"""Command line: python -m deadcells <command> (run from the deadcells/ folder)."""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_layout(path: str | None):
    from .layout import Layout

    return Layout.load(path) if path else Layout.default()


def make_perception(args):
    from .detect import ColorDetector, OnnxDetector
    from .hud import DigitReader
    from .pipeline import Perception

    layout = load_layout(args.layout)
    detector = None
    if args.model:
        detector = OnnxDetector(args.model, conf=args.conf)
        print(f"detector: {args.model} on {detector.provider}", file=sys.stderr)
    elif layout.object_colors:
        detector = ColorDetector(layout.object_colors)
    digits = DigitReader.load(args.digits, layout.digit_threshold) if args.digits else None
    return Perception(layout, detector, digits, map_roi=args.map, limit=args.limit)


def cmd_builds(args):
    from . import builds

    data = builds.load("builds.json")
    problems = builds.validate(data)
    if problems:
        sys.exit("\n".join(problems))
    if args.format == "json":
        print(json.dumps(data, indent=1, ensure_ascii=False))
    elif args.format == "tree":
        print(builds.tree(data), end="")
    else:
        print(builds.table(data), end="")


def cmd_describe(args):
    from .capture import FileSource

    perception = make_perception(args)
    source = FileSource(args.images)
    for path in args.images:
        frame, t = source.grab()
        state = perception.process(frame, t)
        print(json.dumps({"image": str(path), "state": state}, ensure_ascii=False))
    print(json.dumps({"timings_ms": perception.timings.summary()}), file=sys.stderr)


def cmd_live(args):
    from .capture import LatestFrame, find_window, open_source
    from .decide import combat_request, post, route_request

    region = find_window(args.window)
    if region is None:
        print(f"window {args.window!r} not found: capturing the whole screen", file=sys.stderr)
    perception = make_perception(args)
    frames = LatestFrame(open_source(args.source, region, args.fps))
    # Decisions run on their own thread: perception keeps the frame rate while the model thinks.
    pool = ThreadPoolExecutor(max_workers=1)
    pending, last_decision, count = None, 0.0, 0

    def decide(state: dict, frame: int) -> dict:
        request = combat_request(state)
        if args.route and (route := route_request(state)):
            request["questions"].update(route["questions"])
            request["state"]["map"] = state.get("map")
        started = time.perf_counter()
        answers = post(request, args.decide)["answers"]
        return {
            "decision_for_frame": frame,
            "decision": {k: a.get("choice", a.get("score")) for k, a in answers.items()},
            "decision_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    try:
        while args.frames == 0 or count < args.frames:
            item = frames.get()
            if item is None:
                continue
            count += 1
            line = {"frame": count, "state": perception.process(*item)}
            if pending is not None and pending.done():
                line.update(pending.result())
                pending = None
            now = time.monotonic()
            if args.decide and pending is None and now - last_decision >= args.every:
                last_decision = now
                pending = pool.submit(decide, line["state"], count)
            if not args.quiet:
                print(json.dumps(line, ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        frames.close()
        pool.shutdown(wait=False, cancel_futures=True)
        summary = {"timings_ms": perception.timings.summary(), "dropped_frames": frames.dropped}
        print(json.dumps(summary), file=sys.stderr)


def cmd_bench(args):
    from . import synth
    from .detect import ColorDetector
    from .layout import ColorClass
    from .pipeline import Perception

    layout = load_layout(args.layout)
    width, height = map(int, args.size.lower().split("x"))
    cell = layout.cell_px(height)
    x0, y0, x1, y1 = layout.rois["minimap"].pixels(width, height)
    cols, rows = (x1 - x0) // cell, (y1 - y0) // cell
    colors = [ColorClass("player", "#00ff00", tol=10), ColorClass("enemy", "#ff00ff", tol=10)]
    layout.object_colors = colors
    # Worst case for the map: everything explored, targets in the far corners.
    frame = synth.frame(
        layout,
        width,
        height,
        rooms=[(0, 0, cols, rows)],
        markers={"player": (1, 1), "exit": (cols - 2, rows - 2), "door": (cols // 2, rows // 2)},
        objects=[
            ("player", (width // 2 - 20, height // 2 - 40, width // 2 + 20, height // 2 + 40)),
            ("enemy", (width // 2 + 150, height // 2 - 40, width // 2 + 190, height // 2 + 40)),
        ],
    )
    results = {}
    for name, detector in (("no detector", None), ("colour detector", ColorDetector(colors))):
        perception = Perception(layout, detector)
        for i in range(args.frames):
            if args.moving and i % 2:
                perception._map_cache = None  # force a full map parse every other frame
            perception.process(frame, time.perf_counter_ns())
        results[name] = perception.timings.summary()
    print(
        json.dumps({"frame": args.size, "map_grid": [cols, rows], "timings_ms": results}, indent=1)
    )


def cmd_calibrate(args):
    import cv2
    import numpy as np

    layout = load_layout(args.layout)
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        sys.exit(f"Cannot read {args.image}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    h, w = image.shape[:2]
    report = {"frame": [w, h], "rois": {}}
    for name, roi in layout.rois.items():
        x0, y0, x1, y1 = roi.pixels(w, h)
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
        cv2.putText(
            overlay, name, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
        )
        crop = image[y0:y1, x0:x1]
        cv2.imwrite(str(out / f"{name}.png"), crop)
        # Most frequent colours of the region: candidates for the layout's colour classes.
        pixels = crop.reshape(-1, 3) // 8 * 8
        colours, counts = np.unique(pixels, axis=0, return_counts=True)
        top = np.argsort(-counts)[: args.colours]
        report["rois"][name] = [
            {
                "rgb": f"#{int(c[2]):02x}{int(c[1]):02x}{int(c[0]):02x}",
                "share": round(float(n) / len(pixels), 3),
            }
            for c, n in zip(colours[top], counts[top])
        ]
    cv2.imwrite(str(out / "overlay.png"), overlay)
    (out / "colours.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {out}/overlay.png, one crop per region and colours.json", file=sys.stderr)


def cmd_learn_digits(args):
    import cv2

    from .hud import DigitReader

    layout = load_layout(args.layout)
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        sys.exit(f"Cannot read {args.image}")
    reader = (
        DigitReader.load(args.digits, layout.digit_threshold)
        if Path(args.digits).exists()
        else DigitReader({}, layout.digit_threshold)
    )
    learned = DigitReader.learn(
        layout.rois[args.roi].crop(image), args.value, layout.digit_threshold
    )
    reader.templates.update(learned)
    reader.save(args.digits)
    print(f"digits known: {''.join(sorted(reader.templates))}", file=sys.stderr)


def cmd_collect(args):
    import cv2

    from .capture import find_window, open_source

    region = find_window(args.window)
    source = open_source(args.source, region, 60)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        for i in range(args.count):
            frame, _ = source.grab()
            cv2.imwrite(str(out / f"frame_{int(time.time() * 1000)}_{i:05d}.png"), frame)
            time.sleep(args.every)
    except KeyboardInterrupt:
        pass
    finally:
        source.close()


def cmd_nav(args):
    import scipy.sparse.csgraph  # noqa: F401  (import time is not search time)

    from .nav import NavGraph, Physics, parse_ascii

    grid, marks = parse_ascii(Path(args.level).read_text(encoding="utf-8"))
    for mark in ("S", "G"):
        if mark not in marks:
            sys.exit(f"The level needs an {mark} (start) and a G (goal)")
    started = time.perf_counter()
    graph = NavGraph(grid, Physics())
    built = time.perf_counter()
    start, goal = graph.node(*marks["S"][0]), graph.node(*marks["G"][0])
    path = graph.path(start, goal)
    done = time.perf_counter()
    print(
        json.dumps(
            {
                "nodes": len(graph.cells),
                "edges": graph.edges,
                "build_ms": round((built - started) * 1000, 1),
                "search_ms": round((done - built) * 1000, 2),
                "seconds": None if path is None else round(graph.path_cost(path), 3),
                "actions": None if path is None else graph.actions(path),
            },
            indent=1,
        )
    )


def cmd_bridge(args):
    from .bridge import BridgeReader
    from .decide import combat_request, post
    from .world import Navigator

    reader = BridgeReader(args.name)
    nav = Navigator(limit=args.limit, replan_every=args.replan)
    target = args.target if args.target != "none" else None
    pool = ThreadPoolExecutor(max_workers=1)
    pending, last, count = None, 0.0, 0
    try:
        while args.frames == 0 or count < args.frames:
            snap = reader.read()
            if snap is None:
                time.sleep(0.0005)  # no new frame yet
                continue
            count += 1
            line = {
                "frame": snap.frame,
                "state": nav.update(snap, target),
                "timings_ms": dict(nav.timings_ms),
            }
            if pending is not None and pending.done():
                answers = pending.result()["answers"]
                line["decision"] = {k: a.get("choice", a.get("score")) for k, a in answers.items()}
                pending = None
            if args.decide and pending is None and time.monotonic() - last >= args.every:
                last = time.monotonic()
                pending = pool.submit(post, combat_request(line["state"]), args.decide)
            if not args.quiet:
                print(json.dumps(line, ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        print(json.dumps({"torn_reads": reader.torn, "stale_reads": reader.stale}), file=sys.stderr)


def cmd_bench_nav(args):
    import statistics
    import tempfile

    import numpy as np

    from . import synth
    from .bridge import ENTITY_DTYPE, KINDS, BridgeReader, BridgeWriter, Hero
    from .nav import NavGraph, Physics
    from .world import Navigator

    rows, cols = map(int, args.size.lower().split("x"))
    grid = synth.level(args.seed, rows, cols, args.platforms)
    t = time.perf_counter()
    graph = NavGraph(grid, Physics())
    build_ms = (time.perf_counter() - t) * 1000
    rng = np.random.default_rng(args.seed)
    standing = np.argwhere(graph.standable)
    pairs = rng.integers(len(graph.cells), size=(50, 2))

    def timed(fn, n):
        values = []
        for a, b in pairs[:n]:
            t = time.perf_counter()
            fn(int(a), int(b))
            values.append((time.perf_counter() - t) * 1000)
        return values

    graph.path(0, 0)  # scipy import outside the measure
    results = {
        "astar_python_ms": timed(graph.astar, 20),
        "path_scipy_ms": timed(graph.path, 50),
        "flow_field_ms": timed(lambda a, b: graph.flow_field(b), 20),
    }
    exit_cell = standing[len(standing) // 2]

    def entities(n):
        e = np.zeros(n, ENTITY_DTYPE)
        pick = standing[rng.integers(len(standing), size=n)]
        e["kind"], e["x"], e["y"] = KINDS.index("enemy"), pick[:, 1] + 0.5, pick[:, 0] + 0.9
        e["vx"] = rng.normal(0, 3, n)
        e[0] = (1, KINDS.index("exit"), 0, exit_cell[1] + 0.5, exit_cell[0] + 0.9, 0, 0, 0, 0)
        return e

    start = standing[5]
    hero = Hero(start[1] + 0.5, start[0] + 0.9, 0, 0, 80, 100, 2, 3, 10, 100, 1, 3, 1, 2, 0)
    with tempfile.TemporaryDirectory() as tmp:
        writer = BridgeWriter(Path(tmp) / "bench.bin", cols, rows, capacity=128)
        writer.write(hero, entities(args.entities), 0.0, grid)
        reader, nav = BridgeReader(Path(tmp) / "bench.bin"), Navigator()
        nav.update(reader.read(), "exit")
        reads, updates = [], []
        for i in range(args.frames):
            writer.write(hero, entities(args.entities), i / 60)
            t = time.perf_counter()
            snap = reader.read()
            reads.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            nav.update(snap, "exit")
            updates.append((time.perf_counter() - t) * 1000)
        reader.close()
        writer.close()
    results["bridge_read_ms"], results["navigator_update_ms"] = reads, updates

    def stats(values):
        ordered = sorted(values)
        return {
            "p50": round(statistics.median(ordered), 3),
            "p95": round(ordered[round(0.95 * (len(ordered) - 1))], 3),
            "max": round(ordered[-1], 3),
        }

    print(
        json.dumps(
            {
                "level": [rows, cols],
                "nodes": len(graph.cells),
                "edges": graph.edges,
                "graph_build_ms": round(build_ms, 1),
                "entities": args.entities,
                **{k: stats(v) for k, v in results.items()},
            },
            indent=1,
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m deadcells")
    sub = parser.add_subparsers(dest="command", required=True)

    def perception_args(p):
        p.add_argument("--layout", help="layout JSON (default: built-in guesses, not calibrated)")
        p.add_argument("--model", help="YOLO ONNX detector trained on Dead Cells")
        p.add_argument("--conf", type=float, default=0.35)
        p.add_argument("--digits", help="folder of digit templates (learn-digits)")
        p.add_argument("--map", default="minimap", choices=["minimap", "fullmap"])
        p.add_argument("--limit", type=int, default=6, help="max items listed per group")

    p = sub.add_parser("builds", help="build table / tree")
    p.add_argument("--format", choices=["md", "tree", "json"], default="md")
    p.set_defaults(func=cmd_builds)

    p = sub.add_parser("layout", help="print the default layout JSON (starting point)")
    p.set_defaults(func=lambda args: print(load_layout(None).dump(), end=""))

    p = sub.add_parser("describe", help="describe screenshots as JSON lines")
    p.add_argument("images", nargs="+")
    perception_args(p)
    p.set_defaults(func=cmd_describe)

    p = sub.add_parser("live", help="describe the game screen in real time")
    perception_args(p)
    p.add_argument("--source", default="auto", choices=["auto", "dxcam", "mss"])
    p.add_argument("--window", default="Dead Cells")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = never)")
    p.add_argument("--decide", metavar="URL", help="ask a running `rizzo serve` for actions")
    p.add_argument("--every", type=float, default=0.25, help="seconds between decisions")
    p.add_argument("--route", action="store_true", help="also ask where to go on the map")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_live)

    p = sub.add_parser("bench", help="latency of every stage on a synthetic frame")
    p.add_argument("--layout")
    p.add_argument("--size", default="1920x1080")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--moving", action="store_true", help="re-parse the map every other frame")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("calibrate", help="draw the layout on a screenshot, dump region colours")
    p.add_argument("image")
    p.add_argument("--layout")
    p.add_argument("--out", default="calibration")
    p.add_argument("--colours", type=int, default=8)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("learn-digits", help="learn digit templates from a counter of known value")
    p.add_argument("image")
    p.add_argument("--roi", default="cells", choices=["cells", "gold"])
    p.add_argument("--value", type=int, required=True)
    p.add_argument("--layout")
    p.add_argument("--digits", default="calibration/digits")
    p.set_defaults(func=cmd_learn_digits)

    p = sub.add_parser("collect", help="save game frames for labelling the detector dataset")
    p.add_argument("--out", default="dataset/raw")
    p.add_argument("--every", type=float, default=0.5)
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--source", default="auto", choices=["auto", "dxcam", "mss"])
    p.add_argument("--window", default="Dead Cells")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("nav", help="path on an ASCII level (S start, G goal, # . = H ^)")
    p.add_argument("level")
    p.set_defaults(func=cmd_nav)

    p = sub.add_parser("bridge", help="read the game state from the mod's shared memory")
    p.add_argument("--name", default="Local\\RizzoDeadCells", help="mapping name or file path")
    p.add_argument("--target", default="exit", help="entity type to reach, or none")
    p.add_argument("--frames", type=int, default=0)
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--replan", type=int, default=6, help="frames between danger-aware replans")
    p.add_argument("--decide", metavar="URL")
    p.add_argument("--every", type=float, default=0.25)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_bridge)

    p = sub.add_parser("bench-nav", help="pathfinding and bridge latency on a random level")
    p.add_argument("--size", default="200x500", help="rows x cols")
    p.add_argument("--platforms", type=int, default=1600)
    p.add_argument("--entities", type=int, default=40)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_bench_nav)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
