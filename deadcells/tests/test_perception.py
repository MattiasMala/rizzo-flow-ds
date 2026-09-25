import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from deadcells.decide import combat_request, route_request
from deadcells.detect import ColorDetector, Detection, decode_yolo, nms
from deadcells.hud import DigitReader, bar_fraction, count_blobs
from deadcells.layout import ColorClass, Layout, Roi
from deadcells.minimap import EMPTY, classify, compress, parse, read_map
from deadcells.pipeline import Perception
from deadcells.scene import describe
from deadcells.track import Tracker

from deadcells import synth

ROOT = Path(__file__).resolve().parents[2]


def layout_with_objects() -> Layout:
    layout = Layout.default()
    layout.object_colors = [
        ColorClass("player", "#00ff00", tol=10),
        ColorClass("enemy", "#ff00ff", tol=10),
        ColorClass("projectile", "#00ffff", tol=10),
    ]
    return layout


def test_roi_rejects_bad_boxes_and_crops():
    with pytest.raises(ValueError):
        Roi(0.5, 0.1, 0.4, 0.2)
    frame = np.zeros((100, 200, 3), np.uint8)
    assert Roi(0.1, 0.2, 0.3, 0.5).crop(frame).shape == (30, 40, 3)


def test_layout_round_trip(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text(Layout.default().dump(), encoding="utf-8")
    loaded = Layout.load(path)
    assert loaded == Layout.default()
    assert not loaded.calibrated


@pytest.mark.parametrize("health", [0.0, 0.25, 0.6, 1.0])
def test_health_bar(health):
    layout = Layout.default()
    frame = synth.frame(layout, health=health)
    value = bar_fraction(layout.rois["health"].crop(frame), layout.health)
    if health == 0:
        assert value is None
    else:
        assert value == pytest.approx(health, abs=0.01)


def test_flask_pips():
    layout = Layout.default()
    for n in range(4):
        frame = synth.frame(layout, flasks=n)
        assert count_blobs(layout.rois["flask"].crop(frame), layout.flask) == n


def test_digits_learned_from_one_frame_read_other_values():
    layout = Layout.default()
    teach = synth.frame(layout, cells=234567, gold=9876)
    templates = {}
    for roi, value in (("cells", 234567), ("gold", 9876)):
        crop = layout.rois[roi].crop(teach)
        templates.update(DigitReader.learn(crop, value, layout.digit_threshold))
    templates.update(
        DigitReader.learn(
            layout.rois["cells"].crop(synth.frame(layout, cells=10)), 10, layout.digit_threshold
        )
    )
    reader = DigitReader(templates, layout.digit_threshold)
    frame = synth.frame(layout, cells=907, gold=31)
    assert reader.read(layout.rois["cells"].crop(frame)) == 907
    assert reader.read(layout.rois["gold"].crop(frame)) == 31
    assert DigitReader({}, 200).read(layout.rois["gold"].crop(frame)) is None


def test_digit_learning_checks_glyph_count():
    layout = Layout.default()
    frame = synth.frame(layout, cells=12)
    with pytest.raises(ValueError, match="glyphs"):
        DigitReader.learn(layout.rois["cells"].crop(frame), 123, layout.digit_threshold)


def test_digit_templates_saved_and_loaded(tmp_path):
    layout = Layout.default()
    frame = synth.frame(layout, cells=4560)
    crop = layout.rois["cells"].crop(frame)
    reader = DigitReader(DigitReader.learn(crop, 4560, layout.digit_threshold))
    reader.save(tmp_path)
    loaded = DigitReader.load(tmp_path, layout.digit_threshold)
    assert sorted(loaded.templates) == ["0", "4", "5", "6"]
    assert loaded.read(crop) == 4560


def minimap_frame(layout):
    # Two rooms joined by a corridor, a separate unexplored-looking island with an item.
    return synth.frame(
        layout,
        rooms=[(2, 2, 12, 8), (12, 4, 30, 5), (30, 2, 40, 10), (60, 20, 64, 24)],
        markers={"player": (4, 6), "exit": (38, 9), "door": (20, 4), "item": (62, 22)},
    )


def test_minimap_classify_markers_win_over_floor():
    layout = Layout.default()
    frame = minimap_frame(layout)
    crop = layout.rois["minimap"].crop(frame)
    grid = classify(crop, layout.minimap_classes, layout.cell_px(frame.shape[0]))
    names = [c.name for c in layout.minimap_classes]
    assert grid[6, 4] == names.index("player")
    assert grid[4, 20] == names.index("door")
    assert grid[3, 3] == names.index("floor")
    assert grid[0, 0] == EMPTY


def test_minimap_routes():
    layout = Layout.default()
    frame = minimap_frame(layout)
    state = read_map(
        layout.rois["minimap"].crop(frame), layout.minimap_classes, layout.cell_px(frame.shape[0])
    )
    assert state.player == (4, 6)
    assert state.rooms == 2
    by_name = {t.name: t for t in state.targets}
    door, exit_, item = by_name["door"], by_name["exit"], by_name["item"]
    assert door.path_length == 16 + 2  # 16 right, 2 up
    assert exit_.path_length == (38 - 4) + 2 + 5  # up into the corridor (y=4), down to y=9
    assert item.path_length is None and item.route == []
    assert [t.name for t in state.targets] == ["door", "exit", "item"]
    assert sum(n for _, n in exit_.route) == exit_.path_length


def test_compress_runs():
    path = [(0, 0), (1, 0), (2, 0), (2, -1), (2, -2), (1, -2)]
    assert compress(path) == [("right", 2), ("up", 2), ("left", 1)]


def test_map_without_player():
    layout = Layout.default()
    grid = np.full((5, 5), EMPTY, np.int8)
    grid[2, :] = [c.name for c in layout.minimap_classes].index("floor")
    state = parse(grid, layout.minimap_classes)
    assert state.player is None and state.rooms == 1 and state.targets == []


def test_nms_and_yolo_decoding():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5) == [0, 2]
    classes = ("player", "enemy")
    # 3 candidates: two overlapping enemies and one player; input 640, frame scaled by 0.5.
    raw = np.array(
        [
            [105, 106, 305],  # cx
            [105, 105, 305],  # cy
            [10, 10, 20],  # w
            [10, 10, 40],  # h
            [0.0, 0.1, 0.9],  # player
            [0.9, 0.8, 0.0],  # enemy
        ],
        np.float32,
    )[None]
    found = decode_yolo(raw, classes, scale=0.5, pad=(0, 80), conf=0.3, iou=0.5)
    assert sorted(d.cls for d in found) == ["enemy", "player"]
    player = next(d for d in found if d.cls == "player")
    assert player.box == pytest.approx((590, 410, 630, 490))
    with pytest.raises(ValueError, match="classes"):
        decode_yolo(raw, ("a",), 1, (0, 0), 0.3, 0.5)


def test_tracker_ids_and_velocity():
    tracker = Tracker()
    first = tracker.update([Detection("enemy", (100, 100, 110, 110), 0.9)], 0, 1000)
    second = tracker.update([Detection("enemy", (120, 100, 130, 110), 0.9)], 100_000_000, 1000)
    assert first[0].id == second[0].id
    assert second[0].vx == pytest.approx(200) and second[0].vy == pytest.approx(0)
    # A different class never inherits the track; a far jump starts a new one.
    third = tracker.update([Detection("player", (120, 100, 130, 110), 0.9)], 200_000_000, 1000)
    assert third[0].id != first[0].id
    fourth = tracker.update([Detection("enemy", (900, 900, 910, 910), 0.9)], 300_000_000, 1000)
    assert fourth[0].id not in {first[0].id, third[0].id}


def test_scene_relative_positions_and_caps():
    from deadcells.hud import Hud

    tracker = Tracker()
    player = Detection("player", (940, 480, 980, 560), 0.9)  # 80 px tall, centre (960, 520)
    enemies = [
        Detection("enemy", (940 + 80 * k, 480, 980 + 80 * k, 560), 0.8) for k in range(2, 10)
    ]
    shot = Detection("projectile", (700, 510, 710, 520), 0.8)
    tracker.update([player, *enemies, shot], 0, 1080)
    moved = Detection("projectile", (740, 510, 750, 520), 0.8)  # 40 px in 0.1 s, towards player
    tracks = tracker.update([player, *enemies, moved], 100_000_000, 1080)
    state = describe((1920, 1080), Hud(0.5, 1, 10, None), None, tracks, calibrated=False, limit=3)
    threats = state["threats"]
    assert threats["total"] == 9 and len(threats["listed"]) == 3
    first = threats["listed"][0]
    assert first["type"] == "projectile" and first["approaching"] and first["side"] == "left"
    assert first["seconds_to_reach"] == pytest.approx(abs(first["dx"]) / 5.0, abs=0.05)
    assert threats["listed"][1]["dx"] == pytest.approx(2.0)
    assert state["player"]["health_percent"] == 50 and not state["player"]["position_estimated"]
    assert "warning" in state
    json.dumps(state)


def test_pipeline_end_to_end_with_color_detector():
    layout = layout_with_objects()
    layout.calibrated = True
    objects = [("player", (940, 480, 980, 560)), ("enemy", (1100, 480, 1140, 560))]
    frame = synth.frame(
        layout,
        health=0.8,
        flasks=3,
        rooms=[(2, 2, 20, 6)],
        markers={"player": (3, 3), "exit": (18, 5)},
        objects=objects,
    )
    perception = Perception(layout, detector=ColorDetector(layout.object_colors, downscale=2))
    state = perception.process(frame, 0)
    assert state["player"]["health_percent"] == 80 and state["player"]["flask_charges"] == 3
    enemy = state["threats"]["listed"][0]
    assert (
        enemy["type"] == "enemy" and enemy["dx"] == pytest.approx(2.0) and enemy["level"] == "same"
    )
    assert state["map"]["targets"]["listed"][0]["type"] == "exit"
    assert "warning" not in state
    assert {"hud", "map", "detect", "track", "describe", "processing", "latency"} <= set(
        perception.timings.summary()
    )


def test_requests_are_valid_for_rizzo_schema():
    sys.path.insert(0, str(ROOT / "src"))
    schema = pytest.importorskip("rizzo_flow.schema")
    layout = Layout.default()
    frame = minimap_frame(layout)
    state = Perception(layout).process(frame, 0)
    schema.Request.model_validate(combat_request(state))
    route = route_request(state)
    assert route is not None
    schema.Request.model_validate(route)
    assert len(route["questions"]["target"]["options"]) == 2  # the unreachable item is left out
