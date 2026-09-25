"""Perception pipeline: frame -> HUD, map, detections, tracks -> scene description, with timings.

Each stage is timed with perf_counter_ns; `latency_ms` adds the age of the frame (capture to
description), which is what matters for reacting in time. The language model is NOT called per
frame: a decision costs ~45-50 ms, so it reads the latest description when a decision is due.
"""

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from .hud import DigitReader, Hud, bar_fraction, count_blobs
from .layout import Layout
from .minimap import classify, parse
from .scene import describe
from .track import Tracker


@dataclass
class Timings:
    samples: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def add(self, stage: str, ms: float) -> None:
        self.samples[stage].append(ms)

    def summary(self) -> dict[str, dict[str, float]]:
        result = {}
        for stage, values in self.samples.items():
            ordered = sorted(values)
            p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
            result[stage] = {
                "p50": round(statistics.median(ordered), 3),
                "p95": round(p95, 3),
                "max": round(ordered[-1], 3),
                "n": len(ordered),
            }
        return result


class Perception:
    def __init__(
        self,
        layout: Layout,
        detector=None,
        digits: DigitReader | None = None,
        map_roi: str = "minimap",
        limit: int = 6,
    ):
        self.layout, self.detector, self.digits = layout, detector, digits
        self.map_roi, self.limit = map_roi, limit
        self.tracker = Tracker()
        self.timings = Timings()
        self._map_cache: tuple[np.ndarray, object] | None = None

    def process(self, frame: np.ndarray, captured_ns: int) -> dict:
        stamps = [("start", time.perf_counter_ns())]

        def mark(stage: str) -> None:
            stamps.append((stage, time.perf_counter_ns()))

        rois = self.layout.rois
        health = bar_fraction(rois["health"].crop(frame), self.layout.health)
        flask = count_blobs(rois["flask"].crop(frame), self.layout.flask)
        cells = gold = None
        if self.digits is not None:
            cells = self.digits.read(rois["cells"].crop(frame))
            gold = self.digits.read(rois["gold"].crop(frame))
        hud = Hud(health, flask, cells, gold)
        mark("hud")

        map_state = None
        if self.layout.minimap_classes and self.map_roi in rois:
            cell = self.layout.cell_px(frame.shape[0])
            classes = self.layout.minimap_classes
            grid = classify(rois[self.map_roi].crop(frame), classes, cell)
            # The map changes only when the player crosses a cell: reuse the routes otherwise.
            if self._map_cache is not None and np.array_equal(self._map_cache[0], grid):
                map_state = self._map_cache[1]
            else:
                map_state = parse(grid, classes)
                self._map_cache = (grid, map_state)
        mark("map")

        detections = self.detector(frame) if self.detector is not None else []
        mark("detect")
        tracks = self.tracker.update(detections, captured_ns, frame.shape[0])
        mark("track")
        state = describe(
            (frame.shape[1], frame.shape[0]),
            hud,
            map_state,
            tracks,
            self.layout.calibrated,
            self.limit,
        )
        mark("describe")

        for (_, t0), (stage, t1) in pairwise(stamps):
            self.timings.add(stage, (t1 - t0) / 1e6)
        self.timings.add("processing", (stamps[-1][1] - stamps[0][1]) / 1e6)
        self.timings.add("latency", (stamps[-1][1] - captured_ns) / 1e6)
        return state
