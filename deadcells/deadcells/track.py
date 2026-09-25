"""Frame-to-frame tracking: stable IDs and velocities for detections.

Greedy nearest-centre matching within each class (few objects per frame: no Hungarian solver
needed). Velocity is an exponential moving average in pixels per second, so a projectile's
time to reach the player can be estimated from two or more frames.
"""

import itertools
from dataclasses import dataclass, field

from .detect import Detection


@dataclass
class Track:
    id: int
    detection: Detection
    vx: float = 0.0
    vy: float = 0.0
    seen: int = 1  # frames matched
    missed: int = 0
    last_ns: int = 0


@dataclass
class Tracker:
    max_jump: float = 0.08  # max centre move between frames, as a share of frame height
    max_missed: int = 5
    smoothing: float = 0.5
    tracks: list[Track] = field(default_factory=list)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    def update(self, detections: list[Detection], t_ns: int, frame_height: int) -> list[Track]:
        limit = self.max_jump * frame_height
        free = list(self.tracks)
        matched: list[Track] = []
        # Most confident detections pick their track first.
        for det in sorted(detections, key=lambda d: -d.score):
            cx, cy = det.centre
            best, best_d = None, limit
            for track in free:
                if track.detection.cls != det.cls:
                    continue
                tx, ty = track.detection.centre
                d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if d <= best_d:
                    best, best_d = track, d
            if best is None:
                matched.append(Track(next(self._ids), det, last_ns=t_ns))
                continue
            free.remove(best)
            dt = (t_ns - best.last_ns) / 1e9
            if dt > 0:
                tx, ty = best.detection.centre
                a = self.smoothing if best.seen > 1 else 1.0
                best.vx = a * (cx - tx) / dt + (1 - a) * best.vx
                best.vy = a * (cy - ty) / dt + (1 - a) * best.vy
            best.detection, best.seen, best.missed, best.last_ns = det, best.seen + 1, 0, t_ns
            matched.append(best)
        for track in free:
            track.missed += 1
        self.tracks = matched + [t for t in free if t.missed <= self.max_missed]
        return matched  # only what is visible in this frame
