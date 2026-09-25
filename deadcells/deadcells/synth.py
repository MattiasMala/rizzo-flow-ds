"""Synthetic frames drawn with the layout's own colours, for tests and latency benchmarks.

They check that the code reads what was drawn; they say nothing about the real game's pixels.
"""

import cv2
import numpy as np

from .layout import Layout, parse_hex


def fill(frame: np.ndarray, box: tuple[int, int, int, int], rgb: str) -> None:
    x0, y0, x1, y1 = box
    frame[y0:y1, x0:x1] = parse_hex(rgb)


def frame(
    layout: Layout,
    width: int = 1920,
    height: int = 1080,
    health: float = 0.6,
    flasks: int = 2,
    cells: int | None = 123,
    gold: int | None = 4567,
    rooms: list[tuple[int, int, int, int]] | None = None,
    markers: dict[str, tuple[int, int]] | None = None,
    objects: list[tuple[str, tuple[int, int, int, int]]] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Draw a frame. `rooms` and `markers` are in minimap cells (x0, y0, x1, y1 / x, y)."""
    rng = np.random.default_rng(seed)
    image = rng.integers(20, 60, size=(height, width, 3), dtype=np.uint8)  # dark noisy scene
    rois = layout.rois

    x0, y0, x1, y1 = rois["health"].pixels(width, height)
    image[y0:y1, x0:x1] = 30
    fill(image, (x0, y0, x0 + round((x1 - x0) * health), y1), layout.health.rgb)

    x0, y0, x1, y1 = rois["flask"].pixels(width, height)
    pip = max(2, (x1 - x0) // 7)
    for i in range(flasks):
        fill(image, (x0 + i * 2 * pip, y0, x0 + i * 2 * pip + pip, y1), layout.flask.rgb)

    for name, value in (("cells", cells), ("gold", gold)):
        if value is None:
            continue
        x0, y0, x1, y1 = rois[name].pixels(width, height)
        image[y0:y1, x0:x1] = 0
        cv2.putText(
            image,
            str(value),
            (x0 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            (y1 - y0) / 30,
            (255, 255, 255),
            2,
            cv2.LINE_8,
        )

    by_name = {c.name: c for c in layout.minimap_classes}
    floor = next(c for c in layout.minimap_classes if c.kind == "walkable")
    cell = layout.cell_px(height)
    mx, my, mx1, my1 = rois["minimap"].pixels(width, height)
    image[my:my1, mx:mx1] = 10
    for rx0, ry0, rx1, ry1 in rooms or []:
        fill(image, (mx + rx0 * cell, my + ry0 * cell, mx + rx1 * cell, my + ry1 * cell), floor.rgb)
    for name, (cx, cy) in (markers or {}).items():
        base = name.split("#")[0]  # "door#2" draws a second door
        fill(
            image,
            (mx + cx * cell, my + cy * cell, mx + (cx + 1) * cell, my + (cy + 1) * cell),
            by_name[base].rgb,
        )

    colors = {c.name: c for c in layout.object_colors}
    for name, box in objects or []:
        fill(image, box, colors[name].rgb)
    return image
