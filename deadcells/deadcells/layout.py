"""Screen regions and colour classes of the Dead Cells HUD.

Regions are fractions of the game frame, so one layout serves every 16:9 resolution. Colours are
written as RGB hex for people and converted to BGR (OpenCV order) on load. The defaults are NOT
measured on the real game: `calibrated` stays false until someone checks them on a screenshot with
`python -m deadcells calibrate`, and every scene description carries that flag.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Kinds of minimap colour classes: the player marker, explored floor, and points of interest.
MINIMAP_KINDS = ("player", "walkable", "poi")


@dataclass(frozen=True)
class Roi:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self):
        if not (0 <= self.x0 < self.x1 <= 1 and 0 <= self.y0 < self.y1 <= 1):
            raise ValueError(f"ROI must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1: {self}")

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        x0, x1 = round(self.x0 * width), max(round(self.x1 * width), round(self.x0 * width) + 1)
        y0, y1 = round(self.y0 * height), max(round(self.y1 * height), round(self.y0 * height) + 1)
        return x0, y0, x1, y1

    def crop(self, frame: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.pixels(frame.shape[1], frame.shape[0])
        return frame[y0:y1, x0:x1]


def parse_hex(value: str) -> tuple[int, int, int]:
    """'#rrggbb' -> (b, g, r)."""
    text = value.lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Colour must be #rrggbb: {value!r}")
    r, g, b = (int(text[i : i + 2], 16) for i in (0, 2, 4))
    return b, g, r


@dataclass(frozen=True)
class ColorClass:
    """Pixels whose every channel is within `tol` of `rgb` belong to the class."""

    name: str
    rgb: str
    tol: int = 24
    kind: str = "poi"
    min_cover: float = 0.25  # share of a minimap cell the colour must cover to claim it

    def __post_init__(self):
        parse_hex(self.rgb)
        if self.kind not in MINIMAP_KINDS:
            raise ValueError(f"kind must be one of {MINIMAP_KINDS}: {self.kind!r}")

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        bgr = np.array(parse_hex(self.rgb), dtype=np.int16)
        return (
            np.clip(bgr - self.tol, 0, 255).astype(np.uint8),
            np.clip(bgr + self.tol, 0, 255).astype(np.uint8),
        )

    def mask(self, image: np.ndarray) -> np.ndarray:
        """uint8 mask (255 inside) of a BGR uint8 image; cv2.inRange is SIMD, ~10x numpy."""
        low, high = self.bounds
        return cv2.inRange(image, low, high)


@dataclass
class Layout:
    rois: dict[str, Roi]
    health: ColorClass
    flask: ColorClass
    minimap_classes: list[ColorClass]
    minimap_cell: float = 0.004  # minimap cell side as a fraction of frame height (~4 px at 1080p)
    digit_threshold: int = 200  # grey level above which a counter pixel is ink
    calibrated: bool = False
    notes: str = ""
    object_colors: list[ColorClass] = field(default_factory=list)

    def cell_px(self, height: int) -> int:
        return max(1, round(self.minimap_cell * height))

    @classmethod
    def default(cls) -> "Layout":
        # Rough positions of the 16:9 HUD: health bar and flask bottom-left, counters above them,
        # minimap top-right. Guesses to be replaced by `calibrate` on a real screenshot.
        return cls(
            rois={
                "health": Roi(0.030, 0.925, 0.300, 0.950),
                "flask": Roi(0.030, 0.860, 0.090, 0.915),
                "cells": Roi(0.030, 0.810, 0.110, 0.845),
                "gold": Roi(0.120, 0.810, 0.220, 0.845),
                "minimap": Roi(0.780, 0.030, 0.985, 0.300),
                "fullmap": Roi(0.050, 0.080, 0.950, 0.920),
            },
            # Wiki: current health is green (orange = recoverable, yellow = Malaise, blue = bonus).
            health=ColorClass("health", "#4fbf3a", tol=40),
            flask=ColorClass("flask", "#e04a3c", tol=40),
            minimap_classes=[
                ColorClass("player", "#ffffff", tol=20, kind="player", min_cover=0.2),
                ColorClass("exit", "#39d353", tol=30),
                ColorClass("door", "#e3b341", tol=30),
                ColorClass("teleporter", "#58a6ff", tol=30),
                ColorClass("item", "#f778ba", tol=30),
                ColorClass("floor", "#6e7681", tol=28, kind="walkable", min_cover=0.4),
            ],
            notes="Default layout: guessed, not measured on the game. Run `calibrate`.",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Layout":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            rois={name: Roi(**roi) for name, roi in data["rois"].items()},
            health=ColorClass(**data["health"]),
            flask=ColorClass(**data["flask"]),
            minimap_classes=[ColorClass(**c) for c in data["minimap_classes"]],
            minimap_cell=data.get("minimap_cell", 0.004),
            digit_threshold=data.get("digit_threshold", 200),
            calibrated=data.get("calibrated", False),
            notes=data.get("notes", ""),
            object_colors=[ColorClass(**c) for c in data.get("object_colors", [])],
        )

    def dump(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n"
