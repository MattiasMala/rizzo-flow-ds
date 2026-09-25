"""HUD readers: health bar, flask charges and numeric counters (cells, gold).

Everything works on small crops with colour masks and projections: well under a millisecond per
frame, no OCR engine. Counters are read by matching each glyph against digit templates learned
from a screenshot whose value is known (`python -m deadcells learn-digits`).
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .layout import ColorClass

GLYPH_SIZE = (10, 14)  # width, height of normalised digit glyphs


def bar_fraction(crop: np.ndarray, color: ColorClass, min_rows: float = 0.4) -> float | None:
    """Filled share of a left-to-right bar, or None when no pixel of the bar colour is visible."""
    mask = color.mask(crop)
    filled = np.flatnonzero((mask > 0).mean(axis=0) >= min_rows)
    if filled.size == 0:
        return None
    return float(filled[-1] + 1) / mask.shape[1]


def count_blobs(crop: np.ndarray, color: ColorClass, min_area: int = 6) -> int:
    """Separate blobs of one colour (flask charges and similar pips)."""
    mask = color.mask(crop)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return int((stats[1:, cv2.CC_STAT_AREA] >= min_area).sum())


def ink(crop: np.ndarray, threshold: int) -> np.ndarray:
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return grey >= threshold


def glyphs(mask: np.ndarray, min_width: int = 1) -> list[np.ndarray]:
    """Split a binary counter crop into glyphs by empty columns, trimmed and normalised."""
    columns = mask.any(axis=0)
    result, start = [], None
    for x, on in enumerate([*columns, False]):
        if on and start is None:
            start = x
        elif not on and start is not None:
            if x - start >= min_width:
                piece = mask[:, start:x]
                rows = np.flatnonzero(piece.any(axis=1))
                piece = piece[rows[0] : rows[-1] + 1]
                resized = cv2.resize(
                    piece.astype(np.float32), GLYPH_SIZE, interpolation=cv2.INTER_AREA
                )
                result.append(resized)
            start = None
    return result


@dataclass
class DigitReader:
    templates: dict[str, np.ndarray]
    threshold: int = 200
    max_distance: float = 0.35  # mean absolute difference above which a glyph is unknown

    def read(self, crop: np.ndarray) -> int | None:
        if not self.templates:
            return None
        text = ""
        for glyph in glyphs(ink(crop, self.threshold)):
            label, distance = min(
                ((k, float(np.abs(glyph - t).mean())) for k, t in self.templates.items()),
                key=lambda item: item[1],
            )
            if distance > self.max_distance:
                return None
            text += label
        return int(text) if text.isdigit() else None

    @staticmethod
    def learn(crop: np.ndarray, value: int, threshold: int) -> dict[str, np.ndarray]:
        found = glyphs(ink(crop, threshold))
        digits = str(value)
        if len(found) != len(digits):
            raise ValueError(
                f"Found {len(found)} glyphs but {value} has {len(digits)} digits: "
                "check the ROI and the ink threshold"
            )
        return dict(zip(digits, found))

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for label, glyph in self.templates.items():
            cv2.imwrite(str(directory / f"{label}.png"), (glyph * 255).astype(np.uint8))

    @classmethod
    def load(cls, directory: str | Path, threshold: int = 200) -> "DigitReader":
        templates = {}
        for path in sorted(Path(directory).glob("[0-9].png")):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            templates[path.stem] = image.astype(np.float32) / 255
        return cls(templates, threshold)


@dataclass(frozen=True)
class Hud:
    health: float | None  # 0..1
    flask_charges: int
    cells: int | None
    gold: int | None
