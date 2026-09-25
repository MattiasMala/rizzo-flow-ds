"""Object detection on the game frame.

`OnnxDetector` runs a YOLO model (Ultralytics v8/11 ONNX export: output `1 × (4 + classes) × N`,
boxes as centre x, centre y, width, height in input pixels) with onnxruntime. Nothing here is
trained yet: the model must be trained on labelled Dead Cells screenshots (docs/deadcells.md).
`ColorDetector` finds blobs of calibrated colours; it needs no training and suits things with
a unique colour, not enemies in general.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .layout import ColorClass

# Classes the detector is meant to learn. The order is the model's class index.
CLASSES = (
    "player",
    "enemy",
    "elite",
    "boss",
    "projectile",
    "telegraph",  # attack wind-up flash / red warning area
    "trap",
    "weapon_drop",
    "scroll",
    "food",
    "gold",
    "cell",
    "chest",
    "door",
    "exit",
    "teleporter",
    "npc",
    "breakable_wall",
)


@dataclass(frozen=True)
class Detection:
    cls: str
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in frame pixels
    score: float

    @property
    def centre(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """Indices kept by greedy non-maximum suppression, best score first."""
    order = np.argsort(-scores)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        w = np.clip(
            np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]),
            0,
            None,
        )
        h = np.clip(
            np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]),
            0,
            None,
        )
        overlap = w * h / (areas[i] + areas[rest] - w * h + 1e-9)
        order = rest[overlap <= iou]
    return keep


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to a square; returns image, scale and padding."""
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, scale, (pad_x, pad_y)


def decode_yolo(
    output: np.ndarray,
    classes: tuple[str, ...],
    scale: float,
    pad: tuple[int, int],
    conf: float,
    iou: float,
) -> list[Detection]:
    pred = output[0].T  # N × (4 + classes)
    if pred.shape[1] != 4 + len(classes):
        raise ValueError(f"Model has {pred.shape[1] - 4} classes, expected {len(classes)}")
    scores = pred[:, 4:]
    best = scores.argmax(axis=1)
    best_score = scores[np.arange(len(pred)), best]
    keep = best_score >= conf
    pred, best, best_score = pred[keep], best[keep], best_score[keep]
    if not len(pred):
        return []
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    boxes -= np.array([pad[0], pad[1], pad[0], pad[1]])
    boxes /= scale
    result = []
    for c in np.unique(best):  # suppression within each class
        idx = np.flatnonzero(best == c)
        for i in nms(boxes[idx], best_score[idx], iou):
            j = idx[i]
            result.append(
                Detection(classes[c], tuple(float(v) for v in boxes[j]), float(best_score[j]))
            )
    return result


class OnnxDetector:
    def __init__(
        self,
        path: str,
        classes: tuple[str, ...] = CLASSES,
        size: int = 640,
        conf: float = 0.35,
        iou: float = 0.5,
        providers: list[str] | None = None,
    ):
        import onnxruntime as ort

        available = ort.get_available_providers()
        preferred = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        self.session = ort.InferenceSession(
            path, providers=providers or [p for p in preferred if p in available]
        )
        self.input = self.session.get_inputs()[0].name
        self.classes, self.size, self.conf, self.iou = classes, size, conf, iou

    @property
    def provider(self) -> str:
        return self.session.get_providers()[0]

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        image, scale, pad = letterbox(frame, self.size)
        blob = image[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255  # BGR -> RGB
        output = self.session.run(None, {self.input: np.ascontiguousarray(blob)})[0]
        return decode_yolo(output, self.classes, scale, pad, self.conf, self.iou)


class ColorDetector:
    """Blobs of calibrated colours, searched on a downscaled frame for speed."""

    def __init__(self, colors: list[ColorClass], min_area: int = 20, downscale: int = 4):
        self.colors, self.min_area, self.downscale = colors, min_area, downscale

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        d = self.downscale
        small = np.ascontiguousarray(frame[::d, ::d])
        result = []
        for color in self.colors:
            mask = color.mask(small)
            _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for x, y, w, h, area in stats[1:]:
                if area * d * d >= self.min_area:
                    box = (float(x * d), float(y * d), float((x + w) * d), float((y + h) * d))
                    result.append(Detection(color.name, box, 1.0))
        return result
