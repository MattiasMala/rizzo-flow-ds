"""Frame sources. Every source returns (BGR uint8 frame, capture time in perf_counter_ns).

- `DxcamSource` (Windows): DXGI Desktop Duplication, frames straight from the GPU compositor;
  the lowest-latency option (a new frame is available as soon as the game presents it).
- `MssSource`: portable screenshot of a screen region (Windows, Linux X11/XWayland, macOS);
  slower. Native Wayland sessions do not allow it (only through the screencast portal).
- `FileSource`: images from disk, for offline tests and calibration.

`LatestFrame` runs a source on its own thread and keeps only the newest frame, so processing
never works on a stale queue: if analysis is slower than the game, old frames are dropped.
"""

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

Region = tuple[int, int, int, int]  # left, top, right, bottom in screen pixels


def find_window(title: str = "Dead Cells") -> Region | None:
    """Client area of the game window: Win32 on Windows, xdotool on X11 (also XWayland
    windows); None on native Wayland or when the window is not open."""
    if sys.platform == "win32":
        return _find_window_win32(title)
    if sys.platform.startswith("linux") and os.environ.get("DISPLAY") and shutil.which("xdotool"):
        run = subprocess.run(
            [
                "xdotool",
                "search",
                "--onlyvisible",
                "--name",
                f"^{title}$",
                "getwindowgeometry",
                "--shell",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        values = dict(line.split("=", 1) for line in run.stdout.splitlines() if "=" in line)
        if {"X", "Y", "WIDTH", "HEIGHT"} <= values.keys():
            x, y, w, h = (int(values[k]) for k in ("X", "Y", "WIDTH", "HEIGHT"))
            return x, y, x + w, y + h
    return None


def _find_window_win32(title: str) -> Region | None:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    try:
        user32.SetProcessDPIAware()  # physical pixels, not scaled ones
    except OSError:
        pass
    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        return None
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    corner = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(corner))
    return corner.x, corner.y, corner.x + rect.right, corner.y + rect.bottom


class DxcamSource:
    def __init__(self, region: Region | None = None, fps: int = 60):
        import dxcam

        self.camera = dxcam.create(output_color="BGR")
        self.camera.start(region=region, target_fps=fps, video_mode=True)

    def grab(self) -> tuple[np.ndarray, int]:
        frame = self.camera.get_latest_frame()  # blocks until the next frame
        return frame, time.perf_counter_ns()

    def close(self) -> None:
        self.camera.stop()


class MssSource:
    def __init__(self, region: Region | None = None):
        self.region = region
        self.mss = None  # created on the grabbing thread: mss handles are thread-local on Windows

    def grab(self) -> tuple[np.ndarray, int]:
        if self.mss is None:
            import mss

            self.mss = mss.mss()
            if self.region is None:
                self.monitor = self.mss.monitors[1]
            else:
                left, top, right, bottom = self.region
                self.monitor = {
                    "left": left,
                    "top": top,
                    "width": right - left,
                    "height": bottom - top,
                }
        shot = self.mss.grab(self.monitor)
        t = time.perf_counter_ns()
        return np.asarray(shot)[:, :, :3], t  # BGRA -> BGR view, no copy

    def close(self) -> None:
        if self.mss is not None:
            self.mss.close()


class FileSource:
    def __init__(self, paths: list[str | Path], loop: bool = False):
        import cv2

        self.frames = []
        for path in paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Cannot read image {path}")
            self.frames.append(frame)
        self.loop, self.index = loop, 0

    def grab(self) -> tuple[np.ndarray, int]:
        if self.index >= len(self.frames):
            if not self.loop:
                raise StopIteration
            self.index = 0
        frame = self.frames[self.index]
        self.index += 1
        return frame, time.perf_counter_ns()

    def close(self) -> None:
        pass


def open_source(kind: str, region: Region | None, fps: int):
    if kind == "auto":
        kind = "dxcam" if sys.platform == "win32" else "mss"
    if kind == "dxcam":
        return DxcamSource(region, fps)
    if kind == "mss":
        return MssSource(region)
    raise ValueError(f"Unknown source {kind!r}: use auto, dxcam or mss")


class LatestFrame:
    def __init__(self, source):
        self.source = source
        self.frame: tuple[np.ndarray, int] | None = None
        self.dropped = 0
        self._new = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            try:
                item = self.source.grab()
            except StopIteration:
                break
            with self._new:
                if self.frame is not None:
                    self.dropped += 1
                self.frame = item
                self._new.notify()

    def get(self, timeout: float = 1.0) -> tuple[np.ndarray, int] | None:
        with self._new:
            if self.frame is None:
                self._new.wait(timeout)
            item, self.frame = self.frame, None
            return item

    def close(self) -> None:
        self._stop = True
        self._thread.join(timeout=1)
        self.source.close()
