"""One-shot check-up on the machine that runs the game, for remote development.

Development happens in a cloud session that cannot see the player's PC: this command gathers
in one folder everything that session needs (system, game install, capture timings, a
screenshot, bridge status, and the class/field names of the game's HashLink bytecode, which
is what a mod hooks into). The player commits the folder and pushes; the cloud session pulls
it. Every step is independent and reports its own error instead of stopping the probe.
"""

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

GAME_DIR = "Dead Cells"
KEYWORDS = (
    "hero",
    "level",
    "entity",
    "mob",
    "game",
    "collision",
    "camera",
    "player",
    "boss",
    "lifebar",
    "inventory",
    "item",
    "room",
    "door",
    "teleport",
    "loot",
    "bullet",
    "hud",
    "physics",
    "map",
    "cell",
    "flask",
    "scroll",
)


def step(report: dict, name: str):
    """Run one probe step, storing its result or its error under `name`."""

    def run(fn):
        started = time.perf_counter()
        try:
            report[name] = fn()
        except Exception as exc:  # noqa: BLE001 - every failure is reported, none stops the probe
            report[name] = {
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=3),
            }
        report.setdefault("_seconds", {})[name] = round(time.perf_counter() - started, 3)

    return run


def steam_libraries() -> list[Path]:
    roots: list[Path] = []
    if sys.platform == "win32":
        import winreg

        for hive, key in (
            (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam"),
        ):
            try:
                with winreg.OpenKey(hive, key) as k:
                    value = winreg.QueryValueEx(
                        k, "SteamPath" if hive == winreg.HKEY_CURRENT_USER else "InstallPath"
                    )[0]
                    roots.append(Path(value))
            except OSError:
                pass
    else:
        roots += [Path.home() / ".steam/steam", Path.home() / ".local/share/Steam"]
    libraries = []
    for root in roots:
        vdf = root / "steamapps" / "libraryfolders.vdf"
        if vdf.exists():
            text = vdf.read_text(encoding="utf-8", errors="replace")
            libraries += [
                Path(p.replace("\\\\", "\\")) for p in re.findall(r'"path"\s+"(.+?)"', text)
            ]
        libraries.append(root)
    return list(dict.fromkeys(libraries))


def steam_app(libraries: list[Path]) -> dict | None:
    """Dead Cells' Steam app id and folder from the app manifests (robust to renamed folders)."""
    for library in libraries:
        for manifest in sorted((library / "steamapps").glob("appmanifest_*.acf")):
            text = manifest.read_text(encoding="utf-8", errors="replace")
            name = re.search(r'"name"\s+"(.+?)"', text)
            if name and name.group(1) == GAME_DIR:
                appid = re.search(r'"appid"\s+"(\d+)"', text).group(1)
                folder = re.search(r'"installdir"\s+"(.+?)"', text).group(1)
                return {
                    "appid": appid,
                    "library": library,
                    "root": library / "steamapps" / "common" / folder,
                    "proton_prefix": library / "steamapps" / "compatdata" / appid,
                }
    return None


def find_game(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    libraries = steam_libraries()
    app = steam_app(libraries)
    if app and app["root"].exists():
        return app["root"]
    for library in libraries:
        candidate = library / "steamapps" / "common" / GAME_DIR
        if candidate.exists():
            return candidate
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def game_files(root: Path) -> dict:
    files = {}
    names = (
        "hlboot.dat",
        "deadcells",
        "deadcells.exe",
        "deadcells_gl.exe",
        "libhl.so",
        "libhl.dll",
        "res.pak",
    )
    for name in names:
        path = root / name
        if path.is_file():
            with open(path, "rb") as f:
                magic = f.read(4)
            files[name] = {
                "bytes": path.stat().st_size,
                "format": "ELF" if magic == b"\x7fELF" else "PE" if magic[:2] == b"MZ" else None,
                "sha256": sha256(path) if path.stat().st_size < 512 << 20 else None,
            }
    app = steam_app(steam_libraries())
    return {
        "root": str(root),
        "build": "linux-native"
        if "deadcells" in files
        else "windows (Proton)"
        if "deadcells.exe" in files
        else "unknown",
        "steam_appid": app and app["appid"],
        "proton_prefix_exists": bool(app and app["proton_prefix"].exists()),
        "files": files,
        "top_level": sorted(p.name for p in root.iterdir())[:80],
        "core_modding_installed": (root / "coremod").exists(),
    }


def dump_types(hlboot: Path, out: Path, keywords=KEYWORDS) -> dict:
    """Class names, superclasses and own fields of the HashLink bytecode, filtered by keyword."""
    from crashlink import Bytecode, Obj
    from crashlink.disasm import type_name

    code = Bytecode.from_path(str(hlboot))
    lines, total, kept = [], 0, 0
    for index, typ in enumerate(code.types):
        definition = typ.definition
        if not isinstance(definition, Obj):
            continue
        total += 1
        name = definition.name.resolve(code)
        if not any(k in name.lower() for k in keywords):
            continue
        kept += 1
        parent = ""
        if definition.super.value >= 0:
            parent = " extends " + type_name(code, definition.super.resolve(code))
        lines.append(f"class {name}{parent}  // t@{index}")
        for field in definition.fields:
            lines.append(
                f"    {field.name.resolve(code)}: {type_name(code, field.type.resolve(code))}"
            )
        for proto in definition.protos:
            lines.append(f"    fn {proto.name.resolve(code)}()")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"file": out.name, "classes_total": total, "classes_kept": kept, "keywords": keywords}


def capture(folder: Path, frames: int = 60) -> dict:
    import cv2

    from .capture import find_window, open_source

    region = find_window()
    try:
        source = open_source("auto", region, 60)
        times, frame = [], None
        try:
            for _ in range(frames):
                t = time.perf_counter()
                frame, _ = source.grab()
                times.append((time.perf_counter() - t) * 1000)
        finally:
            source.close()
    except Exception as exc:  # noqa: BLE001 - e.g. mss on native Wayland: keep a screenshot
        tool = screenshot_fallback(folder / "screen.png")
        return {
            "window": region,
            "live_capture_error": f"{type(exc).__name__}: {exc}",
            "screenshot": "screen.png" if tool else None,
            "screenshot_tool": tool,
        }
    cv2.imwrite(str(folder / "screen.png"), frame)
    ordered = sorted(times)
    return {
        "window": region,
        "frame": list(frame.shape),
        "screenshot": "screen.png",
        "grab_ms_p50": round(ordered[len(ordered) // 2], 2),
        "grab_ms_p95": round(ordered[int(0.95 * (len(ordered) - 1))], 2),
    }


def screenshot_fallback(out: Path) -> str | None:
    """A still screenshot with the desktop's own tool (slow, but works on Wayland)."""
    commands = {
        "grim": ["grim", str(out)],  # wlroots: Sway, Hyprland
        "spectacle": ["spectacle", "-b", "-n", "-f", "-o", str(out)],  # KDE
        "gnome-screenshot": ["gnome-screenshot", "-f", str(out)],
    }
    for tool, command in commands.items():
        if shutil.which(tool):
            subprocess.run(command, capture_output=True, check=False, timeout=30)
            if out.exists():
                return tool
    return None


def linux() -> dict:
    from .memory import LinuxProcessMemory, basename, find_linux_process, ptrace_hint

    info = {
        "session": os.environ.get("XDG_SESSION_TYPE"),
        "desktop": os.environ.get("XDG_CURRENT_DESKTOP"),
        "wayland_display": os.environ.get("WAYLAND_DISPLAY"),
        "x_display": os.environ.get("DISPLAY"),
        "tools": {
            t: bool(shutil.which(t))
            for t in ("xdotool", "grim", "spectacle", "gnome-screenshot", "scanmem", "gdb")
        },
        "dev_shm_writable": os.access("/dev/shm", os.W_OK),
        "uinput_writable": os.access("/dev/uinput", os.W_OK),
    }
    try:
        info["ptrace_scope"] = Path("/proc/sys/kernel/yama/ptrace_scope").read_text().strip()
    except OSError:
        info["ptrace_scope"] = None
    try:
        pid, mode = find_linux_process()
    except LookupError as exc:
        info["game_process"] = str(exc)
        return info
    argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    info["game_process"] = {"pid": pid, "mode": mode, "argv0": argv[0].decode(errors="replace")}
    maps = Path(f"/proc/{pid}/maps").read_text(errors="replace").splitlines()
    mapped = sorted(
        {basename(line.split(maxsplit=5)[5]) for line in maps if len(line.split(maxsplit=5)) == 6}
    )
    info["modules_of_interest"] = [
        m
        for m in mapped
        if any(k in m for k in ("deadcells", "hl", "sdl", "openal", "fmt", "ui.", "mono", "dotnet"))
    ]
    info["modules_total"] = len(mapped)
    try:
        memory = LinuxProcessMemory(pid=pid)
        main = next(m for m in info["modules_of_interest"] if m.startswith("deadcells"))
        head = memory.read(memory.module_base(main), 4)
        info["memory_read"] = {"ok": True, "module": main, "magic": head.hex()}
        memory.close()
    except Exception as exc:  # noqa: BLE001
        info["memory_read"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "hint": ptrace_hint(),
        }
    return info


def module(args: list[str], timeout: int = 300) -> dict:
    run = subprocess.run(
        [sys.executable, "-m", "deadcells", *args],
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
        cwd=Path(__file__).resolve().parents[1],
    )
    if run.returncode:
        return {"error": run.stderr[-2000:]}
    return json.loads(run.stdout)


def bridge() -> dict:
    from .bridge import BridgeReader

    reader = BridgeReader()
    snap = None
    for _ in range(200):
        snap = reader.read()
        if snap is not None:
            break
        time.sleep(0.005)
    if snap is None:
        return {"status": "no new frame in 1 s"}
    return {
        "frame": snap.frame,
        "level_version": snap.level_version,
        "grid": None if snap.grid is None else list(snap.grid.shape),
        "hero": vars(snap.hero),
        "entities": len(snap.entities),
    }


def system() -> dict:
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    if shutil.which("nvidia-smi"):
        info["nvidia"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
        ).stdout.strip()
    if sys.platform.startswith("linux") and shutil.which("lspci"):
        lines = subprocess.run(
            ["lspci"], capture_output=True, check=False, text=True, timeout=20
        ).stdout.splitlines()
        info["gpus"] = [line for line in lines if "VGA" in line or "3D controller" in line]
    for package in ("numpy", "cv2", "scipy", "mss", "dxcam", "onnxruntime", "crashlink"):
        try:
            info[package] = getattr(__import__(package), "__version__", "installed")
        except Exception:  # noqa: BLE001
            info[package] = None
    return info


def main(args) -> Path:
    folder = Path(args.out) / datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")  # local time
    folder.mkdir(parents=True, exist_ok=False)
    report: dict = {"probe_version": 2}
    step(report, "system")(system)
    if sys.platform.startswith("linux"):
        step(report, "linux")(linux)
    game = find_game(args.game)
    step(report, "game")(
        lambda: game_files(game) if game else {"error": "Dead Cells not found; pass --game PATH"}
    )
    if game and (game / "hlboot.dat").exists() and not args.skip_types:
        step(report, "types")(lambda: dump_types(game / "hlboot.dat", folder / "types.txt"))
    if not args.skip_capture:
        step(report, "capture")(lambda: capture(folder))
    step(report, "bridge")(bridge)
    if not args.skip_bench:
        step(report, "bench_screen")(lambda: module(["bench", "--frames", "100"]))
        step(report, "bench_nav")(lambda: module(["bench-nav", "--frames", "100"]))
    (folder / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return folder
