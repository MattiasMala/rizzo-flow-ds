"""External memory reading: values of the running game through pointer chains, no mod needed.

Second choice after the bridge (bridge.py): reading another process costs a system call per
pointer hop (microseconds each), and pointer chains found with a memory scanner (e.g. Cheat
Engine's pointer scan) can break when the game updates or when HashLink's JIT and garbage
collector lay objects out differently. Good for a handful of values (hero position, health);
the whole entity list and the level grid are far easier to export from a mod.

A chain follows Cheat Engine's convention: read the pointer at `module + base`, then for each
offset but the last read the pointer at `pointer + offset`; `pointer + last offset` is the
value's address (no offsets: the value sits at `module + base`).
Haxe `Float` is a 64-bit double (`f64`), `Int` a 32-bit integer (`i32`).

On Linux the same code reads `/proc/<pid>/mem` and finds module bases in `/proc/<pid>/maps`,
which covers both the native build and the Windows build under Proton (the PE modules of a
Wine process are mapped files too). Arch's default `kernel.yama.ptrace_scope = 1` only lets a
process read its own children: either launch the game from the reader, or allow it until the
next reboot with `sudo sysctl kernel.yama.ptrace_scope=0`.

Spec file (JSON), none known yet for Dead Cells ("process" may be omitted: then `deadcells`
and `deadcells.exe` are both tried):

    {"process": "deadcells",
     "fields": {"hero_x": {"module": "libhl.so", "base": "0x0", "offsets": ["0x0"], "type": "f64"}}}
"""

import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

TYPES = {"f32": "<f", "f64": "<d", "i32": "<i", "u32": "<I", "i64": "<q", "u8": "<B"}


class Memory(Protocol):
    def read(self, address: int, size: int) -> bytes: ...

    def module_base(self, name: str) -> int: ...


def number(value: int | str) -> int:
    return int(value, 0) if isinstance(value, str) else int(value)


@dataclass(frozen=True)
class Chain:
    module: str
    base: int
    offsets: tuple[int, ...]
    type: str = "f64"

    @classmethod
    def parse(cls, spec: dict) -> "Chain":
        if spec.get("type", "f64") not in TYPES:
            raise ValueError(f"type must be one of {sorted(TYPES)}")
        return cls(
            spec["module"],
            number(spec["base"]),
            tuple(number(o) for o in spec.get("offsets", [])),
            spec.get("type", "f64"),
        )

    def address(self, memory: Memory, pointer_size: int = 8) -> int:
        start = memory.module_base(self.module) + self.base
        if not self.offsets:
            return start  # a static value
        fmt = "<Q" if pointer_size == 8 else "<I"

        def deref(address: int) -> int:
            pointer = struct.unpack(fmt, memory.read(address, pointer_size))[0]
            if pointer == 0:
                raise LookupError(f"Null pointer at {address:#x} in chain {self}")
            return pointer

        pointer = deref(start)
        for offset in self.offsets[:-1]:
            pointer = deref(pointer + offset)
        return pointer + self.offsets[-1]

    def read(self, memory: Memory) -> float | int:
        fmt = TYPES[self.type]
        return struct.unpack(fmt, memory.read(self.address(memory), struct.calcsize(fmt)))[0]


GAME_PROCESSES = ("deadcells", "deadcells.exe")


def basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1].lower()


def find_linux_process(names=GAME_PROCESSES) -> tuple[int, str]:
    """(pid, how it runs) of the first process whose executable or argv[0] matches a name."""
    wanted = {n.lower() for n in names}
    for entry in sorted(Path("/proc").iterdir(), key=lambda p: p.name):
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            exe = os.readlink(entry / "exe") if (entry / "exe").exists() else ""
        except OSError:
            continue
        first = basename(argv[0].decode(errors="replace")) if argv and argv[0] else ""
        if first in wanted or basename(exe) in wanted:
            proton = first.endswith(".exe") or "wine" in basename(exe)
            return int(entry.name), "proton" if proton else "native"
    raise LookupError(f"None of {sorted(wanted)} is running")


class LinuxProcessMemory:
    """/proc/<pid>/mem reads with pread: native Linux builds and Proton alike."""

    def __init__(self, process: str | None = None, pid: int | None = None):
        self.pid = (
            pid
            if pid is not None
            else find_linux_process((process,) if process else GAME_PROCESSES)[0]
        )
        try:
            self.fd = os.open(f"/proc/{self.pid}/mem", os.O_RDONLY)
        except PermissionError as exc:
            raise PermissionError(f"Cannot read process {self.pid}: {ptrace_hint()}") from exc
        self._modules: dict[str, int] | None = None

    def module_base(self, name: str) -> int:
        if self._modules is None:
            self._modules = {}
            with open(f"/proc/{self.pid}/maps", encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.split(maxsplit=5)
                    if len(parts) == 6:
                        start = int(parts[0].split("-")[0], 16)
                        key = basename(parts[5].strip())
                        self._modules[key] = min(start, self._modules.get(key, start))
        return self._modules[name.lower()]

    def read(self, address: int, size: int) -> bytes:
        try:
            data = os.pread(self.fd, size, address)
        except PermissionError as exc:
            raise PermissionError(f"Cannot read process {self.pid}: {ptrace_hint()}") from exc
        if len(data) != size:
            raise OSError(f"Short read at {address:#x}")
        return data

    def close(self) -> None:
        os.close(self.fd)


def ptrace_hint() -> str:
    try:
        scope = Path("/proc/sys/kernel/yama/ptrace_scope").read_text().strip()
    except OSError:
        return "no permission"
    if scope == "0":
        return "ptrace_scope is 0; run as the same user as the game"
    return (
        f"kernel.yama.ptrace_scope = {scope}: launch the game from the reader, or run "
        "`sudo sysctl kernel.yama.ptrace_scope=0` (until reboot)"
    )


def open_process(process: str | None = None):
    """Memory of the running game: /proc on Linux, ReadProcessMemory on Windows."""
    return (
        WindowsProcessMemory(process or "deadcells.exe")
        if sys.platform == "win32"
        else LinuxProcessMemory(process)
    )


class WindowsProcessMemory:
    """ReadProcessMemory on a running process."""

    def __init__(self, process: str = "deadcells.exe"):
        import ctypes
        from ctypes import wintypes

        self.ctypes, self.wintypes = ctypes, wintypes
        self.k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        pid = self._find(process)
        access = 0x0010 | 0x0400  # PROCESS_VM_READ | PROCESS_QUERY_INFORMATION
        self.handle = self.k32.OpenProcess(access, False, pid)
        if not self.handle:
            raise OSError(f"OpenProcess failed ({ctypes.get_last_error()})")
        self._modules: dict[str, int] | None = None

    def _find(self, name: str) -> int:
        ctypes, wintypes = self.ctypes, self.wintypes
        pids = (wintypes.DWORD * 4096)()
        used = wintypes.DWORD()
        self.psapi.EnumProcesses(pids, ctypes.sizeof(pids), ctypes.byref(used))
        for pid in pids[: used.value // ctypes.sizeof(wintypes.DWORD)]:
            handle = self.k32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
            if not handle:
                continue
            buf = ctypes.create_unicode_buffer(260)
            size = wintypes.DWORD(260)
            ok = self.k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
            self.k32.CloseHandle(handle)
            if ok and Path(buf.value).name.lower() == name.lower():
                return pid
        raise LookupError(f"Process {name} not running")

    def module_base(self, name: str) -> int:
        if self._modules is None:
            ctypes, wintypes = self.ctypes, self.wintypes
            mods = (ctypes.c_void_p * 1024)()
            used = wintypes.DWORD()
            self.psapi.EnumProcessModulesEx(
                self.handle, mods, ctypes.sizeof(mods), ctypes.byref(used), 0x03
            )  # LIST_MODULES_ALL
            self._modules = {}
            for mod in mods[: used.value // ctypes.sizeof(ctypes.c_void_p)]:
                buf = ctypes.create_unicode_buffer(260)
                self.psapi.GetModuleBaseNameW(self.handle, ctypes.c_void_p(mod), buf, 260)
                self._modules[buf.value.lower()] = mod
        return self._modules[name.lower()]

    def read(self, address: int, size: int) -> bytes:
        ctypes = self.ctypes
        buf = ctypes.create_string_buffer(size)
        got = ctypes.c_size_t()
        if (
            not self.k32.ReadProcessMemory(
                self.handle, ctypes.c_void_p(address), buf, size, ctypes.byref(got)
            )
            or got.value != size
        ):
            raise OSError(f"ReadProcessMemory at {address:#x} failed ({ctypes.get_last_error()})")
        return buf.raw

    def close(self) -> None:
        self.k32.CloseHandle(self.handle)


class MemorySource:
    """Named values read through the chains of a spec file."""

    def __init__(self, spec: dict, memory: Memory):
        self.fields = {name: Chain.parse(f) for name, f in spec["fields"].items()}
        self.memory = memory

    @classmethod
    def from_file(cls, path: str | Path, memory: Memory | None = None) -> "MemorySource":
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(spec, memory or open_process(spec.get("process")))

    def read(self) -> dict[str, float | int | None]:
        result = {}
        for name, chain in self.fields.items():
            try:
                result[name] = chain.read(self.memory)
            except (OSError, LookupError):
                result[name] = None  # a broken chain is reported, never guessed
        return result
