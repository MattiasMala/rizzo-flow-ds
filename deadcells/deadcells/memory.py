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

Spec file (JSON), none known yet for Dead Cells:

    {"process": "deadcells.exe",
     "fields": {"hero_x": {"module": "libhl.dll", "base": "0x0", "offsets": ["0x0"], "type": "f64"}}}
"""

import json
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


class ProcessMemory:
    """ReadProcessMemory on a running process (Windows only)."""

    def __init__(self, process: str = "deadcells.exe"):
        if sys.platform != "win32":
            raise OSError("Reading another process's memory is implemented for Windows only")
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
        return cls(spec, memory or ProcessMemory(spec.get("process", "deadcells.exe")))

    def read(self) -> dict[str, float | int | None]:
        result = {}
        for name, chain in self.fields.items():
            try:
                result[name] = chain.read(self.memory)
            except (OSError, LookupError):
                result[name] = None  # a broken chain is reported, never guessed
        return result
