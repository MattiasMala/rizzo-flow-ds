import ctypes
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from deadcells.memory import Chain, LinuxProcessMemory, basename, find_linux_process

from deadcells import probe

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux /proc")


class SelfMemory(LinuxProcessMemory):
    """Our own process, with a fake module placed at address 0 so chains use real pointers."""

    def module_base(self, name):
        return 0 if name == "fake" else super().module_base(name)


def test_reads_real_process_memory_through_a_pointer_chain():
    value = ctypes.c_double(42.5)
    hp = ctypes.c_int32(77)

    class Hero(ctypes.Structure):
        _fields_ = [("pad", ctypes.c_uint64), ("x", ctypes.c_double), ("hp", ctypes.c_int32)]

    hero = Hero(0, value.value, hp.value)
    holder = ctypes.c_uint64(ctypes.addressof(hero))  # a global pointing at the hero object
    memory = SelfMemory(pid=__import__("os").getpid())
    base = ctypes.addressof(holder)
    assert Chain("fake", base, (8,), "f64").read(memory) == 42.5
    assert Chain("fake", base, (16,), "i32").read(memory) == 77
    python = basename(Path(sys.executable).resolve().as_posix())
    libs = memory.module_base  # a real mapped file resolves to a real address
    assert any(libs(name) > 0 for name in (python, "libc.so.6") if _mapped(name))
    memory.close()


def _mapped(name):
    return name in Path("/proc/self/maps").read_text()


@pytest.fixture
def fake_games(tmp_path):
    native = tmp_path / "deadcells"
    shutil.copy(shutil.which("sleep"), native)
    procs = [subprocess.Popen([str(native), "30"])]
    procs.append(
        subprocess.Popen(
            ["bash", "-c", 'exec -a "Z:\\\\games\\\\Dead Cells\\\\deadcells.exe" sleep 30']
        )
    )
    time.sleep(0.3)
    yield procs
    for p in procs:
        p.kill()


def test_finds_native_and_proton_processes(fake_games):
    native, proton = fake_games
    assert find_linux_process(("deadcells",)) == (native.pid, "native")
    assert find_linux_process(("deadcells.exe",)) == (proton.pid, "proton")


def test_probe_linux_section_reads_the_game_memory(fake_games):
    info = probe.linux()
    assert info["game_process"]["mode"] == "native"
    assert info["memory_read"]["ok"], info["memory_read"]
    assert info["memory_read"]["magic"] == "7f454c46"  # the ELF header of the executable
