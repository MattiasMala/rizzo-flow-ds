import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from deadcells.game_state import DeadCellsReader, probe_report
from deadcells.hashlink import HashLink
from deadcells.memory import LinuxProcessMemory

MOCK = Path(__file__).parent / "data" / "dcmock" / "dcmock.hl"
HL = os.environ.get("HASHLINK_BIN") or shutil.which("hl")

pytestmark = pytest.mark.skipif(
    not (sys.platform.startswith("linux") and HL), reason="needs Linux and a HashLink VM"
)


@pytest.fixture(scope="module")
def hl():
    proc = subprocess.Popen([HL, str(MOCK)], stdout=subprocess.PIPE, cwd=Path(HL).parent)
    assert proc.stdout.readline().strip() == b"ready"
    time.sleep(0.2)
    yield HashLink(LinuxProcessMemory(pid=proc.pid))
    proc.kill()


def test_snapshot_follows_game_me(hl):
    reader = DeadCellsReader(hl)
    state = reader.snapshot()
    hero = state["hero"]
    assert (hero["class"], hero["cx"], hero["cy"], hero["life"], hero["maxLife"]) == (
        "en.Hero",
        3,
        8,
        250,
        250,
    )
    assert state["level"]["id"] == "PrisonStart"
    assert (state["level"]["mobs_left"], state["level"]["mobs_total"]) == (2, 3)
    classes = sorted(e["class"] for e in state["entities"])
    assert classes == ["en.Hero", "en.Mob", "en.Mob"]  # the destroyed mob is left out
    grid = reader.grid(state["level"]["map"])
    assert grid.shape == (10, 20) and grid[9].all() and grid[:, 0].all() and grid[4, 5] == 0
    assert reader.grid(state["level"]["map"]) is grid and reader.map_changes == 1


def test_probe_report_records_everything(hl, capsys):
    report, grid = probe_report(hl, seconds=0.3, hz=100)
    assert report["collisions"]["values"] == {0: int((grid == 0).sum()), 1: int(grid.sum())}
    assert report["entities"]["alive"] == 3 and report["entities"]["by_class"]["en.Mob"] == 2
    rows = report["hero_samples"]["rows"]
    assert len(rows) >= 10 and len({r[3] for r in rows}) > 3  # xr moves in the mock
    assert report["snapshot_ms"]["p50"] < 20
    assert "jump" in capsys.readouterr().out
