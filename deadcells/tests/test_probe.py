import json
from argparse import Namespace
from pathlib import Path

import pytest

from deadcells import probe

SAMPLES = Path(__file__).parent / "data"


def test_game_files_and_explicit_path(tmp_path):
    (tmp_path / "hlboot.dat").write_bytes(b"HLB\x04")
    (tmp_path / "coremod").mkdir()
    info = probe.game_files(probe.find_game(str(tmp_path)))
    assert info["files"]["hlboot.dat"]["bytes"] == 4
    assert len(info["files"]["hlboot.dat"]["sha256"]) == 64
    assert info["core_modding_installed"]


def test_probe_reports_every_step_even_when_they_fail(tmp_path):
    game = tmp_path / "game"
    game.mkdir()
    (game / "hlboot.dat").write_bytes(b"not bytecode")
    args = Namespace(
        game=str(game),
        out=str(tmp_path / "probes"),
        skip_types=False,
        skip_capture=False,
        skip_bench=True,
    )
    folder = probe.main(args)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert report["system"]["python"]
    assert report["game"]["files"]["hlboot.dat"]["bytes"] == 12
    assert "error" in report["types"]  # not a HashLink file (or crashlink missing)
    assert "capture" in report and "bridge" in report  # errors here too, but recorded
    assert set(report["_seconds"]) >= {"system", "game", "types", "capture", "bridge"}


def test_type_dump_on_real_hashlink_bytecode(tmp_path):
    pytest.importorskip("crashlink")
    sample = SAMPLES / "sample.hl"
    result = probe.dump_types(sample, tmp_path / "types.txt", keywords=("string",))
    text = (tmp_path / "types.txt").read_text(encoding="utf-8")
    assert result["classes_kept"] >= 1 and "class String" in text and "length: I32" in text
