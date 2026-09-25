import json
import struct
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from deadcells.bridge import ENTITY_DTYPE, KINDS, BridgeReader, BridgeWriter, Hero
from deadcells.decide import combat_request
from deadcells.memory import Chain, MemorySource
from deadcells.nav import parse_ascii
from deadcells.world import Navigator

from deadcells import bridge

ROOT = Path(__file__).resolve().parents[2]

LEVEL = """
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
.S......................X.....
##############################
"""


def hero(x, y, **kw):
    base = {
        "vx": 0.0,
        "vy": 0.0,
        "hp": 80,
        "hp_max": 100,
        "flask": 2,
        "flask_max": 3,
        "cells": 12,
        "gold": 340,
        "flags": 1,
        "brutality": 3,
        "tactics": 1,
        "survival": 2,
        "curse": 0,
    }
    return Hero(x, y, **{**base, **kw})


def entities(*rows):
    out = np.zeros(len(rows), ENTITY_DTYPE)
    for i, (kind, x, y, vx) in enumerate(rows):
        out[i] = (i + 1, KINDS.index(kind), 0b100 if kind == "enemy" else 0, x, y, vx, 0, 10, 10)
    return out


@pytest.fixture
def level():
    grid, marks = parse_ascii(LEVEL)
    return grid, marks


def test_round_trip_and_grid_cached(tmp_path, level):
    grid, _ = level
    path = tmp_path / "bridge.bin"
    writer = BridgeWriter(path, grid.shape[1], grid.shape[0], capacity=8)
    writer.write(hero(1.5, 8.99), entities(("enemy", 10.5, 8.9, -4.0)), 1.25, grid)
    reader = BridgeReader(path)
    snap = reader.read()
    assert snap.frame == 1 and snap.game_time == 1.25 and snap.level_version == 1
    assert snap.hero.hp == 80 and snap.hero.flag("on_ground") and not snap.hero.flag("on_ladder")
    assert snap.hero.curse == 0 and snap.hero.survival == 2
    assert np.array_equal(snap.grid, grid)
    assert KINDS[snap.entities[0]["kind"]] == "enemy"
    assert reader.read() is None and reader.stale == 1  # no new frame yet
    writer.write(hero(2.5, 8.99), entities(), 1.3)  # same level: grid not rewritten
    again = reader.read()
    assert again.grid is snap.grid and again.hero.x == 2.5 and len(again.entities) == 0
    with pytest.raises(ValueError, match="capacity"):
        writer.write(hero(0, 0), entities(*[("gold", 0, 0, 0)] * 9), 2.0)
    writer.close()
    reader.close()


def test_torn_write_is_retried_then_times_out(tmp_path, level):
    grid, _ = level
    path = tmp_path / "bridge.bin"
    writer = BridgeWriter(path, grid.shape[1], grid.shape[0], capacity=4)
    writer.write(hero(1.5, 8.99), entities(), 0.0, grid)
    struct.pack_into("<I", writer.buf, 8, writer.seq + 1)  # writer stuck mid-write
    writer.buf.flush()
    reader = BridgeReader(path, retries=5)
    with pytest.raises(TimeoutError):
        reader.read()
    assert reader.torn == 5


def test_missing_or_foreign_block(tmp_path):
    empty = tmp_path / "empty.bin"
    empty.write_bytes(bytes(64))
    with pytest.raises(ConnectionError, match="mod"):
        BridgeReader(empty).read()
    other = tmp_path / "other.bin"
    other.write_bytes(b"NOPE" + bytes(60))
    with pytest.raises(ValueError, match="bridge"):
        BridgeReader(other).read()


def test_layout_matches_the_documented_offsets():
    assert bridge.HEADER.size == bridge.HERO.size == 64 and bridge.ENTITY.size == 32
    size, entities_at, grid_at = bridge.layout_size(10, 5, 3)
    assert (entities_at, grid_at, size) == (128, 128 + 96, 128 + 96 + 50)


class FakeMemory:
    def __init__(self):
        self.bytes = {}

    def put(self, address, data):
        for i, b in enumerate(data):
            self.bytes[address + i] = b

    def read(self, address, size):
        try:
            return bytes(self.bytes[address + i] for i in range(size))
        except KeyError:
            raise OSError(f"unmapped {address:#x}") from None

    def module_base(self, name):
        return {"libhl.dll": 0x10000}[name.lower()]


def test_pointer_chain_like_cheat_engine():
    mem = FakeMemory()
    mem.put(0x10000 + 0x40, struct.pack("<Q", 0x5000))  # [module+0x40] -> 0x5000
    mem.put(0x5000 + 0x18, struct.pack("<Q", 0x9000))  # [0x5000+0x18] -> 0x9000
    mem.put(0x9000 + 0x8, struct.pack("<d", 42.5))  # hero x at 0x9000+0x8 (Haxe Float)
    mem.put(0x9000 + 0x10, struct.pack("<i", 77))
    spec = {
        "fields": {
            "hero_x": {"module": "libhl.dll", "base": "0x40", "offsets": ["0x18", "0x8"]},
            "hp": {
                "module": "libhl.dll",
                "base": "0x40",
                "offsets": ["0x18", "0x10"],
                "type": "i32",
            },
            "broken": {"module": "libhl.dll", "base": "0x48", "offsets": ["0x0", "0x0"]},
        }
    }
    values = MemorySource(spec, mem).read()
    assert values == {"hero_x": 42.5, "hp": 77, "broken": None}
    with pytest.raises(ValueError):
        Chain.parse({"module": "m", "base": 0, "type": "f16"})


def test_navigator_routes_to_the_exit_and_describes(tmp_path, level):
    grid, marks = level
    (sr, sc), (xr, xc) = marks["S"][0], marks["X"][0]
    writer = BridgeWriter(tmp_path / "b.bin", grid.shape[1], grid.shape[0], capacity=8)
    writer.write(
        hero(sc + 0.5, sr + 0.9),
        entities(("exit", xc + 0.5, xr + 0.9, 0.0), ("enemy", sc + 6.5, sr + 0.9, -5.0)),
        0.0,
        grid,
    )
    snap = BridgeReader(tmp_path / "b.bin").read()
    nav = Navigator()
    state = nav.update(snap, "exit")
    route = state["route"]
    assert route["next_moves"][0]["dir"] == "right" and route["seconds"] > 0
    threat = state["threats"]["listed"][0]
    assert threat["type"] == "enemy" and threat["dx"] == 6.0 and threat["approaching"]
    assert threat["seconds_to_reach"] == pytest.approx(6 / 5, abs=0.01)
    assert threat["flags"] == ["hostile"]
    assert state["places"]["listed"][0]["type"] == "exit"
    assert "graph_build" in nav.timings_ms
    json.dumps(state)
    # The enemy on the floor makes walking through it costlier than without it.
    calm = Navigator().update(_without_enemies(snap), "exit")
    assert calm["route"]["seconds"] < route["seconds"]
    sys.path.insert(0, str(ROOT / "src"))
    schema = pytest.importorskip("rizzo_flow.schema")
    schema.Request.model_validate(combat_request(state))


def _without_enemies(snap):
    from dataclasses import replace

    keep = snap.entities[snap.entities["kind"] != KINDS.index("enemy")]
    return replace(snap, entities=keep)


def test_danger_map_matches_brute_force(tmp_path, level):
    grid, marks = level
    (sr, sc) = marks["S"][0]
    writer = BridgeWriter(tmp_path / "d.bin", grid.shape[1], grid.shape[0], capacity=8)
    rows = [
        ("enemy", sc + 4.2, sr + 0.9, 0.0),
        ("projectile", sc + 9.7, sr - 0.4, 3.0),
        ("gold", sc + 2.5, sr + 0.9, 0.0),
        ("trap", 0.2, sr + 0.9, 0.0),
    ]
    writer.write(hero(sc + 0.5, sr + 0.9), entities(*rows), 0.0, grid)
    snap = BridgeReader(tmp_path / "d.bin").read()
    nav = Navigator()
    nav.update(snap, None)
    penalty = nav._penalty(snap)
    centres = nav.graph.cells[:, ::-1] + 0.5
    expected = np.zeros(len(centres))
    for kind, x, y, _ in rows:
        if kind != "gold":
            d = np.hypot(centres[:, 0] - x, centres[:, 1] - y)
            expected += nav.danger_cost * np.clip(1 - d / nav.danger_radius, 0, None)
    assert np.allclose(penalty, expected, atol=1e-5)
