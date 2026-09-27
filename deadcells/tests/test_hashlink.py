import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from deadcells.hashlink import HashLink
from deadcells.memory import LinuxProcessMemory

DATA = Path(__file__).parent / "data" / "hltest"
HL = os.environ.get("HASHLINK_BIN") or shutil.which("hl")

pytestmark = pytest.mark.skipif(
    not (sys.platform.startswith("linux") and HL), reason="needs Linux and a HashLink VM"
)


@pytest.fixture(scope="module")
def game():
    proc = subprocess.Popen(
        [HL, str(DATA / "test.hl")], stdout=subprocess.PIPE, cwd=Path(HL).parent
    )
    assert proc.stdout.readline().strip() == b"ready"
    time.sleep(0.2)
    yield HashLink(LinuxProcessMemory(pid=proc.pid))
    proc.kill()


def test_class_layout_matches_the_vm(game):
    hero = game.find_class("en.Hero")
    layout = {f.name: (f.owner, f.kind, f.offset) for f in hero.fields}
    assert layout["cx"] == ("en.Entity", "i32", 8)  # after the 8-byte type pointer
    assert layout["xr"][1:] == ("f64", 16)
    assert layout["flasks"][0] == "en.Hero"
    assert layout["speed"][1] == "f32" and layout["onGround"][1] == "bool"
    assert game.class_at(hero.parent).name == "en.Entity"
    assert hero.size == 72


def test_instances_and_values(game):
    (hero,) = game.instances(game.find_class("en.Hero"))[:1]
    values = game.dump(hero)
    assert values["cx"] == 12 and values["cy"] == 34 and values["life"] == 100
    assert values["name"] == "hero" and values["flasks"] == 3 and values["dx"] == -1.5
    assert values["onGround"] is True and values["speed"] == 2.5 and 0 <= values["xr"] < 1
    mobs = game.instances(game.find_class("en.Mob"))
    assert sorted(game.dump(m)["name"] for m in mobs[:5]) == [f"mob{i}" for i in range(5)]
    assert all(game.dump(m)["target"] == f"en.Hero@{hero:#x}" for m in mobs[:5])


def test_follow_references_and_arrays(game):
    world = game.instances(game.find_class("Game"))[0]
    fields = {f.name: f for f in game.class_at(game.ptr(world)).fields}
    hero = game.read_field(world, fields["hero"])
    assert game.dump(hero)["name"] == "hero"
    mobs = game.array_items(game.read_field(world, fields["mobs"]))
    assert [game.dump(m)["name"] for m in mobs] == [f"mob{i}" for i in range(5)]


def test_live_field_changes(game):
    hero = game.instances(game.find_class("en.Hero"))[0]
    xr = next(f for f in game.find_class("en.Hero").fields if f.name == "xr")
    seen = {round(game.read_field(hero, xr), 2) for _ in range(20) if not time.sleep(0.02)}
    assert len(seen) > 3  # the program moves the hero every 10 ms


def test_static_singleton_without_scanning(game):
    statics = game.statics(game.find_class("Game"))
    assert game.class_at(game.ptr(statics)).name == "$Game"
    me = game.dump(statics)["ME"]
    world = game.instances(game.find_class("Game"))[0]
    assert me == f"Game@{world:#x}"


def test_field_paths_and_int_arrays(game):
    statics = game.statics(game.find_class("Game"))
    world = game.get(statics, "ME")
    assert game.get(world, "hero", "name") == "hero"
    assert (game.get(world, "wid"), game.get(world, "hei")) == (6, 4)
    cells = game.array_bytes(game.get(world, "collisions"))
    assert cells.tolist() == [1 if i % 3 == 0 else 0 for i in range(24)]
