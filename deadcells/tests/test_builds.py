import sys
from pathlib import Path

import pytest

from deadcells import builds

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def data():
    return builds.load("builds.json")


@pytest.fixture(scope="module")
def biomes():
    return builds.load("biomes.json")


def schema():
    sys.path.insert(0, str(ROOT / "src"))
    return pytest.importorskip("rizzo_flow.schema")


def test_build_data_is_consistent(data):
    assert builds.validate(data) == []
    assert {b["color"] for b in data["builds"]} == set(builds.STATS)


def test_validate_catches_unknown_names(data):
    broken = {**data, "builds": [{**data["builds"][0], "weapons": ["Excalibur"]}]}
    assert any("Excalibur" in p for p in builds.validate(broken))


def test_off_colour_items_are_flagged(data):
    bow = next(b for b in data["builds"] if b["id"] == "bow")
    assert builds.off_colour(data, bow) == ["Grappling Hook"]  # Brutality skill in a Tactics build


def test_table_and_tree_render_every_build(data):
    table, tree = builds.table(data), builds.tree(data)
    for b in data["builds"]:
        assert b["name"] in table and b["name"] in tree
    assert table.count("\n") == len(data["builds"]) + 2
    assert "Quick Bow [Tattica]" in tree


def test_fits_rank_the_matching_build_first(data):
    ranked = builds.fits(data, ["Quick Bow", "Marksman's Bow", "Wolf Trap", "Barbed Tips"])
    assert ranked[0].build["id"] == "bow"
    assert ranked[0].owned == ["Barbed Tips", "Marksman's Bow", "Quick Bow", "Wolf Trap"]
    assert 0 < ranked[0].score <= 1
    assert builds.fits(data, [])[0].score == 0


def test_decision_requests_validate(data):
    rizzo = schema()
    owned, stats = ["Oiled Sword", "Fire Grenade"], {"brutality": 4, "tactics": 2, "survival": 1}
    for request in (
        builds.pickup_request(data, owned, stats, ["Torch", "Quick Bow"]),
        builds.scroll_request(data, owned, stats),
        builds.scroll_request(data, owned, stats, ("brutality", "survival")),
        builds.mutation_request(data, owned, stats, ["Combo", "Support", "Necromancy"]),
    ):
        rizzo.Request.model_validate(request)
    pickup = builds.pickup_request(data, owned, stats, ["Torch", "Unknown Thing"])
    texts = [o["description"] for o in pickup["questions"]["pickup"]["options"]]
    assert "Fire / Oil" in texts[0] and "not in the build table" in texts[1]
    assert pickup["state"]["closest_builds"][0]["build"] == "Fire / Oil"


def test_biome_graph(biomes):
    names = {e["from"] for e in biomes["exits"]} | {e["to"] for e in biomes["exits"]}
    assert biomes["start"] in names
    assert set(biomes["bosses"]) <= names
    for edge in biomes["exits"]:
        assert edge["rune"] is None or edge["rune"] in biomes["runes"]
    # Every biome is reachable from the start.
    seen, todo = set(), [biomes["start"]]
    while todo:
        b = todo.pop()
        if b not in seen:
            seen.add(b)
            todo += [e["to"] for e in biomes["exits"] if e["from"] == b]
    assert seen == names


def test_exits_apply_runes_and_boss_cells(biomes):
    start = biomes["start"]
    open_now = {e["to"] for e in builds.exits(biomes, start, set(), 0) if e["open"]}
    assert "Promenade of the Condemned" in open_now and "Toxic Sewers" not in open_now
    with_vine = {e["to"] for e in builds.exits(biomes, start, {"Vine"}, 0) if e["open"]}
    assert "Toxic Sewers" in with_vine
    depths = {e["to"]: e["open"] for e in builds.exits(biomes, "Prison Depths", set(), 0)}
    assert depths["Ancient Sewers"] is False
    depths = {e["to"]: e["open"] for e in builds.exits(biomes, "Prison Depths", set(), 1)}
    assert depths["Ancient Sewers"] is True
    # "2+ BSC and the Giant killed once": the extra clause is checked when `unlocked` is given.
    haven = {
        e["to"]: e["open"] for e in builds.exits(biomes, "Forgotten Sepulcher", set(), 3, set())
    }
    assert haven["Guardian's Haven"] is False
    haven = {
        e["to"]: e["open"]
        for e in builds.exits(biomes, "Forgotten Sepulcher", set(), 3, {"the Giant killed once"})
    }
    assert haven["Guardian's Haven"] is True


def test_biome_request(biomes):
    rizzo = schema()
    request = builds.biome_request(
        biomes, "Promenade of the Condemned", {"Teleportation"}, 0, {"health_percent": 70}
    )
    rizzo.Request.model_validate(request)
    texts = [o["description"] for o in request["questions"]["next_biome"]["options"]]
    assert texts == ["Go to Ramparts.", "Go to Ossuary.", "Go to Morass of the Banished."]
    assert builds.biome_request(biomes, "Clock Tower", set(), 0, {}) is None  # single exit
