"""Build tree, biome graph and the between-fights decisions that use them.

Data lives in `data/builds.json` and `data/biomes.json` (sources and check date inside). This
module validates it, renders the table and the tree, measures how far the current inventory is
from each build, and turns pickup / scroll / mutation / next-biome choices into requests for
Rizzo Flow. These decisions are not time-critical, which is where the model fits best.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).parent / "data"
STATS = ("brutality", "tactics", "survival")
LABEL = {
    "brutality": "Brutalità",
    "tactics": "Tattica",
    "survival": "Sopravvivenza",
    "colorless": "incolore",
}


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def validate(data: dict) -> list[str]:
    """Problems in the build data (empty list = consistent)."""
    problems = []
    items, mutations = data["items"], data["mutations"]
    for name, item in items.items():
        if item["type"] not in ("weapon", "skill"):
            problems.append(f"{name}: unknown type {item['type']}")
        if not item["scaling"] or any(s not in STATS for s in item["scaling"]):
            problems.append(f"{name}: bad scaling {item['scaling']}")
    for name, colour in mutations.items():
        if colour not in (*STATS, "colorless"):
            problems.append(f"mutation {name}: bad colour {colour}")
    ids = [b["id"] for b in data["builds"]]
    if len(ids) != len(set(ids)):
        problems.append("duplicate build ids")
    for build in data["builds"]:
        if build["color"] not in STATS:
            problems.append(f"{build['id']}: bad colour {build['color']}")
        for kind in ("weapons", "skills"):
            for name in build[kind]:
                if name not in items:
                    problems.append(f"{build['id']}: unknown item {name}")
                elif items[name]["type"] != kind[:-1]:
                    problems.append(f"{build['id']}: {name} is a {items[name]['type']}")
        for name in build["mutations"]:
            if name not in mutations:
                problems.append(f"{build['id']}: unknown mutation {name}")
        if not build["sources"]:
            problems.append(f"{build['id']}: no source")
    return problems


def scaling(data: dict, name: str) -> str:
    return "/".join(LABEL[s] for s in data["items"][name]["scaling"])


def off_colour(data: dict, build: dict) -> list[str]:
    """Items of a build that do not scale with the build's colour (they use another stat)."""
    return [
        n
        for n in build["weapons"] + build["skills"]
        if build["color"] not in data["items"][n]["scaling"]
    ]


def table(data: dict) -> str:
    rows = [
        "| Build | Colore | Armi | Abilità | Mutazioni | Affissi | Stile di gioco |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for b in sorted(data["builds"], key=lambda b: (STATS.index(b["color"]), b["name"])):
        name = b["name"] + (" ¹" if b["composed"] else "")
        muts = ", ".join(f"{m} ({LABEL[data['mutations'][m]]})" for m in b["mutations"])
        off = set(off_colour(data, b))
        weapons, skills = (
            ", ".join(f"{n}{' ²' if n in off else ''}" for n in b[kind])
            for kind in ("weapons", "skills")
        )
        rows.append(
            f"| {name} | {LABEL[b['color']]} | {weapons} | {skills} | {muts} "
            f"| {', '.join(b['affixes']) or '—'} | {b['playstyle']} |"
        )
    return "\n".join(rows) + "\n"


def tree(data: dict) -> str:
    lines = []
    for stat in STATS:
        lines.append(f"{LABEL[stat]} ({data['stats'][stat]['colour']})")
        builds = [b for b in data["builds"] if b["color"] == stat]
        for i, b in enumerate(builds):
            last = i == len(builds) - 1
            lines.append(f"{'└──' if last else '├──'} {b['name']}")
            pad = "    " if last else "│   "
            parts = [
                ("armi", [f"{n} [{scaling(data, n)}]" for n in b["weapons"]]),
                ("abilità", [f"{n} [{scaling(data, n)}]" for n in b["skills"]]),
                ("mutazioni", b["mutations"]),
            ]
            for j, (label, names) in enumerate(parts):
                lines.append(
                    f"{pad}{'└──' if j == len(parts) - 1 else '├──'} {label}: " + ", ".join(names)
                )
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class Fit:
    build: dict
    owned: list[str]  # items and mutations of the build the player already has
    score: float  # share of the build's weapon, skill and mutation groups already covered


def fits(data: dict, owned: list[str]) -> list[Fit]:
    """Builds ranked by how much of them the inventory covers (one item per group counts)."""
    have = set(owned)
    result = []
    for b in data["builds"]:
        groups = [set(b["weapons"]), set(b["skills"]), set(b["mutations"])]
        got = sorted(have & set().union(*groups))
        # Two weapons and two skills can be equipped: count up to two per group, three mutations.
        caps = (2, 2, 3)
        covered = sum(min(len(g & have), cap) for g, cap in zip(groups, caps))
        total = sum(min(len(g), cap) for g, cap in zip(groups, caps))
        result.append(Fit(b, got, covered / total))
    return sorted(result, key=lambda f: -f.score)


def item_note(data: dict, name: str) -> str:
    if name not in data["items"] and name not in data["mutations"]:
        return f"{name} (not in the build table)."
    if name in data["mutations"]:
        colour = data["mutations"][name]
        head = f"{name}, mutation ({colour})"
    else:
        item = data["items"][name]
        head = f"{name}, {item['type']} scaling with {' or '.join(item['scaling'])}"
    used = [
        b["name"] for b in data["builds"] if name in b["weapons"] + b["skills"] + b["mutations"]
    ]
    return f"{head}; part of builds: {', '.join(used)}." if used else f"{head}."


def context(data: dict, owned: list[str], stats: dict[str, int], top: int = 3) -> dict:
    ranked = fits(data, owned)[:top]
    return {
        "rules": data["rules"],
        "stats": stats,
        "inventory": [item_note(data, n) for n in owned],
        "closest_builds": [
            {
                "build": f.build["name"],
                "colour": f.build["color"],
                "coverage_percent": round(100 * f.score),
                "already_owned": f.owned,
                "core_weapons": f.build["weapons"],
                "core_skills": f.build["skills"],
                "mutations": f.build["mutations"],
                "playstyle": f.build["playstyle"],
            }
            for f in ranked
        ],
    }


def pickup_request(data: dict, owned: list[str], stats: dict[str, int], offered: list[str]) -> dict:
    """Take one of the items on offer (it replaces the one in its slot) or leave them all."""
    options = [
        {"id": f"take{i}", "description": f"Take it: {item_note(data, n)}"}
        for i, n in enumerate(offered)
    ]
    options.append({"id": "leave", "description": "Leave everything and keep the current gear."})
    return {
        "state": {"game": "Dead Cells: choosing gear during a run.", **context(data, owned, stats)},
        "questions": {
            "pickup": {
                "type": "choice",
                "instructions": "Which option makes the run stronger, given the stats and the builds "
                "closest to the current inventory?",
                "options": options,
                "policy": {"allow_abstain": False},
            }
        },
    }


def scroll_request(
    data: dict, owned: list[str], stats: dict[str, int], choices: tuple[str, ...] = STATS
) -> dict:
    """Which stat a Scroll of Power (or a dual scroll, by passing its two stats) should raise."""
    options = [
        {
            "id": s,
            "description": f"+1 {s} ({data['stats'][s]['colour']}): {data['stats'][s]['themes']}.",
        }
        for s in choices
    ]
    return {
        "state": {"game": "Dead Cells: spending a Scroll of Power.", **context(data, owned, stats)},
        "questions": {
            "scroll": {
                "type": "choice",
                "instructions": "Which stat should this scroll raise to make the current gear hit "
                "hardest while keeping the player alive?",
                "options": options,
                "policy": {"allow_abstain": False},
            }
        },
    }


def mutation_request(
    data: dict, owned: list[str], stats: dict[str, int], offered: list[str]
) -> dict:
    options = [{"id": f"m{i}", "description": item_note(data, n)} for i, n in enumerate(offered)]
    return {
        "state": {"game": "Dead Cells: choosing a mutation.", **context(data, owned, stats)},
        "questions": {
            "mutation": {
                "type": "choice",
                "instructions": "Which mutation fits the current gear and stats best?",
                "options": options,
                "policy": {"allow_abstain": False},
            }
        },
    }


BOSS_CELLS = re.compile(r"^(\d+)\+? Boss Stem Cells?(?: and (.+))?$")


def exits(
    biomes: dict, biome: str, runes: set[str], boss_cells: int, unlocked: set[str] | None = None
) -> list[dict]:
    """Exits of a biome with an `open` flag. Runes and Boss Stem Cells are checked. Other
    conditions ("after beating Dracula", keys, outfits) are checked against `unlocked` when it
    is given; with None they are assumed met and left in the option text for the model."""
    result = []
    for edge in biomes["exits"]:
        if edge["from"] != biome:
            continue
        open_ = edge["rune"] is None or edge["rune"] in runes
        cond = edge["condition"]
        if cond and (match := BOSS_CELLS.match(cond)):
            open_ = open_ and boss_cells >= int(match.group(1))
            cond = match.group(2)
        if cond and unlocked is not None and cond not in unlocked:
            open_ = False
        result.append({**edge, "open": open_})
    return result


def biome_request(
    biomes: dict,
    biome: str,
    runes: set[str],
    boss_cells: int,
    player: dict,
    unlocked: set[str] | None = None,
) -> dict | None:
    """Which exit to take; None when fewer than two exits are open."""
    candidates = [e for e in exits(biomes, biome, runes, boss_cells, unlocked) if e["open"]]
    if len(candidates) < 2:
        return None
    options = []
    for i, e in enumerate(candidates):
        text = f"Go to {e['to']}"
        if e["to"] in biomes["bosses"]:
            text += f" (boss: {biomes['bosses'][e['to']]})"
        if e["condition"]:
            text += f" [{e['condition']}]"
        if e["note"]:
            text += f" ({e['note']})"
        options.append({"id": f"b{i}", "description": text + "."})
    return {
        "state": {
            "game": "Dead Cells: choosing the next biome.",
            "current_biome": biome,
            "runes": sorted(runes),
            "boss_stem_cells": boss_cells,
            "player": player,
        },
        "questions": {
            "next_biome": {
                "type": "choice",
                "instructions": "Which exit gives the best chance to finish the run, given the "
                "player's health, gear and difficulty?",
                "options": options,
                "policy": {"allow_abstain": False},
            }
        },
    }
