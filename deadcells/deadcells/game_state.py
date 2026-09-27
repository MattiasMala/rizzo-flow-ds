"""Dead Cells state straight from the game's objects (HashLink), no mod and no screen.

Chain of references, as listed in the game's classes (probe types.txt, v3.5 Steam build):
`pr.Game.ME` (static) -> `hero: en.Hero`, `curLevel: pr.Level` -> `map: level.LevelMap`
(`wid`, `hei`, `collisions: ArrayBytes_Int`, `id`) and `entities: ArrayObj<Entity>`.
Every entity has the Deepnight-engine position: cell `cx, cy` plus fraction `xr, yr` inside the
cell, speed `dx, dy`, and `life`, `maxLife`, `destroyed`.

The meaning of the collision values and the orientation of the axes are not verified yet: the
probe records them (histogram, grid, hero samples over time) so they can be mapped to nav.py.
"""

import time
from collections import Counter

import numpy as np

from .hashlink import HashLink

ENTITY_FIELDS = ("cx", "cy", "xr", "yr", "dx", "dy", "life", "maxLife", "destroyed")


class DeadCellsReader:
    def __init__(self, hl: HashLink):
        self.hl = hl
        self.game_class = hl.find_class("pr.Game")
        self._statics = hl.statics(self.game_class)
        if not self._statics:
            raise LookupError("pr.Game has no statics: is a run started?")
        self._map = 0
        self._grid: np.ndarray | None = None
        self.map_changes = 0

    def game(self) -> int:
        return self.hl.get(self._statics, "ME") or 0

    def entity(self, obj: int) -> dict:
        values = self.hl.values(obj, ENTITY_FIELDS)
        values["class"] = self.hl.class_at(self.hl.ptr(obj)).name
        return values

    def grid(self, level_map: int) -> np.ndarray:
        """Collision grid (hei × wid), read again only when the level map object changes."""
        if level_map != self._map or self._grid is None:
            wid, hei = self.hl.get(level_map, "wid"), self.hl.get(level_map, "hei")
            cells = self.hl.array_bytes(self.hl.get(level_map, "collisions"))
            if cells.size != wid * hei:
                raise ValueError(f"collisions has {cells.size} cells, map is {wid}x{hei}")
            self._grid = cells.reshape(hei, wid)
            self._map = level_map
            self.map_changes += 1
        return self._grid

    def snapshot(self, with_entities: bool = True) -> dict | None:
        game = self.game()
        if not game:
            return None
        hl = self.hl
        hero = hl.get(game, "hero")
        level = hl.get(game, "curLevel")
        state = {"hero": self.entity(hero) if hero else None, "level": None, "entities": []}
        if level:
            level_map = hl.get(level, "map")
            state["level"] = {
                "id": hl.get(level_map, "id") if level_map else None,
                "map": level_map,
                "mobs_left": hl.get(level, "nbMobsLeft"),
                "mobs_total": hl.get(level, "nbTotalMobs"),
            }
            if with_entities:
                items = hl.array_items(hl.get(level, "entities"))
                state["entities"] = [
                    e for e in (self.entity(o) for o in items if o) if not e["destroyed"]
                ]
        return state


def probe_report(
    hl: HashLink, seconds: float = 8.0, hz: float = 60.0, wait: float = 0.0
) -> tuple[dict, object]:
    """What the probe records: layout, level grid statistics, entities, hero samples."""
    reader = DeadCellsReader(hl)
    started = time.perf_counter()
    state = reader.snapshot()
    first_ms = (time.perf_counter() - started) * 1000
    if state is None:
        return {"error": "pr.Game.ME is null: start a run"}
    report = {
        "hero": state["hero"],
        "level": {k: v for k, v in state["level"].items() if k != "map"}
        if state["level"]
        else None,
    }
    grid = None
    if state["level"] and state["level"]["map"]:
        grid = reader.grid(state["level"]["map"])
        values, counts = np.unique(grid, return_counts=True)
        report["collisions"] = {
            "shape": list(grid.shape),
            "values": {int(v): int(c) for v, c in zip(values, counts)},
        }
    hero = state["hero"]
    if hero:
        near = sorted(
            state["entities"], key=lambda e: abs(e["cx"] - hero["cx"]) + abs(e["cy"] - hero["cy"])
        )
        report["entities"] = {
            "alive": len(state["entities"]),
            "by_class": dict(Counter(e["class"] for e in state["entities"]).most_common(25)),
            "nearest": near[:15],
        }
    timings = []
    for _ in range(50):
        t = time.perf_counter()
        reader.snapshot()
        timings.append((time.perf_counter() - t) * 1000)
    timings.sort()
    report["snapshot_ms"] = {
        "first": round(first_ms, 2),
        "p50": round(timings[25], 3),
        "p95": round(timings[47], 3),
    }
    samples, period = [], 1 / hz
    if wait:
        print(
            f"Tra {wait:g} s registro l'eroe per {seconds:g} s: torna nel gioco, "
            "corri e salta (anche doppio salto).",
            flush=True,
        )
        time.sleep(wait)
    print(f"Recording the hero for {seconds:g} s: run and jump now.", flush=True)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        t = time.perf_counter()
        game = reader.game()
        hero_obj = hl.get(game, "hero") if game else 0
        if hero_obj:
            v = hl.values(hero_obj, ("cx", "cy", "xr", "yr", "dx", "dy"))
            samples.append([round(t - started, 4), *(round(float(x), 4) for x in v.values())])
        time.sleep(max(0.0, period - (time.perf_counter() - t)))
    report["hero_samples"] = {"columns": ["t", "cx", "cy", "xr", "yr", "dx", "dy"], "rows": samples}
    return report, grid
