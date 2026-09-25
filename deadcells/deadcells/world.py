"""Snapshot from memory + navigation graph -> scene for the model and the next move.

The level graph is rebuilt only when `level_version` changes. Enemies, projectiles and traps
add a danger cost to the nodes around them; the flow field towards the current goal is
recomputed when the goal changes or every `replan_every` frames (enemies move), and in between
the next move is a table lookup. Positions are in tiles relative to the hero (right and up
positive); the hero's cell is (floor(y), floor(x)).
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .bridge import ENTITY_FLAGS, KINDS, Snapshot
from .nav import NavGraph, Physics

HOSTILE = {"enemy", "projectile", "trap"}
LOOT = {"weapon_drop", "scroll", "food", "gold", "cell", "chest"}
PLACES = {"door", "exit", "teleporter", "npc"}


def entity_view(snap: Snapshot) -> list[dict]:
    hero = snap.hero
    result = []
    for e in snap.entities:
        kind = KINDS[e["kind"]] if e["kind"] < len(KINDS) else "unknown"
        dx, dy = float(e["x"] - hero.x), float(hero.y - e["y"])
        rvx, rvy = float(e["vx"] - hero.vx), -float(e["vy"] - hero.vy)
        dist = math.hypot(dx, dy)
        closing = -(dx * rvx + dy * rvy) / max(dist, 1e-6)
        item = {
            "id": int(e["id"]),
            "type": kind,
            "dx": round(dx, 1),
            "dy": round(dy, 1),
            "distance": round(dist, 1),
            "side": "right" if dx > 0 else "left",
        }
        flags = [f for i, f in enumerate(ENTITY_FLAGS) if e["flags"] >> i & 1]
        if flags:
            item["flags"] = flags
        if e["hp_max"] > 0:
            item["health_percent"] = round(100 * int(e["hp"]) / int(e["hp_max"]))
        if closing > 0.5:
            item["approaching"] = True
            item["seconds_to_reach"] = round(dist / closing, 2)
        result.append(item)
    return result


@dataclass
class Navigator:
    physics: Physics = field(default_factory=Physics)
    danger_radius: float = 3.0  # tiles around a hostile entity that cost extra
    danger_cost: float = 1.5  # seconds-equivalent at the entity's own cell
    replan_every: int = 6
    limit: int = 6
    graph: NavGraph | None = None
    timings_ms: dict = field(default_factory=dict)
    _version: int = -1
    _field: tuple | None = None
    _goal: int = -1
    _age: int = 0

    def _penalty(self, snap: Snapshot) -> np.ndarray | None:
        """Danger cost per node: linear falloff within `danger_radius` of each hostile entity,
        computed only on the window of cells around it."""
        g = self.graph
        rows, cols = g.node_id.shape
        penalty = None
        reach = math.ceil(self.danger_radius)
        for e in snap.entities:
            if e["kind"] >= len(KINDS) or KINDS[e["kind"]] not in HOSTILE:
                continue
            r0, c0 = math.floor(e["y"]), math.floor(e["x"])
            ra, rb = max(r0 - reach, 0), min(r0 + reach + 1, rows)
            ca, cb = max(c0 - reach, 0), min(c0 + reach + 1, cols)
            if ra >= rb or ca >= cb:
                continue
            ids = g.node_id[ra:rb, ca:cb]
            rr, cc = np.nonzero(ids >= 0)
            if not len(rr):
                continue
            d = np.hypot(cc + ca + 0.5 - e["x"], rr + ra + 0.5 - e["y"])
            if penalty is None:
                penalty = np.zeros(len(g.cells))
            np.add.at(
                penalty,
                ids[rr, cc],
                self.danger_cost * np.clip(1 - d / self.danger_radius, 0, None),
            )
        return penalty

    def _goal_node(self, snap: Snapshot, target: str | tuple[int, int] | None) -> int:
        if target is None:
            return -1
        if isinstance(target, tuple):
            return self.graph.node(*target)
        hero = snap.hero
        best, best_d = -1, math.inf
        for e in snap.entities:
            if e["kind"] < len(KINDS) and KINDS[e["kind"]] == target:
                node = self.graph.node(int(e["y"]), int(e["x"]))
                d = math.hypot(e["x"] - hero.x, e["y"] - hero.y)
                if node >= 0 and d < best_d:
                    best, best_d = node, d
        return best

    def update(self, snap: Snapshot, target: str | tuple[int, int] | None = "exit") -> dict:
        t0 = time.perf_counter()
        if snap.level_version != self._version:
            if snap.grid is None:
                raise ValueError("Snapshot without a level grid")
            self.graph = NavGraph(snap.grid, self.physics)
            self._version, self._field = snap.level_version, None
            self.timings_ms["graph_build"] = round((time.perf_counter() - t0) * 1000, 2)
        t1 = time.perf_counter()
        g = self.graph
        goal = self._goal_node(snap, target)
        self._age += 1
        if goal >= 0 and (
            self._field is None or goal != self._goal or self._age >= self.replan_every
        ):
            self._field = g.flow_field(goal, self._penalty(snap))
            self._goal, self._age = goal, 0
        t2 = time.perf_counter()
        hero = snap.hero
        here = g.node(math.floor(hero.y), math.floor(hero.x))
        route = None
        if goal >= 0 and here >= 0:
            dist, nxt = self._field
            path = g.follow(dist, nxt, here, limit=48)  # enough edges for the next moves
            if path is not None:
                route = {
                    "target": target if isinstance(target, str) else list(target),
                    "seconds": round(float(dist[here]), 2),
                    "next_moves": g.actions(path)[:3],
                }
        entities = entity_view(snap)
        groups = {"threats": [], "loot": [], "places": []}
        for item in entities:
            kind = item["type"]
            key = "threats" if kind in HOSTILE else "loot" if kind in LOOT else "places"
            groups[key].append(item)
        for items in groups.values():
            items.sort(key=lambda i: (i.get("seconds_to_reach", math.inf), i["distance"]))
        state = {
            "units": "tiles relative to the hero; dx>0 right, dy>0 up",
            "hero": {
                "health_percent": round(100 * hero.hp / hero.hp_max) if hero.hp_max else None,
                "flask_charges": hero.flask,
                "cells": hero.cells,
                "gold": hero.gold,
                "stats": {
                    "brutality": hero.brutality,
                    "tactics": hero.tactics,
                    "survival": hero.survival,
                },
                "on_ground": hero.flag("on_ground"),
            },
            **{k: {"total": len(v), "listed": v[: self.limit]} for k, v in groups.items()},
            "route": route,
        }
        self.timings_ms["flow_field"] = round((t2 - t1) * 1000, 3)
        self.timings_ms["describe"] = round((time.perf_counter() - t2) * 1000, 3)
        return state
