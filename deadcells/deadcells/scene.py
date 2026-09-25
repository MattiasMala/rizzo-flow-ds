"""Scene description: HUD, map and tracked objects -> the compact JSON state sent to the model.

Positions are relative to the player, in player heights (right and up are positive), because
that unit survives any resolution and matches how far an attack reaches. Lists are sorted by
urgency and capped, and every cap is declared with the full count: nothing is dropped silently.
"""

import math

from .hud import Hud
from .minimap import MapState
from .track import Track

THREATS = {"enemy", "elite", "boss", "projectile", "telegraph", "trap"}
LOOT = {"weapon_drop", "scroll", "food", "gold", "cell", "chest", "breakable_wall"}
PLACES = {"door", "exit", "teleporter", "npc"}
SAME_LEVEL = 0.5  # |dy| below this many player heights counts as the same level


def relative(track: Track, origin: tuple[float, float], unit: float) -> dict:
    cx, cy = track.detection.centre
    dx, dy = (cx - origin[0]) / unit, (origin[1] - cy) / unit
    item = {
        "id": track.id,
        "type": track.detection.cls,
        "dx": round(dx, 1),
        "dy": round(dy, 1),
        "distance": round(math.hypot(dx, dy), 1),
        "side": "right" if dx > 0 else "left",
        "level": "same" if abs(dy) < SAME_LEVEL else ("above" if dy > 0 else "below"),
    }
    if track.seen > 1:
        vx, vy = track.vx / unit, -track.vy / unit  # player heights per second, up positive
        closing = -(dx * vx + dy * vy) / max(math.hypot(dx, dy), 1e-6)
        item["speed"] = round(math.hypot(vx, vy), 1)
        item["approaching"] = closing > 0.5
        if closing > 0.5:
            item["seconds_to_reach"] = round(math.hypot(dx, dy) / closing, 2)
    return item


def urgency(item: dict) -> tuple:
    return (item.get("seconds_to_reach", math.inf), item["distance"])


def capped(items: list[dict], limit: int) -> dict:
    return {"total": len(items), "listed": items[:limit]}


def describe(
    frame_size: tuple[int, int],
    hud: Hud,
    map_state: MapState | None,
    tracks: list[Track],
    calibrated: bool,
    limit: int = 6,
) -> dict:
    width, height = frame_size
    players = [t for t in tracks if t.detection.cls == "player"]
    if players:
        player = max(players, key=lambda t: t.detection.score).detection
        origin, unit, estimated = player.centre, max(player.height, 1.0), False
    else:
        # The camera keeps the player near the centre; the unit is a typical player height.
        origin, unit, estimated = (width / 2, height / 2), height / 12, True

    groups: dict[str, list[dict]] = {"threats": [], "loot": [], "places": []}
    for track in tracks:
        cls = track.detection.cls
        if cls == "player":
            continue
        group = "threats" if cls in THREATS else "loot" if cls in LOOT else "places"
        groups[group].append(relative(track, origin, unit))
    for items in groups.values():
        items.sort(key=urgency)

    state: dict = {
        "units": "positions relative to the player in player heights; dx>0 right, dy>0 up",
        "player": {
            "health_percent": None if hud.health is None else round(hud.health * 100),
            "flask_charges": hud.flask_charges,
            "cells": hud.cells,
            "gold": hud.gold,
            "position_estimated": estimated,
        },
        **{name: capped(items, limit) for name, items in groups.items()},
    }
    if map_state is not None:
        state["map"] = {
            "player_found": map_state.player is not None,
            "explored_cells": map_state.explored_cells,
            "separate_areas": map_state.rooms,
            "targets": capped(
                [
                    {
                        "type": t.name,
                        "path_cells": t.path_length,
                        "route": ", ".join(f"{step} {n}" for step, n in t.route) or None,
                        "dx": t.cell[0] - map_state.player[0] if map_state.player else None,
                        "dy": map_state.player[1] - t.cell[1] if map_state.player else None,
                    }
                    for t in map_state.targets
                ],
                limit,
            ),
        }
    if not calibrated:
        state["warning"] = "screen layout not calibrated: HUD and map values may be wrong"
    return state
