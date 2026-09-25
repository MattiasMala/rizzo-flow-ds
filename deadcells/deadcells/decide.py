"""Requests for Rizzo Flow's `POST /v1/decisions` built from a scene description.

The model picks among a few options in one forward pass (~45-50 ms with the 4B Q8_0 on a
recent GPU): fine for choosing the next target on the map or reacting a few times per second,
not for frame-perfect dodging, which stays with a reflex layer (not written yet).
"""

import json
import os
import urllib.request

# Action set of a default keyboard/pad binding; ids are what an input layer would execute.
ACTIONS = {
    "attack_primary": "Attack with the main weapon.",
    "attack_secondary": "Attack with the secondary weapon or raise the shield.",
    "roll": "Dodge roll: pass through enemies and avoid damage.",
    "jump": "Jump.",
    "skill_left": "Use the left skill.",
    "skill_right": "Use the right skill.",
    "move_left": "Walk left.",
    "move_right": "Walk right.",
    "heal": "Drink a health flask.",
    "wait": "Do nothing this moment.",
}


def combat_request(state: dict, actions: dict[str, str] | None = None) -> dict:
    options = [{"id": k, "description": v} for k, v in (actions or ACTIONS).items()]
    return {
        "state": {
            "game": "Dead Cells, a 2D side-view action game. Getting hit loses health; at 0 the "
            "run ends. Projectiles and attack wind-ups (telegraph) hit if you stay in their path.",
            **state,
        },
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "Which action should the player take right now?",
                "options": options,
                "policy": {"allow_abstain": False},
            },
            "danger": {
                "type": "score",
                "instructions": "How dangerous is the player's situation right now?",
                "levels": [
                    "Safe: no threat will reach the player in the next second.",
                    "Risky: a threat is close or approaching, but there is time to react.",
                    "Critical: a hit is imminent or health is very low.",
                ],
                "policy": {"allow_abstain": False},
            },
        },
    }


def route_request(state: dict) -> dict | None:
    """Which map target to head for; None when the map shows fewer than two reachable ones."""
    targets = [
        t
        for t in state.get("map", {}).get("targets", {}).get("listed", [])
        if t["path_cells"] is not None
    ]
    if len(targets) < 2:
        return None
    options = [
        {
            "id": f"t{i}",
            "description": f"Go to the {t['type']} ({t['path_cells']} map cells away, "
            f"route: {t['route']}).",
        }
        for i, t in enumerate(targets)
    ]
    return {
        "state": {
            "game": "Dead Cells. The map shows explored areas, the exit, doors, teleporters "
            "and items. Picking up items and scrolls makes the player stronger; the exit leads "
            "to the next biome.",
            "player": state["player"],
            "map": state["map"],
        },
        "questions": {
            "target": {
                "type": "choice",
                "instructions": "Where should the player go next?",
                "options": options,
                "policy": {"allow_abstain": False},
            }
        },
    }


def post(request: dict, url: str = "http://127.0.0.1:8017", timeout: float = 10) -> dict:
    """Send a request to a running `rizzo serve` (bearer auth if RIZZO_API_KEY is set)."""
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("RIZZO_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    body = json.dumps(request).encode()
    req = urllib.request.Request(f"{url.rstrip('/')}/v1/decisions", body, headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())
