"""Platformer pathfinding on the level's tile grid.

The level is a grid of tiles (row 0 at the top). A navigation graph is built once per level:
nodes are the cells where the hero can stand or hang (floor below, or a ladder/vine), edges are
the moves that connect them, each with a cost in seconds:

- walk one tile left/right;
- climb one tile on a ladder;
- drop through a one-way platform;
- walk off a ledge and fall;
- jump: parabolic arcs for every apex height up to the double-jump height and every horizontal
  distance the air speed allows, swept with the hero's body height. The first floor met while
  descending is the landing (as in the game); an arc still airborne at take-off height keeps
  falling straight down. Ceilings block the arc (no ledge grab: conservative);
- links: teleporters or anything else given explicitly.

Edges are generated for all nodes at once with numpy (one vectorised step per sampled arc cell),
then searched with A* (one query) or a reverse Dijkstra "flow field" (distance to a goal from every
node: the next move is a table lookup each frame). Extra per-node costs (enemies, projectiles)
steer the search without rebuilding the graph.

The physics defaults are estimates, not measured in Dead Cells: measure them from the hero's
position over time (memory bridge) and pass a `Physics` with the real values.
"""

import heapq
import math
from dataclasses import dataclass

import numpy as np

EMPTY, SOLID, PLATFORM, LADDER, HAZARD = 0, 1, 2, 3, 4
ASCII = {".": EMPTY, " ": EMPTY, "#": SOLID, "=": PLATFORM, "H": LADDER, "^": HAZARD}
WALK, JUMP, FALL, DROP, CLIMB, LINK = range(6)
KIND = {WALK: "walk", JUMP: "jump", FALL: "fall", DROP: "drop", CLIMB: "climb", LINK: "link"}


@dataclass(frozen=True)
class Physics:
    """Hero movement in tiles and seconds. Estimates until measured on the game."""

    run_speed: float = 9.0
    air_speed: float = 9.0  # max horizontal speed in the air (full air control assumed)
    jump_height: float = 3.0  # apex of a single jump, tiles
    double_jump_height: float = 5.5  # apex with the double jump
    gravity: float = 80.0  # tiles/s^2
    max_fall_speed: float = 35.0
    climb_speed: float = 5.0
    body_height: int = 2  # tiles the hero occupies (standing cell and the ones above)
    half_width: float = 0.35  # half the hero's width, tiles: the arc checks every column it spans
    jump_overhead: float = 0.05  # extra seconds per jump (input, landing)
    hazard_cost: float = 3.0  # seconds-equivalent per hazard cell entered

    def fall_time(self, distance: np.ndarray | float) -> np.ndarray | float:
        d = np.asarray(distance, dtype=np.float64)
        v, g = self.max_fall_speed, self.gravity
        d_acc = v * v / (2 * g)
        return np.where(d <= d_acc, np.sqrt(2 * np.maximum(d, 0) / g), v / g + (d - d_acc) / v)

    @property
    def max_speed(self) -> float:
        """Upper bound of the hero's speed: keeps the A* heuristic admissible."""
        up = math.sqrt(2 * self.gravity * self.double_jump_height)
        return math.hypot(max(self.run_speed, self.air_speed), max(self.max_fall_speed, up))


def parse_ascii(text: str) -> tuple[np.ndarray, dict[str, list[tuple[int, int]]]]:
    """Grid from ASCII art; letters other than H mark points (row, col) and are empty tiles."""
    lines = [line for line in text.strip("\n").splitlines()]
    width = max(len(line) for line in lines)
    grid = np.zeros((len(lines), width), dtype=np.uint8)
    marks: dict[str, list[tuple[int, int]]] = {}
    for r, line in enumerate(lines):
        for c, ch in enumerate(line.ljust(width)):
            if ch in ASCII:
                grid[r, c] = ASCII[ch]
            else:
                marks.setdefault(ch, []).append((r, c))
    return grid, marks


def arc(physics: Physics, apex: float, dx: int) -> tuple[list[tuple[int, int, bool, bool]], float]:
    """Cells swept by a jump reaching `apex` and covering `dx` tiles by the time it is back at
    take-off height, and that time. Each cell is (dx, dy up, descending, centre column): the
    centre column is where the feet land, the others only have to be free."""
    g, hw = physics.gravity, physics.half_width
    v0 = math.sqrt(2 * g * apex)
    t_end = 2 * v0 / g
    vx = dx / t_end
    step = 0.25 / max(abs(vx), v0, 1.0)
    cells: list[tuple[int, int, bool, bool]] = []
    seen: set[tuple[int, int, bool]] = set()
    t = step
    while t <= t_end + 1e-9:
        x, y = vx * t, v0 * t - g * t * t / 2
        # Feet at height y are in the cell whose bottom is floor(y): a 5.5-tile apex reaches
        # a 5-tile ledge, not a 6-tile one.
        oy, down, centre = math.floor(y + 1e-9), v0 - g * t < 0, round(x)
        for ox in sorted({round(x - hw), centre, round(x + hw)}, key=lambda c: c != centre):
            key = (ox, oy, ox == centre)
            if key not in seen:
                seen.add(key)
                cells.append((ox, oy, down, ox == centre))
        t += step
    if (dx, 0, True) not in seen:
        cells.append((dx, 0, True, True))
    return cells, t_end


class NavGraph:
    def __init__(
        self,
        grid: np.ndarray,
        physics: Physics | None = None,
        links: list[tuple[tuple[int, int], tuple[int, int], float]] = (),
    ):
        physics = physics or Physics()
        self.grid, self.physics = grid, physics
        passable = grid != SOLID
        # clear: the hero's body fits with its feet in this cell.
        clear = passable.copy()
        for k in range(1, physics.body_height):
            clear[k:] &= passable[:-k]
            clear[:k] = False
        support = np.zeros_like(passable)
        support[:-1] = (grid[1:] == SOLID) | (grid[1:] == PLATFORM)
        # The top of a ladder or vine is a floor: that is how the hero climbs out of it.
        support[:-1] |= (grid[1:] == LADDER) & (grid[:-1] != LADDER)
        self.clear = clear
        self.standable = clear & support
        self.ladder = clear & (grid == LADDER)
        self.hazard = grid == HAZARD
        nodes = self.standable | self.ladder
        self.node_id = np.full(grid.shape, -1, dtype=np.int32)
        self.cells = np.argwhere(nodes)  # (row, col) per node
        self.node_id[nodes] = np.arange(len(self.cells), dtype=np.int32)
        self.fall_to = self._fall_to()
        self._build(list(links))

    def _fall_to(self) -> np.ndarray:
        """Row where a hero falling straight down from each cell lands (-1: never)."""
        rows, _ = self.grid.shape
        result = np.full(self.grid.shape, -1, dtype=np.int32)
        below = np.full(self.grid.shape[1], -1, dtype=np.int32)
        for r in range(rows - 1, -1, -1):
            here = np.where(self.standable[r], r, below)
            below = np.where(self.clear[r], here, -1)
            result[r] = below
        return result

    def _build(self, links) -> None:
        p = self.physics
        rows, cols = self.grid.shape
        r0, c0 = self.cells[:, 0], self.cells[:, 1]
        parts: list[tuple] = []  # (src, dst, cost, kind, dx, apex)

        def add(src, dst, cost, kind, dx=0, apex=0.0):
            if len(src):
                n = len(src)
                parts.append(
                    (
                        src,
                        dst,
                        np.broadcast_to(cost, n).astype(np.float64),
                        np.full(n, kind, np.int8),
                        np.full(n, dx, np.int16),
                        np.full(n, apex, np.float32),
                    )
                )

        def at(r, c):
            ok = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
            return ok, np.clip(r, 0, rows - 1), np.clip(c, 0, cols - 1)

        ids = np.arange(len(self.cells), dtype=np.int32)
        standing = self.standable[r0, c0]
        for d in (-1, 1):
            ok, r, c = at(r0, c0 + d)
            dst = np.where(ok, self.node_id[r, c], -1)
            walk = standing & (dst >= 0) & self.standable[r, c]
            add(
                ids[walk],
                dst[walk],
                1 / p.run_speed + p.hazard_cost * self.hazard[r, c][walk],
                WALK,
                d,
            )
            # Walk off a ledge: the next cell is free with no floor, fall straight from there.
            off = standing & ok & self.clear[r, c] & ~self.standable[r, c]
            land = np.where(off, self.fall_to[r, c], -1)
            m = land >= 0
            dst = self.node_id[land[m], c[m]]
            cost = (
                1 / p.run_speed
                + p.fall_time(land[m] - r0[m])
                + p.hazard_cost * self.hazard[land[m], c[m]]
            )
            add(ids[m], dst, cost, FALL, d)
        for d in (-1, 1):
            ok, r, c = at(r0 + d, c0)
            dst = np.where(ok, self.node_id[r, c], -1)
            climb = (self.ladder[r0, c0] | (ok & self.ladder[r, c])) & (dst >= 0)
            add(ids[climb], dst[climb], 1 / p.climb_speed, CLIMB)
        # Drop through a one-way platform under the hero.
        on_platform = standing & (r0 + 1 < rows)
        on_platform &= self.grid[np.minimum(r0 + 1, rows - 1), c0] == PLATFORM
        land = np.where(on_platform, self.fall_to[np.minimum(r0 + 1, rows - 1), c0], -1)
        m = land >= 0
        add(ids[m], self.node_id[land[m], c0[m]], p.fall_time(land[m] - r0[m]) + 0.1, DROP)

        apexes = sorted(
            {*range(1, math.floor(p.double_jump_height) + 1), p.jump_height, p.double_jump_height}
        )
        for apex in apexes:
            t_end = 2 * math.sqrt(2 * p.gravity * apex) / p.gravity
            reach = math.floor(p.air_speed * t_end)
            for dx in range(-reach, reach + 1):
                self._jump(add, ids[standing], r0[standing], c0[standing], apex, dx)

        for (ra, ca), (rb, cb), cost in links:
            a, b = self.node_id[ra, ca], self.node_id[rb, cb]
            if a < 0 or b < 0:
                raise ValueError(f"Link endpoint is not a node: {(ra, ca)} -> {(rb, cb)}")
            add(np.array([a]), np.array([b]), max(cost, 1e-6), LINK)
        self.has_links = bool(links)

        src, dst, cost, kind, dx, apex = (np.concatenate(x) for x in zip(*parts))
        keep = src != dst
        src, dst, cost, kind, dx, apex = (x[keep] for x in (src, dst, cost, kind, dx, apex))
        # Keep the cheapest edge per (src, dst), then store as CSR by source.
        order = np.lexsort((cost, dst, src))
        src, dst, cost, kind, dx, apex = (x[order] for x in (src, dst, cost, kind, dx, apex))
        first = np.ones(len(src), bool)
        first[1:] = (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])
        self.src, self.dst, self.cost = src[first], dst[first], cost[first]
        self.kind, self.dx, self.apex = kind[first], dx[first], apex[first]
        self.offsets = np.searchsorted(self.src, np.arange(len(self.cells) + 1))
        self._keys = self.src.astype(np.int64) * len(self.cells) + self.dst  # sorted

    def _jump(self, add, ids, r0, c0, apex, dx) -> None:
        p = self.physics
        rows, cols = self.grid.shape
        cells, t_end = arc(p, apex, dx)
        alive = np.ones(len(ids), bool)
        hazards = np.zeros(len(ids), np.int16)
        overhead = p.jump_overhead * (2 if apex > p.jump_height else 1)  # second press
        g, v0 = p.gravity, math.sqrt(2 * p.gravity * apex)
        for ox, oy, descending, centre in cells:
            r, c = r0 - oy, c0 + ox
            inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
            rc, cc = np.clip(r, 0, rows - 1), np.clip(c, 0, cols - 1)
            alive &= inside & self.clear[rc, cc]
            hazards += alive & self.hazard[rc, cc]
            if descending and centre:
                landed = alive & self.standable[rc, cc]
                if landed.any():
                    # Time when the arc comes down to height oy.
                    t = (v0 + math.sqrt(max(v0 * v0 - 2 * g * oy, 0))) / g
                    add(
                        ids[landed],
                        self.node_id[rc[landed], cc[landed]],
                        t + overhead + p.hazard_cost * hazards[landed],
                        JUMP,
                        dx,
                        apex,
                    )
                    alive &= ~landed
            elif centre and oy > 0:
                grab = alive & self.ladder[rc, cc]  # optional: grab a ladder or vine mid-air
                if grab.any():
                    t = (v0 - math.sqrt(max(v0 * v0 - 2 * g * oy, 0))) / g
                    add(ids[grab], self.node_id[rc[grab], cc[grab]], t + overhead, JUMP, dx, apex)
        # Still in the air at take-off height: fall straight down from there.
        r, c = r0, c0 + dx
        inside = (c >= 0) & (c < cols)
        cc = np.clip(c, 0, cols - 1)
        land = np.where(alive & inside, self.fall_to[r, cc], -1)
        m = land >= 0
        if m.any():
            cost = (
                t_end
                + overhead
                + p.fall_time(land[m] - r[m])
                + p.hazard_cost * (hazards[m] + self.hazard[land[m], cc[m]])
            )
            add(ids[m], self.node_id[land[m], cc[m]], cost, JUMP, dx, apex)

    @property
    def edges(self) -> int:
        return len(self.src)

    def node(self, row: int, col: int) -> int:
        """Node of a cell; a cell in the air maps to where the hero lands falling from it."""
        if self.node_id[row, col] >= 0:
            return int(self.node_id[row, col])
        land = self.fall_to[row, col]
        return int(self.node_id[land, col]) if land >= 0 else -1

    def _heuristic(self, goal: int):
        if self.has_links:  # teleporters break any distance bound
            return lambda n: 0.0
        gr, gc = self.cells[goal]
        speed = self.physics.max_speed
        cells = self.cells
        return lambda n: math.hypot(cells[n, 0] - gr, cells[n, 1] - gc) / speed

    def astar(self, start: int, goal: int, penalty: np.ndarray | None = None) -> list[int] | None:
        """Cheapest edge sequence (edge indices) from start to goal; None if unreachable.
        `penalty[node]` is added when entering that node (danger near enemies, ...)."""
        h = self._heuristic(goal)
        best = {start: 0.0}
        came: dict[int, int] = {}
        queue = [(h(start), 0.0, start)]
        offsets, dst, cost = self.offsets, self.dst, self.cost
        while queue:
            _, g, n = heapq.heappop(queue)
            if n == goal:
                path = []
                while n != start:
                    e = came[n]
                    path.append(e)
                    n = int(self.src[e])
                return path[::-1]
            if g > best.get(n, math.inf):  # stale queue entry
                continue
            for e in range(offsets[n], offsets[n + 1]):
                m = int(dst[e])
                ng = g + cost[e] + (penalty[m] if penalty is not None else 0.0)
                if ng < best.get(m, math.inf):
                    best[m] = ng
                    came[m] = e
                    heapq.heappush(queue, (ng + h(m), ng, m))
        return None

    def _matrix(self, penalty: np.ndarray | None, transpose: bool):
        """Sparse matrix of edge costs (+ penalty of the destination). The structure is built
        once per direction; later calls only refill the weights, without sorting."""
        from scipy.sparse import csr_matrix

        n = len(self.cells)
        cache = self.__dict__.setdefault("_csr", {})
        if transpose not in cache:
            rows, cols = (self.dst, self.src) if transpose else (self.src, self.dst)
            order = csr_matrix((np.arange(1, self.edges + 1), (rows, cols)), shape=(n, n))
            cache[transpose] = (order.data - 1, order.indices, order.indptr)
        perm, indices, indptr = cache[transpose]
        weight = self.cost[perm]
        if penalty is not None:
            weight = weight + penalty[self.dst[perm]]
        return csr_matrix((weight, indices, indptr), shape=(n, n))

    def _edges(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Index of the edge a -> b for each pair (the pairs must be edges)."""
        return np.searchsorted(self._keys, a.astype(np.int64) * len(self.cells) + b)

    def flow_field(
        self, goal: int, penalty: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Seconds to the goal from every node and the edge to take (-1: goal or unreachable).

        Dijkstra from the goal over reversed edges, in C (scipy.sparse.csgraph): recomputing it
        when enemies move is cheap, and each frame the next move is `nxt[node]`."""
        try:
            from scipy.sparse.csgraph import dijkstra
        except ImportError:
            return self._flow_field_python(goal, penalty)
        dist, pred = dijkstra(
            self._matrix(penalty, transpose=True),
            directed=True,
            indices=goal,
            return_predecessors=True,
        )
        nxt = np.full(len(self.cells), -1, dtype=np.int64)
        has = pred >= 0
        nodes = np.flatnonzero(has)
        nxt[nodes] = self._edges(nodes, pred[nodes])
        return dist, nxt

    def _flow_field_python(self, goal, penalty):
        n = len(self.cells)
        order = np.argsort(self.dst, kind="stable")  # incoming edges grouped by destination
        into = np.searchsorted(self.dst[order], np.arange(n + 1))
        dist = np.full(n, np.inf)
        nxt = np.full(n, -1, dtype=np.int64)
        dist[goal] = 0.0
        queue = [(0.0, goal)]
        while queue:
            d, m = heapq.heappop(queue)
            if d > dist[m]:
                continue
            enter = penalty[m] if penalty is not None else 0.0
            for k in range(into[m], into[m + 1]):
                e = order[k]
                s = int(self.src[e])
                nd = d + self.cost[e] + enter
                if nd < dist[s]:
                    dist[s] = nd
                    nxt[s] = e
                    heapq.heappush(queue, (nd, s))
        return dist, nxt

    def path(self, start: int, goal: int, penalty: np.ndarray | None = None) -> list[int] | None:
        """Cheapest path (edge indices), Dijkstra from the start in C; falls back to `astar`."""
        try:
            from scipy.sparse.csgraph import dijkstra
        except ImportError:
            return self.astar(start, goal, penalty)
        dist, pred = dijkstra(
            self._matrix(penalty, transpose=False),
            directed=True,
            indices=start,
            return_predecessors=True,
        )
        if not np.isfinite(dist[goal]):
            return None
        nodes = [goal]
        while nodes[-1] != start:
            nodes.append(int(pred[nodes[-1]]))
        nodes = np.array(nodes[::-1])
        return self._edges(nodes[:-1], nodes[1:]).tolist()

    def follow(
        self, dist: np.ndarray, nxt: np.ndarray, start: int, limit: int | None = None
    ) -> list[int] | None:
        """Edges from `start` to the flow field's goal (the first `limit` only, if given);
        None if the goal is unreachable."""
        if not np.isfinite(dist[start]):
            return None
        path, n = [], start
        while nxt[n] >= 0 and (limit is None or len(path) < limit):
            path.append(int(nxt[n]))
            n = int(self.dst[nxt[n]])
        return path

    def path_cost(self, path: list[int], penalty: np.ndarray | None = None) -> float:
        extra = sum(penalty[self.dst[e]] for e in path) if penalty is not None else 0.0
        return float(sum(self.cost[e] for e in path) + extra)

    def actions(self, path: list[int]) -> list[dict]:
        """Edges as moves for an input layer; runs of walking or climbing are merged."""
        result: list[dict] = []
        for e in path:
            kind = KIND[int(self.kind[e])]
            (r0, c0), (r1, c1) = self.cells[self.src[e]], self.cells[self.dst[e]]
            if kind == "walk" or kind == "climb":
                direction = (
                    ("right" if c1 > c0 else "left")
                    if kind == "walk"
                    else ("up" if r1 < r0 else "down")
                )
                if result and result[-1]["action"] == kind and result[-1]["dir"] == direction:
                    result[-1]["tiles"] += 1
                    result[-1]["to"] = [int(r1), int(c1)]
                    continue
                result.append(
                    {"action": kind, "dir": direction, "tiles": 1, "to": [int(r1), int(c1)]}
                )
                continue
            step = {"action": kind, "to": [int(r1), int(c1)]}
            if kind == "jump":
                step["dx"] = int(self.dx[e])
                step["apex"] = round(float(self.apex[e]), 2)
                step["double_jump"] = bool(self.apex[e] > self.physics.jump_height)
            elif kind == "fall":
                step["dir"] = "right" if self.dx[e] > 0 else "left"
            result.append(step)
        return result
