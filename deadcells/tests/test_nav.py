import heapq
import math

import pytest

np = pytest.importorskip("numpy")

from deadcells.nav import (
    FALL,
    HAZARD,
    JUMP,
    LADDER,
    NavGraph,
    Physics,
    arc,
    parse_ascii,
)

try:
    import scipy  # noqa: F401

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

P = Physics()  # single jump apex 3 tiles, double 5.5, body 2 tiles
SKY = "\n".join(["." * 30] * 8) + "\n"


def graph(text, physics=P, links=()):
    # Open sky above every test level: the top of the grid is a ceiling.
    grid, marks = parse_ascii(SKY + text.strip("\n"))
    g = NavGraph(grid, physics, links=list(links))
    return g, {k: [g.node(*rc) for rc in v] for k, v in marks.items()}, marks


def route(text, physics=P):
    g, m, _ = graph(text, physics)
    return g, g.astar(m["S"][0], m["G"][0])


def route_marks(text, physics=P):
    g, m, marks = graph(text, physics)
    return g, g.astar(m["S"][0], m["G"][0]), marks


def kinds(g, path):
    return [a["action"] for a in g.actions(path)]


def test_flat_walk():
    g, path = route("""
........
.S....G.
########
""")
    (step,) = g.actions(path)
    assert (step["action"], step["dir"], step["tiles"]) == ("walk", "right", 5)
    assert g.path_cost(path) == pytest.approx(5 / P.run_speed)


def test_gap_needs_a_jump_and_too_wide_gap_is_unreachable():
    g, path = route("""
..............
.S.......G....
####....######
""")
    assert "jump" in kinds(g, path)
    _, path = route("""
.........................
.S....................G..
####..................###
""")
    assert path is None


def test_wall_needs_double_jump():
    text = """
..........
......G...
......####
......####
......####
.S....####
##########
"""
    g, path = route(text)
    jumps = [a for a in g.actions(path) if a["action"] == "jump"]
    assert jumps and jumps[-1]["double_jump"]
    single_only = Physics(double_jump_height=3.0)
    assert route(text, single_only)[1] is None


def test_ceiling_blocks_the_jump():
    open_ = """
..........
..........
.......G..
.......###
.S.....###
##########
"""
    capped = """
..........
#######...
.......G..
.......###
.S.....###
##########
"""
    assert route(open_)[1] is not None
    assert route(capped)[1] is None


def test_one_way_platform_jump_through_and_drop():
    g, path = route("""
..........
....G.....
...====...
..........
..........
....S.....
##########
""")
    assert path is not None and kinds(g, path)[-1] == "jump"  # up through the platform
    g, path = route("""
..........
....S.....
..=====...
..........
..........
....G.....
##########
""")
    assert kinds(g, path) == ["drop"]


def test_ladder_climb_and_exit_at_the_top():
    g, path, marks = route_marks("""
.........
.....G...
...H####.
...H####.
...H####.
...H####.
...H####.
.S.H####.
#########
""")
    acts = g.actions(path)
    assert any(a["action"] == "climb" and a["dir"] == "up" for a in acts)
    assert acts[-1]["to"] == list(marks["G"][0])


def test_long_fall_off_a_ledge():
    g, path, marks = route_marks("""
.S.......
####.....
####.....
####.....
####.....
####...G.
#########
""")
    assert path is not None
    edge_kinds = {int(g.kind[e]) for e in path}
    assert edge_kinds & {FALL, JUMP}
    assert g.cells[g.dst[path[-1]]].tolist() == list(marks["G"][0])


def test_teleport_link_and_heuristic_off():
    text = """
.S.#....G.
####....##
####......
##########
"""
    grid, marks = parse_ascii(SKY + text.strip("\n"))
    without = NavGraph(grid, P)
    s, gl = without.node(*marks["S"][0]), without.node(*marks["G"][0])
    base = without.path_cost(without.astar(s, gl))
    linked = NavGraph(grid, P, links=[(marks["S"][0], marks["G"][0], 0.01)])
    path = linked.astar(s, gl)
    assert kinds(linked, path) == ["link"] and linked.path_cost(path) < base
    with pytest.raises(ValueError):
        NavGraph(grid, P, links=[((8, 3), (8, 8), 0.1)])  # (8, 3) is a wall


def test_penalty_diverts_the_route():
    text = """
.............
.............
....====.....
.............
.S.........G.
#############
"""
    g, m, marks = graph(text)
    s, goal = m["S"][0], m["G"][0]
    straight = g.astar(s, goal)
    assert kinds(g, straight) == ["walk"]
    penalty = np.zeros(len(g.cells))
    r, _ = marks["S"][0]
    for col in range(5, 8):  # an enemy standing in the middle of the floor
        penalty[g.node(r, col)] = 10.0
    detour = g.astar(s, goal, penalty)
    assert g.path_cost(detour, penalty) < g.path_cost(straight, penalty)


def test_hazards_are_avoided_when_possible():
    grid, marks = parse_ascii(
        SKY
        + """
...........
.S.......G.
###########
""".strip("\n")
    )
    grid[1, 4:7] = HAZARD
    g = NavGraph(grid, P)
    path = g.astar(g.node(*marks["S"][0]), g.node(*marks["G"][0]))
    visited = {tuple(g.cells[g.dst[e]]) for e in path}
    assert not visited & {(1, 4), (1, 5), (1, 6)}


def test_node_of_a_cell_in_the_air_is_the_landing():
    g, m, marks = graph("""
.....
..A..
.....
#####
""")
    r, c = marks["A"][0]
    assert tuple(g.cells[m["A"][0]]) == (r + 1, c)
    assert g.node(1, 0) == g.node(r + 1, 0)  # falling from high up in the sky


def test_arc_ends_at_takeoff_height():
    cells, t = arc(P, 3.0, 4)
    centre = [c for c in cells if c[3]]
    assert centre[-1] == (4, 0, True, True)
    assert {ox for ox, _, _, _ in cells} == {0, 1, 2, 3, 4}  # sides stay within one column
    assert max(oy for _, oy, _, _ in cells) == 3
    assert t == pytest.approx(2 * math.sqrt(2 * P.gravity * 3) / P.gravity)


def random_level(seed, rows=40, cols=80):
    rng = np.random.default_rng(seed)
    grid = np.zeros((rows, cols), np.uint8)
    grid[-1] = 1
    for _ in range(60):
        r, c, w = rng.integers(2, rows - 1), rng.integers(0, cols - 4), rng.integers(2, 10)
        grid[r, c : c + w] = rng.choice([1, 1, 2])
    for _ in range(4):
        c, r = rng.integers(0, cols), rng.integers(5, rows - 6)
        grid[r : r + 5, c] = LADDER
    return grid


def dijkstra(g, start, goal, penalty):
    dist = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        d, n = heapq.heappop(queue)
        if n == goal:
            return d
        if d > dist[n]:
            continue
        for e in range(g.offsets[n], g.offsets[n + 1]):
            m = int(g.dst[e])
            nd = d + g.cost[e] + penalty[m]
            if nd < dist.get(m, math.inf):
                dist[m] = nd
                heapq.heappush(queue, (nd, m))
    return None


@pytest.mark.parametrize("seed", range(5))
def test_all_searches_are_optimal(seed):
    g = NavGraph(random_level(seed), P)
    rng = np.random.default_rng(100 + seed)
    goal = int(rng.integers(len(g.cells)))
    penalty = rng.uniform(0, 0.5, len(g.cells)) * (rng.random(len(g.cells)) < 0.2)
    fields = [g._flow_field_python(goal, penalty)]
    if HAVE_SCIPY:
        fields.append(g.flow_field(goal, penalty))
    for start in rng.integers(len(g.cells), size=20):
        start = int(start)
        best = dijkstra(g, start, goal, penalty)
        searches = [g.astar(start, goal, penalty), g.path(start, goal, penalty)]
        if best is None:
            assert all(p is None for p in searches)
            for dist, nxt in fields:
                assert not np.isfinite(dist[start]) and g.follow(dist, nxt, start) is None
            continue
        for path in searches:
            assert g.path_cost(path, penalty) == pytest.approx(best)
            assert int(g.dst[path[-1]]) == goal if path else start == goal
        for dist, nxt in fields:
            assert dist[start] == pytest.approx(best)
            assert g.path_cost(g.follow(dist, nxt, start), penalty) == pytest.approx(best)
