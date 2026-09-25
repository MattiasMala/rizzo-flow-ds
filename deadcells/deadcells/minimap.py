"""Minimap (or full map) parsing: colour classes -> cell grid -> player, rooms, targets, routes.

The map crop is reduced to a grid of cells (a few screen pixels each). Every cell takes the first
colour class, in layout order, that covers at least `min_cover` of it, so small markers (player,
doors) listed first win over the floor. Walkable cells are the floor plus every marker. Routes are
breadth-first shortest paths over walkable cells from the player, compressed into runs such as
`right 12, up 3`: they follow the drawn map, not the game's real collision.
"""

from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np

from .layout import ColorClass

EMPTY = -1
STEPS = {(1, 0): "right", (-1, 0): "left", (0, -1): "up", (0, 1): "down"}
CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def classify(crop: np.ndarray, classes: list[ColorClass], cell: int) -> np.ndarray:
    """Grid of class indices (EMPTY where no class covers enough of the cell)."""
    rows, cols = crop.shape[0] // cell, crop.shape[1] // cell
    if rows == 0 or cols == 0:
        return np.full((0, 0), EMPTY, dtype=np.int8)
    image = np.ascontiguousarray(crop[: rows * cell, : cols * cell])
    grid = np.full((rows, cols), EMPTY, dtype=np.int8)
    for index in reversed(range(len(classes))):  # first class in the list is applied last: wins
        # INTER_AREA on an exact multiple is the block mean: the share of the cell covered.
        cover = cv2.resize(classes[index].mask(image), (cols, rows), interpolation=cv2.INTER_AREA)
        grid[cover >= classes[index].min_cover * 255] = index
    return grid


@dataclass(frozen=True)
class Target:
    name: str
    cell: tuple[int, int]  # x, y in grid cells
    size: int  # cells covered
    path_length: int | None  # None: not reachable over drawn floor
    route: list[tuple[str, int]]


@dataclass(frozen=True)
class MapState:
    grid_size: tuple[int, int]  # cols, rows
    player: tuple[int, int] | None
    rooms: int  # separate walkable regions
    explored_cells: int
    targets: list[Target]


def compress(path: list[tuple[int, int]]) -> list[tuple[str, int]]:
    runs: list[tuple[str, int]] = []
    for (x0, y0), (x1, y1) in pairwise(path):
        step = STEPS[(x1 - x0, y1 - y0)]
        if runs and runs[-1][0] == step:
            runs[-1] = (step, runs[-1][1] + 1)
        else:
            runs.append((step, 1))
    return runs


def distances(walkable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Steps from `start` (x, y) to every walkable cell over 4-neighbours; -1 = unreachable.

    Breadth-first search as whole-array frontier expansion (one dilation per ring) instead of
    a Python loop per cell.
    """
    dist = np.full(walkable.shape, -1, dtype=np.int32)
    free = walkable.astype(np.uint8)
    frontier = np.zeros_like(free)
    frontier[start[1], start[0]] = 1
    step = 0
    while True:
        ring = frontier.astype(bool)
        if not ring.any():
            return dist
        dist[ring] = step
        free[ring] = 0
        frontier = cv2.bitwise_and(cv2.dilate(frontier, CROSS), free)
        step += 1


def walk_back(dist: np.ndarray, end: tuple[int, int]) -> list[tuple[int, int]]:
    """Shortest path from the start to `end` (x, y), following decreasing distances."""
    rows, cols = dist.shape
    path = [end]
    x, y = end
    while dist[y, x] > 0:
        for dx, dy in STEPS:  # fixed order: deterministic among equal paths
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows and dist[ny, nx] == dist[y, x] - 1:
                x, y = nx, ny
                break
        path.append((x, y))
    return path[::-1]


def parse(grid: np.ndarray, classes: list[ColorClass], max_route_runs: int = 6) -> MapState:
    rows, cols = grid.shape
    walkable = grid != EMPTY
    player = None
    player_ids = [i for i, c in enumerate(classes) if c.kind == "player"]
    if player_ids:
        ys, xs = np.nonzero(np.isin(grid, player_ids))
        if xs.size:
            # The marker cell nearest to the marker's centroid (the centroid may fall off-floor).
            cx, cy = xs.mean(), ys.mean()
            best = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
            player = (int(xs[best]), int(ys[best]))
    rooms = cv2.connectedComponents(walkable.astype(np.uint8), connectivity=4)[0] - 1
    dist = distances(walkable, player) if player else np.full(grid.shape, -1, np.int32)

    targets = []
    for index, color in enumerate(classes):
        if color.kind != "poi":
            continue
        count, labels = cv2.connectedComponents((grid == index).astype(np.uint8), connectivity=8)
        for label in range(1, count):
            ys, xs = np.nonzero(labels == label)
            cells = list(zip(xs.tolist(), ys.tolist()))
            reachable = dist[ys, xs] >= 0
            if reachable.any():
                i = int(np.argmin(np.where(reachable, dist[ys, xs], np.iinfo(np.int32).max)))
                path = walk_back(dist, cells[i])
                length, route = len(path) - 1, compress(path)[:max_route_runs]
            else:
                length, route = None, []
            centre = (round(float(xs.mean())), round(float(ys.mean())))
            targets.append(Target(color.name, centre, len(cells), length, route))
    targets.sort(key=lambda t: (t.path_length is None, t.path_length or 0, t.name))
    return MapState((cols, rows), player, int(rooms), int(walkable.sum()), targets)


def read_map(crop: np.ndarray, classes: list[ColorClass], cell: int) -> MapState:
    return parse(classify(crop, classes, cell), classes)
