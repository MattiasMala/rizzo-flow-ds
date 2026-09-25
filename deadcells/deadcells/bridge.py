"""Game state straight from the game's memory, through a shared-memory "bridge".

The fastest and most exact source: a mod running inside the game (Dead Cells runs on HashLink;
the MIT-licensed Dead Cells Core Modding API loads C# mods on Windows x64) copies the hero, the
entities and the level's collision grid into a named shared-memory block every frame. Python
maps the same block and reads a consistent snapshot in microseconds: no screenshots, no vision
model, exact tile coordinates. The mod side is NOT written yet (it needs the game to find the
fields); this module fixes the binary layout both sides agree on, and `BridgeWriter` is the
reference writer used by the tests.

Consistency uses a sequence lock: the writer makes `seq` odd, writes, makes it even again; the
reader copies the block and retries if `seq` was odd or changed meanwhile. The level grid is
copied only when `level_version` changes.

Layout v1, little-endian, offsets in bytes:

    header (64)   magic "RZDC", version u16, header_size u16, seq u32, frame u32,
                  game_time f64, level_version u32, level_w u16, level_h u16,
                  n_entities u16, entity_size u16, grid_offset u32, entities_offset u32,
                  capacity u32, reserved
    hero (64)     at 64: x f32, y f32 (tiles: column, row of the feet, fractional),
                  vx f32, vy f32 (tiles/s, y down), hp i32, hp_max i32, flask i32,
                  flask_max i32, cells i32, gold i32, flags u32, brutality u8, tactics u8,
                  survival u8, pad u8, curse i32, reserved
    entities      at entities_offset, entity_size (32) each: id u32, kind u16, flags u16,
                  x f32, y f32, vx f32, vy f32, hp i32, hp_max i32
    grid          at grid_offset: level_h × level_w u8 tile codes (nav.EMPTY, SOLID, ...)
"""

import mmap
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC, VERSION = b"RZDC", 1
HEADER = struct.Struct("<4sHHIIdIHHHHIII16x")
HERO = struct.Struct("<4f6iI4Bi12x")
ENTITY = struct.Struct("<IHH4f2i")
HERO_OFFSET = 64
# Windows: named mapping. Linux: a file in shared memory (a mod under Proton can map it as
# Z:\dev\shm\RizzoDeadCells).
DEFAULT_NAME = "Local\\RizzoDeadCells" if sys.platform == "win32" else "/dev/shm/RizzoDeadCells"

KINDS = (
    "unknown",
    "enemy",
    "projectile",
    "trap",
    "weapon_drop",
    "scroll",
    "food",
    "gold",
    "cell",
    "chest",
    "door",
    "exit",
    "teleporter",
    "npc",
)
ENTITY_FLAGS = ("elite", "boss", "hostile", "attacking")
HERO_FLAGS = ("on_ground", "facing_right", "on_ladder", "invulnerable")
ENTITY_DTYPE = np.dtype(
    [
        ("id", "<u4"),
        ("kind", "<u2"),
        ("flags", "<u2"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("vx", "<f4"),
        ("vy", "<f4"),
        ("hp", "<i4"),
        ("hp_max", "<i4"),
    ]
)
assert HEADER.size == 64 and HERO.size == 64 and ENTITY.size == ENTITY_DTYPE.itemsize == 32


@dataclass
class Hero:
    x: float
    y: float
    vx: float
    vy: float
    hp: int
    hp_max: int
    flask: int
    flask_max: int
    cells: int
    gold: int
    flags: int
    brutality: int
    tactics: int
    survival: int
    curse: int

    def flag(self, name: str) -> bool:
        return bool(self.flags >> HERO_FLAGS.index(name) & 1)


@dataclass
class Snapshot:
    frame: int
    game_time: float
    level_version: int
    hero: Hero
    entities: np.ndarray  # ENTITY_DTYPE records
    grid: np.ndarray | None  # (h, w) tile codes; the same object while level_version holds


def layout_size(width: int, height: int, capacity: int) -> tuple[int, int, int]:
    entities_offset = HERO_OFFSET + HERO.size
    grid_offset = entities_offset + capacity * ENTITY.size
    return grid_offset + width * height, entities_offset, grid_offset


def _open(target: str | Path, size: int, create: bool) -> mmap.mmap:
    path = Path(str(target))
    if sys.platform == "win32" and not path.suffix:
        return mmap.mmap(-1, size, tagname=str(target))  # named mapping shared with the mod
    if create:
        with open(path, "wb") as f:
            f.truncate(size)
    with open(path, "r+b") as f:
        return mmap.mmap(f.fileno(), size or 0)


class BridgeWriter:
    """Reference writer (tests, and the layout a game mod must reproduce)."""

    def __init__(self, target, width: int, height: int, capacity: int = 256):
        size, self.entities_offset, self.grid_offset = layout_size(width, height, capacity)
        self.buf = _open(target, size, create=True)
        self.width, self.height, self.capacity = width, height, capacity
        self.seq = self.frame = self.level_version = 0

    def _header(self, n: int, game_time: float) -> bytes:
        return HEADER.pack(
            MAGIC,
            VERSION,
            HEADER.size,
            self.seq,
            self.frame,
            game_time,
            self.level_version,
            self.width,
            self.height,
            n,
            ENTITY.size,
            self.grid_offset,
            self.entities_offset,
            self.capacity,
        )

    def write(
        self, hero: Hero, entities: np.ndarray, game_time: float, grid: np.ndarray | None = None
    ) -> None:
        if len(entities) > self.capacity:
            raise ValueError(f"{len(entities)} entities > capacity {self.capacity}")
        self.seq += 1  # odd: writing
        struct.pack_into("<I", self.buf, 8, self.seq)
        self.frame += 1
        if grid is not None:
            if grid.shape != (self.height, self.width):
                raise ValueError(f"grid {grid.shape} != {(self.height, self.width)}")
            self.level_version += 1
            self.buf[self.grid_offset : self.grid_offset + grid.size] = grid.astype(
                np.uint8
            ).tobytes()
        self.buf[HERO_OFFSET : HERO_OFFSET + HERO.size] = HERO.pack(
            hero.x,
            hero.y,
            hero.vx,
            hero.vy,
            hero.hp,
            hero.hp_max,
            hero.flask,
            hero.flask_max,
            hero.cells,
            hero.gold,
            hero.flags,
            hero.brutality,
            hero.tactics,
            hero.survival,
            0,
            hero.curse,
        )
        data = np.asarray(entities, dtype=ENTITY_DTYPE).tobytes()
        self.buf[self.entities_offset : self.entities_offset + len(data)] = data
        self.buf[: HEADER.size] = self._header(len(entities), game_time)  # seq still odd
        self.seq += 1  # even: consistent, written last
        struct.pack_into("<I", self.buf, 8, self.seq)

    def close(self) -> None:
        self.buf.close()


@dataclass
class BridgeReader:
    target: str | Path = DEFAULT_NAME
    retries: int = 100
    torn: int = 0  # reads retried because the writer was busy
    _buf: mmap.mmap | None = None
    _grid: np.ndarray | None = None
    _grid_version: int = -1
    _last_frame: int = -1
    stale: int = field(default=0)  # reads that found no new frame

    def _map(self) -> mmap.mmap:
        if self._buf is None:
            probe = _open(self.target, HEADER.size, create=False)
            head = HEADER.unpack_from(probe, 0)
            probe.close()
            if head[0] == b"\0\0\0\0":
                raise ConnectionError(f"Bridge {self.target} is empty: is the game mod running?")
            if head[0] != MAGIC or head[1] != VERSION:
                raise ValueError(f"Not a v{VERSION} bridge: magic {head[0]!r}, version {head[1]}")
            size = layout_size(head[7], head[8], head[13])[0]
            self._buf = _open(self.target, size, create=False)
        return self._buf

    def read(self) -> Snapshot | None:
        """Latest consistent snapshot; None when the writer has not produced a new frame."""
        buf = self._map()
        for _ in range(self.retries):
            seq = struct.unpack_from("<I", buf, 8)[0]
            if seq & 1:
                self.torn += 1
                continue
            head = HEADER.unpack_from(buf, 0)
            hero_raw = HERO.unpack_from(buf, HERO_OFFSET)
            n, entities_offset, grid_offset = head[9], head[12], head[11]
            entities = np.frombuffer(buf, ENTITY_DTYPE, n, entities_offset).copy()
            level_version, w, h = head[6], head[7], head[8]
            grid = self._grid
            if level_version != self._grid_version:
                grid = np.frombuffer(buf, np.uint8, w * h, grid_offset).reshape(h, w).copy()
            if struct.unpack_from("<I", buf, 8)[0] != seq:
                self.torn += 1
                continue
            if head[4] == self._last_frame:
                self.stale += 1
                return None
            self._last_frame = head[4]
            self._grid, self._grid_version = grid, level_version
            hero = Hero(*hero_raw[:14], hero_raw[15])
            return Snapshot(head[4], head[5], level_version, hero, entities, grid)
        raise TimeoutError(f"Writer busy for {self.retries} attempts")

    def close(self) -> None:
        if self._buf is not None:
            self._buf.close()
