"""Read HashLink objects from another process's memory: classes, fields and live instances.

Dead Cells runs on the HashLink VM, which keeps full run-time type information in memory. So
no fragile pointer chains are needed: starting from a class name (as listed by `probe`'s
types.txt) this module finds the VM's own descriptors and reads objects like a debugger would:

1. the class name as a UTF-16 string (HashLink strings are 16-bit);
2. the `hl_type_obj` whose `name` points to it (name at offset 16);
3. the `hl_type` (kind HOBJ = 11) whose `obj` points to that descriptor: every object of the
   class starts with a pointer to this `hl_type`;
4. the class's `hl_runtime_obj`: `fields_indexes` gives the byte offset of every field,
   inherited ones included, exactly as the VM laid them out;
5. instances: heap words equal to the `hl_type` address (verified by reading them).

Offsets are those of HashLink's `hl.h` on 64-bit (identical on Windows and Linux: only ints and
pointers are involved). Checked on a real HashLink 1.14 process in the tests; the game's
own libhl may differ, which the sanity checks below would reveal.
"""

import struct
from dataclasses import dataclass

import numpy as np

HOBJ = 11
KINDS = (
    "void",
    "u8",
    "u16",
    "i32",
    "i64",
    "f32",
    "f64",
    "bool",
    "bytes",
    "dyn",
    "fun",
    "obj",
    "array",
    "type",
    "ref",
    "virtual",
    "dynobj",
    "abstract",
    "enum",
    "null",
    "method",
    "struct",
    "packed",
)
SCALARS = {1: "<B", 2: "<H", 3: "<i", 4: "<q", 5: "<f", 6: "<d", 7: "<?"}
CHUNK = 32 << 20


@dataclass(frozen=True)
class Field:
    name: str
    kind: str
    type_name: str | None  # class name for obj fields
    type_addr: int
    offset: int
    owner: str  # class that declares it


@dataclass(frozen=True)
class ClassInfo:
    name: str
    type_addr: int  # hl_type*: the first word of every instance
    obj_addr: int  # hl_type_obj*
    parent: int  # hl_type* of the superclass, 0 if none
    size: int | None  # instance size in bytes (None: runtime info not built yet)
    fields: tuple[Field, ...]


class HashLink:
    def __init__(self, memory, writable_only: bool = True):
        self.mem = memory
        self.writable_only = writable_only
        self._classes: dict[int, ClassInfo] = {}

    # -- raw reads -------------------------------------------------------------------------
    def u32(self, a: int) -> int:
        return struct.unpack("<I", self.mem.read(a, 4))[0]

    def i32(self, a: int) -> int:
        return struct.unpack("<i", self.mem.read(a, 4))[0]

    def ptr(self, a: int) -> int:
        return struct.unpack("<Q", self.mem.read(a, 8))[0]

    def ustring(self, a: int, limit: int = 256) -> str | None:
        """Zero-terminated UTF-16 string, None if unreadable or not text."""
        try:
            raw = self.mem.read(a, limit * 2)
        except OSError:
            return None
        end = next((i for i in range(0, len(raw), 2) if raw[i : i + 2] == b"\0\0"), None)
        if end is None:
            return None
        try:
            text = raw[:end].decode("utf-16-le")
        except UnicodeDecodeError:
            return None
        return text if text.isprintable() else None

    # -- scanning --------------------------------------------------------------------------
    def _regions(self):
        regions = self.mem.regions(writable=True if self.writable_only else None)
        anon = [r for r in regions if r[3] in ("", "[heap]") or r[3].startswith("[anon")]
        return anon or regions

    def _chunks(self, overlap: int = 0):
        for start, end, _, _ in self._regions():
            at = start
            while at < end:
                size = min(CHUNK, end - at)
                try:
                    data = self.mem.read(at, min(size + overlap, end - at))
                except OSError:
                    break
                yield at, data
                at += size

    def find_bytes(self, needle: bytes, limit: int = 64, align: int = 1) -> list[int]:
        found = []
        for base, data in self._chunks(overlap=len(needle)):
            i = data.find(needle)
            while i != -1 and len(found) < limit:
                if (base + i) % align == 0:
                    found.append(base + i)
                i = data.find(needle, i + 1)
            if len(found) >= limit:
                break
        return sorted(set(found))

    def find_pointers(self, values: list[int], limit: int = 100_000) -> dict[int, list[int]]:
        """Addresses (8-byte aligned) holding any of `values`."""
        wanted = np.array(sorted(set(values)), dtype=np.uint64)
        found: dict[int, list[int]] = {int(v): [] for v in wanted}
        total = 0
        for base, data in self._chunks():
            words = np.frombuffer(data[: len(data) // 8 * 8], dtype=np.uint64)
            hits = np.flatnonzero(np.isin(words, wanted))
            for h in hits:
                found[int(words[h])].append(base + 8 * int(h))
                total += 1
            if total >= limit:
                break
        return found

    # -- classes ---------------------------------------------------------------------------
    def find_class(self, name: str) -> ClassInfo:
        """Locate a class by its full name (e.g. "en.Hero") in the running VM."""
        strings = self.find_bytes(name.encode("utf-16-le") + b"\0\0", align=2)
        if not strings:
            raise LookupError(f"Class name {name!r} not found in memory")
        refs = self.find_pointers(strings)
        candidates = []
        for places in refs.values():
            for p in places:
                obj = p - 16  # hl_type_obj.name
                nfields, nproto = self.i32(obj), self.i32(obj + 4)
                if 0 <= nfields < 4096 and 0 <= nproto < 4096:
                    candidates.append(obj)
        if not candidates:
            raise LookupError(f"No class descriptor points to {name!r}")
        types = self.find_pointers(candidates)
        for obj, places in types.items():
            for p in places:
                t = p - 8  # hl_type.obj
                if self.u32(t) == HOBJ:
                    return self.class_at(t)
        raise LookupError(f"No hl_type of kind HOBJ for {name!r}")

    def class_at(self, t: int) -> ClassInfo:
        if t in self._classes:
            return self._classes[t]
        if self.u32(t) != HOBJ:
            raise ValueError(f"{t:#x} is not an object type")
        obj = self.ptr(t + 8)
        name = self.ustring(self.ptr(obj + 16)) or f"<unnamed {t:#x}>"
        parent = self.ptr(obj + 24)
        rt = self.ptr(obj + 72)
        chain = []  # (type, obj) from the root class down to this one
        cur_t, cur_obj = t, obj
        while True:
            chain.append((cur_t, cur_obj))
            sup = self.ptr(cur_obj + 24)
            if not sup:
                break
            cur_t, cur_obj = sup, self.ptr(sup + 8)
        chain.reverse()
        fields: list[Field] = []
        size = None
        indexes = None
        if rt:
            nfields, size = self.i32(rt + 8), self.i32(rt + 16)
            indexes = (
                struct.unpack(f"<{nfields}i", self.mem.read(self.ptr(rt + 40), 4 * nfields))
                if nfields
                else ()
            )
        position = 0
        for _, o in chain:
            owner = self.ustring(self.ptr(o + 16)) or "?"
            n, array = self.i32(o), self.ptr(o + 32)
            for i in range(n):
                f = array + 24 * i  # hl_obj_field: name, type, hashed_name
                ftype = self.ptr(f + 8)
                kind_code = self.u32(ftype)
                kind = KINDS[kind_code] if kind_code < len(KINDS) else f"kind{kind_code}"
                type_name = None
                if kind_code == HOBJ:
                    type_name = self.ustring(self.ptr(self.ptr(ftype + 8) + 16))
                offset = indexes[position] if indexes is not None else -1
                fields.append(
                    Field(
                        self.ustring(self.ptr(f)) or f"field{i}",
                        kind,
                        type_name,
                        ftype,
                        offset,
                        owner,
                    )
                )
                position += 1
        info = ClassInfo(name, t, obj, parent, size, tuple(fields))
        self._classes[t] = info
        return info

    # -- instances -------------------------------------------------------------------------
    def is_a(self, t: int, ancestor: int) -> bool:
        """Whether object type `t` is `ancestor` or a subclass of it."""
        for _ in range(64):
            if t == ancestor:
                return True
            if not t or self.u32(t) != HOBJ:
                return False
            t = self.ptr(self.ptr(t + 8) + 24)
        return False

    def check(self, obj: int, cls: ClassInfo) -> tuple[int, int, int]:
        """(pointer fields checked, consistent, consistent and non-null). Words equal to the
        type address that are not objects (descriptors, stale memory) fail the check."""
        checked = good = linked = 0
        for f in cls.fields:
            if f.kind != "obj" or f.offset < 0:
                continue
            checked += 1
            try:
                value = self.ptr(obj + f.offset)
                if value == 0:
                    good += 1
                elif self.is_a(self.ptr(value), f.type_addr):
                    good += 1
                    linked += 1
            except (OSError, struct.error):
                pass
        return checked, good, linked

    def instances(self, cls: ClassInfo, limit: int = 1000) -> list[int]:
        """Live objects of exactly this class (subclasses have their own hl_type): candidates
        whose pointer fields are all consistent, most linked first (an object with every
        reference null may still be a false positive)."""
        found = self.find_pointers([cls.type_addr], limit=limit * 20)[cls.type_addr]
        scored = []
        for a in found:
            checked, good, linked = self.check(a, cls)
            if good == checked:
                scored.append((-linked, a))
        return [a for _, a in sorted(scored)][:limit]

    def statics(self, cls: ClassInfo) -> int | None:
        """The class object holding the static variables (e.g. `Game.ME`), through the
        descriptor's `global_value`: no memory scan needed. None if the class has none."""
        slot = self.ptr(cls.obj_addr + 56)
        value = self.ptr(slot) if slot else 0
        return value or None

    def array_items(self, array_obj: int, limit: int = 4096) -> list[int]:
        """Elements (pointers) of an `hl.types.ArrayObj`: {type, length, array: varray*};
        the varray header is 24 bytes (type, element type, size, padding)."""
        cls = self.class_at(self.ptr(array_obj))
        fields = {f.name: f for f in cls.fields}
        length = self.read_field(array_obj, fields["length"])
        native = self.ptr(array_obj + fields["array"].offset)
        count = max(0, min(length, limit))
        return (
            list(struct.unpack(f"<{count}Q", self.mem.read(native + 24, 8 * count)))
            if count
            else []
        )

    def read_field(self, obj: int, field: Field):
        if field.offset < 0:
            raise ValueError(f"No runtime layout for {field.owner}.{field.name}")
        code = KINDS.index(field.kind) if field.kind in KINDS else -1
        if code in SCALARS:
            fmt = SCALARS[code]
            return struct.unpack(fmt, self.mem.read(obj + field.offset, struct.calcsize(fmt)))[0]
        value = self.ptr(obj + field.offset)
        if field.type_name == "String" and value:
            return self.string_object(value)
        return value  # pointer: object, array, closure...

    def string_object(self, s: int) -> str | None:
        """Haxe String object: {hl_type*, bytes: uchar*, length: int}."""
        try:
            data, length = self.ptr(s + 8), self.i32(s + 16)
            if not 0 <= length < 100_000:
                return None
            return self.mem.read(data, 2 * length).decode("utf-16-le")
        except (OSError, UnicodeDecodeError):
            return None

    def dump(self, obj: int) -> dict:
        """Every field of a live object; object fields are shown as "Class@address"."""
        cls = self.class_at(self.ptr(obj))
        out = {}
        for f in cls.fields:
            try:
                value = self.read_field(obj, f)
            except (OSError, ValueError) as exc:
                value = f"<{exc}>"
            if f.kind == "bool" or isinstance(value, (float, str)) or value is None:
                pass
            elif f.kind == "obj" and value:
                value = f"{f.type_name}@{value:#x}"
            elif f.kind not in ("i32", "i64", "u8", "u16"):
                value = f"{f.kind}@{value:#x}" if value else None
            out[f.name] = value
        return out

    def describe_class(self, cls: ClassInfo) -> dict:
        return {
            "name": cls.name,
            "type": f"{cls.type_addr:#x}",
            "size": cls.size,
            "parent": self.class_at(cls.parent).name if cls.parent else None,
            "fields": [
                f"{f.owner}.{f.name}: {f.type_name or f.kind} @{f.offset}" for f in cls.fields
            ],
        }
