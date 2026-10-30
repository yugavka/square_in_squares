"""Compose locally validated rectangular crops along a diagonal fault line.

A gadget with ``w*h + surplus`` unit squares fits a ``(w+f) x (h+f)``
rectangle. Gadgets placed corner to corner share only a small seam square.
The remaining off-diagonal rectangles are filled with upright unit squares.
Every proposed full packing is checked by the independent SAT validator.
"""

from __future__ import annotations

import math
import heapq
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from validate_submission import (ValidationError, make_square, measure_packing,
                                 squares_overlap)


Placement = tuple[float, float, float]
Packing = tuple[float, list[Placement]]


@dataclass(slots=True)
class Gadget:
    w: int
    h: int
    surplus: int
    overhang: float
    squares: tuple[Placement, ...]


def normalize(squares: Iterable[Placement]) -> tuple[tuple[Placement, ...], float, float]:
    """Move the true bounding-box lower-left corner to the origin."""
    rows = tuple(squares)
    geometry = [make_square(i, *row) for i, row in enumerate(rows)]
    min_x = min(square.min_x for square in geometry)
    min_y = min(square.min_y for square in geometry)
    width = max(square.max_x for square in geometry) - min_x
    height = max(square.max_y for square in geometry) - min_y
    return tuple((x - min_x, y - min_y, angle) for x, y, angle in rows), width, height


def orient(squares: Iterable[Placement], turns: int = 0,
           mirror: bool = False) -> tuple[Placement, ...]:
    """Apply one of the eight square symmetries without changing unit sizes."""
    changed = []
    for x, y, angle in squares:
        if mirror:
            x, angle = -x, -angle
        for _ in range(turns):
            x, y, angle = -y, x, angle + math.pi / 2
        changed.append((x, y, math.remainder(angle, math.pi / 2)))
    return normalize(changed)[0]


def extract_corner_crops(donors: Iterable[Packing],
                         max_dim: int = 28,
                         variants_per_signature: int = 1) -> list[Gadget]:
    """Keep low-overhang crops, optionally retaining distinct seam shapes."""
    if variants_per_signature < 1:
        raise ValueError("variants_per_signature must be positive")
    best: dict[tuple[int, int, int, bool], list[Gadget]] = {}
    for _, donor in donors:
        for mirror in (False, True):
            for turns in range(4):
                rows = orient(donor, turns, mirror)
                geometry = [make_square(i, *row) for i, row in enumerate(rows)]
                side = max(max(s.max_x for s in geometry),
                           max(s.max_y for s in geometry))
                base = math.floor(side + 1e-9)
                full_f = side - base
                if base < 2 or not (0 < full_f < 1):
                    continue
                limit = min(base, max_dim)
                # A square appears in crop (w,h) iff both integer thresholds
                # are met. The 2D prefix sum counts crops before rebuilding any.
                buckets = [[0] * (limit + 1) for _ in range(limit + 1)]
                # Prefix extremes of the true bounds give each crop's width
                # and height without rebuilding its squares.
                low_x = [[math.inf] * (limit + 1) for _ in range(limit + 1)]
                low_y = [[math.inf] * (limit + 1) for _ in range(limit + 1)]
                high_x = [[-math.inf] * (limit + 1) for _ in range(limit + 1)]
                high_y = [[-math.inf] * (limit + 1) for _ in range(limit + 1)]
                required = []
                for square in geometry:
                    rw = max(1, math.ceil(square.max_x - full_f - 1e-9))
                    rh = max(1, math.ceil(square.max_y - full_f - 1e-9))
                    required.append((rw, rh))
                    if rw <= limit and rh <= limit:
                        buckets[rw][rh] += 1
                        low_x[rw][rh] = min(low_x[rw][rh], square.min_x)
                        low_y[rw][rh] = min(low_y[rw][rh], square.min_y)
                        high_x[rw][rh] = max(high_x[rw][rh], square.max_x)
                        high_y[rw][rh] = max(high_y[rw][rh], square.max_y)
                for w in range(1, limit + 1):
                    for h in range(1, limit + 1):
                        buckets[w][h] += (buckets[w - 1][h] +
                                          buckets[w][h - 1] - buckets[w - 1][h - 1])
                        low_x[w][h] = min(low_x[w][h], low_x[w - 1][h], low_x[w][h - 1])
                        low_y[w][h] = min(low_y[w][h], low_y[w - 1][h], low_y[w][h - 1])
                        high_x[w][h] = max(high_x[w][h], high_x[w - 1][h], high_x[w][h - 1])
                        high_y[w][h] = max(high_y[w][h], high_y[w - 1][h], high_y[w][h - 1])
                        if w < 2 or h < 2:
                            continue
                        surplus = buckets[w][h] - w * h
                        if surplus <= 0:
                            continue
                        width = high_x[w][h] - low_x[w][h]
                        height = high_y[w][h] - low_y[w][h]
                        f = max(width - w, height - h, 0.0)
                        if f >= 1 or f < 1e-10:
                            continue
                        if variants_per_signature == 1:
                            key = (w, h, surplus, True)
                            previous = best.get(key)
                            if previous is not None and f >= previous[0].overhang - 1e-8:
                                continue
                        min_x, min_y = low_x[w][h], low_y[w][h]
                        crop = tuple((x - min_x, y - min_y, angle)
                                     for (x, y, angle), (rw, rh) in zip(rows, required)
                                     if rw <= w and rh <= h)
                        gadget = Gadget(w, h, surplus, f, crop)
                        if variants_per_signature == 1:
                            best[key] = [gadget]
                            continue
                        # Both seam classes matter: a slightly wider crop with
                        # an empty upper tip may join where the narrowest one
                        # cannot. Keep geometrically different ties as well.
                        key = (w, h, surplus, has_empty_upper_tip(gadget))
                        variants = best.setdefault(key, [])
                        if any(existing.squares == gadget.squares for existing in variants):
                            continue
                        variants.append(gadget)
                        variants.sort(key=lambda item: item.overhang)
                        del variants[variants_per_signature:]
    return [gadget for variants in best.values() for gadget in variants]


def load_gadget_overrides(path: Path | None = None) -> list[Gadget]:
    """Load independently checked local improvements to extracted gadgets."""
    if path is None:
        path = Path(__file__).with_name("gadget_overrides.json")
    if not path.exists():
        return []
    overrides = []
    for entry in json.loads(path.read_text(encoding="utf-8")):
        w, h, surplus = (int(entry[key]) for key in ("w", "h", "surplus"))
        rows = [tuple(map(float, row)) for row in entry["squares"]]
        if w < 2 or h < 2 or surplus < 1 or len(rows) != w * h + surplus:
            raise ValueError(f"invalid gadget size: {w}x{h}+{surplus}")
        squares, width, height = normalize(rows)
        f = max(width - w, height - h, 0.0)
        if not 0 < f < 1 or abs(f - float(entry["overhang"])) > 1e-7:
            raise ValueError(f"invalid gadget overhang: {w}x{h}+{surplus}")
        geometry = sorted((make_square(i, *row) for i, row in enumerate(squares)),
                          key=lambda square: square.min_x)
        active = []
        for square in geometry:
            active = [other for other in active if other.max_x > square.min_x]
            for other in active:
                if (square.max_y > other.min_y and other.max_y > square.min_y
                        and squares_overlap(square, other, tolerance=0)):
                    raise ValueError(f"overlapping gadget: {w}x{h}+{surplus}")
            active.append(square)
        overrides.append(Gadget(w, h, surplus, f, squares))
    return overrides


def seam_compatible(a: Gadget, b: Gadget) -> bool:
    """Check the only region where consecutive diagonal gadgets may meet."""
    a_corner = []
    for i, row in enumerate(a.squares):
        square = make_square(i, *row)
        if square.max_x > a.w and square.max_y > a.h:
            a_corner.append(square)
    if not a_corner:
        return True
    b_corner = []
    for i, (x, y, angle) in enumerate(b.squares):
        square = make_square(i, x + a.w, y + a.h, angle)
        if square.min_x < a.w + a.overhang and square.min_y < a.h + a.overhang:
            b_corner.append(square)
    for first in a_corner:
        for second in b_corner:
            if (first.max_x > second.min_x and second.max_x > first.min_x and
                    first.max_y > second.min_y and second.max_y > first.min_y and
                    squares_overlap(first, second, tolerance=0)):
                return False
    return True


def compose(gadgets: Iterable[Gadget], clearance: float = 1e-8) -> Packing:
    """Fill every off-diagonal rectangle and validate the complete packing."""
    chain = tuple(gadgets)
    if not chain or sum(g.w for g in chain) != sum(g.h for g in chain):
        raise ValueError("gadget widths and heights must sum to the same side")
    f = max(g.overhang for g in chain)
    px = py = 0
    origins = []
    squares: list[Placement] = []
    for gadget in chain:
        origins.append((px, py))
        squares.extend((x + px, y + py, angle) for x, y, angle in gadget.squares)
        px += gadget.w
        py += gadget.h
    for i, first in enumerate(chain):
        x_i, y_i = origins[i]
        for j in range(i + 1, len(chain)):
            second = chain[j]
            x_j, y_j = origins[j]
            squares.extend((x_i + x + 0.5, y_j + f + y + 0.5, 0.0)
                           for x in range(first.w) for y in range(second.h))
            squares.extend((x_j + f + x + 0.5, y_i + y + 0.5, 0.0)
                           for x in range(second.w) for y in range(first.h))
    n = len(squares)
    k = px
    expected = k * k + sum(g.surplus for g in chain)
    if n != expected:
        raise AssertionError(f"composition has {n} squares, expected {expected}")
    # Scaling centres opens a small gap at exact grid contacts after CSV
    # serialization without changing the unit square side length.
    safe = [(x * (1 + clearance), y * (1 + clearance), angle)
            for x, y, angle in squares]
    measured = measure_packing(n, [make_square(i, *row)
                                   for i, row in enumerate(safe)])
    return measured.side, safe


# One upright lattice column or row on the fault between two gadgets. The
# column moves the next gadget one unit right, beyond the upper-right tip of
# the previous one, so that gadget may overhang its seam corner.
COLUMN_GAP = Gadget(1, 0, 0, 0.0, ())
ROW_GAP = Gadget(0, 1, 0, 0.0, ())


def has_empty_upper_tip(gadget: Gadget) -> bool:
    """An empty upper-right seam guarantees compatibility on that side."""
    return not any((square := make_square(i, *row)).max_x > gadget.w and
                   square.max_y > gadget.h
                   for i, row in enumerate(gadget.squares))


def fault_line_compositions(donors: Iterable[Packing],
                            incumbents: dict[int, Packing],
                            max_n: int = 1000,
                            locked: Iterable[int] = (),
                            provenance: dict[int, tuple[Gadget, ...]] | None = None
                            ) -> dict[int, Packing]:
    """Minimax DP over diagonal chains of rectangular gadgets.

    A nonterminal gadget either has an empty upper seam or is followed by a
    lattice column or row, so the next gadget cannot meet its tip. DP states record the columns,
    rows and surplus already used, retaining the smallest maximum overhang.
    """
    extracted = extract_corner_crops(donors)
    by_key = {(g.w, g.h, g.surplus): g for g in extracted}
    for override in load_gadget_overrides():
        key = override.w, override.h, override.surplus
        previous = by_key.get(key)
        if previous is None or override.overhang < previous.overhang - 1e-8:
            if previous is not None and has_empty_upper_tip(previous) and not has_empty_upper_tip(override):
                raise ValueError(f"gadget override closes upper seam: {key}")
            by_key[key] = override
    all_gadgets = list(by_key.values())
    # Pieces are what one DP transition places: a gadget whose upper seam is
    # empty, or any gadget followed by a lattice column or row.
    pieces: list[tuple[Gadget, ...]] = [(g,) for g in all_gadgets
                                        if has_empty_upper_tip(g)]
    pieces.extend((g, gap) for g in all_gadgets for gap in (COLUMN_GAP, ROW_GAP))
    piece_size = [(sum(g.w for g in piece), sum(g.h for g in piece),
                   sum(g.surplus for g in piece), piece[0].overhang)
                  for piece in pieces]
    max_k = math.isqrt(max_n)
    terminal_by_difference: dict[int, list[Gadget]] = defaultdict(list)
    for gadget in all_gadgets:
        terminal_by_difference[gadget.w - gadget.h].append(gadget)

    locked_set = set(locked)
    options: dict[int, list[tuple[float, int, tuple[int, ...], Gadget]]] = defaultdict(list)
    serial = 0
    # A transition increases w+h, so buckets can be processed in one pass.
    states: list[dict[tuple[int, int, int], tuple[float, tuple[int, ...]]]] = [
        {} for _ in range(2 * max_k + 1)
    ]
    for index, (w, h, piece_surplus, overhang) in enumerate(piece_size):
        if w <= max_k - 2 and h <= max_k - 2:
            key = (w, h, piece_surplus)
            bucket = states[w + h]
            previous = bucket.get(key)
            if previous is None or overhang < previous[0]:
                bucket[key] = overhang, (index,)

    for dimension in range(len(states)):
        for (used_w, used_h, surplus), (old_f, chain) in states[dimension].items():
            # Close the chain with any gadget. Equal width and height totals
            # form a square container; the off-diagonal blocks
            # are filled by compose().
            for final in terminal_by_difference[used_h - used_w]:
                k = used_w + final.w
                n = k * k + surplus + final.surplus
                if n > max_n or n in locked_set:
                    continue
                f = max(old_f, final.overhang)
                limit = incumbents[n][0] if n in incumbents else math.isqrt(n - 1) + 1
                if k + f + 2e-7 >= limit:
                    continue
                serial += 1
                item = (-f, serial, chain, final)
                best = options[n]
                if len(best) < 8:
                    heapq.heappush(best, item)
                elif item > best[0]:
                    heapq.heapreplace(best, item)

            for index, (piece_w, piece_h, piece_surplus, overhang) in enumerate(piece_size):
                w = used_w + piece_w
                h = used_h + piece_h
                if w > max_k - 2 or h > max_k - 2:
                    continue
                new_surplus = surplus + piece_surplus
                if (max(w, h) + 2) ** 2 + new_surplus + 1 > max_n:
                    continue
                f = max(old_f, overhang)
                key = (w, h, new_surplus)
                bucket = states[w + h]
                previous = bucket.get(key)
                if previous is None or f < previous[0] - 1e-10:
                    bucket[key] = f, chain + (index,)

    improved: dict[int, Packing] = {}
    for n, best in options.items():
        limit = incumbents[n][0] if n in incumbents else math.isqrt(n - 1) + 1
        for _, _, chain, final in sorted(best, reverse=True):
            sequence = [g for index in chain for g in pieces[index]] + [final]
            try:
                candidate = compose(sequence)
            except ValidationError:
                continue
            if candidate[0] < limit - 1e-9:
                improved[n] = candidate
                if provenance is not None:
                    provenance[n] = tuple(sequence)
                break
    return improved
