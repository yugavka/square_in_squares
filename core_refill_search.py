"""Mutate a tilted rectangular core, refill uprights with MILP, then compact.

This is an optional SciPy experiment. A gadget is split into tilted and upright
squares. Uprights are discarded before macro mutations of the tilted core and
are regenerated from wall and contact anchored candidates. Every accepted
packing passes the independent strict geometry check in search_gadget.py.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from fault_line_dp import Gadget, extract_corner_crops
from patch_refill import milp_refill
from projected_search import load_selected
from search_gadget import checked_gadget, compact_rectangle
from squeeze_layout import separating_axis
from validate_submission import ValidationError, make_square, measure_packing, squares_overlap


Placement = tuple[float, float, float]
ROOT = Path(__file__).resolve().parent


def split_gadget(gadget: Gadget, angle_threshold: float = 0.01
                 ) -> tuple[list[Placement], list[Placement]]:
    tilted, upright = [], []
    for row in gadget.squares:
        (tilted if abs(math.remainder(row[2], math.pi / 2)) > angle_threshold
         else upright).append(row)
    return tilted, upright


def mutate_core(core: list[Placement], upright: list[Placement],
                rng: random.Random, trial: int,
                size: float) -> tuple[list[Placement], int]:
    """Shear, re-angle, walk a taper, or absorb one former upright square."""
    proposal = list(core)
    if trial == 0:
        return proposal, 0
    cx = sum(x for x, _, _ in core) / len(core)
    cy = sum(y for _, y, _ in core) / len(core)
    mode = (trial - 1) % 4
    shift = rng.choice((-1.0, 1.0)) * size * rng.uniform(0.4, 1.0)
    if mode == 0:
        proposal = [(x + shift * (y - cy), y, angle + shift * 0.2)
                    for x, y, angle in core]
    elif mode == 1:
        c, s = math.cos(shift), math.sin(shift)
        proposal = [(cx + c * (x - cx) - s * (y - cy),
                     cy + s * (x - cx) + c * (y - cy), angle + shift)
                    for x, y, angle in core]
    elif mode == 2:
        mean_angle = sum(a for _, _, a in core) / len(core)
        ux, uy = math.cos(mean_angle), math.sin(mean_angle)
        projections = [x * ux + y * uy for x, y, _ in core]
        edge = sorted(projections)[max(0, len(projections) // 6)]
        selected = [i for i, value in enumerate(projections) if value <= edge]
        proposal = [(x + shift * ux, y + shift * uy, angle)
                    if i in selected else (x, y, angle)
                    for i, (x, y, angle) in enumerate(core)]
    else:
        if not upright:
            return proposal, 0
        # A near-interface upright is reclassified as a tilted core square.
        pick = min(upright, key=lambda row:
                   min((row[0] - x) ** 2 + (row[1] - y) ** 2
                       for x, y, _ in core))
        angle = min(core, key=lambda row:
                    (row[0] - pick[0]) ** 2 + (row[1] - pick[1]) ** 2)[2]
        proposal.append((pick[0], pick[1], angle + shift * 0.2))
        return proposal, 1
    return proposal, 0


def repair_core(proposal: list[Placement], width: float, height: float,
                trust: float, clearance: float = 2e-8
                ) -> list[Placement] | None:
    """Find a nearby collision-free core with fixed angles in a rectangle."""
    n = len(proposal)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    rhs: list[float] = []

    def add(entries: tuple[tuple[int, float], ...], bound: float) -> None:
        row = len(rhs)
        for col, value in entries:
            rows.append(row)
            cols.append(col)
            values.append(value)
        rhs.append(bound)

    for i, (x, y, angle) in enumerate(proposal):
        radius = (abs(math.cos(angle)) + abs(math.sin(angle))) / 2
        add(((2 * i, -1),), x - radius - clearance)
        add(((2 * i, 1),), width - x - radius - clearance)
        add(((2 * i + 1, -1),), y - radius - clearance)
        add(((2 * i + 1, 1),), height - y - radius - clearance)
        for dim in range(2):
            add(((2 * i + dim, 1), (2 * n + 2 * i + dim, -1)), 0)
            add(((2 * i + dim, -1), (2 * n + 2 * i + dim, -1)), 0)
    pair_radius = math.sqrt(2) + 2 * math.sqrt(2) * trust + 1e-6
    pairs = cKDTree([(x, y) for x, y, _ in proposal]).query_pairs(pair_radius)
    for i, j in sorted(pairs):
        _, ux, uy, sign, needed = separating_axis(proposal[i], proposal[j])
        dx = proposal[j][0] - proposal[i][0]
        dy = proposal[j][1] - proposal[i][1]
        add(((2 * i, sign * ux), (2 * i + 1, sign * uy),
             (2 * j, -sign * ux), (2 * j + 1, -sign * uy)),
            sign * (dx * ux + dy * uy) - needed - clearance)
    matrix = coo_matrix((values, (rows, cols)),
                        shape=(len(rhs), 4 * n)).tocsr()
    result = linprog([0.0] * (2 * n) + [1.0] * (2 * n),
                     A_ub=matrix, b_ub=rhs,
                     bounds=[(-trust, trust)] * (2 * n) +
                            [(0, trust)] * (2 * n), method="highs",
                     options={"time_limit": 2.0})
    if not result.success:
        return None
    repaired = [(x + float(result.x[2 * i]),
                 y + float(result.x[2 * i + 1]), angle)
                for i, (x, y, angle) in enumerate(proposal)]
    try:
        measure_packing(n, [make_square(i, *row)
                            for i, row in enumerate(repaired)])
    except ValidationError:
        return None
    return repaired


def upright_candidates(core: list[Placement], old_upright: list[Placement],
                       width: float, height: float,
                       maximum: int = 320,
                       overlap_tolerance: float = 0.0) -> list[Placement]:
    """Enumerate four wall grids and exact contact anchored staircase points."""
    margin = 1e-7
    geometry = [make_square(i, *row) for i, row in enumerate(core)]
    def compatible(row: Placement) -> bool:
        square = make_square(-1, *row)
        return not any((square.max_x > other.min_x and
                        other.max_x > square.min_x and
                        square.max_y > other.min_y and
                        other.max_y > square.min_y and
                        squares_overlap(square, other,
                                        tolerance=overlap_tolerance))
                       for other in geometry)

    original = [row for row in old_upright if compatible(row)]
    xs = {round(x, 12) for x, _, _ in old_upright}
    ys = {round(y, 12) for _, y, _ in old_upright}
    for i in range(math.floor(width)):
        xs.update((round(0.5 + i + margin, 12),
                   round(width - 0.5 - i - margin, 12)))
    for i in range(math.floor(height)):
        ys.update((round(0.5 + i + margin, 12),
                   round(height - 0.5 - i - margin, 12)))
    for square in geometry:
        xs.update((round(square.min_x - 0.5 - margin, 12),
                   round(square.max_x + 0.5 + margin, 12)))
        ys.update((round(square.min_y - 0.5 - margin, 12),
                   round(square.max_y + 0.5 + margin, 12)))
    candidates = []
    for x in sorted(xs):
        if not 0.5 - 1e-8 <= x <= width - 0.5 + 1e-8:
            continue
        for y in sorted(ys):
            if not 0.5 - 1e-8 <= y <= height - 0.5 + 1e-8:
                continue
            if not compatible((x, y, 0.0)):
                continue
            priority = min((x - ox) ** 2 + (y - oy) ** 2
                           for ox, oy, _ in old_upright)
            candidates.append((priority, (x, y, 0.0)))
    candidates.sort(key=lambda item: item[0])
    return original + [row for _, row in candidates[:max(0, maximum - len(original))]]


def search(gadget: Gadget, trials: int, seed: int, mutation_size: float,
           trust: float, milp_seconds: float,
           overlap_tolerance: float = 0.0) -> tuple[Gadget, dict]:
    rng = random.Random(seed)
    core, uprights = split_gadget(gadget)
    if not core or not uprights:
        raise ValueError("gadget needs both tilted and upright squares")
    width, height = gadget.w + gadget.overhang, gadget.h + gadget.overhang
    best = gadget
    stats = {"core_repaired": 0, "refill_feasible": 0,
             "assembly_repaired": 0, "strict_candidates": 0,
             "improved_trials": []}
    for trial in range(trials):
        proposal, converted = mutate_core(core, uprights, rng, trial,
                                          mutation_size)
        repaired = core if trial == 0 else repair_core(proposal, width, height, trust)
        if repaired is None:
            continue
        stats["core_repaired"] += 1
        original_uprights = uprights
        if converted:
            original_uprights = [row for row in uprights
                                 if not any(abs(row[0] - q[0]) < 1e-9 and
                                            abs(row[1] - q[1]) < 1e-9
                                            for q in proposal[len(core):])]
        candidates = upright_candidates(repaired, original_uprights,
                                        width, height,
                                        overlap_tolerance=overlap_tolerance)
        chosen = milp_refill(candidates, len(original_uprights),
                             milp_seconds,
                             overlap_tolerance=overlap_tolerance)
        if chosen is None:
            continue
        stats["refill_feasible"] += 1
        joined = repaired + chosen
        try:
            overhang, normalized = checked_gadget(gadget, joined)
        except ValueError:
            if overlap_tolerance <= 0:
                continue
            restored = repair_core(joined, width, height,
                                   max(0.04, overlap_tolerance * 4))
            if restored is None:
                continue
            try:
                overhang, normalized = checked_gadget(gadget, restored)
            except ValueError:
                continue
            stats["assembly_repaired"] += 1
        stats["strict_candidates"] += 1
        aligned = Gadget(gadget.w, gadget.h, gadget.surplus,
                         overhang, normalized)
        compacted = compact_rectangle(aligned, list(normalized))
        if compacted is not None and compacted[0] < overhang:
            overhang, normalized = compacted
        if overhang < best.overhang - 1e-8:
            best = Gadget(gadget.w, gadget.h, gadget.surplus,
                          overhang, normalized)
            stats["improved_trials"].append({"trial": trial,
                                              "overhang": overhang})
    return best, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "submission_improved.csv")
    parser.add_argument("--donor", type=int, default=753)
    parser.add_argument("--w", type=int, default=8)
    parser.add_argument("--h", type=int, default=6)
    parser.add_argument("--surplus", type=int, default=3)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--mutation-size", type=float, default=0.04)
    parser.add_argument("--trust", type=float, default=0.18)
    parser.add_argument("--milp-seconds", type=float, default=2.0)
    parser.add_argument("--overlap-tolerance", type=float, default=0.0,
                        help="temporary overlap allowed during MILP refill")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "reports" / "core_refill_pilot.json")
    args = parser.parse_args()
    if (args.trials < 1 or args.mutation_size <= 0 or args.trust <= 0 or
            args.overlap_tolerance < 0):
        parser.error("invalid trial, mutation, trust or overlap tolerance")
    rows = load_selected(args.input, {args.donor})[args.donor]
    donor_side = measure_packing(args.donor,
                                 [make_square(i, *row)
                                  for i, row in enumerate(rows)]).side
    matches = [g for g in extract_corner_crops([(donor_side, rows)],
                                                max_dim=max(args.w, args.h))
               if (g.w, g.h, g.surplus) == (args.w, args.h, args.surplus)]
    if not matches:
        parser.error("requested gadget signature is absent from donor")
    initial = min(matches, key=lambda g: g.overhang)
    started = time.monotonic()
    best, stats = search(initial, args.trials, args.seed,
                         args.mutation_size, args.trust, args.milp_seconds,
                         args.overlap_tolerance)
    report = {"donor_n": args.donor, "signature": [args.w, args.h, args.surplus],
              "initial_overhang": initial.overhang,
              "final_overhang": best.overhang,
              "trials": args.trials, "seed": args.seed,
              "mutation_size": args.mutation_size, "trust": args.trust,
              "milp_seconds": args.milp_seconds,
              "overlap_tolerance": args.overlap_tolerance,
              "tilted_count": len(split_gadget(initial)[0]),
              "upright_count": len(split_gadget(initial)[1]),
              "seconds": time.monotonic() - started,
              **stats, "squares": best.squares if best is not initial else None}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "squares"},
                     indent=2))


if __name__ == "__main__":
    main()
