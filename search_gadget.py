"""Locally compact a rectangular fault-line gadget with fixed angles."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from fault_line_dp import (Gadget, extract_corner_crops, has_empty_upper_tip,
                           normalize)
from packing_solution import (PUBLIC_PORTFOLIO, goebel_strip_seed,
                              portfolio_packings, record_seeds)
from optimal_locks import five_square_seed
from squeeze_layout import separating_axis
from validate_submission import make_square, squares_overlap


ROOT = Path(__file__).resolve().parent


def current_gadget(w: int, h: int, surplus: int) -> Gadget:
    donors = ([five_square_seed(), goebel_strip_seed(5), goebel_strip_seed(7)]
              + record_seeds()
              + list(portfolio_packings([PUBLIC_PORTFOLIO], 1000).values()))
    return next(g for g in extract_corner_crops(donors)
                if (g.w, g.h, g.surplus) == (w, h, surplus))


def compact_rectangle(gadget: Gadget,
                      start: list[tuple[float, float, float]],
                      clearance: float = 2e-7
                      ) -> tuple[float, tuple[tuple[float, float, float], ...]] | None:
    """Minimize common width/height overhang for fixed separating axes."""
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise RuntimeError("search_gadget.py requires SciPy") from exc
    n = len(start)
    rows, cols, values, limits = [], [], [], []

    def add(entries: tuple[tuple[int, float], ...], bound: float) -> None:
        row = len(limits)
        for col, value in entries:
            rows.append(row)
            cols.append(col)
            values.append(value)
        limits.append(bound)

    for i, (_, _, angle) in enumerate(start):
        radius = (abs(math.cos(angle)) + abs(math.sin(angle))) / 2
        add(((2 * i, -1),), -radius)
        add(((2 * i, 1), (2 * n, -1)), gadget.w - radius)
        add(((2 * i + 1, -1),), -radius)
        add(((2 * i + 1, 1), (2 * n, -1)), gadget.h - radius)
        if has_empty_upper_tip(gadget):
            original = make_square(i, *gadget.squares[i])
            if original.max_x <= gadget.w + 1e-9:
                add(((2 * i, 1),), gadget.w - radius - 1e-6)
            else:
                add(((2 * i + 1, 1),), gadget.h - radius - 1e-6)
    for i, first in enumerate(start):
        for j in range(i + 1, n):
            _, ux, uy, sign, needed = separating_axis(first, start[j])
            add(((2 * i, sign * ux), (2 * i + 1, sign * uy),
                 (2 * j, -sign * ux), (2 * j + 1, -sign * uy)),
                -(needed + clearance))
    matrix = coo_matrix((values, (rows, cols)),
                        shape=(len(limits), 2 * n + 1)).tocsr()
    objective = [0.0] * (2 * n) + [1.0]
    result = linprog(objective, A_ub=matrix, b_ub=limits,
                     bounds=[(None, None)] * (2 * n) + [(0, 1)],
                     method="highs")
    if not result.success:
        return None
    candidate = [(float(result.x[2 * i]), float(result.x[2 * i + 1]), angle)
                 for i, (_, _, angle) in enumerate(start)]
    return checked_gadget(gadget, candidate)


def checked_gadget(gadget: Gadget, squares: list[tuple[float, float, float]]
                   ) -> tuple[float, tuple[tuple[float, float, float], ...]]:
    """Check strict non-overlap and the actual rectangular bounds."""
    if len(squares) != gadget.w * gadget.h + gadget.surplus:
        raise ValueError("wrong square count")
    moved, width, height = normalize(squares)
    geometry = sorted((make_square(i, *row) for i, row in enumerate(moved)),
                      key=lambda square: square.min_x)
    active = []
    for square in geometry:
        active = [other for other in active if other.max_x > square.min_x]
        for other in active:
            if (other.max_y > square.min_y and square.max_y > other.min_y
                    and squares_overlap(other, square, tolerance=0)):
                raise ValueError("squares overlap")
        active.append(square)
    if has_empty_upper_tip(gadget):
        candidate = Gadget(gadget.w, gadget.h, gadget.surplus, 0.0, moved)
        if not has_empty_upper_tip(candidate):
            raise ValueError("upper seam must remain empty")
    return max(width - gadget.w, height - gadget.h, 0.0), moved


def search(gadget: Gadget, trials: int, seed: int,
           position_noise: float, angle_noise: float
           ) -> tuple[float, tuple[tuple[float, float, float], ...] | None, int]:
    rng = random.Random(seed)
    best_f, best = gadget.overhang, None
    feasible = 0
    for trial in range(trials):
        if trial == 0:
            start = list(gadget.squares)
        else:
            start = [(x + rng.gauss(0, position_noise),
                      y + rng.gauss(0, position_noise),
                      angle + rng.gauss(0, angle_noise))
                     for x, y, angle in (best or gadget.squares)]
        try:
            result = compact_rectangle(gadget, start)
        except ValueError:
            continue
        if result is None:
            continue
        feasible += 1
        f, squares = result
        if f < best_f - 1e-6:
            best_f, best = f, squares
            print(f"trial {trial}: f={best_f:.9f}", flush=True)
    return best_f, best, feasible


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w", type=int, default=22)
    parser.add_argument("--h", type=int, default=10)
    parser.add_argument("--surplus", type=int, default=15)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--position-noise", type=float, default=0.05)
    parser.add_argument("--angle-noise", type=float, default=0.002)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "gadget_search_result.json")
    args = parser.parse_args()
    if args.trials < 1 or min(args.w, args.h, args.surplus) < 1:
        parser.error("dimensions, surplus and trials must be positive")
    gadget = current_gadget(args.w, args.h, args.surplus)
    f, squares, feasible = search(gadget, args.trials, args.seed,
                                 args.position_noise, args.angle_noise)
    output = {"w": gadget.w, "h": gadget.h, "surplus": gadget.surplus,
              "initial_overhang": gadget.overhang, "best_overhang": f,
              "trials": args.trials, "feasible": feasible,
              "squares": squares}
    args.output.write_text(json.dumps(output, indent=2) + "\n",
                           encoding="utf-8")
    print(f"f={gadget.overhang:.9f} -> {f:.9f}; "
          f"feasible {feasible}/{args.trials}; {args.output}")


if __name__ == "__main__":
    main()
