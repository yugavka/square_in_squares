"""Try new contact patterns and compact square centres with linear programming.

This is an experimental local search. A candidate is imported only when the
independent strict geometry check confirms an improvement over the submission.
Requires SciPy in the interpreter used for the search; the submission generator
has no SciPy dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from import_published_packings import checked_side, import_entries
from optimal_locks import proven_optimal
from packing_solution import improved_packings


Placement = tuple[float, float, float]


def separating_axis(a: Placement, b: Placement) -> tuple[float, float, float, int, float]:
    """Choose the most separated of the four SAT axes for a pair."""
    ax, ay, aa = a
    bx, by, ba = b
    ac, ass = math.cos(aa), math.sin(aa)
    bc, bs = math.cos(ba), math.sin(ba)
    candidates = []
    for ux, uy in ((ac, ass), (-ass, ac), (bc, bs), (-bs, bc)):
        a_radius = (abs(ac * ux + ass * uy) +
                    abs(-ass * ux + ac * uy)) / 2
        b_radius = (abs(bc * ux + bs * uy) +
                    abs(-bs * ux + bc * uy)) / 2
        projection = (bx - ax) * ux + (by - ay) * uy
        sign = 1 if projection >= 0 else -1
        gap = abs(projection) - a_radius - b_radius
        candidates.append((gap, ux, uy, sign, a_radius + b_radius))
    return max(candidates, key=lambda item: item[0])


def compact_centres(placements: list[Placement],
                    clearance: float = 2e-7) -> list[Placement] | None:
    """Minimize the side with angles and pair-separating axes held fixed."""
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise RuntimeError("squeeze_layout.py requires scipy") from exc

    n = len(placements)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    limits: list[float] = []

    def add(entries: tuple[tuple[int, float], ...], bound: float) -> None:
        row = len(limits)
        for col, value in entries:
            rows.append(row)
            cols.append(col)
            values.append(value)
        limits.append(bound)

    for i, (_, _, angle) in enumerate(placements):
        radius = (abs(math.cos(angle)) + abs(math.sin(angle))) / 2
        add(((2 * i, -1),), -radius)
        add(((2 * i, 1), (2 * n, -1)), -radius)
        add(((2 * i + 1, -1),), -radius)
        add(((2 * i + 1, 1), (2 * n, -1)), -radius)

    for i, a in enumerate(placements):
        for j in range(i + 1, n):
            _, ux, uy, sign, needed = separating_axis(a, placements[j])
            add(((2 * i, sign * ux), (2 * i + 1, sign * uy),
                 (2 * j, -sign * ux), (2 * j + 1, -sign * uy)),
                -(needed + clearance))

    matrix = coo_matrix((values, (rows, cols)),
                        shape=(len(limits), 2 * n + 1)).tocsr()
    objective = [0.0] * (2 * n) + [1.0]
    result = linprog(objective, A_ub=matrix, b_ub=limits,
                     bounds=[(None, None)] * (2 * n) + [(0, None)],
                     method="highs")
    if not result.success:
        return None
    return [(float(result.x[2 * i]), float(result.x[2 * i + 1]), angle)
            for i, (_, _, angle) in enumerate(placements)]


def perturb(placements: list[Placement], rng: random.Random,
            position_noise: float, angle_noise: float) -> list[Placement]:
    """Perturb the contact graph while keeping equal angles in one group."""
    angle_shifts: dict[float, float] = {}
    changed = []
    for x, y, angle in placements:
        key = round(angle, 9)
        if key not in angle_shifts:
            angle_shifts[key] = (0.0 if abs(angle) < 1e-8 else
                                 rng.gauss(0, angle_noise))
        changed.append((x + rng.gauss(0, position_noise),
                        y + rng.gauss(0, position_noise),
                        angle + angle_shifts[key]))
    return changed


def search(n: int, trials: int, position_noise: float, angle_noise: float,
           seed: int) -> tuple[float, list[Placement] | None, int]:
    """Return a strictly checked improvement, if the trials find one."""
    if proven_optimal(n) is not None:
        raise ValueError(f"n={n} has a proof-backed layout")
    base = improved_packings()[n]
    original_side, original = base
    best_side = original_side
    best = None
    feasible = 0
    rng = random.Random(seed)
    for trial in range(trials):
        start = (original if trial == 0 else
                 perturb(original, rng, position_noise, angle_noise))
        candidate = compact_centres(start)
        if candidate is None:
            continue
        try:
            # The importer applies the same normalization and safety spacing.
            side = checked_side(n, [list(row) for row in candidate])
        except ValueError:
            continue
        feasible += 1
        if side < best_side - 1e-6:
            best_side, best = side, candidate
    return best_side, best, feasible


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n", type=int)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--position-noise", type=float, default=0.15)
    parser.add_argument("--angle-noise", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--report", type=Path,
                        default=Path(__file__).with_name("squeeze_results.json"))
    parser.add_argument("--apply", action="store_true",
                        help="import a verified improvement into record_seeds.json")
    args = parser.parse_args()
    if args.trials < 1 or args.position_noise < 0 or args.angle_noise < 0:
        parser.error("trials must be positive and noise scales nonnegative")
    side, squares, feasible = search(args.n, args.trials,
                                     args.position_noise, args.angle_noise,
                                     args.seed)
    report = {"n": args.n, "trials": args.trials, "feasible": feasible,
              "improved_side": side if squares is not None else None,
              "squares": squares}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.apply and squares is not None:
        entry = {"source": "local linear programming search",
                 "published_side": str(side), "squares": squares}
        import_entries(Path(__file__).with_name("record_seeds.json"),
                       [(args.n, entry)])
    print(f"n={args.n}: feasible {feasible}/{args.trials}, "
          f"improved side {report['improved_side']}; report {args.report}")


if __name__ == "__main__":
    main()
