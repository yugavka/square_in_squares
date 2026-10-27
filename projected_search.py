"""Bounded overlap-and-repair search for selected square packings.

Compressed and contact-kicked proposals may overlap. A fixed-angle LP moves
centres inside a trust region to restore a valid packing; a small independent
angle search then repeats the LP. Only strictly verified gains are exported.
SciPy is needed for this optional experiment, not for packing_solution.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

from optimal_locks import proven_optimal
from squeeze_layout import separating_axis
from validate_submission import (make_square, measure_packing,
                                 squares_overlap, total_score,
                                 validate_submission)


Placement = tuple[float, float, float]
ROOT = Path(__file__).resolve().parent


def strict_side(placements: list[Placement]) -> float:
    """Measure with an independent all-pair zero-tolerance SAT check."""
    squares = [make_square(i, *row) for i, row in enumerate(placements)]
    for i, a in enumerate(squares):
        for b in squares[i + 1:]:
            if (a.max_x > b.min_x and b.max_x > a.min_x and
                    a.max_y > b.min_y and b.max_y > a.min_y and
                    squares_overlap(a, b, tolerance=0)):
                raise ValueError(f"overlap between squares {a.square_id} and {b.square_id}")
    return measure_packing(len(squares), squares).side


def normalized(placements: list[Placement]) -> list[Placement]:
    squares = [make_square(i, *row) for i, row in enumerate(placements)]
    left = min(square.min_x for square in squares)
    bottom = min(square.min_y for square in squares)
    return [(x - left, y - bottom, angle) for x, y, angle in placements]


def restore_lp(placements: list[Placement], limit: float, trust: float,
               rng: random.Random, tie: float = 0.0,
               time_limit: float = 2.0) -> list[Placement] | None:
    """Repair a possibly overlapping proposal by one bounded fixed-angle LP."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    from scipy.spatial import cKDTree

    p = normalized(placements)
    n = len(p)
    # Bounds do not require a valid layout; measure the bounding square directly.
    geom = [make_square(i, *row) for i, row in enumerate(p)]
    current = max(max(s.max_x for s in geom), max(s.max_y for s in geom))
    if current - trust >= limit:
        return None
    indices = cKDTree([(x, y) for x, y, _ in p]).query_pairs(
        math.sqrt(2) + 2 * math.sqrt(2) * trust + 1e-7)
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    rhs: list[float] = []

    def add(entries: tuple[tuple[int, float], ...], bound: float) -> None:
        row = len(rhs)
        for col, value in entries:
            row_indices.append(row)
            col_indices.append(col)
            values.append(value)
        rhs.append(bound)

    for i, j in sorted(indices):
        a, b = p[i], p[j]
        _, ux, uy, sign, needed = separating_axis(a, b)
        if tie:
            # A near-tied alternative can change the contact branch.
            alternatives = []
            ac, ass = math.cos(a[2]), math.sin(a[2])
            bc, bs = math.cos(b[2]), math.sin(b[2])
            for vx, vy in ((ac, ass), (-ass, ac), (bc, bs), (-bs, bc)):
                projection = (b[0] - a[0]) * vx + (b[1] - a[1]) * vy
                direction = 1 if projection >= 0 else -1
                ar = (abs(ac * vx + ass * vy) + abs(-ass * vx + ac * vy)) / 2
                br = (abs(bc * vx + bs * vy) + abs(-bs * vx + bc * vy)) / 2
                alternatives.append((abs(projection) - ar - br,
                                     vx, vy, direction, ar + br))
            best_gap = max(item[0] for item in alternatives)
            choices = [item for item in alternatives
                       if item[0] >= best_gap - tie]
            _, ux, uy, sign, needed = rng.choice(choices)
        projection = sign * ((b[0] - a[0]) * ux + (b[1] - a[1]) * uy)
        add(((2 * i, sign * ux), (2 * i + 1, sign * uy),
             (2 * j, -sign * ux), (2 * j + 1, -sign * uy)),
            projection - needed - 3e-9)

    for i, square in enumerate(geom):
        radius = (square.max_x - square.min_x) / 2
        for dim, centre in ((0, square.x), (1, square.y)):
            add(((2 * i + dim, -1),), centre - radius - 3e-9)
            add(((2 * i + dim, 1), (2 * n, -1)),
                current - centre - radius - 3e-9)

    matrix = coo_matrix((values, (row_indices, col_indices)),
                        shape=(len(rhs), 2 * n + 1)).tocsr()
    result = linprog([0.0] * (2 * n) + [1.0], A_ub=matrix, b_ub=rhs,
                     bounds=[(-trust, trust)] * (2 * n) +
                            [(-trust, min(trust, limit - current))],
                     method="highs",
                     options={"time_limit": time_limit,
                              "primal_feasibility_tolerance": 1e-9,
                              "dual_feasibility_tolerance": 1e-9})
    if not result.success:
        return None
    candidate = [(x + float(result.x[2 * i]),
                  y + float(result.x[2 * i + 1]), angle)
                 for i, (x, y, angle) in enumerate(p)]
    try:
        if strict_side(candidate) < limit:
            return normalized(candidate)
    except ValueError:
        pass
    return None


def propose(base: list[Placement], rng: random.Random,
            depth: float, mode: int) -> list[Placement]:
    """Create a near-feasible candidate; overlap is intentional here."""
    p = normalized(base)
    geom = [make_square(i, *row) for i, row in enumerate(p)]
    side = max(max(s.max_x for s in geom), max(s.max_y for s in geom))
    factor = (side - depth - 1) / (side - 1)
    p = [((x - side / 2) * factor + (side - depth) / 2,
          (y - side / 2) * factor + (side - depth) / 2, a)
         for x, y, a in p]
    if mode == 1:
        centre = rng.choice(p)
        nearby = sorted(range(len(p)), key=lambda i:
                        (p[i][0] - centre[0]) ** 2 +
                        (p[i][1] - centre[1]) ** 2)[:min(10, len(p))]
        pivot_x = sum(p[i][0] for i in nearby) / len(nearby)
        pivot_y = sum(p[i][1] for i in nearby) / len(nearby)
        turn = rng.choice((-1, 1)) * rng.uniform(0.004, 0.025)
        c, s = math.cos(turn), math.sin(turn)
        for i in nearby:
            x, y, angle = p[i]
            p[i] = (pivot_x + c * (x - pivot_x) - s * (y - pivot_y),
                    pivot_y + s * (x - pivot_x) + c * (y - pivot_y),
                    angle + turn)
    elif mode == 2:
        direction = rng.uniform(0, math.pi)
        ux, uy = math.cos(direction), math.sin(direction)
        vx, vy = -uy, ux
        projection = sorted(x * ux + y * uy for x, y, _ in p)
        cut = projection[len(projection) // 2]
        kick = rng.uniform(0.005, 0.025)
        p = [(x + (1 if x * ux + y * uy >= cut else -1) * vx * kick,
              y + (1 if x * ux + y * uy >= cut else -1) * vy * kick,
              a) for x, y, a in p]
    elif mode == 3:
        p = [(x + rng.gauss(0, 0.015), y + rng.gauss(0, 0.015),
              a + rng.gauss(0, 0.006)) for x, y, a in p]
    return p


def polish_angles(base: list[Placement], limit: float,
                  rng: random.Random, evaluations: int,
                  trust: float) -> tuple[list[Placement], int]:
    """Release individual angles and reclose centres after each proposal."""
    best = base
    successes = 0
    order = list(range(len(base)))
    rng.shuffle(order)
    # Try tilted squares first, then a few upright squares.
    order.sort(key=lambda i: abs(math.remainder(base[i][2], math.pi / 2)) < 1e-5)
    for attempt in range(evaluations):
        i = order[(attempt // 4) % len(order)]
        step = (0.006, -0.006, 0.0015, -0.0015)[attempt % 4]
        start = best.copy()
        x, y, angle = start[i]
        start[i] = x, y, angle + step
        candidate = restore_lp(start, limit + 0.01, trust, rng,
                               time_limit=1.0)
        if candidate is None:
            continue
        side = strict_side(candidate)
        if side < limit - 1e-9:
            best, limit = candidate, side
            successes += 1
    return best, successes


def load_selected(path: Path, selected: set[int]) -> dict[int, list[Placement]]:
    layouts = {n: [] for n in selected}
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            n = int(row["n"])
            if n in layouts:
                layouts[n].append((float(row["x"]), float(row["y"]),
                                   float(row["angle"])))
    for n, placements in layouts.items():
        if len(placements) != n:
            raise ValueError(f"n={n}: expected {n} squares")
    return layouts


def search_one(base: list[Placement], trials: int, seed: int,
               angle_evaluations: int, trust: float) -> tuple[list[Placement], dict]:
    rng = random.Random(seed)
    initial = strict_side(base)
    best = normalized(base)
    best_side = initial
    repairs = 0
    archive = [best]
    for trial in range(trials):
        mode = trial % 4
        depth = (0.0 if trial == 0 else
                 rng.choice((0.0005, 0.002, 0.006, 0.015, 0.03)))
        start = propose(rng.choice(archive) if trial % 5 == 4 else base,
                        rng, depth, mode)
        candidate = restore_lp(start, best_side + 0.02, trust, rng,
                               tie=0.01 if trial % 3 == 2 else 0.0)
        if candidate is None:
            continue
        repairs += 1
        side = strict_side(candidate)
        if side <= initial + 0.02 and trial % 2 == 0:
            archive.append(candidate)
            if len(archive) > 8:
                archive.pop(1)
        if side < best_side - 1e-8:
            best, best_side = candidate, side
    best, angle_gains = polish_angles(best, best_side - 1e-9, rng,
                                     angle_evaluations, trust / 2)
    final_side = strict_side(best)
    return best, {"before_side": initial, "after_side": final_side,
                  "gain": (initial - final_side) / math.sqrt(len(base)),
                  "repaired_proposals": repairs,
                  "angle_improvements": angle_gains}


def write_partial(path: Path, improved: dict[int, list[Placement]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(("id", "n", "square_id", "x", "y", "angle"))
        row_id = 0
        for n, placements in sorted(improved.items()):
            for square_id, (x, y, angle) in enumerate(placements):
                writer.writerow((row_id, n, square_id, format(x, ".17g"),
                                 format(y, ".17g"), format(angle, ".17g")))
                row_id += 1
    if improved:
        validate_submission(path, allow_partial=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n", nargs="+", type=int)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "submission_improved.csv")
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--angle-evaluations", type=int, default=24)
    parser.add_argument("--trust", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "projected_candidates.csv")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "reports" / "projected_search.json")
    args = parser.parse_args()
    selected = set(args.n)
    if (not all(1 <= n <= 1000 and proven_optimal(n) is None for n in selected)
            or args.trials < 1 or args.angle_evaluations < 0
            or not 0 < args.trust <= 0.5):
        parser.error("choose unlocked n in 1..1000, positive trials and 0<trust<=0.5")
    results = validate_submission(args.input)
    baseline_score = total_score(results)
    layouts = load_selected(args.input, selected)
    improved = {}
    records = []
    for n in sorted(selected):
        started = time.monotonic()
        candidate, record = search_one(layouts[n], args.trials,
                                       args.seed + n, args.angle_evaluations,
                                       args.trust)
        if record["gain"] > 1e-8:
            improved[n] = candidate
        record.update(n=n, accepted=n in improved,
                      seconds=time.monotonic() - started)
        records.append(record)
        print(f"n={n}: gain={record['gain']:.12g}, "
              f"repairs={record['repaired_proposals']}, "
              f"angles={record['angle_improvements']}", flush=True)
    if improved:
        write_partial(args.output, improved)
    report = {"input": str(args.input), "baseline_score": baseline_score,
              "candidate_score": baseline_score - sum(
                  row["gain"] for row in records if row["accepted"]),
              "seed": args.seed, "trials": args.trials,
              "angle_evaluations": args.angle_evaluations,
              "trust": args.trust, "accepted_n": sorted(improved),
              "candidate_csv": str(args.output) if improved else None,
              "records": records}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Accepted {len(improved)} layouts; estimated score "
          f"{report['candidate_score']:.12f}; report {args.report}")


if __name__ == "__main__":
    main()
