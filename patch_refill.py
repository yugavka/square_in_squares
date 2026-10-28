"""Destroy and refill small patches of a validated square packing.

Remove neighbouring squares, reclose the survivors with LP, generate positions
against walls and nearby square edges, and choose a non-overlapping refill with
MILP. Final candidates pass independent strict geometry validation. This is an
optional SciPy search; the normal submission generator has no SciPy dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from joint_polish import polish
from optimal_locks import proven_optimal
from projected_search import (load_selected, normalized, restore_lp,
                              strict_side, write_partial)
from validate_submission import make_square, squares_overlap, total_score, validate_submission


Placement = tuple[float, float, float]
ROOT = Path(__file__).resolve().parent


def select_patch(packing: list[Placement], size: int,
                 rng: random.Random, trial: int) -> tuple[list[Placement], list[Placement]]:
    """Pick a spatially coherent boundary or tilted-core group."""
    squares = [make_square(i, *row) for i, row in enumerate(packing)]
    side = strict_side(packing)
    if trial % 3 == 0:
        anchors = sorted(range(len(squares)), key=lambda i:
                         min(squares[i].min_x, squares[i].min_y,
                             side - squares[i].max_x,
                             side - squares[i].max_y))
        anchor = rng.choice(anchors[:max(size * 3, 1)])
    elif trial % 3 == 1:
        tilted = [i for i, row in enumerate(packing)
                  if abs(math.remainder(row[2], math.pi / 2)) > 0.03]
        anchor = rng.choice(tilted or list(range(len(packing))))
    else:
        anchor = rng.randrange(len(packing))
    x, y, _ = packing[anchor]
    chosen = set(sorted(range(len(packing)), key=lambda i:
                        (packing[i][0] - x) ** 2 +
                        (packing[i][1] - y) ** 2)[:size])
    return ([row for i, row in enumerate(packing) if i not in chosen],
            [row for i, row in enumerate(packing) if i in chosen])


def candidate_positions(survivors: list[Placement], removed: list[Placement],
                        side: float, maximum: int = 180) -> list[Placement]:
    """Enumerate wall and contact anchored placements near the removed patch."""
    geometry = [make_square(i, *row) for i, row in enumerate(survivors)]
    local_left = min(x for x, _, _ in removed) - 1.5
    local_right = max(x for x, _, _ in removed) + 1.5
    local_bottom = min(y for _, y, _ in removed) - 1.5
    local_top = max(y for _, y, _ in removed) + 1.5
    nearby = [square for square in geometry
              if (square.max_x >= local_left - 1 and
                  square.min_x <= local_right + 1 and
                  square.max_y >= local_bottom - 1 and
                  square.min_y <= local_top + 1)]
    angles = [0.0, math.pi / 4]
    for _, _, angle in removed:
        a = math.remainder(angle, math.pi / 2)
        if all(abs(a - old) > 0.025 for old in angles):
            angles.append(a)
    candidates: list[tuple[float, Placement]] = []
    margin = 3e-8
    for angle in angles:
        half = (abs(math.cos(angle)) + abs(math.sin(angle))) / 2
        x_values = [half + margin, side - half - margin]
        y_values = [half + margin, side - half - margin]
        for square in nearby:
            x_values.extend((square.max_x + half + margin,
                             square.min_x - half - margin))
            y_values.extend((square.max_y + half + margin,
                             square.min_y - half - margin))
        x_values.extend(x for x, _, _ in removed)
        y_values.extend(y for _, y, _ in removed)
        xs = sorted({round(x, 9) for x in x_values
                     if half + margin <= x <= side - half - margin and
                     local_left <= x <= local_right})
        ys = sorted({round(y, 9) for y in y_values
                     if half + margin <= y <= side - half - margin and
                     local_bottom <= y <= local_top})
        for x in xs:
            for y in ys:
                square = make_square(-1, x, y, angle)
                if (square.min_x < margin / 2 or square.min_y < margin / 2 or
                        square.max_x > side - margin / 2 or
                        square.max_y > side - margin / 2):
                    continue
                if any((square.max_x > other.min_x and
                        other.max_x > square.min_x and
                        square.max_y > other.min_y and
                        other.max_y > square.min_y and
                        squares_overlap(square, other, tolerance=0))
                       for other in nearby):
                    continue
                priority = min((x - old_x) ** 2 + (y - old_y) ** 2
                               for old_x, old_y, _ in removed)
                candidates.append((priority, (x, y, angle)))
    candidates.sort(key=lambda item: item[0])
    return [placement for _, placement in candidates[:maximum]]


def milp_refill(candidates: list[Placement], count: int,
                seconds: float = 3.0,
                overlap_tolerance: float = 0.0) -> list[Placement] | None:
    """Choose count pairwise compatible placements from a candidate bank."""
    if len(candidates) < count:
        return None
    squares = [make_square(i, *row) for i, row in enumerate(candidates)]
    conflicts = []
    for i, a in enumerate(squares):
        for j in range(i + 1, len(squares)):
            b = squares[j]
            if (a.max_x > b.min_x and b.max_x > a.min_x and
                    a.max_y > b.min_y and b.max_y > a.min_y and
                    squares_overlap(a, b, tolerance=overlap_tolerance)):
                conflicts.append((i, j))
    rows = [0] * len(candidates)
    cols = list(range(len(candidates)))
    values = [1.0] * len(candidates)
    for row, (i, j) in enumerate(conflicts, start=1):
        rows.extend((row, row))
        cols.extend((i, j))
        values.extend((1.0, 1.0))
    matrix = coo_matrix((values, (rows, cols)),
                        shape=(len(conflicts) + 1, len(candidates))).tocsr()
    lower = np.zeros(len(conflicts) + 1)
    upper = np.ones(len(conflicts) + 1)
    lower[0] = upper[0] = count
    objective = np.arange(len(candidates), dtype=float) / max(len(candidates), 1)
    result = milp(objective, integrality=np.ones(len(candidates)),
                  bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, lower, upper),
                  options={"time_limit": seconds})
    if result.x is None:
        return None
    selected = [candidates[i] for i, value in enumerate(result.x)
                if value > 0.5]
    return selected if len(selected) == count else None


def search_one(packing: list[Placement], trials: int, patch_size: int,
               seed: int, target_reductions: tuple[float, ...],
               milp_seconds: float = 3.0) -> tuple[list[Placement], dict]:
    rng = random.Random(seed)
    best = normalized(packing)
    initial_side = best_side = strict_side(best)
    stats = {"survivor_reclosures": 0, "refill_feasible": 0,
             "strict_candidates": 0, "joint_improvements": 0}
    for trial in range(trials):
        survivors, removed = select_patch(best, patch_size, rng, trial)
        target = best_side - target_reductions[trial % len(target_reductions)]
        survivors = restore_lp(survivors, target + 1e-8, 0.18, rng,
                               time_limit=2.0)
        if survivors is None:
            continue
        stats["survivor_reclosures"] += 1
        candidates = candidate_positions(survivors, removed, target)
        chosen = milp_refill(candidates, len(removed), milp_seconds)
        if chosen is None:
            continue
        stats["refill_feasible"] += 1
        joined = normalized(survivors + chosen)
        try:
            side = strict_side(joined)
        except ValueError:
            continue
        stats["strict_candidates"] += 1
        if side >= best_side - 1e-8:
            continue
        polished, gains = polish(joined, 10, strict_side, normalized)
        stats["joint_improvements"] += gains
        side = strict_side(polished)
        if side < best_side - 1e-8:
            best, best_side = polished, side
    return best, {"before_side": initial_side, "after_side": best_side,
                  "score_gain": (initial_side - best_side) /
                                math.sqrt(len(packing)), **stats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n", nargs="+", type=int)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "submission_improved.csv")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--milp-seconds", type=float, default=3.0)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "patch_candidates.csv")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "reports" / "patch_refill.json")
    args = parser.parse_args()
    selected = set(args.n)
    if (not all(1 <= n <= 1000 and proven_optimal(n) is None for n in selected)
            or args.trials < 1 or not 2 <= args.patch_size <= 15
            or any(n <= args.patch_size for n in selected)
            or args.milp_seconds <= 0):
        parser.error("choose unlocked n>patch-size, trials>0, "
                     "2<=patch-size<=15, and milp-seconds>0")
    results = validate_submission(args.input)
    baseline = total_score(results)
    layouts = load_selected(args.input, selected)
    accepted = {}
    records = []
    for n in sorted(selected):
        started = time.monotonic()
        candidate, record = search_one(layouts[n], args.trials,
                                       args.patch_size, args.seed + n,
                                       (1e-5, 1e-4, 1e-3, 0.005),
                                       args.milp_seconds)
        if record["score_gain"] > 1e-8:
            accepted[n] = candidate
        record.update(n=n, accepted=n in accepted,
                      seconds=time.monotonic() - started)
        records.append(record)
        print(f"n={n}: gain={record['score_gain']:.12g}, "
              f"reclosures={record['survivor_reclosures']}, "
              f"MILP={record['refill_feasible']}", flush=True)
    if accepted:
        write_partial(args.output, accepted)
    report = {"input": str(args.input), "baseline_score": baseline,
              "candidate_score": baseline - sum(r["score_gain"] for r in records
                                                if r["accepted"]),
              "seed": args.seed, "trials": args.trials,
              "patch_size": args.patch_size,
              "milp_seconds": args.milp_seconds,
              "accepted_n": sorted(accepted),
              "candidate_csv": str(args.output) if accepted else None,
              "records": records}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Accepted {len(accepted)} layouts; estimated score "
          f"{report['candidate_score']:.12f}; report {args.report}")


if __name__ == "__main__":
    main()
