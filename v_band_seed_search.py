"""Build independent V-band seeds, then fill their corners with a MILP.

The two arms are mirrored tilted square lattices.  The upright bank contains
wall grids and positions anchored to band contacts.  This intentionally does
not borrow a tilted core from the current submission, so it measures whether
the topology can reach the incumbent before spending time on continuous polish.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from core_refill_search import upright_candidates
from optimal_locks import proven_optimal
from projected_search import load_selected, normalized, strict_side, write_partial
from validate_submission import make_square, squares_overlap


Placement = tuple[float, float, float]
ROOT = Path(__file__).resolve().parent


def v_band(side: float, angle: float, rows: int, vertex: float,
           phase: float, seam: float = 0.0) -> list[Placement]:
    """Return two non-overlapping mirrored arms clipped to the square."""
    c, s = math.cos(angle), math.sin(angle)
    radius = (c + s) / 2
    pitch = 1 + 2e-7
    right: list[Placement] = []
    for col in range(math.ceil(2 * side) + 2):
        for row in range(-rows, rows + 1):
            u, v = (col + phase) * pitch, row * pitch
            x = side / 2 + u * c - v * s
            y = vertex + u * s + v * c
            if (side / 2 + seam + radius <= x <= side - radius and
                    radius <= y <= side - radius):
                right.append((x, y, angle))
    core = right + [(side - x, y, -angle) for x, y, _ in right]
    if not core:
        return []
    try:
        strict_side(core)
    except ValueError:
        return []
    return core


def max_refill(candidates: list[Placement], seconds: float
               ) -> list[Placement] | None:
    """Maximum independent set among upright candidates with a time cap."""
    if not candidates:
        return []
    squares = [make_square(i, *row) for i, row in enumerate(candidates)]
    pairs = cKDTree([(x, y) for x, y, _ in candidates]).query_pairs(1.42)
    conflicts = []
    for i, j in sorted(pairs):
        a, b = squares[i], squares[j]
        if (a.max_x > b.min_x and b.max_x > a.min_x and
                a.max_y > b.min_y and b.max_y > a.min_y and
                squares_overlap(a, b, tolerance=0)):
            conflicts.append((i, j))
    rr, cc = [], []
    for k, (i, j) in enumerate(conflicts):
        rr.extend((k, k))
        cc.extend((i, j))
    matrix = coo_matrix((np.ones(len(rr)), (rr, cc)),
                        shape=(len(conflicts), len(candidates))).tocsr()
    result = milp(-np.ones(len(candidates)),
                  integrality=np.ones(len(candidates)), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, 0, 1),
                  options={"time_limit": seconds})
    if result.x is None:
        return None
    chosen = [row for row, value in zip(candidates, result.x) if value > 0.5]
    try:
        strict_side(chosen)
    except ValueError:
        return None
    return chosen


def search_one(n: int, incumbent: list[Placement], trials: int,
               reduction: float, milp_seconds: float,
               maximum: int) -> tuple[list[Placement] | None, dict]:
    before = strict_side(normalized(incumbent))
    side = before - reduction
    grid = [(0.5 + i, 0.5 + j, 0.0)
            for i in range(math.floor(side))
            for j in range(math.floor(side))]
    parameters = itertools.product((0.28, 0.45, math.pi / 4),
                                   (2, 3, 4),
                                   (0.15, 0.4, 0.65, 0.9),
                                   (0.0, 0.5))
    best_count = 0
    best_parameters = None
    valid_cores = solved = 0
    accepted = None
    for trial, (angle, rows, vertex_fraction, phase) in enumerate(parameters):
        if trial >= trials:
            break
        core = v_band(side, angle, rows, vertex_fraction * side, phase)
        if not core:
            continue
        valid_cores += 1
        candidates = upright_candidates(core, grid, side, side,
                                        maximum=maximum)
        chosen = max_refill(candidates, milp_seconds)
        if chosen is None:
            continue
        solved += 1
        count = len(core) + len(chosen)
        if count > best_count:
            best_count = count
            best_parameters = {"angle": angle, "rows": rows,
                               "vertex_fraction": vertex_fraction,
                               "phase": phase, "core": len(core),
                               "upright": len(chosen)}
        if count >= n:
            proposal = normalized(core + chosen[:n - len(core)])
            try:
                actual = strict_side(proposal)
            except ValueError:
                continue
            if actual < before - 1e-8:
                accepted = proposal
                break
    return accepted, {"n": n, "before_side": before, "target_side": side,
                      "trials": min(trials, 72), "valid_cores": valid_cores,
                      "milp_solutions": solved, "best_count": best_count,
                      "count_gap": max(0, n - best_count),
                      "best_parameters": best_parameters,
                      "accepted": accepted is not None,
                      "after_side": strict_side(accepted) if accepted else before}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n", nargs="+", type=int)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "submission_improved.csv")
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--reduction", type=float, default=1e-4)
    parser.add_argument("--milp-seconds", type=float, default=2.0)
    parser.add_argument("--maximum", type=int, default=640)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "v_band_candidates.csv")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "reports" / "v_band_seed_pilot.json")
    args = parser.parse_args()
    if (not all(1 <= n <= 1000 and proven_optimal(n) is None for n in args.n)
            or args.trials < 1 or args.reduction <= 0 or
            args.milp_seconds <= 0 or args.maximum < 1):
        parser.error("use unlocked n and positive search parameters")
    layouts = load_selected(args.input, set(args.n))
    accepted, records = {}, []
    for n in sorted(layouts):
        start = time.monotonic()
        candidate, record = search_one(n, layouts[n], args.trials,
                                       args.reduction, args.milp_seconds,
                                       args.maximum)
        record["seconds"] = time.monotonic() - start
        records.append(record)
        if candidate:
            accepted[n] = candidate
        print(f"n={n}: best {record['best_count']}/{n}, "
              f"accepted={record['accepted']}, "
              f"time={record['seconds']:.1f}s", flush=True)
    if accepted:
        write_partial(args.output, accepted)
    report = {"input": str(args.input), "reduction": args.reduction,
              "trials": args.trials, "milp_seconds": args.milp_seconds,
              "maximum": args.maximum, "accepted_n": sorted(accepted),
              "candidate_csv": str(args.output) if accepted else None,
              "records": records}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
