"""Grow a smaller donor packing by inserted squares, then relax globally.

A donor with m < n squares usually has a smaller side than the incumbent for
n. The search places the missing n-m squares in the emptiest unit windows of
a rasterized donor, scales the donor into a target box and runs the soft
separating-axis relaxation from staircase_soft_search with inserted squares
free to travel. Fixed-angle LP repair and the strict validator decide whether
a candidate is accepted. This is an optional SciPy experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from optimal_locks import proven_optimal
from projected_search import (load_selected, normalized, restore_lp,
                              strict_side, write_partial)
from staircase_soft_search import pair_indices, soft_energy_gradient
from validate_submission import validate_submission

Placement = tuple[float, float, float]
ROOT = Path(__file__).resolve().parent


def free_raster(packing: list[Placement], side: float,
                step: float) -> np.ndarray:
    """Boolean grid of sample points not covered by any square."""
    size = int(math.ceil(side / step))
    centres = (np.arange(size) + .5) * step
    xs, ys = np.meshgrid(centres, centres, indexing="ij")
    free = np.ones((size, size), dtype=bool)
    for x, y, angle in packing:
        reach = (abs(math.cos(angle)) + abs(math.sin(angle))) / 2
        i0, i1 = max(0, int((x - reach) / step)), min(size, int((x + reach) / step) + 2)
        j0, j1 = max(0, int((y - reach) / step)), min(size, int((y + reach) / step) + 2)
        dx = xs[i0:i1, j0:j1] - x
        dy = ys[i0:i1, j0:j1] - y
        c, s = math.cos(angle), math.sin(angle)
        inside = (np.abs(c * dx + s * dy) <= .5) & (np.abs(-s * dx + c * dy) <= .5)
        free[i0:i1, j0:j1] &= ~inside
    return free


def emptiest_windows(packing: list[Placement], side: float, count: int,
                     step: float = .05) -> list[Placement]:
    """Pick count disjoint upright unit windows with the most free area."""
    free = free_raster(packing, side, step).astype(np.int32)
    width = int(round(1 / step))
    table = np.pad(free.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    sums = (table[width:, width:] - table[:-width, width:] -
            table[width:, :-width] + table[:-width, :-width])
    chosen: list[Placement] = []
    blocked = np.zeros_like(sums, dtype=bool)
    for _ in range(count):
        masked = np.where(blocked, -1, sums)
        i, j = np.unravel_index(int(np.argmax(masked)), masked.shape)
        chosen.append(((i + width / 2) * step, (j + width / 2) * step, 0.0))
        blocked[max(0, i - width):i + width, max(0, j - width):j + width] = True
    return chosen


def interaction_pairs(packing: list[Placement], inserted: set[int],
                      trust: float, free_trust: float) -> list[tuple[int, int]]:
    """Pairs that can touch while every square stays inside its bounds."""
    pairs = set(pair_indices(packing, trust))
    tree = cKDTree([(x, y) for x, y, _ in packing])
    radius = math.sqrt(2) * (1 + trust + free_trust) + 1e-6
    for i in inserted:
        for j in tree.query_ball_point(packing[i][:2], radius):
            if i != j:
                pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def relax(start: list[Placement], target: float, inserted: set[int],
          trust: float, free_trust: float, iterations: int,
          inflate: bool) -> tuple[list[Placement], float]:
    """Soft relaxation in a fixed box; inserted squares may move far."""
    reference = np.asarray(start, dtype=float)
    vector = reference.ravel().copy()
    pairs = interaction_pairs(start, inserted, trust, free_trust)
    bounds = []
    for index, (x, y, angle) in enumerate(start):
        reach = free_trust if index in inserted else trust
        bounds.extend(((x - reach, x + reach), (y - reach, y + reach)))
        tilted = abs(math.remainder(angle, math.pi / 2)) > .01
        if index in inserted:
            bounds.append((angle - math.pi / 4, angle + math.pi / 4))
        elif tilted:
            bounds.append((angle - .08, angle + .08))
        else:
            bounds.append((angle - .02, angle + .02))
    stages = (((.9, 10.), (.96, 60.), (.99, 300.), (1., 3000.)) if inflate
              else ((1., 30.), (1., 300.), (1., 3000.)))
    result = None
    for square_size, stiffness in stages:
        result = minimize(soft_energy_gradient, vector, jac=True,
                          args=(reference, pairs, target, stiffness, .01,
                                2e-7, square_size),
                          method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": iterations, "ftol": 1e-12})
        vector = result.x
    relaxed = [tuple(map(float, row)) for row in vector.reshape(-1, 3)]
    return relaxed, float(result.fun)


def search(donor: list[Placement], n: int, targets: tuple[float, ...],
           trials: int, seed: int, iterations: int) -> tuple[list[Placement] | None, dict]:
    rng = random.Random(seed)
    base = normalized(donor)
    donor_side = strict_side(base)
    missing = n - len(base)
    records = []
    best, best_side = None, math.inf
    for trial in range(trials):
        target = targets[trial % len(targets)]
        scale = target / donor_side if target > donor_side else 1.0
        grown = [(x * scale + rng.gauss(0, .004 * (trial % 3)),
                  y * scale + rng.gauss(0, .004 * (trial % 3)), angle)
                 for x, y, angle in base]
        windows = emptiest_windows(grown, target, missing + 2)
        picks = rng.sample(windows, missing) if trial % 2 else windows[:missing]
        start = grown + picks
        inserted = set(range(len(grown), len(start)))
        started = time.monotonic()
        relaxed, energy = relax(start, target, inserted, trust=.25,
                                free_trust=2.0, iterations=iterations,
                                inflate=trial % 2 == 1)
        repaired = restore_lp(relaxed, target + 1e-8, trust=.06, rng=rng,
                              time_limit=20.0)
        side = None
        if repaired is not None:
            try:
                side = strict_side(repaired)
            except ValueError:
                side = None
        records.append({"trial": trial, "target": target, "soft_energy": energy,
                        "lp_repaired": repaired is not None, "side": side,
                        "seconds": round(time.monotonic() - started, 1)})
        print(f"  trial {trial}: target {target:.4f} energy {energy:.3e} "
              f"side {side}", flush=True)
        if side is not None and side < best_side:
            best, best_side = repaired, side
    return best, {"donor_n": len(base), "donor_side": donor_side,
                  "best_side": None if best is None else best_side,
                  "trials": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n", type=int)
    parser.add_argument("--donors", type=int, nargs="+", required=True,
                        help="smaller n whose layouts receive inserted squares")
    parser.add_argument("--targets", type=float, nargs="+", required=True,
                        help="box sides to relax into")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--input", type=Path, default=ROOT / "submission_improved.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "insert_relax_candidates.csv")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "reports" / "insert_relax.json")
    args = parser.parse_args()
    if (proven_optimal(args.n) is not None or not 1 < args.n <= 1000
            or any(not 0 < d < args.n for d in args.donors) or args.trials < 1):
        parser.error("choose an unlocked n and smaller donor sizes")
    incumbent = {r.n: r.side for r in validate_submission(args.input)}[args.n]
    layouts = load_selected(args.input, set(args.donors))
    report = {"n": args.n, "incumbent_side": incumbent, "targets": args.targets,
              "trials": args.trials, "iterations": args.iterations,
              "seed": args.seed, "donors": []}
    best, best_side = None, incumbent
    for donor in args.donors:
        print(f"donor {donor}:", flush=True)
        candidate, record = search(layouts[donor], args.n, tuple(args.targets),
                                   args.trials, args.seed + donor, args.iterations)
        report["donors"].append(record)
        if candidate is not None and record["best_side"] < best_side - 1e-9:
            best, best_side = candidate, record["best_side"]
    report["best_side"] = best_side
    report["score_gain"] = (incumbent - best_side) / math.sqrt(args.n)
    if best is not None:
        write_partial(args.output, {args.n: best})
        report["candidate_csv"] = str(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"incumbent {incumbent:.9f}, best {best_side:.9f}, "
          f"gain {report['score_gain']:.3e}; report {args.report}")


if __name__ == "__main__":
    main()
