"""Rebuild dense corner staircases, then globally relax and re-harden a packing.

The seed keeps the incumbent's irregular tilted contact network.  Unlike a
regular V lattice, it reproduces every square before searching.  A soft
separating-axis penalty lets all centres move and tilted angles change while
the box is held smaller; a fixed-angle LP then restores exact feasibility.
Only strictly validated improvements are exported.
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

from core_refill_search import upright_candidates
from optimal_locks import proven_optimal
from patch_refill import milp_refill
from projected_search import (load_selected, normalized, restore_lp,
                              strict_side, write_partial)


Placement = tuple[float, float, float]
ROOT = Path(__file__).resolve().parent


def split_layout(packing: list[Placement]) -> tuple[list[Placement], list[Placement]]:
    tilted, upright = [], []
    for row in packing:
        (tilted if abs(math.remainder(row[2], math.pi / 2)) > .01
         else upright).append(row)
    return tilted, upright


def rebuild_staircases(packing: list[Placement], maximum: int,
                       milp_seconds: float) -> list[Placement] | None:
    """Re-select aligned squares against the existing irregular tilted band."""
    packing = normalized(packing)
    side = strict_side(packing)
    core, upright = split_layout(packing)
    if not core or not upright:
        return None
    bank = upright_candidates(core, upright, side + 1e-7, side + 1e-7,
                              maximum=maximum)
    chosen = milp_refill(bank, len(upright), seconds=milp_seconds)
    if chosen is None:
        return None
    rebuilt = normalized(core + chosen)
    try:
        if strict_side(rebuilt) <= side + 1e-6:
            return rebuilt
    except ValueError:
        pass
    return None


def pair_indices(packing: list[Placement], trust: float) -> list[tuple[int, int]]:
    radius = math.sqrt(2) + 2 * math.sqrt(2) * trust + 1e-6
    return sorted(cKDTree([(x, y) for x, y, _ in packing]).query_pairs(radius))


def soft_energy_gradient(vector: np.ndarray, reference: np.ndarray,
                         pairs: list[tuple[int, int]], side: float,
                         stiffness: float, spring: float,
                         clearance: float,
                         square_size: float = 1.0) -> tuple[float, np.ndarray]:
    """Squared overlap and wall penalty with a separating-axis subgradient."""
    points = vector.reshape(-1, 3)
    ref = reference.reshape(-1, 3)
    gradient = spring * (points - ref)
    energy = .5 * spring * float(np.sum((points - ref) ** 2))
    cosines = np.cos(points[:, 2])
    sines = np.sin(points[:, 2])
    radii = square_size * (np.abs(cosines) + np.abs(sines)) / 2
    radius_derivative = (-np.sign(cosines) * sines +
                         np.sign(sines) * cosines) * square_size / 2
    for i, (x, y, _) in enumerate(points):
        for dim, coordinate in ((0, x), (1, y)):
            low = radii[i] - coordinate + clearance
            high = coordinate + radii[i] - side + clearance
            if low > 0:
                energy += .5 * stiffness * low * low
                gradient[i, dim] -= stiffness * low
                gradient[i, 2] += stiffness * low * radius_derivative[i]
            if high > 0:
                energy += .5 * stiffness * high * high
                gradient[i, dim] += stiffness * high
                gradient[i, 2] += stiffness * high * radius_derivative[i]
    for i, j in pairs:
        dx = points[j, 0] - points[i, 0]
        dy = points[j, 1] - points[i, 1]
        axes = ((cosines[i], sines[i], i, 0),
                (-sines[i], cosines[i], i, 1),
                (cosines[j], sines[j], j, 0),
                (-sines[j], cosines[j], j, 1))
        best = None
        for ux, uy, owner, kind in axes:
            ai = cosines[i] * ux + sines[i] * uy
            bi = -sines[i] * ux + cosines[i] * uy
            aj = cosines[j] * ux + sines[j] * uy
            bj = -sines[j] * ux + cosines[j] * uy
            projection = dx * ux + dy * uy
            overlap = square_size * (abs(ai) + abs(bi) +
                                     abs(aj) + abs(bj)) / 2 - abs(projection)
            if best is None or overlap < best[0]:
                best = (overlap, ux, uy, owner, kind,
                        ai, bi, aj, bj, projection)
        overlap, ux, uy, owner, kind, ai, bi, aj, bj, projection = best
        penetration = overlap + clearance
        if penetration <= 0:
            continue
        energy += .5 * stiffness * penetration * penetration
        factor = stiffness * penetration
        direction = 1 if projection >= 0 else -1
        gradient[i, 0] += factor * direction * ux
        gradient[i, 1] += factor * direction * uy
        gradient[j, 0] -= factor * direction * ux
        gradient[j, 1] -= factor * direction * uy
        for index, a, b, c, s in ((i, ai, bi, cosines[i], sines[i]),
                                  (j, aj, bj, cosines[j], sines[j])):
            direct = square_size * (np.sign(a) * (-s * ux + c * uy) -
                                    np.sign(b) * (c * ux + s * uy)) / 2
            gradient[index, 2] += factor * direct
        if kind == 0:
            dux, duy = -sines[owner], cosines[owner]
        else:
            dux, duy = -cosines[owner], -sines[owner]
        support_du = square_size * (
            np.sign(ai) * (cosines[i] * dux + sines[i] * duy) +
            np.sign(bi) * (-sines[i] * dux + cosines[i] * duy) +
            np.sign(aj) * (cosines[j] * dux + sines[j] * duy) +
            np.sign(bj) * (-sines[j] * dux + cosines[j] * duy)
        ) / 2
        gradient[owner, 2] += factor * (support_du -
                                        direction * (dx * dux + dy * duy))
    return energy, gradient.ravel()


def soften(packing: list[Placement], target: float, rng: random.Random,
           jitter: float, angle_jitter: float, trust: float,
           iterations: int, mode: int = 0,
           inflate: bool = False) -> tuple[list[Placement], float]:
    """Move a complete packing through a small-overlap state in a fixed box."""
    side = strict_side(packing)
    factor = target / side
    proposal = []
    centre = side / 2
    bend = .065 if side < 16 else .12
    for x, y, angle in packing:
        tilted = abs(math.remainder(angle, math.pi / 2)) > .01
        shift_x = shift_y = shift_angle = 0.0
        if tilted and mode == 1:
            shift_x = bend * (y - centre) / side
            shift_angle = bend / side
        elif tilted and mode == 2:
            position = (x + y) / (2 * side)
            wave = math.sin(2 * math.pi * position)
            shift_x, shift_y = -bend * wave, bend * wave
            shift_angle = bend * math.cos(2 * math.pi * position) / side
        elif tilted and mode == 3:
            sign = 1 if x + y > side else -1
            shift_x = sign * bend / 2
            shift_y = -sign * bend / 2
            shift_angle = sign * bend / side
        proposal.append((x * factor + shift_x +
                         rng.gauss(0, jitter if tilted else jitter / 4),
                         y * factor + shift_y +
                         rng.gauss(0, jitter if tilted else jitter / 4),
                         angle + shift_angle + rng.gauss(0, angle_jitter)
                         if tilted else angle))
    reference = np.asarray(proposal, dtype=float)
    vector = reference.ravel().copy()
    pairs = pair_indices(proposal, trust)
    bounds = []
    for x, y, angle in proposal:
        bounds.extend(((x - trust, x + trust), (y - trust, y + trust)))
        if abs(math.remainder(angle, math.pi / 2)) > .01:
            bounds.append((angle - .08, angle + .08))
        else:
            bounds.append((angle, angle))
    result = None
    stages = ((.94, 20.0), (.97, 80.0), (.99, 300.0),
              (1.0, 3000.0)) if inflate else ((1.0, 30.0),
                                               (1.0, 300.0),
                                               (1.0, 3000.0))
    for square_size, stiffness in stages:
        stage_reference = reference.copy()
        if inflate:
            stage_reference[:, :2] = (target / 2 +
                                      .94 * (reference[:, :2] - target / 2))
        result = minimize(soft_energy_gradient, vector, jac=True,
                          args=(stage_reference, pairs, target, stiffness,
                                .08 if inflate else .03, 2e-7, square_size),
                          method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": iterations, "ftol": 1e-10})
        vector = result.x
    softened = [tuple(map(float, row)) for row in vector.reshape(-1, 3)]
    return softened, float(result.fun)


def search_one(packing: list[Placement], trials: int, seed: int,
               reductions: tuple[float, ...], maximum: int,
               milp_seconds: float, iterations: int,
               inflate: bool = False) -> tuple[list[Placement], dict]:
    base = normalized(packing)
    initial = strict_side(base)
    rebuilt = rebuild_staircases(base, maximum, milp_seconds)
    if rebuilt is None:
        raise ValueError("failed to reproduce the full staircase packing")
    rng = random.Random(seed)
    best, best_side = base, initial
    stats = {"baseline_rebuilt": True, "rebuilt_side": strict_side(rebuilt),
             "soft_trials": 0, "lp_repaired": 0,
             "strict_candidates": 0, "best_soft_energy": None,
             "improvements": []}
    for trial in range(trials):
        target = best_side - reductions[trial % len(reductions)]
        jitter = (0.0, .005, .02, .04)[trial % 4]
        angle_jitter = (0.0, .003, .01, .02)[trial % 4]
        softened, energy = soften(rebuilt if trial == 0 else best, target,
                                  rng, jitter, angle_jitter,
                                  trust=.35 if inflate else .12,
                                  iterations=iterations, mode=trial % 4,
                                  inflate=inflate)
        stats["soft_trials"] += 1
        if stats["best_soft_energy"] is None or energy < stats["best_soft_energy"]:
            stats["best_soft_energy"] = energy
        repaired = restore_lp(softened, target + 1e-8, trust=.08,
                              rng=rng, time_limit=5.0)
        if repaired is None:
            repaired = restore_lp(softened, target + 1e-8, trust=.08,
                                  rng=rng, tie=1e-4, time_limit=5.0)
        if repaired is None:
            continue
        stats["lp_repaired"] += 1
        try:
            side = strict_side(repaired)
        except ValueError:
            continue
        stats["strict_candidates"] += 1
        if side < best_side - 1e-8:
            best, best_side = repaired, side
            stats["improvements"].append({"trial": trial, "side": side,
                                           "soft_energy": energy})
    stats.update(before_side=initial, after_side=best_side,
                 score_gain=(initial - best_side) / math.sqrt(len(packing)))
    return best, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n", nargs="+", type=int)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "submission_improved.csv")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--maximum", type=int, default=640)
    parser.add_argument("--milp-seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--inflate", action="store_true",
                        help="expand temporarily smaller squares under inward pressure")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "staircase_soft_candidates.csv")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "reports" / "staircase_soft_pilot.json")
    args = parser.parse_args()
    if (not all(1 <= n <= 1000 and proven_optimal(n) is None for n in args.n)
            or args.trials < 1 or args.iterations < 1 or
            args.maximum < 1 or args.milp_seconds <= 0):
        parser.error("use unlocked n and positive search parameters")
    layouts = load_selected(args.input, set(args.n))
    accepted, records = {}, []
    for n in sorted(layouts):
        start = time.monotonic()
        candidate, record = search_one(layouts[n], args.trials,
                                       args.seed + n,
                                       (1e-4, .001, .005, .02),
                                       args.maximum, args.milp_seconds,
                                       args.iterations, args.inflate)
        record.update(n=n, seconds=time.monotonic() - start,
                      accepted=record["score_gain"] > 1e-8)
        records.append(record)
        if record["accepted"]:
            accepted[n] = candidate
        print(f"n={n}: {record['before_side']:.12f} -> "
              f"{record['after_side']:.12f}, LP={record['lp_repaired']}, "
              f"time={record['seconds']:.1f}s", flush=True)
    if accepted:
        write_partial(args.output, accepted)
    report = {"input": str(args.input), "trials": args.trials,
              "iterations": args.iterations, "maximum": args.maximum,
              "milp_seconds": args.milp_seconds, "seed": args.seed,
              "inflate": args.inflate,
              "accepted_n": sorted(accepted),
              "candidate_csv": str(args.output) if accepted else None,
              "records": records}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
