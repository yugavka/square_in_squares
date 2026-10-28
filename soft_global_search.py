"""Global soft packing search in a fixed box: diverse seeds or basin hopping.

The energy is the squared separating-axis penetration of every nearby pair plus
a wall penalty, computed with NumPy for all pairs at once. Mode ``seeds``
starts from random, mixed-angle, diagonal-band and tilted-block layouts and
inflates the squares from size 0.9 to 1 with decreasing random shakes. Mode
``hop`` scales the incumbent into the target box and runs Metropolis basin
hopping over cluster shakes, cluster rotations and single-square moves. A
configuration with near-zero energy goes to the fixed-angle LP repair and the
strict validator. This is an optional SciPy experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from optimal_locks import proven_optimal
from projected_search import (load_selected, normalized, restore_lp,
                              strict_side, write_partial)
from validate_submission import validate_submission

Placement = tuple[float, float, float]
ROOT = Path(__file__).resolve().parent


def energy_grad(vec: np.ndarray, pairs: np.ndarray, side: float, size: float,
                stiffness: float) -> tuple[float, np.ndarray]:
    """Penalty and gradient for squares of edge ``size`` in a ``side`` box."""
    points = vec.reshape(-1, 3)
    x, y, a = points[:, 0], points[:, 1], points[:, 2]
    grad = np.zeros_like(points)
    energy = 0.0
    cos, sin = np.cos(a), np.sin(a)
    reach = size / 2 * (np.abs(cos) + np.abs(sin))
    reach_da = size / 2 * (-np.sign(cos) * sin + np.sign(sin) * cos)
    for coordinate, dim in ((x, 0), (y, 1)):
        for excess, sign in ((reach - coordinate, -1.0), (coordinate + reach - side, 1.0)):
            hit = excess > 0
            energy += .5 * stiffness * float(np.sum(excess[hit] ** 2))
            grad[hit, dim] += sign * stiffness * excess[hit]
            grad[hit, 2] += stiffness * excess[hit] * reach_da[hit]
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        dx, dy = x[j] - x[i], y[j] - y[i]
        delta = a[j] - a[i]
        cd, sd = np.cos(delta), np.sin(delta)
        half = size / 2 * (np.abs(cd) + np.abs(sd))
        half_dd = size / 2 * (-np.sign(cd) * sd + np.sign(sd) * cd)
        # Axes i0, i1, j0, j1 and their derivatives by the owner's angle.
        ax = np.stack([cos[i], -sin[i], cos[j], -sin[j]], 1)
        ay = np.stack([sin[i], cos[i], sin[j], cos[j]], 1)
        dax = np.stack([-sin[i], -cos[i], -sin[j], -cos[j]], 1)
        day = np.stack([cos[i], -sin[i], cos[j], -sin[j]], 1)
        proj = dx[:, None] * ax + dy[:, None] * ay
        overlap = size / 2 + half[:, None] - np.abs(proj)
        axis = np.argmin(overlap, 1)
        rows = np.arange(len(axis))
        depth = overlap[rows, axis]
        hit = depth > 0
        if np.any(hit):
            rows, axis, depth = rows[hit], axis[hit], depth[hit]
            ii, jj = i[hit], j[hit]
            sign = np.sign(proj[rows, axis])
            weight = stiffness * depth
            energy += .5 * stiffness * float(np.sum(depth ** 2))
            gx = -sign * ax[rows, axis] * weight
            gy = -sign * ay[rows, axis] * weight
            turn = -sign * (dx[hit] * dax[rows, axis] + dy[hit] * day[rows, axis]) * weight
            own_i = axis < 2
            np.add.at(grad[:, 0], jj, gx)
            np.add.at(grad[:, 1], jj, gy)
            np.add.at(grad[:, 0], ii, -gx)
            np.add.at(grad[:, 1], ii, -gy)
            np.add.at(grad[:, 2], ii, -half_dd[hit] * weight + np.where(own_i, turn, 0.0))
            np.add.at(grad[:, 2], jj, half_dd[hit] * weight + np.where(own_i, 0.0, turn))
    return energy, grad.ravel()


def near_pairs(vec: np.ndarray, cutoff: float) -> np.ndarray:
    centres = vec.reshape(-1, 3)[:, :2]
    return np.array(sorted(cKDTree(centres).query_pairs(cutoff)),
                    dtype=int).reshape(-1, 2)


def minimise(vec: np.ndarray, side: float, size: float, stiffness: float,
             iterations: int) -> tuple[np.ndarray, float]:
    """L-BFGS on fresh neighbour pairs; returns the penalty without stiffness."""
    pairs = near_pairs(vec, size * 1.5 + .3)
    result = minimize(energy_grad, vec, jac=True,
                      args=(pairs, side, size, stiffness), method="L-BFGS-B",
                      options={"maxiter": iterations, "ftol": 1e-15, "gtol": 1e-12})
    return result.x, float(result.fun) / stiffness


def settle(vec: np.ndarray, side: float) -> tuple[np.ndarray, float]:
    energy = math.inf
    for stiffness in (1e3, 1e4, 1e5):
        vec, energy = minimise(vec, side, 1.0, stiffness, 1500)
    return vec, energy


def inflate(vec: np.ndarray, side: float, rng: random.Random, steps: int = 30,
            noise: float = .02) -> tuple[np.ndarray, float]:
    """Grow squares from 0.9 to 1 with shakes that fade to zero."""
    shaker = np.random.default_rng(rng.randrange(1 << 30))
    for step in range(steps + 1):
        vec, _ = minimise(vec, side, .9 + .1 * step / steps, 1e3, 600)
        if step < steps:
            amplitude = noise * (1 - step / steps)
            vec = vec + shaker.normal(0, amplitude, vec.shape)
    energy = math.inf
    for stiffness in (1e4, 1e5, 1e6):
        vec, energy = minimise(vec, side, 1.0, stiffness, 3000)
    return vec, energy


def seed_random(n: int, side: float, rng: random.Random) -> list[Placement]:
    return [(rng.uniform(.5, side - .5), rng.uniform(.5, side - .5),
             rng.uniform(0, math.pi / 2)) for _ in range(n)]


def seed_mixed(n: int, side: float, rng: random.Random) -> list[Placement]:
    return [(x, y, rng.choice((0.0, math.pi / 4)) + rng.gauss(0, .05))
            for x, y, _ in seed_random(n, side, rng)]


def grid_outside(side: float, keep) -> list[Placement]:
    count = int(side)
    scale = side / count
    return [((gx + .5) * scale, (gy + .5) * scale, 0.0)
            for gx in range(count) for gy in range(count)
            if keep(np.array([(gx + .5) * scale, (gy + .5) * scale]))]


def fill_to(points: list[Placement], n: int, side: float,
            rng: random.Random) -> list[Placement]:
    while len(points) < n:
        points.append((rng.uniform(.5, side - .5), rng.uniform(.5, side - .5),
                       rng.uniform(0, 1.5)))
    return points[:n]


def seed_band(n: int, side: float, rng: random.Random) -> list[Placement]:
    """Diagonal band of tilted squares; upright grid elsewhere."""
    theta = math.radians(rng.uniform(28, 45))
    width = rng.choice((2, 3, 4, 5, 6, 8))
    u = np.array([math.cos(theta), math.sin(theta)])
    v = np.array([-u[1], u[0]])
    centre = np.array([side / 2, side / 2]) + rng.uniform(-1.5, 1.5) * v
    band = []
    for row in range(width):
        for k in range(-int(side) * 2, int(side) * 2 + 1):
            p = centre + k * u + (row - (width - 1) / 2) * v
            if .3 < p[0] < side - .3 and .3 < p[1] < side - .3:
                band.append((float(p[0]), float(p[1]), theta))
    grid = grid_outside(side, lambda q: abs((q - centre) @ v) > width / 2 + .8)
    rng.shuffle(grid)
    return fill_to(band + grid, n, side, rng)


def seed_block(n: int, side: float, rng: random.Random) -> list[Placement]:
    """Tilted a x b block near the centre; upright grid elsewhere."""
    edge = max(2, int(side / 1.6))
    a, b = rng.randint(edge - 2, edge + 1), rng.randint(edge - 2, edge + 1)
    theta = math.radians(rng.uniform(35, 45))
    centre = np.array([side / 2 + rng.uniform(-1, 1), side / 2 + rng.uniform(-1, 1)])
    u = np.array([math.cos(theta), math.sin(theta)])
    v = np.array([-u[1], u[0]])
    block = [tuple(map(float, centre + (i - (a - 1) / 2) * u + (j - (b - 1) / 2) * v)) + (theta,)
             for i in range(a) for j in range(b)]
    grid = grid_outside(side, lambda q: abs((q - centre) @ u) > a / 2 + .6
                        or abs((q - centre) @ v) > b / 2 + .6)
    rng.shuffle(grid)
    return fill_to(block + grid, n, side, rng)


SEEDS = {"random": seed_random, "mixed": seed_mixed, "band": seed_band,
         "block": seed_block}


def hop_move(vec: np.ndarray, side: float, rng: random.Random) -> np.ndarray:
    """Shake or rotate the squares near a random point, or move one square."""
    points = vec.reshape(-1, 3).copy()
    centre = np.array([rng.uniform(0, side), rng.uniform(0, side)])
    near = np.where(np.hypot(*(points[:, :2] - centre).T) < rng.uniform(1.0, 3.5))[0]
    kind = rng.random()
    if kind < .5:
        amplitude = rng.uniform(.1, .4)
        for index in near:
            points[index] += (rng.gauss(0, amplitude), rng.gauss(0, amplitude),
                              rng.gauss(0, amplitude))
    elif kind < .75:
        angle = rng.gauss(0, .15)
        c, s = math.cos(angle), math.sin(angle)
        offset = points[near, :2] - centre
        points[near, :2] = centre + offset @ np.array([[c, s], [-s, c]])
        points[near, 2] += angle
    else:
        points[rng.randrange(len(points))] = (
            rng.uniform(.5, side - .5), rng.uniform(.5, side - .5),
            rng.choice((0.0, math.pi / 4, rng.uniform(0, 1.5))))
    return points.ravel()


def repair(vec: np.ndarray, side: float, energy: float, threshold: float,
           rng: random.Random) -> tuple[float | None, list[Placement]]:
    layout = [tuple(map(float, row)) for row in vec.reshape(-1, 3)]
    if energy >= threshold:
        return None, layout
    repaired = restore_lp(layout, side + 1e-8, trust=.05, rng=rng, time_limit=30.0)
    if repaired is None:
        return None, layout
    try:
        return strict_side(repaired), repaired
    except ValueError:
        return None, layout


def run_seed(task: tuple) -> dict:
    n, side, kind, seed = task
    rng = random.Random(seed)
    start = np.asarray(SEEDS[kind](n, side, rng), dtype=float).ravel()
    vec, energy = inflate(start, side, rng)
    found, layout = repair(vec, side, energy, 1e-5, rng)
    return {"kind": kind, "seed": seed, "energy": energy, "side": found,
            "layout": layout if found is not None else None}


def run_hop(task: tuple) -> dict:
    incumbent, side, seed, steps, temperature = task
    rng = random.Random(seed)
    scale = side / strict_side(incumbent)
    vec = np.array([(x * scale, y * scale, a) for x, y, a in incumbent], float).ravel()
    vec, energy = settle(vec, side)
    start_energy = energy
    best_energy, best_vec = energy, vec
    for _ in range(steps):
        candidate, candidate_energy = settle(hop_move(vec, side, rng), side)
        if (candidate_energy < energy or
                rng.random() < math.exp(-(candidate_energy - energy) / temperature)):
            vec, energy = candidate, candidate_energy
            if energy < best_energy:
                best_energy, best_vec = energy, vec
        if best_energy < 1e-10:
            break
    found, layout = repair(best_vec, side, best_energy, 1e-8, rng)
    return {"kind": "hop", "seed": seed, "start_energy": start_energy,
            "energy": best_energy, "side": found,
            "layout": layout if found is not None else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n", type=int)
    parser.add_argument("target", type=float, help="box side to pack into")
    parser.add_argument("--mode", choices=("seeds", "hop"), default="hop")
    parser.add_argument("--kinds", nargs="+", choices=sorted(SEEDS), default=sorted(SEEDS))
    parser.add_argument("--count", type=int, default=12,
                        help="seeds per kind, or basin-hopping chains")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=2e-6)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--input", type=Path, default=ROOT / "submission_improved.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "soft_global_candidates.csv")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "reports" / "soft_global.json")
    args = parser.parse_args()
    if proven_optimal(args.n) is not None or not 1 < args.n <= 1000 or args.count < 1:
        parser.error("choose an unlocked n and a positive count")
    incumbent_side = {r.n: r.side for r in validate_submission(args.input)}[args.n]
    incumbent = normalized(load_selected(args.input, {args.n})[args.n])
    if args.mode == "seeds":
        tasks = [(args.n, args.target, kind, seed)
                 for seed in range(args.count) for kind in args.kinds]
        worker = run_seed
    else:
        tasks = [(incumbent, args.target, seed, args.steps, args.temperature)
                 for seed in range(args.count)]
        worker = run_hop
    started = time.monotonic()
    records, best = [], None
    with Pool(args.workers) as pool:
        for record in pool.imap_unordered(worker, tasks):
            layout = record.pop("layout")
            records.append(record)
            print(f"  {record['kind']} {record['seed']}: energy "
                  f"{record['energy']:.3e} side {record['side']}", flush=True)
            if record["side"] is not None and (best is None or record["side"] < best[0]):
                best = (record["side"], layout)
    records.sort(key=lambda r: (r["kind"], r["seed"]))
    report = {"n": args.n, "incumbent_side": incumbent_side, "target": args.target,
              "mode": args.mode, "count": args.count, "steps": args.steps,
              "temperature": args.temperature,
              "seconds": round(time.monotonic() - started),
              "best_side": None if best is None else best[0], "runs": records}
    if best is not None and best[0] < incumbent_side - 1e-9:
        write_partial(args.output, {args.n: best[1]})
        report["candidate_csv"] = str(args.output)
        report["score_gain"] = (incumbent_side - best[0]) / math.sqrt(args.n)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"incumbent {incumbent_side:.9f}, best {report['best_side']}; report {args.report}")


if __name__ == "__main__":
    main()
