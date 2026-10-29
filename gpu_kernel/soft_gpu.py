"""Batched soft packing on a GPU: many seeds inflate at once in a fixed box.

This file runs as a Kaggle script kernel. gpu_soft_search.py replaces CONFIG
with one job per n (box sides and incumbent layout) and the time budget, which
the jobs share round-robin. Each batch
inflates squares from a smaller size to 1 under a separating-axis overlap
penalty with fading random shakes, then anneals the step size. The best layouts
of every batch are written to candidates.json for local L-BFGS polish, LP
repair and strict validation; summary.json records the energy statistics of every batch.
"""

import json
import math
import time

import numpy as np
import torch

CONFIG = {"jobs": [{"n": 3, "targets": [2.0], "incumbent": [], "incumbent_side": 1.0}],
          "minutes": 1.0, "batch": 64, "seed": 0, "kinds": ["random"],
          "keep": 1e-4, "top": 8}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def energy(points, side, size, i, j):
    """Per-layout penalty: squared wall excess plus squared SAT penetration."""
    x, y, a = points.unbind(-1)
    cos, sin = torch.cos(a), torch.sin(a)
    reach = size / 2 * (cos.abs() + sin.abs())
    wall = (torch.relu(reach - x) ** 2 + torch.relu(x + reach - side) ** 2 +
            torch.relu(reach - y) ** 2 + torch.relu(y + reach - side) ** 2).sum(-1)
    dx, dy = x[:, j] - x[:, i], y[:, j] - y[:, i]
    delta = a[:, j] - a[:, i]
    half = size / 2 * (torch.cos(delta).abs() + torch.sin(delta).abs())
    ci, si, cj, sj = cos[:, i], sin[:, i], cos[:, j], sin[:, j]
    proj = torch.stack([dx * ci + dy * si, -dx * si + dy * ci,
                        dx * cj + dy * sj, -dx * sj + dy * cj], -1).abs()
    depth = torch.relu((size / 2 + half).unsqueeze(-1) - proj).amin(-1)
    return .5 * (wall + (depth ** 2).sum(-1))


def grid_outside(side, keep):
    count = int(side)
    scale = side / count
    return [((gx + .5) * scale, (gy + .5) * scale, 0.0)
            for gx in range(count) for gy in range(count)
            if keep(np.array([(gx + .5) * scale, (gy + .5) * scale]))]


def fill_to(points, n, side, rng):
    while len(points) < n:
        points.append((rng.uniform(.5, side - .5), rng.uniform(.5, side - .5),
                       rng.uniform(0, 1.5)))
    order = rng.permutation(len(points))[:n]
    return [points[k] for k in sorted(order)]


def seed(kind, job, side, rng):
    n = job["n"]
    if kind == "random":
        return [(rng.uniform(.5, side - .5), rng.uniform(.5, side - .5),
                 rng.uniform(0, math.pi / 2)) for _ in range(n)]
    if kind == "mixed":
        return [(rng.uniform(.5, side - .5), rng.uniform(.5, side - .5),
                 rng.choice((0.0, math.pi / 4)) + rng.normal(0, .05)) for _ in range(n)]
    if kind == "incumbent":
        scale = side / job["incumbent_side"]
        sigma = rng.uniform(.05, .4)
        return [(x * scale + rng.normal(0, sigma), y * scale + rng.normal(0, sigma),
                 a + rng.normal(0, sigma)) for x, y, a in job["incumbent"]]
    theta = math.radians(rng.uniform(25, 45))
    u = np.array([math.cos(theta), math.sin(theta)])
    v = np.array([-u[1], u[0]])
    if kind == "band":
        width = int(rng.choice((2, 3, 4, 5, 6, 8)))
        centre = np.array([side / 2, side / 2]) + rng.uniform(-2, 2) * v
        tilted = []
        for row in range(width):
            for k in range(-2 * int(side), 2 * int(side) + 1):
                p = centre + (k + rng.uniform(0, 1) * (row % 2)) * u + (row - (width - 1) / 2) * v
                if .3 < p[0] < side - .3 and .3 < p[1] < side - .3:
                    tilted.append((float(p[0]), float(p[1]), theta))
        grid = grid_outside(side, lambda q: abs((q - centre) @ v) > width / 2 + .8)
        return fill_to(tilted + grid, n, side, rng)
    edge = max(2, int(side / 1.6))
    a_, b_ = int(rng.integers(edge - 3, edge + 2)), int(rng.integers(edge - 3, edge + 2))
    centre = np.array([side / 2 + rng.uniform(-1.5, 1.5), side / 2 + rng.uniform(-1.5, 1.5)])
    tilted = [tuple(map(float, centre + (p - (a_ - 1) / 2) * u + (q - (b_ - 1) / 2) * v)) + (theta,)
              for p in range(a_) for q in range(b_)]
    grid = grid_outside(side, lambda q: abs((q - centre) @ u) > a_ / 2 + .6
                        or abs((q - centre) @ v) > b_ / 2 + .6)
    return fill_to(tilted + grid, n, side, rng)


def run_batch(kind, job, side, rng, i, j):
    batch = CONFIG["batch"]
    # float32: consumer GPUs such as the T4 run float64 about 30 times slower,
    # and the LP repair after download removes the remaining residual.
    start = np.array([seed(kind, job, side, rng) for _ in range(batch)], dtype=np.float32)
    points = torch.tensor(start, device=DEVICE, requires_grad=True)
    optimiser = torch.optim.Adam([points], lr=2e-2)
    stages = [(.6 + .4 * k / 40, 2e-2 * (1 - k / 40) + 2e-3) for k in range(41)]
    for size, lr in stages:
        for group in optimiser.param_groups:
            group["lr"] = lr
        for _ in range(60):
            optimiser.zero_grad()
            energy(points, side, size, i, j).sum().backward()
            optimiser.step()
        with torch.no_grad():
            amplitude = .03 * (1 - size) / .4
            points[..., :2] += amplitude * torch.randn_like(points[..., :2])
            points[..., 2] += amplitude * torch.randn_like(points[..., 2])
    for lr in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
        for group in optimiser.param_groups:
            group["lr"] = lr
        for _ in range(1500):
            optimiser.zero_grad()
            energy(points, side, 1.0, i, j).sum().backward()
            optimiser.step()
    with torch.no_grad():
        final = energy(points, side, 1.0, i, j).cpu().numpy()
    return points.detach().cpu().numpy(), final


def main():
    started = time.time()
    name = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
    print(json.dumps({"device": DEVICE, "name": name}), flush=True)
    rng = np.random.default_rng(CONFIG["seed"])
    pairs = {job["n"]: torch.triu_indices(job["n"], job["n"], 1, device=DEVICE)
             for job in CONFIG["jobs"]}
    tasks = [(job, side, kind) for job in CONFIG["jobs"]
             for side in job["targets"] for kind in CONFIG["kinds"]]
    candidates, summary = [], []
    round_index = 0
    while time.time() - started < CONFIG["minutes"] * 60:
        for job, side, kind in tasks:
            if time.time() - started >= CONFIG["minutes"] * 60:
                break
            n = job["n"]
            layouts, final = run_batch(kind, job, side, rng, *pairs[n])
            order = np.argsort(final)
            summary.append({"round": round_index, "n": n, "side": side, "kind": kind,
                            "min": float(final[order[0]]),
                            "median": float(np.median(final)),
                            "below_keep": int(np.sum(final < CONFIG["keep"])),
                            "seconds": round(time.time() - started)})
            print(json.dumps(summary[-1]), flush=True)
            # The best few of every batch are polished locally with L-BFGS.
            for k in order[:CONFIG["top"]]:
                candidates.append({"n": n, "side": side, "kind": kind,
                                   "energy": float(final[k]),
                                   "layout": layouts[k].tolist()})
        round_index += 1
        candidates.sort(key=lambda c: (c["n"], c["side"], c["energy"]))
        json.dump(candidates, open("candidates.json", "w"))
        config = {k: v for k, v in CONFIG.items() if k != "jobs"}
        config["jobs"] = [{k: v for k, v in job.items() if k != "incumbent"}
                          for job in CONFIG["jobs"]]
        json.dump({"device": DEVICE, "name": name, "config": config,
                   "batches": summary}, open("summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
