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


def energy(points, side, size, pairs):
    """Per-layout penalty: squared wall excess plus squared SAT penetration.

    ``size`` is one edge length or a (layout, square) tensor of them; ``pairs``
    holds flat (layout, i, j) index tensors of nearby squares.
    """
    x, y, a = points.unbind(-1)
    size = torch.as_tensor(size, dtype=points.dtype, device=points.device).expand_as(x)
    cos, sin = torch.cos(a), torch.sin(a)
    reach = size / 2 * (cos.abs() + sin.abs())
    total = (torch.relu(reach - x) ** 2 + torch.relu(x + reach - side) ** 2 +
             torch.relu(reach - y) ** 2 + torch.relu(y + reach - side) ** 2).sum(-1)
    b, i, j = pairs
    dx, dy = x[b, j] - x[b, i], y[b, j] - y[b, i]
    delta = a[b, j] - a[b, i]
    spread = torch.cos(delta).abs() + torch.sin(delta).abs()
    size_i, size_j = size[b, i] / 2, size[b, j] / 2
    # On an axis of square i its own half width is size_i/2; square j
    # projects to size_j/2 * spread, and symmetrically for j's axes.
    on_i, on_j = size_i + size_j * spread, size_j + size_i * spread
    ci, si, cj, sj = cos[b, i], sin[b, i], cos[b, j], sin[b, j]
    proj = torch.stack([dx * ci + dy * si, -dx * si + dy * ci,
                        dx * cj + dy * sj, -dx * sj + dy * cj], -1).abs()
    limit = torch.stack([on_i, on_i, on_j, on_j], -1)
    depth = torch.relu(limit - proj).amin(-1)
    total = total.index_add(0, b, depth ** 2)
    return .5 * total


def neighbours(points, cutoff):
    """Flat (layout, i, j) indices of square pairs closer than ``cutoff``."""
    with torch.no_grad():
        centres = points[..., :2]
        close = torch.cdist(centres, centres) < cutoff
        close = torch.triu(close, diagonal=1)
        b, i, j = close.nonzero(as_tuple=True)
    return b, i, j


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


def destroy(job, side, rng):
    """Incumbent with every square in a random window thrown in at random."""
    scale = side / job["incumbent_side"]
    width = rng.uniform(2.5, min(6.0, side))
    x0, y0 = rng.uniform(0, side - width), rng.uniform(0, side - width)
    points, inside = [], []
    for x, y, a in job["incumbent"]:
        x, y = x * scale, y * scale
        hit = x0 <= x <= x0 + width and y0 <= y <= y0 + width
        if hit:
            x, y, a = (rng.uniform(x0, x0 + width), rng.uniform(y0, y0 + width),
                       rng.uniform(0, math.pi / 2))
        points.append((x, y, a))
        inside.append(hit)
    return points, inside


def descend(points, optimiser, side, size, lr, steps, refresh, margin):
    for group in optimiser.param_groups:
        group["lr"] = lr
    for step in range(steps):
        if step % refresh == 0:
            # Squares move at most about lr per step under Adam, so pairs
            # beyond the margin cannot meet before the next refresh.
            pairs = neighbours(points, float(torch.as_tensor(size).max()) * math.sqrt(2) + margin)
        optimiser.zero_grad()
        energy(points, side, size, pairs).sum().backward()
        optimiser.step()


def run_batch(kind, job, side, rng):
    batch = CONFIG["batch"]
    # float32: consumer GPUs such as the T4 run float64 about 30 times slower,
    # and the local L-BFGS polish after download removes the residual.
    if kind == "lns":
        built = [destroy(job, side, rng) for _ in range(batch)]
        start = np.array([b for b, _ in built], dtype=np.float32)
        inside = torch.tensor([m for _, m in built], device=DEVICE)
    else:
        start = np.array([seed(kind, job, side, rng) for _ in range(batch)], dtype=np.float32)
    points = torch.tensor(start, device=DEVICE, requires_grad=True)
    optimiser = torch.optim.Adam([points], lr=2e-2)
    if kind == "lns":
        # Thrown-in squares grow from half size and alone are shaken while
        # the rest of the incumbent stays at full size around them.
        low, rounds, lr0, shaken = .5, 30, 1e-2, inside.float()
    elif kind == "incumbent":
        # The incumbent keeps its structure: shake it at nearly full size.
        low, rounds, lr0, shaken = .97, 10, 2e-2, torch.ones_like(points[..., 0])
    else:
        # Other seeds organise while all squares grow from 0.6.
        low, rounds, lr0, shaken = .6, 40, 2e-2, torch.ones_like(points[..., 0])
    for k in range(rounds + 1):
        grow = low + (1 - low) * k / rounds
        size = torch.where(inside, grow, 1.0) if kind == "lns" else grow
        descend(points, optimiser, side, size, lr0 * (1 - k / rounds) + lr0 / 10, 60, 10, .6)
        with torch.no_grad():
            amplitude = .03 * (1 - k / rounds) * shaken
            for axis in range(3):
                points[..., axis] += amplitude * torch.randn_like(amplitude)
    for lr in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
        descend(points, optimiser, side, 1.0, lr, 1500, 50, .3)
    with torch.no_grad():
        final = energy(points, side, 1.0, neighbours(points, 1.5)).cpu().numpy()
    return points.detach().cpu().numpy(), final


def main():
    started = time.time()
    name = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
    print(json.dumps({"device": DEVICE, "name": name}), flush=True)
    rng = np.random.default_rng(CONFIG["seed"])
    tasks = [(job, side, kind) for job in CONFIG["jobs"]
             for side in job["targets"] for kind in CONFIG["kinds"]]
    candidates, summary = [], []
    round_index = 0
    while time.time() - started < CONFIG["minutes"] * 60:
        for job, side, kind in tasks:
            if time.time() - started >= CONFIG["minutes"] * 60:
                break
            n = job["n"]
            layouts, final = run_batch(kind, job, side, rng)
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
