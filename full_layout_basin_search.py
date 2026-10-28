"""Basin hopping on complete, strictly checked square packings.

An archive keeps geometrically distinct local minima, including a few that are
worse than the incumbent. Collective perturbations may overlap temporarily;
the existing LP repair and free-angle sequential LP quench make them legal
again. Only a strict full-n improvement is written as a portfolio candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from scipy.spatial import cKDTree

from joint_polish import polish
from optimal_locks import proven_optimal
from projected_search import (load_selected, normalized, restore_lp,
                              strict_side, write_partial)
from squeeze_layout import separating_axis
from validate_submission import total_score, validate_submission


Placement = tuple[float, float, float]
ROOT = Path(__file__).resolve().parent


@dataclass
class State:
    squares: list[Placement]
    side: float
    contacts: frozenset[tuple[int, int]]


def contact_graph(squares: list[Placement], tolerance: float = 0.006
                  ) -> frozenset[tuple[int, int]]:
    """Record near contacts to distinguish the basins found by the search."""
    points = [(x, y) for x, y, _ in squares]
    pairs = cKDTree(points).query_pairs(math.sqrt(2) + tolerance + 1e-6)
    return frozenset((i, j) for i, j in pairs
                     if separating_axis(squares[i], squares[j])[0] < tolerance)


def collective_mutation(squares: list[Placement], rng: random.Random,
                        size: float, mode: int) -> list[Placement]:
    """Move a cluster, a boundary band, or the whole layout across contacts."""
    result = list(squares)
    n = len(result)
    if mode == 0 or mode == 3:
        pivot = rng.randrange(n)
        count = min(n, rng.randint(min(n, 7), min(n, 26)))
        selected = sorted(range(n), key=lambda i:
                          (result[i][0] - result[pivot][0]) ** 2 +
                          (result[i][1] - result[pivot][1]) ** 2)[:count]
        px = sum(result[i][0] for i in selected) / count
        py = sum(result[i][1] for i in selected) / count
        turn = rng.uniform(-size, size) * (0.45 if mode == 0 else 0.9)
        dx, dy = rng.uniform(-size, size), rng.uniform(-size, size)
        c, s = math.cos(turn), math.sin(turn)
        for i in selected:
            x, y, angle = result[i]
            result[i] = (px + c * (x - px) - s * (y - py) + dx,
                         py + s * (x - px) + c * (y - py) + dy,
                         angle + turn)
    elif mode == 1:
        side = strict_side(result)
        wall = rng.randrange(4)
        bandwidth = rng.uniform(1.5, 3.0)
        dx, dy = rng.uniform(-size, size), rng.uniform(-size, size)
        for i, (x, y, angle) in enumerate(result):
            distance = (x if wall == 0 else side - x if wall == 1
                        else y if wall == 2 else side - y)
            weight = max(0.0, 1.0 - distance / bandwidth)
            result[i] = x + weight * dx, y + weight * dy, angle
    elif mode == 2:
        side = strict_side(result)
        phase_x, phase_y = rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)
        for i, (x, y, angle) in enumerate(result):
            result[i] = (x + size * math.sin(2 * math.pi * y / side + phase_x),
                         y + size * math.sin(2 * math.pi * x / side + phase_y),
                         angle + size * 0.2 * math.sin(
                             2 * math.pi * (x + y) / side + phase_x))
    else:
        raise ValueError("unknown mutation mode")
    return result


def add_to_archive(archive: list[State], candidate: State,
                   maximum: int, ceiling: float) -> bool:
    """Keep the best representative of each near-contact topology."""
    if candidate.side > ceiling:
        return False
    for i, state in enumerate(archive):
        # A few contacts near the threshold flicker under tiny LP movements.
        # They must not fill the archive with copies of the same basin.
        minimum_change = max(4, len(state.contacts) // 20)
        if len(state.contacts ^ candidate.contacts) < minimum_change:
            if candidate.side >= state.side - 1e-8:
                return False
            archive[i] = candidate
            archive.sort(key=lambda row: row.side)
            return True
    archive.append(candidate)
    archive.sort(key=lambda row: row.side)
    if len(archive) > maximum:
        archive.pop()
    return candidate in archive


def search_one(base: list[Placement], trials: int, seed: int,
               quench_rounds: int = 6, trust: float = 0.22,
               allowance: float = 0.06, archive_size: int = 8,
               lp_seconds: float = 2.0) -> tuple[list[Placement], dict]:
    """Explore multiple full-layout basins and return the best legal packing."""
    if trials < 1 or quench_rounds < 0 or trust <= 0 or allowance <= 0:
        raise ValueError("invalid search budget or geometry limits")
    rng = random.Random(seed)
    initial_side = strict_side(base)
    initial = normalized(base)
    best = State(initial, initial_side, contact_graph(initial))
    archive = [best]
    walk = best
    size = 0.035
    temperature = allowance / 4
    stats: dict = {"initial_side": initial_side, "final_side": initial_side,
                   "repaired": 0, "quenched": 0, "new_topologies": 0,
                   "uphill_moves": 0, "best_updates": [], "attempts": trials}
    for trial in range(trials):
        if trial and trial % 8 == 0:
            walk = rng.choice(archive)
        anchor = walk if trial % 4 else rng.choice(archive)
        proposal = collective_mutation(anchor.squares, rng, size, trial % 4)
        repaired = restore_lp(proposal, anchor.side + allowance, trust, rng,
                              tie=0.01 if trial % 3 == 2 else 0.0,
                              time_limit=lp_seconds)
        if repaired is None:
            size = max(0.012, size * 0.85)
            continue
        stats["repaired"] += 1
        if quench_rounds:
            quenched, _ = polish(repaired, quench_rounds, strict_side,
                                 normalized, initial_step=min(0.025, size))
        else:
            quenched = repaired
        stats["quenched"] += 1
        side = strict_side(quenched)
        candidate = State(quenched, side, contact_graph(quenched))
        novel = all(len(state.contacts ^ candidate.contacts) >=
                    max(4, len(state.contacts) // 20) for state in archive)
        if add_to_archive(archive, candidate, archive_size,
                          initial_side + allowance):
            if novel:
                stats["new_topologies"] += 1
            size = min(0.11, size * 1.18)
        else:
            size = max(0.012, size * 0.9)
        if side < best.side - 1e-8:
            best = candidate
            stats["best_updates"].append({"trial": trial, "side": side})
        difference = side - walk.side
        if difference <= 0 or rng.random() < math.exp(-difference / temperature):
            if difference > 1e-8:
                stats["uphill_moves"] += 1
            walk = candidate
    stats["final_side"] = best.side
    stats["score_gain"] = (initial_side - best.side) / math.sqrt(len(base))
    stats["archive_size"] = len(archive)
    stats["archive_sides"] = [state.side for state in archive]
    return best.squares, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n", nargs="+", type=int)
    parser.add_argument("--input", type=Path, default=ROOT / "submission_improved.csv")
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--quench-rounds", type=int, default=6)
    parser.add_argument("--trust", type=float, default=0.22)
    parser.add_argument("--allowance", type=float, default=0.06)
    parser.add_argument("--archive-size", type=int, default=8)
    parser.add_argument("--lp-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "full_basin_candidates.csv")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "reports" / "full_basin_pilot.json")
    args = parser.parse_args()
    selected = set(args.n)
    if (not selected or not all(1 <= n <= 1000 and proven_optimal(n) is None
                                for n in selected) or args.trials < 1 or
            args.quench_rounds < 0 or args.trust <= 0 or
            args.allowance <= 0 or args.archive_size < 2 or
            args.lp_seconds <= 0 or
            args.output.resolve() == args.input.resolve()):
        parser.error("choose unlocked n, valid search settings, distinct output")
    baseline_score = total_score(validate_submission(args.input))
    layouts = load_selected(args.input, selected)
    improved = {}
    records = []
    for n in sorted(selected):
        started = time.monotonic()
        candidate, record = search_one(
            layouts[n], args.trials, args.seed + n, args.quench_rounds,
            args.trust, args.allowance, args.archive_size, args.lp_seconds)
        if record["initial_side"] - record["final_side"] > 1e-8:
            improved[n] = candidate
        record.update(n=n, accepted=n in improved,
                      seconds=time.monotonic() - started)
        records.append(record)
        print(f"n={n}: side {record['initial_side']:.12f} -> "
              f"{record['final_side']:.12f}; repaired={record['repaired']}, "
              f"new topologies={record['new_topologies']}", flush=True)
    if improved:
        write_partial(args.output, improved)
    report = {"input": str(args.input), "baseline_score": baseline_score,
              "estimated_score": baseline_score - sum(
                  row["score_gain"] for row in records if row["accepted"]),
              "seed": args.seed, "trials": args.trials,
              "quench_rounds": args.quench_rounds, "trust": args.trust,
              "allowance": args.allowance, "archive_size": args.archive_size,
              "accepted_n": sorted(improved),
              "candidate_csv": str(args.output) if improved else None,
              "records": records}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Accepted {len(improved)} layouts; estimated score "
          f"{report['estimated_score']:.12f}; report {args.report}")


if __name__ == "__main__":
    main()
