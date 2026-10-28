"""Transfer an (n+1)-square layout to n by deletion and fixed-angle LP.

This optional search reads a validated full submission, removes promising
donor squares, recloses the survivors, and exports only strictly valid gains.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from joint_polish import polish
from optimal_locks import proven_optimal
from projected_search import (Placement, load_selected, normalized, restore_lp,
                              strict_side, write_partial)
from validate_submission import (make_square, measure_packing, total_score,
                                 validate_submission)


ROOT = Path(__file__).resolve().parent


def removal_order(donor: list[Placement], seed: int) -> list[int]:
    """Try removals that expose a box edge or open an isolated pocket first."""
    rng = random.Random(seed)
    squares = [make_square(i, *row) for i, row in enumerate(donor)]
    lo_x = min(s.min_x for s in squares)
    hi_x = max(s.max_x for s in squares)
    lo_y = min(s.min_y for s in squares)
    hi_y = max(s.max_y for s in squares)
    candidates = []
    for i, square in enumerate(squares):
        others = squares[:i] + squares[i + 1:]
        width = max(s.max_x for s in others) - min(s.min_x for s in others)
        height = max(s.max_y for s in others) - min(s.min_y for s in others)
        exposed = max(width, height)
        edge = min(square.min_x - lo_x, hi_x - square.max_x,
                   square.min_y - lo_y, hi_y - square.max_y)
        nearest = min(math.hypot(square.x - other.x, square.y - other.y)
                      for other in others)
        candidates.append((exposed, edge, -nearest, rng.random(), i))
    return [i for *_, i in sorted(candidates)]


def transfer_one(base: list[Placement], donor: list[Placement], seed: int,
                 attempts: int = 12, trust: float = 0.25,
                 joint_rounds: int = 12) -> tuple[list[Placement], dict]:
    if len(donor) != len(base) + 1:
        raise ValueError("donor must contain exactly one more square")
    rng = random.Random(seed)
    best = normalized(base)
    before = best_side = measure_packing(
        len(best), [make_square(i, *row) for i, row in enumerate(best)]).side
    donor_side = measure_packing(
        len(donor), [make_square(i, *row) for i, row in enumerate(donor)]).side
    order = removal_order(donor, seed)[:attempts]
    feasible = 0
    lp_gains = 0
    for index in order:
        reduced = normalized(donor[:index] + donor[index + 1:])
        try:
            side = strict_side(reduced)
        except ValueError:
            pass  # The donor may have a tiny contact allowed by the CSV validator.
        else:
            if side < best_side - 1e-8:
                best, best_side = reduced, side
        candidate = restore_lp(reduced, best_side + 1e-7, trust, rng,
                               tie=0.01 if index % 3 == 0 else 0.0,
                               time_limit=2.0)
        if candidate is None:
            continue
        feasible += 1
        side = strict_side(candidate)
        if side < best_side - 1e-8:
            best, best_side = candidate, side
            lp_gains += 1
    joint_gains = 0
    if best_side < before - 1e-8 and joint_rounds:
        polished, joint_gains = polish(best, joint_rounds, strict_side,
                                       normalized)
        side = strict_side(polished)
        if side < best_side - 1e-8:
            best, best_side = polished, side
    return best, {"before_side": before, "donor_side": donor_side,
                  "after_side": best_side,
                  "score_gain": (before - best_side) / math.sqrt(len(base)),
                  "deletions_tested": len(order), "lp_feasible": feasible,
                  "lp_improvements": lp_gains,
                  "joint_improvements": joint_gains}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n", nargs="+", type=int)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "submission_improved.csv")
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--trust", type=float, default=0.25)
    parser.add_argument("--joint-rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "neighbor_transfer_candidates.csv")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "reports" / "neighbor_transfer_lp.json")
    args = parser.parse_args()
    selected = set(args.n)
    if (not all(1 <= n < 1000 and proven_optimal(n) is None for n in selected)
            or args.attempts < 1 or not 0 < args.trust <= 0.5
            or args.joint_rounds < 0):
        parser.error("choose unlocked n in 1..999, attempts>0, "
                     "0<trust<=0.5, joint-rounds>=0")
    results = validate_submission(args.input)
    baseline_score = total_score(results)
    layouts = load_selected(args.input,
                            selected | {n + 1 for n in selected})
    accepted = {}
    records = []
    for n in sorted(selected):
        started = time.monotonic()
        candidate, record = transfer_one(layouts[n], layouts[n + 1],
                                         args.seed + n, args.attempts,
                                         args.trust, args.joint_rounds)
        if record["score_gain"] > 1e-8:
            accepted[n] = candidate
        record.update(n=n, accepted=n in accepted,
                      seconds=time.monotonic() - started)
        records.append(record)
        print(f"n={n}: gain={record['score_gain']:.12g}, "
              f"LP={record['lp_feasible']}/{record['deletions_tested']}",
              flush=True)
    if accepted:
        write_partial(args.output, accepted)
    report = {"input": str(args.input), "baseline_score": baseline_score,
              "candidate_score": baseline_score - sum(
                  row["score_gain"] for row in records if row["accepted"]),
              "seed": args.seed, "attempts": args.attempts,
              "trust": args.trust, "joint_rounds": args.joint_rounds,
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
