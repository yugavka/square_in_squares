"""Try complementary rectangular crops with independently checked seams.

Unlike the fault-line DP, both members of a pair may have occupied upper
tips. This tests candidate seams explicitly rather than requiring an empty tip
on every nonterminal fragment. Several crop geometries per signature can be
retained because the narrowest crop need not have the best joining seam.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from fault_line_dp import (Gadget, extract_corner_crops,
                           load_gadget_overrides, seam_compatible)
from optimal_locks import proven_optimal
from packing_solution import PUBLIC_PORTFOLIO, portfolio_packings, record_seeds
from projected_search import write_partial
from validate_submission import (ValidationError, make_square, measure_packing,
                                 squares_overlap, validate_submission)


ROOT = Path(__file__).resolve().parent


def select_public_donors(portfolio: dict[int, tuple[float, list]],
                         limit: int) -> list[tuple[float, list]]:
    """Sample the public portfolio across the whole range of puzzle sizes."""
    if limit <= 0:
        return []
    numbers = sorted(portfolio)
    selected = {numbers[round(i * (len(numbers) - 1) / max(limit - 1, 1))]
                for i in range(min(limit, len(numbers)))}
    return [portfolio[n] for n in sorted(selected)]


def shifted_seam_compatible(first: Gadget, second: Gadget, shift: float) -> bool:
    """Test collisions between the two fragments after moving the second NE."""
    tip = [make_square(i, *row) for i, row in enumerate(first.squares)]
    tip = [square for square in tip if square.max_x > first.w and
           square.max_y > first.h]
    if not tip:
        return True
    corner = [make_square(i, x + first.w + shift, y + first.h + shift, angle)
              for i, (x, y, angle) in enumerate(second.squares)]
    for a in tip:
        for b in corner:
            if (a.max_x > b.min_x and b.max_x > a.min_x and
                    a.max_y > b.min_y and b.max_y > a.min_y and
                    squares_overlap(a, b, tolerance=0)):
                return False
    return True


def compose_shifted(first: Gadget, second: Gadget,
                    shift: float) -> tuple[float, list]:
    """Fill both off-diagonal grids while leaving a shifted seam gap."""
    f = max(first.overhang, second.overhang)
    rows = list(first.squares)
    rows.extend((x + first.w + shift, y + first.h + shift, angle)
                for x, y, angle in second.squares)
    rows.extend((x + 0.5, first.h + f + y + 0.5, 0.0)
                for x in range(first.w) for y in range(second.h))
    rows.extend((first.w + f + x + 0.5, y + 0.5, 0.0)
                for x in range(second.w) for y in range(first.h))
    safe = [(x * (1 + 1e-8), y * (1 + 1e-8), angle)
            for x, y, angle in rows]
    side = measure_packing(len(safe), [make_square(i, *row)
                                      for i, row in enumerate(safe)]).side
    return side, safe


def search_pairs(gadgets: list[Gadget], incumbents: dict[int, float],
                 max_n: int = 1000) -> tuple[dict[int, tuple[float, list]], dict]:
    """Screen pairs by their best possible side, then validate joined geometry."""
    complements: dict[int, list[Gadget]] = defaultdict(list)
    for gadget in gadgets:
        complements[gadget.w - gadget.h].append(gadget)
    best: dict[int, tuple[float, list]] = {}
    records: dict[int, dict] = {}
    screened = compatible = composed = shifted = 0
    for first in gadgets:
        for second in complements[first.h - first.w]:
            k = first.w + second.w
            n = k * k + first.surplus + second.surplus
            if n > max_n or proven_optimal(n) is not None:
                continue
            limit = min(incumbents.get(n, float(math.isqrt(n - 1) + 1)),
                        best[n][0] if n in best else math.inf)
            if k + max(first.overhang, second.overhang) + 2e-7 >= limit:
                continue
            screened += 1
            shift = 0.0
            if not seam_compatible(first, second):
                margin = limit - k - max(first.overhang, second.overhang) - 2e-7
                if not shifted_seam_compatible(first, second, margin):
                    continue
                low, high = 0.0, margin
                for _ in range(20):
                    middle = (low + high) / 2
                    if shifted_seam_compatible(first, second, middle):
                        high = middle
                    else:
                        low = middle
                shift = high + 1e-8
                shifted += 1
            compatible += 1
            try:
                candidate = compose_shifted(first, second, shift)
            except ValidationError:
                continue
            composed += 1
            if candidate[0] >= limit - 1e-9:
                continue
            best[n] = candidate
            records[n] = {
                "before_side": incumbents.get(n, float(math.isqrt(n - 1) + 1)),
                "after_side": candidate[0],
                "score_gain": (incumbents.get(n, float(math.isqrt(n - 1) + 1))
                               - candidate[0]) / math.sqrt(n),
                "fragments": [[g.w, g.h, g.surplus, g.overhang]
                              for g in (first, second)],
                "seam_shift": shift,
            }
    report = {"gadgets": len(gadgets), "screened_pairs": screened,
              "compatible_pairs": compatible, "shifted_pairs": shifted,
              "validated_pairs": composed,
              "improvements": [dict(n=n, **records[n]) for n in sorted(records)]}
    return best, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "submission_improved.csv")
    parser.add_argument("--public-donors", type=int, default=24)
    parser.add_argument("--max-dim", type=int, default=20)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--polish-rounds", type=int, default=0,
                        help="optional full-packing centre/angle LP refinement")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "reports" / "rectangle_pair_pilot.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "rectangle_pair_candidates.csv")
    args = parser.parse_args()
    if args.polish_rounds < 0:
        parser.error("--polish-rounds must be nonnegative")
    incumbents = {row.n: row.side for row in validate_submission(args.input)}
    donors = record_seeds()
    portfolio = portfolio_packings([PUBLIC_PORTFOLIO], 1000)
    sampled = select_public_donors(portfolio, args.public_donors)
    donors.extend(sampled)
    gadgets = extract_corner_crops(donors, max_dim=args.max_dim,
                                   variants_per_signature=args.variants)
    gadgets.extend(load_gadget_overrides())
    improvements, report = search_pairs(gadgets, incumbents)
    if args.polish_rounds:
        from joint_polish import polish
        from projected_search import normalized, strict_side

        for item in report["improvements"]:
            n = item["n"]
            before, rows = improvements[n]
            polished, steps = polish(rows, args.polish_rounds, strict_side,
                                     normalized, initial_step=0.01)
            side = strict_side(polished)
            item["assembled_side"] = before
            item["polish_steps"] = steps
            if side < before - 1e-10:
                improvements[n] = side, polished
                item["after_side"] = side
                item["score_gain"] = (item["before_side"] - side) / math.sqrt(n)
    report.update({"input": str(args.input), "record_donors": len(donors) - len(sampled),
                   "public_donors": len(sampled), "max_dim": args.max_dim,
                   "variants_per_signature": args.variants,
                   "polish_rounds": args.polish_rounds,
                   "baseline_score": math.fsum(
                       incumbents[n] / math.sqrt(n) - 1 for n in range(1, 1001))})
    report["candidate_score"] = report["baseline_score"] - math.fsum(
        item["score_gain"] for item in report["improvements"])
    if improvements:
        write_partial(args.output, {n: packing[1]
                                    for n, packing in improvements.items()})
        report["candidate_csv"] = str(args.output)
    else:
        report["candidate_csv"] = None
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
