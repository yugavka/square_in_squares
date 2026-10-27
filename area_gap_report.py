"""Rank packings without an optimality lock by excess area over n."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from optimal_locks import proven_optimal
from validate_submission import Result, validate_submission


def ranked_gaps(results: list[Result]) -> list[dict]:
    rows = []
    for result in results:
        if proven_optimal(result.n) is not None:
            continue
        area = result.side * result.side
        rows.append({
            "n": result.n,
            "side": result.side,
            "area": area,
            "area_lower_bound": result.n,
            "excess_area": area - result.n,
            "relative_excess_area": area / result.n - 1,
            "kaggle_score": result.score,
            "gain_vs_grid": result.baseline_score - result.score,
        })
    return sorted(rows, key=lambda row: (-row["relative_excess_area"], row["n"]))


def write_report(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=[
            "rank", "n", "side", "area", "area_lower_bound", "excess_area",
            "relative_excess_area", "kaggle_score", "gain_vs_grid",
        ])
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **{
                key: f"{value:.12f}" if isinstance(value, float) else value
                for key, value in row.items()
            }})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", nargs="?", type=Path,
                        default=Path(__file__).with_name("submission_improved.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path(__file__).with_name("area_gap_report.csv"))
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    results = validate_submission(args.submission)
    rows = ranked_gaps(results)
    write_report(args.report, rows)
    print("rank  n     side       excess area   relative excess")
    for rank, row in enumerate(rows[:args.top], start=1):
        print(f"{rank:<5} {row['n']:<5} {row['side']:<10.6f} "
              f"{row['excess_area']:<13.6f} "
              f"{row['relative_excess_area']:.2%}")
    print(f"Full report: {args.report}")


if __name__ == "__main__":
    main()
