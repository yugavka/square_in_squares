"""Find shell offsets whose packing was only carried over from a smaller shell.

For n=(K-1)^2+j the excess is frac=L-(K-1). Adding a full row and column maps
shell K-1 to shell K with the same j and frac, so equal frac in consecutive
shells marks a transferred layout. A better layout at (K, j) also transfers to
every larger shell where frac is not already lower.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from optimal_locks import proven_optimal
from validate_submission import validate_submission

TIE = 1e-9


def shell(n: int) -> tuple[int, int]:
    k = math.isqrt(n - 1) + 1
    return k, n - (k - 1) ** 2


def plateau_rows(sides: dict[int, float], max_n: int = 1000) -> list[dict]:
    rows = []
    for n, side in sides.items():
        k, j = shell(n)
        frac = side - (k - 1)
        previous = (k - 2) ** 2 + j
        if (k < 3 or frac >= 1 - TIE or j > 2 * k - 3 or previous not in sides
                or proven_optimal(n) is not None):
            continue
        if abs(sides[previous] - (k - 2) - frac) > TIE:
            continue
        start = k - 1
        while (start > 2 and (start - 2) ** 2 + j in sides
               and abs(sides[(start - 2) ** 2 + j] - (start - 2) - frac) <= TIE):
            start -= 1
        # Weight of a unit frac gain carried to all larger shells it can reach.
        weight = 0.0
        upper = k
        while (upper - 1) ** 2 + j <= max_n:
            m = (upper - 1) ** 2 + j
            if m not in sides or sides[m] - (upper - 1) < frac - TIE:
                break
            weight += 1 / math.sqrt(m)
            upper += 1
        rows.append({"n": n, "K": k, "j": j, "side": side, "frac": frac,
                     "plateau_from_K": start, "plateau_length": k - start + 1,
                     "last_in_plateau": upper == k + 1,
                     "transfer_weight": weight})
    return sorted(rows, key=lambda row: (-row["last_in_plateau"], -row["K"],
                                         -row["transfer_weight"], row["n"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", nargs="?", type=Path,
                        default=Path(__file__).with_name("submission_improved.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path(__file__).with_name("plateau_report.csv"))
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    sides = {result.n: result.side for result in validate_submission(args.submission)}
    rows = plateau_rows(sides)
    with args.report.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]) if rows else ["n"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.12f}" if isinstance(value, float) else value
                             for key, value in row.items()})
    print("n     K   j   frac      plateau  last  weight")
    for row in rows[:args.top]:
        print(f"{row['n']:<5} {row['K']:<3} {row['j']:<3} {row['frac']:.6f}  "
              f"{row['plateau_from_K']:>2}..{row['K']:<4} "
              f"{'yes' if row['last_in_plateau'] else 'no ':<5} "
              f"{row['transfer_weight']:.4f}")
    print(f"{len(rows)} transferred cells; full report: {args.report}")


if __name__ == "__main__":
    main()
