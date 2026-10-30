"""Exact overlap check on the decimal text of a submission, as Kaggle does it.

The Kaggle metric certifies the submitted text with interval arithmetic, so a
pair that overlaps by 1e-16 fails there while validate_submission.py, which
allows OVERLAP_TOLERANCE, accepts it. This check redoes the separating-axis
test with 45-digit decimals on the text itself and reports every puzzle whose
worst pair penetrates by more than zero. Exact contacts (penetration 0, as
in the grid) are allowed, as in the metric.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from collections.abc import Iterable
from decimal import Decimal, localcontext
from multiprocessing import Pool
from pathlib import Path

DIGITS = 45
PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494")
TextRow = tuple[str, str, str]


def cos_sin(angle: Decimal) -> tuple[Decimal, Decimal]:
    """cos and sin by Taylor series after reduction by multiples of pi/2."""
    quarter = PI / 2
    turns = int((angle / quarter).to_integral_value())
    r = angle - turns * quarter
    cos, sin = Decimal(0), Decimal(0)
    term_cos, term_sin = Decimal(1), r
    for m in range(40):
        cos += term_cos
        sin += term_sin
        term_cos = -term_cos * r * r / ((2 * m + 1) * (2 * m + 2))
        term_sin = -term_sin * r * r / ((2 * m + 2) * (2 * m + 3))
    for _ in range(turns % 4):
        cos, sin = -sin, cos
    return cos, sin


def worst_penetration(rows: list[TextRow]) -> Decimal:
    """Largest SAT penetration over all nearby pairs; negative means a gap."""
    with localcontext() as context:
        context.prec = DIGITS
        half = Decimal(1) / 2
        centres, axes, corners = [], [], []
        for x_text, y_text, angle_text in rows:
            x, y = Decimal(x_text), Decimal(y_text)
            cos, sin = cos_sin(Decimal(angle_text))
            d1 = (half * cos - half * sin, half * sin + half * cos)
            d2 = (-half * cos - half * sin, -half * sin + half * cos)
            centres.append((float(x), float(y)))
            axes.append(((cos, sin), (-sin, cos)))
            corners.append([(x + d1[0], y + d1[1]), (x + d2[0], y + d2[1]),
                            (x - d1[0], y - d1[1]), (x - d2[0], y - d2[1])])
        worst = Decimal(-1)
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if (abs(centres[i][0] - centres[j][0]) > 1.5
                        or abs(centres[i][1] - centres[j][1]) > 1.5):
                    continue
                gap = None
                for ax, ay in (*axes[i], *axes[j]):
                    first = [ax * u + ay * v for u, v in corners[i]]
                    second = [ax * u + ay * v for u, v in corners[j]]
                    g = max(min(first), min(second)) - min(max(first), max(second))
                    gap = g if gap is None else max(gap, g)
                worst = max(worst, -gap)
        return worst


def read_text(path: Path) -> dict[int, list[TextRow]]:
    """Decimal text rows per n, ordered by square_id."""
    groups: dict[int, dict[int, TextRow]] = {}
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, mode="rt", newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            groups.setdefault(int(row["n"]), {})[int(row["square_id"])] = (
                row["x"].strip(), row["y"].strip(), row["angle"].strip())
    return {n: [rows[k] for k in sorted(rows)] for n, rows in groups.items()}


def _check(item: tuple[int, list[TextRow]]) -> tuple[int, float]:
    n, rows = item
    return n, float(worst_penetration(rows))


def overlapping(groups: dict[int, list[TextRow]],
                only: Iterable[int] | None = None,
                processes: int | None = None) -> dict[int, float]:
    """Puzzles whose text overlaps by more than zero, with the penetration."""
    keep = None if only is None else set(only)
    work = [(n, rows) for n, rows in sorted(groups.items())
            if keep is None or n in keep]
    if len(work) < 8:
        results = [_check(item) for item in work]
    else:
        with Pool(processes) as pool:
            results = pool.map(_check, work, chunksize=4)
    return {n: penetration for n, penetration in results if penetration > 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="submission CSV or .csv.gz")
    args = parser.parse_args()
    bad = overlapping(read_text(args.path))
    print(f"{len(bad)} puzzles overlap in exact decimal arithmetic")
    for n, penetration in sorted(bad.items()):
        print(f"n={n}: penetration {penetration:.3e}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
