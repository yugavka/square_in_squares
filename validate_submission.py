"""Strictly validate a submission and score valid puzzles by Kaggle's formula.

The score uses Kaggle's formula, while geometry is checked with double precision.
Angles are interpreted as radians by default; use --angle-unit degrees if needed.
Unlike Kaggle V17, an invalid submitted puzzle raises an error instead of
receiving the grid-ceiling fallback score. This is deliberate preflight behavior.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
from dataclasses import dataclass
from pathlib import Path


HEADER = ["id", "n", "square_id", "x", "y", "angle"]
OVERLAP_TOLERANCE = 1e-9


class ValidationError(ValueError):
    """A malformed or geometrically invalid submission."""


@dataclass(slots=True)
class Square:
    square_id: int
    x: float
    y: float
    ux: float
    uy: float
    min_x: float
    max_x: float
    min_y: float
    max_y: float


@dataclass(slots=True)
class Result:
    n: int
    side: float
    baseline_side: int

    @property
    def improvement(self) -> float:
        return self.baseline_side - self.side

    @property
    def score(self) -> float:
        return self.side / math.sqrt(self.n) - 1

    @property
    def baseline_score(self) -> float:
        return self.baseline_side / math.sqrt(self.n) - 1


def make_square(square_id: int, x: float, y: float, angle: float) -> Square:
    # A square is unchanged by a quarter turn. Reducing the angle also makes
    # trigonometry more accurate for large, but finite, input angles.
    angle = math.remainder(angle, math.pi / 2)
    ux, uy = math.cos(angle), math.sin(angle)
    half_extent = (abs(ux) + abs(uy)) / 2
    return Square(
        square_id, x, y, ux, uy,
        x - half_extent, x + half_extent,
        y - half_extent, y + half_extent,
    )


def squares_overlap(a: Square, b: Square, tolerance: float = OVERLAP_TOLERANCE) -> bool:
    """Use the separating axis theorem; touching edges are allowed."""
    dx, dy = b.x - a.x, b.y - a.y
    axes = ((a.ux, a.uy), (-a.uy, a.ux),
            (b.ux, b.uy), (-b.uy, b.ux))
    for axis_x, axis_y in axes:
        distance = abs(dx * axis_x + dy * axis_y)
        a_radius = (abs(a.ux * axis_x + a.uy * axis_y)
                    + abs(-a.uy * axis_x + a.ux * axis_y)) / 2
        b_radius = (abs(b.ux * axis_x + b.uy * axis_y)
                    + abs(-b.uy * axis_x + b.ux * axis_y)) / 2
        if distance >= a_radius + b_radius - tolerance:
            return False
    return True


def measure_packing(n: int, squares: list[Square]) -> Result:
    """Check all possible contacts using an x-axis sweep, then measure bounds."""
    ordered = sorted(squares, key=lambda square: square.min_x)
    active: list[Square] = []
    for square in ordered:
        active = [other for other in active
                  if other.max_x > square.min_x]
        for other in active:
            if other.max_y <= square.min_y or square.max_y <= other.min_y:
                continue
            if squares_overlap(other, square):
                raise ValidationError(
                    f"n={n}: squares {other.square_id} and {square.square_id} overlap"
                )
        active.append(square)

    width = max(square.max_x for square in squares) - min(square.min_x for square in squares)
    height = max(square.max_y for square in squares) - min(square.min_y for square in squares)
    return Result(n, max(width, height), math.isqrt(n - 1) + 1)


def validate_submission(path: Path, max_n: int = 1000,
                        angle_unit: str = "radians",
                        allow_partial: bool = False) -> list[Result]:
    """Check CSV identifiers, counts, finite coordinates, and all packings."""
    expected_rows = max_n * (max_n + 1) // 2
    seen_ids: set[int] = set()
    groups: list[dict[int, Square]] = [{} for _ in range(max_n + 1)]
    row_count = 0

    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, mode="rt", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != HEADER:
            raise ValidationError(f"expected CSV header {','.join(HEADER)}")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValidationError(f"line {line_number}: wrong number of columns")
            try:
                row_id = int(row["id"])
                n = int(row["n"])
                square_id = int(row["square_id"])
                x, y, angle = (float(row[key]) for key in ("x", "y", "angle"))
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"line {line_number}: invalid numeric value") from exc
            if row_id in seen_ids:
                raise ValidationError(f"line {line_number}: duplicate id {row_id}")
            seen_ids.add(row_id)
            if not (1 <= n <= max_n):
                raise ValidationError(f"line {line_number}: n {n} outside 1..{max_n}")
            if not (0 <= square_id < n):
                raise ValidationError(f"line {line_number}: square_id {square_id} outside 0..{n - 1}")
            if square_id in groups[n]:
                raise ValidationError(f"line {line_number}: duplicate square_id {square_id} for n={n}")
            if not all(map(math.isfinite, (x, y, angle))):
                raise ValidationError(f"line {line_number}: coordinates and angle must be finite")
            if angle_unit == "degrees":
                angle = math.radians(angle)
            groups[n][square_id] = make_square(square_id, x, y, angle)
            row_count += 1

    if not allow_partial and row_count != expected_rows:
        raise ValidationError(f"expected {expected_rows} rows, found {row_count}; "
                              "use --allow-partial for a partial submission")

    results = []
    for n in range(1, max_n + 1):
        if allow_partial and not groups[n]:
            continue
        if len(groups[n]) != n:
            raise ValidationError(f"n={n}: expected {n} squares, found {len(groups[n])}")
        results.append(measure_packing(n, list(groups[n].values())))
    return results


def total_score(results: list[Result], max_n: int = 1000) -> float:
    """Sum L/sqrt(n)-1; absent puzzles receive the grid-ceiling score."""
    by_n = {result.n: result for result in results}
    return math.fsum(
        by_n[n].score if n in by_n else (math.isqrt(n - 1) + 1) / math.sqrt(n) - 1
        for n in range(1, max_n + 1)
    )


def write_report(path: Path, results: list[Result], max_n: int = 1000) -> None:
    by_n = {result.n: result for result in results}
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["n", "status", "side", "baseline_side", "improvement",
                         "relative_improvement", "score", "baseline_score", "score_gain"])
        for n in range(1, max_n + 1):
            result = by_n.get(n)
            if result is None:
                baseline = math.isqrt(n - 1) + 1
                result = Result(n, float(baseline), baseline)
                status = "missing_grid_penalty"
            else:
                status = "submitted"
            writer.writerow([
                result.n, status, f"{result.side:.12f}", result.baseline_side,
                f"{result.improvement:.12f}",
                f"{result.improvement / result.baseline_side:.12f}",
                f"{result.score:.12f}", f"{result.baseline_score:.12f}",
                f"{result.baseline_score - result.score:.12f}",
            ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "submission", nargs="?", type=Path,
        default=Path(__file__).with_name("submission.csv"),
        help="submission CSV to validate (default: submission.csv beside this script)",
    )
    parser.add_argument("--angle-unit", choices=("radians", "degrees"), default="radians")
    parser.add_argument("--allow-partial", action="store_true",
                        help="score missing puzzles as trivial grid packings")
    parser.add_argument("--report", type=Path, help="write per-n measurements to CSV")
    parser.add_argument("--show-all", action="store_true", help="print all per-n measurements")
    args = parser.parse_args()

    try:
        results = validate_submission(args.submission, angle_unit=args.angle_unit,
                                      allow_partial=args.allow_partial)
        if args.report:
            write_report(args.report, results)
    except (OSError, ValidationError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1

    improved = [result for result in results if result.improvement > 1e-8]
    worse = [result for result in results if result.improvement < -1e-8]
    grid_score = total_score([], max_n=1000)
    score = total_score(results, max_n=1000)
    print(f"VALID: {len(results)} packings, {sum(result.n for result in results)} squares")
    print(f"Kaggle score (lower is better): {score:.12f}")
    print(f"Grid score: {grid_score:.12f}; gain: {grid_score - score:.12f}")
    if args.allow_partial:
        print(f"Missing puzzles charged grid score: {1000 - len(results)}")
    print(f"Improved vs grid: {len(improved)}; worse: {len(worse)}")
    if args.show_all:
        selected = results
    else:
        selected = sorted(improved, key=lambda result: result.improvement, reverse=True)[:10]
    if selected:
        print("n  side         grid  improvement")
        for result in selected:
            print(f"{result.n:<3} {result.side:<12.8f} {result.baseline_side:<5} "
                  f"{result.improvement:+.8f}")
    if args.report:
        print(f"Per-n report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
