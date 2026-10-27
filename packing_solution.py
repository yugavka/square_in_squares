"""Construct improved packings for the Squares in Squares competition.

The starting layouts are classical Goebel 5/27/52-square constructions and
selected catalog layouts in record_seeds.json. Each can be extended by one
outer row and column. Other puzzle sizes retain the grid baseline. No
optimization libraries or network access are required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from optimal_locks import (SAFETY_SCALE, SQRT2, check_locked_sides,
                           fixed_packing, five_square_seed, proven_optimal,
                           write_lock_manifest)
from validate_submission import (make_square, measure_packing, total_score,
                                 validate_submission)


SquarePlacement = tuple[float, float, float]
Packing = tuple[float, list[SquarePlacement]]


def goebel_strip_seed(k: int) -> Packing:
    """Two triangular staircases around a rotated strip (k=5 or 7)."""
    if k not in (5, 7):
        raise ValueError("this seed is used only for k=5 or k=7")
    side = k + SQRT2 / 2
    lower = [
        (x + 0.5, y + 0.5, 0.0)
        for x in range(k - 1)
        for y in range(k - 1 - x)
    ]
    lower.append((side - 0.5, 0.5, 0.0))
    upper = [(side - x, side - y, 0.0) for x, y, _ in lower]

    strip_length = (3 * k - 5) // 2
    strip = []
    for j in range(strip_length):
        offset = j + 0.5 - strip_length / 2
        x = k / 2 + (0.5 - offset) / SQRT2
        y = k / 2 + (0.5 + offset) / SQRT2
        strip.append((x, y, math.pi / 4))
    squares = lower + upper + strip
    assert len(squares) == k * k + (k - 1) // 2
    return side, squares


def extend_by_outer_l(side: float, squares: list[SquarePlacement], k: int) -> Packing:
    """Add 2k+1 squares to increase both the container side and k by one."""
    extended = list(squares)
    extended.extend((side + 0.5, j + 0.5, 0.0) for j in range(k))
    extended.extend((i + 0.5, side + 0.5, 0.0) for i in range(k + 1))
    return side + 1, extended


def record_seeds() -> list[Packing]:
    """Load selected published layouts and move each into the positive quadrant."""
    path = Path(__file__).with_name("record_seeds.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    seeds = []
    for n_text, entry in data.items():
        n = int(n_text)
        squares = [tuple(map(float, row)) for row in entry["squares"]]
        if len(squares) != n:
            raise ValueError(f"record seed n={n} has {len(squares)} squares")
        geometry = [make_square(i, *placement)
                    for i, placement in enumerate(squares)]
        min_x = min(square.min_x for square in geometry)
        min_y = min(square.min_y for square in geometry)
        normalized = [(x - min_x, y - min_y, angle) for x, y, angle in squares]
        side = measure_packing(n, [make_square(i, *placement)
                                  for i, placement in enumerate(normalized)]).side
        seeds.append((side, normalized))
    return seeds


def improved_packings(max_n: int = 1000) -> dict[int, Packing]:
    """Select the best valid construction or record-derived extension for each n."""
    candidates = [(2, five_square_seed()),
                  (5, goebel_strip_seed(5)),
                  (7, goebel_strip_seed(7))]
    candidates.extend((math.isqrt(len(squares)), (side, squares))
                      for side, squares in record_seeds())
    improved: dict[int, Packing] = {}
    for initial_k, (side, squares) in candidates:
        k = initial_k
        while len(squares) <= max_n:
            n = len(squares)
            grid_side = math.isqrt(n - 1) + 1
            # A small outward scaling of centers keeps decimal output safely
            # clear of exact contacts when Kaggle parses it at high precision.
            safe_squares = [(x * SAFETY_SCALE, y * SAFETY_SCALE, angle)
                            for x, y, angle in squares]
            if proven_optimal(n) is None and side * SAFETY_SCALE < grid_side:
                measured = measure_packing(
                    n, [make_square(i, *placement)
                        for i, placement in enumerate(safe_squares)]
                ).side
                previous = improved.get(n)
                if measured < grid_side and (previous is None or measured < previous[0]):
                    improved[n] = measured, safe_squares
            side, squares = extend_by_outer_l(side, squares, k)
            k += 1
    return improved


def write_submission(path: Path, max_n: int = 1000) -> int:
    """Write a complete submission, always preserving proof-backed layouts."""
    improved = improved_packings(max_n)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["id", "n", "square_id", "x", "y", "angle"])
        row_id = 0
        for n in range(1, max_n + 1):
            if proven_optimal(n) is not None:
                squares = fixed_packing(n)[1]
            elif n in improved:
                squares = improved[n][1]
            else:
                k = math.isqrt(n - 1) + 1
                squares = [(i % k + 0.5, i // k + 0.5, 0.0)
                           for i in range(n)]
            for square_id, (x, y, angle) in enumerate(squares):
                writer.writerow([row_id, n, square_id,
                                 format(x, ".17g"), format(y, ".17g"),
                                 format(angle, ".17g")])
                row_id += 1
    locked_improvements = sum(
        fixed_packing(n)[0] < math.isqrt(n - 1) + 1
        for n in range(1, max_n + 1) if proven_optimal(n) is not None
    )
    return len(improved) + locked_improvements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("submission_improved.csv"))
    args = parser.parse_args()
    improved_count = write_submission(args.output)
    lock_count = write_lock_manifest(Path(__file__).with_name("locked_optimal.csv"))
    results = validate_submission(args.output)
    check_locked_sides(results)
    print(f"Wrote {args.output} with {improved_count} improved puzzles")
    print(f"Fixed proof-backed layouts: {lock_count}")
    print(f"Local Kaggle score: {total_score(results):.12f}")


if __name__ == "__main__":
    main()
