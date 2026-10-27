"""Construct improved packings for the Squares in Squares competition.

The starting layouts are classical Goebel 5/27/52-square constructions and
selected catalog layouts in record_seeds.json. Each can be extended by one
outer row and column, cut along a unit-wide strip, or trimmed by removing
boundary squares. Rectangular crops are joined along a diagonal fault by a
minimax dynamic program. Optional validated CSVs form a per-n portfolio.
Other puzzle sizes retain the grid baseline. No optimization libraries or
network access are required.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import csv
import gzip
import json
import math
from pathlib import Path
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fault_line_dp import Gadget

from optimal_locks import (SAFETY_SCALE, SQRT2, check_locked_sides,
                           fixed_packing, five_square_seed, proven_optimal,
                           write_lock_manifest)
from validate_submission import (make_square, measure_packing, total_score,
                                 validate_submission)


SquarePlacement = tuple[float, float, float]
Packing = tuple[float, list[SquarePlacement]]
PUBLIC_PORTFOLIO = (Path(__file__).with_name("published_sources") /
                    "malyshev_v36_partial.csv.gz")
TRIM_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (1, -1), (-1, 1), (-1, -1))
MAX_TRIM = 4


@dataclass(frozen=True)
class FaultTrace:
    source_n: int
    gadgets: tuple[Gadget, ...]
    steps: tuple[str, ...] = ()


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


def trim_one_square(squares: list[SquarePlacement],
                    direction: tuple[int, int]) -> Packing:
    """Remove the square giving the smallest remaining bounding square.

    A directional tie break explores different boundary regions over several
    successive removals. Any subset remains nonoverlapping by construction.
    """
    geometry = [make_square(i, *placement)
                for i, placement in enumerate(squares)]
    if len(geometry) < 2:
        raise ValueError("cannot trim a one-square packing")
    min_x = sorted((square.min_x, i) for i, square in enumerate(geometry))
    max_x = sorted((square.max_x, i) for i, square in enumerate(geometry))
    min_y = sorted((square.min_y, i) for i, square in enumerate(geometry))
    max_y = sorted((square.max_y, i) for i, square in enumerate(geometry))

    best_key = None
    best_index = -1
    for i, square in enumerate(geometry):
        left = min_x[1][0] if min_x[0][1] == i else min_x[0][0]
        right = max_x[-2][0] if max_x[-1][1] == i else max_x[-1][0]
        bottom = min_y[1][0] if min_y[0][1] == i else min_y[0][0]
        top = max_y[-2][0] if max_y[-1][1] == i else max_y[-1][0]
        side = max(right - left, top - bottom)
        # Round only for the tie break: floating point noise must not decide
        # which boundary region is searched next.
        key = (round(side, 9), -(direction[0] * square.x +
                                 direction[1] * square.y), i)
        if best_key is None or key < best_key:
            best_key, best_index = key, i

    remaining = squares[:best_index] + squares[best_index + 1:]
    remaining_geometry = geometry[:best_index] + geometry[best_index + 1:]
    side = max(
        max(square.max_x for square in remaining_geometry) -
        min(square.min_x for square in remaining_geometry),
        max(square.max_y for square in remaining_geometry) -
        min(square.min_y for square in remaining_geometry),
    )
    return side, remaining


def portfolio_packings(paths: Iterable[Path], max_n: int) -> dict[int, Packing]:
    """Select the best independently validated layout for each n in CSV inputs."""
    best: dict[int, Packing] = {}
    for path in paths:
        measured = {result.n: result.side for result in
                    validate_submission(path, max_n=max_n, allow_partial=True)}
        groups: dict[int, list[SquarePlacement | None]] = {
            n: [None] * n for n in measured
        }
        opener = gzip.open if path.suffix == ".gz" else Path.open
        with opener(path, mode="rt", newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                n = int(row["n"])
                groups[n][int(row["square_id"])] = (
                    float(row["x"]), float(row["y"]), float(row["angle"])
                )
        for n, rows in groups.items():
            side = measured[n]
            if (proven_optimal(n) is not None or
                    side >= math.isqrt(n - 1) + 1 - 1e-9):
                continue
            previous = best.get(n)
            if previous is None or side < previous[0] - 1e-9:
                best[n] = side, [row for row in rows if row is not None]
    return best


def slab_candidates(squares: list[SquarePlacement],
                    max_removed: int | None = None) -> Iterator[tuple[int, Packing]]:
    """Remove a unit-wide vertical or horizontal strip and close the gap.

    Every square touching the strip is removed; squares on opposite sides are
    separated by at least one unit before the translation. Candidate positions
    come from square edges, including strips flush with tilted blocks.
    """
    n = len(squares)
    if max_removed is None:
        max_removed = 2 * math.isqrt(n) + 6
    geometry = [make_square(i, *row) for i, row in enumerate(squares)]
    for axis in ("x", "y"):
        lower = [getattr(square, f"min_{axis}") for square in geometry]
        upper = [getattr(square, f"max_{axis}") for square in geometry]
        sorted_lower, sorted_upper = sorted(lower), sorted(upper)
        positions = sorted(set(lower) | set(upper) |
                           {value - 1 for value in lower} |
                           {value - 1 for value in upper})
        for cut in positions:
            if cut <= sorted_lower[0] or cut + 1 >= sorted_upper[-1]:
                continue
            left_count = bisect_right(sorted_upper, cut)
            right_count = n - bisect_left(sorted_lower, cut + 1)
            kept = left_count + right_count
            if kept < 2 or n - kept > max_removed:
                continue
            packed: list[SquarePlacement] = []
            min_x = min_y = math.inf
            max_x = max_y = -math.inf
            for i, (x, y, angle) in enumerate(squares):
                shift_x = shift_y = 0
                if upper[i] <= cut:
                    packed.append((x, y, angle))
                elif lower[i] >= cut + 1:
                    shift_x, shift_y = (1, 0) if axis == "x" else (0, 1)
                    packed.append((x - shift_x, y - shift_y, angle))
                else:
                    continue
                square = geometry[i]
                min_x = min(min_x, square.min_x - shift_x)
                max_x = max(max_x, square.max_x - shift_x)
                min_y = min(min_y, square.min_y - shift_y)
                max_y = max(max_y, square.max_y - shift_y)
            if len(packed) != kept:
                raise AssertionError("slab classification disagrees with counts")
            yield kept, (max(max_x - min_x, max_y - min_y), packed)


def improved_packings(max_n: int = 1000,
                      portfolio_paths: Iterable[Path] = (),
                      include_public: bool = True,
                      include_fault_dp: bool = True,
                      trace: dict[int, FaultTrace] | None = None
                      ) -> dict[int, Packing]:
    """Select the best seed, portfolio, fault, slab and trimmed layout per n."""
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

    sources = list(portfolio_paths)
    if include_public and max_n == 1000:
        sources.insert(0, PUBLIC_PORTFOLIO)
    source_packings = portfolio_packings(sources, max_n)
    for n, candidate in source_packings.items():
        previous = improved.get(n)
        if previous is None or candidate[0] < previous[0] - 1e-9:
            improved[n] = candidate

    if include_fault_dp and max_n == 1000:
        from fault_line_dp import fault_line_compositions

        donors = [packing for _, packing in candidates]
        donors.extend(source_packings.values())
        locked = (n for n in range(1, max_n + 1)
                  if proven_optimal(n) is not None)
        provenance: dict[int, tuple[Gadget, ...]] | None = ({}
            if trace is not None else None)
        compositions = fault_line_compositions(donors, improved, max_n,
                                                locked, provenance)
        improved.update(compositions)
        if trace is not None and provenance is not None:
            trace.update({n: FaultTrace(n, chain)
                          for n, chain in provenance.items()})

    # A unit-wide strip can be removed from a larger packing and the two
    # remaining pieces can be joined without creating a new intersection.
    slab_donors = [(n, squares, trace.get(n) if trace is not None else None)
                   for n, (_, squares) in improved.items()]
    for donor_n, donor_squares, donor_trace in slab_donors:
        for n, candidate in slab_candidates(donor_squares):
            if n > max_n or proven_optimal(n) is not None:
                continue
            side = candidate[0]
            previous = improved.get(n)
            grid_side = math.isqrt(n - 1) + 1
            if side < grid_side and (previous is None or side < previous[0] - 1e-9):
                measured = measure_packing(
                    n, [make_square(i, *row) for i, row in enumerate(candidate[1])]
                ).side
                if abs(measured - side) > 1e-8:
                    raise AssertionError("slab side changed during validation")
                improved[n] = candidate
                if trace is not None:
                    if donor_trace is None:
                        trace.pop(n, None)
                    else:
                        trace[n] = FaultTrace(donor_trace.source_n,
                                              donor_trace.gadgets,
                                              donor_trace.steps +
                                              (f"slab {donor_n}->{n}",))

    # Neighbor transfer: a good (n+d)-square packing can improve n after
    # removing d squares. Explore several directions because equal one-step
    # sides can lead to different later reductions.
    trim_donors = [(n, squares, trace.get(n) if trace is not None else None)
                   for n, (_, squares) in improved.items()]
    for donor_n, donor_squares, donor_trace in trim_donors:
        for direction in TRIM_DIRECTIONS:
            squares = donor_squares
            for _ in range(min(MAX_TRIM, donor_n - 1)):
                side, squares = trim_one_square(squares, direction)
                n = len(squares)
                if proven_optimal(n) is not None:
                    continue
                previous = improved.get(n)
                grid_side = math.isqrt(n - 1) + 1
                if side < grid_side and (previous is None or side < previous[0] - 1e-9):
                    improved[n] = side, squares
                    if trace is not None:
                        if donor_trace is None:
                            trace.pop(n, None)
                        else:
                            trace[n] = FaultTrace(donor_trace.source_n,
                                                  donor_trace.gadgets,
                                                  donor_trace.steps +
                                                  (f"trim {donor_n}->{n}",))
    return improved


def write_submission(path: Path, max_n: int = 1000,
                     portfolio_paths: Iterable[Path] = (),
                     include_public: bool = True,
                     include_fault_dp: bool = True) -> int:
    """Write a complete submission, always preserving proof-backed layouts."""
    improved = improved_packings(max_n, portfolio_paths, include_public,
                                 include_fault_dp)
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
    parser.add_argument("--portfolio", action="append", type=Path, default=[],
                        help="validated full or partial submission CSV; repeat for multiple inputs")
    parser.add_argument("--no-public-portfolio", action="store_true",
                        help="exclude the bundled public replay for ablation")
    parser.add_argument("--no-fault-dp", action="store_true",
                        help="disable diagonal gadget composition for ablation")
    args = parser.parse_args()
    if args.output.resolve() in {path.resolve() for path in args.portfolio}:
        parser.error("--output must differ from every --portfolio input")
    improved_count = write_submission(args.output, portfolio_paths=args.portfolio,
                                      include_public=not args.no_public_portfolio,
                                      include_fault_dp=not args.no_fault_dp)
    lock_count = write_lock_manifest(Path(__file__).with_name("locked_optimal.csv"))
    results = validate_submission(args.output)
    check_locked_sides(results)
    print(f"Wrote {args.output} with {improved_count} improved puzzles")
    print(f"Fixed proof-backed layouts: {lock_count}")
    print(f"Local Kaggle score: {total_score(results):.12f}")


if __name__ == "__main__":
    main()
