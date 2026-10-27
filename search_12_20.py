"""Reproducible local search for packings smaller than the 12/20-square grids.

This is a feasibility search, not a proof of optimality. A candidate is saved
only after the independent submission geometry validator accepts it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from optimal_locks import SAFETY_SCALE
from packing_solution import record_seeds
from validate_submission import ValidationError, make_square, measure_packing


Placement = tuple[float, float, float]


def extents(angle: float) -> float:
    return (abs(math.cos(angle)) + abs(math.sin(angle))) / 2


def boundary_cost(square: Placement, side: float) -> float:
    x, y, angle = square
    radius = extents(angle)
    return sum(max(0.0, gap) ** 2 for gap in
               (radius - x, x + radius - side,
                radius - y, y + radius - side))


def pair_cost(a: Placement, b: Placement) -> float:
    ax, ay, aa = a
    bx, by, ba = b
    dx, dy = bx - ax, by - ay
    au = (math.cos(aa), math.sin(aa))
    av = (-au[1], au[0])
    bu = (math.cos(ba), math.sin(ba))
    bv = (-bu[1], bu[0])
    depths = []
    for ux, uy in (au, av, bu, bv):
        a_radius = (abs(au[0] * ux + au[1] * uy)
                    + abs(av[0] * ux + av[1] * uy)) / 2
        b_radius = (abs(bu[0] * ux + bu[1] * uy)
                    + abs(bv[0] * ux + bv[1] * uy)) / 2
        depth = a_radius + b_radius - abs(dx * ux + dy * uy)
        if depth <= 0:
            return 0.0
        depths.append(depth)
    return min(depths) ** 2


def local_cost(index: int, candidate: Placement,
               squares: list[Placement], side: float) -> float:
    return boundary_cost(candidate, side) + sum(
        pair_cost(candidate, square) for j, square in enumerate(squares)
        if j != index
    )


def initial_layout(n: int, rng: random.Random, restart: int,
                   smaller_seed: list[Placement]) -> list[Placement]:
    k = math.isqrt(n - 1) + 1
    # Explore the grid, its vacancies, and the published (n-1)-square layout.
    if restart == 0:
        return [(i % k + .5, i // k + .5, 0.0) for i in range(n)]
    if restart % 3 == 2:
        geometry = [make_square(i, *placement)
                    for i, placement in enumerate(smaller_seed)]
        width = max(square.max_x for square in geometry)
        height = max(square.max_y for square in geometry)
        shift_x = (k - width) / 2
        shift_y = (k - height) / 2
        return [(x + shift_x + rng.gauss(0, .08),
                 y + shift_y + rng.gauss(0, .08),
                 angle + rng.gauss(0, .04))
                for x, y, angle in smaller_seed] + [
                    (rng.uniform(.5, k - .5), rng.uniform(.5, k - .5),
                     rng.uniform(-.3, .3))]
    if restart % 2 == 0:
        cells = [(i % k, i // k) for i in range(n)]
    else:
        cells = rng.sample([(x, y) for y in range(k) for x in range(k)], n)
    return [(x + .5 + rng.gauss(0, .12),
             y + .5 + rng.gauss(0, .12),
             rng.gauss(0, .10)) for x, y in cells]


def search(n: int, side: float, restarts: int, iterations: int,
           seed: int) -> tuple[float, list[Placement], int]:
    rng = random.Random(seed)
    best_cost = math.inf
    best_squares: list[Placement] = []
    evaluations = 0
    smaller_seed = next(squares for _, squares in record_seeds()
                        if len(squares) == n - 1)
    for restart in range(restarts):
        squares = initial_layout(n, rng, restart, smaller_seed)
        total = sum(boundary_cost(square, side) for square in squares)
        total += sum(pair_cost(squares[i], squares[j])
                     for i in range(n) for j in range(i + 1, n))
        if total < best_cost:
            best_cost, best_squares = total, list(squares)
        for iteration in range(iterations):
            i = rng.randrange(n)
            old = squares[i]
            progress = (iteration % 2000) / 2000
            temperature = 0.003 * (1e-7 / 0.003) ** progress
            step = .22 * (0.03 + 0.97 * (1 - progress))
            candidate = (old[0] + rng.gauss(0, step),
                         old[1] + rng.gauss(0, step),
                         max(-math.pi / 4, min(math.pi / 4,
                             old[2] + rng.gauss(0, step / 2))))
            old_cost = local_cost(i, old, squares, side)
            new_cost = local_cost(i, candidate, squares, side)
            evaluations += 1
            delta = new_cost - old_cost
            if delta < 0 or rng.random() < math.exp(-delta / temperature):
                squares[i] = candidate
                total += delta
                if total < best_cost:
                    best_cost, best_squares = total, list(squares)
    return best_cost, best_squares, evaluations


def verified_candidate(n: int, side: float,
                       squares: list[Placement]) -> list[Placement] | None:
    if not squares:
        return None
    geometry = [make_square(i, *placement)
                for i, placement in enumerate(squares)]
    min_x = min(square.min_x for square in geometry)
    min_y = min(square.min_y for square in geometry)
    normalized = [((x - min_x) * SAFETY_SCALE,
                   (y - min_y) * SAFETY_SCALE, angle)
                  for x, y, angle in squares]
    try:
        result = measure_packing(n, [make_square(i, *placement)
                                     for i, placement in enumerate(normalized)])
    except ValidationError:
        return None
    return normalized if result.side < side else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restarts', type=int, default=12)
    parser.add_argument('--iterations', type=int, default=12000)
    parser.add_argument('--seed', type=int, default=20261027)
    parser.add_argument('--target-gap', type=float, default=0.01,
                        help='target side is grid side minus this amount')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('search_12_20_results.json'))
    args = parser.parse_args()
    if args.restarts <= 0 or args.iterations <= 0 or not 0 < args.target_gap < 0.1:
        parser.error('restarts and iterations must be positive; target-gap must be in (0, 0.1)')
    report = {}
    for n, grid_side in ((12, 4), (20, 5)):
        target = grid_side - args.target_gap
        cost, squares, evaluations = search(n, target, args.restarts,
                                            args.iterations, args.seed + n)
        candidate = verified_candidate(n, grid_side, squares)
        report[str(n)] = {'target_side': target, 'best_penalty': cost,
                          'evaluations': evaluations,
                          'verified_side': None if candidate is None else
                          measure_packing(n, [make_square(i, *p)
                                              for i, p in enumerate(candidate)]).side,
                          'squares': candidate}
        print(f'n={n}: target {target:.5f}, best penalty {cost:.9g}, '
              f'verified side {report[str(n)]["verified_side"]}')
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Report: {args.output}')


if __name__ == '__main__':
    main()
