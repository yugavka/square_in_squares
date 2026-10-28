"""Sequential LP refinement with centres and every square angle free.

Each LP linearizes one separating-axis branch per nearby pair. Both one-sided
slopes are included at absolute-value kinks. Every accepted step is checked
against the exact unit-square geometry before changing the incumbent.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree


Placement = tuple[float, float, float]


def _support_slope(angle: float) -> list[float]:
    c, s = math.cos(angle), math.sin(angle)
    if min(abs(c), abs(s)) < 1e-8:
        return [-0.5, 0.5]
    return [(-s * math.copysign(1, c) + c * math.copysign(1, s)) / 2]


def _pair_slopes(delta: float) -> list[float]:
    return _support_slope(delta)


def _chosen_axis(a: Placement, b: Placement
                 ) -> tuple[float, float, float, int]:
    ax, ay, aa = a
    bx, by, ba = b
    ac, ass = math.cos(aa), math.sin(aa)
    bc, bs = math.cos(ba), math.sin(ba)
    axes = ((ac, ass), (-ass, ac), (bc, bs), (-bs, bc))
    best = None
    for owner, (vx, vy) in enumerate(axes):
        projection = (bx - ax) * vx + (by - ay) * vy
        sign = 1 if projection >= 0 else -1
        ux, uy = sign * vx, sign * vy
        support_a = (abs(ac * ux + ass * uy) +
                     abs(-ass * ux + ac * uy)) / 2
        support_b = (abs(bc * ux + bs * uy) +
                     abs(-bs * ux + bc * uy)) / 2
        gap = abs(projection) - support_a - support_b
        if best is None or gap > best[0]:
            best = gap, ux, uy, owner
    assert best is not None
    return best


def linearized_step(placements: list[Placement], side: float,
                    step: float, time_limit: float = 2.0,
                    height: float | None = None
                    ) -> list[tuple[float, float, float]] | None:
    """Find a local LP step; returned geometry still needs exact validation."""
    n = len(placements)
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    rhs: list[float] = []

    def add(entries: tuple[tuple[int, float], ...], bound: float) -> None:
        row = len(rhs)
        for column, value in entries:
            row_indices.append(row)
            col_indices.append(column)
            values.append(value)
        rhs.append(bound)

    points = [(x, y) for x, y, _ in placements]
    radius = math.sqrt(2) + 2 * math.sqrt(2) * step + 1e-6
    pairs = cKDTree(points).query_pairs(radius)
    for i, j in sorted(pairs):
        a, b = placements[i], placements[j]
        gap, ux, uy, owner = _chosen_axis(a, b)
        dx, dy = b[0] - a[0], b[1] - a[1]
        rotation = -dx * uy + dy * ux
        for slope in _pair_slopes(b[2] - a[2]):
            if owner < 2:
                angle_i, angle_j = -rotation - slope, slope
            else:
                angle_i, angle_j = -slope, slope - rotation
            add(((3 * i, ux), (3 * i + 1, uy),
                 (3 * j, -ux), (3 * j + 1, -uy),
                 (3 * i + 2, angle_i), (3 * j + 2, angle_j)),
                gap - 4e-10)

    if height is None:
        height = side
    for i, (x, y, angle) in enumerate(placements):
        h = (abs(math.cos(angle)) + abs(math.sin(angle))) / 2
        for dimension, centre, bound in ((0, x, side), (1, y, height)):
            for slope in _support_slope(angle):
                add(((3 * i + dimension, -1), (3 * i + 2, slope)),
                    centre - h - 4e-10)
                add(((3 * i + dimension, 1), (3 * i + 2, slope),
                     (3 * n, -1)), bound - centre - h - 4e-10)

    matrix = coo_matrix((values, (row_indices, col_indices)),
                        shape=(len(rhs), 3 * n + 1)).tocsr()
    objective = [0.0] * (3 * n) + [1.0]
    bounds = [(-step, step) if k % 3 != 2 else (-step / 3, step / 3)
              for k in range(3 * n)] + [(-step, step)]
    result = linprog(objective, A_ub=matrix, b_ub=rhs,
                     bounds=bounds, method="highs",
                     options={"time_limit": time_limit,
                              "primal_feasibility_tolerance": 1e-9,
                              "dual_feasibility_tolerance": 1e-9})
    if not result.success:
        return None
    return [tuple(float(result.x[3 * i + k]) for k in range(3))
            for i in range(n)]


def polish(placements: list[Placement], rounds: int,
           strict_side: Callable[[list[Placement]], float],
           normalize: Callable[[list[Placement]], list[Placement]],
           initial_step: float = 0.025) -> tuple[list[Placement], int]:
    """Run bounded sequential LP, accepting only exact SAT-valid decreases."""
    best = normalize(placements)
    side = strict_side(best)
    step = initial_step
    improvements = 0
    for _ in range(rounds):
        move = linearized_step(best, side, step)
        winner = None
        winner_side = side
        if move is not None:
            for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                proposal = [(x + fraction * dx,
                             y + fraction * dy,
                             angle + fraction * da)
                            for (x, y, angle), (dx, dy, da) in zip(best, move)]
                try:
                    measured = strict_side(proposal)
                except ValueError:
                    continue
                if measured < winner_side - 1e-10:
                    winner, winner_side = normalize(proposal), measured
        if winner is None:
            step *= 0.4
            if step < 2e-7:
                break
        else:
            best, side = winner, winner_side
            improvements += 1
            step = min(0.06, step * 1.3)
    return best, improvements


def polish_rectangle(gadget, placements: list[Placement], rounds: int,
                     initial_step: float = 0.025
                     ) -> tuple[list[Placement], int]:
    """Optimize centres, angles and common overhang of a checked gadget."""
    from search_gadget import checked_gadget

    overhang, normalized = checked_gadget(gadget, placements)
    best = list(normalized)
    step = initial_step
    improvements = 0
    for _ in range(rounds):
        move = linearized_step(best, gadget.w + overhang, step,
                               height=gadget.h + overhang)
        winner = None
        winner_overhang = overhang
        if move is not None:
            for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                proposal = [(x + fraction * dx,
                             y + fraction * dy,
                             angle + fraction * da)
                            for (x, y, angle), (dx, dy, da) in zip(best, move)]
                try:
                    measured, normalized = checked_gadget(gadget, proposal)
                except ValueError:
                    continue
                if measured < winner_overhang - 1e-10:
                    winner, winner_overhang = list(normalized), measured
        if winner is None:
            step *= 0.4
            if step < 2e-7:
                break
        else:
            best, overhang = winner, winner_overhang
            improvements += 1
            step = min(0.06, step * 1.3)
    return best, improvements
