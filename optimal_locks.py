"""Proof-backed layouts that the search must leave unchanged.

Proof status is recorded separately from the achieved numerical side. The
Kaggle CSV for n=5 and n=10 uses a tiny safety spacing, so its measured side
is slightly larger than the exact mathematical optimum; the underlying
construction is the proved optimal one.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


SquarePlacement = tuple[float, float, float]
Packing = tuple[float, list[SquarePlacement]]
SQRT2 = math.sqrt(2)
SAFETY_SCALE = 1 + 1e-12

CATALOG = "https://kingbird.myphotos.cc/packing/squares_in_squares.html"
KARAKUS = ("https://github.com/jlevy/squares/blob/main/packing/resources/papers/"
           "karakus-2026-counterexample-nagamochi-scoring-lemma.md")
ATLAS = "https://evand.github.io/square-packing/proofs.html"


@dataclass(frozen=True, slots=True)
class ProofRecord:
    n: int
    exact_side: float
    family: str
    evidence: str
    source: str


def _square_root_if_square(value: int) -> int | None:
    if value < 0:
        return None
    root = math.isqrt(value)
    return root if root * root == value else None


def proven_optimal(n: int) -> ProofRecord | None:
    """Return a proof record only where the current layout reaches its bound."""
    if n < 1:
        return None
    k = _square_root_if_square(n)
    if k is not None:
        return ProofRecord(n, float(k), "perfect_square", "area_bound", CATALOG)

    if n in (5, 10):
        k = 2 if n == 5 else 3
        return ProofRecord(n, k + SQRT2 / 2, "Goebel_construction",
                           "classical_proof", CATALOG)

    k = _square_root_if_square(n + 1)
    if k is not None and k >= 2:
        return ProofRecord(n, float(k), "k_squared_minus_1",
                           "independent_proof_2026", KARAKUS)

    # Older proofs independent of the incomplete Nagamochi k²-2 argument.
    if n in (2, 6, 7, 13, 14, 22, 33, 46):
        side = math.isqrt(n - 1) + 1
        return ProofRecord(n, float(side), "classical_grid_case",
                           "classical_proof", CATALOG)

    k = _square_root_if_square(n + 3)
    if k is not None and k >= 6:
        return ProofRecord(n, float(k), "k_squared_minus_3",
                           "computer_checked_unreviewed", ATLAS)

    k = _square_root_if_square(n + 4)
    if k is not None and k >= 5:
        return ProofRecord(n, float(k), "k_squared_minus_4",
                           "computer_checked_unreviewed", ATLAS)

    return None


def five_square_seed() -> Packing:
    """The exact four-corners-plus-diamond construction, before safety spacing."""
    side = 2 + SQRT2 / 2
    corner = side - 0.5
    return side, [
        (0.5, 0.5, 0.0),
        (corner, 0.5, 0.0),
        (0.5, corner, 0.0),
        (corner, corner, 0.0),
        (1 + SQRT2 / 4, 1 + SQRT2 / 4, math.pi / 4),
    ]


def fixed_packing(n: int) -> Packing:
    """Return the canonical, fixed coordinates for a proof-backed puzzle."""
    record = proven_optimal(n)
    if record is None:
        raise ValueError(f"n={n} has no verified optimal layout in this registry")
    if n in (5, 10):
        side, squares = five_square_seed()
        if n == 10:
            squares = list(squares)
            squares.extend((side + 0.5, j + 0.5, 0.0) for j in range(2))
            squares.extend((i + 0.5, side + 0.5, 0.0) for i in range(3))
            side += 1
        return (side * SAFETY_SCALE,
                [(x * SAFETY_SCALE, y * SAFETY_SCALE, angle)
                 for x, y, angle in squares])

    k = int(record.exact_side)
    return float(k), [(i % k + 0.5, i // k + 0.5, 0.0) for i in range(n)]


def check_locked_sides(results: list) -> None:
    """Catch any future generator change that worsens a proof-backed layout."""
    for result in results:
        record = proven_optimal(result.n)
        if record is None:
            continue
        # Only the two irrational constructions have intentional safety space.
        allowed_margin = 1e-6 if result.n in (5, 10) else 1e-9
        if result.side > record.exact_side + allowed_margin:
            raise ValueError(
                f"locked n={result.n}: side {result.side:.12f} exceeds "
                f"proved optimum {record.exact_side:.12f}"
            )


def write_lock_manifest(path: Path, max_n: int = 1000) -> int:
    """Write a reviewable list of fixed puzzles and their proof sources."""
    records = [record for n in range(1, max_n + 1)
               if (record := proven_optimal(n)) is not None]
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(["n", "exact_optimum_side", "submitted_side",
                         "family", "evidence", "source"])
        for record in records:
            submitted_side = fixed_packing(record.n)[0]
            writer.writerow([record.n, format(record.exact_side, ".15g"),
                             format(submitted_side, ".15g"),
                             record.family, record.evidence, record.source])
    return len(records)
