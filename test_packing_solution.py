import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from area_gap_report import ranked_gaps
from import_published_packings import checked_side, import_entries
from optimal_locks import check_locked_sides, fixed_packing, proven_optimal
from packing_solution import (five_square_seed, goebel_strip_seed,
                              improved_packings, record_seeds,
                              write_submission)
from validate_submission import Result


class ConstructivePackingTests(unittest.TestCase):
    def test_seed_sizes(self):
        for seed, expected_n, expected_side in (
            (five_square_seed(), 5, 2 + math.sqrt(2) / 2),
            (goebel_strip_seed(5), 27, 5 + math.sqrt(2) / 2),
            (goebel_strip_seed(7), 52, 7 + math.sqrt(2) / 2),
        ):
            side, squares = seed
            self.assertEqual(len(squares), expected_n)
            self.assertAlmostEqual(side, expected_side)

    def test_outer_l_families(self):
        packings = improved_packings(70)
        for n in (5, 10):
            self.assertLess(fixed_packing(n)[0], math.isqrt(n - 1) + 1)
        for n in (17, 27, 38, 52, 67):
            self.assertIn(n, packings)
            self.assertEqual(len(packings[n][1]), n)
            self.assertLess(packings[n][0], math.isqrt(n - 1) + 1)

    def test_proof_registry_does_not_assume_k_squared_minus_2(self):
        for n in (23, 34, 47, 62):
            self.assertIsNone(proven_optimal(n))
        for n in (1, 5, 10, 21, 24, 32, 46, 60, 99, 100, 957):
            self.assertIsNotNone(proven_optimal(n))

    def test_locked_layout_wins_over_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "submission.csv"
            fake = [(0.5, 0.5, 0.0)] * 5
            with patch("packing_solution.improved_packings",
                       return_value={5: (1.0, fake)}):
                write_submission(path, max_n=5)
            rows = [line.split(",") for line in path.read_text().splitlines()[1:]
                    if line.split(",")[1] == "5"]
            actual = [(float(row[3]), float(row[4]), float(row[5])) for row in rows]
            self.assertEqual(actual, fixed_packing(5)[1])

    def test_candidates_never_override_proven_layouts(self):
        overlap = set(improved_packings()) & {
            n for n in range(1, 1001) if proven_optimal(n) is not None
        }
        self.assertEqual(overlap, set())

    def test_locked_side_regression_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "locked n=25"):
            check_locked_sides([Result(25, 5.01, 5)])

    def test_record_seeds_are_valid_and_normalized(self):
        seeds = record_seeds()
        self.assertEqual(len(seeds), 40)
        sizes = {len(squares) for _, squares in seeds}
        self.assertTrue({11, 17, 18, 19, 26, 28, 29, 37, 39, 40, 41,
                         50, 51, 54, 55, 68, 70, 295} <= sizes)
        for side, squares in seeds:
            self.assertGreater(side, 0)
            self.assertTrue(all(x >= 0 and y >= 0 for x, y, _ in squares))

    def test_published_import_validates_and_is_idempotent(self):
        entry = {'source': 'fixture', 'published_side': '2',
                 'squares': [[0.5, 0.5, 0.0], [1.5, 0.5, 0.0]]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'seeds.json'
            path.write_text('{}')
            self.assertEqual(import_entries(path, [(2, entry)]), [2])
            self.assertEqual(import_entries(path, [(2, entry)]), [])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            checked_side(2, [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])

    def test_gap_ranking_excludes_locked_cases(self):
        rows = ranked_gaps([Result(25, 5, 5), Result(11, 4, 4),
                            Result(20, 5, 5)])
        self.assertEqual([row["n"] for row in rows], [11, 20])


if __name__ == "__main__":
    unittest.main()
