import csv
import gzip
import math
import sys
import tempfile
import unittest
from pathlib import Path

# PyCharm may run this file with the parent directory as the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_submission import (
    Result,
    ValidationError,
    make_square,
    squares_overlap,
    total_score,
    validate_submission,
)


class GeometryTests(unittest.TestCase):
    def test_touching_and_overlap(self):
        first = make_square(0, 0, 0, math.pi / 4)
        touching = make_square(1, math.sqrt(2), 0, math.pi / 4)
        overlapping = make_square(1, math.sqrt(2) - 0.01, 0, math.pi / 4)
        self.assertFalse(squares_overlap(first, touching))
        self.assertTrue(squares_overlap(first, overlapping))

    def test_rotation_changes_bounds(self):
        square = make_square(0, 0, 0, math.pi / 4)
        self.assertAlmostEqual(square.max_x - square.min_x, math.sqrt(2))
        self.assertAlmostEqual(square.max_y - square.min_y, math.sqrt(2))


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "submission.csv"

    def write_rows(self, rows):
        with self.path.open("w", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(["id", "n", "square_id", "x", "y", "angle"])
            writer.writerows(rows)

    def test_valid_small_submission(self):
        self.write_rows([[0, 1, 0, 0, 0, 0],
                         [1, 2, 0, 0, 0, 0],
                         [2, 2, 1, 1, 0, 0]])
        results = validate_submission(self.path, max_n=2)
        self.assertEqual([round(result.side) for result in results], [1, 2])

    def test_gzip_submission_uses_same_validation(self):
        rows = [[0, 1, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0],
                [2, 2, 1, 1, 0, 0]]
        self.write_rows(rows)
        path = self.path.with_suffix(".csv.gz")
        with gzip.open(path, "wt", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(["id", "n", "square_id", "x", "y", "angle"])
            writer.writerows(rows)
        self.assertEqual(validate_submission(path, max_n=2),
                         validate_submission(self.path, max_n=2))

    def test_detects_overlap(self):
        self.write_rows([[0, 1, 0, 0, 0, 0],
                         [1, 2, 0, 0, 0, 0],
                         [2, 2, 1, 0.9, 0, 0]])
        with self.assertRaisesRegex(ValidationError, "overlap"):
            validate_submission(self.path, max_n=2)

    def test_detects_duplicate_id(self):
        self.write_rows([[0, 1, 0, 0, 0, 0],
                         [1, 2, 0, 0, 0, 0],
                         [1, 2, 1, 1, 0, 0]])
        with self.assertRaisesRegex(ValidationError, "duplicate id"):
            validate_submission(self.path, max_n=2)

    def test_partial_submission_uses_grid_score_for_missing_n(self):
        self.write_rows([[10, 1, 0, 0, 0, 0]])
        results = validate_submission(self.path, max_n=2, allow_partial=True)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(total_score(results, max_n=2), 2 / math.sqrt(2) - 1)

    def test_score_formula(self):
        result = Result(n=5, side=2.75, baseline_side=3)
        self.assertAlmostEqual(result.score, 2.75 / math.sqrt(5) - 1)
        self.assertAlmostEqual(result.baseline_score - result.score,
                               0.25 / math.sqrt(5))


if __name__ == "__main__":
    unittest.main()
