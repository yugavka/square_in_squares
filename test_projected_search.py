import random
import unittest

from projected_search import (propose, restore_lp, strict_side,
                              normalized, write_partial)
from joint_polish import polish
from validate_submission import validate_submission


class ProjectedSearchTests(unittest.TestCase):
    def test_lp_restores_an_overlapping_pair(self):
        overlapping = [(0.5, 0.5, 0.0), (1.4, 0.5, 0.0)]
        with self.assertRaises(ValueError):
            strict_side(overlapping)
        repaired = restore_lp(overlapping, 2.1, 0.2, random.Random(1))
        self.assertIsNotNone(repaired)
        self.assertLess(strict_side(repaired), 2.1)

    def test_proposals_can_cross_a_contact_before_repair(self):
        touching = [(0.5, 0.5, 0.0), (1.5, 0.5, 0.0)]
        proposal = propose(touching, random.Random(2), 0.03, 0)
        with self.assertRaises(ValueError):
            strict_side(proposal)
        repaired = restore_lp(proposal, 2.1, 0.2, random.Random(3))
        self.assertIsNotNone(repaired)

    def test_exported_partial_csv_passes_validator(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.csv"
            write_partial(path, {2: [(0.5, 0.5, 0.0),
                                     (1.50000001, 0.5, 0.0)]})
            results = validate_submission(path, max_n=2, allow_partial=True)
            self.assertEqual([result.n for result in results], [2])

    def test_joint_polish_changes_angles_and_reduces_side(self):
        tilted = [(0.6, 0.6, 0.08), (1.7, 0.6, 0.08)]
        compact = restore_lp(tilted, 2.3, 0.3, random.Random(1))
        before = strict_side(compact)
        polished, improvements = polish(compact, 20, strict_side, normalized)
        self.assertGreater(improvements, 0)
        self.assertLess(strict_side(polished), before - 1e-6)
        self.assertTrue(any(abs(a[2] - b[2]) > 1e-7
                            for a, b in zip(compact, polished)))


if __name__ == "__main__":
    unittest.main()
