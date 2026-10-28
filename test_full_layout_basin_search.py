"""Regression checks for full-layout basin hopping and strict export."""

import random
import tempfile
import unittest
from pathlib import Path

from full_layout_basin_search import collective_mutation, search_one
from projected_search import strict_side, write_partial
from validate_submission import validate_submission


class FullLayoutBasinSearchTest(unittest.TestCase):
    def test_collective_search_returns_legal_full_layout(self):
        base = [(0.6, 0.5, 0.0), (1.7, 0.5, 0.0)]
        before = strict_side(base)
        result, stats = search_one(base, trials=4, seed=17, quench_rounds=3)
        self.assertLess(strict_side(result), before - 0.01)
        self.assertGreater(stats["repaired"], 0)
        self.assertGreater(stats["uphill_moves"], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.csv"
            write_partial(path, {2: result})
            self.assertLess(validate_submission(path, max_n=2,
                                                allow_partial=True)[0].side,
                            before)

    def test_cluster_mutation_crosses_contacts(self):
        base = [(x + 0.5, y + 0.5, 0.0)
                for x in range(4) for y in range(4)]
        changed = collective_mutation(base, random.Random(3), 0.11, 0)
        self.assertEqual(len(changed), len(base))
        self.assertTrue(any(a != b for a, b in zip(base, changed)))
        with self.assertRaises(ValueError):
            strict_side(changed)


if __name__ == "__main__":
    unittest.main()
