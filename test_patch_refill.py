import random
import unittest

from patch_refill import candidate_positions, milp_refill, select_patch
from projected_search import strict_side


class PatchRefillTests(unittest.TestCase):
    def test_refill_finds_missing_grid_cell(self):
        survivors = [(0.5, 0.5, 0.0), (1.5, 0.5, 0.0),
                     (0.5, 1.5, 0.0)]
        removed = [(1.5, 1.5, 0.0)]
        candidates = candidate_positions(survivors, removed, 2.000001)
        self.assertTrue(candidates)
        chosen = milp_refill(candidates, 1)
        self.assertIsNotNone(chosen)
        self.assertLess(strict_side(survivors + chosen), 2.000001)

    def test_milp_rejects_pairwise_conflicts(self):
        candidates = [(0.5, 0.5, 0.0), (0.6, 0.5, 0.0)]
        self.assertIsNone(milp_refill(candidates, 2))

    def test_patch_selection_keeps_square_count(self):
        layout = [(x + 0.5, y + 0.5, 0.0)
                  for x in range(4) for y in range(4)]
        survivors, removed = select_patch(layout, 5, random.Random(1), 0)
        self.assertEqual((len(survivors), len(removed)), (11, 5))
        self.assertEqual(len(set(survivors + removed)), 16)


if __name__ == "__main__":
    unittest.main()
