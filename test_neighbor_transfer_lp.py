import unittest

from neighbor_transfer_lp import removal_order, transfer_one
from projected_search import strict_side


class NeighborTransferTests(unittest.TestCase):
    def test_edge_deletions_precede_middle(self):
        donor = [(0.5, 0.5, 0), (1.6, 0.5, 0), (2.7, 0.5, 0)]
        order = removal_order(donor, 3)
        self.assertIn(order[0], (0, 2))
        self.assertEqual(order[-1], 1)

    def test_lp_recloses_after_deletion(self):
        base = [(0.5, 0.5, 0), (1.7, 0.5, 0)]
        donor = [(0.5, 0.5, 0), (1.6, 0.5, 0), (2.7, 0.5, 0)]
        result, stats = transfer_one(base, donor, 4, attempts=3,
                                     joint_rounds=0)
        self.assertGreater(stats["lp_improvements"], 0)
        self.assertLess(strict_side(result), 2.01)


if __name__ == "__main__":
    unittest.main()
