"""Geometry and reproducibility checks for rectangular core refill."""

import unittest
import math

from core_refill_search import search, split_gadget
from fault_line_dp import Gadget
from search_gadget import checked_gadget


class CoreRefillSearchTest(unittest.TestCase):
    def test_refill_and_compaction_preserve_strict_geometry(self):
        overhang = math.sqrt(2) / 2
        side = 2 + overhang
        gadget = Gadget(2, 2, 1, overhang, (
            (side / 2, side / 2, math.pi / 4),
            (0.5, 0.5, 0.0), (side - 0.5, 0.5, 0.0),
            (0.5, side - 0.5, 0.0), (side - 0.5, side - 0.5, 0.0)))
        tilted, upright = split_gadget(gadget)
        self.assertEqual((len(tilted), len(upright)), (1, 4))
        best, stats = search(gadget, trials=1, seed=20261028,
                             mutation_size=0.04, trust=0.18,
                             milp_seconds=3.0)
        self.assertEqual(stats["strict_candidates"], 1)
        checked, _ = checked_gadget(best, list(best.squares))
        self.assertAlmostEqual(checked, best.overhang, places=7)


if __name__ == "__main__":
    unittest.main()
