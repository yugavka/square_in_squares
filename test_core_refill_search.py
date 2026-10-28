"""Geometry and reproducibility checks for rectangular core refill."""

import unittest
import math

from core_refill_search import repair_core, search, split_gadget
from fault_line_dp import Gadget
from patch_refill import milp_refill
from search_gadget import checked_gadget
from validate_submission import make_square, measure_packing


class CoreRefillSearchTest(unittest.TestCase):
    def test_small_temporary_overlap_can_be_repaired(self):
        candidates = [(0.5, 0.5, 0.0), (1.49, 0.5, 0.0)]
        self.assertIsNone(milp_refill(candidates, 2, overlap_tolerance=0.0))
        selected = milp_refill(candidates, 2, overlap_tolerance=0.02)
        self.assertIsNotNone(selected)
        restored = repair_core(selected, 2.0, 1.0, trust=0.03)
        self.assertIsNotNone(restored)
        measure_packing(2, [make_square(i, *row)
                            for i, row in enumerate(restored)])

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
                             milp_seconds=3.0, polish_rounds=3)
        self.assertEqual(stats["strict_candidates"], 1)
        checked, _ = checked_gadget(best, list(best.squares))
        self.assertAlmostEqual(checked, best.overhang, places=7)


if __name__ == "__main__":
    unittest.main()
