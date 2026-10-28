import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fault_line_dp import extract_corner_crops
from packing_solution import record_seeds
from rectangle_pair_search import compose_shifted, search_pairs, shifted_seam_compatible


class RectanglePairSearchTests(unittest.TestCase):
    def test_valid_pair_is_found_and_shift_preserves_validity(self):
        donor = next(packing for packing in record_seeds()
                     if len(packing[1]) == 11)
        gadgets = extract_corner_crops([donor], variants_per_signature=3)
        first = next(g for g in gadgets if (g.w, g.h, g.surplus) == (3, 2, 1))
        second = next(g for g in gadgets if (g.w, g.h, g.surplus) == (2, 3, 1))
        self.assertTrue(shifted_seam_compatible(first, second, 0.01))
        side, rows = compose_shifted(first, second, 0.01)
        self.assertEqual(len(rows), 27)
        self.assertLess(side, 6)
        best, report = search_pairs([first, second], {27: 6.0})
        self.assertIn(27, best)
        self.assertGreater(report["validated_pairs"], 0)

    def test_rejects_nonpositive_variant_count(self):
        with self.assertRaises(ValueError):
            extract_corner_crops([], variants_per_signature=0)


if __name__ == "__main__":
    unittest.main()
