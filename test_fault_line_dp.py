import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fault_line_dp import (Gadget, compose, extract_corner_crops,
                           has_empty_upper_tip, load_gadget_overrides,
                           orient, seam_compatible)
from packing_solution import five_square_seed, record_seeds
from validate_submission import ValidationError, make_square, measure_packing


class FaultLineTests(unittest.TestCase):
    def test_saved_override_preserves_required_seam(self):
        gadget = next(g for g in load_gadget_overrides()
                      if (g.w, g.h, g.surplus) == (22, 10, 15))
        self.assertTrue(has_empty_upper_tip(gadget))
        self.assertLess(gadget.overhang, 0.917305674)

    def test_rectangular_crops_from_known_packing(self):
        donor = next(packing for packing in record_seeds()
                     if len(packing[1]) == 11)
        gadgets = extract_corner_crops([donor])
        crop = next(g for g in gadgets if (g.w, g.h, g.surplus) == (3, 2, 1))
        self.assertTrue(has_empty_upper_tip(crop))
        self.assertEqual(len(crop.squares), 7)
        measured = measure_packing(7, [make_square(i, *row)
                                       for i, row in enumerate(crop.squares)])
        self.assertLessEqual(measured.side, 3 + crop.overhang + 1e-8)

    def test_composition_checks_seam_and_count(self):
        side, rows = five_square_seed()
        full = Gadget(2, 2, 1, side - 2, tuple(rows))
        self.assertFalse(seam_compatible(full, full))
        with self.assertRaises(ValidationError):
            compose((full, full))

        donor = next(packing for packing in record_seeds()
                     if len(packing[1]) == 11)
        gadgets = extract_corner_crops([donor])
        first = next(g for g in gadgets if (g.w, g.h, g.surplus) == (3, 2, 1))
        second = next(g for g in gadgets if (g.w, g.h, g.surplus) == (2, 3, 1))
        self.assertTrue(seam_compatible(first, second))
        packed_side, packed = compose((first, second))
        self.assertEqual(len(packed), 27)
        self.assertLess(packed_side, 6)
        self.assertAlmostEqual(packed_side, 5 + first.overhang, places=5)

    def test_symmetry_preserves_validity(self):
        side, rows = five_square_seed()
        rotated = orient(rows, turns=2, mirror=True)
        measured = measure_packing(5, [make_square(i, *row)
                                       for i, row in enumerate(rotated)])
        self.assertAlmostEqual(measured.side, side)

    def test_three_gadget_rectangular_chain(self):
        gadgets = extract_corner_crops(record_seeds())
        def find(w, h):
            return next(g for g in gadgets
                        if (g.w, g.h, g.surplus) == (w, h, 1)
                        and has_empty_upper_tip(g))

        first, middle, last = find(3, 2), find(2, 3), find(3, 3)
        last = Gadget(last.w, last.h, last.surplus, last.overhang,
                      orient(last.squares, turns=2))
        side, packed = compose((first, middle, last))
        self.assertEqual(len(packed), 67)
        self.assertLess(side, 9)


if __name__ == "__main__":
    unittest.main()
