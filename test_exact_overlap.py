import math
import unittest

from exact_overlap import overlapping, worst_penetration


class ExactOverlapTests(unittest.TestCase):
    def test_grid_contacts_are_allowed(self):
        rows = [("0.5", "0.5", "0"), ("1.5", "0.5", "0"), ("0.5", "1.5", "0")]
        self.assertEqual(worst_penetration(rows), 0)
        self.assertEqual(overlapping({3: rows}), {})

    def test_sub_float_overlap_is_caught(self):
        rows = [("0.5", "0.5", "0"), ("1.4999999999999999", "0.5", "0")]
        self.assertGreater(worst_penetration(rows), 0)
        self.assertIn(2, overlapping({2: rows}))

    def test_tilted_gap_is_separated(self):
        reach = 0.5 + math.sqrt(2) / 2 + 1e-12
        rows = [("0", "0", "0"), (repr(reach), "0", repr(math.pi / 4))]
        self.assertLess(worst_penetration(rows), 0)
        rows = [("0", "0", "0"), (repr(reach - 2e-12), "0", repr(math.pi / 4))]
        self.assertGreater(worst_penetration(rows), 0)


if __name__ == "__main__":
    unittest.main()
