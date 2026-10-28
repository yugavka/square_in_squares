"""Small geometry checks for independent V-band seed construction."""

import unittest

from projected_search import strict_side
from v_band_seed_search import max_refill, v_band


class VBandSeedSearchTest(unittest.TestCase):
    def test_mirrored_band_has_valid_geometry(self):
        core = v_band(12.65, 0.45, rows=3, vertex=5.0, phase=0.5)
        self.assertGreater(len(core), 0)
        self.assertLessEqual(strict_side(core), 12.65)

    def test_max_refill_selects_independent_squares(self):
        candidates = [(0.5, 0.5, 0.0), (1.5, 0.5, 0.0),
                      (0.5, 1.5, 0.0), (1.5, 1.5, 0.0),
                      (0.5, 0.5, 0.0)]
        chosen = max_refill(candidates, seconds=1.0)
        self.assertIsNotNone(chosen)
        self.assertEqual(len(chosen), 4)
        self.assertLessEqual(strict_side(chosen), 2.0)


if __name__ == "__main__":
    unittest.main()
