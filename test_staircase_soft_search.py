"""Checks for full-seed reconstruction and the soft contact gradient."""

import math
import unittest

import numpy as np

from projected_search import strict_side
from staircase_soft_search import rebuild_staircases, soft_energy_gradient


class StaircaseSoftSearchTest(unittest.TestCase):
    def test_rebuild_preserves_complete_valid_packing(self):
        side = 2 + math.sqrt(2) / 2
        packing = [(side / 2, side / 2, math.pi / 4),
                   (0.5, 0.5, 0.0), (side - 0.5, 0.5, 0.0),
                   (0.5, side - 0.5, 0.0),
                   (side - 0.5, side - 0.5, 0.0)]
        rebuilt = rebuild_staircases(packing, maximum=40, milp_seconds=1)
        self.assertIsNotNone(rebuilt)
        self.assertEqual(len(rebuilt), 5)
        self.assertAlmostEqual(strict_side(rebuilt), side, places=6)

    def test_soft_gradient_matches_finite_difference(self):
        vector = np.array([2.0, 2.0, 0.27, 2.75, 2.12, -0.39])
        reference = vector.copy()
        for square_size in (0.95, 1.0):
            args = (reference, [(0, 1)], 5.0, 10.0, 0.03,
                    1e-4, square_size)
            _, gradient = soft_energy_gradient(vector, *args)
            for index in range(len(vector)):
                step = 1e-6
                plus, minus = vector.copy(), vector.copy()
                plus[index] += step
                minus[index] -= step
                numeric = (soft_energy_gradient(plus, *args)[0] -
                           soft_energy_gradient(minus, *args)[0]) / (2 * step)
                self.assertAlmostEqual(gradient[index], numeric, places=5)


if __name__ == "__main__":
    unittest.main()
