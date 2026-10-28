import math
import random
import unittest

import numpy as np

from soft_global_search import SEEDS, energy_grad, inflate, near_pairs


class SoftGlobalTests(unittest.TestCase):
    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        vec = np.column_stack([rng.uniform(0, 3, 6), rng.uniform(0, 3, 6),
                               rng.uniform(0, 1.5, 6)]).ravel()
        pairs = near_pairs(vec, 3.0)
        energy, grad = energy_grad(vec, pairs, 3.0, 1.0, 1.0)
        self.assertGreater(energy, 0)
        numeric = np.zeros_like(vec)
        for index in range(len(vec)):
            step = np.zeros_like(vec)
            step[index] = 1e-6
            numeric[index] = (energy_grad(vec + step, pairs, 3.0, 1.0, 1.0)[0] -
                              energy_grad(vec - step, pairs, 3.0, 1.0, 1.0)[0]) / 2e-6
        self.assertLess(np.max(np.abs(numeric - grad)), 1e-7)

    def test_disjoint_layout_has_zero_energy(self):
        vec = np.array([.5, .5, 0, 1.5, .5, 0, .8, 1.8, math.pi / 4]).ravel()
        energy, grad = energy_grad(vec, near_pairs(vec, 3.0), 2.6, 1.0, 1.0)
        self.assertEqual(energy, 0.0)
        self.assertFalse(np.any(grad))

    def test_inflation_recovers_shaken_grid(self):
        rng = np.random.default_rng(1)
        grid = np.array([(x + .5, y + .5, 0.0) for x in range(3) for y in range(3)])
        grid += rng.normal(0, .05, grid.shape)
        _, energy = inflate(grid.ravel(), 3.0 + 1e-6, random.Random(0), steps=4)
        self.assertLess(energy, 1e-12)

    def test_seeds_have_requested_size(self):
        for kind, seed in SEEDS.items():
            points = seed(40, 6.6, random.Random(2))
            self.assertEqual(len(points), 40, kind)


if __name__ == "__main__":
    unittest.main()
