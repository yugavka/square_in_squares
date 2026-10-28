import unittest

from insert_relax_search import emptiest_windows, free_raster, interaction_pairs


class InsertRelaxTests(unittest.TestCase):
    def test_emptiest_window_finds_missing_grid_cell(self):
        layout = [(x + .5, y + .5, 0.0) for x in range(3) for y in range(3)
                  if (x, y) != (1, 2)]
        (x, y, angle), = emptiest_windows(layout, 3.0, 1)
        self.assertAlmostEqual(x, 1.5)
        self.assertAlmostEqual(y, 2.5)
        self.assertEqual(angle, 0.0)

    def test_raster_marks_tilted_square(self):
        free = free_raster([(1.0, 1.0, 0.7853981633974483)], 2.0, .1)
        self.assertFalse(free[10, 10])
        self.assertTrue(free[0, 0])

    def test_inserted_square_pairs_reach_far_neighbours(self):
        layout = [(.5, .5, 0.0), (4.5, .5, 0.0), (8.5, .5, 0.0)]
        self.assertEqual(interaction_pairs(layout, set(), .1, 2.0), [])
        self.assertEqual(interaction_pairs(layout, {1}, .1, 2.0),
                         [(0, 1), (1, 2)])


if __name__ == "__main__":
    unittest.main()
