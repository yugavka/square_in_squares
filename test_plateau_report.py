import unittest

from plateau_report import plateau_rows, shell


class PlateauReportTests(unittest.TestCase):
    def test_shell_offsets(self):
        self.assertEqual(shell(26), (6, 1))
        self.assertEqual(shell(36), (6, 11))
        self.assertEqual(shell(37), (7, 1))

    def test_transferred_cells_are_flagged_with_carry_weight(self):
        # j=2 has frac 0.6 in shells 6, 7 and 8, then improves in shell 9.
        sides = {27: 5.6, 38: 6.6, 51: 7.6, 66: 8.5}
        rows = {row["n"]: row for row in plateau_rows(sides, max_n=66)}
        self.assertEqual(sorted(rows), [38, 51])
        self.assertTrue(rows[51]["last_in_plateau"])
        self.assertFalse(rows[38]["last_in_plateau"])
        self.assertEqual(rows[51]["plateau_from_K"], 6)
        self.assertAlmostEqual(rows[38]["transfer_weight"],
                               1 / 38 ** 0.5 + 1 / 51 ** 0.5)

    def test_grid_cells_are_ignored(self):
        self.assertEqual(plateau_rows({27: 6.0, 38: 7.0}, max_n=38), [])


if __name__ == "__main__":
    unittest.main()
