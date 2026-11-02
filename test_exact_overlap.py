import math
import tempfile
import unittest
from pathlib import Path

from exact_overlap import overlapping, read_text, worst_penetration


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

    def test_gap_lost_in_float64_parsing_is_caught(self):
        # 1e-30 clear in the text; the float64 texts are 1 - 1e-15 apart.
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sub.csv"
            path.write_text("id,n,square_id,x,y,angle\n0,2,0,0.6718212205620061,0.5,0\n"
                            "1,2,1,1.671821220562006100000000000001,0.5,0\n")
            exact = read_text(path, as_float=False)
            parsed = read_text(path)
        self.assertEqual(overlapping(exact), {})
        self.assertIn(2, overlapping(parsed))

if __name__ == "__main__":
    unittest.main()
