import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from import_published_packings import checked_side
from squeeze_layout import compact_centres, separating_axis


class SqueezeTests(unittest.TestCase):
    def test_separating_axis_recognizes_contact(self):
        margin, *_ = separating_axis((0.5, 0.5, 0.0),
                                     (1.5, 0.5, 0.0))
        self.assertAlmostEqual(margin, 0.0)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy unavailable")
    def test_linear_compaction_of_spaced_pair(self):
        initial = [(0.5, 0.5, 0.0), (3.5, 0.5, 0.0)]
        compacted = compact_centres(initial)
        self.assertIsNotNone(compacted)
        self.assertLess(checked_side(2, [list(row) for row in compacted]), 2.001)


if __name__ == "__main__":
    unittest.main()
