import math
import unittest

from fault_line_dp import Gadget
from packing_solution import FaultTrace
from report_fault_line import build_report


class ReportTests(unittest.TestCase):
    def test_shared_gain_is_allocated_once_and_lineage_is_preserved(self):
        a = Gadget(2, 2, 1, 0.4, ())
        b = Gadget(3, 3, 1, 0.3, ())
        # Synthetic sides isolate the accounting; no geometric claim is made.
        baseline = {11: (4.0, []), 12: (4.0, [])}
        actual = {11: (3.9, []), 12: (3.8, [])}
        trace = {11: FaultTrace(12, (a, b), ("trim 12->11",)),
                 12: FaultTrace(12, (a, a, b))}
        report = build_report(baseline, actual, trace)
        expected = 0.1 / math.sqrt(11) + 0.2 / math.sqrt(12)
        self.assertAlmostEqual(report["score_gain"], expected)
        self.assertAlmostEqual(sum(g["allocated_gain"] for g in report["gadgets"]),
                               expected)
        self.assertEqual(report["direct_n"], 1)
        self.assertEqual(report["derived_n"], 1)
        gadget_a = next(g for g in report["gadgets"] if g["w"] == 2)
        self.assertEqual(gadget_a["n"], [11, 12])
        self.assertEqual(gadget_a["occurrences"], 3)
        self.assertEqual(gadget_a["unique_bottleneck_uses"], 1)


if __name__ == "__main__":
    unittest.main()
