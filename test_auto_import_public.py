import csv
import gzip
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from auto_import_public import (archive_name, imported_archives, run_stamp,
                                select_better, write_delta)
from validate_submission import validate_submission


class AutoImportTests(unittest.TestCase):
    def test_selection_needs_strict_gain_over_grid_and_snapshots(self):
        candidate = {6: 2.9, 11: 3.87, 12: 3.99, 13: 3.95, 26: 5.62}
        baseline = {11: 3.8770835, 12: 3.99, 26: 5.6}
        # 6 and 13 are proof-locked, 12 only ties and 26 is worse.
        self.assertEqual(select_better(candidate, baseline), [11])
        self.assertEqual(select_better({11: 4.0}, {}), [])

    def test_delta_keeps_original_rows_and_validates(self):
        rows = [("id", "n", "square_id", "x", "y", "angle"),
                ("0", "1", "0", "0.5", "0.5", "0.0"),
                ("1", "2", "0", "0.5", "0.5", "0"),
                ("2", "2", "1", "1.5000000001", "0.5", "0")]
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "submission.csv"
            with source.open("w", newline="") as handle:
                csv.writer(handle).writerows(rows)
            first, second = Path(temp) / "a.csv.gz", Path(temp) / "b.csv.gz"
            self.assertEqual(write_delta(source, [2], first), 2)
            write_delta(source, [2], second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with gzip.open(first, "rt", newline="") as handle:
                self.assertEqual([tuple(row) for row in csv.reader(handle)],
                                 [rows[0], rows[2], rows[3]])
            results = validate_submission(first, max_n=2, allow_partial=True)
            self.assertEqual([result.n for result in results], [2])

    def test_import_log_lists_archives_in_order(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "imports.json"
            self.assertEqual(imported_archives(log), [])
            log.write_text(json.dumps([{"archive": "a.csv.gz"},
                                       {"archive": "b.csv.gz"}]))
            self.assertEqual(imported_archives(log),
                             [Path(temp) / "a.csv.gz", Path(temp) / "b.csv.gz"])

    def test_archive_name_includes_owner_slug_and_run(self):
        stamp = run_stamp(datetime(2026, 10, 28, 16, 29, 32, 780000))
        self.assertEqual(stamp, "20261028T162932")
        self.assertEqual(archive_name("owner/some-notebook", stamp),
                         "owner__some-notebook__20261028T162932.csv.gz")
        self.assertEqual(run_stamp(None), "unknown")


if __name__ == "__main__":
    unittest.main()
