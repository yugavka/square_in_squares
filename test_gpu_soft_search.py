import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gpu_soft_search
from gpu_soft_search import build, slug_for

ROOT = Path(__file__).resolve().parent


class GpuSoftSearchTests(unittest.TestCase):
    def test_slug_names_single_and_multi_n_kernels(self):
        self.assertEqual(slug_for([147], ""), "soft-gpu-147")
        self.assertEqual(slug_for([147], "v2"), "soft-gpu-147-v2")
        self.assertEqual(slug_for([147, 172], "batch"), "soft-gpu-batch")

    def test_build_embeds_jobs_and_private_gpu_metadata(self):
        with tempfile.TemporaryDirectory() as temp, \
                mock.patch.object(gpu_soft_search, "BUILD", Path(temp)):
            folder = build([147, 172], None, [.002, .005], 5.0, 16, ["incumbent"],
                           3, 1e-4, 4, "soft-gpu-test", "someone",
                           ROOT / "submission_improved.csv")
            metadata = json.loads((folder / "kernel-metadata.json").read_text())
            self.assertEqual(metadata["id"], "someone/soft-gpu-test")
            self.assertTrue(metadata["is_private"])
            self.assertTrue(metadata["enable_gpu"])
            code = (folder / "soft_gpu.py").read_text()
            tree = ast.parse(code)
            config = next(ast.literal_eval(node.value) for node in tree.body
                          if isinstance(node, ast.Assign)
                          and getattr(node.targets[0], "id", "") == "CONFIG")
        self.assertEqual([job["n"] for job in config["jobs"]], [147, 172])
        first = config["jobs"][0]
        self.assertEqual(len(first["incumbent"]), 147)
        self.assertAlmostEqual(first["targets"][0], first["incumbent_side"] - .002, places=8)
        self.assertEqual((config["batch"], config["top"], config["seed"]), (16, 4, 3))


if __name__ == "__main__":
    unittest.main()
