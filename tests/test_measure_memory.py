"""T15 (milestone 8, design section 11): tests/gpu/measure_level_memory.py runs end to end on a
tiny level-2 cascade configuration (GPU only; skipped without a GPU).

The local GTX 1080 runs under the 6000 MB cap: a level-2 TINY deform step with T = 7 and no
checkpointing does not fit there (FEAT-002), so the test passes --integration-steps 3 and
--gradient-checkpointing both to get a measured 'ok' result; 'oom' is still accepted.
"""
import tests.tf_setup  # noqa: F401  (TF32 off, GPU memory setup) -- must come first

import importlib.util
import json
import os
import shutil
import tempfile
import unittest

import tensorflow as tf

from tests.runtime_helpers import free_memory

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu", "measure_level_memory.py")
_spec = importlib.util.spec_from_file_location("measure_level_memory", _PATH)
mlm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mlm)


@unittest.skipIf(not tf.config.list_physical_devices("GPU"), "needs a GPU")
class TestMeasureLevelMemory(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mlm_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        tf.keras.mixed_precision.set_global_policy("float32")
        free_memory()

    def test_tiny_level2_cascade(self):
        out = os.path.join(self.tmp, "memory.json")
        rc = mlm.main(["--level", "2", "--tasks", "deform,rigid", "--n", "2", "--widths", "tiny",
                       "--cascade", "true", "--gradient-checkpointing", "both", "--integration-steps", "3",
                       "--steps", "2", "--warmup-steps", "1", "--val-pass", "true",
                       "--gpu-memory-limit-mb", "6000", "--out", out])
        self.assertEqual(rc, 0)
        with open(out) as f:
            res = json.load(f)["results"]
        by_task = {r["task"]: r for r in res}
        self.assertEqual(sorted(by_task), ["deform", "rigid"])
        for r in res:
            print("[T15] %s status %s peak_mb %s warmup_peak_mb %s resident_mb %s step_ms %s frozen_ms %s "
                  "val_status %s val_peak_mb %s val_ms %s" % (
                      r["task"], r["status"], r.get("peak_mb"), r.get("warmup_peak_mb"), r.get("resident_mb"),
                      r.get("step_ms"), r.get("frozen_ms"), r.get("val_status"), r.get("val_peak_mb"),
                      r.get("val_ms")))
            self.assertIn(r["status"], ("ok", "oom"), r.get("error"))
            self.assertEqual((r["level"], r["n"], r["widths"], r["cascade"]), (2, 2, "tiny", True))
            if r["status"] == "ok":
                for k in ("peak_mb", "warmup_peak_mb", "resident_mb", "step_ms", "step_ms_mean", "frozen_ms"):
                    self.assertIsInstance(r[k], float, k)
                self.assertEqual(len(r["step_ms_all"]), 2)
                self.assertTrue(r["finite"])
                self.assertEqual(r["trace_counts"][r["task"]], 1)
        d, rg = by_task["deform"], by_task["rigid"]
        self.assertEqual(d["gradient_checkpointing"], "both")
        self.assertEqual(rg["gradient_checkpointing"], "none")
        self.assertIn(d["val_status"], ("ok", "oom"), d.get("val_error"))
        if d["val_status"] == "ok":
            self.assertIsInstance(d["val_peak_mb"], float)
            self.assertIsInstance(d["val_ms"], float)
        self.assertIsNone(rg["val_peak_mb"])
        self.assertIsNone(rg["val_status"])


if __name__ == "__main__":
    unittest.main()
