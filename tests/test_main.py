"""main.py end to end: DALI check, dry run, the real-data overfit path and the OVERFIT test.

The overfit test (milestone 4 verification): one fixed synthetic group (smooth asymmetric
phantom, per-image random rigid transforms up to +-45 deg / +-15 mm, N = 4, level 4), narrow
model (default dense_norm 'layer'), Adam 3e-4 constant LR, no flips, no dropout. It must reach a
10-step mean training-step centered RMS below OVERFIT_STOP_MM within 1500 steps, and then the
eval-mode validation of the same group must be below 1.0 mm and within 0.5 mm of the
training-mode (no update) value. The RMS curve is printed every 50 steps.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests.runtime_helpers import free_memory, write_fake_export, NARROW_ARGS
import numpy as np

import main as M
from data_loading import dali_loader

OVERFIT_LR = "3e-4"
# Stop a little below 1 mm: the training-step RMS is measured before each update and is noisy
# under a constant LR, so the eval-mode check at the stopping step gets some margin.
OVERFIT_STOP_MM = "0.8"
COMMON = ["--log-backend", "jsonl", "--quiet", "true", "--seed", "0"]


class MainCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_m4_main_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        free_memory()

    def run_dir(self, name):
        return os.path.join(self.tmp, "results", name)


class TestMain(MainCase):
    @unittest.skipIf(dali_loader.DALI_AVAILABLE, "DALI is available here")
    def test_dry_run_without_dali_fails_clearly(self):
        with self.assertRaises(SystemExit) as cm:
            M.main(["--dry-run", "--data", self.tmp, "--results", self.tmp])
        msg = str(cm.exception.code)
        self.assertIn("nvidia.dali", msg)
        self.assertIn("--overfit-one-group", msg)

    def test_synthetic_overfit_dry_run(self):
        rc = M.main(["--overfit-one-group", "--overfit-synthetic", "--dry-run", "--group-size", "4",
                     "--results", os.path.join(self.tmp, "results"), "--run-name", "dry"]
                    + NARROW_ARGS + COMMON)
        self.assertEqual(rc, 0)
        d = self.run_dir("dry")
        with open(os.path.join(d, "config.json")) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["model_config"]["filters"], [16, 32, 32, 48, 64, 96])
        with open(os.path.join(d, "logs", "metrics.jsonl")) as f:
            lines = [json.loads(line) for line in f]
        self.assertEqual(lines[0]["step"], "PARAMETER")
        self.assertFalse(lines[0]["data"]["tf32"])
        steps = [r["step"] for r in lines[1:]]
        self.assertIn(2, steps)
        self.assertTrue(any("val_rigid/pair_rms_mm" in r["data"] for r in lines[1:]))
        self.assertTrue(any(n.startswith("last-2") for n in os.listdir(os.path.join(d, "ckpt", "last"))))

    def test_real_data_overfit_path(self):
        root = write_fake_export(self.tmp, n_train=5, n_val=2)
        rc = M.main(["--overfit-one-group", "--data", root, "--group-size", "4", "--overfit-max-steps", "20",
                     "--log-every", "10", "--val-every", "10", "--results", os.path.join(self.tmp, "results"),
                     "--run-name", "real", "--optimizer", "adam", "--lr", "1e-4", "--schedule", "constant",
                     "--warmup-steps", "0"] + NARROW_ARGS + COMMON)
        self.assertIn(rc, (0, 1))
        with open(os.path.join(self.run_dir("real"), "overfit_summary.json")) as f:
            s = json.load(f)
        self.assertEqual(s["steps_run"], 20)
        with open(os.path.join(self.run_dir("real"), "config.json")) as f:
            self.assertIn("train images", json.load(f)["data"])
        self.assertEqual(len(s["val"]), 2)
        self.assertTrue(np.isfinite(s["val"][-1]["val_rigid/center_rms_mm"]))
        # resume: same run name continues from the last checkpoint (step 20)
        rc = M.main(["--overfit-one-group", "--data", root, "--group-size", "4", "--overfit-max-steps", "25",
                     "--log-every", "5", "--val-every", "0", "--results", os.path.join(self.tmp, "results"),
                     "--run-name", "real", "--optimizer", "adam", "--lr", "1e-4", "--schedule", "constant",
                     "--warmup-steps", "0"] + NARROW_ARGS + COMMON)
        with open(os.path.join(self.run_dir("real"), "overfit_summary.json")) as f:
            s = json.load(f)
        self.assertEqual((s["steps_run"], s["end_step"]), (5, 25))


class TestOverfit(MainCase):
    def test_overfit_synthetic_group_below_1mm(self):
        rc = M.main(["--overfit-one-group", "--overfit-synthetic", "--level", "4", "--group-size", "4",
                     "--optimizer", "adam", "--lr", OVERFIT_LR, "--schedule", "constant", "--warmup-steps", "0",
                     "--overfit-max-steps", "1500", "--overfit-threshold-mm", OVERFIT_STOP_MM, "--log-every", "100",
                     "--val-every", "250", "--results", os.path.join(self.tmp, "results"), "--run-name", "overfit"]
                    + NARROW_ARGS + COMMON)
        with open(os.path.join(self.run_dir("overfit"), "overfit_summary.json")) as f:
            s = json.load(f)
        curve = s["curve"]
        rows = ["step %4d  center RMS %7.3f mm  pair RMS %7.3f mm" % tuple(c) for c in curve
                if c[0] == 1 or c[0] % 50 == 0 or c[0] == curve[-1][0]]
        sys.stderr.write("\n[overfit curve] lr %s, steps %d, reached %s, best 10-step mean %.4f mm, "
                         "median step %.0f ms\n  %s\n" % (OVERFIT_LR, s["steps_run"], s["reached"],
                                                       s["best_window_center_rms_mm"], s["median_compute_ms"],
                                                       "\n  ".join(rows)))
        sys.stderr.write("[overfit val] step, eval-mode center RMS, train-mode center RMS (mm):\n  %s\n" % (
            "\n  ".join("%4d  %7.3f  %7.3f" % tuple(v) for v in s["val_curve"])))
        self.assertTrue(s["reached"], "best 10-step mean %.3f mm" % s["best_window_center_rms_mm"])
        self.assertEqual(rc, 0)
        self.assertLessEqual(s["steps_run"], 1500)
        # Eval mode (the source of truth for 'best') must reach the target too, and agree with
        # training-mode BN on the same group (v3.4: dense path LayerNorm, no group-relative BN).
        ev = s["final_val"]["val_rigid/center_rms_mm"]
        tm = s["final_val"]["val_rigid_trainmode/center_rms_mm"]
        self.assertLess(ev, 1.0, "eval-mode center RMS %.3f mm (train mode %.3f mm)" % (ev, tm))
        self.assertLess(abs(ev - tm), 0.5)


if __name__ == "__main__":
    unittest.main()
