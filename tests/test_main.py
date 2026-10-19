"""main.py end to end: DALI check, dry run, the real-data overfit path and the OVERFIT test.

Milestone 5 adds: the affine overfit (full affine ranges, < 1 mm eval mode), the mixed
rigid:affine overfit on two fixed groups (both < 1 mm; tests/gpu/check_mixed_run.py on its run
dir), the resumed step's control values before the pipelines start, and --init-weights.

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
        self.assertEqual(os.listdir(os.path.join(d, "val_png")), ["rigid_step0000002.png"])   # --val-png default on
        free_memory()
        rc = M.main(["--overfit-one-group", "--overfit-synthetic", "--dry-run", "--group-size", "4",
                     "--results", os.path.join(self.tmp, "results"), "--run-name", "dry_nopng",
                     "--val-png", "false"] + NARROW_ARGS + COMMON)
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.exists(os.path.join(self.run_dir("dry_nopng"), "val_png")))

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


class _Stub:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class TestResumeControlsAndWarmStart(MainCase):
    def test_resumed_step_control_values_before_pipelines(self):
        # Review-curriculum issue 5: _build_dali_data writes the start step's values (stub data
        # module records control_init; no DALI needed).
        from data_loading import params as P
        from runtime import args as A
        seen = {}

        class FakeDM:
            def __init__(self, *a, **kw):
                seen.update(kw)
                self.control_writers = {t: None for t in kw.get("control_init", {})}
                self.files, self.template_dir = ["f"], "t"

            def iterator(self, task):
                return iter(())

        fd = _Stub(load_level_images=lambda *a, **k: (np.zeros((8,) + P.LEVEL_SHAPE[4] + (1,)), []),
                   build_validation_sets=lambda *a, **k: {})
        pp = _Stub(brain_S_matrix=lambda *a, **k: np.eye(4))
        dm = _Stub(LinearDataModule=FakeDM)
        args = A.parse_args(["--data", self.tmp, "--results", self.tmp, "--run-name", "r", "--curriculum", "true",
                             "--curriculum-ramp-steps", "100", "--global-transform", "true"])
        for start in (0, 60, 500):
            M._build_dali_data(args, 4, (fd, P, None, pp, dm), start_step=start)
            ctl = A.task_controls_from_args(args)["rigid"]
            self.assertEqual(seen["control_init"]["rigid"], ctl.values_at(start))
        self.assertAlmostEqual(seen["control_init"]["rigid"]["rotation_deg"], 45.0)
        M._build_dali_data(args, 4, (fd, P, None, pp, dm), start_step=60)
        self.assertAlmostEqual(seen["control_init"]["rigid"]["rotation_deg"], 15.0 + 0.6 * 30.0)

    def test_init_weights_warm_start(self):
        res = os.path.join(self.tmp, "results")
        base = ["--overfit-one-group", "--overfit-synthetic", "--dry-run", "--group-size", "4",
                "--results", res, "--val-png", "false"] + NARROW_ARGS + COMMON
        self.assertEqual(M.main(base + ["--run-name", "src"]), 0)
        free_memory()
        src_ckpt = os.path.join(self.run_dir("src"), "ckpt", "last")
        self.assertEqual(M.main(base + ["--run-name", "warm", "--init-weights", src_ckpt]), 0)
        with open(os.path.join(self.run_dir("warm"), "config.json")) as f:
            cfg = json.load(f)
        self.assertTrue(cfg["init_weights"]["checkpoint"].endswith("last-2"))
        self.assertTrue(cfg["init_weights"]["source_config"].endswith(os.path.join("src", "config.json")))
        self.assertEqual(cfg["args"]["init_weights"], src_ckpt)
        with open(os.path.join(self.run_dir("warm"), "logs", "metrics.jsonl")) as f:
            lines = [json.loads(line) for line in f]
        self.assertEqual(lines[0]["data"]["resumed_step"], 0)           # fresh step / optimizer
        self.assertFalse(lines[0]["data"]["resumed"])
        self.assertTrue(os.path.exists(os.path.join(self.run_dir("warm"), "ckpt", "last", "last-2.index")))
        free_memory()
        # Mutually exclusive with resuming a run dir that has a checkpoint.
        with self.assertRaises(SystemExit) as cm:
            M.main(base + ["--run-name", "src", "--init-weights", src_ckpt])
        self.assertIn("--init-weights", str(cm.exception.code))
        free_memory()
        # Different ModelConfig: clear error.
        with self.assertRaises(ValueError) as cm:
            M.main(base + ["--run-name", "warm_bad", "--init-weights", src_ckpt, "--translation-scale", "7"])
        self.assertIn("translation_scale", str(cm.exception))


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

    def run_task_overfit(self, ratio, name, max_steps, schedule="constant"):
        """Milestone 5: synthetic overfit with --task-ratio `ratio` (affine: full affine ranges
        +-45 deg / +-15 mm / +-10 % scale / +-5 % shear; rigid: pure rigid as above; mixed runs
        alternate on two fixed groups). Stops when EVERY task's 10-step training mean is below
        OVERFIT_STOP_MM; then each task's eval-mode RMS must be < 1 mm and close to train mode."""
        rc = M.main(["--overfit-one-group", "--overfit-synthetic", "--level", "4", "--group-size", "4",
                     "--task-ratio", ratio, "--optimizer", "adam", "--lr", OVERFIT_LR, "--schedule", schedule,
                     "--warmup-steps", "0", "--overfit-max-steps", str(max_steps), "--overfit-threshold-mm",
                     OVERFIT_STOP_MM, "--log-every", "100", "--val-every", "250",
                     "--results", os.path.join(self.tmp, "results"), "--run-name", name] + NARROW_ARGS + COMMON)
        with open(os.path.join(self.run_dir(name), "overfit_summary.json")) as f:
            s = json.load(f)
        tasks = s["tasks"]
        sys.stderr.write("\n[overfit %s] steps %d, reached %s, best worst-task 10-step mean %.4f mm, per task "
                         "%s, median step %.0f ms\n" % (ratio, s["steps_run"], s["reached"],
                                                        s["best_window_center_rms_mm"] or float("nan"),
                                                        s["window_center_rms_by_task"], s["median_compute_ms"]))
        sys.stderr.write("[overfit %s val] %s:\n  %s\n" % (ratio, ", ".join(s["val_curve_columns"]), "\n  ".join(
            "  ".join("%7.3f" % x if isinstance(x, float) else str(x) for x in v) for v in s["val_curve"])))
        self.assertTrue(s["reached"], "best worst-task mean %.3f mm" % (s["best_window_center_rms_mm"] or -1))
        self.assertEqual(rc, 0)
        self.assertLessEqual(s["steps_run"], max_steps)
        self.assertEqual(s["trace_counts"], {t: 1 for t in tasks})
        for t in tasks:
            ev = s["final_val"]["val_%s/center_rms_mm" % t]
            tm = s["final_val"]["val_%s_trainmode/center_rms_mm" % t]
            sys.stderr.write("[overfit %s final] %s eval %.4f mm, train mode %.4f mm\n" % (ratio, t, ev, tm))
            self.assertLess(ev, 1.0, "%s eval-mode center RMS %.3f mm (train mode %.3f mm)" % (t, ev, tm))
            self.assertLess(abs(ev - tm), 0.5)
        return s

    def test_overfit_affine_below_1mm(self):
        s = self.run_task_overfit("affine:1", "overfit_affine", 3000)
        fv = s["final_val"]
        sys.stderr.write("[overfit affine final] scale %.3f %%, shear %.3f %%, QR rotation %.3f deg (centered)\n"
                         % (fv["val_affine/center_scale_pct"], fv["val_affine/center_shear_pct"],
                            fv["val_affine/center_qr_rot_deg"]))

    def test_overfit_mixed_rigid_affine_below_1mm(self):
        # 'Rigid metrics not degraded' at small scale: both tasks reach < 1 mm with one optimizer.
        # Default Adam 3e-4 with the default cosine decay (no warmup, over the 4000 max steps): at
        # a constant LR the two tasks keep perturbing each other and the eval RMS oscillates
        # between ~0.5 and ~2 mm (milestone-5 report).
        s = self.run_task_overfit("rigid:1,affine:1", "overfit_mixed", 4000, schedule="cosine")
        self.assertEqual(set(s["task_curve"]), {"rigid", "affine"})
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu"))
        try:
            import check_mixed_run
        finally:
            sys.path.pop(0)
        ok, res = check_mixed_run.check(self.run_dir("overfit_mixed"), ["rigid", "affine"], 1)
        self.assertTrue(ok, [r for r in res if not r[1]])


if __name__ == "__main__":
    unittest.main()
