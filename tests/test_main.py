"""main.py end to end: DALI check, dry run, the real-data overfit path and the OVERFIT test.

Milestone 6 adds the deform dry run, the real-data deform overfit path and the three-task dry
run (the synthetic deform overfit is tests/test_deform_overfit.py).

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
from tests.runtime_helpers import free_memory, write_fake_export, NARROW_ARGS, TINY_ARGS
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


class TestDeformMain(MainCase):
    """Milestone 6: deform runs through main.py (overfit mode, no DALI)."""

    def read_metrics(self, name):
        with open(os.path.join(self.run_dir(name), "logs", "metrics.jsonl")) as f:
            return [json.loads(line) for line in f]

    def test_synthetic_deform_dry_run(self):
        rc = M.main(["--overfit-one-group", "--overfit-synthetic", "--dry-run", "--group-size", "4",
                     "--task-ratio", "deform:1", "--results", os.path.join(self.tmp, "results"),
                     "--run-name", "ddry"] + NARROW_ARGS + COMMON)
        self.assertEqual(rc, 0)
        d = self.run_dir("ddry")
        with open(os.path.join(d, "config.json")) as f:
            cfg = json.load(f)
        mc = cfg["model_config"]
        self.assertEqual((mc["deform_decoder"], mc["decoder_widths"], mc["integration_steps"]), (True, None, 7))
        for k in ("lambda_pdice", "p_pdice", "lambda_smooth", "smooth_order", "integration_steps",
                  "decoder_widths", "deform_decoder", "deform_residual_affine", "deform_group_size",
                  "gradient_checkpointing", "overfit_dice_threshold", "overfit_fold_max", "val_deform_max_pairs",
                  "lambda_dice_template", "lambda_ncc_template", "lambda_ncc_pair", "ncc_window"):
            self.assertIn(k, cfg["args"])
        self.assertEqual(cfg["args"]["best_metric"], "val_deform/pair_dice_loss")
        self.assertIn("deform_decoder", cfg["param_counts"])
        lines = self.read_metrics("ddry")
        self.assertTrue(any("val_deform/pair_dice" in r["data"] for r in lines[1:]))
        self.assertTrue(any("train_deform/soft_dice_tmpl" in r["data"] for r in lines[1:]))
        self.assertFalse(any("train_deform/ncc_tmpl" in r["data"] for r in lines[1:]))     # LNCC off
        self.assertTrue(any("val_deform/ncc_template" in r["data"] for r in lines[1:]))   # always validated
        self.assertEqual(os.listdir(os.path.join(d, "val_png")), ["deform_step0000002.png"])
        self.assertFalse(os.path.exists(os.path.join(d, "control")))

    def test_synthetic_deform_ncc_dry_run(self):
        # v3.9: Dice + LNCC through main.py.
        rc = M.main(["--overfit-one-group", "--overfit-synthetic", "--dry-run", "--group-size", "4",
                     "--task-ratio", "deform:1", "--results", os.path.join(self.tmp, "results"),
                     "--run-name", "dncc", "--lambda-ncc-template", "1", "--lambda-ncc-pair", "1",
                     "--val-png", "false"] + NARROW_ARGS + COMMON)
        self.assertEqual(rc, 0)
        lines = self.read_metrics("dncc")
        for k in ("train_deform/ncc_tmpl", "train_deform/ncc_pair", "train_deform/ncc_tmpl_loss",
                  "val_deform/ncc_template", "val_deform/ncc_pair", "val_deform/ncc_template_identity"):
            self.assertTrue(any(k in r["data"] for r in lines[1:]), k)

    def test_deform_real_data_overfit_path(self):
        root = write_fake_export(self.tmp, n_train=5, n_val=2, labels=True)
        rc = M.main(["--overfit-one-group", "--data", root, "--group-size", "4", "--task-ratio", "deform:1",
                     "--overfit-max-steps", "3", "--log-every", "1", "--val-every", "3",
                     "--results", os.path.join(self.tmp, "results"), "--run-name", "dreal",
                     "--schedule", "constant", "--warmup-steps", "0"] + NARROW_ARGS + COMMON)
        self.assertIn(rc, (0, 1))
        with open(os.path.join(self.run_dir("dreal"), "overfit_summary.json")) as f:
            s = json.load(f)
        self.assertEqual(s["steps_run"], 3)
        self.assertEqual(s["trace_counts"], {"deform": 1})
        self.assertEqual(s["curve"], [])                                   # no linear task
        self.assertEqual(len(s["deform_curve"]), 3)
        self.assertEqual(s["deform_curve_columns"], ["step", "loss", "soft_dice_tmpl", "soft_dice_pair", "fold_frac"])
        self.assertEqual([r[0] for r in s["deform_val_curve"]], [3])
        iv = s["initial_val"]
        self.assertAlmostEqual(iv["val_deform/pair_dice"], iv["val_deform/pair_dice_identity"], places=10)
        self.assertTrue(np.isfinite(s["final_val"]["val_deform/template_dice"]))
        self.assertIsNone(s["window_deform"])                             # 3 steps < the 10-step window
        self.assertEqual(s["window_center_rms_by_task"], {})
        self.assertFalse(s["reached"])
        with open(os.path.join(self.run_dir("dreal"), "config.json")) as f:
            self.assertIn("train18 labels", json.load(f)["data"])

    def test_mixed_three_task_dry_run(self):
        res = os.path.join(self.tmp, "results")
        base = ["--overfit-one-group", "--overfit-synthetic", "--group-size", "4",
                "--task-ratio", "rigid:1,affine:1,deform:2", "--results", res] + NARROW_ARGS + COMMON
        self.assertEqual(M.main(base + ["--dry-run", "--run-name", "m3dry"]), 0)
        lines = self.read_metrics("m3dry")
        val = [r["data"] for r in lines[1:] if "val_deform/pair_dice" in r["data"]]
        self.assertEqual(len(val), 1)
        for k in ("val_rigid/pair_rms_mm", "val_affine/pair_rms_mm", "val/mean_pair_rms_mm"):
            self.assertIn(k, val[0])
        self.assertAlmostEqual(val[0]["val/mean_pair_rms_mm"],
                               0.5 * (val[0]["val_rigid/pair_rms_mm"] + val[0]["val_affine/pair_rms_mm"]), places=10)
        self.assertEqual(sorted(os.listdir(os.path.join(self.run_dir("m3dry"), "val_png"))),
                         ["affine_step0000002.png", "deform_step0000002.png", "rigid_step0000002.png"])
        free_memory()
        # A short (non-dry) run writes both best files: ckpt/best (linear rule) and ckpt/best_deform.
        rc = M.main(base + ["--run-name", "m3", "--overfit-max-steps", "4", "--val-every", "2", "--log-every", "2",
                            "--val-png", "false"])
        self.assertIn(rc, (0, 1))
        d = self.run_dir("m3")
        self.assertTrue(any(n.startswith("best-") for n in os.listdir(os.path.join(d, "ckpt", "best"))))
        self.assertTrue(any(n.startswith("best_deform-") for n in os.listdir(os.path.join(d, "ckpt", "best_deform"))))
        with open(os.path.join(d, "overfit_summary.json")) as f:
            s = json.load(f)
        self.assertEqual(sorted(s["trace_counts"]), ["affine", "deform", "rigid"])
        self.assertTrue(all(v <= 1 for v in s["trace_counts"].values()))
        self.assertEqual(s["val_curve_columns"], ["step", "rigid_eval_center_rms_mm", "rigid_trainmode_center_rms_mm",
                                                  "affine_eval_center_rms_mm", "affine_trainmode_center_rms_mm"])
        self.assertEqual(len(s["curve"]) + len(s["deform_curve"]), 4)

    def test_dali_data_wiring_with_deform(self):
        # The DALI path cannot run locally: stub the data module and the val loaders, check what
        # _build_dali_data hands them (deform group size, controls / linear val sets linear-only).
        from data_loading import params as P
        from runtime import args as A
        seen, calls = {}, {}
        shape = (6,) + P.LEVEL_SHAPE[4] + (1,)

        class FakeDM:
            def __init__(self, *a, **kw):
                seen.clear()
                seen.update(kw)
                self.control_writers = {t: None for t in kw.get("control_init", {})}
                self.files, self.template_dir = ["f"], "t"

            def iterator(self, task):
                return iter((task,))

        def record(name, value):
            def fn(*a, **k):
                calls[name] = (a, k)
                return value
            return fn

        fd = _Stub(load_level_images=record("images", (np.zeros(shape, np.float32), [])),
                   load_level_images_and_labels=record(
                       "images_labels", (np.zeros(shape, np.float32), np.zeros(shape, np.uint8), [], [])),
                   build_validation_sets=record("linear_val", {}),
                   DeformValidationSet=_Stub(build=record("deform_val", "deform_set")))
        mods = (fd, P, None, _Stub(brain_S_matrix=lambda *a, **k: np.eye(4)), _Stub(LinearDataModule=FakeDM))
        common = ["--data", self.tmp, "--results", self.tmp, "--run-name", "r"]
        args = A.parse_args(common + ["--task-ratio", "rigid:1,deform:2", "--group-size", "8",
                                      "--deform-group-size", "4", "--curriculum", "true",
                                      "--val-groups", "3", "--val-seed", "7"])
        sources, val_sets, _, _, controls = M._build_dali_data(args, 4, mods)
        self.assertEqual((seen["deform_batch_size"], seen["deform_residual_affine"]), (4, False))
        self.assertEqual(sorted(seen["control_init"]), ["rigid"])
        self.assertEqual(sorted(controls), ["rigid"])
        self.assertEqual(sorted(sources), ["deform", "rigid"])
        self.assertEqual(list(calls["linear_val"][0][1]), ["rigid"])
        self.assertEqual(calls["linear_val"][0][3], 6)                    # linear nv = min(8, 6 val cases)
        a, k = calls["deform_val"]
        self.assertEqual((a[2], a[3], a[4], k["seed"]), (4, 4, 3, 7))     # level, nv = min(4, 6), groups
        self.assertEqual(val_sets["deform"], "deform_set")
        calls.clear()
        args = A.parse_args(common + ["--task-ratio", "deform:1", "--group-size", "8",
                                      "--deform-residual-affine", "true"])
        sources, val_sets, _, _, controls = M._build_dali_data(args, 4, mods)
        self.assertEqual((seen["deform_batch_size"], seen["deform_residual_affine"]), (8, True))
        self.assertNotIn("control_init", seen)
        self.assertEqual((sorted(sources), sorted(val_sets), controls), (["deform"], ["deform"], {}))
        self.assertEqual(sorted(calls), ["deform_val", "images_labels"])  # no linear val loading
        self.assertEqual(calls["deform_val"][0][3], 6)                      # nv = min(8, 6 val cases)

    def test_three_task_validation_group_sizes(self):
        # Milestone 7 recipe on 39 val cases (the real T1 val split): linear N 32 -> 2 groups of
        # 32 per linear set, deform N 16 -> 3 groups of 16; no --val-group-size.
        from data_loading import params as P
        from runtime import args as A
        from runtime import fixed_data as real_fd
        shape = (39,) + P.LEVEL_SHAPE[4] + (1,)
        rng = np.random.default_rng(0)
        images = rng.random(shape).astype(np.float32)
        labels = rng.integers(0, 3, shape).astype(np.uint8)

        class FakeDM:
            def __init__(self, *a, **kw):
                self.kw = kw
                self.control_writers, self.files, self.template_dir = {}, ["f"], "t"

            def iterator(self, task):
                return iter((task,))

        fd = _Stub(load_level_images=lambda *a, **k: (images, []),
                   load_level_images_and_labels=lambda *a, **k: (images, labels, [], []),
                   build_validation_sets=real_fd.build_validation_sets,
                   DeformValidationSet=real_fd.DeformValidationSet)
        mods = (fd, P, None, _Stub(brain_S_matrix=lambda *a, **k: np.eye(4)), _Stub(LinearDataModule=FakeDM))
        args = A.parse_args(["--data", self.tmp, "--results", self.tmp, "--run-name", "r",
                             "--task-ratio", "rigid:1,affine:1,deform:2", "--group-size", "32",
                             "--deform-group-size", "16"])
        _, val_sets, _, _, _ = M._build_dali_data(args, 4, mods)
        self.assertEqual(sorted(val_sets), ["affine", "deform", "rigid"])
        for t in ("rigid", "affine"):
            self.assertEqual((val_sets[t].group_size, len(val_sets[t].groups)), (32, 2), t)
        self.assertEqual((val_sets["deform"].group_size, len(val_sets["deform"].groups)), (16, 3))
        self.assertEqual(val_sets["deform"].groups[0]["images"].shape[0], 16)
        self.assertEqual(args.val_deform_max_pairs, 64)                    # 16 * 15 = 240 pairs -> 64


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
        self.assertEqual(cfg["init_weights"]["mode"], "full")
        free_memory()
        # Milestone 7: the linear-only source warm-starts a rigid + deform run (partial restore).
        self.assertEqual(M.main(base + ["--run-name", "warm_deform", "--init-weights", src_ckpt,
                                        "--task-ratio", "rigid:1,deform:1", "--deform-group-size", "2"]), 0)
        with open(os.path.join(self.run_dir("warm_deform"), "config.json")) as f:
            iw = json.load(f)["init_weights"]
        self.assertEqual(iw["mode"], "partial")
        self.assertEqual(sorted(iw["fresh_parts"]), ["deform_decoder"])
        self.assertEqual(sorted(iw["restored_parts"]), ["centering", "encoder", "linear_decoder"])
        self.assertEqual(iw["config_diff"], {"deform_decoder": [False, True]})
        with open(os.path.join(self.run_dir("warm_deform"), "logs", "metrics.jsonl")) as f:
            lines = [json.loads(line) for line in f]
        self.assertEqual(lines[0]["step"], "PARAMETER")
        ev = [r["data"] for r in lines if r["data"].get("event") == "init_weights"]
        self.assertEqual(len(ev), 1)
        self.assertEqual((ev[0]["mode"], ev[0]["fresh_parts"]), ("partial", iw["fresh_parts"]))
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

    def test_three_task_warm_start_smoke(self):
        # Milestone 7 end to end: a tiny linear-only run warm-starts a rigid:affine:deform 1:1:2
        # run with different group sizes (linear 4, deform 2) and --loss-scale-mm2 100; the run
        # completes, logs the encoder grad-norm diagnostics and writes both best checkpoints.
        from runtime.run import TaskSchedule
        res = os.path.join(self.tmp, "results")
        common = ["--overfit-one-group", "--overfit-synthetic", "--group-size", "4", "--log-every", "1",
                  "--schedule", "constant", "--warmup-steps", "0", "--results", res] + NARROW_ARGS + COMMON
        rc = M.main(common + ["--run-name", "lin_src", "--task-ratio", "rigid:1,affine:1", "--overfit-max-steps", "6",
                              "--val-every", "3", "--val-png", "false"])
        self.assertIn(rc, (0, 1))
        free_memory()
        ratio = {"rigid": 1.0, "affine": 1.0, "deform": 2.0}
        seed = next(s for s in range(1000) if {TaskSchedule(ratio, s).draw(i) for i in range(12)} == set(ratio))
        rc = M.main(common + ["--run-name", "m7", "--task-ratio", "rigid:1,affine:1,deform:2", "--deform-group-size",
                              "2", "--init-weights", os.path.join(self.run_dir("lin_src"), "ckpt", "last"),
                              "--overfit-max-steps", "12", "--val-every", "6", "--loss-scale-mm2", "100",
                              "--seed", str(seed)])
        self.assertIn(rc, (0, 1))
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu"))
        try:
            import check_mixed_run
        finally:
            sys.path.pop(0)
        d = self.run_dir("m7")
        ok, results = check_mixed_run.check(d, ["rigid", "affine", "deform"], 12, expect_best=True,
                                            expect_init_weights="partial", expect_enc_grad=True)
        sys.stderr.write("\n[m7 smoke] seed %d\n  %s\n" % (seed, "\n  ".join(
            "%s %-22s %s" % ("PASS" if p else "FAIL", n, x) for n, p, x in results)))
        self.assertTrue(ok, [r for r in results if not r[1]])
        with open(os.path.join(d, "overfit_summary.json")) as f:
            s = json.load(f)
        self.assertEqual(s["trace_counts"], {"rigid": 1, "affine": 1, "deform": 1})
        self.assertEqual(s["deform_group_size"], 2)
        self.assertEqual(s["group_size"], 4)


class TestCascadeMain(MainCase):
    """Milestone 8: --cascade-from through main.py (overfit-synthetic, TINY widths, rigid only with
    the deform decoder built so both levels share the ModelConfig). Resume: tests/test_cascade.py T8."""

    def test_cascade_overfit_smoke_and_level_error(self):
        res = os.path.join(self.tmp, "results")
        common = ["--overfit-one-group", "--overfit-synthetic", "--group-size", "4", "--task-ratio", "rigid:1",
                  "--deform-decoder", "true", "--log-every", "1", "--val-every", "0", "--val-png", "false",
                  "--results", res] + TINY_ARGS + COMMON
        self.assertIn(M.main(common + ["--level", "4", "--run-name", "src4", "--overfit-max-steps", "2"]), (0, 1))
        free_memory()
        src = os.path.join(self.run_dir("src4"), "ckpt", "last")
        rc = M.main(common + ["--level", "2", "--cascade-from", src, "--run-name", "c2", "--overfit-max-steps", "2"])
        self.assertIn(rc, (0, 1))
        with open(os.path.join(self.run_dir("c2"), "config.json")) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["init_weights"]["via"], "cascade")
        self.assertEqual(cfg["init_weights"]["mode"], "full")
        cas = cfg["cascade"]
        self.assertEqual((cas["source_level"], cas["source_step"], cas["skip_prob"]), (4, 2, 0.0))
        self.assertIsNone(cas["S_rel_diff_vs_level4"])                   # overfit-synthetic
        self.assertTrue(cas["source_checkpoint"].endswith("last-2"))
        self.assertGreater(cas["fingerprint"]["n_vars"], 0)
        self.assertNotIn("bn_recalibration", cfg)                        # --bn-recal-batches 0
        with open(os.path.join(self.run_dir("c2"), "overfit_summary.json")) as f:
            s = json.load(f)
        self.assertEqual((s["trace_counts"], s["steps_run"]), ({"rigid": 1}, 2))
        self.assertTrue(np.isfinite(s["final_val"]["val_rigid/center_rms_mm"]))
        self.assertTrue(np.isfinite(s["final_val"]["val_rigid_frozen/center_rms_mm"]))
        free_memory()
        # A level-2 run as the source: SystemExit naming the level (before any model is built).
        with self.assertRaises(SystemExit) as cm:
            M.main(common + ["--level", "2", "--cascade-from", os.path.join(self.run_dir("c2"), "ckpt", "last"),
                             "--run-name", "bad"])
        msg = str(cm.exception.code)
        self.assertTrue(msg.startswith("main.py: --cascade-from"), msg)
        self.assertIn("level 2", msg)
        self.assertIn("level-4 source", msg)
        self.assertFalse(os.path.exists(self.run_dir("bad")))
        # A config mismatch with the source: SystemExit naming the field.
        with self.assertRaises(SystemExit) as cm:
            M.main(common + ["--level", "2", "--cascade-from", src, "--run-name", "bad2", "--dropout", "0.1"])
        self.assertIn("dropout", str(cm.exception.code))


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
