"""main.py end to end: DALI check, dry run, the real-data overfit path and the OVERFIT test.

Milestone 6 adds the deform dry run, the real-data deform overfit path and the three-task dry
run (the synthetic deform overfit is tests/test_deform_overfit.py).

Milestone 5 adds: the affine overfit (full affine ranges, < 1 mm eval mode), the mixed
rigid:affine overfit on two fixed groups (both < 1 mm; tests/gpu/check_mixed_run.py on its run
dir), the resumed step's control values before the pipelines start, and --init-weights.

Milestone 9 phase 2 adds the level 4 -> 2 -> 1 cascade chain with a level-1 resume and the
in-process checker (TestLevel1CascadeMain, T-E1) and the level-1 --cascade-from errors.

The overfit test (milestone 4 verification): one fixed synthetic group (smooth asymmetric
phantom, per-image random rigid transforms up to +-45 deg / +-15 mm, N = 4, level 4), narrow
model (default dense_norm 'layer'), Adam 3e-4 constant LR, no flips, no dropout. It must reach a
10-step mean training-step centered RMS below OVERFIT_STOP_MM within 1500 steps, and then the
eval-mode validation of the same group must be below 1.0 mm and within 0.5 mm of the
training-mode (no update) value. The RMS curve is printed every 50 steps.
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
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

    def test_fp16_mitigation_flags_reach_model(self):
        # nonfinite-fixes a1 / a2 / a3 through main.py under --amp (overfit dry run, 2 steps).
        import tensorflow as tf
        self.addCleanup(tf.keras.mixed_precision.set_global_policy, "float32")       # --amp true
        from models import group_net                             # main() imports the class lazily
        seen = {}
        real_init = group_net.GroupRegLevelModel.__init__

        def spy(model, *a, **kw):
            real_init(model, *a, **kw)
            seen.setdefault("model", model)

        group_net.GroupRegLevelModel.__init__ = spy
        self.addCleanup(setattr, group_net.GroupRegLevelModel, "__init__", real_init)
        rc = M.main(["--overfit-one-group", "--overfit-synthetic", "--dry-run", "--group-size", "4",
                     "--task-ratio", "deform:1", "--results", os.path.join(self.tmp, "results"),
                     "--run-name", "dfp16", "--amp", "true", "--fp16-act-clip", "30000",
                     "--deform-decoder-dtype", "float32", "--bn-epsilon-override", "1e-3",
                     "--nonfinite-patience-task", "5"] + NARROW_ARGS + COMMON)
        self.assertEqual(rc, 0)
        model = seen["model"]
        self.assertEqual((model.fp16_act_clip, model.fp16_act_clip_scope, model.deform_decoder_dtype,
                          model.bn_epsilon_override), (30000.0, "deform_decoder", "float32", 1e-3))
        self.assertEqual(model.deform_decoder.d1.conv.compute_dtype, "float32")
        self.assertEqual(model.encoder.l3.conv.compute_dtype, "float16")
        self.assertEqual(model.deform_decoder.d1.fp16_act_clip, 30000.0)
        with open(os.path.join(self.run_dir("dfp16"), "config.json")) as f:
            args = json.load(f)["args"]
        self.assertEqual((args["fp16_act_clip"], args["fp16_act_clip_scope"], args["deform_decoder_dtype"],
                          args["bn_epsilon_override"], args["nonfinite_patience_task"]),
                         (30000.0, "deform_decoder", "float32", 1e-3, 5))

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

    def test_memory_knobs_dry_run(self):
        # Milestone 9 (design 3.8): every memory knob on, through main.py (Dice + LNCC); the
        # knobs land in config.json args, not in model_config.
        rc = M.main(["--overfit-one-group", "--overfit-synthetic", "--dry-run", "--group-size", "4",
                     "--task-ratio", "rigid:1,deform:1", "--results", os.path.join(self.tmp, "results"),
                     "--run-name", "dknobs", "--gradient-checkpointing", "full", "--lean-sampler", "true",
                     "--lean-losses", "true", "--split-group-conv", "true", "--lambda-ncc-template", "1",
                     "--lambda-ncc-pair", "1", "--val-png", "false"] + NARROW_ARGS + COMMON)
        self.assertEqual(rc, 0)
        with open(os.path.join(self.run_dir("dknobs"), "config.json")) as f:
            cfg = json.load(f)
        self.assertEqual((cfg["args"]["gradient_checkpointing"], cfg["args"]["lean_sampler"],
                          cfg["args"]["lean_losses"], cfg["args"]["split_group_conv"]), ("full", True, True, True))
        for k in ("lean_sampler", "lean_losses", "split_group_conv", "gradient_checkpointing"):
            self.assertNotIn(k, cfg["model_config"])
        lines = self.read_metrics("dknobs")
        self.assertTrue(any("train_deform/ncc_pair" in r["data"] for r in lines[1:]))

    def test_build_n(self):
        from runtime import args as A
        base = ["--data", "x", "--task-ratio", "rigid:1,deform:1", "--group-size", "16", "--deform-group-size", "8"]
        self.assertEqual(M._build_n(A.parse_args(base)), 8)                     # today's rule
        self.assertEqual(M._build_n(A.parse_args(base + ["--split-group-conv", "true"])), 2)
        self.assertEqual(M._build_n(A.parse_args(["--data", "x", "--group-size", "16"])), 16)
        self.assertEqual(M._build_n(A.parse_args(["--data", "x", "--group-size", "16",
                                                  "--split-group-conv", "true"])), 2)
        self.assertEqual(M._build_n(A.parse_args(["--data", "x", "--level", "1", "--split-group-conv", "true"])), 2)

    def test_trainer_lean_mismatch(self):
        from tests.runtime_helpers import narrow_config, synthetic_group
        from models.group_net import GroupRegLevelModel
        from runtime import fixed_data as fd
        from runtime import optimizers as O
        from runtime.run import Trainer, TrainConfig
        g = synthetic_group(4, 4, "rigid")
        for model_lean, cfg_lean in ((False, True), (True, False)):
            model = GroupRegLevelModel(4, narrow_config(deform_decoder=True), lean_sampler=model_lean).build_model(2)
            with self.assertRaises(ValueError) as cm:
                Trainer(model, O.make_optimizer("adam", lr=1e-3), {"rigid": fd.FixedGroupSource(g["batch"])},
                        g["S"], TrainConfig(group_size=4, lean_sampler=cfg_lean))
            self.assertIn("lean_sampler: model %s vs TrainConfig %s" % (model_lean, cfg_lean), str(cm.exception))

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


class TestIntensityAugDevice(MainCase):
    """--intensity-aug-device: DALI config vs TrainConfig.intensity_aug_tf, the overfit parse
    error and the resume guard (`main._check_intensity_aug_device`)."""

    def build(self, extra):
        from data_loading import params as P
        from runtime import args as A
        seen = {}

        class FakeDM:
            def __init__(self, *a, **kw):
                seen.update(kw)
                self.control_writers = {}
                self.files, self.template_dir = ["f"], "t"

            def iterator(self, task):
                return iter(())

        fd = _Stub(load_level_images=lambda *a, **k: (np.zeros((8,) + P.LEVEL_SHAPE[4] + (1,)), []),
                   build_validation_sets=lambda *a, **k: {})
        pp = _Stub(brain_S_matrix=lambda *a, **k: np.eye(4))
        args = A.parse_args(["--data", self.tmp, "--results", self.tmp, "--run-name", "r"] + extra)
        desc = M._build_dali_data(args, 4, (fd, P, None, pp, _Stub(LinearDataModule=FakeDM)))[3]
        return args, seen["intensity_aug"], A.train_config_from_args(args), desc

    def test_device_routing(self):
        args, dali_cfg, tc, desc = self.build([])
        self.assertEqual(args.intensity_aug_device, "dali")
        self.assertTrue(dali_cfg.enabled)
        self.assertIsNone(tc.intensity_aug_tf)
        self.assertIn("intensity aug dali", desc)
        _, dali_cfg, tc, desc = self.build(["--intensity-aug-device", "tf"])
        self.assertFalse(dali_cfg.enabled)
        self.assertTrue(tc.intensity_aug_tf.enabled)
        self.assertIn("intensity aug tf", desc)
        _, dali_cfg, tc, desc = self.build(["--intensity-aug", "false", "--intensity-aug-device", "tf"])
        self.assertFalse(dali_cfg.enabled)
        self.assertIsNone(tc.intensity_aug_tf)
        self.assertIn("intensity aug off", desc)

    def test_overfit_parse_error(self):
        from runtime import args as A
        with self.assertRaises(SystemExit):
            A.parse_args(["--overfit-one-group", "--overfit-synthetic", "--intensity-aug-device", "tf"])
        with self.assertRaises(SystemExit):
            A.parse_args(["--data", self.tmp, "--intensity-aug-device", "gpu"])

    def test_lowres_method(self):
        """retrain-prereqs 11.2 f: --lowres-method tent needs the TF device; default nn."""
        from data_loading import params as P
        from runtime import args as A
        self.assertEqual(P.IntensityAugConfig().low_res_method, "nn")
        with self.assertRaises(SystemExit):
            A.parse_args(["--data", self.tmp, "--lowres-method", "tent", "--intensity-aug-device", "dali"])
        with self.assertRaises(SystemExit):          # dali is the default device
            A.parse_args(["--data", self.tmp, "--lowres-method", "tent"])
        with self.assertRaises(SystemExit):
            A.parse_args(["--data", self.tmp, "--lowres-method", "area", "--intensity-aug-device", "tf"])
        args, _, tc, desc = self.build(["--intensity-aug-device", "tf"])
        self.assertEqual(args.lowres_method, "nn")
        self.assertEqual(tc.intensity_aug_tf.low_res_method, "nn")
        self.assertNotIn("low-res tent", desc)
        args, _, tc, desc = self.build(["--intensity-aug-device", "tf", "--lowres-method", "tent"])
        self.assertEqual(tc.intensity_aug_tf.low_res_method, "tent")
        self.assertIn("intensity aug tf | low-res tent", desc)
        # --intensity-aug false: accepted, no effect.
        args = A.parse_args(["--data", self.tmp, "--intensity-aug", "false", "--lowres-method", "tent"])
        self.assertIsNone(A.train_config_from_args(args).intensity_aug_tf)

    def test_resume_guard(self):
        from runtime import args as A
        path = os.path.join(self.tmp, "config.json")

        def write(saved_args):
            with open(path, "w") as f:
                json.dump({"args": saved_args}, f)

        dali = A.parse_args(["--data", self.tmp])
        tf_args = A.parse_args(["--data", self.tmp, "--intensity-aug-device", "tf"])
        M._check_intensity_aug_device(os.path.join(self.tmp, "missing.json"), tf_args)   # no file: ok
        write({"level": 4})                       # runs before the option = 'dali'
        M._check_intensity_aug_device(path, dali)
        with self.assertRaises(SystemExit) as cm:
            M._check_intensity_aug_device(path, tf_args)
        self.assertIn("--intensity-aug-device tf", str(cm.exception))
        write({"intensity_aug_device": "tf", "intensity_aug": True})
        M._check_intensity_aug_device(path, tf_args)
        with self.assertRaises(SystemExit):
            M._check_intensity_aug_device(path, dali)
        off = A.parse_args(["--data", self.tmp, "--intensity-aug-device", "tf", "--intensity-aug", "false"])
        with self.assertRaises(SystemExit):        # same device, but the aug generators differ
            M._check_intensity_aug_device(path, off)


class TestOptimizerMain(MainCase):
    """Retrain-prereqs design 2.2 / test 11.1 f: the startup optimizer line and NOTE, the
    config.json ``optimizer`` record (sgdw / adamw only) and the resume guard ``_check_optimizer``."""

    BASE = ["--overfit-one-group", "--overfit-synthetic", "--level", "4", "--group-size", "4",
            "--task-ratio", "rigid:1", "--log-every", "1", "--val-every", "0", "--val-png", "false",
            "--schedule", "constant", "--warmup-steps", "0"] + TINY_ARGS + COMMON + ["--quiet", "false"]

    def run_main(self, name, extra, steps=2):
        argv = self.BASE + ["--results", os.path.join(self.tmp, "results"), "--run-name", name,
                            "--overfit-max-steps", str(steps)] + extra
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = M.main(argv)
        free_memory()
        self.assertIn(rc, (0, 1))
        with open(os.path.join(self.run_dir(name), "config.json")) as f:
            return buf.getvalue(), json.load(f)

    def test_adam_note_and_no_record(self):
        out, cfg = self.run_main("adam", ["--optimizer", "adam", "--weight-decay", "1e-4"])
        self.assertIn("NOTE: --weight-decay is ignored by --optimizer adam", out)
        self.assertNotIn("optimizer", cfg)
        out, cfg = self.run_main("adam0", ["--optimizer", "adam", "--weight-decay", "0"])
        self.assertNotIn("NOTE: --weight-decay", out)
        self.assertNotIn("optimizer", cfg)

    def test_adamw_record_and_resume(self):
        from models.group_net import GroupRegLevelModel
        from runtime import args as A
        out, cfg = self.run_main("w", ["--optimizer", "adamw", "--lr", "3e-4", "--weight-decay", "0.01"])
        rec = cfg["optimizer"]
        n = len(GroupRegLevelModel(4, A.model_config_from_args(A.parse_args(self.BASE)))
                .build_model(4).trainable_variables)
        free_memory()
        self.assertEqual((rec["name"], rec["lr"], rec["weight_decay"]), ("adamw", 3e-4, 0.01))
        self.assertEqual(rec["decayed_variables"] + rec["excluded_variables"], n)
        self.assertGreater(rec["decayed_variables"], 0)
        self.assertGreater(rec["excluded_variables"], 0)
        self.assertIn("optimizer adamw | lr 0.0003 | weight decay 0.01 (decoupled, lr-scaled; excluded "
                      "gamma/beta/bias: %d of %d variables)" % (rec["excluded_variables"], n), out)
        self.assertNotIn("NOTE: --weight-decay", out)
        # Resume with another optimizer: SystemExit before any restore.
        with self.assertRaises(SystemExit) as cm:
            self.run_main("w", ["--optimizer", "adam", "--lr", "3e-4"], steps=3)
        free_memory()
        msg = str(cm.exception.code)
        self.assertIn("--optimizer adam differs from the run's saved adamw", msg)
        # Resume with a changed --lr: allowed, logged as an event, record refreshed.
        out, cfg = self.run_main("w", ["--optimizer", "adamw", "--lr", "1e-4", "--weight-decay", "0.01"], steps=3)
        self.assertEqual(cfg["optimizer"]["lr"], 1e-4)
        self.assertEqual(cfg["optimizer"]["weight_decay"], 0.01)
        with open(os.path.join(self.run_dir("w"), "logs", "metrics.jsonl")) as f:
            events = [r for r in (json.loads(line) for line in f)
                      if isinstance(r.get("data"), dict)
                      and r["data"].get("event") == "optimizer_hparam_changed_on_resume"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["step"], 2)
        self.assertEqual(events[0]["data"]["old"], {"lr": 3e-4, "weight_decay": 0.01})
        self.assertEqual(events[0]["data"]["new"], {"lr": 1e-4, "weight_decay": 0.01})

    def test_check_optimizer_unit(self):
        from runtime import args as A
        path = os.path.join(self.tmp, "config.json")
        with open(path, "w") as f:
            json.dump({"args": {"lr": 3e-4, "weight_decay": 3e-5}}, f)      # no key = 'adam'
        self.assertIsNone(M._check_optimizer(os.path.join(self.tmp, "missing.json"),
                                             A.parse_args(["--data", self.tmp, "--optimizer", "sgdw"])))
        self.assertIsNone(M._check_optimizer(path, A.parse_args(["--data", self.tmp])))
        with self.assertRaises(SystemExit):
            M._check_optimizer(path, A.parse_args(["--data", self.tmp, "--optimizer", "adamw"]))
        changed = M._check_optimizer(path, A.parse_args(["--data", self.tmp, "--weight-decay", "0"]))
        self.assertEqual(changed, {"old": {"lr": 3e-4, "weight_decay": 3e-5},
                                   "new": {"lr": 3e-4, "weight_decay": 0.0}})
        for bad in ("-1", "nan", "inf"):
            with self.assertRaises(SystemExit):
                A.parse_args(["--data", self.tmp, "--weight-decay", bad])


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

    def test_stage_skip_record_and_resume(self):
        """Retrain-prereqs 11.3 a (resume part): the cascade record's skip_mode / skip_probs /
        skip_consistency_warnings, the logged per-stage key, and a resume with a changed spec
        refreshing them and writing the cascade_skip_changed_on_resume event."""
        res = os.path.join(self.tmp, "results")
        common = ["--overfit-one-group", "--overfit-synthetic", "--group-size", "4", "--task-ratio", "rigid:1",
                  "--deform-decoder", "true", "--log-every", "1", "--val-every", "0", "--val-png", "false",
                  "--results", res] + TINY_ARGS + COMMON
        self.assertIn(M.main(common + ["--level", "4", "--run-name", "src4", "--overfit-max-steps", "2"]), (0, 1))
        free_memory()
        run = common + ["--level", "2", "--cascade-from", os.path.join(self.run_dir("src4"), "ckpt", "last"),
                        "--run-name", "c2"]
        self.assertIn(M.main(run + ["--overfit-max-steps", "2", "--cascade-skip-linear", "4:0.5"]), (0, 1))
        free_memory()
        with open(os.path.join(self.run_dir("c2"), "config.json")) as f:
            cas = json.load(f)["cascade"]
        self.assertEqual((cas["skip_mode"], cas["skip_probs"], cas["skip_consistency_warnings"], cas["skip_prob"]),
                         ("stage", {"rigid": {"4": 0.5}}, [], 0.0))
        self.assertIn(M.main(run + ["--overfit-max-steps", "4", "--cascade-skip-linear", "4:0.2"]), (0, 1))
        free_memory()
        d = self.run_dir("c2")
        with open(os.path.join(d, "config.json")) as f:
            cas2 = json.load(f)["cascade"]
        self.assertEqual(cas2["skip_probs"], {"rigid": {"4": 0.2}})
        self.assertEqual({k: v for k, v in cas2.items() if k != "skip_probs"},
                         {k: v for k, v in cas.items() if k != "skip_probs"})            # everything else carried
        with open(os.path.join(d, "logs", "metrics.jsonl")) as f:
            recs = [json.loads(line) for line in f if line.strip()]
        self.assertTrue(any("train_rigid/cascade_skip_l4" in r["data"] for r in recs))
        self.assertFalse(any("train/cascade_skip_frac" in r["data"] for r in recs))
        ev = [r for r in recs if r["data"].get("event") == "cascade_skip_changed_on_resume"]
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["data"]["old"]["skip_probs"], {"rigid": {"4": 0.5}})
        self.assertEqual(ev[0]["data"]["new"]["skip_probs"], {"rigid": {"4": 0.2}})
        self.assertEqual(ev[0]["step"], 2)

    def test_random_width_sources_chain(self):
        """fp32-decoder-width FEAT-002: the minimal source recipe of slurm/m11_random_sources.sh
        (one rigid step, deform decoder built, no validation, only the final ckpt/last save) at a
        non-default width (stem and D2/D1 narrower than TINY) builds a level-4 and a level-2 source
        that a level-1 cascade accepts through the normal --cascade-from copy path."""
        import tensorflow as tf
        self.addCleanup(tf.keras.mixed_precision.set_global_policy, "float32")       # level 1 --amp true
        res = os.path.join(self.tmp, "results")
        widths = ["--filters", "4,16,16,16,16,16", "--decoder-widths", "8,8,8,4,4"]
        recipe = ["--overfit-one-group", "--overfit-synthetic", "--task-ratio", "rigid:1", "--deform-decoder", "true",
                  "--group-size", "2", "--overfit-max-steps", "1", "--warmup-steps", "0", "--val-every", "0",
                  "--ckpt-every", "0", "--val-png", "false", "--log-every", "1", "--resume", "false",
                  "--results", res] + TINY_ARGS + widths + COMMON
        self.assertIn(M.main(recipe + ["--level", "4", "--run-name", "l4_w"]), (0, 1))
        free_memory()
        self.assertIn(M.main(recipe + ["--level", "2", "--run-name", "l2_w", "--cascade-from",
                                       os.path.join(self.run_dir("l4_w"), "ckpt", "last")]), (0, 1))
        free_memory()
        src2 = os.path.join(self.run_dir("l2_w"), "ckpt", "last")
        keys = [k for k, _ in tf.train.list_variables(src2)]
        self.assertTrue(any(k.startswith("frozen_stage/") for k in keys))
        self.assertIn(M.main(recipe + ["--level", "1", "--run-name", "l1_w", "--cascade-from", src2,
                                       "--amp", "true", "--split-group-conv", "true"]), (0, 1))
        free_memory()
        for name, level in (("l2_w", 2), ("l1_w", 1)):
            with open(os.path.join(self.run_dir(name), "config.json")) as f:
                cfg = json.load(f)
            with self.subTest(name):
                self.assertEqual(cfg["model_config"]["filters"], [4, 16, 16, 16, 16, 16])
                self.assertEqual(cfg["model_config"]["decoder_widths"], [8, 8, 8, 4, 4])
                self.assertTrue(cfg["model_config"]["deform_decoder"])
                iw = cfg["init_weights"]
                self.assertEqual((iw["via"], iw["mode"], iw["n_fresh_vars"]), ("cascade", "full", 0))
                self.assertEqual(iw["cross_level"], [4 if level == 2 else 2, level])
                self.assertEqual(cfg["cascade"]["source_step"], 1)
        with open(os.path.join(self.run_dir("l1_w"), "config.json")) as f:
            cas = json.load(f)["cascade"]
        self.assertEqual([s["level"] for s in cas["stages"]], [4, 2])
        self.assertEqual([s["origin_step"] for s in cas["stages"]], [1, 1])


class TestStageSkipArgs(unittest.TestCase):
    """Retrain-prereqs 11.3 a (parser) and g (`main._skip_consistency`), `_refresh_skip_record`."""

    BASE = ["--data", "x"]

    def parse(self, extra):
        from runtime import args as A
        return A.parse_args(self.BASE + extra)

    def parse_error(self, extra):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            self.parse(extra)
        return err.getvalue()

    def test_valid_specs(self):
        from runtime import args as A
        three = ["--task-ratio", "rigid:1,affine:1,deform:2"]
        a = self.parse(three + ["--level", "1", "--cascade-from", "y", "--cascade-skip-linear", "4:0,2:0.15",
                                "--cascade-skip-deform", " 4:0.15 , 2:0.15 "])
        want = {"rigid": {4: 0.0, 2: 0.15}, "affine": {4: 0.0, 2: 0.15}, "deform": {4: 0.15, 2: 0.15}}
        self.assertEqual(a.cascade_skip_stage_parsed, want)
        self.assertEqual(A.train_config_from_args(a).cascade_skip_stage, want)
        self.assertEqual(M._skip_record(a), {"skip_mode": "stage", "skip_probs": {
            "rigid": {"4": 0.0, "2": 0.15}, "affine": {"4": 0.0, "2": 0.15}, "deform": {"4": 0.15, "2": 0.15}}})
        b = self.parse(["--level", "2", "--cascade-from", "y", "--cascade-skip-linear", ""])
        self.assertEqual(b.cascade_skip_stage_parsed, {"rigid": {}})
        self.assertEqual(M._skip_record(b), {"skip_mode": "off"})
        c = self.parse(["--level", "2", "--cascade-from", "y", "--cascade-skip-prob", "0.1",
                        "--cascade-skip-linear", "4:0"])                       # p 0 with legacy: allowed
        self.assertEqual(M._skip_record(c), {"skip_mode": "legacy"})
        d = self.parse([])
        self.assertEqual(d.cascade_skip_stage_parsed, {"rigid": {}})
        self.assertEqual(A.train_config_from_args(d).cascade_skip_stage, {"rigid": {}})
        self.assertEqual(A.parse_skip_spec("2:0.5,4:0.25", "--x", [4, 2]), {2: 0.5, 4: 0.25})

    def test_errors(self):
        c2 = ["--level", "2", "--cascade-from", "y"]
        cases = [
            (c2 + ["--cascade-skip-linear", "2:0.1"], ["--cascade-skip-linear", "level 2", "2:0.1"]),
            (c2 + ["--cascade-skip-deform", "4:1"], ["--cascade-skip-deform", "4:1", "[0, 1)"]),
            (c2 + ["--cascade-skip-deform", "4:nan"], ["--cascade-skip-deform", "4:nan"]),
            (c2 + ["--cascade-skip-linear", "4:0.1,4:0.2"], ["--cascade-skip-linear", "duplicate level 4"]),
            (c2 + ["--cascade-skip-linear", "4"], ["--cascade-skip-linear", "malformed", "'4'"]),
            (c2 + ["--cascade-skip-linear", "4:abc"], ["--cascade-skip-linear", "malformed", "'4:abc'"]),
            (c2 + ["--cascade-skip-linear", "x:0.1"], ["--cascade-skip-linear", "malformed"]),
            (c2 + ["--cascade-skip-prob", "0.1", "--cascade-skip-deform", "4:0.2"],
             ["--cascade-skip-prob and --cascade-skip-linear / --cascade-skip-deform are mutually exclusive"]),
        ]
        for flag in ("--cascade-skip-linear", "--cascade-skip-deform"):
            for spec in ("4:0", "", "4:0.2"):
                for lv in ("2", "4"):
                    cases.append((["--level", lv, flag, spec], ["%s needs --cascade-from" % flag]))
        for extra, needles in cases:
            msg = self.parse_error(extra)
            for n in needles:
                self.assertIn(n, msg, (extra, msg))

    def test_skip_consistency(self):
        lin = {"rigid": {4: 0.1}, "affine": {4: 0.1}, "deform": {}}
        dfm = {"rigid": {}, "affine": {}, "deform": {4: 0.15, 2: 0.15}}
        src = {"skip_prob": 0.0, "skip_mode": "stage",
               "skip_probs": {"rigid": {"4": 0.0}, "affine": {"4": 0.0}, "deform": {"4": 0.15}}}
        self.assertEqual(M._skip_consistency(src, dfm), [])
        self.assertEqual(M._skip_consistency(src, lin), ["rigid", "affine"])
        self.assertEqual(M._skip_consistency({"skip_prob": 0.1}, dict(lin, **dfm)), [])
        self.assertEqual(M._skip_consistency({"skip_prob": 0.0}, {"rigid": {4: 0.1}, "affine": {}, "deform": {4: 0.15}}),
                         ["rigid", "deform"])
        self.assertEqual(M._skip_consistency(None, lin), [])
        self.assertEqual(M._skip_consistency(src, {"rigid": {2: 0.2}, "deform": {4: 0.0, 2: 0.3}}), [])
        lines = M._skip_consistency_lines(src, ["rigid"]) + M._skip_consistency_lines({"skip_prob": 0.0}, ["deform"])
        self.assertEqual(lines, [
            "WARNING: --cascade-skip-linear skips level 4 for rigid but the level-2 source was trained with level-4 "
            "skip p = 0 for it; the frozen level-2 stage will see out-of-distribution inputs",
            "WARNING: --cascade-skip-deform skips level 4 for deform but the level-2 source was trained with level-4 "
            "skip p = none for it; the frozen level-2 stage will see out-of-distribution inputs"])

    def test_refresh_skip_record(self):
        old = {"source_level": 4, "skip_prob": 0.0, "fingerprint": {"n_vars": 1}}
        a = self.parse(["--level", "2", "--cascade-from", "y"])
        rec, changed = M._refresh_skip_record(old, a)
        self.assertIsNone(changed)                         # pre-feature record, flags off: no event
        self.assertEqual(rec, dict(old, skip_mode="off", skip_consistency_warnings=[]))
        b = self.parse(["--level", "2", "--cascade-from", "y", "--cascade-skip-linear", "4:0.3"])
        rec2, changed2 = M._refresh_skip_record(rec, b)
        self.assertEqual(rec2["skip_probs"], {"rigid": {"4": 0.3}})
        self.assertEqual(changed2["old"]["skip_mode"], "off")
        self.assertEqual(changed2["new"], {"skip_mode": "stage", "skip_probs": {"rigid": {"4": 0.3}},
                                           "skip_consistency_warnings": []})
        self.assertIsNone(M._refresh_skip_record(rec2, b)[1])
        # Level 1: the warnings come from the level-2 source's config.json (JSON only).
        tmp = tempfile.mkdtemp(prefix="bgr_skip_")
        self.addCleanup(shutil.rmtree, tmp, True)
        src_path = os.path.join(tmp, "config.json")
        with open(src_path, "w") as f:
            json.dump({"cascade": {"skip_prob": 0.0, "skip_probs": {"deform": {"4": 0.0}}}}, f)
        c = self.parse(["--level", "1", "--cascade-from", "y", "--task-ratio", "rigid:1,deform:1",
                        "--cascade-skip-deform", "4:0.15"])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rec3, changed3 = M._refresh_skip_record({"source_config": src_path, "skip_prob": 0.0}, c)
        self.assertEqual(rec3["skip_consistency_warnings"], ["deform"])
        self.assertIn("WARNING: --cascade-skip-deform skips level 4 for deform", out.getvalue())
        self.assertIsNotNone(changed3)
        rec4, _ = M._refresh_skip_record(dict(rec3, source_config=os.path.join(tmp, "missing.json")), c)
        self.assertEqual(rec4["skip_consistency_warnings"], ["deform"])    # unreadable source: carried


class TestTaskStartArgs(unittest.TestCase):
    """Retrain-prereqs 11.4 d: --task-start-step / --task-start-ramp-steps parsing and errors."""

    BASE = ["--data", "x", "--steps", "100", "--task-ratio", "rigid:1,affine:1,deform:2"]

    def parse(self, extra, base=None):
        from runtime import args as A
        return A.parse_args((self.BASE if base is None else base) + extra)

    def parse_error(self, extra, base=None):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            self.parse(extra, base)
        return err.getvalue()

    def test_valid(self):
        from runtime import args as A
        a = self.parse([])
        self.assertEqual(a.task_start_steps_parsed, {})
        self.assertEqual(a.task_start_ramp_steps, 0)
        tc = A.train_config_from_args(a)
        self.assertEqual((tc.task_start_steps, tc.task_start_ramp_steps), ({}, 0))
        b = self.parse(["--task-start-step", " deform:30 , affine:0 ", "--task-start-ramp-steps", "5"])
        self.assertEqual(b.task_start_steps_parsed, {"deform": 30, "affine": 0})
        tc = A.train_config_from_args(b)
        self.assertEqual((tc.task_start_steps, tc.task_start_ramp_steps), ({"deform": 30, "affine": 0}, 5))
        c = self.parse(["--task-start-step", "deform:0"])                 # all at 0: no ramp, allowed
        self.assertEqual(c.task_start_steps_parsed, {"deform": 0})
        # The start < steps check uses the overfit step count.
        o = self.parse(["--task-start-step", "deform:6"],
                       base=["--overfit-one-group", "--overfit-synthetic", "--overfit-max-steps", "20",
                             "--task-ratio", "rigid:1,affine:1,deform:2"])
        self.assertEqual((o.steps, o.task_start_steps_parsed), (20, {"deform": 6}))

    def test_errors(self):
        cases = [
            (["--task-start-step", "foo:10"], ["unknown task 'foo'"]),
            (["--task-start-step", "deform:10"], ["task 'deform' is not in --task-ratio"], ["--data", "x"]),
            (["--task-start-step", "deform:100"], ["deform:100 >= --steps 100", "never train"]),
            (["--task-start-step", "rigid:5,affine:5,deform:5"], ["at least one task of --task-ratio must start at step 0"]),
            (["--task-start-step", "deform:-1"], ["integer >= 0", "'deform:-1'"]),
            (["--task-start-step", "deform:1.5"], ["integer >= 0", "'deform:1.5'"]),
            (["--task-start-step", "deform"], ["needs task:step"]),
            (["--task-start-step", "deform:3,deform:4"], ["duplicate task 'deform'"]),
            (["--task-start-step", "deform:3", "--task-start-ramp-steps", "-1"], ["--task-start-ramp-steps must be >= 0"]),
            (["--task-start-ramp-steps", "4"], ["--task-start-ramp-steps needs a --task-start-step entry > 0"]),
            (["--task-start-step", "deform:0", "--task-start-ramp-steps", "4"],
             ["--task-start-ramp-steps needs a --task-start-step entry > 0"]),
        ]
        for case in cases:
            extra, needles = case[0], case[1]
            base = case[2] if len(case) > 2 else None
            msg = self.parse_error(extra, base)
            for n in needles:
                self.assertIn(n, msg, (extra, msg))


class TestMaxPairsArgs(unittest.TestCase):
    """Retrain-prereqs design 16.3 / 16.4: --max-pairs parsing, TrainConfig, startup NOTE lines."""

    BASE = ["--data", "x", "--task-ratio", "rigid:1,affine:1,deform:2", "--group-size", "32",
            "--deform-group-size", "16"]

    def parse(self, extra):
        from runtime import args as A
        return A.parse_args(self.BASE + extra)

    def test_default_and_value(self):
        from runtime import args as A
        a = self.parse([])
        self.assertEqual(a.max_pairs, 0)
        self.assertEqual(A.train_config_from_args(a).max_pairs, 0)
        b = self.parse(["--max-pairs", "64"])
        self.assertEqual(A.train_config_from_args(b).max_pairs, 64)
        self.assertEqual(A.args_to_dict(b)["max_pairs"], 64)

    def test_errors(self):
        for bad in ("-1", "1.5", "x"):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit, msg=bad):
                self.parse(["--max-pairs", bad])
            self.assertIn("--max-pairs", err.getvalue(), bad)

    def test_notes(self):
        self.assertEqual(M._pair_notes(self.parse([])), [])
        self.assertEqual(M._pair_notes(self.parse(["--max-pairs", "64"])), [])
        self.assertEqual(M._pair_notes(self.parse(["--max-pairs", "64", "--min-pairs", "n"])), [])
        notes = M._pair_notes(self.parse(["--max-pairs", "20", "--min-pairs", "n"]))
        self.assertEqual(notes, ["NOTE: max-pairs 20 < N (32): the min-pairs n floor is capped"])
        notes = M._pair_notes(self.parse(["--max-pairs", "8", "--min-pairs", "n", "--pair-recompute", "true"]))
        self.assertEqual(len(notes), 2)
        self.assertIn("(16, 32)", notes[0])
        self.assertIn("--pair-recompute is a level-1 option", notes[1])


class TestLevel1CascadeMain(MainCase):
    """Milestone 9 phase 2, T-E1 (design-m9-phase2 section 11): main.py level 4 -> level 2 ->
    level 1 overfit chain (TINY widths, N 2, deform decoder built at every level so the three
    ModelConfigs are dict-equal), a resume of the level-1 run and the in-process checker.
    Local deviation from the design (GTX 1080, 6000 MB cap): the deform train step does not fit
    at level 2 (fp32, default sampler: host-pinned int32 sampler indices OOM) nor at level 1 (even
    with every memory knob, FEAT-002 probe), so all runs train rigid only, and level 1 uses AMP +
    --split-group-conv (the float32 level-1 build pass with the deform decoder exceeds the cap).
    The level-1 deform chain branches are covered by tests/test_cascade.py (T-C2 / T-C6 / T-V1)."""

    def test_level1_chain_resume_and_checker(self):
        import importlib.util
        import tensorflow as tf
        self.addCleanup(tf.keras.mixed_precision.set_global_policy, "float32")       # --amp true
        res = os.path.join(self.tmp, "results")
        common = ["--overfit-one-group", "--overfit-synthetic", "--task-ratio", "rigid:1", "--deform-decoder", "true",
                  "--deform-residual-affine", "false", "--group-size", "2", "--log-every", "1", "--val-png", "false",
                  "--results", res] + TINY_ARGS + COMMON
        t0 = time.time()
        self.assertIn(M.main(common + ["--level", "4", "--run-name", "r1", "--overfit-max-steps", "3"]), (0, 1))
        free_memory()
        self.assertIn(M.main(common + ["--level", "2", "--run-name", "r2", "--cascade-from",
                                       os.path.join(self.run_dir("r1"), "ckpt", "last"),
                                       "--overfit-max-steps", "3", "--bn-recal-batches", "4"]), (0, 1))
        free_memory()
        run3 = common + ["--level", "1", "--run-name", "r3", "--cascade-from",
                         os.path.join(self.run_dir("r2"), "ckpt", "last"), "--deform-group-size", "2",
                         "--min-pairs", "n", "--amp", "true", "--split-group-conv", "true",
                         "--bn-recal-batches", "4", "--val-every", "3", "--val-png", "true"]
        self.assertIn(M.main(run3 + ["--overfit-max-steps", "3"]), (0, 1))
        free_memory()
        self.assertIn(M.main(run3 + ["--overfit-max-steps", "5"]), (0, 1))            # resume
        free_memory()
        print("\n[T-E1] level 4 -> 2 -> 1 + resume: %.0f s" % (time.time() - t0), file=sys.stderr)
        d = self.run_dir("r3")
        with open(os.path.join(d, "config.json")) as f:
            cfg = json.load(f)
        cas, iw = cfg["cascade"], cfg["init_weights"]
        self.assertEqual(cas["source_level"], 2)
        self.assertEqual([s["level"] for s in cas["stages"]], [4, 2])
        self.assertEqual([s["checkpoint_key"] for s in cas["stages"]], ["frozen_stage", "model"])
        self.assertEqual([s["origin_step"] for s in cas["stages"]], [3, 3])
        self.assertEqual((iw["via"], iw["cross_level"]), ("cascade", [2, 1]))
        self.assertEqual(cfg["args"]["min_pairs"], "n")
        keys = [k for k, _ in tf.train.list_variables(os.path.join(d, "ckpt", "last"))]
        self.assertTrue(any(k.startswith("frozen_stage_4/") for k in keys))
        self.assertTrue(any(k.startswith("frozen_stage_2/") for k in keys))
        self.assertFalse(any(k.startswith("frozen_stage/") for k in keys))
        with open(os.path.join(d, "logs", "metrics.jsonl")) as f:
            recs = [json.loads(line) for line in f if line.strip()]
        n_pairs = [r["data"]["train_rigid/n_pairs"] for r in recs if "train_rigid/n_pairs" in r["data"]]
        self.assertTrue(n_pairs)
        self.assertTrue(all(v == 2 for v in n_pairs), n_pairs)
        final = [r["data"] for r in recs if r.get("step") == 5 and "val_rigid/center_rms_mm" in r["data"]]
        self.assertTrue(final)
        self.assertTrue(np.isfinite(final[-1]["val_rigid_frozen4/center_rms_mm"]))
        self.assertTrue(np.isfinite(final[-1]["val/wall_s"]))
        self.assertEqual(sum(1 for r in recs if r["data"].get("event") == "bn_recalibration"), 1)
        self.assertTrue(os.listdir(os.path.join(d, "ckpt", "best")))
        self.assertTrue(os.path.exists(os.path.join(d, "val_png", "rigid_step0000005.png")))
        spec = importlib.util.spec_from_file_location(
            "check_mixed_run", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu", "check_mixed_run.py"))
        cmr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cmr)
        ok, results = cmr.check(d, ["rigid"], 5, expect_best=True, expect_cascade=True, expect_cascade_stages=2,
                                expect_min_pairs="n")
        self.assertTrue(ok, [r for r in results if not r[1]])
        self.assertTrue(all(p for _, p, _ in results))
        self.assertIn("cascade_stages", [n for n, _, _ in results])


class TestLevel1CascadeErrors(MainCase):
    """Milestone 9 phase 2 (design 4.3, 10): main.py --level 1 --cascade-from failures exit with
    'main.py: --cascade-from ...'; _check_cascade_fingerprints names the level."""

    def fake_source(self, common):
        """A level-2 'cascade run' whose checkpoint holds only a global step (no model /
        frozen_stage objects) and whose config.json has a cascade record."""
        import tensorflow as tf
        from runtime import args as A
        from runtime import checkpoint as C
        from runtime.logging import make_run_dir
        paths = make_run_dir(os.path.join(self.tmp, "results"), "m8")
        st = tf.Variable(4, dtype=tf.int64)
        tf.train.CheckpointManager(tf.train.Checkpoint(step=st), paths["ckpt_last"], 1,
                                   checkpoint_name="last").save(checkpoint_number=4)
        args = A.parse_args(common + ["--level", "2"])
        rec = {"source_checkpoint": "/x/src4/ckpt/last/last-3", "source_level": 4, "source_step": 3,
               "fingerprint": {"n_vars": 7, "abs_sum": 1.0, "sq_sum": 1.0}, "skip_prob": 0.0}
        C.write_config(paths["config"], A.model_config_from_args(args), {"level": 2}, extra={"cascade": rec})
        return paths["ckpt_last"]

    def test_level1_cascade_from_errors(self):
        import tensorflow as tf
        self.addCleanup(tf.keras.mixed_precision.set_global_policy, "float32")       # main.py --amp true
        res = os.path.join(self.tmp, "results")
        common = ["--overfit-one-group", "--overfit-synthetic", "--group-size", "2", "--task-ratio", "rigid:1",
                  "--deform-decoder", "true", "--log-every", "1", "--val-every", "0", "--val-png", "false",
                  "--results", res] + TINY_ARGS + COMMON
        with self.assertRaises(SystemExit) as cm:          # unresolvable prefix: before any model
            M.main(common + ["--level", "1", "--cascade-from", os.path.join(self.tmp, "nope", "ckpt", "best"),
                             "--run-name", "bad"])
        msg = str(cm.exception.code)
        self.assertTrue(msg.startswith("main.py: --cascade-from"), msg)
        self.assertFalse(os.path.exists(self.run_dir("bad")))
        # A level-2 cascade source whose checkpoint lacks the model objects: the FrozenChain restore
        # (TF AssertionError, wrapped into ValueError) becomes the same SystemExit.
        src = self.fake_source(common)
        # Phase-2 memory knobs: the level-1 build pass (deform decoder) exceeds the 6000 MB cap in float32.
        knobs = ["--amp", "true", "--gradient-checkpointing", "full", "--lean-sampler", "true",
                 "--split-group-conv", "true", "--task-ratio", "rigid:1,deform:1", "--deform-residual-affine", "false"]
        # The level-1 build pass only creates the variables (they do not depend on N), and this
        # test checks the error routing after it. In the full local suite the eager level-1 deform
        # build at N 2 OOMed inside the fragmented 6000 MB region (twice, tf-intensity-aug report),
        # so the build runs with N 1 here (half the activations).
        from unittest import mock
        from models.group_net import GroupRegLevelModel
        orig_build = GroupRegLevelModel.build_model

        def small_build(model, n=1):
            return orig_build(model, 1)

        with self.assertRaises(SystemExit) as cm, mock.patch.object(GroupRegLevelModel, "build_model", small_build):
            M.main(common + knobs + ["--level", "1", "--cascade-from", src, "--run-name", "bad2"])
        msg = str(cm.exception.code)
        print("\n[level-1 cascade-from] %s" % msg[:200], file=sys.stderr)
        self.assertTrue(msg.startswith("main.py: --cascade-from"), msg)
        self.assertIn("FrozenChain.from_checkpoint", msg)
        self.assertIn("AssertionError", msg)

    def test_check_cascade_fingerprints(self):
        fp4 = {"n_vars": 5, "abs_sum": 2.0, "sq_sum": 1.5}
        fp2 = {"n_vars": 6, "abs_sum": 3.0, "sq_sum": 2.5}

        class FakeChain:
            level = 2
            fingerprints = {4: fp4, 2: fp2}
            fingerprint = fp2

        rec = {"fingerprint": dict(fp2), "stages": [{"level": 4, "fingerprint": dict(fp4)},
                                                    {"level": 2, "fingerprint": dict(fp2)}]}
        M._check_cascade_fingerprints(rec, FakeChain, "config.json")                 # no error
        for lv in (4, 2):
            bad = json.loads(json.dumps(rec))
            st = [s for s in bad["stages"] if s["level"] == lv][0]
            st["fingerprint"]["abs_sum"] *= 1.0 + 1e-9
            with self.assertRaises(RuntimeError) as cm:
                M._check_cascade_fingerprints(bad, FakeChain, "config.json")
            self.assertIn("level-%d" % lv, str(cm.exception))
        bad = dict(rec, fingerprint=dict(fp2, n_vars=99))
        with self.assertRaises(RuntimeError) as cm:
            M._check_cascade_fingerprints(bad, FakeChain, "config.json")
        self.assertIn("cascade.fingerprint", str(cm.exception))


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
