"""Args (runtime.args), logging, checkpoints and TF-side fixed data (runtime.fixed_data)."""
import argparse
import json
import os
import shutil
import tempfile
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests.runtime_helpers import free_memory, narrow_config, write_fake_export, NARROW_ARGS
import numpy as np
import tensorflow as tf

from data_loading import params as P
from data_loading.postprocess import LinearPostprocess
from models.group_net import GroupRegLevelModel
from runtime import args as A
from runtime import checkpoint as C
from runtime import fixed_data as fd
from runtime import metrics as M
from runtime import optimizers as O
from runtime.logging import RunLogger, make_run_dir


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_m4_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        free_memory()


class TestArgs(unittest.TestCase):
    def test_str2bool(self):
        for v in ("yes", "True", "t", "1", "on", True):
            self.assertTrue(A.str2bool(v))
        for v in ("no", "False", "f", "0", "off", False):
            self.assertFalse(A.str2bool(v))
        with self.assertRaises(argparse.ArgumentTypeError):
            A.str2bool("maybe")

    def test_level_defaults_and_overfit_dropout(self):
        for level, n in ((4, 32), (2, 8), (1, 2)):
            self.assertEqual(A.parse_args(["--data", "x", "--level", str(level)]).group_size, n)
        self.assertEqual(A.parse_args(["--data", "x"]).dropout, 0.15)
        self.assertEqual(A.parse_args(["--overfit-one-group", "--overfit-synthetic"]).dropout, 0.0)
        self.assertEqual(A.parse_args(["--overfit-one-group", "--overfit-synthetic", "--dropout", "0.2"]).dropout, 0.2)
        a = A.parse_args(["--data", "x", "--val-every", "100"])
        self.assertEqual(a.ckpt_every, 100)
        self.assertTrue(a.run_name.startswith("l4_rigid1_"))

    def test_model_config_round_trip(self):
        a = A.parse_args(["--data", "x", "--group-config", "B", "--translation-scale", "10"] + NARROW_ARGS)
        cfg = A.model_config_from_args(a)
        self.assertEqual(cfg.group_config, "B")
        self.assertEqual(cfg.translation_scale, 10.0)
        self.assertEqual(cfg.filters, (16, 32, 32, 48, 64, 96))
        self.assertEqual(type(cfg).from_dict(cfg.to_dict()), cfg)
        tc = A.train_config_from_args(a)
        self.assertEqual((tc.level, tc.group_size, tc.lambda_pair, tc.p_pair), (4, 32, 1.0, 0.1))
        self.assertEqual(tc.schedule.base_lr, 3e-4)
        self.assertEqual(a.optimizer, "adam")
        self.assertEqual(A.ranges_from_args(a)["rigid"], dict(P.DEFAULT_LINEAR_RANGES["rigid"]))
        json.dumps(A.args_to_dict(a))

    def test_loss_scale_and_head_activation_flags(self):
        a = A.parse_args(["--data", "x"])
        self.assertEqual((a.loss_scale_mm2, a.head_activation), (1.0, "relu"))
        self.assertEqual(A.train_config_from_args(a).loss_scale_mm2, 1.0)
        self.assertEqual(A.model_config_from_args(a).head_activation, "relu")
        a = A.parse_args(["--data", "x", "--loss-scale-mm2", "100", "--head-activation", "none"])
        self.assertEqual(A.train_config_from_args(a).loss_scale_mm2, 100.0)
        self.assertEqual(A.model_config_from_args(a).head_activation, "none")
        d = A.args_to_dict(a)
        self.assertEqual((d["loss_scale_mm2"], d["head_activation"]), (100.0, "none"))
        import contextlib
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            for bad in (["--loss-scale-mm2", "0"], ["--loss-scale-mm2", "-1"], ["--head-activation", "gelu"]):
                with self.assertRaises(SystemExit):
                    A.parse_args(["--data", "x"] + bad)

    def test_best_metric_defaults(self):
        # Milestone 5: combined val/mean_pair_rms_mm by default with more than one task.
        from runtime.run import MEAN_PAIR_METRIC, resolve_best_metric
        self.assertEqual(A.MEAN_PAIR_METRIC, MEAN_PAIR_METRIC)
        cases = [([], "val_rigid/pair_rms_mm"),
                 (["--task-ratio", "affine:1"], "val_affine/pair_rms_mm"),
                 (["--task-ratio", "rigid:1,affine:1"], MEAN_PAIR_METRIC),
                 (["--best-metric", "mean_pair_rms_mm"], MEAN_PAIR_METRIC),
                 (["--task-ratio", "rigid:1,affine:1", "--best-metric", "val_rigid/pair_rms_mm"],
                  "val_rigid/pair_rms_mm"),
                 (["--task-ratio", "affine:1", "--best-metric", "val_rigid/center_rms_mm"],
                  "val_affine/center_rms_mm")]
        for extra, want in cases:
            a = A.parse_args(["--data", "x"] + extra)
            self.assertEqual(a.best_metric, want, extra)
            self.assertEqual(A.train_config_from_args(a).best_metric, want)
        self.assertEqual(resolve_best_metric(None, ["rigid"]), "val_rigid/pair_rms_mm")
        self.assertEqual(resolve_best_metric(None, ["rigid", "affine"]), MEAN_PAIR_METRIC)
        self.assertEqual(resolve_best_metric("mean_pair_rms_mm", ["rigid"]), MEAN_PAIR_METRIC)
        self.assertEqual(resolve_best_metric("val_x/y", ["rigid", "affine"]), "val_x/y")

    def test_ratio_and_required_data(self):
        self.assertEqual(dict(A.parse_task_ratio("rigid:3, affine:1")), {"rigid": 3.0, "affine": 1.0})
        # Milestone 6: deform is accepted (was NotImplementedError before).
        self.assertEqual(dict(A.parse_task_ratio("rigid:1,deform:2")), {"rigid": 1.0, "deform": 2.0})
        self.assertEqual(list(A.parse_task_ratio("deform:1,affine:1")), ["deform", "affine"])
        with self.assertRaises(ValueError):
            A.parse_task_ratio("rigid:0")
        with self.assertRaises(ValueError):
            A.parse_task_ratio("warp:1")
        a = A.parse_args(["--data", "x", "--task-ratio", "deform:1"])
        self.assertEqual(dict(a.task_ratio_parsed), {"deform": 1.0})
        self.assertTrue(a.run_name.startswith("l4_deform1_"))
        devnull = open(os.devnull, "w")
        try:
            import contextlib
            with contextlib.redirect_stderr(devnull):
                with self.assertRaises(SystemExit):
                    A.parse_args([])                      # --data missing
        finally:
            devnull.close()

    def test_deform_flag_defaults(self):
        a = A.parse_args(["--data", "x"])
        want = {"lambda_pdice": 1.0, "p_pdice": 0.1, "lambda_smooth": 0.5, "smooth_order": 1,
                "integration_steps": 7, "decoder_widths": None, "deform_decoder": "auto",
                "deform_residual_affine": False, "deform_group_size": 32,
                "gradient_checkpointing": "none", "overfit_dice_threshold": 0.9,
                "overfit_fold_max": 0.001, "val_deform_max_pairs": 64}
        d = A.args_to_dict(a)
        for k, v in want.items():
            self.assertEqual(getattr(a, k), v, k)
            self.assertIn(k, d)                                  # saved in config.json args
        mc = A.model_config_from_args(a)
        self.assertEqual((mc.deform_decoder, mc.decoder_widths, mc.integration_steps), (False, None, 7))
        tc = A.train_config_from_args(a)
        for k in ("lambda_pdice", "p_pdice", "lambda_smooth", "smooth_order", "deform_residual_affine",
                  "deform_group_size", "overfit_dice_threshold", "overfit_fold_max", "val_deform_max_pairs"):
            self.assertEqual(getattr(tc, k), want[k], k)
        self.assertEqual(tc.n_classes, P.N_TRAIN_LABELS)
        # --deform-decoder auto: on iff deform is trained; true forces it for a linear run.
        self.assertTrue(A.model_config_from_args(A.parse_args(["--data", "x", "--task-ratio", "deform:1"])).deform_decoder)
        self.assertTrue(A.model_config_from_args(A.parse_args(["--data", "x", "--deform-decoder", "true"])).deform_decoder)
        a = A.parse_args(["--data", "x", "--task-ratio", "rigid:1,deform:1", "--group-size", "8",
                          "--deform-group-size", "4", "--decoder-widths", "8,8,8,4,4", "--integration-steps", "5",
                          "--lambda-pdice", "2", "--p-pdice", "0.5", "--lambda-smooth", "1", "--smooth-order", "2",
                          "--deform-residual-affine", "true", "--gradient-checkpointing", "both",
                          "--overfit-dice-threshold", "0.8", "--overfit-fold-max", "0.01",
                          "--val-deform-max-pairs", "12"])
        mc = A.model_config_from_args(a)
        self.assertEqual((mc.deform_decoder, mc.decoder_widths, mc.integration_steps), (True, (8, 8, 8, 4, 4), 5))
        tc = A.train_config_from_args(a)
        self.assertEqual((tc.group_size, tc.deform_group_size, tc.lambda_pdice, tc.p_pdice, tc.lambda_smooth,
                          tc.smooth_order, tc.deform_residual_affine, tc.overfit_dice_threshold,
                          tc.overfit_fold_max, tc.val_deform_max_pairs),
                         (8, 4, 2.0, 0.5, 1.0, 2, True, 0.8, 0.01, 12))
        self.assertEqual(a.gradient_checkpointing, "both")
        json.dumps(A.args_to_dict(a))
        import contextlib
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            for bad in (["--task-ratio", "deform:1", "--deform-decoder", "false"],
                        ["--decoder-widths", "8,8"], ["--deform-group-size", "1"], ["--smooth-order", "3"],
                        ["--gradient-checkpointing", "all"], ["--p-pdice", "0"], ["--integration-steps", "-1"],
                        ["--val-deform-max-pairs", "0"]):
                with self.assertRaises(SystemExit, msg=str(bad)):
                    A.parse_args(["--data", "x"] + bad)

    def test_ncc_flags(self):
        # v3.9 LNCC: defaults off (Dice weights 1), window per level; values reach TrainConfig.
        a = A.parse_args(["--data", "x"])
        self.assertEqual((a.lambda_dice_template, a.lambda_ncc_template, a.lambda_ncc_pair, a.ncc_window),
                         (1.0, 0.0, 0.0, 5))
        tc = A.train_config_from_args(a)
        self.assertEqual((tc.lambda_dice_tmpl, tc.lambda_ncc_tmpl, tc.lambda_ncc_pair, tc.ncc_window),
                         (1.0, 0.0, 0.0, 5))
        for k in ("lambda_dice_template", "lambda_ncc_template", "lambda_ncc_pair", "ncc_window"):
            self.assertIn(k, A.args_to_dict(a))
        self.assertEqual(A.parse_args(["--data", "x", "--level", "2"]).ncc_window, 7)
        self.assertEqual(A.parse_args(["--data", "x", "--level", "1"]).ncc_window, 9)
        a = A.parse_args(["--data", "x", "--task-ratio", "deform:1", "--lambda-dice-template", "0",
                          "--lambda-pdice", "0", "--lambda-ncc-template", "1.5", "--lambda-ncc-pair", "0.5",
                          "--ncc-window", "3"])
        tc = A.train_config_from_args(a)
        self.assertEqual((tc.lambda_dice_tmpl, tc.lambda_pdice, tc.lambda_ncc_tmpl, tc.lambda_ncc_pair,
                          tc.ncc_window), (0.0, 0.0, 1.5, 0.5, 3))
        # All Dice weights 0 is fine for a linear-only run (deform not trained).
        self.assertEqual(A.parse_args(["--data", "x", "--lambda-dice-template", "0", "--lambda-pdice",
                                       "0"]).lambda_pdice, 0.0)
        import contextlib
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            for bad in (["--lambda-ncc-template", "-1"], ["--lambda-ncc-pair", "nan"],
                        ["--lambda-dice-template", "inf"], ["--ncc-window", "4"], ["--ncc-window", "1"],
                        ["--ncc-window", "41"],
                        ["--task-ratio", "deform:1", "--lambda-dice-template", "0", "--lambda-pdice", "0"]):
                with self.assertRaises(SystemExit, msg=str(bad)):
                    A.parse_args(["--data", "x"] + bad)

    def test_task_loss_weights_flag(self):
        # Milestone 7: default all 1 (linear-only runs unchanged), missing tasks = 1.
        a = A.parse_args(["--data", "x"])
        self.assertIsNone(a.task_loss_weights)
        self.assertEqual(a.task_loss_weights_parsed, {"rigid": 1.0, "affine": 1.0, "deform": 1.0})
        self.assertEqual(A.train_config_from_args(a).task_loss_weights, {"rigid": 1.0, "affine": 1.0, "deform": 1.0})
        a = A.parse_args(["--data", "x", "--task-ratio", "rigid:1,affine:1,deform:2", "--task-loss-weights",
                          "deform:4, affine:0.5"])
        self.assertEqual(A.train_config_from_args(a).task_loss_weights, {"rigid": 1.0, "affine": 0.5, "deform": 4.0})
        json.dumps(A.args_to_dict(a))
        self.assertEqual(A.parse_task_loss_weights(""), {"rigid": 1.0, "affine": 1.0, "deform": 1.0})
        import contextlib
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            for bad in ("deform:0", "foo:1", "rigid:nan", "rigid:-1", "rigid:inf", "deform"):
                with self.assertRaises(SystemExit, msg=bad):
                    A.parse_args(["--data", "x", "--task-loss-weights", bad])

    def test_task_grad_norm_flag(self):
        # v3.10: per-task gradient normalisation, default off (0) with minimum raw norm 1e-3.
        a = A.parse_args(["--data", "x"])
        self.assertEqual((a.task_grad_norm, a.task_grad_norm_min), (0.0, 1e-3))
        tc = A.train_config_from_args(a)
        self.assertEqual((tc.task_grad_norm, tc.task_grad_norm_min), (0.0, 1e-3))
        a = A.parse_args(["--data", "x", "--task-grad-norm", "12", "--task-grad-norm-min", "0"])
        tc = A.train_config_from_args(a)
        self.assertEqual((tc.task_grad_norm, tc.task_grad_norm_min), (12.0, 0.0))
        json.dumps(A.args_to_dict(a))
        import contextlib
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            for flag in ("--task-grad-norm", "--task-grad-norm-min"):
                for bad in ("-1", "nan", "inf"):
                    with self.assertRaises(SystemExit, msg="%s %s" % (flag, bad)):
                        A.parse_args(["--data", "x", flag, bad])

    def test_best_metric_with_deform(self):
        from runtime.run import DEFORM_BEST_METRIC, MEAN_PAIR_METRIC, resolve_best_metric
        self.assertEqual(A.DEFORM_BEST_METRIC, DEFORM_BEST_METRIC)
        cases = [("deform:1", DEFORM_BEST_METRIC), ("rigid:1,deform:2", "val_rigid/pair_rms_mm"),
                 ("deform:1,affine:1", "val_affine/pair_rms_mm"),
                 ("rigid:1,affine:1,deform:2", MEAN_PAIR_METRIC)]
        for ratio, want in cases:
            a = A.parse_args(["--data", "x", "--task-ratio", ratio])
            self.assertEqual(a.best_metric, want, ratio)
            self.assertEqual(resolve_best_metric(None, list(a.task_ratio_parsed)), want, ratio)
        a = A.parse_args(["--data", "x", "--task-ratio", "deform:1,affine:1", "--best-metric", "val_rigid/center_rms_mm"])
        self.assertEqual(a.best_metric, "val_affine/center_rms_mm")

    def test_controls_skip_deform(self):
        a = A.parse_args(["--data", "x", "--task-ratio", "rigid:1,deform:1", "--curriculum", "true",
                          "--global-transform", "true"])
        self.assertEqual(sorted(A.task_controls_from_args(a)), ["rigid"])
        self.assertEqual(A.linear_tasks(a), ["rigid"])
        import contextlib
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            for flag in ("--curriculum", "--global-transform"):
                with self.assertRaises(SystemExit):
                    A.parse_args(["--data", "x", "--task-ratio", "deform:1", flag, "true"])


class TestLogging(TempDirCase):
    def test_jsonl_and_tensorboard(self):
        paths = make_run_dir(self.tmp, "run")
        for k in ("logs", "tb", "ckpt_last", "ckpt_best"):
            self.assertTrue(os.path.isdir(paths[k]), k)
        log = RunLogger(paths, backend="jsonl", quiet=True)
        log.log_params({"lr": 0.01, "name": "x", "arr": np.arange(3)})
        for s in (1, 2, 3):
            log.log_metrics(s, {"loss": np.float32(1.5 * s), "rms": tf.constant(2.0 * s), "nan": float("nan")})
        log.close()
        with open(log.metrics_path) as f:
            lines = [json.loads(line) for line in f]
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0]["step"], "PARAMETER")
        self.assertEqual(lines[0]["data"]["arr"], [0, 1, 2])
        self.assertEqual([r["step"] for r in lines[1:]], [1, 2, 3])
        self.assertEqual(lines[3]["data"]["loss"], 4.5)
        self.assertEqual(lines[3]["data"]["rms"], 6.0)
        self.assertIsInstance(lines[3]["data"]["loss"], float)
        self.assertEqual(lines[1]["data"]["nan"], "nan")
        self.assertEqual(log.backend_name, "jsonl")
        self.assertTrue(any(n.startswith("events.out.tfevents") for n in os.listdir(paths["tb"])))


class TestCheckpoint(TempDirCase):
    def make(self):
        model = GroupRegLevelModel(4, narrow_config(dropout=0.0)).build_model(3)
        opt = O.make_optimizer("sgdw", lr=0.01)
        O.build_optimizer(opt, model.trainable_variables)
        step = tf.Variable(0, dtype=tf.int64)
        gen = tf.random.Generator.from_seed(5)
        ck = C.RunCheckpoint(os.path.join(self.tmp, "last"), os.path.join(self.tmp, "best"), model, opt,
                             step, extra={"gen": gen})
        return model, opt, step, gen, ck

    def train_two_steps(self, model, opt, step):
        x = tf.random.stateless_uniform((3,) + P.LEVEL_SHAPE[4] + (1,), seed=(1, 2))
        for _ in range(2):
            with tf.GradientTape() as tape:
                a = model(x, task="rigid", training=True)["A"]
                loss = tf.reduce_sum(tf.square(a[:, :3, 3] - 1.0))
            tv = model.trainable_variables
            gv = [(g, v) for g, v in zip(tape.gradient(loss, tv), tv) if g is not None]
            opt.apply_gradients(gv)
            step.assign_add(1)

    def test_round_trip_bit_exact(self):
        model, opt, step, gen, ck = self.make()
        self.train_two_steps(model, opt, step)
        gen.uniform([3])
        ck.save_last()
        ref_model = [v.numpy() for v in model.variables]
        ref_opt = [v.numpy() for v in opt.variables()]
        ref_gen = gen.state.numpy()
        model2, opt2, step2, gen2, ck2 = self.make()
        self.assertTrue(ck2.restore_last())
        self.assertEqual(int(step2.numpy()), 2)
        self.assertEqual(int(opt2.iterations.numpy()), 2)
        self.assertEqual(len(ref_opt), len(opt2.variables()))
        for a, b in zip(ref_model, model2.variables):
            self.assertTrue(np.array_equal(a, b.numpy()), b.name)
        for a, b in zip(ref_opt, opt2.variables()):
            self.assertTrue(np.array_equal(a, b.numpy()), b.name)
        np.testing.assert_array_equal(gen2.state.numpy(), ref_gen)
        bn = [v for v in model2.non_trainable_variables if "moving_mean" in v.name]
        self.assertTrue(any(np.any(v.numpy() != 0) for v in bn))   # statistics moved and restored

    def test_best_and_keep_last(self):
        model, opt, step, gen, ck = self.make()
        for s, val in ((1, 5.0), (2, 6.0), (3, 4.0)):
            step.assign(s)
            saved = ck.maybe_save_best(val)           # Trainer order: validate, best, then last
            ck.save_last()
            self.assertEqual(saved, val != 6.0)
        self.assertFalse(ck.maybe_save_best(float("nan")))
        self.assertTrue(ck.latest_best.endswith("best-3"))
        self.assertEqual(len(ck.last.checkpoints), 2)
        self.assertTrue(ck.latest_last.endswith("last-3"))
        model2, opt2, step2, gen2, ck2 = self.make()
        ck2.restore_last()
        self.assertEqual(float(ck2.best_metric.numpy()), 4.0)

    def test_extra_best(self):
        # Milestone 6: ckpt/best_deform next to ckpt/best (deform + linear runs only).
        model, opt, step, gen, _ = self.make()
        ck = C.RunCheckpoint(os.path.join(self.tmp, "last2"), os.path.join(self.tmp, "best2"), model, opt, step,
                             extra={"gen": gen}, extra_best={"deform": os.path.join(self.tmp, "best_deform")})
        self.assertIsNone(ck.latest_best_named("deform"))
        for s, lin, dfm in ((1, 5.0, 0.5), (2, 4.0, 0.6), (3, 6.0, 0.3)):
            step.assign(s)
            self.assertEqual(ck.maybe_save_best(lin), lin != 6.0)
            self.assertEqual(ck.maybe_save_best(dfm, name="deform"), dfm != 0.6)
            ck.save_last()
        self.assertTrue(ck.latest_best.endswith("best-2"))
        self.assertTrue(ck.latest_best_named("deform").endswith("best_deform-3"))
        self.assertEqual((float(ck.best_metric.numpy()), float(ck.extra_best_metric["deform"].numpy())), (4.0, 0.3))
        self.assertFalse(ck.maybe_save_best(float("nan"), name="deform"))
        with self.assertRaises(KeyError):
            ck.maybe_save_best(0.1, name="other")
        # Restores with the extra variable, and into a RunCheckpoint without it.
        model2, opt2, step2, gen2, _ = self.make()
        ck2 = C.RunCheckpoint(os.path.join(self.tmp, "last2"), os.path.join(self.tmp, "x"), model2, opt2, step2,
                              extra={"gen": gen2}, extra_best={"deform": os.path.join(self.tmp, "y")})
        self.assertTrue(ck2.restore_last())
        self.assertEqual(float(ck2.extra_best_metric["deform"].numpy()), 0.3)
        model3, opt3, step3, gen3, ck3 = self.make()
        ck3 = C.RunCheckpoint(os.path.join(self.tmp, "last2"), os.path.join(self.tmp, "z"), model3, opt3, step3,
                              extra={"gen": gen3})
        self.assertTrue(ck3.restore_last())
        self.assertEqual((int(step3.numpy()), float(ck3.best_metric.numpy())), (3, 4.0))
        # An old checkpoint (no extra best) restores into an old-style RunCheckpoint unchanged.
        _, _, step4, _, ck4 = self.make()
        step4.assign(7)
        ck4.save_last()
        _, _, step5, _, ck5 = self.make()
        self.assertTrue(ck5.restore_last())
        self.assertEqual(int(step5.numpy()), 7)
        self.assertEqual(ck5.extra_best, {})

    def test_load_model_weights_warm_start(self):
        # --init-weights: model variables only, from a saved tiny run; config mismatch raises.
        root = os.path.join(self.tmp, "src")
        paths = make_run_dir(self.tmp, "src")
        model = GroupRegLevelModel(4, narrow_config(dropout=0.0)).build_model(3)
        opt = O.make_optimizer("adam", lr=1e-3)
        O.build_optimizer(opt, model.trainable_variables)
        step = tf.Variable(0, dtype=tf.int64)
        ck = C.RunCheckpoint(paths["ckpt_last"], paths["ckpt_best"], model, opt, step)
        self.train_two_steps(model, opt, step)
        ck.save_last()
        C.write_config(paths["config"], narrow_config(dropout=0.0), {})
        ref = [v.numpy() for v in model.variables]
        self.assertTrue(any(np.any(v != 0) for v in [w.numpy() for w in model.linear_decoder.rigid_head.weights]))
        for src in (paths["ckpt_last"], ck.latest_last):              # directory and prefix forms
            m2 = GroupRegLevelModel(4, narrow_config(dropout=0.0, seed=99)).build_model(3)
            opt2 = O.make_optimizer("adam", lr=1e-3)
            O.build_optimizer(opt2, m2.trainable_variables)
            step2 = tf.Variable(0, dtype=tf.int64)
            prefix, cfg = C.load_model_weights(m2, narrow_config(dropout=0.0, seed=99), src)
            self.assertEqual(prefix, ck.latest_last)
            self.assertEqual(os.path.abspath(cfg), os.path.abspath(os.path.join(root, "config.json")))
            for a, b in zip(ref, m2.variables):
                self.assertTrue(np.array_equal(a, b.numpy()), b.name)
            self.assertEqual(int(opt2.iterations.numpy()), 0)        # optimizer not loaded
            self.assertTrue(all(not np.any(v.numpy()) for v in opt2.variables()))
            self.assertEqual(int(step2.numpy()), 0)
        m3 = GroupRegLevelModel(4, narrow_config(dropout=0.0, translation_scale=10.0)).build_model(3)
        with self.assertRaises(ValueError) as cm:
            C.load_model_weights(m3, narrow_config(dropout=0.0, translation_scale=10.0), paths["ckpt_last"])
        self.assertIn("translation_scale", str(cm.exception))
        with self.assertRaises(ValueError):
            C.load_model_weights(m3, narrow_config(dropout=0.0), os.path.join(self.tmp, "missing"))
        with self.assertRaises(ValueError):
            C.load_model_weights(m3, narrow_config(dropout=0.0), paths["ckpt_best"])   # empty dir

    def save_source(self, name, cfg):
        """A narrow model of `cfg` trained 2 rigid steps, saved as <tmp>/<name> with config.json.
        Returns (ckpt/last dir, source model)."""
        paths = make_run_dir(self.tmp, name)
        model = GroupRegLevelModel(4, cfg).build_model(3)
        opt = O.make_optimizer("adam", lr=1e-3)
        O.build_optimizer(opt, model.trainable_variables)
        step = tf.Variable(0, dtype=tf.int64)
        ck = C.RunCheckpoint(paths["ckpt_last"], paths["ckpt_best"], model, opt, step)
        self.train_two_steps(model, opt, step)
        ck.save_last()
        C.write_config(paths["config"], cfg, {})
        return paths["ckpt_last"], model

    @staticmethod
    def part_values(model, part):
        return [v.numpy().copy() for v in getattr(model, part).weights]

    def assert_parts_equal(self, src, dst, parts=C.SHARED_PARTS):
        for part in parts:
            a, b = self.part_values(src, part), self.part_values(dst, part)
            self.assertEqual(len(a), len(b), part)
            for x, y, v in zip(a, b, getattr(dst, part).weights):
                self.assertTrue(np.array_equal(x, y), v.name)

    def test_warm_start_linear_into_deform(self):
        # Milestone 7: a linear-only checkpoint warm-starts a deform model (encoder + linear heads
        # restored, deform decoder fresh); the optimizer is never read.
        src_dir, src = self.save_source("lin", narrow_config(dropout=0.0))
        bn = [v for v in src.encoder.weights if "moving_" in v.name]
        self.assertTrue(bn)
        self.assertTrue(any(np.any(v.numpy() != (0.0 if "moving_mean" in v.name else 1.0)) for v in bn))
        cases = {"default": dict(deform_decoder=True),
                 "widths_steps": dict(deform_decoder=True, decoder_widths=(8, 8, 8, 4, 4), integration_steps=5)}
        for label, kw in cases.items():
            with self.subTest(label):
                cfg = narrow_config(dropout=0.0, seed=99, **kw)
                m = GroupRegLevelModel(4, cfg).build_model(3)
                fresh0 = self.part_values(m, "deform_decoder")
                enc0 = self.part_values(m, "encoder")
                info = C.warm_start_model(m, cfg, src_dir)
                self.assertEqual(info["mode"], "partial")
                self.assertEqual(info["fresh_parts"], {"deform_decoder": len(m.deform_decoder.weights)})
                self.assertEqual(info["ignored_source_parts"], [])
                self.assertEqual(sorted(info["restored_parts"]), sorted(C.SHARED_PARTS))
                self.assertEqual(info["n_restored_vars"] + info["n_fresh_vars"], len(m.weights))
                self.assertIn("deform_decoder", info["config_diff"])
                self.assertEqual(info["config_diff"]["deform_decoder"], [False, True])
                if label == "widths_steps":
                    self.assertEqual(sorted(info["config_diff"]),
                                     ["decoder_widths", "deform_decoder", "integration_steps"])
                self.assertTrue(info["checkpoint"].endswith("last-2"))
                self.assert_parts_equal(src, m)
                self.assertFalse(all(np.array_equal(a, v.numpy()) for a, v in zip(enc0, m.encoder.weights)))
                for a, v in zip(fresh0, m.deform_decoder.weights):          # untouched
                    self.assertTrue(np.array_equal(a, v.numpy()), v.name)
                opt = O.make_optimizer("adam", lr=1e-3)                    # built afterwards: fresh
                O.build_optimizer(opt, m.trainable_variables)
                self.assertEqual(int(opt.iterations.numpy()), 0)
                self.assertTrue(all(not np.any(v.numpy()) for v in opt.variables()))
                # The deform forward runs on the warm-started model.
                out = m(tf.zeros((2,) + P.LEVEL_SHAPE[4] + (1,)), task="deform", training=False)
                self.assertTrue(np.all(np.isfinite(out["u"].numpy())))
        # Any non-decoder field still raises, naming the field.
        for kw, field in ((dict(translation_scale=10.0), "translation_scale"),
                          (dict(filters=(16, 32, 32, 48, 64, 128)), "filters")):
            cfg = narrow_config(dropout=0.0, deform_decoder=True, **kw)
            m = GroupRegLevelModel(4, cfg).build_model(3)
            with self.assertRaises(ValueError) as cm:
                C.warm_start_model(m, cfg, src_dir)
            self.assertIn("in ['%s']" % field, str(cm.exception))

    def test_warm_start_deform_into_linear_and_full(self):
        cfg_d = narrow_config(dropout=0.0, deform_decoder=True)
        src_dir, src = self.save_source("dfm", cfg_d)
        cfg = narrow_config(dropout=0.0, seed=7)
        m = GroupRegLevelModel(4, cfg).build_model(3)
        info = C.warm_start_model(m, cfg, src_dir)
        self.assertEqual((info["mode"], info["fresh_parts"], info["ignored_source_parts"]),
                         ("partial", {}, ["deform_decoder"]))
        self.assertEqual(info["n_restored_vars"], len(m.weights))
        self.assert_parts_equal(src, m)
        # Identical configs (seed aside): the strict whole-model restore, deform decoder included.
        m2 = GroupRegLevelModel(4, cfg_d.replace(seed=3)).build_model(3)
        info = C.warm_start_model(m2, cfg_d.replace(seed=3), src_dir)
        self.assertEqual((info["mode"], info["config_diff"], info["fresh_parts"]), ("full", {}, {}))
        self.assertEqual(info["n_restored_vars"], len(m2.weights))
        self.assert_parts_equal(src, m2, C.SHARED_PARTS + ("deform_decoder",))
        self.assertEqual(C.load_model_weights(m2, cfg_d, src_dir)[0], info["checkpoint"])

    def test_peek_last_step(self):
        # Review-curriculum issue 5: main.py reads the resumed step before building the pipelines.
        self.assertEqual(C.peek_last_step(os.path.join(self.tmp, "nothing")), 0)
        model, opt, step, gen, ck = self.make()
        self.assertEqual(C.peek_last_step(os.path.join(self.tmp, "last")), 0)
        for s in (7, 12):
            step.assign(s)
            ck.save_last()
            self.assertEqual(C.peek_last_step(os.path.join(self.tmp, "last")), s)

    def test_config_json_and_compatibility(self):
        path = os.path.join(self.tmp, "config.json")
        cfg = narrow_config()
        C.write_config(path, cfg, {"lr": 0.01, "filters": [1, 2]})
        with open(path) as f:
            d = json.load(f)
        self.assertEqual(d["args"]["lr"], 0.01)
        self.assertIn("tensorflow", d["versions"])
        C.check_config_compatible(path, cfg)
        with self.assertRaises(ValueError) as cm:
            C.check_config_compatible(path, cfg.replace(dropout=0.3, group_config="B"))
        self.assertIn("dropout", str(cm.exception))
        self.assertIn("group_config", str(cm.exception))


class TestFixedData(TempDirCase):
    def test_transform_ranges(self):
        rng = np.random.default_rng(0)
        T, tgt = fd.random_linear_transforms(rng, 200, "rigid")
        r = P.DEFAULT_LINEAR_RANGES["rigid"]
        self.assertLessEqual(float(np.max(M.rotation_angle_deg(tgt[:, :3, :3]))), r["rotation_deg"] + 1e-6)
        self.assertLessEqual(float(np.max(np.abs(tgt[:, :3, 3]))), r["translation_mm"])
        rr = tgt[:, :3, :3]
        np.testing.assert_allclose(rr @ np.swapaxes(rr, 1, 2), np.tile(np.eye(3), (200, 1, 1)), atol=1e-12)
        np.testing.assert_allclose(T[:, :3, 3], tgt[:, :3, 3])
        k = np.linalg.inv(tgt) @ T                                # Rt^-1 T = nuisance K (upper triangular)
        self.assertLess(float(np.max(np.abs(np.tril(k[:, :3, :3], -1)))), 1e-12)
        self.assertGreater(float(np.max(np.abs(k[:, :3, :3] - np.eye(3)))), 1e-4)
        Tn, tn = fd.random_linear_transforms(rng, 5, "rigid", nuisance=False)
        np.testing.assert_allclose(Tn, tn)
        Ta, ta = fd.random_linear_transforms(rng, 5, "affine")
        np.testing.assert_allclose(Ta, ta)

    def test_dali_batch_round_trip(self):
        rng = np.random.default_rng(1)
        for task in ("rigid", "affine"):
            T, tgt = fd.random_linear_transforms(rng, 4, task)
            imgs = np.zeros((4,) + P.LEVEL_SHAPE[4] + (1,), np.float32)
            batch = fd.to_dali_batch(imgs, T, tgt, task, 4)
            self.assertEqual(len(batch), 3 if task == "rigid" else 2)
            self.assertEqual(tuple(batch[1].shape), (4, 3, 4))
            out = LinearPostprocess(task, 4, flip_prob=0.0)(*batch)
            np.testing.assert_allclose(out["T_target"].numpy(), tgt, atol=1e-4)
            np.testing.assert_allclose(out["T_render"].numpy(), T, atol=1e-4)
            a = out["A_star"].numpy().astype(np.float64)
            np.testing.assert_allclose(a @ tgt, np.broadcast_to((a @ tgt)[:1], a.shape), atol=1e-4)

    def test_load_level_images(self):
        root = write_fake_export(self.tmp)
        x, files = fd.load_level_images(root, 4, "train", n=2, seed=3)
        self.assertEqual(x.shape, (2,) + P.LEVEL_SHAPE[4] + (1,))
        self.assertEqual(x.dtype, np.float32)
        y, files2 = fd.load_level_images(root, 4, "train", n=2, seed=3)
        self.assertEqual(files, files2)
        np.testing.assert_array_equal(x, y)
        v, vf = fd.load_level_images(root, 4, "val")
        self.assertEqual(len(vf), 2)
        with self.assertRaises(ValueError):
            fd.load_level_images(root, 4, "train", n=5)

    def test_validation_set_deterministic(self):
        rng = np.random.default_rng(2)
        imgs = np.stack([fd.smooth_phantom(P.LEVEL_SHAPE[4], rng)[0][..., None] for _ in range(5)])
        a = fd.ValidationSet.build(imgs, "rigid", 4, group_size=3, seed=7)
        b = fd.ValidationSet.build(imgs, "rigid", 4, group_size=3, seed=7)
        self.assertEqual(len(a.groups), 2)                 # ceil(5 / 3)
        self.assertEqual(a.group_size, 3)
        covered = set()
        for ga, gb in zip(a.groups, b.groups):
            np.testing.assert_array_equal(ga["index"], gb["index"])
            np.testing.assert_array_equal(ga["images"], gb["images"])
            np.testing.assert_array_equal(ga["T"], gb["T"])
            self.assertEqual(len(set(ga["index"].tolist())), 3)
            covered |= set(ga["index"].tolist())
        self.assertEqual(covered, set(range(5)))
        c = fd.ValidationSet.build(imgs, "rigid", 4, group_size=3, seed=8)
        self.assertFalse(np.array_equal(c.groups[0]["T"], a.groups[0]["T"]))
        with self.assertRaises(ValueError):
            fd.ValidationSet.build(imgs, "rigid", 4, group_size=6)

    def test_validation_set_independent_of_task_order(self):
        # Review issue 4: the rigid set must not depend on the task's position in the ratio.
        rng = np.random.default_rng(3)
        imgs = np.stack([fd.smooth_phantom(P.LEVEL_SHAPE[4], rng)[0][..., None] for _ in range(5)])
        ranges = A.ranges_from_args(A.parse_args(["--data", "x"]))
        alone = fd.build_validation_sets(imgs, list(A.parse_task_ratio("rigid:1")), 4, 3, ranges=ranges, seed=7)
        mixed = fd.build_validation_sets(imgs, list(A.parse_task_ratio("affine:1,rigid:2")), 4, 3,
                                         ranges=ranges, seed=7)
        self.assertEqual(sorted(mixed), ["affine", "rigid"])
        for ga, gb in zip(alone["rigid"].groups, mixed["rigid"].groups):
            np.testing.assert_array_equal(ga["index"], gb["index"])
            np.testing.assert_array_equal(ga["T"], gb["T"])
            np.testing.assert_array_equal(ga["images"], gb["images"])
        self.assertFalse(np.array_equal(mixed["affine"].groups[0]["T"], mixed["rigid"].groups[0]["T"]))
        # group g depends on (seed, task, g) only, not on how many groups are built
        one = fd.ValidationSet.build(imgs, "rigid", 4, group_size=3, n_groups=1, ranges=ranges["rigid"], seed=7)
        np.testing.assert_array_equal(one.groups[0]["T"], alone["rigid"].groups[0]["T"])


class TestDLLoggerBackend(TempDirCase):
    def test_dllogger_backend(self):
        try:
            import dllogger  # noqa: F401
        except ImportError:
            self.skipTest("dllogger is not importable here (the container may have it)")
        paths = make_run_dir(self.tmp, "dll")
        log = RunLogger(paths, backend="dllogger", tensorboard=False, quiet=True)
        self.assertEqual(log.backend_name, "dllogger")
        log.log_params({"lr": 0.01, "arr": np.arange(2)})
        log.log_metrics(1, {"loss": np.float32(1.5), "rms": tf.constant(2.0)})
        log.log_event(1, {"event": "x"})
        log.close()
        self.assertTrue(os.path.isfile(log.metrics_path))
        with open(log.metrics_path) as f:
            text = f.read()
        self.assertIn("loss", text)
        self.assertIn("PARAMETER", text)


if __name__ == "__main__":
    unittest.main()
