"""Trainer loop (runtime.run): task schedule, None-gradient filtering, NaN guard, trace counts,
mixed rigid + affine interleaving with one optimizer (milestone 5: seeded alternation, one trace
per task with flips / curriculum / G on or off, Adam slots for the affine head before the first
affine step, affine validation errors, the combined best metric and affine PNGs),
rebuild_steps after a BN freeze, resume, validation (eval mode and the non-updating training-mode
diagnostics), a mixed-precision smoke test and a forced float16 overflow (loss-scale skip).
Milestone 6: the deform step (None-gradient routing between the linear and deform decoders, NaN
guard, collapse diagnostics), three tasks with one optimizer (one trace per task), deform
validation (identity baseline, PNG) and the deform best checkpoints.

Narrow model, level 4, N = 4, synthetic phantom groups (tests.runtime_helpers) and the synthetic
deform group (`fd.synthetic_deform_group`).
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests.runtime_helpers import free_memory, narrow_config, synthetic_group
import numpy as np
import tensorflow as tf

from data_loading import params as P
from models import transforms as tfm
from models.group_net import GroupRegLevelModel
from runtime import checkpoint as C
from runtime import fixed_data as fd
from runtime import metrics as M
from runtime import optimizers as O
from runtime.logging import RunLogger, make_run_dir
from runtime.run import (DEFORM_BEST_METRIC, DEFORM_DIAG_KEYS, DEFORM_LOSS_KEYS, DIAG_KEYS, ENC_GRAD_KEYS,
                         TaskSchedule, Trainer, TrainConfig, collapse_diagnostics)
from runtime.schedulers import ScheduleConfig

N = 4


def config(**kw):
    d = dict(level=4, group_size=N, steps=100, val_every=0, log_every=1, task_ratio={"rigid": 1.0},
             schedule=ScheduleConfig("constant", base_lr=1e-3, total_steps=100, warmup_steps=0), seed=0)
    d.update(kw)
    return TrainConfig(**d)


class TrainerCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups = {t: synthetic_group(4, N, t, seed=i) for i, t in enumerate(("rigid", "affine"))}

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_m4_trainer_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        free_memory()

    def make(self, cfg=None, sources=None, val_sets=None, logger=None, opt="adam", amp=False, dropout=0.0,
             **model_kw):
        cfg = cfg or config()
        model = GroupRegLevelModel(4, narrow_config(dropout=dropout, seed=1, **model_kw)).build_model(N)
        optimizer = O.make_optimizer(opt, lr=1e-3, amp=amp)
        if sources is None:
            sources = {t: fd.FixedGroupSource(g["batch"]) for t, g in self.groups.items()}
        return Trainer(model, optimizer, sources, self.groups["rigid"]["S"], cfg, logger=logger,
                       val_sets=val_sets)


class TestTaskSchedule(unittest.TestCase):
    def test_ratio_and_determinism(self):
        s = TaskSchedule({"rigid": 3, "affine": 1}, seed=5)
        draws = [s.draw(i) for i in range(20000)]
        self.assertAlmostEqual(draws.count("rigid") / 20000.0, 0.75, delta=0.02)
        self.assertEqual(draws, [TaskSchedule("rigid:3,affine:1", seed=5).draw(i) for i in range(20000)])
        rev = {i: s.draw(i) for i in reversed(range(500))}
        self.assertEqual([rev[i] for i in range(500)], draws[:500])      # independent of call order
        self.assertNotEqual(draws[:200], [TaskSchedule({"rigid": 3, "affine": 1}, seed=6).draw(i) for i in range(200)])
        self.assertEqual({TaskSchedule({"rigid": 1}).draw(i) for i in range(50)}, {"rigid"})
        # Milestone 6: deform is a schedulable task (was NotImplementedError before).
        s3 = TaskSchedule({"rigid": 1, "deform": 2}, seed=1)
        self.assertEqual(s3.tasks, ["rigid", "deform"])
        d3 = [s3.draw(i) for i in range(6000)]
        self.assertAlmostEqual(d3.count("deform") / 6000.0, 2.0 / 3.0, delta=0.03)


class TestTrainStep(TrainerCase):
    def test_none_gradients_filtered(self):
        tr = self.make()
        model = tr.model
        dec = model.linear_decoder
        ss_before = [v.numpy().copy() for v in dec.scale_shear_head.trainable_variables]
        rigid_before = dec.rigid_head.kernel.numpy().copy()
        enc_before = model.encoder.l3.conv.kernel.numpy().copy()
        # Two steps: at init the head kernels are zero, so step 1 only moves the heads.
        for _ in range(2):
            res = tr.train_step("rigid", self.groups["rigid"]["batch"])
            self.assertTrue(res["finite"])
            self.assertEqual(res["n_vars_updated"], len(model.trainable_variables) - 2)
        for b, v in zip(ss_before, dec.scale_shear_head.trainable_variables):
            self.assertTrue(np.array_equal(b, v.numpy()), v.name)
        self.assertFalse(np.array_equal(rigid_before, dec.rigid_head.kernel.numpy()))
        self.assertFalse(np.array_equal(enc_before, model.encoder.l3.conv.kernel.numpy()))
        res = tr.train_step("affine", self.groups["affine"]["batch"])       # slots exist for the head
        self.assertEqual(res["n_vars_updated"], len(model.trainable_variables))
        self.assertFalse(np.array_equal(ss_before[0], dec.scale_shear_head.kernel.numpy()))

    def test_nan_guard_skips_update(self):
        imgs = self.groups["rigid"]["images"].copy()
        imgs[0, 3, 4, 5, 0] = np.nan
        g = self.groups["rigid"]
        nan_batch = fd.to_dali_batch(imgs, g["T"], g["T_target"], "rigid", 4)
        src = {"rigid": fd.FixedGroupSource(g["batch"])}
        logger = RunLogger(make_run_dir(self.tmp, "nan"), backend="jsonl", tensorboard=False, quiet=True)
        tr = self.make(config(nonfinite_patience=2), sources=src, logger=logger)
        tr.fit(2)                                    # finite steps so the optimizer has state
        tr.sources["rigid"] = fd.FixedGroupSource(nan_batch)
        model, opt = tr.model, tr.optimizer
        before_t = [v.numpy().copy() for v in model.trainable_variables]
        before_s = [v.numpy().copy() for v in model.non_trainable_variables]
        before_opt = [v.numpy().copy() for v in opt.variables()]
        it = int(opt.iterations.numpy())
        total0 = tr.nonfinite_total
        tr.fit(1)
        self.assertFalse(tr.history[-1]["finite"])
        self.assertEqual(tr.nonfinite_total, total0 + 1)
        self.assertEqual(int(opt.iterations.numpy()), it)
        for b, v in zip(before_t, model.trainable_variables):
            self.assertTrue(np.array_equal(b, v.numpy()), v.name)
        for b, v in zip(before_s, model.non_trainable_variables):
            self.assertTrue(np.array_equal(b, v.numpy()), v.name)
        for b, v in zip(before_opt, opt.variables()):
            self.assertTrue(np.array_equal(b, v.numpy()), v.name)
        self.assertEqual(int(tr.step.numpy()), 3)                 # the step counter still advances
        tr.sources["rigid"] = fd.FixedGroupSource(g["batch"])
        tr.fit(1)
        self.assertTrue(tr.history[-1]["finite"])
        self.assertEqual(int(opt.iterations.numpy()), it + 1)
        tr.sources["rigid"] = fd.FixedGroupSource(nan_batch)
        with self.assertRaises(RuntimeError):
            tr.fit(3)                                 # patience 2: the 3rd consecutive skip raises
        logger.close()
        with open(logger.metrics_path) as f:
            events = [json.loads(line)["data"] for line in f]
        self.assertTrue(any(e.get("event") == "nonfinite_step_skipped" for e in events))

    def test_trace_once_per_task(self):
        tr = self.make(config(task_ratio={"rigid": 1.0, "affine": 1.0}))
        tr.fit(8)
        self.assertEqual({h["task"] for h in tr.history}, {"rigid", "affine"})
        self.assertEqual(tr.trace_counts, {"rigid": 1, "affine": 1})
        self.assertEqual(int(tr.step.numpy()), 8)
        self.assertTrue(all(h["finite"] for h in tr.history))

    def test_rebuild_steps_after_bn_freeze(self):
        tr = self.make()
        batch = self.groups["rigid"]["batch"]
        tr.train_step("rigid", batch)
        stats = lambda: [v.numpy().copy() for v in tr.model.non_trainable_variables]  # noqa: E731
        affine = [v for v in tr.model.trainable_variables if v.name.split("/")[-1].split(":")[0] in ("gamma", "beta")]
        self.assertGreater(tr.model.set_bn_stats_frozen(True), 0)
        s0 = stats()
        tr.train_step("rigid", batch)                 # trap: the traced step ignores the freeze
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(s0, stats())))
        tr.rebuild_steps()
        s1 = stats()
        g1 = [v.numpy().copy() for v in affine]
        tr.train_step("rigid", batch)
        tr.train_step("rigid", batch)
        for a, b in zip(s1, stats()):
            self.assertTrue(np.array_equal(a, b))
        self.assertTrue(any(not np.array_equal(a, v.numpy()) for a, v in zip(g1, affine)))
        self.assertEqual(tr.trace_counts["rigid"], 2)


def _seed_starting_rigid(k=3):
    """A schedule seed whose rigid:1,affine:1 draws start with k rigid steps."""
    for seed in range(1000):
        s = TaskSchedule({"rigid": 1.0, "affine": 1.0}, seed)
        if [s.draw(i) for i in range(k)] == ["rigid"] * k:
            return seed
    raise AssertionError("no seed found")


class TestMixedTasks(TrainerCase):
    """Milestone 5: rigid + affine interleaved with ONE optimizer (design 5.2)."""

    def control_sources(self):
        """Per-task controls (curriculum + G, writers under tmp) and batches with extras."""
        from runtime import curriculum as CU
        controls, sources = {}, {}
        for i, t in enumerate(("rigid", "affine")):
            end = dict(P.DEFAULT_LINEAR_RANGES[t])
            start = dict(end, rotation_deg=10.0, translation_mm=2.0)
            controls[t] = CU.TaskControl(end, CU.LinearCurriculum(start, end, 10),
                                         CU.GlobalTransformSampler(45.0, 10.0, 3, seed=i),
                                         writer=CU.ControlWriter(os.path.join(self.tmp, "control", t + ".npy")))
            g = np.tile(np.eye(4)[:3][None], (N, 1, 1)).astype(np.float32)
            extras = (tf.constant(g), tf.constant(np.ones([N], np.float32)))
            sources[t] = fd.FixedGroupSource(tuple(self.groups[t]["batch"]) + extras)
        return controls, sources

    def test_interleaving_starting_rigid(self):
        seed = _seed_starting_rigid(3)
        n_steps = 30
        for variant in ("flips_off", "flips_controls_on"):
            with self.subTest(variant=variant):
                cfg = config(task_ratio={"rigid": 1.0, "affine": 1.0}, seed=seed,
                             flip_prob=0.0 if variant == "flips_off" else 0.5)
                controls, sources = ({}, None) if variant == "flips_off" else self.control_sources()
                model = GroupRegLevelModel(4, narrow_config(dropout=0.0, seed=1)).build_model(N)
                opt = O.make_optimizer("adam", lr=1e-3)
                tr = Trainer(model, opt, sources or {t: fd.FixedGroupSource(g["batch"])
                                                     for t, g in self.groups.items()},
                             self.groups["rigid"]["S"], cfg, controls=controls)
                tvars = model.trainable_variables
                # Adam slots exist for every trainable variable (incl. the affine-only head)
                # before the first step: m and v per variable (Keras 2.10 experimental Adam;
                # variables() does not list the iteration counter).
                n_opt = len(opt.variables())
                self.assertEqual(n_opt, 2 * len(tvars))
                head = model.linear_decoder.scale_shear_head
                self.assertEqual(sum("scale_shear_head" in v.name for v in opt.variables()),
                                 2 * len(head.trainable_variables))
                ss0 = [v.numpy().copy() for v in head.trainable_variables]
                tr.fit(3)                                           # rigid only
                self.assertEqual([h["task"] for h in tr.history], ["rigid"] * 3)
                for a, v in zip(ss0, head.trainable_variables):
                    self.assertTrue(np.array_equal(a, v.numpy()), v.name)
                self.assertEqual(tr.trace_counts, {"rigid": 1, "affine": 0})
                tr.fit(n_steps - 3)
                tasks = [h["task"] for h in tr.history]
                self.assertEqual(tasks, [tr.schedule.draw(s) for s in range(n_steps)])
                self.assertGreater(tasks.count("affine"), 5)
                self.assertGreater(tasks.count("rigid"), 5)
                self.assertEqual(tr.trace_counts, {"rigid": 1, "affine": 1})
                self.assertEqual(len(opt.variables()), n_opt)       # no slot created later
                self.assertEqual(int(opt.iterations.numpy()), n_steps)
                self.assertTrue(all(h["finite"] for h in tr.history))
                for h in tr.history:
                    want = len(tvars) - 2 if h["task"] == "rigid" else len(tvars)
                    self.assertEqual(h["n_vars_updated"], want)
                self.assertFalse(np.array_equal(ss0[0], head.kernel.numpy()))
                if controls:
                    for t, c in controls.items():
                        self.assertGreater(c.writer.version, 1)
                        rec = next(h for h in tr.history if h["task"] == t)
                        self.assertIn("control_rot_deg", rec)
                        self.assertEqual(rec["control_g_rot_max_deg"], 45.0)   # no G warm-up
                free_memory()

    def test_mixed_validation_best_metric_and_pngs(self):
        val_sets = {t: fd.ValidationSet.from_group(g["images"], g["T"], g["T_target"], t, 4)
                    for t, g in self.groups.items()}
        png = os.path.join(self.tmp, "png")
        paths = make_run_dir(self.tmp, "mixed")
        cfg = config(task_ratio={"rigid": 1.0, "affine": 1.0}, val_every=2, val_png_dir=png, val_png_k=2)
        tr = self.make(cfg, val_sets=val_sets)
        from runtime.run import MEAN_PAIR_METRIC
        self.assertEqual(tr.best_metric, MEAN_PAIR_METRIC)
        tr.checkpoint = C.RunCheckpoint(paths["ckpt_last"], paths["ckpt_best"], tr.model, tr.optimizer,
                                        tr.step, extra=tr.checkpoint_variables())
        tr.fit(4)
        self.assertEqual([v["step"] for v in tr.val_history], [2, 4])
        for v in tr.val_history:
            self.assertAlmostEqual(v[MEAN_PAIR_METRIC],
                                   0.5 * (v["val_rigid/pair_rms_mm"] + v["val_affine/pair_rms_mm"]), places=12)
            for k in ("center_scale_pct", "center_shear_pct", "center_qr_rot_deg", "pair_scale_pct",
                      "pair_shear_pct", "pair_qr_rot_deg"):
                self.assertTrue(np.isfinite(v["val_affine/" + k]), k)
                self.assertNotIn("val_rigid/" + k, v)
                self.assertNotIn("val_affine_trainmode/" + k, v)
        best = min(v[MEAN_PAIR_METRIC] for v in tr.val_history)
        self.assertAlmostEqual(float(tr.checkpoint.best_metric.numpy()), best, places=10)
        self.assertIsNotNone(tr.checkpoint.latest_best)
        self.assertEqual(sorted(os.listdir(png)), ["affine_step0000002.png", "affine_step0000004.png",
                                                   "rigid_step0000002.png", "rigid_step0000004.png"])
        # Identity model at init: the affine val errors equal those of the identity matrices.
        tr0 = self.make(config(), val_sets={"affine": val_sets["affine"]})
        m = tr0.validate()
        a_star = val_sets["affine"].groups[0]["A_star"]
        ref = M.affine_matrix_errors(np.tile(np.eye(4), (N, 1, 1)), a_star)
        self.assertAlmostEqual(m["val_affine/center_scale_pct"], float(np.mean(ref["scale_pct"])), places=4)
        self.assertGreater(m["val_affine/center_scale_pct"], 0.5)   # the fixed group has real scale


class TestResumeAndValidation(TrainerCase):
    def with_checkpoint(self, tr, paths):
        tr.checkpoint = C.RunCheckpoint(paths["ckpt_last"], paths["ckpt_best"], tr.model, tr.optimizer,
                                        tr.step, extra=tr.checkpoint_variables())
        return tr

    def test_resume_continues_step_count(self):
        paths = make_run_dir(self.tmp, "resume")
        log1 = RunLogger(paths, backend="jsonl", tensorboard=False, quiet=True)
        tr = self.with_checkpoint(self.make(logger=log1), paths)
        tr.fit(3)
        log1.close()
        w_ref = [v.numpy() for v in tr.model.variables]
        free_memory()
        log2 = RunLogger(paths, backend="jsonl", tensorboard=False, quiet=True)
        tr2 = self.with_checkpoint(self.make(logger=log2), paths)
        self.assertTrue(tr2.checkpoint.restore_last())
        self.assertEqual(int(tr2.step.numpy()), 3)
        for a, b in zip(w_ref, tr2.model.variables):
            self.assertTrue(np.array_equal(a, b.numpy()))
        summary = tr2.fit(2)
        log2.close()
        self.assertEqual((summary["start_step"], summary["end_step"]), (3, 5))
        with open(log2.metrics_path) as f:
            steps = [json.loads(line)["step"] for line in f]
        self.assertEqual(steps, [1, 2, 3, 4, 5])
        self.assertTrue(tr2.checkpoint.latest_last.endswith("last-5"))

    def test_validation_identity_model(self):
        g = self.groups["rigid"]
        vs = fd.ValidationSet.from_group(g["images"], g["T"], g["T_target"], "rigid", 4)
        tr = self.make(val_sets={"rigid": vs})
        m = tr.validate()
        for k in ("center_rms_mm", "center_rot_deg", "center_trans_mm", "pair_rms_mm", "pair_rot_deg",
                  "pair_trans_mm", "cycle_rms_mm"):
            self.assertTrue(np.isfinite(m["val_rigid/" + k]), k)
        a_star = vs.groups[0]["A_star"]
        d = np.eye(4) - a_star
        ref = np.mean(np.sqrt(np.einsum("nij,jk,nik->n", d, g["S"], d)))
        self.assertAlmostEqual(m["val_rigid/center_rms_mm"], ref, places=4)
        self.assertLess(m["val_rigid/cycle_rms_mm"], 1e-3)
        # training-mode diagnostics: identity heads at init -> same RMS; no BN statistics update
        self.assertAlmostEqual(m["val_rigid_trainmode/center_rms_mm"], ref, places=4)
        self.assertTrue(np.isfinite(m["val_rigid_trainmode/pair_rms_mm"]))

    def test_trainmode_validation_keeps_bn_statistics(self):
        g = self.groups["rigid"]
        vs = fd.ValidationSet.from_group(g["images"], g["T"], g["T_target"], "rigid", 4)
        tr = self.make(val_sets={"rigid": vs}, dropout=0.15)
        tr.fit(5)
        stats = [v.numpy().copy() for v in tr.model.non_trainable_variables]
        self.assertTrue(stats)
        m1 = tr.validate()
        m2 = tr.validate()
        for a, v in zip(stats, tr.model.non_trainable_variables):
            self.assertTrue(np.array_equal(a, v.numpy()), v.name)
        # deterministic (dropout off in the train-mode forward too)
        self.assertAlmostEqual(m1["val_rigid_trainmode/center_rms_mm"],
                               m2["val_rigid_trainmode/center_rms_mm"], places=4)
        self.assertEqual(tr.model.linear_decoder.dropout.rate, 0.15)
        self.assertEqual(tr.val_trace_counts, {("rigid", N, False): 1, ("rigid", N, True): 1})

    def test_mixed_precision_smoke(self):
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            tr = self.make(amp=True)
            self.assertIsInstance(tr.optimizer, tf.keras.mixed_precision.LossScaleOptimizer)
            out = tr._steps["rigid"](*self.groups["rigid"]["batch"])
            self.assertEqual(out["loss"].dtype, tf.float32)
            # Smoke test only: the loss is float32 and stays finite under mixed_float16. Whether
            # a step overflows at the default initial scale (2^15) is not asserted here (the
            # overflow path is test_mixed_precision_overflow_skips). Keras' LSO advances the
            # inner optimizer's `iterations` on applied AND skipped steps, so it counts calls.
            tr.fit(20)
            self.assertTrue(all(np.isfinite(h["loss"]) for h in tr.history))
            self.assertGreater(sum(h["finite"] for h in tr.history), 0)
            self.assertEqual(int(O.inner_optimizer(tr.optimizer).iterations.numpy()), 21)
            self.assertLessEqual(float(tr.optimizer.loss_scale.numpy()), 2.0 ** 15)
        finally:
            tf.keras.mixed_precision.set_global_policy("float32")

    def test_mixed_precision_overflow_skips(self):
        # Review issue 2: force float16 overflow with a huge initial loss scale (2^40; the scaled
        # mm^2 loss is ~1e15, so float16 gradients become inf). The LSO must skip the update and
        # halve its scale, and the step must roll back the BN statistics.
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            model = GroupRegLevelModel(4, narrow_config(dropout=0.0, seed=1)).build_model(N)
            # The heads are float32 with a zero kernel at init, so no gradient reaches the float16
            # layers on step 1. A small random kernel lets the scaled gradient flow into them.
            head = model.linear_decoder.rigid_head
            head.kernel.assign(np.random.default_rng(0).normal(0, 1e-3, head.kernel.shape).astype(np.float32))
            inner = O.make_optimizer("adam", lr=1e-3)
            opt = tf.keras.mixed_precision.LossScaleOptimizer(inner, initial_scale=2.0 ** 40)
            tr = Trainer(model, opt, {"rigid": fd.FixedGroupSource(self.groups["rigid"]["batch"])},
                         self.groups["rigid"]["S"], config(amp=True))
            batch = self.groups["rigid"]["batch"]
            n_skipped = 0
            for _ in range(40):
                scale0 = float(opt.loss_scale.numpy())
                it0 = int(inner.iterations.numpy())
                w0 = [v.numpy().copy() for v in model.trainable_variables]
                s0 = [v.numpy().copy() for v in model.non_trainable_variables]
                slots0 = [v.numpy().copy() for v in inner.variables() if "iteration" not in v.name]
                res = tr.train_step("rigid", batch)
                self.assertTrue(np.isfinite(res["loss"]))
                self.assertEqual(int(inner.iterations.numpy()), it0 + 1)   # skipped steps count too
                if res["finite"]:
                    break
                n_skipped += 1
                self.assertEqual(float(opt.loss_scale.numpy()), scale0 / 2.0)
                for a, v in zip(w0, model.trainable_variables):
                    self.assertTrue(np.array_equal(a, v.numpy()), v.name)
                for a, v in zip(s0, model.non_trainable_variables):
                    self.assertTrue(np.array_equal(a, v.numpy()), v.name)
                for a, v in zip(slots0, [v for v in inner.variables() if "iteration" not in v.name]):
                    self.assertTrue(np.array_equal(a, v.numpy()), v.name)
            sys.stderr.write("\n[amp overflow] %d skipped steps, first applied step at loss scale 2^%d\n" % (
                n_skipped, int(round(np.log2(float(opt.loss_scale.numpy()))))))
            self.assertGreater(n_skipped, 0)
            self.assertTrue(res["finite"], "no finite step after %d halvings" % n_skipped)
            self.assertTrue(any(not np.array_equal(a, v.numpy()) for a, v in zip(w0, model.trainable_variables)))
            self.assertTrue(any(not np.array_equal(a, v.numpy()) for a, v in zip(s0, model.non_trainable_variables)))
        finally:
            tf.keras.mixed_precision.set_global_policy("float32")


def _random_head_kernel(model, seed=0):
    head = model.linear_decoder.rigid_head
    head.kernel.assign(np.random.default_rng(seed).normal(0, 1e-3, head.kernel.shape).astype(np.float32))


class TestLossScale(TrainerCase):
    def test_gradient_scales_and_logged_terms_do_not(self):
        scale = 100.0
        tr1 = self.make(config(loss_scale_mm2=1.0))
        _random_head_kernel(tr1.model)          # gradient into the encoder too, not only the heads
        tr2 = self.make(config(loss_scale_mm2=scale))
        tr2.model.set_weights(tr1.model.get_weights())
        batch = self.groups["rigid"]["batch"]
        r1 = tr1.train_step("rigid", batch)
        r2 = tr2.train_step("rigid", batch)
        self.assertTrue(r1["finite"] and r2["finite"])
        self.assertGreater(r1["grad_norm"], 0.0)
        np.testing.assert_allclose(r2["grad_norm"], r1["grad_norm"] / scale, rtol=1e-3)
        for k in ("loss", "center", "pair", "center_rms_mm", "pair_rms_mm"):   # unscaled mm^2 / mm
            np.testing.assert_allclose(r2[k], r1[k], rtol=1e-4, err_msg=k)
        np.testing.assert_allclose(r2["loss"], r2["center"] + r2["pair"], rtol=1e-5)
        self.assertGreater(r2["center"], 1.0)    # mm^2 scale, not divided by 100
        with self.assertRaises(ValueError):
            self.make(config(loss_scale_mm2=0.0))

    def make_clip(self, cfg, clipnorm, weights_from=None):
        model = GroupRegLevelModel(4, narrow_config(dropout=0.0, seed=1)).build_model(N)
        if weights_from is None:
            _random_head_kernel(model)
        else:
            model.set_weights(weights_from.get_weights())
        sources = {t: fd.FixedGroupSource(g["batch"]) for t, g in self.groups.items()}
        return Trainer(model, O.make_optimizer("adam", lr=1e-3, clipnorm=clipnorm), sources,
                       self.groups["rigid"]["S"], cfg)

    def test_task_loss_weight_and_enc_grad_norms(self):
        # Milestone 7: the task weight multiplies the applied gradient; logged terms and the raw
        # encoder norm (unscaled loss) do not change; _eff follows the applied encoder gradient.
        batch = self.groups["rigid"]["batch"]
        tr1 = self.make_clip(config(), clipnorm=None)
        tr4 = self.make_clip(config(task_loss_weights={"rigid": 4.0}), None, weights_from=tr1.model)
        trs = self.make_clip(config(task_loss_weights={"rigid": 4.0}, loss_scale_mm2=100.0), None,
                             weights_from=tr1.model)
        trc = self.make_clip(config(), clipnorm=1e-3, weights_from=tr1.model)    # global clip 1e-3
        r1, r4, rs, rc = (t.train_step("rigid", batch) for t in (tr1, tr4, trs, trc))
        self.assertGreater(r1["diag_enc_grad_norm_raw"], 0.0)
        np.testing.assert_allclose(r4["grad_norm"], 4.0 * r1["grad_norm"], rtol=1e-4)
        np.testing.assert_allclose(rs["grad_norm"], 0.04 * r1["grad_norm"], rtol=1e-4)
        for r in (r4, rs):
            for k in ("loss", "center", "pair"):
                np.testing.assert_allclose(r[k], r1[k], rtol=1e-5, err_msg=k)
            np.testing.assert_allclose(r["diag_enc_grad_norm_raw"], r1["diag_enc_grad_norm_raw"], rtol=1e-4)
        # Unclipped: _eff is the applied encoder norm = raw * w / loss_scale.
        np.testing.assert_allclose(r1["diag_enc_grad_norm_eff"], r1["diag_enc_grad_norm_raw"], rtol=1e-6)
        np.testing.assert_allclose(r4["diag_enc_grad_norm_eff"], 4.0 * r1["diag_enc_grad_norm_eff"], rtol=1e-4)
        np.testing.assert_allclose(rs["diag_enc_grad_norm_eff"], 0.04 * r1["diag_enc_grad_norm_eff"], rtol=1e-4)
        self.assertLess(r1["diag_enc_grad_norm_raw"], r1["grad_norm"] * (1 + 1e-6))
        # Clipped (global clip 1e-3): _eff = encoder norm * clip / total norm.
        self.assertGreater(rc["grad_norm"], 1e-3)
        np.testing.assert_allclose(rc["diag_enc_grad_norm_eff"],
                                   rc["diag_enc_grad_norm_raw"] * 1e-3 / rc["grad_norm"], rtol=1e-4)
        np.testing.assert_allclose(rc["diag_enc_grad_norm_raw"], r1["diag_enc_grad_norm_raw"], rtol=1e-4)
        for bad in ({"rigid": 0.0}, {"deform": float("nan")}, {"warp": 1.0}):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.make(config(task_loss_weights=bad))


GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "m7_linear_golden.json")
GOLDEN_STEPS = 6


def _linear_golden_run():
    """6 mixed rigid/affine steps of a seeded narrow model (dropout 0, flips on): per-step losses
    and per-variable (float64 sum, max |v|) keyed by the scope-local name."""
    ratio = {"rigid": 1.0, "affine": 1.0}
    seed = _seed_with_all_tasks(ratio, GOLDEN_STEPS)
    groups = {t: synthetic_group(4, N, t, seed=i) for i, t in enumerate(("rigid", "affine"))}
    model = GroupRegLevelModel(4, narrow_config(dropout=0.0, seed=0)).build_model(N)
    tr = Trainer(model, O.make_optimizer("adam", lr=1e-3),
                 {t: fd.FixedGroupSource(g["batch"]) for t, g in groups.items()},
                 groups["rigid"]["S"], config(task_ratio=ratio, seed=seed))
    tr.fit(GOLDEN_STEPS)
    out = {"seed": seed, "tasks": [h["task"] for h in tr.history],
           "loss": [float(h["loss"]) for h in tr.history],
           "variables": {GroupRegLevelModel._local_name(v): [float(np.sum(v.numpy().astype(np.float64))),
                                                             float(np.max(np.abs(v.numpy())))]
                         for v in model.variables}}
    return out


class TestLinearOnlyUnchanged(unittest.TestCase):
    """Milestone 7 guard: linear-only training is unchanged by the three-task additions (task loss
    weights, encoder grad-norm diagnostics). The fixture was written on the milestone-6 code
    (commit 0fb998b) with BGR_WRITE_GOLDEN=1."""

    def tearDown(self):
        free_memory()

    def test_matches_golden(self):
        got = _linear_golden_run()
        if os.environ.get("BGR_WRITE_GOLDEN") == "1":
            os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
            with open(GOLDEN_PATH, "w") as f:
                json.dump(dict(got, tolerance=0.0), f, indent=1, sort_keys=True)
            sys.stderr.write("\n[golden] wrote %s\n" % GOLDEN_PATH)
        with open(GOLDEN_PATH) as f:
            want = json.load(f)
        tol = float(want.get("tolerance", 0.0))
        self.assertEqual(got["seed"], want["seed"])
        self.assertEqual(got["tasks"], want["tasks"])
        self.assertEqual(set(got["tasks"]), {"rigid", "affine"})
        self.assertEqual(sorted(got["variables"]), sorted(want["variables"]))
        diffs = [abs(a - b) for a, b in zip(got["loss"], want["loss"])]
        for name, (s, m) in want["variables"].items():
            gs, gm = got["variables"][name]
            diffs += [abs(gs - s), abs(gm - m)]
        sys.stderr.write("\n[golden] max |diff| %.3g (tolerance %.3g)\n" % (max(diffs), tol))
        if tol == 0.0:
            self.assertEqual(got["loss"], want["loss"])
            self.assertEqual(got["variables"], want["variables"])
        else:
            self.assertLessEqual(max(diffs), tol)


class TestCollapseDiagnostics(TrainerCase):
    def identical_group_source(self):
        g = self.groups["rigid"]
        rep = lambda a: np.repeat(a[:1], N, axis=0)  # noqa: E731
        return fd.FixedGroupSource(fd.to_dali_batch(rep(g["images"]), rep(g["T"]), rep(g["T_target"]),
                                                    "rigid", 4))

    def test_crafted_values(self):
        pre = np.array([[1.0, -1.0, 0.0, 2.0], [-1.0, -2.0, 0.0, 4.0]], np.float32)
        params = np.zeros((2, 19), np.float32)
        params[0, :2] = (3.0, 4.0)
        params[1, :2] = (6.0, 8.0)
        params[0, 10:13] = (1.0, 2.0, 3.0)
        params[1, 10:13] = (1.0, 2.0, 5.0)
        params[:, 13:] = np.arange(12, dtype=np.float32).reshape(2, 6) * 100   # ignored (affine)
        d = {k: float(v) for k, v in collapse_diagnostics(tf.constant(pre), tf.constant(params)).items()}
        self.assertEqual(set(d), set(DIAG_KEYS))
        self.assertAlmostEqual(d["diag_dead_frac"], 0.5, places=6)          # channels 1 and 2
        self.assertAlmostEqual(d["diag_feat_spread"], 0.625, places=6)      # stds 1, .5, 0, 1
        self.assertAlmostEqual(d["diag_feat_rel_spread"], 0.625 / 1.375, places=6)
        self.assertAlmostEqual(d["diag_rot10_norm"], 7.5, places=5)         # norms 5 and 10
        self.assertAlmostEqual(d["diag_head_spread"], 4.5 / 13, places=6)   # stds 1.5, 2, 1
        z = collapse_diagnostics(tf.zeros((3, 4)), tf.zeros((3, 13)))
        self.assertEqual(float(z["diag_feat_rel_spread"]), 0.0)
        self.assertEqual(float(z["diag_dead_frac"]), 1.0)

    def test_identical_images_zero_spread_and_warning(self):
        paths = make_run_dir(self.tmp, "collapse")
        logger = RunLogger(paths, backend="jsonl", tensorboard=False, quiet=True)
        tr = self.make(config(collapse_patience=3), sources={"rigid": self.identical_group_source()},
                       logger=logger)
        tr.fit(5)
        logger.close()
        for h in tr.history:
            self.assertTrue(h["finite"])
            self.assertLess(h["diag_feat_spread"], 1e-5)
            self.assertLess(h["diag_feat_rel_spread"], 1e-5)
            self.assertLess(h["diag_head_spread"], 1e-5)
        self.assertEqual((tr.collapse_streak, tr.collapse_warnings), (5, 1))
        self.assertEqual(tr.trace_counts, {"rigid": 1})
        with open(logger.metrics_path) as f:
            recs = [json.loads(line) for line in f]
        events = [r for r in recs if r["data"].get("event") == "collapse_suspect"]
        self.assertEqual([r["step"] for r in events], [3])
        counters = [r["data"]["train/collapse_suspect"] for r in recs if "train/collapse_suspect" in r["data"]]
        self.assertEqual(counters, [1, 2, 3, 4, 5])

    def test_healthy_group_logs_diagnostics(self):
        paths = make_run_dir(self.tmp, "healthy")
        logger = RunLogger(paths, backend="jsonl", tensorboard=False, quiet=True)
        tr = self.make(config(log_every=3, collapse_patience=2), logger=logger)
        tr.fit(3)
        logger.close()
        self.assertEqual(tr.trace_counts["rigid"], 1)
        self.assertEqual((tr.collapse_streak, tr.collapse_warnings), (0, 0))
        with open(logger.metrics_path) as f:
            recs = [json.loads(line)["data"] for line in f]
        self.assertEqual(len(recs), 1)
        m = recs[0]
        for k in DIAG_KEYS:
            self.assertTrue(np.isfinite(m["train_rigid/" + k]), k)
        self.assertGreater(m["train_rigid/diag_feat_rel_spread"], 1e-3)
        self.assertGreaterEqual(m["train_rigid/diag_dead_frac"], 0.0)
        self.assertLessEqual(m["train_rigid/diag_dead_frac"], 1.0)
        self.assertEqual(m["train/collapse_suspect"], 0)
        # rot10 norm at init: the identity bias of the 10D rotation outputs
        np.testing.assert_allclose(tr.history[0]["diag_rot10_norm"],
                                   np.linalg.norm(tfm.identity_rigid_params()[:tfm.ROTATION_DIMS]),
                                   rtol=1e-4)


class TestHeadActivation(TrainerCase):
    def test_build_and_run(self):
        for act in ("relu", "none"):
            # negative_slope=0 so 'relu' is a true ReLU (the model default is LeakyReLU 0.01).
            tr = self.make(head_activation=act, negative_slope=0.0)
            dec = tr.model.linear_decoder
            self.assertEqual(dec.head_norm_act.act is None, act == "none")
            self.assertIsNotNone(dec.head_norm_act.norm)                   # LayerNorm kept
            res = tr.train_step("rigid", self.groups["rigid"]["batch"])
            self.assertTrue(res["finite"], act)
            out = tr.model(self.groups["rigid"]["images"], task="rigid", training=False)
            pre = out["pre_head"].numpy()
            np.testing.assert_array_equal(pre, out["dense"].numpy())
            self.assertEqual(out["params"].shape, (N, tfm.RIGID_PARAMS))
            if act == "relu":
                self.assertGreaterEqual(pre.min(), 0.0)
            else:
                self.assertLess(pre.min(), 0.0)                             # no activation clamp
            self.assertLess(res["diag_dead_frac"], 1.0)
            free_memory()
        with self.assertRaises(ValueError):
            narrow_config(head_activation="gelu")


class TestControls(TrainerCase):
    """Curriculum / global transform hooks (runtime.curriculum, design 3.2 v3.5)."""

    def extras(self, mixed=False, version=1.0):
        g = np.tile(np.eye(4)[:3][None], (N, 1, 1)).astype(np.float32)
        if mixed:
            g[1, 0, 3] = 2.0
        return (tf.constant(g), tf.constant(np.full([N], version, np.float32)))

    def test_off_is_unchanged(self):
        batch = self.groups["rigid"]["batch"]
        runs = []
        for b in (batch, tuple(batch) + self.extras()):
            tf.random.set_seed(0)
            tr = self.make(sources={"rigid": fd.FixedGroupSource(b)})
            tr.fit(3)
            runs.append([h["loss"] for h in tr.history])
            self.assertEqual(tr.controls, {})
            self.assertFalse(any(k.startswith("control") or k == "g_mixed_batch" for k in tr.history[-1]))
            free_memory()
        np.testing.assert_allclose(runs[0], runs[1], rtol=1e-4)   # trailing extras are dropped
        self.assertEqual(os.listdir(self.tmp), [])         # no control file anywhere

    def test_controls_written_each_step_and_logged(self):
        from runtime import curriculum as CU
        start = {"rotation_deg": 10.0, "translation_mm": 2.0, "log_scale": 0.0, "shear": 0.0}
        end = {"rotation_deg": 50.0, "translation_mm": 10.0, "log_scale": 0.0, "shear": 0.0}
        path = os.path.join(self.tmp, "control", "rigid.npy")
        ctl = CU.TaskControl(end, CU.LinearCurriculum(start, end, 4),
                             CU.GlobalTransformSampler(45.0, 10.0, resample_every=2, seed=1),
                             writer=CU.ControlWriter(path))
        batch = tuple(self.groups["rigid"]["batch"]) + self.extras(mixed=True, version=1.0)
        logger = RunLogger(make_run_dir(self.tmp, "ctl"), backend="jsonl", tensorboard=False, quiet=True)
        model = GroupRegLevelModel(4, narrow_config(dropout=0.0, seed=1)).build_model(N)
        tr = Trainer(model, O.make_optimizer("adam", lr=1e-3), {"rigid": fd.FixedGroupSource(batch)},
                     self.groups["rigid"]["S"], config(), logger=logger, controls={"rigid": ctl})
        tr.fit(4)
        logger.close()
        self.assertEqual(ctl.writer.version, 4)            # values changed at steps 0..3
        on_disk = ctl.writer.read()
        self.assertEqual(on_disk["version"], 4)
        self.assertAlmostEqual(on_disk["rotation_deg"], 40.0, places=4)   # step 3: 10 + 3/4 * 40
        self.assertAlmostEqual(on_disk["g_angle_deg"], ctl.sampler.params(3)["angle_deg"], places=4)
        h = tr.history[-1]
        self.assertEqual(h["g_mixed_batch"], 1.0)
        self.assertEqual(h["control_version_lag"], 3.0)
        self.assertAlmostEqual(h["control_rot_deg"], 40.0)
        self.assertAlmostEqual(h["control_trans_mm"], 8.0)
        self.assertEqual(h["control_g_angle_deg"], ctl.sampler.params(3)["angle_deg"])
        self.assertEqual([x["control_rot_deg"] for x in tr.history], [10.0, 20.0, 30.0, 40.0])
        self.assertTrue(all(x["finite"] for x in tr.history))
        with open(logger.metrics_path) as f:
            keys = set().union(*[json.loads(line)["data"].keys() for line in f])
        for k in ("g_mixed_batch", "control_version_lag", "control_rot_deg", "control_trans_mm",
                  "control_g_angle_deg"):
            self.assertIn("train_rigid/" + k, keys)
        # The values are a pure function of the step (resume reproduces them).
        fresh = CU.TaskControl(end, CU.LinearCurriculum(start, end, 4),
                               CU.GlobalTransformSampler(45.0, 10.0, resample_every=2, seed=1))
        self.assertEqual(fresh.values_at(3), ctl.values_at(3))

    def test_g_warmup_logged(self):
        from runtime import curriculum as CU
        end = {"rotation_deg": 45.0, "translation_mm": 15.0, "log_scale": 0.0, "shear": 0.0}
        ctl = CU.TaskControl(end, None, CU.GlobalTransformSampler(45.0, 10.0, resample_every=2, seed=1,
                                                                  start_step=2, ramp_steps=2),
                             writer=CU.ControlWriter(os.path.join(self.tmp, "control", "rigid.npy")))
        batch = tuple(self.groups["rigid"]["batch"]) + self.extras()
        model = GroupRegLevelModel(4, narrow_config(dropout=0.0, seed=1)).build_model(N)
        tr = Trainer(model, O.make_optimizer("adam", lr=1e-3), {"rigid": fd.FixedGroupSource(batch)},
                     self.groups["rigid"]["S"], config(), controls={"rigid": ctl})
        tr.fit(5)
        self.assertEqual([h["control_g_rot_max_deg"] for h in tr.history], [0.0, 0.0, 0.0, 22.5, 45.0])
        self.assertEqual([h["control_g_angle_deg"] for h in tr.history[:3]], [0.0, 0.0, 0.0])
        self.assertNotEqual(tr.history[4]["control_g_angle_deg"], 0.0)
        self.assertAlmostEqual(ctl.writer.read()["g_angle_deg"], ctl.values_at(4)["g_angle_deg"], places=4)

    def test_named_val_sets_and_pngs(self):
        g = self.groups["rigid"]
        vs = fd.ValidationSet.from_group(g["images"], g["T"], g["T_target"], "rigid", 4)
        png = os.path.join(self.tmp, "png")
        tr = self.make(config(val_png_dir=png, val_png_k=2), val_sets={"rigid": vs, "rigid_g": vs})
        val = tr.validate()
        for name in ("val_rigid", "val_rigid_g", "val_rigid_trainmode", "val_rigid_g_trainmode"):
            self.assertIn(name + "/center_rms_mm", val)
        self.assertAlmostEqual(val["val_rigid/center_rms_mm"], val["val_rigid_g/center_rms_mm"], places=5)
        self.assertEqual(sorted(os.listdir(png)), ["rigid_g_step0000000.png", "rigid_step0000000.png"])


# ============================================================================ milestone 6: deform
def _vals(variables):
    return [v.numpy().copy() for v in variables]


def _all_equal(before, variables):
    return all(np.array_equal(a, v.numpy()) for a, v in zip(before, variables))


def _seed_with_all_tasks(ratio, n_steps):
    """A schedule seed whose first n_steps draws contain every task of `ratio`."""
    for seed in range(1000):
        s = TaskSchedule(ratio, seed)
        if {s.draw(i) for i in range(n_steps)} == set(ratio):
            return seed
    raise AssertionError("no seed found")


class DeformCase(TrainerCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.dgroup = fd.synthetic_deform_group(4, N, seed=0)

    def make_deform(self, cfg=None, sources=None, val_sets=None, logger=None, amp=False, **model_kw):
        cfg = cfg or config(task_ratio={"deform": 1.0})
        model = GroupRegLevelModel(4, narrow_config(dropout=0.0, seed=1, deform_decoder=True,
                                                    **model_kw)).build_model(N)
        if sources is None:
            sources = {t: fd.FixedGroupSource(g["batch"]) for t, g in self.groups.items()}
            sources["deform"] = fd.FixedGroupSource(self.dgroup["batch"])
            sources = {t: s for t, s in sources.items() if t in cfg.task_ratio}
        return Trainer(model, O.make_optimizer("adam", lr=1e-3, amp=amp), sources,
                       self.groups["rigid"]["S"], cfg, logger=logger, val_sets=val_sets)


class TestDeformStep(DeformCase):
    def test_none_gradients_filtered(self):
        tr = self.make_deform(config(task_ratio={"rigid": 1.0, "deform": 1.0}))
        model = tr.model
        lin, dfm = model.linear_decoder, model.deform_decoder
        n_tv = len(model.trainable_variables)
        lin_before, dfm_before = _vals(lin.variables), _vals(dfm.variables)
        enc_before = model.encoder.l3.conv.kernel.numpy().copy()
        head_before = dfm.svf_head.kernel.numpy().copy()
        # Two steps: at init the SVF head kernel is zero, so step 1 only moves the head.
        for _ in range(2):
            res = tr.train_step("deform", self.dgroup["batch"])
            self.assertTrue(res["finite"])
            self.assertEqual(res["n_vars_updated"], n_tv - len(lin.trainable_variables))
            for k in DEFORM_LOSS_KEYS + DEFORM_DIAG_KEYS:
                self.assertTrue(np.isfinite(res[k]), k)
        self.assertTrue(_all_equal(lin_before, lin.variables))        # whole linear decoder untouched
        self.assertFalse(np.array_equal(head_before, dfm.svf_head.kernel.numpy()))
        self.assertFalse(np.array_equal(enc_before, model.encoder.l3.conv.kernel.numpy()))
        self.assertGreater(res["diag_u_max_vox"], 0.0)          # step 2 sees the updated head
        self.assertGreaterEqual(res["diag_fold_frac"], 0.0)
        self.assertLessEqual(res["diag_fold_frac"], 1.0)
        dfm_after = _vals(dfm.variables)
        self.assertFalse(_all_equal(dfm_before, dfm.variables))
        res = tr.train_step("rigid", self.groups["rigid"]["batch"])
        self.assertTrue(res["finite"])
        self.assertEqual(res["n_vars_updated"], n_tv - 2 - len(dfm.trainable_variables))
        self.assertTrue(_all_equal(dfm_after, dfm.variables))         # deform decoder untouched (BN too)
        self.assertEqual(tr.trace_counts, {"rigid": 1, "deform": 1})

    def test_nan_guard_skips_update(self):
        imgs = self.dgroup["images"].copy()
        imgs[0, 3, 4, 5, 0] = np.nan
        nan_batch = fd.to_deform_batch(imgs, self.dgroup["labels"], level=4)
        tr = self.make_deform()
        self.assertTrue(tr.train_step("deform", self.dgroup["batch"])["finite"])
        model, opt = tr.model, tr.optimizer
        before_t, before_s, before_o = (_vals(model.trainable_variables), _vals(model.non_trainable_variables),
                                        _vals(opt.variables()))
        it = int(opt.iterations.numpy())
        res = tr.train_step("deform", nan_batch)
        self.assertFalse(res["finite"])
        self.assertEqual(int(opt.iterations.numpy()), it)
        self.assertTrue(_all_equal(before_t, model.trainable_variables))
        self.assertTrue(_all_equal(before_s, model.non_trainable_variables))   # BN statistics restored
        self.assertTrue(_all_equal(before_o, opt.variables()))
        self.assertEqual(int(tr.step.numpy()), 2)
        self.assertTrue(tr.train_step("deform", self.dgroup["batch"])["finite"])
        self.assertEqual(tr.trace_counts, {"deform": 1})

    def test_residual_affine_step(self):
        # --deform-residual-affine: the DALI tuple carries T_dali [N, 3, 4] (FEAT-002 format);
        # the deform step takes three inputs and split_batch keeps them, dropping later extras.
        T = np.tile(np.eye(4), (N, 1, 1))
        T[:, 0, 3] = 1.0
        batch = fd.to_deform_batch(self.dgroup["images"], self.dgroup["labels"], T, level=4)
        self.assertEqual(len(batch), 3)
        tr = self.make_deform(config(task_ratio={"deform": 1.0}, deform_residual_affine=True),
                              sources={"deform": fd.FixedGroupSource(batch)})
        self.assertEqual(len(tr.split_batch("deform", tuple(batch) + ("extra",))[0]), 3)
        tr.fit(2)
        self.assertTrue(all(h["finite"] for h in tr.history))
        self.assertEqual(tr.trace_counts, {"deform": 1})

    def test_collapse_diag_deform(self):
        rep = lambda a: np.repeat(a[:1], N, axis=0)  # noqa: E731
        src = {"deform": fd.FixedGroupSource(fd.to_deform_batch(rep(self.dgroup["images"]),
                                                                rep(self.dgroup["labels"]), level=4))}
        paths = make_run_dir(self.tmp, "dcollapse")
        logger = RunLogger(paths, backend="jsonl", tensorboard=False, quiet=True)
        tr = self.make_deform(config(task_ratio={"deform": 1.0}, collapse_patience=3), sources=src, logger=logger)
        tr.fit(4)
        logger.close()
        for h in tr.history:
            self.assertTrue(h["finite"])
            self.assertLess(h["diag_feat_rel_spread"], 1e-5)
        self.assertEqual((tr.collapse_streak, tr.collapse_warnings), (4, 1))
        with open(logger.metrics_path) as f:
            recs = [json.loads(line)["data"] for line in f]
        self.assertEqual([r["train/collapse_suspect"] for r in recs if "train/collapse_suspect" in r], [1, 2, 3, 4])
        for k in DEFORM_LOSS_KEYS + DEFORM_DIAG_KEYS:
            self.assertIn("train_deform/" + k, recs[-1])
        self.assertNotIn("train_deform/center", recs[-1])
        # A healthy (distinct) group has a clearly non-zero spread.
        tr2 = self.make_deform()
        self.assertGreater(tr2.train_step("deform", self.dgroup["batch"])["diag_feat_rel_spread"], 1e-3)


class TestThreeTasks(DeformCase):
    def test_trace_once_mixed(self):
        ratio = {"rigid": 1.0, "affine": 1.0, "deform": 2.0}
        n_steps = 16
        seed = _seed_with_all_tasks(ratio, n_steps)
        tr = self.make_deform(config(task_ratio=ratio, seed=seed, flip_prob=0.5))
        tvars = tr.model.trainable_variables
        n_opt = len(tr.optimizer.variables())
        self.assertEqual(n_opt, 2 * len(tvars))         # slots for every variable before step 1
        tr.fit(n_steps)
        tasks = [h["task"] for h in tr.history]
        self.assertEqual(tasks, [tr.schedule.draw(s) for s in range(n_steps)])
        self.assertEqual(set(tasks), set(ratio))
        self.assertEqual(tr.trace_counts, {"rigid": 1, "affine": 1, "deform": 1})
        self.assertTrue(all(h["finite"] for h in tr.history))
        self.assertEqual(len(tr.optimizer.variables()), n_opt)  # no slot created later
        self.assertEqual(int(tr.optimizer.iterations.numpy()), n_steps)
        n_lin = len(tr.model.linear_decoder.trainable_variables)
        n_dfm = len(tr.model.deform_decoder.trainable_variables)
        want = {"rigid": len(tvars) - 2 - n_dfm, "affine": len(tvars) - n_dfm, "deform": len(tvars) - n_lin}
        for h in tr.history:
            self.assertEqual(h["n_vars_updated"], want[h["task"]], h["task"])
        # Flips are on: the deform postprocess draws a flip state from its own generator.
        self.assertEqual(sorted(tr.postprocess), ["affine", "deform", "rigid"])

    def test_enc_grad_diagnostics_logged(self):
        # Milestone 7: per-task encoder grad norms and per-task grad_norm in the window logs.
        ratio = {"rigid": 1.0, "affine": 1.0, "deform": 2.0}
        seed = _seed_with_all_tasks(ratio, 8)
        paths = make_run_dir(self.tmp, "encgrad")
        logger = RunLogger(paths, backend="jsonl", tensorboard=False, quiet=True)
        tr = self.make_deform(config(task_ratio=ratio, seed=seed, log_every=1), logger=logger)
        tr.fit(8)
        logger.close()
        with open(logger.metrics_path) as f:
            recs = [json.loads(line)["data"] for line in f]
        for t in ratio:
            keys = ["train_%s/%s" % (t, k) for k in ENC_GRAD_KEYS] + ["train_%s/grad_norm" % t]
            got = [r for r in recs if keys[0] in r]
            self.assertTrue(got, t)
            for r in got:
                for k in keys:
                    self.assertTrue(np.isfinite(r[k]), k)
                    self.assertGreaterEqual(r[k], 0.0, k)
                self.assertEqual(r["train_%s/grad_norm" % t], r["train/grad_norm"])   # one task per step
        for h in tr.history:
            for k in ENC_GRAD_KEYS:
                self.assertIn(k, h)

    def test_deform_group_size_differs(self):
        # Linear N = 4, deform N = 2: each task's step has its own signature, validation uses each
        # set's own group size, both best files are written.
        ratio = {"rigid": 1.0, "affine": 1.0, "deform": 2.0}
        n_steps = 8
        seed = _seed_with_all_tasks(ratio, n_steps)
        imgs, labs = self.dgroup["images"][:2], self.dgroup["labels"][:2]
        sources = {t: fd.FixedGroupSource(g["batch"]) for t, g in self.groups.items()}
        sources["deform"] = fd.FixedGroupSource(fd.to_deform_batch(imgs, labs, level=4))
        val_sets = {t: fd.ValidationSet.from_group(g["images"], g["T"], g["T_target"], t, 4)
                    for t, g in self.groups.items()}
        val_sets["deform"] = fd.DeformValidationSet.from_group(imgs, labs, 4)
        cfg = config(task_ratio=ratio, seed=seed, deform_group_size=2, val_every=n_steps)
        tr = self.make_deform(cfg, sources=sources, val_sets=val_sets)
        self.assertEqual(tr.extra_best, {"deform": DEFORM_BEST_METRIC})
        self.assertEqual(tr._steps["deform"].input_signature[0].shape[0], 2)
        self.assertEqual(tr._steps["rigid"].input_signature[0].shape[0], N)
        paths = make_run_dir(self.tmp, "dn2")
        tr.checkpoint = C.RunCheckpoint(paths["ckpt_last"], paths["ckpt_best"], tr.model, tr.optimizer, tr.step,
                                        extra=tr.checkpoint_variables(),
                                        extra_best={"deform": os.path.join(paths["root"], "ckpt", "best_deform")})
        tr.fit(n_steps)
        self.assertEqual(tr.trace_counts, {"rigid": 1, "affine": 1, "deform": 1})
        self.assertTrue(all(h["finite"] for h in tr.history))
        v = tr.val_history[-1]
        for k in ("val_deform/pair_dice", "val/mean_pair_rms_mm", "val_rigid/pair_rms_mm", "val_affine/pair_rms_mm"):
            self.assertTrue(np.isfinite(v[k]), k)
        self.assertEqual(sorted(tr.val_trace_counts),
                         sorted([("rigid", N, False), ("rigid", N, True), ("affine", N, False),
                                 ("affine", N, True), ("deform", 2, False), ("deform", 2, True)]))
        self.assertTrue(tr.checkpoint.latest_best.endswith("best-%d" % n_steps))
        self.assertTrue(tr.checkpoint.latest_best_named("deform").endswith("best_deform-%d" % n_steps))


class TestTaskLossWeightDeform(DeformCase):
    def test_deform_weight(self):
        def make(w, clipnorm=None, like=None):
            model = GroupRegLevelModel(4, narrow_config(dropout=0.0, seed=1, deform_decoder=True)).build_model(N)
            if like is None:
                head = model.deform_decoder.svf_head
                head.kernel.assign(np.random.default_rng(0).normal(0, 1e-3, head.kernel.shape).astype(np.float32))
            else:
                model.set_weights(like.get_weights())
            cfg = config(task_ratio={"deform": 1.0}, task_loss_weights={"deform": w})
            return Trainer(model, O.make_optimizer("adam", lr=1e-3, clipnorm=clipnorm),
                           {"deform": fd.FixedGroupSource(self.dgroup["batch"])}, self.groups["rigid"]["S"], cfg)
        batch = self.dgroup["batch"]
        tr1 = make(1.0)
        tr3 = make(3.0, like=tr1.model)
        trc = make(1.0, clipnorm=1e-4, like=tr1.model)
        r1, r3, rc = (t.train_step("deform", batch) for t in (tr1, tr3, trc))
        self.assertGreater(r1["diag_enc_grad_norm_raw"], 0.0)
        for k in DEFORM_LOSS_KEYS:
            np.testing.assert_allclose(r3[k], r1[k], rtol=1e-5, err_msg=k)
        np.testing.assert_allclose(r3["grad_norm"], 3.0 * r1["grad_norm"], rtol=1e-4)
        np.testing.assert_allclose(r3["diag_enc_grad_norm_raw"], r1["diag_enc_grad_norm_raw"], rtol=1e-4)
        np.testing.assert_allclose(r1["diag_enc_grad_norm_eff"], r1["diag_enc_grad_norm_raw"], rtol=1e-6)
        np.testing.assert_allclose(r3["diag_enc_grad_norm_eff"], 3.0 * r1["diag_enc_grad_norm_eff"], rtol=1e-4)
        self.assertGreater(rc["grad_norm"], 1e-4)
        np.testing.assert_allclose(rc["diag_enc_grad_norm_eff"],
                                   rc["diag_enc_grad_norm_raw"] * 1e-4 / rc["grad_norm"], rtol=1e-4)


class TestDeformValidation(DeformCase):
    def test_deform_validation_and_pngs(self):
        vs = fd.DeformValidationSet.from_group(self.dgroup["images"], self.dgroup["labels"], 4)
        png = os.path.join(self.tmp, "png")
        tr = self.make_deform(config(task_ratio={"deform": 1.0}, val_png_dir=png), val_sets={"deform": vs})
        m = tr.validate()
        for k in ("template_dice", "pair_dice", "neg_jac_pct", "inv_consistency_mm", "inv_consistency_max_mm",
                  "centrality_mm", "raw_centrality_mm", "pair_dice_loss", "pair_dice_identity",
                  "template_dice_identity", "groups", "time_s"):
            self.assertTrue(np.isfinite(m["val_deform/" + k]), k)
        for k in ("template_dice", "pair_dice"):
            self.assertTrue(np.isfinite(m["val_deform_trainmode/" + k]), k)
        self.assertNotIn("val/mean_pair_rms_mm", m)
        # Identity model at init (zero SVF head): metrics equal the identity baseline.
        self.assertAlmostEqual(m["val_deform/pair_dice"], m["val_deform/pair_dice_identity"], places=10)
        self.assertAlmostEqual(m["val_deform/template_dice"], m["val_deform/template_dice_identity"], places=10)
        self.assertAlmostEqual(m["val_deform/pair_dice_loss"], 1.0 - m["val_deform/pair_dice"], places=12)
        self.assertGreater(m["val_deform/pair_dice"], 0.4)
        self.assertLess(m["val_deform/pair_dice"], 0.9)
        self.assertEqual(m["val_deform/neg_jac_pct"], 0.0)
        self.assertEqual(m["val_deform/inv_consistency_mm"], 0.0)
        self.assertEqual(m["val_deform/centrality_mm"], 0.0)
        from runtime import val_png as V
        path = os.path.join(png, "deform_step0000000.png")
        self.assertEqual(os.listdir(png), ["deform_step0000000.png"])
        from tests.test_val_png import png_size
        self.assertEqual(png_size(path), V.figure_size_px(4, N + 1))
        tr.fit(2)
        m2 = tr.validate()
        self.assertGreater(m2["val_deform/inv_consistency_mm"], 0.0)     # u != 0 after training steps
        self.assertTrue(np.isfinite(m2["val_deform/pair_dice"]))
        self.assertEqual(m2["val_deform/pair_dice_identity"], m["val_deform/pair_dice_identity"])
        self.assertEqual(tr.val_trace_counts, {("deform", N, False): 1, ("deform", N, True): 1})
        # Pair cap: N (N - 1) = 12 > 5 -> a fixed seeded subset of 5 pairs.
        tr.cfg.val_deform_max_pairs = 5
        i, j = tr._val_pairs(N, seed=0)
        self.assertEqual(len(i), 5)
        self.assertTrue(np.all(i != j))
        np.testing.assert_array_equal(i, tr._val_pairs(N, seed=0)[0])
        tr.cfg.val_deform_max_pairs = 64
        self.assertEqual(len(tr._val_pairs(N, seed=0)[0]), 12)

    def test_best_metrics_with_deform(self):
        vs = fd.DeformValidationSet.from_group(self.dgroup["images"], self.dgroup["labels"], 4)
        # deform-only: ckpt/best on val_deform/pair_dice_loss, no extra best file.
        paths = make_run_dir(self.tmp, "donly")
        tr = self.make_deform(config(task_ratio={"deform": 1.0}, val_every=2), val_sets={"deform": vs})
        self.assertEqual(tr.best_metric, DEFORM_BEST_METRIC)
        self.assertEqual(tr.extra_best, {})
        tr.checkpoint = C.RunCheckpoint(paths["ckpt_last"], paths["ckpt_best"], tr.model, tr.optimizer, tr.step,
                                        extra=tr.checkpoint_variables())
        tr.fit(2)
        self.assertTrue(tr.checkpoint.latest_best.endswith("best-2"))
        self.assertAlmostEqual(float(tr.checkpoint.best_metric.numpy()),
                               tr.val_history[-1][DEFORM_BEST_METRIC], places=10)
        free_memory()
        # rigid + deform: ckpt/best keeps the linear rule, ckpt/best_deform tracks the deform key.
        g = self.groups["rigid"]
        val_sets = {"rigid": fd.ValidationSet.from_group(g["images"], g["T"], g["T_target"], "rigid", 4),
                    "deform": vs}
        paths = make_run_dir(self.tmp, "mixed")
        tr = self.make_deform(config(task_ratio={"rigid": 1.0, "deform": 1.0}, val_every=2), val_sets=val_sets)
        self.assertEqual(tr.best_metric, "val_rigid/pair_rms_mm")
        self.assertEqual(tr.extra_best, {"deform": DEFORM_BEST_METRIC})
        best_deform = os.path.join(paths["root"], "ckpt", "best_deform")
        tr.checkpoint = C.RunCheckpoint(paths["ckpt_last"], paths["ckpt_best"], tr.model, tr.optimizer, tr.step,
                                        extra=tr.checkpoint_variables(), extra_best={"deform": best_deform})
        tr.fit(2)
        v = tr.val_history[-1]
        self.assertIn("val/mean_pair_rms_mm", v)                 # over the linear task only
        self.assertEqual(v["val/mean_pair_rms_mm"], v["val_rigid/pair_rms_mm"])
        self.assertTrue(tr.checkpoint.latest_best.endswith("best-2"))
        self.assertTrue(tr.checkpoint.latest_best_named("deform").endswith("best_deform-2"))
        self.assertAlmostEqual(float(tr.checkpoint.best_metric.numpy()), v["val_rigid/pair_rms_mm"], places=6)
        self.assertAlmostEqual(float(tr.checkpoint.extra_best_metric["deform"].numpy()), v[DEFORM_BEST_METRIC],
                               places=10)


if __name__ == "__main__":
    unittest.main()
