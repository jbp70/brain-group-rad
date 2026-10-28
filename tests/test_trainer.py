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
import contextlib
import io
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
from runtime.run import (DEFORM_BEST_METRIC, DEFORM_DIAG_KEYS, DEFORM_LOSS_KEYS, DEFORM_NCC_KEYS, DIAG_KEYS,
                         ENC_GRAD_KEYS, LOSS_SCALE_KEY, NONFINITE_WINDOW_KEYS, PAIR_COUNT_KEY,
                         POSTPROCESS_SEED_OFFSET, TASK_GRAD_NORM_EPS, TASK_GRAD_NORM_KEYS, TaskSchedule, Trainer, TrainConfig, collapse_diagnostics, scale_gradients,
                         task_grad_norm_scale)
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

    @staticmethod
    def _old_draws(ratio, seed, n):
        """Hard-coded replica of the pre-delayed-start TaskSchedule.draw loop."""
        import random
        ratio = [(k, float(v)) for k, v in ratio.items() if float(v) > 0]
        total = sum(v for _, v in ratio)
        acc, cum = 0.0, []
        for k, v in ratio:
            acc += v / total
            cum.append((acc, k))
        out = []
        for step in range(n):
            if len(cum) == 1:
                out.append(cum[0][1])
                continue
            u = random.Random(seed * 1000003 + step).random()
            for c, k in cum:
                if u < c:
                    out.append(k)
                    break
            else:
                out.append(cum[-1][1])
        return out

    def test_no_start_steps_unchanged(self):
        # Retrain-prereqs 11.4 a: without starts (None, {} or all 0) the draws are today's, exactly.
        for ratio, seed in (({"rigid": 1, "affine": 1, "deform": 2}, 0), ({"rigid": 3, "affine": 1}, 5),
                            ({"deform": 1}, 2), ({"rigid": 1, "deform": 0.5}, 17)):
            want = self._old_draws(ratio, seed, 10000)
            for starts in (None, {}, {t: 0 for t in ratio}):
                s = TaskSchedule(ratio, seed, starts)
                self.assertEqual([s.draw(i) for i in range(10000)], want, (ratio, seed, starts))
            self.assertEqual([TaskSchedule(ratio, seed).draw(i) for i in range(10000)], want)

    def test_delayed_start(self):
        # Retrain-prereqs 11.4 b: deform:3000 -> no deform before 3000; 1:1 before, 1:1:2 after.
        ratio = {"rigid": 1, "affine": 1, "deform": 2}
        s = TaskSchedule(ratio, 0, {"deform": 3000})
        self.assertEqual(s.active_tasks(2999), ["rigid", "affine"])
        self.assertEqual(s.active_tasks(3000), ["rigid", "affine", "deform"])
        before = [s.draw(i) for i in range(3000)]
        self.assertNotIn("deform", before)
        self.assertAlmostEqual(before.count("rigid") / 3000.0, 0.5, delta=0.02)
        after = [s.draw(i) for i in range(3000, 23000)]
        for t, f in (("rigid", 0.25), ("affine", 0.25), ("deform", 0.5)):
            self.assertAlmostEqual(after.count(t) / 20000.0, f, delta=0.02, msg=t)
        # From the start on every task is active: today's draws exactly.
        self.assertEqual(after, [TaskSchedule(ratio, 0).draw(i) for i in range(3000, 23000)])
        # 1:1 over 20000 draws before a (far) start; deterministic and call-order independent.
        late = TaskSchedule(ratio, 0, {"deform": 30000})
        pre = [late.draw(i) for i in range(20000)]
        self.assertNotIn("deform", pre)
        self.assertAlmostEqual(pre.count("rigid") / 20000.0, 0.5, delta=0.02)
        self.assertEqual(pre, [TaskSchedule(ratio, 0, {"deform": 30000}).draw(i) for i in range(20000)])
        rev = {i: late.draw(i) for i in reversed(range(500))}
        self.assertEqual([rev[i] for i in range(500)], pre[:500])
        # A single active task is returned without a draw.
        one = TaskSchedule({"rigid": 1, "deform": 2}, 0, {"deform": 10})
        self.assertEqual({one.draw(i) for i in range(10)}, {"rigid"})
        for bad in ({"affine": 3}, {"rigid": 2, "deform": 3}, {"deform": -1}):
            with self.assertRaises(ValueError):
                TaskSchedule({"rigid": 1, "deform": 2}, 0, bad)


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


class TestMinPairsTrainer(DeformCase):
    """T-G1 (milestone 9 phase 2, design-m9-phase2 section 5): --min-pairs '1' builds no min_pairs
    op and no n_pairs result key; 'n' adds both; validation pairs are unaffected."""

    def _pp_ops(self, tr):
        pp = tr.postprocess["deform"]
        x = tf.TensorSpec((N,) + tuple(P.LEVEL_SHAPE[4]) + (1,), tf.float32)
        y = tf.TensorSpec((N,) + tuple(P.LEVEL_SHAPE[4]) + (1,), tf.uint8)
        fn = tf.function(lambda a, b: pp(a, b), input_signature=[x, y])
        gd = fn.get_concrete_function().graph.as_graph_def()
        # Outer nodes plus the nodes of the function library (tf.cond branch bodies).
        return [n.name for n in gd.node] + [n.name for f in gd.library.function for n in f.node_def]

    def test_graphs_and_results(self):
        ratio = {"rigid": 1.0, "deform": 1.0}
        tr1 = self.make_deform(config(task_ratio=ratio))
        trn = self.make_deform(config(task_ratio=ratio, min_pairs="n"))
        for tr, on in ((tr1, False), (trn, True)):
            for task, batch in (("rigid", self.groups["rigid"]["batch"]), ("deform", self.dgroup["batch"])):
                res = tr.train_step(task, batch)
                self.assertTrue(res["finite"])
                self.assertEqual(PAIR_COUNT_KEY in res, on, (task, on))
                if on:
                    self.assertEqual(res[PAIR_COUNT_KEY], float(N))      # N 4, p 0.1 -> k = N (cycle)
                ops = [op.name for op in tr._steps[task].get_concrete_function().graph.get_operations()]
                self.assertEqual(any("min_pairs/" in o for o in ops), on, (task, on))
        ops1, opsn = self._pp_ops(tr1), self._pp_ops(trn)
        self.assertFalse(any("min_pairs/" in o for o in ops1))
        self.assertTrue(any("min_pairs/" in o for o in opsn))
        self.assertGreater(len(opsn), len(ops1))
        free_memory()

    def test_logged_key(self):
        paths = make_run_dir(self.tmp, "minpairs")
        logger = RunLogger(paths, backend="jsonl", tensorboard=False, quiet=True)
        tr = self.make_deform(config(task_ratio={"deform": 1.0}, min_pairs="n"), logger=logger)
        tr.fit(2)
        logger.close()
        with open(logger.metrics_path) as f:
            recs = [json.loads(line)["data"] for line in f]
        self.assertEqual([r["train_deform/n_pairs"] for r in recs if "train_deform/n_pairs" in r][-1], float(N))
        tr1 = self.make_deform(config(task_ratio={"deform": 1.0}))
        tr1.fit(1)
        self.assertNotIn(PAIR_COUNT_KEY, tr1.history[-1])

    def test_validation_unaffected_and_invalid(self):
        g = self.groups["rigid"]
        val_sets = {"rigid": fd.ValidationSet.from_group(g["images"], g["T"], g["T_target"], "rigid", 4),
                    "deform": fd.DeformValidationSet.from_group(self.dgroup["images"], self.dgroup["labels"], 4)}
        ratio = {"rigid": 1.0, "deform": 1.0}
        m1 = self.make_deform(config(task_ratio=ratio), val_sets=val_sets).validate(max_groups=1)
        mn = self.make_deform(config(task_ratio=ratio, min_pairs="n"), val_sets=val_sets).validate(max_groups=1)
        self.assertEqual(sorted(m1), sorted(mn))
        for k in m1:
            if "time" in k:
                continue
            self.assertEqual(m1[k], mn[k], k)
        with self.assertRaises(ValueError):
            self.make_deform(config(task_ratio=ratio, min_pairs="x"))

    def test_max_pairs(self):
        """--max-pairs (retrain-prereqs design 16.3): the cap reaches both post-processes, the
        step results carry the capped n_pairs (also with min_pairs '1'), invalid values raise."""
        ratio = {"rigid": 1.0, "deform": 1.0}
        for mp, cap, want in (("n", 3, 3.0), ("1", 3, 1.0), ("n", 64, float(N))):
            tr = self.make_deform(config(task_ratio=ratio, min_pairs=mp, max_pairs=cap))
            self.assertEqual({t: p.max_pairs for t, p in tr.postprocess.items()},
                             {"rigid": cap, "deform": cap})
            for task, batch in (("rigid", self.groups["rigid"]["batch"]), ("deform", self.dgroup["batch"])):
                res = tr.train_step(task, batch)
                self.assertTrue(res["finite"])
                self.assertEqual(res[PAIR_COUNT_KEY], want, (mp, cap, task))
            free_memory()
        for bad in (-1, 1.5, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.make_deform(config(task_ratio=ratio, max_pairs=bad))


class TestTFIntensityAug(DeformCase):
    """--intensity-aug-device tf: TrainConfig.intensity_aug_tf reaches every post-process, its
    generators are checkpointed (only then) and resume continues their stream."""

    def test_checkpoint_keys(self):
        ratio = {"rigid": 1.0, "deform": 1.0}
        tr = self.make_deform(config(task_ratio=ratio))
        self.assertEqual(sorted(tr.checkpoint_variables()), ["postprocess_deform", "postprocess_rigid"])
        free_memory()
        tr = self.make_deform(config(task_ratio=ratio, intensity_aug_tf=P.IntensityAugConfig()))
        self.assertEqual(sorted(tr.checkpoint_variables()),
                         ["postprocess_deform", "postprocess_deform_aug", "postprocess_rigid",
                          "postprocess_rigid_aug"])
        for t, pp in tr.postprocess.items():
            self.assertIsNotNone(pp.intensity_aug, t)

    def test_steps_finite_and_resume_continues_aug_stream(self):
        cfg = config(task_ratio={"rigid": 1.0, "deform": 1.0}, intensity_aug_tf=P.IntensityAugConfig())
        paths = make_run_dir(self.tmp, "tfaug")

        def with_ckpt(tr):
            tr.checkpoint = C.RunCheckpoint(paths["ckpt_last"], paths["ckpt_best"], tr.model, tr.optimizer,
                                            tr.step, extra=tr.checkpoint_variables())
            return tr

        tr = with_ckpt(self.make_deform(cfg))
        for task, batch in (("rigid", self.groups["rigid"]["batch"]), ("deform", self.dgroup["batch"])):
            res = tr.train_step(task, batch)
            self.assertTrue(res["finite"], task)
            self.assertTrue(np.isfinite(float(res["loss"])), task)
        tr.fit(3)
        states = {t: pp.aug_generator.state.numpy().copy() for t, pp in tr.postprocess.items()}
        for t in states:                    # the stream advanced (2 + fit steps)
            start = tf.random.Generator.from_seed(
                cfg.seed + POSTPROCESS_SEED_OFFSET[t] + P.INTENSITY_AUG_SEED_OFFSET).state.numpy()
            self.assertFalse(np.array_equal(states[t], start), t)
        free_memory()
        tr2 = with_ckpt(self.make_deform(cfg))
        self.assertTrue(tr2.checkpoint.restore_last())
        for t, pp in tr2.postprocess.items():
            np.testing.assert_array_equal(pp.aug_generator.state.numpy(), states[t], t)

    def test_tent_lowres_step_finite(self):
        """retrain-prereqs 3: low_res_method 'tent' runs inside the Trainer post-process."""
        aug = P.IntensityAugConfig(low_res_prob=1.0, low_res_method="tent")
        tr = self.make_deform(config(task_ratio={"rigid": 1.0, "deform": 1.0}, intensity_aug_tf=aug))
        for task, batch in (("rigid", self.groups["rigid"]["batch"]), ("deform", self.dgroup["batch"])):
            res = tr.train_step(task, batch)
            self.assertTrue(res["finite"], task)
            self.assertTrue(np.isfinite(float(res["loss"])), task)


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


class TestDelayedTaskStart(DeformCase):
    """Retrain-prereqs 11.4 c / e: the delayed task's LR ramp in fit() and a short three-task run
    with a delayed deform start."""

    def test_lr_ramp(self):
        # 11.4 c: affine starts at 6 with a 4-step ramp; rigid steps keep lr_at(s).
        from runtime.schedulers import lr_at
        start, W, n_steps = 6, 4, 14
        seed = next(sd for sd in range(1000)
                    if {TaskSchedule({"rigid": 1, "affine": 1}, sd, {"affine": start}).draw(i)
                        for i in range(start, start + W)} == {"rigid", "affine"}
                    and "affine" in {TaskSchedule({"rigid": 1, "affine": 1}, sd, {"affine": start}).draw(i)
                                     for i in range(start + W, n_steps)})
        sched = ScheduleConfig("cosine", base_lr=1e-3, total_steps=n_steps, warmup_steps=2)
        cfg = config(task_ratio={"rigid": 1.0, "affine": 1.0}, seed=seed, schedule=sched, steps=n_steps,
                     task_start_steps={"affine": start}, task_start_ramp_steps=W)
        tr = self.make(cfg)
        tr.fit()
        self.assertEqual(len(tr.history), n_steps)
        seen = set()
        for h in tr.history:
            s = h["step"] - 1
            base = lr_at(s, sched)
            if h["task"] == "affine":
                self.assertGreaterEqual(s, start)
                want = base * (s - start + 1) / W if s < start + W else base
                seen.add("ramp" if s < start + W else "after")
            else:
                want = base
            self.assertEqual(h["lr"], want, (s, h["task"]))
            self.assertTrue(h["finite"])
        self.assertEqual(seen, {"ramp", "after"})
        self.assertAlmostEqual(float(tr.optimizer.learning_rate.numpy()), tr.history[-1]["lr"], places=9)
        # Default config: no multiplication anywhere.
        tr0 = self.make(config(task_ratio={"rigid": 1.0, "affine": 1.0}, seed=seed, schedule=sched, steps=4))
        tr0.fit()
        self.assertEqual([h["lr"] for h in tr0.history], [lr_at(s, sched) for s in range(4)])
        with self.assertRaises(ValueError):
            self.make(config(task_ratio={"rigid": 1.0}, task_start_steps={"affine": 3}))
        with self.assertRaises(ValueError):
            self.make(config(task_ratio={"rigid": 1.0}, task_start_steps={"rigid": 3}))
        with self.assertRaises(ValueError):
            self.make(config(task_ratio={"rigid": 1.0, "affine": 1.0}, task_start_steps={"affine": 3},
                             task_start_ramp_steps=-1))

    def test_short_three_task_run(self):
        # 11.4 e: TINY synthetic three-task run, 12 steps, deform starts at 6.
        ratio = {"rigid": 1.0, "affine": 1.0, "deform": 2.0}
        start, n_steps = 6, 12
        seed = next(sd for sd in range(1000)
                    if "deform" in {TaskSchedule(ratio, sd, {"deform": start}).draw(i)
                                    for i in range(start, n_steps)})
        tr = self.make_deform(config(task_ratio=ratio, seed=seed, steps=n_steps,
                                     task_start_steps={"deform": start}, task_start_ramp_steps=3))
        tr.fit()
        tasks = [h["task"] for h in tr.history]
        self.assertEqual(len(tasks), n_steps)
        self.assertNotIn("deform", tasks[:start])
        self.assertIn("deform", tasks[start:])
        self.assertTrue(all(h["finite"] for h in tr.history))
        for h in tr.history:
            self.assertTrue(np.isfinite(h["loss"]), h)
        self.assertEqual(tr.trace_counts.get("deform"), 1)


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


class TestDeformNCC(DeformCase):
    """v3.9: LNCC terms in the deform step (weights > 0)."""

    def make_ncc(self, like=None, **kw):
        cfg = config(task_ratio={"deform": 1.0}, **kw)
        tr = self.make_deform(cfg)
        if like is None:
            head = tr.model.deform_decoder.svf_head
            head.kernel.assign(np.random.default_rng(0).normal(0, 1e-3, head.kernel.shape).astype(np.float32))
        else:
            tr.model.set_weights(like.get_weights())
        return tr

    def test_ncc_step(self):
        batch = self.dgroup["batch"]
        tr0 = self.make_ncc()
        tr1 = self.make_ncc(like=tr0.model, lambda_ncc_tmpl=1.0, lambda_ncc_pair=1.0)
        self.assertEqual(tr1.ncc_window, P.ncc_window_for_level(4))
        r0 = tr0.train_step("deform", batch)
        for _ in range(2):
            r1 = tr1.train_step("deform", batch)
            self.assertTrue(r1["finite"])
            for k in DEFORM_NCC_KEYS + DEFORM_LOSS_KEYS:
                self.assertTrue(np.isfinite(r1[k]), k)
            for k in ("ncc_tmpl", "ncc_pair", "ncc_tmpl_loss", "ncc_pair_loss"):
                self.assertGreaterEqual(r1[k], 0.0, k)
                self.assertLessEqual(r1[k], 1.0, k)
        for k in DEFORM_NCC_KEYS:
            self.assertNotIn(k, r0)
        self.assertEqual(r1["n_vars_updated"], r0["n_vars_updated"])
        self.assertEqual(len(tr1.model.variables), len(tr0.model.variables))
        self.assertEqual(tr1.trace_counts, {"deform": 1})
        # From the same weights: the LNCC terms add to the loss and change the gradient.
        tr2 = self.make_ncc(like=tr0.model, lambda_ncc_tmpl=1.0, lambda_ncc_pair=1.0)
        tr0b = self.make_ncc(like=tr0.model)
        r2 = tr2.train_step("deform", batch)
        np.testing.assert_allclose(r2["loss"], r2["tmpl"] + r2["pdice"] + 0.5 * r2["smooth"]
                                   + r2["ncc_tmpl_loss"] + r2["ncc_pair_loss"], rtol=1e-5)
        self.assertNotEqual(r2["grad_norm"], tr0b.train_step("deform", batch)["grad_norm"])

    def test_lncc_only_and_validation_errors(self):
        tr = self.make_ncc(lambda_dice_tmpl=0.0, lambda_pdice=0.0, lambda_ncc_tmpl=1.0, lambda_ncc_pair=1.0)
        r = tr.train_step("deform", self.dgroup["batch"])
        self.assertTrue(r["finite"])
        self.assertTrue(np.isfinite(r["soft_dice_tmpl"]))
        np.testing.assert_allclose(r["loss"], 0.5 * r["smooth"] + r["ncc_tmpl_loss"] + r["ncc_pair_loss"],
                                   rtol=1e-5)
        for bad in (dict(lambda_ncc_tmpl=-1.0), dict(lambda_ncc_pair=float("nan")),
                    dict(lambda_dice_tmpl=float("inf")), dict(ncc_window=4), dict(ncc_window=41)):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.make_deform(config(task_ratio={"deform": 1.0}, **bad))

    def test_three_task_logs_ncc(self):
        ratio = {"rigid": 1.0, "affine": 1.0, "deform": 2.0}
        seed = _seed_with_all_tasks(ratio, 8)
        paths = make_run_dir(self.tmp, "ncc3")
        logger = RunLogger(paths, backend="jsonl", tensorboard=False, quiet=True)
        tr = self.make_deform(config(task_ratio=ratio, seed=seed, log_every=1, lambda_ncc_tmpl=1.0,
                                     lambda_ncc_pair=1.0), logger=logger)
        tr.fit(8)
        logger.close()
        self.assertEqual(tr.trace_counts, {"rigid": 1, "affine": 1, "deform": 1})
        self.assertTrue(all(h["finite"] for h in tr.history))
        with open(logger.metrics_path) as f:
            recs = [json.loads(line)["data"] for line in f]
        got = [r for r in recs if "train_deform/loss" in r]
        self.assertTrue(got)
        for r in got:
            for k in DEFORM_NCC_KEYS:
                self.assertTrue(np.isfinite(r["train_deform/" + k]), k)
        self.assertFalse(any("train_rigid/ncc_tmpl" in r for r in recs))


DEFORM_GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                                  "lncc_deform_golden.json")
DEFORM_GOLDEN_STEPS = 8


def _three_task_run(steps=DEFORM_GOLDEN_STEPS, clipnorm=12.0, **cfg_kw):
    """`steps` seeded three-task steps (rigid:1, affine:1, deform:2, flips on) of a narrow deform
    model (dropout 0, SVF head kernel N(0, 1e-3) so deform gradients reach the encoder), Adam lr
    1e-3 with global clip `clipnorm`, extra TrainConfig fields `cfg_kw`. Returns ({seed, tasks,
    per-step loss, per-step deform terms, per-variable (float64 sum, max |v|)}, history)."""
    ratio = {"rigid": 1.0, "affine": 1.0, "deform": 2.0}
    seed = _seed_with_all_tasks(ratio, steps)
    groups = {t: synthetic_group(4, N, t, seed=i) for i, t in enumerate(("rigid", "affine"))}
    dgroup = fd.synthetic_deform_group(4, N, seed=0)
    model = GroupRegLevelModel(4, narrow_config(dropout=0.0, seed=1, deform_decoder=True)).build_model(N)
    head = model.deform_decoder.svf_head
    head.kernel.assign(np.random.default_rng(0).normal(0, 1e-3, head.kernel.shape).astype(np.float32))
    sources = {t: fd.FixedGroupSource(g["batch"]) for t, g in groups.items()}
    sources["deform"] = fd.FixedGroupSource(dgroup["batch"])
    tr = Trainer(model, O.make_optimizer("adam", lr=1e-3, clipnorm=clipnorm), sources, groups["rigid"]["S"],
                 config(task_ratio=ratio, seed=seed, flip_prob=0.5, **cfg_kw))
    tr.fit(steps)
    deform = [[float(h[k]) for k in ("tmpl", "pdice", "smooth")] for h in tr.history if h["task"] == "deform"]
    out = {"seed": seed, "tasks": [h["task"] for h in tr.history],
           "loss": [float(h["loss"]) for h in tr.history], "deform_terms": deform,
           "variables": {GroupRegLevelModel._local_name(v): [float(np.sum(v.numpy().astype(np.float64))),
                                                             float(np.max(np.abs(v.numpy())))]
                         for v in model.variables}}
    return out, list(tr.history)


def _deform_golden_run():
    """The golden three-task run: `_three_task_run` with its defaults (clip 12, no extra config)."""
    return _three_task_run()[0]


class _OpDeterminism:
    """Op determinism on GPU for the duration of a block, restored afterwards.

    Without it the deform golden run is not reproducible: the gradients of the warp's gathers are
    GPU segment sums with nondeterministic accumulation order; Adam amplifies the ~1e-7 noise of
    step 2 to max |diff| ~100-300 over 8 steps (variable sums incl. BN statistics, measured on
    the GTX 1080). With determinism on, reruns are bit-identical."""

    def __enter__(self):
        from tensorflow.python.framework import config as tf_config
        self._cfg = tf_config
        self._was = tf_config.is_op_determinism_enabled()
        tf_config.enable_op_determinism()
        return self

    def __exit__(self, *exc):
        if not self._was:
            self._cfg.disable_op_determinism()
        return False


class TestDeformUnchanged(unittest.TestCase):
    """LNCC guard: three-task training with the LNCC weights at their defaults (0) is unchanged.
    The fixture was written on the milestone-7 code (commit b35bb54, before any LNCC change) with
    BGR_WRITE_GOLDEN=1, under `_OpDeterminism` (tolerance 0: reruns are bit-identical)."""

    def tearDown(self):
        free_memory()

    def test_matches_golden(self):
        with _OpDeterminism():
            got = _deform_golden_run()
        if os.environ.get("BGR_WRITE_GOLDEN") == "1":
            os.makedirs(os.path.dirname(DEFORM_GOLDEN_PATH), exist_ok=True)
            with open(DEFORM_GOLDEN_PATH, "w") as f:
                json.dump(dict(got, tolerance=0.0), f, indent=1, sort_keys=True)
            sys.stderr.write("\n[golden] wrote %s\n" % DEFORM_GOLDEN_PATH)
        with open(DEFORM_GOLDEN_PATH) as f:
            want = json.load(f)
        tol = float(want.get("tolerance", 0.0))
        self.assertEqual(got["seed"], want["seed"])
        self.assertEqual(got["tasks"], want["tasks"])
        self.assertEqual(set(got["tasks"]), {"rigid", "affine", "deform"})
        self.assertEqual(sorted(got["variables"]), sorted(want["variables"]))
        diffs = [abs(a - b) for a, b in zip(got["loss"], want["loss"])]
        for a, b in zip(got["deform_terms"], want["deform_terms"]):
            diffs += [abs(x - y) for x, y in zip(a, b)]
        for name, (s, m) in want["variables"].items():
            gs, gm = got["variables"][name]
            diffs += [abs(gs - s), abs(gm - m)]
        sys.stderr.write("\n[golden deform] max |diff| %.3g (tolerance %.3g)\n" % (max(diffs), tol))
        if tol == 0.0:
            self.assertEqual(got["loss"], want["loss"])
            self.assertEqual(got["deform_terms"], want["deform_terms"])
            self.assertEqual(got["variables"], want["variables"])
        else:
            self.assertLessEqual(max(diffs), tol)


# ============================================================ per-task gradient normalisation (v3.10)
class TestTaskGradNormHelpers(unittest.TestCase):
    """`task_grad_norm_scale` / `scale_gradients` on crafted values (eager)."""

    def scale(self, g, target=12.0, min_norm=1e-3):
        s, gated = task_grad_norm_scale(tf.constant(g, tf.float32), target, min_norm)
        return float(s.numpy()), float(gated.numpy())

    def test_scale_and_gate(self):
        self.assertEqual(self.scale(24.0), (0.5, 0.0))
        self.assertEqual(self.scale(5e-4), (1.0, 1.0))          # below the minimum: unscaled, gated
        self.assertEqual(self.scale(1e-3), (np.float32(12.0) * (np.float32(1.0) / np.float32(1e-3)), 0.0))
        for bad in (float("nan"), float("inf")):                # non-finite: never rescaled, not gated
            self.assertEqual(self.scale(bad), (1.0, 0.0), bad)

    def test_eps_zero_gradients(self):
        s, gated = self.scale(0.0, min_norm=0.0)
        self.assertTrue(np.isfinite(s))
        self.assertAlmostEqual(s, 12.0 / TASK_GRAD_NORM_EPS, delta=1e-6 * 12.0 / TASK_GRAD_NORM_EPS)
        self.assertEqual(gated, 0.0)
        out = scale_gradients([tf.zeros((3, 2))], tf.constant(s))
        self.assertTrue(np.all(out[0].numpy() == 0.0))
        self.assertEqual(self.scale(0.0), (1.0, 1.0))           # default minimum: gated

    def test_indexed_slices(self):
        g = tf.IndexedSlices(tf.constant([[1.0, 2.0], [3.0, 4.0]]), tf.constant([0, 2]), tf.constant([3, 2]))
        d = tf.constant([1.0, -2.0])
        out = scale_gradients([g, d], tf.constant(0.5))
        self.assertIsInstance(out[0], tf.IndexedSlices)
        np.testing.assert_array_equal(out[0].values.numpy(), [[0.5, 1.0], [1.5, 2.0]])
        np.testing.assert_array_equal(out[0].indices.numpy(), [0, 2])
        np.testing.assert_array_equal(out[1].numpy(), [0.5, -1.0])


class TestTaskGradNorm(DeformCase):
    """--task-grad-norm: normalise each task's raw gradient to the target (gated below the
    minimum), then multiply by the task weight; raw grad_norm logged, _eff = applied encoder norm."""

    RATIO = {"rigid": 1.0, "affine": 1.0, "deform": 2.0}

    def make3(self, like=None, clipnorm=None, opt=None, amp=False, logger=None, sources=None, **cfg_kw):
        """Three-task trainer on a narrow deform model; fresh models get random rigid and SVF head
        kernels N(0, 1e-3) so every task's gradient reaches the encoder."""
        model = GroupRegLevelModel(4, narrow_config(dropout=0.0, seed=1, deform_decoder=True)).build_model(N)
        if like is None:
            rng = np.random.default_rng(0)
            for head in (model.linear_decoder.rigid_head, model.deform_decoder.svf_head):
                head.kernel.assign(rng.normal(0, 1e-3, head.kernel.shape).astype(np.float32))
        else:
            model.set_weights(like.get_weights())
        if sources is None:
            sources = {t: fd.FixedGroupSource(g["batch"]) for t, g in self.groups.items()}
            sources["deform"] = fd.FixedGroupSource(self.dgroup["batch"])
        cfg_kw.setdefault("task_ratio", dict(self.RATIO))
        optimizer = opt or O.make_optimizer("adam", lr=1e-3, clipnorm=clipnorm, amp=amp)
        return Trainer(model, optimizer, sources, self.groups["rigid"]["S"], config(amp=amp, **cfg_kw),
                       logger=logger)

    def batch(self, task):
        return self.dgroup["batch"] if task == "deform" else self.groups[task]["batch"]

    def test_off_no_new_keys(self):
        tr = self.make3()
        self.assertEqual(tr.task_grad_norm, 0.0)
        for t in ("rigid", "affine", "deform"):
            res = tr.train_step(t, self.batch(t))
            self.assertTrue(res["finite"])
            for k in TASK_GRAD_NORM_KEYS:
                self.assertNotIn(k, res)
        self.assertEqual(tr.trace_counts, {"rigid": 1, "affine": 1, "deform": 1})

    def test_linear_and_deform_normalised(self):
        target, lr = 5.0, 1e-2
        base = self.make3().model
        rel_err = {}
        for t in ("rigid", "affine", "deform"):
            on = self.make3(like=base, task_grad_norm=target)
            off = self.make3(like=base)
            with _OpDeterminism():          # deform gathers: nondeterministic gradient sums otherwise
                r_on, r_off = on.train_step(t, self.batch(t)), off.train_step(t, self.batch(t))
            self.assertTrue(r_on["finite"] and r_off["finite"], t)
            self.assertNotAlmostEqual(r_on["grad_norm"], target, places=2)    # really rescaled
            np.testing.assert_allclose(r_on["grad_norm_normed"], target, rtol=1e-5, err_msg=t)
            np.testing.assert_allclose(r_on["grad_norm"] * r_on["grad_norm_scale"], target, rtol=1e-5, err_msg=t)
            self.assertEqual(r_on["grad_norm_gated_frac"], 0.0)
            np.testing.assert_allclose(r_on["grad_norm"], r_off["grad_norm"], rtol=1e-6, err_msg=t)
            for k in (DEFORM_LOSS_KEYS if t == "deform" else ("loss", "center", "pair")):
                np.testing.assert_allclose(r_on[k], r_off[k], rtol=1e-6, err_msg=k)
            # Independent check of the applied update: plain SGD moves the variables by lr * g.
            sgd = self.make3(like=base, task_grad_norm=target,
                             opt=O.make_optimizer("sgd", lr=lr, momentum=0.0, nesterov=False, clipnorm=None))
            tv = sgd.model.trainable_variables
            before = _vals(tv)
            self.assertTrue(sgd.train_step(t, self.batch(t))["finite"])
            step_norm = np.sqrt(sum(np.sum((b.astype(np.float64) - v.numpy()) ** 2) for b, v in zip(before, tv)))
            rel_err[t] = abs(step_norm / lr - target) / target
            self.assertLess(rel_err[t], 1e-4, t)
        sys.stderr.write("\n[grad-norm] SGD |delta| / lr vs target rel. error %s\n"
                         % ", ".join("%s %.2g" % kv for kv in sorted(rel_err.items())))

    def test_gate_below_min(self):
        base = self.make3().model
        with _OpDeterminism():
            on = self.make3(like=base, task_grad_norm=12.0, task_grad_norm_min=1e9)
            off = self.make3(like=base)
            for t in ("rigid", "deform"):
                r_on, r_off = on.train_step(t, self.batch(t)), off.train_step(t, self.batch(t))
                self.assertEqual((r_on["grad_norm_scale"], r_on["grad_norm_gated_frac"]), (1.0, 1.0), t)
                self.assertEqual(r_on["grad_norm_normed"], r_on["grad_norm"], t)
                self.assertEqual(r_on["grad_norm"], r_off["grad_norm"], t)
                self.assertEqual(r_on["diag_enc_grad_norm_eff"], r_off["diag_enc_grad_norm_eff"], t)
                for a, b in zip(on.model.trainable_variables, off.model.trainable_variables):
                    self.assertTrue(np.array_equal(a.numpy(), b.numpy()), (t, a.name))

    def test_eps_zero_grads(self):
        # Identical images, zero head kernels: GroupCentering cancels everything -> (near) zero
        # gradient. With min 0 the eps path keeps the step finite; with the default min it is gated.
        g = self.groups["rigid"]
        rep = lambda a: np.repeat(a[:1], N, axis=0)  # noqa: E731
        batch = fd.to_dali_batch(rep(g["images"]), rep(g["T"]), rep(g["T_target"]), "rigid", 4)
        src = {"rigid": fd.FixedGroupSource(batch)}
        for min_norm in (0.0, 1e-3):
            tr = self.make(config(task_grad_norm=12.0, task_grad_norm_min=min_norm), sources=src)
            res = tr.train_step("rigid", batch)
            self.assertTrue(res["finite"], min_norm)
            self.assertLess(res["grad_norm"], 1e-3)
            for k in TASK_GRAD_NORM_KEYS:
                self.assertTrue(np.isfinite(res[k]), k)
            for v in tr.model.variables:
                self.assertTrue(np.all(np.isfinite(v.numpy())), v.name)
            if min_norm == 0.0:
                self.assertEqual(res["grad_norm_gated_frac"], 0.0)
                if res["grad_norm"] == 0.0:
                    # The global norm underflows to 0 (squares of ~1e-20 residues), so the eps path
                    # applies 12 / eps = 1.2e13: finite, but the residues are amplified to a
                    # normed value ~1e-6 (measured 3.9e-6) -> the reason for the default minimum.
                    np.testing.assert_allclose(res["grad_norm_scale"], 12.0 / TASK_GRAD_NORM_EPS, rtol=1e-6)
                    self.assertLess(res["grad_norm_normed"], 1e-3)
            else:
                self.assertEqual((res["grad_norm_scale"], res["grad_norm_gated_frac"]), (1.0, 1.0))
                self.assertEqual(res["grad_norm_normed"], res["grad_norm"])
            sys.stderr.write("\n[grad-norm eps] min %g: raw grad_norm %.3g, scale %.3g, normed %.3g\n"
                             % (min_norm, res["grad_norm"], res["grad_norm_scale"], res["grad_norm_normed"]))

    def test_weights_after_normalisation(self):
        target = 4.0
        w = {"deform": 0.25, "rigid": 2.0}
        base = self.make3().model
        for t in ("rigid", "deform"):        # fresh twins per task: both start from `base`
            on = self.make3(like=base, task_grad_norm=target, task_loss_weights=w)
            off = self.make3(like=base)
            with _OpDeterminism():          # deform gathers: nondeterministic gradient sums otherwise
                r_on, r_off = on.train_step(t, self.batch(t)), off.train_step(t, self.batch(t))
            np.testing.assert_allclose(r_on["grad_norm_normed"], target * w[t], rtol=1e-5, err_msg=t)
            np.testing.assert_allclose(r_on["grad_norm"], r_off["grad_norm"], rtol=1e-6, err_msg=t)  # w not in it
            np.testing.assert_allclose(r_on["diag_enc_grad_norm_raw"], r_off["diag_enc_grad_norm_raw"],
                                       rtol=1e-6, err_msg=t)
            self.assertGreater(r_on["diag_enc_grad_norm_raw"], 0.0)
            np.testing.assert_allclose(r_on["diag_enc_grad_norm_eff"],
                                       r_on["diag_enc_grad_norm_raw"] * r_on["grad_norm_scale"] * w[t],
                                       rtol=1e-5, err_msg=t)
        # Global clip 1 below the applied norm 8: _eff includes the clip factor 1 / normed.
        clipped = self.make3(like=base, clipnorm=1.0, task_grad_norm=target, task_loss_weights=w)
        rc = clipped.train_step("rigid", self.batch("rigid"))
        np.testing.assert_allclose(rc["grad_norm_normed"], 8.0, rtol=1e-5)
        np.testing.assert_allclose(rc["diag_enc_grad_norm_eff"],
                                   rc["diag_enc_grad_norm_raw"] * rc["grad_norm_scale"] * 2.0 / rc["grad_norm_normed"],
                                   rtol=1e-4)

    def test_nonfinite_unchanged(self):
        # NaN pixel: the images are zeroed (finite gradient) and the step is skipped as before.
        imgs = self.groups["rigid"]["images"].copy()
        imgs[0, 3, 4, 5, 0] = np.nan
        g = self.groups["rigid"]
        nan_batch = fd.to_dali_batch(imgs, g["T"], g["T_target"], "rigid", 4)
        tr = self.make3(task_grad_norm=12.0)
        self.assertTrue(tr.train_step("rigid", self.batch("rigid"))["finite"])
        model, opt = tr.model, tr.optimizer
        before = (_vals(model.trainable_variables), _vals(model.non_trainable_variables), _vals(opt.variables()))
        it = int(opt.iterations.numpy())
        res = tr.train_step("rigid", nan_batch)
        self.assertFalse(res["finite"])
        self.assertTrue(np.isfinite(res["grad_norm_scale"]))
        self.assertEqual(int(opt.iterations.numpy()), it)
        self.assertTrue(_all_equal(before[0], model.trainable_variables))
        self.assertTrue(_all_equal(before[1], model.non_trainable_variables))
        self.assertTrue(_all_equal(before[2], opt.variables()))

    def test_amp_normalised_and_overflow(self):
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            tr = self.make3(amp=True, task_grad_norm=12.0, task_ratio={"rigid": 1.0, "deform": 1.0})
            self.assertIsInstance(tr.optimizer, tf.keras.mixed_precision.LossScaleOptimizer)
            n_ok = {"rigid": 0, "deform": 0}
            for _ in range(20):
                for t in n_ok:
                    res = tr.train_step(t, self.batch(t))
                    if res["finite"]:
                        n_ok[t] += 1
                        np.testing.assert_allclose(res["grad_norm_normed"], 12.0, rtol=1e-3, err_msg=t)
                if min(n_ok.values()) >= 2:
                    break
            self.assertGreaterEqual(min(n_ok.values()), 2, n_ok)
            # Forced float16 overflow (initial loss scale 2^40): inf gradients are never rescaled
            # (scale 1, not gated), the LSO skips the update and halves its scale.
            inner = O.make_optimizer("adam", lr=1e-3, clipnorm=None)
            lso = tf.keras.mixed_precision.LossScaleOptimizer(inner, initial_scale=2.0 ** 40)
            tr2 = self.make3(opt=lso, amp=True, task_grad_norm=12.0, task_ratio={"rigid": 1.0})
            w0 = _vals(tr2.model.trainable_variables)
            res = tr2.train_step("rigid", self.batch("rigid"))
            self.assertFalse(res["finite"])
            self.assertEqual((res["grad_norm_scale"], res["grad_norm_gated_frac"]), (1.0, 0.0))
            self.assertEqual(float(lso.loss_scale.numpy()), 2.0 ** 39)
            self.assertTrue(_all_equal(w0, tr2.model.trainable_variables))
        finally:
            tf.keras.mixed_precision.set_global_policy("float32")

    def test_trainer_validation(self):
        for bad in (dict(task_grad_norm=-1.0), dict(task_grad_norm=float("nan")),
                    dict(task_grad_norm=float("inf")), dict(task_grad_norm_min=-1.0),
                    dict(task_grad_norm_min=float("nan"))):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.make(config(**bad))

    def test_window_logs(self):
        seed = _seed_with_all_tasks(self.RATIO, 8)
        recs = {}
        for name, kw in (("on", dict(task_grad_norm=12.0)), ("off", {})):
            paths = make_run_dir(self.tmp, "gn_" + name)
            logger = RunLogger(paths, backend="jsonl", tensorboard=False, quiet=True)
            tr = self.make3(logger=logger, seed=seed, log_every=1, **kw)
            tr.fit(8)
            logger.close()
            self.assertTrue(all(h["finite"] for h in tr.history))
            self.assertEqual(tr.trace_counts, {"rigid": 1, "affine": 1, "deform": 1})
            with open(logger.metrics_path) as f:
                recs[name] = [json.loads(line)["data"] for line in f]
        for t in self.RATIO:
            pre = "train_%s/" % t
            got = [r for r in recs["on"] if pre + "grad_norm" in r]
            self.assertTrue(got, t)
            for r in got:
                self.assertTrue(np.isfinite(r[pre + "grad_norm_scale"]))
                self.assertGreaterEqual(r[pre + "grad_norm_gated_frac"], 0.0)
                self.assertLessEqual(r[pre + "grad_norm_gated_frac"], 1.0)
                np.testing.assert_allclose(r[pre + "grad_norm_normed"], 12.0, rtol=1e-5)
                self.assertEqual(r["train/grad_norm"], r[pre + "grad_norm"])      # raw, not normalised
        for r in recs["off"]:
            for k in r:
                self.assertFalse(any(k.endswith("/" + key) for key in TASK_GRAD_NORM_KEYS), k)


# ============================================================ non-finite guard (nonfinite-fixes)
def _find_seed(ratio, pred, n=64):
    """A schedule seed whose first n draws (0-based step indices) satisfy pred(draws)."""
    for seed in range(5000):
        s = TaskSchedule(ratio, seed)
        draws = [s.draw(i) for i in range(n)]
        if pred(draws):
            return seed, draws
    raise AssertionError("no seed found")


def _positions(draws, task):
    return [i for i, t in enumerate(draws) if t == task]


def _open_streaks(history):
    """{step: largest per-task non-finite streak open after that step} from Trainer.history."""
    cur, out = {}, {}
    for h in history:
        cur[h["task"]] = 0 if h["finite"] else cur.get(h["task"], 0) + 1
        out[h["step"]] = max(cur.values())
    return out


class _SwitchSource:
    """Yields `bad` for the first n_bad calls, then `good` (iterator, like FixedGroupSource)."""

    def __init__(self, bad, good, n_bad):
        self.bad, self.good, self.n_bad, self.calls = tuple(bad), tuple(good), int(n_bad), 0

    def __iter__(self):
        return self

    def __next__(self):
        self.calls += 1
        return self.bad if self.calls <= self.n_bad else self.good


class _KernelSwitchSource:
    """Always yields `batch`; before call i (1-based) sets `kernel` to `bad` if i in bad_calls,
    else to `good` (a forward-NaN deform decoder for exactly those steps, within one fit)."""

    def __init__(self, batch, kernel, good, bad, bad_calls):
        self.batch, self.kernel, self.good, self.bad = tuple(batch), kernel, good, bad
        self.bad_calls, self.calls = set(bad_calls), 0

    def __iter__(self):
        return self

    def __next__(self):
        self.calls += 1
        self.kernel.assign(self.bad if self.calls in self.bad_calls else self.good)
        return self.batch


class TestNonfiniteGuard(DeformCase):
    """nonfinite-fixes D1-D6: per-task streak guard, ckpt/last protection, LSO skip, window logs."""

    RATIO = {"rigid": 1.0, "deform": 1.0}

    def nan_deform_batch(self):
        imgs = self.dgroup["images"].copy()
        imgs[0, 3, 4, 5, 0] = np.nan
        return fd.to_deform_batch(imgs, self.dgroup["labels"], level=4)

    def sources(self, deform_source):
        return {"rigid": fd.FixedGroupSource(self.groups["rigid"]["batch"]), "deform": deform_source}

    def logger(self, name):
        return RunLogger(make_run_dir(self.tmp, name), backend="jsonl", tensorboard=False, quiet=True)

    @staticmethod
    def records(logger):
        with open(logger.metrics_path) as f:
            return [json.loads(line) for line in f]

    def interleaved_seed(self, n_deform=4):
        """Seed with a finite-task (rigid) step between the 1st and the n_deform-th deform step."""
        def pred(d):
            p = _positions(d, "deform")
            return len(p) >= n_deform and "rigid" in d[p[0]:p[n_deform - 1]]
        return _find_seed(self.RATIO, pred)

    def attach_checkpoint(self, tr, name):
        paths = make_run_dir(self.tmp, name)
        tr.checkpoint = C.RunCheckpoint(paths["ckpt_last"], paths["ckpt_best"], tr.model, tr.optimizer,
                                        tr.step, extra=tr.checkpoint_variables())
        saves, orig = [], tr.checkpoint.save_last

        def spy():
            saves.append(int(tr.step.numpy()))
            return orig()

        tr.checkpoint.save_last = spy
        return saves

    def test_per_task_guard_trips_in_mixed_run(self):
        seed, draws = self.interleaved_seed()
        dpos = _positions(draws, "deform")[:4]
        first, last = dpos[0] + 1, dpos[3] + 1
        logger = self.logger("trip")
        tr = self.make_deform(config(task_ratio=dict(self.RATIO), seed=seed, nonfinite_patience_task=3),
                              sources=self.sources(fd.FixedGroupSource(self.nan_deform_batch())), logger=logger)
        self.assertEqual(tr.cfg.nonfinite_patience, 20)
        with self.assertRaises(RuntimeError) as cm:
            tr.fit(60)
        logger.close()
        msg = str(cm.exception)
        for part in ("task deform", "4 consecutive non-finite steps of this task", "first at step %d" % first,
                     "last at step %d" % last, "finite steps of other tasks do not reset"):
            self.assertIn(part, msg)
        self.assertEqual(tr.history[-1]["step"], last)
        self.assertTrue(all(not h["finite"] for h in tr.history if h["task"] == "deform"))
        # finite rigid steps inside the streak: they reset the old global counter
        self.assertTrue(any(h["task"] == "rigid" and h["finite"] and first < h["step"] < last for h in tr.history))
        events = [r["data"] for r in self.records(logger) if "event" in r["data"]]
        abort = [e for e in events if e["event"] == "nonfinite_abort"]
        self.assertEqual(len(abort), 1)
        self.assertEqual((abort[0]["task"], abort[0]["streak"], abort[0]["first_step"], abort[0]["last_step"]),
                         ("deform", 4, first, last))
        skipped = [e for e in events if e["event"] == "nonfinite_step_skipped"]
        self.assertEqual([e["task_streak"] for e in skipped], [1, 2, 3, 4])
        self.assertTrue(all("lso_skipped" not in e for e in skipped))           # no AMP
        free_memory()
        tr2 = self.make_deform(config(task_ratio=dict(self.RATIO), seed=seed, nonfinite_patience_task=0),
                               sources=self.sources(fd.FixedGroupSource(self.nan_deform_batch())))
        tr2.fit(30)                                                              # 0 = off: no raise
        self.assertGreater(sum(1 for h in tr2.history if h["task"] == "deform"), 4)

    def test_validation_of_patience(self):
        for bad in (-1, 1.5, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.make_deform(config(task_ratio=dict(self.RATIO), nonfinite_patience_task=bad))

    def test_ckpt_last_not_written_during_streak(self):
        ckpt_every, n_bad = 4, 4

        def pred(d):
            p = _positions(d, "deform")
            if len(p) < 5 or p[4] + 1 >= 22:
                return False
            blocked = range(p[1] + 1, p[4] + 1)                 # done values with an open streak >= 2
            return any(s % ckpt_every == 0 for s in blocked) and (p[4] + 1) % ckpt_every != 0
        seed, draws = _find_seed(self.RATIO, pred)
        logger = self.logger("protect")
        src = self.sources(_SwitchSource(self.nan_deform_batch(), self.dgroup["batch"], n_bad))
        tr = self.make_deform(config(task_ratio=dict(self.RATIO), seed=seed, ckpt_every=ckpt_every,
                                     nonfinite_patience_task=6), sources=src, logger=logger)
        saves = self.attach_checkpoint(tr, "protect_ckpt")
        tr.fit(24)
        logger.close()
        streaks = _open_streaks(tr.history)
        blocked = {s for s, v in streaks.items() if v >= 2}
        self.assertTrue(blocked)
        self.assertFalse(set(saves) & blocked, (saves, sorted(blocked)))
        deferred = [r for r in self.records(logger) if r["data"].get("event") == "ckpt_last_deferred"]
        self.assertTrue(deferred)
        for r in deferred:
            self.assertEqual(r["step"] % ckpt_every, 0)
            self.assertEqual(r["data"]["streaks"], {"deform": streaks[r["step"]]})
            after = min(s for s in streaks if s > r["step"] and s not in blocked)
            self.assertIn(after, saves)                          # the deferred save, off the cadence
        self.assertTrue(any(s % ckpt_every for s in saves))
        regular = [s for s in streaks if s % ckpt_every == 0 and s not in blocked]
        self.assertTrue(set(regular) <= set(saves))              # finite cadence unchanged
        free_memory()
        # Raising variant: the last ckpt/last on disk predates the 2nd failing deform step.
        seed, draws = self.interleaved_seed()
        dpos = _positions(draws, "deform")
        tr = self.make_deform(config(task_ratio=dict(self.RATIO), seed=seed, ckpt_every=1, nonfinite_patience_task=3),
                              sources=self.sources(fd.FixedGroupSource(self.nan_deform_batch())))
        saves = self.attach_checkpoint(tr, "protect_raise")
        with self.assertRaises(RuntimeError):
            tr.fit(60)
        self.assertIn(dpos[0] + 1, saves)                        # a single failure does not block
        self.assertLess(max(saves), dpos[1] + 1)
        self.assertTrue(tr.checkpoint.latest_last.endswith("last-%d" % max(saves)), tr.checkpoint.latest_last)

    def amp_model(self):
        model = GroupRegLevelModel(4, narrow_config(dropout=0.0, seed=1, deform_decoder=True)).build_model(N)
        _random_head_kernel(model)
        return model

    def test_lso_not_halved_by_streaking_task(self):
        def pred(d):
            p = _positions(d, "deform")
            return len(p) >= 6 and p[5] < 20 and "rigid" in d[p[0]:p[4]]
        seed, draws = _find_seed(self.RATIO, pred)
        dpos = _positions(draws, "deform")
        n_steps = dpos[5] + 1
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            model = self.amp_model()
            inner = O.make_optimizer("adam", lr=1e-3, clipnorm=None)
            opt = tf.keras.mixed_precision.LossScaleOptimizer(inner, initial_scale=2.0 ** 4)
            kern = model.deform_decoder.d1.conv.kernel
            good = kern.numpy().copy()
            bad = good.copy()
            bad.flat[0] = np.nan
            src = self.sources(_KernelSwitchSource(self.dgroup["batch"], kern, good, bad, range(1, 6)))
            logger = self.logger("lso")
            tr = Trainer(model, opt, src, self.groups["rigid"]["S"],
                         config(task_ratio=dict(self.RATIO), seed=seed, amp=True, nonfinite_patience_task=50),
                         logger=logger)
            head0 = model.linear_decoder.rigid_head.kernel.numpy().copy()
            tr.fit(n_steps)
            logger.close()
            dh = [h for h in tr.history if h["task"] == "deform"]
            rh = [h for h in tr.history if h["task"] == "rigid"]
            self.assertEqual([h["finite"] for h in dh], [False] * 5 + [True])
            self.assertEqual([h["lso_skipped"] for h in dh], [False, True, True, True, True, False])
            self.assertTrue(all(not np.isfinite(h["loss"]) for h in dh[:5]))
            self.assertTrue(rh and all(h["finite"] and not h["lso_skipped"] for h in rh))
            self.assertFalse(np.array_equal(head0, model.linear_decoder.rigid_head.kernel.numpy()))
            self.assertEqual(float(opt.loss_scale.numpy()), 2.0 ** 3)        # halved once, not 5 times
            # LSO calls: every rigid step, the first deform failure and the finite deform step
            self.assertEqual(int(inner.iterations.numpy()), len(rh) + 2)
            self.assertEqual(tr.trace_counts, {"rigid": 1, "deform": 1})
            recs = [r["data"] for r in self.records(logger) if LOSS_SCALE_KEY in r["data"]]
            self.assertEqual(len(recs), n_steps)
            self.assertEqual(recs[-1][LOSS_SCALE_KEY], float(opt.loss_scale.numpy()))
            # Control: the same forward-NaN step without the flag halves every time; with it, never.
            kern.assign(bad)
            scale = float(opt.loss_scale.numpy())
            for i in range(3):
                res = tr.train_step("deform", self.dgroup["batch"], lso_skip=False)
                self.assertFalse(res["finite"] or res["lso_skipped"])
                self.assertEqual(float(opt.loss_scale.numpy()), scale / 2.0 ** (i + 1))
            scale = float(opt.loss_scale.numpy())
            res = tr.train_step("deform", self.dgroup["batch"], lso_skip=True)
            self.assertTrue(res["lso_skipped"])
            self.assertEqual(float(opt.loss_scale.numpy()), scale)
            self.assertEqual(tr.trace_counts, {"rigid": 1, "deform": 1})
            self.assertNotIn("lso_skip_deform", [v.name.split(":")[0] for v in model.variables])
        finally:
            tf.keras.mixed_precision.set_global_policy("float32")

    def test_gradient_overflow_and_isolated_nan_unchanged(self):
        """Gradient-only overflows (finite loss) keep halving exactly as with plain train_step
        calls; an isolated forward-NaN step halves once, as today."""
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            with _OpDeterminism():
                models, opts, trs = [], [], []
                for i in range(2):
                    m = GroupRegLevelModel(4, narrow_config(dropout=0.0, seed=1)).build_model(N)
                    _random_head_kernel(m)
                    if models:
                        m.set_weights(models[0].get_weights())
                    o = tf.keras.mixed_precision.LossScaleOptimizer(O.make_optimizer("adam", lr=1e-3),
                                                                    initial_scale=2.0 ** 40)
                    logger = self.logger("ovf%d" % i) if i == 0 else None
                    trs.append(Trainer(m, o, {"rigid": fd.FixedGroupSource(self.groups["rigid"]["batch"])},
                                       self.groups["rigid"]["S"],
                                       config(amp=True, nonfinite_patience=60, nonfinite_patience_task=60,
                                              ckpt_every=1),
                                       logger=logger))
                    models.append(m)
                    opts.append(o)
                k = 45
                saves = self.attach_checkpoint(trs[0], "ovf_ckpt")
                trs[0].fit(k)
                # warm-up overflows (finite loss) change no weights: they never block ckpt/last
                self.assertEqual(saves, list(range(1, k + 1)))
                trs[0].logger.close()
                scales_fit = [r["data"][LOSS_SCALE_KEY] for r in self.records(trs[0].logger)
                              if LOSS_SCALE_KEY in r["data"]]
                scales_step, finite_step = [], []
                for _ in range(k):
                    res = trs[1].train_step("rigid", self.groups["rigid"]["batch"])
                    scales_step.append(float(opts[1].loss_scale.numpy()))
                    finite_step.append(res["finite"])
                self.assertGreater(finite_step.count(False), 1)              # consecutive overflows
                self.assertTrue(all(np.isfinite(h["loss"]) for h in trs[0].history))
                self.assertEqual([h["finite"] for h in trs[0].history], finite_step)
                self.assertFalse(any(h["lso_skipped"] for h in trs[0].history))
                self.assertEqual(scales_fit, scales_step)
                self.assertEqual(int(O.inner_optimizer(opts[0]).iterations.numpy()),
                                 int(O.inner_optimizer(opts[1]).iterations.numpy()))
                for a, b in zip(models[0].variables, models[1].variables):
                    self.assertTrue(np.array_equal(a.numpy(), b.numpy()), a.name)
            free_memory()
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            # Isolated forward NaN in a mixed run: reaches the LSO (one halving), next step fine.
            def pred(d):
                p = _positions(d, "deform")
                return len(p) >= 2 and p[1] < 16
            seed, draws = _find_seed(self.RATIO, pred)
            dpos = _positions(draws, "deform")
            model = self.amp_model()
            inner = O.make_optimizer("adam", lr=1e-3, clipnorm=None)
            opt = tf.keras.mixed_precision.LossScaleOptimizer(inner, initial_scale=2.0 ** 4)
            kern = model.deform_decoder.d1.conv.kernel
            good = kern.numpy().copy()
            bad = good.copy()
            bad.flat[0] = np.nan
            src = self.sources(_KernelSwitchSource(self.dgroup["batch"], kern, good, bad, [1]))
            tr = Trainer(model, opt, src, self.groups["rigid"]["S"],
                         config(task_ratio=dict(self.RATIO), seed=seed, amp=True))
            tr.fit(dpos[1] + 1)
            dh = [h for h in tr.history if h["task"] == "deform"]
            self.assertEqual([(h["finite"], h["lso_skipped"]) for h in dh], [(False, False), (True, False)])
            self.assertEqual(float(opt.loss_scale.numpy()), 2.0 ** 3)
        finally:
            tf.keras.mixed_precision.set_global_policy("float32")

    def test_window_log_keys_and_warning(self):
        seed, draws = self.interleaved_seed()
        logger = self.logger("win")
        src = self.sources(_SwitchSource(self.nan_deform_batch(), self.dgroup["batch"], 3))
        tr = self.make_deform(config(task_ratio=dict(self.RATIO), seed=seed, log_every=4), sources=src,
                              logger=logger)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            tr.fit(16)
        logger.close()
        recs = [r for r in self.records(logger) if "lr" in r["data"]]
        self.assertEqual([r["step"] for r in recs], [4, 8, 12, 16])
        warned = []
        for r in recs:
            window = [h for h in tr.history if r["step"] - 4 < h["step"] <= r["step"]]
            for t in ("rigid", "deform"):
                hs = [h for h in window if h["task"] == t]
                keys = ["train_%s/%s" % (t, k) for k in NONFINITE_WINDOW_KEYS]
                if not hs:
                    for k in keys:
                        self.assertNotIn(k, r["data"])
                    continue
                bad = sum(1 for h in hs if not h["finite"])
                self.assertEqual([r["data"][k] for k in keys], [len(hs), bad, bad / float(len(hs))])
                if bad / float(len(hs)) > 0.1:
                    warned.append("WARNING step %d: task %s non-finite in %d of %d steps" % (r["step"], t, bad, len(hs)))
            self.assertNotIn(LOSS_SCALE_KEY, r["data"])                       # no AMP
        self.assertTrue(warned)
        for w in warned:
            self.assertIn(w, err.getvalue())
        self.assertEqual(err.getvalue().count("non-finite in"), len(warned))
        free_memory()
        logger = self.logger("win_ok")
        tr = self.make_deform(config(task_ratio=dict(self.RATIO), seed=seed, log_every=4), logger=logger)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            tr.fit(8)
        logger.close()
        recs = [r["data"] for r in self.records(logger) if "lr" in r["data"]]
        for r in recs:
            for t in ("rigid", "deform"):
                if "train_%s/window_steps" % t in r:
                    self.assertEqual((r["train_%s/nonfinite_steps" % t], r["train_%s/nonfinite_frac" % t]), (0, 0.0))
        self.assertNotIn("non-finite in", err.getvalue())


class TestTaskGradNormClipEquivalence(unittest.TestCase):
    """KEY check: --task-grad-norm 12 without weights equals the clip-12 recipe with weights 2^14
    (every step's raw norm > 12, so every step is clipped). Powers of two scale every float value
    exactly, and `task_grad_norm_scale` computes target * (1.0 / norm) like
    `tf.clip_by_global_norm`, so with the optimizer clip OFF (A0) the runs agree bit for bit;
    with the clip kept at 12 (A12, the real recipe) the extra clip of a norm-12 gradient is a
    no-op up to ~1 ulp per step (near-equality only, see the tolerance comment below)."""

    W = 2.0 ** 14

    def tearDown(self):
        free_memory()

    @staticmethod
    def max_diff(a, b):
        d = [abs(x - y) for x, y in zip(a["loss"], b["loss"])]
        for p, q in zip(a["deform_terms"], b["deform_terms"]):
            d += [abs(x - y) for x, y in zip(p, q)]
        for name, (s, m) in b["variables"].items():
            gs, gm = a["variables"][name]
            d += [abs(gs - s) / (1.0 + abs(s)), abs(gm - m) / (1.0 + abs(m))]
        return max(d)

    def test_equivalent_to_clipping_when_all_clipped(self):
        with _OpDeterminism():
            b, hb = _three_task_run(clipnorm=12.0,
                                    task_loss_weights={t: self.W for t in ("rigid", "affine", "deform")})
            a0, h0 = _three_task_run(clipnorm=None, task_grad_norm=12.0)
            a12, h12 = _three_task_run(clipnorm=12.0, task_grad_norm=12.0)
        self.assertEqual(set(b["tasks"]), {"rigid", "affine", "deform"})
        self.assertEqual(a0["tasks"], b["tasks"])
        self.assertTrue(all(h["finite"] for h in hb + h0 + h12))
        self.assertTrue(all(h["grad_norm"] > 12.0 for h in hb), [h["grad_norm"] for h in hb])
        self.assertTrue(all(h["grad_norm_gated_frac"] == 0.0 for h in h0 + h12))
        for ha, hw in zip(h0, hb):                         # raw norm of the unweighted objective
            np.testing.assert_allclose(ha["grad_norm"] * self.W, hw["grad_norm"], rtol=1e-6)
        d0, d12 = self.max_diff(a0, b), self.max_diff(a12, b)
        loss12 = max(abs(x - y) / abs(y) for x, y in zip(a12["loss"], b["loss"]))
        sys.stderr.write("\n[grad-norm equiv] A0 (clip off) vs B max |diff| %.3g, A12 (clip 12) vs B "
                         "max |diff| %.3g (variables relative to 1 + |value|), A12 loss max rel. diff "
                         "%.3g\n" % (d0, d12, loss12))
        if d0 != 0.0:
            sys.stderr.write("[grad-norm equiv] A0 vs B not bit-identical\n")
        self.assertLessEqual(d0, 1e-5)
        # A12: the optimizer re-clips the normalised gradient whose norm is 12 +- 1 ulp; when it is
        # above 12 the clip factor is 1 - 1 ulp. Measured on the GTX 1080: losses agree to ~1e-6
        # relative, but Adam's updates of near-zero-gradient elements (m / sqrt(v) of tiny,
        # cancelling moments) turn the ulp-level differences into variable-sum differences of
        # up to ~1e-3 relative after 8 steps (per-element max |v| ~2e-4).
        for h in h12:
            np.testing.assert_allclose(h["grad_norm_normed"], 12.0, rtol=1e-6)
        self.assertLessEqual(loss12, 1e-5)
        self.assertLessEqual(d12, 1e-2)


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
        # v3.9 LNCC metrics (always computed): identity model = identity baseline.
        for k in ("ncc_template", "ncc_pair", "ncc_template_identity", "ncc_pair_identity"):
            self.assertTrue(np.isfinite(m["val_deform/" + k]), k)
            self.assertGreater(m["val_deform/" + k], 0.0, k)
            self.assertLessEqual(m["val_deform/" + k], 1.0, k)
        self.assertAlmostEqual(m["val_deform/ncc_template"], m["val_deform/ncc_template_identity"], places=10)
        self.assertAlmostEqual(m["val_deform/ncc_pair"], m["val_deform/ncc_pair_identity"], places=10)
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
        for k in ("ncc_template_identity", "ncc_pair_identity"):
            self.assertEqual(m2["val_deform/" + k], m["val_deform/" + k], k)
        self.assertTrue(np.isfinite(m2["val_deform/ncc_template"]))
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
