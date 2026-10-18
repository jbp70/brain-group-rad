"""Trainer loop (runtime.run): task schedule, None-gradient filtering, NaN guard, trace counts,
rebuild_steps after a BN freeze, resume, validation (eval mode and the non-updating training-mode
diagnostics), a mixed-precision smoke test and a forced float16 overflow (loss-scale skip).

Narrow model, level 4, N = 4, synthetic phantom groups (tests.runtime_helpers).
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

from models.group_net import GroupRegLevelModel
from runtime import checkpoint as C
from runtime import fixed_data as fd
from runtime import optimizers as O
from runtime.logging import RunLogger, make_run_dir
from runtime.run import TaskSchedule, Trainer, TrainConfig
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

    def make(self, cfg=None, sources=None, val_sets=None, logger=None, opt="adam", amp=False, dropout=0.0):
        cfg = cfg or config()
        model = GroupRegLevelModel(4, narrow_config(dropout=dropout, seed=1)).build_model(N)
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
        with self.assertRaises(NotImplementedError):
            TaskSchedule({"rigid": 1, "deform": 2})


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


if __name__ == "__main__":
    unittest.main()
