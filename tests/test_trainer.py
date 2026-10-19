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

from models import transforms as tfm
from models.group_net import GroupRegLevelModel
from runtime import checkpoint as C
from runtime import fixed_data as fd
from runtime import optimizers as O
from runtime.logging import RunLogger, make_run_dir
from runtime.run import DIAG_KEYS, TaskSchedule, Trainer, TrainConfig, collapse_diagnostics
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


if __name__ == "__main__":
    unittest.main()
