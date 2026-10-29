"""``--gradient-checkpointing full`` with BN moving statistics updated exactly once per step
(`models.checkpointing.recompute`, `GroupRegLevelModel._call_deform`; milestone 9 design 3.2, 8).

(a) gradients under 'full' equal 'none' within 1e-4 relative, fp32 and AMP (level 4, narrow,
    N 3, randomised weights so the SVF head is non-zero; op determinism on and Grappler off:
    with Grappler on, its different rewrites of the two graphs alone gave 4.7e-4 under AMP on
    the GTX 1080, with it off the gradients are bitwise equal);
(b) the BN moving mean / variance after one 'full' Trainer step equal the 'none' step (<= 1e-7
    relative) while 'both' still shows the documented double update of the decoder statistics;
(b') two tape.gradient calls of a persistent tape under 'full' give one update;
(c) frozen statistics stay unchanged under 'full';
(d) recalibrate_bn under 'full' makes one update per layer and batch (the T7 pattern of
    tests/test_cascade.py, level-2 TINY cascade);
(e) one trace per task with 'full' over 6 steps of rigid:1,affine:1,deform:2;
(f) build_model under 'full' creates the same variables (names, shapes) as under 'none';
(g) retrain recipe: level-2 cascade train steps under 'full' after a recal with freeze False
    update the trainable BN once per step, the frozen stage never; freeze True: nothing moves.
"""
import sys
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests.runtime_helpers import free_memory, narrow_config
from tests.test_cascade import (CascadeTrainerCase, CountingSource, _GrapplerOff, _all_equal, _vals, bn_stats,
                                cascade_config, randomize)
from tests.test_trainer import (DeformCase, _OpDeterminism, _seed_with_all_tasks, config)
import numpy as np
import tensorflow as tf

from models import layers as L
from models.group_net import GroupRegLevelModel
from runtime import fixed_data as fd
from runtime import optimizers as O
from runtime.run import Trainer

N3 = 3


def rel(a, b):
    a = np.concatenate([np.ravel(np.asarray(x, np.float64)) for x in a])
    b = np.concatenate([np.ravel(np.asarray(x, np.float64)) for x in b])
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def local_names(model, variables=None):
    return [(GroupRegLevelModel._local_name(v), tuple(v.shape)) for v in (variables or model.weights)]


def stats_of(model, parts=None):
    layers = L.batch_norm_layers(parts if parts is not None else [model])
    return [v.numpy().copy() for m in layers for v in (m.bn.moving_mean, m.bn.moving_variance)]


def pair(mode_a, mode_b, seed=3, n=N3):
    """Two narrow deform models with identical randomised weights and the given modes."""
    cfg = narrow_config(dropout=0.0, seed=1, deform_decoder=True)
    a = GroupRegLevelModel(4, cfg, gradient_checkpointing=mode_a).build_model(n)
    randomize(a, seed, 0.02)
    b = GroupRegLevelModel(4, cfg, gradient_checkpointing=mode_b).build_model(n)
    b.copy_weights_from(a)
    return a, b


class TestFullGradients(unittest.TestCase):
    def tearDown(self):
        tf.keras.mixed_precision.set_global_policy("float32")
        free_memory()

    def grads(self, model, x, w):
        @tf.function
        def f():
            with tf.GradientTape() as tape:
                out = model(x, task="deform", training=True)
                loss = (tf.reduce_sum(out["u"] * w) + 0.5 * tf.reduce_sum(out["u_inv"] * w)
                        + 1e-3 * tf.reduce_sum(tf.cast(out["dec_feat"], tf.float32)))
            gs = tape.gradient(loss, model.trainable_variables)
            return [g for g in gs if g is not None]
        return [g.numpy() for g in f()]

    def run_case(self, amp):
        if amp:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        grp = fd.synthetic_deform_group(4, N3, seed=0)
        x = tf.constant(grp["images"])
        w = tf.constant(np.random.default_rng(0).normal(size=grp["images"].shape[:4] + (3,)).astype(np.float32))
        with _GrapplerOff(), _OpDeterminism():
            m_none, m_full = pair("none", "full")
            g_none = self.grads(m_none, x, w)
            g_full = self.grads(m_full, x, w)
        self.assertEqual(len(g_none), len(g_full))
        self.assertGreater(len(g_none), 10)
        e = rel(g_full, g_none)
        per = max(rel([a], [b]) for a, b in zip(g_full, g_none) if np.linalg.norm(b) > 0)
        print("\n[ckpt full] %s gradients vs none: rel %.2e (worst variable %.2e)"
              % ("AMP" if amp else "fp32", e, per), file=sys.stderr)
        self.assertLessEqual(e, 1e-4)

    def test_fp32(self):
        self.run_case(False)

    def test_amp(self):
        self.run_case(True)


class TestFullBNOnce(DeformCase):
    """(b), (b'), (c), (e) with Trainer / tf.function steps at level 4 (narrow, N 4)."""

    def trainer(self, mode, cfg=None, model=None):
        cfg = cfg or config(task_ratio={"deform": 1.0}, flip_prob=0.0)
        if model is None:
            model = GroupRegLevelModel(4, narrow_config(dropout=0.0, seed=1, deform_decoder=True),
                                       gradient_checkpointing=mode).build_model(4)
        sources = {t: fd.FixedGroupSource(g["batch"]) for t, g in self.groups.items()}
        sources["deform"] = fd.FixedGroupSource(self.dgroup["batch"])
        sources = {t: s for t, s in sources.items() if t in cfg.task_ratio}
        return Trainer(model, O.make_optimizer("adam", lr=1e-3), sources, self.groups["rigid"]["S"], cfg)

    def test_one_update_per_step(self):
        res = {}
        models = {}
        ref = None
        for mode in ("none", "full", "both"):
            m = GroupRegLevelModel(4, narrow_config(dropout=0.0, seed=1, deform_decoder=True),
                                   gradient_checkpointing=mode).build_model(4)
            if ref is None:
                randomize(m, 5, 0.02)
                ref = m
            else:
                m.copy_weights_from(ref)
            models[mode] = m
        before = stats_of(ref, [ref.encoder, ref.deform_decoder])   # the parts a deform step runs
        for mode, m in models.items():
            tr = self.trainer(mode, model=m)
            self.assertTrue(tr.train_step("deform", self.dgroup["batch"])["finite"])
            res[mode] = {"enc": stats_of(m, [m.encoder]), "dec": stats_of(m, [m.deform_decoder]),
                         "all": stats_of(m, [m.encoder, m.deform_decoder])}
        for a, b in zip(before, res["none"]["all"]):
            self.assertFalse(np.array_equal(a, b))                       # the step updated them
        e_full = rel(res["full"]["all"], res["none"]["all"])
        e_both_dec = rel(res["both"]["dec"], res["none"]["dec"])
        e_both_enc = rel(res["both"]["enc"], res["none"]["enc"])
        print("\n[ckpt full] BN statistics after one step vs none: full %.2e; both: encoder %.2e, decoder %.2e"
              % (e_full, e_both_enc, e_both_dec), file=sys.stderr)
        self.assertLessEqual(e_full, 1e-7)
        self.assertLessEqual(e_both_enc, 1e-7)
        self.assertGreater(e_both_dec, 1e-6)                             # documented double update

    def test_persistent_tape_two_gradients(self):
        m_none, m_full = pair("none", "full", n=4)
        x = tf.constant(self.dgroup["images"])

        @tf.function
        def two(model):
            with tf.GradientTape(persistent=True) as tape:
                out = model(x, task="deform", training=True)
                loss = tf.reduce_sum(out["u"] ** 2) + tf.reduce_sum(out["u_inv"])
            vs = model.trainable_variables
            g1 = [g for g in tape.gradient(loss, vs) if g is not None]
            g2 = [g for g in tape.gradient(loss, vs) if g is not None]
            return g1, g2

        @tf.function
        def fwd(model):
            return model(x, task="deform", training=True)["u"]

        fwd(m_none)
        g1, g2 = two(m_full)
        e = rel([s for s in stats_of(m_full)], stats_of(m_none))
        print("\n[ckpt full] persistent tape, 2 gradient calls: BN stats vs one forward %.2e" % e, file=sys.stderr)
        self.assertLessEqual(e, 1e-7)
        self.assertLessEqual(rel([g.numpy() for g in g2], [g.numpy() for g in g1]), 1e-6)

    def test_frozen_stats(self):
        tr = self.trainer("full")
        tr.model.set_bn_stats_frozen(True)
        tr.rebuild_steps()
        before = stats_of(tr.model)
        for _ in range(2):
            self.assertTrue(tr.train_step("deform", self.dgroup["batch"])["finite"])
        for a, b in zip(before, stats_of(tr.model)):
            self.assertTrue(np.array_equal(a, b))

    def test_one_trace_per_task(self):
        ratio = {"rigid": 1.0, "affine": 1.0, "deform": 2.0}
        seed = _seed_with_all_tasks(ratio, 6)
        tr = self.trainer("full", cfg=config(task_ratio=ratio, seed=seed))
        tr.fit(6)
        self.assertEqual(set(h["task"] for h in tr.history), set(ratio))
        self.assertTrue(all(h["finite"] for h in tr.history))
        self.assertEqual(tr.trace_counts, {"rigid": 1, "affine": 1, "deform": 1})


class TestFullBuild(unittest.TestCase):
    def tearDown(self):
        free_memory()

    def test_same_variables(self):
        cfg = narrow_config(dropout=0.0, seed=1, deform_decoder=True)
        a = GroupRegLevelModel(4, cfg, gradient_checkpointing="none").build_model(2)
        b = GroupRegLevelModel(4, cfg, gradient_checkpointing="full").build_model(2)
        self.assertEqual(local_names(a), local_names(b))
        self.assertEqual(local_names(a, a.trainable_variables), local_names(b, b.trainable_variables))
        with self.assertRaises(ValueError):
            GroupRegLevelModel(4, cfg, gradient_checkpointing="fully")


class TestFullRecal(CascadeTrainerCase):
    """(d): forward-only BN recalibration under 'full' makes one update per layer and batch."""

    def test_recalibrate_one_update(self):
        cfg = cascade_config(task_ratio={"deform": 1.0}, flip_prob=0.0)
        batches = [fd.to_deform_batch(self.dgroup["images"] * s, self.dgroup["labels"], level=2)
                   for s in (1.0, 1.5)]
        tr = self.make(cfg, sources={"deform": fd.FixedGroupSource(batches[0])}, gradient_checkpointing="full")
        self.assertEqual(tr.model.gradient_checkpointing, "full")
        parts = tr._bn_parts("deform")
        per = []
        for b in batches:
            tr.sources["deform"] = fd.FixedGroupSource(b)
            tr.recalibrate_bn(1, freeze=False, val=False)
            per.append(bn_stats(tr.model, parts))
        want = [np.mean([p[i] for p in per], axis=0) for i in range(len(per[0]))]
        counts = {}
        orig = L.BatchNormWithSeparateAffine.call

        def counting(layer, x, training=None):
            if training and not layer.stats_frozen:
                counts[id(layer)] = counts.get(id(layer), 0) + 1
            return orig(layer, x, training=training)

        L.BatchNormWithSeparateAffine.call = counting
        try:
            tr.sources["deform"] = CountingSource(batches)
            tr.recalibrate_bn(2, freeze=False, val=False)
        finally:
            L.BatchNormWithSeparateAffine.call = orig
        layers = Trainer._bn_layers(parts)
        self.assertEqual([counts.get(id(m), 0) for m in layers], [2] * len(layers))
        for g, w in zip(bn_stats(tr.model, parts), want):
            np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-5)


class TestFullCascadeTrainableBN(CascadeTrainerCase):
    """Retrain recipe (level 2, BN_RECAL_FREEZE false): after a step-0 recalibration with
    freeze=False, cascade TRAIN steps under 'full' update the trainable stage's BN statistics
    exactly once per step (reference: 'integrator', which never recomputes the encoder / decoder)
    and never touch the frozen level-4 stage; with freeze=True nothing moves after the recal.

    The recalibration runs on a deform batch with images x 1.5 and the train steps on the
    unscaled one, so a train step moves the recalibrated statistics by ~1 % (momentum 0.99): a
    second update in the 'full' recompute would double that and exceed the tolerance."""

    def tearDown(self):
        free_memory()
        super().tearDown()

    def _recal_sources(self):
        scaled = fd.to_deform_batch(self.dgroup["images"] * 1.5, self.dgroup["labels"], level=2)
        return {"rigid": fd.FixedGroupSource(self.batches["rigid"]),
                "affine": fd.FixedGroupSource(self.batches["affine"]),
                "deform": fd.FixedGroupSource(scaled)}

    def _recal_then_deform(self, mode):
        """(trainer, BN stats of the deform parts after the recal, after one deform step)."""
        cfg = cascade_config(flip_prob=0.0)
        tr = self.make(cfg, sources=self._recal_sources(), gradient_checkpointing=mode)
        self.assertEqual(tr.model.gradient_checkpointing, mode)
        tr.recalibrate_bn(4, freeze=False, val=False)
        self.assertFalse(any(m.stats_frozen for m in Trainer._bn_layers([tr.model])))
        parts = tr._bn_parts("deform")
        s0 = bn_stats(tr.model, parts)
        self.assertTrue(tr.train_step("deform", self.batches["deform"])["finite"])
        return tr, s0, bn_stats(tr.model, parts)

    def test_freeze_false_updates_once(self):
        ref, s0_ref, s1_ref = self._recal_then_deform("integrator")
        del ref
        free_memory()
        tr, s0, s1 = self._recal_then_deform("full")
        f0 = _vals(tr.cascade.model.weights)
        e_recal = rel(s0, s0_ref)
        e_full = rel(s1, s1_ref)
        e_moved = rel(s1, s0)
        print("\n[ckpt full, cascade L2, freeze False] recal full vs integrator %.2e; after one deform step: "
              "full vs integrator %.2e, step change %.2e" % (e_recal, e_full, e_moved), file=sys.stderr)
        self.assertLessEqual(e_recal, 1e-6)
        self.assertLessEqual(e_full, 1e-6)
        self.assertGreater(e_moved, 1e-4)
        for a, b in zip(s0, s1):
            self.assertFalse(np.array_equal(a, b))                       # every BN layer trained
        # Further steps keep training the trainable stage's BN; the frozen stage never changes.
        lin_parts = tr._bn_parts("rigid")
        before = bn_stats(tr.model, lin_parts)
        self.assertTrue(tr.train_step("rigid", self.batches["rigid"])["finite"])
        after = bn_stats(tr.model, lin_parts)
        self.assertTrue(all(not np.array_equal(a, b) for a, b in zip(before, after)))
        self.assertTrue(tr.train_step("deform", self.batches["deform"])["finite"])
        s2 = bn_stats(tr.model, tr._bn_parts("deform"))
        self.assertTrue(all(not np.array_equal(a, b) for a, b in zip(s1, s2)))
        self.assertTrue(_all_equal(f0, tr.cascade.model.weights))       # incl. frozen BN statistics

    def test_freeze_true_full(self):
        cfg = cascade_config(flip_prob=0.0)
        tr = self.make(cfg, sources=self._recal_sources(), gradient_checkpointing="full")
        tr.recalibrate_bn(4, freeze=True, val=False)
        self.assertTrue(all(m.stats_frozen for m in Trainer._bn_layers([tr.model])))
        s0 = bn_stats(tr.model)
        f0 = _vals(tr.cascade.model.weights)
        for t in ("rigid", "affine", "deform"):
            self.assertTrue(tr.train_step(t, self.batches[t])["finite"], t)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(s0, bn_stats(tr.model))))
        self.assertTrue(_all_equal(f0, tr.cascade.model.weights))
        self.assertEqual(tr.trace_counts, {"rigid": 1, "affine": 1, "deform": 1})


if __name__ == "__main__":
    unittest.main()
