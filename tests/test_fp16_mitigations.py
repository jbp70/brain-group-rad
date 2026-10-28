"""nonfinite-fixes a1 / a2 / a3 (`.agents/tasks/nonfinite-fixes/plan.md` D8-D10): opt-in float16
overflow mitigations of `GroupRegLevelModel` (--fp16-act-clip, --deform-decoder-dtype,
--bn-epsilon-override). The overflow case is the local repro of the m9 level-1 deform NaN
(`.agents/tasks/m9-level1-nan/repro_fp16_overflow.py`: tiny level-2 deform model, frozen BN, the d2
conv kernel x 1e5, SVF head N(0, 0.02))."""
import os
import shutil
import tempfile
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests.runtime_helpers import free_memory, narrow_config, synthetic_group, tiny_config
import numpy as np
import tensorflow as tf

from data_loading import params as P
from models import layers as L
from models.group_net import GroupRegLevelModel
from runtime import fixed_data as fd
from runtime import losses
from runtime import optimizers as O
from runtime.run import Trainer, TrainConfig
from runtime.schedulers import ScheduleConfig

MP = tf.keras.mixed_precision
REPRO_LEVEL, REPRO_N = 2, 3


def _nonfinite(t):
    return int(tf.reduce_sum(tf.cast(tf.logical_not(tf.math.is_finite(tf.cast(t, tf.float32))), tf.int32)).numpy())


class MixedPolicyCase(unittest.TestCase):
    """Every test runs under mixed_float16 (restored to float32 afterwards)."""

    def setUp(self):
        MP.set_global_policy("mixed_float16")

    def tearDown(self):
        MP.set_global_policy("float32")
        free_memory()

    @staticmethod
    def repro_data():
        rng = np.random.default_rng(0)
        shape = P.LEVEL_SHAPE[REPRO_LEVEL]
        imgs, labs = zip(*[fd.synthetic_label_phantom(shape, rng) for _ in range(REPRO_N)])
        return np.stack(imgs)[..., None].astype(np.float32), np.stack(labs).astype(np.int32)

    @staticmethod
    def repro_model(policy=None, **kw):
        """Tiny level-2 deform model with frozen BN; ``policy`` builds it under that global policy."""
        if policy is not None:
            MP.set_global_policy(policy)
        try:
            m = GroupRegLevelModel(REPRO_LEVEL, tiny_config(deform_decoder=True, seed=1, integration_steps=7),
                                   **kw).build_model(REPRO_N)
        finally:
            if policy is not None:
                MP.set_global_policy("mixed_float16")
        m.set_bn_stats_frozen(True)
        return m

    @staticmethod
    def overflow_weights(model, k=1e5):
        """SVF head N(0, 0.02) (so v != 0) and the d2 conv kernel scaled by k; returns the weights."""
        head = model.deform_decoder.svf_head
        head.kernel.assign(np.random.default_rng(1).normal(0, 0.02, head.kernel.shape).astype(np.float32))
        kern = model.deform_decoder.d2.conv.kernel
        kern.assign(kern.numpy() * k)
        return model.get_weights()

    @staticmethod
    def deform_terms(out, lab):
        onehot = tf.one_hot(lab, int(lab.max()) + 1, dtype=tf.float32)
        batch = {"onehot": onehot, "pair_i": tf.constant([0, 1, 2]), "pair_j": tf.constant([1, 2, 0])}
        return losses.deform_loss(out, batch)


class TestOffAddsNoOps(MixedPolicyCase):

    def graph_ops(self, model, n=3):
        f = tf.function(lambda x: model(x, task="deform", training=True))
        spec = tf.TensorSpec((n,) + tuple(P.LEVEL_SHAPE[4]) + (1,), tf.float32)
        return f.get_concrete_function(spec).graph.get_operations()

    def model(self, **kw):
        return GroupRegLevelModel(4, narrow_config(dropout=0.0, seed=1, deform_decoder=True), **kw).build_model(3)

    def test_defaults_vs_explicit_off(self):
        a = self.model()
        b = self.model(fp16_act_clip=0.0, fp16_act_clip_scope="all", deform_decoder_dtype="policy",
                       bn_epsilon_override=0.0)
        b.set_weights(a.get_weights())
        ops_a, ops_b = self.graph_ops(a), self.graph_ops(b)
        self.assertEqual([o.type for o in ops_a], [o.type for o in ops_b])
        # float16 under mixed_float16 is the policy dtype: same graph too
        c = self.model(deform_decoder_dtype="float16")
        self.assertEqual([o.type for o in self.graph_ops(c)], [o.type for o in ops_a])

    @staticmethod
    def n_clip(ops):
        """Clip ops inside the deform decoder (the sampler clips coordinates elsewhere)."""
        return len([o for o in ops if "clip_by_value" in o.name and "deform_decoder" in o.name])

    def test_clip_scope(self):
        base = self.n_clip(self.graph_ops(self.model()))
        self.assertEqual(base, 0)
        m = self.model(fp16_act_clip=30000.0)
        self.assertGreater(self.n_clip(self.graph_ops(m)), 0)
        dec = m.deform_decoder
        for name in ("d5", "d4", "d3", "d2", "d1"):
            self.assertEqual(getattr(dec, name).fp16_act_clip, 30000.0, name)
            self.assertEqual(getattr(dec, name).norm.fp16_act_clip, 30000.0, name)
        self.assertEqual(dec.svf_norm_act.norm.fp16_act_clip, 30000.0)
        enc = [x for x in [m.encoder] + list(m.encoder.submodules) + list(m.linear_decoder.submodules)
               if isinstance(x, (L.GroupConvBlock, L.BatchNormWithSeparateAffine))]
        self.assertTrue(enc)
        self.assertTrue(all(x.fp16_act_clip == 0.0 for x in enc))
        free_memory()
        m = self.model(fp16_act_clip=30000.0, fp16_act_clip_scope="all")
        self.assertEqual(m.encoder.l3.fp16_act_clip, 30000.0)
        self.assertEqual(m.encoder.l3.norm.fp16_act_clip, 30000.0)
        self.assertEqual(m.deform_decoder.d1.fp16_act_clip, 30000.0)

    def test_clip_is_noop_under_float32(self):
        MP.set_global_policy("float32")
        ref = [o.type for o in self.graph_ops(self.model())]
        m = self.model(fp16_act_clip=30000.0, fp16_act_clip_scope="all")
        self.assertEqual([o.type for o in self.graph_ops(m)], ref)


class TestFp16ActClip(MixedPolicyCase):
    """a1: the clip turns the repro overflow (d2 x 1e5) into finite fields and a finite loss."""

    def test_clip_prevents_repro_overflow(self):
        x, lab = self.repro_data()
        ref = self.repro_model()
        w = self.overflow_weights(ref)
        out = ref(x, task="deform", training=False)
        self.assertGreater(_nonfinite(out["v_raw"]), 0)               # precondition: the repro overflows
        self.assertGreater(_nonfinite(self.deform_terms(out, lab)["loss"]), 0)
        for split in (False, True):
            with self.subTest(split_group_conv=split):
                m = self.repro_model(fp16_act_clip=30000.0, split_group_conv=split)
                m.set_weights(w)
                out = m(x, task="deform", training=False)
                for k in ("v_raw", "v", "u", "u_inv"):
                    self.assertEqual(_nonfinite(out[k]), 0, k)
                self.assertEqual(_nonfinite(self.deform_terms(out, lab)["loss"]), 0)

    def test_clip_matches_unclipped_without_overflow(self):
        x, _ = self.repro_data()
        ref = self.repro_model()
        head = ref.deform_decoder.svf_head
        head.kernel.assign(np.random.default_rng(1).normal(0, 0.02, head.kernel.shape).astype(np.float32))
        m = self.repro_model(fp16_act_clip=30000.0)
        m.set_weights(ref.get_weights())
        a = ref(x, task="deform", training=False)["v_raw"].numpy()
        b = m(x, task="deform", training=False)["v_raw"].numpy()
        np.testing.assert_array_equal(a, b)              # no activation near the clip: identical values


class TestFloat32Decoder(MixedPolicyCase):
    """a2: --deform-decoder-dtype float32 under mixed_float16."""

    def test_dtypes_and_match_float32_model(self):
        x, _ = self.repro_data()
        m = self.repro_model(deform_decoder_dtype="float32")
        dec = m.deform_decoder
        for layer in [dec] + list(dec.submodules):
            if isinstance(layer, tf.keras.layers.Layer):
                self.assertEqual(layer.compute_dtype, "float32", layer.name)
        self.assertEqual(m.encoder.compute_dtype, "float16")
        self.assertEqual(m.encoder.l3.conv.compute_dtype, "float16")
        self.assertEqual(MP.global_policy().name, "mixed_float16")      # scope restored
        head = dec.svf_head
        head.kernel.assign(np.random.default_rng(1).normal(0, 0.02, head.kernel.shape).astype(np.float32))
        m32 = self.repro_model(policy="float32")
        m32.set_weights(m.get_weights())
        v = m(x, task="deform", training=False)["v_raw"].numpy()
        v32 = m32(x, task="deform", training=False)["v_raw"].numpy()
        self.assertTrue(np.all(np.isfinite(v)))
        tol = 2e-2 * np.abs(v32).max() + 1e-6
        self.assertLessEqual(np.abs(v - v32).max(), tol)
        # the repro overflow: finite in the float32 decoder
        for k_layer in (m, m32):
            kern = k_layer.deform_decoder.d2.conv.kernel
            kern.assign(kern.numpy() * 1e5)
        out = m(x, task="deform", training=False)
        for k in ("v_raw", "u", "u_inv"):
            self.assertEqual(_nonfinite(out[k]), 0, k)

    def test_amp_trainer_full_ckpt_lean_split(self):
        n = 4
        model = GroupRegLevelModel(4, narrow_config(dropout=0.0, seed=1, deform_decoder=True),
                                   gradient_checkpointing="full", lean_sampler=True, split_group_conv=True,
                                   deform_decoder_dtype="float32").build_model(n)
        self.assertEqual(model.deform_decoder.d1.conv.compute_dtype, "float32")
        head = model.deform_decoder.svf_head
        head.kernel.assign(np.random.default_rng(0).normal(0, 1e-3, head.kernel.shape).astype(np.float32))
        dgroup = fd.synthetic_deform_group(4, n, seed=0)
        opt = MP.LossScaleOptimizer(O.make_optimizer("adam", lr=1e-3), initial_scale=2.0 ** 8)
        cfg = TrainConfig(level=4, group_size=n, steps=100, val_every=0, log_every=0, task_ratio={"deform": 1.0},
                          schedule=ScheduleConfig("constant", base_lr=1e-3, total_steps=100, warmup_steps=0),
                          seed=0, amp=True, lean_sampler=True)
        tr = Trainer(model, opt, {"deform": fd.FixedGroupSource(dgroup["batch"])},
                     synthetic_group(4, n, "rigid", seed=0)["S"], cfg)
        before = [v.numpy().copy() for v in model.deform_decoder.trainable_variables]
        for _ in range(3):
            res = tr.train_step("deform", dgroup["batch"])
            self.assertTrue(res["finite"], res)
        self.assertEqual(tr.trace_counts, {"deform": 1})
        self.assertTrue(any(not np.array_equal(a, v.numpy())
                            for a, v in zip(before, model.deform_decoder.trainable_variables)))


class TestCheckpointCompatibility(MixedPolicyCase):

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="bgr_fp16_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def test_checkpoint_roundtrip_between_dtypes(self):
        a2 = self.repro_model(deform_decoder_dtype="float32")
        self.overflow_weights(a2, k=3.0)                                 # some non-init values
        default = self.repro_model()
        self.assertEqual([v.dtype for v in a2.variables], [v.dtype for v in default.variables])
        self.assertEqual([GroupRegLevelModel._local_name(v) for v in a2.variables],
                         [GroupRegLevelModel._local_name(v) for v in default.variables])
        prefix = tf.train.Checkpoint(model=a2).write(os.path.join(self.tmp, "a2"))
        tf.train.Checkpoint(model=default).read(prefix).assert_existing_objects_matched()
        for s, d in zip(a2.variables, default.variables):
            np.testing.assert_array_equal(s.numpy(), d.numpy(), d.name)
        back = self.repro_model(deform_decoder_dtype="float32")
        prefix = tf.train.Checkpoint(model=default).write(os.path.join(self.tmp, "default"))
        tf.train.Checkpoint(model=back).read(prefix).assert_existing_objects_matched()
        for s, d in zip(default.variables, back.variables):
            np.testing.assert_array_equal(s.numpy(), d.numpy(), d.name)
        fresh = self.repro_model()
        self.assertEqual(fresh.copy_weights_from(a2), len(a2.weights))
        for s, d in zip(a2.weights, fresh.weights):
            np.testing.assert_array_equal(s.numpy(), d.numpy(), d.name)


class TestBnEpsilonOverride(MixedPolicyCase):
    """a3: --bn-epsilon-override sets the BN epsilon only (LayerNorm / config unchanged)."""

    def test_override(self):
        default = self.repro_model()
        m = self.repro_model(bn_epsilon_override=1e-2)
        bns = L.batch_norm_layers(m)
        self.assertTrue(bns)
        self.assertTrue(all(b.bn.epsilon == 1e-2 for b in bns))
        self.assertTrue(all(b.bn.epsilon == default.config.bn_epsilon for b in L.batch_norm_layers(default)))
        lns = [x for x in m.submodules if isinstance(x, (L.LayerNormWithAffine, L.InstanceNormWithAffine))]
        self.assertTrue(lns)
        self.assertTrue(all(x.epsilon == m.config.bn_epsilon for x in lns))
        self.assertEqual(m.config.to_dict(), default.config.to_dict())
        m.set_weights(default.get_weights())
        for model in (default, m):                       # a near-zero recalibrated variance
            mv = model.deform_decoder.d1.norm.bn.moving_variance
            mv.assign(np.full(mv.shape, 1e-6, np.float32))
        x, _ = self.repro_data()
        head = np.random.default_rng(1).normal(0, 0.02, default.deform_decoder.svf_head.kernel.shape)
        for model in (default, m):
            model.deform_decoder.svf_head.kernel.assign(head.astype(np.float32))
        a = default(x, task="deform", training=False)["v_raw"].numpy()
        b = m(x, task="deform", training=False)["v_raw"].numpy()
        self.assertFalse(np.allclose(a, b))


class TestInvalidKwargs(MixedPolicyCase):

    def test_invalid(self):
        cfg = tiny_config(deform_decoder=True)
        for kw in (dict(fp16_act_clip=-1.0), dict(fp16_act_clip=40000.0), dict(fp16_act_clip_scope="encoder"),
                   dict(deform_decoder_dtype="bfloat16"), dict(bn_epsilon_override=-1.0),
                   dict(bn_epsilon_override=float("nan"))):
            with self.assertRaises(ValueError, msg=repr(kw)):
                GroupRegLevelModel(4, cfg, **kw)
        with self.assertRaises(ValueError):                 # deform_decoder scope without a decoder
            GroupRegLevelModel(4, tiny_config(), fp16_act_clip=30000.0)
        GroupRegLevelModel(4, tiny_config(), fp16_act_clip=30000.0, fp16_act_clip_scope="all")
        MP.set_global_policy("float32")
        with self.assertRaises(ValueError):                 # float16 decoder under a float32 policy
            GroupRegLevelModel(4, cfg, deform_decoder_dtype="float16")
        GroupRegLevelModel(4, cfg, deform_decoder_dtype="float32")   # no-op under float32


if __name__ == "__main__":
    unittest.main()
