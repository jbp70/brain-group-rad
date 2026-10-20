"""Encoder, linear decoder, heads and centering (models.group_net, milestone 3).

Level-1 (full resolution) models are only run forward (inference, N <= 2) to keep GPU memory small.
"""
import gc
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests import np_reference as npr
from tests.runtime_helpers import NARROW
import numpy as np
import tensorflow as tf
from scipy.linalg import expm as sp_expm

from data_loading import params as P
from models import group_net as G
from models import layers as L
from models import transforms as tfm
from models.config import ModelConfig

SKIP_SHAPES = {  # design 4.3: stem, L2, L3, L4, L5, L6 (spatial) per level
    4: [(40, 56, 48), (40, 56, 48), (40, 56, 48), (20, 28, 24), (10, 14, 12), (5, 7, 6)],
    2: [(80, 112, 96), (80, 112, 96), (40, 56, 48), (20, 28, 24), (10, 14, 12), (5, 7, 6)],
    1: [(160, 224, 192), (80, 112, 96), (40, 56, 48), (20, 28, 24), (10, 14, 12), (5, 7, 6)],
}
FILTERS = (32, 64, 128, 192, 256, 384)


def images(level, n, seed=0):
    rng = np.random.default_rng(seed)
    shape = P.LEVEL_SHAPE[level]
    vols = [npr.smooth_phantom(shape, rng, sigma=4.0 * 4 / level) for _ in range(n)]
    return tf.constant(np.stack(vols)[..., None].astype(np.float32))


def set_random_heads(model, x, rng, rot=0.3, trans_mm=8.0, ss=0.05, training=False):
    """Nonzero head kernels scaled so the head outputs vary by ~rot (10D), ~trans_mm, ~ss.

    Scale with the features of the mode the test runs in: training-mode BatchNorm changes the
    feature magnitudes, and too-large rotations (relative angles near 180 deg) make the
    barycenter's matrix log ill-defined.
    """
    feat = model(x, task="affine", training=training)["dense"].numpy()
    dec = model.linear_decoder
    units = feat.shape[-1]
    scale = 1.0 / (np.sqrt(units) * max(float(np.sqrt(np.mean(feat ** 2))), 1e-6))
    k = rng.normal(size=(units, 13)) * scale
    k[:, :10] *= rot
    k[:, 10:] *= trans_mm
    dec.rigid_head.kernel.assign(k.astype(np.float32))
    dec.scale_shear_head.kernel.assign((rng.normal(size=(units, 6)) * scale * ss).astype(np.float32))


def random_rigid(rng, max_deg=20.0, max_mm=8.0):
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    ang = np.deg2rad(rng.uniform(-max_deg, max_deg))
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    t = np.eye(4)
    t[:3, :3] = sp_expm(k * ang)
    t[:3, 3] = rng.uniform(-max_mm, max_mm, size=3)
    return t


def kernels(variables):
    return [v for v in variables if v.name.split("/")[-1].startswith("kernel")]


def free_memory():
    """Keras layers hold reference cycles: collect them so the next test gets the GPU memory back."""
    tf.keras.backend.clear_session()
    gc.collect()


class NetTestCase(unittest.TestCase):
    def tearDown(self):
        free_memory()


class TestShapesAndLevels(NetTestCase):
    """Shapes at all three levels, identical parameters and weight copying across levels."""

    @classmethod
    def setUpClass(cls):
        free_memory()
        cls.models = {}
        cls.outputs = {}
        for level in (1, 2, 4):                       # full resolution first, on a clean GPU
            m = G.GroupRegLevelModel(level, ModelConfig(group_config="A"))
            out = m(images(level, cls.N[level], seed=level), task="affine", training=False)
            cls.models[level] = m
            cls.outputs[level] = {k: out[k].numpy() for k in ("A_raw", "A", "params", "bottleneck")}
            cls.outputs[level]["skips"] = [tuple(s.shape) for s in out["skips"]]
            del out
            free_memory()

    N = {4: 2, 2: 2, 1: 1}   # N = 1 at full resolution: ~1 GB per float32 stem output (8 GB GPU)

    @classmethod
    def tearDownClass(cls):
        del cls.models
        free_memory()

    def test_shapes(self):
        for level in P.LEVELS:
            o, n = self.outputs[level], self.N[level]
            self.assertEqual(o["bottleneck"].shape, (n, 5, 7, 6, 384), level)
            self.assertEqual(o["skips"], [(n,) + s + (f,) for s, f in zip(SKIP_SHAPES[level], FILTERS)])
            self.assertEqual(o["A_raw"].shape, (n, 4, 4))
            self.assertEqual(o["A"].shape, (n, 4, 4))
            self.assertEqual(o["params"].shape, (n, 19))

    def test_identity_at_init_all_levels(self):
        for level in P.LEVELS:
            eye = np.broadcast_to(np.eye(4), (self.N[level], 4, 4))
            np.testing.assert_allclose(self.outputs[level]["A_raw"], eye, atol=1e-6)
            np.testing.assert_allclose(self.outputs[level]["A"], eye, atol=1e-6)

    def test_identical_parameters_across_levels(self):
        sig = {}
        for level, m in self.models.items():
            sig[level] = [(m._local_name(v), tuple(v.shape)) for v in m.weights]
        self.assertEqual(sig[4], sig[2])
        self.assertEqual(sig[4], sig[1])
        counts = {lvl: m.count_params() for lvl, m in self.models.items()}
        self.assertEqual(len(set(counts.values())), 1, counts)
        self.assertFalse(any(isinstance(l, tf.keras.layers.AveragePooling3D) and l.weights
                             for l in self.models[1].encoder.submodules))

    def test_copy_weights_4_to_2(self):
        m4, m2 = self.models[4], self.models[2]
        set_random_heads(m4, images(4, 2), np.random.default_rng(7))
        n = m2.copy_weights_from(m4)
        self.assertEqual(n, len(m4.weights))
        for a, b in zip(m4.weights, m2.weights):
            np.testing.assert_array_equal(a.numpy(), b.numpy())
        out = m2(images(2, 2, seed=11), task="affine", training=False)
        self.assertTrue(np.all(np.isfinite(out["A"].numpy())))

    def test_copy_rejects_mismatch(self):
        other = G.GroupRegLevelModel(4, ModelConfig(group_config="B")).build_model()
        with self.assertRaises(ValueError):
            other.copy_weights_from(self.models[4])

    def test_copy_rejects_shape_neutral_config_difference(self):
        # Same variable shapes and names, different function: must be refused.
        other = G.GroupRegLevelModel(2, ModelConfig(translation_scale=10.0)).build_model()
        with self.assertRaises(ValueError):
            other.copy_weights_from(self.models[4])
        # The init seed alone is irrelevant after copying.
        seeded = G.GroupRegLevelModel(2, ModelConfig(seed=123)).build_model()
        seeded.copy_weights_from(self.models[4])


class TestCopyRoundTrip(NetTestCase):
    def test_same_level_exact(self):
        x = images(4, 3, seed=3)
        a = G.GroupRegLevelModel(4).build_model()
        b = G.GroupRegLevelModel(4).build_model()
        set_random_heads(a, x, np.random.default_rng(1))
        a(x, task="rigid", training=True)                  # move the BN statistics off init
        oa = a(x, task="affine", training=False)
        ob0 = b(x, task="affine", training=False)
        self.assertGreater(np.abs(oa["A_raw"].numpy() - ob0["A_raw"].numpy()).max(), 1e-3)
        b.copy_weights_from(a)
        ob = b(x, task="affine", training=False)
        for k in ("A_raw", "A", "bottleneck"):
            np.testing.assert_array_equal(oa[k].numpy(), ob[k].numpy())


class TestGroupBehaviour(NetTestCase):
    """Permutation equivariance and where group information enters (eval mode)."""

    def _model(self, group_config, n=4, seed=0):
        tf.random.set_seed(seed)
        x = images(4, n, seed=seed)
        m = G.GroupRegLevelModel(4, ModelConfig(group_config=group_config)).build_model()
        set_random_heads(m, x, np.random.default_rng(seed))
        return m, x

    def test_permutation_equivariance(self):
        for cfg in ("A", "B"):
            m, x = self._model(cfg, n=5)
            perm = np.array([3, 0, 4, 1, 2])
            for task in ("rigid", "affine"):
                o = m(x, task=task, training=False)
                op = m(tf.gather(x, perm), task=task, training=False)
                a_raw, a = o["A_raw"].numpy(), o["A"].numpy()
                self.assertGreater(np.abs(a_raw - np.eye(4)).max(), 0.05)   # non-trivial
                # float32 rounding: 1e-5 absolute on the rotation block; translations are mm of
                # magnitude ~10, so 1e-5 relative to the group's largest translation (~1e-4 mm)
                for got, ref in ((op["A_raw"].numpy(), a_raw[perm]), (op["A"].numpy(), a[perm])):
                    np.testing.assert_allclose(got[:, :3, :3], ref[:, :3, :3], atol=1e-5,
                                               err_msg="%s %s" % (cfg, task))
                    t_tol = 1e-5 * max(1.0, float(np.abs(ref[:, :3, 3]).max()))
                    np.testing.assert_allclose(got[:, :3, 3], ref[:, :3, 3], rtol=0, atol=t_tol,
                                               err_msg="%s %s" % (cfg, task))
                    np.testing.assert_array_equal(got[:, 3], ref[:, 3])
                # centered: barycenter of A is the identity
                mu = tfm.barycenter(tf.constant(a, tf.float64), iters=10).numpy()
                np.testing.assert_allclose(mu, np.eye(4), atol=1e-3)

    def test_group_information_flow(self):
        rng = np.random.default_rng(9)
        for cfg in ("A", "B"):
            m, x = self._model(cfg, n=4, seed=1)
            x2 = x.numpy().copy()
            x2[2] = npr.smooth_phantom(x2.shape[1:4], rng)[..., None]
            o1 = m(x, task="affine", training=False)
            o2 = m(tf.constant(x2), task="affine", training=False)
            others = [0, 1, 3]
            d_enc = np.abs(o1["bottleneck"].numpy()[others] - o2["bottleneck"].numpy()[others]).max()
            d_skip = max(np.abs(s1.numpy()[others] - s2.numpy()[others]).max()
                         for s1, s2 in zip(o1["skips"], o2["skips"]))
            d_out = np.abs(o1["A_raw"].numpy()[others] - o2["A_raw"].numpy()[others]).max()
            if cfg == "A":
                self.assertGreater(d_enc, 1e-3)
                self.assertGreater(d_skip, 1e-3)
            else:
                self.assertEqual(d_enc, 0.0)                    # per-image encoder in config B
                self.assertEqual(d_skip, 0.0)
            self.assertGreater(d_out, 1e-4, cfg)                 # decoder group blocks


class TestGradients(NetTestCase):
    """Regression test for the groupwise reference bug (kernels without gradient)."""

    def _check(self, cfg, task):
        tf.random.set_seed(0)
        rng = np.random.default_rng(0)
        n = 3
        x = images(4, n, seed=5)
        m = G.GroupRegLevelModel(4, cfg).build_model()
        set_random_heads(m, x, rng, training=True)
        t = np.stack([random_rigid(rng) for _ in range(n)])
        target = tfm.barycentric_targets(tf.constant(t, tf.float32))
        s = tfm.brain_second_moment(np.ones(P.LEVEL_SHAPE[4]), P.LEVEL_SPACING[4])
        variables = m.trainable_variables
        # Two passes, the second one checked: on the local GTX 1080 the first backward pass of a
        # new conv shape can return 512 NaN entries in the encoder l3 kernel gradient even with
        # cuDNN autotune off (CPU finite; milestone-4 report, milestone-6 report). The routing
        # checks below are unaffected by the extra pass (BN statistics only move slightly).
        for _ in range(2):
            with tf.GradientTape() as tape:
                out = m(x, task=task, training=True)
                loss = tf.reduce_mean(tfm.point_loss(out["A"] - target, s))
            grads = tape.gradient(loss, variables)
        self.assertTrue(np.isfinite(float(loss)))
        unused = set()
        if task == "rigid":
            unused = {id(v) for v in m.linear_decoder.scale_shear_head.trainable_variables}
        for v, g in zip(variables, grads):
            if id(v) in unused:
                self.assertIsNone(g, v.name)
                continue
            self.assertIsNotNone(g, v.name)
            g = tf.convert_to_tensor(g).numpy()
            self.assertTrue(np.all(np.isfinite(g)), v.name)
        for v in kernels(variables):
            if id(v) in unused:
                continue
            g = tf.convert_to_tensor(grads[[id(u) for u in variables].index(id(v))]).numpy()
            self.assertGreater(np.abs(g).max(), 0.0, v.name)
        return len(variables), len(kernels(variables))

    def test_every_variable_gets_gradient(self):
        for cfg in (ModelConfig(group_config="A"), ModelConfig(group_config="B"),
                    # Narrow residual model: on the local GTX 1080, cuDNN autotuning of the
                    # full-width residual L5 conv (3x3, 512 -> 256 at 10x14x12) takes ~2 min.
                    ModelConfig(group_config="A", block="residual",
                                filters=(8, 16, 16, 24, 32, 48)),
                    ModelConfig(group_config="B", dense_group_concat_first=True)):
            for task in ("affine", "rigid"):
                with self.subTest(cfg=cfg.group_config, block=cfg.block, task=task):
                    nv, nk = self._check(cfg, task)
                    self.assertGreater(nk, 10)
                    free_memory()


class TestDynamicN(NetTestCase):
    def test_no_retrace(self):
        m = G.GroupRegLevelModel(4).build_model()
        traces = []

        @tf.function(input_signature=[tf.TensorSpec([None, 40, 56, 48, 1], tf.float32)])
        def fwd(x):
            traces.append(1)
            return m(x, task="rigid", training=False)["A"]

        for n in (2, 3, 5):
            self.assertEqual(tuple(fwd(images(4, n, seed=n)).shape), (n, 4, 4))
        self.assertEqual(len(traces), 1)


class TestMixedPrecision(NetTestCase):
    def test_mixed_float16(self):
        old = tf.keras.mixed_precision.global_policy()
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            x = images(4, 3)
            m = G.GroupRegLevelModel(4).build_model()
            set_random_heads(m, x, np.random.default_rng(2))
            with tf.GradientTape() as tape:
                out = m(x, task="affine", training=True)
                loss = tf.reduce_mean(tf.square(out["A"] - tf.eye(4)))
            grads = tape.gradient(loss, m.trainable_variables)
            self.assertEqual(out["bottleneck"].dtype, tf.float16)
            self.assertEqual(out["skips"][0].dtype, tf.float16)
            for k in ("A_raw", "A", "params"):
                self.assertEqual(out[k].dtype, tf.float32, k)
            self.assertEqual(m.linear_decoder.rigid_head.compute_dtype, "float32")
            self.assertEqual(m.centering.compute_dtype, "float32")
            self.assertTrue(all(v.dtype == tf.float32 for v in m.trainable_variables))
            self.assertTrue(np.all(np.isfinite(out["A"].numpy())))
            self.assertTrue(all(g is not None for g in grads))
            o_eval = m(x, task="rigid", training=False)
            self.assertTrue(np.all(np.isfinite(o_eval["A"].numpy())))
        finally:
            tf.keras.mixed_precision.set_global_policy(old)
        self.assertEqual(tf.keras.mixed_precision.global_policy().name, old.name)


class TestBNFreezeAndDropout(NetTestCase):
    def test_bn_freeze(self):
        x = images(4, 3)
        m = G.GroupRegLevelModel(4).build_model()
        set_random_heads(m, x, np.random.default_rng(4))
        bns = [l for l in m.submodules if isinstance(l, L.BatchNormWithSeparateAffine)]
        self.assertEqual(m.set_bn_stats_frozen(True), len(bns))
        stats0 = [v.numpy().copy() for v in m.non_trainable_variables]
        with tf.GradientTape() as tape:
            out = m(x, task="affine", training=True)
            loss = tf.reduce_mean(tf.square(out["A"] - tf.eye(4)))
        affine_vars = [w for b in bns for w in b.trainable_weights]
        grads = tape.gradient(loss, affine_vars)
        for a, b in zip(stats0, m.non_trainable_variables):
            np.testing.assert_array_equal(a, b.numpy())
        self.assertTrue(all(g is not None for g in grads))
        self.assertGreater(max(float(tf.reduce_max(tf.abs(g))) for g in grads), 0.0)
        m.set_bn_stats_frozen(False)
        m(x, task="affine", training=True)
        changed = [np.abs(a - b.numpy()).max() > 0 for a, b in zip(stats0, m.non_trainable_variables)]
        self.assertTrue(all(changed))

    def test_dropout_training_only(self):
        x = images(4, 4)
        m = G.GroupRegLevelModel(4).build_model()
        m.set_bn_stats_frozen(True)        # deterministic pre-dropout features in both modes
        o_eval = m(x, task="rigid", training=False)
        np.testing.assert_array_equal(o_eval["head_input"].numpy(), o_eval["dense"].numpy())
        o_tr = m(x, task="rigid", training=True)
        pre, post = o_tr["dense"].numpy(), o_tr["head_input"].numpy()
        nz = pre != 0
        dropped = (post == 0) & nz
        frac = dropped.sum() / nz.sum()
        self.assertGreater(frac, 0.08)
        self.assertLess(frac, 0.22)
        kept = nz & ~dropped
        np.testing.assert_allclose(post[kept], pre[kept] / 0.85, rtol=1e-5)


class TestParameterCounts(NetTestCase):
    def test_counts(self):
        rows = {}
        for cfg_name in ("A", "B"):
            for first in (False, True):
                m = G.GroupRegLevelModel(4, ModelConfig(group_config=cfg_name,
                                                        dense_group_concat_first=first))
                rows[(cfg_name, first)] = m.build_model().count_params_by_part()
        keys = list(rows[("A", False)].keys())
        print("\nParameter counts (trainable; level-independent):")
        print("%-40s %14s %14s %14s %14s" % ("part", "A", "B", "A+first", "B+first"))
        for k in keys:
            print("%-40s %14s %14s %14s %14s" % (
                k, *["{:,}".format(rows[c][k]) for c in
                     (("A", False), ("B", False), ("A", True), ("B", True))]))
        d = rows[("A", False)]
        self.assertEqual(d["linear_decoder/dense_0/dense"], 5040 * 2048 + 2048)     # 10.3 M
        self.assertEqual(rows[("A", True)]["linear_decoder/dense_0/dense"], 2 * 5040 * 2048 + 2048)
        self.assertEqual(d["linear_decoder/dense_1/dense"], 2 * 2048 * 512 + 512)    # group block
        overhead = d["group_overhead/linear_decoder"]
        self.assertAlmostEqual(overhead / 1e6, 1.15, delta=0.05)
        self.assertAlmostEqual((d["linear_decoder"] - overhead) / 1e6, 11.5, delta=0.1)
        for first in (False, True):   # the decoder is the same in A and B
            self.assertEqual(rows[("A", first)]["linear_decoder"],
                             rows[("B", first)]["linear_decoder"])
        self.assertEqual(rows[("B", False)]["group_overhead/encoder"], 0)
        self.assertEqual(rows[("A", False)]["encoder"] - rows[("B", False)]["encoder"],
                         rows[("A", False)]["group_overhead/encoder"])
        self.assertEqual(d["trainable"], d["encoder"] + d["linear_decoder"])


class TestModelConfig(unittest.TestCase):
    def test_defaults_and_round_trip(self):
        c = ModelConfig()
        self.assertEqual(c.filters, FILTERS)
        self.assertEqual((c.reductions, c.dense, c.dropout), ((192, 96, 48, 24), (2048, 512), 0.15))
        self.assertEqual((c.group_config, c.dense_group_concat_first, c.norm, c.block),
                         ("A", False, "batch", "single"))
        self.assertEqual((c.negative_slope, c.centering_iters, c.translation_scale), (0.01, 4, 1.0))
        self.assertEqual(c.dense_norm, "layer")                      # v3.4; conv norm stays batch
        for ok in ("layer", "batch", "none"):
            self.assertEqual(ModelConfig(dense_norm=ok).dense_norm, ok)
        self.assertEqual(ModelConfig.from_dict(c.to_dict()), c)
        self.assertEqual(c.replace(group_config="B").group_config, "B")
        for bad in (dict(group_config="C"), dict(block="dense"), dict(norm="group"),
                    dict(dense_norm="instance"), dict(filters=(1, 2)), dict(dropout=1.0)):
            with self.assertRaises(ValueError):
                ModelConfig(**bad)
        with self.assertRaises(ValueError):
            ModelConfig.from_dict({"nope": 1})

    def test_deform_fields(self):
        c = ModelConfig()
        self.assertEqual((c.deform_decoder, c.decoder_widths, c.integration_steps), (False, None, 7))
        self.assertEqual(c.resolved_decoder_widths(), (128, 96, 64, 32, 16))
        d = ModelConfig(deform_decoder=True, decoder_widths=[8, 8, 8, 8, 4], integration_steps=0)
        self.assertEqual(d.decoder_widths, (8, 8, 8, 8, 4))
        self.assertEqual(d.resolved_decoder_widths(), (8, 8, 8, 8, 4))
        self.assertEqual(ModelConfig.from_dict(d.to_dict()), d)
        old = c.to_dict()
        for k in ("deform_decoder", "decoder_widths", "integration_steps"):
            old.pop(k)
        self.assertEqual(ModelConfig.from_dict(old), c)               # pre-milestone-6 config.json
        for bad in (dict(decoder_widths=(8, 8)), dict(decoder_widths=(8, 8, 0, 8, 8)),
                    dict(integration_steps=-1)):
            with self.assertRaises(ValueError):
                ModelConfig(**bad)

    def test_translation_scale(self):
        x = images(4, 2)
        m = G.GroupRegLevelModel(4, ModelConfig(translation_scale=10.0)).build_model()
        k = np.zeros((512, 13), np.float32)
        k[:, 10:] = 1e-3
        m.linear_decoder.rigid_head.kernel.assign(k)
        o = m(x, task="rigid", training=False)
        raw = o["head_input"].numpy() @ k + m.linear_decoder.rigid_head.bias.numpy()
        np.testing.assert_allclose(o["params"].numpy()[:, 10:], 10.0 * raw[:, 10:], rtol=1e-5)
        np.testing.assert_allclose(o["params"].numpy()[:, :10], raw[:, :10], atol=1e-6)
        np.testing.assert_allclose(o["A_raw"].numpy()[:, :3, 3], o["params"].numpy()[:, 10:],
                                   rtol=1e-6)


def deform_config(**kw):
    d = dict(NARROW, deform_decoder=True)
    d.update(kw)
    return ModelConfig(**d)


def deform_model(level=4, n=1, **kw):
    return G.GroupRegLevelModel(level, deform_config(**kw)).build_model(n)


def randomize_svf_head(model, rng, std=0.05):
    """Nonzero SVF head kernel: v of roughly one voxel with the narrow decoder."""
    k = model.deform_decoder.svf_head.kernel
    k.assign((rng.normal(size=k.shape) * std).astype(np.float32))


def part_ids(layer):
    return {id(v) for v in layer.trainable_variables}


class TestDeformShapes(NetTestCase):
    # Tiny widths: at level 1 a narrow D1 group block alone needs > 6 GB (64 float32 channels at
    # 160x224x192); shapes and identity do not depend on the widths.
    TINY = dict(filters=(4, 8, 8, 8, 8, 8), reductions=(8, 8, 8, 8), dense=(32, 16),
                decoder_widths=(8, 8, 8, 4, 4))

    def test_shapes_and_identity_at_init(self):
        for level, n in ((1, 1), (2, 2), (4, 3)):
            with self.subTest(level=level):
                m = G.GroupRegLevelModel(level, ModelConfig(deform_decoder=True, **self.TINY))
                m.build_model(n)
                x = images(level, n, seed=level)
                out = m(x, task="deform", training=False, return_warped=True)
                shape = (n,) + tuple(P.LEVEL_SHAPE[level])
                for k in ("v_raw", "v", "u", "u_inv"):
                    self.assertEqual(tuple(out[k].shape), shape + (3,), k)
                    self.assertEqual(out[k].dtype, tf.float32, k)
                    np.testing.assert_array_equal(out[k].numpy(), 0.0)
                self.assertEqual(tuple(out["dec_feat"].shape),
                                 shape + (m.config.resolved_decoder_widths()[-1],))
                np.testing.assert_array_equal(out["warped"].numpy(), x.numpy())
                self.assertEqual(tuple(m(x, task="rigid", training=False)["A"].shape), (n, 4, 4))
                del m, out
                free_memory()

    def test_off_by_default(self):
        m = G.GroupRegLevelModel(4, ModelConfig(**NARROW)).build_model()
        self.assertIsNone(m.deform_decoder)
        self.assertFalse(any("deform" in v.name for v in m.weights))
        self.assertNotIn("deform_decoder", m.count_params_by_part())
        with self.assertRaises(ValueError):
            m(images(4, 2), task="deform", training=False)
        with self.assertRaises(ValueError):
            G.GroupRegLevelModel(4, gradient_checkpointing="all")
        self.assertEqual(G.ALL_TASKS, ("rigid", "affine", "deform"))
        self.assertEqual(G.TASKS, ("rigid", "affine"))


class TestDeformCopyWeights(NetTestCase):
    def test_copy_4_to_2(self):
        m4, m2 = deform_model(4, 2), deform_model(2, 2)
        sig = [[(m._local_name(v), tuple(v.shape)) for v in m.weights] for m in (m4, m2)]
        self.assertEqual(sig[0], sig[1])
        c4, c2 = m4.count_params_by_part(), m2.count_params_by_part()
        self.assertEqual(c4, c2)
        self.assertEqual(c4["trainable"], c4["encoder"] + c4["linear_decoder"] + c4["deform_decoder"])
        self.assertEqual(c4["deform_decoder"], sum(c4["deform_decoder/" + s] for s in
                                                   G.DECODER_STAGES + ("svf_head",)))
        self.assertGreater(c4["group_overhead/deform_decoder"], 0)
        randomize_svf_head(m4, np.random.default_rng(3))
        self.assertEqual(m2.copy_weights_from(m4), len(m4.weights))
        for a, b in zip(m4.weights, m2.weights):
            np.testing.assert_array_equal(a.numpy(), b.numpy())
        out = m2(images(2, 2, seed=4), task="deform", training=False)
        self.assertTrue(np.all(np.isfinite(out["u"].numpy())))
        self.assertGreater(np.abs(out["u"].numpy()).max(), 1e-3)


class TestDeformGradients(NetTestCase):
    def test_gradient_routing(self):
        tf.random.set_seed(0)
        m = deform_model(4, 3)
        x = images(4, 3, seed=6)
        randomize_svf_head(m, np.random.default_rng(6))
        variables = m.trainable_variables
        linear = part_ids(m.linear_decoder)
        deform = part_ids(m.deform_decoder)
        with tf.GradientTape() as tape:
            out = m(x, task="deform", training=True, return_warped=True)
            w = out["warped"]
            loss = (tf.reduce_mean(tf.square(w - tf.reduce_mean(w, axis=0, keepdims=True)))
                    + 1e-3 * tf.reduce_mean(tf.square(out["u_inv"])))
        grads = tape.gradient(loss, variables)
        self.assertTrue(np.isfinite(float(loss)))
        for v, g in zip(variables, grads):
            if id(v) in linear:
                self.assertIsNone(g, v.name)
            else:
                self.assertIsNotNone(g, v.name)
                self.assertTrue(np.all(np.isfinite(tf.convert_to_tensor(g).numpy())), v.name)
        self.assertGreater(len(deform), 10)
        g_head = grads[[id(v) for v in variables].index(id(m.deform_decoder.svf_head.kernel))]
        self.assertGreater(float(tf.reduce_max(tf.abs(g_head))), 0.0)

        # Rigid step: the deform decoder does not run, so its variables get no gradient.
        with tf.GradientTape() as tape:
            out = m(x, task="rigid", training=True)
            loss = tf.reduce_mean(tf.square(out["A_raw"] - tf.eye(4)))
        grads = tape.gradient(loss, variables)
        for v, g in zip(variables, grads):
            if id(v) in deform:
                self.assertIsNone(g, v.name)


class TestDeformCentering(NetTestCase):
    def test_mean_zero(self):
        m = deform_model(4, 4)
        randomize_svf_head(m, np.random.default_rng(8))
        out = m(images(4, 4, seed=8), task="deform", training=False)
        v, v_raw = out["v"].numpy(), out["v_raw"].numpy()
        self.assertLess(np.abs(v.mean(axis=0)).max(), 1e-5)
        self.assertGreater(np.abs(v_raw.mean(axis=0)).max(), 1e-3)
        np.testing.assert_allclose(v - v[:1], v_raw - v_raw[:1], atol=1e-5)


class TestDeformPermutation(NetTestCase):
    def test_permutation_equivariance(self):
        perm = np.array([2, 0, 3, 1])
        for cfg in ("A", "B"):
            with self.subTest(cfg=cfg):
                m = deform_model(4, 4, group_config=cfg)
                randomize_svf_head(m, np.random.default_rng(10))
                x = images(4, 4, seed=10)
                o = m(x, task="deform", training=False)
                op = m(tf.gather(x, perm), task="deform", training=False)
                self.assertGreater(np.abs(o["u"].numpy()).max(), 0.1)
                for k in ("v", "u", "u_inv"):
                    np.testing.assert_allclose(op[k].numpy(), o[k].numpy()[perm], atol=1e-4,
                                               err_msg="%s %s" % (cfg, k))
                del m
                free_memory()


class TestGradientCheckpointing(NetTestCase):
    def test_recompute_matches(self):
        tf.random.set_seed(1)
        m = deform_model(4, 3)
        x = images(4, 3, seed=12)
        randomize_svf_head(m, np.random.default_rng(12))
        variables = [v for v in m.trainable_variables
                     if id(v) not in part_ids(m.linear_decoder)]

        def loss_and_grads():
            with tf.GradientTape() as tape:
                out = m(x, task="deform", training=True, return_warped=True)
                w = out["warped"]
                loss = (tf.reduce_mean(tf.square(w - tf.reduce_mean(w, axis=0, keepdims=True)))
                        + 1e-3 * tf.reduce_mean(tf.square(out["u_inv"])))
            return loss, tape.gradient(loss, variables)

        graph_fn = tf.function(loss_and_grads)
        results = {}
        for mode in ("none", "integrator", "decoder", "both"):
            m.gradient_checkpointing = mode
            results[mode] = loss_and_grads()
        m.gradient_checkpointing = "both"
        results["both_graph"] = graph_fn()
        m.gradient_checkpointing = "none"
        ref_loss, ref_grads = results["none"]
        # Variables whose gradient is analytically ~0 (e.g. a conv bias right before a BN) only
        # carry rounding noise: floor the per-variable tolerance at 1e-6 of the largest gradient.
        g_max = max(float(np.abs(tf.convert_to_tensor(r).numpy()).max()) for r in ref_grads)
        for mode, (loss, grads) in results.items():
            self.assertAlmostEqual(float(loss), float(ref_loss), delta=1e-4 * abs(float(ref_loss)),
                                   msg=mode)
            for v, g, r in zip(variables, grads, ref_grads):
                self.assertIsNotNone(g, "%s %s" % (mode, v.name))
                g, r = tf.convert_to_tensor(g).numpy(), tf.convert_to_tensor(r).numpy()
                tol = 1e-4 * float(np.abs(r).max()) + 1e-6 * g_max
                self.assertLessEqual(float(np.abs(g - r).max()), tol, "%s %s" % (mode, v.name))


class TestDeformMixedPrecision(NetTestCase):
    def test_mixed_float16(self):
        old = tf.keras.mixed_precision.global_policy()
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            m = deform_model(4, 3)
            randomize_svf_head(m, np.random.default_rng(14))
            x = images(4, 3, seed=14)
            with tf.GradientTape() as tape:
                out = m(x, task="deform", training=True, return_warped=True)
                loss = tf.reduce_mean(tf.square(out["warped"] - x))
            grads = tape.gradient(loss, m.deform_decoder.trainable_variables)
            for k in ("v_raw", "v", "u", "u_inv", "warped"):
                self.assertEqual(out[k].dtype, tf.float32, k)
                self.assertTrue(np.all(np.isfinite(out[k].numpy())), k)
            self.assertEqual(out["dec_feat"].dtype, tf.float16)
            self.assertEqual(m.deform_decoder.svf_head.compute_dtype, "float32")
            self.assertGreater(np.abs(out["u"].numpy()).max(), 1e-3)
            self.assertTrue(all(g is not None for g in grads))
        finally:
            tf.keras.mixed_precision.set_global_policy(old)
        self.assertEqual(tf.keras.mixed_precision.global_policy().name, old.name)


class TestSmokeTraining(NetTestCase):
    def test_overfit_one_group(self):
        tf.random.set_seed(0)
        rng = np.random.default_rng(0)
        n, level = 4, 4
        shape, spacing = P.LEVEL_SHAPE[level], P.LEVEL_SPACING[level]
        base = npr.smooth_phantom(shape, rng, sigma=4.0)
        t = np.stack([random_rigid(rng, 20.0, 8.0) for _ in range(n)]).astype(np.float32)
        src = tf.constant(np.repeat(base[None, ..., None], n, 0).astype(np.float32))
        x = L.transform_image(src, tfm.affine_inverse(tf.constant(t)), spacing)  # I(T^-1 y)
        target = tfm.barycentric_targets(tf.constant(t))
        s = tfm.brain_second_moment(base > 0.2 * base.max(), spacing)
        m = G.GroupRegLevelModel(level).build_model()
        opt = tf.keras.optimizers.Adam(1e-4)

        @tf.function
        def step():
            with tf.GradientTape() as tape:
                out = m(x, task="rigid", training=True)
                loss = tf.reduce_mean(tfm.point_loss(out["A"] - target, s))
            grads = tape.gradient(loss, m.trainable_variables)
            opt.apply_gradients([(g, v) for g, v in zip(grads, m.trainable_variables)
                                 if g is not None])
            return loss

        # 60 steps: with the v3.4 dense LayerNorm the first ~20 steps move the loss only ~12%
        # (the old group-relative dense BN amplified the small inter-image differences early on).
        losses = [float(step()) for _ in range(60)]
        print("\nsmoke training (level 4, N=4, rigid) RMS mm per step: " +
              " ".join("%.2f" % np.sqrt(v) for v in losses))
        self.assertTrue(np.all(np.isfinite(losses)))
        # training-mode BN on N = 4 plus dropout make single steps noisy: compare averages
        self.assertLess(np.mean(losses[-5:]), 0.7 * losses[0])


if __name__ == "__main__":
    unittest.main()
