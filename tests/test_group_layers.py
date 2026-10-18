"""Group block layers (models.layers, milestone 3) against numpy references."""
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
import numpy as np
import tensorflow as tf

from models import layers as L


def relu(x, slope=0.0):
    return np.where(x > 0, x, slope * x)


class TestGroupMeanConcat(unittest.TestCase):
    def test_matches_numpy(self):
        x = np.random.default_rng(0).normal(size=(3, 4, 5, 2, 6)).astype(np.float32)
        out = L.group_mean_concat(tf.constant(x)).numpy()
        ref = np.concatenate([x, np.broadcast_to(x.mean(0, keepdims=True), x.shape)], -1)
        np.testing.assert_allclose(out, ref, atol=1e-6)

    def test_dynamic_n_keeps_static_channels(self):
        @tf.function(input_signature=[tf.TensorSpec([None, 4, 4, 4, 3], tf.float32)])
        def f(x):
            y = L.group_mean_concat(x)
            assert y.shape[-1] == 6, y.shape
            return y
        self.assertEqual(tuple(f(tf.zeros((5, 4, 4, 4, 3))).shape), (5, 4, 4, 4, 6))


class TestGroupConvBlock(unittest.TestCase):
    def test_group_block_matches_manual(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(3, 6, 5, 4, 2)).astype(np.float32)
        blk = L.GroupConvBlock(4, kernel_size=3, group=True, norm="none", negative_slope=0.1)
        out = blk(tf.constant(x)).numpy()
        self.assertEqual(tuple(blk.conv.kernel.shape), (3, 3, 3, 4, 4))   # doubled input
        h = relu(x, 0.1)
        hc = np.concatenate([h, np.broadcast_to(h.mean(0, keepdims=True), h.shape)], -1)
        ref = tf.nn.conv3d(hc, blk.conv.kernel, [1] * 5, "SAME") + blk.conv.bias
        np.testing.assert_allclose(out, ref.numpy(), atol=1e-5)
        self.assertEqual(blk.group_overhead_params(), 27 * 2 * 4)

    def test_plain_block_and_stem(self):
        x = tf.random.normal((2, 6, 6, 6, 3))
        plain = L.GroupConvBlock(5, group=False, norm="batch")
        y = plain(x, training=False)
        self.assertEqual(tuple(plain.conv.kernel.shape), (3, 3, 3, 3, 5))
        self.assertEqual(plain.group_overhead_params(), 0)
        self.assertEqual(tuple(y.shape), (2, 6, 6, 6, 5))
        stem = L.GroupConvBlock(5, group=False, pre_activation=False)
        out = stem(x)
        ref = tf.nn.conv3d(x, stem.conv.kernel, [1] * 5, "SAME") + stem.conv.bias
        np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=1e-5)   # no norm / act
        self.assertIsNone(stem.norm)

    def test_kaiming_init_variance(self):
        blk = L.GroupConvBlock(256, kernel_size=3, group=True, norm="none", negative_slope=0.0)
        blk(tf.zeros((1, 4, 4, 4, 32)))
        fan_in = 27 * 64
        self.assertAlmostEqual(float(np.var(blk.conv.kernel.numpy())) * fan_in / 2.0, 1.0,
                               delta=0.05)

    def test_strided(self):
        blk = L.GroupConvBlock(8, strides=2)
        self.assertEqual(tuple(blk(tf.zeros((2, 10, 14, 12, 4)), training=False).shape),
                         (2, 5, 7, 6, 8))


class TestNorms(unittest.TestCase):
    def test_bn_stats_freeze(self):
        rng = np.random.default_rng(2)
        x = tf.constant(rng.normal(3.0, 2.0, size=(4, 5, 5, 5, 3)).astype(np.float32))
        blk = L.GroupConvBlock(4, group=True, norm="batch")
        blk(x, training=False)
        bn = blk.norm
        self.assertIsInstance(bn, L.BatchNormWithSeparateAffine)
        mm0 = bn.bn.moving_mean.numpy().copy()
        mv0 = bn.bn.moving_variance.numpy().copy()
        self.assertEqual(L.set_bn_stats_frozen(blk, True), 1)
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(blk(x, training=True) ** 2)
        g_gamma, g_beta = tape.gradient(loss, [bn.affine.gamma, bn.affine.beta])
        np.testing.assert_array_equal(bn.bn.moving_mean.numpy(), mm0)
        np.testing.assert_array_equal(bn.bn.moving_variance.numpy(), mv0)
        self.assertGreater(float(tf.reduce_max(tf.abs(g_gamma))), 0.0)
        self.assertGreater(float(tf.reduce_max(tf.abs(g_beta))), 0.0)
        # frozen + training=True normalizes with the moving statistics (= inference output)
        np.testing.assert_allclose(blk(x, training=True).numpy(), blk(x, training=False).numpy(),
                                   atol=1e-6)
        L.set_bn_stats_frozen(blk, False)
        blk(x, training=True)
        self.assertGreater(np.abs(bn.bn.moving_mean.numpy() - mm0).max(), 1e-3)
        self.assertEqual(len(bn.trainable_weights), 2)                    # gamma, beta only

    def test_bn_uses_group_statistics(self):
        x = np.random.default_rng(3).normal(1.0, 3.0, size=(6, 7)).astype(np.float32)
        bn = L.BatchNormWithSeparateAffine(epsilon=1e-5)
        out = bn(tf.constant(x), training=True).numpy()
        ref = (x - x.mean(0)) / np.sqrt(x.var(0) + 1e-5)
        np.testing.assert_allclose(out, ref, atol=1e-4)

    def test_instance_norm(self):
        x = np.random.default_rng(4).normal(2.0, 5.0, size=(2, 6, 5, 4, 3)).astype(np.float32)
        out = L.InstanceNormWithAffine()(tf.constant(x)).numpy()
        ref = (x - x.mean((1, 2, 3), keepdims=True)) / np.sqrt(
            x.var((1, 2, 3), keepdims=True) + 1e-5)
        np.testing.assert_allclose(out, ref, atol=1e-4)
        blk = L.GroupConvBlock(4, norm="instance")
        self.assertEqual(tuple(blk(tf.constant(x)).shape), (2, 6, 5, 4, 4))

    def test_bad_norm(self):
        with self.assertRaises(ValueError):
            L.make_norm("group")
        with self.assertRaises(ValueError):
            L.GroupDenseBlock(4, norm="instance")


class TestGroupResBlock(unittest.TestCase):
    def test_projection_and_identity_skip(self):
        x = tf.random.normal((3, 8, 8, 8, 4))
        down = L.GroupResBlock(6, strides=2)
        y = down(x, training=False)
        self.assertEqual(tuple(y.shape), (3, 4, 4, 4, 6))
        self.assertIsNotNone(down.projection)
        same = L.GroupResBlock(4, strides=1)
        y2 = same(x, training=False)
        self.assertIsNone(same.projection)
        ref = same.block2(same.block1(x, training=False), training=False) + x
        np.testing.assert_allclose(y2.numpy(), ref.numpy(), atol=1e-6)
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(down(x, training=True) ** 2)
        grads = tape.gradient(loss, down.trainable_weights)
        self.assertTrue(all(g is not None for g in grads))
        self.assertEqual(down.group_overhead_params(), 27 * 4 * 6 + 27 * 6 * 6)


class TestGroupDenseBlock(unittest.TestCase):
    def test_matches_numpy(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(5, 7)).astype(np.float32)
        blk = L.GroupDenseBlock(3, group=True, norm="none", negative_slope=0.2)
        out = blk(tf.constant(x)).numpy()
        w = blk.dense.kernel.numpy()
        b = blk.dense.bias.numpy()
        self.assertEqual(w.shape, (14, 3))
        h = relu(x, 0.2)
        ref = h @ w[:7] + h.mean(0, keepdims=True) @ w[7:] + b    # per-group bias W_b mean(h)
        np.testing.assert_allclose(out, ref, atol=1e-5)
        self.assertEqual(blk.group_overhead_params(), 21)

    def test_no_group(self):
        blk = L.GroupDenseBlock(3, group=False)
        blk(tf.zeros((2, 7)), training=False)
        self.assertEqual(tuple(blk.dense.kernel.shape), (7, 3))
        self.assertEqual(blk.group_overhead_params(), 0)

    def test_layer_norm_per_sample(self):
        # Default dense norm (v3.4): per-vector LayerNorm, independent of the other group members
        # and identical in training and inference.
        rng = np.random.default_rng(6)
        x = rng.normal(2.0, 4.0, size=(5, 9)).astype(np.float32)
        ln = L.make_norm("layer")
        self.assertIsInstance(ln, L.LayerNormWithAffine)
        out = ln(tf.constant(x), training=True).numpy()
        ref = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
        np.testing.assert_allclose(out, ref, atol=1e-4)
        np.testing.assert_array_equal(out, ln(tf.constant(x), training=False).numpy())
        np.testing.assert_allclose(ln(tf.constant(x[:2]), training=True).numpy(), out[:2], atol=1e-6)
        self.assertEqual(sorted(w.name.split("/")[-1].split(":")[0] for w in ln.trainable_weights),
                         ["beta", "gamma"])
        self.assertEqual(ln.non_trainable_weights, [])
        blk = L.GroupDenseBlock(3, group=False)                   # default norm is 'layer'
        self.assertIsInstance(blk.norm, L.LayerNormWithAffine)
        y_all = blk(tf.constant(x), training=True).numpy()
        np.testing.assert_allclose(blk(tf.constant(x[:1]), training=True).numpy(), y_all[:1], atol=1e-5)
        self.assertIsInstance(L.NormAct("layer").norm, L.LayerNormWithAffine)
        self.assertIsNone(L.NormAct("none").norm)
        self.assertIsNotNone(L.NormAct("none").act)

    def test_layer_norm_mixed_precision(self):
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            ln = L.LayerNormWithAffine()
            x = tf.constant(np.random.default_rng(7).normal(100.0, 50.0, size=(3, 512)), tf.float16)
            y = ln(x)
            self.assertEqual(y.dtype, tf.float16)
            self.assertTrue(np.all(np.isfinite(y.numpy())))
            self.assertEqual(ln.affine.gamma.dtype, tf.float32)
        finally:
            tf.keras.mixed_precision.set_global_policy("float32")


if __name__ == "__main__":
    unittest.main()
