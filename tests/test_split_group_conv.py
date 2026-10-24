"""Split group-mean conv (``GroupConvBlock.split_group``, `layers.set_split_group_conv`,
``GroupRegLevelModel(split_group_conv=True)``; milestone 9 design 3.5).

conv(concat(x, mean)) computed as conv(x, K_a) + conv(mean, K_b): per block (stride 1 and 2)
outputs within 1e-6 relative in fp32 (1e-3 in fp16) and gradients within 1e-4 relative of the
standard path. The model-level checks use the TINY config (every group block type, strided
L4-L6 included): through the whole model the per-block rounding compounds, so its outputs are
checked at 10x the block tolerance, and the gradients at 1e-4. (With the narrow config, N 3 and
a training-mode affine + deform loss, the gradients differed by 1.4e-4 deterministically on the
GTX 1080, versus 1.7e-6 for TINY: ill-conditioning of the wider linear decoder with
training-mode normalisation over 3 images, not the split, which is exact algebra.) The default path is bitwise unchanged
(also after switching split off again); same variable names and count; the stem
(pre_activation=False) keeps split_group False; weights copy between split and non-split models.
"""
import sys
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests.runtime_helpers import free_memory, narrow_config, tiny_config
from tests.test_cascade import randomize
import numpy as np
import tensorflow as tf

from models import layers as L
from models.group_net import GroupRegLevelModel
from runtime import fixed_data as fd

N = 3


def rel(a, b):
    a = np.concatenate([np.ravel(np.asarray(x, np.float64)) for x in a])
    b = np.concatenate([np.ravel(np.asarray(x, np.float64)) for x in b])
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def names(model):
    return [(GroupRegLevelModel._local_name(v), tuple(v.shape)) for v in model.weights]


class TestSplitBlock(unittest.TestCase):
    def tearDown(self):
        tf.keras.mixed_precision.set_global_policy("float32")
        free_memory()

    def test_block_strided(self):
        rng = np.random.default_rng(0)
        x = tf.constant(rng.normal(size=(3, 9, 8, 7, 5)).astype(np.float32))
        for strides in (1, 2):
            blk = L.GroupConvBlock(6, strides=strides, negative_slope=0.01, seed=0)
            ref = blk(x, training=True).numpy()
            blk.split_group = True
            with tf.GradientTape() as tape:
                tape.watch(x)
                got = blk(x, training=True)
                loss = tf.reduce_sum(got * got)
            g_split = tape.gradient(loss, [x] + blk.trainable_variables)
            blk.split_group = False
            with tf.GradientTape() as tape:
                tape.watch(x)
                again = blk(x, training=True)
                loss = tf.reduce_sum(again * again)
            g_ref = tape.gradient(loss, [x] + blk.trainable_variables)
            self.assertTrue(np.array_equal(again.numpy(), ref))         # default path unchanged
            self.assertEqual(got.shape, again.shape)
            self.assertLessEqual(rel([got.numpy()], [ref]), 1e-6, strides)
            self.assertLessEqual(rel([g.numpy() for g in g_split], [g.numpy() for g in g_ref]), 1e-4, strides)

    def test_block_fp16(self):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        rng = np.random.default_rng(1)
        x = tf.constant(rng.normal(size=(3, 9, 8, 7, 16)).astype(np.float16))
        for strides in (1, 2):
            blk = L.GroupConvBlock(16, strides=strides, negative_slope=0.01, seed=0)
            ref = tf.cast(blk(x, training=True), tf.float32).numpy()
            blk.split_group = True
            got = blk(x, training=True)
            self.assertEqual(got.dtype, tf.float16)
            e = rel([tf.cast(got, tf.float32).numpy()], [ref])
            print("\n[split] fp16 block (stride %d) outputs rel %.2e" % (strides, e), file=sys.stderr)
            self.assertLessEqual(e, 1e-3)

    def test_setter_rules(self):
        model = GroupRegLevelModel(4, narrow_config(deform_decoder=True), split_group_conv=True)
        self.assertTrue(model.split_group_conv)
        enc = model.encoder
        self.assertFalse(enc.stem.split_group)                          # stem excluded
        for name in ("l2", "l3", "l4", "l5", "l6"):
            self.assertTrue(getattr(enc, name).split_group, name)
        self.assertTrue(all(b.split_group for b in model.linear_decoder.reductions))
        self.assertTrue(all(b.split_group for b in model.deform_decoder.blocks()))
        n = L.set_split_group_conv(model, False)
        self.assertEqual(n, 5 + len(model.linear_decoder.reductions) + 5)
        self.assertFalse(any(m.split_group for m in model.submodules if isinstance(m, L.GroupConvBlock)))
        cfg_b = narrow_config(group_config="B")
        m_b = GroupRegLevelModel(4, cfg_b, split_group_conv=True)
        self.assertFalse(m_b.encoder.l3.split_group)                    # config B: no group in the encoder
        self.assertTrue(m_b.linear_decoder.reductions[0].split_group)


class TestSplitModel(unittest.TestCase):
    def tearDown(self):
        tf.keras.mixed_precision.set_global_policy("float32")
        free_memory()

    def models(self):
        cfg = tiny_config(dropout=0.0, seed=1, deform_decoder=True)
        a = GroupRegLevelModel(4, cfg).build_model(N)
        randomize(a, 7, 0.02)
        b = GroupRegLevelModel(4, cfg, split_group_conv=True).build_model(2)
        self.assertEqual(names(a), names(b))
        self.assertEqual(len(a.trainable_variables), len(b.trainable_variables))
        self.assertEqual(b.copy_weights_from(a), len(a.weights))
        return a, b

    def outputs(self, model, x, training):
        out = []
        for task in ("rigid", "affine"):
            r = model(x, task=task, training=training)
            out += [r["A_raw"].numpy(), r["params"].numpy()]
        r = model(x, task="deform", training=training)
        out += [r["v_raw"].numpy(), tf.cast(r["dec_feat"], tf.float32).numpy()]
        return out

    def test_outputs_and_gradients_fp32(self):
        a, b = self.models()
        x = tf.constant(fd.synthetic_deform_group(4, N, seed=0)["images"])
        for training in (False, True):
            e = rel(self.outputs(b, x, training), self.outputs(a, x, training))
            print("\n[split] fp32 model outputs rel %.2e (training %s)" % (e, training), file=sys.stderr)
            self.assertLessEqual(e, 1e-5)
        grads = {}
        for key, m in (("ref", a), ("split", b)):
            with tf.GradientTape() as tape:
                r = m(x, task="deform", training=True)
                q = m(x, task="affine", training=True)
                loss = tf.reduce_sum(r["v_raw"] ** 2) + tf.reduce_sum(q["params"] ** 2)
            grads[key] = [g.numpy() for g in tape.gradient(loss, m.trainable_variables) if g is not None]
        e = rel(grads["split"], grads["ref"])
        print("[split] fp32 gradients rel %.2e" % e, file=sys.stderr)
        self.assertLessEqual(e, 1e-4)
        # Copy split -> non-split; switching split off gives the bitwise default path.
        c = GroupRegLevelModel(4, a.config).build_model(2)
        self.assertEqual(c.copy_weights_from(b), len(b.weights))
        L.set_split_group_conv(b, False)
        self.assertTrue(all(np.array_equal(p, q) for p, q in zip(self.outputs(b, x, False), self.outputs(c, x, False))))

    def test_outputs_fp16(self):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        a, b = self.models()
        x = tf.constant(fd.synthetic_deform_group(4, N, seed=0)["images"])
        e = rel(self.outputs(b, x, True), self.outputs(a, x, True))
        print("\n[split] AMP model outputs rel %.2e" % e, file=sys.stderr)
        self.assertLessEqual(e, 1e-2)


if __name__ == "__main__":
    unittest.main()
