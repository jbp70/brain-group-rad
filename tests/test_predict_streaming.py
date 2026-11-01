"""Streaming prediction engine (predict-script design 4).

FEAT-001: guard of the bit-identical GroupConvBlock refactor (design 4.2).
`GroupConvBlock._split_call` now delegates to `pre` and `conv_with_group_mean`; its outputs must
equal a verbatim copy of the pre-refactor body bitwise (float32 and mixed_float16, fp16_act_clip 0
and 100 with the clip active, strides 1 and 2). `conv_with_group_mean(pre(x), mean(pre(x)))`
reproduces the block's own call: bitwise with split on, within float rounding with split off (the
concat conv is different arithmetic); the stem-like block (pre_activation=False) has an identity
`pre`.
"""
import sys
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory limit)
from tests.runtime_helpers import free_memory
import numpy as np
import tensorflow as tf

from models import layers as L
from models.layers import clip_fp16
from models.precision import policy_scope


def _legacy_split_call(block, x, training):
    """Verbatim copy of `GroupConvBlock._split_call` at commit 95e58c8 (models/layers.py
    691-704), with ``self`` -> ``block``; reads only block._pre, block.conv (kernel, bias),
    block.strides and block.fp16_act_clip."""
    x = block._pre(x, training, concat=False)
    channels = int(x.shape[-1])
    mean = tf.reduce_mean(x, axis=0, keepdims=True)
    k = tf.cast(block.conv.kernel, x.dtype)
    strides = [1, block.strides, block.strides, block.strides, 1]
    c = block.fp16_act_clip
    y = (clip_fp16(tf.nn.conv3d(x, k[..., :channels, :], strides, "SAME"), c)
         + clip_fp16(tf.nn.conv3d(mean, k[..., channels:, :], strides, "SAME"), c))
    return clip_fp16(tf.nn.bias_add(y, tf.cast(block.conv.bias, x.dtype)), c)


SHAPE = (3, 8, 8, 8, 4)
SCALE = 100.0  # input scale: under float16 some pre-clip values exceed 100 (asserted)


def _input(seed=0):
    rng = np.random.default_rng(seed)
    return (SCALE * rng.normal(size=SHAPE)).astype(np.float32)


def _block(policy, strides, clip, pre_activation=True, filters=6):
    """Built GroupConvBlock (group=True, slope 0.01, seed 0) under ``policy``; the first call
    takes the standard concat path and builds the conv."""
    with policy_scope(policy):
        blk = L.GroupConvBlock(filters, strides=strides, group=True, negative_slope=0.01,
                               pre_activation=pre_activation, seed=0)
    x = tf.cast(tf.constant(_input()), blk.compute_dtype)
    blk(x, training=False)
    blk.fp16_act_clip = float(clip)
    return blk, x


def _pre_clip_max(block, x):
    """Largest |value| among the unclipped conv terms and their bias-added sum."""
    p = block.pre(x, False)
    channels = int(p.shape[-1])
    mean = tf.reduce_mean(p, axis=0, keepdims=True)
    k = tf.cast(block.conv.kernel, p.dtype)
    s = [1, block.strides, block.strides, block.strides, 1]
    a = tf.nn.conv3d(p, k[..., :channels, :], s, "SAME")
    b = tf.nn.conv3d(mean, k[..., channels:, :], s, "SAME")
    y = tf.nn.bias_add(a + b, tf.cast(block.conv.bias, p.dtype))
    return max(float(tf.reduce_max(tf.abs(tf.cast(t, tf.float32)))) for t in (a, b, y))


class TestRefactorGuard(unittest.TestCase):
    """Legacy vs refactored `_split_call`: bitwise equal in all 8 combinations."""

    def tearDown(self):
        free_memory()

    def test_bitwise_all_combinations(self):
        n = 0
        for policy in ("float32", "mixed_float16"):
            for clip in (0.0, 100.0):
                for strides in (1, 2):
                    blk, x = _block(policy, strides, clip)
                    blk.split_group = True
                    if policy == "mixed_float16" and clip > 0:
                        m = _pre_clip_max(blk, x)
                        print("\n[refactor guard] fp16 stride %d max |pre-clip| %.1f"
                              % (strides, m), file=sys.stderr)
                        self.assertGreater(m, clip)  # the clip is active
                    ref = _legacy_split_call(blk, x, False)
                    got = blk._split_call(x, False)
                    self.assertEqual(got.dtype, ref.dtype)
                    self.assertEqual(got.dtype, tf.as_dtype(blk.compute_dtype))
                    np.testing.assert_array_equal(got.numpy(), ref.numpy(),
                                                  err_msg="%s clip %g stride %d"
                                                  % (policy, clip, strides))
                    if clip > 0 and policy == "mixed_float16":
                        self.assertLessEqual(float(tf.reduce_max(tf.abs(got))), clip)
                    n += 1
        self.assertEqual(n, 8)
        print("\n[refactor guard] %d combinations bitwise equal" % n, file=sys.stderr)


class TestConvWithGroupMean(unittest.TestCase):
    """`conv_with_group_mean(pre(x), mean(pre(x)), inner_clip=split_group)` vs `block(x)`."""

    def tearDown(self):
        free_memory()

    def _engine(self, blk, x):
        p = blk.pre(x, False)
        mean = tf.reduce_mean(p, axis=0, keepdims=True)
        return blk.conv_with_group_mean(p, mean, inner_clip=blk.split_group)

    def test_split_on_bitwise(self):
        for policy in ("float32", "mixed_float16"):
            for clip in (0.0, 100.0):
                for strides in (1, 2):
                    blk, x = _block(policy, strides, clip)
                    blk.split_group = True
                    np.testing.assert_array_equal(self._engine(blk, x).numpy(),
                                                  blk(x, training=False).numpy(),
                                                  err_msg="%s clip %g stride %d"
                                                  % (policy, clip, strides))

    def test_split_off_close_float32(self):
        worst = 0.0
        for strides in (1, 2):
            blk, x = _block("float32", strides, 0.0)
            self.assertFalse(blk.split_group)
            got = self._engine(blk, x).numpy()
            ref = blk(x, training=False).numpy()
            worst = max(worst, float(np.max(np.abs(got - ref)) / np.max(np.abs(ref))))
            np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5 * float(np.max(np.abs(ref))))
        print("\n[conv_with_group_mean] split off float32 max |diff| / max |ref| %.2e" % worst,
              file=sys.stderr)

    def test_stem_pre_identity(self):
        for strides in (1, 2):
            blk, x = _block("float32", strides, 0.0, pre_activation=False, filters=4)
            self.assertIsNone(blk.norm)
            self.assertIsNone(blk.act)
            np.testing.assert_array_equal(blk.pre(x, False).numpy(), x.numpy())
            got = self._engine(blk, x).numpy()
            ref = blk(x, training=False).numpy()
            np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5 * float(np.max(np.abs(ref))))


if __name__ == "__main__":
    unittest.main()
