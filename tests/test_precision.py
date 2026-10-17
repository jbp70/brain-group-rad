"""TF32 must be off: on Ampere GPUs it corrupts float32 grid / 4x4 matrix math (models/precision.py)."""
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
import numpy as np
import tensorflow as tf

from models.precision import configure_tf_precision


class TestPrecision(unittest.TestCase):
    def test_tf32_disabled_by_test_setup(self):
        self.assertFalse(tf.config.experimental.tensor_float_32_execution_enabled())

    def test_configure_returns_state(self):
        self.assertFalse(configure_tf_precision(tf32=False))

    def test_float32_matmul_is_full_precision(self):
        # With TF32 a [64, 64] float32 matmul has ~1e-3 relative error on Ampere; full float32
        # stays ~1e-6. On CPU or pre-Ampere GPUs this passes trivially.
        rng = np.random.default_rng(0)
        a = rng.normal(size=(64, 64))
        b = rng.normal(size=(64, 64))
        ref = a @ b
        out = tf.matmul(tf.constant(a, tf.float32), tf.constant(b, tf.float32)).numpy()
        rel = np.abs(out - ref).max() / np.abs(ref).max()
        self.assertLess(rel, 1e-5)


if __name__ == "__main__":
    unittest.main()
