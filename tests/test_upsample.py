"""Trilinear x2 upsampling (models.layers.upsample_trilinear2x, half-pixel semantics; milestone 6)."""
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
import numpy as np
import scipy
import tensorflow as tf
from scipy.ndimage import map_coordinates, zoom

from models import layers as L


def scipy_version():
    return tuple(int(v) for v in scipy.__version__.split(".")[:2])


def reference(x):
    """map_coordinates(order=1, mode='nearest') at coarse coordinate (p - 0.5) / 2 per axis."""
    shape = x.shape
    fine = [np.arange(2 * n, dtype=np.float64) for n in shape]
    coords = np.stack(np.meshgrid(*[(p - 0.5) / 2.0 for p in fine], indexing="ij"))
    return map_coordinates(x.astype(np.float64), coords, order=1, mode="nearest")


def up(x):
    """numpy [X, Y, Z(, C)] -> upsampled numpy via the TF op (batch of 1)."""
    t = x if x.ndim == 4 else x[..., None]
    out = L.upsample_trilinear2x(tf.constant(t[None].astype(np.float32))).numpy()[0]
    return out if x.ndim == 4 else out[..., 0]


class TestUpsampleTrilinear2x(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)

    def test_matches_map_coordinates(self):
        x = self.rng.normal(size=(5, 7, 6))
        np.testing.assert_allclose(up(x), reference(x), atol=1e-6)

    def test_multichannel(self):
        x = self.rng.normal(size=(3, 4, 5, 2))
        out = up(x)
        self.assertEqual(out.shape, (6, 8, 10, 2))
        for c in range(2):
            np.testing.assert_allclose(out[..., c], reference(x[..., c]), atol=1e-6)

    def test_matches_zoom_grid_mode(self):
        if scipy_version() < (1, 6):
            self.skipTest("scipy < 1.6 has no zoom(grid_mode=True)")
        x = self.rng.normal(size=(5, 7, 6))
        ref = zoom(x, 2, order=1, mode="nearest", grid_mode=True)
        np.testing.assert_allclose(up(x), ref, atol=1e-6)

    def test_ramp_exact_in_interior(self):
        a, b = np.array([0.7, -1.3, 2.1]), 0.4
        shape = (6, 5, 7)
        q = np.stack(np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij"),
                     axis=-1)
        x = q @ a + b
        out = up(x)
        p = np.stack(np.meshgrid(*[np.arange(2 * n, dtype=np.float64) for n in shape],
                                 indexing="ij"), axis=-1)
        ref = ((p - 0.5) / 2.0) @ a + b
        inner = (slice(1, -1),) * 3        # edge replication differs only at the outer fine voxel
        np.testing.assert_allclose(out[inner], ref[inner], atol=1e-5)

    def test_dynamic_n_no_retrace(self):
        traces = []

        @tf.function(input_signature=[tf.TensorSpec([None, 4, 5, 3, 2], tf.float32)])
        def f(x):
            traces.append(1)
            return L.upsample_trilinear2x(x)

        for n in (2, 3):
            x = self.rng.normal(size=(n, 4, 5, 3, 2)).astype(np.float32)
            out = f(tf.constant(x)).numpy()
            self.assertEqual(out.shape, (n, 8, 10, 6, 2))
            np.testing.assert_allclose(out[n - 1, ..., 1], reference(x[n - 1, ..., 1]), atol=1e-6)
        self.assertEqual(len(traces), 1)

    def test_layer_and_gradient(self):
        layer = L.TrilinearUpsample2x()
        x = tf.constant(self.rng.normal(size=(2, 3, 4, 5, 1)).astype(np.float32))
        self.assertEqual(layer.weights, [])
        self.assertEqual(tuple(layer.compute_output_shape(x.shape)), (2, 6, 8, 10, 1))
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = tf.reduce_sum(tf.square(layer(x)))
        g = tape.gradient(loss, x)
        self.assertIsNotNone(g)
        self.assertTrue(np.all(np.isfinite(g.numpy())))
        self.assertGreater(float(tf.reduce_max(tf.abs(g))), 0.0)

    def test_float16(self):
        x = self.rng.normal(size=(1, 3, 4, 5, 2)).astype(np.float16)
        out = L.upsample_trilinear2x(tf.constant(x))
        self.assertEqual(out.dtype, tf.float16)
        np.testing.assert_allclose(out.numpy()[0, ..., 0].astype(np.float64),
                                   reference(x[0, ..., 0].astype(np.float64)), atol=1e-2)


if __name__ == "__main__":
    unittest.main()
