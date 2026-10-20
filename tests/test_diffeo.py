"""Displacement fields, composition, SVF integration and Jacobians (models.diffeo, milestone 6).

numpy/scipy references; conventions in the models.diffeo docstring (plan section 0).
"""
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter, map_coordinates

from models import diffeo as D
from models import layers as L


def smooth_field(shape, rng, n=1, sigma=3.0, max_vox=2.0):
    """[n, X, Y, Z, 3] Gaussian-filtered noise scaled so max_voxel |v| = max_vox (float64)."""
    v = np.stack([np.stack([gaussian_filter(rng.normal(size=shape), sigma, mode="nearest")
                            for _ in range(3)], axis=-1) for _ in range(n)])
    return v * (max_vox / np.linalg.norm(v, axis=-1).max())


def grid(shape):
    return np.stack(np.meshgrid(*[np.arange(s, dtype=np.float64) for s in shape], indexing="ij"),
                    axis=-1)


def interior(a, margin):
    """Crop the spatial axes 1..3 of [N, X, Y, Z, ...] by margin."""
    s = slice(margin, -margin) if margin else slice(None)
    return a[:, s, s, s]


def scipy_sample(vol, coords, order=1, mode="nearest"):
    """vol [X, Y, Z], coords [X', Y', Z', 3] -> [X', Y', Z']."""
    return map_coordinates(vol.astype(np.float64), np.moveaxis(coords, -1, 0), order=order,
                           mode=mode, cval=0.0)


def norm(u):
    return np.linalg.norm(np.asarray(u), axis=-1)


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_constant_velocity_is_translation(self):
        shape = (10, 12, 8)
        c = np.array([[0.7, -1.2, 2.5], [-3.1, 0.4, 0.0]], np.float32)
        v = np.broadcast_to(c[:, None, None, None, :], (2,) + shape + (3,)).astype(np.float32)
        u, u_inv = D.exp_and_inverse(tf.constant(v), steps=7)
        np.testing.assert_allclose(u.numpy(), v, atol=1e-6, rtol=0)
        np.testing.assert_allclose(u_inv.numpy(), -v, atol=1e-6, rtol=0)
        np.testing.assert_array_equal(D.integrate_svf(v, steps=0).numpy(), v)

    def test_exp_inverse_round_trip(self):
        shape = (40, 56, 48)
        v = smooth_field(shape, self.rng, n=2, sigma=3.0, max_vox=2.0).astype(np.float32)
        u, u_inv = D.exp_and_inverse(tf.constant(v), steps=7)
        e1 = interior(norm(D.compose(u, u_inv)), 3)
        e2 = interior(norm(D.compose(u_inv, u)), 3)
        print("\nexp(v) o exp(-v) round trip (40x56x48, sigma 3, max|v| 2 vox, T=7, 3-voxel "
              "margin): u o u_inv max %.5f mean %.6f vox; u_inv o u max %.5f mean %.6f vox; "
              "max|u| %.3f" % (e1.max(), e1.mean(), e2.max(), e2.mean(), norm(u).max()))
        for e in (e1, e2):
            self.assertLess(e.max(), 0.1)
            self.assertLess(e.mean(), 0.01)
        self.assertGreater(norm(u).max(), 1.0)                       # non-trivial deformation

    def test_gradients_finite(self):
        v = tf.Variable(smooth_field((12, 10, 8), self.rng, n=2, max_vox=1.5).astype(np.float32))
        img = tf.constant(self.rng.uniform(size=(2, 12, 10, 8, 1)).astype(np.float32))
        with tf.GradientTape() as tape:
            u, u_inv = D.exp_and_inverse(v, steps=7)
            loss = tf.reduce_sum(tf.square(D.warp(img, u))) + tf.reduce_sum(tf.square(u_inv))
        g = tape.gradient(loss, v)
        self.assertIsNotNone(g)
        g = g.numpy()
        self.assertTrue(np.all(np.isfinite(g)))
        self.assertGreater(np.abs(g).max(), 0.0)

    def test_centering_mean_zero(self):
        v = self.rng.normal(size=(4, 6, 7, 5, 3)).astype(np.float32) + 3.0
        c = D.center_velocities(tf.constant(v)).numpy()
        self.assertLess(np.abs(c.mean(axis=0)).max(), 1e-6)
        np.testing.assert_allclose(c[1] - c[0], v[1] - v[0], atol=1e-5)
        np.testing.assert_allclose(c[3] - c[2], v[3] - v[2], atol=1e-5)


class TestWarpCompose(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_compose_matches_scipy(self):
        shape = (14, 16, 12)
        u_a = smooth_field(shape, self.rng, n=2, sigma=2.0, max_vox=3.0)
        u_b = smooth_field(shape, self.rng, n=2, sigma=2.0, max_vox=3.0)
        got = D.compose(tf.constant(u_a, tf.float32), tf.constant(u_b, tf.float32)).numpy()
        x = grid(shape)
        for i in range(2):
            ref = u_b[i] + np.stack([scipy_sample(u_a[i, ..., c], x + u_b[i]) for c in range(3)],
                                    axis=-1)
            np.testing.assert_allclose(got[i], ref, atol=1e-5)

    def test_warp_matches_scipy(self):
        shape = (14, 16, 12)
        vol = self.rng.uniform(size=(2,) + shape + (2,)).astype(np.float32)
        u = smooth_field(shape, self.rng, n=2, sigma=2.0, max_vox=4.0)
        x = grid(shape)
        lin = D.warp(vol, tf.constant(u, tf.float32)).numpy()
        near = D.warp(vol, tf.constant(u, tf.float32), interpolation="nearest").numpy()
        for i in range(2):
            coords = x + u[i]
            inside = np.all((coords >= 0) & (coords <= np.array(shape) - 1), axis=-1)
            self.assertGreater(inside.sum(), 1000)
            for c in range(2):
                ref = scipy_sample(vol[i, ..., c], coords, order=1, mode="grid-constant")
                np.testing.assert_allclose(lin[i, ..., c], ref, atol=2e-5)  # incl. the fall-off
                ref0 = scipy_sample(vol[i, ..., c], coords, order=0, mode="nearest")
                np.testing.assert_array_equal(near[i, ..., c][inside], ref0[inside])

    def test_nearest_labels_keep_dtype(self):
        labels = self.rng.integers(0, 5, size=(1, 8, 9, 7, 1)).astype(np.uint8)
        u = tf.constant(np.full((1, 8, 9, 7, 3), 1.0, np.float32))
        out = D.warp(labels, u, interpolation="nearest")
        self.assertEqual(out.dtype, tf.uint8)
        np.testing.assert_array_equal(out.numpy()[0, :-1, :-1, :-1], labels[0, 1:, 1:, 1:])

    def test_border_keeps_constant_field(self):
        shape = (8, 9, 7)
        field = np.broadcast_to(np.array([1.5, -2.0, 0.25], np.float32), (1,) + shape + (3,))
        u = smooth_field(shape, self.rng, sigma=1.0, max_vox=6.0).astype(np.float32)
        np.testing.assert_allclose(D.warp_field(field, u).numpy(), field, atol=1e-6)

    def test_pairwise_displacement(self):
        shape = (12, 10, 8)
        v = smooth_field(shape, self.rng, n=3, max_vox=1.0).astype(np.float32)
        u, u_inv = D.exp_and_inverse(v)
        got = D.pairwise_displacement(u, u_inv, [0, 2], [1, 0]).numpy()
        np.testing.assert_allclose(got[0], D.compose(u[:1], u_inv[1:2]).numpy()[0], atol=1e-6)
        np.testing.assert_allclose(got[1], D.compose(u[2:3], u_inv[:1]).numpy()[0], atol=1e-6)
        self.assertEqual(D.pairwise_displacement(u, u_inv, 1, 1).shape, (1,) + shape + (3,))
        self.assertLess(interior(norm(D.pairwise_displacement(u, u_inv, 1, 1)), 2).max(), 0.05)

    def test_to_mm(self):
        u = np.ones((1, 2, 3, 4, 3), np.float32)
        np.testing.assert_allclose(D.to_mm(u, 4.0).numpy(), 4.0 * u)
        np.testing.assert_allclose(D.to_mm(u, (1.0, 2.0, 3.0)).numpy()[0, 0, 0, 0], [1, 2, 3])


class TestJacobian(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)

    def test_jacobian_affine_field(self):
        shape = (9, 10, 8)
        m = np.eye(3) + self.rng.uniform(-0.2, 0.2, (3, 3))
        b = np.array([0.5, -1.0, 2.0])
        u = (grid(shape) @ (m - np.eye(3)).T + b)[None].astype(np.float32)
        jac = D.jacobian_determinant(u).numpy()
        self.assertEqual(jac.shape, (1, 7, 8, 6))
        np.testing.assert_allclose(jac, np.linalg.det(m), atol=1e-5)

    def test_jacobian_vs_np_gradient(self):
        shape = (12, 11, 10)
        u = smooth_field(shape, self.rng, n=2, sigma=1.5, max_vox=3.0)
        jac = D.jacobian_determinant(tf.constant(u, tf.float32)).numpy()
        for i in range(2):
            g = np.stack([np.stack(np.gradient(u[i, ..., c]), axis=-1) for c in range(3)], axis=-2)
            ref = np.linalg.det(np.eye(3) + g)[1:-1, 1:-1, 1:-1]        # [..., c, a]
            np.testing.assert_allclose(jac[i], ref, atol=1e-5)

    def test_folding_fraction(self):
        shape = (32, 10, 8)
        x0 = grid(shape)[..., 0]
        u = np.zeros((2,) + shape + (3,), np.float32)
        u[0, ..., 0] = -1.5 * np.clip(x0, 10, 20)          # J = -0.5 at x0 = 11..19, 0.25 at 10, 20
        frac = D.folding_fraction(u).numpy()
        np.testing.assert_allclose(frac, [9.0 / 30.0, 0.0], atol=1e-6)
        mask = ((x0 >= 11) & (x0 <= 19))[None]
        np.testing.assert_allclose(D.folding_fraction(u, mask=mask).numpy(), [1.0, 0.0], atol=1e-6)
        jac = D.jacobian_determinant(u).numpy()
        np.testing.assert_allclose(jac[0, 10:19], -0.5, atol=1e-6)    # interior index = x0 - 1
        np.testing.assert_allclose(jac[0, 9], 0.25, atol=1e-6)


class TestUpsampleAndCascade(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(4)

    def test_upsample_displacement_consistency(self):
        coarse = (10, 12, 9)
        m = self.rng.uniform(-0.05, 0.05, (3, 3))
        b = np.array([0.4, -0.6, 0.3])
        u_c = (grid(coarse) @ m.T + b)[None]
        up = D.upsample_displacement(tf.constant(u_c, tf.float32)).numpy()
        p = grid(tuple(2 * n for n in coarse))
        c = (p - 0.5) / 2.0
        ref = 2.0 * (c @ m.T + b)
        np.testing.assert_allclose(interior(up, 1), interior(ref[None], 1), atol=1e-5)

        # Warping consistency: fine warp with the upsampled field vs upsampled coarse warp.
        fine_shape = (40, 48, 36)
        coarse_shape = tuple(n // 2 for n in fine_shape)

        def f(pts):                                     # continuous image in fine coordinates
            return (np.sin(2 * np.pi * pts[..., 0] / 40.0) * np.cos(2 * np.pi * pts[..., 1] / 50.0)
                    + 0.5 * np.sin(2 * np.pi * pts[..., 2] / 30.0))

        img_f = f(grid(fine_shape))[None, ..., None].astype(np.float32)
        img_c = f(2.0 * grid(coarse_shape) + 0.5)[None, ..., None].astype(np.float32)
        v_c = smooth_field(coarse_shape, self.rng, sigma=3.0, max_vox=1.5).astype(np.float32)
        u_c2 = D.integrate_svf(v_c)
        fine_warp = D.warp(img_f, D.upsample_displacement(u_c2)).numpy()
        coarse_up = L.upsample_trilinear2x(D.warp(img_c, u_c2)).numpy()
        err = np.abs(interior(fine_warp, 6) - interior(coarse_up, 6))
        print("\nupsampled-field warp vs upsampled coarse warp: max %.4f mean %.5f (range 3)"
              % (err.max(), err.mean()))
        self.assertLess(err.max(), 0.06)

    def test_cascade_inverse(self):
        coarse, fine = (20, 28, 24), (40, 56, 48)
        v1 = smooth_field(coarse, self.rng, sigma=3.0, max_vox=1.0).astype(np.float32)
        v2 = smooth_field(fine, self.rng, sigma=3.0, max_vox=1.5).astype(np.float32)
        u1, u1_inv = D.exp_and_inverse(v1)
        u2, u2_inv = D.exp_and_inverse(v2)
        u, u_inv = D.compose_cascade(D.upsample_displacement(u1), D.upsample_displacement(u1_inv),
                                     u2, u2_inv)
        e1 = interior(norm(D.compose(u, u_inv)), 4)
        e2 = interior(norm(D.compose(u_inv, u)), 4)
        print("\ncascade round trip (20x28x24 -> 40x56x48, 4-voxel margin): u o u_inv max %.5f "
              "mean %.6f; u_inv o u max %.5f mean %.6f vox" % (e1.max(), e1.mean(), e2.max(),
                                                              e2.mean()))
        for e in (e1, e2):                                  # same bounds as the round trip
            self.assertLess(e.max(), 0.1)
            self.assertLess(e.mean(), 0.01)
        # One factor (zero previous level) is the single-level map.
        z = tf.zeros_like(u2)
        a, ai = D.compose_cascade(z, z, u2, u2_inv)
        np.testing.assert_allclose(a.numpy(), u2.numpy(), atol=1e-6)
        np.testing.assert_allclose(ai.numpy(), u2_inv.numpy(), atol=1e-6)


class TestDtypes(unittest.TestCase):
    def test_dtypes(self):
        rng = np.random.default_rng(5)
        v = smooth_field((8, 9, 7), rng, max_vox=1.0)
        u16, inv16 = D.exp_and_inverse(tf.constant(v, tf.float16))
        self.assertEqual((u16.dtype, inv16.dtype), (tf.float32, tf.float32))
        u64, _ = D.exp_and_inverse(tf.constant(v, tf.float64))
        self.assertEqual(u64.dtype, tf.float64)
        u32, _ = D.exp_and_inverse(tf.constant(v, tf.float32))
        np.testing.assert_allclose(u64.numpy(), u32.numpy(), atol=1e-5)
        self.assertEqual(D.center_velocities(tf.constant(v, tf.float16)).dtype, tf.float32)
        self.assertEqual(D.upsample_displacement(tf.constant(v, tf.float16)).dtype, tf.float32)
        self.assertEqual(D.jacobian_determinant(tf.constant(v, tf.float16)).dtype, tf.float32)
        img = tf.constant(rng.uniform(size=(1, 8, 9, 7, 1)), tf.float16)
        self.assertEqual(D.warp(img, u32).dtype, tf.float32)
        self.assertEqual(D.compose(u32, u64).dtype, tf.float64)


if __name__ == "__main__":
    unittest.main()
