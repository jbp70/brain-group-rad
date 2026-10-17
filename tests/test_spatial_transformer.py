"""Tests for models.layers.transform_image (pull warp in centered mm coordinates)."""
import unittest

import numpy as np

from tests import tf_setup  # noqa: F401  (log level, GPU memory growth)
from tests import np_reference as npr
import tensorflow as tf
from scipy.ndimage import map_coordinates

from models import layers
from models import transforms as tfm


def small_affine(rng, max_deg=20.0, max_trans=4.0, scale=0.0, shear=0.0):
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    from scipy.linalg import expm
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    m = np.eye(4)
    m[:3, :3] = expm(k * np.deg2rad(rng.uniform(-max_deg, max_deg)))
    kk = np.diag(1.0 + rng.uniform(-scale, scale, 3)) + np.triu(rng.uniform(-shear, shear, (3, 3)), 1)
    m[:3, :3] = m[:3, :3] @ kk
    m[:3, 3] = rng.uniform(-max_trans, max_trans, 3)
    return m


def sample_coords(m_mm, shape, spacing):
    """Input voxel coordinates [3, X, Y, Z] that out(y) samples, via the numpy reference."""
    mv = npr.voxel_matrix(m_mm, shape, spacing)
    grid = np.stack(np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij"))
    return np.einsum("ij,j...->i...", mv[:3, :3], grid) + mv[:3, 3].reshape(3, 1, 1, 1)


def interior(shape, margin):
    m = np.zeros(shape, bool)
    m[margin:shape[0] - margin, margin:shape[1] - margin, margin:shape[2] - margin] = True
    return m


class TestTransformImage(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(10)

    def test_a_matches_scipy_map_coordinates(self):
        shape, spacing = (14, 16, 12), 2.0
        vol = self.rng.uniform(size=(3,) + shape + (2,)).astype(np.float32)
        mats = np.stack([small_affine(self.rng, 30.0, 6.0, 0.1, 0.05) for _ in range(3)])
        out = layers.transform_image(vol, mats.astype(np.float32), spacing).numpy()
        checked = 0
        for i in range(3):
            coords = sample_coords(mats[i], shape, spacing)
            inside = np.all([(coords[a] >= 0) & (coords[a] <= shape[a] - 1) for a in range(3)], axis=0)
            far = np.any([(coords[a] < -1) | (coords[a] > shape[a]) for a in range(3)], axis=0)
            for ch in range(2):
                ref = map_coordinates(vol[i, ..., ch].astype(np.float64), coords, order=1,
                                      mode="constant", cval=0.0)
                np.testing.assert_allclose(out[i, ..., ch][inside], ref[inside], atol=2e-5)
                np.testing.assert_array_equal(out[i, ..., ch][far], 0.0)
            checked += inside.sum()
        self.assertGreater(checked, 1000)

    def test_b_integer_translation_shifts_exactly(self):
        shape, spacing = (10, 12, 8), 2.0
        vol = self.rng.uniform(size=(1,) + shape + (1,)).astype(np.float32)
        k = np.array([2, -3, 1])
        m = tfm.translation_matrix((k * spacing).astype(np.float32))[None]
        out = layers.transform_image(vol, m, spacing).numpy()[0, ..., 0]
        ref = np.zeros(shape, np.float32)
        ref[0:shape[0] - 2, 3:, 0:shape[2] - 1] = vol[0, 2:, :shape[1] - 3, 1:, 0]
        np.testing.assert_array_equal(out, ref)
        # The same shift as a displacement field in voxels, with an identity matrix.
        disp = np.broadcast_to(k.astype(np.float32), (1,) + shape + (3,))
        out_d = layers.transform_image(vol, np.eye(4, dtype=np.float32)[None], spacing,
                                       displacement=disp).numpy()[0, ..., 0]
        np.testing.assert_array_equal(out_d, ref)

    def test_c_round_trip(self):
        shape, spacing = (32, 36, 28), 1.0
        vol = npr.smooth_phantom(shape, self.rng, sigma=3.0).astype(np.float32)[None, ..., None]
        m = small_affine(self.rng, 20.0, 3.0, 0.05, 0.02)
        m_inv = np.linalg.inv(m)
        once = layers.transform_image(vol, m[None].astype(np.float32), spacing)
        twice = layers.transform_image(once, m_inv[None].astype(np.float32), spacing).numpy()
        inner = interior(shape, 8)
        err = np.sqrt(np.mean((twice[0, ..., 0] - vol[0, ..., 0])[inner] ** 2))
        # "0.5 voxel equivalent": RMS change of the phantom under a 0.5-voxel shift along each axis
        half = []
        for a in range(3):
            t = np.zeros(3, np.float32)
            t[a] = 0.5 * spacing
            sh = layers.transform_image(vol, tfm.translation_matrix(t)[None], spacing).numpy()
            half.append(np.sqrt(np.mean((sh[0, ..., 0] - vol[0, ..., 0])[inner] ** 2)))
        self.assertLess(err, min(half), "round-trip RMS {} vs 0.5-voxel RMS {}".format(err, half))
        # Composed matrix applied once is identity-close.
        comp = tfm.compose(m_inv.astype(np.float32), m.astype(np.float32)).numpy()
        np.testing.assert_allclose(comp, np.eye(4), atol=1e-5)
        once_id = layers.transform_image(vol, comp[None], spacing).numpy()
        np.testing.assert_allclose(once_id[0, ..., 0][inner], vol[0, ..., 0][inner], atol=1e-4)

    def test_d_nearest_preserves_labels(self):
        shape, spacing = (12, 14, 10), 2.0
        labels = self.rng.choice(np.array([0, 2, 5, 7, 18], np.uint8), size=(2,) + shape + (1,))
        mats = np.stack([small_affine(self.rng, 30.0, 5.0, 0.1, 0.05) for _ in range(2)]).astype(np.float32)
        out = layers.transform_image(labels, mats, spacing, interpolation="nearest")
        self.assertEqual(out.dtype, tf.uint8)
        self.assertTrue(set(np.unique(out.numpy())) <= {0, 2, 5, 7, 18})
        # identity -> unchanged; float labels stay integral too
        ident = np.broadcast_to(np.eye(4, dtype=np.float32), (2, 4, 4))
        np.testing.assert_array_equal(
            layers.transform_image(labels, ident, spacing, interpolation="nearest").numpy(), labels)
        out_f = layers.transform_image(labels.astype(np.float32), mats, spacing, interpolation="nearest").numpy()
        np.testing.assert_array_equal(out_f, out.numpy().astype(np.float32))

    def test_e_cpu_out_of_bounds(self):
        shape, spacing = (10, 12, 8), 4.0
        vol = (1.0 + self.rng.uniform(size=(2,) + shape + (1,))).astype(np.float32)
        far = np.eye(4, dtype=np.float32)
        far[:3, 3] = [1000.0, -1000.0, 1e7]
        half = np.eye(4, dtype=np.float32)
        half[0, 3] = 0.5 * shape[0] * spacing    # half the volume pushed out along x
        with tf.device("/CPU:0"):
            out = layers.transform_image(vol, np.stack([far, half]), spacing).numpy()
            out_nn = layers.transform_image(vol, np.stack([far, half]), spacing,
                                            interpolation="nearest").numpy()
            out_fill = layers.transform_image(vol, np.stack([far, half]), spacing, fill_value=-1.0).numpy()
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_array_equal(out[0], 0.0)
        np.testing.assert_array_equal(out_nn[0], 0.0)
        np.testing.assert_array_equal(out_fill[0], -1.0)
        self.assertTrue(np.all(out[1, shape[0] // 2 + 1:] == 0.0))
        self.assertTrue(np.all(out[1, :shape[0] // 2 - 1] > 0.5))

    def test_f_gradient_matches_finite_difference(self):
        shape, spacing = (16, 18, 14), 2.0
        vol = tf.constant(npr.smooth_phantom(shape, self.rng, sigma=3.0)[None, ..., None])
        m = tf.constant(small_affine(self.rng, 15.0, 3.0, 0.05, 0.02)[None])
        w = tf.constant(self.rng.normal(size=(1,) + shape + (1,)))
        v = np.zeros((1, 4, 4))
        v[:, :3] = self.rng.normal(size=(1, 3, 4)) * np.array([0.01, 0.01, 0.01, 0.5])
        v = tf.constant(v)

        def f(mat):
            return tf.reduce_sum(w * layers.transform_image(vol, mat, spacing))

        with tf.GradientTape() as tape:
            tape.watch(m)
            y = f(m)
        g = tape.gradient(y, m)
        self.assertIsNotNone(g)
        ad = float(tf.reduce_sum(g * v))
        eps = 1e-5
        fd = float((f(m + eps * v) - f(m - eps * v)) / (2 * eps))
        self.assertLess(abs(ad - fd) / abs(fd), 1e-4, "autodiff {} vs FD {}".format(ad, fd))
        # float32 matrices, displacement gradient
        disp = tf.zeros((1,) + shape + (3,), tf.float32)
        m32 = tf.cast(m, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch([m32, disp])
            y = tf.reduce_sum(tf.cast(w, tf.float32) *
                              layers.transform_image(tf.cast(vol, tf.float32), m32, spacing, displacement=disp))
        gm, gd = tape.gradient(y, [m32, disp])
        self.assertIsNotNone(gm)
        self.assertIsNotNone(gd)
        self.assertGreater(float(tf.reduce_max(tf.abs(gd))), 0.0)

    def test_g_dynamic_group_size_in_tf_function(self):
        shape, spacing = (8, 10, 6), 4.0
        traces = []

        @tf.function(input_signature=[tf.TensorSpec([None] + list(shape) + [1], tf.float32),
                                      tf.TensorSpec([None, 4, 4], tf.float32)])
        def warp_and_center(images, mats):
            traces.append(1)
            centered = tfm.center_group(mats)
            return layers.transform_image(images, tfm.affine_inverse(centered), spacing), centered

        demo = npr.load_demo()
        for n in (2, 3, 5):
            vol = self.rng.uniform(size=(n,) + shape + (1,)).astype(np.float32)
            mats = np.stack([demo.random_rigid(20.0, 5.0) for _ in range(n)]).astype(np.float32)
            out, centered = warp_and_center(vol, mats)
            self.assertEqual(out.shape, (n,) + shape + (1,))
            self.assertEqual(centered.shape, (n, 4, 4))
        self.assertEqual(len(traces), 1)

    def test_h_mixed_float16_policy(self):
        prev = tf.keras.mixed_precision.global_policy()
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            demo = npr.load_demo()
            shape, spacing = (12, 14, 10), 2.0
            raw = np.stack([demo.random_rigid() for _ in range(4)]).astype(np.float32)
            layer = tfm.GroupCentering()
            self.assertEqual(layer.compute_dtype, "float32")
            centered = layer(tf.constant(raw))
            self.assertEqual(centered.dtype, tf.float32)
            ref = tfm.center_group(tf.constant(raw.astype(np.float64)), 4).numpy()
            np.testing.assert_allclose(centered.numpy(), ref, atol=1e-3)
            # float16 head output -> float32 matrices
            params16 = tf.cast(tf.constant(self.rng.normal(size=(4, 19))), tf.float16)
            self.assertEqual(tfm.affine_matrix(params16).dtype, tf.float32)
            # float16 images (and float16 matrices) warp with a float32 grid; output stays float16
            vol = self.rng.uniform(size=(4,) + shape + (1,)).astype(np.float32)
            mats = tfm.affine_inverse(centered)
            out16 = layers.transform_image(tf.cast(vol, tf.float16), tf.cast(mats, tf.float16), spacing)
            out32 = layers.transform_image(vol, mats, spacing)
            self.assertEqual(out16.dtype, tf.float16)
            # differences come only from float16 storage of images, matrices and output
            np.testing.assert_allclose(out16.numpy().astype(np.float32), out32.numpy(), atol=2e-2)
            out16b = layers.transform_image(tf.cast(vol, tf.float16), mats, spacing)
            np.testing.assert_allclose(out16b.numpy().astype(np.float32), out32.numpy(), atol=2e-3)
        finally:
            tf.keras.mixed_precision.set_global_policy(prev)
        self.assertEqual(tf.keras.mixed_precision.global_policy().name, prev.name)

    def test_i_multilevel_consistency(self):
        shape_f = (48, 56, 40)
        vol_f = npr.smooth_phantom(shape_f, self.rng, n_blobs=8, sigma=5.0).astype(np.float32)
        vol_c = npr.block_average(vol_f, 2)
        m = small_affine(self.rng, 20.0, 4.0, 0.05, 0.02).astype(np.float32)
        ident = np.eye(4, dtype=np.float32)[None]
        # identity at both levels: block averaging commutes exactly
        np.testing.assert_allclose(
            npr.block_average(layers.transform_image(vol_f[None, ..., None], ident, 1.0).numpy()[0, ..., 0], 2),
            layers.transform_image(vol_c[None, ..., None], ident, 2.0).numpy()[0, ..., 0], atol=1e-6)
        out_f = layers.transform_image(vol_f[None, ..., None], m[None], 1.0).numpy()[0, ..., 0]
        ref = npr.block_average(out_f, 2)
        out_c = layers.transform_image(vol_c[None, ..., None], m[None], 2.0).numpy()[0, ..., 0]
        inner = interior(out_c.shape, 4)
        diff = (ref - out_c)[inner]
        rms, err_max = np.sqrt(np.mean(diff ** 2)), np.abs(diff).max()
        # Same check with the coarse matrix off by 0.5 mm (a quarter coarse voxel) on each axis:
        # it must be clearly worse, so the test is sensitive to center / spacing errors. The
        # remaining error is coarse-grid linear interpolation error (~h^2 f'' / 8).
        shift = np.eye(4, dtype=np.float32)
        shift[:3, 3] = 0.5
        out_bad = layers.transform_image(vol_c[None, ..., None], (m @ shift)[None], 2.0).numpy()[0, ..., 0]
        rms_bad = np.sqrt(np.mean((ref - out_bad)[inner] ** 2))
        rng_val = vol_f.max() - vol_f.min()
        self.assertLess(rms, 0.01 * rng_val, "rms {} range {}".format(rms, rng_val))
        self.assertLess(err_max, 0.05 * rng_val, "max {} range {}".format(err_max, rng_val))
        self.assertLess(rms, 0.25 * rms_bad, "rms {} vs 0.5 mm shifted {}".format(rms, rms_bad))

    def test_flip_commutes_with_conjugated_warp(self):
        shape, spacing = (16, 20, 14), 2.0
        vol = npr.smooth_phantom(shape, self.rng, sigma=3.0).astype(np.float32)[None, ..., None]
        t = small_affine(self.rng, 20.0, 4.0, 0.05, 0.02)
        for axes in [(True, False, False), (False, True, True), (True, True, True)]:
            f = tfm.flip_matrix(axes, tf.float64)
            # render with T (pull with T^-1), then flip
            a = tfm.flip_images(layers.transform_image(vol, np.linalg.inv(t)[None].astype(np.float32),
                                                       spacing), axes).numpy()
            # flip first, then render with F T F
            ftf = tfm.flip_conjugate(tf.constant(t[None]), f)
            b = layers.transform_image(tfm.flip_images(vol, axes),
                                       tf.cast(tfm.affine_inverse(ftf), tf.float32), spacing).numpy()
            inner = interior(shape, 3)
            self.assertLess(np.abs(a - b)[0, ..., 0][inner].max(), 1e-5, str(axes))


if __name__ == "__main__":
    unittest.main()
