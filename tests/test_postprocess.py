"""Tests for data_loading.postprocess (pure TF, no DALI)."""
import os
import shutil
import tempfile
import unittest

import numpy as np

from tests import tf_setup  # noqa: F401  (log level, GPU memory growth)
from tests import np_reference as npr
import tensorflow as tf

from data_loading import params as P
from data_loading import postprocess as pp
from models import layers
from models import transforms as tfm

FLIPS = [(True, False, False), (False, True, True), (True, True, True), (False, False, False)]


def random_dali_batch(rng, task, level, n):
    """DALI-order matrices for n images via the numpy mirror of the DALI graph."""
    r = P.linear_ranges(task, level)
    t, rt = [], []
    for _ in range(n):
        m = P.linear_matrices_np(
            rng.uniform(-r.rotation_deg, r.rotation_deg), rng.normal(size=3),
            rng.uniform(-r.translation_vox, r.translation_vox, 3),
            rng.uniform(-r.log_scale, r.log_scale, 3), rng.uniform(-r.shear, r.shear, 3),
            P.LEVEL_SHAPE[level])
        t.append(m["T"][:3])
        rt.append(m["Rt"][:3])
    return np.asarray(t, np.float32), np.asarray(rt, np.float32)


def interior(shape, margin):
    m = np.zeros(shape, bool)
    m[margin:shape[0] - margin, margin:shape[1] - margin, margin:shape[2] - margin] = True
    return m


class TestFlips(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)

    def test_image_and_matrix_flip_consistent(self):
        shape, spacing = (24, 28, 20), 4.0
        vol = np.stack([npr.smooth_phantom(shape, self.rng, sigma=2.5) for _ in range(2)])
        vol = vol.astype(np.float32)[..., None]
        t = np.stack([tfm.affine_matrix(np.concatenate([
            tfm.encode_10d(tf.constant(npr_rot(self.rng, 30.0))).numpy(),
            self.rng.uniform(-6, 6, 3), self.rng.uniform(-0.05, 0.05, 6)]).astype(np.float32)).numpy()
            for _ in range(2)])
        rendered = layers.transform_image(vol, tfm.affine_inverse(t), spacing)
        inner = interior(shape, 4)
        for flags in FLIPS:
            img_f, (t_f,) = pp.apply_flip(rendered, [t], flags)
            # Flipped render == rendering the flipped template with (F T F)^-1.
            ref = layers.transform_image(tfm.flip_images(vol, flags), tfm.affine_inverse(t_f), spacing)
            np.testing.assert_allclose(img_f.numpy()[:, inner], ref.numpy()[:, inner], atol=1e-5)
            # Undoing with F T F gives the flipped undo of the unflipped batch.
            undo_f = layers.transform_image(img_f, t_f, spacing)
            undo = tfm.flip_images(layers.transform_image(rendered, t, spacing), flags)
            np.testing.assert_allclose(undo_f.numpy()[:, inner], undo.numpy()[:, inner], atol=1e-5)
            np.testing.assert_allclose(t_f.numpy(), tfm.flip_conjugate(t, tfm.flip_matrix(flags)).numpy(),
                                       atol=1e-6)

    def test_flip_sampling(self):
        g = tf.random.Generator.from_seed(5)
        self.assertFalse(np.any(pp.sample_flip_axes(g, 0.0).numpy()))
        self.assertTrue(np.all(pp.sample_flip_axes(g, 1.0).numpy()))
        self.assertEqual(list(pp.sample_flip_axes(g, (1.0, 0.0, 1.0)).numpy()), [True, False, True])
        a = [pp.sample_flip_axes(tf.random.Generator.from_seed(9)).numpy() for _ in range(2)]
        np.testing.assert_array_equal(a[0], a[1])
        g = tf.random.Generator.from_seed(11)
        counts = np.mean([pp.sample_flip_axes(g, 0.5).numpy() for _ in range(400)], axis=0)
        self.assertTrue(np.all(np.abs(counts - 0.5) < 0.1), counts)


def npr_rot(rng, max_deg):
    from scipy.linalg import expm
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return expm(k * np.deg2rad(rng.uniform(-max_deg, max_deg))).astype(np.float32)


class TestPairs(unittest.TestCase):
    def test_counts(self):
        for n, want in [(2, 1), (3, 1), (8, 6), (12, 13), (48, 226)]:
            self.assertEqual(int(pp.num_pairs(n, 0.1)), want, n)
        self.assertEqual(int(pp.num_pairs(4, 1.0)), 12)

    def test_valid_unique_deterministic(self):
        for n in (2, 5, 48):
            i, j = pp.sample_pairs(tf.random.Generator.from_seed(1), n, 0.1)
            i, j = i.numpy(), j.numpy()
            self.assertEqual(len(i), int(pp.num_pairs(n, 0.1)))
            self.assertTrue(np.all(i != j))
            self.assertTrue(np.all((i >= 0) & (i < n) & (j >= 0) & (j < n)))
            self.assertEqual(len(set(zip(i, j))), len(i))
            i2, j2 = pp.sample_pairs(tf.random.Generator.from_seed(1), n, 0.1)
            np.testing.assert_array_equal(i, i2.numpy())
            np.testing.assert_array_equal(j, j2.numpy())
        # All ordered pairs when p_pair = 1.
        i, j = pp.sample_pairs(tf.random.Generator.from_seed(2), 4, 1.0)
        self.assertEqual(sorted(zip(i.numpy(), j.numpy())),
                         sorted((a, b) for a in range(4) for b in range(4) if a != b))
        # Successive draws from one generator differ.
        g = tf.random.Generator.from_seed(3)
        a, b = pp.sample_pairs(g, 48)[0].numpy(), pp.sample_pairs(g, 48)[0].numpy()
        self.assertFalse(np.array_equal(a, b))


class TestLinearPostprocess(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(21)
        self.n = 6
        self.images = np.zeros((self.n,) + P.LEVEL_SHAPE[4] + (1,), np.float32)

    def test_dali_to_mm_matches_models_transforms(self):
        t, _ = random_dali_batch(self.rng, "affine", 4, 3)
        ref = tfm.to_homogeneous(tfm.dali_voxel_to_tf_mm(t, P.LEVEL_SHAPE[4], 4.0)).numpy()
        np.testing.assert_allclose(pp.dali_to_mm(t, 4).numpy(), ref, atol=1e-5)
        self.assertEqual(pp.dali_to_mm(t, 4).shape, (3, 4, 4))

    def test_rigid_targets_use_rt_and_are_central(self):
        t, rt = random_dali_batch(self.rng, "rigid", 4, self.n)
        post = pp.LinearPostprocess("rigid", 4, seed=4)
        out = post(self.images, t, rt, flip_axes=[False, False, False])
        rt_mm = pp.dali_to_mm(rt, 4).numpy()
        np.testing.assert_allclose(out["T_render"].numpy(), pp.dali_to_mm(t, 4).numpy(), atol=1e-5)
        np.testing.assert_allclose(out["T_target"].numpy(), rt_mm, atol=1e-5)
        ref = tfm.barycentric_targets(tf.constant(rt_mm, tf.float64)).numpy()
        np.testing.assert_allclose(out["A_star"].numpy(), ref, atol=1e-5)
        a = tf.constant(out["A_star"].numpy(), tf.float64)
        mu = tfm.barycenter(a, iters=10).numpy()
        np.testing.assert_allclose(mu, np.eye(4), atol=1e-4)
        # Rigid targets stay rigid (the nuisance is excluded).
        lin = out["A_star"].numpy()[:, :3, :3]
        np.testing.assert_allclose(lin @ np.transpose(lin, (0, 2, 1)), np.broadcast_to(np.eye(3), lin.shape),
                                   atol=1e-5)
        pred = tfm.pair_predictions(a, out["pair_i"], out["pair_j"]).numpy()
        np.testing.assert_allclose(pred, out["P_star"].numpy(), atol=1e-4)
        self.assertEqual(out["pair_i"].shape[0], int(pp.num_pairs(self.n)))
        self.assertEqual(out["images"].dtype, tf.float32)
        with self.assertRaises(ValueError):
            post(self.images, t)

    def test_affine_target_is_full_t_and_flip_equivariant(self):
        t, _ = random_dali_batch(self.rng, "affine", 4, self.n)
        post = pp.LinearPostprocess("affine", 4, seed=4)
        out0 = post(self.images, t, flip_axes=[False, False, False])
        np.testing.assert_allclose(out0["T_target"].numpy(), out0["T_render"].numpy())
        flags = [True, False, True]
        out1 = post(self.images, t, flip_axes=flags)
        f = tfm.flip_matrix(flags).numpy()
        np.testing.assert_allclose(out1["T_render"].numpy(), f @ out0["T_render"].numpy() @ f, atol=1e-5)
        np.testing.assert_allclose(out1["A_star"].numpy(), f @ out0["A_star"].numpy() @ f, atol=1e-4)
        np.testing.assert_array_equal(out1["flip_axes"].numpy(), flags)

    def test_images_flipped_with_random_state_and_tf_function(self):
        vol = self.rng.uniform(size=(2,) + P.LEVEL_SHAPE[4] + (1,)).astype(np.float32)
        t, rt = random_dali_batch(self.rng, "rigid", 4, 2)
        post = pp.LinearPostprocess("rigid", 4, flip_prob=0.5, seed=8)
        traces = []

        @tf.function
        def step(x, a, b):
            traces.append(1)
            return post(x, a, b)

        for _ in range(4):
            out = step(vol, t, rt)
            flags = out["flip_axes"].numpy()
            np.testing.assert_array_equal(out["images"].numpy(), tfm.flip_images(vol, flags).numpy())
            f = tfm.flip_matrix(flags).numpy()
            np.testing.assert_allclose(out["T_render"].numpy(), f @ pp.dali_to_mm(t, 4).numpy() @ f, atol=1e-5)
        self.assertEqual(len(traces), 1)
        # Same seed -> same flip / pair sequence.
        a = pp.LinearPostprocess("rigid", 4, seed=8)(vol, t, rt)
        b = pp.LinearPostprocess("rigid", 4, seed=8)(vol, t, rt)
        np.testing.assert_array_equal(a["flip_axes"].numpy(), b["flip_axes"].numpy())
        np.testing.assert_array_equal(a["pair_i"].numpy(), b["pair_i"].numpy())


class TestBrainS(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="m2_template_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_s_matrix_from_synthetic_mask(self):
        for level in (4, 2):
            shape = P.LEVEL_SHAPE[level]
            g = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
            c = [(n - 1) / 2.0 + 3 for n in shape]
            mask = (sum(((g[a] - c[a]) / (0.3 * shape[a])) ** 2 for a in range(3)) < 1).astype(np.uint8)
            np.save(os.path.join(self.tmp, "template_mask%s.npy" % P.level_suffix(level)), mask)
            s = pp.brain_S_matrix(level, self.tmp)
            np.testing.assert_allclose(s, tfm.brain_second_moment(mask, P.LEVEL_SPACING[level]))
            if level == 4:
                np.testing.assert_allclose(s, npr.second_moment_bruteforce(mask, 4.0), rtol=1e-10)
            self.assertAlmostEqual(s[3, 3], 1.0)
            self.assertTrue(np.all(np.linalg.eigvalsh(s) > 0))
        np.save(os.path.join(self.tmp, "template_mask_4.npy"), np.ones((10, 10, 10), np.uint8))
        with self.assertRaises(ValueError):
            pp.brain_S_matrix(4, self.tmp)


if __name__ == "__main__":
    unittest.main()
