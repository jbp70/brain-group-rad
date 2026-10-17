"""Tests for models.transforms against numpy / scipy references."""
import os
import unittest

import numpy as np

from tests import tf_setup  # noqa: F401  (log level, GPU memory growth)
from tests import np_reference as npr
import tensorflow as tf
from scipy.linalg import expm as sp_expm, logm as sp_logm

from models import transforms as tfm

REF_ROT = os.path.join(npr.REPO, "reference", "1_rigid_registration_src", "models",
                       "rotation_representations.py")


def random_rotation(rng, max_deg=45.0):
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    ang = np.deg2rad(rng.uniform(-max_deg, max_deg))
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return sp_expm(k * ang)


def fd_check(fn, x, rng, eps=1e-6):
    """Directional derivative: autodiff vs central difference (float64). Returns (ad, fd, rel)."""
    v = tf.constant(rng.normal(size=x.shape))
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = fn(x)
    g = tape.gradient(y, x)
    ad = float(tf.reduce_sum(g * v))
    fd = float((fn(x + eps * v) - fn(x - eps * v)) / (2 * eps))
    return ad, fd, abs(ad - fd) / max(abs(fd), 1e-12)


class TestRotation10D(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_identity_encoding(self):
        r = tfm.decode_10d(tfm.identity_rotation_10d()).numpy()
        np.testing.assert_allclose(r, np.eye(3), atol=1e-7)
        batched = tfm.decode_10d(np.tile(tfm.identity_rotation_10d(), (2, 3, 1)))
        self.assertEqual(batched.shape, (2, 3, 3, 3))
        np.testing.assert_allclose(batched.numpy(), np.broadcast_to(np.eye(3), (2, 3, 3, 3)), atol=1e-7)

    def test_random_params_are_rotations(self):
        v = tf.constant(self.rng.normal(size=(4, 50, 10)))
        r = tfm.decode_10d(v).numpy()
        rtr = np.einsum("...ji,...jk->...ik", r, r)
        np.testing.assert_allclose(rtr, np.broadcast_to(np.eye(3), rtr.shape), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-10)

    def test_matches_numpy_rederivation(self):
        v = self.rng.normal(size=(40, 10))
        r = tfm.decode_10d(tf.constant(v)).numpy()
        for vi, ri in zip(v, r):
            np.testing.assert_allclose(ri, npr.decode_10d_np(vi), atol=1e-9)

    def test_matches_reference_implementation(self):
        ref = npr.load_module("ref_rotation_representations", REF_ROT)
        if ref is None:
            self.skipTest("reference/ rotation_representations.py not present (local-only, gitignored)")
        v = self.rng.normal(size=(64, 10)).astype(np.float32)
        np.testing.assert_allclose(tfm.decode_10d(v).numpy(), ref.decode_10d(tf.constant(v)).numpy(),
                                   atol=1e-6)
        np.testing.assert_allclose(ref.identity_encoding("10d"), tfm.identity_rotation_10d())
        rots = np.stack([random_rotation(self.rng, 180.0) for _ in range(32)]).astype(np.float32)
        np.testing.assert_allclose(tfm.encode_10d(rots).numpy(), ref.encode_10d(tf.constant(rots)).numpy(),
                                   atol=1e-6)

    def test_encode_decode_round_trip(self):
        rots = np.stack([random_rotation(self.rng, 180.0) for _ in range(64)])
        back = tfm.decode_10d(tfm.encode_10d(tf.constant(rots))).numpy()
        np.testing.assert_allclose(back, rots, atol=1e-9)

    def test_gradients_finite(self):
        for v_np in [np.tile(tfm.identity_rotation_10d(), (3, 1)),
                     self.rng.normal(size=(16, 10)).astype(np.float32)]:
            v = tf.constant(v_np)
            w = tf.constant(self.rng.normal(size=(v_np.shape[0], 3, 3)), tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(v)
                y = tf.reduce_sum(w * tfm.decode_10d(v))
            g = tape.gradient(y, v)
            self.assertIsNotNone(g)
            self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(g))))
            self.assertGreater(float(tf.reduce_max(tf.abs(g))), 0.0)

    def test_gradient_matches_finite_difference(self):
        v = tf.constant(self.rng.normal(size=(8, 10)))
        w = tf.constant(self.rng.normal(size=(8, 3, 3)))
        _, _, rel = fd_check(lambda x: tf.reduce_sum(w * tfm.decode_10d(x)), v, self.rng)
        self.assertLess(rel, 1e-6)


class TestBuilders(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)

    def test_identity_params(self):
        rid = tfm.identity_rigid_params()
        self.assertEqual(rid.shape, (13,))
        self.assertEqual(tfm.identity_scale_shear_params().shape, (6,))
        np.testing.assert_allclose(tfm.rigid_matrix(rid).numpy(), np.eye(4), atol=1e-7)
        aff = np.concatenate([rid, tfm.identity_scale_shear_params()])
        np.testing.assert_array_equal(tfm.affine_matrix(aff).numpy(), np.eye(4, dtype=np.float32))

    def test_rigid_layout(self):
        p = self.rng.normal(size=(5, 13))
        m = tfm.rigid_matrix(tf.constant(p)).numpy()
        np.testing.assert_allclose(m[:, :3, :3], tfm.decode_10d(p[:, :10]).numpy(), atol=1e-12)
        np.testing.assert_allclose(m[:, :3, 3], p[:, 10:], atol=1e-12)
        np.testing.assert_allclose(m[:, 3], np.broadcast_to([0, 0, 0, 1], (5, 4)))

    def test_affine_is_r_times_k(self):
        p = self.rng.normal(size=(2, 4, 19)) * 0.3
        m = tfm.affine_matrix(tf.constant(p)).numpy()
        rot = tfm.decode_10d(p[..., :10]).numpy()
        s, sh = p[..., 13:16], p[..., 16:19]
        k = np.zeros(p.shape[:-1] + (3, 3))
        k[..., 0, 0], k[..., 1, 1], k[..., 2, 2] = np.exp(s[..., 0]), np.exp(s[..., 1]), np.exp(s[..., 2])
        k[..., 0, 1], k[..., 0, 2], k[..., 1, 2] = sh[..., 0], sh[..., 1], sh[..., 2]
        np.testing.assert_allclose(m[..., :3, :3], rot @ k, atol=1e-12)
        np.testing.assert_allclose(m[..., :3, 3], p[..., 10:13], atol=1e-12)
        np.testing.assert_allclose(tfm.scale_shear_matrix(p[..., 13:]).numpy(), k, atol=1e-12)

    def test_qr_round_trip_recovers_scales(self):
        for _ in range(20):
            p = np.concatenate([self.rng.normal(size=10), self.rng.uniform(-20, 20, 3),
                                self.rng.uniform(-0.1, 0.1, 3), self.rng.uniform(-0.05, 0.05, 3)])
            m = tfm.affine_matrix(tf.constant(p)).numpy()
            q, r = np.linalg.qr(m[:3, :3])
            sign = np.sign(np.diag(r))
            q, r = q * sign, r * sign[:, None]
            np.testing.assert_allclose(np.diag(r), np.exp(p[13:16]), atol=1e-10)
            self.assertTrue(np.all(np.diag(r) > 0))
            np.testing.assert_allclose([r[0, 1], r[0, 2], r[1, 2]], p[16:19], atol=1e-10)
            np.testing.assert_allclose(q, tfm.decode_10d(p[:10]).numpy(), atol=1e-10)
            self.assertAlmostEqual(np.linalg.det(q), 1.0, places=10)


class TestMatrixHelpers(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.demo = npr.load_demo()

    def test_inverse_compose_layout(self):
        a = np.stack([self.demo.random_affine() for _ in range(6)]).reshape(2, 3, 4, 4)
        inv = tfm.affine_inverse(tf.constant(a)).numpy()
        np.testing.assert_allclose(inv, np.linalg.inv(a), atol=1e-12)
        inv34 = tfm.affine_inverse(tf.constant(a[..., :3, :])).numpy()
        self.assertEqual(inv34.shape, (2, 3, 3, 4))
        np.testing.assert_allclose(inv34, np.linalg.inv(a)[..., :3, :], atol=1e-12)
        r = np.stack([self.demo.random_rigid() for _ in range(4)])
        np.testing.assert_allclose(tfm.affine_inverse(r, rigid=True).numpy(), np.linalg.inv(r), atol=1e-12)
        np.testing.assert_allclose(tfm.compose(a, a[..., :3, :], inv).numpy(), a, atol=1e-10)
        np.testing.assert_allclose(tfm.to_homogeneous(a[..., :3, :]).numpy(), a)
        self.assertEqual(tfm.to_3x4(a).shape, (2, 3, 3, 4))
        np.testing.assert_allclose(tfm.translation_matrix([1.0, 2.0, 3.0]).numpy()[:3, 3], [1, 2, 3])

    def test_mm_voxel_conversion(self):
        shape, spacing = (40, 56, 48), 4.0
        m = self.demo.random_affine()
        mv = tfm.mm_to_voxel_matrix(tf.constant(m), shape, spacing).numpy()
        np.testing.assert_allclose(mv, npr.voxel_matrix(m, shape, spacing), atol=1e-10)
        np.testing.assert_allclose(tfm.voxel_to_mm_matrix(mv, shape, spacing).numpy(), m, atol=1e-10)
        # point check: voxel p -> mm -> M -> back to voxel equals M_vox p
        p = np.array([3.0, 50.0, 17.0, 1.0])
        x = np.r_[(p[:3] - (np.array(shape) - 1) / 2.0) * spacing, 1.0]
        y = m @ x
        np.testing.assert_allclose(mv @ p, np.r_[y[:3] / spacing + (np.array(shape) - 1) / 2.0, 1.0],
                                   atol=1e-10)
        # anisotropic spacing and 3x4 layout
        mv34 = tfm.mm_to_voxel_matrix(tf.constant(m[:3]), shape, (1.0, 2.0, 4.0)).numpy()
        np.testing.assert_allclose(mv34, npr.voxel_matrix(m, shape, (1.0, 2.0, 4.0))[:3], atol=1e-10)

    def test_coarse_levels_share_centers(self):
        # Block averaging by 2 or 4 keeps voxel centers on the same world points.
        for f in (2, 4):
            fine = np.array([160, 224, 192])
            coarse = fine // f
            pc = np.array([0.0, 7.0, 11.0])
            fine_idx = pc * f + (f - 1) / 2.0  # mean index of the fine block
            np.testing.assert_allclose((pc - (coarse - 1) / 2.0) * f, (fine_idx - (fine - 1) / 2.0) * 1.0)

    def test_logm_expm(self):
        a = np.stack([self.demo.random_affine() for _ in range(5)])
        np.testing.assert_allclose(tfm.logm_real(a).numpy(), [np.real(sp_logm(x)) for x in a], atol=1e-10)
        x = tfm.logm_real(a)
        np.testing.assert_allclose(tfm.expm(x).numpy(), a, atol=1e-10)
        w = tf.constant(self.rng.normal(size=(5, 4, 4)))
        _, _, rel = fd_check(lambda z: tf.reduce_sum(w * tfm.logm_real(z)), tf.constant(a), self.rng)
        self.assertLess(rel, 1e-6)
        # float16 input is promoted to float32
        self.assertEqual(tfm.logm_real(tf.cast(a, tf.float16)).dtype, tf.float32)


class TestBarycenter(unittest.TestCase):
    def setUp(self):
        self.demo = npr.load_demo()
        self.rng = np.random.default_rng(4)

    def _groups(self, kind, g=3, n=6):
        gen = self.demo.random_rigid if kind == "rigid" else self.demo.random_affine
        return np.stack([[gen() for _ in range(n)] for _ in range(g)])  # +-45 deg, +-30 mm

    def test_matches_demo_numpy(self):
        for kind in ("rigid", "affine"):
            groups = self._groups(kind)
            mu_ref = np.stack([self.demo.barycenter(list(g), iters=20) for g in groups])
            centered_ref = np.stack([np.linalg.inv(m)[None] @ g for m, g in zip(mu_ref, groups)])
            mu64 = tfm.barycenter(tf.constant(groups), iters=10).numpy()
            np.testing.assert_allclose(mu64, mu_ref, atol=1e-9, err_msg=kind)
            out64 = tfm.GroupCentering(iters=10, dtype="float64")(tf.constant(groups)).numpy()
            np.testing.assert_allclose(out64, centered_ref, atol=1e-9, err_msg=kind)
            out32 = tfm.GroupCentering()(tf.constant(groups, tf.float32))
            self.assertEqual(out32.dtype, tf.float32)
            err = np.abs(out32.numpy() - centered_ref)
            self.assertLess(err[..., :3, :3].max(), 1e-4, kind)   # linear part (unitless)
            self.assertLess(err[..., :3, 3].max(), 1e-3, kind)    # translations (mm, up to ~60)

    def test_centered_output_and_pairwise(self):
        for kind in ("rigid", "affine"):
            groups = self._groups(kind)
            out = tfm.GroupCentering()(tf.constant(groups, tf.float32)).numpy().astype(np.float64)
            for o in out:
                mu = self.demo.barycenter(list(o), iters=20)
                # 4 float32 iterations: center within ~1e-3 mm over a brain-sized box
                self.assertLess(self.demo.diff_mm(mu, np.eye(4)), 1e-3, kind)
            for o, g in zip(out, groups):
                for i in range(6):
                    for j in range(6):
                        p_out = np.linalg.inv(o[j]) @ o[i]
                        p_in = np.linalg.inv(g[j]) @ g[i]
                        self.assertLess(np.abs(p_out[:3, :3] - p_in[:3, :3]).max(), 5e-6, kind)
                        self.assertLess(np.abs(p_out[:3, 3] - p_in[:3, 3]).max(), 1e-4, kind)

    def test_invariance_to_common_left_multiplication(self):
        groups = self._groups("affine", g=2)
        gm = self.demo.random_rigid()
        layer = tfm.GroupCentering(iters=8, dtype="float64")
        a = layer(tf.constant(gm[None, None] @ groups)).numpy()
        b = layer(tf.constant(groups)).numpy()
        np.testing.assert_allclose(a, b, atol=1e-8)

    def test_batched_leading_axes(self):
        groups = self._groups("rigid", g=4).reshape(2, 2, 6, 4, 4)
        mu = tfm.barycenter(tf.constant(groups), iters=6).numpy()
        self.assertEqual(mu.shape, (2, 2, 4, 4))
        single = tfm.barycenter(tf.constant(groups[1, 0]), iters=6).numpy()
        np.testing.assert_allclose(mu[1, 0], single, atol=1e-12)

    def test_gradient_centering_plus_point_loss(self):
        pts = np.c_[self.rng.uniform([-70, -85, -60], [70, 85, 60], (5000, 3)), np.ones(5000)]
        s = tf.constant(pts.T @ pts / len(pts))
        a_raw = tf.constant(self._groups("affine", g=2))
        a_star = tf.constant(self._groups("rigid", g=2))
        layer = tfm.GroupCentering(iters=4, dtype="float64")

        def loss(x):
            return tf.reduce_mean(tfm.point_loss(layer(x) - a_star, s))

        ad, fd, rel = fd_check(loss, a_raw, self.rng)
        self.assertLess(rel, 1e-5, "autodiff {} vs FD {}".format(ad, fd))

    def test_barycentric_targets(self):
        t = tf.constant(self._groups("affine", g=2))
        with tf.GradientTape() as tape:
            tape.watch(t)
            targets = tfm.barycentric_targets(t)
            y = tf.reduce_sum(targets)
        self.assertIsNone(tape.gradient(y, t))
        targets = targets.numpy()
        t_np = t.numpy()
        for a_g, t_g in zip(targets, t_np):
            mu = self.demo.barycenter(list(a_g), iters=20)
            self.assertLess(self.demo.diff_mm(mu, np.eye(4)), 1e-8)
            for i in range(6):
                for j in range(6):
                    np.testing.assert_allclose(np.linalg.inv(a_g[j]) @ a_g[i],
                                               t_g[j] @ np.linalg.inv(t_g[i]), atol=1e-9)
        # pair helpers agree with each other on the targets
        i_idx, j_idx = [0, 1, 5, 3], [2, 4, 0, 3]
        pt = tfm.pair_targets(t, i_idx, j_idx).numpy()
        pp = tfm.pair_predictions(tf.constant(targets), i_idx, j_idx).numpy()
        self.assertEqual(pt.shape, (2, 4, 4, 4))
        np.testing.assert_allclose(pp, pt, atol=1e-9)
        np.testing.assert_allclose(pt[1, 2], t_np[1, 0] @ np.linalg.inv(t_np[1, 5]), atol=1e-12)


class TestFlipAlgebra(unittest.TestCase):
    def test_conjugated_rigid_stays_rotation(self):
        demo = npr.load_demo()
        t = np.stack([demo.random_rigid() for _ in range(8)])
        for axes in [(True, False, False), (False, True, True), (True, True, True)]:
            f = tfm.flip_matrix(axes, tf.float64)
            out = tfm.flip_conjugate(tf.constant(t), f).numpy()
            r = out[:, :3, :3]
            np.testing.assert_allclose(r.transpose(0, 2, 1) @ r, np.broadcast_to(np.eye(3), r.shape), atol=1e-12)
            np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-12)
            fn = np.diag([-1.0 if a else 1.0 for a in axes] + [1.0])
            np.testing.assert_allclose(out, fn @ t @ fn, atol=1e-12)
            # pairwise maps transform the same way
            np.testing.assert_allclose(out[1] @ np.linalg.inv(out[0]),
                                       fn @ t[1] @ np.linalg.inv(t[0]) @ fn, atol=1e-12)


class TestPointLoss(unittest.TestCase):
    def test_second_moment_and_point_loss_bruteforce(self):
        rng = np.random.default_rng(5)
        mask = rng.uniform(size=(9, 11, 7)) > 0.4
        for spacing in (1.0, 2.0, (1.0, 2.0, 4.0)):
            s = tfm.brain_second_moment(mask, spacing)
            np.testing.assert_allclose(s, npr.second_moment_bruteforce(mask, spacing), atol=1e-10)
            np.testing.assert_allclose(tfm.brain_second_moment(mask[..., None], spacing), s)
            d_a = rng.normal(size=(3, 4, 4))
            d_a[..., 3, :] = 0.0
            loss = tfm.point_loss(tf.constant(d_a), s).numpy()
            self.assertEqual(loss.shape, (3,))
            for k in range(3):
                bf = npr.mean_sq_displacement_bruteforce(d_a[k], mask, spacing)
                self.assertAlmostEqual(loss[k] / bf, 1.0, places=10)
            # 3x4 layout gives the same value
            np.testing.assert_allclose(tfm.point_loss(tf.constant(d_a[:, :3]), s).numpy(), loss, rtol=1e-12)


class TestDaliConversion(unittest.TestCase):
    def setUp(self):
        self.demo = npr.load_demo()

    def test_involution_and_composition(self):
        m1, m2 = self.demo.random_affine(), self.demo.random_affine()
        conv = lambda m: tfm.dali_to_tf_matrix(tf.constant(m)).numpy()  # noqa: E731
        np.testing.assert_allclose(conv(conv(m1)), m1, atol=1e-12)
        np.testing.assert_allclose(conv(m1 @ m2), conv(m1) @ conv(m2), atol=1e-10)
        np.testing.assert_allclose(tfm.tf_to_dali_matrix(tf.constant(m1[:3])).numpy(), conv(m1)[:3])
        # acting on reversed-axis points: M_tf x = reverse(M_dali reverse(x))
        x = np.array([10.0, -20.0, 30.0, 1.0])
        rev = np.array([2, 1, 0, 3])
        np.testing.assert_allclose(conv(m1) @ x, (m1 @ x[rev])[rev], atol=1e-10)

    def test_dali_voxel_mm_round_trip_and_meaning(self):
        shape, spacing = (40, 56, 48), 4.0
        m = self.demo.random_affine()
        mm = tfm.dali_voxel_to_tf_mm(tf.constant(m), shape, spacing).numpy()
        np.testing.assert_allclose(tfm.tf_mm_to_dali_voxel(mm, shape, spacing).numpy(), m, atol=1e-9)
        # A DALI-voxel translation by d (DALI axis order) is a mm translation by reverse(d) * s.
        d = np.array([1.0, -2.0, 3.0])
        t_dali = np.eye(4)
        t_dali[:3, 3] = d
        t_mm = tfm.dali_voxel_to_tf_mm(tf.constant(t_dali), shape, spacing).numpy()
        np.testing.assert_allclose(t_mm[:3, 3], d[::-1] * spacing, atol=1e-12)
        np.testing.assert_allclose(t_mm[:3, :3], np.eye(3), atol=1e-12)
        # A rotation about the DALI image center (n / 2 in DALI coordinates, assumed) is a pure
        # rotation about the mm origin.
        r = random_rotation(np.random.default_rng(6))
        c_dali = np.array(shape[::-1], np.float64) / 2.0
        rot_dali = np.eye(4)
        rot_dali[:3, :3] = r
        rot_dali[:3, 3] = c_dali - r @ c_dali
        rot_mm = tfm.dali_voxel_to_tf_mm(tf.constant(rot_dali), shape, spacing).numpy()
        np.testing.assert_allclose(rot_mm[:3, 3], 0.0, atol=1e-10)
        rev3 = np.eye(3)[::-1]
        np.testing.assert_allclose(rot_mm[:3, :3], rev3 @ r @ rev3, atol=1e-12)
        self.assertEqual(tfm.dali_voxel_to_tf_mm(tf.constant(m[:3]), shape, spacing).shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
