"""Unit tests of inference/tta.py (TTA design 2-5, 7.2, 8 T1-T4, T8, T14).

The design checks .agents/tasks/tta/check_flip_signs.py (C1-C7) and check_gauge_free.py (E1-E5)
as unit tests:
* TestSigns (T1, C1-C3): Jay's literal 13-entry patterns == the derived x / y / z signs, combined
  flips == products, matrix(sigma_f * p) == F matrix(p) F (float64 / float32).
* TestCentering (T2, C4): center_group(F A F) == F center_group(A) F; unflip_matrices == F M F.
* TestFields (T3, C5, C6): SVF unflip involution, exp / warp / transform_image commute with flips,
  flip_np == tfm.flip_images.
* TestEncoding (T4, E1, E2): encode / decode round trip, sign covariance, raw K shears.
* TestAveraging (C7, T4): identical encodings, eigh read-out of +-theta, readout deviation,
  float32 field accumulation.
* TestGaugeFree (E3-E5): gauge-free residual averaging vs the v1 raw-averaging reference,
  pipeline equivariance with a non-equivariant group, 4-iteration centering, identity-only path,
  N = 2.
* TestDiagnostics (T8), TestTrainFlip (T14), TestConfig.
Measured maxima are printed to stderr.
"""
from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory limit)
import itertools
import os
import subprocess
import sys
import unittest

import numpy as np
import tensorflow as tf

from inference import tta as T
from models import cascade as CS
from models import diffeo
from models import layers as L
from models import transforms as tfm
from runtime import metrics as RM

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Jay's reference patterns (reference/1_rigid_registration_src/models/reg_net.py 168-180), literal.
JAY = {"x": np.array([1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1], np.float64),
       "y": np.array([1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1], np.float64),
       "z": np.array([1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1], np.float64)}
AXIS_FLIP = {"x": (True, False, False), "y": (False, True, False), "z": (False, False, True)}


def _say(msg):
    print("\n[tta] " + msg, file=sys.stderr)


def _maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def _rot(axis, deg):
    a = np.asarray(axis, np.float64)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def _rand_affine(rng, deg=30, mm=15, ls=0.08, sh=0.04):
    m = np.eye(4)
    k = np.eye(3)
    k[np.diag_indices(3)] = np.exp(rng.uniform(-ls, ls, 3))
    k[0, 1], k[0, 2], k[1, 2] = rng.uniform(-sh, sh, 3)
    m[:3, :3] = _rot(rng.normal(size=3), rng.uniform(-deg, deg)) @ k
    m[:3, 3] = rng.uniform(-mm, mm, 3)
    return m


def _rand_rigid(rng, deg=30, mm=15):
    m = np.eye(4)
    m[:3, :3] = _rot(rng.normal(size=3), rng.uniform(-deg, deg))
    m[:3, 3] = rng.uniform(-mm, mm, 3)
    return m


def _group(rng, n, kind="affine", **kw):
    f = _rand_affine if kind == "affine" else _rand_rigid
    return np.stack([f(rng, **kw) for _ in range(n)])


def _flip_field(v, fl):
    """v^F(p) = F v(F p), written independently of inference.tta (as in check_flip_signs.py)."""
    out = tfm.flip_images(v, fl)
    return out * tf.constant([-1.0 if b else 1.0 for b in fl], v.dtype)


def _points_moment(rng, n=500, r=60.0):
    h = np.c_[rng.uniform(-r, r, (n, 3)), np.ones(n)]
    return h.T @ h / n


def _pair_err(a, b, S):
    """max over ordered pairs i != j of the point RMS of A_j^-1 A_i - B_j^-1 B_i (mm)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    e = 0.0
    for i in range(len(a)):
        for j in range(len(a)):
            if i != j:
                d = np.linalg.inv(a[j]) @ a[i] - np.linalg.inv(b[j]) @ b[i]
                e = max(e, float(RM.point_rms(d, S)))
    return e


def _random_params(rng, b=64):
    p_rot = rng.normal(size=(b, 10)) + tfm.identity_rotation_10d()[None] * 2.0
    p_t = rng.normal(scale=20.0, size=(b, 3))
    p_ss = np.concatenate([rng.normal(scale=0.15, size=(b, 3)), rng.normal(scale=0.2, size=(b, 3))], 1)
    return np.concatenate([p_rot, p_t, p_ss], 1)


# ============================================================================ T1 / C1-C3
class TestSigns(unittest.TestCase):
    def test_axes_equal_jay(self):
        for ax, fl in AXIS_FLIP.items():
            for task in ("rigid", "affine"):
                s = T.linear_flip_signs(fl, task)
                self.assertEqual(s.dtype, np.float64)
                self.assertEqual(s.shape, (13 if task == "rigid" else 19,))
                self.assertTrue(np.array_equal(s[:13], JAY[ax]), (ax, task, s[:13]))

    def test_combined_are_products(self):
        for fl in T.FLIPS_8:
            prod13, prod19 = np.ones(13), np.ones(19)
            for ax, b in zip("xyz", fl):
                if b:
                    prod13 = prod13 * JAY[ax]
                    prod19 = prod19 * T.linear_flip_signs(AXIS_FLIP[ax], "affine")
            self.assertTrue(np.array_equal(T.linear_flip_signs(fl, "rigid"), prod13), fl)
            self.assertTrue(np.array_equal(T.linear_flip_signs(fl, "affine"), prod19), fl)
            s = T.linear_flip_signs(fl, "affine")
            self.assertTrue(np.array_equal(s * s, np.ones(19)))      # its own inverse
        self.assertTrue(np.array_equal(T.linear_flip_signs((False,) * 3, "affine"), np.ones(19)))
        # flipping all three axes keeps the rotation and negates t
        s = T.linear_flip_signs((True, True, True), "affine")
        self.assertTrue(np.array_equal(s[:10], np.ones(10)))
        self.assertTrue(np.array_equal(s[10:13], -np.ones(3)))

    def _check_matrices(self, dtype, tol):
        rng = np.random.default_rng(0)
        p19 = _random_params(rng).astype(dtype.as_numpy_dtype)
        worst_r = worst_a = 0.0
        for fl in T.FLIPS_8:
            F = tfm.flip_matrix(fl, dtype)
            s = T.linear_flip_signs(fl, "affine").astype(dtype.as_numpy_dtype)
            m_r = tfm.rigid_matrix(tf.constant(p19[:, :13]))
            m_rf = tfm.rigid_matrix(tf.constant(p19[:, :13] * s[:13]))
            worst_r = max(worst_r, _maxdiff(m_rf, tfm.flip_conjugate(m_r, F)))
            m_a = tfm.affine_matrix(tf.constant(p19))
            m_af = tfm.affine_matrix(tf.constant(p19 * s))
            worst_a = max(worst_a, _maxdiff(m_af, tfm.flip_conjugate(m_a, F)))
        _say("C1/C2 %s max |M(p s_f) - F M(p) F| rigid %.2e, affine %.2e (8 flips x 64)"
             % (dtype.name, worst_r, worst_a))
        self.assertLess(worst_r, tol)
        self.assertLess(worst_a, tol)

    def test_matrix_covariance_float64(self):
        self._check_matrices(tf.float64, 1e-12)

    def test_matrix_covariance_float32(self):
        self._check_matrices(tf.float32, 1e-5)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            T.linear_flip_signs((True, False), "rigid")
        with self.assertRaises(ValueError):
            T.linear_flip_signs((True, False, False), "deform")


# ============================================================================ T2 / C4
class TestCentering(unittest.TestCase):
    def test_center_commutes_with_conjugation(self):
        rng = np.random.default_rng(1)
        A = tf.constant(_group(rng, 12))
        worst = 0.0
        for fl in T.FLIPS_8:
            F = tfm.flip_matrix(fl, tf.float64)
            lhs = tfm.center_group(tfm.flip_conjugate(A, F), iters=4)
            rhs = tfm.flip_conjugate(tfm.center_group(A, iters=4), F)
            worst = max(worst, _maxdiff(lhs, rhs))
        _say("C4 max |center(F A F) - F center(A) F| (N 12, float64): %.2e" % worst)
        self.assertLess(worst, 1e-10)

    def test_unflip_matrices(self):
        rng = np.random.default_rng(2)
        A = _group(rng, 5)
        for fl in T.FLIPS_8:
            F = T.flip_matrix_np(fl)
            self.assertTrue(np.array_equal(F, np.asarray(tfm.flip_matrix(fl, tf.float64))))
            want = F @ A @ F
            got = T.unflip_matrices(A, fl)
            self.assertEqual(got.dtype, np.float64)
            self.assertEqual(_maxdiff(got, want), 0.0)
            a32 = A.astype(np.float32)
            self.assertEqual(T.unflip_matrices(a32, fl).dtype, np.float32)
            got_tf = T.unflip_matrices(tf.constant(a32), fl)
            self.assertIsInstance(got_tf, tf.Tensor)
            self.assertEqual(got_tf.dtype, tf.float32)
            self.assertTrue(np.array_equal(got_tf.numpy(), tfm.flip_conjugate(a32, tfm.flip_matrix(fl)).numpy()))
            self.assertTrue(np.array_equal(T.unflip_matrices(T.unflip_matrices(a32, fl), fl), a32))
        self.assertIs(T.unflip_matrices(A, (False, False, False)), A)


# ============================================================================ T3 / C5, C6
class TestFields(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        v = tf.constant(rng.normal(scale=1.5, size=(2, 5, 7, 6, 3)).astype(np.float32))
        for _ in range(3):
            v = L.upsample_trilinear2x(v)                  # smooth [2, 40, 56, 48, 3] (level 4)
        cls.v = v
        cls.img = rng.uniform(size=(2, 40, 56, 48, 1)).astype(np.float32)

    def test_unflip_involution_and_convention(self):
        v = self.v
        self.assertIs(T.unflip_field(v, (False, False, False)), v)
        for fl in T.FLIPS_8:
            vf = T.unflip_field(v, fl)
            self.assertTrue(np.array_equal(vf.numpy(), _flip_field(v, fl).numpy()), fl)
            self.assertTrue(np.array_equal(T.unflip_field(vf, fl).numpy(), v.numpy()), fl)

    def test_exp_and_warp_commute(self):
        v, img = self.v, self.img
        u = diffeo.integrate_svf(v, 7)
        worst_u = worst_w = 0.0
        for fl in T.FLIPS_8:
            uf = diffeo.integrate_svf(T.unflip_field(v, fl), 7)
            worst_u = max(worst_u, _maxdiff(uf, T.unflip_field(u, fl)))
            lhs = diffeo.warp(tfm.flip_images(img, fl), uf)
            rhs = tfm.flip_images(diffeo.warp(img, u), fl)
            worst_w = max(worst_w, _maxdiff(lhs, rhs))
        _say("C6 max |v| %.2f vox; max |exp(v^F) - (exp v)^F| %.2e vox; max |warp(flip I, u^F) - "
             "flip warp(I, u)| %.2e" % (float(tf.reduce_max(tf.abs(v))), worst_u, worst_w))
        self.assertLess(worst_u, 1e-5)
        self.assertLess(worst_w, 1e-4)

    def test_transform_image_commutes(self):
        rng = np.random.default_rng(4)
        img = self.img[:1]
        M = tf.constant(_rand_affine(rng)[None].astype(np.float32))
        worst = 0.0
        for fl in T.FLIPS_8:
            F = tfm.flip_matrix(fl, tf.float32)
            lhs = L.transform_image(tfm.flip_images(img, fl), tfm.flip_conjugate(M, F), 4.0)
            rhs = tfm.flip_images(L.transform_image(img, M, 4.0), fl)
            worst = max(worst, _maxdiff(lhs, rhs))
        _say("C5 max |T(flip I, F M F) - flip T(I, M)| (level 4, float32): %.2e" % worst)
        self.assertLess(worst, 1e-4)

    def test_flip_np(self):
        x = np.asarray(self.v.numpy(), np.float64)          # float64 input -> float32 output
        for fl in T.FLIPS_8:
            got = T.flip_np(x, fl)
            self.assertEqual(got.dtype, np.float32)
            self.assertTrue(got.flags["C_CONTIGUOUS"])
            want = tfm.flip_images(tf.constant(x.astype(np.float32)), fl).numpy()
            self.assertTrue(np.array_equal(got, want), fl)


# ============================================================================ T4 / E1, E2
class TestEncoding(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(5)
        for task, kind in (("rigid", "rigid"), ("affine", "affine")):
            M = _group(rng, 50, kind)
            p = T.encode_linear(M, task)
            self.assertEqual(p.shape, (50, 13 if task == "rigid" else 19))
            back = T.decode_linear(p, task, np.float64)
            err = _maxdiff(back, M)
            _say("E1 %s encode/decode round trip max |diff|: %.2e" % (task, err))
            self.assertLess(err, 1e-12)
            self.assertEqual(T.decode_linear(p, task).dtype, np.float32)

    def test_sign_covariance(self):
        rng = np.random.default_rng(6)
        worst = 0.0
        for task, kind in (("rigid", "rigid"), ("affine", "affine")):
            M = _group(rng, 20, kind)
            p = T.encode_linear(M, task)
            for fl in T.FLIPS_8:
                F = T.flip_matrix_np(fl)
                pf = T.encode_linear(F @ M @ F, task)
                worst = max(worst, _maxdiff(pf, T.linear_flip_signs(fl, task)[None] * p))
        _say("E2 max |encode(F M F) - sigma_f * encode(M)| over 8 flips x 20: %.2e" % worst)
        self.assertLess(worst, 1e-12)

    def test_raw_k_entries(self):
        K = np.array([[1.3, 0.07, -0.05], [0.0, 0.8, 0.03], [0.0, 0.0, 1.15]])
        M = np.eye(4)
        M[:3, :3] = _rot((0.3, -1.0, 0.4), 23.0) @ K
        M[:3, 3] = (4.0, -2.0, 7.0)
        p = T.encode_linear(M[None], "affine")[0]
        self.assertLess(_maxdiff(p[16:19], [K[0, 1], K[0, 2], K[1, 2]]), 1e-12)
        self.assertLess(_maxdiff(p[13:16], np.log(np.diag(K))), 1e-12)
        _, _, ratios = RM.qr_factors(M[:3, :3])
        self.assertGreater(_maxdiff(p[16:19], ratios), 1e-3)      # not the qr_factors ratios
        r = T.encode_linear(M[None], "rigid")
        self.assertEqual(r.shape, (1, 13))
        self.assertTrue(np.array_equal(r[0], p[:13]))

    def test_invalid_rows_nan(self):
        rng = np.random.default_rng(7)
        M = _group(rng, 4)
        M[1, 0, 0] = np.nan
        M[2, :3, :3] = np.diag([-1.0, 1.0, 1.0])            # det <= 0
        M[3, :3, :3] = 0.0                                  # singular
        p = T.encode_linear(M, "affine")
        self.assertTrue(np.all(np.isfinite(p[0])))
        for i in (1, 2, 3):
            self.assertTrue(np.all(np.isnan(p[i])), i)


# ============================================================================ C7 / T4 averaging
class TestAveraging(unittest.TestCase):
    def test_identical_encodings(self):
        rng = np.random.default_rng(8)
        p = T.encode_linear(_group(rng, 10), "affine")
        p_bar, diag = T.mean_linear([p] * 8)
        rel = float(np.max(np.abs(p_bar - p) / np.maximum(1.0, np.abs(p))))
        _say("C7 mean_linear of 8 identical encodings: max rel diff %.2e, rot_gap min %.15f, "
             "readout_dev max %.2e deg" % (rel, float(diag["rot_gap"].min()),
                                           float(diag["readout_dev_deg"].max())))
        self.assertLess(rel, 1e-15)
        np.testing.assert_allclose(diag["rot_gap"], 1.0, rtol=0, atol=1e-12)
        self.assertLess(float(diag["readout_dev_deg"].max()), 1e-6)

    def test_plus_minus_theta(self):
        base = _rot((0.2, 0.9, -0.4), 37.0)
        worst = 0.0
        devs = []
        for ax in ((0, 0, 1), (1, 0, 0), (0.3, -0.5, 0.8)):
            ms = []
            for sgn in (1.0, -1.0):
                m = np.eye(4)
                m[:3, :3] = base @ _rot(ax, sgn * 30.0)
                ms.append(m[None])
            p_bar, diag = T.mean_linear([T.encode_linear(m, "rigid") for m in ms])
            r_bar = T.decode_linear(p_bar, "rigid", np.float64)[0, :3, :3]
            worst = max(worst, float(RM.rotation_angle_deg(r_bar @ base.T)))
            devs.append(float(diag["readout_dev_deg"][0]))
            self.assertGreater(float(diag["rot_gap"][0]), 0.0)
        _say("eigh read-out of base * R(+-30 deg): angle to base max %.2e deg; readout_dev_deg %s"
             % (worst, ["%.3g" % d for d in devs]))
        self.assertLess(worst, 1e-9)
        self.assertGreater(min(devs), 1e-6)       # decode_10d of the mean != eigh mean

    def test_nonfinite_rows_stay_nan(self):
        rng = np.random.default_rng(9)
        p = T.encode_linear(_group(rng, 3), "affine")
        q = p.copy()
        q[1] = np.nan
        p_bar, diag = T.mean_linear([p, q])
        self.assertTrue(np.all(np.isfinite(p_bar[[0, 2]])))
        self.assertTrue(np.all(np.isnan(p_bar[1])))
        self.assertTrue(np.isnan(diag["rot_gap"][1]))

    def test_float32_field_accumulation(self):
        rng = np.random.default_rng(10)
        v = tf.constant(rng.normal(scale=2.0, size=(2, 8, 10, 6, 3)).astype(np.float32))
        acc = tf.zeros_like(v)
        for fl in T.FLIPS_8:
            acc = acc + T.unflip_field(T.unflip_field(v, fl), fl)   # equivariant prediction, unflipped
        avg = (acc / 8.0).numpy()
        rel = float(np.max(np.abs(avg - v.numpy()) / np.maximum(np.abs(v.numpy()), 1e-30)))
        _say("C7 float32 running sum of 8 identical fields / 8: max rel diff %.2e" % rel)
        self.assertLess(rel, 2e-7)


# ============================================================================ E3-E5
class TestGaugeFree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cen30 = tfm.GroupCentering(iters=30, dtype="float64")
        cls.cen4 = tfm.GroupCentering(iters=4, dtype="float64")

    def _setup(self, n, seed, centering):
        rng = np.random.default_rng(seed)
        truth = _group(rng, n)
        a_prev = centering(tf.constant(_group(rng, n, deg=10, mm=5))).numpy()
        res_true = truth @ np.linalg.inv(a_prev)
        gauges = [_rand_affine(rng, 15, 6) for _ in T.FLIPS_8]
        raws = [np.einsum("ij,njk->nik", g, res_true) for g in gauges]     # G_f R_i (unflipped)
        S = _points_moment(rng)
        return a_prev, res_true, raws, S

    def _v1(self, raws, a_prev, centering):
        """Version-1 reference: average the raw (unflipped) encodings, decode, compose, center."""
        p_bar, _ = T.mean_linear([T.encode_linear(r, "affine") for r in raws])
        r_bar = tf.constant(T.decode_linear(p_bar, "affine", np.float64))
        return CS.compose_linear(r_bar, tf.constant(a_prev), centering).numpy()

    def _check_gauge(self, n, seed):
        a_prev, res_true, raws, S = self._setup(n, seed, self.cen30)
        ref = CS.compose_linear(tf.constant(res_true), tf.constant(a_prev), self.cen30).numpy()
        out = T.linear_tta_level([tf.constant(r) for r in raws], tf.constant(a_prev), self.cen30,
                                 "affine", T.FLIPS_8)
        A = out["A"].numpy()
        self.assertEqual(out["A"].dtype, tf.float64)
        self.assertEqual(out["nonfinite"], [])
        self.assertEqual(len(out["A_flips"]), 8)
        self.assertEqual(len(out["params"]), 8)
        e_v2 = _pair_err(A, ref, S)
        e_v1 = _pair_err(self._v1(raws, a_prev, self.cen30), ref, S)
        cen = _maxdiff(A, ref)
        return e_v1, e_v2, cen

    def test_e3_gauge_free(self):
        e_v1, e_v2, cen = self._check_gauge(6, 11)
        _say("E3 flip-dependent raw gauge, zero model error (N 6): v1 raw averaging pair err %.3f mm; "
             "v2 pair err %.2e mm; v2 centered max |diff| %.2e" % (e_v1, e_v2, cen))
        self.assertLess(e_v2, 1e-8)
        self.assertLess(cen, 1e-8)
        self.assertGreater(e_v1, 0.05)

    def test_n2(self):
        e_v1, e_v2, cen = self._check_gauge(2, 12)
        _say("E3 at N 2: v1 %.3f mm, v2 %.2e mm, centered %.2e" % (e_v1, e_v2, cen))
        self.assertLess(e_v2, 1e-8)
        self.assertLess(cen, 1e-8)

    def test_e4_equivariance(self):
        rng = np.random.default_rng(13)
        n = 5
        worst = 0.0
        for task, kind in (("affine", "affine"), ("rigid", "rigid")):
            a_prev = self.cen30(tf.constant(_group(rng, n, kind, deg=10, mm=5))).numpy()
            outs = {fl: _group(rng, n, kind, deg=20, mm=8) for fl in T.FLIPS_8}   # flipped-frame raws
            for prev in (a_prev, None):
                kw = {} if prev is not None else {"dtype": tf.float64}
                prev_t = None if prev is None else tf.constant(prev)
                tta_x = T.linear_tta_level([T.unflip_matrices(outs[fl], fl) for fl in T.FLIPS_8],
                                           prev_t, self.cen30, task, T.FLIPS_8, **kw)["A"].numpy()
                for g in ((True, False, False), (False, True, False), (False, False, True), (True, True, True)):
                    G = T.flip_matrix_np(g)
                    prev_g = None if prev is None else tf.constant(G @ prev @ G)
                    raws_g = [T.unflip_matrices(outs[tuple(a != b for a, b in zip(fl, g))], fl)
                              for fl in T.FLIPS_8]
                    tta_g = T.linear_tta_level(raws_g, prev_g, self.cen30, task, T.FLIPS_8, **kw)["A"].numpy()
                    worst = max(worst, _maxdiff(tta_g, G @ tta_x @ G))
        _say("E4 TTA(flip_g x) vs F_g TTA(x) F_g, non-equivariant group, rigid + affine, first and "
             "later level: max |diff| %.2e" % worst)
        self.assertLess(worst, 1e-8)

    def test_e5_four_iterations(self):
        a_prev, res_true, raws, S = self._setup(6, 11, self.cen30)
        ref4 = CS.compose_linear(tf.constant(res_true), tf.constant(a_prev), self.cen4).numpy()
        A = T.linear_tta_level([tf.constant(r) for r in raws], tf.constant(a_prev), self.cen4,
                               "affine", T.FLIPS_8)["A"].numpy()
        e5 = _pair_err(A, ref4, S)
        cen5 = float(np.max(np.abs(A - ref4)[:, :3, 3]))
        _say("E5 4-iteration centering: v2 pair err %.2e mm, centered translation max |diff| %.2e mm"
             % (e5, cen5))
        self.assertLess(e5, 1e-8)
        self.assertLess(cen5, 1e-5)

    def test_identity_only_exact(self):
        rng = np.random.default_rng(14)
        cen = tfm.GroupCentering(iters=4)                   # the model's float32 centering
        M = tf.constant(_group(rng, 4).astype(np.float32))
        a_prev = tf.cast(cen(tf.constant(_group(rng, 4, deg=10, mm=5).astype(np.float32))), tf.float32)
        for task in ("rigid", "affine"):
            out = T.linear_tta_level([M], a_prev, cen, task, T.IDENTITY_ONLY)
            want = CS.compose_linear(M, a_prev, cen)
            self.assertEqual(out["A"].dtype, want.dtype)
            self.assertEqual(_maxdiff(out["A"], want), 0.0)
            self.assertIsNone(out["A_flips"])
            self.assertIsNone(out["params"])
            out0 = T.linear_tta_level([M], None, cen, task, T.IDENTITY_ONLY)
            want0 = tf.cast(cen(M), tf.float32)
            self.assertEqual(out0["A"].dtype, tf.float32)
            self.assertEqual(_maxdiff(out0["A"], want0), 0.0)
        with self.assertRaises(ValueError):
            T.linear_tta_level([M], None, cen, "rigid", T.FLIPS_8)

    def test_float32_path_and_nonfinite_flip(self):
        """Model dtype path (float32 centering) is finite; a non-finite flip is named and makes the
        affected image's result non-finite (identity 'centering' stub: no logm of NaN)."""
        rng = np.random.default_rng(15)
        cen = tfm.GroupCentering(iters=4)
        n = 3
        raws = [tf.constant(_group(rng, n).astype(np.float32)) for _ in T.FLIPS_8]
        out = T.linear_tta_level(raws, None, cen, "affine", T.FLIPS_8)
        self.assertEqual(out["A"].dtype, tf.float32)
        self.assertTrue(np.all(np.isfinite(out["A"].numpy())))
        self.assertEqual(out["nonfinite"], [])
        bad = list(raws)
        b = bad[3].numpy()
        b[1, 0, 3] = np.nan
        bad[3] = tf.constant(b)
        out = T.linear_tta_level(bad, None, lambda m: m, "affine", T.FLIPS_8)
        self.assertEqual(out["nonfinite"], [3])
        A = out["A"].numpy()
        self.assertTrue(np.all(np.isnan(A[1])))
        self.assertTrue(np.all(np.isfinite(A[[0, 2]])))


# ============================================================================ T8 diagnostics
class TestDiagnostics(unittest.TestCase):
    def test_translation_spread(self):
        rng = np.random.default_rng(16)
        n, d = 4, 1.7
        A_bar = _group(rng, n)
        signs = [1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0]
        A_flips = []
        for s in signs:
            a = A_bar.copy()
            a[:, 0, 3] += s * d
            A_flips.append(a)
        params = [T.encode_linear(a, "affine") for a in A_flips]
        p_bar = T.encode_linear(A_bar, "affine")
        S = _points_moment(rng)
        for S_arg in (S, np.repeat(S[None], n, 0)):
            out = T.linear_spread(A_flips, A_bar, S_arg, params, p_bar, "affine")
            np.testing.assert_allclose(out["spread_rms_mm"], d, rtol=1e-12)
            np.testing.assert_allclose(out["spread_max_mm"], d, rtol=1e-12)
            np.testing.assert_allclose(out["shift_mm"], d, rtol=1e-12)
            np.testing.assert_allclose(out["trans_mm_rms"], d, rtol=1e-9)
            np.testing.assert_allclose(out["rot_deg_rms"], 0.0, atol=1e-6)
            np.testing.assert_allclose(out["scale_pct_rms"], 0.0, atol=1e-9)
            np.testing.assert_allclose(out["shear_pct_rms"], 0.0, atol=1e-9)
        self.assertNotIn("spread_p99_mm", out)              # deform only (review v2 NIT 3)

    def test_rotation_scale_shear_spread(self):
        theta, ds, dk = 3.0, 0.02, 0.01
        A_flips, signs = [], [1.0, -1.0] * 4
        for s in signs:
            K = np.eye(3)
            K[0, 0] = np.exp(s * ds)
            K[0, 1] = s * dk
            m = np.eye(4)
            m[:3, :3] = _rot((0, 0, 1), s * theta) @ K
            A_flips.append(m[None])
        params = [T.encode_linear(a, "affine") for a in A_flips]
        p_bar, diag = T.mean_linear(params)
        A_bar = T.decode_linear(p_bar, "affine", np.float64)
        np.testing.assert_allclose(A_bar[0], np.eye(4), atol=1e-12)
        out = T.linear_spread(A_flips, A_bar, np.eye(4), params, p_bar, "affine", mean_diag=diag)
        _say("T8 rot_deg_rms %.6f (want %.1f), scale_pct_rms %.6f (want %.1f), shear_pct_rms %.6f "
             "(want %.1f), rot_gap %.4f, readout_dev %.2e" % (
                 out["rot_deg_rms"][0], theta, out["scale_pct_rms"][0], 100 * ds,
                 out["shear_pct_rms"][0], 100 * dk, out["rot_gap"][0], out["readout_dev_deg"][0]))
        self.assertAlmostEqual(float(out["rot_deg_rms"][0]), theta, places=9)
        self.assertAlmostEqual(float(out["scale_pct_rms"][0]), 100 * ds, places=9)
        self.assertAlmostEqual(float(out["shear_pct_rms"][0]), 100 * dk, places=9)
        self.assertIn("rot_gap", out)
        self.assertIn("readout_dev_deg", out)

    def test_gauge_size(self):
        angles, shifts = [0.0, 2.0, 5.0, 1.0, 7.5, 3.0, 0.5, 4.0], [0, 1, 2, 3, 4, 2.5, 6.0, 1]
        mus = []
        for a, t in zip(angles, shifts):
            m = np.eye(4)
            m[:3, :3] = _rot((0.2, 1.0, 0.1), a)
            m[:3, 3] = (t, 0.0, 0.0)
            mus.append(m)
        deg, mm = T.gauge_size(mus)
        self.assertAlmostEqual(deg, 7.5, places=9)
        # inv(mu_0) = I, so the translations are the shifts
        self.assertAlmostEqual(mm, 6.0, places=9)

    def test_gauge_barycenters(self):
        """mu_f of G_f A equals G_f mu(A) (converged), so gauge_size returns G_f G_0^-1's size."""
        rng = np.random.default_rng(17)
        A = tfm.center_group(tf.constant(_group(rng, 5)), 30).numpy()
        Gs = []
        for a in (0.0, 4.0, -9.0, 2.0, 6.0, -1.0, 3.0, 8.5):
            g = np.eye(4)
            g[:3, :3] = _rot((1.0, -0.2, 0.3), a)
            g[:3, 3] = (0.1 * a, -0.2 * a, 0.0)
            Gs.append(g)
        M = [np.einsum("ij,njk->nik", g, A) for g in Gs]
        mus = T.gauge_barycenters(M, None, 30)
        deg, mm = T.gauge_size(mus)
        self.assertAlmostEqual(deg, 9.0, places=6)
        self.assertAlmostEqual(mm, float(np.linalg.norm([0.9, -1.8, 0.0])), places=6)
        mus_prev = T.gauge_barycenters([np.eye(4)[None].repeat(5, 0)], tf.constant(A), 30)
        self.assertLess(_maxdiff(mus_prev[0], np.eye(4)), 1e-8)

    def test_summarize_spread(self):
        rows = [{"level": 4, "task": "rigid", "spread_rms_mm": v, "spread_p99_mm": ""} for v in (1.0, 2.0, 3.0)]
        rows += [{"level": 2, "task": "deform", "spread_rms_mm": 0.5, "spread_p99_mm": 0.9,
                  "rot_gap": None, "shift_mm": float("nan")}]
        s = T.summarize_spread(rows)
        self.assertEqual(sorted(s), ["l2/deform", "l4/rigid"])
        self.assertEqual(s["l4/rigid"]["spread_rms_mm"], {"mean": 2.0, "median": 2.0, "p95": 2.9, "max": 3.0})
        self.assertNotIn("spread_p99_mm", s["l4/rigid"])
        self.assertEqual(s["l2/deform"]["spread_p99_mm"]["max"], 0.9)
        self.assertNotIn("shift_mm", s["l2/deform"])
        self.assertNotIn("rot_gap", s["l2/deform"])


# ============================================================================ T14
class TestTrainFlip(unittest.TestCase):
    def test_probs_and_warnings(self):
        chain = {4: ("/a/config.json", {"args": {"flip_prob": 0.5}}),
                 2: ("/b/config.json", {"args": {"flip_prob": 0.2}}),
                 1: ("/c/config.json", {"args": {"flip_prob": 0.0}})}
        self.assertEqual(T.train_flip_probs(chain), {4: 0.5, 2: 0.2, 1: 0.0})
        other = {4: (None, {"args": {"flip_prob": 0.5, "overfit_one_group": True}}),
                 2: (None, None), 1: ("/x", {"args": {}})}
        probs = T.train_flip_probs(other)
        self.assertEqual(probs, {4: 0.0, 2: None, 1: None})
        w = T.flip_prob_warnings(T.train_flip_probs(chain), (4, 2, 1))
        self.assertEqual(w, ["TTA level 2: training flip probability 0.2 < 0.5: flips partly out of "
                             "training distribution",
                             "TTA level 1: training flip probability 0: flips never seen in training; "
                             "TTA averages out-of-distribution predictions"])
        w2 = T.flip_prob_warnings(probs, (4, 2))
        self.assertEqual(w2, ["TTA level 4: training flip probability 0: flips never seen in training; "
                              "TTA averages out-of-distribution predictions",
                              "TTA level 2: training flip probability unknown"])
        self.assertEqual(T.flip_prob_warnings({4: 0.5, 2: 0.5}, (4, 2)), [])
        self.assertEqual(T.flip_prob_warnings({4: 0.2}, ()), [])


class TestConfig(unittest.TestCase):
    def test_flips_and_config(self):
        self.assertEqual(len(T.FLIPS_8), 8)
        self.assertEqual(len(set(T.FLIPS_8)), 8)
        self.assertEqual(T.FLIPS_8[0], (False, False, False))
        self.assertEqual(T.FLIPS_8, tuple(itertools.product((False, True), repeat=3)))
        self.assertEqual(T.IDENTITY_ONLY, ((False, False, False),))
        c = T.TTAConfig("flip8", (4, 2, 1))
        self.assertEqual(c.to_dict(), {"mode": "flip8", "levels": [4, 2, 1],
                                       "flips": [list(f) for f in T.FLIPS_8]})
        self.assertEqual(T.TTAConfig("flip8", [1], T.IDENTITY_ONLY).to_dict()["flips"], [[False, False, False]])
        with self.assertRaises(ValueError):
            T.TTAConfig("flip", (4,))
        self.assertEqual(T.SPREAD_COLUMNS[:3], ("spread_rms_mm", "spread_max_mm", "spread_p99_mm"))

    def test_import_is_tf_free(self):
        code = "import inference.tta, sys; assert 'tensorflow' not in sys.modules"
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
