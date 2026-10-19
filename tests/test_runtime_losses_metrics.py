"""Linear losses (runtime.losses) and metrics (runtime.metrics) against numpy brute force."""
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests import np_reference as npr
import numpy as np
import tensorflow as tf
from scipy.linalg import expm as sp_expm

from data_loading import params as P
from models import transforms as tfm
from runtime import losses
from runtime import metrics as M

SPACING = P.LEVEL_SPACING[4]


def rot(axis, deg):
    ax = np.asarray(axis, np.float64)
    ax = ax / np.linalg.norm(ax)
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return sp_expm(k * np.deg2rad(deg))


def random_affine(rng, deg=20.0, mm=10.0, lin=0.05):
    m = np.eye(4)
    m[:3, :3] = rot(rng.normal(size=3), rng.uniform(-deg, deg)) @ (np.eye(3) + lin * rng.normal(size=(3, 3)))
    m[:3, 3] = rng.uniform(-mm, mm, size=3)
    return m


def synthetic_mask(rng):
    shape = P.LEVEL_SHAPE[4]
    g = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    r2 = sum(((gi - (n - 1) / 2.0) / (0.35 * n)) ** 2 for gi, n in zip(g, shape))
    return np.logical_and(r2 <= 1.0, rng.uniform(size=shape) > 0.3)


class TestLosses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(0)
        cls.mask = synthetic_mask(cls.rng)
        cls.S = tfm.brain_second_moment(cls.mask, SPACING)
        cls.A = np.stack([random_affine(cls.rng) for _ in range(5)])
        cls.A_star = np.stack([random_affine(cls.rng) for _ in range(5)])
        cls.T = np.stack([random_affine(cls.rng) for _ in range(5)])

    def bf_center(self, A, A_star):
        return np.mean([npr.mean_sq_displacement_bruteforce(A[i] - A_star[i], self.mask, SPACING)
                        for i in range(A.shape[0])])

    def test_center_loss_matches_bruteforce(self):
        ref = self.bf_center(self.A, self.A_star)
        for dtype, rtol in ((tf.float64, 1e-10), (tf.float32, 1e-5)):
            got = losses.linear_center_loss(tf.constant(self.A, dtype), tf.constant(self.A_star, dtype),
                                            self.S).numpy()
            np.testing.assert_allclose(got, ref, rtol=rtol)

    def test_center_loss_units_mm2(self):
        a = np.tile(np.eye(4), (3, 1, 1))
        b = a.copy()
        b[:, :3, 3] = [3.0, 0.0, 0.0]          # pure 3 mm translation
        terms = losses.linear_loss(tf.constant(b), {"A_star": a, "T_target": a,
                                                    "pair_i": [0], "pair_j": [1]}, self.S)
        self.assertAlmostEqual(float(terms["center"]), 9.0, places=9)
        self.assertAlmostEqual(float(terms["center_rms_mm"]), 3.0, places=9)
        self.assertAlmostEqual(float(terms["pair"]), 0.0, places=9)

    def test_pair_loss_matches_bruteforce(self):
        i = np.array([0, 3, 2, 4, 1])
        j = np.array([1, 0, 4, 2, 3])
        ref = np.mean([npr.mean_sq_displacement_bruteforce(
            np.linalg.inv(self.A[b]) @ self.A[a] - self.T[b] @ np.linalg.inv(self.T[a]), self.mask, SPACING)
            for a, b in zip(i, j)])
        got64 = losses.linear_pair_loss(tf.constant(self.A), tf.constant(self.T), i, j, self.S).numpy()
        np.testing.assert_allclose(got64, ref, rtol=1e-10)
        got32 = losses.linear_pair_loss(tf.constant(self.A, tf.float32), tf.constant(self.T, tf.float32),
                                        i, j, self.S).numpy()
        np.testing.assert_allclose(got32, ref, rtol=1e-4)
        # P_star from postprocess-style float64 pair targets gives the same value
        p_star = tfm.pair_targets(tf.constant(self.T), i, j)
        got_p = losses.linear_pair_loss(tf.constant(self.A), None, i, j, self.S, P_star=p_star).numpy()
        np.testing.assert_allclose(got_p, ref, rtol=1e-10)

    def test_total_and_rms(self):
        i, j = np.array([0, 1, 2]), np.array([4, 3, 0])
        batch = {"A_star": self.A_star, "T_target": self.T, "pair_i": i, "pair_j": j}
        terms = losses.linear_loss(tf.constant(self.A), batch, self.S, lambda_pair=0.7)
        center = float(terms["center"])
        pair = float(terms["pair"])
        np.testing.assert_allclose(center, self.bf_center(self.A, self.A_star), rtol=1e-10)
        np.testing.assert_allclose(float(terms["loss"]), center + 0.7 * pair, rtol=1e-12)
        np.testing.assert_allclose(float(terms["center_rms_mm"]), np.sqrt(center), rtol=1e-12)
        np.testing.assert_allclose(float(terms["pair_rms_mm"]), np.sqrt(pair), rtol=1e-12)

    def test_gradients_finite(self):
        a = tf.Variable(self.A.astype(np.float32))
        batch = {"A_star": self.A_star.astype(np.float32), "T_target": self.T.astype(np.float32),
                 "pair_i": [0, 1], "pair_j": [2, 3]}
        with tf.GradientTape() as tape:
            loss = losses.linear_loss(a, batch, self.S)["loss"]
        g = tape.gradient(loss, a).numpy()
        self.assertTrue(np.all(np.isfinite(g)))
        self.assertGreater(np.abs(g).sum(), 0.0)


class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(1)
        cls.S = tfm.brain_second_moment(synthetic_mask(cls.rng), SPACING)

    def test_rotation_angle(self):
        for deg in (0.0, 0.5, 13.0, 45.0, 120.0, 179.0):
            r = rot(self.rng.normal(size=3), deg)
            self.assertAlmostEqual(float(M.rotation_angle_deg(r)), deg, delta=1e-6)
        # polar part: a scaled/sheared rotation reports the rotation angle
        r = rot([0, 0, 1], 30.0)
        self.assertAlmostEqual(float(M.rotation_angle_deg(r @ np.diag([1.0, 1.0, 1.0]) * 1.3)), 30.0, delta=1e-6)

    def test_matrix_errors(self):
        a, b = np.eye(4), np.eye(4)
        a[:3, :3] = rot([1, 2, 3], 7.0)
        a[:3, 3] = [1.0, -2.0, 2.0]
        b[:3, 3] = [0.0, 0.0, 0.0]
        e = M.matrix_errors(a, b, self.S)
        self.assertAlmostEqual(float(e["rot_deg"]), 7.0, delta=1e-6)
        self.assertAlmostEqual(float(e["trans_mm"]), 3.0, delta=1e-12)
        np.testing.assert_allclose(float(e["rms_mm"]) ** 2, np.trace((a - b) @ self.S @ (a - b).T), rtol=1e-12)

    def test_pairwise_all_ordered_pairs(self):
        n = 5
        A = np.stack([random_affine(self.rng) for _ in range(n)])
        T = np.stack([random_affine(self.rng) for _ in range(n)])
        e = M.pairwise_errors(A, T, self.S)
        self.assertEqual(e["rms_mm"].shape, (n * (n - 1),))
        ref_rms, ref_rot, ref_tr = [], [], []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pred = np.linalg.inv(A[j]) @ A[i]
                true = T[j] @ np.linalg.inv(T[i])
                d = pred - true
                ref_rms.append(np.sqrt(np.trace(d @ self.S @ d.T)))
                ref_tr.append(np.linalg.norm(d[:3, 3]))
                ref_rot.append(M.rotation_angle_deg(M.polar_rotation(pred[:3, :3]) @ M.polar_rotation(true[:3, :3]).T))
        np.testing.assert_allclose(e["rms_mm"], ref_rms, rtol=1e-10)
        np.testing.assert_allclose(e["trans_mm"], ref_tr, rtol=1e-10)
        np.testing.assert_allclose(e["rot_deg"], ref_rot, atol=1e-8)

    def test_cycle_error_zero(self):
        A = np.stack([random_affine(self.rng) for _ in range(6)])
        c = M.cycle_errors(A, self.S, n_triplets=50, seed=3)
        self.assertEqual(c["triplets"].shape, (50, 3))
        for t in c["triplets"]:
            self.assertEqual(len(set(t.tolist())), 3)
        self.assertLess(float(np.max(c["rms_mm"])), 1e-6)
        c32 = M.cycle_errors(A.astype(np.float32), self.S, n_triplets=50, seed=3)
        self.assertLess(float(np.max(c32["rms_mm"])), 1e-3)
        self.assertEqual(M.cycle_errors(A[:2], self.S)["rms_mm"].size, 0)

    def test_group_metrics_keys_and_accumulator(self):
        A = np.stack([random_affine(self.rng) for _ in range(4)])
        A_star = np.stack([random_affine(self.rng) for _ in range(4)])
        T = np.stack([random_affine(self.rng) for _ in range(4)])
        m = M.group_metrics(A, A_star, T, self.S, "val_rigid")
        for k in ("center_rms_mm", "center_rot_deg", "center_trans_mm", "pair_rms_mm", "pair_rot_deg",
                  "pair_trans_mm", "cycle_rms_mm", "center_rms_mm_p95", "pair_rms_mm_p95"):
            self.assertIn("val_rigid/" + k, m)
        ce = M.centered_errors(A, A_star, self.S)
        self.assertAlmostEqual(m["val_rigid/center_rms_mm"], float(np.mean(ce["rms_mm"])), places=12)
        acc = M.MeanAccumulator()
        acc.update({"a": 1.0, "b": 2.0})
        acc.update({"a": 3.0})
        self.assertEqual(acc.result(), {"a": 2.0, "b": 2.0})
        acc.reset()
        self.assertEqual(acc.result(), {})


def known_affine(rng, deg=40.0, ls=0.1, sh=0.05, mm=10.0):
    """[R U D | t] from known factors -> (matrix, R, log-scales, shears (a, b, c))."""
    r = rot(rng.normal(size=3), rng.uniform(-deg, deg))
    s = rng.uniform(-ls, ls, 3)
    a, b, c = rng.uniform(-sh, sh, 3)
    u = np.array([[1.0, a, b], [0.0, 1.0, c], [0.0, 0.0, 1.0]])
    m = np.eye(4)
    m[:3, :3] = r @ u @ np.diag(np.exp(s))
    m[:3, 3] = rng.uniform(-mm, mm, 3)
    return m, r, s, np.array([a, b, c])


def ref_angle_deg(r1, r2):
    c = (np.trace(r1 @ r2.T) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


class TestAffineMetrics(unittest.TestCase):
    """Milestone 5: scale / shear / rotation-factor errors (QR, the model's A = [R K | t])."""

    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(11)
        cls.S = tfm.brain_second_moment(synthetic_mask(cls.rng), SPACING)

    def test_qr_factors_recover_known_factors(self):
        items = [known_affine(self.rng) for _ in range(6)]
        L = np.stack([m[:3, :3] for m, _, _, _ in items]).reshape(2, 3, 3, 3)    # batched shape
        q, ls, sh = M.qr_factors(L)
        self.assertEqual((q.shape, ls.shape, sh.shape), ((2, 3, 3, 3), (2, 3, 3), (2, 3, 3)))
        for n, (_, r, s, k) in enumerate(items):
            i, j = divmod(n, 3)
            np.testing.assert_allclose(q[i, j], r, atol=1e-12)
            np.testing.assert_allclose(ls[i, j], s, atol=1e-12)
            np.testing.assert_allclose(sh[i, j], k, atol=1e-12)
        q, ls, sh = M.qr_factors(np.eye(3))
        np.testing.assert_allclose(q, np.eye(3), atol=1e-15)
        np.testing.assert_allclose(np.concatenate([ls, sh]), 0.0, atol=1e-15)

    def test_affine_matrix_errors_reference(self):
        pred = [known_affine(self.rng) for _ in range(5)]
        true = [known_affine(self.rng) for _ in range(5)]
        e = M.affine_matrix_errors(np.stack([p[0] for p in pred]), np.stack([t[0] for t in true]))
        for n, (p, t) in enumerate(zip(pred, true)):
            self.assertAlmostEqual(e["scale_pct"][n], 100.0 * np.mean(np.abs(p[2] - t[2])), places=9)
            self.assertAlmostEqual(e["shear_pct"][n], 100.0 * np.mean(np.abs(p[3] - t[3])), places=9)
            self.assertAlmostEqual(e["qr_rot_deg"][n], ref_angle_deg(p[1], t[1]), places=6)
        z = M.affine_matrix_errors(pred[0][0], pred[0][0])
        for k in ("scale_pct", "shear_pct", "qr_rot_deg"):
            self.assertLess(float(z[k]), 1e-6, k)
        # 5% scale on one axis only -> 100 * log(1.05) / 3 percent
        m = np.eye(4)
        m[0, 0] = 1.05
        self.assertAlmostEqual(float(M.affine_matrix_errors(m, np.eye(4))["scale_pct"]),
                               100.0 * np.log(1.05) / 3.0, places=10)

    def test_group_errors_affine_keys_and_pairs(self):
        n = 4
        A = np.stack([known_affine(self.rng)[0] for _ in range(n)])
        A_star = np.stack([known_affine(self.rng)[0] for _ in range(n)])
        T = np.stack([known_affine(self.rng)[0] for _ in range(n)])
        rigid = M.group_errors(A, A_star, T, self.S)
        self.assertFalse(any("scale" in k or "shear" in k or "qr_rot" in k for k in rigid))
        aff = M.group_errors(A, A_star, T, self.S, affine=True)
        for k in rigid:
            np.testing.assert_array_equal(aff[k], rigid[k])
        ce = M.affine_matrix_errors(A, A_star)
        for k in ("scale_pct", "shear_pct", "qr_rot_deg"):
            np.testing.assert_allclose(aff["center_" + k], ce[k], rtol=1e-12)
        ref = {k: [] for k in ("scale_pct", "shear_pct", "qr_rot_deg")}
        for i in range(n):                         # brute force over ordered pairs, row-major
            for j in range(n):
                if i != j:
                    e = M.affine_matrix_errors(np.linalg.inv(A[j]) @ A[i], T[j] @ np.linalg.inv(T[i]))
                    for k in ref:
                        ref[k].append(float(e[k]))
        for k in ref:
            np.testing.assert_allclose(aff["pair_" + k], ref[k], rtol=1e-10, atol=1e-12)
        m = M.summarize(aff, "val_affine")
        for k in ("center_scale_pct", "center_shear_pct", "center_qr_rot_deg", "pair_scale_pct",
                  "pair_shear_pct", "pair_qr_rot_deg"):
            self.assertIn("val_affine/" + k, m)

    def test_mean_pair_rms(self):
        m = {"val_rigid/pair_rms_mm": 2.0, "val_affine/pair_rms_mm": 4.0, "val_rigid_g/pair_rms_mm": 100.0}
        self.assertEqual(M.mean_pair_rms(m, ["rigid", "affine"]), 3.0)
        self.assertEqual(M.mean_pair_rms(m, ["rigid"]), 2.0)
        self.assertIsNone(M.mean_pair_rms({"val_rigid/pair_rms_mm": 2.0}, ["rigid", "affine"]))


if __name__ == "__main__":
    unittest.main()
