"""Linear and deformable losses (runtime.losses) and metrics (runtime.metrics) against numpy
references."""
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests import np_reference as npr
import numpy as np
import tensorflow as tf
from scipy.linalg import expm as sp_expm

from data_loading import params as P
from models import diffeo
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


# ============================================================================ deformable (m6)
def np_soft_dice(a, b, eps=1e-5):
    """Numpy reference of losses.soft_dice (foreground channels, masked mean over present)."""
    a, b = np.asarray(a, np.float64)[..., 1:], np.asarray(b, np.float64)[..., 1:]
    out = []
    for n in range(a.shape[0]):
        vals = []
        for c in range(a.shape[-1]):
            sa, sb = a[n, ..., c].sum(), b[n, ..., c].sum()
            if sa + sb > 1e-3:
                vals.append((2.0 * (a[n, ..., c] * b[n, ..., c]).sum() + eps) / (sa + sb + eps))
        out.append(np.mean(vals) if vals else 1.0)
    return np.asarray(out)


def np_grad_l2(u):
    u = np.asarray(u, np.float64)
    return np.mean([np.mean(np.diff(u, axis=a + 1) ** 2) for a in range(3)])


def np_bending(u):
    """Plain numpy re-implementation of the bending energy: per interior voxel
    sum_c [u_xx^2 + u_yy^2 + u_zz^2 + 2 (u_xy^2 + u_xz^2 + u_yz^2)], mean over N, voxels, c."""
    u = np.asarray(u, np.float64)
    n, X, Y, Z, C = u.shape
    tot, cnt = 0.0, 0
    for x in range(1, X - 1):
        for y in range(1, Y - 1):
            for z in range(1, Z - 1):
                f = u[:, x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]      # [n, 3, 3, 3, C]
                dxx = f[:, 0, 1, 1] - 2 * f[:, 1, 1, 1] + f[:, 2, 1, 1]
                dyy = f[:, 1, 0, 1] - 2 * f[:, 1, 1, 1] + f[:, 1, 2, 1]
                dzz = f[:, 1, 1, 0] - 2 * f[:, 1, 1, 1] + f[:, 1, 1, 2]
                dxy = (f[:, 0, 0, 1] - f[:, 2, 0, 1] - f[:, 0, 2, 1] + f[:, 2, 2, 1]) / 4
                dxz = (f[:, 0, 1, 0] - f[:, 2, 1, 0] - f[:, 0, 1, 2] + f[:, 2, 1, 2]) / 4
                dyz = (f[:, 1, 0, 0] - f[:, 1, 2, 0] - f[:, 1, 0, 2] + f[:, 1, 2, 2]) / 4
                tot += np.sum(dxx ** 2 + dyy ** 2 + dzz ** 2 + 2 * (dxy ** 2 + dxz ** 2 + dyz ** 2))
                cnt += n * C
    return tot / cnt


def block_labels(shape, n_classes, rng):
    """Integer map with a few axis-aligned blocks of foreground labels away from the border."""
    lab = np.zeros(shape, np.int64)
    for k in range(1, n_classes):
        lo = [int(rng.integers(2, s // 2)) for s in shape]
        hi = [min(l + int(rng.integers(3, s // 2)), s - 3) for l, s in zip(lo, shape)]
        lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = k
    return lab


class TestDeformLosses(unittest.TestCase):
    SHAPE = (12, 14, 10)

    def setUp(self):
        self.rng = np.random.default_rng(5)

    def onehot(self, lab, c):
        return np.eye(c, dtype=np.float32)[lab]

    def test_soft_dice_matches_numpy_and_excludes_absent(self):
        a = self.rng.uniform(size=(3,) + self.SHAPE + (5,)).astype(np.float32)
        b = self.rng.uniform(size=(3,) + self.SHAPE + (5,)).astype(np.float32)
        a[0, ..., 3] = 0.0
        b[0, ..., 3] = 0.0                       # channel 3 absent in image 0
        got = losses.soft_dice(tf.constant(a), tf.constant(b)).numpy()
        np.testing.assert_allclose(got, np_soft_dice(a, b), rtol=1e-5)
        # Counted, the absent channel would add a Dice of 1 (eps / eps) and inflate the mean.
        inflated = (np_soft_dice(a, b)[0] * 3 + 1.0) / 4.0
        self.assertLess(got[0], inflated - 0.05)
        # No present foreground channel at all -> 1.
        z = np.zeros((1,) + self.SHAPE + (3,), np.float32)
        z[..., 0] = 1.0
        self.assertEqual(float(losses.soft_dice(z, z)[0]), 1.0)
        # Background included on request.
        full = losses.soft_dice(tf.constant(a), tf.constant(b), exclude_background=False).numpy()
        self.assertFalse(np.allclose(full, got))

    def test_perfect_alignment_zero_and_shift(self):
        c = 4
        lab = block_labels(self.SHAPE, c, self.rng)
        oh = np.stack([self.onehot(lab, c)] * 3)
        zeros = tf.zeros((3,) + self.SHAPE + (3,))
        tmpl, dice_t, template = losses.template_dice_loss(diffeo.warp(oh, zeros))
        self.assertLess(float(tmpl), 1e-6)
        np.testing.assert_allclose(template.numpy()[0], oh[0], atol=1e-7)
        pdice, dice_p = losses.pair_dice_loss(oh, zeros, zeros, [0, 1, 2], [1, 2, 0])
        self.assertLess(float(pdice), 1e-6)
        self.assertEqual(dice_p.shape, (3,))
        # Image 1 is image 0 shifted by one voxel along x: unregistered Dice = numpy Dice.
        shifted = np.zeros_like(lab)
        shifted[:-1] = lab[1:]                   # shifted(x) = lab(x + 1)
        oh2 = np.stack([self.onehot(lab, c), self.onehot(shifted, c)])
        z2 = tf.zeros((2,) + self.SHAPE + (3,))
        _, d = losses.pair_dice_loss(oh2, z2, z2, [0], [1])
        np.testing.assert_allclose(d.numpy(), np_soft_dice(oh2[:1], oh2[1:]), rtol=1e-5)
        self.assertLess(float(d[0]), 0.95)
        # u_inv_1 = +1 voxel along x maps 1's coords to 0's: onehot_0(x + 1) = onehot_1(x).
        u_inv = np.zeros((2,) + self.SHAPE + (3,), np.float32)
        u_inv[1, ..., 0] = 1.0
        _, d = losses.pair_dice_loss(oh2, z2, tf.constant(u_inv), [0], [1])
        self.assertGreater(float(d[0]), 1.0 - 1e-6)

    def test_template_gradient_flows_through_template(self):
        c = 3
        labs = [block_labels(self.SHAPE, c, self.rng) for _ in range(3)]
        oh = tf.constant(np.stack([self.onehot(l, c) for l in labs]))
        u = tf.Variable(self.rng.normal(scale=0.3, size=(3,) + self.SHAPE + (3,)).astype(np.float32))

        def grads(stop):
            with tf.GradientTape() as tape:
                w = diffeo.warp(oh, u)
                if stop:
                    t = tf.stop_gradient(tf.reduce_mean(w, axis=0, keepdims=True))
                    loss = tf.reduce_mean(1.0 - losses.soft_dice(w, tf.broadcast_to(t, tf.shape(w))))
                else:
                    loss = losses.template_dice_loss(w)[0]
            return tape.gradient(loss, u).numpy()

        g, g_stop = grads(False), grads(True)
        for i in range(3):
            self.assertGreater(float(np.abs(g[i]).sum()), 0.0)
        self.assertTrue(np.all(np.isfinite(g)))
        self.assertGreater(float(np.abs(g - g_stop).max()), 1e-6)   # the template term contributes

    def test_smoothness_orders(self):
        u = self.rng.normal(size=(2,) + self.SHAPE + (3,))
        np.testing.assert_allclose(losses.smoothness_loss(tf.constant(u), 1).numpy(), np_grad_l2(u),
                                   rtol=1e-10)
        np.testing.assert_allclose(losses.smoothness_loss(tf.constant(u), 2).numpy(), np_bending(u),
                                   rtol=1e-10)
        self.assertEqual(losses.smoothness_loss(tf.constant(u, tf.float32), 1).dtype, tf.float32)
        const = np.ones((2,) + self.SHAPE + (3,)) * np.array([0.5, -1.0, 2.0])
        for order in (1, 2):
            self.assertEqual(float(losses.smoothness_loss(tf.constant(const), order)), 0.0)
        # Affine field u = B x + b: bending energy 0; first order = mean_a sum_c B[c, a]^2 / 3.
        g = np.stack(np.meshgrid(*[np.arange(n, dtype=np.float64) for n in self.SHAPE],
                                 indexing="ij"), -1)
        B = self.rng.normal(scale=0.1, size=(3, 3))
        aff = (g @ B.T + np.array([0.3, 0.1, -0.2]))[None]
        self.assertLess(float(losses.smoothness_loss(tf.constant(aff), 2)), 1e-20)
        np.testing.assert_allclose(float(losses.smoothness_loss(tf.constant(aff), 1)),
                                   np.sum(B ** 2) / 9.0, rtol=1e-10)
        with self.assertRaises(ValueError):
            losses.smoothness_loss(tf.constant(u), 3)

    def test_total_loss(self):
        c = 4
        labs = [block_labels(self.SHAPE, c, self.rng) for _ in range(3)]
        oh = np.stack([self.onehot(l, c) for l in labs])
        v = tf.constant(self.rng.normal(scale=0.3, size=(3,) + self.SHAPE + (3,)).astype(np.float32))
        u, u_inv = diffeo.exp_and_inverse(v, 4)
        batch = {"onehot": oh, "pair_i": tf.constant([0, 2]), "pair_j": tf.constant([1, 0])}
        terms = losses.deform_loss({"u": u, "u_inv": u_inv}, batch, lambda_pdice=0.7,
                                   lambda_smooth=0.3, smooth_order=2)
        for k in ("loss", "tmpl", "pdice", "smooth", "soft_dice_tmpl", "soft_dice_pair"):
            self.assertEqual(terms[k].dtype, tf.float32, k)
            self.assertTrue(np.isfinite(float(terms[k])), k)
        np.testing.assert_allclose(float(terms["loss"]), float(terms["tmpl"]) + 0.7 * float(terms["pdice"])
                                   + 0.3 * float(terms["smooth"]), rtol=1e-6)
        tmpl, dice_t, _ = losses.template_dice_loss(diffeo.warp(oh, u))
        np.testing.assert_allclose(float(terms["tmpl"]), float(tmpl), rtol=1e-6)
        np.testing.assert_allclose(float(terms["soft_dice_tmpl"]), float(np.mean(dice_t)), rtol=1e-6)
        np.testing.assert_allclose(float(terms["smooth"]), float(losses.smoothness_loss(u, 2)), rtol=1e-6)
        pd, _ = losses.pair_dice_loss(oh, u, u_inv, [0, 2], [1, 0])
        np.testing.assert_allclose(float(terms["pdice"]), float(pd), rtol=1e-6)
        np.testing.assert_allclose(float(terms["soft_dice_pair"]), 1.0 - float(pd), rtol=1e-6)

    # ---------------------------------------------------------------- LNCC wiring (v3.9)
    def ncc_case(self, c=4, n=3):
        labs = [block_labels(self.SHAPE, c, self.rng) for _ in range(n)]
        oh = np.stack([self.onehot(l, c) for l in labs])
        # Intensities: a per-label value + noise (zero background), so LNCC sees edges.
        levels = np.concatenate([[0.0], self.rng.uniform(0.3, 1.0, c - 1)])
        imgs = (levels[np.stack(labs)] + 0.05 * self.rng.normal(size=(n,) + self.SHAPE))[..., None]
        imgs = np.where(np.stack(labs)[..., None] > 0, imgs, 0.0).astype(np.float32)
        v = tf.constant(self.rng.normal(scale=0.3, size=(n,) + self.SHAPE + (3,)).astype(np.float32))
        u, u_inv = diffeo.exp_and_inverse(v, 4)
        batch = {"onehot": oh, "images": imgs, "pair_i": tf.constant([0, 2]), "pair_j": tf.constant([1, 0])}
        return {"u": u, "u_inv": u_inv}, batch

    def test_ncc_defaults_unchanged(self):
        out, batch = self.ncc_case()
        ref = losses.deform_loss(out, batch)
        same = losses.deform_loss(out, batch, 1.0, 0.5, 1, lambda_dice_tmpl=1.0, lambda_ncc_tmpl=0.0,
                                  lambda_ncc_pair=0.0, ncc_window=3)
        self.assertEqual(sorted(ref), sorted(same))
        self.assertEqual(sorted(ref), sorted(("loss", "tmpl", "pdice", "smooth", "soft_dice_tmpl",
                                              "soft_dice_pair")))
        for k in ref:
            self.assertEqual(float(ref[k]), float(same[k]), k)        # bit-identical
        no_img = {k: v for k, v in batch.items() if k != "images"}
        self.assertEqual(float(losses.deform_loss(out, no_img)["loss"]), float(ref["loss"]))
        with self.assertRaisesRegex(ValueError, "images"):
            losses.deform_loss(out, no_img, lambda_ncc_pair=1.0)

    def test_ncc_total_and_keys(self):
        out, batch = self.ncc_case()
        t = losses.deform_loss(out, batch, lambda_pdice=0.7, lambda_smooth=0.3, smooth_order=1,
                               lambda_dice_tmpl=0.5, lambda_ncc_tmpl=0.4, lambda_ncc_pair=0.6, ncc_window=3)
        for k in ("loss", "tmpl", "pdice", "smooth", "soft_dice_tmpl", "soft_dice_pair", "ncc_tmpl",
                  "ncc_pair", "ncc_tmpl_loss", "ncc_pair_loss"):
            self.assertEqual(t[k].dtype, tf.float32, k)
            self.assertTrue(np.isfinite(float(t[k])), k)
        f = {k: float(v) for k, v in t.items()}
        np.testing.assert_allclose(f["loss"], 0.5 * f["tmpl"] + 0.7 * f["pdice"] + 0.3 * f["smooth"]
                                   + 0.4 * f["ncc_tmpl_loss"] + 0.6 * f["ncc_pair_loss"], rtol=1e-6)
        for k in ("ncc_tmpl", "ncc_pair"):
            self.assertGreater(f[k], 0.0)
            self.assertLessEqual(f[k], 1.0)
            np.testing.assert_allclose(f[k + "_loss"], 1.0 - f[k], rtol=1e-6)
        # Only the active term's keys.
        only_t = losses.deform_loss(out, batch, lambda_ncc_tmpl=1.0, ncc_window=3)
        self.assertIn("ncc_tmpl", only_t)
        self.assertNotIn("ncc_pair", only_t)
        # The pair term with the shared displacement equals a separate recomputation.
        oh = tf.constant(batch["onehot"])
        disp = diffeo.pairwise_displacement(out["u"], out["u_inv"], batch["pair_i"], batch["pair_j"])
        moved = diffeo.warp(tf.gather(oh, batch["pair_i"]), disp)
        _, ncc_p = losses.pair_ncc_loss(batch["images"], disp, batch["pair_i"], batch["pair_j"],
                                        losses.foreground(moved), losses.foreground(tf.gather(oh, batch["pair_j"])), 3)
        np.testing.assert_allclose(f["ncc_pair"], float(np.mean(ncc_p)), rtol=1e-6)
        w = diffeo.warp(oh, out["u"])
        _, ncc_t = losses.template_ncc_loss(diffeo.warp(batch["images"], out["u"]),
                                            losses.foreground(tf.reduce_mean(w, axis=0, keepdims=True)), 3)
        np.testing.assert_allclose(f["ncc_tmpl"], float(np.mean(ncc_t)), rtol=1e-6)

    def test_dice_template_weight_zero(self):
        out, batch = self.ncc_case()
        u = tf.Variable(out["u"])
        u_inv = tf.constant(out["u_inv"])
        with tf.GradientTape(persistent=True) as tape:
            t0 = losses.deform_loss({"u": u, "u_inv": u_inv}, batch, lambda_dice_tmpl=0.0)
            ref = (losses.pair_dice_loss(batch["onehot"], u, u_inv, batch["pair_i"], batch["pair_j"])[0]
                   + 0.5 * losses.smoothness_loss(u, 1))
        g0, gr = tape.gradient(t0["loss"], u).numpy(), tape.gradient(ref, u).numpy()
        np.testing.assert_allclose(g0, gr, rtol=1e-5, atol=1e-9)
        self.assertGreater(float(t0["soft_dice_tmpl"]), 0.0)          # still computed and logged


def np_local_ncc(a, b, k, eps=losses.NCC_EPS, signed=False):
    """float64 numpy reference of losses.local_ncc for one [X, Y, Z] pair (zero padding, true
    window counts)."""
    from scipy import ndimage
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    cnt = ndimage.uniform_filter(np.ones_like(a), k, mode="constant")
    m = lambda x: ndimage.uniform_filter(x, k, mode="constant") / cnt  # noqa: E731
    ma, mb = m(a), m(b)
    cov = m(a * b) - ma * mb
    va, vb = np.maximum(m(a * a) - ma ** 2, 0.0), np.maximum(m(b * b) - mb ** 2, 0.0)
    return cov / np.sqrt(va * vb + eps) if signed else cov ** 2 / (va * vb + eps)


def smooth_noise(rng, shape, sigma=1.0):
    from scipy import ndimage
    x = ndimage.gaussian_filter(rng.normal(size=shape), sigma)
    return (x - x.mean()) / x.std()


class TestLNCC(unittest.TestCase):
    SHAPE = (12, 14, 10)

    def setUp(self):
        self.rng = np.random.default_rng(11)

    def test_box_sum_and_counts(self):
        from scipy import ndimage
        x = self.rng.uniform(size=(2,) + self.SHAPE + (2,)).astype(np.float32)
        for k in (3, 5):
            s = losses.box_sum(tf.constant(x), k).numpy()
            self.assertEqual(s.shape, x.shape)
            cnt = losses.window_count(self.SHAPE, k)
            np.testing.assert_allclose(losses.box_sum(tf.ones((1,) + self.SHAPE + (1,)), k).numpy()[0, ..., 0],
                                       cnt, rtol=1e-6)
            ones = ndimage.uniform_filter(np.ones(self.SHAPE), k, mode="constant")
            for b in range(2):
                for c in range(2):
                    want = ndimage.uniform_filter(x[b, ..., c].astype(np.float64), k, mode="constant") / ones
                    np.testing.assert_allclose(s[b, ..., c] / cnt, want, rtol=1e-5, err_msg="k %d" % k)
            self.assertEqual(cnt[0, 0, 0], (k // 2 + 1) ** 3)                  # corner
            self.assertEqual(cnt[6, 7, 5], k ** 3)                             # interior

    def test_local_ncc_matches_numpy(self):
        a = smooth_noise(self.rng, (2,) + self.SHAPE, 0.7)
        b = 0.6 * a + 0.8 * smooth_noise(self.rng, (2,) + self.SHAPE, 0.7)
        for k in (3, 5):
            for signed in (False, True):
                got = losses.local_ncc(tf.constant(a, tf.float32), tf.constant(b, tf.float32), k,
                                       signed=signed).numpy()
                self.assertEqual(got.shape, (2,) + self.SHAPE)
                for i in range(2):
                    np.testing.assert_allclose(got[i], np_local_ncc(a[i], b[i], k, signed=signed),
                                               rtol=1e-4, atol=1e-5, err_msg="k %d signed %s" % (k, signed))

    def test_affine_intensity_invariance(self):
        a = self.rng.uniform(size=(1,) + self.SHAPE).astype(np.float32)
        from scipy import ndimage
        cnt = ndimage.uniform_filter(np.ones(self.SHAPE), 5, mode="constant")
        m = lambda x: ndimage.uniform_filter(x, 5, mode="constant") / cnt  # noqa: E731
        var = m(a[0].astype(np.float64) ** 2) - m(a[0].astype(np.float64)) ** 2
        mask = var > 1e-4
        self.assertGreater(mask.mean(), 0.9)
        for b, sq, sg in ((2.0 * a + 0.3, 1.0, 1.0), (-a, 1.0, -1.0)):
            cc = losses.local_ncc(a, b, 5).numpy()[0]
            cs = losses.local_ncc(a, b, 5, signed=True).numpy()[0]
            np.testing.assert_allclose(cc[mask], sq, atol=1e-4)
            np.testing.assert_allclose(cs[mask], sg, atol=1e-4)
        # Self-similarity = 1 in the masked mean.
        roi = np.ones((1,) + self.SHAPE, np.float32)
        self.assertGreater(float(losses.masked_mean(losses.local_ncc(a, a, 5), roi)[0]), 1.0 - 1e-4)
        # Independent noise ~ 0.
        b = self.rng.uniform(size=(1,) + self.SHAPE).astype(np.float32)
        self.assertLess(float(losses.masked_mean(losses.local_ncc(a, b, 5), roi)[0]), 0.05)

    def blocks(self, n=2):
        """Constant blocks (value per image) on a zero background, and their foreground."""
        imgs = np.zeros((n,) + self.SHAPE + (1,), np.float32)
        imgs[:, 3:9, 4:10, 3:7] = np.linspace(0.3, 0.9, n)[:, None, None, None, None]
        fg = (imgs[..., 0] > 0).astype(np.float32)
        return imgs, fg

    def test_flat_regions_finite(self):
        imgs, fg = self.blocks(3)
        cc = losses.local_ncc(imgs[:1], imgs[1:2], 3).numpy()[0]
        self.assertTrue(np.all(np.isfinite(cc)))
        self.assertLess(float(np.abs(cc[5, 6, 4])), 1e-3)          # both flat (block interior)
        self.assertLess(float(np.abs(cc[0, 0, 0])), 1e-3)          # both flat (background)
        self.assertGreater(float(cc[3, 6, 4]), 0.9)                 # at the block edge: aligned
        u = tf.Variable(0.2 * self.rng.normal(size=(3,) + self.SHAPE + (3,)).astype(np.float32))
        x = tf.Variable(imgs)
        with tf.GradientTape(persistent=True) as tape:
            lt, _ = losses.template_ncc_loss(diffeo.warp(x, u), fg[:1], 3)
            disp = diffeo.pairwise_displacement(u, -u, [0, 1], [1, 2])
            lp, _ = losses.pair_ncc_loss(x, disp, [0, 1], [1, 2], fg[:2], fg[1:], 3)
            total = lt + lp
        for g in (tape.gradient(total, u), tape.gradient(total, x)):
            self.assertTrue(np.all(np.isfinite(g.numpy())))
            self.assertGreater(float(np.abs(g.numpy()).sum()), 0.0)
        # All-zero images: finite loss (cc = 0 -> loss 1) and finite gradients.
        z = tf.Variable(np.zeros_like(imgs))
        with tf.GradientTape() as tape:
            lz, ncc = losses.template_ncc_loss(diffeo.warp(z, u), fg[:1], 3)
        self.assertAlmostEqual(float(lz), 1.0, places=6)
        self.assertTrue(np.all(np.isfinite(tape.gradient(lz, [z, u])[1].numpy())))

    def test_roi_excludes_far_background_and_border(self):
        imgs, fg = self.blocks(3)
        imgs = imgs + 0.1 * self.rng.uniform(size=imgs.shape).astype(np.float32) * (imgs > 0)
        roi = losses.ncc_roi(fg[:1], 3).numpy()[0]
        np.testing.assert_array_equal(roi[2:10, 3:11, 2:8], 1.0)    # dilated by k // 2 = 1
        self.assertEqual(float(roi.sum()), 8 * 8 * 6)
        loss0, _ = losses.template_ncc_loss(imgs, fg[:1], 3)
        # Noise more than 2 * (k // 2) voxels from the foreground never enters an ROI window.
        far = np.ones(self.SHAPE, bool)
        far[1:11, 2:12, 1:9] = False
        noisy = imgs + (self.rng.uniform(size=imgs.shape) * far[None, ..., None]).astype(np.float32)
        loss1, _ = losses.template_ncc_loss(noisy, fg[:1], 3)
        np.testing.assert_allclose(float(loss1), float(loss0), rtol=1e-6)
        # Foreground touching the volume border: finite.
        edge = np.zeros_like(imgs)
        edge[:, :4] = self.rng.uniform(size=edge[:, :4].shape)
        lb, nb = losses.template_ncc_loss(edge, (edge[:1, ..., 0] > 0).astype(np.float32), 5)
        self.assertTrue(np.isfinite(float(lb)))
        self.assertTrue(np.all(np.isfinite(nb.numpy())))

    def test_leave_one_out(self):
        roi = np.ones((1,) + self.SHAPE, np.float32)
        noise = self.rng.uniform(size=(4,) + self.SHAPE + (1,)).astype(np.float32)
        _, loo = losses.template_ncc_loss(noise, roi, 5)
        _, inc = losses.template_ncc_loss(noise, roi, 5, leave_one_out=False)
        self.assertLess(float(np.mean(loo)), 0.05)
        self.assertGreater(float(np.mean(inc)), 0.15)                # the ~1/N self-term floor
        same = np.repeat(noise[:1], 4, axis=0)
        _, one = losses.template_ncc_loss(same, roi, 5)
        np.testing.assert_allclose(one.numpy(), 1.0, atol=1e-4)

    def test_template_gradient_flows_through_template(self):
        imgs, fg = self.blocks(3)
        imgs = imgs + 0.1 * self.rng.uniform(size=imgs.shape).astype(np.float32)
        u = tf.Variable(0.3 * self.rng.normal(size=(3,) + self.SHAPE + (3,)).astype(np.float32))
        roi = losses.ncc_roi(fg[:1], 3)

        def grads(stop):
            with tf.GradientTape() as tape:
                w = diffeo.warp(imgs, u)
                if stop:
                    t = tf.stop_gradient((tf.reduce_sum(w, axis=0, keepdims=True) - w) / 2.0)
                    loss = tf.reduce_mean(1.0 - losses.masked_mean(losses.local_ncc(w, t, 3), roi))
                else:
                    loss = losses.template_ncc_loss(w, fg[:1], 3)[0]
            return tape.gradient(loss, u).numpy()

        g, g_stop = grads(False), grads(True)
        self.assertTrue(np.all(np.isfinite(g)))
        self.assertGreater(float(np.abs(g - g_stop).max()), 1e-6)

    def test_pair_shift(self):
        img0 = smooth_noise(self.rng, self.SHAPE, 1.0)
        img1 = np.zeros_like(img0)
        img1[:-1] = img0[1:]                          # img1(x) = img0(x + 1)
        imgs = np.stack([img0, img1])[..., None].astype(np.float32)
        fg = np.ones((1,) + self.SHAPE, np.float32)
        z = tf.zeros((2,) + self.SHAPE + (3,))
        u_inv = np.zeros((2,) + self.SHAPE + (3,), np.float32)
        u_inv[1, ..., 0] = 1.0
        _, aligned = losses.pair_ncc_loss(imgs, diffeo.pairwise_displacement(z, u_inv, [0], [1]), [0], [1],
                                          fg, fg, 5)
        _, ident = losses.pair_ncc_loss(imgs, diffeo.pairwise_displacement(z, z, [0], [1]), [0], [1],
                                        fg, fg, 5)
        self.assertGreater(float(aligned[0]), 0.999)
        self.assertLess(float(ident[0]), 0.99)

    def test_dtypes_and_window_validation(self):
        a = self.rng.uniform(size=(1,) + self.SHAPE + (1,))
        out = losses.local_ncc(tf.constant(a, tf.float16), tf.constant(a, tf.float16), 3)
        self.assertEqual(out.dtype, tf.float32)
        loss, ncc = losses.template_ncc_loss(tf.constant(np.repeat(a, 2, 0), tf.float16),
                                             np.ones((1,) + self.SHAPE, np.float16), 3)
        self.assertEqual((loss.dtype, ncc.dtype), (tf.float32, tf.float32))
        for bad in (4, 1, 11, 2.5):
            with self.assertRaises(ValueError, msg=str(bad)):
                losses.local_ncc(a, a, bad)
        with self.assertRaises(ValueError):
            losses.box_sum(tf.constant(a), 13)                      # larger than the volume (10)
        self.assertEqual(losses.check_ncc_window(5.0), 5)


class TestDeformMetrics(unittest.TestCase):
    def test_hard_dice_hand_made(self):
        a = np.array([0, 1, 1, 2, 2, 0])
        b = np.array([0, 1, 2, 2, 2, 0])
        d = M.hard_dice_per_label(a, b, 4)
        self.assertEqual(d.shape, (3,))
        np.testing.assert_allclose(d[:2], [2.0 / 3.0, 0.8])
        self.assertTrue(np.isnan(d[2]))                      # label 3 absent in both
        self.assertAlmostEqual(M.mean_fg_dice(a, b, 4), (2.0 / 3.0 + 0.8) / 2.0)
        self.assertEqual(M.mean_fg_dice(a, a, 4), 1.0)
        self.assertEqual(M.mean_fg_dice(np.array([1, 1, 0, 0]), np.array([0, 0, 2, 2]), 3), 0.0)
        self.assertEqual(M.mean_fg_dice(np.zeros(5, int), np.zeros(5, int), 3), 1.0)
        with self.assertRaises(ValueError):
            M.hard_dice_per_label(a, b[:3], 4)

    def test_deform_group_errors(self):
        rng = np.random.default_rng(0)
        shape = (6, 7, 5)
        seg = rng.integers(0, 3, size=(3,) + shape)
        jac = np.ones((3, 4, 5, 3))
        jac[1, 0, 0, :2] = -0.5                              # 2 folded voxels of 60
        err = np.zeros((3,) + shape + (3,))
        err[2, ..., 0] = 0.25                                # 0.25 vox everywhere in image 2
        err[2, 0, 0, 0] = [0.0, 3.0, 4.0]                    # max 5 vox
        v = rng.normal(size=(3,) + shape + (3,))
        v_c = v - v.mean(axis=0, keepdims=True)
        e = M.deform_group_errors(seg, seg[0], seg[:2], seg[1:], jac, err, v_c, v, 4.0, 3)
        np.testing.assert_allclose(e["template_dice"][0], 1.0)
        self.assertEqual(e["template_dice"].shape, (3,))
        for p in range(2):
            self.assertAlmostEqual(e["pair_dice"][p], M.mean_fg_dice(seg[p], seg[p + 1], 3))
        np.testing.assert_allclose(e["neg_jac_pct"], [0.0, 100.0 * 2 / 60, 0.0])
        n_vox = float(np.prod(shape))
        np.testing.assert_allclose(e["inv_consistency_mm"],
                                   [0.0, 0.0, 4.0 * (0.25 * (n_vox - 1) + 5.0) / n_vox])
        np.testing.assert_allclose(e["inv_consistency_max_mm"], [0.0, 0.0, 20.0])
        self.assertLess(float(e["centrality_mm"][0]), 1e-12)
        ref = 4.0 * np.mean(np.linalg.norm(v.mean(axis=0), axis=-1))
        np.testing.assert_allclose(e["raw_centrality_mm"], [ref])
        m = M.summarize(e, "val_deform")
        for k in ("template_dice", "pair_dice", "neg_jac_pct", "inv_consistency_mm",
                  "inv_consistency_max_mm", "centrality_mm", "raw_centrality_mm"):
            self.assertIn("val_deform/" + k, m)
        self.assertAlmostEqual(M.deform_pair_dice_loss(m, "val_deform"), 1.0 - m["val_deform/pair_dice"])
        self.assertIsNone(M.deform_pair_dice_loss({}, "val_deform"))
        # Norm input [N, X, Y, Z] is accepted too.
        e2 = M.deform_group_errors(seg, seg[0], seg[:2], seg[1:], jac, np.linalg.norm(err, axis=-1),
                                   v_c, v, 4.0, 3)
        np.testing.assert_allclose(e2["inv_consistency_mm"], e["inv_consistency_mm"])


if __name__ == "__main__":
    unittest.main()
