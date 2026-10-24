"""Lean deform losses (`runtime.losses`, ``lean_losses`` / ``lean_sampler``; milestone 9 design
3.4): Dice without materialised slices or broadcast copies, LNCC under tf.recompute_grad, and the
warps with the lean sampler. Against the default path (fp32, level 4, N 4, synthetic phantom):
every loss term within 1e-6 relative and the gradients w.r.t. the SVF within 1e-4 relative, for
lean_losses on / off combined with lean_sampler on / off; with both off the terms are bitwise
those of the call without the new arguments.
"""
import sys
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests.field_helpers import smooth_field
from tests.runtime_helpers import free_memory
from tests.test_trainer import _OpDeterminism
import numpy as np
import tensorflow as tf

from models import diffeo as D
from runtime import fixed_data as fd
from runtime import losses as LS

N = 4
N_CLASSES = 9


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


class TestLeanLosses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        grp = fd.synthetic_deform_group(4, N, seed=0)
        cls.images = tf.constant(grp["images"])
        cls.onehot = tf.one_hot(tf.constant(grp["labels"][..., 0], tf.int32), N_CLASSES, dtype=tf.float32)
        shape = tuple(grp["images"].shape[1:4])
        cls.v = tf.Variable(smooth_field(shape, np.random.default_rng(1), n=N, sigma=3.0, max_vox=2.5)
                            .astype(np.float32))
        cls.batch = {"onehot": cls.onehot, "images": cls.images,
                     "pair_i": tf.constant([0, 2, 3], tf.int32), "pair_j": tf.constant([1, 3, 0], tf.int32)}

    @classmethod
    def tearDownClass(cls):
        free_memory()

    def run_loss(self, ncc=True, **kw):
        lam = dict(lambda_ncc_tmpl=1.0, lambda_ncc_pair=1.0) if ncc else {}
        with tf.GradientTape() as tape:
            u, u_inv = D.exp_and_inverse(self.v, 7)
            terms = LS.deform_loss({"u": u, "u_inv": u_inv}, self.batch, ncc_window=5, **dict(lam, **kw))
        g = tape.gradient(terms["loss"], self.v).numpy()
        return {k: float(t.numpy()) for k, t in terms.items()}, g

    def test_against_default(self):
        with _OpDeterminism():          # GPU segment-sum gradients are otherwise nondeterministic
            ref, g_ref = self.run_loss()
            plain, g_plain = self.run_loss(lean_sampler=False, lean_losses=False)
        self.assertEqual(plain, ref)                                   # bitwise: same ops
        self.assertTrue(np.array_equal(g_plain, g_ref))
        worst_t, worst_g = 0.0, 0.0
        for ls in (False, True):
            for ll in (False, True):
                if not (ls or ll):
                    continue
                got, g = self.run_loss(lean_sampler=ls, lean_losses=ll)
                self.assertEqual(sorted(got), sorted(ref))
                for k in ref:
                    e = abs(got[k] - ref[k]) / max(abs(ref[k]), 1e-12)
                    worst_t = max(worst_t, e)
                    self.assertLessEqual(e, 1e-6, (ls, ll, k, got[k], ref[k]))
                e = rel(g, g_ref)
                worst_g = max(worst_g, e)
                self.assertLessEqual(e, 1e-4, (ls, ll))
        print("\n[lean losses] worst rel. error: terms %.2e, gradient %.2e" % (worst_t, worst_g), file=sys.stderr)

    def test_dice_only(self):
        ref, g_ref = self.run_loss(ncc=False)
        got, g = self.run_loss(ncc=False, lean_sampler=True, lean_losses=True)
        for k in ref:
            self.assertLessEqual(abs(got[k] - ref[k]) / max(abs(ref[k]), 1e-12), 1e-6, k)
        self.assertLessEqual(rel(g, g_ref), 1e-4)

    def test_soft_dice_lean(self):
        rng = np.random.default_rng(2)
        a = tf.constant(rng.uniform(size=(3, 6, 5, 4, 5)).astype(np.float32))
        b = tf.constant(rng.uniform(size=(1, 6, 5, 4, 5)).astype(np.float32))
        b = tf.concat([b[..., :2], tf.zeros_like(b[..., 2:3]), b[..., 3:]], axis=-1)   # an absent class
        a = tf.concat([a[..., :2], tf.zeros_like(a[..., 2:3]), a[..., 3:]], axis=-1)
        want = LS.soft_dice(a, tf.broadcast_to(b, tf.shape(a))).numpy()
        got = LS._soft_dice_lean(a, b).numpy()
        np.testing.assert_allclose(got, want, rtol=1e-6)
        np.testing.assert_allclose(LS._soft_dice_lean(a, b, exclude_background=False).numpy(),
                                   LS.soft_dice(a, tf.broadcast_to(b, tf.shape(a)), exclude_background=False).numpy(),
                                   rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
