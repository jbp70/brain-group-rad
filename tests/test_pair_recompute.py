"""--pair-recompute (tf-intensity-aug addendum): `losses.pair_terms_recompute` gives the same
deform loss and gradients as the default pairwise path (plain, lean sampler, lean losses; eager
and tf.function), the static pair count matches `postprocess.num_pairs`, the Trainer step matches
the default step (also with --gradient-checkpointing full + lean sampler), and on a GPU the
gradient peak with 4 pairs stays close to the 1-pair peak (default path: grows per pair)."""
import sys
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory setup)
from tests.runtime_helpers import free_memory, narrow_config, synthetic_group
import numpy as np
import tensorflow as tf

from data_loading import params as P
from data_loading import postprocess as pp
from models import diffeo
from models.group_net import GroupRegLevelModel
from runtime import fixed_data as fd
from runtime import losses
from runtime import optimizers as O
from runtime.run import Trainer, TrainConfig
from runtime.schedulers import ScheduleConfig


def smooth_field(rng, shape, n, amp=2.0):
    """[n, X, Y, Z, 3] smooth random displacements (voxels)."""
    g = np.meshgrid(*[np.linspace(0, 2 * np.pi, s) for s in shape], indexing="ij")
    out = np.zeros((n,) + tuple(shape) + (3,), np.float32)
    for i in range(n):
        for c in range(3):
            ph = rng.uniform(0, 6, 3)
            out[i, ..., c] = amp * np.sin(g[0] + ph[0]) * np.cos(g[1] + ph[1]) * np.sin(0.5 * g[2] + ph[2])
    return out


def make_inputs(shape=(16, 20, 12), n=4, n_classes=5, seed=0):
    rng = np.random.default_rng(seed)
    u = smooth_field(rng, shape, n)
    u_inv = -u                                       # not an exact inverse; fine for value checks
    lab = rng.integers(0, n_classes, size=(n,) + tuple(shape))
    onehot = np.eye(n_classes, dtype=np.float32)[lab]
    images = rng.uniform(size=(n,) + tuple(shape) + (1,)).astype(np.float32)
    return u, u_inv, onehot, images


def loss_and_grads(u, u_inv, batch, n_pairs, recompute, fn_mode=False, **kw):
    uv, uiv = tf.Variable(u), tf.Variable(u_inv)

    def run():
        with tf.GradientTape() as tape:
            terms = losses.deform_loss({"u": uv, "u_inv": uiv}, batch, lambda_ncc_tmpl=1.0, lambda_ncc_pair=1.0,
                                       ncc_window=3, pair_recompute=recompute,
                                       n_pairs=n_pairs if recompute else None, **kw)
        g = [tf.convert_to_tensor(x) for x in tape.gradient(terms["loss"], [uv, uiv])]   # dense (gathers)
        return terms, g

    terms, g = tf.function(run)() if fn_mode else run()
    return {k: float(v.numpy()) for k, v in terms.items()}, [x.numpy() for x in g]


class TestPairRecomputeLoss(unittest.TestCase):

    def setUp(self):
        self.u, self.u_inv, oh, img = make_inputs()
        self.pi = np.array([0, 1, 2, 3], np.int32)
        self.pj = np.array([1, 2, 3, 0], np.int32)
        self.batch = {"onehot": tf.constant(oh), "images": tf.constant(img),
                      "pair_i": tf.constant(self.pi), "pair_j": tf.constant(self.pj)}

    def compare(self, fn_mode=False, **kw):
        a, ga = loss_and_grads(self.u, self.u_inv, self.batch, 4, False, fn_mode, **kw)
        b, gb = loss_and_grads(self.u, self.u_inv, self.batch, 4, True, fn_mode, **kw)
        self.assertEqual(sorted(a), sorted(b))
        for k in a:
            self.assertAlmostEqual(a[k], b[k], places=5, msg=k)
        worst = 0.0
        for x, y in zip(ga, gb):
            scale = max(float(np.abs(x).max()), 1e-12)
            worst = max(worst, float(np.abs(x - y).max()) / scale)
            np.testing.assert_allclose(y, x, atol=1e-5 * scale, rtol=1e-4)
        return worst

    def test_values_and_gradients_match(self):
        worst = max(self.compare(), self.compare(fn_mode=True),
                    self.compare(lean_sampler=True), self.compare(lean_losses=True),
                    self.compare(fn_mode=True, lean_sampler=True, lean_losses=True))
        sys.stderr.write("\n[pair recompute] max relative |grad diff| %.2e\n" % worst)

    def test_without_pair_ncc_and_static_count(self):
        a, ga = loss_and_grads(self.u, self.u_inv, dict(self.batch), 4, False)
        terms = losses.deform_loss({"u": tf.constant(self.u), "u_inv": tf.constant(self.u_inv)}, self.batch,
                                   pair_recompute=True)            # n_pairs from the static shape
        ref = losses.deform_loss({"u": tf.constant(self.u), "u_inv": tf.constant(self.u_inv)}, self.batch)
        for k in ref:
            self.assertAlmostEqual(float(ref[k]), float(terms[k]), places=5, msg=k)
        with self.assertRaises(ValueError):
            losses.pair_terms_recompute(self.batch["onehot"], self.u, self.u_inv, self.pi, self.pj, 0)
        self.assertTrue(np.isfinite(a["loss"]))

    def test_graph_has_recompute_scope_only_when_on(self):
        def names(on):
            fn = tf.function(lambda u, ui: losses.deform_loss({"u": u, "u_inv": ui}, self.batch, pair_recompute=on,
                                                              n_pairs=4 if on else None)["loss"])
            g = fn.get_concrete_function(tf.TensorSpec(self.u.shape), tf.TensorSpec(self.u.shape)).graph
            return [op.name for op in g.get_operations()]
        self.assertFalse([s for s in names(False) if "pair_recompute" in s])
        self.assertTrue([s for s in names(True) if "pair_recompute" in s])

    def test_num_pairs_static(self):
        for n in range(2, 34):
            for p in (0.1, 0.05, 0.25, 0.5, 1.0):
                for mp in pp.MIN_PAIRS:
                    self.assertEqual(pp.num_pairs_static(n, p, mp), int(pp.num_pairs(n, p, mp)), (n, p, mp))


class TestPairRecomputeTrainer(unittest.TestCase):
    """One deform step with and without --pair-recompute from the same init: same loss, same
    updated weights (min_pairs 'n', LNCC on)."""

    @classmethod
    def setUpClass(cls):
        cls.dgroup = fd.synthetic_deform_group(4, 4, seed=0)
        cls.S = synthetic_group(4, 4, "rigid", seed=0)["S"]

    def tearDown(self):
        free_memory()

    def step(self, recompute, ckpt="none", lean=False):
        cfg = TrainConfig(level=4, group_size=4, steps=10, val_every=0, log_every=1, task_ratio={"deform": 1.0},
                          schedule=ScheduleConfig("constant", base_lr=1e-3, total_steps=10, warmup_steps=0), seed=0,
                          min_pairs="n", lambda_ncc_tmpl=1.0, lambda_ncc_pair=1.0, pair_recompute=recompute,
                          lean_sampler=lean)
        model = GroupRegLevelModel(4, narrow_config(dropout=0.0, seed=1, deform_decoder=True),
                                   gradient_checkpointing=ckpt, lean_sampler=lean).build_model(4)
        tr = Trainer(model, O.make_optimizer("adam", lr=1e-3), {"deform": fd.FixedGroupSource(self.dgroup["batch"])},
                     self.S, cfg)
        res = [tr.train_step("deform", self.dgroup["batch"]) for _ in range(2)]   # step 1 moves the zero head
        return res, [v.numpy() for v in model.trainable_variables]

    def test_step_matches_default(self):
        for ckpt, lean in (("none", False), ("full", True)):
            ra, wa = self.step(False, ckpt, lean)
            free_memory()
            rb, wb = self.step(True, ckpt, lean)
            free_memory()
            for x, y in zip(ra, rb):
                self.assertTrue(bool(x["finite"]) and bool(y["finite"]))
                self.assertAlmostEqual(float(x["loss"]), float(y["loss"]), places=5, msg=ckpt)
            for x, y in zip(wa, wb):
                # Two Adam steps (lr 1e-3): summation-order noise of near-zero gradients is
                # amplified by the normalisation, so 1e-4 = 10 % of one step.
                np.testing.assert_allclose(y, x, atol=1e-4, err_msg=ckpt)


@unittest.skipUnless(tf.config.list_physical_devices("GPU"), "needs a GPU")
class TestPairRecomputeMemory(unittest.TestCase):
    """Gradient peak at 64x80x64 (19 classes, N 4) with 1 vs 4 pairs, pair LNCC on (template
    LNCC off to stay inside the 6 GB local test limit)."""

    def peak(self, n_pairs, recompute, lean=False):
        shape = (64, 80, 64)
        u, u_inv, oh, img = make_inputs(shape, 4, P.N_TRAIN_LABELS, seed=1)
        pi = np.arange(n_pairs, dtype=np.int32) % 4
        pj = (pi + 1) % 4
        with tf.device("GPU:0"):
            batch = {"onehot": tf.constant(oh), "images": tf.constant(img), "pair_i": tf.constant(pi),
                     "pair_j": tf.constant(pj)}
            uv, uiv = tf.Variable(u), tf.Variable(u_inv)

            @tf.function
            def run():
                with tf.GradientTape() as tape:
                    t = losses.deform_loss({"u": uv, "u_inv": uiv}, batch, lambda_ncc_pair=1.0,
                                           ncc_window=5, pair_recompute=recompute,
                                           n_pairs=n_pairs if recompute else None, lean_sampler=lean)
                g = [tf.convert_to_tensor(x) for x in tape.gradient(t["loss"], [uv, uiv])]
                return tf.add_n([tf.reduce_sum(x) for x in g])

            run().numpy()
            base = tf.config.experimental.get_memory_info("GPU:0")["current"]
            tf.config.experimental.reset_memory_stats("GPU:0")
            run().numpy()
            peak = tf.config.experimental.get_memory_info("GPU:0")["peak"] - base
        del batch, uv, uiv
        free_memory()
        return peak / 2.0 ** 20

    def test_peak_1_vs_4_pairs(self):
        for lean in (False, True):
            p = {(k, rc): self.peak(k, rc, lean) for k in (1, 4) for rc in (False, True)}
            sys.stderr.write("\n[pair recompute memory, 64x80x64 N 4, %s sampler] default 1 / 4 pairs: %.0f / %.0f "
                             "MiB; recompute 1 / 4 pairs: %.0f / %.0f MiB\n"
                             % ("lean" if lean else "default", p[(1, False)], p[(4, False)], p[(1, True)],
                                p[(4, True)]))
            grow_default = p[(4, False)] - p[(1, False)]
            grow_rc = p[(4, True)] - p[(1, True)]
            self.assertGreater(grow_default, 0.0)
            self.assertLess(grow_rc, 0.35 * grow_default)
            self.assertLess(p[(4, True)], p[(4, False)])


if __name__ == "__main__":
    unittest.main()
