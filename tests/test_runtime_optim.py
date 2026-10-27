"""LR schedules (runtime.schedulers) and optimizers (runtime.optimizers).

TestAdamW: retrain-prereqs design 2 / test 11.1 a-e (AdamW at wd 0 == Adam, decay targets and
magnitude, AMP, checkpoint round trip); 11.1 f is in tests/test_main.py.
"""
import os
import shutil
import tempfile
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests.runtime_helpers import free_memory, narrow_config
import numpy as np
import tensorflow as tf

from models.group_net import GroupRegLevelModel
from runtime import optimizers as O
from runtime.schedulers import ScheduleConfig, lr_at


class TestSchedules(unittest.TestCase):
    def test_warmup_and_endpoints(self):
        for name in ("cosine", "poly"):
            cfg = ScheduleConfig(name=name, base_lr=0.01, total_steps=5000, warmup_steps=1000,
                                 min_lr_factor=1e-3)
            self.assertLess(lr_at(0, cfg), 1e-9)
            self.assertAlmostEqual(lr_at(500, cfg), 0.005, delta=1e-9)
            self.assertAlmostEqual(lr_at(1000, cfg), 0.01, places=12)
            self.assertAlmostEqual(lr_at(5000, cfg), 1e-5, places=12)
            self.assertAlmostEqual(lr_at(9000, cfg), 1e-5, places=12)
            post = [lr_at(s, cfg) for s in range(1000, 5001, 50)]
            self.assertTrue(all(b <= a + 1e-15 for a, b in zip(post, post[1:])), name)
            warm = [lr_at(s, cfg) for s in range(0, 1001, 50)]
            self.assertTrue(all(b > a for a, b in zip(warm, warm[1:])))

    def test_cosine_midpoint(self):
        cfg = ScheduleConfig(name="cosine", base_lr=1.0, total_steps=200, warmup_steps=0, min_lr_factor=0.0)
        self.assertAlmostEqual(lr_at(100, cfg), 0.5, places=12)

    def test_cycle_restarts(self):
        cfg = ScheduleConfig(name="cycle", base_lr=0.01, total_steps=10000, warmup_steps=0,
                             min_lr_factor=1e-3, cycle_steps=1000, cycle_mult=1.5, cycle_decay=10.0)
        self.assertAlmostEqual(lr_at(0, cfg), 0.01, places=12)
        self.assertLess(lr_at(999, cfg), 1e-4)
        self.assertAlmostEqual(lr_at(1000, cfg), 0.001, places=12)       # restart, peak / 10
        self.assertAlmostEqual(lr_at(2500, cfg), 0.0001, places=12)      # 2nd cycle lasts 1500
        self.assertAlmostEqual(lr_at(1750, cfg), 0.5 * (0.001 + 1e-6), places=12)

    def test_constant(self):
        cfg = ScheduleConfig(name="constant", base_lr=3e-4, total_steps=100, warmup_steps=0)
        self.assertEqual({lr_at(s, cfg) for s in (0, 50, 100, 1000)}, {3e-4})

    def test_bad_name(self):
        with self.assertRaises(ValueError):
            ScheduleConfig(name="step")


class TestOptimizers(unittest.TestCase):
    def tearDown(self):
        free_memory()

    def model(self):
        return GroupRegLevelModel(4, narrow_config()).build_model(2)

    def test_weight_decay_exclusion(self):
        model = self.model()
        opt = O.make_optimizer("sgdw")
        O.build_optimizer(opt, model.trainable_variables)
        excluded = [v for v in model.trainable_variables if not O.uses_weight_decay(opt, v)]
        included = [v for v in model.trainable_variables if O.uses_weight_decay(opt, v)]
        leaf = lambda v: v.name.split("/")[-1].split(":")[0]  # noqa: E731
        for v in excluded:
            self.assertIn(leaf(v), ("gamma", "beta", "bias"), v.name)
        for v in included:
            self.assertEqual(leaf(v), "kernel", v.name)
        n_kernel = sum(leaf(v) == "kernel" for v in model.trainable_variables)
        self.assertEqual(len(included), n_kernel)
        self.assertEqual(len(excluded) + len(included), len(model.trainable_variables))
        self.assertGreater(len(excluded), 0)
        no_ex = O.make_optimizer("sgdw", exclude_wd=False)
        self.assertTrue(all(O.uses_weight_decay(no_ex, v) for v in model.trainable_variables))
        self.assertFalse(any(O.uses_weight_decay(O.make_optimizer("adam"), v) for v in model.trainable_variables))

    def test_decoupled_decay_functional(self):
        model = self.model()
        opt = O.make_optimizer("sgdw", lr=1.0, weight_decay=0.1, momentum=0.0, decoupled=True, clipnorm=None)
        tv = model.trainable_variables
        O.build_optimizer(opt, tv)
        before = [v.numpy().copy() for v in tv]
        opt.apply_gradients(zip([tf.zeros_like(v) for v in tv], tv))
        for v, b in zip(tv, before):
            if O.uses_weight_decay(opt, v):
                np.testing.assert_allclose(v.numpy(), 0.9 * b, rtol=1e-6, err_msg=v.name)
            else:
                np.testing.assert_array_equal(v.numpy(), b, err_msg=v.name)

    def test_sgdw_matches_reference_update(self):
        v = tf.Variable([1.0, -2.0], name="w/kernel")
        opt = O.SGDW(learning_rate=0.1, weight_decay=0.01, momentum=0.9, dampening=0.0, nesterov=True)
        opt.build([v])
        m = np.zeros(2)
        w = np.array([1.0, -2.0])
        for g in ([0.5, 0.5], [-1.0, 2.0]):
            opt.apply_gradients([(tf.constant(g), v)])
            b = np.asarray(g) + 0.01 * w
            m = 0.9 * m + b
            w = w - 0.1 * (b + 0.9 * m)
        np.testing.assert_allclose(v.numpy(), w, rtol=1e-6)

    def test_every_optimizer_steps_and_lr_helpers(self):
        for name in O.OPTIMIZERS:
            for amp in (False, True):
                v = tf.Variable([1.0, 2.0], name="x/kernel")
                opt = O.make_optimizer(name, lr=0.1, amp=amp)
                O.build_optimizer(opt, [v])
                opt.apply_gradients([(tf.constant([1.0, 1.0]), v)])
                self.assertFalse(np.allclose(v.numpy(), [1.0, 2.0]), (name, amp))
                O.set_learning_rate(opt, 0.0123)
                self.assertAlmostEqual(O.get_learning_rate(opt), 0.0123, places=7)
                self.assertEqual(int(O.inner_optimizer(opt).iterations.numpy()), 1)

    def test_weight_decay_optimizers_constant(self):
        self.assertEqual(O.WEIGHT_DECAY_OPTIMIZERS, ("sgdw", "adamw"))
        self.assertTrue(set(O.WEIGHT_DECAY_OPTIMIZERS) <= set(O.OPTIMIZERS))

    def test_default_clipnorm(self):
        opt = O.make_optimizer()
        self.assertEqual(opt.global_clipnorm, 12.0)
        self.assertIsInstance(opt, O.SGDW)
        self.assertEqual((opt.momentum, opt.nesterov, opt.weight_decay), (0.99, True, 3e-5))
        self.assertIsNone(O.make_optimizer(clipnorm=0).global_clipnorm)


def _leaf(v):
    return v.name.split("/")[-1].split(":")[0]


def _opt_state(opt):
    """Numpy copies of the (inner) optimizer's slot variables (every m / v; Keras 2.10
    ``variables()`` does not list ``iterations``, compared separately)."""
    return [np.array(s.numpy()) for s in O.inner_optimizer(opt).variables()]


class TestAdamW(unittest.TestCase):
    """Retrain-prereqs design 2 / test 11.1 a-e: --optimizer adamw = Keras 2.10 experimental AdamW."""

    LRS = (3e-4, 1e-3, 5e-4, 2e-4, 1e-4)

    def tearDown(self):
        free_memory()

    def model(self):
        return GroupRegLevelModel(4, narrow_config()).build_model(2)

    @staticmethod
    def grads(variables, steps, seed=0):
        rng = np.random.default_rng(seed)
        return [[rng.standard_normal(v.shape).astype(np.float32) for v in variables] for _ in range(steps)]

    @staticmethod
    def run_steps(opt, variables, grads, lrs):
        for g, lr in zip(grads, lrs):
            O.set_learning_rate(opt, lr)
            opt.apply_gradients(zip([tf.constant(x) for x in g], variables))

    def test_a_wd0_equals_adam(self):
        rng = np.random.default_rng(1)
        init = [rng.standard_normal(s).astype(np.float32) for s in ((3, 4), (4,), (2, 2, 5), (5,))]
        names = ("a/kernel", "a/bias", "b/kernel", "b/gamma")
        vs = {}
        for name in ("adam", "adamw"):
            vs[name] = [tf.Variable(x, name=n) for x, n in zip(init, names)]
        opts = {"adam": O.make_optimizer("adam", lr=3e-4),
                "adamw": O.make_optimizer("adamw", lr=3e-4, weight_decay=0.0)}
        self.assertIsInstance(O.inner_optimizer(opts["adamw"]), tf.keras.optimizers.experimental.AdamW)
        grads = self.grads(vs["adam"], 5, seed=2)
        for name in opts:
            O.build_optimizer(opts[name], vs[name])
            self.run_steps(opts[name], vs[name], grads, self.LRS)
        for a, w in zip(vs["adam"], vs["adamw"]):
            np.testing.assert_array_equal(a.numpy(), w.numpy(), err_msg=a.name)
        sa, sw = _opt_state(opts["adam"]), _opt_state(opts["adamw"])
        self.assertEqual(len(sa), len(sw))
        self.assertEqual(len(sa), 2 * len(init))              # m, v per variable
        for x, y in zip(sa, sw):
            np.testing.assert_array_equal(x, y)
        for name in opts:
            self.assertEqual(int(O.inner_optimizer(opts[name]).iterations.numpy()), 5)
        self.assertFalse(np.array_equal(vs["adam"][0].numpy(), init[0]))   # the steps did move

    def test_b_decay_only_on_kernels(self):
        model = self.model()
        tv = model.trainable_variables
        opt = O.make_optimizer("adamw", weight_decay=1e-2)
        O.build_optimizer(opt, tv)
        excluded = [v for v in tv if not O.uses_weight_decay(opt, v)]
        for v in tv:
            ex = any(k in _leaf(v) for k in O.DEFAULT_WD_EXCLUDE)
            self.assertEqual(O.uses_weight_decay(opt, v), not ex, v.name)
            if _leaf(v) == "kernel":
                self.assertTrue(O.uses_weight_decay(opt, v), v.name)
        self.assertGreater(len(excluded), 0)
        self.assertGreater(len(tv) - len(excluded), 0)
        no_ex = O.make_optimizer("adamw", weight_decay=1e-2, exclude_wd=False)
        self.assertTrue(all(O.uses_weight_decay(no_ex, v) for v in tv))

    def test_c_decay_magnitude(self):
        model = self.model()
        tv = model.trainable_variables
        opt = O.make_optimizer("adamw", lr=0.1, weight_decay=0.01, clipnorm=None)
        O.build_optimizer(opt, tv)
        before = [v.numpy().copy() for v in tv]
        opt.apply_gradients(zip([tf.zeros_like(v) for v in tv], tv))
        n_dec = 0
        for v, b in zip(tv, before):
            if O.uses_weight_decay(opt, v):
                n_dec += 1
                np.testing.assert_allclose(v.numpy(), (1.0 - 1e-3) * b, rtol=1e-6, err_msg=v.name)
            else:
                np.testing.assert_array_equal(v.numpy(), b, err_msg=v.name)
        self.assertGreater(n_dec, 0)

    def test_d_amp_loss_scale_optimizer(self):
        opt = O.make_optimizer("adamw", lr=0.1, weight_decay=0.01, clipnorm=None, amp=True)
        self.assertIsInstance(opt, tf.keras.mixed_precision.LossScaleOptimizer)
        self.assertIsInstance(O.inner_optimizer(opt), tf.keras.optimizers.experimental.AdamW)
        k = tf.Variable([1.0, -2.0, 3.0], name="amp/kernel")
        b = tf.Variable([0.5, 0.25], name="amp/bias")
        O.build_optimizer(opt, [k, b])
        opt.apply_gradients([(tf.zeros(3), k), (tf.zeros(2), b)])          # finite: decays
        np.testing.assert_allclose(k.numpy(), (1.0 - 1e-3) * np.array([1.0, -2.0, 3.0]), rtol=1e-6)
        np.testing.assert_array_equal(b.numpy(), [0.5, 0.25])
        state = _opt_state(opt)
        kb, bb = k.numpy().copy(), b.numpy().copy()
        opt.apply_gradients([(tf.constant([np.inf, 0.0, 0.0]), k), (tf.zeros(2), b)])   # skipped
        np.testing.assert_array_equal(k.numpy(), kb)
        np.testing.assert_array_equal(b.numpy(), bb)
        for x, y in zip(state, _opt_state(opt)):
            np.testing.assert_array_equal(x, y)
        # Keras' LSO still counts the skipped step in `iterations` (its do-not-apply branch);
        # the Trainer's own tf.cond skip around apply_gradients never reaches the LSO.
        self.assertEqual(int(O.inner_optimizer(opt).iterations.numpy()), 2)

    def test_e_checkpoint_round_trip(self):
        tmp = tempfile.mkdtemp(prefix="bgr_adamw_ckpt_")
        try:
            make = lambda: O.make_optimizer("adamw", lr=3e-4, weight_decay=1e-2)  # noqa: E731
            ref = self.model()
            init = ref.get_weights()
            grads = self.grads(ref.trainable_variables, 5, seed=3)
            opt_ref = O.build_optimizer(make(), ref.trainable_variables)
            self.run_steps(opt_ref, ref.trainable_variables, grads, self.LRS)
            first = self.model()
            first.set_weights(init)
            opt1 = O.build_optimizer(make(), first.trainable_variables)
            self.run_steps(opt1, first.trainable_variables, grads[:3], self.LRS[:3])
            path = tf.train.Checkpoint(model=first, optimizer=opt1).save(os.path.join(tmp, "ck"))
            second = self.model()
            opt2 = O.build_optimizer(make(), second.trainable_variables)
            tf.train.Checkpoint(model=second, optimizer=opt2).restore(path).assert_existing_objects_matched()
            self.assertEqual(int(O.inner_optimizer(opt2).iterations.numpy()), 3)
            self.run_steps(opt2, second.trainable_variables, grads[3:], self.LRS[3:])
            for a, c in zip(ref.weights, second.weights):
                np.testing.assert_array_equal(a.numpy(), c.numpy(), err_msg=a.name)
            self.assertEqual(int(O.inner_optimizer(opt2).iterations.numpy()), 5)
            sr, s2 = _opt_state(opt_ref), _opt_state(opt2)
            self.assertEqual(len(sr), len(s2))
            for x, y in zip(sr, s2):
                np.testing.assert_array_equal(x, y)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
