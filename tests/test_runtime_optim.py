"""LR schedules (runtime.schedulers) and optimizers (runtime.optimizers)."""
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

    def test_default_clipnorm(self):
        opt = O.make_optimizer()
        self.assertEqual(opt.global_clipnorm, 12.0)
        self.assertIsInstance(opt, O.SGDW)
        self.assertEqual((opt.momentum, opt.nesterov, opt.weight_decay), (0.99, True, 3e-5))
        self.assertIsNone(O.make_optimizer(clipnorm=0).global_clipnorm)


if __name__ == "__main__":
    unittest.main()
