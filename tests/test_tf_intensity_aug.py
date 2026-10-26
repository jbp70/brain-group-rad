"""Tests for data_loading.tf_intensity_aug (TF intensity augmentation, --intensity-aug-device tf)
against the numpy reference data_loading.intensity_ref (T1-T14 of
.agents/tasks/tf-intensity-aug/plan.md). DALI-vs-TF comparisons: tests/gpu/verify_dali_pipeline.py
check m (cluster)."""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

from tests import tf_setup  # noqa: F401  (log level, GPU memory growth)
import tensorflow as tf
from scipy import stats

from data_loading import intensity_ref as ref
from data_loading import params as P
from data_loading import tf_intensity_aug as tfa

C = P.AUG_COLUMNS
OPS = ("noise", "brightness", "contrast", "low_res", "gamma")


def only(op, prob=0.5):
    """Config with only `op` enabled (probability `prob`; urow forces it with <op>_apply=0)."""
    kw = {o: (o == op) for o in OPS}
    kw[op + "_prob"] = prob
    return P.IntensityAugConfig(**kw)


def urow(**cols):
    """A row of uniforms: 0.999 everywhere (nothing applied), overridden by column name."""
    u = np.full([P.N_AUG_UNIFORMS], 0.999, np.float32)
    for k, v in cols.items():
        u[C[k]] = v
    return u


def phantom(shape, seed=0, lo=0.0, hi=1.0):
    """Smooth phantom in [lo, hi]: sum of a few low-frequency cosines."""
    rng = np.random.default_rng(seed)
    g = np.meshgrid(*[np.linspace(0, 1, n) for n in shape], indexing="ij")
    x = sum(np.cos(2 * np.pi * (rng.uniform(0.5, 2) * g[0] + rng.uniform(0.5, 2) * g[1]
                                + rng.uniform(0.5, 2) * g[2]) + rng.uniform(0, 6)) for _ in range(3))
    x = (x - x.min()) / (x.max() - x.min())
    return (lo + (hi - lo) * x).astype(np.float32)[..., None]


def aug1(x, u, cfg, seed=(1, 2)):
    return tfa.augment_image(tf.constant(x), tf.constant(u), tf.constant(seed, tf.int64), cfg).numpy()


class TestOps(unittest.TestCase):
    """T1-T7: per-op TF vs numpy with forced parameters."""

    def setUp(self):
        self.x = phantom((12, 14, 16), seed=1, lo=0.1, hi=0.9)

    def test_t1_noise(self):
        cfg = only("noise")
        u = urow(noise_apply=0.0, noise_std=0.5)
        seed = np.array([5, 7], np.int64)
        out = aug1(self.x, u, cfg, seed)
        z = tf.random.stateless_normal(self.x.shape, seed, dtype=tf.float32).numpy()
        want = np.clip(self.x + 0.5 * cfg.noise_std_max * z, 0, 1)
        np.testing.assert_allclose(out, want, atol=1e-6)
        np.testing.assert_allclose(out, ref.augment_np(self.x, u, z, cfg), atol=1e-5)
        # Probability 0 -> exact identity (cond skip).
        np.testing.assert_array_equal(aug1(self.x, u, only("noise", 0.0), seed), self.x)

    def test_t2_brightness(self):
        cfg = only("brightness")
        for low, want in ((0.2, 0.75 + 0.3 * 0.25), (0.8, 1.0 + 0.4 * 0.25)):
            u = urow(brightness_apply=0.0, brightness_low=low, brightness_lo_val=0.3, brightness_hi_val=0.4)
            p = ref.factors_from_uniforms(u, cfg)
            self.assertAlmostEqual(p["brightness_factor"], want, places=6)
            np.testing.assert_allclose(aug1(self.x, u, cfg), np.clip(self.x * want, 0, 1), atol=1e-6)
        np.testing.assert_array_equal(aug1(self.x, urow(), cfg), self.x)

    def test_t3_contrast(self):
        cfg = only("contrast")
        for low in (0.2, 0.8):
            u = urow(contrast_apply=0.0, contrast_low=low, contrast_lo_val=0.3, contrast_hi_val=0.9)
            f = ref.factors_from_uniforms(u, cfg)["contrast_factor"]
            out = aug1(self.x, u, cfg)
            np.testing.assert_allclose(out, ref.contrast_np(self.x, f), atol=1e-6)
            # Clamped to the PRE-contrast range.
            self.assertGreaterEqual(out.min(), self.x.min() - 1e-7)
            self.assertLessEqual(out.max(), self.x.max() + 1e-7)
        np.testing.assert_array_equal(aug1(self.x, urow(), cfg), self.x)

    def test_t4_lowres_matches_two_stage_reference(self):
        rng = np.random.default_rng(4)
        worst = 0.0
        for shape in ((11, 16, 13), (40, 56, 48)):
            x = rng.uniform(size=shape + (1,)).astype(np.float32)
            for roi in tfa.LOWRES_ROIS:
                for f in (0.5, 0.55, 0.63, 0.8, 0.97, 1.0):
                    out = tfa.lowres_resample(x, f, roi).numpy()
                    want = ref.lowres_np(x, f, roi)
                    d = float(np.abs(out - want).max())
                    worst = max(worst, d)
                    self.assertLess(d, 1e-5, (shape, roi, f))
            np.testing.assert_array_equal(tfa.lowres_resample(x, 1.0).numpy(), x)
        sys.stderr.write("\n[T4] lowres TF vs two-stage numpy: max |diff| %.3g\n" % worst)
        fn = tf.function(lambda v, f: tfa.lowres_resample(v, f),
                         input_signature=[tf.TensorSpec([11, 16, 13, 1], tf.float32),
                                          tf.TensorSpec([], tf.float64)])
        cf = fn.get_concrete_function()
        self.assertEqual(cf.structured_outputs.shape.as_list(), [11, 16, 13, 1])
        x = rng.uniform(size=(11, 16, 13, 1)).astype(np.float32)
        np.testing.assert_allclose(fn(x, tf.constant(0.6, tf.float64)).numpy(), ref.lowres_np(x, 0.6),
                                   atol=1e-5)
        with self.assertRaises(ValueError):
            tfa.lowres_resample(x, 0.5, "bogus")

    def test_t5_lowres_effective_resolution(self):
        x = np.random.default_rng(5).normal(size=(32, 32, 32, 1)).astype(np.float32)

        def lag1(v, axis):
            v = v[..., 0] - v.mean()
            a = np.take(v, range(31), axis=axis)
            b = np.take(v, range(1, 32), axis=axis)
            return float((a * b).mean() / v.var())

        out = tfa.lowres_resample(x, 0.5).numpy()
        for axis in range(3):
            self.assertLess(abs(lag1(x, axis)), 0.05)
            self.assertGreaterEqual(lag1(out, axis), 0.4)
        for n, f in ((32, 0.5), (160, 0.63), (224, 0.5), (192, 0.77)):
            self.assertEqual(len(np.unique(ref.nn_source_index(n, f))), ref.lowres_size(n, f))
        np.testing.assert_array_equal(tfa.lowres_resample(x, 1.0).numpy(), x)

    def test_t6_gamma(self):
        cfg = only("gamma")
        for g_low in (0.2, 0.8):
            u = urow(gamma_apply=0.0, gamma_low=g_low, gamma_lo_val=0.4, gamma_hi_val=0.6)
            g = ref.factors_from_uniforms(u, cfg)["gamma_factor"]
            out = tfa.gamma(tf.constant(self.x), tf.constant(g, tf.float32)).numpy()
            np.testing.assert_allclose(out.mean(), self.x.mean(), rtol=1e-4)
            np.testing.assert_allclose(out.std(), self.x.std(), rtol=1e-4)
            np.testing.assert_allclose(out, ref.gamma_np(self.x, g), atol=1e-5)
            np.testing.assert_allclose(aug1(self.x, u, cfg), np.clip(ref.gamma_np(self.x, g), 0, 1), atol=1e-5)
        # Not applied: g = 1 and the renormalisation still runs (~identity).
        np.testing.assert_allclose(aug1(self.x, urow(), cfg), self.x, atol=1e-5)

    def test_t7_clamp_and_disabled(self):
        x = np.random.default_rng(7).uniform(-0.2, 1.3, size=(12, 14, 16, 1)).astype(np.float32)
        cfg = P.IntensityAugConfig()
        g = tf.random.Generator.from_seed(3)
        out = tfa.augment_group(tf.constant(np.stack([x] * 6)), cfg, g).numpy()
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)
        off = P.IntensityAugConfig.off()
        state = g.state.numpy().copy()
        np.testing.assert_array_equal(tfa.augment_group(tf.constant(x[None]), off, g).numpy(), x[None])
        np.testing.assert_array_equal(tfa.augment_group(tf.constant(x[None]), None, g).numpy(), x[None])
        np.testing.assert_array_equal(g.state.numpy(), state)   # disabled: no draw
        np.testing.assert_array_equal(aug1(x, urow(), off), x)


class TestChain(unittest.TestCase):
    """T8-T10: full chain vs the reference, per image and statistically."""

    def test_t8_group_matches_reference(self):
        cfg = P.IntensityAugConfig(noise_prob=0.5, brightness_prob=0.5, contrast_prob=0.5,
                                   low_res_prob=0.5, gamma_prob=0.5)
        x = np.stack([phantom((12, 14, 16), seed=s) for s in range(8)])
        g = tf.random.Generator.from_seed(11)
        u, seeds = tfa.draw_aug_params(tf.random.Generator.from_seed(11), 8)
        out = tfa.augment_group(tf.constant(x), cfg, g).numpy()
        applied = np.zeros(len(OPS))
        for i in range(8):
            z = tf.random.stateless_normal(x.shape[1:], seeds[i], dtype=tf.float32).numpy()
            np.testing.assert_allclose(out[i], ref.augment_np(x[i], u[i].numpy(), z, cfg), atol=1e-5)
            p = ref.factors_from_uniforms(u[i].numpy(), cfg)
            applied += [p[o] for o in OPS]
        self.assertTrue(np.all(applied > 0), applied)       # every op exercised at least once

    def test_t9_parameter_statistics(self):
        cfg = P.IntensityAugConfig()
        n = 20000
        u, seeds = tfa.draw_aug_params(tf.random.Generator.from_seed(9), n)
        u = u.numpy()
        self.assertEqual(u.shape, (n, P.N_AUG_UNIFORMS))
        self.assertEqual(seeds.shape, (n, 2))
        self.assertEqual(len(np.unique(seeds.numpy()[:, 0])), n)
        rows = [ref.factors_from_uniforms(r, cfg) for r in u]
        for op in OPS:
            p = getattr(cfg, op + "_prob")
            freq = np.mean([r[op] for r in rows])
            self.assertLess(abs(freq - p), 4 * np.sqrt(p * (1 - p) / n), (op, freq))
        for op, (lo, hi) in (("brightness", cfg.brightness_range), ("contrast", cfg.contrast_range),
                             ("gamma", cfg.gamma_range)):
            fs = np.array([r[op + "_factor"] for r in rows if r[op]])
            self.assertTrue(np.all((fs >= lo) & (fs <= hi)), op)
            low = np.mean(fs < 1.0)
            self.assertLess(abs(low - 0.5), 4 * np.sqrt(0.25 / len(fs)), (op, low))
        f = np.array([r["low_res_factor"] for r in rows if r["low_res"]])
        self.assertTrue(np.all((f >= cfg.low_res_min_factor) & (f <= 1.0)))
        std = np.array([r["noise_std"] for r in rows if r["noise"]])
        self.assertTrue(np.all((std >= 0) & (std <= cfg.noise_std_max)))

    def test_t10_chain_statistics_vs_reference(self):
        cfg = P.IntensityAugConfig()
        x = phantom((16, 20, 24), seed=3)
        n = 400
        tf_out = tfa.augment_group(tf.constant(np.stack([x] * n)), cfg,
                                   tf.random.Generator.from_seed(10)).numpy()
        rng = np.random.RandomState(10)
        uu = ref.sample_uniforms_np(rng, n)
        np_out = np.stack([ref.augment_np(x, uu[i], rng.normal(size=x.shape), cfg) for i in range(n)])

        def feats(a):
            a = a.reshape(n, -1)
            return {"mean": a.mean(1), "std": a.std(1), "p1": np.percentile(a, 1, axis=1),
                    "p50": np.percentile(a, 50, axis=1), "p99": np.percentile(a, 99, axis=1),
                    "frac0": (a <= 0).mean(1), "frac1": (a >= 1).mean(1)}

        ft, fn = feats(tf_out), feats(np_out)
        f0 = {k: v[0] for k, v in feats(np.stack([x.astype(np.float64)] * n)).items()}
        lines = []
        for k in ft:
            # Many images keep their statistics (an atom at the input's value): bin the values
            # in 1e-4 steps CENTERED on the input's value, so float32-vs-float64 jitter (~1e-7)
            # inside the atom does not dominate the KS statistic.
            p = stats.ks_2samp(np.round((ft[k] - f0[k]) / 1e-4), np.round((fn[k] - f0[k]) / 1e-4)).pvalue
            lines.append("%s p=%.3g" % (k, p))
            self.assertGreaterEqual(p, 1e-3, k)
        sys.stderr.write("\n[T10] KS TF vs numpy (400 each): " + ", ".join(lines) + "\n")
        self.assertGreaterEqual(tf_out.min(), 0.0)
        self.assertLessEqual(tf_out.max(), 1.0)


class TestDeterminism(unittest.TestCase):
    """T11-T13."""

    def setUp(self):
        self.x = tf.constant(np.stack([phantom((12, 14, 16), seed=s) for s in range(4)]))
        self.cfg = P.IntensityAugConfig(noise_prob=0.5, brightness_prob=0.5, contrast_prob=0.5,
                                        low_res_prob=0.5, gamma_prob=0.5)

    def run_calls(self, g, k):
        return [tfa.augment_group(self.x, self.cfg, g).numpy() for _ in range(k)]

    def test_t11_same_seed_same_output(self):
        a = self.run_calls(tf.random.Generator.from_seed(1), 2)
        b = self.run_calls(tf.random.Generator.from_seed(1), 2)
        c = self.run_calls(tf.random.Generator.from_seed(2), 2)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        self.assertFalse(np.array_equal(a[0], c[0]))
        self.assertFalse(np.array_equal(a[0], a[1]))

    def test_t12_checkpoint_resume_continues_stream(self):
        tmp = tempfile.mkdtemp(prefix="tfaug_")
        try:
            g = tf.random.Generator.from_seed(5)
            self.run_calls(g, 3)
            path = tf.train.Checkpoint(g=g).write(os.path.join(tmp, "ck"))
            want = self.run_calls(g, 2)
            g2 = tf.random.Generator.from_seed(999)
            tf.train.Checkpoint(g=g2).read(path).assert_consumed()
            got = self.run_calls(g2, 2)
            for x, y in zip(want, got):
                np.testing.assert_array_equal(x, y)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_t13_mixed_precision_policy_float32(self):
        ref32 = tfa.augment_group(self.x, self.cfg, tf.random.Generator.from_seed(4))
        old = tf.keras.mixed_precision.global_policy()
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            out = tfa.augment_group(self.x, self.cfg, tf.random.Generator.from_seed(4))
        finally:
            tf.keras.mixed_precision.set_global_policy(old)
        self.assertEqual(out.dtype, tf.float32)
        np.testing.assert_array_equal(out.numpy(), ref32.numpy())


@unittest.skipUnless(tf.config.list_physical_devices("GPU"), "needs a GPU")
class TestMemory(unittest.TestCase):
    """T14: transient GPU memory of one augment_group call (printed for the report)."""

    def measure(self, level, n):
        shape = P.LEVEL_SHAPE[level]
        vol_bytes = float(np.prod(shape)) * 4.0
        cfg = P.IntensityAugConfig(noise_prob=1.0, brightness_prob=1.0, contrast_prob=1.0,
                                   low_res_prob=1.0, gamma_prob=1.0)
        g = tf.random.Generator.from_seed(1)
        fn = tf.function(lambda v: tf.reduce_sum(tfa.augment_group(v, cfg, g)))
        with tf.device("GPU:0"):
            x = tf.random.uniform((n,) + shape + (1,))
            fn(x).numpy()                                   # trace + warm-up
            base = tf.config.experimental.get_memory_info("GPU:0")["current"]
            tf.config.experimental.reset_memory_stats("GPU:0")
            fn(x).numpy()
            peak = tf.config.experimental.get_memory_info("GPU:0")["peak"]
        extra = peak - base
        sys.stderr.write("\n[T14] level %d N %d: transient peak %.1f MiB (group %.1f MiB, volume %.1f MiB)\n"
                         % (level, n, extra / 2.0 ** 20, n * vol_bytes / 2.0 ** 20, vol_bytes / 2.0 ** 20))
        return extra, vol_bytes

    def test_t14_level2_n8(self):
        extra, vol = self.measure(2, 8)
        self.assertLessEqual(extra, 3 * 8 * vol + 8 * vol)

    def test_t14_level1_n2(self):
        try:
            extra, vol = self.measure(1, 2)
        except tf.errors.ResourceExhaustedError:
            self.skipTest("level 1 N 2 does not fit the local test memory limit")
        self.assertLessEqual(extra, 3 * 2 * vol + 8 * vol)


if __name__ == "__main__":
    unittest.main()
