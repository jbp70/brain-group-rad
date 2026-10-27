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
from data_loading import tf_intensity_aug as tfa
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


def _is_single_cycle(i, j, n):
    succ = dict(zip(i.tolist(), j.tolist()))
    seen, cur = [], 0
    while cur not in seen:
        seen.append(cur)
        cur = succ[cur]
    return len(seen) == n and cur == 0


class TestMinPairs(unittest.TestCase):
    """--min-pairs (milestone 9 phase 2, design-m9-phase2 section 5): T-P1..T-P5."""

    def test_counts_table(self):                                    # T-P1
        want_1 = {2: 1, 3: 1, 4: 1, 5: 2, 6: 3, 8: 6, 10: 9, 11: 11, 12: 13, 16: 24, 32: 99}
        want_n = {2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 8: 8, 10: 10, 11: 11, 12: 13, 16: 24, 32: 99}
        for n in want_1:
            self.assertEqual(int(pp.num_pairs(n, 0.1, "1")), want_1[n], n)
            self.assertEqual(int(pp.num_pairs(n, 0.1)), want_1[n], n)
            self.assertEqual(int(pp.num_pairs(n, 0.1, "n")), want_n[n], n)
        with self.assertRaises(ValueError):
            pp.num_pairs(4, 0.1, "x")

    def test_counts_match_memory_model(self):                       # T-P1 mirror
        from models import memory_model as MM
        for n in range(2, 41):
            for p in (0.05, 0.1, 0.25, 1.0):
                for m in pp.MIN_PAIRS:
                    self.assertEqual(int(pp.num_pairs(n, p, m)), MM.num_pairs(n, p, m), (n, p, m))

    def test_counts_match_checker(self):                            # T-P1 mirror (design section 9)
        import importlib.util
        from models import memory_model as MM
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu", "check_mixed_run.py")
        spec = importlib.util.spec_from_file_location("check_mixed_run", path)
        cmr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cmr)
        for n in range(2, 41):
            for p in (0.05, 0.1, 0.25, 1.0):
                for m in pp.MIN_PAIRS:
                    floor = 1 if m == "1" else n
                    self.assertEqual(cmr.expected_pairs(floor, n, p), int(pp.num_pairs(n, p, m)), (n, p, m))
                    self.assertEqual(cmr.expected_pairs(floor, n, p), MM.num_pairs(n, p, m), (n, p, m))

    def _check_cycle(self, i, j, n):
        i, j = np.asarray(i), np.asarray(j)
        self.assertEqual(sorted(i.tolist()), list(range(n)))
        self.assertEqual(sorted(j.tolist()), list(range(n)))
        self.assertTrue(np.all(i != j))
        self.assertTrue(_is_single_cycle(i, j, n))

    def test_cycle_eager_and_graph(self):                           # T-P2
        for n in (2, 3, 4, 8, 11):
            self.assertEqual(int(pp.num_pairs(n, 0.1, "n")), n)
            g_graph = tf.random.Generator.from_seed(0)

            @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
            def draw(nn, g=g_graph):
                i, j = pp.sample_pairs(g, nn, 0.1, "n")
                self.assertEqual(i.shape.as_list(), [None])
                return i, j

            for seed in range(200):
                i, j = pp.sample_pairs(tf.random.Generator.from_seed(seed), n, 0.1, "n")
                self._check_cycle(i.numpy(), j.numpy(), n)
                gi, gj = draw(tf.constant(n))
                self._check_cycle(gi.numpy(), gj.numpy(), n)

    def test_more_pairs_than_n_equal_today(self):                   # T-P3
        for n in (12, 16):
            for seed in range(5):
                a = pp.sample_pairs(tf.random.Generator.from_seed(seed), n, 0.1, "1")
                b = pp.sample_pairs(tf.random.Generator.from_seed(seed), n, 0.1, "n")
                np.testing.assert_array_equal(a[0].numpy(), b[0].numpy())
                np.testing.assert_array_equal(a[1].numpy(), b[1].numpy())

    def test_generator_advance_unchanged(self):                     # T-P4
        for n in (2, 4, 8, 16):
            g1, gn = tf.random.Generator.from_seed(3), tf.random.Generator.from_seed(3)
            pp.sample_pairs(g1, n, 0.1, "1")
            pp.sample_pairs(gn, n, 0.1, "n")
            np.testing.assert_array_equal(g1.state.numpy(), gn.state.numpy())
            np.testing.assert_array_equal(pp.sample_flip_axes(g1).numpy(), pp.sample_flip_axes(gn).numpy())

    def test_same_seed_same_pairs(self):                            # T-P5
        for n in (4, 11):
            a = pp.sample_pairs(tf.random.Generator.from_seed(9), n, 0.1, "n")
            b = pp.sample_pairs(tf.random.Generator.from_seed(9), n, 0.1, "n")
            np.testing.assert_array_equal(a[0].numpy(), b[0].numpy())
            np.testing.assert_array_equal(a[1].numpy(), b[1].numpy())


def _load_checker():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu", "check_mixed_run.py")
    spec = importlib.util.spec_from_file_location("check_mixed_run", path)
    cmr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmr)
    return cmr


class TestMaxPairs(unittest.TestCase):
    """--max-pairs K (retrain-prereqs design 16.3): k = min(K, max(floor, round(p N (N - 1))))."""

    NS = (4, 8, 32, 48, 64)
    KS = (0, 16, 64)

    def test_counts(self):
        uncapped = {"1": {4: 1, 8: 6, 32: 99, 48: 226, 64: 403},
                    "n": {4: 4, 8: 8, 32: 99, 48: 226, 64: 403}}
        for m in pp.MIN_PAIRS:
            for n in self.NS:
                for k in self.KS:
                    want = uncapped[m][n] if k == 0 else min(k, uncapped[m][n])
                    self.assertEqual(int(pp.num_pairs(n, 0.1, m, k)), want, (m, n, k))
                    self.assertEqual(pp.num_pairs_static(n, 0.1, m, k), want, (m, n, k))
                self.assertEqual(int(pp.num_pairs(n, 0.1, m)), uncapped[m][n])
                self.assertEqual(pp.num_pairs_static(n, 0.1, m), uncapped[m][n])
        # A cap below the min-pairs floor wins (applied last).
        self.assertEqual(int(pp.num_pairs(32, 0.1, "n", 16)), 16)
        self.assertEqual(int(pp.num_pairs(8, 0.1, "n", 5)), 5)

    def test_mirrors_agree(self):
        from models import memory_model as MM
        cmr = _load_checker()
        for n in list(range(2, 41)) + [48, 64, 96]:
            for p in (0.05, 0.1, 0.25, 1.0):
                for m in pp.MIN_PAIRS:
                    floor = 1 if m == "1" else n
                    for k in (0, 1, 5, 16, 64):
                        want = int(pp.num_pairs(n, p, m, k))
                        self.assertEqual(pp.num_pairs_static(n, p, m, k), want, (n, p, m, k))
                        self.assertEqual(MM.num_pairs(n, p, m, k), want, (n, p, m, k))
                        self.assertEqual(cmr.expected_pairs(floor, n, p, k), want, (n, p, m, k))

    def test_invalid(self):
        for bad in (-1, 1.5, True, "4", None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                pp.num_pairs(8, 0.1, "1", bad)
            with self.assertRaises(ValueError, msg=repr(bad)):
                pp.num_pairs_static(8, 0.1, "1", bad)
        with self.assertRaises(ValueError):
            pp.LinearPostprocess("rigid", 4, max_pairs=-1)
        with self.assertRaises(ValueError):
            pp.DeformPostprocess(4, max_pairs=-2)

    def test_zero_is_bit_exact(self):
        for m in pp.MIN_PAIRS:
            for n in (4, 8, 16, 48):
                ga, gb = tf.random.Generator.from_seed(5), tf.random.Generator.from_seed(5)
                a = pp.sample_pairs(ga, n, 0.1, m)
                b = pp.sample_pairs(gb, n, 0.1, m, 0)
                np.testing.assert_array_equal(a[0].numpy(), b[0].numpy())
                np.testing.assert_array_equal(a[1].numpy(), b[1].numpy())
                np.testing.assert_array_equal(ga.state.numpy(), gb.state.numpy())

    def test_cap_keeps_rng_and_prefix(self):
        """The cap does not change the RNG draw: same generator advance, and (min-pairs 1, top_k
        sorted) the capped pairs are the first K of the uncapped ones."""
        for n in (8, 32, 48):
            ga, gb = tf.random.Generator.from_seed(11), tf.random.Generator.from_seed(11)
            a = pp.sample_pairs(ga, n, 0.1, "1")
            b = pp.sample_pairs(gb, n, 0.1, "1", 16)
            k = min(16, int(pp.num_pairs(n, 0.1)))
            self.assertEqual(len(b[0].numpy()), k)
            np.testing.assert_array_equal(a[0].numpy()[:k], b[0].numpy())
            np.testing.assert_array_equal(a[1].numpy()[:k], b[1].numpy())
            np.testing.assert_array_equal(ga.state.numpy(), gb.state.numpy())

    def test_capped_pairs_valid(self):
        for m in pp.MIN_PAIRS:
            for n in (8, 32, 64):
                for k in (5, 16, 64):
                    i, j = pp.sample_pairs(tf.random.Generator.from_seed(n + k), n, 0.1, m, k)
                    i, j = i.numpy(), j.numpy()
                    self.assertEqual(len(i), int(pp.num_pairs(n, 0.1, m, k)), (m, n, k))
                    self.assertTrue(np.all(i != j))
                    self.assertTrue(np.all((i >= 0) & (i < n) & (j >= 0) & (j < n)))
                    self.assertEqual(len(set(zip(i, j))), len(i))

    def test_cap_on_n_gives_cycle(self):
        """--min-pairs n with a cap landing exactly on k == N: the N-cycle (eager and graph)."""
        n, k = 8, 8                       # round(0.5 * 56) = 28 -> capped to 8 == N
        self.assertEqual(int(pp.num_pairs(n, 0.5, "n", k)), n)
        g_graph = tf.random.Generator.from_seed(0)

        @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
        def draw(nn):
            return pp.sample_pairs(g_graph, nn, 0.5, "n", k)

        for seed in range(50):
            i, j = pp.sample_pairs(tf.random.Generator.from_seed(seed), n, 0.5, "n", k)
            for a, b in ((i.numpy(), j.numpy()), tuple(x.numpy() for x in draw(tf.constant(n)))):
                self.assertEqual(sorted(a.tolist()), list(range(n)))
                self.assertEqual(sorted(b.tolist()), list(range(n)))
                self.assertTrue(_is_single_cycle(a, b, n))

    def test_postprocess_counts(self):
        n = 8
        images = np.zeros((n,) + P.LEVEL_SHAPE[4] + (1,), np.float32)
        labels = np.zeros((n,) + P.LEVEL_SHAPE[4] + (1,), np.uint8)
        out = pp.DeformPostprocess(4, p_pdice=0.5, seed=3, min_pairs="n", max_pairs=5)(images, labels)
        self.assertEqual(len(out["pair_i"].numpy()), 5)
        t, rt = random_dali_batch(np.random.default_rng(0), "rigid", 4, n)
        out = pp.LinearPostprocess("rigid", 4, p_pair=0.5, seed=3, max_pairs=7)(images, t, rt)
        self.assertEqual(out["pair_i"].shape[0], 7)
        self.assertEqual(out["P_star"].shape[0], 7)


class TestMinPairsPostprocess(unittest.TestCase):
    """T-P6: both postprocess classes with min_pairs='n' at N 4."""

    def setUp(self):
        self.n = 4
        self.images = np.zeros((self.n,) + P.LEVEL_SHAPE[4] + (1,), np.float32)
        self.labels = np.zeros((self.n,) + P.LEVEL_SHAPE[4] + (1,), np.uint8)

    def test_deform_cycle(self):
        post = pp.DeformPostprocess(4, seed=5, min_pairs="n")
        self.assertEqual(post.min_pairs, "n")
        self.assertEqual(pp.DeformPostprocess(4).min_pairs, "1")
        out = post(self.images, self.labels)
        i, j = out["pair_i"].numpy(), out["pair_j"].numpy()
        self.assertEqual(len(i), 4)
        self.assertEqual(sorted(i.tolist()), [0, 1, 2, 3])
        self.assertEqual(sorted(j.tolist()), [0, 1, 2, 3])
        self.assertTrue(_is_single_cycle(i, j, 4))
        with self.assertRaises(ValueError):
            pp.DeformPostprocess(4, min_pairs="x")

    def test_linear_count(self):
        t, rt = random_dali_batch(np.random.default_rng(2), "rigid", 4, self.n)
        post = pp.LinearPostprocess("rigid", 4, seed=5, min_pairs="n")
        self.assertEqual(post.min_pairs, "n")
        out = post(self.images, t, rt)
        self.assertEqual(out["pair_i"].shape[0], 4)
        self.assertEqual(out["P_star"].shape[0], 4)
        self.assertTrue(np.all(out["pair_i"].numpy() != out["pair_j"].numpy()))
        with self.assertRaises(ValueError):
            pp.LinearPostprocess("rigid", 4, min_pairs="x")


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


class TestDeformPostprocess(unittest.TestCase):
    """Milestone 6: shared flips of images and labels, one-hot, pairs, residual T."""

    def setUp(self):
        self.rng = np.random.default_rng(31)
        self.n = 4
        shape = (self.n,) + P.LEVEL_SHAPE[4] + (1,)
        self.images = self.rng.uniform(size=shape).astype(np.float32)
        self.labels = self.rng.integers(0, P.N_TRAIN_LABELS, size=shape).astype(np.uint8)

    def test_flips_shared_and_onehot(self):
        post = pp.DeformPostprocess(4, seed=2)
        for flags in FLIPS:
            out = post(self.images, self.labels, flip_axes=flags)
            axes = tuple(a + 1 for a in range(3) if flags[a])
            ref_img = np.flip(self.images, axis=axes) if axes else self.images
            ref_lab = (np.flip(self.labels, axis=axes) if axes else self.labels)[..., 0]
            np.testing.assert_array_equal(out["images"].numpy(), ref_img)
            np.testing.assert_array_equal(out["labels"].numpy(), ref_lab.astype(np.int32))
            self.assertEqual(out["labels"].dtype, tf.int32)
            oh = out["onehot"].numpy()
            self.assertEqual(oh.shape, self.labels.shape[:4] + (P.N_TRAIN_LABELS,))
            self.assertEqual(oh.dtype, np.float32)
            np.testing.assert_array_equal(oh.sum(axis=-1), 1.0)
            np.testing.assert_array_equal(np.argmax(oh, axis=-1), ref_lab)
            np.testing.assert_array_equal(out["flip_axes"].numpy(), flags)
            self.assertNotIn("T_render", out)
        with self.assertRaises(ValueError):
            post(self.images, self.labels, np.zeros((self.n, 3, 4), np.float32))

    def test_pairs_distinct_and_deterministic(self):
        a = pp.DeformPostprocess(4, p_pdice=0.5, seed=7)(self.images, self.labels)
        b = pp.DeformPostprocess(4, p_pdice=0.5, seed=7)(self.images, self.labels)
        i, j = a["pair_i"].numpy(), a["pair_j"].numpy()
        self.assertEqual(len(i), int(pp.num_pairs(self.n, 0.5)))
        self.assertTrue(np.all(i != j))
        self.assertEqual(len(set(zip(i, j))), len(i))
        np.testing.assert_array_equal(i, b["pair_i"].numpy())
        np.testing.assert_array_equal(j, b["pair_j"].numpy())
        np.testing.assert_array_equal(a["flip_axes"].numpy(), b["flip_axes"].numpy())
        np.testing.assert_array_equal(a["labels"].numpy(), b["labels"].numpy())

    def test_residual_t_conjugated(self):
        t, _ = random_dali_batch(self.rng, "affine", 4, self.n)
        post = pp.DeformPostprocess(4, seed=3, residual_affine=True)
        flags = [True, False, True]
        out = post(self.images, self.labels, t, flip_axes=flags)
        f = tfm.flip_matrix(flags).numpy()
        np.testing.assert_allclose(out["T_render"].numpy(), f @ pp.dali_to_mm(t, 4).numpy() @ f,
                                   atol=1e-5)
        with self.assertRaises(ValueError):
            post(self.images, self.labels)

    def test_tf_function_uint8_random_flips(self):
        post = pp.DeformPostprocess(4, flip_prob=0.5, seed=8)
        traces = []

        @tf.function
        def step(x, y):
            traces.append(1)
            return post(x, y)

        x = tf.constant(self.images)
        y = tf.constant(self.labels)
        for _ in range(4):
            out = step(x, y)
            flags = out["flip_axes"].numpy()
            img = tfm.flip_images(x, flags).numpy()
            np.testing.assert_array_equal(out["images"].numpy(), img)
            ref = tfm.flip_images(y, flags).numpy()[..., 0].astype(np.int32)
            np.testing.assert_array_equal(out["labels"].numpy(), ref)
        self.assertEqual(len(traces), 1)


def _graph_op_names(fn, *args):
    return [op.name for op in fn.get_concrete_function(*args).graph.get_operations()]


class TestIntensityAugPostprocess(unittest.TestCase):
    """--intensity-aug-device tf: TF intensity augmentation inside both post-processes."""

    def setUp(self):
        self.rng = np.random.default_rng(41)
        self.n = 4
        g = np.meshgrid(*[np.linspace(0, 1, s) for s in P.LEVEL_SHAPE[4]], indexing="ij")
        vol = (0.2 + 0.6 * (g[0] * g[1] + 0.5 * g[2]) / 1.5).astype(np.float32)
        self.images = np.stack([vol] * self.n)[..., None]
        self.labels = self.rng.integers(0, P.N_TRAIN_LABELS, size=self.images.shape).astype(np.uint8)
        # Every op applied, so the images change with certainty.
        self.cfg = P.IntensityAugConfig(noise_prob=1.0, brightness_prob=1.0, contrast_prob=1.0,
                                        low_res_prob=1.0, gamma_prob=1.0)

    def test_default_builds_no_op(self):
        t, rt = random_dali_batch(self.rng, "rigid", 4, self.n)
        post = pp.LinearPostprocess("rigid", 4, seed=4)
        self.assertIsNone(post.aug_generator)
        self.assertIsNone(post.intensity_aug)
        names = _graph_op_names(tf.function(lambda x, a, b: post(x, a, b)), self.images, t, rt)
        self.assertFalse([s for s in names if "intensity_aug" in s])
        dpost = pp.DeformPostprocess(4, seed=4, intensity_aug=P.IntensityAugConfig.off())
        self.assertIsNone(dpost.aug_generator)
        names = _graph_op_names(tf.function(lambda x, y: dpost(x, y)), self.images, self.labels)
        self.assertFalse([s for s in names if "intensity_aug" in s])
        on = pp.LinearPostprocess("rigid", 4, seed=4, intensity_aug=self.cfg)
        names = _graph_op_names(tf.function(lambda x, a, b: on(x, a, b)), self.images, t, rt)
        self.assertTrue([s for s in names if s.startswith("intensity_aug/")])

    def test_linear_images_change_rest_unchanged(self):
        for task in P.TASKS:
            t, rt = random_dali_batch(self.rng, task, 4, self.n)
            rt = rt if task == "rigid" else None
            base = pp.LinearPostprocess(task, 4, seed=9)
            aug = pp.LinearPostprocess(task, 4, seed=9, intensity_aug=self.cfg)
            self.assertIsNotNone(aug.aug_generator)
            self.assertIsNot(aug.aug_generator, aug.generator)
            for _ in range(2):
                a, b = base(self.images, t, rt), aug(self.images, t, rt)
                self.assertFalse(np.allclose(a["images"].numpy(), b["images"].numpy(), atol=1e-3))
                for k in ("T_render", "T_target", "A_star", "P_star", "pair_i", "pair_j", "flip_axes"):
                    np.testing.assert_array_equal(a[k].numpy(), b[k].numpy(), k)
                img = b["images"].numpy()
                self.assertEqual(b["images"].dtype, tf.float32)
                self.assertGreaterEqual(img.min(), 0.0)
                self.assertLessEqual(img.max(), 1.0)
                self.assertEqual(tuple(img.shape), self.images.shape)

    def test_deform_labels_never_augmented(self):
        base = pp.DeformPostprocess(4, seed=5)
        aug = pp.DeformPostprocess(4, seed=5, intensity_aug=self.cfg)
        for _ in range(2):
            a, b = base(self.images, self.labels), aug(self.images, self.labels)
            self.assertFalse(np.allclose(a["images"].numpy(), b["images"].numpy(), atol=1e-3))
            for k in ("labels", "onehot", "pair_i", "pair_j", "flip_axes"):
                np.testing.assert_array_equal(a[k].numpy(), b[k].numpy(), k)

    def test_flip_commutes(self):
        # Fixed flip state: augmenting then flipping == flipping the augmentation of the same draw.
        cfg = self.cfg
        x = tf.constant(self.images)
        aug = pp.DeformPostprocess(4, seed=6, intensity_aug=cfg)
        flags = [True, False, True]
        out = aug(self.images, self.labels, flip_axes=flags)["images"].numpy()
        g = tf.random.Generator.from_seed(6 + P.INTENSITY_AUG_SEED_OFFSET)
        ref = tfm.flip_images(tfa.augment_group(x, cfg, g), flags).numpy()
        np.testing.assert_array_equal(out, ref)


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
