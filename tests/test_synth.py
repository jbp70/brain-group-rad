"""Tests for data_loading.synth and the synthesis path of data_loading.postprocess
(synth-contrast design 4.1-4.4, 5.2, 7.1; orchestrator decisions OD1, OD5, OD6). Level 4, N 4."""
import collections
import dataclasses
import unittest
from unittest import mock

import numpy as np

from tests import tf_setup  # noqa: F401  (log level, GPU memory growth)
import tensorflow as tf
from scipy import ndimage

from data_loading import params as P
from data_loading import postprocess as pp
from data_loading import synth
from models import layers
from models import transforms as tfm
from preprocessing import luts
from runtime import fixed_data as fd
from tests import runtime_helpers as rh

LEVEL = 4
N = 4
SHAPE = P.LEVEL_SHAPE[LEVEL]
SPACING = P.LEVEL_SPACING[LEVEL]
# Seed of the trace-once test: with p 0.5 its first 8 calls give both outcomes (found once).
TRACE_SEED = 3

# Everything but the contrast off: no std, blur only the partial-volume term, no bias, gamma 1,
# no noise, no jitter, no parcel deviation, always re-masked.
PLAIN = P.SynthConfig(prob=1.0, jitter=0.0, std_max=0.0, fwhm_mr_mm=(0.0, 0.0),
                      fwhm_pet_mm=(0.0, 0.0), fwhm_image_jitter=0.0, bias_std_max=0.0,
                      gamma_log_std=0.0, noise_std_max=0.0, remask_prob=1.0, hier_dev=0.0,
                      hier_dev_cortex=0.0)
ON = P.SynthConfig(prob=1.0)


def make_group(seed=0, n=N):
    """(images float32 [n, X, Y, Z, 1], train18 labels uint8 [n, ..., 1], fine uint8 [n, ..., 1])."""
    rng = np.random.default_rng(seed)
    imgs, labs, fines = [], [], []
    for _ in range(n):
        img, lab = fd.synthetic_label_phantom(SHAPE, rng)
        imgs.append(img)
        labs.append(lab)
        fines.append(rh.fine_labels_from_train18(lab, rng))
    return (np.stack(imgs)[..., None].astype(np.float32), np.stack(labs)[..., None],
            np.stack(fines)[..., None])


def linear_batch(task, seed=0, n=N):
    """DALI-format (images, T_dali[, Rt_dali]) + fine labels (template space) + T (mm)."""
    images, labs, fine = make_group(seed, n)
    rng = np.random.default_rng(seed + 100)
    T, tgt = fd.random_linear_transforms(rng, n, task)
    batch = fd.to_dali_batch(images, T, tgt, task, LEVEL)
    return batch, fine, labs


def draw(cfg, seed=0, n=N):
    return synth.draw_synth_params(tf.random.Generator.from_seed(seed), n, cfg)


def onehot_depths(fn, *args):
    """Multiset (Counter) of OneHot depths in the FULL graph-def of a traced call: the outer
    graph and every library function (tf.cond branches, map_fn / while bodies)."""
    gd = fn.get_concrete_function(*args).graph.as_graph_def()
    lists = [list(gd.node)] + [list(f.node_def) for f in gd.library.function]
    depths = collections.Counter()
    for nodes in lists:
        by_name = {nd.name: nd for nd in nodes}
        for nd in nodes:
            if nd.op != "OneHot":
                continue
            src = by_name.get(nd.input[1].lstrip("^").split(":")[0])
            if src is not None and src.op == "Const":
                depths[int(tf.make_ndarray(src.attr["value"].tensor))] += 1
            else:
                depths["dynamic"] += 1
    return depths


class TestTables(unittest.TestCase):
    def test_fine_to_train18_table(self):
        table = synth.fine_to_train18_table().numpy()
        self.assertEqual(table.dtype, np.int32)
        self.assertEqual(table.shape, (256,))
        fine_map, t18_map = luts.LUTS["synthfine"]["map"], luts.LUTS["train18"]["map"]
        self.assertEqual(len(fine_map), 103)
        for raw, k in fine_map.items():
            self.assertEqual(table[k], t18_map[raw], raw)
        np.testing.assert_array_equal(table[103:], 0)
        np.testing.assert_array_equal(table, luts.synthfine_to_train18_table().astype(np.int32))

    def test_dev_halfwidth_od1(self):
        cfg = P.SynthConfig()
        self.assertEqual((cfg.hier_dev, cfg.hier_dev_cortex), (0.1, 0.25))
        half = synth.dev_halfwidth_table(cfg).numpy()
        cortex = luts.synthfine_cortex_ids()
        self.assertEqual(len(cortex), 72)
        want = np.zeros(256, np.float32)
        want[1:103] = 0.1
        want[cortex] = 0.25
        np.testing.assert_array_equal(half, want)
        table = luts.synthfine_to_train18_table()
        np.testing.assert_array_equal(np.nonzero(half == np.float32(0.25))[0],
                                      np.nonzero(table == 2)[0])

    def test_mean_table_deviation_od1(self):
        cfg = P.SynthConfig(prob=0.5)
        g = luts.synthfine_to_train18_table().astype(np.int64)
        cortex = np.zeros(256, bool)
        cortex[luts.synthfine_cortex_ids()] = True
        other = np.zeros(256, bool)
        other[1:103] = True
        other &= ~cortex
        max_cortex, max_other = 0.0, 0.0
        for seed in range(5):
            prm = draw(cfg, seed)
            mean, std = prm["mean"].numpy(), prm["std"].numpy()
            dev = np.abs(mean - prm["cls_mean"].numpy()[g])
            self.assertLessEqual(dev[cortex].max(), 0.25 + 1e-6)
            self.assertLessEqual(dev[other].max(), 0.1 + 1e-6)
            max_cortex = max(max_cortex, float(dev[cortex].max()))
            max_other = max(max_other, float(dev[other].max()))
            self.assertEqual(mean[0], 0.0)
            self.assertEqual(std[0], 0.0)
            np.testing.assert_array_equal(mean[103:], 0.0)
            self.assertTrue(np.all((mean >= 0) & (mean <= 1)))
            self.assertTrue(np.all((std >= 0) & (std <= cfg.std_max + 1e-7)))
            # The std is shared by every fine id of a class (hierarchical).
            for c in range(1, P.N_TRAIN_LABELS):
                s = std[:103][g[:103] == c]
                np.testing.assert_array_equal(s, s[0], err_msg=str(c))
        self.assertGreater(max_cortex, 0.1)

    def test_draw_pattern_fixed(self):
        """Same generator advance whatever p (incl. 0 and 1) and the flag value."""
        states = []
        for p in (0.0, 0.3, 1.0):
            g = tf.random.Generator.from_seed(7)
            prm = synth.draw_synth_params(g, N, P.SynthConfig(prob=p))
            states.append(g.state.numpy())
            self.assertEqual(prm["flag"].dtype, tf.bool)
            self.assertEqual(prm["flag"].shape, ())
            self.assertEqual(prm["jitter"].shape, (N, 256))
            self.assertEqual(prm["fwhm_mm"].shape, (N, 3))
            self.assertEqual(prm["noise_std"].shape, (N,))
            self.assertEqual(prm["bias"].shape, (N, 64))
            self.assertEqual(prm["seeds"].shape, (N, 2, 2))
        for s in states[1:]:
            np.testing.assert_array_equal(s, states[0])
        self.assertFalse(bool(draw(P.SynthConfig(prob=0.0))["flag"]))
        self.assertTrue(bool(draw(P.SynthConfig(prob=1.0))["flag"]))

    def test_operators(self):
        for n, fwhm in ((40, 0.0), (56, 3.0), (48, 8.0)):
            k = synth.gaussian_operator(n, fwhm, SPACING).numpy()
            self.assertEqual(k.shape, (n, n))
            np.testing.assert_allclose(k.sum(axis=1), 1.0, atol=1e-6)
            self.assertTrue(np.all(k >= 0))
            self.assertEqual(int(np.argmax(k[n // 2])), n // 2)
        # FWHM 0 keeps only the partial-volume term sigma = 1 / sqrt(12) voxel.
        k0 = synth.gaussian_operator(40, 0.0, SPACING).numpy()
        w = np.exp(-1.0 / (2.0 / 12.0))
        self.assertAlmostEqual(float(k0[20, 21] / k0[20, 20]), w, places=5)
        for n in (40, 80, 160):
            m = synth.bias_interp_matrix(n, 4)
            self.assertEqual(m.shape, (n, 4))
            np.testing.assert_allclose(m.sum(axis=1), 1.0, atol=1e-6)
            self.assertTrue(np.all(m >= 0))
        # Control points at the same physical position at every level: the field of a linear
        # ramp in control values is the same function of mm.
        a, b = synth.bias_interp_matrix(40, 4) @ np.arange(4.0), synth.bias_interp_matrix(160, 4) @ np.arange(4.0)
        np.testing.assert_allclose(a, b.reshape(40, 4).mean(axis=1), atol=1e-5)


class TestSynthesize(unittest.TestCase):
    def setUp(self):
        self.images, self.labs, self.fine = make_group(0)

    def test_shapes_range_finite(self):
        for lab in (self.fine, self.fine[..., 0]):
            out = synth.synthesize_group(tf.constant(lab), draw(ON, 1), ON, LEVEL).numpy()
            self.assertEqual(out.shape, (N,) + SHAPE + (1,))
            self.assertEqual(out.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(out)))
            self.assertGreaterEqual(out.min(), 0.0)
            self.assertLessEqual(out.max(), 1.0)
            self.assertGreater(out.max(), 0.5)

    def test_seeded_determinism(self):
        with tf.device("/cpu:0"):
            a = synth.synthesize_group(tf.constant(self.fine), draw(ON, 5), ON, LEVEL).numpy()
            b = synth.synthesize_group(tf.constant(self.fine), draw(ON, 5), ON, LEVEL).numpy()
            c = synth.synthesize_group(tf.constant(self.fine), draw(ON, 6), ON, LEVEL).numpy()
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def _interior_means(self, out):
        """Per image {fine id: mean over the 2-voxel eroded label region}."""
        res = []
        for i in range(N):
            d = {}
            for f in np.unique(self.fine[i, ..., 0]):
                if f == 0:
                    continue
                core = ndimage.binary_erosion(self.fine[i, ..., 0] == f, iterations=2)
                if core.sum() >= 5:
                    d[int(f)] = float(out[i, ..., 0][core].mean())
            res.append(d)
        return res

    def test_one_contrast_per_group(self):
        out = synth.synthesize_group(tf.constant(self.fine), draw(PLAIN, 2), PLAIN, LEVEL).numpy()
        means = self._interior_means(out)
        common = set.intersection(*[set(m) for m in means])
        # Labels below the clip value; the brightest of them is the per-image reference (removes
        # the per-image normalisation factor).
        common = [f for f in common if all(m[f] < 1.0 - 1e-6 for m in means)]
        self.assertGreaterEqual(len(common), 3)
        ref = max(common, key=lambda f: means[0][f])
        for f in common:
            r = [m[f] / m[ref] for m in means]
            np.testing.assert_allclose(r, r[0], rtol=1e-5, err_msg=str(f))

    def test_jitter_bounds(self):
        j = 0.08
        cfg = dataclasses.replace(PLAIN, jitter=j)
        prm = draw(cfg, 3)
        d = prm["jitter"].numpy()
        self.assertTrue(np.all(np.abs(d) <= j + 1e-7))
        self.assertGreater(np.abs(d).max(), 0.5 * j)
        idx = tf.constant(np.arange(256, dtype=np.int32).reshape(4, 8, 8))
        x = [synth.contrast_image(idx, prm["mean"], prm["std"], prm["jitter"][i],
                                  prm["seeds"][i, 0]).numpy().reshape(-1) for i in range(N)]
        mean = prm["mean"].numpy()
        ok = mean > 1e-3
        lo, hi = (1 - j) / (1 + j), (1 + j) / (1 - j)
        for a in range(N):
            for b in range(N):
                r = x[a][ok] / x[b][ok]
                self.assertTrue(np.all((r >= lo - 1e-5) & (r <= hi + 1e-5)), (a, b))
        np.testing.assert_allclose(x[0][ok] / mean[ok], 1.0 + d[0][ok], rtol=1e-5)

    def test_background_zero_with_remask(self):
        cfg = P.SynthConfig(prob=1.0, remask_prob=1.0, noise_std_max=0.2, fwhm_mr_mm=(3.0, 3.0),
                            fwhm_pet_mm=(8.0, 8.0))
        out = synth.synthesize_group(tf.constant(self.fine), draw(cfg, 4), cfg, LEVEL).numpy()
        bg = self.fine == 0
        np.testing.assert_array_equal(out[bg], 0.0)
        self.assertGreater(out[~bg].max(), 0.0)
        # Without the re-mask the blur spills into the background.
        cfg0 = dataclasses.replace(cfg, remask_prob=0.0, noise_std_max=0.0)
        out0 = synth.synthesize_group(tf.constant(self.fine), draw(cfg0, 4), cfg0, LEVEL).numpy()
        self.assertGreater(out0[bg].max(), 0.0)

    def test_normalisation(self):
        cfg = P.SynthConfig(prob=1.0, noise_std_max=0.0, gamma_log_std=0.0, remask_prob=1.0)
        for seed in range(3):
            out = synth.synthesize_group(tf.constant(self.fine), draw(cfg, seed), cfg, LEVEL).numpy()
            for i in range(N):
                x = out[i, ..., 0]
                mask = self.fine[i, ..., 0] > 0
                self.assertLessEqual(x.max(), 1.0)
                self.assertEqual(x.max(), 1.0)
                q = float(np.quantile(x[mask], 0.995))
                self.assertGreater(q, 0.9, (seed, i))
                self.assertLessEqual(q, 1.0)

    def test_empty_mask_finite(self):
        lab = np.zeros((2,) + SHAPE + (1,), np.uint8)
        out = synth.synthesize_group(tf.constant(lab), draw(ON, 0, 2), ON, LEVEL).numpy()
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_array_equal(out, 0.0)

    def test_maybe_synthesize_false_is_identity(self):
        x = tf.constant(self.images)
        prm = draw(ON, 0)
        out = synth.maybe_synthesize(x, tf.constant(self.fine), tf.constant(False), prm, ON, LEVEL)
        np.testing.assert_array_equal(out.numpy(), self.images)
        out = synth.maybe_synthesize(x, tf.constant(self.fine), tf.constant(True), prm, ON, LEVEL)
        self.assertFalse(np.allclose(out.numpy(), self.images, atol=1e-3))


class TestPostprocessSynth(unittest.TestCase):
    def test_default_off_builds_nothing(self):
        for cfg in (None, P.SynthConfig(prob=0.0)):
            for post in (pp.LinearPostprocess("rigid", LEVEL, synth=cfg),
                         pp.LinearPostprocess("affine", LEVEL, synth=cfg),
                         pp.DeformPostprocess(LEVEL, synth=cfg)):
                self.assertIsNone(post.synth)
                self.assertIsNone(post.synth_generator)
        # prob 0 traces exactly the graph of synth=None.
        batch, fine, labs = linear_batch("rigid")
        images = batch[0]
        gds = []
        for cfg in (None, P.SynthConfig(prob=0.0)):
            post = pp.LinearPostprocess("rigid", LEVEL, seed=1, synth=cfg)
            fn = tf.function(lambda x, a, b, post=post: post(x, a, b))
            gd = fn.get_concrete_function(*batch).graph.as_graph_def()
            gds.append(([nd.op for nd in gd.node], len(gd.library.function)))
            out = post(*batch)
            self.assertNotIn("synth", out)
        self.assertEqual(gds[0], gds[1])
        dgds = []
        for cfg in (None, P.SynthConfig(prob=0.0)):
            post = pp.DeformPostprocess(LEVEL, seed=1, synth=cfg)
            fn = tf.function(lambda x, y, post=post: post(x, y))
            gd = fn.get_concrete_function(images, tf.constant(labs)).graph.as_graph_def()
            dgds.append(([nd.op for nd in gd.node], len(gd.library.function)))
            self.assertFalse([nd.name for nd in gd.node if "synth" in nd.name])
        self.assertEqual(dgds[0], dgds[1])

    def test_on_has_own_generator(self):
        post = pp.LinearPostprocess("rigid", LEVEL, seed=4, synth=ON)
        self.assertIs(post.synth, ON)
        self.assertIsNotNone(post.synth_generator)
        self.assertIsNot(post.synth_generator, post.generator)
        ref = tf.random.Generator.from_seed(4 + P.SYNTH_SEED_OFFSET)
        np.testing.assert_array_equal(post.synth_generator.state.numpy(), ref.state.numpy())
        with self.assertRaises(ValueError):
            pp.DeformPostprocess(LEVEL, synth=P.SynthConfig(prob=0.5, device="tpu"))

    def test_linear_without_labels_raises(self):
        batch, fine, _ = linear_batch("rigid")
        post = pp.LinearPostprocess("rigid", LEVEL, seed=4, synth=ON)
        with self.assertRaisesRegex(ValueError, "labels= required"):
            post(*batch)

    def test_override_false_equals_off(self):
        aug = P.IntensityAugConfig(noise_prob=1.0, brightness_prob=1.0, contrast_prob=1.0,
                                   low_res_prob=1.0, gamma_prob=1.0)
        cfg = P.SynthConfig(prob=0.5)
        for task in P.TASKS:
            batch, fine, _ = linear_batch(task, seed=1)
            on = pp.LinearPostprocess(task, LEVEL, seed=9, intensity_aug=aug, synth=cfg)
            off = pp.LinearPostprocess(task, LEVEL, seed=9, intensity_aug=aug)
            for flips in ((True, False, True), None):
                a = on(*batch, flip_axes=flips, labels=tf.constant(fine), synth_override=False)
                b = off(*batch, flip_axes=flips)
                self.assertFalse(bool(a["synth"].numpy()))
                self.assertEqual(set(a) - {"synth"}, set(b))
                for k in b:
                    np.testing.assert_array_equal(a[k].numpy(), b[k].numpy(), err_msg=k)
        images, labs, fine = make_group(2)
        on = pp.DeformPostprocess(LEVEL, seed=9, intensity_aug=aug, synth=cfg)
        off = pp.DeformPostprocess(LEVEL, seed=9, intensity_aug=aug)
        for flips in ((False, True, True), None):
            a = on(images, fine, flip_axes=flips, synth_override=False)
            b = off(images, labs, flip_axes=flips)
            self.assertFalse(bool(a["synth"].numpy()))
            self.assertEqual(set(a) - {"synth"}, set(b))
            for k in b:
                np.testing.assert_array_equal(a[k].numpy(), b[k].numpy(), err_msg=k)

    def test_deform_override_true(self):
        images, labs, fine = make_group(3)
        post = pp.DeformPostprocess(LEVEL, seed=2, synth=ON)
        out = post(images, fine, flip_axes=(False, False, False), synth_override=True)
        self.assertTrue(bool(out["synth"].numpy()))
        np.testing.assert_array_equal(out["labels"].numpy(), labs[..., 0].astype(np.int32))
        np.testing.assert_array_equal(np.argmax(out["onehot"].numpy(), -1), labs[..., 0])
        img = out["images"].numpy()
        self.assertFalse(np.allclose(img, images, atol=1e-3))
        self.assertTrue(np.all((img >= 0) & (img <= 1)))
        # Same draws as synthesize_group from the post-process's synth generator.
        g = tf.random.Generator.from_seed(2 + P.SYNTH_SEED_OFFSET)
        ref = synth.synthesize_group(tf.constant(fine), synth.draw_synth_params(g, N, ON), ON, LEVEL)
        np.testing.assert_allclose(img, ref.numpy(), atol=1e-6)

    def test_render_consistency(self):
        for task in P.TASKS:
            batch, fine, _ = linear_batch(task, seed=4)
            seed = 6
            on = pp.LinearPostprocess(task, LEVEL, seed=seed, synth=ON)
            off = pp.LinearPostprocess(task, LEVEL, seed=seed)
            flips = (False, False, False)
            a = on(*batch, flip_axes=flips, labels=tf.constant(fine), synth_override=True)
            b = off(*batch, flip_axes=flips)
            self.assertTrue(bool(a["synth"].numpy()))
            g = tf.random.Generator.from_seed(seed + P.SYNTH_SEED_OFFSET)
            s = synth.synthesize_group(tf.constant(fine), synth.draw_synth_params(g, N, ON), ON, LEVEL)
            t_mm = pp.dali_to_mm(batch[1], LEVEL)
            ref = layers.transform_image(s, tfm.affine_inverse(t_mm), SPACING, interpolation="linear",
                                         fill_value=0.0).numpy()
            img = a["images"].numpy()
            np.testing.assert_allclose(img, ref, atol=1e-5)
            self.assertFalse(np.allclose(img, s.numpy(), atol=1e-2))   # really rendered
            self.assertFalse(np.allclose(img, batch[0].numpy(), atol=1e-3))
            for k in ("T_render", "T_target", "A_star", "P_star", "pair_i", "pair_j", "flip_axes"):
                np.testing.assert_array_equal(a[k].numpy(), b[k].numpy(), err_msg=k)

    def test_cpu_device_equals_default(self):
        cpu = dataclasses.replace(ON, device="cpu")
        with tf.device("/cpu:0"):
            batch, fine, _ = linear_batch("affine", seed=5)
            fine_t = tf.constant(fine)
            a = pp.LinearPostprocess("affine", LEVEL, seed=3, synth=ON)(
                *batch, flip_axes=(True, False, False), labels=fine_t, synth_override=True)
            b = pp.LinearPostprocess("affine", LEVEL, seed=3, synth=cpu)(
                *batch, flip_axes=(True, False, False), labels=fine_t, synth_override=True)
            images, _, dfine = make_group(5)
            c = pp.DeformPostprocess(LEVEL, seed=3, synth=ON)(tf.constant(images), tf.constant(dfine),
                                                              synth_override=True)
            d = pp.DeformPostprocess(LEVEL, seed=3, synth=cpu)(tf.constant(images), tf.constant(dfine),
                                                               synth_override=True)
        np.testing.assert_allclose(a["images"].numpy(), b["images"].numpy(), atol=1e-6)
        np.testing.assert_allclose(c["images"].numpy(), d["images"].numpy(), atol=1e-6)
        # And traced: the cpu variant builds and runs inside a tf.function too.
        post = pp.LinearPostprocess("affine", LEVEL, seed=3, synth=cpu)
        fn = tf.function(lambda x, t, lab: post(x, t, labels=lab))
        out = fn(batch[0], batch[1], fine_t)
        self.assertTrue(np.all(np.isfinite(out["images"].numpy())))
        gd = fn.get_concrete_function(batch[0], batch[1], fine_t).graph.as_graph_def()
        devs = [nd.device for f in gd.library.function for nd in f.node_def]
        self.assertTrue([d for d in devs if "CPU:0" in d.upper()], "true branch not placed on CPU")

    def test_trace_once_drawn_flag(self):
        cfg = P.SynthConfig(prob=0.5)
        img_spec = tf.TensorSpec([None] + list(SHAPE) + [1], tf.float32)
        lab_spec = tf.TensorSpec([None] + list(SHAPE) + [1], tf.uint8)
        mat_spec = tf.TensorSpec([None, 3, 4], tf.float32)
        batch, fine, _ = linear_batch("rigid", seed=6)
        post = pp.LinearPostprocess("rigid", LEVEL, seed=TRACE_SEED, synth=cfg)
        traces = []

        @tf.function(input_signature=[img_spec, mat_spec, mat_spec, lab_spec])
        def step(x, t, rt, lab):
            traces.append(1)
            return post(x, t, rt, labels=lab)

        seen = []
        for _ in range(8):
            out = step(batch[0], batch[1], batch[2], tf.constant(fine))
            seen.append(bool(out["synth"].numpy()))
            img = out["images"].numpy()
            self.assertTrue(np.all(np.isfinite(img)))
            if not seen[-1]:
                flags = out["flip_axes"].numpy()
                np.testing.assert_array_equal(img, tfm.flip_images(batch[0], flags).numpy())
        self.assertEqual(len(traces), 1)
        self.assertEqual(set(seen), {True, False}, seen)

        images, labs, dfine = make_group(6)
        dpost = pp.DeformPostprocess(LEVEL, seed=TRACE_SEED, synth=cfg)
        dtraces = []

        @tf.function(input_signature=[img_spec, lab_spec])
        def dstep(x, lab):
            dtraces.append(1)
            return dpost(x, lab)

        dseen = []
        for _ in range(8):
            out = dstep(tf.constant(images), tf.constant(dfine))
            dseen.append(bool(out["synth"].numpy()))
            flags = out["flip_axes"].numpy()
            np.testing.assert_array_equal(out["labels"].numpy(),
                                          tfm.flip_images(labs, flags).numpy()[..., 0].astype(np.int32))
        self.assertEqual(len(dtraces), 1)
        self.assertEqual(set(dseen), {True, False}, dseen)


class TestNoFineOneHot(unittest.TestCase):
    """OD6 / design 7.1: synthesis adds no OneHot (full graph-def incl. nested functions)."""

    def setUp(self):
        self.batch, self.fine, _ = linear_batch("rigid", seed=7)
        self.images, self.labs, self.dfine = make_group(7)

    def _depths(self, aug, cfg):
        lin = pp.LinearPostprocess("rigid", LEVEL, seed=1, intensity_aug=aug, synth=cfg)
        dpost = pp.DeformPostprocess(LEVEL, seed=1, intensity_aug=aug, synth=cfg)
        if cfg is None:
            lfn = tf.function(lambda x, t, rt: lin(x, t, rt))
            largs = tuple(self.batch)
            dargs = (tf.constant(self.images), tf.constant(self.labs))
        else:
            lfn = tf.function(lambda x, t, rt, lab: lin(x, t, rt, labels=lab))
            largs = tuple(self.batch) + (tf.constant(self.fine),)
            dargs = (tf.constant(self.images), tf.constant(self.dfine))
        dfn = tf.function(lambda x, y: dpost(x, y))
        return onehot_depths(lfn, *largs), onehot_depths(dfn, *dargs)

    def test_multiset_on_equals_off(self):
        cfg = P.SynthConfig(prob=0.5)
        tent = P.IntensityAugConfig(low_res_method="tent")
        for aug in (None, tent):
            l_off, d_off = self._depths(aug, None)
            l_on, d_on = self._depths(aug, cfg)
            self.assertEqual(l_on, l_off, aug)
            self.assertEqual(d_on, d_off, aug)
            self.assertEqual(d_off[P.N_TRAIN_LABELS], 1)
            if aug is not None:   # the tent low-res one-hots are seen inside nested functions
                for n in SHAPE:
                    self.assertGreater(l_off[n], 0, n)
                    self.assertGreater(d_off[n], 0, n)

    def test_negative_control(self):
        """A synthesis variant that one-hots the fine labels inside tf.cond / map_fn makes the
        comparison fail (the scan sees nested functions)."""
        orig = synth.synthesize_group

        def leaky(labels_u8, prm, cfg, level):
            out = orig(labels_u8, prm, cfg, level)
            lab = tf.convert_to_tensor(labels_u8)[..., 0]
            extra = tf.map_fn(lambda l: tf.reduce_sum(tf.one_hot(tf.cast(l, tf.int32), 103)), lab,
                              fn_output_signature=tf.float32)
            return out + 0.0 * tf.reshape(extra, [-1, 1, 1, 1, 1])

        cfg = P.SynthConfig(prob=0.5)
        l_off, d_off = self._depths(None, None)
        with mock.patch.object(synth, "synthesize_group", leaky):
            l_on, d_on = self._depths(None, cfg)
        self.assertNotEqual(l_on, l_off)
        self.assertNotEqual(d_on, d_off)
        self.assertEqual(l_on[103], 1)
        self.assertEqual(d_on[103], 1)


class TestFineLabelHelper(unittest.TestCase):
    def test_fine_labels_map_back(self):
        rng = np.random.default_rng(0)
        _, lab = fd.synthetic_label_phantom(SHAPE, rng)
        fine = rh.fine_labels_from_train18(lab, rng)
        self.assertEqual(fine.dtype, np.uint8)
        table = luts.synthfine_to_train18_table()
        np.testing.assert_array_equal(table[fine], lab)
        self.assertGreaterEqual(len(np.unique(fine[lab == 2])), 3)
        cortex = set(luts.synthfine_cortex_ids().tolist())
        self.assertTrue(set(np.unique(fine[lab == 2]).tolist()) <= cortex)

    def test_write_fake_export_fine(self):
        import csv
        import os
        import shutil
        import tempfile
        tmp = tempfile.mkdtemp(prefix="synth_export_")
        try:
            a = rh.write_fake_export(os.path.join(tmp, "a"), labels=True)
            b = rh.write_fake_export(os.path.join(tmp, "b"), labels=True, fine_labels=True)
            with open(os.path.join(b, "export_manifest.csv")) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["labels_l4"], "labels_train18_4.npy;labels_synthfine_4.npy")
            for r in rows:
                d_a = os.path.join(a, "LUMIR", r["subject_dir"])
                d_b = os.path.join(b, "LUMIR", r["subject_dir"])
                np.testing.assert_array_equal(np.load(os.path.join(d_a, "image_4.npy")),
                                              np.load(os.path.join(d_b, "image_4.npy")))
                lab = np.load(os.path.join(d_b, "labels_train18_4.npy"))
                fine = np.load(os.path.join(d_b, "labels_synthfine_4.npy"))
                np.testing.assert_array_equal(luts.synthfine_to_train18_table()[fine], lab)
            with open(os.path.join(a, "export_manifest.csv")) as f:
                self.assertEqual(list(csv.DictReader(f))[0]["labels_l4"], "labels_train18_4.npy")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
