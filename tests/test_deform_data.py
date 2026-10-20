"""Milestone 6: synthetic deformable data, the deform batch format, DeformValidationSet and the
numpy (image, label) loader (runtime.fixed_data)."""
import os
import shutil
import tempfile
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests import runtime_helpers as rh
import numpy as np
import tensorflow as tf

from data_loading import dali_loader
from data_loading import data_module as dm
from data_loading import params as P
from data_loading import postprocess as pp
from models import diffeo
from runtime import fixed_data as fd
from runtime import metrics as M

SHAPE = P.LEVEL_SHAPE[4]


def identity_pair_dice(labels, n_classes):
    n = labels.shape[0]
    return [M.mean_fg_dice(labels[i], labels[j], n_classes)
            for i in range(n) for j in range(n) if i != j]


class TestLabelPhantom(unittest.TestCase):
    def test_labels_present_and_inside_ellipsoid(self):
        for n_labels in (8, 5):
            rng = np.random.default_rng(3)
            img, lab = fd.synthetic_label_phantom(SHAPE, rng, n_labels=n_labels)
            self.assertEqual((img.shape, lab.shape), (SHAPE, SHAPE))
            self.assertEqual((img.dtype, lab.dtype), (np.float32, np.uint8))
            self.assertEqual(np.unique(lab).tolist(), list(range(n_labels + 1)))
            _, mask = fd.smooth_phantom(SHAPE, np.random.default_rng(3))
            np.testing.assert_array_equal(lab > 0, mask)
            self.assertGreaterEqual(float(img.min()), 0.0)
            self.assertLessEqual(float(img.max()), 1.0)
            counts = np.bincount(lab.reshape(-1), minlength=n_labels + 1)[1:]
            self.assertGreater(int(counts.min()), 125, counts)     # cells >= ~5 voxels across
        a = fd.synthetic_label_phantom(SHAPE, np.random.default_rng(5))
        b = fd.synthetic_label_phantom(SHAPE, np.random.default_rng(5))
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[0], b[0])

    def test_random_svf(self):
        v = fd.random_svf(SHAPE, np.random.default_rng(0), max_vox=2.0, sigma_vox=4.0)
        self.assertEqual(v.shape, SHAPE + (3,))
        self.assertEqual(v.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(v, axis=-1).max()), 2.0, places=5)
        # Smooth: neighbouring differences are small compared to the field itself.
        self.assertLess(float(np.abs(np.diff(v, axis=0)).max()), 0.5)


class TestSyntheticDeformGroup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = fd.synthetic_deform_group(level=4, n=4, seed=0)

    def test_shapes_and_batch_format(self):
        g = self.g
        self.assertEqual(g["images"].shape, (4,) + SHAPE + (1,))
        self.assertEqual(g["labels"].dtype, np.uint8)
        self.assertEqual(g["u_true"].shape, (4,) + SHAPE + (3,))
        shapes, dtypes = dali_loader.deform_output_signature(4, 4)
        self.assertEqual(len(g["batch"]), 2)
        for t, s, d in zip(g["batch"], shapes, dtypes):
            self.assertEqual(tuple(t.shape), s)
            self.assertEqual(t.dtype, tf.as_dtype(d))
        np.testing.assert_allclose(g["v_true"].mean(axis=0), 0.0, atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(g["v_true"], axis=-1).max()), 4.0, places=5)
        # The batch runs through DeformPostprocess.
        out = pp.DeformPostprocess(4, flip_prob=0.0, seed=1)(*g["batch"])
        np.testing.assert_array_equal(out["labels"].numpy(), g["labels"][..., 0].astype(np.int32))

    def test_images_and_labels_aligned(self):
        g = self.g
        u = tf.constant(g["u_true"])
        mask = np.broadcast_to((g["base_labels"] > 0)[None, ..., None], (4,) + SHAPE + (1,))
        warped_mask = diffeo.warp(tf.constant(mask.astype(np.uint8)), u, "nearest").numpy() > 0
        np.testing.assert_array_equal(g["labels"] > 0, warped_mask)
        img = diffeo.warp(tf.constant(np.broadcast_to(g["base_image"][None, ..., None],
                                                      (4,) + SHAPE + (1,))), u).numpy()
        np.testing.assert_allclose(g["images"], img, atol=1e-6)
        # Per-label mean intensity of the deformed images matches the base phantom.
        for k in range(1, 9):
            base_mean = float(g["base_image"][g["base_labels"] == k].mean())
            for i in range(4):
                self.assertAlmostEqual(float(g["images"][i][g["labels"][i] == k].mean()), base_mean,
                                       delta=0.05)

    def test_initial_pairwise_dice_non_trivial(self):
        d = identity_pair_dice(self.g["labels"][..., 0], 9)
        print("\n[m6] synthetic deform group (level 4, n 4, seed 0): identity pairwise hard Dice "
              "mean %.4f min %.4f max %.4f, max|u| %.3f vox"
              % (np.mean(d), np.min(d), np.max(d),
                 float(np.linalg.norm(self.g["u_true"], axis=-1).max())))
        self.assertGreaterEqual(float(np.mean(d)), 0.4)
        self.assertLessEqual(float(np.mean(d)), 0.85)

    def test_deterministic_and_residual_batch(self):
        g2 = fd.synthetic_deform_group(level=4, n=4, seed=0)
        np.testing.assert_array_equal(g2["labels"], self.g["labels"])
        np.testing.assert_array_equal(g2["images"], self.g["images"])
        T = np.tile(np.eye(4), (4, 1, 1))
        T[:, 0, 3] = 2.0
        batch = fd.to_deform_batch(self.g["images"], self.g["labels"][..., 0], T, level=4)
        shapes, dtypes = dali_loader.deform_output_signature(4, 4, residual_affine=True)
        self.assertEqual(tuple(batch[2].shape), shapes[2])
        self.assertEqual(batch[1].dtype, tf.uint8)
        np.testing.assert_allclose(pp.dali_to_mm(batch[2], 4).numpy(), T, atol=1e-4)


class TestDeformValidationSet(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.images = rng.uniform(size=(7,) + SHAPE + (1,)).astype(np.float32)
        self.labels = rng.integers(0, 19, size=(7,) + SHAPE + (1,)).astype(np.uint8)

    def test_build_deterministic_and_seeded(self):
        a = fd.DeformValidationSet.build(self.images, self.labels, 4, 3, seed=11)
        b = fd.DeformValidationSet.build(self.images, self.labels, 4, 3, seed=11)
        c = fd.DeformValidationSet.build(self.images, self.labels, 4, 3, seed=12)
        self.assertEqual((a.task, a.level, a.group_size, len(a.groups)), ("deform", 4, 3, 3))
        perm = np.random.default_rng([11, fd.VAL_DEFORM_SEED]).permutation(7)
        for g, (ga, gb) in enumerate(zip(a.groups, b.groups)):
            np.testing.assert_array_equal(ga["index"], gb["index"])
            np.testing.assert_array_equal(ga["index"], [perm[(g * 3 + k) % 7] for k in range(3)])
            np.testing.assert_array_equal(ga["images"], self.images[ga["index"]])
            np.testing.assert_array_equal(ga["labels"], self.labels[ga["index"]])
            self.assertEqual(ga["labels"].dtype, np.uint8)
        self.assertFalse(all(np.array_equal(x["index"], y["index"]) for x, y in zip(a.groups, c.groups)))
        self.assertEqual(len(a.subset(2).groups), 2)
        self.assertEqual(len(fd.DeformValidationSet.build(self.images, self.labels, 4, 3, n_groups=5).groups), 5)
        with self.assertRaises(ValueError):
            fd.DeformValidationSet.build(self.images, self.labels, 4, 8)
        with self.assertRaises(ValueError):
            fd.DeformValidationSet.build(self.images, self.labels[:5], 4, 3)

    def test_from_group(self):
        vs = fd.DeformValidationSet.from_group(self.images[:4], self.labels[:4, ..., 0], 4)
        self.assertEqual((vs.task, vs.group_size, len(vs.groups)), ("deform", 4, 1))
        self.assertEqual(vs.groups[0]["labels"].shape, (4,) + SHAPE + (1,))
        np.testing.assert_array_equal(vs.groups[0]["index"], np.arange(4))


class TestLoaderWithLabels(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="m6_export_")
        self.root = rh.write_fake_export(self.tmp, n_train=4, n_val=3, labels=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_aligned_images_and_labels(self):
        imgs, labs, files, lfiles = fd.load_level_images_and_labels(self.root, 4, "train", n=3, seed=1)
        self.assertEqual((imgs.shape, labs.shape), ((3,) + SHAPE + (1,), (3,) + SHAPE + (1,)))
        self.assertEqual((imgs.dtype, labs.dtype), (np.float32, np.uint8))
        _, ref_files = fd.load_level_images(self.root, 4, "train", n=3, seed=1)
        self.assertEqual(files, ref_files)                  # same seeded selection
        for k, (f, lf) in enumerate(zip(files, lfiles)):
            self.assertEqual(os.path.dirname(f), os.path.dirname(lf))
            np.testing.assert_array_equal(imgs[k, ..., 0], np.load(os.path.join(self.root, f)))
            np.testing.assert_array_equal(labs[k, ..., 0], np.load(os.path.join(self.root, lf)))
            self.assertEqual(np.unique(labs[k]).size, 9)
        # Every label region has the phantom's per-label intensity structure (image and labels
        # come from the same phantom): mean intensity inside the labels > outside.
        self.assertGreater(float(imgs[labs > 0].mean()), float(imgs[labs == 0].mean()))
        imgs, labs, files, _ = fd.load_level_images_and_labels(self.root, 4, "val")
        self.assertEqual(len(files), 3)
        data = dm.LinearDataModule(self.root, level=4, batch_size=2)
        self.assertEqual(len(data.label_files), 4)
        for lf in data.label_files:
            self.assertTrue(os.path.isfile(os.path.join(self.root, lf)))

    def test_without_labels_column(self):
        root = rh.write_fake_export(os.path.join(self.tmp, "nolabels"), n_train=2, n_val=2)
        with self.assertRaises(KeyError):
            fd.load_level_images_and_labels(root, 4, "train")
        data = dm.LinearDataModule(root, level=4, batch_size=2)
        with self.assertRaises(KeyError):
            data.label_files


if __name__ == "__main__":
    unittest.main()
