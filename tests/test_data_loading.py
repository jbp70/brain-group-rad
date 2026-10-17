"""Tests for data_loading.params, data_loading.data_module (manifest) and the DALI import guard.

DALI-dependent tests skip cleanly when nvidia.dali is not importable (local WSL envs).
"""
import csv
import math
import os
import shutil
import tempfile
import unittest

import numpy as np

from tests import tf_setup  # noqa: F401  (log level, GPU memory growth)
import tensorflow as tf

from data_loading import params as P
from data_loading import dali_loader
from data_loading import data_module as dm
from models import transforms as tfm


def _random_params(rng, ranges):
    ax = rng.normal(size=3)
    return dict(angle_deg=rng.uniform(-ranges.rotation_deg, ranges.rotation_deg), axis_dali=ax,
                t_vox_dali=rng.uniform(-ranges.translation_vox, ranges.translation_vox, 3),
                log_scale_xyz=rng.uniform(-ranges.log_scale, ranges.log_scale, 3),
                shear_xyz=rng.uniform(-ranges.shear, ranges.shear, 3))


class TestLevelsAndRanges(unittest.TestCase):
    def test_level_constants(self):
        self.assertEqual(P.LEVEL_SHAPE[4], (40, 56, 48))
        self.assertEqual(P.LEVEL_SHAPE[2], (80, 112, 96))
        self.assertEqual(P.LEVEL_SHAPE[1], (160, 224, 192))
        for k in P.LEVELS:
            self.assertEqual(P.LEVEL_SPACING[k], float(k))
            # Same physical extent at every level.
            self.assertEqual(tuple(np.multiply(P.LEVEL_SHAPE[k], k)), (160, 224, 192))
        self.assertEqual([P.level_suffix(k) for k in (1, 2, 4)], ["", "_2", "_4"])
        with self.assertRaises(ValueError):
            P.check_level(3)

    def test_translation_mm_to_voxels_per_level(self):
        for k in P.LEVELS:
            r = P.linear_ranges("rigid", k)
            self.assertEqual(r.translation_mm, 15.0)
            self.assertAlmostEqual(r.translation_vox, 15.0 / k)
            self.assertAlmostEqual(P.mm_to_voxels(15.0, k) * P.LEVEL_SPACING[k], 15.0)
        self.assertAlmostEqual(P.linear_ranges("affine", 2, translation_mm=30).translation_vox, 15.0)

    def test_default_ranges(self):
        rig, aff = P.linear_ranges("rigid", 4), P.linear_ranges("affine", 4)
        self.assertEqual((rig.rotation_deg, aff.rotation_deg), (45.0, 45.0))
        self.assertAlmostEqual(rig.log_scale, math.log(1.005))
        self.assertAlmostEqual(rig.shear, 0.0025)
        self.assertAlmostEqual(aff.log_scale, math.log(1.10))
        self.assertAlmostEqual(aff.shear, 0.05)
        self.assertEqual(P.linear_ranges("rigid", 4, rotation_deg=10).rotation_deg, 10.0)
        self.assertFalse(P.DEFAULT_DEFORM_RESIDUAL_AFFINE)
        self.assertEqual(P.DEFAULT_DEFORM_RESIDUAL_RANGES["rotation_deg"], 1.0)
        with self.assertRaises(ValueError):
            P.linear_ranges("deform", 4)
        with self.assertRaises(ValueError):
            P.linear_ranges("rigid", 4, shear=-1)

    def test_auto_initial_fill(self):
        self.assertEqual(P.auto_initial_fill(3384, 48, P.LEVEL_SHAPE[4]), 2496)
        self.assertEqual(P.auto_initial_fill(3384, 4, P.LEVEL_SHAPE[1]), 39)
        self.assertEqual(P.auto_initial_fill(100, 48, P.LEVEL_SHAPE[4]), 100)
        self.assertEqual(P.auto_initial_fill(3384, 64, P.LEVEL_SHAPE[1]), 64)  # >= batch

    def test_intensity_config(self):
        cfg = P.IntensityAugConfig()
        self.assertTrue(cfg.enabled and cfg.noise and cfg.gamma)
        self.assertEqual((cfg.noise_prob, cfg.brightness_prob, cfg.contrast_prob,
                          cfg.low_res_prob, cfg.gamma_prob), (0.1, 0.15, 0.15, 0.125, 0.3))
        self.assertFalse(P.IntensityAugConfig.off().enabled)


class TestDaliMatrixFormulas(unittest.TestCase):
    """The formulas the DALI graph evaluates, checked in numpy and against models.transforms."""

    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_scale_shear_rows(self):
        dx, dy, dz, a, b, c = 1.07, 0.93, 1.02, 0.04, -0.03, 0.05
        fwd, inv = P.scale_shear_rows(dx, dy, dz, a, b, c)
        k = np.vstack([np.asarray(fwd), [0, 0, 0, 1]])
        k_inv = np.vstack([np.asarray(inv), [0, 0, 0, 1]])
        np.testing.assert_allclose(k @ k_inv, np.eye(4), atol=1e-12)
        p = np.eye(4)[[2, 1, 0, 3]]
        k_tf = (p @ k @ p)[:3, :3]
        u = np.array([[1, a, b], [0, 1, c], [0, 0, 1]])
        np.testing.assert_allclose(k_tf, u @ np.diag([dx, dy, dz]), atol=1e-12)

    def test_dali_center(self):
        self.assertEqual(P.dali_center((40, 56, 48), 0.5), [24.0, 28.0, 20.0])
        self.assertEqual(P.dali_center((40, 56, 48), 0.0), [23.5, 27.5, 19.5])

    def test_mirror_composition_and_conversion(self):
        for level in P.LEVELS:
            shape, s = P.LEVEL_SHAPE[level], P.LEVEL_SPACING[level]
            for task in P.TASKS:
                ranges = P.linear_ranges(task, level)
                prm = _random_params(self.rng, ranges)
                m = P.linear_matrices_np(shape_xyz=shape, **prm)
                np.testing.assert_allclose(m["T"] @ m["T_inv"], np.eye(4), atol=1e-10)
                t_mm = tfm.dali_voxel_to_tf_mm(m["T"], shape, s).numpy()
                rt_mm = tfm.dali_voxel_to_tf_mm(m["Rt"], shape, s).numpy()
                # Rigid part: orthonormal, translation = reversed DALI voxel offset * spacing.
                np.testing.assert_allclose(rt_mm[:3, :3] @ rt_mm[:3, :3].T, np.eye(3), atol=1e-10)
                np.testing.assert_allclose(rt_mm[:3, 3], prm["t_vox_dali"][::-1] * s, atol=1e-9)
                # Nuisance S = Rt^-1 T: upper-triangular U D about the center, no translation.
                s_mm = np.linalg.inv(rt_mm) @ t_mm
                np.testing.assert_allclose(s_mm[:3, 3], 0.0, atol=1e-9)
                a, b, c = prm["shear_xyz"]
                d = np.exp(prm["log_scale_xyz"])
                u = np.array([[1, a, b], [0, 1, c], [0, 0, 1]])
                np.testing.assert_allclose(s_mm[:3, :3], u @ np.diag(d), atol=1e-10)

    def test_rotation_about_center_has_no_mm_translation(self):
        # Centering with dali_center(pc) and converting with the same pc is consistent.
        for pc in (0.5, 0.0):
            m = P.linear_matrices_np(33.0, [0.3, -1.0, 0.4], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                                     P.LEVEL_SHAPE[4], pixel_center=pc)
            t_mm = tfm.dali_voxel_to_tf_mm(m["T"], P.LEVEL_SHAPE[4], 4.0, pixel_center=pc).numpy()
            np.testing.assert_allclose(t_mm[:3, 3], 0.0, atol=1e-9)
        # A mismatched pc shows up as a spurious translation (what GPU check (a) detects).
        m = P.linear_matrices_np(33.0, [0.3, -1.0, 0.4], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                                 P.LEVEL_SHAPE[4], pixel_center=0.5)
        t_mm = tfm.dali_voxel_to_tf_mm(m["T"], P.LEVEL_SHAPE[4], 4.0, pixel_center=0.0).numpy()
        self.assertGreater(np.abs(t_mm[:3, 3]).max(), 0.5)


def _write_manifest(path, rows):
    cols = ["dataset", "subject_dir", "contrast", "split", "npy_dir", "image_l1", "image_l2",
            "image_l4", "shape_l4", "labels_l4", "spacing_l4", "missing_optional_labels"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _row(i, split, labels="labels_train18_4.npy;labels_lumir10_4.npy", shape="40x56x48"):
    return {"dataset": "LUMIR25", "subject_dir": "case%04d" % i, "contrast": "T1w", "split": split,
            "npy_dir": "/data/npy_v2/LUMIR25/case%04d" % i, "image_l1": "image.npy",
            "image_l2": "image_2.npy", "image_l4": "image_4.npy", "shape_l4": shape,
            "labels_l4": labels, "spacing_l4": "4x4x4"}


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="m2_manifest_")
        self.path = os.path.join(self.tmp, dm.MANIFEST_FILE)
        rows = [_row(i, "train") for i in range(5)] + [_row(i, "val") for i in range(5, 8)]
        _write_manifest(self.path, rows)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_file_lists_per_split_and_level(self):
        rows = dm.read_manifest(self.path)
        self.assertEqual(len(rows), 8)
        train = dm.manifest_files(rows, 4, "train")
        val = dm.manifest_files(rows, 4, "val")
        self.assertEqual(train, ["LUMIR25/case%04d/image_4.npy" % i for i in range(5)])
        self.assertEqual(val, ["LUMIR25/case%04d/image_4.npy" % i for i in range(5, 8)])
        self.assertEqual(dm.manifest_files(rows, 1, "val")[0], "LUMIR25/case0005/image.npy")
        self.assertEqual(dm.manifest_files(rows, 2, None)[-1], "LUMIR25/case0007/image_2.npy")
        self.assertEqual(dm.manifest_files(rows, 4, "train", kind="labels")[0],
                         "LUMIR25/case0000/labels_train18_4.npy")
        self.assertEqual(dm.manifest_files(rows, 4, "train", contrast="T2w"), [])

    def test_missing_label_and_column(self):
        rows = [_row(0, "train", labels="labels_lumir10_4.npy")]
        with self.assertRaises(ValueError):
            dm.manifest_files(rows, 4, "train", kind="labels")
        with self.assertRaises(KeyError):
            dm.manifest_files([{"dataset": "a", "subject_dir": "b", "split": "train"}], 4, "train")

    def test_data_module_without_building_pipelines(self):
        data = dm.LinearDataModule(self.tmp, level=4, batch_size=4, seed=3)
        self.assertEqual(len(data.files), 5)
        self.assertEqual(data.template_dir, os.path.join(self.tmp, "template"))
        self.assertFalse(data.deform_residual_affine)
        self.assertEqual(data.prefetch_queue_depth, 1)
        with self.assertRaises(ValueError):
            dm.LinearDataModule(self.tmp, level=4, batch_size=4, split="val")  # 3 < 4
        _write_manifest(self.path, [_row(0, "train", shape="40x56x47")] * 4)
        with self.assertRaises(ValueError):
            dm.LinearDataModule(self.tmp, level=4, batch_size=2)

    def test_output_signature(self):
        shapes, dtypes = dali_loader.linear_output_signature("rigid", 4, 48)
        self.assertEqual(shapes, ((48, 40, 56, 48, 1), (48, 3, 4), (48, 3, 4)))
        self.assertEqual(dtypes, ("float32",) * 3)
        shapes, _ = dali_loader.linear_output_signature("affine", 2, 8)
        self.assertEqual(shapes, ((8, 80, 112, 96, 1), (8, 3, 4)))

    def test_reader_kwargs_shuffle_modes(self):
        files = ["a.npy", "b.npy"]
        kw = dali_loader.numpy_reader_kwargs(files, file_root="/root", seed=3)  # default 'epoch'
        self.assertTrue(kw["shuffle_after_epoch"])
        self.assertFalse(kw["random_shuffle"])  # exclusive with shuffle_after_epoch in DALI 1.18
        self.assertNotIn("initial_fill", kw)
        self.assertFalse(kw["pad_last_batch"])
        self.assertEqual((kw["files"], kw["file_root"], kw["seed"]), (files, "/root", 3))
        kw = dali_loader.numpy_reader_kwargs(files, shuffle_mode="buffer", initial_fill=7)
        self.assertTrue(kw["random_shuffle"])
        self.assertEqual(kw["initial_fill"], 7)
        self.assertNotIn("shuffle_after_epoch", kw)
        self.assertNotIn("file_root", kw)
        for mode in dali_loader.SHUFFLE_MODES:
            kw = dali_loader.numpy_reader_kwargs(files, shuffle=False, shuffle_mode=mode)
            self.assertFalse(kw["random_shuffle"])
            self.assertNotIn("shuffle_after_epoch", kw)
            self.assertNotIn("initial_fill", kw)
        with self.assertRaises(ValueError):
            dali_loader.numpy_reader_kwargs(files, shuffle_mode="window")

    def test_memory_growth_helper(self):
        self.assertIsInstance(dm.enable_memory_growth(), list)


class TestDaliGuard(unittest.TestCase):
    @unittest.skipIf(dali_loader.DALI_AVAILABLE, "nvidia.dali is installed")
    def test_pipeline_needs_dali(self):
        with self.assertRaises(ImportError):
            dali_loader.LinearPipeline("rigid", 4, ["a.npy"], 2)

    @unittest.skipUnless(dali_loader.DALI_AVAILABLE and tf.config.list_physical_devices("GPU"),
                         "needs nvidia.dali and a GPU")
    def test_pipeline_runs_on_synthetic_files(self):
        tmp = tempfile.mkdtemp(prefix="m2_dali_")
        try:
            rng = np.random.default_rng(0)
            files = []
            for i in range(4):
                f = "img%d.npy" % i
                np.save(os.path.join(tmp, f), rng.uniform(size=P.LEVEL_SHAPE[4]).astype(np.float32))
                files.append(f)
            for mode in dali_loader.SHUFFLE_MODES:
                pipe = dali_loader.LinearPipeline("rigid", 4, files, 2, seed=1, file_root=tmp,
                                                  shuffle_mode=mode)
                pipe.build()
                for _ in range(3):  # crosses the epoch boundary (4 files, groups of 2)
                    imgs, t, rt = pipe.run()
                self.assertEqual(imgs.as_cpu().as_array().shape, (2, 40, 56, 48, 1))
                self.assertEqual(t.as_cpu().as_array().shape, (2, 3, 4))
                self.assertEqual(rt.as_cpu().as_array().shape, (2, 3, 4))
                del pipe
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
