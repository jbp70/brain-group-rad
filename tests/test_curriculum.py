"""Curriculum and global transform G (design 3.2 v3.5): runtime.curriculum, the control-file
layout (data_loading.params), the numpy mirror with G, post-processing targets with G, the
val_<task>_g sets, the CLI flags and the data-module / pipeline wiring. DALI-dependent tests skip
without nvidia.dali.
"""
import argparse
import contextlib
import math
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests.runtime_helpers import write_fake_export
import scipy.linalg
import tensorflow as tf

from data_loading import dali_loader
from data_loading import data_module as dm
from data_loading import params as P
from data_loading import postprocess as pp
from models import transforms as tfm
from runtime import args as A
from runtime import curriculum as C
from runtime import fixed_data as fd
from runtime import metrics as M

START = {"rotation_deg": 15.0, "translation_mm": 5.0, "log_scale": 0.01, "shear": 0.0}
END = {"rotation_deg": 45.0, "translation_mm": 15.0, "log_scale": 0.03, "shear": 0.02}


def quiet_parse(argv):
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        return A.parse_args(argv)


def np_barycentric_targets(t, iters=30):
    """Independent numpy reference: mu* = barycenter{T_i^-1} (scipy logm/expm), A_i* = mu*^-1 T_i^-1."""
    t_inv = np.linalg.inv(np.asarray(t, np.float64))
    mu = np.eye(4)
    for _ in range(iters):
        x = np.mean([np.real(scipy.linalg.logm(np.linalg.inv(mu) @ m)) for m in t_inv], axis=0)
        mu = mu @ scipy.linalg.expm(x)
    return np.linalg.inv(mu)[None] @ t_inv


class TestLinearCurriculum(unittest.TestCase):
    def test_endpoints_and_midpoint(self):
        c = C.LinearCurriculum(START, END, 1000)
        self.assertEqual(c.values(0), START)
        for s in (1000, 1001, 10 ** 6):
            self.assertEqual(c.values(s), END)
        mid = c.values(500)
        for k in C.RANGE_KEYS:
            self.assertAlmostEqual(mid[k], 0.5 * (START[k] + END[k]), places=12)
        self.assertAlmostEqual(c.values(250)["rotation_deg"], 22.5)
        self.assertEqual(C.LinearCurriculum(START, END, 0).values(0), END)

    def test_validation(self):
        with self.assertRaises(ValueError):
            C.LinearCurriculum(dict(START, rotation_deg=50.0), END, 10)
        with self.assertRaises(ValueError):
            C.LinearCurriculum(START, END, -1)
        with self.assertRaises(ValueError):
            C.LinearCurriculum({"rotation_deg": 1.0}, END, 10)


class TestGlobalTransformSampler(unittest.TestCase):
    def test_determinism_and_resample_interval(self):
        s = C.GlobalTransformSampler(45.0, 10.0, resample_every=5, seed=3)
        a = [s.params(i) for i in range(40)]
        b = [C.GlobalTransformSampler(45.0, 10.0, resample_every=5, seed=3).params(i)
             for i in reversed(range(40))][::-1]
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x["axis"], y["axis"])
            self.assertEqual(x["angle_deg"], y["angle_deg"])
            np.testing.assert_array_equal(x["translation_mm"], y["translation_mm"])
        for i in range(40):
            same_block = a[(i // 5) * 5]
            self.assertEqual(a[i]["angle_deg"], same_block["angle_deg"])
            self.assertEqual(a[i]["block"], i // 5)
        angles = [a[5 * k]["angle_deg"] for k in range(8)]
        self.assertEqual(len(set(angles)), 8)                     # a new G every block
        other = C.GlobalTransformSampler(45.0, 10.0, resample_every=5, seed=4).params(0)
        self.assertNotEqual(other["angle_deg"], a[0]["angle_deg"])
        k10 = C.GlobalTransformSampler(45.0, 10.0, resample_every=10, seed=3)
        self.assertEqual(k10.params(9)["angle_deg"], k10.params(0)["angle_deg"])
        self.assertNotEqual(k10.params(10)["angle_deg"], k10.params(9)["angle_deg"])

    def test_axis_and_ranges(self):
        s = C.GlobalTransformSampler(30.0, 8.0, resample_every=1, seed=0)
        ps = [s.params(i) for i in range(2000)]
        axes = np.array([p["axis"] for p in ps])
        np.testing.assert_allclose(np.linalg.norm(axes, axis=1), 1.0, atol=1e-12)
        self.assertLess(float(np.abs(axes.mean(0)).max()), 0.06)  # uniform on the sphere
        ang = np.array([p["angle_deg"] for p in ps])
        t = np.array([p["translation_mm"] for p in ps])
        self.assertLessEqual(float(np.abs(ang).max()), 30.0)
        self.assertGreater(float(np.abs(ang).max()), 29.0)
        self.assertLessEqual(float(np.abs(t).max()), 8.0)
        self.assertGreater(float(np.abs(t).max()), 7.9)
        z = C.GlobalTransformSampler(0.0, 0.0, 5, 0).params(7)
        self.assertEqual(z["angle_deg"], 0.0)
        np.testing.assert_array_equal(z["translation_mm"], np.zeros(3))
        with self.assertRaises(ValueError):
            C.GlobalTransformSampler(45.0, 10.0, resample_every=0)
        with self.assertRaises(ValueError):
            C.GlobalTransformSampler(-1.0, 10.0)

    def test_global_matrix(self):
        g = C.global_matrix_mm((0.0, 0.0, 2.0), 90.0, (1.0, 2.0, 3.0))
        np.testing.assert_allclose(g[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
        np.testing.assert_allclose(g[:3, 3], [1, 2, 3])
        self.assertAlmostEqual(float(M.rotation_angle_deg(g[None, :3, :3])[0]), 90.0, places=6)


class TestControlFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_ctl_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_layout_round_trip(self):
        self.assertEqual(P.CONTROL_SIZE, 12)
        self.assertEqual(P.CONTROL_FIELDS[0], "rot_deg")
        self.assertEqual(P.CONTROL_FIELDS[-1], "version")
        self.assertEqual([P.CONTROL_INDEX[k] for k in ("g_axis_x", "g_angle_deg", "g_tx_mm")], [4, 7, 8])
        v = P.control_vector(10.0, 3.0, 0.02, 0.01, g_axis=(0.0, 3.0, 4.0), g_angle_deg=-20.0,
                             g_translation_mm=(1.0, -2.0, 5.0), version=7)
        self.assertEqual(v.dtype, np.float32)
        self.assertEqual(v.shape, (12,))
        d = P.parse_control(v)
        self.assertAlmostEqual(d["rotation_deg"], 10.0)
        self.assertAlmostEqual(d["log_scale"], 0.02, places=7)
        np.testing.assert_allclose(d["g_axis"], [0.0, 0.6, 0.8], atol=1e-7)   # normalized
        self.assertAlmostEqual(d["g_angle_deg"], -20.0)
        np.testing.assert_allclose(d["g_translation_mm"], [1.0, -2.0, 5.0])
        self.assertEqual(d["version"], 7)
        again = P.control_vector(version=d.pop("version"), **d)
        np.testing.assert_array_equal(again, v)
        for bad in (dict(rotation_deg=-1.0), dict(translation_mm=float("nan")), dict(g_axis=(0, 0, 0)),
                    dict(g_angle_deg=float("inf"))):
            kw = dict(rotation_deg=1.0, translation_mm=1.0, log_scale=0.0, shear=0.0)
            kw.update(bad)
            with self.assertRaises(ValueError):
                P.control_vector(**kw)
        with self.assertRaises(ValueError):
            P.parse_control(np.zeros(11))

    def test_writer_atomic_version_and_noop(self):
        path = os.path.join(self.tmp, "control", "rigid.npy")
        w = C.ControlWriter(path)
        self.assertEqual(w.version, 0)
        vals = dict(START, **C.identity_global())
        self.assertTrue(w.write(vals))
        self.assertEqual(w.version, 1)
        self.assertEqual(w.read()["version"], 1)
        self.assertAlmostEqual(w.read()["rotation_deg"], 15.0)
        mtime = os.stat(path).st_mtime_ns
        self.assertFalse(w.write(dict(vals)))                      # unchanged -> no write
        self.assertEqual(w.version, 1)
        self.assertEqual(os.stat(path).st_mtime_ns, mtime)
        self.assertTrue(w.write(dict(vals, rotation_deg=16.0)))
        self.assertEqual((w.version, w.read()["version"]), (2, 2))
        self.assertAlmostEqual(w.read()["rotation_deg"], 16.0)
        self.assertEqual(os.listdir(os.path.dirname(path)), ["rigid.npy"])   # no temp files left
        # A failing write leaves the previous file intact and no temp file behind.
        with mock.patch("runtime.curriculum.np.save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                w.write(dict(vals, rotation_deg=30.0))
        self.assertEqual(w.version, 2)
        self.assertAlmostEqual(w.read()["rotation_deg"], 16.0)
        self.assertEqual(os.listdir(os.path.dirname(path)), ["rigid.npy"])
        with self.assertRaises(ValueError):
            w.write(dict(vals, rotation_deg=-1.0))

    def test_task_control(self):
        cur = C.LinearCurriculum(START, END, 100)
        smp = C.GlobalTransformSampler(45.0, 10.0, 5, seed=1)
        tc = C.TaskControl(END, cur, smp, writer=C.ControlWriter(os.path.join(self.tmp, "a.npy")))
        v0, written = tc.update(0)
        self.assertTrue(written)
        self.assertEqual(v0["rotation_deg"], 15.0)
        self.assertEqual(v0["g_angle_deg"], smp.params(0)["angle_deg"])
        self.assertTrue(tc.update(1)[1])                            # curriculum moved
        plain = C.TaskControl(END, writer=C.ControlWriter(os.path.join(self.tmp, "b.npy")))
        self.assertTrue(plain.update(0)[1])
        self.assertFalse(plain.update(1)[1])                        # constant: no rewrite
        self.assertEqual(plain.values_at(5)["g_angle_deg"], 0.0)
        self.assertEqual(plain.values_at(5)["rotation_deg"], 45.0)
        with self.assertRaises(RuntimeError):
            C.TaskControl(END).update(0)


class TestNumpyMirrorWithG(unittest.TestCase):
    def test_composition_and_conversion(self):
        rng = np.random.default_rng(0)
        shape, level = P.LEVEL_SHAPE[4], 4
        axis_tf, g_t_mm = np.array([0.3, -0.5, 0.8]), np.array([4.0, -6.0, 8.0])
        axis_d, t_d = P.control_global_dali(axis_tf, g_t_mm, level)
        np.testing.assert_allclose(t_d, [2.0, -1.5, 1.0])          # reversed, / 4 mm
        prm = dict(angle_deg=20.0, axis_dali=rng.normal(size=3), t_vox_dali=rng.uniform(-2, 2, 3),
                   log_scale_xyz=rng.uniform(-0.05, 0.05, 3), shear_xyz=rng.uniform(-0.03, 0.03, 3),
                   shape_xyz=shape)
        m0 = P.linear_matrices_np(**prm)
        np.testing.assert_allclose(m0["G"], np.eye(4), atol=1e-12)
        m = P.linear_matrices_np(g_angle_deg=30.0, g_axis_dali=axis_d, g_t_vox_dali=t_d, **prm)
        np.testing.assert_allclose(m["T"], m["G"] @ m0["T"], atol=1e-10)
        np.testing.assert_allclose(m["Rt"], m["G"] @ m0["Rt"], atol=1e-10)
        np.testing.assert_allclose(m["T"] @ m["T_inv"], np.eye(4), atol=1e-10)
        g_mm = tfm.dali_voxel_to_tf_mm(m["G"], shape, 4.0).numpy()
        np.testing.assert_allclose(g_mm[:3, 3], g_t_mm, atol=1e-9)  # about the center, TF order
        self.assertAlmostEqual(float(M.rotation_angle_deg(g_mm[None, :3, :3])[0]), 30.0, places=6)
        r = g_mm[:3, :3]
        ax = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
        self.assertAlmostEqual(abs(float(ax @ axis_tf / np.linalg.norm(ax) / np.linalg.norm(axis_tf))), 1.0, 6)


class TestPostprocessWithG(unittest.TestCase):
    def test_targets_from_g_rt_and_flips(self):
        rng = np.random.default_rng(5)
        n, level = 5, 4
        shape = P.LEVEL_SHAPE[level]
        G = C.global_matrix_mm(rng.normal(size=3), 35.0, (3.0, -4.0, 6.0))
        T0, R0 = fd.random_linear_transforms(rng, n, "rigid", {"rotation_deg": 30.0, "translation_mm": 8.0})
        T, Rt = G @ T0, G @ R0                                  # T_i = G Rt_i S_i, target G Rt_i
        imgs = np.zeros((n,) + shape + (1,), np.float32)
        for flips in ((False, False, False), (True, False, True)):
            F = tfm.flip_matrix(flips, tf.float64).numpy()
            out = pp.LinearPostprocess("rigid", level, seed=0)(*fd.to_dali_batch(imgs, T, Rt, "rigid", level),
                                                               flip_axes=flips)
            tgt_ref = F @ Rt @ F
            np.testing.assert_allclose(out["T_target"].numpy(), tgt_ref, atol=2e-4)
            np.testing.assert_allclose(out["T_render"].numpy(), F @ T @ F, atol=2e-4)
            a_ref = np_barycentric_targets(tgt_ref)
            np.testing.assert_allclose(out["A_star"].numpy(), a_ref, atol=2e-4)
            # Not the G-free targets: the shared G moves the central space (A*_G = G' A*_0 G'^-1).
            a0 = np_barycentric_targets(F @ R0 @ F)
            gf = F @ G @ F
            np.testing.assert_allclose(a_ref, gf[None] @ a0 @ np.linalg.inv(gf)[None], atol=1e-6)
            self.assertGreater(float(np.abs(a_ref - a0).max()), 0.1)
            i, j = out["pair_i"].numpy(), out["pair_j"].numpy()
            np.testing.assert_allclose(out["P_star"].numpy(), tgt_ref[j] @ np.linalg.inv(tgt_ref[i]), atol=2e-4)


class TestValidationWithG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(2)
        cls.imgs = np.stack([fd.smooth_phantom(P.LEVEL_SHAPE[4], rng)[0][..., None] for _ in range(6)])

    def test_deterministic_same_groups_and_shifted_barycenter(self):
        gt = {"rotation_deg": 45.0, "translation_mm": 10.0}
        sets = fd.build_validation_sets(self.imgs, ["rigid", "affine"], 4, 3, seed=7, global_transform=gt)
        self.assertEqual(sorted(sets), ["affine", "affine_g", "rigid", "rigid_g"])
        again = fd.ValidationSet.build(self.imgs, "rigid", 4, 3, seed=7, global_transform=gt)
        plain = fd.build_validation_sets(self.imgs, ["rigid"], 4, 3, seed=7)
        self.assertEqual(sorted(plain), ["rigid"])
        for gg, ga, g0, g00 in zip(sets["rigid_g"].groups, again.groups, sets["rigid"].groups,
                                   plain["rigid"].groups):
            np.testing.assert_array_equal(gg["images"], ga["images"])          # deterministic
            np.testing.assert_array_equal(gg["G"], ga["G"])
            np.testing.assert_array_equal(g0["T"], g00["T"])                   # base set unchanged
            self.assertNotIn("G", g0)
            np.testing.assert_array_equal(gg["index"], g0["index"])            # same groups
            np.testing.assert_allclose(np.linalg.inv(gg["G"])[None] @ gg["T"], g0["T"], atol=1e-12)
            np.testing.assert_allclose(np.linalg.inv(gg["G"])[None] @ gg["T_target"], g0["T_target"], atol=1e-12)
            ang = abs(gg["G_params"]["angle_deg"])
            self.assertLessEqual(ang, 45.0)
            self.assertLessEqual(float(np.abs(gg["G"][:3, 3]).max()), 10.0)
            # barycenter of the G set's T_target^-1 is ~G^-1 times the plain one
            mu_g = tfm.barycenter(tf.constant(np.linalg.inv(gg["T_target"])), 10).numpy()
            mu_0 = tfm.barycenter(tf.constant(np.linalg.inv(g0["T_target"])), 10).numpy()
            rot_g = float(M.rotation_angle_deg(mu_g[None, :3, :3])[0])
            rot_0 = float(M.rotation_angle_deg(mu_0[None, :3, :3])[0])
            self.assertLessEqual(abs(rot_g - ang), rot_0 + 1e-3)
            np.testing.assert_allclose(mu_g, mu_0 @ np.linalg.inv(gg["G"]), atol=1e-5)
            np.testing.assert_allclose(gg["A_star"] @ gg["T_target"],
                                       np.broadcast_to((gg["A_star"] @ gg["T_target"])[:1], (3, 4, 4)), atol=1e-8)
        angles = [abs(g["G_params"]["angle_deg"]) for g in sets["rigid_g"].groups]
        self.assertGreater(max(angles), 5.0)
        self.assertFalse(np.array_equal(sets["rigid_g"].groups[0]["G"], sets["affine_g"].groups[0]["G"]))


class TestCurriculumArgs(unittest.TestCase):
    def test_defaults_off(self):
        a = A.parse_args(["--data", "x"])
        self.assertFalse(a.curriculum)
        self.assertFalse(a.global_transform)
        self.assertFalse(A.controls_enabled(a))
        self.assertEqual(A.task_controls_from_args(a), {})
        self.assertEqual((a.curriculum_ramp_steps, a.global_rot_max_deg, a.global_trans_max_mm,
                          a.global_resample_every), (2000, 45.0, 10.0, 5))
        self.assertEqual((a.rigid_rotation_deg_start, a.rigid_translation_mm_start), (15.0, 5.0))
        self.assertEqual(a.rigid_log_scale_start, a.rigid_log_scale)       # nuisance does not ramp
        self.assertEqual(a.rigid_shear_start, a.rigid_shear)
        self.assertAlmostEqual(a.affine_log_scale_start, math.log(1.03))
        self.assertEqual(a.affine_shear_start, 0.01)
        d = A.args_to_dict(a)
        for k in ("curriculum", "curriculum_ramp_steps", "affine_shear_start", "global_transform",
                  "global_rot_max_deg", "global_trans_max_mm", "global_resample_every"):
            self.assertIn(k, d)
        # Existing commands with small end ranges still parse (start <= end only with --curriculum).
        self.assertEqual(A.parse_args(["--data", "x", "--rigid-rotation-deg", "10"]).rigid_rotation_deg, 10.0)

    def test_validation(self):
        for bad in (["--curriculum", "true", "--rigid-rotation-deg", "10"],
                    ["--curriculum", "true", "--rigid-translation-mm-start", "20"],
                    ["--curriculum", "true", "--curriculum-ramp-steps", "-1"],
                    ["--global-transform", "true", "--global-resample-every", "0"],
                    ["--global-transform", "true", "--global-rot-max-deg", "-1"],
                    ["--curriculum", "maybe"]):
            with self.assertRaises(SystemExit, msg=str(bad)):
                quiet_parse(["--data", "x"] + bad)
        with self.assertRaises(SystemExit):
            quiet_parse(["--overfit-one-group", "--overfit-synthetic", "--global-transform", "true"])
        # affine bounds are only checked when affine is trained
        a = A.parse_args(["--data", "x", "--curriculum", "true", "--affine-rotation-deg", "5"])
        self.assertTrue(a.curriculum)
        self.assertIsInstance(A.str2bool("on"), bool)
        with self.assertRaises(argparse.ArgumentTypeError):
            A.str2bool("x")

    def test_task_controls(self):
        a = A.parse_args(["--data", "x", "--task-ratio", "rigid:1,affine:1", "--curriculum", "true",
                          "--curriculum-ramp-steps", "100", "--global-transform", "true",
                          "--global-resample-every", "7"])
        ctl = A.task_controls_from_args(a)
        self.assertEqual(sorted(ctl), ["affine", "rigid"])
        r0, r100 = ctl["rigid"].values_at(0), ctl["rigid"].values_at(100)
        self.assertEqual((r0["rotation_deg"], r0["translation_mm"]), (15.0, 5.0))
        self.assertEqual((r100["rotation_deg"], r100["translation_mm"]), (45.0, 15.0))
        self.assertAlmostEqual(ctl["affine"].values_at(0)["log_scale"], math.log(1.03))
        self.assertEqual(ctl["rigid"].sampler.resample_every, 7)
        self.assertNotEqual(ctl["rigid"].values_at(0)["g_angle_deg"], ctl["affine"].values_at(0)["g_angle_deg"])
        self.assertEqual(ctl["rigid"].values_at(3)["g_angle_deg"], ctl["rigid"].values_at(0)["g_angle_deg"])
        only_g = A.task_controls_from_args(A.parse_args(["--data", "x", "--global-transform", "true"]))
        self.assertIsNone(only_g["rigid"].curriculum)
        self.assertEqual(only_g["rigid"].values_at(0)["rotation_deg"], 45.0)


class TestDataModuleControl(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_ctl_dm_")
        self.root = write_fake_export(self.tmp, n_train=3)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_off_by_default(self):
        data = dm.LinearDataModule(self.root, 4, 2)
        self.assertEqual(data.control_writers, {})
        self.assertIsNone(data.control_file("rigid"))
        self.assertFalse(any("control" in d for _, ds, _ in os.walk(self.tmp) for d in ds))
        shapes, _ = dali_loader.linear_output_signature("rigid", 4, 2)
        self.assertEqual(len(shapes), 3)

    def test_initial_files(self):
        cdir = os.path.join(self.tmp, "run", "control")
        data = dm.LinearDataModule(self.root, 4, 2, control_dir=cdir,
                                   rigid_ranges={"rotation_deg": 20.0})
        self.assertEqual(sorted(os.listdir(cdir)), ["affine.npy", "rigid.npy"])
        d = data.control_writers["rigid"].read()
        self.assertEqual(d["version"], 1)
        self.assertAlmostEqual(d["rotation_deg"], 20.0)
        self.assertEqual(d["g_angle_deg"], 0.0)
        self.assertEqual(data.control_file("rigid"), os.path.abspath(os.path.join(cdir, "rigid.npy")))
        init = {"rigid": dict(START, **C.identity_global())}
        cdir2 = os.path.join(self.tmp, "run2", "control")
        data2 = dm.LinearDataModule(self.root, 4, 2, control_dir=cdir2, control_init=init)
        self.assertEqual(os.listdir(cdir2), ["rigid.npy"])
        self.assertAlmostEqual(data2.control_writers["rigid"].read()["rotation_deg"], 15.0)

    def test_signature_and_reader_kwargs(self):
        for task, k in (("rigid", 3), ("affine", 2)):
            shapes, dtypes = dali_loader.linear_output_signature(task, 4, 6, control=True)
            self.assertEqual(len(shapes), k + 2)
            self.assertEqual(shapes[-2:], ((6, 3, 4), (6,)))
            self.assertEqual(set(dtypes), {"float32"})
            self.assertEqual(dali_loader.base_output_count(task), k)
        kw = dali_loader.control_reader_kwargs("rel/x.npy")
        self.assertEqual(kw["files"], [os.path.abspath("rel/x.npy")])
        self.assertEqual((kw["random_shuffle"], kw["shuffle_after_epoch"], kw["read_ahead"],
                          kw["dont_use_mmap"], kw["pad_last_batch"], kw["num_shards"]),
                         (False, False, False, True, True, 1))

    @unittest.skipUnless(dali_loader.DALI_AVAILABLE and tf.config.list_physical_devices("GPU"),
                         "needs nvidia.dali and a GPU")
    def test_pipeline_with_control_file(self):
        rows = dm.read_manifest(os.path.join(self.root, dm.MANIFEST_FILE))
        files = dm.manifest_files(rows, 4, "train")
        path = os.path.join(self.tmp, "control", "rigid.npy")
        w = C.ControlWriter(path)
        w.write({"rotation_deg": 10.0, "translation_mm": 2.0, "log_scale": 0.0, "shear": 0.0,
                 "g_axis": (0.0, 0.0, 1.0), "g_angle_deg": 25.0, "g_translation_mm": (2.0, 0.0, -4.0)})
        pipe = dali_loader.LinearPipeline("rigid", 4, files, 2, seed=1, file_root=self.root,
                                          control_file=path)
        pipe.build()
        out = [o.as_cpu().as_array() for o in pipe.run()]
        self.assertEqual(len(out), 5)
        self.assertEqual(out[3].shape, (2, 3, 4))
        np.testing.assert_array_equal(out[4].reshape(-1), [1.0, 1.0])
        np.testing.assert_allclose(out[3][0], out[3][1], atol=1e-6)
        del pipe


if __name__ == "__main__":
    unittest.main()
