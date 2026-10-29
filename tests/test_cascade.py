"""Milestone-8 level cascade (models/cascade.py, runtime/checkpoint.py cascade helpers, the
cascade paths of runtime/run.py).

Design `.agents/tasks/milestone-8/design.md` §16 tests T1-T9 and T16b plus the validation
subtests of §8 (T8 includes main.py-level fresh / resume runs). Models use the TINY config (`tests.runtime_helpers.TINY`) so a frozen level-4
stage plus a level-2 stage fit the 6000 MB GTX 1080 cap. Measured values are printed (they go
into the report).
"""
import json
import os
import random
import shutil
import sys
import tempfile
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests.runtime_helpers import TINY_ARGS, free_memory, synthetic_group, tiny_config
from tests.test_trainer import _OpDeterminism
import numpy as np
import scipy.linalg
import tensorflow as tf

import main as MAIN
from data_loading import params as P
from models import cascade
from models import diffeo
from models import layers as L
from models import transforms as tfm
from models.group_net import GroupRegLevelModel
from runtime import checkpoint as C
from runtime import fixed_data as fd
from runtime import losses as LS
from runtime import metrics as M
from runtime import optimizers as O
from runtime.logging import RunLogger, make_run_dir
from runtime.run import (CASCADE_SKIP_KEY, Trainer, TrainConfig, bn_recal_plan, cascade_skip_draw,
                         cascade_stage_skip_draw)
from runtime.schedulers import ScheduleConfig

SHAPE2, SHAPE4 = tuple(P.LEVEL_SHAPE[2]), tuple(P.LEVEL_SHAPE[4])
SP2, SP4 = P.LEVEL_SPACING[2], P.LEVEL_SPACING[4]


def randomize(model, seed, scale=0.05):
    """Perturb every variable (heads, BN statistics, ...) so copies are non-trivial."""
    rng = np.random.default_rng(seed)
    for v in model.weights:
        shape = tuple(v.shape)
        if "moving_variance" in v.name:
            val = rng.uniform(0.5, 1.5, shape)
        else:
            val = v.numpy() + rng.normal(0.0, scale, shape)
        v.assign(val.astype(v.dtype.as_numpy_dtype))


def save_run(root, name, cfg, level_arg=4, step=7, seed=0):
    """A randomised level-4 model of `cfg` saved as <root>/<name> (RunCheckpoint 'last' at `step`
    + config.json with args.level = `level_arg`; None = no level key). Returns (paths, model)."""
    paths = make_run_dir(root, name)
    model = GroupRegLevelModel(4, cfg).build_model(2)
    randomize(model, seed)
    opt = O.make_optimizer("adam", lr=1e-3)
    O.build_optimizer(opt, model.trainable_variables)
    st = tf.Variable(int(step), dtype=tf.int64)
    ck = C.RunCheckpoint(paths["ckpt_last"], paths["ckpt_best"], model, opt, st)
    ck.save_last()
    C.write_config(paths["config"], cfg, {} if level_arg is None else {"level": int(level_arg)})
    return paths, model


def ncc(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


def interior(x, m):
    """Crop the spatial axes 1-3 of [N, X, Y, Z, ...] by m voxels."""
    return x[:, m:-m, m:-m, m:-m]


def block_mean(vol, k=2):
    x, y, z = vol.shape
    return vol.reshape(x // k, k, y // k, k, z // k, k).mean(axis=(1, 3, 5))


def np_center(mats, iters):
    """Numpy reference of GroupCentering: mu <- mu expm(mean_i logm(mu^-1 X_i)) from mu = I,
    `iters` iterations (scipy logm/expm, float64); returns (mu^-1 X_i, mu)."""
    mats = np.asarray(mats, np.float64)
    mu = np.eye(4)
    for _ in range(int(iters)):
        x = np.mean([np.real(scipy.linalg.logm(np.linalg.inv(mu) @ m)) for m in mats], axis=0)
        mu = mu @ scipy.linalg.expm(x)
    return np.linalg.inv(mu)[None] @ mats, mu


def rotation_deg(lin):
    """Angle (deg) of the rotation factor of a 3x3 linear map (polar decomposition)."""
    u, _, vt = np.linalg.svd(lin)
    r = u @ vt
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


class CascadeCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_m8_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        tf.keras.mixed_precision.set_global_policy("float32")
        free_memory()


class TestWeightTransfer(CascadeCase):
    """T1: level-4 checkpoint -> frozen stage + level-2 copy, init record, units, S."""

    def test_a_checkpoint_to_frozen_and_level2(self):
        cfg = tiny_config(deform_decoder=True)
        paths, src = save_run(self.tmp, "src", cfg, step=7)
        ref = [v.numpy().copy() for v in src.weights]
        prefix, src_cfg, info = C.load_cascade_source(paths["ckpt_last"])
        self.assertEqual(src_cfg, cfg)
        self.assertTrue(os.path.isabs(prefix) and prefix.endswith("last-7"))
        self.assertEqual(info, {"checkpoint": prefix, "source_config": os.path.abspath(paths["config"]),
                                "source_level": 4, "source_step": 7})
        cfg2 = cfg.replace(seed=11)
        C.check_cascade_config(cfg2, src_cfg)                     # seed may differ
        frozen = cascade.FrozenStage.from_checkpoint(prefix, src_cfg)
        self.assertEqual((frozen.level, frozen.model.level), (4, 4))
        self.assertEqual(len(frozen.model.weights), len(ref))
        for a, v in zip(ref, frozen.model.weights):
            self.assertTrue(np.array_equal(a, v.numpy()), v.name)
        model = GroupRegLevelModel(2, cfg2).build_model(2)
        n = model.copy_weights_from(frozen.model)
        self.assertEqual(n, len(model.weights))
        self.assertEqual(n, len(ref))
        for a, v in zip(ref, model.weights):
            self.assertTrue(np.array_equal(a, v.numpy()), v.name)
        fp = frozen.fingerprint
        self.assertEqual(fp, C.weight_fingerprint(src))
        self.assertEqual(fp["n_vars"], len(ref))
        self.assertAlmostEqual(fp["abs_sum"], float(sum(np.abs(a.astype(np.float64)).sum() for a in ref)),
                               places=6)
        self.assertGreater(fp["sq_sum"], 0.0)
        record = C.cascade_init_record(model, info)
        want = {"mode": "full", "via": "cascade", "checkpoint": prefix,
                "source_config": os.path.abspath(paths["config"]), "source_level": 4, "source_step": 7,
                "cross_level": [4, 2],
                "restored_parts": {"encoder": len(model.encoder.weights),
                                   "linear_decoder": len(model.linear_decoder.weights), "centering": 0,
                                   "deform_decoder": len(model.deform_decoder.weights)},
                "fresh_parts": {}, "ignored_source_parts": [], "n_restored_vars": len(model.weights),
                "n_fresh_vars": 0, "config_diff": {}}
        self.assertEqual(sorted(record), sorted(want))
        for k in want:
            self.assertEqual(record[k], want[k], k)
        self.assertIsInstance(MAIN._init_weights_lines(record), str)
        # Without the deform decoder the record has no deform_decoder part.
        lin = GroupRegLevelModel(2, tiny_config()).build_model(2)
        self.assertEqual(C.cascade_init_record(lin, info)["restored_parts"],
                         {"encoder": len(lin.encoder.weights), "linear_decoder": len(lin.linear_decoder.weights),
                          "centering": 0})

    def test_b_source_checks(self):
        cfg = tiny_config(deform_decoder=True)
        paths, _ = save_run(self.tmp, "src", cfg)
        _, src_cfg, _ = C.load_cascade_source(paths["ckpt_last"])
        with self.assertRaises(ValueError) as cm:
            C.check_cascade_config(cfg.replace(dropout=0.1), src_cfg)
        self.assertIn("dropout", str(cm.exception))
        with self.assertRaises(ValueError) as cm:                  # every differing field named
            C.check_cascade_config(cfg.replace(dropout=0.1, decoder_widths=(8, 8, 8, 4, 4)), src_cfg)
        self.assertIn("['decoder_widths', 'dropout']", str(cm.exception))
        # Source run of level 2, without args.level, without config.json, unresolvable path.
        root = os.path.join(self.tmp, "src")
        for name, level in (("lvl2", 2), ("nolevel", None)):
            dst = os.path.join(self.tmp, name)
            shutil.copytree(root, dst)
            C.write_config(os.path.join(dst, "config.json"), cfg, {} if level is None else {"level": level})
            with self.assertRaises(ValueError) as cm:
                C.load_cascade_source(os.path.join(dst, "ckpt", "last"))
            self.assertIn("level %s" % level, str(cm.exception))
        dst = os.path.join(self.tmp, "nocfg")
        shutil.copytree(root, dst)
        os.remove(os.path.join(dst, "config.json"))
        with self.assertRaises(ValueError) as cm:
            C.load_cascade_source(os.path.join(dst, "ckpt", "last"))
        self.assertIn("config.json", str(cm.exception))
        with self.assertRaises(ValueError):
            C.load_cascade_source(os.path.join(self.tmp, "missing"))
        with self.assertRaises(ValueError):
            C.load_cascade_source(os.path.join(root, "ckpt", "best"))   # empty directory

    def test_c_units(self):
        # A constant level-4 field c (level-4 voxels) is the same mm displacement after upsampling.
        c = np.array([0.7, -1.3, 2.1], np.float32)
        u4 = tf.constant(np.broadcast_to(c, (2,) + SHAPE4 + (3,)).copy())
        up = diffeo.upsample_displacement(u4)
        self.assertEqual(tuple(up.shape[1:4]), SHAPE2)
        np.testing.assert_allclose(diffeo.to_mm(up, SP2).numpy(), diffeo.to_mm(u4, SP4).numpy()[:1, :1, :1, :1]
                                   + np.zeros(up.shape, np.float32), atol=1e-6)
        # The copied rigid head gives the same mm matrix at both levels, and that matrix moves a
        # world point identically through either level's voxel grid.
        cfg = tiny_config()
        m4 = GroupRegLevelModel(4, cfg).build_model(2)
        randomize(m4, 3)
        m2 = GroupRegLevelModel(2, cfg).build_model(2)
        m2.copy_weights_from(m4)
        h = tf.constant(np.random.default_rng(0).normal(size=(3, cfg.dense[-1])).astype(np.float32))
        a4 = tfm.rigid_matrix(m4.linear_decoder.rigid_head(h)).numpy()
        a2 = tfm.rigid_matrix(m2.linear_decoder.rigid_head(h)).numpy()
        np.testing.assert_array_equal(a4, a2)
        x_mm = np.array([10.0, -20.0, 5.0, 1.0])
        for shape, sp in ((SHAPE4, SP4), (SHAPE2, SP2)):
            vox = tfm.mm_to_voxel_matrix(tf.constant(a4.astype(np.float64)), shape, sp).numpy()
            p = tfm.mm_to_index_matrix(shape, sp) @ x_mm
            y = tfm.index_to_mm_matrix(shape, sp) @ np.einsum("nij,j->ni", vox, p).T
            np.testing.assert_allclose(y.T, np.einsum("nij,j->ni", a4.astype(np.float64), x_mm), atol=1e-5)

    def test_d_second_moment(self):
        # A synthetic level-2 brain mask and its mode-pooled level-4 mask (majority over 2x2x2,
        # ties -> background, i.e. `preprocessing.export.block_mode` for a binary mask).
        rng = np.random.default_rng(4)
        _, mask2 = fd.smooth_phantom(SHAPE2, rng)
        mask2 = mask2.astype(np.uint8)
        x, y, z = SHAPE2
        counts = mask2.reshape(x // 2, 2, y // 2, 2, z // 2, 2).sum(axis=(1, 3, 5))
        mask4 = (counts > 4).astype(np.uint8)
        s2 = tfm.brain_second_moment(mask2, SP2)
        s4 = tfm.brain_second_moment(mask4, SP4)
        rel = float(np.linalg.norm(s2 - s4) / np.linalg.norm(s4))
        print("\n[T1d] S rel. Frobenius diff level 2 vs mode-pooled level 4: %.5f" % rel, file=sys.stderr)
        self.assertLess(rel, 0.05)


class TestRenderConsistency(CascadeCase):
    """T2 (AC7.2): render at level 2 then pool == block-average then render at level 4."""

    def test_pooled_render_matches_level4_render(self):
        rng = np.random.default_rng(5)
        vol2, _ = fd.smooth_phantom(SHAPE2, rng)
        T, _ = fd.random_linear_transforms(rng, 3, "affine")
        r2 = fd.render_group(vol2, T, SP2)
        pooled = cascade.pool_to_coarser(tf.constant(r2))
        self.assertEqual(pooled.dtype, tf.float32)
        self.assertEqual(tuple(pooled.shape), (3,) + SHAPE4 + (1,))
        np.testing.assert_allclose(pooled.numpy()[0, ..., 0], block_mean(r2[0, ..., 0]), atol=1e-6)
        r4 = fd.render_group(block_mean(vol2), T, SP4)
        vals = [ncc(interior(pooled.numpy(), 4)[i], interior(r4, 4)[i]) for i in range(3)]
        print("\n[T2] interior NCC pooled level-2 render vs level-4 render: %s"
              % ", ".join("%.5f" % v for v in vals), file=sys.stderr)
        for v in vals:
            self.assertGreater(v, 0.98)


class TestLinearCascade(CascadeCase):
    """T3 (AC3.1-3.4): compose_linear, centering, pre-warp, pair predictions."""

    def setUp(self):
        super().setUp()
        rng = np.random.default_rng(6)
        self.cfg = tiny_config()
        self.A4, _ = fd.random_linear_transforms(rng, 5, "affine", {"rotation_deg": 20.0, "translation_mm": 10.0})
        self.R, _ = fd.random_linear_transforms(rng, 5, "affine", {"rotation_deg": 5.0, "translation_mm": 3.0,
                                                                   "log_scale": 0.02, "shear": 0.02})
        self.c64 = tfm.GroupCentering(iters=self.cfg.centering_iters, dtype="float64")

    def test_compose_matches_numpy_reference(self):
        ref, _ = np_center(self.R @ self.A4, self.cfg.centering_iters)
        out64 = cascade.compose_linear(tf.constant(self.R), tf.constant(self.A4), self.c64)
        self.assertEqual(out64.dtype, tf.float64)
        np.testing.assert_allclose(out64.numpy(), ref, atol=1e-5)
        model = GroupRegLevelModel(2, self.cfg)
        out32 = cascade.compose_linear(tf.constant(self.R, tf.float32), tf.constant(self.A4, tf.float32),
                                       model.centering)
        self.assertEqual(out32.dtype, tf.float32)
        np.testing.assert_allclose(out32.numpy(), ref, atol=1e-4)
        # The composed group is centered: its (converged) barycenter is the identity.
        _, mu = np_center(out64.numpy(), 50)
        rot, trans = rotation_deg(mu[:3, :3]), float(np.abs(mu[:3, 3]).max())
        sym = float(np.abs(np.linalg.svd(mu[:3, :3])[1] - 1.0).max())
        print("\n[T3] barycenter of the composed group: rotation %.2e deg, translation %.2e mm, "
              "scale %.2e" % (rot, trans, sym), file=sys.stderr)
        self.assertLess(rot, 1e-4)
        self.assertLess(trans, 1e-4)
        self.assertLess(sym, 1e-6)
        # R = I: Center(A4).
        eye = np.tile(np.eye(4), (5, 1, 1))
        out_i = cascade.compose_linear(tf.constant(eye), tf.constant(self.A4), self.c64).numpy()
        np.testing.assert_allclose(out_i, tfm.center_group(tf.constant(self.A4), self.cfg.centering_iters).numpy(),
                                   atol=1e-5)
        np.testing.assert_allclose(out_i, np_center(self.A4, self.cfg.centering_iters)[0], atol=1e-5)
        # Pair predictions of the composed A: A_j^-1 A_i, equal to the raw (R A4) pairwise maps.
        i_idx, j_idx = np.array([0, 1, 2, 4]), np.array([1, 3, 0, 2])
        pairs = tfm.pair_predictions(out64, i_idx, j_idx).numpy()
        a = out64.numpy()
        x = self.R @ self.A4
        for p, (i, j) in enumerate(zip(i_idx, j_idx)):
            np.testing.assert_allclose(pairs[p], np.linalg.inv(a[j]) @ a[i], atol=1e-10)
            np.testing.assert_allclose(pairs[p], np.linalg.inv(x[j]) @ x[i], atol=1e-5)

    def test_prewarp_consistency(self):
        rng = np.random.default_rng(7)
        vol, _ = fd.smooth_phantom(SHAPE2, rng)
        imgs = tf.constant(np.broadcast_to(vol[None, ..., None], (2,) + SHAPE2 + (1,)).copy())
        a4 = tf.constant(self.A4[:2], tf.float32)
        r = tf.constant(self.R[:2], tf.float32)
        twice = cascade.prewarp_linear(cascade.prewarp_linear(imgs, a4, SP2), r, SP2)
        once = cascade.prewarp_linear(imgs, tf.matmul(r, a4), SP2)
        vals = [ncc(interior(twice.numpy(), 8)[i], interior(once.numpy(), 8)[i]) for i in range(2)]
        print("\n[T3] pre-warp NCC (two warps vs one): %s" % ", ".join("%.5f" % v for v in vals), file=sys.stderr)
        for v in vals:
            self.assertGreater(v, 0.99)


class TestDeformCascade(CascadeCase):
    """T4 (AC4.1-4.4): composition of the upsampled frozen fields with the level-2 residual."""

    def setUp(self):
        super().setUp()
        rng = np.random.default_rng(8)
        v4 = np.stack([fd.random_svf(SHAPE4, rng, 2.0) for _ in range(2)])
        v2 = np.stack([fd.random_svf(SHAPE2, rng, 2.0) for _ in range(2)])
        u4, u4i = diffeo.exp_and_inverse(tf.constant(v4), 7)
        self.U, self.Ui = diffeo.upsample_displacement(u4), diffeo.upsample_displacement(u4i)
        self.u2, self.u2i = diffeo.exp_and_inverse(tf.constant(v2), 7)

    def frozen_and_model(self):
        cfg = tiny_config(deform_decoder=True)
        m4 = GroupRegLevelModel(4, cfg).build_model(2)
        randomize(m4, 9)
        m2 = GroupRegLevelModel(2, cfg).build_model(2)
        m2.copy_weights_from(m4)
        return cascade.FrozenStage.from_model(m4), m4, m2

    def test_zero_residual_and_frozen_fields(self):
        zero = tf.zeros_like(self.U)
        u, ui = diffeo.compose_cascade(self.U, self.Ui, zero, zero)
        np.testing.assert_allclose(u.numpy(), self.U.numpy(), atol=1e-6)
        np.testing.assert_allclose(ui.numpy(), self.Ui.numpy(), atol=1e-6)
        # FrozenStage.deform = upsampled frozen u, u_inv, v (no gradient); .linear = frozen A.
        frozen, m4, _ = self.frozen_and_model()
        g = fd.synthetic_deform_group(2, 2)
        x = tf.constant(g["images"])
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            prev = frozen.deform(x)
            a_prev = frozen.linear(x, "affine")
        self.assertIsNone(tape.gradient(tf.reduce_sum(prev["U"]), x))
        self.assertIsNone(tape.gradient(tf.reduce_sum(a_prev), x))
        del tape
        out = m4(cascade.pool_to_coarser(x), task="deform", training=False)
        for k, s in (("U", "u"), ("U_inv", "u_inv"), ("V", "v")):
            self.assertEqual(tuple(prev[k].shape), (2,) + SHAPE2 + (3,))
            np.testing.assert_allclose(prev[k].numpy(), diffeo.upsample_displacement(out[s]).numpy(), atol=1e-6)
        self.assertGreater(float(tf.reduce_max(tf.abs(prev["U"]))), 1e-3)
        np.testing.assert_allclose(a_prev.numpy(), m4(cascade.pool_to_coarser(x), task="affine",
                                                      training=False)["A"].numpy(), atol=1e-6)
        self.assertEqual((a_prev.dtype, tuple(a_prev.shape)), (tf.float32, (2, 4, 4)))

    def test_two_level_round_trip(self):
        # Tolerances revised from the provisional 0.05 / 0.01 at a 3-voxel margin (design §16 T4,
        # FEAT-001 findings): the composed error is the sum of the two stages' own
        # scaling-and-squaring inverse errors (upsampled level-4 pair alone ~0.066 / 0.0074, level-2
        # residual alone ~0.031 / 0.0069 level-2 voxels), and at a 3-voxel margin u_inv o u samples
        # u_inv up to max |u| ~4.7 voxels outside the interior (border clamping, max ~0.21). So the
        # check uses a margin of ceil(max |u|) + 3 voxels, absolute bounds max <= 0.1 / mean <= 0.02
        # and "the composition adds no error": <= 1.25 x (upsampled-only + residual-only error).
        u, ui = diffeo.compose_cascade(self.U, self.Ui, self.u2, self.u2i)
        margin = int(np.ceil(float(np.linalg.norm(u.numpy(), axis=-1).max()))) + 3

        def err(a, b, m):
            norm = np.linalg.norm(interior(diffeo.compose(a, b).numpy(), m), axis=-1)
            return float(norm.max()), float(norm.mean())

        rows = {}
        for m in (3, margin):
            rows[m] = {"u o u_inv": err(u, ui, m), "u_inv o u": err(ui, u, m),
                       "U o U_inv": err(self.U, self.Ui, m), "U_inv o U": err(self.Ui, self.U, m),
                       "u2 o u2_inv": err(self.u2, self.u2i, m), "u2_inv o u2": err(self.u2i, self.u2, m)}
            print("\n[T4] round trip (level-2 voxels, %d-voxel margin): %s" % (m, "; ".join(
                "%s max %.4f mean %.5f" % (k, a, b) for k, (a, b) in rows[m].items())), file=sys.stderr)
        r = rows[margin]
        for k, up, res in (("u o u_inv", "U o U_inv", "u2 o u2_inv"), ("u_inv o u", "U_inv o U", "u2_inv o u2")):
            mx, mean = r[k]
            self.assertLessEqual(mx, 0.1, k)
            self.assertLessEqual(mean, 0.02, k)
            self.assertLessEqual(mx, 1.25 * (r[up][0] + r[res][0]), k)
            self.assertLessEqual(mean, 1.25 * (r[up][1] + r[res][1]), k)

    def test_residual_centered_and_loss_on_composed_field(self):
        frozen, _, m2 = self.frozen_and_model()
        g = fd.synthetic_deform_group(2, 2)
        images = tf.constant(g["images"])
        x_in = cascade.prewarp_deform(images, self.U)
        self.assertEqual(x_in.dtype, tf.float32)
        out = m2(x_in, task="deform", training=False)
        res_mean = float(tf.reduce_max(tf.abs(tf.reduce_mean(out["v"], axis=0))))
        raw_mean = float(tf.reduce_max(tf.abs(tf.reduce_mean(out["v_raw"], axis=0))))
        print("\n[T4] residual group mean max |mean_i v_i| %.2e (raw %.2e)" % (res_mean, raw_mean), file=sys.stderr)
        self.assertLessEqual(res_mean, 1e-5)
        self.assertGreater(raw_mean, 1e-4)                    # the centering did something
        u, ui = diffeo.compose_cascade(self.U, self.Ui, out["u"], out["u_inv"])
        out_c = dict(out, u=u, u_inv=ui, u_res=out["u"])
        onehot = tf.one_hot(tf.cast(g["labels"][..., 0], tf.int32), int(g["labels"].max()) + 1)
        pi, pj = tf.constant([0, 1]), tf.constant([1, 0])
        batch = {"onehot": onehot, "pair_i": pi, "pair_j": pj, "images": images}
        kw = dict(lambda_pdice=1.0, lambda_smooth=0.5, smooth_order=1, lambda_dice_tmpl=1.0,
                  lambda_ncc_tmpl=1.0, lambda_ncc_pair=1.0, ncc_window=7)
        got = LS.deform_loss(out_c, batch, **kw)
        # Manual computation with the composed field (independent of the dict plumbing).
        uc = diffeo.compose(self.U, out["u"])
        uic = diffeo.compose(out["u_inv"], self.Ui)
        tmpl, _, template = LS.template_dice_loss(diffeo.warp(onehot, uc))
        disp = diffeo.pairwise_displacement(uc, uic, pi, pj)
        pdice, _, moved, target = LS.pair_dice_loss(onehot, uc, uic, pi, pj, disp=disp, return_moved=True)
        smooth = LS.smoothness_loss(uc, 1)
        nt, _ = LS.template_ncc_loss(diffeo.warp(images, uc), LS.foreground(template), 7)
        npl, _ = LS.pair_ncc_loss(images, disp, pi, pj, LS.foreground(moved), LS.foreground(target), 7)
        manual = tmpl + pdice + 0.5 * smooth + nt + npl
        np.testing.assert_allclose(float(got["loss"]), float(manual), rtol=1e-6, atol=1e-6)
        for k, v in (("tmpl", tmpl), ("pdice", pdice), ("smooth", smooth), ("ncc_tmpl_loss", nt),
                     ("ncc_pair_loss", npl)):
            np.testing.assert_allclose(float(got[k]), float(v), rtol=1e-6, atol=1e-7, err_msg=k)
        res_only = LS.deform_loss(out, batch, **kw)
        self.assertGreater(abs(float(res_only["loss"]) - float(got["loss"])), 1e-4)


class TestFrozenStageFactories(CascadeCase):
    """T9 factory part: every factory builds a float32 frozen model under mixed_float16 and
    restores the global policy."""

    def test_float32_under_mixed_policy(self):
        cfg = tiny_config(deform_decoder=True)
        paths, _ = save_run(self.tmp, "src", cfg)
        prefix, src_cfg, _ = C.load_cascade_source(paths["ckpt_last"])
        mp = tf.keras.mixed_precision
        mp.set_global_policy("mixed_float16")
        src16 = GroupRegLevelModel(4, cfg).build_model(2)        # built under mixed_float16
        self.assertEqual(src16.encoder.l2.compute_dtype, "float16")
        x = tf.zeros((2,) + SHAPE2 + (1,))
        factories = (("from_checkpoint", lambda: cascade.FrozenStage.from_checkpoint(prefix, src_cfg)),
                     ("from_model", lambda: cascade.FrozenStage.from_model(src16)),
                     ("from_config", lambda: cascade.FrozenStage.from_config(cfg)))
        for name, make in factories:
            with self.subTest(name):
                frozen = make()
                self.assertEqual(mp.global_policy().name, "mixed_float16")
                self.assertEqual(frozen.model.compute_dtype, "float32")
                self.assertEqual(frozen.model.encoder.l2.compute_dtype, "float32")
                self.assertEqual(frozen.model.gradient_checkpointing, "none")
                self.assertTrue(all(v.dtype == tf.float32 for v in frozen.model.weights))
                self.assertEqual(frozen.linear(x, "rigid").dtype, tf.float32)
        frozen = cascade.FrozenStage.from_model(src16)
        for a, v in zip(src16.weights, frozen.model.weights):
            self.assertTrue(np.array_equal(a.numpy(), v.numpy()), v.name)

    @staticmethod
    def _keras_layers(layer):
        return [x for x in [layer] + list(layer.submodules) if isinstance(x, tf.keras.layers.Layer)]

    def test_cascade_copy_into_float32_decoders(self):
        """fp32-decoder-width (nonfinite-fixes review NIT 3): the main.py cascade copy
        (load_cascade_source -> check_cascade_config -> FrozenStage.from_checkpoint ->
        copy_weights_from) into trainable models with --deform-decoder-dtype float32 /
        float32_fine; frozen stages are float32 everywhere, decoder included (D2)."""
        cfg = tiny_config(deform_decoder=True)
        paths, _ = save_run(self.tmp, "src", cfg)
        prefix, src_cfg, _ = C.load_cascade_source(paths["ckpt_last"])
        C.check_cascade_config(cfg, src_cfg)
        mp = tf.keras.mixed_precision
        mp.set_global_policy("mixed_float16")
        for dtype in ("float32", "float32_fine"):
            with self.subTest(deform_decoder_dtype=dtype):
                frozen = cascade.FrozenStage.from_checkpoint(prefix, src_cfg)
                self.assertEqual(mp.global_policy().name, "mixed_float16")
                for x in self._keras_layers(frozen.model):
                    self.assertEqual(x.compute_dtype, "float32", x.name)
                trainable = GroupRegLevelModel(2, cfg, deform_decoder_dtype=dtype).build_model(2)
                self.assertEqual(trainable.copy_weights_from(frozen.source_model), len(trainable.weights))
                for s, d in zip(frozen.source_model.weights, trainable.weights):
                    self.assertTrue(np.array_equal(s.numpy(), d.numpy()), d.name)
                dec = trainable.deform_decoder
                self.assertEqual(dec.d1.conv.compute_dtype, "float32")
                self.assertEqual(dec.d3.conv.compute_dtype, "float16" if dtype == "float32_fine" else "float32")
                self.assertEqual(trainable.encoder.l2.compute_dtype, "float16")
                del frozen, trainable, dec
                free_memory()
                mp.set_global_policy("mixed_float16")

    def test_chain_copy_into_float32_fine_level1(self):
        """Level-1 variant: FrozenChain (float32 level 4 + level 2) -> level-1 float32_fine copy."""
        cfg = tiny_config(deform_decoder=True)
        _, model4 = save_run(self.tmp, "src", cfg)
        model2 = GroupRegLevelModel(2, cfg).build_model(2)
        randomize(model2, 5)
        mp = tf.keras.mixed_precision
        mp.set_global_policy("mixed_float16")
        chain = cascade.FrozenChain.from_models(model4, model2)
        self.assertEqual(mp.global_policy().name, "mixed_float16")
        for m in chain.models:
            for x in self._keras_layers(m):
                self.assertEqual(x.compute_dtype, "float32", x.name)
        trainable = GroupRegLevelModel(1, cfg, deform_decoder_dtype="float32_fine").build_model(1)
        self.assertEqual(trainable.copy_weights_from(chain.source_model), len(trainable.weights))
        for s, d in zip(model2.weights, trainable.weights):
            self.assertTrue(np.array_equal(s.numpy(), d.numpy()), d.name)
        dec = trainable.deform_decoder
        self.assertIsNotNone(dec.up_d1)
        for x in self._keras_layers(dec.d1) + [dec.up_d1, dec.up_d2, dec.svf_norm_act]:
            self.assertEqual(x.compute_dtype, "float32", x.name)
        self.assertEqual(dec.d3.conv.compute_dtype, "float16")


# ============================================================ Trainer integration (FEAT-002)
N2, ND2 = 4, 2                      # level-2 linear / deform group sizes of the trainer tests
RATIO3 = {"rigid": 1.0, "affine": 1.0, "deform": 2.0}
# Level-2 deform training under the 6000 MB cap (measured on the GTX 1080, TINY, N 2): T = 7
# without checkpointing OOMs (scaling-and-squaring tape ~14 compositions of 1.7 M voxels), T = 3
# with gradient_checkpointing 'both' peaks at ~2.9 GB (~2 s per step). The synthetic label
# phantom has 8 labels + background, so the one-hot has 9 classes instead of 19.
STAGE_KW = dict(deform_decoder=True, seed=1, integration_steps=3)
STAGE_CKPT = "both"
N_CLASSES = 9


def cascade_config(**kw):
    d = dict(level=2, group_size=N2, deform_group_size=ND2, steps=100, val_every=0, log_every=1,
             task_ratio=dict(RATIO3), seed=0, n_classes=N_CLASSES,
             schedule=ScheduleConfig("constant", base_lr=1e-3, total_steps=100, warmup_steps=0))
    d.update(kw)
    return TrainConfig(**d)


def _vals(variables):
    return [v.numpy().copy() for v in variables]


def _all_equal(before, variables):
    return all(np.array_equal(a, v.numpy()) for a, v in zip(before, variables))


def _var_sums(model):
    """float64 sum of every variable, in model.weights order (names differ between models)."""
    return [float(np.sum(v.numpy().astype(np.float64))) for v in model.weights]


def bn_stats(model, parts=None):
    """[moving_mean, moving_variance] arrays of every BN layer in `parts` (default: whole model)."""
    layers = Trainer._bn_layers(parts if parts is not None else [model])
    return [v.numpy().copy() for m in layers for v in m.bn.non_trainable_variables]


def block_mode(lab, k):
    """`preprocessing.export.block_mode` (majority over k^3 blocks, ties -> smallest id); the
    tf_gpu env has no nibabel (imported by preprocessing.export), so the same algorithm is used
    when the import fails."""
    try:
        from preprocessing.export import block_mode as bm
    except ImportError:
        bm = None
    if bm is not None:
        return bm(lab, k)
    x, y, z = (int(n) // k for n in lab.shape)
    b = np.asarray(lab).reshape(x, k, y, k, z, k).transpose(0, 2, 4, 1, 3, 5).reshape(-1, k ** 3)
    counts = np.stack([(b == c).sum(axis=1) for c in range(int(b.max()) + 1)], axis=1)
    return counts.argmax(axis=1).astype(lab.dtype).reshape(x, y, z)


class _GrapplerOff:
    """Grappler's meta optimizer disabled for the functions traced inside the block (restored)."""

    def __enter__(self):
        opts = tf.config.optimizer.get_experimental_options()
        self._was = bool(opts.get("disable_meta_optimizer", False))
        tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
        return self

    def __exit__(self, *exc):
        tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": self._was})
        return False


class CountingSource:
    """Iterator over a list of batch tuples (StopIteration at the end) counting next() calls."""

    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.calls >= len(self.batches):
            raise StopIteration
        self.calls += 1
        return self.batches[self.calls - 1]


class CascadeTrainerCase(CascadeCase):
    """Level-2 TINY cascade trainers on synthetic level-2 groups (linear N 4, deform N 2)."""

    @classmethod
    def setUpClass(cls):
        cls.groups = {t: synthetic_group(2, N2, t, seed=i) for i, t in enumerate(("rigid", "affine"))}
        cls.dgroup = fd.synthetic_deform_group(2, ND2, seed=0)
        cls.batches = {"rigid": cls.groups["rigid"]["batch"], "affine": cls.groups["affine"]["batch"],
                       "deform": cls.dgroup["batch"]}

    def stages(self, seed=3, gradient_checkpointing=STAGE_CKPT):
        """(trainable level-2 model, FrozenStage): a randomised TINY level-4 model (non-trivial
        heads and BN statistics) copied into both; deterministic for a given seed."""
        cfg = tiny_config(**STAGE_KW)
        m4 = GroupRegLevelModel(4, cfg).build_model(2)
        randomize(m4, seed, 0.02)
        m2 = GroupRegLevelModel(2, cfg, gradient_checkpointing=gradient_checkpointing).build_model(2)
        m2.copy_weights_from(m4)
        return m2, cascade.FrozenStage.from_model(m4)

    def make(self, cfg=None, sources=None, val_sets=None, logger=None, amp=False, cascade_on=True,
             gradient_checkpointing=STAGE_CKPT):
        cfg = cfg or cascade_config()
        model, frozen = self.stages(gradient_checkpointing=gradient_checkpointing)
        if sources is None:
            sources = {t: fd.FixedGroupSource(b) for t, b in self.batches.items() if t in cfg.task_ratio}
        return Trainer(model, O.make_optimizer("adam", lr=1e-3, amp=amp), sources, self.groups["rigid"]["S"],
                       cfg, logger=logger, val_sets=val_sets, cascade=frozen if cascade_on else None)

    def val_sets(self, scale=1.0, deform_group=None):
        out = {}
        for t in ("rigid", "affine"):
            g = self.groups[t]
            out[t] = fd.ValidationSet.from_group(g["images"] * scale, g["T"], g["T_target"], t, 2)
        d = deform_group or self.dgroup
        out["deform"] = fd.DeformValidationSet.from_group(d["images"] * scale, d["labels"], 2)
        return out

    def logger(self, name):
        return RunLogger(make_run_dir(self.tmp, name), backend="jsonl", tensorboard=False, quiet=True)

    @staticmethod
    def records(logger):
        logger.close()
        with open(logger.metrics_path) as f:
            return [json.loads(line) for line in f]


class TestFrozenStageTraining(CascadeTrainerCase):
    """T5 (AC2.1, AC2.4): the frozen stage never trains; cascade=None keeps today's signatures."""

    def test_three_task_steps(self):
        tr = self.make()
        ref = self.make(cascade_on=False)
        frozen = tr.cascade
        fz_refs = {v.ref() for v in frozen.model.weights}
        fz_before = _vals(frozen.model.weights)
        self.assertFalse(fz_refs & {v.ref() for v in tr.optimizer.variables()})
        tv_before = _vals(tr.model.trainable_variables)
        for t in ("rigid", "affine", "deform"):
            res = tr.train_step(t, self.batches[t])
            res0 = ref.train_step(t, self.batches[t])
            self.assertTrue(res["finite"], t)
            self.assertTrue(np.isfinite(res["loss"]), t)
            self.assertEqual(res["n_vars_updated"], res0["n_vars_updated"], t)
        self.assertTrue(np.isfinite(res["diag_u_res_max_vox"]))
        self.assertGreater(res["diag_u_max_vox"], 0.0)
        self.assertNotIn("diag_u_res_max_vox", res0)
        self.assertTrue(_all_equal(fz_before, frozen.model.weights))          # incl. frozen BN statistics
        self.assertFalse(_all_equal(tv_before, tr.model.trainable_variables))
        self.assertFalse(fz_refs & {v.ref() for v in tr.optimizer.variables()})
        self.assertEqual(len(tr.optimizer.variables()), len(ref.optimizer.variables()))
        self.assertEqual(tr.trace_counts, {"rigid": 1, "affine": 1, "deform": 1})
        # Input signatures: cascade = today's + one trailing bool scalar; cascade=None unchanged.
        for t, n in (("rigid", 3), ("affine", 2), ("deform", 2)):
            sig0, sig = ref._steps[t].input_signature, tr._steps[t].input_signature
            self.assertEqual(len(sig0), n, t)
            self.assertFalse(any(s.dtype == tf.bool for s in sig0), t)
            self.assertEqual(len(sig), n + 1, t)
            self.assertEqual(list(sig[:n]), list(sig0), t)
            self.assertEqual((sig[-1].dtype, tuple(sig[-1].shape)), (tf.bool, ()), t)
        self.assertTrue(ref.train_step("rigid", self.batches["rigid"])["finite"])
        self.assertTrue(ref.train_step("rigid", self.batches["rigid"], skip=False)["finite"])
        self.assertEqual(ref.trace_counts, {"rigid": 1, "affine": 1, "deform": 1})


class TestSkip(CascadeTrainerCase):
    """T6 (AC5.2): level skipping. The identity test runs under `_OpDeterminism` with
    TF_CUDNN_USE_AUTOTUNE=0 (and Grappler off, see the test) and requires max |diff| 0
    (design-review NIT 3: only if a bitwise mismatch cannot be traced to the skip ops would the
    fallback tolerance 1e-6 relative apply, recorded in the report). The graph-op check is strict."""

    def test_draw_reproducible(self):
        draws = [cascade_skip_draw(0, s, 0.3) for s in range(4000)]
        self.assertEqual(draws, [cascade_skip_draw(0, s, 0.3) for s in range(4000)])
        self.assertAlmostEqual(np.mean(draws), 0.3, delta=0.03)
        self.assertNotEqual(draws[:200], [cascade_skip_draw(1, s, 0.3) for s in range(200)])
        self.assertFalse(any(cascade_skip_draw(0, s, 0.0) for s in range(200)))
        self.assertEqual(cascade_skip_draw(5, 17, 0.5), random.Random("cascade-skip-5-17").random() < 0.5)

    def test_off_path_is_identity(self):
        """p = 0 vs p = 0.5 fed skip=False: bitwise equal losses and variables over 3 three-task
        rounds. Grappler is off for this comparison: with it on, the mere presence of the
        SelectV2 ops changes Grappler's rewrites of the deform step (measured on the GTX 1080:
        max |var diff| 8.6e-8 after the first deform step, amplified by Adam; two p = 0 trainers
        stay bitwise equal), i.e. the mismatch is traced to the skip ops' graph, not their values."""
        with _OpDeterminism(), _GrapplerOff():
            a = self.make(cascade_config(cascade_skip_prob=0.0))
            b = self.make(cascade_config(cascade_skip_prob=0.5))
            la, lb = [], []
            for _ in range(3):
                for t in ("rigid", "affine", "deform"):
                    la.append(a.train_step(t, self.batches[t])["loss"])
                    lb.append(b.train_step(t, self.batches[t], skip=False)["loss"])
            sa, sb = _var_sums(a.model), _var_sums(b.model)
        diffs = [abs(x - y) for x, y in zip(la, lb)] + [abs(x - y) for x, y in zip(sa, sb)]
        print("\n[T6] skip off-path identity: max |diff| %.3g over %d losses and %d variable sums"
              % (max(diffs), len(la), len(sa)), file=sys.stderr)
        self.assertEqual(la, lb)
        self.assertEqual(sa, sb)

    @staticmethod
    def skip_ops(fn):
        g = fn.get_concrete_function().graph
        return [op.name for op in g.get_operations()
                if any(p.startswith(cascade.SKIP_SCOPE) for p in op.name.split("/"))]

    def test_graph_and_forced_skip(self):
        tr0 = self.make(cascade_config(cascade_skip_prob=0.0))
        tr5 = self.make(cascade_config(cascade_skip_prob=0.5))
        ref = self.make(cascade_on=False)
        for t in ("rigid", "affine", "deform"):
            self.assertEqual(self.skip_ops(tr0._steps[t]), [], t)
            self.assertTrue(self.skip_ops(tr5._steps[t]), t)
        tr0.rebuild_steps()
        tr5.rebuild_steps()
        tr5.trace_counts = {t: 0 for t in tr5.trace_counts}
        # Forced skip: the pre-warp input is the raw image (A_prev = I) and U = U_inv = 0.
        x = tf.constant(self.groups["rigid"]["images"])
        a_prev = tr5.cascade.linear(x, "rigid")
        warped = cascade.prewarp_linear(x, cascade.skip_linear(tf.constant(True), a_prev), SP2)
        np.testing.assert_allclose(warped.numpy(), x.numpy(), atol=1e-5)
        self.assertGreater(float(tf.reduce_max(tf.abs(cascade.prewarp_linear(x, a_prev, SP2) - x))), 1e-3)
        np.testing.assert_array_equal(cascade.skip_linear(tf.constant(False), a_prev).numpy(), a_prev.numpy())
        prev = tr5.cascade.deform(tf.constant(self.dgroup["images"]))
        z = cascade.skip_fields(tf.constant(True), [prev["U"], prev["U_inv"]])
        self.assertTrue(all(float(tf.reduce_max(tf.abs(f))) == 0.0 for f in z))
        np.testing.assert_array_equal(cascade.skip_fields(tf.constant(False), [prev["U"]])[0].numpy(),
                                      prev["U"].numpy())
        # A forced-skip cascade step = a no-cascade step on the same weights (first step per task),
        # and alternating skip values never retrace.
        for t in ("rigid", "affine", "deform"):
            rs = tr5.train_step(t, self.batches[t], skip=True)
            rn = ref.train_step(t, self.batches[t])
            np.testing.assert_allclose(rs["loss"], rn["loss"], rtol=1e-4, err_msg=t)
        for i in range(4):
            for t in ("rigid", "deform"):
                self.assertTrue(tr5.train_step(t, self.batches[t], skip=bool(i % 2))["finite"])
        self.assertEqual(tr5.trace_counts, {"rigid": 1, "affine": 1, "deform": 1})

    def test_logging_and_validation(self):
        vs = self.val_sets()
        a = self.make(cascade_config(cascade_skip_prob=0.0, log_every=3), val_sets=vs,
                      logger=self.logger("p0"))
        b = self.make(cascade_config(cascade_skip_prob=0.9, log_every=3), val_sets=vs,
                      logger=self.logger("p9"))
        va, vb = a.validate(), b.validate()
        self.assertEqual(sorted(va), sorted(vb))
        for k in va:
            if not k.endswith("/time_s"):
                self.assertEqual(va[k], vb[k], k)             # validation never skips
        a.fit(6)
        b.fit(6)
        ra = [r["data"] for r in self.records(a.logger)]
        rb = [r["data"] for r in self.records(b.logger)]
        self.assertFalse(any(CASCADE_SKIP_KEY in r for r in ra))
        fr = [r[CASCADE_SKIP_KEY] for r in rb if CASCADE_SKIP_KEY in r]
        self.assertEqual(len(fr), 2)
        draws = [cascade_skip_draw(0, s, 0.9) for s in range(6)]
        np.testing.assert_allclose(fr, [np.mean(draws[:3]), np.mean(draws[3:])], atol=1e-12)
        self.assertEqual([h["cascade_skip"] for h in b.history], draws)
        for recs, tr in ((ra, a), (rb, b)):
            keys = set().union(*recs)
            for t in set(h["task"] for h in tr.history):
                self.assertIn("train_%s/data_ms" % t, keys)
                self.assertIn("train_%s/compute_ms" % t, keys)
            if any(h["task"] == "deform" for h in tr.history):
                self.assertIn("train_deform/diag_u_res_max_vox", keys)


class TestCascadeValidation(CascadeTrainerCase):
    """§8 (AC7.3, AC8.1): composed validation, frozen-stage baselines (cached), NIT 1 chunk."""

    def test_baselines_and_cache(self):
        tr = self.make(val_sets=self.val_sets())
        calls = {"n": 0}
        orig = tr._deform_errors

        def counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        tr._deform_errors = counting
        v1 = tr.validate()
        first = calls["n"]
        for name in ("rigid", "affine"):
            for k in ("center_rms_mm", "pair_rms_mm", "center_rot_deg", "pair_trans_mm"):
                for pre in ("val_%s" % name, "val_%s_frozen" % name):
                    self.assertTrue(np.isfinite(v1["%s/%s" % (pre, k)]), (pre, k))
            self.assertIn("val_%s_trainmode/center_rms_mm" % name, v1)
        for k in ("center_scale_pct", "center_shear_pct", "pair_qr_rot_deg"):
            self.assertTrue(np.isfinite(v1["val_affine_frozen/%s" % k]), k)
        self.assertNotIn("val_rigid_frozen/center_scale_pct", v1)
        for k in ("template_dice", "pair_dice", "neg_jac_pct", "inv_consistency_mm", "inv_consistency_max_mm",
                  "ncc_template", "ncc_pair"):
            self.assertTrue(np.isfinite(v1["val_deform_frozen/%s" % k]), k)
        self.assertGreaterEqual(v1["val_deform/composed_mean_disp_mm"], 0.0)
        self.assertTrue(np.isfinite(v1["val_deform/composed_mean_disp_mm"]))
        self.assertNotIn("val_deform_trainmode/composed_mean_disp_mm", v1)
        # The linear baseline is exactly the frozen stage's A4 on the same group.
        g = tr.val_sets["rigid"].groups[0]
        a4 = tr.cascade.linear(tf.constant(g["images"]), "rigid").numpy().astype(np.float64)
        want = M.summarize(M.group_errors(a4, g["A_star"], g["T_target"], tr.S_np, seed=0), "val_rigid_frozen")
        for k, v in want.items():                      # traced vs eager frozen forward: float32 rounding
            self.assertAlmostEqual(v1[k], v, delta=1e-4 * max(1.0, abs(v)), msg=k)
        # Cache hit: no frozen collection or recomputation; baselines unchanged; one trace per key.
        tr.train_step("rigid", self.batches["rigid"])
        v2 = tr.validate()
        self.assertEqual(first, 4)                     # composed eval + trainmode + identity + frozen
        self.assertEqual(calls["n"] - first, 2)        # composed eval + trainmode only
        for k in v1:
            if "_frozen/" in k or k.endswith("_identity"):
                self.assertEqual(v1[k], v2[k], k)
        self.assertNotEqual(v1["val_rigid/center_rms_mm"], v2["val_rigid/center_rms_mm"])
        self.assertTrue(all(c == 1 for c in tr.val_trace_counts.values()), tr.val_trace_counts)
        self.assertEqual(len(tr._frozen_cache), 3)

    def test_val_pair_chunk(self):
        dg4 = fd.synthetic_deform_group(2, 4, seed=1)
        vs = {"deform": fd.DeformValidationSet.from_group(dg4["images"], dg4["labels"], 2)}
        tr = self.make(cascade_config(task_ratio={"deform": 1.0}, val_pair_chunk=4), val_sets=vs)
        sizes = []
        orig = diffeo.pairwise_displacement

        def spy(u, u_inv, i, j):
            sizes.append(len(i))
            return orig(u, u_inv, i, j)

        diffeo.pairwise_displacement = spy
        try:
            v = tr.validate()
            n_validate = len(sizes)
            g = vs["deform"].groups[0]
            z = np.zeros(tuple(g["images"].shape[:4]) + (3,), np.float32)
            tr._deform_errors(g["labels"], z, z, z, z, seed=0, chunk=16)
        finally:
            diffeo.pairwise_displacement = orig
        # 12 ordered pairs in chunks of 4 for the composed eval / trainmode, identity and frozen passes.
        self.assertEqual(sizes[:n_validate], [4] * 12)
        self.assertEqual(sizes[n_validate:], [12])       # an explicit chunk still wins
        self.assertTrue(np.isfinite(v["val_deform_frozen/pair_dice"]))
        self.assertTrue(np.isfinite(v["val_deform/pair_dice_identity"]))


class TestBNRecal(CascadeTrainerCase):
    """T7 (AC6.1-AC6.4, AC6.6): forward-only BN recalibration with exact averages."""

    def scaled_linear(self, task, scale):
        g = self.groups[task]
        return fd.to_dali_batch(g["images"] * scale, g["T"], g["T_target"], task, 2)

    def scaled_deform(self, scale):
        return fd.to_deform_batch(self.dgroup["images"] * scale, self.dgroup["labels"], level=2)

    def nan_linear(self, task):
        g = self.groups[task]
        imgs = g["images"].copy()
        imgs[0, 3, 4, 5, 0] = np.nan
        return fd.to_dali_batch(imgs, g["T"], g["T_target"], task, 2)

    def test_plan(self):
        self.assertEqual(bn_recal_plan(8, RATIO3), ["rigid", "affine", "deform", "rigid", "affine", "deform",
                                                     "deform", "deform"])
        p = bn_recal_plan(64, RATIO3)
        self.assertEqual([p.count(t) for t in RATIO3], [16, 16, 32])
        self.assertEqual(p[:6], ["rigid", "affine", "deform"] * 2)
        self.assertEqual(p[48:], ["deform"] * 16)
        self.assertEqual(bn_recal_plan(3, {"rigid": 1.0, "affine": 1.0, "deform": 100.0}),
                         ["rigid", "affine", "deform"])          # every task >= 1
        self.assertEqual(bn_recal_plan(5, {"rigid": 1.0}), ["rigid"] * 5)
        with self.assertRaises(ValueError):
            bn_recal_plan(2, RATIO3)

    def test_forward_only_and_momentum_restore(self):
        logger = self.logger("recal")
        tr = self.make(logger=logger)
        model, frozen = tr.model, tr.cascade
        layers = Trainer._bn_layers([model])
        layers[1].bn.momentum = 0.9                      # one layer with its own momentum
        moms = [m.bn.momentum for m in layers]
        before = [_vals(model.trainable_variables), _vals(tr.optimizer.variables()), _vals(frozen.model.weights)]
        stats0 = bn_stats(model)
        ev = tr.recalibrate_bn(8, freeze=False, val=False)
        self.assertTrue(_all_equal(before[0], model.trainable_variables))
        self.assertTrue(_all_equal(before[1], tr.optimizer.variables()))
        self.assertTrue(_all_equal(before[2], frozen.model.weights))
        self.assertEqual(int(tr.step.numpy()), 0)
        for a, b in zip(stats0, bn_stats(model)):
            self.assertFalse(np.array_equal(a, b))       # every BN moving mean / variance changed
        self.assertEqual([m.bn.momentum for m in layers], moms)
        self.assertEqual(layers[1].bn.momentum, 0.9)
        self.assertFalse(any(m.stats_frozen for m in layers))
        want = {"rigid": 2, "affine": 2, "deform": 4}
        self.assertEqual(ev["planned"], want)
        self.assertEqual(ev["batches"], want)
        self.assertEqual(ev["skipped_nonfinite"], {"rigid": 0, "affine": 0, "deform": 0})
        self.assertEqual(ev["layers_without_updates"], [])
        self.assertEqual(ev["n_layers_updated"], len(layers))
        self.assertEqual((ev["gap"], ev["freeze"], ev["num_batches"]), ({}, False, 8))
        self.assertGreaterEqual(ev["elapsed_s"], 0.0)
        recs = [r["data"] for r in self.records(logger)]
        self.assertEqual([r["event"] for r in recs if "event" in r], ["bn_recalibration"])
        self.assertFalse(any(k.startswith("bnrecal_") for r in recs for k in r))
        with self.assertRaises(RuntimeError) as cm:
            self.make(cascade_on=False).recalibrate_bn(4, freeze=False, val=False)
        self.assertIn("needs a cascade trainer", str(cm.exception))

    def test_exact_average_and_nonfinite_slots(self):
        """MANDATORY: Keras 2.10 re-reads ``momentum`` per eager call, so K batches give the exact
        mean of the per-batch statistics (per-batch statistics = a K = 1 recalibration, which
        replaces the moving values; train-mode forwards do not depend on moving statistics)."""
        cfg = cascade_config(task_ratio={"rigid": 1.0}, flip_prob=0.0)
        batches = [self.scaled_linear("rigid", s) for s in (1.0, 1.7, 0.6)]
        tr = self.make(cfg, sources={"rigid": fd.FixedGroupSource(batches[0])})
        per = []
        for b in batches:
            tr.sources["rigid"] = fd.FixedGroupSource(b)
            tr.recalibrate_bn(1, freeze=False, val=False)
            per.append(bn_stats(tr.model))
        want = [np.mean([p[i] for p in per], axis=0) for i in range(len(per[0]))]
        self.assertFalse(np.allclose(per[0][0], per[1][0]))
        src = CountingSource(batches)
        tr.sources["rigid"] = src
        ev = tr.recalibrate_bn(3, freeze=False, val=False)
        self.assertEqual((src.calls, ev["batches"]), (3, {"rigid": 3}))
        err = max(float(np.max(np.abs(g - w) / (np.abs(w) + 1e-3))) for g, w in zip(bn_stats(tr.model), want))
        print("\n[T7] exact average: max rel. error %.2e" % err, file=sys.stderr)
        for g, w in zip(bn_stats(tr.model), want):
            np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-5)
        # A non-finite batch consumes its slot (no replacement draw); the average is over the rest.
        # (First move the statistics away from the mean: a recalibration that reproduces the
        # current statistics bitwise would look like a layer without updates.)
        tr.sources["rigid"] = fd.FixedGroupSource(batches[1])
        tr.recalibrate_bn(1, freeze=False, val=False)
        src = CountingSource([batches[0], self.nan_linear("rigid"), batches[1], batches[2]])
        tr.sources["rigid"] = src
        ev = tr.recalibrate_bn(4, freeze=False, val=False)
        self.assertEqual(src.calls, 4)
        self.assertEqual((ev["batches"], ev["skipped_nonfinite"]), ({"rigid": 3}, {"rigid": 1}))
        for g, w in zip(bn_stats(tr.model), want):
            np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-5)
        tr.sources["rigid"] = CountingSource([self.nan_linear("rigid")] * 2)
        with self.assertRaises(RuntimeError) as cm:
            tr.recalibrate_bn(2, freeze=False, val=False)
        self.assertIn("no finite batch", str(cm.exception))
        tr.sources["rigid"] = CountingSource([batches[0]])
        with self.assertRaises(RuntimeError) as cm:
            tr.recalibrate_bn(2, freeze=False, val=False)
        self.assertIn("bn recalibration: source rigid exhausted", str(cm.exception))

    def test_gradient_checkpointing_both(self):
        """No double updates with a recompute_grad decoder: one BN update per layer and batch."""
        cfg = cascade_config(task_ratio={"deform": 1.0}, flip_prob=0.0)
        batches = [self.scaled_deform(s) for s in (1.0, 1.5)]
        tr = self.make(cfg, sources={"deform": fd.FixedGroupSource(batches[0])}, gradient_checkpointing="both")
        self.assertEqual(tr.model.gradient_checkpointing, "both")
        parts = tr._bn_parts("deform")
        per = []
        for b in batches:
            tr.sources["deform"] = fd.FixedGroupSource(b)
            tr.recalibrate_bn(1, freeze=False, val=False)
            per.append(bn_stats(tr.model, parts))
        want = [np.mean([p[i] for p in per], axis=0) for i in range(len(per[0]))]
        counts = {}
        orig = L.BatchNormWithSeparateAffine.call

        def counting(layer, x, training=None):
            if training and not layer.stats_frozen:
                counts[id(layer)] = counts.get(id(layer), 0) + 1
            return orig(layer, x, training=training)

        L.BatchNormWithSeparateAffine.call = counting
        try:
            tr.sources["deform"] = CountingSource(batches)
            tr.recalibrate_bn(2, freeze=False, val=False)
        finally:
            L.BatchNormWithSeparateAffine.call = orig
        layers = Trainer._bn_layers(parts)
        self.assertEqual([counts.get(id(m), 0) for m in layers], [2] * len(layers))
        for g, w in zip(bn_stats(tr.model, parts), want):
            np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-5)

    def test_freeze_retraces(self):
        tr = self.make()
        for t in ("rigid", "affine", "deform"):
            tr.train_step(t, self.batches[t])
        ev = tr.recalibrate_bn(4, freeze=True, val=False)
        self.assertTrue(ev["freeze"])
        layers = Trainer._bn_layers([tr.model])
        self.assertTrue(all(m.stats_frozen for m in layers))
        stats = bn_stats(tr.model)
        for t in ("rigid", "affine", "deform"):
            self.assertTrue(tr.train_step(t, self.batches[t])["finite"])
        self.assertEqual(tr.trace_counts, {"rigid": 2, "affine": 2, "deform": 2})
        for a, b in zip(stats, bn_stats(tr.model)):
            self.assertTrue(np.array_equal(a, b))

    def test_gap_shrinks_with_mismatched_intensity(self):
        """Statistics recalibrated on the level-2 inputs x 1, then validation and recalibration
        batches x 3 (the deliberately mismatched intensity scale): the eval/trainmode gap is
        smaller after recalibrating on the x 3 batches. Rigid only and flips off, so the post
        statistics are those of the validation group itself."""
        cfg = cascade_config(task_ratio={"rigid": 1.0}, flip_prob=0.0)
        tr = self.make(cfg, sources={"rigid": fd.FixedGroupSource(self.scaled_linear("rigid", 1.0))},
                       val_sets={"rigid": self.val_sets(scale=3.0)["rigid"]})
        tr.recalibrate_bn(2, freeze=False, val=False)
        tr.sources["rigid"] = fd.FixedGroupSource(self.scaled_linear("rigid", 3.0))
        gap = tr.recalibrate_bn(2, freeze=False, val=True)["gap"]
        print("\n[T7] eval/trainmode gap pre -> post (stats of x 1 inputs, data x 3): %s" % "; ".join(
            "%s %.4g -> %.4g" % (k, gap["gap_pre/rigid/%s" % k], gap["gap_post/rigid/%s" % k])
            for k in ("center_rms_mm", "pair_rms_mm")), file=sys.stderr)
        for k in ("center_rms_mm", "pair_rms_mm"):
            self.assertLess(gap["gap_post/rigid/%s" % k], gap["gap_pre/rigid/%s" % k], k)

    def test_validation_records(self):
        """val=True: bnrecal_pre/post records at step 0 (PNGs off), the gap table of every set and
        the logged event; freeze after the post validation."""
        cfg = cascade_config(val_png_dir=os.path.join(self.tmp, "png"))
        logger = self.logger("gap")
        tr = self.make(cfg, val_sets=self.val_sets(), logger=logger)
        ev = tr.recalibrate_bn(4, freeze=True, val=True)
        self.assertFalse(os.path.exists(cfg.val_png_dir))
        self.assertEqual(tr.cfg.val_png_dir, cfg.val_png_dir)
        self.assertTrue(all(m.stats_frozen for m in Trainer._bn_layers([tr.model])))
        gap = ev["gap"]
        sets = {"rigid": ("center_rms_mm", "pair_rms_mm"), "affine": ("center_rms_mm", "pair_rms_mm"),
                "deform": ("template_dice", "pair_dice")}
        want = sorted("gap_%s/%s/%s" % (w, s, k) for w in ("pre", "post") for s, ks in sets.items() for k in ks)
        self.assertEqual(sorted(gap), want)
        self.assertTrue(all(np.isfinite(v) and v >= 0 for v in gap.values()))
        print("\n[T7] three-task eval/trainmode gap pre -> post: %s" % "; ".join(
            "%s/%s %.4g -> %.4g" % (s, k, gap["gap_pre/%s/%s" % (s, k)], gap["gap_post/%s/%s" % (s, k)])
            for s, ks in sets.items() for k in ks), file=sys.stderr)
        recs = self.records(logger)
        pre = [r for r in recs if "bnrecal_pre/val_rigid/center_rms_mm" in r["data"]]
        post = [r for r in recs if "bnrecal_post/val_rigid/center_rms_mm" in r["data"]]
        self.assertEqual(([r["step"] for r in pre], [r["step"] for r in post]), ([0], [0]))
        for k in ("bnrecal_pre/val_rigid_trainmode/center_rms_mm", "bnrecal_pre/val_deform/template_dice",
                  "bnrecal_post/val_deform_trainmode/pair_dice", "bnrecal_post/val_affine_frozen/center_rms_mm"):
            self.assertIn(k, set().union(*[r["data"] for r in recs]), k)
        events = [r for r in recs if r["data"].get("event") == "bn_recalibration"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["data"]["gap"], gap)


class TestCascadeAMP(CascadeTrainerCase):
    """T9 step part: cascade deform and affine steps under mixed_float16."""

    def test_mixed_precision_steps(self):
        mp = tf.keras.mixed_precision
        mp.set_global_policy("mixed_float16")
        cfg = cascade_config(task_ratio={"affine": 1.0, "deform": 1.0}, amp=True)
        tr = self.make(cfg, amp=True)
        self.assertEqual(mp.global_policy().name, "mixed_float16")
        self.assertEqual(tr.model.encoder.l2.compute_dtype, "float16")
        frozen = tr.cascade
        self.assertEqual(frozen.model.compute_dtype, "float32")
        self.assertTrue(all(v.dtype == tf.float32 for v in frozen.model.weights))
        fz_before = _vals(frozen.model.weights)
        # The mm^2 linear loss overflows float16 at the LSO's initial scale 2^15: those steps are
        # skipped and the scale halves (as tests.test_trainer's AMP tests), so step until finite.
        for t in ("affine", "deform"):
            for _ in range(24):
                res = tr.train_step(t, self.batches[t])
                self.assertTrue(np.isfinite(res["loss"]), t)
                if res["finite"]:
                    break
            self.assertTrue(res["finite"], t)
        self.assertTrue(_all_equal(fz_before, frozen.model.weights))
        # The composed quantities of the step's input path are float32.
        x = tf.constant(self.groups["affine"]["images"])
        a_prev = frozen.linear(x, "affine")
        out = tr.model(cascade.prewarp_linear(x, a_prev, SP2), task="affine", training=False)
        A = cascade.compose_linear(out["A_raw"], a_prev, tr.model.centering)
        dx = tf.constant(self.dgroup["images"])
        prev = frozen.deform(dx)
        dout = tr.model(cascade.prewarp_deform(dx, prev["U"]), task="deform", training=False)
        u, ui = diffeo.compose_cascade(prev["U"], prev["U_inv"], dout["u"], dout["u_inv"])
        self.assertEqual((A.dtype, u.dtype, ui.dtype, a_prev.dtype, prev["U"].dtype), (tf.float32,) * 5)
        self.assertEqual(mp.global_policy().name, "mixed_float16")


class TestCascadeBeatsFrozen(CascadeCase):
    """T16b (AC3.5 / AC4.5 locally): a frozen level-4 stage trained on the pooled SAME group, then
    the level-2 cascade on that group: the composed result (train-mode statistics) beats the frozen
    baseline. Flips are off in both stages so that the 100-step overfit is a clean comparison."""

    STEPS = 100

    def test_composed_beats_frozen(self):
        g2 = synthetic_group(2, 4, "rigid")
        d2 = fd.synthetic_deform_group(2, 2)
        pooled = cascade.pool_to_coarser(g2["images"]).numpy()
        r4 = fd.to_dali_batch(pooled, g2["T"], g2["T_target"], "rigid", 4)
        pooled_images = cascade.pool_to_coarser(d2["images"]).numpy()
        pooled_labels = np.stack([block_mode(d2["labels"][i, ..., 0], 2) for i in range(2)])[..., None]
        d4 = fd.to_deform_batch(pooled_images, pooled_labels, level=4)
        mcfg = tiny_config(**STAGE_KW)
        sched = ScheduleConfig("constant", base_lr=1e-3, total_steps=1000, warmup_steps=0)
        common = dict(group_size=4, deform_group_size=2, steps=1000, val_every=0, log_every=0,
                      task_ratio={"rigid": 1.0, "deform": 1.0}, seed=0, flip_prob=0.0, schedule=sched,
                      n_classes=N_CLASSES)
        m4 = GroupRegLevelModel(4, mcfg).build_model(2)
        t4 = Trainer(m4, O.make_optimizer("adam", lr=1e-3), {"rigid": fd.FixedGroupSource(r4),
                                                             "deform": fd.FixedGroupSource(d4)},
                     g2["S"], TrainConfig(level=4, **common))
        for t, b in (("rigid", r4), ("deform", d4)):
            for _ in range(self.STEPS):
                self.assertTrue(t4.train_step(t, b)["finite"])
        frozen = cascade.FrozenStage.from_model(m4)
        m2 = GroupRegLevelModel(2, mcfg, gradient_checkpointing=STAGE_CKPT).build_model(2)
        m2.copy_weights_from(m4)
        vs = {"rigid": fd.ValidationSet.from_group(g2["images"], g2["T"], g2["T_target"], "rigid", 2),
              "deform": fd.DeformValidationSet.from_group(d2["images"], d2["labels"], 2)}
        t2 = Trainer(m2, O.make_optimizer("adam", lr=1e-3), {"rigid": fd.FixedGroupSource(g2["batch"]),
                                                             "deform": fd.FixedGroupSource(d2["batch"])},
                     g2["S"], TrainConfig(level=2, **common), val_sets=vs, cascade=frozen)
        v0 = t2.validate()
        for _ in range(self.STEPS):
            for t, b in (("rigid", g2["batch"]), ("deform", d2["batch"])):
                self.assertTrue(t2.train_step(t, b)["finite"])
        v = t2.validate()
        c_rms, f_rms = v["val_rigid_trainmode/center_rms_mm"], v["val_rigid_frozen/center_rms_mm"]
        c_dice, f_dice = v["val_deform_trainmode/template_dice"], v["val_deform_frozen/template_dice"]
        print("\n[T16b] rigid center RMS composed (trainmode) %.3f mm vs frozen %.3f mm (margin %.3f; "
              "step 0 composed %.3f); deform template Dice composed %.4f vs frozen %.4f (margin %.4f; "
              "identity %.4f, step 0 composed %.4f)"
              % (c_rms, f_rms, f_rms - c_rms, v0["val_rigid_trainmode/center_rms_mm"], c_dice, f_dice,
                 c_dice - f_dice, v["val_deform/template_dice_identity"], v0["val_deform_trainmode/template_dice"]),
              file=sys.stderr)
        self.assertLess(c_rms, f_rms)
        self.assertGreater(c_dice, f_dice)


# ============================================================ main.py wiring and resume (FEAT-003)
MAIN_COMMON = ["--log-backend", "jsonl", "--quiet", "true", "--seed", "0"]
PROVENANCE = ("init_weights", "cascade", "bn_recalibration")


def ckpt_bn_stats(run_dir, scope="model/"):
    """{checkpoint key: array} of the BN moving statistics under `scope` in the latest
    ckpt/last checkpoint of `run_dir` (read from the file, no model)."""
    prefix = tf.train.latest_checkpoint(os.path.join(run_dir, "ckpt", "last"))
    names = [n for n, _ in tf.train.list_variables(prefix)
             if n.startswith(scope) and ("moving_mean" in n or "moving_variance" in n)]
    return prefix, {n: tf.train.load_variable(prefix, n) for n in names}


class TestCascadeResume(CascadeTrainerCase):
    """T8 (AC2.3, AC6.1, design §9): the frozen stage in the run checkpoint, the fingerprint
    check, and main.py's fresh / resume paths (provenance carried over, BN freeze re-applied from
    the saved record, no second recalibration, the source never read on a resume)."""

    def test_checkpoint_roundtrip(self):
        cfg = cascade_config(task_ratio={"rigid": 1.0})
        tr = self.make(cfg=cfg)
        paths = make_run_dir(self.tmp, "ck")
        ck = C.RunCheckpoint(paths["ckpt_last"], paths["ckpt_best"], tr.model, tr.optimizer, tr.step,
                             extra=dict(tr.checkpoint_variables(), frozen_stage=tr.cascade.model))
        ck.save_last()
        x = tf.constant(self.groups["rigid"]["images"])
        dx = tf.constant(self.dgroup["images"])
        ref_lin, ref_def, ref_fp = tr.cascade.linear(x, "rigid").numpy(), tr.cascade.deform(dx), tr.cascade.fingerprint
        mcfg = tr.model.config
        model2 = GroupRegLevelModel(2, mcfg, gradient_checkpointing=STAGE_CKPT).build_model(2)
        frozen2 = cascade.FrozenStage.from_config(mcfg)
        self.assertNotEqual(frozen2.fingerprint, ref_fp)                          # random init
        tr2 = Trainer(model2, O.make_optimizer("adam", lr=1e-3), {"rigid": fd.FixedGroupSource(self.batches["rigid"])},
                      self.groups["rigid"]["S"], cfg, cascade=frozen2)
        ck2 = C.RunCheckpoint(paths["ckpt_last"], paths["ckpt_best"], tr2.model, tr2.optimizer, tr2.step,
                              extra=dict(tr2.checkpoint_variables(), frozen_stage=frozen2.model))
        self.assertTrue(ck2.restore_last())
        self.assertEqual(frozen2.fingerprint, ref_fp)
        MAIN._check_cascade_fingerprint(ref_fp, frozen2.fingerprint, "config.json")          # no error
        np.testing.assert_allclose(frozen2.linear(x, "rigid").numpy(), ref_lin, rtol=0, atol=1e-7)
        out2 = frozen2.deform(dx)
        for k in ("U", "U_inv", "V"):
            np.testing.assert_allclose(out2[k].numpy(), ref_def[k].numpy(), rtol=0, atol=1e-7, err_msg=k)
        for bad in (dict(ref_fp, abs_sum=ref_fp["abs_sum"] * (1 + 1e-9)), dict(ref_fp, n_vars=ref_fp["n_vars"] + 1),
                    dict(ref_fp, sq_sum=None), None):
            with self.assertRaises(RuntimeError) as cm:
                MAIN._check_cascade_fingerprint(bad, frozen2.fingerprint, "config.json")
            self.assertIn("fingerprint", str(cm.exception))
        # Relative 1e-12: a difference below the tolerance passes.
        MAIN._check_cascade_fingerprint(dict(ref_fp, abs_sum=ref_fp["abs_sum"] * (1 + 1e-14)), frozen2.fingerprint, "c")
        # config.json without cascade (or missing) -> RuntimeError naming the path.
        cfg_path = os.path.join(self.tmp, "config.json")
        with open(cfg_path, "w") as f:
            json.dump({"init_weights": None}, f)
        for p in (cfg_path, os.path.join(self.tmp, "missing.json")):
            with self.assertRaises(RuntimeError) as cm:
                MAIN._saved_cascade_config(p)
            self.assertIn("resume of a cascade run without config.json cascade: %s" % p, str(cm.exception))

    def read_config(self, d):
        with open(os.path.join(d, "config.json")) as f:
            return json.load(f)

    def events(self, d, name):
        with open(os.path.join(d, "logs", "metrics.jsonl")) as f:
            recs = [json.loads(line) for line in f]
        return [r for r in recs if isinstance(r.get("data"), dict) and r["data"].get("event") == name]

    def test_main_fresh_and_resume(self):
        res = os.path.join(self.tmp, "results")
        # Rigid only with the deform decoder built (identical ModelConfig on both levels, no level-2
        # deform step under the 6000 MB cap); TINY widths.
        common = (["--overfit-one-group", "--overfit-synthetic", "--group-size", "4", "--task-ratio", "rigid:1",
                   "--deform-decoder", "true", "--log-every", "1", "--val-every", "0", "--val-png", "false",
                   "--schedule", "constant", "--warmup-steps", "0", "--results", res] + TINY_ARGS + MAIN_COMMON)
        self.assertIn(MAIN.main(common + ["--level", "4", "--run-name", "src", "--overfit-max-steps", "3"]), (0, 1))
        free_memory()
        src = os.path.join(res, "src", "ckpt", "last")
        lvl2 = common + ["--level", "2", "--cascade-from", src, "--run-name", "c"]
        self.assertIn(MAIN.main(lvl2 + ["--bn-recal-batches", "8", "--overfit-max-steps", "2"]), (0, 1))
        free_memory()
        d = os.path.join(res, "c")
        cfg0 = self.read_config(d)
        blocks = {k: cfg0.get(k) for k in PROVENANCE}
        self.assertEqual(blocks["init_weights"]["via"], "cascade")
        self.assertEqual((blocks["cascade"]["source_level"], blocks["cascade"]["source_step"]), (4, 3))
        self.assertIsNone(blocks["cascade"]["S_rel_diff_vs_level4"])
        self.assertEqual(blocks["cascade"]["fingerprint"]["n_vars"], blocks["init_weights"]["n_restored_vars"])
        self.assertTrue(blocks["bn_recalibration"]["freeze"])
        self.assertEqual(blocks["bn_recalibration"]["batches"], {"rigid": 8})
        self.assertEqual(blocks["bn_recalibration"]["layers_without_updates"], [])
        self.assertTrue(blocks["bn_recalibration"]["gap"])                       # val=True in main.py
        self.assertEqual(len(self.events(d, "bn_recalibration")), 1)
        self.assertEqual(len(self.events(d, "init_weights")), 1)
        prefix0, bn0 = ckpt_bn_stats(d)
        self.assertTrue(bn0)
        self.assertTrue(any(n.startswith("frozen_stage/") for n, _ in tf.train.list_variables(prefix0)))
        # Move the source away: a resume must not read it.
        os.rename(os.path.join(res, "src"), os.path.join(res, "src_moved"))
        for steps, k in (("3", "0"), ("4", "8")):
            self.assertIn(MAIN.main(lvl2 + ["--bn-recal-batches", k, "--overfit-max-steps", steps]), (0, 1))
            free_memory()
            cfg = self.read_config(d)
            self.assertEqual({b: cfg.get(b) for b in PROVENANCE}, blocks, "resume to step %s" % steps)
            self.assertEqual(cfg["args"]["bn_recal_batches"], int(k))
            prefix, bn = ckpt_bn_stats(d)
            self.assertTrue(prefix.endswith("last-%s" % steps), prefix)
            # Frozen statistics re-applied from the SAVED record (training steps would change them
            # otherwise) and no second recalibration (it would replace them).
            self.assertEqual(sorted(bn), sorted(bn0))
            for n in bn0:
                self.assertTrue(np.array_equal(bn[n], bn0[n]), "%s after resume to %s" % (n, steps))
            self.assertEqual(len(self.events(d, "bn_recalibration")), 1)
            self.assertEqual(len(self.events(d, "init_weights")), 1)
        # Tampered fingerprint -> RuntimeError naming both fingerprints; config.json untouched.
        bad = json.loads(json.dumps(cfg0))
        bad["cascade"]["fingerprint"]["abs_sum"] *= 1.0 + 1e-9
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(bad, f)
        with self.assertRaises(RuntimeError) as cm:
            MAIN.main(lvl2 + ["--overfit-max-steps", "5"])
        self.assertIn("fingerprint", str(cm.exception))
        self.assertIn(repr(bad["cascade"]["fingerprint"]["abs_sum"]), str(cm.exception))
        free_memory()
        # config.json without cascade -> RuntimeError before any model is built.
        del bad["cascade"]
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(bad, f)
        with self.assertRaises(RuntimeError) as cm:
            MAIN.main(lvl2 + ["--overfit-max-steps", "5"])
        self.assertIn("resume of a cascade run without config.json cascade", str(cm.exception))

    def test_main_freeze_false_resume(self):
        """Retrain recipe (r_level2.sh BN_RECAL_FREEZE false): one recalibration on the fresh run,
        BN keeps training on the fresh run AND after a resume (no freeze re-applied, no second
        recalibration); the frozen stage's statistics never change."""
        res = os.path.join(self.tmp, "results")
        common = (["--overfit-one-group", "--overfit-synthetic", "--group-size", "4", "--task-ratio", "rigid:1",
                   "--deform-decoder", "true", "--log-every", "1", "--val-every", "0", "--val-png", "false",
                   "--schedule", "constant", "--warmup-steps", "0", "--results", res] + TINY_ARGS + MAIN_COMMON)
        self.assertIn(MAIN.main(common + ["--level", "4", "--run-name", "src", "--overfit-max-steps", "3"]), (0, 1))
        free_memory()
        src = os.path.join(res, "src", "ckpt", "last")
        lvl2 = common + ["--level", "2", "--cascade-from", src, "--run-name", "c",
                         "--bn-recal-batches", "8", "--bn-recal-freeze", "false"]
        self.assertIn(MAIN.main(lvl2 + ["--overfit-max-steps", "2"]), (0, 1))
        free_memory()
        d = os.path.join(res, "c")
        rec = self.read_config(d)["bn_recalibration"]
        self.assertIs(rec["freeze"], False)
        self.assertEqual(rec["batches"], {"rigid": 8})
        self.assertEqual(len(self.events(d, "bn_recalibration")), 1)
        prefix2, bn2 = ckpt_bn_stats(d)
        _, fz2 = ckpt_bn_stats(d, scope="frozen_stage/")
        self.assertTrue(prefix2.endswith("last-2"), prefix2)
        enc = sorted(n for n in bn2 if "encoder" in n)
        self.assertTrue(enc, sorted(bn2))
        self.assertTrue(fz2)
        self.assertIn(MAIN.main(lvl2 + ["--overfit-max-steps", "3"]), (0, 1))
        free_memory()
        self.assertEqual(self.read_config(d)["bn_recalibration"], rec)
        self.assertEqual(len(self.events(d, "bn_recalibration")), 1)
        prefix3, bn3 = ckpt_bn_stats(d)
        _, fz3 = ckpt_bn_stats(d, scope="frozen_stage/")
        self.assertTrue(prefix3.endswith("last-3"), prefix3)
        self.assertEqual(sorted(bn3), sorted(bn2))
        for n in enc:                                       # the resumed step trained the statistics
            self.assertFalse(np.array_equal(bn3[n], bn2[n]), n)
        self.assertEqual(sorted(fz3), sorted(fz2))
        for n in fz2:
            self.assertTrue(np.array_equal(fz3[n], fz2[n]), n)


# ============================================================ 3-level frozen chain (milestone 9 phase 2)
# Design `.agents/tasks/milestone-9-phase2/design-m9-phase2.md` sections 2-3, tests T-C1..T-C9 and
# T-V1. Level-1 tensors at N 2 with TINY widths; the trainable level-1 model uses the phase-2
# recipe knobs (AMP, gradient checkpointing 'full', lean sampler, split group conv). Measured on the
# GTX 1080 under the 6000 MB cap: the level-1 LINEAR train step (~1.1 GB) and the frozen chain run;
# the level-1 DEFORM train step and the traced deform validation forward OOM even with every knob,
# T 1 and 2 label classes. So the deform chain branches are checked by tracing the step / validation
# functions (no execution) plus eager checks of the chain outputs and the baseline code.
SHAPE1, SP1 = tuple(P.LEVEL_SHAPE[1]), P.LEVEL_SPACING[1]
L1_CKPT = "full"
_STANDIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu", "frozen_chain_standin.py")


def standin_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("frozen_chain_standin", _STANDIN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def chain_config(**kw):
    d = dict(level=1, group_size=2, deform_group_size=2, steps=100, val_every=0, log_every=1,
             task_ratio={"rigid": 1.0, "deform": 1.0}, seed=0, n_classes=N_CLASSES, lean_sampler=True, amp=True,
             schedule=ScheduleConfig("constant", base_lr=1e-3, total_steps=100, warmup_steps=0))
    d.update(kw)
    return TrainConfig(**d)


class _Level4Chain:
    """Chain wrapper whose linear / deform return the level-4 stage alone (skip reference, T-C6)."""

    level = 2
    n_stages = 2

    def __init__(self, chain):
        self.chain = chain

    @property
    def models(self):
        return self.chain.models

    def linear(self, x, task):
        return self.chain.linear_stages(x, task)["A4"]

    def deform(self, x):
        st = self.chain.deform_stages(x)
        return {"U": st["U4"], "U_inv": st["U4_inv"], "V": st["V4"]}


class _FakeStage:
    """Only level / linear / deform (no n_stages): the single-stage path (T-C9)."""

    level = 4

    def linear(self, x, task):
        raise AssertionError("not called")

    def deform(self, x):
        raise AssertionError("not called")


class ChainCase(CascadeCase):
    """Level-1 synthetic groups (rigid N 2, deform N 2) and TINY randomised level-4 / level-2 models."""

    @classmethod
    def setUpClass(cls):
        cls.lgroup = synthetic_group(1, 2, "rigid", seed=0)
        cls.dgroup1 = fd.synthetic_deform_group(1, 2, seed=0)
        cls.batches1 = {"rigid": cls.lgroup["batch"], "deform": cls.dgroup1["batch"]}

    def frozen_models(self, seed=3):
        """(cfg, m4, m2): a randomised level-4 model and a level-2 copy perturbed again (so the
        level-2 stage is not the level-4 one at another resolution)."""
        cfg = tiny_config(**STAGE_KW)
        m4 = GroupRegLevelModel(4, cfg).build_model(2)
        randomize(m4, seed, 0.02)
        m2 = GroupRegLevelModel(2, cfg).build_model(2)
        m2.copy_weights_from(m4)
        randomize(m2, seed + 1, 0.01)
        return cfg, m4, m2

    def chain(self, seed=3):
        _, m4, m2 = self.frozen_models(seed)
        return cascade.FrozenChain.from_models(m4, m2)

    def level1_model(self, chain):
        """Trainable level-1 copy of the chain's level-2 model under mixed_float16 (the recipe's
        AMP; float32 level-1 deform forwards at N 2 exceed the 6000 MB cap), built at N 1."""
        tf.keras.mixed_precision.set_global_policy("mixed_float16")     # restored in tearDown
        m1 = GroupRegLevelModel(1, chain.model2.config, gradient_checkpointing=L1_CKPT,
                                lean_sampler=True, split_group_conv=True).build_model(1)
        m1.copy_weights_from(chain.model2)          # = FrozenChain.source_model (the stand-in has none)
        return m1

    def make1(self, cfg=None, chain=None, val_sets=None, logger=None):
        cfg = cfg or chain_config()
        chain = chain if chain is not None else self.chain()
        sources = {t: fd.FixedGroupSource(b) for t, b in self.batches1.items() if t in cfg.task_ratio}
        return Trainer(self.level1_model(getattr(chain, "chain", chain)),
                       O.make_optimizer("adam", lr=1e-3, amp=True), sources, self.lgroup["S"], cfg,
                       logger=logger, val_sets=val_sets, cascade=chain)

    def val_sets1(self):
        g, d = self.lgroup, self.dgroup1
        return {"rigid": fd.ValidationSet.from_group(g["images"], g["T"], g["T_target"], "rigid", 1),
                "deform": fd.DeformValidationSet.from_group(d["images"], d["labels"], 1)}


class TestFrozenChain(ChainCase):
    """T-C1, T-C2, T-C3, T-C5: constants, stand-in identity, 3-level composition, from_checkpoint."""

    def test_constants(self):
        self.assertEqual(cascade.PREV_LEVEL, {2: 4, 1: 2})
        self.assertEqual(cascade.PREV_LEVEL, C.CASCADE_SOURCE_LEVEL)
        self.assertEqual(cascade.FROZEN_LEVEL, 4)
        self.assertEqual((cascade.FrozenChain.level, cascade.FrozenChain.n_stages), (2, 2))
        self.assertEqual(cascade.FrozenStage.n_stages, 1)

    def test_matches_standin(self):
        """T-C2: FrozenChain and the phase-1 stand-in with the same weights -> max |diff| 0."""
        sc = standin_module()
        cfg, m4, m2 = self.frozen_models()
        x = tf.constant(self.lgroup["images"])
        dx = tf.constant(self.dgroup1["images"])
        diffs = {}
        with _OpDeterminism():
            chain = cascade.FrozenChain.from_models(m4, m2)
            m2c = cascade._build_frozen_model(cfg, level=2)
            m2c.copy_weights_from(m2)
            stand = sc.FrozenChainStandIn(cascade.FrozenStage.from_model(m4), m2c)
            self.assertEqual(stand.n_stages, 2)
            outs = []
            for obj in (chain, stand):
                o = {"linear/A": obj.linear(x, "rigid")}
                st = obj.linear_stages(x, "rigid")
                o.update({"linear_stages/A": st["A"], "linear_stages/A4": st["A4"]})
                d = obj.deform(dx)
                o.update({"deform/%s" % k: d[k] for k in ("U", "U_inv", "V")})
                ds = obj.deform_stages(dx)
                o.update({"deform_stages/%s" % k: ds[k] for k in ("U", "U_inv", "V", "U4", "U4_inv", "V4")})
                outs.append({k: v.numpy() for k, v in o.items()})
        for k in outs[0]:
            diffs[k] = float(np.max(np.abs(outs[0][k] - outs[1][k])))
        print("\n[T-C2] FrozenChain vs stand-in max |diff|: %s" % ", ".join(
            "%s %.3g" % kv for kv in sorted(diffs.items())), file=sys.stderr)
        for k, v in diffs.items():
            self.assertEqual(v, 0.0, k)
        self.assertEqual(outs[0]["deform/U"].shape, (2,) + SHAPE1 + (3,))
        self.assertGreater(float(np.abs(outs[0]["deform/U"]).max()), 1e-3)
        self.assertGreater(float(np.abs(outs[0]["linear_stages/A"] - outs[0]["linear_stages/A4"]).max()), 1e-4)

    def test_three_level_linear_composition(self):
        """T-C3: A4 = FrozenStage.linear(pool(x)); A2 = Center2(R2_raw A4); A1 = Center1(R1_raw A2)."""
        chain = self.chain()
        x = tf.constant(self.lgroup["images"])
        st = chain.linear_stages(x, "affine")
        x2 = cascade.pool_to_coarser(x)
        a4 = chain.stage4.linear(x2, "affine").numpy()
        np.testing.assert_array_equal(st["A4"].numpy(), a4)
        np.testing.assert_array_equal(chain.linear(x, "affine").numpy(), st["A"].numpy())
        iters = chain.model2.config.centering_iters
        out2 = chain.model2(cascade.prewarp_linear(x2, tf.constant(a4), SP2), task="affine", training=False)
        r2 = out2["A_raw"].numpy().astype(np.float64)
        a2_ref, _ = np_center(r2 @ a4.astype(np.float64), iters)
        err2 = float(np.max(np.abs(st["A"].numpy() - a2_ref)))
        np.testing.assert_allclose(st["A"].numpy(), a2_ref, atol=1e-4)
        m1 = self.level1_model(chain)
        out1 = m1(cascade.prewarp_linear(x, st["A"], SP1), task="affine", training=False)
        r1 = out1["A_raw"].numpy().astype(np.float64)
        a1_ref, _ = np_center(r1 @ st["A"].numpy().astype(np.float64), iters)
        c64 = tfm.GroupCentering(iters=iters, dtype="float64")
        a1_64 = cascade.compose_linear(tf.constant(r1), tf.constant(st["A"].numpy().astype(np.float64)), c64).numpy()
        np.testing.assert_allclose(a1_64, a1_ref, rtol=1e-5, atol=1e-8)
        a1_32 = cascade.compose_linear(out1["A_raw"], st["A"], m1.centering).numpy()
        err1 = float(np.max(np.abs(a1_32 - a1_ref)))
        np.testing.assert_allclose(a1_32, a1_ref, atol=1e-4)
        print("\n[T-C3] linear chain vs numpy float64: A2 max |diff| %.2e, A1 (float32) %.2e"
              % (err2, err1), file=sys.stderr)
        # deform_stages U4 = up(up(u4)) of the level-4 model on the twice-pooled images.
        dx = tf.constant(self.dgroup1["images"])
        ds = chain.deform_stages(dx)
        out4 = chain.stage4.model(cascade.pool_to_coarser(cascade.pool_to_coarser(dx)), task="deform",
                                  training=False)
        up = diffeo.upsample_displacement
        for k, s in (("U4", "u"), ("U4_inv", "u_inv"), ("V4", "v")):
            np.testing.assert_allclose(ds[k].numpy(), up(up(out4[s])).numpy(), atol=1e-6, err_msg=k)

    def test_from_checkpoint(self):
        """T-C5: an m8-style checkpoint (model = level 2, frozen_stage = level 4) restores both;
        every failure is a ValueError."""
        cfg, m4, m2 = self.frozen_models()
        prefix = tf.train.Checkpoint(model=m2, frozen_stage=m4).save(os.path.join(self.tmp, "m8", "ck"))
        fp4 = C.weight_fingerprint(m4)
        chain = cascade.FrozenChain.from_checkpoint(prefix, cfg, fp4)
        for a, b in ((m4, chain.stage4.model), (m2, chain.model2)):
            for va, vb in zip(a.weights, b.weights):
                self.assertTrue(np.array_equal(va.numpy(), vb.numpy()), vb.name)
        self.assertEqual(chain.fingerprints, {4: fp4, 2: C.weight_fingerprint(m2)})
        self.assertEqual(chain.fingerprint, C.weight_fingerprint(m2))
        self.assertIs(chain.source_model, chain.model2)
        self.assertEqual(sorted(chain.checkpoint_extra()), ["frozen_stage_2", "frozen_stage_4"])
        self.assertEqual(chain.models, [chain.stage4.model, chain.model2])
        self.assertTrue(all(v.dtype == tf.float32 for m in chain.models for v in m.weights))
        self.assertEqual([m.gradient_checkpointing for m in chain.models], ["none", "none"])
        with self.assertRaises(ValueError) as cm:
            cascade.FrozenChain.from_checkpoint(prefix, cfg, dict(fp4, abs_sum=fp4["abs_sum"] * 2.0))
        self.assertIn("cascade.fingerprint", str(cm.exception))
        only2 = tf.train.Checkpoint(model=m2).save(os.path.join(self.tmp, "only2", "ck"))
        with self.assertRaises(ValueError) as cm:
            cascade.FrozenChain.from_checkpoint(only2, cfg, fp4)
        self.assertIn("AssertionError", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            cascade.FrozenChain.from_checkpoint(os.path.join(self.tmp, "missing", "ck-1"), cfg, fp4)
        self.assertIn("NotFoundError", str(cm.exception))
        print("\n[T-C5] missing prefix -> %s" % str(cm.exception)[:160], file=sys.stderr)
        # Constructor checks: wrong levels / different configs.
        with self.assertRaises(ValueError):
            cascade.FrozenChain(cascade.FrozenStage(chain.stage4.model), chain.stage4.model)
        other = cascade._build_frozen_model(tiny_config(**dict(STAGE_KW, integration_steps=4)), level=2)
        with self.assertRaises(ValueError) as cm:
            cascade.FrozenChain(chain.stage4, other)
        self.assertIn("integration_steps", str(cm.exception))
        # FrozenStage additions (pure).
        fs = chain.stage4
        self.assertIs(fs.source_model, fs.model)
        self.assertEqual(fs.fingerprints, {4: fs.fingerprint})
        self.assertEqual(list(fs.checkpoint_extra()), ["frozen_stage"])


class TestThreeLevelRoundTrip(CascadeCase):
    """T-C4: three-level deform composition / unrolled inverse on smooth synthetic fields at the
    real level shapes (N 1)."""

    def test_three_level_round_trip(self):
        rng = np.random.default_rng(11)
        v4 = fd.random_svf(SHAPE4, rng, 2.0)[None]
        v2 = fd.random_svf(SHAPE2, rng, 2.0)[None]
        v1 = fd.random_svf(SHAPE1, rng, 2.0)[None]
        # Lean sampler throughout (same values as the default sampler, design-m9-phase1 3.3): the
        # default sampler's level-1 index tensors exceed the 6000 MB cap.
        up = diffeo.upsample_displacement
        u4, u4i = diffeo.exp_and_inverse(tf.constant(v4), 7)
        u2, u2i = diffeo.exp_and_inverse(tf.constant(v2), 7)
        u1, u1i = diffeo.exp_and_inverse(tf.constant(v1), 7, lean=True)
        c2, c2i = diffeo.compose_cascade(up(u4), up(u4i), u2, u2i)          # level 2
        U, Ui = up(c2), up(c2i)                                             # level-1 voxels
        del c2, c2i
        u, ui = diffeo.compose_cascade(U, Ui, u1, u1i, lean=True)
        del U, Ui
        margin = int(np.ceil(float(np.linalg.norm(u.numpy(), axis=-1).max()))) + 3

        def err(a, b):
            norm = np.linalg.norm(interior(diffeo.compose(a, b, lean=True).numpy(), margin), axis=-1)
            return float(norm.max()), float(norm.mean())

        r = {"u o u_inv": err(u, ui), "u_inv o u": err(ui, u)}
        del u, ui
        U4, U4i = up(up(u4)), up(up(u4i))
        r.update({"U4 o U4_inv": err(U4, U4i), "U4_inv o U4": err(U4i, U4)})
        del U4, U4i
        U2, U2i = up(u2), up(u2i)
        r.update({"U2 o U2_inv": err(U2, U2i), "U2_inv o U2": err(U2i, U2)})
        del U2, U2i
        r.update({"u1 o u1_inv": err(u1, u1i), "u1_inv o u1": err(u1i, u1)})
        print("\n[T-C4] three-level round trip (level-1 voxels, %d-voxel margin): %s" % (margin, "; ".join(
            "%s max %.4f mean %.5f" % (k, a, b) for k, (a, b) in r.items())), file=sys.stderr)
        for k, a, b, c in (("u o u_inv", "U4 o U4_inv", "U2 o U2_inv", "u1 o u1_inv"),
                           ("u_inv o u", "U4_inv o U4", "U2_inv o U2", "u1_inv o u1")):
            mx, mean = r[k]
            self.assertLess(mx, margin, k)                     # far below the interior margin
            self.assertLessEqual(mx, 0.5, k)
            self.assertLessEqual(mean, 0.05, k)
            # The composition adds no error beyond the sum of the stages' own inverse errors.
            self.assertLessEqual(mx, 1.25 * (r[a][0] + r[b][0] + r[c][0]), k)
            self.assertLessEqual(mean, 1.25 * (r[a][1] + r[b][1] + r[c][1]), k)


class TestChainTrainer(ChainCase):
    """T-C4 centering part, T-C6, T-C7, T-C9, T-V1: the level-1 Trainer with a FrozenChain (deform
    steps traced, not executed: see the section comment)."""

    def opt(self):
        return O.make_optimizer("adam", lr=1e-3, amp=True)

    def test_dispatch(self):
        """T-C9: _n_stages from getattr(cascade, 'n_stages', 1); a chain needs a level-1 trainer."""
        sc = standin_module()
        chain = self.chain()
        self.assertEqual(self.make1(chain=chain)._n_stages, 2)
        stand = sc.FrozenChainStandIn(chain.stage4, chain.model2)
        self.assertEqual(self.make1(chain=stand)._n_stages, 2)
        m1 = self.level1_model(chain)
        cfg1 = chain_config(task_ratio={"rigid": 1.0})
        src1 = {"rigid": fd.FixedGroupSource(self.batches1["rigid"])}
        self.assertEqual(Trainer(m1, self.opt(), src1, self.lgroup["S"], cfg1)._n_stages, 0)
        with self.assertRaises(ValueError):                       # level 1 rejects a single stage
            Trainer(m1, self.opt(), src1, self.lgroup["S"], cfg1, cascade=chain.stage4)
        tf.keras.mixed_precision.set_global_policy("float32")
        cfg2 = cascade_config(task_ratio={"rigid": 1.0})
        g2 = synthetic_group(2, N2, "rigid", seed=0)
        m2 = GroupRegLevelModel(2, tiny_config(**STAGE_KW)).build_model(2)
        src2 = {"rigid": fd.FixedGroupSource(g2["batch"])}
        opt2 = O.make_optimizer("adam", lr=1e-3)
        self.assertEqual(Trainer(m2, opt2, src2, g2["S"], cfg2, cascade=chain.stage4)._n_stages, 1)
        self.assertEqual(Trainer(m2, opt2, src2, g2["S"], cfg2, cascade=_FakeStage())._n_stages, 1)
        for bad in (chain, stand):                                # a chain needs a level-1 trainer
            with self.assertRaises(ValueError) as cm:
                Trainer(m2, opt2, src2, g2["S"], cfg2, cascade=bad)
            self.assertIn("level-1 trainer", str(cm.exception))

    def test_residual_centering(self):
        """T-C4 (centering): the trainable level-1 residual SVF has group mean ~0 and the composed
        A is centred (Center1 applied again changes it by < tolerance)."""
        chain = self.chain()
        m1 = self.level1_model(chain)
        dx = tf.constant(self.dgroup1["images"])
        x_in = cascade.prewarp_deform(dx, chain.deform(dx)["U"])
        out = m1(x_in, task="deform", training=False)
        v, v_raw = out["v"].numpy(), out["v_raw"].numpy()
        del out, x_in
        mean_max = float(np.abs(v.mean(axis=0)).max())
        raw_mean = float(np.abs(v_raw.mean(axis=0)).max())
        v_max = float(np.abs(v).max())
        x = tf.constant(self.lgroup["images"])
        a_prev = chain.linear(x, "affine")
        o1 = m1(cascade.prewarp_linear(x, a_prev, SP1), task="affine", training=False)
        A = cascade.compose_linear(o1["A_raw"], a_prev, m1.centering)
        again = float(tf.reduce_max(tf.abs(m1.centering(A) - A)))
        print("\n[T-C4] level-1 residual: max |mean_i v_i| %.2e (raw %.2e, max |v| %.3g); re-centering the "
              "composed A changes it by %.2e" % (mean_max, raw_mean, v_max, again), file=sys.stderr)
        self.assertGreater(v_max, 0.0)
        self.assertGreater(raw_mean, 10.0 * mean_max)               # the centering did something
        self.assertLessEqual(mean_max, 1e-5 * v_max)
        self.assertLess(again, 1e-5)

    @staticmethod
    def skip_ops(fn):
        g = fn.get_concrete_function().graph
        return [op.name for op in g.get_operations()
                if any(p.startswith(cascade.SKIP_SCOPE) for p in op.name.split("/"))]

    def test_skip_and_frozen_unchanged(self):
        """T-C6 (p 0: no skip ops; forced skip = the level-4 stage alone) and T-C7 (the frozen chain
        never changes: train steps, recalibrate_bn)."""
        chain = self.chain()
        fz = [v for m in chain.models for v in m.weights]
        fz_refs = {v.ref() for v in fz}
        before = _vals(fz)
        tr0 = self.make1(chain=chain)
        tr5 = self.make1(chain_config(cascade_skip_prob=0.5), chain=chain)
        # Graphs (traced, the deform step is not executed): skip ops only with p > 0.
        for t in ("rigid", "deform"):
            self.assertEqual(self.skip_ops(tr0._steps[t]), [], t)
            self.assertTrue(self.skip_ops(tr5._steps[t]), t)
        tr5.rebuild_steps()
        tr5.trace_counts = {t: 0 for t in tr5.trace_counts}
        # Forced skip (linear, executed): the step equals a chain whose output is level 4 alone.
        ref = self.make1(chain_config(task_ratio={"rigid": 1.0}), chain=_Level4Chain(chain))
        self.assertEqual(ref._n_stages, 2)
        rs = tr5.train_step("rigid", self.batches1["rigid"], skip=True)
        rn = ref.train_step("rigid", self.batches1["rigid"])
        print("\n[T-C6] rigid forced skip loss %.8g vs level-4-only chain %.8g" % (rs["loss"], rn["loss"]),
              file=sys.stderr)
        # (AMP: the first steps may be skipped by the loss-scale warm-up, so 'finite' is not
        # required; the loss itself is finite.)
        self.assertTrue(np.isfinite(rs["loss"]))
        np.testing.assert_allclose(rs["loss"], rn["loss"], rtol=1e-4)
        self.assertTrue(np.isfinite(tr5.train_step("rigid", self.batches1["rigid"], skip=False)["loss"]))
        self.assertEqual(tr5.trace_counts["rigid"], 1)                 # no retrace on the skip value
        self.assertTrue(np.isfinite(tr0.train_step("rigid", self.batches1["rigid"])["loss"]))
        # Deform skip semantics on the chain outputs (eager): U = U4, U_inv = U4_inv when skipped.
        st = chain.deform_stages(tf.constant(self.dgroup1["images"]))
        for k, k4 in (("U", "U4"), ("U_inv", "U4_inv")):
            np.testing.assert_array_equal(cascade.skip_to(tf.constant(True), st[k4], st[k]).numpy(), st[k4].numpy())
            np.testing.assert_array_equal(cascade.skip_to(tf.constant(False), st[k4], st[k]).numpy(), st[k].numpy())
        self.assertGreater(float(np.abs(st["U"].numpy() - st["U4"].numpy()).max()), 1e-4)
        del st
        # T-C7: the frozen chain never changes and is never in the optimizer.
        self.assertTrue(_all_equal(before, fz))
        for tr in (tr0, tr5, ref):
            inner = getattr(tr.optimizer, "inner_optimizer", tr.optimizer)      # AMP: LossScaleOptimizer
            self.assertTrue(inner.variables())
            self.assertFalse(fz_refs & {v.ref() for v in inner.variables()})
        rec = self.make1(chain_config(task_ratio={"rigid": 1.0}), chain=chain)
        ev = rec.recalibrate_bn(2, freeze=False, val=False)
        self.assertEqual(ev["batches"], {"rigid": 2})
        self.assertTrue(_all_equal(before, fz))

    def test_validation(self):
        """T-V1: level-1 chain validation has the _frozen (4 + 2) and _frozen4 baselines and logs
        val/wall_s. Rigid set end to end; deform: the traced eval forward's outputs and the
        baseline code on the chain's frozen fields."""
        from runtime.run import CHAIN_DEFORM_FROZEN_KEYS, VAL_WALL_KEY
        logger = RunLogger(make_run_dir(self.tmp, "l1"), backend="jsonl", tensorboard=False, quiet=True)
        vs = self.val_sets1()
        tr = self.make1(chain_config(task_ratio={"rigid": 1.0}), val_sets={"rigid": vs["rigid"]}, logger=logger)
        v = tr.validate(max_groups=1)
        for pre in ("val_rigid", "val_rigid_frozen", "val_rigid_frozen4"):
            for k in ("center_rms_mm", "pair_rms_mm", "center_rot_deg", "pair_trans_mm"):
                self.assertTrue(np.isfinite(v["%s/%s" % (pre, k)]), (pre, k))
        self.assertNotEqual(v["val_rigid_frozen/center_rms_mm"], v["val_rigid_frozen4/center_rms_mm"])
        self.assertEqual(sorted(k.replace("_frozen4/", "_frozen/") for k in v if "_frozen4/" in k),
                         sorted(k for k in v if "_frozen/" in k))
        self.assertNotIn(VAL_WALL_KEY, v)
        # The linear baselines are exactly the group errors of A (4 + 2) and A4 on the same group.
        g = tr.val_sets["rigid"].groups[0]
        st = tr.cascade.linear_stages(tf.constant(g["images"]), "rigid")
        for key, pre in (("A", "val_rigid_frozen"), ("A4", "val_rigid_frozen4")):
            a = st[key].numpy().astype(np.float64)
            want = M.summarize(M.group_errors(a, g["A_star"], g["T_target"], tr.S_np, seed=0), pre)
            for k, w in want.items():
                self.assertAlmostEqual(v[k], w, delta=1e-4 * max(1.0, abs(w)), msg=k)
        val = tr._validate_and_log(0)
        self.assertGreater(val[VAL_WALL_KEY], 0.0)
        self.assertIn(VAL_WALL_KEY, tr.val_history[-1])
        recs = [r["data"] for r in CascadeTrainerCase.records(logger)]
        self.assertTrue(any(VAL_WALL_KEY in r for r in recs))
        print("\n[T-V1] level-1 rigid validation wall time %.1f s; frozen center rms %.3f (4 + 2) / %.3f (4)"
              % (val[VAL_WALL_KEY], v["val_rigid_frozen/center_rms_mm"], v["val_rigid_frozen4/center_rms_mm"]),
              file=sys.stderr)
        # Deform: the eval-mode validation forward returns the six frozen fields, train mode none.
        trd = self.make1(chain_config(task_ratio={"deform": 1.0}), chain=tr.cascade, val_sets={"deform": vs["deform"]})
        ev = trd._val_deform_fn(2, False).get_concrete_function().structured_outputs
        self.assertEqual(sorted(ev), sorted(("u", "u_inv", "v", "v_raw") + CHAIN_DEFORM_FROZEN_KEYS))
        tm = trd._val_deform_fn(2, True).get_concrete_function().structured_outputs
        self.assertEqual(sorted(tm), ["u", "u_inv", "v", "v_raw"])
        lin = tr._val_fn("rigid", 2, False).get_concrete_function().structured_outputs
        self.assertEqual(sorted(lin), ["A", "A_prev", "A_prev4"])
        # Deform baselines from the chain's frozen fields (what _val_deform_errors collects).
        dgrp = vs["deform"].groups[0]
        ds = tr.cascade.deform_stages(tf.constant(dgrp["images"]))
        fo = {k: ds[k].numpy() for k in CHAIN_DEFORM_FROZEN_KEYS}
        del ds
        base = trd._frozen_baseline_metrics("deform", vs["deform"], [dgrp], [fo])
        for pre in ("val_deform_frozen", "val_deform_frozen4"):
            for k in ("template_dice", "pair_dice", "neg_jac_pct", "inv_consistency_mm", "ncc_template", "ncc_pair"):
                self.assertTrue(np.isfinite(base["%s/%s" % (pre, k)]), (pre, k))
        self.assertEqual(sorted(k.replace("_frozen4/", "_frozen/") for k in base if "_frozen4/" in k),
                         sorted(k for k in base if "_frozen/" in k))
        self.assertNotEqual(base["val_deform_frozen/template_dice"], base["val_deform_frozen4/template_dice"])
        self.assertEqual(trd._frozen_baseline_metrics("deform", vs["deform"], [dgrp], None), base)   # cached
        print("\n[T-V1] level-1 deform frozen template Dice %.4f (4 + 2) / %.4f (4)"
              % (base["val_deform_frozen/template_dice"], base["val_deform_frozen4/template_dice"]), file=sys.stderr)


class TestLevel2Unchanged(CascadeTrainerCase):
    """T-V1 level-2 part: a milestone-8 trainer keeps its validation keys (no _frozen4, no
    val/wall_s) and its single-stage dispatch."""

    def test_level2_keys(self):
        from runtime.run import VAL_WALL_KEY
        tr = self.make(val_sets=self.val_sets())
        self.assertEqual(tr._n_stages, 1)
        val = tr._validate_and_log(0)
        self.assertFalse([k for k in val if "_frozen4" in k])
        self.assertNotIn(VAL_WALL_KEY, val)
        for name in ("rigid", "affine", "deform"):
            self.assertTrue(any(k.startswith("val_%s_frozen/" % name) for k in val), name)


# ============================================================ per-stage skip (retrain-prereqs 4, 11.3)
def _skip_op_names(fn):
    g = fn.get_concrete_function().graph
    return [op.name for op in g.get_operations()
            if any(p.startswith(cascade.SKIP_SCOPE) for p in op.name.split("/"))]


class TestStageSkipDraw(unittest.TestCase):
    """11.3 b: `cascade_stage_skip_draw` is deterministic per (seed, step, task, level), independent
    across levels and tasks, and never skips at p 0."""

    def test_draws(self):
        n, p4, p2 = 20000, 0.15, 0.3
        d4 = [cascade_stage_skip_draw(0, s, "deform", 4, p4) for s in range(n)]
        d2 = [cascade_stage_skip_draw(0, s, "deform", 2, p2) for s in range(n)]
        r4 = [cascade_stage_skip_draw(0, s, "rigid", 4, p4) for s in range(n)]
        self.assertEqual(d4[:500], [cascade_stage_skip_draw(0, s, "deform", 4, p4) for s in range(500)])
        self.assertEqual(cascade_stage_skip_draw(5, 17, "affine", 2, 0.5),
                         random.Random("cascade-skip-affine-l2-5-17").random() < 0.5)
        self.assertNotEqual(d4[:500], [cascade_stage_skip_draw(1, s, "deform", 4, p4) for s in range(500)])
        rates = {"deform l4": np.mean(d4), "deform l2": np.mean(d2), "rigid l4": np.mean(r4),
                 "deform l4 & l2": np.mean(np.logical_and(d4, d2)),
                 "deform l4 & rigid l4": np.mean(np.logical_and(d4, r4))}
        print("\n[11.3 b] stage skip rates over %d steps: %s" % (n, ", ".join(
            "%s %.4f" % kv for kv in rates.items())), file=sys.stderr)
        self.assertAlmostEqual(rates["deform l4"], p4, delta=0.01)
        self.assertAlmostEqual(rates["deform l2"], p2, delta=0.01)
        self.assertAlmostEqual(rates["rigid l4"], p4, delta=0.01)
        self.assertAlmostEqual(rates["deform l4 & l2"], p4 * p2, delta=0.01)
        self.assertAlmostEqual(rates["deform l4 & rigid l4"], p4 * p4, delta=0.01)
        self.assertNotEqual(d4, r4)
        self.assertFalse(any(cascade_stage_skip_draw(0, s, "deform", 4, 0.0) for s in range(n)))
        # The legacy draw is untouched (its own seed string).
        self.assertEqual(cascade_skip_draw(5, 17, 0.5), random.Random("cascade-skip-5-17").random() < 0.5)


class TestStageSkipLevel2(CascadeTrainerCase):
    """11.3 d / e (level 2): stage mode with a single frozen stage."""

    STAGE = {"rigid": {4: 0.5}, "affine": {4: 0.5}, "deform": {4: 0.5}}

    def test_matches_legacy(self):
        """11.3 d: stage mode with 4:p and skip True / False gives the legacy step results (same
        seeds and weights, max |diff| 0), for all three tasks."""
        with _OpDeterminism(), _GrapplerOff():
            a = self.make(cascade_config(cascade_skip_prob=0.5))
            b = self.make(cascade_config(cascade_skip_stage=self.STAGE))
            self.assertEqual((a._skip_mode, b._skip_mode), ("legacy", "stage"))
            la, lb = [], []
            for sk in (True, False):
                for t in ("rigid", "affine", "deform"):
                    la.append(a.train_step(t, self.batches[t], skip=sk)["loss"])
                    lb.append(b.train_step(t, self.batches[t], skip=(sk,))["loss"])
            sa, sb = _var_sums(a.model), _var_sums(b.model)
        diffs = [abs(x - y) for x, y in zip(la, lb)] + [abs(x - y) for x, y in zip(sa, sb)]
        print("\n[11.3 d] level-2 stage vs legacy skip: max |diff| %.3g over %d losses and %d variable sums"
              % (max(diffs), len(la), len(sa)), file=sys.stderr)
        self.assertEqual(la, lb)
        self.assertEqual(sa, sb)

    def test_trainer_stage_mode(self):
        """11.3 e: [n_stages] bool signature, one trace per task, no skip op for p 0 tasks,
        logged keys only for p > 0, validation without skip ops; constructor checks."""
        stage = {"rigid": {}, "affine": {4: 0.0}, "deform": {4: 0.5}}
        tr = self.make(cascade_config(cascade_skip_stage=stage, log_every=3), val_sets=self.val_sets(),
                       logger=self.logger("stage"))
        self.assertEqual((tr._skip_mode, tr._frozen_levels), ("stage", [4]))
        for t in ("rigid", "affine", "deform"):
            spec = tr._steps[t].input_signature[-1]
            self.assertEqual((spec.dtype, tuple(spec.shape)), (tf.bool, (1,)), t)
        self.assertEqual(_skip_op_names(tr._steps["rigid"]), [])
        self.assertEqual(_skip_op_names(tr._steps["affine"]), [])
        self.assertTrue(_skip_op_names(tr._steps["deform"]))
        self.assertEqual(_skip_op_names(tr._val_fn("rigid", N2, False)), [])
        self.assertEqual(_skip_op_names(tr._val_deform_fn(ND2, False)), [])
        tr.rebuild_steps()
        tr.trace_counts = {t: 0 for t in tr.trace_counts}
        with self.assertRaises(TypeError):
            tr.train_step("rigid", self.batches["rigid"], skip=True)
        for sk in ((True,), (False,), [True]):
            for t in ("rigid", "deform"):
                self.assertTrue(np.isfinite(tr.train_step(t, self.batches[t], skip=sk)["loss"]), t)
        tr.fit(6)
        self.assertEqual(tr.trace_counts, {"rigid": 1, "affine": 1, "deform": 1})
        recs = [r["data"] for r in self.records(tr.logger)]
        keys = set().union(*recs)
        self.assertIn("train_deform/cascade_skip_l4", keys)
        self.assertFalse([k for k in keys if "cascade_skip" in k and k != "train_deform/cascade_skip_l4"])
        start = tr.history[0]["step"] - 1
        for h in tr.history:
            if h["task"] == "deform":
                self.assertEqual(h["cascade_skip_l4"], cascade_stage_skip_draw(0, h["step"] - 1, "deform", 4, 0.5))
            else:
                self.assertNotIn("cascade_skip_l4", h)
            self.assertNotIn("cascade_skip", h)
        self.assertEqual(start, 6)
        # Constructor checks (Trainer.__init__ re-validation).
        for bad in ({"deform": {2: 0.1}}, {"deform": {4: 1.0}}, {"deform": {4: float("nan")}}):
            with self.assertRaises(ValueError):
                self.make(cascade_config(cascade_skip_stage=bad))
        with self.assertRaises(ValueError) as cm:
            self.make(cascade_config(cascade_skip_prob=0.1, cascade_skip_stage={"deform": {4: 0.1}}))
        self.assertIn("mutually exclusive", str(cm.exception))
        with self.assertRaises(ValueError):
            self.make(cascade_config(cascade_skip_stage={"deform": {4: 0.1}}), cascade_on=False)
        off = self.make(cascade_config(cascade_skip_stage={"rigid": {}, "deform": {4: 0.0}}))
        self.assertEqual(off._skip_mode, "off")
        self.assertEqual(tuple(off._steps["deform"].input_signature[-1].shape), ())
        self.assertEqual(_skip_op_names(off._steps["deform"]), [])


class _IdentityChain:
    """Chain wrapper whose linear returns the identity (skip-both reference, 11.3 c / e)."""

    level = 2
    n_stages = 2

    def __init__(self, chain):
        self.chain = chain

    @property
    def models(self):
        return self.chain.models

    def linear(self, x, task):
        return tf.eye(4, batch_shape=[int(x.shape[0])], dtype=tf.float32)


class TestStageSkipChain(ChainCase):
    """11.3 c / e (level 1): `FrozenChain.linear_skip` / `deform_skip` and the level-1 Trainer in
    stage mode (deform steps traced, not executed: see the chain section comment)."""

    def test_chain_methods(self):
        """11.3 c: skip4=None equals linear_stages / deform_stages + skip_to (max |diff| 0); skip4
        True = level 2 on the identity-pre-warped x2; both True = identity."""
        chain = self.chain()
        x = tf.constant(self.lgroup["images"])
        dx = tf.constant(self.dgroup1["images"])
        T, F = tf.constant(True), tf.constant(False)
        with _OpDeterminism():
            st = chain.linear_stages(x, "rigid")
            np.testing.assert_array_equal(chain.linear_skip(x, "rigid").numpy(), st["A"].numpy())
            for s2 in (T, F):
                np.testing.assert_array_equal(chain.linear_skip(x, "rigid", skip2=s2).numpy(),
                                              cascade.skip_to(s2, st["A4"], st["A"]).numpy())
            a4s = chain.linear_skip(x, "rigid", skip4=T).numpy()
            x2 = cascade.pool_to_coarser(x)
            eye = tf.eye(4, batch_shape=[int(x.shape[0])], dtype=tf.float32)
            x2i = cascade.prewarp_linear(x2, eye, SP2)
            out = chain.model2(x2i, task="rigid", training=False)
            ref = cascade.compose_linear(out["A_raw"], eye, chain.model2.centering).numpy()
            d_ref = float(np.max(np.abs(a4s - ref)))
            d_unskipped = float(np.max(np.abs(a4s - st["A"].numpy())))
            d_pre = float(np.max(np.abs(x2i.numpy() - x2.numpy())))
            both = chain.linear_skip(x, "rigid", skip4=T, skip2=T).numpy()
            np.testing.assert_array_equal(chain.linear_skip(x, "rigid", skip4=F, skip2=F).numpy(), st["A"].numpy())
            ds = chain.deform_stages(dx)
            d0 = chain.deform_skip(dx)
            for k in ("U", "U_inv"):
                np.testing.assert_array_equal(d0[k].numpy(), ds[k].numpy(), err_msg=k)
            del d0
            for s2 in (T, F):
                d = chain.deform_skip(dx, skip2=s2)
                for k, k4 in (("U", "U4"), ("U_inv", "U4_inv")):
                    np.testing.assert_array_equal(d[k].numpy(), cascade.skip_to(s2, ds[k4], ds[k]).numpy(),
                                                  err_msg=k)
                del d
            del ds
            d4 = chain.deform_skip(dx, skip4=T)
            dx2 = cascade.pool_to_coarser(dx)
            zero = tf.zeros(tuple(dx2.shape[:4]) + (3,), tf.float32)
            dx2i = cascade.prewarp_deform(dx2, zero)
            o2 = chain.model2(dx2i, task="deform", training=False)
            up = diffeo.upsample_displacement
            d_u = max(float(np.max(np.abs(d4[k].numpy() - up(o2[s]).numpy()))) for k, s in (("U", "u"), ("U_inv", "u_inv")))
            d_pre_d = float(np.max(np.abs(dx2i.numpy() - dx2.numpy())))
            self.assertGreater(float(np.abs(d4["U"].numpy()).max()), 1e-4)
            del d4, o2
            dboth = chain.deform_skip(dx, skip4=T, skip2=T)
            for k in ("U", "U_inv"):
                self.assertEqual(float(np.max(np.abs(dboth[k].numpy()))), 0.0, k)
            del dboth
        print("\n[11.3 c] level-1 skip4: A vs compose(model2 on prewarp(x2, I), I) max |diff| %.3g (vs unskipped "
              "%.3g); |prewarp_linear(x2, I) - x2| %.3g; U4 = 0: |U - up(u2)| %.3g, |prewarp_deform(x2, 0) - x2| %.3g"
              % (d_ref, d_unskipped, d_pre, d_u, d_pre_d), file=sys.stderr)
        self.assertEqual(d_ref, 0.0)
        self.assertGreater(d_unskipped, 1e-4)
        self.assertLess(d_pre, 1e-4)
        self.assertLess(d_u, 1e-5)
        self.assertLess(d_pre_d, 1e-4)
        np.testing.assert_array_equal(both, np.broadcast_to(np.eye(4, dtype=np.float32), both.shape))

    def test_trainer_stage_mode(self):
        """11.3 e (level 1): [2] bool signature, no skip op for a p 0 task, one trace over mixed
        values, skip semantics of the rigid step (skip 2 = level 4 alone, skip both = identity),
        validation without skip ops."""
        chain = self.chain()
        a = self.make1(chain_config(cascade_skip_stage={"rigid": {}, "deform": {4: 0.15, 2: 0.15}}), chain=chain)
        self.assertEqual((a._skip_mode, a._frozen_levels), ("stage", [4, 2]))
        for t in ("rigid", "deform"):
            spec = a._steps[t].input_signature[-1]
            self.assertEqual((spec.dtype, tuple(spec.shape)), (tf.bool, (2,)), t)
        self.assertEqual(_skip_op_names(a._steps["rigid"]), [])
        self.assertTrue(_skip_op_names(a._steps["deform"]))
        self.assertEqual(_skip_op_names(a._val_fn("rigid", 2, False)), [])
        self.assertEqual(_skip_op_names(a._val_deform_fn(2, False)), [])
        del a
        free_memory()
        cfg = chain_config(task_ratio={"rigid": 1.0}, cascade_skip_stage={"rigid": {4: 0.1, 2: 0.1}})
        b = self.make1(cfg, chain=chain)
        self.assertTrue(_skip_op_names(b._steps["rigid"]))
        with self.assertRaises(TypeError):
            b.train_step("rigid", self.batches1["rigid"], skip=True)
        with self.assertRaises(TypeError):
            b.train_step("rigid", self.batches1["rigid"], skip=(True,))
        r_l4 = self.make1(chain_config(task_ratio={"rigid": 1.0}), chain=_Level4Chain(chain))
        r_id = self.make1(chain_config(task_ratio={"rigid": 1.0}), chain=_IdentityChain(chain))
        rs2 = b.train_step("rigid", self.batches1["rigid"], skip=(False, True))
        rn2 = r_l4.train_step("rigid", self.batches1["rigid"])
        del r_l4
        b2 = self.make1(cfg, chain=chain)
        rsb = b2.train_step("rigid", self.batches1["rigid"], skip=(True, True))
        rnb = r_id.train_step("rigid", self.batches1["rigid"])
        print("\n[11.3 e] level-1 rigid: skip l2 loss %.8g vs level-4-only chain %.8g; skip both %.8g vs identity "
              "chain %.8g" % (rs2["loss"], rn2["loss"], rsb["loss"], rnb["loss"]), file=sys.stderr)
        np.testing.assert_allclose(rs2["loss"], rn2["loss"], rtol=1e-4)
        np.testing.assert_allclose(rsb["loss"], rnb["loss"], rtol=1e-4)
        for sk in ((True, False), (False, False), (True, True)):
            self.assertTrue(np.isfinite(b.train_step("rigid", self.batches1["rigid"], skip=sk)["loss"]))
        self.assertEqual(b.trace_counts["rigid"], 1)


if __name__ == "__main__":
    unittest.main()
