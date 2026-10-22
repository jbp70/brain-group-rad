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
from runtime.run import CASCADE_SKIP_KEY, Trainer, TrainConfig, bn_recal_plan, cascade_skip_draw
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


if __name__ == "__main__":
    unittest.main()
