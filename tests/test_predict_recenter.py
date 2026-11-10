"""Tests of `predict.py run --recenter K` (post-hoc group re-centering, inference/recenter.py;
.agents/tasks/predict-recenter/report.md).

* TestRecentererUnit: `Recenterer` on a small synthetic group phi_i = exp(v_i + c) with a common,
  uncentred c: the mean forward displacement drops to ~0 in K rounds, the corrected fields equal an
  independent scipy recomputation of the documented algorithm (evaluation.fields.compose), the
  pairwise maps compose(inv_j', fwd_i') equal the originals, inverse consistency does not degrade,
  the temporary directory is removed.
* TestRecenterCLI: --recenter validation (exit 2 for negative values and rigid / affine branches;
  default 0).
* TestRecenterRun: `run` end to end on a tiny level-4 checkpoint (N 3) whose outputs get a known
  common warp (to_canonical patched: fwd o C, C^-1 o inv): recenter 0 vs 2 (predict.json record,
  centrality from the written files, moved / atlas / atlas_seg recomputed from the corrected
  fields, pairwise maps unchanged, inverse error not worse), affine+deform with --tta flip8, a
  level-2 cascade with --levels 4 and --integration-grid fine, non-finite outputs (re-centering
  skipped, exit 3). The default (no flag) is covered bit-exactly by tests.test_predict_tta_golden.
Measured values are printed to stderr.
"""
from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory limit)
from tests.predict_helpers import save_cascade_run, save_level4_run, write_subjects
from tests.runtime_helpers import free_memory, tiny_config
from tests.test_cascade import STAGE_KW, randomize
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np
import tensorflow as tf
from scipy import ndimage

from evaluation import fields as F
from evaluation import metrics as EM
from inference import canonical as CN
from inference import cli as CLI
from inference import predict as PR
from inference import recenter as RC
from models import cascade as CS
from models import diffeo
from models.group_net import GroupRegLevelModel

SMALL = (32, 36, 28)


def _say(msg):
    print("\n[recenter] " + msg, file=sys.stderr)


def _smooth_field(shape, rng, amp, sigma=4.0):
    f = np.stack([ndimage.gaussian_filter(rng.standard_normal(shape), sigma) for _ in range(3)], -1)
    return (amp * f / np.abs(f).max()).astype(np.float32)


def _exp_pair_np(v):
    u, ui = diffeo.exp_and_inverse(tf.constant(v[None]), 7)
    return u.numpy()[0], ui.numpy()[0]


def _mean_norm(u, mask=None):
    n = F.norm_mm(u, (1.0, 1.0, 1.0))
    return float(n[mask].mean() if mask is not None else n.mean())


def _interior(shape, margin):
    m = np.zeros(shape, bool)
    m[margin:-margin, margin:-margin, margin:-margin] = True
    return m


class TestRecentererUnit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        cls.n = 5
        c = _smooth_field(SMALL, rng, 2.0, 6.0)
        cls.fwd, cls.inv = [], []
        for _ in range(cls.n):
            u, ui = _exp_pair_np(_smooth_field(SMALL, rng, 2.5) + c)
            cls.fwd.append(u)
            cls.inv.append(ui)
        cls.tmp = tempfile.mkdtemp(prefix="bgr_rc_unit_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        free_memory()

    def _run(self, rounds):
        rc = RC.Recenterer(rounds, self.n, SMALL, (1.0, 1.0, 1.0), self.tmp)
        ones = np.ones(SMALL, np.float32)
        for i in range(self.n):
            rc.add(i, self.fwd[i], self.inv[i], ones)
        out = {}

        def write(i, f, v):
            out[i] = (f.copy(), v.copy())
            return ones
        try:
            rec = rc.run(write, np.ones((self.n,) + SMALL, np.float32))
        finally:
            rc.close()
        self.assertFalse(os.path.exists(rc.dir))
        self.assertEqual(sorted(out), list(range(self.n)))
        return rec, [out[i][0] for i in range(self.n)], [out[i][1] for i in range(self.n)]

    def test_rounds_reduce_mean_and_match_reference(self):
        rec, fwd2, inv2 = self._run(2)
        hist = [h["mean_mm"] for h in rec["centrality_by_round"]]
        _say("unit centrality by round %s (max %s)" % (["%.5f" % h for h in hist],
                                                       ["%.4f" % h["max_mm"] for h in rec["centrality_by_round"]]))
        self.assertEqual(rec["rounds"], 2)
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(len(hist), 3)
        self.assertGreater(hist[0], 0.3)
        self.assertLess(hist[1], 0.1 * hist[0])
        self.assertLess(hist[2], 0.02 * hist[0])
        mean2 = np.mean(np.stack(fwd2), 0)
        self.assertAlmostEqual(_mean_norm(mean2), hist[2], places=5)
        # independent recomputation (scipy compose, evaluator semantics F.compose(a, b) = b o a)
        w = np.mean(np.stack(self.fwd).astype(np.float64), 0).astype(np.float32)
        p, _ = _exp_pair_np(-w)
        m1 = np.mean(np.stack([F.compose(p, f) for f in self.fwd]).astype(np.float64), 0).astype(np.float32)
        w2 = (w.astype(np.float64) + m1).astype(np.float32)
        p2, q2 = _exp_pair_np(-w2)
        d_f = max(float(np.abs(F.compose(p2, f) - g).max()) for f, g in zip(self.fwd, fwd2))
        d_i = max(float(np.abs(F.compose(v, q2) - g).max()) for v, g in zip(self.inv, inv2))
        _say("unit vs scipy reference: fwd %.2e, inv %.2e vox" % (d_f, d_i))
        self.assertLess(d_f, 1e-3)
        self.assertLess(d_i, 1e-3)
        # pairwise maps (evaluator u_ij = compose(inv_j, fwd_i)) unchanged, inverse error not worse
        inner = _interior(SMALL, 6)
        worst = 0.0
        for i, j in ((0, 1), (2, 4), (3, 0)):
            a = F.compose(self.inv[j], self.fwd[i])
            b = F.compose(inv2[j], fwd2[i])
            worst = max(worst, float(F.norm_mm(a - b, (1, 1, 1))[inner].max()))
        e0 = np.mean([_mean_norm(F.compose(f, v), inner) for f, v in zip(self.fwd, self.inv)])
        e1 = np.mean([_mean_norm(F.compose(f, v), inner) for f, v in zip(fwd2, inv2)])
        _say("unit pairwise max change %.4f vox; inverse error %.5f -> %.5f vox" % (worst, e0, e1))
        self.assertLess(worst, 0.05)
        self.assertLess(e1, e0 + 0.01)
        ie = rec["inv_err"]
        self.assertEqual(ie["subjects"], [0, 1, 2, 3])
        self.assertLess(ie["after"]["inv_err_fwd_mm"], ie["before"]["inv_err_fwd_mm"] + 0.01)

    def test_one_round_and_validation(self):
        rec, _, _ = self._run(1)
        hist = [h["mean_mm"] for h in rec["centrality_by_round"]]
        self.assertEqual(len(hist), 2)
        self.assertLess(hist[1], 0.1 * hist[0])
        self.assertEqual(sorted(rec["timing_s"]), ["collect", "round_1"])
        with self.assertRaises(ValueError):
            RC.Recenterer(0, 2, SMALL, (1, 1, 1), self.tmp)
        rc = RC.Recenterer(1, 2, SMALL, (1, 1, 1), self.tmp)
        try:
            rc.add(0, self.fwd[0], self.inv[0], np.ones(SMALL))
            with self.assertRaises(RuntimeError):
                rc.run(lambda i, f, v: np.ones(SMALL), np.ones((2,) + SMALL))
        finally:
            rc.close()

    def test_atlas_mask_rule(self):
        cnt = np.array([0, 1, 2, 3], np.int32)
        np.testing.assert_array_equal(RC.atlas_mask(cnt, 4), [False, False, True, True])
        np.testing.assert_array_equal(RC.atlas_mask(cnt, 5), [False, False, False, True])


class TestRecenterCLI(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_rc_cli_")
        self.ck = os.path.join(self.tmp, "ck")
        os.makedirs(self.ck)
        self.proc = os.path.join(self.tmp, "proc")
        os.makedirs(self.proc)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _parse(self, *extra):
        return CLI.parse_args(["run", "--checkpoint", self.ck, "--processed-dir", self.proc,
                               "--out", os.path.join(self.tmp, "out")] + list(extra))

    def _exit2(self, *extra):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            self._parse(*extra)
        self.assertEqual(cm.exception.code, 2)
        return err.getvalue()

    def test_flag(self):
        self.assertEqual(self._parse().recenter, 0)
        self.assertEqual(self._parse("--branches", "deform", "--recenter", "2").recenter, 2)
        self.assertEqual(self._parse("--recenter", "1").recenter, 1)          # default affine+deform
        self.assertEqual(self._parse("--branches", "rigid", "--recenter", "0").recenter, 0)
        self.assertIn("must be an integer >= 0", self._exit2("--recenter", "-1"))
        self.assertIn("is not an integer", self._exit2("--recenter", "x"))
        for b in ("rigid", "affine"):
            self.assertIn("--recenter needs --branches deform or affine+deform", self._exit2("--branches", b,
                                                                                          "--recenter", "2"))
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:   # `run` only
            CLI.parse_args(["check-streaming", "--checkpoint", self.ck, "--processed-dir", self.proc,
                            "--out", os.path.join(self.tmp, "c.json"), "--recenter", "1"])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("unrecognized arguments: --recenter", err.getvalue())


def _read(path):
    return np.asarray(nib.load(path).dataobj)


def _common_warp():
    """A smooth common SVF c on the level-1 grid (amplitude ~1.5 vox) -> (C, C^-1) TF [1, X, Y, Z, 3]."""
    rng = np.random.default_rng(7)
    shape = CN.GRID_SHAPE
    small = tuple(s // 8 for s in shape)
    c = np.stack([ndimage.zoom(ndimage.gaussian_filter(rng.standard_normal(small), 1.5), 8, order=1)
                  for _ in range(3)], -1)
    c = (1.5 * c / np.abs(c).max()).astype(np.float32)
    return diffeo.exp_and_inverse(tf.constant(c[None]), 7)


class TestRecenterRun(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgr_rc_run_")
        cfg = tiny_config(**STAGE_KW)
        cls.ckpt, _ = save_level4_run(cls.tmp, "l4", cfg, seed=5)
        cls.proc = os.path.join(cls.tmp, "proc")
        cls.dirs, cls.images = write_subjects(cls.proc, 3, textured=True, seed=2)
        cls.sdirs = [os.path.basename(d) for d in cls.dirs]
        cls.labels = [_read(os.path.join(d, "labels_dkt31aseg.nii.gz")) for d in cls.dirs]
        cls.C, cls.C_inv = _common_warp()
        orig = CN.to_canonical

        def warped(*a, **k):                 # fwd o C, C^-1 o inv (a known common warp)
            fwd, inv = orig(*a, **k)
            return diffeo.compose(fwd, cls.C), diffeo.compose(cls.C_inv, inv)
        cls.codes, cls.out = {}, {}
        for name, k in (("rc0", "0"), ("rc2", "2")):
            out = os.path.join(cls.tmp, "out_" + name)
            args = CLI.parse_args(["run", "--checkpoint", cls.ckpt, "--processed-dir", cls.proc, "--out", out,
                                   "--branches", "deform", "--check-internal", "0", "--recenter", k,
                                   "--store-dir", os.path.join(cls.tmp, "store")])
            with mock.patch.object(CN, "to_canonical", side_effect=warped):
                cls.codes[name] = PR.run_predict(args, configure_device=False)
            cls.out[name] = out
            free_memory()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        free_memory()

    def _rec(self, name):
        with open(os.path.join(self.out[name], "predict.json")) as f:
            return json.load(f)

    def _fields(self, name):
        out = self.out[name]
        return ([_read(os.path.join(out, s + "_fwd.nii.gz")).astype(np.float32) for s in self.sdirs],
                [_read(os.path.join(out, s + "_inv.nii.gz")).astype(np.float32) for s in self.sdirs],
                [_read(os.path.join(out, s + "_moved.nii.gz")).astype(np.float32) for s in self.sdirs])

    def test_records(self):
        self.assertEqual(self.codes, {"rc0": 0, "rc2": 0}, self._rec("rc2").get("message"))
        r0, r2 = self._rec("rc0"), self._rec("rc2")
        self.assertEqual(set(r2), set(PR.PREDICT_KEYS))
        self.assertEqual(r0["recenter"], {"rounds": 0})
        self.assertNotIn("re-centering", r0["pipeline"])
        self.assertNotIn("recenter", r0["timing_s"])
        rc = r2["recenter"]
        self.assertEqual((rc["rounds"], rc["status"], rc["integration_steps"]), (2, "ok", 7))
        self.assertIn("re-centering, 2 round(s)", r2["pipeline"])
        self.assertIn("recenter", r2["timing_s"])
        self.assertEqual(sorted(rc["timing_s"]), ["collect", "round_1", "round_2"])
        self.assertFalse(os.path.exists(rc["tmp_dir"]))
        self.assertTrue(rc["tmp_dir"].startswith(os.path.join(self.tmp, "store")))
        self.assertAlmostEqual(rc["tmp_gb"], 3 * 2 * 4 * 3 * float(np.prod(CN.GRID_SHAPE)) / 1e9, places=6)
        hist = [h["mean_mm"] for h in rc["centrality_by_round"]]
        ie = rc["inv_err"]
        _say("run centrality by round %s, final mask %.5f; inv_err fwd %.4f -> %.4f, inv %.4f -> %.4f mm"
             % (["%.5f" % h for h in hist], rc["centrality_final_mask"]["mean_mm"], ie["before"]["inv_err_fwd_mm"],
                ie["after"]["inv_err_fwd_mm"], ie["before"]["inv_err_inv_mm"], ie["after"]["inv_err_inv_mm"]))
        self.assertGreater(hist[0], 0.3)
        self.assertLess(hist[2], 0.02 * hist[0])
        self.assertEqual(ie["subjects"], [0, 1, 2])
        for k in ("inv_err_fwd_mm", "inv_err_inv_mm"):
            self.assertLess(ie["after"][k], ie["before"][k] + 0.01)
        self.assertEqual(r2["outputs"], r0["outputs"])
        self.assertEqual(r2["nonfinite"]["total"], 0)

    def test_files_from_corrected_fields(self):
        fwd, inv, moved = self._fields("rc2")
        n = len(fwd)
        cnt = np.zeros(CN.GRID_SHAPE, np.int32)
        for k in range(n):
            ref = F.warp(self.images[k], fwd[k], order=1)
            np.testing.assert_array_equal(moved[k], ref)
            cnt += moved[k] > 0
        atlas = _read(os.path.join(self.out["rc2"], "atlas.nii.gz"))
        ref_atlas = (np.sum(np.stack(moved).astype(np.float64), 0) / n).astype(np.float32)
        np.testing.assert_allclose(atlas, ref_atlas, atol=1e-6)
        seg = _read(os.path.join(self.out["rc2"], "atlas_seg.nii.gz"))
        wl = np.stack([F.warp(self.labels[k], fwd[k], order=0) for k in range(n)])
        np.testing.assert_array_equal(seg, EM.MajorityVote(wl).full())
        # centrality as `python -m evaluation run` computes it from the written files
        mask = np.mean(np.stack(moved) > 0, axis=0) >= 0.5
        cen = F.norm_mm(np.mean(np.stack(fwd).astype(np.float64), 0).astype(np.float32), (1, 1, 1))
        rc = self._rec("rc2")["recenter"]
        self.assertEqual(rc["centrality_final_mask"]["atlas_mask_voxels"], int(mask.sum()))
        self.assertAlmostEqual(rc["centrality_final_mask"]["mean_mm"], float(cen[mask].mean()), places=5)
        f0, _, m0 = self._fields("rc0")
        mask0 = np.mean(np.stack(m0) > 0, axis=0) >= 0.5
        cen0 = F.norm_mm(np.mean(np.stack(f0).astype(np.float64), 0).astype(np.float32), (1, 1, 1))
        self.assertAlmostEqual(rc["centrality_by_round"][0]["mean_mm"], float(cen0[mask0].mean()), places=5)

    def test_pairwise_unchanged(self):
        f0, i0, _ = self._fields("rc0")
        f2, i2, _ = self._fields("rc2")
        worst, mean = 0.0, 0.0
        for i, j in ((0, 1), (1, 2), (2, 0)):
            sm = self.images[j] > 0
            d = F.norm_mm(F.compose(i0[j], f0[i]) - F.compose(i2[j], f2[i]), (1, 1, 1))[sm]
            worst, mean = max(worst, float(np.percentile(d, 99.9))), max(mean, float(d.mean()))
        _say("run pairwise change (subject mask): mean %.5f, p99.9 %.4f mm" % (mean, worst))
        self.assertLess(mean, 0.01)
        self.assertLess(worst, 0.05)

    def test_tta_affine_deform_levels_fine_nonfinite(self):
        def run(ck, name, *extra):
            out = os.path.join(self.tmp, "x_" + name)
            args = CLI.parse_args(["run", "--checkpoint", ck, "--processed-dir", self.proc, "--out", out,
                                   "--check-internal", "1"] + list(extra))
            code = PR.run_predict(args, configure_device=False)
            free_memory()
            with open(os.path.join(out, "predict.json")) as f:
                return code, json.load(f), out
        code, rec, out = run(self.ckpt, "tta", "--branches", "affine+deform", "--tta", "flip8", "--recenter", "1")
        self.assertEqual(code, 0, rec["message"])
        self.assertEqual((rec["recenter"]["status"], rec["tta"]), ("ok", "flip8"))
        self.assertTrue(os.path.isfile(os.path.join(out, "tta_spread.csv")))
        h = [x["mean_mm"] for x in rec["recenter"]["centrality_by_round"]]
        _say("affine+deform flip8 centrality %s" % h)
        self.assertTrue(h[1] < h[0] or h[0] < 1e-5, h)
        self.assertTrue(rec["internal_check"]["passed"])
        # level-2 cascade, --levels 4 (truncated) and --integration-grid fine
        cfg = tiny_config(**STAGE_KW)
        m4 = GroupRegLevelModel(4, cfg).build_model(2)
        randomize(m4, 3, 0.02)
        frozen = CS.FrozenStage.from_model(m4)
        m2 = GroupRegLevelModel(2, cfg, lean_sampler=True, split_group_conv=True).build_model(1)
        m2.copy_weights_from(m4)
        randomize(m2, 4, 0.01)
        ck2 = save_cascade_run(self.tmp, "l2", m2, frozen)
        del m4, m2, frozen
        for name, extra in (("l2lv4", ("--levels", "4")), ("l2fine", ("--integration-grid", "fine"))):
            code, rec, _ = run(ck2, name, "--branches", "deform", "--recenter", "1", *extra)
            self.assertEqual(code, 0, rec["message"])
            self.assertEqual(rec["recenter"]["status"], "ok")
            h = [x["mean_mm"] for x in rec["recenter"]["centrality_by_round"]]
            _say("level-2 %s centrality %s" % (name, h))
            self.assertTrue(h[1] < h[0] or h[0] < 1e-5, h)
        # non-finite model outputs: re-centering skipped, uncorrected fields written, exit 3
        orig = CN.to_canonical

        def nan_fwd(*a, **k):
            fwd, inv = orig(*a, **k)
            f = fwd.numpy()
            f[0, 50, 60, 70, 1] = np.nan
            return tf.constant(f), inv
        with mock.patch.object(CN, "to_canonical", side_effect=nan_fwd):
            code, rec, out = run(self.ckpt, "nan", "--branches", "deform", "--recenter", "2", "--check-internal", "0")
        self.assertEqual(code, CLI.EXIT_NONFINITE)
        self.assertEqual(rec["recenter"]["status"], "skipped_nonfinite")
        self.assertFalse(os.path.exists(os.path.join(out, "atlas.nii.gz")))
        self.assertFalse(os.path.exists(os.path.join(out, "_recenter_tmp")))
        for s in self.sdirs:
            self.assertTrue(os.path.isfile(os.path.join(out, s + "_fwd.nii.gz")))


if __name__ == "__main__":
    unittest.main()
