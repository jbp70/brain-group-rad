"""Tests of `predict.py run --tta flip8` (TTA design `.agents/tasks/tta/design.md` 3-5, 7, 8;
FEAT-002 of the TTA implementation): engine `StreamedLevel.deform_tta`, runner TTA stages, CLI
flags and predict.json / tta_spread.csv records.

* TestIdentityOnly (T5): `StreamedRunner(..., tta=TTAConfig(flips=IDENTITY_ONLY))` == no TTA,
  bitwise, all four branches, level 4 and the level-2 cascade.
* TestEquivariance (T6): float32 random-weight models: TTA(flip_g x) vs F_g TTA(x) F_g (A, point
  RMS) and (TTA u)^F (U1, U1_inv, u_ckpt, u_inv_ckpt), g = x, y, z, xyz, all four branches,
  level 4 and level 2 with TTA at all levels, N 3 and one N 2 run; the store ends empty and
  'src_affine' is gone after every run.
* TestEquivariantStubs (T7): `StreamedLevel` stubs injected via ``runner._engines``: (a) linear
  stub (intensity centroid / covariance, exactly raw-equivariant), (b) gauge stub B (a) (centered-
  but not raw-equivariant), (c) deform stub alpha grad(I) + c; TTA == single pass; spreads ~ 0;
  gauge > 0 only for (b); v1 raw averaging fails (b) (regression guard); a NaN flip is reported.
* TestCLIAndRecords (T10): flags, exit codes, --overwrite protection of tta_spread.csv, an unrun
  --tta-levels level -> exit 2 with predict.json written.
* TestEndToEnd: `run` on the golden checkpoints (tests/test_predict_tta_golden.py) with --tta
  none (hashes == the pre-TTA fixture) and flip8 (records, tta_spread.csv), --tta-levels 2.
Measured maxima are printed to stderr.
"""
from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory limit)
from tests.predict_helpers import save_cascade_run, save_level4_run, write_subjects
from tests.runtime_helpers import free_memory, tiny_config
from tests.test_cascade import STAGE_KW, randomize
from tests.test_predict_runner import RunnerCase, _collect, _full_res, _level2, _level4
from tests.test_predict_streaming import _images
from tests.test_predict_tta_golden import GOLDEN_PATH, _nifti_hash
from tests.test_trainer import _OpDeterminism
import contextlib
import csv
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np
import tensorflow as tf

from data_loading import params as P
from inference import cli as CLI
from inference import predict as PR
from inference import streaming as ST
from inference import tta as TTA
from inference.runner import BRANCHES, SRC_AFFINE, StreamedRunner, _pool_to
from inference.store import RamStore
from models import cascade as CS
from models import transforms as tfm
from models.group_net import GroupRegLevelModel
from runtime import checkpoint as C

N = 3
TOL_A_MM, TOL_U_VOX = 1e-4, 1e-4                         # T6
TOL_STUB_A_MM, TOL_STUB_PAIR_MM, TOL_STUB_U_VOX = 1e-4, 1e-5, 1e-6   # T7
FLIP_G = {"x": (True, False, False), "y": (False, True, False), "z": (False, False, True),
          "xyz": (True, True, True)}
FIELD_KEYS = ("U1", "U1_inv", "u_ckpt", "u_inv_ckpt")
ALL_KEYS = ("A",) + FIELD_KEYS + ("v", "v_raw")
CHUNKS = {"l1": 2, "l2": 2, "l4": 2}
GOLDEN_BRANCHES = ("rigid", "deform", "affine+deform")


def _say(msg):
    print("\n[tta] " + msg, file=sys.stderr)


def _cfg(levels, flips=TTA.FLIPS_8):
    return TTA.TTAConfig("flip8", levels, flips)


def _second_moments(x1):
    return np.stack([tfm.brain_second_moment(np.asarray(x1[i, ..., 0]) > 0, 1.0)
                     for i in range(x1.shape[0])])


def _point_rms(dM, S):
    dM = np.asarray(dM, np.float64)
    return np.sqrt(np.maximum(np.einsum("...ij,...jk,...ik->...", dM, np.asarray(S, np.float64), dM), 0.0))


def _pair_err(A, B, S):
    """max over ordered pairs i != j of point_rms(A_j^-1 A_i - B_j^-1 B_i, S_i) (float64)."""
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    worst = 0.0
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if i != j:
                d = np.linalg.inv(A[j]) @ A[i] - np.linalg.inv(B[j]) @ B[i]
                worst = max(worst, float(_point_rms(d, S[i])))
    return worst


def _unflip_np(v, flip):
    return TTA.unflip_field(tf.constant(np.asarray(v, np.float32)), flip).numpy()


def _assert_store_clean(case, store, label):
    case.assertNotIn(SRC_AFFINE, store, label)
    case.assertEqual(store.nbytes(), 0, "%s: stores left %s" % (label, store.names()))


# ============================================================================ T5
class TestIdentityOnly(RunnerCase):
    """T5: the identity-only flip list runs today's code: bitwise equal to tta=None."""

    def _run(self, label, loaded):
        x1 = _full_res(_images(2), 2)
        cfg = _cfg(loaded.levels, TTA.IDENTITY_ONLY)
        with _OpDeterminism():
            for br in BRANCHES:
                ref, _ = _collect(StreamedRunner(loaded, RamStore(), chunks=CHUNKS), x1, br, ALL_KEYS)
                st = RamStore()
                got, summ = _collect(StreamedRunner(loaded, st, chunks=CHUNKS, tta=cfg), x1, br, ALL_KEYS)
                self.assertEqual(sorted(got), sorted(ref), "%s %s" % (label, br))
                for key in ref:
                    np.testing.assert_array_equal(got[key], ref[key], "%s %s %s" % (label, br, key))
                self.assertEqual(summ["tta"]["rows"], [])
                _assert_store_clean(self, st, "%s %s" % (label, br))
        _say("T5 %s: identity-only TTA bitwise == no TTA for %s" % (label, ", ".join(BRANCHES)))

    def test_level4(self):
        self._run("L4", _level4("float32"))

    def test_level2(self):
        self._run("L2", _level2("float32"))


# ============================================================================ T6
class TestEquivariance(RunnerCase):
    """T6: TTA(flip_g x) == F_g TTA(x) F_g and (TTA u)^F, float32 random models."""

    def _check(self, label, loaded, x1, branches, gs):
        n = x1.shape[0]
        S = _second_moments(x1)
        worst = {}
        n_rows = {"rigid": 1, "affine": 1, "deform": 1, "affine+deform": 2}
        with _OpDeterminism():
            for br in branches:
                lin, dfm = BRANCHES[br]
                st = RamStore()
                runner = StreamedRunner(loaded, st, chunks=CHUNKS, tta=_cfg(loaded.levels))
                base, summ = _collect(runner, x1, br, ALL_KEYS)
                _assert_store_clean(self, st, "%s %s base" % (label, br))
                self.assertEqual(len(summ["tta"]["rows"]), n * len(loaded.levels) * n_rows[br])
                for g in gs:
                    flip = FLIP_G[g]
                    msg = "%s %s g=%s" % (label, br, g)
                    got, _ = _collect(runner, TTA.flip_np(x1, flip), br, ALL_KEYS)
                    _assert_store_clean(self, st, msg)
                    if lin is not None:
                        F = TTA.flip_matrix_np(flip)
                        want = F @ np.asarray(base["A"], np.float64) @ F
                        err = float(np.max(_point_rms(np.asarray(got["A"], np.float64) - want, F @ S @ F)))
                        worst["A_mm"] = max(worst.get("A_mm", 0.0), err)
                        self.assertLess(err, TOL_A_MM, msg + " A")
                    if dfm is not None:
                        for key in FIELD_KEYS:
                            err = float(np.max(np.abs(got[key] - _unflip_np(base[key], flip))))
                            worst[key] = max(worst.get(key, 0.0), err)
                            self.assertLess(err, TOL_U_VOX, "%s %s" % (msg, key))
                    del got
                del base
                st.close()
        _say("T6 %s N %d: %s" % (label, n, ", ".join("%s %.2e" % kv for kv in sorted(worst.items()))))

    def test_level4(self):
        self._check("L4", _level4("float32"), _full_res(_images(2), 2), tuple(BRANCHES), tuple(FLIP_G))

    def test_level2(self):
        self._check("L2", _level2("float32"), _full_res(_images(2), 2), tuple(BRANCHES), tuple(FLIP_G))

    def test_level4_n2(self):
        self._check("L4", _level4("float32"), _full_res(_images(2, n=2, seed=4), 2),
                    ("affine+deform",), ("xyz", "y"))


# ============================================================================ T7 stubs
def _coords_mm(shape, level):
    s = float(P.LEVEL_SPACING[level])
    return [(np.arange(int(k), dtype=np.float64) - (int(k) - 1) / 2.0) * s for k in shape]


def _rodrigues(axis, deg):
    a = np.asarray(axis, np.float64)
    a = a / np.linalg.norm(a)
    t = np.deg2rad(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def _gauge_matrix(task):
    """The constant non-flip-symmetric left factor B of stub (b) (TTA design T7 (b))."""
    L = _rodrigues((1.0, 0.3, 0.2), 10.0)
    if task == "affine":
        K = np.array([[np.exp(0.05), 0.03, 0.03], [0.0, np.exp(0.05), 0.03], [0.0, 0.0, np.exp(0.05)]])
        L = L @ K
    B = np.eye(4)
    B[:3, :3] = L
    B[:3, 3] = (3.0, -2.0, 4.0)
    return B


class _LinearStub(ST.StreamedLevel):
    """Stub (a): translation = intensity centroid (mm); 10-D Q = diag(0, 1, 1, 1) + eps C-pattern;
    log-scales 0.1 log(C_aa / 900); shears eps2 (C_xy, C_xz, C_yz) (C = intensity covariance, mm^2).
    Exactly raw-equivariant under array reversal. ``gauge``: left factor B (stub (b))."""

    EPS, EPS2 = 2e-4, 1e-4
    gauge = None

    def _params(self, x, task):
        cx, cy, cz = _coords_mm(x.shape[1:4], self.level)
        X = cx[None, :, None, None]
        Y = cy[None, None, :, None]
        Z = cz[None, None, None, :]
        w = x / np.sum(x, axis=(1, 2, 3), keepdims=True)
        m = [np.sum(w * c, axis=(1, 2, 3)) for c in (X, Y, Z)]
        d = [X - m[0][:, None, None, None], Y - m[1][:, None, None, None], Z - m[2][:, None, None, None]]
        C = np.zeros((x.shape[0], 3, 3))
        for a in range(3):
            for b in range(3):
                C[:, a, b] = np.sum(w * d[a] * d[b], axis=(1, 2, 3))
        e = self.EPS
        cxx, cyy, czz, cxy, cxz, cyz = (C[:, 0, 0], C[:, 1, 1], C[:, 2, 2], C[:, 0, 1], C[:, 0, 2],
                                        C[:, 1, 2])
        rot = np.stack([0 * cxx, e * cyz, e * cxz, e * cxy, 1 + e * cxx, e * cxy, e * cxz,
                        1 + e * cyy, e * cyz, 1 + e * czz], axis=1)
        p = np.concatenate([rot, np.stack(m, axis=1)], axis=1)
        if task == "affine":
            ls = 0.1 * np.log(np.stack([cxx, cyy, czz], axis=1) / 900.0)
            sh = self.EPS2 * np.stack([cxy, cxz, cyz], axis=1)
            p = np.concatenate([p, ls, sh], axis=1)
        return p

    def linear(self, n, input_fn, task):
        n = int(n)
        k, bounds = self._bounds(n, task)
        stats = self._new_stats(task, n, k)
        x = np.concatenate([np.asarray(input_fn(i0, i1), np.float64) for i0, i1 in bounds])[..., 0]
        stats["chunks"]["P0"] = list(bounds)
        p = tf.constant(self._params(x, task), tf.float64)
        A = (tfm.rigid_matrix(p) if task == "rigid" else tfm.affine_matrix(p)).numpy()
        if self.gauge is not None:
            A = self.gauge[task][None] @ A
        self.last_stats = self._finish_stats(stats)
        return tf.constant(A.astype(np.float32))


class _GaugeStub(_LinearStub):
    gauge = {"rigid": _gauge_matrix("rigid"), "affine": _gauge_matrix("affine")}


class _DeformStub(ST.StreamedLevel):
    """Stub (c): v_raw = alpha * central-difference gradient of the level input (np.gradient,
    exactly equivariant under array reversal) + a constant non-symmetric field c (deform gauge).
    ``nan_call``: the call (1-based) of `_v_raw_passes` that puts a NaN into v_raw."""

    ALPHA = 2.0
    C = np.array([0.7, -0.4, 0.25])
    nan_call = None

    def _v_raw_passes(self, n, bounds, stats, input_fn):
        self.calls = getattr(self, "calls", 0) + 1
        acc = None
        for i0, i1 in bounds:
            x = np.asarray(input_fn(i0, i1), np.float64)[..., 0]
            g = np.stack(np.gradient(x, axis=(1, 2, 3)), axis=-1)
            v = (self.ALPHA * g + self.C).astype(np.float32)
            if self.nan_call == self.calls and i0 == 0:
                v[0, 1, 1, 1, 0] = np.nan
            self._put("v_raw", n, i0, v)
            s = np.sum(v, axis=0, keepdims=True, dtype=np.float32)
            acc = s if acc is None else acc + s
        stats["chunks"]["P0"] = list(bounds)
        return tf.constant(acc)


class _DeformLinearStub(_DeformStub, _LinearStub):
    """Stub (c) for the deform task, stub (a) for the linear part of affine+deform."""


def _stub_runner(loaded, stub_cls, tta, **attrs):
    st = RamStore()
    runner = StreamedRunner(loaded, st, chunks=CHUNKS, tta=tta)
    for lv in loaded.levels:
        eng = stub_cls(runner.models[lv], st, CHUNKS["l%d" % lv])
        for k, v in attrs.items():
            setattr(eng, k, v)
        runner._engines[lv] = eng
    return runner, st


def _v1_raw_average_A(loaded, x1, task, stub_cls):
    """Version-1 TTA (raw-output averaging) of a level-4 stub, in the test (regression guard)."""
    model = loaded.stage_models()[4]
    eng = stub_cls(model, RamStore(), CHUNKS["l4"])
    x4 = _pool_to(tf.constant(x1), 4).numpy()
    n = x1.shape[0]
    params = []
    for flip in TTA.FLIPS_8:
        a = eng.linear(n, lambda i0, i1, flip=flip: TTA.flip_np(x4[i0:i1], flip), task)
        params.append(TTA.encode_linear(np.asarray(TTA.unflip_matrices(a, flip), np.float64), task))
    p_bar, _ = TTA.mean_linear(params)
    return tf.cast(model.centering(tf.constant(TTA.decode_linear(p_bar, task))), tf.float32).numpy()


class TestEquivariantStubs(RunnerCase):
    """T7: equivariant / gauge stubs: TTA == single pass; gauge visible only for (b)."""

    def _linear_case(self, label, loaded, stub_cls, gauge_expected):
        x1 = _full_res(_images(2), 2)
        S = _second_moments(x1)
        worst = {}
        for task in ("rigid", "affine"):
            single, st1 = _stub_runner(loaded, stub_cls, None)
            A1 = single.run(x1, task)["A"]
            runner, st = _stub_runner(loaded, stub_cls, _cfg(loaded.levels))
            summ = runner.run(x1, task)
            A8 = summ["A"]
            _assert_store_clean(self, st, label)
            err = float(np.max(_point_rms(np.asarray(A8, np.float64) - np.asarray(A1, np.float64), S)))
            pair = _pair_err(A8, A1, S)
            worst["A_mm"] = max(worst.get("A_mm", 0.0), err)
            worst["pair_mm"] = max(worst.get("pair_mm", 0.0), pair)
            self.assertLess(err, TOL_STUB_A_MM, "%s %s A" % (label, task))
            # pairwise 1e-5 mm at the first level; at level 2 the level-2 input is pre-warped with
            # the (rounding-different) level-4 result, so 1e-4 mm (measured 2.7e-5; test log)
            tol_pair = TOL_STUB_PAIR_MM if loaded.level == 4 else 10.0 * TOL_STUB_PAIR_MM
            self.assertLess(pair, tol_pair, "%s %s pairwise" % (label, task))
            tt = summ["tta"]
            self.assertEqual(len(tt["rows"]), x1.shape[0] * len(loaded.levels))
            for r in tt["rows"]:
                self.assertIsNone(r["spread_p99_mm"])
                self.assertLess(r["spread_rms_mm"], 1e-3, "%s %s spread" % (label, task))
                self.assertLess(r["spread_max_mm"], 1e-3)
                self.assertLess(r["shift_mm"], 1e-3)
                self.assertLess(r["rot_deg_rms"], 1e-3)
                self.assertGreater(r["rot_gap"], 0.5)
                self.assertLess(r["readout_dev_deg"], 1e-3)
                if task == "rigid":
                    self.assertIsNone(r["scale_pct_rms"])
                else:
                    self.assertLess(r["scale_pct_rms"], 1e-3)
            for lv in loaded.levels:
                deg, mm = tt["gauge"]["l%d/%s" % (lv, task)]
                worst["gauge_deg_l%d" % lv] = max(worst.get("gauge_deg_l%d" % lv, 0.0), deg)
                if gauge_expected and lv == loaded.levels[0]:
                    self.assertGreater(deg, 1.0, "%s %s gauge" % (label, task))
                    self.assertGreater(mm, 1.0, "%s %s gauge" % (label, task))
                elif not gauge_expected:
                    self.assertLess(deg, 1e-2, "%s %s gauge" % (label, task))
                    self.assertLess(mm, 1e-2, "%s %s gauge" % (label, task))
            self.assertEqual(tt["nonfinite_flips"], {"l%d/%s" % (lv, task): [] for lv in loaded.levels})
            self.assertEqual(sorted(tt["timing_s"]), sorted("l%d/%s" % (lv, task) for lv in loaded.levels))
            if gauge_expected and loaded.level == 4:
                v1 = _v1_raw_average_A(loaded, x1, task, stub_cls)
                v1_pair = _pair_err(v1, A1, S)
                worst["v1_pair_mm"] = max(worst.get("v1_pair_mm", 0.0), v1_pair)
                self.assertGreater(v1_pair, 1e-3, "%s %s: raw averaging should be off" % (label, task))
        _say("T7 %s %s: %s" % (label, stub_cls.__name__,
                                ", ".join("%s %.2e" % kv for kv in sorted(worst.items()))))

    def test_linear_stub_level4(self):
        self._linear_case("L4", _level4("float32"), _LinearStub, False)

    def test_gauge_stub_level4(self):
        self._linear_case("L4", _level4("float32"), _GaugeStub, True)

    def test_gauge_stub_level2(self):
        self._linear_case("L2", _level2("float32"), _GaugeStub, True)

    def _deform_case(self, label, loaded, x1=None):
        x1 = _full_res(_images(2), 2) if x1 is None else x1
        worst = {}
        for br in ("deform", "affine+deform"):
            single, st1 = _stub_runner(loaded, _DeformLinearStub, None)
            runner, st = _stub_runner(loaded, _DeformLinearStub, _cfg(loaded.levels))
            ref, _ = _collect(single, x1, br, ALL_KEYS)
            got, summ = _collect(runner, x1, br, ALL_KEYS)
            _assert_store_clean(self, st, "%s %s" % (label, br))
            # Tolerances (deviation from design T7 '1e-6 vox' for u, recorded in the test log): the
            # SVF v of a first level without a linear pre-warp is the float32 8-term average
            # (~2e-7 vox, C7) -> 1e-6; float32 scaling and squaring and, at level 2 / after the
            # affine pre-warp, the stub's gradient of an input pre-warped with rounding-different
            # upstream results amplify it to ~1e-5 vox -> fields within the T6 tolerance 1e-4.
            for key in FIELD_KEYS + ("v",):
                err = float(np.max(np.abs(got[key] - ref[key])))
                worst[br + "_" + key] = max(worst.get(br + "_" + key, 0.0), err)
                self.assertLess(err, TOL_U_VOX, "%s %s %s" % (label, br, key))
            if loaded.level == 4 and br == "deform":
                err = float(np.max(np.abs(got["v"] - ref["v"])))
                self.assertLess(err, TOL_STUB_U_VOX, "%s %s v" % (label, br))
            rows = [r for r in summ["tta"]["rows"] if r["task"] == "deform"]
            self.assertEqual(len(rows), x1.shape[0] * len(loaded.levels))
            for r in rows:
                for c in ("spread_rms_mm", "spread_p99_mm", "shift_mm"):
                    worst["deform_" + c] = max(worst.get("deform_" + c, 0.0), r[c])
                    self.assertLess(r[c], 1e-4, "%s %s %s" % (label, br, c))
                for c in ("spread_max_mm", "rot_deg_rms", "rot_gap", "readout_dev_deg"):
                    self.assertIsNone(r[c])
            self.assertNotIn("l%d/deform" % loaded.level, summ["tta"]["gauge"])
        _say("T7 %s deform stub: %s" % (label, ", ".join("%s %.2e" % kv for kv in sorted(worst.items()))))

    def test_deform_stub_level4(self):
        self._deform_case("L4", _level4("float32"))

    def test_deform_stub_level2(self):
        self._deform_case("L2", _level2("float32"))


    def test_nonfinite_flip_reported(self):
        loaded = _level4("float32")
        x1 = _full_res(_images(2), 2)
        runner, st = _stub_runner(loaded, _DeformStub, _cfg((4,)), nan_call=4)
        with self.assertLogs("inference.streaming", "WARNING") as logs:
            got, summ = _collect(runner, x1, "deform", ALL_KEYS)
        self.assertEqual(summ["tta"]["nonfinite_flips"], {"l4/deform": [3]})
        self.assertTrue(any("flip 3" in m and "non-finite" in m for m in logs.output), logs.output)
        self.assertFalse(np.all(np.isfinite(got["U1"])))
        _assert_store_clean(self, st, "nan flip")


# ============================================================================ T10
class TestCLIAndRecords(unittest.TestCase):
    """T10: --tta / --tta-levels parsing and validation (TF-free part), exit 2 for an unrun level."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgr_tta_cli_")
        cls.proc = os.path.join(cls.tmp, "proc")
        write_subjects(cls.proc, 2, seed=2)
        cls.ck4, _ = save_level4_run(cls.tmp, "l4", tiny_config(**STAGE_KW), seed=5,
                                     extra_args={"flip_prob": 0.5})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        free_memory()

    def _argv(self, out, *extra):
        return ["run", "--checkpoint", self.ck4, "--processed-dir", self.proc, "--out", out] + list(extra)

    def _exit2(self, argv):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            CLI.parse_args(argv)
        self.assertEqual(cm.exception.code, 2, err.getvalue())
        return err.getvalue()

    def test_parse(self):
        out = os.path.join(self.tmp, "o_parse")
        a = CLI.parse_args(self._argv(out))
        self.assertEqual((a.tta, a.tta_levels), ("none", None))
        a = CLI.parse_args(self._argv(out, "--tta", "flip8"))
        self.assertEqual((a.tta, a.tta_levels), ("flip8", None))
        a = CLI.parse_args(self._argv(out, "--tta", "flip8", "--tta-levels", "2,1"))
        self.assertEqual(a.tta_levels, (2, 1))
        self.assertIn("--tta-levels needs --tta flip8", self._exit2(self._argv(out, "--tta-levels", "4")))
        self.assertIn("--mode streamed", self._exit2(self._argv(out, "--tta", "flip8", "--mode", "unchunked")))
        for bad in (("--tta", "flip"), ("--tta", "flip8", "--tta-levels", "3"),
                    ("--tta", "flip8", "--tta-levels", "2,2"), ("--tta", "flip8", "--tta-levels", "1,2")):
            self._exit2(self._argv(out, *bad))
        a = CLI.parse_args(self._argv(out, "--mode", "unchunked"))
        self.assertEqual(a.tta, "none")
        chk = ["check-streaming", "--checkpoint", self.ck4, "--processed-dir", self.proc,
               "--out", os.path.join(self.tmp, "c.json")]
        self.assertFalse(hasattr(CLI.parse_args(chk), "tta"))
        self._exit2(chk + ["--tta", "flip8"])

    def test_overwrite_protects_tta_spread(self):
        out = os.path.join(self.tmp, "o_csv")
        os.makedirs(out)
        open(os.path.join(out, "tta_spread.csv"), "w").close()
        self.assertIn("tta_spread.csv", self._exit2(self._argv(out, "--tta", "flip8")))
        self.assertTrue(CLI.parse_args(self._argv(out, "--tta", "flip8", "--overwrite")).overwrite)

    def test_unrun_tta_level_exit2_record(self):
        out = os.path.join(self.tmp, "o_bad_level")
        args = CLI.parse_args(self._argv(out, "--tta", "flip8", "--tta-levels", "1", "--branches", "rigid"))
        self.assertEqual(PR.run_predict(args, configure_device=False), 2)
        with open(os.path.join(out, "predict.json")) as f:
            rec = json.load(f)
        self.assertEqual(set(rec), set(PR.PREDICT_KEYS))
        self.assertEqual(rec["status"], "error")
        self.assertEqual(rec["tta"], "flip8")
        self.assertEqual(rec["tta_record"]["mode"], "flip8")
        self.assertEqual(rec["tta_record"]["levels"], [1])
        self.assertIn("not run by the level plan", rec["tta_record"]["error"])
        self.assertIn("--tta-levels 1", rec["message"])
        self.assertFalse(os.path.exists(os.path.join(out, "tta_spread.csv")))


# ============================================================================ end to end
def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class TestEndToEnd(unittest.TestCase):
    """`predict.py run` on the golden checkpoints: --tta none == fixture; flip8 records."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgr_tta_e2e_")
        cls.proc = os.path.join(cls.tmp, "proc")
        # the construction of tests/test_predict_tta_golden._golden_run (same seeds), plus
        # flip_prob 0.5 in the config args and the level-4 source config of the level-2 run
        write_subjects(cls.proc, 3, textured=True, seed=11)
        cfg = tiny_config(**STAGE_KW)
        cls.ck4, _ = save_level4_run(cls.tmp, "l4", cfg, seed=5, extra_args={"flip_prob": 0.5})
        m4 = GroupRegLevelModel(4, cfg).build_model(2)
        randomize(m4, 3, 0.02)
        frozen = CS.FrozenStage.from_model(m4)
        m2 = GroupRegLevelModel(2, cfg, lean_sampler=True, split_group_conv=True).build_model(1)
        m2.copy_weights_from(m4)
        randomize(m2, 4, 0.01)
        cls.ck2 = save_cascade_run(cls.tmp, "l2", m2, frozen, args={"flip_prob": 0.5})
        cfg2 = C.source_config_path(cls.ck2)
        with open(cfg2) as f:
            d = json.load(f)
        d["cascade"]["source_config"] = C.source_config_path(cls.ck4)
        with open(cfg2, "w") as f:
            json.dump(d, f, indent=1)
        del m4, m2, frozen
        free_memory()
        with open(GOLDEN_PATH) as f:
            cls.golden = json.load(f)["hashes"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        free_memory()

    def _predict(self, ck, out, *extra):
        args = CLI.parse_args(["run", "--checkpoint", ck, "--processed-dir", self.proc, "--out", out,
                               "--store", "ram", "--check-internal", "0"] + list(extra))
        t0 = time.time()
        code = PR.run_predict(args, configure_device=False)
        with open(os.path.join(out, "predict.json")) as f:
            rec = json.load(f)
        free_memory()
        return code, rec, time.time() - t0

    def test_tta_none_equals_golden(self):
        n = 0
        with _OpDeterminism():
            for lname, ck in (("l4", self.ck4), ("l2", self.ck2)):
                for b in GOLDEN_BRANCHES:
                    out = os.path.join(self.tmp, "none_%s_%s" % (lname, b.replace("+", "_")))
                    code, rec, sec = self._predict(ck, out, "--branches", b, "--tta", "none")
                    self.assertEqual(code, 0, rec["message"])
                    self.assertEqual(set(rec), set(PR.PREDICT_KEYS))
                    self.assertEqual(rec["tta"], "none")
                    self.assertIsNone(rec["tta_record"])
                    self.assertNotIn("TTA", rec["pipeline"])
                    self.assertFalse(os.path.exists(os.path.join(out, "tta_spread.csv")))
                    self.assertNotIn("tta_spread.csv", rec["outputs"])
                    files = sorted(f for f in os.listdir(out) if f.endswith(".nii.gz"))
                    self.assertTrue(files)
                    for fname in files:
                        key = "%s/%s/%s" % (lname, b, fname)
                        self.assertEqual(_nifti_hash(os.path.join(out, fname)), self.golden[key], key)
                        n += 1
        _say("E2E --tta none: %d NIfTI hashes == pre-TTA fixture" % n)

    def _check_flip8(self, rec, out, levels, tasks):
        self.assertEqual(rec["status"], "ok", rec["message"])
        self.assertEqual(set(rec), set(PR.PREDICT_KEYS))
        self.assertEqual(rec["tta"], "flip8")
        self.assertEqual(rec["nonfinite"]["total"], 0)
        tr = rec["tta_record"]
        self.assertEqual(tr["mode"], "flip8")
        self.assertEqual(tr["levels"], list(levels))
        self.assertEqual(tr["flips"], [list(f) for f in TTA.FLIPS_8])
        self.assertEqual(tr["linear_averaging"], TTA.LINEAR_AVERAGING)
        self.assertEqual(tr["svf_accumulation"], "float32")
        self.assertTrue(rec["pipeline"].endswith(" with 8-flip TTA at levels %s"
                                                 % ",".join(str(lv) for lv in levels)), rec["pipeline"])
        keys = sorted("l%d/%s" % (lv, t) for lv in levels for t in tasks)
        self.assertEqual(sorted(tr["spread"]), keys)
        self.assertEqual(sorted(tr["nonfinite_flips"]), keys)
        self.assertEqual(sorted(tr["timing_s"]), keys)
        for key in keys:
            self.assertEqual(tr["nonfinite_flips"][key], [])
            self.assertEqual(len(tr["timing_s"][key]), 8)
            s = tr["spread"][key]
            for stat in ("mean", "median", "p95", "max"):
                self.assertTrue(np.isfinite(s["spread_rms_mm"][stat]))
            if key.endswith("deform"):
                self.assertNotIn("gauge_deg", s)
                self.assertIn("spread_p99_mm", s)
                self.assertNotIn("spread_max_mm", s)
            else:
                self.assertTrue(np.isfinite(s["gauge_deg"]) and np.isfinite(s["gauge_mm"]))
                self.assertIn("spread_max_mm", s)
                self.assertNotIn("spread_p99_mm", s)
        self.assertIn("tta_spread.csv", rec["outputs"])
        self.assertFalse(any("training flip probability" in w for w in rec["warnings"]), rec["warnings"])
        rows = _read_csv(os.path.join(out, "tta_spread.csv"))
        self.assertEqual(len(rows), 3 * len(levels) * len(tasks))
        self.assertEqual(list(rows[0]), ["subject", "subject_dir", "level", "task"] + list(TTA.SPREAD_COLUMNS))
        self.assertEqual(sorted({(r["subject"], r["level"], r["task"]) for r in rows}),
                         sorted((s, str(lv), t) for s in rec["subjects"] for lv in levels for t in tasks))
        for r in rows:
            self.assertTrue(np.isfinite(float(r["spread_rms_mm"])))
            self.assertTrue(np.isfinite(float(r["shift_mm"])))
            if r["task"] == "deform":
                self.assertEqual(r["spread_max_mm"], "")
                self.assertTrue(np.isfinite(float(r["spread_p99_mm"])))
                for c in ("rot_deg_rms", "trans_mm_rms", "scale_pct_rms", "shear_pct_rms", "rot_gap",
                          "readout_dev_deg"):
                    self.assertEqual(r[c], "", c)
            else:
                self.assertEqual(r["spread_p99_mm"], "")
                for c in ("spread_max_mm", "rot_deg_rms", "trans_mm_rms", "rot_gap", "readout_dev_deg"):
                    self.assertTrue(np.isfinite(float(r[c])), c)
                if r["task"] == "rigid":
                    self.assertEqual((r["scale_pct_rms"], r["shear_pct_rms"]), ("", ""))
                else:
                    self.assertTrue(np.isfinite(float(r["scale_pct_rms"])))
        for f in os.listdir(out):
            if f.endswith(".nii.gz"):
                import nibabel as nib
                self.assertTrue(np.all(np.isfinite(np.asarray(nib.load(os.path.join(out, f)).dataobj))), f)
        return rows

    def test_flip8(self):
        tasks = {"rigid": ("rigid",), "deform": ("deform",), "affine+deform": ("affine", "deform")}
        for lname, ck, levels in (("l4", self.ck4, (4,)), ("l2", self.ck2, (4, 2))):
            for b in GOLDEN_BRANCHES:
                out = os.path.join(self.tmp, "flip8_%s_%s" % (lname, b.replace("+", "_")))
                code, rec, sec = self._predict(ck, out, "--branches", b, "--tta", "flip8")
                self.assertEqual(code, 0, rec["message"])
                self._check_flip8(rec, out, levels, tasks[b])
                self.assertEqual(rec["tta_record"]["train_flip_prob"],
                                 {str(lv): 0.5 for lv in levels})
                spread = rec["tta_record"]["spread"]
                _say("E2E flip8 %s %s: exit 0 in %.0f s; spread_rms_mm max %s" % (
                    lname, b, sec, {k: round(v["spread_rms_mm"]["max"], 4) for k, v in spread.items()}))

    def test_tta_levels_2(self):
        out = os.path.join(self.tmp, "flip8_l2_only2")
        code, rec, sec = self._predict(self.ck2, out, "--branches", "deform", "--tta", "flip8",
                                       "--tta-levels", "2")
        self.assertEqual(code, 0, rec["message"])
        rows = self._check_flip8(rec, out, (2,), ("deform",))
        self.assertEqual({r["level"] for r in rows}, {"2"})
        self.assertEqual(rec["tta_record"]["train_flip_prob"], {"4": 0.5, "2": 0.5})


if __name__ == "__main__":
    unittest.main()
