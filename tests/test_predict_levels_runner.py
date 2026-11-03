"""Runner tests of predict.py level selection (predict-levels design v2, section 11 runner part).

Tiny random-weight cascades (tests/test_predict_runner.py helpers; level 1 on /CPU:0):
1. Default bit-identity: a non-normalised full plan through the general `_linear_sel` /
   `_deform_sel` path == the default path (np.array_equal on A, U1, U1_inv; one streamed runner,
   chunk 1 and N, plan toggled, op determinism on), level 2 and level 1, deform and affine+deform.
2. Truncation (L2 `4`; L1 `4,2`, `4`) and 3. skip (L2 `2`; L1 `2,1`, `1`, `2`, `4,1`): the runner
   outputs vs a direct reference written here from the trainer's own functions
   (`FrozenStage.deform` / `linear`, `FrozenChain.deform` / `deform_stages` / `linear_stages` /
   `deform_skip` / `linear_skip`, `cascade.skip_fields` / `skip_linear` with tf.constant(True),
   `cascade.prewarp_*`, `model(..., training=False)`, `diffeo.compose_cascade`,
   `cascade.compose_linear`, `diffeo.upsample_displacement`): unchunked <= 1e-5, streamed within
   S_stream.
4. Streamed == unchunked (`RunnerCase.compare_modes` with the plan) for every combination.
5. Zero-field sanity: the L1 `1` pre-warp is warp(x1, 0) and the result differs from the default.
Measured maxima are printed to stderr.
"""
import gc
import sys
import time
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory limit)
from tests.test_predict_runner import (B_EXACT_FALLBACK, B_STREAM_A, B_STREAM_FIELD, FIELD_KEYS, N,
                                       RunnerCase, _collect, _diff, _full_res, _level1, _level2,
                                       _note, _print)
from tests.test_predict_streaming import _images
from tests.test_trainer import _OpDeterminism
import numpy as np
import tensorflow as tf

from data_loading import params as P
from inference import levels as LV
from inference.runner import BRANCHES, StreamedRunner, UnchunkedRunner, _pool_to, _up_to_level1
from inference.store import RamStore
from models import cascade as CS
from models import diffeo

LIN_FIELDS = ("U1", "U1_inv")


def _true():
    return tf.constant(True)


def _resampled(x, A):
    return x if A is None else CS.prewarp_linear(x, A, P.LEVEL_SPACING[1])


def _ref_level2(loaded, x1, sel, branch):
    """Direct reference at a level-2 checkpoint (trainer stage-mode level-2 bodies)."""
    frozen, model = loaded.frozen, loaded.model
    x = tf.constant(x1)
    lin, dfm = BRANCHES[branch]
    A, U1, U1i = None, None, None
    if lin is not None:
        a = frozen.linear(_pool_to(x, 2), lin)
        if 4 not in sel:
            a = CS.skip_linear(_true(), a)
        if sel[-1] == 2:
            x2 = _pool_to(x, 2)
            out = model(CS.prewarp_linear(x2, a, P.LEVEL_SPACING[2]), task=lin, training=False)
            a = CS.compose_linear(out["A_raw"], a, model.centering)
        A = a
    if dfm is not None:
        x2 = _pool_to(_resampled(x, A), 2)
        p = frozen.deform(x2)
        U, Ui = p["U"], p["U_inv"]
        if 4 not in sel:
            U, Ui = CS.skip_fields(_true(), [U, Ui])
        if sel[-1] == 2:
            out = model(CS.prewarp_deform(x2, U), task="deform", training=False)
            U, Ui = diffeo.compose_cascade(U, Ui, out["u"], out["u_inv"])
        U1, U1i = _up_to_level1(U, 2), _up_to_level1(Ui, 2)
    return _np(A, U1, U1i)


def _ref_level1(loaded, x1, sel, branch):
    """Direct reference at a level-1 checkpoint (trainer stage-mode level-1 bodies)."""
    chain, model = loaded.frozen, loaded.model
    plan = LV.plan((4, 2, 1), sel)
    sk4, sk2 = 4 in plan.skipped, 2 in plan.skipped
    s4, s2 = (_true() if sk4 else None), (_true() if sk2 else None)
    x = tf.constant(x1)
    lin, dfm = BRANCHES[branch]
    A, U1, U1i = None, None, None
    if lin is not None:
        if plan.last == 4:
            a = chain.linear_stages(x, lin)["A4"]
        elif sk4 or sk2:
            a = chain.linear_skip(x, lin, skip4=s4, skip2=s2)
        else:
            a = chain.linear(x, lin)
        if plan.last == 1:
            out = model(CS.prewarp_linear(x, a, P.LEVEL_SPACING[1]), task=lin, training=False)
            a = CS.compose_linear(out["A_raw"], a, model.centering)
        A = a
    if dfm is not None:
        xs = _resampled(x, A)
        if plan.last == 4:
            st = chain.deform_stages(xs)
            U, Ui = st["U4"], st["U4_inv"]
        elif sk4 or sk2:
            p = chain.deform_skip(xs, skip4=s4, skip2=s2)
            U, Ui = p["U"], p["U_inv"]
        else:
            p = chain.deform(xs)
            U, Ui = p["U"], p["U_inv"]
        if plan.last == 1:
            out = model(CS.prewarp_deform(xs, U), task="deform", training=False)
            U, Ui = diffeo.compose_cascade(U, Ui, out["u"], out["u_inv"])
        U1, U1i = U, Ui
    return _np(A, U1, U1i)


def _np(A, U1, U1i):
    return {k: None if v is None else v.numpy() for k, v in (("A", A), ("U1", U1), ("U1_inv", U1i))}


class LevelsCase(RunnerCase):

    def check_against_ref(self, label, ref, got, exact, worst):
        """Runner result ``got`` vs the direct reference ``ref`` (unchunked: <= 1e-5; streamed:
        S_stream)."""
        for key in ("A", "U1", "U1_inv"):
            if ref[key] is None:
                self.assertNotIn(key, got, "%s %s" % (label, key))
                continue
            mx, _ = _diff(got[key], ref[key])
            _note(worst, ("exact_" if exact else "stream_") + key, mx)
            if exact:
                bound = B_EXACT_FALLBACK
            else:
                bound = B_STREAM_A if key == "A" else B_STREAM_FIELD
            self.assertLessEqual(mx, bound, "%s %s vs reference" % (label, key))

    def selection(self, label, loaded, x1, sel, branches, chunks, ref_fn, worst, default=None,
                  keep=False):
        """compare_modes with the plan, plus streamed and unchunked vs the direct reference, one
        branch at a time (level-1 host memory). ``default`` {branch: {'U1': ...}}: the selection's
        unchunked U1 must differ from it (the selection is active). Returns {branch: unchunked
        arrays} if ``keep``, else {}."""
        plan = LV.plan(loaded.levels, sel)
        self.assertFalse(plan.default)
        fields = FIELD_KEYS if plan.last == loaded.level else LIN_FIELDS
        if loaded.level == 1:
            fields = LIN_FIELDS                      # at level 1 u_ckpt is U1 (compared once)
        res = {}
        total, worst = worst, {}                      # per-selection maxima, merged at the end
        for br in branches:
            ref = ref_fn(loaded, x1, sel, br)

            def on_streamed(b, chunk, got, ref=ref):
                self.check_against_ref("%s %s %s chunk %d" % (label, sel, b, chunk), ref, got,
                                       False, worst)
            out = self.compare_modes("%s %s" % (label, sel), loaded, x1, chunks, (br,), False, worst,
                                     fields=fields, plan=plan, on_streamed=on_streamed, keep_ref=True)
            un = out[br].pop("ref")
            self.check_against_ref("%s %s %s unchunked" % (label, sel, br), ref, un, True, worst)
            if default is not None and br in default:
                d = _diff(un["U1"], default[br]["U1"])[0]
                worst["min_vs_default_U1"] = min(worst.get("min_vs_default_U1", float("inf")), d)
                self.assertGreater(d, 1e-5, "%s %s %s: same as the default" % (label, sel, br))
            if keep:
                res[br] = un
            del ref, un, out
            gc.collect()
        _print("%s --levels %s (%s): exact_* = unchunked vs reference, stream_* = streamed vs "
               "reference, plain keys = streamed vs unchunked" % (label, LV.levels_text(sel),
                                                                  ", ".join(branches)), worst)
        for k, v in worst.items():
            if k.startswith("min_"):
                total[k] = min(total.get(k, float("inf")), v)
            else:
                _note(total, k, v)
        return res

    def default_identity(self, label, loaded, x1, chunks, worst):
        """Test 1(a): plan None vs the non-normalised full plan on the same streamed runner."""
        full = LV.plan(loaded.levels, loaded.levels, normalise=False)
        self.assertFalse(full.default)
        self.assertTrue(LV.plan(loaded.levels, loaded.levels).default)
        self.assertTrue(LV.plan(loaded.levels, None).default)
        defaults = {}
        with _OpDeterminism():
            for chunk in chunks:
                sr = StreamedRunner(loaded, RamStore(), chunks={"l1": chunk, "l2": chunk, "l4": chunk})
                self.assertTrue(sr._is_default())
                for br in ("deform", "affine+deform"):
                    sr.plan = None
                    got_d, sum_d = _collect(sr, x1, br, ("A", "U1", "U1_inv"))
                    sr.plan = full
                    self.assertFalse(sr._is_default())
                    got_s, sum_s = _collect(sr, x1, br, ("A", "U1", "U1_inv"))
                    self.assertEqual(sorted(got_d), sorted(got_s))
                    for key in got_d:
                        same = np.array_equal(got_d[key], got_s[key])
                        _note(worst, "default_vs_full_" + key, _diff(got_d[key], got_s[key])[0])
                        self.assertTrue(same, "%s %s chunk %d %s: general path != default" % (
                            label, br, chunk, key))
                    self.assertEqual(sum_d["chunks"], sum_s["chunks"])
                    self.assertTrue(sum_d["levels"]["default"])
                    self.assertFalse(sum_s["levels"]["default"])
                    defaults[br] = got_d
                    del got_s
                sr.store.close()
        return defaults


class TestLevelsLevel2(LevelsCase):
    """Level-2 cascade on the GPU: default identity, `4` truncation, `2` skip."""

    def setUp(self):
        self.loaded = _level2("float32")
        self.x1 = _full_res(_images(2), 2)

    def test_default_bit_identity(self):
        worst = {}
        self.default_identity("L2", self.loaded, self.x1, (1, N), worst)
        _print("L2 default path vs general path (full plan)", worst)

    def test_truncation_and_skip(self):
        worst = {}
        branches = ("rigid", "deform", "affine+deform")
        default = {br: _collect(UnchunkedRunner(self.loaded), self.x1, br, ("A", "U1"))[0]
                   for br in ("deform", "affine+deform")}
        res4 = self.selection("L2", self.loaded, self.x1, (4,), branches, (1, 2, N), _ref_level2, worst,
                              default=default, keep=True)
        res2 = self.selection("L2", self.loaded, self.x1, (2,), branches, (1, 2, N), _ref_level2, worst,
                              default=default, keep=True)
        # the selections are active: A differs from the default, and `4` from `2`
        for res in (res4, res2):
            self.assertGreater(_diff(res["affine+deform"]["A"], default["affine+deform"]["A"])[0], 1e-4)
        self.assertGreater(_diff(res4["deform"]["U1"], res2["deform"]["U1"])[0], 1e-3)
        _print("L2 truncation 4 / skip 2 (vs reference and streamed vs unchunked)", worst)


class TestLevelsLevel1(LevelsCase):
    """Level-1 FrozenChain + float32 level-1 stage on /CPU:0: every non-default selection."""

    def test_level1_default_bit_identity_gpu(self):
        """Test 1(a) at level 1 on the GPU under op determinism. Not on /CPU:0: there the default
        path itself is not run-to-run bitwise reproducible (two runs of the same default runner
        differ by ~1e-6 vox, measured), so bit identity is only testable on the GPU. Chunk 1 only
        (chunks 2 and N = 3 run out of the 6 GB local GPU limit; level 2 covers chunk N)."""
        if not tf.config.list_physical_devices("GPU"):
            self.skipTest("needs a GPU")
        worst = {}
        loaded = _level1()
        self.default_identity("L1 GPU", loaded, _images(1), (1,), worst)
        _print("L1 GPU default path vs general path (full plan)", worst)

    def test_level1_cpu(self):
        if not tf.config.list_physical_devices("GPU"):
            self.skipTest("tf_gpu env requires a visible GPU (the level-1 case runs on /CPU:0)")
        t0 = time.time()
        worst = {}
        with tf.device("/CPU:0"):
            loaded = _level1()
            x1 = _images(1)
            default = {"deform": _collect(UnchunkedRunner(loaded), x1, "deform", ("U1",))[0]}
            branches = ("deform", "affine+deform")
            # Streamed chunk 2 only (N 3: a full chunk and a remainder); one level-1 streamed run
            # on the CPU takes ~1.5 min.
            for sel in ((4, 2), (4,), (2, 1), (1,), (2,), (4, 1)):
                self.selection("L1", loaded, x1, sel, branches, (2,), _ref_level1, worst, default=default)
                print("[levels] L1 %s done after %.0f s" % (sel, time.time() - t0), file=sys.stderr)
            del default
            # truncation `4` == deform_skip(skip2) (design 2.2: both are up(up(u4)))
            U4s = _ref_level1(loaded, x1, (4,), "deform")["U1"]
            U4 = loaded.frozen.deform_skip(tf.constant(x1), skip2=_true())["U"].numpy()
            _note(worst, "trunc4_vs_skip2", _diff(U4s, U4)[0])
            self.assertLessEqual(_diff(U4s, U4)[0], B_EXACT_FALLBACK)
            del U4s, U4
            # zero-field sanity of `1`: the pre-warp is warp(x1, 0) (lean, as the runner warps)
            sr = StreamedRunner(loaded, RamStore(), chunks={"l1": 2, "l2": 2, "l4": 2},
                                plan=LV.plan(loaded.levels, (1,)))
            pre = sr._prewarp_deform_zero_fn(1)(x1[:2]).numpy()
            ref = diffeo.warp(tf.constant(x1[:2]), tf.zeros((2,) + tuple(P.LEVEL_SHAPE[1]) + (3,)),
                              lean=True).numpy()
            np.testing.assert_array_equal(pre, ref)
            _note(worst, "warp0_vs_x1", _diff(pre, x1[:2])[0])
            sr.store.close()
        _print("L1 CPU level selections", worst, "; total %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    unittest.main()
