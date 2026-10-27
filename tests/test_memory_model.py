"""Calibrated memory model, no TF (milestone 9 phase 1, design 4.6; AC7, AC8).

Uses the checked-in `models/memory_calibration.json` and `models/memory_coefficients.json`
(regenerate the latter with `python scripts/memory_model.py fit` after any change to the
calibration file or the inventory; `test_coefficients_file_current` fails otherwise).

Pre-fit pins (all coefficients at their initial values, default sampler, int32 indices host-resident)
follow design 2.3 / 2.5 / 4.6 with two documented corrections (FEAT-002 findings):
  * resident: `runtime.optimizers.build_optimizer` creates Adam slots for every trainable variable,
    so the deform decoder's slots (8 B per parameter) are resident in linear-only records too; the
    design 2.3 linear table is compared after adding exactly that term (tolerance unchanged, 0.05 GiB).
  * the f0d1 LS N4 level-1 pin is the design's own component sum (172 + 55 + 490 + 170 + 384 B =
    1271 B per V1 voxel per image = 8.15 GiB, 33.60 GiB in total); the 33.5 headline of design 2.5
    rounds 8.13 GiB per image. Tolerance unchanged (0.1 GiB).
"""
import copy
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

from data_loading import params as DP
from models import memory_model as MM
from models import width_presets as WP

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB = os.path.join(REPO, "models", "memory_calibration.json")
COEFFS = os.path.join(REPO, "models", "memory_coefficients.json")
DEFORM = {"deform_decoder": True}
LS = {"lean_sampler": True, "lean_losses": False, "split_group_conv": True}
LLS = {"lean_sampler": True, "lean_losses": True, "split_group_conv": True}
V1 = float(np.prod(DP.LEVEL_SHAPE[1]))


def _load_cli():
    spec = importlib.util.spec_from_file_location("memory_model_cli", os.path.join(REPO, "scripts", "memory_model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _l2(n, ck, amp, **kw):
    return MM.Workload(2, "deform", n, DEFORM, ck, amp, 7, cascade="frozen4", **kw)


def _f0d1(n=4, knobs=LS):
    return MM.level1_workload("f0d1", n, 7, knobs)


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calib = MM.load_calibration(CALIB)
        cls.co = MM.load_coefficients(COEFFS)
        cls.pre = MM.default_coefficients()


class TestCalibrationAndFit(_Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        buf = io.StringIO()
        cls.result, cls.failures = MM.fit(cls.calib, out=buf, lomo=True)
        cls.report = buf.getvalue()
        print("\n" + cls.report)

    def test_no_tensorflow(self):
        if "tensorflow" not in sys.modules:          # alone / bgr_preproc38 (stage 1 of run_tests_wsl.sh)
            return
        # Full-suite discovery: other test modules import TF first, so check the module imports.
        import ast
        import inspect
        for src in (inspect.getsource(MM), open(os.path.join(REPO, "scripts", "memory_model.py")).read()):
            tree = ast.parse(src)
            names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
            names += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
            self.assertFalse(any(x.split(".")[0] == "tensorflow" for x in names))

    def test_calibration_schema(self):
        self.assertTrue(MM.validate_calibration(self.calib))
        bad = copy.deepcopy(self.calib)
        del bad["records"][0]["cap_gib"]
        with self.assertRaisesRegex(ValueError, "cap_gib"):
            MM.validate_calibration(bad)
        bad = copy.deepcopy(self.calib)
        bad["records"][1]["status"] = "bogus"
        with self.assertRaisesRegex(ValueError, "status"):
            MM.validate_calibration(bad)
        bad = copy.deepcopy(self.calib)
        bad["records"].append(dict(bad["records"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MM.validate_calibration(bad)
        ids = [r["id"] for r in self.calib["records"]]
        for rid in ("l2_deform_fp32_none_n6", "l2_linear_fp32_n12", "l4_linear_range_n32", "l4_a6000_default_none_n16",
                    "l4_gtx_m6_narrow_none_n6", "l4_gtx_lncc_on_narrow_none_n4"):
            self.assertIn(rid, ids)

    def test_every_point_within_tolerance(self):
        rows = MM.evaluate_records([r for r in self.calib["records"] if r["status"] != "holdout"], self.co)
        bad = [(r["id"], r["status"], r["note"]) for r in rows if not r["good"]]
        self.assertEqual(bad, [])
        ok = [r for r in rows if r["status"] == "ok"]
        self.assertEqual(len(ok), 45)
        self.assertTrue(all(abs(r["err"]) <= 0.15 for r in ok))
        for st in ("oom", "failed_d2"):
            self.assertTrue([r for r in rows if r["status"] == st])
        self.assertEqual(self.failures, [])

    def test_failed_d2_via_build_flag(self):
        r = [x for x in self.calib["records"] if x["id"] == "l2_linear_fp32_n12"][0]
        p = MM.predict(MM.Workload.from_record(r), self.co)
        self.assertTrue(any(f.startswith("build/d2 group concat") for f in p["limit_flags"]), p["limit_flags"])
        self.assertTrue(p["peak_gib"] < r["cap_gib"])  # infeasible only through the limit flag

    def test_leave_one_mode_out_printed(self):
        self.assertIn("Leave-one-mode-out", self.report)
        self.assertEqual(list(self.result["leave_one_mode_out"]), list(MM.LOMO_GROUPS))

    def test_coefficients_file_current(self):
        with open(COEFFS) as fh:
            saved = json.load(fh)
        for k in MM.COEFF_NAMES:
            self.assertAlmostEqual(saved["coefficients"][k], self.result["coefficients"][k], places=5, msg=k)
        self.assertEqual(saved["at_bound"], self.result["at_bound"])
        print("coefficients at a bound: %s" % (self.result["at_bound"] or "none"))

    def test_fake_oom_feasible_fails_fit(self):
        recs = [r for r in self.calib["records"] if r["id"].startswith("l2_deform_amp_both")]
        fake = copy.deepcopy([x for x in self.calib["records"] if x["id"] == "l4_gtx_m6_narrow_none_n2"][0])
        fake["id"], fake["status"] = "fake_oom_feasible", "oom"
        fake.pop("peak_mb")
        calib = {"records": copy.deepcopy(recs) + [fake]}
        tmp = tempfile.mkdtemp(prefix="bgr_mm_")
        try:
            path = os.path.join(tmp, "calib.json")
            with open(path, "w") as fh:
                json.dump(calib, fh)
            buf = io.StringIO()
            _, failures = MM.fit(MM.load_calibration(path), out=buf, lomo=False)
            self.assertEqual([r["id"] for r in failures], ["fake_oom_feasible"])
            self.assertIn("FIT FAILURE: fake_oom_feasible", buf.getvalue())
            cli = _load_cli()
            rc = cli.main(["fit", "--calibration", path, "--out", os.path.join(tmp, "co.json"), "--no-lomo"])
            self.assertEqual(rc, 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_oom_hinge(self):
        self.assertEqual(MM.oom_hinge(0.99 * 6.0, 6.0), 0.0)
        self.assertEqual(MM.oom_hinge(7.0, 6.0), 0.0)
        self.assertAlmostEqual(MM.oom_hinge(4.88, 6.0), (0.98 * 6.0 - 4.88) / 6.0)

    def test_holdout_gate(self):
        gate = MM.holdout_gate(self.calib["records"], self.co)
        if not gate["evaluated"]:
            self.skipTest("no hold-out records in models/memory_calibration.json yet; this gate must run "
                          "(not skip) after the FEAT-004 hold-out measurements (plan item 18)")
        buf = io.StringIO()
        MM._print_rows(gate["rows"], buf, "Hold-out:")
        print("\n" + buf.getvalue())
        print("one-sided gate passed %s, worst under-prediction %.1f %%, symmetric passed %s (max |err| "
              "%.1f %%, %d over +20 %%)" % (gate["passed"], 100 * gate["max_under_err"], gate["symmetric_passed"],
                                            100 * gate["max_abs_err"], gate["n_over"]))
        # One-sided gate (FEAT-004 deviation from design 4.3): only under-predictions are gated.
        self.assertTrue(gate["passed"], buf.getvalue())
        self.assertLessEqual(gate["max_under_err"], MM.HOLDOUT_GATE)
        self.assertAlmostEqual(gate["margin"], max(0.15, gate["max_under_err"]))

    def test_holdout_gate_one_sided_rule(self):
        """Synthetic rows: +30 % passes (flagged), -25 % fails, margin from the under-prediction."""
        base = [r for r in self.calib["records"] if r["status"] == "ok" and r["gpu"] == "gtx1080"][0]
        pred = MM.predict(MM.Workload.from_record(base), self.co)["peak_gib"] * 1024.0

        def hold(rid, err):
            return dict(base, id=rid, status="holdout", measured_status="ok", peak_mb=pred / (1.0 + err))

        g = MM.holdout_gate([hold("over", 0.30), hold("under", -0.10)], self.co)
        self.assertTrue(g["passed"])
        self.assertFalse(g["symmetric_passed"])
        self.assertEqual(g["n_over"], 1)
        self.assertAlmostEqual(g["max_under_err"], 0.10, places=6)
        self.assertAlmostEqual(g["margin"], 0.15)
        g = MM.holdout_gate([hold("under", -0.25)], self.co)
        self.assertFalse(g["passed"])
        self.assertAlmostEqual(g["margin"], 0.25, places=6)


class TestMinPairs(_Base):
    """Milestone 9 phase 2 (design-m9-phase2 section 5): the --min-pairs mirror."""

    def test_counts(self):
        self.assertEqual([MM.num_pairs(n, 0.1, "n") for n in (2, 4, 8, 11, 12, 16)], [2, 4, 8, 11, 13, 24])
        self.assertEqual([MM.num_pairs(n, 0.1) for n in (2, 4, 8, 11, 12, 16)], [1, 1, 6, 11, 13, 24])
        with self.assertRaises(ValueError):
            MM.num_pairs(4, 0.1, "x")
        # --max-pairs K (retrain-prereqs 16.3): applied after the floor; 0 = no cap
        self.assertEqual([MM.num_pairs(n, 0.1, "n", 64) for n in (16, 32, 48, 64)], [24, 64, 64, 64])
        self.assertEqual([MM.num_pairs(n, 0.1, "n", 0) for n in (16, 32, 48, 64)], [24, 99, 226, 403])
        self.assertEqual(MM.num_pairs(32, 0.1, "n", 16), 16)
        for bad in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                MM.num_pairs(8, 0.1, "n", bad)

    def test_level1_deform_n4(self):
        w1 = MM.level1_workload("default", 4, 7, LS)
        wn = w1.replace(min_pairs="n")
        self.assertEqual((w1.min_pairs, wn.min_pairs), ("1", "n"))
        p1, pn = MM.predict(w1, self.co), MM.predict(wn, self.co)
        self.assertGreater(pn["phases"]["Fw"], p1["phases"]["Fw"])
        self.assertEqual(wn.to_dict()["min_pairs"], "n")
        self.assertEqual(MM.Workload.from_record(wn.to_dict()).min_pairs, "n")
        old = w1.to_dict()
        del old["min_pairs"]
        self.assertEqual(MM.Workload.from_record(old).min_pairs, "1")
        self.assertEqual(MM.predict(MM.Workload.from_record(old), self.co)["peak_gib"], p1["peak_gib"])
        with self.assertRaises(ValueError):
            w1.replace(min_pairs="x")


class TestDaliTerm(_Base):
    """tf-intensity-aug: DALI device-memory term (calibration job 10517362) and the budgets."""

    def test_calibration_points_and_scaling(self):
        d = MM.dali_memory(1, {"rigid": 8, "affine": 8, "deform": 4}, "dali")
        self.assertAlmostEqual(d["total_gib"], 14.0, places=6)
        for t, want in (("rigid", 5.6), ("affine", 5.6), ("deform", 2.8)):
            self.assertAlmostEqual(d["per_pipeline"][t], want, places=6)
        self.assertAlmostEqual(MM.dali_memory(1, None, "tf")["total_gib"], 4.5, places=6)
        self.assertAlmostEqual(MM.dali_memory(1, None, "off")["total_gib"], 4.5, places=6)
        self.assertAlmostEqual(MM.dali_memory(2, {"rigid": 16, "affine": 16, "deform": 8}, "dali")["total_gib"],
                               3.5, places=6)
        self.assertAlmostEqual(MM.dali_memory(4, {"rigid": 32, "affine": 32, "deform": 16}, "dali")["total_gib"],
                               0.875, places=6)
        self.assertIn("10517362", d["calibration"])
        with self.assertRaises(ValueError):
            MM.dali_memory(1, None, "gpu")
        with self.assertRaises(ValueError):
            MM.dali_memory(1, {"bogus": 2}, "tf")

    def test_predict_and_feasible_budget(self):
        w = MM.level1_workload("default", 2, 7, LS)
        p0 = MM.predict(w, self.co, 0.15)
        p1 = MM.predict(w, self.co, 0.15, dali_gib=0.0)
        for k in ("peak_gib", "feasible_44", "feasible_36", "val_peak_gib", "phases"):
            self.assertEqual(p0[k], p1[k], k)
        self.assertEqual(p0["dali_gib"], 0.0)
        self.assertEqual(p0["total_gib"], p0["peak_gib"])
        m = 0.15
        room = MM.BUDGET_GIB - p0["peak_gib"] * (1 + m)
        self.assertTrue(MM.feasible(p0, m, MM.BUDGET_GIB)[0])
        p2 = MM.predict(w, self.co, m, dali_gib=room + 0.5)
        self.assertEqual(p2["peak_gib"], p0["peak_gib"])
        self.assertAlmostEqual(p2["total_gib"], p2["peak_gib"] + room + 0.5)
        self.assertFalse(MM.feasible(p2, m, MM.BUDGET_GIB)[0])
        self.assertFalse(p2["feasible_44"])
        p3 = MM.predict(w, self.co, m, dali_gib=max(room - 0.5, 0.0))
        self.assertEqual(MM.feasible(p3, m, MM.BUDGET_GIB)[0], room > 0.5 and p0["feasible_44"])

    def test_cli_dali_flag(self):
        cli = _load_cli()
        outs = {}
        for dali in ("none", "tf"):
            tmp = tempfile.mkdtemp(prefix="mm_dali_")
            try:
                path = os.path.join(tmp, "p.json")
                buf = io.StringIO()
                old = sys.stdout
                sys.stdout = buf
                try:
                    rc = cli.main(["predict", "--level", "1", "--n", "4", "--checkpointing", "full", "--amp", "true",
                                   "--lean-sampler", "true", "--split-group-conv", "true", "--cascade", "chain42",
                                   "--dali", dali, "--out", path])
                finally:
                    sys.stdout = old
                self.assertEqual(rc, 0)
                with open(path) as f:
                    outs[dali] = json.load(f)
                if dali == "tf":
                    self.assertIn("DALI 4.50 GiB (aug device tf", buf.getvalue())
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        a, b = outs["none"], outs["tf"]
        self.assertIsNone(a["dali"])
        self.assertAlmostEqual(b["dali_gib"], 4.5, places=6)
        diff = sorted(k for k in a if a[k] != b[k])
        self.assertTrue(set(diff) <= {"dali", "dali_gib", "total_gib", "feasible_44", "feasible_36"}, diff)


class TestPresets(unittest.TestCase):
    def test_uniform_and_counts(self):
        for name, m in WP.UNIFORM_FACTORS.items():
            self.assertEqual(WP.uniform(m), {k: tuple(v) for k, v in WP.PRESETS[name].items()})
        # Built trainable counts (milestone-6 memory.json `params`).
        self.assertEqual(WP.count_params(MM.resolve_config("default"))["total"], 29628638)
        narrow = dict(filters=(16, 32, 32, 48, 64, 96), reductions=(64, 48, 24, 24), dense=(512, 256),
                      deform_decoder=True)
        self.assertEqual(WP.count_params(narrow)["total"], 4032846)
        self.assertEqual(WP.count_params("f0d1"), WP.count_params("f0d1"))


class TestPrefitPins(_Base):
    def test_level2_deform_rows(self):
        pins = [("none", False, {2: 18.5, 3: 26.7, 4: 35.0}), ("integrator", False, {2: 15.1, 4: 29.3, 6: 43.6}),
                ("both", False, {2: 12.5, 3: 18.3, 4: 24.1, 6: 35.8}), ("none", True, {2: 13.8, 4: 25.6, 6: 39.6}),
                ("integrator", True, {2: 10.4, 4: 19.9, 6: 29.4, 8: 38.9}),
                ("both", True, {2: 7.7, 4: 14.5, 6: 21.3, 8: 28.2})]
        for ck, amp, row in pins:
            for n, g in row.items():
                p = MM.predict(_l2(n, ck, amp), self.pre)
                self.assertAlmostEqual(p["peak_gib"], g, delta=0.1, msg="%s amp=%s N%d" % (ck, amp, n))
                self.assertEqual(p["peak_phase"], {"none": "F", "integrator": "I", "both": "D"}[ck])
                self.assertEqual(p["phases"]["Fw"], 0.0)
                self.assertEqual(p["phases"]["Fb"], 0.0)

    def test_level2_linear_table(self):
        parts = WP.count_params(MM.resolve_config("default"))
        corr = 8.0 * parts["deform_decoder"] / MM.GIB   # all-slot resident (module docstring)
        for amp, row in ((False, {4: 5.46, 8: 10.15}), (True, {4: 3.58, 8: 6.35, 12: 9.13, 16: 11.90})):
            for n, g in row.items():
                w = MM.Workload(2, "rigid", n, DEFORM, "none", amp, cascade="frozen4", build_n=n)
                self.assertAlmostEqual(MM.predict(w, self.pre)["peak_gib"], g + corr, delta=0.05, msg="amp=%s N%d" % (amp, n))
        rows = MM.evaluate_records([r for r in self.calib["records"] if r.get("group", "").startswith("linear")], self.co)
        self.assertTrue(all(r["good"] for r in rows))

    def test_level1_f0d1_ls(self):
        p = MM.predict(_f0d1(), self.pre)
        self.assertAlmostEqual(p["peak_gib"], 33.6, delta=0.1)
        self.assertEqual(p["peak_phase"], "D")
        self.assertEqual(p["phase_info"]["D_block"], "d1")
        self.assertAlmostEqual(p["phases"]["Fw"], 32.0, delta=0.2)
        self.assertAlmostEqual(p["phases"]["Fb"], 25.5, delta=0.2)
        self.assertFalse(p["limit_flags"])
        self.assertAlmostEqual(MM.predict(_f0d1(), self.pre, fwd_transient_bound=True)["peak_gib"], 41.0, delta=0.2)
        self.assertAlmostEqual(MM.predict(_f0d1(knobs=LLS), self.pre, fwd_transient_bound=True)["peak_gib"], 35.0,
                               delta=0.3)
        self.assertAlmostEqual(MM.predict(_f0d1(), self.pre, skip_grad_bound=True)["peak_gib"], 34.9, delta=0.1)

    def test_m8_recipe(self):
        p = MM.predict(_l2(8, "both", True), self.co)
        self.assertLessEqual(abs(p["peak_gib"] - 27.8) / 27.8, 0.15)


class TestFwFbRules(_Base):
    def _bytes(self, w, **kw):
        inv = MM._Inventory(w)
        ph, info = MM._deform_phases(inv, self.pre, kw.get("skip", False), kw.get("fwd", False))
        return inv, ph, info

    def test_timed_fw_is_shared_plus_onehot_moment(self):
        w = _f0d1()
        inv, ph, info = self._bytes(w)
        self.assertEqual(info["Fw_call"], "tdice")
        shared = sum(inv.batch.values()) + inv.kept_skips + sum(inv.cascade.values())
        self.assertAlmostEqual(ph["Fw"], shared + MM.lean_fwd_moment(19), delta=1.0)
        self.assertEqual(MM.lean_fwd_moment(19), 936.0)
        p = MM.predict(w, self.pre)
        scale = w.n * V1 / MM.GIB
        self.assertAlmostEqual(p["phases"]["Fw"] - p["resident_gib"], scale * (shared + 936.0), delta=scale)

    def test_bound_and_fb(self):
        for knobs, d_fw, d_fb in ((LS, 554.0, -54.0), (LLS, 698.0, 90.0)):
            w = _f0d1(knobs=knobs)
            _, ph, _ = self._bytes(w)
            _, phb, _ = self._bytes(w, fwd=True)
            self.assertAlmostEqual(phb["Fw"] - ph["F"], d_fw, delta=1.0, msg=str(knobs))
            self.assertAlmostEqual(ph["Fb"] - ph["F"], d_fb, delta=1.0, msg=str(knobs))

    def test_lean_integrator_moments(self):
        self.assertEqual(MM.lean_bwd_moment(3, True), 172.0)
        self.assertEqual(MM.lean_fwd_moment(3), 360.0)
        w = _f0d1()
        inv, ph, _ = self._bytes(w)
        T = w.integration_steps
        alive = sum(inv.batch.values()) + inv.kept_skips
        # Hold-out correction (FEAT-004): under `full` both integrator blocks (exp(v), exp(-v)) are
        # recomputed concurrently, so I holds two blocks.
        self.assertAlmostEqual(ph["I"], alive + 2 * ((T - 1) * 24.0 + 12.0 + max(360.0, 172.0)), places=6)

    def test_no_lean_no_fw_fb(self):
        for ck in ("none", "integrator", "both", "full"):
            p = MM.predict(MM.level1_workload("f0d1", 2, 7, dict(LS, lean_sampler=False), checkpointing=ck), self.pre)
            self.assertEqual((p["phases"]["Fw"], p["phases"]["Fb"]), (0.0, 0.0))


class TestSkipGradients(_Base):
    def test_level2_both_only_stem_skip(self):
        w = _l2(2, "both", False)
        inv = MM._Inventory(w)
        ph, info = MM._deform_phases(inv, self.pre, False, False)
        self.assertEqual(info["D_block"], "d2")
        tr_d2 = [s for s in inv.sub if s[0] == "d2"][0][1]
        base = sum(inv.batch.values()) + inv.enc_bytes + inv.dec_bytes + 2.0 * tr_d2
        self.assertAlmostEqual(ph["D"] - base, 32 * 4.0, places=6)   # d(stem skip), 32 ch fp32
        amp = MM._Inventory(_l2(2, "both", True))
        pha, _ = MM._deform_phases(amp, self.pre, False, False)
        base = sum(amp.batch.values()) + amp.enc_bytes + amp.dec_bytes + 2.0 * tr_d2
        self.assertAlmostEqual(pha["D"] - base, 32 * 2.0, places=6)

    def test_level1_full(self):
        inv = MM._Inventory(_f0d1())
        ph, info = MM._deform_phases(inv, self.pre, False, False)
        self.assertEqual(info["D_block"], "d1")
        tr_d1 = [s for s in inv.sub if s[0] == "d1"][0][1]
        alive = sum(inv.batch.values()) + inv.kept_skips + inv.dec_bytes
        self.assertAlmostEqual(ph["D"], alive + 2.0 * tr_d1, places=6)          # B_sg = 0 at D1
        self.assertAlmostEqual(inv.skip_grads_all, 52.9, delta=0.1)
        self.assertAlmostEqual(ph["E"], sum(inv.batch.values()) + inv.enc_bytes + inv.skip_grads_all
                               + 2.0 * inv.enc_tr, places=6)
        phb, _ = MM._deform_phases(inv, self.pre, True, False)
        self.assertAlmostEqual(phb["D"] - ph["D"], inv.skip_grads_all, places=6)


class TestLimitsAndBuild(_Base):
    def test_f0d1_split(self):
        unsplit = MM.predict(_f0d1(knobs={"lean_sampler": True, "lean_losses": False, "split_group_conv": False}),
                             self.pre)
        self.assertTrue(any("train/d1 group concat" in f and "2642411520" in f for f in unsplit["limit_flags"]),
                        unsplit["limit_flags"])
        self.assertEqual(MM.predict(_f0d1(), self.pre)["limit_flags"], [])

    def test_build_n(self):
        w2 = _f0d1(6)
        self.assertEqual(w2.build_n, 2)
        self.assertEqual(MM.predict(w2, self.pre)["limit_flags"], [])
        w6 = w2.replace(build_n=6)
        flags = MM.predict(w6, self.pre)["limit_flags"]
        self.assertTrue(any(f.startswith("build/d1 group concat") and "3963617280" in f for f in flags), flags)
        self.assertEqual(MM.predict(w2, self.pre)["peak_gib"], MM.predict(w6, self.pre)["peak_gib"])
        self.assertEqual(MM.predict(w2, self.co)["peak_gib"], MM.predict(w6, self.co)["peak_gib"])

    def test_tensor_limits_override(self):
        tmp = tempfile.mkdtemp(prefix="bgr_mm_")
        try:
            path = os.path.join(tmp, "tensor_limit.json")
            with open(path, "w") as fh:
                json.dump({"verified_max_elements": {"fp16": {"fwd": 10 ** 10, "bwd": 10 ** 9}}}, fh)
            lim = MM.load_tensor_limits(path)
            self.assertEqual(lim["fp16"]["bwd"], 10 ** 9)
            self.assertEqual(lim["fp32"], MM.DEFAULT_LIMITS["fp32"])
            self.assertTrue(MM.predict(_f0d1(), self.pre, tensor_limits=lim)["limit_flags"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestResidentAndLncc(_Base):
    def test_resident_parts(self):
        lin = MM.Workload(2, "rigid", 4, DEFORM, "none", True, cascade="frozen4")
        train = lin.replace(tasks_run=MM.TASKS)
        parts = WP.count_params(lin.cfg)
        diff = MM.resident_bytes(train) - MM.resident_bytes(lin)
        self.assertAlmostEqual(diff, parts["deform_decoder"] * (4.0 + 2.0))      # gradient + fp16 cast
        self.assertAlmostEqual(MM.resident_bytes(lin), 12.0 * parts["total"] + 6.0 * (parts["encoder"]
                               + parts["linear_decoder"]) + 4.0 * parts["total"])

    def test_lncc_on_off(self):
        for wd, n in (("default", 2), ("narrow", 2), ("narrow", 3), ("narrow", 4)):
            mc = dict(DEFORM)
            if wd == "narrow":
                mc.update(filters=(16, 32, 32, 48, 64, 96), reductions=(64, 48, 24, 24), dense=(512, 256))
            for c_loss in (0.0, 0.25, 0.5, 0.75, 1.0):
                co = dict(self.pre, c_loss=c_loss)
                on = MM.predict(MM.Workload(4, "deform", n, mc, "none", False, lncc=True, gpu="gtx1080"), co)["peak_gib"]
                off = MM.predict(MM.Workload(4, "deform", n, mc, "none", False, lncc=False, gpu="gtx1080"), co)["peak_gib"]
                self.assertGreaterEqual(on - off, 0.0)
                self.assertLessEqual(on - off, 0.05 * on, msg="%s N%d c_loss %s" % (wd, n, c_loss))


class TestRecommendation(_Base):
    def test_t_rule(self):
        m = 0.15
        self.assertIsNone(MM.choose_t(False, 50, False, 49, True, m))
        self.assertEqual(MM.choose_t(False, 50, True, 30, True, m), 6)       # T6 raises N
        self.assertEqual(MM.choose_t(False, 50, True, 30, False, m), None)   # not allowed
        self.assertEqual(MM.choose_t(True, 35.0, True, 34.0, True, m), 6)    # restores >= 10 % headroom
        self.assertEqual(MM.choose_t(True, 35.0, True, 34.9, True, m), 7)    # T6 headroom still < 10 %
        self.assertEqual(MM.choose_t(True, 30.0, True, 29.0, True, m), 7)    # enough headroom at T7
        self.assertEqual(MM.choose_t(True, 35.0, True, 34.0, False, m), 7)
        ok, msg = MM.read_t6_verdict(None)
        self.assertFalse(ok)
        self.assertIn("excluded", msg)
        tmp = tempfile.mkdtemp(prefix="bgr_mm_")
        try:
            for val, want in ((False, False), (True, True)):
                path = os.path.join(tmp, "v.json")
                with open(path, "w") as fh:
                    json.dump({"t6_accuracy_ok": val}, fh)
                self.assertEqual(MM.read_t6_verdict(path)[0], want)
            self.assertFalse(MM.read_t6_verdict(os.path.join(tmp, "missing.json"))[0])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        rows = MM.candidate_table(self.pre, m, True, widths=["f0d1"], ns=[4])
        row = [r for r in rows if r["knobs"] == LS][0]
        self.assertEqual(row["T"], 7)   # with the lean sampler T does not move the peak (phase D)

    def test_prefit_per_n(self):
        table = MM.per_n_table(MM.candidate_table(self.pre, 0.15, False))
        buf = io.StringIO()
        MM.print_per_n(table, buf, "pre-fit per-N table (timed rule)")
        print("\n" + buf.getvalue())
        self.assertFalse(any(r["feasible"] for r in table if r["n"] >= 5))
        n4 = [r for r in table if r["n"] == 4][0]
        self.assertEqual((n4["widths"], n4["knobs"]), ("f0d1", LS))

    def test_recommend_fields(self):
        buf = io.StringIO()
        rec, code = MM.recommend(self.co, self.calib, n_target=4, out=buf,
                                 time_model=MM.StepTimeModel(self.calib["step_times"]))
        text = buf.getvalue()
        print("\n" + text)
        self.assertIn("Per-N table (timed rule", text)
        self.assertIn("skip-gradient bound", text)
        self.assertIn("order-independent estimate", text)
        self.assertIn("Level-1 grid summary", text)
        if code == 3:
            self.assertIn("R3.7 VERDICT", text)
            return
        self.assertEqual(code, 0)
        self.assertIn("PROVISIONAL", text)
        for k in ("provisional", "margin", "n_target", "coefficients", "widths", "filters", "decoder_widths",
                  "integration_steps", "knobs", "level1", "level2", "level4", "step_s", "level1_long_run_h", "per_n"):
            self.assertIn(k, rec)
        for k in ("amp", "gradient_checkpointing", "lean_sampler", "lean_losses", "split_group_conv"):
            self.assertIn(k, rec["knobs"])
        l1 = rec["level1"]
        for k in ("deform_n", "n_target", "linear_n", "val_pair_chunk", "deform_peak_gib", "deform_peak_margin_gib",
                  "linear_peak_gib", "a100", "deform_peak_fwd_bound_gib", "deform_peak_fwd_bound_margin_gib",
                  "bound_alt_knobs"):
            self.assertIn(k, l1)
        for k in ("deform_n_margin", "deform_n_no_margin", "linear_n", "deform_n_no_margin_fwd_bound"):
            self.assertIn(k, l1["a100"])
        for k in ("recipe", "deform_n", "linear_n", "amp", "gradient_checkpointing", "deform_peak_gib",
                  "largest_deform_n"):
            self.assertIn(k, rec["level2"])
        for k in ("recipe", "deform_n", "linear_n", "amp", "gradient_checkpointing", "deform_peak_gib"):
            self.assertIn(k, rec["level4"])
        self.assertGreaterEqual(l1["deform_n"], 4)
        self.assertLessEqual(l1["deform_peak_margin_gib"], 44.0)
        p = MM.predict(MM.level1_workload(rec["widths"], l1["deform_n"], rec["integration_steps"],
                                          {k: rec["knobs"][k] for k in MM.KNOBS}), self.co, rec["margin"])
        self.assertLessEqual(p["phases"]["Z"] * (1 + rec["margin"]), 44.0)        # frozen stages
        self.assertLessEqual(p["val_by_chunk"][l1["val_pair_chunk"]] * (1 + rec["margin"]), 44.0)  # validation
        json.dumps(rec)
        buf = io.StringIO()
        MM.print_env(rec, buf)
        self.assertIn("EXTRA_ARGS=", buf.getvalue())
        self.assertIn('KNOBS="--lean-sampler', buf.getvalue())

    def test_grid_printed(self):
        buf = io.StringIO()
        res = MM.grid(self.co, out=buf)
        self.assertEqual(len(res["deform"]), len(WP.PRESETS) * 3 * 2 * 2 * 8)
        self.assertEqual(len(res["linear"]), len(WP.PRESETS) * 2 * 2 * 2)
        self.assertIn("Frozen-chain cost per N", buf.getvalue())


class TestStepTimes(_Base):
    def test_m7_chain(self):
        t = {"rigid": 1.0, "affine": 1.0, "deform": 2.0}
        r = {"linear": 0.5, "deform": 0.5}
        keep = MM.m7_consistency(t, r, 0.75)
        self.assertEqual(keep["mode"], "r_task")
        self.assertAlmostEqual(keep["a100"]["deform"], 1.0)
        sw = MM.m7_consistency(t, r, 2.0)
        self.assertEqual(sw["mode"], "scalar")
        self.assertAlmostEqual(sw["s"], 2.0 / 1.5)
        self.assertAlmostEqual(sw["a100"]["deform"], 2.0 * 2.0 / 1.5)
        self.assertEqual(MM.m7_consistency(t, r, 0.75 / 1.19)["mode"], "r_task")
        self.assertEqual(MM.m7_consistency(t, r, 0.75 / 1.21)["mode"], "scalar")

    def test_model_reproduces_its_points(self):
        spec = self.calib["step_times"]
        tm = MM.StepTimeModel(spec)
        for kind in ("linear", "deform"):
            w = MM.Workload.from_record(spec["rtx8000"][kind]["workload"])
            self.assertAlmostEqual(tm.predict(w), spec["rtx8000"][kind]["seconds"], places=6)
            self.assertAlmostEqual(tm.predict(w, "a100"), spec["rtx8000"][kind]["seconds"] * spec["r_task"][kind],
                                   places=6)
        p = MM.predict(_f0d1(), self.co, time_model=tm)
        self.assertGreater(p["step_s"], 0.0)
        self.assertEqual(p["step_s_range"], [0.5 * p["step_s"], 1.5 * p["step_s"]])


if __name__ == "__main__":
    unittest.main()
