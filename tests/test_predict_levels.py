"""TF-free tests of predict.py level selection (predict-levels design v2, section 11 unit part).

`inference.levels`: parse_levels, plan (run / skipped / truncated per checkpoint level and
selection, normalisation of the full set to the default), run_skip_probs (stage mode with string
keys, legacy at levels 2 and 1, absent keys), exposure on the retrained-chain and old-chain
fixtures (design 6.3 table: 0.0225 for `1`, refusals for linear), the args vs cascade.skip_probs
cross-check (test 7); `--levels` / `--allow-untrained-skip` parsing without importing TF.
Runs in the py3.8 TF-free list of tests/run_tests_wsl.sh.
"""
import argparse
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest

from inference import cli as CLI
from inference import levels as LV


def _r1sw():
    """Retrained level-1 run r1sw_lr1e-4_wd1e-1_n8_d4 (cluster facts of the brief)."""
    return {"args": {"level": 1, "cascade_skip_prob": 0.0,
                     "cascade_skip_stage_parsed": {"rigid": {"4": 0.0, "2": 0.15},
                                                   "affine": {"4": 0.0, "2": 0.15},
                                                   "deform": {"4": 0.15, "2": 0.15}}},
            "cascade": {"skip_mode": "stage",
                        "skip_probs": {"rigid": {"4": 0.0, "2": 0.15}, "affine": {"4": 0.0, "2": 0.15},
                                       "deform": {"4": 0.15, "2": 0.15}}}}


def _r2():
    """Its level-2 source r2_s15000_lr3e-4_wd1e-1_n32_d16."""
    return {"args": {"level": 2, "cascade_skip_prob": 0.0,
                     "cascade_skip_stage_parsed": {"rigid": {"4": 0.0}, "affine": {"4": 0.0},
                                                   "deform": {"4": 0.15}}},
            "cascade": {"skip_mode": "stage", "skip_probs": {"rigid": {"4": 0.0}, "affine": {"4": 0.0},
                                                            "deform": {"4": 0.15}}}}


def _old(level):
    """m9_level1_cascade / m8_level2_gradnorm: skip prob 0, no per-stage skip keys."""
    return {"args": {"level": level, "cascade_skip_prob": 0.0}, "cascade": {"skip_prob": 0.0}}


RETRAINED = {1: ("/r/r1sw/config.json", _r1sw()), 2: ("/r/r2/config.json", _r2())}
OLD = {1: ("/r/m9/config.json", _old(1)), 2: ("/r/m8/config.json", _old(2))}


def _status(exp):
    return {(e["task"], e["level"]): (e["status"], e["joint"]) for e in exp["entries"]}


class TestParseAndPlan(unittest.TestCase):

    def test_parse_levels(self):
        self.assertEqual(LV.parse_levels("4,2,1"), (4, 2, 1))
        self.assertEqual(LV.parse_levels(" 2 , 1 "), (2, 1))
        self.assertEqual(LV.parse_levels("1"), (1,))
        for bad in ("", " ", "4,,1", "4,2,", "1,2", "4,4", "3", "all", "4,2,1,1", "x", "2,4"):
            with self.assertRaises(ValueError, msg=bad):
                LV.parse_levels(bad)

    def test_plan_level1(self):
        ck = (4, 2, 1)
        cases = {(4, 2, 1): ((4, 2, 1), (), ()), (4, 2): ((4, 2), (), (1,)), (4,): ((4,), (), (2, 1)),
                 (2, 1): ((2, 1), (4,), ()), (1,): ((1,), (4, 2), ()), (2,): ((2,), (4,), (1,)),
                 (4, 1): ((4, 1), (2,), ())}
        for sel, (run, skipped, trunc) in cases.items():
            p = LV.plan(ck, sel)
            self.assertEqual((p.run, p.skipped, p.truncated, p.last), (run, skipped, trunc, sel[-1]), sel)
            self.assertEqual(p.default, sel == ck, sel)
            for lv in ck:
                kind = "run" if lv in run else ("skipped" if lv in skipped else "truncated")
                self.assertEqual(p.kind(lv), kind, (sel, lv))
            d = p.to_dict()
            self.assertEqual(d["selected"], list(sel))
            self.assertEqual(d["requested"], ",".join(map(str, sel)))
            self.assertEqual(d["upsampled_from_level"], sel[-1])
        self.assertEqual(LV.plan(ck, (1,)).skipped_before(1), (4, 2))
        self.assertEqual(LV.plan(ck, (4, 1)).skipped_before(1), (2,))
        self.assertEqual(LV.plan(ck, (2,)).skipped_before(2), (4,))
        self.assertIn("skipped (identity", LV.plan(ck, (2, 1)).to_dict()["per_level"]["4"])
        self.assertIn("upsampled from level 2", LV.plan(ck, (4, 2)).to_dict()["per_level"]["1"])

    def test_plan_default_and_normalise(self):
        p = LV.plan((4, 2, 1), None)
        self.assertTrue(p.default)
        self.assertIsNone(p.requested)
        self.assertEqual(p.selected, (4, 2, 1))
        self.assertTrue(LV.plan((4, 2, 1), (4, 2, 1)).default)
        full = LV.plan((4, 2, 1), (4, 2, 1), normalise=False)
        self.assertFalse(full.default)
        self.assertEqual((full.skipped, full.truncated), ((), ()))
        self.assertTrue(LV.plan((4, 2), (4, 2)).default)
        self.assertTrue(LV.plan((4,), None).default)

    def test_plan_level_not_in_checkpoint(self):
        with self.assertRaises(ValueError) as cm:
            LV.plan((4, 2), (1,))
        self.assertIn("level-2 run with levels [4, 2]", str(cm.exception))
        with self.assertRaises(ValueError):
            LV.plan((4,), (2,))
        p2 = LV.plan((4, 2), (2,))
        self.assertEqual((p2.skipped, p2.truncated, p2.default), ((4,), (), False))
        p4 = LV.plan((4, 2), (4,))
        self.assertEqual((p4.skipped, p4.truncated), ((), (2,)))


class TestSkipProbs(unittest.TestCase):

    def test_stage_mode_string_keys(self):
        info = LV.run_skip_probs(_r1sw())
        self.assertEqual(info["skip_mode"], "stage")
        self.assertEqual(info["source"], "args.cascade_skip_stage_parsed")
        self.assertEqual(info["probs"]["deform"], {"4": 0.15, "2": 0.15})
        self.assertEqual(info["probs"]["affine"], {"4": 0.0, "2": 0.15})
        self.assertFalse(info["args_vs_record_mismatch"])
        # int keys and a missing task / level -> 0
        cfg = {"args": {"level": 2, "cascade_skip_stage_parsed": {"deform": {4: 0.3}}}}
        info = LV.run_skip_probs(cfg)
        self.assertEqual(info["probs"], {"rigid": {"4": 0.0}, "affine": {"4": 0.0}, "deform": {"4": 0.3}})

    def test_legacy_and_absent(self):
        l2 = LV.run_skip_probs({"args": {"level": 2, "cascade_skip_prob": 0.2}})
        self.assertEqual(l2["skip_mode"], "legacy")
        self.assertEqual(l2["probs"]["rigid"], {"4": 0.2})
        l1 = LV.run_skip_probs({"args": {"level": 1, "cascade_skip_prob": 0.2}})
        self.assertEqual(l1["probs"]["deform"], {"4": 0.0, "2": 0.2})       # legacy drops level 2
        old = LV.run_skip_probs(_old(1))
        self.assertEqual(old["skip_mode"], "off")
        self.assertIn("absent", old["source"])
        self.assertEqual(old["probs"]["deform"], {"4": 0.0, "2": 0.0})

    def test_resume_mismatch_warning(self):
        cfg = _r2()
        cfg["cascade"]["skip_probs"]["deform"]["4"] = 0.3              # record differs from args
        info = LV.run_skip_probs(cfg)
        self.assertTrue(info["args_vs_record_mismatch"])
        self.assertEqual(info["probs"]["deform"]["4"], 0.15)          # decision from args
        p = LV.plan((4, 2), (2,))
        with self.assertLogs("inference.levels", "WARNING") as logs:
            exp = LV.exposure(p, ("deform",), {2: ("/r/r2/config.json", cfg)})
        self.assertTrue(any("differ from cascade.skip_probs" in m for m in logs.output), logs.output)
        self.assertTrue(exp["runs"]["2"]["args_vs_record_mismatch"])
        self.assertEqual(_status(exp), {("deform", 2): ("trained", 0.15)})
        # legacy record mismatch
        leg = {"args": {"level": 2, "cascade_skip_prob": 0.2}, "cascade": {"skip_prob": 0.1}}
        self.assertTrue(LV.run_skip_probs(leg)["args_vs_record_mismatch"])


class TestExposure(unittest.TestCase):

    def _exp(self, sel, tasks, chain=RETRAINED, ck=(4, 2, 1)):
        return LV.exposure(LV.plan(ck, sel), tasks, chain)

    def test_retrained_chain_table(self):
        e = self._exp((2, 1), ("deform", "affine"))
        self.assertEqual(_status(e), {("deform", 2): ("trained", 0.15), ("deform", 1): ("trained", 0.15),
                                      ("affine", 2): ("untrained", 0.0), ("affine", 1): ("untrained", 0.0)})
        ent = {(x["task"], x["level"]): x for x in e["entries"]}
        self.assertEqual(ent[("deform", 2)]["run_config"], "/r/r2/config.json")
        self.assertEqual(ent[("deform", 1)]["run_config"], "/r/r1sw/config.json")
        self.assertAlmostEqual(ent[("deform", 1)]["exact_pattern"], 0.15 * 0.85)
        self.assertAlmostEqual(ent[("deform", 2)]["exact_pattern"], 0.15)
        self.assertEqual([x["task"] for x in LV.bad_entries(e)], ["affine", "affine"])
        one = self._exp((1,), ("deform", "rigid"))
        st = _status(one)
        self.assertEqual(st[("deform", 1)][0], "trained")
        self.assertAlmostEqual(st[("deform", 1)][1], 0.0225)
        self.assertEqual(st[("rigid", 1)], ("untrained", 0.0))
        self.assertAlmostEqual(one["min_joint"]["deform"], 0.0225)
        self.assertEqual(_status(self._exp((2,), ("deform", "affine"))),
                         {("deform", 2): ("trained", 0.15), ("affine", 2): ("untrained", 0.0)})
        st41 = _status(self._exp((4, 1), ("deform", "affine")))
        self.assertEqual(st41, {("deform", 1): ("trained", 0.15), ("affine", 1): ("trained", 0.15)})
        for sel in ((4,), (4, 2), (4, 2, 1)):                      # truncation / default: no entries
            self.assertEqual(self._exp(sel, ("deform", "affine"))["entries"], [], sel)

    def test_old_chain_refused_truncation_allowed(self):
        for sel in ((2, 1), (1,), (2,), (4, 1)):
            e = self._exp(sel, ("deform", "rigid"), chain=OLD)
            self.assertTrue(e["entries"])
            self.assertTrue(all(x["status"] == "untrained" for x in e["entries"]), sel)
        for sel in ((4,), (4, 2)):
            self.assertEqual(self._exp(sel, ("deform",), chain=OLD)["entries"], [])
        e2 = LV.exposure(LV.plan((4, 2), (2,)), ("deform",), {2: OLD[2]})
        self.assertEqual(_status(e2), {("deform", 2): ("untrained", 0.0)})

    def test_unknown_source(self):
        chain = {1: RETRAINED[1], 2: (None, None)}
        e = self._exp((2, 1), ("deform",), chain=chain)
        st = _status(e)
        self.assertEqual(st[("deform", 2)], ("unknown", None))
        self.assertEqual(st[("deform", 1)], ("trained", 0.15))
        self.assertEqual(len(LV.bad_entries(e)), 1)
        self.assertIn("unknown", e["runs"]["2"]["source"])


class TestCLILevels(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_lv_cli_")
        os.makedirs(os.path.join(self.tmp, "ck"))
        os.makedirs(os.path.join(self.tmp, "proc"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _argv(self, cmd, *extra):
        out = os.path.join(self.tmp, "o" if cmd == "run" else "c.json")
        return [cmd, "--checkpoint", os.path.join(self.tmp, "ck"), "--processed-dir",
                os.path.join(self.tmp, "proc"), "--out", out] + list(extra)

    def test_levels_parse(self):
        for cmd in ("run", "check-streaming"):
            a = CLI.parse_args(self._argv(cmd))
            self.assertIsNone(a.levels)
            self.assertFalse(a.allow_untrained_skip)
            a = CLI.parse_args(self._argv(cmd, "--levels", "2, 1", "--allow-untrained-skip"))
            self.assertEqual(a.levels, (2, 1))
            self.assertTrue(a.allow_untrained_skip)
            for bad in ("1,2", "4,,1", "3", "all", ""):
                err = io.StringIO()
                with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                    CLI.parse_args(self._argv(cmd, "--levels", bad))
                self.assertEqual(cm.exception.code, 2, bad)
                self.assertIn("--levels", err.getvalue())
        self.assertNotIn("tensorflow", sys.modules)

    def test_levels_type_error_is_argparse_error(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            CLI._levels("2,4")


if __name__ == "__main__":
    unittest.main()
