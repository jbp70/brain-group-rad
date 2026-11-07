"""Tests of ``predict.py linear-eval`` (TTA design 7.3, T12; plan FEAT-003, D9).

* TestLinearEval: 3 textured phantoms, tiny level-4 run and level-2 cascade run (construction of
  tests/test_predict_tta.TestEndToEnd). rigid / affine at level 4 with --tta none and flip8: exit 0,
  per_subject.csv / per_pair.csv layout, summary.json keys, note, seed, ranges, finite means,
  tta_spread.csv only with flip8; the rendered group is bit-identical across the two arms (sha256
  of `fixed_data.render_group`'s output); level-2 cascade affine flip8 with spread rows at levels 4
  and 2; one subject -> exit 2 with summary.json status error; --tta-levels without flip8 and
  existing outputs without --overwrite -> SystemExit 2.
Measured timings are printed to stderr.
"""
from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory limit)
from tests.predict_helpers import save_cascade_run, save_level4_run, write_subjects
from tests.runtime_helpers import free_memory, tiny_config
from tests.test_cascade import STAGE_KW, randomize
import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

from data_loading import params as P
from inference import cli as CLI
from inference import linear_eval as LE
from inference import tta as TTA
from models import cascade as CS
from models.group_net import GroupRegLevelModel
from runtime import checkpoint as C
from runtime import fixed_data as FD

N_SUBJECTS = 3
SUBJECT_COLS = {"rigid": ["center_rms_mm", "center_rot_deg", "center_trans_mm"],
                "affine": ["center_rms_mm", "center_rot_deg", "center_trans_mm", "center_scale_pct",
                           "center_shear_pct"]}


def _say(msg):
    print("\n[linear-eval] " + msg, file=sys.stderr)


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class TestLinearEval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgr_linear_eval_")
        cls.proc = os.path.join(cls.tmp, "proc")
        write_subjects(cls.proc, N_SUBJECTS, textured=True, seed=21)
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

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        free_memory()

    def _argv(self, ck, out, *extra):
        return ["linear-eval", "--checkpoint", ck, "--processed-dir", self.proc, "--out", out,
                "--store", "ram"] + list(extra)

    def _eval(self, ck, out, *extra):
        """(exit code, summary.json, sha256 of every rendered group, seconds)."""
        args = CLI.parse_args(self._argv(ck, out, *extra))
        hashes = []
        real = FD.render_group

        def recording(*a, **kw):
            res = real(*a, **kw)
            hashes.append(hashlib.sha256(np.ascontiguousarray(res).tobytes()).hexdigest())
            return res
        t0 = time.time()
        with mock.patch.object(FD, "render_group", recording):
            code = LE.run_linear_eval(args, configure_device=False)
        sec = time.time() - t0
        with open(os.path.join(out, "summary.json")) as f:
            rec = json.load(f)
        free_memory()
        return code, rec, hashes, sec

    def _exit2(self, argv):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            CLI.parse_args(argv)
        self.assertEqual(cm.exception.code, 2, err.getvalue())
        return err.getvalue()

    def _check_ok(self, rec, out, task, tta, levels):
        self.assertEqual(rec["status"], "ok", rec["message"])
        self.assertEqual(set(rec), set(LE.LINEAR_EVAL_KEYS))
        self.assertEqual(rec["note"], "errors include the residual alignment of the processed subjects")
        self.assertEqual((rec["task"], rec["seed"], rec["tta"]), (task, 0, tta))
        self.assertEqual(rec["ranges"], dict(P.DEFAULT_LINEAR_RANGES[task]))
        self.assertEqual(rec["n_subjects"], N_SUBJECTS)
        self.assertEqual(rec["nonfinite"], {"A": 0})
        for key in ("center_rms_mm", "pair_rms_mm", "center_rot_deg", "pair_trans_mm", "cycle_rms_mm"):
            self.assertTrue(np.isfinite(rec["means"][key]), key)
            self.assertTrue(np.isfinite(rec["p95"][key]), key)
        if task == "affine":
            for key in ("center_scale_pct", "center_shear_pct", "pair_scale_pct", "pair_shear_pct"):
                self.assertTrue(np.isfinite(rec["means"][key]), key)
        else:
            self.assertNotIn("center_scale_pct", rec["means"])
        for key in ("read", "render", "model", "metrics", "total"):
            self.assertIn(key, rec["timing_s"])
        subj = _read_csv(os.path.join(out, "per_subject.csv"))
        self.assertEqual(len(subj), N_SUBJECTS)
        self.assertEqual(list(subj[0]), ["subject", "subject_dir"] + SUBJECT_COLS[task] + ["pair_rms_mm_mean"])
        self.assertEqual([r["subject"] for r in subj], rec["subjects"])
        pairs = _read_csv(os.path.join(out, "per_pair.csv"))
        self.assertEqual(len(pairs), N_SUBJECTS * (N_SUBJECTS - 1))
        self.assertEqual(list(pairs[0]), ["moving", "fixed"] + [c.replace("center_", "pair_")
                                                                 for c in SUBJECT_COLS[task]])
        self.assertEqual(sorted((r["moving"], r["fixed"]) for r in pairs),
                         sorted((a, b) for a in rec["subjects"] for b in rec["subjects"] if a != b))
        for r in subj + pairs:
            for k, v in r.items():
                if k not in ("subject", "subject_dir", "moving", "fixed"):
                    self.assertTrue(np.isfinite(float(v)), k)
        # per-subject pair mean = mean over the pairs with the subject as moving image
        for r in subj:
            mine = [float(p["pair_rms_mm"]) for p in pairs if p["moving"] == r["subject"]]
            self.assertAlmostEqual(float(r["pair_rms_mm_mean"]), float(np.mean(mine)), places=5)
        self.assertAlmostEqual(rec["means"]["pair_rms_mm"],
                               float(np.mean([float(p["pair_rms_mm"]) for p in pairs])), places=5)
        spread = os.path.join(out, "tta_spread.csv")
        if tta == "none":
            self.assertIsNone(rec["tta_record"])
            self.assertFalse(os.path.exists(spread))
            self.assertEqual(rec["outputs"], ["per_subject.csv", "per_pair.csv", "summary.json"])
        else:
            tr = rec["tta_record"]
            self.assertEqual((tr["mode"], tr["levels"]), ("flip8", list(levels)))
            self.assertEqual(tr["flips"], [list(f) for f in TTA.FLIPS_8])
            self.assertEqual(sorted(tr["spread"]), sorted("l%d/%s" % (lv, task) for lv in levels))
            self.assertIn("tta_spread.csv", rec["outputs"])
            rows = _read_csv(spread)
            self.assertEqual(len(rows), N_SUBJECTS * len(levels))
            self.assertEqual(sorted((r["subject"], r["level"]) for r in rows),
                             sorted((s, str(lv)) for s in rec["subjects"] for lv in levels))
            self.assertEqual({r["task"] for r in rows}, {task})
            for r in rows:
                self.assertTrue(np.isfinite(float(r["spread_rms_mm"])))

    def test_level4_tasks_and_arms(self):
        for task in ("rigid", "affine"):
            got = {}
            for tta in ("none", "flip8"):
                out = os.path.join(self.tmp, "l4_%s_%s" % (task, tta))
                code, rec, hashes, sec = self._eval(self.ck4, out, "--task", task, "--tta", tta)
                self.assertEqual(code, 0, rec["message"])
                self._check_ok(rec, out, task, tta, (4,))
                self.assertEqual(len(hashes), 1)
                got[tta] = hashes[0]
                _say("L4 %s --tta %s: exit 0 in %.1f s (model %.1f s); center %.3f / pair %.3f mm"
                     % (task, tta, sec, rec["timing_s"]["model"], rec["means"]["center_rms_mm"],
                        rec["means"]["pair_rms_mm"]))
            self.assertEqual(got["none"], got["flip8"], "rendered inputs differ across TTA arms")
            _say("L4 %s: rendered group sha256 identical in both arms (%s...)" % (task, got["none"][:12]))

    def test_level2_affine_flip8(self):
        out = os.path.join(self.tmp, "l2_affine_flip8")
        code, rec, _, sec = self._eval(self.ck2, out, "--task", "affine", "--tta", "flip8")
        self.assertEqual(code, 0, rec["message"])
        self._check_ok(rec, out, "affine", "flip8", (4, 2))
        self.assertEqual(rec["tta_record"]["train_flip_prob"], {"4": 0.5, "2": 0.5})
        _say("L2 affine flip8: exit 0 in %.1f s (model %.1f s)" % (sec, rec["timing_s"]["model"]))

    def test_one_subject_exit2(self):
        sf = os.path.join(self.tmp, "one_subject.txt")
        with open(sf, "w") as f:
            f.write("S0\n")
        out = os.path.join(self.tmp, "one")
        args = CLI.parse_args(self._argv(self.ck4, out, "--task", "rigid", "--subjects-file", sf))
        self.assertEqual(LE.run_linear_eval(args, configure_device=False), 2)
        with open(os.path.join(out, "summary.json")) as f:
            rec = json.load(f)
        self.assertEqual(rec["status"], "error")
        self.assertEqual(set(rec), set(LE.LINEAR_EVAL_KEYS))
        self.assertFalse(os.path.exists(os.path.join(out, "per_subject.csv")))

    def test_cli_errors(self):
        out = os.path.join(self.tmp, "cli")
        a = CLI.parse_args(self._argv(self.ck4, out, "--task", "affine"))
        self.assertEqual((a.branches, a.mode, a.tta, a.tta_levels, a.seed), ("affine", "streamed", "none",
                                                                             None, 0))
        self.assertIn("--tta-levels needs --tta flip8",
                      self._exit2(self._argv(self.ck4, out, "--task", "rigid", "--tta-levels", "4")))
        self._exit2(self._argv(self.ck4, out))                                   # --task required
        self._exit2(self._argv(self.ck4, out, "--task", "deform"))
        self._exit2(self._argv(self.ck4, out, "--task", "rigid", "--branches", "rigid"))
        self._exit2(self._argv(self.ck4, out, "--task", "rigid", "--seed", "-1"))
        os.makedirs(out, exist_ok=True)
        for name in ("summary.json", "per_pair.csv"):
            path = os.path.join(out, name)
            open(path, "w").close()
            self.assertIn(name, self._exit2(self._argv(self.ck4, out, "--task", "rigid")))
            self.assertTrue(CLI.parse_args(self._argv(self.ck4, out, "--task", "rigid", "--overwrite")).overwrite)
            os.remove(path)
        open(os.path.join(out, "atlas.nii.gz"), "w").close()     # a `run` output: not protected here
        CLI.parse_args(self._argv(self.ck4, out, "--task", "rigid"))


if __name__ == "__main__":
    unittest.main()
