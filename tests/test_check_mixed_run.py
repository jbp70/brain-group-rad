"""Unit tests of tests/gpu/check_mixed_run.py (stdlib only, no TF) on fabricated run directories:
rigid+affine (milestone-5 behaviour unchanged), deform-only and rigid+affine+deform (milestone 6).
"""
import importlib.util
import json
import os
import shutil
import tempfile
import unittest

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu", "check_mixed_run.py")
_spec = importlib.util.spec_from_file_location("check_mixed_run", _PATH)
cmr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmr)

STEPS = 20


def linear_val(task, pair):
    d = {"val_%s/center_rms_mm" % task: 1.0, "val_%s/pair_rms_mm" % task: pair}
    if task == "affine":
        d.update({"val_affine/" + k: 0.5 for k in cmr.AFFINE_KEYS})
    return d


def deform_val():
    return {"val_deform/template_dice": 0.8, "val_deform/pair_dice": 0.7, "val_deform/neg_jac_pct": 0.0,
            "val_deform/inv_consistency_mm": 0.05, "val_deform/centrality_mm": 1e-7,
            "val_deform/pair_dice_loss": 0.3}


def make_run(root, tasks, val_overrides=None, drop_pngs=(), best_dirs=None, trace=None, init_weights=None,
             enc_grad=False, enc_grad_bad=None):
    """Fabricated run dir: metrics.jsonl with train/val records, fit_summary.json, PNGs, ckpts;
    optional config.json init_weights record and train_<task>/diag_enc_grad_norm_* keys
    (``enc_grad_bad``: a task whose _eff values are NaN)."""
    os.makedirs(os.path.join(root, "logs"))
    os.makedirs(os.path.join(root, "val_png"))
    recs = []
    for step in range(5, STEPS + 1, 5):
        data = {"train_%s/loss" % t: 1.0 / step for t in tasks}
        if enc_grad:
            for t in tasks:
                eff = float("nan") if t == enc_grad_bad else (2.0 if t == "deform" else 1.0)
                data.update({"train_%s/diag_enc_grad_norm_raw" % t: 10.0, "train_%s/diag_enc_grad_norm_eff" % t: eff,
                             "train_%s/grad_norm" % t: 5.0})
        recs.append({"step": step, "data": data})
    if init_weights is not None:
        with open(os.path.join(root, "config.json"), "w") as f:
            json.dump({"init_weights": init_weights}, f)
    val = {}
    pairs = []
    for t in tasks:
        if t == "deform":
            val.update(deform_val())
        else:
            p = {"rigid": 6.0, "affine": 7.0}[t]
            pairs.append(p)
            val.update(linear_val(t, p))
    if len(pairs) >= 2:
        val["val/mean_pair_rms_mm"] = sum(pairs) / len(pairs)
    val.update(val_overrides or {})
    recs.append({"step": STEPS, "data": val})
    with open(os.path.join(root, "logs", "metrics.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(root, "fit_summary.json"), "w") as f:
        json.dump({"trace_counts": trace or {t: 1 for t in tasks}, "nonfinite_total": 0}, f)
    for t in tasks:
        if t not in drop_pngs:
            open(os.path.join(root, "val_png", "%s_step%07d.png" % (t, STEPS)), "wb").close()
    if best_dirs is None:
        best_dirs = {"best": "best"}
        if "deform" in tasks and any(t != "deform" for t in tasks):
            best_dirs["best_deform"] = "best_deform"
    for sub, prefix in best_dirs.items():
        d = os.path.join(root, "ckpt", sub)
        os.makedirs(d)
        open(os.path.join(d, "%s-2.index" % prefix), "w").close()
    return root


class CheckMixedRunCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_check_mixed_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_dir(self, name, tasks, **kw):
        return make_run(os.path.join(self.tmp, name), tasks, **kw)

    def failed(self, res):
        return [n for n, p, _ in res if not p]


class TestLinear(CheckMixedRunCase):
    def test_rigid_affine_pass(self):
        ok, res = cmr.check(self.run_dir("ra", ["rigid", "affine"]), ["rigid", "affine"], STEPS)
        self.assertTrue(ok, self.failed(res))
        self.assertEqual([n for n, _, _ in res],
                         ["final_step", "train_rigid", "train_affine", "val_rigid", "val_affine",
                          "val_mean_pair_rms_mm", "trace_once", "finite_steps", "val_png", "best_checkpoint"])

    def test_rigid_affine_wrong_mean_fails(self):
        d = self.run_dir("ra", ["rigid", "affine"], val_overrides={"val/mean_pair_rms_mm": 1.0})
        ok, res = cmr.check(d, ["rigid", "affine"], STEPS)
        self.assertFalse(ok)
        self.assertEqual(self.failed(res), ["val_mean_pair_rms_mm"])

    def test_missing_affine_metric_fails(self):
        d = self.run_dir("ra", ["rigid", "affine"], val_overrides={"val_affine/pair_shear_pct": float("nan")})
        ok, res = cmr.check(d, ["rigid", "affine"], STEPS)
        self.assertEqual(self.failed(res), ["val_affine"])

    def test_main_exit_code(self):
        d = self.run_dir("ra", ["rigid", "affine"])
        self.assertEqual(cmr.main([d, "--tasks", "rigid,affine", "--steps", str(STEPS)]), 0)
        self.assertEqual(cmr.main([d, "--tasks", "rigid,affine", "--steps", str(STEPS + 10)]), 1)


class TestDeform(CheckMixedRunCase):
    def test_deform_only_pass(self):
        ok, res = cmr.check(self.run_dir("d", ["deform"]), ["deform"], STEPS)
        self.assertTrue(ok, self.failed(res))
        names = [n for n, _, _ in res]
        self.assertIn("val_deform", names)
        self.assertNotIn("best_deform_checkpoint", names)    # deform-only: ckpt/best only

    def test_mixed3_pass(self):
        tasks = ["rigid", "affine", "deform"]
        ok, res = cmr.check(self.run_dir("m3", tasks), tasks, STEPS)
        self.assertTrue(ok, self.failed(res))
        self.assertIn("best_deform_checkpoint", [n for n, _, _ in res])
        self.assertEqual(cmr.main([os.path.join(self.tmp, "m3"), "--tasks", ",".join(tasks)]), 0)

    def test_mixed3_missing_best_deform_fails(self):
        tasks = ["rigid", "affine", "deform"]
        d = self.run_dir("m3", tasks, best_dirs={"best": "best"})
        ok, res = cmr.check(d, tasks, STEPS)
        self.assertFalse(ok)
        self.assertEqual(self.failed(res), ["best_deform_checkpoint"])

    def test_deform_nonfinite_val_and_missing_png_fail(self):
        d = self.run_dir("d", ["deform"], val_overrides={"val_deform/neg_jac_pct": float("nan")},
                         drop_pngs=("deform",))
        ok, res = cmr.check(d, ["deform"], STEPS)
        self.assertFalse(ok)
        self.assertEqual(self.failed(res), ["val_deform", "val_png"])

    def test_rigid_deform_skips_mean(self):
        tasks = ["rigid", "deform"]
        ok, res = cmr.check(self.run_dir("rd", tasks), tasks, STEPS)
        self.assertTrue(ok, self.failed(res))
        detail = dict((n, d) for n, _, d in res)["val_mean_pair_rms_mm"]
        self.assertIn("skipped", detail)

    def test_init_weights_and_enc_grad(self):
        # Milestone 7: optional warm-start and encoder grad-norm checks (default off = old behaviour).
        tasks = ["rigid", "affine", "deform"]
        partial = {"mode": "partial", "restored_parts": {"encoder": 30, "linear_decoder": 20, "centering": 0},
                   "fresh_parts": {"deform_decoder": 40}, "config_diff": {"deform_decoder": [False, True]}}
        d = self.run_dir("m7", tasks, init_weights=partial, enc_grad=True)
        ok, res = cmr.check(d, tasks, STEPS, expect_init_weights="partial", expect_enc_grad=True)
        self.assertTrue(ok, self.failed(res))
        names = [n for n, _, _ in res]
        self.assertEqual(names[-5:], ["init_weights", "enc_grad_rigid", "enc_grad_affine", "enc_grad_deform",
                                      "enc_grad_ratio"])
        self.assertIn("rigid 0.5", dict((n, x) for n, _, x in res)["enc_grad_ratio"])
        self.assertEqual(cmr.main([d, "--tasks", ",".join(tasks), "--expect-init-weights", "partial",
                                   "--expect-enc-grad", "true"]), 0)
        self.assertEqual(cmr.main([d, "--tasks", ",".join(tasks), "--expect-init-weights", "full"]), 1)
        # defaults: neither check runs
        self.assertNotIn("init_weights", [n for n, _, _ in cmr.check(d, tasks, STEPS)[1]])
        # no record / wrong fresh parts / missing or NaN diagnostics fail
        d2 = self.run_dir("m7_none", tasks)
        ok, res = cmr.check(d2, tasks, STEPS, expect_init_weights="partial", expect_enc_grad=True)
        self.assertEqual(self.failed(res), ["init_weights", "enc_grad_rigid", "enc_grad_affine", "enc_grad_deform"])
        d3 = self.run_dir("m7_bad", tasks, init_weights=dict(partial, fresh_parts={}), enc_grad=True,
                          enc_grad_bad="affine")
        ok, res = cmr.check(d3, tasks, STEPS, expect_init_weights="partial", expect_enc_grad=True)
        self.assertEqual(self.failed(res), ["init_weights", "enc_grad_affine"])
        # linear-only partial warm start (deform source): fresh parts must be empty
        lin = ["rigid", "affine"]
        d4 = self.run_dir("lin", lin, init_weights=dict(partial, fresh_parts={}), enc_grad=True)
        ok, res = cmr.check(d4, lin, STEPS, expect_init_weights="partial", expect_enc_grad=True)
        self.assertTrue(ok, self.failed(res))
        self.assertNotIn("enc_grad_ratio", [n for n, _, _ in res])

    def test_retrace_fails(self):
        tasks = ["rigid", "affine", "deform"]
        d = self.run_dir("m3", tasks, trace={"rigid": 1, "affine": 1, "deform": 2})
        ok, res = cmr.check(d, tasks, STEPS)
        self.assertEqual(self.failed(res), ["trace_once"])


if __name__ == "__main__":
    unittest.main()
