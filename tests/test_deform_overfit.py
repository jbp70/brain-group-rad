"""Synthetic deformable overfit through main.py (milestone 6, plan item 16).

One fixed synthetic group (`fd.synthetic_deform_group(4, 4, seed 0)`: a label phantom with 8
labels deformed by 4 random centred diffeomorphisms, max |v| 4 level-4 voxels), narrow model
with the deform decoder, Adam 3e-4 constant LR, no flips, no dropout, default deform losses
(template soft Dice + pairwise soft Dice + 0.5 first-order smoothness). It must reach a 10-step
window mean soft template Dice >= 0.91 (test threshold, >= the 0.9 default) with a window
folding fraction < 0.1 % within 1500 steps (exit code 0); then the eval-mode validation of the
same group must reach a template hard Dice >= 0.85 and >= identity + 0.1 (the identity baseline
is the first validation, before training), a pair hard Dice >= identity + 0.1, and folding
< 0.1 %. Reference run (GTX 1080): stop at step 306, eval template / pair Dice 0.971 / 0.950
(identity 0.864 / 0.774), no folding, median step ~1.2 s, ~8 min.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests.runtime_helpers import free_memory, NARROW_ARGS

import main as M

MAX_STEPS = 1500
# Test-only tuning (defaults unchanged): with the default BN momentum 0.99 the soft template Dice
# reached 0.90 at step 189, but the eval-mode moving statistics still lagged (eval template /
# pair hard Dice 0.873 / 0.799 vs train mode 0.966 / 0.941). BN momentum 0.9 lets the moving
# statistics follow the fixed group, and the 0.91 stop threshold (still >= 0.9) adds margin.
BN_MOMENTUM = "0.9"
DICE_THRESHOLD = "0.91"
FLAGS = ["--overfit-one-group", "--overfit-synthetic", "--task-ratio", "deform:1", "--level", "4",
         "--group-size", "4", "--optimizer", "adam", "--lr", "3e-4", "--schedule", "constant",
         "--warmup-steps", "0", "--overfit-max-steps", str(MAX_STEPS), "--val-every", "250",
         "--log-every", "50", "--log-backend", "jsonl", "--quiet", "true", "--seed", "0",
         "--bn-momentum", BN_MOMENTUM, "--overfit-dice-threshold", DICE_THRESHOLD] + NARROW_ARGS


def overfit_report(s):
    """Human-readable curve summary of an overfit_summary.json dict."""
    rows = ["step %4d  loss %.4f  soft tmpl Dice %.4f  soft pair Dice %.4f  fold %.5f" % tuple(c)
            for c in s["deform_curve"] if c[0] == 1 or c[0] % 50 == 0 or c[0] == s["deform_curve"][-1][0]]
    iv = s["initial_val"]
    lines = ["[deform overfit] steps %d, reached %s, window %s, median step %.0f ms, gpu peak %s MB"
             % (s["steps_run"], s["reached"], s["window_deform"], s["median_compute_ms"] or float("nan"),
                s["gpu_peak_mb"]),
             "  identity (initial val): template Dice %.4f, pair Dice %.4f"
             % (iv["val_deform/template_dice"], iv["val_deform/pair_dice"])]
    lines += ["  " + r for r in rows]
    lines.append("[deform overfit val] step, template Dice, pair Dice, neg J %:")
    lines += ["  %4d  %.4f  %.4f  %.4f" % tuple(v) for v in s["deform_val_curve"]]
    return "\n".join(lines)


class TestDeformOverfit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgr_m6_overfit_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        free_memory()

    def test_synthetic_deform_overfit(self):
        res = os.path.join(self.tmp, "results")
        rc = M.main(FLAGS + ["--results", res, "--run-name", "doverfit"])
        with open(os.path.join(res, "doverfit", "overfit_summary.json")) as f:
            s = json.load(f)
        sys.stderr.write("\n" + overfit_report(s) + "\n")
        self.assertTrue(s["reached"], "window %s" % s["window_deform"])
        self.assertEqual(rc, 0)
        self.assertLessEqual(s["steps_run"], MAX_STEPS)
        self.assertEqual(s["trace_counts"], {"deform": 1})
        self.assertGreaterEqual(s["window_deform"]["soft_dice_tmpl"], 0.9)
        self.assertLess(s["window_deform"]["fold_frac"], 0.001)
        iv, fv = s["initial_val"], s["final_val"]
        self.assertAlmostEqual(iv["val_deform/template_dice"], iv["val_deform/template_dice_identity"], places=10)
        self.assertGreaterEqual(fv["val_deform/template_dice"], 0.85)
        self.assertGreaterEqual(fv["val_deform/template_dice"], iv["val_deform/template_dice"] + 0.1)
        self.assertGreaterEqual(fv["val_deform/pair_dice"], iv["val_deform/pair_dice"] + 0.1)
        self.assertLess(fv["val_deform/neg_jac_pct"], 0.1)
        # Eval mode (moving BN statistics) agrees with training-mode BN on the same group.
        self.assertLess(abs(fv["val_deform/template_dice"] - fv["val_deform_trainmode/template_dice"]), 0.02)


if __name__ == "__main__":
    unittest.main()
