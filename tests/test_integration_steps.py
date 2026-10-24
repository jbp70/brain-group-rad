"""Scaling-and-squaring accuracy T = 6 vs T = 7 and the t6_accuracy_ok verdict (milestone 9
design 3.6, R2.3 Option B, R3.5).

Fields from `tests.field_helpers.smooth_field` (n = 1), max |v| in {2, 4, 8} voxels, on the
level-4 grid (40x56x48, sigma 3) and the level-1 grid (160x224x192, sigma 6). Per case: max and
mean |exp(v) o exp(-v) - id| and |exp(-v) o exp(v) - id| over the interior (margin
ceil(max |u|) + 3 voxels, as `test_two_level_round_trip`) and folding_fraction(exp(v)). Only
the documented case is asserted (level 4, max 2, T 7: max < 0.1, mean < 0.01).

Verdict, for max |v| = --max-v (default 8) on the level-1 grid: e(T) = max over the two orders
of the mean interior inverse error; t6_accuracy_ok iff e(6) <= 2 e(7), e(6) <= 0.05 voxel and
folding(T6) <= max(0.1 %, folding(T7) + 0.05 percentage points).

Script mode writes {"t6_accuracy_ok", "cases", "criteria"} as JSON::

    python -m tests.test_integration_steps --max-v 8 --out <json>

Default --out: $BGR_SCRATCH/m9/integration_steps.json, else the repo's scratch/m9/ (local runs
pass --out .agents/tasks/milestone-9/integration_steps.json: the repo's scratch/ is not
git-ignored).
"""
import argparse
import json
import math
import os
import sys
import time
import unittest

from tests import tf_setup  # noqa: F401  (log level, TF32 off, GPU memory growth)
from tests.field_helpers import smooth_field
import numpy as np
import tensorflow as tf

from models import diffeo as D

GRIDS = {"level4": ((40, 56, 48), 3.0), "level1": ((160, 224, 192), 6.0)}
MAX_V = (2.0, 4.0, 8.0)
STEPS = (6, 7)
CRITERIA = {"e6_over_e7_max": 2.0, "e6_max_vox": 0.05, "fold6_abs_max": 0.001, "fold6_over_fold7_add": 0.0005,
            "error": "mean interior inverse error, max over the two orders", "grid": "level1"}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_out():
    root = os.environ.get("BGR_SCRATCH")
    base = os.path.join(root, "m9") if root else os.path.join(REPO, "scratch", "m9")
    return os.path.join(base, "integration_steps.json")


def _norm(u):
    return tf.norm(u, axis=-1)


def run_case(grid, max_v, steps, base):
    """One case: base field (max |v| 1) scaled to max_v, integrated with T = steps."""
    shape, sigma = GRIDS[grid]
    v = tf.constant((base * max_v).astype(np.float32))
    u, u_inv = D.exp_and_inverse(v, steps)
    umax = float(tf.reduce_max(_norm(u)).numpy())
    m = int(math.ceil(umax)) + 3
    out = {"grid": grid, "shape": list(shape), "sigma": sigma, "max_v": max_v, "steps": steps,
           "max_u": umax, "margin": m}
    for name, (a, b) in (("fwd_inv", (u, u_inv)), ("inv_fwd", (u_inv, u))):
        e = _norm(D.compose(a, b))[:, m:-m, m:-m, m:-m]
        out[name + "_max"] = float(tf.reduce_max(e).numpy())
        out[name + "_mean"] = float(tf.reduce_mean(e).numpy())
    out["folding"] = float(tf.reduce_mean(D.folding_fraction(u)).numpy())
    return out


def run_cases(max_vs=MAX_V, grids=tuple(GRIDS), seed=0):
    cases = []
    for g in grids:
        shape, sigma = GRIDS[g]
        base = smooth_field(shape, np.random.default_rng(seed), n=1, sigma=sigma, max_vox=1.0)
        for mv in max_vs:
            for t in STEPS:
                cases.append(run_case(g, float(mv), t, base))
    return cases


def verdict(cases, max_v):
    """t6_accuracy_ok for the level-1 cases at max |v| = max_v."""
    sel = {c["steps"]: c for c in cases if c["grid"] == "level1" and c["max_v"] == float(max_v)}
    e = {t: max(c["fwd_inv_mean"], c["inv_fwd_mean"]) for t, c in sel.items()}
    fold = {t: c["folding"] for t, c in sel.items()}
    ok = (e[6] <= CRITERIA["e6_over_e7_max"] * e[7] and e[6] <= CRITERIA["e6_max_vox"]
          and fold[6] <= max(CRITERIA["fold6_abs_max"], fold[7] + CRITERIA["fold6_over_fold7_add"]))
    return bool(ok), {"e6": e[6], "e7": e[7], "fold6": fold[6], "fold7": fold[7], "max_v": float(max_v)}


def print_table(cases, stream=sys.stderr):
    print("\n[integration steps] interior inverse error (voxels) and folding fraction", file=stream)
    print("  %-7s %5s %2s %7s %10s %10s %10s %10s %9s" % ("grid", "max_v", "T", "max|u|", "fi max",
                                                         "fi mean", "if max", "if mean", "folding"), file=stream)
    for c in cases:
        print("  %-7s %5.1f %2d %7.3f %10.5f %10.6f %10.5f %10.6f %9.2e" % (
            c["grid"], c["max_v"], c["steps"], c["max_u"], c["fwd_inv_max"], c["fwd_inv_mean"],
            c["inv_fwd_max"], c["inv_fwd_mean"], c["folding"]), file=stream)


class TestIntegrationSteps(unittest.TestCase):
    def test_cases_and_verdict(self):
        t0 = time.time()
        cases = run_cases()
        print_table(cases)
        doc = next(c for c in cases if c["grid"] == "level4" and c["max_v"] == 2.0 and c["steps"] == 7)
        for order in ("fwd_inv", "inv_fwd"):
            self.assertLess(doc[order + "_max"], 0.1)
            self.assertLess(doc[order + "_mean"], 0.01)
        self.assertEqual(len(cases), len(GRIDS) * len(MAX_V) * len(STEPS))
        for c in cases:
            for k in ("fwd_inv_max", "fwd_inv_mean", "inv_fwd_max", "inv_fwd_mean", "folding"):
                self.assertTrue(np.isfinite(c[k]), (c, k))
        ok, info = verdict(cases, 8.0)
        print("[integration steps] t6_accuracy_ok (level 1, max|v| 8): %s %s (%.0f s)"
              % (ok, json.dumps(info), time.time() - t0), file=sys.stderr)
        self.assertIsInstance(ok, bool)


def main(argv=None):
    ap = argparse.ArgumentParser(description="T = 6 vs 7 scaling-and-squaring accuracy verdict (design 3.6)")
    ap.add_argument("--max-v", type=float, default=8.0, help="max |v| (voxels) of the verdict, level-1 grid")
    ap.add_argument("--out", default=None, help="JSON path (default %s)" % default_out())
    args = ap.parse_args(argv)
    if not (math.isfinite(args.max_v) and args.max_v > 0):
        ap.error("--max-v must be finite and > 0, got %r" % args.max_v)
    max_vs = tuple(sorted(set(MAX_V) | {float(args.max_v)}))
    cases = run_cases(max_vs)
    print_table(cases, sys.stdout)
    ok, info = verdict(cases, args.max_v)
    out = args.out or default_out()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"t6_accuracy_ok": ok, "verdict": info, "cases": cases, "criteria": CRITERIA}, f, indent=1)
    print("t6_accuracy_ok %s %s -> %s" % (ok, json.dumps(info), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
