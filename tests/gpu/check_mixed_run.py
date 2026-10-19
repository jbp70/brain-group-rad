#!/usr/bin/env python
"""Check a finished mixed-task main.py run directory (milestone 5, `dryrun_mixed`). numpy-free,
TF-free (Python 3.8 stdlib only), so it runs inside or outside the container.

    python tests/gpu/check_mixed_run.py <run_dir> [--tasks rigid,affine] [--steps 20]

Passes (exit 0) when:
* every task trained: ``train_<task>/loss`` logged and finite, and the logged steps reach --steps;
* every task was validated at the final step: ``val_<task>/{center,pair}_rms_mm`` finite, the
  affine set also with its scale/shear/rotation errors, and ``val/mean_pair_rms_mm`` (the
  default best metric of mixed runs) equal to the mean of the tasks' pair RMS;
* each step function was traced exactly once (``fit_summary.json`` or ``overfit_summary.json``
  ``trace_counts``) and no step was non-finite;
* ``val_png/<task>_step<final>.png`` exists for every task, and a 'best' checkpoint was written
  (only checked when --expect-best, the default).
Prints one line per criterion and writes nothing.
"""
import argparse
import json
import math
import os
import sys

AFFINE_KEYS = ("center_scale_pct", "center_shear_pct", "center_qr_rot_deg", "pair_scale_pct",
               "pair_shear_pct", "pair_qr_rot_deg")


def _finite(v):
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def check(run_dir, tasks, steps, expect_best=True):
    """Returns (ok, [(name, passed, detail)])."""
    res = []
    with open(os.path.join(run_dir, "logs", "metrics.jsonl")) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    recs = [r for r in recs if isinstance(r.get("step"), int)]
    final = max(r["step"] for r in recs) if recs else 0
    res.append(("final_step", final >= steps, "last logged step %d (expected >= %d)" % (final, steps)))
    for t in tasks:
        losses = [r["data"]["train_%s/loss" % t] for r in recs if "train_%s/loss" % t in r["data"]]
        res.append(("train_%s" % t, bool(losses) and all(_finite(v) for v in losses),
                    "%d logged train_%s/loss values, all finite" % (len(losses), t)))
    val = {}
    for r in recs:
        if r["step"] == final:
            val.update({k: v for k, v in r["data"].items() if k.startswith("val")})
    pairs = []
    for t in tasks:
        keys = ["val_%s/center_rms_mm" % t, "val_%s/pair_rms_mm" % t]
        if t == "affine":
            keys += ["val_affine/" + k for k in AFFINE_KEYS]
        missing = [k for k in keys if not _finite(val.get(k))]
        res.append(("val_%s" % t, not missing, "final-step validation %s" % (
            "complete" if not missing else "missing/non-finite: %s" % missing)))
        pairs.append(val.get("val_%s/pair_rms_mm" % t))
    mean = val.get("val/mean_pair_rms_mm")
    mean_ok = (len(tasks) < 2 or (_finite(mean) and all(_finite(p) for p in pairs)
                                  and abs(float(mean) - sum(pairs) / len(pairs)) < 1e-6))
    res.append(("val_mean_pair_rms_mm", mean_ok, "val/mean_pair_rms_mm = %s, task pair RMS %s" % (mean, pairs)))
    summ = None
    for name in ("fit_summary.json", "overfit_summary.json"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            with open(p) as f:
                summ = json.load(f)
            break
    tc = (summ or {}).get("trace_counts")
    res.append(("trace_once", tc is not None and all(tc.get(t) == 1 for t in tasks),
                "trace counts %s (1 per task)" % tc))
    nf = (summ or {}).get("nonfinite_total", 0)
    res.append(("finite_steps", nf == 0, "non-finite steps: %s" % nf))
    pngs = ["%s_step%07d.png" % (t, final) for t in tasks]
    have = set(os.listdir(os.path.join(run_dir, "val_png"))) if os.path.isdir(os.path.join(run_dir, "val_png")) else set()
    res.append(("val_png", all(p in have for p in pngs), "PNGs %s present: %s" % (pngs, [p in have for p in pngs])))
    if expect_best:
        best = os.path.join(run_dir, "ckpt", "best")
        ok_best = os.path.isdir(best) and any(n.startswith("best-") for n in os.listdir(best))
        res.append(("best_checkpoint", ok_best, "ckpt/best has a checkpoint: %s" % ok_best))
    return all(p for _, p, _ in res), res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--tasks", default="rigid,affine")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--expect-best", default="true", choices=["true", "false"])
    a = ap.parse_args(argv)
    tasks = [t for t in a.tasks.split(",") if t]
    ok, res = check(a.run_dir, tasks, a.steps, a.expect_best == "true")
    for name, passed, detail in res:
        print("%s %-22s %s" % ("PASS" if passed else "FAIL", name, detail))
    print("check_mixed_run: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
