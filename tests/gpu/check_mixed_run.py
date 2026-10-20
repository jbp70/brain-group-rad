#!/usr/bin/env python
"""Check a finished mixed-task main.py run directory (milestone 5 `dryrun_mixed`, milestone 6
`dryrun_deform`). numpy-free, TF-free (Python 3.8 stdlib only), so it runs inside or outside the
container.

    python tests/gpu/check_mixed_run.py <run_dir> [--tasks rigid,affine] [--steps 20]

Tasks: rigid, affine (linear) and deform. Passes (exit 0) when:
* every task trained: ``train_<task>/loss`` logged and finite, and the logged steps reach --steps;
* every task was validated at the final step: linear ``val_<task>/{center,pair}_rms_mm`` finite,
  the affine set also with its scale/shear/rotation errors; deform
  ``val_deform/{template_dice,pair_dice,neg_jac_pct,inv_consistency_mm,centrality_mm}`` finite;
  with >= 2 linear tasks ``val/mean_pair_rms_mm`` (the default best metric of mixed linear runs)
  equal to the mean of the linear tasks' pair RMS (skipped with fewer linear tasks);
* each step function was traced exactly once (``fit_summary.json`` or ``overfit_summary.json``
  ``trace_counts``) and no step was non-finite;
* ``val_png/<task>_step<final>.png`` exists for every task, and a 'best' checkpoint was written
  in ``ckpt/best`` (only checked when --expect-best, the default); with deform AND a linear task
  also ``ckpt/best_deform`` (the deform best file; ``ckpt/best`` keeps the linear criterion);
* milestone 7, optional: ``--expect-init-weights full|partial``: config.json ``init_weights``
  has that mode; partial: encoder + linear_decoder restored and ``fresh_parts`` exactly
  [deform_decoder] when deform is trained (else none). ``--expect-enc-grad true``: every task
  logged finite ``train_<task>/diag_enc_grad_norm_{raw,eff}``; prints the medians and the
  linear / deform median _eff ratio as an INFO line (never a failure).
Prints one line per criterion and writes nothing.
"""
import argparse
import json
import math
import os
import sys

LINEAR_TASKS = ("rigid", "affine")
DEFORM_TASK = "deform"
AFFINE_KEYS = ("center_scale_pct", "center_shear_pct", "center_qr_rot_deg", "pair_scale_pct",
               "pair_shear_pct", "pair_qr_rot_deg")
DEFORM_VAL_KEYS = ("template_dice", "pair_dice", "neg_jac_pct", "inv_consistency_mm", "centrality_mm")


def _finite(v):
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _has_checkpoint(run_dir, sub, prefix):
    d = os.path.join(run_dir, "ckpt", sub)
    return os.path.isdir(d) and any(n.startswith(prefix + "-") for n in os.listdir(d))


def _median(values):
    v = sorted(float(x) for x in values)
    if not v:
        return float("nan")
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


def _check_init_weights(run_dir, tasks, expect):
    """config.json init_weights record of a warm-started run (milestone 7)."""
    path = os.path.join(run_dir, "config.json")
    iw = None
    if os.path.exists(path):
        with open(path) as f:
            iw = json.load(f).get("init_weights")
    if not iw:
        return [("init_weights", False, "config.json has no init_weights record (expected %s)" % expect)]
    ok = iw.get("mode") == expect
    detail = "mode %s (expected %s)" % (iw.get("mode"), expect)
    if expect == "partial":
        restored = sorted(iw.get("restored_parts") or {})
        fresh = sorted(iw.get("fresh_parts") or {})
        want_fresh = [DEFORM_TASK + "_decoder"] if DEFORM_TASK in tasks else []
        ok = ok and "encoder" in restored and "linear_decoder" in restored and fresh == want_fresh
        detail += ", restored %s, fresh %s (expected %s), config diff %s" % (
            restored, fresh, want_fresh, iw.get("config_diff"))
    return [("init_weights", ok, detail)]


def _check_enc_grad(recs, tasks):
    """train_<task>/diag_enc_grad_norm_{raw,eff} logged and finite for every task; the median
    _eff ratio linear / deform is an info line (never a failure)."""
    res, med = [], {}
    for t in tasks:
        keys = ["train_%s/diag_enc_grad_norm_%s" % (t, k) for k in ("raw", "eff")]
        vals = {k: [r["data"][k] for r in recs if k in r["data"]] for k in keys}
        ok = all(vals[k] and all(_finite(v) for v in vals[k]) for k in keys)
        med[t] = {k.rsplit("_", 1)[1]: _median(vals[k]) for k in keys}
        gn = [r["data"]["train_%s/grad_norm" % t] for r in recs if "train_%s/grad_norm" % t in r["data"]]
        med[t]["grad_norm"] = _median(gn)
        res.append(("enc_grad_%s" % t, ok, "%d/%d logged values, median raw %.4g, eff %.4g, task grad_norm %.4g" % (
            len(vals[keys[0]]), len(vals[keys[1]]), med[t]["raw"], med[t]["eff"], med[t]["grad_norm"])))
    if DEFORM_TASK in tasks:
        d = med[DEFORM_TASK]["eff"]
        ratios = {t: (med[t]["eff"] / d if d and math.isfinite(d) and d > 0 else float("nan"))
                  for t in tasks if t in LINEAR_TASKS}
        if ratios:
            res.append(("enc_grad_ratio", True, "INFO median eff ratio linear/deform %s (target [0.1, 10])"
                        % ", ".join("%s %.3g" % kv for kv in ratios.items())))
    return res


def check(run_dir, tasks, steps, expect_best=True, expect_init_weights=None, expect_enc_grad=False):
    """Returns (ok, [(name, passed, detail)]). ``expect_init_weights`` None (not checked) |
    'none' (not checked) | 'full' | 'partial'; ``expect_enc_grad``: check the milestone-7
    encoder grad-norm diagnostics."""
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
    linear = [t for t in tasks if t in LINEAR_TASKS]
    pairs = []
    for t in tasks:
        if t == DEFORM_TASK:
            keys = ["val_deform/" + k for k in DEFORM_VAL_KEYS]
        else:
            keys = ["val_%s/center_rms_mm" % t, "val_%s/pair_rms_mm" % t]
            if t == "affine":
                keys += ["val_affine/" + k for k in AFFINE_KEYS]
            pairs.append(val.get("val_%s/pair_rms_mm" % t))
        missing = [k for k in keys if not _finite(val.get(k))]
        res.append(("val_%s" % t, not missing, "final-step validation %s" % (
            "complete" if not missing else "missing/non-finite: %s" % missing)))
    mean = val.get("val/mean_pair_rms_mm")
    if len(linear) < 2:
        res.append(("val_mean_pair_rms_mm", True, "skipped (%d linear task(s))" % len(linear)))
    else:
        mean_ok = (_finite(mean) and all(_finite(p) for p in pairs)
                   and abs(float(mean) - sum(pairs) / len(pairs)) < 1e-6)
        res.append(("val_mean_pair_rms_mm", mean_ok,
                    "val/mean_pair_rms_mm = %s, linear task pair RMS %s" % (mean, pairs)))
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
        ok_best = _has_checkpoint(run_dir, "best", "best")
        res.append(("best_checkpoint", ok_best, "ckpt/best has a checkpoint: %s" % ok_best))
        if DEFORM_TASK in tasks and linear:
            ok_bd = _has_checkpoint(run_dir, "best_deform", "best_deform")
            res.append(("best_deform_checkpoint", ok_bd, "ckpt/best_deform has a checkpoint: %s" % ok_bd))
    if expect_init_weights not in (None, "none"):
        res += _check_init_weights(run_dir, tasks, expect_init_weights)
    if expect_enc_grad:
        res += _check_enc_grad(recs, tasks)
    return all(p for _, p, _ in res), res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--tasks", default="rigid,affine")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--expect-best", default="true", choices=["true", "false"])
    ap.add_argument("--expect-init-weights", default="none", choices=["none", "full", "partial"],
                    help="check config.json init_weights (warm start mode); none = not checked")
    ap.add_argument("--expect-enc-grad", default="false", choices=["true", "false"],
                    help="check train_<task>/diag_enc_grad_norm_{raw,eff} (milestone 7)")
    a = ap.parse_args(argv)
    tasks = [t for t in a.tasks.split(",") if t]
    unknown = [t for t in tasks if t not in LINEAR_TASKS + (DEFORM_TASK,)]
    if unknown:
        ap.error("unknown task(s) %s" % unknown)
    ok, res = check(a.run_dir, tasks, a.steps, a.expect_best == "true",
                    expect_init_weights=a.expect_init_weights, expect_enc_grad=a.expect_enc_grad == "true")
    for name, passed, detail in res:
        print("%s %-22s %s" % ("PASS" if passed else "FAIL", name, detail))
    print("check_mixed_run: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
