#!/usr/bin/env python
"""Summarize the per-task encoder gradient norms of main.py runs (milestone 7 loss balance).

    python .agents/tasks/milestone-7/summarize_grad_norms.py <run_dir>... [--min-step K] [--out grad_balance.json]

For every run and task: medians over the logged records of ``train_<task>/diag_enc_grad_norm_raw``
(unscaled loss), ``_eff`` (applied: after loss scale, task weight and global clip) and
``train_<task>/grad_norm`` (pre-clip total norm of the objective), the fraction of the task's
logged steps with grad_norm > clipnorm (clipped), plus the median _eff ratio linear / deform.
Stdlib only (Python 3.8).
"""
import argparse
import json
import math
import os
import sys

LINEAR = ("rigid", "affine")
TASKS = LINEAR + ("deform",)


def median(v):
    v = sorted(float(x) for x in v if x is not None and math.isfinite(float(x)))
    if not v:
        return float("nan")
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


def summarize(run_dir, min_step=0):
    with open(os.path.join(run_dir, "logs", "metrics.jsonl")) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    params = next((r["data"] for r in recs if r.get("step") == "PARAMETER"), {})
    recs = [r["data"] for r in recs if isinstance(r.get("step"), int) and r["step"] > int(min_step)]
    clip = float(params.get("clipnorm") or 0.0)
    out = {"run": os.path.basename(os.path.normpath(run_dir)),
           "loss_scale_mm2": params.get("loss_scale_mm2"), "task_loss_weights": params.get("task_loss_weights"),
           "init_weights": params.get("init_weights"), "lr": params.get("lr"), "clipnorm": clip,
           "min_step": int(min_step), "tasks": {}}
    for t in TASKS:
        raw = [r["train_%s/diag_enc_grad_norm_raw" % t] for r in recs if "train_%s/diag_enc_grad_norm_raw" % t in r]
        if not raw:
            continue
        eff = [r["train_%s/diag_enc_grad_norm_eff" % t] for r in recs if "train_%s/diag_enc_grad_norm_eff" % t in r]
        gn = [r["train_%s/grad_norm" % t] for r in recs if "train_%s/grad_norm" % t in r]
        clipped = sum(1 for g in gn if clip > 0 and float(g) > clip)
        out["tasks"][t] = {"n": len(raw), "enc_raw": median(raw), "enc_eff": median(eff), "grad_norm": median(gn),
                           "clipped_frac": clipped / float(len(gn)) if gn else float("nan")}
        if t in LINEAR:
            c = [r["train_%s/center" % t] for r in recs if "train_%s/center" % t in r]
            out["tasks"][t]["center_rms_mm"] = math.sqrt(max(median(c), 0.0)) if c else float("nan")
        else:
            out["tasks"][t]["soft_dice_tmpl"] = median(r["train_deform/soft_dice_tmpl"] for r in recs
                                                       if "train_deform/soft_dice_tmpl" in r)
    d = out["tasks"].get("deform", {}).get("enc_eff")
    out["eff_ratio_linear_over_deform"] = {
        t: (out["tasks"][t]["enc_eff"] / d if d and math.isfinite(d) and d > 0 else float("nan"))
        for t in LINEAR if t in out["tasks"]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-step", type=int, default=0, help="only records logged after this step")
    a = ap.parse_args(argv)
    rows = [summarize(d, a.min_step) for d in a.run_dirs]
    print("%-22s %6s | %-6s %10s %10s %10s %7s | ratio eff lin/deform" % (
        "run", "S", "task", "enc_raw", "enc_eff", "grad_norm", "clipped"))
    for r in rows:
        for t, v in r["tasks"].items():
            print("%-22s %6s | %-6s %10.4g %10.4g %10.4g %7.2f | %s" % (
                r["run"], r["loss_scale_mm2"], t, v["enc_raw"], v["enc_eff"], v["grad_norm"], v["clipped_frac"],
                "%.3g" % r["eff_ratio_linear_over_deform"][t] if t in r["eff_ratio_linear_over_deform"] else ""))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"runs": rows}, f, indent=1, sort_keys=True)
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
