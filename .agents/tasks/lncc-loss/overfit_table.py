"""Table of the three LNCC overfit runs (overfit_compare.sh): final eval-mode metrics vs the
identity baseline, folding, train grad-norm window means, median step time, GPU peak.

Usage: python overfit_table.py <results dir with run_A, run_B, run_C>
Prints a markdown table and writes overfit_table.json next to this file.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"A": "Dice-only", "B": "Dice+LNCC 1/1", "C": "LNCC-only"}


def run_row(root, run):
    d = os.path.join(root, "run_" + run)
    with open(os.path.join(d, "overfit_summary.json")) as f:
        s = json.load(f)
    with open(os.path.join(d, "logs", "metrics.jsonl")) as f:
        recs = [json.loads(line) for line in f]
    train = [r["data"] for r in recs[1:] if "train_deform/grad_norm" in r["data"]]
    fv, iv = s["final_val"], s["initial_val"]
    p = "val_deform/"
    g = [r["train_deform/grad_norm"] for r in train]
    return {"run": run, "name": NAMES[run], "steps": s["steps_run"],
            "template_dice": fv[p + "template_dice"], "template_dice_identity": iv[p + "template_dice"],
            "pair_dice": fv[p + "pair_dice"], "pair_dice_identity": iv[p + "pair_dice"],
            "ncc_template": fv[p + "ncc_template"], "ncc_template_identity": fv[p + "ncc_template_identity"],
            "ncc_pair": fv[p + "ncc_pair"], "ncc_pair_identity": fv[p + "ncc_pair_identity"],
            "neg_jac_pct": fv[p + "neg_jac_pct"], "trainmode_template_dice": fv[p.rstrip("/") + "_trainmode/template_dice"],
            "grad_norm_first_window": g[0] if g else None, "grad_norm_last_window": g[-1] if g else None,
            "grad_norm_mean": float(np.mean(g)) if g else None, "grad_norm_max": float(np.max(g)) if g else None,
            "train_ncc_tmpl_last": train[-1].get("train_deform/ncc_tmpl") if train else None,
            "train_soft_dice_tmpl_last": train[-1].get("train_deform/soft_dice_tmpl") if train else None,
            "median_step_ms": s["median_compute_ms"], "gpu_peak_mb": s["gpu_peak_mb"],
            "val_curve": [[v["step"], v[p + "template_dice"], v[p + "pair_dice"], v[p + "ncc_template"],
                           v[p + "ncc_pair"], v[p + "neg_jac_pct"]] for v in s["val"]]}


def main():
    root = sys.argv[1]
    rows = [run_row(root, r) for r in "ABC"]
    cols = [("run", "%s"), ("name", "%s"), ("steps", "%d"), ("template_dice", "%.4f"), ("pair_dice", "%.4f"),
            ("ncc_template", "%.4f"), ("ncc_pair", "%.4f"), ("neg_jac_pct", "%.4f"),
            ("grad_norm_mean", "%.3f"), ("grad_norm_max", "%.3f"), ("median_step_ms", "%.0f"),
            ("gpu_peak_mb", "%.0f")]
    print("| " + " | ".join(c for c, _ in cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join((fmt % r[c]) if r[c] is not None else "-" for c, fmt in cols) + " |")
    r = rows[0]
    print("identity: template Dice %.4f, pair Dice %.4f, ncc_template %.4f, ncc_pair %.4f"
          % (r["template_dice_identity"], r["pair_dice_identity"], r["ncc_template_identity"], r["ncc_pair_identity"]))
    for r in rows:
        print("%s val curve [step, tmpl Dice, pair Dice, ncc_t, ncc_p, negJ%%]: %s"
              % (r["run"], [[round(x, 4) for x in v] for v in r["val_curve"]]))
    with open(os.path.join(HERE, "overfit_table.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
