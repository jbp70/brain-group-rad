"""Per-variable statistics of TF object-based checkpoints (no model build; CPU is enough).

For the m9 level-1 NaN investigation (.agents/tasks/m9-level1-nan/report.md): which trainable
variables (and Adam slots) of the level-1 run are non-finite or grew, comparing ckpt/last (step
8000), ckpt/best (linear, 6500), ckpt/best_deform (5000) and the level-2 source the stage was
initialised from (m8_level2_gradnorm ckpt/best, step 6000; same variable names under 'model/').

Usage (inside tf10_nv.sif, CUDA_VISIBLE_DEVICES= for CPU):
  python ckpt_stats.py --out <json> name=<ckpt dir or prefix> [name=...]
The FIRST checkpoint is the subject; the others are references (relative change ||w - w_ref|| /
||w_ref|| per variable). Printed: non-finite variables, loss-scale / iteration scalars, BN
moving-variance minima (max frozen-BN gain 1 / sqrt(var + 1e-5)), and the top variables by growth
of max |w| and by relative change, per part (encoder / deform_decoder / linear_decoder).
"""
import argparse
import json
import math
import sys

import numpy as np
import tensorflow as tf

SUFFIX = "/.ATTRIBUTES/VARIABLE_VALUE"
BN_EPS = 1e-5


def resolve(path):
    if tf.io.gfile.isdir(path):
        p = tf.train.latest_checkpoint(path)
        if p is None:
            raise SystemExit("no checkpoint in %s" % path)
        return p
    return path


def load(prefix):
    r = tf.train.load_checkpoint(prefix)
    out = {}
    for k, dt in r.get_variable_to_dtype_map().items():
        if not k.endswith(SUFFIX) or not dt.is_floating:
            continue
        out[k[: -len(SUFFIX)]] = r.get_tensor(k)
    return out


def part(name):
    for p in ("deform_decoder", "encoder", "linear_decoder"):
        if "/" + p + "/" in "/" + name + "/":
            return p
    return "other"


def stats(a):
    a = np.asarray(a, np.float64)
    fin = np.isfinite(a)
    af = np.where(fin, a, 0.0)
    return {"n": int(a.size), "nonfinite": int(a.size - fin.sum()),
            "max_abs": float(np.max(np.abs(af))) if a.size else 0.0,
            "rms": float(np.sqrt(np.mean(af * af))) if a.size else 0.0,
            "min": float(np.min(af)) if a.size else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("ckpts", nargs="+", help="name=path")
    a = ap.parse_args()
    names, data = [], {}
    for item in a.ckpts:
        name, path = item.split("=", 1)
        prefix = resolve(path)
        print("loading %s = %s" % (name, prefix), flush=True)
        names.append(name)
        data[name] = load(prefix)
    subj = names[0]
    S = data[subj]
    report = {"ckpts": names, "scalars": {}, "nonfinite": [], "vars": {}}

    # Scalars: step, optimizer iterations, loss scale (keys vary with the LSO version).
    for k, v in S.items():
        if np.asarray(v).size == 1 and not k.startswith("model/"):
            report["scalars"][k] = float(np.asarray(v).reshape(-1)[0])
    print("\nscalars of %s (non-model):" % subj)
    for k, v in sorted(report["scalars"].items()):
        print("  %-80s %.6g" % (k, v))

    for k, v in S.items():
        if not k.startswith("model/"):
            continue
        st = stats(v)
        st["part"] = part(k)
        st["slot"] = ".OPTIMIZER_SLOT" in k
        for ref in names[1:]:
            w0 = data[ref].get(k)
            if w0 is not None and np.shape(w0) == np.shape(v):
                d = np.asarray(v, np.float64) - np.asarray(w0, np.float64)
                n0 = float(np.linalg.norm(np.asarray(w0, np.float64)))
                st["rel_change_vs_" + ref] = (float(np.linalg.norm(d)) / n0 if n0 > 0 and np.all(np.isfinite(d))
                                              else float("nan") if not np.all(np.isfinite(d)) else None)
                st["max_abs_" + ref] = stats(w0)["max_abs"]
        report["vars"][k] = st
        if st["nonfinite"]:
            report["nonfinite"].append(k)

    print("\nnon-finite variables in %s: %d" % (subj, len(report["nonfinite"])))
    for k in report["nonfinite"][:50]:
        print("  ", k, report["vars"][k]["nonfinite"], "/", report["vars"][k]["n"])
    for ref in names[1:]:
        bad = [k for k, v in data[ref].items() if k.startswith("model/") and not np.all(np.isfinite(v))]
        print("non-finite variables in %s: %d" % (ref, len(bad)))

    # Frozen-BN gains of the trainable stage.
    print("\nBN moving_variance minima (gain = 1/sqrt(var+eps)), lowest 20, %s:" % subj)
    bn = []
    for k, v in S.items():
        if k.startswith("model/") and k.endswith("moving_variance") and ".OPTIMIZER_SLOT" not in k:
            v = np.asarray(v, np.float64)
            bn.append((float(v.min()), k, float(1.0 / math.sqrt(max(v.min(), 0.0) + BN_EPS)), int((v < 1e-3).sum()), v.size))
    for mn, k, g, small, n in sorted(bn)[:20]:
        print("  min %-10.3g gain %-8.1f n(var<1e-3) %d/%d  %s" % (mn, g, small, n, k))

    weights = {k: v for k, v in report["vars"].items() if not v["slot"]}
    for ref in names[1:]:
        key = "rel_change_vs_" + ref
        for p in ("deform_decoder", "encoder", "linear_decoder", "other"):
            rows = [(v.get(key), k, v) for k, v in weights.items() if v["part"] == p and v.get(key) is not None]
            rows = [r for r in rows if r[0] == r[0]]
            if not rows:
                continue
            rows.sort(key=lambda r: -r[0])
            print("\n[%s] top %d relative change vs %s:" % (p, a.top, ref))
            for rc, k, v in rows[: a.top]:
                print("  rel %-9.4g max|w| %-9.4g (ref %-9.4g) rms %-9.4g %s"
                      % (rc, v["max_abs"], v.get("max_abs_" + ref, float("nan")), v["rms"], k))
    print("\n[slots] largest max|.| of Adam slots (subject):")
    slots = sorted(((v["max_abs"], k) for k, v in report["vars"].items() if v["slot"]), reverse=True)
    for m, k in slots[: a.top]:
        print("  %-10.4g %s" % (m, k))
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
