"""Probe (GPU, read-only for the repo): does integrating every level's SVF on the 1 mm grid at
inference time reduce the inverse error / centrality, and how much does it change the forward map?

Runs `predict.py run` in-process on one checkpoint (deform branch, all levels) with a wrapper around
`inference.streaming.StreamedLevel._f_p12` that stashes each level's centred SVF v (exactly what the
model integrates). Then rebuilds the final fields per subject in several variants and writes them as
NIfTI next to the baseline fields, for `python -m evaluation run` (template metrics):

  A   native (reproduces predict): exp at each level's grid, compose at that level, upsample
  D   native integration, but all compositions at 1 mm (separates "where we compose")
  B   L4 SVF integrated at 1 mm (exp(up(up(v4)))), L2 / L1 native, composed at 1 mm
  C   all SVFs integrated at 1 mm, composed at 1 mm
  L4n level 4 only, native (= predict --levels 4)
  L4f level 4 only, integrated at 1 mm

Model inputs of levels 2 / 1 are NOT changed (they use the native pre-warp, as in training); only
the output reconstruction changes. Nothing in the repo is written; outputs go under --out.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

T0 = time.time()


def log(msg):
    print("[%6.0f s] %s" % (time.time() - T0, msg), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--grid-reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="A,D,B,C,L4n,L4f")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--keep-fields", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.chdir(a.repo)
    sys.path.insert(0, a.repo)

    from inference import cli
    from inference import streaming as ST

    stash = {}
    orig = ST.StreamedLevel._f_p12

    def wrapped(self, cascade):
        f = orig(self, cascade)
        lv = int(self.level)

        def g(*args):
            out = f(*args)
            stash.setdefault(lv, []).append(out["v"].numpy().astype(np.float32))
            return out
        return g

    ST.StreamedLevel._f_p12 = wrapped

    base = os.path.join(a.out, "base")
    argv = ["run", "--checkpoint", a.checkpoint, "--processed-dir", a.processed_dir,
            "--grid-reference", a.grid_reference, "--branches", "deform", "--out", base,
            "--check-internal", "0"]
    args = cli.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from inference import predict as PR
    log("predict run: " + " ".join(argv))
    rc = PR.run_predict(args, configure_device=True)
    log("predict rc %s" % rc)
    if rc not in (0, None):
        sys.exit("predict failed rc %s" % rc)
    ST.StreamedLevel._f_p12 = orig

    import nibabel as nib
    import tensorflow as tf
    from models import diffeo as D

    v = {lv: np.concatenate(chunks, axis=0) for lv, chunks in stash.items()}
    del stash
    log("stashed v: " + str({lv: x.shape for lv, x in v.items()}))
    subs = sorted(os.path.basename(p)[:-len("_fwd.nii.gz")] for p in glob.glob(os.path.join(base, "*_fwd.nii.gz")))
    n = len(subs)
    for lv, x in v.items():
        if x.shape[0] != n:
            sys.exit("level %d: %d stashed SVFs for %d subjects" % (lv, x.shape[0], n))
    levels = sorted(v.keys(), reverse=True)          # e.g. [4, 2, 1]
    if levels != [4, 2, 1]:
        sys.exit("expected levels 4, 2, 1, got %s" % levels)
    pools = {4: 2, 2: 1, 1: 0}
    steps = 7
    lean = True

    def up(u, k):
        for _ in range(k):
            u = D.upsample_displacement(u)
        return u

    @tf.function
    def build(v4, v2, v1):
        e = {}
        for lv, vv in ((4, v4), (2, v2), (1, v1)):
            e[lv] = D.exp_and_inverse(vv, steps, lean=lean)                       # native grid
            e[str(lv) + "f"] = D.exp_and_inverse(up(vv, pools[lv]), steps, lean=lean)  # at 1 mm
        out = {}
        # A: native cascade (as predict: compose at each level, upsample by one level step)
        U, Ui = e[4]
        U, Ui = D.compose_cascade(up(U, 1), up(Ui, 1), e[2][0], e[2][1], lean=lean)
        U, Ui = D.compose_cascade(up(U, 1), up(Ui, 1), e[1][0], e[1][1], lean=lean)
        out["A"] = (U, Ui)

        def at1(f4, f2, f1):
            U, Ui = f4
            U, Ui = D.compose_cascade(U, Ui, f2[0], f2[1], lean=lean)
            return D.compose_cascade(U, Ui, f1[0], f1[1], lean=lean)
        n4 = (up(e[4][0], 2), up(e[4][1], 2))
        n2 = (up(e[2][0], 1), up(e[2][1], 1))
        out["D"] = at1(n4, n2, e[1])
        out["B"] = at1(e["4f"], n2, e[1])
        out["C"] = at1(e["4f"], e["2f"], e[1])
        out["L4n"] = n4
        out["L4f"] = e["4f"]
        return out

    variants = [s for s in a.variants.split(",") if s]
    ref = nib.load(os.path.join(base, subs[0] + "_fwd.nii.gz"))
    shape, aff = ref.shape, ref.affine
    log("written field shape %s" % (shape,))
    from inference.canonical import write_nifti
    stats = {s: {"fwd_change_vs_A_mean_mm": [], "fwd_change_vs_A_max_mm": [], "allvox_inv_err_mm": []} for s in variants}
    sanity = []
    # pass 1: statistics of every variant (no files)
    for i, sub in enumerate(subs):
        out = build(tf.constant(v[4][i:i + 1]), tf.constant(v[2][i:i + 1]), tf.constant(v[1][i:i + 1]))
        A = out["A"][0][0].numpy()
        w = np.asarray(nib.load(os.path.join(base, sub + "_fwd.nii.gz")).dataobj, np.float32).reshape(A.shape)
        sanity.append(float(np.abs(A - w).max()))
        for s in variants:
            U, Ui = out[s]
            dchg = np.sqrt(((U[0].numpy() - A) ** 2).sum(-1))
            stats[s]["fwd_change_vs_A_mean_mm"].append(float(dchg.mean()))
            stats[s]["fwd_change_vs_A_max_mm"].append(float(dchg.max()))
            e = D.compose(U, Ui)[0].numpy()                    # fwd o inv residual, all voxels
            stats[s]["allvox_inv_err_mm"].append(float(np.sqrt((e ** 2).sum(-1)).mean()))
        if i % 10 == 0 or i == n - 1:
            log("stats %d/%d %s sanity max|A - written| %.2e mm" % (i + 1, n, sub, sanity[-1]))
    summary = {"checkpoint": a.checkpoint, "n_subjects": n, "sanity_max_abs_A_minus_written_mm": max(sanity),
               "variants": {s: {k: float(np.mean(x)) for k, x in d.items()} for s, d in stats.items()}}
    summary["variants_max"] = {s: {"fwd_change_vs_A_max_mm": float(np.max(d["fwd_change_vs_A_max_mm"]))} for s, d in stats.items()}
    with open(os.path.join(a.out, "probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log("sanity max |A - written| = %.3e mm" % max(sanity))
    log(json.dumps(summary["variants"]))
    if max(sanity) > 0.05:
        sys.exit("variant A does not reproduce the written fields (max %.3f mm): conventions differ" % max(sanity))

    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="")
    # pass 2: one variant at a time (write 101 x 2 fields, evaluate, delete) to bound disk use
    for s in variants:
        d = os.path.join(a.out, s)
        os.makedirs(d, exist_ok=True)
        for i, sub in enumerate(subs):
            out = build(tf.constant(v[4][i:i + 1]), tf.constant(v[2][i:i + 1]), tf.constant(v[1][i:i + 1]))
            U, Ui = out[s]
            write_nifti(os.path.join(d, sub + "_fwd.nii.gz"), U[0].numpy().reshape(shape), aff)
            write_nifti(os.path.join(d, sub + "_inv.nii.gz"), Ui[0].numpy().reshape(shape), aff)
        log("wrote fields of " + s)
        cmd = [sys.executable, "-m", "evaluation", "run", "--processed-dir", a.processed_dir,
               "--method", "probe1mm_" + s, "--fwd", os.path.join(d, "{subject_dir}_fwd.nii.gz"),
               "--inv", os.path.join(d, "{subject_dir}_inv.nii.gz"), "--components", "xyz", "--signs", "+++",
               "--units", "mm", "--pairs", "none", "--workers", str(a.workers), "--out", os.path.join(a.out, "eval_" + s)]
        log("eval " + s)
        r = subprocess.run(cmd, cwd=a.repo, env=env, stdin=subprocess.DEVNULL)
        log("eval %s rc %d" % (s, r.returncode))
        if r.returncode == 0 and not a.keep_fields:
            shutil.rmtree(d)
    del v
    if not a.keep_fields:
        for p in glob.glob(os.path.join(base, "*_fwd.nii.gz")) + glob.glob(os.path.join(base, "*_inv.nii.gz")) + \
                glob.glob(os.path.join(base, "*_moved.nii.gz")):
            os.remove(p)
    rows = []
    for s in variants:
        f = os.path.join(a.out, "eval_" + s, "summary.json")
        if not os.path.exists(f):
            rows.append((s, None))
            continue
        m = json.load(open(f))
        sm, g = m["subject_metrics"], m.get("group", {})
        rows.append((s, {k: sm[k]["mean"] for k in ("template_dice", "template_dice_cortical", "template_dice_subcortical",
                                                    "template_lncc", "fold_fwd_pct", "fold_inv_pct", "logjac_std_fwd",
                                                    "inv_err_fwd_mm", "inv_err_inv_mm") if k in sm},
                     g.get("centrality_mean_mm")))
    summary["eval"] = {r[0]: (None if r[1] is None else dict(r[1], centrality_mean_mm=r[2])) for r in rows}
    with open(os.path.join(a.out, "probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log("done")
    for s, d in summary["eval"].items():
        log("%s %s" % (s, d))


if __name__ == "__main__":
    main()
