"""Offline probe (CPU, numpy/scipy only; py3.8): does post-hoc inverse refinement and post-hoc
re-centering fix our inverse error / centrality on Mindboggle, and where is the inverse error?

Reads the written fields of one predict run (canonical xyz +++ mm on the processed 1 mm grid) and
the processed images (brain mask = image > 0). Writes one JSON. Nothing in the repo is modified.

  inverse  for the first --n subjects:
           e_fwd = |compose(fwd, inv)| (atlas mask), e_inv = |compose(inv, fwd)| (subject mask),
           before and after K fixed-point iterations  w <- -fwd(y + w)  (w0 = inv);
           error binned by distance (mm) to the atlas-mask border and by |fwd|.
  center   all subjects: m = mean_i fwd_i; centrality |m| (atlas mask); then R rounds of
           fwd_i <- compose(-m, fwd_i) (= phi_i o (id - m)), recomputing m each round.

python probe_inverse_centrality.py --run-dir RUN --processed-dir P --out OUT.json [--n 10 --iters 10 --rounds 2]
"""
import argparse
import glob
import json
import os
import time

import nibabel as nib
import numpy as np
from scipy import ndimage


def load_mm(path):
    a = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    a = a.reshape([d for d in a.shape if d != 1]) if a.ndim > 4 else a
    return a                                                   # [X, Y, Z, 3] mm == vox (1 mm grid)


_G = {}


def grid(shape):
    if shape not in _G:
        _G[shape] = np.indices(shape, dtype=np.float32)
    return _G[shape]


def sample(field, disp):
    """field(x + disp(x)) per channel, trilinear, edge clamp (evaluation.fields.compose)."""
    c = grid(disp.shape[:3]) + np.moveaxis(disp, -1, 0)
    out = np.empty_like(disp)
    for k in range(3):
        out[..., k] = ndimage.map_coordinates(field[..., k], c, order=1, mode="nearest", output=np.float32,
                                              prefilter=False)
    return out


def compose(a, b):
    """Displacement of x -> x + a(x) followed by b (evaluation.fields.compose)."""
    return a + sample(b, a)


def norm(u):
    return np.sqrt(np.sum(u * u, axis=-1))


def stats(e, m):
    v = e[m]
    return {"mean": float(v.mean()), "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--fields-dir", default=None, help="default <run-dir>/fields (MultiMorph: mindboggle_results)")
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--skip-center", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    fdir = a.fields_dir or os.path.join(a.run_dir, "fields")
    subs = sorted(os.path.basename(p)[:-len("_fwd.nii.gz")] for p in glob.glob(os.path.join(fdir, "*_fwd.nii.gz")))
    img = {s: np.asarray(nib.load(os.path.join(a.processed_dir, s, "image.nii.gz")).dataobj, np.float32) for s in subs}
    shape = img[subs[0]].shape
    # atlas mask as in evaluation: >= 50 % of the warped images > 0
    cnt = np.zeros(shape, np.int32)
    for s in subs:
        fwd = load_mm(os.path.join(fdir, s + "_fwd.nii.gz"))
        c = grid(shape) + np.moveaxis(fwd, -1, 0)
        cnt += ndimage.map_coordinates(img[s], c, order=1, mode="constant", cval=0.0, prefilter=False) > 0
    am = cnt >= 0.5 * len(subs)
    dist = ndimage.distance_transform_edt(am)                 # mm inside the atlas mask
    out = {"run": a.run_dir, "n_subjects": len(subs), "atlas_mask_voxels": int(am.sum()),
           "grid_voxels": int(np.prod(shape)), "inverse": {}, "center": {}}
    bins = [0, 3, 6, 10, 15, 25, 100]
    for s in subs[: a.n]:
        fwd = load_mm(os.path.join(fdir, s + "_fwd.nii.gz"))
        inv = load_mm(os.path.join(fdir, s + "_inv.nii.gz"))
        sm = img[s] > 0
        e1, e2 = norm(compose(fwd, inv)), norm(compose(inv, fwd))
        rec = {"before": {"fwd_then_inv": stats(e1, am), "inv_then_fwd": stats(e2, sm),
                          "fwd_then_inv_all_voxels_mean": float(e1.mean())},
               "by_border_distance_mm": {}, "by_disp_mm": {}}
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = am & (dist > lo) & (dist <= hi)
            if m.any():
                rec["by_border_distance_mm"]["%d-%d" % (lo, hi)] = float(e1[m].mean())
        mag = norm(fwd)
        for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 100)):
            m = am & (mag > lo) & (mag <= hi)
            if m.any():
                rec["by_disp_mm"]["%d-%d" % (lo, hi)] = [float(e1[m].mean()), int(m.sum())]
        w = inv.copy()
        trace = []
        for k in range(a.iters):
            w = -sample(fwd, w)                                # w(y) = -fwd(y + w(y))
            trace.append(float(norm(compose(fwd, w))[am].mean()))
        e1r, e2r = norm(compose(fwd, w)), norm(compose(w, fwd))
        rec["after_fixed_point"] = {"fwd_then_inv": stats(e1r, am), "inv_then_fwd": stats(e2r, sm),
                                    "trace_fwd_then_inv_mean": trace,
                                    "max_change_vs_inv_mm": float(norm(w - inv)[sm].max()),
                                    "mean_change_vs_inv_mm": float(norm(w - inv)[sm].mean())}
        out["inverse"][s] = rec
        print("%s inv_err fwd %.3f -> %.3f, inv %.3f -> %.3f (%.0f s)" % (
            s, rec["before"]["fwd_then_inv"]["mean"], rec["after_fixed_point"]["fwd_then_inv"]["mean"],
            rec["before"]["inv_then_fwd"]["mean"], rec["after_fixed_point"]["inv_then_fwd"]["mean"], time.time() - t0),
            flush=True)
    if not a.skip_center:
        m = np.zeros(shape + (3,), np.float64)
        for s in subs:
            m += load_mm(os.path.join(fdir, s + "_fwd.nii.gz"))
        m = (m / len(subs)).astype(np.float32)
        hist = [stats(norm(m), am)]
        for r in range(a.rounds):
            neg = -m
            m_new = np.zeros(shape + (3,), np.float64)
            for s in subs:          # fwd_i' = compose(-m, fwd_i) applied to the ORIGINAL fwd with the summed shift
                m_new += compose(neg, load_mm(os.path.join(fdir, s + "_fwd.nii.gz")))
            m_new = (m_new / len(subs)).astype(np.float32)
            hist.append(stats(norm(m_new), am))
            # next round: total shift = previous shift + residual mean (first-order update)
            m = -(neg - m_new)
            print("round %d centrality %.3f mm (%.0f s)" % (r + 1, hist[-1]["mean"], time.time() - t0), flush=True)
        out["center"] = {"centrality_by_round": hist}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out, "%.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
