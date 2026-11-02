"""Per-image check: do a method's warps remove cortical detail that the unwarped images have?

For every subject and method (canonical fields, see evaluation/fields.py):
  native   processed image I_i with its own labels L_i
  moved    the method's moved image (subject -> atlas) with L_i warped by fwd_i (nearest)
Cortex mask = cortical labels (1000-2999) dilated by --dilate voxels (includes sulcal CSF), restricted to
nonzero (skull-stripped brain) voxels, in each space.
  grad_*        mean |grad image| in the cortex mask
  csf_frac_*    fraction of cortex-mask voxels darker than --csf-thr (sulcal CSF; images are [0, 1])
  csf_vol_*     number of those voxels (mm^3 at 1 mm); csf_vol_moved_jac = sum over moved CSF voxels of
                det J (fwd maps atlas -> subject, so this is their native volume)
  logjac_*      in the moved cortex mask: mean |log det J|, std, % voxels det J > 1.5 / < 1/1.5,
                and mean log det J on moved CSF voxels vs moved tissue voxels. det J < 1 = the
                subject's content is magnified (stretched) into the atlas, which blurs it
Ratios moved / native summarize how much detail a method removes (1 = preserved).

    python scripts/moved_detail.py --processed-dir P --out OUT --workers 4 \\
        --method 'ours=F/{subject_dir}_moved.nii.gz,F/{subject_dir}_fwd.nii.gz' \\
        --method 'multimorph=M/{subject_dir}_moved.nii.gz,M/{subject_dir}_fwd.nii.gz'
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import fields as F  # noqa: E402
from evaluation.evaluate import find_subjects  # noqa: E402

S = {}


def load(path, dtype=np.float32):
    a = np.asanyarray(nib.load(path).dataobj, dtype=dtype)
    return a.reshape(a.shape[:3])


def cortex_mask(lab, dilate):
    m = (lab >= 1000) & (lab < 3000)
    return ndimage.binary_dilation(m, ndimage.generate_binary_structure(3, 1), iterations=dilate)


def grad_mean(img, m):
    g = np.gradient(img)
    return float(np.sqrt(sum(x * x for x in g))[m].mean())


def one(task):
    mi, si = task
    name, moved_t, fwd_t = S["methods"][mi]
    sub = S["subs"][si]
    thr, dil = S["thr"], S["dilate"]
    img = load(sub["image"])
    lab = np.asanyarray(nib.load(sub["labels"]).dataobj).reshape(img.shape).astype(np.int32)
    fmt = dict(subject=sub["subject"], subject_dir=sub["dir"])
    moved = load(moved_t.format(**fmt))
    fwd = F.load_field(fwd_t.format(**fmt), img.shape, S["spacing"], F.Convention("xyz", "+++", "mm"))
    wlab = F.warp(lab, fwd, 0)
    # Inside the skull-stripped brain only (background is exactly 0 and would count as "CSF").
    cn, cm = cortex_mask(lab, dil) & (img > 0), cortex_mask(wlab, dil) & (moved > 0)
    csf_n, csf_m = cn & (img < thr), cm & (moved < thr)
    det = F.jacobian_det(fwd)
    logj = np.log(np.maximum(det, 1e-6))
    r = {"method": name, "subject": sub["subject"],
         "grad_native": grad_mean(img, cn), "grad_moved": grad_mean(moved, cm),
         "csf_frac_native": float(csf_n.sum() / max(cn.sum(), 1)), "csf_frac_moved": float(csf_m.sum() / max(cm.sum(), 1)),
         "csf_vol_native": int(csf_n.sum()), "csf_vol_moved": int(csf_m.sum()),
         # fwd maps atlas -> subject (x -> x + fwd(x)): an atlas voxel covers det J mm^3 of the subject.
         "csf_vol_moved_jac": float(np.maximum(det[csf_m], 0).sum()),
         "cortex_vol_native": int(cn.sum()), "cortex_vol_moved": int(cm.sum()),
         "logjac_abs_mean": float(np.abs(logj[cm]).mean()), "logjac_std": float(logj[cm].std()),
         "pct_expand_1p5": 100.0 * float((det[cm] > 1.5).mean()), "pct_shrink_1p5": 100.0 * float((det[cm] < 1 / 1.5).mean()),
         "logjac_mean_on_csf": float(logj[csf_m].mean()) if csf_m.any() else float("nan"),
         "logjac_mean_on_tissue": float(logj[cm & ~csf_m].mean())}
    r["grad_ratio"] = r["grad_moved"] / r["grad_native"]
    r["csf_frac_ratio"] = r["csf_frac_moved"] / max(r["csf_frac_native"], 1e-9)
    return r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", required=True)
    p.add_argument("--method", action="append", required=True, metavar="NAME=MOVED_TEMPLATE,FWD_TEMPLATE")
    p.add_argument("--subjects-file", default=None)
    p.add_argument("--label-name", default="dkt31aseg")
    p.add_argument("--csf-thr", type=float, default=0.3)
    p.add_argument("--dilate", type=int, default=2)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    subs = find_subjects(a.processed_dir, a.label_name, a.subjects_file)
    aff = nib.load(subs[0]["image"]).affine
    methods = []
    for m in a.method:
        name, rest = m.split("=", 1)
        mv, fw = rest.split(",", 1)
        methods.append((name, mv, fw))
    S.update(methods=methods, subs=subs, thr=a.csf_thr, dilate=a.dilate,
             spacing=tuple(float(x) for x in np.sqrt((aff[:3, :3] ** 2).sum(0))))
    tasks = [(mi, si) for mi in range(len(methods)) for si in range(len(subs))]
    rows = []
    if a.workers > 1:
        with mp.get_context("fork").Pool(a.workers) as pool:
            for k, r in enumerate(pool.imap(one, tasks, chunksize=1)):
                rows.append(r)
                if (k + 1) % 20 == 0:
                    print("%d / %d" % (k + 1, len(tasks)), flush=True)
    else:
        rows = [one(t) for t in tasks]
    keys = list(rows[0].keys())
    with open(os.path.join(a.out, "per_subject.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summ = {}
    for name, _, _ in methods:
        rr = [r for r in rows if r["method"] == name]
        summ[name] = {k: {"mean": float(np.nanmean([r[k] for r in rr])), "std": float(np.nanstd([r[k] for r in rr]))}
                      for k in keys if k not in ("method", "subject")}
        summ[name]["n"] = len(rr)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({"csf_thr": a.csf_thr, "dilate": a.dilate, "methods": summ}, f, indent=1)
    names = [m[0] for m in methods]
    print("%-24s" % "metric" + "".join("%22s" % n[:21] for n in names))
    for k in keys:
        if k in ("method", "subject"):
            continue
        print("%-24s" % k + "".join("%22s" % ("%.4f +- %.4f" % (summ[n][k]["mean"], summ[n][k]["std"])) for n in names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
