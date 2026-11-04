"""Compare atlas sharpness of several methods on the same processed grid (numpy/scipy/nibabel only).

Each method: NAME=ATLAS[,MOVED_GLOB]. MOVED_GLOB (optional) matches the method's moved images; their
mean is recomputed and compared with the saved atlas, and their voxelwise spread is measured.
Labels for the masks come from --seg (one or more atlas segmentations, DKT31+aseg IDs): brain mask =
union of seg > 0, cortex mask = union of cortical labels (1000-2999); both dilated by --dilate voxels
so sulcal CSF boundaries are included. Each atlas is divided by its 99.5th percentile in the brain
mask, so methods with different intensity scales are comparable.

Metrics per method (atlas and recomputed mean):
  grad_brain / grad_cortex   mean |grad atlas| (per voxel, normalized intensity)
  hf_cortex                  mean |atlas - gaussian(atlas, 1 vox)| in the cortex mask
  spread_cortex              mean voxelwise std of the normalized moved images in the cortex mask
  mean_vs_atlas_max / _ncc   saved atlas vs recomputed mean of the moved images

    python scripts/atlas_sharpness.py --out OUT --seg A/atlas_seg.nii.gz --seg B/atlas_seg.nii.gz \\
        --method 'ours=A/atlas.nii.gz,A/*_moved.nii.gz' --method 'multimorph=B/atlas.nii.gz,B/*_T1w_moved.nii.gz'
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402


def load(path):
    a = np.asanyarray(nib.load(path).dataobj, dtype=np.float32)
    return a.reshape(a.shape[:3])


def grad_mag(x):
    g = np.gradient(x)
    return np.sqrt(sum(gi * gi for gi in g))


def ncc(a, b, m):
    x = a[m].astype(np.float64) - a[m].mean()
    y = b[m].astype(np.float64) - b[m].mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def sharpness(img, brain, cortex):
    g = grad_mag(img)
    hf = np.abs(img - ndimage.gaussian_filter(img, 1.0))
    return {"grad_brain": float(g[brain].mean()), "grad_cortex": float(g[cortex].mean()),
            "hf_cortex": float(hf[cortex].mean())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", action="append", required=True, metavar="NAME=ATLAS[,MOVED_GLOB]")
    p.add_argument("--seg", action="append", required=True, help="atlas segmentation(s) for the masks")
    p.add_argument("--dilate", type=int, default=2)
    p.add_argument("--out", required=True)
    p.add_argument("--show-subject", action="append", default=[], metavar="SUBJECT_DIR",
                   help="also plot this subject's moved image per method (file name contains "
                        "'<SUBJECT_DIR>_moved'), with its native image from --native-dir; repeatable")
    p.add_argument("--native-dir", default=None, help="processed dir with <SUBJECT_DIR>/image.nii.gz")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    brain = cortex = None
    for s in a.seg:
        lab = np.rint(load(s)).astype(np.int64)
        b, c = lab > 0, (lab >= 1000) & (lab < 3000)
        brain = b if brain is None else brain | b
        cortex = c if cortex is None else cortex | c
    st = ndimage.generate_binary_structure(3, 1)
    brain = ndimage.binary_dilation(brain, st, iterations=a.dilate)
    cortex = ndimage.binary_dilation(cortex, st, iterations=a.dilate) & brain
    print("brain mask %d voxels, cortex mask %d voxels" % (brain.sum(), cortex.sum()))

    res, shows = {}, {}
    for spec in a.method:
        name, rest = spec.split("=", 1)
        parts = rest.split(",", 1)
        atlas = load(parts[0])
        scale = float(np.percentile(atlas[brain], 99.5)) or 1.0
        r = {"atlas": parts[0], "scale_p995": scale}
        r.update({"atlas_" + k: v for k, v in sharpness(atlas / scale, brain, cortex).items()})
        shows[name] = [atlas / scale]
        if len(parts) > 1:
            files = sorted(glob.glob(parts[1]))
            if not files:
                print("WARNING: %s: no moved images match %s; atlas metrics only" % (name, parts[1]))
                res[name] = r
                continue
            s1 = np.zeros(atlas.shape, np.float64)
            s2 = np.zeros(atlas.shape, np.float64)
            for f in files:
                x = load(f) / scale
                s1 += x
                s2 += x * x
            n = max(len(files), 1)
            mean = (s1 / n).astype(np.float32)
            std = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0)).astype(np.float32)
            r.update({"n_moved": len(files), "mean_vs_atlas_max": float(np.abs(mean - atlas / scale)[brain].max()),
                      "mean_vs_atlas_ncc": ncc(mean, atlas / scale, brain),
                      "spread_cortex": float(std[cortex].mean()), "spread_brain": float(std[brain].mean())})
            r.update({"mean_" + k: v for k, v in sharpness(mean, brain, cortex).items()})
            shows[name].append(std)
        res[name] = r
        print(name, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}))

    with open(os.path.join(a.out, "sharpness.json"), "w") as f:
        json.dump({"brain_voxels": int(brain.sum()), "cortex_voxels": int(cortex.sum()), "dilate": a.dilate,
                   "methods": res}, f, indent=1)

    # Same slices and window for every method: atlas rows, then moved-image spread rows.
    X, Y, Z = brain.shape
    ix = np.argwhere(cortex)
    cx = int(np.percentile(ix[:, 0], 25))          # one hemisphere, through cortex
    cy, cz = Y // 2, int(np.percentile(ix[:, 2], 60))
    views = [("axial z=%d" % cz, lambda v: v[:, :, cz].T), ("coronal y=%d" % cy, lambda v: v[:, cy, :].T),
             ("sagittal x=%d" % cx, lambda v: v[cx, :, :].T)]
    zoom = (slice(X // 2 - 60, X // 2 + 20), slice(Y // 2 - 10, Y // 2 + 90))
    names = list(shows)
    fig, axes = plt.subplots(len(names), 4, figsize=(16, 4 * len(names)), squeeze=False)
    for i, n in enumerate(names):
        for j, (title, f) in enumerate(views):
            axes[i][j].imshow(f(shows[n][0]), cmap="gray", vmin=0, vmax=1, origin="lower")
            axes[i][j].set_title("%s  %s" % (n, title))
        axes[i][3].imshow(shows[n][0][zoom[0], zoom[1], cz].T, cmap="gray", vmin=0, vmax=1, origin="lower",
                          interpolation="nearest")
        axes[i][3].set_title("%s  axial zoom (no interpolation)" % n)
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "atlases.png"), dpi=110)
    plt.close(fig)
    spread = [n for n in names if len(shows[n]) > 1]
    if spread:
        vmax = max(float(np.percentile(shows[n][1][brain], 99)) for n in spread)
        fig, axes = plt.subplots(len(spread), 3, figsize=(12, 4 * len(spread)), squeeze=False)
        for i, n in enumerate(spread):
            for j, (title, f) in enumerate(views):
                axes[i][j].imshow(f(shows[n][1]), cmap="magma", vmin=0, vmax=vmax, origin="lower")
                axes[i][j].set_title("%s  spread (std)  %s" % (n, title))
                axes[i][j].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "spread.png"), dpi=110)
        plt.close(fig)
    # Moved images of selected subjects (same slices / window; native image first, unwarped).
    for subj in a.show_subject:
        rows = []
        if a.native_dir and os.path.isfile(os.path.join(a.native_dir, subj, "image.nii.gz")):
            nat = load(os.path.join(a.native_dir, subj, "image.nii.gz"))
            rows.append(("native (unwarped)", nat / (float(np.percentile(nat[nat > 0], 99.5)) or 1.0)))
        for spec in a.method:
            name, rest = spec.split("=", 1)
            parts = rest.split(",", 1)
            if len(parts) < 2:
                continue
            hits = [f for f in glob.glob(parts[1]) if os.path.basename(f).startswith(subj + "_moved")]
            if hits:
                x = load(hits[0])
                rows.append((name, x / (float(np.percentile(x[brain], 99.5)) or 1.0)))
        if not rows:
            print("WARNING: no images found for subject %s" % subj)
            continue
        fig, axes = plt.subplots(len(rows), 4, figsize=(16, 4 * len(rows)), squeeze=False)
        for i, (n, img) in enumerate(rows):
            for j, (title, f) in enumerate(views):
                axes[i][j].imshow(f(img), cmap="gray", vmin=0, vmax=1, origin="lower")
                axes[i][j].set_title("%s  %s  %s" % (subj, n, title), fontsize=9)
            axes[i][3].imshow(img[zoom[0], zoom[1], cz].T, cmap="gray", vmin=0, vmax=1, origin="lower",
                              interpolation="nearest")
            axes[i][3].set_title("%s  axial zoom" % n, fontsize=9)
            for ax in axes[i]:
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "moved_%s.png" % subj), dpi=100)
        plt.close(fig)
    print("wrote", os.path.join(a.out, "sharpness.json"), "atlases.png", "spread.png",
          " ".join("moved_%s.png" % s for s in a.show_subject))
    return 0


if __name__ == "__main__":
    sys.exit(main())
