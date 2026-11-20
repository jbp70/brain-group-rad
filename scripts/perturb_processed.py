#!/usr/bin/env python
"""Copy processed subjects with a known random affine per subject (off-MNI test inputs, paper F2).

For every selected subject folder ``<src>/<subject_dir>/`` (image.nii.gz + labels_<name>.nii.gz on the
LUMIR target grid), draw a reproducible random affine P (seed + subject id) and write
``<out>/<subject_dir>/`` with

    image.nii.gz              out(y) = in(P^-1 y), trilinear (scipy order 1), zero outside
    labels_<name>.nii.gz      the same map, nearest neighbour, source dtype
    perturbation_world.txt    P (4x4, world RAS mm; maps the source image to the written image)
    meta.json                 row.subject (read by evaluation.find_subjects), the parameters,
                              brain field-of-view loss and the displacement magnitude over the brain

and ``<out>/perturbation_summary.csv`` (one row per subject, sorted by subject_dir).

P uses the same parameterisation as the MIRIAD centroid backend (preprocessing.pipeline
perturbation_params / perturbation_matrix): rotation angle U(-r, r) about an axis uniform on the
sphere, translation U(-t, t) mm per axis, log-scale U(-s, s) per axis, 3 shears U(-h, h), about the
subject's brain centroid (labels > 0). Defaults = the AFFINE training ranges at every level
(data_loading.params.DEFAULT_LINEAR_RANGES['affine']: 45 deg, 15 mm, ln 1.10, 0.05).

--shared-rotation-deg A (default 0) adds ONE rotation G by exactly A degrees, the same for every
subject (axis drawn from the seed, about the grid centre, no translation): P = G @ P_individual.
Without it the group's central space stays near MNI (the individual draws average out); with it the
central space is rotated by ~A degrees away from MNI.

This resamples the processed image a second time (the processed LUMIR images were resampled once
from native). Brain voxels mapped outside the 160 x 224 x 192 grid are lost (meta fov.lost_frac);
the training augmentation has the same property.

    python scripts/perturb_processed.py --src $PROJ/data/processed/lumir_ants \
        --manifest $PROJ/data/npy_v3/export_manifest.csv --split val \
        --shared-rotation-deg 30 --out $PROJ/data/processed/lumir_ants_val_affpert45_rot30 --workers 8

Python 3.8, numpy 1.21 / scipy 1.4 / nibabel 3.1 (tf10_nv.sif).
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import nibabel as nib
import numpy as np
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from preprocessing.pipeline import perturbation_matrix, perturbation_params  # noqa: E402

AFFINE_TRAIN_RANGES = {"rotation_deg": 45.0, "translation_mm": 15.0, "log_scale": math.log(1.10), "shear": 0.05}
SUMMARY = "perturbation_summary.csv"


def select_dirs(a):
    if a.subjects_file:
        with open(a.subjects_file) as f:
            names = [x.split("#", 1)[0].strip() for x in f]
        names = [n for n in names if n]
    elif a.manifest:
        with open(a.manifest, newline="") as f:
            rows = list(csv.DictReader(f))
        names = [r["subject_dir"] for r in rows if (a.split is None or r["split"] == a.split)]
    else:
        names = sorted(d for d in os.listdir(a.src) if os.path.isfile(os.path.join(a.src, d, "image.nii.gz")))
    names = sorted(set(names))
    missing = [n for n in names if not os.path.isfile(os.path.join(a.src, n, "image.nii.gz"))]
    if missing:
        raise SystemExit("missing image.nii.gz for %d subjects, e.g. %s" % (len(missing), missing[:3]))
    if len(names) < 2:
        raise SystemExit("need at least 2 subjects, got %d" % len(names))
    return names


def subject_id(src_dir, name):
    meta = os.path.join(src_dir, "meta.json")
    if os.path.isfile(meta):
        try:
            with open(meta) as f:
                s = json.load(f).get("row", {}).get("subject")
            if s:
                return s
        except ValueError:
            pass
    return name[:-4] if name.endswith("_T1w") else name


def shared_rotation(seed, angle_deg, center):
    """4x4 world rotation G (the SAME for every subject) by exactly ``angle_deg`` about an axis
    uniform on the sphere drawn from the seed, about the fixed point ``center`` (no translation of
    that point). Returns (G, axis)."""
    import hashlib
    h = hashlib.sha256(("%d:__shared_rotation__" % int(seed)).encode("utf-8")).hexdigest()
    rng = np.random.RandomState(int(h[:8], 16))
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    params = {"rotation_axis": axis.tolist(), "rotation_deg": float(angle_deg), "translation_mm": [0.0] * 3,
              "log_scale": [0.0] * 3, "shear_xy_xz_yz": [0.0] * 3}
    return perturbation_matrix(params, center), axis


def rotation_angle_deg(A):
    """Angle (deg) of the rotation factor R of the polar decomposition A = R S."""
    from scipy.linalg import polar
    R = polar(np.asarray(A, np.float64))[0]
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def grid_center_world(aff, shape):
    return (aff @ np.r_[(np.asarray(shape[:3], np.float64) - 1) / 2.0, 1.0])[:3]


def _pull(vol, M, order, dtype):
    """out[v] = vol[M[:3,:3] v + M[:3,3]] (voxel map out -> in)."""
    return ndimage.affine_transform(vol, M[:3, :3], offset=M[:3, 3], output_shape=vol.shape, order=order,
                                    mode="constant", cval=0, prefilter=False, output=dtype)


def perturb_one(job):
    a, name = job
    t0 = time.time()
    src_dir, out_dir = os.path.join(a.src, name), os.path.join(a.out, name)
    done = os.path.join(out_dir, "meta.json")
    cfg = {"seed": a.seed, "rotation_deg": a.rotation_deg, "translation_mm": a.translation_mm,
           "log_scale": a.log_scale, "shear": a.shear, "shared_rotation_deg": a.shared_rotation_deg}
    if os.path.isfile(done) and not a.overwrite:
        with open(done) as f:
            old = json.load(f)
        if old["perturbation"]["config"] != cfg:
            raise RuntimeError("%s exists with perturbation %s != requested %s (use --overwrite or another --out)"
                               % (out_dir, old["perturbation"]["config"], cfg))
        return old["summary"]
    img = nib.load(os.path.join(src_dir, "image.nii.gz"))
    aff = np.asarray(img.affine, np.float64)
    data = np.asarray(img.dataobj, np.float32)
    labs = {}
    for ln in a.label_name:
        li = nib.load(os.path.join(src_dir, "labels_%s.nii.gz" % ln))
        if not np.allclose(li.affine, aff, atol=1e-4) or li.shape != img.shape:
            raise RuntimeError("%s: labels_%s grid differs from image" % (name, ln))
        labs[ln] = (li, np.asarray(li.dataobj))
    mask = labs[a.label_name[0]][1] > 0 if labs else data > 0
    idx = np.argwhere(mask).astype(np.float64)
    if not len(idx):
        raise RuntimeError("%s: empty brain mask" % name)
    world = idx @ aff[:3, :3].T + aff[:3, 3]
    center = world.mean(0)
    subj = subject_id(src_dir, name)
    params = perturbation_params(cfg, subj)
    P_ind = perturbation_matrix(params, center)
    g_center = grid_center_world(aff, img.shape)
    G, g_axis = shared_rotation(a.seed, a.shared_rotation_deg, g_center)
    P = G @ P_ind                       # individual perturbation first, then the shared rotation
    # voxel pull map out -> in: aff^-1 P^-1 aff
    M = np.linalg.inv(aff) @ np.linalg.inv(P) @ aff
    # brain voxels after the map (forward push), FOV loss and displacement magnitude
    moved_world = world @ P[:3, :3].T + P[:3, 3]
    vox = (moved_world - aff[:3, 3]) @ np.linalg.inv(aff[:3, :3]).T
    shape = np.asarray(img.shape[:3], np.float64)
    inside = np.all((vox >= -0.5) & (vox <= shape - 0.5), axis=1)
    disp = np.linalg.norm(moved_world - world, axis=1)
    os.makedirs(out_dir, exist_ok=True)
    out_img = np.clip(_pull(data, M, 1, np.float32), 0.0, None)
    hdr = img.header.copy()
    hdr.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(out_img, aff, hdr), os.path.join(out_dir, "image.nii.gz"))
    lab_counts = {}
    for ln, (li, arr) in labs.items():
        out_lab = _pull(arr, M, 0, arr.dtype)
        h = li.header.copy()
        h.set_data_dtype(arr.dtype)
        nib.save(nib.Nifti1Image(out_lab, aff, h), os.path.join(out_dir, "labels_%s.nii.gz" % ln))
        lab_counts[ln] = {"before": int((arr > 0).sum()), "after": int((out_lab > 0).sum())}
    np.savetxt(os.path.join(out_dir, "perturbation_world.txt"), P, fmt="%.10f")
    rot = abs(params["rotation_deg"])
    summ = {"subject_dir": name, "subject": subj, "rotation_deg": rot,
            "shared_rotation_deg": float(a.shared_rotation_deg),
            "total_rotation_deg": rotation_angle_deg(P[:3, :3]),
            "translation_norm_mm": float(np.linalg.norm(params["translation_mm"])),
            "scale_max_pct": float(100.0 * np.max(np.abs(np.expm1(params["log_scale"])))),
            "shear_max": float(np.max(np.abs(params["shear_xy_xz_yz"]))),
            "disp_mean_mm": float(disp.mean()), "disp_rms_mm": float(np.sqrt((disp ** 2).mean())),
            "disp_max_mm": float(disp.max()), "lost_frac": float(1.0 - inside.mean())}
    meta = {"row": {"dataset": a.dataset, "subject": subj, "split": a.split or "",
                    "source_dir": os.path.abspath(src_dir)},
            "perturbation": {"config": cfg, "params": params, "about_mm": center.tolist(),
                             "matrix_world_ras": P.tolist(),
                             "individual_matrix_world_ras": P_ind.tolist(),
                             "shared": {"rotation_deg": float(a.shared_rotation_deg), "axis": g_axis.tolist(),
                                        "about_mm": g_center.tolist(), "matrix_world_ras": G.tolist()},
                             "note": "out(y) = in(P^-1 y): P = G @ P_individual maps the source image to this "
                                     "image (RAS mm); P_individual about the source brain centroid "
                                     "(labels > 0), G = shared rotation (same for every subject) about the "
                                     "grid centre",
                             "parameterisation": "preprocessing.pipeline.perturbation_params/_matrix"},
            "interpolation": {"image": "trilinear (scipy order 1), clipped at 0", "labels": "nearest"},
            "fov": {"lost_frac": summ["lost_frac"], "label_voxels": lab_counts},
            "summary": summ, "seconds": time.time() - t0}
    with open(done + ".tmp", "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(done + ".tmp", done)
    return summ


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="processed dataset root (subject folders)")
    p.add_argument("--out", required=True, help="output root (must differ from --src)")
    p.add_argument("--manifest", default=None, help="export_manifest.csv; selects subject_dir by --split")
    p.add_argument("--split", default=None, help="manifest split, e.g. val")
    p.add_argument("--subjects-file", default=None, help="alternative: one subject_dir per line")
    p.add_argument("--label-name", action="append", default=None,
                   help="labels_<name>.nii.gz to carry along (repeatable; default synthseg)")
    p.add_argument("--dataset", default="lumir_ants_affpert45", help="meta row.dataset")
    p.add_argument("--seed", type=int, default=2026)
    for k, v in AFFINE_TRAIN_RANGES.items():
        p.add_argument("--" + k.replace("_", "-"), type=float, default=v, help="half-width (default %(default).4g)")
    p.add_argument("--shared-rotation-deg", type=float, default=0.0,
                   help="rotation by exactly this angle applied to EVERY subject after its own perturbation "
                        "(axis drawn from --seed, about the grid centre; moves the group's central space off "
                        "MNI). Default 0 = none")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    a.label_name = a.label_name or ["synthseg"]
    if os.path.abspath(a.src) == os.path.abspath(a.out):
        raise SystemExit("--out must differ from --src")
    names = select_dirs(a)
    os.makedirs(a.out, exist_ok=True)
    print("%d subjects: %s -> %s (seed %d, rot %.1f deg, trans %.1f mm, log-scale %.4f, shear %.3f, "
          "shared rotation %.1f deg)" % (len(names), a.src, a.out, a.seed, a.rotation_deg, a.translation_mm,
                                         a.log_scale, a.shear, a.shared_rotation_deg), flush=True)
    jobs = [(a, n) for n in names]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers) as pool:
            rows = pool.map(perturb_one, jobs, chunksize=1)
    else:
        rows = [perturb_one(j) for j in jobs]
    rows.sort(key=lambda r: r["subject_dir"])
    with open(os.path.join(a.out, SUMMARY), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lost = np.array([r["lost_frac"] for r in rows])
    disp = np.array([r["disp_mean_mm"] for r in rows])
    print("done: brain displacement mean %.1f mm (range %.1f-%.1f); FOV loss mean %.3f %%, max %.3f %% (%d > 0.5 %%)"
          % (disp.mean(), disp.min(), disp.max(), 100 * lost.mean(), 100 * lost.max(), int((lost > 0.005).sum())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
