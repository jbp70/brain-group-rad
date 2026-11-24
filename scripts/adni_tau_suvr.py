#!/usr/bin/env python
"""ADNI tau-PET experiment (paper Figure 5): SUVR change in four resampling arms, then statistics.

Per patient the group {baseline PET, follow-up PET} is registered once by our model (rigid branch,
slurm/predict_miriad.sh on the processed PETs -> <pred>/<patient>/fields/matrices.npz). Every arm
uses those same transforms, so the arms differ ONLY in which PET is resampled:

    native    no PET resampled; the T1 labels are mapped into each native PET grid
    forward   reference = baseline: follow-up resampled into the baseline grid (fixed-baseline practice)
    backward  reference = follow-up: baseline resampled into the follow-up grid
    central   both PETs resampled once into the central space (grid with the native PET voxel size)

Geometry (world RAS mm point maps): F_i = matrices.npz fwd_world (central -> processed PET i),
C_i = affine_to_template_world.txt of processed PET i (native -> processed), so
W_i = C_i^-1 F_i maps central points to native points of PET i. Pulling PET j into the grid of PET i
uses W_j W_i^-1. The T1 is registered rigidly (ANTs, Mattes MI) to the model's central PET mean
(fields/atlas.nii.gz); R maps central points to processed-T1 points, so T1 labels reach any grid G
with map_G (grid world -> central) via R map_G. Labels are sampled at k^3 sub-voxel points per PET
voxel (nearest T1 label each), so every region mean is a partial-volume weighted mean of the PET
voxels (no label interpolation). Voxels whose resampled value needs data outside a PET's field of view
are dropped from both time points of that arm (reported as frac_invalid).

SUVR = region mean / reference mean (reference: SynthSeg cerebellar cortex 8 + 47, optional erosion in
T1 space). Regions: every SynthSeg label, bilateral merges, and composites (meta-temporal, Braak
stages after Scholl et al. 2016 restricted to the parcels SynthSeg outputs, all cortex).
Interpolation orders (--orders, default 1 = trilinear and 3 = cubic B-spline) are run side by side.

Subcommands:
  run     per patient -> <out>/<patient>/{suvr.csv, patient.json, qc.png[, images/]}; resumable
  stats   <out>/*/suvr.csv -> <stats-dir>/{suvr_long.csv, stats.csv, summary.md}
  selftest  synthetic patient with known motion: geometry, composition and arm logic (no ANTs needed)

    P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
    python scripts/adni_tau_suvr.py run --inventory $P/data/adni_tau/inventory.csv \
        --static-dir $P/data/adni_tau/static --pet-processed $P/data/processed/adni_tau_pet \
        --t1-processed $P/data/processed/adni_tau_t1 --pred $P/results/predict_adni_tau/<name> \
        --out $P/results/adni_tau_suvr/<name>
    python scripts/adni_tau_suvr.py stats --root $P/results/adni_tau_suvr/<name>

Python 3.8, numpy 1.21 / scipy 1.4 / nibabel 3.1 / matplotlib 3.5 (tf10_nv.sif); ANTs 2.4.4 for `run`.
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import adni_tau_common as C  # noqa: E402
from preprocessing.luts import DK_PARCELS, SYNTHSEG_STRUCTURES  # noqa: E402
from preprocessing.nifti import read_matrix_txt  # noqa: E402

ARMS = ("native", "forward", "backward", "central")
REF_LABELS = (8, 47)                 # SynthSeg left / right cerebellum cortex
REF_ID = 9000                        # reference region id after optional erosion
N_IDS = REF_ID + 1


# ------------------------------------------------------------------ regions
def _dk(name: str) -> List[int]:
    i = DK_PARCELS.index(name) + 1
    return [1000 + i, 2000 + i]


BRAAK = OrderedDict([
    ("braak_1_2", ["entorhinal"]),        # + hippocampus (17, 53), added below
    ("braak_3", ["parahippocampal", "fusiform", "lingual"]),   # + amygdala (18, 54)
    ("braak_4", ["middletemporal", "caudalanteriorcingulate", "rostralanteriorcingulate", "posteriorcingulate",
                 "isthmuscingulate", "insula", "inferiortemporal", "temporalpole"]),
    ("braak_5", ["superiorfrontal", "lateralorbitofrontal", "medialorbitofrontal", "frontalpole",
                 "caudalmiddlefrontal", "rostralmiddlefrontal", "parsopercularis", "parsorbitalis",
                 "parstriangularis", "lateraloccipital", "supramarginal", "inferiorparietal", "superiortemporal",
                 "superiorparietal", "precuneus", "bankssts", "transversetemporal"]),
    ("braak_6", ["pericalcarine", "postcentral", "cuneus", "precentral", "paracentral"]),
])
META_TEMPORAL = ["entorhinal", "parahippocampal", "fusiform", "inferiortemporal", "middletemporal"]  # + amygdala


def region_table() -> "OrderedDict[str, List[int]]":
    """name -> member label ids. Composites first (meta_temporal = primary region)."""
    t: "OrderedDict[str, List[int]]" = OrderedDict()
    t["meta_temporal"] = sum((_dk(n) for n in META_TEMPORAL), []) + [18, 54]
    for k, names in BRAAK.items():
        ids = sum((_dk(n) for n in names), [])
        if k == "braak_1_2":
            ids += [17, 53]
        if k == "braak_3":
            ids += [18, 54]
        t[k] = ids
    t["braak_3_4"] = t["braak_3"] + t["braak_4"]
    t["braak_5_6"] = t["braak_5"] + t["braak_6"]
    t["cortex_all"] = [3, 42] + list(range(1001, 1036)) + list(range(2001, 2036))
    t["reference"] = [REF_ID]
    for n in DK_PARCELS:
        t["bi:ctx-" + n] = _dk(n)
    names = dict(SYNTHSEG_STRUCTURES)
    for lid, name in SYNTHSEG_STRUCTURES:
        if lid and lid not in REF_LABELS and name.startswith("left "):
            right = [k for k, v in names.items() if v == "right " + name[5:]]
            t["bi:" + name[5:]] = [lid] + right
    for lid, name in SYNTHSEG_STRUCTURES:
        if lid and lid not in REF_LABELS:      # cerebellar cortex = the reference region
            t[name] = [lid]
    for base, hemi in ((1000, "lh"), (2000, "rh")):
        for i, n in enumerate(DK_PARCELS):
            t["ctx-%s-%s" % (hemi, n)] = [base + i + 1]
    return t


# ------------------------------------------------------------------ geometry
def grid_points(shape: Sequence[int]) -> np.ndarray:
    """Voxel indices [V, 4] (homogeneous) in C order."""
    g = np.indices(tuple(int(s) for s in shape), dtype=np.float64).reshape(3, -1).T
    return np.c_[g, np.ones(len(g))]


def sample_labels(lab: np.ndarray, lab_aff: np.ndarray, shape, grid_aff: np.ndarray, grid_to_t1: np.ndarray,
                  k: int) -> np.ndarray:
    """T1 label ids at k^3 sub-voxel points of every grid voxel: [k^3, V] int32 (0 outside the T1)."""
    M = np.linalg.inv(lab_aff) @ grid_to_t1 @ grid_aff          # grid voxel -> T1 voxel
    pts = grid_points(shape)
    offs = (np.arange(k) + 0.5) / k - 0.5
    out = np.zeros((k ** 3, len(pts)), np.int32)
    n = 0
    lshape = np.asarray(lab.shape)
    for ox in offs:
        for oy in offs:
            for oz in offs:
                q = pts.copy()
                q[:, :3] += (ox, oy, oz)
                v = np.rint(q @ M[:3].T).astype(np.int64)
                ok = np.all((v >= 0) & (v < lshape), axis=1)
                out[n, ok] = lab[v[ok, 0], v[ok, 1], v[ok, 2]]
                n += 1
    return out


def resample(src: np.ndarray, src_aff: np.ndarray, shape, grid_aff: np.ndarray, grid_to_src_world: np.ndarray,
             order: int) -> Tuple[np.ndarray, np.ndarray]:
    """src pulled onto the grid: value(v) = src(N_src^-1 map grid_aff v). Returns (flat values, valid)."""
    M = np.linalg.inv(src_aff) @ grid_to_src_world @ grid_aff
    c = (grid_points(shape) @ M[:3].T).T
    eps = 1e-6
    valid = np.all((c >= -eps) & (c <= np.asarray(src.shape)[:, None] - 1 + eps), axis=0)
    vals = ndimage.map_coordinates(src.astype(np.float64), c, order=order, mode="nearest")
    return vals, valid


def region_sums(labels: np.ndarray, values: np.ndarray, weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """sum(value) and count over all sub-voxel samples per label id (weight 0 drops a voxel)."""
    s = np.zeros(N_IDS)
    n = np.zeros(N_IDS)
    w = weight.astype(np.float64)
    vw = values * w
    for lab in labels:
        s += np.bincount(lab, weights=vw, minlength=N_IDS)[:N_IDS]
        n += np.bincount(lab, weights=w, minlength=N_IDS)[:N_IDS]
    return s, n


def region_means(s: np.ndarray, n: np.ndarray, table) -> Dict[str, Tuple[float, float]]:
    out = {}
    for name, ids in table.items():
        ids = [i for i in ids if i < N_IDS]
        cnt = float(n[ids].sum())
        out[name] = (float(s[ids].sum()) / cnt if cnt > 0 else float("nan"), cnt)
    return out


def central_grid(target_shape, target_aff: np.ndarray, spacing: Sequence[float]) -> Tuple[Tuple[int, ...], np.ndarray]:
    """Axis-aligned grid with the given spacing covering the target (LUMIR) grid's world box."""
    lo = (target_aff @ np.r_[-0.5, -0.5, -0.5, 1])[:3]
    hi = (target_aff @ np.r_[np.asarray(target_shape) - 0.5, 1])[:3]
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    sp = np.asarray(spacing, float)
    n = np.ceil((hi - lo) / sp - 1e-6).astype(int)
    centre = (lo + hi) / 2
    aff = np.eye(4)
    aff[:3, :3] = np.diag(sp)
    aff[:3, 3] = centre - (n - 1) / 2.0 * sp
    return tuple(int(x) for x in n), aff


def world_spacing(aff: np.ndarray) -> np.ndarray:
    """Voxel size along world x, y, z (for near-axis-aligned grids: column norm of the dominant axis)."""
    a = np.asarray(aff, float)[:3, :3]
    norms = np.sqrt((a ** 2).sum(0))
    out = np.zeros(3)
    for j in range(3):
        out[int(np.argmax(np.abs(a[:, j])))] = norms[j]
    return out


# ------------------------------------------------------------------ per patient
def load_inventory(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, newline="") as f:
        return {r["patient"]: r for r in csv.DictReader(f) if r["status"] == "ok"}


def compute_patient(pid: str, inv: Dict[str, str], pet_native: Dict[str, Tuple[np.ndarray, np.ndarray]],
                    W: Dict[str, np.ndarray], lab: np.ndarray, lab_aff: np.ndarray, R: np.ndarray,
                    target_shape, target_aff: np.ndarray, orders: Sequence[int], k: int, ref_erode: int,
                    keep: Optional[Dict[str, object]] = None) -> List[Dict[str, object]]:
    """SUVR rows for one patient. W[tp]: central -> native PET world; R: central -> T1 world."""
    table = region_table()
    # Reference voxels (cerebellar cortex, optionally eroded) get REF_ID; an eroded-away rim keeps its
    # 8 / 47 id but belongs to no reported region.
    ref = np.isin(lab, REF_LABELS)
    if ref_erode > 0:
        ref = ndimage.binary_erosion(ref, iterations=ref_erode)
    lab_ref = lab.astype(np.int32).copy()
    lab_ref[ref] = REF_ID
    years = float(inv["years"])
    Winv = {tp: np.linalg.inv(W[tp]) for tp in ("bl", "fu")}
    grids = {"bl": (pet_native["bl"][0].shape, pet_native["bl"][1]),
             "fu": (pet_native["fu"][0].shape, pet_native["fu"][1])}
    sp = (world_spacing(grids["bl"][1]) + world_spacing(grids["fu"][1])) / 2
    cshape, caff = central_grid(target_shape, target_aff, sp)
    grids["central"] = (cshape, caff)
    to_central = {"bl": Winv["bl"], "fu": Winv["fu"], "central": np.eye(4)}   # grid world -> central world
    labs = {g: sample_labels(lab_ref, lab_aff, grids[g][0], grids[g][1], R @ to_central[g], k)
            for g in grids}
    rows: List[Dict[str, object]] = []

    def emit(arm, order, grid, v_bl, v_fu, weight):
        s_b, n_b = region_sums(labs[grid], v_bl, weight)
        s_f, n_f = region_sums(labs[grid], v_fu, weight)
        mb, mf = region_means(s_b, n_b, table), region_means(s_f, n_f, table)
        rb, rf = mb["reference"][0], mf["reference"][0]
        frac_invalid = float(1.0 - weight.mean())
        for name in table:
            sb, sf = mb[name][0] / rb, mf[name][0] / rf
            rows.append(OrderedDict(
                patient=pid, arm=arm, order=order, region=name, suvr_bl=sb, suvr_fu=sf, delta=sf - sb,
                rel_change=(sf - sb) / sb if sb else float("nan"), annual_delta=(sf - sb) / years, years=years,
                n_bl=mb[name][1] / k ** 3, n_fu=mf[name][1] / k ** 3, frac_invalid=frac_invalid))

    # native: each PET on its own grid, unresampled (order-independent: written once as order 0).
    xb, ab = pet_native["bl"]
    xf, af = pet_native["fu"]
    s_b, n_b = region_sums(labs["bl"], xb.ravel().astype(np.float64), np.ones(xb.size, bool))
    s_f, n_f = region_sums(labs["fu"], xf.ravel().astype(np.float64), np.ones(xf.size, bool))
    mb, mf = region_means(s_b, n_b, table), region_means(s_f, n_f, table)
    for name in table:
        sb, sf = mb[name][0] / mb["reference"][0], mf[name][0] / mf["reference"][0]
        rows.append(OrderedDict(
            patient=pid, arm="native", order=0, region=name, suvr_bl=sb, suvr_fu=sf, delta=sf - sb,
            rel_change=(sf - sb) / sb if sb else float("nan"), annual_delta=(sf - sb) / years, years=years,
            n_bl=mb[name][1] / k ** 3, n_fu=mf[name][1] / k ** 3, frac_invalid=0.0))
    for order in orders:
        # forward: follow-up pulled into the baseline grid (baseline untouched)
        v_fu, ok = resample(xf, af, xb.shape, ab, W["fu"] @ Winv["bl"], order)
        emit("forward", order, "bl", xb.ravel().astype(np.float64), v_fu, ok)
        # backward: baseline pulled into the follow-up grid
        v_bl, ok = resample(xb, ab, xf.shape, af, W["bl"] @ Winv["fu"], order)
        emit("backward", order, "fu", v_bl, xf.ravel().astype(np.float64), ok)
        # central: both pulled into the central grid
        c_bl, ok_b = resample(xb, ab, cshape, caff, W["bl"], order)
        c_fu, ok_f = resample(xf, af, cshape, caff, W["fu"], order)
        emit("central", order, "central", c_bl, c_fu, ok_b & ok_f)
        if keep is not None and order == orders[0]:
            keep.update(central_shape=cshape, central_aff=caff, central_bl=c_bl.reshape(cshape),
                        central_fu=c_fu.reshape(cshape), central_labels=labs["central"][(k ** 3) // 2].reshape(cshape))
    return rows


def _write_csv(rows: List[Dict[str, object]], path: str) -> None:
    tmp = path + ".part"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("%.6g" % v if isinstance(v, float) else v) for k, v in r.items()})
    os.replace(tmp, path)


def _save_nii(arr, aff, path, dtype=np.float32):
    o = nib.Nifti1Image(np.asarray(arr).astype(dtype), aff)
    o.set_qform(aff, code=1)
    o.set_sform(aff, code=1)
    nib.save(o, path)


def _qc_png(keep: Dict[str, object], path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bl, fu, lab = keep["central_bl"], keep["central_fu"], keep["central_labels"]
    vmax = np.percentile(np.r_[bl.ravel(), fu.ravel()], 99.5)
    ctx = np.isin(lab, [3, 42] + list(range(1001, 1036)) + list(range(2001, 2036)))
    ref = lab == REF_ID
    cx, cy, cz = [int(np.argwhere(lab > 0)[:, i].mean()) for i in range(3)]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for row, (img, name) in enumerate(((bl, "baseline"), (fu, "follow-up"))):
        for col, sl in enumerate((np.s_[cx, :, :], np.s_[:, cy, :], np.s_[:, :, cz])):
            ax = axes[row, col]
            ax.imshow(np.rot90(img[sl]), cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
            ax.contour(np.rot90(ctx[sl].astype(float)), levels=[0.5], colors="c", linewidths=0.5)
            ax.contour(np.rot90(ref[sl].astype(float)), levels=[0.5], colors="orange", linewidths=0.7)
            ax.set_axis_off()
            if col == 0:
                ax.set_title(name, fontsize=9, loc="left")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_ants_t1(atlas: str, t1: str, work: str, threads: int, ants_bin: str) -> np.ndarray:
    prefix = os.path.join(work, "t1_to_central_")
    argv = C.ants_rigid_cmd(ants_bin, atlas, t1, prefix, init=1, threads=threads)
    C.run(argv, os.path.join(work, "ants_t1.log"), threads)
    return C.read_itk_affine(prefix + "0GenericAffine.mat")     # central RAS -> T1 RAS


def cmd_run(a) -> int:
    inv_all = load_inventory(a.inventory)
    pids = a.patients.split() if a.patients else sorted(inv_all)
    orders = [int(x) for x in a.orders.split(",") if x.strip()]
    ants_bin = a.ants_bin if a.ants_bin is not None else C.default_ants_bin()
    tmp_root = os.environ.get("TMPDIR") or a.out
    os.makedirs(a.out, exist_ok=True)
    n_done = n_skip = n_fail = 0
    for pid in pids:
        od = os.path.join(a.out, pid)
        if not a.overwrite and os.path.isfile(os.path.join(od, "suvr.csv")):
            n_skip += 1
            continue
        t0 = time.time()
        try:
            if pid not in inv_all:
                raise ValueError("not a usable inventory patient")
            inv = inv_all[pid]
            fd = os.path.join(a.pred, pid, "fields")
            z = np.load(os.path.join(fd, "matrices.npz"), allow_pickle=False)
            names = [str(s) for s in z["subjects"]]
            Fw = {}
            for tp in ("bl", "fu"):
                key = "%s_%s_PET" % (pid, tp)
                if key not in names:
                    raise ValueError("%s missing from %s (has %s)" % (key, fd, names))
                Fw[tp] = np.asarray(z["fwd_world"][names.index(key)], float)
            W, rig = {}, {}
            for tp in ("bl", "fu"):
                Cm = read_matrix_txt(os.path.join(a.pet_processed, "%s_%s_PET" % (pid, tp),
                                                    "affine_to_template_world.txt"))
                W[tp] = np.linalg.inv(Cm) @ Fw[tp]
                rig[tp] = C.rigid_deviation(W[tp])
            if max(rig.values()) > 1e-3:
                raise ValueError("central -> native maps are not rigid (deviation %s)" % rig)
            pet = {}
            for tp in ("bl", "fu"):
                img = nib.load(os.path.join(a.static_dir, pid, "pet_%s.nii.gz" % tp))
                pet[tp] = (np.asanyarray(img.dataobj).astype(np.float32), np.asarray(img.affine, float))
            t1d = os.path.join(a.t1_processed, "%s_T1w" % pid)
            limg = nib.load(os.path.join(t1d, "labels_synthseg.nii.gz"))
            lab = np.asanyarray(limg.dataobj).astype(np.int32)
            atlas = os.path.join(fd, "atlas.nii.gz")
            timg = nib.load(atlas)
            os.makedirs(od, exist_ok=True)
            work = tempfile.mkdtemp(prefix="adni_suvr_", dir=tmp_root)
            try:
                R = run_ants_t1(atlas, os.path.join(t1d, "image.nii.gz"), work, a.threads, ants_bin)
                shutil.copyfile(os.path.join(work, "t1_to_central_0GenericAffine.mat"),
                                os.path.join(od, "t1_to_central_0GenericAffine.mat"))
            finally:
                shutil.rmtree(work, ignore_errors=True)
            keep: Dict[str, object] = {}
            rows = compute_patient(pid, inv, pet, W, lab, np.asarray(limg.affine, float), R,
                                   timg.shape[:3], np.asarray(timg.affine, float), orders, a.supersample,
                                   a.ref_erode, keep)
            _write_csv(rows, os.path.join(od, "suvr.csv"))
            # Displacement between the time points (native bl -> native fu) over the head of the baseline.
            pts = (grid_points(pet["bl"][0].shape)[pet["bl"][0].ravel() > 0.2 * np.percentile(pet["bl"][0], 99)]
                   @ pet["bl"][1].T)[:, :3]
            rec = {"patient": pid, "years": float(inv["years"]), "pet_bl_date": inv["pet_bl_date"],
                   "pet_fu_date": inv["pet_fu_date"], "t1_date": inv["t1_date"], "orders": orders,
                   "supersample": a.supersample, "ref_erode_vox": a.ref_erode,
                   "W_central_to_native": {tp: W[tp].tolist() for tp in W}, "rigid_deviation": rig,
                   "R_central_to_t1": R.tolist(), "central_shape": list(keep["central_shape"]),
                   "central_affine": np.asarray(keep["central_aff"]).tolist(),
                   "bl_to_fu_native_world": (W["fu"] @ np.linalg.inv(W["bl"])).tolist(),
                   "bl_to_fu_displacement": C.displacement_stats(W["bl"] @ np.linalg.inv(W["fu"]), pts),
                   "seconds": time.time() - t0}
            with open(os.path.join(od, "patient.json"), "w") as f:
                json.dump(rec, f, indent=1)
            _qc_png(keep, os.path.join(od, "qc.png"), "%s  central space (order %d), cortex cyan, reference orange"
                    % (pid, orders[0]))
            if a.save_images:
                im = os.path.join(od, "images")
                os.makedirs(im, exist_ok=True)
                _save_nii(keep["central_bl"], keep["central_aff"], os.path.join(im, "central_bl.nii.gz"))
                _save_nii(keep["central_fu"], keep["central_aff"], os.path.join(im, "central_fu.nii.gz"))
                _save_nii(keep["central_labels"], keep["central_aff"], os.path.join(im, "central_labels.nii.gz"),
                          np.int32)
            mt = [r for r in rows if r["region"] == "meta_temporal" and r["order"] in (0, orders[0])]
            print("%s: %.0f s, meta-temporal delta SUVR %s" % (
                pid, time.time() - t0, ", ".join("%s %.4f" % (r["arm"], r["delta"]) for r in mt)))
            n_done += 1
        except Exception as e:  # noqa: BLE001 (log the patient, continue)
            print("%s: FAILED: %s" % (pid, e), file=sys.stderr)
            n_fail += 1
    print("run: %d done, %d already done, %d failed" % (n_done, n_skip, n_fail))
    return 1 if n_fail else 0


# ------------------------------------------------------------------ statistics
def holm(p: Sequence[float]) -> List[float]:
    p = np.asarray([np.nan if x is None else x for x in p], float)
    out = np.full(p.shape, np.nan)
    idx = [i for i in np.argsort(p) if np.isfinite(p[i])]
    m = len(idx)
    run_max = 0.0
    for rank, i in enumerate(idx):
        run_max = max(run_max, min(1.0, (m - rank) * p[i]))
        out[i] = run_max
    return out.tolist()


def wilcoxon_p(d: np.ndarray) -> Optional[float]:
    from scipy.stats import wilcoxon
    d = d[np.isfinite(d)]
    if d.size < 2 or not np.any(d != 0):
        return None
    try:
        return float(wilcoxon(d).pvalue)
    except ValueError:
        return None


CONTRASTS = (("forward", "backward"), ("forward", "central"), ("backward", "central"), ("central", "native"),
             ("forward", "native"), ("backward", "native"))


def cmd_stats(a) -> int:
    files = sorted(glob.glob(os.path.join(a.root, "*", "suvr.csv")))
    if not files:
        print("no */suvr.csv under %s" % a.root, file=sys.stderr)
        return 2
    rows = []
    for p in files:
        with open(p, newline="") as f:
            rows += list(csv.DictReader(f))
    out_dir = a.stats_dir or os.path.join(a.root, "_stats")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "suvr_long.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    val = {}
    for r in rows:
        val[(r["patient"], r["arm"], int(r["order"]), r["region"])] = float(r[a.metric])
    patients = sorted({r["patient"] for r in rows})
    regions = list(OrderedDict.fromkeys(r["region"] for r in rows))
    orders = sorted({int(r["order"]) for r in rows if int(r["order"]) > 0})
    stats = []
    for order in orders:
        def get(arm, reg):
            o = 0 if arm == "native" else order
            return np.array([val.get((p, arm, o, reg), np.nan) for p in patients])
        block = []
        for reg in regions:
            if reg == "reference":
                continue
            d = {arm: get(arm, reg) for arm in ARMS}
            rec = OrderedDict(order=order, region=reg, n=int(np.isfinite(d["central"]).sum()))
            for arm in ARMS:
                rec["mean_" + arm] = float(np.nanmean(d[arm]))
            for x, y in CONTRASTS:
                diff = d[x] - d[y]
                rec["diff_%s_%s" % (x, y)] = float(np.nanmean(diff))
                rec["sd_%s_%s" % (x, y)] = float(np.nanstd(diff, ddof=1)) if np.isfinite(diff).sum() > 1 else float("nan")
                rec["p_%s_%s" % (x, y)] = wilcoxon_p(diff)
            lo = np.minimum(d["forward"], d["backward"])
            hi = np.maximum(d["forward"], d["backward"])
            ok = np.isfinite(lo) & np.isfinite(d["central"])
            rec["frac_central_between"] = float(((d["central"] >= lo) & (d["central"] <= hi))[ok].mean()) if ok.any() else float("nan")
            asym = np.abs(d["forward"] - d["backward"])
            rec["median_abs_fwd_bwd"] = float(np.nanmedian(asym))
            rec["median_abs_central"] = float(np.nanmedian(np.abs(d["central"])))
            block.append(rec)
        for x, y in CONTRASTS:
            key = "p_%s_%s" % (x, y)
            for rec, ph in zip(block, holm([r[key] for r in block])):
                rec[key + "_holm"] = ph
        stats += block
    keys = list(stats[0].keys())
    with open(os.path.join(out_dir, "stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in stats:
            w.writerow({k: ("" if v is None else ("%.6g" % v if isinstance(v, float) else v)) for k, v in r.items()})
    lines = ["# ADNI tau SUVR arms (%s; %d patients)" % (a.metric, len(patients)), "",
             "Paired over patients. diff = mean of (arm x - arm y); p = Wilcoxon signed-rank, Holm over regions "
             "(per contrast and order). 'between' = share of patients whose central value lies between forward "
             "and backward.", ""]
    show = [r for r in stats if r["region"] in a.summary_regions.split(",")] if a.summary_regions else stats
    for order in orders:
        lines += ["## interpolation order %d" % order, "",
                  "| region | n | native | forward | backward | central | fwd-bwd (p_holm) | fwd-cen (p_holm) | "
                  "bwd-cen (p_holm) | cen-nat (p_holm) | between |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in show:
            if r["order"] != order:
                continue

            def c(x, y):
                ph = r["p_%s_%s_holm" % (x, y)]
                return "%+.4f (%s)" % (r["diff_%s_%s" % (x, y)], "n/a" if ph is None or not np.isfinite(ph) else "%.2g" % ph)
            lines.append("| %s | %d | %.4f | %.4f | %.4f | %.4f | %s | %s | %s | %s | %.2f |" % (
                r["region"], r["n"], r["mean_native"], r["mean_forward"], r["mean_backward"], r["mean_central"],
                c("forward", "backward"), c("forward", "central"), c("backward", "central"), c("central", "native"),
                r["frac_central_between"]))
        lines.append("")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("-> %s" % out_dir)
    return 0


# ------------------------------------------------------------------ self-test
def _rot(axis, deg):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.deg2rad(deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def cmd_selftest(a) -> int:
    """Synthetic patient: one 'true' tau image in a reference world; baseline and follow-up are copies
    seen through known rigid motions on native PET grids (sampled from a 1 mm truth, no resampling chain).
    Checks: ITK .mat round trip, T1 labels reach each grid correctly, native arm == truth SUVR, and with
    an identical activity map (zero true change) every arm's delta is small and native's is ~0."""
    rng = np.random.RandomState(0)
    ok = True
    # ITK round trip
    m = np.eye(4)
    m[:3, :3] = _rot([1, 2, 3], 7)
    m[:3, 3] = [3, -2, 5]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.mat")
        C.write_itk_affine(m, p)
        err = float(np.abs(C.read_itk_affine(p) - m).max())
    print("itk round trip max err %.2e" % err)
    ok &= err < 1e-6
    # truth on a 1 mm grid (T1 space = truth space), labels: background 0, ref 8/47 block, region 1006 sphere
    tshape = (96, 112, 96)
    taff = np.eye(4)
    taff[:3, 3] = -(np.asarray(tshape) - 1) / 2.0
    g = np.indices(tshape).astype(float)
    xyz = [g[i] - (tshape[i] - 1) / 2.0 for i in range(3)]
    r = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2)
    lab = np.zeros(tshape, np.int32)
    lab[r < 40] = 2
    lab[(np.abs(xyz[0]) < 25) & (xyz[2] < -20) & (r < 40)] = 8
    lab[(np.sqrt((xyz[0] - 12) ** 2 + (xyz[1] - 5) ** 2 + (xyz[2] - 5) ** 2) < 9)] = 1006
    act = np.where(lab == 0, 0.05, 1.0)
    act[lab == 1006] = 2.0
    act = ndimage.gaussian_filter(act, 3.0)          # PET-like blur at 1 mm
    # native PET grids: 2 x 2 x 3.27 mm, LAS, different origins; motions T_i: truth world -> native world
    pshape = (64, 72, 40)
    paff = np.diag([-2.0, 2.0, 3.27, 1.0])
    paff[:3, 3] = [63.0, -71.0, -63.8]
    T = {}
    for tp, (ax, deg, tr) in {"bl": ([0, 0, 1], 4.0, [2.0, -3.0, 1.5]), "fu": ([1, 0.3, 0], -6.0, [-4.0, 2.5, 3.0])}.items():
        M = np.eye(4)
        M[:3, :3] = _rot(ax, deg)
        M[:3, 3] = tr
        T[tp] = M
    pet = {}
    for tp in ("bl", "fu"):
        # native value at voxel v = truth at T^-1 paff v (forward model, sampled from the 1 mm truth)
        vals, _ = resample(act, taff, pshape, paff, np.linalg.inv(T[tp]), 1)
        pet[tp] = (vals.reshape(pshape).astype(np.float32) * (1 + 0.01 * rng.randn(*pshape)).astype(np.float32), paff)
    # central space = truth world rotated by a midpoint G: W_i = T_i G^-1 (central -> native), R = G^-1
    G = np.eye(4)
    G[:3, :3] = _rot([0.2, 1, 0.4], 3.0)
    G[:3, 3] = [1.0, 0.5, -0.5]
    W = {tp: T[tp] @ np.linalg.inv(G) for tp in T}
    R = np.linalg.inv(G)                              # central -> T1 (= truth) world
    inv = {"years": "2.0"}
    rows = compute_patient("selftest", inv, pet, W, lab, taff, R, tshape, taff, [1, 3], 3, 0)
    by = {(r["arm"], r["order"], r["region"]): r for r in rows}
    s_true = (ndimage.mean(act, lab == 1006)) / ndimage.mean(act, lab == 8)
    for arm, order in (("native", 0), ("forward", 1), ("backward", 1), ("central", 1), ("central", 3)):
        rr = by[(arm, order, "ctx-lh-entorhinal")]
        print("%-8s order %d  suvr bl %.4f fu %.4f delta %+.5f  (truth %.4f, invalid %.3f)"
              % (arm, order, rr["suvr_bl"], rr["suvr_fu"], rr["delta"], s_true, rr["frac_invalid"]))
    nat = by[("native", 0, "ctx-lh-entorhinal")]
    ok &= abs(nat["delta"]) < 0.02 and abs(nat["suvr_bl"] - s_true) / s_true < 0.08
    for arm in ("forward", "backward", "central"):
        ok &= abs(by[(arm, 1, "ctx-lh-entorhinal")]["delta"]) < 0.05
    # a deliberately wrong W (swap) must break the native-arm agreement with the truth
    rows_bad = compute_patient("bad", inv, pet, {"bl": W["fu"], "fu": W["bl"]}, lab, taff, R, tshape, taff,
                               [1], 3, 0)
    bad = [r for r in rows_bad if r["arm"] == "native" and r["region"] == "ctx-lh-entorhinal"][0]
    print("swapped transforms: native suvr bl %.4f (truth %.4f) -> must be off" % (bad["suvr_bl"], s_true))
    ok &= abs(bad["suvr_bl"] - s_true) > abs(nat["suvr_bl"] - s_true) + 0.02
    print("SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("run")
    s.add_argument("--inventory", required=True)
    s.add_argument("--static-dir", required=True, help="native static PETs <patient>/pet_<bl|fu>.nii.gz")
    s.add_argument("--pet-processed", required=True, help="processed PET folders <patient>_<bl|fu>_PET")
    s.add_argument("--t1-processed", required=True, help="processed T1 folders <patient>_T1w (labels_synthseg)")
    s.add_argument("--pred", required=True, help="predict root with <patient>/fields/{matrices.npz,atlas.nii.gz}")
    s.add_argument("--out", required=True)
    s.add_argument("--patients", default="", help="space-separated subset (default: all usable)")
    s.add_argument("--orders", default="1,3", help="interpolation orders of the resampling arms")
    s.add_argument("--supersample", type=int, default=3, help="label samples per PET voxel and axis")
    s.add_argument("--ref-erode", type=int, default=0, help="erode the reference region by N T1 voxels")
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--ants-bin", default=None)
    s.add_argument("--save-images", action="store_true", help="also write central-space PETs + labels")
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_run)
    s = sub.add_parser("stats")
    s.add_argument("--root", required=True)
    s.add_argument("--stats-dir", default=None, help="default <root>/_stats")
    s.add_argument("--metric", default="delta", choices=("delta", "annual_delta", "rel_change", "suvr_bl", "suvr_fu"))
    s.add_argument("--summary-regions", default="meta_temporal,braak_1_2,braak_3,braak_4,braak_3_4,braak_5_6,cortex_all",
                   help="rows of summary.md ('' = all regions)")
    s.set_defaults(func=cmd_stats)
    s = sub.add_parser("selftest")
    s.set_defaults(func=cmd_selftest)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.func(a))


if __name__ == "__main__":
    sys.exit(main())
