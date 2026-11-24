#!/usr/bin/env python
"""ADNI tau-PET experiment (paper Figure 5), data preparation. Jay runs these steps (data handling).

Raw layout (after unzipping data_adni/patients_*.zip into one folder):
    <raw>/<patient>/Baseline.3D_T1_weighted.<Iid>.<series>.<date>.nii.gz
    <raw>/<patient>/Baseline.tauPET.<Iid>.<series>.<date>.nii.gz          4D, e.g. 6 x 5 min frames
    <raw>/<patient>/Follow_up_<k>.tauPET.<Iid>.<series>.<date>.nii.gz
Patient folders are named like 168_S_6051. Only the role, modality and date are read from file names.

Subcommands (run in this order):

  inventory   one row per patient: T1, baseline PET, follow-up PET (Follow_up_1 by default; --followup
              last picks the latest), dates, years between the PETs, PET shape / spacing / frames, status
              (ok, or the reason the patient is excluded). Header reads only.
  average     per usable patient and time point: rigid frame-to-mean alignment (ANTs, Mattes MI, linear
              interpolation; --align none skips it), then the mean of the aligned frames -> native static
              PET <static>/<patient>/pet_<bl|fu>.nii.gz (native grid, not skull stripped) + frames.json
              (per-frame displacement over the head). Both time points get identical treatment.
              Chunkable with --start/--end (inventory rows), resumable (done files are skipped).
  manifests   preprocessing manifests: <out>/adni_tau_pet.csv (static PETs, subject <patient>_<bl|fu>,
              contrast PET, bias_corrected true = no N4) and <out>/adni_tau_t1.csv (raw T1, contrast T1w).
  pet-labels  after `python -m preprocessing run` on the PET manifest: writes labels_mask.nii.gz
              (image > 0, uint16) into every processed PET folder; predict.py requires a label file.

    P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
    python scripts/adni_tau_prepare.py inventory --raw $P/data_adni/raw --out $P/data/adni_tau/inventory.csv
    python scripts/adni_tau_prepare.py average --inventory $P/data/adni_tau/inventory.csv \
        --static-dir $P/data/adni_tau/static --threads 4
    python scripts/adni_tau_prepare.py manifests --inventory $P/data/adni_tau/inventory.csv \
        --static-dir $P/data/adni_tau/static --out-dir $P/data/manifests
    python scripts/adni_tau_prepare.py pet-labels --processed-dir $P/data/processed/adni_tau_pet

Python 3.8, numpy 1.21 / scipy 1.4 / nibabel 3.1 (tf10_nv.sif); ANTs 2.4.4 for `average`.
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
from typing import Dict, List

import nibabel as nib
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import adni_tau_common as C  # noqa: E402
from preprocessing.manifest import ManifestRow, write_manifest  # noqa: E402

INVENTORY_COLUMNS = ["patient", "status", "notes", "t1_path", "t1_date", "pet_bl_path", "pet_bl_date",
                     "pet_fu_path", "pet_fu_date", "fu_role", "years", "pet_bl_shape", "pet_bl_spacing",
                     "pet_bl_frames", "pet_fu_shape", "pet_fu_spacing", "pet_fu_frames", "n_followups"]


def _header(path: str) -> Dict[str, object]:
    img = nib.load(path)
    shape = tuple(int(s) for s in img.shape)
    sp = np.sqrt((np.asarray(img.affine)[:3, :3] ** 2).sum(0))
    return {"shape": "x".join(str(s) for s in shape[:3]), "spacing": "x".join("%.3f" % s for s in sp),
            "frames": shape[3] if len(shape) == 4 else 1}


def cmd_inventory(a) -> int:
    pdirs = sorted(d for d in glob.glob(os.path.join(a.raw, "*")) if os.path.isdir(d))
    rows, unknown = [], 0
    for d in pdirs:
        pid = os.path.basename(d)
        if not C.PATIENT_DIR.match(pid):
            print("skip folder %s (not a patient id)" % pid, file=sys.stderr)
            continue
        files = sorted(glob.glob(os.path.join(d, "*.nii*")))
        parsed = []
        for f in files:
            info = C.parse_raw_name(f)
            if info is None or info["modality"] == "unknown":
                unknown += 1
                print("%s: unrecognised file name %s" % (pid, os.path.basename(f)), file=sys.stderr)
                continue
            info["path"] = os.path.abspath(f)
            parsed.append(info)
        t1 = [x for x in parsed if x["modality"] == "T1"]
        bl = [x for x in parsed if x["modality"] == "PET" and x["role"] == "bl"]
        fu = sorted((x for x in parsed if x["modality"] == "PET" and x["role"] != "bl"),
                    key=lambda x: (x["date"], x["followup_index"]))
        row = {k: "" for k in INVENTORY_COLUMNS}
        row.update(patient=pid, n_followups=len(fu))
        problems = []
        if len(t1) != 1:
            problems.append("%d T1 files" % len(t1))
        if len(bl) != 1:
            problems.append("%d baseline PET files" % len(bl))
        if not fu:
            problems.append("no follow-up PET")
        if len(t1) >= 1:
            t1.sort(key=lambda x: x["date"])
            row.update(t1_path=t1[0]["path"], t1_date=t1[0]["date"])
        if len(bl) >= 1:
            row.update(pet_bl_path=bl[0]["path"], pet_bl_date=bl[0]["date"])
        if fu:
            pick = fu[-1] if a.followup == "last" else fu[0]
            row.update(pet_fu_path=pick["path"], pet_fu_date=pick["date"], fu_role=pick["role"])
        notes = []
        for key in ("pet_bl", "pet_fu"):
            p = row[key + "_path"]
            if p:
                try:
                    h = _header(p)
                except Exception as e:  # noqa: BLE001 (report and continue)
                    problems.append("%s unreadable: %s" % (key, e))
                    continue
                row.update({key + "_shape": h["shape"], key + "_spacing": h["spacing"], key + "_frames": h["frames"]})
        if row["pet_bl_date"] and row["pet_fu_date"]:
            yrs = C.years_between(row["pet_bl_date"], row["pet_fu_date"])
            row["years"] = "%.4f" % yrs
            if yrs <= 0:
                problems.append("follow-up not after baseline (%.2f y)" % yrs)
        if row["t1_date"] and row["pet_bl_date"]:
            gap = C.years_between(row["t1_date"], row["pet_bl_date"])
            notes.append("T1-to-baseline-PET %.2f y" % gap)
        if row["pet_bl_spacing"] and row["pet_fu_spacing"] and row["pet_bl_spacing"] != row["pet_fu_spacing"]:
            notes.append("PET spacing differs between time points")
        row["status"] = "ok" if not problems else "excluded: " + "; ".join(problems)
        row["notes"] = "; ".join(notes)
        rows.append(row)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INVENTORY_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    ok = [r for r in rows if r["status"] == "ok"]
    print("%d patient folders, %d usable, %d excluded, %d unrecognised files -> %s"
          % (len(rows), len(ok), len(rows) - len(ok), unknown, a.out))
    for key in ("pet_bl", "pet_fu"):
        combos = {}
        for r in ok:
            c = "%s @ %s, %s frames" % (r[key + "_shape"], r[key + "_spacing"], r[key + "_frames"])
            combos[c] = combos.get(c, 0) + 1
        print("%s grids:" % key)
        for c, n in sorted(combos.items(), key=lambda x: -x[1]):
            print("  %4d  %s" % (n, c))
    if ok:
        yrs = np.array([float(r["years"]) for r in ok])
        print("years between PETs: min %.2f / median %.2f / max %.2f" % (yrs.min(), np.median(yrs), yrs.max()))
    return 0


def read_inventory(path: str, only_ok: bool = True) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["status"] == "ok"] if only_ok else rows


def _head_points(mean: np.ndarray, affine: np.ndarray, step: int = 2) -> np.ndarray:
    """World coordinates of voxels above 20 % of the 99th percentile (the head), subsampled."""
    thr = 0.2 * np.percentile(mean, 99)
    idx = np.argwhere(mean[::step, ::step, ::step] > thr) * step
    return (np.c_[idx, np.ones(len(idx))] @ np.asarray(affine, float).T)[:, :3]


def average_pet(src: str, out: str, align: str, threads: int, ants_bin: str, tmp_root: str) -> Dict[str, object]:
    """Frame alignment + mean of one 4D PET (native grid). Returns the frames.json record."""
    img = nib.load(src)
    data = np.asanyarray(img.dataobj).astype(np.float32)
    aff = np.asarray(img.affine, float)
    rec = {"source": src, "shape": list(data.shape), "align": align, "frames": []}
    if data.ndim == 3 or (data.ndim == 4 and data.shape[3] == 1):
        static = data.reshape(data.shape[:3])
        rec["note"] = "3D input: copied without averaging"
    else:
        n = data.shape[3]
        means = [float(data[..., t].mean()) for t in range(n)]
        rec["frame_means"] = means
        mean0 = data.mean(3)
        if align == "none" or n == 1:
            static = mean0
            rec["frames"] = [{"frame": t, "transform": None} for t in range(n)]
        else:
            tmp = tempfile.mkdtemp(prefix="adni_frames_", dir=tmp_root)
            try:
                fixed = os.path.join(tmp, "mean.nii.gz")
                nib.save(nib.Nifti1Image(mean0, aff), fixed)
                pts = _head_points(mean0, aff)
                aligned = np.zeros_like(data)
                for t in range(n):
                    mov = os.path.join(tmp, "f%02d.nii.gz" % t)
                    nib.save(nib.Nifti1Image(data[..., t], aff), mov)
                    prefix = os.path.join(tmp, "f%02d_" % t)
                    # Same grid: init 0 (image centres) = identity. 2 levels: PET voxels are 2-4 mm.
                    argv = C.ants_rigid_cmd(ants_bin, fixed, mov, prefix, init=0, levels="2x1", sigmas="1x0vox",
                                            iters="200x100")
                    C.run(argv, os.path.join(tmp, "ants.log"), threads)
                    m = C.read_itk_affine(prefix + "0GenericAffine.mat")   # mean point -> frame point
                    # Pull the frame onto the mean's grid: value(x) = frame(m x).
                    vox = np.linalg.inv(aff) @ m @ aff
                    from scipy import ndimage
                    aligned[..., t] = ndimage.affine_transform(data[..., t], vox[:3, :3], vox[:3, 3], order=1,
                                                               mode="constant", cval=0.0)
                    rec["frames"].append({"frame": t, "transform_ras": m.tolist(),
                                          "displacement": C.displacement_stats(m, pts)})
                static = aligned.mean(3)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    o = nib.Nifti1Image(static.astype(np.float32), aff)
    o.set_qform(aff, code=1)
    o.set_sform(aff, code=1)
    tmp_out = out + ".part.nii.gz"
    nib.save(o, tmp_out)
    os.replace(tmp_out, out)
    return rec


def cmd_average(a) -> int:
    rows = read_inventory(a.inventory)
    rows = rows[a.start:a.end]
    ants_bin = a.ants_bin if a.ants_bin is not None else C.default_ants_bin()
    tmp_root = os.environ.get("TMPDIR") or a.static_dir
    if not os.path.isdir(tmp_root) or not os.access(tmp_root, os.W_OK):
        tmp_root = a.static_dir
    os.makedirs(a.static_dir, exist_ok=True)
    n_done = n_skip = n_fail = 0
    for r in rows:
        pid = r["patient"]
        d = os.path.join(a.static_dir, pid)
        rec_path = os.path.join(d, "frames.json")
        outs = {tp: os.path.join(d, "pet_%s.nii.gz" % tp) for tp in ("bl", "fu")}
        if not a.overwrite and os.path.isfile(rec_path) and all(os.path.isfile(p) for p in outs.values()):
            n_skip += 1
            continue
        t0 = time.time()
        try:
            rec = {"patient": pid}
            for tp in ("bl", "fu"):
                rec[tp] = average_pet(r["pet_%s_path" % tp], outs[tp], a.align, a.threads, ants_bin, tmp_root)
            with open(rec_path, "w") as f:
                json.dump(rec, f, indent=1)
            mx = [fr["displacement"]["max_mm"] for tp in ("bl", "fu") for fr in rec[tp]["frames"]
                  if fr.get("displacement")]
            print("%s: done in %.0f s (max frame displacement %s mm)"
                  % (pid, time.time() - t0, "%.2f" % max(mx) if mx else "n/a"))
            n_done += 1
        except Exception as e:  # noqa: BLE001 (log the patient, continue with the others)
            print("%s: FAILED: %s" % (pid, e), file=sys.stderr)
            n_fail += 1
    print("average: %d done, %d already done, %d failed" % (n_done, n_skip, n_fail))
    return 1 if n_fail else 0


def cmd_manifests(a) -> int:
    rows = read_inventory(a.inventory)
    pet, t1, missing = [], [], 0
    for r in rows:
        pid = r["patient"]
        statics = {tp: os.path.abspath(os.path.join(a.static_dir, pid, "pet_%s.nii.gz" % tp)) for tp in ("bl", "fu")}
        if not all(os.path.isfile(p) for p in statics.values()):
            missing += 1
            continue
        for tp in ("bl", "fu"):
            pet.append(ManifestRow(dataset=a.pet_dataset, subject="%s_%s" % (pid, tp), contrast="PET",
                                   image_path=statics[tp], bias_corrected=True))
        t1.append(ManifestRow(dataset=a.t1_dataset, subject=pid, contrast="T1w", image_path=r["t1_path"]))
    pet_csv = os.path.join(a.out_dir, a.pet_dataset + ".csv")
    t1_csv = os.path.join(a.out_dir, a.t1_dataset + ".csv")
    write_manifest(pet, pet_csv)
    write_manifest(t1, t1_csv)
    print("%d patients (%d without static PETs skipped): %s (%d rows), %s (%d rows)"
          % (len(t1), missing, pet_csv, len(pet), t1_csv, len(t1)))
    return 0


def cmd_pet_labels(a) -> int:
    n = 0
    for d in sorted(glob.glob(os.path.join(a.processed_dir, "*_PET"))):
        src = os.path.join(d, "image.nii.gz")
        out = os.path.join(d, "labels_mask.nii.gz")
        if not os.path.isfile(src) or (os.path.isfile(out) and not a.overwrite):
            continue
        img = nib.load(src)
        lab = (np.asanyarray(img.dataobj) > 0).astype(np.uint16)
        o = nib.Nifti1Image(lab, img.affine)
        o.set_qform(img.affine, code=4)
        o.set_sform(img.affine, code=4)
        o.header.set_data_dtype(np.uint16)
        nib.save(o, out)
        n += 1
    print("labels_mask.nii.gz written for %d folders in %s" % (n, a.processed_dir))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("inventory")
    s.add_argument("--raw", required=True, help="folder with one <patient> folder per patient")
    s.add_argument("--out", required=True, help="inventory CSV")
    s.add_argument("--followup", choices=("first", "last"), default="first",
                   help="follow-up PET when there are several (default: first = Follow_up_1)")
    s.set_defaults(func=cmd_inventory)
    s = sub.add_parser("average")
    s.add_argument("--inventory", required=True)
    s.add_argument("--static-dir", required=True)
    s.add_argument("--align", choices=("ants", "none"), default="ants")
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--ants-bin", default=None, help="ANTs bin folder (default: Martinos ANTs 2.4.4 or $ANTS_BIN)")
    s.add_argument("--start", type=int, default=None, help="first usable inventory row (inclusive)")
    s.add_argument("--end", type=int, default=None, help="last usable inventory row (exclusive)")
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_average)
    s = sub.add_parser("manifests")
    s.add_argument("--inventory", required=True)
    s.add_argument("--static-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--pet-dataset", default="adni_tau_pet")
    s.add_argument("--t1-dataset", default="adni_tau_t1")
    s.set_defaults(func=cmd_manifests)
    s = sub.add_parser("pet-labels")
    s.add_argument("--processed-dir", required=True)
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_pet_labels)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.func(a))


if __name__ == "__main__":
    sys.exit(main())
