"""Export processed NIfTI outputs to .npy in a parallel layout.

<export_root>/<dataset>/<subject_dir>/{image.npy, labels_<name>.npy[, mask.npy]}
plus <export_root>/export_manifest.csv, <export_root>/label_luts.json and
optionally <export_root>/template/{template.npy, template_mask.npy}. Arrays
keep nibabel (x, y, z) order, C-contiguous (160x224x192 for LUMIR-grid outputs).

Defaults:
  image   float16 (values are already in [0, 1]; max abs rounding error ~2.4e-4
          near 1.0); --image-dtype float32 keeps full precision.
  labels  remapped to contiguous ids with a FIXED LUT per label name
          (preprocessing/luts.py: synthseg, lumir10, seg4, seg35), saved as
          uint8 when the LUT fits (else uint16). The LUTs used (original id ->
          new id, plus names) are written to label_luts.json and referenced in
          export_manifest.csv. A value outside the LUT is an error. Label names
          without a LUT are exported unchanged as uint16 (warning).
          --no-remap-labels keeps the original ids (uint16).
  mask    not exported; --include-mask writes mask.npy (uint8) from
          mask.nii.gz if it was saved, else image > 0 (drops filled interior holes
          and any in-mask voxel normalized to 0, see qc.load_mask).
  existing files  kept without --overwrite only if their dtype and shape
          match this run (else the export fails); after reprocessing, use
          --overwrite or delete the old export folder.

Size per LUMIR-grid case (160x224x192 = 6.88 M voxels, uncompressed .npy):
  image float16 13.8 MB + labels_synthseg uint8 6.9 MB = ~21 MB
  (+ 6.9 MB per extra uint8 label map, e.g. lumir10; + 6.9 MB with --include-mask)
  vs ~41 MB before (float32 image 27.5 MB + uint16 synthseg 13.8 MB).

    python -m preprocessing export-npy --out-root /data/processed --export-root /data/npy --dataset lumir
"""
import csv
import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import nifti
from .labels import to_label_dtype
from .luts import apply_lut, get_lut, lut_dtype, lut_json

log = logging.getLogger(__name__)

LUT_FILE = "label_luts.json"


def _save(arr: np.ndarray, path: str, overwrite: bool) -> bool:
    """Write arr to path; returns False when an existing file was kept.

    Without overwrite an existing .npy is kept only if its dtype and shape match
    what this run would write; otherwise the export fails, so the manifest and
    label_luts.json never describe arrays that are not on disk (e.g. a float32 /
    raw-uint16 export from an earlier run).
    """
    if os.path.isfile(path) and not overwrite:
        old = np.load(path, mmap_mode="r")
        if old.dtype != arr.dtype or old.shape != arr.shape:
            raise ValueError("export: existing %s is %s %s but this export writes %s %s; rerun with "
                             "--overwrite or delete the old export folder"
                             % (path, old.dtype, "x".join(map(str, old.shape)), arr.dtype,
                                "x".join(map(str, arr.shape))))
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.ascontiguousarray(arr))
    return True


def export_npy(out_root: str, export_root: str, datasets: Optional[Sequence[str]] = None,
               image_dtype: str = "float16", label_dtype: str = "auto",
               template_paths: Optional[Dict[str, str]] = None, overwrite: bool = False,
               remap_labels: bool = True, include_mask: bool = False,
               luts: Optional[Dict[str, Dict[str, object]]] = None) -> int:
    """Export all completed subjects; returns the number exported.

    label_dtype: "auto" (uint8 if the LUT fits, else uint16; uint16 without
    remapping) or an explicit "uint8"/"uint16" (error if values do not fit).
    luts: override of preprocessing.luts.LUTS (tests).
    """
    from .qc import load_mask
    if image_dtype not in ("float32", "float16"):
        raise ValueError("image_dtype must be float32 or float16")
    if label_dtype not in ("auto", "uint16", "uint8"):
        raise ValueError("label_dtype must be auto, uint16 or uint8")
    if datasets:
        ds_dirs = [os.path.join(out_root, d) for d in datasets]
    else:
        ds_dirs = sorted(d for d in glob.glob(os.path.join(out_root, "*"))
                         if os.path.isdir(d) and not os.path.basename(d).startswith("_"))
    rows: List[Dict[str, Any]] = []
    used_luts: Dict[str, Dict[str, object]] = {}
    warned = set()
    lut_path = os.path.join(export_root, LUT_FILE)
    kept: List[str] = []

    def save(arr: np.ndarray, path: str) -> None:
        if not _save(arr, path, overwrite):
            kept.append(path)

    for ds in ds_dirs:
        for meta_path in sorted(glob.glob(os.path.join(ds, "*", "meta.json"))):
            d = os.path.dirname(meta_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("status") != "completed" or not os.path.isfile(os.path.join(d, "image.nii.gz")):
                log.warning("export: skipping %s (status=%s)", d, meta.get("status"))
                continue
            dst = os.path.join(export_root, os.path.basename(ds), os.path.basename(d))
            img = nifti.load_data(os.path.join(d, "image.nii.gz"), np.float32)[0]
            save(img.astype(image_dtype), os.path.join(dst, "image.npy"))
            mask_source = ""
            if include_mask:
                mask, mask_source = load_mask(d, img)
                save(mask.astype(np.uint8), os.path.join(dst, "mask.npy"))
            labels, dtypes = [], []
            for p in sorted(glob.glob(os.path.join(d, "labels_*.nii.gz"))):
                name = os.path.basename(p)[:-len(".nii.gz")]
                lname = name[len("labels_"):]
                lab = nifti.load_data(p, np.float64)[0]
                lut = get_lut(lname, luts) if remap_labels else None
                if lut is not None:
                    dt = lut_dtype(lut) if label_dtype == "auto" else label_dtype
                    arr = apply_lut(lab, lut, dt, what=p)
                    used_luts[lname] = lut
                else:
                    if remap_labels and lname not in warned:
                        log.warning("export: no fixed LUT for label set %r; exporting original ids", lname)
                        warned.add(lname)
                    dt = "uint16" if label_dtype == "auto" else label_dtype
                    arr = to_label_dtype(lab, dt)
                save(arr, os.path.join(dst, name + ".npy"))
                labels.append(lname)
                dtypes.append("%s:%s%s" % (lname, dt, "" if lut is not None else ":raw"))
            row = meta.get("row", {})
            rows.append({"dataset": os.path.basename(ds), "subject_dir": os.path.basename(d),
                         "subject": row.get("subject", ""), "session": row.get("session", ""),
                         "contrast": row.get("contrast", ""), "split": row.get("split", ""),
                         "npy_dir": dst, "labels": ";".join(labels), "shape": "x".join(map(str, img.shape)),
                         "image_dtype": image_dtype, "label_dtypes": ";".join(dtypes),
                         "label_luts": lut_path if any(not x.endswith(":raw") for x in dtypes) else "",
                         "mask": mask_source})
    if template_paths:
        t = nifti.load_data(template_paths["template"], np.float32)[0]
        save(t.astype(image_dtype), os.path.join(export_root, "template", "template.npy"))
        tm = nifti.load_data(template_paths["template_mask"], np.float32)[0]
        save(to_label_dtype(tm, "uint8"), os.path.join(export_root, "template", "template_mask.npy"))
    if kept:
        log.warning("export-npy: kept %d existing .npy files with matching dtype/shape (contents not "
                    "re-checked; use --overwrite or delete %s after reprocessing), e.g. %s",
                    len(kept), export_root, kept[0])
    os.makedirs(export_root, exist_ok=True)
    with open(lut_path, "w", encoding="utf-8") as f:
        json.dump({"remap_labels": bool(remap_labels),
                   "luts": {n: lut_json(lut) for n, lut in sorted(used_luts.items())}}, f, indent=2)
    cols = ["dataset", "subject_dir", "subject", "session", "contrast", "split", "npy_dir", "labels", "shape",
            "image_dtype", "label_dtypes", "label_luts", "mask"]
    with open(os.path.join(export_root, "export_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    log.info("export-npy: %d subjects -> %s", len(rows), export_root)
    return len(rows)
