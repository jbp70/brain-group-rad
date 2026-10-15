"""Export processed NIfTI outputs to .npy in a parallel layout.

<export_root>/<dataset>/<subject_dir>/{image.npy, mask.npy, labels_<name>.npy}
plus <export_root>/export_manifest.csv and optionally
<export_root>/template/{template.npy, template_mask.npy}. Arrays keep nibabel
(x, y, z) order, C-contiguous (160x224x192 for LUMIR-grid outputs).

    python -m preprocessing export-npy --out-root /data/processed --export-root /data/npy --image-dtype float16
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

log = logging.getLogger(__name__)


def _save(arr: np.ndarray, path: str, overwrite: bool) -> None:
    if os.path.isfile(path) and not overwrite:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.ascontiguousarray(arr))


def export_npy(out_root: str, export_root: str, datasets: Optional[Sequence[str]] = None,
               image_dtype: str = "float32", label_dtype: str = "uint16",
               template_paths: Optional[Dict[str, str]] = None, overwrite: bool = False) -> int:
    """Export all completed subjects; returns the number exported."""
    if image_dtype not in ("float32", "float16"):
        raise ValueError("image_dtype must be float32 or float16")
    if label_dtype not in ("uint16", "uint8"):
        raise ValueError("label_dtype must be uint16 or uint8")
    if datasets:
        ds_dirs = [os.path.join(out_root, d) for d in datasets]
    else:
        ds_dirs = sorted(d for d in glob.glob(os.path.join(out_root, "*"))
                         if os.path.isdir(d) and not os.path.basename(d).startswith("_"))
    rows: List[Dict[str, Any]] = []
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
            _save(img.astype(image_dtype), os.path.join(dst, "image.npy"), overwrite)
            mask = nifti.load_data(os.path.join(d, "mask.nii.gz"), np.float32)[0]
            _save(to_label_dtype(mask, "uint8"), os.path.join(dst, "mask.npy"), overwrite)
            labels = []
            for p in sorted(glob.glob(os.path.join(d, "labels_*.nii.gz"))):
                name = os.path.basename(p)[:-len(".nii.gz")]
                lab = nifti.load_data(p, np.float64)[0]
                _save(to_label_dtype(lab, label_dtype), os.path.join(dst, name + ".npy"), overwrite)
                labels.append(name[len("labels_"):])
            row = meta.get("row", {})
            rows.append({"dataset": os.path.basename(ds), "subject_dir": os.path.basename(d),
                         "subject": row.get("subject", ""), "session": row.get("session", ""),
                         "contrast": row.get("contrast", ""), "split": row.get("split", ""),
                         "npy_dir": dst, "labels": ";".join(labels), "shape": "x".join(map(str, img.shape))})
    if template_paths:
        t = nifti.load_data(template_paths["template"], np.float32)[0]
        _save(t.astype(image_dtype), os.path.join(export_root, "template", "template.npy"), overwrite)
        tm = nifti.load_data(template_paths["template_mask"], np.float32)[0]
        _save(to_label_dtype(tm, "uint8"), os.path.join(export_root, "template", "template_mask.npy"), overwrite)
    os.makedirs(export_root, exist_ok=True)
    cols = ["dataset", "subject_dir", "subject", "session", "contrast", "split", "npy_dir", "labels", "shape"]
    with open(os.path.join(export_root, "export_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    log.info("export-npy: %d subjects -> %s", len(rows), export_root)
    return len(rows)
