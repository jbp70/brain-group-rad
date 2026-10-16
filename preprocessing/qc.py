"""Quality control: per-subject metrics + optional PNG, and per-dataset summaries.

Per subject (qc.json, always written; qc.png only with outputs.save_qc_png):
  shape, affine_match_final, intensity p1/p50/p99 in mask, frac_clipped_high,
  mask_volume_cc, n_labels_<name>, dice_mask_synthseg (mask vs synthseg>0),
  dice_mask_template, ncc_to_template (inside the template mask), grid_mismatch,
  mask_source. The mask is mask.nii.gz when it was saved (outputs.save_mask),
  else image > 0 ("image>0"). That is a subset of the saved mask: normalization
  clips to [0, 1] and zeroes outside the mask, so the derived mask loses filled
  interior holes and every in-mask voxel normalized to 0 (original intensity at or
  below the lower bound: <= 0 by default, the whole lower_percentile tail when that
  is set). mask_volume_cc and the Dice values can therefore be slightly smaller.
qc.png: 2x3 mid-slices -- row 1 subject image + mask contour (red) + SynthSeg
boundaries (cyan); row 2 template + subject mask contour (registration check).

Dataset summary (`qc-summary`): robust z = (x - median) / (1.4826 MAD) for
OUTLIER_METRICS; |z| > mad_threshold flags the subject. Hard flags:
grid_mismatch, affine_match_final false, synthseg with < 1 label, status !=
completed. Writes qc_summary.csv, qc_summary.json and manifest_processed.csv.

PNGs on demand (needs --config for the template):
    python -m preprocessing qc-summary --out-root /data/processed --dataset lumir --config cfg.yaml --png flagged
    ... --png sample:20   (20 random rows, --png-seed)   ... --png all

    python -m preprocessing qc-summary --out-root /data/processed --dataset neurite_oasis
"""
import csv
import glob
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import nifti
from .manifest import COLUMNS, format_label_paths
from .masks import dice

log = logging.getLogger(__name__)

OUTLIER_METRICS = ["mask_volume_cc", "p50", "p99", "ncc_to_template", "dice_mask_template",
                   "dice_mask_synthseg"]


def label_files(subject_dir: str) -> Dict[str, str]:
    out = {}
    for p in sorted(glob.glob(os.path.join(subject_dir, "labels_*.nii.gz"))):
        name = os.path.basename(p)[len("labels_"):-len(".nii.gz")]
        out[name] = p
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else None


def load_mask(subject_dir: str, image: Optional[np.ndarray] = None) -> Tuple[np.ndarray, str]:
    """(bool mask, source): mask.nii.gz if saved, else image.nii.gz > 0.

    image > 0 drops filled interior holes and every in-mask voxel normalized to 0
    (intensity <= the normalization lower bound), see the module docstring.
    """
    p = os.path.join(subject_dir, "mask.nii.gz")
    if os.path.isfile(p):
        return nifti.load_data(p, np.uint8)[0] > 0, "mask.nii.gz"
    if image is None:
        image = nifti.load_data(os.path.join(subject_dir, "image.nii.gz"))[0]
    return np.asarray(image) > 0, "image>0"


def compute_qc(subject_dir: str, template: np.ndarray, template_affine, template_mask: np.ndarray,
               tol: float = 1e-3, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Metrics for one processed subject directory."""
    meta = meta or {}
    img_data, img = nifti.load_data(os.path.join(subject_dir, "image.nii.gz"))
    mask, mask_source = load_mask(subject_dir, img_data)
    q: Dict[str, Any] = {"shape": list(img_data.shape), "mask_source": mask_source}
    q["affine_match_final"] = nifti.grid_matches(img_data.shape, img.affine, template.shape,
                                                 template_affine, tol)
    v = img_data[mask]
    if v.size:
        q["p1"], q["p50"], q["p99"] = [float(x) for x in np.percentile(v, [1, 50, 99])]
        q["frac_at_one"] = float(np.mean(v >= 1.0))
    else:
        q["p1"] = q["p50"] = q["p99"] = q["frac_at_one"] = None
    q["frac_clipped_high"] = (meta.get("normalization") or {}).get("frac_clipped_high")
    q["mask_volume_cc"] = float(mask.sum() * np.prod(nifti.voxel_sizes(img.affine)) / 1000.0)
    q["dice_mask_template"] = dice(mask, template_mask) if mask.shape == template_mask.shape else None
    tm = template_mask > 0
    if img_data.shape == template.shape and tm.any():
        q["ncc_to_template"] = _pearson(img_data[tm], template[tm])
    else:
        q["ncc_to_template"] = None
    q["dice_mask_synthseg"] = None
    for name, p in label_files(subject_dir).items():
        lab = nifti.load_data(p, np.int64)[0]
        q["n_labels_%s" % name] = int(len(np.unique(lab[lab > 0])))
        if name == "synthseg" and lab.shape == mask.shape:
            q["dice_mask_synthseg"] = dice(mask, lab > 0)
    q["grid_mismatch"] = bool(meta.get("grid_mismatch", False))
    return q


def _boundaries(lab2d: np.ndarray) -> np.ndarray:
    b = np.zeros(lab2d.shape, bool)
    b[1:, :] |= lab2d[1:, :] != lab2d[:-1, :]
    b[:, 1:] |= lab2d[:, 1:] != lab2d[:, :-1]
    return b


def write_qc_png(path: str, image: np.ndarray, mask: np.ndarray, template: np.ndarray,
                 labels: Optional[np.ndarray] = None, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for row, base in enumerate((image, template)):
        for ax_i in range(3):
            ax = axes[row, ax_i]
            idx = base.shape[ax_i] // 2
            sl = [slice(None)] * 3
            sl[ax_i] = idx  # type: ignore[call-overload]
            sl_t = tuple(sl)
            ax.imshow(base[sl_t].T, cmap="gray", origin="lower", vmin=0, vmax=max(1e-6, float(base.max())))
            m2 = mask[sl_t].astype(float)
            if m2.any() and (m2 < 1).any():
                ax.contour(m2.T, levels=[0.5], colors="red", linewidths=0.7)
            if row == 0 and labels is not None:
                b = _boundaries(labels[sl_t])
                ov = np.ma.masked_where(~b.T, b.T.astype(float))
                ax.imshow(ov, cmap="cool", origin="lower", alpha=0.6, vmin=0, vmax=1)
            ax.set_axis_off()
            ax.set_title(("subject" if row == 0 else "template") + " axis %d slice %d" % (ax_i, idx), fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def render_qc_png(subject_dir: str, template: np.ndarray) -> str:
    """Write <subject_dir>/qc.png from the files on disk (mask via load_mask); returns the path."""
    img = nifti.load_data(os.path.join(subject_dir, "image.nii.gz"))[0]
    mask = load_mask(subject_dir, img)[0].astype(np.uint8)
    lp = os.path.join(subject_dir, "labels_synthseg.nii.gz")
    lab = nifti.load_data(lp, np.int32)[0] if os.path.isfile(lp) else None
    out = os.path.join(subject_dir, "qc.png")
    write_qc_png(out, img, mask, template, lab, title=os.path.basename(subject_dir))
    return out


def parse_png_mode(text: str) -> Tuple[str, int]:
    """'none' | 'flagged' | 'all' | 'sample:N' -> (mode, N)."""
    t = (text or "none").strip().lower()
    if t in ("none", "flagged", "all"):
        return t, 0
    if t.startswith("sample:"):
        try:
            n = int(t.split(":", 1)[1])
        except ValueError:
            n = -1
        if n > 0:
            return "sample", n
    raise ValueError("--png must be none, flagged, all or sample:N (N > 0), got %r" % text)


def select_png_dirs(dataset_dir: str, mode: str, n: int = 0, seed: int = 0) -> List[str]:
    """Subject dirs to render, using qc_summary.csv (written by summarize) for 'flagged'."""
    dirs = [d for d in _subject_dirs(dataset_dir) if os.path.isfile(os.path.join(d, "image.nii.gz"))]
    if mode == "none":
        return []
    if mode == "all":
        return dirs
    if mode == "sample":
        k = min(n, len(dirs))
        return sorted(random.Random(seed).sample(dirs, k))
    flagged = set()
    with open(os.path.join(dataset_dir, "qc_summary.csv"), "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("flagged") == "True":
                flagged.add(r["subject_dir"])
    return [d for d in dirs if os.path.basename(d) in flagged]


def robust_z(values: List[Optional[float]]) -> np.ndarray:
    """Robust z-scores; None/NaN -> 0; MAD == 0 -> all zeros."""
    x = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    ok = np.isfinite(x)
    z = np.zeros(len(x))
    if ok.sum() < 3:
        return z
    med = np.median(x[ok])
    mad = np.median(np.abs(x[ok] - med))
    if mad == 0:
        return z
    z[ok] = (x[ok] - med) / (1.4826 * mad)
    return z


def _subject_dirs(dataset_dir: str) -> List[str]:
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(dataset_dir, "*", "meta.json")))


def summarize(dataset_dir: str, mad_threshold: float = 4.0) -> Tuple[int, int]:
    """Write qc_summary.csv/.json and manifest_processed.csv; returns (n_subjects, n_flagged)."""
    rows: List[Dict[str, Any]] = []
    for d in _subject_dirs(dataset_dir):
        with open(os.path.join(d, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        q: Dict[str, Any] = {}
        qp = os.path.join(d, "qc.json")
        if os.path.isfile(qp):
            with open(qp, "r", encoding="utf-8") as f:
                q = json.load(f)
        rows.append({"dir": d, "meta": meta, "qc": q})
    zs = {m: robust_z([r["qc"].get(m) for r in rows]) for m in OUTLIER_METRICS}
    metric_cols = sorted({k for r in rows for k in r["qc"] if k != "shape"})
    out_rows = []
    n_flag = 0
    for i, r in enumerate(rows):
        meta, q = r["meta"], r["qc"]
        reasons = []
        for m in OUTLIER_METRICS:
            if abs(zs[m][i]) > mad_threshold:
                reasons.append("%s_z=%.1f" % (m, zs[m][i]))
        status = meta.get("status", "unknown")
        if status != "completed":
            reasons.append("status=%s" % status)
        if not q:
            reasons.append("missing_qc")
        if q.get("grid_mismatch") or meta.get("grid_mismatch"):
            reasons.append("grid_mismatch")
        if q and q.get("affine_match_final") is False:
            reasons.append("affine_mismatch_final")
        if "n_labels_synthseg" in q and q["n_labels_synthseg"] < 1:
            reasons.append("synthseg_empty")
        row = meta.get("row", {})
        d = {"subject_dir": os.path.basename(r["dir"]), "dataset": row.get("dataset", ""),
             "subject": row.get("subject", ""), "session": row.get("session", ""),
             "contrast": row.get("contrast", ""), "status": status}
        for c in metric_cols:
            d[c] = q.get(c)
        for m in OUTLIER_METRICS:
            d["z_" + m] = round(float(zs[m][i]), 3)
        d["flagged"] = bool(reasons)
        d["reasons"] = ";".join(reasons)
        n_flag += int(bool(reasons))
        out_rows.append(d)
    cols = ["subject_dir", "dataset", "subject", "session", "contrast", "status"] + metric_cols + \
        ["z_" + m for m in OUTLIER_METRICS] + ["flagged", "reasons"]
    with open(os.path.join(dataset_dir, "qc_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    pcols = COLUMNS + ["status", "out_dir", "image", "mask", "labels", "affine_to_template"]
    with open(os.path.join(dataset_dir, "manifest_processed.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pcols)
        w.writeheader()
        for r in rows:
            row = dict(r["meta"].get("row", {}))
            row["label_paths"] = format_label_paths(row.get("label_paths") or {})
            for b in ("skull_stripped", "in_template_space", "bias_corrected"):
                row[b] = "true" if row.get(b) else "false"
            d = r["dir"]
            aff = os.path.join(d, "affine_to_template.mat")
            mpath = os.path.join(d, "mask.nii.gz")
            row.update({"status": r["meta"].get("status", "unknown"), "out_dir": d,
                        "image": os.path.join(d, "image.nii.gz"),
                        "mask": mpath if os.path.isfile(mpath) else "",   # empty: use image > 0
                        "labels": format_label_paths(label_files(d)),
                        "affine_to_template": aff if os.path.isfile(aff) else ""})
            w.writerow({c: row.get(c, "") for c in pcols})
    summary = {"n_subjects": len(rows), "n_flagged": n_flag, "mad_threshold": mad_threshold}
    tmp = os.path.join(dataset_dir, "qc_summary.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, os.path.join(dataset_dir, "qc_summary.json"))
    log.info("qc-summary %s: %d subjects, %d flagged", dataset_dir, len(rows), n_flag)
    return len(rows), n_flag
