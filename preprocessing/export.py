"""Export processed NIfTI outputs to per-level .npy files (training export v2).

Layout (one folder per case; suffix "" = full resolution, _2 = 1/2, _4 = 1/4):

  <export_root>/<dataset>/<subject_dir>/
      image.npy  image_2.npy  image_4.npy                       (160x224x192, 80x112x96, 40x56x48)
      labels_train18.npy  labels_train18_2.npy  labels_train18_4.npy
      [labels_synthseg.npy] [labels_<provided>.npy, e.g. labels_lumir10.npy] [mask.npy]
  <export_root>/manifest_parts/<part>.csv, <part>.luts.json   (one per export-npy run / chunk)
  <export_root>/export_manifest.csv, label_luts.json            (written by the merge step)
  <export_root>/template/template[_k].npy, template_mask[_k].npy, template_info.json
                                                                 (--include-template)

Arrays keep nibabel (x, y, z) order, C-contiguous. Use a NEW export root for
this layout (e.g. $D/npy_v2); never mix it with an older export.

Levels (--levels, default 1,2,4; every dimension must be divisible by k, else
the export fails):
  image   EXACT block average over non-overlapping k x k x k blocks, computed in
          float64 from the full-resolution float32 image.nii.gz (never from a
          float16 export), then cast to --image-dtype.
  labels  train18 only: merged first (fixed LUT, preprocessing/luts.py), then
          block majority vote (mode of each block). Tie-break: the SMALLEST
          merged label id wins (deterministic; e.g. a 4/4 background/cortex block
          becomes background). synthseg (103-id remap) and provided label sets
          (lumir10, seg4, seg35, ...) are FULL RESOLUTION ONLY (evaluation).
  affine  level k voxel q has its center at full-resolution index k*q + (k-1)/2
          (2q+0.5, 4q+1.5), so A_k = A_1 @ [[k,0,0,(k-1)/2], ...] where A_1 is the
          corrected target-grid affine stamped on image.nii.gz. The world center
          (index (n-1)/2) is identical at every level. export_manifest.csv stores
          per-level file, shape, spacing, affine and world center.
  template  image block-averaged; mask by block majority vote (tie -> 0, i.e. a
          coarse voxel is brain only when MORE than half of its block is brain).

Defaults:
  image       float32: augmentation (warps, intensity) runs in float32 in DALI and
              the cast to float16 for mixed precision happens at the model input.
              --image-dtype float16 halves the image size (values in [0, 1]; max
              abs rounding error ~2.4e-4 near 1.0).
  label sets  --label-sets train18 (default). Opt-in: synthseg (the 103-id
              SynthSeg --parc remap, uint8) and provided labels such as lumir10.
              A validation export passes --optional-label-sets lumir10. train18 and
              synthseg are read from the RAW SynthSeg ids in labels_synthseg.nii.gz;
              any id missing from a fixed LUT fails the export (ids and case listed).
              A requested label set missing for a case fails the export, except
              sets in --optional-label-sets (evaluation only, added to --label-sets;
              never train18): a case without one skips that file, is logged, and
              lists the set in the manifest column missing_optional_labels
              (';'-separated; empty otherwise). LUMIR val: 10 of the 39 T1w cases
              (3484-3493, landmarks only) have no lumir10 map.
              Provided label names without a fixed LUT are exported unchanged as
              uint16 (warning); --no-remap-labels keeps original ids (not with train18).
  contrast    --contrast T1w (default; repeatable or comma list, matched against the
              manifest contrast or its last '_' token). This drops LUMIR's 10 non-T1
              unknown0001 validation cases. --contrast any exports every contrast.
  split       --split train|val|... (repeatable or comma list; default any).
  mask        not exported; --include-mask writes mask.npy (uint8, full resolution)
              from mask.nii.gz if it was saved, else image > 0.

Row selection and chunking (SLURM arrays, slurm/export_array.sh):
  --start/--end/--rows index the rows of <out_root>/<dataset>/manifest_processed.csv
  (written by qc-summary; datasets concatenated in order). Filters apply inside
  the selection. Each run writes its own manifest part (named after the
  datasets, filters and row range), so concurrent chunks never write the same
  file. A run WITHOUT a row selection merges all parts into export_manifest.csv
  and label_luts.json at the end; after chunked runs, run `export-merge` once
  all chunks are done.

Writes are atomic (temp file in the same folder, then rename), so a killed job
never leaves a truncated .npy. Resume: a case is skipped when every requested
file exists with the expected dtype and shape (an optional label set without a
source for that case is not requested for it); a file with a different dtype or
shape fails the export (it belongs to another export configuration);
--overwrite rewrites everything.

Size per LUMIR case (uncompressed): image 27.5 / 3.4 / 0.43 MB (float32, full/_2/_4),
labels_train18 6.9 / 0.86 / 0.11 MB (uint8): ~39 MB; ~135 GB for 3,433 cases.
(+6.9 MB per full-resolution extra label map, e.g. lumir10 or synthseg.)

    python -m preprocessing export-npy --out-root /data/processed --export-root /data/npy_v2 --dataset lumir
    python -m preprocessing export-npy ... --split val --label-sets train18 --optional-label-sets lumir10
    python -m preprocessing export-npy ... --split train --start 0 --end 200   (one chunk)
    python -m preprocessing export-merge --export-root /data/npy_v2
    python -m preprocessing export-check --export-root /data/npy_v2 --sample 20
"""
import csv
import glob
import hashlib
import io
import json
import logging
import os
import random
import re
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import nifti
from .labels import to_label_dtype
from .luts import apply_lut, get_lut, lut_dtype, lut_json

log = logging.getLogger(__name__)

LUT_FILE = "label_luts.json"
MANIFEST_FILE = "export_manifest.csv"
PARTS_DIR = "manifest_parts"
TEMPLATE_INFO = "template_info.json"
ALL_LEVELS = (1, 2, 4)
DEFAULT_LEVELS = (1, 2, 4)
DEFAULT_LABEL_SETS = ("train18",)
DEFAULT_CONTRASTS = ("T1w",)
# Label sets read from labels_synthseg.nii.gz (raw SynthSeg ids); others from labels_<name>.nii.gz.
SYNTHSEG_SETS = ("train18", "synthseg")
# Label sets exported at every requested level; all others are full resolution only.
PYRAMID_LABEL_SETS = ("train18",)
_NAME = re.compile(r"^[A-Za-z0-9_-]+$")

MISSING_OPTIONAL = "missing_optional_labels"
BASE_COLUMNS = ["dataset", "subject_dir", "subject", "session", "contrast", "split", "npy_dir", "levels",
                "shape", "image_dtype", "labels", "label_dtypes", "label_levels", "label_luts", "mask",
                MISSING_OPTIONAL]
LEVEL_FIELDS = ["image", "shape", "labels", "spacing", "affine", "world_center"]
MANIFEST_COLUMNS = BASE_COLUMNS + ["%s_l%d" % (f, k) for k in ALL_LEVELS for f in LEVEL_FIELDS]
# Parts written before missing_optional_labels existed (merged with that column empty).
OLD_MANIFEST_COLUMNS = [c for c in MANIFEST_COLUMNS if c != MISSING_OPTIONAL]


# ----------------------------------------------------------------- geometry / pooling

def level_suffix(k: int) -> str:
    """File name suffix of level k: "" (full resolution), "_2", "_4"."""
    return "" if k == 1 else "_%d" % k


def _check_divisible(shape: Sequence[int], k: int, what: str = "array") -> None:
    bad = [int(n) for n in shape if int(n) % k]
    if bad:
        raise ValueError("%s: shape %s is not divisible by %d (--levels); dimensions %s"
                         % (what, "x".join(str(int(n)) for n in shape), k, bad))


def block_average(arr: np.ndarray, k: int, what: str = "image") -> np.ndarray:
    """Exact mean over non-overlapping k x k x k blocks, in float64."""
    _check_divisible(arr.shape, k, what)
    x, y, z = (int(n) // k for n in arr.shape)
    return np.asarray(arr).reshape(x, k, y, k, z, k).mean(axis=(1, 3, 5), dtype=np.float64)


def block_mode(lab: np.ndarray, k: int, what: str = "labels") -> np.ndarray:
    """Majority vote over non-overlapping k x k x k blocks; ties -> the smallest label id.

    Vectorized: one bincount over (block index, label) pairs; argmax returns the
    first (smallest) id among equal counts. Returns the input dtype.
    """
    if not np.issubdtype(lab.dtype, np.integer):
        raise ValueError("%s: block_mode needs an integer label map, got %s" % (what, lab.dtype))
    _check_divisible(lab.shape, k, what)
    x, y, z = (int(n) // k for n in lab.shape)
    b = np.asarray(lab).reshape(x, k, y, k, z, k).transpose(0, 2, 4, 1, 3, 5).reshape(-1, k ** 3)
    if b.size and int(b.min()) < 0:
        raise ValueError("%s: negative label values" % what)
    vals = None
    top = int(b.max()) if b.size else 0
    if top >= 1024:  # sparse ids (e.g. raw FreeSurfer): compress to ranks, order preserved
        vals, inv = np.unique(b, return_inverse=True)
        b = inv.reshape(b.shape)
        top = len(vals) - 1
    n_cls = top + 1
    nb = b.shape[0]
    idx = np.arange(nb, dtype=np.int64)[:, None] * n_cls + b.astype(np.int64)
    counts = np.bincount(idx.ravel(), minlength=nb * n_cls).reshape(nb, n_cls)
    win = counts.argmax(axis=1)
    if vals is not None:
        win = vals[win]
    return win.astype(lab.dtype).reshape(x, y, z)


def level_affine(affine, k: int) -> np.ndarray:
    """Voxel -> world affine of level k: coarse index q -> full index k*q + (k-1)/2."""
    s = np.diag([float(k), float(k), float(k), 1.0])
    s[:3, 3] = (k - 1) / 2.0
    return np.asarray(affine, dtype=np.float64) @ s


def world_center(shape: Sequence[int], affine) -> np.ndarray:
    """World coordinate of the volume center, voxel index (n - 1) / 2."""
    a = np.asarray(affine, dtype=np.float64)
    c = (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0
    return a[:3, :3] @ c + a[:3, 3]


def _fmt_vec(v, sep: str = " ") -> str:
    return sep.join("%.10g" % float(x) for x in np.asarray(v).ravel())


def _parse_vec(text: str) -> np.ndarray:
    return np.array([float(x) for x in text.replace("x", " ").split()], dtype=np.float64)


# ----------------------------------------------------------------- atomic files

def _atomic_save_npy(path: str, arr: np.ndarray) -> None:
    """np.save to a temp file in the same folder, then rename (no truncated .npy)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = "%s.tmp%d" % (path, os.getpid())
    try:
        with open(tmp, "wb") as f:
            np.save(f, np.ascontiguousarray(arr))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _atomic_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = "%s.tmp%d" % (path, os.getpid())
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _csv_text(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in cols})
    return buf.getvalue()


def _npy_state(path: str, dtype: str, shape: Sequence[int]) -> Tuple[str, str]:
    """("missing"|"unreadable"|"mismatch"|"ok", description of the existing file)."""
    if not os.path.isfile(path):
        return "missing", ""
    try:
        old = np.load(path, mmap_mode="r")
        dt, sh = old.dtype, tuple(old.shape)
        del old
    except Exception:  # truncated / not an .npy: rewrite
        return "unreadable", ""
    if dt != np.dtype(dtype) or sh != tuple(int(n) for n in shape):
        return "mismatch", "%s %s" % (dt, "x".join(map(str, sh)))
    return "ok", ""


# (path, dtype, shape, producer)
Item = Tuple[str, str, Tuple[int, ...], Callable[[], np.ndarray]]


def _needs_write(items: List[Item], overwrite: bool) -> bool:
    """True when the items must be (re)written; raises on dtype/shape mismatch without overwrite."""
    if overwrite:
        return True
    states = [(it, _npy_state(it[0], it[1], it[2])) for it in items]
    for it, (st, desc) in states:
        if st == "mismatch":
            raise ValueError("export: existing %s is %s but this export writes %s %s; rerun with "
                             "--overwrite or use a new --export-root"
                             % (it[0], desc, it[1], "x".join(map(str, it[2]))))
    return any(st != "ok" for _, (st, _d) in states)


def _write_items(items: List[Item]) -> None:
    for path, dtype, shape, make in items:
        arr = make()
        if arr.dtype != np.dtype(dtype) or tuple(arr.shape) != tuple(shape):  # internal consistency
            raise AssertionError("%s: produced %s %s, planned %s %s" % (path, arr.dtype, arr.shape, dtype, shape))
        _atomic_save_npy(path, arr)


# ----------------------------------------------------------------- case selection

def _split_list(values: Optional[Sequence[str]]) -> List[str]:
    out: List[str] = []
    for v in values or []:
        for x in str(v).split(","):
            x = x.strip()
            if x and x not in out:
                out.append(x)
    return out


def _matches(value: str, wanted: Optional[List[str]]) -> bool:
    if not wanted:
        return True
    value = value or ""
    return value in wanted or value.split("_")[-1] in wanted


def _dataset_dirs(out_root: str, datasets: Optional[Sequence[str]]) -> List[str]:
    if datasets:
        return [os.path.join(out_root, d) for d in datasets]
    return sorted(d for d in glob.glob(os.path.join(out_root, "*"))
                  if os.path.isdir(d) and not os.path.basename(d).startswith("_"))


def case_list(out_root: str, datasets: Optional[Sequence[str]] = None,
              require_manifest: bool = False) -> List[Tuple[str, str]]:
    """[(dataset_dir, subject_dir)] in manifest_processed.csv order, datasets concatenated.

    Without manifest_processed.csv (qc-summary not run) the subject folders with a
    meta.json are used in sorted order (same order qc-summary writes), unless
    require_manifest (row selection) is set.
    """
    out: List[Tuple[str, str]] = []
    for ds in _dataset_dirs(out_root, datasets):
        mp = os.path.join(ds, "manifest_processed.csv")
        if os.path.isfile(mp):
            with open(mp, "r", newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    name = os.path.basename(os.path.normpath(r.get("out_dir") or ""))
                    if name:
                        out.append((ds, os.path.join(ds, name)))
        elif require_manifest:
            raise ValueError("export: --start/--end/--rows index %s, which does not exist; run qc-summary "
                             "first" % mp)
        else:
            log.warning("export: %s not found (run qc-summary); using the subject folders in sorted order", mp)
            out += [(ds, os.path.dirname(p)) for p in sorted(glob.glob(os.path.join(ds, "*", "meta.json")))]
    return out


def select_cases(cases: List[Tuple[str, str]], start: Optional[int], end: Optional[int],
                 rows: Optional[Sequence[int]]) -> List[Tuple[str, str]]:
    if rows is not None:
        if start is not None or end is not None:
            raise ValueError("use either --rows or --start/--end")
        bad = [i for i in rows if i < 0 or i >= len(cases)]
        if bad:
            raise ValueError("row indices out of range (0..%d): %s" % (len(cases) - 1, bad))
        return [cases[i] for i in rows]
    s = 0 if start is None else max(0, start)
    e = len(cases) if end is None else min(len(cases), end)
    return cases[s:e]


def part_name(datasets: Optional[Sequence[str]], splits: List[str], contrasts: List[str],
              start: Optional[int], end: Optional[int], rows: Optional[Sequence[int]]) -> str:
    """Stable, unique-per-selection manifest part name (no two concurrent chunks share one)."""
    if rows is not None:
        sel = "rows-" + hashlib.sha1(",".join(str(i) for i in rows).encode()).hexdigest()[:10]
    elif start is None and end is None:
        sel = "rows-all"
    else:
        sel = "rows-%s-%s" % (0 if start is None else start, "end" if end is None else end)
    parts = ["+".join(datasets) if datasets else "all", "split-" + ("+".join(splits) or "any"),
             "contrast-" + ("+".join(contrasts) or "any"), sel]
    return re.sub(r"[^A-Za-z0-9._+-]", "_", "_".join(parts))


# ----------------------------------------------------------------- export

def _label_source(subject_dir: str, label_set: str) -> str:
    name = "synthseg" if label_set in SYNTHSEG_SETS else label_set
    return os.path.join(subject_dir, "labels_%s.nii.gz" % name)


def export_npy(out_root: str, export_root: str, datasets: Optional[Sequence[str]] = None,
               image_dtype: str = "float32", label_dtype: str = "auto",
               template_paths: Optional[Dict[str, str]] = None, overwrite: bool = False,
               remap_labels: bool = True, include_mask: bool = False,
               luts: Optional[Dict[str, Dict[str, object]]] = None,
               levels: Sequence[int] = DEFAULT_LEVELS, label_sets: Sequence[str] = DEFAULT_LABEL_SETS,
               contrasts: Optional[Sequence[str]] = DEFAULT_CONTRASTS, splits: Optional[Sequence[str]] = None,
               start: Optional[int] = None, end: Optional[int] = None,
               rows: Optional[Sequence[int]] = None,
               optional_label_sets: Optional[Sequence[str]] = None) -> int:
    """Export the selected completed cases; returns the number of cases in this run's manifest part.

    label_dtype: "auto" (uint8 if the LUT fits, else uint16; uint16 without
    remapping) or an explicit "uint8"/"uint16" (error if values do not fit).
    optional_label_sets: evaluation-only provided sets (e.g. lumir10), added to
    label_sets; a case without one skips that file and lists it in the
    manifest column missing_optional_labels. train18 cannot be optional.
    contrasts: None/[]/["any"] = no contrast filter. luts: override of
    preprocessing.luts.LUTS (tests). Without start/end/rows the manifest parts
    are merged into export_manifest.csv at the end (merge_parts).
    """
    from .qc import load_mask
    if image_dtype not in ("float32", "float16"):
        raise ValueError("image_dtype must be float32 or float16")
    if label_dtype not in ("auto", "uint16", "uint8"):
        raise ValueError("label_dtype must be auto, uint16 or uint8")
    levels = sorted({int(k) for k in levels})
    if not levels or any(k not in ALL_LEVELS for k in levels):
        raise ValueError("levels must be a non-empty subset of %s, got %s" % (ALL_LEVELS, levels))
    optional = _split_list(optional_label_sets)
    if "train18" in optional:
        raise ValueError("train18 is the training label set and cannot be optional; drop it from "
                         "--optional-label-sets")
    sets = _split_list(list(label_sets or []) + optional)
    for ls in sets:
        if not _NAME.match(ls):
            raise ValueError("invalid label set name %r" % ls)
        if ls not in PYRAMID_LABEL_SETS and 1 not in levels:
            raise ValueError("label set %r is exported at full resolution only; add 1 to --levels" % ls)
    if not remap_labels and "train18" in sets:
        raise ValueError("train18 is a remapped label set; drop --no-remap-labels or train18")
    want_contrast = [c for c in _split_list(contrasts) if c.lower() != "any"]
    want_split = [s for s in _split_list(splits) if s.lower() != "any"]
    selected = start is not None or end is not None or rows is not None
    cases = select_cases(case_list(out_root, datasets, require_manifest=selected), start, end, rows)

    lut_path = os.path.join(export_root, LUT_FILE)
    out_rows: List[Dict[str, Any]] = []
    used_luts: Dict[str, Dict[str, object]] = {}
    warned = set()
    n = {"contrast": 0, "split": 0, "status": 0, "written": 0, "kept": 0}
    n_missing: Dict[str, int] = {}

    for ds, d in cases:
        meta_path = os.path.join(d, "meta.json")
        if not os.path.isfile(meta_path):
            log.warning("export: skipping %s (no meta.json)", d)
            n["status"] += 1
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        row = meta.get("row", {})
        if not _matches(row.get("contrast", ""), want_contrast):
            n["contrast"] += 1
            continue
        if not _matches(row.get("split", ""), want_split):
            n["split"] += 1
            continue
        img_path = os.path.join(d, "image.nii.gz")
        if meta.get("status") != "completed" or not os.path.isfile(img_path):
            log.warning("export: skipping %s (status=%s)", d, meta.get("status"))
            n["status"] += 1
            continue
        dst = os.path.join(export_root, os.path.basename(ds), os.path.basename(d))
        hdr = nifti.load(img_path)
        shape = nifti.spatial_shape(hdr)
        aff = np.asarray(hdr.affine, dtype=np.float64)
        for k in levels:
            _check_divisible(shape, k, img_path)
        cache: Dict[str, Any] = {}

        def image() -> np.ndarray:
            if "img" not in cache:
                cache["img"] = nifti.load_data(img_path, np.float32)[0]
            return cache["img"]

        def image_level(k: int) -> Callable[[], np.ndarray]:
            if k == 1:
                return lambda: image().astype(image_dtype)
            return lambda: block_average(image(), k, img_path).astype(image_dtype)

        items: List[Item] = []
        lshape = {k: tuple(int(s) // k for s in shape) for k in levels}
        for k in levels:
            items.append((os.path.join(dst, "image%s.npy" % level_suffix(k)), image_dtype, lshape[k],
                          image_level(k)))
        label_info: List[Tuple[str, str, bool, List[int]]] = []  # (set, dtype, remapped, levels)
        missing: List[str] = []
        for ls in sets:
            src = _label_source(d, ls)
            if not os.path.isfile(src) and ls in optional:
                missing.append(ls)
                n_missing[ls] = n_missing.get(ls, 0) + 1
                continue
            if not os.path.isfile(src):
                raise ValueError("export: %s has no %s (needed for label set %r); drop it from --label-sets "
                                 "or restrict --split/--contrast" % (d, os.path.basename(src), ls))
            lut = get_lut(ls, luts) if remap_labels else None
            if lut is not None:
                dt = lut_dtype(lut) if label_dtype == "auto" else label_dtype
                used_luts[ls] = lut
            else:
                if remap_labels and ls not in warned:
                    log.warning("export: no fixed LUT for label set %r; exporting original ids", ls)
                    warned.add(ls)
                dt = "uint16" if label_dtype == "auto" else label_dtype
            ls_levels = levels if ls in PYRAMID_LABEL_SETS else [1]
            label_info.append((ls, dt, lut is not None, ls_levels))

            def full_labels(ls=ls, src=src, lut=lut, dt=dt) -> np.ndarray:
                key = "lab:" + ls
                if key not in cache:
                    raw_key = "raw:" + src
                    if raw_key not in cache:
                        cache[raw_key] = nifti.load_data(src, np.float64)[0]
                    raw = cache[raw_key]
                    if lut is not None:
                        cache[key] = apply_lut(raw, lut, dt, what="%s (case %s, label set %s)"
                                               % (src, os.path.basename(d), ls))
                    else:
                        cache[key] = to_label_dtype(raw, dt)
                return cache[key]

            for k in ls_levels:
                make = full_labels if k == 1 else (lambda k=k, fl=full_labels, ls=ls: block_mode(
                    fl(), k, "%s labels_%s" % (d, ls)))
                items.append((os.path.join(dst, "labels_%s%s.npy" % (ls, level_suffix(k))), dt, lshape.get(
                    k, tuple(shape)), make))
        if missing:
            log.info("export: %s has no %s (optional label set%s); skipped, listed in %s", d,
                     ", ".join(os.path.basename(_label_source(d, ls)) for ls in missing),
                     "s" if len(missing) > 1 else "", MISSING_OPTIONAL)
        mask_source = ""
        if include_mask:
            mask_source = "mask.nii.gz" if os.path.isfile(os.path.join(d, "mask.nii.gz")) else "image>0"
            items.append((os.path.join(dst, "mask.npy"), "uint8", tuple(shape),
                          lambda: load_mask(d, image())[0].astype(np.uint8)))
        if _needs_write(items, overwrite):
            for stale in glob.glob(os.path.join(dst, "*.tmp[0-9]*")):  # leftovers of a killed job
                os.remove(stale)
            _write_items(items)
            n["written"] += 1
        else:
            n["kept"] += 1
        cache.clear()

        r: Dict[str, Any] = {
            "dataset": os.path.basename(ds), "subject_dir": os.path.basename(d),
            "subject": row.get("subject", ""), "session": row.get("session", ""),
            "contrast": row.get("contrast", ""), "split": row.get("split", ""), "npy_dir": dst,
            "levels": ";".join(str(k) for k in levels), "shape": "x".join(map(str, shape)),
            "image_dtype": image_dtype, "labels": ";".join(s for s, _, _, _ in label_info),
            "label_dtypes": ";".join("%s:%s%s" % (s, dt, "" if rm else ":raw") for s, dt, rm, _ in label_info),
            "label_levels": ";".join("%s:%s" % (s, ",".join(map(str, lv))) for s, _, _, lv in label_info),
            "label_luts": lut_path if any(rm for _, _, rm, _ in label_info) else "", "mask": mask_source,
            MISSING_OPTIONAL: ";".join(missing)}
        for k in levels:
            ak = level_affine(aff, k)
            r["image_l%d" % k] = "image%s.npy" % level_suffix(k)
            r["shape_l%d" % k] = "x".join(map(str, lshape[k]))
            r["labels_l%d" % k] = ";".join("labels_%s%s.npy" % (s, level_suffix(k))
                                           for s, _, _, lv in label_info if k in lv)
            r["spacing_l%d" % k] = _fmt_vec(nifti.voxel_sizes(ak), "x")
            r["affine_l%d" % k] = _fmt_vec(ak)
            r["world_center_l%d" % k] = _fmt_vec(world_center(lshape[k], ak))
        out_rows.append(r)

    if template_paths:
        _export_template(template_paths, export_root, levels, image_dtype, overwrite)
    want_c, want_s = want_contrast, want_split
    name = part_name(datasets, want_s, want_c, start, end, rows)
    pdir = os.path.join(export_root, PARTS_DIR)
    _atomic_write_text(os.path.join(pdir, name + ".csv"), _csv_text(out_rows, MANIFEST_COLUMNS))
    _atomic_write_text(os.path.join(pdir, name + ".luts.json"), json.dumps(
        {"remap_labels": bool(remap_labels),
         "luts": OrderedDict((k, lut_json(v)) for k, v in sorted(used_luts.items()))}, indent=2))
    log.info("export-npy: %d cases in part %s (%d written, %d already complete and kept); skipped %d by "
             "--contrast %s, %d by --split %s, %d not completed", len(out_rows), name, n["written"], n["kept"],
             n["contrast"], ",".join(want_c) or "any", n["split"], ",".join(want_s) or "any", n["status"])
    for ls in optional:
        log.info("export-npy: optional label set %s missing for %d of %d cases (listed in %s)", ls,
                 n_missing.get(ls, 0), len(out_rows), MISSING_OPTIONAL)
    if n["kept"]:
        log.info("export-npy: kept %d complete cases (dtype/shape checked, contents not re-checked; use "
                 "--overwrite after reprocessing)", n["kept"])
    if not selected:
        merge_parts(export_root)
    else:
        log.info("export-npy: row selection given; run `export-merge --export-root %s` after all chunks finish",
                 export_root)
    return len(out_rows)


def _export_template(tp: Dict[str, str], export_root: str, levels: Sequence[int], image_dtype: str,
                     overwrite: bool) -> None:
    """template[_k].npy (block average) and template_mask[_k].npy (block majority vote, tie -> 0)."""
    hdr = nifti.load(tp["template"])
    shape = nifti.spatial_shape(hdr)
    aff = np.asarray(hdr.affine, dtype=np.float64)
    for k in levels:
        _check_divisible(shape, k, tp["template"])
    cache: Dict[str, np.ndarray] = {}

    def t() -> np.ndarray:
        if "t" not in cache:
            cache["t"] = nifti.load_data(tp["template"], np.float32)[0]
        return cache["t"]

    def m() -> np.ndarray:
        if "m" not in cache:
            cache["m"] = to_label_dtype(nifti.load_data(tp["template_mask"], np.float32)[0], "uint8")
        return cache["m"]

    tdir = os.path.join(export_root, "template")
    items: List[Item] = []
    info: Dict[str, Any] = {"source": {"template": tp["template"], "template_mask": tp["template_mask"]},
                            "image_dtype": image_dtype, "mask_rule": "block majority vote, tie -> 0",
                            "levels": OrderedDict()}
    for k in levels:
        sh = tuple(int(s) // k for s in shape)
        sfx = level_suffix(k)
        if k == 1:
            items.append((os.path.join(tdir, "template.npy"), image_dtype, sh, lambda: t().astype(image_dtype)))
            items.append((os.path.join(tdir, "template_mask.npy"), "uint8", sh, m))
        else:
            items.append((os.path.join(tdir, "template%s.npy" % sfx), image_dtype, sh,
                          lambda k=k: block_average(t(), k, "template").astype(image_dtype)))
            items.append((os.path.join(tdir, "template_mask%s.npy" % sfx), "uint8", sh,
                          lambda k=k: block_mode(m(), k, "template_mask")))
        ak = level_affine(aff, k)
        info["levels"][str(k)] = {"image": "template%s.npy" % sfx, "mask": "template_mask%s.npy" % sfx,
                                  "shape": list(sh), "spacing": [float(x) for x in nifti.voxel_sizes(ak)],
                                  "affine": ak.tolist(), "world_center": world_center(sh, ak).tolist()}
    if _needs_write(items, overwrite):
        _write_items(items)
    _atomic_write_text(os.path.join(tdir, TEMPLATE_INFO), json.dumps(info, indent=2))


# ----------------------------------------------------------------- merge

def merge_parts(export_root: str) -> int:
    """Union of manifest_parts/*.csv -> export_manifest.csv, *.luts.json -> label_luts.json.

    Rows are keyed by (dataset, subject_dir); identical duplicates (overlapping
    chunks) collapse, conflicting ones fail (delete the stale part). Returns the row count.
    """
    pdir = os.path.join(export_root, PARTS_DIR)
    parts = sorted(glob.glob(os.path.join(pdir, "*.csv")))
    if not parts:
        raise ValueError("export-merge: no manifest parts in %s (run export-npy first)" % pdir)
    rows: "OrderedDict[Tuple[str, str], Dict[str, str]]" = OrderedDict()
    origin: Dict[Tuple[str, str], str] = {}
    for p in parts:
        with open(p, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if list(reader.fieldnames or []) not in (MANIFEST_COLUMNS, OLD_MANIFEST_COLUMNS):
                raise ValueError("export-merge: %s has different columns (older export version?)" % p)
            for r in reader:
                r = {c: r.get(c) or "" for c in MANIFEST_COLUMNS}  # old parts: missing_optional_labels empty
                key = (r["dataset"], r["subject_dir"])
                if key in rows and rows[key] != r:
                    raise ValueError("export-merge: %s/%s differs between %s and %s (different export options); "
                                     "delete the stale part and rerun" % (key[0], key[1], origin[key], p))
                rows[key] = dict(r)
                origin.setdefault(key, p)
    luts: Dict[str, Any] = {}
    flags = set()
    for p in parts:
        jp = p[:-len(".csv")] + ".luts.json"
        if not os.path.isfile(jp):
            raise ValueError("export-merge: %s is missing" % jp)
        with open(jp, "r", encoding="utf-8") as f:
            j = json.load(f)
        flags.add(bool(j.get("remap_labels", True)))
        for name, lut in j.get("luts", {}).items():
            if name in luts and luts[name] != lut:
                raise ValueError("export-merge: LUT %r differs between parts (%s)" % (name, jp))
            luts[name] = lut
    if len(flags) > 1:
        raise ValueError("export-merge: parts mix remapped and --no-remap-labels exports; use separate roots")
    out = [rows[k] for k in sorted(rows)]
    _atomic_write_text(os.path.join(export_root, LUT_FILE), json.dumps(
        {"remap_labels": flags.pop(), "luts": OrderedDict(sorted(luts.items()))}, indent=2))
    _atomic_write_text(os.path.join(export_root, MANIFEST_FILE), _csv_text(out, MANIFEST_COLUMNS))
    log.info("export-merge: %d parts -> %d cases in %s", len(parts), len(out),
             os.path.join(export_root, MANIFEST_FILE))
    return len(out)


# ----------------------------------------------------------------- check

def export_check(export_root: str, sample: int = 20, seed: int = 0) -> Tuple[int, List[str], List[str]]:
    """Verify files listed in export_manifest.csv: dtype/shape, label range, world centers.

    sample > 0 checks that many random cases (seed), 0 checks all. Returns
    (n_checked, problems, report lines).
    """
    with open(os.path.join(export_root, MANIFEST_FILE), "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n_cls: Dict[str, int] = {}
    lp = os.path.join(export_root, LUT_FILE)
    if os.path.isfile(lp):
        with open(lp, "r", encoding="utf-8") as f:
            n_cls = {k: int(v["n_labels"]) for k, v in json.load(f).get("luts", {}).items()}
    if sample and sample < len(rows):
        rows = random.Random(seed).sample(rows, sample)
    problems: List[str] = []
    report: List[str] = []
    label_max: Dict[str, int] = {}
    max_dev = 0.0
    for r in rows:
        case = "%s/%s" % (r["dataset"], r["subject_dir"])
        d = os.path.join(export_root, r["dataset"], r["subject_dir"])
        dts = {}
        for x in r["label_dtypes"].split(";"):
            if x:
                p = x.split(":")
                dts[p[0]] = p[1]
        centers = []
        for k in [int(x) for x in r["levels"].split(";") if x]:
            sh = tuple(int(s) for s in r["shape_l%d" % k].split("x"))
            files = [(r["image_l%d" % k], r["image_dtype"], None)]
            sfx = level_suffix(k)
            for fn in [x for x in r["labels_l%d" % k].split(";") if x]:
                ls = fn[len("labels_"):-len(".npy")]
                ls = ls[:-len(sfx)] if sfx else ls
                files.append((fn, dts.get(ls, ""), ls))
            for fn, dt, ls in files:
                st, desc = _npy_state(os.path.join(d, fn), dt, sh)
                if st != "ok":
                    problems.append("%s: %s %s (%s; expected %s %s)" % (case, fn, st, desc, dt, r["shape_l%d" % k]))
                    continue
                if ls is not None:
                    top = int(np.load(os.path.join(d, fn), mmap_mode="r").max())
                    label_max[ls] = max(label_max.get(ls, 0), top)
                    if ls in n_cls and top >= n_cls[ls]:
                        problems.append("%s: %s max label %d outside 0..%d" % (case, fn, top, n_cls[ls] - 1))
            ak = _parse_vec(r["affine_l%d" % k]).reshape(4, 4)
            wc = world_center(sh, ak)
            if np.max(np.abs(wc - _parse_vec(r["world_center_l%d" % k]))) > 1e-4:
                problems.append("%s: world_center_l%d does not match affine_l%d" % (case, k, k))
            centers.append((k, wc))
        for k, wc in centers[1:]:
            dev = float(np.max(np.abs(wc - centers[0][1])))
            max_dev = max(max_dev, dev)
            if dev > 1e-4:
                problems.append("%s: world center of level %d differs by %.6g mm from level %d"
                                % (case, k, dev, centers[0][0]))
    if rows:
        r = rows[0]
        report.append("example %s/%s:" % (r["dataset"], r["subject_dir"]))
        for k in [int(x) for x in r["levels"].split(";") if x]:
            report.append("  level %d: %s %s spacing %s labels [%s] world center (%s)"
                          % (k, r["shape_l%d" % k], r["image_dtype"], r["spacing_l%d" % k], r["labels_l%d" % k],
                             r["world_center_l%d" % k]))
    report.append("label max over checked cases: %s" % ", ".join(
        "%s=%d (0..%s)" % (k, v, n_cls[k] - 1 if k in n_cls else "raw") for k, v in sorted(label_max.items())))
    report.append("max world-center difference between levels: %.3g mm" % max_dev)
    return len(rows), problems, report
