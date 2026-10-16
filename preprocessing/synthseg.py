"""SynthSeg labels in batch mode (one mri_synthseg call, model loaded once).

For the selected subject dirs (image.nii.gz present; labels_synthseg.nii.gz
missing unless overwrite), writes inputs.txt / outputs.txt / vols.txt /
qc.txt into the batch dir and runs a single

    mri_synthseg --i inputs.txt --o outputs.txt [--parc] [--robust] --vol vols.txt --qc qc.txt --threads N [--cpu]

(txt-list mode per the SynthSeg README; verify --vol/--qc txt support on the
cluster version). Raw outputs go to <subject>/work/synthseg_raw.nii.gz; each
is checked against the target grid -- a mismatch is nearest-resampled (world
space) and logged as ``synthseg_resampled`` in meta.json -- then saved as
labels_synthseg.nii.gz (uint16, target affine), re-read and verified, and QC
is refreshed. Then, for rows whose status is "completed", work/ is deleted
(unless outputs.keep_work / --keep-work). Only image.nii.gz is needed as input,
so the batch works after `run` has already removed work/.

    python -m preprocessing synthseg --config cfg.yaml --manifest oasis.csv --out-root /data/processed --start 0 --end 100
"""
import json
import logging
import os
import shutil
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import nifti
from .commands import synthseg_cmd
from .grid import Target
from .labels import resample_labels_nearest, to_label_dtype
from .meta import load_meta, now, record_step, write_meta_atomic
from .tools import ToolRunner

log = logging.getLogger(__name__)


def postprocess_synthseg(subject_dir: str, raw_path: str, target: Target) -> Dict[str, Any]:
    """Raw SynthSeg output -> labels_synthseg.nii.gz on the target grid (uint16)."""
    data, img = nifti.load_data(raw_path, np.float64)
    resampled = False
    if not nifti.grid_matches(data.shape, img.affine, target.shape, target.affine, target.tol):
        log.warning("%s: SynthSeg output grid %s differs from target %s; nearest resampling",
                    subject_dir, data.shape, target.shape)
        data = resample_labels_nearest(data, img.affine, target.shape, target.affine)
        resampled = True
    lab = to_label_dtype(data, "uint16")
    out = os.path.join(subject_dir, "labels_synthseg.nii.gz")
    nifti.save_like(lab, target.affine, out, np.uint16, xform_code=4)
    return {"synthseg_resampled": resampled, "raw_shape": list(img.shape[:3]),
            "n_labels": int(len(np.unique(lab[lab > 0])))}


def verify_labels(path: str, target: Target) -> None:
    """Re-read a written label file and check it is on the target grid (before work/ is deleted)."""
    data, img = nifti.load_data(path, np.int64)
    if not nifti.grid_matches(data.shape, img.affine, target.shape, target.affine, target.tol):
        raise ValueError("%s is not on the target grid after writing" % path)


def refresh_qc(subject_dir: str, target: Target, cfg: Dict[str, Any], meta: Dict[str, Any]) -> None:
    """Rewrite qc.json (always) and qc.png (only with outputs.save_qc_png)."""
    from .qc import compute_qc, render_qc_png
    q = compute_qc(subject_dir, target.data(), target.affine, target.mask_data(), target.tol, meta)
    with open(os.path.join(subject_dir, "qc.json"), "w", encoding="utf-8") as f:
        json.dump(q, f, indent=2)
    if cfg["outputs"]["save_qc_png"]:
        render_qc_png(subject_dir, target.data())


def clean_work_dir(subject_dir: str, meta: Dict[str, Any]) -> bool:
    """Delete <subject_dir>/work (a real directory, not a symlink) and set meta['work_cleaned']."""
    work = os.path.join(subject_dir, "work")
    if os.path.islink(work) or not os.path.isdir(work):
        return False
    shutil.rmtree(work)
    meta["work_cleaned"] = True
    return True


def run_synthseg_batch(subject_dirs: List[str], cfg: Dict[str, Any], runner: ToolRunner, batch_dir: str,
                       target: Optional[Target], overwrite: bool = False,
                       echo: Optional[Callable[[str], None]] = None,
                       clean_work: Optional[bool] = None) -> Dict[str, int]:
    """Run one SynthSeg call over all pending subjects; returns counts.

    clean_work (default: not outputs.keep_work): after a row's labels are
    written and verified and its QC refreshed, delete its work/ (which holds
    the raw SynthSeg output) if the row's status is "completed". The per-row
    `run --steps ...,synthseg` path passes False and cleans at the end of the row.
    """
    echo = echo or print
    if clean_work is None:
        clean_work = not cfg["outputs"]["keep_work"]
    pending = []
    for d in subject_dirs:
        img = os.path.join(d, "image.nii.gz")
        # Trust an existing output only if meta.json recorded it as done (not a half-written file).
        done = (os.path.isfile(os.path.join(d, "labels_synthseg.nii.gz")) and
                load_meta(os.path.join(d, "meta.json")).get("steps", {}).get("synthseg", {}).get("status") == "done")
        if done and not overwrite:
            continue
        if not os.path.isfile(img) and not runner.dry_run:
            log.warning("%s: image.nii.gz missing; skipping SynthSeg (run the pipeline first)", d)
            continue
        pending.append(d)
    counts = {"pending": len(pending), "done": 0, "failed": 0, "skipped": len(subject_dirs) - len(pending)}
    if not pending:
        log.info("synthseg: nothing to do (%d already done or missing images)", len(subject_dirs))
        return counts
    files = {
        "inputs": [os.path.join(d, "image.nii.gz") for d in pending],
        "outputs": [os.path.join(d, "work", "synthseg_raw.nii.gz") for d in pending],
        "vols": [os.path.join(d, "synthseg_vols.csv") for d in pending],
        "qc": [os.path.join(d, "synthseg_qc.csv") for d in pending],
    }
    lists = {k: os.path.join(batch_dir, "%s.txt" % k) for k in files}
    sc = cfg["synthseg"]
    argv = synthseg_cmd(runner.tool(cfg, "mri_synthseg"), lists["inputs"], lists["outputs"], sc,
                        vol=lists["vols"], qc=lists["qc"])
    if runner.dry_run:
        for k in files:
            echo("PY synthseg: write %s (%d lines)" % (lists[k], len(files[k])))
        runner.run(argv, os.path.join(batch_dir, "synthseg.log"))
        echo("PY synthseg: postprocess %d outputs -> labels_synthseg.nii.gz" % len(pending))
        return counts
    os.makedirs(batch_dir, exist_ok=True)
    for k, lines in files.items():
        with open(lists[k], "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    started = now()
    for d in pending:
        os.makedirs(os.path.join(d, "work"), exist_ok=True)
        # Mark running: a concurrent `cleanup` refuses the row, and a killed job is redone.
        meta_path = os.path.join(d, "meta.json")
        meta = load_meta(meta_path)
        record_step(meta, "synthseg", "running", outputs=[os.path.join(d, "labels_synthseg.nii.gz")],
                    started=started)
        write_meta_atomic(meta, meta_path)
    log.info("synthseg: one call over %d subjects", len(pending))
    runner.run(argv, os.path.join(batch_dir, "synthseg.log"))
    assert target is not None
    for d, raw in zip(pending, files["outputs"]):
        meta_path = os.path.join(d, "meta.json")
        meta = load_meta(meta_path)
        labels_path = os.path.join(d, "labels_synthseg.nii.gz")
        try:
            info = postprocess_synthseg(d, raw, target)
            verify_labels(labels_path, target)
        except Exception as e:  # keep going for the rest of the batch
            log.error("%s: SynthSeg postprocess failed: %s", d, e)
            record_step(meta, "synthseg", "failed", params={"error": str(e)}, started=started)
            write_meta_atomic(meta, meta_path)
            counts["failed"] += 1
            continue
        meta["synthseg_resampled"] = info["synthseg_resampled"]
        record_step(meta, "synthseg", "done", outputs=[labels_path],
                    commands=[argv], params=dict(info, batch_dir=batch_dir), started=started,
                    log=os.path.join(batch_dir, "synthseg.log"))
        write_meta_atomic(meta, meta_path)
        refresh_qc(d, target, cfg, meta)
        if clean_work and meta.get("status") == "completed" and clean_work_dir(d, meta):
            write_meta_atomic(meta, meta_path)
        counts["done"] += 1
    return counts
