"""Free disk space in already-processed outputs (`cleanup` command).

Deletes ONLY these known per-subject items inside <out_root>/<dataset>/<subject>/:
  --work     work/          intermediates (LUMIR: stripped copy of the input,
                            native mask, raw SynthSeg output; ~8.6 MB/case)
  --mask     mask.nii.gz    QC/export/qc-summary fall back to image > 0
                            (loses filled holes and in-mask voxels normalized
                            to 0; ~0.14 MB/case)
  --qc-png   qc.png         re-render with `qc-summary --png ...` (~0.25 MB/case)
Subject folders are those with a meta.json directly under the dataset folder.
Rows whose meta.json status is not "completed", or with any step recorded
as not "done" (e.g. a SynthSeg batch still running), are refused and left
untouched. Symlinks are never followed or deleted. Everything else (image,
labels, meta/qc json, logs, other files) is never touched. Deleting work/ sets
meta.json "work_cleaned" so `run` still treats the row as complete. Prints
every path with its size and the total; exit status 3 if any row was refused.

    python -m preprocessing cleanup --out-root /data/processed --dataset lumir --work --mask --qc-png --dry-run
"""
import glob
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Sequence, Tuple

from .meta import write_meta_atomic

log = logging.getLogger(__name__)


def _size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            if not os.path.islink(p):
                total += os.path.getsize(p)
    return total


def _inside(path: str, root: str) -> bool:
    path, root = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([path, root]) == root and path != root


def _check_dataset(out_root: str, dataset: str) -> str:
    if not dataset or dataset in (".", "..") or "/" in dataset or "\\" in dataset or dataset.startswith("_"):
        raise ValueError("invalid --dataset %r (plain dataset folder name expected)" % dataset)
    d = os.path.join(out_root, dataset)
    if os.path.islink(d) or not os.path.isdir(d):
        raise ValueError("no such dataset folder: %s" % d)
    return d


def _refusal(meta: Dict[str, Any]) -> str:
    if meta.get("status") != "completed":
        return "status=%s" % meta.get("status")
    bad = ["%s=%s" % (s, r.get("status")) for s, r in sorted((meta.get("steps") or {}).items())
           if r.get("status") != "done"]
    return ("steps " + ",".join(bad)) if bad else ""


def cleanup(out_root: str, datasets: Sequence[str], work: bool = False, mask: bool = False,
            qc_png: bool = False, dry_run: bool = False, echo=print) -> Dict[str, Any]:
    """Delete the selected items; returns {"deleted": [(path, bytes)], "refused": [(dir, reason)], "bytes": n}."""
    if not (work or mask or qc_png):
        raise ValueError("choose at least one of --work, --mask, --qc-png")
    out_root = os.path.abspath(out_root)
    deleted: List[Tuple[str, int]] = []
    refused: List[Tuple[str, str]] = []
    for ds in datasets:
        ds_dir = _check_dataset(out_root, ds)
        for meta_path in sorted(glob.glob(os.path.join(ds_dir, "*", "meta.json"))):
            d = os.path.dirname(meta_path)
            if os.path.islink(d) or not _inside(d, ds_dir):
                continue
            targets = []
            if work:
                targets.append((os.path.join(d, "work"), True))
            if mask:
                targets.append((os.path.join(d, "mask.nii.gz"), False))
            if qc_png:
                targets.append((os.path.join(d, "qc.png"), False))
            present = [(p, is_dir) for p, is_dir in targets if not os.path.islink(p) and
                       (os.path.isdir(p) if is_dir else os.path.isfile(p))]
            if not present:
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError) as e:
                refused.append((d, "unreadable meta.json: %s" % e))
                continue
            reason = _refusal(meta)
            if reason:
                refused.append((d, reason))
                echo("REFUSE %s (%s)" % (d, reason))
                continue
            for p, is_dir in present:
                if not _inside(p, d):
                    continue
                n = _size(p)
                echo("%s %s (%d bytes)" % ("WOULD DELETE" if dry_run else "DELETE", p, n))
                if not dry_run:
                    if is_dir:
                        shutil.rmtree(p)
                    else:
                        os.remove(p)
                deleted.append((p, n))
            if not dry_run:
                names = {os.path.basename(p) for p, _ in present}
                if "work" in names:
                    meta["work_cleaned"] = True
                if "mask.nii.gz" in names:
                    meta["mask_saved"] = False
                if names & {"work", "mask.nii.gz"}:
                    write_meta_atomic(meta, meta_path)
    total = sum(n for _, n in deleted)
    echo("%s %d items, %.1f MB%s; %d rows refused" % (
        "would delete" if dry_run else "deleted", len(deleted), total / 1e6,
        " (dry run, nothing changed)" if dry_run else "", len(refused)))
    return {"deleted": deleted, "refused": refused, "bytes": total}
