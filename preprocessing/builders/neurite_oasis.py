"""Neurite-OASIS manifest builder.

Every folder under ``root`` that contains ``norm.nii.gz`` is one subject
(subject = folder name, e.g. OASIS_OAS1_0001_MR1 -- folder naming is an
assumption to confirm on disk). Uses the scanner-space 256^3 files only:
``norm.nii.gz`` (or ``orig.nii.gz`` with --image orig) plus ``seg35``/``seg4``
labels. ``aligned_*`` (MNI305, cropped) and ``slice_*`` files are never used.

norm: skull-stripped + bias-corrected (FreeSurfer) -> SynthStrip/N4 skipped,
mask derived from (norm > 0) | (seg35 > 0). orig: full pipeline.

    python -m preprocessing build-manifest --dataset neurite_oasis --root /data/neurite-oasis --out oasis.csv
"""
import argparse
import logging
from collections import OrderedDict
from pathlib import Path
from typing import List

from ..manifest import ManifestRow

log = logging.getLogger(__name__)
LABELS = ("seg35", "seg4")


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--image", choices=("norm", "orig"), default="norm",
                   help="scanner-space image to use (default %(default)s)")


def build(root: str, args: argparse.Namespace) -> List[ManifestRow]:
    which = getattr(args, "image", "norm")
    rows = []
    # Exact filename match, so aligned_norm.nii.gz / slice_norm.nii.gz never match.
    for norm in sorted(Path(root).rglob("norm.nii.gz")):
        d = norm.parent
        img = d / ("%s.nii.gz" % which)
        if not img.is_file():
            log.warning("neurite_oasis: %s missing, skipping %s", img.name, d)
            continue
        labels: "OrderedDict[str, str]" = OrderedDict()
        for name in LABELS:
            p = d / ("%s.nii.gz" % name)
            if p.is_file():
                labels[name] = str(p.resolve())
        pre = which == "norm"
        rows.append(ManifestRow(dataset="neurite_oasis", subject=d.name, contrast="T1w",
                                image_path=str(img.resolve()), label_paths=labels,
                                skull_stripped=pre, in_template_space=False, bias_corrected=pre))
    log.info("neurite_oasis: %d subjects (%d with seg35)", len(rows),
             sum(1 for r in rows if "seg35" in r.label_paths))
    return rows
