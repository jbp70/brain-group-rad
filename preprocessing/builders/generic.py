"""Generic manifest builder (OASIS-2, Mindboggle-101, OASIS-3, ...).

BIDS mode (root has dataset_description.json and no --image-glob):
raw ``sub-*/[ses-*/]anat/*_<suffix>`` images.

Folder mode: ``--image-glob`` relative to root; subject from the parent
folder name (default) or the file stem; optional ``--mask-glob`` and
repeatable ``--label NAME=GLOB`` (globs relative to root, ``{subject}``
placeholder). Flags (--skull-stripped etc.) apply to every row.

    python -m preprocessing build-manifest --dataset generic --root /data/Mindboggle101 --name mindboggle \\
        --image-glob '*/*/t1weighted.nii.gz' --label 'manual=*/{subject}/labels.DKT31.manual+aseg.nii.gz'
    python -m preprocessing build-manifest --dataset generic --root /data/oasis3_bids --name oasis3 --suffix T2w

Mindboggle-101, MNI152 versions (already on FSL's MNI152 1 mm grid -> no registration;
strip + N4, then world-space resampling onto the LUMIR grid; run with mask.union_labels: true):

    python -m preprocessing build-manifest --dataset generic --root /data/data_mindboggle --name mindboggle \\
        --image-glob '*/*/t1weighted.MNI152.nii.gz' \\
        --label 'dkt31aseg=*/{subject}/labels.DKT31.manual+aseg.MNI152.nii.gz' --in-template-space
"""
import argparse
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

from ..manifest import ManifestRow
from .bids import contrast_from, find_bids_images, parse_bids_name, strip_nii
from .lumir import DEFAULT_GLOB as LUMIR_GLOB

log = logging.getLogger(__name__)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--name", default="generic", help="dataset name for the rows (default %(default)s)")
    p.add_argument("--suffix", default="T1w", help="BIDS mode: image suffix (default %(default)s)")
    p.add_argument("--image-glob", default=None,
                   help="image glob relative to root (generic folder mode; lumir default %s)" % LUMIR_GLOB)
    p.add_argument("--subject-from", choices=("parent", "stem"), default="parent")
    p.add_argument("--mask-glob", default=None, help="mask glob relative to root ({subject} placeholder)")
    p.add_argument("--label", action="append", default=[], metavar="NAME=GLOB",
                   help="label glob relative to root ({subject} placeholder); repeatable")
    p.add_argument("--contrast", default="T1w", help="folder mode: contrast column (default %(default)s)")
    p.add_argument("--split", default="", help="split column value for every row")
    p.add_argument("--skull-stripped", action="store_true")
    p.add_argument("--bias-corrected", action="store_true")
    p.add_argument("--in-template-space", action="store_true")


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]", "_", s)


def _one(rootp: Path, pattern: Optional[str], subject: str, what: str) -> str:
    if not pattern:
        return ""
    hits = sorted(rootp.glob(pattern.format(subject=subject)))
    if len(hits) > 1:
        raise ValueError("%s glob matched %d files for %s: %s" % (what, len(hits), subject, hits[:3]))
    if not hits:
        log.warning("generic: no %s for %s", what, subject)
        return ""
    return str(hits[0].resolve())


def build(root: str, args: argparse.Namespace) -> List[ManifestRow]:
    rootp = Path(root)
    label_specs = []
    for spec in getattr(args, "label", []) or []:
        if "=" not in spec:
            raise ValueError("--label must be NAME=GLOB, got %r" % spec)
        label_specs.append(tuple(spec.split("=", 1)))
    entries = []  # (subject, session, contrast, path)
    if getattr(args, "image_glob", None) is None and (rootp / "dataset_description.json").is_file():
        for img in find_bids_images(root, args.suffix):
            ents, suffix = parse_bids_name(img.name)
            entries.append(("sub-" + ents["sub"], ents.get("ses", ""), contrast_from(ents, suffix), img))
    else:
        if not getattr(args, "image_glob", None):
            raise ValueError("generic: --image-glob is required outside BIDS mode")
        for img in sorted({p.resolve() for p in rootp.glob(args.image_glob) if p.is_file()}):
            subj = img.parent.name if args.subject_from == "parent" else (strip_nii(img.name) or img.stem)
            entries.append((subj, "", args.contrast, img))
    rows = []
    for raw_subj, ses, contrast, img in entries:
        # Globs get the raw folder/file name; only the manifest subject column is sanitized.
        labels: "OrderedDict[str, str]" = OrderedDict()
        for name, pattern in label_specs:
            p = _one(rootp, pattern, raw_subj, "label %s" % name)
            if p:
                labels[name] = p
        rows.append(ManifestRow(dataset=args.name, subject=_safe(raw_subj), session=ses, contrast=contrast,
                                split=args.split, image_path=str(img),
                                mask_path=_one(rootp, args.mask_glob, raw_subj, "mask"), label_paths=labels,
                                skull_stripped=args.skull_stripped, in_template_space=args.in_template_space,
                                bias_corrected=args.bias_corrected))
    log.info("generic(%s): %d images, %d with masks, %d with labels", args.name, len(rows),
             sum(1 for r in rows if r.mask_path), sum(1 for r in rows if r.label_paths))
    return rows
