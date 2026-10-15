"""UltraCortex (OpenNeuro ds005216, BIDS) manifest builder.

Raw images: ``sub-*/[ses-*/]anat/*_T1w.nii[.gz]`` (derivatives excluded).
Derivatives (``derivatives/**``) are matched by sub/ses (and acq/run when the
derivative names them):
  * brain masks: filename contains "mask" (paths containing "synthstrip" win);
  * manual labels: path contains "manual" or suffix "dseg" (label name "manual").
The exact derivative folder/file names are UNVERIFIED -- check the logged
match counts (expected: 86 images, 12 with manual labels).

N4 is always run (bias_corrected=false). Provided masks are used unless
--ignore-provided-masks (then SynthStrip runs).

    python -m preprocessing build-manifest --dataset ultracortex --root /data/ds005216 --out uc.csv
"""
import argparse
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

from ..manifest import ManifestRow
from .bids import contrast_from, find_bids_images, matches, parse_bids_name, strip_nii

log = logging.getLogger(__name__)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ignore-provided-masks", action="store_true",
                   help="do not use derivative masks (run SynthStrip instead)")
    p.add_argument("--suffix", default="T1w", help="BIDS suffix of raw images (default %(default)s)")


def scan_derivatives(root: Path) -> Tuple[List[Tuple[Path, Dict[str, str]]], List[Tuple[Path, Dict[str, str]]]]:
    masks, labels = [], []
    droot = root / "derivatives"
    if not droot.is_dir():
        return masks, labels
    for p in sorted(droot.rglob("*.nii*")):
        if strip_nii(p.name) is None:
            continue
        try:
            ents, suffix = parse_bids_name(p.name)
        except ValueError:
            continue
        if "sub" not in ents:
            continue
        low = str(p.relative_to(root)).lower()
        if "mask" in p.name.lower():
            masks.append((p.resolve(), ents))
        elif "manual" in low or suffix == "dseg":
            labels.append((p.resolve(), ents))
    return masks, labels


def _pick(cands: List[Path], prefer: str, what: str, subj: str) -> Path:
    preferred = [c for c in cands if prefer in str(c).lower()]
    pool = preferred or cands
    if len(pool) > 1:
        log.warning("ultracortex: %d %s candidates for %s, using %s", len(pool), what, subj, pool[0])
    return pool[0]


def build(root: str, args: argparse.Namespace) -> List[ManifestRow]:
    rootp = Path(root)
    masks, labels = scan_derivatives(rootp)
    ignore_masks = getattr(args, "ignore_provided_masks", False)
    rows = []
    for img in find_bids_images(root, getattr(args, "suffix", "T1w")):
        ents, suffix = parse_bids_name(img.name)
        subj = "sub-" + ents["sub"]
        m = [p for p, e in masks if matches(ents, e)]
        lab = [p for p, e in labels if matches(ents, e)]
        mask_path = ""
        if m and not ignore_masks:
            mask_path = str(_pick(m, "synthstrip", "mask", subj))
        label_paths: "OrderedDict[str, str]" = OrderedDict()
        if lab:
            label_paths["manual"] = str(_pick(lab, "manual", "label", subj))
        rows.append(ManifestRow(dataset="ultracortex", subject=subj, session=ents.get("ses", ""),
                                contrast=contrast_from(ents, suffix), image_path=str(img),
                                mask_path=mask_path, label_paths=label_paths,
                                skull_stripped=False, in_template_space=False, bias_corrected=False))
    log.info("ultracortex: %d images, %d with derivative masks%s, %d with manual labels "
             "(derivatives scanned: %d mask files, %d label files)", len(rows),
             sum(1 for r in rows if r.mask_path), " (ignored)" if ignore_masks else "",
             sum(1 for r in rows if r.label_paths), len(masks), len(labels))
    return rows
