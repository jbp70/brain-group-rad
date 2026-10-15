"""LUMIR manifest builder.

Scans ``root`` for ``LUMIRMRI_<id>_0000.nii.gz`` (nnU-Net style names). The
split comes from the parent folder: imagesTr -> train, imagesVal -> val,
imagesTs -> test, otherwise "". LUMIR is already skull-stripped, N4-corrected
and on the LUMIR grid, so all flags are true.

Optional labels (e.g. LUMIR25 val 10-class maps): ``--label-glob`` relative to
root with an ``{id}`` placeholder (the subject, e.g. LUMIRMRI_3454), stored as
label ``--label-name`` (default lumir10). The label file naming is UNVERIFIED.

    python -m preprocessing build-manifest --dataset lumir --root /data/LUMIR --out lumir.csv \\
        --label-glob 'labelsVal/{id}*.nii.gz'
"""
import argparse
import logging
from collections import OrderedDict
from pathlib import Path
from typing import List

from ..manifest import ManifestRow

log = logging.getLogger(__name__)

SPLITS = {"imagestr": "train", "imagesval": "val", "imagests": "test"}
DEFAULT_GLOB = "**/LUMIRMRI_*_0000.nii.gz"


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--image-glob", default=DEFAULT_GLOB, help="image glob relative to root (default %(default)s)")
    p.add_argument("--label-glob", default=None, help="label glob relative to root with {id} placeholder")
    p.add_argument("--label-name", default="lumir10", help="label name for --label-glob (default %(default)s)")


def split_of(path: Path) -> str:
    return SPLITS.get(path.parent.name.lower(), "")


def subject_of(path: Path) -> str:
    name = path.name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    if name.endswith("_0000"):
        name = name[:-5]
    return name


def find_images(root: str, image_glob: str = DEFAULT_GLOB) -> List[Path]:
    return sorted({p.resolve() for p in Path(root).glob(image_glob) if p.is_file()})


def build(root: str, args: argparse.Namespace) -> List[ManifestRow]:
    rows = []
    rootp = Path(root)
    # --image-glob is shared with the generic builder (default None there).
    for img in find_images(root, getattr(args, "image_glob", None) or DEFAULT_GLOB):
        subj = subject_of(img)
        labels: "OrderedDict[str, str]" = OrderedDict()
        if getattr(args, "label_glob", None):
            hits = sorted(rootp.glob(args.label_glob.format(id=subj)))
            if len(hits) > 1:
                raise ValueError("label glob matched %d files for %s: %s" % (len(hits), subj, hits[:3]))
            if hits:
                labels[args.label_name] = str(hits[0].resolve())
        rows.append(ManifestRow(dataset="lumir", subject=subj, session="", contrast="T1w",
                                split=split_of(img), image_path=str(img), label_paths=labels,
                                skull_stripped=True, in_template_space=True, bias_corrected=True))
    n_lab = sum(1 for r in rows if r.label_paths)
    log.info("lumir: %d images (%d with labels)", len(rows), n_lab)
    return rows
