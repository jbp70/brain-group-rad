"""LUMIR manifest builder.

Scans ``root`` for ``LUMIRMRI_<id>_<chan>.nii.gz`` (nnU-Net style names,
default glob ``**/images*/LUMIRMRI_*_000[0-9].nii.gz``: only images* folders,
because label folders such as simplelabelsVal reuse the image file names). The split comes from the
parent folder: imagesTr -> train, imagesVal -> val, imagesTs -> test,
otherwise "". LUMIR is already skull-stripped, N4-corrected and on the LUMIR
grid, so all flags are true (no registration, so no contrast-dependent
template choice is made for these rows).

Contrast comes from the nnU-Net channel suffix:
  _0000            -> contrast T1w, subject LUMIRMRI_<id>           (unchanged ids)
  _0001 .. _0009   -> contrast unknown<chan> (e.g. unknown0001), subject
                      LUMIRMRI_<id>_<chan> (LUMIR25 extra validation cases,
                      non-T1 contrast; deliberately not guessed as T2/FLAIR)

Optional labels (e.g. LUMIR25 simplelabelsVal 10-class maps): ``--label-glob``
relative to root with placeholders ``{id}`` (the subject id above) and
``{name}`` (the image file stem, e.g. LUMIRMRI_3454_0000), stored as label
``--label-name`` (default lumir10). Label files named for a different
channel are ignored (``{id}*`` for LUMIRMRI_3454 never picks
LUMIRMRI_3454_0001.nii.gz); if several files still match, the one named like
the image is used.

    python -m preprocessing build-manifest --dataset lumir --root /data/LUMIR --out lumir.csv \\
        --image-glob 'imagesVal/LUMIRMRI_*_000[0-9].nii.gz' --label-glob 'simplelabelsVal/{id}*.nii.gz'
"""
import argparse
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

from ..manifest import ManifestRow

log = logging.getLogger(__name__)

SPLITS = {"imagestr": "train", "imagesval": "val", "imagests": "test"}
# images* folders only: LUMIR25 label folders (simplelabelsVal, SanityCheckLabelsTr) reuse the image names.
DEFAULT_GLOB = "**/images*/LUMIRMRI_*_000[0-9].nii.gz"
_NAME = re.compile(r"^(LUMIRMRI_.+)_(\d{4})$")


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--image-glob", default=DEFAULT_GLOB, help="image glob relative to root (default %(default)s)")
    p.add_argument("--label-glob", default=None,
                   help="label glob relative to root with {id} (subject) or {name} (image stem) placeholder")
    p.add_argument("--label-name", default="lumir10", help="label name for --label-glob (default %(default)s)")


def split_of(path: Path) -> str:
    return SPLITS.get(path.parent.name.lower(), "")


def _stem(path: Path) -> str:
    name = path.name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def parse_name(path: Path) -> Tuple[str, str]:
    """(subject, contrast) from the file name (see module docstring)."""
    stem = _stem(path)
    m = _NAME.match(stem)
    if not m:
        return stem, "T1w"
    if m.group(2) == "0000":
        return m.group(1), "T1w"
    return stem, "unknown" + m.group(2)


def _other_channel(label: Path, image: Path) -> bool:
    """True if the label file is named for a different channel (e.g. _0001 label vs _0000 image)."""
    ml, mi = _NAME.match(_stem(label)), _NAME.match(_stem(image))
    return bool(ml and mi and ml.group(2) != mi.group(2))


def subject_of(path: Path) -> str:
    return parse_name(path)[0]


def find_images(root: str, image_glob: str = DEFAULT_GLOB) -> List[Path]:
    return sorted({p.resolve() for p in Path(root).glob(image_glob) if p.is_file()})


def build(root: str, args: argparse.Namespace) -> List[ManifestRow]:
    rows = []
    rootp = Path(root)
    # --image-glob is shared with the generic builder (default None there).
    for img in find_images(root, getattr(args, "image_glob", None) or DEFAULT_GLOB):
        subj, contrast = parse_name(img)
        labels: "OrderedDict[str, str]" = OrderedDict()
        if getattr(args, "label_glob", None):
            hits = sorted(rootp.glob(args.label_glob.format(id=subj, name=_stem(img))))
            hits = [h for h in hits if not _other_channel(h, img)]
            if len(hits) > 1:
                same = [h for h in hits if _stem(h) == _stem(img)]
                if len(same) != 1:
                    raise ValueError("label glob matched %d files for %s: %s" % (len(hits), subj, hits[:3]))
                hits = same
            if hits:
                labels[args.label_name] = str(hits[0].resolve())
        rows.append(ManifestRow(dataset="lumir", subject=subj, session="", contrast=contrast,
                                split=split_of(img), image_path=str(img), label_paths=labels,
                                skull_stripped=True, in_template_space=True, bias_corrected=True))
    n_lab = sum(1 for r in rows if r.label_paths)
    n_t1 = sum(1 for r in rows if r.contrast == "T1w")
    log.info("lumir: %d images (%d T1w, %d other channel; %d with labels)", len(rows), n_t1, len(rows) - n_t1,
             n_lab)
    return rows
