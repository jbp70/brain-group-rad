"""Mindboggle-101 manifest builder.

Layout: ``<root>/<group>_volumes/<subject>/`` (e.g. OASIS-TRT-20_volumes/OASIS-TRT-20-1),
subject = folder name. Every subject has raw (native) and MNI152 versions of the
T1 image and of the DKT31 manual + aseg labels.

- MNI152 rows (default): ``t1weighted.MNI152.nii.gz`` + ``labels.DKT31.manual+aseg.MNI152.nii.gz``,
  in_template_space (already on FSL's MNI152 1 mm grid): no registration; SynthStrip + N4,
  then world-space resampling onto the LUMIR grid.
- Native rows (subjects matching --mb-native, default ``HLN-*``, whose provided MNI152
  images are misregistered): ``t1weighted.nii.gz`` + ``labels.DKT31.manual+aseg.nii.gz``,
  full pipeline: reorient, SynthStrip, N4, 9-DOF FLIRT to MNI152, one resampling onto
  the LUMIR grid (labels nearest-neighbour with the same transform).

Labels are named ``dkt31aseg`` in both cases, so outputs look the same. Config:
preprocessing/config_mindboggle.yaml (SynthStrip mask only, labels zeroed outside it).

    # current recipe: every subject from the native files with our own registration
    python -m preprocessing build-manifest --dataset mindboggle --root /data/data_mindboggle --out mb.csv \\
        --mb-native '*'
    # mixed: provided MNI152 files except HLN-*
    python -m preprocessing build-manifest --dataset mindboggle --root /data/data_mindboggle --out mb.csv
"""
import argparse
import fnmatch
import logging
from collections import OrderedDict
from pathlib import Path
from typing import List

from ..manifest import ManifestRow

log = logging.getLogger(__name__)

LABEL_NAME = "dkt31aseg"
FILES = {
    "mni": ("t1weighted.MNI152.nii.gz", "labels.DKT31.manual+aseg.MNI152.nii.gz"),
    "native": ("t1weighted.nii.gz", "labels.DKT31.manual+aseg.nii.gz"),
}


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mb-native", action="append", default=None, metavar="GLOB",
                   help="mindboggle: subject-name glob processed from the native files with our own "
                        "registration; repeatable (default HLN-*)")
    p.add_argument("--mb-select", choices=("all", "native", "mni"), default="all",
                   help="mindboggle: which rows to write (default %(default)s)")


def build(root: str, args: argparse.Namespace) -> List[ManifestRow]:
    native_globs = getattr(args, "mb_native", None) or ["HLN-*"]
    select = getattr(args, "mb_select", "all")
    rows = []
    for d in sorted(p for p in Path(root).glob("*/*") if p.is_dir()):
        subject = d.name
        kind = "native" if any(fnmatch.fnmatchcase(subject, g) for g in native_globs) else "mni"
        if select != "all" and kind != select:
            continue
        img, lab = (d / n for n in FILES[kind])
        if not img.is_file() or not lab.is_file():
            log.warning("mindboggle: %s missing %s or %s, skipping", d, img.name, lab.name)
            continue
        rows.append(ManifestRow(dataset="mindboggle", subject=subject, contrast="T1w", image_path=str(img.resolve()),
                                label_paths=OrderedDict([(LABEL_NAME, str(lab.resolve()))]),
                                in_template_space=kind == "mni"))
    subjects = [r.subject for r in rows]
    dup = sorted({s for s in subjects if subjects.count(s) > 1})
    if dup:
        raise ValueError("mindboggle: duplicate subject names across groups: %s" % dup[:5])
    n_native = sum(1 for r in rows if not r.in_template_space)
    log.info("mindboggle: %d subjects (%d native + own registration [%s], %d MNI152)", len(rows), n_native,
             ",".join(native_globs), len(rows) - n_native)
    return rows
