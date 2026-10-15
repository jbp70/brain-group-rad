"""Per-dataset manifest builders.

Each builder scans a dataset root and returns ManifestRow objects (absolute
paths, sorted). Use through the CLI:

    python -m preprocessing build-manifest --dataset lumir --root /data/LUMIR --out lumir.csv
    python -m preprocessing build-manifest --dataset neurite_oasis --root /data/neurite-oasis --out oasis.csv --exclude ids.txt
    python -m preprocessing build-manifest --dataset ultracortex --root /data/ds005216 --out uc.csv
    python -m preprocessing build-manifest --dataset generic --root /data/Mindboggle101 --name mindboggle \\
        --image-glob '*/*/t1weighted.nii.gz' --label 'manual={subject}/labels.DKT31.manual+aseg.nii.gz'
"""
import logging
from typing import List, Optional

from ..manifest import ManifestRow
from . import generic, lumir, neurite_oasis, ultracortex

log = logging.getLogger(__name__)

BUILDERS = {
    "lumir": lumir,
    "neurite_oasis": neurite_oasis,
    "ultracortex": ultracortex,
    "generic": generic,
}


def read_id_list(path: str) -> List[str]:
    """One subject ID per line; blank lines and '#' comments ignored."""
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                ids.append(line)
    return ids


def apply_exclude(rows: List[ManifestRow], exclude_path: Optional[str]) -> List[ManifestRow]:
    """Drop rows whose subject is listed (exact match) in exclude_path."""
    if not exclude_path:
        return rows
    ids = set(read_id_list(exclude_path))
    kept = [r for r in rows if r.subject not in ids]
    matched = {r.subject for r in rows} & ids
    log.info("exclude list: %d IDs, removed %d rows (%d IDs matched)", len(ids),
             len(rows) - len(kept), len(matched))
    unmatched = ids - matched
    if unmatched:
        log.warning("%d exclude IDs matched no subject, e.g. %s", len(unmatched), sorted(unmatched)[:5])
    return kept
