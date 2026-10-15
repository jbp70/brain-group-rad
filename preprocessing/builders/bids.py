"""Minimal BIDS filename helpers (entity parsing and derivative matching).

Example:
    parse_bids_name("sub-01_ses-2_acq-mp2rage_T1w.nii.gz")
    # -> ({'sub': '01', 'ses': '2', 'acq': 'mp2rage'}, 'T1w')
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

NII_EXTS = (".nii.gz", ".nii")


def strip_nii(name: str) -> Optional[str]:
    for ext in NII_EXTS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return None


def parse_bids_name(name: str) -> Tuple[Dict[str, str], str]:
    """Return (entities, suffix) for a BIDS-like NIfTI filename."""
    stem = strip_nii(os.path.basename(name))
    if stem is None:
        raise ValueError("not a NIfTI file: %s" % name)
    ents: Dict[str, str] = {}
    suffix = ""
    for part in stem.split("_"):
        if "-" in part:
            k, v = part.split("-", 1)
            ents[k] = v
        else:
            suffix = part
    return ents, suffix


def find_bids_images(root: str, suffix: str = "T1w") -> List[Path]:
    """Raw images sub-*/[ses-*/]anat/*_<suffix>.nii[.gz] (derivatives excluded)."""
    rootp = Path(root)
    found = []
    for pattern in ("sub-*/anat/*_%s.nii*" % suffix, "sub-*/ses-*/anat/*_%s.nii*" % suffix):
        for p in rootp.glob(pattern):
            if strip_nii(p.name) is not None:
                found.append(p.resolve())
    return sorted(set(found))


def contrast_from(ents: Dict[str, str], suffix: str) -> str:
    """Contrast column: suffix, prefixed by acq/run entities when present (keeps rows unique)."""
    parts = []
    for k in ("acq", "rec", "run"):
        if k in ents:
            parts.append("%s-%s" % (k, ents[k]))
    parts.append(suffix)
    return "_".join(parts)


def matches(target: Dict[str, str], cand: Dict[str, str]) -> bool:
    """Candidate derivative matches when sub/ses agree and any acq/run it names agree."""
    if cand.get("sub") != target.get("sub"):
        return False
    if cand.get("ses", "") != target.get("ses", ""):
        return False
    for k in ("acq", "run"):
        if k in cand and cand[k] != target.get(k):
            return False
    return True
