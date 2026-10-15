"""Input manifest (CSV) read/write.

Columns (in order):
    dataset, subject, session, contrast, split, image_path, mask_path,
    label_paths, skull_stripped, in_template_space, bias_corrected

``label_paths`` is ``name:path;name:path`` (name = [A-Za-z0-9_-]+; the path may
contain ':' but not ';'). Booleans are written as true/false.

Example:
    rows = read_manifest("lumir.csv")
    print(subject_dir_name(rows[0]))   # e.g. LUMIRMRI_0001_T1w
"""
import csv
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

COLUMNS = ["dataset", "subject", "session", "contrast", "split", "image_path", "mask_path",
           "label_paths", "skull_stripped", "in_template_space", "bias_corrected"]
BOOL_COLUMNS = ("skull_stripped", "in_template_space", "bias_corrected")
_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._+-]*$")


@dataclass
class ManifestRow:
    dataset: str
    subject: str
    image_path: str
    session: str = ""
    contrast: str = ""
    split: str = ""
    mask_path: str = ""
    label_paths: Dict[str, str] = field(default_factory=OrderedDict)
    skull_stripped: bool = False
    in_template_space: bool = False
    bias_corrected: bool = False

    def key(self) -> Tuple[str, str, str, str]:
        return (self.dataset, self.subject, self.session, self.contrast)

    def to_dict(self) -> Dict[str, object]:
        d = {c: getattr(self, c) for c in COLUMNS}
        d["label_paths"] = dict(self.label_paths)
        return d


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "y", "t"):
        return True
    if v in ("", "0", "false", "no", "n", "f"):
        return False
    raise ValueError("cannot parse boolean from %r" % (value,))


def parse_label_paths(text: str) -> "OrderedDict[str, str]":
    out: "OrderedDict[str, str]" = OrderedDict()
    for part in (text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("label entry %r must be name:path" % part)
        name, path = part.split(":", 1)
        name = name.strip()
        if not _NAME.match(name):
            raise ValueError("invalid label name %r" % name)
        if name in out:
            raise ValueError("duplicate label name %r" % name)
        out[name] = path.strip()
    return out


def format_label_paths(labels: Dict[str, str]) -> str:
    parts = []
    for name, path in labels.items():
        if not _NAME.match(name):
            raise ValueError("invalid label name %r" % name)
        if ";" in path:
            raise ValueError("label path may not contain ';': %r" % path)
        parts.append("%s:%s" % (name, path))
    return ";".join(parts)


def validate_row(row: ManifestRow) -> None:
    for attr in ("dataset", "subject", "session", "contrast"):
        val = getattr(row, attr)
        if not _SAFE_ID.match(val):
            raise ValueError("%s %r has characters outside [A-Za-z0-9._+-]" % (attr, val))
    if not row.dataset or not row.subject or not row.image_path:
        raise ValueError("dataset, subject and image_path are required (%s)" % (row.key(),))


def check_unique(rows: List[ManifestRow]) -> None:
    seen = set()
    for r in rows:
        if r.key() in seen:
            raise ValueError("duplicate manifest key (dataset, subject, session, contrast) = %s" % (r.key(),))
        seen.add(r.key())


def subject_dir_name(row: ManifestRow) -> str:
    """<subject>[_ses-<session>][_<contrast>]"""
    name = row.subject
    if row.session:
        name += "_ses-" + row.session
    if row.contrast:
        name += "_" + row.contrast
    return name


def write_manifest(rows: List[ManifestRow], path: str) -> None:
    for r in rows:
        validate_row(r)
    check_unique(rows)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            d = {c: getattr(r, c) for c in COLUMNS}
            d["label_paths"] = format_label_paths(r.label_paths)
            for b in BOOL_COLUMNS:
                d[b] = "true" if d[b] else "false"
            w.writerow(d)


def read_manifest(path: str) -> List[ManifestRow]:
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ("dataset", "subject", "image_path") if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("%s: missing columns %s" % (path, missing))
        for i, d in enumerate(reader):
            try:
                row = ManifestRow(
                    dataset=(d.get("dataset") or "").strip(),
                    subject=(d.get("subject") or "").strip(),
                    image_path=(d.get("image_path") or "").strip(),
                    session=(d.get("session") or "").strip(),
                    contrast=(d.get("contrast") or "").strip(),
                    split=(d.get("split") or "").strip(),
                    mask_path=(d.get("mask_path") or "").strip(),
                    label_paths=parse_label_paths(d.get("label_paths") or ""),
                    skull_stripped=parse_bool(d.get("skull_stripped", "")),
                    in_template_space=parse_bool(d.get("in_template_space", "")),
                    bias_corrected=parse_bool(d.get("bias_corrected", "")),
                )
                validate_row(row)
            except ValueError as e:
                raise ValueError("%s row %d: %s" % (path, i, e))
            rows.append(row)
    check_unique(rows)
    return rows
