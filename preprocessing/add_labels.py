"""export-add-labels: add one label set to an EXISTING export, add-only (synth-contrast design 3.3).

The case list is <export_root>/export_manifest.csv itself (no export filters needed); the only
new files are labels_<set>{,_2,_4}.npy per case. v1 supports the set `synthfine` (the 103-id
SynthSeg --parc remap, nested majority vote at levels 2/4 so that
luts.synthfine_to_train18_table()[synthfine_k] == train18_k voxel for voxel).

Phase files (array-safe; writes only per-case npy files and one status file per chunk), for
each manifest data row r in the half-open range [--start, --end):
  1. source <out_root>/<dataset>/<subject_dir>/labels_synthseg.nii.gz (missing -> case FAILED);
     npy dir <export_root>/<dataset>/<subject_dir> (the trainer's rule; the npy_dir column is
     only compared, a difference is a WARNING);
  2. fine = LUT(source) (unknown raw id -> FAILED); its shape must equal shape_l1;
  3. per level k: f_k (nested vote) is verified against the existing labels_train18<sfx>.npy
     BEFORE any write (table[f_k] == train18_k; a reprocessed case fails at k = 1);
  4. labels_<set><sfx>.npy written atomically when missing or unreadable; an existing file must be
     bit-identical (rerun -> "kept"), else FAILED and left alone (also a wrong dtype/shape);
     image / train18 / other files are only read;
  5. status <export_root>/add_labels/<set>/rows-<start>-<end|end>.json (per case written / kept /
     failed + reason); exit 0 iff no case failed.
  Range rule: --start/--end index the data rows of export_manifest.csv, half-open; omitted start
  = 0, omitted end = all rows; end beyond the row count is clipped (last chunk shorter); start >=
  rows = empty selection (status file with 0 cases, exit 0); negative values or start > end exit
  2. The status file is named after the REQUESTED range.

Phase manifest (one process, after all file chunks):
  0. read-only pre-check: the UNMODIFIED manifest_parts/ must merge (export-merge) to the current
     export_manifest.csv byte for byte, else exit 1 "pre-existing divergence ..." with the
     differing (dataset, subject_dir) keys; nothing changed, no backup;
  1. re-check every row (files exist, uint8, level shape, table consistency), else exit 1;
  2. backup export_manifest.csv, label_luts.json, manifest_parts/ to
     <export_root>/backup_pre_<set>_<UTC timestamp>/ (copies);
  3. append the set to the six label columns (LABEL_COLUMNS) of every manifest row and of the
     part rows that are in the manifest; every other column and the row/column order unchanged;
  4. add the set's LUT to every changed <part>.luts.json and to label_luts.json;
  5. the changed parts must merge to the changed manifest byte for byte; then everything is
     written atomically;
  6. diff check old vs new manifest (only LABEL_COLUMNS differ, by appended entries) printed and
     written to add_labels/<set>/manifest_diff.json.
  Any failure after the backup restores every file from it (exit 1). A second run is a no-op.
"""
import csv
import glob
import io
import json
import logging
import os
import shutil
import tempfile
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import export as ex
from . import nifti
from .luts import LUTS, apply_lut, lut_json, synthfine_to_train18_table

log = logging.getLogger(__name__)

LABEL_COLUMNS = ("labels", "label_dtypes", "label_levels") + tuple("labels_l%d" % k for k in ex.ALL_LEVELS)
SUPPORTED_SETS = ("synthfine",)
STATUS_DIR = "add_labels"
LABEL_DTYPE = "uint8"


# ----------------------------------------------------------------- helpers

def resolve_range(n_rows: int, start: Optional[int], end: Optional[int]) -> Tuple[int, int]:
    """Half-open row range [s, e) clipped to n_rows; ValueError on negative values or start > end."""
    if (start is not None and start < 0) or (end is not None and end < 0):
        raise ValueError("--start/--end must be >= 0, got start=%s end=%s" % (start, end))
    s = 0 if start is None else int(start)
    if end is not None and s > end:
        raise ValueError("--start %d is greater than --end %d" % (s, end))
    e = n_rows if end is None else min(int(end), n_rows)
    s = min(s, n_rows)
    return s, max(s, e)


def _check_set(label_set: str) -> None:
    if label_set not in SUPPORTED_SETS:
        raise ValueError("label set %r is not supported (supported: %s)" % (label_set, ", ".join(SUPPORTED_SETS)))


def _read_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        return list(reader.fieldnames or []), rows


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _key(row: Dict[str, str]) -> Tuple[str, str]:
    return row["dataset"], row["subject_dir"]


def _case(row: Dict[str, str]) -> str:
    return "%s/%s" % _key(row)


def _row_levels(row: Dict[str, str], label_set: str) -> List[int]:
    allowed = ex.LABEL_SET_LEVELS.get(label_set, (1,))
    return [int(x) for x in (row.get("levels") or "").split(";") if x and int(x) in allowed]


def _level_shape(row: Dict[str, str], k: int) -> Tuple[int, ...]:
    return tuple(int(s) for s in row["shape_l%d" % k].split("x"))


def _set_file(label_set: str, k: int) -> str:
    return "labels_%s%s.npy" % (label_set, ex.level_suffix(k))


def _listed(row: Dict[str, str], k: int, name: str) -> bool:
    return name in [x for x in (row.get("labels_l%d" % k) or "").split(";") if x]


def _npy_dir(export_root: str, row: Dict[str, str]) -> str:
    return os.path.join(export_root, row["dataset"], row["subject_dir"])


def _train18(row: Dict[str, str], d: str, k: int) -> np.ndarray:
    """The existing labels_train18<sfx>.npy of level k (must be listed, uint8, level shape)."""
    name = _set_file("train18", k)
    if not _listed(row, k, name):
        raise ValueError("%s is not listed in labels_l%d" % (name, k))
    path = os.path.join(d, name)
    st, desc = ex._npy_state(path, LABEL_DTYPE, _level_shape(row, k))
    if st != "ok":
        raise ValueError("%s is %s %s" % (name, st, desc))
    return np.load(path)


def _differ(table: np.ndarray, f_k: np.ndarray, t18: np.ndarray) -> int:
    return int(np.count_nonzero(table[f_k] != t18))


# ----------------------------------------------------------------- phase files

def _process_case(row: Dict[str, str], out_root: str, export_root: str, label_set: str,
                  table: np.ndarray) -> Dict[str, Any]:
    """One case of phase files; never raises (failure -> status 'failed' with the reason)."""
    levels = _row_levels(row, label_set)
    res: Dict[str, Any] = {"case": _case(row), "status": "failed", "reason": "", "levels": levels}
    try:
        d = _npy_dir(export_root, row)
        if row.get("npy_dir") and os.path.normpath(row["npy_dir"]) != os.path.normpath(d):
            log.warning("export-add-labels: %s npy_dir column %s differs from %s (tree moved?); using %s",
                        res["case"], row["npy_dir"], d, d)
        src = os.path.join(out_root, row["dataset"], row["subject_dir"], "labels_synthseg.nii.gz")
        if not os.path.isfile(src):
            res["reason"] = "no source %s" % src
            return res
        fine = apply_lut(nifti.load_data(src, np.float64)[0], LUTS[label_set], LABEL_DTYPE,
                         what="%s (case %s, label set %s)" % (src, res["case"], label_set))
        if tuple(fine.shape) != _level_shape(row, 1):
            res["reason"] = "source shape %s differs from shape_l1 %s" % (
                "x".join(map(str, fine.shape)), row["shape_l1"])
            return res
        planned = []
        for k in levels:  # verify every level BEFORE any write
            f_k = ex.synthfine_level(fine, k)
            n = _differ(table, f_k, _train18(row, d, k))
            if n:
                res["reason"] = "level %d: %d voxels differ from %s (source differs from the one used " \
                                "for train18?)" % (k, n, _set_file("train18", k))
                return res
            planned.append((k, os.path.join(d, _set_file(label_set, k)), f_k))
        to_write = []
        for k, path, f_k in planned:
            st, desc = ex._npy_state(path, LABEL_DTYPE, f_k.shape)
            if st == "mismatch":
                res["reason"] = "existing %s is %s, expected %s %s" % (
                    os.path.basename(path), desc, LABEL_DTYPE, "x".join(map(str, f_k.shape)))
                return res
            if st == "ok":
                if not np.array_equal(np.load(path), f_k):
                    res["reason"] = "existing %s differs from the recomputed labels (left alone)" % (
                        os.path.basename(path))
                    return res
            else:  # missing / unreadable
                to_write.append((path, f_k))
        for path, f_k in to_write:
            ex._atomic_save_npy(path, f_k)
        res["status"] = "written" if to_write else "kept"
        res["reason"] = "" if to_write else "all files present and identical"
        return res
    except Exception as e:  # one case never stops the chunk
        res["status"], res["reason"] = "failed", "%s: %s" % (type(e).__name__, e)
        return res


def status_path(export_root: str, label_set: str, start: Optional[int], end: Optional[int]) -> str:
    """Status file named after the REQUESTED range: rows-<start>-<end|end>.json."""
    name = "rows-%d-%s.json" % (0 if start is None else start, "end" if end is None else end)
    return os.path.join(export_root, STATUS_DIR, label_set, name)


def add_labels_files(out_root: str, export_root: str, label_set: str = "synthfine", start: Optional[int] = None,
                     end: Optional[int] = None) -> Tuple[Dict[str, Any], int]:
    """Phase files over manifest rows [start, end); returns (status dict, exit code 0/1).

    ValueError for an unsupported set or an invalid range (the CLI exits 2).
    """
    _check_set(label_set)
    _, rows = _read_csv(os.path.join(export_root, ex.MANIFEST_FILE))
    s, e = resolve_range(len(rows), start, end)
    table = synthfine_to_train18_table()
    cases = []
    for r in rows[s:e]:
        res = _process_case(r, out_root, export_root, label_set, table)
        if res["status"] == "failed":
            log.error("export-add-labels: %s FAILED: %s", res["case"], res["reason"])
        cases.append(res)
    counts = OrderedDict((c, sum(1 for x in cases if x["status"] == c)) for c in ("written", "kept", "failed"))
    status: Dict[str, Any] = OrderedDict([
        ("label_set", label_set), ("requested", [start, end]), ("rows", [s, e]), ("n_manifest_rows", len(rows)),
        ("counts", counts), ("cases", cases)])
    sp = status_path(export_root, label_set, start, end)
    ex._atomic_write_text(sp, json.dumps(status, indent=2))
    print("export-add-labels files: rows [%d, %d) of %d: %d written, %d kept, %d failed -> %s"
          % (s, e, len(rows), counts["written"], counts["kept"], counts["failed"], sp))
    for x in cases:
        if x["status"] == "failed":
            print("FAILED %s: %s" % (x["case"], x["reason"]))
    return status, (1 if counts["failed"] else 0)


# ----------------------------------------------------------------- phase manifest

def _append(old: str, entry: str) -> str:
    return entry if not old else old + ";" + entry


def add_set(row: Dict[str, str], label_set: str) -> Dict[str, str]:
    """Row with label_set appended to the six label columns; unchanged if `labels` already lists it."""
    new = dict(row)
    if label_set in [x for x in (row.get("labels") or "").split(";") if x]:
        return new
    levels = _row_levels(row, label_set)
    new["labels"] = _append(row.get("labels") or "", label_set)
    new["label_dtypes"] = _append(row.get("label_dtypes") or "", "%s:%s" % (label_set, LABEL_DTYPE))
    new["label_levels"] = _append(row.get("label_levels") or "", "%s:%s" % (label_set, ",".join(map(str, levels))))
    for k in levels:
        col = "labels_l%d" % k
        new[col] = _append(row.get(col) or "", _set_file(label_set, k))
    return new


def _tmp_root(export_root: str, label_set: str, name: str) -> str:
    try:
        return tempfile.mkdtemp(prefix="bgr_%s_" % name)
    except OSError:
        d = os.path.join(export_root, STATUS_DIR, label_set, name)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        return d


def _merged_manifest(parts: Dict[str, str], export_root: str, label_set: str, name: str) -> bytes:
    """merge_parts over the given part files {file name: text} in a temp root; the merged manifest bytes."""
    tmp = _tmp_root(export_root, label_set, name)
    try:
        pdir = os.path.join(tmp, ex.PARTS_DIR)
        os.makedirs(pdir, exist_ok=True)
        for fn, text in parts.items():
            with open(os.path.join(pdir, fn), "w", newline="", encoding="utf-8") as f:
                f.write(text)
        ex.merge_parts(tmp)
        return _read_bytes(os.path.join(tmp, ex.MANIFEST_FILE))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _read_parts(export_root: str) -> "OrderedDict[str, str]":
    out: "OrderedDict[str, str]" = OrderedDict()
    pdir = os.path.join(export_root, ex.PARTS_DIR)
    for p in sorted(glob.glob(os.path.join(pdir, "*.csv")) + glob.glob(os.path.join(pdir, "*.luts.json"))):
        with open(p, "r", newline="", encoding="utf-8") as f:
            out[os.path.basename(p)] = f.read()
    return out


def _csv_rows(text: str) -> Tuple[List[str], List[Dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text, newline=""))
    rows = [dict(r) for r in reader]
    return list(reader.fieldnames or []), rows


def _diff_keys(a: bytes, b: bytes) -> List[str]:
    """(dataset, subject_dir) keys whose rows differ between two manifests (or exist in one only)."""
    _, ra = _csv_rows(a.decode("utf-8"))
    _, rb = _csv_rows(b.decode("utf-8"))
    da = {_key(r): r for r in ra}
    db = {_key(r): r for r in rb}
    keys = sorted(k for k in set(da) | set(db) if da.get(k) != db.get(k))
    if not keys and [_key(r) for r in ra] != [_key(r) for r in rb]:
        return ["(row order differs)"]
    if not keys:
        return ["(header or formatting differs)"]
    return ["%s/%s%s" % (k[0], k[1], "" if k in da and k in db else (" (only in %s)" % (
        "the merged parts" if k in da else "export_manifest.csv"))) for k in keys]


def _recheck(rows: List[Dict[str, str]], export_root: str, label_set: str, table: np.ndarray) -> List[str]:
    """Step 1: every row has the set's files (uint8, level shape) consistent with train18."""
    problems = []
    for r in rows:
        d = _npy_dir(export_root, r)
        try:
            for k in _row_levels(r, label_set):
                path = os.path.join(d, _set_file(label_set, k))
                st, desc = ex._npy_state(path, LABEL_DTYPE, _level_shape(r, k))
                if st != "ok":
                    problems.append("%s: %s %s %s" % (_case(r), os.path.basename(path), st, desc))
                    continue
                n = _differ(table, np.load(path), _train18(r, d, k))
                if n:
                    problems.append("%s: %s inconsistent with %s (%d voxels differ)"
                                    % (_case(r), os.path.basename(path), _set_file("train18", k), n))
        except Exception as e:
            problems.append("%s: %s: %s" % (_case(r), type(e).__name__, e))
    return problems


def _luts_text(text: str, label_set: str) -> str:
    j = json.loads(text, object_pairs_hook=OrderedDict)
    luts = OrderedDict(j.get("luts", {}))
    luts[label_set] = json.loads(json.dumps(lut_json(LUTS[label_set])), object_pairs_hook=OrderedDict)
    return json.dumps({"remap_labels": bool(j.get("remap_labels", True)), "luts": OrderedDict(sorted(luts.items()))},
                      indent=2)


def _diff_report(old_rows: List[Dict[str, str]], new_rows: List[Dict[str, str]], label_set: str) -> Dict[str, Any]:
    """Step 6: old vs new manifest; problems unless only LABEL_COLUMNS differ by appended entries."""
    problems: List[str] = []
    changed = 0
    cols_changed = set()
    if len(old_rows) != len(new_rows):
        problems.append("row count %d -> %d" % (len(old_rows), len(new_rows)))
    elif [_key(r) for r in old_rows] != [_key(r) for r in new_rows]:
        problems.append("(dataset, subject_dir) order changed")
    else:
        for o, n in zip(old_rows, new_rows):
            if list(o) != list(n):
                problems.append("%s: columns changed" % _case(o))
                continue
            row_changed = False
            for c in o:
                if o[c] == n[c]:
                    continue
                if c not in LABEL_COLUMNS:
                    problems.append("%s: non-label column %s changed (%r -> %r)" % (_case(o), c, o[c], n[c]))
                    continue
                rest = n[c][len(o[c]):]
                if not n[c].startswith(o[c]) or not rest or (o[c] and not rest.startswith(";")) \
                        or label_set not in rest:
                    problems.append("%s: %s is not the old value plus appended entries (%r -> %r)"
                                    % (_case(o), c, o[c], n[c]))
                    continue
                cols_changed.add(c)
                row_changed = True
            changed += int(row_changed)
    return OrderedDict([("label_set", label_set), ("rows", len(new_rows)), ("rows_changed", changed),
                        ("columns_changed", sorted(cols_changed)), ("allowed_columns", list(LABEL_COLUMNS)),
                        ("problems", problems)])


def _backup(export_root: str, label_set: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    bdir = os.path.join(export_root, "backup_pre_%s_%s" % (label_set, stamp))
    i = 0
    while os.path.exists(bdir):
        i += 1
        bdir = os.path.join(export_root, "backup_pre_%s_%s_%d" % (label_set, stamp, i))
    os.makedirs(bdir)
    shutil.copy2(os.path.join(export_root, ex.MANIFEST_FILE), bdir)
    lp = os.path.join(export_root, ex.LUT_FILE)
    if os.path.isfile(lp):
        shutil.copy2(lp, bdir)
    shutil.copytree(os.path.join(export_root, ex.PARTS_DIR), os.path.join(bdir, ex.PARTS_DIR))
    return bdir


def _restore(export_root: str, bdir: str) -> None:
    for fn in (ex.MANIFEST_FILE, ex.LUT_FILE):
        b = os.path.join(bdir, fn)
        if os.path.isfile(b):
            ex._atomic_write_text(os.path.join(export_root, fn), _read_bytes(b).decode("utf-8"))
    bp = os.path.join(bdir, ex.PARTS_DIR)
    for b in sorted(glob.glob(os.path.join(bp, "*"))):
        ex._atomic_write_text(os.path.join(export_root, ex.PARTS_DIR, os.path.basename(b)),
                              _read_bytes(b).decode("utf-8"))


def add_labels_manifest(export_root: str, label_set: str = "synthfine") -> int:
    """Phase manifest (steps 0-6 of the module docstring); returns the exit code 0/1."""
    _check_set(label_set)
    mpath = os.path.join(export_root, ex.MANIFEST_FILE)
    lpath = os.path.join(export_root, ex.LUT_FILE)
    old_bytes = _read_bytes(mpath)
    parts = _read_parts(export_root)
    # 0. read-only pre-check: the unmodified parts reproduce the current manifest.
    try:
        merged = _merged_manifest(parts, export_root, label_set, "tmp_merge_pre")
        diverged = [] if merged == old_bytes else _diff_keys(merged, old_bytes)
    except ValueError as e:
        diverged = ["export-merge of the parts failed: %s" % e]
    if diverged:
        msg = ("pre-existing divergence between manifest_parts and export_manifest.csv (not caused by "
               "export-add-labels); nothing changed. Differing (dataset, subject_dir): %s" % "; ".join(diverged))
        log.error("export-add-labels: %s", msg)
        print(msg)
        for k in diverged:
            print("DIVERGED %s" % k)
        return 1
    cols, rows = _csv_rows(old_bytes.decode("utf-8"))
    # 1. re-check every row.
    problems = _recheck(rows, export_root, label_set, synthfine_to_train18_table())
    if problems:
        print("export-add-labels manifest: %d problems, nothing changed (run phase files first):" % len(problems))
        for p in problems:
            print("PROBLEM " + p)
        return 1
    with open(lpath, "r", encoding="utf-8") as f:
        luts_old = f.read()
    has_lut = label_set in json.loads(luts_old).get("luts", {})
    if has_lut and all(add_set(r, label_set) == r for r in rows):
        print("export-add-labels manifest: every row already lists %s; nothing to do" % label_set)
        return 0
    # 2. backup.
    bdir = _backup(export_root, label_set)
    print("export-add-labels manifest: backup in %s" % bdir)
    try:
        # 3. transform rows of the manifest and of the part rows that are in the manifest.
        keys = {_key(r) for r in rows}
        new_rows = [add_set(r, label_set) for r in rows]
        new_manifest = ex._csv_text(new_rows, cols)
        new_parts: "OrderedDict[str, str]" = OrderedDict(parts)
        changed_parts = []
        for fn, text in parts.items():
            if not fn.endswith(".csv"):
                continue
            pcols, prow = _csv_rows(text)
            tr = [add_set(r, label_set) if _key(r) in keys else r for r in prow]
            if tr != prow:
                new_parts[fn] = ex._csv_text(tr, pcols)
                changed_parts.append(fn)
        # 4. LUT into every changed part's luts.json and label_luts.json.
        for fn in changed_parts:
            jn = fn[:-len(".csv")] + ".luts.json"
            if jn not in parts:
                raise ValueError("%s is missing" % jn)
            new_parts[jn] = _luts_text(parts[jn], label_set)
        new_luts = _luts_text(luts_old, label_set)
        # 5. the transformed parts reproduce the transformed manifest, then write atomically.
        merged = _merged_manifest(new_parts, export_root, label_set, "tmp_merge")
        if merged != new_manifest.encode("utf-8"):
            raise ValueError("merging the transformed parts does not reproduce the transformed manifest (%s)"
                             % "; ".join(_diff_keys(merged, new_manifest.encode("utf-8"))))
        pdir = os.path.join(export_root, ex.PARTS_DIR)
        for fn in changed_parts:
            jn = fn[:-len(".csv")] + ".luts.json"
            ex._atomic_write_text(os.path.join(pdir, jn), new_parts[jn])
            ex._atomic_write_text(os.path.join(pdir, fn), new_parts[fn])
        ex._atomic_write_text(lpath, new_luts)
        ex._atomic_write_text(mpath, new_manifest)
        # 6. diff check of the written manifest.
        _, written = _csv_rows(_read_bytes(mpath).decode("utf-8"))
        report = _diff_report(rows, written, label_set)
        report["changed_parts"] = changed_parts
        report["backup"] = bdir
        dp = os.path.join(export_root, STATUS_DIR, label_set, "manifest_diff.json")
        ex._atomic_write_text(dp, json.dumps(report, indent=2))
        print("export-add-labels manifest: %d rows, %d changed, columns changed %s, %d parts changed, "
              "%d problems -> %s" % (report["rows"], report["rows_changed"], ",".join(report["columns_changed"]),
                                     len(changed_parts), len(report["problems"]), dp))
        if report["problems"]:
            for p in report["problems"][:50]:
                print("PROBLEM " + p)
            raise ValueError("diff check failed (%d problems)" % len(report["problems"]))
    except Exception as e:
        log.error("export-add-labels: %s; restoring from %s", e, bdir)
        _restore(export_root, bdir)
        print("export-add-labels manifest FAILED (%s); every file restored from %s" % (e, bdir))
        return 1
    return 0


def run_phase(phase: str, export_root: str, label_set: str, out_root: Optional[str] = None,
              start: Optional[int] = None, end: Optional[int] = None) -> int:
    """CLI helper: one phase; ValueError for usage errors (exit 2)."""
    _check_set(label_set)
    if phase == "files":
        if not out_root:
            raise ValueError("--phase files needs --out-root")
        return add_labels_files(out_root, export_root, label_set, start, end)[1]
    if phase == "manifest":
        if start is not None or end is not None:
            raise ValueError("--start/--end apply to --phase files only")
        return add_labels_manifest(export_root, label_set)
    raise ValueError("unknown phase %r" % phase)
