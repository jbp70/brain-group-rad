"""Patient-level groups (e.g. MIRIAD): one groupwise registration per patient, results averaged.

write-groups  one subjects file per group from the processed subject folders. The group of a folder
              is the first match of --group-regex (default: MIRIAD patient
              ``miriad_<id>_<AD|HC>_<sex>`` from ``miriad_<id>_<AD|HC>_<sex>_<visit>_MR_<rep>_T1w``).
              Writes <out-dir>/<group>.txt (folder names, sorted) and <out-dir>/groups.csv
              (group, n_scans). Groups with fewer than --min-size scans are listed but get no file.
aggregate     reads <root>/<group>/summary.json (python -m evaluation run, one per group) and
              averages per group first (the subject / pair means of that group), then across
              groups, so every patient counts once. Writes per_group.csv and summary.json in --out
              (also split by diagnosis when the group name holds _AD_ / _HC_).

Metrics kept (folding / Jacobian / inverse-consistency metrics are not meaningful for linear
maps and are left out unless --all-metrics): template_lncc, template_ncc, template_dice,
template_dice_cortical, template_dice_subcortical, pair_lncc, pair_ncc, pair_dice,
pair_dice_cortical, pair_dice_subcortical, centrality_mean_mm, centrality_max_mm,
atlas_intensity_std.

    python -m evaluation.groups write-groups --processed-dir $PROJ/data/processed/miriad --out-dir G
    python -m evaluation.groups aggregate --root R/<method> --method <method> --out R/<method>/_aggregate
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

MIRIAD_REGEX = r"^(miriad_\d+_(?:AD|HC)_[MF])_"
SUBJECT_METRICS = ("template_lncc", "template_ncc", "template_dice", "template_dice_cortical",
                   "template_dice_subcortical")
PAIR_METRICS = ("pair_lncc", "pair_ncc", "pair_dice", "pair_dice_cortical", "pair_dice_subcortical")
GROUP_METRICS = ("centrality_mean_mm", "centrality_max_mm", "atlas_intensity_std")


def find_groups(processed_dir: str, regex: str = MIRIAD_REGEX) -> Dict[str, List[str]]:
    """{group: sorted subject folder names} over folders holding image.nii.gz."""
    pat = re.compile(regex)
    groups: Dict[str, List[str]] = {}
    for d in sorted(glob.glob(os.path.join(processed_dir, "*"))):
        if not os.path.isfile(os.path.join(d, "image.nii.gz")):
            continue
        m = pat.search(os.path.basename(d))
        if not m:
            continue
        groups.setdefault(m.group(1), []).append(os.path.basename(d))
    return groups


def cmd_write_groups(a) -> int:
    groups = find_groups(a.processed_dir, a.group_regex)
    if not groups:
        print("no subject folders matching %r in %s" % (a.group_regex, a.processed_dir), file=sys.stderr)
        return 2
    os.makedirs(a.out_dir, exist_ok=True)
    n_written = 0
    with open(os.path.join(a.out_dir, "groups.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n_scans", "written"])
        for g, dirs in sorted(groups.items()):
            ok = len(dirs) >= a.min_size
            w.writerow([g, len(dirs), int(ok)])
            if ok:
                with open(os.path.join(a.out_dir, g + ".txt"), "w") as gf:
                    gf.write("\n".join(dirs) + "\n")
                n_written += 1
    sizes = [len(v) for v in groups.values()]
    print("%d groups (%d written, min size %d), %d scans; scans per group min %d / median %.1f / max %d"
          % (len(groups), n_written, a.min_size, sum(sizes), min(sizes), float(np.median(sizes)), max(sizes)))
    return 0


def _mean(d: dict, key: str) -> Optional[float]:
    v = d.get(key) or {}
    return float(v["mean"]) if v.get("n") else None


def group_row(summary: dict, metrics_all: bool = False) -> dict:
    row = {"n_subjects": summary.get("n_subjects"), "n_pairs": summary.get("n_pairs")}
    subj, pair = summary.get("subject_metrics", {}), summary.get("pair_metrics", {})
    s_keys = list(subj) if metrics_all else [k for k in SUBJECT_METRICS if k in subj]
    p_keys = list(pair) if metrics_all else [k for k in PAIR_METRICS if k in pair]
    for k in s_keys:
        row[k] = _mean(subj, k)
    for k in p_keys:
        row[k] = _mean(pair, k)
    for k, v in summary.get("group", {}).items():
        if metrics_all or k in GROUP_METRICS:
            row[k] = v
    return row


def _stats(vals: Sequence[Optional[float]]) -> dict:
    v = np.asarray([x for x in vals if x is not None], np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return {"n": 0}
    return {"mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v)),
            "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def _diagnosis(group: str) -> Optional[str]:
    for dx in ("AD", "HC"):
        if "_%s_" % dx in group + "_":
            return dx
    return None


def cmd_aggregate(a) -> int:
    # Folders starting with "_" (e.g. the _aggregate output itself) are not groups.
    paths = sorted(p for p in glob.glob(os.path.join(a.root, "*", a.summary_name))
                   if not os.path.basename(os.path.dirname(p)).startswith("_"))
    if not paths:
        print("no %s under %s/*/" % (a.summary_name, a.root), file=sys.stderr)
        return 2
    rows = []
    for p in paths:
        with open(p) as f:
            s = json.load(f)
        r = {"group": os.path.basename(os.path.dirname(p)), "diagnosis": _diagnosis(os.path.basename(os.path.dirname(p)))}
        r.update(group_row(s, a.all_metrics))
        rows.append(r)
    if a.expected is not None and len(rows) != a.expected:
        print("WARNING: %d groups found, %d expected" % (len(rows), a.expected), file=sys.stderr)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    metric_keys = [k for k in keys if k not in ("group", "diagnosis", "n_subjects", "n_pairs")]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "per_group.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    out = {"method": a.method, "root": os.path.abspath(a.root), "n_groups": len(rows),
           "n_subjects_total": int(sum(r.get("n_subjects") or 0 for r in rows)),
           "averaging": "mean over subjects (pairs) within each group, then over groups",
           "metrics": {k: _stats([r.get(k) for r in rows]) for k in metric_keys}, "by_diagnosis": {}}
    for dx in ("AD", "HC"):
        sub = [r for r in rows if r["diagnosis"] == dx]
        if sub:
            out["by_diagnosis"][dx] = {"n_groups": len(sub),
                                       "metrics": {k: _stats([r.get(k) for r in sub]) for k in metric_keys}}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("method %s: %d groups, %d scans" % (a.method, out["n_groups"], out["n_subjects_total"]))
    for k in metric_keys:
        v = out["metrics"][k]
        if v.get("n"):
            print("  %-28s %.4f +- %.4f  (median %.4f, n %d)" % (k, v["mean"], v["std"], v["median"], v["n"]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m evaluation.groups", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write-groups", help="one subjects file per group")
    w.add_argument("--processed-dir", required=True)
    w.add_argument("--out-dir", required=True)
    w.add_argument("--group-regex", default=MIRIAD_REGEX, help="group = first capture (default MIRIAD patient)")
    w.add_argument("--min-size", type=int, default=2, help="smallest group that gets a file (default 2)")
    w.set_defaults(func=cmd_write_groups)
    g = sub.add_parser("aggregate", help="average per-group summary.json files")
    g.add_argument("--root", required=True,
                   help="folder with one <group>/summary.json per group (folders starting with _ are skipped)")
    g.add_argument("--method", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--summary-name", default="summary.json")
    g.add_argument("--expected", type=int, default=None, help="warn if the number of groups differs")
    g.add_argument("--all-metrics", action="store_true", help="keep every metric of summary.json")
    g.set_defaults(func=cmd_aggregate)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.func(a))


if __name__ == "__main__":
    sys.exit(main())
