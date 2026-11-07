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

compare       side-by-side table of several methods from their <root>/<method>/_aggregate/per_group.csv,
              over the patients every method has: mean +- std per method, and per method the paired
              difference to --reference (default identity), two-sided Wilcoxon signed-rank p over
              patients and the share of patients where it beats the reference. Overall and AD / HC.
              Writes compare.md and compare.csv in --out.

    python -m evaluation.groups compare --root $PROJ/results/eval_miriad \\
        --methods identity brainmorph_rigid brainmorph_affine --out $PROJ/results/eval_miriad/_compare
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


HIGHER_IS_BETTER = ("lncc", "ncc", "dice")


def _better(metric: str) -> str:
    return "higher" if any(k in metric for k in HIGHER_IS_BETTER) else "lower"


def _read_per_group(path: str) -> Dict[str, dict]:
    with open(path, newline="") as f:
        return {r["group"]: r for r in csv.DictReader(f)}


def _num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _wilcoxon(d: np.ndarray) -> Optional[float]:
    """Two-sided Wilcoxon signed-rank p over paired per-group differences (None if undefined)."""
    if d.size < 2 or not np.any(d != 0):
        return None
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue)
    except ValueError:
        return None


def compare_rows(tables: Dict[str, Dict[str, dict]], methods: Sequence[str], reference: Optional[str],
                 metrics: Sequence[str], groups: Sequence[str], subset: str = "all") -> List[dict]:
    """One row per (metric, method): mean / std / median over ``groups`` (the patients every method
    has), plus the paired difference to ``reference`` (mean, Wilcoxon p, fraction of groups where
    the method is better than the reference)."""
    rows = []
    for m in metrics:
        ref = None
        if reference:
            ref = np.array([_num(tables[reference][g].get(m)) for g in groups], dtype=object)
        for meth in methods:
            vals = np.array([_num(tables[meth][g].get(m)) for g in groups], dtype=object)
            ok = np.array([v is not None for v in vals])
            v = vals[ok].astype(np.float64)
            row = {"subset": subset, "metric": m, "better": _better(m), "method": meth, "n": int(v.size),
                   "mean": float(v.mean()) if v.size else None, "std": float(v.std()) if v.size else None,
                   "median": float(np.median(v)) if v.size else None}
            if ref is not None and meth != reference:
                both = ok & np.array([r is not None for r in ref])
                d = vals[both].astype(np.float64) - ref[both].astype(np.float64)
                sign = 1.0 if row["better"] == "higher" else -1.0
                row.update({"diff_vs_ref": float(d.mean()) if d.size else None, "n_paired": int(d.size),
                            "p_wilcoxon": _wilcoxon(d),
                            "frac_better": float(np.mean(sign * d > 0)) if d.size else None})
            rows.append(row)
    return rows


def _fmt(x, nd=4):
    return "" if x is None else ("%.*f" % (nd, x))


def _markdown(rows: List[dict], methods: Sequence[str], reference: Optional[str]) -> str:
    """Wide table per subset: metric | method mean +- std ... | diff vs reference (p)."""
    out = []
    for subset in sorted({r["subset"] for r in rows}, key=lambda s: (s != "all", s)):
        sub = [r for r in rows if r["subset"] == subset]
        n = max(r["n"] for r in sub) if sub else 0
        out.append("### %s (%d patients)\n" % (subset, n))
        head = ["metric"] + list(methods)
        others = [m for m in methods if reference and m != reference]
        head += ["%s - %s (p, frac better)" % (m, reference) for m in others]
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "---|" * len(head))
        for metric in dict.fromkeys(r["metric"] for r in sub):
            by = {r["method"]: r for r in sub if r["metric"] == metric}
            cells = ["%s (%s)" % (metric, by[methods[0]]["better"])]
            for m in methods:
                r = by[m]
                cells.append("%s +- %s" % (_fmt(r["mean"]), _fmt(r["std"])) if r["mean"] is not None else "n/a")
            for m in others:
                r = by[m]
                if r.get("diff_vs_ref") is None:
                    cells.append("n/a")
                else:
                    p = r["p_wilcoxon"]
                    cells.append("%+.4f (p %s, %.0f %%)" % (r["diff_vs_ref"], "n/a" if p is None else "%.2g" % p,
                                                            100 * r["frac_better"]))
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def cmd_compare(a) -> int:
    a.reference = a.reference or None
    tables = {}
    for m in a.methods:
        p = os.path.join(a.root, m, "_aggregate", "per_group.csv")
        if not os.path.isfile(p):
            print("missing %s (run eval_miriad.sh / aggregate for %s first)" % (p, m), file=sys.stderr)
            return 2
        tables[m] = _read_per_group(p)
    if a.reference and a.reference not in a.methods:
        print("--reference %s is not one of --methods" % a.reference, file=sys.stderr)
        return 2
    common = sorted(set.intersection(*(set(t) for t in tables.values())))
    for m, t in tables.items():
        extra = sorted(set(t) - set(common))
        if extra:
            print("WARNING: %s has %d patients not in every method (left out): %s"
                  % (m, len(extra), " ".join(extra[:5]) + (" ..." if len(extra) > 5 else "")), file=sys.stderr)
    if not common:
        print("no patients common to all methods", file=sys.stderr)
        return 2
    if a.metrics:
        metrics = a.metrics
    else:   # aggregate metrics present as a column for every method
        metrics = [k for k in SUBJECT_METRICS + PAIR_METRICS + GROUP_METRICS
                   if all(k in tables[m][common[0]] for m in a.methods)]
    rows = compare_rows(tables, a.methods, a.reference, metrics, common, "all")
    for dx in ("AD", "HC"):
        gs = [g for g in common if _diagnosis(g) == dx]
        if gs:
            rows += compare_rows(tables, a.methods, a.reference, metrics, gs, dx)
    os.makedirs(a.out, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(a.out, "compare.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    md = _markdown(rows, a.methods, a.reference)
    with open(os.path.join(a.out, "compare.md"), "w") as f:
        f.write("# Patient-level comparison (%s)\n\nMean +- std over patients (each patient = mean over its "
                "scans / pairs). Differences are paired per patient vs %s; p = two-sided Wilcoxon "
                "signed-rank; frac better = share of patients where the method beats the reference "
                "(direction per metric in parentheses). Centrality is 0 for identity by construction.\n\n%s"
                % (", ".join(a.methods), a.reference or "(none)", md))
    print(md)
    print("wrote %s/compare.{md,csv}" % a.out)
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
    c = sub.add_parser("compare", help="side-by-side table of several methods (paired per patient)")
    c.add_argument("--root", required=True, help="evaluation root with <method>/_aggregate/per_group.csv")
    c.add_argument("--methods", nargs="+", required=True)
    c.add_argument("--reference", default="identity", help="paired differences vs this method ('' = none)")
    c.add_argument("--metrics", nargs="+", default=None, help="default: the aggregate metrics")
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_compare)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.func(a))


if __name__ == "__main__":
    sys.exit(main())
