#!/usr/bin/env python3
"""Summarize and rank training runs (retrain-prereqs design sections 7.3 and 8).

  python3 scripts/summarize_runs.py <results_root> [--glob 'r4sw_*'] [--csv out.csv] [--last-k 2]
                                    [--rank true|false] [--length-curve RUN8,RUN12,RUN16]

Stdlib only, Python 3.6 syntax, read-only (login node). For every <results_root>/<run> matching
the glob: <run>/logs/metrics.jsonl ({"step", "time", "data"} per line; malformed lines are skipped
and counted on stderr; a missing file -> 'no metrics') and <run>/config.json ``args`` (lr,
weight_decay, optimizer, group_size, deform_group_size, steps, val_group_size; missing -> '?').
"Final" = the value in the LAST record that contains the key; best = min (RMS) / max (Dice).

Ranking (7.3): L = mean of val/mean_pair_rms_mm over the last --last-k validations (lower is
better), Dp = mean of val_deform/pair_dice over the last --last-k validations (higher is better).
Excluded (X): no L / Dp, incomplete (the last record holding val/mean_pair_rms_mm is not at
config.json args.steps: the run died or hit its time limit; it stays in the table), a non-finite
final L or Dp metric, nonfinite_total / steps > 0.06, final
train/collapse_suspect > 0, final val_deform/neg_jac_pct > 0.5 (a key never logged does not
exclude; shown as '?'). Score = rank(L) + rank(Dp) (1 = best); tie-break by lower final
neg_jac_pct, then lower final val_affine_g/pair_rms_mm. Runs with different val_group_size values
are NOT ranked (ERROR on stderr, unranked table, exit 0). The rev column comes from the last
<results_root>/r4sw_submissions.jsonl record of the run ('?' if none); ranked runs with different
revs give a WARNING on stderr.

--length-curve RUN8,RUN12,RUN16: the phase-C length decision `length_decision` on (steps from
config.json, L, Dp) of the three runs: 'LENGTH <steps> (run <name>)', 'LENGTH 20000 (new run;
command: ...)' or 'LENGTH UNDECIDED (<reason>)'.

Exit code 0 (also with incomplete runs); 2 only if <results_root> does not exist.
"""
import argparse
import csv
import fnmatch
import json
import math
import os
import sys

L_KEY = "val/mean_pair_rms_mm"
DP_KEY = "val_deform/pair_dice"
NEG_JAC_KEY = "val_deform/neg_jac_pct"
TIE_KEY = "val_affine_g/pair_rms_mm"
LINEAR_SETS = ("val_rigid", "val_affine", "val_rigid_g", "val_affine_g")
DEFORM_FINAL_KEYS = ("ncc_template", "ncc_pair", "neg_jac_pct", "inv_consistency_mm")
MAX_NONFINITE_FRAC = 0.06
MAX_NEG_JAC_PCT = 0.5
SUBMISSIONS = "r4sw_submissions.jsonl"
CFG_KEYS = (("lr", "lr"), ("wd", "weight_decay"), ("opt", "optimizer"), ("N", "group_size"),
            ("DN", "deform_group_size"), ("steps", "steps"), ("vgs", "val_group_size"))


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _str2bool(s):
    if s.lower() in ("true", "1", "yes"):
        return True
    if s.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("expected true or false, got %r" % s)


# ------------------------------------------------------------------------------------- loading

def load_records(path):
    """(records [(step, time, data)], malformed line count); records need a data dict and an int step."""
    recs, bad = [], 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if not isinstance(rec, dict) or not isinstance(rec.get("data"), dict):
                continue
            step = rec.get("step")
            if not isinstance(step, int) or isinstance(step, bool):
                continue
            recs.append((step, rec.get("time"), rec["data"]))
    return recs, bad


def load_config_args(run_dir):
    try:
        with open(os.path.join(run_dir, "config.json")) as f:
            return json.load(f).get("args") or {}
    except (OSError, ValueError, AttributeError):
        return {}


def load_revs(results_root):
    """{run name: rev} from the LAST submissions record of each run."""
    revs = {}
    path = os.path.join(results_root, SUBMISSIONS)
    if not os.path.isfile(path):
        return revs
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and rec.get("run"):
                revs[rec["run"]] = rec.get("rev")
    return revs


def series(recs, key):
    return [(s, d[key]) for s, _, d in recs if key in d]


def final(recs, key):
    v = series(recs, key)
    return v[-1][1] if v else None


def best(recs, key, highest):
    """(best value, its step) over the finite values, or (None, None)."""
    v = [(val, s) for s, val in series(recs, key) if _finite(val)]
    if not v:
        return None, None
    val, s = (max if highest else min)(v, key=lambda t: t[0])
    return val, s


def last_k_mean(recs, key, k):
    v = [val for _, val in series(recs, key)][-k:]
    if not v:
        return None
    if not all(_finite(x) for x in v):
        return float("nan")
    return sum(float(x) for x in v) / len(v)


# ----------------------------------------------------------------------------------- summarize

def summarize_run(run_dir, name, last_k=2, revs=None):
    """One row dict (column -> value) of a run dir."""
    args = load_config_args(run_dir)
    row = {"run": name, "rev": ((revs or {}).get(name) or "?")[:8]}
    for col, key in CFG_KEYS:
        row[col] = args.get(key)
    row["_has_vgs"] = "val_group_size" in args
    path = os.path.join(run_dir, "logs", "metrics.jsonl")
    row["_bad_lines"] = 0
    if not os.path.isfile(path):
        row["_no_metrics"] = True
        recs = []
    else:
        row["_no_metrics"] = False
        recs, row["_bad_lines"] = load_records(path)
    row["last_step"] = max(s for s, _, _ in recs) if recs else None
    times = [t for _, t, _ in recs if _finite(t)]
    row["wall_h"] = (max(times) - min(times)) / 3600.0 if len(times) > 1 else None
    peaks = [d["gpu_peak_mb"] for _, _, d in recs if _finite(d.get("gpu_peak_mb"))]
    row["peak_GiB"] = max(peaks) / 1024.0 if peaks else None
    row["nonfinite"] = final(recs, "nonfinite_total")
    row["collapse"] = final(recs, "train/collapse_suspect")
    for st in LINEAR_SETS:
        for m in ("center", "pair"):
            key = "%s/%s_rms_mm" % (st, m)
            col = "%s.%s" % (st[len("val_"):], m)
            row[col] = final(recs, key)
            row[col + ".best"], row[col + ".at"] = best(recs, key, False)
    row["mean_pair"] = final(recs, L_KEY)
    row["mean_pair.best"], row["mean_pair.at"] = best(recs, L_KEY, False)
    for m in ("template_dice", "pair_dice"):
        key = "val_deform/" + m
        row[m] = final(recs, key)
        row[m + ".best"], row[m + ".at"] = best(recs, key, True)
    for m in DEFORM_FINAL_KEYS:
        row[m] = final(recs, "val_deform/" + m)
    l_steps = [s for s, _ in series(recs, L_KEY)]
    row["_last_val_step"] = l_steps[-1] if l_steps else None
    row["L"] = last_k_mean(recs, L_KEY, last_k)
    row["Dp"] = last_k_mean(recs, DP_KEY, last_k)
    row["excluded"] = exclusion(row)
    row["score"] = row["rank"] = None
    return row


def exclusion(row):
    """Reason string if the run is excluded from the ranking (7.3), else None."""
    if row["L"] is None or row["Dp"] is None:
        return "no L / Dp"
    # Same completion rule as sweep_level4.run_progress: the last record holding L_KEY is at
    # config.json args.steps (a run that died or hit its --time limit is not ranked).
    if row["_last_val_step"] != row["steps"]:
        return "incomplete (last validation %s of %s)" % (row["_last_val_step"],
                                                         "?" if row["steps"] is None else row["steps"])
    if not (_finite(row["L"]) and _finite(row["Dp"]) and _finite(row["mean_pair"]) and _finite(row["pair_dice"])):
        return "non-finite final metric"
    steps = row["steps"] if _is_int(row["steps"]) and row["steps"] > 0 else row["last_step"]
    if _finite(row["nonfinite"]) and steps and float(row["nonfinite"]) / steps > MAX_NONFINITE_FRAC:
        return "nonfinite_total %s / %s steps > %g" % (row["nonfinite"], steps, MAX_NONFINITE_FRAC)
    if _finite(row["collapse"]) and row["collapse"] > 0:
        return "collapse_suspect %s" % row["collapse"]
    if _finite(row["neg_jac_pct"]) and row["neg_jac_pct"] > MAX_NEG_JAC_PCT:
        return "neg_jac_pct %s > %g" % (row["neg_jac_pct"], MAX_NEG_JAC_PCT)
    return None


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _ranks(values, highest):
    """Competition ranks (1 = best, ties share the lower rank)."""
    order = sorted(values, reverse=highest)
    return [order.index(v) + 1 for v in values]


def _inf(v):
    return float(v) if _finite(v) else float("inf")


def rank_rows(rows):
    """Sets score / rank on the non-excluded rows; returns all rows sorted (ranked first, excluded last)."""
    ok = [r for r in rows if r["excluded"] is None]
    rl = _ranks([float(r["L"]) for r in ok], highest=False)
    rd = _ranks([float(r["Dp"]) for r in ok], highest=True)
    for r, a, b in zip(ok, rl, rd):
        r["score"] = a + b
    ok.sort(key=lambda r: (r["score"], _inf(r["neg_jac_pct"]), _inf(r["affine_g.pair"]), r["run"]))
    for i, r in enumerate(ok):
        r["rank"] = i + 1
    rest = sorted((r for r in rows if r["excluded"] is not None), key=lambda r: r["run"])
    return ok + rest


def length_decision(points, tol_l=0.05, tol_dp=0.003):
    """Final level-4 length (7.3) from points [(steps, L, Dp)] of the 8000 / 12000 / 16000 runs.

    s* = the shortest length with L_s <= L_long + tol_l and Dp_s >= Dp_long - tol_dp. If s* is the
    longest run and it beats the middle one by more than the tolerance on BOTH metrics, the curve
    is still steep -> a new run of longest + (longest - middle) steps (20000). Otherwise s*.
    A missing / non-finite point -> 'UNDECIDED (<reason>)'."""
    pts = list(points)
    if len(pts) != 3:
        return "UNDECIDED (need 3 points, got %d)" % len(pts)
    for s, l_val, dp in pts:
        if not _is_int(s) or not (_finite(l_val) and _finite(dp)):
            return "UNDECIDED (missing point at %s steps)" % (s,)
    pts.sort(key=lambda p: p[0])
    if len(set(p[0] for p in pts)) != 3:
        return "UNDECIDED (duplicate run lengths %s)" % [p[0] for p in pts]
    (_, _, _), (s2, l2, d2), (s3, l3, d3) = pts
    s_star = next(s for s, l_val, dp in pts if l_val <= l3 + tol_l and dp >= d3 - tol_dp)
    if s_star == s3 and (l2 - l3) > tol_l and (d3 - d2) > tol_dp:
        return s3 + (s3 - s2)
    return s_star


# -------------------------------------------------------------------------------------- output

COLUMNS = (["run", "rev", "lr", "wd", "opt", "N", "DN", "vgs", "step", "wall_h", "peak_GiB", "nonfinite"]
           + ["%s.%s%s" % (st[len("val_"):], m, sfx) for st in LINEAR_SETS for m in ("center", "pair")
              for sfx in ("", ".best", ".at")]
           + ["mean_pair", "mean_pair.best", "mean_pair.at"]
           + ["%s%s" % (m, sfx) for m in ("template_dice", "pair_dice") for sfx in ("", ".best", ".at")]
           + list(DEFORM_FINAL_KEYS) + ["L", "Dp", "score", "rank", "X"])


def _cell(row, col):
    if col == "step":
        if row["_no_metrics"]:
            return "no metrics"
        return "%s/%s" % ("?" if row["last_step"] is None else row["last_step"],
                          "?" if row["steps"] is None else row["steps"])
    if col == "X":
        return "" if row["excluded"] is None else "X (%s)" % row["excluded"]
    if col == "vgs" and not row["_has_vgs"]:
        return "?"
    v = row.get(col)
    if v is None:
        return "?" if col not in ("score", "rank") else ""
    if isinstance(v, bool):
        return str(v)
    if _is_int(v):
        return str(v)
    if isinstance(v, float):
        return "%.4g" % v
    return str(v)


def write_table(rows, out):
    cells = [[_cell(r, c) for c in COLUMNS] for r in rows]
    widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(COLUMNS)]
    out.write("  ".join(c.ljust(w) for c, w in zip(COLUMNS, widths)).rstrip() + "\n")
    for row in cells:
        out.write("  ".join(v.ljust(w) for v, w in zip(row, widths)).rstrip() + "\n")


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([_cell(r, c) for c in COLUMNS])


def length_curve(results_root, names, last_k, revs, out):
    """Prints the LENGTH line for the three --length-curve runs."""
    rows = []
    for name in names:
        run_dir = os.path.join(results_root, name)
        if not os.path.isdir(run_dir):
            out.write("LENGTH UNDECIDED (run %s not found)\n" % name)
            return
        row = summarize_run(run_dir, name, last_k, revs)
        if row["excluded"] is not None:
            out.write("LENGTH UNDECIDED (run %s excluded: %s)\n" % (name, row["excluded"]))
            return
        rows.append(row)
    by_steps = {}
    for r in rows:
        by_steps[r["steps"]] = r
    decision = length_decision([(r["steps"], r["L"], r["Dp"]) for r in rows])
    if not _is_int(decision):
        out.write("LENGTH %s\n" % decision)
    elif decision in by_steps:
        out.write("LENGTH %d (run %s)\n" % (decision, by_steps[decision]["run"]))
    else:
        r = by_steps[max(by_steps)]
        vgs = r["vgs"] if r["vgs"] is not None else 16
        out.write("LENGTH %d (new run; command: RUN_NAME=r4_final_s%d STEPS=%d LR=%s WD=%s N=%s DEFORM_N=%s "
                  "VAL_GROUP_SIZE=%s sbatch --job-name=r4final --time=30:00:00 --mem=64G slurm/r_level4.sh)\n"
                  % (decision, decision, decision, _num(r["lr"]), _num(r["wd"]), r["N"], r["DN"], vgs))


def _num(v):
    return "%g" % v if _finite(v) else "?"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_root")
    ap.add_argument("--glob", default="r4sw_*", help="run directory name pattern (fnmatch)")
    ap.add_argument("--csv", default=None, help="also write the table to this CSV file")
    ap.add_argument("--last-k", type=int, default=2, help="validations averaged for L and Dp")
    ap.add_argument("--rank", type=_str2bool, default=True)
    ap.add_argument("--length-curve", default=None, metavar="RUN8,RUN12,RUN16")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.results_root):
        sys.stderr.write("ERROR: results root %s does not exist\n" % a.results_root)
        return 2
    if a.last_k < 1:
        ap.error("--last-k must be >= 1")
    revs = load_revs(a.results_root)
    rows = []
    for name in sorted(os.listdir(a.results_root)):
        run_dir = os.path.join(a.results_root, name)
        if not (os.path.isdir(run_dir) and fnmatch.fnmatch(name, a.glob)):
            continue
        row = summarize_run(run_dir, name, a.last_k, revs)
        if row["_bad_lines"]:
            sys.stderr.write("%s: skipped %d malformed metrics.jsonl line(s)\n" % (name, row["_bad_lines"]))
        if row["_no_metrics"]:
            sys.stderr.write("%s: no metrics\n" % name)
        rows.append(row)
    ranked = a.rank
    if ranked:
        ok = [r for r in rows if r["excluded"] is None]
        vgs = sorted(set(repr(r["vgs"]) for r in ok))
        if len(vgs) > 1:
            sys.stderr.write("ERROR: runs use different validation groups (%s); not ranked\n" % ", ".join(vgs))
            ranked = False
        else:
            rev_set = sorted(set(r["rev"] for r in ok))
            if len(rev_set) > 1:
                sys.stderr.write("WARNING: ranked runs have different code revisions (%s)\n" % ", ".join(rev_set))
    if ranked:
        rows = rank_rows(rows)
    else:
        for r in rows:
            r["score"] = r["rank"] = None
        rows.sort(key=lambda r: r["run"])
    write_table(rows, sys.stdout)
    if a.csv:
        write_csv(rows, a.csv)
    if a.length_curve:
        names = [n for n in a.length_curve.split(",") if n]
        if len(names) != 3:
            sys.stdout.write("LENGTH UNDECIDED (--length-curve needs 3 runs, got %d)\n" % len(names))
        else:
            length_curve(a.results_root, names, a.last_k, revs, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
