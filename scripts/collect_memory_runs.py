#!/usr/bin/env python3
"""Collect the per-config results of a real-trainer memory job (fp32-decoder-width FEAT-003,
slurm/m11_fp32_width_memory.sh) into one JSON file and print a table.

  python scripts/collect_memory_runs.py --status <status.tsv> --runs <runs_dir> --out <json>
                                        [--skip-steps 2] [--job JOB]

Stdlib only, Python 3.8 syntax, read-only apart from --out. ``status.tsv`` (written by the job,
tab-separated, header line) has the columns

  name  level  dtype  widths  n  max_pairs  rc  elapsed_s  log

For every row: <runs>/<name>/logs/metrics.jsonl (runtime/logging.make_run_dir layout; one
{"step", "time", "data"} object per line, malformed lines skipped) and <runs>/<name>/config.json
(args.filters, args.decoder_widths, param_counts.trainable, param_counts.deform_decoder); a missing
run directory gives nulls. Record fields: name, level, dtype, widths, filters, decoder_widths, n,
max_pairs, pairs (max train_deform/n_pairs), status, rc, elapsed_s, steps_logged (records with
time/compute_ms), gpu_peak_mb (max) / gpu_peak_gib, mean_deform_step_s / mean_linear_step_s
((time/data_ms + time/compute_ms) / 1000 averaged over the records with step > --skip-steps; a
record is linear iff one of its keys starts with train_rigid/ or train_affine/, else deform; null
when there is none), nonfinite_total (last record holding it), param_count, deform_decoder_params,
error (last log line containing 'Error' or 'error', <= 300 chars).

Status: rc -1 -> skipped_time (job time guard), rc -2 -> skipped_source (the width's random
cascade source was not built), rc 124 / 137 -> timeout, rc != 0 and the log contains
ResourceExhaustedError / 'OOM when allocating' / 'out of memory' -> oom, rc 0 and >= 1 logged step
-> ok, else abort. The OOM text is only checked for failed runs: a successful TF run may log the
BFC allocator warning "ran out of memory trying to allocate ... this is not a failure".

Writes {"job": ..., "skip_steps": k, "records": [...]} (indent 1) and prints a fixed-width table.
Exit code 2 if the status file is missing, else 0.
"""
import argparse
import json
import os
import sys

COLUMNS = ("name", "level", "dtype", "widths", "n", "max_pairs", "rc", "elapsed_s", "log")
OOM_TEXTS = ("ResourceExhaustedError", "OOM when allocating", "out of memory")
LINEAR_PREFIXES = ("train_rigid/", "train_affine/")
PAIRS_KEY = "train_deform/n_pairs"
ERROR_MAX_CHARS = 300


def _int(v, default=None):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def _num(v):
    try:
        x = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return int(x) if x == int(x) else x


def read_status(path):
    """Rows of status.tsv as dicts (header line skipped; short rows padded with '')."""
    rows = []
    with open(path) as f:
        lines = f.read().splitlines()
    for ln in lines[1:]:
        if not ln.strip():
            continue
        parts = ln.split("\t")
        parts += [""] * (len(COLUMNS) - len(parts))
        rows.append(dict(zip(COLUMNS, parts)))
    return rows


def read_metrics(path):
    """metrics.jsonl records (dicts with step / data); malformed lines are skipped."""
    out = []
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for ln in f:
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            if isinstance(rec, dict) and isinstance(rec.get("data"), dict):
                out.append(rec)
    return out


def read_json(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except ValueError:
        return None


def read_text(path):
    if not path or not os.path.isfile(path):
        return ""
    with open(path, errors="replace") as f:
        return f.read()


def _is_linear(data):
    return any(k.startswith(LINEAR_PREFIXES) for k in data)


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def classify(rc, steps_logged, log_text):
    if rc == -1:
        return "skipped_time"
    if rc == -2:
        return "skipped_source"
    if rc in (124, 137):
        return "timeout"
    if rc != 0 and any(t in log_text for t in OOM_TEXTS):
        return "oom"
    if rc == 0 and steps_logged > 0:
        return "ok"
    return "abort"


def last_error(log_text):
    err = None
    for ln in log_text.splitlines():
        if "Error" in ln or "error" in ln:
            err = ln.strip()[:ERROR_MAX_CHARS]
    return err


def collect_row(row, runs, skip_steps):
    name = row["name"]
    run_dir = os.path.join(runs, name)
    recs = read_metrics(os.path.join(run_dir, "logs", "metrics.jsonl"))
    cfg = read_json(os.path.join(run_dir, "config.json")) or {}
    args = cfg.get("args") or {}
    counts = cfg.get("param_counts") or {}
    log_text = read_text(row["log"])
    rc = _int(row["rc"], default=-99)
    steps_logged = 0
    peaks, pairs, deform_s, linear_s = [], [], [], []
    nonfinite = None
    for rec in recs:
        d = rec["data"]
        if "gpu_peak_mb" in d and isinstance(d["gpu_peak_mb"], (int, float)):
            peaks.append(d["gpu_peak_mb"])
        if PAIRS_KEY in d and isinstance(d[PAIRS_KEY], (int, float)):
            pairs.append(d[PAIRS_KEY])
        if "nonfinite_total" in d:
            nonfinite = d["nonfinite_total"]
        if "time/compute_ms" not in d:
            continue
        steps_logged += 1
        step = _int(rec.get("step"), default=0)
        if step <= skip_steps:
            continue
        secs = (float(d.get("time/data_ms") or 0.0) + float(d["time/compute_ms"])) / 1000.0
        (linear_s if _is_linear(d) else deform_s).append(secs)
    peak = max(peaks) if peaks else None
    md, ml = _mean(deform_s), _mean(linear_s)
    return {"name": name, "level": _int(row["level"]), "dtype": row["dtype"], "widths": row["widths"],
            "filters": args.get("filters"), "decoder_widths": args.get("decoder_widths"),
            "n": _int(row["n"]), "max_pairs": _int(row["max_pairs"]),
            "pairs": int(max(pairs)) if pairs else None,
            "status": classify(rc, steps_logged, log_text), "rc": rc, "elapsed_s": _num(row["elapsed_s"]),
            "steps_logged": steps_logged, "gpu_peak_mb": peak,
            "gpu_peak_gib": round(peak / 1024.0, 2) if peak is not None else None,
            "mean_deform_step_s": round(md, 3) if md is not None else None,
            "mean_linear_step_s": round(ml, 3) if ml is not None else None,
            "nonfinite_total": nonfinite, "param_count": counts.get("trainable"),
            "deform_decoder_params": counts.get("deform_decoder"), "error": last_error(log_text)}


def _fmt(v, spec="%s"):
    return "-" if v is None else spec % v


def format_table(records):
    head = "%-22s %3s %-8s %-8s %4s %5s %-14s %8s %8s %8s %8s" % (
        "name", "lvl", "dtype", "widths", "N", "pairs", "status", "peakGiB", "deform_s", "linear_s", "paramsM")
    lines = [head, "-" * len(head)]
    for r in records:
        params = r["param_count"] / 1e6 if isinstance(r["param_count"], (int, float)) else None
        lines.append("%-22s %3s %-8s %-8s %4s %5s %-14s %8s %8s %8s %8s" % (
            r["name"], _fmt(r["level"]), r["dtype"], r["widths"], _fmt(r["n"]), _fmt(r["pairs"]), r["status"],
            _fmt(r["gpu_peak_gib"], "%.2f"), _fmt(r["mean_deform_step_s"], "%.2f"),
            _fmt(r["mean_linear_step_s"], "%.2f"), _fmt(params, "%.2f")))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--status", required=True, help="status.tsv written by the job")
    p.add_argument("--runs", required=True, help="directory holding <name>/ run dirs")
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--skip-steps", type=int, default=2, help="steps <= this are excluded from step times")
    p.add_argument("--job", default=None, help="job id stored in the JSON")
    a = p.parse_args(argv)
    if not os.path.isfile(a.status):
        print("ERROR: status file %s not found" % a.status, file=sys.stderr)
        return 2
    records = [collect_row(row, a.runs, a.skip_steps) for row in read_status(a.status)]
    out_dir = os.path.dirname(os.path.abspath(a.out))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(a.out, "w") as f:
        json.dump({"job": a.job, "skip_steps": a.skip_steps, "records": records}, f, indent=1)
    print(format_table(records))
    print("wrote %s (%d records)" % (a.out, len(records)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
