"""Step times (and extra diagnostics) from training logs, TF-free (milestone 9 design 4.5, 3.6).

    python scripts/step_times_from_metrics.py --runs <run dir> [<run dir> ...] [--min-step 1000] \
        [--extra-keys diag_u_res_max_vox] [--args-keys global_start_step,group_size] --out step_times.json

`metrics.jsonl` (``<run>/logs/metrics.jsonl``, one ``{"step", "time", "data"}`` object per log call)
holds 25-step WINDOW MEANS, not per-step values (`runtime.run.Trainer.fit`). Per run, over the
records with ``step >= --min-step``, this reports for ``time/compute_ms`` (the task-mixed mean,
every run) and for each ``train_<task>/compute_ms`` (cascade runs only): the median and IQR of
the window means and the number of windows. ``per_task`` is true iff a ``train_<task>/compute_ms``
key exists. ``--extra-keys``: each key is matched with or without a ``train_<task>/`` prefix and
reported as the median and the MAX of the window means (null if absent). ``--args-keys``: copied
from ``config.json`` ``args`` (null if missing). A missing run dir, metrics file or config.json
is marked ``missing``; unparsable lines are counted and skipped. Always exits 0.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

TASK_TIME = re.compile(r"^train_([A-Za-z0-9]+)/compute_ms$")
MIXED = "time/compute_ms"


def _csv(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def _stats(values):
    v = np.asarray(values, np.float64)
    q25, q50, q75 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    return {"median": q50, "iqr": [q25, q75], "iqr_width": q75 - q25, "n_windows": int(v.size)}


def _metrics_path(run):
    for p in (os.path.join(run, "logs", "metrics.jsonl"), os.path.join(run, "metrics.jsonl")):
        if os.path.isfile(p):
            return p
    return None


def read_metrics(path):
    """-> (list of (step, data dict), unparsable line count)."""
    recs, bad = [], 0
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                step = r["step"]
                data = r.get("data", {})
                if not isinstance(step, int) or not isinstance(data, dict):
                    raise ValueError("bad record")
            except (ValueError, KeyError, TypeError):
                bad += 1
                continue
            recs.append((step, data))
    return recs, bad


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def summarize_run(run, min_step=1000, extra_keys=(), args_keys=()):
    out = {"run": run, "status": "ok", "metrics": "ok", "config": "ok", "n_records": 0, "n_windows": 0,
           "unparsable_lines": 0, "per_task": False, "timing": {}, "extra": {}, "args": {}}
    if not os.path.isdir(run):
        out.update(status="missing", metrics="missing", config="missing")
        out["extra"] = {k: None for k in extra_keys}
        out["args"] = {k: None for k in args_keys}
        return out
    path = _metrics_path(run)
    if path is None:
        out["metrics"] = "missing"
        out["extra"] = {k: None for k in extra_keys}
    else:
        recs, bad = read_metrics(path)
        out["n_records"] = len(recs)
        out["unparsable_lines"] = bad
        win = [d for s, d in recs if s >= int(min_step) and "event" not in d]
        out["n_windows"] = len(win)
        keys = set()
        for d in win:
            keys.update(d)
        timing = {}
        task_keys = sorted(k for k in keys if TASK_TIME.match(k))
        for k in ([MIXED] if MIXED in keys else []) + task_keys:
            vals = [d[k] for d in win if _is_num(d.get(k))]
            if vals:
                timing[k] = _stats(vals)
        out["timing"] = timing
        out["per_task"] = bool(task_keys)
        for ek in extra_keys:
            matched = sorted(k for k in keys if k == ek or re.match(r"^train_[A-Za-z0-9]+/%s$" % re.escape(ek), k))
            vals = [d[k] for d in win for k in matched if _is_num(d.get(k))]
            out["extra"][ek] = ({"median": float(np.median(vals)), "max": float(np.max(vals)),
                                 "n_windows": len(vals), "keys": matched} if vals else None)
    cfg = os.path.join(run, "config.json")
    if not os.path.isfile(cfg):
        out["config"] = "missing"
        out["args"] = {k: None for k in args_keys}
    else:
        try:
            with open(cfg) as fh:
                args = (json.load(fh).get("args") or {})
        except (ValueError, OSError):
            out["config"] = "unparsable"
            args = {}
        out["args"] = {k: args.get(k) for k in args_keys}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run directories")
    ap.add_argument("--min-step", type=int, default=1000)
    ap.add_argument("--extra-keys", default="", help="comma list, matched with or without train_<task>/")
    ap.add_argument("--args-keys", default="", help="comma list of config.json args entries")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = {"min_step": a.min_step, "note": "window means (25-step), not per-step values", "runs": []}
    for run in a.runs:
        s = summarize_run(run, a.min_step, _csv(a.extra_keys), _csv(a.args_keys))
        res["runs"].append(s)
        print("%s: %s (metrics %s, config %s, %d windows >= step %d, %d unparsable, per_task %s)" % (
            run, s["status"], s["metrics"], s["config"], s["n_windows"], a.min_step, s["unparsable_lines"],
            s["per_task"]))
        for k, v in s["timing"].items():
            print("  %-28s median %8.1f ms  IQR %8.1f-%8.1f  (%d windows)" % (k, v["median"], v["iqr"][0], v["iqr"][1],
                                                                          v["n_windows"]))
        for k, v in s["extra"].items():
            print("  %-28s %s" % (k, "median %.4g max %.4g (%d windows)" % (v["median"], v["max"], v["n_windows"])
                                  if v else "null"))
        if s["args"]:
            print("  args: %s" % json.dumps(s["args"]))
    if a.out:
        d = os.path.dirname(os.path.abspath(a.out))
        if not os.path.isdir(d):
            os.makedirs(d)
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
            fh.write("\n")
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
