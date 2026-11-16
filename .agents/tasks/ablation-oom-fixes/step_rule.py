#!/usr/bin/env python3
"""STEPS of the from-scratch level-1 run (ablation A1) from a smoke's metrics.jsonl (report.md 5).

    python3 .agents/tasks/ablation-oom-fixes/step_rule.py <smoke run dir> [DEFORM_START 100] [VAL_EVERY 500]

t_lin = mean of time/compute_ms + time/data_ms over the log windows ending in (25, DEFORM_START]
(the first window holds the linear traces), t_mix = the same over windows ending after
DEFORM_START + 25, v = mean val/wall_s without the first pass (tracing). Budget 229860 s (63.85 h);
STEPS = 0.99 x 229860 / (3/16 t_lin + 13/16 t_mix + v / VAL_EVERY). Stdlib only (login node ok).
"""
import json
import os
import sys

BUDGET_S = 229860.0


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    run = argv[1]
    ds = int(argv[2]) if len(argv) > 2 else 100
    val_every = int(argv[3]) if len(argv) > 3 else 500
    with open(os.path.join(run, "logs", "metrics.jsonl")) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    win = [(r["step"], (r["data"]["time/compute_ms"] + r["data"].get("time/data_ms", 0.0)) / 1e3)
           for r in recs if isinstance(r.get("data"), dict) and "time/compute_ms" in r["data"]]
    lin = [t for s, t in win if 25 < s <= ds]
    mix = [t for s, t in win if s > ds + 25]
    val = [r["data"]["val/wall_s"] for r in recs if isinstance(r.get("data"), dict) and "val/wall_s" in r["data"]]
    if not lin or not mix:
        print("not enough log windows: %d linear-only, %d mixed" % (len(lin), len(mix)))
        return 1
    t_lin, t_mix = sum(lin) / len(lin), sum(mix) / len(mix)
    v = sum(val[1:]) / len(val[1:]) if len(val) > 1 else (val[0] if val else 0.0)
    per_step = 3.0 / 16 * t_lin + 13.0 / 16 * t_mix + v / val_every
    print("windows: %d linear-only, %d mixed; validation passes %d" % (len(lin), len(mix), len(val)))
    print("t_lin %.3f s | t_mix %.3f s | v %.0f s | VAL_EVERY %d" % (t_lin, t_mix, v, val_every))
    print("STEPS ~ %d  (= 0.99 x %.0f / %.3f s per step)" % (0.99 * BUDGET_S / per_step, BUDGET_S, per_step))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
