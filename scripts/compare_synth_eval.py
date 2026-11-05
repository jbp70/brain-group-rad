"""Re-print the paired synthetic-contrast table of eval_synth.json (synth-contrast design 8 step 5).

    python scripts/compare_synth_eval.py <out>/eval_synth.json
    python scripts/compare_synth_eval.py <jobA>/eval_synth.json <jobB>/eval_synth.json

One JSON: one row per (preset, metric) with one column per run and diff = second - first.
Two JSONs (fallback when the two runs were evaluated by separate jobs): the runs of both files
are merged, and every preset in both files must have the same sha256 per synthetic set in every
run; a mismatch is an error naming the preset (exit 1). ``--metrics`` filters the metric names
(substring match, comma list). Stdlib only, Python 3.8.
"""
import argparse
import json
import sys


def load(path):
    with open(path) as f:
        rep = json.load(f)
    if "results" not in rep or "runs" not in rep:
        raise ValueError("%s is not an eval_synth.json (no results / runs)" % path)
    return rep


def check_sha(reports, paths):
    """Error lines for presets whose synthetic sets differ (sha256) between any two runs."""
    errors = []
    presets = []
    for rep in reports:
        for p in rep.get("presets", []):
            if p not in presets:
                presets.append(p)
    for p in presets:
        seen = []
        for rep, path in zip(reports, paths):
            res = rep["results"].get(p)
            if not res:
                continue
            for label, r in sorted(res["runs"].items()):
                seen.append(("%s:%s" % (path, label), r.get("sha256") or {}))
        if len(seen) < 2:
            continue
        ref_name, ref = seen[0]
        for name, sha in seen[1:]:
            if sha != ref:
                diff = sorted(k for k in set(sha) | set(ref) if sha.get(k) != ref.get(k))
                errors.append("preset %s: synthetic sets differ between %s and %s (%s)"
                              % (p, ref_name, name, ", ".join(diff)))
    return errors


def table(reports, metrics_filter=None):
    """Rows [preset, metric, value per run..., diff] over the merged runs."""
    labels = []
    for i, rep in enumerate(reports):
        for run in rep["runs"]:
            lb = run["label"] if len(reports) == 1 else "%d:%s" % (i, run["label"])
            labels.append((lb, rep, run["label"]))
    presets = []
    for rep in reports:
        for p in rep.get("presets", []):
            if p not in presets:
                presets.append(p)
    rows = []
    for p in presets:
        per = []
        for lb, rep, orig in labels:
            res = rep["results"].get(p)
            per.append(((res or {}).get("runs") or {}).get(orig, {}).get("metrics", {}))
        if not per or not all(per):
            continue
        keys = sorted(set.intersection(*[set(m) for m in per]))
        if metrics_filter:
            keys = [k for k in keys if any(s in k for s in metrics_filter)]
        for k in keys:
            vals = [m[k] for m in per]
            diff = ""
            if len(vals) == 2:
                try:
                    diff = float(vals[1]) - float(vals[0])
                except (TypeError, ValueError):
                    diff = ""
            rows.append([p, k] + vals + [diff])
    return [lb for lb, _, _ in labels], rows


def _fmt(v):
    if isinstance(v, float):
        return "%.6g" % v
    return str(v)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("json", nargs="+", help="one or two eval_synth.json files")
    ap.add_argument("--metrics", default="", help="comma list of metric substrings to keep")
    a = ap.parse_args(argv)
    if len(a.json) > 2:
        ap.error("give one or two eval_synth.json files")
    try:
        reports = [load(p) for p in a.json]
    except (OSError, ValueError) as e:
        print("compare_synth_eval: error: %s" % e, file=sys.stderr)
        return 2
    errors = check_sha(reports, a.json)
    for line in errors:
        print("compare_synth_eval: error: %s" % line, file=sys.stderr)
    filt = [s.strip() for s in a.metrics.split(",") if s.strip()]
    labels, rows = table(reports, filt)
    header = ["preset", "metric"] + labels + (["diff"] if len(labels) == 2 else [])
    print("\t".join(header))
    for r in rows:
        print("\t".join(_fmt(v) for v in (r if len(labels) == 2 else r[:-1])))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
