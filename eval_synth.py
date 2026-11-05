"""Post-training synthetic-contrast evaluation of one or two trained runs (synth-contrast design 8).

    python eval_synth.py --runs <real_only_run> <mixed_run> --level 1 --out <dir>
    python eval_synth.py --runs <run> --level 4 --presets random_0,t2_like --out <dir>

Restores each run's ckpt/last read-only (nothing is written into a run directory), checks it
against the run's logged validation (sanity gate), then validates both models on the same
synthetic validation sets per preset (`data_loading.synth.EVAL_PRESETS`). Output in --out:
eval_synth.json and eval_synth.csv (`runtime.eval_run`). SLURM wrapper: slurm/eval_synth.sh.

Exit codes: 0 OK, 2 usage / run / data error, 5 sanity gate failed (restore or config mismatch).
"""
import argparse
import os
import sys

PRESET_NAMES = ("random_0", "random_1", "random_2", "random_3", "t2_like", "flair_like", "fdg_like",
                "amyloid_pos", "amyloid_neg", "tau_like", "t1_like")   # = synth.PRESET_NAMES


def build_parser():
    p = argparse.ArgumentParser(description="Synthetic-contrast evaluation of trained runs (no TF at --help)")
    p.add_argument("--runs", nargs="+", required=True, metavar="RUN",
                   help="1 or 2 run directories (<results>/<run>; first = real-only, second = mixed)")
    p.add_argument("--level", type=int, default=1, choices=(4, 2, 1),
                   help="pyramid level; must equal each run's level (default 1)")
    p.add_argument("--out", required=True, help="output directory (created; must not be inside a run dir)")
    p.add_argument("--presets", default="", help="comma list of presets (default all: %s)" % ",".join(PRESET_NAMES))
    p.add_argument("--data", default=None, help="export root (default: each run's saved args.data)")
    p.add_argument("--sanity-gate", choices=("on", "off"), default="on",
                   help="compare the restored model with the run's logged validation (off: tests only)")
    p.add_argument("--quiet", action="store_true", help="no progress lines")
    return p


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= len(args.runs) <= 2:
        parser.error("--runs takes 1 or 2 run directories, got %d" % len(args.runs))
    presets = [s.strip() for s in args.presets.split(",") if s.strip()] or list(PRESET_NAMES)
    bad = [s for s in presets if s not in PRESET_NAMES]
    if bad:
        parser.error("unknown preset(s) %s; known: %s" % (",".join(bad), ",".join(PRESET_NAMES)))
    if len(set(presets)) != len(presets):
        parser.error("--presets lists a preset twice")
    args.presets_parsed = presets
    for r in args.runs:
        if not os.path.isfile(os.path.join(r, "config.json")):
            parser.error("run directory %s has no config.json" % r)
        out, run = os.path.realpath(args.out), os.path.realpath(r)
        if out == run or out.startswith(run.rstrip(os.sep) + os.sep):
            parser.error("--out %s is inside the run directory %s (eval never writes into a run)" % (args.out, r))
    return args


def main(argv=None):
    args = parse_args(argv)                         # argparse errors exit 2, no TF imported yet
    import tensorflow as tf                          # noqa: F401
    from models.precision import configure_tf_precision
    configure_tf_precision(tf32=False, memory_growth=True)   # FIRST TF call (as main.py)
    from runtime import eval_run as E
    try:
        return E.run_eval(args.runs, args.level, args.out, args.presets_parsed, data_root=args.data,
                          sanity_gate_on=args.sanity_gate == "on", quiet=args.quiet)
    except E.EvalError as e:
        print("eval_synth.py: error: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
