"""CLI of the calibrated memory model (milestone 9 phase 1, design 4.1-4.5). TF-free.

    python scripts/memory_model.py fit [--calibration C] [--out models/memory_coefficients.json]
    python scripts/memory_model.py predict --level 1 --task deform --n 4 --widths f0d1 --amp true \
        --checkpointing full --lean-sampler true --split-group-conv true --cascade chain42 [--json]
    python scripts/memory_model.py grid [--level 1] [--margin 0.15] [--out grid.json]
    python scripts/memory_model.py recommend [--n-target 4] [--t6-verdict integration_steps.json] \
        [--margin M] [--print-env] [--out recommendation.json]

`fit` writes the coefficients JSON (default models/memory_coefficients.json), which every other
command reads by default (pre-fit initial values with a warning if it is missing). Exit codes:
fit 1 = an ok point outside +-15 %, an OOM point predicted feasible, a failed_d2 point without a
limit flag or a range point above its upper end; recommend 3 = no configuration reaches
--n-target (R3.7 verdict, stop for Jay), 4 = hold-out gate failed (stop for Jay).
Local outputs go to .agents/tasks/milestone-9/ (the repo's scratch/ is not git-ignored).
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from models import memory_model as MM  # noqa: E402
from models import width_presets as WP  # noqa: E402

DEFAULT_CALIBRATION = os.path.join("models", "memory_calibration.json")
DEFAULT_COEFFICIENTS = os.path.join("models", "memory_coefficients.json")


def _bool(s):
    v = str(s).strip().lower()
    if v in ("1", "true", "yes", "y", "t"):
        return True
    if v in ("0", "false", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError("expected a boolean, got %r" % s)


def _path(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def _coefficients(path, required=False):
    full = _path(path)
    if os.path.exists(full):
        return MM.load_coefficients(full)
    if required:
        raise SystemExit("coefficients file %s not found (run `scripts/memory_model.py fit`)" % path)
    sys.stderr.write("WARNING: %s not found; using the pre-fit initial coefficients\n" % path)
    return MM.default_coefficients()


def _time_model(calib):
    spec = calib.get("step_times")
    return MM.StepTimeModel(spec) if spec else None


def _write(path, obj):
    full = _path(path)
    d = os.path.dirname(full)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(full, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
        fh.write("\n")
    print("wrote %s" % path)


def cmd_fit(a):
    calib = MM.load_calibration(_path(a.calibration))
    limits = MM.load_tensor_limits(a.tensor_limits)
    result, failures = MM.fit(calib, limits, lomo=not a.no_lomo)
    result["calibration"] = a.calibration
    _write(a.out or DEFAULT_COEFFICIENTS, result)
    if failures:
        print("fit: %d check(s) failed (tolerance not loosened; design 4.3)" % len(failures))
        return 1
    print("fit: every ok point within +-15 %, every OOM / failed point infeasible")
    return 0


def _workload_from_args(a):
    mc = a.widths
    if a.filters or a.decoder_widths:
        mc = {"filters": [int(x) for x in a.filters.split(",")] if a.filters else list(WP.DEFAULT_FILTERS),
              "decoder_widths": ([int(x) for x in a.decoder_widths.split(",")] if a.decoder_widths else None),
              "deform_decoder": True}
    tasks = tuple(a.tasks_run.split(",")) if a.tasks_run else None
    return MM.Workload(a.level, a.task, a.n, mc, a.checkpointing, a.amp, a.integration_steps, lncc=a.lncc,
                       lean_sampler=a.lean_sampler, lean_losses=a.lean_losses, split_group_conv=a.split_group_conv,
                       cascade=a.cascade, val_pair_chunk=a.val_pair_chunk, build_n=a.build_n, tasks_run=tasks,
                       gpu=a.gpu)


def cmd_predict(a):
    co = _coefficients(a.coefficients)
    calib = MM.load_calibration(_path(a.calibration))
    w = _workload_from_args(a)
    p = MM.predict(w, co, a.margin, skip_grad_bound=a.skip_grad_bound, fwd_transient_bound=a.fwd_transient_bound,
                   tensor_limits=MM.load_tensor_limits(a.tensor_limits), time_model=_time_model(calib))
    if a.json:
        print(json.dumps(p, indent=1, default=str))
    else:
        print("peak %.2f GiB (%.2f with margin %.2f), phase %s, resident %.2f GiB" % (
            p["peak_gib"], p["peak_margin_gib"], a.margin, p["peak_phase"], p["resident_gib"]))
        print("phases (GiB): " + ", ".join("%s %.2f" % kv for kv in p["phases"].items()))
        print("largest tensor: %s %s (%d elements, %s)" % (p["largest_tensor"]["desc"], p["largest_tensor"]["shape"],
                                                           p["largest_tensor"]["elements"],
                                                           p["largest_tensor"]["direction"]))
        print("limit flags: %s" % (p["limit_flags"] or "none"))
        print("feasible 44 / 36: %s / %s, validation %.2f GiB (chunk %d)" % (
            p["feasible_44"], p["feasible_36"], p["val_peak_gib"], w.val_pair_chunk))
        if p["step_s"] is not None:
            print("step %.1f s (estimate %.1f-%.1f s)" % (p["step_s"], p["step_s_range"][0], p["step_s_range"][1]))
    if a.out:
        _write(a.out, p)
    return 0


def cmd_grid(a):
    co = _coefficients(a.coefficients)
    res = MM.grid(co, a.margin, MM.load_tensor_limits(a.tensor_limits), level=a.level)
    if a.out:
        _write(a.out, res)
    return 0


def cmd_recommend(a):
    co = _coefficients(a.coefficients)
    calib = MM.load_calibration(_path(a.calibration))
    rec, code = MM.recommend(co, calib, n_target=a.n_target, margin=a.margin, t6_verdict=a.t6_verdict,
                             limits=MM.load_tensor_limits(a.tensor_limits), time_model=_time_model(calib),
                             coefficients_path=a.coefficients)
    if code == 0 and a.print_env:
        MM.print_env(rec)
    if a.out:
        _write(a.out, rec)
    return code


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    common.add_argument("--coefficients", default=DEFAULT_COEFFICIENTS)
    common.add_argument("--tensor-limits", default=None, help="probe_tensor_limit.py JSON (verified_max_elements)")
    common.add_argument("--out", default=None)
    f = sub.add_parser("fit", parents=[common], help="fit the coefficients, print the accuracy table")
    f.add_argument("--no-lomo", action="store_true", help="skip the leave-one-mode-out refits")
    p = sub.add_parser("predict", parents=[common], help="one workload")
    p.add_argument("--level", type=int, choices=(4, 2, 1), default=1)
    p.add_argument("--task", choices=MM.TASKS, default="deform")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--widths", default="default", help="preset name (%s)" % ", ".join(WP.PRESETS))
    p.add_argument("--filters", default=None)
    p.add_argument("--decoder-widths", default=None)
    p.add_argument("--checkpointing", choices=MM.CHECKPOINTING, default="none")
    p.add_argument("--amp", type=_bool, default=False)
    p.add_argument("--integration-steps", type=int, default=None)
    p.add_argument("--lncc", type=_bool, default=True)
    p.add_argument("--lean-sampler", type=_bool, default=False)
    p.add_argument("--lean-losses", type=_bool, default=False)
    p.add_argument("--split-group-conv", type=_bool, default=False)
    p.add_argument("--cascade", choices=MM.CASCADES, default="none")
    p.add_argument("--val-pair-chunk", type=int, default=16)
    p.add_argument("--build-n", type=int, default=None)
    p.add_argument("--tasks-run", default=None, help="comma list (default: the task)")
    p.add_argument("--gpu", choices=MM.GPUS, default="rtx8000")
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--skip-grad-bound", action="store_true")
    p.add_argument("--fwd-transient-bound", action="store_true",
                   help="order-independent estimate (one overlapping sampler forward)")
    p.add_argument("--json", action="store_true")
    g = sub.add_parser("grid", parents=[common], help="the R3.4 grid")
    g.add_argument("--level", type=int, choices=(4, 2, 1), default=1)
    g.add_argument("--margin", type=float, default=MM.DEFAULT_MARGIN)
    r = sub.add_parser("recommend", parents=[common], help="level-1 recommendation (design 4.4)")
    r.add_argument("--n-target", type=int, default=4)
    r.add_argument("--margin", type=float, default=None, help="default max(0.15, worst hold-out error)")
    r.add_argument("--t6-verdict", default=None, help="integration_steps.json (t6_accuracy_ok)")
    r.add_argument("--print-env", action="store_true")
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if not a.cmd:
        ap.error("choose a command: fit, predict, grid, recommend")
    if getattr(a, "n_target", 2) < 2:
        ap.error("--n-target must be >= 2")
    if getattr(a, "margin", 0.0) is not None and getattr(a, "margin", 0.0) < 0:
        ap.error("--margin must be >= 0")
    if a.cmd == "predict" and (a.n < 1 or a.val_pair_chunk < 1 or (a.build_n is not None and a.build_n < 1)):
        ap.error("--n, --val-pair-chunk and --build-n must be >= 1")
    return {"fit": cmd_fit, "predict": cmd_predict, "grid": cmd_grid, "recommend": cmd_recommend}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
