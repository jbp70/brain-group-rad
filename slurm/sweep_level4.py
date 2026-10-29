#!/usr/bin/env python3
"""Level-4 retrain sweep helper (retrain-prereqs design sections 7 and 10): list / status / submit
the runs of phase A (lr x wd), B (group sizes) and C (number of iterations) of slurm/r_level4.sh.

Runs on the MLSC login node: stdlib only, Python 3.6 syntax, one-shot (never a daemon, no SLURM
arrays). Account qtim is hard-coded; at most --max-active (1 or 2) of the user's jobs are active
(ALL lines of `squeue -h -u <user>` count, including Jay's own jobs).

  python3 slurm/sweep_level4.py list   [--phase A|B|C] [grid options]   # configs + exact sbatch lines
  python3 slurm/sweep_level4.py status [--phase A|B|C] [grid options]   # done / failed / running / pending / not started
  python3 slurm/sweep_level4.py submit [--phase A|B|C] [grid options] [--max-active 2] [--dry-run] [--a100-ok]

Configurations (7.1): phase A = lr x wd at (N, DEFORM_N), wd-major with wd 0 first and the
baseline lr 3e-4 first within each wd; phase B = (N_small, DEFORM_N) and (N, DEFORM_N_small) at
--best-lr / --best-wd; phase C = --c-steps (default 12000, 16000) at the A/B winner with the
final schedule. One global job index across phases (A 0-8, B 9-10, C 11-12 with the default grid),
job names r4sw<k>. The B / C indices start after len(--lrs) x len(--wds) of the CURRENT call, so
pass the same grid options (--lrs, --wds) in every phase and command; other values shift the B / C
names, which can then collide with a still-active phase-A job (jobs are matched by name). Every config passes VAL_GROUP_SIZE = min(16, DEFORM_N_small) (fixed for the
whole sweep; review v1.3 NIT 6) so the validation groups are identical in every run. Every config
also passes the retrain width d12hs explicitly (FILTERS=16,64,128,192,256,384
DECODER_WIDTHS=128,96,64,16,8, identical to the r_level*.sh defaults) so the sweep trains the
final width.

Status (evaluated in this order): running / pending = the job name is in squeue (RUNNING ->
running, else pending); done = the run dir exists and the last record of logs/metrics.jsonl that
holds val/mean_pair_rms_mm has step == config.json args.steps; failed = the run dir exists and the
run is not done, OR (review v1.3 NIT 4) there is a <results>/r4sw_submissions.jsonl record for the
run name but neither a squeue entry nor a run dir ("died before start"); not started = otherwise.
`submit` only submits configs that are not started; failed configs are printed and NEVER
resubmitted automatically (resume by hand with the same RUN_NAME, after Jay's decision).

Code revision: `submit` (not --dry-run) refuses a repo with uncommitted changes (exit 2) and
records `git rev-parse HEAD` per submission. A job runs the checkout present when it STARTS, so do
not `git pull` in the cluster clone while sweep jobs are pending. --results must be the results
root r_level4.sh writes to (its RESULTS default).

Exit codes: 0 ok; 1 squeue / sbatch / git missing or failing (nothing further submitted);
2 configuration error (bad grid value, phase B / C without --best-lr / --best-wd, --max-active
outside 1..2, unknown partition, dgx-a100 without --a100-ok, dirty repo).
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

PROJ = "/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg"
ACCOUNT = "qtim"
PARTITIONS = ("rtx8000", "dgx-a100")
A100_REMINDER = "dgx-a100 only after a smoke with INFO gpu_peak <= 36864 MB; pass --a100-ok to confirm"
SCRIPT = "slurm/r_level4.sh"
SUBMISSIONS = "r4sw_submissions.jsonl"
BASELINE_LR = 3e-4
MEM = "64G"
# Every env name slurm/r_level4.sh reads; dropped from the caller's environment before cfg_env is
# added, so a stale value in the login shell (e.g. EXTRA_ARGS) never leaks into a sweep job.
SCRIPT_ENV_KEYS = (
    "RUN_NAME", "STEPS", "VAL_EVERY", "CKPT_EVERY", "WARMUP", "LR", "WD", "OPTIMIZER", "N", "DEFORM_N",
    "TASK_RATIO", "DEFORM_START", "DEFORM_RAMP", "CURRICULUM", "CURRICULUM_RAMP", "G_START", "G_RAMP",
    "G_ROT", "G_TRANS", "TF_MEM_LIMIT_MB", "GRAD_CKPT", "PAIR_RECOMPUTE", "MIN_PAIRS", "MAX_PAIRS",
    "LEAN_LOSSES", "LOWRES_METHOD", "GRAD_BALANCE", "VAL_PAIR_CHUNK", "VAL_GROUP_SIZE", "DEFORM_DECODER",
    "INIT", "EXTRA_ARGS", "BGR_DRY_RUN", "PROJECT_STEPS", "PROJECT_VAL_EVERY",
    "FP16_ACT_CLIP", "FP16_ACT_CLIP_SCOPE", "DEFORM_DECODER_DTYPE", "BN_EPSILON_OVERRIDE",
    "FILTERS", "DECODER_WIDTHS",
    "PROJ", "REPO", "DATA", "RESULTS", "SIF", "SCRATCH_ROOT", "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK")
# The full env set of every submission (7.1), in this order.
CFG_KEYS = ("RUN_NAME", "LR", "WD", "N", "DEFORM_N", "STEPS", "VAL_EVERY", "WARMUP", "DEFORM_START",
            "DEFORM_RAMP", "CURRICULUM_RAMP", "G_START", "G_RAMP", "VAL_GROUP_SIZE",
            "FILTERS", "DECODER_WIDTHS")
# The retrain width d12hs, identical to the slurm/r_level*.sh defaults (FILTERS / DECODER_WIDTHS).
WIDTH_FILTERS = "16,64,128,192,256,384"
WIDTH_DECODER = "128,96,64,16,8"
# Schedules: phases A / B (8000 steps, scaled) and phase C (the final schedule).
SCHEDULE_AB = {"DEFORM_START": 2000, "DEFORM_RAMP": 500, "CURRICULUM_RAMP": 2000, "G_START": 2000,
               "G_RAMP": 2000, "WARMUP": 1000, "VAL_EVERY": 500}
SCHEDULE_C = {"DEFORM_START": 3000, "DEFORM_RAMP": 500, "CURRICULUM_RAMP": 3000, "G_START": 3000,
              "G_RAMP": 3000, "WARMUP": 1000, "VAL_EVERY": 500}
TIME_RE = re.compile(r"^(\d+-)?\d+:\d\d:\d\d$")
DONE_KEY = "val/mean_pair_rms_mm"


def fmt(x):
    """Compact float for run names: 0 -> '0', 3e-4 -> '3e-4', 2.5e-4 -> '2.5e-4' (design 7.1)."""
    x = float(x)
    if x == 0:
        return "0"
    s = ("%.0e" % x).replace("e-0", "e-").replace("e+0", "e")
    if float(s) != x:
        s = ("%.1e" % x).replace("e-0", "e-").replace("e+0", "e")
    return s


def env_float(x):
    """Env value of a float: fmt(x) when it round-trips exactly, else repr."""
    s = fmt(x)
    return s if float(s) == float(x) else repr(float(x))


# ----------------------------------------------------------------------------------- arguments

def _float_list(text, name, positive):
    out = []
    for part in text.split(","):
        try:
            v = float(part)
        except ValueError:
            raise ValueError("%s: %r is not a number" % (name, part))
        if not math.isfinite(v) or (v <= 0 if positive else v < 0):
            raise ValueError("%s: %r must be finite and %s" % (name, part, "> 0" if positive else ">= 0"))
        out.append(v)
    if not out:
        raise ValueError("%s is empty" % name)
    return out


def _int(text, name, minimum):
    try:
        v = int(text)
    except (TypeError, ValueError):
        raise ValueError("%s: %r is not an integer" % (name, text))
    if v < minimum:
        raise ValueError("%s must be >= %d, got %d" % (name, minimum, v))
    return v


def build_parser():
    p = argparse.ArgumentParser(prog="sweep_level4.py", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    grid = argparse.ArgumentParser(add_help=False)
    g = grid.add_argument_group("grid options")
    g.add_argument("--phase", default="A", choices=["A", "B", "C"])
    g.add_argument("--lrs", default="1e-4,3e-4,1e-3",
                   help="phase A learning rates (> 0); pass the same --lrs / --wds in every phase "
                   "(the B / C job indices start after the phase-A grid)")
    g.add_argument("--wds", default="0,1e-2,1e-1",
                   help="phase A AdamW weight decays (>= 0); same value in every phase (see --lrs)")
    g.add_argument("--n", default="128", help="N_max, linear group size")
    g.add_argument("--deform-n", default="64", help="DEFORM_N_max, deform group size")
    g.add_argument("--n-small", default=None, help="phase B N_small (default N_max / 2)")
    g.add_argument("--deform-n-small", default=None, help="phase B DEFORM_N_small (default DEFORM_N_max / 2)")
    g.add_argument("--steps", default="8000", help="phases A / B run length")
    g.add_argument("--time", default="14:00:00", help="phases A / B --time")
    g.add_argument("--c-steps", default="12000,16000", help="phase C run lengths (strictly increasing)")
    g.add_argument("--c-time", default="24:00:00", help="phase C --time: derive from the phase-A INFO step_time")
    g.add_argument("--val-group-size", default=None,
                   help="VAL_GROUP_SIZE of every run (default min(16, DEFORM_N_small))")
    g.add_argument("--partition", default="rtx8000", help="rtx8000 or dgx-a100")
    g.add_argument("--best-lr", default=None, help="phase-A winner lr (phases B, C: required)")
    g.add_argument("--best-wd", default=None, help="phase-A winner wd (phases B, C: required)")
    g.add_argument("--best-n", default=None, help="phase C N (default --n)")
    g.add_argument("--best-deform-n", default=None, help="phase C DEFORM_N (default --deform-n)")
    g.add_argument("--results", default=PROJ + "/results", help="results root (r_level4.sh's RESULTS)")
    g.add_argument("--repo", default=PROJ + "/brain-group-rad", help="repo clone (sbatch cwd)")
    g.add_argument("--user", default="jn85")
    sub.add_parser("list", parents=[grid], help="configs + exact sbatch lines")
    sub.add_parser("status", parents=[grid], help="per config: done / failed / running / pending / not started")
    s = sub.add_parser("submit", parents=[grid], help="submit not-started configs up to --max-active")
    s.add_argument("--max-active", type=int, default=2, help="1 or 2 (cluster rule: at most two active jobs)")
    s.add_argument("--dry-run", action="store_true", help="print the commands only (no git check, no sbatch)")
    s.add_argument("--a100-ok", action="store_true", help="confirm dgx-a100 (" + A100_REMINDER + ")")
    return p


def resolve(args):
    """Validated grid dict from the parsed args; ValueError on a bad value."""
    g = {"phase": args.phase}
    g["lrs"] = _float_list(args.lrs, "--lrs", True)
    g["wds"] = _float_list(args.wds, "--wds", False)
    g["n"] = _int(args.n, "--n", 2)
    g["deform_n"] = _int(args.deform_n, "--deform-n", 2)
    g["n_small"] = _int(args.n_small if args.n_small is not None else g["n"] // 2, "--n-small", 2)
    g["deform_n_small"] = _int(args.deform_n_small if args.deform_n_small is not None else g["deform_n"] // 2,
                               "--deform-n-small", 2)
    g["steps"] = _int(args.steps, "--steps", 1)
    c_steps = [_int(x, "--c-steps", 1) for x in args.c_steps.split(",")]
    if any(b <= a for a, b in zip(c_steps, c_steps[1:])):
        raise ValueError("--c-steps must be strictly increasing, got %s" % args.c_steps)
    g["c_steps"] = c_steps
    for name, val in (("--time", args.time), ("--c-time", args.c_time)):
        if not TIME_RE.match(val):
            raise ValueError("%s must look like HH:MM:SS or D-HH:MM:SS, got %r" % (name, val))
    g["time"], g["c_time"] = args.time, args.c_time
    vgs = args.val_group_size
    g["val_group_size"] = _int(vgs if vgs is not None else min(16, g["deform_n_small"]), "--val-group-size", 2)
    if args.partition not in PARTITIONS:
        raise ValueError("--partition must be one of %s, got %r" % (", ".join(PARTITIONS), args.partition))
    g["partition"] = args.partition
    if args.phase in ("B", "C"):
        if args.best_lr is None or args.best_wd is None:
            raise ValueError("phase %s needs --best-lr and --best-wd (the phase-A winner)" % args.phase)
        g["best_lr"] = _float_list(args.best_lr, "--best-lr", True)[0]
        g["best_wd"] = _float_list(args.best_wd, "--best-wd", False)[0]
    g["best_n"] = _int(args.best_n if args.best_n is not None else g["n"], "--best-n", 2)
    g["best_deform_n"] = _int(args.best_deform_n if args.best_deform_n is not None else g["deform_n"],
                              "--best-deform-n", 2)
    return g


# ------------------------------------------------------------------------------------ configs

def _config(k, phase, run, lr, wd, n, dn, steps, sched, time_limit, val_group_size):
    env = {"RUN_NAME": run, "LR": env_float(lr), "WD": env_float(wd), "N": str(n), "DEFORM_N": str(dn),
           "STEPS": str(steps), "VAL_GROUP_SIZE": str(val_group_size),
           "FILTERS": WIDTH_FILTERS, "DECODER_WIDTHS": WIDTH_DECODER}
    for key, val in sched.items():
        env[key] = str(val)
    return {"k": k, "phase": phase, "job": "r4sw%d" % k, "run": run, "time": time_limit, "env": env}


def configs(g, phase=None):
    """Configs of `phase` (default g['phase']) in sweep order, with the global job index."""
    phase = phase or g["phase"]
    n_a = len(g["lrs"]) * len(g["wds"])
    vgs = g["val_group_size"]
    out = []
    if phase == "A":
        lrs = sorted(g["lrs"], key=lambda x: x != BASELINE_LR)       # stable: baseline lr first
        wds = sorted(g["wds"], key=lambda x: x != 0)                 # stable: wd 0 first
        k = 0
        for wd in wds:
            for lr in lrs:
                run = "r4sw_A_lr%s_wd%s_n%d_d%d" % (fmt(lr), fmt(wd), g["n"], g["deform_n"])
                out.append(_config(k, "A", run, lr, wd, g["n"], g["deform_n"], g["steps"], SCHEDULE_AB,
                                   g["time"], vgs))
                k += 1
    elif phase == "B":
        lr, wd = g["best_lr"], g["best_wd"]
        for i, (n, dn) in enumerate(((g["n_small"], g["deform_n"]), (g["n"], g["deform_n_small"]))):
            run = "r4sw_B_lr%s_wd%s_n%d_d%d" % (fmt(lr), fmt(wd), n, dn)
            out.append(_config(n_a + i, "B", run, lr, wd, n, dn, g["steps"], SCHEDULE_AB, g["time"], vgs))
    else:
        lr, wd, n, dn = g["best_lr"], g["best_wd"], g["best_n"], g["best_deform_n"]
        for i, steps in enumerate(g["c_steps"]):
            run = "r4sw_C_s%d_lr%s_wd%s_n%d_d%d" % (steps, fmt(lr), fmt(wd), n, dn)
            out.append(_config(n_a + 2 + i, "C", run, lr, wd, n, dn, steps, SCHEDULE_C, g["c_time"], vgs))
    return out


def sbatch_argv(cfg, partition):
    return ["sbatch", "--parsable", "-A", ACCOUNT, "-p", partition, "--job-name", cfg["job"],
            "--time", cfg["time"], "--mem", MEM, SCRIPT]


def env_prefix(cfg):
    return " ".join("%s=%s" % (k, cfg["env"][k]) for k in CFG_KEYS)


def job_env(cfg_env):
    """The caller's environment without SCRIPT_ENV_KEYS, plus cfg_env (all strings)."""
    env = {k: v for k, v in os.environ.items() if k not in SCRIPT_ENV_KEYS}
    env.update({k: str(v) for k, v in cfg_env.items()})
    return env


# ------------------------------------------------------------------------------------- status

def _run(argv, what):
    """check_output decoded; RuntimeError naming `what` when missing or non-zero."""
    try:
        return subprocess.check_output(argv, stderr=subprocess.PIPE).decode("utf-8")
    except OSError as e:
        raise RuntimeError("%s: cannot run %s (%s)" % (what, argv[0], e))
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError("%s failed with exit code %d: %s" % (what, e.returncode, err))


def squeue(user):
    """(number of the user's active jobs, {job name: state}) from `squeue -h -u <user> -o '%j %T'`."""
    text = _run(["squeue", "-h", "-u", user, "-o", "%j %T"], "squeue")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    states = {}
    for ln in lines:
        parts = ln.split()
        states[parts[0]] = parts[1] if len(parts) > 1 else "?"
    return len(lines), states


def submitted_runs(results):
    """Run names with a record in <results>/r4sw_submissions.jsonl (malformed lines skipped)."""
    path = os.path.join(results, SUBMISSIONS)
    runs = set()
    if not os.path.isfile(path):
        return runs
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and rec.get("run"):
                runs.add(rec["run"])
    return runs


def run_progress(run_dir):
    """(done, last logged step or None) of a run dir (see the module docstring)."""
    steps = None
    try:
        with open(os.path.join(run_dir, "config.json")) as f:
            steps = (json.load(f).get("args") or {}).get("steps")
    except (OSError, ValueError):
        pass
    last, last_done_key = None, None
    path = os.path.join(run_dir, "logs", "metrics.jsonl")
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not (isinstance(rec, dict) and isinstance(rec.get("data"), dict)):
                    continue
                step = rec.get("step")
                if isinstance(step, int):
                    last = step if last is None else max(last, step)
                    if DONE_KEY in rec["data"]:
                        last_done_key = step
    done = steps is not None and last_done_key is not None and last_done_key == steps
    return done, last


def state_of(cfg, results, queue, submitted):
    """(state, detail) with state in running / pending / done / failed / not started."""
    if cfg["job"] in queue:
        return ("running" if queue[cfg["job"]] == "RUNNING" else "pending"), None
    run_dir = os.path.join(results, cfg["run"])
    if os.path.isdir(run_dir):
        done, last = run_progress(run_dir)
        return ("done" if done else "failed"), (None if last is None else "last step %d" % last)
    if cfg["run"] in submitted:
        return "failed", "died before start"
    return "not started", None


# ----------------------------------------------------------------------------------- commands

def cmd_list(g, out):
    if g["partition"] == "dgx-a100":
        out.write("NOTE: %s\n" % A100_REMINDER)
    for cfg in configs(g):
        argv = [a for a in sbatch_argv(cfg, g["partition"]) if a != "--parsable"]
        out.write("%d %s %s\n" % (cfg["k"], cfg["job"], cfg["run"]))
        out.write("  %s %s\n" % (env_prefix(cfg), " ".join(argv)))
    return 0


def cmd_status(g, user, out):
    _, queue = squeue(user)
    submitted = submitted_runs(g["results"])
    for cfg in configs(g):
        state, detail = state_of(cfg, g["results"], queue, submitted)
        out.write("%d %s %s %s%s\n" % (cfg["k"], cfg["job"], cfg["run"], state,
                                       "" if detail is None else " (%s)" % detail))
    return 0


def git_rev(repo):
    """HEAD sha of a clean repo; ValueError if it has uncommitted changes, RuntimeError on git errors."""
    status = _run(["git", "-C", repo, "status", "--porcelain"], "git status")
    if status.strip():
        raise ValueError("repo has uncommitted changes; commit/pull first")
    return _run(["git", "-C", repo, "rev-parse", "HEAD"], "git rev-parse").strip()


def cmd_submit(g, args, out, err):
    if args.max_active not in (1, 2):
        err.write("ERROR: --max-active must be 1 or 2 (cluster rule: at most two active jobs), got %d\n"
                  % args.max_active)
        return 2
    if g["partition"] == "dgx-a100":
        if not args.dry_run and not args.a100_ok:
            err.write("ERROR: %s\n" % A100_REMINDER)
            return 2
        out.write("NOTE: %s\n" % A100_REMINDER)
    rev = None
    if not args.dry_run:
        try:
            rev = git_rev(g["repo"])
        except ValueError as e:
            err.write("ERROR: %s\n" % e)
            return 2
    active, queue = squeue(args.user)
    submitted = submitted_runs(g["results"])
    todo = []
    for cfg in configs(g):
        state, detail = state_of(cfg, g["results"], queue, submitted)
        if state == "failed":
            out.write("FAILED, not resubmitted: %s%s\n" % (cfg["run"], "" if detail is None else " (%s)" % detail))
        elif state == "not started":
            todo.append(cfg)
    free = args.max_active - active
    if not todo:
        out.write("nothing to submit (%d active)\n" % active)
        return 0
    if free <= 0:
        out.write("NOT submitted: %d active\n" % active)
        return 0
    for cfg in todo[:free]:
        argv = sbatch_argv(cfg, g["partition"])
        line = "%s %s" % (env_prefix(cfg), " ".join(argv))
        if args.dry_run:
            out.write("DRY RUN: %s\n" % line)
            continue
        try:
            text = subprocess.check_output(argv, env=job_env(cfg["env"]), cwd=g["repo"],
                                           stderr=subprocess.PIPE).decode("utf-8")
        except OSError as e:
            raise RuntimeError("sbatch: cannot run (%s)" % e)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("sbatch failed with exit code %d for %s: %s" % (
                e.returncode, cfg["run"], (e.stderr or b"").decode("utf-8", "replace").strip()))
        job_id = text.strip().split(";")[0]
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "job_id": job_id, "run": cfg["run"], "rev": rev,
               "env": dict(cfg["env"])}
        if not os.path.isdir(g["results"]):
            os.makedirs(g["results"])
        with open(os.path.join(g["results"], SUBMISSIONS), "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        out.write("SUBMITTED job %s: %s\n" % (job_id, line))
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.error("choose a command: list, status or submit")
    try:
        g = resolve(args)
    except ValueError as e:
        parser.error(str(e))
    g["results"], g["repo"] = args.results, args.repo
    try:
        if args.cmd == "list":
            return cmd_list(g, sys.stdout)
        if args.cmd == "status":
            return cmd_status(g, args.user, sys.stdout)
        return cmd_submit(g, args, sys.stdout, sys.stderr)
    except RuntimeError as e:
        sys.stderr.write("ERROR: %s\n" % e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
