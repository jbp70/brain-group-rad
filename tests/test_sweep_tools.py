"""Retrain-prereqs FEAT-005 (design sections 6, 7, 8, tests 11.5 a-h): the level-4 sweep helper
slurm/sweep_level4.py, the run summarizer scripts/summarize_runs.py and the from-scratch SLURM
scripts slurm/r_level{4,2,1}.sh (static header checks, `bash -n`, BGR_DRY_RUN=1 argument lines,
one stub run). Stdlib only, no TensorFlow (runtime.args is imported lazily; it is TF-free). The
fake squeue / sbatch / git / singularity scripts need a POSIX shell; those cases are skipped
without bash.
"""
import ast
import contextlib
import csv
import importlib.util
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(REPO, "slurm", "sweep_level4.py")
SUMMARIZE = os.path.join(REPO, "scripts", "summarize_runs.py")
R4 = os.path.join(REPO, "slurm", "r_level4.sh")
R2 = os.path.join(REPO, "slurm", "r_level2.sh")
R1 = os.path.join(REPO, "slurm", "r_level1.sh")
SCRIPTS = (R4, R2, R1)
LOGS = "/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs"
SIF = "/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif"
BASH = shutil.which("bash")
POSIX = os.name == "posix" and BASH is not None
FAKE_SHA = "0123456789abcdef0123456789abcdef01234567"
PY37_TOKENS = ("capture_output", "text=True", "import dataclasses", "from dataclasses", "fromisoformat")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sweep = _load(SWEEP, "sweep_level4")
summ = _load(SUMMARIZE, "summarize_runs")
# Env names of the level-2 / level-1 scripts that the level-4 key list does not cover.
CASCADE_ENV = ("CASCADE_FROM", "BN_RECAL_BATCHES", "BN_RECAL_FREEZE", "BN_RECAL_VALIDATE", "SKIP_LINEAR",
               "SKIP_DEFORM", "VAL_IMAGE_CHUNK", "TMPDIR")
STRIP_ENV = tuple(sweep.SCRIPT_ENV_KEYS) + CASCADE_ENV


def _read(path):
    with open(path) as f:
        return f.read()


def _has_seq(args, seq):
    n = len(seq)
    return any(args[i:i + n] == seq for i in range(len(args) - n + 1))


def _grid(*argv):
    return sweep.resolve(sweep.build_parser().parse_args(["list"] + list(argv)))


def _write_exec(path, body):
    with open(path, "w") as f:
        f.write(body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


# ------------------------------------------------------------------------- synthetic run trees

def write_run(root, name, args, records, bad_lines=0, metrics=True):
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "logs"))
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"args": args}, f)
    if metrics:
        with open(os.path.join(d, "logs", "metrics.jsonl"), "w") as f:
            for i, (step, data) in enumerate(records):
                f.write(json.dumps({"step": step, "time": 1000.0 + 1800.0 * i, "data": data}) + "\n")
                if i == 0:
                    for _ in range(bad_lines):
                        f.write("{not json\n")
    return d


def train_rec(step, nonfinite=0, collapse=0, peak=20480.0):
    return (step, {"loss": 1.0, "nonfinite_total": nonfinite, "train/collapse_suspect": collapse,
                   "gpu_peak_mb": peak})


def val_rec(step, l_val, dp, neg=0.01, affg=6.0):
    return (step, {"val/mean_pair_rms_mm": l_val, "val_deform/pair_dice": dp, "val_deform/template_dice": dp + 0.05,
                   "val_deform/neg_jac_pct": neg, "val_deform/ncc_template": 0.8, "val_deform/ncc_pair": 0.7,
                   "val_deform/inv_consistency_mm": 0.1, "val_affine_g/pair_rms_mm": affg,
                   "val_affine_g/center_rms_mm": affg / 2, "val_rigid/center_rms_mm": l_val / 2,
                   "val_rigid/pair_rms_mm": l_val})


def run_args(steps=1000, vgs=16, lr=3e-4, wd=0.0, n=64, dn=32):
    return {"lr": lr, "weight_decay": wd, "optimizer": "adamw", "group_size": n, "deform_group_size": dn,
            "steps": steps, "val_group_size": vgs}


def done_records(steps, vals, **train_kw):
    """Train windows + validations at the given steps: vals = [(step, L, Dp, neg)]."""
    recs = []
    for s, l_val, dp, neg in vals:
        recs.append(train_rec(s - 25, **train_kw))
        recs.append(val_rec(s, l_val, dp, neg))
    return recs


# --------------------------------------------------------------------------------- 11.5 a

class TestSweepConfigs(unittest.TestCase):

    def test_fmt(self):
        for x, want in ((0, "0"), (0.0, "0"), (1e-4, "1e-4"), (3e-4, "3e-4"), (1e-3, "1e-3"), (1e-2, "1e-2"),
                        (1e-1, "1e-1"), (2.5e-4, "2.5e-4")):
            self.assertEqual(sweep.fmt(x), want, x)

    def test_phase_a(self):
        cfgs = sweep.configs(_grid())
        self.assertEqual([c["k"] for c in cfgs], list(range(9)))
        self.assertEqual([c["job"] for c in cfgs], ["r4sw%d" % k for k in range(9)])
        want = ["r4sw_A_lr%s_wd%s_n128_d64" % (lr, wd) for wd in ("0", "1e-2", "1e-1")
                for lr in ("3e-4", "1e-4", "1e-3")]
        self.assertEqual([c["run"] for c in cfgs], want)
        for c in cfgs:
            env = c["env"]
            self.assertEqual(set(env), set(sweep.CFG_KEYS))
            self.assertEqual(env["RUN_NAME"], c["run"])
            for k, v in (("STEPS", "8000"), ("DEFORM_START", "2000"), ("DEFORM_RAMP", "500"),
                         ("CURRICULUM_RAMP", "2000"), ("G_START", "2000"), ("G_RAMP", "2000"), ("WARMUP", "1000"),
                         ("VAL_EVERY", "500"), ("VAL_GROUP_SIZE", "16"), ("N", "128"), ("DEFORM_N", "64")):
                self.assertEqual(env[k], v, k)
            self.assertEqual(c["time"], "14:00:00")
            self.assertTrue(all(isinstance(v, str) for v in env.values()))
        self.assertEqual((cfgs[0]["env"]["LR"], cfgs[0]["env"]["WD"]), ("3e-4", "0"))
        self.assertEqual(float(cfgs[4]["env"]["LR"]), 1e-4)
        self.assertEqual(float(cfgs[4]["env"]["WD"]), 1e-2)

    def test_phase_b_and_c(self):
        b = sweep.configs(_grid("--phase", "B", "--best-lr", "3e-4", "--best-wd", "1e-2"))
        self.assertEqual([c["k"] for c in b], [9, 10])
        self.assertEqual([c["job"] for c in b], ["r4sw9", "r4sw10"])
        self.assertEqual([c["run"] for c in b], ["r4sw_B_lr3e-4_wd1e-2_n64_d64", "r4sw_B_lr3e-4_wd1e-2_n128_d32"])
        self.assertEqual([(c["env"]["N"], c["env"]["DEFORM_N"]) for c in b], [("64", "64"), ("128", "32")])
        self.assertTrue(all(c["env"]["STEPS"] == "8000" and c["env"]["DEFORM_START"] == "2000" for c in b))
        c = sweep.configs(_grid("--phase", "C", "--best-lr", "1e-4", "--best-wd", "0", "--best-n", "32"))
        self.assertEqual([x["k"] for x in c], [11, 12])
        self.assertEqual([x["run"] for x in c], ["r4sw_C_s12000_lr1e-4_wd0_n32_d64", "r4sw_C_s16000_lr1e-4_wd0_n32_d64"])
        for x, steps in zip(c, ("12000", "16000")):
            env = x["env"]
            self.assertEqual(set(env), set(sweep.CFG_KEYS))
            for k, v in (("STEPS", steps), ("DEFORM_START", "3000"), ("DEFORM_RAMP", "500"),
                         ("CURRICULUM_RAMP", "3000"), ("G_START", "3000"), ("G_RAMP", "3000"), ("WARMUP", "1000"),
                         ("VAL_GROUP_SIZE", "16")):
                self.assertEqual(env[k], v, k)
            self.assertEqual(x["time"], "24:00:00")

    def test_val_group_size_follows_deform_n_small(self):
        """Review v1.3 NIT 6: VAL_GROUP_SIZE = min(16, DEFORM_N_small), the same in every phase."""
        for phase_args in ((), ("--phase", "B", "--best-lr", "3e-4", "--best-wd", "0"),
                           ("--phase", "C", "--best-lr", "3e-4", "--best-wd", "0")):
            g = _grid("--deform-n", "24", *phase_args)
            self.assertEqual(g["deform_n_small"], 12)
            self.assertEqual({c["env"]["VAL_GROUP_SIZE"] for c in sweep.configs(g)}, {"12"})
            g = _grid(*phase_args)
            self.assertEqual({c["env"]["VAL_GROUP_SIZE"] for c in sweep.configs(g)}, {"16"})

    def _main_exit(self, argv):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                sweep.main(argv)
        return cm.exception.code, err.getvalue()

    def test_errors(self):
        for argv in (["list", "--phase", "B"], ["list", "--phase", "C", "--best-lr", "3e-4"],
                     ["list", "--lrs", "0,1e-4"], ["list", "--wds", "-1"], ["list", "--n", "x"],
                     ["list", "--c-steps", "16000,12000"], ["list", "--partition", "pubgpu"],
                     ["list", "--time", "14h"], ["list", "--val-group-size", "1"], []):
            code, err = self._main_exit(argv)
            self.assertEqual(code, 2, (argv, err))
        code, err = self._main_exit(["list", "--phase", "B"])
        self.assertIn("--best-lr", err)

    def test_list_output(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(sweep.main(["list"]), 0)
        text = out.getvalue()
        self.assertIn("0 r4sw0 r4sw_A_lr3e-4_wd0_n128_d64\n", text)
        self.assertEqual(text.count("sbatch -A qtim -p rtx8000 --job-name r4sw"), 9)
        self.assertIn("RUN_NAME=r4sw_A_lr3e-4_wd0_n128_d64 LR=3e-4 WD=0 N=128 DEFORM_N=64 STEPS=8000", text)
        self.assertNotIn("entre", text)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(sweep.main(["list", "--partition", "dgx-a100"]), 0)
        self.assertIn("NOTE: dgx-a100 only after a smoke with INFO gpu_peak <= 36864 MB", out.getvalue())

    def test_job_env(self):
        old = dict(os.environ)
        try:
            os.environ.update(EXTRA_ARGS="x", TASK_RATIO="deform:1", BGR_SWEEP_TEST_KEEP="1")
            env = sweep.job_env({"RUN_NAME": "r", "LR": "3e-4"})
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertNotIn("EXTRA_ARGS", env)
        self.assertNotIn("TASK_RATIO", env)
        self.assertEqual(env["BGR_SWEEP_TEST_KEEP"], "1")
        self.assertEqual((env["RUN_NAME"], env["LR"]), ("r", "3e-4"))

    def test_script_env_keys_cover_r_level4(self):
        """11.5 g: SCRIPT_ENV_KEYS is a superset of every ${NAME:- default in r_level4.sh."""
        names = set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*):-", _read(R4)))
        self.assertIn("EXTRA_ARGS", names)
        self.assertEqual(names - set(sweep.SCRIPT_ENV_KEYS), set())
        self.assertTrue(set(sweep.CFG_KEYS) <= set(sweep.SCRIPT_ENV_KEYS))


# ------------------------------------------------------------------------ 11.5 b, c, g

@unittest.skipUnless(POSIX, "needs a POSIX shell for the fake squeue / sbatch / git")
class TestSweepCluster(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sweep_test_")
        self.bin = os.path.join(self.tmp, "bin")
        self.sb = os.path.join(self.tmp, "sbatch_calls")
        self.results = os.path.join(self.tmp, "results")
        self.repo = os.path.join(self.tmp, "repo")
        for d in (self.bin, self.sb, self.results, self.repo):
            os.makedirs(d)
        self.squeue_file = os.path.join(self.tmp, "squeue.txt")
        self.git_status = os.path.join(self.tmp, "git_status.txt")
        self.set_queue([])
        with open(self.git_status, "w"):
            pass
        _write_exec(os.path.join(self.bin, "squeue"),
                    '#!/bin/sh\n[ -n "$FAKE_SQUEUE_FAIL" ] && { echo boom >&2; exit 3; }\n'
                    'echo "$*" > "$FAKE_SQUEUE_ARGS"\ncat "$FAKE_SQUEUE"\n')
        _write_exec(os.path.join(self.bin, "sbatch"),
                    '#!/bin/bash\n[ -n "$FAKE_SBATCH_FAIL" ] && { echo "sbatch: error" >&2; exit 1; }\n'
                    'd="$FAKE_SBATCH_DIR"\nn=$(cat "$d/count" 2>/dev/null || echo 0)\nn=$((n + 1))\n'
                    'echo "$n" > "$d/count"\nprintf "%s\\n" "$@" > "$d/args_$n"\nenv > "$d/env_$n"\n'
                    'pwd > "$d/cwd_$n"\necho "$((1000 + n));cluster"\n')
        _write_exec(os.path.join(self.bin, "git"),
                    '#!/bin/sh\ncase "$*" in\n*"status --porcelain"*) cat "$FAKE_GIT_STATUS" ;;\n'
                    '*"rev-parse HEAD"*) echo %s ;;\n*) exit 1 ;;\nesac\n' % FAKE_SHA)
        env = {k: v for k, v in os.environ.items() if k not in STRIP_ENV}
        env.update(PATH=self.bin + os.pathsep + os.environ.get("PATH", ""), FAKE_SQUEUE=self.squeue_file,
                   FAKE_SQUEUE_ARGS=os.path.join(self.tmp, "squeue_args.txt"), FAKE_SBATCH_DIR=self.sb,
                   FAKE_GIT_STATUS=self.git_status, PYTHONDONTWRITEBYTECODE="1")
        self.env = env
        self.cfgs = sweep.configs(_grid())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def set_queue(self, lines):
        with open(self.squeue_file, "w") as f:
            f.write("".join(ln + "\n" for ln in lines))

    def run_sweep(self, *argv, **env_extra):
        env = dict(self.env)
        env.update(env_extra)
        r = subprocess.run([sys.executable, SWEEP] + list(argv) + ["--results", self.results, "--repo", self.repo],
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                           stdin=subprocess.DEVNULL)
        return r

    def sbatch_calls(self):
        n = int(_read(os.path.join(self.sb, "count"))) if os.path.exists(os.path.join(self.sb, "count")) else 0
        calls = []
        for i in range(1, n + 1):
            args = _read(os.path.join(self.sb, "args_%d" % i)).splitlines()
            env = dict(ln.split("=", 1) for ln in _read(os.path.join(self.sb, "env_%d" % i)).splitlines() if "=" in ln)
            cwd = _read(os.path.join(self.sb, "cwd_%d" % i)).strip()
            calls.append((args, env, cwd))
        return calls

    def submissions(self):
        path = os.path.join(self.results, sweep.SUBMISSIONS)
        if not os.path.exists(path):
            return []
        return [json.loads(ln) for ln in _read(path).splitlines() if ln.strip()]

    def make_done(self, cfg):
        write_run(self.results, cfg["run"], run_args(steps=8000),
                  done_records(8000, [(7500, 4.1, 0.75, 0.01), (8000, 4.0, 0.76, 0.01)]))

    def make_failed(self, cfg):
        write_run(self.results, cfg["run"], run_args(steps=8000),
                  done_records(8000, [(500, 5.0, 0.70, 0.01), (1000, 4.8, 0.71, 0.01)]))

    def test_two_active_nothing_submitted(self):
        self.set_queue(["other RUNNING", "jays_job PENDING"])
        r = self.run_sweep("submit")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("NOT submitted: 2 active", r.stdout)
        self.assertEqual(self.sbatch_calls(), [])
        self.assertEqual(self.submissions(), [])
        self.assertEqual(_read(os.path.join(self.tmp, "squeue_args.txt")).split(), ["-h", "-u", "jn85", "-o", "%j", "%T"])

    def test_one_active_submits_one(self):
        self.set_queue(["other RUNNING"])
        r = self.run_sweep("submit")
        self.assertEqual(r.returncode, 0, r.stderr)
        calls = self.sbatch_calls()
        self.assertEqual(len(calls), 1)
        args, env, cwd = calls[0]
        self.assertEqual(args, ["--parsable", "-A", "qtim", "-p", "rtx8000", "--job-name", "r4sw0", "--time",
                                "14:00:00", "--mem", "64G", "slurm/r_level4.sh"])
        self.assertNotIn("entre", " ".join(args))
        self.assertEqual(os.path.realpath(cwd), os.path.realpath(self.repo))
        for k, v in self.cfgs[0]["env"].items():
            self.assertEqual(env.get(k), v, k)
        subs = self.submissions()
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]["job_id"], "1001")
        self.assertEqual(subs[0]["run"], self.cfgs[0]["run"])
        self.assertEqual(subs[0]["rev"], FAKE_SHA)
        self.assertEqual(subs[0]["env"], self.cfgs[0]["env"])
        self.assertIn("time", subs[0])
        self.assertIn("SUBMITTED job 1001", r.stdout)

    def test_existing_and_failed_runs_skipped(self):
        self.make_done(self.cfgs[0])
        self.make_failed(self.cfgs[1])
        r = self.run_sweep("submit", "--max-active", "2")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("FAILED, not resubmitted: %s" % self.cfgs[1]["run"], r.stdout)
        jobs = [c[0][6] for c in self.sbatch_calls()]
        self.assertEqual(jobs, ["r4sw2", "r4sw3"])

    def test_died_before_start_not_resubmitted(self):
        """Review v1.3 NIT 4: a submissions record, no run dir, not in squeue -> failed."""
        with open(os.path.join(self.results, sweep.SUBMISSIONS), "w") as f:
            f.write(json.dumps({"time": "t", "job_id": "7", "run": self.cfgs[0]["run"], "rev": FAKE_SHA,
                                "env": self.cfgs[0]["env"]}) + "\n")
            f.write("{malformed\n")
        r = self.run_sweep("status")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("0 r4sw0 %s failed (died before start)\n" % self.cfgs[0]["run"], r.stdout)
        r = self.run_sweep("submit", "--max-active", "1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("FAILED, not resubmitted: %s (died before start)" % self.cfgs[0]["run"], r.stdout)
        self.assertEqual([c[0][6] for c in self.sbatch_calls()], ["r4sw1"])

    def test_status_states(self):
        self.set_queue(["r4sw0 RUNNING", "r4sw1 PENDING", "unrelated RUNNING"])
        self.make_done(self.cfgs[2])
        self.make_failed(self.cfgs[3])
        with open(os.path.join(self.results, sweep.SUBMISSIONS), "w") as f:
            f.write(json.dumps({"run": self.cfgs[4]["run"], "job_id": "9"}) + "\n")
        r = self.run_sweep("status")
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.splitlines()
        self.assertEqual(len(lines), 9)
        want = ["running", "pending", "done (last step 8000)", "failed (last step 1000)",
                "failed (died before start)"] + ["not started"] * 4
        for i, (ln, w) in enumerate(zip(lines, want)):
            self.assertEqual(ln, "%d r4sw%d %s %s" % (i, i, self.cfgs[i]["run"], w))

    def test_config_errors(self):
        r = self.run_sweep("submit", "--max-active", "3")
        self.assertEqual(r.returncode, 2, r.stderr)
        r = self.run_sweep("submit", "--partition", "dgx-a100")
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("--a100-ok", r.stderr)
        self.assertEqual(self.sbatch_calls(), [])
        r = self.run_sweep("submit", "--partition", "dgx-a100", "--a100-ok", "--max-active", "1")
        self.assertEqual(r.returncode, 0, r.stderr)
        args = self.sbatch_calls()[0][0]
        self.assertTrue(_has_seq(args, ["-A", "qtim", "-p", "dgx-a100"]))

    def test_env_isolation_and_rev(self):
        """11.5 g: caller env EXTRA_ARGS / TASK_RATIO never reach the job; cfg_env does."""
        r = self.run_sweep("submit", "--max-active", "1", EXTRA_ARGS="x", TASK_RATIO="deform:1",
                           BGR_SWEEP_TEST_KEEP="kept")
        self.assertEqual(r.returncode, 0, r.stderr)
        _, env, _ = self.sbatch_calls()[0]
        self.assertNotIn("EXTRA_ARGS", env)
        self.assertNotIn("TASK_RATIO", env)
        self.assertEqual(env.get("BGR_SWEEP_TEST_KEEP"), "kept")
        for k in sweep.CFG_KEYS:
            self.assertEqual(env[k], self.cfgs[0]["env"][k], k)
        self.assertEqual(self.submissions()[0]["rev"], FAKE_SHA)

    def test_dirty_repo_refused(self):
        with open(self.git_status, "w") as f:
            f.write(" M x.py\n")
        r = self.run_sweep("submit")
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("ERROR: repo has uncommitted changes; commit/pull first", r.stderr)
        self.assertEqual(self.sbatch_calls(), [])
        # --dry-run: no git check, commands printed only
        r = self.run_sweep("submit", "--dry-run")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.count("DRY RUN: "), 2)
        self.assertIn("sbatch --parsable -A qtim -p rtx8000 --job-name r4sw0", r.stdout)
        self.assertEqual(self.sbatch_calls(), [])
        self.assertEqual(self.submissions(), [])

    def test_cluster_errors_exit_1(self):
        r = self.run_sweep("submit", FAKE_SQUEUE_FAIL="1")
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("squeue failed", r.stderr)
        r = self.run_sweep("submit", FAKE_SBATCH_FAIL="1")
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("sbatch failed", r.stderr)
        self.assertEqual(self.submissions(), [])
        env = dict(self.env)
        env["PATH"] = self.bin                     # git found, but squeue missing -> exit 1
        os.remove(os.path.join(self.bin, "squeue"))
        r = subprocess.run([sys.executable, SWEEP, "status", "--results", self.results], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(r.returncode, 1, r.stderr)


# --------------------------------------------------------------------------------- 11.5 d

class TestSummarize(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="summ_test_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def run_cli(self, *argv):
        return subprocess.run([sys.executable, SUMMARIZE] + list(argv), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True)

    def tree(self):
        write_run(self.root, "r4sw_A_a", run_args(),
                  done_records(1000, [(500, 5.0, 0.70, 0.01), (1000, 4.0, 0.75, 0.02)]), bad_lines=1)
        write_run(self.root, "r4sw_A_b", run_args(),
                  done_records(1000, [(500, 3.0, 0.80, 0.2), (1000, 3.0, 0.80, 0.8)]))      # folding
        write_run(self.root, "r4sw_A_c", run_args(),
                  done_records(1000, [(500, 4.5, 0.74, 0.01), (1000, 4.2, 0.76, 0.01)]))
        recs = done_records(1000, [(500, 2.0, 0.9, 0.0), (1000, 2.0, 0.9, 0.0)], nonfinite=100)
        write_run(self.root, "r4sw_A_d", run_args(), recs)                                  # 100/1000 skipped
        write_run(self.root, "r4sw_A_e", run_args(), [], metrics=False)
        write_run(self.root, "other_run", run_args(), done_records(1000, [(1000, 1.0, 0.9, 0.0)]))

    def test_summarize_run(self):
        self.tree()
        row = summ.summarize_run(os.path.join(self.root, "r4sw_A_a"), "r4sw_A_a")
        self.assertEqual(row["_bad_lines"], 1)
        self.assertAlmostEqual(row["L"], 4.5)
        self.assertAlmostEqual(row["Dp"], 0.725)
        self.assertEqual((row["mean_pair"], row["mean_pair.best"], row["mean_pair.at"]), (4.0, 4.0, 1000))
        self.assertEqual((row["pair_dice"], row["pair_dice.best"], row["pair_dice.at"]), (0.75, 0.75, 1000))
        self.assertEqual((row["rigid.center.best"], row["rigid.center.at"]), (2.0, 1000))
        self.assertEqual(row["last_step"], 1000)
        self.assertEqual(row["nonfinite"], 0)
        self.assertAlmostEqual(row["wall_h"], 1.5)
        self.assertAlmostEqual(row["peak_GiB"], 20.0)
        self.assertIsNone(row["excluded"])
        b = summ.summarize_run(os.path.join(self.root, "r4sw_A_b"), "r4sw_A_b")
        self.assertIn("neg_jac_pct", b["excluded"])
        d = summ.summarize_run(os.path.join(self.root, "r4sw_A_d"), "r4sw_A_d")
        self.assertIn("nonfinite_total", d["excluded"])
        e = summ.summarize_run(os.path.join(self.root, "r4sw_A_e"), "r4sw_A_e")
        self.assertTrue(e["_no_metrics"])
        self.assertEqual(e["excluded"], "no L / Dp")

    def test_final_is_last_record_with_key(self):
        """A collapse window before the last validation still excludes (final = last record with the key)."""
        recs = [train_rec(475), val_rec(500, 4.0, 0.7), train_rec(975, collapse=3), val_rec(1000, 4.0, 0.7)]
        write_run(self.root, "r4sw_x", run_args(), recs)
        row = summ.summarize_run(os.path.join(self.root, "r4sw_x"), "r4sw_x")
        self.assertEqual(row["collapse"], 3)
        self.assertIn("collapse_suspect", row["excluded"])
        recs = [(s, {"val/mean_pair_rms_mm": 4.0, "val_deform/pair_dice": 0.7}) for s in (500, 1000)]   # no train keys
        write_run(self.root, "r4sw_y", run_args(), recs)
        row = summ.summarize_run(os.path.join(self.root, "r4sw_y"), "r4sw_y")
        self.assertIsNone(row["excluded"])
        self.assertEqual(summ._cell(row, "nonfinite"), "?")

    def test_incomplete_run_not_ranked(self):
        """Review MEDIUM: a run whose last validation is not at args.steps is excluded but listed."""
        write_run(self.root, "r4sw_A_done", run_args(),
                  done_records(1000, [(500, 4.5, 0.74, 0.0), (1000, 4.2, 0.76, 0.0)]))
        write_run(self.root, "r4sw_A_cut", run_args(),
                  done_records(1000, [(500, 3.0, 0.80, 0.0)]) + [train_rec(975)])   # died after step 500
        row = summ.summarize_run(os.path.join(self.root, "r4sw_A_cut"), "r4sw_A_cut")
        self.assertEqual(row["excluded"], "incomplete (last validation 500 of 1000)")
        write_run(self.root, "r4sw_A_nosteps", {"lr": 3e-4}, done_records(1000, [(1000, 2.0, 0.9, 0.0)]))
        row = summ.summarize_run(os.path.join(self.root, "r4sw_A_nosteps"), "r4sw_A_nosteps")
        self.assertEqual(row["excluded"], "incomplete (last validation 1000 of ?)")
        r = self.run_cli(self.root, "--glob", "r4sw_A_*")
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.splitlines()
        self.assertEqual(len(lines), 4)                       # header + all 3 runs listed
        self.assertTrue(lines[1].startswith("r4sw_A_done "))  # the only ranked run
        cut = [ln for ln in lines if ln.startswith("r4sw_A_cut ")][0]
        self.assertIn("975/1000", cut)
        self.assertIn("X (incomplete (last validation 500 of 1000))", cut)

    def test_cli_ranking_and_csv(self):
        self.tree()
        with open(os.path.join(self.root, "r4sw_submissions.jsonl"), "w") as f:
            f.write(json.dumps({"run": "r4sw_A_a", "rev": "a" * 40}) + "\n")
            f.write(json.dumps({"run": "r4sw_A_c", "rev": "b" * 40}) + "\n")
        out_csv = os.path.join(self.root, "out.csv")
        r = self.run_cli(self.root, "--csv", out_csv)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("r4sw_A_a: skipped 1 malformed metrics.jsonl line(s)", r.stderr)
        self.assertIn("r4sw_A_e: no metrics", r.stderr)
        self.assertIn("WARNING: ranked runs have different code revisions", r.stderr)
        lines = r.stdout.splitlines()
        self.assertEqual(len(lines), 6)                       # header + 5 r4sw_* runs (glob)
        self.assertTrue(lines[1].startswith("r4sw_A_c "))     # better on L and Dp
        self.assertTrue(lines[2].startswith("r4sw_A_a "))
        self.assertNotIn("other_run", r.stdout)
        with open(out_csv) as f:
            rows = list(csv.reader(f))
        head = rows[0]
        self.assertEqual(head, list(summ.COLUMNS))
        by_run = {row[0]: dict(zip(head, row)) for row in rows[1:]}
        self.assertEqual(by_run["r4sw_A_c"]["rank"], "1")
        self.assertEqual(by_run["r4sw_A_a"]["rank"], "2")
        self.assertEqual(by_run["r4sw_A_c"]["score"], "2")
        self.assertEqual(by_run["r4sw_A_a"]["score"], "4")
        self.assertEqual(by_run["r4sw_A_a"]["rev"], "a" * 8)
        self.assertEqual(by_run["r4sw_A_b"]["rev"], "?")
        self.assertTrue(by_run["r4sw_A_b"]["X"].startswith("X (neg_jac_pct"))
        self.assertEqual(by_run["r4sw_A_b"]["rank"], "")
        self.assertEqual(by_run["r4sw_A_a"]["step"], "1000/1000")
        self.assertEqual(by_run["r4sw_A_e"]["step"], "no metrics")
        self.assertEqual(by_run["r4sw_A_a"]["mean_pair.best"], "4")
        self.assertEqual(by_run["r4sw_A_a"]["L"], "4.5")

    def test_mixed_val_groups_not_ranked(self):
        write_run(self.root, "r4sw_1", run_args(vgs=16), done_records(1000, [(1000, 4.0, 0.7, 0.0)]))
        write_run(self.root, "r4sw_2", run_args(vgs=32), done_records(1000, [(1000, 3.0, 0.8, 0.0)]))
        r = self.run_cli(self.root, "--glob", "r4sw_*")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ERROR: runs use different validation groups (16, 32); not ranked", r.stderr)
        lines = r.stdout.splitlines()
        self.assertTrue(lines[1].startswith("r4sw_1 "))       # unranked: by name
        score_col = lines[0].split().index("score")
        self.assertEqual(len(lines[1].split()), score_col)   # empty score / rank / X cells

    def test_missing_root(self):
        r = self.run_cli(os.path.join(self.root, "nope"))
        self.assertEqual(r.returncode, 2)

    def test_length_curve_cli(self):
        write_run(self.root, "r4sw_A_w", run_args(steps=8000),
                  done_records(8000, [(7500, 4.6, 0.70, 0.0), (8000, 4.4, 0.70, 0.0)]))
        write_run(self.root, "r4sw_C_s12000", run_args(steps=12000),
                  done_records(12000, [(11500, 4.22, 0.748, 0.0), (12000, 4.22, 0.748, 0.0)]))
        write_run(self.root, "r4sw_C_s16000", run_args(steps=16000),
                  done_records(16000, [(15500, 4.2, 0.75, 0.0), (16000, 4.2, 0.75, 0.0)]))
        r = self.run_cli(self.root, "--length-curve", "r4sw_C_s16000,r4sw_A_w,r4sw_C_s12000")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("LENGTH 12000 (run r4sw_C_s12000)", r.stdout)
        r = self.run_cli(self.root, "--length-curve", "r4sw_A_w,r4sw_C_s12000,missing")
        self.assertIn("LENGTH UNDECIDED (run missing not found)", r.stdout)
        shutil.rmtree(os.path.join(self.root, "r4sw_C_s12000"))
        write_run(self.root, "r4sw_C_s12000", run_args(steps=12000),
                  done_records(12000, [(11500, 4.5, 0.72, 0.0), (12000, 4.5, 0.72, 0.0)]))
        r = self.run_cli(self.root, "--length-curve", "r4sw_A_w,r4sw_C_s12000,r4sw_C_s16000")
        self.assertIn("LENGTH 20000 (new run; command: RUN_NAME=r4_final_s20000 STEPS=20000 LR=0.0003 WD=0 N=64 "
                      "DEFORM_N=32 VAL_GROUP_SIZE=16 sbatch --job-name=r4final --time=30:00:00 --mem=64G "
                      "slurm/r_level4.sh)", r.stdout)
        # Review MEDIUM: a 16000-step run cut off by its time limit is not a 16000-step point.
        shutil.rmtree(os.path.join(self.root, "r4sw_C_s16000"))
        write_run(self.root, "r4sw_C_s16000", run_args(steps=16000),
                  done_records(16000, [(9500, 4.9, 0.66, 0.0), (10000, 4.8, 0.67, 0.0)]))
        r = self.run_cli(self.root, "--length-curve", "r4sw_A_w,r4sw_C_s12000,r4sw_C_s16000")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("LENGTH UNDECIDED (run r4sw_C_s16000 excluded: incomplete (last validation 10000 of 16000))",
                      r.stdout)


# --------------------------------------------------------------------------------- 11.5 e

class TestLengthDecision(unittest.TestCase):

    def test_cases(self):
        f = summ.length_decision
        self.assertEqual(f([(8000, 4.5, 0.70), (12000, 4.22, 0.748), (16000, 4.2, 0.75)]), 12000)      # (i)
        self.assertEqual(f([(8000, 4.23, 0.749), (12000, 4.21, 0.75), (16000, 4.2, 0.75)]), 8000)      # (ii)
        self.assertEqual(f([(8000, 4.8, 0.70), (12000, 4.5, 0.72), (16000, 4.2, 0.75)]), 20000)       # (iii)
        self.assertEqual(f([(8000, 4.8, 0.70), (12000, 4.5, 0.749), (16000, 4.2, 0.75)]), 16000)      # (iv)
        self.assertEqual(f([(8000, 4.8, 0.70), (12000, 4.22, 0.72), (16000, 4.2, 0.75)]), 16000)      # (iv) Dp only
        self.assertEqual(f([(16000, 4.2, 0.75), (8000, 4.5, 0.70), (12000, 4.22, 0.748)]), 12000)      # any order
        for pts in ([(8000, 4.5, 0.7), (12000, None, 0.7), (16000, 4.2, 0.75)],                       # (v)
                    [(8000, 4.5, 0.7), (16000, 4.2, 0.75)],
                    [(8000, 4.5, 0.7), (12000, float("nan"), 0.7), (16000, 4.2, 0.75)],
                    [(8000, 4.5, 0.7), (8000, 4.4, 0.7), (16000, 4.2, 0.75)]):
            d = f(pts)
            self.assertTrue(isinstance(d, str) and d.startswith("UNDECIDED ("), (pts, d))


# --------------------------------------------------------------------------------- 11.5 f

class TestPython36(unittest.TestCase):

    def test_helpers_parse_as_36(self):
        for path in (SWEEP, SUMMARIZE):
            src = _read(path)
            ast.parse(src, path, feature_version=(3, 6))
            for tok in PY37_TOKENS:
                self.assertNotIn(tok, src, (path, tok))
            self.assertNotIn("entre", src)


# ---------------------------------------------------------------------------- scripts, 11.5 h

class TestScriptsStatic(unittest.TestCase):

    def test_headers(self):
        for path, limit in ((R4, "24:00:00"), (R2, "24:00:00"), (R1, "60:00:00")):
            text = _read(path)
            header = text[:text.index("set -euo pipefail")]
            with self.subTest(os.path.basename(path)):
                self.assertNotIn("\r", text)
                self.assertTrue(all(ord(c) < 128 for c in text))
                self.assertIn("#SBATCH -A qtim\n", text)
                self.assertNotIn("entre", text.lower())
                self.assertNotRegex(text, r"(?m)^#SBATCH -A (?!qtim$)")
                for line in ("#SBATCH --gpus=1\n", "#SBATCH -N 1\n", "#SBATCH --ntasks-per-node=1\n",
                             "#SBATCH --cpus-per-task=3\n", "#SBATCH -p rtx8000\n", "#SBATCH --mem=64G\n",
                             "#SBATCH --time=%s\n" % limit,
                             "#SBATCH --output=%s/%%x-%%j.out\n" % LOGS, "#SBATCH --error=%s/%%x-%%j.err\n" % LOGS):
                    self.assertIn(line, text)
                self.assertIn(SIF, text)
                self.assertIn('export TMPDIR="$SCRATCH_ROOT/$JOB_ID"', text)
                self.assertIn("DO NOT submit without Jay's approval; at most two active jobs (squeue -u jn85)", header)
                self.assertIn("mkdir -p %s" % LOGS, header)
                self.assertIn("BGR_DRY_RUN=1", header)
                self.assertIn('repo rev $(git -C "$REPO" rev-parse HEAD', text)
                self.assertNotIn("46000", text)
                self.assertEqual(len(re.findall(r"</dev/null \|\| c?rc=\$\?", text)), 2)
                for var in ("N", "DEFORM_N"):
                    m = re.search(r"(?m)^# (Group sizes|Deform N) .*\n%s=\"\$\{%s:-" % (var, var), text)
                    self.assertIsNotNone(m, var)
                self.assertIn("set --time from the smoke's 'INFO step_time' projection x 1.3", header)
                self.assertIn("resubmit with the same RUN_NAME", header)
        for path in (R4, R2):
            header = _read(path)
            for k in ("fits(record, partition) := record.peak_mb <= BUDGET[partition][level]",
                      "BUDGET.rtx8000  = {4: 42000, 2: 41000}, BUDGET.dgx-a100 = 34000 at every level.",
                      "Unit: MiB, as written by tests/gpu/measure_level_memory.py (MB = 2**20)",
                      "46080 MiB usable - ~500 CUDA", "TF_MEM_LIMIT_MB=0 (memory growth) is kept at levels 4 / 2",
                      "TF_MEM_LIMIT_MB=<budget>", "job 10519650", "PAIR_RECOMPUTE stays false"):
                self.assertIn(k, header, (path, k))
        r4 = _read(R4)
        for k in ("RUN_NAME=r4_lin TASK_RATIO=rigid:1,affine:1 DEFORM_DECODER=true sbatch --job-name=r4lin "
                  "--time=24:00:00 --mem=64G slurm/r_level4.sh",
                  "RUN_NAME=r4_3t TASK_RATIO=rigid:1,affine:1,deform:2 INIT=$RESULTS/r4_lin/ckpt/best DEFORM_START=0 "
                  "CURRICULUM=false G_START=0 G_RAMP=0 sbatch --job-name=r4_3t --time=24:00:00 --mem=64G "
                  "slurm/r_level4.sh",
                  "RUN_NAME=r4_smoke STEPS=300 VAL_EVERY=100 WARMUP=50 DEFORM_START=100 DEFORM_RAMP=100 "
                  "CURRICULUM_RAMP=100 "
                  "G_START=100 G_RAMP=100", "allowed only if the m10 \"L4 deform no ckpt\" record fits",
                  "NOT a collapse signal"):
            self.assertIn(k, r4, k)
        r1 = _read(R1)
        for k in ("36.7 GB", "MIN_PAIRS=1", "39000 MB", "TF_MEM_LIMIT_MB=34000"):
            self.assertIn(k, r1, k)

    @unittest.skipIf(BASH is None, "bash not available")
    def test_bash_syntax(self):
        for path in SCRIPTS:
            r = subprocess.run([BASH, "-n", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
            self.assertEqual(r.returncode, 0, r.stderr)


@unittest.skipUnless(POSIX, "bash not available")
class TestScriptsDryRun(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rlevel_test_")
        self.log = os.path.join(self.tmp, "calls.log")
        self.bin = os.path.join(self.tmp, "bin")
        self.ckpt = os.path.join(self.tmp, "ckpt")
        for d in (self.bin, self.ckpt, os.path.join(self.tmp, "repo")):
            os.makedirs(d)
        _write_exec(os.path.join(self.bin, "singularity"),
                    '#!/bin/bash\nprintf "%%s|" "$@" >> "%s"\nprintf "\\n" >> "%s"\ncase "$*" in\n'
                    '*main.py*) exit ${FAKE_MAIN_RC:-0} ;;\nesac\nexit 0\n' % (self.log, self.log))
        _write_exec(os.path.join(self.bin, "nvidia-smi"), "#!/bin/sh\nexit 0\n")
        _write_exec(os.path.join(self.bin, "git"), "#!/bin/sh\necho %s\n" % FAKE_SHA)
        env = {k: v for k, v in os.environ.items() if k not in STRIP_ENV}
        env.update(PATH=self.bin + os.pathsep + os.environ.get("PATH", ""), PROJ=self.tmp,
                   REPO=os.path.join(self.tmp, "repo"), RESULTS=os.path.join(self.tmp, "results"),
                   DATA=os.path.join(self.tmp, "data"), SIF=os.path.join(self.tmp, "x.sif"),
                   SCRATCH_ROOT=os.path.join(self.tmp, "scratch"))
        self.env = env

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _tree(self):
        out = []
        for root, dirs, files in os.walk(self.tmp):
            out += [os.path.join(root, x) for x in dirs + files]
        return sorted(out)

    def run_script(self, path, dry=True, **extra):
        env = dict(self.env)
        if dry:
            env["BGR_DRY_RUN"] = "1"
        for k, v in extra.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
        return subprocess.run([BASH, path], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, stdin=subprocess.DEVNULL)

    def dry(self, path, **extra):
        before = self._tree()
        r = self.run_script(path, **extra)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self._tree(), before)                # no mkdir / write / singularity call
        main = [ln for ln in r.stdout.splitlines() if ln.startswith("MAIN: ")]
        chk = [ln for ln in r.stdout.splitlines() if ln.startswith("CHECK: ")]
        self.assertEqual((len(main), len(chk)), (1, 1), r.stdout)
        main, chk = main[0].split()[1:], chk[0].split()[1:]
        self.assertEqual(main[:2], ["python", "main.py"])
        self.assertEqual(chk[:2], ["python", "tests/gpu/check_mixed_run.py"])
        for seq in (["--amp", "true"], ["--split-group-conv", "true"], ["--intensity-aug-device", "tf"],
                    ["--max-nonfinite-frac", "0.06"], ["--memory-budget-mb", "36864"]):
            self.assertTrue(_has_seq(main, seq) or _has_seq(chk, seq), seq)
        self.assertNotIn("entre", " ".join(main + chk))
        return main[2:], chk[2:], r.stdout

    def assert_seqs(self, args, seqs):
        for seq in seqs:
            self.assertTrue(_has_seq(args, seq), (seq, args))

    def l2(self, **kw):
        d = dict(CASCADE_FROM=self.ckpt)
        d.update(kw)
        return d

    def test_r4_default(self):
        main, chk, _ = self.dry(R4)
        self.assert_seqs(main, (["--level", "4"], ["--group-size", "128"], ["--deform-group-size", "64"],
                                ["--task-ratio", "rigid:1,affine:1,deform:2"], ["--steps", "16000"],
                                ["--gradient-checkpointing", "full", "--lean-sampler", "true", "--pair-recompute",
                                 "false"],
                                ["--task-start-step", "deform:3000", "--task-start-ramp-steps", "500"],
                                ["--val-group-size", "16"], ["--min-pairs", "n"], ["--max-pairs", "64"],
                                ["--curriculum", "true", "--curriculum-ramp-steps", "3000"],
                                ["--global-transform", "true", "--global-rot-max-deg", "45", "--global-trans-max-mm",
                                 "10", "--global-start-step", "3000", "--global-ramp-steps", "3000"],
                                ["--optimizer", "adamw", "--lr", "3e-4", "--weight-decay", "0", "--schedule", "cosine",
                                 "--warmup-steps", "1000", "--min-lr-factor", "1e-3"],
                                ["--task-grad-norm", "12"], ["--lowres-method", "tent"], ["--ckpt-every", "250"],
                                ["--lambda-dice-template", "1", "--lambda-pdice", "1", "--lambda-ncc-template", "1",
                                 "--lambda-ncc-pair", "1", "--lambda-smooth", "0.5", "--p-pdice", "0.1"],
                                ["--p-pair", "0.1"], ["--val-pair-chunk", "16"], ["--deform-decoder", "auto"],
                                ["--num-threads", "3"], ["--log-every", "25", "--log-backend", "jsonl"],
                                ["--val-png", "true"]))
        for flag in ("--lean-losses", "--gpu-memory-limit-mb", "--init-weights", "--task-loss-weights",
                     "--cascade-from"):
            self.assertNotIn(flag, main)
        self.assert_seqs(chk, ([os.path.join(self.tmp, "results", "r4_manual")], ["--tasks", "rigid,affine,deform"],
                               ["--steps", "16000"], ["--grad-balance-rule", "true", "--grad-balance-from-step", "3100"],
                               ["--expect-init-weights", "none"], ["--expect-min-pairs", "n"],
                               ["--project-steps", "16000"], ["--project-val-every", "500"],
                               ["--expect-enc-grad", "true"]))
        for flag in ("--expect-cascade", "--expect-skip"):
            self.assertNotIn(flag, chk)
        self.check_parses(main)

    def test_r4_linear_only_stage1(self):
        main, chk, out = self.dry(R4, TASK_RATIO="rigid:1,affine:1", DEFORM_DECODER="true", RUN_NAME="r4_lin")
        self.assertIn("NOTE: linear-only ratio, deform-only flags dropped", out)
        self.assert_seqs(main, (["--gradient-checkpointing", "none"], ["--deform-decoder", "true"]))
        for flag in ("--lean-sampler", "--pair-recompute", "--task-start-step", "--lean-losses", "--lambda-pdice",
                     "--p-pdice"):
            self.assertNotIn(flag, main)
        self.assert_seqs(chk, (["--tasks", "rigid,affine"],))
        self.assertNotIn("--grad-balance-rule", chk)
        self.check_parses(main)

    def test_r4_deform_only(self):
        """Review v1.3 NIT 3: no task-start flags (main.py needs a task starting at 0)."""
        main, chk, out = self.dry(R4, TASK_RATIO="deform:1")
        self.assertIn("NOTE: deform-only ratio, DEFORM_START ignored", out)
        for flag in ("--task-start-step", "--task-start-ramp-steps", "--curriculum-ramp-steps"):
            self.assertNotIn(flag, main)
        self.assert_seqs(main, (["--curriculum", "false", "--global-transform", "false"],))
        self.assertNotIn("--grad-balance-rule", chk)
        self.assert_seqs(chk, (["--tasks", "deform"],))
        self.check_parses(main)

    def test_r4_grad_ckpt_none(self):
        main, _, out = self.dry(R4, GRAD_CKPT="none", PAIR_RECOMPUTE="true", LEAN_LOSSES="true")
        self.assertIn("NOTE: GRAD_CKPT=none uses the m10 no-ckpt knob set", out)
        self.assert_seqs(main, (["--gradient-checkpointing", "none"],))
        for flag in ("--lean-sampler", "--pair-recompute", "--lean-losses"):
            self.assertNotIn(flag, main)
        self.check_parses(main)

    def test_r4_fallback_stage2(self):
        """Review v1.2 finding 4: curriculum false without the ramp flag; warm start from INIT."""
        main, chk, _ = self.dry(R4, TASK_RATIO="rigid:1,affine:1,deform:2", INIT=self.ckpt, DEFORM_START="0",
                                CURRICULUM="false", G_START="0", G_RAMP="0")
        self.assert_seqs(main, (["--curriculum", "false", "--global-transform", "true"],
                                ["--init-weights", self.ckpt], ["--global-start-step", "0", "--global-ramp-steps", "0"]))
        for flag in ("--curriculum-ramp-steps", "--task-start-step"):
            self.assertNotIn(flag, main)
        self.assert_seqs(chk, (["--expect-init-weights", "full"],
                               ["--grad-balance-rule", "true", "--grad-balance-from-step", "100"]))
        self.check_parses(main)

    def test_r4_overrides(self):
        main, chk, _ = self.dry(R4, LR="1e-3", WD="1e-2", OPTIMIZER="adam", N="32", DEFORM_N="16",
                                TF_MEM_LIMIT_MB="42000", GRAD_BALANCE="deform64", VAL_GROUP_SIZE="",
                                MAX_PAIRS="0", LOWRES_METHOD="nn", EXTRA_ARGS="--seed 3")
        self.assert_seqs(main, (["--lr", "1e-3", "--weight-decay", "1e-2"], ["--optimizer", "adam"],
                                ["--gpu-memory-limit-mb", "42000"], ["--task-loss-weights", "deform:64"],
                                ["--max-pairs", "0"], ["--lowres-method", "nn"], ["--group-size", "32"]))
        self.assertEqual(main[-2:], ["--seed", "3"])
        for flag in ("--val-group-size", "--task-grad-norm"):
            self.assertNotIn(flag, main)

    def test_r4_invalid_env(self):
        for env in (dict(LR="abc"), dict(TASK_RATIO="foo"), dict(GRAD_CKPT="some"), dict(CURRICULUM="yes"),
                    dict(MAX_PAIRS="-1"), dict(VAL_GROUP_SIZE="1"), dict(STEPS="08"), dict(N="1.5"),
                    dict(OPTIMIZER="sgdw"), dict(LOWRES_METHOD="box"), dict(DEFORM_DECODER="yes"),
                    dict(INIT=os.path.join(self.tmp, "missing")), dict(GRAD_BALANCE="other"),
                    dict(PAIR_RECOMPUTE="yes"), dict(MIN_PAIRS="2"), dict(VAL_PAIR_CHUNK="0")):
            r = self.run_script(R4, **env)
            self.assertEqual(r.returncode, 2, (env, r.stdout, r.stderr))
            self.assertIn("ERROR", r.stderr)
            self.assertNotIn("MAIN: ", r.stdout)

    def test_r2_default(self):
        main, chk, _ = self.dry(R2, **self.l2())
        self.assert_seqs(main, (["--level", "2"], ["--cascade-from", self.ckpt], ["--group-size", "32"],
                                ["--deform-group-size", "16"], ["--steps", "8000"], ["--warmup-steps", "500"],
                                ["--gradient-checkpointing", "full", "--lean-sampler", "true", "--pair-recompute",
                                 "false"],
                                ["--min-pairs", "n"], ["--max-pairs", "64"], ["--bn-recal-batches", "64"],
                                ["--bn-recal-freeze", "true"], ["--cascade-skip-linear", "4:0"],
                                ["--cascade-skip-deform", "4:0.15"], ["--val-pair-chunk", "4"],
                                ["--global-transform", "true", "--global-rot-max-deg", "45", "--global-trans-max-mm",
                                 "10", "--global-start-step", "0", "--global-ramp-steps", "0", "--curriculum", "false"],
                                ["--optimizer", "adamw"], ["--lowres-method", "tent"], ["--task-grad-norm", "12"]))
        for flag in ("--lean-losses", "--val-group-size", "--gpu-memory-limit-mb", "--init-weights",
                     "--task-start-step", "--cascade-skip-prob", "--val-image-chunk"):
            self.assertNotIn(flag, main)
        self.assert_seqs(chk, (["--tasks", "rigid,affine,deform"], ["--steps", "8000"],
                               ["--grad-balance-rule", "true", "--grad-balance-from-step", "100"],
                               ["--expect-cascade", "true"], ["--expect-skip", "true"],
                               ["--expect-init-weights", "full"], ["--expect-bn-recal", "true"]))
        self.assertNotIn("--expect-cascade-stages", chk)
        self.check_parses(main)
        # linear-only at level 2 with SKIP_LINEAR 4:0: no skip key is logged -> no --expect-skip
        _, chk, _ = self.dry(R2, **self.l2(TASK_RATIO="rigid:1,affine:1"))
        self.assertNotIn("--expect-skip", chk)
        _, chk, _ = self.dry(R2, **self.l2(BN_RECAL_BATCHES="0"))
        self.assertNotIn("--expect-bn-recal", chk)

    def test_r1_default(self):
        main, chk, _ = self.dry(R1, **self.l2())
        self.assert_seqs(main, (["--level", "1"], ["--cascade-from", self.ckpt], ["--group-size", "8"],
                                ["--deform-group-size", "4"],
                                ["--gradient-checkpointing", "full", "--lean-sampler", "true", "--pair-recompute",
                                 "true", "--lean-losses", "true"],
                                ["--min-pairs", "n"], ["--max-pairs", "64"], ["--gpu-memory-limit-mb", "40000"],
                                ["--val-image-chunk", "1"], ["--val-pair-chunk", "1"], ["--bn-recal-validate", "false"],
                                ["--cascade-skip-linear", "4:0,2:0.15"], ["--cascade-skip-deform", "4:0.15,2:0.15"],
                                ["--curriculum", "false"], ["--optimizer", "adamw"], ["--lowres-method", "tent"]))
        self.assertNotIn("--val-group-size", main)
        self.assert_seqs(chk, (["--expect-cascade", "true", "--expect-cascade-stages", "2"], ["--expect-skip", "true"],
                               ["--expect-init-weights", "full"], ["--expect-bn-recal", "true"],
                               ["--grad-balance-from-step", "100"], ["--project-steps", "8000"]))
        self.check_parses(main)

    def test_cascade_guards(self):
        for path in (R2, R1):
            r = self.run_script(path)                                          # CASCADE_FROM unset
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("set CASCADE_FROM", r.stderr)
            missing = os.path.join(self.tmp, "missing")
            r = self.run_script(path, CASCADE_FROM=missing, RUN_NAME="rr")     # fresh run, missing source
            self.assertEqual(r.returncode, 2, r.stderr)
            for env in (dict(SKIP_DEFORM="4:x"), dict(SKIP_LINEAR="4"), dict(BN_RECAL_FREEZE="yes")):
                r = self.run_script(path, **self.l2(**env))
                self.assertEqual(r.returncode, 2, (path, env, r.stderr))
        last = os.path.join(self.tmp, "results", "rr", "ckpt", "last")
        os.makedirs(last)
        with open(os.path.join(last, "checkpoint"), "w") as f:
            f.write('model_checkpoint_path: "last-10"\n')
        for path in (R2, R1):                                                  # resume: guard skipped
            main, _, _ = self.dry(path, CASCADE_FROM=os.path.join(self.tmp, "missing"), RUN_NAME="rr")
            self.assertTrue(_has_seq(main, ["--cascade-from", os.path.join(self.tmp, "missing")]))

    def test_r4_stub_run(self):
        """The real path: both container calls carry exactly the dry-run argument lists; exit codes."""
        main, chk, _ = self.dry(R4)
        r = self.run_script(R4, dry=False)
        self.assertEqual(r.returncode, 0, r.stderr)
        calls = _read(self.log).splitlines()
        self.assertEqual(len(calls), 2, calls)
        self.assertTrue(calls[0].startswith("exec|--nv|-B|/scratch|-B|/autofs|-B|/cluster|-B|/space|-B|/vast|"))
        a0, a1 = calls[0].rstrip("|").split("|"), calls[1].rstrip("|").split("|")
        self.assertEqual(a0[a0.index("python") + 2:], main)
        self.assertEqual(a1[a1.index("python") + 2:], chk)
        self.assertIn("repo rev %s" % FAKE_SHA, r.stdout)
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, "scratch", "manual")))
        os.remove(self.log)
        r = self.run_script(R4, dry=False, FAKE_MAIN_RC="4")
        self.assertEqual(r.returncode, 4, r.stderr)
        self.assertEqual(len(_read(self.log).splitlines()), 1)

    # ------------------------------------------------------------------ main.py accepts the lines

    def check_parses(self, main_args):
        """Acceptance criterion: every main.py flag of a dry run (incl. --max-pairs) is accepted by
        runtime.args.parse_args."""
        from runtime import args as A
        argv = list(main_args)
        self.assertIn("--max-pairs", argv)
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                A.parse_args(argv)
        except SystemExit as e:
            self.fail("main.py rejects the dry-run line (exit %s): %s\n%s" % (e.code, err.getvalue(), argv))


class TestMaxPairsFlag(unittest.TestCase):

    def test_max_pairs_flag(self):
        from runtime import args as A
        a = A.parse_args(["--data", "x", "--task-ratio", "rigid:1,affine:1,deform:2", "--max-pairs", "64"])
        self.assertEqual(a.max_pairs, 64)


if __name__ == "__main__":
    unittest.main()
