"""T13 (milestone 8, design sections 12 and 16): static checks of slurm/m8_memory.sh and
slurm/m8_level2_cascade.sh, `bash -n`, and stub runs (stub `singularity` / `nvidia-smi` / `git`
first on PATH, every path redirected into a temp dir through the scripts' env overrides).
Stdlib only (no TF); the bash parts are skipped when bash is missing.
"""
import os
import re
import shutil
import stat
import subprocess
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY = os.path.join(REPO, "slurm", "m8_memory.sh")
TRAIN = os.path.join(REPO, "slurm", "m8_level2_cascade.sh")
M9 = os.path.join(REPO, "slurm", "m9_memory.sh")
M9L1 = os.path.join(REPO, "slurm", "m9_level1_cascade.sh")
SCRIPTS = (MEMORY, TRAIN, M9, M9L1)
M9_KNOBS = ["--lean-sampler", "true", "--lean-losses", "false", "--split-group-conv", "true"]
LOGS = "/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs"
SIF = "/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif"
BASH = shutil.which("bash")
# Recipe / run env removed from the inherited environment for the stub runs.
RECIPE_ENV = ("GRAD_BALANCE", "BN_RECAL_BATCHES", "BN_RECAL_FREEZE", "SKIP_PROB", "GRAD_CKPT", "AMP",
              "EXTRA_ARGS", "RUN_NAME", "STEPS", "VAL_EVERY", "WARMUP", "SLURM_JOB_ID", "N", "DEFORM_N",
              "CASCADE_FROM", "PROJ", "REPO", "RESULTS", "DATA", "SIF", "SCRATCH_ROOT", "TMPDIR", "SLURM_CPUS_PER_TASK",
              "WIDTHS", "KNOBS", "MIN_PAIRS", "LEAN_LOSSES", "PREFETCH", "CKPT_EVERY")


def _read(path):
    with open(path) as f:
        return f.read()


class TestStatic(unittest.TestCase):

    def test_sbatch_header_and_paths(self):
        for path in SCRIPTS:
            text = _read(path)
            with self.subTest(os.path.basename(path)):
                self.assertNotIn("\r", text)
                self.assertIn("#SBATCH -A qtim\n", text)
                self.assertNotIn("entre", text.lower())
                for line in ("#SBATCH --gpus=1\n", "#SBATCH -N 1\n", "#SBATCH --ntasks-per-node=1\n",
                             "#SBATCH --cpus-per-task=3\n", "#SBATCH -p rtx8000\n"):
                    self.assertIn(line, text)
                mem = re.findall(r"^#SBATCH --mem=(\d+)G$", text, re.M)
                self.assertEqual(len(mem), 1)
                self.assertLessEqual(int(mem[0]), 128)
                self.assertRegex(text, r"(?m)^#SBATCH --time=\d\d:\d\d:\d\d$")
                self.assertIn("#SBATCH --output=%s/%%x-%%j.out\n" % LOGS, text)
                self.assertIn("#SBATCH --error=%s/%%x-%%j.err\n" % LOGS, text)
                self.assertIn(SIF, text)
                self.assertIn("set -euo pipefail\n", text)
                self.assertIn('SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"', text)
                self.assertIn('export TMPDIR="$SCRATCH_ROOT/$JOB_ID"', text)
                self.assertIn("</dev/null", text)
                self.assertNotIn("--val-group-size", text)
                self.assertIn("DO NOT submit without Jay's approval", text)
                self.assertIn("mkdir -p %s" % LOGS, text)

    def test_memory_script(self):
        text = _read(MEMORY)
        self.assertIn('--num-threads "${SLURM_CPUS_PER_TASK:-3}"', text)
        self.assertIn("verify_dali_pipeline.py --data \"$DATA\" --level 2 --checks b,c,d,e,l", text)
        self.assertIn("--n-throughput 12", text)
        self.assertIn("tests/gpu/measure_level_memory.py --level 2", text)
        self.assertEqual(len(re.findall(r'"\$\{M\[@\]\}"', text)), 4)              # four memory calls
        self.assertEqual(len(re.findall(r"</dev/null \|\| rc_x=\$\?", text)), 5)    # every call
        self.assertIn("sbatch --job-name=m8mem --time=04:00:00 --mem=64G slurm/m8_memory.sh", text)

    def test_m9_memory_script(self):
        """Milestone 9 phase 1 (design 5.5, AC11): header, </dev/null on every container call."""
        text = _read(M9)
        header = text[:text.index("set -euo pipefail")]
        self.assertIn("#SBATCH --mem=64G\n", text)
        self.assertIn("#SBATCH --time=12:00:00\n", text)
        self.assertIn("-p dgx-a100", header)
        self.assertIn("sbatch --job-name=m9mem --time=12:00:00 --mem=64G slurm/m9_memory.sh", header)
        self.assertNotIn("-A entre", text)
        self.assertNotRegex(text, r"(?m)^#SBATCH -A (?!qtim$)")
        run_lines = [ln for ln in text.splitlines() if '"${RUN[@]}"' in ln]
        devnull = re.findall(r"</dev/null \|\| rc_x=\$\?", text)
        self.assertEqual(len(run_lines), 15)
        self.assertEqual(len(run_lines), len(devnull))
        self.assertIn('RUN=(singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF")',
                      text)
        self.assertIn('OUT_DIR="$PROJ/scratch/m9"', text)
        for k in ("tests/gpu/probe_tensor_limit.py --mode run", "tests/gpu/probe_tensor_limit.py --mode m8-errors",
                  "scripts/step_times_from_metrics.py", 'verify_dali_pipeline.py --data "$DATA" --level 1',
                  "tests/gpu/measure_level_memory.py", "--n-throughput 16",
                  "tests the Fw cap: predicted N5 39.6 GiB, N6 47.3 GiB -> OOM expected"):
            self.assertIn(k, text)
        self.assertTrue(all(ord(c) < 128 for c in text))

    def test_train_script(self):
        text = _read(TRAIN)
        self.assertIn('gradnorm) BAL_ARGS="--task-grad-norm 12"', text)
        self.assertIn('deform64) BAL_ARGS="--task-loss-weights deform:64"', text)
        self.assertIn('[ ! -d "$CASCADE_FROM" ] && [ ! -f "$CASCADE_FROM.index" ]', text)
        self.assertIn('if [ ! -f "$RESULTS/$RUN_NAME/ckpt/last/checkpoint" ]; then', text)
        for k in ("grad_balance KEEP", "FALLBACK", "UNDECIDED", "scancel", "squeue -u jn85",
                  "peak (job 10504614, RTX 8000, scratch/m8/memory_level2.json)",
                  "verdict: fits the 40 GB A100 (peak 27.8 GB <= 36 GB)",
                  '--max-nonfinite-frac 0.06', "-p dgx-a100",
                  "sbatch --job-name=m8mem --time=04:00:00 --mem=64G slurm/m8_memory.sh",
                  "--job-name=m8smoke --time=02:00:00", "--job-name=m8long --time=24:00:00", "T_smoke"):
            self.assertIn(k, text)
        # every recipe default is assigned once, in the variable block before the first `case`
        body = text[re.search(r"(?m)^case ", text).start():]
        for var in ("BN_RECAL_BATCHES", "SKIP_PROB", "GRAD_CKPT", "AMP", "BN_RECAL_FREEZE", "GRAD_BALANCE"):
            self.assertNotIn("${%s:-" % var, body, var)
        for flag in ("--level 2", '--cascade-from "$CASCADE_FROM"', '--group-size "$N"',
                     '--deform-group-size "$DEFORM_N"', "--task-ratio rigid:1,affine:1,deform:2",
                     "--lambda-ncc-template 1 --lambda-ncc-pair 1", '--bn-recal-batches "$BN_RECAL_BATCHES"',
                     '--bn-recal-freeze "$BN_RECAL_FREEZE"', '--cascade-skip-prob "$SKIP_PROB"',
                     '--gradient-checkpointing "$GRAD_CKPT"', '--amp "$AMP"', "--global-rot-max-deg 45",
                     "--global-trans-max-mm 10", "--curriculum false", "--log-backend jsonl",
                     "--expect-init-weights full --expect-enc-grad true --expect-cascade true "
                     "--grad-balance-rule true"):
            self.assertIn(flag, body, flag)

    def test_m9_level1_script(self):
        """Milestone 9 phase 2 (design section 8): header comments, recipe, both container calls."""
        text = _read(M9L1)
        header = text[:text.index("set -euo pipefail")]
        self.assertIn("#SBATCH --mem=64G\n", text)
        self.assertIn("#SBATCH --time=48:00:00\n", text)
        self.assertNotRegex(text, r"(?m)^#SBATCH -A (?!qtim$)")
        self.assertTrue(all(ord(c) < 128 for c in text))
        for k in ("RUN_NAME=m9_l1_smoke STEPS=200 VAL_EVERY=100 WARMUP=50 sbatch --job-name=m9smoke "
                  "--time=04:00:00 --mem=64G slurm/m9_level1_cascade.sh",
                  "RUN_NAME=m9_level1_cascade sbatch --job-name=m9long --time=48:00:00 --mem=64G "
                  "slurm/m9_level1_cascade.sh",
                  "RUN_NAME=m9_level1_cascade sbatch -p dgx-a100 --job-name=m9long --time=36:00:00 --mem=64G "
                  "slurm/m9_level1_cascade.sh",
                  "grad_balance KEEP task-grad-norm", "FALLBACK", "UNDECIDED", "PREFETCH=2", "MaxRSS > 56 GB",
                  "--mem=96G", "36 GiB", "31.0 GB, 15.6 s/step", "git pull", "Resume plan"):
            self.assertIn(k, header, k)
        body = text[re.search(r"(?m)^case ", text).start():]
        for var in ("BN_RECAL_BATCHES", "SKIP_PROB", "MIN_PAIRS", "LEAN_LOSSES", "PREFETCH", "GRAD_BALANCE",
                    "N", "DEFORM_N", "CASCADE_FROM"):
            self.assertNotIn("${%s:-" % var, body, var)
        for flag in ("--filters", "--decoder-widths", "--integration-steps", "--val-group-size"):
            self.assertNotIn(flag, body, flag)
        self.assertEqual(len(re.findall(r"</dev/null \|\| c?rc=\$\?", text)), 2)
        self.assertIn('CASCADE_FROM="${CASCADE_FROM:-$RESULTS/m8_level2_gradnorm/ckpt/best}"', text)

    @unittest.skipIf(BASH is None, "bash not available")
    def test_bash_syntax(self):
        for path in SCRIPTS:
            r = subprocess.run([BASH, "-n", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
            self.assertEqual(r.returncode, 0, r.stderr)


@unittest.skipIf(BASH is None, "bash not available")
class TestStubRuns(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="slurm_stub_")
        self.log = os.path.join(self.tmp, "calls.log")
        bindir = os.path.join(self.tmp, "bin")
        os.makedirs(bindir)
        stubs = {"singularity": '#!/bin/sh\nprintf "%%s\\n" "$*" >> "%s"\nexit 0\n' % self.log,
                 "nvidia-smi": "#!/bin/sh\nexit 0\n", "git": "#!/bin/sh\nexit 0\n"}
        for name, body in stubs.items():
            p = os.path.join(bindir, name)
            with open(p, "w") as f:
                f.write(body)
            os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        os.makedirs(os.path.join(self.tmp, "repo"))
        os.makedirs(os.path.join(self.tmp, "ckpt"))
        env = {k: v for k, v in os.environ.items() if k not in RECIPE_ENV}
        env.update(PATH=bindir + os.pathsep + os.environ.get("PATH", ""), PROJ=self.tmp,
                   REPO=os.path.join(self.tmp, "repo"), RESULTS=os.path.join(self.tmp, "results"),
                   DATA=os.path.join(self.tmp, "data"), SIF=os.path.join(self.tmp, "x.sif"),
                   SCRATCH_ROOT=os.path.join(self.tmp, "scratch"))
        self.base_env = env

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_script(self, path, **extra):
        env = dict(self.base_env)
        for k, v in extra.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
        if os.path.exists(self.log):
            os.remove(self.log)
        r = subprocess.run([BASH, path], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, stdin=subprocess.DEVNULL)
        calls = _read(self.log).splitlines() if os.path.exists(self.log) else None
        return r, calls

    def train_env(self, **kw):
        d = dict(CASCADE_FROM=os.path.join(self.tmp, "ckpt"), N="4", DEFORM_N="2")
        d.update(kw)
        return d

    def test_train_default_recipe(self):
        r, calls = self.run_script(TRAIN, **self.train_env())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(calls), 2, calls)
        for k in ("main.py", "--level 2", "--task-grad-norm 12", "--bn-recal-batches 64"):
            self.assertIn(k, calls[0])
        self.assertNotIn("--task-loss-weights", calls[0])
        self.assertNotIn("--val-group-size", calls[0])
        self.assertIn("check_mixed_run.py", calls[1])
        self.assertIn("--expect-bn-recal true", calls[1])
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, "scratch", "manual")))

    def test_train_requires_n(self):
        r, calls = self.run_script(TRAIN, **self.train_env(N=None))
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIsNone(calls)
        self.assertIn("set N", r.stderr)

    def test_train_deform64(self):
        r, calls = self.run_script(TRAIN, **self.train_env(GRAD_BALANCE="deform64"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(calls), 2, calls)
        self.assertIn("--task-loss-weights deform:64", calls[0])
        self.assertNotIn("--task-grad-norm", calls[0])

    def test_train_resume_skips_guard(self):
        env = self.train_env(CASCADE_FROM=os.path.join(self.tmp, "missing"), RUN_NAME="r")
        r, calls = self.run_script(TRAIN, **env)                 # fresh run, source missing
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIsNone(calls)
        last = os.path.join(self.tmp, "results", "r", "ckpt", "last")
        os.makedirs(last)
        with open(os.path.join(last, "checkpoint"), "w") as f:
            f.write('model_checkpoint_path: "last-10"\n')
        r, calls = self.run_script(TRAIN, **env)                 # resume: guard skipped
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(calls), 2, calls)
        self.assertIn("--cascade-from %s" % os.path.join(self.tmp, "missing"), calls[0])

    def test_memory_script(self):
        r, calls = self.run_script(MEMORY)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(calls), 5, calls)
        self.assertIn("verify_dali_pipeline.py", calls[0])
        self.assertIn("--level 2", calls[0])
        self.assertIn("--num-threads 3", calls[0])
        for c in calls[1:]:
            self.assertIn("measure_level_memory.py --level 2", c)
        self.assertIn("--val-pass true", calls[1])
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, "scratch", "manual")))

    def l1_env(self, **kw):
        d = dict(CASCADE_FROM=os.path.join(self.tmp, "ckpt"))
        d.update(kw)
        return d

    def test_m9_level1_default_recipe(self):
        """Design section 8: default argv of main.py and of the checker."""
        self._m9_stub()
        r, calls = self.run_script(M9L1, **self.l1_env())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(calls), 2, calls)
        self.assertTrue(calls[0].startswith("exec|--nv|-B|/scratch|-B|/autofs|-B|/cluster|-B|/space|-B|/vast|"))
        main, chk = self._args(calls[0]), self._args(calls[1])
        self.assertEqual(main[1], "main.py")
        for seq in (["--level", "1"], ["--cascade-from", os.path.join(self.tmp, "ckpt")], ["--group-size", "16"],
                    ["--deform-group-size", "4"], ["--min-pairs", "n"], ["--gradient-checkpointing", "full"],
                    ["--val-pair-chunk", "4"], ["--task-grad-norm", "12"], ["--amp", "true"],
                    ["--lean-sampler", "true"], ["--split-group-conv", "true"], ["--lean-losses", "false"],
                    ["--prefetch", "1"], ["--ckpt-every", "250"], ["--bn-recal-batches", "64"],
                    ["--cascade-skip-prob", "0"], ["--task-ratio", "rigid:1,affine:1,deform:2"],
                    ["--p-pair", "0.1", "--p-pdice", "0.1"], ["--num-threads", "3"]):
            self.assertTrue(self._has_seq(main, seq), seq)
        for flag in ("--filters", "--decoder-widths", "--integration-steps", "--task-loss-weights",
                     "--val-group-size"):
            self.assertNotIn(flag, main)
        self.assertEqual(chk[1], "tests/gpu/check_mixed_run.py")
        for seq in (["--expect-cascade-stages", "2"], ["--expect-min-pairs", "n"], ["--expect-bn-recal", "true"],
                    ["--memory-budget-mb", "36864"], ["--project-steps", "8000"],
                    ["--project-val-every", "500"], ["--expect-cascade", "true"],
                    ["--max-nonfinite-frac", "0.06"], ["--tasks", "rigid,affine,deform"]):
            self.assertTrue(self._has_seq(chk, seq), seq)
        # BN_RECAL_BATCHES 0 -> no --expect-bn-recal
        r, calls = self.run_script(M9L1, **self.l1_env(BN_RECAL_BATCHES="0"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("--expect-bn-recal", self._args(calls[1]))

    def test_m9_level1_deform64_and_guards(self):
        self._m9_stub()
        r, calls = self.run_script(M9L1, **self.l1_env(GRAD_BALANCE="deform64", MIN_PAIRS="1"))
        self.assertEqual(r.returncode, 0, r.stderr)
        main = self._args(calls[0])
        self.assertTrue(self._has_seq(main, ["--task-loss-weights", "deform:64"]))
        self.assertNotIn("--task-grad-norm", main)
        self.assertTrue(self._has_seq(main, ["--min-pairs", "1"]))
        self.assertTrue(self._has_seq(self._args(calls[1]), ["--expect-min-pairs", "1"]))
        for env in (dict(MIN_PAIRS="x"), dict(DEFORM_N="abc"), dict(N="1.5"), dict(GRAD_BALANCE="other")):
            r, calls = self.run_script(M9L1, **self.l1_env(**env))
            self.assertEqual(r.returncode, 2, (env, r.stderr))
            self.assertIsNone(calls)

    def test_m9_level1_resume_skips_guard(self):
        self._m9_stub()
        env = self.l1_env(CASCADE_FROM=os.path.join(self.tmp, "missing"), RUN_NAME="r1")
        r, calls = self.run_script(M9L1, **env)                  # fresh run, source missing
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIsNone(calls)
        last = os.path.join(self.tmp, "results", "r1", "ckpt", "last")
        os.makedirs(last)
        with open(os.path.join(last, "checkpoint"), "w") as f:
            f.write('model_checkpoint_path: "last-10"\n')
        r, calls = self.run_script(M9L1, **env)                  # resume: guard skipped
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(calls), 2, calls)
        self.assertTrue(self._has_seq(self._args(calls[0]), ["--cascade-from", os.path.join(self.tmp, "missing")]))

    def test_m9_level1_exit_codes(self):
        """main.py's exit code if non-zero (checker not run), else the checker's."""
        self._m9_stub('*main.py*) exit 4;;')
        r, calls = self.run_script(M9L1, **self.l1_env())
        self.assertEqual(r.returncode, 4, r.stderr)
        self.assertEqual(len(calls), 1, calls)
        self._m9_stub('*check_mixed_run*) exit 1;;')
        r, calls = self.run_script(M9L1, **self.l1_env())
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertEqual(len(calls), 2, calls)

    def _m9_stub(self, fail_cases=""):
        """singularity stub logging every argument separately ('|'-joined) so that word splitting
        of $KNOBS is visible; ``fail_cases``: extra `case` arms (pattern) exit code;;."""
        body = ('#!/bin/bash\nprintf "%%s|" "$@" >> "%s"\nprintf "\\n" >> "%s"\ncase "$*" in\n%s\n*) ;;\nesac\n'
                'exit 0\n' % (self.log, self.log, fail_cases))
        p = os.path.join(self.tmp, "bin", "singularity")
        with open(p, "w") as f:
            f.write(body)

    @staticmethod
    def _args(call):
        """Arguments after the sif path (python and its arguments)."""
        a = call.rstrip("|").split("|")
        return a[a.index("python"):]

    def _has_seq(self, args, seq):
        n = len(seq)
        return any(args[i:i + n] == seq for i in range(len(args) - n + 1))

    def test_m9_memory_script(self):
        self._m9_stub()
        r, calls = self.run_script(M9)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(calls), 18, calls)
        args = [self._args(c) for c in calls]
        for c in calls:
            self.assertTrue(c.startswith("exec|--nv|-B|/scratch|-B|/autofs|-B|/cluster|-B|/space|-B|/vast|"), c)
        out = os.path.join(self.tmp, "scratch", "m9")
        # 0a / 0b / 0c / 1
        self.assertEqual(args[0][:4], ["python", "tests/gpu/probe_tensor_limit.py", "--mode", "run"])
        self.assertIn(os.path.join(out, "tensor_limit.json"), args[0])
        self.assertEqual(args[1][:4], ["python", "tests/gpu/probe_tensor_limit.py", "--mode", "m8-errors"])
        self.assertTrue(self._has_seq(args[1], ["--m8-json", os.path.join(self.tmp, "scratch", "m8",
                                                                          "memory_level2.json")]))
        self.assertEqual(args[2][1], "scripts/step_times_from_metrics.py")
        self.assertTrue(self._has_seq(args[2], ["--min-step", "1000", "--extra-keys", "diag_u_res_max_vox"]))
        self.assertIn(os.path.join(self.tmp, "results", "m7_three_task_lncc"), args[2])
        self.assertIn(os.path.join(self.tmp, "results", "m8_level2_gradnorm"), args[2])
        self.assertEqual(args[3][1], "tests/gpu/verify_dali_pipeline.py")
        for seq in (["--level", "1"], ["--checks", "b,c,d,e,l"], ["--num-threads", "3"], ["--n-throughput", "16"]):
            self.assertTrue(self._has_seq(args[3], seq), seq)
        # 2p .. 6b: measure_level_memory.py, all into memory_m9.json with LNCC 1/1
        for a in args[4:]:
            self.assertEqual(a[1], "tests/gpu/measure_level_memory.py")
            self.assertTrue(self._has_seq(a, ["--lambda-ncc-template", "1", "--lambda-ncc-pair", "1"]))
            self.assertTrue(self._has_seq(a, ["--out", os.path.join(out, "memory_m9.json")]))
            self.assertTrue(self._has_seq(a, ["--isolate", "true"]), a)       # one config per process
        l1 = ["--level", "1", "--amp", "true", "--warmup-steps", "1", "--steps", "3"]
        p1, p2 = args[4], args[5]                                       # 2p: priority (FEAT-004)
        self.assertTrue(self._has_seq(p1, l1 + ["--cascade", "true", "--tasks", "deform", "--widths", "f0d1",
                                                "--integration-steps", "7", "--n", "3,4,5",
                                                "--gradient-checkpointing", "full"] + M9_KNOBS
                                      + ["--val-pass", "true", "--val-pair-chunk", "4"]), p1)
        self.assertTrue(self._has_seq(p2, ["--widths", "fr50", "--integration-steps", "7", "--n", "4",
                                           "--gradient-checkpointing", "full", "--lean-sampler", "true",
                                           "--lean-losses", "false", "--split-group-conv", "false"]), p2)
        g = args[6]                                                     # 2: level-1 deform grid
        for seq in (l1, ["--cascade", "true", "--tasks", "deform", "--widths", "f0d1,default,fr50",
                         "--integration-steps", "6,7", "--n", "2,3,4,5,6,8", "--gradient-checkpointing", "full"]
                    + M9_KNOBS + ["--val-pass", "true", "--val-pair-chunk", "4"]):
            self.assertTrue(self._has_seq(g, seq), seq)
        b = args[7]                                                     # 2b: no cascade
        self.assertTrue(self._has_seq(b, l1 + ["--cascade", "false", "--tasks", "deform", "--widths", "f0d1"]))
        self.assertTrue(self._has_seq(b, ["--n", "2,4", "--gradient-checkpointing", "full"] + M9_KNOBS))
        c2 = args[8]                                                    # 2c: fr50 + lean losses
        self.assertTrue(self._has_seq(c2, ["--widths", "fr50", "--integration-steps", "7", "--n", "5,6,7"]))
        self.assertTrue(self._has_seq(c2, ["--lean-sampler", "true", "--lean-losses", "true",
                                           "--split-group-conv", "true"]))
        ablations = [["--lean-sampler", "false", "--lean-losses", "false", "--split-group-conv", "true",
                      "--gradient-checkpointing", "full"],
                     ["--lean-sampler", "true", "--lean-losses", "false", "--split-group-conv", "false",
                      "--gradient-checkpointing", "full"],
                     ["--lean-sampler", "true", "--lean-losses", "true", "--split-group-conv", "true",
                      "--gradient-checkpointing", "full"],
                     ["--lean-sampler", "true", "--lean-losses", "false", "--split-group-conv", "true",
                      "--gradient-checkpointing", "both"]]
        for a, ab in zip(args[9:13], ablations):                        # 3: ablations, split into args
            self.assertTrue(self._has_seq(a, ["--widths", "f0d1", "--integration-steps", "7", "--n", "4"] + ab
                                          + ["--val-pass", "false"]), a)
        lin = args[13]                                                  # 4: level-1 linear
        self.assertTrue(self._has_seq(lin, l1 + ["--cascade", "true", "--tasks", "rigid,affine", "--widths", "f0d1",
                                                 "--n", "4,8,12,16", "--split-group-conv", "true"]))
        a5, b5 = args[14], args[15]                                     # 5a / 5b
        self.assertTrue(self._has_seq(a5, ["--level", "2", "--cascade", "true", "--amp", "true"]))
        self.assertTrue(self._has_seq(a5, ["--n", "8", "--gradient-checkpointing", "both", "--lean-sampler", "false",
                                           "--lean-losses", "false", "--split-group-conv", "false",
                                           "--val-pass", "true"]))
        self.assertTrue(self._has_seq(b5, ["--level", "2", "--cascade", "true", "--amp", "true"]))
        self.assertTrue(self._has_seq(b5, ["--n", "8,12,16", "--gradient-checkpointing", "full"] + M9_KNOBS
                                      + ["--val-pass", "true"]))
        s6, s6b = args[16], args[17]                                    # 6 / 6b: level 4
        self.assertTrue(self._has_seq(s6, ["--level", "4", "--cascade", "false", "--amp", "false"]))
        self.assertTrue(self._has_seq(s6, ["--tasks", "deform", "--widths", "default,f0d1", "--n", "16,24,32",
                                           "--gradient-checkpointing", "none"]))
        self.assertTrue(self._has_seq(s6b, ["--tasks", "rigid", "--widths", "default,f0d1", "--n", "16,32"]))
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, "scratch", "manual")))
        self.assertTrue(os.path.isdir(os.path.join(out, "dali")))

    def test_m9_memory_env_overrides(self):
        self._m9_stub()
        r, calls = self.run_script(M9, WIDTHS="fr50,f0d1", KNOBS="--lean-sampler true --lean-losses true")
        self.assertEqual(r.returncode, 0, r.stderr)
        args = [self._args(c) for c in calls]
        self.assertTrue(self._has_seq(args[6], ["--widths", "fr50,f0d1"]))
        self.assertTrue(self._has_seq(args[7], ["--widths", "fr50", "--integration-steps", "7"]))
        self.assertTrue(self._has_seq(args[15], ["--gradient-checkpointing", "full", "--lean-sampler", "true",
                                                 "--lean-losses", "true", "--val-pass", "true"]))

    def test_m9_memory_exit_code(self):
        """Every call runs; the script exits with the first non-zero exit code."""
        self._m9_stub('*verify_dali_pipeline*) exit 3;;\n*"--level 4"*) exit 5;;')
        r, calls = self.run_script(M9)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertEqual(len(calls), 18, calls)
        self.assertIn("verify_dali_pipeline (level 1) exit code 3", r.stdout)
        self.assertIn("measure_level_memory (level 4 deform) exit code 5", r.stdout)
        self.assertIn("finished m9_memory exit code 3", r.stdout)


if __name__ == "__main__":
    unittest.main()
