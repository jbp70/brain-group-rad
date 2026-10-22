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
SCRIPTS = (MEMORY, TRAIN)
LOGS = "/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs"
SIF = "/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif"
BASH = shutil.which("bash")
# Recipe / run env removed from the inherited environment for the stub runs.
RECIPE_ENV = ("GRAD_BALANCE", "BN_RECAL_BATCHES", "BN_RECAL_FREEZE", "SKIP_PROB", "GRAD_CKPT", "AMP",
              "EXTRA_ARGS", "RUN_NAME", "STEPS", "VAL_EVERY", "WARMUP", "SLURM_JOB_ID", "N", "DEFORM_N",
              "CASCADE_FROM", "PROJ", "REPO", "RESULTS", "DATA", "SIF", "SCRATCH_ROOT", "TMPDIR", "SLURM_CPUS_PER_TASK")


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

    def test_train_script(self):
        text = _read(TRAIN)
        self.assertIn('gradnorm) BAL_ARGS="--task-grad-norm 12"', text)
        self.assertIn('deform64) BAL_ARGS="--task-loss-weights deform:64"', text)
        self.assertIn('[ ! -d "$CASCADE_FROM" ] && [ ! -f "$CASCADE_FROM.index" ]', text)
        self.assertIn('if [ ! -f "$RESULTS/$RUN_NAME/ckpt/last/checkpoint" ]; then', text)
        for k in ("grad_balance KEEP", "FALLBACK", "UNDECIDED", "scancel", "squeue -u jn85",
                  "peak: unmeasured (fill in from scratch/m8/memory_level2.json)", "-p dgx-a100",
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


if __name__ == "__main__":
    unittest.main()
