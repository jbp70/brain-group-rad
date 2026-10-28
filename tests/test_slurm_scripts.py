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
M10C = os.path.join(REPO, "slurm", "m10_batch_memory_capped.sh")
M11SRC = os.path.join(REPO, "slurm", "m11_random_sources.sh")
M11 = os.path.join(REPO, "slurm", "m11_fp32_width_memory.sh")
# Env of slurm/m11_random_sources.sh removed from the inherited environment (TestM11Sources).
M11_SOURCE_ENV = ("SOURCES_DIR", "SOURCE_WIDTHS", "SOURCE_N", "SOURCE_TIMEOUT", "BGR_DRY_RUN", "STUB_FAIL_LEVEL",
                  "STUB_FAIL_RUN")
M9_KNOBS = ["--lean-sampler", "true", "--lean-losses", "false", "--split-group-conv", "true"]
LOGS = "/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs"
SIF = "/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif"
BASH = shutil.which("bash")
# Recipe / run env removed from the inherited environment for the stub runs.
RECIPE_ENV = ("GRAD_BALANCE", "BN_RECAL_BATCHES", "BN_RECAL_FREEZE", "SKIP_PROB", "GRAD_CKPT", "AMP",
              "EXTRA_ARGS", "RUN_NAME", "STEPS", "VAL_EVERY", "WARMUP", "SLURM_JOB_ID", "N", "DEFORM_N",
              "CASCADE_FROM", "PROJ", "REPO", "RESULTS", "DATA", "SIF", "SCRATCH_ROOT", "TMPDIR", "SLURM_CPUS_PER_TASK",
              "WIDTHS", "KNOBS", "MIN_PAIRS", "LEAN_LOSSES", "PREFETCH", "CKPT_EVERY",
              "INTENSITY_AUG_DEVICE", "TF_MEM_LIMIT_MB", "PAIR_RECOMPUTE", "VAL_IMAGE_CHUNK", "VAL_PAIR_CHUNK",
              "BN_RECAL_VALIDATE", "FP16_ACT_CLIP", "FP16_ACT_CLIP_SCOPE", "DEFORM_DECODER_DTYPE",
              "BN_EPSILON_OVERRIDE", "L1_SOURCE", "L2_SOURCE", "SRC_RESULTS", "CONFIG_TIMEOUT", "TIME_BUDGET_S",
              "ONLY", "BGR_DRY_RUN")


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
                  "TF_MEM_LIMIT_MB=34000 RUN_NAME=m9_level1_cascade sbatch -p dgx-a100 --job-name=m9long "
                  "--time=36:00:00 --mem=64G slurm/m9_level1_cascade.sh",
                  "grad_balance KEEP task-grad-norm", "FALLBACK", "UNDECIDED", "PREFETCH=2", "MaxRSS > 56 GB",
                  "--mem=96G", "36 GiB", "31.0 GB, 15.6 s/step", "git pull", "Resume plan",
                  "job 10517362", "INTENSITY_AUG_DEVICE (default tf)", "TF_MEM_LIMIT_MB (default 40000)",
                  "46080 MiB usable - CUDA context ~500 - DALI without aug ~4600", "34412",
                  "PAIR_RECOMPUTE (default true)", "job 10517366", "VAL_IMAGE_CHUNK (default 1)",
                  "BN_RECAL_VALIDATE (default false)", "job 10517503"):
            self.assertIn(k, header, k)
        body = text[re.search(r"(?m)^case ", text).start():]
        for var in ("BN_RECAL_BATCHES", "SKIP_PROB", "MIN_PAIRS", "LEAN_LOSSES", "PREFETCH", "GRAD_BALANCE",
                    "N", "DEFORM_N", "CASCADE_FROM", "INTENSITY_AUG_DEVICE", "TF_MEM_LIMIT_MB", "PAIR_RECOMPUTE",
                    "VAL_IMAGE_CHUNK", "VAL_PAIR_CHUNK", "BN_RECAL_VALIDATE"):
            self.assertNotIn("${%s:-" % var, body, var)
        for flag in ("--filters", "--decoder-widths", "--integration-steps", "--val-group-size"):
            self.assertNotIn(flag, body, flag)
        self.assertEqual(len(re.findall(r"</dev/null \|\| c?rc=\$\?", text)), 2)
        self.assertIn('CASCADE_FROM="${CASCADE_FROM:-$RESULTS/m8_level2_gradnorm/ckpt/best}"', text)

    def test_m10_capped_script(self):
        """Retrain-prereqs design 16.6 (FEAT-007): capped batch-size re-measurement header."""
        text = _read(M10C)
        self.assertNotIn("\r", text)
        self.assertTrue(all(ord(c) < 128 for c in text))
        for line in ("#SBATCH -A qtim\n", "#SBATCH -p rtx8000\n", "#SBATCH --gpus=1\n", "#SBATCH -N 1\n",
                     "#SBATCH --ntasks-per-node=1\n", "#SBATCH --cpus-per-task=3\n", "#SBATCH --mem=64G\n",
                     "#SBATCH --output=%s/%%x-%%j.out\n" % LOGS, "#SBATCH --error=%s/%%x-%%j.err\n" % LOGS):
            self.assertIn(line, text)
        self.assertRegex(text, r"(?m)^#SBATCH --time=\d\d:\d\d:\d\d$")
        self.assertNotIn("entre", text.lower())
        self.assertIn(SIF, text)
        self.assertIn("DO NOT submit without Jay's approval", text)
        self.assertNotIn("--pair-recompute", text)
        for k in ("--isolate true", "--max-pairs 64", "--int32-check true", "memory_batch_capped.json",
                  "timeout 2400", "</dev/null", "--n 48,64,80,96", "--n 128,144,160,192", "--n 32,36,40",
                  "--val-pair-chunk 16", "--min-pairs n", "--gradient-checkpointing full"):
            self.assertIn(k, text, k)

    def test_m11_header(self):
        """fp32-decoder-width FEAT-003: exact SBATCH settings and the static rules of every script."""
        text = _read(M11)
        self.assertNotIn("\r", text)
        self.assertTrue(all(ord(c) < 128 for c in text))
        self.assertNotIn("entre", text.lower())
        for line in ("#SBATCH --job-name=bgr-m11-fp32w", "#SBATCH -A qtim", "#SBATCH -p rtx8000",
                     "#SBATCH --gpus=1", "#SBATCH -N 1", "#SBATCH --ntasks-per-node=1",
                     "#SBATCH --cpus-per-task=3", "#SBATCH --mem=64G", "#SBATCH --time=04:00:00",
                     "#SBATCH --output=%s/%%x-%%j.out" % LOGS, "#SBATCH --error=%s/%%x-%%j.err" % LOGS):
            self.assertIn(line + "\n", text)
        self.assertEqual(len(re.findall(r"(?m)^#SBATCH ", text)), 11)
        self.assertNotIn("--gpus=2", text)
        self.assertNotIn("--cascade-trainable-init", text)
        self.assertIn(SIF, text)
        self.assertIn("set -uo pipefail\n", text)
        self.assertIn("</dev/null", text)
        self.assertIn("DO NOT submit without Jay's approval", text)
        self.assertIn("mkdir -p %s" % LOGS, text)
        self.assertIn("sbatch slurm/m11_fp32_width_memory.sh", text)
        self.assertIn('source "$M11_LIB"', text)
        for k in ("31.0 GB", "29.7 GB", "29.3 GB", "38.5 GB", "36.7 GB", "~3.0 h of 4 h", "skipped_time",
                  "skipped_source", "ONLY=<names, comma-separated>", "--time=02:00:00", "float32_fine"):
            self.assertIn(k, text, k)

    @unittest.skipIf(BASH is None, "bash not available")
    def test_bash_syntax(self):
        for path in SCRIPTS + (M10C, M11):
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
        for seq in (["--level", "1"], ["--cascade-from", os.path.join(self.tmp, "ckpt")], ["--group-size", "8"],
                    ["--deform-group-size", "4"], ["--min-pairs", "n"], ["--gradient-checkpointing", "full"],
                    ["--val-pair-chunk", "1"], ["--val-image-chunk", "1"], ["--bn-recal-validate", "false"],
                    ["--task-grad-norm", "12"], ["--amp", "true"],
                    ["--lean-sampler", "true"], ["--split-group-conv", "true"], ["--lean-losses", "false"],
                    ["--prefetch", "1"], ["--ckpt-every", "250"], ["--bn-recal-batches", "64"],
                    ["--cascade-skip-prob", "0"], ["--task-ratio", "rigid:1,affine:1,deform:2"],
                    ["--p-pair", "0.1", "--p-pdice", "0.1"], ["--num-threads", "3"],
                    ["--intensity-aug-device", "tf"], ["--gpu-memory-limit-mb", "40000"],
                    ["--pair-recompute", "true"]):
            self.assertTrue(self._has_seq(main, seq), seq)
        for flag in ("--filters", "--decoder-widths", "--integration-steps", "--task-loss-weights",
                     "--val-group-size", "--fp16-act-clip", "--fp16-act-clip-scope", "--deform-decoder-dtype",
                     "--bn-epsilon-override", "--nonfinite-patience-task"):
            self.assertNotIn(flag, main)
        self.assertEqual(chk[1], "tests/gpu/check_mixed_run.py")
        for seq in (["--expect-cascade-stages", "2"], ["--expect-min-pairs", "n"], ["--expect-bn-recal", "true"],
                    ["--memory-budget-mb", "36864"], ["--project-steps", "8000"],
                    ["--project-val-every", "500"], ["--expect-cascade", "true"],
                    ["--max-nonfinite-frac", "0.06", "--max-window-nonfinite-frac", "0.05"],
                    ["--tasks", "rigid,affine,deform"]):
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

    def test_m9_level1_memory_recipe(self):
        """TF_MEM_LIMIT_MB / INTENSITY_AUG_DEVICE / PAIR_RECOMPUTE overrides and their guards."""
        self._m9_stub()
        r, calls = self.run_script(M9L1, **self.l1_env(TF_MEM_LIMIT_MB="0"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("--gpu-memory-limit-mb", self._args(calls[0]))
        r, calls = self.run_script(M9L1, **self.l1_env(INTENSITY_AUG_DEVICE="dali", TF_MEM_LIMIT_MB="28000",
                                                       PAIR_RECOMPUTE="false", VAL_IMAGE_CHUNK="0",
                                                       VAL_PAIR_CHUNK="4", BN_RECAL_VALIDATE="true"))
        self.assertEqual(r.returncode, 0, r.stderr)
        main = self._args(calls[0])
        for seq in (["--intensity-aug-device", "dali"], ["--gpu-memory-limit-mb", "28000"],
                    ["--pair-recompute", "false"], ["--val-image-chunk", "0"], ["--val-pair-chunk", "4"],
                    ["--bn-recal-validate", "true"]):
            self.assertTrue(self._has_seq(main, seq), seq)
        for env in (dict(INTENSITY_AUG_DEVICE="dali"), dict(INTENSITY_AUG_DEVICE="gpu"),
                    dict(TF_MEM_LIMIT_MB="4e4"), dict(PAIR_RECOMPUTE="yes"), dict(VAL_IMAGE_CHUNK="-1"),
                    dict(VAL_PAIR_CHUNK="0"), dict(BN_RECAL_VALIDATE="no")):
            r, calls = self.run_script(M9L1, **self.l1_env(**env))
            self.assertEqual(r.returncode, 2, (env, r.stderr))
            self.assertIsNone(calls)

    def test_m9_level1_mitigations(self):
        """nonfinite-fixes D11: env overrides emit their flags before EXTRA_ARGS; invalid values exit 2."""
        self._m9_stub()
        r, calls = self.run_script(M9L1, **self.l1_env(FP16_ACT_CLIP="30000", DEFORM_DECODER_DTYPE="float32",
                                                       BN_EPSILON_OVERRIDE="1e-3", EXTRA_ARGS="--seed 3"))
        self.assertEqual(r.returncode, 0, r.stderr)
        main = self._args(calls[0])
        self.assertTrue(self._has_seq(main, ["--fp16-act-clip", "30000", "--fp16-act-clip-scope", "deform_decoder",
                                             "--deform-decoder-dtype", "float32", "--bn-epsilon-override", "1e-3",
                                             "--seed", "3"]), main)
        self.assertIn("fp16 act clip 30000 (deform_decoder)", r.stdout)
        r, calls = self.run_script(M9L1, **self.l1_env(FP16_ACT_CLIP="20000", FP16_ACT_CLIP_SCOPE="all"))
        self.assertEqual(r.returncode, 0, r.stderr)
        main = self._args(calls[0])
        self.assertTrue(self._has_seq(main, ["--fp16-act-clip", "20000", "--fp16-act-clip-scope", "all"]))
        for flag in ("--deform-decoder-dtype", "--bn-epsilon-override"):
            self.assertNotIn(flag, main)
        # fp32-decoder-width: float32_fine is accepted and passed on
        r, calls = self.run_script(M9L1, **self.l1_env(DEFORM_DECODER_DTYPE="float32_fine"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(self._has_seq(self._args(calls[0]), ["--deform-decoder-dtype", "float32_fine"]))
        self.assertIn("deform decoder dtype float32_fine", r.stdout)
        for env in (dict(FP16_ACT_CLIP="abc"), dict(FP16_ACT_CLIP="40000"), dict(FP16_ACT_CLIP_SCOPE="enc"),
                    dict(DEFORM_DECODER_DTYPE="half"), dict(BN_EPSILON_OVERRIDE="-1")):
            r, calls = self.run_script(M9L1, **self.l1_env(**env))
            self.assertEqual(r.returncode, 2, (env, r.stderr))
            self.assertIsNone(calls)
        r, _ = self.run_script(M9L1, **self.l1_env(DEFORM_DECODER_DTYPE="half"))
        self.assertIn("must be policy, float16, float32 or float32_fine", r.stderr)

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


@unittest.skipIf(BASH is None, "bash not available")
class TestM11Sources(unittest.TestCase):
    """fp32-decoder-width FEAT-002: slurm/m11_random_sources.sh (random-weight level-4 / level-2
    cascade sources at the non-default m11 widths; no --cascade-trainable-init flag)."""
    ORDER = ["l4_d12h", "l2_d12h", "l4_d12hs", "l2_d12hs", "l4_fr50", "l2_fr50"]
    WIDTH_FLAGS = {"d12h": ["--decoder-widths", "128,96,64,16,8"],
                   "d12hs": ["--filters", "16,64,128,192,256,384", "--decoder-widths", "128,96,64,16,8"],
                   "fr50": ["--filters", "16,32,128,192,256,384", "--decoder-widths", "128,96,64,16,8"]}

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="slurm_m11src_")
        self.log = os.path.join(self.tmp, "calls.log")
        bindir = os.path.join(self.tmp, "bin")
        os.makedirs(bindir)
        # singularity stub: logs the arguments ('|'-joined) and writes the run's config.json and
        # ckpt/last/checkpoint, unless --level equals $STUB_FAIL_LEVEL (exit 1, nothing written).
        body = ('#!/bin/bash\nprintf "%%s|" "$@" >> "%s"\nprintf "\\n" >> "%s"\nres=""; run=""; lvl=""\n'
                'while [ $# -gt 0 ]; do case "$1" in --results) res="$2";; --run-name) run="$2";; '
                '--level) lvl="$2";; esac; shift; done\n'
                'if [ "$lvl" = "${STUB_FAIL_LEVEL:-}" ]; then exit 1; fi\n'
                'mkdir -p "$res/$run/ckpt/last"; echo "{}" > "$res/$run/config.json"\n'
                'echo x > "$res/$run/ckpt/last/checkpoint"\nexit 0\n' % (self.log, self.log))
        for name, text in (("singularity", body), ("nvidia-smi", "#!/bin/sh\nexit 0\n"), ("git", "#!/bin/sh\nexit 0\n")):
            p = os.path.join(bindir, name)
            with open(p, "w") as f:
                f.write(text)
            os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        os.makedirs(os.path.join(self.tmp, "repo"))
        self.src = os.path.join(self.tmp, "src")
        env = {k: v for k, v in os.environ.items() if k not in RECIPE_ENV + M11_SOURCE_ENV}
        env.update(PATH=bindir + os.pathsep + os.environ.get("PATH", ""), PROJ=self.tmp,
                   REPO=os.path.join(self.tmp, "repo"), DATA=os.path.join(self.tmp, "data"),
                   SIF=os.path.join(self.tmp, "x.sif"), SCRATCH_ROOT=os.path.join(self.tmp, "scratch"),
                   SOURCES_DIR=self.src)
        self.env = env

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_script(self, **extra):
        env = dict(self.env, **extra)
        if os.path.exists(self.log):
            os.remove(self.log)
        r = subprocess.run([BASH, M11SRC], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, stdin=subprocess.DEVNULL)
        calls = _read(self.log).splitlines() if os.path.exists(self.log) else None
        return r, calls

    def status_rows(self):
        return [ln.split("\t") for ln in _read(os.path.join(self.src, "status.tsv")).splitlines()]

    def test_static(self):
        text = _read(M11SRC)
        self.assertNotIn("\r", text)
        self.assertTrue(all(ord(c) < 128 for c in text))
        self.assertNotIn("entre", text.lower())
        self.assertNotIn("#SBATCH", text)
        self.assertNotIn("--cascade-trainable-init", text)
        self.assertIn(SIF, text)
        self.assertIn("</dev/null", text)
        r = subprocess.run([BASH, "-n", M11SRC], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_dry_run_argv_parses_and_matches_grid_config(self):
        import contextlib
        import io
        import shlex
        from runtime import args as A
        r, calls = self.run_script(BGR_DRY_RUN="1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIsNone(calls)
        self.assertFalse(os.path.exists(self.src))
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("SOURCE ")]
        self.assertEqual([ln.split(":")[0][len("SOURCE "):] for ln in lines], self.ORDER)

        def config(argv):
            return {k: v for k, v in A.model_config_from_args(A.parse_args(argv)).to_dict().items() if k != "seed"}

        default = config(["--data", "x", "--task-ratio", "deform:1"])
        for ln in lines:
            name = ln.split(":")[0][len("SOURCE "):]
            level, width = int(name[1]), name.split("_", 1)[1]
            argv = shlex.split(ln.split(": ", 1)[1])
            self.assertEqual(argv[:2], ["python", "main.py"])
            a = A.parse_args(argv[2:])
            with self.subTest(name):
                self.assertEqual((a.level, a.run_name, a.results, a.steps, a.val_every, a.ckpt_every, a.resume),
                                 (level, name, self.src, 1, 0, 0, False))
                self.assertEqual(list(a.task_ratio_parsed), ["rigid"])
                self.assertEqual(a.cascade_from, os.path.join(self.src, "l4_%s" % width, "ckpt", "last")
                                 if level == 2 else None)
                self.assertEqual(a.deform_decoder_dtype, "policy")
                self.assertFalse(a.amp)
                # The ModelConfig equals the one of the m11 grid lines of that width (deform-only).
                grid = config(["--data", "x", "--task-ratio", "deform:1"] + self.WIDTH_FLAGS[width])
                self.assertEqual(config(argv[2:]), grid)
                self.assertTrue(grid["deform_decoder"])
                self.assertNotEqual(grid, default)
        err = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(err):
            A.parse_args(["--data", "x", "--cascade-trainable-init", "fresh"])    # no such flag

    def test_stub_run_and_resume(self):
        r, calls = self.run_script(SOURCE_WIDTHS="d12h fr50")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(calls), 4, calls)
        for c, name in zip(calls, ["l4_d12h", "l2_d12h", "l4_fr50", "l2_fr50"]):
            self.assertTrue(c.startswith("exec|--nv|-B|/scratch|-B|/autofs|-B|/cluster|-B|/space|-B|/vast|"), c)
            self.assertIn("|python|main.py|", c)
            self.assertIn("|--run-name|%s|" % name, c)
        self.assertIn("|--cascade-from|%s|" % os.path.join(self.src, "l4_d12h", "ckpt", "last"), calls[1])
        rows = self.status_rows()
        self.assertEqual(rows[0], ["name", "level", "widths", "rc", "ok", "elapsed_s", "log"])
        self.assertEqual([(x[0], x[3], x[4]) for x in rows[1:]],
                         [("l4_d12h", "0", "1"), ("l2_d12h", "0", "1"), ("l4_fr50", "0", "1"), ("l2_fr50", "0", "1")])
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, "scratch", "manual")))
        r, calls = self.run_script(SOURCE_WIDTHS="d12h fr50")                   # resumable: all present
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIsNone(calls)
        self.assertEqual(r.stdout.count("present, kept"), 4)
        self.assertEqual(len(self.status_rows()), 5)

    def test_failed_level4_skips_level2(self):
        r, calls = self.run_script(SOURCE_WIDTHS="d12h", STUB_FAIL_LEVEL="4")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertEqual(len(calls), 1, calls)
        self.assertEqual([(x[0], x[3], x[4]) for x in self.status_rows()[1:]],
                         [("l4_d12h", "1", "0"), ("l2_d12h", "-2", "0")])
        self.assertIn("source l2_d12h missing", r.stdout)

    def test_invalid_settings(self):
        for extra in ({"SOURCE_WIDTHS": "d12h foo"}, {"SOURCE_WIDTHS": "default"}, {"SOURCE_N": "1"},
                      {"SOURCE_TIMEOUT": "x"}):
            with self.subTest(extra):
                r, calls = self.run_script(**extra)
                self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                self.assertIsNone(calls)

    def test_sourced_functions(self):
        script = ('source "%s"; m11_width_args default; echo "[d12hs] $(m11_width_args d12hs)"; '
                  'm11_width_args bad; echo "rc=$?"; SOURCES_DIR=/s DATA=/d; m11_source_cmd 2 fr50; '
                  'echo "cmd: ${M11_CMD[*]}"; echo "end"' % M11SRC)
        r = subprocess.run([BASH, "-c", script], env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, stdin=subprocess.DEVNULL)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = r.stdout.splitlines()
        self.assertEqual(out[0], "[d12hs] --filters 16,64,128,192,256,384 --decoder-widths 128,96,64,16,8")
        self.assertEqual(out[1], "rc=2")
        self.assertIn("unknown width 'bad'", r.stderr)
        self.assertTrue(out[2].startswith("cmd: python main.py --data /d --results /s --run-name l2_fr50 "), out[2])
        self.assertTrue(out[2].endswith("--decoder-widths 128,96,64,16,8 --cascade-from /s/l4_fr50/ckpt/last"), out[2])
        self.assertEqual(out[3], "end")


def _has_seq(args, seq):
    n = len(seq)
    return any(args[i:i + n] == seq for i in range(len(args) - n + 1))


@unittest.skipIf(BASH is None, "bash not available")
class TestM11Job(unittest.TestCase):
    """fp32-decoder-width FEAT-003: slurm/m11_fp32_width_memory.sh (real-trainer memory grid; dry run,
    argv parsing, ONLY / guards, stubbed run with the random-width sources of m11_random_sources.sh)."""
    ORDER = ["l1_f32fine_default", "l1_f32_default", "l1_f32fine_d12h", "l1_f32_d12h", "l1_f32fine_d12hs",
             "l1_f32_d12hs", "l1_f32fine_fr50", "l1_f32_fr50", "l1_f16_default", "l1_f16_d12h", "l1_f16_d12hs",
             "l1_f16_fr50", "l2_f32fine_default", "l2_f32_default", "l2_f32fine_d12h", "l2_f32_d12h",
             "l4_f32fine_default", "l4_f32_default", "l4_f32fine_d12h", "l4_f32_d12h"]
    SOURCES = ["l4_d12h", "l2_d12h", "l4_d12hs", "l2_d12hs", "l4_fr50", "l2_fr50"]
    WIDTH_FLAGS = TestM11Sources.WIDTH_FLAGS
    DTYPES = {"f16": "float16", "f32": "float32", "f32fine": "float32_fine"}
    COMMON = (["--task-ratio", "deform:1"], ["--val-every", "0"], ["--ckpt-every", "0"], ["--bn-recal-batches", "0"],
              ["--resume", "false"], ["--amp", "true"], ["--gradient-checkpointing", "full"],
              ["--lean-sampler", "true"], ["--split-group-conv", "true"], ["--log-every", "1"],
              ["--intensity-aug-device", "tf"], ["--min-pairs", "n"], ["--steps", "8"], ["--optimizer", "adamw"],
              ["--lr", "3e-4"], ["--weight-decay", "0"], ["--task-grad-norm", "12"], ["--lowres-method", "tent"],
              ["--global-transform", "false"], ["--curriculum", "false"], ["--warmup-steps", "0"],
              ["--num-threads", "3"], ["--prefetch", "1"], ["--val-png", "false"])

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="slurm_m11_")
        self.log = os.path.join(self.tmp, "calls.log")
        bindir = os.path.join(self.tmp, "bin")
        os.makedirs(bindir)
        # singularity stub: logs the arguments ('|'-joined); a main.py run (--results / --run-name)
        # writes config.json and ckpt/last/checkpoint unless its name is in $STUB_FAIL_RUN (exit 1).
        body = ('#!/bin/bash\nprintf "%%s|" "$@" >> "%s"\nprintf "\\n" >> "%s"\nres=""; run=""\n'
                'while [ $# -gt 0 ]; do case "$1" in --results) res="$2";; --run-name) run="$2";; esac; shift; done\n'
                'if [ -n "$run" ]; then\n'
                '  case " ${STUB_FAIL_RUN:-} " in *" $run "*) echo "stub failure of $run"; exit 1;; esac\n'
                '  mkdir -p "$res/$run/ckpt/last"; echo "{}" > "$res/$run/config.json"\n'
                '  echo x > "$res/$run/ckpt/last/checkpoint"\nfi\nexit 0\n' % (self.log, self.log))
        for name, text in (("singularity", body), ("nvidia-smi", "#!/bin/sh\nexit 0\n"), ("git", "#!/bin/sh\nexit 0\n")):
            p = os.path.join(bindir, name)
            with open(p, "w") as f:
                f.write(text)
            os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.ckpt = os.path.join(self.tmp, "ckpt")
        os.makedirs(self.ckpt)
        env = {k: v for k, v in os.environ.items() if k not in RECIPE_ENV + M11_SOURCE_ENV}
        # REPO = this repo: the job sources slurm/m11_random_sources.sh from it (all container calls are stubs).
        env.update(PATH=bindir + os.pathsep + os.environ.get("PATH", ""), PROJ=self.tmp, REPO=REPO,
                   DATA=os.path.join(self.tmp, "data"), SIF=os.path.join(self.tmp, "x.sif"),
                   SCRATCH_ROOT=os.path.join(self.tmp, "scratch"), L1_SOURCE=self.ckpt, L2_SOURCE=self.ckpt)
        self.env = env
        self.out = os.path.join(self.tmp, "scratch", "m11", "manual")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_script(self, **extra):
        env = dict(self.env)
        for k, v in extra.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
        if os.path.exists(self.log):
            os.remove(self.log)
        r = subprocess.run([BASH, M11], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, stdin=subprocess.DEVNULL)
        calls = _read(self.log).splitlines() if os.path.exists(self.log) else None
        return r, calls

    def dry_lines(self, prefix, stdout):
        out = []
        for ln in stdout.splitlines():
            if ln.startswith(prefix + " "):
                name, cmd = ln[len(prefix) + 1:].split(": ", 1)
                out.append((name, cmd.split()))
        return out

    def test_m11_dry_run(self):
        r, calls = self.run_script(BGR_DRY_RUN="1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIsNone(calls)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "scratch")))
        src = os.path.join(self.out, "sources")
        self.assertEqual([n for n, _ in self.dry_lines("SOURCE", r.stdout)], self.SOURCES)
        configs = self.dry_lines("CONFIG", r.stdout)
        self.assertEqual([n for n, _ in configs], self.ORDER)
        for name, argv in configs:
            lv, tag, width = name.split("_")
            with self.subTest(name):
                self.assertEqual(argv[:2], ["python", "main.py"])
                self.assertEqual(argv[-4:], ["--run-name", name, "--deform-decoder-dtype", self.DTYPES[tag]])
                for seq in self.COMMON + (["--results", os.path.join(self.out, "runs")],
                                          ["--data", os.path.join(self.tmp, "data")]):
                    self.assertTrue(_has_seq(argv, seq), seq)
                if lv == "l1":
                    want = os.path.join(src, "l2_%s" % width, "ckpt", "last") if width != "default" else self.ckpt
                    for seq in (["--level", "1"], ["--cascade-from", want], ["--deform-group-size", "4"],
                                ["--group-size", "4"], ["--pair-recompute", "true"], ["--lean-losses", "true"],
                                ["--gpu-memory-limit-mb", "40000"], ["--val-image-chunk", "1"],
                                ["--val-pair-chunk", "1"]):
                        self.assertTrue(_has_seq(argv, seq), seq)
                    self.assertNotIn("--max-pairs", argv)
                elif lv == "l2":
                    want = os.path.join(src, "l4_%s" % width, "ckpt", "last") if width != "default" else self.ckpt
                    for seq in (["--level", "2"], ["--cascade-from", want], ["--deform-group-size", "16"],
                                ["--max-pairs", "64"], ["--pair-recompute", "false"], ["--val-pair-chunk", "4"]):
                        self.assertTrue(_has_seq(argv, seq), seq)
                else:
                    for seq in (["--level", "4"], ["--deform-group-size", "64"], ["--max-pairs", "64"],
                                ["--pair-recompute", "false"], ["--val-pair-chunk", "16"]):
                        self.assertTrue(_has_seq(argv, seq), seq)
                    self.assertNotIn("--cascade-from", argv)
                self.assertNotIn("--cascade-trainable-init", argv)
                self.assertNotIn("--gpu-memory-limit-mb", argv if lv != "l1" else [])
                if width == "default":
                    self.assertNotIn("--filters", argv)
                    self.assertNotIn("--decoder-widths", argv)
                else:
                    self.assertTrue(_has_seq(argv, self.WIDTH_FLAGS[width]), argv)
                    self.assertEqual(argv.count("--decoder-widths"), 1)
                    self.assertEqual(argv.count("--filters"), 0 if width == "d12h" else 1)
        collect = [ln for ln in r.stdout.splitlines() if ln.startswith("COLLECT: ")]
        self.assertEqual(collect, ["COLLECT: python scripts/collect_memory_runs.py --status %s --runs %s --out %s "
                                   "--job manual" % (os.path.join(self.out, "status.tsv"),
                                                     os.path.join(self.out, "runs"),
                                                     os.path.join(self.out, "fp32_width_memory.json"))])

    def test_m11_dry_run_argv_parses(self):
        """Every generated argv parses with runtime.args (TF-free); non-default lines get the
        ModelConfig of their random source (m11_random_sources.sh), default lines the default one."""
        import shlex
        from runtime import args as A
        r, _ = self.run_script(BGR_DRY_RUN="1")
        self.assertEqual(r.returncode, 0, r.stderr)

        def config(argv):
            return {k: v for k, v in A.model_config_from_args(A.parse_args(argv)).to_dict().items() if k != "seed"}

        default_cfg = config(["--data", "x", "--task-ratio", "deform:1"])
        default_filters = A.parse_args(["--data", "x"]).filters
        sources = {n: shlex.split(ln.split(": ", 1)[1])[2:] for n, ln in
                   ((ln[len("SOURCE "):].split(":")[0], ln) for ln in r.stdout.splitlines()
                    if ln.startswith("SOURCE "))}
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("CONFIG ")]
        self.assertEqual(len(lines), 20)
        for ln in lines:
            name = ln[len("CONFIG "):].split(":")[0]
            lv, tag, width = name.split("_")
            argv = shlex.split(ln.split(": ", 1)[1])
            with self.subTest(name):
                a = A.parse_args(argv[2:])
                self.assertEqual(a.level, int(lv[1]))
                self.assertEqual(a.deform_decoder_dtype, self.DTYPES[tag])
                self.assertTrue(a.amp)
                self.assertEqual(list(a.task_ratio_parsed), ["deform"])
                self.assertFalse(hasattr(a, "cascade_trainable_init"))
                if width == "default":
                    self.assertIsNone(a.decoder_widths)
                    self.assertEqual(list(a.filters), list(default_filters))
                    self.assertEqual(config(argv[2:]), default_cfg)
                else:
                    self.assertEqual(list(a.decoder_widths), [128, 96, 64, 16, 8])
                    self.assertEqual(list(a.filters), [int(x) for x in self.WIDTH_FLAGS[width][1].split(",")]
                                     if width != "d12h" else list(default_filters))
                    if lv != "l4":   # cascade source config == this line's config (seed aside)
                        src = sources[("l2_" if lv == "l1" else "l4_") + width]
                        self.assertEqual(config(src), config(argv[2:]))
                        self.assertNotEqual(config(argv[2:]), default_cfg)

    def test_m11_only_and_guards(self):
        r, calls = self.run_script(BGR_DRY_RUN="1", ONLY="l4_f32_default,l2_f32_d12h",
                                   L1_SOURCE=os.path.join(self.tmp, "missing"),
                                   L2_SOURCE=os.path.join(self.tmp, "missing"))
        self.assertEqual(r.returncode, 0, r.stderr)     # no default-width L1 / L2 line selected
        self.assertEqual([n for n, _ in self.dry_lines("CONFIG", r.stdout)], ["l2_f32_d12h", "l4_f32_default"])
        self.assertEqual([n for n, _ in self.dry_lines("SOURCE", r.stdout)], ["l4_d12h", "l2_d12h"])
        r, _ = self.run_script(BGR_DRY_RUN="1", ONLY="l4_f32_default")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.dry_lines("SOURCE", r.stdout), [])
        self.assertEqual([n for n, _ in self.dry_lines("CONFIG", r.stdout)], ["l4_f32_default"])
        prefix = os.path.join(self.tmp, "pfx", "best-100")
        os.makedirs(os.path.dirname(prefix))
        with open(prefix + ".index", "w") as f:
            f.write("x")
        r, _ = self.run_script(BGR_DRY_RUN="1", L1_SOURCE=prefix)      # checkpoint prefix accepted
        self.assertEqual(r.returncode, 0, r.stderr)
        for extra in (dict(ONLY="l4_f32_default,l3_f32_default"), dict(L1_SOURCE=os.path.join(self.tmp, "missing")),
                      dict(ONLY="l2_f32_default", L2_SOURCE=os.path.join(self.tmp, "missing")),
                      dict(STEPS="abc"), dict(STEPS="2"), dict(CONFIG_TIMEOUT="0"), dict(TIME_BUDGET_S="1h"),
                      dict(SOURCE_N="1"), dict(REPO=os.path.join(self.tmp, "norepo"))):
            for dry in ("1", None):
                with self.subTest(extra=extra, dry=dry):
                    r, calls = self.run_script(BGR_DRY_RUN=dry, **extra)
                    self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                    self.assertIsNone(calls)
                    self.assertIn("ERROR", r.stderr)
                    self.assertFalse(os.path.exists(os.path.join(self.tmp, "scratch")))

    def status_rows(self):
        return [ln.split("\t") for ln in _read(os.path.join(self.out, "status.tsv")).splitlines()]

    def test_m11_stub_run(self):
        r, calls = self.run_script()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(calls), 6 + 20 + 1, calls)
        mains = [c for c in calls if "|python|main.py|" in c]
        self.assertEqual(len(mains), 26)
        for c in mains:
            self.assertTrue(c.startswith("exec|--nv|-B|/scratch|-B|/autofs|-B|/cluster|-B|/space|-B|/vast|"), c)
        names = [c.split("|--run-name|")[1].split("|")[0] for c in mains]
        self.assertEqual(names, self.SOURCES + self.ORDER)
        self.assertTrue(calls[-1].startswith("exec|-B|/autofs|-B|/cluster|-B|/space|%s|python|"
                                             "scripts/collect_memory_runs.py|" % os.path.join(self.tmp, "x.sif")),
                        calls[-1])
        self.assertIn("|--out|%s|" % os.path.join(self.out, "fp32_width_memory.json"), calls[-1])
        rows = self.status_rows()
        self.assertEqual(len(rows), 21)
        self.assertEqual(rows[0], ["name", "level", "dtype", "widths", "n", "max_pairs", "rc", "elapsed_s", "log"])
        self.assertEqual([x[0] for x in rows[1:]], self.ORDER)
        lvl = {"l1": ("1", "4", "0"), "l2": ("2", "16", "64"), "l4": ("4", "64", "64")}
        for x in rows[1:]:
            lv, tag, width = x[0].split("_")
            self.assertEqual((x[1], x[4], x[5]), lvl[lv])
            self.assertEqual((x[2], x[3], x[6]), (self.DTYPES[tag], width, "0"))
            self.assertEqual(x[8], os.path.join(self.out, "runs", x[0] + ".log"))
            self.assertTrue(os.path.isfile(os.path.join(self.out, "runs", x[0], "config.json")))
            self.assertFalse(os.path.exists(os.path.join(self.out, "runs", x[0], "ckpt")))   # deleted
        for s in self.SOURCES:                                     # sources keep their checkpoints
            self.assertTrue(os.path.isfile(os.path.join(self.out, "sources", s, "ckpt", "last", "checkpoint")))
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, "scratch", "manual")))
        self.assertTrue(os.path.isdir(os.path.join(self.out, "runs")))

    def test_m11_failed_source_and_config(self):
        """A failed source skips its width's grid lines (rc -2); a failed config does not stop the loop."""
        r, calls = self.run_script(STUB_FAIL_RUN="l4_d12h l1_f32_default")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(calls), 5 + 15 + 1, calls)
        rows = {x[0]: x for x in self.status_rows()[1:]}
        self.assertEqual(len(rows), 20)
        skipped = {n for n, x in rows.items() if x[6] == "-2"}
        self.assertEqual(skipped, {"l1_f32fine_d12h", "l1_f32_d12h", "l1_f16_d12h", "l2_f32fine_d12h", "l2_f32_d12h"})
        self.assertEqual(rows["l1_f32_default"][6], "1")
        self.assertIn("stub failure of l1_f32_default", r.stdout)       # tail -n 5 of the failed log
        self.assertEqual({n for n, x in rows.items() if x[6] == "0"}, set(self.ORDER) - skipped - {"l1_f32_default"})
        self.assertIn("=== l1_f32_d12h skipped: random source", r.stdout)

    def test_m11_time_guard(self):
        r, calls = self.run_script(TIME_BUDGET_S="200", ONLY="l4_f32_default,l4_f32fine_d12h")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(calls), 1, calls)                  # the collector only
        self.assertIn("collect_memory_runs.py", calls[0])
        rows = self.status_rows()
        self.assertEqual([(x[0], x[6]) for x in rows[1:]], [("l4_f32_default", "-1"), ("l4_f32fine_d12h", "-1")])

    def test_m11_collector_exit_code(self):
        stub = os.path.join(self.tmp, "bin", "singularity")
        with open(stub) as f:
            body = f.read()
        with open(stub, "w") as f:
            # before the argument loop (it shifts "$@" away)
            f.write(body.replace('res=""; run=""\n', 'res=""; run=""\ncase "$*" in *collect_memory_runs*) exit 2;; esac\n'))
        r, calls = self.run_script(ONLY="l4_f32_default")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertEqual(len(calls), 2, calls)
        self.assertIn("collector exit code 2", r.stdout)


if __name__ == "__main__":
    unittest.main()
