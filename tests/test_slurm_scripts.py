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
PREDICT = os.path.join(REPO, "slurm", "predict_mindboggle.sh")
EVAL_SYNTH = os.path.join(REPO, "slurm", "eval_synth.sh")
TTA_MB = os.path.join(REPO, "slurm", "tta_mindboggle.sh")
TTA_CMP = os.path.join(REPO, "slurm", "compare_tta.sh")
PREDICT_MULTI = os.path.join(REPO, "slurm", "predict_mindboggle_multi.sh")
# Env of slurm/m11_random_sources.sh removed from the inherited environment (TestM11Sources).
M11_SOURCE_ENV = ("SOURCES_DIR", "SOURCE_WIDTHS", "SOURCE_L4_ONLY", "SOURCE_N", "SOURCE_TIMEOUT", "BGR_DRY_RUN",
                  "STUB_FAIL_LEVEL",
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
              "ONLY", "BGR_DRY_RUN", "FILTERS", "DECODER_WIDTHS", "TTA", "TTA_LEVELS", "SKIP_CHECKS",
              "FROZEN_INTEGRATION_GRID", "L4_SOURCE", "GROUP_ARGS", "SMOOTH", "STUB_WRITE_CKPT")


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

    def test_predict_script(self):
        """predict-script design 9: header, gate / record steps, no sbatch call, grid reference."""
        text = _read(PREDICT)
        self.assertNotIn("\r", text)
        self.assertTrue(all(ord(c) < 128 for c in text))
        self.assertNotIn("entre", text.lower())
        for line in ("#SBATCH --job-name=bgr-predict", "#SBATCH -A qtim", "#SBATCH -p rtx8000",
                     "#SBATCH --gpus=1", "#SBATCH -N 1", "#SBATCH --ntasks-per-node=1",
                     "#SBATCH --cpus-per-task=3", "#SBATCH --mem=96G", "#SBATCH --time=05:00:00",
                     "#SBATCH --output=%s/%%x-%%j.out" % LOGS, "#SBATCH --error=%s/%%x-%%j.err" % LOGS):
            self.assertIn(line + "\n", text)
        self.assertIn(SIF, text)
        self.assertIn("DO NOT submit without Jay's approval", text)
        self.assertIn("never\n# drop the gate", text)
        self.assertIn("N_CHECK=3 CHECK_CHUNKS='--chunk-l4 2 --chunk-l2 2 --chunk-l1 2'", text)
        self.assertIn('CHECK_CHUNKS=${CHECK_CHUNKS:-"--chunk-l4 3 --chunk-l2 3 --chunk-l1 3"}', text)
        self.assertIn("BRANCHES=${BRANCHES:-deform}\n", text)
        self.assertIn("CKPT=${CKPT:-$PROJ/results/m9_level1_cascade/ckpt/best_deform}\n", text)
        self.assertIn("GRID_REF=${GRID_REF:-$PROJ/data/target_grid/target_grid.json}\n", text)
        self.assertIn("EVAL_PAIRS=${EVAL_PAIRS:-none}\n", text)
        lines = text.splitlines()
        self.assertFalse([ln for ln in lines if ln.lstrip().startswith("sbatch")])
        sing = [ln for ln in lines if ln.startswith("singularity ")]
        self.assertEqual(len(sing), 6)
        for ln in sing:
            self.assertIn("</dev/null", ln)
        pred = [ln for ln in sing if " predict.py " in ln]
        self.assertEqual(len(pred), 4)
        for ln in pred:
            self.assertIn('--grid-reference "$GRID_REF"', ln)
            self.assertIn("--nv", ln)
        checks = [ln for ln in pred if "check-streaming" in ln]
        self.assertEqual(len(checks), 3)
        for ln in checks:
            self.assertIn(" $CHECK_CHUNKS ", ln)
            self.assertIn('--n-subjects "$N_CHECK"', ln)
        a, b, c = checks
        self.assertIn("--precision float32", a)
        self.assertIn('--branches "$BRANCHES"', a)
        self.assertTrue(a.endswith("|| exit $?"), a)
        self.assertIn("--precision checkpoint", b)
        self.assertIn("--precision float32", c)
        self.assertIn("--branches affine+deform", c)
        for ln in (b, c):
            self.assertRegex(ln, r'\|\| echo "WARNING[^"]*"$')
        run = [ln for ln in pred if " predict.py run " in ln]
        self.assertEqual(len(run), 1)
        self.assertTrue(run[0].endswith("$EXTRA </dev/null || exit $?"), run[0])
        # predict-levels: LEVELS / ALLOW_UNTRAINED_SKIP -> $LEVEL_FLAGS (empty by default) on 1a-1c, 2.
        self.assertIn("LEVELS=${LEVELS:-}\n", text)
        self.assertIn("ALLOW_UNTRAINED_SKIP=${ALLOW_UNTRAINED_SKIP:-}\n", text)
        self.assertIn('LEVEL_FLAGS=""\n', text)
        self.assertIn('if [ -n "$LEVELS" ]; then LEVEL_FLAGS="--levels $LEVELS"; fi\n', text)
        self.assertIn('if [ "$ALLOW_UNTRAINED_SKIP" = "1" ]; then LEVEL_FLAGS="$LEVEL_FLAGS '
                      '--allow-untrained-skip"; fi\n', text)
        for ln in pred:
            self.assertEqual(ln.count(" $LEVEL_FLAGS "), 1, ln)
        self.assertIn(' --out "$FIELDS" $LEVEL_FLAGS $TTA_FLAGS $EXTRA </dev/null || exit $?', run[0])
        self.assertTrue(run[0].endswith('--out "$FIELDS" $LEVEL_FLAGS $TTA_FLAGS $EXTRA </dev/null || exit $?'))
        # TTA (milestone 10, plan D7): TTA / TTA_LEVELS reach step 2 only; SKIP_CHECKS=1 skips 1a-1c.
        self.assertIn("TTA=${TTA:-none}\n", text)
        self.assertIn("TTA_LEVELS=${TTA_LEVELS:-}\n", text)
        self.assertIn("SKIP_CHECKS=${SKIP_CHECKS:-}\n", text)
        self.assertIn('TTA_FLAGS="--tta $TTA"\n', text)
        self.assertIn('if [ -n "$TTA_LEVELS" ]; then TTA_FLAGS="$TTA_FLAGS --tta-levels $TTA_LEVELS"; fi\n', text)
        self.assertIn('if [ "$SKIP_CHECKS" != "1" ]; then\n', text)
        self.assertIn('echo "checks 1a-1c skipped (SKIP_CHECKS=1; run in the TTA-none arm of the same job)"\n',
                      text)
        for ln in checks:
            self.assertNotIn("TTA_FLAGS", ln)
            self.assertNotIn("--tta", ln)
        self.assertEqual(len([ln for ln in lines if "$TTA_FLAGS" in ln and ln.startswith("singularity ")]), 1)
        ev = [ln for ln in sing if " -m evaluation " in ln]
        self.assertEqual(len(ev), 2)
        self.assertIn("check-convention", ev[0])
        self.assertIn('--pairs "$EVAL_PAIRS" --workers 3', ev[1])
        self.assertIn('CONV="--components xyz --signs +++ --units mm"', text)
        self.assertIn("sbatch slurm/eval_mindboggle.sh", text)

    def test_eval_synth_header(self):
        """synth-contrast design 8 (FEAT-005): exact SBATCH policy lines, usage line, approval note."""
        text = _read(EVAL_SYNTH)
        self.assertNotIn("\r", text)
        self.assertNotIn("entre", text.lower())
        for line in ("#SBATCH -A qtim\n", "#SBATCH -p rtx8000\n", "#SBATCH --gpus=1\n", "#SBATCH -N 1\n",
                     "#SBATCH --ntasks-per-node=1\n", "#SBATCH --cpus-per-task=3\n", "#SBATCH --mem=128G\n",
                     "#SBATCH --time=08:00:00\n", "#SBATCH --output=%s/%%x-%%j.out\n" % LOGS,
                     "#SBATCH --error=%s/%%x-%%j.err\n" % LOGS, "set -euo pipefail\n"):
            self.assertIn(line, text)
        self.assertEqual(len(re.findall(r"^#SBATCH --mem=", text, re.M)), 1)
        for k in ("DO NOT submit without Jay's approval", SIF, "mkdir -p %s" % LOGS,
                  'RUNS="<real-only config B L1 run> $R/r1_cfgB_synth05" sbatch --job-name=evalsyn '
                  '--time=08:00:00 --mem=128G slurm/eval_synth.sh',
                  'sstat -j "$SLURM_JOB_ID.batch" --format=MaxRSS >> "$OUT/sstat.txt"',
                  'OUT="${OUT:-$PROJ/results/eval_synth/$JOB_ID}"', "singularity exec --nv", "</dev/null"):
            self.assertIn(k, text, k)
        self.assertTrue(all(ord(c) < 128 for c in text))

    @unittest.skipIf(BASH is None, "bash not available")
    def test_bash_syntax(self):
        for path in SCRIPTS + (M10C, M11, PREDICT, EVAL_SYNTH, TTA_MB, TTA_CMP, PREDICT_MULTI):
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

    def test_predict_levels_flags(self):
        """predict-levels: LEVELS unset -> no level flag (the predict.py calls equal the LEVELS run's
        calls with the flags removed); LEVELS / ALLOW_UNTRAINED_SKIP reach steps 1a, 1b, 1c and 2."""
        base = dict(OUT=os.path.join(self.tmp, "out"), EXTRA=None, CHECK_CHUNKS=None, BRANCHES=None)
        r, calls = self.run_script(PREDICT, LEVELS=None, ALLOW_UNTRAINED_SKIP=None, **base)
        self.assertEqual(r.returncode, 0, r.stderr)
        pred = [c for c in calls if " predict.py " in c]
        self.assertEqual(len(pred), 4, calls)
        for c in calls:
            self.assertNotIn("--levels", c)
            self.assertNotIn("--allow-untrained-skip", c)
        r2, calls2 = self.run_script(PREDICT, LEVELS="2,1", ALLOW_UNTRAINED_SKIP="1", **base)
        self.assertEqual(r2.returncode, 0, r2.stderr)
        pred2 = [c for c in calls2 if " predict.py " in c]
        self.assertEqual(len(pred2), 4, calls2)
        for c in pred2:
            self.assertEqual((c + " ").count(" --levels 2,1 --allow-untrained-skip "), 1, c)
        self.assertEqual([c.replace(" --levels 2,1 --allow-untrained-skip", "") for c in calls2], calls)
        r3, calls3 = self.run_script(PREDICT, LEVELS="4", ALLOW_UNTRAINED_SKIP=None, **base)
        self.assertEqual(r3.returncode, 0, r3.stderr)
        for c in (c for c in calls3 if " predict.py " in c):
            self.assertIn(" --levels 4 ", c + " ")
            self.assertNotIn("--allow-untrained-skip", c)

    def test_predict_tta_flags(self):
        """TTA plan D7: default env -> today's calls plus `--tta none` on step 2 only; TTA=flip8
        TTA_LEVELS=1 SKIP_CHECKS=1 -> no check-streaming call, `--tta flip8 --tta-levels 1` on step 2."""
        base = dict(OUT=os.path.join(self.tmp, "out"), EXTRA=None, CHECK_CHUNKS=None, BRANCHES=None,
                    LEVELS=None, ALLOW_UNTRAINED_SKIP=None)
        r, calls = self.run_script(PREDICT, **base)
        self.assertEqual(r.returncode, 0, r.stderr)
        pred = [c for c in calls if " predict.py " in c]
        self.assertEqual(len(pred), 4, calls)
        for c in pred[:3]:
            self.assertIn("check-streaming", c)
            self.assertNotIn("--tta", c)
        self.assertIn(" predict.py run ", pred[3])
        self.assertTrue(pred[3].endswith(" --out %s --tta none" % os.path.join(self.tmp, "out", "fields")), pred[3])
        self.assertNotIn("checks 1a-1c skipped", r.stdout)
        r2, calls2 = self.run_script(PREDICT, TTA="flip8", TTA_LEVELS="1", SKIP_CHECKS="1", **base)
        self.assertEqual(r2.returncode, 0, r2.stderr)
        self.assertIn("checks 1a-1c skipped (SKIP_CHECKS=1; run in the TTA-none arm of the same job)", r2.stdout)
        self.assertFalse([c for c in calls2 if "check-streaming" in c], calls2)
        pred2 = [c for c in calls2 if " predict.py " in c]
        self.assertEqual(len(pred2), 1, calls2)
        self.assertTrue(pred2[0].endswith(" --tta flip8 --tta-levels 1"), pred2[0])
        self.assertEqual(pred2[0].replace(" --tta flip8 --tta-levels 1", " --tta none"), pred[3])
        self.assertEqual(len([c for c in calls2 if " -m evaluation " in c]), 2)

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

    def test_level4_only_widths(self):
        """SOURCE_L4_ONLY widths build l4_W only (level-2 grid lines need nothing else)."""
        r, _ = self.run_script(SOURCE_WIDTHS="d12h fr50", SOURCE_L4_ONLY="fr50", BGR_DRY_RUN="1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual([ln.split(":")[0][len("SOURCE "):] for ln in r.stdout.splitlines()
                          if ln.startswith("SOURCE ")], ["l4_d12h", "l2_d12h", "l4_fr50"])
        r, calls = self.run_script(SOURCE_WIDTHS="d12hs fr50", SOURCE_L4_ONLY="d12hs fr50")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual([c.split("|--run-name|")[1].split("|")[0] for c in calls], ["l4_d12hs", "l4_fr50"])
        self.assertFalse(os.path.exists(os.path.join(self.src, "l2_d12hs")))

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
    # Follow-up L4 / L2 lines at d12hs / fr50: 12 main lines, then the smaller-group fallbacks.
    MAIN12 = ["l4_f32fine_d12hs", "l4_f32_d12hs", "l4_f16_d12hs", "l4_f32fine_fr50", "l4_f32_fr50", "l4_f16_fr50",
              "l2_f32fine_d12hs", "l2_f32_d12hs", "l2_f16_d12hs", "l2_f32fine_fr50", "l2_f32_fr50", "l2_f16_fr50"]
    FALLBACKS = ["%s_%s_%s_n%d" % (lv, tag, w, n) for lv, n in (("l4", 48), ("l4", 32), ("l2", 12), ("l2", 8))
                 for w in ("d12hs", "fr50") for tag in ("f32fine", "f32")]
    ORDER = ORDER + MAIN12 + FALLBACKS
    SOURCES = ["l4_d12h", "l2_d12h", "l4_d12hs", "l2_d12hs", "l4_fr50", "l2_fr50"]
    WIDTH_FLAGS = TestM11Sources.WIDTH_FLAGS
    DTYPES = {"f16": "float16", "f32": "float32", "f32fine": "float32_fine"}
    DEFAULT_N = {"l1": 4, "l2": 16, "l4": 64}

    @classmethod
    def parse(cls, name):
        """<level>_<tag>_<width>[_n<N>] -> (lv, tag, width, N)."""
        parts = name.split("_")
        n = int(parts.pop()[1:]) if len(parts) == 4 else cls.DEFAULT_N[parts[0]]
        return parts[0], parts[1], parts[2], n
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
            lv, tag, width, n = self.parse(name)
            with self.subTest(name):
                self.assertEqual(argv[:2], ["python", "main.py"])
                self.assertTrue(_has_seq(argv, ["--deform-group-size", str(n)]), argv)
                self.assertTrue(_has_seq(argv, ["--group-size", str(n)]), argv)
                self.assertEqual(argv.count("--deform-group-size"), 1)
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
                    for seq in (["--level", "2"], ["--cascade-from", want],
                                ["--max-pairs", "64"], ["--pair-recompute", "false"], ["--val-pair-chunk", "4"]):
                        self.assertTrue(_has_seq(argv, seq), seq)
                else:
                    for seq in (["--level", "4"], ["--max-pairs", "64"],
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
        self.assertEqual(len(lines), 48)
        for ln in lines:
            name = ln[len("CONFIG "):].split(":")[0]
            lv, tag, width, n = self.parse(name)
            argv = shlex.split(ln.split(": ", 1)[1])
            with self.subTest(name):
                a = A.parse_args(argv[2:])
                self.assertEqual(a.level, int(lv[1]))
                self.assertEqual((a.deform_group_size, a.group_size), (n, n))
                self.assertEqual(a.max_pairs, 0 if lv == "l1" else 64)
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
        self.assertEqual([n for n, _ in self.dry_lines("SOURCE", r.stdout)], ["l4_d12h"])   # L2 line: l4_W only
        # Follow-up job: the 12 main lines need l4_d12hs + l4_fr50 only (no l2_W, no L1 / L2 real source).
        r, _ = self.run_script(BGR_DRY_RUN="1", ONLY=",".join(reversed(self.MAIN12)),
                               L1_SOURCE=os.path.join(self.tmp, "missing"),
                               L2_SOURCE=os.path.join(self.tmp, "missing"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual([n for n, _ in self.dry_lines("SOURCE", r.stdout)], ["l4_d12hs", "l4_fr50"])
        self.assertEqual([n for n, _ in self.dry_lines("CONFIG", r.stdout)], self.MAIN12)    # list order kept
        r, _ = self.run_script(BGR_DRY_RUN="1", ONLY="l2_f32_fr50_n8,l4_f32fine_d12hs_n48,l1_f32_fr50")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual([n for n, _ in self.dry_lines("SOURCE", r.stdout)], ["l4_fr50", "l2_fr50"])
        configs = dict(self.dry_lines("CONFIG", r.stdout))
        self.assertEqual(list(configs), ["l1_f32_fr50", "l4_f32fine_d12hs_n48", "l2_f32_fr50_n8"])
        self.assertTrue(_has_seq(configs["l4_f32fine_d12hs_n48"], ["--deform-group-size", "48"]))
        self.assertTrue(_has_seq(configs["l2_f32_fr50_n8"], ["--deform-group-size", "8"]))
        self.assertTrue(_has_seq(configs["l2_f32_fr50_n8"], ["--cascade-from", os.path.join(
            self.out, "sources", "l4_fr50", "ckpt", "last")]))
        r, _ = self.run_script(BGR_DRY_RUN="1", ONLY="l4_f32_d12hs_n64")      # not in the list
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
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
        self.assertEqual(len(calls), 6 + 48 + 1, calls)
        mains = [c for c in calls if "|python|main.py|" in c]
        self.assertEqual(len(mains), 54)
        for c in mains:
            self.assertTrue(c.startswith("exec|--nv|-B|/scratch|-B|/autofs|-B|/cluster|-B|/space|-B|/vast|"), c)
        names = [c.split("|--run-name|")[1].split("|")[0] for c in mains]
        self.assertEqual(names, self.SOURCES + self.ORDER)
        self.assertTrue(calls[-1].startswith("exec|-B|/autofs|-B|/cluster|-B|/space|%s|python|"
                                             "scripts/collect_memory_runs.py|" % os.path.join(self.tmp, "x.sif")),
                        calls[-1])
        self.assertIn("|--out|%s|" % os.path.join(self.out, "fp32_width_memory.json"), calls[-1])
        rows = self.status_rows()
        self.assertEqual(len(rows), 49)
        self.assertEqual(rows[0], ["name", "level", "dtype", "widths", "n", "max_pairs", "rc", "elapsed_s", "log"])
        self.assertEqual([x[0] for x in rows[1:]], self.ORDER)
        lvl = {"l1": ("1", "0"), "l2": ("2", "64"), "l4": ("4", "64")}
        for x in rows[1:]:
            lv, tag, width, n = self.parse(x[0])
            self.assertEqual((x[1], x[4], x[5]), (lvl[lv][0], str(n), lvl[lv][1]))
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
        self.assertEqual(len(calls), 5 + 43 + 1, calls)
        rows = {x[0]: x for x in self.status_rows()[1:]}
        self.assertEqual(len(rows), 48)
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


VERIFY_SYNTH = os.path.join(REPO, "slurm", "verify_synth.sh")
SYNTH_PREVIEW = os.path.join(REPO, "slurm", "synth_preview.sh")
SYNTH_PREVIEW_ENV = ("PROJ", "REPO", "DATA", "SIF", "SLURM_JOB_ID", "N", "SEED", "N_GROUPS", "SPLIT", "LABEL_SET",
                     "BGR_DRY_RUN")


class TestSynthScripts(unittest.TestCase):
    """synth-contrast FEAT-004 (plan deviations D-c / D-d): slurm/verify_synth.sh (GPU checks s1-s3,
    rtx8000) and slurm/synth_preview.sh (CPU preview PNGs on the basic partition)."""

    def common(self, text):
        self.assertNotIn("\r", text)
        self.assertTrue(all(ord(c) < 128 for c in text))
        self.assertIn("#SBATCH -A qtim\n", text)
        self.assertNotIn("entre", text.lower())
        self.assertNotRegex(text, r"(?m)^#SBATCH -A (?!qtim$)")
        for line in ("#SBATCH -N 1\n", "#SBATCH --ntasks-per-node=1\n",
                     "#SBATCH --output=%s/%%x-%%j.out\n" % LOGS, "#SBATCH --error=%s/%%x-%%j.err\n" % LOGS):
            self.assertIn(line, text)
        self.assertIn(SIF, text)
        self.assertIn("DO NOT submit without Jay's approval", text)
        self.assertEqual(len(re.findall(r"(?m)^#SBATCH --mem=", text)), 1)

    def test_verify_synth_header(self):
        text = _read(VERIFY_SYNTH)
        self.common(text)
        for line in ("#SBATCH -p rtx8000\n", "#SBATCH --gpus=1\n", "#SBATCH --cpus-per-task=3\n",
                     "#SBATCH --mem=64G\n", "#SBATCH --time=01:00:00\n"):
            self.assertIn(line, text)
        for k in ("--checks s", "for lvl in 4 1; do", '--out "$OUT/verify_synth_l$lvl.json"',
                  'OUT="$PROJ/scratch/verify_synth"', "sbatch slurm/verify_synth.sh", "</dev/null"):
            self.assertIn(k, text, k)

    def test_synth_preview_header(self):
        text = _read(SYNTH_PREVIEW)
        self.common(text)
        for line in ("#SBATCH -p basic\n", "#SBATCH --cpus-per-task=2\n", "#SBATCH --mem=16G\n",
                     "#SBATCH --time=00:30:00\n"):
            self.assertIn(line, text)
        self.assertNotIn("--gpus", text)
        self.assertNotIn("--nv", text)                         # CPU only, container without --nv
        self.assertIn("singularity exec -B /autofs -B /cluster -B /space", text)
        self.assertIn("BGR_DRY_RUN=1", text)
        self.assertIn("mkdir -p %s" % LOGS, text)

    @unittest.skipIf(BASH is None, "bash not available")
    def test_bash_syntax(self):
        for path in (VERIFY_SYNTH, SYNTH_PREVIEW):
            r = subprocess.run([BASH, "-n", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
            self.assertEqual(r.returncode, 0, r.stderr)

    @unittest.skipIf(BASH is None, "bash not available")
    def test_synth_preview_dry_run(self):
        tmp = tempfile.mkdtemp(prefix="synth_preview_test_")
        try:
            env = {k: v for k, v in os.environ.items() if k not in SYNTH_PREVIEW_ENV}
            env.update(PROJ=tmp, BGR_DRY_RUN="1", SLURM_JOB_ID="77")

            def run(**extra):
                e = dict(env, **extra)
                return subprocess.run([BASH, SYNTH_PREVIEW], env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      universal_newlines=True, stdin=subprocess.DEVNULL)

            r = run()
            self.assertEqual(r.returncode, 0, r.stderr)
            lines = [ln for ln in r.stdout.splitlines() if ln.startswith("PREVIEW: ")]
            self.assertEqual(len(lines), 3, r.stdout)
            for lvl, ln in zip((4, 2, 1), lines):
                self.assertIn("singularity exec -B /autofs -B /cluster -B /space %s python scripts/synth_preview.py"
                              % SIF, ln)
                self.assertIn("--data %s/data/npy_v2 --level %d --n 8 --seed 0 --groups 2 --split train "
                              "--label-set synthfine --out %s/scratch/synth_preview/77/l%d.png" % (tmp, lvl, tmp, lvl),
                              ln)
                self.assertNotIn("--nv", ln)
            self.assertEqual(os.listdir(tmp), [])                # dry run writes nothing
            for bad in (dict(N="x"), dict(N_GROUPS="-1"), dict(LABEL_SET="a b")):
                r = run(**bad)
                self.assertEqual(r.returncode, 2, (bad, r.stdout, r.stderr))
                self.assertIn("ERROR", r.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


EVAL_SYNTH_ENV = ("RUNS", "LEVEL", "OUT", "PRESETS", "DATA", "BGR_DRY_RUN", "PROJ", "REPO", "SIF",
                  "SCRATCH_ROOT", "SLURM_JOB_ID", "TMPDIR")


@unittest.skipIf(BASH is None, "bash not available")
class TestEvalSynthScript(unittest.TestCase):
    """synth-contrast FEAT-005: slurm/eval_synth.sh dry run (argv parses with eval_synth.py's
    parser, nothing written), validation exits 2, stub run (one --nv container call)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="evalsyn_")
        self.env = {k: v for k, v in os.environ.items() if k not in EVAL_SYNTH_ENV}
        self.env.update(PROJ=os.path.join(self.tmp, "proj"), SLURM_JOB_ID="88")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_script(self, **extra):
        e = dict(self.env, **extra)
        return subprocess.run([BASH, EVAL_SYNTH], env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, stdin=subprocess.DEVNULL)

    def test_dry_run_argv(self):
        import sys
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        import eval_synth as ES
        r = self.run_script(BGR_DRY_RUN="1", RUNS="a b")
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("MAIN: ")]
        self.assertEqual(len(lines), 1, r.stdout)
        argv = lines[0][len("MAIN: "):].split()
        self.assertEqual(argv[:2], ["python", "eval_synth.py"])
        self.assertEqual(argv[2:], ["--runs", "a", "b", "--level", "1", "--out",
                                    os.path.join(self.tmp, "proj", "results", "eval_synth", "88")])
        ns = ES.build_parser().parse_args(argv[2:])
        self.assertEqual((ns.runs, ns.level, ns.presets, ns.data, ns.sanity_gate), (["a", "b"], 1, "", None, "on"))
        r = self.run_script(BGR_DRY_RUN="1", RUNS="a", LEVEL="4", PRESETS="random_0,t2_like", DATA="/d",
                            OUT="/o")
        self.assertEqual(r.returncode, 0, r.stderr)
        argv = [ln for ln in r.stdout.splitlines() if ln.startswith("MAIN: ")][0].split()[3:]
        self.assertEqual(argv, ["--runs", "a", "--level", "4", "--out", "/o", "--presets", "random_0,t2_like",
                                "--data", "/d"])
        ES.build_parser().parse_args(argv)
        self.assertEqual(os.listdir(self.tmp), [])           # dry run writes nothing

    def test_validation_exits_2(self):
        for bad in (dict(RUNS=""), dict(RUNS="a b c"), dict(RUNS="a", LEVEL="3"),
                    dict(RUNS="a", PRESETS="t2 like")):
            r = self.run_script(BGR_DRY_RUN="1", **bad)
            self.assertEqual(r.returncode, 2, (bad, r.stdout, r.stderr))
            self.assertIn("ERROR", r.stderr)
            self.assertNotIn("MAIN:", r.stdout)

    def test_stub_run(self):
        bindir = os.path.join(self.tmp, "bin")
        os.makedirs(bindir)
        log = os.path.join(self.tmp, "calls.log")
        for name, body in (("singularity", '#!/bin/sh\nprintf "%%s\\n" "$*" >> "%s"\nexit 5\n' % log),
                           ("git", "#!/bin/sh\nexit 0\n"), ("sstat", "#!/bin/sh\necho MaxRSS\necho 1234K\n")):
            p = os.path.join(bindir, name)
            with open(p, "w") as f:
                f.write(body)
            os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        os.makedirs(os.path.join(self.tmp, "repo"))
        out = os.path.join(self.tmp, "out")
        r = self.run_script(RUNS="r1 r2", OUT=out, REPO=os.path.join(self.tmp, "repo"),
                            SIF=os.path.join(self.tmp, "x.sif"), SCRATCH_ROOT=os.path.join(self.tmp, "scratch"),
                            PATH=bindir + os.pathsep + os.environ.get("PATH", ""))
        self.assertEqual(r.returncode, 5, r.stderr)              # the container's exit code
        calls = _read(log).splitlines()
        self.assertEqual(len(calls), 1, calls)
        self.assertIn("exec --nv", calls[0])
        self.assertIn("python eval_synth.py --runs r1 r2 --level 1 --out %s" % out, calls[0])
        self.assertIn("1234K", _read(os.path.join(out, "sstat.txt")))
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, "scratch", "88")))


TTA_ENV = ("CKPT", "NAME", "PROCESSED", "GRID_REF", "PARTS", "SEED", "TTA_LEVELS", "DRY_RUN", "REPO", "SIF",
           "SCRATCH_ROOT", "SLURM_JOB_ID", "TMPDIR", "TTA", "SKIP_CHECKS", "A", "B", "OUT", "KIND")
PROJ_DIR = "/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg"


class TestTTAScripts(unittest.TestCase):
    """TTA FEAT-004 (design 9.4, plan D5 / D8): slurm/tta_mindboggle.sh and slurm/compare_tta.sh."""

    def common(self, text):
        self.assertNotIn("\r", text)
        self.assertTrue(all(ord(c) < 128 for c in text))
        self.assertNotIn("entre", text.lower())
        self.assertIn("DO NOT submit without Jay's approval", text)
        self.assertIn("set -uo pipefail\n", text)
        self.assertIn(SIF, text)
        self.assertFalse([ln for ln in text.splitlines() if ln.lstrip().startswith("sbatch")])

    def test_tta_mindboggle_static(self):
        text = _read(TTA_MB)
        self.common(text)
        sb = ["#SBATCH --job-name=bgr-tta-mb", "#SBATCH -A qtim", "#SBATCH -p rtx8000", "#SBATCH --gpus=1",
              "#SBATCH -N 1", "#SBATCH --ntasks-per-node=1", "#SBATCH --cpus-per-task=3", "#SBATCH --mem=96G",
              "#SBATCH --time=10:00:00", "#SBATCH --output=%s/%%x-%%j.out" % LOGS,
              "#SBATCH --error=%s/%%x-%%j.err" % LOGS]
        self.assertEqual(re.findall(r"(?m)^#SBATCH .*$", text), sb)
        self.assertIn("two-active-jobs rule", text)
        self.assertIn("export TMPDIR=${SCRATCH_ROOT:-/scratch}/${SLURM_JOB_ID:-manual}\n", text)
        self.assertIn("PARTS=${PARTS:-predict linear compare}\n", text)
        self.assertIn("SEED=${SEED:-0}\n", text)
        self.assertIn("TTA_LEVELS=${TTA_LEVELS:-}\n", text)
        lines = text.splitlines()
        arms = [ln for ln in lines if 'bash "$REPO/slurm/predict_mindboggle.sh"' in ln]
        self.assertEqual(len(arms), 2)
        self.assertIn('NAME="${NAME}_tta_none"', arms[0])
        self.assertIn(" TTA=none ", arms[0])
        self.assertIn('NAME="${NAME}_tta_flip8"', arms[1])
        self.assertIn(" TTA=flip8 ", arms[1])
        self.assertIn(" SKIP_CHECKS=1 ", arms[1])
        self.assertIn('TTA_LEVELS="$TTA_LEVELS"', arms[1])
        self.assertIn("    for task in rigid affine; do\n        for arm in none flip8; do\n", text)
        lin = [ln for ln in lines if " predict.py linear-eval " in ln and not ln.startswith("#")]
        self.assertEqual(len(lin), 1)
        for k in ("singularity exec --nv", '--checkpoint "$CKPT"', '--processed-dir "$PROCESSED"',
                  '--grid-reference "$GRID_REF"', '--task "$task"', '--seed "$SEED"', '--tta "$arm"',
                  '--out "$LIN/${NAME}_${task}_tta_${arm}"', "</dev/null"):
            self.assertIn(k, lin[0], k)
        cmp_lines = [ln for ln in lines if "scripts/compare_tta.py" in ln and "singularity exec" in ln]
        self.assertEqual(len(cmp_lines), 2)
        for ln in cmp_lines:
            self.assertNotIn("--nv", ln)
        self.assertIn('--out "$CMP/${NAME}_template" --kind eval', cmp_lines[0])
        self.assertIn('--out "$CMP/${NAME}_${task}_linear" --kind linear', cmp_lines[1])
        self.assertIn('CONV=\\"--components xyz --signs +++ --units mm\\" sbatch slurm/eval_mindboggle.sh"', text)
        self.assertIn("OUT=$CMP/${NAME}_allpairs sbatch slurm/compare_tta.sh", text)
        self.assertIn('exit "$failed"', text)

    def test_compare_tta_static(self):
        text = _read(TTA_CMP)
        self.common(text)
        sb = ["#SBATCH --job-name=bgr-tta-compare", "#SBATCH -A qtim", "#SBATCH -p basic", "#SBATCH -N 1",
              "#SBATCH --ntasks-per-node=1", "#SBATCH --cpus-per-task=1", "#SBATCH --mem=4G",
              "#SBATCH --time=00:15:00", "#SBATCH --output=%s/%%x-%%j.out" % LOGS,
              "#SBATCH --error=%s/%%x-%%j.err" % LOGS]
        self.assertEqual(re.findall(r"(?m)^#SBATCH .*$", text), sb)
        self.assertNotIn("--gpus", text)
        self.assertIn("KIND=${KIND:-eval}\n", text)
        self.assertIn("export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n", text)
        self.assertIn('singularity exec -B /autofs -B /scratch "$SIF" bash -c \'cd "$1" && shift && python "$@"\' '
                      'cmp "$REPO" scripts/compare_tta.py --a "$A" --b "$B" --out "$OUT" --kind "$KIND" </dev/null\n',
                      text)
        self.assertIn("KIND=linear sbatch slurm/compare_tta.sh", text)
        self.assertIn("_allpairs sbatch slurm/compare_tta.sh", text)

    @unittest.skipIf(BASH is None, "bash not available")
    def test_bash_syntax(self):
        for path in (PREDICT, TTA_MB, TTA_CMP):
            r = subprocess.run([BASH, "-n", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
            self.assertEqual(r.returncode, 0, r.stderr)

    def _run(self, path, **extra):
        env = {k: v for k, v in os.environ.items() if k not in TTA_ENV}
        env.update(extra)
        return subprocess.run([BASH, path], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, stdin=subprocess.DEVNULL)

    @unittest.skipIf(BASH is None, "bash not available")
    def test_tta_mindboggle_dry_run(self):
        tmp = tempfile.mkdtemp(prefix="tta_dry_")
        try:
            r = self._run(TTA_MB, DRY_RUN="1", NAME="x", CKPT="/tmp/x", SCRATCH_ROOT=tmp)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(os.listdir(tmp), [])                    # nothing created
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        dry = [ln[len("DRY: "):] for ln in r.stdout.splitlines() if ln.startswith("DRY: ")]
        self.assertEqual(len(dry), 2 + 4 + 3, r.stdout)
        pred = "%s/results/predict_mindboggle" % PROJ_DIR
        a, b = dry[0], dry[1]
        for k in ("env NAME=x_tta_none OUT=%s/x_tta_none TTA=none " % pred, "CKPT=/tmp/x",
                  "bash %s/brain-group-rad/slurm/predict_mindboggle.sh" % PROJ_DIR):
            self.assertIn(k, a, k)
        self.assertNotIn("SKIP_CHECKS=1", a)
        for k in ("env NAME=x_tta_flip8 OUT=%s/x_tta_flip8 TTA=flip8 " % pred, " SKIP_CHECKS=1 "):
            self.assertIn(k, b, k)
        lin = [c for c in dry if " predict.py linear-eval " in c]
        self.assertEqual(len(lin), 4)
        got = [(re.search(r"--task (\w+)", c).group(1), re.search(r"--tta (\w+)", c).group(1)) for c in lin]
        self.assertEqual(got, [("rigid", "none"), ("rigid", "flip8"), ("affine", "none"), ("affine", "flip8")])
        for c, (task, arm) in zip(lin, got):
            self.assertIn("--seed 0 ", c)
            self.assertNotIn("--tta-levels", c)
            self.assertTrue(c.endswith("--out %s/results/linear_eval_mindboggle/x_%s_tta_%s" % (PROJ_DIR, task, arm)), c)
        cmp_calls = [c for c in dry if "scripts/compare_tta.py" in c]
        self.assertEqual(len(cmp_calls), 3)
        self.assertIn("--a %s/x_tta_none/eval_template --b %s/x_tta_flip8/eval_template" % (pred, pred), cmp_calls[0])
        self.assertIn("--kind eval", cmp_calls[0])
        for c, task in zip(cmp_calls[1:], ("rigid", "affine")):
            self.assertIn("--out %s/results/tta_compare/x_%s_linear --kind linear" % (PROJ_DIR, task), c)
        printed = [ln.strip() for ln in r.stdout.splitlines() if ln.strip().endswith("sbatch slurm/eval_mindboggle.sh")]
        self.assertEqual(len(printed), 2)
        self.assertTrue(printed[0].startswith("METHOD=x_tta_none FWD='%s/x_tta_none/fields/{subject_dir}_fwd.nii.gz'"
                                              % pred), printed[0])
        self.assertTrue(printed[1].startswith("METHOD=x_tta_flip8 "), printed[1])
        self.assertIn("A=%s/results/eval_mindboggle/x_tta_none B=%s/results/eval_mindboggle/x_tta_flip8 "
                      "OUT=%s/results/tta_compare/x_allpairs sbatch slurm/compare_tta.sh" % (PROJ_DIR, PROJ_DIR, PROJ_DIR),
                      r.stdout)
        # TTA_LEVELS reaches the flip8 arm and the flip8 linear-eval runs only; PARTS subset
        r = self._run(TTA_MB, DRY_RUN="1", NAME="x", CKPT="/tmp/x", TTA_LEVELS="1", PARTS="linear")
        self.assertEqual(r.returncode, 0, r.stderr)
        dry = [ln for ln in r.stdout.splitlines() if ln.startswith("DRY: ")]
        self.assertEqual(len(dry), 4)
        self.assertEqual([("--tta-levels 1" in c) for c in dry], [False, True, False, True])

    @unittest.skipIf(BASH is None, "bash not available")
    def test_tta_mindboggle_invalid(self):
        for bad in (dict(NAME="x"), dict(CKPT="/c"), dict(NAME="a,b", CKPT="/c"), dict(NAME="a:b", CKPT="/c"),
                    dict(NAME="x", CKPT="/c", PARTS="predict eval")):
            r = self._run(TTA_MB, DRY_RUN="1", **bad)
            self.assertEqual(r.returncode, 2, (bad, r.stdout, r.stderr))
            self.assertIn("ERROR", r.stderr)
            self.assertNotIn("DRY:", r.stdout)
        r = self._run(TTA_CMP, A="a", B="b")
        self.assertEqual(r.returncode, 2)
        self.assertIn("set A, B and OUT", r.stderr)


CASCADE = os.path.join(REPO, "slurm", "r_cascade.sh")
LEVEL_SCRIPTS = tuple(os.path.join(REPO, "slurm", "r_level%d.sh" % k) for k in (4, 2, 1))
CASCADE_ENV = ("S4", "S2", "S1", "SYNTH_PROB", "RUN_NAME_L4", "RUN_NAME_L2", "RUN_NAME_L1", "GROUP_ARGS",
               "EXTRA_ARGS_L4", "EXTRA_ARGS_L2", "EXTRA_ARGS_L1", "STEPS_L4", "STEPS_L2", "STEPS_L1", "LR", "WD",
               "VAL_EVERY", "WARMUP_L4", "DEFORM_START_L4", "DEFORM_RAMP_L4", "CURRICULUM_RAMP_L4", "G_START_L4",
               "G_RAMP_L4", "VAL_GROUP_SIZE_L4", "TF_MEM_LIMIT_MB_L1", "GPU_POLL", "START_LEVEL",
               "CASCADE_FROM_START", "L4_SCRIPT", "L2_SCRIPT", "L1_SCRIPT", "STUB_FAIL_LEVEL", "STUB_BAD_LEVEL",
               "CODE_COMMIT", "ALLOW_DIRTY", "FORCE_RECIPE_CHANGE", "GIT_DIR", "GIT_WORK_TREE")
# Env of the level scripts that r_cascade.sh passes through unchanged (all others are unset per level).
CASCADE_PASSTHROUGH = {"PROJ", "REPO", "DATA", "RESULTS", "SIF", "SCRATCH_ROOT", "SLURM_JOB_ID",
                       "SLURM_CPUS_PER_TASK", "CUDA_VISIBLE_DEVICES"}
GIT = shutil.which("git")
# Stub level script: logs what it sees (incl. REPO, its own path and a tag), fails on request, else
# writes ckpt/last at last-$STEPS.
STUB_LEVEL = r"""#!/bin/bash
echo "L%(k)s RUN_NAME=$RUN_NAME STEPS=$STEPS PROJECT_STEPS=$PROJECT_STEPS CASCADE_FROM=${CASCADE_FROM-UNSET} N=${N-UNSET} FILTERS=${FILTERS-UNSET} INIT=${INIT-UNSET} SYNTH_PROB=$SYNTH_PROB REPO=${REPO-UNSET} SELF=$0 TAG=%(tag)s EXTRA_ARGS=$EXTRA_ARGS" >> "%(log)s"
if [ "${STUB_FAIL_LEVEL:-}" = "%(k)s" ]; then exit 7; fi
mkdir -p "$RESULTS/$RUN_NAME/ckpt/last"
s="$STEPS"
if [ "${STUB_BAD_LEVEL:-}" = "%(k)s" ]; then s=1; fi
printf 'model_checkpoint_path: "last-%%s"\nall_model_checkpoint_paths: "last-%%s"\n' "$s" "$s" > "$RESULTS/$RUN_NAME/ckpt/last/checkpoint"
exit 0
"""


class TestCascadeStatic(unittest.TestCase):
    """cascade-chain: slurm/r_cascade.sh header and level-env coverage."""

    def test_header(self):
        text = _read(CASCADE)
        self.assertNotIn("\r", text)
        self.assertTrue(all(ord(c) < 128 for c in text))
        self.assertNotIn("entre", text.lower())
        for line in ("#SBATCH --job-name=bgr-cascade", "#SBATCH -A qtim", "#SBATCH -p rtx8000",
                     "#SBATCH --gpus=1", "#SBATCH -N 1", "#SBATCH --ntasks-per-node=1",
                     "#SBATCH --cpus-per-task=3", "#SBATCH --mem=128G", "#SBATCH --time=72:00:00",
                     "#SBATCH --output=%s/%%x-%%j.out" % LOGS, "#SBATCH --error=%s/%%x-%%j.err" % LOGS):
            self.assertIn(line + "\n", text)
        self.assertEqual(len(re.findall(r"(?m)^#SBATCH ", text)), 11)
        self.assertIn("set -uo pipefail\n", text)
        for k in ("DO NOT submit without Jay's approval", "at most two active jobs", "mkdir -p %s" % LOGS,
                  "env S4=<S4> S2=<S2> S1=<S1> sbatch slurm/r_cascade.sh", "SYNTH_PROB=0", "Resume rule"):
            self.assertIn(k, text, k)
        self.assertNotIn("FILTERS=", text)
        self.assertNotIn("DECODER_WIDTHS=", text)
        self.assertNotIn("Do not `git pull`", text)
        for k in ("git archive", "scratch/cascade_code/<RUN_NAME_L4>", "cascade_chains/<RUN_NAME_L4>.json",
                  "CODE_COMMIT", "ALLOW_DIRTY=1", "FORCE_RECIPE_CHANGE=1", "can be deleted"):
            self.assertIn(k, text, k)

    def test_level_env_coverage(self):
        """Every env variable a level script reads is either unset per level or passed through."""
        m = re.search(r"LEVEL_ENV_KEYS=\(([^)]*)\)", _read(CASCADE))
        keys = set(m.group(1).split())
        self.assertFalse(keys & CASCADE_PASSTHROUGH)
        for path in LEVEL_SCRIPTS:
            names = set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*):?-", _read(path)))
            with self.subTest(os.path.basename(path)):
                self.assertEqual(names - keys - CASCADE_PASSTHROUGH, set())

    @unittest.skipIf(BASH is None, "bash not available")
    def test_bash_syntax(self):
        r = subprocess.run([BASH, "-n", CASCADE], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True)
        self.assertEqual(r.returncode, 0, r.stderr)


@unittest.skipIf(BASH is None or GIT is None, "bash or git not available")
class TestCascadeRuns(unittest.TestCase):
    """cascade-chain: validation, dry run, and stub level scripts (success chain, failing level,
    rerun skips, ckpt without marker reruns), plus the frozen-code copy (v2): REPO is a temp git
    repo whose committed slurm/r_level{4,2,1}.sh are stubs."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cascade_")
        self.log = os.path.join(self.tmp, "levels.log")
        self.results = os.path.join(self.tmp, "results")
        self.proj = os.path.join(self.tmp, "proj")
        self.repo = os.path.join(self.tmp, "repo")
        self.stubs = {}
        for k in (4, 2, 1):
            p = os.path.join(self.tmp, "stub_l%d.sh" % k)
            with open(p, "w") as f:
                f.write(STUB_LEVEL % {"k": k, "log": self.log, "tag": "hook"})
            self.stubs[k] = p
        os.makedirs(os.path.join(self.repo, "slurm"))
        self.git("init", "-q")
        self.sha1 = self.commit_stubs("v1")
        env = {k: v for k, v in os.environ.items() if k not in CASCADE_ENV + RECIPE_ENV}
        env.update(PROJ=self.proj, REPO=self.repo, RESULTS=self.results,
                   SLURM_JOB_ID="4242", S4="0.5", S2="2", S1="3")
        self.env = env

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def git(self, *args, repo=None):
        r = subprocess.run([GIT, "-C", repo or self.repo, "-c", "user.name=t", "-c", "user.email=t@example.com",
                            "-c", "commit.gpgsign=false"] + list(args),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip()

    def write_stubs(self, tag):
        for k in (4, 2, 1):
            with open(os.path.join(self.repo, "slurm", "r_level%d.sh" % k), "w") as f:
                f.write(STUB_LEVEL % {"k": k, "log": self.log, "tag": tag})
        with open(os.path.join(self.repo, "main.py"), "w") as f:
            f.write("print('%s')\n" % tag)

    def commit_stubs(self, tag):
        """Commit stub level scripts that log TAG=<tag>; returns the full sha."""
        self.write_stubs(tag)
        self.git("add", "slurm", "main.py")
        self.git("commit", "-q", "-m", tag)
        return self.git("rev-parse", "HEAD")

    def code_root(self, sha, chain="r4_cfgB_synth05"):
        return os.path.join(self.proj, "scratch", "cascade_code", chain, sha[:12])

    def state(self, chain="r4_cfgB_synth05"):
        import json
        return json.loads(_read(os.path.join(self.results, "cascade_chains", chain + ".json")))

    def run_cascade(self, stubs=True, **extra):
        e = dict(self.env)
        if stubs:
            e.update(L4_SCRIPT=self.stubs[4], L2_SCRIPT=self.stubs[2], L1_SCRIPT=self.stubs[1])
        for k, v in extra.items():
            if v is None:
                e.pop(k, None)
            else:
                e[k] = v
        return subprocess.run([BASH, CASCADE], env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, stdin=subprocess.DEVNULL)

    def calls(self):
        return _read(self.log).splitlines() if os.path.exists(self.log) else []

    def marker(self, run):
        import json
        return json.loads(_read(os.path.join(self.results, run, "cascade_done.json")))

    def write_ckpt(self, run, step):
        d = os.path.join(self.results, run, "ckpt", "last")
        os.makedirs(d)
        with open(os.path.join(d, "checkpoint"), "w") as f:
            f.write('model_checkpoint_path: "last-%d"\n' % step)

    def test_missing_or_bad_smoothness_exits_2(self):
        for bad in (dict(S4=None), dict(S2=None), dict(S1=None), dict(S1=""), dict(S2="0"), dict(S4="-1"),
                    dict(S4="abc"), dict(START_LEVEL="3"), dict(SYNTH_PROB="1.5"),
                    dict(RUN_NAME_L2="r4_cfgB_synth05")):
            r = self.run_cascade(BGR_DRY_RUN="1", **bad)
            self.assertEqual(r.returncode, 2, (bad, r.stdout, r.stderr))
            self.assertIn("ERROR", r.stderr)
            self.assertNotIn("LEVEL ", r.stdout)
        self.assertEqual(self.calls(), [])

    def test_dry_run(self):
        r = self.run_cascade(stubs=False, BGR_DRY_RUN="1", REPO=REPO)       # the real level scripts
        self.assertEqual(r.returncode, 0, r.stderr)
        out = r.stdout
        head = self.git("rev-parse", "HEAD", repo=REPO)
        root = self.code_root(head)
        self.assertIn("code: frozen %s at %s (clone HEAD %s)" % (head, root, head), out)
        self.assertFalse(os.path.exists(self.tmp + "/proj"))
        self.assertFalse(os.path.exists(self.results))       # nothing created
        env = {k: [ln for ln in out.splitlines() if ln.startswith("ENV L%d: " % k)] for k in (4, 2, 1)}
        cmd = {k: [ln for ln in out.splitlines() if ln.startswith("CMD L%d: " % k)] for k in (4, 2, 1)}
        for k, run, steps, s in ((4, "r4_cfgB_synth05", 16000, "0.5"), (2, "r2_cfgB_synth05", 15000, "2"),
                                 (1, "r1_cfgB_synth05", 4000, "3")):
            self.assertEqual(len(env[k]), 1, out)
            self.assertEqual(len(cmd[k]), 1, out)
            self.assertIn("LEVEL %d: fresh" % k, out)
            e = env[k][0] + " "
            for item in ("RUN_NAME=%s " % run, "STEPS=%d " % steps, "PROJECT_STEPS=%d " % steps, "LR=3e-4 ",
                         "WD=1e-1 ", "VAL_EVERY=500 ", "SYNTH_PROB=0.5 ", "GPU_POLL=false ",
                         "EXTRA_ARGS=--group-config B --lambda-smooth %s " % s):
                self.assertIn(item, e, (k, item))
            self.assertIn("EXTRA_ARGS='--group-config B --lambda-smooth %s' bash %s"
                          % (s, os.path.join(root, "slurm", "r_level%d.sh" % k)), cmd[k][0])
            self.assertIn("REPO=%s " % root, e)
            self.assertNotIn("FILTERS", e)
            self.assertNotIn("DECODER_WIDTHS", e)
        for item in ("WARMUP=1000 ", "DEFORM_START=3000 ", "DEFORM_RAMP=500 ", "CURRICULUM_RAMP=3000 ",
                     "G_START=3000 ", "G_RAMP=3000 ", "VAL_GROUP_SIZE=16 "):
            self.assertIn(item, env[4][0] + " ")
            self.assertNotIn(item.split("=")[0] + "=", env[2][0])
        self.assertNotIn("CASCADE_FROM", env[4][0])
        self.assertIn("CASCADE_FROM=%s/r4_cfgB_synth05/ckpt/last " % self.results, env[2][0])
        self.assertIn("CASCADE_FROM=%s/r2_cfgB_synth05/ckpt/last " % self.results, env[1][0])
        self.assertIn("TF_MEM_LIMIT_MB=42500 ", env[1][0])
        self.assertNotIn("TF_MEM_LIMIT_MB", env[2][0])
        # real-only chain + user extras
        r = self.run_cascade(stubs=False, BGR_DRY_RUN="1", REPO=REPO, SYNTH_PROB="0", RUN_NAME_L4="r4_cfgB_final",
                             RUN_NAME_L2="r2_cfgB_final", RUN_NAME_L1="r1_cfgB_final", EXTRA_ARGS_L1="--lean-losses false")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("CASCADE_FROM=%s/r2_cfgB_final/ckpt/last " % self.results, r.stdout)
        self.assertIn("SYNTH_PROB=0 ", r.stdout)
        self.assertIn("EXTRA_ARGS=--group-config B --lambda-smooth 3 --lean-losses false", r.stdout)

    def test_start_level_needs_source(self):
        for lvl in ("2", "1"):
            r = self.run_cascade(BGR_DRY_RUN="1", START_LEVEL=lvl)
            self.assertEqual(r.returncode, 2, r.stdout)
            self.assertIn("CASCADE_FROM_START", r.stderr)
            r = self.run_cascade(BGR_DRY_RUN="1", START_LEVEL=lvl, CASCADE_FROM_START=os.path.join(self.tmp, "nope"))
            self.assertEqual(r.returncode, 2, r.stdout)
        r = self.run_cascade(BGR_DRY_RUN="1", CASCADE_FROM_START=self.tmp)     # only with START_LEVEL 2 / 1
        self.assertEqual(r.returncode, 2, r.stdout)
        r = self.run_cascade(BGR_DRY_RUN="1", START_LEVEL="2", CASCADE_FROM_START=self.tmp)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("LEVEL 4: not run (START_LEVEL 2)", r.stdout)
        self.assertNotIn("ENV L4:", r.stdout)
        self.assertIn("CASCADE_FROM=%s " % self.tmp, [ln for ln in r.stdout.splitlines()
                                                     if ln.startswith("ENV L2: ")][0])
        self.assertIn("CASCADE_FROM=%s/r2_cfgB_synth05/ckpt/last " % self.results, r.stdout)
        r = self.run_cascade(START_LEVEL="2", CASCADE_FROM_START=self.tmp)     # real (stub) run
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual([c.split()[0] for c in self.calls()], ["L2", "L1"])

    def test_success_chain_and_rerun_skip(self):
        r = self.run_cascade(N="999", FILTERS="", INIT="/x", EXTRA_ARGS="--stale")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        calls = self.calls()
        self.assertEqual([c.split()[0] for c in calls], ["L4", "L2", "L1"])
        for c in calls:                                          # stale submit env never reaches a level
            for item in ("N=UNSET", "FILTERS=UNSET", "INIT=UNSET"):
                self.assertIn(item, c)
            self.assertNotIn("--stale", c)
        self.assertIn("CASCADE_FROM=UNSET", calls[0])
        self.assertIn("CASCADE_FROM=%s/r4_cfgB_synth05/ckpt/last " % self.results, calls[1])
        self.assertIn("CASCADE_FROM=%s/r2_cfgB_synth05/ckpt/last " % self.results, calls[2])
        self.assertIn("EXTRA_ARGS=--group-config B --lambda-smooth 2", calls[1])
        for k, run, steps in ((4, "r4_cfgB_synth05", 16000), (2, "r2_cfgB_synth05", 15000),
                              (1, "r1_cfgB_synth05", 4000)):
            m = self.marker(run)
            self.assertEqual((m["level"], m["steps"], m["checkpoint"], m["job_id"], m["run_name"]),
                             (k, steps, "last-%d" % steps, "4242", run))
        dates = [self.marker(r_)["date"] for r_ in ("r4_cfgB_synth05", "r2_cfgB_synth05", "r1_cfgB_synth05")]
        self.assertEqual(dates, sorted(dates))
        out = r.stdout
        self.assertLess(out.index("=== level 4 start"), out.index("=== level 4 end"))
        self.assertLess(out.index("=== level 4 end"), out.index("=== level 2 start"))
        self.assertLess(out.index("=== level 2 end"), out.index("=== level 1 start"))
        self.assertIn("CASCADE DONE", out)
        # identical resubmission: every level skipped, no level script called
        os.remove(self.log)
        r = self.run_cascade()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.calls(), [])
        for k in (4, 2, 1):
            self.assertIn("=== level %d skip" % k, r.stdout)

    def test_failing_level_stops_chain(self):
        r = self.run_cascade(STUB_FAIL_LEVEL="2")
        self.assertEqual(r.returncode, 7, r.stdout + r.stderr)
        self.assertEqual([c.split()[0] for c in self.calls()], ["L4", "L2"])
        self.assertIn("CASCADE FAILED at level 2", r.stderr)
        self.assertIn("=== level 2 end", r.stdout)
        self.assertNotIn("=== level 1 start", r.stdout)
        self.assertTrue(os.path.isfile(os.path.join(self.results, "r4_cfgB_synth05", "cascade_done.json")))
        self.assertFalse(os.path.exists(os.path.join(self.results, "r2_cfgB_synth05", "cascade_done.json")))
        self.assertFalse(os.path.exists(os.path.join(self.results, "r1_cfgB_synth05")))
        # resubmission continues at level 2
        os.remove(self.log)
        r = self.run_cascade()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual([c.split()[0] for c in self.calls()], ["L2", "L1"])
        # exit 0 with a checkpoint short of STEPS is a failure too (exit 3, no marker)
        shutil.rmtree(self.results)
        os.remove(self.log)
        r = self.run_cascade(STUB_BAD_LEVEL="4")
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertEqual([c.split()[0] for c in self.calls()], ["L4"])
        self.assertFalse(os.path.exists(os.path.join(self.results, "r4_cfgB_synth05", "cascade_done.json")))

    def test_ckpt_without_marker_is_rerun(self):
        self.write_ckpt("r4_cfgB_synth05", 16000)                # finished but no marker (e.g. checker failed)
        self.write_ckpt("r2_cfgB_synth05", 750)                  # interrupted
        # runs exist but no chain state: the commit they ran is unknown -> CODE_COMMIT required
        r = self.run_cascade(BGR_DRY_RUN="1")
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("CODE_COMMIT", r.stderr)
        r = self.run_cascade(BGR_DRY_RUN="1", CODE_COMMIT="HEAD")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("LEVEL 4: resume (ckpt/last at last-16000, no cascade_done.json", r.stdout)
        self.assertIn("LEVEL 2: resume (ckpt/last at last-750", r.stdout)
        self.assertIn("LEVEL 1: fresh", r.stdout)
        r = self.run_cascade(CODE_COMMIT="HEAD")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual([c.split()[0] for c in self.calls()], ["L4", "L2", "L1"])
        self.assertEqual(self.marker("r2_cfgB_synth05")["checkpoint"], "last-15000")
        self.assertEqual(self.state()["code_mode"], "pin")

    # ---- frozen code (v2)
    def test_frozen_fresh_chain_runs_from_code_root(self):
        r = self.run_cascade(stubs=False)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        root = self.code_root(self.sha1)
        self.assertIn("code: frozen %s at %s (clone HEAD %s)" % (self.sha1, root, self.sha1), r.stdout)
        self.assertIn("dirty check: clean", r.stdout)
        calls = self.calls()
        self.assertEqual([c.split()[0] for c in calls], ["L4", "L2", "L1"])
        for k, c in zip((4, 2, 1), calls):
            self.assertIn("REPO=%s " % root, c)
            self.assertIn("SELF=%s " % os.path.join(root, "slurm", "r_level%d.sh" % k), c)
            self.assertIn("TAG=v1 ", c)
        self.assertEqual(_read(os.path.join(root, ".cascade_extract_ok")).strip(), self.sha1)
        self.assertTrue(os.path.isfile(os.path.join(root, "main.py")))
        st = self.state()
        for key, val in (("commit", self.sha1), ("code_root", root), ("code_mode", "fresh"), ("s4", "0.5"),
                         ("s2", "2"), ("s1", "3"), ("synth_prob", "0.5"), ("run_name_l4", "r4_cfgB_synth05"),
                         ("run_name_l2", "r2_cfgB_synth05"), ("run_name_l1", "r1_cfgB_synth05"),
                         ("created_job_id", "4242"), ("last_job_id", "4242"), ("start_level", "4")):
            self.assertEqual(st[key], val, key)
        for run in ("r4_cfgB_synth05", "r2_cfgB_synth05", "r1_cfgB_synth05"):
            self.assertEqual(self.marker(run)["git_rev"], self.sha1)
            self.assertEqual(self.marker(run)["code_root"], root)
        # hooks override the frozen path, REPO is still the frozen copy
        shutil.rmtree(self.results)
        os.remove(self.log)
        r = self.run_cascade()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for c in self.calls():
            self.assertIn("TAG=hook ", c)
            self.assertIn("REPO=%s " % root, c)

    def test_resume_ignores_new_clone_commit(self):
        r = self.run_cascade(stubs=False, STUB_FAIL_LEVEL="2")
        self.assertEqual(r.returncode, 7, r.stdout + r.stderr)
        sha2 = self.commit_stubs("v2")                            # Jay commits / pulls meanwhile
        os.remove(self.log)
        r = self.run_cascade(stubs=False, SLURM_JOB_ID="4343")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        root = self.code_root(self.sha1)
        self.assertIn("code: frozen %s at %s (clone HEAD %s)" % (self.sha1, root, sha2), r.stdout)
        self.assertIn("code source: resume", r.stdout)
        calls = self.calls()
        self.assertEqual([c.split()[0] for c in calls], ["L2", "L1"])
        for c in calls:
            self.assertIn("TAG=v1 ", c)
            self.assertIn("REPO=%s " % root, c)
        self.assertFalse(os.path.exists(self.code_root(sha2)))
        st = self.state()
        self.assertEqual((st["commit"], st["created_job_id"], st["last_job_id"]), (self.sha1, "4242", "4343"))
        self.assertEqual(self.marker("r1_cfgB_synth05")["git_rev"], self.sha1)

    def test_dirty_clone(self):
        with open(os.path.join(self.repo, "slurm", "r_level4.sh"), "a") as f:
            f.write('echo DIRTY >> "%s"\n' % self.log)
        r = self.run_cascade(stubs=False)
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("uncommitted changes", r.stderr)
        self.assertIn("slurm/r_level4.sh", r.stderr)
        self.assertEqual(self.calls(), [])
        self.assertFalse(os.path.exists(self.proj))
        self.assertFalse(os.path.exists(self.results))
        r = self.run_cascade(stubs=False, BGR_DRY_RUN="1")       # dry run reports it, exit 0
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("dirty check: DIRTY", r.stdout)
        self.assertIn("a real run would exit 2", r.stdout)
        r = self.run_cascade(stubs=False, ALLOW_DIRTY="1")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("WARNING: ALLOW_DIRTY=1", r.stderr)
        calls = self.calls()
        self.assertEqual([c.split()[0] for c in calls], ["L4", "L2", "L1"])
        self.assertNotIn("DIRTY", "\n".join(calls))               # the uncommitted change is not frozen
        self.assertEqual(self.state()["allow_dirty"], "1")

    def test_code_commit_pin(self):
        sha2 = self.commit_stubs("v2")
        with open(os.path.join(self.repo, "main.py"), "a") as f:   # dirty tree is irrelevant with a pin
            f.write("# dirty\n")
        r = self.run_cascade(stubs=False, CODE_COMMIT="deadbeef")
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("CODE_COMMIT", r.stderr)
        r = self.run_cascade(stubs=False, CODE_COMMIT=self.sha1[:9])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        root = self.code_root(self.sha1)
        self.assertIn("code: frozen %s at %s (clone HEAD %s)" % (self.sha1, root, sha2), r.stdout)
        self.assertIn("dirty check: skipped (CODE_COMMIT pin", r.stdout)
        for c in self.calls():
            self.assertIn("TAG=v1 ", c)
        self.assertEqual((self.state()["commit"], self.state()["code_mode"]), (self.sha1, "pin"))

    def test_recipe_change_on_resume(self):
        r = self.run_cascade(stubs=False, STUB_FAIL_LEVEL="2")
        self.assertEqual(r.returncode, 7, r.stdout + r.stderr)
        os.remove(self.log)
        for bad in (dict(S2="5"), dict(SYNTH_PROB="0"), dict(RUN_NAME_L1="r1_other")):
            r = self.run_cascade(stubs=False, **bad)
            self.assertEqual(r.returncode, 2, (bad, r.stdout))
            self.assertIn("FORCE_RECIPE_CHANGE", r.stderr)
        sha2 = self.commit_stubs("v2")
        r = self.run_cascade(stubs=False, CODE_COMMIT=sha2)       # commit change needs the force flag too
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertEqual(self.calls(), [])
        r = self.run_cascade(stubs=False, S2="5", FORCE_RECIPE_CHANGE="1")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("WARNING: FORCE_RECIPE_CHANGE=1", r.stderr)
        calls = self.calls()
        self.assertEqual([c.split()[0] for c in calls], ["L2", "L1"])
        self.assertIn("--lambda-smooth 5", calls[0])
        self.assertIn("TAG=v1 ", calls[0])                        # still the recorded commit
        self.assertEqual((self.state()["s2"], self.state()["commit"]), ("5", self.sha1))
        prev = [f for f in os.listdir(os.path.join(self.results, "cascade_chains")) if ".json.prev-" in f]
        self.assertEqual(len(prev), 1, prev)

    def test_missing_or_partial_code_root_is_reextracted(self):
        r = self.run_cascade(stubs=False, STUB_FAIL_LEVEL="2")
        self.assertEqual(r.returncode, 7, r.stdout + r.stderr)
        root = self.code_root(self.sha1)
        self.commit_stubs("v2")
        shutil.rmtree(root)                                       # scratch cleaned
        r = self.run_cascade(stubs=False, BGR_DRY_RUN="1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("extract: needed: git archive %s" % self.sha1, r.stdout)
        self.assertFalse(os.path.exists(root))
        os.makedirs(os.path.join(root, "slurm"))                  # partial extract: no stamp
        r = self.run_cascade(stubs=False, BGR_DRY_RUN="1")
        self.assertIn("extract: partial copy", r.stdout)
        os.remove(self.log)
        r = self.run_cascade(stubs=False)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("extracting %s into %s" % (self.sha1, root), r.stdout)
        self.assertEqual(_read(os.path.join(root, ".cascade_extract_ok")).strip(), self.sha1)
        for c in self.calls():
            self.assertIn("TAG=v1 ", c)
            self.assertIn("REPO=%s " % root, c)

    def test_frozen_dry_run_creates_nothing(self):
        r = self.run_cascade(stubs=False, BGR_DRY_RUN="1")
        self.assertEqual(r.returncode, 0, r.stderr)
        root = self.code_root(self.sha1)
        for item in ("code: frozen %s at %s (clone HEAD %s)" % (self.sha1, root, self.sha1),
                     "code source: fresh (fresh chain: clone HEAD)", "dirty check: clean",
                     "extract: needed: git archive %s | tar -x -C %s" % (self.sha1, root),
                     "bash %s" % os.path.join(root, "slurm", "r_level4.sh")):
            self.assertIn(item, r.stdout, item)
        self.assertFalse(os.path.exists(self.proj))
        self.assertFalse(os.path.exists(self.results))
        self.assertEqual(self.calls(), [])
        self.assertEqual(self.git("status", "--porcelain"), "")
        # after a run: the dry run reports the reuse of the existing copy
        self.assertEqual(self.run_cascade(stubs=False, STUB_FAIL_LEVEL="4").returncode, 7)
        sha2 = self.commit_stubs("v2")
        r = self.run_cascade(stubs=False, BGR_DRY_RUN="1")
        self.assertIn("code: frozen %s at %s (clone HEAD %s)" % (self.sha1, root, sha2), r.stdout)
        self.assertIn("extract: present", r.stdout)
        self.assertIn("dirty check: skipped (resume", r.stdout)




R_LEVEL2 = os.path.join(REPO, "slurm", "r_level2.sh")
R_LEVEL1 = os.path.join(REPO, "slurm", "r_level1.sh")
FGI = os.path.join(REPO, "slurm", "fgi_memory.sh")


@unittest.skipIf(BASH is None, "bash not available")
class TestFrozenGridKnob(unittest.TestCase):
    """fine-grid-integration FEAT-003 (plan D8): FROZEN_INTEGRATION_GRID in r_level2.sh / r_level1.sh
    (unset / empty -> no flag, today's calls; native | current -> one flag just before EXTRA_ARGS;
    anything else -> exit 2). Stub runs as in TestStubRuns."""
    setUp = TestStubRuns.setUp
    tearDown = TestStubRuns.tearDown
    run_script = TestStubRuns.run_script

    def runs(self, path, **kw):
        env = dict(CASCADE_FROM=os.path.join(self.tmp, "ckpt"), EXTRA_ARGS="--lambda-smooth 10")
        env.update(kw)
        return self.run_script(path, **env)

    def test_level_scripts(self):
        from runtime import args as A
        for path, level in ((R_LEVEL2, 2), (R_LEVEL1, 1)):
            with self.subTest(os.path.basename(path)):
                r, base = self.runs(path)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(len(base), 2, base)
                self.assertNotIn("--frozen-integration-grid", "\n".join(base))
                self.assertIn("frozen integration grid: not set", r.stdout)
                r, empty = self.runs(path, FROZEN_INTEGRATION_GRID="")
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(empty, base)
                r, dry = self.runs(path, BGR_DRY_RUN="1")
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertNotIn("--frozen-integration-grid", r.stdout)
                for grid in ("current", "native"):
                    r, calls = self.runs(path, FROZEN_INTEGRATION_GRID=grid)
                    self.assertEqual(r.returncode, 0, r.stderr)
                    self.assertIn("frozen integration grid: %s" % grid, r.stdout)
                    flag = " --frozen-integration-grid %s " % grid
                    self.assertEqual(calls[0].count(flag), 1, calls[0])
                    self.assertEqual(calls[0].count("--frozen-integration-grid"), 1, calls[0])
                    self.assertTrue(calls[0].endswith(flag + "--lambda-smooth 10"), calls[0])
                    self.assertEqual(calls[0].replace(flag, " ", 1), base[0])
                    self.assertEqual(calls[1], base[1])
                    argv = calls[0].split()
                    a = A.parse_args(argv[argv.index("main.py") + 1:])
                    self.assertEqual((a.level, a.frozen_integration_grid, a.lambda_smooth), (level, grid, 10.0))
                r, calls = self.runs(path, FROZEN_INTEGRATION_GRID="bogus")
                self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                self.assertIsNone(calls)
                self.assertIn("FROZEN_INTEGRATION_GRID must be one of: native current", r.stderr)

    def test_static(self):
        for path in (R_LEVEL2, R_LEVEL1):
            text = _read(path)
            with self.subTest(os.path.basename(path)):
                self.assertNotIn("\r", text)
                self.assertTrue(all(ord(c) < 128 for c in text))
                self.assertIn('FROZEN_INTEGRATION_GRID="${FROZEN_INTEGRATION_GRID:-}"\n', text)
                self.assertIn("$SYNTH_ARGS $GRID_ARGS $EXTRA_ARGS)", text)
        self.assertIn("FROZEN_INTEGRATION_GRID", re.search(r"LEVEL_ENV_KEYS=\(([^)]*)\)", _read(CASCADE)).group(1))
        self.assertIn("EXTRA_ARGS_L2='--frozen-integration-grid current'", _read(CASCADE))


@unittest.skipIf(BASH is None, "bash not available")
class TestFgiMemory(unittest.TestCase):
    """fine-grid-integration FEAT-003 (plan D7): slurm/fgi_memory.sh (real-trainer memory check of
    --frozen-integration-grid current vs native at levels 2 and 1)."""
    ORDER = ["l2_native", "l2_current", "l1_native", "l1_current"]

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="slurm_fgi_")
        self.log = os.path.join(self.tmp, "calls.log")
        bindir = os.path.join(self.tmp, "bin")
        os.makedirs(bindir)
        # singularity stub: logs the arguments ('|'-joined); with STUB_WRITE_CKPT=1 a main.py run
        # (--results / --run-name) writes ckpt/last/checkpoint.
        body = ('#!/bin/bash\nprintf "%%s|" "$@" >> "%s"\nprintf "\\n" >> "%s"\nres=""; run=""\n'
                'while [ $# -gt 0 ]; do case "$1" in --results) res="$2";; --run-name) run="$2";; esac; shift; done\n'
                'if [ -n "$run" ] && [ "${STUB_WRITE_CKPT:-}" = 1 ]; then\n'
                '  mkdir -p "$res/$run/ckpt/last"; echo x > "$res/$run/ckpt/last/checkpoint"\nfi\nexit 0\n'
                % (self.log, self.log))
        for name, text in (("singularity", body), ("nvidia-smi", "#!/bin/sh\nexit 0\n"), ("git", "#!/bin/sh\nexit 0\n")):
            p = os.path.join(bindir, name)
            with open(p, "w") as f:
                f.write(text)
            os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.l4 = os.path.join(self.tmp, "l4src")
        os.makedirs(self.l4)
        env = {k: v for k, v in os.environ.items() if k not in RECIPE_ENV + ("CUDA_VISIBLE_DEVICES",)}
        # REPO = this repo: the job runs the real r_level{2,1}.sh dry runs (all container calls are stubs).
        env.update(PATH=bindir + os.pathsep + os.environ.get("PATH", ""), PROJ=self.tmp, REPO=REPO,
                   DATA=os.path.join(self.tmp, "data"), SIF=os.path.join(self.tmp, "x.sif"),
                   SCRATCH_ROOT=os.path.join(self.tmp, "scratch"), L4_SOURCE=self.l4)
        self.env = env
        self.out = os.path.join(self.tmp, "scratch", "fgi", "manual")
        self.runs = os.path.join(self.out, "runs")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_fgi(self, **extra):
        env = dict(self.env)
        for k, v in extra.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
        if os.path.exists(self.log):
            os.remove(self.log)
        r = subprocess.run([BASH, FGI], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, stdin=subprocess.DEVNULL)
        calls = _read(self.log).splitlines() if os.path.exists(self.log) else None
        return r, calls

    def configs(self, stdout):
        out = []
        for ln in stdout.splitlines():
            if ln.startswith("CONFIG "):
                name, cmd = ln[len("CONFIG "):].split(": ", 1)
                out.append((name, cmd))
        return out

    def status_rows(self):
        return [ln.split("\t") for ln in _read(os.path.join(self.out, "status.tsv")).splitlines()]

    def test_header(self):
        text = _read(FGI)
        self.assertNotIn("\r", text)
        self.assertTrue(all(ord(c) < 128 for c in text))
        self.assertNotIn("entre", text.lower())
        for line in ("#SBATCH --job-name=bgr-fgi-mem", "#SBATCH -A qtim", "#SBATCH -p rtx8000",
                     "#SBATCH --gpus=1", "#SBATCH -N 1", "#SBATCH --ntasks-per-node=1",
                     "#SBATCH --cpus-per-task=3", "#SBATCH --mem=128G", "#SBATCH --time=03:00:00",
                     "#SBATCH --output=%s/%%x-%%j.out" % LOGS, "#SBATCH --error=%s/%%x-%%j.err" % LOGS):
            self.assertIn(line + "\n", text)
        self.assertEqual(len(re.findall(r"(?m)^#SBATCH ", text)), 11)
        for k in ("set -uo pipefail\n", SIF, "</dev/null", "DO NOT submit without Jay's approval",
                  "at most two active jobs", "mkdir -p %s" % LOGS, "sbatch slurm/fgi_memory.sh",
                  "r4_cfgB_dice1_smooth2_lr3e-4_wd1e-1/ckpt/last", "collect_memory_runs.py", "--skip-steps 2",
                  "timeout -k 60", "~2 h", "BGR_DRY_RUN=1"):
            self.assertIn(k, text, k)

    def test_dry_run(self):
        import shlex
        from runtime import args as A
        r, calls = self.run_fgi(BGR_DRY_RUN="1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIsNone(calls)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "scratch")))
        configs = self.configs(r.stdout)
        self.assertEqual([n for n, _ in configs], self.ORDER)
        for name, cmd in configs:
            level, grid = name[1], name[3:]
            with self.subTest(name):
                words = shlex.split(cmd)
                self.assertEqual(words[-2:], ["bash", "slurm/r_level%s.sh" % level])
                env = dict(w.split("=", 1) for w in words[:-2])
                src = self.l4 if level == "2" else os.path.join(self.runs, "l2_%s" % grid, "ckpt", "last")
                want = {"RESULTS": self.runs, "RUN_NAME": name, "STEPS": "16", "VAL_EVERY": "8", "CKPT_EVERY": "0",
                        "BN_RECAL_BATCHES": "4", "PROJECT_STEPS": "16",
                        "EXTRA_ARGS": "--group-config B --lambda-smooth 10", "CASCADE_FROM": src}
                if grid == "current":
                    want["FROZEN_INTEGRATION_GRID"] = "current"
                self.assertEqual(env, want)
                # the env reproduces a level-script recipe that parses with the expected values
                os.makedirs(src, exist_ok=True)
                full = dict(self.env, **env)
                full["BGR_DRY_RUN"] = "1"
                d = subprocess.run([BASH, os.path.join(REPO, words[-1])], env=full, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, universal_newlines=True, stdin=subprocess.DEVNULL)
                self.assertEqual(d.returncode, 0, d.stderr)
                main = [ln for ln in d.stdout.splitlines() if ln.startswith("MAIN: python main.py ")]
                self.assertEqual(len(main), 1, d.stdout)
                a = A.parse_args(main[0].split()[3:])
                self.assertEqual((a.level, a.run_name, a.results, a.steps, a.val_every, a.ckpt_every),
                                 (int(level), name, self.runs, 16, 8, 0))
                self.assertEqual((a.bn_recal_batches, a.lambda_smooth, a.group_config, a.cascade_from),
                                 (4, 10.0, "B", src))
                self.assertEqual(a.frozen_integration_grid, "current" if grid == "current" else None)
        shutil.rmtree(os.path.join(self.tmp, "scratch"))
        r, _ = self.run_fgi(BGR_DRY_RUN="1", ONLY="l2_current")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual([n for n, _ in self.configs(r.stdout)], ["l2_current"])
        r, _ = self.run_fgi(BGR_DRY_RUN="1", ONLY="l1_current,l2_native", SMOOTH="4", GROUP_ARGS="")
        self.assertEqual([n for n, _ in self.configs(r.stdout)], ["l2_native", "l1_current"])
        self.assertIn("EXTRA_ARGS='--lambda-smooth 4'", r.stdout)

    def test_guards(self):
        prefix = os.path.join(self.tmp, "pfx", "last-16000")
        os.makedirs(os.path.dirname(prefix))
        with open(prefix + ".index", "w") as f:
            f.write("x")
        r, _ = self.run_fgi(BGR_DRY_RUN="1", L4_SOURCE=prefix)        # checkpoint prefix accepted
        self.assertEqual(r.returncode, 0, r.stderr)
        for extra in (dict(L4_SOURCE=os.path.join(self.tmp, "missing")), dict(ONLY="l2_native,l4_native"),
                      dict(STEPS="abc"), dict(SMOOTH="x"), dict(CONFIG_TIMEOUT="0"),
                      dict(REPO=os.path.join(self.tmp, "norepo"))):
            for dry in ("1", None):
                with self.subTest(extra=extra, dry=dry):
                    r, calls = self.run_fgi(BGR_DRY_RUN=dry, **extra)
                    self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                    self.assertIsNone(calls)
                    self.assertIn("ERROR", r.stderr)
                    self.assertFalse(os.path.exists(os.path.join(self.tmp, "scratch")))

    def test_stub_run_without_level2_checkpoints(self):
        r, calls = self.run_fgi()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(calls), 3, calls)             # two L2 main.py runs + the collector
        mains = calls[:2]
        for c, name in zip(mains, self.ORDER[:2]):
            self.assertTrue(c.startswith("exec|--nv|-B|/scratch|-B|/autofs|-B|/cluster|-B|/space|-B|/vast|%s|"
                                         "python|main.py|" % os.path.join(self.tmp, "x.sif")), c)
            self.assertIn("|--run-name|%s|" % name, c)
            self.assertIn("|--level|2|", c)
            self.assertIn("|--cascade-from|%s|" % self.l4, c)
            self.assertEqual(c.count("--frozen-integration-grid"), 1 if name.endswith("current") else 0)
            self.assertTrue(os.path.isfile(os.path.join(self.runs, name, "gpu_poll.csv")))
        self.assertIn("|--frozen-integration-grid|current|--group-config|B|--lambda-smooth|10|", mains[1])
        self.assertTrue(calls[2].startswith("exec|-B|/autofs|-B|/cluster|-B|/space|%s|python|"
                                            "scripts/collect_memory_runs.py|" % os.path.join(self.tmp, "x.sif")),
                        calls[2])
        for seq in ("|--status|%s|" % os.path.join(self.out, "status.tsv"), "|--runs|%s|" % self.runs,
                    "|--out|%s|" % os.path.join(self.out, "fgi_memory.json"), "|--skip-steps|2|", "|--job|manual|"):
            self.assertIn(seq, calls[2])
        rows = self.status_rows()
        self.assertEqual(rows[0], ["name", "level", "dtype", "widths", "n", "max_pairs", "rc", "elapsed_s", "log"])
        self.assertEqual([x[:7] for x in rows[1:]],
                         [["l2_native", "2", "native", "d12hs", "16", "64", "0"],
                          ["l2_current", "2", "current", "d12hs", "16", "64", "0"],
                          ["l1_native", "1", "native", "d12hs", "4", "64", "-2"],
                          ["l1_current", "1", "current", "d12hs", "4", "64", "-2"]])
        for x in rows[1:]:
            self.assertEqual(x[8], os.path.join(self.runs, x[0] + ".log"))
        self.assertIn("=== l1_native skipped: level-2 source", r.stdout)
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, "scratch", "manual")))

    def test_stub_run_full_chain(self):
        r, calls = self.run_fgi(STUB_WRITE_CKPT="1")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(calls), 5, calls)
        names = [c.split("|--run-name|")[1].split("|")[0] for c in calls[:4]]
        self.assertEqual(names, self.ORDER)
        for c, name in zip(calls[2:4], self.ORDER[2:]):
            grid = name[3:]
            self.assertIn("|--level|1|", c)
            self.assertIn("|--cascade-from|%s|" % os.path.join(self.runs, "l2_%s" % grid, "ckpt", "last"), c)
            self.assertEqual(c.count("--frozen-integration-grid"), 1 if grid == "current" else 0)
        self.assertEqual([x[6] for x in self.status_rows()[1:]], ["0", "0", "0", "0"])
        for name in self.ORDER[:2]:                        # L2 checkpoints kept (L1 sources)
            self.assertTrue(os.path.isfile(os.path.join(self.runs, name, "ckpt", "last", "checkpoint")))
        for name in self.ORDER[2:]:                        # L1 checkpoints deleted
            self.assertFalse(os.path.exists(os.path.join(self.runs, name, "ckpt")))

    def test_collector_exit_code(self):
        stub = os.path.join(self.tmp, "bin", "singularity")
        with open(stub) as f:
            body = f.read()
        with open(stub, "w") as f:
            f.write(body.replace('res=""; run=""\n', 'res=""; run=""\ncase "$*" in *collect_memory_runs*) exit 3;; esac\n'))
        r, calls = self.run_fgi(ONLY="l2_native")
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertEqual(len(calls), 2, calls)
        self.assertIn("collector exit code 3", r.stdout)

    @unittest.skipIf(BASH is None, "bash not available")
    def test_bash_syntax(self):
        for path in (FGI, R_LEVEL2, R_LEVEL1, CASCADE):
            r = subprocess.run([BASH, "-n", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
            self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
