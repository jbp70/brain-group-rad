# Review FEAT-004, iteration 1 (checker, r_level scripts, r_level1 128G / 42500, GPU check s, preview tool)

Verdict: APPROVED (0 HIGH, 0 MEDIUM, 2 NIT).

## Evidence checked
- Test evidence (impl-log.md, "FEAT-004 iteration 1"): bgr_preproc38 `tests.test_sweep_tools tests.test_slurm_scripts`
  Ran 94, OK (baseline 84 + 6 + 4); tf_gpu `tests.test_check_mixed_run tests.test_synth_preview` Ran 46, OK, including
  the NIT 1 test (4 x 25 deform steps, deform:100 PASS / deform:101 FAIL); `bash -n` of all changed/new scripts OK;
  default dry runs byte-compared with `git show HEAD:` (r_level4/2 identical, r_level1 differs only by 40000 -> 42500);
  synth dry-run MAIN/CHECK/POLL lines pasted for all three levels; preview PNG produced locally
  (.agents/tmp/feat004_preview/synth_preview_l4.png). FEAT-004 changes no model/trainer code, so no golden rerun was
  needed (FEAT-003 goldens 0). Evidence is consistent with the code; I did not rerun.
- Checker (tests/gpu/check_mixed_run.py 634-701, 814-817, 873-912): `task_steps` sums `train_<t>/window_steps`
  (run.py 1972 writes it per window per task), synth_frac is window-weighted, tolerance 4 sqrt(p(1-p)/n) with exact
  equality at p 0/1, `bn_recal_synth` compares with `floor(planned[t] * p)`, the same formula and keys the trainer
  writes (run.py 1930-1936). `--min-task-steps` requires trace count 1 and n >= threshold; regex parse with
  argparse error; defaults (None) add no result lines (test_exact_p0_p1_and_absent_keys asserts this).
- Scripts (r_level4/2/1.sh, identical blocks, e.g. r_level1.sh 279-309, 360, 368, 373, 398-419): env defaults and
  validation as specified (need_float is non-negative, awk <= 1, need_choice, label-set regex, MIN_TASK_STEPS reuses
  RATIO_RE); the four `--synth-*` flags only when `is_pos SYNTH_PROB`, placed before `$EXTRA_ARGS`, checker gets
  `--expect-synth-prob`; POLL dry-run line; real run starts nvidia-smi after `mkdir -p $RESULTS/$RUN_NAME`
  (run dir creation is harmless: logging uses `exist_ok=True`, the CASCADE_FROM guard keys on ckpt/last), EXIT trap
  plus `stop_poll` right after main.py. Paired smoke header lines match design 7.2 (r4/r2/r1syn_smoke_p$T,
  CASCADE_FROM chain, MIN_TASK_STEPS values, L1 42500 / 128G). H4: default command lines unchanged
  (test_synth_default_off + byte comparison).
- OD3: r_level1.sh `#SBATCH --mem=128G`, `TF_MEM_LIMIT_MB` default 42500, header reasons (jobs, 94 GB MaxRSS),
  threshold 41500 MB, usage lines 128G, A100 34000 sentence kept; tests updated.
- sweep_level4.py: six names + CUDA_VISIBLE_DEVICES in SCRIPT_ENV_KEYS (only stripped from the login-shell env
  before sbatch; SLURM sets it on the node), py3.6 syntax.
- verify_dali_pipeline check s (1055-1123, registered at 1229-1231, not in the default list) reuses the existing
  render/undo pattern (cf. 333-336), `manifest_files(..., kind="labels", label_set=...)` exists, output order
  (images, T, Rt, labels) matches LinearPipeline (dali_loader.py 558-569); slurm/verify_synth.sh follows policy
  (qtim, rtx8000, 1 GPU, 3 CPUs, 64G, 1 h, approval note).
- scripts/synth_preview.py: GPU hidden with set_visible_devices only, mutable SynthConfig dataclass, synth API names
  and dict keys match data_loading/synth.py (99-157, 240); slurm/synth_preview.sh: basic, qtim, 2 CPUs, 16G, 30 min,
  no --nv, dry run writes nothing (tested). Python 3.8 compatible (ast check logged).
- Labels keyword-only / cascade-skip guard tests: not touched by FEAT-004.

## Findings

### NIT 1: GPU_POLL device index under an unconstrained CUDA_VISIBLE_DEVICES
r_level*.sh (e.g. r_level1.sh 306-309) passes the first CUDA_VISIBLE_DEVICES entry to `nvidia-smi -i`. With SLURM
device cgroups (the usual MLSC setup) this is 0 and correct. Without cgroups, nvidia-smi indexes in PCI order while
CUDA uses FASTEST_FIRST, so on a mixed RTX 8000 / A6000 node the poll could watch the wrong GPU. Check the first
poll CSV's memory.total (49152 for both, so check memory.used tracks the job) or use `--id=<PCI bus id>` later.

### NIT 2: synth_preview has not run in the container on a CPU node yet
The local run used the tf_gpu env with the GPU hidden; the cluster job runs the NVIDIA TF image without `--nv` on
`basic`. TF 2.10 normally falls back to CPU with warnings when libcuda is absent, but the first job is the real test;
read its .err log before relying on the PNGs.
