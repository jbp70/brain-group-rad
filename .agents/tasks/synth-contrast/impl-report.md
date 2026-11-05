# Implementation report: synthetic-contrast training (synth-contrast)

Workflow wf_8d6aeb00dacc1ac8. Design `.agents/tasks/synth-contrast/design.md` v4 (APPROVED) + review NITs 1/2 +
orchestrator decisions OD1-OD6 (`plan.md`). Base HEAD 4bfe227; all work is UNCOMMITTED on `main` (the orchestrator
commits/pushes). Per-command log: `.agents/tasks/synth-contrast/impl-log.md`.

## 1. What was built

### Files

| Area | Files |
|---|---|
| Exporter (FEAT-001) | `preprocessing/luts.py` (`LUTS["synthfine"]`, `synthfine_to_train18_table()`, `synthfine_cortex_ids()`, `CORTEX_TRAIN18_ID`), `preprocessing/export.py` (`synthfine` at levels 1/2/4, `nested_block_mode`, `synthfine_level`, export-check consistency rule), `preprocessing/add_labels.py` (new: add-only files + manifest phases), `preprocessing/cli.py` (`export-add-labels`), `preprocessing/slurm/add_labels_array.sh` (new) |
| Synthesis core (FEAT-002) | `data_loading/params.py` (`SynthConfig`, `SYNTH_SEED_OFFSET` 104729, `SYNTH_DEVICES`, `SYNTH_COLUMNS`), `data_loading/synth.py` (new), `data_loading/postprocess.py` (`synth=None`, keyword-only `labels=` / `synth_override=`, `tf.cond`, linear render with T_mm^-1, deform fine->train18 gather) |
| Data / trainer / CLI (FEAT-003) | `data_loading/dali_loader.py` (label reader in `LinearPipeline`), `data_loading/data_module.py` (`synth_label_set`), `runtime/args.py` (5 `--synth-*` flags), `runtime/run.py` (`TrainConfig.synth`, `Trainer._pp_call`, uint8 signatures, synth logging, checkpointed generator, BN-recal synthetic slots), `main.py` (config record + resume guard `_check_synth_config`) |
| Checker / scripts / tools (FEAT-004) | `tests/gpu/check_mixed_run.py` (`--expect-synth-prob`, `--min-task-steps`, `bn_recal_synth`), `slurm/r_level{4,2,1}.sh` (SYNTH_* / GPU_POLL / MIN_TASK_STEPS), `slurm/r_level1.sh` (128G / 42500), `slurm/sweep_level4.py`, `tests/gpu/verify_dali_pipeline.py` (check s1-s3), `slurm/verify_synth.sh` (new), `scripts/synth_preview.py` (new), `slurm/synth_preview.sh` (new) |
| Post-training eval (FEAT-005) | `runtime/fixed_data.py` (`group_images_fn` hooks), `data_loading/synth.py` (`EVAL_PRESETS` 11, `preset_params`), `runtime/eval_run.py` (new), `eval_synth.py` (new), `scripts/compare_synth_eval.py` (new), `slurm/eval_synth.sh` (new) |
| Tests | new: `preprocessing/tests/test_add_labels.py`, `tests/test_synth.py` (25), `tests/test_synth_preview.py` (2), `tests/test_eval_synth.py` (14); extended: `preprocessing/tests/test_export.py`, `test_slurm_scripts.py`, `test_cli.py`, `run_tests_wsl.sh`; `tests/runtime_helpers.py`, `test_postprocess.py`, `test_data_loading.py`, `test_runtime_infra.py`, `test_main.py`, `test_trainer.py`, `test_cascade.py`, `test_check_mixed_run.py`, `test_sweep_tools.py`, `test_slurm_scripts.py`, `tests/run_tests_wsl.sh` |

### Training flags (`main.py`, `runtime/args.py`)

| Flag | Default | Limits / notes |
|---|---|---|
| `--synth-prob` | 0 | [0, 1]; 0 = off: no op, no generator, no checkpoint key, no config.json `synth` record. > 0 needs `--intensity-aug-device tf` (or `--intensity-aug false`); refused with `--overfit-one-group` or `--shuffle-mode buffer` |
| `--synth-jitter` | 0.08 | [0, 0.5], per-image per-label relative jitter of the group means / stds |
| `--synth-device` | gpu | gpu \| cpu (memory fallback, same values) |
| `--synth-label-set` | synthfine | `^[A-Za-z0-9_-]+$`; manifest label set read as fine labels |
| `--synth-hier-dev-cortex` | 0.25 | [0, 0.5] (OD1); per-parcel mean deviation of the 72 cortical fine ids; other ids 0.1 |

Other `SynthConfig` defaults (no flag): std_max 0.08, FWHM MR U(0, 3) mm / PET U(4, 8) mm (PET family p 0.5), FWHM image
jitter 0.10, bias grid 4 with std <= 0.3, log-gamma std 0.25, noise std <= 0.03, re-mask p 0.5, hier_dev 0.1, robust
normalisation at p99.5. With synthesis on, config.json gets a top-level `synth` record (asdict of SynthConfig) and
metrics.jsonl gets `train_<t>/synth_frac`, `loss_synth`, `loss_real`; the BN-recal event gets `synth_batches` /
`synth_batches_used`.

### Checker (`tests/gpu/check_mixed_run.py`)

| Flag | Meaning |
|---|---|
| `--expect-synth-prob p` | per task the `window_steps`-weighted mean of `train_<t>/synth_frac` within 4 sqrt(p(1-p)/n_task_steps) of p (exact at p 0 / 1); FAIL if no synth_frac is logged; with `--expect-bn-recal true` and p > 0 also `synth_batches[t] == floor(planned[t] p)` |
| `--min-task-steps rigid:20,affine:20,deform:80` | per listed task FAIL unless trace count 1 AND n_task_steps (sum of `train_<t>/window_steps`) >= threshold (OD2: 4 records x 25 steps passes deform:100, fails deform:101) |

### Script env (`slurm/r_level{4,2,1}.sh`)

| Env | Default | Effect |
|---|---|---|
| `SYNTH_PROB` | 0 | > 0 adds `--synth-prob/--synth-jitter/--synth-device/--synth-label-set` to MAIN (right before EXTRA_ARGS) and `--expect-synth-prob` to CHECK; at 0 the command lines are byte-identical to HEAD (r_level4 / r_level2 verified) |
| `SYNTH_JITTER` | 0.08 | |
| `SYNTH_DEVICE` | gpu | gpu \| cpu |
| `SYNTH_LABEL_SET` | synthfine | |
| `GPU_POLL` | false | true: host `nvidia-smi` memory poll every 5 s into `$RESULTS/$RUN_NAME/gpu_poll.csv` while main.py runs |
| `MIN_TASK_STEPS` | empty | passed to the checker as `--min-task-steps` |

`slurm/r_level1.sh` (OD3): `#SBATCH --mem=128G`, `TF_MEM_LIMIT_MB` default 42500 (L1 MaxRSS ~94 GB, 64G OOM-killed in
validation; 40000 MB OOM in d1 backward). Its default MAIN differs from HEAD only by `--gpu-memory-limit-mb 42500`.
`slurm/sweep_level4.py`: SCRIPT_ENV_KEYS gained the six names + CUDA_VISIBLE_DEVICES.

### `python -m preprocessing export-add-labels`

`--export-root R` (required), `--phase files|manifest` (required), `--out-root O` (required for files), `--label-set`
(default synthfine, only supported set), `--start S --end E` (files only; half-open [S, E) over the
export_manifest.csv data rows, E clipped to the row count, S >= rows -> 0-case status file and exit 0; negative,
S > E, or `--start/--end` with `--phase manifest` -> exit 2). Files phase writes only `labels_synthfine{,_2,_4}.npy`
per case after verifying LUT(synthfine_k) == train18_k against the existing files (never modifies a file); status
`R/add_labels/synthfine/rows-<S>-<E|end>.json`; exit 1 if any case failed. Manifest phase: read-only pre-check
(merge of parts == manifest), re-check of every case, backup `R/backup_pre_synthfine_<UTC>/`, append-only edit of the
six label columns in parts + manifest + LUTs, merge reproduction, diff check (`R/add_labels/synthfine/manifest_diff.json`),
restore on any late failure; a second run prints "nothing to do". `export-check` verifies the synthfine rule.

Wrapper `preprocessing/slurm/add_labels_array.sh`: `PHASE` files|manifest|check (default files; check =
`export-check --sample 0`), `CHUNK` 500, `LABEL_SET` synthfine; basic partition, account qtim, 8G, 1 CPU, 2 h,
`%2` array cap, runs through `preprocessing/run_in_container.sh` (no GPU).

### Post-training eval (`eval_synth.py`)

`--runs RUN [RUN]` (first = real-only, second = mixed), `--level 1|2|4` (1), `--out DIR` (required, not inside a
run dir), `--presets` (comma list, default all 11: random_0..random_3, t2_like, flair_like, fdg_like, amyloid_pos,
amyloid_neg, tau_like, t1_like), `--data` (default: each run's saved args.data), `--sanity-gate on|off` (on).
Restores `<run>/ckpt/last` read-only; exit 5 if the restore does not reproduce the logged validation (mm metrics
within 1e-2, Dice within 1e-3); exit 2 on usage / level mismatch / missing synthfine. Output `eval_synth.json`,
`eval_synth.csv`, `sstat.txt`. Wrapper `slurm/eval_synth.sh` (env RUNS required, LEVEL, OUT, PRESETS, DATA).

## 2. Test results

| Suite | Result |
|---|---|
| Preprocessing (`preprocessing/tests/run_tests_wsl.sh bgr_preproc38`) | 167 passed, 0 failed, 0 skipped (118 s); bash -n of all preprocessing scripts and `--help` loop incl. export-add-labels OK; "ALL CHECKS PASSED (bgr_preproc38)" |
| Full suite stage 1 (bgr_preproc38) | py_compile OK; ast py3.8 OK (119 files), py3.6 OK (2 login helpers); TF-free imports of data_loading / runtime.args / `main.py --help` / `eval_synth.py --help` OK; bash -n OK (20 scripts); TF-free unittests 181 OK without TF |
| Full suite stage 2 (tf_gpu, one process per module) | **910 tests, 4 skipped, 0 failing invocations**; suite status 0; unittest wall time 24221 s (6 h 44 min; whole run 01:40-08:25) |
| Skips | test_curriculum 1, test_data_loading 2 (DALI), test_runtime_infra 1 (pre-existing / DALI-only); the two deferred memory tests ran in their own processes |
| Goldens | `[golden] max |diff| 0 (tolerance 0)` (TestLinearOnlyUnchanged), `[golden deform] max |diff| 0 (tolerance 0)` (TestDeformUnchanged) |
| Seam fixes in integration | none needed |

New / touched modules in the full run (all exit 0): test_synth 25, test_synth_preview 2, test_eval_synth 14, test_trainer
74, test_cascade 52 (cascade-skip guards TestSkip / TestChainTrainer / TestStageSkipLevel2 / TestStageSkipChain unchanged,
plus synth-on twins), test_postprocess 36, test_main 44, test_check_mixed_run 44, test_data_loading 30,
test_runtime_infra 38, test_slurm_scripts 46, test_sweep_tools 52, test_tf_intensity_aug 20.

Trace-once and memory-neutrality evidence (tests.test_synth / tests.test_trainer):
- Post-process trace once on the drawn path: tf.function with input signature, p 0.5, seed 3, 8 calls -> one trace,
  synth True and False both seen (linear rigid and deform).
- OneHot depth multiset on the full graph-def (outer graph + library functions) identical with synthesis on vs off,
  linear and deform, with and without tent TF aug; negative control (fine one-hot depth 103) makes it differ.
  Trainer train-step multisets on == off: rigid {4: 1}, affine {4: 1}, deform {19: 1}.
- synth_override=False bit-equal to synth=None (rigid, affine, deform); SynthConfig(prob=0) vs default over 4 three-task
  trainer steps under op determinism: bit-equal.

E2E tiny trainer runs (TestSynthTrainer.test_e2e_p1_and_p0; level 4, N 4, narrow model, rigid:1 affine:1 deform:2,
8 steps, flips on):
- p 1: every step finite, trace_counts {rigid 1, affine 1, deform 1}; final losses rigid 2219.33, affine 1198.53,
  deform 0.414402; synth == 1 every step; window logs synth_frac 1 + loss_synth, no loss_real.
- p 0 (synth None): every step finite, trace_counts {rigid 1, affine 1, deform 1}; final losses rigid 2214.32,
  affine 1198.6, deform 0.415442; no synth key anywhere. (Values vary in the last digits between runs: GPU float noise,
  no op determinism.)
- p 0.5 (12 steps): history flags == the simulated per-task generator draws; window synth_frac in (0, 1); loss_real /
  loss_synth = means over the matching steps.

Not run locally (cluster only): verify_dali_pipeline check s (DALI + GPU, via `slurm/verify_synth.sh`); a cascade
(L2/L1) run through eval_synth; a passing sanity gate on real logged validation.

## 3. Deviations from the design

- D-a `add_labels_array.sh` runs through `preprocessing/run_in_container.sh` instead of a hand-written
  `singularity exec`: `preprocessing/tests/test_slurm_scripts.py` requires that pattern for every preprocessing SLURM
  script. Same effect (container, no `--nv`, binds incl. /autofs /cluster /space).
- D-a' (consequence) the export verification (design 3.4: a `singularity exec ... export-check` on the login node) is
  a wrapper phase `PHASE=check` = `export-check --sample 0`, so it runs on a basic compute node, not on mlsc-login.
- D-b The synthesis-on forced-skip twins and the BN-recal wiring tests are in `tests/test_cascade.py`
  (TestSynthSkip, TestSynthRecal, TestSynthSkipChain; they need its fixtures), not in tests/test_synth.py.
- D-c GPU checks s1-s3 have their own job `slurm/verify_synth.sh` (levels 4 and 1 ->
  `$PROJ/scratch/verify_synth/verify_synth_l<k>.json`), so the r_level*.sh default command lines stay byte-identical.
  Check s is not in verify_dali_pipeline's default `--checks` (it needs the synthfine export).
- D-d `scripts/synth_preview.py` has a CPU-only basic-partition wrapper `slurm/synth_preview.sh` (no compute on
  mlsc-login). Its env uses `N_GROUPS` (not `GROUPS`, a bash special variable).
- D-e `eval_synth.py` has `--data` (default: the run's saved args.data) and `--sanity-gate on|off` (default on; off
  only for local tests whose fake run has no comparable logged validation).
- Found during implementation:
  - `export-npy` refuses `synthfine` with `--no-remap-labels` (raw ids would break the nested vote).
  - `Trainer._synth_on = any(pp.synth is not None ...)` instead of `cfg.synth is not None`: the post-processes treat
    SynthConfig(prob=0) as off, so a TrainConfig with p 0 stays on today's path (main.py passes None at p 0 anyway).
  - `_pp_call` raises a ValueError naming the task when synthesis is on and a linear input list has no label entry
    (instead of an IndexError).
  - `_check_synth_config` (resume guard) compares the JSON round trip of asdict(SynthConfig) (tuples -> lists).
  - BN-recal event: `synth_batches` = floor(plan.count(t) p) per task; `synth_batches_used` counts used slots with
    override True (non-finite slots excluded).
  - The checker's `synth_frac` is the window mean over every executed step (finite or not), weighted by
    `window_steps`; `loss_synth` / `loss_real` cover finite steps only and are absent from windows without such steps.
  - Eval presets: named presets have no random parcel deviation (fixed class means + DK parcel overrides), re-mask on,
    gamma 1, std / jitter / FWHM jitter / bias / noise drawn as in training; MR presets draw the group FWHM from
    U(0, 3) mm; `random_*` = training draws with SynthConfig defaults (prob forced 1). Contrast seed per group
    = TASK_SEED[task] * 100000 + g (rigid 1, affine 2, deform 3). Synthetic eval sets are built on `/cpu:0`; AMP runs
    are validated under the mixed_float16 policy.
  - `sweep_level4.SCRIPT_ENV_KEYS` also lists CUDA_VISIBLE_DEVICES (r_level4 reads it for GPU_POLL; the env-key
    coverage test scans every `${NAME:-`).

## 4. Commands

`P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg`, `R=$P/results`. Everything below needs the
commit pushed and `git pull` in `$P/brain-group-rad`. All jobs use account qtim; check `squeue -u jn85` before each
submission (at most two active jobs; Jay's own jobs count).

### (a) Jay: synthfine export (basic partition, CPU only)

Data: sources `$P/data/processed/lumir/<case>/labels_synthseg.nii.gz`, export `$P/data/npy_v2` with
`export_manifest.csv` (3423 data rows = 3384 train + 39 val). The `%2` array cap keeps the array within the 2-active-job
rule (an array counts its running tasks).

```
P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
cd $P/brain-group-rad && git pull
mkdir -p $P/logs
squeue -u jn85
N=$(( $(wc -l < $P/data/npy_v2/export_manifest.csv) - 1 )); echo $N    # 3423 expected
# phase files: 500 rows per task (7 tasks for 3423 rows), at most two tasks at once, ~2-4 h wall time
sbatch --array=0-$(( (N + 499) / 500 - 1 ))%2 --export=ALL,CHUNK=500,LABEL_SET=synthfine preprocessing/slurm/add_labels_array.sh
# only after the array has FINISHED (squeue -u jn85 shows none of its tasks); check
# $P/data/npy_v2/add_labels/synthfine/rows-*.json and $P/logs/bgr_add_labels_*.out/.err for failed cases
# (rerunning a task is safe: identical files are kept):
sbatch --array=0-0%2 --export=ALL,PHASE=manifest,LABEL_SET=synthfine preprocessing/slurm/add_labels_array.sh
# verification after the manifest phase finished (all cases; includes the synthfine consistency rule):
sbatch --array=0-0%2 --export=ALL,PHASE=check preprocessing/slurm/add_labels_array.sh
```

Expected results: every `rows-*.json` with 0 failed; manifest log "3423 rows, 3423 changed, columns changed
label_dtypes,label_levels,labels,labels_l1,labels_l2,labels_l4 ..., 0 problems" and a backup dir
`$P/data/npy_v2/backup_pre_synthfine_<UTC>/`; check log "checked 3423 cases: OK".

### (b) Orchestrator: preview, optional DALI check, paired smokes

**The smokes and the preview need the synthfine export (files phase AND manifest phase of (a)) to exist first.**
With synthesis on, the data module resolves `labels_synthfine_<k>.npy` for every train row at startup and fails with a
ValueError naming the case otherwise. The p 0 smokes do not read synthfine, but they must run on the same commit as
their p 0.5 twins, so submit the pairs only after the export.

Preview PNGs (basic partition, CPU, 2 CPUs, 16G, 30 min; levels 4/2/1 -> `$P/scratch/synth_preview/<job>/l<k>.png`;
env N 8, SEED 0, N_GROUPS 2, SPLIT train, LABEL_SET synthfine):
```
cd $P/brain-group-rad && mkdir -p $P/logs
sbatch slurm/synth_preview.sh
```

Optional DALI checks s1-s3 (rtx8000, 1 GPU, 3 CPUs, 64G, 1 h; output `$P/scratch/verify_synth/verify_synth_l{4,1}.json`):
```
sbatch slurm/verify_synth.sh
```

Paired smokes, in the order L4 -> L2 -> L1. Script headers set rtx8000, `-A qtim`, 1 GPU, 3 CPUs. Each pair = 2 jobs =
the 2-job limit, so submit a pair only when no other job of yours is active and the previous pair has finished.
Smoothness 0.5 in both runs of every pair (memory and trace checks do not depend on it). Rule: if the p 0 run of a pair
fails (OOM, int32 limit, checker FAIL), STOP the chain and report to Jay as a config B / recipe issue; the p 0.5 result
is then only informative and no further level runs.

```
P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
R=$P/results
cd $P/brain-group-rad
# L4 pair (64G, 2 h; deform / curriculum / G from step 100)
for P_ in 0 0.5; do T=${P_/./}
  RUN_NAME=r4syn_smoke_p$T STEPS=300 VAL_EVERY=100 WARMUP=50 DEFORM_START=100 DEFORM_RAMP=100 CURRICULUM_RAMP=100 G_START=100 G_RAMP=100 PROJECT_STEPS=16000 PROJECT_VAL_EVERY=500 WD=1e-1 SYNTH_PROB=$P_ GPU_POLL=true MIN_TASK_STEPS=rigid:20,affine:20,deform:80 EXTRA_ARGS="--group-config B --lambda-smooth 0.5" sbatch --job-name=r4synsm$T --time=02:00:00 --mem=64G slurm/r_level4.sh
done
# L2 pair, after both L4 smokes finished (64G, 3 h)
for P_ in 0 0.5; do T=${P_/./}
  RUN_NAME=r2syn_smoke_p$T CASCADE_FROM=$R/r4syn_smoke_p$T/ckpt/last STEPS=200 VAL_EVERY=100 WARMUP=50 PROJECT_STEPS=15000 PROJECT_VAL_EVERY=500 WD=1e-1 SYNTH_PROB=$P_ GPU_POLL=true MIN_TASK_STEPS=rigid:20,affine:20,deform:40 EXTRA_ARGS="--group-config B --lambda-smooth 0.5" sbatch --job-name=r2synsm$T --time=03:00:00 --mem=64G slurm/r_level2.sh
done
# L1 pair, after both L2 smokes finished (128G, 4 h)
for P_ in 0 0.5; do T=${P_/./}
  RUN_NAME=r1syn_smoke_p$T CASCADE_FROM=$R/r2syn_smoke_p$T/ckpt/last STEPS=200 VAL_EVERY=100 WARMUP=50 PROJECT_STEPS=4000 PROJECT_VAL_EVERY=500 LR=3e-4 WD=1e-1 TF_MEM_LIMIT_MB=42500 SYNTH_PROB=$P_ GPU_POLL=true MIN_TASK_STEPS=rigid:20,affine:20,deform:40 EXTRA_ARGS="--group-config B --lambda-smooth 0.5" sbatch --job-name=r1synsm$T --time=04:00:00 --mem=128G slurm/r_level1.sh
done
```

(The loop variable is `P_` so it does not overwrite the project path `P`; the r_level script headers and design 7.2
use `P` with `R` spelled out in full, which is equivalent. `T` is `0` or `05`: run names `r<k>syn_smoke_p0` /
`r<k>syn_smoke_p05`.)

Pass criteria per level (design 7.2): checker PASS incl. trace_once, `--min-task-steps`, and `--expect-synth-prob` for
the p 0.5 run; `gpu_peak_mb` of p 0.5 within 0.5 GB of p 0; `max(memory.used) <= memory.total - 512` MiB in
`gpu_poll.csv` and within 0.5 GB of p 0; sacct MaxRSS < 64 GB (L2) / < 128 GB (L1); at L2 and L1
`bn_recalibration.synth_batches` == {rigid 8, affine 8, deform 16}; non-finite steps only in the AMP warm-up;
step-time overhead within design 6.5 (L4/L2 <= 5 %, L1 <= 7 %).

### (c) Later: synthetic-contrast evaluation (rtx8000, 1 GPU, 3 CPUs, 128G, 8 h)

After the real-only config B cascade (Jay's) and the mixed `r1_cfgB_synth05` run exist (needs the synthfine val labels
from (a)):
```
P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
R=$P/results
cd $P/brain-group-rad && mkdir -p $P/logs
RUNS="<Jay's real-only config B L1 run> $R/r1_cfgB_synth05" sbatch --job-name=evalsyn --time=08:00:00 --mem=128G slurm/eval_synth.sh
```
Optional env: LEVEL (1), OUT (default `$R/eval_synth/<job>`), PRESETS (comma list, empty = all 11), DATA (empty = the
runs' saved args.data). Compare with `python scripts/compare_synth_eval.py` (see its `--help`). Dry run:
`BGR_DRY_RUN=1 RUNS="a b" bash slurm/eval_synth.sh`.
