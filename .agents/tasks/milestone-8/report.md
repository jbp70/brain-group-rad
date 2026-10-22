# Milestone 8 report: level 2 with a frozen level-4 cascade (local implementation)

Status: implemented and verified locally (GTX 1080, WSL, TF memory cap 6000 MB). Not committed, not on the cluster yet. Source of truth: `design.md` v5 (approved, 4 NITs), `requirements.md`, `plan.md`. Feature artifacts with full numbers: `task-m8-level2-cascade/features/FEAT-001..005.json`.

## Files (design section 17)

New:
- `models/cascade.py`: `FrozenStage` (`from_checkpoint` / `from_model` / `from_config`, float32 policy, inference-only, `stop_gradient`), `pool_to_coarser`, `prewarp_linear`, `compose_linear`, `prewarp_deform`, `skip_linear`, `skip_fields` (`cascade_skip` name scope).
- `tests/gpu/measure_level_memory.py`, `slurm/m8_memory.sh`, `slurm/m8_level2_cascade.sh`.
- `tests/test_cascade.py` (29 tests), `tests/test_slurm_scripts.py` (9), `tests/test_measure_memory.py` (1), `tests/test_verify_dali_level.py` (2).
- `.agents/tasks/milestone-8/report.md` (this file).

Modified:
- `runtime/args.py`: `--cascade-from`, `--cascade-skip-prob`, `--bn-recal-batches`, `--bn-recal-freeze`, `--val-pair-chunk` and the section 10 `parser.error` rules.
- `runtime/run.py`: `TrainConfig.cascade_skip_prob` / `val_pair_chunk`, `Trainer(cascade=...)`, cascade train/deform steps with `skip`, cascade validation, `_frozen_baseline_metrics`, `recalibrate_bn`, per-task timing, `composed_mean_disp_mm`.
- `runtime/checkpoint.py`: `load_cascade_source`, `check_cascade_config`, `weight_fingerprint`, `cascade_init_record`, the cross-level branch of `warm_start_model`.
- `main.py`: cascade wiring, fresh vs resume decided once (NIT 4), `frozen_stage` in `RunCheckpoint`, fingerprint check, recalibration and re-freeze, config.json `cascade` / `bn_recalibration`, `S_rel_diff_vs_level4`.
- `tests/gpu/check_mixed_run.py`: `--expect-cascade`, `--expect-bn-recal`, `--expect-skip`, `--grad-balance-rule`, `--grad-balance-from-step`; INFO lines `cascade_gain`, data wait, `g_gap`, `bn_gap`, `grad_balance`.
- `tests/gpu/verify_dali_pipeline.py`: `--level {4,2}`, `_set_level`, `_template_brain`.
- `tests/runtime_helpers.py` (`TINY`, `TINY_ARGS`, `tiny_config`), `tests/test_runtime_infra.py` (+2), `tests/test_check_mixed_run.py` (+10), `tests/test_main.py` (+1), `tests/run_tests_wsl.sh` (both m8 scripts in the `bash -n` loop).
- `preprocessing/export.py` (`LABEL_SET_LEVELS`, synthseg at levels 1 and 2), `preprocessing/cli.py` (help text), `preprocessing/tests/test_export.py`.
- `.agents/tasks/network-design/design.md` (v3.12, design step) and `.agents/tasks/milestone-8/design.md` (NIT 2 in section 7, NIT 4 in section 9, T4 tolerance revision in section 16).

No change to `models/group_net.py`, `models/diffeo.py`, `runtime/losses.py`, the DALI pipelines or `data_loading/params.py`.

## Tests run

- Full local suite (`tests/run_tests_wsl.sh`, once): `Ran 381 tests in 10295 s`, `OK (skipped=4)`, unittest exit code 0 (327 baseline + 54 new; the 4 skips are the 3 DALI tests and dllogger, as before). Wall time 10319 s (2 h 52 min), over the planned ~2 h: `test_cascade` (~30 min) and the cascade main-level tests add ~1 h. The 2 h `timeout` wrapper was paused at 1.5 h so the run could finish. Its step-1 checks all passed too (py_compile and ast 3.8 on 57 files, `main.py --help`, `bash -n` on all six scripts). `test_group_net.TestGradientCheckpointing.test_recompute_matches` (a GPU-noise flake seen in FEAT-001) passed. Plan about 3 h for future full runs.
- Goldens: `[golden] max |diff| 0` and `[golden deform] max |diff| 0` (level 4 bit-identical, FR13), in the full suite and after every `run.py` / `main.py` / `checkpoint.py` change.
- Python 3.8 (`bgr_preproc38`): `py_compile` and `ast` `feature_version=(3, 8)` OK on all 19 new/modified `.py` files; `python main.py --help` OK without TF; `bash -n` OK on both m8 scripts.
- Preprocessing suite: 142 passed, `ALL CHECKS PASSED (bgr_preproc38)`.
- New tests per module: `test_cascade` 29, `test_check_mixed_run` +10, `test_slurm_scripts` 9, `test_runtime_infra` +2, `test_verify_dali_level` 2, `test_main` +1, `test_measure_memory` 1 (54 total), plus `test_synthseg_pyramid_levels_1_2` in the preprocessing suite.

## Integration verification (cross-FEAT, after FEAT-005)

First integration pass (no `impl-review.md` yet). No seam needed a fix, so no code changed and the full suite (FEAT-005, 381 OK) was not rerun.

- `bgr_preproc38`: `py_compile OK`, `ast 3.8 OK 19` on all new/modified `.py` files; `bash -n` OK on `slurm/m8_memory.sh` and `slurm/m8_level2_cascade.sh`; `python main.py --help` and `tests/gpu/check_mixed_run.py --help` OK without TF.
- Seam SLURM -> `runtime.args`: the exact `main.py` argv of `slurm/m8_level2_cascade.sh` (smoke 200/100/50 and long 8000/500/500, both `GRAD_BALANCE` arms, N 16 / DEFORM_N 8) passes `parse_args` (level 2, K 64, freeze true, `val_pair_chunk` 16; gradnorm: `task_grad_norm` 12, weights all 1; deform64: weights deform 64, `task_grad_norm` 0). Every flag the two m8 scripts pass to `measure_level_memory.py`, `verify_dali_pipeline.py` and `check_mixed_run.py` exists there. The m8 defaults and the m7 LNCC source both use default model flags, so `check_cascade_config` sees equal configs.
- Targeted modules + goldens, one `tf_gpu` run (GTX 1080, `BGR_TF_GPU_LIMIT_MB=6000`, autotune off): `tests.test_trainer.TestLinearOnlyUnchanged tests.test_trainer.TestDeformUnchanged tests.test_slurm_scripts tests.test_verify_dali_level tests.test_check_mixed_run tests.test_runtime_infra tests.test_measure_memory tests.test_main tests.test_cascade` -> `Ran 109 tests in 5608 s`, `OK (skipped=1)` (dllogger), exit code 0; `[golden] max |diff| 0 (tolerance 0)` and `[golden deform] max |diff| 0 (tolerance 0)`. Printed values reproduce the FEAT numbers: T1(d) 0.00682, T4 8-voxel margin max 0.0604 / 0.0677, T16b rigid center composed 6.99 vs frozen 12.38 mm (margin 5.40), deform template Dice 0.9715 vs 0.9400 (margin 0.0315).
- Not verifiable locally: the DALI level-2 path with a cascade (real `S_rel_diff_vs_level4`, recalibration on DALI sources, `verify_dali_pipeline.py --level 2`); these run in cluster job (a) and the smoke run.

## Local verification numbers

- T1(d) S rel. Frobenius diff, level 2 vs mode-pooled level 4: 0.0068 (< 0.05).
- T2 render consistency, interior NCC: 0.9998.
- T3 composed group barycenter: rotation 1.2e-6 deg, translation 3e-9 mm; pre-warp NCC 0.99997.
- T4 two-level round trip (level-2 voxels, `random_svf` max 2 vox per level, T = 7): see "Tolerance revision" below.
- T6 off-path identity (p = 0 vs p = 0.5 fed `skip=False`): max |diff| 0 over 9 losses and 94 variable sums, with Grappler's meta optimizer off in the test. With Grappler on, the mere presence of the `SelectV2` skip ops changes Grappler's rewrites of the deform step (8.6e-8 after one step, Adam-amplified), while two p = 0 trainers stay bitwise equal. The mismatch traces to the skip ops' graph, not their values; the NIT 3 fallback tolerance (1e-6) was not used. The graph-op check (no `cascade_skip` op at p = 0) is strict.
- T7 BN recalibration: exact average max rel. error 1.8e-7 (Keras 2.10 re-reads `momentum` per eager call: confirmed). `gradient_checkpointing='both'`: exactly one update per layer and batch. Gap with x3 intensity mismatch, rigid: center 0.354 -> 0.0005 mm, pair 0.494 -> 0.0007 mm.
- T16b cascade beats frozen (train-mode stats): rigid center RMS composed 6.88 vs frozen 12.76 mm (margin 5.88); deform template hard Dice composed 0.973 vs frozen 0.941 (margin 0.032; identity 0.930).
- Local e2e (TINY, overfit-synthetic, level 4 -> level 2 cascade -> resume): no traceback, checker `PASS` twice, one trace per task, fingerprint OK on resume, the three provenance blocks unchanged after resume. Step 150 composed vs frozen: rigid center 18.24 vs 22.12 mm, affine 23.39 vs 26.34 mm, deform template Dice 0.948 vs 0.931; folding 0. BN recal (K 8): rigid center gap 0.623 -> 0.246 mm, deform template Dice gap 0.0049 -> 9e-6. `grad_balance KEEP task-grad-norm` (ratios rigid 1.15, affine 1.10; after resume 1.19 / 1.26). Steady step times rigid ~950 ms, affine ~930 ms, deform ~2570 ms; peak 4556 MB. The level-4 phantom differs from the level-2 phantom, so this mainly shows the residual learns; T16b is the meaningful local evidence.
- T15 local memory (level 2, TINY, N 2, cascade, T = 3): deform with checkpointing `both` peak 3.7-4.6 GB (varies run to run), step ~2.5 s, frozen forward 100 ms, val pass peak 1.3 GB; rigid peak 0.66 GB, step 0.63 s, frozen 64 ms.
- NF1 frozen share: not measured at real widths locally; job (a) has the cascade false configs.

Local limits: a level-2 deform training step does not fit 6000 MB even at TINY N = 2 with T = 7 unless gradient checkpointing is `both`. The local tests therefore use `integration_steps 3` and `gradient_checkpointing='both'` at level 2. On GPU, the int32 index arithmetic of `layers.sample_voxel_coords` runs in host memory (TF int32 kernels are HostMemory), so level-2 warps may be partly CPU-bound. Watch the step times in job (a).

## Memory recommendation

Pending cluster job (a) (`scratch/m8/memory_level2.json`). The group-size rule of design section 11 picks DEFORM_N (cheapest option fitting N = 8 at <= 44 GB, else the largest N >= 4) and N (16 if <= 44 GB without AMP, else the largest N >= 8). The A100 verdict (`peak <= 36 GB`) goes into the script header once measured (currently `peak: unmeasured`).

## Cluster commands (order: memory -> smoke -> long; each needs Jay's approval, max two active jobs)

First, `git pull` in the cluster clone and `mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs`. All three jobs: `-A qtim`, `-p rtx8000`, 1 GPU, 3 CPUs, 64 GB.

```bash
sbatch --job-name=m8mem --time=04:00:00 --mem=64G slurm/m8_memory.sh
RUN_NAME=m8_smoke N=<n> DEFORM_N=<nd> STEPS=200 VAL_EVERY=100 WARMUP=50 sbatch --job-name=m8smoke --time=02:00:00 --mem=64G slurm/m8_level2_cascade.sh
RUN_NAME=m8_level2_cascade N=<n> DEFORM_N=<nd> sbatch --job-name=m8long --time=24:00:00 --mem=64G slurm/m8_level2_cascade.sh
```

- Smoke `--time` (provisional 02:00:00): T_smoke = 1.5 x (200 x s_step + 4 x s_val + s_recal + 600 s), floor 01:30:00, rounded up to 15 min. s_step = (rigid + affine + 2 x deform) / 4 of the median `step_ms` at the chosen N; s_val = sum over the 5 validation sets of n_groups x 2 x 0.5 x median `step_ms` of that task; s_recal = 64 x 0.5 x task step.
- Long `--time` (provisional 24:00:00): T = 1.3 x (8000 x mean step s + 16 x val s), from the smoke run.
- A100: add `-p dgx-a100` (same CPUs and memory) only if the measured peak is <= 36 GB.
- Defaults of `m8_level2_cascade.sh`: `CASCADE_FROM=$RESULTS/m7_three_task_lncc/ckpt/best`, `GRAD_BALANCE=gradnorm` (`--task-grad-norm 12`, no task weights), Dice+LNCC 1/1, `BN_RECAL_BATCHES=64`, freeze true, skip 0, G 45 deg / 10 mm from step 0, curriculum off, 8000 steps, VAL_EVERY 500.

## C4 grad-balance decision rule

The checker (`--grad-balance-rule true`, records with window-end step >= 50) computes per linear task median(`train_<task>/diag_enc_grad_norm_eff`) / median(`train_deform/diag_enc_grad_norm_eff`).
- Both ratios in [0.1, 10]: `grad_balance KEEP task-grad-norm`, keep the gradnorm long run.
- Otherwise `FALLBACK deform:64`, and missing values `UNDECIDED`, which is treated as FALLBACK and the run inspected. If the gradnorm long run is already running: `scancel` our own m8long job, check `squeue -u jn85` shows fewer than two active jobs, and resubmit the long run with `GRAD_BALANCE=deform64` (needs Jay's approval: partition, GPU, CPUs, memory, `--time`).

## G gap at level 2 (fill from the checker's `g_gap` lines)

| Metric | Level 4 (LNCC run) none / G | Level 2 composed none / G | Level 2 frozen none / G |
|---|---|---|---|
| rigid pair_rms_mm | 5.45 / 5.53 | | |
| affine pair_rms_mm | 6.92 / 9.45 | | |
| affine pair_rot_deg | 4.9 / 7.1 | | |
| affine pair_shear_pct | 4.0 / 5.3 | | |
| affine pair_scale_pct | ~same | | |
| rigid / affine center_rms_mm, trans_mm, qr_rot_deg | (from level-4 metrics) | | |

Jay's question (message 11): does the affine G gap shrink at level 2? LUMIR inputs are not perfectly affinely aligned, which sets an absolute floor but does not explain the none-vs-G difference.

## Open decision: label sampling

(B) applies: per-step label sampling is deferred to milestone 9 unless Jay chooses (A) (design section 14). Milestone-8 runs use train18 at level 2. Only the exporter change is implemented (synthseg at levels 1 and 2, no `_4`).

Jay must export / provide (after pulling this change):
1. Train: `env LABEL_SETS=train18,synthseg sbatch --array=0-$((N-1))%2 --export=ALL,SPLIT=train preprocessing/slurm/export_array.sh` (N chunks of 200).
2. T1 val: `env LABEL_SETS=train18,synthseg OPTIONAL_LABEL_SETS=lumir10 sbatch --array=0-0%2 --export=ALL,SPLIT=val,CHUNK_SIZE=100000,INCLUDE_TEMPLATE=0 preprocessing/slurm/export_array.sh`.
3. After both: `sbatch --array=0-0%2 --export=ALL,MERGE_ONLY=1 preprocessing/slurm/export_array.sh`.
4. The core-label list as raw SynthSeg ids (e.g. 4/43 lateral ventricles, 14, 15, 17/53 hippocampi) and K, the number of sampled non-core labels per step (8-12 suggested). The label table is `label_luts.json` (lut `synthseg`).

## Design-review NITs

1. `val_pair_chunk` reaches every `_deform_errors` call: `chunk=None` defaults to `cfg.val_pair_chunk`; call sites pass nothing; the memory script passes `chunk=16`. Test: identity and frozen baselines honour `val_pair_chunk=4` (`TestCascadeValidation`).
2. Recalibration index `b` = global plan index 0..K-1, `controls[t].update(b)` only if `t in self.controls`; docstring and design section 7 step 3 state that `fit` restarts at `update(0)` and that K > 0 changes the data order vs K = 0.
3. T6 identity under `_OpDeterminism` with autotune off: max |diff| 0 held (with Grappler off, see above); fallback tolerance not used.
4. `cascade_resume` decided once in main.py before any model is built; selects `from_config` vs `from_checkpoint`, gates recalibration; `restore_last()` must return True on a cascade resume. Design section 9 states the `--resume false` semantics.

## Implementation-review NITs (APPROVED, 0 HIGH / 0 MEDIUM; left open)

1. Design section 16 T6 states an unconditional max |diff| 0; the identity holds with Grappler's meta optimizer off (with it on, the p > 0 skip graph differs by ~1e-7 per step). Harmless for milestone 8 (p = 0 builds no skip ops).
2. `S_rel_diff_vs_level4` is null for every overfit run, including `--overfit-one-group` on real data; design section 2.1 says null only for overfit-synthetic. Align the design text or compute it on the real-data overfit path.
3. `check_mixed_run.py --expect-cascade` does not require finite `val_rigid_g_frozen` / `val_affine_g_frozen` keys; a missing one only drops the `g_gap ... frozen` INFO line needed for the G-gap table. Check this by hand in the smoke output.
4. `recalibrate_bn` treats the whole encoder as run by every task; a single-task recalibration plan could raise "got no update" if a BN layer were linear-only. Not relevant for the 1:1:2 recipe.

## Deviations and tolerance revisions

- T4 tolerance (design section 16 updated): at a 3-voxel margin the measured round-trip errors were max 0.076 / mean 0.011 (u o u_inv) and max 0.206 / mean 0.015 (u_inv o u), above the provisional 0.05 / 0.01. The stages alone give 0.066 / 0.0074 (upsampled level-4) and 0.031 / 0.0069 (level-2 residual): scaling-and-squaring inverse error; the 0.206 is border clamping. The test uses margin ceil(max|u|) + 3 = 8 voxels (measured 0.060 / 0.011 and 0.068 / 0.015) with max <= 0.1, mean <= 0.02, and composed <= 1.25 x (upsampled-only + residual-only).
- Local tests and e2e use `--integration-steps 3` and `--gradient-checkpointing both` at level 2 (6000 MB cap); main-level tests are rigid-only with `--deform-decoder true`; the e2e covers the three-task path.
- `measure_level_memory.py` has an extra `--integration-steps` option (default = ModelConfig 7; the cluster script does not pass it). The val pass also runs after a training OOM of that config. `m8_memory.sh` uses bash arrays for the calls (quoting-safe).
- `S_rel_diff_vs_level4` is computed only on the DALI path (null in overfit runs).
- `Trainer.__init__` also rejects a cascade of the wrong level, cascade + residual affine, and skip outside [0, 1) (defence in depth).
- Checker INFO lines print with label INFO (the existing m7 `enc_grad_ratio` line too); tally unchanged.
- Cosmetic: TF prints "triggered retracing" warnings in cascade runs (one tf.function per validation key, plus `rebuild_steps()` after the freeze).
