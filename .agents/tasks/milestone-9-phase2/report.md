# Milestone 9, phase 2: report (3-level frozen cascade, level-1 training, minimum pair count)

Design: `design-m9-phase2.md` v3 (review NITs 1-3 applied). Features: FEAT-001 (`--min-pairs`), FEAT-002 (`FrozenChain`, level-1 paths), FEAT-003 (checker, SLURM script, end-to-end chain, docs). Nothing was committed or pushed; the cluster was not touched.

## Implemented (per design section)

- **2 Cascade** (`models/cascade.py`): `PREV_LEVEL = {2: 4, 1: 2}`, `FrozenChain` (levels 4 + 2 from ONE m8 checkpoint: objects `model` + `frozen_stage`, level-4 fingerprint checked against the m8 `cascade.fingerprint`; `from_checkpoint` / `from_config` / `from_models`; `linear(_stages)`, `deform(_stages)` = the stand-in math verbatim), `skip_to` (skip of the last frozen stage, recipe 0), `FrozenStage` additions, `_build_frozen_model(cfg, level=...)` (NIT 2).
- **3 Trainer** (`runtime/run.py`): `self._n_stages` dispatch (0 / 1 / 2), chain branches in the train steps and validation, `val_<set>_frozen` (4 + 2) and `val_<set>_frozen4` (level 4 alone) baselines, `val/wall_s` at level 1 only, `TrainConfig.min_pairs`, `train_<task>/n_pairs` (only with `--min-pairs n`).
- **4 Args / checkpoint / main**: `--cascade-from` with `--level 1` (level-2 cascade source, `upstream_cascade`), `fingerprints_match`, level-1 fresh run (cascade record with `stages`, `S_rel_diff_vs_level2`, `skip_stage` 2; checkpoint keys `frozen_stage_4` / `frozen_stage_2`), level-1 resume with per-stage fingerprint checks, `_load_cascade_source(..., level)` (NIT 3).
- **5 Minimum pair count** (`data_loading/postprocess.py`, `models/memory_model.py`, measurement tool): `--min-pairs {1,n}`; k = min(max(N, round(p N(N-1))), N(N-1)), random N-cycle at k = N, same single RNG draw; `--p-pair` range check.
- **8 SLURM**: `slurm/m9_level1_cascade.sh` (rtx8000, `-A qtim`, 1 GPU, 3 CPUs, 64G, 48 h; env N 16 / DEFORM_N 4 / MIN_PAIRS n / GRAD_BALANCE gradnorm|deform64 / LEAN_LOSSES / PREFETCH / SKIP_PROB 0 / BN_RECAL_BATCHES 64; resume-safe `CASCADE_FROM` guard; no width flags; header with the measured facts, A100 / grad-balance / prefetch / MaxRSS rules, time formulas, submit lines and resume plan). Added to the `bash -n` loop of `tests/run_tests_wsl.sh`.
- **9 Checker** (`tests/gpu/check_mixed_run.py`, stdlib only): `--expect-cascade-stages {1,2}`, `--expect-min-pairs {none,1,n}` with `expected_pairs` (float32 mirror; NIT 1 comment on `deform_group_size`), `--memory-budget-mb` (INFO `gpu_peak`), `--project-steps` (INFO `step_time`) with `--project-val-every` (validation cadence of the projected run, default 0 = the analysed run's `val_every`; the m9 script passes 500, so the smoke's VAL_EVERY 100 no longer projects 80 passes instead of 16), `_frozen4` gain / g_gap / `deform_extra` INFO lines, `S_rel_diff_vs_level2`, grad-balance label `KEEP current balance (deform64)` for `task_grad_norm == 0`. Defaults reproduce the milestone-8 checks (all existing fixtures unchanged).
- **13 design.md**: v3.14 -> **v3.15** (status line, 4.7 "As specified for milestone 9 phase 2" + skip-0 deviation, 6.1 / 6.2 `--min-pairs`, 9 level-1 4-pair estimate 35-38 GB + A100 rule, 11 row 9). The brief said v3.13 -> v3.14, but the file was already v3.14 after phase 1.

## Deviations

1. design.md version number (above).
2. **T-E1 trains rigid only, level 1 with AMP + `--split-group-conv`** (design: `rigid:1,deform:1`, `--amp false`). On the local GTX 1080 at the 6000 MB cap the deform train step fits neither at level 2 (fp32, default sampler: OOM in the host-pinned allocator on the int32 sampler indices, measured in this step) nor at level 1 (even TINY, N 2, AMP + `full` + lean sampler + split + lean losses, T 1; FEAT-002 probe), and the float32 level-1 build pass with the deform decoder exceeds the cap. The deform decoder is still built at every level (identical ModelConfigs). Consequences: `train_deform/n_pairs`, `val_deform_frozen4/*`, `ckpt/best_deform` and `val_png/deform_step0000005.png` are not asserted end to end; the level-1 deform chain branches are covered by `tests/test_cascade.py` (T-C2 outputs max |diff| 0, T-C6 skip graph / eager, T-V1 deform `_frozen` / `_frozen4` from the chain's frozen fields) and the checker's deform paths by fake run dirs. With AMP the three fresh level-1 steps were loss-scale skips (`nonfinite_step_skipped`); steps 4-5 after the resume trained (checker `finite_steps` counts the resumed process: 0).

## Test evidence (local WSL, GTX 1080, 6000 MB)

| Command | Result |
|---|---|
| FEAT-001: `tests.test_postprocess tests.test_runtime_infra tests.test_memory_model tests.test_measure_memory` (tf_gpu) | Ran 97, OK (skipped 1) |
| FEAT-001: `tests.test_trainer` (tf_gpu) | 53 tests OK (T-G1 fixed and re-run 3/3) |
| FEAT-002: `tests.test_cascade` | Ran 39, OK (40.5 min) |
| FEAT-002: `tests.test_runtime_infra tests.test_main tests.test_measure_memory` | Ran 66, OK (skipped 1) |
| Goldens (FEAT-001 and FEAT-002) | `[golden] max \|diff\| 0 (tolerance 0)`, `[golden deform] max \|diff\| 0 (tolerance 0)` |
| T-C2 FrozenChain vs stand-in | max \|diff\| 0 for every output |
| T-C4 3-level round trip (level-1 voxels, margin 13) | u o u_inv max 0.1029 voxel (mean 0.0210), u_inv o u max 0.0997 |
| FEAT-003: `tests.test_check_mixed_run tests.test_postprocess` (tf_gpu) | Ran 50, OK (56 s) |
| FEAT-003: `tests.test_slurm_scripts tests.test_check_mixed_run` (bgr_preproc38, TF not imported) | Ran 46, OK |
| FEAT-003: `tests.test_main.TestLevel1CascadeMain` (T-E1) | Ran 1, OK; measured 416 s (design estimate 8-10 min) |
| py_compile + ast (3, 8), `bash -n` | OK (checker, changed tests, `slurm/m9_level1_cascade.sh`, `tests/run_tests_wsl.sh`); script ASCII, LF |
| Integration: py_compile + ast (3, 8) on all 20 changed .py (bgr_preproc38), `bash -n` on `slurm/m9_level1_cascade.sh` and `tests/run_tests_wsl.sh`, goldens (148 s) | OK; `[golden] max \|diff\| 0`, `[golden deform] max \|diff\| 0` |
| Full suite (`tests/run_tests_wsl.sh`, integration step) | Stage 1 OK (py3.8 checks on 75 files, `bash -n` on 8 scripts, 65 TF-free tests without TF); stage 2 Ran 520 tests, OK (skipped=4), unittest wall 11906 s (3 h 18 min); goldens max \|diff\| 0. Baseline was 477 OK; no failures, no fixes needed |
| Code-review fixes (iteration 2): MEDIUM `--project-val-every` + `test_project_val_every` (fixture `val_every` 100: 80 passes = 4164 s vs 16 passes = 1668 s); NIT stand-in / measurement docstrings ("schema-compatible", `val_peak_mb` includes the `_frozen4` fields) | py_compile + ast (3, 8) on the 5 changed .py, `bash -n` (ASCII, no CR); `tests.test_check_mixed_run tests.test_slurm_scripts` Ran 47, OK (bgr_preproc38). Only the checker, its tests, the SLURM script and docstrings changed (no shared module), so goldens and the full suite were not re-run; the 520-test result above stands (now 521 tests with the new one) |

Scripted CLI chain (`main.py` level 4 -> 2 -> 1 + resume, TINY, N 2, rigid, `--gpu-memory-limit-mb 6000`, then `python tests/gpu/check_mixed_run.py <run3> --tasks rigid --steps 5 --expect-cascade true --expect-cascade-stages 2 --expect-min-pairs n --expect-bn-recal true --memory-budget-mb 36864 --project-steps 8000`): exit 0, `check_mixed_run: PASS`, wall 466 s. Final step 5 (untrained TINY overfit, values only show the plumbing): val_rigid center / pair 21.115 / 41.481 mm, `_frozen` (4 + 2) 21.116 / 41.483, `_frozen4` 21.116 / 41.483; `cascade_stages` level 4 key `frozen_stage` origin step 3, level 2 key `model` origin step 3; `n_pairs_rigid` 2 (N 2); one `bn_recalibration` event (55 s); `INFO gpu_peak 3710 MiB (3.62 GiB) vs A100 budget 36 GiB: fits`; `INFO step_time` (run before `--project-val-every` existed; uses the run's own cadence): `per task rigid 17.69 s; ... val/wall_s median 37.64 (first 35.96, 2 passes); recal 55.0 s; projected 8000 steps ... = 314535 s` (local TINY numbers include tracing; meaningless for the cluster).

The real level-1 size (default widths, linear N 16 / deform N 4, 160x224x192) cannot run locally; the smoke run on the cluster is the first real level-1 test.

## Code review

`code-review.json`: verdict **APPROVED** (pass 2; high 0, medium 0, nit 0). Pass-1 MEDIUM (`--project-val-every`) and NIT (stand-in / measurement docstrings) are fixed.

## Open NITs

- None from the code review. Carried over from phase 1 (still to watch on the cluster): f0d1 lean+split N4 forward bound with margin 44.4 > 44 GiB (not used here: phase 2 runs default widths). Cosmetic, from m8: the checker's grad-balance label now reads `KEEP current balance (deform64)` for `task_grad_norm == 0` (fixed in this phase).
- Caveat (review): the level-1 deform chain is first exercised end to end by the cluster smoke run (local T-E1 is rigid only, see Deviations 2).

## Changed / new files (`git status --short`, not committed)

```
 M .agents/tasks/network-design/design.md
 M data_loading/postprocess.py
 M main.py
 M models/cascade.py
 M models/memory_model.py
 M runtime/args.py
 M runtime/checkpoint.py
 M runtime/run.py
 M scripts/memory_model.py
 M tests/gpu/check_mixed_run.py
 M tests/gpu/frozen_chain_standin.py
 M tests/gpu/measure_level_memory.py
 M tests/run_tests_wsl.sh
 M tests/test_cascade.py
 M tests/test_check_mixed_run.py
 M tests/test_main.py
 M tests/test_measure_memory.py
 M tests/test_memory_model.py
 M tests/test_postprocess.py
 M tests/test_runtime_infra.py
 M tests/test_slurm_scripts.py
 M tests/test_trainer.py
?? .agents/tasks/milestone-9-phase2/
?? slurm/m9_level1_cascade.sh
```

Suggested commit message:

```
feat: m9 phase 2 - 3-level frozen cascade (L4+L2) with level-1 training, --min-pairs n, level-1 checker and slurm/m9_level1_cascade.sh
```

Cleanup: `.agents/tmp` empty, `__pycache__` removed under models/ data_loading/ runtime/ tests/ scripts/, no local run outputs (results/ only has the old `m4_local`), no local scratch/.

## Open items for Jay

- **D6 skip probability:** the recipe uses 0 (as in m8), design.md 4.7 says 0.15 for the full level. Confirm 0 or set `SKIP_PROB=0.15`.
- **Order:** git pull -> smoke -> long. A100 only if the smoke's `INFO gpu_peak` is <= 36 GiB. If MaxRSS of the smoke > 56 GB, use `--mem=96G` for the long run. Grad balance from the smoke's `grad_balance` line (KEEP -> gradnorm, else `GRAD_BALANCE=deform64`); data wait > 0.10 -> `PREFETCH=2`. If the smoke OOMs: `LEAN_LOSSES=true`, then `MIN_PAIRS=1`.

## Cluster commands (design section 8; need Jay's approval, account qtim, at most two active jobs)

```bash
cd /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad && git pull --ff-only
mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
# smoke (rtx8000, 1 GPU, 3 CPUs, 64G, 4 h)
RUN_NAME=m9_l1_smoke STEPS=200 VAL_EVERY=100 WARMUP=50 sbatch --job-name=m9smoke --time=04:00:00 --mem=64G slurm/m9_level1_cascade.sh
# long (rtx8000, 48 h; --time from the smoke's INFO step_time line)
RUN_NAME=m9_level1_cascade sbatch --job-name=m9long --time=48:00:00 --mem=64G slurm/m9_level1_cascade.sh
# A100 variant (only if the smoke peak <= 36 GiB)
RUN_NAME=m9_level1_cascade sbatch -p dgx-a100 --job-name=m9long --time=36:00:00 --mem=64G slurm/m9_level1_cascade.sh
```
