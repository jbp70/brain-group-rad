# 3-level frozen cascade, level-1 training path and `--min-pairs` (milestone 9 phase 2), review pass 2

Phase 2 makes level 1 trainable as the third stage of a frozen 4 -> 2 cascade. `models.cascade.FrozenChain` restores both frozen stages from one milestone-8 level-2 checkpoint (`model` + `frozen_stage`, level-4 fingerprint checked against the m8 `cascade.fingerprint`). The trainable level-1 model has the identical default ModelConfig and copies the level-2 weights. The change also adds an opt-in `--min-pairs n` floor for the linear and deform pair losses, `_frozen4` baselines, `val/wall_s`, per-stage provenance and resume checks, `slurm/m9_level1_cascade.sh`, and checker / memory-tool extensions. Defaults stay bit-identical (goldens max |diff| 0). This pass checks the fixes for the pass-1 findings and looks for new concerns.

Watch for: nothing blocking. Both pass-1 findings are fixed. The MEDIUM `step_time` projection now takes `--project-val-every 500` from the SLURM script (**confirmed**). The NIT docstrings now say "schema-compatible" and note the `_frozen4` val-peak increase (**confirmed**). As before, the level-1 deform chain has not run end to end locally, so the cluster smoke run is its first real test.

**Verdict**: APPROVED (0 HIGH, 0 MEDIUM, 0 NIT)

## High-level view

The pass-1 MEDIUM is fixed in `tests/gpu/check_mixed_run.py`. `_step_time` takes `project_val_every` (0 = the analysed run's `val_every`). The CLI rejects negative values. `slurm/m9_level1_cascade.sh` passes `--project-val-every 500`, so a VAL_EVERY 100 smoke run projects the design's 16 validation passes, not 80. The new `test_project_val_every` fixture uses `val_every` 100. It covers both cadences (80 passes = 4164 s, 16 passes = 1668 s) through `check()` and `main()`, plus the negative-value error. `test_slurm_scripts` asserts the new flag in the script's checker call.

The pass-1 NIT is fixed in the `frozen_chain_standin.py` and `measure_level_memory.py` docstrings. They now say the records are schema-compatible with job 10516237 but that `val_peak_mb` includes U4 / U4_inv / V4 (~+1 GB at level-1 N 4).

A second read of the rest of the diff found no new concerns: chain dispatch, validation baselines, resume fingerprints, `--min-pairs` sampling and the SLURM script. Two points that are easy to miss were checked. BN recalibration reaches the chain through the generic `cascade.linear` / `cascade.deform` interface. Both `ckpt/best` and `ckpt/last` share one `tf.train.Checkpoint`, so the m8 `ckpt/best` holds the `frozen_stage` object that `FrozenChain.from_checkpoint` needs.

<details>
<summary>Issues (0)</summary>

No open issues.

</details>

<details><summary>Details</summary>

### Fix verification

```python
val_every = project_val_every if project_val_every > 0 else args.get("val_every")
n_val = project_steps / float(val_every) if _is_int(val_every) and val_every > 0 else 0.0
```

The script header's long-run `--time` text now matches the checker call: `--project-steps 8000 --project-val-every 500`. The projection still uses 8000 steps when `STEPS` is overridden. That only matters for the smoke run, which is the projection's purpose, and the line is INFO only.

### Coverage carried over from pass 1

T-E1 (scripted level 4 -> 2 -> 1 + resume) trains rigid only, because the local GTX 1080 cannot fit a level-1 deform step (report deviation 2). So these are not exercised end to end before the cluster: `train_deform/n_pairs`, `val_deform_frozen4/*`, `ckpt/best_deform` and the deform PNG. Unit tests cover the deform branches (T-C2 outputs max |diff| 0, T-C6 skip, T-V1 `_frozen` / `_frozen4`). The smoke run's checker asserts all of them (`--expect-cascade-stages 2 --expect-min-pairs n`).

### Test evidence

Recorded in FEAT-001..003 and report.md:

- Targeted suites: 97 + 53 + 39 + 66 + 50 + 46 tests, all OK.
- Goldens: `[golden]` and `[golden deform]` max |diff| 0.
- T-C2 (FrozenChain vs the stand-in): max |diff| 0.
- T-C4 3-level round trip: max 0.103 voxel, mean 0.021.
- T-E1 and the scripted CLI chain with the checker: PASS.
- Full suite: 520 OK, skipped 4, 3 h 18 min.
- Iteration-2 fix: `tests.test_check_mixed_run tests.test_slurm_scripts` 47 OK.

Re-run in this pass (bgr_preproc38, py 3.8.20):

- `py_compile` on the checker, the stand-in, the measurement tool, the two changed test modules, `models/cascade.py`, `runtime/run.py` and `main.py`: OK.
- `bash -n slurm/m9_level1_cascade.sh`: OK.
- `tests.test_slurm_scripts tests.test_check_mixed_run`: Ran 47, OK.

The fix touched only the checker, its tests, the script and docstrings, so the goldens and the full suite were not re-run.

</details>

<details>
<summary>File map</summary>

- `models/cascade.py`: `PREV_LEVEL {2: 4, 1: 2}`, `CHAIN_LEVEL`, `_build_frozen_model(cfg, level)`, `FrozenStage` additions, `FrozenChain`, `skip_to`.
- `runtime/run.py`: `_n_stages` dispatch, chain train / val branches, `_frozen4` baselines, `val/wall_s`, `TrainConfig.min_pairs`, `train_<task>/n_pairs`.
- `runtime/args.py`: `--cascade-from` at level 1, `--min-pairs`, `--p-pair` range check.
- `runtime/checkpoint.py`: `CASCADE_SOURCE_LEVEL` dict, `fingerprints_match`, `load_cascade_source(path, level)` with `upstream_cascade`.
- `main.py`: level-1 fresh / resume paths, per-stage fingerprints, `checkpoint_extra()`.
- `data_loading/postprocess.py`: `MIN_PAIRS`, `num_pairs` / `sample_pairs` with the cycle branch.
- `models/memory_model.py`, `scripts/memory_model.py`: `min_pairs` mirror.
- `tests/gpu/check_mixed_run.py`: `--expect-cascade-stages`, `--expect-min-pairs`, `--memory-budget-mb`, `--project-steps`, `--project-val-every` (pass-2 fix), deform64 label.
- `tests/gpu/frozen_chain_standin.py`, `tests/gpu/measure_level_memory.py`: `n_stages`, `*_stages`, `--min-pairs` axis, corrected docstrings (pass-2 fix).
- `slurm/m9_level1_cascade.sh` (new), `tests/run_tests_wsl.sh` (`bash -n` loop).
- `.agents/tasks/network-design/design.md`: v3.15.
- Tests: `test_cascade`, `test_main`, `test_postprocess`, `test_trainer`, `test_check_mixed_run`, `test_slurm_scripts`, `test_runtime_infra`, `test_memory_model`, `test_measure_memory`.

Full diff: `git diff` plus the untracked `slurm/m9_level1_cascade.sh`.

</details>
