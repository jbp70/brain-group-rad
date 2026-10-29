# Review: retrain recipe (d12hs widths, BN trainable at L4/L2, L1 float32_fine)

Verdict: APPROVED, no blockers. I reviewed the uncommitted diff (10 files + `.agents/tasks/retrain-recipe/`) and the coder's verification log. I did not rerun the suites.

## Checks against the brief's reviewer list

- **BN trainable at L2 with freeze false (code path).** `BatchNormWithSeparateAffine.call` (models/layers.py 493) passes `training` through unless `stats_frozen`. The only setter is `set_bn_stats_frozen`. `recalibrate_bn` (runtime/run.py 1844-1846) calls it only under `if freeze:`, and it restores the temporary momentum in `finally`. `main.py` 430 runs the recal only on fresh cascade runs. `main.py` 407-411 re-applies the freeze on resume only if the saved record has `freeze is True`. So a freeze-false run trains BN on the fresh run and after a resume. `r_level2.sh` now defaults to `BN_RECAL_FREEZE=false` with `BN_RECAL_BATCHES=64`; the dry run shows `--bn-recal-batches 64 --bn-recal-freeze false`.
- **One BN update per step under 'full'.** In `models/checkpointing.py`, `recompute` runs call 1 (the forward) as is. Later calls snapshot, recompute and restore. The restore is built iff `training` is true and some layer is unfrozen, and this is read at trace time (freeze triggers `rebuild_steps()`). The new `TestFullCascadeTrainableBN.test_freeze_false_updates_once` compares 'full' with the 'integrator' reference, which does one update by construction: they agree to 0.0 while one step moves the statistics by 8e-4. The test passed locally under 'full' within the 6000 MB cap.
- **Frozen stages untouched.** Both new checkpointing tests assert the frozen-stage weights stay bitwise equal, including its BN statistics. `test_main_freeze_false_resume` checks the `frozen_stage/` checkpoint entries across the resume. `test_freeze_true_full` shows nothing moves after a freeze-true recal (one trace per task).
- **Widths identical across the three scripts and the sweep.** All three scripts default to `FILTERS="${FILTERS-16,64,128,192,256,384}"` and `DECODER_WIDTHS="${DECODER_WIDTHS-128,96,64,16,8}"`, with a format check and `$WIDTH_ARGS` emitted before `$MITIG_ARGS $EXTRA_ARGS`. The sweep's `_config` adds the same strings to every config env, and they are part of `CFG_KEYS` and `SCRIPT_ENV_KEYS`. `test_cascade_model_configs_identical` parses the three dry-run lines into `ModelConfig`, round-trips them through JSON, and finds them equal in every field but seed, which matches `check_cascade_config` (runtime/checkpoint.py 237). `deform_decoder_dtype` is a `GroupRegLevelModel` constructor kwarg (main.py 281), not a `ModelConfig` field, so the L1 float32_fine default does not break the L2 -> L1 config check.
- **L1 float32_fine default.** `DEFORM_DECODER_DTYPE="${DEFORM_DECODER_DTYPE:-float32_fine}"` in `r_level1.sh`. The dry run emits `--deform-decoder-dtype float32_fine` and keeps recal 64, freeze true, N 8 / 4, TF limit 40000, pair recompute, lean losses and the val chunks. A linear-only ratio drops the flag with a NOTE. L4 and L2 keep `policy` (no flag), and tests assert that.
- **Goldens bit-identical.** The only non-script, non-test code changes are docstrings (`models/width_presets.py`, `runtime/args.py`). `TestLinearOnlyUnchanged` and `TestDeformUnchanged` passed with the existing fixtures.

## Verification evidence (from the coder's log)

- py_compile and `bash -n` passed for every changed file.
- 90 TF-free tests passed (sweep tools, slurm scripts, width presets).
- The dry runs of all three scripts and the sweep show the final main.py lines.
- TF tests passed: the new checkpointing class (2 tests), the new cascade resume test, the goldens, the whole `test_checkpointing` module (10 tests) and the related cascade classes (11 tests).
- The level-2 recal decision (keep recal 64, no freeze) is justified in the script header and the report. The smoke commands follow the cluster rules (rtx8000, qtim, 1 GPU, 3 CPUs, 64G).

## Findings

None.
