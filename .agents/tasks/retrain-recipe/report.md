# Retrain recipe: d12hs widths, BN trainable at L4/L2, L1 float32_fine

Workflow wf_8f2ae48e6d1ea33d. Plan: `plan.md` (this folder). Not committed or pushed (the
orchestrator does that). No cluster job submitted.

## Summary

- All three retrain scripts and the level-4 sweep now pass the d12hs width explicitly:
  `--filters 16,64,128,192,256,384 --decoder-widths 128,96,64,16,8`. The env overrides are
  `FILTERS` / `DECODER_WIDTHS` (unset -> d12hs, set but empty -> no flag = ModelConfig default,
  bad format -> exit 2).
- Level 2: `BN_RECAL_FREEZE` default `false` (BN trainable for the whole run), step-0
  recalibration kept (`BN_RECAL_BATCHES 64`).
- Level 1: `DEFORM_DECODER_DTYPE` default `float32_fine`; recal 64 + freeze true and every other
  level-1 default unchanged.
- Levels 4 and 2 keep the fp16 deform decoder (`DEFORM_DECODER_DTYPE` default `policy`, no flag).
- Group sizes unchanged (128/64, 32/16, 8/4).
- **No BN code fix was needed.** The code trace and the new tests show `--bn-recal-freeze false`
  leaves BN training after the recal and after a resume, and that 'full' checkpointing updates
  the trainable statistics exactly once per step while the frozen stage stays untouched.
- Goldens unchanged (no model / trainer code touched).

## Changed files

| File | Change |
|---|---|
| `slurm/r_level4.sh` | Width header paragraph + BN / decoder-dtype note; `FILTERS` / `DECODER_WIDTHS` vars, format check, `$WIDTH_ARGS` before `$MITIG_ARGS $EXTRA_ARGS`; `widths:` echo line |
| `slurm/r_level2.sh` | Same width changes; `BN_RECAL_FREEZE` default `false`; header explains the recal-without-freeze decision; resume-plan line updated |
| `slurm/r_level1.sh` | Same width changes; `DEFORM_DECODER_DTYPE` default `float32_fine` (header paragraph with the m9 NaN cause and the m11 10528396 numbers); N 8 comment notes the d12hs int32 limit (~N 19), N stays 8 |
| `slurm/sweep_level4.py` | `WIDTH_FILTERS` / `WIDTH_DECODER` constants; every config env carries `FILTERS` / `DECODER_WIDTHS` (appended to `CFG_KEYS`, added to `SCRIPT_ENV_KEYS`); docstring note |
| `models/width_presets.py`, `runtime/args.py` | Docstring only: d12hs = the from-scratch retrain width |
| `tests/test_sweep_tools.py` | d12hs env in phase A/B/C configs; env-key regex also catches `${NAME-`; header checks (d12hs, flag line, `BN_RECAL_FREEZE false`, `float32_fine`, `36.95 GiB`); default dry-run lines (d12hs flags, r2 freeze false and no dtype flag, r1 float32_fine + recal 64 + freeze true, linear-only r1 drops the dtype, `policy` drops the flag); `test_mitigations_default_off` expects the r1 dtype flag; new `test_widths_env`, `test_cascade_model_configs_identical` |
| `tests/test_slurm_scripts.py` | `FILTERS`, `DECODER_WIDTHS` added to `RECIPE_ENV` (stripped from the inherited env) |
| `tests/test_checkpointing.py` | New `TestFullCascadeTrainableBN` (point 3) |
| `tests/test_cascade.py` | New `TestCascadeResume.test_main_freeze_false_resume` |

## Level-2 recalibration decision: keep the recal, no freeze

`BN_RECAL_BATCHES=64`, `BN_RECAL_FREEZE=false`. The level-2 trainable stage starts as a copy of
the level-4 weights (`model.copy_weights_from(frozen.model)` in the `main.py` cascade path), so
its BN moving statistics are level-4 statistics computed on different inputs (4 mm vs 2 mm,
pre-warped). With Keras momentum 0.99 these need about 460 steps to fade (0.99^460 is about 1 %).
Until then, the step-0 and early eval-mode validations, the first best-checkpoint picks and the
smoke's VAL_EVERY 100 validations would use mismatched statistics. The recal replaces them with
the exact mean over 64 level-2 batches. That costs about 9 min including its pre/post validation
passes (m8 job 10505028: 515 s) and logs the eval/train gap. Training then keeps updating them,
because nothing freezes them. `BN_RECAL_BATCHES=0` skips the recal; `BN_RECAL_FREEZE=true`
restores the old m8 recipe. Both choices are documented in the r_level2.sh header.

## Point 1: `--bn-recal-freeze false` keeps BN in training mode (code trace)

- `models/layers.py` `BatchNormWithSeparateAffine` (484-495): `stats_frozen=False` by default.
  `call` uses `bn_training = False if self.stats_frozen else training`. The only setter is
  `set_bn_stats_frozen` (552-559).
- `runtime/run.py` `recalibrate_bn` (1753-1854): momentum is set temporarily and restored in
  `finally`. It freezes only inside `if freeze:` (1844-1846: `set_bn_stats_frozen(True)` +
  `rebuild_steps()`). With freeze False, `stats_frozen` stays False and the event records
  `"freeze": False`.
- `main.py` 430-433: recal runs only when `cascade_on and not cascade_resume and
  args.bn_recal_batches > 0`, so it never runs on a resume. `main.py` 407-411 (resume): the
  freeze is re-applied only if the SAVED record has `freeze is True`. A freeze-false run
  therefore resumes with `stats_frozen` False (the same state as the fresh run), and the trained
  statistics come back from ckpt/last.
- Train steps call `model(x_in, training=True)`. The BN backup is restored only for a
  non-finite step. Validation and diagnostic train-mode forwards restore
  `model.non_trainable_variables` afterwards, so validation never moves the statistics.
- Frozen stages (`models/cascade.py`) are built float32 with `gradient_checkpointing='none'` and
  always called with `training=False` plus `stop_gradient`. They are not in the trainable model's
  variables or in the optimizer.

## Point 2: once-per-step BN update under `--gradient-checkpointing full`

- `models/group_net.py` 404-456: deform under 'full' goes through `_call_deform_full`. Encoder
  and decoder are each wrapped in `checkpointing.recompute(fn, bn_layers, training)`; the
  integrators use plain `tf.recompute_grad` (no BN). Linear tasks are never checkpointed (one
  forward = one update).
- `models/checkpointing.py` `_restores_needed` (31-37): snapshot/restore is built iff `training`
  is truthy and some layer has `stats_frozen False`. This is read at trace time, after the recal,
  because freeze calls `rebuild_steps()`. `recompute` (40-64): the first call (forward) updates,
  and every later call (gradient recompute) snapshots, runs and restores.
- With freeze False the layers stay unfrozen, so the restore path is active: one update per
  step. With freeze True no snapshot ops are built and nothing updates.

## Point 3: local tests (tiny synthetic level-2 cascade)

The config is `cascade_config(flip_prob=0.0)`: TINY widths, T 3, linear N 4 / deform N 2,
RATIO3, fp32, a randomised level-4 model copied into both the frozen stage and the trainable
stage. **'full' ran under the 6000 MB local cap; no fallback was needed.**

- `TestFullCascadeTrainableBN.test_freeze_false_updates_once`: recal 4 batches with freeze False.
  The deform recal batch uses images x 1.5, so the following train step on the unscaled batch
  moves the statistics measurably. The test then runs one deform train step under 'full' and
  under the 'integrator' reference (which never recomputes encoder/decoder, so one update by
  construction); the two trainers run one after the other to save memory. Printed result:
  `recal full vs integrator 0.00e+00; after one deform step: full vs integrator 0.00e+00, step change 8.05e-04`.
  So 'full' equals the one-update reference exactly. A double update would differ by about the
  step change (8e-4), far above the 1e-6 tolerance. Every trainable BN array changed. A
  following rigid step changed every encoder + linear-decoder BN array, and a second deform step
  changed every deform-part BN array again. The frozen stage weights (incl. its BN statistics)
  stayed bitwise equal.
- `test_freeze_true_full`: recal with freeze True puts every layer in `stats_frozen`. After
  rigid, affine and deform steps under 'full', all trainable-model BN statistics and the frozen
  stage are bitwise unchanged, with one trace per task.
- `TestCascadeResume.test_main_freeze_false_resume` (main.py, rigid-only TINY, deform decoder
  built): level-4 source 3 steps, then level 2 with `--bn-recal-batches 8 --bn-recal-freeze
  false` for 2 steps. config.json has `bn_recalibration.freeze` False and there is one
  recalibration event. Resume to step 3: still one recal event, the record is unchanged, the
  checkpoint is `last-3`, every trainable encoder BN entry changed (BN still trains after the
  resume, matching the fresh run), and every `frozen_stage/` BN entry is bitwise equal.

## Point 4/5: script tests, widths across levels, goldens

`test_cascade_model_configs_identical` parses the three default dry-run lines with
`runtime.args`, builds `ModelConfig` and round-trips it through JSON like config.json. It then
applies the `check_cascade_config` rule (every field but seed), replicated TF-free.
`r_level4 == r_level2 == r_level1`, with filters (16,64,128,192,256,384), decoder_widths
(128,96,64,16,8), deform_decoder True and T 7 at all three levels. So level 2 cascading from a
level-4 run of these scripts, and level 1 from level 2, passes the check. The goldens pass
unchanged.

## Cluster commands (next jobs; Jay approves each submission)

All three scripts are rtx8000, `-A qtim`, 1 GPU, 3 CPUs, `--mem=64G` (in the SBATCH headers).
Before submitting: commit + push, `git pull` in the cluster clone, check `squeue -u jn85` (at
most two active jobs), and `mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs`.

1. Level-4 smoke (defaults: N 128 / deform 64, d12hs, fp16 decoder, BN trainable):
   ```
   RUN_NAME=r4_smoke STEPS=300 VAL_EVERY=100 WARMUP=50 DEFORM_START=100 DEFORM_RAMP=100 CURRICULUM_RAMP=100 G_START=100 G_RAMP=100 PROJECT_STEPS=16000 PROJECT_VAL_EVERY=500 sbatch --job-name=r4smoke --time=02:00:00 --mem=64G slurm/r_level4.sh
   ```
   Check: `INFO gpu_peak` <= 42000 MiB budget (m11 measured 25.6 GiB at deform N64 d12hs fp16),
   config.json filters / decoder_widths = d12hs, checker PASS.
2. Level-2 smoke after it (pipeline check only, the source is barely trained):
   ```
   RUN_NAME=r2_smoke CASCADE_FROM=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results/r4_smoke/ckpt/best STEPS=200 VAL_EVERY=100 WARMUP=50 PROJECT_STEPS=8000 PROJECT_VAL_EVERY=500 sbatch --job-name=r2smoke --time=03:00:00 --mem=64G slurm/r_level2.sh
   ```
   Check: the `bn_recalibration` event has `freeze` false, the cascade config check passes
   (identical ModelConfig), `gpu_peak` <= 41000 MiB (m11: 27.2 GiB at deform N16 d12hs fp16),
   the BN gap shrinks, checker PASS.
3. Later, level-1 smoke from the level-2 long run (from the r_level1.sh header; float32_fine is
   now the default):
   ```
   RUN_NAME=r1_smoke CASCADE_FROM=$RESULTS/r2_long/ckpt/best STEPS=200 VAL_EVERY=100 WARMUP=50 PROJECT_STEPS=8000 PROJECT_VAL_EVERY=500 sbatch --job-name=r1smoke --time=04:00:00 --mem=64G slurm/r_level1.sh
   ```

## Verification log

All commands were run from PowerShell through `wsl -d Ubuntu` with throwaway scripts under
`.agents/tmp/` (conda `bgr_preproc38` for TF-free checks, `tf_gpu` with
`BGR_TF_GPU_LIMIT_MB=6000 TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 TF_CUDNN_USE_AUTOTUNE=0` for TF).

- V1 syntax (bgr_preproc38): `python -m py_compile slurm/sweep_level4.py models/width_presets.py
  runtime/args.py tests/test_sweep_tools.py tests/test_slurm_scripts.py tests/test_checkpointing.py
  tests/test_cascade.py && bash -n slurm/r_level4.sh && bash -n slurm/r_level2.sh && bash -n
  slurm/r_level1.sh` -> `SYNTAX_OK`.
- V2 TF-free tests (bgr_preproc38): `tests.test_sweep_tools tests.test_slurm_scripts
  tests.test_width_presets` -> **Ran 90 tests, OK**; `tensorflow` not imported.
- V3 dry runs (`BGR_DRY_RUN=1 RESULTS=/tmp/rr_results CASCADE_FROM=/tmp/rr_ckpt`, FILTERS /
  DECODER_WIDTHS unset), each rc 0:
  - r_level4: `MAIN: python main.py --data /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/data/npy_v2 --results /tmp/rr_results --run-name r4_manual --level 4 --group-size 128 --deform-group-size 64 --task-ratio rigid:1,affine:1,deform:2 --steps 16000 --val-every 500 --ckpt-every 250 --optimizer adamw --lr 3e-4 --weight-decay 0 --schedule cosine --warmup-steps 1000 --min-lr-factor 1e-3 --task-grad-norm 12 --p-pair 0.1 --min-pairs n --max-pairs 64 --amp true --split-group-conv true --intensity-aug-device tf --lowres-method tent --deform-decoder auto --task-start-step deform:3000 --task-start-ramp-steps 500 --curriculum true --curriculum-ramp-steps 3000 --global-transform true --global-rot-max-deg 45 --global-trans-max-mm 10 --global-start-step 3000 --global-ramp-steps 3000 --val-pair-chunk 16 --val-png true --num-threads 3 --log-every 25 --log-backend jsonl --val-group-size 16 --gradient-checkpointing full --lean-sampler true --pair-recompute false --lambda-dice-template 1 --lambda-pdice 1 --lambda-ncc-template 1 --lambda-ncc-pair 1 --lambda-smooth 0.5 --p-pdice 0.1 --filters 16,64,128,192,256,384 --decoder-widths 128,96,64,16,8`
  - r_level2: `MAIN: python main.py --data /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/data/npy_v2 --results /tmp/rr_results --run-name r2_manual --level 2 --cascade-from /tmp/rr_ckpt --group-size 32 --deform-group-size 16 --task-ratio rigid:1,affine:1,deform:2 --steps 8000 --val-every 500 --ckpt-every 250 --optimizer adamw --lr 3e-4 --weight-decay 0 --schedule cosine --warmup-steps 500 --min-lr-factor 1e-3 --task-grad-norm 12 --p-pair 0.1 --min-pairs n --max-pairs 64 --amp true --split-group-conv true --intensity-aug-device tf --lowres-method tent --bn-recal-batches 64 --bn-recal-freeze false --cascade-skip-linear 4:0 --cascade-skip-deform 4:0.15 --global-transform true --global-rot-max-deg 45 --global-trans-max-mm 10 --global-start-step 0 --global-ramp-steps 0 --curriculum false --val-pair-chunk 4 --val-png true --num-threads 3 --log-every 25 --log-backend jsonl --gradient-checkpointing full --lean-sampler true --pair-recompute false --lambda-dice-template 1 --lambda-pdice 1 --lambda-ncc-template 1 --lambda-ncc-pair 1 --lambda-smooth 0.5 --p-pdice 0.1 --filters 16,64,128,192,256,384 --decoder-widths 128,96,64,16,8`
  - r_level1: `MAIN: python main.py --data /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/data/npy_v2 --results /tmp/rr_results --run-name r1_manual --level 1 --cascade-from /tmp/rr_ckpt --group-size 8 --deform-group-size 4 --task-ratio rigid:1,affine:1,deform:2 --steps 8000 --val-every 500 --ckpt-every 250 --optimizer adamw --lr 3e-4 --weight-decay 0 --schedule cosine --warmup-steps 500 --min-lr-factor 1e-3 --task-grad-norm 12 --p-pair 0.1 --min-pairs n --max-pairs 64 --amp true --split-group-conv true --intensity-aug-device tf --lowres-method tent --bn-recal-batches 64 --bn-recal-freeze true --bn-recal-validate false --cascade-skip-linear 4:0,2:0.15 --cascade-skip-deform 4:0.15,2:0.15 --global-transform true --global-rot-max-deg 45 --global-trans-max-mm 10 --global-start-step 0 --global-ramp-steps 0 --curriculum false --val-pair-chunk 1 --val-png true --num-threads 3 --log-every 25 --log-backend jsonl --gradient-checkpointing full --lean-sampler true --pair-recompute true --lean-losses true --lambda-dice-template 1 --lambda-pdice 1 --lambda-ncc-template 1 --lambda-ncc-pair 1 --lambda-smooth 0.5 --p-pdice 0.1 --val-image-chunk 1 --gpu-memory-limit-mb 40000 --filters 16,64,128,192,256,384 --decoder-widths 128,96,64,16,8 --deform-decoder-dtype float32_fine`
  - `python3 slurm/sweep_level4.py list` (first config): `RUN_NAME=r4sw_A_lr3e-4_wd0_n128_d64 LR=3e-4 WD=0 N=128 DEFORM_N=64 STEPS=8000 VAL_EVERY=500 WARMUP=1000 DEFORM_START=2000 DEFORM_RAMP=500 CURRICULUM_RAMP=2000 G_START=2000 G_RAMP=2000 VAL_GROUP_SIZE=16 FILTERS=16,64,128,192,256,384 DECODER_WIDTHS=128,96,64,16,8 sbatch -A qtim -p rtx8000 --job-name r4sw0 --time 14:00:00 --mem 64G slurm/r_level4.sh`
- TF tests (tf_gpu, one `python -m unittest ... -v` process per line; log `.agents/tmp/rr_v45.log`):
  - `tests.test_checkpointing.TestFullCascadeTrainableBN` -> Ran 2 tests in 198 s, OK
    (printed line in point 3).
  - `tests.test_cascade.TestCascadeResume.test_main_freeze_false_resume` -> Ran 1 test in 139 s, OK.
  - Goldens `tests.test_trainer.TestLinearOnlyUnchanged tests.test_trainer.TestDeformUnchanged`
    -> Ran 2 tests in 110 s, OK (existing fixtures, `BGR_WRITE_GOLDEN` not set).
  - `tests.test_checkpointing` (whole module, incl. the 2 new tests) -> Ran 10 tests in 449 s, OK.
  - `tests.test_cascade.TestBNRecal tests.test_cascade.TestCascadeResume
    tests.test_cascade.TestFrozenStageTraining` (incl. the new resume test) -> Ran 11 tests in
    731 s, OK (`[T7] exact average: max rel. error 1.76e-07`).
- Full suite not run (about 3 h); last full pass 709 tests at 1d5df0f.
