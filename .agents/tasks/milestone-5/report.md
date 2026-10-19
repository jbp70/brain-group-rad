# Milestone 5 report: mixed rigid + affine schedule, affine metrics, affine/mixed overfit, curriculum follow-ups, G warm-up, warm start

Outcome: rigid and affine now train interleaved with one shared Adam, with one trace per task. Affine validation reports scale, shear and rotation-factor errors. Mixed runs keep the best checkpoint on `val/mean_pair_rms_mm`. On the synthetic phantom, the affine overfit and the mixed rigid:affine overfit both reach < 1 mm in eval mode. Review-curriculum issues 2-5 are fixed, and the orchestrator relay (G warm-up, `--init-weights`) is in. Local full suite: **194 tests, OK, 3 skips** (`test-log.txt`). Nothing was committed or pushed, nothing touched Martinos, and the glacier commands below were not run.

## Files changed

| File | Change |
|---|---|
| `runtime/metrics.py` | `qr_factors`, `affine_matrix_errors` (scale_pct, shear_pct, qr_rot_deg), `group_errors(..., affine=)`, `group_metrics(..., affine=)`, `mean_pair_rms` |
| `runtime/run.py` | `MEAN_PAIR_METRIC`, `resolve_best_metric`. `TrainConfig.best_metric` defaults to None (auto). `Trainer.best_metric`. Affine val errors. `val/mean_pair_rms_mm` in `validate()`. Per-task overfit windows (`window_center_rms_by_task`). `control_g_rot_max_deg` log. Docstring on the mixed schedule and the optimizer slots |
| `runtime/args.py` | `--best-metric` default None (auto), `mean_pair_rms_mm` alias, `--global-start-step`, `--global-ramp-steps`, `--init-weights`. The overfit help text now covers multiple tasks |
| `runtime/curriculum.py` | `GlobalTransformSampler(start_step, ramp_steps)`, `fraction`, `rot_max_at`, `trans_max_at` |
| `runtime/checkpoint.py` | `peek_last_step`, `resolve_checkpoint`, `source_config_path`, `load_model_weights` |
| `main.py` | Overfit mode builds one fixed group per trained task. The resumed step's control values are written before the pipelines start. `--init-weights` (with the resume exclusion and logged in config.json). `fit_summary.json`. The overfit summary is per task |
| `data_loading/params.py`, `data_loading/dali_loader.py` | `G_DALI_ANGLE_SIGN = -1` (with derivation), `control_global_dali_angle`. `control_global_params` negates the angle |
| `tests/gpu/verify_dali_pipeline.py` | Check i is strict on the signed axis and angle, and the recompose sub-check is gone. Check j gates `barycenter_equivariance_max_abs` with `--tol-bary` (default 1e-6) |
| `tests/gpu/run_overfit_glacier.sh` | New `dryrun_mixed` mode (writes under `scratch/m5`) |
| `tests/gpu/check_mixed_run.py` (new) | Stdlib-only checker of a finished mixed run dir |
| Tests | `test_runtime_losses_metrics` (TestAffineMetrics, 4), `test_trainer` (TestMixedTasks 2, `test_g_warmup_logged`), `test_runtime_infra` (best-metric defaults, `peek_last_step`, warm start), `test_curriculum` (G angle sign, G warm-up, warm-up flags), `test_main` (resume control values, `--init-weights`, affine overfit, mixed overfit) |
| `.agents/tasks/network-design/design.md` | v3.6: 3.2 (G angle sign, G collapse note, warm-up / warm start), 5.2 (mixed tasks, combined best metric), 7 (affine metrics) |
| `.agents/tasks/milestone-5/` | This report, `test-log.txt`, overfit logs and summaries, `overfit_run.sh`, `run_some.sh`, `run_py.sh`, `measure_bary.py` |

## API changes

- **Mixed schedule.** `--task-ratio rigid:1,affine:1` was already parsed. The task of step s comes from `TaskSchedule.draw(s)`, which depends only on (seed, s).
  - Each task has its own `tf.function`.
  - The single optimizer is built on all trainable variables in `Trainer.__init__` (`O.build_optimizer`). The scale/shear head therefore has Adam m/v slots before any affine step.
  - Rigid steps drop that head's None gradients statically, so the first affine step creates no variable and does not retrace.
  - Tested: 30 steps starting with 3 rigid steps. The step order equals the seeded draw, both tasks trace once, the optimizer's variable count stays at 2 × trainable, and the scale/shear head is unchanged until the first affine step. This holds with flips off and controls off, and with flips on plus curriculum and G on.
- **Affine metrics** (`runtime.metrics`). Each linear part is factored as L = Q·U·D with `qr_factors`, looping per matrix because numpy 1.21 has no batched QR.
  - `scale_pct` = 100·mean|Δ log D|.
  - `shear_pct` = 100·mean|ΔU| over (k_xy, k_xz, k_yz), the augmentation's shear convention.
  - `qr_rot_deg` = ∠(Q Q_trueᵀ).
  - These are logged as `val_affine/{center,pair}_{scale_pct,shear_pct,qr_rot_deg}` (eval mode only). The tests build matrices from known factors as the numpy reference.
- **Validation sets.** `val_affine` (and `val_affine_g` with G on) already existed from milestone 4: fixed seeded affine-range transforms, built by `build_validation_sets`. Validation PNGs follow the set name, so they are written as `affine_step*.png` (tested).
- **Best checkpoint.**
  - The `--best-metric` default is now None, which resolves to `val/mean_pair_rms_mm` (the mean of `val_<task>/pair_rms_mm` over the trained tasks, computed only when every one was validated) when more than one task is trained. With one task it resolves to `val_<task>/pair_rms_mm`, as before.
  - `mean_pair_rms_mm` is accepted as an alias. Explicit keys behave as before, including the `val_rigid/` → `val_<task>/` rewrite.
  - This changes the default only for runs with more than one task, as the brief asked.
- **Overfit mode with several tasks.** There is one fixed group per task over the same images. Rigid keeps its milestone-4 transforms; affine draws from `default_rng([seed, 202])` with the `--affine-*` ranges. The run stops when every task's own 10-step mean is below the threshold. `overfit_summary.json` gains `tasks`, `trace_counts`, `window_center_rms_by_task` and `task_curve`, plus a per-task `val_curve`; the rigid-only layout is unchanged.
- **G warm-up** (orchestrator relay). `--global-start-step S` (default 0) and `--global-ramp-steps R` (default 0).
  - Factor f = 0 before S, (s − S)/R on [S, S + R], and 1 afterwards. The block draw (seed, step // K) is unchanged; its angle and translation are scaled by f, so the defaults reproduce the old sampler exactly (tested).
  - Logged as `train_<task>/control_g_rot_max_deg`.
  - Being a pure function of the step, a resumed run restores it.
- **Warm start** (orchestrator relay). `--init-weights PATH` accepts a checkpoint prefix or a `ckpt/best` / `ckpt/last` directory (its latest checkpoint).
  - `checkpoint.load_model_weights` restores only `model` (`tf.train.Checkpoint(model=...)`, `expect_partial`, `assert_existing_objects_matched`).
  - It compares the source run's `config.json` ModelConfig with the new one, ignoring the init seed (as `copy_weights_from` does). A mismatch raises `ValueError` naming the fields.
  - The run starts with a fresh optimizer at step 0.
  - `main` exits if the run dir already has a checkpoint and `--resume` is on.
  - `config.json` records `init_weights: {checkpoint, source_config}`.
  - Tests: round trip from a saved tiny run (prefix and directory forms; weights equal; optimizer and step untouched), config mismatch, missing path, empty dir, the main dry run, and the resume conflict.
- **`main.py`** writes `<run>/fit_summary.json` (trace counts, steps, non-finite count) for non-overfit runs.

## Review-curriculum follow-ups

- **(a) Issue 2, G angle sign, determined analytically.**
  - DALI 1.18 `fn.transforms.rotation` is the right-handed Rodrigues matrix: `rotation3D(axis, deg2rad(angle))` in `include/dali/core/geom/transform.h` and `transform_rotation.cc`, read at the v1.18.0 tag. It is built in DALI's reversed axis order with a_d = P a.
  - M_tf = P M_dali P, and P (swapping x and z) is improper. For orthogonal Q, Q R(u, θ) Qᵀ = R(det Q · Q u, θ), so P R(Pa, θ) P = R(a, −θ). The old code therefore emitted R(axis, −angle) in TF space.
  - Fix: `control_global_params` passes `angle × G_DALI_ANGLE_SIGN (−1)`, so the emitted G is R(axis, +angle) = `global_matrix_mm`, matching the logged `control_g_angle_deg` and `val_<task>_g`.
  - The local test `test_g_angle_sign` checks it through the numpy mirror and `dali_voxel_to_tf_mm`: +angle → R(a, −θ), and the corrected angle → `global_matrix_mm`.
  - Check i is now strict: signed axis dot > 1 − 1e-5, angle within 0.01°, |G − global_matrix_mm(written)| < 1e-3, and the single-sign mirror < 1e-3.
  - DALI itself was not run here. Check i on glacier confirms it.
- **(b) Issue 3.** The tautological recompose sub-check is removed. Check j's render equality covers T independently.
- **(c) Issue 4.** Check j's `task_ok` now requires `barycenter_equivariance_max_abs < --tol-bary` (default 1e-6). `measure_bary.py` measured a maximum of 1.7e-14 over 20 synthetic groups per task (float32-rounded inputs, float64 math, check j's ranges).
- **(d) Issue 5.** `main` reads the resumed step with `checkpoint.peek_last_step` before `_build_dali_data`, which writes `values_at(start_step)`. Tested with a stub data module (steps 0 / 60 / 500).
- Issue 6 (batch-level non-reproducibility with controls on) is unchanged and documented in milestone 4.

## Overfit results (synthetic phantom, level 4, N = 4, Adam 3e-4, flips and dropout off, stop at a 10-step training mean < 0.8 mm per task, then eval mode at the stopping step)

Affine ranges are ±45° / ±15 mm / ±10% log-scale / ±5% shear. Rigid is pure rigid ±45° / ±15 mm.

| Run | Model | LR schedule | Steps | Eval center RMS (train mode) | Eval pair RMS | Affine centered scale / shear / QR rot | Median step |
|---|---|---|---|---|---|---|---|
| affine (suite test) | narrow | constant | 1846 | **0.606 mm** (0.618) | 0.971 mm | 0.50% / 0.55% / 0.62° | 562 ms |
| affine | full width (13.9 M) | constant | 1878 | **0.502 mm** (0.503) | 0.801 mm | 0.37% / 0.44% / 0.32° | 1605 ms |
| mixed rigid:affine (suite test) | narrow | cosine, no warmup, over 4000 | 2199 | rigid **0.318** (0.336), affine **0.605** (0.620) | 0.547 / 1.013 mm | 0.38% / 0.40% / 0.67° | 528 ms |
| mixed rigid:affine | full width | cosine, no warmup, over 4000 | 2171 | rigid **0.417** (0.408), affine **0.771** (0.783) | 0.705 / 1.239 mm | 0.49% / 0.90% / 0.74° | 1595 ms |
| mixed rigid:affine (exploratory) | narrow | constant | 3095 | rigid **1.142** (1.144), affine 0.769 | 1.910 / 1.260 mm | | 592 ms |
| rigid only (suite, milestone-4 test) | narrow | constant | 593 | 0.763 (0.589) | | | 516 ms |

The suite numbers are those of the logged suite run in `test-log.txt`; they match the standalone runs. Curves are in `overfit_summary_*.json` and logs in `overfit_*.log`.

- **The affine overfit works** at both widths. It needs about 3× the steps of rigid alone (1846 vs ~600), and the eval/train-mode gap is ≤ 0.012 mm.
- **Mixed, "rigid not degraded" at small scale.** Both tasks reach < 1 mm with one optimizer. Rigid's eval RMS (0.32 narrow / 0.42 full) is below the rigid-only run's (0.76). It costs about 2200 steps for both tasks, about 1100 per task.
- **Interference at a constant LR.** With constant Adam 3e-4 the mixed run hits the 10-step training criterion at 3095 steps, but rigid eval at that step is 1.14 mm. Across validations the eval RMS oscillates between 0.5 and 2.2 mm for rigid and between 1.4 and 2.4 mm for affine from step 1250 on: each task's updates perturb the other. The default cosine decay removes this, so the mixed test uses `--schedule cosine --warmup-steps 0` (Adam 3e-4, the default optimizer). No default changed. Real mixed runs use the default cosine schedule anyway.
- Peak TF memory: about 1.3 GB narrow and 3.9-4.0 GB full width. Interleaving adds no memory, since only one task's step runs at a time.

## Tests and how they were run

- Development runs used `wsl -d Ubuntu bash .agents/tasks/milestone-5/run_some.sh <modules>` and `overfit_run.sh <name> <flags>` (results/m5_local, deleted afterwards; summaries copied here).
- The full suite ran after the last code edit, with `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/tests/run_tests_wsl.sh` and output to `test-log.txt` (finished 10:36:47; the last code edit was at 04:13).
  - py_compile and the ast feature_version=(3, 8) check of 44 files are OK.
  - The TF-free imports and `main.py --help` are OK.
  - `bash -n` passes on all three scripts.
  - **Ran 194 tests in 3852 s, OK (skipped=3)**: the DALI pipeline tests (2) and dllogger. That is 16 new tests (178 in the milestone-4 curriculum log). The wall time grew from 1437 s to 3852 s, mostly from the two new overfit tests (about 17 and 19 min).

## Glacier commands for the orchestrator (NOT run)

On glacier, after `git pull` in the clone, pick a free GPU with `nvidia-smi`.

```bash
# h-k with the strict check i (signed axis / angle), gated check j barycenter, no recompose sub-check
bash /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad/tests/gpu/run_verify_glacier.sh <gpu> 1800 verify h,i,j,k
# short real-data mixed run: rigid:1,affine:1, N = 8, level 4, 20 steps, val + PNGs at 10 and 20,
# then tests/gpu/check_mixed_run.py on the run dir (exit code = the mode's result); writes scratch/m5
bash /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad/tests/gpu/run_overfit_glacier.sh <gpu> 1800 dryrun_mixed
```

Expected results:
- **Check i:** `G_axis_signed_dot` ≈ 1, `G_angle_deg` ≈ 30, `G_vs_global_matrix_mm_max_abs` < 1e-3 and `G_vs_numpy_mirror_max_abs` < 1e-3. If check i fails only on the sign, the DALI rotation convention differs from the source read above; flip `params.G_DALI_ANGLE_SIGN` to +1.
- **Check j:** `barycenter_equivariance_max_abs` < 1e-6.
- **`dryrun_mixed`:** `check_mixed_run: PASS`, with both tasks trained and validated, affine errors finite, `val/mean_pair_rms_mm` equal to the task mean, trace counts {rigid: 1, affine: 1}, `rigid_step0000020.png` / `affine_step0000020.png`, and a best checkpoint.

## Not verified

- The DALI side of the sign fix, and the gated checks (glacier h-k).
- The real-data mixed run (`dryrun_mixed`).
- G warm-up and warm start on real data.
- Whether mixed training on real data degrades rigid at N = 32. Only the synthetic overfit was run.
- Python 3.8 / numpy 1.21 at runtime. Only the syntax check ran here; `qr_factors` deliberately avoids the batched `np.linalg.qr` added in numpy 1.22.
