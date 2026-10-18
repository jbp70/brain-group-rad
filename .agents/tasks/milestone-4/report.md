# Milestone 4 report: linear losses, metrics, custom loop (rigid, level 4), logging, checkpoints, main, SLURM debug template

Outcome: the rigid training loop runs end to end locally (TF-side data, no DALI). The eval-mode gap of the first version is fixed: the dense path now uses LayerNorm by default (design v3.4), and on the overfit group eval mode now tracks training mode within ~0.01 mm. The full suite and its overfit numbers are in `test-log.txt` (see "Follow-up: dense-path LayerNorm" for the counts). Nothing was committed, pushed, ssh'd or submitted.

Findings for Jay (details in "Findings"):
1. **Resolved: the eval-mode BN gap on the dense path.** With `dense_norm='layer'` (new default) the full-width model reaches an eval-mode center RMS of 0.51 mm (Adam) and 0.006 mm (SGDW defaults) on the overfit group, with training mode at 0.51 / 0.006 mm. Conv-path BatchNorm still lags briefly when the weights move fast (SGDW, steps 200-450).
2. **Default LR lowered to 3e-3.** The reference SGDW lr 0.01 drove the overfit error from 22 mm up to ~90 mm before recovering. The plan's decision 6 authorized this change.

## Files

| File | Content |
|---|---|
| `runtime/__init__.py` | Package docstring only |
| `runtime/args.py` | TF-free argparse surface, `str2bool`, `int_list`, `parse_task_ratio`, config builders |
| `runtime/schedulers.py` | `ScheduleConfig`, `lr_at(step, cfg)` (warmup + cosine / poly / cycle / constant), pure Python |
| `runtime/losses.py` | `linear_center_loss`, `linear_pair_loss`, `linear_loss` (mm²) |
| `runtime/metrics.py` | numpy float64 errors (rms mm, rotation deg, translation mm), all-pairs, cycle, `summarize`, `MeanAccumulator` |
| `runtime/optimizers.py` | SGDW port (identity-based exclusion), `make_optimizer`, LR get/set, `build_optimizer`, `uses_weight_decay` |
| `runtime/checkpoint.py` | `RunCheckpoint` (last keep 2, best keep 1), `write_config`, `check_config_compatible` |
| `runtime/logging.py` | `RunLogger` (dllogger if importable, else JSON lines; TensorBoard scalars), `make_run_dir`, `to_jsonable` |
| `runtime/fixed_data.py` | Phantom, numpy transform sampler, TF render, DALI-format batches, manifest numpy loader, `FixedGroupSource`, `ValidationSet` |
| `runtime/run.py` | `TaskSchedule`, `TrainConfig`, `Trainer` |
| `main.py` | Entry point (`main(argv) -> int`), DALI / overfit / dry-run modes |
| `slurm/train_debug.sh` | sbatch template (NOT submitted) |
| `tests/gpu/run_overfit_glacier.sh` | Glacier overfit / DALI dry-run runner (NOT run) |
| `tests/runtime_helpers.py` | Narrow config, `free_memory`, synthetic groups, fake export |
| `tests/test_runtime_losses_metrics.py`, `test_runtime_optim.py`, `test_runtime_infra.py`, `test_trainer.py`, `test_main.py` | New tests (45) |
| `tests/tf_setup.py` | Optional `BGR_TF_GPU_LIMIT_MB` (fixed TF memory limit instead of growth; local WSL only) |
| `tests/run_tests_wsl.sh` | Adds `runtime/*.py main.py` to the 3.8 check, the TF-free import check, `main.py --help`, `bash -n` on the two new scripts, the local GPU env vars, line-buffered output |
| `models/precision.py` | `configure_tf_precision` also ignores `ValueError` from `set_memory_growth` (raised when a logical memory limit is configured) |
| `.agents/tasks/milestone-4/` | `record_tests.sh`, `test-log.txt`, `overfit_summary_*.json`, `overfit_full.sh`, `profile_step.py`, `diag_eval_mode.py`, `run_some.sh`, `diag_nan_grad.py`, `run_py.sh`, `overfit_full_*_layer.log` |

## Public API

`runtime.losses` (TF; A is float32 even under mixed_float16):
- `linear_center_loss(A, A_star, S)`: mean_i tr(ΔA_i S ΔA_iᵀ), in mm².
- `linear_pair_loss(A, T_target, pair_i, pair_j, S, P_star=None)`: mean over the pairs of tr(ΔP S ΔPᵀ), with ΔP = A_j⁻¹A_i − T_jT_i⁻¹. It uses `P_star` (float64 targets from postprocess) when given, otherwise `pair_targets(T_target)`.
- `linear_loss(A, batch, S, lambda_pair=1.0)` returns `{loss, center, pair, center_rms_mm, pair_rms_mm}`.

`runtime.metrics` (numpy float64):
- `rotation_angle_deg(L)` (atan2 form, angle of the SVD polar rotation) and `polar_rotation(L)`.
- `matrix_errors(M, M_true, S)` returns `{rms_mm, rot_deg, trans_mm}`.
- `centered_errors`, `pairwise_errors` (all N(N−1) ordered pairs), `cycle_errors(A, S, n_triplets=64, seed=0)` (distinct triplets; empty if N < 3).
- `group_errors` gives per-matrix arrays; `summarize(errors, prefix)` gives means plus `*_rms_mm_p95`; `group_metrics` combines the two.
- `concat_errors` pools validation groups; `MeanAccumulator`.

`runtime.schedulers`:
- `ScheduleConfig(name, base_lr, total_steps, warmup_steps, min_lr_factor, power, cycle_steps, cycle_mult, cycle_decay)`.
- `lr_at(step, cfg)`.

`runtime.optimizers`:
- `SGDW`, `DEFAULT_WD_EXCLUDE = ('gamma', 'beta', 'bias')`.
- `make_optimizer(name, lr, weight_decay, momentum, dampening, nesterov, decoupled, clipnorm, exclude_wd, amp, jit_compile)`: names sgdw, sgd, adam, adamw; wrapped in an LSO when `amp`.
- `inner_optimizer`, `build_optimizer(opt, variables)`, `set_learning_rate`, `get_learning_rate`, `uses_weight_decay`.

`runtime.checkpoint`:
- `RunCheckpoint(ckpt_dir_last, ckpt_dir_best, model, optimizer, step, extra=None, keep_last=2)` with `save_last()`, `maybe_save_best(value) -> bool`, `restore_last() -> bool`, `restore_best()`, `latest_last`, `latest_best`, and `best_metric` (a float64 variable, stored in the checkpoint).
- `write_config(path, model_config, args_dict, extra=None)` and `check_config_compatible(path, model_config)`.

`runtime.logging`:
- `make_run_dir(results, run_name)` returns paths for `root, logs, tb, ckpt_last, ckpt_best, config`.
- `RunLogger(run_dir, backend='auto'|'dllogger'|'jsonl', tensorboard=True, quiet=False)` with `log_params`, `log_metrics(step, dict)`, `log_event(step, dict)`, `flush`, `close`, `.backend_name`, `.metrics_path`.
- Log line format: `{"step", "time", "data"}`.

`runtime.fixed_data`:
- `smooth_phantom(shape, rng)` returns `(vol, mask)`.
- `random_linear_transforms(rng, n, task, ranges=None, nuisance=True)` returns `(T, T_target)`.
- `render_group(images, T, spacing)`, `to_dali_batch(images, T, T_target, task, level)`.
- `load_level_images(export_root, level, split, n=None, seed=0)` returns `(images, files)`.
- `FixedGroupSource(batch)`.
- `ValidationSet.build(images, task, level, group_size, n_groups=None, ranges=None, seed=1234, target_iters=10)`, `ValidationSet.from_group(...)`, `.subset(k)`.

`runtime.run`:
- `parse_task_ratio` (re-exported), `TaskSchedule(ratio, seed).draw(step)`, `TrainConfig` (dataclass, fields as in the plan).
- `Trainer(model, optimizer, sources, S, cfg, logger=None, checkpoint=None, val_sets=None)`.
  - Attributes: `step` (int64 variable), `postprocess[task]`, `trace_counts`, `nonfinite_total`, `history`, `val_history`, `reached`.
  - Methods: `rebuild_steps()`, `train_step(task, batch_tuple)`, `validate(max_groups=None)`, `fit(num_steps=None)`, `checkpoint_variables()`.

`main.main(argv=None) -> int`. Run `python main.py --help` for the flags. The flag list matches the plan, with two changes: `--lr` defaults to 3e-3, and there is a new `--gpu-memory-limit-mb` (0 = off).

## Trainer loop and step functions

One `tf.function` per task. It is built in `rebuild_steps()` and traced once:

```
rigid:  tf.function(step(images, t_dali, rt_dali), input_signature=[
            TensorSpec([N, 40, 56, 48, 1], f32), TensorSpec([N, 3, 4], f32), TensorSpec([N, 3, 4], f32)])
affine: tf.function(step(images, t_dali), input_signature=[TensorSpec([N, X, Y, Z, 1], f32), TensorSpec([N, 3, 4], f32)])
        (shapes from dali_loader.linear_output_signature(task, level, N))
val:    tf.function(fwd(x), input_signature=[TensorSpec([Nv, X, Y, Z, 1], f32)]) -> A, eval mode, per (task, Nv)
```

The step body:
1. `LinearPostprocess` runs outside the tape (flips, mm matrices, float64 targets, pairs).
2. Non-finite input images are zeroed and mark the step non-finite.
3. BN moving statistics are backed up.
4. Under the tape: forward pass, `linear_loss`, and the scaled loss under AMP.
5. Gradients are computed, and None gradients are dropped statically (the scale/shear head in rigid steps). Under AMP the gradients are unscaled.
6. `global_norm` is computed and `finite = loss & norm & inputs finite`.
7. `tf.cond(finite, apply_gradients, skip)`. Under AMP the gate is `inputs finite` only, so the LSO still sees overflows and lowers its scale.
8. `tf.cond(finite, no-op, restore BN statistics)`.
9. `step += 1`. The step returns the loss terms, `grad_norm`, `finite` and `n_vars_updated`.

The optimizer is built on all trainable variables in `Trainer.__init__`, so the affine head has slots even when the run starts with rigid steps.

`fit()` loop, per step (Python):
1. `task = schedule.draw(step)`. The draw is stateless and seeded per step, so a resumed run gets the same sequence.
2. `set_learning_rate(lr_at(step))`.
3. A timed `next(source)` gives the data wait.
4. A timed step call (including the host sync) gives the compute time.
5. History record. A non-finite step logs a `nonfinite_step_skipped` event, and more than `nonfinite_patience` consecutive skips raise `RuntimeError`.
6. Every `log_every` steps, log per-task loss/center/pair means and RMS, grad norm, lr, data/compute ms, `gpu_peak_mb` and the non-finite count.
7. Every `val_every` steps, `validate()` then `maybe_save_best(best_metric)`.
8. Every `ckpt_every` steps, and at the end, `save_last()`.
9. In overfit mode, stop when the 10-step mean `center_rms_mm` is below the threshold.

`fit` starts from `int(step)`, so a resumed run continues the step count (tested).

## Defaults and why

| Setting | Default | Why |
|---|---|---|
| Optimizer | SGDW, momentum 0.99, Nesterov, dampening 0, coupled wd 3e-5, `global_clipnorm` 12 | Rigid-project values (design 5.2) |
| WD exclusion | names matching gamma / beta / bias (on) | BN affine and biases. The reference list ('normalization') matches nothing here. Compared by identity, not `in` |
| Peak LR | **3e-3** (reference 0.01) | Full-width overfit: 0.01 pushed the RMS from 22 to ~90 mm (steps 100-250) before the cosine decay recovered (< 1 mm at step 454). 3e-3 reached < 1 mm at step 325 after a smaller excursion (~50 mm, steps 50-150). The loss is in mm², about 10³ at init, so early gradients (norm 10⁶-10⁷) are always clipped to 12, and momentum 0.99 integrates them 100× |
| Schedule | warmup 1000 (debug job: 200), cosine to lr × 1e-3 at `--steps` | Reference warmup length; pure function of the step |
| Loss | λ_pair 1, p_pair 0.1, flips 0.5 per axis, 10 target iterations | Design 6.1 / 3.3 |
| Group size | 32 / 8 / 2 for levels 4 / 2 / 1 | Design 5.1 |
| Dropout | 0.15, or 0.0 in overfit mode | Dropout noise would keep the overfit RMS above 1 mm |
| AMP / XLA | off / off | Correctness first. Under AMP, the default initial loss scale 2¹⁵ overflows float16 on the first steps (mm² losses); the LSO skips those steps and lowers its scale, as the AMP test shows |
| Best checkpoint | `val_rigid/pair_rms_mm` (lower is better), eval mode | Plan decision 10. `val_rigid_trainmode/*` is logged next to it as a diagnostic |
| Dense-path norm | `layer` (`--dense-norm layer/batch/none`) | Design v3.4, finding 1. Conv norm stays `batch` |

## Overfit results, first version (dense_norm = batch; superseded, see the follow-up section)

Synthetic phantom, level 4, N = 4, ±45° / ±15 mm pure rigid, flips and dropout off.

Criterion: the mean of the last 10 training-step `center_rms_mm` values < 1.0 mm. These are training-mode BN values, measured before each update. Init: 22.29 mm centered, 41.48 mm pairwise.

| Run | Model | Optimizer | Steps to < 1 mm | Center RMS (mm) at steps 1 / 50 / 100 / 150 / 200 / 250 / 300 / end | Median step | Peak TF memory |
|---|---|---|---|---|---|---|
| Unit test (logged suite run) | narrow | Adam 3e-4 const | **352** (other runs: 380, 399, 483; cuDNN is not bit-deterministic) | 22.3 / 13.2 / 9.8 / 6.1 / 3.6 / 2.8 / 2.3 / 1.01 (10-step mean 0.99) | 556 ms | ~1.4 GB |
| `full_adam` | default (13.9 M) | Adam 3e-4 const | **242** | 22.3 / 11.1 / 6.1 / 2.4 / 1.8 / – / – / 0.88 | 1033 ms | 3.9 GB |
| `full_sgdw_default` | default | SGDW 0.01, warmup 100, cosine over 500 | 454 | 22.3 / 28.6 / 60.6 / 83.9 / 9.2 / 92.7 / 15.9 / 0.87 | 1015 ms | 3.8 GB |
| `full_sgdw_3e-3` | default | SGDW 3e-3, warmup 100, cosine over 500 | **325** | 22.3 / 46.4 / 49.4 / 50.9 / 3.2 / 1.7 / 8.4 / 0.73 | 1176 ms | 3.8 GB |
| `narrow_adam_eval` | narrow | Adam 3e-4, 1000 steps, no stop | (0.36 best) | train: 0.69 at 400, 0.36 at 950; **eval: 22.2 → 19.8** | 541 ms | 1.3 GB |

The < 1 mm target is reachable at the default model size locally. The suite uses the narrow config only for speed (plan decision 13). LR tuning: Adam 3e-4 was the first value tried and worked, so 1e-4 and 1e-3 were not needed. Curves are in `overfit_summary_*.json` (every step) and in the test log (every 50 steps).

## Follow-up: dense-path LayerNorm (design v3.4) and review issues 2-4

Suite (`test-log.txt`, started 08:38:49, after the last edit at 08:38:47): **145 tests, OK, 2 skips** (the DALI pipeline test and the new dllogger test, since neither package is importable locally). Wall time 972 s. Six tests are new: LayerNorm per-sample and mixed precision, the non-updating train-mode validation, the forced AMP overflow, the dllogger backend and val-set task-order independence.

### Changes

| Area | Change |
|---|---|
| `models/layers.py` | `LayerNormWithAffine`: per-vector over the last axis, float32 math, ε, `AffineParams` γ/β. `make_norm('layer')`. `GroupDenseBlock` accepts `layer` / `batch` / `none` (default `layer`), and so does `NormAct` |
| `models/config.py`, `runtime/args.py` | `dense_norm` / `--dense-norm` take `layer` / `batch` / `none`, **default `layer`**, with the reason documented. Conv `norm` stays `batch`. `main.py` takes the flag through `model_config_from_args` |
| `models/group_net.py` | Docstrings only. The pre-head norm-act already followed `dense_norm` |
| `runtime/run.py` | `validate()` adds `val_<task>_trainmode/{center,pair}_rms_mm`. The forward runs with `training=True`, dropout rates zeroed at trace time, BN moving statistics backed up and restored in the same `tf.function`, and its own trace key |
| `main.py` | Overfit mode runs one final `validate()` at the stopping step. `overfit_summary.json` gains `final_val` and `val_curve` (step, eval RMS, train-mode RMS). Val sets come from `fd.build_validation_sets` |
| `runtime/fixed_data.py` | `ValidationSet.build` seeds the permutation from (seed, task id) and group g's transforms from (seed, task id, g), with `VAL_TASK_SEED = {rigid: 101, affine: 202}`. `build_validation_sets` is new. `main.py` passes `--val-seed` unchanged for every task (it used to pass `+ i`). Validation sets from before this change differ |
| `slurm/train_debug.sh`, `tests/gpu/run_overfit_glacier.sh` | `--log-backend jsonl` |
| Tests | `test_group_layers`: LayerNorm per-sample, train == eval, γ/β only, float16 in → float16 out. `test_trainer`: train-mode val keeps BN statistics and is deterministic. The AMP overflow test forces initial scale 2⁴⁰ with a small random head kernel, because zero-initialized float32 heads send no gradient into the float16 layers on step 1. It asserts that each skipped step halves the scale, leaves weights, BN statistics and optimizer slots unchanged, and advances `iterations` by 1. In the logged run, 33 steps were skipped and the first applied step was at 2⁷. The AMP smoke-test comment and assertion are fixed: Keras' LSO advances `iterations` on skipped steps too, so the count is 21 calls, and the earlier `n_finite + first_applied` claim was wrong. `test_runtime_infra`: rigid val set identical for `rigid:1` and `affine:1,rigid:2`, independent of `n_groups`; guarded dllogger test. `test_main` overfit: stops at a 10-step train mean < 0.8 mm, then asserts final **eval-mode < 1.0 mm** and \|eval − train mode\| < 0.5 mm |
| `test_group_net` smoke | 60 steps instead of 20 (Adam 1e-4, full width, dropout on). With LayerNorm the loss plateaus for ~30 steps (RMS 7.9 → 7.5 at step 20, failing the 0.7× check), then drops to ~5.2 mm |
| `tests/run_tests_wsl.sh` | `TF_CUDNN_USE_AUTOTUNE=0` (local only), see finding 5 |

### Overfit results (synthetic phantom, level 4, N = 4, ±45° / ±15 mm pure rigid, flips and dropout off, dense_norm = layer)

Full width (13.9 M, default model), 1000 steps, no early stop (`--overfit-threshold-mm 0`), validation of the same group every 50 steps. Eval = `val_rigid/center_rms_mm` (moving statistics), train mode = `val_rigid_trainmode/center_rms_mm` (batch statistics, no update). All values are center RMS in mm.

| Step | Adam 3e-4 const: eval | train mode | SGDW defaults (3e-3, warmup 100, cosine 1000): eval | train mode |
|---|---|---|---|---|
| 50 | 20.57 | 13.59 | 21.88 | 13.53 |
| 100 | 10.90 | 8.27 | 20.70 | 12.21 |
| 200 | 6.65 | 3.12 | 7.58 | 2.31 |
| 250 | 3.31 | 1.85 | 7.74 | 0.97 |
| 300 | 3.36 | 2.25 | 7.77 | 0.76 |
| 400 | 1.25 | 1.68 | 3.44 | 0.39 |
| 500 | 1.07 | 0.99 | 0.71 | 0.32 |
| 600 | 1.21 | 1.20 | 0.29 | 0.25 |
| 700 | 0.73 | 0.72 | 0.065 | 0.044 |
| 800 | 0.90 | 0.89 | 0.025 | 0.015 |
| 900 | 0.84 | 0.83 | 0.012 | 0.010 |
| **1000** | **0.508** | **0.514** | **0.0062** | **0.0058** |

- Both runs meet the target: eval < 1 mm at the end, and the gap is < 0.01 mm (eval/train ≈ 1.0 for Adam and 1.07 for SGDW).
- Training-step 10-step mean < 1 mm was first reached at step 593 (Adam) and step 238 (SGDW). Median step time was 988 / 896 ms, and peak TF memory 3.8 GB.
- Convergence is slower than with dense BN: Adam took 242 steps with BN against 593 now. The Adam curve stays noisy at constant LR (0.5-1.4 mm after step 650).
- SGDW shows a transient gap at steps 200-450 (eval 7.7 mm vs train mode 0.97 mm at step 250) that closes by step 500. That is conv-BN moving-statistics lag (momentum 0.99) while the weights move fast, not the structural dense-BN gap.
- No LR or other default changed. `dense_norm=none` was not run.
- Narrow model in the suite (Adam 3e-4): stopped at step 583. At that step eval was 0.599 mm and train mode 0.599 mm. At step 250 they were 17.1 vs 4.0 mm (the same transient lag). Other runs today stopped at 532 / 841 steps with final eval 0.74 / 0.65 mm.
- Files: `overfit_summary_full_{adam,sgdw}_layer.json`, `overfit_full_{adam,sgdw}_layer.log`.

## Findings

1. **Eval-mode BN gap on the dense path: resolved by LayerNorm (design v3.4).**
   - Cause: with one group per batch, the dense-path BatchNorms (inputs 5040, 512 and 256 wide in the narrow model) normalized over the N = 4 group members. In training mode that is group-relative: it subtracts the group mean and divides by the group spread. The eval-mode moving statistics mix between-group and within-group variation, so they can't reproduce it. `diag_eval_mode.py` measured moving_var / batch_var of 7278, 193 and 252 there, against 1.4-1.7 in the conv BNs. Training mode reached 0.36 mm while eval mode stayed at ~20 mm.
   - Fix (Jay's decision): `ModelConfig.dense_norm` and `--dense-norm` accept `layer` / `batch` / `none`, default `layer`. `models.layers.LayerNormWithAffine` normalizes each vector over its own features in float32 (ε = `bn_epsilon`) with trainable γ/β (`AffineParams`). It has no statistics and doesn't depend on the other group members. The dense blocks and the pre-head norm-act follow `dense_norm`. `none` keeps the activation and drops the norm. `batch` remains for ablations. The conv path keeps BatchNorm.
   - Diagnostics: every validation now also logs `val_<task>_trainmode/{center,pair}_rms_mm`. This is the same fixed groups with training-mode BN and dropout off, and the BN moving statistics are restored afterwards (tested). Eval mode (`val_<task>/...`) stays the source of truth for `best`.
   - Result: see "Follow-up: dense-path LayerNorm". The full-width gap at the end is under 0.01 mm. The remaining transient gap with SGDW (eval 7.7 mm vs train mode 0.97 mm at step 250, closed by step 500) is conv-BN lag while the weights move fast, not a structural gap. `dense_norm='none'` was not run, because LayerNorm met the target.
2. **NaN inputs crash instead of propagating.** A NaN voxel makes `GroupCentering`'s matrix inverse raise `InvalidArgumentError` ("Input is not invertible") rather than return NaN. The step therefore zeroes non-finite inputs and marks the step non-finite (skip plus BN rollback, tested). Non-finite values that arise inside the forward pass from finite inputs and weights would still raise. That is unlikely with finite weights, which the guard protects.
3. **The local GPU slows down sharply near its memory limit (WSL).** With memory growth, cuDNN autotune workspaces push TF to the last GB of the 8 GB GTX 1080. WSL then pages GPU memory to host RAM, and one full-width training step takes 175-430 s instead of 0.8 s. `TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512` plus a fixed 6 GB TF memory limit (`--gpu-memory-limit-mb 6000`, or `BGR_TF_GPU_LIMIT_MB=6000` for tests) fix it. Both are set only by the local scripts; defaults are unchanged (memory growth). This is not expected on the 40-48 GB cluster GPUs, but that is unverified.
4. **Step-time breakdown** (GTX 1080, full width, N = 4, `profile_step.py`): post-processing (float64 targets, 10 barycenter iterations) 85-98 ms, encoder forward+backward 660 ms, model forward+backward 790 ms, full step about 1.0 s. With the narrow model the costs are 79 / 212 / 330 / 420 ms, so the logm/expm centering plus targets cost about 0.2 s per step whatever the width. N = 8 at full width does not fit in 6 GB on the GTX 1080: the l4 padded input alone is [8, 256, 41, 57, 49] float32 = 0.9 GB.

5. **First-call cuDNN NaN on the GTX 1080 (local, unrelated to LayerNorm).** With cuDNN autotune on, the first full-width backward pass of a process returned 512 NaN entries (of 442,368) in the encoder l3 conv-kernel gradient. The gradient into l2 and every later call stayed finite. `diag_nan_grad.py` reproduced it with dense_norm batch and layer alike, with and without the 512 MB workspace cap. It did not occur on the CPU or with `TF_CUDNN_USE_AUTOTUNE=0`. It made `TestGradients` flaky, so the local runner now sets autotune off. In training, the NaN guard would skip such a step. Whether the cluster GPUs show it is unverified.

## Commands for the orchestrator / Jay

1. Glacier (pick a free GPU with `nvidia-smi` first, then `git pull` in the clone). These write only under `.../GroupwiseReg/scratch/m4`:

   ```bash
   bash /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad/tests/gpu/run_overfit_glacier.sh <gpu> 3600 overfit
   bash /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad/tests/gpu/run_overfit_glacier.sh <gpu> 1800 dryrun
   ```

   - `overfit`: 8 real level-4 training images, fixed rigid transforms, default SGDW (lr 3e-3, warmup 100, cosine over 3000 steps). Exit 0 if the 10-step mean RMS drops below 1 mm. Summary in `scratch/m4/results/overfit_*/overfit_summary.json`.
   - `dryrun`: real DALI pipeline, N = 8. Builds the DALI datasets and the validation set (1 group of 8 val cases), then runs 2 steps + 1 validation and saves a checkpoint. Exit 0 if every value is finite.

2. Proposed SLURM debug job, **for Jay to approve** (not submitted):

   ```bash
   mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
   cd /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad && git pull
   sbatch slurm/train_debug.sh
   ```

   - Resources: account `qtim`, partition `rtx8000` (Quadro RTX 8000, 48 GB), 1 GPU, 1 node / 1 task, 3 CPUs, 64 GB RAM, time 01:00:00.
   - Run: `main.py` inside `tf10_nv.sif` on rigid only, level 4, N = 32, 2000 steps (warmup 200, cosine, SGDW lr 3e-3), validation every 250 steps on the 39 T1w val cases (2 fixed groups of 32), checkpoints every 250 steps.
   - Logs go to `GroupwiseReg/logs/bgr-m4-debug-<jobid>.{out,err}`; the run goes to `GroupwiseReg/results/m4_debug_<jobid>/`.
   - Runtime estimate (unmeasured on an RTX 8000): ~2-3 s/step at N = 32, so 2000 steps need about 1.1-1.7 h. With 1 h the job will likely stop at the time limit after ~1200-1800 steps, keeping the last checkpoint (every 250 steps). For a complete run, approve either `STEPS=1000 sbatch slurm/train_debug.sh` or `--time=02:00:00`.
   - Memory: ~30 GB extrapolated. On OOM, use `N=16` or `AMP=true`.

## Not verified

- The DALI training path (`LinearDataModule` iterators feeding the step signature) and the DALI dry run. The glacier `dryrun` covers them.
- The container runtime (Python 3.8.10, numpy 1.21.1, dllogger backend). Locally only the py3.8 syntax check, TF-free imports and `--help` ran in `bgr_preproc38`; the TF tests ran in Python 3.9 / numpy 1.24.
- RTX 8000 / A100 step time and memory at N = 32. XLA. AMP speed: on the GTX 1080 AMP was only smoke-tested, and that card has no fp16 tensor cores.
- Real-data overfit (glacier `overfit`) and the train/eval gap on real data with LayerNorm (finding 1; the `val_rigid_trainmode/*` logs will show it).
- `dense_norm='none'` training (only constructed and validated as a config value).
- The dllogger backend (the guarded test skips locally; the debug and glacier scripts now force `jsonl`).
- The cosine/cycle schedules and the SGDW defaults beyond the 500-step synthetic sanity checks.
