# Milestone 4 implementation plan: linear losses, metrics, custom loop (rigid, level 4), logging, checkpointing, main, SLURM debug template

Scope: design v3.3 sections 2, 3.2-3.3, 4.4, 5.1-5.2, 6.1, 7, 10, 11 (row 4). LOCAL ONLY: verify in WSL (`tf_gpu` for TF tests, `bgr_preproc38` for the Python 3.8 syntax check). No git commit/push, no ssh, no SLURM submission. Reference code under `reference/` is read-only.

## Ground truth found during exploration (what the plan builds on)

- `models.transforms`: `point_loss(dA, S)` returns tr(dA S dAᵀ) per matrix (mm²); `pair_predictions(A, i, j)` = A_j⁻¹ A_i; `pair_targets(T, i, j)` = T_j T_i⁻¹ (stop_gradient); `barycentric_targets`, `affine_inverse(m, rigid=False)`, `to_3x4`, `to_homogeneous`, `tf_mm_to_dali_voxel(m_mm, shape, spacing, pc)` (keeps 3×4 layout), `brain_second_moment(mask, spacing)`.
- `data_loading.postprocess.LinearPostprocess(task, level, flip_prob, p_pair, seed, target_iters)`: called as `(images, T_dali[, Rt_dali])`, returns `images, T_render, T_target, A_star, pair_i, pair_j, P_star, flip_axes`. It owns a seeded `tf.random.Generator` (flips, pairs) and works inside `tf.function`. `brain_S_matrix(level, template_dir)` gives S (float64).
- `data_loading.dali_loader.linear_output_signature(task, level, N)` gives the static DALI shapes (rigid: images, T_dali, Rt_dali; affine: images, T_dali; all float32). `DALI_AVAILABLE`, `require_dali()`. DALI `.npy` images are reshaped to `[X, Y, Z, 1]`.
- `data_loading.data_module`: `read_manifest`, `manifest_files(rows, level, split, kind, contrast)`, `LinearDataModule(...).iterator(task)`, `template_dir`. The export already contains only the 39 T1w val cases (export default `--contrast T1w`).
- `models.group_net.GroupRegLevelModel(level, config)`: `call(images, task, training)` → dict with float32 `A`, `A_raw`, `params`; `build_model(n)`, `set_bn_stats_frozen`, `count_params_by_part`. Rigid steps give `scale_shear_head` a `None` gradient. BN `stats_frozen` is read at trace time.
- `models.layers.transform_image(images, M, spacing)` pull warp: out_i(y) = in_i(M_i y). Rendering I_i(y) = I(T_i⁻¹ y) is `transform_image(I, affine_inverse(T))`.
- Trainable variable names: `.../gamma`, `.../beta` (AffineParams of every BN), `.../bias` (conv/dense/heads), `.../kernel`. BN moving stats are non-trainable (17,984 values).
- Reference defaults (rigid `runtime/args.py`, `main2.py`): SGDW lr 0.01, momentum 0.99, dampening 0, Nesterov on, coupled weight decay 3e-5 (2.5e-5 in main2), `global_clipnorm=12`, warmup 1000 iterations, exclude list `['normalization', 'bias']` (off by default there), cosine-cycle schedule (cycle ×1.5 length, peak and floor ÷10 per cycle), poly power 0.9, `str2bool`. The reference SGDW `_use_weight_decay` does `variable in list`, which compares tf.Variables elementwise: port it with identity comparison.
- Local env: `tf_gpu` has tensorboard, no dllogger, tqdm, tensorflow_addons or DALI. Two GTX 1080 (8 GB).
- `tests/run_tests_wsl.sh` runs the py3.8 syntax check over `PYFILES` and `bash -n` on GPU scripts, then `python -m unittest discover -s tests -t . -v`. Every TF test imports `tests.tf_setup` first. Tests free memory with `tf.keras.backend.clear_session()` + `gc.collect()` (`tests/test_group_net.py:free_memory`).
- `.gitignore` already ignores `results/`, `logs/`, `ckpt/`, `*.npy`.

## Decisions (with rationale)

1. **NaN guard = `tf.cond` around `apply_gradients` inside the one per-task `tf.function`, plus BN-statistics rollback.** Keras' own LossScaleOptimizer wraps `apply_gradients` in a cond the same way, so it traces in TF 2.10. Training-mode BN updates its moving statistics during the forward pass, so a NaN batch would poison them even if the update is skipped. The step snapshots `model.non_trainable_variables` (about 18 k floats) before the forward pass and restores them in the skip branch. Under AMP the LossScaleOptimizer must still see non-finite gradients to lower its scale, so with `--amp` the step always calls `apply_gradients` (it skips internally) and only the BN rollback and logging use our finite flag. Contingency if a cond around `apply_gradients` fails to trace: split into a compute `tf.function` that returns the gradients and a per-task apply `tf.function` gated by a Python check of one scalar. Record it in the report if used.
2. **Build the optimizer on ALL trainable variables before the first trace** (`optimizer.build(model.trainable_variables)`, inner optimizer for LSO). Otherwise a rigid-first run creates no slots for `scale_shear_head`, and a later affine step fails the experimental optimizer's variable-key lookup. None gradients are filtered statically (at trace time) before unscaling and applying (finding a).
3. **The step input is the raw DALI tuple; post-processing runs inside the step, before the tape.** One signature per task comes from `linear_output_signature(task, level, N)`, which pins the fixed N per level. The overfit and synthetic sources emit the same DALI-format tuple (T_mm → `tf_mm_to_dali_voxel` 3×4), so the local tests exercise exactly the code path DALI uses.
4. **Task schedule is stateless per step:** `task = draw(seed, step)` with `random.Random(seed * 1_000_003 + step)` and cumulative weights. It's deterministic, and resuming from a checkpoint reproduces the sequence with no saved RNG state. `deform` in the ratio raises `NotImplementedError` ("milestone 6") at parse time. Affine is supported now (the pipeline and heads already exist).
5. **LR is set from Python every step**, `set_learning_rate(opt, lr_at(step))`, assigning the optimizer's LR variable (inner optimizer under LSO). That keeps the schedule a pure function of the step and adds no retrace.
6. **Optimizer defaults: the reference values** (SGDW, lr 0.01, momentum 0.99, Nesterov, coupled wd 3e-5, `global_clipnorm` 12), with **weight-decay exclusion ON by default** for names matching `gamma`, `beta`, `bias` (our names; the reference's `normalization` does not occur here). Rationale: design 5.2 says to port SGDW. Caveat: our loss is in mm² (≈ 10²–10³ at init), so early gradients are always clipped, and lr 0.01 at momentum 0.99 may be too hot. Item 9 runs a sanity check. If SGDW at 0.01 diverges or stalls on the synthetic overfit, the coder lowers the default to the largest of {3e-3, 1e-3} that trains and records it. Adam, AdamW and SGD come from `tf.keras.optimizers.experimental` as options. Optimizer `jit_compile` follows `--xla` (default off).
7. **Schedule default:** `cosine` with linear warmup (`warmup_steps` 1000, as in the reference) down to `min_lr = lr × 1e-3` at `--steps`. The debug job uses `--warmup-steps 200`. Also `poly` (power 0.9, end lr × 1e-3), `cycle` (port of `CustomLearningRateSchedules`: cycle length × 1.5, peak and floor ÷ 10 per cycle) and `constant`.
8. **Loss weights:** λ_pair = 1, p_pair = 0.1, flip probability 0.5 per axis, target iterations 10 (design 6.1, 3.3). The loss stays in mm² as designed; RMS (√ of the means) is logged.
9. **Overfit mode** (`--overfit-one-group`): one fixed group of N images, fixed pure-rigid transforms (±45° random axis, ±15 mm, no nuisance), rendered TF-side, with flips off and intensity aug off. **Dropout defaults to 0 in overfit mode** (`--dropout` given explicitly wins), because 15% dropout noise on 512 features would hold the RMS above 1 mm. Training-mode BN on a fixed group is deterministic. Stop criterion: mean of the last 10 step `center_rms_mm` < `--overfit-threshold-mm` (1.0), or `--overfit-max-steps` (1500). `overfit_summary.json` holds the curve. Exit 0 if reached, else 1.
10. **Validation:** a fixed set with deterministic groups and fixed seeded transforms (numpy, design ranges incl. rigid nuisance), rendered once with `transform_image`, flips and intensity aug off, cached on the host as float32. Targets are computed once in float64. Val group size = `min(N, n_val)` (`--val-group-size` overrides; it must be ≤ n_val so no group holds duplicates). Groups use a seeded permutation, with group g taking `perm[(g·Nv + k) % n_val]`. Default `val_groups = ceil(n_val / Nv)`, so every image appears at least once. Forward is a separate `tf.function` per task (eval mode, its own signature). Best checkpoint metric: `val_rigid/pair_rms_mm` (lower is better; configurable `--best-metric`).
11. **Logging:** dllogger if importable (JSONStreamBackend → `logs/dllogger.json`, plus stdout unless `--quiet`), else our JSON-lines backend → `logs/metrics.jsonl`. TensorBoard scalars always go to `tb/`. Run dir: `<results>/<run_name>/{logs,tb,ckpt/last,ckpt/best,config.json}`.
12. **Not restructuring the workflow into FEATs.** The FEAT coder contract requires per-FEAT local commits, and the user forbade commits in this workflow. The existing implement-and-review loop runs this plan as one unit.
13. **Unit-test model size:** the trainer and overfit unit tests use a narrow `ModelConfig` (filters 16,32,32,48,64,96; reductions 64,48,24,24; dense 512,256). That keeps the suite fast on the GTX 1080 and avoids the full-width cuDNN autotune stalls seen in milestone 3. The full-width overfit is run once via `main.py` and recorded in the report, not in the suite.

## Public APIs (new modules)

`runtime/__init__.py`: docstring only, no imports.

`runtime/losses.py` (TF, float32/64; no Keras state):
- `linear_center_loss(A, A_star, S) -> scalar`: `mean_i point_loss(A_i − A*_i, S)` (mm²).
- `linear_pair_loss(A, P_star, pair_i, pair_j, S) -> scalar`: `mean_p point_loss(pair_predictions(A, i, j) − P*_p, S)` (mm²). P* comes from postprocess (`pair_targets` of T_target, float64 → float32).
- `linear_loss(A, batch, S, lambda_pair=1.0) -> dict(loss, center, pair, center_rms_mm, pair_rms_mm)`, where `batch` is the postprocess dict. Inputs are cast to float32 (A is float32 even under mixed precision; S is passed as a float32 constant).

`runtime/metrics.py` (numpy float64, pure functions plus an accumulator):
- `rotation_angle_deg(L)`: angle of the polar (SVD U Vᵀ) rotation of a 3×3 matrix (arccos((tr − 1)/2), clipped).
- `matrix_errors(M, M_true, S) -> dict(rms_mm, rot_deg, trans_mm)` per matrix: rms = √tr(ΔM S ΔMᵀ), rot = angle of polar(L_M) polar(L_true)ᵀ, trans = ‖t_M − t_true‖ (displacement of the rotation center).
- `centered_errors(A, A_star, S)`: per image.
- `pairwise_errors(A, T_target, S)`: ALL ordered pairs i ≠ j, predicted A_j⁻¹A_i vs true T_jT_i⁻¹.
- `cycle_errors(A, S, n_triplets=64, seed=0)`: random distinct triplets, C = P_ki P_jk P_ij (i → j → k → i), rms of C − I.
- `group_metrics(A, A_star, T_target, S, prefix) -> dict` of means plus p95 of rms, keyed `<prefix>/center_rms_mm`, `center_rot_deg`, `center_trans_mm`, `pair_rms_mm`, `pair_rot_deg`, `pair_trans_mm`, `cycle_rms_mm`.
- `MeanAccumulator`: `update(dict)`, `result() -> dict`, `reset()`. For train-step running means and pooling val metrics over groups (val metrics pool per-matrix values over all groups, then take means).

`runtime/schedulers.py` (pure Python, no TF):
- `ScheduleConfig(name='cosine', base_lr=0.01, total_steps, warmup_steps=1000, min_lr_factor=1e-3, power=0.9, cycle_steps=None, cycle_mult=1.5, cycle_decay=10.0)`.
- `lr_at(step, cfg) -> float`:
  - warmup: base·max(step, 1e-8 floor)/warmup, linear from ≈ base/1e8 to base at `step == warmup_steps`;
  - then by name: cosine (min + ½(base − min)(1 + cos πp)), poly (end + (base − end)(1 − p)^power), cycle (the reference port), constant;
  - p = clip((step − warmup)/(total − warmup), 0, 1).

`runtime/optimizers.py`:
- `SGDW(tf.keras.optimizers.experimental.Optimizer)`: port of the reference (`build`, `update_step`, `get_config`, `exclude_from_weight_decay(var_list, var_names)`), with `_use_weight_decay` by identity (`any(v is variable ...)`) plus regex names.
- `DEFAULT_WD_EXCLUDE = ('gamma', 'beta', 'bias')`.
- `make_optimizer(name, lr, weight_decay, momentum, dampening, nesterov, decoupled, clipnorm, exclude_wd=True, amp=False, jit_compile=False)`. Names: sgdw, sgd, adam, adamw (experimental Keras). Exclusion applies to sgdw/adamw. Wraps `tf.keras.mixed_precision.LossScaleOptimizer` when `amp`.
- `inner_optimizer(opt)`, `build_optimizer(opt, variables)`, `set_learning_rate(opt, value)`, `get_learning_rate(opt)`, `uses_weight_decay(opt, variable) -> bool`.

`runtime/logging.py`:
- `JsonLinesBackend(path)`: writes `{"step": ..., "time": ..., "data": {...}}` per line; `"PARAMETER"` step for hyperparameters.
- `RunLogger(run_dir, backend='auto'|'dllogger'|'jsonl', tensorboard=True, quiet=False)` with `log_params(dict)`, `log_metrics(step, dict)` (backend line + `tf.summary.scalar` per numeric key), `flush()`, `close()`. Attributes `.backend_name`, `.metrics_path`. Values are sanitized to JSON (numpy/tf scalars → float).
- `make_run_dir(results, run_name) -> dict of paths` (logs, tb, ckpt_last, ckpt_best, config).

`runtime/checkpoint.py`:
- `RunCheckpoint(ckpt_dir_last, ckpt_dir_best, model, optimizer, step, extra=None, keep_last=2)`. It holds `tf.train.Checkpoint(model, optimizer, step, best_metric (tf.Variable float64 = +inf), **extra)`, where `extra` is the postprocess generators. Methods:
  - `save_last()`, `maybe_save_best(value) -> bool` (lower is better, keep 1);
  - `restore_last() -> bool` (True if restored; `assert_existing_objects_matched`);
  - `restore_best()`; `latest_last` / `latest_best`.
- `write_config(path, model_config, args_dict, extra=None)`: JSON of model config, args, python/tf/numpy versions and the git commit when available.
- `check_config_compatible(path, model_config)`: raises ValueError naming the differing keys on resume.

`runtime/fixed_data.py` (TF-side data that needs no DALI):
- `smooth_phantom(shape, rng, n_blobs=8, sigma=4.0)`: asymmetric Gaussian blobs inside an ellipsoid envelope; returns `(volume float32 in [0, 1], mask bool)`.
- `random_linear_transforms(rng, n, task, ranges=None, nuisance=True) -> (T, T_target)`: float64 [n,4,4], TF centered mm. Rotation U(±r) about a uniform random axis, translation U(±t) per axis, K = U·D (log-scale, shear). T = [R K | t], Rt = [R | t]; target = Rt (rigid) or T (affine). This matches the DALI semantics (T = C+ Tr R K C− is [RK | t] in centered mm).
- `render_group(template_or_images, T, spacing) -> float32 [n,X,Y,Z,1]` via `transform_image(x, affine_inverse(T))`.
- `to_dali_batch(images, T, T_target, task, level)`: the DALI-format tuple (3×4 via `tf_mm_to_dali_voxel`; rigid adds Rt).
- `load_level_images(export_root, level, split, n=None, seed=0, contrast=None)`: numpy `np.load` of manifest files → float32 [n,X,Y,Z,1], with a seeded selection.
- `FixedGroupSource(batch_tuple)`: an iterator that yields the same tensors forever.
- `ValidationSet.build(images, task, level, group_size, n_groups, ranges, seed, target_iters=10)`: holds per group `images`, `T_target`, `A_star` (float64), with `.groups`, `.task`, `.group_size`.

`runtime/run.py`:
- `parse_task_ratio('rigid:1,affine:1') -> OrderedDict`, `TaskSchedule(ratio, seed)` with `draw(step) -> task`.
- `TrainConfig` dataclass: level, group_size, steps, val_every, log_every, ckpt_every, task_ratio, lambda_pair, p_pair, flip_prob, target_iters, amp, seed, schedule (ScheduleConfig), nonfinite_patience=20, best_metric, overfit (bool), overfit_threshold_mm, overfit_window=10.
- `Trainer(model, optimizer, sources: {task: iterator}, S, cfg, logger=None, checkpoint=None, val_sets: {task: ValidationSet} or None)`. Attributes: `step` (tf.Variable int64), `postprocess[task]`, `trace_counts[task]`, `nonfinite_total`, `history` (list of per-step dicts for overfit/tests). Methods:
  - `rebuild_steps()`: re-creates every train and val `tf.function` and resets nothing else (finding b);
  - `train_step(task, batch_tuple) -> dict of Python floats`;
  - `validate() -> dict`;
  - `fit(num_steps=None) -> dict summary`;
  - `checkpoint_variables() -> dict` (for `RunCheckpoint` extra: the postprocess generators).

### Train step (one `tf.function` per task, traced once)

```
signature = [tf.TensorSpec(s, tf.float32) for s in linear_output_signature(task, level, N)[0]]
@tf.function(input_signature=signature)
def step(*dali):                       # rigid: images, T_dali, Rt_dali ; affine: images, T_dali
    self.trace_counts[task] += 1       # Python side effect: runs only while tracing
    batch = self.postprocess[task](*dali)                     # outside the tape
    backup = [tf.identity(v) for v in model.non_trainable_variables]
    with tf.GradientTape() as tape:
        out = model(batch["images"], task=task, training=True)
        terms = losses.linear_loss(out["A"], batch, S, lambda_pair)
        loss = terms["loss"]
        scaled = opt.get_scaled_loss(loss) if amp else loss
    grads = tape.gradient(scaled, model.trainable_variables)
    pairs = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]  # static
    gs = [g for g, _ in pairs]
    if amp: gs = opt.get_unscaled_gradients(gs)
    gnorm = tf.linalg.global_norm(gs)
    finite = tf.math.is_finite(loss) & tf.math.is_finite(gnorm)
    if amp: opt.apply_gradients(zip(gs, vars_))               # LSO skips non-finite itself
    else:   tf.cond(finite, apply, tf.no_op)
    tf.cond(finite, tf.no_op, restore_bn(backup))             # BN stats rollback
    self.step.assign_add(1)
    return dict(terms, grad_norm=gnorm, finite=finite, n_vars_updated=len(pairs))
```

`fit` loop per step (Python):
1. `task = schedule.draw(step)`.
2. `set_learning_rate(opt, lr_at(step))`.
3. Timed `batch = next(sources[task])` (data wait).
4. Timed `out = steps[task](*batch)`, then `float(out['loss'])` (compute, including the host sync).
5. Accumulate. On a non-finite step, log a warning record and count it. Raise `RuntimeError` after `nonfinite_patience` consecutive skips.
6. Every `log_every` steps: log the means of loss/center/pair RMS per task (`train_<task>/...`), lr, grad norm, data/compute ms, peak GPU memory (`tf.config.experimental.get_memory_info('GPU:0')['peak']` when a GPU exists), and the non-finite count.
7. Every `val_every`: `validate()` then `checkpoint.maybe_save_best(metrics[best_metric])`.
8. Every `ckpt_every` and at the end: `save_last()`.
9. Overfit mode: stop when the windowed mean `center_rms_mm` < threshold.

`fit` starts from `int(step)`, so a resumed run continues the step count.

Val forward: `tf.function(input_signature=[TensorSpec([Nv, X, Y, Z, 1])])` returns `model(x, task, training=False)['A']`. Metrics are computed in numpy float64 with `group_metrics` and pooled over groups.

`runtime/args.py` (TF-free; `python main.py --help` must work without TF):
- `str2bool`, `int_list`, `build_parser()`, `parse_args(argv=None)` (validation plus resolved per-level defaults), `model_config_from_args(args) -> ModelConfig`, `train_config_from_args(args) -> TrainConfig`, `ranges_from_args(args) -> {task: dict}`, `args_to_dict(args)`.

Flags (default):
- `--data` (None; required unless `--overfit-one-group --overfit-synthetic`), `--results` (./results), `--run-name` (auto `l<level>_<ratio>_<YYYYmmdd-HHMMSS>`), `--resume` (True: restore the last checkpoint if one exists).
- `--level` 4, `--group-size` (None → {4: 32, 2: 8, 1: 2}), `--steps` 50000, `--val-every` 250, `--val-groups` (None → auto), `--val-group-size` (None), `--val-seed` 1234, `--log-every` 50, `--ckpt-every` (None → val_every), `--best-metric` (val_rigid/pair_rms_mm).
- `--optimizer` sgdw, `--lr` 0.01, `--momentum` 0.99, `--dampening` 0, `--nesterov` True, `--decoupled` False, `--weight-decay` 3e-5, `--exclude-from-weight-decay` True, `--clipnorm` 12.
- `--schedule` cosine, `--warmup-steps` 1000, `--min-lr-factor` 1e-3, `--power` 0.9, `--cycle-steps` (None → steps/4).
- `--amp` False, `--xla` False.
- `--task-ratio` 'rigid:1', `--lambda-pair` 1.0, `--p-pair` 0.1, `--flip-prob` 0.5, `--target-iters` 10.
- Augmentation ranges: `--rigid-rotation-deg/-translation-mm/-log-scale/-shear` and `--affine-...`, with defaults from `data_loading.params.DEFAULT_LINEAR_RANGES`. `--intensity-aug` True.
- DALI: `--num-threads` 4, `--prefetch` 1, `--shuffle-mode` epoch.
- Model: `--group-config` A, `--block` single, `--norm` batch, `--dense-norm` batch, `--filters`, `--reductions`, `--dense` (comma lists; ModelConfig defaults), `--dropout` (None → 0.15, or 0.0 in overfit mode), `--dense-group-concat-first` False, `--negative-slope` 0, `--translation-scale` 1, `--centering-iters` 4, `--bn-momentum` 0.99.
- `--seed` 0, `--nonfinite-patience` 20, `--quiet` False, `--log-backend` auto.
- Modes: `--dry-run` (2 steps + 1 validation of 1 group, then exit 0) and `--overfit-one-group` with `--overfit-synthetic` (False), `--overfit-threshold-mm` 1.0, `--overfit-max-steps` 1500.

`main.py`: `main(argv=None) -> int`, plus `if __name__ == '__main__': sys.exit(main())`. Order:
1. Parse args (no TF).
2. Unless overfit mode: if `data_loading.dali_loader.DALI_AVAILABLE` is False, `raise SystemExit("main.py: nvidia.dali is not importable (...). Run inside tf10_nv.sif, or use --overfit-one-group (TF-side warp, no DALI needed).")`. This fires before any TF model work.
3. Import TF; `configure_tf_precision(tf32=False, memory_growth=True)` FIRST.
4. Seeds (random, numpy, tf).
5. `--amp` → `mixed_float16` global policy; `--xla` → `tf.config.optimizer.set_jit(True)`.
6. Data:
   - DALI mode: `LinearDataModule` iterators per task in the ratio; val images via `load_level_images(split='val')` → a `ValidationSet` per task; S from `brain_S_matrix(level, template_dir)`.
   - Overfit mode: real (`load_level_images(split='train', n=N)`) or synthetic phantom images; fixed rigid T; `FixedGroupSource`; a validation set of that one group; S from the template mask (real) or the phantom mask (synthetic).
7. Model `GroupRegLevelModel(level, cfg).build_model(N)`; log `count_params_by_part()`.
8. Optimizer → `build_optimizer`.
9. Run dir, `RunLogger`, `write_config`, `RunCheckpoint`. Resume: `check_config_compatible` + `restore_last`.
10. `Trainer.fit`.
11. Overfit: write `overfit_summary.json` (curve of step, center_rms_mm, plus the eval-mode metrics at each validation; best and reached). Return 0/1.

The model and policy are reset in tests by the caller.

## Ordered implementation items

- [ ] 1. **Pure-Python scaffolding: `runtime/__init__.py`, `runtime/schedulers.py`, `runtime/args.py`.** Implement `ScheduleConfig`/`lr_at` (warmup + cosine/poly/cycle/constant) and the full argparse surface above (TF-free; `TrainConfig` imported lazily or defined as a plain dict builder so `args.py` never imports TF). Use `str2bool` and comma-list parsers, resolve per-level N and the overfit dropout default, and validate (e.g. `deform` in the ratio → error naming milestone 6; `--data` required outside synthetic overfit).
      Files: `runtime/__init__.py`, `runtime/schedulers.py`, `runtime/args.py`, `tests/test_runtime_optim.py` (scheduler part), `tests/test_runtime_infra.py` (args part).
      Verify: in WSL `bgr_preproc38`, `python -c "import runtime.args, runtime.schedulers"` (no TF). Then `python -m unittest tests.test_runtime_optim tests.test_runtime_infra -v` in `tf_gpu`. Tests: scheduler start ≈ 0, `lr_at(warmup) == base`, `lr_at(total) == base·min_factor` for cosine and poly, monotone non-increasing after warmup, cycle restarts at the cycle boundary with peak/10, constant. Args: `str2bool` variants, the per-level N defaults, the overfit dropout default 0.0 vs explicit `--dropout`, `model_config_from_args` round trip, and the ratio parse error for `deform`.

- [ ] 2. **`runtime/losses.py` and `runtime/metrics.py`** with the APIs above.
      Files: `runtime/losses.py`, `runtime/metrics.py`, `tests/test_runtime_losses_metrics.py`.
      Verify: `python -m unittest tests.test_runtime_losses_metrics -v` (tf_gpu). Tests:
      - (a) center loss vs `np_reference.mean_sq_displacement_bruteforce` averaged over images, on a random synthetic level-4-like mask (spacing 4 mm; S from `brain_second_moment`; rtol 1e-5 float32, 1e-10 float64), checking mm² units (a pure 3 mm translation → 9 mm², RMS 3 mm);
      - (b) pair loss vs a numpy loop over the sampled pairs (A_j⁻¹A_i − T_jT_i⁻¹, brute-force displacement), plus `linear_loss` total = center + λ·pair and rms = √;
      - (c) finite gradients w.r.t. A;
      - (d) metrics: `rotation_angle_deg` on known θ about a random axis (1e-6 deg); translation error = norm of the t difference; `pairwise_errors` vs an explicit double loop over all N(N−1) ordered pairs; cycle error ≈ 0 (< 1e-6 mm float64, < 1e-3 mm from float32 A) and triplets distinct; `group_metrics` keys/prefix; `MeanAccumulator` means.

- [ ] 3. **`runtime/optimizers.py`**: the SGDW port, `make_optimizer`, LR get/set helpers (plain and LSO-wrapped), `build_optimizer`, `uses_weight_decay`.
      Files: `runtime/optimizers.py`, `tests/test_runtime_optim.py` (optimizer part).
      Verify: `python -m unittest tests.test_runtime_optim -v`. Tests:
      - exclusion list on a built narrow `GroupRegLevelModel`: every `gamma`/`beta`/`bias` excluded, every `kernel` included, and the count matches;
      - functional check with SGDW (momentum 0, decoupled, lr 1, wd 0.1) and zero gradients: kernels shrink by 10%, biases and gammas unchanged;
      - each optimizer name builds and applies one step;
      - `set_learning_rate` changes the value read by `get_learning_rate` for plain and LSO-wrapped optimizers;
      - `global_clipnorm` is set to 12 by default.

- [ ] 4. **`runtime/logging.py` and `runtime/checkpoint.py`.**
      Files: `runtime/logging.py`, `runtime/checkpoint.py`, `tests/test_runtime_infra.py` (logging + checkpoint parts).
      Verify: `python -m unittest tests.test_runtime_infra -v`. Tests:
      - logging: with `backend='jsonl'`, `log_params` + 3× `log_metrics` write 4 parseable JSON lines with the right steps and float values; TB event files appear in `tb/`; `make_run_dir` layout;
      - checkpoint (narrow model, SGDW): run 2 manual optimizer steps, `save_last`. A fresh model + optimizer (built) + `restore_last()` gives every model variable (incl. BN stats) and every optimizer variable bit-exactly equal (`np.array_equal`), with equal `step` and `optimizer.iterations`;
      - `maybe_save_best` saves only on improvement and `best_metric` survives a restore; keep_last = 2 prunes older checkpoints;
      - `write_config` / `check_config_compatible` raises on a changed `ModelConfig` field.

      All use `tempfile.mkdtemp()` and clean up.

- [ ] 5. **`runtime/fixed_data.py`**: phantom, numpy transform sampler, TF render, DALI-format conversion, manifest numpy loader, `FixedGroupSource`, `ValidationSet`.
      Files: `runtime/fixed_data.py`, `tests/test_runtime_infra.py` (fixed-data part).
      Verify: `python -m unittest tests.test_runtime_infra -v`. Tests:
      - sampled transforms respect the ranges (rotation angle ≤ r + 1e-6, |t| ≤ t; rigid target orthonormal; Rt⁻¹T upper-triangular for nuisance);
      - `to_dali_batch` → `LinearPostprocess(flip_prob=0)` round trip recovers T_target to 1e-4 mm and A* centered;
      - `load_level_images` on a tiny fake export in a temp dir (manifest CSV with `image_l4` columns + level-4 `.npy` files + `template/template_mask_4.npy`) returns [n,40,56,48,1] float32 deterministically for a seed;
      - `ValidationSet.build` gives fixed group composition and identical images across two builds with the same seed, Nv ≤ n_val enforced.

- [ ] 6. **`runtime/run.py`: `parse_task_ratio`, `TaskSchedule`, `TrainConfig`, `Trainer`** (train steps, NaN guard + BN rollback, `rebuild_steps`, validate, fit, logging cadence, checkpoint hooks, overfit stopping). Depends on items 1-5.
      Files: `runtime/run.py`, `tests/test_trainer.py`.
      Verify: `python -m unittest tests.test_trainer -v`. All tests use the narrow config, level 4, N = 4, synthetic `FixedGroupSource` batches, `free_memory()` in tearDown:
      - task schedule: ratio `rigid:3,affine:1` gives frequency 0.75 ± 0.02 over 20 k draws; the same seed gives an identical sequence; `draw(step)` is independent of the call order (resume determinism); `deform` raises.
      - None gradients: one rigid step runs without error. `scale_shear_head` kernel/bias are bit-unchanged, `rigid_head` and encoder kernels changed, `n_vars_updated == len(trainable) − 2`.
      - NaN guard: a batch with a NaN image returns `finite=False`. Every trainable variable, every BN moving statistic and `optimizer.iterations` are unchanged, and `nonfinite_total == 1`. The next finite step applies. `nonfinite_patience=2` with three NaN batches raises `RuntimeError`.
      - Trace count: `fit(6)` with ratio `rigid:1,affine:1` (both tasks drawn) gives `trace_counts == {'rigid': 1, 'affine': 1}`.
      - `rebuild_steps` regression (finding b): after training steps, `model.set_bn_stats_frozen(True)` + `rebuild_steps()` + 2 steps leave every moving mean/var unchanged while gamma/beta change. Without `rebuild_steps()` the already-traced step still updates the stats; assert that too, to document the trap.
      - Resume: `fit(3)` + `save_last`; a new trainer + restore → `int(step) == 3`, then `fit(2)` ends at step 5 and the log shows steps 4-5.
      - Validation: `validate()` on a 1-group synthetic `ValidationSet` returns all `val_rigid/*` keys, finite; the identity model gives center RMS equal to the numpy RMS of A* vs I.
      - AMP smoke: under a `mixed_float16` policy (restored in `finally`) with LSO, 2 steps finite and losses float32.

- [ ] 7. **`main.py`** (the `main(argv)` flow above), wiring args → data → model → optimizer → logger/checkpoint → Trainer. Also the overfit and dry-run modes and `overfit_summary.json`.
      Files: `main.py`, `tests/test_main.py`.
      Verify: `python -m unittest tests.test_main -v`. Tests:
      - (a) with DALI unavailable (skip if `DALI_AVAILABLE`), `main(['--dry-run', '--data', <tmp>])` raises `SystemExit` whose message contains `nvidia.dali` and `--overfit-one-group`;
      - (b) `main(['--overfit-one-group', '--overfit-synthetic', '--dry-run', ...narrow...])` returns 0 after 2 steps + 1 validation, writing `config.json`, a jsonl log and a `ckpt/last` checkpoint;
      - (c) `--overfit-one-group --data <fake export>` (real-data loader path, 20 steps) runs and writes its summary;
      - (d) **the overfit test**: `--overfit-one-group --overfit-synthetic --level 4 --group-size 4`, narrow config, `--optimizer adam --lr <tuned> --schedule constant --warmup-steps 0 --overfit-max-steps 1500`, asserting `overfit_summary.json` has `reached == true` (10-step mean center RMS < 1.0 mm) and printing the RMS curve every 50 steps. LR tuning: try 3e-4 first, then 1e-4 or 1e-3; record what was tried. If < 1 mm is not reachable within 1500 steps, assert the best achievable bound measured (e.g. < 2 mm), mark the gap clearly, and explain in the report (candidates: `translation_scale` 10 if translation lags; BN-on-N=4 noise).

      Also run `python main.py --help` in `bgr_preproc38` (no TF import at parse time).

- [ ] 8. **SLURM and glacier scripts.**
      - `slurm/train_debug.sh`, an sbatch template that is never submitted here:
        - header: `#!/bin/bash`, `-A qtim`, `-p rtx8000` (commented alternatives `dgx-a100` with `--cpus-per-task` up to 16, and `pubgpu`), `--gpus=1`, `-N 1 --ntasks-per-node=1`, `--cpus-per-task=3`, `--mem=64G`, `--time=01:00:00`, `--job-name=bgr-m4-debug`, `--output`/`--error` under `/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.{out,err}`;
        - a comment that the logs dir must exist before `sbatch` (`mkdir -p` it);
        - `set -euo pipefail`; env-overridable `PROJ`, `REPO`, `DATA` (`$PROJ/data/npy_v2`), `RESULTS` (`$PROJ/results`), `SIF`, `RUN_NAME` (default `m4_debug_$SLURM_JOB_ID`), `LEVEL` 4, `N` 32, `STEPS` 2000, `VAL_EVERY` 250, `WARMUP` 200, `TASK_RATIO` rigid:1, `GROUP_CONFIG` A, `AMP` false, `EXTRA_ARGS`;
        - exports `TMPDIR=/scratch/$SLURM_JOB_ID`, `KERAS_HOME=$PROJ/scratch/keras`, `CUDA_CACHE_PATH=$PROJ/scratch/cuda_cache`, `PYTHONDONTWRITEBYTECODE=1`, plus the `SINGULARITYENV_*` copies;
        - logs `hostname`, `nvidia-smi`, `git rev-parse`;
        - `singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF" python main.py --data "$DATA" --results "$RESULTS" --run-name "$RUN_NAME" --level ... --num-threads 3 ... $EXTRA_ARGS </dev/null`.

        The header comment documents the expected runtime as an estimate derived from the local GTX 1080 step time measured in item 9 (scaled, stated as unmeasured on RTX 8000), the peak-memory extrapolation for N = 32, the fallback (N = 16 or `AMP=true` on OOM), and that `last` is saved every 250 steps so a timeout keeps results. No `-B /usr/local/freesurfer`: training doesn't use FreeSurfer, and that path may not exist on the compute nodes.
      - `tests/gpu/run_overfit_glacier.sh <gpu_index> [timeout_s] [overfit|dryrun]`, mirroring `run_verify_glacier.sh`:
        - integer validation; `CUDA_VISIBLE_DEVICES`; everything under `$PROJ/scratch/m4` (TMPDIR, CUDA cache, KERAS_HOME, results);
        - `timeout` + `</dev/null` + tee to `scratch/m4/<mode>.log`;
        - overfit mode: `main.py --overfit-one-group --data $PROJ/data/npy_v2 --level 4 --group-size 8 --overfit-max-steps 3000 --results $SCRATCH/results`;
        - dryrun mode: `main.py --dry-run --data ... --level 4 --group-size 8 --results $SCRATCH/results` (real DALI pipeline).

        TF32 is turned off by `main.py`.
      - Update `tests/run_tests_wsl.sh`: add `runtime/*.py main.py` to `PYFILES`, add `bash -n` for both new scripts, and add the TF-free check `python -c "import runtime.args, runtime.schedulers"` plus `python main.py --help >/dev/null` in `bgr_preproc38`.
      Files: `slurm/train_debug.sh`, `tests/gpu/run_overfit_glacier.sh`, `tests/run_tests_wsl.sh`.
      Verify: `wsl -d Ubuntu bash tests/run_tests_wsl.sh` stage 1 passes (py_compile, ast 3.8, `bash -n` on all three scripts, TF-free imports, `--help`).

- [ ] 9. **Full local verification, measurements and report.**
      - Create `.agents/tasks/milestone-4/record_tests.sh` (copy the milestone-3 pattern, writing `test-log.txt`) and run it. The full suite (previous 94 tests + new) must pass, with only DALI skips.
      - Run `python main.py --overfit-one-group --overfit-synthetic --level 4 --group-size 4 --results results/m4_local` once with the **default full-width config** (Adam at the tuned lr) and once with **SGDW defaults** (300-500 steps sanity check per decision 6; adjust the default lr if needed). Copy each `overfit_summary.json` to `.agents/tasks/milestone-4/`.
      - Measure the local step time and peak memory (level 4, N = 4, full width) for the SLURM estimate.
      - Write `.agents/tasks/milestone-4/report.md`: files, public API, the decisions above (and any deviations, such as a changed default lr or the cond fallback), the test → code map, the overfit curves (step vs RMS table every 50-100 steps; best and reached), the measured step time and memory with the RTX 8000 extrapolation, and the exact commands for Jay/orchestrator (glacier scripts, sbatch template with resources).

        Not verified: DALI path, container py3.8.10/numpy 1.21.1 runtime, A100/RTX 8000 timing, XLA, AMP speed.
      - Delete local `results/m4_local` checkpoints after copying the summaries (keep the folder out of git; it is ignored anyway).
      Files: `.agents/tasks/milestone-4/record_tests.sh`, `test-log.txt`, `report.md`, `overfit_summary_*.json`.
      Verify: `wsl -d Ubuntu bash .agents/tasks/milestone-4/record_tests.sh` exits 0, and the log timestamps postdate the last source edit.

## Test → code map

| Required test | File / test | Code under test |
|---|---|---|
| Losses vs numpy brute force (center, pair, mm², S-weighted over mask) | `tests/test_runtime_losses_metrics.py` | `runtime/losses.py` |
| Metrics vs numpy (rotation, translation, pairwise, cycle) | `tests/test_runtime_losses_metrics.py` | `runtime/metrics.py` |
| Scheduler shape | `tests/test_runtime_optim.py` | `runtime/schedulers.py` |
| Weight-decay exclusion | `tests/test_runtime_optim.py` | `runtime/optimizers.py` |
| None-gradient filtering (rigid) | `tests/test_trainer.py` | `Trainer` step |
| NaN guard skips update (+ BN rollback, patience) | `tests/test_trainer.py` | `Trainer` step / fit |
| Task schedule ratio + determinism | `tests/test_trainer.py` | `TaskSchedule` |
| Checkpoint round trip (bit-exact, optimizer step, resume continues) | `tests/test_runtime_infra.py`, `tests/test_trainer.py` | `runtime/checkpoint.py`, `Trainer.fit` |
| Logging writes JSON lines | `tests/test_runtime_infra.py` | `runtime/logging.py` |
| Trace count == 1 per task | `tests/test_trainer.py` | `Trainer` step functions |
| `rebuild_steps()` after BN freeze (finding b) | `tests/test_trainer.py` | `Trainer.rebuild_steps` |
| Overfit < 1 mm on synthetic N = 4 in ≤ 1500 steps | `tests/test_main.py` | `main.py` overfit mode, `runtime/fixed_data.py` |
| `--dry-run` without DALI fails clearly | `tests/test_main.py` | `main.py` |

## Open points and recommendations (for Jay, non-blocking)

1. **Default SGDW lr.** The reference value 0.01 (momentum 0.99) was tuned for a different loss scale. Recommendation: keep it as the default unless item 9's sanity run shows divergence, in which case use 3e-3 or 1e-3 and say so in the report.
2. **translation_scale.** Left at 1 (milestone 3). If the overfit curve shows translation error lagging rotation, try 10 and report. It's a config flag, so no code change.
3. **AMP default off** for the first debug job (correctness first). The RTX 8000 has fp16 tensor cores, so a second debug run with `AMP=true` is the natural follow-up.
4. **Debug job memory at N = 32** is extrapolated, not measured. If it OOMs on the 48 GB RTX 8000, rerun with `N=16`.
