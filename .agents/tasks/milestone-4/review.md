# Linear-task training loop for level 4: losses, metrics, Trainer, checkpoints, main.py and the SLURM debug template

This change adds a `runtime/` package and a `main.py` entry point for training one pyramid level on the rigid and affine tasks. The pieces are mm² center and pair losses, numpy float64 metrics, a ported SGDW, an LR schedule, a dllogger/JSONL plus TensorBoard logger, last/best checkpoints and TF-side fixed data (an overfit group and a fixed validation set). The `Trainer` uses one shared optimizer and builds one traced `tf.function` per task, with a NaN guard and BN-statistics rollback. `slurm/train_debug.sh` and `tests/gpu/run_overfit_glacier.sh` hold the cluster and glacier commands. Neither was run or submitted. The test log started at 00:19:39, after the last source edit (`runtime/args.py`, 00:19:32), and nothing in the tree is newer. It records 139 tests OK, 1 DALI skip, and a real overfit run (22.3 mm down to a 10-step mean of 0.99 mm in 352 steps).

Watch for: eval-mode (moving-statistics BN) validation stays near identity on the overfit group while training-mode RMS reaches 0.36 mm. The coder found and reported this (confirmed by their diagnostic; the real-data impact is unverified). As it stands, the `best` checkpoint and the `training=False` frozen-level path that milestone 5 needs both depend on eval-mode BN, so Jay needs to pick a fix before those numbers mean anything. The AMP smoke test does not exercise the loss-scale overflow path that its comment and the report say it shows (likely). The dllogger backend that `auto` picks inside the container has never run (likely).

**Verdict**: APPROVED

## High-level view

The loss and metric conventions match design 2 and 6.1. Pairs are predicted as A_j⁻¹A_i against T_jT_i⁻¹ through `transforms.pair_predictions` / `pair_targets`. The center loss is tr(ΔA S ΔAᵀ) against barycentric A*, in mm², with RMS as √mean. The tests check both against the voxel brute force in `np_reference` at float64 and float32. The numpy metrics use the same pair direction over all ordered pairs.

The train step does the work in the right order. Post-processing runs outside the tape, gradients come from the single shared optimizer (built on every trainable variable up front), and None gradients are dropped at trace time. The update is a `tf.cond` on loss, gradient-norm and input finiteness, and a skipped step also rolls back the BN moving statistics. Under AMP the LossScaleOptimizer is always called, so it can lower its scale, while the loss and A stay float32. `rebuild_steps()` exists, and the test pins the trap it guards against: a BN freeze is ignored until the step is re-traced. Trace count is asserted at one per task.

The validation set is fixed. Groups come from a seeded permutation, transforms are seeded and include the rigid nuisance, images are rendered once with `transform_image`, and there are no flips or intensity augmentation. With N = 32 the 39 T1w cases give 2 groups. Checkpointing wraps model, optimizer, step, best value and the postprocess RNGs in one `tf.train.Checkpoint`, saved by a 'last' manager (keep 2) and a 'best' manager (keep 1). Resume continues the step count, and the stateless per-step task draw keeps the task sequence identical. `configure_tf_precision()` runs before any TF GPU work in `main.py`, and every new TF test imports `tests.tf_setup` first.

The SLURM template follows policy: `-A qtim`, `rtx8000`, one GPU on one node and one task, 3 CPUs, 64 GB, 1 h, logs under `GroupwiseReg/logs`, and the standard binds and sif. Runtime writes stay under `GroupwiseReg/{results,scratch,logs}` and `/scratch/$JOB_ID`. Nothing was committed, `reference/` is untouched, and no script in the diff calls ssh or sbatch.

The open problem is behavioral rather than a code defect: eval-mode BN on the dense path. The default LR also changed from the reference 0.01 to 3e-3, as plan decision 6 allows. The SGDW curves at 3e-3 still swing widely (50 mm, then 3.2, then 8.4 mm before converging), so the debug job is the real test of that default.

<details>
<summary>Issues (4)</summary>

1. **Eval-mode BN gap on the dense path** — validation and `best` selection use moving statistics, and on the overfit group those stay around 20 mm while training mode reaches 0.36 mm. Jay should pick one of the options in the report (training-mode/group-statistics validation, lower `bn_momentum`, or `--dense-norm none`) before trusting `best` or freezing level 4 for milestone 5. Also log train-mode val metrics in the debug job. Non-blocking for milestone 4.
2. **AMP smoke test doesn't exercise overflow skips** — Keras' LossScaleOptimizerV3 advances `inner.iterations` on skipped steps too, so `iterations == n_finite + first_applied` can only hold if every step was finite. Fix the comment and the report claim, or add a test that forces an overflow (e.g. a huge initial scale) and asserts the scale halves and BN rolls back.
3. **dllogger backend unexercised** — `--log-backend auto` picks dllogger whenever it imports, and that path has no test. Either default the debug job to `--log-backend jsonl` or add a test for `_DLLoggerBackend` behind an import guard.
4. **Validation seed depends on task order in the ratio** — `seed=args.val_seed + i` uses the task's position in `--task-ratio`, so `rigid:1` and `affine:1,rigid:2` produce different rigid validation sets. Key the seed on the task name (e.g. the `POSTPROCESS_SEED_OFFSET`-style constant) so val metrics stay comparable across runs.

</details>

<details>
<summary>Details</summary>

### Eval-mode BN and what the 'best' checkpoint means

The coder's `diag_eval_mode.py` puts the cause in the three dense-path BatchNorms, which normalize over N = 4 feature vectors. There, moving_var / batch_var comes out at 7278, 193 and 252 after 400 steps, and the means are about 30 batch std apart. The inter-image variance in those layers is tiny. Training-mode BN rescales it to unit variance, but the moving variance still carries part of its initial 1.0 (0.99⁴⁰⁰ ≈ 0.018 of it, which is large next to a batch variance near 1e-5). On real N = 32 groups of different subjects the batch variance should be far larger, so the gap may shrink. That is unverified (likely, not confirmed).

It matters beyond this milestone. `Trainer.validate()` runs `training=False`, `fit()` feeds `val_rigid/pair_rms_mm` into `maybe_save_best`, and design 5.2 runs frozen prior levels with `training=False`. If the gap persists on real data, the debug job's validation curve and its `best` checkpoint won't track training, and the level-2 cascade would see a near-identity level 4. The code does what the plan specified, and each fix changes the design, so the decision belongs to Jay. In the meantime, the debug job's `train_rigid/*` logs are the trustworthy signal.

### AMP path and the smoke test's claim

In the step, the AMP branch gates `apply_gradients` on input finiteness only. That lets the LSO see overflowed gradients and halve its scale. The BN rollback and the skip counter use the full `finite` flag, computed from the unscaled gradient norm. The test's assertion is weaker than its comment. In TF 2.10 Keras, `LossScaleOptimizerV3.apply_gradients` runs `do_not_apply_fn` on overflow, which calls `self._optimizer.iterations.assign_add(1)`. Applied and skipped steps both advance the inner iteration count, so it reaches 21 after the 21 calls either way. `assertEqual(inner.iterations, n_finite + first_applied)` can therefore only pass when all 21 steps were finite, which means this run had no overflow. The comment "the first steps overflow … the LSO skips them" and the report's "as the AMP test shows" are not supported (likely; traced in `keras/mixed_precision/loss_scale_optimizer.py`, not run). Overflow steps would also count toward `nonfinite_patience` (20 consecutive). With an initial scale of 2¹⁵ that leaves room for about 20 halvings, which should be enough, but it has not been tested.

### Logging backend in the container

`RunLogger(backend='auto')` switches to `_DLLoggerBackend` whenever `import dllogger` succeeds. Locally it never does, so every test uses JSONL. The verified contents of the container image don't list dllogger, and if it is present, the first time this code runs will be the SLURM job (likely). The calls look right for the dllogger API (`init(backends=[JSONStreamBackend, StdOutBackend])`, `log(step="PARAMETER", data=...)`). Still, a failure there would only show up after the job leaves the queue. Passing `--log-backend jsonl` in `train_debug.sh` removes the risk at no cost.

### Validation-set determinism across task ratios

`_build_dali_data` builds each task's `ValidationSet` with `seed=args.val_seed + i`, where `i` is the task's index in `args.task_ratio_parsed`. Across runs with a different task order, though, the rigid set changes composition and transforms, so val numbers from a `rigid:1` debug job and a later `affine:1,rigid:2` job aren't on the same groups (confirmed by reading `main.py`).

</details>

<details>
<summary>File map</summary>

- `runtime/losses.py`: center/pair/total linear losses in mm² (new).
- `runtime/metrics.py`: numpy float64 rms/rot/trans errors, all-pairs, cycle, summarize, accumulator (new).
- `runtime/optimizers.py`: SGDW port with identity-based exclusion, factory, LSO wrapping, LR helpers (new).
- `runtime/schedulers.py`: warmup plus cosine/poly/cycle/constant, pure Python (new).
- `runtime/checkpoint.py`: `RunCheckpoint` last/best, config JSON, resume compatibility check (new).
- `runtime/logging.py`: dllogger or JSONL backend, TensorBoard, run-dir layout (new).
- `runtime/fixed_data.py`: phantom, numpy transform sampler, TF render, DALI-format batches, `ValidationSet` (new).
- `runtime/run.py`: `TaskSchedule`, `TrainConfig`, `Trainer` with the step functions, NaN guard, validate, fit (new).
- `runtime/args.py`: TF-free argparse surface and config builders (new).
- `main.py`: entry point with DALI, overfit and dry-run modes (new).
- `slurm/train_debug.sh`: sbatch template, not submitted (new).
- `tests/gpu/run_overfit_glacier.sh`: glacier overfit / DALI dry-run runner, not run (new).
- `tests/test_runtime_*.py`, `tests/test_trainer.py`, `tests/test_main.py`, `tests/runtime_helpers.py`: 45 new tests and helpers (new).
- `tests/tf_setup.py`: optional fixed GPU memory limit via `BGR_TF_GPU_LIMIT_MB`.
- `tests/run_tests_wsl.sh`: extends the py3.8 check, TF-free import, `--help` and `bash -n` checks, local GPU env vars.
- `models/precision.py`: also ignores `ValueError` from `set_memory_growth`.

Full diff: `git diff` plus the untracked files listed by `git status` in the workspace.

</details>
