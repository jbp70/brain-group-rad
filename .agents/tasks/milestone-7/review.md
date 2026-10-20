# Milestone 7 review: three-task training at level 4 (partial warm start, loss-balance knobs, m7 SLURM script)

The diff lets a deform-enabled model warm-start from a linear-only checkpoint. It adds a per-task loss weight next to `--loss-scale-mm2`, logs per-task encoder gradient norms (raw and post-clip), and wires up a three-task level-4 recipe in `slurm/m7_three_task.sh` that runs `check_mixed_run.py` afterwards. The partial warm start builds a throwaway model with the source config, restores it strictly, and copies the encoder, linear decoder and centering after count / shape / local-name checks. The coverage guard makes sure no model part is skipped silently. The loss-balance recommendation (keep S = 1, no task weights) is backed by a measured probe grid in `grad_balance.json`. Linear-only behaviour is pinned by an exact-equality golden test written on the milestone-6 code.

Watch for: the checker's `enc_grad_ratio` INFO line takes medians over the whole run, so the zero-SVF-head start skews it in a 200-step smoke (likely). One row of the report's balance table (`warm3`, steps > 50) can't be reproduced from `grad_balance.json` (confirmed). `tests/fixtures/` is untracked and must be committed with the code (confirmed). An unrelated `.gitignore` edit is in the working tree (confirmed).

**Verdict**: APPROVED

## High-level view

The warm start goes through a temporary source model and only three fields (`deform_decoder`, `decoder_widths`, `integration_steps`) may differ. That reuses the existing strict restore and covers BN moving statistics and LayerNorm (all in `part.weights`), and it works in both directions. Any other config difference still raises `ValueError` naming the fields. Identical configs, or a source with no config.json, keep the old strict whole-model restore. The optimizer is never read: main.py still builds it after the warm start, and the tests assert `iterations == 0` and all-zero slots.

The loss balance has two parts. The task weight multiplies the objective only when w != 1, decided at trace time. The encoder-norm diagnostics are read-only ops on gradients that are already computed. So the applied update in the default graph is unchanged, and `TestLinearOnlyUnchanged` asserts that exactly: 6 losses plus the float64 sum and max |v| of every variable, tolerance 0. Two reruns on the unmodified code (P0) showed zero spread first. The probe shows that the global clip of 12, not the loss units, decides what Adam sees on the shared encoder. Raw mm² encoder gradients are about 2000x the Dice ones, but after clipping the linear/deform ratio is 1.25-1.66 for every S in 1..1000. So S = 1 is a measured choice, not a default kept by inertia.

The SLURM script matches the plan's D8 recipe and the cluster rules: `-A qtim`, `rtx8000`, 1 GPU, 3 CPUs, 64 GB, 12 h, no `--val-group-size`, and `entre` appears only in the "never" comment. The report has the smoke and long sbatch lines with time estimates. The deform step time at N 16 is extrapolated, not measured, and the smoke run is the place to measure it.

<details>
<summary>Issues (4)</summary>

1. **Ratio INFO line biased by the zero-init SVF head** — `_check_enc_grad` takes medians over every window, so in a 200-step smoke the early near-zero deform `_eff` inflates the linear/deform ratio. Add a `--enc-grad-min-step` (or tail-fraction) option to the checker, or make P2 explicitly read the jsonl windows after step ~100 (the report already says this in prose).
2. **`warm3` row not reproducible from `grad_balance.json`** — the report's ratio for warm3 is 1.27 / 1.72 "steps > 50", but the JSON stores warm3 with `min_step 0` and ratio 2.97 / 3.95. Either add the step-filtered entry to the JSON or label the table row as computed separately.
3. **Golden fixture untracked** — `tests/fixtures/m7_linear_golden.json` is a new untracked directory. The suite fails on the cluster clone if it is left out of the commit, so stage it explicitly.
4. **Unrelated `.gitignore` change in the tree** — the `.kiro/skills/` and `.agents/tmp/` ignore lines predate this milestone (the coder says so). The orchestrator should decide whether to commit them separately rather than bundling them into the milestone-7 commit.

</details>

<details>
<summary>Details</summary>

### Partial warm start through a temporary source model

`warm_start_model` first computes the config diff (seed excluded) and raises on any field outside `DECODER_ONLY_FIELDS`. The tests check `translation_scale` and `filters` and look for the field name in the message. With no diff it does the old strict restore. Otherwise it builds `GroupRegLevelModel(model.level, src_cfg).build_model(2)` and restores it with `assert_existing_objects_matched`, so a truncated or wrong checkpoint fails before any copy. Every pair is checked before the first `assign`, so a mismatch leaves the target untouched.

Name matching relies on `_local_name` stripping the top-level scope. A second model built in the same process gets a uniquified top-level name, so that is exactly the case being exercised. `test_warm_start_linear_into_deform` builds the target first and the source second, the same order as main.py, so the case is covered (confirmed). The test also proves the BN check is meaningful: it asserts that a source encoder `moving_mean` / `moving_variance` has moved off 0/1 before comparing.

```
target (deform cfg)          src (linear cfg, strict restore)
  encoder        <-- copy --   encoder         (trainable + BN stats)
  linear_decoder <-- copy --   linear_decoder  (incl. LayerNorm)
  centering      <-- copy --   centering
  deform_decoder   fresh       (none)
coverage guard: model.weights == restored ∪ deform_decoder.weights, else RuntimeError
```

For the real m5 source, the old config.json lacks the milestone-6 fields and `ModelConfig.from_dict` fills in defaults, so the expected diff is `{deform_decoder}`. That isn't checked against the actual m5 file locally. The smoke run's `init weights (partial)` line and the checker's `init_weights` row will confirm it.

### Encoder gradient diagnostics and the measured balance

`_enc_grad_diagnostics` runs on gradients that are already filtered (static None filter) and already AMP-unscaled. `raw = enc_norm / (w / S)` for linear and `enc_norm / w` for deform. `eff = enc_norm · min(1, global_clipnorm / gnorm)`. This matches how `make_optimizer` clips: `global_clipnorm` on the inner optimizer, read through `O.inner_optimizer` so it works under the LSO too. Non-finite steps aren't accumulated, so the NaN diagnostics of skipped steps don't pollute the window means. Tests cover the 4x / 3x weight relations for linear and deform, the clipped factor with clipnorm 1e-3, and per-task keys in a three-task fit.

The probe numbers in `grad_balance.json` agree with the report table for the S grid and for the fresh-decoder and scratch rows. Rigid/affine eff stays at 0.239 / 0.202 against deform 0.144 at S = 1, 10 and 100. Affine drops to 0.179 at S = 1000, where clipping falls to 50 %. The `warm3` row is the exception (Issue 2). The probe is synthetic, narrow and N 4 / 2, which the report states, and P2 has a concrete rescaling rule for both the clipped and the unclipped case. Both formulas move in the right direction for ratios above 10 and below 0.1.

The fresh-decoder probe matters for reading the cluster smoke. Deform encoder grad is 3.8e-10 until the zero-init SVF head kernel grows, so ratios from the first ~100 steps are around 1e9. `_check_enc_grad` takes the median over all logged windows. With `--log-every 25` and 200 steps, that is 8 windows per task, and with deform at ratio 2 several of the deform windows can fall in the dead-head regime. The INFO line can therefore show a ratio far outside [0.1, 10] even when the balance after warm-up is fine (likely). It is never a failure, but it is the number P2 tells the reader to look at first (Issue 1).

### Linear-only invariance

The default graph has no extra ops on the update path: `w != 1.0` and `loss_scale != 1.0` are Python checks at trace time, and the diagnostics only read `gs`. The golden test asserts exact equality after the change, and the full-suite run reproduced max |diff| 0. It covers one narrow config, 6 steps, rigid + affine and the default optimizer path, but not AMP. The AMP path changed only by the read-only diagnostic, so that gap is low-risk. Linear-only metrics.jsonl gains `train_<task>/diag_enc_grad_norm_{raw,eff}` and `train_<task>/grad_norm` keys. That changes the log content, not the training.

### SLURM script and cluster commands

The header and env defaults match D8. `INIT` is checked to be a directory (exit 2 otherwise). `TASK_LOSS_WEIGHTS` is passed only when non-empty, and the checker runs inside the container with `--expect-init-weights partial --expect-enc-grad true`. The exit code is main.py's if it failed, else the checker's. `bash -n` passes, both locally and in stage 1 of the suite. The `.sh` files are LF through `.gitattributes`. The report's sbatch lines (`--time=00:45:00` smoke, `--time=12:00:00` long, `--mem=64G`) match the script's env names. They state that Jay's approval and a `squeue -u jn85` check come first.

### Verification evidence

`test-log.txt` ends with `Ran 295 tests in 6127.028s` / `OK (skipped=4)`: the 284 baseline plus 11 new tests, with the same 4 DALI/dllogger skips. `py38_check.sh` ran py_compile and `ast.parse(feature_version=(3, 8))` on all changed .py files, and stage 1 of the suite repeated that on 51 files. The new code uses `typing.Dict`, `math.isfinite`, `tuple(shape)` and no PEP 585/604 runtime annotations. The local end-to-end smoke passed all 18 `check_mixed_run` lines, including both best files and `init_weights` mode partial. No spot-check re-run was needed.

</details>

<details>
<summary>File map</summary>

- `runtime/checkpoint.py`: `DECODER_ONLY_FIELDS`, `SHARED_PARTS`, `warm_start_model` (full/partial + info dict), `load_model_weights` wrapper.
- `main.py`: uses `warm_start_model`, logs an `init_weights` event, prints the stdout summary, docstring example.
- `runtime/args.py`: `--task-loss-weights` + parser, `--init-weights` help.
- `runtime/run.py`: `TrainConfig.task_loss_weights`, objective weighting, `_enc_grad_diagnostics`, per-task grad-norm logging.
- `tests/gpu/check_mixed_run.py`: `--expect-init-weights`, `--expect-enc-grad`, ratio INFO line.
- `slurm/m7_three_task.sh`: new three-task level-4 job (qtim / rtx8000).
- `tests/run_tests_wsl.sh`: `bash -n` on the new script.
- `tests/test_{trainer,runtime_infra,main,check_mixed_run}.py`, `tests/fixtures/m7_linear_golden.json`: golden test, warm-start, weights, diagnostics, group sizes, end-to-end smoke.
- `.agents/tasks/network-design/design.md`: v3.8.
- `.gitignore`: unrelated ignore lines.

Full diff: `git diff HEAD` plus the untracked `slurm/m7_three_task.sh`, `tests/fixtures/` and `.agents/tasks/milestone-7/`.

</details>
