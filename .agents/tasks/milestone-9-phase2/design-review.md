# Design review: milestone 9 phase 2 (`design-m9-phase2.md` v3)

Reviewer: design review step, wf_ea0e306fd89bcd92, iteration 3. Code read at the working tree: `models/cascade.py`, `tests/gpu/frozen_chain_standin.py`, `runtime/checkpoint.py` (RunCheckpoint, `_strict_restore`, `resolve_checkpoint`, `load_cascade_source`, `check_cascade_config`, `weight_fingerprint`, `cascade_init_record`), `runtime/run.py` (Trainer init / cascade branches / postprocess construction), `runtime/args.py` (`parse_args` validation, `train_config_from_args`), `data_loading/postprocess.py` (`num_pairs`, `sample_pairs`, both postprocessors), `models/memory_model.py` (`num_pairs`, `Workload.FIELDS_`), `main.py` (cascade fresh / resume flow, overfit data, helpers), `slurm/m8_level2_cascade.sh`, `tests/gpu/check_mixed_run.py` (`check()` signature, `_check_cascade`), `tests/test_main.py` (`TestCascadeMain`, overfit resume pattern).

Verdict: **APPROVED** (0 HIGH, 0 MEDIUM, 3 NIT).

All iteration-2 findings are resolved as stated. T-E1 now uses `--overfit-max-steps`, `--val-png true` for run 3 and its resume, `rc in (0, 1)`, and an in-process `check()` call with new keyword arguments. T-G1 compares the default graph with the `"n"` graph through the `min_pairs` name scope. D6 is reworded as a recipe choice to confirm with Jay. `from_checkpoint` has an explicit exception contract. No new blocking issue was found.

## Findings

### 1. NIT: section 9 says config.json stores `deform_group_size: null` when the flag is unset

Where: section 9, `--expect-min-pairs` ("config.json stores `null` when `--deform-group-size` is unset").

`parse_args` sets `args.deform_group_size = args.group_size` before anything is written (args.py ~427), so config.json always holds an int. The mirror `args.deform_group_size or args.group_size` still gives the right value, so nothing breaks. Fix: reword to "config.json stores the resolved value (= `--group-size` when unset); the `or` fallback is kept for robustness".

### 2. NIT: how the frozen level-2 model is built is not tied to an existing helper

Where: section 2.2 ("Both models are constructed and built inside `_float32_policy()` ... `build_model(2)`").

`models/cascade._build_frozen_model(cfg)` hard-codes `FROZEN_LEVEL` (4). FrozenChain needs the same rules for a level-2 model. Fix: state `_build_frozen_model(cfg, level=FROZEN_LEVEL)` (default keeps `FrozenStage` unchanged). FrozenChain calls it with `level=2` and `level=4`, so both models follow one code path. This is the stand-in's `build_standin` rule.

### 3. NIT: `_load_cascade_source` gains a `level` argument that section 12 does not list

Where: section 4.3 step 1 (`_load_cascade_source(C, path, model_cfg, level)`) vs main.py today (`_load_cascade_source(C, path, model_cfg)`, which calls `C.load_cascade_source(path)`).

The change is implied but missing from the main.py file entry. Fix: add "`_load_cascade_source(C, path, model_cfg, level=2)` passes `level` to `C.load_cascade_source`" to section 12 `main.py`. The default keeps the level-2 call sites and their tests unchanged.

## Verified assumptions

- Composition (a): the stand-in's `linear` computes `compose_linear(out["A_raw"], A4, model2.centering)` on `prewarp_linear(pool(x1), A4, 2 mm)`, with `stage4.linear` pooling level 2 → 4 itself. Its `deform` computes `compose_cascade(U4₂, U4₂⁻, u2, u2⁻)` at level 2, then upsamples, with V = up(v2). This is exactly section 2.2's math. The trainable step's `compose_cascade(U, U_inv, out.u, out.u_inv, lean=...)` and the single trainable `centering` are unchanged trainer code (run.py 649-663, 743-748), so the unrolled inverse u⁻ = compose(exp(−v1), up(compose(exp(−v2), up(φ4⁻¹)))) follows. Residual SVFs are group-mean centred inside each model; the linear chain has one centering per stage, which matches m8's training of level 2.
- Checkpoint reuse (a/b): `RunCheckpoint` puts `model` and the extras (`frozen_stage` in m8) at the top level of one `tf.train.Checkpoint` that serves last / best / best_deform. The `best` manager has `max_to_keep=1`, so `resolve_checkpoint(<run>/ckpt/best)` returns the single best prefix (step 6000 per notes; recorded from the checkpoint's step variable). `tf.train.Checkpoint(model=model2, frozen_stage=model4).restore(...).expect_partial().assert_existing_objects_matched()` is the `_strict_restore` pattern extended by one key. The m8 `cascade.fingerprint` is `FrozenStage.fingerprint` of the restored level-4 model (main.py 280). `weight_fingerprint` iterates `model.weights` in order, so an identically built level-4 model gives an exact match.
- Resume (b): `RunCheckpoint._restore` asserts existing objects. The new extras `frozen_stage_4` / `frozen_stage_2` do not collide with fields. Fresh vs resume is decided once via `peek_last_step` (main.py 227-235), and overfit-mode resume works (`test_real_data_overfit_path`), so the T-E1 resume is feasible.
- Config equality (d): `lean_sampler`, `split_group_conv` and `gradient_checkpointing` are constructor / TrainConfig knobs, not ModelConfig fields. So the level-1 run with memory knobs passes `check_cascade_config` against m8's stored config, provided no width flags are passed (D1). The document-only N = 5 route is consistent with the strict check.
- Dispatch: `Trainer.__init__` checks `CS.PREV_LEVEL.get(self.level) != cascade.level`. With `{2: 4, 1: 2}`, a level-1 trainer accepts the chain (level 2) and still rejects a bare `FrozenStage`. `getattr(cascade, "n_stages", 1)` keeps today's objects on today's path.
- `--min-pairs` (c): today's `num_pairs` / `sample_pairs` are op-for-op the design's `"1"` branch (one `uniform([n(n−1)])`, `tf.maximum(k, 1)`, `tf.minimum(.., total)`, top_k). The `"n"` branch adds no stateful op and draws the same single uniform, so the RNG stream is unchanged. `perm, roll(perm, −1)` from `top_k(keys[:n])` is one N-cycle with i ≠ j for n ≥ 2. `n = tf.shape(images)[0]` is dynamic, and today's pair tensors are already `[None]`, so consumers are unaffected. Both postprocessors call `sample_pairs` and are constructed inside `Trainer` from `TrainConfig`, so one flag covers the linear pair loss, pair Dice and pair LNCC. Validation pairs do not use `sample_pairs`. `memory_model.num_pairs` / `Workload.FIELDS_` exist as the design says.
- `--p-pair` (0, 1] check: no test, script or tool passes another value.
- S matrices: `brain_S_matrix(level, template_dir)` loads `template_mask{'', '_2', '_4'}.npy`, so both `S_rel_diff_vs_level4` and `_vs_level2` are computable on real data.
- Checker: `check()` takes keyword arguments today, so the new `expect_cascade_stages` / `expect_min_pairs` / `memory_budget_mb` / `project_steps` keywords with defaults equal to today's checks are a pure addition. `_check_cascade` hard-codes `source_level == 4`; the design switches it under `--expect-cascade-stages 2`.
- SLURM (e): the header matches the m8 template and cluster rules (`-A qtim`, no `entre`, `--gpus=1`, `-N 1`, `--ntasks-per-node=1`, 3 CPUs, 64G, logs in the project folder). rtx8000 is the default. dgx-a100 is gated on a smoke peak ≤ 36 GiB. The smoke `--time` comes from measured phase-1 step times × 1.5. The long `--time` is re-derived from the smoke via `--project-steps`. The guard, `RECAL_CHECK` and the checker exit-code pattern exist in m8.
- Off by default (f): `n_pairs` only with `"n"`, `val/wall_s` only at level 1, no skip ops at p = 0. `FrozenStage.checkpoint_extra()` = `{"frozen_stage": model}` keeps m8 keys. Level-2 `load_cascade_source` behaviour is unchanged. The only universal change is `args.min_pairs` in config.json, which the design states.

## Unverified / wrong assumptions

- Wrong (NIT 1): `deform_group_size` is stored as null when unset. It is stored resolved.
- Unverified (cluster only, flagged by the design as A1-A5): `m8_level2_gradnorm/ckpt/best` is step 6000 and holds `frozen_stage`. The 4-pair level-1 peak is 35-38 GB and the deform step ~18 s. Validation host peak is ~30-35 GB. These are checked at runtime (`from_checkpoint`, fingerprint, smoke checker INFO lines, `sacct MaxRSS` rule) before the long run.
- Unverified: T-E1 run time (~8-10 min) on the local GTX 1080 at level 1, N 2, tiny widths. The existing level-1 `--cascade true` measurement test shows level-1 tiny models fit locally.
