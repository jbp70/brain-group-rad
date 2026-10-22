# Milestone 8 design review (design v5)

Reviewer: fresh-context design review. Inputs: `design.md` v5 (incl. section 18 responses), `requirements.md`, `.agents/tasks/network-design/design.md` v3.12, and the code (`models/group_net.py`, `models/layers.py`, `models/diffeo.py`, `models/config.py`, `runtime/checkpoint.py`, `runtime/args.py`, `runtime/run.py`, `runtime/curriculum.py`, `main.py`, `tests/gpu/*`). No builds or tests were run.

Verdict: **APPROVED** (0 HIGH, 0 MEDIUM, 4 NIT). The NITs don't block. They're carried as implementation notes.

## Specific checks requested

- BN recalibration is cascade-only. Confirmed in every place it needs to appear: Q9, section 7 (scope paragraph and flags), the section 10 flag table (`parser.error("--bn-recal-batches > 0 needs --cascade-from (...)")`), the section 15 error table, and test 11 (K = 4 without `--cascade-from` at level 4 and at level 2 → error naming `--cascade-from`; K = 0 parses). As defence in depth, `recalibrate_bn` raises `RuntimeError` when `cascade is None` (test 7). This fits `runtime/args.py`, where every cross-flag rule is already a `parser.error` in the post-parse validation block (lines ~357–447). network-design v3.12 (status line and the 5.1 "BN recalibration as specified" paragraph) says the same thing.
- Resume semantics are defined in one place (section 7, "When it runs"):
  - Recalibration runs once, at step 0 of a fresh run (no `ckpt/last`), after the copy and before `fit`.
  - On resume it does not run again. The moving statistics are non-trainable variables of the trainable model, so `RunCheckpoint` (`tf.train.Checkpoint(model=...)`) restores them. Only `stats_frozen` (a Python attribute, `layers.py` line ~307, not checkpointed) is re-applied, from the saved `config.json bn_recalibration.freeze` (section 9 step 5).
  - A run preempted before its first checkpoint restarts as a fresh run.
  - Test 8 asserts a single `bn_recalibration` event and bitwise-restored statistics on a resume that passes K > 0.
- The 6 prior NITs are addressed (section 18, v4 → v5):
  - #2: `train_step` keeps the non-cascade call unchanged. This matches today's `train_step` (run.py line ~696).
  - #3: each plan slot draws exactly one batch, with no replacement draw.
  - #4: `val=False` is defined, the gap table is then `{}`, and the checker accepts it.
  - #5: the cut applies to the record's `step` field (window end), and test 12 checks it.
  - #6: `--val-pass true` measurement and the `--val-pair-chunk` fallback (see NIT 1 for one remaining gap).
  - #7: the cascade `_val_fn` returns a dict and the non-cascade `_val_fn` is unchanged. This matches today's `_val_fn` / `_val_errors` (run.py lines ~707–755).

## Findings

### 1. NIT: `--val-pair-chunk` must reach every `_deform_errors` call, not just the composed pass

Where: sections 10 and 11 (`TrainConfig.val_pair_chunk` → `_deform_errors(chunk=C)`), section 17.

Problem: `_deform_errors(..., chunk=16, ...)` (run.py line ~805) is called from three places: `_val_deform_errors` (the composed eval and trainmode passes), `_identity_deform_metrics` (run.py line ~870, `self._deform_errors(grp["labels"], z, z, z, z, seed=g, images=...)`), and the new `_frozen_baseline_metrics`. The identity and frozen baselines are computed at the FIRST validation, which is `bnrecal_pre` at step 0. If only the composed call gets `chunk=cfg.val_pair_chunk`, lowering the chunk can't stop an OOM there.

Fix: change the default instead of the call sites: `def _deform_errors(self, labels, u, u_inv, v, v_raw, seed, chunk=None, images=None): chunk = int(self.cfg.val_pair_chunk if chunk is None else chunk)`. Because `val_pair_chunk` defaults to 16, level-4 runs are unchanged. The memory script's `--val-pass` keeps passing `chunk=16` explicitly. Add one assertion to test 7 or 8 that the identity and frozen baselines honour `val_pair_chunk=4` (for example, patch `_val_pairs` to count chunk sizes).

### 2. NIT: say which index `b` is in the recalibration's `controls[t].update(b)`, and note that recalibration consumes training-stream state

Where: section 7 step 3.

Problem: "`self.controls[t].update(b)` ... so G is resampled as during steps 0..K−1" doesn't say whether `b` is the global plan index (0..K−1) or the per-task slot index. Recalibration also advances the postprocess flip generators (`postprocess_<t>`, checkpointed) and the DALI sources before step 0. So a K > 0 run sees a different training batch sequence than a K = 0 run with the same seed. That's harmless (a resume restores the generators, and `TaskControl.update` / `ControlWriter.write` are idempotent and versioned, `curriculum.py` lines ~205–257), but it isn't stated.

Fix: write "b = global plan index 0..K−1 (the step the batch would have had); `fit` then starts again at `update(0)`, which is allowed because control values are a pure function of the step. Recalibration consumes K source batches and K postprocess draws, so K > 0 changes the training data order relative to K = 0 (reproducible on resume through the checkpointed generators)."

### 3. NIT: test 6's bitwise equality across two different graphs may be fragile

Where: section 16 test 6 ("a cascade trainer with p = 0 vs one with p = 0.5 whose steps are all fed `skip=False` ... losses and variable sums equal (max |diff| 0)").

Problem: `tf.where(False, I, A_prev)` is the identity. But the two traced graphs differ, and Grappler or cuDNN algorithm selection can in principle reorder floating-point work even under op determinism. A failure there wouldn't mean the feature is wrong.

Fix: keep max |diff| 0 as the target, and run it with `TF_CUDNN_USE_AUTOTUNE=0` plus `_OpDeterminism` (as the goldens do). Pin the fallback in the test docstring: if a bitwise mismatch appears that can't be traced to the skip ops, the tolerance becomes 1e-6 relative. The report then records it. Separately, the graph-op check (no `cascade_skip/*` ops at p = 0) stays as the strict proof.

### 4. NIT: decide resume vs fresh once, before the FrozenStage is built

Where: section 9 ("Resume (`args.resume` and `ckpt/last` exists, i.e. `C.peek_last_step(.../ckpt/last) > 0`)") vs main.py, which today decides with `trainer.checkpoint.latest_last` (line ~244) AFTER building the Trainer.

Problem: the cascade has to choose `FrozenStage.from_checkpoint` (fresh) or `from_config` (resume) BEFORE `Trainer(..., cascade=frozen)` and `RunCheckpoint` exist. So main.py needs a pre-Trainer decision, and it must agree with the later `restore_last()`. The case `--resume false` with an existing `ckpt/last` is also implicit: it's a fresh run that reads the source, recalibrates, overwrites config.json and appends a second `bn_recalibration` event to the append-mode metrics.jsonl.

Fix: in main.py compute `cascade_resume = bool(args.resume) and C.peek_last_step(<run>/ckpt/last) > 0` once, before any model is built. Use it for both the FrozenStage factory and the recalibration gate, and assert `restore_last()` returns True when `cascade_resume` is True. State in section 9 that `--resume false` on an existing run dir is a fresh run (today's semantics), with the duplicate-event consequence.

## Verified assumptions (read in code)

- `runtime/args.py`: all cross-flag validation is `parser.error` after parsing. `--norm {batch,instance,none}`, `--dry-run`, `--init-weights`, `--resume` (str2bool, default True), `--deform-residual-affine` (default False), `--clipnorm` 12 and `--val-group-size` exist. `--level` choices come from `P.LEVELS`.
- `models/layers.py`: `BatchNormWithSeparateAffine` wraps a Keras BN as `.bn` with its own `momentum`. `call` passes `training=False` when `stats_frozen` is set. `set_bn_stats_frozen` walks submodules and returns the count. So freezing after recalibration and re-applying on resume work as the design says, as long as `rebuild_steps()` follows (trace-time attribute).
- `models/group_net.py`: the single `Dropout` sits after `head_norm_act`, before the heads, after every norm. So the design's claim that dropout during the recal forward doesn't affect any BN statistics is correct. The linear path doesn't run the deform decoder and vice versa, so the per-task part lists of section 7 step 4 are right. Decoder `tf.recompute_grad` is only used once the decoder is built, and the docstring confirms double BN updates under a tape (the reason for test 7's `both` check). `build_model(n)` runs rigid plus deform.
- `runtime/run.py`: `train_step(task, batch_tuple)` calls `self._steps[task](*batch_tuple)`. `_val_fn` returns `model(...)["A"]` (tensor). `_val_errors` calls `.numpy()`. `_val_deform_fn` returns `{u, u_inv, v, v_raw}`. Train-mode validation backs up and restores only `model.non_trainable_variables` of the trainable model. `_deform_errors(..., chunk=16, images=None)` uses eager pair chunks. The identity cache is keyed by (name, len(groups)). `validate()` emits `val_<task>_trainmode/{center,pair}_rms_mm` and `val_deform_trainmode/{template,pair}_dice`. `fit` uses `next(self.sources[task])` and `controls.update(s)`.
- `runtime/checkpoint.py`: `RunCheckpoint(extra=...)` rejects key collisions, and `_restore` uses `assert_existing_objects_matched`. So a built `FrozenStage.from_config` model restores strictly. Also verified: `STEP_KEY`, `peek_last_step`, `resolve_checkpoint` (dir or `<prefix>.index`), `source_config_path`, `DECODER_ONLY_FIELDS`, `SHARED_PARTS`, and `_strict_restore` (`expect_partial` plus matched).
- `main.py`: `_init_weights_lines` reads `info["ignored_source_parts"]` unconditionally (line 398). config.json is written once after the init/resume branch (line ~247), then `log_params`, then the `init_weights` event. The `--init-weights` plus existing `ckpt/last` SystemExit doesn't interact with `--cascade-from` (mutually exclusive). Overfit sources are `fd.FixedGroupSource` iterators, so `next(sources[t])` works in recalibration and in the e2e plan.
- `runtime/curriculum.py`: `TaskControl.update(step)` is a pure function of the step, and `ControlWriter.write` is versioned and skips unchanged values. Calling it during recalibration and then from step 0 in `fit` is safe (NIT 2 only asks to document it).
- `models/diffeo.py`: `upsample_displacement`, `compose`, `compose_cascade`, `pairwise_displacement`, `center_velocities`, `exp_and_inverse` and `to_mm` exist as the design uses them.
- `tests/gpu/` holds `check_mixed_run.py`, `measure_deform_memory.py` and `verify_dali_pipeline.py`. The new `measure_level_memory.py` doesn't exist yet, as expected.
- network-design `design.md` is at v3.12 and agrees with milestone-8 v5 on: cascade-only recalibration, once-only and not on resume, freeze from saved config, `--val-pass` / `--val-pair-chunk`, provenance schema.
- Requirements coverage: FR1–FR14 and NF1–NF4 each map to a design section and a test. The design states the documented relaxation of AC10.5 (fail-fast guard skipped on resume, because resume doesn't read the source). AC12.1 is recorded as "(B) unless Jay chooses (A)", with the open decision carried into the report.

## Unverified or possibly wrong assumptions

- Keras 2.10 BN re-reads `momentum` on every eager call (fused `exponential_avg_factor = 1 − momentum`, non-fused `_assign_moving_average`). This is plausible, and test 7's exact-average check is mandatory, but it's unverified until that test runs.
- `tf.recompute_grad` runs the wrapped function once without a tape (eager). Test 7 with `both` asserts it.
- The two-level inverse tolerances (0.05 / 0.01 level-2 voxels) are provisional. Test 4 prints the measured values.
- All level-2 memory, step-time, validation-memory and `--time` numbers are provisional until job (a) and the smoke run.
- The `verify_dali_pipeline` tolerances were calibrated at 4 mm and may need adjusting at 2 mm.
- The re-export with `LABEL_SETS=train18,synthseg` leaving image / train18 bytes identical depends on `_needs_write` per item. I didn't trace it here.
- Test 6's bitwise equality across different graphs (NIT 3).
