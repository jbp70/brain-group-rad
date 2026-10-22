# Milestone 8 implementation plan: level 2 with a frozen level-4 cascade

Source of truth: `design.md` v5 (approved, `design-review.md`: 0 HIGH / 0 MEDIUM / 4 NIT), `requirements.md` (FR1-FR14, C1-C8, NF1-NF4). Section numbers (§) refer to `design.md`; test numbers (T1-T17) refer to §16. This plan sequences the design; it does not re-decide it. Label sampling is (B) deferred (§14); only the exporter change is implemented.

Repo: `d:\Research\Brain_Registration_Projects\brain-group-rad` (no worktree). No commits, pushes or cluster actions (C8). Shell scripts for WSL go in `.agents/tmp/` and are deleted afterwards.

## Shell mechanics (all items)

PowerShell 5.1 mangles quotes, so every multi-line WSL command goes into a script file in `.agents/tmp/` and runs with:

```
wsl -d Ubuntu bash -c "tr -d '\r' < /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.agents/tmp/<script>.sh | bash -s"
```

Targeted unit tests (`.agents/tmp/ut.sh`, edit the module list):

```bash
source ~/miniconda3/etc/profile.d/conda.sh
set +u
conda activate tf_gpu
cd /mnt/d/Research/Brain_Registration_Projects/brain-group-rad
export PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 BGR_TF_GPU_LIMIT_MB=6000 TF_CUDNN_USE_AUTOTUNE=0
python -m unittest -v tests.test_cascade 2>&1 | grep -v "cuBLAS factory"
```

`BGR_TF_GPU_LIMIT_MB` / `--gpu-memory-limit-mb` never above 6000. Python 3.8 check (`.agents/tmp/py38.sh`):

```bash
source ~/miniconda3/etc/profile.d/conda.sh
set +u
conda activate bgr_preproc38
cd /mnt/d/Research/Brain_Registration_Projects/brain-group-rad
F="models/cascade.py runtime/args.py runtime/run.py runtime/checkpoint.py main.py tests/gpu/check_mixed_run.py tests/gpu/verify_dali_pipeline.py tests/gpu/measure_level_memory.py tests/runtime_helpers.py tests/test_cascade.py tests/test_slurm_scripts.py tests/test_measure_memory.py tests/test_verify_dali_level.py tests/test_runtime_infra.py tests/test_check_mixed_run.py tests/test_main.py preprocessing/export.py preprocessing/cli.py preprocessing/tests/test_export.py"
PYTHONDONTWRITEBYTECODE=1 python -m py_compile $F && echo py_compile OK
python -c "import ast,sys; [ast.parse(open(f).read(), f, feature_version=(3, 8)) for f in sys.argv[1:]]; print('ast 3.8 OK', len(sys.argv)-1)" $F
for s in slurm/m8_memory.sh slurm/m8_level2_cascade.sh; do bash -n "$s" && echo "bash -n $s OK"; done
```

(Only list files that exist at that point.) Goldens (FR13, after every `runtime/run.py` change): `python -m unittest -v tests.test_trainer.TestLinearOnlyUnchanged tests.test_trainer.TestDeformUnchanged` → both OK, stderr `[golden] max |diff| 0` and `[golden deform] max |diff| 0`.

## Design-review NITs carried (must be implemented)

- NIT 1 (item 5): `_deform_errors(self, labels, u, u_inv, v, v_raw, seed, chunk=None, images=None)` with `chunk = int(self.cfg.val_pair_chunk if chunk is None else chunk)`; call sites (`_val_deform_errors`, `_identity_deform_metrics`, `_frozen_baseline_metrics`) do not pass `chunk`; the memory script passes `chunk=16` explicitly. T7 or T8 asserts the identity and frozen baselines honour `val_pair_chunk=4` (patch/wrap `_val_pairs` or count chunk sizes).
- NIT 2 (item 6): in `recalibrate_bn`, `b` = the global plan index 0..K-1 (`self.controls[t].update(b)` only if `t in self.controls`); docstring states that `fit` restarts at `update(0)` (control values are pure functions of the step) and that recalibration consumes K source batches and K postprocess draws, so K > 0 changes the data order vs K = 0 (reproducible on resume via the checkpointed generators). Mirror the sentence in `design.md` §7 step 3.
- NIT 3 (item 4, T6): run the p=0 vs p=0.5-with-skip=False identity test under `_OpDeterminism` (import from `tests.test_trainer`) with `TF_CUDNN_USE_AUTOTUNE=0`; target max |diff| 0; docstring pins the fallback: if a bitwise mismatch cannot be traced to the skip ops, tolerance 1e-6 relative and the report records it. The graph-op check (no op name starting `cascade_skip` at p = 0) stays strict.
- NIT 4 (item 8): main.py computes `cascade_resume = bool(args.resume) and C.peek_last_step(<run>/ckpt/last) > 0` ONCE before any model is built; it selects `FrozenStage.from_config` (resume) vs `from_checkpoint` (fresh) and gates recalibration; after `restore_last()` assert it returned True when `cascade_resume`. `--resume false` on an existing run dir = fresh run (today's semantics; overwrites config.json, appends a second `bn_recalibration` event); add this sentence to `design.md` §9.

## Plan

- [ ] 1. Tiny test config and the cascade module.
      Add `TINY = dict(filters=(8, 16, 16, 16, 16, 16), reductions=(16, 8), dense=(64, 32), decoder_widths=(8, 8, 8, 8, 8), dropout=0.0)`, `TINY_ARGS` (the matching `--filters/--reductions/--dense/--decoder-widths` CLI list) and `tiny_config(**kw)` to `tests/runtime_helpers.py`. Create `models/cascade.py` exactly per §3: `PREV_LEVEL = {2: 4}`, `pool_to_coarser` (`tf.nn.avg_pool3d`, float32), `_float32_policy()` context manager (save/set `'float32'`/restore in `finally`), `FrozenStage` with `from_checkpoint(prefix, cfg)` (build `GroupRegLevelModel(4, cfg, gradient_checkpointing='none')` + `build_model(2)` inside `_float32_policy`, then `checkpoint._strict_restore`), `from_model(model)` (fresh float32 model from `model.config`, `copy_weights_from`), `from_config(cfg)`, `.model`, `.level = 4`, `.fingerprint` (property → `checkpoint.weight_fingerprint`), `linear(images_fine, task)` and `deform(images_fine)` (always `training=False`, `tf.stop_gradient`, deform fields upsampled with `diffeo.upsample_displacement`), and the free functions `prewarp_linear`, `compose_linear`, `prewarp_deform`, `skip_linear`, `skip_fields` (both build their `tf.where` under `tf.name_scope("cascade_skip")`). Import `runtime.checkpoint` lazily inside the factory/fingerprint to avoid a cycle if needed.
      Files: `tests/runtime_helpers.py`, `models/cascade.py`, `tests/test_cascade.py` (new; `import tests.tf_setup` first).
      Tests added now: T1(c) units (`to_mm(upsample_displacement(c)) == to_mm(c, 4)`, same rigid params → same mm matrix at both levels), T2 `TestRenderConsistency` (AC7.2, NCC > 0.98), T3 `TestLinearCascade` (AC3.1-3.4: numpy Karcher reference with `scipy.linalg.logm/expm`, 1e-5 float64 with `tfm.GroupCentering(iters=cfg.centering_iters, dtype="float64")`, 1e-4 float32, barycenter = I, R = I case, pre-warp NCC > 0.99, pair predictions), T4 `TestDeformCascade` (AC4.1-4.4: zero residual 1e-6; two-level round trip with `fd.random_svf` max 2 vox, T=7, 3-voxel margin, max ≤ 0.05 / mean ≤ 0.01, print measured values; residual group mean ≤ 1e-5; `losses.deform_loss` on `out_c` = manual composed computation), T9 factory part (`compute_dtype == "float32"` for all three factories under a `mixed_float16` global policy, policy restored to `mixed_float16` after each).
      Verify: `ut.sh` with `tests.test_cascade` → all OK (print measured round-trip errors); `tests.test_group_net tests.test_diffeo` still OK.

- [ ] 2. Checkpoint helpers and cross-level warm start (FR1, §2.2-2.3).
      In `runtime/checkpoint.py` add `load_cascade_source(path)` (resolve via `resolve_checkpoint`; `source_config_path` must exist; source `args.level` must be 4; `info` = checkpoint abs prefix, source_config abs path, source_level 4, `source_step = int(tf.train.load_variable(prefix, STEP_KEY))`; ValueError otherwise), `check_cascade_config(model_config, src_cfg)` (ValueError naming every differing field, seed excepted), `weight_fingerprint(model)` (`n_vars`, float64 `abs_sum`, `sq_sum` over `model.weights`), a helper `cascade_init_record(model, info)` building the §2.2 cascade dict verbatim (`via: "cascade"`, `cross_level [4, 2]`, parts from `len(part.weights)` of the level-2 model, `deform_decoder` only if present). Extend `warm_start_model`: source level from the source config.json `args.level` (absent → `model.level`); same level → existing code path untouched; different level → build `src = GroupRegLevelModel(src_level, src_cfg).build_model(2)`, `_strict_restore(src, prefix)`, identical configs → `model.copy_weights_from(src)` and the full record; decoder-only diff → the existing `SHARED_PARTS` loop on `src`; both add exactly `source_level`, `source_step`, `cross_level` (§2.3 dicts, `n_restored_vars = len(pairs)` in partial). Do not extend `DECODER_ONLY_FIELDS`.
      Files: `runtime/checkpoint.py`, `tests/test_cascade.py` (T1(a)(b)(d)), `tests/test_runtime_infra.py` (T10 `TestCrossLevelWarmStart`).
      Acceptance: AC1.1-AC1.3, AC1.5 (T1(d) prints the S rel. diff < 0.05), AC13.3 (level-4→level-4 record key set = today's nine keys).
      Verify: `ut.sh` with `tests.test_cascade tests.test_runtime_infra` → OK; `tests.test_main.TestResumeControlsAndWarmStart` still OK.

- [ ] 3. CLI flags and config plumbing (§10).
      `runtime/args.py`: `--cascade-from` (str, None), `--cascade-skip-prob` (float 0.0), `--bn-recal-batches` (int 0), `--bn-recal-freeze` (str2bool True), `--val-pair-chunk` (int 16). Validation in the post-parse block, all `parser.error`: cascade needs `--level 2`, excludes `--init-weights`, needs `--deform-residual-affine false`; skip finite in [0, 1), > 0 needs cascade; K ≥ 0, K > 0 needs `--cascade-from` with the exact message `--bn-recal-batches > 0 needs --cascade-from (BN recalibration is defined for the level-transfer cascade only)`, K ≥ len(task ratio), `--norm batch`, not `--dry-run`; `--val-pair-chunk` ≥ 1. `runtime/run.py` `TrainConfig`: `cascade_skip_prob: float = 0.0`, `val_pair_chunk: int = 16` (dataclass fields only, no behaviour yet); `train_config_from_args` passes both. Help texts state the semantics.
      Files: `runtime/args.py`, `runtime/run.py` (TrainConfig only), `tests/test_runtime_infra.py` (T11, T16).
      Verify: `ut.sh` with `tests.test_runtime_infra` → OK; `py38.sh` on `runtime/args.py`; `python main.py --help` works in `bgr_preproc38` (no TF import); goldens OK.

- [ ] 4. Trainer cascade train steps and skip (FR2-FR5, §4-§6). Depends on 1, 3.
      `Trainer.__init__(..., cascade=None)` stores `self.cascade` (never added to the optimizer). `_make_train_step` / `_make_deform_step`: a cascade branch chosen at trace time exactly per the §4 / §5 pseudo-code (frozen call + pre-warp before the tape, `compose_linear(out["A_raw"], A_prev, model.centering)` / `diffeo.compose_cascade` inside the tape, losses on composed results, `deform_diagnostics(out_c)` plus `diag_u_res_max_vox`), with the extra `skip` input (`tf.TensorSpec([], tf.bool)`) appended to the input signature only in the cascade branch; skip ops only when `cfg.cascade_skip_prob > 0`. Non-cascade signatures and bodies untouched. `train_step(task, batch_tuple, skip=False)` per §6. `fit`: stateless draw `random.Random("cascade-skip-%d-%d" % (seed, s)).random() < p` only with cascade and p > 0; `train/cascade_skip_frac` window mean only when p > 0; with a cascade also `train_<task>/data_ms` and `train_<task>/compute_ms` window means. Add `DEFORM_CASCADE_DIAG_KEYS`/logging of `diag_u_res_max_vox` only in the cascade branch.
      Files: `runtime/run.py`, `tests/test_cascade.py` (T5 `TestFrozenStageTraining`, T6 `TestSkip` per NIT 3, T9 step part `TestCascadeAMP`).
      Acceptance: AC2.1, AC2.4, AC3.6, AC4.6, AC5.2.
      Verify: `ut.sh` with `tests.test_cascade` → OK; goldens max |diff| 0; `tests.test_trainer` → OK.

- [ ] 5. Cascade validation and frozen-stage baselines (FR7-FR8, §8). Depends on 4.
      `_val_fn` / `_val_deform_fn` cascade branches (same keys, traced once; frozen recomputed inside the traced function; eval returns `{"A", "A_prev"}` / usual four keys + `U, U_inv, V`; train-mode returns `{"A"}` / the usual four; non-cascade functions unchanged). `_val_errors` / `_val_deform_errors` read per §8 and collect `frozen_outs` in eval mode with a cascade. `_frozen_baseline_metrics(name, vs, groups, frozen_outs)` cached by (name, len(groups)): linear `M.group_errors(A4, ...)` → `M.summarize(..., "val_%s_frozen" % name)` (affine factor errors included), deform `_deform_errors(labels, U, U_inv, V, V, ...)` → `val_deform_frozen/*`. `val_deform/composed_mean_disp_mm` (RMS over voxels of |mean_i u_i| in mm) with a cascade only. NIT 1 (`chunk=None` default from `cfg.val_pair_chunk`). Validation never skips.
      Files: `runtime/run.py`, `tests/test_cascade.py` (validation subtests: baseline keys present and finite for rigid/affine/rigid_g-style sets and deform, cache hit on second `validate()`, NIT 1 chunk assertion with `val_pair_chunk=4`, `validate()` identical for p=0 and p=0.9 from T6).
      Acceptance: AC7.3, AC7.4, AC8.1, AC8.4.
      Verify: `ut.sh` with `tests.test_cascade tests.test_trainer.TestDeformValidation tests.test_trainer.TestResumeAndValidation` → OK; goldens max |diff| 0.

- [ ] 6. BN recalibration (FR6, §7). Depends on 5.
      `Trainer.recalibrate_bn(num_batches, freeze, val=True)` exactly per §7 steps 1-7: `RuntimeError` when `self.cascade is None`; pre validation (PNGs off) logged at step 0 as `bnrecal_pre/...`; largest-remainder task plan interleaved round-robin; per slot one draw (NIT 2 index `b`), non-finite batch consumes the slot (`skipped_nonfinite`), 0 used → `RuntimeError`, `StopIteration` → `RuntimeError("bn recalibration: source <task> exhausted")`; eager forward pool → frozen → pre-warp → `model(x_in, task, training=True)` without tape; per-layer `m.bn.momentum = c/(c+1)` over `BatchNormWithSeparateAffine` layers of the parts the task runs, originals restored in `finally`; post validation `bnrecal_post/...`; freeze → `model.set_bn_stats_frozen(True)` + `rebuild_steps()`; returns and logs the `bn_recalibration` event (batches, skipped_nonfinite, n_layers_updated, layers_without_updates must be empty else RuntimeError, elapsed_s, freeze, gap table `gap_pre/<set>/<metric>`, `gap_post/...`; `{}` when `val=False`).
      Files: `runtime/run.py`, `tests/test_cascade.py` (T7 `TestBNRecal`, every subtest listed in §16 T7 incl. the mandatory exact-average test, `gradient_checkpointing='both'` update-count test, differing-momentum layer, non-finite slot, `val=True` subtest).
      Acceptance: AC6.1-AC6.4, AC6.6.
      Verify: `ut.sh` with `tests.test_cascade.TestBNRecal` → OK; goldens max |diff| 0.

- [ ] 7. Local "cascade beats frozen" evidence (T16b, AC3.5 / AC4.5 local). Depends on 6.
      `TestCascadeBeatsFrozen` in `tests/test_cascade.py` exactly per §16 T16b (level-4 TINY model trained 100 rigid + 100 deform steps on the pooled same group built with `fd.to_dali_batch(..., 4)` / `fd.to_deform_batch(..., level=4)` and `preprocessing.export.block_mode` labels, wrapped with `FrozenStage.from_model`, level-2 cascade trainer 100 steps per task). Print margins (they go into the report).
      Files: `tests/test_cascade.py`.
      Verify: `ut.sh` with `tests.test_cascade.TestCascadeBeatsFrozen` → OK (≈1-2 min); record printed composed vs frozen center RMS and template Dice.

- [ ] 8. main.py cascade wiring, provenance and resume (§2.1, §9, NIT 4). Depends on 2, 3, 6.
      In `main.py`: `cascade_resume` decided once before model build (NIT 4). Fresh: `C.load_cascade_source` + `C.check_cascade_config` (ValueError → `SystemExit("main.py: --cascade-from ...: <msg>")`), `FrozenStage.from_checkpoint`, `model.copy_weights_from(frozen.model)`, `init_info = C.cascade_init_record(...)`; `cascade` block (§9: source_checkpoint, source_config, source_level, source_step, fingerprint, `S_rel_diff_vs_level4` (DALI path: `postprocess.brain_S_matrix(4, template_dir)` vs level-2 S; overfit path: `null`), skip_prob). `Trainer(..., cascade=frozen)`; `RunCheckpoint(extra=dict(trainer.checkpoint_variables(), frozen_stage=frozen.model))`. Write order per §9 (config.json, log_params, `init_weights` event, `recalibrate_bn` if K > 0 and fresh, second `write_config` with `bn_recalibration`). Resume steps 1-6 of §9 (config.json `cascade` required, source not read, `from_config`, restore, fingerprint check rel 1e-12 → RuntimeError, re-freeze from saved `bn_recalibration.freeze` + `rebuild_steps()`, provenance blocks carried verbatim). Non-cascade paths byte-identical. Update `design.md` §9 with the NIT 4 sentence.
      Files: `main.py`, `models/cascade.py` (if helpers needed), `tests/test_cascade.py` (T8 `TestCascadeResume`, main-level via `main.main([...])` on overfit-synthetic TINY setup), `tests/test_main.py` (cascade overfit dry-run smoke: `--level 2 --cascade-from <tiny level-4 run>/ckpt/last` exits 0, config.json has `cascade` and `init_weights.via == "cascade"`; `--cascade-from` with a level-2 source → SystemExit naming the level).
      Acceptance: AC1.3, AC2.3, AC6.1 (config/event), AC10.5 (relaxation on resume documented), AC13.3.
      Verify: `ut.sh` with `tests.test_cascade.TestCascadeResume tests.test_main` → OK; goldens OK.

- [ ] 9. Checker extensions (FR11, §13). Independent of 4-8 (stdlib only).
      `tests/gpu/check_mixed_run.py`: options `--expect-cascade`, `--expect-bn-recal`, `--expect-skip`, `--grad-balance-rule`, `--grad-balance-from-step` (default 50, help text: window-end `step` ≥ value; with `--log-every 25` the first window is 26-50); `cascade_gain` INFO lines, data-wait fraction INFO, `g_gap <task>` INFO lines for composed and frozen sets with the §13 keys (rot/trans and affine scale/shear/qr_rot for the G-gap question, AC8.3); `grad_balance KEEP task-grad-norm` / `FALLBACK deform:64` / `UNDECIDED (treat as FALLBACK and inspect the run)` (never fails). Defaults off → level-4 dirs unchanged. Update module docstring.
      Files: `tests/gpu/check_mixed_run.py`, `tests/test_check_mixed_run.py` (T12 incl. the step-cut case and empty gap table).
      Acceptance: AC11.1-AC11.4.
      Verify: `ut.sh` with `tests.test_check_mixed_run` → OK.

- [ ] 10. Level memory script (FR9, §11). Depends on 4, 5.
      New `tests/gpu/measure_level_memory.py` reusing `measure_deform_memory.py`'s structure: `configure_tf_precision()` first, options of §11 (`--level`, `--tasks`, `--n`, `--widths narrow,default,tiny`, `--cascade`, `--gradient-checkpointing`, `--amp`, `--lambda-ncc-template/pair`, `--val-pass`, `--warmup-steps`, `--steps`, `--gpu-memory-limit-mb`, `--out`), `main(argv)` returning 0, synthetic data, OOM → `status oom` and larger N `skipped`, other errors `status error`, JSON merged by config key, per-result fields of §11 incl. `frozen_ms`, `val_peak_mb/val_status/val_ms` (deform `--val-pass` with `chunk=16` explicit), AMP policy reset after each config.
      Files: `tests/gpu/measure_level_memory.py`, `tests/test_measure_memory.py` (T15, GPU-only skip, `--gpu-memory-limit-mb 6000`).
      Acceptance: AC9.1, AC9.2.
      Verify: `ut.sh` with `tests.test_measure_memory` → OK on the GTX 1080 (record the printed tiny-config peaks).

- [ ] 11. verify_dali_pipeline `--level` (AC7.1, §8). Independent.
      `tests/gpu/verify_dali_pipeline.py`: `--level` (int, choices (4, 2), default 4), `_set_level(level)` rebinding `LEVEL`, `SHAPE`, `SPACING` via `global`, called right after `parse_args()`; `_template_brain(args)` loading `"template_mask%s.npy" % P.level_suffix(LEVEL)` replacing both literals; docstring line rewritten (no literal `template_mask_4` / `labels_train18_4` left).
      Files: `tests/gpu/verify_dali_pipeline.py`, `tests/test_verify_dali_level.py` (T16a).
      Verify: `ut.sh` with `tests.test_verify_dali_level` → OK.

- [ ] 12. SLURM recipes (FR10, §12). Depends on 9, 10, 11 (flags they call).
      `slurm/m8_memory.sh` and `slurm/m8_level2_cascade.sh` per §12 (m7 template, `SCRATCH_ROOT` override, variable block assigned once, `N`/`DEFORM_N`/`BN_RECAL_BATCHES` integer checks, fresh-run-only `CASCADE_FROM` guard accepting dir or `<prefix>.index`, `GRAD_BALANCE=gradnorm|deform64` with the C4 decision rule and the scancel/resubmit procedure in the header, the exact sbatch lines (a)-(c) with provisional `--time` and the formula, `-p dgx-a100` override with `peak: unmeasured (fill in from scratch/m8/memory_level2.json)`, checker call with `--expect-cascade true --grad-balance-rule true $RECAL_CHECK`, exit-code rule). No `--val-group-size`, no `entre`. Add both scripts to the `bash -n` loop in `tests/run_tests_wsl.sh`.
      Files: `slurm/m8_memory.sh`, `slurm/m8_level2_cascade.sh`, `tests/test_slurm_scripts.py` (T13 static checks + the four stub-run cases + memory stub run), `tests/run_tests_wsl.sh`.
      Acceptance: AC10.1-AC10.5.
      Verify: `ut.sh` with `tests.test_slurm_scripts` → OK; `bash -n` both scripts in `py38.sh`.

- [ ] 13. Exporter: synthseg at levels 1 and 2 (§14). Independent.
      `preprocessing/export.py`: `LABEL_SET_LEVELS = {"train18": (1, 2, 4), "synthseg": (1, 2)}` replacing `PYRAMID_LABEL_SETS` (line ~117, uses at ~404 and ~483), `ls_levels = [k for k in levels if k in LABEL_SET_LEVELS.get(ls, (1,))]`, ValueError when empty with the existing full-resolution message for `(1,)` sets and `label set %r is exported at levels 1,2 only; ...` for synthseg; `_2` via `block_mode`; docstrings. `preprocessing/cli.py` `--label-sets` help.
      Files: `preprocessing/export.py`, `preprocessing/cli.py`, `preprocessing/tests/test_export.py` (T14; update the line ~119 `labels_synthseg_2` assertion).
      Verify: `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/preprocessing/tests/run_tests_wsl.sh bgr_preproc38` → `ALL CHECKS PASSED (bgr_preproc38)`.

- [ ] 14. Local end-to-end tiny cascade (§16 local e2e plan). Depends on 8, 9.
      One script `.agents/tmp/e2e.sh` (tf_gpu, `TF_CUDNN_USE_AUTOTUNE=0`, results under `.agents/tmp/e2e_results`, `--gpu-memory-limit-mb 6000`, `TINY_ARGS` flags, `--val-png true`): (1) level-4 three-task overfit 150 steps `e2e_l4`; (2) level-2 cascade `--cascade-from .../e2e_l4/ckpt/last --bn-recal-batches 8 --task-grad-norm 12 --lambda-ncc-template 1 --lambda-ncc-pair 1` 150 steps `e2e_l2` (overfit exit code 1 = threshold not reached is acceptable), then `check_mixed_run.py ... --expect-best false --expect-init-weights full --expect-enc-grad true --expect-cascade true --expect-bn-recal true --grad-balance-rule true`; (3) resume with `--overfit-max-steps 160 --resume true`, diff the three config.json blocks, checker with `--steps 160`. Record: trace counts, composed vs frozen train-mode errors, folding, bn gap pre/post, grad_balance verdict, step times.
      Files: none committed (results and script deleted after recording numbers in the FEAT findings / report).
      Verify: checker prints `check_mixed_run: PASS` for steps (2) and (3); fingerprint check passes on resume.

- [ ] 15. Full verification (FR13).
      `py38.sh` over every new/modified file (AC13.4) and `bash -n` of both m8 scripts; preprocessing suite (item 13 command); full suite in the background with a ~2 h timeout: `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/tests/run_tests_wsl.sh` (baseline 327 OK, 4 skips; expect 327 + new, 0 failures, goldens max |diff| 0).
      Verify: unittest exit code 0, `py_compile OK`, `ast feature_version=(3, 8) OK`, every `bash -n ... OK`.

- [ ] 16. Report and docs (FR14).
      Write `.agents/tasks/milestone-8/report.md`: files changed, tests run with counts, local verification numbers (T4 round-trip errors, T7 gap, T16b margins, e2e results, local memory test peaks, NF1 frozen share if measured), memory recommendation "pending cluster job (a)", the exact sbatch lines in cluster order (memory → smoke → long) with provisional times and the T_smoke/T formulas, C4 decision rule, G-gap table template (level-4 LNCC numbers: rigid pair 5.45 vs 5.53, affine pair 6.92 vs 9.45, rotation 4.9→7.1°, shear 4.0→5.3 %) to be filled from the checker's `g_gap` lines, open decision: label sampling (B) unless Jay chooses (A), plus Jay's export commands (§14). Update `design.md` (NIT 2 and NIT 4 sentences; measured tolerances if T4 values differ) and the network-design status line only if a decision changed.
      Files: `.agents/tasks/milestone-8/report.md`, `.agents/tasks/milestone-8/design.md`.
      Verify: report lists every file of §17 and the counts from item 15.

## Requirement → item map

FR1 → 1, 2, 8 · FR2 → 1, 4, 8 · FR3 → 1, 4, 7 · FR4 → 1, 4, 7 · FR5 → 4 · FR6 → 6, 8 · FR7 → 5, 11, 12 · FR8 → 5, 9 · FR9 → 10, 12 · FR10 → 12 · FR11 → 9 · FR12 → 13, 16 (B deferred) · FR13 → every item (goldens), 15 · FR14 → 16 · NF1 → 10 (`frozen_ms`) · NF2 → 1, 4 (float32 grids, T9) · NF3 → 4, 6 · NF4 → 4, 6, 8.

## Assumptions / gaps

- `TINY` reductions (16, 8) are valid (`ModelConfig.validate` only requires non-empty reductions).
- If `copy_weights_from` rejects the TINY level-4 → level-2 pair for any reason, fix it in `models/cascade.py`/`checkpoint.py`, not `group_net.py` (§17 says group_net needs no change); if a change there is unavoidable, record it in the findings and report.
- `tests.test_trainer._OpDeterminism` is importable from the new test module; if import order causes double test discovery, copy the context manager into `tests/test_cascade.py`.
- Two-level inverse tolerances are provisional (§16 T4); if measured values exceed them, revise in `design.md` and the report with the measured numbers, do not silently loosen.
