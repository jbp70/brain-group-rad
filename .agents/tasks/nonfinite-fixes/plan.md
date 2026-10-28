# Implementation Plan: non-finite robustness fixes + opt-in fp16 overflow mitigations

Source of truth: `.agents/tasks/m9-level1-nan/report.md` (sections 1, 2, 5). Work directly in
`d:\Research\Brain_Registration_Projects\brain-group-rad` (no worktree). Do NOT commit, push or touch
the cluster; the orchestrator does git and cluster. Python 3.8 syntax everywhere (container py3.8.10);
`slurm/sweep_level4.py` must also stay Python 3.6 / stdlib only.

## Decisions (fixed; do not re-decide)

D1. Per-task guard. `TrainConfig.nonfinite_patience_task: int = 20` (`--nonfinite-patience-task K`,
0 = off). `fit` keeps `consecutive_task[task]` (+ `streak_first[task]` = step number `done` of the first
non-finite step of the current streak), reset to 0 ONLY by a finite step of the same task. Raise when
`consecutive_task[task] > K` (same `>` semantics as the old guard, so K = 20 allows 20 and raises on
the 21st). The old global `consecutive` counter and `--nonfinite-patience` (default 20) stay exactly
as they are and are checked FIRST (single-task runs keep the old message; there both counters are equal).
Rationale: in a 1:1:2 mixed run the global counter is reset by every finite linear step (job 10518952).

D2. Abort. Before either RuntimeError: `logger.log_event(done, {"event": "nonfinite_abort", "task", "streak",
"first_step", "last_step", "last_ckpt_step"})` and `logger.flush()`. Per-task message:
`"task %s: %d consecutive non-finite steps of this task (first at step %d, last at step %d; finite steps
of other tasks do not reset this count); stopping. ckpt/last was not written during the streak (last
save at step %s); restart from ckpt/last or ckpt/best*, e.g. with a lower --lr, --fp16-act-clip or
--deform-decoder-dtype float32"`. No auto-recovery.

D3. ckpt/last protection. Constant `SAVE_LAST_MAX_STREAK = 2`: `save_last` is NOT called while any task
has `consecutive_task[t] >= 2` (a single non-finite step does not block: it changed no weights, and isolated
AMP overflows must not shift the ckpt cadence). A due save that is blocked sets `pending_save = True`,
logs one `{"event": "ckpt_last_deferred", "streaks": {...}}` per deferral, and is performed at the first
later step where no task is in a streak >= 2 (`saved_at` updated). The final save at the end of `fit`
follows the same rule (blocked -> event `ckpt_last_skipped_final`). For finite runs nothing changes.
Documented caveat: "pre-failure" = no task in a streak >= 2 at save time; with an intermittent onset the
weights just before the streak may already be near the failure; `ckpt/best_deform` is the safer
restart point. ckpt/best* saving is unchanged (out of scope; note it in the docstring).

D4. LSO skip (AMP only, trace-time switch `amp`). Constant `LSO_SKIP_STREAK = 2`. A step of task t is
"LSO-skipped" iff ALL of: AMP on, the previous step of the SAME task was non-finite
(`consecutive_task[t] >= LSO_SKIP_STREAK - 1` before the step), and THIS step's forward loss
`terms["loss"]` is non-finite. Then `apply_gradients` is not called (no scale halving, no
LossScaleOptimizer counter reset, inner `iterations` NOT incremented: `iterations` counts LSO calls).
The extra "forward loss non-finite" condition is deliberate: gradient-only overflows (finite loss, inf
gradients) are the normal AMP loss-scale warm-up, which needs consecutive halvings; they ALWAYS reach the
LSO exactly as today. A forward NaN is scale-independent (the fp16 overflow in report section 2), so
halving cannot help it. Isolated forward-NaN steps (previous same-task step finite) also reach the LSO
as today. Non-AMP graphs are untouched.

D5. Skip flag without retracing: one non-trainable `tf.Variable(False, dtype=tf.bool,
trainable=False, name="lso_skip_<task>")` per task in `self._lso_skip` (dict), created in
`Trainer.__init__` BEFORE `rebuild_steps()` and only when `cfg.amp` (else `{}`). It is a Trainer attribute,
not part of the model (so not in `model.non_trainable_variables`, BN backup/restore, `checkpoint_variables()`
or any checkpoint). The traced step reads it (captured variable: no signature change, no retrace, trace
counts stay 1). `train_step(self, task, batch_tuple, skip=False, lso_skip=False)` assigns
`self._lso_skip[task].assign(bool(lso_skip))` before calling the step when AMP is on (ignored otherwise);
`fit` passes the D4 flag. Gate in BOTH `_make_train_step` and `_make_deform_step`:

```python
if amp:
    lso_skipped = tf.logical_and(self._lso_skip[task], tf.logical_not(tf.math.is_finite(terms["loss"])))
    gate = tf.logical_and(in_finite, tf.logical_not(lso_skipped))
else:
    gate = finite
tf.cond(gate, apply, lambda: tf.constant(False))
# result: res["lso_skipped"] = lso_skipped   (AMP only; train_step converts it to bool like "finite")
```

(`task` inside `_make_deform_step` is `DEFORM`.) BN rollback (`tf.cond(finite, ..., restore)`) is unchanged.
`train_step` must treat "lso_skipped" like "finite" (bool, excluded from the float conversion).

D6. Window logging. Per log window `fit` counts per task `win_steps[t]` and `win_nonfinite[t]` (plain ints,
not the MeanAccumulator), and at every log record writes, for every task with >= 1 step in the window:
`train_<t>/window_steps`, `train_<t>/nonfinite_steps`, `train_<t>/nonfinite_frac` (= nonfinite / steps);
plus `optimizer/loss_scale` = `float(self.optimizer.loss_scale.numpy())` iff
`isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer)`. If a task's window fraction
> `NONFINITE_WARN_FRAC = 0.1`: stderr `WARNING step %d: task %s non-finite in %d of %d steps of the last
log window (%.0f %%)` (flush=True). Constants: `NONFINITE_WINDOW_KEYS = ("window_steps", "nonfinite_steps",
"nonfinite_frac")`, `LOSS_SCALE_KEY = "optimizer/loss_scale"`. The `nonfinite_step_skipped` event gains
fields `task_streak` and `lso_skipped` (AMP only).

D7. Checker. New `check(...)` kwargs / CLI flags, default OFF so existing result-name lists are unchanged:
`--max-window-nonfinite-frac F` (default 0 = off; scripts pass 0.05), `--nonfinite-window-steps W`
(default 500), `--nonfinite-from-step S` (default 100). With F > 0, per task in `--tasks` one result
`nonfinite_window_<task>`: aggregate the logged `train_<t>/nonfinite_steps` / `window_steps` of train records
with `step > S` into buckets `(step - 1) // W`; buckets with < `MIN_WINDOW_TASK_STEPS = 20` task steps are
skipped; FAIL if any bucket's fraction > F, detail = first such bucket's step range, counts, percent and
the first record step in it with nonfinite_steps > 0. Task without the keys (old runs) -> PASS
"skipped: no train_<t>/nonfinite_steps logged". Loss scale: when any record has `optimizer/loss_scale`
(independent of F), one result `loss_scale` passed=True, detail
`"INFO last %g, min %g (step %d), %d of %d log windows at the floor (<= 1)"`, and when the LAST value <= 1
the detail is `"INFO WARN loss scale at the floor (1): healthy tasks train with fp16 gradient underflow
risk; ..."` (INFO label, never a failure). `--max-nonfinite-frac` stays as is.

D8. a1 `--fp16-act-clip C` (float, default 0 = off; allowed 0 or 0 < C <= 32000; > 0 needs `--amp true`)
and `--fp16-act-clip-scope {deform_decoder, all}` (default `deform_decoder`; `deform_decoder` with C > 0
needs 'deform' in `--task-ratio`). Scope `deform_decoder` = d5..d1 GroupConvBlocks + every
BatchNormWithSeparateAffine inside the deform decoder (incl. `svf_norm_act` and the blocks' pre-activation
BN); `all` = every GroupConvBlock and BatchNormWithSeparateAffine of the trainable model (encoder incl.
stem, linear-decoder 1x1 reductions, deform decoder). Default scope is the decoder because the report ranks
it first and clipping the encoder would also change the healthy linear tasks; `all` is the fallback if the
probe names an encoder layer. Clip = `tf.clip_by_value(y, -C, C)` (constants in y.dtype) applied ONLY when
`C > 0 and y.dtype == tf.float16` (trace-time Python check: no op otherwise). Applied to: GroupConvBlock
conv output (standard path) and, in `_split_call`, to EACH of the two conv terms and to the bias-added sum
(two clipped terms sum to <= 2C = 64000 < 65504, hence the 32000 cap); BatchNormWithSeparateAffine
output (after the affine). LeakyReLU, group mean, concat and trilinear upsampling cannot exceed their
inputs' range, so these producers cover every fp16 overflow source of the conv path. Clipped voxels get
zero gradient (documented: symptom treatment). Frozen cascade stages are never clipped (built separately).

D9. a2 `--deform-decoder-dtype {policy, float16, float32}` (default `policy` = global policy, today).
`float16` and `float32` need `--amp true` (`float16` is then identical to `policy`; `float32` without AMP
would be a no-op, so it is a parser error); `float32` needs 'deform' in `--task-ratio`.
Implementation: `GroupRegLevelModel(..., deform_decoder_dtype="policy")`; for "float32" while the
global policy's compute dtype is not float32, construct `DeformDecoder` inside
`models.precision.policy_scope("float32")` so the decoder AND all its sublayers (blocks, convs, BN, act,
upsample, svf_norm_act, svf_head) get the float32 policy (Keras captures the policy at layer construction;
all decoder sublayers are created in `__init__`). "float16" under a float32 global policy -> ValueError.
`DeformDecoder.call` casts its input explicitly: `x = tf.cast(skips[5], self.compute_dtype)` (no-op when
the dtype already matches, so the default graph is unchanged; Keras autocast would do it too). Encoder
stays fp16; gradients flow back through the casts. Variables are float32 in both modes with identical
names/order, so checkpoints, warm start (`C.warm_start_model`), `copy_weights_from` (cascade init) and
resume are compatible between dtypes. It is a Python attribute like `lean_sampler` (NOT a ModelConfig field:
config equality checks for warm start / cascade sources stay valid). Works with every
`--gradient-checkpointing` mode (the decoder is called through `__call__` in all of them), lean sampler and
split group conv (`_split_call` casts the kernel to x.dtype). Memory: ~2x decoder activations; documented as
fine at levels 4/2 and not fitting level 1 at 4 pairs under the 40000 MB limit.

D10. a3 (cheap, done as a runtime override, not a ModelConfig change). `ModelConfig.bn_epsilon` already
exists (1e-5) but is shared by BN, LayerNorm and InstanceNorm and is part of the config that warm start and
`--cascade-from` compare, so changing it is unusable exactly where a near-zero recalibrated variance
occurs (levels 2/1). Instead: `--bn-epsilon-override E` (float, default 0 = off; E > 0 finite):
`GroupRegLevelModel(..., bn_epsilon_override=0.0)` sets `bn.epsilon = E` on every
BatchNormWithSeparateAffine of the trainable model in `__init__` (before build), via
`layers.set_bn_epsilon(layer, eps)`. LayerNorm / InstanceNorm and frozen cascade stages untouched; the
override is recorded in config.json args; it must be passed again on resume (the scripts do).

D11. Scripts env (all default off, no flag emitted when off): `FP16_ACT_CLIP` (default 0, `need_float`;
> 0 -> `--fp16-act-clip $FP16_ACT_CLIP --fp16-act-clip-scope $FP16_ACT_CLIP_SCOPE`),
`FP16_ACT_CLIP_SCOPE` (default deform_decoder; `need_choice deform_decoder all`),
`DEFORM_DECODER_DTYPE` (default policy; `need_choice policy float16 float32`; != policy -> flag),
`BN_EPSILON_OVERRIDE` (default 0; `need_float`; > 0 -> flag). In linear-only ratios of r_level4.sh,
deform-scoped clip and the decoder dtype are dropped with a NOTE (same pattern as the deform flags).
Checker call in all four scripts gets `--max-window-nonfinite-frac 0.05`. `--nonfinite-patience-task` is not
exposed (default 20; EXTRA_ARGS can override).

D12. Restructuring: not done. The two halves (guard/logging/checker vs fp16 flags) share `runtime/args.py`,
`main.py` and the four scripts; the existing implement/review loop runs this plan.

## Existing tests that touch the changed behaviour (checked)

- No test asserts the exact key set of a train log record (checked `tests/test_trainer.py` 750, 904-906,
  1079-1088, 1625-1634; `tests/test_step_times.py` 84/99 only look at `*compute_ms`; `scripts/summarize_runs.py`
  reads `nonfinite_total`, `gpu_peak_mb`, `train/collapse_suspect`). New keys must not end in
  `/grad_norm_scale|/grad_norm_gated_frac|/grad_norm_normed` (they don't). Re-grep before finishing:
  `assertEqual(set(` / `sorted(` over `recs` / `history` keys in tests/.
- `tests/test_check_mixed_run.py:147-150` asserts the exact default result-name list -> the new results must
  appear only with F > 0 / when `optimizer/loss_scale` is logged (fixture logs neither). 291-294 and 504-507
  assert absence of milestone names with defaults: keep.
- `tests/test_trainer.py` `test_nan_guard_skips_update` (patience 2, rigid-only, raises on the 3rd skip),
  `test_mixed_precision_smoke` (inner iterations == 21 after fit(20)+1: all overflow steps there have
  finite loss -> still reach the LSO), `test_mixed_precision_overflow_skips` (direct `train_step`,
  `lso_skip` default False), `TestDeformStep.test_nan_guard_skips_update` (non-AMP, iterations unchanged),
  `test_amp_normalised_and_overflow`: must pass unchanged.
- `tests/test_sweep_tools.py::test_script_env_keys_cover_r_level4` -> add the new env names to
  `SCRIPT_ENV_KEYS` in `slurm/sweep_level4.py`. `test_r4_*` / r2 / r1 dry-run tests and
  `tests/test_slurm_scripts.py::test_m9_level1_default_recipe` (asserts some flags are NOT in main.py's argv)
  -> new flags must be absent by default.
- `tests/test_runtime_infra.py:364` asserts ModelConfig has no new fields: we add none.
- Goldens: `tests/test_trainer.py` `TestLinearOnlyUnchanged` (fixtures/m7_linear_golden.json) and
  `TestDeformUnchanged` (fixtures/lncc_deform_golden.json): fp32, finite -> must print `max |diff| 0`.
  NEVER set BGR_WRITE_GOLDEN.

## Verification helpers (create once, delete at the end)

`.agents/tmp/nf_py38.sh` (TF-free: syntax + TF-free tests):

```bash
#!/usr/bin/env bash
cd /mnt/d/Research/Brain_Registration_Projects/brain-group-rad
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bgr_preproc38
export PYTHONDONTWRITEBYTECODE=1
F="runtime/run.py runtime/args.py models/layers.py models/group_net.py models/precision.py models/cascade.py main.py tests/gpu/check_mixed_run.py tests/test_trainer.py tests/test_fp16_mitigations.py tests/test_check_mixed_run.py tests/test_runtime_infra.py tests/test_main.py tests/test_slurm_scripts.py tests/test_sweep_tools.py slurm/sweep_level4.py"
python -m py_compile $F && echo "py_compile OK"
python - $F <<'EOF'
import ast, sys
for f in sys.argv[1:]:
    ast.parse(open(f).read(), f, feature_version=(3, 8))
print("ast feature_version=(3, 8) OK (%d files)" % len(sys.argv[1:]))
EOF
python -c "import ast; ast.parse(open('slurm/sweep_level4.py').read(), feature_version=(3, 6)); print('sweep py3.6 OK')"
python -c "import runtime.args, sys; assert 'tensorflow' not in sys.modules; print('runtime.args TF-free')"
python main.py --help >/dev/null && echo "main.py --help OK"
for s in slurm/r_level4.sh slurm/r_level2.sh slurm/r_level1.sh slurm/m9_level1_cascade.sh; do bash -n "$s" && echo "bash -n $s OK"; done
python -m unittest tests.test_slurm_scripts tests.test_sweep_tools 2>&1 | tail -n 4
```

`.agents/tmp/nf_tf.sh` (TF modules, arguments = unittest names; 6000 MB limit, never higher):

```bash
#!/usr/bin/env bash
cd /mnt/d/Research/Brain_Registration_Projects/brain-group-rad
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
export PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 BGR_TF_GPU_LIMIT_MB=6000 TF_CUDNN_USE_AUTOTUNE=0
for m in "$@"; do python -m unittest "$m" -v 2>&1 | grep -v "cuBLAS factory" | tail -n 40; done
```

Run from PowerShell (one line each):
- `wsl -d Ubuntu bash -c "tr -d '\r' < /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.agents/tmp/nf_py38.sh | bash -s"`
- `wsl -d Ubuntu bash -c "tr -d '\r' < /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.agents/tmp/nf_tf.sh | bash -s -- tests.test_trainer"` (replace the module list as needed)

Full suite (baseline 647 OK / 4 skipped; ~3 h now, start with the background-process tool or redirect to
`.agents/tmp/nf_suite.log`): `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/tests/run_tests_wsl.sh`.
If any `tests/test_*.py` or script is written with CRLF, run it through `tr -d '\r'`; keep repo files LF.

## Items

- [ ] 1. Add `policy_scope(name)` to `models/precision.py` (contextlib context manager: save
      `tf.keras.mixed_precision.global_policy()`, `set_global_policy(name)`, restore in `finally`) and make
      `models/cascade.py::_float32_policy()` delegate to it (`return precision.policy_scope("float32")`;
      keep the name, callers unchanged).
      Files: models/precision.py, models/cascade.py
      Verify: `nf_tf.sh tests.test_precision tests.test_cascade` -> all OK.

- [ ] 2. Layers (D8, D10). In `models/layers.py`: module helper `clip_fp16(x, c)` (returns x unchanged unless
      `c > 0 and x.dtype == tf.float16`); attribute `fp16_act_clip = 0.0` on `GroupConvBlock` and
      `BatchNormWithSeparateAffine` (set in `__init__`); apply it in `GroupConvBlock.call` (conv output),
      `GroupConvBlock._split_call` (each conv term, then the bias-added sum) and
      `BatchNormWithSeparateAffine.call` (after the affine). Add `set_fp16_act_clip(layer, clip)` (sets the
      attribute on every GroupConvBlock and BatchNormWithSeparateAffine in `[layer] + layer.submodules`,
      returns the count; "read at trace time: re-trace after changing it", like `set_split_group_conv`) and
      `set_bn_epsilon(layer, eps)` (sets `m.bn.epsilon = float(eps)` on every BatchNormWithSeparateAffine,
      returns the count). Docstrings state the defaults add no op.
      Files: models/layers.py
      Verify: `nf_tf.sh tests.test_group_layers tests.test_split_group_conv` -> all OK (unchanged).

- [ ] 3. Model kwargs (D8, D9, D10). `GroupRegLevelModel.__init__(..., fp16_act_clip=0.0,
      fp16_act_clip_scope="deform_decoder", deform_decoder_dtype="policy", bn_epsilon_override=0.0)`:
      validate (clip 0 or (0, 32000]; scope in `FP16_ACT_CLIP_SCOPES = ("deform_decoder", "all")`; dtype in
      `DEFORM_DECODER_DTYPES = ("policy", "float16", "float32")`; "float16" needs the global compute dtype
      float16; override 0 or finite > 0), store them as attributes; build `DeformDecoder` inside
      `precision.policy_scope("float32")` iff dtype == "float32" and the global compute dtype is not
      float32 (else `contextlib.nullcontext()`); after `set_split_group_conv`, call
      `L.set_fp16_act_clip(self.deform_decoder or self, clip)` when clip > 0 (scope deform_decoder without a
      deform decoder -> ValueError), and `L.set_bn_epsilon(self, eps)` when eps > 0. `DeformDecoder.call`:
      `x = tf.cast(skips[5], self.compute_dtype)`. Extend the class docstring (Python attributes, not
      ModelConfig; checkpoint compatibility; memory note for float32 at level 1).
      Files: models/group_net.py
      Verify: `nf_tf.sh tests.test_group_net tests.test_checkpointing tests.test_cascade` -> all OK.

- [ ] 4. New tests for items 1-3 in `tests/test_fp16_mitigations.py` (first import `tests.tf_setup`; set
      `mixed_float16` in try/finally and restore `float32`, like `test_trainer.test_mixed_precision_smoke`;
      `free_memory()` in tearDown):
      (a) off = no new ops: under mixed_float16 build a narrow level-4 deform model with defaults and one
      with explicit off kwargs, same weights; trace `tf.function(lambda x: m(x, task="deform",
      training=True))` and compare `[op.type for op in graph.get_operations()]` -> equal; with clip 30000
      the graph contains `ClipByValue` ops; with scope deform_decoder no encoder layer has
      `fp16_act_clip > 0`; scope all sets it on encoder blocks too.
      (b) a1 prevents the repro overflow: level 2, N 3, `tiny_config(deform_decoder=True, seed=1,
      integration_steps=7)`, `set_bn_stats_frozen(True)`, phantoms via `fd.synthetic_label_phantom` (as in
      `.agents/tasks/m9-level1-nan/repro_fp16_overflow.py`), svf_head kernel N(0, 0.02), d2 kernel x 1e5:
      assert the precondition (unclipped fp16 eval forward: `v_raw` has non-finite values), then the same
      weights in a model with `fp16_act_clip=30000` give finite `v_raw`, `u`, `u_inv` and a finite
      `losses.deform_loss`; also with `split_group_conv=True`.
      (c) a2: `deform_decoder_dtype="float32"` under mixed_float16 -> every Layer in
      `model.deform_decoder.submodules` (and the decoder) has compute_dtype float32, the encoder float16;
      eval `v_raw` vs a float32-policy model with the same weights (svf head N(0, 0.02)):
      max|diff| <= 2e-2 * max|v32| + 1e-6; the k = 1e5 case of (b) is finite; AMP Trainer step with
      `gradient_checkpointing="full", lean_sampler=True, split_group_conv=True` (cfg `lean_sampler=True`,
      `amp=True`, deform-only) runs 3 finite deform steps with `trace_counts == {"deform": 1}`.
      (d) checkpoint compatibility: `tf.train.Checkpoint(model=a2_model).write(prefix)` restored into a
      default-dtype model (and back) -> all variables equal; `default.copy_weights_from(a2_model)` works.
      (e) a3: `bn_epsilon_override=1e-2` -> every `BatchNormWithSeparateAffine.bn.epsilon == 1e-2`,
      LayerNorm epsilons unchanged, `model.config.to_dict()` unchanged; with a moving_variance set to 1e-6
      and frozen BN the eval output differs from the default model with the same weights.
      (f) invalid kwargs raise ValueError (clip -1 / 40000, unknown scope / dtype, float16 under float32
      policy, scope deform_decoder without deform decoder).
      Files: tests/test_fp16_mitigations.py
      Verify: `nf_tf.sh tests.test_fp16_mitigations` -> all new tests OK.

- [ ] 5. Per-task guard, LSO skip, ckpt protection (D1-D5) in `runtime/run.py`. `TrainConfig`: add
      `nonfinite_patience_task: int = 20` with a comment (0 = off, D1). Module constants `LSO_SKIP_STREAK`,
      `SAVE_LAST_MAX_STREAK`. `Trainer.__init__`: validate `nonfinite_patience_task` (int >= 0), create
      `self._lso_skip` before `rebuild_steps()`. Gate change in both step builders exactly as D5, result key
      `lso_skipped` (AMP only); `train_step` gets `lso_skip=False` and converts `lso_skipped` to bool.
      `fit`: `consecutive_task`, `streak_first`, LSO flag computed before `train_step`, old global check
      first, then per-task check, D2 event + flush before both raises, D3 deferred save logic for the
      periodic and the final `save_last` (`last_saved_step` tracked for the message), new fields on
      `nonfinite_step_skipped`. Update the module docstring (train-step pseudo code line
      `tf.cond(finite, ...)`, the "Under --amp the LossScaleOptimizer is always called" paragraph, a new
      "Non-finite guard" paragraph with D1-D4) and document the NOT implemented future mode: keep a
      `ckpt/good` manager saved only when no task had a non-finite step in the last window; on a trip restore
      model + optimizer + LSO + RNG from it, multiply the failing task's lr by 0.5, allow at most 2 recoveries.
      Files: runtime/run.py
      Verify: `nf_tf.sh tests.test_trainer` -> all existing tests OK, golden lines print `max |diff| 0`.

- [ ] 6. Window logging (D6) in `Trainer.fit` (`runtime/run.py`): per-task window counters reset with
      `acc.reset()`, keys written into `m` before `self._log(done, m)`, `optimizer/loss_scale` under an LSO,
      stderr warning above 0.1. Constants exported next to `ENC_GRAD_KEYS`.
      Files: runtime/run.py
      Verify: `nf_tf.sh tests.test_trainer tests.test_cascade` -> all OK.

- [ ] 7. Trainer tests for items 5-6 in `tests/test_trainer.py` (new classes after `TestTaskGradNorm`,
      reuse `DeformCase`, `config`, `fd.to_deform_batch`, `RunLogger`/`make_run_dir`,
      `C.RunCheckpoint(paths["ckpt_last"], paths["ckpt_best"], tr.model, tr.optimizer, tr.step,
      extra=tr.checkpoint_variables())` as in `TestResumeAndValidation.with_checkpoint`, `_OpDeterminism`,
      `_seed_with_all_tasks`):
      (a) per-task trip: ratio rigid:1, deform:1, deform source = NaN-image batch (as
      `TestDeformStep.test_nan_guard_skips_update`), `nonfinite_patience_task=3`, default
      `nonfinite_patience=20`: `fit(60)` raises RuntimeError whose message contains "task deform", "4
      consecutive" and the first failing step; history shows finite rigid steps interleaved (the old
      counter alone would not trip); the `nonfinite_abort` event is in metrics.jsonl. With
      `nonfinite_patience_task=0` the same `fit(30)` does not raise.
      (b) ckpt/last protection: same setup with `ckpt_every=1`, `nonfinite_patience_task=6`, wrap
      `tr.checkpoint.save_last` in a spy recording `int(tr.step.numpy())`: no save happens at a step where a
      task streak >= 2 is open; after switching the deform source back to the finite batch the deferred
      save happens at the first step with no open streak and a `ckpt_last_deferred` event exists; in the
      raising variant the last spy step < first failing step + 2.
      (c) LSO not halved by a streaking task (AMP; mixed_float16 try/finally): `LossScaleOptimizer(inner,
      initial_scale=2.0 ** 4)` (inner `O.make_optimizer("adam", lr=1e-3, clipnorm=None)`), make3-style
      model, ratio rigid:1, deform:1, one NaN in `model.deform_decoder.d1.conv.kernel` (forward loss NaN):
      `fit` over >= 5 deform steps with `nonfinite_patience_task=50` -> loss scale == 2^4 / 2 exactly (only
      the first failure of the streak halved), `lso_skipped` True for deform failures 2.., rigid steps
      finite and applied (weights change), trace counts 1. Restore the kernel: deform finite again, streak
      reset. Control: the same failing sequence through `train_step(..., lso_skip=False)` halves every time.
      (d) isolated / gradient-overflow unchanged (AMP): rigid-only, `initial_scale=2.0 ** 40`
      (consecutive gradient overflows with FINITE loss, as `test_mixed_precision_overflow_skips`): `fit(k)`
      vs a twin trainer (same weights) driven by plain `train_step` calls -> identical loss-scale
      sequence, inner `iterations` and weights (`np.array_equal`, under `_OpDeterminism`); one isolated
      forward-NaN deform step (kernel NaN for one deform step only) halves the scale once, exactly as today.
      (e) log keys: `log_every=4`, rigid:1, deform:1, NaN deform source for some steps -> records hold
      `train_<t>/{window_steps, nonfinite_steps, nonfinite_frac}` with the right counts; stderr (captured with
      `contextlib.redirect_stderr`) has the WARNING when frac > 0.1; no `optimizer/loss_scale` without AMP,
      present (== current scale) with AMP; all-finite run: nonfinite_steps 0 and no warning.
      Files: tests/test_trainer.py
      Verify: `nf_tf.sh tests.test_trainer` -> all OK, both goldens `max |diff| 0`.

- [ ] 8. CLI + main wiring. `runtime/args.py`: `--nonfinite-patience-task` (misc group, default from
      `TrainConfig`-matching 20, help explains D1), `--fp16-act-clip`, `--fp16-act-clip-scope`,
      `--deform-decoder-dtype`, `--bn-epsilon-override` (model / memory group next to `--split-group-conv`,
      help texts with D8-D10 rationale), validation in the existing post-parse block (`parser.error`
      for: K < 0; C outside {0} U (0, 32000]; C > 0 without `--amp true`; scope deform_decoder or dtype
      float32 without 'deform' in `--task-ratio`; dtype float16/float32 without `--amp true`; override < 0 or
      non-finite); `train_config_from_args` passes `nonfinite_patience_task`; module docstring bullet.
      `main.py`: pass the four model kwargs to `GroupRegLevelModel(...)` (line ~278; the global policy is set
      before, line ~228). Tests in `tests/test_runtime_infra.py` (defaults, each error, mapping to
      TrainConfig / `args_to_dict`) and one `tests/test_main.py` run via `self.run_main(name, [...], steps=2)`
      with a deform ratio and `--amp true --fp16-act-clip 30000 --deform-decoder-dtype float32
      --bn-epsilon-override 1e-3` that finishes and records the values in config.json args (follow the
      existing deform `run_main` tests for the required flags).
      Files: runtime/args.py, main.py, tests/test_runtime_infra.py, tests/test_main.py
      Verify: `nf_py38.sh` (runtime.args stays TF-free, `main.py --help` OK); `nf_tf.sh
      tests.test_runtime_infra tests.test_main` -> all OK.

- [ ] 9. Checker (D7) in `tests/gpu/check_mixed_run.py`: new `check()` kwargs `max_window_nonfinite_frac=0.0,
      nonfinite_window_steps=500, nonfinite_from_step=100`, helpers `_check_nonfinite_windows(recs, tasks, ...)`
      and `_loss_scale_info(recs)`, CLI flags, module docstring section. Tests in
      `tests/test_check_mixed_run.py` using `make_run(..., train_extra=...)` (extend `make_run` with an
      optional per-record callback if needed): deform 30 % non-finite in one 500-step window after step 100
      -> FAIL naming that window and its first step; the same before step 100 -> PASS; isolated (1 of 125
      rigid steps) -> PASS; small trailing bucket (< 20 task steps) skipped; keys missing -> PASS
      "skipped"; F = 0 -> no `nonfinite_window_*` names (existing list test unchanged); loss-scale INFO line
      present only when logged, WARN text when the last value <= 1, never a failure; CLI flags reach
      `check()` (`cmr.main([...])` exit codes).
      Files: tests/gpu/check_mixed_run.py, tests/test_check_mixed_run.py
      Verify: `nf_tf.sh tests.test_check_mixed_run` -> all OK.

- [ ] 10. Scripts (D11): `slurm/r_level4.sh`, `slurm/r_level2.sh`, `slurm/r_level1.sh`,
      `slurm/m9_level1_cascade.sh`: env defaults + validation (`need_float` / `need_choice`, or the existing
      error style in m9), a `MITIG_ARGS` string appended to main.py's argv before `$EXTRA_ARGS`, an echo line
      (`fp16 act clip ... | deform decoder dtype ... | bn epsilon override ...`), header comment block
      ("Non-finite mitigations (nonfinite-fixes, default off; decide after the m9 NaN probe): ...", include
      that level-1 DEFORM_DECODER_DTYPE=float32 is not expected to fit 4 pairs under TF_MEM_LIMIT_MB 40000)
      and `--max-window-nonfinite-frac 0.05` in the checker call. `slurm/sweep_level4.py`: add
      `FP16_ACT_CLIP`, `FP16_ACT_CLIP_SCOPE`, `DEFORM_DECODER_DTYPE`, `BN_EPSILON_OVERRIDE` to
      `SCRIPT_ENV_KEYS`. Tests: `tests/test_sweep_tools.py` (r4/r2/r1 dry runs: defaults emit none of the four
      flags and the checker has `--max-window-nonfinite-frac 0.05`; each env override emits its flag and
      `check_parses(main)` passes; invalid values exit 2; linear-only r4 drops deform-scoped ones with a NOTE)
      and `tests/test_slurm_scripts.py` (m9 default recipe has no new flags but the new checker flag;
      override case).
      Files: slurm/r_level4.sh, slurm/r_level2.sh, slurm/r_level1.sh, slurm/m9_level1_cascade.sh,
      slurm/sweep_level4.py, tests/test_sweep_tools.py, tests/test_slurm_scripts.py
      Verify: `nf_py38.sh` -> bash -n OK for the four scripts, sweep py3.6 OK, TF-free tests OK.

- [ ] 11. Final verification and cleanup. Re-grep tests for exact key-set assertions (see above). Run
      `nf_py38.sh`, then `nf_tf.sh tests.test_trainer.TestLinearOnlyUnchanged tests.test_trainer.TestDeformUnchanged`
      (both print `max |diff| 0`), then the full suite
      `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/tests/run_tests_wsl.sh`
      (expect 647 + new tests OK, 4 skipped, "suite status: 0"). Record in this plan's checkboxes / the
      step output: test counts, golden diffs, any deviation. Delete `.agents/tmp/nf_*.sh` and logs.
      Files: none (verification only)
      Verify: suite status 0; goldens 0; no stray files from `git status --short` beyond the planned ones.

## Gaps / assumptions

- The LSO skip adds the "forward loss non-finite" condition to the brief's "streak >= 2" rule (D4) so the
  AMP warm-up's consecutive gradient overflows keep halving the scale; this is the behaviour-preserving
  reading of "isolated early AMP overflows must still reach the LSO exactly as today".
- a3 is a runtime override instead of a ModelConfig field (D10) because the existing field is shared with
  LayerNorm and checked by warm start / cascade; flag the deviation in the step output.
- The probe job (cluster) decides which mitigation to enable; nothing here enables one by default.
