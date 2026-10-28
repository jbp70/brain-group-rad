# Per-task non-finite guard, LSO protection and opt-in fp16 overflow mitigations

The diff fixes the two structural bugs from the m9 level-1 NaN investigation. First, the global non-finite counter was reset by every finite linear step, so `--nonfinite-patience` could never trip in a mixed run. Second, a dead deform task kept halving the shared AMP loss scale until it reached 1. The fix adds a per-task streak guard, keeps ckpt/last from being overwritten during a streak, keeps a streaking task's forward-NaN steps away from the LossScaleOptimizer, and adds per-window non-finite logging plus a checker window test. It also adds three opt-in mitigations, all off by default: an fp16 activation clip (a1), a float32 deform decoder (a2) and a BN epsilon override (a3). The coder's evidence is the goldens at `max |diff| 0`, an AMP twin test that gives bit-identical weights and loss-scale sequences for warm-up overflows, trace counts of 1, and 677/4-skipped tests with the three failing modules rerun clean in isolation. That evidence supports the bit-identity claim. Watch for: (likely) an isolated forward NaN that directly follows a gradient-only warm-up overflow of the same task is LSO-skipped, which contradicts the "isolated forward NaNs always reach the LSO" wording; (possible) with `--nonfinite-patience-task 0` a permanently dead task blocks ckpt/last for the rest of the run. Neither one blocks.

**Verdict**: APPROVED

## High-level view

The per-task guard sits next to the old global counter. The global check runs first, so single-task runs keep the old message. Each task's streak is reset only by a finite step of the same task. Both raises log a `nonfinite_abort` event and flush the logs first. The ckpt/last protection counts only "hard" failures: AMP gradient-only overflows are exempt. This exemption deviates from plan D3, and it is justified by a real failure in T-E1, where the warm-up blocked the final save of a 3-step run. A due save that is blocked is deferred to the first step with no open streak. A blocked final save is logged and skipped.

The LSO skip is a captured `tf.Variable(bool)` per task, assigned eagerly before each AMP step and read inside the traced gate. The signature does not change and nothing re-traces. Under fp32 `_apply_gate` returns the old `finite` gate unchanged, so the golden graphs are unchanged. Under AMP the extra term evaluates to `in_finite` whenever the flag is false, so finite AMP steps are numerically unchanged. The twin test confirms this. The flag is gated on a non-finite forward loss, so the normal warm-up halvings still reach the LSO.

Window logging adds `train_<task>/{window_steps, nonfinite_steps, nonfinite_frac}`, plus `optimizer/loss_scale` only under an LSO, and prints a stderr WARNING above 10 %. The checker's per-task window test is off by default, so the existing exact result-name list test stays as it was. All four scripts pass `--max-window-nonfinite-frac 0.05`. The test diffs extend or tighten their assertions (new absent-flag checks, the checker flag sequence); none were loosened.

a1 adds `clip_by_value` only when C > 0 and the tensor is float16, decided at trace time. a2 builds the decoder under a float32 policy scope, and the variables stay float32 with the same names in every mode. a3 is a runtime override instead of a ModelConfig field, a documented and sensible deviation, because the config field is compared by warm start and the cascade. The env overrides in all four scripts emit no flag at their defaults. The level-1 scripts warn that a2 is not expected to fit there.

<details>
<summary>Issues (4)</summary>

1. **Warm-up overflow then forward NaN is LSO-skipped** (likely): `lso_flag` is computed from `consecutive_task`, which also counts gradient-only overflows. A forward NaN right after a same-task warm-up overflow therefore skips the LSO even though it is the first forward failure. The behaviour is harmless, since halving does not cure a forward NaN, but the docstring and report say isolated forward NaNs "always" reach the LSO. Either base the flag on `save_streak` (hard failures) or reword the docstring.
2. **ckpt/last frozen forever with the per-task guard off** (possible): with `--nonfinite-patience-task 0`, a task stuck in a hard-failure streak defers every ckpt/last save and also the final one. A resume then rewinds to the pre-streak state, possibly thousands of steps back. Document this next to the flag, or let deferral lapse after K' steps when the guard is off.
3. **a2 warm start not directly tested** (possible): `copy_weights_from` and Checkpoint round trips across dtypes are tested, but `C.warm_start_model` / `--cascade-from` with `--deform-decoder-dtype float32` is not. It is low risk because the variable names, shapes and dtypes are identical. A one-line test would close the gap.
4. **Jay's float32_fine / width grid not in this diff** (confirmed, scope): the project state says Jay added a `float32_fine` option (fp32 only for d1/d2/svf) and `slurm/m11_fp32_width_memory.sh` to this workflow. The report explicitly defers both as needing their own plan item. The orchestrator must schedule them before the m11 memory job, which was pre-approved to be submitted when this workflow lands.

</details>

<details>
<summary>Details</summary>

### Guard counters and the abort path in `Trainer.fit`

The global `consecutive` counter is untouched and is checked first, so `test_nan_guard_skips_update` (rigid only, patience 2) keeps its exact message. The per-task branch raises on `consecutive_task[task] > K`, with the message and `nonfinite_abort` fields from D2. `last_saved_step` is `None` until the first save inside this `fit()`, and the message prints "none since this fit() started" in that case. That is accurate, but after a resume it can read as if no ckpt/last exists. Streaks start empty on resume; this is fine because the ckpt/last being resumed from is pre-streak by construction.

### ckpt/last deferral and the warm-up exemption

```python
if not (amp and math.isfinite(res["loss"]) and not math.isfinite(res["grad_norm"])):
    save_streak[task] += 1
```

Gradient-only AMP failures leave the weights unchanged (the LSO skips the update), so exempting them from the save streak keeps ckpt/last accurate. The exemption has one consequence. A task with a persistent backward overflow at loss scale 1 never blocks ckpt/last. That is acceptable, because its weights do not move and the other tasks keep training. The per-task guard still trips for it, since `consecutive_task` counts every non-finite step.

The flip side is issue 2. With K = 0, a hard-failing task defers ckpt/last indefinitely. `ckpt_last_deferred` events make this visible, and `ckpt/best*` saving is unchanged, but neither the docstring nor the flag help mentions it.

### LSO skip gate

```
fit: lso_flag = amp and consecutive_task[task] >= 1   (assigned to self._lso_skip[task])
graph: lso_skipped = flag & ~isfinite(loss);  gate = in_finite & ~lso_skipped
```

The step counter and the lr schedule use `self.step` and `lr_at(s, ...)`, not `optimizer.iterations`, so an LSO-skipped step does not shift the schedule. Not incrementing `iterations` affects only Adam's bias correction. That is negligible.

Because `consecutive_task` includes gradient-only overflows, the sequence "deform warm-up overflow, then deform forward NaN" skips the LSO on the forward NaN (issue 1). The case is rare and harmless, but the documented contract claims otherwise.

The variables are Trainer attributes and are not reachable from `model.variables`, the BN backup or `checkpoint_variables()`. The tests also cover AMP plus cascade, and the trace counts stay at 1.

### Window logs and checker

The window counters reset together with the accumulator. `_check_nonfinite_windows` buckets records by `(step - 1) // W`. With `--log-every 25`, a record's window can straddle a bucket edge by at most 24 steps, which is acceptable at W = 500. The test counts gradient-only warm-up overflows as non-finite steps too. In `r_level4.sh` the deform task starts at step 3000, and its onset may produce a few consecutive halvings after step 100. The report estimates these at well under 5 % of about 250 deform steps per window. That holds for a handful of overflows, but there is no evidence from a real delayed-start AMP run yet. The first long L4 run will show whether `--nonfinite-from-step` needs raising. The `loss_scale` INFO line is emitted whenever the key is logged and is never a failure, as D7 specifies.

### fp16 mitigations

a1's scope `deform_decoder` covers d5..d1, `svf_norm_act` and the BN layers inside the blocks. That includes the d1 split-conv AddV2 the probe named as the first inf (job 10524598). The repro test covers `split_group_conv=True`. The clipped voxels get zero gradient, and the report states this. a2 covers the decoder and its sublayers, which the tests confirm by checking the compute dtype of every submodule. It also relies on all sublayers being created in `__init__`; a future sublayer created in `build()` would silently fall back to float16. The memory figures for a2 in the report are estimates, not measurements.

</details>

<details>
<summary>File map</summary>

- `runtime/run.py`: per-task streaks, abort event, ckpt/last deferral, LSO-skip variables and the `_apply_gate` helper, window metrics, docstring.
- `runtime/args.py`: `--nonfinite-patience-task`, `--fp16-act-clip(-scope)`, `--deform-decoder-dtype`, `--bn-epsilon-override` and their validation.
- `main.py`: passes the four model kwargs.
- `models/layers.py`: `clip_fp16`, `set_fp16_act_clip`, `set_bn_epsilon`, clip hooks in GroupConvBlock and BN.
- `models/group_net.py`: kwargs and validation, the float32 decoder policy scope, the input cast in `DeformDecoder.call`.
- `models/precision.py`, `models/cascade.py`: the shared `policy_scope`.
- `tests/gpu/check_mixed_run.py`: the window check and the loss-scale INFO line.
- `slurm/r_level{4,2,1}.sh`, `slurm/m9_level1_cascade.sh`, `slurm/sweep_level4.py`: env overrides and the checker flag.
- Tests: new `tests/test_fp16_mitigations.py`; additions to the trainer, checker, args, main, slurm and sweep tests.

Full diff: `git diff` in the repo (uncommitted).

</details>
