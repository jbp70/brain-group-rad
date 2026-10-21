# Optional per-task gradient normalisation for the three-task trainer

The change adds `--task-grad-norm T` / `--task-grad-norm-min m` (default T = 0, off). When on, each task step rescales its raw, AMP-unscaled gradient to global norm T, unless the raw norm is below m or non-finite. It then multiplies by the task weight, which is removed from the objective, and passes the result to the existing optimizer with the global clip intact. This replaces the hand-tuned `deform:64` weight with an explicit mechanism. The off path is a trace-time Python branch, so existing runs are unaffected. SLURM env overrides, logging keys, checker output and a design.md v3.10 note come with it.

Watch for: the gate threshold is compared against the linear norm after `--loss-scale-mm2` (documented, possible surprise if S is changed); the checker's "gated frac" is an unweighted mean of window means (info only). Neither blocks.

**Verdict**: APPROVED

## High-level view

The scale is `T * (1 / max(g, 1e-12))` when g is finite and ≥ m, otherwise 1. This matches the requested rule exactly: eps plus a min gate, with gated and non-finite steps left raw. The factor is written to mirror `tf.clip_by_global_norm`, which is why "T 12 without clip" is bit-identical to "clip 12 with weights 2^14" in the equivalence test.

Weights are applied after normalisation (`factor = scale * w`), and the objective multiplication is suppressed with `not norm_on` in both the linear and deform steps. The global clip is untouched. With T ≤ clipnorm it is a no-op up to 1 ulp, which the report measures (losses 9e-7 relative, variable sums ≤ 1.1e-3 relative after Adam amplification).

The off path is guarded by `norm_on = self.task_grad_norm > 0` at trace time, so no ops are added. The report shows both goldens (`m7_linear_golden.json`, `lncc_deform_golden.json`) at max |diff| 0 with fixtures not regenerated. The full suite ran 327 tests OK (313 + 14 new, same 4 skips).

Logging keeps `grad_norm` as the raw norm, computed before normalisation. `_eff` now includes `post = scale * w` and the clip factor evaluated on `g * post`. The new keys `grad_norm_scale`, `grad_norm_gated_frac` and `grad_norm_normed` are emitted only when the flag is on. Non-finite handling is unchanged: `finite` still comes from the raw norm, inf/NaN norms get scale 1, and under AMP the LSO still receives the inf gradients (covered by `test_amp_normalised_and_overflow`).

The SLURM script defaults both env vars to empty, so no flags are passed. It warns when weights are combined with normalisation and when MIN is set without a target. `bash -n` is reported clean. Python 3.8 compatibility is reported via py_compile and `ast.parse(feature_version=(3, 8))` on all changed .py files.

<details>
<summary>Issues (2)</summary>

1. **Gate sees loss-scaled linear norm** — for linear tasks the gate compares m against the gradient of `loss / loss_scale_mm2`, so a large S could push linear steps under the gate. It's documented in design.md; consider noting it in the `--task-grad-norm-min` help text as well. Non-blocking.
2. **Checker gated fraction is unweighted** — `check_mixed_run.py` averages per-window `grad_norm_gated_frac` means without weighting by each window's task step count. That's fine for an INFO line; weight by counts if it ever becomes a pass/fail criterion. Non-blocking.

</details>

<details><summary>Details</summary>

### Normalisation math and ordering

`task_grad_norm_scale` computes `ok = finite & (g >= min)` and `scale = where(ok, T * (1/max(g, eps)), 1)`. `gated` is set only for finite sub-min norms, so NaN/inf steps are not counted as gated. Both `tf.where` branches are evaluated, and eps keeps the unselected division finite, so there is no NaN leakage through the gradient-free `where`. `_normalise_gradients` multiplies by `scale * w` (skipping the constant when w == 1) and returns `global_norm(applied)` as `grad_norm_normed`. `scale_gradients` preserves `IndexedSlices`. None of the current variables produce them, but the handling is correct if any do.

The default min of 1e-3 is justified with measured norms: deform 0.03–0.5, linear ≥ 1e3, so nothing is gated in the 400-step local run. `test_eps_zero_grads` shows the gate is necessary: a collapsed group with min 0 gets a 1.2e13 gain on ~1e-19 residues. The gate is a discontinuity (scale jumps from 1 to T/m), and that is documented. Because of how the loss scale enters the linear objective, the gate for linear tasks sees `raw / loss_scale_mm2` (**confirmed**, `objective / loss_scale` before the tape gradient). This is harmless at the recommended S = 1 but would shift gating if S were raised to 1e6 or more.

### Diagnostics

`_enc_grad_diagnostics(..., post=factor)` passes `mult` = 1 (deform) or `1/S` (linear), so `_raw` remains the encoder norm of the unweighted, unscaled loss. The `_eff` clip factor uses `gnorm * post`, which equals the global norm of the applied gradients for dense tensors. `test_weights_after_normalisation` checks both the unclipped and the clip-1 cases. In the checker (`tests/gpu/check_mixed_run.py`), the gated fraction is `sum(gated)/len(gated)` over window means, which is an unweighted mean of per-window fractions (**confirmed**). This is info only and adds no new pass/fail names.

### Equivalence and test coverage

`TestTaskGradNormClipEquivalence` covers the A0 case (T 12, clip off) against B (clip 12, weights 2^14) with a bit-exact result. A12 (clip kept) is near-equal, and the report attributes the difference to 1-ulp clip factors that Adam amplifies on elements with near-zero moments. The test tolerances are 1e-5 relative on losses and 1e-2 on variables. The remaining tests cover the gate (bit-identical to flag-off), the eps path, weight ordering, the NaN skip, AMP overflow, window logs, CLI validation and the absence of new keys when the flag is off. Not tested: hovering at the gate across many steps (logged but not exercised), and real-data encoder-share behaviour. The report flags the latter, noting that in the narrow synthetic run equal total norms gave a linear/deform encoder _eff ratio of ~0.05; the cluster A/B has to settle it.

</details>

<details>
<summary>File map</summary>

- `runtime/run.py`: `TASK_GRAD_NORM_EPS`, `TASK_GRAD_NORM_KEYS`, `task_grad_norm_scale`, `scale_gradients`, `Trainer._normalise_gradients`, wiring in the linear and deform steps, `_eff` with `post`, window logging, docstrings.
- `runtime/args.py`: the two flags, finite and ≥ 0 validation, passed into `TrainConfig`.
- `slurm/m7_three_task.sh`: `TASK_GRAD_NORM` / `TASK_GRAD_NORM_MIN` env overrides (empty = off), warnings, echo line, example sbatch line.
- `tests/gpu/check_mixed_run.py`: scale median and gated fraction in the `enc_grad_<task>` detail.
- `tests/test_trainer.py`, `tests/test_runtime_infra.py`, `tests/test_check_mixed_run.py`: the new tests; `_three_task_run` refactor of the golden helper.
- `.agents/tasks/network-design/design.md`: v3.10 status line and a section 5.2 bullet.

Full diff: `git diff` in the workspace (uncommitted).

</details>
