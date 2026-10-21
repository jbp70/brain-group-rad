# Report: optional per-task gradient normalisation (`--task-grad-norm`)

First iteration (no review.json). Not committed, not pushed, cluster not touched.

## Files changed

- `runtime/args.py`: `--task-grad-norm` (float, default 0 = off) and `--task-grad-norm-min` (float, default 1e-3) next to `--task-loss-weights`; `parse_args` rejects non-finite or negative values (`parser.error`); passed into `TrainConfig`.
- `runtime/run.py`: `TrainConfig.task_grad_norm` / `task_grad_norm_min`; `Trainer.__init__` validation (ValueError); module-level `TASK_GRAD_NORM_EPS = 1e-12`, `TASK_GRAD_NORM_KEYS`, `task_grad_norm_scale()`, `scale_gradients()`; `Trainer._normalise_gradients()`; both train steps (linear, deform) wired; `_enc_grad_diagnostics(..., post=None)`; window logging in `fit`; module docstring.
- `slurm/m7_three_task.sh`: env `TASK_GRAD_NORM`, `TASK_GRAD_NORM_MIN` (empty = flags not passed), comment recommending `TASK_LOSS_WEIGHTS=none`, warnings for weights + norm and for min without norm, echo line, example sbatch line.
- `tests/gpu/check_mixed_run.py`: when `train_<task>/grad_norm_scale` exists, appends `grad_norm_scale median ..., gated frac ...` to the `enc_grad_<task>` detail (info only, no new result names).
- Tests: `tests/test_trainer.py` (new classes `TestTaskGradNormHelpers`, `TestTaskGradNorm`, `TestTaskGradNormClipEquivalence`; `_deform_golden_run` now wraps a parameterised `_three_task_run` with identical defaults), `tests/test_runtime_infra.py` (`test_task_grad_norm_flag`), `tests/test_check_mixed_run.py` (`grad_norm_keys` option + assertions).
- `.agents/tasks/network-design/design.md`: v3.10 status line and a section 5.2 bullet "Per-task gradient normalisation".

## Semantics

In each task step, after the gradients are computed and AMP-unscaled and before `apply_gradients`:

- g = global norm of the task's raw gradient.
- If g is finite and g >= min: grads × target × (1.0 / max(g, 1e-12)).
- If g < min, or g is non-finite: scale 1 (raw gradients kept).
- Then × the task weight w.

With the flag on, w is not in the objective, so `--task-loss-weights` become ratios of applied norms (target × w). Gated steps get raw × w; with `TASK_LOSS_WEIGHTS=none` that is exactly the raw gradient.

Other behaviour:

- The global clip is kept.
- Non-finite handling is unchanged: the finite flag still comes from the raw norm, NaN/inf norms are never rescaled, and the AMP LSO still sees the inf gradients.
- With the flag off, a Python check at trace time adds no op.

The scale is written as `target * (1.0 / max(g, eps))` on purpose, to mirror `tf.clip_by_global_norm`. That is what makes the clip-equivalence test bit-exact.

Logged values:

- `grad_norm`: unchanged, the RAW norm. With the flag on it is the unweighted norm.
- `diag_enc_grad_norm_eff`: the applied encoder norm, i.e. raw encoder norm × scale × w × min(1, clip / (g × scale × w)).
- New per step and per window, flag on only:
  - `grad_norm_scale`: the applied factor, 1 when gated.
  - `grad_norm_gated_frac`: per step 1 or 0. The window mean is the gated fraction; gated count = fraction × the task's steps in the window.
  - `grad_norm_normed`: norm passed to the optimizer, before its clip.

## Default minimum: 1e-3

- Healthy deform total norms are 0.03-0.5, which is 30-500× above it. Linear totals are ≥ 1e3. So in normal training nothing is gated, and the local run below gated 0 steps.
- It equals `collapse_grad_norm` (1e-3). The steps the collapse detector already calls "zero gradient" (group collapse, degenerate batches) are left alone instead of being amplified.
- It caps the gain at target / min = 1.2e4 for target 12.
- The test `test_eps_zero_grads` shows why the gate is needed. For a collapsed group (identical images) the global norm underflows to exactly 0 while the gradients still hold ~1e-19 residues. With min 0 the eps path multiplies them by 1.2e13: the result is finite, with an applied norm of 3.9e-6, but it is pure noise. With the default min the step is gated (scale 1).
- The gate is a discontinuity: the scale jumps from 1 to target/min as g crosses min. A capped-gain alternative, min(target/g, max_gain), is documented in the design doc but not implemented.

## Test results

- Targeted runs (tf_gpu, GTX 1080, 6000 MB limit): all new tests pass.
- Golden tests `TestLinearOnlyUnchanged` and `TestDeformUnchanged`: `max |diff| 0 (tolerance 0)` for both, fixtures not regenerated.
- `test_linear_and_deform_normalised` (target 5): applied norm 5 at rtol 1e-5 for rigid, affine and deform. The raw norm and loss terms equal the unnormalised twin's at rtol 1e-6, under op determinism. The independent SGD check, |Δθ| / lr vs target, gave relative errors rigid 7.7e-8, affine 3.6e-8, deform 2.2e-8 to 9.1e-8 across runs.
- `test_gate_below_min` (min 1e9): scale 1, gated 1. Applied norm == raw norm exactly, and variables are bit-identical to a flag-off twin after one rigid and one deform step.
- `test_eps_zero_grads`: min 0 gave raw 0, scale 1.2e13, normed 3.89e-6, all finite. Default min gave scale 1, gated 1.
- `test_weights_after_normalisation`: target 4 with deform:0.25 / rigid:2 gives applied norms 1 and 8 (rtol 1e-5). Raw and encoder-raw norms equal the unweighted twin's. `_eff` == enc_raw × scale × w unclipped, and × clip/normed with clip 1.
- `test_nonfinite_unchanged` (NaN pixel): step skipped; variables, BN stats and optimizer state unchanged.
- `test_amp_normalised_and_overflow`:
  - Under mixed_float16, applied norm is 12 (rtol 1e-3) on the finite steps.
  - With a forced overflow (loss scale 2^40), the inf gradients get scale 1 (not gated), the LSO halves its scale and the weights are unchanged.
- Also passing: `test_off_no_new_keys`, `test_trainer_validation`, `test_window_logs` (normed 12 in every window, `train/grad_norm` stays raw, an off-run logs none of the keys), `test_task_grad_norm_flag` (CLI defaults; -1 / nan / inf rejected for both flags; target 0 = off) and `test_check_mixed_run`.

Two test bugs were found and fixed on the way:

- Deform twin comparisons need op determinism. Without it, the warp gathers give 9e-5 relative noise.
- One test reused the same twin trainers across tasks, so they started from different weights.

## Key equivalence test (`TestTaskGradNormClipEquivalence`)

Setup: 8 three-task steps, the golden setup, under `_OpDeterminism`. Run B is clip 12 with weights 2^14 for every task; every raw norm is > 12, so every step is clipped.

- **A0 (target 12, no weights, optimizer clip off) vs B: bit-identical, max |diff| 0.** Losses, deform terms and every variable's sum and max match. The per-step raw norm × 2^14 equals B's norm.
- **A12 (target 12, no weights, clip 12 kept, the real recipe) vs B: near-equal.** Losses have max relative diff 9.2e-7. Variable sums differ by up to 1.1e-3 relative to (1 + |s|), worst case `encoder/l4/conv/kernel` with sum -3.616 vs -3.621. Per-element max |v| differs by ≤ 2e-4 (stem bias), and applied norms are 12 ± 1 ulp.
  - Cause: when the normalised norm comes out at 12 + 1 ulp, the optimizer's clip applies a factor of 1 - 1 ulp. Adam turns these ulp differences into visible differences only for elements with tiny, cancelling moments.
  - These are not systematic. Test tolerances: losses 1e-5 relative, variables 1e-2.

## Local synthetic three-task run (WSL GTX 1080, 6000 MB limit)

Setup:

- Narrow widths; linear N 4 / deform N 2; Adam 3e-4 constant; clip 12.
- Warm start from a 300-step rigid+affine run (`lin_src`, partial warm start, deform decoder fresh).
- 400 steps, rigid:affine:deform 1:1:2, windows of 25 steps.
- gn_on uses `--task-grad-norm 12`; gn_off is the unweighted control.
- Both runs exited with code 1, as expected: the overfit target was not reached by design. All metrics were finite, with no non-finite steps.

Medians over windows after step 100:

| run | task | raw grad_norm | scale | gated frac | normed | enc _eff |
|---|---|---|---|---|---|---|
| gn_on | rigid | 1.24e4 | 1.14e-3 | 0 | 12 | 0.221 |
| gn_on | affine | 1.67e4 | 9.2e-4 | 0 | 12 | 0.326 |
| gn_on | deform | 0.656 | 19.3 | 0 | 12 | 4.62 |
| gn_off | rigid | 1.25e4 | - | - | - | 0.136 |
| gn_off | affine | 1.46e4 | - | - | - | 0.240 |
| gn_off | deform | 0.792 | - | - | - | 0.284 |

- **No step was gated** (gated frac 0 in every window of every task). That includes the first deform window right after the warm start, while the SVF head was still near zero. That window had raw 0.37, scale 38, enc _eff 0.55 and u_max 0.38 vox. Those early steps mostly grew the head (u_max 0.38 → 2.6 → 4.0 vox by step 75) and were harmless: no folding, all finite.
- **Encoder _eff ratio linear/deform: rigid 0.048, affine 0.071 with the flag on, vs 0.48 / 0.85 unweighted.** This is NOT O(1). The cause: in this narrow synthetic setup the encoder is ~38% of the deform total norm (0.25 / 0.66) but only ~1.7-2.6% of the linear totals (214 / 12350). Equal total norms therefore give deform ~20× the applied encoder norm. This setup also does not reproduce the real-data imbalance: unweighted, the ratio is already ~0.5-0.9 here, vs 250-1500 on the cluster.
- On real level-4 data, every step was clipped under deform:64, which is effectively the same as normalising to 12. That gave ratios of rigid 1.32 / affine 0.72 (LNCC smoke, job 10494447). The cluster A/B should land in that range. If not, the encoder-share mismatch above is the explanation, and per-task encoder-only normalisation would be the follow-up.
- Validation at step 400 (one synthetic group, one seed, not conclusive):
  - gn_on: deform template / pair Dice 0.9345 / 0.8695, rigid / affine center 1.93 / 3.13 mm.
  - gn_off: 0.9264 / 0.8554, 2.31 / 5.54 mm.
  - Identity: 0.8345 / 0.6953.
- `check_mixed_run.py` on gn_on passes everything except `val_png`, which was expected because the run used `--val-png false`. Its detail lines:
  - `enc_grad_rigid ... grad_norm_scale median 0.001108, gated frac 0`
  - `enc_grad_deform ... grad_norm_scale median 18.78, gated frac 0`
  - `enc_grad_ratio INFO median eff ratio linear/deform rigid 0.0543, affine 0.085`

## Python 3.8 check

`bgr_preproc38` (Python 3.8.20) on all 6 changed .py files: py_compile OK and `ast.parse(feature_version=(3, 8))` OK. `bash -n slurm/m7_three_task.sh` OK, and `main.py --help` lists both flags without importing TF.

## Full local suite

`tests/run_tests_wsl.sh`, run once at the end:

- Step 1 (Python 3.8): py_compile OK, ast (3, 8) OK on 51 files, `main.py --help` OK with no TF, all 4 `bash -n` checks OK.
- Step 2: **Ran 327 tests in 7135 s, OK (skipped=4)**, unittest exit code 0. That is the 313 baseline plus 14 new; the 4 skips are the same as the baseline (DALI / dllogger not installed).
- Both golden lines: max |diff| 0.

## Recommended cluster A/B (NOT submitted)

Compare against the deform:64 control `results/m7_three_task_long_w64` (job 10493723, A100: step 12000 rigid 3.84, affine 4.79, rigid_g 3.89, affine_g 6.52 mm; deform template / pair Dice 0.726 / 0.642). Same recipe, same partition (dgx-a100, A100 40 GB, 1 GPU, script defaults 3 CPUs, 64 GB):

```
RUN_NAME=m7_three_task_gradnorm TASK_GRAD_NORM=12 TASK_LOSS_WEIGHTS=none sbatch --job-name=m7gn --time=08:00:00 --mem=64G -p dgx-a100 slurm/m7_three_task.sh
```

Requires a `git pull` on the cluster after the orchestrator pushes. Watch these:

- `enc_grad_ratio` from the checker, and `train_<task>/grad_norm_gated_frac`, which should be 0.
- `train_deform/grad_norm_scale`, expected ~24-400 early (raw 0.03-0.5), then ~10-20 (w64 deform totals 40-75 → raw ~0.6-1.2).
- Deform Dice vs w64 at matching steps (2000 / 4000 / 8000).

By construction the run should track w64 closely: w64 already clipped every step, so the arm mainly confirms the mechanism. Its value is that it removes the dependence on picking 64.
