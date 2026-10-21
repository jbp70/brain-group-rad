# LNCC image-similarity terms for the deformable loss

The change adds a local normalised cross-correlation loss to the deform task, next to the existing template and pairwise soft Dice. It has two terms. The template term compares each warped image with a leave-one-out mean intensity template. The pair term compares image i moved into j's space with image j, on the same sampled pairs and displacements as the pairwise Dice. LNCC is the squared VoxelMorph form computed on true-count local means with a separable ones-kernel `tf.nn.conv3d`, so the model gains no variables. It is evaluated inside a label-derived ROI (foreground dilated by the window radius). All new weights default to 0, and the default graph is rebuilt op-for-op. A determinism-locked golden fixture written on b35bb54 guards bit-identity, and validation always reports LNCC plus identity baselines.

Watch for: train-mode validation now computes and discards LNCC on every val group (confirmed, cost only). Memory at N = 16 is extrapolated, not measured. The local A/B shows LNCC does not close the linear/deform encoder-gradient gap. None of these block.

**Verdict**: APPROVED

## High-level view

The math follows the reference's `fast_nlcc_metric`, with deliberate changes. Local means use the true per-voxel window count, which fixes border normalisation and gives ε = 1e-7 a real meaning in intensity² units. Variances are clamped at 0. The window is per level (5/7/9). The loss is 1 − cc instead of −cc. The squared form bounds cc in [0, 1] and avoids a sqrt, so constant regions give cc = 0 with finite gradients. Unit tests check this, and they also check box sums and LNCC maps against a float64 scipy reference (rtol 1e-5 on box means, rtol 1e-4 / atol 1e-5 on the LNCC map).

The leave-one-out template handles the self-similarity floor of the inclusive mean (≈ 1/N for unrelated images). `test_leave_one_out` measures it: < 0.05 with LOO vs > 0.15 inclusive at N = 4. Gradients still flow through the template, and a test checks this against a stop-gradient variant. Collapse to a flat region maximises the loss, so it is not a degenerate optimum. Per-level SVF centring rules out a global group shrink.

The pair term reuses the work of the pairwise Dice. `deform_loss` calls `pairwise_displacement` once when λ_ncc_pair > 0 and passes `disp` into `pair_dice_loss`, which returns its moved/target one-hots for the ROI. There is no extra integration or composition, and a test checks that shared and recomputed values are equal. Out-of-FOV zero fill reads as background because the ROI uses Σ_{c≥1} onehot instead of 1 − onehot[...,0]. Noise outside the dilated foreground provably does not change the loss.

Bit-identity of the default path holds by construction and is enforced. `template_dice_loss`'s template is now kept instead of discarded, the template-Dice factor is applied only when λ ≠ 1, and every LNCC op sits behind static Python `> 0` checks. `TestDeformUnchanged` runs 8 three-task steps under `enable_op_determinism`, because GPU warp-gather gradients were otherwise nondeterministic (diff 100-300). It compares losses, deform terms and every variable at tolerance 0. Per the report it passed at 0 before and after the change and inside the full-suite process. `TestLinearOnlyUnchanged` also stays at 0.

Wiring matches the Dice metrics. `DEFORM_NCC_KEYS` are logged only when present. Validation adds `ncc_template` / `ncc_pair` per item and cached `_identity` baselines. The CLI flags default to 1/0/0/per-level and are validated in both `args.py` and `Trainer`. A deform run with all four similarity weights at 0 is rejected.

The recorded evidence is complete. The full suite ran 313 tests OK (4 skipped, previously 295), with py3.8 stage 1 OK. The `FAIL` lines in the log come from the negative `check_mixed_run` tests. The other recorded artefacts are memory.json, grad_probe.json and the A/B/C overfit runs. A matches milestone 6. B keeps Dice within 0.007 of A and gains +0.04/+0.07 LNCC. C (LNCC-only) beats identity by +0.10 / +0.16 Dice with no folding.

<details>
<summary>Issues (4)</summary>

1. **Discarded train-mode LNCC** — `_val_deform_errors(vs, groups, True)` passes `images`, so every validation computes an extra image warp, the pair warps and the box filters for the train-mode pass, then keeps only template/pair Dice. Pass `images=None` when `train_mode` is True (non-blocking).
2. **N = 16 memory unmeasured** — locally there was no measurable slope (N ≤ 4); the +0.5 GB at default widths N = 16 is an estimate. Confirm `gpu_peak_mb` in the LNCC smoke before the 14 h run, especially on the 40 GB A100.
3. **Gradient imbalance unchanged** — LNCC raises the deform encoder gradient only ≈ 2-3× (synthetic), still orders of magnitude below the clipped linear steps. Keep `--task-loss-weights` on the table for the real-data A/B instead of expecting LNCC to fix it.
4. **Golden provenance by assertion** — the fixture is untracked, and "written on b35bb54" is backed only by the report and docstring. Commit the fixture in the same commit as the code so a future regeneration is visible in history.

</details>

<details><summary>Details</summary>

## LNCC numerics vs the reference

The reference normalises 9³ box sums by k³ and uses ε = 1e-7 on those sums. That is ≈ 2e-13 in means units, effectively no guard, and border windows are under-normalised. This change divides by `window_count` (an analytic numpy constant of shape and window) and clamps `var_a`, `var_b` at 0. `cov` is left unclamped (the square removes its sign). With [0, 1] intensities, float32 cancellation in var and cov is ~1e-7, so the clamped case gives cc ≈ cov²/ε ≈ 1e-7, not a spike. The signed variant exists for tests only. All quantities go through `_float` (float16 → float32), and there are no Keras layers, so AMP cannot downcast it.

## Template term and ROI

```
W_i = warp(img_i, u_i)          T_-i = (ΣW - W_i)/(N-1)      (gradients through T)
ROI = box_sum(fg(mean_i warp(onehot_i, u_i))) > 1e-3         (stop_gradient, shared over i)
L   = mean_i (1 - masked_mean(lncc(W_i, T_-i), ROI))
```

The ROI comes from the inclusive one-hot template, so image i's own labels shape it, but it is a stop-gradient region choice, not a similarity target. During training the DALI noise in the background lies inside the dilated rim. There the LOO template averages noise down, so the rim contributes low cc at identity and after alignment alike. That dilutes the absolute value but not the gradient direction.

## Validation cost

`_deform_errors(..., images=...)` is called for the eval pass, the identity baseline (cached), and the train-mode pass. Only the first two use the LNCC values. The train-mode result goes through `M.summarize`, and only `template_dice` / `pair_dice` are copied out (run.py ~line 832). The LNCC work on that pass is wasted. It is small at level 4 but grows with val group count and level.

## Test coverage

Covered: scipy reference match at k = 3/5 with borders, self = 1, affine intensity = 1, −I gives squared 1 / signed −1, independent noise ~0, flat/zero regions finite with finite gradients through `diffeo.warp`, ROI invariance and border foreground, LOO vs inclusive, template gradient flow, one-voxel pair shift, float16 → float32, bad windows, λ = 0 bit-identity and key set, missing `images` only when on, total = weighted sum, shared-disp equality, LNCC-only config, trace counts and `n_vars_updated` unchanged, three-task logging, validation identity = baseline (places 10), CLI defaults per level and round trip, and a main.py dry run with 1/1.

Not tested: real-T1 intensities (no local level-4 export), and peak memory at N ≥ 8 with default widths.

</details>

<details>
<summary>File map</summary>

- `runtime/losses.py`: LNCC primitives (`check_ncc_window`, `box_sum`, `window_count`, `local_ncc`, `foreground`, `ncc_roi`, `masked_mean`, `template_ncc_loss`, `pair_ncc_loss`); `pair_dice_loss(disp=, return_moved=)`; `deform_loss` new weights.
- `runtime/run.py`: TrainConfig fields, validation of weights/window, step wiring, `DEFORM_NCC_KEYS` logging, validation LNCC + identity baselines.
- `runtime/args.py`: four CLI flags, validation, TrainConfig pass-through.
- `data_loading/params.py`: `NCC_WINDOW_DEFAULT`, `ncc_window_for_level`.
- `runtime/metrics.py`: docstring only.
- `slurm/m7_three_task.sh`: comment with the LNCC submit line.
- `tests/gpu/measure_deform_memory.py`: LNCC weight flags.
- `tests/test_runtime_losses_metrics.py`, `tests/test_trainer.py`, `tests/test_runtime_infra.py`, `tests/test_main.py`, `tests/fixtures/lncc_deform_golden.json`: new tests and golden fixture.
- `.agents/tasks/network-design/design.md`: v3.9 LNCC section.

Full diff: `git diff` against HEAD (b35bb54) plus the untracked fixture and `.agents/tasks/lncc-loss/`.

</details>
