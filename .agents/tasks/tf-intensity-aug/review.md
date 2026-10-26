# Intensity augmentation moved from DALI into the TF post-process (plus level-1 memory recipe)

The change adds `--intensity-aug-device dali|tf`. With `tf`, the DALI pipelines get `IntensityAugConfig.off()` and the same noise -> brightness -> contrast -> low-res -> gamma -> clamp chain runs in `LinearPostprocess` / `DeformPostprocess` right after the float32 cast. It uses its own checkpointed generator, so the flip and pair stream is unchanged. The goal is to free the ~9.5 GB of DALI device memory measured at level 1 (job 10517362). The same diff also adds the level-1 recipe in `slurm/m9_level1_cascade.sh` (fixed TF memory limit, TF augmentation, `--pair-recompute`, chunked deform validation, `--bn-recal-validate false`) and a DALI term in the memory model. The default path builds no new op and no new generator. Both goldens report max |diff| 0 in the targeted run and in the final full suite (558 OK, 4 skips, `suite3.log`).

Watch for: the A100 rule in the level-1 script (smoke peak <= 36 GiB) contradicts the 34000 MB TF limit it prescribes for A100 (confirmed). The 40000 MB RTX limit leaves 980 MiB by the script's own arithmetic, not the stated >= 1024 (confirmed, though 40000 already ran in job 10517503). DALI-vs-TF equivalence (check m) has not run yet and must gate the level-1 smoke.

**Verdict**: APPROVED

## High-level view

The TF chain in `data_loading/tf_intensity_aug.py` mirrors `dali_loader.intensity_augment` op by op. Apply coins are `u < prob`, factors use the 0.5 low/high side rule, contrast clamps to the pre-contrast min/max, gamma renormalises to the input mean and population std, and the final clamp runs whenever the config is enabled. The placement is equivalent: DALI augmented after the warp and before any flip, and the TF version runs on the DALI output before the flips. A fixed draw pattern of 16 uniforms plus one stateless noise seed per image makes the stream independent of the enable flags. The generator is created only when TF augmentation is on, so `dali` checkpoints keep their keys, and a device switch on resume is rejected with a clear error.

Low-res is a separable, static-shape NN-down / linear-up. The DALI NN source convention is still open between three ROI modes and is documented as such. Locally it matches an independent two-stage numpy resize to 1.7e-7. The real DALI comparison is check m1 in `verify_dali_pipeline.py`, still to run on the cluster. It is the only test that pins the TF code to DALI rather than to a reference that shares its formulas.

The augmentation maps one image at a time with no variables. The measured transient is about 2.3-3 group tensors (157.5 MiB at level 1 N 2), which extrapolates to ~0.5-0.6 GiB at linear N 8, before the forward and outside the tape.

The level-1 script defaults to `tf` and `TF_MEM_LIMIT_MB=40000`, rejects `dali` with a limit above 30000, and leaves the level-4/2 scripts' behaviour unchanged. The headroom arithmetic and the A100 guidance have the inconsistencies listed above.

The memory model gets a two-point DALI term (109.2 / 35.1 B per image voxel) that reduces the 44 / 36 GiB budgets. The library default `dali_gib=0` keeps the fit and the hold-out gate unchanged, and the CLI defaults to `--dali tf`.

<details>
<summary>Issues (6)</summary>

1. **A100 rule vs TF limit**: the script still says to use dgx-a100 if the smoke `gpu_peak` <= 36 GiB, but the A100 line sets `TF_MEM_LIMIT_MB=34000` (~33.2 GiB). A 34-36 GiB smoke peak would pass the rule and OOM on A100. Change the rule to a peak at or below the A100 TF limit (~33 GiB), or document why the TF limit can be raised there (confirmed).
2. **RTX 8000 headroom below the stated 1 GiB**: 46080 - 500 - 4600 - 40000 = 980 MiB. The comment claims >= 1024 and then rounds 39956 up. Either round down (e.g. 39900) or state that 40000 was validated empirically (job 10517503 ran 100 steps with DALI aug off at this limit) (confirmed, low risk).
3. **m8 comment overstates the level-2 saving**: "~3.5 GB less DALI memory at level 2" is the total DALI estimate with aug (16/16/8). The saving by the model's own coefficients is 3.50 - 1.13 ≈ 2.4 GiB. Fix the number (confirmed).
4. **Check m3 likely under-reports DALI memory**: DALI's device pool is not returned on `del pipe`. m3 runs after m2 has built many pipelines, and `aug_off` is measured after `aug_on`, so both deltas will be near zero. Run m3 first in a fresh process (or one process per config) before using it for calibration (likely).
5. **DALI equivalence not yet measured**: m1 (ROI convention, mean |diff| <= 5e-3) and m2 (KS per op and full chain) are the only DALI-anchored evidence. Run `--checks m` at levels 4 and 1 before the level-1 smoke, and flip `LOWRES_ROI` if m1 picks another mode (confirmed pending; process gate, not a code defect).
6. **Resume guard over-strict when augmentation is off**: a run saved with `--intensity-aug false --intensity-aug-device tf` refuses to resume with `--intensity-aug false` (default device dali), although neither builds TF aug generators. Compare only the effective device (`A.intensity_aug_device`) (confirmed, NIT).

</details>

<details>
<summary>Details</summary>

### Contract check against the DALI chain

`dali_loader.intensity_augment` runs after `warp_linear` (linear) and after the label/image warp (deform), on the warped volume, before any flip (flips live in the post-process). The TF chain runs on the DALI output right after the float32 cast and before `apply_flip`, on images only. Labels, one-hot and matrices are untouched. Per-op semantics match: noise std `U(0, max)` multiplexed with prob, brightness factor, contrast with pre-contrast min/max bounds, low-res with ONE isotropic factor (DALI multiplies a scalar uniform by the shape vector, which the docstring records against the task text's "per axis"), gamma with population std, and final clamp. Non-applied noise / contrast / low-res are exact identities in TF, while DALI does a near-identity there (resize to the full size, clamp at factor 1). That difference is float-rounding-level only. Gamma always runs, as in DALI. Everything is explicit float32, and T13 shows the mixed_float16 policy does not change the output.

### RNG, reproducibility and resume

`aug_generator = Generator.from_seed(pp_seed + 7919)` keeps `pp.generator` (flips, pairs) consuming the same stream as a `dali` run. This is the documented reading of "seeded from the postprocess generator", chosen because `split()` would advance the original stream. The generators are checkpointed as `postprocess_<task>_aug` only when present. T11/T12 and the Trainer resume test cover same-seed determinism and bit-exact continuation. On a fresh cascade run the BN-recalibration batches also consume aug draws, consistently between runs.

The resume guard raises on any mismatch in the raw `intensity_aug_device` string. When `--intensity-aug false` both runs are effectively "off", yet a saved `tf` vs current `dali` still exits (issue 6). Fail-closed, so harmless, but it can block a legitimate resume.

### Low-res resample and what is actually measured

```
per axis n:  m = max(1, floor(f n + 0.5))
             c = (o + 0.5) m / n - 0.5      -> lerp between low-res l0, l1 (edge clamp)
             src(l) = floor(start + (l + 0.5) s), s = 1/f ('subpixel') | n/m ('integer')
```

T4 compares this fused form to `intensity_ref.lowres_np`, which really is two-stage (`np.ix_` NN gather then per-axis linear up). The NN source index formula and the upsample coordinate rule are the same equations in both implementations, though. T4 proves the fusion is exact, not that DALI uses these conventions. That evidence comes only from check m1 (three ROI modes, PASS if the active mode is best and <= 5e-3 mean |diff|), which cannot run locally (issue 5). The residual differences (fractional footprint m/f vs n, float32 vs float64 index rounding) are documented in the module docstring.

### GPU check m

m1 reuses the exact DALI resize calls with constant sizes in DHW layout order, matching how `volume_shape` is passed in the loader. m2 compares DALI-augmented batches against TF-augmented copies of DALI's un-augmented output, per op at prob 1 and for the full chain, with KS on per-image features binned to 1e-4 around the un-augmented value. Untouched images then form atoms rather than spurious differences. m3 measures process memory around building one pipeline with and without aug. DALI's device pool keeps its memory after `del pipe`, and m3 runs after m2 and in aug_on -> aug_off order, so it will likely report near-zero deltas (issue 4). It is INFO only and does not affect PASS, but the report names it as calibration input.

### Memory: transient and the level-1 budget

`tf.map_fn(parallel_iterations=1)` with a static output signature keeps one image's temporaries alive. T14's measured 60.9 MiB (level 2 N 8, group 26.2 MiB) and 157.5 MiB (level 1 N 2, group 52.5 MiB) put the transient at ~2.3-3 group tensors (unstacked input plus stacked output plus per-image temporaries). That supports the ~0.5-0.6 GiB estimate for linear N 8 at level 1. It runs before the model forward, so it should not stack on the backward peak.

The script arithmetic is RTX 8000 46080 - 500 - 4600 - 1024 = 39956, then rounded UP to 40000. That leaves 980 MiB outside TF, contradicting the ">= 1024 headroom" in the same comment (issue 2). Job 10517503 (DALI aug off, 40000 limit) ran 100 steps without handle failures, so the practical risk is small. The A100 line uses 34000 (headroom 1436). The A100 rule a few lines above still allows dgx-a100 for a smoke peak up to 36 GiB, which exceeds the 34000 MB (~33.2 GiB) TF region (issue 1).

### Memory-model DALI term

`dali_memory` scales linearly in images x voxels from two points of one job (14.0 / 4.5 GiB at level 1, 20 images, prefetch 1). The docstring states this is an extrapolation assumption for other levels and group sizes. `predict` and `feasible` subtract `dali_gib` from both the peak and the validation-chunk budgets. The default 0 keeps the calibration records, fit and gate unchanged, and the tests cover the level 1/2/4 values, budget flips and the `--dali none` equivalence. The m8 comment converts this into "~3.5 GB less at level 2", but 3.5 GiB is the total with aug, and the saving is ~2.4 GiB (issue 3).

### Addenda bundled in the same diff

`--pair-recompute` (losses `pair_terms_recompute`, one pair per `tf.recompute_grad` block chained through pass-through u / u_inv) and `--val-image-chunk` / `--bn-recal-validate` belong to this workflow by Jay's request. They are opt-in, gated by trace-time Python switches, and locally shown equal to the default paths (loss 1e-5, gradients 2.4e-6 relative; chunked metrics 1e-4). The static pair count comes from `num_pairs_static` (float32 product, half-to-even like `tf.round`), and `tf.ensure_shape` catches a mismatch at run time. The Grappler constant-folding of the original `0 * s` dependency was caught by the suite's memory test and fixed with a data-dependent zero. The 4-pair level-1 step under recompute is still unmeasured on real hardware; the report lists LEAN_LOSSES / MIN_PAIRS=1 as fallbacks.

### Test coverage

Covered: per-op and full-chain TF vs numpy, parameter frequencies (20000 draws, 4 sigma), chain-level KS vs numpy (400 vs 400), determinism and checkpoint continuation, the AMP policy, transient memory, post-process default-has-no-op, flip/pair stream unchanged, labels untouched, checkpoint keys, Trainer resume, CLI plumbing and the overfit parse error, slurm env validation, memory-model DALI term. Python 3.8 syntax was checked with `ast.parse(feature_version=(3, 8))` on all 22 changed files, and no 3.9+ stdlib usage appears in the diff.

Not tested: DALI vs TF statistics and the ROI convention (check m, cluster only), the level-1 process memory with TF aug and the 40000 limit with 4 pairs under recompute, and resume of a real DALI-backed `tf` run (only the Trainer-level resume is exercised).

</details>

<details>
<summary>File map</summary>

- `data_loading/tf_intensity_aug.py` (new): TF chain, fused low-res, `augment_group` via map_fn.
- `data_loading/intensity_ref.py` (new): numpy reference, two-stage low-res.
- `data_loading/params.py`: device list, seed offset, uniform column map.
- `data_loading/postprocess.py`: `intensity_aug=` / `aug_generator`, `num_pairs_static`.
- `runtime/run.py`: `TrainConfig.intensity_aug_tf`, `pair_recompute`, `val_image_chunk`; aug checkpoint keys; chunked deform validation.
- `runtime/losses.py`: `pair_terms_recompute`, `deform_loss(pair_recompute=, n_pairs=)`.
- `runtime/args.py`, `main.py`: new flags, DALI config by device, resume guard, `bn_recal_validate`.
- `models/memory_model.py`, `scripts/memory_model.py`: DALI term, `--dali` / `--dali-groups`.
- `slurm/m9_level1_cascade.sh`: new env overrides and recipe; `slurm/m8_level2_cascade.sh`: comment only.
- `tests/gpu/verify_dali_pipeline.py`: check m; `tests/gpu/measure_level_memory.py`: `--pair-recompute`.
- Tests: new `test_tf_intensity_aug`, `test_pair_recompute`, `test_val_chunk`; extended postprocess, trainer, main, memory model, slurm, verify level, measure memory.

Full diff: `git diff` plus the untracked files above.

</details>
