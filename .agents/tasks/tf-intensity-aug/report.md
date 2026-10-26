# Report: TF intensity augmentation, level-1 memory recipe, DALI term, pair recompute, chunked validation

Not committed (orchestrator owns git / cluster). Verification details: `verification.md` (same folder).

## Files changed

- New: `data_loading/tf_intensity_aug.py` (TF chain), `data_loading/intensity_ref.py` (TF-free numpy
  reference), `tests/test_tf_intensity_aug.py`, `tests/test_pair_recompute.py`, `tests/test_val_chunk.py`.
- `data_loading/params.py` (INTENSITY_AUG_DEVICES, INTENSITY_AUG_SEED_OFFSET 7919, AUG_COLUMNS,
  N_AUG_UNIFORMS), `data_loading/postprocess.py` (`intensity_aug=`, `aug_generator`, `num_pairs_static`),
  `runtime/run.py` (TrainConfig `intensity_aug_tf`, `pair_recompute`, `val_image_chunk`; checkpoint keys
  `postprocess_<task>_aug`; deform step wiring; `_deform_errors_chunked`), `runtime/losses.py`
  (`pair_terms_recompute`, `deform_loss(pair_recompute=, n_pairs=)`), `runtime/args.py`
  (`--intensity-aug-device`, `--pair-recompute`, `--val-image-chunk`, `--bn-recal-validate`), `main.py`
  (DALI config by device, desc, resume guard `_check_intensity_aug_device`, `recalibrate_bn(val=)`).
- `models/memory_model.py` (`dali_memory`, `predict(dali_gib=)`, budgets), `scripts/memory_model.py`
  (`--dali`, `--dali-groups`).
- `tests/gpu/verify_dali_pipeline.py` (check m), `tests/gpu/measure_level_memory.py` (`--pair-recompute`).
- `slurm/m9_level1_cascade.sh` (INTENSITY_AUG_DEVICE, TF_MEM_LIMIT_MB, PAIR_RECOMPUTE, VAL_IMAGE_CHUNK,
  VAL_PAIR_CHUNK, BN_RECAL_VALIDATE), `slurm/m8_level2_cascade.sh` (comment only).
- Tests extended: `test_postprocess`, `test_trainer`, `test_main`, `test_memory_model`,
  `test_slurm_scripts`, `test_verify_dali_level`, `test_measure_memory`. `plan.md` has addenda A / B.

## Design

- Flag `--intensity-aug-device dali|tf` (default dali: no TF op built, checkpoint keys unchanged, goldens
  max |diff| 0). With tf, DALI gets `IntensityAugConfig.off()` and `TrainConfig.intensity_aug_tf`
  reaches both post-processes. Not allowed with `--overfit-one-group`. Resume with a different device
  exits with a clear message.
- Placement: in `LinearPostprocess` / `DeformPostprocess` right after the float32 cast, before the flips
  (commute), images only. Also runs in the BN-recalibration input path, as DALI did. Validation never
  augments.
- RNG: separate `aug_generator` = `Generator.from_seed(pp seed + 7919)`, so the flip / pair stream is
  identical to a dali run. Checkpointed as `postprocess_<task>_aug`. Per call, `uniform([N, 16])` plus
  one stateless noise seed per image (fixed draw pattern).
- Semantics mirror `dali_loader.intensity_augment`: noise -> brightness -> contrast (clamped to the
  pre-contrast min/max) -> low-res -> gamma (renormalised to the input mean / population std) -> clamp
  [0, 1]. The low-res factor is ONE isotropic f per image: the DALI code multiplies a scalar uniform by
  the shape (the task text said "per axis", the code wins).
- dtype: float32 always, explicit (the mixed_float16 policy has no effect; test T13).
- Low-res method: a fused separable NN-down + linear-up per axis (two gathers + lerp, static output
  shape). m = max(1, floor(f n + 0.5)) (half up, like DALI's rounding); pixel center 0.5, edge clamp,
  no antialias. NN source `src(l) = min(floor(start + (l + 0.5) s), n - 1)`; `LOWRES_ROI` = 'subpixel'
  (s = 1/f, DALI subpixel_scale), alternatives 'integer' (s = n/m) and 'subpixel_centered'. Which one
  DALI uses is measured by check m1 on the cluster (fix: flip the constant). Remaining known
  difference: index arithmetic in float64 here vs DALI's float32 (only exact-integer boundaries).
  Locally: TF vs the independent two-stage numpy resize, max |diff| 1.7e-7 (all modes, f 0.5-1.0,
  odd/even sizes).
- Memory: `tf.map_fn(parallel_iterations=1)` over images, no variables / persistent buffers.
  Measured transient (T14, GTX 1080): level 2 N 8: 60.9 MiB (group 26.2 MiB); level 1 N 2: 157.5 MiB
  (group 52.5 MiB) = ~2.3-3 group tensors. Expected at level 1: linear N 8 ~0.5-0.6 GiB, deform N 4
  ~0.3 GiB, before the model forward (outside the tape), so the step peak should not change; worst
  case +0.6 GiB if the allocator overlaps it.

## Equivalence tests (local)

- Per op TF vs numpy (forced parameters): noise, brightness, contrast, low-res, gamma (mean/std kept to
  rtol 1e-4), clamp, disabled = identity; full chain per image (8 images) atol 1e-5.
- Parameters (20000 draws): every apply frequency within 4 binomial sigma, low side ~0.5, factors in
  range. Chain statistics vs numpy (400 vs 400, KS on mean / std / p1 / p50 / p99 / frac 0 / frac 1):
  p = 1, 0.111, 0.468, 0.998, 0.131, 0.994, 0.367.
- Determinism: same seed identical, checkpoint restore continues the stream bit-exactly; Trainer
  resume restores the aug generator states.
- DALI vs TF cannot run locally (no DALI): new check m in `verify_dali_pipeline.py` (below).

## Expected memory effect at level 1 and the TF limit

- DALI: ~14 GB -> ~4.5 GB (job 10517362), ~9.5 GB freed; TF transient +~0.5 GiB before the forward.
- TF_MEM_LIMIT_MB (default 40000, `--gpu-memory-limit-mb`): RTX 8000 46080 MiB usable - CUDA context
  ~500 - DALI without aug ~4600 - >= 1024 headroom outside TF (cuBLAS / cuDNN handles, DALI pool
  growth) = 39956 -> 40000. A100 40 GB: 40536 - 500 - 4600 - 1024 = 34412 -> TF_MEM_LIMIT_MB=34000.
  INTENSITY_AUG_DEVICE=dali with a limit > 30000 exits 2. TF_MEM_LIMIT_MB=0 = memory growth.
- Memory model: `dali_memory` = 109.2 B (aug in DALI) / 35.1 B (not) per image voxel, calibrated on the
  20-image level-1 configuration; level 2 16/16/8 = 3.50 GiB, level 4 32/32/16 = 0.875 GiB (aug in
  DALI). `predict(dali_gib=)` subtracts it from the 44 / 36 GiB budgets; default 0 keeps the fit and
  the hold-out gate. CLI default `--dali tf` (4.50 GiB at level 1), `--dali none` = old numbers.

## Pair recompute (addendum A)

- Local equality: loss terms to 1e-5 and gradients max relative diff 2.2e-6 (plain, lean sampler,
  lean losses, eager and tf.function); Trainer step loss equal, weights after two Adam steps within
  1e-4 (also with --gradient-checkpointing full + lean sampler).
- Serialisation: block p's input carries `s * z` (s = sum of block p-1's result, z a data-dependent
  zero). The first version used `0 * s`, which Grappler constant-folds away; the full suite caught it
  (4-pair growth 221 MiB vs 160 allowed) and the fix was re-measured.
- Local GPU (64x80x64, N 4, 19 classes, pair LNCC), after the fix: default sampler 1 / 4 pairs
  1740 / 2278 MiB vs recompute 1889 / 1964 MiB; lean sampler 1040 / 1438 MiB vs recompute 970 / 968 MiB
  (flat in the pair count with the lean sampler used at level 1).
- Scaled to level 1 (x21 voxels): the default path costs ~3.3 GiB per extra pair (3 extra = ~9.8 GiB,
  consistent with the job-10517366 OOM). With recompute the 4-pair step is expected at about the
  1-pair peak plus at most ~3 GiB overhead: 35.1 GB (job 10517503, 1 pair) -> ~35-38 GiB with 4
  pairs, inside the 40000 MB limit but with little room. Unverified until the smoke; fallbacks:
  LEAN_LOSSES=true, then MIN_PAIRS=1.

## Level-1 validation memory (addendum B)

- `--val-image-chunk 1 --val-pair-chunk 1`: every validation post-processing tensor holds one image /
  one pair (one-hot ~0.52 GB, field ~0.08 GB at level 1) instead of [4, ..., 19] (2.1 GB) one-hots,
  their warps and sampler transients. The eval forward of the group (groupwise model, cannot be split)
  is unchanged and is now the largest validation allocation. Linear validation unchanged (group
  forward only).
- Chunked == unchunked metrics to 1e-4 (incl. identity baselines and LNCC; frozen baselines use the
  same function).
- Local GPU (level 2 TINY, N 2, LNCC, lean sampler, fixed 6000 MB): train step peak 4703 MiB;
  validation peak unchunked 1345 MiB -> chunk 1 / pair chunk 1 839 MiB (-38 %); train -> validate ->
  train ran in both modes and again unchunked after the chunked pass.
- `--bn-recal-validate false` skips the ~20 min of bnrecal_pre / bnrecal_post passes per fresh run.
- Time: measured 600 s per validation pass unchunked; chunking adds launches, not FLOPs; expected
  600-800 s per pass. Long run on the RTX 8000: 8000 x 9.1 s = 20.2 h + 16 passes x 0.17-0.22 h =
  2.7-3.6 h + recal / startup ~1 h = ~24-25 h; x1.3 = ~32 h -> keep --time 48:00:00 (36 h also fits).

## Cluster checks to run (orchestrator)

1. `git pull --ff-only`, then DALI equivalence (rtx8000, ~20-40 min):
   `python tests/gpu/verify_dali_pipeline.py --data <npy_v2> --scratch <proj>/scratch/verify_m --out <proj>/scratch/verify_m/verify_m_l4.json --checks m --level 4`
   and the same with `--level 1` (`--out .../verify_m_l1.json`). PASS needed for m1 (else flip
   `LOWRES_ROI` to the best mode reported) and m2; m3 is INFO (DALI MB per pipeline with / without aug).
2. Level-1 smoke with the script defaults (INTENSITY_AUG_DEVICE=tf, TF_MEM_LIMIT_MB=40000,
   PAIR_RECOMPUTE=true, VAL_IMAGE_CHUNK=1, VAL_PAIR_CHUNK=1, BN_RECAL_VALIDATE=false):
   `RUN_NAME=m9_l1_smoke STEPS=200 VAL_EVERY=100 WARMUP=50 sbatch --job-name=m9smoke --time=04:00:00 --mem=64G slurm/m9_level1_cascade.sh`
   Look at: the first deform step with 4 pairs, the deform step right after the step-100 validation,
   gpu_peak, validation time per pass.

## Not verified locally

DALI-vs-TF statistics and the low-res ROI convention (check m), the real level-1 memory (DALI term,
TF limit headroom, 4 pairs with recompute, the validation peak and the post-validation train step),
the level-1 validation time.

## Test results

Final full suite (`suite3.log`): 558 tests OK, 4 skips (baseline 520), exit 0; both goldens max
|diff| 0. Two earlier suite runs found the constant-folded pair-recompute dependency (fixed) and an
in-suite OOM of `TestLevel1CascadeErrors`' eager level-1 build pass at N 2 under the 6000 MB cap
(the test now builds with N 1; it only checks the error routing). Details: `verification.md`.
