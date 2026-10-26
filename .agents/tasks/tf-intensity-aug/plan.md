# Implementation Plan: TF-side intensity augmentation, level-1 TF memory limit, DALI memory term

Goal: with `--intensity-aug-device tf` the DALI pipelines run without intensity augmentation
(`IntensityAugConfig.off()`) and the SAME augmentation runs in TF inside the post-process. This
frees ~9.5 GB of DALI device memory at level 1 (measured, job 10517362: DALI ~14 GB with aug vs
~4.5 GB without). Default stays `dali`, so every existing run and both goldens are bit-identical.

Ground rules for the implementer
- Work in `d:\Research\Brain_Registration_Projects\brain-group-rad` (no worktree). Do NOT commit or
  push, do NOT ssh / sbatch (a cluster job runs from 41491dd; the orchestrator owns git + cluster).
- Python 3.8 syntax (no PEP 585/604 at runtime, no 3.9+ stdlib), TF 2.10, numpy 1.21.1, scipy
  1.4.1, DALI 1.18. `data_loading/params.py`, `data_loading/dali_loader.py` (import), `runtime/args.py`,
  `models/memory_model.py`, `scripts/memory_model.py` must stay TF-free.
- Every TF test module imports `tests.tf_setup` first. unittest only.
- Local test command (tf_gpu env; quote exactly like this from PowerShell, or activate the
  `shell-and-remote-jobs` skill):
  `wsl -d Ubuntu bash -lc "source ~/miniconda3/etc/profile.d/conda.sh; conda activate tf_gpu; cd /mnt/d/Research/Brain_Registration_Projects/brain-group-rad; export TF_CPP_MIN_LOG_LEVEL=3 TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 BGR_TF_GPU_LIMIT_MB=6000 TF_CUDNN_USE_AUTOTUNE=0; python -m unittest <modules> -v"`
  TF-free tests: same with `conda activate bgr_preproc38`.
- Goldens: `tests.test_trainer.TestLinearOnlyUnchanged` and `tests.test_trainer.TestDeformUnchanged`
  must print `[golden] max |diff| 0` / `[golden deform] max |diff| 0` and pass (tolerance 0).

## Design decisions (one line of rationale each)

D1. Option: `--intensity-aug-device {dali,tf}` (default `dali`) in `runtime/args.py`, next to
    `--intensity-aug`. `--intensity-aug false` still turns augmentation off everywhere. With `tf`:
    `main._build_dali_data` gives the data module `IntensityAugConfig.off()`, and
    `train_config_from_args` sets `TrainConfig.intensity_aug_tf = IntensityAugConfig()`. In every
    other case `intensity_aug_tf = None`, so no TF op is built. `--intensity-aug-device tf` together
    with `--overfit-one-group` is a parse error (`p.error`): the overfit path has no augmentation at
    all, and a silent no-op flag would mislead.

D2. New TF module `data_loading/tf_intensity_aug.py`, plus a TF-free numpy reference
    `data_loading/intensity_ref.py`. The reference is used by the unit tests and by the GPU check,
    and it is a second, independent implementation. In particular its low-res is a real two-stage
    resize, not the fused gather.

D3. Placement: inside `LinearPostprocess.__call__` and `DeformPostprocess.__call__`, right after
    the float32 cast of `images` and BEFORE flip sampling / `apply_flip`. Intensity ops commute with
    flips; the per-image statistics (mean/std/min/max) are flip-invariant. It touches images only,
    never labels, one-hot or matrices. Because the augmentation is part of the post-process, it
    also runs in the BN-recalibration input path (`Trainer._recal_input` calls `pp(...)`), as the
    DALI augmentation did. Validation (`runtime/fixed_data.py`) never calls the post-process, so it
    stays unaugmented.

D4. RNG: a SEPARATE `tf.random.Generator.from_seed(seed + P.INTENSITY_AUG_SEED_OFFSET)` per
    post-process (`self.aug_generator`, offset 7919; `seed` = the post-process seed, i.e.
    `cfg.seed + POSTPROCESS_SEED_OFFSET[task]`). It is created only when TF augmentation is on, so
    the default path is unchanged and `pp.generator` (flips, pairs) consumes exactly the same
    stream in `dali` and `tf` mode. That keeps A/B comparisons controlled.
    `generator.split()` would advance `pp.generator`, so a seed derived from the post-process seed
    is used instead. This is the documented reading of "seeded from the postprocess generator".
    Checkpointing: `Trainer.checkpoint_variables()` adds `"postprocess_<task>_aug": pp.aug_generator`
    only for post-processes that have one. A `dali` run's checkpoint keys are therefore unchanged,
    and resume restores the aug stream like the existing RNGs. Switching the device on resume is
    rejected with a clear SystemExit in `main.py` (see item 5); otherwise TF's
    `assert_existing_objects_matched` would raise an obscure error.

D5. Draw pattern (fixed, independent of probabilities and enable flags, so the stream is stable):
    per call, `u = aug_generator.uniform([N, 16], dtype=tf.float32)` and
    `seeds = tf.transpose(aug_generator.make_seeds(N))` ([N, 2] int64, one stateless noise seed per
    image). Each column has the same meaning as the DALI `_multiplex` / `_intensity_factor`
    semantics in `dali_loader.py` (coin "apply" = `u < prob`; `_intensity_factor` = with prob 0.5
    the low side U(low, 1), else the high side U(1, high)):

    | cols | op | meaning |
    |---|---|---|
    | 0, 1 | noise | apply = u0 < noise_prob; std = u1 * noise_std_max; img + std * stateless_normal(shape, seed_i) |
    | 2-5 | brightness | apply = u2 < p; low side = u3 < 0.5; f = lo + u4 (1 - lo) or 1 + u5 (hi - 1); img * f |
    | 6-9 | contrast | same factor rule; clamp((img - mean) f + mean, min(img), max(img)), with min/max of the PRE-contrast image |
    | 10, 11 | low_res | apply = u10 < p; f = min_factor + u11 (1 - min_factor); ONE isotropic factor per image |
    | 12-15 | gamma | factor rule with gamma_range; when not applied g = 1 and the full gamma + mean/std renormalisation still runs, as in DALI |
    | - | clamp | final clamp to [0, 1] whenever cfg.enabled |

    The low-res factor is isotropic: in `dali_loader.intensity_augment`, `fn.random.uniform` has no
    shape (a scalar) and is multiplied by the shape vector. The task text said "per axis"; the code
    wins, and this is recorded in the module docstring.
    Ops with `cfg.<op> = False` are skipped, exactly like DALI, which does not build them. Their
    columns are still drawn. std is the population std (ddof 0, as `fn.reductions.std_dev`).
    Noise, brightness, contrast and low-res are skipped per image with `tf.cond` when not applied,
    so non-applied ops are bit-exact identities. Gamma always runs, mirroring DALI (pow 1 + renorm
    is not bit-exact in DALI either).

D6. Low-res in TF: `lowres_resample(x, f)` for one image [X, Y, Z, C]. It is a fused, separable,
    static-shape NN-down + linear-up: one pass per spatial axis with two `tf.gather` calls and a
    lerp. For an axis of length n:
    `m = max(1, round(f n))` (int32, dynamic); output voxel o in [0, n):
    `c = (o + 0.5) m / n - 0.5`, `l0 = floor(c)`, `w = c - l0`,
    `l0' = clip(l0, 0, m - 1)`, `l1' = clip(l0 + 1, 0, m - 1)`,
    NN source of low-res voxel l: `src(l) = min(floor((l + 0.5) / f), n - 1)`
    (`LOWRES_ROI = "subpixel"`: DALI 1.18 `subpixel_scale=True` keeps the fractional scale f);
    `out[o] = (1 - w) x[src(l0')] + w x[src(l1')]`.
    NN-then-linear resizes are tensor-product operators, so the separable form is exactly the
    two-stage 3-D resize; the output shape is always n, so everything stays traceable with static
    shapes. Pixel-center convention 0.5 and edge clamping match `P.DALI_PIXEL_CENTER` and DALI's
    linear resize (antialias off when upsampling, as in `dali_loader`).
    Known, documented possible differences from DALI, to be measured by GPU check (m1):
    (a) DALI might use the integer ratio n/m instead of 1/f for the NN source (`subpixel_scale`
    semantics). The alternative `LOWRES_ROI = "integer"`, `src(l) = min(floor((l + 0.5) n / m), n - 1)`,
    is implemented behind the module constant, and the check reports both.
    (b) Rounding of m: `tf.round` is half-to-even, DALI rounds half away from zero; f n is almost
    never exactly x.5 (difference <= 1 low-res voxel in that case).
    (c) DALI's upsample treats the m low-res voxels as spanning the full extent n (the same as here);
    with fractional f the true footprint m/f differs from n by < 1 voxel. This mismatch exists in
    DALI itself, so it is mirrored, not "fixed".
    The existing trilinear helpers in `models/layers.py` were not used: they sample a dense 3-D
    coordinate grid (8 corner gathers of the whole volume), which is heavier and not exactly NN.

D7. Memory: `augment_group(images, cfg, generator)` uses
    `tf.map_fn(..., parallel_iterations=1, fn_output_signature=tf.TensorSpec(images.shape[1:], tf.float32))`
    over (image, u-row, seed-row), so only ONE image's intermediates are alive at a time. Within an
    image at most ~4 single-volume temporaries are alive (noise field + sum; contrast; per-axis
    gather pair + lerp; gamma pow + renorm). The augmentation creates no tf.Variables and no
    persistent buffers besides the 16 floats / 2 int64 per image and the generator state.
    Expected extra TF transient at level 1 (one volume = 6,881,280 voxels x 4 B = 26.25 MiB):
    linear N 8 = map_fn TensorArray elements + stacked output, about 2 x 210 MiB, plus ~4 x 26 MiB,
    so ~0.5 GiB; deform N 4 ~0.3 GiB. This runs before the model forward, outside the tape, and is
    released before the step's backward peak (~30.7 GB), so the step peak is unchanged; worst case
    +0.5 GiB if the allocator overlaps it. Item 3 measures it locally at level 2 (and level 1, N 2).

D8. dtype: the augmentation always computes in float32 with explicit dtypes (`tf.float32`
    uniforms, `tf.random.stateless_normal(dtype=tf.float32)`, float32 math). The `mixed_float16`
    Keras policy does not affect raw tf ops, and the post-process already casts images to float32.
    The model casts its inputs under AMP. pow / std / mean in fp16 would be inaccurate, so there is
    no fp16 path.

D9. TF memory limit at level 1 (`slurm/m9_level1_cascade.sh`): `TF_MEM_LIMIT_MB` defaults to 40000.
    RTX 8000 usable 46080 MiB - CUDA context ~500 - DALI without aug ~4600 (measured 4.5 GB) -
    >= 1024 headroom outside TF (cuBLAS/cuDNN handles, DALI pool growth) = 39956, rounded to 40000.
    A100 40 GB (~40536 MiB usable): 40536 - 500 - 4600 - 1024 = 34412, so `TF_MEM_LIMIT_MB=34000`
    goes on the documented `-p dgx-a100` line. `TF_MEM_LIMIT_MB=0` means memory growth (flag
    omitted). `INTENSITY_AUG_DEVICE` defaults to `tf`. With `INTENSITY_AUG_DEVICE=dali` and
    `TF_MEM_LIMIT_MB > 30000` the script exits 2: DALI needs ~14 GB then, and job 10517362 showed
    cuBLAS handle creation failing at 38000. The level-4/2 scripts (m7, m8) stay unchanged by
    default; m8 gets a two-line comment on how to switch (`EXTRA_ARGS="--intensity-aug-device tf"`).

D10. Memory-model DALI term (TF-free, `models/memory_model.py`): DALI device memory is linear in
    images x voxels, calibrated on job 10517362 (level 1, rigid N 8 + affine N 8 + deform N 4 =
    20 images, prefetch 1, exec_async/pipelined on):
    `DALI_BYTES_PER_IMAGE_VOXEL = {"dali": 14.0 * GIB / (20 * 6881280), "tf": 4.5 * GIB / (20 * 6881280)}`
    (about 109.2 B and 35.1 B; key = intensity-aug device; `"off"` (`--intensity-aug false`) uses
    the `"tf"` value). `dali_memory(level, groups, aug_device)` returns
    `{"total_gib", "per_pipeline": {task: gib}, "bytes_per_image_voxel", "calibration": "<job, config>"}`
    with `groups` like `{"rigid": 8, "affine": 8, "deform": 4}`. Labels (uint8) and the fixed pool
    overhead are absorbed in the coefficient; it is calibrated at prefetch 1 only (documented).
    Check values: level 2 (16/16/8) aug 3.50 GiB, level 4 (32/32/16) 0.88 GiB. Feasibility:
    `predict(..., dali_gib=0.0)` adds `out["dali_gib"]` and `out["total_gib"] = peak + dali_gib`;
    `feasible_44` / `feasible_36` and `feasible(p, margin, budget)` use the effective budget
    `budget - dali_gib` for the peak AND the validation chunks (DALI stays resident the whole run).
    `grid`, `candidate_table` and `recommend` get a `dali_gib=0.0` kwarg forwarded to every
    `predict`. The Python default 0.0 keeps the fit, hold-out gate and every existing test
    unchanged: calibration records were measured without DALI. The CLI (`scripts/memory_model.py`
    predict/grid/recommend) gets `--dali {none,dali,tf}` (default `tf`, the new level-1 recipe) and
    `--dali-groups` (default `rigid:8,affine:8,deform:4`). `none` reproduces the old numbers; this
    is noted in the CLI docstring.

D11. GPU check letter: `verify_dali_pipeline.py` already uses `i` (`i_global_shared`), so the new
    DALI-vs-TF check is `m` (next free letter). It is added to the default `--checks` list.

## Items

- [ ] 1. Constants + TF-free numpy reference.
      Add `INTENSITY_AUG_DEVICES = ("dali", "tf")` and `INTENSITY_AUG_SEED_OFFSET = 7919` to
      `data_loading/params.py` (near `IntensityAugConfig`, with a short comment), plus
      `N_AUG_UNIFORMS = 16` and the column map (D5) as a named dict `AUG_COLUMNS`.
      Create `data_loading/intensity_ref.py` (numpy only, py3.8):
      `factors_from_uniforms(u, cfg)` (D5 table -> dict of per-image apply flags / values);
      `lowres_np(img, f, roi="subpixel")`, a true two-stage resize: build the low-res array
      [m_x, m_y, m_z] by NN index arrays, then upsample per axis with the pixel-center linear rule
      and edge clamp (written independently of the fused formula);
      `gamma_np`, `contrast_np`;
      `augment_np(img, u_row, noise, cfg)`, the full chain for one image given its uniforms and its
      pre-drawn unit-normal noise field;
      `sample_uniforms_np(rng, n)` (numpy RNG with DALI's sampling semantics for statistical tests).
      Files: data_loading/params.py, data_loading/intensity_ref.py
      Verify: bgr_preproc38 env: `python -c "import data_loading.intensity_ref, sys; assert 'tensorflow' not in sys.modules"`
      and py_compile + `ast.parse(..., feature_version=(3, 8))` of both files. Both pass.

- [ ] 2. TF augmentation module `data_loading/tf_intensity_aug.py` (depends on 1).
      Functions: `draw_aug_params(generator, n)` -> (u [N, 16] float32, seeds [N, 2] int64);
      `lowres_resample(x, f, roi=LOWRES_ROI)` (D6; per-axis helper `_lowres_axis(x, axis, n, f)`
      with static n from `x.shape`); `augment_image(x, u_row, seed, cfg)` (D5 order: noise ->
      brightness -> contrast -> low-res -> gamma -> clamp, float32, `tf.cond` skips);
      `augment_group(images, cfg, generator)` (D7 `tf.map_fn`, parallel_iterations=1; returns
      images unchanged if `cfg is None or not cfg.enabled`, without drawing). Wrap the body in
      `tf.name_scope("intensity_aug")` (marker for the "no ops on default path" test). The module
      docstring records D5-D8 including the isotropic factor and the DALI resize differences.
      Files: data_loading/tf_intensity_aug.py
      Verify: covered by item 3's tests; `python -m py_compile data_loading/tf_intensity_aug.py` in
      bgr_preproc38 passes.

- [ ] 3. Unit tests for the augmentation (depends on 1, 2). New `tests/test_tf_intensity_aug.py`
      (imports `tests.tf_setup` first; small volumes such as 12x14x16 unless stated):
      Per-op TF vs numpy (forced parameters; atol 1e-5 unless stated):
      - T1 noise: given u and seed, TF output == img + std * stateless_normal(seed); the numpy
        reference with the same noise field matches; prob 0 -> exact identity.
      - T2 brightness: low/high side factor ranges; product matches.
      - T3 contrast: clamp bounds are the PRE-contrast min/max; matches `contrast_np`.
      - T4 low-res: `lowres_resample` vs `lowres_np` (the two-stage reference) for f in
        {0.5, 0.55, 0.63, 0.8, 0.97, 1.0} and odd/even sizes (e.g. 11x16x13, 40x56x48), both ROI
        modes; f = 1 is an exact identity; output shape static under `tf.function` with an
        `input_signature`.
      - T5 low-res effective resolution: on a 1-voxel checkerboard / white-noise volume (32^3),
        f = 0.5 -> lag-1 autocorrelation along each axis >= 0.4 (input ~0) and per-axis distinct
        NN source indices == round(f n); f = 1 leaves it unchanged.
      - T6 gamma: output mean and std equal the input's (rtol 1e-4); g = 1 is ~identity
        (atol 1e-5); matches `gamma_np`.
      - T7 clamp: chain output in [0, 1] for inputs in [-0.2, 1.3]; disabled config (`off()`) ->
        input returned unchanged (no clamp), as in DALI.
      Full chain:
      - T8 `augment_group` vs `augment_np` per image for 8 images with params from
        `draw_aug_params` (the test re-creates the noise fields from the same seeds) (atol 1e-5).
      - T9 statistical, parameters: 20000 rows from `draw_aug_params` -> per-op apply frequency
        within 4 binomial sigma of its probability; low-side frequency ~0.5; factors inside the
        configured ranges; low-res f in [min, 1].
      - T10 statistical, chain: fixed smooth phantom (16x20x24, values in [0, 1]); 400 images from
        TF (generator seeds) vs 400 from `augment_np` with `sample_uniforms_np` + numpy normal
        noise: two-sample KS (`scipy.stats.ks_2samp`) on per-image mean, std, p1, p50, p99 and the
        fraction at 0 / 1, each p >= 1e-3.
      Determinism / resume:
      - T11 same seed -> identical `augment_group` output; different seed -> different output.
      - T12 resume: run 3 calls, `tf.train.Checkpoint(g=generator).write(...)`, 2 more calls ->
        A; a fresh generator with a different seed restored from the checkpoint gives 2 calls == A
        (bit-exact).
      dtype / memory:
      - T13 under `tf.keras.mixed_precision.set_global_policy("mixed_float16")` (restored in
        `finally`) the output dtype is float32 and it equals the float32-policy output.
      - T14 GPU only (skip without GPU): `tf.config.experimental.reset_memory_stats("GPU:0")`, then
        one `augment_group` at level 2 N 8 under `tf.function`; peak - baseline <= 3 x group bytes
        + 8 x volume bytes. Also level 1 with N 2 if it fits the 6 GB test limit (skip on OOM).
        Print the measured peaks (MiB) to stderr for the report.
      Files: tests/test_tf_intensity_aug.py
      Verify: `python -m unittest tests.test_tf_intensity_aug -v` (tf_gpu): all pass. Record the
      T4 max diff and the T14 peaks for the report.

- [ ] 4. Wire into the post-process and the Trainer (depends on 2).
      `data_loading/postprocess.py`: both classes get `intensity_aug=None` (an
      `IntensityAugConfig` or None). When it is enabled: `self.intensity_aug = cfg`,
      `self.aug_generator = tf.random.Generator.from_seed(int(seed) + P.INTENSITY_AUG_SEED_OFFSET)`.
      Otherwise `self.intensity_aug = None` and `self.aug_generator = None`. In `__call__`, right
      after the float32 cast: `if self.intensity_aug is not None: images = tfa.augment_group(images, self.intensity_aug, self.aug_generator)`.
      This is a Python switch, so the default builds no op. Update both class docstrings and the
      module docstring (step "0. intensity augmentation (optional, --intensity-aug-device tf)").
      `runtime/run.py`: add `TrainConfig.intensity_aug_tf: Optional[object] = None` (comment:
      `IntensityAugConfig` or None, D1); pass `intensity_aug=cfg.intensity_aug_tf` to both
      post-process constructors in `Trainer.__init__`; `checkpoint_variables()` adds
      `"postprocess_%s_aug" % t: pp.aug_generator` for every pp whose `aug_generator` is not None.
      Add tests to `tests/test_postprocess.py`:
      - default (`intensity_aug=None`): `aug_generator is None`, and the traced graph of a
        `tf.function` call contains no op under `intensity_aug/`;
      - with the config: images change, labels / one-hot / matrices / pairs / flip_axes equal those
        of the default post-process with the same seed (the flip/pair stream is unchanged, D4);
        output in [0, 1];
      - the deform post-process never augments labels.
      Add tests to `tests/test_trainer.py` (`TestResumeAndValidation` or a new
      `TestTFIntensityAug(TrainerCase)`):
      - `checkpoint_variables()` keys unchanged by default; `postprocess_rigid_aug` (etc.) present
        with `intensity_aug_tf`;
      - resume: train 3 steps with TF aug and a checkpoint, restore into a fresh Trainer; the next
        post-process output (`tr.postprocess["rigid"].aug_generator.state`) equals the
        uninterrupted run's state after 3 steps;
      - one step runs and is finite with TF aug (rigid and deform).
      Files: data_loading/postprocess.py, runtime/run.py, tests/test_postprocess.py, tests/test_trainer.py
      Verify: `python -m unittest tests.test_postprocess tests.test_trainer.TestLinearOnlyUnchanged tests.test_trainer.TestDeformUnchanged tests.test_trainer.TestResumeAndValidation -v`
      (plus the new class): all pass; both goldens print max |diff| 0.

- [ ] 5. CLI + main plumbing (depends on 4).
      `runtime/args.py` (already imports `data_loading.params as P`, TF-free):
      `g.add_argument("--intensity-aug-device", choices=list(P.INTENSITY_AUG_DEVICES), default="dali", help=...)`
      right after `--intensity-aug`. Add the parse-time error for `tf` + `--overfit-one-group` (D1)
      where `parse_args` validates the other flags, a docstring bullet, and
      `intensity_aug_tf=(P.IntensityAugConfig() if args.intensity_aug and args.intensity_aug_device == "tf" and not args.overfit_one_group else None)`
      in `train_config_from_args`.
      `main.py::_build_dali_data`: `aug = P.IntensityAugConfig() if (args.intensity_aug and args.intensity_aug_device == "dali") else P.IntensityAugConfig.off()`;
      append `"; intensity aug %s" % ("off" | "dali" | "tf")` to `desc`.
      Resume guard: new helper `_check_intensity_aug_device(config_path, args)`. It reads the saved
      config.json args (look up how `C.write_config` stores `A.args_to_dict(args)`; a missing key
      means `"dali"`) and raises `SystemExit("main.py: --intensity-aug-device %s differs from the run's saved %s ...")`
      on a mismatch. Call it in `main()` before `trainer.checkpoint.restore_last()` on the resume
      branch, and for cascade resumes too (same branch).
      Tests in `tests/test_main.py` (follow the `FakeDM` kwarg-capturing pattern at ~line 265-300):
      default -> data module gets an enabled `IntensityAugConfig` and `TrainConfig.intensity_aug_tf is None`;
      `tf` -> data module gets `off()` and `intensity_aug_tf` is enabled;
      `--intensity-aug false --intensity-aug-device tf` -> both off; the overfit + tf parse error;
      the resume-guard helper (mismatch -> SystemExit, missing key treated as dali).
      Files: runtime/args.py, main.py, tests/test_main.py
      Verify: `python -m unittest tests.test_main -v` (tf_gpu) passes; bgr_preproc38:
      `python main.py --help` works without TF and lists `--intensity-aug-device`.

- [ ] 6. GPU check (m) in `tests/gpu/verify_dali_pipeline.py` (depends on 2; DALI only on the cluster).
      Add `check_m(files, tmp, args)` and register it as `"m"` in the default `--checks` string,
      the `main()` dispatch (`guarded("m_tf_intensity_aug", check_m, files, tmp, args)`) and the
      module docstring. Use `LEVEL` / `SHAPE` globals only (no level-4 literals;
      `tests/test_verify_dali_level.py::test_no_level4_literals`).
      m1 low-res voxelwise: an `FnPipeline` graph reading `phantom.npy` from tmp (as `check_g`),
      `fn.resize` NN to `f * SHAPE` and linear back (antialias False), i.e. the exact
      `dali_loader.intensity_augment` low-res code with a constant size, for f in
      (0.5, 0.6, 0.75, 0.9). Compare with `tfa.lowres_resample` in both ROI modes: max / mean abs
      diff per f. PASS if the active `LOWRES_ROI` has mean abs diff <= 5e-3 and is not worse than
      the alternative; else FAIL with both numbers (the fix is to flip the constant).
      m2 statistical per op and full chain: real image `files[0]` repeated N = `--n-real` times;
      `dl.LinearPipeline(task="rigid", rotation_deg=0, translation_mm=0, log_scale=0, shear=0, shuffle=False)`
      with (i) the default augmentation and (ii) `off()`, plus TF `augment_group` applied to the
      output of (ii). Configs: full default, and one per op with only that op enabled at prob 1
      (noise, brightness, contrast, low_res, gamma). Run ~16 batches per config. Compare per-image
      mean, std, p1/p50/p99 and the fraction at 0 / 1 with `scipy.stats.ks_2samp`: PASS if every
      p >= 1e-3. Also check min >= 0 and max <= 1 for both, and gamma: |mean - input mean| and
      |std - input std| < 1e-4 for both. Record all numbers in RESULTS.
      m3 INFO: `gpu_memory()` before / after building one LinearPipeline with aug on vs off at
      `--level` (DALI footprint per pipeline, for the memory-model calibration).
      Extend `tests/test_verify_dali_level.py`: the default `--checks` includes `m`, `check_m`
      exists, and the source has no level-4 literals (existing test).
      Files: tests/gpu/verify_dali_pipeline.py, tests/test_verify_dali_level.py
      Verify: py3.8 py_compile + ast of the script; `python -m unittest tests.test_verify_dali_level -v`
      (tf_gpu) passes. Cluster command for the orchestrator (not run locally):
      `python tests/gpu/verify_dali_pipeline.py --data <npy_v2> --scratch <proj>/scratch/verify_m --out <...>/verify_m.json --checks m --level 1`
      (and `--level 4` for speed).

- [ ] 7. Level-1 recipe in `slurm/m9_level1_cascade.sh` (independent of 6).
      Add `INTENSITY_AUG_DEVICE="${INTENSITY_AUG_DEVICE:-tf}"` and
      `TF_MEM_LIMIT_MB="${TF_MEM_LIMIT_MB:-40000}"`, with validation: device in {dali, tf}, limit a
      non-negative integer, `dali` + limit > 30000 -> exit 2 with a message. Build `MEM_ARGS`
      (`--gpu-memory-limit-mb $TF_MEM_LIMIT_MB` iff > 0) and pass
      `--intensity-aug-device "$INTENSITY_AUG_DEVICE" $MEM_ARGS` before `$EXTRA_ARGS`. Echo both in
      the run line. Header comment: the D9 arithmetic (RTX 8000 and A100), the measurement source
      (job 10517362), and the A100 line becomes
      `TF_MEM_LIMIT_MB=34000 RUN_NAME=m9_level1_cascade sbatch -p dgx-a100 --job-name=m9long --time=36:00:00 --mem=64G slurm/m9_level1_cascade.sh`
      (keep the existing substring the test asserts). ASCII only.
      `slurm/m8_level2_cascade.sh`: a two-line comment on how to switch (EXTRA_ARGS), no behaviour
      change. Leave `slurm/m7_three_task.sh` untouched.
      Tests in `tests/test_slurm_scripts.py`:
      `test_m9_level1_default_recipe` additionally asserts `["--intensity-aug-device", "tf"]` and
      `["--gpu-memory-limit-mb", "40000"]`; a new test checks that `TF_MEM_LIMIT_MB=0` omits the
      flag, `INTENSITY_AUG_DEVICE=dali TF_MEM_LIMIT_MB=28000` passes both, and
      `INTENSITY_AUG_DEVICE=dali` (default limit), `INTENSITY_AUG_DEVICE=gpu` and
      `TF_MEM_LIMIT_MB=4e4` exit 2 without calls. Add the header strings to the asserted list.
      Files: slurm/m9_level1_cascade.sh, slurm/m8_level2_cascade.sh, tests/test_slurm_scripts.py
      Verify: `bash -n slurm/m9_level1_cascade.sh slurm/m8_level2_cascade.sh` (WSL); bgr_preproc38:
      `python -m unittest tests.test_slurm_scripts -v` passes.

- [ ] 8. Memory-model DALI term (independent of 2-7).
      `models/memory_model.py`: constants `DALI_BYTES_PER_IMAGE_VOXEL`, `DALI_CALIBRATION` (text:
      job 10517362, RTX 8000, level 1, rigid 8 + affine 8 + deform 4, prefetch 1, 14.0 / 4.5 GiB),
      `DALI_DEVICES = ("dali", "tf", "off")`; `dali_memory(level, groups, aug_device)` (D10);
      `predict(..., dali_gib=0.0)` adding `dali_gib` and `total_gib` and using the effective
      budget for `feasible_44` / `feasible_36`; `feasible(p, margin, budget)` subtracts
      `p.get("dali_gib", 0.0)` from `budget` for both the peak and the chunk tests; `dali_gib` kwarg
      threaded through `candidate_table`, `grid`, `recommend` (every internal `predict` call) and
      reported in the recommendation dict / printout when > 0. Module docstring paragraph.
      `scripts/memory_model.py`: `--dali {none,dali,tf}` (default `tf`) and `--dali-groups`
      (default `rigid:8,affine:8,deform:4`) for predict / grid / recommend, using the subcommand's
      `--level` where present (else 1). Print the DALI line (`DALI x.xx GiB (aug device ..., groups ...)`)
      and include it in `--json`. Docstring note: `--dali none` = the pre-DALI-term numbers.
      Tests (TF-free, `tests/test_memory_model.py`, new `TestDaliTerm`):
      level 1 recipe -> 14.0 / 4.5 GiB (places=6) and per-pipeline split 5.6 / 5.6 / 2.8 (aug);
      level 2 16/16/8 aug ~3.50 GiB; level 4 32/32/16 ~0.875 GiB; unknown device / task ->
      ValueError; `predict` with dali_gib 0 == the old output (feasible flags, peak);
      `feasible(p, m, 44)` flips from True to False when dali_gib pushes peak (1+m) + dali over 44;
      `total_gib == peak_gib + dali_gib`; `MM.grid(co, dali_gib=0.0)` row counts unchanged; CLI
      `predict --dali none` vs `--dali tf` JSON differ only in dali / feasibility fields. The
      existing `test_coefficients_file_current`, the hold-out gate and the recommendation tests stay
      green WITHOUT refitting (the calibration records are untouched).
      Files: models/memory_model.py, scripts/memory_model.py, tests/test_memory_model.py
      Verify: bgr_preproc38 (TF-free): `python -m unittest tests.test_memory_model -v` passes, with
      no TF import (`test_no_tensorflow`).

- [ ] 9. Report + full verification (depends on all).
      Write `.agents/tasks/tf-intensity-aug/report.md` (short). Cover: what changed; the flag and
      the recipe; D4-D8 decisions incl. the isotropic low-res factor and the DALI resize
      differences with the local T4 numbers; the measured T14 peaks and the expected level-1 TF
      transient (~0.5 GiB linear N 8, ~0.3 GiB deform N 4, before the model forward); the
      TF_MEM_LIMIT_MB arithmetic; the DALI term calibration; the cluster commands for the
      orchestrator (check m at levels 4 and 1; then the level-1 smoke with the script defaults);
      what could not be verified locally (DALI equivalence, real level-1 memory).
      Verification, in order:
      a. Goldens: `python -m unittest tests.test_trainer.TestLinearOnlyUnchanged tests.test_trainer.TestDeformUnchanged -v`
         -> both pass, stderr shows max |diff| 0.
      b. py3.8: in bgr_preproc38, `python -m py_compile` of every changed / new .py, and
         `ast.parse(src, f, feature_version=(3, 8))` (the snippet from `tests/run_tests_wsl.sh`
         stage 1).
      c. `bash -n` on slurm/m9_level1_cascade.sh and slurm/m8_level2_cascade.sh.
      d. Full suite in the background (~3 h+; poll the log, do not block):
         `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/tests/run_tests_wsl.sh > <workspace>/.agents/tasks/tf-intensity-aug/suite.log 2>&1`
         -> exit 0, stage 1 OK lines, `unittest exit code: 0`; record the test count (previously
         520) in the report.
      e. Remove any temporary files; `__pycache__` cleanup is done by the suite script.
      Files: .agents/tasks/tf-intensity-aug/report.md
      Verify: suite.log ends with `unittest exit code: 0` and the script exits 0.

## Notes / assumptions
- DALI semantics that cannot be checked locally (resize ROI rounding, op seeding) are covered by
  check m on the cluster. If m1 fails, flip `LOWRES_ROI` and rerun m (one-line change).
- TF aug and DALI aug draw from different RNGs, so a `tf` run is statistically, not bitwise,
  equivalent to a `dali` run. That is the required equivalence.
- The DALI-term calibration has only two points (with / without aug at one configuration). The
  linear-in-images scaling to other levels and group sizes is an assumption; check m3 adds
  per-pipeline points.

## Addendum A (orchestrator, Jay approved): per-pair recompute of the deform pairwise terms

- `--pair-recompute true|false` (default false = today's ops, goldens bit-identical); `TrainConfig.pair_recompute`.
- `runtime/losses.pair_terms_recompute`: static Python loop over the pairs (count from
  `postprocess.num_pairs_static`, `tf.ensure_shape`); per pair ONE `tf.recompute_grad` block (inputs
  u_i, u_inv_j [1, X, Y, Z, 3] and the pair indices as floats; one-hot / images captured): compose,
  warp of the one-hot (and image), pairwise soft Dice (lean Dice with --lean-losses), pairwise LNCC
  with the label ROI. Block p's u_i input carries `s * z` (s = sum(result_{p-1}), z a data-dependent zero; a literal
  `0 * s` is removed by Grappler constant folding) to serialise forward and backward. Template terms unchanged. float32 throughout.
- Wired in `Trainer._make_deform_step`; flag in `tests/gpu/measure_level_memory.py` (record field
  `pair_recompute`, key field); `PAIR_RECOMPUTE` (default true) in `slurm/m9_level1_cascade.sh`.
- Tests `tests/test_pair_recompute.py`: values + gradients vs the default path (plain, lean sampler,
  lean losses, eager / tf.function), static count, graph marker, Trainer step equality (also with
  --gradient-checkpointing full + lean sampler), GPU peak 1 vs 4 pairs.

## Addendum B (orchestrator, Jay approved): level-1 validation memory

- `--val-image-chunk k` (default 0 = unchunked, levels 4/2 bit-identical): `Trainer._deform_errors_chunked`
  computes the deform validation metrics k images / `--val-pair-chunk` pairs at a time (one-hot built
  per chunk, label template from a running sum, leave-one-out intensity template from host-side warped
  images, Jacobian / inverse consistency per chunk); same numpy metrics. Identity and frozen baselines go
  through the same function. Linear validation is unchanged (its group forward cannot be split: the
  model is groupwise).
- `--bn-recal-validate true|false` (default true): passed as `recalibrate_bn(..., val=)`; false skips
  the bnrecal_pre / bnrecal_post passes (checker bn_gap INFO lines are then absent, not a failure).
- `slurm/m9_level1_cascade.sh`: VAL_IMAGE_CHUNK=1, VAL_PAIR_CHUNK=1, BN_RECAL_VALIDATE=false.
- Tests `tests/test_val_chunk.py`: chunked == unchunked metrics (plain / lean sampler, chunk 1 and 3);
  GPU train -> validate -> train under the fixed 6000 MB limit with the validation peaks.