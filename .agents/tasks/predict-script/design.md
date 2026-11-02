# Standalone groupwise prediction script: requirements and design

Version 5 (2026-11-01). Status: revision after design review v4 (CHANGES_REQUESTED, 0 HIGH,
2 MEDIUM, 3 NIT); responses in section 16 (v3 in section 15, v2 in section 14, v1 in section 13).
Jay's decision D1 (branch set) and D2 (GPU job scope) are recorded in sections 3.4, 9 and 12.
New in v5: check-streaming has its own chunk defaults (`min(cap, max(1, n - 1))` per level, so
every level streams with a remainder chunk); the cluster gate runs at `--precision float32` with
the float32 bounds, the fp16 and affine+deform checks are recorded but do not gate; the model
input of every level is stored once as `x_in`; test (b) names the compared fields per level;
optional `--grid-reference` checks the processed affine against the training grid.
(New in v4: nibabel in tf_gpu (10.0), one tolerance set for streamed vs any float32 reference,
lean sampler at inference, git-commit fallback, rigid check tolerances.)

## 0. Summary

`predict.py` (thin CLI) + a new package `inference/` load a saved run checkpoint (config restored
from the run's `config.json`, nothing hardcoded), read a folder of processed subjects on the
160x224x192 1 mm grid, run the selected branches for the WHOLE group at once and write, per
subject, one forward and one inverse displacement (mm, canonical `evaluation/fields.py`
convention) and the moved image, plus `atlas.nii.gz`, `atlas_seg.nii.gz` and `predict.json`. A
full-resolution group of N = 101 runs on one GPU through exact group streaming: in inference mode
every group interaction of the network is a group MEAN (or a computation on N small matrices),
so the network can run one group-block at a time over chunks of images, accumulating each block's
group mean in a first sweep and applying it in the next. The same code runs on CPU
(`--device cpu`). An unchunked mode (`--mode unchunked`) calls exactly the trainer's validation
math and is the reference of the equivalence check.

## 1. Requirements

### 1.1 Functional

- F1 Inputs: `--checkpoint` (checkpoint prefix or checkpoint directory, e.g.
  `.../m9_level1_cascade/ckpt/best_deform`), `--processed-dir` with
  `<subject_dir>/image.nii.gz` and `labels_<label-name>.nii.gz` (default `dkt31aseg`), optional
  `--subjects-file`; subject names and order exactly as `evaluation.evaluate.find_subjects`.
- F2 Model restored from the run directory: `config.json` (`model_config`, `args`, `cascade`)
  plus the run checkpoint's `model` and frozen stages. Supported run kinds: level-4 single stage,
  level-2 cascade (`frozen_stage`), level-1 cascade (`frozen_stage_4`, `frozen_stage_2`).
- F3 Branch selection `--branches` in {rigid, affine, deform, affine+deform} (section 3.4,
  Jay's D1); deform-only must work; any value other than these four (e.g. all, rigid+deform,
  rigid+affine, deform+affine) is an argparse error (exit 2).
- F4 The whole group in one model evaluation (all group means, the centering and the atlas over
  all N subjects; no subsets).
- F5 Outputs in `--out`: `<subject_dir>_fwd.nii.gz`, `<subject_dir>_inv.nii.gz` ([X,Y,Z,3]
  float32, mm, components x,y,z = grid axes i,j,k, canonical roles), `<subject_dir>_moved.nii.gz`,
  `atlas.nii.gz`, `atlas_seg.nii.gz`, `predict.json`. Affine = the processed grid affine.
- F6 `predict.py check-streaming`: streamed vs unchunked on a small group of real subjects, JSON
  report with max/mean differences and a pass/fail gate. Its chunk defaults split every level
  into at least two chunks including a remainder (section 4.6), so the check exercises chunked
  accumulation at every cascade level.
- F6b Optional `--grid-reference PATH` (target_grid.json or a NIfTI on the training grid): the
  processed affine must equal the reference affine (section 5.1).
- F7 GPU (default) and CPU device (section 3.3, mechanism and precision); streamed (default)
  and unchunked modes.
- F7b Internal consistency check (section 5.3): for the first `--check-internal K` subjects
  (default 2), the written canonical fields are compared with the model's own TF warp ops
  (independent of how the moved file is produced); the numbers go to predict.json.
- F8 Room for test-time flips: `--tta` with the single choice `none`.
- F9 SLURM script `slurm/predict_mindboggle.sh` (section 9).

### 1.2 Non-functional

- Python 3.8 syntax, TF 2.10, numpy 1.21, scipy 1.4, nibabel 3.1 (tf10_nv.sif); runs locally in
  WSL `tf_gpu` (py3.9) for tests.
- TF32 off (`configure_tf_precision()` before any TF op).
- Streamed result equals the unchunked model output within float tolerance (section 8, gate G1).
- N = 101 at level 1 fits one RTX 8000 (48 GB) with default widths; host RAM <= 96 GB (with a disk
  spill fallback).
- No change to training behaviour: model-code changes are pure refactors (bit-identical graphs).

### 1.3 Acceptance criteria

0. Tolerance sets used by AC 1 and AC 2 (one set per kind of comparison; every test prints the
   measured maxima and the report states them):
   - S_stream (streamed mode vs ANY float32 reference: the unchunked script or the trainer):
     fields (u, u_inv, fwd, inv) max abs diff <= 1e-4 vox; A max abs diff <= 1e-5 (all 12
     entries; translation in mm); Dice |diff| <= 1e-4; warped / moved image at the comparison
     level max abs diff <= 1e-4.
   - S_exact (unchunked script vs the trainer; identical ops): fields and A <= 1e-6, Dice
     identical, warped image <= 1e-5. If 1e-6 fails because of graph-level fusion differences
     (traced whole-cascade function vs per-stage functions), the field / A bound falls back to
     1e-5 and Dice to <= 1e-6; the measured maximum is printed and reported.
   - S_fp16 (streamed vs unchunked with a mixed_float16 trainable stage): fields <= 2e-2 vox max,
     <= 1e-3 vox mean.
1. (a) `tests/test_predict_streaming.py`: streamed == unchunked on random-weight TINY models at
   levels 4 and 2 (cascade), every branch, chunk sizes 1, 2 and N, float32 policy (S_stream) and
   mixed_float16 trainable stage (S_fp16).
   Level 1 (`FrozenChain` + trainable level-1 stage: `pool2`, `up_d1`, the fused stem / d1
   passes, the 3-stage orchestration, the linear chain A4 -> A2 -> A1) runs on the CPU inside the
   GPU test process (`with tf.device('/CPU:0')`, section 10 (a)): TINY widths, float32, N = 3,
   chunks 1 and 2, branches deform and affine+deform, same tolerances. It is never skipped for
   memory; a `skipTest` is allowed only when no GPU is visible (tf_gpu env requirement).
2. (b) `tests/test_predict.py`: trainer equivalence at level 2 (cascade `Trainer` with a
   `FrozenStage`) and level 1 (`Trainer` with a `FrozenChain`, CPU scope as in AC 1), both modes.
   References: `Trainer._val_deform_fn(nv)` and `Trainer._val_fn(task, nv)`, which dispatch to
   `_cascade_val_deform_fn(nv)` / `_cascade_val_fn(task, nv)`. Unchunked mode, both sides run
   under `tests.test_trainer._OpDeterminism` (deterministic ops, cuDNN autotune off) and the
   unchunked runner wraps each level stage in one `tf.function` with the trainer's input
   signature: u / u_inv, A, Dice (`Trainer._deform_errors` on the script's fields) and the
   trainer's warped image at its level (`diffeo.warp(x, u_trainer)`) vs the script's internal
   warp at that level, all within S_exact (AC 0, including its fallback rule). Streamed mode with
   chunk 1 and chunk N: the same four quantities within S_stream (AC 0), the same bound set as
   AC 1. Which fields are compared:
   - Level 2 (trainer fields live on the level-2 grid): the runner's INTERNAL fields at the
     checkpoint level (u, u_inv after the level-2 `compose_cascade`, before the output pass's
     upsampling and canonical conversion) and the internal A.
   - Level 1 (deform branch): the runner's internal u / u_inv as above, AND the written
     `<subject_dir>_fwd.nii.gz` / `_inv.nii.gz` files read back with nibabel and converted to
     voxels (divided by the spacing, 1 mm) vs the trainer's u / u_inv (fwd_vox = u, inv_vox =
     u_inv for the deform branch, section 5.1), within the same bound set (S_exact for the
     unchunked script, S_stream for streamed). This ties the files to the trainer where both
     live on the same grid.
   - Linear branches at either level: the internal A vs the trainer's A (the written linear
     fields are tied to A by the AC 4 / 5.3 checks).
3. (c) Convention evidence that does not depend on how the moved file is written (section 5.3):
   for every branch, with randomised heads giving rotation >= 5 deg, translation >= 3 mm and
   |u| >= 2 vox: (i) coordinate test: the model's internal warp ops applied to a coordinate volume
   give the map p -> p + fwd_vox(p) to <= 1e-3 vox on the valid interior, and the internal
   subject-space pull gives y -> y + inv_vox(y) to <= 1e-3 vox; (ii) natural image, single-
   interpolation branches (rigid, affine, deform): max |F.warp(I, fwd) - internal_moved| <= 1e-3
   on the valid interior; (iii) inverse consistency, gated only where it is a single
   interpolation: |F.compose(fwd, inv)| <= 1e-3 mm on the interior for rigid and affine, and
   |F.compose(fwd, inv) - diffeo.compose(U1_inv, U1)| <= 1e-3 mm for deform. For affine+deform
   both numbers are recorded and printed but not gated (F.compose samples the stored inv at
   off-grid points, a second interpolation of the rotated u_inv; the coordinate checks (i) prove
   that branch's maps in both directions). Then `python -m evaluation check-convention` on the outputs
   (affine+deform) prints `RESULT: use  --components xyz --signs +++ --units mm` and exits 0, and
   `python -m evaluation run` with those flags and `--pairs 2` exits 0 with finite metrics. (On a
   1 mm grid check-convention cannot tell mm from vox; the units are covered by AC 5.)
4. (d) Branch combinations: every supported value runs and produces the documented composition
   (rigid: the 3x3 linear part R of Mv(A^-1), read from the runner output, and the
   least-squares linear part of p -> p + fwd_vox(p) fitted on the interior Omega (5.3) agree to
   1e-4; the runner's R satisfies |det R - 1| <= 1e-5 and max |R^T R - I| <= 1e-5 (the centered
   rigid A is rigid up to logm / expm rounding), the fitted one the same two checks at 1e-4
   (float32 rounding of the stored field); deform: equals the deform cascade on the raw images;
   affine+deform: equals the deform cascade run on the images resampled with the script's A,
   composed); any other value (e.g. `all`, `rigid+deform`, `rigid+affine`, `deform+affine`) exits
   2 with a message naming the four supported values.
5. Unit tests for the canonical conversion (synthetic A and u, exact formulas, including a
   non-unit spacing case: a 2 mm synthetic grid at the converter level so the `x spacing[a]` step
   is tested), store backends, chunk cap, unchunked group-size bound (default widths level 1:
   deform n_max 4, linear n_max 9; `--mode unchunked` / `check-streaming` above it -> exit 2),
   `deform_decoder_dtype 'float16'` checkpoint loaded under a float32 build (mapped to `policy`,
   recorded), CLI validation, `--device cpu` setup (subprocess, or the `setup_device` call-order
   unit test where the local env cannot run CPU-only, section 10), predict.json schema,
   nonfinite handling, internal-check gate, check-streaming chunk defaults (`--n-subjects 4`
   without chunk flags resolves to (3, 3, 3); explicit flags override; values above the int32
   cap are capped), `--grid-reference` (json and NIfTI reference; equal affine passes, an
   LR-mirrored affine (first row negated, offset adjusted) exits 2; absent -> INFO and skipped;
   predict.json records the processed affine and the reference path).
6. Full suite (`tests/run_tests_wsl.sh`) passes; new files in the py3.8 syntax check and
   `bash -n` list. Precondition: nibabel 3.1.1 is installed in the local tf_gpu env after Jay's
   approval (section 10.0); stage 2 of the runner fails loudly if it is missing; no nibabel test
   is skipped for a missing nibabel.
7. Cluster job (after Jay's approval): step 1a gate G1 passes on the real level-1 checkpoint for
   `$BRANCHES` (deform) at `--precision float32` with the float32 bound set and chunks 3 / 3 / 3
   at N_CHECK = 4 (two chunks with a remainder at every level; JSON records `chunks` and
   `gpu_peak_mb`); the fp16 check (1b) and the float32 affine+deform check (1c) are run and
   recorded, non-blocking (section 9),
   step 2 writes 101 x 3 subject files + atlas files with zero non-finite values and
   predict.json `internal_check` passes for the checked subjects (coordinate max <= 1e-3 vox,
   moved vs internal <= 1e-3), step 3 check-convention selects the canonical flags and the
   template-only evaluation runs.

### 1.4 Out of scope

Retraining, TTA, preprocessing, changing evaluator metric definitions, the full pairwise
evaluation inside the GPU job (it runs on `basic` with `slurm/eval_mindboggle.sh`, section 9).

## 2. Findings from the code (what the trained model actually does)

### 2.1 Branch inputs during training

- Linear tasks (`rigid`, `affine`): DALI renders MNI-aligned training images with random
  transforms (rigid: rotation + translation + nuisance scale/shear S_i; affine: full affine) and
  the targets are A_i* = mu*^-1 T_i^-1 (`data_loading/postprocess.py` steps 1-5). Rigid and affine
  are two heads (`rigid_head`, `scale_shear_head`) on the same encoder and the same input
  distribution. Each training step runs ONE task; no step feeds one linear head's output to the
  other. They are alternatives, never trained in sequence.
- Deform task: `DeformPipeline` reads the images as they are ("already in MNI space (A = I)");
  the residual affine is off by default and `--cascade-from` requires
  `--deform-residual-affine false` (`runtime/args.py`). The m9 run has
  `deform_residual_affine False`. So the deform branch was trained on 12-dof MNI-aligned images
  and never on the output of a linear branch.
- Cascade: each task chains over LEVELS, not over tasks. Linear at level 1:
  x2 = pool(x1); A4 = Center4(model4 on pool(x2)); A2 = Center2(R2_raw A4) with model2 on
  x2 o A4^-1; A = Center1(R1_raw A2) with model1 on x1 o A2^-1 (`models/cascade.py`,
  `Trainer._cascade_val_fn`). Deform at level 1: level-4 fields upsampled to level 2, level-2
  residual on warp(x2, U4), composed (`diffeo.compose_cascade`), upsampled to level 1, level-1
  residual on warp(x1, U), composed (`Trainer._cascade_val_deform_fn`).
- Shared global transform G: only a training augmentation (DALI control file, validation sets
  `<task>_g`); no model component. At inference nothing has to be switched off; predict.json
  records `"global_transform": "none (training augmentation only)"`.
- Flips: random group flips are training augmentation; inference uses none (`--tta none`).
- Low-res inputs: the pyramid inside the cascade is 2x2x2 average pooling of the full-res images
  (`cascade.pool_to_coarser`), equal to the export's exact block average. `--lowres-method`
  (nn | tent) only selects the simulated-low-res INTENSITY AUGMENTATION, which is off at
  inference. So the script builds levels 2 and 4 by pooling, for every checkpoint; the
  checkpoint's lowres_method is recorded in predict.json but changes nothing.

### 2.2 The m9 checkpoint (read from the cluster, read-only)

`results/m9_level1_cascade/config.json`: level 1, amp True, split_group_conv True,
lean_sampler True, gradient_checkpointing full, deform_group_size 4, group_size 8,
task_ratio rigid:1,affine:1,deform:2, default widths (filters 32..384, decoder None = 128..16),
integration_steps 7, BN recalibrated 64 batches and frozen; `cascade.stages` = level 4
(`frozen_stage`, origin m7_three_task_lncc best-11000) and level 2 (`model`, origin
m8_level2_gradnorm best-6000) with fingerprints. Written by commit e6815d9, so the args have no
`deform_decoder_dtype`, `fp16_act_clip` or `bn_epsilon_override` keys: absent keys take the
current argparse defaults (`policy`, 0, 0). `ckpt/best_deform/best_deform-5000` exists. The
Mindboggle folder holds 101 `*_T1w` subjects.

### 2.3 Group interactions in inference mode (every one, and whether it decomposes)

With `training=False` BatchNorm uses moving statistics (per image), LayerNorm / InstanceNorm are
per sample, dropout is off. What remains couples the group only through:

| # | Where | Interaction | Decomposes exactly? |
|---|---|---|---|
| 1 | encoder stem (config A) | conv(concat(x_i, mean_j x_j)) on the raw image (no norm/act) | yes: mean = sum over chunks / N |
| 2 | encoder l2..l6 (config A) | `GroupConvBlock`: conv(concat(pre(x_i), mean_j pre(x_j))), pre = BN(eval) -> LeakyReLU; split form conv_a(pre) + conv_b(mean) + bias at L1 | yes: mean of pre over chunks |
| 3 | linear decoder reductions | 1x1 `GroupConvBlock`s on [N,5,7,6,C] | yes; run unchunked on the gathered bottleneck (tiny) |
| 4 | linear decoder dense | `GroupDenseBlock` (group concat), LayerNorm | unchunked on N vectors (tiny) |
| 5 | GroupCentering | barycenter of N 4x4 matrices (iterative logm/expm), mu^-1 A_raw | not a mean, but N matrices only: computed on all N at once, exactly the model's op |
| 6 | cascade compose_linear | Center(R_raw A_prev) | as 5 |
| 7 | deform decoder d5..d1 | `GroupConvBlock`s on concat(up(x), skip) | yes: mean of pre over chunks |
| 8 | SVF centering | v_i - mean_j v_raw_j | yes |
| 9 | integration, warps, pooling, upsampling, compose_cascade, pre-warps | per image | trivially |
| 10 | atlas, atlas_seg | mean of moved images, majority vote of warped labels | sum / vote counts over all N |

Nothing is non-decomposable. Streamed and unchunked results differ only by float rounding, from
these sources: (i) summation order of the group means (chunk sums accumulated in float32, divided
by N once); (ii) split vs concat convolution: the engine computes the stem and every frozen-stage
block (frozen stages are built without `split_group_conv`) as conv_a(x) + conv_b(mean) + bias,
while the model computes conv(concat(x, mean)); (iii) cuDNN may pick a different convolution
algorithm per chunk batch size; (iv) under mixed_float16, the rounding of the fp16 group mean
(the engine sums in float32 and casts the mean once; the model calls `reduce_mean` on the fp16
tensor, whose internal accumulator type is TF's choice). Training-
mode BatchNorm (batch statistics over the group) would not decompose this way, but inference
never uses it. Residual blocks (`block='residual'`) decompose the same way but no checkpoint uses
them; the streaming engine supports `block='single'` only and fails fast otherwise (YAGNI).

### 2.4 Distribution note (not a blocker)

The trained group sizes are small (level 1: deform 4, linear 8; level 2: 8 / 16; level 4:
16 / 32). With N = 101 the group-mean channels average over many more images (less noisy) than
in training. This is what Jay's decision 1 asks for; the effect is unmeasured. predict.json
records N and the training group sizes; a later ablation can compare N = 4 groups vs N = 101.

## 3. Architecture

### 3.1 Files

| File | Content |
|---|---|
| `predict.py` | CLI entry (`run`, `check-streaming`), imports nothing from TF before argument parsing |
| `inference/__init__.py` | package marker |
| `inference/cli.py` | argparse, validation of every option (section 7), exit codes |
| `inference/loader.py` | checkpoint resolution, config read, model / frozen-stage construction under the right dtype policies, restore, fingerprint checks -> `LoadedModel` |
| `inference/store.py` | `RamStore` / `DiskStore` (per-image host arrays, np.memmap), size estimate |
| `inference/streaming.py` | `StreamedLevel`: exact chunked execution of one `GroupRegLevelModel` (encoder, linear decoder, deform decoder + SVF head) |
| `inference/runner.py` | branch orchestration over the cascade levels for both modes (`StreamedRunner`, `UnchunkedRunner`), returns per-subject internal results |
| `inference/canonical.py` | internal (A, u, u_inv) -> canonical fwd / inv in mm; NIfTI writing; moved image, atlas, atlas_seg; internal consistency check (`internal_moved`, `internal_pull_inv`, coordinate / natural-image / inverse-consistency metrics, section 5.3) |
| `inference/predict.py` | `run_predict(args)`, `run_check_streaming(args)`, predict.json, timing / memory / nonfinite bookkeeping |
| `models/layers.py` | refactor only: `GroupConvBlock.pre(x, training)` and `GroupConvBlock.conv_with_group_mean(pre_x, mean, inner_clip)`; `_split_call` calls the latter (same ops in the same order) |
| `tests/test_predict_streaming.py` | tests (a) |
| `tests/test_predict.py` | tests (b), (c), (d) and unit tests |
| `tests/test_slurm_scripts.py` | static checks of the new SLURM script |
| `tests/run_tests_wsl.sh` | add `predict.py inference/*.py evaluation/*.py` to the py3.8 checks, `bash -n slurm/predict_mindboggle.sh`, the stage-2 nibabel import guard (section 10.0) |
| `slurm/predict_mindboggle.sh` | cluster job (section 9) |

No change to `runtime/`, `main.py`, `evaluation/` (the script imports `evaluation.evaluate`
`find_subjects`, `load_grid_data`, `spacing_of`, `evaluation.fields` `warp`, and
`evaluation.metrics.MajorityVote`, so subjects, grid checks, warps and the vote are the
evaluator's own code).

Technology stack (locked): Python 3.8-compatible, TensorFlow 2.10 (Keras layers of the trained
models, tf.function per chunk shape), numpy, scipy.ndimage (via evaluation.fields), nibabel,
argparse, json. No new dependencies.

### 3.2 Loading (`inference/loader.py`)

1. `prefix = runtime.checkpoint.resolve_checkpoint(path)`; `cfg_path =
   source_config_path(prefix)`; missing config.json -> fatal (exit 2, "no config.json next to the
   checkpoint").
2. `model_cfg = ModelConfig.from_dict(config["model_config"])`; `args = config["args"]`;
   `level = int(args["level"])`; runtime knobs with defaults for absent keys: amp False,
   split_group_conv False, deform_decoder_dtype 'policy', fp16_act_clip 0.0,
   fp16_act_clip_scope 'deform_decoder', bn_epsilon_override 0.0. Unknown ModelConfig keys ->
   fatal (from_dict raises).
3. Branch / model check: deform requested but `model_cfg.deform_decoder` False -> exit 2.
4. Trainable model: global policy `mixed_float16` iff amp and `--precision checkpoint`, else
   `float32`; `GroupRegLevelModel(level, model_cfg, gradient_checkpointing='none',
   lean_sampler=True, split_group_conv=args.split_group_conv, fp16_act_clip=...,
   fp16_act_clip_scope=..., deform_decoder_dtype=..., bn_epsilon_override=...)
   .build_model(1)`. (Checkpointing changes no forward value. The lean sampler gives bitwise the
   same forward values as the default sampler (`layers.sample_voxel_coords`) and keeps its int64
   corner indices on the GPU, while the default sampler's int32 index tensors live in host memory
   on GPU, which at level 1 would move 8 corner tensors of 6.88 M elements per image through the
   3 CPUs in every warp; so inference always uses `lean_sampler=True`, independent of the
   checkpoint's value, and predict.json `knobs.lean_sampler` records `{"checkpoint": <value>,
   "used": true}`. The build pass at N = 1 also creates every variable with split_group_conv on,
   as in training.) The global policy
   is restored to float32 right after construction (layers keep their captured policy).
   `deform_decoder_dtype` under a float32 build (`--precision float32`, or `--device cpu` on an
   AMP checkpoint, or a non-AMP checkpoint): `'float16'` would raise in
   `GroupRegLevelModel.__init__` ("needs a float16 global policy"), so the loader passes
   `'policy'` instead, logs INFO and records
   `precision.deform_decoder_dtype = {"checkpoint": "float16", "used": "policy (float32 build)"}`.
   `'float32'` / `'float32_fine'` / `'policy'` are passed unchanged (no-ops under float32; the
   record then has `used` == `checkpoint`). Variables are float32 in every policy, so the restore
   is the same.
4b. Unchunked group-size bound: after the config is known, the loader computes
   `n_max(level, model_cfg, tasks, split_group_conv) = floor((2^31 - 1) / E_max)` with the
   section 4.6 function, here for the model's own call path (with the checkpoint's
   `split_group_conv`: an unsplit `GroupConvBlock` also builds concat(x, mean) at 2C channels).
   Default widths, level 1, split on: deform E_max = 440 M (the d1 skip concat
   concat(up_d1(d2), stem), 64 ch x 6.88 M voxels) -> n_max 4; linear E_max = 220 M (stem output,
   32 ch) -> n_max 9. `--mode unchunked` with N > n_max, or `check-streaming --n-subjects` >
   n_max (its unchunked reference), exits 2 before any model call: "unchunked mode at level L
   supports at most n_max subjects (largest tensor <name>, E_max elements per image); use --mode
   streamed or fewer --n-subjects". The bound is the minimum over the tasks run (both for
   affine+deform) and over all stages of the cascade (frozen stages use their own widths, level
   and split flag False; the checkpoint level dominates for default widths).
5. Frozen stages: level 2 -> `FrozenStage.from_config(model_cfg)`; level 1 ->
   `FrozenChain.from_config(model_cfg)` (both float32, built inside `_float32_policy`).
6. Restore: `tf.train.Checkpoint(model=m, **extra).restore(prefix)` with extra = {} (level 4),
   {'frozen_stage': stage.model} (level 2), {'frozen_stage_4': ..., 'frozen_stage_2': ...}
   (level 1); `expect_partial()` (optimizer, RNGs, best metrics not read) then
   `assert_existing_objects_matched()`; failure -> exit 2 with the TF message.
7. Fingerprints: level 2: `config.cascade.fingerprint` vs the frozen level-4 model; level 1:
   every `cascade.stages[k].fingerprint` vs the model of that level
   (`runtime.checkpoint.fingerprints_match`). Mismatch -> exit 2.
8. Step: `tf.train.load_variable(prefix, STEP_KEY)`.
9. `LoadedModel` = (level, model, frozen (None | FrozenStage | FrozenChain), model_cfg, knobs,
   step, prefix, run_dir, config subset, training group sizes).

### 3.3 Precision and BN at inference

- Device setup (`inference/predict.py: setup_device(args)`), the first TF calls of the process,
  before any op or model construction:
  - `--device gpu` (default): `configure_tf_precision(tf32=False, memory_growth=not
    --gpu-memory-limit-mb)`; with `--gpu-memory-limit-mb M` a logical-device limit on GPU 0 as
    main.py does. No visible GPU -> fatal, exit 2 ("no GPU visible; use --device cpu").
  - `--device cpu`: `tf.config.set_visible_devices([], 'GPU')` first, then
    `configure_tf_precision(tf32=False, memory_growth=False)` (TF32 off is a no-op on CPU but
    keeps one code path). Hiding the GPUs before initialisation means no CUDA context and no GPU
    memory is taken (a `tf.device` scope would still initialise the GPU). A RuntimeError from
    `set_visible_devices` (devices already initialised; only possible when called as a library
    after TF ops) -> fatal, exit 1 with the message. predict.json records `"device": "cpu"` and
    `visible_devices` = the names from `tf.config.get_visible_devices()`.
  - CPU precision: `--device cpu` with `--precision checkpoint` on an AMP checkpoint logs a
    WARNING and builds the trainable stage in float32 (`precision.trainable = "float32 (cpu)"`;
    `deform_decoder_dtype 'float16'` mapped to `'policy'` as in 3.2 step 4).
    Reason: mixed_float16 Conv3D on CPU is slow and fp16 CPU kernels for every op used (AvgPool3D,
    the sampler's gathers) are not verified; the result differs from the GPU AMP run only by fp16
    rounding. `gpu_peak_mb` is null on CPU; `peak_rss_gb` (from `resource.getrusage`) is recorded.
  - Tests use a different mechanism on purpose: the tf_gpu env crashes without a visible GPU, so
    the level-1 tests run inside a GPU process under `with tf.device('/CPU:0')` (section 10); that
    scope is a test device for placement only, not the CLI's CPU mode.
- Every model call uses `training=False`: BN normalises with its moving statistics (recalibrated
  and frozen ones for the m9 trainable stage), dropout off. The engine calls the blocks' own norm
  layers with `training=False`, so `bn_epsilon_override` and frozen statistics apply unchanged.
- Frozen stages are float32 always (as in training). The trainable stage runs in the
  checkpoint's policy by default (`--precision checkpoint`: mixed_float16 for amp runs,
  `deform_decoder_dtype` honoured), which reproduces the trainer's validation. `--precision
  float32` builds the trainable stage under float32 (more exact evaluation of the same weights;
  differs from the AMP validation by fp16 rounding only).
- Fields, integration, warps and matrix math are float32 (diffeo / transforms rules); the 4x4
  voxel matrices of the canonical conversion are formed in float64 numpy and cast to float32.
- fp16 headroom (streamed mode, fp16 blocks only): max |output| per block over all chunks is
  recorded in predict.json (`fp16_max_abs`, with ratio to 65504), because the m9 run went NaN by
  fp16 overflow after step ~5500.

### 3.4 Branch combinations (decision D1, Jay 2026-10-31)

Findings: rigid and affine are alternative heads trained on the same inputs; the deform branch
was trained on MNI-aligned images without a linear branch in front; nothing was trained in
sequence across tasks.

Supported set (exactly these four values of `--branches`; default `affine+deform`):

| `--branches` | Pipeline | fwd / inv |
|---|---|---|
| `rigid` | rigid cascade on the raw images | linear field of A_rigid |
| `affine` | affine cascade on the raw images | linear field of A_affine |
| `deform` | deform cascade on the raw images (inputs must already be MNI-affine-aligned, like training; Mindboggle is) | u, u_inv |
| `affine+deform` | affine cascade; full-res images resampled once into the affine central space (x_i o A_i^-1, trilinear, zero fill); deform cascade on them; one composed field | A^-1 o (id + u) and its inverse |

Rejected (Jay chose option A for each): `all`, `rigid+deform`, `rigid+affine` (and any other
string, including reordered spellings such as `deform+affine`). Reasons: rigid and affine are
alternative heads, so a rigid -> affine chain or "all three" has no trained meaning; the deform
branch never saw rigidly aligned inputs that keep per-subject scale / shear. The CLI uses
`choices=("rigid", "affine", "deform", "affine+deform")`; any other value is an argparse error
(exit 2) whose message lists the four values and says "rigid and affine are alternatives; the
deform branch was trained on affine-aligned inputs". The help of `deform` states that the inputs
must already be MNI-affine-aligned (as processed Mindboggle is); the script cannot check this and
does not try.

Branch semantics in code: `BRANCHES = {"rigid": ("rigid", None), "affine": ("affine", None),
"deform": (None, "deform"), "affine+deform": ("affine", "deform")}` -> (linear task or None,
deform or None). The runner runs the linear task first (if any), then the deform cascade on the
images resampled by the linear A (or on the raw images), then the output pass composes. The
Mindboggle cluster run uses `deform` (Jay: the images are already affine-aligned; rigid / affine
are tested later on another dataset), while the local tests cover all four values.

## 4. Streaming engine (`inference/streaming.py`)

### 4.1 Store

`Store` maps a name to a per-image array [N, ...] on the host. `RamStore`: `np.empty`
preallocated. `DiskStore`: `numpy.lib.format.open_memmap(<store-dir>/<name>.npy, 'w+')`;
`free(name)` deletes the file; the directory is removed at the end (also on error, `finally`).
dtype = the tensor's dtype (float16 activations stay float16). API: `alloc(name, shape, dtype)`,
`put(name, i0, chunk)`, `get(name, i0, i1)`, `free(name)`, `nbytes()`.
`--store auto` (default): RAM if the estimate (section 6) is <= 0.7 x the budget, budget =
`SLURM_MEM_PER_NODE` (MB) when set, else `MemAvailable` from /proc/meminfo; else disk in
`--store-dir` (default `$TMPDIR` when set, else `<out>/_stream_tmp`).

### 4.2 Per-block primitives (exactly the layers' math)

New refactor in `models/layers.py` (no behaviour change):

```python
class GroupConvBlock:
    def pre(self, x, training=None):            # norm -> act, no group concat
        return self._pre(x, training, concat=False)
    def conv_with_group_mean(self, x, mean, inner_clip=True):
        # conv(concat(x, mean)) as conv_a(x) + conv_b(mean) + bias (the current _split_call body,
        # same ops in the same order); inner_clip False = only the outer fp16 clip (the unsplit
        # path's clip semantics); no clip op at all when fp16_act_clip == 0.
```

`_split_call(x, training)` becomes `x = self._pre(...); mean = reduce_mean(x, 0, keepdims=True);
return self.conv_with_group_mean(x, mean)`, so the traced graph is unchanged (the existing golden
tests check this). The engine applies a block to a chunk as:

- group block: `y = block.conv_with_group_mean(block.pre(x, False), mean, inner_clip=block.split_group)`
  (stem: `pre` is the identity because the stem has no norm / act);
- non-group block (config B): `y = clip_fp16(block.conv(block.pre(x, False)), block.fp16_act_clip)`.

Lean sampler in the engine: every engine call of a diffeo function that takes `lean`
(`exp_and_inverse`, `compose_cascade`, `warp` (also the deform pre-warp, called as
`diffeo.warp(tf.cast(x, tf.float32), U, lean=lean)`, the body of `cascade.prewarp_deform` with
the flag added), `warp_field`, `compose`) passes `lean = model.lean_sampler` (True, 3.2 step 4),
for the frozen stages too (their models are built with the default False, but the flag is the
engine's choice per call and changes no value). `layers.transform_image` (linear pre-warps,
5.3 internal pulls) has no lean option and is called unchanged. The unchunked runner calls the
models' own `call` (which uses `model.lean_sampler` in the integrator) and `cascade` functions
unchanged; since lean and default are bitwise equal in the forward, test (b) stays exact.

The group mean of a block is accumulated as `S += reduce_sum(cast(pre(x_chunk), float32), 0)`
over all chunks (GPU tensor [1, X, Y, Z, C]), then `mean = cast(S / N, pre dtype)`. Inputs are
cast exactly as the model does (`tf.cast(images, model.compute_dtype)`, the decoder's per-stage
`tf.cast(x, block.compute_dtype)` and `tf.cast(skip, x.dtype)`, `svf_norm_act.compute_dtype`).

### 4.3 Pass schedule of one level model (deform task)

Notation: x = this level's MODEL INPUT, stored as `x_in` (float32 store, [N, X, Y, Z, 1]).
Rule: `x_in` is computed once per chunk in P0 and stored; P1-P11 read it (the stem is
recomputed from it in P1, P2, P10, P11); it is freed after P11 (after P6 for a linear task).
`x_in` is the level's input after the cascade pre-warp: deform `diffeo.warp(tf.cast(x_l,
float32), U_up, lean)` (the body of `cascade.prewarp_deform`) at levels 2 / 1 of a cascade,
linear `transform_image(x_l, A_prev^-1, spacing)` (`cascade.prewarp_linear`), and the pooled
or raw image x_l itself at the first stage (level 4, or a single-stage checkpoint). x_l is the
level's image pooled per chunk from the level-1 source store (raw images, or the
affine-resampled store for affine+deform). The values are those of the model's own path
(deterministic per-voxel ops), so storing instead of recomputing changes nothing; P0 also
accumulates S_x from the stored chunk. `pool2` only at level 1,
`pool3` at levels 2 and 1, `up_d2` at levels 2 / 1, `up_d1` at level 1 (the model's attributes).
Every pass iterates over chunks [i0, i1) and runs one tf.function per (pass, chunk length)
(at most two traces per pass: full chunk and the remainder).

| Pass | Per chunk | Stored | Accumulated |
|---|---|---|---|
| P0 | x_in = pre-warp / pool of the source chunk (see rule above); cast x_in as the model does | `x_in` | S_x (stem group input) |
| P1 | y0 = stem(x, mean_x); a = pool2(y0) or y0 | - | S_l2 = sum pre_l2(a) |
| P2 | recompute a; y = l2(a, mean_l2); b = pool3(y) or y | `l2` | S_l3 = sum pre_l3(b) |
| P3-P5 | y = lk(prev, mean_lk) | `l3`, `l4`, `l5` | S of the next block |
| P6 | y = l6(...) | `l6` | S_d5 = sum pre_d5(concat(up(y), l5)) |
| P7-P9 | y = dk(concat(up(prev), skip), mean_dk) | `d5`, `d4`, `d3`; free the used skip | S of the next stage |
| P10 | y = d2(concat(up_d2(d3), l2), mean_d2) | `d2`; free `l2`, `d3` | S_d1 = sum pre_d1(concat(up_d1(y), stem(x))) with the stem recomputed from x and mean_x |
| P11 | recompute stem; f = d1(concat(up_d1(d2), stem), mean_d1); v_raw = svf_head(svf_norm_act(f, False)) cast float32 | `v_raw`; free `d2`, `x_in` | S_v = sum v_raw |
| P12 | v = v_raw - S_v / N; (u_res, u_inv_res) = diffeo.exp_and_inverse(v, T); with previous-level fields U, U_inv (upsampled per chunk): (u, u_inv) = compose_cascade | `u`, `u_inv` (except at the last level, see 4.5) | - |

The stem output is never stored (at level 1 it is 32 channels at full resolution, 440 MB per image
in fp16); it is recomputed from the stored x_in and mean_x in P1, P2, P10 and P11 (one conv on 2
channels).
The d1 output is never stored (fused into P11).

Linear task: P0-P6 only (keeping only the latest encoder store), then the bottleneck [N,5,7,6,C]
is gathered into one tensor and `model.linear_decoder(bottleneck, task, training=False)` gives
A_raw [N,4,4] for all N at once (reductions, dense group blocks, heads: exactly the model's code
on the whole group). Centering / compose_linear also on all N matrices.

### 4.4 Cascade orchestration (`inference/runner.py`)

Raw full-res images I [N,160,224,192,1] float32 (host). Per level the input store is derived per
chunk: x2 = pool(x1), x4 = pool(x2) (`cascade.pool_to_coarser`).

Linear branch (task t), level-1 checkpoint: A4 = model4.centering(A_raw4) from x4;
x2' = transform_image(x2, A4^-1, 2 mm) per chunk; A2 = compose_linear(A_raw2, A4, model2.centering);
x1' = transform_image(x1, A2^-1, 1 mm); A = compose_linear(A_raw1, A2, model1.centering).
Level-2 checkpoint: the first two steps with model2 = trainable; level 4: A = centering(A_raw4).

Deform branch, level-1 checkpoint: level-4 pass schedule on x4 -> (u4, u4_inv) at level 4;
U4 = up(u4) per chunk; x2' = warp(x2, U4) (computed in the level-2 P0 and stored as that
level's `x_in`, 4.3); level-2 schedule -> compose_cascade(U4, U4_inv, u2,
u2_inv) stored at level 2; U = up(.) per chunk; x1' = warp(x1, U); level-1 schedule ->
compose_cascade(U, U_inv, u1, u1_inv). These are the exact ops of `FrozenStage.deform`,
`FrozenChain._deform` and `_cascade_val_deform_fn`, applied per image.

`affine+deform`: the affine branch first (A), then the deform branch with
x1 := transform_image(I, A^-1, 1 mm) (per chunk, into a new input store), then the canonical
composition (section 5).

Unchunked mode (`UnchunkedRunner`): the same orchestration with each level stage executed as one
direct call on all N (tensors on the device): linear `chain.linear(x, t)` / `stage.linear`,
`CS.prewarp_linear`, `model(x_in, task=t, training=False)["A_raw"]`, `CS.compose_linear`;
deform `chain.deform(x)` / `stage.deform`, `CS.prewarp_deform`, `model(x_in, 'deform',
training=False)`, `diffeo.compose_cascade` (= the trainer's `_cascade_val_fn` /
`_cascade_val_deform_fn` without the baseline extras). Each level stage is wrapped in one
`tf.function` (one trace per input shape). Limited to small groups by the int32 per-tensor
element limit: unchunked level-1 deform at default widths N <= 4 (E_max = 440 M, the d1 skip
concat), linear N <= 9 (the 32-channel stem); the loader enforces this with exit 2
(section 3.2 step 4b). GPU memory of the unchunked level-1 deform reference at N = 4 is a
forward pass only (no gradients, no Dice one-hots); it is measured in the cluster check
(predict.json of check-streaming, `gpu_peak_mb`).

### 4.5 Last level and output pass

At the checkpoint's level the last pass (P12) does not store u / u_inv for all N: per chunk it
upsamples them to level 1 if the checkpoint level is 2 or 4 (repeated
`diffeo.upsample_displacement`), applies the canonical conversion (section 5) on the device,
copies fwd / inv to the host and hands them to the writer (section 5.2). Linear-only branches run
the output pass from A alone.

### 4.6 Chunk sizes and the int32 limit

`run`: `--chunk-l1` (default 2), `--chunk-l2` (8), `--chunk-l4` (32); `0` = the whole group.

`check-streaming`: the same three flags, but their default is None, resolved per level after the
subjects are known as `chunk_l = min(cap_l, max(1, n_subjects - 1))`, so every level runs at
least two chunks and the last one is a remainder (N = 4 -> 3 + 1 at each level; N = 3 -> 2 + 1;
N = 2 -> 1 + 1). An explicit flag overrides the default for its level (`0` = whole group, which
check-streaming accepts but logs a WARNING "level L not chunked"). The resolver is
`inference/cli.py: resolve_check_chunks(n_subjects, flags, caps) -> {l1, l2, l4}` (pure
function, unit-tested: N = 4 without flags -> (3, 3, 3); explicit flags kept; a value above
cap_l capped). The resolved values and whether each came from the default or a flag go to the
JSON `chunks` {l1, l2, l4, source, capped}. Only the levels present in the checkpoint's cascade
are used (a level-2 checkpoint ignores `l1`).

The engine
computes, from the model config and level, the largest per-image tensor element count E_max over
every block input, concat, output and the warps (level 1 default widths: concat(up(d2), stem) =
64 ch x 6.88 M = 440 M); a chunk with k x E_max >= 2^31 is capped to floor((2^31 - 1) / E_max)
with a warning (recorded in predict.json `chunks.capped`). Default level-1 widths -> cap 4.
The function is `inference/streaming.py: max_elements_per_image(level, model_cfg, tasks,
split_group_conv)`, returning (E_max, tensor name). The engine always uses the split form
(`conv_with_group_mean`), so for the chunk cap it is called with `split_group_conv=True`; the
unchunked bound (3.2 step 4b) calls it with the checkpoint's flag. Both are unit-tested against
hand counts for default widths at levels 4, 2 and 1.

## 5. Canonical fields (`inference/canonical.py`)

### 5.1 Conversion

Internal conventions: arrays keep nibabel (i, j, k) order (export.py), so internal channel a =
grid axis a; level-1 voxels at 1 mm; centered mm x = (p - c) s, c = (n - 1) / 2 (layers.py);
A_i maps image i -> central (mm), images are pulled with A_i^-1; u maps central -> image coords
(warped = I o (id + u)), u_inv its exact inverse (exp(-v), compose_cascade). Atlas grid = central
grid = the processed grid.

With V = `tfm.index_to_mm_matrix(shape, spacing)` and Mv(B) = V^-1 B V (voxel-index matrix of a
centered-mm matrix B, `tfm.mm_to_voxel_matrix`), for atlas voxel p and subject voxel y:

| Branches | fwd_vox(p) (atlas grid) | inv_vox(y) (subject grid) |
|---|---|---|
| linear (rigid / affine) | Mv(A^-1) [p; 1] - p | Mv(A) [y; 1] - y |
| deform | u(p) | u_inv(y) |
| affine + deform | Mv(A^-1) [p + u(p); 1] - p | z + u_inv(z) - y, z = Mv(A) [y; 1], u_inv sampled trilinearly with border clamping (`layers.sample_voxel_coords(boundary='border')`, the `diffeo.compose` rule) |

Then fwd_mm[..., a] = fwd_vox[..., a] x spacing[a] (and inv). Stored [X,Y,Z,3] float32 with the
processed affine. In evaluator terms: warped_i(x) = I_i(x + fwd_i(x)) and A_i(y) = A(y + inv_i(y))
(evaluation/fields.py), convention flags `--components xyz --signs +++ --units mm`, no
`--swap-roles`. Grid requirement checked at load: shape == (160, 224, 192), spacing (1, 1, 1)
within 1e-3 (the model was trained at 1 mm); otherwise exit 2.

Training-grid check (`--grid-reference PATH`, optional; `inference/canonical.py:
load_reference_affine(path) -> (affine 4x4 float64, shape or None)`): shape and spacing alone
would accept an LR-mirrored grid (same shape, x axis reversed), on which the model would run on
mirrored anatomy without error. The `npy_v2` training arrays come from `image.nii.gz` files on
the corrected LUMIR target grid (`preprocessing/export.py`, x flip settled by
`build-target-grid --force-flip yes`), and `config_mindboggle.yaml` resamples onto the same
`target_dir`. Reference formats:
- path ending in `.json`: read with `json.load` (no import of `preprocessing`); key `affine`
  required (4x4 list of numbers, else exit 2 "grid reference has no 4x4 affine"); key `shape`
  optional (if present it must equal the processed shape, else exit 2). This is the
  `target_grid.json` written by `build_target_grid`.
- any other path: loaded with nibabel (`img.affine`, `img.shape[:3]` must equal the processed
  shape); unreadable -> exit 2 with the nibabel message.
The processed affine (the first subject's `image.nii.gz` affine, which `load_grid_data` /
the evaluator use for all outputs; `load_grid_data` already raises ValueError, mapped to exit 2,
if any subject's image or labels deviate from it by more than 1e-3) must satisfy max |affine - reference| <= 1e-3 over all 16
entries (the preprocessing `grid_tolerance_mm`); otherwise exit 2, ERROR naming both affines.
Without the flag the check is skipped with an INFO line "no --grid-reference; affine not checked
against the training grid". In every case predict.json (and the check-streaming JSON) records
`grid` {shape, spacing, affine (the processed affine), reference_path (or null),
reference_max_abs_diff (or null), checked (bool)}. The path is validated in the CLI (must be an
existing file, else argparse error, exit 2).

### 5.2 Writer, moved image, atlas

Per subject (CPU, while the GPU waits; sequential, simple): `fwd`, `inv` written with nibabel
(`<out>/<subject_dir>_fwd.nii.gz`, atomic: temp name + `os.replace`); moved = `evaluation.fields.warp(
image, fwd_vox, order=1)` (the evaluator's own warp, one trilinear interpolation of the original
image for every branch, so the moved file, the atlas and the evaluator all use the same warp);
warped labels = `F.warp(labels, fwd_vox, order=0)` kept in RAM (uint16, 1.4 GB for 101);
atlas sum += moved (float64). After all subjects: `atlas.nii.gz` = sum / N (float32),
`atlas_seg.nii.gz` = `evaluation.metrics.MajorityVote(warped_labels).full()` (ties to the
smaller label; the labels' dtype). Labels never enter the model. Writing that the moved file
comes from `fwd` is a layout choice, not evidence of the convention: check-convention on these
files always prefers the canonical flags by construction. The evidence is section 5.3.

gzip level: `--compress-level` (0-9, default 1, recorded in predict.json) is applied by setting
`nibabel.openers.Opener.default_compresslevel` before writing (exists in nibabel 3.1).

### 5.3 Internal consistency check (`inference/canonical.py`)

Purpose: tie the canonical files to the warp the model actually computed, using the model's own
TF ops (`models.layers.transform_image`, `models.diffeo.warp`, `models.diffeo.compose`) and not
the conversion formulas of 5.1.

Internal pull functions (TF, float32, on the device; inputs per subject at level 1, 1 mm; U1,
U1_inv = the internal deform fields at level 1, upsampled with `diffeo.upsample_displacement` for
level-2/4 checkpoints exactly as the output pass does):

| Branches | `internal_moved(V)` (atlas grid) | `internal_pull_inv(J)` (subject grid) |
|---|---|---|
| rigid / affine | `transform_image(V, A^-1, 1)` | `transform_image(J, A, 1)` |
| deform | `diffeo.warp(V, U1)` | `diffeo.warp(J, U1_inv)` |
| affine+deform | `diffeo.warp(transform_image(V, A^-1, 1), U1)` (the pipeline's two steps) | `transform_image(diffeo.warp(J, U1_inv), A, 1)` |

Checks per checked subject (all on the valid interior Omega, defined below):
- coordinate check (gated, every branch): C = the voxel-index coordinate volume [X,Y,Z,3]
  (`layers.index_grid`). Trilinear interpolation reproduces an affine function exactly, so
  `internal_moved(C)(p)` is the model's map p -> phi(p) with no interpolation error even through
  the two steps of affine+deform. Metric `coord_fwd_max_vox` = max |internal_moved(C) - (p +
  fwd_vox(p))|; likewise `coord_inv_max_vox` = max |internal_pull_inv(C) - (y + inv_vox(y))|.
  Gate 1e-3 vox. This catches a sign error, Mv(A) used for Mv(A^-1), swapped u / u_inv, or a wrong
  composition order.
- natural image (gated only for rigid, affine, deform): `moved_vs_internal_max_abs` = max
  |F.warp(I, fwd_vox) - internal_moved(I)| (scipy order-1 vs TF trilinear, one interpolation
  each); gate 1e-3 (intensities in [0, 1]). For affine+deform the internal path interpolates
  twice, so the value is recorded but not gated (the coordinate check covers that branch).
- inverse consistency (recorded in predict.json; gated in tests only for single-interpolation
  branches): rigid / affine: max |F.compose(fwd, inv)| (mm), test gate 1e-3 mm; deform: max
  |F.compose(fwd, inv) - diffeo.compose(U1_inv, U1)| (mm; both evaluate u(p) +
  u_inv_tri(p + u(p)) with border clamping), test gate 1e-3 mm. affine+deform: both numbers are
  recorded only. F.compose samples the stored inv at the off-grid point Mv(A^-1)(p + u(p)); on the
  grid, inv contains u_inv_tri(Mv(A) y'), a piecewise-trilinear field seen through a rotated grid,
  so its trilinear interpolation differs from the internal u_inv_tri(p + u(p)) by an
  interpolation error of the rotated u_inv. That is a property of storing the field on a grid,
  not of the conversion; the coordinate checks cover this branch exactly. No smoke bound is
  gated (YAGNI); the test prints the values.

Valid interior Omega: voxels p with 2 <= p_a <= n_a - 3 on every axis (the TF and scipy
zero-fill rules differ in the last voxel) where `internal_moved(ones)` (resp.
`internal_pull_inv(ones)`) equals 1 within 1e-5, i.e. every interpolation stage sampled inside
the volume. Omega's voxel count is recorded; a count below 50 % of the volume -> WARNING (the
check says little).

In the script: `--check-internal K` (default 2, 0 = off) checks the first K subjects in subject
order during the output pass (the per-chunk fields are on the device anyway; extra cost: about
four warps of a 3-channel and two of a 1-channel volume per subject). predict.json
`internal_check` = {k, gate {coord_vox: 1e-3, moved: 1e-3}, per subject {coord_fwd_max_vox,
coord_inv_max_vox, moved_vs_internal_max_abs, moved_gated, inv_consistency_max_mm,
interior_voxels}, passed}. Failure -> all files are still written, status "internal_mismatch",
ERROR log with the numbers, exit 5. The same functions are the reference of tests (b)-(d).

## 6. Memory and time estimates (level 1, default widths, N = 101, deform only)

Host (RAM store, trainable stage fp16): raw images 2.8 GB + labels 1.4 + masks 0.7 (from
`load_grid_data`) + level-1 `x_in` store 2.8 + encoder/decoder stores at their peak (l2 110 MB,
l3 28, l4 5, d3 14, d2 55 MB per image ~ 21 GB) + v_raw 8.3 GB + level-2 composed fields 2.1 GB
~ 40 GB. Input stores (rule of 4.3): the level-1 SOURCE for deform only is the raw image array
returned by `load_grid_data` (no copy; levels 2 / 4 are pooled from it per chunk), and the
level-1 model input `x_in` = warp(x1, U) is a separate stored float32 array (2.8 GB, freed after
P11); the level-2 / level-4 `x_in` stores are 0.35 / 0.04 GB. So the deform-only total above
(~40 GB) is: raw images 2.8 + labels 1.4 + masks 0.7 + `x_in` 2.8 + activation stores ~21 +
v_raw 8.3 + level-2 composed fields 2.1 + level-1 accumulators / small items ~1. The level-2
frozen stage (float32: l2 220 MB, d2 110 MB per image) peaks ~ 35 GB but is freed before level 1.
For affine+deform the affine-resampled level-1 source transform_image(I, A^-1) is materialised
once (2.8 GB) next to the raw images (which stay in RAM because the writer warps the ORIGINAL
images with fwd, 5.2) and is freed after the level-1 P0 has produced `x_in`; peak ~43 GB.
`--precision float32` (trainable stage float32: activation stores and v_raw unchanged in count,
activations doubled) ~ 63 GB; still RAM under `--store auto` with `--mem=96G` (0.7 x 96 = 67 GB),
otherwise disk. The lean sampler (3.2 step 4) keeps the warps' index tensors on the GPU,
so the per-warp host traffic of the default sampler does not apply. GPU: chunk 2 at level 1 ~ 6-10 GB (stem 2 x 440 MB, d1 inputs, integration of 2 fields,
accumulators <= 1.8 GB). Time: ~12 passes per level, each a forward of one block plus the
recomputed stem; expected 20-40 min for the model on an RTX 8000. Writing is sequential on one
core: per subject two gzip-compressed 83 MB fields, the moved image, a scipy trilinear warp and a
nearest warp of the labels, roughly 15-25 s, so 25-40 min for 101 subjects at
`--compress-level 1`. All of these are estimates; the measured values go to predict.json
(`timing_s`) and the report. `--device cpu` (fallback) is expected to be 10-50x slower for the
model part; not planned for the Mindboggle run.

## 7. CLI and validation (`inference/cli.py`)

```
python predict.py run --checkpoint P --processed-dir D --out O [--subjects-file F]
    [--label-name dkt31aseg] [--branches <see 3.4>] [--mode streamed|unchunked]
    [--device gpu|cpu] [--precision checkpoint|float32] [--chunk-l1 2] [--chunk-l2 8]
    [--chunk-l4 32] [--store auto|ram|disk] [--store-dir DIR] [--gpu-memory-limit-mb M]
    [--check-internal 2] [--compress-level 1] [--tta none] [--grid-reference R]
    [--overwrite] [-v]
python predict.py check-streaming --checkpoint P --processed-dir D --out JSON
    [--n-subjects 4] [--subjects-file F] [--branches ...] [--precision checkpoint|float32]
    [--device gpu|cpu] [--chunk-l1 C] [--chunk-l2 C] [--chunk-l4 C]   (default per level:
    min(cap, max(1, n_subjects - 1)), section 4.6) [--grid-reference R]
    [--tol-field-max-mm X] [--tol-field-mean-mm Y] [--tol-moved W] [--tol-matrix Z]
    [--tol-translation-mm T]
```

Tolerance flags default to None (= the row of the section 8 table for the precision used);
when given they must be floats > 0 (argparse error otherwise).

Validation (all before TF is imported; failure = argparse error, exit 2): checkpoint path exists
(prefix .index or directory); processed-dir is a directory; out is not an existing non-empty
directory with any of the output names unless `--overwrite`; chunk values integers >= 0;
`--n-subjects` >= 2; branches in the supported set; `--grid-reference` an existing file; `--tta` only `none` (help names
milestone 10); `--store-dir` creatable; `--gpu-memory-limit-mb` > 0; `--device cpu` with
`--gpu-memory-limit-mb` -> error; `--check-internal` integer >= 0 (values above N are clipped to
N after loading, logged INFO); `--compress-level` integer 0-9. After the config is read (before
any model call): unchunked group-size bound (3.2 step 4b) -> exit 2. After TF: subjects >= 2 (find_subjects), grid check (5.1),
input images finite (else exit 2 naming the subject), intensity range warning (max > 1.5 or
min < -1e-3: logged WARNING, recorded).

Exit codes: 0 ok; 2 invalid input / config / checkpoint / no GPU visible with `--device gpu`;
3 non-finite outputs; 4 check-streaming gate failed; 5 internal consistency check failed
(section 5.3); 1 unexpected error (traceback logged). If both 3 and 5 apply, 3 wins (non-finite
values make the internal numbers meaningless). Logging: `logging` INFO to
stderr (one line per pass and level with time and TF peak memory), WARNING for caps / ranges.

## 8. predict.json and checks

`predict.json` (written last, atomically; on failure a best-effort file with `status` "error" /
"nonfinite" and the message): `status`, `command` (argv), `git_commit` (see below), `checkpoint` {prefix,
step, run_dir, config path, level, frozen stage origins + fingerprints matched}, `model_config`,
`knobs` (amp, split_group_conv, lean_sampler {checkpoint, used}, deform_decoder_dtype,
fp16_act_clip, bn_epsilon_override, lowres_method), `precision` per stage, `branches`, `pipeline` (textual chain),
`global_transform`, `tta`, `n_subjects`, `subjects`, `training_group_sizes`, `mode`, `device`,
`chunks` {l1, l2, l4, capped}, `grid` (5.1: shape, spacing, affine, reference_path,
reference_max_abs_diff, checked), `store` {kind, dir, estimate_gb, peak_rss_gb}, `timing_s` (load,
read, each level stage, write, atlas, total), `gpu_peak_mb` per phase
(`tf.config.experimental.get_memory_info` after `reset_memory_stats`), `nonfinite` {per subject
fwd / inv / moved counts, total}, `stats` {max |fwd| mm, max |inv| mm, max |v| per level},
`fp16_max_abs` per block, `internal_check` (section 5.3), `compress_level`, `visible_devices`,
`convention` {"components": "xyz", "signs": "+++", "units": "mm",
"swap_roles": false}, `outputs` (file names), `warnings`.

`git_commit` (git may be missing inside tf10_nv.sif; never fatal): (1)
`subprocess.run(["git", "rev-parse", "HEAD"], cwd=<repo root>, capture_output=True, text=True,
timeout=10, check=True)` (py3.8 API), stripped stdout; on FileNotFoundError,
CalledProcessError, TimeoutExpired or OSError -> (2) read `<repo>/.git/HEAD`: a 40-hex line is the
commit; `ref: refs/heads/<b>` -> read `<repo>/.git/<ref>`, else look the ref up in
`<repo>/.git/packed-refs` (`<sha> <ref>` lines); (3) anything else -> `null`, WARNING
"git commit unknown", appended to `warnings`. predict.json also records `git_commit_source`
(`git`, `git_dir` or `none`). Unit-tested with the subprocess patched to raise
FileNotFoundError and a temp `.git` directory (loose ref, packed ref, missing ref).

Non-finite policy: any non-finite value in a field or moved image -> the files are still written
(for inspection), atlas files are not, status "nonfinite", exit 3.

Gate G1 (`check-streaming`): per branch, max / mean |streamed - unchunked| of fwd and inv (mm),
moved image, A (when linear; split into the 3x3 linear part dA_lin = max |dA[:3,:3]| and the
translation dA_t = max |dA[:3,3]| in mm). The bound set is chosen by the trainable stage's
precision actually used (`precision.trainable`):

| Quantity | float32 trainable stage | mixed_float16 trainable stage | flag |
|---|---|---|---|
| field max (fwd, inv) | 1e-3 mm | 0.1 mm | `--tol-field-max-mm` |
| field mean | (not gated) | 2e-3 mm | `--tol-field-mean-mm` |
| moved max | 1e-3 | 5e-3 | `--tol-moved` |
| dA_lin | 1e-5 | 1e-3 | `--tol-matrix` |
| dA_t | 1e-4 mm | 0.05 mm | `--tol-translation-mm` |

A flag given on the command line overrides that row for the run; the JSON records the bound set
name (`float32` / `mixed_float16`), the bounds used and every measured number, whether or not
they pass. Exit 4 if any bound is exceeded. The fp16 bounds are provisional (fp16 rounding of
the encoder group means): they are never used as the gate of the cluster job. The exactness
proof on the real checkpoint is the float32 run (section 9 step 1a, float32 bounds); the fp16
run (step 1b) is a non-blocking record whose measured maxima the report states and which can
later replace the provisional bounds.

The check-streaming JSON contains: `status` (pass / fail), `precision` (bound set name and
`precision.trainable`), `bounds`, `chunks` (4.6: l1, l2, l4, source per level, capped),
`n_subjects`, `subjects`, `branches`, per branch and per quantity the max / mean differences,
`gpu_peak_mb` {unchunked, streamed} (`get_memory_info` peak after `reset_memory_stats` before
each side; null on CPU), `timing_s` {unchunked, streamed}, `grid` (5.1), `git_commit`,
`checkpoint` {prefix, step}.

## 9. SLURM script `slurm/predict_mindboggle.sh`

Header: `-A qtim -p rtx8000 --gpus=1 -N 1 --ntasks-per-node=1 --cpus-per-task=3 --mem=96G
--time=05:00:00` (4 h for the prediction and writing estimates of section 6 plus three
check-streaming runs of a few minutes to ~15 min each), logs `$PROJ/logs/%x-%j.out/.err`. Env
overrides: CKPT (default `$PROJ/results/m9_level1_cascade/ckpt/best_deform`), NAME (default
`m9_best_deform_5000_deform`), BRANCHES (default `deform`), PROCESSED
(`$PROJ/data/processed/mindboggle`), OUT (`$PROJ/results/predict_mindboggle/$NAME`), N_CHECK (4),
CHECK_CHUNKS (default `--chunk-l4 3 --chunk-l2 3 --chunk-l1 3`), GRID_REF
(`$PROJ/data/target_grid/target_grid.json`, the `target_dir` of `config_mindboggle.yaml`),
EVAL_PAIRS (`none`), EXTRA. Every predict.py call passes `--grid-reference $GRID_REF`. Steps
inside tf10_nv.sif (`--nv`, `</dev/null`, TMPDIR = /scratch/$SLURM_JOB_ID):
1a. Gating exactness check (float32): `python predict.py check-streaming --checkpoint $CKPT
   --processed-dir $PROCESSED --grid-reference $GRID_REF --precision float32 --n-subjects
   $N_CHECK $CHECK_CHUNKS --branches $BRANCHES --out $OUT/check_streaming_f32.json || exit $?`.
   Float32 bound set (field max 1e-3 mm, moved 1e-3, dA_lin 1e-5, dA_t 1e-4 mm). Chunks 3 / 3 / 3
   at N = 4 give 3 + 1 at every level, so levels 4, 2 and 1 all stream through a full chunk and
   a remainder (the N = 101 run has remainders at every level too: 32 x 3 + 5, 8 x 12 + 5,
   2 x 50 + 1). Chunk 3 at level 1 is below the int32 cap of 4. The float32 unchunked level-1
   reference at N = 4 is a forward pass whose largest tensor is the d1 input [4, 160, 224, 192,
   64] float32 (7 GB, 440 M x 4 elements < 2^31); its `gpu_peak_mb` is recorded. `$N_CHECK` must
   be <= n_max = 4 (the loader rejects larger values with exit 2). OOM fallback (documented in
   the script header, not automatic): resubmit with `N_CHECK=3
   CHECK_CHUNKS="--chunk-l4 2 --chunk-l2 2 --chunk-l1 2"` (2 + 1 at every level; chunk 3 at N = 3
   would be one chunk); the gate is never skipped. The script expands `$CHECK_CHUNKS` unquoted on
   purpose (word splitting into flags; a comment says so), and the static test asserts that
   steps 1a-1c pass it and that 1a has `--precision float32` and `|| exit $?` while 1b / 1c end
   in `|| echo "WARNING`.
1b. fp16 record (non-blocking): the same command with `--precision checkpoint` (mixed_float16
   trainable stage, provisional bounds) and `--out $OUT/check_streaming_fp16.json || echo
   "WARNING: fp16 streaming check rc $? (non-blocking), see $OUT/check_streaming_fp16.json"`.
   Its measured maxima are the report's fp16 numbers.
1c. Linear-path record (non-blocking): the step-1a command with `--branches affine+deform` and
   `--out $OUT/check_streaming_affine_deform_f32.json || echo "WARNING: affine+deform streaming
   check rc $? (non-blocking) ..."`. It exercises the linear streaming (A4 -> A2 -> A1 and the
   affine resampling) on the real checkpoint without blocking the deform-only Mindboggle
   prediction (Jay's clarification 1); the report states its result either way.
2. `python predict.py run --checkpoint $CKPT --processed-dir $PROCESSED --grid-reference
   $GRID_REF --branches $BRANCHES --out $OUT/fields $EXTRA` (checkpoint precision, default
   chunks 32 / 8 / 2).
3. `python -m evaluation check-convention --fwd '$OUT/fields/{subject_dir}_fwd.nii.gz' ...
   --moved ... --out $OUT/convention.json`; then `python -m evaluation run --method $NAME --atlas
   ... --atlas-seg ... --components xyz --signs +++ --units mm --pairs $EVAL_PAIRS --workers 3
   --out $OUT/eval_template`. Step 3 confirms only that the files are readable with the
   canonical flags (by construction, see 5.2); the convention evidence of the cluster run is
   step 2's predict.json `internal_check` (exit 5 on failure stops the job before step 3).
Per Jay's D2, the full pairwise evaluation (10100 pairs, ~2 h on 8 CPUs) runs separately on `basic`
with the existing script, so the GPU allocation is not held idle (the script prints this command
at the end):
`METHOD=$NAME FWD=$OUT/fields/{subject_dir}_fwd.nii.gz INV=... ATLAS=... ATLAS_SEG=...
CONV="--components xyz --signs +++ --units mm" sbatch slurm/eval_mindboggle.sh`. Nothing is
submitted without Jay's approval (exact command, resources and squeue count in the request).

## 10. Tests

### 10.0 Environment prerequisite (first task of the plan)

The local tf_gpu env (python 3.9.13, numpy 1.23.4, scipy 1.13.0) has no nibabel (probed by the
v3 review: `import nibabel` -> ModuleNotFoundError); `bgr_preproc38` has 3.1.1 but no TF. Every
test that writes the synthetic NIfTI subjects, runs `predict.py run` in-process, or calls
`evaluation.evaluate.main` needs TF and nibabel in one process: (b) through the files, (c), (d),
the predict.json / CLI / nonfinite / internal-check unit tests, the `--device cpu` subprocess
test, and the existing `tests/test_evaluation.py` (which already fails at import in stage 2).

- Install, matching the container version: `conda activate tf_gpu && pip install
  nibabel==3.1.1` (pure Python, works with numpy 1.23). This changes Jay's local env, so the
  coder asks Jay first (workflow pause with the exact command) and does not install without his
  OK; the report records the install. After the install the coder checks
  `python -c "import nibabel, numpy; print(nibabel.__version__, numpy.__version__)"` in tf_gpu
  (unverified that 3.1.1 imports cleanly under numpy 1.23; expected, since numpy 1.23 still has
  the deprecated aliases). If it does not import, the coder reports the error and asks Jay again
  (the next candidate is the oldest nibabel that supports numpy 1.23, e.g. 3.2.2, recorded as a
  deviation); the tests are never weakened instead.
- Guard in `tests/run_tests_wsl.sh` stage 2, right after the version print:
  `python -c "import nibabel" || { echo "tf_gpu needs nibabel (pip install nibabel==3.1.1)"; status=1; }`.
  The suite continues (other modules still run) but ends with status 1, so a missing dependency
  is one clear line instead of import errors module by module.
- No `skipTest` for a missing nibabel anywhere: that would silently drop checks (b)-(d).
- scipy difference: tf_gpu has scipy 1.13, the container 1.4.1; `map_coordinates` edge handling
  changed in scipy 1.6. `evaluation.fields.warp` / `compose` use it, so values at the volume
  border can differ between the envs. The interior Omega of 5.3 (2-voxel margin plus the
  all-corners-valid mask) excludes the affected voxels, so no gate depends on the version; the
  report states the difference.

### 10.1 Test modules

All TF tests import `tests.tf_setup` first; TINY widths from `tests/runtime_helpers.py`;
random weights via the existing `randomize` helper so heads and SVF heads are non-zero; synthetic
full-res subjects written as NIfTI to a temp dir (N = 3, 160x224x192 image + labels, affine
diag(1,1,1)); level-2 synthetic groups made full-res by `np.repeat` x2 per axis (so pooling
returns the level-2 images exactly).

- (a) streaming: levels 4 and 2 (FrozenStage + trainable), branches rigid / affine / deform /
  affine+deform, chunks 1, 2, N; float32 and mixed_float16 trainable stage; split_group_conv on
  and off; group_config A and B; fp16_act_clip > 0 variant. Engine primitives alone: one block
  (`conv_with_group_mean` == `GroupConvBlock.call` with split on and off). Refactor guard: the
  test module embeds a verbatim copy of today's `_split_call` body (`models/layers.py` lines
  691-705 at the HEAD before the refactor; the copy is made by the coder BEFORE editing
  layers.py and carries a comment naming that commit) as a local function `_legacy_split_call(
  block, x, training)` that reads only `block._pre`, `block.conv` (kernel, bias),
  `block.strides` and `block.fp16_act_clip` (all kept by the refactor). It is compared
  bitwise (`assert_array_equal`) with the refactored `block._split_call` on a fixed random
  input, for float32 and mixed_float16 blocks, with `fp16_act_clip` 0 and 100 (the clip
  active on that input), strides 1 and 2.
  Lean flag: the engine with `lean=True` vs `lean=False` on one level-2 deform case gives
  bitwise-equal fields (guards the 4.2 claim).
  Level 1 (`TestStreamingLevel1`): a `FrozenChain` from TINY randomised level-4 / level-2 models
  (`ChainCase.frozen_models` / `chain` helpers of `tests/test_cascade.py`) plus a trainable
  level-1 copy built under float32 (not the AMP `level1_model` helper), N = 3, chunks 1 and 2,
  branches deform and affine+deform, both streamed and unchunked inside
  `with tf.device('/CPU:0'):` in the GPU process. Tolerances as levels 4/2 float32. Never skipped
  for memory; skip only when `tf.config.list_physical_devices('GPU')` is empty. The level-1
  models (frozen copies, chain, trainable level-1 model) are built inside the same
  `with tf.device('/CPU:0')` scope, so no level-1 variable or build activation lands on the
  6000 MB GPU. Runtime and the peak RSS (`resource.getrusage`) are printed to the test log.
- (b) trainer equivalence as in AC 2: level 2 with the cascade Trainer helpers of
  `tests/test_cascade.py` (`CascadeCase`); level 1 with `ChainCase.make1`-style construction but
  `amp=False` and the float32 level-1 model, under the CPU scope (models, chain and `Trainer`
  constructed inside it), with `val_pair_chunk=1` and `n_classes=N_CLASSES` (9, matching the
  synthetic labels of `tests/test_cascade.py`). Host memory at N = 3: the trainer's Dice path
  holds one-hot [3,160,224,192,9] float32 (0.74 GB) plus one warped pair at a time (0.25 GB),
  next to the level-1 activations (largest [3,160,224,192,16] float32 = 1.3 GB, a few alive at
  once) and the stored fields; estimated peak 6-9 GB, measured and printed (peak RSS) by the
  test. For each: script unchunked and
  streamed (chunk 1, chunk N) vs `Trainer._val_deform_fn(nv)` / `_val_fn(task, nv)` (reaching
  `_cascade_val_deform_fn` / `_cascade_val_fn`), Dice through `Trainer._deform_errors`, and the
  trainer's warped image at its level vs the script's internal warp; bounds S_exact for
  unchunked and S_stream for streamed (AC 0). Compared fields per level as listed in AC 2: at
  level 2 the runner's internal level-2 u / u_inv (before the output pass); at level 1 also the
  written `_fwd` / `_inv` files (read back, mm / spacing -> voxels) vs the trainer's u / u_inv,
  same bound set. For the file comparison the level-1 test runs `run_predict` (in-process, CPU
  scope) on the same synthetic subjects, so the files exist. In-process calls (here and in (c))
  use `run_predict(args, configure_device=False)`: `tests.tf_setup` has already configured TF in
  the test process (TF32 off, GPU limit), and `setup_device` after TF initialisation would raise;
  the CLI entry always passes `configure_device=True`. predict.json records the flag.
- (c) convention: for each branch at level 2 TINY with randomised heads scaled so that the
  rotation is >= 5 deg, translation >= 3 mm and |u| >= 2 vox (the test asserts these magnitudes
  first, so a degenerate draw fails loudly instead of passing trivially). Synthetic images for
  this test are textured so that check-convention can separate the conventions: Gaussian-smoothed
  white noise (sigma 1.5 vox, fixed seed, rescaled to [0, 1]) times an ellipsoid mask (semi-axes
  ~0.4 of the grid extent), different noise per subject; labels = a few ellipsoid shells inside
  the mask. The three 5.3 checks
  with the gates of AC 3, computed from the written NIfTI files and the 5.3 internal functions;
  plus one deliberately wrong conversion (fwd built with Mv(A) instead of Mv(A^-1), and u_inv
  swapped with u) must FAIL the coordinate check (proves the check has power). Then
  `predict.py run` in-process (CLI function, affine+deform), `evaluation.evaluate.main`
  check-convention (with `--out conv.json`) -> exit 0 and the canonical RESULT line, and from
  conv.json: `ranking[0]` is components xyz / signs +++ / no swap, `ranking[0].ncc -
  ranking[1].ncc > 0.002` and `ranking[0].ncc > identity_ncc + 0.002` (the evaluator's own
  margins, asserted explicitly so a near-tie fails with the numbers); `run --pairs 2` -> exit 0, finite
  summary; predict.json `internal_check.passed` true.
- (d) branch combinations and rejections; canonical conversion unit tests against independent
  numpy formulas (rigid A of a known rotation, a constant u, their composition and inverse
  consistency |compose(fwd, inv)| < 1e-4 mm for linear fields), at spacing 1 mm and on a
  synthetic 2 mm grid (converter level, so the mm scaling is tested).
- Device: `predict.py run --device cpu` in a subprocess with a TINY level-4 checkpoint, N = 2
  (the subprocess's tf_gpu env has a GPU, the flag hides it): asserts that the child's
  `tf.config.get_visible_devices()` (printed via `-v`) lists no GPU, predict.json records
  `"device": "cpu"`, and for an AMP checkpoint `precision.trainable == "float32 (cpu)"` and a
  WARNING line. The local tf_gpu env is known to crash in CPU-only mode with
  `CUDA_VISIBLE_DEVICES=""` (`scripts/run_in_tf_env.sh`); `set_visible_devices([], 'GPU')`
  keeps CUDA initialised and probably avoids it, but this is unverified. The coder runs the
  subprocess probe first. If the child dies by a signal (return code < 0) locally, the test
  calls `skipTest("tf_gpu env crashes without a visible GPU (rc ...)")` on exactly that signal
  (any other failure still fails); the mechanism stays as designed (correct for the cluster
  container). Independently of the subprocess, a unit test of `setup_device` with
  `tf.config.set_visible_devices`, `configure_tf_precision` and the memory-limit call patched
  (`unittest.mock`) asserts the call order (hide GPUs first, then `configure_tf_precision(
  tf32=False, memory_growth=False)`) for `--device cpu`, and for `--device gpu` the TF32-off call
  with memory growth / the logical-device limit.
- Loader precision: a TINY level-4 config with `amp True` and `deform_decoder_dtype 'float16'`,
  saved as a run checkpoint, loads with `--precision float32` (and with the CPU precision
  rule, called directly without the subprocess): no exception, the model's d-stage policies are
  float32, `precision.deform_decoder_dtype == {"checkpoint": "float16", "used": "policy (float32
  build)"}`, and every restored variable equals the saved one; with `--precision checkpoint` on
  the GPU the same checkpoint builds with float16 decoder stages.
- Unchunked bound: `max_elements_per_image` for default widths at level 1 gives (440.4 M,
  d1 skip concat) for deform and (220.2 M, stem) for linear -> n_max 4 / 9; with split off the
  deform bound is lower (group concat 2C); `--mode unchunked` with N = 5 on a level-1 config
  (bound computed from the config alone, no model build) and `check-streaming --n-subjects 5`
  exit 2 with the n_max message.
- Unit: store backends (RAM / disk, free, cleanup on error), chunk cap arithmetic, CLI errors and
  exit codes, predict.json keys, nonfinite path (a NaN injected into a field -> exit 3, atlas not
  written), internal-check failure (canonical conversion monkeypatched to add 0.01 vox to fwd -> exit 5, status "internal_mismatch"),
  config with absent knob keys (old e6815d9 config) loads with defaults, fingerprint mismatch ->
  exit 2.
- Slurm static test: header values (`--time=05:00:00`, `--mem=96G`, 3 CPUs, one GPU), `-A qtim`,
  rtx8000, `</dev/null` on every container call, no `sbatch` inside the script, step 1a with
  `--precision float32` and `|| exit $?`, steps 1b / 1c non-blocking, `$CHECK_CHUNKS` default
  `--chunk-l4 3 --chunk-l2 3 --chunk-l1 3` on 1a-1c, `--grid-reference` on every predict.py call.
- check-streaming end to end (TINY level-2 checkpoint, N = 3, no chunk flags): the JSON `chunks`
  are {l2: 2, l4: 2, source: default}, every level's streamed side ran two chunks (the engine
  logs / returns the chunk boundaries per level, asserted [(0, 2), (2, 3)]), `gpu_peak_mb` and
  `grid` are present.

Run order: the new modules first, then the full suite in the background.

## 11. Error handling summary

| Operation | Failure | Kind | Caller gets | Log |
|---|---|---|---|---|
| resolve checkpoint / config.json | missing, unreadable, unknown keys | fatal | exit 2 | ERROR |
| restore | missing variables, shape mismatch | fatal | exit 2 | ERROR with TF message |
| fingerprint | mismatch | fatal | exit 2 | ERROR naming the level |
| subjects / grid | < 2 subjects, wrong shape / spacing, subjects not on one grid | fatal | exit 2 | ERROR |
| `--grid-reference` | unreadable, no 4x4 affine, shape differs, affine differs > 1e-3 | fatal before any model call | exit 2 | ERROR with both affines |
| no `--grid-reference` | - | recoverable | check skipped, `grid.checked` false | INFO |
| input image | non-finite | fatal | exit 2 | ERROR naming the subject |
| chunk too large for int32 | - | recoverable | capped | WARNING |
| unchunked mode / check-streaming reference, N > n_max | int32 per-tensor limit | fatal before any model call | exit 2 naming n_max and the tensor | ERROR |
| `deform_decoder_dtype 'float16'` under a float32 build | would raise in the model constructor | recoverable | mapped to `policy`, recorded | INFO |
| GPU OOM | ResourceExhaustedError | fatal | exit 1, message suggests a smaller --chunk-l1 or --device cpu | ERROR |
| disk store | ENOSPC | fatal | exit 1, store dir cleaned | ERROR |
| outputs non-finite | - | fatal after writing fields | exit 3 | ERROR with counts |
| check-streaming gate | exceeded | fatal | exit 4 | ERROR with the numbers |
| internal consistency (5.3) | coordinate or moved gate exceeded | fatal after writing | exit 5, status "internal_mismatch" | ERROR with the numbers |
| `--device gpu`, no GPU | no physical GPU | fatal | exit 2 | ERROR suggesting `--device cpu` |
| `--device cpu` setup | GPUs already initialised | fatal | exit 1 | ERROR |
| interior too small (5.3) | Omega < 50 % of voxels | recoverable | numbers recorded | WARNING |

Invariant owners: group-statistic exactness = `inference/streaming.py` (engine) with tests (a);
canonical convention = `inference/canonical.py`, verified against the model's own warp ops by
the internal check (5.3, owned by `inference/canonical.py`, enforced at run time in
`inference/predict.py` and in tests (b)-(c)); model function identity = `models/layers.py`
refactor guarded by an explicit bitwise test of `_split_call` plus the existing goldens; device
and precision = `inference/predict.py: setup_device`, because it must run before any TF op.

## 12. Decisions recorded

- D1 (Jay, 2026-10-31): branch set = rigid, affine, deform, affine+deform. `all` dropped,
  rigid+deform rejected, rigid -> affine rejected (option A for each).
- D2 (Jay, 2026-10-31): the GPU job runs the equivalence check, the prediction (deform only for
  Mindboggle), check-convention and a template-only evaluation (`--pairs none`); the full
  10100-pair evaluation runs separately on `basic` with `slurm/eval_mindboggle.sh`.
- Cluster submissions: each one needs Jay's approval (exact sbatch command, partition, GPU, CPUs,
  memory, time limit and the current `squeue -u jn85` count); the 2-job limit is at his
  discretion.

## 13. Responses to design review v1 (CHANGES_REQUESTED)

1. HIGH, circular convention check: addressed. New section 5.3 checks the canonical fields
   against the model's own TF warp ops (`transform_image`, `diffeo.warp`, `diffeo.compose`). The
   main gate uses a coordinate volume instead of an image: trilinear interpolation of an affine
   function is exact, so the internal map is read off directly, also through the two
   interpolation steps of affine+deform (a natural-image 1e-3 bound cannot hold there because
   the internal path interpolates twice; the natural-image check is gated only for
   single-interpolation branches). Inverse checks: coordinate pull on the subject grid, and
   inverse consistency compared with the internal `diffeo.compose(U1_inv, U1)` (an absolute
   0.05 mm bound would test the random heads' inverse consistency, not the conversion). Test (c)
   also proves the check has power with a deliberately wrong conversion. "moved equals
   F.warp(image, fwd)" is removed as acceptance evidence (5.2 states why). predict.json records
   `internal_check` for `--check-internal K` subjects (default 2), and the cluster job stops with
   exit 5 before check-convention if it fails (section 9).
2. MEDIUM, level-1 paths untested locally: addressed. Level-1 streamed vs unchunked and the
   level-1 trainer equivalence run under `tf.device('/CPU:0')` in the GPU test process, float32,
   N = 3, chunks 1 and 2, deform and affine+deform; no skip for memory (AC 1, AC 2, 10 (a)/(b)).
   The trainable level-1 model is built under float32 (the existing `level1_model` helper builds
   it under mixed_float16, which would need fp16 CPU kernels).
3. MEDIUM, mode and Dice tolerance of (b): addressed as proposed (AC 2): both modes, unchunked
   1e-6 and exact Dice, streamed 1e-5 vox / A 1e-6 / Dice <= 1e-4, reference functions named,
   plus the trainer's warped image at its level vs the script's internal warp (1e-5).
4. MEDIUM, CPU mechanism: addressed (3.3): `tf.config.set_visible_devices([], 'GPU')` as the
   first TF call, float32 trainable stage with a WARNING for AMP checkpoints, recorded in
   predict.json, subprocess test. No visible GPU with `--device gpu` -> exit 2.
5. NIT, rounding sources: addressed (2.3, four sources listed).
6. NIT, rejected-value list: addressed (F3, AC 4: any value other than the four).
7. NIT, write time: addressed (section 6: 25-40 min estimate, measured values reported;
   `--compress-level`, default 1, recorded).
8. NIT, mm vs vox: addressed (AC 3 note; the converter test uses a 2 mm grid, AC 5 / 10 (d)).
   The report will state that check-convention cannot distinguish units on a 1 mm grid.

Other unverified items from the review: the loader fails fast (exit 2, naming the key) if any
config key listed in 2.2 that the loader needs is missing; the GPU memory / time estimates stay
estimates until the cluster run measures them.

## 14. Responses to design review v2 (CHANGES_REQUESTED, 0 HIGH, 4 MEDIUM, 5 NIT)

1. MEDIUM, affine+deform inverse-consistency gate infeasible: addressed as proposed. AC 3 (iii)
   and 5.3 gate inverse consistency only for rigid / affine (|F.compose(fwd, inv)| <= 1e-3 mm)
   and deform (vs `diffeo.compose(U1_inv, U1)`, <= 1e-3 mm). affine+deform records both numbers
   without a gate; 5.3 states why (second interpolation of the rotated u_inv at off-grid points)
   and that the exact coordinate checks cover the branch. No smoke bound (it would test the random
   heads, not the conversion).
2. MEDIUM, G1 without an fp16 matrix bound; affine+deform check blocking the deform-only run:
   addressed. Section 8 now has a bound table per trainable-stage precision, with mixed_float16
   dA_lin <= 1e-3 and dA_t <= 0.05 mm, new flags `--tol-moved` and `--tol-translation-mm`, all
   values recorded either way. Section 9 step 1: only the `$BRANCHES` check gates (`|| exit $?`);
   the affine+deform check runs after it with `|| echo "WARNING ..."` and its JSON is kept.
3. MEDIUM, `deform_decoder_dtype 'float16'` crash under a float32 build: addressed. 3.2 step 4
   maps `'float16'` -> `'policy'` for every float32 build (`--precision float32`, CPU on an AMP
   checkpoint, non-AMP checkpoint), logs INFO, records `precision.deform_decoder_dtype`; other
   values pass unchanged. 3.3, the error table and a unit test (section 10 "Loader precision")
   cover it.
4. MEDIUM, wrong unchunked group-size limit: addressed. 4.4 now says level-1 default widths
   deform N <= 4 (d1 skip concat, 440 M elements per image), linear N <= 9 (stem). New 3.2 step 4b
   computes n_max from `max_elements_per_image` (4.6; with the checkpoint's split flag for the
   model's own call path, minimum over tasks and cascade stages) and exits 2 for
   `--mode unchunked` / `check-streaming` above it, before any model call. Section 9 notes
   N_CHECK 4 = n_max. Unit tests in section 10 ("Unchunked bound").
5. NIT, level-1 CPU test memory: addressed. Level-1 trainer test uses `val_pair_chunk=1`,
   `n_classes=9`; models, chain and Trainer are built inside the CPU scope; the estimate is
   restated (6-9 GB) and the peak RSS is printed by the test.
6. NIT, tf_gpu CPU-only crash: addressed. The coder probes the subprocess first; a signal death
   locally -> `skipTest` on exactly that signal; plus a mocked call-order unit test of
   `setup_device` for both devices.
7. NIT, check-convention margins on synthetic images: addressed. Test (c) uses textured images
   (smoothed noise sigma 1.5 vox x ellipsoid) and asserts the ranking margins from the
   `--out` JSON (`ranking`, `identity_ncc`).
8. NIT, determinism for the 1e-6 comparison: addressed. AC 2: both sides under `_OpDeterminism`
   (`tests/test_trainer.py`), one `tf.function` per stage in the unchunked runner (4.4), fallback
   to 1e-5 with the measured value reported.
9. NIT, frozen reference for the `_split_call` bitwise test: addressed. 10 (a) embeds a verbatim
   copy of the current body (copied before the refactor, commit named in a comment), compared
   bitwise in float32 and mixed_float16, clip 0 and 100, strides 1 and 2.

No finding needed a decision from Jay: the branch set (D1) is unchanged, and every group
interaction remains exactly decomposable (section 2.3).

## 15. Responses to design review v3 (CHANGES_REQUESTED, 1 HIGH, 1 MEDIUM, 4 NIT)

1. HIGH, no nibabel in tf_gpu: addressed as proposed. New section 10.0 (first task of the plan):
   `pip install nibabel==3.1.1` into tf_gpu only after Jay's OK (the coder pauses the workflow
   with the exact command), an import check after the install with a named fallback if it fails
   (ask Jay again, never weaken tests), a stage-2 import guard in `tests/run_tests_wsl.sh` that
   sets status 1 with one clear line, no `skipTest` for missing nibabel, and the scipy 1.13 vs
   1.4.1 edge-handling note (Omega excludes the affected voxels). AC 6 names the precondition;
   the files table lists the runner change.
2. MEDIUM, contradictory streamed tolerances: addressed. New AC 0 defines three bound sets:
   S_stream (streamed vs any float32 reference: fields 1e-4 vox, A 1e-5, Dice 1e-4, warped /
   moved image 1e-4), S_exact (unchunked vs trainer: 1e-6, Dice identical, warped image 1e-5,
   with the fallback to 1e-5 / Dice 1e-6), S_fp16 (unchanged). AC 1 and AC 2 reference them, so
   streamed vs unchunked and streamed vs trainer use the same bounds. Section 10 (b) refers to
   AC 0. The tolerances in the v1 response 3 (section 13) are superseded by AC 0.
3. NIT, lean sampler: addressed. 3.2 step 4 builds the trainable model with `lean_sampler=True`
   (bitwise-equal forward, int64 indices on the GPU); 4.2 lists every engine call that passes
   `lean=model.lean_sampler` (exp_and_inverse, compose_cascade, warp incl. the deform pre-warp,
   warp_field, compose; `transform_image` has no option); predict.json `knobs.lean_sampler`
   records checkpoint vs used; a bitwise lean on/off engine test is added to 10 (a); section 6
   notes the removed host traffic.
4. NIT, git in the container: addressed in section 8 (git rev-parse with timeout -> `.git/HEAD`
   loose / packed refs -> null with a WARNING; `git_commit_source`; unit test).
5. NIT, rigid det check: addressed in AC 4 (the runner's R from Mv(A^-1): |det - 1| and
   max |R^T R - I| <= 1e-5; the least-squares fit of p + fwd_vox(p) on Omega agrees with R to
   1e-4 and passes both checks at 1e-4).
6. NIT, input-store wording: addressed in section 6 (deform only: the level-1 input store is the
   raw image array, no copy, ~37 GB; affine+deform materialises only the resampled store next to
   the raw images, which the writer needs, ~40 GB; pre-warped cascade inputs are per chunk).

No finding needed a decision from Jay at design level: the branch set (D1) is unchanged, every
group interaction remains exactly decomposable (section 2.3), and the nibabel install is a local
environment change the coder asks Jay about before making it (section 10.0).

## 16. Responses to design review v4 (CHANGES_REQUESTED, 0 HIGH, 2 MEDIUM, 3 NIT)

1. MEDIUM, cluster gate does not stream at levels 4 / 2: addressed as proposed. check-streaming
   has its own chunk defaults `min(cap_l, max(1, n_subjects - 1))` per level (4.6, pure resolver
   `resolve_check_chunks`), so N = 4 resolves to 3 + 1 at every level and every level runs a
   remainder chunk; explicit flags override. The JSON records `chunks` with the source per level.
   Section 9 passes `--chunk-l4 3 --chunk-l2 3 --chunk-l1 3` explicitly (`CHECK_CHUNKS`); the
   N_CHECK = 3 OOM fallback comes with chunks 2 / 2 / 2. Tests: resolver unit test (AC 5) and a
   check-streaming end-to-end test asserting two chunks per level (10.1).
2. MEDIUM, fp16 bounds as the only real-checkpoint gate: addressed as proposed. Step 1a gates on
   `--precision float32` with the float32 bound set and records `gpu_peak_mb` of both sides
   (OOM fallback N_CHECK = 3, never skipping the gate); step 1b runs the same check at checkpoint
   precision (fp16) as a non-blocking record; step 1c runs the affine+deform check at float32,
   non-blocking. Section 8 states that the provisional fp16 bounds never gate the cluster job and
   lists the check-streaming JSON keys. Time limit raised from 4 h to 5 h. AC 7 updated.
3. NIT, x_in stored vs recomputed: addressed. 4.3 states one rule: each level's model input
   (after the cascade pre-warp, or the pooled / raw image at the first stage) is computed once per
   chunk in P0, stored as `x_in` (float32), read by P1-P11 and freed after P11 (P6 for linear).
   Section 6 counts it: ~40 GB deform only, ~43 GB affine+deform peak (the affine-resampled
   source is freed after the level-1 P0), ~63 GB with `--precision float32` (still RAM under
   `--store auto` at `--mem=96G`).
4. NIT, which fields test (b) compares: addressed in AC 2 and 10 (b): level 2 compares the
   runner's internal level-2 fields (before the output pass); level 1 additionally compares the
   written `_fwd` / `_inv` files, converted to voxels, with the trainer's u / u_inv, within the
   same bound set; linear branches compare A. In-process `run_predict` calls in tests skip
   device setup (`configure_device=False`), because `tests.tf_setup` has configured TF already.
5. NIT, LR-mirrored grid accepted: addressed. New optional `--grid-reference` (target_grid.json
   via `json.load`, key `affine` required, `shape` optional; or any NIfTI via nibabel): the
   processed affine must match within 1e-3 (else exit 2 with both affines); absent -> INFO and
   skipped. predict.json and the check-streaming JSON record `grid` (processed affine, reference
   path, max diff, checked). `slurm/predict_mindboggle.sh` passes
   `$PROJ/data/target_grid/target_grid.json` (the `target_dir` of `config_mindboggle.yaml`) on
   every call. Unit tests: json and NIfTI reference, mirrored affine -> exit 2, absent flag.

No finding needed a decision from Jay: the branch set (D1) is unchanged and every group
interaction remains exactly decomposable (section 2.3).
