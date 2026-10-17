# Milestone 3 report: group blocks, encoder, linear decoder, heads, centering

Status: implemented and tested locally (WSL, GTX 1080 8 GB). Not committed (per instructions). Full test output: `test-log.txt` in this folder (recorded after the last source edit, 2026-10-17 17:05-17:08).

## Files

| File | Change |
|---|---|
| `models/layers.py` | Added the group block layers after `transform_image` (which is unchanged) |
| `models/group_net.py` | New: `GroupEncoder`, `LinearDecoder`, `GroupRegLevelModel` |
| `models/config.py` | New: `ModelConfig` dataclass (no TF import) |
| `tests/test_group_layers.py` | New: 13 tests (blocks against numpy references, norms, BN freeze) |
| `tests/test_group_net.py` | New: 17 tests (shapes, levels, copy, identity, equivariance, group flow, gradients, dynamic N, mixed precision, BN freeze, dropout, parameter counts, config, smoke training) |
| `.agents/tasks/milestone-3/record_tests.sh` | Helper that runs `tests/run_tests_wsl.sh` and writes `test-log.txt` |

## Public API

`models.layers` (group blocks):
- `KaimingNormal(negative_slope=0, seed=None)`: VarianceScaling(2/(1+a²), fan_in, untruncated normal), as in the rigid project.
- `AffineParams`: per-channel gamma (ones) and beta (zeros).
- `BatchNormWithSeparateAffine(momentum=0.99, epsilon=1e-5, stats_frozen=False)`: Keras BN with `center=scale=False`, then `AffineParams`. If `stats_frozen` is set, it normalizes with the moving statistics and never updates them, even with `training=True`. gamma/beta keep training.
- `InstanceNormWithAffine(epsilon)`: normalizes per image and channel over the spatial axes, in float32. No tensorflow_addons needed.
- `make_norm('batch'|'instance'|'none')`, `set_bn_stats_frozen(layer, frozen)` (returns the number of BN layers it set).
- `group_mean_concat(x)`: concat(x_i, mean over axis 0), broadcast with `tf.shape`, so N can be dynamic and the channel count stays static.
- `GroupConvBlock(filters, kernel_size=3, strides=1, group=True, norm='batch', pre_activation=True, negative_slope=0, ...)`: norm → act → [group concat] → Conv3D ('same', bias, Kaiming). `pre_activation=False` gives the stem (no norm/act).
- `GroupResBlock(...)`: block1 (carries the stride) → block2, plus a skip. The skip is the identity, or a 1×1 Conv3D with the stride when channels or stride change.
- `GroupDenseBlock(units, group=True, norm='batch'|'none')`: norm → act → [concat(h_i, mean h)] → Dense.
- `NormAct(norm)`: norm → act.
- Every group block has `group_overhead_params()`, the size of the kernel half that multiplies the group mean.

`models.config.ModelConfig` defaults: `filters=(32,64,128,192,256,384)`, `reductions=(192,96,48,24)`, `dense=(2048,512)`, `dropout=0.15`, `group_config='A'`, `dense_group_concat_first=False`, `norm='batch'`, `dense_norm='batch'`, `block='single'`, `negative_slope=0.0`, `kernel_size=3`, `bn_momentum=0.99`, `bn_epsilon=1e-5`, `centering_iters=4`, `translation_scale=1.0`, `seed=None`. Also `validate()`, `to_dict()`, `from_dict()` (rejects unknown keys) and `replace()`.

`models.group_net`:
- `GroupEncoder(level, config)`: `call(x, training)` returns `(bottleneck, skips)`, with skips = [stem, L2, L3, L4, L5, L6].
- `LinearDecoder(config)`: `call(bottleneck, task='rigid'|'affine', training)` returns `(A_raw, params, {'dense', 'head_input'})`.
- `GroupRegLevelModel(level, config)`: `call(images, task, training)` returns a dict with `A_raw`, `A` (centered), `params` [N,13|19], `bottleneck`, `skips`, `dense` (pre-dropout features) and `head_input` (post-dropout). Helpers:
  - `build_model(n=1)`: one inference pass on zeros, which creates every variable;
  - `copy_weights_from(other)`: copies all variables, BN statistics included, matched by order and checked by shape and scope-local name; raises on any mismatch;
  - `count_params_by_part()`;
  - `set_bn_stats_frozen(frozen)`.
  - `task='deform'` raises `NotImplementedError`. `self.deform_decoder = None` is the milestone 6 hook, and it reads `skips`.

## Architecture (per level; output shapes [N, X, Y, Z, C]; trainable parameters)

| Layer | Op | Level 4 | Level 2 | Level 1 | Params A | Params B |
|---|---|---|---|---|---|---|
| stem | Conv3D 32, 3³, no norm/act (A: concat(image, mean image)) | 40×56×48×32 | 80×112×96×32 | 160×224×192×32 | 1,760 | 896 |
| pool2 | AvgPool 2, parameter-free | — | — | 80×112×96 | 0 | 0 |
| L2 | block 64 | 40×56×48×64 | 80×112×96×64 | 80×112×96×64 | 110,720 | 55,424 |
| pool3 | AvgPool 2, parameter-free | — | 40×56×48 | 40×56×48 | 0 | 0 |
| L3 | block 128 | 40×56×48×128 | same | same | 442,624 | 221,440 |
| L4 | block 192, stride 2 | 20×28×24×192 | same | same | 1,327,552 | 664,000 |
| L5 | block 256, stride 2 | 10×14×12×256 | same | same | 2,654,848 | 1,327,744 |
| L6 | block 384, stride 2 (bottleneck) | 5×7×6×384 | same | same | 5,309,312 | 2,655,104 |
| **encoder** | | | | | **9,846,816** | **4,924,608** |
| reduction 0-3 | 1×1 group blocks 192, 96, 48, 24 (both configs) | 5×7×6×24 | same | same | 148,416 + 37,344 + 9,456 + 2,424 | same |
| flatten | | 5,040 | same | same | 0 | 0 |
| dense_0 | BN(5040) → act → Dense 2048 (no concat by default) | 2048 | same | same | 10,334,048 (Dense 10,323,968) | same |
| dense_1 | BN → act → concat(mean) → Dense 512 | 512 | same | same | 2,101,760 (Dense 2,097,664) | same |
| head_norm_act | BN → act, then Dropout 0.15 | 512 | same | same | 1,024 | same |
| rigid_head | Dense 13, float32, zero kernel, identity bias | 13 | same | same | 6,669 | same |
| scale_shear_head | Dense 6, float32, zeros (affine only) | 6 | same | same | 3,078 | same |
| **linear decoder** | | | | | **12,644,219** | same |
| centering | GroupCentering (4 iterations), float32 | N×4×4 | same | same | 0 | 0 |
| **total trainable** | | | | | **22,491,035** | **17,568,827** |

Non-trainable (BN moving statistics): 17,984 in every configuration. Parameters are identical at all three levels (tested: same scope-local names, shapes and counts).

Parameter counts (from `test-log.txt`):

| | A | B | A + `dense_group_concat_first` | B + flag |
|---|---|---|---|---|
| encoder | 9,846,816 | 4,924,608 | 9,846,816 | 4,924,608 |
| first Dense (5040 → 2048) | 10,323,968 | 10,323,968 | 20,645,888 | 20,645,888 |
| linear decoder | 12,644,219 | 12,644,219 | 22,966,139 | 22,966,139 |
| group overhead, encoder | 4,922,208 | 0 | 4,922,208 | 0 |
| group overhead, linear decoder | 1,146,496 | 1,146,496 | 11,468,416 | 11,468,416 |
| total trainable | 22,491,035 | 17,568,827 | 32,812,955 | 27,890,747 |

The linear decoder without group overhead is 11,497,723 (≈ 11.5 M), and the default overhead is 1.15 M, matching design v3.2. The flag adds 10.3 M.

## Decisions taken

1. **Translation scaling.** The head's 3 translation outputs are in mm, multiplied by a fixed `translation_scale`, which defaults to **1** (raw mm). Conditioning note: under the point loss, a unit change of a 10D rotation entry moves brain voxels by roughly the brain radius (~60 mm), while a unit translation output moves them by 1 mm. So translations have far lower curvature per output unit. A value around 10 would even this out somewhat. I left it at 1 because the rigid project regressed raw translations successfully. It's a one-line config change if milestone 4's overfit shows slow translation convergence. Tested: the scale applies only to the translation outputs.
2. **BN placement relative to the group mean** (design 4.1): norm → act → mean over N of the *normalized, activated* features → concat → conv. In training mode, BN statistics come from the N images of the single group, over voxels as well. For dense blocks they come from the N vectors.
3. **Pre-head norm-act.** I added this; the spec doesn't list it explicitly. The blocks are pre-activation, so the 512-unit output of dense_1 is linear. A final BN → act (1,024 parameters) feeds dropout and the heads. This mirrors the rigid project's Dense → act → dropout → head. Without it, the head would be a linear map of a linear map. Likewise, the flatten feeds the raw output of the last 1×1 conv, and dense_0's own BN → act normalizes it (BN over 5,040 features, 10,080 parameters).
4. **Activation**: `ReLU(negative_slope)`, default 0.0, which is the rigid project's `--negative-slope` default. Kaiming init uses the same slope.
5. **Residual flag**: applies to L2-L6. The stem and the 1×1 reductions are always single convs.
6. **Both heads are built in `LinearDecoder.build`**, so the variable set (checkpoints, copying, counts) doesn't depend on which task ran first. In a rigid step the scale/shear head gets a `None` gradient (it's unused), so the training loop must skip `None` gradients, as the smoke test does.
7. **`stats_frozen` is a Python attribute read at trace time.** Re-trace the train step after toggling it. The level-transition code (milestone 8) builds new steps anyway.
8. **Weight copying** is by variable order, verified by shape and by name with the uniquified top-level model scope stripped. Explicit sublayer names (`encoder/l2/...`) make those names stable.
9. **Dtypes under `mixed_float16`**: inputs are cast to float16 at the model input, and conv/dense compute is float16. Heads (`dtype='float32'`), centering and all matrix math are float32, and every variable is float32. The group mean is a `reduce_mean` in the compute dtype (float16 under mixed precision).

## Test results

`wsl -d Ubuntu bash tests/run_tests_wsl.sh` (via `record_tests.sh`):
- Python 3.8 syntax check in `bgr_preproc38`: py_compile and `ast feature_version=(3, 8)` OK on 22 files; the numpy reference self-check is OK.
- unittest in `tf_gpu` (Python 3.9.13, TF 2.10.0, GTX 1080): **94 tests, OK, 1 skipped** (the DALI pipeline test needs nvidia.dali). 30 of them are new. Wall time 164 s.

What the new tests show:
- **Shapes** at levels 4, 2 and 1: bottleneck 5×7×6×384, every skip shape per the design 4.3 table, A_raw/A [N,4,4], params [N,19].
- **Levels**: identical parameter names, shapes and counts at 4, 2 and 1. `copy_weights_from` reproduces the outputs bit-exactly between two level-4 models (after moving the BN statistics and setting nonzero heads). Level 4 → 2 copies every variable equally and runs. A copy between config A and config B is rejected.
- **Identity at init**: A_raw = A = I at all levels (atol 1e-6).
- **Permutation equivariance** (configs A and B, rigid and affine, N = 5, nonzero heads, eval mode): the rotation block within 1e-5 absolute. Translations (~10 mm) within 1e-5 × the largest translation in the group, about 1e-4 mm; a plain 1e-5 absolute failed at 1.7e-5 mm from float32 rounding. The centered group's barycenter is I.
- **Group information flow** (eval mode): in config A, changing image 2 changes the other images' bottleneck and skip features. In config B the other images' encoder features are bit-identical. In both configs the other images' A_raw changes, through the decoder group blocks.
- **Gradients** (the regression test for the reference `fast_group_conv_3d` bug), with a point loss to barycentric targets of random rigid T, training mode, nonzero heads. Configs: A, B, A residual (narrow widths, see below), and B with `dense_group_concat_first`; tasks affine and rigid. Every trainable variable has a non-None, finite gradient, and every conv/dense kernel's gradient is nonzero. The one exception is the scale/shear head in rigid steps, which correctly gets `None`.
- **Dynamic N**: a `tf.function` with input signature [None, 40, 56, 48, 1] called with N = 2, 3, 5 traces once.
- **mixed_float16**: bottleneck and skips are float16; A_raw, A and params are float32; heads and centering compute in float32; all variables are float32; gradients are non-None. The policy is restored afterwards.
- **BN freeze**: with statistics frozen and `training=True`, every moving mean/variance is unchanged, while every gamma/beta gets a gradient (nonzero max). Unfreezing changes all of them.
- **Dropout**: identity with `training=False`. With `training=True` about 15% of the nonzero features are dropped (accepted range 8-22%) and the kept ones are scaled by 1/0.85.
- **Parameter counts**: the table above; asserts on the Dense sizes, the 11.5 M + 1.15 M split and A − B = encoder overhead.
- **Smoke training** (level 4, N = 4, rigid, phantom images warped by random T at ±20° / ±8 mm, Adam 1e-4, 20 steps, training mode): RMS error went from 7.85 mm to about 5.0-5.3 mm. The mean of the last 5 losses must be < 0.7 × the first loss. It is noisy because of training-mode BN on N = 4 plus dropout.

## Issues found locally (not code bugs)

- **cuDNN autotuning on the GTX 1080**: the full-width residual L5 conv (3×3, 512 → 256 channels at 10×14×12) took 107 s to autotune forward and 31 s backward on first use. With `TF_CUDNN_USE_AUTOTUNE=0` the backward pass stalled for more than 3 minutes. Single-conv configs aren't affected. The residual gradient test therefore uses narrow filters (8, 16, 16, 24, 32, 48); the structure is identical. Check the full-width residual step time on the A100/RTX 8000 before using `block='residual'`.
- **GPU memory on 8 GB**: tests free Keras reference cycles between tests (`clear_session` plus `gc.collect`). Level 1 is run forward only, with N = 1: one float32 stem output is 1.1 GB per image. A level-4 config A training step at N = 3 peaks at about 3.4 GB (float32).
- **Head init in tests**: random head weights scaled from eval-mode features gave very large rotations in training mode, because BN on 2-3 samples rescales features. Relative angles near 180° make the barycenter's matrix log ill-defined and produced non-finite gradients. The tests now scale the heads with features from the mode they run in. This also matters in practice: if the heads ever produce relative rotations near 180°, centering breaks down. The ±45° training range keeps relative angles below 90°.

## Not verified

- Not run in the container `tf10_nv.sif` (Python 3.8.10, numpy 1.21.1) or on an A100/RTX 8000. Locally only a Python 3.8 syntax check ran, plus tests under Python 3.9 / numpy 1.24. The orchestrator should run `python -m unittest discover -s tests -t .` in the container on glacier.
- Mixed precision ran on a GPU without float16 tensor cores (compute 6.1), so correctness only, no speed. XLA isn't tested.
- No gradient or training step at levels 2 and 1, and no memory figures for real group sizes (N = 8-12 at level 2, 2-4 at level 1). That belongs to milestones 8-9.
- In config B, encoder independence holds in inference mode (or with frozen BN statistics). Training-mode BN statistics still couple the group through the batch.
