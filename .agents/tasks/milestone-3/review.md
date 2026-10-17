# Group blocks, level-invariant encoder, linear decoder and in-graph centering (milestone 3)

This change adds the single-level groupwise network for the rigid and affine tasks. `models/layers.py` gains the group block family (`group_mean_concat`, `GroupConvBlock`, `GroupResBlock`, `GroupDenseBlock`, `NormAct`, plus `BatchNormWithSeparateAffine` and `InstanceNormWithAffine`). `models/group_net.py` assembles `GroupEncoder`, `LinearDecoder` and `GroupRegLevelModel`, and `models/config.py` holds a TF-free `ModelConfig` dataclass. Group interaction is a concat of the axis-0 mean after norm and activation. Weights stay level-invariant because parameter-free average pools are inserted before L2/L3 at finer levels. The heads are zero-kernel, identity-bias float32 Dense layers that feed the existing `GroupCentering`. The test log (17:05–17:08) postdates every source edit (latest: `tests/test_group_net.py` at 17:04). It shows 94 tests OK with 1 DALI skip, and the 30 new tests cover every behavior the milestone requires.

Watch for: `copy_weights_from` only checks variable shapes and names, so shape-neutral config differences copy silently (confirmed). There is a pre-head BN → act layer that the spec doesn't list (confirmed, documented by the coder). Runtime has not yet been verified in the `tf10_nv.sif` container (confirmed gap).

**Verdict**: APPROVED

## High-level view

The group block follows design 4.1 exactly: norm → act → concat(x, mean over axis 0 broadcast with `tf.shape`) → conv/Dense. That is the only place the code takes a group mean, so the only coupling between members is the intended mean, plus training-mode BatchNorm statistics, which design 4.1 asks for. Variables come from Keras sublayers and are used whole in `call()`. The gradient test asserts a non-None, finite gradient on every variable and a nonzero gradient on every conv/dense kernel, for configs A, B, residual-A and B+`dense_group_concat_first`. That closes the reference `fast_group_conv_3d` bug.

Config placement matches 4.2–4.4 and v3.2. In A, the stem and L2–L6 are group blocks; in B, none of them are. The 1×1 reductions (192, 96, 48, 24) are group blocks unconditionally. The 5040 → 2048 Dense gets the concat only when `dense_group_concat_first` is set, and 2048 → 512 is always a group block. The parameter table confirms this: dense_0 has 10.32 M parameters, or 20.65 M with the flag, and the decoder overhead is 1.15 M.

The pooling trick puts `pool2` only at level 1 and `pool3` at levels 2 and 1. The test asserts identical scope-local names, shapes and counts at levels 4, 2 and 1, plus a bit-exact output after `copy_weights_from` with moved BN statistics. Identity at init (A_raw = A = I) is tested at all three levels. Under `mixed_float16`, encoder outputs are float16 while heads, params, A_raw, A and every variable stay float32. `GroupCentering` was already float32 from milestone 1.

The remaining items are non-blocking hand-offs to milestones 4 and 8, plus one small hardening gap in weight copying.

<details>
<summary>Issues (6)</summary>

1. **Weight copy ignores config identity**: `copy_weights_from` copies between models whose configs differ only in shape-neutral fields (`translation_scale`, `negative_slope`, `dropout`, `centering_iters`). Compare `self.config.to_dict()` with `other.config.to_dict()` and raise on a mismatch.
2. **Pre-head norm-act not in spec**: `head_norm_act` (BN → act before dropout and heads) is a design addition. Have Jay confirm it and add it to design 4.4 so the doc stays the single source of truth.
3. **Container runtime unverified**: the TF tests ran under Python 3.9 / numpy 1.24; only a syntax check ran under 3.8. Run `python -m unittest discover -s tests -t .` inside `tf10_nv.sif` on glacier, with one GPU pinned, before milestone 4 relies on it.
4. **`stats_frozen` read at trace time**: toggling it doesn't affect an already-traced `tf.function`. Milestone 8's recalibrate-then-freeze code must build new train steps after freezing, and a test should cover that.
5. **None gradients for unused head**: rigid steps produce `None` for `scale_shear_head`. Milestone 4's custom loop must filter `None` gradients, as the smoke test does, or `apply_gradients` will fail or warn.
6. **Full-width residual untested**: the residual gradient test uses narrow filters because of GTX 1080 cuDNN autotune times. Time one full-width `block='residual'` step on an RTX 8000/A100 before using that flag.

</details>

<details>
<summary>Details</summary>

### Group mean and where members couple

`group_mean_concat` is `reduce_mean(x, axis=0, keepdims=True)` broadcast to `tf.shape(x)`. The channel dimension stays static and N stays dynamic: `test_dynamic_n_keeps_static_channels` and the model-level `test_no_retrace` (N = 2, 3, 5, one trace) cover this. `_PreActivation._pre` applies it after norm and act, as design 4.1 requires. `GroupDenseBlock` reuses the same front end, which gives W_a hᵢ + W_b m + b; the numpy test splits the kernel into those two halves. The block helper is the only call site of the mean.

The group-information test is the meaningful one here. In config B the other members' bottleneck and every skip are bit-identical when one image changes (eval mode), while A_raw still changes through the decoder group blocks. In config A both change. In config B, training-mode BN still couples members through the batch statistics (confirmed, and documented in the encoder docstring and the report). Design 4.1 calls for this behavior. It only matters for B's "run the encoder in chunks" claim if chunking ever happens in training mode.

### Config placement and dense-concat rule

`GroupEncoder` computes `g = group_config == "A"` and passes it to the stem and to L2–L6. `LinearDecoder` hard-codes `group=True` for the reductions and `group=(i > 0 or dense_group_concat_first)` for the dense blocks. If someone later adds a third dense width, it becomes a group block too, which matches the intent of v3.2.

### Level invariance and weight copying

At level 2 the input is 80×112×96: stem and L2 run at that size, then `pool3` halves it to 40×56×48. At level 1 `pool2` follows the stem and `pool3` follows L2, so the shapes match the 4.3 table, which `SKIP_SHAPES` asserts. `copy_weights_from` matches variables by order and checks shape and name after stripping the uniquified top-level scope. That rejects A ↔ B copies (tested). But two models with equal shapes and different shape-neutral hyperparameters copy without complaint (confirmed from the code). Example: a level-2 model built with a different `translation_scale` or `negative_slope` than the level-4 model it inherits from would produce a silently different function. Comparing `config.to_dict()` would close that.

### Heads, identity and precision

Both heads are built in `LinearDecoder.build`, so the variable set doesn't depend on which task ran first; that is necessary for the order-based copy. As a consequence, a rigid step leaves `scale_shear_head` with a `None` gradient (asserted by the gradient test), which milestone 4's loop must filter.

The added `head_norm_act` is defensible: with pre-activation blocks, the output of dense_1 is linear, so without it the head would be a linear map of a linear map. It is still an architectural addition that isn't in design 4.4. In training mode it adds one more BN over only N vectors, which is noisy at small N until the statistics are frozen.

### Test quality

The tests are independent: each class frees the Keras session, and `TestShapesAndLevels` builds once in `setUpClass` but only reads from it. Permutation equivariance runs for A and B, rigid and affine, with N = 5 and non-trivial heads. The translation tolerance is relative to the group's largest translation, which the report justifies with a measured 1.7e-5 mm float32 residual. The BN-freeze test checks unchanged moving statistics, nonzero γ/β gradients and that unfreezing changes every statistic. The dropout test checks training-only behavior, a drop fraction around 15% and the 1/0.85 rescale. The smoke run's loss fell to under half its starting value in 20 steps (RMS 7.85 → ~5.3 mm). Not tested: gradients or a training step at levels 2 and 1, any run inside the Python 3.8.10 / numpy 1.21.1 container, and a full-width residual step.

</details>

<details>
<summary>File map</summary>

- `models/layers.py`: adds the group block family, norms, `KaimingNormal`, `set_bn_stats_frozen` (`transform_image` unchanged).
- `models/group_net.py` (new): `GroupEncoder`, `LinearDecoder`, `GroupRegLevelModel` with `build_model`, `copy_weights_from`, `count_params_by_part`.
- `models/config.py` (new): `ModelConfig` dataclass with validation and dict round trip.
- `tests/test_group_layers.py` (new): block-level numpy references, norms, BN freeze, residual skip.
- `tests/test_group_net.py` (new): shapes/levels, copy, identity, equivariance, group flow, gradients, dynamic N, mixed precision, BN freeze, dropout, counts, config, smoke training.
- `.agents/tasks/milestone-3/`: report, test log, `record_tests.sh`.

Full diff: `git diff` plus the untracked files listed by `git status` in the repo root.

</details>
