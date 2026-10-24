# Milestone 9, Phase 1: technical design (level-1 memory root cause, knobs, calibrated model, tooling)

Status: v5 (2026-10-22), fourth revision. v5 answers the review of v4 (`design-review.md`: 0 HIGH, 1 MEDIUM, 7 NIT): the lean sampler's backward accumulates its 8 corners sequentially through a control-dependency chain on the accumulators (3.3); the lean sampler's forward and backward transients enter the inventory at their call sites as the new sub-phases Fw / Fb (2.1, 4.2), the level-1 tables are re-checked against them (2.5, 4.4), and the graph-mode memory test checks the modelled transients and the saving on a level-2-size field (3.3). The responses are in "v4 → v5 responses" at the end. v2 answered the review of v1 (0 HIGH, 10 MEDIUM, 9 NIT). v3 answered the review of v2 (0 HIGH, 2 MEDIUM, 8 NIT). v4 answers the review of v3 (`design-review.md`: 1 HIGH, 1 MEDIUM, 7 NIT): the lean sampler's backward is made data-dependent on `dy` so Grappler cannot merge it with the forward (3.3), with a graph-mode memory test; the step-time data sources are corrected to the window-mean keys that `metrics.jsonl` actually holds (4.5). Responses to all three reviews are at the end. Inputs: `requirements.md` (this folder), `network-design/design.md` (v3.13, revised with this design), the code at commit `8f850cd`, `milestone-6/memory.json`, `lncc-loss/memory.json`, `milestone-8/report.md`, the Keras 2.10 BatchNormalization source in the local `tf_gpu` env, and the level-2 cluster numbers transcribed in requirements section 2.

All hand numbers in sections 2 and 4 come from one inventory calculation (the same formulas `models/memory_model.py` implements, section 4.2), evaluated with the pre-fit coefficient values `c_net = c_warp = c_rec = c_enc32 = c_enc16 = c_loss = 1`, `c_bwd = 2`, `c_0 = 0.3 GiB`. Where a table also shows "expected" values, they use the expected coefficients of 4.3.

## 0. Overview and decisions

At level 1 a deformable step at today's widths does not fit for these reasons, largest first:

1. The trilinear sampler keeps 8 corner weights and 8 gathered value tensors per call under the tape. The 14 scaling-and-squaring compositions alone hold 2.5 KB per voxel per image, 16 GiB per image at level 1.
2. The one-hot warps and Dice keep the same 8-corner copies of a 19-channel tensor plus sliced and broadcast copies: 1.2 KB per voxel per image, plus 1.4 KB per sampled pair.
3. D1 runs at 160×224×192. At default widths its group-concat input has 128 channels. Under recompute, D1 and its backward transient hold about 1.3 KB per voxel per image in fp16.
4. The encoder is never recomputed. This matters at level 2 (stem and L2 at full resolution) but little at level 1, where only the stem runs at full resolution.

The analytic tensor inventory in section 2 reproduces all 21 measured level-2 deform points within −5 % … +1 % before fitting (−5 % … +3 % with the v4 skip-gradient term, 2.1). It uses one hand-set transient coefficient: the backward transient of the largest block, `c_bwd = 2` × its input channels × 4 B. With the same values it reproduces the six level-2 linear points within −8 % … +12 % and the level-4 points within −3 % … +13 % (the largest level-4 errors are at N = 2, where the constant `c_0` dominates). The linear residuals have opposite signs by precision (fp32 low, AMP high), so the fit carries one encoder coefficient per precision (`c_enc32`, `c_enc16`). With their expected values the linear points fall within −1 % … +6 %. The LNCC on/off pairs differ by at most 4 % in prediction and are within the measurement noise (section 2.3).

Phase 1 adds four memory knobs, all off by default, with levels 4 and 2 bit-identical when the defaults are used. It also adds a pure-Python calibrated memory model with a knob hold-out set, a forward and backward tensor-size probe, and level-1 measurement tooling. That tooling covers a frozen-chain stand-in, a `--cascade false` point and the frozen-stage peak. Phase 1 runs nothing on the cluster.

| Decision | Choice |
|---|---|
| Width configuration family | Named presets that expand to explicit `--filters` / `--decoder-widths` (no new `ModelConfig` field). Grid: `default`; uniform `u075` / `u050` / `u0375`; targeted full/half-resolution `f0d1` / `fr50` / `fr25`. |
| Selection rule (changed from R3.6, see 4.4) | Level 1 uses AMP and `full` (Jay, message 12). Maximise the model width (trainable parameters) subject to deform N ≥ `--n-target` (default **4**, Jay's minimum), predicted peak × (1 + margin) ≤ 44 GiB, margin = max(0.15, worst hold-out error), and the tensor-size bound. `recommend` always prints the per-N table "N → widest feasible configuration, peak, parameters" for N 2…8, so Jay can choose a larger N instead. |
| Checkpointing surface | New `--gradient-checkpointing full` (deform task) recomputes the encoder, the deform decoder and the integrator. The integrator runs as two blocks, exp(v) and exp(−v). BN moving statistics are updated exactly once per step: call 1 is the forward, and every later call snapshots and restores the statistics. `none / integrator / decoder / both` are unchanged, including today's documented double BN update. |
| Other knobs | `--lean-sampler`: custom-gradient trilinear sampler with int64 GPU index math; its backward recomputes the corner math from `coords + dep(dy)` (the `tf.recompute_grad` trick), so Grappler's dedup cannot merge it with the forward and keep the corner tensors alive, and it accumulates the 8 corners **sequentially** (each corner's ops carry a control dependency on the previous corner's accumulators), so at most one corner's temporaries are alive (3.3). Its forward is today's op graph (bit-identical); the forward's corner tensors are a transient while the call runs, modelled as sub-phase Fw (2.1). `--lean-losses`: Dice without materialised slices or broadcast, LNCC under `tf.recompute_grad`. `--split-group-conv`: conv_a(x) + conv_b(mean), so the doubled-channel tensor is never built. All default false. |
| Integration steps | Option B of R2.3: T stays one `ModelConfig` value, identical at every cascade level. `check_cascade_config` is unchanged. With the lean sampler, T does not change the peak phase (D), so the expected choice is **T = 7**. `recommend` selects T = 6 only if `integration_steps.json` says `t6_accuracy_ok` and the rule in 4.4 step 4 applies. |
| Frozen stages in the measurement | Both. (a) `--cascade false` level-1 points (R4.2, literal). (b) The **stand-in** `FrozenChainStandIn`: the existing level-4 `FrozenStage` plus a float32 level-2 model with the real m8 composition rules, inside the trainable step. JSON label `cascade_kind: "standin_chain_4_2"`. (c) `frozen_peak_mb`, the frozen-stage inference peak measured with the trainer resident. The 3-level cascade itself is implemented after Phase 1 (section 6.1). |
| Per-tensor limit | The bound is the largest size verified for forward **and** backward kernels (`tests/gpu/probe_tensor_limit.py --mode run`, conv / BN / LeakyReLU under `tf.GradientTape`). Until the probe JSON exists, tensors that enter a backward kernel are bounded at 2³¹ − 1 elements in both precisions. Inference-only tensors are bounded at 2³¹ − 1 in fp32 and 3.52·10⁹ in fp16 (observed OK). The build forward is kept small: the tool builds every configuration at `min(N, --build-n 2)`, and `main.py` builds at `min(n_build, 2)` when `--split-group-conv` is true. `limit_flags` also checks the build forward at the record's `build_n`. |
| Expected outcome (pre-fit, provisional until `m9_memory.sh`) | **`f0d1`**: filters (16, 64, 128, 192, 256, 384), decoder (128, 96, 64, 32, 8). AMP, `full`, lean sampler + split, T = 7. Level-1 deform N = 4 at 33.5 GiB predicted (38.5 GiB with the 15 % margin; 34.9 / 40.1 under the conservative skip-gradient bound of 2.5). On the A100, N = 4 is 33.5 ≤ 36 without the margin; N = 3 is 29.4 with it. Linear N = 16 at 12.7 GiB. v5: the lean sampler's forward transient (Fw, 2.1) does not move the N = 4 peak (Fw 32.0 < D 33.5 GiB), but it caps every lean configuration at N = 5 without the margin and N = 4 with it, so the v4 alternatives `fr50` N 5–6 / `fr25` N 7 are gone. Per-N table: N 2 `default`, N 3 `default` LS, N 4 `f0d1` LS, N 5 none within the margin (`f0d1` LS 41.4 GiB without it), N ≥ 6 none. Under the order-independent forward bound N = 4 stays `f0d1` but needs lean losses as well (35.0 / 40.3 GiB). The code-generated per-N table is authoritative. All of this assumes the lean sampler frees its corner tensors in graph mode and sequences its backward corners, which the graph-mode memory test of 3.3 verifies before any number is used. |

Technology stack (locked): Python 3.8 syntax, TF 2.10 / Keras 2.10, numpy 1.21, scipy 1.4 (`scipy.optimize.least_squares`), unittest, bash + SLURM + `tf10_nv.sif`. No new dependencies.

## 1. Assumptions and unverified items

Carried from the requirements: A1, A2, A3, A4, A5 and A6.
- A1: 44 GiB budget on the 48 GB cards, 36 GiB on the A100. "GB" means 1024 MB = GiB, as `peak_mb / 1024` in the existing tools.
- A2: `full` recomputes the encoder, the decoder and the integrator.
- A3: the reduced widths apply at every level. Reductions, dense layers and the 5×7×6×24 bottleneck are unchanged.
- One correction to the requirements (review F16): requirements section 2 says all deform numbers include LNCC. The level-4 A6000 and GTX 1080 points come from milestone 6, which predates LNCC, so they are Dice only. `memory_calibration.json` records `lncc: false` and the source for them, and the report states the correction.

| Id | Assumption | Status / verification |
|---|---|---|
| U1 | fp16 forward kernels (Conv3D, concat, BN, LeakyReLU) accept tensors up to 3.52·10⁹ elements. Evidence: the eager inference forward in `build_model(16)` ran at linear AMP N = 16 at level 2 ([16, 80, 112, 96, 256] group concat). | Forward and inference only, one size. Backward kernels (Conv3DBackpropInputV2 / FilterV2, FusedBatchNormGradV3, LeakyReluGrad) are unverified; the probe's run mode tests them (2.4). |
| U2 | The fp32 N = 12 linear failure in `d2` is a per-tensor limit (2.64·10⁹ elements), not an OOM. | Unverified. `m9_memory.sh` step 0b prints the error text. The probe tests fp32 at 2³¹ ± 2²⁴. |
| U3 | Under fp32, Grappler's layout optimizer runs Conv3D in NCDHW (the OOM shapes [N, 256, 80, 112, 96] and [N, 3, 80, 112, 96] are NCDHW); under fp16 the graph stays NDHWC. | Inferred from the OOM messages. Absorbed by `c_bwd` and `c_net`. |
| U4 | The default sampler's int32 index tensors live in host memory (TF registers int32 elementwise GPU kernels with HostMemory) and do not count in the GPU peak. | Unverified. It moves the `none` / `integrator` points by +3 … +5 % (still within ±15 %) and does not affect the lean sampler, which keeps no index tensors. The fit absorbs it in `c_warp`. |
| U5 | Graph-mode liveness: a forward tensor stays allocated only while a reachable gradient op references it. Per group block that is the BN input x, the BN output y (for gamma), the affine output z (LeakyReLU input) and the concat (conv input): 5·C_in. The block output is consumed by linear ops (upsample, pool, concat), which keep nothing. | Standard TF behaviour. Supported by the section 2.3 cross-check. |
| U6 | Keras 2.10 `BatchNormalization(axis=-1)` on rank-5 inputs uses the fused kernel (NDHWC), so under AMP it keeps the fp16 input (no fp32 copy). | Verified in the Keras 2.10 source (`build`: `self.fused = rank in (4, 5)` under V2 behaviour; axis [4] → `NDHWC`), local `tf_gpu` env. Container Keras is assumed to be the same 2.10 release that TF 2.10 bundles (not checked on the container). |
| U7 | DALI's GPU buffers are outside the TF allocator (≤ ~2 GiB at level 1, N ≤ 16), so the 44 GiB TF budget leaves room for them on a 48 GB card. | Unverified at level 1. `verify_dali_pipeline.py --level 1` check e records it. |
| U8 | Stateful `assign` ops created inside a `tf.recompute_grad` gradient function run in creation order under tf.function automatic control dependencies. | Verified by the phase-1 test AC3(b). |
| U9 | Step-time predictions scale from a few measured points and carry ±50 % uncertainty. | Stated with every printed estimate (4.5). |
| U10 | RTX A6000 and RTX 8000 have the same TF allocator behaviour (the level-4 A6000 points). | Unverified. A single resident constant is used. |
| U11 | The transcribed level-2 numbers (requirements section 2) are `peak_mb`, the steady state after the warm-up reset, not `warmup_peak_mb`. | Unverified. The m8 report table was built from the tool's printed `peak` column. `m9_memory.sh` step 0b prints both fields of `scratch/m8/memory_level2.json`. If they are `warmup_peak_mb`, the calibration file is corrected before `fit`. |
| U12 | A cuDNN autotune workspace allocation that fails during the warm-up is not fatal: TF's BFC allocator reports "not a failure" and the autotuner tries the next algorithm. | Standard TF 2.10 behaviour (`TF_CUDNN_WORKSPACE_LIMIT_IN_MB` default 4096). The warm-up excess is reported, not budgeted (4.1). |
| U14 | The level-4 linear fp32 N = 32 "26–34 GB" figure (project notes) is a TF allocator peak. | Unverified: the source run and the quantity (allocator peak or `nvidia-smi` use) are not recorded. It is stored as `status: "range"` and checked for upper containment only (predicted ≤ 34 GiB). Job step 6b measures level-4 linear N 16 / 32 with the tool, and those records replace it. |
| U15 | The loss terms' local tensors (warped one-hot, Dice slices, LNCC filters and moments) need not all be alive at once. TF's graph executor starts a term's backward as soon as its inputs exist, so the end of the forward pass is not one simultaneous live set. | Hypothesis. It is consistent with the level-4 LNCC on/off pairs (no measurable LNCC cost). It is modelled by the fitted `c_loss` (4.3), with the conservative sum rule (`c_loss = 1`) before the fit. |
| U16 | Keras 2.10 `OptimizerV2.apply_gradients` creates Adam slots only for the variables it receives. The Trainer passes only variables with a gradient (the static `pairs` filter in `run.py`). A one-task measurement therefore holds slots and gradients only for the parts that task uses. A linear-only record has none for the deform decoder. | The Trainer filter is verified in `run.py`. Lazy slot creation is standard Keras 2.10 behaviour; the source was not re-read in this revision. It is used for the resident term of single-task records (4.2). Training with all three tasks holds slots for every part. |
| U17 | The A100 / RTX 8000 step-time ratio per task measured at level 2 under AMP (deform 4.9 / 7.0 s ≈ 0.70, linear 0.73 / 1.16 s ≈ 0.63) also applies at level 4 (fp32) and level 1 (AMP). | Unverified for fp32: without TF32 the A100 / RTX 8000 fp32 throughput ratio differs from the fp16 one. Guarded by the m7 consistency check of 4.5 (±20 %, else a single fitted scalar replaces the level-4 ratios). The RTX 8000 level-2 times come from 5-step memory-tool runs (noisy). |
| U18 | Grappler's arithmetic optimizer (DedupComputations) merges identical stateless nodes of the forward and the custom-gradient backward inside one `tf.function` (review v3 F1). Adding `dep(dy)` to the coordinates makes every recomputed node's input chain differ, so nothing is merged, as in `tf.recompute_grad`. | The dedup behaviour is taken from the review and the TF 2.10 `recompute_grad` source comment; not reproduced locally. The graph-mode memory test of 3.3 verifies the outcome (v5: case B, the lean integrator block at least 0.85 GiB below the default one on a level-2-size field, Grappler on) and prints the case-B numbers without the dependency for information. |
| U19 | (v5) Inside a `tf.function`, the TF 2.10 executor honours control edges, and Grappler's dependency optimizer removes only control edges that are implied by other paths. A chain "corner k's ops depend on corner k−1's accumulators" therefore runs the corners one after another, and the previous corner's temporaries are freed before the next corner allocates. Independent branches without such an edge (the 8 forward corners, the independent loss terms) may run in any order and overlap. | Standard TF behaviour, not reproduced locally. Verified by case C of the graph-mode memory test (3.3): the backward alone, lean peak ≤ 1.5 × the sequenced model, which an unsequenced backward exceeds by about 4–6×. If case C fails, the backward switches to the `tf.while_loop` form specified in 3.3. |
| U13 | The backward transient in the peak phase is about `c_bwd` × (input channels of the largest block alive in that phase) × 4 B, at that block's resolution, in both precisions. That covers the gradient of the concat or conv input, dx, and the conv backward workspace. | Fitted (`c_bwd`, expected about 2). Pre-fit `c_bwd = 2` reproduces the level-2 `both` points within 1 % (2.3). At level 1 the block is D1 at full resolution, so the term scales 8×. |

## 2. Memory root cause (R1)

### 2.1 Method

For one training step the inventory lists every tensor that a reachable gradient op keeps (U5), per image and per voxel of the resolution it lives at, in bytes (fp32 4, fp16 2, int32 0 on GPU per U4, int64 8). Each tensor is assigned to the phases in which it is alive:

- **F**: end of the forward pass, start of the loss backward (everything taped and not recomputed). Loss-local tensors are grouped into four terms: template Dice (warped one-hot, Dice slices, smoothness), template LNCC, pair Dice and pair LNCC. The pair displacement is shared by both pair terms and counts as shared. F holds the shared tensors plus `max_term + c_loss · (sum_terms − max_term)` (U15). `c_loss = 1` is the sum rule, used before the fit and in every hand table. `c_loss = 0` means only the largest term is alive.
- **Fw** (v5, lean sampler only): the forward moment of each lean sampler call c. It is the tensors alive at c's call site plus c's forward moment (below). Calls are taken in the deform step's program order: cascade compositions, integrator compositions, template one-hot warp, pair displacement, pair one-hot warp, template image warp, pair image warp. The **timed rule** (default) counts as alive at c the shared tensors (batch and fields, kept skips, cascade outputs) plus the loss-term tensors of the calls before c, with the `c_loss` weighting of F. The **order-independent bound** (`fwd_transient_bound=True`) instead counts every other loss term as alive: F − 150 − (the tensors of c's own term created after c) − 12 + c's forward moment (the 12 B are c's coordinates, which the moment already contains). It covers TF's executor running c last, because neither the forward corners nor the independent loss branches are sequenced (U19). Both are printed. Selection uses the timed rule, as for the skip gradients. Fw is the maximum over the calls.
- **Fb** (v5, lean sampler only): the loss backward at each lean sampler call c, F − 150 − (c's own term tensors created after c, already consumed by then) + share_c · (4·C_c + T_bwd(c)), with share = 1 for template calls and P/N for pair calls. With the default sampler Fb is part of F: its backward transient is within the calibrated 150 B F transient.
- **I**: integrator backward (the recomputed integrator tape plus what is still alive).
- **D**: deform-decoder backward (the recomputed decoder tape).
- **E**: encoder backward (`full` only).
- **Z**: frozen-stage inference (before the tape opens).
- **V**: validation (no tape).

Peak = resident + max over phases of (live bytes + the phase's backward transient). For linear tasks the peak phase is the encoder backward with the whole encoder tape alive: the encoder bytes plus the transient of the largest encoder block (`c_bwd` × its C_in × 4 B; at level 2 that is L2 at V2 with C_in 32, so 256 B), plus the batch.

The build forward (`build_model(n)`) never enters the predicted `peak_mb`. The tool measures `warmup_peak_mb` and `peak_mb` only after `reset_memory_stats`, which runs after `build_model` (verified in `measure_level_memory.measure()`). The build forward is evaluated only by `limit_flags`, at the record's `build_n`, as an inference forward with the largest tensor being the unsplit D1 (level 1) or D2 (level 2) group concat (2.4). The modes move tapes between phases:
- without recompute, every taped tensor is alive in F;
- `integrator` moves the integrator tape to I, and the loss tape is freed before I;
- `both` also moves the decoder tape to D;
- `full` moves the encoder tape to E. Only the encoder outputs (the skips) stay alive from E's forward to D's backward.

Block inventories:
- Group block (pre-activation BN → affine → LeakyReLU → concat(group mean) → conv): x, y, z and the concat = **5·C_in**.
- With `--split-group-conv`: x, y, z, a = leaky(z) (conv_a input) and the group mean [1, V, C_in] (conv_b input) = **4·C_in + C_in/N**.
- Stem (no pre-activation): its 2-channel concat.
- SVF head (NormAct → conv): x, y, z and the conv input a = **4·C**, with a kept in fp32 under AMP because the head conv is float32 (review F13).

Upsamples, pools, centering and adds keep nothing.

Skip gradients (review v3 F9). The decoder backward produces d(skip_k) when the decoder block that consumes skip_k (D1 ← stem, D2 ← L2, …) runs its backward, and d(skip_k) stays alive until the encoder backward of the stage that produced skip_k (it also waits for the gradient from the coarser encoder path). The inventory therefore carries the term `B_sg` = d(skips) in the compute dtype of the encoder, per voxel of each skip's resolution, timed per sub-phase:
- in the D phase evaluated at the backward of block b (the block that sets the transient), `B_sg` counts the skips consumed by the decoder blocks finer than b, whose backward has completed. The skip slice of b itself is part of b's input gradient dx, which the transient `c_bwd · C_in(b) · 4 B` already covers (C_in(b) includes the skip channels), and the separate slice copy is created only when the conv-input gradient of b has been freed;
- in E (and in the taped encoder backward of the other modes) it counts every skip gradient.

Effect: at level 2 the peak block is D2, so `B_sg` = d(stem skip), 32 ch at V2: 64 B (AMP) / 128 B (fp32) per V2 voxel, +1.5 % / +1.8 % on the `both` rows of 2.3. At level 1 the peak block is D1, the first decoder backward, so `B_sg` = 0 at the peak moment; the later D sub-phase (D2 backward) and E carry it, and both stay below the D1 moment (2.5). The conservative alternative, every skip gradient counted at the D1 moment, is shown in 2.5 and does not change the N = 4 recommendation. The two conservative bounds (skip gradients, forward transient) act on different phases (D, Fw), so they never add up.

Backward transient (U13): `c_bwd · C_in,max · 4 B` at the resolution of the largest block alive in the phase. For example, D2 at level 2 (C_in = 128) gives 1024 B per voxel; D1 at level 1 (C_in = 64 at default widths, 48 for `f0d1`) gives 512 / 384 B per V1 voxel. Other transients:
- F: 150 B (the loss gradients);
- I: one composition's backward, 250 B with the default sampler. With the lean sampler (v5) I is the larger of two moments. The backward moment is dy 12 B + T_bwd(3, with input gradient) 160 B = 172 B (v4: 48 B). The forward moment of the recomputed composition is 360 B (below);
- E: `c_bwd` × the largest encoder block input.

Trilinear sampler (`layers.sample_voxel_coords`, linear mode, C channels), per output voxel per image:
- coordinates 12 B (for the clip gradient);
- per-axis weights and their products, 16 + 24 B;
- the 8 corner weights, 32 B;
- the 8 gathered value tensors, 32·C B;
- the 8 int32 corner indices, 32 B (host, U4).

GPU total **84 + 32·C B**: C = 3 → 180 B, C = 19 → 692 B, C = 1 → 116 B. With the default sampler all of it is taped, so it is already in F (or I) and has no separate transient. Only the untaped products and the running sum (≤ 2·4C B) are not taped, and they fall below F's 150 B transient plus the loss terms not yet created.

Lean sampler (v5 inventory, review v4 F1), per call, per output voxel per image:
- **Retained** under the tape: the coordinates and the captured input, 12 + 12 B for a field composition. For a data volume (one-hot, image) the input is already alive as a batch tensor, so only the 12 B count. The output is counted where it is needed.
- **Forward transient** T_fwd(C) = **84 + 32·C** B of float tensors plus **168 B of int64 index tensors** on the GPU. The float tensors are the same ones as in the default inventory: the lean forward computes them but does not keep them. The int64 tensors are lo 24, the six per-axis clipped offsets 48, the four xy offsets 32 and the eight corner indices 64. The default path's int32 copies are host-resident (U4), which is why the default total has no index term. These tensors live only while the call runs, but because the forward corners are not sequenced (the op graph stays today's) all of them can be alive at once. The **forward moment** of a call is T_fwd(C) + 4·C (the output / running sum) = 252 + 36·C: C = 19 → **936 B**, C = 3 → 360 B, C = 1 → 288 B. It enters phase Fw (and phase I for the recomputed integrator compositions).
- **Backward transient** with the sequenced corners (3.3): T_bwd(C) = **88 + 8·C** B, plus **16·C** B when the input gradient is needed (`images_grad=True`, field compositions).
  - The fixed 88 B: `coords_b` 12, lo (int64) 24, frac 12, the d_coords accumulator 12, and one corner's index 8, weight 4, s_k 4 and d_coords partial 12.
  - The 8·C: one corner's g_k and dy · g_k.
  - The 16·C: the d_flat accumulator, dy · w_k, the segment sum and the new accumulator.
  - Values: C = 19 without the input gradient (one-hot) → **240 B**; C = 3 with it (fields) → 160 B; C = 1 without it (image) → 96 B. The per-corner index and weight temporaries below 40 B are inside the 1.5× tolerance of the memory test.
  - The **backward moment** of a call is the live set + dy (4·C) + T_bwd(C), and enters phases Fb and I.
- An unsequenced backward would instead hold up to 8 corners' g_k and products at once: about 1500 B (C = 19) and 870 B (C = 3). This is the v4 F1 case.

### 2.2 Level-2 per-image inventory (default widths, Dice + LNCC 1/1, cascade, T = 7)

Bytes per level-2 voxel per image. 1000 B per voxel = 0.80 GiB per image at level 2, 6.41 GiB at level 1 and 0.100 GiB at level 4. "Ch" = channel-equivalents at 80×112×96; coarser stages are divided by 8 per halving.

| Component | Ch-equivalent | fp32 B | AMP B | none | integrator | both | full (new) |
|---|---|---|---|---|---|---|---|
| Encoder full-res: stem 2, L2 5·32 | 162 | 648 | 324 | F | F | F, I, D (never recomputed) | E (skips kept from E to D: 96 ch) |
| Encoder coarse L3–L6 | 138 | 550 | 275 | F | F | F, I, D | E |
| Decoder coarse D5–D3 | 171 | 684 | 342 | F | F | D | D |
| Decoder full-res D2 5·128, D1 5·64, SVF head 4·16 | 1024 | 4096 | 2080 (head input fp32) | F | F | D | D |
| Backward transient, largest block (D2, C_in 128) | — | 1024 (c_bwd 2) | 1024 | (D-like sub-phase, below F) | — | D | D |
| Integration, 14 compositions × 180 B (+ 14 × 32 B host) | — | 2520 | same | F | I | I | I (two blocks, 7 at a time) |
| Cascade composition (2 compositions, in the tape) | — | 360 | same | F | F | F | F |
| Template: one-hot warp (C = 19) 692, Dice slices 144, smoothness 36 | — | 872 | same | F | F | F | F |
| LNCC template: image warp 116, box filters and moments 168 | — | 284 | same | F | F | F | F |
| Per sampled pair (× P/N): displacement 204, one-hot warp 692, Dice slices 144, LNCC 284, target slice 72 | — | 1396 per pair | same | F | F | F | F |
| Batch and fields: images ×2 8, labels int32 4, one-hot 76, U/U⁻ 24, v/u/u⁻ 36 | — | 148 | same | all | all | all | all |

P = max(1, round(0.1·N(N−1))): N 2/3/4/6/8 → P 1/1/1/3/6. Over N = 2…4 the pair count is constant, and the pairs live only in F, so they do not affect the `both` peak (phase D) at any N.

Largest single tensors at level 2: the D2 group concat [N, 80, 112, 96, 256] (2.20·10⁸ elements per image), the one-hot [N, 80, 112, 96, 19] and the fields [N, 80, 112, 96, 3]. These are exactly the shapes in the OOM messages.

### 2.3 Which phase sets the peak, cross-checked against every measured point

Pre-fit prediction (int32 index tensors excluded per U4; resident = trainable parameters × 16 B (+2 B with AMP) + frozen parameters × 4 B + `c_0` 0.3 GiB) vs measured, level 2, RTX 8000, GiB:

| Mode | Peak phase (per image) | N2 pred / meas | N3 | N4 | N6 | N8 | Error range |
|---|---|---|---|---|---|---|---|
| fp32 none | F: network 5978 + integration 2520 + cascade 360 + template 1156 + pairs 1396·P/N + batch 148 + 150 | 18.5 / 19.0 | 26.7 / 27.5 | 35.0 / 36.4 | 53.8 (> 48: OOM ✓) | — | −4 … −3 % |
| fp32 integrator | I: 5978 + 2520 + 148 + 250 | 15.1 / 15.6 | — | 29.3 / 30.7 | 43.6 / 45.3 | — | −5 … −3 % |
| fp32 both | D: encoder 1198 + decoder 4780 + 148 + 1024 | 12.3 / 12.2 | 18.0 / 17.9 | 23.7 / 23.9 | 35.2 / 35.2 | — | −1 … +1 % |
| AMP none | F: network 3021 + 2520 + 360 + 1156 + pairs + 148 + 150 | 13.8 / 13.9 | — | 25.6 / 27.0 | 39.6 / 39.9 | — | −5 … −1 % |
| AMP integrator | I: 3021 + 2520 + 148 + 250 | 10.4 / 10.9 | — | 19.9 / 20.8 | 29.4 / 30.8 | 38.9 / 40.9 | −5 … −4 % |
| AMP both | D: 599 + 2422 + 148 + 1024 (I = 3517 is next) | 7.6 / 7.6 | — | 14.3 / 14.4 | 21.0 / 21.2 | 27.8 / 27.8 | −1 … 0 % |

The `both` rows above do not yet include the skip-gradient term `B_sg` of 2.1 (d(stem skip) alive at the D2 backward). With it they move by +64 B (AMP) / +128 B (fp32) per V2 voxel: AMP N8 27.8 → 28.2 (+1.5 % vs measured 27.8), fp32 N2 12.3 → 12.5 (+2.5 %), so the `both` range becomes 0 … +3 %. The pre-fit statements below that quote −1 … +1 % refer to the table without it; `tests/test_memory_model.py` checks the version with it.

The peak phases:
- `none`: the end of the forward pass.
- `integrator`: the integrator backward (the integrator tape and the loss tape never coexist).
- `both`: the decoder backward in both precisions.

v1 of this design compared `both` against the wrong phase and omitted the backward transient, which produced the 11 % / 25 % gaps the review flagged (F1). The AMP-both residual of v1 (0.68 GiB per image) is the D-phase transient: the gradient of the D2 input and concat plus the conv-backward workspace, about 1024 B per level-2 voxel. It is precision-independent in bytes, which is consistent with an fp32 workspace or accumulation. At level 1 the same term sits on D1 at full resolution, and the level-1 tables in 2.5 include it.

With the int32 indices on the GPU instead (U4 false), `none` and `integrator` move by +3 … +5 % (fp32 none N4 35.0 → 36.4 predicted vs 36.4 measured). The `both` and lean-sampler predictions do not change. The fit absorbs this in `c_warp`, and the level-1 recommendation is not sensitive to it.

Level-4 cross-check (Dice only, no LNCC, no cascade; nothing is pooled at level 4, so every stage down to L3 is at 40×56×48):

| Point | Predicted | Measured | Error |
|---|---|---|---|
| A6000 default fp32 none N8 / N12 / N16 (GB) | 16.1 / 24.2 / 32.7 | 16.6 / 24.8 / 33.8 | −3 % |
| GTX 1080 default none N2 / both N2 (MB) | 4608 / 4140 | 4437 / 3909 | +4 / +6 % |
| GTX 1080 narrow none N2 / N3 / N4 (MB) | 2364 / 3240 / 4115 | 2089 / 3008 / 4010 | +13 / +8 / +3 % |
| GTX 1080 narrow both N4 / N8 (MB) | 2732 / 4964 | 2510 / 4781 | +9 / +4 % |

The GTX intercepts are high because `c_0 = 0.3 GiB` is a cluster-sized constant (measured GTX `resident_mb` is 67–424 MB). The fit lowers `c_0`, so these points tighten.

The v1 "narrow both over-predicted by 19 %" (review F2) came from subtracting pair growth from the narrow-both slope. Under `both` the pairs live only in F, which is not the peak (phase D: 5330 B vs F 3752 B at N = 8). The raw slope (568 MB per image) matches the predicted 546 MB per image (−4 %). So the level-4 and level-2 `both` residuals do not have opposite signs.

Level-2 linear cross-check (review F1). Rigid and affine have the same inventory. The peak phase is the encoder backward. Per level-2 voxel and image:
- encoder 1198 B (fp32) / 599 B (AMP);
- transient 256 B (L2, C_in 32);
- batch about 10 B (image and transform only, no labels).

The deform build forward is excluded (2.1). The resident is task-touched (U16): encoder + linear decoder (22.49 M parameters) × 16 B (+2 B with AMP), the untouched deform decoder (7.14 M) × 4 B, the frozen level-4 stage (29.63 M) × 4 B, plus `c_0`. That gives 0.77 GiB (fp32) and 0.81 GiB (AMP). Measured intercepts from the two-point fits are 0.80 (fp32) and 0.63 (AMP).

| Point (GiB) | Measured | Pre-fit (c_enc = 1) | Error | Expected (c_enc32 1.10, c_enc16 0.89) | Error |
|---|---|---|---|---|---|
| fp32 N4 | 5.9 | 5.46 | −7 % | 5.87 | −1 % |
| fp32 N8 | 11.0 | 10.15 | −8 % | 10.97 | 0 % |
| fp32 N12 | `failed_d2` | build forward D2 concat 2.64·10⁹ fp32 > 2³¹ − 1 → limit flag | infeasible ✓ | same | ✓ |
| AMP N4 | 3.2 | 3.58 | +12 % | 3.38 | +6 % |
| AMP N8 | 5.8 | 6.35 | +9 % | 5.95 | +3 % |
| AMP N12 | 8.5 | 9.13 | +7 % | 8.51 | 0 % |
| AMP N16 | 10.9 | 11.90 | +9 % | 11.08 | +2 % |

Per-image slopes: measured 1.275 (fp32) and 0.642 GiB (AMP), pre-fit 1.173 and 0.693. The opposite signs are a precision effect on the encoder tape, not a task effect. The expected values come from those slopes: (1592 − 266) / 1198 = 1.107 and (801 − 266) / 599 = 0.893 B per B. The mechanism is not identified. A plausible fp32 component is the NCDHW layout copies of the full-resolution stem and L2 conv inputs (U3). The fp16 deficit is unexplained.

The deform points barely move with these values. The encoder is 10–17 % of the deform bytes in the peak phase, so `c_enc16 = 0.89` moves AMP `both` by −1.4 % (N8 27.8 → 27.4) and `c_enc32 = 1.10` moves fp32 `both` by +1.8 % (still within −1 … +3 %). The c_net16 fallback of v2 is dropped. It would have scaled the decoder bytes as well, which the deform AMP points do not support.

LNCC on/off pairs (`lncc-loss/memory.json`, GTX 1080, level 4, `none`, Dice + LNCC 1/1 vs Dice only, MB):

| Point | Off measured | On measured | Measured Δ | Predicted Δ, c_loss = 1 (sum) | Predicted Δ, c_loss = 0 |
|---|---|---|---|---|---|
| default N2 | 4668 | 4619 | −49 | +92 (+2.0 %) | 0 |
| narrow N2 | 2095 | 2160 | +65 | +92 (+4.4 %) | 0 |
| narrow N3 | 3188 | 3171 | −17 | +122 (+3.8 %) | 0 |
| narrow N4 | 4087 | 4047 | −40 | +153 (+3.7 %) | 0 |

Per image, LNCC costs 284 B × 107 520 V4 voxels = 30.5 MB per template image and the same per pair (P = 1 for N 2–4). Both rules put every LNCC point within ±15 %, because the difference is at most 4.4 % of the peak. The measured Δ has no sign and is within ±65 MB of run-to-run noise, which favours `c_loss` near 0 at level 4.

The `none` peak phase, reconciled with design.md section 9 (v3.9). That section says the level-4 `none` peak is the integration backward, with the LNCC tensors freed before it. In this inventory that sub-phase is I₀: the network and integration tapes, the batch and the composition transient (250 B), with every loss-local tensor freed. At level 2, I₀ per image is 5978 + 2520 + 148 + 250 = 8896 B. It predicts fp32 `none` N2 at 15.1 GiB against 19.0 measured (−20 %). F with the sum rule gives 18.5 (−3 %). F with only the largest term (`c_loss = 0`: shared 9006 + template Dice 872 + 150 B) gives 16.9 (−11 %). So the level-2 data reject I₀ as the `none` peak and put it in F. The level-4 LNCC data say that not every loss term is alive at F. The model therefore keeps F as the `none` peak and fits `c_loss` ∈ [0, 1] (4.3) to both data sets. The design.md section 9 sentence is corrected in v3.13: "the peak is in the forward-end phase, and the loss terms need not coexist (LNCC adds no measurable memory at level 4)". The level-1 consequences are in 2.5.

### 2.4 The linear fp32 N = 12 `d2` failure (R1.3)

Hypothesis: `measure_level_memory.measure()` calls `build_model(n)` with the deform decoder on, so a linear N = 12 configuration runs one deform inference forward at N = 12. Its D2 group concat [12, 80, 112, 96, 256] has 2.64·10⁹ elements, above 2³¹ − 1 = 2.147·10⁹. N = 8 (1.76·10⁹) worked. The same forward at AMP N = 16 (3.52·10⁹ fp16 elements) also worked. So if this is a size limit, it lives on the fp32 path only: the NDHWC→NCDHW layout transpose or the cuDNN NCDHW fp32 convolution (U1, U3), not concat or BN, which run in both precisions. The alternative is a plain OOM of the fp32 inference forward: concat 10.6 GB + its NCDHW copy 10.6 GB + skips + autotune workspace. The code cannot decide between them. Three pieces of evidence settle it:

1. `tests/gpu/probe_tensor_limit.py --mode graph` runs locally on the CPU with no execution. It builds the level-2 model at N = 12 in a TF1-style `tf.Graph` with a fixed-shape placeholder. If Keras refuses: eager `build_model(1)` at level 4, then `tf.function(...).get_concrete_function(TensorSpec)` on the level-2 model, which creates the variables during that single trace. It then lists, sorted, every op output with more than 2³¹ − 1 elements (op type, scope, shape, dtype, elements, bytes). Expected first entries: `deform_decoder/d2/...ConcatV2` [12, 80, 112, 96, 256] and the d2 `Conv3D` input. It prints `hypothesis limit: 2147483647 elements (int32 index); run mode decides`. This satisfies AC6 locally.
2. `--mode run` (GPU; `m9_memory.sh` step 0a) tests sizes E ∈ {2³¹ − 2²⁴, 2³¹ + 2²⁴} in fp32 and fp16, plus fp16 at 3.52·10⁹ and 4.4·10⁹. The ops:
   - forward: `tf.nn.conv3d` (NDHWC, [1, X, Y, Z, 8] → 8 channels), `tf.transpose` NDHWC→NCDHW, `group_mean_concat`, Keras BN training call, LeakyReLU, `reduce_mean(axis 0)`, `tf.gather` of rows of [E/8, 8], `unsorted_segment_sum`;
   - backward (review F4): each conv / BN / LeakyReLU probe also runs under a `tf.GradientTape`, with `tape.gradient` with respect to the input and (conv) the kernel. That exercises Conv3DBackpropInputV2, Conv3DBackpropFilterV2, FusedBatchNormGradV3 and LeakyReluGrad on the same E.
   
   Each op and direction is wrapped in try/except, with status `ok | oom | error` and the first error line. Before each op the script estimates the bytes (inputs + outputs × 1.5 for forward, × 3 for backward). If that is above the device limit it records `skipped_mem`; on the local GTX 1080 most entries are skipped. Output: `scratch/m9/tensor_limit.json` with per-op `{dtype, elements, direction, status, error}` and a summary `verified_max_elements = {"fp32": {"fwd": E, "bwd": E}, "fp16": {...}}`, the largest E with status `ok` for every op of that direction.
3. `--mode m8-errors --m8-json <file>` prints every non-`ok` record of the m8 memory JSON (config key and `error`), and the `peak_mb` and `warmup_peak_mb` columns of all records (U11). This is step 0b of the job.

Tool fix (either way; review F2 option a): **every** configuration, deform included, builds with `build_model(min(N, --build-n))`, default `--build-n 2`. This is safe because the variables do not depend on N: group blocks take the mean over the batch axis, and the kernels and BN shapes depend only on the channels. `build_model` runs with `training=False`, so it makes no BN update and has no numerical effect on the measured steps. The JSON records `build_n`.

`main.py` gets the matching rule: after today's `n_build = min(group_size, deform_group_size)` it adds `if args.split_group_conv: n_build = min(n_build, 2)`. This is a Python switch, so the default path is unchanged (the goldens do not run with split). It is needed because the build forward always takes the unsplit path (3.5). At level 1 `f0d1` that path would materialise [N, V1, 96] (N4 2.64·10⁹, N6 3.96·10⁹ fp16 elements), and default widths at N4 would give exactly 3.52·10⁹. fp32 builds exceed 2³¹ − 1 at all of these sizes. At build N 2 the largest build tensor is the default-width D1 concat [2, V1, 128] = 1.76·10⁹, which is under every bound.

The model checks the build forward too. `limit_flags` evaluates it at the record's `build_n` (default `min(N, 2)`; the m8 calibration records carry `build_n = N`) against the forward bound of the build dtype. A configuration whose build would exceed the bound is infeasible in step 2(d) of 4.4. This is how the m8 linear fp32 N12 record is predicted infeasible.

Feasibility bound used by the memory model (`limit_flags`; `--tensor-limits <tensor_limit.json>` overrides the defaults):
- Tensors that enter a backward kernel (conv inputs and output gradients, BN inputs, LeakyReLU inputs of the trainable stage): ≤ `verified_max_elements[dtype]["bwd"]`, default 2³¹ − 1 in both precisions.
- Forward-only tensors (frozen stages, validation, the eval forward): ≤ `verified_max_elements[dtype]["fwd"]`, default 2³¹ − 1 (fp32) and 3.52·10⁹ (fp16).

### 2.5 Level 1 (R1.4)

At level 1 these run at 160×224×192 (V1, 8× level 2): the stem, D1, the SVF head, the integration, every warp, the one-hot and the losses. L2 and D2 run at 80×112×96 (V2, the same size as at level 2). L3 and coarser are unchanged. Values are pre-fit, AMP, `full`, cascade, LNCC 1/1, T = 7, resident about 1.0 GiB (trainable parameters × 18 B, two frozen fp32 copies, `c_0`). Each cell is predicted peak in GiB / peak phase.

| Widths | Knobs | N2 | N3 | N4 | N5 | N6 | N7 | N8 | Largest N ≤ 44 / with 15 % margin |
|---|---|---|---|---|---|---|---|---|---|
| default | `both`, no knobs | 39.7 I | — | — | — | — | — | — | 2 / 1 (N1 22.2) |
| default | none | 34.6 F | 46.9 F | — | — | — | — | — | 2 / 2 |
| default | lean sampler (L) | 23.7 D | 35.1 D (limit ✗) | 46.4 D | — | — | — | — | 2 / 2 (N3: the D1 concat, 2.64·10⁹ elements, exceeds the backward bound) |
| default | L + split (LS) | 22.6 D | 32.9 D | 43.2 D | 53.4 | — | — | — | 4 / 3 |
| u075 | LS | 18.1 D | 26.3 D | 34.6 D | 42.8 D | 51.0 | — | — | 5 / 4 |
| **f0d1** | **LS** | 17.7 D | 25.6 D | **33.5 D** | 41.4 D | 49.4 | — | — | 5 / **4** |
| fr50 | LS | 16.4 Fw | 24.1 Fw | 31.8 Fw | 39.6 Fw | 47.3 Fw | — | — | 5 / 4 (v4: 5 / 5) |
| fr50 | L + lean losses + split (LLS) | 16.4 Fw | 24.1 Fw | 31.8 Fw | 39.6 Fw | 47.3 Fw | — | — | 5 / 4 (v4: 7 / 6) |
| u050 | LLS | 16.4 Fw | 24.1 Fw | 31.8 Fw | 39.5 Fw | 47.1 Fw | — | — | 5 / 4 (v4: 7 / 6) |
| u0375 | LLS | 16.4 Fw | 24.1 Fw | 31.8 Fw | 39.5 Fw | 47.1 Fw | — | — | 5 / 4 (v4: 7 / 6) |
| fr25 | LLS | 16.2 Fw | 23.7 Fw | 31.3 Fw | 38.9 Fw | 46.5 Fw | — | — | 5 / 4 (v4: 8 / 7) |

v5 re-check against the lean-sampler transients (2.1, timed rule, `c_loss = 1`). Each lean row is now the maximum of its v4 value and two new sub-phases.
- **Fw.** The binding call is the template one-hot warp (C = 19), the first loss call in program order. Its live set is the shared tensors: batch and fields 172 B, the kept skips (`default` 87, `u075` 66, `f0d1` 55, `fr50` 47, `u050` / `u0375` 44, `fr25` 27 B) and the cascade compositions 48 B. Adding its forward moment of 936 B gives Fw = 1243 / 1222 / 1211 / 1203 / 1200 / 1199 / 1183 B per V1 voxel per image, i.e. 7.97 / 7.83 / 7.76 / 7.71 / 7.69 / 7.69 / 7.58 GiB per image. The other calls are smaller. For example, the template image warp, C = 1, sees the template Dice term and the pair Dice share already alive: about 840 B for `f0d1`.
- **Fb.** The binding call is again the template one-hot warp. Its Dice slices and warped output have been consumed by then, so Fb = F − 150 − 220 + 76 + 240 = F − 54 B with the default Dice. With lean losses (no slices) it is F − 150 − 76 + 76 + 240 = F + 90 B. Fb is below Fw in every row.
- The default-sampler rows (`both` / `none`, no L) are unchanged (2.1).

Fw is nearly the same for all presets (≈ 7.6–8.0 GiB per image), because it is set by the one-hot warp and the batch, not by the network. Two consequences:
- **The recommended row is unchanged.** `f0d1` LS N4: Fw 32.0, Fb 25.5, D **33.5** GiB, so D still sets the peak.
- **Every lean configuration is capped** at N = 5 without the margin (N × 7.6 + 1.0 ≤ 44) and N = 4 with it. The F-bound and lean-loss rows of v4 (`fr50` N 5–6, `fr25` N 7–8) become Fw-bound and lose one to three N.

How the knobs act on the phases:
- The lean sampler moves the peak from F (sampler tapes) to D, or to Fw for the narrow presets.
- Split trims D1, so it helps the D-bound configurations.
- Lean losses lower F, but under the timed rule F no longer binds anywhere in the table (Fw ≥ F for every lean row), so they gain nothing there. They do matter under the order-independent forward bound (below), where the other loss terms count at the one-hot call. The fitted `c_loss` does not change the timed Fw, because the binding call is the first loss call. Job step 2c (`fr50` with lean losses) and ablation 3 (lean losses on at the centre widths) measure the difference.
- T = 6 vs 7 changes only phase I with the lean sampler (`f0d1` LS N4: 33.53 GiB at both T). Without it, T matters: default `both` per image is 17.1 vs 19.4 GiB in phase I.

Order-independent forward bound (`fwd_transient_bound=True`): Fw = F − 150 − 220 − 12 + 936 = **F + 554 B** with the default Dice (own-term tensors after the call: warped output 76 + Dice slices 144). With lean losses it is F − 150 − 76 − 12 + 936 = F + 698 B, on a much smaller F.
- `f0d1` LS N4: 1562 B → 10.01 GiB per image → **41.0 GiB (47.2 with the margin)**. That is ≤ 44 without the margin but infeasible with it.
- `f0d1` LLS N4: F ≈ 628 B → 1326 B → **35.0 GiB (40.3 with the margin)**. Feasible, so under the bound the N = 4 row keeps the `f0d1` widths and adds lean losses.
- N3 is unchanged (`default` LS, D 32.9; its bound value is about 32.6). N ≥ 5 is infeasible with the margin.

The bound is printed next to the timed rule, and the measurement decides between them:
- the hold-out records with the lean sampler (4.3) enter the margin;
- job step 2 measures `f0d1` LS at N 2…8, and ablation 3 measures `f0d1` LLS at N 4;
- the measured-N override of 6.4 step 4 uses measured peaks.

If the measured lean peaks follow the bound rather than the timed rule, the first remedy is the backlogged forward corner sequencing (3.7). It does not change values and brings the forward moment of the one-hot warp from 936 B to about 400 B, which puts Fw below F and D again and restores the v4 table.

Per-phase split of the expected configuration (`f0d1` LS, N4, per image): F 6.46, Fw 7.76 (timed; 10.01 under the order-independent bound), Fb 6.12, I 4.67 (v4 2.83: the sequenced lean backward and the recomputed composition's forward moment, 2.1), **D 8.13**, E 1.86 GiB. D (at the D1 backward) breaks down per V1 voxel as:
- batch and fields, 172 B;
- skips kept (forward skips, the recompute inputs), about 55 B;
- decoder recompute: D1 (4·48 + 48/4 channels) and the head, about 490 B; D2 and coarser, about 170 B;
- D1 backward transient, 384 B (covers D1's input gradient, including its stem-skip slice);
- skip gradients `B_sg` of finer blocks: 0 B (D1 is the first decoder backward, 2.1);
- total about 1270 B.

Later D sub-phase (D2 backward): the D1 / head tapes are freed (−490 B), d(stem skip) appears (16 ch fp16, 32 B), the transient is D2's at V2 (2 · 128 · 4 / 8 = 128 B per V1 voxel) → about 700 B, well below the D1 moment. E: encoder recompute tape + all skip gradients (stem 32 B, L2 16 B, L3 and coarser 4 B per V1 voxel ≈ 52 B) + transient → 1.53 + 0.33 = 1.86 GiB per image, also below D.

Conservative bound (review v3 F9): counting all ≈ 52 B of skip gradients at the D1 moment gives D = 1322 B (8.47 GiB per image) and `f0d1` LS N4 = **34.9 GiB (40.1 with the margin)**, N5 43.1. N = 4 stays feasible on the 48 GB cards with the margin and on the A100 without it (34.9 ≤ 36). The same bound moves other D-bound rows of the table by (f0 · 2 + f1 · 2/8 + f2 · 2/64) B per V1 voxel per image (default 84, u075 63, fr50 44, u050 42, u0375 40, fr25 24 B); it makes `default` LS N4 infeasible (45.4 > 44; it already was with the margin). In v4 it also moved the N6 row. In v5 there is no N6 row (Fw caps the lean rows), and the Fw-bound rows are unaffected because the skip gradients are counted in D only. The fitted model uses the timed rule of 2.1; the bound is reported next to it.

Linear tasks at level 1 (AMP, no recompute, `full` does not apply): default N 4/8/12/16 = 4.8 / 8.7 / 12.6 / 16.4 GiB; `f0d1` 3.9 / 6.9 / 9.8 / 12.7; `fr50` 3.8 / 6.6 / 9.4 / 12.2. Linear N = 16 fits both cards for every preset. This answers review F12.

Largest single tensors at level 1 that enter backward kernels (elements; the bound is 2.147·10⁹ until the probe says otherwise):

| Tensor | default | `f0d1` | `fr50` | `fr25` |
|---|---|---|---|---|
| D1 group concat [N, V1, 2·(d2 + f0)], no split | N4 3.52·10⁹ ✗, N2 1.76·10⁹ | N4 2.64·10⁹ ✗, N3 1.98·10⁹ | N4 1.76·10⁹, N5 2.20·10⁹ ✗ | N8 1.76·10⁹ |
| D1 conv_a input [N, V1, d2 + f0], split | N4 1.76·10⁹, N5 2.20·10⁹ ✗ | N6 1.98·10⁹, N7 2.31·10⁹ ✗ | N9 1.98·10⁹ | N ≤ 19 |
| Stem output gradient [N, V1, f0] (linear N16) | 3.52·10⁹ ✗ (linear N ≤ 9) | 1.76·10⁹ | 1.76·10⁹ | 0.88·10⁹ |
| One-hot [N, V1, 19] fp32 (sampler gather rows, backward segment sum) | N8 1.05·10⁹ | same | same | same |
| Frozen level-2 D2 concat fp32 [N, V2, 2·(d3 + f1)] (inference only) | N8 1.76·10⁹ | N8 1.76·10⁹ | N8 1.32·10⁹ | N8 1.10·10⁹ |

So without split, the default and `f0d1` presets cannot reach N = 4 under the conservative bound. With split, the training-step tensors of `f0d1` stay under 2³¹ up to N = 6, which covers its whole feasible range. The build forward is unsplit, but it runs at `min(N, 2)` (2.4), so it is not a constraint. The default preset would cap linear N at 9, while `f0d1` / `fr50` keep linear N = 16. The int32 index math in `sample_voxel_coords` (max N·V1 = 5.5·10⁷ rows at N = 8) is far from its limit.

## 3. Memory knobs (R2)

All new options default to today's behaviour. None of them is a `ModelConfig` field, so the `config.json` files of `m7_three_task_lncc` and `m8_level2_gradnorm` restore unchanged, and `check_cascade_config` / `copy_weights_from` / `warm_start_model` see no new keys.

### 3.1 Widths (R2.1)

`ModelConfig` keeps its fields, and `resolved_decoder_widths()` is unchanged (None → half the matching encoder widths). `ModelConfig.validate()` gains one check: every entry of `filters`, `reductions` and `dense` is an integer ≥ 1 (ValueError naming the field). Existing configs all pass. `runtime/args.py` adds the same check with `parser.error` for `--filters`, `--reductions` and `--dense`. The counts stay as today: 6 / ≥ 1 / ≥ 1. `--decoder-widths` (5 entries ≥ 1) is already checked.

New TF-free module `models/width_presets.py`:

- `PRESETS` maps a name to `dict(filters=..., decoder_widths=...)`, always both explicit:

| Name | filters (stem, L2..L6) | decoder (D5..D1) | Trainable parameters (approx., incl. dense) |
|---|---|---|---|
| `default` | (32, 64, 128, 192, 256, 384) | (128, 96, 64, 32, 16) | 29.63 M (built: encoder 9.85, linear decoder 12.64, deform decoder 7.14) |
| `u075` | (24, 48, 96, 144, 192, 288) | (96, 72, 48, 24, 16) | ≈ 22.1 M |
| `u050` | (16, 32, 64, 96, 128, 192) | (64, 48, 32, 16, 8) | ≈ 16.8 M |
| `u0375` | (16, 24, 48, 72, 96, 144) | (48, 40, 24, 16, 8) | ≈ 14.9 M |
| `f0d1` (full-res only: f0, D1) | (16, 64, 128, 192, 256, 384) | (128, 96, 64, 32, 8) | 29.54 M (built) |
| `fr50` (full + half res: f0, f1, D2, D1 halved) | (16, 32, 128, 192, 256, 384) | (128, 96, 64, 16, 8) | ≈ 29.1 M |
| `fr25` | (8, 16, 128, 192, 256, 384) | (128, 96, 64, 8, 8) | ≈ 28.9 M |

`default` and `f0d1` were counted by building the level-4 models locally (`tf_gpu`, `model.trainable_variables`). v2's hand formula undercounted both by about 1.0 M. The ≈ rows carry that offset and are approximate. The ranking, which is all the selection rule uses, is the same. `count_params` is checked against the built models (test below). The resident terms of 2.3 and 4.2 use the built counts.

- Uniform rule `uniform(m)`: every width w → max(8, 8·floor(w·m/8 + 0.5)), multiples of 8 for fp16 tensor cores. It is applied to the default filters and decoder widths. The table is generated by it and pinned by a test.
- `preset_args(name)` returns the `--filters` / `--decoder-widths` CLI strings; `preset_kwargs(name)` returns ModelConfig kwargs; an unknown name raises a ValueError listing the names. `count_params(model_config_dict)` (TF-free, the same formula as the memory model) gives the parameter column. A TF test checks it against `model.count_params()` within 1 % for every preset.

Training scripts and the measurement tool pass the explicit lists. `config.json` stores them, so all three levels carry identical widths and the cascade check applies unchanged. Reductions (192, 96, 48, 24) and dense (2048, 512) stay, so the bottleneck stays 5×7×6×24. The parameter count is dominated by the dense layer and the coarse stages, which is why the targeted presets keep about 28 M parameters.

### 3.2 `--gradient-checkpointing full` and BN updated once (R2.2)

`models.group_net.GRADIENT_CHECKPOINTING` becomes `("none", "integrator", "decoder", "both", "full")`, and the `runtime/args.py` choices follow. Like the other modes, `full` applies to the deform task only; linear calls are unchanged (their level-1 cost is in section 2.5, linear row). `--gradient-checkpointing full` without `deform` in `--task-ratio` fails with `parser.error("--gradient-checkpointing full needs 'deform' in --task-ratio")`.

New module `models/checkpointing.py`:

```python
def recompute(fn, bn_layers=()):
    """tf.recompute_grad(fn) whose recompute passes leave the BN moving statistics as the forward
    pass left them (exactly one update per step). fn(*tensors) -> tensor or flat tuple."""
```

Implementation: the returned function wraps `fn` in a closure with a call counter, created per model call. Call 1 is the forward: `fn(*args)` as is. **Every later call** (the recompute inside the gradient, and any further `tape.gradient` or diagnostic gradient; review F11) runs:
1. `saved = [tf.identity(v) for v in stats]`, where stats are `moving_mean` and `moving_variance` of every `BatchNormWithSeparateAffine.bn` in `bn_layers`;
2. `out = fn(*args)`;
3. `with tf.control_dependencies(tf.nest.flatten(out)): v.assign(s)` for each pair;
4. return `out`.

A forward without a gradient (inference, `recalibrate_bn`) calls the closure once and never restores. There is no switch for the restore (v2's `restore_bn` parameter is dropped, review F6). With `bn_layers=()` nothing is snapshotted, which is the integrator's case. The whole wrapped closure goes through `tf.recompute_grad`. Ordering: the forward's moving-average assigns, the snapshot reads, the recompute's assigns and the restores all touch the same resource variables and are created in that order, so tf.function's automatic control dependencies serialise them (U8). The recompute normalises with the batch statistics (BN training mode), which the restore does not touch, so the gradients equal the non-checkpointed ones.

Restore skipping (trace time, Python): no snapshot or restore ops are built when `training` is a Python False/None, or when every layer in `bn_layers` has `stats_frozen=True` (cascade after recalibration). Nothing is updated in either pass then. If `training` is a tensor, the ops are built (harmless).

Build guard (review F19): `full` takes the recompute path only if `self.encoder.built and self.deform_decoder.built`. Otherwise the whole deform forward runs directly, as today's decoder guard does. During `build_model` the encoder is built by the rigid pass but the decoder is not, so the deform build pass runs directly.

`GroupRegLevelModel._call_deform` under `full` with both parts built:
1. `skips = recompute(lambda x: encoder(x, training)[1], encoder_bn)(x)` returns the 6 skips (bottleneck = skips[5]).
2. `v_raw, dec_feat = recompute(lambda *s: decoder(list(s), training), decoder_bn)(*skips)`.
3. `v = center_velocities(v_raw)`, then `u = tf.recompute_grad(lambda w: integrate_svf(w, T, lean))(v)` and `u_inv = tf.recompute_grad(lambda w: integrate_svf(w, T, lean))(-v)`. These are two blocks, so the backward holds one integration's tape at a time. There is no BN inside.

`encoder_bn` / `decoder_bn` come from a new helper, `layers.batch_norm_layers(layer)`: the Trainer's `_bn_layers` logic moves to `models/layers.py`, and the Trainer calls it. The encoder and decoder have no dropout, so the recompute is deterministic.

BN rule (documented in design 4.5):
- Under `full`, every BN moving statistic is updated exactly once per training step, from the forward pass, however many gradient calls follow.
- With frozen statistics they are never updated.
- `recalibrate_bn` (eager, tape-free forward) makes one update per layer and batch.
- `decoder` / `both` keep their documented double update, so the m8 recipe reproduces. They are not changed.
- The Trainer's restore of the pre-step statistics on a non-finite step is unchanged and composes with the BN-once restore.

Availability: `main.py` passes `args.gradient_checkpointing` to the model constructor, so nothing changes beyond the choices. `FrozenStage` stays `'none'`. The measurement tool accepts `full`.

### 3.3 `--lean-sampler` (R2.4, adopted)

`layers.sample_voxel_coords(..., lean=False)`: `lean=False` executes today's code unchanged (golden path). `lean=True` with `interpolation="linear"` calls `_sample_linear_lean(images, coords, fill_value, boundary)`, a `tf.custom_gradient` function.

Forward:
- The same float operations in the same order as today's linear branch, except that the corner indices are computed in int64, which stays on the GPU (U4).
- The values are bitwise identical: the integer math is exact and the same rows are gathered.
- Only `images` and `coords` are captured by the gradient closure, so no corner tensor outlives the forward.

Backward(dy):
- First a `dy`-dependent zero, the same construction TF 2.10's `tf.recompute_grad` uses for its block inputs (review v3 F1, U18). Without it, Grappler's arithmetic optimizer would deduplicate the recomputed Floor / Cast / Clip / Mul / GatherV2 nodes with the identical forward nodes inside the `tf.function`, and the forward's corner weights and gathers would stay alive until the backward, which is the default sampler's memory profile:

```python
def grad(dy):
    # dy-dependent zero (as tf.recompute_grad, TF 2.10 custom_gradient.py): the recomputed corner
    # math below depends on dy, so Grappler cannot merge it with the forward ops.
    elem = tf.reduce_max(tf.reshape(dy, [-1])[:1])
    elem_ok = tf.cast(elem, tf.bool)
    dep = tf.where(tf.equal(elem_ok, elem_ok), 0.0, float("nan"))   # always 0: data dependency only
    coords_b = coords + tf.cast(dep, coords.dtype)
    flat = tf.reshape(images, [-1, C])          # a view of the captured input; no copy
    lo, frac, valid = _axis_terms(coords_b)     # per axis, int64 lo; nothing per corner yet
    d_coords = tf.zeros_like(coords_b)
    d_flat = tf.zeros_like(flat) if images_grad else None   # images_grad = lean_images_grad (Python bool)
    for k in range(8):                          # bits (bx, by, bz) in the forward's corner order
        accs = [d_coords] if d_flat is None else [d_coords, d_flat]
        with tf.control_dependencies(accs):     # corner k starts after corner k-1 is accumulated
            idx_k, w_k, dw_k = _corner(lo, frac, valid, k)   # int64 [N*V'], [N,V'], 3 x [N,V']
            g_k = tf.gather(flat, idx_k)                      # [N, V', C], transient
            s_k = tf.reduce_sum(dy * g_k, -1)
            d_coords = d_coords + tf.stack([s_k * dw_k[a] for a in range(3)], -1)
            if d_flat is not None:
                d_flat = d_flat + tf.math.unsorted_segment_sum(
                    tf.reshape(dy * w_k[..., None], [-1, C]), idx_k, tf.shape(flat)[0])
    ...  # fill / border corrections, cast, reshape; returns (d_images or None, d_coords)
```

  Only `coords` gets the dependency. Every recomputed node (lo, frac, the corner weights, the int64 indices) descends from `coords_b`, and every `gather(flat, idx_k)` node has an `idx_k` input that descends from it, so no recomputed node has the same inputs as a forward node. `flat` itself is a reshape of the captured `images`, which the forward also keeps, so merging that reshape is harmless. This deviates from the review's sketch (`flat + dep` as well): perturbing `flat` would materialise a second copy of the input (4C B per voxel in the backward) for no extra protection. The cost of the dependency is one [N, V, 3] add (12 B per voxel, transient). Mechanism, corrected in v5 (review v4 F2): `dep` is always 0. `elem` is only the first element of `dy`, `tf.cast(NaN, bool)` is True, and `equal(x, x)` on booleans is True, so `dep` only adds a data dependency. A non-finite `dy` still makes the gradient non-finite, because it propagates through the products dy · g_k and dy · w_k, and the Trainer's existing non-finite step skip handles it. The outcome is unchanged from v4; only the stated mechanism was wrong. The dependency is added in eager mode too (no branch on `tf.executing_eagerly()`); there it has no effect besides the add.
- The forward gets no extra dependency, so its graph and values are those described above.
- It recomputes lo, frac and the per-axis validity once from `coords_b` (`_axis_terms`, the fixed part of T_bwd). Each corner's clipped int64 index, weight w_k and the three partials ∂w_k/∂f_a are formed inside that corner's block (`_corner`). k's bits select i0 / i1 and w0 / w1 per axis, in the forward's loop order x, y, z, so the corner order matches the forward.
- **Sequencing (v5, review v4 F1; orchestrator decision).** The 8 corners are accumulated **sequentially**: every op of corner k is created under `tf.control_dependencies` on the accumulators produced by corner k − 1 (d_coords, plus d_flat when it is computed). Corner k's gather therefore cannot be enqueued before corner k − 1's contributions are summed, and corner k − 1's g_k, products and segment sum have then been consumed and freed. At most one corner's temporaries are alive at a time, which is what T_bwd (2.1) assumes. The edges are not implied by any data path, so Grappler's dependency optimizer keeps them (U19). They also make the per-corner ops distinct nodes with distinct control inputs, so no corner can be merged with another one or with the forward. Together with the `dy` dependency above, the Grappler CSE guard of v4 is **kept unchanged**.
- Per corner: g_k = gather(flat, idx_k) (transient) and s_k = reduce_sum(dy · g_k, −1). It accumulates d coords_a += s_k · ∂w_k/∂f_a, with ∂w0/∂f = −v0 and ∂w1/∂f = +v1 in zero mode, −1/+1 in border mode. When the input gradient is needed it also accumulates d flat += unsorted_segment_sum(dy · w_k, idx_k, N·V).
- **Input gradient (v5).** `images_grad` is a Python argument of the lean path (`sample_voxel_coords(..., lean=True, lean_images_grad=True)`). It is needed because the control chain would keep an unused d_flat branch alive (control edges count as fan-in, so Grappler cannot prune it). With `lean_images_grad=False` the gradient for `images` is `None` and no d_flat op is built. The default is True, which is always correct. The losses pass False for their data-volume warps (the template and pair one-hot warps, the LNCC image warps). Field compositions keep True. The test below checks that the d coords are identical under both settings.
- Fallback, used only if case C of the memory test fails with the chain: the same per-corner body inside `tf.while_loop(cond k < 8, body, loop_vars=(k, d_coords, d_flat), parallel_iterations=1)`. In that form `_corner` selects the bits with `k // 4 % 2`, `k // 2 % 2` and `k % 2` through `tf.where` on the per-axis lo / lo + 1 and w0 / w1, and when `lean_images_grad=False` the loop carries only d_coords. The forward is not affected either way. The implementer records in the report which form was used.
- `fill_value ≠ 0` (zero mode): d coords −= fill · reduce_sum(dy, −1) · ∂wsum/∂f.
- Border mode: d coords is masked like TF's `clip_by_value` gradient (zero where coords < 0 or > n − 1, passed at equality).
- It returns d images, reshaped and cast to the images dtype, and d coords. Unused gradient outputs (data volumes) are pruned by Grappler.

Other rules:
- `nearest` mode is unchanged (no gradient), and `lean` is ignored there by design (documented; there is nothing to save).
- Plumbing gains `lean=False` in: `diffeo.warp`, `warp_field`, `compose`, `integrate_svf`, `exp_and_inverse`, `pairwise_displacement` and `compose_cascade`. `diffeo.warp` also gains `lean_images_grad=True` (v5), which is passed through to `sample_voxel_coords` and ignored when `lean` is False.
- The losses `losses.deform_loss`, `pair_dice_loss`, `template_ncc_loss` and `pair_ncc_loss` gain `lean_sampler=False` and pass it to their warps.
- `GroupRegLevelModel(..., lean_sampler=False)` is a constructor attribute like `gradient_checkpointing` and passes it to the integrator.
- The Trainer passes `cfg.lean_sampler` to `compose_cascade` and `deform_loss` in the deform step only. Validation keeps `lean=False` (no tape, same forward values).
- Consistency: `Trainer.__init__` raises `ValueError("lean_sampler: model %s vs TrainConfig %s")` if `model.lean_sampler != cfg.lean_sampler`.

Memory per call (v5, replaces v4's "2·(4C + 12) + 12 B" backward-only figure; formulas in 2.1):
- retained under the tape: the coordinates and the captured input;
- forward moment while the call runs: T_fwd(C) + 4C = 252 + 36·C B (936 B for the one-hot);
- backward moment: dy + T_bwd(C) = 4C + 88 + 8C (+ 16C with the input gradient): 316 B for the one-hot, 172 B for a field composition.

The same holds inside the recomputed integrator blocks of `full`: `tf.recompute_grad` protects the block inputs with its own `dresult` dependency, and each lean composition inside the block protects its own corner math with the `dy` it receives from the block's inner tape.

Tolerance: forward max |diff| 0 vs `lean=False`, compared **eagerly** (no `tf.function`), so no Grappler rewrite can move values (review v3 F4; the m8 T6 precedent showed ~1e-8 Grappler differences between different graphs). A second forward comparison inside `tf.function` runs under `_GrapplerOff` + `_OpDeterminism` (from `tests/test_cascade.py`) and must also give 0. The test docstring states both modes. For gradients the fp32 target is ≤ 1e-6 relative (L2 norm of the difference over the norm), the test threshold is 1e-4 (R2.4), and float64 inputs must reach ≤ 1e-12.

Graph-mode memory test (review v3 F1, rewritten in v5 for review v4 F1), `tests/test_lean_sampler.py::test_graph_mode_memory`.

Setup:
- Local GPU; skipped without one (`tf.config.list_physical_devices("GPU")` empty → `skipTest`).
- Grappler at its defaults (no `_GrapplerOff`), float32.
- **Level-2-size field**: grid 80×112×96 (V2 = 860 160), N = 2. So 1 B per voxel per image = 1 720 320 B = **0.001602 GiB**.
- Inputs are created before the measurement: the one-hot, the field `tf.Variable`s (`tests/field_helpers.smooth_field`, max 2 voxels, seed 0), the fixed random weight volumes w and, for case C, dy and coords.
- Measurement per case and flag:
  1. one call (trace and warm-up), then delete its outputs;
  2. `tf.config.experimental.reset_memory_stats("GPU:0")`;
  3. `base = get_memory_info("GPU:0")["current"]` (this includes the inputs);
  4. a second call;
  5. `peak = get_memory_info("GPU:0")["peak"] − base` ("peak minus inputs").
- Every modelled value M below is computed in the test from the 2.1 formulas, so the assertion and the inventory cannot drift apart. The test prints M and the measured peak for every case.

Cases:
- **Case A, one-hot warp, transient check.** `tf.function(lambda: gradient of reduce_sum(warp(onehot, u, lean=flag, lean_images_grad=False) · w) w.r.t. u)`, C = 19, zero mode.
  - Lean model: M_A = max(forward moment 252 + 36·19 = 936, backward moment 12 + 76 + 240 = 328) = **936 B → 1.50 GiB**.
  - **Assert** `peak_lean ≤ 1.5 × M_A = 2.25 GiB`.
  - Default, printed only: ≈ 920–1450 B → 1.5–2.3 GiB. In a single isolated call the lean peak is the forward transient, so there is little saving here. The saving is in what stays taped across calls (case B). This is why v4's "≤ 0.6 × default" assertion is dropped: it would fail for case A by construction.
- **Case B, integrator block as under `full`, saving check.** `tf.recompute_grad(lambda v: integrate_svf(v, 7, lean=flag))` on [2, 80, 112, 96, 3]; gradient of a weighted sum w.r.t. v.
  - Lean model: M_B,lean = the recompute's last forward moment = 6 × 24 retained + 12 (v / 2^T) + 360 = **516 B → 0.83 GiB**. The backward moment, 168 + 12 + 12 + 160 + 24 = 376 B, is smaller.
  - Default model: M_B,def = 7 × 180 taped + 12 + 12 output + 12 dy + 250 transient = **1546 B → 2.48 GiB**.
  - **Predicted saving of the lean sampler: 1030 B → 1.65 GiB (1.77 GB).**
  - **Assert** `peak_lean ≤ 1.5 × M_B,lean = 1.24 GiB`.
  - **Assert** `peak_def − peak_lean ≥ 0.85 GiB`. This floor is the predicted saving less two allowances: the 1.5× allowance on the lean side (0.41 GiB) and the 15 % model band on the default side (0.37 GiB).
  - If Grappler merged the recompute with the forward (the v3 F1 failure), the lean case would keep the default tape, and both assertions would fail.
- **Case C, the backward alone, sequencing check.** The grad body is factored as the module function `_lean_linear_backward(images, coords, dy, fill_value, boundary, images_grad)`, which `grad(dy)` calls. The test runs that function alone in a `tf.function`. Without the forward there is no forward transient, so the peak is the backward transient alone.
  - C1, one-hot C = 19, zero mode, `images_grad=False`: M = T_bwd = **240 B → 0.38 GiB**. **Assert** `peak ≤ 0.58 GiB`. Unsequenced, it would be ≈ 1500 B → 2.40 GiB.
  - C2, field C = 3, border mode, `images_grad=True`: M = T_bwd + the border mask 12 = **172 B → 0.28 GiB**. **Assert** `peak ≤ 0.41 GiB`. Unsequenced, it would be ≈ 870 B → 1.40 GiB.
  - These cases separate the sequenced and unsequenced backward by 4–6×, which the 1.5× tolerance cannot hide. If C1 or C2 fails, the backward switches to the `tf.while_loop` form above, and the test is rerun.
- **Gradient equality of `lean_images_grad`.** Eager, no memory measurement: `lean_images_grad=False` returns `None` for images and the same d coords (max |diff| 0) as `lean_images_grad=True`.
- **Information only** (printed, not asserted): case B with the private switch `_DY_DEP = False` (module constant, tests only), which drops the dependency, so the report can state whether Grappler's dedup was actually observed in TF 2.10.

The largest allocation is the default case B (≈ 2.5 GiB plus about 40 MB of inputs), well inside the 8 GB GTX 1080. Its time is about 1 min, as in v4.

The hold-out measurements (5.6) run the tool's traced training step, so they check the same thing at model scale.

### 3.4 `--lean-losses` (R2.4, adopted)

Signatures (review F5). Every new parameter defaults to today's path:
- `losses.deform_loss(..., lean_sampler=False, lean_losses=False)` passes `lean_losses` on to the three functions below and uses `_soft_dice_lean` for the template Dice when it is True;
- `losses.pair_dice_loss(..., lean_sampler=False, lean_losses=False)` uses `_soft_dice_lean` for the pair Dice;
- `losses.template_ncc_loss(..., lean_sampler=False, lean=False)` and `losses.pair_ncc_loss(..., lean_sampler=False, lean=False)` wrap `local_ncc` in `tf.recompute_grad` when `lean` is True. `local_ncc` itself is unchanged.

Hand-off: the Trainer's `_make_deform_step` passes `lean_sampler=cfg.lean_sampler, lean_losses=cfg.lean_losses` to `deform_loss`. Every validation path (`_val_deform_fn`, `_deform_errors`, the LNCC validation metrics) keeps the defaults (False). Validation has no tape, so the lean variants would save nothing, and the validation numbers stay comparable with m7 / m8.

With `lean_losses=True`:

- Lean `soft_dice` variant `_soft_dice_lean(a, b)`: `inter = reduce_sum(a * b, [1, 2, 3])` and `total = reduce_sum(a, [1,2,3]) + reduce_sum(b, [1,2,3])` over all channels. b is the [1, X, Y, Z, C] template, used through implicit broadcasting and never `broadcast_to`. Then `[..., 1:]` on the [N, C] results; presence mask and mean as today.
- `local_ncc` is evaluated as `tf.recompute_grad(lambda a, b: local_ncc(a, b, window, eps))(a, b)` in both LNCC terms.
- Tolerance: losses ≤ 1e-6 relative in fp32, gradients ≤ 1e-4 relative. The default path is bit-identical (goldens).
- `--lean-losses` without `deform` in `--task-ratio` is a parser error.

### 3.5 `--split-group-conv` (R2.4, adopted)

`GroupConvBlock` gets a Python attribute `split_group = False`, read at trace time. With `True`, `group=True` and `self.conv.built`:
1. x → norm → act (as `_pre`, without the concat);
2. `mean = reduce_mean(x, 0, keepdims=True)` and `k = tf.cast(self.conv.kernel, x.dtype)`;
3. `y = conv3d(x, k[..., :C, :]) + conv3d(mean, k[..., C:, :])` with the layer's strides, `"SAME"`, NDHWC;
4. `tf.nn.bias_add(y, cast(bias))`.

Variables, names and checkpoints are unchanged: the first call (`build_model`) always takes the standard path, so the variables keep today's names. That unsplit build forward runs at `min(N, 2)` in both the tool and `main.py` (2.4), so its concat stays at ≤ 1.76·10⁹ elements at level 1. Non-group blocks and the dense blocks are unchanged.

Stem (v5, review v4 F5): blocks with `pre_activation=False` are **excluded**. The stem is a `GroupConvBlock(group=True, pre_activation=False)` whose concat has only 2 channels, so it gains nothing from the split. `set_split_group_conv` sets `split_group` only on `GroupConvBlock`s with `group=True and pre_activation=True`, and the stem always takes the standard path. The inventory keeps the stem's 2-channel concat (2.1). `tests/test_split_group_conv.py` asserts that the stem's `split_group` stays False after `set_split_group_conv(model, True)`.

Setter `layers.set_split_group_conv(layer, on)`, like `set_bn_stats_frozen`. `GroupRegLevelModel(..., split_group_conv=False)` calls it after construction. Frozen stages keep `False`. The knob is valid for every task.

Tolerance: outputs ≤ 1e-6 relative in fp32 (≤ 1e-3 in fp16), gradients ≤ 1e-4 relative. The default path is bit-identical.

Split is not only a memory knob. It keeps the D1 conv input of `f0d1` under 2³¹ elements at N ≤ 6 (2.5).

### 3.6 Integration steps 6 vs 7 (R2.3, R3.5)

Decision: Option B. T is set once for the retraining and used at all levels, and the cascade check keeps requiring identical ModelConfigs. Every level is retrained with the reduced widths anyway, so allowing a per-stage T would add a config-check exception (and a fingerprint/resume rule) for no benefit.

Memory saving of T = 6:
- With the lean sampler: 24 B per voxel per image per removed composition, in phase I only (about 0.15 GiB per image at level 1). Phase I is not the peak, so this saves nothing in the recommended configurations.
- Without the lean sampler: 180 B (about 1.2 GiB per image) in phase I or F.

Local measurement (review F9), a one-off step done together with the hold-out runs (5.6) on the GTX 1080 with `TF_CUDNN_USE_AUTOTUNE=0`:

```bash
python tests/gpu/measure_level_memory.py --level 2 --cascade true --tasks deform --widths tiny --n 2 \
  --amp true --integration-steps 6,7 --lean-sampler false,true --gradient-checkpointing integrator,full \
  --warmup-steps 1 --steps 3 --gpu-memory-limit-mb 6000 --out .agents/tasks/milestone-9/memory_holdout.json
```

That is 8 records, merged into `memory_holdout.json` (the `integration_steps` key field keeps them apart from the other hold-out records). The report prints the measured T6 − T7 difference in GB per image for each (sampler, checkpointing) pair, next to the model's prediction at level 2 tiny and at level 1 `f0d1`. With the lean sampler the expected difference is about 0, because phase I is not the peak. The `integrator` + default-sampler pair shows the full saving, about 180 B per voxel per image. These records are information, not hold-out gate points; the gate is in 4.3.

Accuracy test `tests/test_integration_steps.py` (CPU or GPU, float32). It imports `tests.tf_setup` first, like every TF test, which also turns TF32 off. It can also be run as a script (`python -m tests.test_integration_steps --max-v 8 --out <json>`), which writes the verdict:
- Fields from `tests.test_diffeo.smooth_field` (moved to `tests/field_helpers.py` and imported by both), n = 1, max |v| ∈ {2, 4, 8} voxels.
- Grids: level 4 (40×56×48, sigma 3) and level 1 (160×224×192, sigma 6). T ∈ {6, 7}.
- Printed per case: max and mean |exp(v) ∘ exp(−v) − id| and |exp(−v) ∘ exp(v) − id|, with an interior margin of ceil(max |u|) + 3 voxels as in `test_two_level_round_trip`, and `folding_fraction(exp(v))`.
- Assertions only for the documented case: level 4, max 2, T 7 must give max < 0.1 and mean < 0.01.

The verdict is computed for max |v| = 8 voxels on the level-1 grid. This magnitude is a conservative choice, not an estimate (review F8). The m8 level-2 runs log `train_deform/diag_u_res_max_vox` in `metrics.jsonl`. That value is a 25-step window mean of the per-step maximum residual displacement (review v3 F3), so percentiles of it underestimate per-step percentiles. Job step 0c extracts the **maximum over windows** with step ≥ 1000 from `results/m8_level2_gradnorm` (`step_times_from_metrics.py --extra-keys diag_u_res_max_vox`, which reports median and max of window means). The report states that maximum × 2, in level-1 voxels. The SVF magnitude is not logged, and |v| ≤ |u| is not guaranteed in general. If 2 × max(window means) exceeds 8, the test is rerun with `--max-v` set to it (rounded up) and the verdict comes from that run. Let e(T) = max over the two orders (exp(v) ∘ exp(−v) and exp(−v) ∘ exp(v)) of the mean interior inverse error. `t6_accuracy_ok` holds iff:
- e(6) ≤ 2 × e(7);
- e(6) ≤ 0.05 voxel;
- folding (T6) ≤ max(0.1 %, folding (T7) + 0.05 percentage points) (review F15).

It is written to `scratch/m9/integration_steps.json` (`--out`; default `$BGR_SCRATCH/m9/integration_steps.json`, else the repo's `scratch/m9/`) as `{"t6_accuracy_ok": bool, "cases": [...], "criteria": {...}}`, and printed. The report copies it, and `recommend --t6-verdict` reads it (4.4).

### 3.7 Candidates not adopted

| Candidate | Decision | Reason |
|---|---|---|
| Chunked one-hot creation / label warp with chunked Dice | Backlog | After the lean sampler the one-hot costs 76 B (input) + 76 B (warped, needed by the Dice gradient) + 12 B per image, in F only. Chunking helps only if each chunk's Dice is recomputed too, which needs the full-group template first (two passes). Revisit with per-step label sampling. |
| Per-squaring-step checkpointing | Not needed (conditional on the 3.3 graph-mode test) | With the lean sampler one integration block keeps 7 × 24 B in phase I, which is not the peak. If the graph-mode test (case B) failed, phase I would return to the default-sampler size and this becomes the next knob. |
| Splitting full-res convs over the batch axis | Backlog, conditional | Only if the probe's backward bound is below the recommended configuration's largest tensor. `f0d1` + split at N = 4 is 1.32·10⁹. |
| fp16 one-hot | Backlog | Saves 38 B per image; the warp computes in float32. |
| Forward corner sequencing in the lean sampler (v5) | Backlog, conditional; the first remedy if the measured lean peaks follow the order-independent forward bound (2.5), or if Jay wants N ≥ 5 at level 1 with the margin | The same control-dependency chain as the backward, applied to the lean forward's partial sums (`out`, `wsum`). The forward sum is already left to right, so values stay bitwise identical. It brings the forward moment from 252 + 36·C to about 170 + 12·C B (936 → about 400 B for the one-hot), which puts Fw below F and D and restores the v4 per-N table (`fr50` N 5–6, `fr25` N 7). It is not adopted in phase 1, by decision: the forward stays today's op graph. |

### 3.8 Flags and validation (R2.6)

| Flag / field | Type, default | Validation (parse time) | Where read |
|---|---|---|---|
| `--gradient-checkpointing` | choice, `none` | choices include `full`; `full` needs `deform` in `--task-ratio` | model constructor |
| `--lean-sampler` | `str2bool`, false | needs `deform` in `--task-ratio` | model constructor, `TrainConfig.lean_sampler` |
| `--lean-losses` | `str2bool`, false | needs `deform` in `--task-ratio` | `TrainConfig.lean_losses` |
| `--split-group-conv` | `str2bool`, false | none beyond bool parsing | model constructor |
| `--filters`, `--reductions`, `--dense` | int lists, today's defaults | entries ≥ 1, counts as today | ModelConfig |
| `--integration-steps` | int, 7 | ≥ 0 (existing) | ModelConfig |
| `--val-pair-chunk` | int, 16 | ≥ 1 (existing) | Trainer |

Invalid values fail with `parser.error` naming the option; `str2bool` errors already name the flag via argparse. `TrainConfig` gets `lean_sampler: bool = False` and `lean_losses: bool = False`, and `main.py` fills them from the args. The new flags appear in the `config.json` `args` automatically. Resume does not check them, because they are memory and numerics knobs, not variables.

### 3.9 Bit-identity and compatibility (R2.5)

With every new option at its default, no new op is created: the lean paths, the split path, the `full` branch and the BN restore are Python-level switches. The goldens `tests/fixtures/m7_linear_golden.json` and `tests/fixtures/lncc_deform_golden.json` must report max |diff| 0. A new test restores a ModelConfig dict and an `args` dict of today's format (no new keys) and gets today's defaults. `check_cascade_config` accepts today's level-4 → level-2 pair.

## 4. Calibrated memory model (R3)

### 4.1 Module, CLI and the calibrated quantity

The model lives in `models/memory_model.py`. It is TF-free (stdlib + numpy; scipy only inside `fit`).
- CLI: `scripts/memory_model.py {fit, predict, grid, recommend}`. It prints tables and writes JSON (`--out`).
- Calibration data: `models/memory_calibration.json`.
- Fitted coefficients: `models/memory_coefficients.json`, written by `fit` and read by default by every other command and by the measurement tool.

Calibrated quantity (review F9): every calibration and hold-out point is the tool's `peak_mb`, the steady state after the warm-up reset. The transcribed m8 numbers are assumed to be `peak_mb` (U11, checked by job step 0b). `pred_over_meas` uses `peak_mb` too.

The warm-up (trace + cuDNN autotune) is handled separately:
- `fit` reports `warmup_excess_gib` = max over points with both fields of (`warmup_peak_mb` − `peak_mb`) / 1024. Locally this is ≤ 0.21 GiB (narrow `both` N8: 4994 vs 4781 MB).
- Feasibility uses the steady peak. The autotune excess is **accepted, not budgeted**: a failed workspace allocation during autotune is not fatal (U12). TF skips that algorithm, which costs speed, not correctness, and the steady state then uses an algorithm that fits.
- The level-1 m9 records include `warmup_peak_mb`. The report states the measured level-1 excess, and any level-1 `ok` record whose warm-up peak exceeded the card is flagged in the report.

`Workload` dataclass (inputs):
- `level` (4/2/1), `task` (rigid/affine/deform), `n`;
- `model_config` (ModelConfig, a dict, or a preset name);
- `checkpointing`, `amp`, `integration_steps`;
- `lncc` (template and pair weights > 0), `p_pdice` (0.1), `n_classes` (19);
- `lean_sampler`, `lean_losses`, `split_group_conv`;
- `cascade` (`none` | `frozen4` | `chain42`), `val_pair_chunk`;
- `build_n` (default `min(n, 2)`), `tasks_run` (default `(task,)`; 4.2);
- `gpu` (`rtx8000` | `a6000` | `a100` | `gtx1080`), which selects the `c_0` family and the time model.

`predict(workload, coeffs, margin=0.0)` returns:
- `peak_gib`, `peak_margin_gib` (= peak × (1 + margin)), `resident_gib`;
- `phases` {F, Fw, Fb, I, D, E, Z, V: GiB} and `peak_phase`. Fw and Fb are 0 without the lean sampler. `predict(..., skip_grad_bound=False, fwd_transient_bound=False)` selects the conservative variants (2.1);
- `components` (name, phase set, bytes per image, class `net | rec | warp | other | transient`);
- `largest_tensor` {desc, shape, dtype, elements, bytes, direction `bwd | fwd`} and `limit_flags` (list of strings);
- `feasible_44`, `feasible_36` (both at the given margin), `val_peak_gib` with the chosen `val_pair_chunk`;
- `step_s` (estimate) and `step_s_range` (×0.5 … ×1.5).

### 4.2 Inventory construction

The inventory is built from the architecture with the rules of 2.1, not hard-coded per preset. For each stage of `GroupEncoder` / `DeformDecoder` / `LinearDecoder` it derives from the level:
- the resolution (pools at levels 2/1 as in `group_net`: `pool2` at level 1 only, `pool3` at levels 2 and 1);
- C_in, including skip concats; C_out; the group flag;
- the taped channels: 5·C_in, or 4·C_in + C_in/N with split; the head 4·C.

Skip gradients (`B_sg`, review v3 F9): for each decoder block the inventory records which encoder skip it consumes. In a D sub-phase evaluated at the backward of block b, the d(skips) of the decoder blocks finer than b are added (encoder compute dtype, at each skip's resolution); in E, and in the taped encoder backward of the linear tasks and of `none` / `integrator` / `both`, every d(skip) is added. The D phase value is the maximum over its sub-phases (one per decoder block b). `predict(..., skip_grad_bound=True)` instead adds all d(skips) at every D sub-phase; `recommend` prints the per-N table under both rules and selects with the timed rule.

Sampler calls are enumerated from `deform_loss`, the integrator (2T compositions; two blocks of T under `full`) and the cascade composition, with their C and P. v5: each call is a record `(name, C, share, images_grad, term, order)`, where `share` is 1 or P/N, `term` is the loss term the call belongs to, or `shared`, and `order` is its position in program order (2.1 Fw). The order follows `runtime/losses.deform_loss` at the code commit: template one-hot warp, then pair displacement and pair one-hot warp, then template image warp, then pair image warp. With the lean sampler each call contributes:
- its retained bytes to the phases of its mode (as in v4);
- its forward moment to Fw, at its call site, under the timed rule or `fwd_transient_bound`;
- its backward moment to Fb (loss calls) or I (integrator compositions, together with the recomputed composition's forward moment).

Each term lists the tensors created after its sampler call (warped output, Dice slices, LNCC moments), which are what Fw and Fb subtract. LNCC costs 284 B per call by default and 24 B with lean losses. Batch tensors are 148 B, plus 24 B at level 1 for the upsampled U, U⁻.

Each component carries the phase set of its mode (2.2) and a class:
- `net`: taped network tensors that are not recomputed;
- `rec`: the block recomputed in that phase (decoder in D, integrator in I, encoder in E);
- `warp`: sampler-held tensors, not recomputed;
- `other`: losses, batch and fields;
- `transient`: c_bwd units.

Resident = touched trainable parameters × 16 B (fp32 weight, gradient, two Adam slots) + 2 B × touched parameters (fp16 weight casts, AMP) + untouched trainable parameters × 4 B + frozen parameters × 4 B. The parameter counts come from `width_presets.count_params`, per part. "Touched" means the parts used by the tasks of the workload (U16):
- a single-task measurement record: deform touches the encoder and the deform decoder; rigid or affine touches the encoder and the linear decoder;
- a training workload (`tasks` = the task ratio, default all three) touches every part.

`Workload` gets the field `tasks_run` (default: the workload's own task). `recommend` predicts training with `tasks_run = (rigid, affine, deform)`, which is the larger resident term.

The build forward is not part of any phase (2.1). Its tensors enter only `limit_flags`, at `Workload.build_n` (default `min(n, 2)`; calibration records carry their own `build_n`).

Phase Z (frozen chain inference, `chain42`): batch tensors + the frozen level-2 fp32 forward at V2 (largest live set: group concat + its NCDHW copy + skips) + the upsampled U, U⁻ at V1. Phase V (validation): the eval forward of the trainable stage (largest live set without tape, AMP dtype) and the eager `_deform_errors` pair chunk. Per pair and V1 voxel that chunk costs one-hot gather 76 + moved 76 + sampler forward transient 3·(4 + 4C) + coordinates 12 B.

### 4.3 Coefficients and fitting

Prediction for a phase p, per image (pairs as P/N per image):

`S_p = c_enc[prec] · B_enc,p + c_net · B_net,p + c_rec · B_rec,p + c_warp · B_warp,p + B_other,p + B_loss,p(c_loss) + c_bwd · B_tr,p`

The terms:
- `B_enc,p`: encoder tensors in phase p, taped or recomputed. `prec` is fp32 or fp16, the compute dtype of the trainable stage.
- `B_net` / `B_rec`: decoder and head tensors only.
- `B_loss,p`: zero outside F. In F it is `max_term + c_loss · (sum_terms − max_term)` over the four loss terms of 2.1.

`peak = R + c_0[gpu family] + N · max_p S_p` (Z and V are evaluated as their own totals). B_tr,p = C_in,max of the largest block alive or recomputed in phase p × 4 B at its resolution (U13).

The coefficients are pre-registered, with initial value, bounds and the value expected from section 2.3:

| Coefficient | Meaning | Initial | Bounds | Expected (from 2.3) |
|---|---|---|---|---|
| `c_net` | taped decoder and head tensors | 1.0 | [0.6, 2.0] | ≈ 1.03 (level-2 `none` / `integrator` −3 … −5 %) |
| `c_enc32` | encoder tensors, fp32 compute (review F1) | 1.0 | [0.6, 1.6] | ≈ 1.10 (level-2 linear fp32 −7 / −8 % pre-fit) |
| `c_enc16` | encoder tensors, fp16 compute (AMP) | 1.0 | [0.6, 1.6] | ≈ 0.89 (level-2 linear AMP +7 … +12 % pre-fit) |
| `c_loss` | share of the non-largest loss terms alive at F (U15) | 1.0 | [0, 1] | between 0 (level-4 LNCC pairs) and 1 (level-2 `none`, which `c_net` can also absorb); the fit decides |
| `c_rec` | recomputed block in its backward phase (phase-aware, review F2) | 1.0 | [0.5, 2.0] | ≈ 1.0 (level-2 `both` within ±1 %, level-4 narrow `both` +4 … +9 % before `c_0`) |
| `c_warp` | default-sampler tensors (absorbs U4) | 1.0 | [0.6, 2.0] | ≈ 1.0–1.05 |
| `c_bwd` | backward transient (U13) | 2.0 | [0, 4.0] | ≈ 2.0 |
| `c_0[cluster]` | allocator / context constant, RTX 8000 / A6000 / A100 | 0.3 GiB | [0, 2] | ≈ 0.3 |
| `c_0[gtx1080]` | the same, local GTX 1080 under the 6000 MB logical limit | 0.1 GiB | [0, 1] | ≈ 0.05–0.1 (measured `resident_mb` 67–424 MB) |

Nine coefficients over 37 transcribed and local points plus the 8 LNCC points of `lncc-loss/memory.json`.

Status of each point group before and with the expected values (2.3):

| Point group | Pre-fit (all 1, c_bwd 2) | With the expected values |
|---|---|---|
| Level-2 deform, 21 points | −5 … +1 % | −4 … +3 % (the `c_enc` shifts) |
| Level-2 linear, 6 points | −8 … +12 % | −1 … +6 % |
| Level-4 A6000, 3 points | −3 % | about −3 % |
| GTX 1080 default and narrow, 7 points | +3 … +13 % | ≤ ±5 % once `c_0[gtx1080]` is lowered |
| LNCC on/off, 8 points | ≤ +4.4 % difference between on and off | `c_loss` absorbs the difference |
| Failed or OOM points | all predicted infeasible, including linear fp32 N12 via the build-forward limit flag (2.4) | same |

The pre-fit values alone put every point within ±13 %, so ±15 % is reachable without the fit. The fit only has to avoid trading one group against another, and the precision-split `c_enc` removes the one opposite-sign pair the review found. If either `c_enc` ends at a bound, the report states it as a model-form problem.

Fit: `scipy.optimize.least_squares` on the residuals (pred − meas)/meas over the `ok` points, with `loss="soft_l1"` and `f_scale=0.05`, plus one hinge residual per `oom` calibration point (review v3 F5): r = max(0, 0.98 · cap − pred) / cap, so the fit is pushed to keep every OOM point above 98 % of its card's cap (the 2 % covers the allocator headroom TF keeps below the logical limit). `failed_d2` points are infeasible through `limit_flags`, which does not depend on the coefficients, so they get no residual. There is no optional extra coefficient. v2's `c_net16` is dropped (2.3).

`fit` exits 1, printing the offending points, if either holds after convergence:
- any `ok` point outside ±15 %;
- any `oom` point predicted feasible (pred ≤ cap) or any `failed_d2` point without a limit flag. The marginal case the review names is GTX narrow `none` N6 (≈ 5.9–6.2 GB pre-fit against the 6000 MB cap).

The report states it. The tolerance is not loosened silently; the recommendation is then printed with the measured error band and flagged for Jay.

Calibration points (`models/memory_calibration.json`; each with `source`, `gpu`, `cap_gib`, `status`, `measure: "peak_mb"`, `lncc`, `cascade`):
- Level 2, RTX 8000, deform, cascade, LNCC (`lncc: true`):
  - fp32: none N2/3/4 (N6 `oom`); integrator N2/4/6; both N2/3/4/6;
  - AMP: none N2/4/6; integrator N2/4/6/8; both N2/4/6/8.
- Level 2 linear: fp32 N4/8 (N12 `failed_d2`); AMP N4/8/12/16. These carry `build_n = n`, the old tool behaviour. The build forward at that N is evaluated **only** by `limit_flags`. It is never added to the predicted `peak_mb`, which the tool measures after the post-build reset (2.1). The fp32 N12 record is predicted infeasible by that flag.
- Level 4, A6000, deform fp32 none N8/12/16 (`lncc: false`, milestone 6, Dice only; review F16).
- GTX 1080 (cap 6000 MB, `TF_CUDNN_USE_AUTOTUNE=0`, `lncc: false`):
  - default none N2 (N3, N4 `oom`); default both N2 (N4 `oom`);
  - narrow none N2/3/4 (N6, N8 `oom`); narrow both N4/8 (N16 `oom`).
- The LNCC points of `lncc-loss/memory.json` (`lncc: true`).
- The level-4 linear N = 32 point (26–34 GB, training-log range) is stored with `status: "range"` and checked only for containment.

Hold-out set (review F3; `status: "holdout"`, never used by `fit`): local GTX 1080 measurements made by the implementer with the Phase-1 tool (5.6). Configurations:
- level 4 narrow N 2/3/4, level 4 default N 2, and level 2 tiny N 2 (T = 7 if it fits, else T = 3, recorded);
- each in four knob combinations: `full`; `full` + lean sampler; `full` + lean sampler + lean losses; `full` + all three;
- each in fp32 and AMP.

That is up to 40 points (OOMs recorded as such). `tests/test_memory_model.py` and the report print every hold-out error.
- Gate: |error| ≤ 20 % on every `ok` hold-out point, and every hold-out OOM predicted infeasible.
- If a knob fails the gate, its inventory (2.1 / 3.3–3.5) is corrected and the correction is documented in the report. The gate is not widened.
- The margin used by `recommend` is m = max(0.15, max |hold-out error|). Because the gate fails above 0.20, m ≤ 0.20 whenever the gate has passed. v4's "m > 0.30 → exit 4" could never fire and is dropped (review v4 F3).
- Exit 4 is redefined as the gate bypass case: `recommend` re-evaluates the gate on the hold-out points of the calibration file it reads. It exits 4 and stops for Jay if any `ok` hold-out point has |error| > 0.20 or any hold-out OOM is predicted feasible, for example after a hand-edited calibration file or with coefficients fitted before a knob's inventory was corrected.
- Lean-sampler hold-out records also test the Fw / Fb terms (v5). Their errors enter m like any other.
- **Implementation deviation (FEAT-004, orchestrator decision; `report.md`).**
  - The gate is **one-sided**: it fails only on an under-prediction beyond −20 % or on a hold-out OOM predicted feasible.
  - Margin m = max(0.15, the worst under-prediction).
  - Reason: with the measurements made one configuration per process, 18 of the 40 local points are over-predicted by more than 20 %, which is conservative. These are small level-4 AMP and split points (+20 … +52 %). The worst under-prediction is −13.0 %, so m = 0.15.
  - The symmetric verdict (FAILED) stays in the report as an open model-form question.
  - Also from the hold-out: under `full` both integrator blocks are recomputed concurrently, so phase I holds two blocks. The inventory was corrected.
  - The tool now runs one configuration per process (`--isolate true`), because memory leaked across configurations in one process.

Accuracy report (R3.3):
- a table of every point: predicted, measured, error %;
- every OOM / failed point must be predicted infeasible (OOM: predicted peak > cap; `failed_d2`: a limit flag or predicted > cap);
- leave-one-mode-out: for each of the six level-2 groups (fp32/AMP × none/integrator/both), refit without it and print its errors;
- the hold-out table;
- `warmup_excess_gib`.

### 4.4 Level-1 grid and recommendation (R3.4, R3.6, R3.7)

`grid` covers the R3.4 grid in full and prints, per widths × checkpointing × AMP × T × knobs, the largest feasible deform N and its peak. Axes:
- widths {default, u075, u050, u0375, f0d1, fr50, fr25};
- checkpointing {none, both, full}, AMP {false, true}, T {6, 7};
- knob subsets of {lean sampler, lean losses, split} (8);
- deform N 2…8 (cascade `chain42`, LNCC 1/1); linear N 4…16 (rigid and affine, AMP {false, true});
- the frozen-chain cost per N (phase Z, frozen weights, persistent U/U⁻).

Recommendation rule (`recommend`). This changes the R3.6 rule (review F5, option b), because the R3.6 rule with an arbitrary N cap of 8 would choose the 8-channel `fr25` stem by construction. Jay's message 12 asks for "what size model will fit … at least batch size 4". `recommend` therefore maximises the model subject to a target N, and the per-N table keeps every alternative visible.

1. Candidates. Level 1 always uses AMP = true and checkpointing = `full` (Jay, message 12; `none` / `both` stay in `grid` for information). Every widths preset × knob subset × T × N 2…8, with T ∈ {7}, plus {6} only if `--t6-verdict <integration_steps.json>` has `t6_accuracy_ok: true`. Without the file, T = 6 is excluded and a warning is printed (fail-safe, review F6).
2. Feasible(config, N) iff all four hold with m from 4.3:
   - (a) predicted steady training peak × (1 + m) ≤ 44 GiB;
   - (b) phase Z × (1 + m) ≤ 44 GiB;
   - (c) validation peak × (1 + m) ≤ 44 GiB for some `val_pair_chunk` ∈ {16, 8, 4, 2, 1} (the largest that fits is chosen);
   - (d) no limit flag (bounds of 2.4), for the training step and for the build forward at `build_n = min(N, 2)`. Under that rule the build never binds, but the check stays in case `--build-n` is changed.
3. Per-N table, always printed. For every N in 2…8 it shows the widest feasible configuration, chosen by:
   - most trainable parameters;
   - then the T rule of step 4 (T = 7 preferred unless step 4 makes T = 6 the choice; judged against the T = 7 candidate with the same widths and knobs);
   - then fewer non-default knobs among lean sampler / lean losses / split;
   - then the lowest predicted step time.

   This is R3.6's order (wider, T = 7, fewer knobs; review v3 F7). v3 had swapped the T rule and the knob count; since the T rule only ever compares candidates with equal widths and knobs, the order changes no expected row.
   
   It prints its peak, peak with margin, parameters, knobs and A100 verdict.
4. T rule (R3.5 with the headroom clause, review F6). For equal widths and knobs, T = 6 is chosen over T = 7 only if `t6_accuracy_ok` and either:
   - T = 7 is infeasible at that N and T = 6 is feasible (T = 6 raises N); or
   - both are feasible, peak(T7) × (1 + m) > 0.9 × 44 GiB (less than 10 % headroom) and peak(T6) × (1 + m) ≤ 0.9 × 44 GiB (T = 6 restores ≥ 10 % headroom).
   
   Otherwise T = 7.
5. Selection: config* = the per-N row at N = `--n-target` (default 4, Jay's stated minimum). N* = the largest N at which config* stays feasible (N* ≥ `n_target`). `recommendation.json` stores **`level1.deform_n` = N\*** and `n_target` next to it (review v4 F4). In the expected case N* = `n_target` = 4. If no configuration is feasible at `--n-target`, `recommend` prints the R3.7 verdict (the smallest extra change that reaches it, chosen from the backlog in 3.7 by predicted saving), exits with code 3 and stops for Jay.
6. Printed and written to `scratch/m9/recommendation.json` (`--out`; review F10). Fields the retraining commands use:

```json
{"provisional": true, "margin": 0.15, "n_target": 4, "coefficients": "models/memory_coefficients.json",
 "widths": "f0d1", "filters": [16, 64, 128, 192, 256, 384], "decoder_widths": [128, 96, 64, 32, 8],
 "integration_steps": 7,
 "knobs": {"amp": true, "gradient_checkpointing": "full", "lean_sampler": true, "lean_losses": false,
           "split_group_conv": true},
 "level1": {"deform_n": 4, "n_target": 4, "linear_n": 16, "val_pair_chunk": 4, "deform_peak_gib": 33.5,
            "deform_peak_margin_gib": 38.5, "linear_peak_gib": 12.7,
            "a100": {"deform_n_margin": 3, "deform_n_no_margin": 4, "linear_n": 16}},
 "level2": {"recipe": "m8", "deform_n": 8, "linear_n": 16, "amp": true, "gradient_checkpointing": "both",
            "deform_peak_gib": 25.2, "largest_deform_n": {"both": 8, "full": 12}},
 "level4": {"recipe": "m7", "deform_n": 16, "linear_n": 32, "amp": false, "gradient_checkpointing": "none",
            "deform_peak_gib": 32.6},
 "step_s": {"rtx8000": {"l1_deform": [25, 40], "l1_linear": [5, 9]}, "a100": {"...": "..."}},
 "level1_long_run_h": {"rtx8000": [33, 60]}, "per_n": [{"n": 2, "widths": "default", "...": "..."}]}
```

   The `level2` / `level4` entries are the pinned retraining recipes of section 6.4, with their predicted peaks under config*. `recommend` flags them if they are infeasible. It also prints the largest feasible N per checkpointing mode at those levels, for information.
   
   `--print-env` prints the shell assignments the SLURM scripts take, for example `EXTRA_ARGS="--filters 16,64,128,192,256,384 --decoder-widths 128,96,64,32,8 ..." N=16 DEFORM_N=4`. No script parses the JSON.
   
   Also printed:
   - the level-1 deform peak at N* and the linear peak at the largest linear N ≤ 16 (≥ 8 targeted);
   - the A100 verdicts (≤ 36 GiB, with and without the margin) and the largest deform N on the A100;
   - step times per task at levels 4, 2, 1 on RTX 8000 and A100 (4.5);
   - the predicted wall time of a level-1 long run (8000 steps at 1:1:2 plus 16 validations), flagged if > 7 days (partition maximum) or > 72 h (needs a resume chain);
   - the line "PROVISIONAL until slurm/m9_memory.sh measures it" and the N that still fits at `--margin 0.15` next to the no-margin N.

Pre-fit expectation (2.5, m = 0.15). The per-N table:

| N | Widest feasible | Knobs | Peak / with margin (GiB) | Params |
|---|---|---|---|---|
| 2 | `default` | `full` only | 34.6 / 39.8 | 29.6 M |
| 3 | `default` | L + split (default + L is excluded: its D1 concat, 2.64·10⁹ elements, exceeds the backward bound) | 32.9 / 37.8 | 29.6 M |
| 4 | **`f0d1`** | L + split | 33.5 / 38.5 (Fw 32.0) | 29.5 M |
| 5 | none within the margin. Without it: `f0d1` LS 41.4 D. Every lean row has Fw ≥ 38.9 GiB at N 5, which × 1.15 > 44. | — | — | — |
| 6–8 | none (Fw ≥ 46.5 GiB at N 6) | — | — | — |

v5 change (2.5): the lean sampler's forward transient Fw removes the v4 rows N 5 `fr50` LS (35.4 → 39.6 / 45.5), N 6 `fr50` LLS (37.7 → 47.3) and N 7 `fr25` LLS (33.0 → 54.1). The N 2–4 rows are unchanged.

Under the conservative bounds (2.5):
- **Skip gradients** (`skip_grad_bound`): N 3 `default` LS 34.5 / 39.7; N 4 `f0d1` LS 34.9 / 40.1.
- **Forward transient** (`fwd_transient_bound`): N 3 `default` LS unchanged at 32.9 / 37.8 (its bound value is about 32.6); N 4 **`f0d1` LLS 35.0 / 40.3**, because `f0d1` LS at 41.0 / 47.2 fails the margin.
- N ≥ 5 is none under both.

The N = 4 widths are `f0d1` under every rule; only the forward bound adds lean losses. `recommend` prints the per-N table under the timed rule and under each bound, and selects with the timed rule.

This hand table applies the rule of steps 2–3 by hand, with the sum rule `c_loss = 1`. The table `recommend` generates from the fitted coefficients is authoritative.

With `--n-target 4` the expected recommendation is `f0d1`, AMP, `full`, lean sampler + split, T = 7, deform N* = 4, linear N = 16 (12.7 GiB). The A100 fits N = 4 only without the margin (33.5 ≤ 36; 41.0 under the forward bound), so level 1 trains on the 48 GB cards (6.4).

The decision left for Jay, stated in the report: `--n-target 4` keeps today's half-resolution widths and halves only f0 / D1. N ≥ 5 at level 1 needs either running without the margin (`f0d1` LS N 5 at 41.4 GiB predicted, checked by the m9 measurement) or the backlogged forward corner sequencing (3.7), which restores the v4 alternatives (`fr50` with lean losses at N 6).

### 4.5 Step-time model

Per task and GPU class: `t = a_gpu · F_conv + b_gpu · W_warp + t0_gpu`.
- F_conv counts 2·27·C_in·C_out·V FLOPs per conv: forward × 1, backward × 2, recompute + 1 forward for the recomputed parts. fp32 convs count × 2.2, the measured linear AMP speed-up.
- W_warp counts sampler voxel-channels: forward + backward × 3, recompute + 1; the lean sampler's backward × 4 for the per-corner gathers.

Data (corrected in v4, review v3 F2). What the training logs hold, verified in `runtime/run.py` `fit`: the per-step record `{step, task, data_ms, compute_ms, ...}` goes only into the in-memory `Trainer.history`. `metrics.jsonl` receives window means (`acc.result()`, every `--log-every` = 25 steps in the m7 / m8 scripts). The timing keys there are `time/compute_ms` (mean over all tasks of the window) for every run, and `train_<task>/compute_ms` (per task) only when `self.cascade is not None`. Neither the per-step task nor the per-window task counts are logged. So:
- `results/m8_level2_gradnorm` (cascade, A100, level 2) has per-task window means: `train_rigid/compute_ms`, `train_affine/compute_ms`, `train_deform/compute_ms`.
- `results/m7_three_task_lncc` (no cascade, A100, level 4) has only the task-mixed `time/compute_ms`; per-task times cannot be identified from it.

New TF-free `scripts/step_times_from_metrics.py --runs <run dirs> --min-step 1000 [--extra-keys k1,k2] [--args-keys a1,a2] --out <json>`:
- per run it reads `metrics.jsonl` and, over records with `step ≥ --min-step`, reports for `time/compute_ms` and for each `train_<task>/compute_ms` present: the median and IQR of the window means, and the number of windows. Validation time is not inside `compute_ms`.
- `per_task: true` if at least one `train_<task>/compute_ms` key exists, else `false` (only the mixed mean is reported).
- `--extra-keys`: each named key is matched with or without a `train_<task>/` prefix; reported as median and **max** of the window means (3.6); missing keys are `null`.
- `--args-keys`: copies the named entries of the run's `config.json` `args` (missing → `null`; 6.4 step 1 uses this for the m5 G schedule).
- A missing run dir, `metrics.jsonl` or `config.json` marks that part `missing`; unparsable lines are counted and skipped; exit 0.

`m9_memory.sh` step 0c runs it on `results/m7_three_task_lncc`, `results/m8_level2_gradnorm` and `results/m5_mixed_long_curr_dG_20261019-205526` → `scratch/m9/step_times.json`.

Per-task time sources per level and GPU:
- Level 2, A100: `m8_level2_gradnorm` `train_<task>/compute_ms` medians (expected about 0.73 s linear, 4.9 s deform). Level 2, RTX 8000: `step_ms` of the m8 memory JSON (deform AMP both N8 7.0 s, linear AMP N16 1.16 s). Their ratios define r_deform ≈ 0.70 and r_linear ≈ 0.63 (A100 / RTX 8000, U17).
- Level 4, RTX 8000: `step_ms` of m9 job step 6 (deform N 16, default and centre widths; default N 24 / 32 are expected to OOM and are recorded as such) and step 6b (rigid N 16 / 32). Affine uses the rigid time (same network and inventory; ±10 % assumption, stated in the printout). Level 4, A100: those times × r_task.
- Level-4 consistency check (A100): predicted mixed mean (t_rigid + t_affine + 2 · t_deform) / 4 at the m7 recipe (default widths, N 32 / 16, fp32, LNCC) against the median `time/compute_ms` of `m7_three_task_lncc` (expected ≈ 1.65 s). Within ±20 %: the r_task scaling is kept. Outside: the level-4 A100 times are replaced by the RTX 8000 times × one scalar s = measured mean / predicted RTX-8000 mixed mean, and the printout says so.
- Level 1, RTX 8000: the m9 job's level-1 `step_ms`; A100: × r_task (level 1 is AMP like level 2).

Before the m9 job has run, the level-4 and level-1 entries are model estimates only (the coefficients a, b, t0 below) and labelled `source: "model"`.

Identification is weak and is stated in the printout. Per GPU, `a` comes from the linear points, and `b` is fixed exactly by a single deform point, so there is no residual to estimate the error. Every estimate carries ±50 % (U9).

Pre-fit estimate for `f0d1` at level 1:
- RTX 8000: deform N4 25–40 s, linear N16 5–9 s;
- A100: 18–28 s and 3.5–6.5 s.

A 1:1:2 long run of 8000 steps on the RTX 8000 is 4000 × 32 s + 4000 × 7 s ≈ 43 h, plus 16 validations (≈ 0.8 h, 6.3) and the BN recalibration (≈ 10 min): ≈ 45 h (range 33–60 h). That is under 72 h and well under the 7-day partition maximum.

### 4.6 Tests (no TF)

`tests/test_memory_model.py` checks:
- every `ok` calibration point is within ±15 %, and every OOM / failed point is infeasible;
- the leave-one-mode-out errors are printed;
- hold-out points are within ±20 % and their OOMs infeasible, and the table is printed;
- the presets table is reproduced by `uniform`, and `count_params` is stable;
- the level-1 grid, the per-N table and the recommendation are printed (deform N ≥ 4 at ≤ 44 GiB with the margin, including frozen stages and validation, or the R3.7 verdict);
- `recommendation.json` contains every field listed in 4.4;
- T = 6 is excluded without `--t6-verdict` and with `t6_accuracy_ok: false`, and selected under each branch of step 4 on synthetic predictions;
- `predict` of the m8 recipe (level 2, default, AMP, both, N 8) is within 15 % of 27.8 GB;
- `limit_flags` flags `f0d1` without split at N = 4 (D1 concat 2.64·10⁹ > 2³¹ − 1, backward) and passes it with split;
- build forward (review F2): `f0d1` + split at N 6 with the default `build_n = min(6, 2)` has no build flag. The same workload with `build_n = 6` is flagged (unsplit build D1 concat 3.96·10⁹ fp16 > 3.52·10⁹). The m8 linear fp32 N12 record (`build_n = 12`) is flagged by its build forward (D2 concat 2.64·10⁹ fp32);
- the build forward never changes `peak_gib`: two workloads that differ only in `build_n` have the same predicted peak;
- the level-2 linear table of 2.3 is reproduced: the pre-fit values within 0.05 GiB of the printed numbers, and the fitted errors within ±15 %;
- pre-fit pins (v5, review v4 F8). All coefficients are at their initial values, U4 is as stated (int32 indices host-resident), and the default sampler is used. The six level-2 deform rows of 2.3 must be reproduced within **0.1 GiB** each. The `both` rows include `B_sg`, i.e. +64 B (AMP) / +128 B (fp32) per V2 voxel per image:
  - fp32 none N2/3/4 = 18.5 / 26.7 / 35.0;
  - fp32 integrator N2/4/6 = 15.1 / 29.3 / 43.6;
  - fp32 both N2/3/4/6 = 12.5 / 18.3 / 24.1 / 35.8;
  - AMP none N2/4/6 = 13.8 / 25.6 / 39.6;
  - AMP integrator N2/4/6/8 = 10.4 / 19.9 / 29.4 / 38.9;
  - AMP both N2/4/6/8 = 7.7 / 14.5 / 21.3 / 28.2.

  Level-1 pins: `f0d1` LS N4 (AMP, `full`, cascade `chain42`, LNCC 1/1, T 7) has `peak_gib` **33.5 ± 0.1** with `peak_phase` D and phases Fw 32.0 ± 0.2 and Fb 25.5 ± 0.2. The default-sampler level-2 pins do not depend on any v5 term;
- the Fw / Fb rules (v5) on the same `f0d1` LS N4 workload:
  - the binding Fw call is the template one-hot warp, and its timed live set is the shared tensors, so Fw − resident equals N · (shared + 936 B) to within 1 B per V1 voxel per image;
  - with `fwd_transient_bound=True`, Fw − F = N · 554 B (± 1 B) per V1 voxel per image, and with lean losses N · 698 B;
  - Fb = F − N · 54 B (default Dice) and F + N · 90 B (lean losses);
  - phase I with the lean sampler uses the 172 B backward moment and the 360 B forward moment of the recomputed composition;
  - without the lean sampler, Fw = Fb = 0 and every v4 prediction is unchanged;
  - `fwd_transient_bound=True` gives `f0d1` LS N4 41.0 ± 0.2 GiB and `f0d1` LLS N4 35.0 ± 0.3 GiB;
- the per-N table under the timed rule has no feasible row with the margin at N ≥ 5, and its N 4 row is `f0d1` LS (pre-fit coefficients);
- the LNCC on/off pairs: the predicted on − off difference is ≥ 0 and ≤ 5 % of the peak for every `c_loss` in [0, 1];
- resident: a linear-only record's resident excludes the deform decoder's slots, and a training workload includes them;
- `fit` with one OOM point deliberately predicted feasible (a fake calibration file) exits 1 and names the point; the hinge residual is zero for OOM points already above 0.98 · cap;
- the skip-gradient term: at level 2 `both` the D phase includes d(stem skip) only; at level 1 `full` the D1 sub-phase has `B_sg` = 0, E includes all skip gradients; `skip_grad_bound=True` gives the 2.5 conservative numbers (`f0d1` LS N4 34.9 GiB within 0.1);
- the step-time chain: the m7 consistency check keeps r_task within ±20 % and switches to the scalar s outside it (synthetic inputs).

The tests run in `bgr_preproc38` too; they are added to the TF-free block of `run_tests_wsl.sh`. `tests/test_step_times.py` runs `step_times_from_metrics.py` on two small fake run dirs whose `metrics.jsonl` hold 25-step **window-mean** records, as the Trainer writes them (review v3 F2): one cascade-style (`time/compute_ms`, `train_rigid/compute_ms`, `train_deform/compute_ms`, `train_deform/diag_u_res_max_vox`) and one non-cascade (`time/compute_ms` only). It checks `per_task` true / false, the medians / IQR, the max of `diag_u_res_max_vox` window means, `--args-keys` from a fake `config.json`, and a missing run marked `missing`.

## 5. Level-1 measurement tooling (R4)

### 5.1 `tests/gpu/measure_level_memory.py`

Arguments:
- `--level` choices (4, 2, 1).
- `--widths`: a comma list of `narrow`, `tiny`, any `models.width_presets` name, or `custom`. With `custom`, `--filters` and `--decoder-widths` are required, and the widths key is `custom:f=16-32-...;d=...`.
- `--gradient-checkpointing` accepts `full`. `--lean-sampler`, `--lean-losses` and `--split-group-conv` take comma lists of bools (grid axes, like `--amp`). `--integration-steps` takes a comma list of ints (0 = the ModelConfig default, resolved before the record is written) and is a grid axis.
- `--build-n` (default 2): **every** configuration (linear and deform, trainable and frozen models) builds with `build_model(min(n, build_n))` (2.4, review F2 of v2). `--build-n < 1` fails with `ap.error("--build-n must be >= 1")` (review F18 of v1). So a build failure can no longer be recorded as a training-configuration `limit` / `oom` and skip the larger N. A level-1 build at N = 2 is far below every bound.
- `--val-pair-chunk` (default 16, was hard-coded) is used by the validation pass; values < 1 fail with `ap.error`.
- `--cascade true` at `--level 1` uses `FrozenChainStandIn` (5.2); at level 2 the existing `FrozenStage`; at level 4 it is an error as today. `--cascade false` is valid at every level (R4.2 literal: the level-1 trainable step alone).

Frozen-stage peak (review F7): for cascade configurations with training status `ok`, after the existing `frozen_ms` timing:
1. `reset_memory_stats("GPU:0")`;
2. `ffn(x).numpy()`;
3. `frozen_peak_mb = _peak_mb()`.

The trainer, model, optimizer slots and batch are still alive, so the value includes the resident memory, as it would during training. A `ResourceExhaustedError` there sets `frozen_status: "oom"` and leaves the training status unchanged. Fields: `frozen_peak_mb`, `frozen_status` (`ok | oom | error | null`).

New result fields:
- `lean_sampler`, `lean_losses`, `split_group_conv`, `build_n`, `val_pair_chunk`;
- `cascade_kind` (`none` | `frozen4` | `standin_chain_4_2`);
- `frozen_peak_mb`, `frozen_status`;
- `predicted_peak_mb`, `predicted_phase`, `pred_over_meas` (= `predicted_peak_mb / peak_mb`), `largest_tensor_elems`, `largest_tensor_desc`. These come from `models.memory_model.predict` with the stored coefficients and margin 0. They are `null`, with a stderr warning, if the coefficients file is missing.

Status `limit` (review F14):
- The classification happens in `main()`'s per-configuration `except` chain, which is where a `build_model` error arrives, and in `measure()`'s inner `try`.
- An exception is `limit` iff its type is one of `tf.errors.InvalidArgumentError`, `tf.errors.InternalError`, `tf.errors.UnknownError` or `ValueError`, and its message contains one of `"2^31"`, `"2**31"`, `"INT_MAX"`, `"2147483647"`, `"num_elements"`. Everything else not OOM is `error`.
- After `oom` or `limit`, the larger N of that configuration are `skipped`.

Configuration key (review F8):
- `KEY_FIELDS = ("level", "task", "n", "widths", "cascade", "gradient_checkpointing", "amp", "lambda_ncc_tmpl", "lambda_ncc_pair", "integration_steps", "lean_sampler", "lean_losses", "split_group_conv", "val_pair_chunk", "cascade_kind")`.
- `build_n`, `frozen_*`, `predicted_*`, `pred_over_meas` and `largest_*` are **not** keys.
- `config_key(r)` reads missing fields of old records with these defaults: `lean_sampler`, `lean_losses`, `split_group_conv` = `False`; `val_pair_chunk` = `16`; `cascade_kind` = `"frozen4" if r.get("cascade") else "none"`.
- `integration_steps` (review v3 F6): in the tool, `--integration-steps 0` means "the ModelConfig default" (tool-only convention; `main.py`'s 0 means T = 0, which the tool cannot measure, documented in its help). `config_key` normalises a stored 0 to the resolved T, `model_config(r["widths"], 0).integration_steps` (7 for every current preset, including `narrow` / `tiny`, which set no T). New records store the resolved T, never 0.
- New records always write all fields. A re-measured m8 configuration therefore gets the same key as its old record and replaces it.
- `tests/test_measure_memory.py::test_merge_old_record` merges an m8-format record (no new fields, `cascade: true`, `integration_steps: 0`) with a re-measured one of the same configuration measured with `--integration-steps 7`, and expects exactly one record, the new one, with `integration_steps: 7`. A second case merges a record with T = 6 and keeps both.

Incremental, atomic output (v5, review v4 F7). Today `main()` merges and writes `--out` once, after the whole grid. v5 instead merges and writes it **after every configuration**, whatever its status (`ok`, `oom`, `limit`, `error`, `skipped`). The merge semantics are unchanged: read the existing file, replace records with the same `config_key`, append new ones, keep the order. The write goes to `<out>.tmp.<pid>` in the same directory, with flush and `os.fsync`, followed by `os.replace(tmp, out)`. A time-limit kill or a crash therefore loses at most the configuration being measured, and the file is never half-written. A failing write (for example a full disk, `OSError`) is fatal for the invocation: it propagates with its traceback and a non-zero exit, which `m9_memory.sh` records as that step's exit code. The temp file is removed in `finally`. New test `tests/test_measure_memory.py::test_incremental_write`:
- patch `measure` with a fake that returns records for two configurations, and wrap the writer to snapshot the file after each write;
- after the first write the file holds record 1, and after the second it holds records 1 and 2;
- no `*.tmp.*` file remains;
- a pre-existing record with the same key as record 2 is replaced, not duplicated.

The output and existing level-4 / level-2 invocations are otherwise unchanged, and the existing `tests/test_measure_memory.py` keeps passing.

### 5.2 Frozen-chain stand-in (R4.2)

The 3-level cascade is implemented after Phase 1 (R5.1). For measurement the stand-in lives in the new module `tests/gpu/frozen_chain_standin.py`, not in `models/` (review F4). It contains the class `FrozenChainStandIn`, the factory `build_standin(model_config, build_n=2)` and the context manager `patched_prev_level()`. `measure_level_memory.py` imports them only when `--level 1 --cascade true`.

```python
class FrozenChainStandIn:
    level = 2                       # the trainable level-1 stage's previous level
    # stage4: models.cascade.FrozenStage.from_model(random level-4 model of cfg)
    # model2: float32 GroupRegLevelModel(2, cfg, 'none') built under cascade._float32_policy()
    def deform(self, images_l1):    # -> dict U, U_inv, V at level 1 (stop_gradient)
        x2 = pool_to_coarser(images_l1)
        p4 = self.stage4.deform(x2)                     # level-4 fields upsampled to level 2
        out = self.model2(prewarp_deform(x2, p4["U"]), task="deform", training=False)
        u2, u2i = diffeo.compose_cascade(p4["U"], p4["U_inv"], out["u"], out["u_inv"])
        up = diffeo.upsample_displacement
        return {"U": stop(up(u2)), "U_inv": stop(up(u2i)), "V": stop(up(out["v"]))}
    def linear(self, images_l1, task):   # -> A_prev [N, 4, 4] float32 (stop_gradient)
        x2 = pool_to_coarser(images_l1)
        A4 = self.stage4.linear(x2, task)
        out = self.model2(prewarp_linear(x2, A4, 2.0), task=task, training=False)
        return stop(compose_linear(out["A_raw"], A4, self.model2.centering))
```

The stand-in is passed to `Trainer(cascade=standin)` inside `with patched_prev_level():`. The context manager sets `models.cascade.PREV_LEVEL = {2: 4, 1: 2}` and restores the original dict in `finally`. `FROZEN_LEVEL` is bound at import, so `FrozenStage` is unaffected (verified by the review). The level-1 trainable step therefore measures the real cascade input path (pre-warp, in-tape `compose_cascade`, persistent U/U⁻) plus the stand-in frozen inference inside the same step function.

Other fields:
- `frozen_ms` times `standin.deform` / `.linear` alone, and `frozen_peak_mb` measures it (5.1).
- `frozen_params` = both frozen models' parameter counts.
- Weights are random (memory and time only).
- The `--cascade false` level-1 points (job step 2b) measure the same trainable step without the stand-in. The difference bounds the frozen chain's share, and `frozen_peak_mb` calibrates phase Z.

### 5.3 `tests/gpu/probe_tensor_limit.py`

Modes `graph | run | m8-errors` (2.4), with the backward probes in `run`.

Arguments:
- `--level`, `--n`, `--widths`, `--task`, `--amp` (graph);
- `--gpu-memory-limit-mb` (run, local), `--sizes` (run, default as in 2.4);
- `--m8-json` (m8-errors), `--out`.

Exit 0 unless it crashes. `run` without a GPU exits 2.

TF32 (review F7): `graph` and `run` import TF, and their first statement after the imports is `models.precision.configure_tf_precision(tf32=False, memory_growth=...)`. `memory_growth` is True unless `--gpu-memory-limit-mb` is given. That matches `measure_level_memory.py` and `tests/tf_setup.py`. `m8-errors` does not import TF. The same rule holds for every new GPU-capable entry point of Phase 1: `tests/gpu/frozen_chain_standin.py` is only imported by the tool, which configures precision first, and the new TF tests import `tests.tf_setup` first.

Covered by `tests/test_probe_tensor_limit.py`:
- graph mode on the level-2 N = 12 default config: the D2 concat is listed first, > 2³¹ − 1;
- run mode with a tiny `--sizes` list on the local GPU: every op and direction has a status, and the summary has `verified_max_elements` with `fwd` / `bwd`;
- m8-errors mode on a small fake JSON.

### 5.4 `verify_dali_pipeline.py --level 1`

`--level` choices become (4, 2, 1); `_set_level` already rebinds LEVEL / SHAPE / SPACING. The level-1 template mask comes from the export's `template/`. This is unverified until the job runs, and a failing check does not stop the memory steps. `tests/test_verify_dali_level.py` extends to level 1 for the pure-Python parts.

### 5.5 `slurm/m9_memory.sh`

```bash
#!/bin/bash
#SBATCH --job-name=bgr-m9-mem
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Milestone 9 phase 1, memory job (design .agents/tasks/milestone-9/design-m9-phase1.md section 5.5).
# Untrained weights; synthetic groups (data needed only for step 1, logs only for step 0c).
#   0a. probe_tensor_limit.py --mode run (fp32/fp16 single ops at 2^31 +- 2^24 elements, forward AND
#       backward via GradientTape) -> tensor_limit.json
#   0b. probe_tensor_limit.py --mode m8-errors: failed records + peak_mb / warmup_peak_mb of
#       scratch/m8/memory_level2.json
#   0c. step_times_from_metrics.py on results/m7_three_task_lncc, results/m8_level2_gradnorm and
#       the m5 run -> step_times.json (window-mean compute_ms: per task for the m8 cascade run,
#       task-mixed time/compute_ms for m7; median / max of the diag_u_res_max_vox window means for
#       the T = 6 magnitude check, design 3.6; the m5 G-schedule args for retraining step 1, 6.4)
#   1.  verify_dali_pipeline.py --level 1, checks b,c,d,e,l (--n-throughput 16) -> verify_level1.json
#   2.  level 1 deform, cascade (frozen-chain stand-in), WIDTHS x T 6,7 x N 2,3,4,5,6,8, ckpt full,
#       AMP, $KNOBS, validation pass (pair chunk 4): <= 36 configs (larger N skipped after an OOM)
#   2b. level 1 deform, --cascade false, centre widths, N 2,4, T 7, $KNOBS: 2 configs
#   2c. level 1 deform, cascade, fr50 + lean losses (the N 5-6 alternative), N 5,6,7, T 7: 3 configs
#   3.  level 1 deform ablations at the centre widths, N 4, T 7: lean sampler off / split off /
#       lean losses on / checkpointing both: 4 configs
#   4.  level 1 rigid + affine, cascade, centre widths, AMP, N 4,8,12,16: 8 configs
#   5.  level 2 deform (frozen level-4 cascade), centre widths, AMP: the pinned m8 retraining recipe
#       (ckpt both, all knobs off, N 8) + ckpt full with $KNOBS (N 8,12,16): 4 configs
#   6.  level 4 deform, default + centre widths, fp32, none, N 16,24,32: <= 6 configs (level-4
#       per-task step times for the m7 consistency check, design 4.5)
#   6b. level 4 rigid, default + centre widths, fp32, N 16,32: 4 configs (replaces the unverified
#       "26-34 GB" linear N 32 range, design U14)
# Level 1 is measured with AMP only (selection rule, design 4.4 step 1); fp32 level-1 rows are
# model-only (scripts/memory_model.py grid).
# Every configuration builds at min(N, 2) (--build-n default); the build never sets a peak.
# All memory results merge into $OUT_DIR/memory_m9.json. Every call runs; the exit code is the first
# non-zero one (the tools record OOM / limit results and return 0 unless they crash).
# Time: level-1 deform config ~6 min (trace + cuDNN autotune ~3 min, 1 warm-up + 3 measured steps at
# ~25-50 s, validation pass ~1 min) x 45 (worst case, no OOM skips) = 4.5 h; linear 8 x 4 min = 0.5 h;
# level 2: 4 x 3 min; level 4: 10 x 1.5 min; probe, DALI, step times ~0.5 h -> ~6 h;
# --time 10:00:00 (1.7x margin). Peak memory is the point of the job; the 48 GB RTX 8000 is the
# default. A100 variant (40 GB; the level-1 N >= 4 configs are expected to OOM or exceed 36 GiB there,
# which is recorded): add -p dgx-a100 (same CPUs and memory).
# Memory: --mem 64G (host: synthetic level-1 groups ~0.5 GB, DALI buffers); up to ~128G is a fair
# share on rtx8000 / dgx-a100 if host OOMs appear.
# WIDTHS = recommendation, one step wider, one step narrower; KNOBS = the recommendation's knobs.
# `python scripts/memory_model.py recommend --print-env` prints both lines; replace the defaults below
# if the calibrated model picks another configuration.
#
# DO NOT submit without Jay's approval (cluster-access rules). Account qtim only.
# Before sbatch, the log folder must exist (SLURM does not create it):
#   mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
# Submit (RTX 8000 / A6000 48 GB):
#   sbatch --job-name=m9mem --time=10:00:00 --mem=64G slurm/m9_memory.sh
set -euo pipefail

PROJ="${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}"
REPO="${REPO:-$PROJ/brain-group-rad}"
DATA="${DATA:-$PROJ/data/npy_v2}"
SIF="${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}"
JOB_ID="${SLURM_JOB_ID:-manual}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"
OUT_DIR="$PROJ/scratch/m9"
WIDTHS="${WIDTHS:-f0d1,default,fr50}"
CENTER="${WIDTHS%%,*}"
KNOBS="${KNOBS:---lean-sampler true --lean-losses false --split-group-conv true}"

SCRATCH="$PROJ/scratch"
mkdir -p "$OUT_DIR/dali" "$SCRATCH/keras" "$SCRATCH/cuda_cache"
export TMPDIR="$SCRATCH_ROOT/$JOB_ID"
mkdir -p "$TMPDIR"
export KERAS_HOME="$SCRATCH/keras"
export CUDA_CACHE_PATH="$SCRATCH/cuda_cache"
export PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_TMPDIR="$TMPDIR"
export SINGULARITYENV_KERAS_HOME="$KERAS_HOME"
export SINGULARITYENV_CUDA_CACHE_PATH="$CUDA_CACHE_PATH"
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1

cd "$REPO"
echo "host $(hostname) | job $JOB_ID | repo $(git rev-parse --short HEAD 2>/dev/null || echo '?') | $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
echo "data $DATA | out $OUT_DIR | widths $WIDTHS (centre $CENTER) | knobs $KNOBS"

RUN=(singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF")
rc=0
rc_x=0   # 0a: per-tensor limit probe (forward + backward)
"${RUN[@]}" python tests/gpu/probe_tensor_limit.py --mode run --out "$OUT_DIR/tensor_limit.json" </dev/null || rc_x=$?
echo "probe_tensor_limit (run) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 0b: the m8 failure texts and peak columns (read only)
"${RUN[@]}" python tests/gpu/probe_tensor_limit.py --mode m8-errors \
    --m8-json "$PROJ/scratch/m8/memory_level2.json" </dev/null || rc_x=$?
echo "probe_tensor_limit (m8-errors) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 0c: per-task step times from the m7 / m8 training logs (read only)
"${RUN[@]}" python scripts/step_times_from_metrics.py --min-step 1000 --extra-keys diag_u_res_max_vox \
    --args-keys global_transform,global_start_step,global_ramp_steps,global_rot_max_deg,global_trans_max_mm,curriculum,curriculum_ramp_steps,task_ratio,group_size,steps,lr,warmup_steps \
    --runs "$PROJ/results/m7_three_task_lncc" "$PROJ/results/m8_level2_gradnorm" \
           "$PROJ/results/m5_mixed_long_curr_dG_20261019-205526" \
    --out "$OUT_DIR/step_times.json" </dev/null || rc_x=$?
echo "step_times_from_metrics exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 1: level-1 DALI checks
"${RUN[@]}" python tests/gpu/verify_dali_pipeline.py --data "$DATA" --level 1 --checks b,c,d,e,l \
    --num-threads "${SLURM_CPUS_PER_TASK:-3}" --n-throughput 16 \
    --scratch "$OUT_DIR/dali" --out "$OUT_DIR/verify_level1.json" </dev/null || rc_x=$?
echo "verify_dali_pipeline (level 1) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi

M=(python tests/gpu/measure_level_memory.py --lambda-ncc-template 1 --lambda-ncc-pair 1
   --out "$OUT_DIR/memory_m9.json")
L1=(--level 1 --amp true --warmup-steps 1 --steps 3)
rc_x=0   # 2: level-1 deform grid (<= 36 configs, validation pass each)
# shellcheck disable=SC2086
"${RUN[@]}" "${M[@]}" "${L1[@]}" --cascade true --tasks deform --widths "$WIDTHS" --integration-steps 6,7 \
    --n 2,3,4,5,6,8 --gradient-checkpointing full $KNOBS --val-pass true --val-pair-chunk 4 </dev/null || rc_x=$?
echo "measure_level_memory (level 1 deform grid) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 2b: level-1 trainable step without the frozen chain (2 configs)
# shellcheck disable=SC2086
"${RUN[@]}" "${M[@]}" "${L1[@]}" --cascade false --tasks deform --widths "$CENTER" --integration-steps 7 \
    --n 2,4 --gradient-checkpointing full $KNOBS --val-pass false </dev/null || rc_x=$?
echo "measure_level_memory (level 1 no cascade) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 2c: fr50 with lean losses (3 configs)
"${RUN[@]}" "${M[@]}" "${L1[@]}" --cascade true --tasks deform --widths fr50 --integration-steps 7 \
    --n 5,6,7 --gradient-checkpointing full --lean-sampler true --lean-losses true --split-group-conv true \
    --val-pass true --val-pair-chunk 4 </dev/null || rc_x=$?
echo "measure_level_memory (level 1 fr50 lean losses) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 3: level-1 ablations at the centre widths (4 configs)
for ab in "--lean-sampler false --lean-losses false --split-group-conv true --gradient-checkpointing full" \
          "--lean-sampler true --lean-losses false --split-group-conv false --gradient-checkpointing full" \
          "--lean-sampler true --lean-losses true --split-group-conv true --gradient-checkpointing full" \
          "--lean-sampler true --lean-losses false --split-group-conv true --gradient-checkpointing both"; do
    # shellcheck disable=SC2086
    "${RUN[@]}" "${M[@]}" "${L1[@]}" --cascade true --tasks deform --widths "$CENTER" --integration-steps 7 \
        --n 4 $ab --val-pass false </dev/null || rc_x=$?
done
echo "measure_level_memory (level 1 ablations) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 4: level-1 linear (8 configs)
"${RUN[@]}" "${M[@]}" "${L1[@]}" --cascade true --tasks rigid,affine --widths "$CENTER" --n 4,8,12,16 \
    --split-group-conv true </dev/null || rc_x=$?
echo "measure_level_memory (level 1 linear) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 5: level-2 deform at the centre widths (4 configs)
# 5a: the pinned level-2 retraining recipe (m8: both, no knobs) at the reduced widths, N 8
"${RUN[@]}" "${M[@]}" --level 2 --cascade true --amp true --warmup-steps 2 --steps 5 --tasks deform \
    --widths "$CENTER" --n 8 --gradient-checkpointing both --lean-sampler false --lean-losses false \
    --split-group-conv false --val-pass true </dev/null || rc_x=$?
# 5b: full with the recommendation's knobs, N 8,12,16
# shellcheck disable=SC2086
"${RUN[@]}" "${M[@]}" --level 2 --cascade true --amp true --warmup-steps 2 --steps 5 --tasks deform \
    --widths "$CENTER" --n 8,12,16 --gradient-checkpointing full $KNOBS --val-pass true </dev/null || rc_x=$?
echo "measure_level_memory (level 2 deform) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 6: level-4 deform, default and centre widths (<= 6 configs; default N 24 OOM -> 32 skipped)
"${RUN[@]}" "${M[@]}" --level 4 --cascade false --amp false --warmup-steps 2 --steps 5 --tasks deform \
    --widths "default,$CENTER" --n 16,24,32 --gradient-checkpointing none </dev/null || rc_x=$?
echo "measure_level_memory (level 4 deform) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 6b: level-4 linear, default and centre widths (4 configs)
"${RUN[@]}" "${M[@]}" --level 4 --cascade false --amp false --warmup-steps 2 --steps 5 --tasks rigid \
    --widths "default,$CENTER" --n 16,32 </dev/null || rc_x=$?
echo "measure_level_memory (level 4 linear) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi

echo "finished m9_memory exit code $rc $(date -Is)"
exit "$rc"
```

Test coverage:
- `tests/run_tests_wsl.sh` adds `slurm/m9_memory.sh` to its `bash -n` loop.
- `tests/test_slurm_scripts.py` gets `test_m9_memory_script`. It runs `bash -n` and asserts `-A qtim`, no `entre`, `--gpus=1`, `-N 1`, `--ntasks-per-node=1`, `--cpus-per-task=3`, `--mem=64G`, `-p rtx8000`, the two log paths, the sif path, `-p dgx-a100` in the header, the exact sbatch line, and `</dev/null` on every container call (count of `"${RUN[@]}"` lines = count of `</dev/null || rc_x=$?`).
- The same test does a stub run: stub `singularity` / `nvidia-smi` / `git` on PATH, `PROJ` and `SCRATCH_ROOT` pointing into a temp dir. It checks that the calls are made in order with the expected arguments, and that the exit code propagates. The argument checks include:
- `--cascade false` in step 2b;
- `$KNOBS` expanded into separate arguments;
- step 5 as two calls (v5, review v4 F6): 5a with `--gradient-checkpointing both --lean-sampler false --lean-losses false --split-group-conv false --n 8`, and 5b with `--gradient-checkpointing full` followed by the `$KNOBS` arguments and `--n 8,12,16`.

Step 5a measures the level-2 retraining recipe of 6.4 step 3 at the reduced widths. Its peak was so far only bounded by the default-width 27.8 GB. 5b uses `$KNOBS`, so lean losses follow the recommendation (off by default) instead of being forced on.

### 5.6 Local runs (R4.4) and the hold-out measurements

`tests/test_measure_memory.py` gains `test_tiny_level1`: `--level 1 --tasks deform,rigid --n 2 --widths tiny --cascade true --gradient-checkpointing full --lean-sampler true --lean-losses true --integration-steps 3 --amp true --steps 1 --warmup-steps 1 --val-pass true --val-pair-chunk 1 --gpu-memory-limit-mb 6000`. Checks:
- status `ok` or `oom` (both accepted), no crash;
- the JSON has `predicted_peak_mb`, `largest_tensor_elems`, `cascade_kind: "standin_chain_4_2"` and `frozen_peak_mb` (or `frozen_status`);
- a second invocation merges.

The synthetic level-1 group (`fd.synthetic_deform_group(1, 2)`) runs in numpy/scipy on 6.88 M voxels; its time is unverified, expected < 1 min. `test_merge_old_record` is described in 5.1.

Hold-out measurements (4.3) are made once by the implementer, not in the suite, with `TF_CUDNN_USE_AUTOTUNE=0` and `--gpu-memory-limit-mb 6000`. The results are transcribed into `models/memory_calibration.json` with `status: "holdout"`, and the raw tool JSON is kept as `.agents/tasks/milestone-9/memory_holdout.json`:

```bash
python tests/gpu/measure_level_memory.py --level 4 --cascade false --tasks deform --widths narrow --n 2,3,4 \
  --gradient-checkpointing full --amp false,true --lean-sampler false,true --lean-losses false,true \
  --split-group-conv false,true --warmup-steps 1 --steps 3 --gpu-memory-limit-mb 6000 --out <holdout.json>
# + the same for --widths default --n 2, and --level 2 --cascade true --widths tiny --n 2
```

Only the four knob combinations of 4.3 are used. The tool measures the full 2³ grid, but the other combinations are kept as extra information and not gated.

## 6. Rest of milestone 9 (design only, R5)

### 6.1 3-level cascade (R5.1)

- `models/cascade.py`: `PREV_LEVEL = {2: 4, 1: 2}`. New `FrozenChain(stages)`, with `.level` = the last stage's level and the `FrozenStage` interface (`linear`, `deform`, `fingerprint`, `models`). Stages are ordered coarse → fine (level 4, level 2). Each is a float32 `GroupRegLevelModel` with `'none'` checkpointing, inference only. `FrozenStage` stays as the one-stage case, so level-2 runs are unchanged.
- Inputs: the level-4 stage gets `pool(pool(images_l1))` and the level-2 stage `pool(images_l1)`, both from the same flipped, augmented group.
- Linear:
  - A4 = stage4(x4).A;
  - x2 = prewarp_linear(pool(I), A4, 2 mm);
  - A2 = Center₂(R2_raw · A4) with the level-2 model's centering (exactly the level-2 run's composed output); A_prev = A2;
  - trainable level 1: x1 = prewarp_linear(I, A2, 1 mm) and A = Center(R1_raw · A2) with the trainable model's centering.
  
  One extra interpolation per level, as in m8.
- Deform:
  - U4 = up(u4) at level 2, and x2 = warp(pool(I), U4);
  - (u2, u2⁻) = compose_cascade(U4, U4⁻, res2) at level 2;
  - U = up(u2) and U⁻ = up(u2⁻) at level 1; x1 = warp(I, U);
  - u = compose(U, u1_res), u⁻ = compose(u1_res⁻, U⁻).
  
  The inverse is unrolled factor by factor: exp(−v1) ∘ up(exp(−v2) ∘ up(φ4⁻¹)). Its error is the sum of the stage errors plus interpolation. It gets a test like `test_two_level_round_trip` over three levels, with margin ceil(max |u|) + 3.
- Source: `--cascade-from <level-2 cascade run>/ckpt/best`. The level-2 model is restored from the run's model variables, and the level-4 stage from its `frozen_stage` extra. The source config.json must hold `cascade` (level 2 trained as a cascade), level 2 and an identical ModelConfig. `--level 1` is allowed with `--cascade-from`.
- Checkpoint: extras `frozen_stage_4` and `frozen_stage_2` (level-2 runs keep `frozen_stage`). config.json has `cascade.stages = [{level, source_checkpoint, source_step, fingerprint}, ...]`. Resume restores both stages and checks both fingerprints (RuntimeError on mismatch). The BN freeze comes from the saved record, and provenance is carried verbatim (m8 rules).
- Skip: `--cascade-skip-prob` is the probability of skipping the LAST frozen stage (level 2 at level 1). A skipped step returns the level-4 output upsampled to level 1 (A_prev = A4, U = up(up(u4))). The default is 0 in code; the level-1 recipe uses 0.15 (design 4.7). The level-4 stage is never skipped at level 1.
- BN recalibration at level 1 uses the unchanged procedure: 64 batches (16/16/32), eager, tape-free, through the cascade input path. With `f0d1` at N 4/16 it takes about 64 × 5–10 s ≈ 5–11 min.
- Memory and time of the frozen stages:
  - phase Z before the tape: `f0d1` N4 ≈ 7 GiB, the frozen level-2 fp32 D2 concat [4, V2, 256] plus its NCDHW copy and skips; well below the training peak;
  - weights: 2 × 4 B × 29.5 M ≈ 0.22 GiB;
  - persistent U/U⁻: 24 B per V1 voxel per image (0.15 GiB per image);
  - time ≈ one level-2 eval forward + one level-4 forward per step (≈ 0.3–1 s, measured as `frozen_ms`; memory as `frozen_peak_mb`).

### 6.2 Level-1 DALI pipeline (R5.2)

`LinearPipeline` / `DeformPipeline` at level 1 read `image.npy` (27.5 MB) and `labels_train18.npy` (6.9 MB) per case from autofs.
- Per step: deform N = 4 → 138 MB, linear N = 16 → 440 MB.
- At the predicted step times (deform ≥ 18 s, linear ≥ 3.5 s) the needed read rates are ≤ 8 MB/s and ≤ 126 MB/s.
- `verify_dali_pipeline.py --level 1` check e measures the throughput at N = 16 with 3 threads.
- If that is below 1.5× the linear need, set `prefetch_queue_depth 2` for the linear pipelines (host memory +440 MB) before considering a node-local copy. A node-local copy is 116 GB for the train set, too large to assume for `$TMPDIR`.

The rules are unchanged: `warp_affine` with `inverse_map=True` and T⁻¹, never float16 inputs, pixel centre 0.5, M_tf = P M_dali P; labels nearest. Checks to run at level 1: a (pixel centre), b, c, d, e, l. Shuffle buffer: `auto_initial_fill` gives ~39 samples at level 1 (1 GiB budget), at least the batch.

### 6.3 Level-1 validation (R5.3)

- Group sizes: deform `--val-group-size` = deform N (4 → 10 groups of the 39 T1 cases, the last padded per the existing rule); linear = linear N (16 → 3 groups). All ordered pairs of a deform group are used (12).
- `--val-pair-chunk 4`: per pair about 404 B per V1 voxel (2.6 GiB), so a chunk is ≈ 10.4 GiB plus the group fields. That stays ≤ 36 GiB, and the m9 job's validation pass confirms it. No label chunking is needed (hard Dice is already argmax-based on the host).
- Frozen baselines:
  - `val_<set>_frozen/...` = the chain's output (levels 4 + 2 composed, the level-2 run's own metrics on the same groups);
  - `val_<set>_frozen4/...` = level 4 only, from the chain's intermediate.
  
  Both are cached per validation set as in m8.
- Wall time per validation: deform 10 groups × (forward ~3 s + 12 pairs × ~1 s) ≈ 2.5 min; linear 4 sets × 3 groups × ~3 s ≈ 0.6 min; ≈ 3 min in total. 16 validations ≈ 0.8 h per long run.

### 6.4 Retraining sequence with the recommended configuration (R5.4)

All steps use the recommended widths (explicit `--filters` / `--decoder-widths` from `recommend --print-env`), the recommended T, and Dice + LNCC 1/1 wherever deform runs. Every submission needs Jay's approval, at most two of our jobs may be active, and the steps are sequential because each warm-starts from the previous one. The m9 memory job may run in parallel with step 1.

GPU type and N are pinned per step (review F10). Steps 1–3 reuse the measured m5 / m7 / m8 recipes unchanged except for the widths. The reduced model only removes channels, so every tensor is the same size or smaller than at default widths at the same level, N and recipe. The measured default-width peaks are therefore upper bounds. Keeping the recipes also makes the comparisons in 6.5 like-for-like.

| Step | Script and env | Recipe | GPU (partition) | Group sizes (source) | Steps | Wall time |
|---|---|---|---|---|---|---|
| 1 | `slurm/train_debug.sh`, `EXTRA_ARGS` = `--filters … --decoder-widths …` (the recommended lists, decoder widths passed explicitly even though the linear run never uses the deform decoder) + `--curriculum true --curriculum-ramp-steps 3000` + the m5 G schedule (below) | m5: rigid + affine 1:1, fp32 | RTX 8000 / A6000 (`rtx8000`) | linear N 32 (m5; measured 26–34 GB at default widths) | 12 000 | ≈ 4.5 h (1.36 s/step); `--time 08:00:00` |
| 2 | `slurm/m7_three_task.sh`, `INIT_WEIGHTS` = step 1 `ckpt/best`, `EXTRA_ARGS` widths + LNCC 1/1, `TASK_GRAD_NORM=12`, `TASK_LOSS_WEIGHTS=none` | m7 three-task 1:1:2, fp32, checkpointing none; Jay's rule: fall back to clip 12 + deform:64 if the smoke enc-grad ratio is outside [0.1, 10] | A100 (`dgx-a100`) | linear 32 / deform 16 (`recommendation.json` `level4.linear_n` / `level4.deform_n`; measured 34.5 GB at default widths) | smoke 200, long 12 000 | ≈ 6 h (m7 LNCC 5 h 51); smoke `--time 00:45:00`, long `--time 10:00:00` |
| 3 | `slurm/m8_level2_cascade.sh`, `CASCADE_FROM` = step 2 `ckpt/best`, widths in `EXTRA_ARGS` | m8: weight copy, BN recal 64, gradnorm 12, AMP, checkpointing both | A100 (`dgx-a100`) | linear 16 / deform 8 (`level2.linear_n` / `level2.deform_n`; measured 27.8 GB at default widths) | smoke 200, long 8000 | ≈ 7.5 h; smoke `--time 02:00:00`, long `--time 12:00:00` |
| 4 | `slurm/m9_level1_cascade.sh` (new, phase 2; template: the m8 script) | 3-level chain, BN recal 64, gradnorm 12, AMP, `full`, the recommended knobs, skip 0.15 | RTX 8000 / A6000 (`rtx8000`). The A100 is excluded: N = 4 fits there only without the margin. | deform N = `level1.deform_n` (= N*, ≥ `n_target`; 4.4 step 5), linear N = `level1.linear_n` (expected 4 / 16), **overridden by the m9 measurement**: the largest measured N with `peak_mb` ≤ 44 GiB for that configuration, never above the model's no-margin N | smoke 100, long 8000 (cut to 6000 if the level-2 plateau at ~6000 repeats) | smoke `--time 03:00:00`; long ≈ 45 h (33–60 h), `--time 72:00:00`; a `--resume` resubmission (Jay's approval) only if the job fails or is preempted |

Step 1 pinning (review v3 F8). The m5 run's exact G schedule is not recorded in this repository (the project notes say only "curriculum + delayed G", G max 45° / 10 mm; its `config.json` is on the cluster and was not read for this design). It is therefore taken verbatim from `results/m5_mixed_long_curr_dG_20261019-205526/config.json` `args`, which m9 job step 0c copies into `scratch/m9/step_times.json` (`--args-keys`): `--global-transform`, `--global-start-step`, `--global-ramp-steps`, `--global-rot-max-deg` (expected 45), `--global-trans-max-mm` (expected 10), `--curriculum`, `--curriculum-ramp-steps` (expected 3000), `--task-ratio`, `--group-size` (expected 32), `--steps` (expected 12 000), `--lr`, `--warmup-steps`. Step 1's `EXTRA_ARGS` / env use exactly those values; the report prints the resulting command. If a key is `null` (old config without it), the argparse default of that flag at the m5 commit applies and the report says so. `--decoder-widths` is passed explicitly in steps 1–4 with the same list, so the ModelConfig in every `config.json` is identical and step 2's partial warm start (`runtime/checkpoint.warm_start_model`, which allows only `deform_decoder` / `decoder_widths` / `integration_steps` to differ) restores with a config diff of exactly `{deform_decoder}`, as in m7. Step 1 also passes the recommended `--integration-steps` for the same reason (it has no effect on a linear run).

Steps 2 and 3 reuse the A100 recipes because their measured default-width peaks (34.5 and 27.8 GB) are ≤ 36 GB. `recommend` checks that the predicted reduced-width peaks of these pinned recipes are ≤ 36 GiB (pre-fit: level 4 32.6, level 2 25.2), and flags them otherwise.

### 6.5 Comparison criteria (R5.5, proposed, A5)

The retrained narrower model is compared against:
- `m7_three_task_lncc` (level 4): rigid 3.82, affine 4.80 mm center; Dice 0.757 / 0.701;
- `m8_level2_gradnorm` (level 2): rigid 2.03, affine 2.53 mm center; Dice 0.846 / 0.789; folding 0.012 %.

Criteria:
- center RMS ≤ +10 %: level 4 rigid ≤ 4.20, affine ≤ 5.28; level 2 rigid ≤ 2.23, affine ≤ 2.78;
- template / pair Dice ≥ −0.01: level 4 ≥ 0.747 / 0.691; level 2 ≥ 0.836 / 0.779;
- folding ≤ 0.05 %.

A failure returns the decision to Jay. Options: `--n-target` lower or a wider preset (only with a different level-1 configuration, because weights copy across levels).

## 7. Error handling

| Operation | Failure | Recoverable? | Caller receives | Logged |
|---|---|---|---|---|
| Parse new flags | invalid value / incompatible task | fatal | `parser.error` (exit 2) naming the option | stderr by argparse |
| Measurement tool `--build-n`, `--val-pair-chunk` | < 1 | fatal | `ap.error` (exit 2) | stderr |
| `ModelConfig.validate` | non-positive width | fatal | ValueError naming the field | — |
| `GroupRegLevelModel(...)` | unknown checkpointing mode | fatal | ValueError (existing) | — |
| `Trainer.__init__` | `model.lean_sampler != cfg.lean_sampler` | fatal | ValueError | — |
| `full` before the parts are built | variables missing | recoverable | forward runs directly (guard: `encoder.built and deform_decoder.built`) | — |
| Lean sampler | non-float coords | fatal | ValueError (existing check) | — |
| Split conv before build | conv not built | recoverable | standard path for that call | — |
| `width_presets` | unknown preset | fatal | ValueError listing the names | — |
| `memory_model.fit` | not converged / an `ok` point outside ±15 % | fatal for `fit` | RuntimeError / exit 1 with the failing points | stdout table |
| Hold-out gate | hold-out error > 20 % | fatal for the test | test failure naming the knob; the inventory is fixed, the gate is not widened | test output |
| `memory_model.recommend` | the hold-out gate fails on the calibration file it reads (an `ok` hold-out point with \|error\| > 0.20, or a hold-out OOM predicted feasible), i.e. the gate was bypassed | stop for Jay | exit 4, the failing points printed | stdout |
| `memory_model.recommend` | nothing feasible at `--n-target` | stop for Jay | exit 3, R3.7 verdict printed | stdout |
| `recommend --t6-verdict` | file missing / unreadable / `t6_accuracy_ok` false | recoverable | T = 6 excluded | stdout warning |
| Coefficients file missing in the tool | — | recoverable | `predicted_*: null` | stderr warning |
| Measurement config | OOM / size limit / other | recoverable | status `oom` / `limit` / `error` (classified in `main()`'s except chain and `measure()`'s inner try); larger N `skipped` after `oom` / `limit` | stdout line, traceback on stderr for `error` |
| Frozen-peak measurement | OOM / error | recoverable | `frozen_status` `oom` / `error`, training status unchanged | stdout |
| Stand-in `PREV_LEVEL` patch | any exception | — | restored in `finally`, the exception propagates to the per-config handler | — |
| Probe op (forward or backward) | OOM / error / too big for the device | recoverable | per-op, per-direction status | stdout + JSON |
| `step_times_from_metrics.py` | run dir, `metrics.jsonl` or `config.json` missing / unparsable line / requested key absent | recoverable | that part marked `missing`, bad lines counted and skipped, absent keys `null`; exit 0 | stdout |
| `memory_model.fit` | an `oom` point predicted feasible, or a `failed_d2` point without a limit flag | fatal for `fit` | exit 1 listing the points | stdout table |
| Lean sampler backward | `dy` contains NaN / Inf | recoverable (step-level) | `dep` stays 0. The non-finite values propagate through dy · g_k and dy · w_k into a non-finite gradient, and the Trainer's existing non-finite skip handles it | existing non-finite counters |
| Lean sampler backward ordering | memory-test case C fails (the control chain is not honoured) | implementation-time | the backward is switched to the specified `tf.while_loop(parallel_iterations=1)` form and the test is rerun; recorded in the report | test output |
| Measurement tool incremental write | `OSError` writing `<out>.tmp.<pid>` or in `os.replace` | fatal for the invocation | traceback, non-zero exit; earlier configurations stay in `--out`; the temp file is removed in `finally` | stderr |
| Level-4 step-time consistency check | predicted m7 mixed mean outside ±20 % of the measured one | recoverable | level-4 A100 times rescaled by the single scalar s | stdout note in `recommend` |
| `m9_memory.sh` call | non-zero exit | continue | first non-zero exit code at the end | `echo ... exit code` lines |

Invariants and owners:
- identical variables at all levels → `ModelConfig` + `copy_weights_from` (unchanged);
- BN updated once under `full` → `models/checkpointing.recompute`, the only place that recomputes BN under that mode;
- lean-sampler consistency between model and losses → `Trainer.__init__`;
- default-path bit-identity → Python-level switches, enforced by the golden tests;
- the selection rule and margin → `memory_model.recommend`, the only producer of `recommendation.json`;
- the measured-N override in retraining step 4 → the report and the phase-2 script header, because Jay submits with explicit env values.

## 8. Tests (unit vs integration) and time budget

Unit, TF (narrow or tiny widths, levels 4 / 2, `tests.tf_setup` first):
- `tests/test_checkpointing.py`:
  - (a) `full` vs `none` gradients within 1e-4 relative, fp32 and AMP (level 4 narrow N 3);
  - (b) BN moving mean/variance after one `full` step equal the `none` step's (≤ 1e-7 relative), and `both` still shows the double update;
  - (b') two `tape.gradient` calls in one step under `full` (persistent tape) still give exactly one update (review F11);
  - (c) frozen statistics unchanged under `full`;
  - (d) `recalibrate_bn` under `full` makes one update per layer and batch (existing T7 pattern);
  - (e) one trace per task with `full` over 6 steps of rigid:1,affine:1,deform:2;
  - (f) the build guard: `build_model` under `full` creates the same variables as under `none`.
- `tests/test_lean_sampler.py`:
  - forward max |diff| 0 vs default (zero and border, fill 0 and 0.3, C 1/3/19, float32/float64), eagerly, and once more inside `tf.function` under `_GrapplerOff` + `_OpDeterminism` (3.3);
  - `test_graph_mode_memory` (local GPU, Grappler on, level-2-size field, N 2; v5, 3.3). Case A: lean one-hot warp peak ≤ 1.5 × its modelled forward moment (2.25 GiB). Case B: lean integrator block ≤ 1.24 GiB and at least 0.85 GiB below the default one (predicted 1.65 GiB). Case C: the backward alone ≤ 0.58 GiB (one-hot) and ≤ 0.41 GiB (field), the sequencing check. Case B is printed without the dependency for information;
  - `lean_images_grad=False`: images gradient `None`, d coords identical to `True`;
  - gradients with respect to coords and volume within 1e-4 relative (target 1e-6);
  - `exp_and_inverse(lean=True)` vs default;
  - deform loss gradients with lean on/off.
- `tests/test_lean_losses.py` and `tests/test_split_group_conv.py`:
  - the on/off tolerances of 3.4 / 3.5;
  - the default path is bit-identical;
  - variable names and count are unchanged with split on;
  - the stem (`pre_activation=False`) keeps `split_group = False` (v5, 3.5);
  - `copy_weights_from` works between split and non-split models.
- `tests/test_width_presets_tf.py`: `count_params` within 1 % of `model.count_params()` for every preset.
- Parser tests in the existing `test_runtime_infra.py` / `test_main.py`:
  - every new flag's invalid values;
  - config/args dicts of today's format restore with the defaults;
  - `check_cascade_config` accepts today's pair;
  - `main.py`'s build N: `--split-group-conv true` with group size 16 and deform group size 8 builds at 2, and without the flag at 8, as today (the helper is factored as `_build_n(args)` so it can be tested without building).
- `tests/test_integration_steps.py` (3.6), including writing `integration_steps.json`.
- Goldens unchanged: `m7_linear_golden.json`, `lncc_deform_golden.json` at max |diff| 0.

Unit, no TF: `tests/test_memory_model.py`, `tests/test_width_presets.py`, `tests/test_step_times.py`, `tests/test_slurm_scripts.py` (m9), and the probe's m8-errors mode.

Integration (local GPU):
- `tests/test_measure_memory.py::test_tiny_level1`, `::test_merge_old_record` and `::test_incremental_write` (v5, 5.1);
- probe graph mode and run mode (tiny sizes);
- an e2e `main.py` overfit-synthetic deform run at level 4 with `--gradient-checkpointing full --lean-sampler true --lean-losses true --split-group-conv true`, passing `check_mixed_run.py`.

The hold-out measurements (5.6) are a one-off implementer step, about 1 h on the GTX 1080, and are not part of the suite.

Added suite time (estimate; the report states the measured value): ≈ 16 min (the graph-mode memory test adds < 1 min). That is the level-1 tiny tool run ~5 min, the level-1 integration-step fields ~3 min, the checkpointing / lean / split tests ~6 min, and the rest < 2 min, within the 20-minute budget (N3).

## 9. Files

New:
- `models/checkpointing.py`, `models/width_presets.py`, `models/memory_model.py`;
- `models/memory_calibration.json` (calibration + hold-out), `models/memory_coefficients.json`;
- `scripts/memory_model.py`, `scripts/step_times_from_metrics.py`;
- `tests/gpu/probe_tensor_limit.py`, `tests/gpu/frozen_chain_standin.py`, `slurm/m9_memory.sh`, `tests/field_helpers.py`;
- the tests in section 8;
- `.agents/tasks/milestone-9/memory_holdout.json` and `.agents/tasks/milestone-9/report.md` (after implementation).

Modified:
- `models/layers.py`: lean sampler, `batch_norm_layers`, split conv and its setter;
- `models/diffeo.py`: `lean` plumbing;
- `models/group_net.py`: `full`, constructor `lean_sampler` / `split_group_conv`;
- `models/config.py`: positivity check only;
- `runtime/losses.py`: lean sampler / lean losses plumbing;
- `runtime/run.py`: `TrainConfig` fields, consistency check, plumbing, `_bn_layers` via the helper;
- `runtime/args.py`: flags and validation; `main.py`: pass the new flags, and `n_build = min(n_build, 2)` when `--split-group-conv` is true (2.4);
- `tests/gpu/measure_level_memory.py`, `tests/gpu/verify_dali_pipeline.py` (`--level 1`);
- `tests/run_tests_wsl.sh`, `tests/test_slurm_scripts.py`, `tests/test_measure_memory.py`, `tests/test_diffeo.py` (imports the moved helper);
- `.agents/tasks/network-design/design.md` (v3.13).

## 10. Design document v3.13 (updated with this revision)

Updated sections:
- the status line;
- 4.5: checkpointing modes including `full`; the BN rule "call 1 = forward, every later call restores";
- 4.6: T as one value at every level (Option B), the T = 6 verdict file, the revised folding criterion;
- 9: the level-1 root cause with the corrected cross-check (level-2 points −5 … +1 % pre-fit, backward transient), the knobs, the forward + backward tensor bound, the expected configuration `f0d1` N = 4 and the target-N selection rule. v3 adds: the linear cross-check and the precision-split encoder coefficients; the build-forward rule (never in the peak, built at `min(N, 2)`, checked by `limit_flags`); the corrected v3.9 LNCC sentence (the `none` peak is in the forward-end phase, and loss terms need not coexist).
- 11: the milestone-9 row (phase 1 scope and verification, pinned retraining GPUs).
- v4 adds to 9: the lean sampler's `dy`-dependent backward recompute (Grappler dedup guard) and its graph-mode memory test; the timed skip-gradient term and the conservative bound (`f0d1` N4 33.5 / 34.9 GiB); and the step-time data sources (window means; per task only for cascade runs).
- v5 adds to 9 (still v3.13, as in the earlier revisions of this design):
  - the sequenced lean backward (control-dependency chain on the accumulators, with the `while_loop` fallback; the dy guard is kept, and `dep` is always 0);
  - the lean forward / backward transients and the sub-phases Fw / Fb;
  - the re-checked level-1 numbers: N 4 unchanged at 33.5 GiB, N ≥ 5 infeasible with the margin, and under the order-independent forward bound N 4 = `f0d1` + lean losses;
  - the rewritten graph-mode memory test (level-2 size: predicted saving 1.65 GiB, asserted ≥ 0.85 GiB);
  - forward corner sequencing as the backlogged remedy.

  4.5 adds `lean_images_grad`.

## v4 → v5 responses (design review of v4)

| Finding | Resolution |
|---|---|
| F1 (MEDIUM) lean backward corners not sequenced; forward transient missing from the inventory | **Addressed (orchestrator decision).** The lean sampler's backward ORDERS its 8 trilinear-corner computations so that at most one corner's temporaries are live at a time. The corners are accumulated SEQUENTIALLY with explicit `tf.control_dependencies` chaining on the accumulators (d_coords, d_flat). The specified fallback, used only if the memory test's case C fails, is `tf.while_loop(parallel_iterations=1)` over k = 0..7 carrying the accumulators (3.3, U19). The v4 Grappler-CSE guard (the dy dependency that stops CSE from undoing the backward recompute) is KEPT. `lean_images_grad` (Python, default True) avoids an unprunable d_flat branch for data-volume warps. Inventory (2.1, 4.2): the forward transient T_fwd = 84 + 32·C B plus the 168 B of int64 GPU indices, live only while the call runs, in the new sub-phase Fw at its call site. Fw uses a timed program-order rule, with an order-independent bound. The sequenced backward transient T_bwd = 88 + 8C (+16C) B is in the new sub-phase Fb and in I. Re-check (2.5, 4.4): `f0d1` LS N4 is unchanged at 33.5 / 38.5 GiB (Fw 32.0, Fb 25.5; D binds). Fw caps every lean configuration at N 4 with the margin, so the v4 alternatives at N 5–7 are gone. Under the order-independent bound N 4 is `f0d1` + lean losses, 35.0 / 40.3 GiB. Forward corner sequencing is backlogged as the remedy (3.7). Test (3.3) on a level-2-size field, N 2, three cases. Case A: lean one-hot warp ≤ 1.5 × its modelled 936 B moment (2.25 GiB). Case B: lean integrator block ≤ 1.24 GiB, and the predicted saving vs the default sampler is 1.65 GiB (1.77 GB), asserted ≥ 0.85 GiB. Case C: the backward alone ≤ 1.5 × T_bwd (0.58 / 0.41 GiB), which separates sequenced from unsequenced by 4–6×. v4's "≤ 0.6 × default" assertion is dropped because case A cannot meet it by construction (the forward transient). |
| F2 (NIT) `dep` NaN mechanism | Fixed (3.3, section 7). `dep` is always 0 and only adds a data dependency. A non-finite dy propagates through dy · g_k (and dy · w_k), and the Trainer's non-finite skip handles it. The outcome is unchanged. Folds into F1. |
| F3 (NIT) stale "optional coefficient"; unreachable exit 4 | Fixed. The section 7 `fit` row now reads "an `ok` point outside ±15 %". The "m > 0.30" clause is dropped, since m ≤ 0.20 whenever the gate passed. Exit 4 is redefined as the bypass case: `recommend` re-checks the hold-out gate on the file it reads and exits 4 if it fails (4.3, section 7). |
| F4 (NIT) `level1.deform_n` ambiguous | Fixed: `level1.deform_n` = N* (≥ `n_target`), and `n_target` is added to the `level1` block (4.4 step 5, the JSON example, 6.4 step 4). |
| F5 (NIT) split conv on the stem | Fixed by exclusion: `set_split_group_conv` skips `pre_activation=False` blocks, so the stem always uses the standard path, with a test (3.5, 8). |
| F6 (NIT) job step 5 knobs | Fixed. Step 5a measures the pinned m8 recipe at the reduced widths (`both`, all knobs false, N 8). Step 5b runs `full` with `$KNOBS` at N 8, 12, 16, so lean losses follow the recommendation. 4 configurations instead of 6; the stub test checks both calls (5.5). |
| F7 (NIT) JSON written only at the end | Fixed: merge and atomic write (`<out>.tmp.<pid>`, fsync, `os.replace`) after every configuration, with the same merge semantics, `OSError` fatal, and `test_incremental_write` (5.1, 7, 8). |
| F8 (NIT) deform pre-fit table not pinned | Fixed (4.6). Pre-fit assertions within 0.1 GiB pin the six level-2 deform rows of 2.3 (the `both` rows with `B_sg`, values listed) and `f0d1` LS N4 = 33.5 ± 0.1 GiB. Pins for the new Fw / Fb terms are added as well. |

Unchanged from v4: `f0d1` widths, AMP, `--gradient-checkpointing full` with BN updated once per step, `--lean-sampler` + `--split-group-conv`, T = 7, the frozen-chain stand-in, and level-1 deform N = 4, provisional until the cluster memory job.

## Review responses (design review of v3)

| Finding | Response |
|---|---|
| F1 (HIGH) Grappler dedup undoes the lean sampler's recompute | Addressed (3.3, U18). The backward builds `dep` from `dy` exactly as TF 2.10's `recompute_grad` does and recomputes all corner math from `coords_b = coords + dep`. Only `coords` is perturbed, not `flat`: every recomputed node, including every gather (through its index input), descends from `coords_b`, so nothing can be merged, and perturbing `flat` would cost a full input copy (4C B per voxel). The forward is unchanged and bit-identical. Inside the `full` integrator blocks each composition gets its own `dy`-dependency from the block's inner tape. New `test_graph_mode_memory` (local GPU, Grappler on, `tf.function` + gradient): one-hot warp at level 4, C = 19, N = 2, and the `recompute_grad` integrator block; assert lean peak ≤ 0.6 × default peak (expected ≈ 0.3 / 0.25), and print the ratio with the dependency switched off. Dependent conclusions (2.5, 3.6, 3.7, 4.4) are marked conditional on this test; 3.7 names per-squaring-step checkpointing as the next knob if case B failed. |
| F2 (MEDIUM) `metrics.jsonl` has window means; m7 has no per-task times | Addressed (4.5). Verified in `runtime/run.py`: per-step records only in `Trainer.history`; `metrics.jsonl` holds `time/compute_ms` for every run and `train_<task>/compute_ms` only with a cascade. `step_times_from_metrics.py` reads those window-mean keys, reports median / IQR and `per_task`. Level-2 per-task A100 times come from m8; level-4 per-task times from m9 job steps 6 / 6b (step 6 now also measures default widths) × the m8 A100 / RTX 8000 per-task ratio (U17); the m7 mixed mean is a ±20 % consistency check, with a single-scalar fallback. `tests/test_step_times.py` uses fake window-mean files, cascade and non-cascade. |
| F3 (NIT) window-mean `diag_u_res_max_vox` | Addressed: max over window means; rerun if 2 × max > 8 (3.6). |
| F4 (NIT) eager vs graph forward test | Addressed: the max \|diff\| 0 comparison runs eagerly, plus once under `_GrapplerOff` + `_OpDeterminism`; stated in the docstring (3.3, 8). |
| F5 (NIT) OOM point predicted feasible | Addressed: hinge residual max(0, 0.98 · cap − pred) / cap per OOM point, and `fit` exits 1 listing any OOM point predicted feasible or `failed_d2` point without a limit flag (4.3, 4.6, 7). |
| F6 (NIT) `integration_steps` 0 vs 7 | Addressed: `config_key` normalises 0 to the resolved ModelConfig T; new records store the resolved T; `test_merge_old_record` extended (5.1). |
| F7 (NIT) tie-break order | Addressed by matching R3.6: parameters → T rule → fewer knobs → step time; the change from v3 alters no expected row (4.4). |
| F8 (NIT) step 1 G schedule and decoder widths | Addressed with a deviation: the m5 `config.json` is on the cluster and not in the repo, so the values cannot be pinned as numbers here. m9 job step 0c copies the named `args` (G schedule, curriculum, task ratio, N, steps, lr, warm-up) from it via `--args-keys`, and step 1 uses them verbatim. `--decoder-widths` and `--integration-steps` are passed explicitly in steps 1–4, so step 2's partial warm start sees a config diff of exactly `{deform_decoder}` (6.4). |
| F9 (NIT) skip gradients missing from D | Addressed with a timed rule (2.1, 4.2): d(skip_k) appears at the backward of the block that consumes it and lives until the encoder backward. At the D1 moment (the level-1 peak) only finer blocks' skip gradients count, which is none; D1's own skip slice is part of its input-gradient transient. The term is in the later D sub-phase and in E (`f0d1` E 1.53 → 1.86 GiB per image). At level 2 it adds +1.5 / +1.8 % to the `both` rows. The review's reading (all ≈ 52 B at D1) is kept as a conservative bound, `predict(skip_grad_bound=True)`: `f0d1` N4 34.9 / 40.1 GiB, still feasible; the N6 per-N row moves to `fr25` (2.5, 4.4). |

## Review responses (design review of v2)

| Finding | Response |
|---|---|
| F1 (MEDIUM) ±15 % shown for deform only; build forward misplaced; `none` peak conflict | Addressed. (1) 2.3 adds the level-2 linear table: pre-fit −8 … +12 %, opposite signs by precision. The task-touched resident (U16) explains part of the AMP intercept. The rest is pre-registered as the precision-split encoder coefficients `c_enc32` (expected 1.10) and `c_enc16` (expected 0.89), bounds [0.6, 1.6], derived from the measured slopes. With them the linear points are −1 … +6 % and the deform points move by ≤ 2 %. `c_net16` is dropped. (2) Rule stated in 2.1, 4.2 and 4.3: the build forward never enters the predicted `peak_mb` (the tool resets after the build) and is evaluated only by `limit_flags` at the record's `build_n`. (3) LNCC on/off rows added: the predicted Δ is ≤ 4.4 % of the peak, against a measured noise of ±65 MB. The `none` phase is reconciled: level-2 data reject the integrator-backward sub-phase I₀ (−20 %) and keep F. The level-4 LNCC data are modelled by `c_loss` (loss terms need not coexist, U15). design.md section 9 is corrected. The `fr50` / `fr25` lean-loss conclusions are restated as `c_loss`-dependent; the N = 4 recommendation is not affected (2.5). The level-4 linear N 32 range is marked unverified (U14), and job step 6b measures it. |
| F2 (MEDIUM) level-1 build forward unsplit and outside feasibility | Addressed with option (a). The tool builds every configuration at `min(N, --build-n 2)`. `main.py` builds at `min(n_build, 2)` when `--split-group-conv` is true. `limit_flags` and rule 4.4 step 2(d) also check the build at `build_n`. Tests: `f0d1` N6 passes at `build_n` 2 and is flagged at `build_n` 6; m8 linear fp32 N12 is flagged; `main.py` `_build_n` (2.4, 3.5, 4.6, 5.1, 8). |
| F3 (NIT) per-N rows N3 / N5, 2.5 column | Fixed: N3 → `default` + L + split (32.9 / 37.8), N5 → `fr50` + L + split (35.4 / 40.7), `default` + L largest N 2 / 2. The generated table is stated as authoritative (4.4). |
| F4 (NIT) stand-in module | `tests/gpu/frozen_chain_standin.py` with `FrozenChainStandIn`, `build_standin`, `patched_prev_level()` (5.2, 9). |
| F5 (NIT) lean-loss plumbing | Signatures of `deform_loss`, `pair_dice_loss`, `template_ncc_loss` and `pair_ncc_loss`. `_make_deform_step` passes the cfg values; validation keeps False (3.4). |
| F6 (NIT) `restore_bn` | Dropped (3.2). |
| F7 (NIT) TF32 | `probe_tensor_limit.py` calls `configure_tf_precision(tf32=False, ...)` first. TF tests import `tests.tf_setup` (verified to turn TF32 off) (3.6, 5.3). |
| F8 (NIT) T = 6 quantity and magnitude | e(T) = max over both orders of the mean interior error. Magnitude 8 is stated as conservative. Step 0c extracts `diag_u_res_max_vox` from m8, and the test is rerun at a larger magnitude if 2 × p95 exceeds 8 (3.6). *Superseded in v4: max of the window means (review v3 F3).* |
| F9 (NIT) local T = 6 / 7 measurement | Exact command (`--integration-steps 6,7 --lean-sampler false,true --gradient-checkpointing integrator,full`), output `memory_holdout.json`, reported in GB per image next to the model (3.6). |
| F10 (NIT) AMP-only level 1 | Header line in `m9_memory.sh` (5.5). |

Also corrected without a finding: the trainable parameter counts. `default` and `f0d1` were built locally (29.63 M / 29.54 M); v2 said 28.6 / 28.5 M. The other presets are approximate, and the ranking is unchanged (3.1).

## Review responses (design review of v1)

| Finding | Response |
|---|---|
| F1 (MEDIUM) accuracy claim / AMP-both residual | Addressed. v1 compared `both` against phase I and had no backward transient. With the D-phase transient (`c_bwd` × the largest block input × 4 B, U13) and the corrected head (4C), all 21 level-2 deform points are within −5 … +1 % pre-fit, and AMP `both` is exact at N2/4/6/8 (2.3). The v1 residual is attributed to that named term. It scales with the largest block at its resolution (D1 at V1 at level 1), and the level-1 tables (2.5) include it. U4 is shown to move only `none` / `integrator` (+3 … +5 %) and not the lean-sampler predictions. The claim is restated in section 0 and design.md section 9. |
| F2 (MEDIUM) coefficient form vs opposite-sign residuals | Addressed. The opposite sign was an artefact: v1 subtracted pair growth from the narrow-`both` slope, but under `both` the pairs live only in F, which is not the peak. Corrected, narrow `both` is +4 … +9 % pre-fit (2.3). The phase-aware `c_rec` (bounds [0.5, 2.0]) and a per-GPU-family `c_0` are pre-registered anyway, with expected values from the hand numbers (4.3). Exit 1 stays the last resort. |
| F3 (MEDIUM) no measured point for the knobs | Addressed. A GTX 1080 hold-out set (level 4 narrow N 2/3/4, default N 2, level 2 tiny N 2 × four knob combinations × fp32/AMP), status `holdout`, gate ±20 %. The margin is max(0.15, worst hold-out error) and stops for Jay above 0.30 (4.3, 5.6). The recommendation is printed and stored as provisional, always with the margin N (4.4). |
| F4 (MEDIUM) tensor-limit evidence forward only | Addressed. The probe runs Conv3DBackpropInput/Filter, FusedBatchNormGradV3 and LeakyReluGrad under `GradientTape` and reports `verified_max_elements` per direction. The model's backward bound defaults to 2³¹ − 1 in both precisions (2.4). Re-evaluated in 2.5: the default and `f0d1` presets need split to reach N = 4, and `fr50` without split stops at N = 4. Split is now part of the expected configuration. |
| F5 (MEDIUM) rule vs expected outcome | Addressed with option (b). Level 1 uses AMP + `full`; the rule maximises parameters subject to N ≥ `--n-target` (default 4) with the margin, and the per-N table is always printed (4.4). The pre-fit outcome follows from the rule: `f0d1` at N = 4. The N 5–7 alternatives (`fr50`, `fr25`) are listed for Jay. This changes R3.6, justified by Jay's message 12, and the change is stated. |
| F6 (MEDIUM) T = 6 verdict not wired | Addressed. `integration_steps.json` holds `t6_accuracy_ok`, `recommend --t6-verdict` excludes T = 6 without it, and rule step 4 implements "raises N or restores ≥ 10 % headroom" (3.6, 4.4). |
| F7 (MEDIUM) frozen-stage peak, no `--cascade false` point | Addressed. `frozen_peak_mb` / `frozen_status` (reset → frozen forward → peak, trainer alive), job step 2b (level 1 `--cascade false`, N 2/4), `cascade_kind` on every record, `--time` raised to 10 h (5.1, 5.2, 5.5). |
| F8 (MEDIUM) key fields and old-record defaults | Addressed. The exact `KEY_FIELDS`, the non-key fields, the old-record defaults (`cascade_kind` from `cascade`) and `test_merge_old_record` are all specified (5.1). |
| F9 (MEDIUM) `peak_mb` vs `warmup_peak_mb` | Addressed. Calibration and feasibility use `peak_mb`; U11 is checked by job step 0b; `warmup_excess_gib` is reported. The autotune excess is accepted, with the reason (U12): failed workspace allocations are non-fatal (4.1). |
| F10 (MEDIUM) retraining GPU type and N | Addressed. One GPU type and N per step. Steps 1–3 reuse the measured m5 / m7 / m8 recipes (a monotonicity argument and like-for-like comparison). Step 4 runs on the 48 GB cards with N from `recommendation.json` `level1.*`, overridden by the m9 measurement, `--time 72:00:00` (6.4). The JSON field names are fixed in 4.4, and the scripts take env values from `--print-env`. |
| F11 (NIT) BN counter | Addressed: every call after the first restores, with test (b') (3.2, 8). |
| F12 (NIT) cross-reference | Addressed: the linear level-1 row is in 2.5, and 3.2 points to it. |
| F13 (NIT) SVF head 4C | Addressed (2.1, 2.2). |
| F14 (NIT) `limit` classification | Addressed: in `main()`'s except chain and `measure()`'s inner try, by exception type plus the substrings listed (5.1). |
| F15 (NIT) folding criterion | Addressed: folding(T6) ≤ max(0.1 %, folding(T7) + 0.05 pp) (3.6). |
| F16 (NIT) LNCC correction | Addressed: `lncc: false` on the m6 points, and the correction is noted in section 1 and in the report. |
| F17 (NIT) step-time identification | Addressed: per-task `compute_ms` from the m7 / m8 `metrics.jsonl` (job step 0c, `step_times_from_metrics.py`), and the weak identification is stated (4.5). *Corrected in v4 (review v3 F2): window means; per task only for the m8 cascade run.* |
| F18 (NIT) `--build-n` validation | Addressed: `ap.error` if < 1 (5.1). |
| F19 (NIT) `full` build guard | Addressed: recompute only if `encoder.built and deform_decoder.built` (3.2). |
