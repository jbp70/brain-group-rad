# Unified groupwise rigid / affine / deformable registration: design document

Status: v3 (2026-10-16). Incorporates Jay's answers to the v1 and v2 open questions. Decisions that changed since v1 are marked *(v2)* or *(v3)*. Remaining items are in section 12.

Related documents:
- Reference-code review: `.agents/tasks/reference-code-review/report.md` (local only)
- Verified centering math: `scripts/demo_central_transform.py` (numpy reference), `scripts/check_tf_group_centering.py` (TF 2.10 `logm_real`, `GroupCentering`)
- Preprocessing: `preprocessing/` and `.agents/tasks/preprocessing-pipeline/final_report.md`

---

## 1. Goal and claims

One model registers a group of N 3D brain MRIs to an unbiased central space with a rigid, affine or diffeomorphic deformable transform. It has one shared encoder and task-specific decoders. Pairwise transforms between any two members follow by composition through the central space.

Claims the design has to support:
1. **One model, three transform families.** Shared encoder; rigid, affine and deformable decoders.
2. **Groupwise by construction.** Outputs are centered in the graph: a Karcher barycenter for linear transforms, the group-mean velocity for deformable ones.
3. **Pairwise for free.** φ_ij = Ψ_j⁻¹ ∘ Ψ_i is exactly invertible and exactly transitive (cycle-consistent).
4. **Full 1 mm resolution** for the final deformable level.
5. **Keypoint-free.** Linear parameters are regressed directly.
6. **Scales to large N at test time.** This is structural in config B (4.2).

Baselines: BrainMorph, KeyMorph, SynthMorph (affine/joint), MultiMorph, ANTs, FreeSurfer `mri_robust_template`, and arXiv 2501.13010 (longitudinal rigid).

---

## 2. Conventions (single source of truth)

| Item | Convention |
|---|---|
| Array axis order | `[..., X, Y, Z, C]`, channel-last. For LUMIR the array axes are RAS-like (corrected affine x-row `[1, 0, 0, -78]`; grid built with `--force-flip yes`) |
| Physical units | mm. World coordinate of voxel index p is (p − c) · s, with c = (n − 1)/2 per axis and s = 4 / 2 / 1 mm at levels 1/4, 1/2, full. Block averaging by 2 or 4 preserves this exactly (coarse voxel centers land on the same world points) |
| Rotation center | Image center c, the same physical point at every level |
| Matrices | 4×4 homogeneous in heads, centering and losses; sliced to 3×4 for warping |
| Augmentation T_i | Maps template (pre-augmentation) points into image i. Image i is rendered by pulling: I_i(y) = I(T_i⁻¹ y) |
| Linear prediction A_i | Maps image-i coordinates to central space. Image i in central space = I_i ∘ A_i⁻¹ (pull with A_i⁻¹) |
| Deformable φ_i | Maps central-space points to affine-central coordinates of image i (pull). Full map Ψ_i = A_i⁻¹ ∘ φ_i: central → image i. Warped image = I_i ∘ Ψ_i |
| Pairwise map | Image j coordinates → image i coordinates: Ψ_i ∘ Ψ_j⁻¹. Linear-only: A_j⁻¹ A_i maps i → j, with true value T_j T_i⁻¹ |
| DALI ↔ TF matrices | DALI uses reversed axis order. M_tf = P · M_dali · P (P = axis reversal), verified in the reference review |
| Translation ranges | Specified once in mm and identical at every level *(v2)*. The rigid project's per-level doubling compensated for voxel units and is no longer needed. The pipeline converts mm to voxels with that level's spacing |
| Sampling grids | float32 (float64 when the matrices are float64, as in gradient tests), never float16, even under mixed precision. Images may be float16 |
| Displacement fields | Voxel units of the level they live at; ×2 when upsampling to the next level |

---

## 3. Data

### 3.1 Training data and on-disk layout *(v2)*

- Train: LUMIR25 training (3,384 T1w). Validation: the **39 T1w LUMIR25 val images only** *(v3)*. The 10 non-T1 `unknown0001` cases are excluded until synthetic-contrast training exists.
- **Precomputed per-level `.npy` files** (decided). One set per level: full (160×224×192), `_2` (80×112×96), `_4` (40×56×48). This uses more disk but loads faster and keeps the resize work out of DALI and off the GPU.
  - **Images:** block average (2×2×2 or 4×4×4), stored as **float32** *(v3)*. Augmentation (warp, intensity) runs in float32, and images are cast to float16 only afterwards, at the model input, for mixed precision. The full-level network's own inserted average pools see exactly this block-averaged signal, so coarse-level training inputs match what the finer levels compute internally.
  - **Labels:** merge with the training label LUT first (3.4), then block mode (majority vote) for `_2`/`_4`, uint8.
  - Full-resolution 103-ID SynthSeg labels are kept for evaluation.
  - **Requires a preprocessing change:** `export-npy` gets per-level outputs and the merged-label option. Jay runs the export.
- Later test sets: OASIS-2 (rigid/affine, longitudinal) and Mindboggle-101 (deformable, manual DKT labels), preprocessed into the same grid. Synthetic, contrast-randomized images come in a later phase.
- Disk estimate per case (float32 images, uint8 labels): full 27.5 + 6.9 MB, `_2` 3.4 + 0.9 MB, `_4` 0.4 + 0.1 MB, about 39 MB per case. Roughly 135 GB for train plus T1 val.

### 3.2 DALI pipelines

- **Three pipelines:** `RigidPipeline`, `AffinePipeline`, `DeformPipeline` (rigid and affine can share one class with different ranges). Each is a separate `DALIDataset` with a fixed output signature. No `external_source`.
- **One group per batch:** DALI batch size = N (fixed per level), tensor `[N, X, Y, Z, 1]`, group mean over axis 0.
- **Sampling:** numpy reader over the level's files, cross-subject groups. Default `shuffle_after_epoch=True` (the file list is pre-permuted by seed, then reshuffled every epoch; no host buffer). The `random_shuffle` + `initial_fill` buffer mode is optional, since a full-dataset buffer is only affordable at level 4. Same-subject groups are deferred. *(v3.1, milestone 2)*
- **DALI conventions, verified on glacier (milestone 2):** pixel center 0.5, `warp_affine(inverse_map=True)` given T_i⁻¹, DALI matrices in reversed axis order (M_tf = P M_dali P). `warp_affine` does not accept float16 in DALI 1.18, so images stay float32 through augmentation.
- **Per-image transforms:** DALI random ops are per sample, so every image gets an independent T_i about the center. Warped with `fn.warp_affine(inverse_map=True)` given T_i⁻¹, on GPU, linear interpolation. Labels use nearest neighbour.

**Augmentation ranges** *(v2)*. All are uniform and independent per image. Scale is sampled in log space; shear is the upper-triangular entries of K.

| Pipeline | Rotation | Translation | Scale | Shear | Network undoes |
|---|---|---|---|---|---|
| Rigid | ±45° (random axis) | ±15 mm *(v3; may increase later)* | nuisance ±0.5% | nuisance ±0.25% | rotation + translation only |
| Affine | ±45° | ±15 mm | ±10% | ±5% | everything |
| Deformable | residual: ±1° | ±1 mm | ±0.5% | ±0.25% | (absorbed by the deformable field) |

- **The deformable residual affine is a flag** (`deform_residual_affine`, default **off**) *(v2)*. When off, `DeformPipeline` does no warp at all: it only reads images and labels, so it's the cheapest pipeline.
- **Outputs:**
  - Rigid/affine: images `[N,X,Y,Z,1]` and T_i `[N,3,4]` (DALI order).
  - Deformable: images and labels `[N,X,Y,Z,1]` (labels uint8, already merged).
- **Intensity augmentation** from the rigid project (noise, brightness, contrast, gamma, simulated low resolution), per image. **No flips in DALI.**
- **Memory:** `prefetch_queue_depth=1` per pipeline to start, plus TF memory growth (or a fixed TF memory limit). Measure, then tune.
- **Determinism:** one seed per pipeline (seed + 0/1/2), and a seeded Python RNG for the task schedule and flips.

### 3.3 Post-DALI processing in TF (in the train step, outside the tape)

1. **Flips:** one random flip state F per batch (= group), applied to all members.
   - Images: `tf.reverse`.
   - Linear: T_i ← F T_i F.
   - Labels: flipped. No L/R label swap is needed, and left/right are merged anyway (3.4).
2. **Conversion:** DALI to TF matrix order; voxels to mm.
3. **Targets** (linear tasks only):
   - μ* = barycenter{T_i⁻¹}, run to convergence (8–10 iterations).
   - A_i* = μ*⁻¹ T_i⁻¹.
   - Pair targets: T_j T_i⁻¹.
4. **Deformable labels:** one-hot over the merged label set (3.4), fp16.

### 3.4 Training label set *(v2)*

The SynthSeg `--parc` output (103 IDs) is merged into a compact training set:

| Merge rule | Result |
|---|---|
| All 68 DK cortical parcels (1001–1035, 2001–2035) and SynthSeg cortex (3, 42) | one **cortex** label |
| Left/right pairs | one label each: cerebral WM, cortex, lateral ventricle, inferior lateral ventricle, cerebellum WM, cerebellum cortex, thalamus, caudate, putamen, pallidum, hippocampus, amygdala, accumbens, ventral DC (14) |
| Midline structures | 3rd ventricle, 4th ventricle, brainstem, CSF (4) |
| Background | 0 |

The result is **18 foreground labels plus background**: 19 channels, about 260 MB per full-res image in fp16, versus 1.4 GB for 103 labels.

The merge is a fixed LUT applied at export (`labels_train18`), so the coarse-level mode pooling is computed on merged labels. Full-res 103-ID labels are kept for evaluation.

**Decided** *(v3)*: train with this 18-label set. A richer set (DK parcels merged L/R, about 52 labels) stays a possible later ablation for cortical Dice on Mindboggle.

---

## 4. Architecture

### 4.1 Group block

**Block = norm → act → concat(group mean) → conv.**
- The group mean is the mean over the N axis of the normalized and activated features, broadcast back to N.
- The concat doubles the conv's input channels, identically at every level, so weights copy across levels.
- **Normalization:** BatchNorm (`BatchNormWithSeparateAffine` from the rigid project: statistics can be frozen while γ/β stay trainable). Statistics come from the N images of the single group. The norm type is configurable.
- A non-group block is the same without the concat.
- **Single conv vs residual (flag):** the residual variant is two blocks plus a skip, with a 1×1 projection where channels or stride change. Default: single conv.

### 4.2 Group interaction configurations (flag)

| Config | Group blocks in | Purpose |
|---|---|---|
| **A** | stem conv, every encoder layer, linear decoder, deformable decoder | Full group interaction (MultiMorph-style) |
| **B** | only layers after the encoder bottleneck: the linear decoder (1×1 reductions and dense layers) and every deformable decoder stage | Per-image encoder; the decoders learn the groupwise reasoning. At test time the encoder can run in chunks and its features can be cached |

A vs B is a planned ablation.

### 4.3 Encoder (shared by all tasks, one instance per pyramid level)

Starts from the rigid project's `init_small` with filters (32, 64, 128, 192, 256, 384), and uses your parameter-free pool trick.

**Stem** *(v2)*: the input image is convolved once (32 filters, 3³), with no norm or act before it, immediately ahead of the first normalized block. Normalizing an already-normalized input image serves no purpose. In config A the stem is a group conv: concat(image, group-mean image) → conv. That also covers v1's question about a "group-mean input channel".

| Layer | Op | 1/4 level | 1/2 level | Full level |
|---|---|---|---|---|
| Stem | conv (group conv in A) | 40×56×48 | 80×112×96 | 160×224×192 |
| (pool) | parameter-free avg | — | — | ↓2 → 80×112×96 |
| L2 | block | 40×56×48 | 80×112×96 | 80×112×96 |
| (pool) | parameter-free avg | — | ↓2 → 40×56×48 | ↓2 → 40×56×48 |
| L3 | block | 40×56×48 | 40×56×48 | 40×56×48 |
| L4 | block, stride 2 | 20×28×24 | 20×28×24 | 20×28×24 |
| L5 | block, stride 2 | 10×14×12 | 10×14×12 | 10×14×12 |
| L6 | block, stride 2 | **5×7×6** | **5×7×6** | **5×7×6** |

Every level has identical trainable parameters, so every parameter is first trained at 1/4 scale with a large group.

### 4.4 Linear decoder (rigid and affine)

1. 1×1 reductions after L6 with filters (192, 96, 48, 24). These are group blocks in both configs.
2. Flatten the 5×7×6×24 = 5,040 features per image. This is deliberate: it keeps the spatial layout that translation regression needs.
3. Concat with the group mean of the flattened vectors, then dense layers (2048, 512), each with a group-mean concat, then **standard dropout p = 0.15** *(v2)*. There's no recurrent dropout, because prior levels are frozen.
4. **Heads** *(v2: shared rotation branch plus an extra scale/shear branch)*:
   - **Rotation + translation branch (shared by rigid and affine):** 10D rotation representation (validated at ±45° in the rigid project) plus t in mm. 13 outputs. Zero kernel with identity-encoding bias.
   - **Scale/shear branch (affine only):** 3 log-scales and 3 shears, giving K (upper-triangular, diagonal exp(s)). 6 outputs, zero-initialized, so K = I at start.
   - Rigid: A = [R | t]. Affine: A = [R · K | t]. This covers every orientation-preserving affine (QR factorization), is always invertible, and keeps the matrix log in centering well-defined.
5. **Centering layer (`GroupCentering`):** an unrolled 4-iteration group barycenter μ of the raw predictions (`logm_real` with a custom gradient, plus `tf.linalg.expm`), then A_i = μ⁻¹ A_i_raw. Gradients flow through μ. Verified in `scripts/check_tf_group_centering.py`.

### 4.5 Deformable decoder (U-Net)

- **One conv per stage** *(v2)*: trilinear ×2 upsample, concat the encoder skip, one block. These are group blocks in both configs.
- **Decoder widths:** half the matching encoder widths by default (e.g. 192, 128, 96, 64, 32, 16), configurable to 1×.
- **Parameter-free trilinear ×2 upsamples** are inserted at finer levels exactly where the encoder inserted pools, so every level has the same trainable decoder.

| Stage | Skip | 1/4 level | 1/2 level | Full level |
|---|---|---|---|---|
| D5 | L5 | ↑ → 10×14×12 | same | same |
| D4 | L4 | ↑ → 20×28×24 | same | same |
| D3 | L3 | ↑ → 40×56×48 | same | same |
| D2 | L2 | 40×56×48 (no upsample) | **↑ param-free** → 80×112×96 | **↑** → 80×112×96 |
| D1 | stem | 40×56×48 | 80×112×96 | **↑** → 160×224×192 |
| SVF head | — | 3-ch conv, zero-init | same | same |

- Trilinear upsampling: a custom reshape-based implementation (the rigid project found TF's linear upsampling slow), unit-tested against `scipy.ndimage.zoom`.
- No deep supervision.

### 4.6 Velocity fields and integration

At level k, for each image i:
1. v_i^(k) comes from the SVF head, in voxels of level k, in central space.
2. **Centering:** v_i^(k) ← v_i^(k) − mean_j v_j^(k), at every voxel.
3. **Integration:** scaling and squaring with T = 7 steps (configurable), at level k's resolution.
4. **Inverse:** exp(−v_i^(k)).

### 4.7 Multi-level composition

**Linear (level k ≥ 2):**
- Warp image i into the current central estimate (pull with A_i^(k−1)⁻¹).
- Predict a residual R_i.
- A_i^(k) = Center(R_i · A_i^(k−1)).

**Deformable:**
- Level k sees I_i ∘ A_i⁻¹ ∘ φ_i^(k−1) and predicts v_i^(k).
- Compose: φ_i^(k) = φ_i^(k−1) ∘ exp(v_i^(k)). Inverse: φ_i^(k)⁻¹ = exp(−v_i^(k)) ∘ φ_i^(k−1)⁻¹.
- Every v is stored, so the exact inverse is unrolled factor by factor.
- Properties:
  - invertibility is exact up to interpolation;
  - pairwise consistency is exact by construction;
  - centrality is approximate, the same class of approximation as MultiMorph's.
- Inverse consistency is tracked as a metric.

**Frozen levels:** prior levels run outside the GradientTape (`training=False`, dropout off). Composition with their outputs happens inside the tape.

**Level skipping** *(v2)*:
- Per training step, and per prior level ℓ, that level is **bypassed** with probability p_skip[ℓ]. Otherwise (probability 1 − p_skip[ℓ]) its output is used as normal.
- A bypassed level passes on its own input unchanged: for level 1 the identity, otherwise the output of the level before it.
- The decision is made once for the whole group and applies to both linear and deformable cascades.
- Defaults:
  - p_skip[level 1] ≈ 0 (rarely or never), because level 1 must learn the largest transforms;
  - p_skip[level 2] = 0.15 while training the full level, so the full level also learns to correct larger residuals.
- Inference never skips.

---

## 5. Training procedure

### 5.1 Coarse-to-fine for every task

| Level | Group size N *(v2)* | Notes |
|---|---|---|
| 1/4 | 32–48 | Train encoder, linear decoder and deformable decoder from scratch to convergence |
| 1/2 | 8–12 | Copy weights (adds parameter-free pools and upsamples only); recalibrate BN statistics with forward-only passes, then freeze them; 1/4 model frozen as the first cascade stage |
| Full | 2–4 | Same procedure; prior levels frozen; p_skip as in 4.7 |

N is finalized from measured memory. With N = 2–4 at full resolution, BN statistics are already frozen, and group-mean features come from very few images. That's an extra reason for the test-time N sweep in section 7.

### 5.2 Task schedule and optimizer

- Each step trains exactly one task. There are three `tf.function` train steps (rigid, affine, deformable), each traced once per level.
- **Task ratio** deformable : rigid : affine = **2 : 1 : 1** (hyperparameter). Rigid and affine share most of the linear decoder, so this roughly balances updates between the linear and deformable decoders.
- **One shared optimizer** *(v2)*: SGDW with `global_clipnorm` from the rigid project, with its LR warmup/cycle schedule and file-backed range curriculum. Log per-task losses to catch interference.
- Mixed precision (`mixed_float16`) and XLA are optional flags. A custom training loop replaces Keras `fit`.

---

## 6. Losses

### 6.1 Linear tasks

- **Barycentric:** L_center = mean_i tr(ΔA_i S ΔA_iᵀ), ΔA_i = A_i − A_i*.
  - S = E[x xᵀ] over template brain voxels (homogeneous mm coordinates, origin at the rotation center), precomputed once.
  - The loss is the mean squared voxel displacement in mm²; RMS mm is reported.
- **Pairwise:** L_pair = mean over sampled ordered pairs of tr(ΔP_ij S ΔP_ijᵀ), with ΔP_ij = A_j⁻¹ A_i − T_j T_i⁻¹. Pair fraction p_pair (pairs are cheap).
- **Total:** L_lin = L_center + λ_pair · L_pair, applied to the final composed and re-centered A_i at the current level.
- **Defaults** *(v3)*: λ_pair = 1, p_pair = 0.1. Tune later.

### 6.2 Deformable task *(v2)*

Labels: one-hot over the merged 19-channel set (3.4), warped with Ψ_i using linear interpolation of the one-hot channels.

- **Template soft Dice:**
  - The template is computed **once per batch from all N images**, including each image itself, as P̄ = mean_i onehot(L_i) ∘ Ψ_i.
  - Each moved label map is compared with it: L_tmpl = mean_i (1 − softDice(onehot(L_i) ∘ Ψ_i, P̄)), averaged over foreground labels.
  - **Gradients flow through the template** (no stop-gradient).
- **Pairwise soft Dice** over sampled ordered pairs (i, j): L_pdice = mean (1 − softDice(onehot(L_i) ∘ Ψ_i ∘ Ψ_j⁻¹, onehot(L_j))). Pair fraction p_pdice.
- **Smoothness on displacements** *(v2)*:
  - Penalize the deformable displacement u_i = φ_i − id (the composite deformable part, excluding the affine) at the current level. This is the same field the Jacobian determinant is computed on.
  - Options: first-order ‖∇u‖² (**default**) or second-order (bending energy).
  - Weight λ_smooth = 0.5 to start, tuned later.
- **Total:** L_def = L_tmpl + λ_pdice · L_pdice + λ_smooth · L_smooth. Defaults *(v3)*: λ_pdice = 1, p_pdice = 0.1, λ_smooth = 0.5.
- **Deformable-batch inputs** *(v3, decided)*: LUMIR's existing affine alignment. In deformable steps A_i = I and Ψ_i = φ_i. Running the linear heads first would add compute for no clear gain during training.

---

## 7. Metrics and validation

| Task | Metrics |
|---|---|
| Rigid / affine | RMS displacement error (mm), rotation error (deg), translation error (mm): centered and all pairs. Cycle error over triplets (≈ 0 by construction; a numerics check) |
| Deformable | Dice (template and pairwise) on the merged set; full-label Dice at evaluation; % negative Jacobian determinant; inverse consistency ‖Ψ_i ∘ Ψ_i⁻¹ − id‖ (mm); centrality ‖mean_i v_i‖ |
| All | Per-task loss curves (task interference), step time, peak GPU memory |

Validation uses LUMIR val groups of fixed composition. Later: OASIS-2 (longitudinal rigid/affine, including a **test-time N sweep** of 4, 8, 16, 32) and Mindboggle-101 (deformable Dice with manual labels).

---

## 8. Inference

1. Linear cascade (1/4 → 1/2 → full): A_i.
2. Resample into central affine space: I_i ∘ A_i⁻¹.
3. Deformable cascade: all v_i^(k) and φ_i.
4. Outputs: A_i, all v_i^(k) (for the exact inverse), warped images, pairwise maps on demand.
5. **Large N:** the config B encoder runs in chunks with cached features. For the decoders, stream the group means layer by layer (accumulate the sum over chunks, then broadcast) to keep everything on the GPU. This is an implementation task, attempted if the engineering is feasible.

---

## 9. Memory estimates (to measure, not trust)

| Item | Estimate |
|---|---|
| Volume, full res | 6.88 M voxels: 13.8 MB fp16, 27.5 MB as 1-ch float32 |
| Encoder activations, full res, single conv | ~1–1.5 GB per image |
| Deformable decoder, full res, half widths, single conv | ~0.5–1 GB per image |
| Scaling and squaring, full res | ~0.5–1 GB per image (forward plus inverse) |
| One-hot labels (19 ch) | ~260 MB per image fp16, plus warped copies |
| **Full-res deformable step** | Plausibly 10–20 GB at N = 2–4. Gradient checkpointing (`tf.recompute_grad`) is the fallback |

---

## 10. Repository layout and reuse

| Module | Purpose | Starts from |
|---|---|---|
| `main.py` | Env flags, seeds, build data and model, run the loop | rigid `main2.py` |
| `runtime/args.py` | Config: levels, N per level, task ratio, ranges, loss weights, config A/B, block type, p_skip per level, residual-affine flag | rigid `args.py` (pattern) |
| `runtime/run.py` | Custom loop: task schedule, three train steps, validation, hooks | groupwise `run.py` skeleton plus rigid `train_step` |
| `runtime/losses.py` | Barycentric and pairwise linear losses, template and pairwise Dice, displacement smoothness | new; Dice and gradient losses from groupwise `losses.py` |
| `runtime/metrics.py` | Section 7 metrics | new |
| `runtime/optimizers.py`, `schedulers.py` | SGDW, LR schedules | rigid, near as-is |
| `runtime/checkpoint.py` | Per-level checkpoint managers | rigid plus groupwise |
| `runtime/logging.py` | dllogger plus TensorBoard | groupwise plus rigid |
| `data_loading/dali_loader.py` | Rigid, affine and deformable pipelines | rigid `dali_loader.py` |
| `data_loading/data_module.py` | Per-level LUMIR file lists, the three datasets, validation | rigid `data_module.py` |
| `data_loading/curriculum.py` | Range curriculum | rigid, as-is |
| `models/layers.py` | Group block, residual block, BN-with-separate-affine, trilinear up/down, `transform_image` | rigid `layers.py`; groupwise `group_conv_3d` (concat form) |
| `models/transforms.py` | 10D rotation, R·K affine, `logm_real`, `GroupCentering`, barycenter targets, DALI ↔ TF conversion, flips | `scripts/*`; rigid `rotation_representations.py` |
| `models/diffeo.py` | SVF centering, scaling and squaring, inverse, composition, upsampling | new |
| `models/group_net.py` | Encoder, linear decoder, deformable decoder, level model, cascade with skip | rigid `reg_net.py` orchestration (rewritten) |
| `preprocessing/` (existing) | Add per-level `.npy` export and the `train18` merged-label LUT | existing `export.py`, `luts.py` |
| `tests/` | Unit tests against numpy references | new |

Runtime environment: `tf10_nv.sif` (Python 3.8, TF 2.10, DALI 1.18). Available: `tensorflow_addons`, `horovod`, `dllogger`, `tqdm`, `pandas`, `sklearn`, `nv_norms`. Missing: `einops`.

---

## 11. Milestones and verification

| # | Milestone | Verification |
|---|---|---|
| 0 | `export-npy`: per-level files (float32 block-average images, merge-then-mode `train18` labels), contrast filter (T1w only), float32 default | Shapes 40/80/160 grids; world-center consistency across levels; label counts per level; Jay runs the export |
| 1 | `models/transforms.py`, conventions, `transform_image` | Numpy-reference unit tests; centering invariance; pairwise preservation; T/T⁻¹ round trip < 0.5 voxel; F T F matches a flipped-image warp |
| 2 | Rigid DALI pipeline at 1/4 | Re-warp check (NCC > 0.99); pixel-center test; float16 input to `warp_affine` |
| 3 | Group block, encoder (with stem), linear decoder, rigid head, centering (configs A and B) | Permutation equivariance (1e-5); non-None gradient on every kernel |
| 4 | Linear losses, custom loop, logging, checkpoint (rigid, 1/4) | Overfit one fixed group to < 1 mm; short real run (debug job approved by Jay) |
| 5 | Affine pipeline and scale/shear branch; task schedule | No retracing across tasks; affine overfit; rigid metrics not degraded |
| 6 | Deformable pipeline, U-Net decoder, `diffeo.py`, deformable losses (1/4) | Constant v gives an exact translation; exp(v) ∘ exp(−v) ≈ id; mean v = 0; overfit one group (Dice ↑, folding < 0.1%) |
| 7 | Three-task schedule at 1/4 | All per-task losses decrease; no collapse |
| 8 | 1/2 level: weight copy, BN recalibration, frozen cascade, skip | Linear composition test; unrolled inverse over 2 levels; 1/2 error < 1/4 error |
| 9 | Full level, mixed precision, checkpointing, memory profiling | Peak memory per N logged; full-res deformable step fits on 48 GB |

---

## 12. Open questions (remaining)

All v2 questions are resolved *(v3)*:
- rigid translation ±15 mm;
- 18-label set;
- deformable inputs use LUMIR's existing alignment;
- pair defaults λ = 1, p = 0.1.

Deferred:
1. Same-subject training.
2. The pose-shortcut ablation: configs A vs B, group blocks on vs off, evaluated on same-subject groups.
3. A richer label set for cortical Dice.
4. A larger rigid translation range.

New questions will be added here as implementation surfaces them.
