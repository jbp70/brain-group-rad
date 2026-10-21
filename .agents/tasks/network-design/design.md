# Unified groupwise rigid / affine / deformable registration: design document

Status: v3.10 (2026-10-21). v3.10: optional per-task gradient normalisation `--task-grad-norm` / `--task-grad-norm-min` (section 5.2; default off, bit-identical). Previous: v3.9 (2026-10-20). v3.9: optional LNCC intensity loss for the deformable task (template term vs a leave-one-out intensity template and a pairwise term on the pairwise-Dice pairs, label-ROI masked, squared LNCC, window 5 / 7 / 9 at levels 4 / 2 / 1; weights default 0 so existing runs are bit-identical), `--lambda-dice-template` for an LNCC-only ablation, LNCC validation metrics (sections 6.2, 7, 9). Previous: v3.8 (2026-10-20). v3.8 (milestone 7): partial warm start of a deform model from a linear-only checkpoint, per-task loss weights and encoder grad-norm diagnostics with the measured loss-balance recommendation (keep `--loss-scale-mm2 1`: the global clip already balances the applied encoder gradients), the warm-started three-task recipe and per-task validation group sizes (sections 5.2, 6, 9, 11). Previous: v3.7 (2026-10-20). Incorporates Jay's answers to the v1 and v2 open questions. Decisions that changed since v1 are marked *(v2)*, *(v3)* or *(v3.x)*. v3.4: the dense path uses LayerNorm by default (section 4.4). v3.5: opt-in augmentation curriculum and shared global transform G (section 3.2). v3.6: mixed rigid + affine task schedule with one shared optimizer, the combined best metric, affine metrics (sections 5.2, 7), the G angle sign, G warm-up and warm start (section 3.2). v3.7 (milestone 6): the deformable decoder as implemented (4.5: stage table, widths (128, 96, 64, 32, 16), pre-norm-act zero-init SVF head), the diffeomorphic conventions (4.6: voxel units, pull warps, border vs zero fill, composition order, inverse via −v, half-pixel ×2 upsampling, interior Jacobian), the deformable losses as implemented (6.2: masked soft Dice over present labels, forward-difference smoothness, fp32 one-hot), per-task best files (5.2) and measured level-4 memory (9). Remaining items are in section 12.

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
| TF32 | **Off** in every process (`models.precision.configure_tf_precision()` at startup). On Ampere GPUs (A100, A6000, A40), TF 2.10 runs float32 matmul/einsum in TensorFloat-32 by default, which corrupted the grid and 4×4 matrix math by ~1e-3 on glacier. Convolutions stay fast through `mixed_float16` *(v3.1, milestone 2)* |
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

**Curriculum and global transform** *(v3.5)*. Both are opt-in (`--curriculum`, `--global-transform`, default off). With both off nothing below exists and the pipelines run with constant ranges as before.

- **Control file**, one per pipeline: `<run_dir>/control/<task>.npy`, float32 `[rot_deg, trans_mm, log_scale, shear, g_axis_x, g_axis_y, g_axis_z, g_angle_deg, g_tx_mm, g_ty_mm, g_tz_mm, version]` (TF axis order, mm; defined once in `data_loading/params.py`). A second `fn.readers.numpy` reads it every batch, following the reference project's curriculum scale reader: one file, no shuffle, `dont_use_mmap=True`, `read_ahead=False` (so the file is re-read, not cached), and `pad_last_batch=True` (so every sample of a batch sees the same record). No `external_source`. The trainer rewrites it atomically (temp file + `os.replace`) at the start of each step, only when values change, and increments `version`. The data module writes the step-0 file before the pipelines start. Because of the prefetch, a rewrite reaches the batches after about one batch. The emitted matrices are always the truth, so the targets are correct regardless.
- **Curriculum:** the per-image half-widths are U(−1, 1) × control value. They ramp linearly from `--<task>-*-start` to the `--<task>-*` values (which become the end values) over `--curriculum-ramp-steps` (default 2000), then stay constant. Start defaults: rigid 15° / 5 mm (scale/shear nuisance not ramped); affine 15° / 5 mm / log 1.03 / 0.01.
- **Global transform G:** one rigid G per group (all N images of a batch), about the image center. Rotation is U(−`--global-rot-max-deg`, +) (default 45°) about an axis uniform on the sphere; translation is U(±`--global-trans-max-mm`) mm per axis (default 10). It uses a fixed range (no curriculum) and is resampled every `--global-resample-every` steps (default 5), deterministic per (seed, step // K).
- **Composition:** T_i = G · Rt_i · S_i, with G applied last (template side) in one combined matrix, so there is still one interpolation. Rigid target: G · Rt_i (the network undoes G and Rt_i, not the nuisance S_i). Affine target: the full T_i. Flips are still F T F on T and target. The pipeline also emits G_dali `[N,3,4]` and `control_version` `[N]`. The trainer logs `g_mixed_batch`, the control values and the version lag.
- **Why G:** with independent per-image poses the group barycenter stays close to the template (MNI) pose, so "predict back to MNI" is nearly the barycentric answer. A shared G moves the barycenter (μ* → μ* G⁻¹), so the central space is a random large pose. Note: **G does not remove the per-image pose shortcut.** A network can still estimate each image's absolute pose and center the group. G only shifts the training distribution towards large shared poses, in which that pose estimate has to cover G's range as well.
- **G angle sign** *(v3.6)*: the file stores the axis in TF order; DALI gets the reversed axis. Reversing x and z is a reflection P, and P R(P a, θ) P = R(a, −θ), so DALI receives −angle (`params.G_DALI_ANGLE_SIGN`). The emitted G is then R(axis, +angle) in TF mm, i.e. `global_matrix_mm` of the file's values, as logged and as used by `val_<task>_g`. Glacier check (i) is strict on the signed axis and angle.
- **G from step 0 collapsed on real data; enable G after the per-image task is learned (start step / ramp, or warm start)** *(v3.6)*. A level-4 rigid run with the curriculum and G on from step 0 (±45° / ±10 mm, every 5 steps) reached the identity fixed point (relative pre-head feature spread 0.071 → 3e-5 by step 400, val_rigid stuck at 26.5 mm), while Adam runs without G learned at ±15° and ±45°. Options: `--global-start-step S` (G = identity before S) and `--global-ramp-steps R` (G's angle and translation half-widths ramp linearly from 0 at S to the maxima at S + R; still deterministic per (seed, step // K); logged as `train_<task>/control_g_rot_max_deg`), or `--init-weights` (load only the model weights of a finished run, same ModelConfig, fresh optimizer at step 0).
- **Validation:** `val_<task>` stays G = identity (the best checkpoint stays on `val_<task>/pair_rms_mm`, or the combined metric of section 5.2). With `--global-transform` there is also `val_<task>_g`: the same groups and per-image transforms plus one fixed seeded G per group from the same range, rendered TF-side. It writes its own PNGs (`<task>_g_step*.png`).

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
4. **Deformable labels:** one-hot over the merged label set (3.4), float32 *(v3.7: 8.2 MB per image at 1/4, so fp16 brings nothing there; revisit at full resolution)*. `DeformPostprocess` flips images and labels with the same F (no L/R label swap, train18 is L/R-merged) and samples the pairs.

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
3. Dense layers (2048, 512), then **standard dropout p = 0.15** *(v2)*. There's no recurrent dropout, because prior levels are frozen.
   - **Group concat in dense layers** *(v3.2)*: a dense group block is norm → act → concat(hᵢ, mean over the N group members of h) → Dense. That is W_a hᵢ + W_b m + b, i.e. a per-group bias.
   - **Default: no group concat at the first dense layer** (flatten 5,040 → 2048). There it would double the largest weight matrix (10.3 M → 20.6 M parameters). The 1×1 reductions before it are group blocks, so the flattened vector already carries group context.
   - The 2048 → 512 layer **is** a group block (+1.05 M).
   - Total linear-decoder overhead ≈ 1.1 M instead of 11.4 M.
   - Flag `dense_group_concat_first` (default false) enables it at the first dense layer too, for ablations. Config B is unchanged in meaning: all of these layers are after the bottleneck.
   - **Pre-head norm → act** *(v3.3, milestone 3; confirmed by Jay)*: norm → activation after the 512 layer, before dropout and the heads. With pre-activation blocks the 512 output is otherwise linear, so the heads would be a linear map of a linear map. Its norm follows `dense_norm` *(v3.4)*.
   - **Dense-path norm = LayerNorm by default** *(v3.4, milestone 4; decided by Jay)*: the dense blocks (inputs of width 5,040, 2,048 and 512) and the pre-head norm-act use a per-vector LayerNorm (statistics over the features of each image, float32, trainable γ/β). `dense_norm='none'` (activation only, like the rigid project's dense layers) is the alternative. `'batch'` is kept for ablations only. With one group per batch, BatchNorm on these vectors normalizes over the N group members, so in training mode it is group-relative (it subtracts the group's mean and divides by its spread). The eval-mode moving statistics mix between-group and within-group variation and can't reproduce that. On the milestone-4 overfit group, training mode reached 0.36 mm while eval mode stayed at ~20 mm. The conv path (encoder and 1×1 reductions) keeps BatchNorm.
4. **Heads** *(v2: shared rotation branch plus an extra scale/shear branch)*:
   - **Rotation + translation branch (shared by rigid and affine):** 10D rotation representation (validated at ±45° in the rigid project) plus t in mm. 13 outputs. Zero kernel with identity-encoding bias.
   - **Scale/shear branch (affine only):** 3 log-scales and 3 shears, giving K (upper-triangular, diagonal exp(s)). 6 outputs, zero-initialized, so K = I at start.
   - Rigid: A = [R | t]. Affine: A = [R · K | t]. This covers every orientation-preserving affine (QR factorization), is always invertible, and keeps the matrix log in centering well-defined.
5. **Centering layer (`GroupCentering`):** an unrolled 4-iteration group barycenter μ of the raw predictions (`logm_real` with a custom gradient, plus `tf.linalg.expm`), then A_i = μ⁻¹ A_i_raw. Gradients flow through μ. Verified in `scripts/check_tf_group_centering.py`.

### 4.5 Deformable decoder (U-Net)

- **One conv per stage** *(v2)*: trilinear ×2 upsample, concat the encoder skip, one block. These are group blocks (pre-activation norm → act → concat(group mean) → conv, 3×3×3, no residual) in **both** configs A and B.
- **Decoder widths** *(v3.7, implemented)*: half the matching encoder widths, `ModelConfig.decoder_widths = None` → (f4, f3, f2, f1, f0) / 2 = **(128, 96, 64, 32, 16)** for stages D5 … D1 with the default encoder filters; any 5 positive widths via `--decoder-widths`. The decoder exists only with `ModelConfig.deform_decoder = True` (default False, so linear-only models and checkpoints are unchanged; `main.py --deform-decoder auto` turns it on when `deform` is in the task ratio).
- **Parameter-free trilinear ×2 upsamples** are inserted at finer levels exactly where the encoder inserted pools, so every level has the same trainable decoder (verified: identical variable names and shapes at levels 4, 2, 1; `copy_weights_from` copies the decoder).
- **Parameters** (default widths): D5 4.43 M, D4 1.66 M, D3 0.77 M, D2 0.22 M, D1 0.055 M, SVF head 1.3 k; decoder 7.14 M (of which 3.57 M are group-mean input channels); whole model with encoder + linear decoder 29.6 M.
- **SVF head** *(v3.7)*: NormAct (the configured norm, then the activation) → Conv3D(3, 3×3×3, kernel and bias **zero-initialized**, float32 even under mixed precision). At initialization v = 0, so φ = id. `call` returns v_raw and the D1 features (used by the collapse diagnostic).
- **Gradient checkpointing** (constructor flag, not part of `ModelConfig`): `--gradient-checkpointing none|integrator|decoder|both` wraps the integrator and/or the decoder in `tf.recompute_grad` (gradients verified equal to 1e-4). Recomputing the decoder in training mode applies its BN moving-statistic updates twice per step. Default none.

| Stage | Skip | 1/4 level | 1/2 level | Full level |
|---|---|---|---|---|
| D5 | L5 | ↑ → 10×14×12 | same | same |
| D4 | L4 | ↑ → 20×28×24 | same | same |
| D3 | L3 | ↑ → 40×56×48 | same | same |
| D2 | L2 | 40×56×48 (no upsample) | **↑ param-free** → 80×112×96 | **↑** → 80×112×96 |
| D1 | stem | 40×56×48 | 80×112×96 | **↑** → 160×224×192 |
| SVF head | — | 3-ch conv, zero-init | same | same |

- Trilinear upsampling: a custom reshape-based implementation (`models.layers.upsample_trilinear2x`, layer `TrilinearUpsample2x`; no `tf.image`), with **half-pixel** semantics: coarse voxel q sits at fine coordinate 2q + 0.5, so out[2k] = 0.75 x[k] + 0.25 x[k−1], out[2k+1] = 0.75 x[k] + 0.25 x[k+1], edge-replicated. Unit-tested against `scipy.ndimage.map_coordinates(order=1, mode='nearest')` at (p − 0.5)/2 and `zoom(grid_mode=True, mode='nearest')` (not align-corners).
- No deep supervision.

### 4.6 Velocity fields and integration

At level k, for each image i:
1. v_i^(k) comes from the SVF head, in voxels of level k, in central space.
2. **Centering:** v_i^(k) ← v_i^(k) − mean_j v_j^(k), at every voxel.
3. **Integration:** scaling and squaring with T = 7 steps (configurable, `--integration-steps`), at level k's resolution: u ← v / 2^T, then T times u ← compose(u, u). T = 0 returns v.
4. **Inverse:** exp(−v_i^(k)), never a numerical inversion.

Conventions as implemented *(v3.7, `models/diffeo.py`)*:

| Item | Convention |
|---|---|
| Units | v, u in **voxel-index units of their level** (mm = voxels × level spacing). Fields act on voxel indices, so the grid center does not enter them. |
| Pull warp | warp(vol, u)(x) = vol(x + u(x)). φ(x) = x + u(x) maps central-space points to image-i points; the warped image is I_i ∘ φ_i. |
| Composition | compose(u_a, u_b) = displacement of φ_a ∘ φ_b (φ_b first) = u_b + warp_field(u_a, u_b). |
| Boundaries | Fields are resampled with **border** clamping (edge replication; a constant field stays exactly constant). Images and one-hot labels use **zero** fill (the `transform_image` soft fall-off over the last voxel). Labels in nearest mode keep their integer dtype. |
| Pairwise | image j → image i: φ_i ∘ φ_j⁻¹, displacement compose(u_i, u_inv_j); one label interpolation per pair. |
| Cascade (milestone 8) | φ^(k) = up(φ^(k−1)) ∘ exp(v^(k)) → compose(up(u_prev), u_k); inverse compose(u_inv_k, up(u_inv_prev)); up = 2 × trilinear ×2 (half-pixel, so fine p ↔ coarse (p − 0.5)/2 maps to p + 2u). |
| Jacobian | det(I + ∇u) by central differences on **interior voxels only** ([N, X−2, Y−2, Z−2]); folding fraction = mean(J ≤ 0). |
| Dtypes | Grids, fields, integration and losses in float32 (float64 kept if given); float16 inputs are cast to float32; never float16 grids. |

Measured (level-4 grid 40×56×48, smooth random v with max 2 voxels, T = 7, 3-voxel interior margin): |exp(v) ∘ exp(−v) − id| max 0.010 / mean 0.0019 voxels, both orders; a constant v gives the exact translation (1e-6).

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
- **One shared optimizer** *(v2)*: SGDW with `global_clipnorm` from the rigid project, with its LR warmup/cycle schedule and file-backed range curriculum. Log per-task losses to catch interference. *(Milestone 4: the default is Adam 3e-4, since SGDW collapsed on real data.)*
- **Mixed linear tasks** *(v3.6, milestone 5)*: `--task-ratio rigid:1,affine:1`. The task of step s is a stateless seeded draw from (seed, s), so a resumed run gets the same sequence. Each task has its own `tf.function`, traced once over the whole run (flips, curriculum and G on or off). The single optimizer is built on **all** trainable variables before the first step, so the affine-only scale/shear head has its Adam slots even when the run starts with rigid steps; rigid steps drop that head's None gradients at trace time, and the first affine step creates no variable and retraces nothing.
- **Best checkpoint with several tasks** *(v3.6)*: `--best-metric` defaults to `mean_pair_rms_mm` (logged as `val/mean_pair_rms_mm`), the mean of `val_<task>/pair_rms_mm` over the trained tasks (the G = I sets only), when more than one task is trained, and to `val_<task>/pair_rms_mm` for a single task. Any explicit key still works.
- **Best files with the deformable task** *(v3.7)*: `val/mean_pair_rms_mm` averages the **linear** tasks only. Deform only: `ckpt/best` on `val_deform/pair_dice_loss` (= 1 − mean pairwise hard Dice, lower is better). Deform + linear: `ckpt/best` keeps the linear criterion (single linear task's pair RMS, or the mean over rigid and affine) and a second file `ckpt/best_deform` tracks `val_deform/pair_dice_loss` (its own `best_metric_deform` variable, present only in such runs, so older checkpoints restore unchanged).
- **Deform step** *(v3.7)*: `--task-ratio` accepts `deform` (e.g. `rigid:1,affine:1,deform:2`); one more `tf.function` traced once. Deform steps give None gradients to the whole linear decoder, linear steps to the whole deform decoder (filtered at trace time); the one optimizer covers all variables. Curriculum, G and control files stay linear-only. `--deform-group-size` (default = `--group-size`) sets the deform group size.
- **Partial warm start** *(v3.8, milestone 7)*: `--init-weights` (model variables only; the optimizer, step, best values and RNGs are never read, and a fresh Adam is built afterwards) now accepts a source whose ModelConfig differs only in the deform-decoder fields `deform_decoder`, `decoder_widths`, `integration_steps` (`runtime.checkpoint.DECODER_ONLY_FIELDS`). A model with the source config is built and restored strictly, then the encoder, linear decoder + heads and centering (trainable variables plus BN / LayerNorm state) are copied after a count / shape / scope-local-name check; the target's deform decoder keeps its fresh init (a source deform decoder is ignored in the reverse direction). A coverage guard raises if a model part is neither restored nor the fresh decoder. Any other config difference still raises naming the fields; identical configs keep the strict whole-model restore. The result (mode full / partial, restored / fresh parts with variable counts, config diff) goes to config.json `init_weights`, an `init_weights` event in metrics.jsonl and stdout.
- **Loss balance between tasks** *(v3.8)*: knobs `--loss-scale-mm2 S` (linear objective / S) and `--task-loss-weights rigid:a,affine:b,deform:c` (objective × w; default all 1). Linear objective = w · L_lin / S, deform objective = w · L_def; logged losses stay unscaled. Each factor is skipped at trace time when it is 1, so linear-only runs are bit-identical (golden test `tests/fixtures/m7_linear_golden.json`). Diagnostics per task (window means): `train_<task>/diag_enc_grad_norm_raw` (encoder-only gradient norm of the unscaled loss), `train_<task>/diag_enc_grad_norm_eff` (encoder norm actually applied: after S, w and the global-clip factor min(1, 12 / grad_norm)), `train_<task>/grad_norm` (the task's pre-clip total norm).
- **Measured balance** *(v3.8, local probe, synthetic phantom, narrow widths, linear N 4 / deform N 2, warm start from a 300-step linear run (train-mode center RMS ≈ 6.5 mm) + 100 three-task steps; medians of 40 measurement steps at lr 1e-12, `.agents/tasks/milestone-7/grad_balance.json`)*: raw encoder norms rigid 306 / affine 257 (mm²) vs deform 0.144 (≈ 2000×). But every linear step is clipped by the global clip norm 12 (task grad_norm 1.3–1.5 × 10⁴ at S = 1, still 128–148 at S = 100), so the applied encoder norms are rigid 0.24 / affine 0.20 vs deform 0.14: ratio linear/deform **1.66 / 1.40 at S = 1, identical at S = 10 and 100, 1.66 / 1.25 at S = 1000** (where clipping stops). The clip, not the loss units, sets what Adam's second moment sees on the shared encoder. **Recommendation: `--loss-scale-mm2 1`, no task weights** (ratio inside the [0.1, 10] target). Confirm on real data from the smoke run's `enc_grad_ratio` line (if outside [0.1, 10]: unclipped linear steps → S × ratio / 3; still clipped → `--task-loss-weights deform:<ratio/3 rounded to 2^k>`). Caveat: right after a warm start the SVF head kernel is zero, so the deform task sends **zero** gradient to the encoder (3.8 × 10⁻¹⁰ measured) until the head grows; only the linear tasks move the encoder at first (after 100 steps the deform encoder norm was 0.14–0.31).
- **Three-task recipe at level 4** *(v3.8, `slurm/m7_three_task.sh`)*: `--init-weights` the m5 mixed checkpoint (`results/m5_mixed_long_curr_dG_20261019-205526/ckpt/best`, partial restore: config diff exactly `deform_decoder`), `--task-ratio rigid:1,affine:1,deform:2`, `--group-size 32`, `--deform-group-size 16`, `--loss-scale-mm2 1`, G at 45° / 10 mm from step 0 (`--global-start-step 0 --global-ramp-steps 0`), **curriculum off** (milestone 5: warm start + G from step 0 works; m5's curriculum already ended at the full ranges), warmup 500 steps (fresh Adam moments on pretrained weights), Adam 3e-4 cosine, 12000 steps, val every 250, PNGs on.
- **Per-task gradient normalisation** *(v3.10, `--task-grad-norm T --task-grad-norm-min m`, default T = 0 = off)*: why: linear raw total norms are 1e3–6e3 (mm², every step clipped to 12), deform totals 0.03–0.5 (never clipped), so unweighted the shared Adam saw deform encoder _eff ~0.05 vs ~3.4 for rigid; `--task-loss-weights deform:64` made every deform step clipped too (de facto normalisation) and raised val deform template / pair Dice at step 8000 from 0.704 / 0.608 to 0.723 / 0.638 with linear errors unchanged. Rule, per step after the AMP unscale and before `apply_gradients`: g = global norm of the task's raw gradient; if g is finite and g ≥ m: grads × T / max(g, ε) (ε = 1e-12, `TASK_GRAD_NORM_EPS`); if g < m or non-finite: scale 1 (raw gradient; non-finite steps are skipped exactly as before). THEN × the task weight w, which is no longer in the objective, so with the flag on the weights are ratios of applied norms (T · w); `--loss-scale-mm2` cancels except for which raw norm the gate sees. Default **m = 1e-3**: healthy deform totals are 30–500× above it and linear totals ≥ 1e3, so normally nothing is gated; it equals `collapse_grad_norm`, so exactly the steps the collapse detector calls "zero gradient" (group collapse, degenerate batches) are not amplified, and the maximum gain is T / m = 1.2e4 at T 12. Measured: a collapsed rigid group's global norm underflows to 0 while the gradients still carry ~1e-19 residues; with m = 0 the ε path multiplies them by 1.2e13 (applied norm ~4e-6, finite). Zero-head deform steps right after a warm start have total norms 0.03–0.5 (not gated): they are scaled ×24–400, which mostly grows the SVF head, while their ~1e-10 encoder part stays negligible. The gate is a discontinuity (scale jumps 1 → T / m as g crosses m); it only matters if a task hovers at the gate, which is logged. Alternative not implemented: capped gain min(T / g, G_max) (continuous). The global clip stays: with T ≤ clipnorm it is a no-op up to 1 ulp, but weights > 1 get clipped back, so use `TASK_LOSS_WEIGHTS=none` (or T · max w ≤ clipnorm). Logs: `train_<task>/grad_norm` stays the raw pre-normalisation norm (unweighted with the flag on), `diag_enc_grad_norm_eff` = the applied encoder norm (× scale × w × clip factor), new `train_<task>/grad_norm_scale`, `grad_norm_gated_frac` (window mean = gated fraction), `grad_norm_normed`. Off adds no op (trace-time Python check; goldens `m7_linear_golden.json` / `lncc_deform_golden.json` max abs diff 0). Equivalence (`TestTaskGradNormClipEquivalence`, 8 three-task steps, op determinism): T 12 without clip vs clip 12 with all weights 2^14 (every raw norm > 12): bit-identical (max abs diff 0); with clip 12 kept: losses 9e-7 rel., variable sums ≤ 1.1e-3 rel. (1-ulp clip factors amplified by Adam on near-zero-gradient elements). Caveat: equal total norms do not mean equal encoder norms; in the local narrow synthetic run (T 12, 400 steps) the encoder share of the total norm was ~38 % for deform vs ~2 % for linear, so encoder _eff linear / deform fell to ~0.05–0.07 (unweighted: ~0.5–0.9); on real level-4 data deform:64 (≈ normalisation, all steps clipped) gave applied ratios rigid 1.32 / affine 0.72 (LNCC smoke, job 10494447), the expected regime for the cluster A/B. SLURM: `TASK_GRAD_NORM` / `TASK_GRAD_NORM_MIN` in `slurm/m7_three_task.sh` (empty = off).
- **Validation group sizes** *(v3.8)*: no new flag. Linear sets use min(`--group-size`, n_val) = 32 (2 groups for the 39 val cases), the deform set min(`--deform-group-size`, n_val) = 16 (3 groups); deform pairs capped at 64 per group (`--val-deform-max-pairs`). `--val-group-size` would override both, so the m7 scripts do not pass it. Best files unchanged: `ckpt/best` on `val/mean_pair_rms_mm`, `ckpt/best_deform` on `val_deform/pair_dice_loss`.
- **BN momentum** stays 0.99. The deform decoder's BN statistics start fresh, so eval-mode `val_deform` lags `val_deform_trainmode` early (local probe after 100 steps: eval template / pair Dice 0.872 / 0.753 vs train mode 0.947 / 0.898). Report the gap; BN recalibration is milestone 8.
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
- **Training objective** *(v3.8)*: w_task · L_lin / `loss_scale_mm2` (defaults 1 and 1); the logged terms stay in mm² / mm.

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
- **Total:** L_def = L_tmpl + λ_pdice · L_pdice + λ_smooth · L_smooth. Defaults *(v3)*: λ_pdice = 1, p_pdice = 0.1, λ_smooth = 0.5. *(v3.9: optional LNCC terms and a template-Dice weight, see "LNCC" below; defaults leave this total unchanged.)*

As implemented *(v3.7, `runtime/losses.py`)*:
- **Soft Dice** per image and channel (2Σab + ε) / (Σa + Σb + ε), ε = 1e-5, background excluded, then a **masked mean over present foreground channels** (present: Σa + Σb > 1e-3). Channels absent in both maps are excluded, so a group that contains only some labels is not inflated by empty channels. The same rule applies to the hard Dice metrics (NaN for absent labels, nanmean).
- **One-hot** float32, 19 channels (train18 + background), warped linearly with zero fill (`diffeo.warp`).
- **Template** P̄ = mean_i warped_i (no stop-gradient). **Pairs:** p_pdice of the ordered pairs (at least one), `warp(onehot_i, compose(u_i, u_inv_j))` vs onehot_j.
- **Smoothness** on u (not v): order 1 (default) = forward differences, (1/3) Σ_axes mean((u[x + e_a] − u[x])²) over images, voxels and components (VoxelMorph Grad-l2); order 2 (`--smooth-order 2`) = bending energy (central second differences with 2× cross terms, interior crop, mean).
- The objective is not rescaled (no mm² factor) except by the optional task weight *(v3.8: objective = w_deform · L_def, default 1; logged terms unweighted)*; mixed precision uses the same loss scaling as the linear steps.
- **Deformable-batch inputs** *(v3, decided)*: LUMIR's existing affine alignment. In deformable steps A_i = I and Ψ_i = φ_i. Running the linear heads first would add compute for no clear gain during training.

**LNCC (v3.9, `runtime/losses.py`, optional):** an intensity term next to the label Dice, for training on T1 images with labels (joint objective) and as preparation for label-free / synthetic-contrast training.
- **Formula:** local means over a k³ box (zero padded, divided by the true per-voxel window count, so border windows are normalised correctly): μ_a, μ_b, cov = ⟨ab⟩ − μ_a μ_b, var = max(⟨a²⟩ − μ², 0). **Squared** LNCC cc = cov² / (var_a var_b + ε), in [0, 1] (no sqrt: finite gradients on flat regions). ε = 1e-7 in intensity² units (local std product ≈ 3e-4 for [0, 1] intensities; bounds the gradient on flat regions). A signed variant cov / sqrt(var_a var_b + ε) exists for tests only.
- **Box filter:** separable `tf.nn.conv3d` with ones kernels [k,1,1], [1,k,1], [1,1,k] (15 vs 125 MACs per voxel at k = 5), the five quantities [a, b, a², b², ab] stacked along the batch axis; no Keras layer, so the model gains no variables (checkpoints and warm starts unchanged). Window counts are a numpy constant of the shape.
- **Window** `--ncc-window` (odd, ≥ 3), default per level 5 / 7 / 9 voxels at levels 4 / 2 / 1 (20 / 14 / 9 mm; 9 at 1 mm is the VoxelMorph / reference value, 9 at 4 mm would be nearly global on 40×56×48).
- **Template term:** W_i = warp(I_i, u_i) (linear, zero fill); **leave-one-out** template T₋ᵢ = (Σ_j W_j − W_i) / (N − 1), gradients through it (as the Dice template). The inclusive mean has a self term that gives a squared-LNCC floor ≈ 1/N even for unrelated images (0.25 at N = 4, 0.06 at N = 16) and rewards each image for staying put; LOO costs one subtraction. L_ncc_tmpl = mean_i (1 − cc̄_i).
- **Pairwise term:** on the **same sampled pairs** as the pairwise Dice, reusing the same `pairwise_displacement` (computed once, no extra integration): warp(I_i, disp_p) vs I_j. L_ncc_pair = mean_p (1 − cc̄_p).
- **ROI / masking:** the DALI intensity augmentation adds noise to the zero background, so a "nonzero voxel" mask (reference) selects almost the whole volume. Instead fg = Σ_{c≥1} onehot (zero fill gives 0 outside the FOV, unlike 1 − onehot₀), ROI = box_sum(fg) > 1e-3 (foreground dilated by k/2, no gradient). Template term: ROI of the warped one-hot template; pair term: max(fg(moved one-hot_i), fg(one-hot_j)). Per-image masked mean, then the mean over images. Out-of-FOV voxels enter only if the brain is within k/2 voxels of the border (finite either way).
- **Total:** L_def = λ_dt L_tmpl + λ_pdice L_pdice + λ_smooth L_smooth + λ_nt L_ncc_tmpl + λ_np L_ncc_pair. Defaults λ_dt = 1, **λ_nt = λ_np = 0** (terms not built: the default graph is bit-identical, golden test `tests/fixtures/lncc_deform_golden.json` under TF op determinism); recommended experiment 1 / 1 (the Dice terms' 1 : 1). `--lambda-dice-template 0 --lambda-pdice 0` with LNCC on is the LNCC-only ablation (soft Dice still logged); a deform run needs at least one similarity weight > 0.
- **Numerics:** float32 always (inputs cast, no float16, AMP-safe since no Keras layer). Logged `train_deform/ncc_tmpl`, `ncc_pair` (mean squared LNCC, higher is better) and `ncc_tmpl_loss`, `ncc_pair_loss` (only the active terms).
- **Reference** (`fast_nlcc_metric` / `Joint_NLCC_Grad_Loss` of the deformable-groupwise reference code): kept the squared form, ε = 1e-7, the 5-quantity single filter pass, the mean template with gradients; changed raw sums / k³ to true local means (border normalisation, ε meaning), separable functional convs instead of a dense 9³ Keras Conv3D, per-level window instead of a fixed 9, LOO template instead of the inclusive mean, label ROI instead of nonzero voxels (noise-augmented background), loss 1 − cc instead of −cc (O(1) and ≥ 0 like 1 − Dice), plus a pairwise term.
- **Gradient magnitude** (local probe, synthetic phantom, narrow N = 4, after 50 Dice-only steps; `.agents/tasks/lncc-loss/grad_probe.json`): total grad norm Dice-only 0.45 vs Dice+LNCC 1/1 1.25 (encoder 0.29 vs 0.69), each LNCC term ≈ 1.5–2× a Dice term; far below the global clip 12, so the deform step stays unclipped and its applied encoder gradient grows ~2.3× (still ≪ the clipped linear steps).

---

## 7. Metrics and validation

| Task | Metrics |
|---|---|
| Rigid / affine | RMS displacement error (mm), rotation error (deg), translation error (mm): centered and all pairs. Cycle error over triplets (≈ 0 by construction; a numerics check). Affine also *(v3.6)*: from the QR factors L = Q·U·D of predicted and true matrices, scale error 100·mean\|Δ log D\| (%), shear error 100·mean\|ΔU\| (%), and the rotation-factor error ∠(Q Q_trueᵀ) (deg) |
| Deformable | Dice (template and pairwise) on the merged set; full-label Dice at evaluation; % negative Jacobian determinant; inverse consistency ‖Ψ_i ∘ Ψ_i⁻¹ − id‖ (mm); centrality ‖mean_i v_i‖. *(v3.9)* `val_deform/ncc_template`, `ncc_pair` (squared LNCC with the training definition: LOO template, label ROI, level window; always computed, whatever the LNCC weights) and the cached identity baselines `ncc_template_identity`, `ncc_pair_identity` |
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

**Measured, level 4 deformable training step** *(v3.7, milestone 6)*: local GTX 1080 under WSL, TF logical-device limit 6000 MB, `TF_CUDNN_USE_AUTOTUNE=0`, workspace limit 512 MB, fp32, Adam, default losses (p_pdice 0.1 → 1 pair for N ≤ 4, 6 for N = 8), synthetic group; TF allocator peak of the measured steps (`tests/gpu/measure_deform_memory.py`, `.agents/tasks/milestone-6/memory.json`).

| Widths | Checkpointing | N = 2 | N = 3 | N = 4 | N = 8 | Step (GTX 1080) |
|---|---|---|---|---|---|---|
| narrow (4.0 M params) | none | 2.09 GB | 3.01 GB | 4.01 GB | OOM (also N = 6) | 0.71 / 0.95 / 1.08 s |
| narrow | both | | | 2.51 GB | 4.78 GB (N = 16 OOM) | 1.38 / 2.32 s |
| default (29.6 M params) | none | 4.44 GB | OOM | OOM | | 1.92 s |
| default | both | 3.91 GB | | OOM | | 2.29 s |

- Slopes: narrow ≈ 0.96 GB per image without checkpointing, ≈ 0.57 GB with `both` (−40 %, +30–115 % step time). Default widths ≈ 2.0 GB per image (one point, minus the 0.41 GB of weights and Adam slots); `both` saves only ≈ 12 % there because the encoder is not recomputed.
- **Estimate (not measured):** default widths at N = 32 need ≈ 32 × 2.0 GB + ~0.1 GB per sampled pair × 99 pairs ≈ 70+ GB, which does not fit 48 GB (RTX 8000) or 40 GB (A100). N = 16 (24 pairs) ≈ 35 GB is expected to fit 48 GB. Use `--deform-group-size 12-16` at level 4 (linear tasks can keep N = 32) or measure AMP / checkpointing on the cluster first (`measure_deform_memory.py` runs there unchanged, with memory growth).
- **Three-task run in one process** *(v3.8)*: the linear step at N = 32 (~34 GB measured in training, milestone 4) and the deform step at N = 16 (33.8 GB measured, SLURM job 10492943, RTX A6000) never run at the same time; one step's activations are freed before the next, so the peak is the larger of the two (~34 GB), not the sum, plus the resident weights and Adam slots for all parts (~0.44 GB). Fits the 48 GB RTX 8000 / A6000; fallback `DEFORM_N=12` (24.8 GB). The smoke run's `gpu_peak_mb` confirms it.
- **LNCC** *(v3.9, `.agents/tasks/lncc-loss/memory.json`, same local setup, 5 measured steps)*: LNCC 1/1 vs off changes the measured peak by −41 … +65 MB at narrow N = 2–4 and −49 MB at default N = 2 (within run-to-run noise; no measurable per-image slope), step time +6 / +8 / +21 % (narrow N = 2 / 3 / 4) and +3 % (default N = 2). The peak is the integration backward; the LNCC activations (≈ 3 tensors of [5N] volumes for the template term and of [5P] for the pairs, 0.43 MB per level-4 volume) are freed before it. **Estimate (not measured)** at default widths N = 16 (24 pairs): ≈ 0.1 GB template + ≈ 0.15 GB pairs stored, ≤ 0.5 GB in the worst case, on top of the 33.8 GB, so N = 16 still fits the 40 GB A100 and the 48 GB cards.

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
| 6 | Deformable pipeline, U-Net decoder, `diffeo.py`, deformable losses (1/4) | Constant v gives an exact translation; exp(v) ∘ exp(−v) ≈ id; mean v = 0; overfit one group (Dice ↑, folding < 0.1%). *(v3.7: implemented and verified locally: round trip max 0.010 voxel; synthetic overfit (narrow, N = 4) eval template / pair hard Dice 0.864 / 0.774 → 0.971 / 0.950 at step 306, 0 % folding; one trace per task with rigid:1,affine:1,deform:2. DALI `DeformPipeline` (check l) and real-data runs (`dryrun_deform`) are pending on glacier. Report: `.agents/tasks/milestone-6/report.md`.)* |
| 7 | Three-task schedule at 1/4 | All per-task losses decrease; no collapse. *(v3.8: implemented locally: partial warm start (linear-only → deform model), `--task-loss-weights`, per-task encoder grad-norm diagnostics, measured balance (keep S = 1), `slurm/m7_three_task.sh`; local three-task warm-start smoke with linear N 4 / deform N 2 passes `check_mixed_run.py` incl. both best files. Cluster smoke + long run pending. Report: `.agents/tasks/milestone-7/report.md`.)* |
| 8 | 1/2 level: weight copy, BN recalibration, frozen cascade, skip; **full label set with per-step label sampling** (below) | Linear composition test; unrolled inverse over 2 levels; 1/2 error < 1/4 error; label sampling: same subset across the group, core labels in every step, full-label validation Dice matches an unsampled reference |

**Milestone 8 addendum: full label set with per-step label sampling** *(added 2026-10-20, Jay)*. Goal: use the full SynthSeg label set for the deformable Dice at levels 2 and 1 without one-hot memory growing with the number of classes. Level 4 keeps `train18`, because many fine labels are only a few voxels at 4 mm.
- **Data (Jay runs the export):** the full label map remapped to contiguous ids (uint8), per level, next to the existing `labels_train18*`, plus a label table marking the **core** labels (e.g. ventricles, and other structures chosen by Jay). DALI is unchanged (labels are already warped nearest-neighbour).
- **Sampling:** each deform step draws one subset S = core ∪ K random non-core labels (seeded per step like the flips, CLI `--dice-labels full|train18`, `--dice-core-labels`, `--dice-sample-k`, default K about 8-12). The **same S for every image of the group**: the template Dice averages each channel over the group, so every image needs the same channels. Flips are group-wide, so no left/right swap is needed.
- **One-hot in TF after DALI** (in `DeformPostprocess`): build only the |S| binary maps `labels == id`, with no "rest" channel in the loss. Memory then scales with |S|, not the label count (level 1: about 520 MB per image for 19 float32 channels, estimated).
- **Weighting:** uniform sampling is unbiased for the all-label mean Dice in expectation. Core labels appear every step, so they get extra weight, which is intended. Optionally reweight non-core terms by (#non-core / K) for an exact all-label estimate.
- **Validation** always scores every label (hard Dice, argmax over all classes, computed in label chunks without gradients) so runs stay comparable. Report core-label and all-label means separately.
- **Checks:** gradient noise via the milestone-7 per-task encoder gradient diagnostics; the sampled loss averaged over many seeds should match the full-label loss on a fixed group.
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
3. A richer label set for cortical Dice: planned for milestone 8 as the full label set with per-step label sampling (section 11 addendum). Open: which labels are core, and K.
4. A larger rigid translation range.

New questions will be added here as implementation surfaces them.
