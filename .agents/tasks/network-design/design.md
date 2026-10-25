# Unified groupwise rigid / affine / deformable registration: design document

Status: v3.15 (2026-10-24). v3.15 (milestone-9 phase 2 design, `.agents/tasks/milestone-9-phase2/design-m9-phase2.md`): 3-level frozen cascade `FrozenChain` (levels 4 + 2 from the milestone-8 level-2 checkpoint `m8_level2_gradnorm/ckpt/best` step 6000, its stored `frozen_stage` = level 4, fingerprint-checked), level-1 trainable stage with the identical default ModelConfig (no different-width stage; the N = 5 fr50 route needs a full 4 → 2 → 1 retrain), skip of the last frozen stage (recipe 0), `val_<set>_frozen` (4 + 2) and `val_<set>_frozen4` baselines, `val/wall_s`; opt-in `--min-pairs n` (pairwise-loss minimum = group size, random N-cycle at k = N, same RNG draw); level-1 recipe and `slurm/m9_level1_cascade.sh` (sections 4.7, 6, 9, 11). v3.14 (2026-10-23). v3.14 (milestone-9 phase 1 implemented and measured locally, `.agents/tasks/milestone-9/report.md`): knobs `--gradient-checkpointing full`, `--lean-sampler`, `--lean-losses`, `--split-group-conv` and the width presets implemented (all off by default, goldens max |diff| 0); calibrated memory model fitted (45 points −3.7 … +2.1 %), local hold-out of 40 points with a one-sided gate (worst under-prediction −13 %, margin 0.15; 18 small level-4 AMP / split points over-predicted by > 20 %), inventory correction (under `full` both integrator blocks are recomputed concurrently), measurement tool one configuration per process (`--isolate`); provisional level-1 recommendation `f0d1` + lean sampler + split, AMP, `full`, T = 7, deform N = 4 at 35.0 GiB (40.2 with margin), linear N = 16; T = 6 accurate but not adopted (sections 4.5, 4.6, 9, 11). v3.13 (2026-10-22). v3.13 (milestone-9 phase-1 design, `.agents/tasks/milestone-9/design-m9-phase1.md`, design v5 after its fourth review: the lean sampler's backward accumulates its 8 corners sequentially (control-dependency chain on the accumulators, `tf.while_loop(parallel_iterations=1)` fallback; dy guard kept), its forward / backward transients are in the inventory as sub-phases Fw / Fb, level-1 `f0d1` N = 4 unchanged at 33.5 GiB but N ≥ 5 no longer fits with the margin, graph-mode memory test on a level-2-size field; design v4 after its third review: the lean sampler's backward recomputes its corner math from coords + a dy-dependent zero (the `tf.recompute_grad` trick) so Grappler cannot merge it with the forward, verified by a graph-mode memory test; timed skip-gradient term with a conservative bound; step times from the window-mean `metrics.jsonl` keys (per task only for cascade runs); design v3 after its second review: linear calibration cross-check with precision-split encoder coefficients, the build forward kept out of the peak and built at `min(N, 2)` (`main.py` with `--split-group-conv`), LNCC/`none`-phase sentence of section 9 corrected; revised with design v2 after its first review: the inventory includes the decoder-backward transient and reproduces the level-2 deform points within −5 … +1 % before fitting; selection = widest model with level-1 deform N ≥ `--n-target` 4 under a 15 % margin, expected `f0d1` at N = 4; tensor bound covers backward kernels; knob hold-out set; pinned retraining GPUs): level-1 memory root cause (sampler 8-corner tape in the 14 integrator compositions and the one-hot warps, full-res D1, encoder never recomputed); new knobs, all off by default and bit-identical when off: `--gradient-checkpointing full` (encoder + decoder + integrator as two blocks, BN moving statistics updated exactly once per step), `--lean-sampler`, `--lean-losses`, `--split-group-conv`; widths stay explicit `--filters` / `--decoder-widths` (named presets in `models/width_presets.py`, no new ModelConfig field); T stays one ModelConfig value at every cascade level (retraining uses one T); calibrated TF-free memory model `models/memory_model.py`; per-tensor-limit probe; level-1 measurement with a frozen-chain stand-in, a `--cascade false` point and the frozen-stage peak (`slurm/m9_memory.sh`); 3-level cascade, level-1 DALI / validation and the retraining sequence designed (not implemented). v3.12 (2026-10-21). v3.12 (milestone-8 design v5): BN recalibration is cascade-only (`--bn-recal-batches > 0` requires `--cascade-from`, else a parser error), runs once at step 0 of a fresh run and is never repeated on resume (statistics come from the checkpoint, the freeze flag from the saved config.json); non-finite recalibration batches consume their slot; measured level-2 validation memory (`--val-pass`) with the `--val-pair-chunk` fallback; C4 step cut on the log record's window-end step (sections 4.7, 5.1). Previous: v3.11 (2026-10-21). v3.11 (milestone 8 design, `.agents/tasks/milestone-8/design.md`): level-2 cascade with a frozen level-4 stage (`models/cascade.py`), weight copy, linear and deformable cascade equations as specified for implementation, level skipping with default 0 and the cascade-position naming, BN recalibration by exact per-layer averaging then freezing, level-2 memory plan and group-size rule, frozen stage stored in the checkpoint, label sampling deferred unless Jay chooses otherwise, exporter change for synthseg at level 2; revised after the milestone-8 design review: resume provenance, validation recompute, export via the existing `synthseg` label set (sections 4.7, 5.1, 9, 11, 12); second revision (milestone-8 design v3): level-2 template mask in the DALI verify script, `SCRATCH_ROOT` override and stub test for the m8 SLURM scripts, frozen baselines from the same validation pass, derived smoke `--time`, Jay's approval for a fallback resubmission, a local same-group "composed beats frozen" test (section 12 row 8); third revision (milestone-8 design v4): pinned `init_weights` record schema (keeps `ignored_source_parts`; cross-level adds `source_level`, `source_step`, `cross_level`), config.json rewrite with `bn_recalibration` before `fit`, the resume-safe `CASCADE_FROM` guard, the always-documented A100 fit verdict, `g_gap` reporting keys, `label_luts.json` as the label table (sections 4.7, 11). Previous: v3.10 (2026-10-21). v3.10: optional per-task gradient normalisation `--task-grad-norm` / `--task-grad-norm-min` (section 5.2; default off, bit-identical). Previous: v3.9 (2026-10-20). v3.9: optional LNCC intensity loss for the deformable task (template term vs a leave-one-out intensity template and a pairwise term on the pairwise-Dice pairs, label-ROI masked, squared LNCC, window 5 / 7 / 9 at levels 4 / 2 / 1; weights default 0 so existing runs are bit-identical), `--lambda-dice-template` for an LNCC-only ablation, LNCC validation metrics (sections 6.2, 7, 9). Previous: v3.8 (2026-10-20). v3.8 (milestone 7): partial warm start of a deform model from a linear-only checkpoint, per-task loss weights and encoder grad-norm diagnostics with the measured loss-balance recommendation (keep `--loss-scale-mm2 1`: the global clip already balances the applied encoder gradients), the warm-started three-task recipe and per-task validation group sizes (sections 5.2, 6, 9, 11). Previous: v3.7 (2026-10-20). Incorporates Jay's answers to the v1 and v2 open questions. Decisions that changed since v1 are marked *(v2)*, *(v3)* or *(v3.x)*. v3.4: the dense path uses LayerNorm by default (section 4.4). v3.5: opt-in augmentation curriculum and shared global transform G (section 3.2). v3.6: mixed rigid + affine task schedule with one shared optimizer, the combined best metric, affine metrics (sections 5.2, 7), the G angle sign, G warm-up and warm start (section 3.2). v3.7 (milestone 6): the deformable decoder as implemented (4.5: stage table, widths (128, 96, 64, 32, 16), pre-norm-act zero-init SVF head), the diffeomorphic conventions (4.6: voxel units, pull warps, border vs zero fill, composition order, inverse via −v, half-pixel ×2 upsampling, interior Jacobian), the deformable losses as implemented (6.2: masked soft Dice over present labels, forward-difference smoothness, fp32 one-hot), per-task best files (5.2) and measured level-4 memory (9). Remaining items are in section 12.

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
- **Gradient checkpointing** (constructor flag, not part of `ModelConfig`): `--gradient-checkpointing none|integrator|decoder|both` wraps the integrator and/or the decoder in `tf.recompute_grad` (gradients verified equal to 1e-4). Recomputing the decoder in training mode applies its BN moving-statistic updates twice per step. Default none. *(v3.13, milestone-9 phase-1 design; implemented in v3.14)*: new mode `full` (deform task only; needs `deform` in `--task-ratio`) recomputes the encoder, the decoder and the integrator, the integrator as two separate blocks (exp(v), exp(−v)). Under `full` every BN moving statistic is updated **exactly once** per step: `models/checkpointing.recompute` treats call 1 as the forward and, on every later call (each recompute and any further gradient call), snapshots the statistics before it and restores them after it; the recompute path is taken only once `encoder.built and deform_decoder.built` (frozen statistics: no update and no restore ops; `recalibrate_bn`: one update per layer and batch). `decoder` / `both` keep their double update so the milestone-8 recipe reproduces. Further memory knobs (all default false, bit-identical when off): `--lean-sampler` (custom-gradient trilinear sampler that keeps only the coordinates; int64 GPU index math; the backward recomputes the corners from coords + a dy-dependent zero and accumulates them one corner at a time through a control-dependency chain; `lean_images_grad=False` for data-volume warps in the losses), `--lean-losses` (Dice without sliced / broadcast copies, LNCC under `tf.recompute_grad`), `--split-group-conv` (conv_a(x) + conv_b(group mean) with the existing kernel split in two, no doubled-channel tensor). *(v3.14, implemented and verified)*: `full` gradients are bitwise equal to `none` (fp32 and AMP, Grappler off) and its BN statistics equal one forward (rel 0); `both` keeps its double update. The lean sampler uses the control-dependency chain (no `while_loop` fallback needed): forward max |diff| 0, gradients ≤ 1e-7 rel; graph-mode memory on a level-2 field, N 2: integrator block 0.52 GiB lean vs 2.44 default (saving 1.92, asserted ≥ 0.85); without the dy dependency the lean block keeps ~0.6 GiB more (Grappler dedup is real in TF 2.10). Lean losses: terms rel 0, gradients 2.7e-7. Split: fp32 ≤ 1e-6 (block), model level ≤ 1e-5. Memory note from the hold-out: under `full` the two integrator blocks exp(v), exp(−v) are recomputed concurrently, so both block tapes are alive in the integrator backward.

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

*T at level 1 (v3.13)*: T stays one `ModelConfig` value used at every cascade level (the cascade config check is unchanged); if T = 6 is adopted, all levels are retrained with it. `tests/test_integration_steps.py` reports the inverse error (both orders) and folding for T = 6 / 7 at max |v| 2 / 4 / 8 voxels on the level-4 and level-1 grids. T = 6 is adopted only if the mean inverse error stays ≤ 2× the T = 7 value and ≤ 0.05 voxel with folding ≤ max(0.1 %, folding(T = 7) + 0.05 percentage points) at max |v| = 8 on the level-1 grid (verdict `t6_accuracy_ok` in `scratch/m9/integration_steps.json`, read by `scripts/memory_model.py recommend --t6-verdict`; without it T = 6 is excluded), and it raises the feasible level-1 deform N or restores ≥ 10 % memory headroom. With the lean sampler one composition costs ~24 B per voxel per image, so T = 7 is expected to stay. *(v3.14, measured)*: `t6_accuracy_ok = true` (level-1 grid, max |v| 8: mean inverse error 0.00207 vs 0.00202 voxel, folding 0 / 0; `.agents/tasks/milestone-9/integration_steps.json`). Memory at level 2 tiny, N 2: with the lean sampler T 6 vs 7 differs by ≈ 0; with the default sampler under `full` T 7 costs 721 MB more (two compositions per step). At the level-1 recommendation phase D binds, so T = 6 raises neither N nor headroom: **T = 7 stays**. The m8 `diag_u_res_max_vox` magnitude check is pending from the m9 job.

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
- *Naming (v3.11)*: "level 1 / level 2" in these defaults are **cascade positions** (level 1 = the 1/4 stage = code `--level 4`, level 2 = the 1/2 stage = code `--level 2`), not the code's level numbers.

**As specified for milestone 8** *(v3.11; full detail in `.agents/tasks/milestone-8/design.md`)*:
- **Frozen stage:** `models.cascade.FrozenStage` = `GroupRegLevelModel(4, cfg)` restored strictly from `--cascade-from` (source config.json required, source level 4, ModelConfig equal to the level-2 model, seed excepted), built under a float32 Keras policy, always `training=False`, called before the GradientTape opens, outputs through `tf.stop_gradient`, not in the optimizer. Its input is the 2×2×2 average pool (`tf.nn.avg_pool3d`) of the flipped, augmented level-2 group (same T_i, G, flips, noise), in training and validation. It is never recalibrated. Its weights are stored in the level-2 checkpoint (`extra` key `frozen_stage`) and verified on resume by a fingerprint in config.json `cascade`.
- **Weight copy:** the frozen model is restored, then `copy_weights_from` gives the level-2 model every variable verbatim (encoder, 1×1 reductions, dense, both linear heads, deform decoder, SVF head; nothing re-initialised, no SVF ×2 rescale: v is in level-2 voxels, half the mm per unit, the safe direction for a residual). Shapes are identical at both levels (bottleneck 5×7×6×24 = 5040), translations are in mm.
- **Linear:** x_i = I_i ∘ A4_i⁻¹ (`transform_image`, zero fill; one extra interpolation accepted), residual R_raw_i from the copied heads, A_i = Center(R_raw_i · A4_i) with ONE `GroupCentering` (the level-2 model's). Losses, pairs A_j⁻¹ A_i, targets (from T_i, unchanged) and metrics use the composed A.
- **Deformable:** U = up(u4), U⁻ = up(u_inv4) (u4 integrated at level 4, `upsample_displacement`); level-2 input warp(I_i, U_i); residual v2 group-mean centered; u = compose(U, u2), u_inv = compose(u2_inv, U⁻) (`diffeo.compose_cascade`). Template / pair Dice, LNCC, smoothness (on the composed u), Jacobian and inverse consistency use the composed fields; one label interpolation. New metric `val_deform/composed_mean_disp_mm` (composite centrality).
- **Skip:** `--cascade-skip-prob` (default 0; stateless draw per step from (seed, step); A_prev = I and U = U⁻ = 0 when skipped; no retracing; never in validation). Not used in the milestone-8 recipes.
- **Validation:** the frozen stage is recomputed inside the traced validation function on every pass (no field cache); only the frozen-stage baseline metrics (`val_<set>_frozen/...`) are cached per validation set.
- **Resume:** the source checkpoint is not re-read; the frozen stage is restored from the level-2 checkpoint and fingerprint-checked; BN recalibration is not repeated (statistics come from the checkpoint) and the BN freeze is re-applied from the saved config.json `bn_recalibration` record; config.json `init_weights`, `cascade`, `bn_recalibration` are carried over verbatim. `--cascade-from` requires `--deform-residual-affine false`. The m8 train script skips its `CASCADE_FROM` existence guard when `ckpt/last/checkpoint` exists (requeue).
- **Provenance records** *(milestone-8 design v4)*: every `init_weights` record keeps today's nine keys (incl. `ignored_source_parts`, which `main._init_weights_lines` reads unconditionally); cross-level records (`--cascade-from`, or `--init-weights` from another level) add `source_level`, `source_step`, `cross_level`; the cascade record adds `via: "cascade"`. Same-level records are unchanged. On a fresh run config.json is written as today, then `bn_recalibration` is added by a second write after the recalibration and before `fit` (so before the first checkpoint).
- **Reporting:** the checker's `g_gap` INFO lines (rigid vs rigid_g, affine vs affine_g, composed and `_frozen`, incl. the affine scale / shear / QR-rotation factors) feed the level-2 G-gap table that revisits the level-4 affine gap (section 7 / milestone 7). The SLURM header always states whether the chosen recipe's measured peak fits the 40 GB A100 (≤ 36 GB).

**As specified for milestone 9 phase 2** *(v3.15; full detail in `.agents/tasks/milestone-9-phase2/design-m9-phase2.md`)*:
- **Frozen chain:** `models.cascade.FrozenChain` = frozen level 4 + frozen level 2, both restored from ONE milestone-8 level-2 cascade checkpoint (`m8_level2_gradnorm/ckpt/best`, step 6000: object `model` = level 2, object `frozen_stage` = level 4, the level-4 weights fingerprint-checked against the m8 config.json `cascade.fingerprint`), float32 policy, `training=False`, `stop_gradient`, outside the optimizer and never recalibrated. Checkpoint keys of a level-1 run: `frozen_stage_4`, `frozen_stage_2`; config.json `cascade` gains `source_level` 2, `stages` (level, checkpoint key, fingerprint, origin checkpoint / step per stage), `S_rel_diff_vs_level2`, `skip_stage` 2.
- **Trainable level 1:** identical default ModelConfig to the frozen levels (`check_cascade_config` stays strict; no different-width stage, Jay's decision), weights copied from the frozen level-2 model; one `GroupCentering` (the level-1 model's) on the composed linear transform A = Center₁(R1_raw · Center₂(R2_raw · A4)); deform u = compose(up(compose(up(u4), u2)), u1) with the inverse unrolled factor by factor. The math equals the phase-1 stand-in (`tests/gpu/frozen_chain_standin.py`) exactly (max |diff| 0).
- **Dispatch:** the Trainer branches once, at trace time, on `n_stages` (0 none, 1 `FrozenStage`, 2 `FrozenChain`); single-stage (level-2) runs build today's ops, keys and records.
- **Validation:** `val_<set>_frozen/...` = levels 4 + 2 composed, `val_<set>_frozen4/...` = level 4 alone (both cached per set), `val/wall_s` logged at level 1 only. `val_deform_frozen/{centrality_mm, raw_centrality_mm}` describe the level-2 residual SVF only.
- **Resume / provenance:** both frozen stages restored from the run checkpoint and fingerprint-checked per stage; BN recalibration once at the start of a fresh run, never on resume (milestone-8 rules); `slurm/m9_level1_cascade.sh` skips its `CASCADE_FROM` guard on requeue.
- *Deviation:* the level-1 recipe uses skip probability **0** (as in the milestone-8 runs), not the 0.15 default above. Skip of the last frozen stage (level 2: A_prev = A4, U = U4) is implemented (`--cascade-skip-prob`), so 0.15 can be tried without code changes. *Jay (2026-10-24): keep 0 for now.*
- **Planned skip schedule (Jay, 2026-10-24; not implemented yet):** set per task head and per frozen stage.
  - Rigid/affine heads: level-4 stage skip **0.0**, never skipped, because it corrects most of the linear error and must always run; level-2 stage skip **0.15**.
  - Deformable head: level-4 and level-2 stages skip **0.15** each, since deformation can be learned at any scale.
  - Needs per-task, per-stage skip probabilities. Today there's a single `--cascade-skip-prob`, which only skips the last frozen stage. Skipping level 4 means A_prev = I, U = 0 for the whole frozen chain. Skipping level 2 alone means A_prev = A4, U = up(u4).

---

## 5. Training procedure

### 5.1 Coarse-to-fine for every task

| Level | Group size N *(v2)* | Notes |
|---|---|---|
| 1/4 | 32–48 | Train encoder, linear decoder and deformable decoder from scratch to convergence |
| 1/2 | 8–12 (v3.11: from the measured level-2 memory; targets linear 16 / deform 8, floors 8 / 4) | Copy weights (adds parameter-free pools and upsamples only); recalibrate BN statistics with forward-only passes, then freeze them; 1/4 model frozen as the first cascade stage |
| Full | 2–4 | Same procedure; prior levels frozen; p_skip as in 4.7 |

N is finalized from measured memory. With N = 2–4 at full resolution, BN statistics are already frozen, and group-mean features come from very few images. That's an extra reason for the test-time N sweep in section 7.

**BN recalibration as specified** *(v3.11, milestone 8; scope and resume pinned in v3.12)*: `--bn-recal-batches K` (default 0 = off; recipe 64) and `--bn-recal-freeze` (default true). Cascade only: K > 0 without `--cascade-from` fails argument parsing (`parser.error`, message names `--cascade-from`), so level-4 runs and a no-cascade level-2 warm start never recalibrate. It runs ONCE, at the start of a fresh run (step 0, no `ckpt/last`), after the weight copy and before any training step. On resume it is NOT repeated: the recalibrated moving statistics are restored from the checkpoint and only the freeze flag (not checkpointed) is re-applied from the saved config.json `bn_recalibration.freeze`; a run preempted before its first checkpoint restarts as a fresh run and recalibrates again. Each plan slot draws one batch; a non-finite batch uses up its slot without an update (no replacement), and a task left with 0 used batches is an error. Procedure: K forward-only passes of the trainable stage in training mode on real training batches through the cascade input path (pool → frozen stage → pre-warp; flips and G on, G resampled as over steps 0..K−1), tasks allocated in proportion to the task ratio (1:1:2 → 16 / 16 / 32), eager calls with `bn.momentum = c/(c+1)` per layer so each moving statistic becomes the exact mean of its per-batch statistics (the level-4 values are replaced). No gradient, no optimizer or step change. Validation at step 0 before (`bnrecal_pre/...`) and after (`bnrecal_post/...`, statistics still unfrozen) gives the eval vs train-mode gap; then the statistics are frozen (`set_bn_stats_frozen(True)`, `rebuild_steps()`; re-applied on resume). After freezing, `val_<task>_trainmode` equals eval mode by construction. Level-2 schedule: fresh Adam 3e-4 cosine, warmup 500, 8000 steps, validation and checkpoints every 500, ratio 1:1:2, Dice + LNCC 1/1, G 45° / 10 mm from step 0, curriculum off. Gradient balance (Jay, C4): `--task-grad-norm 12` first; if the smoke run's encoder-gradient ratio linear/deform (median `diag_enc_grad_norm_eff`, windows from step 50) is outside [0.1, 10] for either linear task, fall back to clip 12 + `--task-loss-weights deform:64`. The step cut uses the log record's `step` (window end) ≥ 50; with `--log-every 25` the first used window is steps 26–50 (v3.12).

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
- **Minimum pair count** *(v3.15)*: `--min-pairs {1,n}` (default `1` = unchanged). With `n` the sampled pairs per step are k = min(max(N, round(p_pair N(N−1))), N(N−1)); at k = N they form a random N-cycle (every image once source, once target), with the same single RNG draw as the default. Linear N 16 / 32 are unchanged (24 / 99 pairs); validation pairs are unaffected.

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
- **Minimum pair count** *(v3.15)*: the same `--min-pairs n` applies to the pairwise Dice and the pair LNCC (same pairs): level-1 deform N 4 → 4 pairs (a random 4-cycle) instead of 1, level-2 N 8 → 8 instead of 6; N ≥ 12 unchanged.

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
- **LNCC** *(v3.9, `.agents/tasks/lncc-loss/memory.json`, same local setup, 5 measured steps)*: LNCC 1/1 vs off changes the measured peak by −41 … +65 MB at narrow N = 2–4 and −49 MB at default N = 2 (within run-to-run noise; no measurable per-image slope), step time +6 / +8 / +21 % (narrow N = 2 / 3 / 4) and +3 % (default N = 2). *(Corrected in v3.13; was "the peak is the integration backward, and the LNCC activations are freed before it".)* The level-2 measurements put the `none` peak at the end of the forward pass: an integrator-backward peak under-predicts level-2 fp32 `none` N2 by 20 %. The loss terms' local tensors need not all be alive at that point, which is why LNCC (≈ 3 tensors of [5N] volumes for the template term and of [5P] for the pairs, 0.43 MB per level-4 volume) adds no measurable memory at level 4. The milestone-9 memory model fits this as `c_loss`. **Estimate (not measured)** at default widths N = 16 (24 pairs): ≈ 0.1 GB template + ≈ 0.15 GB pairs stored, ≤ 0.5 GB in the worst case, on top of the 33.8 GB, so N = 16 still fits the 40 GB A100 and the 48 GB cards.

**Level 2 (milestone 8 plan, v3.11; to measure on the cluster before the smoke run)**: only the stem, L2, D2, D1, the SVF head, integration, warps, one-hot and LNCC grow 8× at level 2; L3–L6 and D3–D5 keep their level-4 sizes. Estimate ≈ 7–10 GB per image for a trainable deform step at default widths (upper bound 8 × 2.15 ≈ 17 GB), linear ≈ 2–3 GB per image; the frozen level-4 stage adds inference only (no taped activations) and ~0.12 GB of weights. `tests/gpu/measure_level_memory.py` measures one level-2 step per task with cascade on/off, `--gradient-checkpointing none|integrator|decoder|both`, `--amp false|true` (heads, fields, integration, warps, Dice and LNCC stay float32; the frozen stage stays float32), LNCC 1/1, N grid (deform 2–8, linear 4–16), ≥ 10 measured steps (mean and median). Group-size rule: deform N = 8 with the cheapest option that fits ≤ 44 GB (none < integrator < both < AMP), else the largest N ≥ 4; linear N = 16 if it fits, else the largest N ≥ 8; "fits the A100" only if ≤ 36 GB. Fallback order: checkpointing, then AMP, then smaller N. *(v3.12)* Deform validation is measured too (`--val-pass true`: one eager `_deform_errors` pass with pair chunk 16, `val_peak_mb`), since its pair chunk does not scale with N; if it exceeds 44 GB, `--val-pair-chunk` (default 16 = current behaviour) is lowered to 8 or 4.

**Level 1 (v3.13, milestone-9 phase-1 design v2; details and tables in `.agents/tasks/milestone-9/design-m9-phase1.md` sections 2 and 4)**: an analytic tensor inventory (taped tensors per image per voxel: 5·C_in per group block, 4·C for the SVF head, 84 + 32·C B per default-sampler call; phases forward-end F / integrator backward I / decoder backward D / encoder backward E / frozen inference Z / validation V; plus a backward transient of ≈ 2 × the largest block's input channels × 4 B in the peak phase) reproduces all 21 measured level-2 deform points within −5 … +1 % before fitting (AMP `both` 7.6 / 14.3 / 21.0 / 27.8 vs 7.6 / 14.4 / 21.2 / 27.8 GB at N 2/4/6/8) and the level-4 points within −3 … +13 % (the GTX intercepts). The peak under `none` is the end of the forward pass, under `integrator` the integrator backward, under `both` the decoder backward in both precisions (the int32 sampler indices, assumed host-resident, move only `none` / `integrator` by +3 … +5 %). Root cause at level 1: the sampler tapes (14 compositions ≈ 2.5 KB and each 19-channel one-hot warp ≈ 0.7 KB per voxel per image), then D1 at full resolution with its backward transient; the encoder matters little at level 1 (only the stem is at full resolution). Pre-fit level-1 peaks (AMP, `full`, cascade, LNCC, T = 7): default widths with lean sampler + split N 3 32.9 / N 4 43.2 GiB; preset `f0d1` (filters 16, 64, 128, 192, 256, 384; decoder 128, 96, 64, 32, 8) with lean sampler + split N 4 33.5 GiB (38.5 with the 15 % margin), (`fr50` + lean losses N 6 37.7, `fr25` N 7 33.0: withdrawn in design v5, see below); linear N = 16 ≈ 12.7 GiB. Selection rule (`scripts/memory_model.py recommend`, replaces the earlier maximise-N rule): AMP + `full` at level 1, the widest model (trainable parameters) with deform N ≥ `--n-target` (default 4) at predicted peak × (1 + margin) ≤ 44 GiB, margin = max(0.15, worst knob hold-out error), plus the per-N table for Jay; expected `f0d1`, lean sampler + split, T = 7, N = 4 (the A100 fits N = 4 only without the margin, so level 1 trains on the 48 GB cards). The model fits nine pre-registered coefficients to `peak_mb` (the steady state; the autotune warm-up excess is reported, not budgeted): c_net, c_rec, c_warp, c_bwd, c_enc32, c_enc16, c_loss, and c_0 per GPU family. *(Design v3)* The level-2 linear points are −8 … +12 % pre-fit, with fp32 low and AMP high. The precision-split encoder coefficients (expected 1.10 / 0.89) bring them to −1 … +6 %. The resident term counts optimizer slots only for the parts the measured task touches. The build forward never enters the predicted peak (the tool resets after it). It is checked only by the limit flags, and every build runs at `min(N, 2)`: in the tool always, in `main.py` when `--split-group-conv` is true. Per-tensor limit: the linear fp32 N = 12 `d2` failure is either a > 2³¹-element limit on the fp32 conv path (2.64·10⁹ elements) or an OOM; `tests/gpu/probe_tensor_limit.py` tests forward and backward kernels (conv, BN, LeakyReLU under GradientTape). Until it has run, tensors entering backward kernels are bounded at 2³¹ − 1 elements in both precisions, which is why `--split-group-conv` is needed for N ≥ 4 at the default and `f0d1` widths. The measurement tool builds every config at `min(N, --build-n 2)`. Measurement: `slurm/m9_memory.sh` (rtx8000, 10 h; level-1 grid around the recommendation with a frozen-chain stand-in for levels 4 + 2, a `--cascade false` point and the frozen-stage peak). *(Design v4)* The lean sampler's backward adds `dep = where(cast(max(dy[:1]), bool) == itself, 0, NaN)` to the coordinates before recomputing lo / frac / indices / gathers, so no recomputed node shares its inputs with a forward node and Grappler's dedup cannot keep the corner tensors alive; the forward is unchanged (bit-identical). Every level-1 number above assumes this and is gated by `tests/test_lean_sampler.py::test_graph_mode_memory` (Grappler on; assertions revised in design v5, below). Skip gradients d(skip_k) live from the backward of the consuming decoder block to the encoder backward; at the level-1 D1 moment none from finer blocks exist, so the peak is unchanged; counting them all there (conservative bound) gives `f0d1` N 4 34.9 GiB (40.1 with margin), still feasible. Level-4 per-task step times come from the m9 memory job × the m8 A100 / RTX 8000 per-task ratio, checked against the m7 mixed `time/compute_ms` (±20 %). *(Design v5)* **The lean backward is sequenced.** The 8 corners are accumulated one after another: corner k's ops carry a control dependency on corner k − 1's accumulators. If the memory test shows the chain is not honoured, the fallback is `tf.while_loop(parallel_iterations=1)`. The dy guard stays. `dep` is always 0, and a non-finite dy propagates through dy · g_k. **The lean sampler's transients are now in the inventory**, per voxel per image:
- the forward transient while a call runs, 84 + 32·C B plus 168 B of int64 GPU indices; its moment with the output is 936 B for the one-hot. It enters the sub-phase Fw at its call site: program order by default, with an order-independent bound;
- the sequenced backward transient, 88 + 8·C B (+16·C with the input gradient). It enters the sub-phase Fb and phase I.

**Level-1 consequences.** Fw is about 7.6–8.0 GiB per image for every preset, set by the one-hot warp. So:
- `f0d1` LS N4 is unchanged at 33.5 GiB (Fw 32.0, D binds);
- every lean configuration is capped at N 4 with the margin and N 5 without it; the earlier `fr50` N 6 / `fr25` N 7 alternatives are withdrawn;
- under the order-independent bound, N 4 is `f0d1` + lean losses at 35.0 GiB (40.3 with the margin).

Forward corner sequencing is the backlogged remedy; it keeps values bitwise identical and restores the alternatives. **The graph-mode memory test is replaced.** It runs on a level-2-size field at N 2:
- lean peaks ≤ 1.5 × the modelled transients (one-hot warp 2.25 GiB, backward alone 0.58 / 0.41 GiB);
- the lean integrator block at least 0.85 GiB below the default one, against a predicted 1.65 GiB.

**Measured and fitted (v3.14, milestone-9 phase 1; `.agents/tasks/milestone-9/report.md`).**
- Calibration: nine coefficients fitted to 45 points. Errors are −3.7 … +2.1 % (pre-fit −9.5 … +13.7 %). No coefficient is at a bound: c_net 1.00, c_enc32 1.06, c_enc16 0.75, c_loss 0.53, c_rec 0.91, c_warp 1.20, c_bwd 2.69, c_0 0.10 / 0.09 GiB. Every OOM and failed point is predicted infeasible.
- Leave-one-mode-out errors are within −6.5 … +2.6 %.
- Hold-out: 40 local GTX points, measured one configuration per process. The gate is one-sided; it is passed. The worst under-prediction is −13 %, so the margin stays 0.15. The symmetric ±20 % gate fails: 18 small level-4 AMP / split points are over-predicted by 20–52 %. This is conservative and is an open model-form question.
- Inventory correction: under `full` phase I holds both recomputed integrator blocks.
- Tool fix: in one process the allocator kept memory across configurations (`resident_mb` 73 → 329 MB), so `measure_level_memory.py --isolate true` runs one configuration per process. `slurm/m9_memory.sh` uses it, with `--time` 12 h.
- Per-tensor limit: graph mode finds exactly one op above 2³¹ − 1 elements for the m8 linear fp32 N 12 failure, the D2 concat [12, 80, 112, 96, 256] (2.64·10⁹). Run mode is pending from the cluster job.
- Level-1 recommendation (provisional): `f0d1` + lean sampler + split, AMP, `full`, T = 7.
  - Deform N = 4 at 35.0 GiB (40.2 with the margin, phase D); N 5 fits without the margin.
  - Order-independent estimate 38.6 GiB; under that bound N 4 also needs lean losses.
  - Linear N = 16 at 13.5 GiB.
  - A100: N 4 only without the margin, so level 1 trains on the 48 GB cards.
- Pinned recipes at `f0d1` widths: level 2 (m8) deform N 8 at 25.6 GiB; level 4 (m7) deform N 16 at 33.1 GiB. Both fit the A100.
- The cluster job measures the recommendation at N 3 / 4 / 5 and the `fr50` lean-sampler fallback first (step 2p).

**Level 1, phase 2 (v3.15; `.agents/tasks/milestone-9-phase2/design-m9-phase2.md` section 7).** Measured (job 10516237, RTX 8000, stand-in chain, AMP, `full`, lean sampler + split, T = 7, one deform pair): default widths N 4 31.0 GB, 15.6 s/step; validation 22.3 GB at `--val-pair-chunk 4`; N 5 OOM except `fr50` + lean losses (38.4 GB). Phase 2 keeps the default widths (identical ModelConfig across levels) and adds `--min-pairs n` (4 deform pairs at N 4). **Estimate (not measured):** +4 … +7 GB (the fitted model over-predicts level 1, so its +7.8 GiB delta is scaled down) → **35–38 GB** peak. Fits the 48 GB RTX 8000 / A6000 with margin; the 40 GB A100 (36 GiB budget) is marginal. Rule: the level-1 run trains on rtx8000; `-p dgx-a100` only if the smoke run's measured `gpu_peak_mb` ≤ 36 GiB (checker line `INFO gpu_peak`). If the smoke OOMs: `--lean-losses true` (model −3.7 GiB), then `--min-pairs 1`.

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
| 8 | 1/2 level: weight copy, BN recalibration, frozen cascade, skip; **full label set with per-step label sampling** (below) | Linear composition test; unrolled inverse over 2 levels; 1/2 error < 1/4 error; label sampling: same subset across the group, core labels in every step, full-label validation Dice matches an unsampled reference. *(v3.11 design: `.agents/tasks/milestone-8/design.md`. Warm start and frozen stage both from `results/m7_three_task_lncc/ckpt/best` (step 11000). Flags `--cascade-from`, `--cascade-skip-prob` (0), `--bn-recal-batches` (0), `--bn-recal-freeze` (true); all off by default, goldens must stay max \|diff\| 0. "1/2 error < 1/4 error" is measured on identical groups: every level-2 validation also reports the frozen stage's own metrics (`val_<set>_frozen/...`). Cluster order: `slurm/m8_memory.sh` (DALI level-2 checks + memory grid), then `slurm/m8_level2_cascade.sh` smoke (200 steps) and long (8000 steps), `GRAD_BALANCE=gradnorm\|deform64`; `check_mixed_run.py` prints `grad_balance KEEP\|FALLBACK` and composed-vs-frozen lines. On FALLBACK: cancel our own gradnorm long run, resubmit with `GRAD_BALANCE=deform64` after Jay approves and with fewer than two active jobs. Smoke `--time` derived from the memory job's step times, provisional 02:00:00. Local "1/2 error < 1/4 error" evidence: a tiny frozen level-4 stage trained on the pooled same group must be beaten by the composed level-2 result (`TestCascadeBeatsFrozen`).)* |

**Milestone 8 addendum: full label set with per-step label sampling** *(added 2026-10-20, Jay)*. Goal: use the full SynthSeg label set for the deformable Dice at levels 2 and 1 without one-hot memory growing with the number of classes. Level 4 keeps `train18`, because many fine labels are only a few voxels at 4 mm.
- **Data (Jay runs the export):** the existing `synthseg` label set of the exporter (the 103-id `--parc` map remapped to contiguous uint8 ids, ids in `label_luts.json`), exported at full resolution and `_2` (`labels_synthseg.npy`, `labels_synthseg_2.npy`) next to `labels_train18*`, plus Jay's list of **core** labels (raw SynthSeg ids, e.g. ventricles, mapped through `label_luts.json`). DALI is unchanged (labels are already warped nearest-neighbour).
- **Sampling:** each deform step draws one subset S = core ∪ K random non-core labels (seeded per step like the flips, CLI `--dice-labels full|train18`, `--dice-core-labels`, `--dice-sample-k`, default K about 8-12). The **same S for every image of the group**: the template Dice averages each channel over the group, so every image needs the same channels. Flips are group-wide, so no left/right swap is needed.
- **One-hot in TF after DALI** (in `DeformPostprocess`): build only the |S| binary maps `labels == id`, with no "rest" channel in the loss. Memory then scales with |S|, not the label count (level 1: about 520 MB per image for 19 float32 channels, estimated).
- **Weighting:** uniform sampling is unbiased for the all-label mean Dice in expectation. Core labels appear every step, so they get extra weight, which is intended. Optionally reweight non-core terms by (#non-core / K) for an exact all-label estimate.
- **Validation** always scores every label (hard Dice, argmax over all classes, computed in label chunks without gradients) so runs stay comparable. Report core-label and all-label means separately.
- **Checks:** gradient noise via the milestone-7 per-task encoder gradient diagnostics; the sampled loss averaged over many seeds should match the full-label loss on a fixed group.
- **Status (v3.11): deferral to milestone 9 ((B)) applies unless Jay chooses to implement it now ((A)); nothing in milestone 8 depends on it, and the decision stays open in the milestone-8 report.** Reasons: no full-label export yet and the core list / K are open; at level 2 the 19-channel one-hot is ~65 MB per image against an estimated 7–17 GB per image for the step, so sampling saves nothing that matters there (the gain is at full resolution); keeping train18 isolates the cascade's effect. Milestone-8 runs use train18 at level 2.
- **Export (milestone 8 does the exporter change either way):** `preprocessing/export.py` gets `LABEL_SET_LEVELS = {"train18": (1, 2, 4), "synthseg": (1, 2)}` (replaces `PYRAMID_LABEL_SETS`; same block mode, smallest-id tie-break; no synthseg `_4`). Jay then runs `preprocessing/slurm/export_array.sh` with `LABEL_SETS=train18,synthseg` for train (`SPLIT=train`) and T1 val (`SPLIT=val`, `OPTIONAL_LABEL_SETS=lumir10`, one task), then `MERGE_ONLY=1`. Outputs per case `labels_synthseg.npy`, `labels_synthseg_2.npy`, listed automatically in the manifest's `labels_l1` / `labels_l2` (3384 train + 39 T1 val). Jay also supplies the core-label list (raw SynthSeg ids) and K (8–12 suggested). Exact commands: `.agents/tasks/milestone-8/design.md` section 14. The label table (contiguous id, SynthSeg id, name) is the exporter's `label_luts.json` lut `synthseg` (`map` + `names`, from `preprocessing/luts.py`); the core flag is derived at load time from Jay's raw-id list, so no extra table file.
| 9 | Full level, mixed precision, checkpointing, memory profiling | Peak memory per N logged; full-res deformable step fits on 48 GB. *(v3.13: phase 1 = memory root cause, knobs `--gradient-checkpointing full` / `--lean-sampler` / `--lean-losses` / `--split-group-conv` (off by default, goldens max \|diff\| 0), width presets, calibrated memory model with a knob hold-out set and the level-1 recommendation (widest model with deform N ≥ 4 within 44 GB under a 15 % margin, expected `f0d1` at N = 4; A100 verdict ≤ 36 GB), forward + backward tensor-limit probe, level-1 measurement tool + `slurm/m9_memory.sh`; design `.agents/tasks/milestone-9/design-m9-phase1.md`. Later phases: 3-level frozen chain (`PREV_LEVEL = {2: 4, 1: 2}`, `--cascade-from` a level-2 cascade run, skip of the last frozen stage 0.15), level-1 DALI and validation (`--val-pair-chunk 4`), then retraining levels 4 → 2 → 1 with the recommended widths (levels 4 / 2 on the A100 with the m7 / m8 recipes, level 1 on the 48 GB cards with the measured N); acceptance vs `m7_three_task_lncc` / `m8_level2_gradnorm`: center RMS ≤ +10 %, Dice ≥ −0.01, folding ≤ 0.05 %.)* *(v3.14: phase 1 implemented and verified locally (full suite, goldens max \|diff\| 0, hold-out one-sided gate passed). Provisional recommendation `f0d1` + lean sampler + split, AMP, `full`, T 7, level-1 deform N 4 (35.0 GiB predicted) / linear N 16. Next: Jay approves `slurm/m9_memory.sh` (RTX 8000, 12 h), then phase 2 (R5). Report `.agents/tasks/milestone-9/report.md`.)* *(v3.15: memory job 10516237 done (default widths N 4 31.0 GB, widths barely matter, so no retraining of levels 4 / 2; Jay: default widths, deform N 4, the N = 5 route documented only). Phase 2 implemented locally: `FrozenChain` (levels 4 + 2 from `m8_level2_gradnorm/ckpt/best`), level-1 trainable stage with the identical ModelConfig, `_frozen` / `_frozen4` baselines, `val/wall_s`, `--min-pairs n`, checker `--expect-cascade-stages 2` / `--expect-min-pairs` / `--memory-budget-mb` / `--project-steps`, `slurm/m9_level1_cascade.sh` (rtx8000, 3 CPUs, 64G; smoke 4 h, long 48 h; A100 only if the smoke peak ≤ 36 GiB). Next: Jay approves the level-1 smoke, then the long run. Report `.agents/tasks/milestone-9-phase2/report.md`.)* |

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
3. A richer label set for cortical Dice: the full label set with per-step label sampling (section 11 addendum), deferred to milestone 9 unless Jay chooses milestone 8 *(v3.11)*. Open: (A) now or (B) later, which labels are core (raw SynthSeg ids), and K; Jay's `train18,synthseg` export.
4. A larger rigid translation range.
5. *(v3.11)* Level-2 group sizes and whether AMP / checkpointing are needed: from the milestone-8 memory job.
6. *(v3.11)* The affine G gap (level 4: affine pair 6.92 vs 9.45 mm, rotation and shear): revisit with the level-2 numbers (`val_affine` vs `val_affine_g` and their `_frozen` baselines), as Jay tabled.
7. *(v3.11)* Encoder-only gradient normalisation, if the level-2 smoke run triggers the C4 fallback (after all milestones).

New questions will be added here as implementation surfaces them.
