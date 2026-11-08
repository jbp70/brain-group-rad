# Smoothness-loss scaling across levels (read-only investigation)

Date: 2026-11. No code changed, no jobs submitted. Cluster reads: run `config.json` and `logs/metrics.jsonl` files, MultiMorph clone under `baselines/multimorph`.

## Summary

1. **There is no per-level scaling bug.** The smoothness term is the mean over all voxels, components and axes of squared forward differences of the displacement in the voxel units of the current level (`runtime/losses.py:172` `smoothness_loss`, order 1). For a fixed physical deformation, a one-voxel difference of a voxel-unit field equals the dimensionless spatial derivative ∂d/∂x at every level, and the mean does not depend on the voxel count. So λ_smooth = 0.5 applies the same physical penalty at levels 4, 2 and 1. All four runs in the chain used λ_smooth 0.5, order 1, Dice 1/1, LNCC 1/1 (table below).
2. At level 1 the penalty is applied to the **composed** field (frozen level 4+2, upsampled, composed with the level-1 residual), not only to the residual (`runtime/run.py:938-942`). The gradient reaches only the residual, because the frozen fields carry `stop_gradient`.
3. **MultiMorph uses the same regulariser formula** (`Grad3d('l2')`: per-axis mean of squared forward differences, averaged over 3 axes, in voxel units). Its 1 mm data put its λ numbers on the same scale as ours at level 1. Paper values: λ_reg = 1.0, Dice γ = 0.5, NCC weight 1, template terms only. **Its training script is not in the repo** (README: "Training (Coming soon)"). Its field is a 2 mm SVF (stride-2 first conv), integrated with 5 steps at 2 mm and upsampled x2 trilinearly to 1 mm. It therefore has no 1 mm degrees of freedom.
4. **Conclusion.** Relative to the similarity terms, our level-1 regulariser is weaker than MultiMorph's:
   - **2x** on λ alone (0.5 vs 1.0);
   - **~4-8x** against the Dice terms (we weight template + pair Dice 1 + 1, they weight template Dice 0.5);
   - **roughly 10-20x** against LNCC (estimate). We have two LNCC terms, and ours is a mean over a foreground ROI while MultiMorph's NCC is a mean over the whole volume. If the ROI is ~20-25 % of the volume (estimate, not measured), each of our LNCC terms pushes ~4-5x harder per brain voxel.

   At levels 4 and 2 this weakness was hidden because the coarse grids cannot represent fine-scale deformation. Level 1 is the first stage with 1 mm degrees of freedom, a 9 mm LNCC window and a 1 mm one-class cortex Dice. It adds only ~+0.005 to the composed smoothness value (0.059 to 0.064, i.e. +0.0025 in loss), while folding rises 14x. Recommendation: a level-1-only sweep with λ_smooth 2 and 4, plus a MultiMorph-ratio run (Dice 0.5/0.5, λ 1). No code change is needed to make order-1 λ resolution-consistent; it already is. Order 2 (bending energy) is **not** resolution-consistent (see 2.3).

## 1. How the losses are defined (evidence)

### 1.1 Smoothness
- `runtime/losses.py:172-198` `smoothness_loss(u, order=1)`.
  - Order 1: for each axis a, `tf.reduce_mean(tf.square(u[x+e_a] - u[x]))` over [N, X-1, Y, Z, 3] (all images, all voxels including background, all 3 components), summed over the 3 axes and divided by 3. Equivalently, L_smooth = (1/9) Σ_{a,c} ⟨(Δ_a u_c)²⟩ = ⟨‖∇u‖_F²⟩ / 9 with unit voxel spacing.
  - Squared. Mean, not sum. No brain mask.
  - Order 2: bending energy (central second differences plus 2x cross terms, interior only).
- Field: the **displacement u**, not the SVF v. Units are voxels of the current level (`models/diffeo.py` header: "Displacements and velocities are in voxel-index units of the level they live at").
- Called in `deform_loss` at `runtime/losses.py:492`: `smooth = smoothness_loss(u, smooth_order)`, with u = `out["u"]`.
- Cascade: `runtime/run.py:938-942`:
  ```
  out = model(x_in, ...)                       # residual (level-1 stage)
  u, u_inv = diffeo.compose_cascade(U, U_inv, out["u"], out["u_inv"])
  out = dict(out, u=u, u_inv=u_inv, u_res=out["u"])
  terms = losses.deform_loss(out, batch, lam_p, lam_s, order, **ncc_kw)
  ```
  `compose_cascade` (`models/diffeo.py:163`) returns compose(U_prev_up, u_new) = u_new + U(x + u_new). So smoothness, Dice and LNCC all see the **composed** level-1 field. U is the frozen 4→2 chain field, upsampled by `upsample_displacement` (2x values, trilinear) with `tf.stop_gradient` (`models/cascade.py:251-253`, `296`). The penalty therefore includes the frozen field's gradient energy. Its derivative with respect to the residual contains a cross term ⟨∇U, ∇u_res⟩, so the residual is also nudged to smooth the coarse field.
- Flags (`runtime/args.py:373-375`): `--lambda-smooth` (default 0.5) and `--smooth-order` {1, 2} (default 1). `TrainConfig.lambda_smooth = 0.5`, `smooth_order = 1` (`runtime/run.py:548-549`). The scripts hard-code `--lambda-smooth 0.5` in `DEFORM_LOSSES` (`slurm/r_level1.sh:260`, `r_level2.sh:268`, `r_level4.sh:264`). `$EXTRA_ARGS` comes last on the command line (`r_level1.sh:309`), so `EXTRA_ARGS="--lambda-smooth 2"` overrides it (argparse last-wins). The r4_cfgB run confirms this: its config.json has λ_smooth 1.0 and Dice 0.5.

### 1.2 Dice
- `soft_dice` (`losses.py:96`): per channel (2Σab + eps)/(Σa + Σb + eps) over voxels, background excluded, then the mean over **present** foreground channels.
  - Template term (`template_dice_loss`): warped one-hots vs their group mean (gradient through the mean), mean_i (1 − dice).
  - Pair term (`pair_dice_loss`): warp(onehot_i, compose(u_i, u_inv_j)) vs onehot_j over sampled pairs (p 0.1, `--min-pairs n`; at level 1 with deform N 4, that is 4 pairs).
- Labels: train18 = 18 foreground classes + background (`data_loading/params.py:46` N_TRAIN_LABELS 19; `preprocessing/luts.py:94-114`). **All cortex (3/42 + DK parcels) is one class (id 2).** Left/right are merged. Cortex therefore has weight 1/18 within each Dice term, but as a thin ribbon at 1 mm it is the class most sensitive to fine local stretching.

### 1.3 LNCC
- `local_ncc` (`losses.py:240`): squared LNCC cov²/(var_a var_b + 1e-7), box window k³ in level voxels. Defaults per level `NCC_WINDOW_DEFAULT = {4: 5, 2: 7, 1: 9}` (`data_loading/params.py:63`), i.e. 20 / 14 / 9 mm.
- Loss = 1 − **masked mean** over the ROI = foreground dilated by k//2 (`ncc_roi`, `masked_mean`, divided by the ROI voxel count).
- Template term uses a leave-one-out intensity template. The pair term uses the Dice pairs' displacements.

### 1.4 Combination and gradient normalisation
- `losses.py:494-519`: L = λ_dt·L_tmpl + λ_pd·L_pdice + λ_s·L_smooth + λ_nt·L_ncc_tmpl + λ_np·L_ncc_pair.
- `--task-grad-norm 12` (`run.py:812` `_normalise_gradients`) rescales the **total** deform gradient to global norm 12 (then multiplies by the task weight, null = 1). This changes the step size, not the relative balance of the terms inside the deform loss. Changing λ_smooth therefore changes only the gradient direction, and the effective step stays fixed.

### 1.5 Actual training configs (cluster, `results/<run>/config.json`)

| Run | Level | cascade_from | λ_dice_t | λ_pdice | λ_ncc_t | λ_ncc_p | λ_smooth | order | ncc_window | task_grad_norm / weights | lr / wd | deform N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r4sw_C_s16000_lr3e-4_wd1e-1_n128_d64 | 4 | none | 1 | 1 | 1 | 1 | 0.5 | 1 | 5 | 12 / null | 3e-4 / 0.1 | 64 |
| r2_s15000_lr3e-4_wd1e-1_n32_d16 | 2 | r4sw_C…/ckpt/last | 1 | 1 | 1 | 1 | 0.5 | 1 | 7 | 12 / null | 3e-4 / 0.1 | 16 |
| r1sw_lr1e-4_wd1e-1_n8_d4 | 1 | r2_s15000…/ckpt/last | 1 | 1 | 1 | 1 | 0.5 | 1 | 9 | 12 / null | 1e-4 / 0.1 | 4 |
| m9_level1_cascade (old) | 1 | m8_level2_gradnorm/ckpt/best | 1 | 1 | 1 | 1 | 0.5 | 1 | 9 | 12 / null | 3e-4 / 3e-5 | 4 |
| r4_cfgB_dice05_smooth1 (ref) | 4 | none | 0.5 | 0.5 | 1 | 1 | 1.0 | 1 | 5 | 12 / null | 3e-4 / 0.1 | 64 |

All runs used integration_steps 7 and p_pdice 0.1. λ is the same at every level of the production chain.

Last logged training values (`logs/metrics.jsonl`, final record; single 25-step log records, so noisy):

| Run | smooth (composed) | tmpl | pdice | ncc_tmpl_loss | ncc_pair_loss | max \|u\| vox | max \|u_res\| vox |
|---|---|---|---|---|---|---|---|
| L4 r4sw_C (16000) | 0.036 | 0.309 | 0.260 | 0.174 | 0.169 | 4.6 (18 mm) | – |
| L2 r2_s15000 (15000) | 0.059 | 0.248 | 0.214 | 0.155 | 0.183 | 9.6 (19 mm) | 5.6 |
| L1 r1sw_lr1e-4 (4000) | 0.064 | 0.166 | 0.166 | 0.164 | 0.207 | 15.8 (16 mm) | 5.3 |
| L4 cfgB dice0.5 smooth1 (16000) | 0.031 | 0.321 | 0.275 | 0.170 | 0.171 | 3.6 | – |

Smoothness at L1 includes the frozen L4+L2 part. The level-1 stage adds only ~0.005 (≈ +8 %, ≈ 0.0025 in weighted loss) yet raises folding 0.015 → 0.21 % and log-Jacobian std 0.46 → 0.68 (predict-levels report §6). The volume-mean squared first-order penalty hardly "sees" localised fine-scale stretch. Because it averages over the whole 160x224x192 volume, mostly background, local cortical stretching is diluted further. MultiMorph has the same dilution.

## 2. Effective strength across levels (derivation)

### 2.1 Order 1 (what we train with)
Let d(x) be a fixed physical displacement in mm and s the level spacing in mm (4, 2, 1).
- Voxel-unit field: u_s(p) = d(p·s)/s.
- Forward difference: u_s(p+e_a) − u_s(p) = [d(ps + s e_a) − d(ps)]/s ≈ ∂d/∂x_a (dimensionless, s cancels).
- Squared and averaged over voxels gives ⟨(∂_a d_c)²⟩, a spatial average independent of the voxel count.

**Net factor between levels: 1** (to first order, for content smooth on the voxel scale). A deformation with structure at the voxel scale is underestimated by finite differences, but a coarse level cannot represent such a field anyway. The "x2 values per finer level" and the "/2 per-voxel gradient" cancel exactly. The mean also makes the term's gradient per voxel scale as 1/N_vox, exactly like the Dice sums (ratios) and the masked LNCC means. So the relative weighting does not drift with resolution either.

What does change across levels:
- (a) The degrees of freedom. Level 1 is the first stage whose residual can vary on 1-2 mm scales.
- (b) The similarity signal becomes more local: the LNCC window shrinks from 20 mm to 9 mm, and 1 mm labels resolve the 2-3 mm cortical ribbon.
- (c) The level-1 residual is penalised only through the composed field.

λ = 0.5 was set at level 4, where the grid itself acted as the regulariser. MultiMorph's 2 mm SVF plays that role in its model.

### 2.2 Composed vs residual
The level-1 penalty acts on u = u_res + U(x + u_res). Its value includes ⟨‖∇U‖²⟩, which is constant with respect to the trainable weights, so the absolute value (0.064) overstates what the level-1 stage pays. The marginal cost of a level-1 residual is ≈ λ·[⟨‖∇u_res‖²⟩ + 2⟨∇U : ∇u_res⟩]/9 + higher-order terms. The cross term can be negative, which makes cancelling coarse-field gradients cheap. Penalising only the residual would leave the level-1 increment the same to first order. It would not strengthen the regulariser; only a larger λ (or a different penalty) does.

### 2.3 Order 2 (bending energy): not resolution-consistent
Second differences of the voxel-unit field equal s·∂²d/∂x². Squared, they scale as s². For the same physical field, the value at level 1 is 1/16 of level 4 and 1/4 of level 2. If `--smooth-order 2` is ever used across levels, λ must be scaled by (s_ref/s)². This option is unused today.

## 3. MultiMorph (cluster clone `.../baselines/multimorph`, git `2de0265`)

- **Training code is not present.** README: "Training (Coming soon)". There is no training script, optimiser or loss assembly; `src/` only has inference, model, layers, loss classes and the dataloader. The weights are `trained_models/model_cvpr.pt`.
- Loss classes (`src/losses.py`):
  - `Grad3d('l2')` (lines 173-198): squared forward differences along the 3 spatial axes, `torch.mean` of each (over all elements including the channel axis), summed, /3. **This is identical to our order-1 `smoothness_loss`.**
  - `local_NCC_3d` (404-440): `FastNCC` with window 9; squared cc = cross²/(var·var + 1e-5); `-torch.mean(cc)` over the **whole volume** (no mask; background cc ≈ 0). The template is the group mean including self (not leave-one-out). It returns `lncc + lbd * Grad3d(warps)`, where `warps` is the displacement used by `Warp3d`, i.e. the full-resolution field in voxel units.
  - `DiceWarpLoss` (325-368): MONAI `DiceLoss(include_background=False)` of warped segmentations vs their group mean (template only, no pairs), with an optional `do_half_res`.
- Paper (arXiv 2504.00247, Implementation details and Suppl. 7; [paper](https://arxiv.org/html/2504.00247), paraphrased):
  - Loss = NCC(template, warped) + λ‖∇φ‖² + γ·softDice.
  - λ = 1.0 and γ = 0.5, chosen by grid search. λ in [0.5, 2] behaved well; γ in [0.1, 0.7] was acceptable.
  - Adam 1e-4, groups of 2-12, 80k iterations, 1 mm isotropic affinely aligned scans, 50 % synthetic groups.
- Field parametrisation (`src/models.py` `GroupNet3D`, instantiated in `src/build_atlas_inference.py:72`):
  - features [32,128,128,128], `do_half_res=True`. The first conv has stride 2, so the decoder ends at half resolution.
  - `SubtractMean` centrality on the SVF.
  - `VecIntGroup` scaling and squaring with **5 steps at half resolution** (`img_size/2`).
  - Then `ResizeTransform(1/2)` multiplies by 2 and upsamples trilinearly (align_corners=True) to full resolution (models.py:103-107, 159-178; layers.py:189-215). The inverse is exp(−v), handled the same way.
  - So MultiMorph's finest field degrees of freedom are 2 mm, roughly our level 2, and it never predicts a 1 mm residual.
- Unverified: whether training applied `Grad3d` to the upsampled 1 mm displacement or to the half-resolution one. By 2.1 the value is the same either way, up to interpolation.

## 4. Conclusions and recommendations

### 4.1 Is level-1 smoothness weaker than intended or than MultiMorph's?
- **Than intended (per-level scaling):** no. λ = 0.5 order-1 is physically identical at every level.
- **Than MultiMorph's:** yes. The weights are directly comparable (same formula, 1 mm voxel units at level 1).
  - λ alone: 0.5 vs 1.0, so **2x weaker**.
  - Against Dice: our sum of Dice weights is 2 (template + pair) vs their 0.5, so λ/Dice is 0.25 vs 2, **~8x weaker** (~4x if the pair term is counted as half, since it shares the same fields).
  - Against NCC: our LNCC weights sum to 2, and each is a ROI mean; theirs is 1 as a whole-volume mean. With an ROI fraction ~0.2-0.25 (estimate, not measured), the λ/NCC ratio is ~0.25·0.2 vs 1, **roughly 10-20x weaker**.
  - MultiMorph additionally has no 1 mm degrees of freedom.
- The level-4 and level-4+2 regularity matched MultiMorph because their grids are as coarse or coarser than MultiMorph's 2 mm SVF. That masked the weak relative λ. The level-1 stage exposes it, consistent with folding 0.21 % and 41 % of cortical voxels stretched > 1.5x for +0.019 Dice.
- The one-class cortex Dice at 1 mm and the 9 mm LNCC window plausibly drive the cortical stretching. They reward aligning the thin ribbon voxel by voxel. This is unverified as the specific cause.

### 4.2 Level-1-only retrain sweep (frozen L4 + L2 kept)
Base: `slurm/r_level1.sh` defaults with CASCADE_FROM `results/r2_s15000_lr3e-4_wd1e-1_n32_d16/ckpt/last`, LR 1e-4, WD 1e-1, 4000 steps. Use rtx8000, `--mem=128G`, TF_MEM_LIMIT_MB 42500 (≈14-15 h each). Only EXTRA_ARGS change:

| Run | EXTRA_ARGS | λ_s / Dice / LNCC | Purpose |
|---|---|---|---|
| S1 | `--lambda-smooth 2` | 2 / 1+1 / 1+1 | 4x stronger, otherwise the current recipe |
| S2 | `--lambda-smooth 4` | 4 / 1+1 / 1+1 | ~MultiMorph λ/Dice ratio (2) |
| S3 | `--lambda-dice-template 0.5 --lambda-pdice 0.5 --lambda-smooth 1` | 1 / 0.5+0.5 / 1+1 | MultiMorph weights; the same recipe already ran at L4 (r4_cfgB_dice05_smooth1) |
| S4 (optional) | `--lambda-dice-template 0.5 --lambda-pdice 0.5 --lambda-smooth 2` | 2 / 0.5+0.5 / 1+1 | combined |

- Order by priority because of the 2-job limit: S1 + S3 first, then S2 / S4 depending on the result.
- Judge on Mindboggle with `predict.py` (levels 4,2,1, deform), not only on validation:
  - template Dice;
  - folding %;
  - log-Jacobian std;
  - the per-image cortical-gradient retention and stretch > 1.5x fraction;
  - atlas sharpness (atlas_intensity_std).
- Also compare against the frozen 4+2 baseline (val tD/pD 0.873/0.794). The goal is the best regularity at a Dice gain above 4+2.

### 4.3 Code changes
- **None needed for resolution consistency** with order 1.
- Not needed for this sweep, but useful:
  1. Log the residual-only smoothness at level 1 as a diagnostic (e.g. `diag_smooth_res` = `smoothness_loss(u_res)`), because the composed value hides what the stage adds.
  2. If `--smooth-order 2` is ever used across levels, scale λ by (s/s_ref)⁻² (2.3).
  3. If folding stays above the target after the λ sweep, add a direct Jacobian penalty, e.g. a hinge on det J < ε or ⟨(log det J)²⟩ over the brain. This targets the stretch and fold statistics that the volume-mean gradient penalty dilutes. It is new code and needs a design note.
  4. A structural alternative is MultiMorph-style: predict the level-1 SVF at half resolution, then upsample, or drop the full-resolution decoder stage. This means an architecture change and a level-1 retrain, but it saves memory.

### 4.4 Risks
- Higher λ lowers Dice and LNCC. Level-1 Dice gains are small (+0.019 template on Mindboggle), so λ 4 may push composed Dice down to the frozen 4+2 level. Check that it stays above.
- The composed-field penalty also rewards residuals that smooth the frozen L2 field (cross term, 2.2). At high λ the level-1 stage could partly undo level-2 detail.
- Atlas sharpness may go either way. Less stretching should keep more native cortical gradient, but weaker alignment blurs the mean.
- Task grad-norm normalisation keeps the step size fixed, so there is no lr retune with λ. With heavier smoothness the deform-gradient direction changes, so the encoder-gradient ratio diagnostic (`diag_enc_grad_norm_eff`) should be rechecked in the smoke window.
- The ROI-fraction factor in 4.1 is an estimate. It could be measured from one validation batch (ROI voxels / volume voxels at level 1).
- Training metrics quoted in 1.5 are single log records.
