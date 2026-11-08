# Inverse error, centrality and subcortical Dice on Mindboggle: read-only investigation

Date 2026-11-06. Read-only: no tracked file was changed, no job submitted. Numbers come from the existing
`summary.json` / `per_label.csv` / `predict.json` / `metrics.jsonl` files on the cluster, read with the
small scripts in this folder (`summ.py`, `q23.py`, `q2b.py`, `sub_groups.py`; cluster copies in
`scratch/agent/invcen/`). "Ours" = `r1_cfgB_sm10_s2500_bnfreeze_true` (L4 cfgB smooth 0.5, L2/L1 cfgB
smooth 10), template-only eval, 101 subjects.

## Summary

1. **Inverse error (0.485 mm vs MultiMorph 0.231 mm).** The prediction and evaluation code are correct.
   The cascade composes forward and inverse in the right order, the written fields equal the internal ones
   to 2e-6 mm, and the evaluation measures ANTs at 0.013 mm with the same code. The error is
   discretisation error of scaling-and-squaring done separately for exp(v) and exp(-v). It is about
   0.1 voxel of the grid the SVF is integrated on, and **most of it is created at level 4** (4 mm grid).
   On LUMIR validation at level 1 (all voxels), the frozen L4 field alone has inverse error 0.110 mm,
   L4+L2 0.120 mm, and the full model 0.124 mm. So L4 accounts for about 89 %. MultiMorph integrates once,
   on a 2 mm grid, and its error is about the same in grid voxels (0.115 voxel at 2 mm vs about 0.10 voxel
   at 4 mm for our L4). Ranking: (a, spatial part, at L4) >> cascade accumulation of the L2/L1 pairs >
   (c) upsampling > (d) centering (no effect on the inverse) ≈ (e) grid/convention (verified zero) ≈
   (f) evaluation method (verified fine). More squaring steps would not help. The training-time value
   (0.12-0.19 mm) is low because it averages over **all** grid voxels, and about 70 % of them are
   background. Divided by the brain fraction (0.30), 0.124 mm becomes about 0.41 mm, which is close to the
   0.485 mm measured on Mindboggle.
2. **Centrality (0.42 mm vs 0.19 mm).** Evaluation measures |mean_i fwd_i| of the final composed
   displacement. The model centers each level's SVF in log space, exactly over all 101 subjects
   (streaming). The group mean of exp(v_i) is not zero (second-order term ≈ ½ mean_i (Dv_i) v_i), and each
   cascade composition adds a cross term mean_i (DU_i) u_i. So a non-zero value is expected, and it grows
   with each level. Config A ablation: L4 0.30 → L4+L2 0.51 → full model 0.63 mm. Ours (cfgB, S10):
   L4+L2 0.37 → full model 0.42 mm. ANTs reaches 0.32 mm. The training `centrality_mm` metric is the
   centered residual SVF (~1e-8 by construction), so training never measured this quantity.
3. **Subcortical Dice (0.727 vs 0.735).** The gap is 45 % corpus callosum. CC_Central /
   CC_Mid_Anterior / CC_Mid_Posterior lose 0.050 / 0.060 / 0.027, and the CC mean is 0.706 vs 0.734. The CC
   is not in the training labels: SynthSeg never outputs it, and its voxels fall into cerebral white
   matter. The 30 labels that train18 does contain lose only 0.005 on average (0.804 vs 0.809; ANTs
   0.805). The other 7 unlabeled structures (vessel, choroid plexus, hypointensities, optic chiasm) are
   level (−0.002). The planned `synthfine` set (SynthSeg --parc, 103 ids) still has no CC.
4. **Options** (details in section 4). Inference-only, cheap, and testable with the offline probe in
   this folder: (I1) fixed-point refinement of the inverse, and (C1) post-hoc re-centering of the
   composed maps. Medium cost: (I2) integrate the L4/L2 SVFs on the 1 mm grid at inference. Retraining:
   (I3) higher L4 smoothness in the planned cascade chain. Subcortical: L/R-split loss labels and/or a CC
   label source (data work for Jay), or report Dice without the untrained labels.

## 1. Inverse error

### 1.1 How the fields are produced (verified in code)

- Per level: `v = center_velocities(v_raw)` (group mean over axis 0) and
  `(u, u_inv) = exp_and_inverse(v, T=7)`. These are two independent scaling-and-squaring runs, each with
  7 `compose(u, u)` steps that resample trilinearly with border clamping (`models/diffeo.py` `integrate_svf`
  l.78-89, `exp_and_inverse` l.92-95, `compose` l.69-75). At N = 101 the streamed engine centers with
  the exact group sum over all chunks: `v = v_raw - S_v / N`, then `exp_and_inverse`
  (`inference/streaming.py` P12, l.~384-391).
- Cascade: `compose_cascade(up(U_prev), up(U_prev_inv), u_k, u_inv_k)` returns
  `(compose(U_prev, u_k), compose(u_inv_k, U_prev_inv))`, which is φ_prev∘exp(v_k) and
  exp(−v_k)∘φ_prev⁻¹ (`models/diffeo.py` l.163-170). This is the exact inverse of the composed map, and the
  order matches the pre-warp `warp(x, up(U_prev))` (`inference/runner.py` `_prewarp_deform_fn`,
  `_deform`). **(b) is correct.**
- Upsampling: `upsample_displacement = 2 * upsample_trilinear2x` (half-pixel, edge replicate), once per
  level step. Coarse checkpoint fields are upsampled to level 1 by `_up_to_level1`
  (`inference/runner.py`). Forward and inverse are upsampled separately.
- Writing (deform branch): `to_canonical` returns `U1, U1_inv` unchanged (`inference/canonical.py`
  l.~94), and `to_mm` multiplies by spacing 1. There is no resampling, because the processed Mindboggle
  grid is the level-1 grid. The internal check in `predict.json`
  (r1_cfgB_sm10_s2500_bnfreeze_true_deform/fields/predict.json, `internal_check`) gives
  `inv_consistency_max_mm` 2.9e-6 / 1.9e-6 (written vs internal composition), `coord_fwd_max_vox` 6e-5,
  and `inv_consistency_raw_max_mm` 9.2 / 7.8 mm (the intrinsic error itself). **(e) contributes nothing.**
- Evaluation: `inv_err_fwd = |compose(fwd, inv)|` = ψ(φ(x)) − x over the atlas mask (≥ 50 % of the
  warped images > 0, about 2.06 M voxels = 30 % of the 6.88 M grid). `inv_err_inv = |compose(inv, fwd)|`
  is taken over the subject mask (image > 0). Both use trilinear interpolation with edge clamping on the
  1 mm grid (`evaluation/evaluate.py` `_pass2` l.~240-245, `evaluation/fields.py` `compose` l.~227).
  The same code gives ANTs SyN 0.013 mm and BrainMorph TPS 0.017 mm, so it can measure near-zero error.
  **(f) is not the cause.** MultiMorph's inverse is also exp(−v): one centered SVF on the half-resolution
  grid, `VecInt` with 5 steps, then a single ×2 `ResizeTransform` (trilinear, align_corners=True)
  (`baselines/multimorph/src/models.py` l.~100-180, `load_model`: `diffeo_steps=5, do_half_res=True`).
  ANTs SyN optimises forward and inverse jointly and inverts numerically, so its inverse is near exact by
  design. The comparison with MultiMorph is like-for-like. The comparison with ANTs is not.

### 1.2 Measured numbers (Mindboggle, 101 subjects)

| run | levels | inv_err fwd / inv mm | max fwd mm | mean \|fwd\| mm | fold fwd % | centrality mm |
|---|---|---|---|---|---|---|
| r1sw (cfgA, smooth 0.5) | 4 | 0.404 / 0.424 | 6.0 | 2.03 | 0.0002 | 0.302 |
| r1sw | 4,2 | 0.621 / 0.644 | 11.9 | 2.53 | 0.015 | 0.508 |
| r1sw | 4,2,1 | 0.765 / 0.747 | 14.2 | 2.79 | 0.211 | 0.630 |
| r1sw | 2,1 (L4 skipped) | 0.631 / 0.770 | 10.6 | 2.33 | 0.280 | 0.514 |
| r1sw | 1 (only) | 0.323 / 0.423 | 4.9 | 1.74 | 0.110 | 0.280 |
| r2 cfgB S4 / S8 / S10 / S12 @15k | 4,2 | 0.487 / 0.444 / 0.433 / 0.426 | 9.7 / 8.1 / 7.6 / 7.2 | 2.25 / 2.18 / 2.16 / 2.15 | 0.0016 … 0.00002 | 0.423 / 0.386 / 0.372 / 0.364 |
| **ours** r1 cfgB S10 | 4,2,1 | **0.485 / 0.484** | 8.8 | 2.36 | 0.0016 | **0.420** |
| MultiMorph | 1 SVF @2 mm | 0.231 / 0.278 | 4.2 | 2.22 | 0.0002 | 0.186 |
| ANTs SyN | — | 0.013 / 0.013 | 0.23 | 2.94 | 0 | 0.316 |

LUMIR validation, `metrics.jsonl` last record. The inverse error is a mean over **all** voxels of the
level grid (`runtime/run.py` l.1487-1490 → `runtime/metrics.py` `deform_group_errors` l.~229-240):

| run (step) | measured grid | key | inv mm |
|---|---|---|---|
| r4_cfgB_dice1_smooth05 (16000) | L4 | val_deform | 0.282 |
| r2_cfgB smooth10 (15000) | L2 | val_deform_frozen (= L4 field) / val_deform | 0.167 / 0.189 |
| r2_cfgB smooth0.5 (15000) | L2 | val_deform | 0.287 |
| r1_cfgB S10 frozen (2500) | L1 | val_deform_frozen4 (L4) / _frozen (L4+L2) / val_deform | **0.110 / 0.120 / 0.124** |

### 1.3 Interpretation and ranking

- **L4 dominates.** Measured on the 1 mm grid, the cfgB L4 field alone already has 0.110 of the final
  0.124 mm (89 %). The L2 sweep fits this: raising L2 smoothness from 4 to 12 only moves the Mindboggle
  L4+L2 error from 0.487 to 0.426 mm, and it levels off toward about 0.40 mm. That is the measured config-A
  L4-only value. Unverified for the cfgB L4 on Mindboggle: no `--levels 4` prediction exists for that
  checkpoint.
- **What kind of error.** With T = 7 and max |v| = 7.5 L4 voxels (`predict.json` `stats.max_abs_v.l4`),
  each squaring step starts from v/128 < 0.06 voxel, so the time-step error is negligible. The error
  comes from trilinear resampling inside the squarings. It scales with grid spacing² × field curvature,
  and the same error in voxels costs four times more mm at 4 mm than at 1 mm. In integration-grid voxels,
  our L4 (about 0.10) and MultiMorph (0.115) are equal. Part of the L4 value measured at L4 (0.282 mm)
  is measurement interpolation, because the same field measures 0.167 at L2 and 0.110 at L1. Even so,
  0.110 mm of genuine inconsistency remains at 1 mm (unverified split between the integration error and
  the measurement error at each grid).
- **Accumulation.** Each level adds its own exp/exp(−) mismatch, carried through the composition
  (ψ∘φ = exp(−v₁)∘[up(U₂⁻¹)∘up(U₂)]∘exp(v₁)). Config A shows this clearly (0.40 → 0.62 → 0.77 mm).
  For ours, L2 and L1 at smoothness 10 add little: 0.110 → 0.124 on LUMIR, and on Mindboggle ≈ 0.43 →
  0.485 mm.
- **(c) upsampling** of the forward and inverse separately is a second-order effect at the 2 mm / 1 mm
  scale, where the upsampled fields are smooth. It is part of the "measured at a finer grid" drop above
  and not separable with existing data (unverified, small).
- **(d) centering** subtracts a common field from all v_i before exp. It does not change exp(v_i) versus
  exp(−v_i) consistency.
- **Training val vs Mindboggle.** The main reason for the difference is the mask: all voxels versus the
  brain (30 %). 0.124 / 0.30 ≈ 0.41 mm compares with 0.485 mm measured, and 0.110 / 0.30 ≈ 0.37 mm
  compares with 0.40 mm for the config-A L4. Domain shift (Mindboggle vs LUMIR) and N = 101 vs 4-16 explain
  the rest (unverified).
- **Where the error sits:** max 8.8 mm. Not measured per region; the probe below bins it by distance to
  the mask border and by |fwd|.

## 2. Centrality

- **Definition, evaluation:** `centrality_mean_mm` = mean over the atlas mask of |mean_i fwd_i| (mm), with
  fwd the final composed atlas→subject displacement (`evaluation/evaluate.py` l.~394-399).
- **Definition, model:** per level, v_i ← v_i − mean_j v_j in log space (`models/diffeo.py`
  `center_velocities`; streaming exact over N = 101). Nothing constrains the mean of the composed
  displacement.
- **Why non-zero is expected:** exp(v) = v + ½(Dv)v + O(v³). The first order cancels in the group mean,
  but the second-order mean ½ mean_i (Dv_i)v_i does not. For composition, compose(U, u) ≈ U + u + (DU)u,
  and mean_i (DU_i)u_i ≠ 0 because each level's correction u_i is correlated with the previous level's
  field U_i of the same subject. With |v| of a few mm and |Dv| ≈ 0.1-0.2, the expected size is 0.1-0.4 mm.
  This matches the measurements (unverified analytically per voxel).
- **Measured** (table 1.2): one level alone gives 0.28-0.30 mm (L4 or L1 alone, cfgA), and each added level
  adds to it (cfgA 0.30 → 0.51 → 0.63). Ours: L4+L2 0.372 → full model 0.420. Smoothness at L2 lowers it
  slowly (S4 0.423 → S12 0.364), as it does for the inverse error. MultiMorph (one level) 0.186, ANTs
  0.316. `mean_logjac_abs`: ours 0.144, MultiMorph 0.118, ANTs 0.052.
- **Training metrics cannot show it:** `val_deform/centrality_mm` is |mean v| of the centered residual
  (6e-8 at L4, 1e-8 at L1, `runtime/metrics.py` l.~239). `raw_centrality_mm` is the head output before
  centering: 1.09 mm at L4, 1.02 at L2, 0.35 at L1. Neither describes the composed map. The N = 101 vs
  4-16 difference does not affect the centering itself, which is exact. It may affect the group-mean
  features (distribution shift, unverified).

## 3. Subcortical Dice

Per-label template Dice (ours / MultiMorph / ANTs; `per_label.csv`). The worst labels relative to
MultiMorph:

| label | structure | in train18? | ours | MM | ANTs | Δ vs MM |
|---|---|---|---|---|---|---|
| 254 | CC_Mid_Anterior | no (in WM) | 0.677 | 0.737 | 0.739 | −0.060 |
| 253 | CC_Central | no | 0.642 | 0.692 | 0.681 | −0.050 |
| 62 | R vessel | no | 0.496 | 0.527 | 0.532 | −0.032 |
| 252 | CC_Mid_Posterior | no | 0.654 | 0.681 | 0.663 | −0.027 |
| 26 | L accumbens | yes | 0.660 | 0.679 | 0.687 | −0.019 |
| 255 | CC_Anterior | no | 0.764 | 0.782 | 0.794 | −0.018 |
| 41 / 2 | R / L cerebral WM | yes (L/R merged) | 0.881 / 0.881 | 0.894 / 0.893 | 0.855 | −0.013 / −0.012 |
| 52 | R pallidum | yes | 0.804 | 0.816 | 0.825 | −0.012 |
| 77, 85 | WM hypo, optic chiasm | no | 0.279, 0.558 | 0.291, 0.570 | | −0.012 each |
| 63, 31, 251 | choroid plexus R/L, CC_Posterior | no | | | | +0.026, +0.019, +0.014 (ours better) |

Groups (`sub_groups.py`):

| group | n | ours | MM | ANTs | Δ | share of the summed gap |
|---|---|---|---|---|---|---|
| all subcortical | 42 | 0.7274 | 0.7348 | 0.7309 | −0.0074 | 100 % (−0.310) |
| corpus callosum 251-255 | 5 | 0.706 | 0.734 | 0.734 | −0.028 | 45 % |
| other labels absent from train18 (30, 31, 62, 63, 77, 80, 85) | 7 | 0.414 | 0.416 | 0.411 | −0.002 | 4 % |
| labels present in train18 | 30 | 0.804 | 0.809 | 0.805 | −0.005 | 50 % |
| subcortical without CC | 37 | 0.730 | 0.735 | 0.731 | −0.005 | |

- Label mapping (`preprocessing/luts.py` `TRAIN18_CLASSES`): the 18 classes merge left and right and merge
  all cortex. They hold only SynthSeg structures. SynthSeg does not output the CC (the docstring notes that
  1004/2004 corpuscallosum is listed but never produced), vessel, choroid plexus, hypointensities or optic
  chiasm, so those voxels sit inside WM / ventricles / background in the training labels. Mindboggle's
  evaluated list (labels present in all subjects) contains all of them. The CC segments are cut out of the
  WM along the anterior-posterior axis by FreeSurfer. Our Dice loss sees CC voxels as WM interior, so only
  LNCC (and the neighbouring ventricle and cortex labels) aligns them. MultiMorph's training labels
  (SynthSeg, 30 labels) also lack the CC, so why MultiMorph and ANTs do better there is unverified. Both
  rely more on intensity than on labels.
- Transfer Dice (atlas segmentation pulled into the subject by inv) over all subcortical labels: ours
  0.732 vs MM 0.728, so ours is better. The leave-one-out majority vote gives 0.724 vs 0.731. The template
  gap therefore does not come from how the atlas segmentation is built.
- Subcortical Dice is flat across L2 smoothness (0.710-0.714) because the gap sits in structures that the
  label loss does not define. L1 adds +0.014 (0.7135 → 0.7274).
- In-progress label sets (read only): `synthfine` = SynthSeg --parc, 103 ids (L/R split, 35 DK parcels
  per hemisphere, the same table as `synthseg`), exported at all levels with nested majority so that
  LUT(synthfine) == train18 (`preprocessing/luts.py` docstring, `preprocessing/add_labels.py`). In the
  synth-contrast design it is used only for image synthesis; losses stay on train18. It adds left/right
  and cortical-parcel boundaries. It adds no CC, vessels, choroid plexus or chiasm.

## 4. Ranked change options (none implemented)

### Inverse error

| # | change | expected effect | cost | verify |
|---|---|---|---|---|
| I1 | Inference post-processing: refine the final inverse at 1 mm with the fixed-point iteration w ← −U1(y + w), starting from w₀ = U1_inv, 5-10 iterations (new opt-in `predict.py --refine-inverse K`) | inv_err toward ANTs-like values (expected < 0.05 mm, unverified); pairwise maps (which use inv_j) become consistent too; forward map, Dice and atlas unchanged; transfer Dice may change slightly | small: about 10-20 `warp_field` calls per subject at 1 mm, seconds on GPU | First run the offline probe (below) on the existing fields: before/after inv_err and iteration trace. Then a Mindboggle predict + eval with the flag |
| I2 | Inference: upsample the L4 (and L2) **SVFs** to 1 mm and integrate exp(±v) there, then compose at 1 mm (keep each level's v_k; currently u_k is integrated at the level grid and upsampled) | removes most of the L4 discretisation share (89 % on LUMIR); exp(−v) semantics kept; slightly changes the forward map, and so the L2/L1 inputs (by about the inconsistency, < 0.5 mm), so Dice needs a recheck | medium: new streaming path in `inference/streaming.py` / `runner.py`; extra 1 mm integrations (2 × 7 composes per level per subject) | `--levels 4` Mindboggle prediction with and without, then the full cascade; compare inv_err, Dice and folding |
| I3 | Recipe: higher L4 smoothness (currently 0.5) in the planned `r_cascade.sh` chains (S4 is a required parameter there) | lower L4 curvature, so lower inverse error and centrality. The L2 sweep shows the trend (S4 → S12: 0.487 → 0.426 mm) but saturates against the fixed L4 share. Size at L4 unverified; may cost cortical Dice, as at L2 | no extra cost if folded into the chains already planned (L4 ≈ 14 h + L2 ≈ 24 h + L1 ≈ 9 h) | LUMIR `val_deform_frozen4/inv_consistency_mm` at L1, plus Mindboggle `--levels 4` |
| I4 | More squaring steps (T 7 → 9) | ~none (time-step error already negligible) | trivial | not recommended |

Also worth adding as a diagnostic: a brain-masked `inv_consistency_mm` in training validation (LUMIR
label mask > 0), so training numbers can be compared with Mindboggle.

### Centrality

| # | change | expected effect | cost | verify |
|---|---|---|---|---|
| C1 | Inference post-processing (ANTs-style shape update): m = mean_i U1_i (one streamed pass), then φ_i ← φ_i∘(id − m) and φ_i⁻¹ ← (id + m)∘φ_i⁻¹; 1-2 rounds (or use exp(∓m) for exact diffeomorphisms) | centrality from 0.42 to near 0 (unverified); Dice/atlas essentially unchanged (common warp of about 0.4 mm); inverse error barely changed | small (one extra pass over N plus two composes per subject) | probe `center` mode on the existing fields (below), then a predict flag |
| C2 | Training validation: add `val_deform/disp_centrality_mm` = \|mean_i U_i\| of the composed map (brain mask) | makes the problem visible during training (no model change) | small | appears in metrics.jsonl |
| C3 | Training: re-center the composed map at every cascade level (subtract the group mean of the composed displacement before the next level) | could lower centrality inside the model; changes the next level's inputs, so retraining is needed; benefit over C1 unclear | high (code + full retrain) | not recommended before C1 |

### Subcortical Dice

| # | change | expected effect | cost | verify |
|---|---|---|---|---|
| S0 | Reporting: give subcortical Dice also over the train18-defined structures (30 labels; ours 0.804 vs MM 0.809 vs ANTs 0.805) and list the CC separately | honest description; no model change | none (`--labels` file for `python -m evaluation`) | eval with a label list |
| S1 | Loss labels with the L/R split (SynthSeg structures without merging, about 33 classes; available now via `labels_synthseg` / `synthfine`) | adds a midline constraint (L/R WM and cortex boundary through the CC region) and the L/R subcortical boundaries; effect on the CC unverified; may also help WM/accumbens/pallidum (−0.005 group) | retraining; one-hot memory at L1 grows about 19 → 34 classes (the one-hot [4,160,224,192,C] was a level-1 OOM point), so lean losses / per-step label sampling are needed at L1 | short L2 A/B on LUMIR, then Mindboggle `--levels 4,2` per-label |
| S2 | A CC label source (FreeSurfer `mri_cc` on aseg-like inputs, FastSurfer's CC module, unverified) exported as an extra label (Jay's data pipeline) | directly trains the 5 CC segments (45 % of the gap) | data processing by Jay, then retraining | per-label CC Dice |
| S3 | Higher LNCC weight or smaller window at L1 (the CC is an intensity-defined thin structure) | unverified; may help the CC, may raise folding | one L1 run (≈ 9 h) | per-label CC Dice + folding |

## 5. Offline probe for Jay (CPU only; not run)

`probe_inverse_centrality.py` (in this folder; a copy is at
`/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/scratch/agent/invcen/`). It reads one
run's written fields and the processed images, then:

- (inverse) for 10 subjects: inv_err before and after 10 fixed-point iterations (option I1), the
  iteration trace, and the error binned by distance to the atlas-mask border and by |fwd|;
- (center) for all subjects: centrality before and after 2 re-centering rounds (option C1).

It writes JSON only. Expected runtime < 1 h, about 6 GB RAM. Suggested command (basic partition, not
submitted):

```bash
P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
SIF=/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif
sbatch -A qtim -p basic -N 1 --ntasks-per-node=1 -c 2 --mem=24G --time=02:00:00 \
  --job-name=bgr-invprobe --output=$P/logs/invprobe-%j.out --error=$P/logs/invprobe-%j.out \
  --wrap "singularity exec -B /autofs $SIF python $P/scratch/agent/invcen/probe_inverse_centrality.py \
    --run-dir $P/results/predict_mindboggle/r1_cfgB_sm10_s2500_bnfreeze_true_deform \
    --processed-dir $P/data/processed/mindboggle --out $P/scratch/agent/invcen/probe_r1_cfgB_sm10.json \
    --n 10 --iters 10 --rounds 2"
```

To run the same reference on MultiMorph, add `--fields-dir $P/baselines/multimorph/mindboggle_results`
and set `--run-dir` to the same folder. To localise the L4 share for the current cfgB model directly, a
`predict.py --levels 4` Mindboggle run of the r1_cfgB checkpoint (rtx6000, about 35 min, as in the
predict-levels ablation) would give the cfgB L4-only inverse error and centrality.

## 6. Conclusions

- No bug was found in composition order, conventions, resampling to the Mindboggle grid, or the
  evaluation.
- The inverse-error gap comes from integrating the L4 SVF on a 4 mm grid. In grid voxels the error is the
  same as MultiMorph's, but our coarsest level's voxels are twice as large as MultiMorph's integration grid
  (4 mm vs 2 mm). Smoothness at L2/L1 cannot remove it.
- Centrality is a second-order effect of log-space centering plus cascade composition, and it grows with
  each level. It is easy to remove after prediction.
- Half of the subcortical gap is the corpus callosum, which no training label defines. On the structures
  we train on, we are within 0.005 of MultiMorph and level with ANTs.
- Recommended order: run the probe (I1 + C1, CPU) → if confirmed, add opt-in `--refine-inverse` /
  `--recenter` to `predict.py` (small) → consider I2 and a higher L4 smoothness for the cascade chains → S0
  reporting now, S1/S2 as data and recipe decisions.
