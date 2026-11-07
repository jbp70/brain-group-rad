# Milestone 10: 8-flip test-time augmentation (TTA) — design

Version 2 (2026-11-06). Design only: no source changes, no commits, no cluster jobs. Starting point: the milestone-10 addendum of `.agents/tasks/network-design/design.md` (after the milestones table, commit c733468). Numerical checks of the conventions used below: `.agents/tasks/tta/check_flip_signs.py` (signs, flips, fields; `tf_gpu`) and `.agents/tasks/tta/check_gauge_free.py` (v2 linear averaging; numpy + scipy, `bgr_preproc38`); outputs in section 2.7. Changes against version 1 and the response to every review finding: changelog at the end.

## 0. Overview

TTA runs every TTA-enabled cascade level 8 times, once per group-level flip combination F ∈ {diag(±1, ±1, ±1)} applied to the whole group (the same flip for every image, as in training). Each pass's output is mapped back to the unflipped frame ("unflipped"), the 8 results are averaged, and the average then goes through the same downstream operations as a non-TTA output: group centering (linear) or centering + scaling-and-squaring integration (deform), composition with the previous level, and use as the pre-warp of the next level. Linear outputs are averaged as gauge-free residuals (the per-flip centered and composed result divided by the previous level's result), re-encoded in the model's own parameter coordinates (10-D rotation matrix, translation, log-scales, shears); deform outputs are averaged as raw SVFs, which the per-level SVF centering makes gauge-free exactly. TTA lives in the streamed prediction path (`predict.py run`, `inference/runner.py`, `inference/streaming.py`) and in a new `predict.py linear-eval` subcommand that measures rigid / affine errors against known synthetic misalignments. It is off by default (`--tta none`); with `none` the existing code path runs unchanged, so outputs are bit-identical. Evaluation on Mindboggle-101 runs one checkpoint with and without TTA on the same 101-subject group and compares per-subject metrics with paired statistics (`scripts/compare_tta.py`). Technology stack (locked): Python 3.8.10, TensorFlow 2.10, numpy 1.21, scipy 1.4, nibabel 3.1 inside `tf10_nv.sif`; no new dependencies.

## 1. Facts from the code that the design depends on

| Item | Where | Fact |
|---|---|---|
| Linear head outputs | `models/group_net.py` `LinearDecoder.call`, `models/transforms.py` | rigid params [13] = 10-D rotation `[a00, a01, a02, a03, a11, a12, a13, a22, a23, a33]` (upper triangle, row-major, of a symmetric 4×4 matrix Q over the quaternion (w, x, y, z)) + `t_x, t_y, t_z` in mm (already multiplied by `translation_scale`); affine params [19] = rigid 13 + `[s_x, s_y, s_z, k_xy, k_xz, k_yz]`. `rigid_matrix(p) = [R | t]`, `affine_matrix(p) = [R K | t]`, K = `scale_shear_matrix` = [[e^s_x, k_xy, k_xz], [0, e^s_y, k_yz], [0, 0, e^s_z]] (raw entries in the strict upper triangle). `A_raw = matrix(params)`; `A = GroupCentering(A_raw)`. |
| What training constrains | `runtime/run.py` 1104 (train), 1333-1335 (validation) | The linear losses see only `A = out["A"]` or `CS.compose_linear(out["A_raw"], A_prev, model.centering)` = Center(A_raw A_prev). A common left factor C on the whole group (A_raw,i → C A_raw,i) is a zero-gradient direction ("gauge"): set by initialisation, head biases and weight decay, never trained to be flip-equivariant. |
| Centering | `models/transforms.py` `barycenter`, `center_group` | mu ← mu expm(mean_i logm(mu⁻¹ A_i)), from mu = I, `centering_iters` iterations (default 4); `center_group` = mu⁻¹ A_i. The converged group-exponential barycenter is left-equivariant (barycenter(C A) = C barycenter(A)), so Center removes the gauge; with 4 iterations up to the fixed-point residual (measured: 1e-7 mm, section 2.7 E5). |
| Rotation read-out | `decode_10d`, `encode_10d` | `decode_10d`: the quaternion is the adjugate column of Q with the largest diagonal entry, normalised (equals the smallest-eigenvalue eigenvector when Q is rank-deficient, e.g. Q = I − q qᵀ). `encode_10d(R)` = 10-D entries of I − q qᵀ (Shepperd quaternion), `decode_10d(encode_10d(R)) = R`. Ported from Jay's rigid project. |
| Jay's TTA | `reference/1_rigid_registration_src/models/reg_net.py` 168-180, 636-660 | Same 13-entry layout. Per flip: forward pass on the flipped input, multiply the 13 outputs by the flip's sign vector, average the 8 vectors, decode with the same 10-D decoder. Combined flips = elementwise products of the per-axis vectors. His network's raw output was supervised directly (no group centering), so it had no gauge freedom. |
| Coordinates | `models/transforms.py` docstring | Centered mm, x = (p − (n−1)/2)·s. Array reversal of axis a is exactly x_a → −x_a. Flipping the group turns T_i into F T_i F; `apply_flip` in `data_loading/postprocess.py`. |
| Training flips | `data_loading/postprocess.py` `sample_flip_axes`, `apply_flip`; `runtime/args.py` 365, 947 | One flip state per group, each axis with `--flip-prob` (default 0.5; forced to 0 with `--overfit-one-group`); images and labels reversed; targets F T F. Frozen cascade stages in training also see the flipped group. All real runs so far used the default, so the 8 flips are in-distribution; TTA checks this per stage from the configs (section 7.4). |
| Deform fields | `models/diffeo.py` | v, u in voxel units of the level, pull warp warp(I, u)(p) = I(p + u(p)); centering v_i − mean_j v_j; exp by scaling and squaring, T = `integration_steps` (7); u_inv = exp(−v). Cascade (u, u_inv) = compose_cascade(up(U), up(U_inv), u_k, u_inv_k). |
| Streamed engine | `inference/streaming.py` `StreamedLevel` | `linear(n, input_fn, task)` runs passes P0-P6 then `linear_decoder` once on all N and returns A_raw; `deform(n, input_fn, prev_fn, sink)` runs P0-P11 (stores v_raw, accumulates its group sum), then P12 (center, integrate, compose with the previous level, sink). Group means are exact over all chunks. |
| Runner | `inference/runner.py` `StreamedRunner` | Per level: input_fn = pool(x1) or pre-warp with the previous level's result; A = `tf.cast(model.centering(a_raw), float32)` at the first level, else `CS.compose_linear(a_raw, a_prev, model.centering)` (float32); deform levels chained through prev_fn. `_linear_sel` / `_deform_sel` handle `--levels` plans. For `affine+deform`, `_resample` stores `src_affine` and `_deform` / `_deform_sel` (lines ~317-325 / ~495-503) wrap the last level's `input_fn` so that `store.free(SRC_AFFINE)` runs after the last P0 chunk. `UnchunkedRunner` runs each level stage (frozen chain included) as one tf.function. |
| Host store estimate | `inference/store.py` `estimate_store_gb` | Max over levels of a per-image term; for `affine+deform` the deform term already includes the full-resolution `src_affine` (4 B per level-1 voxel) at every deform level. `choose_store` picks memory when the estimate is < 0.7 × budget. |
| CLI / records | `inference/cli.py`, `inference/predict.py` | `run` already has `--tta {none}` (placeholder); `predict.json` key `tta`; `PREDICT_KEYS` lists every key (`tests/test_predict.py` asserts `set(rec) == set(PR.PREDICT_KEYS)`); `_peek_chain_configs` reads the checkpoint's config.json and, for a level-1 run, its level-2 source config. |
| G (global transform) | `inference/predict.py` `GLOBAL_TRANSFORM` | Training augmentation only, never a model output: nothing to unflip at inference. |
| Grids | `data_loading/params.py` | Level shapes 160×224×192 / 80×112×96 / 40×56×48: every axis length is even, so flips commute with the 2×2×2 pooling (`models/cascade.py` docstring), and the trilinear ×2 upsampling of fields (half-pixel, edge-replicated) is symmetric under reversal. |
| Mindboggle today | `slurm/predict_mindboggle.sh`, `predict_mindboggle_multi.sh`, `eval_mindboggle.sh` | 101 subjects in one group, `--branches deform` (processed Mindboggle is MNI-affine aligned), streamed chunks l4 32 / l2 8 / l1 2, checks 1a-1c (1a is a gate, ~10 min), then `predict.py run`, `evaluation check-convention`, template-only `evaluation run --pairs none --workers 3` into `$OUT/eval_template`; all-pairs evaluation separately on `basic` (8 CPUs, 96G, 12 h). Predict job: rtx8000, 3 CPUs, 96G, 5 h. |
| Measured cost (predict.json on the cluster) | `results/predict_mindboggle/r1sw_lr1e-4_s3000_deform/fields/predict.json` | Level-1 checkpoint, N 101, AMP: phases deform:l4 51 s, deform:l2 115 s, deform:l1 1049 s of which writing (inside the level-1 sink) 886 s → level-1 model time ≈ 163 s; total 1290 s; TF GPU peak 11.0 / 10.9 / 8.5 GB (l4 / l2 / l1); host store peak 31.7 GB, process RSS 42.8 GB. Level-2 checkpoint (`r2_cfgB_smooth10_s15000_deform`): deform:l4 57 s, deform:l2 958 s incl. 877 s writing; total 1074 s. |

## 2. Linear TTA

### 2.1 Per-flip unflip of our parameters (derivation)

Let F = diag(f_x, f_y, f_z) (f = ±1) and the group input flipped by F (array reversal of the flipped axes). A level model that is flip-equivariant predicts A^F = F A F for every image. Unflipping is A = F A^F F, F being its own inverse; on 4×4 matrices this is an exact sign change of entries (no rounding).

In parameters: F [R K | t] F = [(F R F)(F K F) | F t].
- Translation: t → F t, sign f_a per component.
- K: (F K F)_ab = f_a f_b K_ab. The diagonal exp(s_a) is unchanged (log-scales sign +1); shears k_xy, k_xz, k_yz get f_x f_y, f_x f_z, f_y f_z. F K F is still upper-triangular with positive diagonal, so the QR parameterisation is closed under the conjugation and the QR factorisation of F L F is (F Q F)(F K F) (uniqueness with positive diagonal).
- Rotation: F R F for unit quaternion (w, x, y, z) is the quaternion S_q q with S_q = diag(1, f_y f_z, f_x f_z, f_x f_y) (the global sign of q is irrelevant). For the 10-D matrix Q this is Q → S_q Q S_q, i.e. entry a_ij gets sign s_i s_j. This holds for any symmetric Q: adj(S Q S) = S adj(Q) S (det S² = 1), the adjugate diagonal is unchanged, so `decode_10d` picks the same column and returns ±S_q q, which decodes to F R F.

Resulting sign vectors σ_f (13 rigid entries, then the 6 scale/shear entries):

| Flip | a00 a01 a02 a03 a11 a12 a13 a22 a23 a33 | t_x t_y t_z | s_x s_y s_z | k_xy k_xz k_yz |
|---|---|---|---|---|
| x | 1 1 −1 −1 1 −1 −1 1 1 1 | −1 1 1 | 1 1 1 | −1 −1 1 |
| y | 1 −1 1 −1 1 −1 1 1 −1 1 | 1 −1 1 | 1 1 1 | −1 1 −1 |
| z | 1 −1 −1 1 1 1 −1 1 −1 1 | 1 1 −1 | 1 1 1 | 1 −1 −1 |

The 13-entry rows equal Jay's reference patterns exactly (`param_factor_{x,y,z}_10d`), so our 13-output order and sign behaviour match his (verified numerically, C3; and by the reviewer by hand). Combined flips are elementwise products. The flip of all three axes leaves the rotation unchanged (S_q = I) and negates t. σ_f is its own inverse. Hence for every parameter vector p: `matrix(σ_f ⊙ p) = F matrix(p) F` (C1, C2: max |diff| 0 in float64 for general, non-rank-deficient 10-D inputs), and for every affine M with positive determinant `encode(F M F) = σ_f ⊙ encode(M)` for the re-encoding of 2.3 (E2: max |diff| 0).

`inference/tta.py` derives σ_f from F with these formulas (`linear_flip_signs(flip, task)`, float64 numpy); a unit test asserts the per-axis 13-entry vectors equal Jay's literal patterns.

### 2.2 Why the raw outputs must not be averaged (review finding 1)

Version 1 averaged σ_f ⊙ p_f, the unflipped raw head outputs, as Jay's net does. That is only invariant when the raw outputs themselves are flip-equivariant. Our losses see Center(A_raw A_prev) only (section 1), so training can at best make the centered outputs equivariant; the raw outputs then are unflipped as G_f A_raw,i with a flip-dependent common factor G_f (a constant head bias B unflips to F B F, different for every flip unless B commutes with all F). The 10-D, translation and K averages are nonlinear in the product G_f A_i, so the mean does not factor as (mean gauge) · A_i, and centering afterwards cannot remove the gauge: pairwise maps A_j⁻¹ A_i change although the model has zero error. Measured: 0.48 mm maximum pairwise point RMS in the reviewer's demo (`review_gauge_demo.py`, 10° / 5.4 mm gauge), 0.18 mm in `check_gauge_free.py` E3 (gauges up to 15° / 6 mm, affine). The gauge size of real checkpoints is unknown; version 2 averages gauge-free quantities instead, and logs the gauge (`gauge_deg`, `gauge_mm`, section 5) so its size becomes visible.

### 2.3 Averaging (gauge-free residuals)

For level k, task τ ∈ {rigid, affine}, image i, flip f (8 flips, fixed order, identity first: `itertools.product((False, True), repeat=3)` over (x, y, z)); A_prev,i = the previous level's TTA result (float32), or I at the first run level:

1. Run the level on the flipped level input (pre-warped with F A_prev F, i.e. `flip_np` of the unflipped pre-warped input, since pre-warping commutes with flips, C5): A^F_raw,f (float32, the unchanged `eng.linear`).
2. Unflip the raw matrices: M_f,i = F A^F_raw,f,i F (exact sign changes).
3. Per-flip composed, centered result, the same op as the non-TTA path: A_f = `model.centering(M_f)` (first run level) or `CS.compose_linear(M_f, A_prev, model.centering)` (float32). Center removes G_f (section 1).
4. Gauge-free residual in float64 (numpy): R_f,i = A_f,i · A_prev,i⁻¹ (= A_f,i at the first run level).
5. Re-encode R_f,i = [L | t] (`inference.tta.encode_linear(R, task)`, float64): L = Q K by numpy QR with the sign fix of `runtime.metrics.qr_factors` (positive diagonal of K, Q a rotation since det L > 0); rotation entries = the 10-D entries of I − q qᵀ for the quaternion q of Q (`tfm.encode_10d(Q)` in float64; its sign is irrelevant); translation t; affine adds s = log diag(K) and the RAW upper-triangle entries (K01, K02, K12) in the `scale_shear_matrix` convention. Note: `qr_factors` returns shear ratios K01/K11, K02/K22, K12/K22, which are a different parameterisation; `encode_linear` returns K entries so that `affine_matrix(encode_linear(R)) = R` (E1: 3.6e-16). For the rigid task K is I up to float32 rounding and is dropped (the 13 entries only); the deviation max|K − I| is logged at DEBUG.
6. Average the 13 / 19 entries over the 8 flips in float64 (fixed flip order): p̄_i.
7. Rotation read-out of the averaged 10-D matrix Q̄ (section 2.4): q̄ = eigenvector of the smallest eigenvalue of Q̄ (float64 `numpy.linalg.eigh`), replace the 10-D entries of p̄ by those of I − q̄ q̄ᵀ.
8. Decode R̄_i = `tfm.rigid_matrix(p̄)` / `tfm.affine_matrix(p̄)` in float64 (`decode_10d` is exact on the rank-deficient I − q̄ q̄ᵀ), cast to float32.
9. Centering and composition once, exactly as the non-TTA path: Ā = `tf.cast(model.centering(R̄), float32)` at the first run level, else `CS.compose_linear(R̄, A_prev, model.centering)`. Ā is the level's TTA result and the next level's A_prev.

Identity-only flip list (`IDENTITY_ONLY`, tests) takes today's two lines unchanged (raw → centering / `compose_linear`, no re-encoding), so it is bit-identical to no TTA (T5).

Cost: 8 extra centerings of N matrices per level and task (float32 TF, a few ms at N 101) and N × 8 float64 QR / eigh calls (numpy, < 1 s at N 101).

### 2.4 Rotation averaging, sign ambiguity and read-out

After step 5 every flip's rotation is encoded as I − q_f qᵀ_f, which is quadratic in q_f: q and −q give the same entries, so the arithmetic mean has no hemisphere / sign-alignment problem (a quaternion mean would need one; a matrix mean leaves SO(3)). The mean is Q̄ = I − W with W = (1/8) Σ_f q_f q_fᵀ (symmetric PSD, trace 1). Its smallest-eigenvalue eigenvector is the dominant eigenvector of W, which is Markley's quaternion average (the chordal L2 mean of the rotations; Markley et al. 2007). When all flips agree, W = q qᵀ, Q̄ = I − q qᵀ and q̄ = ±q exactly (up to eigh rounding, ~1e-16).

Read-out choice. Alternatives: (a) `decode_10d` of Q̄ (the model's adjugate column read-out); (b) the exact smallest-eigenvalue eigenvector (float64 eigh). Chosen: (b). When the flips disagree, Q̄ has full rank (smallest eigenvalue 1 − w_max > 0) and the adjugate column read-out is no longer the eigenvector (adj Q̄ = det(Q̄) Q̄⁻¹, whose column k is a single power-iteration-like step from e_k), so (a) would be an approximation that depends on which column wins; (b) is the defined Markley mean. Version 1 chose (a) to keep identity-only TTA bit-identical; version 2 keeps that through the separate identity-only path (2.3). Projecting q̄ back to I − q̄ q̄ᵀ before `rigid_matrix` / `affine_matrix` keeps the model's decoder in the path (exact on rank-deficient input). Diagnostics: `readout_dev_deg` = angle between `decode_10d(Q̄)` and the eigh rotation (how much (a) would have differed), `rot_gap` = relative eigengap (λ₂ − λ₁) / λ_max of Q̄ (small = ambiguous average). Weighting: plain mean, as in Jay's net (open question 7).

Translations, log-scales and shears are averaged arithmetically in the same residual coordinates. Under (centered) equivariance all 8 values coincide, so the coordinate choice only matters for disagreeing flips; the parameter mean keeps the decoded K a valid upper-triangular, positive-diagonal factor (always invertible, positive determinant).

### 2.5 Centering and composition

Centering commutes with flip conjugation, Center({F A_i F}) = F Center({A_i}) F (logm and expm commute with conjugation by an orthogonal F, so every barycenter iteration does; C4: max |diff| 0 in float64). The averaged group {R̄_i A_prev,i} is not centered (a mean of centered groups is not centered in the barycentric sense), so step 9 re-centers it, exactly where the non-TTA path centers. The central-space constraint (barycenter of the group = I, up to the `centering_iters` fixed point as without TTA) therefore holds for the TTA output, and pairwise maps A_j⁻¹ A_i follow as before. Pairwise maps are independent of the left factor of Center, so they are exactly gauge-free (E3: 3e-14 mm).

### 2.6 Consistency

Definition. A level model is centered-equivariant if its per-flip composed and centered results agree after unflipping: A_f = A_id for all f (with A_f as in 2.3 step 3), i.e. Center(F A^F_raw F · A_prev) = Center(A_raw A_prev). This is the only equivariance training can teach (the gauge is free), and it is weaker than raw-parameter equivariance.

Claim 1 (TTA == non-TTA for a centered-equivariant model). R_f = A_f A_prev⁻¹ is then the same for all f, so its encoding is the same for all f, the float64 mean equals it exactly, the eigh read-out returns ±q (I − q̄ q̄ᵀ = I − q qᵀ to ~1e-16), and R̄ = decode(encode(R)) = R up to float64 rounding (E1: 3.6e-16) and the float32 cast. Then Ā = Center(R A_prev) = Center(A_id): the centering of an already centered group, which returns it up to the 4-iteration fixed-point residual (E5: centered translation 1e-7 mm; pairwise maps 3e-14 mm). In float32 the end-to-end tolerance is < 1e-4 mm point RMS for A and exact-to-rounding (< 1e-5 mm) for pairwise maps (T7). By induction over levels, the next level's inputs are the same up to these tolerances. This replaces version 1's Claim 1, which assumed raw-parameter equivariance.

Claim 2 (exact equivariance of the TTA pipeline, any model). TTA(flip_g x) = F_g TTA(x) F_g. The 8 flipped versions of flip_g x are the 8 flipped versions of x in another order, and their unflipped results are conjugated by F_g; centering commutes with conjugation (2.5); `encode_linear(F M F) = σ_g ⊙ encode_linear(M)` (E2, exact), and the arithmetic mean, the eigh read-out (S_q conjugation of Q̄) and the decoder are covariant under σ_g; pre-warping commutes with flips (C5). So the property propagates through the cascade (E4: 6e-15 with a random, non-equivariant model). This is the main correctness test with real (random-weight) models: a wrong sign or a wrong spatial convention breaks it (T6).

Identity-only TTA (internal flips list `[(False, False, False)]`) equals no TTA bit-for-bit: it runs the unchanged two-line path (T5).

### 2.7 Numerical verification (done for this design)

`.agents/tasks/tta/check_flip_signs.py`, local `tf_gpu` (TF 2.10, GTX 1080), using `models/transforms.py`, `models/layers.py` and `models/diffeo.py`:

```
C3 axis x/y/z derived == Jay: True / True / True; combined flips == products of Jay's patterns: True
C1 rigid  max |M(p s_f) - F M(p) F| over 8 flips x 64 random (float64): 0.00e+00
C2 affine max |M(p s_f) - F M(p) F| over 8 flips x 64 random (float64): 0.00e+00
C4 max |center(F A F) - F center(A) F| (N 12, float64): 0.00e+00
C5 max |T(flip I, F M F) - flip T(I, M)| (level 4, float32): 7.75e-06
C6 max |v| 4.94 vox; unflip(flip(v)) == v exactly: True; max |exp(v^F) - (exp v)^F| 2.15e-06 vox;
   max |warp(flip I, u^F) - flip warp(I, u)| 3.81e-06
C7 average of 8 unflipped equivariant predictions: float64 accumulation max diff 0.0e+00 (exact);
   float32 running sum max rel diff 1.5e-07 (rounding of 3x, 5x, 7x)
ALL PASS
```

`.agents/tasks/tta/check_gauge_free.py`, local `bgr_preproc38` (numpy 1.21-era API, scipy logm / expm), N 6 affine images, the v2 algorithm of 2.3 in float64 (numpy port of the encoding, eigh read-out, converged centering unless stated):

```
E1 affine encode/decode round trip max |diff|: 3.61e-16
E2 max |encode(F M F) - sigma_f * encode(M)| over 8 flips x 20: 0.00e+00
E3 flip-dependent raw gauge, zero model error: v1 raw averaging pair err 0.183 mm;
   v2 gauge-free pair err 3.40e-14 mm; v2 centered max |diff| 1.42e-14
E4 v2 TTA(flip_g x) vs F_g TTA(x) F_g, non-equivariant model, max |diff|: 6.22e-15
E5 4-iteration centering: v2 pair err 3.02e-14 mm, centered translation max |diff| 1.03e-07 mm
ALL PASS
```

These checks become unit tests (section 8).

## 3. Deformable TTA

### 3.1 SVF unflip

Fields are in voxel indices of the level. For flip F the flipped field is v^F(p) = F v(F p) where F p reverses the flipped array axes (p_a → n_a − 1 − p_a, i.e. x_a → −x_a about the centre (n−1)/2) and the leading F negates the flipped components. If the input group is flipped and the model is equivariant, its output is v^F; unflipping applies the same operation again (an involution): `unflip_field(v, flip) = tf.reverse(v, flipped spatial axes) * sign`, sign = (f_x, f_y, f_z). Consistency with the pull convention: I^F(p + u^F(p)) = I(Fp + u(Fp)) with u^F(p) = F_lin u(Fp); integration commutes with the flip, exp(v^F) = (exp v)^F (C6: 2.2e-6 vox at |v| ≤ 5 vox; scaling and squaring clamps to [0, n−1], symmetric under reversal), and warp(flip I, u^F) = flip(warp(I, u)) (C6: 3.8e-6). The level-input flip is the array reversal `flip_images` used in training (`apply_flip`).

### 3.2 Averaging, centering, integration

Per TTA level, after the 8 passes P0-P11 (one per flip) each produced v_raw,f in the flipped frame:

- r_{i,f} = unflip(v_raw,{i,f}); m_f = unflip(group mean of v_raw,f) (the P11 group sum / N, unflipped);
- v̄_raw,i = (1/8) Σ_f r_{i,f} and m̄ = (1/8) Σ_f m_f (float32 running sums in the fixed flip order);
- P12 unchanged on the average: v_i = v̄_raw,i − m̄, (u_i, u_inv,i) = `exp_and_inverse(v_i, T)` with the checkpoint's `integration_steps` (7), then `compose_cascade` with the previous level when there is one.

Gauge. The deform analogue of the linear gauge is a common field c added to every v_raw,i; the losses see only the centered v, so c is free and flip-dependent. Centering is linear and commutes with unflip, so v̄_raw,i − m̄ = (1/8) Σ_f (r_{i,f} − m_f): the average of the per-flip centered fields, in which every c_f cancels exactly. No re-encoding is needed for the deform branch. m̄ equals the group mean of v̄_raw, so v is group-mean-centered per level exactly as without TTA (central space preserved).

Integrating once after averaging (log-domain average) keeps u a diffeomorphism-by-construction with the exact inverse exp(−v); averaging 8 integrated displacements would neither keep the exact inverse nor the folding guarantees of a single SVF. Pairwise maps (`compose(u_i, u_inv_j)`, the evaluator's compose(inv_j, fwd_i)) and the cascade composition are unchanged.

Accumulation precision: float32 for the fields (host memory, section 6). With a perfectly equivariant model the 8 unflipped fields are identical and the float32 running sum differs from v by at most ~1.5e-7 relative (C7); float64 would make it exact but costs +12 B per voxel per image (+8.3 GB host at level 1, N 101). The identity-only flip list runs the unchanged `deform()` (bit-exact). Claim 2 of 2.6 holds for the deform branch too (to float rounding, ~1e-6 vox).

### 3.3 Engine changes (`inference/streaming.py`)

- Move the P0-P11 body of `StreamedLevel.deform` verbatim into `_v_raw_passes(n, bounds, stats, input_fn) -> sum_v` (stores `v_raw`, returns the float32 group sum); `deform()` = `_v_raw_passes` + the unchanged P12 loop. Same tf.function objects, same call order, so the non-TTA results are unchanged (golden test T9).
- New `deform_tta(n, input_fn, flips, prev_fn=None, sink=None, mask_fn=None)`:
  1. allocate host stores `tta_s1` [n, X, Y, Z, 3] f32 (Σ r), `tta_s2` [n, X, Y, Z, 1] f32 (Σ |r − m_f|², for the spread), `tta_c0` [n, X, Y, Z, 3] f32 (identity-flip centered field, for the shift); `sm` [1, X, Y, Z, 3] f32 on the GPU;
  2. for each flip: `input_fn_f(i0, i1) = flip_np(input_fn(i0, i1), f)` (numpy reversal + `np.ascontiguousarray`, float32); `sum_f = _v_raw_passes(...)`; m_f = unflip(sum_f / n); sm += m_f; one chunk pass with tf.function `_f_tta_acc` (inputs v_raw chunk, m_f, s1, s2 chunks; returns updated s1, s2 and, for the identity flip, c0) back to the stores; free `v_raw`; merge per-flip stats (fp16 maxima: max over flips; chunk boundaries from flip 0; per-flip max |v_raw| finite flag). `input_fn` is called 8 times per chunk, so it must not free its source (3.4);
  3. P12 loop with v_raw chunk = s1 / 8 and mean = sm / 8 through the existing `_f_p12(cascade)` body, sink or store as today (the sink's `v_raw` = the averaged raw field, `v` = the centered average);
  4. spread pass (`_f_tta_spread`, per chunk, with the level mask from `mask_fn`): per image spread and shift scalars (section 5), the voxel spread map pulled to the host only for the p99;
  5. free `tta_s1`, `tta_s2`, `tta_c0`; `last_stats` gets `tta` = {flips, per-flip finite, timing per flip}.
- `StreamedLevel.linear` is unchanged (the linear TTA works on its A_raw output).

### 3.4 Runner changes (`inference/runner.py`)

`StreamedRunner(loaded, store, chunks, lean, record_fp16, plan, tta=None)`; `tta` is `inference.tta.TTAConfig(mode='flip8', levels=(…))` or None; `self._tta_at(lv)` = `tta is not None and lv in tta.levels`. The four call sites `eng.linear(...)` (in `_linear`, `_linear_sel`) and `eng.deform(...)` (in `_deform`, `_deform_sel`) go through two helpers:

- `_linear_stage(lv, eng, model, n, input_fn, task, a_prev)`: if not `_tta_at(lv)`, exactly today's two lines (`eng.linear` then centering / `compose_linear`); else 2.3 steps 1-9: 8 × `eng.linear(n, flipped input_fn, task)`, unflip, per-flip `A_f` (cached tf.function per (level, task) around centering / `compose_linear`), float64 residuals, re-encoding, mean, eigh read-out, decode (cached float64 tf.function around `rigid_matrix` / `affine_matrix`), final centering / `compose_linear`; returns Ā and the per-flip A_f, M_f for the diagnostics (section 5).
- `_deform_stage(lv, eng, n, input_fn, prev_fn, sink)`: `eng.deform(...)` or `eng.deform_tta(..., flips, mask_fn=self._mask_fn(lv))`.

`affine+deform` source lifetime (review finding 2). In `_deform` and `_deform_sel` the free-after-last-P0 wrapper is installed only when `is_last and free_src and not self._tta_at(lv)`. When the last level is a TTA level, `_deform` frees `SRC_AFFINE` right after `_deform_stage` returns (also on the exception path: `try/finally`, so a disk store does not keep the file). The host estimate already counts `src_affine` during every deform level (section 1), so its longer life needs no new term (section 6).

Level mask for the deform spread: m1 = (mean_i 1[src_i > 0]) ≥ 0.5 at level 1 (src = the deform input: raw images for `deform`, the affine-resampled images for `affine+deform`), pooled to each level with `pool_to_coarser` and thresholded at 0.5. It is computed once, at the start of `_deform` / `_deform_sel` and only when some deform level is a TTA level, in one chunked pass over `src` (chunk l1) before the first level's P0; held on the host as uint8 per level (level 1: 6.9 MB).

Linear spread uses per-image second moments S_i = `brain_second_moment(x1_i > 0, 1.0)` of the raw level-1 image i (A_f and Ā both map image-i coordinates); computed once per linear branch when any linear level is a TTA level.

`UnchunkedRunner` gets no TTA: `--mode unchunked --tta flip8` is refused at argument validation (exit 2). Reason: its level-1 stage runs the frozen 4 → 2 chain inside one tf.function, so per-level TTA would need a second implementation of the cascade; the streamed path is the one used for every real run. `check-streaming` keeps validating the engine without TTA.

### 3.5 Level plans (`--levels`)

TTA applies to every run level in `tta.levels` (default: all levels the plan runs). Skipped levels contribute the identity and truncated levels are not run, as today; `--tta-levels` naming a level that the plan does not run is an input error (exit 2).

## 4. Cascade: sequential per-level TTA

Options:
- (A) Sequential per level (chosen): level-k TTA runs on the inputs pre-warped with the TTA-averaged result of the previous levels; its averaged residual is composed with that result.
- (B) Only at the last level: earlier levels single-pass.
- (C) Eight independent full cascades (one per flip) averaged at the end.

Choice: (A). Every level's input then benefits from the variance reduction of the earlier levels, it is exactly the plan's "cascade TTA per level sequentially", and it keeps the averaging in each level's own coordinates (residual parameters, SVF). (C) would have to average composed outputs: the composed deform field is a composition of three SVF exponentials, so it could only be averaged as displacements (no exact inverse, no folding guarantee) or as per-level SVFs conditioned on eight different pre-warps. (B) saves little: level-1 model time is about half of the model time and the coarse levels are cheap (section 6), while coarse-level errors are the largest. `--tta-levels` allows (B) and other subsets for an ablation (open question 5). Frozen stages are unchanged: same weights, BN moving statistics, float32 policy; TTA only changes their inputs (flipped groups, which training also fed them) and averages their outputs.

## 5. Flip-consistency diagnostics

Per image, level and task, from the 8 unflipped per-flip results and the TTA result at that level:

| Column | Linear (rigid / affine) | Deform |
|---|---|---|
| `spread_rms_mm` | sqrt(mean_f point_rms(A_f − Ā, S_i)²), A_f the per-flip composed and centered matrix (2.3 step 3), Ā the TTA result (`runtime.metrics.point_rms`) | sqrt(mean over the level mask of σ²)·spacing, σ²(p) = max(0, S2/8 − \|c̄\|²) with c̄ = v̄_raw − m̄ the centered average: the RMS over flips of the centered SVF deviation |
| `spread_max_mm` | max_f point_rms(A_f − Ā, S_i) | p99 of σ over the level mask, mm |
| `shift_mm` | point_rms(A_id − Ā, S_i): change of the single-pass (identity-flip) result by TTA | sqrt(mean_mask \|c_0 − c̄\|²)·spacing |
| `rot_deg_rms` | RMS_f angle(Q_f Q̄ᵀ), Q = QR rotation factor of the residual (`runtime.metrics.qr_factors`, `rotation_angle_deg`) | – |
| `trans_mm_rms` | RMS_f \|t_f − t̄\| (residual translations) | – |
| `scale_pct_rms`, `shear_pct_rms` | affine only: RMS_f of \|Δ log-scale\| ×100 and \|Δ K_ab\| ×100 (residual K) | – |
| `rot_gap` | relative eigengap (λ₂ − λ₁) / λ_max of the averaged 10-D matrix Q̄ (float64 eigh) | – |
| `readout_dev_deg` | angle between the `decode_10d` rotation of Q̄ and the eigh rotation (2.4) | – |

Per level and task (not per image), linear only: `gauge_deg` = max_f rotation angle of mu_id⁻¹ mu_f and `gauge_mm` = max_f \|translation of mu_id⁻¹ mu_f\|, mu_f = `tfm.barycenter({M_f,i A_prev,i}, centering_iters)` (float64), the left factor that step 3's centering removed for flip f. It shows how large the raw gauge is, i.e. how much version 1's raw averaging would have been off; it does not affect the TTA output.

For the deform branch the SVF spread approximates the displacement spread to first order (v ≈ u for small v); displacement spreads would need 8 integrations per level and are not computed. Outputs: `tta_spread.csv` in `--out` (one row per subject × level × task: `subject, subject_dir, level, task`, then the columns above, empty where not applicable) and a summary in `predict.json` `tta_record.spread` (per level and task: mean, median, p95, max over subjects of each column; plus `gauge_deg`, `gauge_mm`). Logged at INFO per level: mean and max `spread_rms_mm`, mean `shift_mm`, and (linear) `gauge_deg` / `gauge_mm`.

## 6. Memory and time

GPU. Each flip pass is one non-TTA level pass on the flipped group with the same chunk sizes, so the per-chunk GPU peak is unchanged (measured non-TTA peaks at N 101: 11.0 / 10.9 / 8.5 GB at l4 / l2 / l1, chunks 32 / 8 / 2). The extra tf.functions (`_f_tta_acc`, `_f_tta_spread`) work on one chunk of fields: at level 1 with chunk 2, inputs v_raw, s1, c0 (2 × 82.6 MB each) and s2 (2 × 27.5 MB) plus temporaries ≈ 1 GB, below the deform pass peak. Passes run sequentially (the 8 flips are separate groups with their own group means; batching flips would multiply the chunk memory by up to 8 for no speed gain on a GPU already busy at chunk 2). Level 1 on the RTX 8000 48 GB: unchanged ~8.5 GB TF peak; no fixed memory limit needed. The linear centerings (8 per level) are N × 4 × 4 matrices.

Host (per image, per TTA deform level, V voxels of the level): s1 12 V + s2 4 V + c0 12 V = 28 V bytes alive during the level, on top of the existing per-flip v_raw store (12 V, freed after each flip's accumulation). Level 1: V = 6.88 M → 193 MB per image → +19.5 GB at N 101 (level 2: +2.4 GB, level 4: +0.3 GB, not simultaneous). `store.estimate_store_gb(level, model_cfg, n, branches, trainable_dtype, tta_levels=())` adds 28 V to the per-image deform term d of every level in `tta_levels` inside its existing loop (the max over levels then picks the worst level): 39.7 → 59.2 GB for the level-1 Mindboggle run, under the 0.7 × 103 GB = 72 GB RAM threshold of a 96G job (disk store otherwise, slower but correct). For `affine+deform` the d term already contains the full-resolution `src_affine` (4 B per level-1 voxel) at every deform level, which covers its longer life at a TTA last level (3.4); no further term. Expected process RSS 42.8 → ~63 GB; evaluation runs in a separate process (MaxRSS 74.5 GB measured for the whole job). Linear TTA host cost: 8 × N × 19 float64 params and 8 × N matrices, negligible. `--mem=96G` stays sufficient.

Time (N 101, RTX 8000, measured non-TTA per-level model times from section 1: l4 51 s, l2 115 s, l1 ≈ 163 s excluding writing): TTA model time ≈ 8 × (51 + 115 + 163) s + accumulation (level 1: 8 × 101 chunked reads/writes of ~0.2 GB per image, ~1-2 min) ≈ 46 min, versus 5.5 min; writing (886 s), reading and atlas are unchanged. `predict.py run` total ≈ 21.5 → ~62 min for a level-1 checkpoint and ≈ 18 → ~35 min for a level-2 checkpoint. Linear TTA at N 101 (`linear-eval`): no measured number exists; the linear passes (P0-P6) are at most the deform encoder share, so ≤ 8 × ~200 s ≈ 27 min per task for a level-1 checkpoint (upper bound; the job records the real value in `timing_s`).

## 7. CLI, API and records

### 7.1 Flags (`inference/cli.py`)

- `run`: `--tta {none,flip8}` (default `none`; the existing placeholder flag gets the new choice). `--tta-levels LEVELS` (comma-separated, e.g. `1` or `2,1`; default: all levels the plan runs; parsed with `inference.levels.parse_levels`). Validation before TF (exit 2 via `parser.error`): `--tta-levels` without `--tta flip8`; `--tta flip8` with `--mode unchunked`; levels not in {4, 2, 1} or duplicates. After the checkpoint level is peeked (`_setup_levels`): a level the plan does not run → PredictError(2), recorded in predict.json.
- `check-streaming`: no TTA flags (engine check only).
- New subcommand `linear-eval` (section 7.3) with `--tta`, `--tta-levels` as for `run`.
- `OUTPUT_PATTERNS`/`OUTPUT_NAMES` gain `tta_spread.csv` (so `--overwrite` protection covers it).

### 7.2 Library (`inference/tta.py`, new; TF + numpy)

- `FLIPS_8`: tuple of 8 bool triples, identity first; `IDENTITY_ONLY = ((False, False, False),)` for tests.
- `TTAConfig(mode, levels, flips=FLIPS_8)`; `TTAConfig.to_dict()`.
- `linear_flip_signs(flip, task) -> float64 [13|19]`; `flip_matrix_np(flip) -> float64 [4, 4]`; `unflip_matrices(M, flip)` (F M F, numpy or TF).
- `encode_linear(R, task) -> float64 [N, 13|19]` (2.3 step 5); `mean_linear(params_list) -> (p̄ float64, diagnostics)` (steps 6-7: float64 mean, eigh read-out, `rot_gap`, `readout_dev_deg`); `decode_linear(p̄, task)` (step 8).
- `flip_np(x, flip)` (host arrays [k, X, Y, Z, C]); `unflip_field(v, flip)` (TF).
- `linear_spread(A_f_list, A_bar, A_id, S, residual_params_f, task) -> dict of per-image arrays`; `gauge_size(mu_list) -> (deg, mm)`.
- Deform accumulation / spread tf.function bodies used by `StreamedLevel.deform_tta`.
- `train_flip_probs(chain_configs) -> {level: float or None}` (TF-free; 7.4).
TF-free parts (`linear_flip_signs`, `train_flip_probs`, CLI parsing) live in functions without TF imports so the CLI stays TF-free.

### 7.3 `predict.py linear-eval` (new; `inference/linear_eval.py`)

Purpose: rigid / affine errors with ground truth on any processed dataset (Mindboggle has no linear ground truth because it is MNI-aligned). Steps: read subjects as `run` does (`_read_subjects`, grid check); template-space mask m = (mean_i 1[I_i > 0]) ≥ 0.5 and S = `brain_second_moment(m, 1.0)`; seeded transforms `runtime.fixed_data.random_linear_transforms(np.random.default_rng(seed), N, task)` with the training validation ranges (`params.DEFAULT_LINEAR_RANGES`: rigid ±45° / ±15 mm with the small scale/shear nuisance, affine ±45° / ±15 mm / log-scale ±log 1.1 / shear ±0.05); render I_i ∘ T_i⁻¹ at level 1 with `fixed_data.render_group` (chunk 2); run `StreamedRunner.run(rendered, task)` (with or without TTA); targets `transforms.barycentric_targets(T_target, 10)`; errors with `runtime.metrics.group_errors(A, A_star, T_target, S, affine=task == 'affine')` (center and all-ordered-pair RMS mm, rotation deg, translation mm, affine scale / shear %, cycle).

Ground truth caveat (review finding 9): the targets treat the un-perturbed processed subjects as mutually aligned. They are only MNI-affine aligned, so the residual inter-subject misalignment adds an error floor. The floor is the same in both arms (same rendered group), so the TTA comparison stays fair, but absolute numbers are upper bounds of the model error. `summary.json` carries `"note": "errors include the residual alignment of the processed subjects"`.

Flags: `--checkpoint`, `--processed-dir`, `--subjects-file`, `--label-name` (unused but accepted for subject discovery), `--task {rigid,affine}` (required), `--seed` (default 0), `--tta`, `--tta-levels`, `--levels`, `--chunk-*`, `--precision`, `--device`, `--grid-reference`, `--gpu-memory-limit-mb`, `--out` (directory). Outputs: `per_subject.csv` (`subject`, center_rms_mm, center_rot_deg, center_trans_mm, [center_scale_pct, center_shear_pct], pair_rms_mm_mean = mean over pairs with the subject as moving image), `per_pair.csv` (`moving`, `fixed`, pair_* errors), `summary.json` (means, p95, the transforms' seed and ranges, tta record, timing, note), `tta_spread.csv` with TTA. Same seed → same rendered group in both arms (paired). Exit codes as `run` (2 input, 3 non-finite A).

### 7.4 Records

`predict.json`: `tta` stays the mode string (`"none"` / `"flip8"`; existing test unchanged). New key `tta_record`, added to `PREDICT_KEYS` (present in every record: null with `none` and in error records written before the TTA setup) = {`mode`, `levels`, `flips` (list of [fx, fy, fz] bools in order), `linear_averaging` "gauge-free residual, float64, eigh (Markley) read-out", `svf_accumulation` "float32", `train_flip_prob` {level: p or null}, `spread` (section 5 summary incl. `gauge_deg`, `gauge_mm`), `nonfinite_flips` {level: [flip indices]}, `timing_s` per level}. `pipeline` text gains " with 8-flip TTA at levels …". `estimate_gb` includes the TTA term. The same `tta_record` goes into `linear-eval`'s `summary.json`.

Training-flip check (review finding 8). `train_flip_probs` reads the effective training flip probability of every stage: the checkpoint run's config.json and the chain (level-2 source via `_peek_chain_configs`; level-4 source via the level-2 config's `cascade.source_config`, else the `cascade.stages` origin checkpoint's `source_config_path`); p = `args.flip_prob`, or 0 when `args.overfit_one_group` is true; null when the config is unreadable. Rule: warn only (no refusal). A WARNING is logged when a TTA level's stage has p < 0.5 ("flips partly out of training distribution") or p = 0 ("flips never seen in training; TTA averages out-of-distribution predictions"), and when p is unknown (null). Reason for not refusing: no real checkpoint has p < 0.5 (only overfit tests), TTA is an evaluation choice, and the record makes the condition visible in every result.

### 7.5 Errors

| Condition | Class | Result | Log |
|---|---|---|---|
| Invalid TTA flag combination (7.1) | input, fatal | exit 2 before TF (argparse) or PredictError(2) after the level peek, predict.json written with status error | argparse message / ERROR |
| GPU OOM in any flip pass | fatal | `ResourceExhaustedError` handled by the existing wrapper: exit 1 with `OOM_HINT` (same chunk advice; TTA does not raise the GPU peak) | ERROR |
| Non-finite raw params / A_f / v_raw in a flip | recoverable for diagnosis, fatal for the run | averaged output non-finite → existing non-finite detection, exit 3, atlas not written; `tta_record.nonfinite_flips` names the flips | WARNING per flip at detection, ERROR at the end |
| QR / eigh in float64 on a non-finite or singular residual (det L ≤ 0) | fatal | treated as a non-finite flip of that image (exit 3 path); `numpy.linalg.LinAlgError` caught and converted | ERROR with subject, level, flip |
| Host store over budget | recoverable | `--store auto` picks the disk store from the estimate | INFO (existing) |
| Disk store write failure | fatal | exception → exit 1; `src_affine` freed in `finally` | ERROR (existing) |
| `readout_dev_deg` > 1° or `rot_gap` < 1e-3 for any image | recoverable | output kept | WARNING with the subject and level |
| Stage trained with flip p < 0.5, p = 0, or unknown | recoverable | output kept, `train_flip_prob` recorded | WARNING |
| linear-eval: fewer than 2 subjects / grid mismatch | input, fatal | exit 2 | ERROR |

## 8. Tests

New `tests/test_tta.py` (unit, every module imports `tests.tf_setup` first) and `tests/test_predict_tta.py` (runner / CLI / end to end on the tiny randomized checkpoints of `tests/predict_helpers.py`, level 4 and level-2 cascade, narrow widths, N 3-4, GPU memory limit ≤ 6000 MB via the existing test setup):

- T1 sign vectors: per-axis 13-entry vectors == Jay's literal patterns; all 8 combined vectors == products; rigid and affine `matrix(σ_f ⊙ p) == F matrix(p) F` for 64 random general (non-rank-deficient) parameter vectors in float64 (max |diff| < 1e-12) and float32 (< 1e-5).
- T2 centering: `center_group(F A F) == F center_group(A) F` (float64, < 1e-10).
- T3 fields: `unflip_field(unflip_field(v, f), f) == v` bitwise; `integrate_svf(v^F) == (integrate_svf v)^F` (< 1e-5 vox); `warp(flip I, u^F) == flip warp(I, u)` and `transform_image(flip I, F M F) == flip transform_image(I, M)` (< 1e-4).
- T4 encoding and averaging: `decode_linear(encode_linear(M)) == M` for random rigid / affine M (< 1e-12, float64); `encode_linear(F M F) == σ_f ⊙ encode_linear(M)` (< 1e-12); `encode_linear` shear entries equal K01, K02, K12 (not the `qr_factors` ratios); `mean_linear` of 8 identical encodings returns them (< 1e-15) with `rot_gap` ≈ 1; eigh read-out of a mean of two encodings 2θ apart about one axis = the rotation by the mean angle (< 1e-9 deg); deform accumulation of 8 identical fields within 2e-7 relative.
- T5 identity-only TTA == no TTA, bitwise (max |diff| 0), linear (rigid, affine) and deform, level 4 and level-2 cascade, via `TTAConfig(flips=IDENTITY_ONLY)`.
- T6 TTA equivariance with a random (non-equivariant) model, run with `--precision float32` (tiny checkpoints, float32 policy; under AMP the 8 flipped passes round differently in fp16 and the tolerances do not hold): `TTA(flip_g x)` vs `F_g TTA(x) F_g` (A, tolerance 1e-4 mm point RMS) and vs `(TTA u)^F` (u, u_inv, tolerance 1e-4 vox), for g = each single axis and xyz, branches rigid, affine, deform, affine+deform (TTA at all levels, i.e. including the last level (2) of the tiny level-2 cascade, so `src_affine` must survive the 8 P0 passes), level 4 and level-2 cascade. This is the test that fails for a wrong sign or a wrong spatial convention.
- T7 equivariant stub models (fake `StreamedLevel` injected into the runner), float32: (a) linear stub with translation = intensity centroid (mm), 10-D Q = diag(0, 1, 1, 1) + ε·[[0, C_yz, C_xz, C_xy], [·, C_xx, C_xy, C_xz], [·, ·, C_yy, C_yz], [·, ·, ·, C_zz]] and shears ∝ (C_xy, C_xz, C_yz), log-scales ∝ log of the variances (C = intensity covariance, exactly equivariant by construction); (b) gauge stub: raw = B · (stub (a) output) with a constant non-flip-symmetric B (10° about (1, 0.3, 0.2) + (3, −2, 4) mm, + log-scale 0.05 and shear 0.03 for affine), centered-equivariant but not raw-equivariant; (c) deform stub v_raw = α ∇I (central differences, equivariant under array reversal) plus a constant non-symmetric field c (the deform gauge). TTA output == single pass: centered A within 1e-4 mm point RMS, pairwise maps A_j⁻¹ A_i within 1e-5 mm, deform u within 1e-6 vox; spreads ≈ 0; `gauge_deg` / `gauge_mm` of (b) ≈ the angle / shift of B's flip variation (> 0), of (a) ≈ 0. Stub (b) fails with version 1's raw averaging (regression guard).
- T8 spread / shift / gauge diagnostics: synthetic per-flip results with known offsets give the expected `spread_rms_mm`, `shift_mm`, `rot_deg_rms`, `gauge_deg` (analytic values).
- T9 off = bit-identical: before any code change, a fixture script (committed with the change, run once on the pre-change commit with `tf.config.experimental.enable_op_determinism()`) writes sha256 hashes of the DECODED outputs of `run` (for each NIfTI: `np.asarray(nib.load(p).dataobj).tobytes()` plus `nib.load(p).affine.tobytes()`, not the `.nii.gz` file bytes, whose gzip header carries a timestamp; plus the A matrices from predict.json / the summary as float64 bytes) for the tiny level-4 and level-2 checkpoints and branches rigid / deform / affine+deform into `tests/fixtures/predict_tta_off_golden.json`; after the change the same run with `--tta none` must reproduce the hashes exactly. Also: `deform()` after the `_v_raw_passes` refactor vs the fixture.
- T10 CLI and records: `--tta flip8` accepted; `--tta-levels` without flip8, with `--mode unchunked`, and an unrun level → exit 2; `--tta flip` still rejected; predict.json `tta`, `tta_record` (key in `PREDICT_KEYS`; null with `none` and in early error records; the existing exact-key-set test keeps passing); `tta_spread.csv` present only with flip8; `--overwrite` protection covers it; `affine+deform --tta flip8` on the tiny level-2 cascade runs to the end and `src_affine` is freed after the level (store has no `src_affine` entry afterwards).
- T11 store estimate: TTA term = 28 V per image added to the deform term of each TTA level; unchanged without TTA; `affine+deform` with TTA includes `src_affine` once.
- T12 linear-eval end to end on synthetic processed subjects (3 phantoms from `fixed_data`), tiny checkpoint: CSV/JSON written incl. the note; same seed → identical rendered inputs in both arms; errors finite.
- T13 `compare_tta.py`: synthetic CSV pairs with a known shift give the expected mean difference, CI covering it, Wilcoxon p < 0.05; identical inputs (all differences zero) give difference 0, `p = 1.0` and `note = "all differences zero"` (the `ValueError` of `scipy.stats.wilcoxon` is caught; the sign test with no non-tied subjects also gives p = 1.0); the subject-cluster bootstrap on a synthetic pair matrix with a known shift recovers it, and with pair outcomes that share a strong per-subject effect gives a wider CI than the naive per-subject bootstrap; mismatched subject sets → exit 2.
- T14 training-flip check: configs with `flip_prob` 0.5 / 0.2 / 0 / `overfit_one_group` true / unreadable give `train_flip_prob` 0.5 / 0.2 / 0 / 0 / null and the expected WARNINGs.

Verification after implementation: full suite (`tests/run_tests_wsl.sh`, background, ~7 h now); `python -m py_compile` under `bgr_preproc38` (Python 3.8 syntax); `check_flip_signs.py` and `check_gauge_free.py` again; on the cluster the first TTA job's predict.json (spread magnitudes plausible: linear spread_rms ≲ a few mm, deform spread ≲ 1 mm; `readout_dev_deg` small; `gauge_deg` / `gauge_mm` recorded) before reading results.

## 9. Mindboggle evaluation plan

### 9.1 What is compared

One checkpoint, the same 101 subjects, one group, same chunks and precision, arm A `--tta none`, arm B `--tta flip8` (all levels). Deform: `--branches deform` (as all Mindboggle runs so far). Linear: `linear-eval --task rigid` and `--task affine`, seed 0, arms A/B. Checkpoints (Jay's last-checkpoint rule): proposed (1) the current best real-only level-1 cascade `results/r1_cfgB_sm10_s2500_bnfreeze_true/ckpt/last` and (2) its level-2 source `results/r2_cfgB_s15000_smooth10_lr3e-4_wd1e-1_n32_d16/ckpt/last` (shows whether TTA matters more without level 1); the final paper models when Jay fixes them (open question 1).

### 9.2 Metrics

From `evaluation run` (`per_subject.csv`, `per_pair.csv`, `summary.json`): template_dice (+ cortical, subcortical), template_dice_loo, transfer_dice, template_lncc, fold_fwd_pct, fold_inv_pct, logjac_std_fwd/inv, inv_err_fwd_mm, inv_err_inv_mm, per-pair pair_dice (+ cortical/subcortical), pair_lncc, pair_fold_pct; group metrics centrality_mean_mm, mean_logjac_abs, atlas_intensity_std. From `linear-eval`: per-subject center_rms_mm, pair_rms_mm_mean, rotation, translation, scale / shear (affine). From TTA: `tta_spread.csv`, `gauge_deg` / `gauge_mm`.

### 9.3 Statistics (`scripts/compare_tta.py`, numpy + scipy 1.4 only)

Scope of inference (review finding 3a). All 101 subjects are registered as one group: they share one atlas and one central space, and every subject's pair mean shares the other 100 subjects. The per-subject outcomes are therefore not independent draws from a population of groups. The analysis is conditional on the fixed 101-subject Mindboggle group: it measures the effect of TTA on these subjects' registrations under the standard Mindboggle protocol. P values and CIs refer to resampling of subjects within this group and do not generalise to new groups; `compare.md` states this in its header.

Unit of analysis: the subject (n = 101). For each per-subject metric, paired differences d_i = B_i − A_i:
- mean, median and SD of d; number of subjects improved / worse / tied;
- 95 % CI of the mean difference by paired bootstrap over subjects (10 000 resamples, seed 0, percentile method);
- two-sided Wilcoxon signed-rank test `scipy.stats.wilcoxon(d, zero_method="pratt")` (handles exact zeros, e.g. folding 0 at high smoothness; no `alternative`/`mode` kwargs so the scipy 1.4 signature suffices) and a sign test `scipy.stats.binom_test`. All-zero differences: scipy 1.4 raises `ValueError`; it is caught and recorded as `p = 1.0`, `note = "all differences zero"` (same for the sign test with no untied subject);
- effect sizes: d_z = mean(d)/SD(d) and matched-pairs rank-biserial r.

Pair metrics (P2 and secondary pair metrics). Per-subject values d_i = mean over the subject's pairs as moving image of (B − A), for the Wilcoxon test and the description. The CI uses a subject-cluster bootstrap that accounts for the shared fixed subject: per resample draw subjects with replacement (counts c_k), keep all ordered pairs among the resampled subjects, excluding self-pairs, and compute the mean pair difference Σ_{i≠j} c_i c_j W_ij D_ij / Σ_{i≠j} c_i c_j W_ij (D = pair-difference matrix, W = 1 for evaluated pairs; this supports the seeded-subset fallback); 10 000 resamples, seed 0, percentile CI; vectorised as cᵀ(W ⊙ D)c, a few seconds.

Primary endpoints (fixed before looking at results; Holm correction over these three): P1 template_dice, P2 pair_dice (needs the all-pairs evaluation or the seeded-subset fallback), P3 fold_fwd_pct. Secondary (reported, Holm within the family, exploratory): the remaining metrics of 9.2, the linear-eval errors, and per-label template Dice (Wilcoxon per label, Holm over labels). Group metrics (one value per arm) are reported as values only.

Decision rule. "Measurable impact" on an endpoint = Holm-adjusted Wilcoxon p < 0.05 AND the 95 % CI of the mean difference (P1, P3: paired subject bootstrap; P2: subject-cluster bootstrap) excludes 0. "Practically relevant" additionally needs |mean d| ≥ the smallest effect of interest (SESOI):
- Dice (P1, P2): 0.002 (the size of the template-Dice changes between neighbouring smoothness weights in the level-2 S sweep on Mindboggle: 0.001-0.003 per step, S4 0.675 → S12 0.667);
- forward folding (P3, review finding 3b): SESOI_fold = max(0.25 × mean_A(fold_fwd_pct), 0.0002 percentage points), i.e. a 25 % relative change, with an absolute floor at MultiMorph's forward folding on Mindboggle (0.00021 %), below which changes are not practically relevant for a near-fold-free model;
- linear pair RMS (linear-eval): 0.5 mm.
With the CI fully inside ±SESOI the report states "no practically relevant effect". A TTA decrease in folding together with no Dice loss is reported as a regularity gain. Outputs: `compare.json` (all numbers), `compare.csv` (one row per metric), `compare.md` (table for the report, with the scope statement).

Replication (optional, open question 8): repeat both arms on 4 disjoint seeded subgroups of ~25 subjects (separate groups, so separate atlases and central spaces) and report the sign of the P1-P3 differences per subgroup; consistent signs in 4/4 support that the effect does not depend on the one group.

### 9.4 Commands (design only; Jay approves before any submission)

Changes to scripts: `slurm/predict_mindboggle.sh` gets `TTA` (default `none`) and `TTA_LEVELS` (default empty) passed only to step 2 (`predict.py run`), and `SKIP_CHECKS` (default empty; `1` skips steps 1a-1c and logs "checks 1a-1c skipped (SKIP_CHECKS=1; run in the TTA-none arm of the same job)"); checks 1a-1c stay TTA-free. New `slurm/tta_mindboggle.sh` (rtx8000, `-A qtim`, 1 GPU, 3 CPUs, 96G, 10 h): for one CKPT it runs `predict_mindboggle.sh` twice with `bash` inside the same allocation (NAME `${NAME}_tta_none`; then `${NAME}_tta_flip8` with `TTA=flip8 SKIP_CHECKS=1`). The inner script's `#SBATCH` lines are ignored when it is called with `bash`, and both calls reuse `TMPDIR=/scratch/$SLURM_JOB_ID` (the store frees its entries at the end of each run, so the second run starts clean). Then `predict.py linear-eval` rigid and affine for both arms (outputs `results/linear_eval_mindboggle/${NAME}_{rigid,affine}_tta_{none,flip8}`), then `compare_tta.py` on the two template-only evaluations (`$OUT/eval_template`) and on the linear evaluations; it prints the two all-pairs `eval_mindboggle.sh` commands and the final compare command. Estimated wall time for a level-1 checkpoint: arm A ~35 min (checks ~10 + run ~22 + template eval), arm B ~65 min (run ~62 + template eval, no checks), linear-eval 4 runs ≤ ~70 min → ~3 h (10 h limit for safety).

```bash
# GPU job (one per checkpoint), after Jay's approval
NAME=r1_cfgB_sm10_s2500_bnF CKPT=$PROJ/results/r1_cfgB_sm10_s2500_bnfreeze_true/ckpt/last \
  sbatch --job-name=bgr-tta-mb slurm/tta_mindboggle.sh
# all-pairs evaluation of both arms (basic partition, 8 CPUs, 96G, 12 h; existing script)
for ARM in none flip8; do
  F=$PROJ/results/predict_mindboggle/r1_cfgB_sm10_s2500_bnF_tta_$ARM/fields
  METHOD=r1_cfgB_sm10_s2500_bnF_tta_$ARM FWD="$F/{subject_dir}_fwd.nii.gz" INV="$F/{subject_dir}_inv.nii.gz" \
    ATLAS=$F/atlas.nii.gz ATLAS_SEG=$F/atlas_seg.nii.gz CONV="--components xyz --signs +++ --units mm" \
    sbatch slurm/eval_mindboggle.sh
done
# final comparison (CPU, seconds; never on the login node -> inside a basic job, 1 CPU, 4G, 15 min)
python scripts/compare_tta.py --a $PROJ/results/eval_mindboggle/r1_cfgB_sm10_s2500_bnF_tta_none \
  --b $PROJ/results/eval_mindboggle/r1_cfgB_sm10_s2500_bnF_tta_flip8 --out $PROJ/results/tta_compare/r1_cfgB_sm10_s2500_bnF
```

Fallback if the basic queue is long: in-job seeded pairs `EVAL_PAIRS=1000` (same seed in both arms, ~25 min per arm on 3 workers) for P2, recorded as such (the cluster bootstrap uses W for the evaluated pairs). Job count: 1 GPU job + 2 basic jobs per checkpoint, within the two-active-jobs rule if submitted in sequence (Jay's own jobs counted).

## 10. Edge cases

- N = 2 (smallest group): group means and centering work as today; spreads and gauge still defined.
- Groups smaller than in training (PET: 2-4 images of one patient) or larger (Mindboggle 101): TTA is per group and does not change the group size.
- Config A vs B, synthetic-contrast checkpoints, float32_fine / policy deform decoder, AMP or `--precision float32`: TTA only flips inputs and averages outputs of the engine, which already handles all of these; no TTA-specific branch.
- Rigid task: residual K = I up to rounding, dropped; the rigid output stays exactly rigid (decoded from 13 entries).
- Averaged rotations of strongly disagreeing flips (e.g. a 180° ambiguity): small `rot_gap` → WARNING; the eigh result is still a valid rotation.
- Fp16 headroom record: max over the 8 flips per block.
- `--levels` with skipped levels: no passes, no TTA there (3.5).
- `affine+deform` with TTA at the last level: `src_affine` lives until the level ends (3.4).
- Images with content near the border: zero-fill pull warps and border-clamped fields are symmetric under reversal, so flips introduce no asymmetry beyond the network's own.
- Non-finite flip: see 7.5; the average is non-finite, the run fails with exit 3 and names the flips.
- Disk store: the TTA stores use the same store object (prefix `L<level>_tta_`), freed at the end of the level.

## 11. Out of scope

- TTA in training-time validation (Jay: no new training-time validation), `runtime/eval_run.py` / `eval_synth.py` (they evaluate through the Trainer's single-graph cascade validation functions; LUMIR-val TTA would need a second cascade implementation; open question 4), `check-streaming`, `--mode unchunked`.
- Augmentations other than the 8 flips (rotations, intensity), weighted averaging, displacement-space averaging.
- MIRIAD / PET evaluation scripts (later; `predict.py run --branches rigid --tta flip8` and `linear-eval` already cover the mechanics).

## 12. Open questions for Jay

1. Which checkpoints: the proposal in 9.1 (current best config-B L1 + its L2 source) now, the final paper models later?
2. Primary endpoints and smallest effects of interest: template Dice and pair Dice 0.002; forward folding max(25 % of arm A's mean, 0.0002 pp); linear pair RMS 0.5 mm. OK?
3. All-pairs evaluation for both arms on `basic` (2 jobs, ~2 h each), or 1000 seeded pairs inside the GPU job?
4. Do you want TTA in `eval_synth` / LUMIR-val evaluation later? (Needs its own design: the Trainer's validation path.)
5. A `--tta-levels 1` (last level only) ablation in the Mindboggle comparison, or only none vs all levels?
6. For rigid PET later: same 8 flips (all three axes), or only the left/right flip if PET anterior/posterior asymmetry makes y/z flips less reliable?
7. Plain mean of the 10-D entries (your rigid net) is the design; any interest in eigengap-weighted averaging as an option?
8. Replication on 4 disjoint seeded subgroups of ~25 subjects (adds 8 smaller predict runs, ~1.5 h GPU): wanted for the paper, or the single 101-subject group only?
9. Linear averaging deviates from your rigid net in one point: we average the re-encoded gauge-free residuals (per-flip centered result divided by the previous level's), not the raw head outputs, because our losses never constrain the raw outputs' common factor (section 2.2). The 10-D averaging and sign patterns are yours unchanged. OK?

### 12.1 Jay's answers (added by the orchestrator; binding for the implementation)

1. Checkpoint now: the current best config-B level-1 run `results/r1_cfgB_sm10_s2500_bnfreeze_true/ckpt/last` (its level-2 source as in 9.1 may also be run); swapped for the final models later.
2. Smallest effects of interest as proposed (Dice 0.002; folding max(25 % of arm A mean, 0.0002 pp); linear pair RMS 0.5 mm).
3. All-pairs evaluation afterwards as a separate CPU job on `basic` (not inside the GPU job).
4. No TTA in training-time validation, `eval_synth` or LUMIR-val: TTA is inference/results only (out of scope, as in section 11).
5. Comparison: `--tta none` vs all levels only; keep `--tta-levels` in the code but no last-level ablation job.
6. All 8 flips (for PET too, if metrics are the same or better).
7. Plain mean of the re-encoded 10-D entries (orchestrator's recommendation, Jay deferred): after re-encoding every flip's residual as I − q qᵀ the per-flip confidence is discarded, so eigengap weights would be equal anyway; plain mean = Markley average. No weighting option.
8. Only the full 101-subject group (no subgroup replication); TTA must work for any group size N ≥ 2 (tests cover small N).
9. Gauge-free residual averaging (section 2.3) approved.

## Changelog

- v1 (2026-11-06): first version.
- v2 (2026-11-06), response to review v1 (CHANGES_REQUESTED, 1 HIGH, 2 MEDIUM, 7 NIT); all findings addressed:
  - Finding 1 (HIGH, raw gauge): linear TTA now averages gauge-free residuals R_f = A_f A_prev⁻¹ (A_f = per-flip unflipped, composed, centered result), re-encoded in float64 (QR with positive diagonal; 10-D of I − q qᵀ; raw K entries in the `scale_shear_matrix` convention, not the `qr_factors` ratios), averaged, decoded, then centered / composed once (2.2, 2.3). Identity-only path kept bit-identical (2.3, T5). Claim 1 rewritten for centered-equivariant models with tolerances (2.6). Rotation read-out changed from `decode_10d` of the mean to the exact eigh (Markley) eigenvector, projected back to I − q̄ q̄ᵀ, because the averaged matrix is full rank when flips disagree (2.4; `readout_dev_deg` keeps the comparison). New diagnostic `gauge_deg` / `gauge_mm` (5). New check script `check_gauge_free.py` (2.7: v1 0.18 mm vs v2 3e-14 mm pairwise error under a gauge; E4 pipeline equivariance). Tests: T4 rewritten for the encoding, T7 gets the gauge stub (b) and the deform gauge (c). `StreamedLevel.linear` no longer needs `return_params`. Open question 9 informs Jay of the deviation from his raw averaging. Deform branch: gauge argument made explicit (3.2; centering cancels a common field exactly).
  - Finding 2 (MEDIUM, `src_affine` freed after flip 0): free-after-P0 wrapper only when the last level is not a TTA level; otherwise `SRC_AFFINE` freed after `deform_tta` in `try/finally`; level mask computed at the start of `_deform` / `_deform_sel` before any P0 (3.4); estimate already counts `src_affine` at every deform level, stated (6); T6 / T10 / T11 cases added.
  - Finding 3 (MEDIUM, statistics): conditional-inference scope statement (9.3, `compare.md`); subject-cluster bootstrap for pair metrics incl. the seeded-subset weights; optional 4-subgroup replication (9.3, open question 8); folding SESOI = max(25 % of arm A's mean, 0.0002 pp) (9.3, open question 2).
  - Finding 4 (NIT): `SKIP_CHECKS` added to `predict_mindboggle.sh`; nested `bash` call ignores `#SBATCH`, shared `TMPDIR` noted (9.4); arm-B time estimate updated.
  - Finding 5 (NIT): `tta_record` added to `PREDICT_KEYS`, null with `none` and in early error records (7.4, T10).
  - Finding 6 (NIT): T9 hashes decoded arrays + affines + A, not `.nii.gz` bytes.
  - Finding 7 (NIT): T6 runs with `--precision float32`; T7 float32 stated.
  - Finding 8 (NIT): `train_flip_prob` per stage read from the chain configs (effective p, 0 with `--overfit-one-group`); warn-only rule chosen with reason (7.4, 7.5, T14).
  - Finding 9 (NIT): residual-alignment error floor stated in 7.3 and in `summary.json`.
  - Finding 10 (NIT): all-zero differences → `ValueError` caught, `p = 1.0`, `note = "all differences zero"` (9.3, T13).
