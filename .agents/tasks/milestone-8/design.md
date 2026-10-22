# Milestone 8 technical design: level 2 with a frozen level-4 cascade

Status: design v5 (2026-10-21), fourth revision after the design review of v4 (`design-review.md`, 0 HIGH, 1 MEDIUM, 6 NIT, all addressed; responses in section 18, table "v4 → v5"; the earlier responses are kept below it). Requirements: `.agents/tasks/milestone-8/requirements.md`. Baseline design doc: `.agents/tasks/network-design/design.md` v3.10 (this design is merged into it as v3.11, updated to v3.12 for v5).

## 0. Overview

Level 2 (80×112×96, 2 mm) is trained as the second stage of a two-stage cascade. A new module `models/cascade.py` holds a `FrozenStage`: a `GroupRegLevelModel(4, cfg)` restored from the level-4 checkpoint, always called with `training=False`, outside the GradientTape, under a float32 Keras policy. The trainable level-2 `GroupRegLevelModel(2, cfg)` gets a verbatim copy of the same checkpoint's variables (`copy_weights_from`). Every train and validation step pools the (flipped, augmented) level-2 group 2×2×2 to level 4, runs the frozen stage on it, pre-warps the level-2 images with the frozen estimate and lets the trainable stage predict a residual, which is composed with the frozen estimate inside the tape. Losses, metrics, PNGs and best files use the composed result. BN statistics of the trainable stage are recalibrated with forward-only passes (exact average of per-batch statistics) and then frozen. Level skipping is implemented with default 0. The level-2 memory is measured on the cluster first by a new `tests/gpu/measure_level_memory.py`; group sizes follow from it. Every new path is gated by flags whose defaults leave level-4 runs bit-identical. Label sampling is recommended for deferral to milestone 9 (needs Jay's export).

Technology stack (locked): Python 3.8 syntax, TensorFlow 2.10 (Keras 2.10 layers, `tf.function`, `tf.GradientTape`, `tf.train.Checkpoint`), DALI 1.18 (existing pipelines, no new DALI ops), numpy 1.21 (`np.random.default_rng` is available), scipy 1.4.1 only in tests, unittest. No new dependencies.

## 1. Decisions at a glance (open questions Q1–Q18)

| Q | Decision |
|---|---|
| Q1 warm-start checkpoint | One checkpoint seeds both stages: `results/m7_three_task_lncc/ckpt/best` (step 11000). Reason: one shared encoder means one consistent model; two frozen models (linear from 11000, deform from 10500) would double the frozen compute and memory for a deform pair-Dice difference of ~0.001 (best_deform 10500: 0.702 vs 0.701 at 12000; 11000 sits in the plateau). The trainable level-2 stage is initialised from the same checkpoint. `best_deform` and `last` stay usable via `--cascade-from`. |
| Q2 what the trainable stage copies | Everything, verbatim (no head re-initialisation, no SVF rescale). Section 2. |
| Q3 level-4 inputs of the frozen stage | 2×2×2 average pooling of the level-2 tensor in TF (`tf.nn.avg_pool3d`), after flips, in train and validation. One DALI read, the same T_i, G, flips and intensity augmentation by construction. |
| Q4 linear composition | A_raw_i = R_raw_i · A4_i, then ONE `GroupCentering` (the trainable model's `centering` layer). The residual heads are the copied heads (10D rotation + t, R·K), unchanged ranges. |
| Q5 pre-warp interpolation | Accepted: one extra trilinear interpolation (`layers.transform_image`, zero fill) on the DALI-augmented level-2 image. The un-augmented source is not available in TF, and a single-interpolation route would need DALI to know A4. |
| Q6 deform details | Residual v^(2) group-mean centered (existing `_call_deform`); smoothness on the COMPOSED field (design 6.2: "composite deformable part at the current level"); u4 integrated at level 4 then upsampled with `diffeo.upsample_displacement`; in validation the frozen stage is recomputed inside the traced validation function (no field cache), only the frozen-stage baseline METRICS are cached (section 8). New metric `val_deform/composed_mean_disp_mm`. |
| Q7 gradient flow | Frozen stage runs before the tape is opened, its outputs pass through `tf.stop_gradient`; its variables are not in `model.trainable_variables` nor in the optimizer. Composition, centering and losses are inside the tape. |
| Q8 skip | Implemented, `--cascade-skip-prob` default 0 (design 4.7: p_skip for the first cascade stage ≈ 0). No range or curriculum change for skipped linear steps (the copied model already handles ±45°). Section 6. |
| Q9 BN recalibration | `--bn-recal-batches K` (default 0 = off; recipe 64; K > 0 only together with `--cascade-from`, else `parser.error`), task plan proportional to the ratio, on cascade pre-warped training batches with flips and G on, exact per-layer average of per-batch statistics, no gradient. Gap = eval vs train-mode-statistics metrics at step 0 before and after recalibration (logged before freezing). Section 7. |
| Q10 after recalibration | Freeze (`--bn-recal-freeze true`, default true when recalibrating), design 5.1; re-applied on resume from the SAVED config.json `bn_recalibration.freeze` record (section 9), not from the current args. |
| Q11 group sizes | From the memory job (section 11). Targets: linear N 16, deform N 8; floors linear 8, deform 4. Validation group sizes = training group sizes (existing rule min(N, 39)); frozen-stage baselines are computed on the same groups and N. |
| Q12 memory fallback order | (1) `--gradient-checkpointing integrator`, (2) `both`, (3) `--amp true`, (4) smaller N down to the floors. |
| Q13 schedule | Fresh Adam 3e-4 cosine, warmup 500, 8000 steps, val every 500, ckpt every 500, ratio 1:1:2, LNCC 1/1, G 45°/10 mm from step 0, curriculum off. `--time` from the smoke run's step times (section 12). |
| Q14 frozen stage in checkpoints | Stored in the level-2 checkpoint (`RunCheckpoint(extra={"frozen_stage": ...})`), plus a weight fingerprint in config.json `cascade`, verified after every restore. Resume does not need or read the source path; config.json `init_weights`, `cascade`, `bn_recalibration` are carried over verbatim (section 9). |
| Q15 C4 mechanics | Ratio of the medians (as `check_mixed_run._check_enc_grad` computes): median linear `train_<task>/diag_enc_grad_norm_eff` / median deform `..._eff`, over log records whose `step` field (the window end) is ≥ 50; with `--log-every 25` the first used window is steps 26–50 (no zero-head warm-up: the SVF head is copied). KEEP only if BOTH linear ratios are in [0.1, 10]; otherwise FALLBACK. UNDECIDED (values missing) is treated as FALLBACK and the run is inspected. |
| Q16 label sampling | **(B) defer to milestone 9 applies unless Jay chooses (A); nothing in milestone 8 depends on it.** The decision stays flagged as open in the milestone-8 report until Jay records it. The small exporter change that lets Jay export the full set at level 2 is done now either way. Section 14. |
| Q17 G at level 2 | G is part of the DALI render at level 2 (and of the `_g` validation renders at level 2); the frozen stage sees the pooled render, so it handles G exactly as at level 4. |
| Q18 no-cascade baseline | Supported (`--level 2 --init-weights <level-4 ckpt>`, cross-level warm start, section 2.3), not run in milestone 8 (ablations out of scope). |

## 2. Weight transfer level 4 → level 2

### 2.1 Per-part level dependence (AC1.4)

Evidence: `models/group_net.py` (`GroupEncoder`, `LinearDecoder`, `DeformDecoder`), `tests/test_group_net.py::TestShapesAndLevels.test_identical_parameters_across_levels`, `test_copy_weights_4_to_2`, `TestDeformCopyWeights.test_copy_4_to_2`.

| Part | Variables | Shapes level-dependent? | Output units level-dependent? | Handling |
|---|---|---|---|---|
| Stem (`encoder/stem`, group conv in A) | conv kernel 3³ × (2 → f0), bias | no | features at the level grid (2 mm voxels at level 2: half the receptive field in mm) | copy; BN-free (no norm before the stem) |
| Encoder L2 (`encoder/l2`) | BN stats + γ/β, conv | no | runs at 80×112×96 at level 2 (40×56×48 at level 4) | copy; BN recalibrated (section 7) |
| `pool3` | none (parameter-free `AveragePooling3D`, only at levels 2/1) | — | — | no variables |
| Encoder L3–L6 | BN, conv | no; bottleneck 5×7×6 at both levels | same grid at both levels | copy; BN recalibrated |
| 1×1 reductions (192, 96, 48, 24) | BN, conv | no | same | copy; BN recalibrated |
| Flatten | none; 5×7×6×24 = 5040 at both levels | no | — | — |
| Dense 2048, 512, `head_norm_act` | Dense + LayerNorm γ/β | no (input 5040 at both) | LayerNorm has no statistics | copy |
| `rigid_head` (13) | kernel, bias | no | 10D rotation unitless; t in mm (`translation_scale` 1) | copy |
| `scale_shear_head` (6) | kernel, bias | no | log-scales, shears unitless | copy |
| `centering` (`GroupCentering`) | none | — | — | — |
| Deform decoder D5–D3 | BN, conv | no | same grid | copy; BN recalibrated |
| D2 (`up_d2` parameter-free at level 2), D1 | BN, conv | no | 80×112×96 at level 2 | copy; BN recalibrated |
| `svf_norm_act`, `svf_head` | BN, conv 3³ → 3 (float32) | no | **v in level voxels: 2 mm per unit at level 2 vs 4 mm at level 4** | copy verbatim, no ×2 rescale |

Rationale for the SVF head: the level-2 stage sees images already warped by up(u4) and its v is group-centered, so the expected residual is small; a copied head predicting in level-2 voxels under-scales a full-size prediction by 2, which is the safe direction. A zero head would bring back the zero-encoder-gradient warm-up of milestone 7. The linear heads are copied too: on pre-aligned inputs a copied head predicts roughly a common pose Q plus small residuals, and the barycenter is left-equivariant (bary(Q X_i) = Q bary(X_i)), so Center(Q A4_i) = A4_i; no harm, and no information is thrown away.

The brain second moment S (linear losses) is computed for level 2 from `template_mask_2.npy` (`postprocess.brain_S_matrix(2, ...)`, already the case in `main._build_dali_data`). With real data, `main._build_dali_data` also computes the level-4 S from `template_mask_4.npy` (`postprocess.brain_S_matrix(4, ...)`) and records `S_rel_diff_vs_level4 = ‖S2 − S4‖_F / ‖S4‖_F` in config.json `cascade` (AC1.5). In overfit-synthetic runs (`_build_overfit_data`, S from a level-specific phantom, no level-4 counterpart) the key is written as `null`.

### 2.2 Restore mechanism (cascade)

Mechanism: restore the checkpoint into the frozen level-4 model (same level as the source, so the object graph matches exactly), then `GroupRegLevelModel.copy_weights_from(frozen.model)` into the level-2 model (existing, tested; matches by order, shape and scope-local name, and requires equal configs, seed excepted). No direct cross-level `tf.train.Checkpoint` restore, so the extra parameter-free layers (`pool3`, `up_d2`) never meet the object-graph matcher.

New in `runtime/checkpoint.py`:

```python
def load_cascade_source(path):
    """--cascade-from: (prefix, src_cfg ModelConfig, info dict). Raises ValueError when the path
    does not resolve (resolve_checkpoint), the source config.json is missing, or the source
    run's args.level != 4. info: checkpoint (abs prefix), source_config (abs path), source_level 4,
    source_step (tf.train.load_variable(prefix, STEP_KEY))."""

def check_cascade_config(model_config, src_cfg):
    """ValueError naming every ModelConfig field (seed excepted) that differs; no field may differ."""

def weight_fingerprint(model):
    """{'n_vars': int, 'abs_sum': float64 sum of |w| over model.weights, 'sq_sum': float64}."""
```

`DECODER_ONLY_FIELDS` is NOT extended for the cascade: the cascade needs a source with the deform decoder (the m7 LNCC run has one) and identical config, so the strict rule is simpler and safer. `FrozenStage.from_checkpoint(prefix, src_cfg)` uses the existing `_strict_restore` (expect_partial for the optimizer etc., `assert_existing_objects_matched` on the model).

Init-record schema (AC1.3; pinned here, used verbatim by 2.3 and tests 1(a) / 10). Base = the dict `warm_start_model` builds today (runtime/checkpoint.py line ~171): `checkpoint`, `source_config`, `mode`, `config_diff`, `restored_parts`, `fresh_parts`, `ignored_source_parts`, `n_restored_vars`, `n_fresh_vars`. `main._init_weights_lines` indexes `info["ignored_source_parts"]` unconditionally (main.py line 398), so every record MUST carry that key. Cross-level / cascade records add exactly three keys: `source_level` (int), `source_step` (`int(tf.train.load_variable(prefix, C.STEP_KEY))`) and `cross_level` ([src_level, model.level]); the cascade record adds a fourth, `via: "cascade"`. Same-level records gain nothing, so level-4 records stay byte-identical.

The cascade record: `{"mode": "full", "via": "cascade", "checkpoint": <abs prefix>, "source_config": <abs path>, "source_level": 4, "source_step": <int, 11000 for m7 LNCC ckpt/best>, "cross_level": [4, 2], "restored_parts": {"encoder": n_enc, "linear_decoder": n_lin, "centering": 0, "deform_decoder": n_def}, "fresh_parts": {}, "ignored_source_parts": [], "n_restored_vars": len(model.weights), "n_fresh_vars": 0, "config_diff": {}}` (n_* = `len(part.weights)` of the level-2 model), written to config.json `init_weights`, the `init_weights` event and stdout (`_init_weights_lines`). So `check_mixed_run.py --expect-init-weights full` passes.

### 2.3 Cross-level `--init-weights` (no cascade, Q18)

`warm_start_model(model, model_config, path)` reads the source level from the source config.json `args.level` (absent → `model.level`, so older behaviour is unchanged). If it equals `model.level`: unchanged code path. If it differs: build `src = GroupRegLevelModel(src_level, src_cfg).build_model(2)`, `_strict_restore(src, prefix)`, then identical configs → `model.copy_weights_from(src)` (`mode='full'`), decoder-only difference → the existing partial copy loop over `SHARED_PARTS` (now with `src` at `src_level` instead of `model.level`). `info["source_level"]`, `info["source_step"]` and `info["cross_level"] = [src_level, model.level]` are added only in the cross-level branch (schema of section 2.2), so level-4 records are unchanged.

Exact record of the cross-level paths (test 10 asserts these dicts key by key; values from the same helpers as the same-level branch):
- Identical configs: the same-level full dict plus the three cross-level keys: `{"mode": "full", "checkpoint": <abs prefix>, "source_config": <abs path>, "source_level": 4, "source_step": int, "cross_level": [4, 2], "restored_parts": {"encoder": n_enc, "linear_decoder": n_lin, "centering": 0, "deform_decoder": n_def}, "fresh_parts": {}, "ignored_source_parts": [], "n_restored_vars": len(model.weights), "n_fresh_vars": 0, "config_diff": {}}` (n_* = `len(part.weights)` of the level-2 model; `deform_decoder` absent when the model has none, as in the same-level full branch).
- Decoder-only difference: the same-level partial dict plus the three cross-level keys: `{"mode": "partial", "checkpoint", "source_config", "source_level": 4, "source_step": int, "cross_level": [4, 2], "restored_parts": {"encoder": n_enc, "linear_decoder": n_lin, "centering": 0}, "fresh_parts": {"deform_decoder": len(fresh)}, "ignored_source_parts": [...], "n_restored_vars": len(pairs), "n_fresh_vars": len(fresh), "config_diff": {<field>: [src, dst], ...}}`, where `fresh_parts` has the `deform_decoder` entry only if the model has a deform decoder and `ignored_source_parts` is `["deform_decoder"]` iff the source has one, else `[]`. Note `n_restored_vars = len(pairs)` (copied variable pairs), not `len(model.weights)`, exactly as the same-level partial branch (checkpoint.py line ~226).
- No `via` key in either (only the cascade record carries `"via": "cascade"`, section 2.2).
- Tests 1(a) and 10 also call `main._init_weights_lines(info)` on every record (cascade, cross-level full, cross-level partial) and assert it returns a `str` (guards the `KeyError` on `ignored_source_parts`).

## 3. The frozen stage (`models/cascade.py`, new)

```python
PREV_LEVEL = {2: 4}                       # milestone 8: level 2 only (level 1 = milestone 9)

def pool_to_coarser(images):              # [N,X,Y,Z,C] float32 -> [N,X/2,Y/2,Z/2,C]
    return tf.nn.avg_pool3d(tf.cast(images, tf.float32), 2, 2, "VALID")

class FrozenStage:
    """The previous cascade level, inference only. .model: GroupRegLevelModel(4, cfg) (built under
    a float32 Keras policy, gradient_checkpointing 'none'); .level = 4; .fingerprint."""
    @classmethod
    def from_checkpoint(cls, prefix, cfg): ...
    @classmethod
    def from_model(cls, model): ...           # tests and the memory script: float32 copy of model's weights
    @classmethod
    def from_config(cls, cfg): ...            # resume: random init, then restored by RunCheckpoint
    def linear(self, images_fine, task):      # -> A_prev [N,4,4] float32 (stop_gradient)
        out = self.model(pool_to_coarser(images_fine), task=task, training=False)
        return tf.stop_gradient(tf.cast(out["A"], tf.float32))
    def deform(self, images_fine):            # -> dict U, U_inv, V (level-2 voxels, stop_gradient)
        out = self.model(pool_to_coarser(images_fine), task="deform", training=False)
        up = diffeo.upsample_displacement
        return {k: tf.stop_gradient(up(out[s])) for k, s in (("U", "u"), ("U_inv", "u_inv"), ("V", "v"))}

def prewarp_linear(images, A_prev, spacing):  # I_i o A_prev_i^-1 (pull), zero fill
    return L.transform_image(images, tfm.affine_inverse(A_prev), spacing)

def compose_linear(R_raw, A_prev, centering): # Center(R_raw_i @ A_prev_i), dtype of A_prev
    # train step: A_prev float32 (FrozenStage.linear), centering = model.centering (float32);
    # test 3 float64 case: float64 inputs + tfm.GroupCentering(iters=cfg.centering_iters, dtype="float64")
    return centering(tf.matmul(tf.cast(R_raw, A_prev.dtype), A_prev))

def prewarp_deform(images, U):                # warp(I, U), linear, zero fill
    return diffeo.warp(tf.cast(images, tf.float32), U)

def skip_linear(skip, A_prev):                # A_prev or I (skip: bool scalar tensor)
def skip_fields(skip, fields):                # each field or zeros
```

Float32 policy: Keras layers capture the dtype policy in `__init__`, so BOTH the construction `GroupRegLevelModel(4, cfg)` AND `build_model(n)` happen inside a small context manager `_float32_policy()` (saves `tf.keras.mixed_precision.global_policy()`, sets `'float32'`, restores in `finally`). This applies to every factory: `FrozenStage.from_checkpoint`, `FrozenStage.from_model` (which builds a fresh float32 model from `model.config` and copies weights, never wraps a model built under another policy) and the resume path (`FrozenStage.from_config(cfg)`, random init, restored by `RunCheckpoint`). Its outputs (and the frozen baselines) do not depend on `--amp`. Inference-only: its methods are only ever called with `training=False`; its BN layers use moving statistics (level-4 values, never recalibrated, AC6.6).

Pooling consistency: level-2 voxel centers (p − 39.5)·2 mm; the block {2q, 2q+1} averages to (q − 19.5)·4 mm, the level-4 center (design 2: block averaging preserves world points). All level-2 axis lengths are even, so a flip commutes with the pooling. The pooled tensor differs from a native level-4 DALI render by the order of warp and block averaging and by the per-image noise averaged over 8 voxels (std / √8); the frozen model was trained with intensity augmentation, so this is accepted (tested in AC7.2 with a phantom).

## 4. Linear cascade (rigid, affine)

Matrices (4×4 homogeneous, centered mm, design 2): A4_i maps image i to the level-4 central frame c4; the level-2 stage sees x_i = I_i ∘ A4_i⁻¹ (in c4) and predicts R_i: c4 → new central space. Then

    A_raw_i = R_raw_i · A4_i,      A_i = μ⁻¹ A_raw_i,   μ = barycenter{A_raw_j} (GroupCentering, 4 unrolled iterations)

Pairwise i → j: A_j⁻¹ A_i (`tfm.pair_predictions`), true value T_j T_i⁻¹. Targets A_i* = μ*⁻¹ T_i⁻¹ and P_ij* come from `LinearPostprocess` unchanged (they depend only on the emitted T). Linear losses (`losses.linear_loss(A, batch, S, lam)`) take the composed, centered A.

Train step (`Trainer._make_train_step`, cascade branch chosen at trace time when `self.cascade is not None`):

```
batch = pp(images, t_dali, rt_dali)                    # flips etc., outside the tape (unchanged)
x = where(in_finite, batch["images"], 0)               # unchanged
A_prev = self.cascade.linear(x, task)                  # outside the tape, eval mode, stop_gradient
if p_skip > 0: A_prev = cascade.skip_linear(skip, A_prev)      # trace-time gate
x_in = cascade.prewarp_linear(x, A_prev, spacing)      # outside the tape
with tape:
    out = model(x_in, task=task, training=True)
    A = cascade.compose_linear(out["A_raw"], A_prev, model.centering)
    terms = losses.linear_loss(A, batch, S, lam) ...   # objective, weights, AMP as today
```

`collapse_diagnostics(out["pre_head"], out["params"])` stays on the trainable stage's features. The trainable model still computes its own `out["A"]` (unused, negligible). Under AMP `out["A_raw"]` is float32 (heads are float32), `A_prev` is float32.

## 5. Deformable cascade

Conventions (design 4.6 Cascade row, `diffeo.compose_cascade`): U = up(u4), U⁻ = up(u_inv4) with up = 2 × `upsample_trilinear2x` (half-pixel: fine p ↔ coarse (p − 0.5)/2, so the fine map is p + 2u4((p − 0.5)/2)); residual v2 = center(v_raw2) at level 2, (u2, u2⁻) = exp_and_inverse(v2); composed

    Ψ_i = up(φ4_i) ∘ exp(v2_i):     u_i     = compose(U_i, u2_i)          (diffeo.compose_cascade)
    Ψ_i⁻¹ = exp(−v2_i) ∘ up(φ4_i)⁻¹: u_inv_i = compose(u2inv_i, U⁻_i)

The level-2 stage input is I_i ∘ up(φ4_i) = `diffeo.warp(x, U)` (design 4.7: "level k sees I_i ∘ A_i⁻¹ ∘ φ_i^(k−1)", A = I in deform steps).

`--deform-residual-affine` (3-input deform step `images, labels, t_dali`) is not combined with the cascade: parser rule "`--cascade-from` requires `--deform-residual-affine false`" (no recipe uses it; its residual-affine pre-warp would have to be ordered against up(φ4), which is a design question for later). So the cascade deform step always has the 2-input signature plus `skip`.

Deform step (cascade branch of `_make_deform_step`):

```
batch = pp(images, labels)                             # unchanged
x = where(in_finite, batch["images"], 0)
prev = self.cascade.deform(x)                          # outside the tape: U, U_inv (level-2 voxels)
if p_skip > 0: U, U_inv = cascade.skip_fields(skip, [U, U_inv])
x_in = cascade.prewarp_deform(x, U)                    # outside the tape
with tape:
    out = model(x_in, task="deform", training=True)    # v_raw, v (centered), u, u_inv = residual
    u, u_inv = diffeo.compose_cascade(U, U_inv, out["u"], out["u_inv"])
    out_c = dict(out, u=u, u_inv=u_inv, u_res=out["u"])
    terms = losses.deform_loss(out_c, batch, ...)      # batch["images"], batch["onehot"]: ORIGINAL
```

All loss terms act on the composed field: template / pair soft Dice warp the original one-hot once with u (no double label interpolation), both LNCC terms warp the original images with u and use `pairwise_displacement(u, u_inv, i, j)`, smoothness is `smoothness_loss(u)` of the composed u (Q6). `deform_diagnostics(out_c)` reports `diag_u_max_vox` and `diag_fold_frac` of the composed field; a new key `diag_u_res_max_vox` (max |u2|) is added in the cascade branch only. The residual centering is the existing `center_velocities` in `_call_deform` (mean_i v_i = 0); centrality of the composite is approximate (design 4.7) and tracked by `val_deform/composed_mean_disp_mm` = RMS over voxels of |mean_i u_i| in mm. `val_deform/centrality_mm` keeps its meaning (residual v).

## 6. Level skipping

Naming (recorded in design.md 4.7): the design's "level 1 / level 2" in the p_skip defaults are cascade positions (level 1 = the 1/4 stage = code level 4; level 2 = the 1/2 stage = code level 2). In milestone 8 the only prior stage is the 1/4 stage, whose design default is ≈ 0.

Flag `--cascade-skip-prob p` (float, default 0.0, must be in [0, 1), > 0 needs `--cascade-from`). Per step s: `skip = random.Random("cascade-skip-%d-%d" % (seed, s)).random() < p` (stateless, so a resume reproduces it; string seeds are hashed deterministically by `random.Random`). One draw per step = per group; applies to the linear and deform cascade alike. A skipped stage gives A_prev = I (linear) and U = U⁻ = 0 (deform), i.e. the stage passes on its input (design 4.7). Cascade step functions take one extra input `skip` (`tf.TensorSpec([], tf.bool)`); with p = 0 the `tf.where` ops are not built (trace-time gate) and the input is unused, so the step equals the cascade step without the feature. Non-cascade step functions keep today's input signature, and `train_step` passes `skip` only in the cascade branch (the non-cascade call is byte-identical to today's, which the goldens need):

```python
def train_step(self, task, batch_tuple, skip=False):
    batch_tuple, _ = self.split_batch(task, batch_tuple)
    if self.cascade is None:
        out = self._steps[task](*batch_tuple)                                 # unchanged
    else:
        out = self._steps[task](*batch_tuple, tf.constant(bool(skip)))
    ...                                                                        # result dict as today
```

`fit` draws `skip` only when `self.cascade is not None and cfg.cascade_skip_prob > 0` (otherwise `skip=False`). A skip changes no shapes, so no retracing. Logged: `train/cascade_skip_frac` (window mean of 0/1) only when p > 0. Validation never skips. Not used in the milestone-8 recipes (p = 0).

## 7. BN recalibration

Scope (v5): recalibration is defined for the cascade only. `--bn-recal-batches K > 0` without `--cascade-from` is rejected by the parser (`parser.error`, section 10), so level-4 runs and the Q18 no-cascade level-2 baseline can never recalibrate. `recalibrate_bn` additionally raises `RuntimeError("bn recalibration needs a cascade trainer")` when `self.cascade is None` (defence in depth for direct API calls; the parser rule is the user-facing owner).

When it runs (fresh vs resume): recalibration runs ONCE, at the start of a fresh run (`trainer.step == 0`, no `ckpt/last`), after the weight copy and before the first training step. On resume (`ckpt/last` exists, step > 0) it is NOT repeated: the recalibrated moving statistics are BN non-trainable variables of the trainable model and come back with `restore_last()`; only the Python-side freeze flag (`stats_frozen`, not checkpointed) is re-applied from the saved config.json `bn_recalibration.freeze` (section 9 step 5). A run preempted before its first checkpoint has no `ckpt/last`, so it restarts as a fresh run and recalibrates again from the copied level-4 statistics (same deterministic plan).

New `Trainer.recalibrate_bn(num_batches, freeze, val=True)`, called by main.py once, after the weight copy and before `fit`, only for a fresh cascade run (step 0, not resumed). `val=True` (main.py always) runs steps 1 and 5; `val=False` skips both `validate()` calls (unit tests only, to keep them fast); the event's `gap` table is then `{}` and `check_mixed_run --expect-bn-recal` does not require gap keys (it only prints them when present). Procedure:

1. Pre validation: `validate()` with PNGs off (`val_png_dir` temporarily None), logged at step 0 with prefix `bnrecal_pre/` (e.g. `bnrecal_pre/val_rigid/center_rms_mm`, `bnrecal_pre/val_rigid_trainmode/center_rms_mm`).
2. Task plan: largest-remainder allocation of K over the task ratio (each task ≥ 1; K = 64 with 1:1:2 → rigid 16, affine 16, deform 32), interleaved round-robin (deterministic, no RNG).
3. For batch b: linear tasks first call `self.controls[t].update(b)` **only if `t in self.controls`** (so G is resampled as during steps 0..K−1; overfit-synthetic runs and DALI runs without curriculum/G have `controls == {}` and skip the call), then `next(sources[t])`, `split_batch`, postprocess (flips on, same `pp` as training). b is the global plan index 0..K−1 (NIT 2): `fit` restarts at `update(0)` (control values are pure functions of the step), and the recalibration consumes K source batches and K postprocess draws, so K > 0 changes the data order vs K = 0 (reproducible on resume via the checkpointed generators). Each plan slot draws exactly one batch: a batch with non-finite input images consumes its slot without a forward or BN update (no replacement draw, so the loop always ends after K draws), a warning goes to stderr and `skipped_nonfinite[t]` is incremented; a task whose slots were all non-finite (0 used batches) → `RuntimeError`. Forward exactly as the training step's input path (pool → frozen → pre-warp), then `model(x_in, task, training=True)` EAGERLY, without a tape, with dropout as in training (dropout sits after every BN, so it does not affect statistics).
4. Exact averages: for every `BatchNormWithSeparateAffine` m in the parts the task runs (linear: `model.encoder`, `model.linear_decoder`; deform: `model.encoder`, `model.deform_decoder`), with c_m = number of earlier recal updates of m, set `m.bn.momentum = c_m / (c_m + 1)` before the call (Keras reads it per eager call), then c_m += 1. So moving = mean over that layer's batches of the batch statistics (first update replaces the level-4 values). Before the loop `orig = {id(m): m.bn.momentum for m in bn_layers}` is stored; in `finally` every layer gets its own saved value back (not `cfg.bn_momentum`, in case layers differ).
5. Post validation (statistics still unfrozen, so `_trainmode` = batch statistics): prefix `bnrecal_post/`.
6. If freeze: `model.set_bn_stats_frozen(True)` and `rebuild_steps()`.
7. Event `bn_recalibration` (metrics.jsonl, step 0) and config.json key `bn_recalibration`: batches per task (used), `skipped_nonfinite` per task, n_layers_updated, layers with 0 updates (must be empty), elapsed_s, freeze, and the gap table: `gap_pre/<set>/<metric>` and `gap_post/<set>/<metric>` = |eval − trainmode| for linear `center_rms_mm`, `pair_rms_mm` and deform `template_dice`, `pair_dice`.

Only the trainable model is touched; trainable variables, optimizer slots and `trainer.step` are untouched (forward-only, no `apply_gradients`, no `step.assign_add`). With `--gradient-checkpointing decoder|both`, a forward without a tape is expected to run the decoder once (`tf.recompute_grad` only recomputes under a tape), so there are no double updates during recalibration; this is asserted by a test (section 16, test 7: per-layer update count = number of that task's batches with `gradient_checkpointing='both'`), not assumed. After freezing, training uses moving statistics, so later `val_<task>_trainmode` equals `val_<task>` by construction; the gap is therefore only defined by the step-0 pre/post records. On resume (checkpoint present) the recalibration is not repeated; main.py re-applies `set_bn_stats_frozen(True)` + `rebuild_steps()` after the restore iff the SAVED config.json `bn_recalibration.freeze` is true (stats_frozen is a Python attribute, not checkpointed; the current args are not consulted, so a resume with different `--bn-recal-*` flags cannot change the frozen state). Procedure in section 9.

Flags: `--bn-recal-batches K` (int ≥ 0, default 0 = off; K > 0 requires `--cascade-from`, must be ≥ number of tasks; parser error with `--norm` other than `batch`), `--bn-recal-freeze` (bool, default true; ignored when K = 0).

## 8. Data, validation, metrics, PNGs at level 2

- Train data: the existing `LinearDataModule(args.data, level=2, ...)` builds `LinearPipeline` (rigid, affine) and `DeformPipeline` from `image_2.npy` / `labels_train18_2.npy` (manifest columns `image_l2`, `labels_l2`), 3 CPU threads; translations converted with the level-2 spacing (`P.linear_ranges`). No pipeline code changes.
- `tests/gpu/verify_dali_pipeline.py` gets `--level {4,2}` (argparse `type=int, choices=(4, 2)`, default 4): `main()` rebinds the module globals `LEVEL`, `SHAPE`, `SPACING` through a new helper `_set_level(level)` (`global` statement, from `P.LEVEL_SHAPE` / `P.LEVEL_SPACING`) right after `parse_args()`, before `env` is built and any check runs. Rebinding is not enough on its own: `main()` loads the template mask as the literal `"template_mask_4.npy"` in two places (checks b/c/d at line 963 and check j at line 989), which at level 2 would AND a (40,56,48) mask with the (80,112,96) `valid_mask` in `check_b_c` and FAIL b/c. Both loads become `"template_mask%s.npy" % P.level_suffix(LEVEL)` (one helper `_template_brain(args)` called in both places, after the rebind). Every other level use in the file already reads the module global at call time (`check_e` `level=LEVEL`, `check_l` `fd.load_level_images_and_labels(args.data, LEVEL, ...)` / `dl.DeformPipeline(LEVEL, ...)`, `dm.manifest_files(rows, LEVEL, ...)`, `pp.brain_S_matrix(LEVEL, ...)`, `phantom(SHAPE)`; no function captures them as default arguments, checked by grep). The module docstring line "real level-4 val cases with labels_train18_4.npy" becomes "real val cases of level `--level` with `labels_train18<suffix>.npy`". The memory job runs checks b, c, d, e (N = 12) and l at level 2 (AC7.1). Check a (pixel center) is level-independent.
- Validation: unchanged builders at level 2 (`fd.build_validation_sets(val_images, lin, 2, nv, ...)` with `_g` sets rendered TF-side at level 2; `fd.DeformValidationSet.build(..., 2, nvd, ...)` from `load_level_images_and_labels(data, 2, "val")`). Group sizes: existing rule in `main._build_dali_data`, nv = `args.val_group_size or min(N, 39)`, nvd = `args.val_group_size or min(DEFORM_N, 39)`. The m8 scripts do not pass `--val-group-size` (it would override both linear and deform, as the m7 header warns), so nv = N and nvd = DEFORM_N; test 13 asserts the flag's absence from `slurm/m8_level2_cascade.sh`. The frozen stage gets the pooled validation renders, as in training (AC7.3).
- Validation forward with the cascade: `_val_fn` / `_val_deform_fn` gain a cascade branch (same key, traced once): pool → frozen (eval) → pre-warp → trainable (eval, or train-mode BN with restored statistics as today) → compose. The frozen stage is RECOMPUTED inside the validation `tf.function` on every pass (a tape-free level-4 forward, cheap, identical to training, no cache invalidation, no ~0.8 GB host cache of U / U⁻); no frozen fields are cached anywhere. Never skips.
- Return types (pinned in v5): the non-cascade `_val_fn` is unchanged (returns the tensor A; `_val_errors` calls `.numpy()` on it as today). The cascade `_val_fn` returns a dict in both modes: eval `{"A": composed, "A_prev": A4}`, train-mode `{"A": composed}`. `_val_errors` branches at the Python level: `out = fn(images)`; `A = (out["A"] if self.cascade is not None else out).numpy().astype(np.float64)`; in eval mode with a cascade it also appends `out["A_prev"]` to `frozen_outs`. `_val_deform_fn` already returns a dict (`u, u_inv, v, v_raw`); its cascade eval branch adds `U, U_inv, V`, its cascade train-mode branch returns only the usual four keys; `_val_deform_errors` reads the four keys as today and, in eval mode with a cascade, appends `{U, U_inv, V}` to `frozen_outs`.
- Frozen-stage baselines (AC8.1): `Trainer._frozen_baseline_metrics(name, vs, groups, frozen_outs)`, cached by (name, len(groups)) like `_identity_cache`. Source of the frozen estimates: the cascade branch of `_val_fn` returns `{"A": composed, "A_prev": A4}` and the cascade branch of `_val_deform_fn` returns its usual composed outputs plus `"U", "U_inv", "V"` (the upsampled frozen fields of the same pass, eval mode; `V` = up(v4) for the inverse-consistency / centrality helpers). `validate()` collects these per group into `frozen_outs` during the eval-mode pass; on a cache miss `_frozen_baseline_metrics` computes the metrics from them (no extra `tf.function`, no extra forward), on a hit the extra outputs are ignored. The train-mode-statistics pass (`_trainmode`) does not feed the baselines (the frozen stage is always eval mode, so its outputs are identical in both passes anyway). Linear: `M.group_errors(A4, A_star, T_target, S_level2, ...)` → `M.summarize(..., "val_%s_frozen" % name)` (all summary keys, affine factor errors included). Deform: `_deform_errors(labels, U, U_inv, V, V, ...)` → `val_deform_frozen/{template_dice, pair_dice, neg_jac_pct, inv_consistency_mm, inv_consistency_max_mm, ncc_template, ncc_pair, ...}`. Same groups, N, S and level-2 labels as the composed metrics.
- PNGs (AC7.4): existing writers with the composed A (linear) and composed u (deform); level-2 spacing and mid-slice z = 48 come from `P.LEVEL_SHAPE[self.level]`.
- Best files unchanged (AC8.4): `ckpt/best` on `val/mean_pair_rms_mm` (composed), `ckpt/best_deform` on `val_deform/pair_dice_loss` (composed).
- Throughput (AC7.5): with the cascade on, `fit` also accumulates `train_<task>/data_ms` and `train_<task>/compute_ms` (window means); the checker prints the per-task data-wait fraction data/(data + compute) as INFO.

## 9. Checkpoints, resume, provenance

- `RunCheckpoint(..., extra=dict(trainer.checkpoint_variables(), frozen_stage=frozen.model))` (key `frozen_stage`, only in cascade runs; level-4 checkpoints unchanged).
- config.json `cascade`: `{source_checkpoint, source_config, source_level: 4, source_step, fingerprint: {n_vars, abs_sum, sq_sum}, S_rel_diff_vs_level4, skip_prob}`.
- Fresh run: frozen from `--cascade-from` (source), trainable ← `copy_weights_from(frozen.model)`. Write order (main.py today writes config.json once at line ~247, then `logger.log_params`, then `fit`): main.py writes config.json once as today (with `init_weights` = the cascade record and `cascade`), creates the logger and logs params, runs `trainer.recalibrate_bn(...)` if K > 0 (it needs the logger for the step-0 `bnrecal_pre/post` records and the event), then calls `C.write_config` again with the same `extra` plus `bn_recalibration`, all before `trainer.fit()`. The first `save_last` happens inside `fit`, so every checkpoint has a matching config.json (a run preempted after its first checkpoint resumes with the saved `bn_recalibration.freeze`). With K = 0 the second write is skipped.
- Resume (`args.resume` and `ckpt/last` exists, i.e. `C.peek_last_step(.../ckpt/last) > 0`), in main.py, in this order:
  1. `saved = json.load(paths["config"])`; require `saved.get("cascade")` (else `RuntimeError("resume of a cascade run without config.json cascade: <path>")`, exit 1).
  2. Do NOT call `C.load_cascade_source` / `C.check_cascade_config` (the source path may be gone; `--cascade-from` must still be passed because it selects the cascade code path, but it is not resolved or read). `C.check_config_compatible(paths["config"], model_cfg)` as today.
  3. `frozen = FrozenStage.from_config(model_cfg)` (random init, float32 policy), `Trainer(..., cascade=frozen)`, `RunCheckpoint(extra=dict(trainer.checkpoint_variables(), frozen_stage=frozen.model))`, `restore_last()`.
  4. `weight_fingerprint(frozen.model)` vs `saved["cascade"]["fingerprint"]` (n_vars exact, abs_sum / sq_sum rel 1e-12) → `RuntimeError` naming both fingerprints on mismatch.
  5. Re-freeze BN iff `(saved.get("bn_recalibration") or {}).get("freeze") is True` (the saved record, not the current args), then `trainer.rebuild_steps()`.
  6. `C.write_config(paths["config"], model_cfg, A.args_to_dict(args), extra={"param_counts": ..., "data": data_desc, "init_weights": saved.get("init_weights"), "cascade": saved["cascade"], "bn_recalibration": saved.get("bn_recalibration")})`: the three provenance blocks are carried over VERBATIM (today's code writes `init_info`, which is `None` on resume; the cascade branch replaces it with the saved value). So `check_mixed_run --expect-cascade` / `--expect-init-weights full` still pass on a resumed long run.
  A fresh cascade run (no `ckpt/last`) never reads config.json; it builds the blocks from `load_cascade_source` and the recalibration.
- Fresh vs resume (NIT 4): main.py computes `cascade_resume = bool(args.resume) and C.peek_last_step(<run>/ckpt/last) > 0` ONCE, before any model is built; it selects `FrozenStage.from_config` (resume) vs `load_cascade_source` + `FrozenStage.from_checkpoint` (fresh) and gates the recalibration, and on a resume `restore_last()` must return True (else `RuntimeError`). `--resume false` on an existing run dir is a fresh run (today's semantics): the source is read again, config.json is overwritten, and a second `bn_recalibration` event is appended to metrics.jsonl. A saved `bn_recalibration` key that was absent stays absent on resume (it is copied only when present).

## 10. CLI flags, config plumbing, off-by-default guarantee

| Flag (runtime/args.py) | Type / default | Validation (parser.error unless noted) | TrainConfig / use |
|---|---|---|---|
| `--cascade-from PATH` | str, None | requires `--level 2`; mutually exclusive with `--init-weights`; requires `--deform-residual-affine false`; on a FRESH run, resolution, source config, source level 4 and config equality are checked in main.py (`C.load_cascade_source`, `C.check_cascade_config`; failure → `SystemExit` with the message, exit 1); on a resume the path is not read (section 9) | `Trainer(cascade=FrozenStage)` |
| `--cascade-skip-prob p` | float, 0.0 | finite, 0 ≤ p < 1; p > 0 needs `--cascade-from` | `TrainConfig.cascade_skip_prob` |
| `--bn-recal-batches K` | int, 0 | K ≥ 0; K > 0 requires `--cascade-from` (`parser.error("--bn-recal-batches > 0 needs --cascade-from (BN recalibration is defined for the level-transfer cascade only)")`), needs ≥ len(task ratio) batches, `--norm batch`, and not `--dry-run` | main.py → `trainer.recalibrate_bn` on a fresh run only (never on resume, section 7) |
| `--bn-recal-freeze` | str2bool, True | — | same |
| `--val-pair-chunk C` (v5) | int, 16 | C ≥ 1 | `TrainConfig.val_pair_chunk` → `_deform_errors(chunk=C)`; 16 = today's literal, so default runs are unchanged; only the level-2 validation-memory fallback (section 11) changes it |

Gating (FR13): `Trainer(cascade=None)` (default) builds exactly today's step and validation functions (Python `if self.cascade is not None` at trace time); `cascade_skip_prob == 0` builds no skip ops; recalibration code runs only when called; the per-task timing keys are logged only with a cascade; `warm_start_model`'s new branch runs only when source level ≠ model level. Level-4 runs never construct a `FrozenStage` (parser requires `--level 2`). Goldens `tests/fixtures/m7_linear_golden.json` and `lncc_deform_golden.json` must keep max |diff| 0. New config.json keys (`cascade`, `bn_recalibration`) are written only when their feature is on; new args keys appear in `args` with off defaults (as every earlier milestone).

## 11. Memory measurement and group sizes

New script `tests/gpu/measure_level_memory.py` (the m6 `measure_deform_memory.py` stays as is for reproducibility; the new script reuses its structure: `configure_gpu`, per-config build, OOM caught, larger N skipped, JSON merged by config key).

- Options: `--level {4,2}` (default 2), `--tasks rigid,affine,deform`, `--n` (comma list), `--widths narrow,default,tiny`, `--cascade true,false` (true: a `FrozenStage.from_model` of a random level-4 model with the same config), `--gradient-checkpointing none,integrator,decoder,both` (deform only; linear uses none), `--amp false,true` (sets `mixed_float16` before building the trainable model and an AMP optimizer; the frozen stage stays float32; the policy is reset to float32 after each config), `--lambda-ncc-template 1 --lambda-ncc-pair 1` (m8 recipe default), `--val-pass false,true` (v5, deform only: after the measured steps, `reset_memory_stats`, then one validation pass on the synthetic group exactly as `_val_deform_errors` does it: the eval-mode `trainer._val_deform_fn(n)` forward (cascade branch when `--cascade true`) followed by `trainer._deform_errors(labels, u, u_inv, v, v_raw, seed=0, chunk=16, images=images)`; records `val_peak_mb` and `val_status ok|oom|error`; an OOM here does not mark the training config as OOM), `--warmup-steps 2`, `--steps 10` (≥ 10 measured steps; mean and median reported), `--gpu-memory-limit-mb`, `--out`.
- Data: synthetic, no DALI: linear `tests.runtime_helpers.synthetic_group(level, n, task)`, deform `fd.synthetic_deform_group(level, n)`.
- Per result: `level, task, n, widths, cascade, gradient_checkpointing, amp, lambda_ncc_*, params, status ok|oom|error|skipped, peak_mb, warmup_peak_mb, resident_mb, step_ms (median), step_ms_mean, step_ms_all, frozen_ms (median time of the frozen forward alone, cascade only), trace_counts, finite`, plus `val_peak_mb`, `val_status`, `val_ms` when `--val-pass true` (deform only; `null` otherwise).
- AMP keeps float32: SVF head and linear heads (`dtype='float32'`), `diffeo` (casts float16 → float32), losses (`_float`), LNCC (float32 always), `transform_image` grids (float32), BN math (Keras computes in float32). Only convs / dense activations become float16.

Estimate (to be replaced by measurement): only the stem, L2, D2, D1, the SVF head, integration, warps, one-hot and LNCC scale by 8 at level 2; L3–L6 and D3–D5 keep their level-4 size. A per-voxel tally gives ≈ 7–10 GB per image for a trainable deform step at default widths (the hi-res parts are ≈ 0.9 GB of the 2.15 GB/image at level 4, × 8, plus the unchanged low-res parts, scaled by the 1.35 ratio of measured to tallied memory at level 4); 8 × 2.15 ≈ 17 GB/image is the upper bound. So deform N = 8 likely needs checkpointing and/or AMP; N 2–4 may fit without. Linear ≈ 2–3 GB/image (vs ~1 GB at level 4) → N 12–16.

Cluster grid (job (a), RTX 8000 / A6000 48 GB): deform N {2, 3, 4, 6, 8} × checkpointing {none, integrator, both} × AMP {false, true}, cascade true, default widths, LNCC 1/1; deform N {2, 4} cascade false (frozen-stage share, NF1); rigid and affine N {4, 8, 12, 16} × AMP {false, true}, cascade true; rigid N {8} cascade false.

Group-size rule (report): per task, fit peak(N) ≈ resident + N · per_image on the measured points. Deform: the cheapest option (order none < integrator < both < AMP+none < AMP+both) that fits N = 8 at ≤ 44 GB; if none does, the largest N ≥ 4 with any option; below 4 the report flags it and stops before the smoke run. Linear: N = 16 if ≤ 44 GB without AMP, else the largest fitting N ≥ 8. The three-task process peak = max(linear, deform) step peak (+ resident weights of both stages and Adam slots), confirmed by the smoke run's `gpu_peak_mb`. "Fits the 40 GB A100" only if ≤ 36 GB; stated per recipe. Validation memory is measured, not assumed (v5): the linear validation forward is a tape-free subset of the training step, but `_deform_errors` runs eagerly with pair chunks of 16 independent of DEFORM_N (gathered 19-channel one-hot of 16 level-2 volumes ≈ 1.05 GB plus warp temporaries and pair LNCC), so the deform cascade configs are measured with `--val-pass true` and the report states `val_peak_mb` next to the training peak for the chosen DEFORM_N. If `val_peak_mb` > 44 GB (or `val_status` is `oom`) at the chosen option, the fallback is a smaller pair chunk: `TrainConfig.val_pair_chunk` (new, default 16 = today's literal, so level-4 validation is unchanged; flag `--val-pair-chunk`, int ≥ 1) passed to `_deform_errors(chunk=...)`, set to 8 or 4 in the recipe's `EXTRA_ARGS`. The smoke run's step-0 `bnrecal_pre` validation is the confirming check on real data.

## 12. SLURM recipes

Both scripts copy the m7 template (`slurm/m7_three_task.sh`): `#SBATCH -A qtim`, `-p rtx8000`, `--gpus=1`, `-N 1`, `--ntasks-per-node=1`, `--cpus-per-task=3`, `--mem=64G`, `--time`, `--output/--error` under `.../GroupwiseReg/logs/%x-%j.{out,err}`, results under `.../GroupwiseReg/results`, container `/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif`, `</dev/null` on every singularity call, `set -euo pipefail`. No `entre` anywhere. One deviation from the m7 template in both scripts: the node-local temp root is an override, `SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"` in the variable block, then `export TMPDIR="$SCRATCH_ROOT/$JOB_ID"` (cluster behaviour unchanged: `/scratch/$SLURM_JOB_ID`; the default keeps the literal `/scratch/` for the AC10.2 grep). This and the existing `PROJ`/`REPO`/`RESULTS`/`DATA`/`SIF` overrides are what lets the local stub test (test 13) run the script without touching `/autofs` or `/scratch`.

(a) `slurm/m8_memory.sh`. Variable block as m7 (`PROJ`, `REPO`, `DATA`, `SIF`, `JOB_ID`, `OUT_DIR="$PROJ/scratch/m8"`), then, each call inside `singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF"` with `</dev/null` and `|| rc_x=$?`:

```bash
python tests/gpu/verify_dali_pipeline.py --data "$DATA" --level 2 --checks b,c,d,e,l \
    --num-threads "${SLURM_CPUS_PER_TASK:-3}" --n-throughput 12 \
    --scratch "$OUT_DIR/dali" --out "$OUT_DIR/verify_level2.json"
M="python tests/gpu/measure_level_memory.py --level 2 --widths default --lambda-ncc-template 1 --lambda-ncc-pair 1 --warmup-steps 2 --steps 10 --out $OUT_DIR/memory_level2.json"
$M --tasks deform --n 2,3,4,6,8 --cascade true  --gradient-checkpointing none,integrator,both --amp false,true --val-pass true   # 30 configs
$M --tasks deform --n 2,4       --cascade false --gradient-checkpointing none --amp false                      # 2 (frozen-stage share)
$M --tasks rigid,affine --n 4,8,12,16 --cascade true  --gradient-checkpointing none --amp false,true           # 16
$M --tasks rigid  --n 8         --cascade false --gradient-checkpointing none --amp false                      # 1
```

All four memory invocations merge into one JSON (by config key, as `measure_deform_memory.py`). `--num-threads` is the requested 3 CPUs (AC7.1), so check e's throughput describes the training configuration. Exit code: the first non-zero of the five calls, every call still runs (the memory script records OOMs as results and returns 0 unless it crashes). Time: 49 configs × ~2 min (trace + 12 steps at level 2, provisional) + 30 validation passes × ~0.5 min + ~15 min DALI checks ≈ 2.1 h → `--time=04:00:00` (still ≈ 2× margin, since level-2 step times are unmeasured).

(b)/(c) `slurm/m8_level2_cascade.sh`. Every env default is assigned ONCE in the variable block and only the plain variables are used afterwards (the script runs under `set -euo pipefail`, so a later `"$VAR"` on an unset variable would abort):

```bash
PROJ="${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}"
REPO="${REPO:-$PROJ/brain-group-rad}"; DATA="${DATA:-$PROJ/data/npy_v2}"; RESULTS="${RESULTS:-$PROJ/results}"
SIF="${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}"
JOB_ID="${SLURM_JOB_ID:-manual}"; RUN_NAME="${RUN_NAME:-m8_level2_cascade_$JOB_ID}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"          # later: export TMPDIR="$SCRATCH_ROOT/$JOB_ID"
STEPS="${STEPS:-8000}"; VAL_EVERY="${VAL_EVERY:-500}"; WARMUP="${WARMUP:-500}"
N="${N:-}"; DEFORM_N="${DEFORM_N:-}"
CASCADE_FROM="${CASCADE_FROM:-$RESULTS/m7_three_task_lncc/ckpt/best}"
GRAD_BALANCE="${GRAD_BALANCE:-gradnorm}"
BN_RECAL_BATCHES="${BN_RECAL_BATCHES:-64}"; BN_RECAL_FREEZE="${BN_RECAL_FREEZE:-true}"
SKIP_PROB="${SKIP_PROB:-0}"; GRAD_CKPT="${GRAD_CKPT:-none}"; AMP="${AMP:-false}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
case "$N" in ''|*[!0-9]*) echo "ERROR: set N (linear group size) from the memory job" >&2; exit 2;; esac
case "$DEFORM_N" in ''|*[!0-9]*) echo "ERROR: set DEFORM_N from the memory job" >&2; exit 2;; esac
case "$BN_RECAL_BATCHES" in ''|*[!0-9]*) echo "ERROR: BN_RECAL_BATCHES must be an integer" >&2; exit 2;; esac
# Guard only for fresh runs: a resume / requeue (ckpt/last exists) does not read the source (section 9).
if [ ! -f "$RESULTS/$RUN_NAME/ckpt/last/checkpoint" ]; then
  if [ ! -d "$CASCADE_FROM" ] && [ ! -f "$CASCADE_FROM.index" ]; then
    echo "ERROR: CASCADE_FROM $CASCADE_FROM is neither a checkpoint directory nor a prefix" >&2; exit 2; fi
fi
case "$GRAD_BALANCE" in
    gradnorm) BAL_ARGS="--task-grad-norm 12" ;;            # --task-loss-weights not passed (= none)
    deform64) BAL_ARGS="--task-loss-weights deform:64" ;;  # global clip 12 only
    *) echo "ERROR: GRAD_BALANCE must be gradnorm or deform64" >&2; exit 2 ;;
esac
RECAL_CHECK=""
if [ "$BN_RECAL_BATCHES" -gt 0 ]; then RECAL_CHECK="--expect-bn-recal true"; fi
```

- main.py: `--data "$DATA" --results "$RESULTS" --run-name "$RUN_NAME" --level 2 --cascade-from "$CASCADE_FROM" --group-size "$N" --deform-group-size "$DEFORM_N" --task-ratio rigid:1,affine:1,deform:2 --steps "$STEPS" --val-every "$VAL_EVERY" --warmup-steps "$WARMUP" $BAL_ARGS --lambda-ncc-template 1 --lambda-ncc-pair 1 --bn-recal-batches "$BN_RECAL_BATCHES" --bn-recal-freeze "$BN_RECAL_FREEZE" --cascade-skip-prob "$SKIP_PROB" --gradient-checkpointing "$GRAD_CKPT" --amp "$AMP" --global-transform true --global-rot-max-deg 45 --global-trans-max-mm 10 --global-start-step 0 --global-ramp-steps 0 --curriculum false --val-png true --num-threads "${SLURM_CPUS_PER_TASK:-3}" --log-every 25 --log-backend jsonl $EXTRA_ARGS </dev/null`. (`--bn-recal-batches 0` passes the parser; `--bn-recal-freeze` is then ignored.)
- After main.py: `check_mixed_run.py "$RESULTS/$RUN_NAME" --tasks rigid,affine,deform --steps "$STEPS" --expect-init-weights full --expect-enc-grad true --expect-cascade true --grad-balance-rule true $RECAL_CHECK`; exit code main.py's if non-zero, else the checker's (as m7).
- Header text also states: the `CASCADE_FROM` existence guard is skipped when `$RESULTS/$RUN_NAME/ckpt/last/checkpoint` exists (resume / requeue; `--cascade-from` is still passed to select the cascade path but not read).
- Header text: the C4 rule — KEEP iff the checker prints `grad_balance KEEP`; `FALLBACK` or `UNDECIDED` → treat as FALLBACK and inspect the run; on FALLBACK while the gradnorm long run is already running (Jay's submit-early preference, message 5): `scancel <our own m8long job id>` (never a job we didn't submit), then resubmit the long run with `GRAD_BALANCE=deform64`. The resubmission is a new job and needs Jay's approval (cluster rules, C8). Jay's message 12 already agrees to the deform64 fallback recipe, so the approval request names only partition, GPU type, CPUs, memory and `--time`. Before resubmitting, `squeue -u jn85` must show fewer than two active jobs of ours (running plus pending), counted after the `scancel` has taken effect.
- sbatch lines (cluster order C8):
  - `sbatch --job-name=m8mem --time=04:00:00 --mem=64G slurm/m8_memory.sh`
  - `RUN_NAME=m8_smoke N=<n> DEFORM_N=<nd> STEPS=200 VAL_EVERY=100 WARMUP=50 sbatch --job-name=m8smoke --time=<T_smoke> --mem=64G slurm/m8_level2_cascade.sh`, with T_smoke = 1.5 × (200 × s_step + 4 × s_val + s_recal + 600 s), floor 01:30:00, rounded up to 15 min; provisional **02:00:00** until job (a) reports. Inputs from job (a)'s `memory_level2.json` at the chosen N / options: s_step = the ratio-weighted mean of the per-task median `step_ms` (1:1:2 → (rigid + affine + 2 · deform) / 4); s_val = one full level-2 validation pass = Σ over validation sets (4 linear: rigid, affine, rigid_g, affine_g; 1 deform) of n_groups(set) × 2 (eval + trainmode) × 0.5 × median `step_ms` of that set's task (a tape-free forward with frozen stage ≈ 1/3 of a training step, × 1.5 for the eager metrics and pair Dice/LNCC); four such passes in the smoke run (bnrecal_pre, bnrecal_post, step 100, step 200); s_recal = K × (forward ≈ 0.5 × task step, eager) with K = 64 in the 1:1:2 plan; 600 s covers the traces (3 train steps, val functions, recal is eager). The measured s_step, s_val (from the smoke log's validation timestamps) and s_recal (`bn_recalibration.elapsed_s`) are recorded in the report and replace the estimates for the long run.
  - `RUN_NAME=m8_level2_cascade N=<n> DEFORM_N=<nd> sbatch --job-name=m8long --time=<T> --mem=64G slurm/m8_level2_cascade.sh`, T = 1.3 × (8000 × mean step s + 16 × val s) from the smoke run (provisional 24:00:00 until measured).
  - A100 override: the header ALWAYS documents `sbatch -p dgx-a100 ...` (same CPUs 3, `--mem=64G`) together with the measured peak of the chosen recipe (from job (a) / the smoke `gpu_peak_mb`) and an explicit verdict line: `fits the 40 GB A100 (peak X GB <= 36 GB)` or `does NOT fit the 40 GB A100 (peak X GB > 36 GB)` (AC10.3). Before job (a) reports, the header says `peak: unmeasured (fill in from scratch/m8/memory_level2.json)`.

## 13. Checker (`tests/gpu/check_mixed_run.py`)

New options (all default off, so level-4 directories check exactly as before):
- `--expect-cascade true`: config.json `cascade` has `source_checkpoint`, `source_level == 4`, integer `source_step`, `fingerprint.n_vars > 0`; `init_weights.via == "cascade"`; final-step validation has finite `val_<task>_frozen/center_rms_mm`, `pair_rms_mm` per linear task and `val_deform_frozen/{template_dice, pair_dice}`. INFO lines `cascade_gain <set>`: composed vs frozen (center / pair RMS, Dice) with `better`/`worse` (never a failure; AC8.2 is read from them for the long run), and per-task data-wait fraction.
- `--expect-bn-recal true`: a `bn_recalibration` event exists, its `n_layers_updated > 0` and `layers_without_updates == []`; INFO lines with gap_pre / gap_post per set and metric when the gap table is non-empty (an empty table, from `val=False`, is not a failure).
- `--expect-skip true`: `train/cascade_skip_frac` logged and finite.
- G gap at level 2 (AC8.3; no new metric code, existing `M.summarize` keys): with `--expect-cascade true` the checker prints INFO lines `g_gap <task>` comparing `rigid` vs `rigid_g` and `affine` vs `affine_g`, for the composed sets (`val_<set>/...`) and the frozen baselines (`val_<set>_frozen/...`): `{center,pair}_rms_mm`, `{center,pair}_rot_deg`, `{center,pair}_trans_mm`, and for affine additionally `{center,pair}_{scale_pct, shear_pct, qr_rot_deg}`. The report's G-gap table lists the same keys next to the level-4 LNCC-run numbers (rigid pair 5.45 vs 5.53 mm; affine pair 6.92 vs 9.45 mm, rotation 4.9 → 7.1°, shear 4.0 → 5.3 %), which answers Jay's message 11 question (does the affine G gap shrink at level 2). Never a failure.
- `--grad-balance-rule true` (implies the enc-grad keys): per linear task the ratio of the medians, median(`train_<task>/diag_enc_grad_norm_eff`) / median(`train_deform/diag_enc_grad_norm_eff`), each median over log records whose `step` field (the window end) is ≥ `--grad-balance-from-step` (default 50; with `--log-every 25` the first used window is steps 26–50; the help text states exactly this) (same computation as `_check_enc_grad`, plus the step filter); prints `grad_balance KEEP task-grad-norm` when both ratios are in [0.1, 10], else `grad_balance FALLBACK deform:64`, with the ratios. Missing deform or linear values → `grad_balance UNDECIDED (treat as FALLBACK and inspect the run)`. Informational: it never fails the run (the PASS/FAIL tally ignores it).

## 14. Label sampling (design 11 addendum): (B) defer to milestone 9 unless Jay chooses (A)

Decision status (AC12.1): **(B) applies unless Jay chooses (A); nothing in milestone 8 depends on it.** Jay's message 2 ("add it to the design doc for milestone 8") asked for the design entry, which exists (design.md section 11 addendum); it did not choose between implementing now and later. The milestone-8 report must list this as an open decision. If Jay picks (A), it is implemented per AC12.2 behind `--dice-labels train18` (default) without touching the cascade code.

Reasons for (B): (1) the full-label export does not exist and the core list and K are Jay's open choices, so an implementation now could only be tested on phantoms and would need a second pass once real labels arrive; (2) at level 2 one-hot memory is not the problem (19 float32 channels = 65 MB per image vs ≈ 7–17 GB per image for the step), so sampling buys nothing for milestone 8's memory; the gain is at level 1 (≈ 520 MB per image for 19 channels, 1.4 GB fp16 for 103); (3) milestone 8 already changes the train step, validation, checkpointing and BN, and keeping train18 isolates the cascade's effect on Dice. The cascade does not depend on it. Milestone-8 runs use train18 at level 2.

The full label set already exists in the exporter (`preprocessing/export.py`): label set `synthseg` = the 103-id SynthSeg `--parc` map remapped to contiguous uint8 ids through the fixed LUT (`get_lut("synthseg")`, read from the raw ids in `labels_synthseg.nii.gz`, like train18), file `labels_synthseg.npy`, ids recorded in `<export_root>/label_luts.json` (luts `synthseg`), manifest `labels_l<k>` columns built automatically (`";".join("labels_%s%s.npy" ...)` for every set whose level list contains k). It is full resolution only today (`PYRAMID_LABEL_SETS = ("train18",)`, `ls_levels = levels if ls in PYRAMID_LABEL_SETS else [1]`). No new file names or label tables are invented.

Code change, done in milestone 8 regardless of (A)/(B) (agent, small, off the training path):
- `preprocessing/export.py`: replace `PYRAMID_LABEL_SETS` by `LABEL_SET_LEVELS = {"train18": (1, 2, 4), "synthseg": (1, 2)}` (any other set: `(1,)`), `ls_levels = [k for k in levels if k in LABEL_SET_LEVELS.get(ls, (1,))]`; the existing `ValueError` check fires when `ls_levels` is empty, keeping the current message for `(1,)` sets ("label set %r is exported at full resolution only; add 1 to --levels", matched by the existing test) and "label set %r is exported at levels 1,2 only; ..." for synthseg. Tests are pytest (as the rest of `preprocessing/tests`). `_2` uses the same `block_mode` (smallest-id tie-break) as train18. No `labels_synthseg_4.npy` (level 4 keeps train18; many fine labels are a few voxels at 4 mm). Docstrings and the `--label-sets` help text updated.
- Tests in `preprocessing/tests/test_export.py`: synthseg at `--levels 1,2,4` writes `labels_synthseg.npy` + `labels_synthseg_2.npy` (no `_4`), `_2` equals `block_mode(full, 2)`, manifest `labels_l2 == "labels_train18_2.npy;labels_synthseg_2.npy"`, `labels_l4` has train18 only, `label_levels` column `synthseg:1,2`; the existing assertion `not os.path.exists(".../labels_synthseg_2.npy")  # full resolution only` (test_export.py line ~119) is updated to the new rule; train18 outputs byte-identical to before.

Jay's commands (after that change is pulled; same array script and chunking as the v2 export, so the manifest part names are the same and the merge replaces them; `_needs_write` sees the missing synthseg files, rewrites those cases atomically with identical image / train18 contents, a few seconds per case):
- Train: `env LABEL_SETS=train18,synthseg sbatch --array=0-$((N-1))%2 --export=ALL,SPLIT=train preprocessing/slurm/export_array.sh` (N chunks of 200 as in the script header).
- T1 val: `env LABEL_SETS=train18,synthseg OPTIONAL_LABEL_SETS=lumir10 sbatch --array=0-0%2 --export=ALL,SPLIT=val,CHUNK_SIZE=100000,INCLUDE_TEMPLATE=0 preprocessing/slurm/export_array.sh`.
- After both arrays finish: `sbatch --array=0-0%2 --export=ALL,MERGE_ONLY=1 preprocessing/slurm/export_array.sh` (export-merge).
- Outputs: per case `labels_synthseg.npy` and `labels_synthseg_2.npy`; manifest `labels_l1` / `labels_l2` list them next to train18 (3384 train + 39 T1 val); ids from `label_luts.json`.
- Label table (AC12.3): `<export_root>/label_luts.json`, lut `synthseg`, written by the exporter from `preprocessing/luts.py` (`_build(_synthseg_entries())`), already holds `map` (raw SynthSeg id → contiguous id) and `names` (contiguous id → name), i.e. the columns contiguous id, SynthSeg id and name. The core flag is not stored: it is derived at load time from Jay's raw-id core list mapped through `map`. No new table file.
- Jay's inputs (the report's "Jay must export / provide" list): (1) the export commands above (train + T1 val + merge) producing `labels_synthseg.npy` / `labels_synthseg_2.npy` and the updated `label_luts.json`; (2) the core-label list as raw SynthSeg ids (e.g. 4/43 lateral ventricles, 14, 15, 17/53 hippocampi, ...); (3) K, the number of sampled non-core labels per step (design suggests 8–12).

## 15. Error handling

| Operation | Failure | Recoverable? | Caller receives / logging |
|---|---|---|---|
| Parse cascade / recal / skip flags | invalid combination or value (table 10) | no | `parser.error` → exit 2, message on stderr |
| `C.load_cascade_source` | path not a prefix/dir, no config.json next to it, source `args.level` ≠ 4 | no | `ValueError` → main.py `SystemExit("main.py: --cascade-from ...: <msg>")`, exit 1, stderr |
| `C.check_cascade_config` | any ModelConfig field (seed excepted) differs | no | `ValueError` naming the fields (AC1.2) → SystemExit as above |
| `FrozenStage.from_checkpoint` restore | missing variables (`assert_existing_objects_matched`) | no | TF `AssertionError` propagates (traceback), exit 1 |
| `copy_weights_from` | count / shape / name mismatch | no | `ValueError` (existing) propagates |
| Resume fingerprint check | frozen weights differ from config.json | no | `RuntimeError` naming both fingerprints, exit 1 |
| Resume of a cascade run | config.json missing or without `cascade` | no | `RuntimeError("resume of a cascade run without config.json cascade: <path>")`, exit 1 |
| Parse `--cascade-from` with `--deform-residual-affine true` | unsupported combination | no | `parser.error`, exit 2 |
| Parse `--bn-recal-batches K > 0` without `--cascade-from` | recalibration is cascade-only (v5) | no | `parser.error("--bn-recal-batches > 0 needs --cascade-from ...")`, exit 2 |
| `recalibrate_bn` called on a trainer without a cascade | API misuse (parser prevents it from the CLI) | no | `RuntimeError("bn recalibration needs a cascade trainer")` |
| Exporter, synthseg with `--levels 4` only | no allowed level requested | no | `ValueError` (message names the allowed levels), export exits non-zero |
| `recalibrate_bn` source | `StopIteration` from a source | no | `RuntimeError("bn recalibration: source <task> exhausted")` |
| `recalibrate_bn` batch | non-finite input images | yes | the batch consumes its plan slot without an update (no replacement draw), warning on stderr, `skipped_nonfinite[task]` in the event; if a task then has 0 used batches → `RuntimeError` |
| `measure_level_memory.py --val-pass` | OOM / exception in the validation pass | yes | `val_status: oom|error` on that result; the training result and later configs are unaffected |
| `recalibrate_bn` coverage | a BN layer of a used part got 0 updates | no | `RuntimeError` listing the layers (stale level-4 statistics otherwise) |
| `recalibrate_bn` momenta | any exception mid-way | no | `finally` restores `bn.momentum`; exception propagates |
| Cascade train step | non-finite composed loss / gradient | yes | existing skip path (no update, BN rollback, `nonfinite_step_skipped` event; patience 20) |
| Frozen baseline / validation | errors | no | propagate as validation errors do today |
| PNG writing | any | yes | existing: warning + `val_png_failed` event |
| `measure_level_memory.py` | OOM | yes | result `status: oom`, larger N of that config `skipped` |
| `measure_level_memory.py` | other exception | yes | result `status: error` with repr, traceback on stderr, continues |
| `verify_dali_pipeline.py --level` | invalid level | no | argparse error, exit 2 |
| SLURM train script | `CASCADE_FROM` missing on a fresh run (no `$RESULTS/$RUN_NAME/ckpt/last/checkpoint`), `N`/`DEFORM_N` unset, unknown `GRAD_BALANCE` | no | `exit 2`, message on stderr, nothing runs |
| Checker | missing / non-finite keys | — | `FAIL <criterion>` line, exit 1; grad-balance verdict and cascade_gain are INFO only |

Invariants and owners: frozen weights never change (Trainer: not in the optimizer var list, called outside the tape, `stop_gradient`; test AC2.1); one group seen by both stages (the step function derives the frozen input from the same post-processed tensor); composed A is centered (`compose_linear`, i.e. the trainable model's `GroupCentering`); residual SVF centered (`GroupRegLevelModel._call_deform`); config equality of source and level-2 model (`checkpoint.check_cascade_config`, before any weights are built into the trainer); level-4 bit-identity (trace-time gates in `Trainer`, parser rule that the cascade needs level 2; goldens).

## 16. Tests

New file `tests/test_cascade.py` (every TF test imports `tests.tf_setup` first; models use a new `TINY` config in `tests/runtime_helpers.py`: filters (8, 16, 16, 16, 16, 16), reductions (16, 8), dense (64, 32), decoder widths (8, 8, 8, 8, 8), dropout 0):

1. `TestWeightTransfer`: (a) level-4 model with random heads and BN stats → level 2 via a written checkpoint + `config.json` (`load_cascade_source`, `FrozenStage.from_checkpoint`, `copy_weights_from`): every variable array-equal, count = len(weights), init record equal to the section 2.2 cascade dict key by key (incl. `ignored_source_parts == []`, `source_step` = the saved step) and `main._init_weights_lines(record)` returns a `str`; (b) config differing in `dropout` → ValueError naming `dropout`; source level 2 → ValueError; missing config.json → ValueError; (c) units: for a constant u4 = c, `to_mm(upsample_displacement(u4), 2) == to_mm(u4, 4)` (1e-6) and the same rigid params give the same mm matrix at both levels; (d) S: a synthetic level-2 mask and its mode-pooled level-4 mask give `brain_second_moment` rel. Frobenius difference < 0.05 (measured value printed).
2. `TestRenderConsistency` (AC7.2): phantom rendered at level 2 with T (`transform_image`, T⁻¹) then 2×2×2 pooled vs the block-averaged phantom rendered at level 4 with the same T: interior NCC > 0.98.
3. `TestLinearCascade`: `compose_linear` vs a numpy reference (`tests/np_reference`-style barycenter: R A4, `scipy.linalg.logm/expm` Karcher iterations) to 1e-5 (float64: float64 R_raw / A_prev and `tfm.GroupCentering(iters=cfg.centering_iters, dtype="float64")`) / 1e-4 (float32: `model.centering`); barycenter of the result = I (rotation < 1e-4°, translation < 1e-4 mm); R = I → Center(A4) to 1e-5; pre-warp consistency: transform_image(transform_image(I, A4⁻¹), R⁻¹) vs transform_image(I, (R A4)⁻¹) interior NCC > 0.99 at level 2; pair predictions from the composed A equal A_j⁻¹ A_i.
4. `TestDeformCascade`: zero residual → (u, u_inv) = (U, U⁻) to 1e-6; two-level round trip with smooth random fields (max 2 voxels per level, `fd.random_svf`, T = 7): interior (3-voxel margin) |compose(u, u_inv)| and |compose(u_inv, u)| max ≤ 0.05, mean ≤ 0.01 level-2 voxels (measured values printed; tolerances revised in the report if needed). **Revised after measurement (FEAT-001):** at the 3-voxel margin the measured errors are u∘u⁻¹ max 0.076 / mean 0.011 and u⁻¹∘u max 0.206 / mean 0.015; the stages alone already give U∘U⁻ (upsampled level-4 pair) max 0.066 / mean 0.0074 and u2∘u2⁻ (level-2 residual) max 0.031 / mean 0.0069 (scaling-and-squaring inverse errors, T = 7; T = 11 changes them by < 5 %), and the 0.206 max comes from border clamping (u⁻¹ sampled up to max |u| ≈ 4.7 voxels outside the 3-voxel interior). The test therefore uses a margin of ceil(max |u|) + 3 voxels (8 here: measured max 0.060 / 0.068, mean 0.011 / 0.015) with max ≤ 0.1, mean ≤ 0.02 and the "composition adds no error" bound ≤ 1.25 × (upsampled-only + residual-only error) for both max and mean; residual mean over the group ≤ 1e-5 at every voxel; `deform_loss` on `out_c` equals a manual computation with the composed field.
5. `TestFrozenStageTraining`: tiny level-2 trainer with a cascade, 3 three-task steps (one of each task): frozen variables array-equal before/after, none of their refs in the optimizer's variables, frozen BN statistics unchanged, trace counts 1 per task, `n_vars_updated` equal to the no-cascade value; cascade linear and deform steps finite; (v5) a `cascade=None` trainer's step functions keep today's input signature (no `skip` TensorSpec) and `train_step(task, batch)` / `train_step(task, batch, skip=False)` both work on it.
6. `TestSkip`: draw reproducible for (seed, step); identity of the off path: a cascade trainer with p = 0 vs one with p = 0.5 whose steps are all fed `skip=False` through the step input, same seed and batches, 3 three-task steps: losses and variable sums equal (max |diff| 0, under `_OpDeterminism`), which proves the `tf.where` path is the identity when skip is false; separately, the p = 0 traced graphs contain no op whose name starts with `cascade_skip` (the skip ops are built under `tf.name_scope("cascade_skip")`); p → forced skip: pre-warp input equals the raw images (linear) and U = 0 (deform); alternating skip values do not retrace; `train/cascade_skip_frac` logged; `validate()` output identical for p = 0 and p = 0.9.
7. `TestBNRecal`: trainable variables, `optimizer.variables()` and `step` bitwise unchanged; every BN moving mean/variance in the used parts changed; exact average: K batches of one fixed group → moving mean = that group's batch mean (1e-5); freeze → every BN `stats_frozen`, steps retraced (trace count 2), statistics unchanged over 3 training steps; a deliberately mismatched intensity scale (level-2 inputs × 3) gives a smaller eval/trainmode gap after recalibration than before; frozen stage untouched; event fields present; `layers_without_updates == []`; each BN layer's momentum after the call equals its own value before (one layer deliberately built with a different momentum); with `gradient_checkpointing='both'` the per-layer update count equals that task's number of recal batches and the exact-average check still holds (no double updates). The exact-average test is mandatory: it is the check that Keras 2.10 re-reads `momentum` per eager call. (v5) A source that yields one non-finite batch among a task's slots: K draws in total (no replacement), `skipped_nonfinite == 1` for that task, the exact average is over the finite batches only; all slots of a task non-finite → `RuntimeError`. `val=False` → no `bnrecal_pre/post` records and an empty gap table. `recalibrate_bn` on a trainer with `cascade=None` → `RuntimeError`. Most of these subtests use `val=False`; one subtest uses `val=True` and checks the pre/post records and gap keys.
8. `TestCascadeResume`: short tiny cascade run with `RunCheckpoint(extra frozen_stage)`, save, rebuild trainer + random frozen, restore → frozen outputs on a fixed group identical (1e-7) and fingerprint equal; tampered fingerprint → RuntimeError; config.json without `cascade` → RuntimeError; after the resume, config.json `init_weights`, `cascade` and `bn_recalibration` are equal to their pre-resume values (main-level test through `main.main([...])` on the tiny overfit-synthetic setup, `--resume true` with a moved `--cascade-from` source to prove it is not read); BN `stats_frozen` re-applied from the saved record when `--bn-recal-batches 0` is passed on the resume; (v5) on a resume with `--bn-recal-batches 8` (same as the fresh run) recalibration is NOT repeated: metrics.jsonl holds exactly one `bn_recalibration` event (step 0) and the BN moving statistics right after the restore equal the checkpointed values (bitwise), i.e. no recalibration forward ran.
9. `TestCascadeAMP`: one cascade deform and one affine step under `mixed_float16`: finite, `u`, `u_inv`, `A` float32, frozen model variables float32, `frozen.model.compute_dtype == "float32"` for `from_checkpoint`, `from_model` (source model built under `mixed_float16`) and `from_config`; the global policy is `mixed_float16` again after each factory returns.
10. `TestCrossLevelWarmStart` (in `tests/test_runtime_infra.py`): `warm_start_model` from a level-4 run into a level-2 model, identical config (`mode full`, array-equal) and decoder-only difference (`mode partial`); each record equals the section 2.3 dict key by key (`ignored_source_parts`, `source_level`, `source_step` = the saved step, `cross_level == [4, 2]`, partial `n_restored_vars == len(pairs)`); `main._init_weights_lines(info)` returns a `str` for both (and for the cascade record in test 1(a)); level-4 → level-4 record unchanged (key set equals today's nine keys, no new keys).
11. Args (`tests/test_runtime_infra.py`): defaults (cascade None, skip 0, recal 0, freeze true); errors: `--cascade-from` with `--level 4`, with `--init-weights`, skip 1.0 / −0.1 / > 0 without cascade, `--bn-recal-batches 2` with three tasks (with `--cascade-from`), recal with `--norm instance` (with `--cascade-from`), (v5) `--bn-recal-batches 4` without `--cascade-from` at `--level 4` and at `--level 2` → parser error whose message contains `--cascade-from`; `--bn-recal-batches 0` without `--cascade-from` parses; `--val-pair-chunk 0` → parser error, default 16.
12. `tests/test_check_mixed_run.py`: synthetic run dirs: cascade pass; missing `val_rigid_frozen` → FAIL; bn-recal event missing → FAIL; grad-balance KEEP (ratios 1.3 / 0.7), FALLBACK (one ratio 40), UNDECIDED; (v5) step cut: records at steps 25, 50, 75 with a deliberately extreme value at step 25 only → the record at step 25 is excluded and the record at step 50 is included (KEEP); a `bn_recalibration` event with an empty gap table passes `--expect-bn-recal`; existing level-4 fixtures pass unchanged.
13. `tests/test_slurm_scripts.py` (new): for `slurm/m8_*.sh`: `#SBATCH -A qtim`, no `entre`, `--gpus=1`, `-N 1`, `--ntasks-per-node=1`, `--cpus-per-task=3`, `-p rtx8000`, `--mem=` ≤ 128G, `--time=`, log paths under `.../GroupwiseReg/logs`, the sif path, `TMPDIR=/scratch/`, `</dev/null`, `GRAD_BALANCE` switch with both recipes; `bash -n` passes (subprocess, skipped if bash is missing); no `${BN_RECAL_BATCHES:-`, `${SKIP_PROB:-`, `${GRAD_CKPT:-`, `${AMP:-`, `${BN_RECAL_FREEZE:-`, `${GRAD_BALANCE:-` occurrence after the variable block (the line of the first `case`); `CASCADE_FROM` guard accepts a directory or `<prefix>.index`; `m8_memory.sh` passes `--num-threads "${SLURM_CPUS_PER_TASK:-3}"` and four `measure_level_memory.py` invocations. `m8_level2_cascade.sh` does not contain `--val-group-size`; both scripts define `SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"` and `TMPDIR="$SCRATCH_ROOT/$JOB_ID"`. Stub run (bash only, skipped otherwise): in a temp dir `<tmp>`, a stub `singularity` (logs its argv to `<tmp>/calls.log`, exit 0) and stub `nvidia-smi` / `git` (exit 0; both are already guarded in the script, the stubs only keep the output quiet) first on PATH; environment `PROJ=<tmp> REPO=<tmp>/repo RESULTS=<tmp>/results DATA=<tmp>/data SIF=<tmp>/x.sif SCRATCH_ROOT=<tmp>/scratch CASCADE_FROM=<tmp>/ckpt` (a directory) `N=4 DEFORM_N=2`, `SLURM_JOB_ID` unset, and no other *recipe* env (`GRAD_BALANCE`, `BN_RECAL_BATCHES`, `BN_RECAL_FREEZE`, `SKIP_PROB`, `GRAD_CKPT`, `AMP`, `EXTRA_ARGS` removed from the inherited environment); `<tmp>/repo` exists. Assertions: exit 0, `calls.log` has exactly two invocations, the first contains `main.py`, `--level 2`, `--task-grad-norm 12`, `--bn-recal-batches 64`, the second contains `check_mixed_run.py` and `--expect-bn-recal true`; `<tmp>/scratch/manual` was created. Second case: same env with `N` unset → exit 2 and no `calls.log`. Third case: `GRAD_BALANCE=deform64` → first invocation contains `--task-loss-weights deform:64` and not `--task-grad-norm`. Fourth case (resume): `CASCADE_FROM=<tmp>/missing` (does not exist), `RUN_NAME=r` and an existing file `<tmp>/results/r/ckpt/last/checkpoint` → exit 0 and two invocations (the guard is skipped); the same with no `ckpt/last/checkpoint` → exit 2. `m8_memory.sh` gets the same stub run (same env minus `N`/`DEFORM_N`/`CASCADE_FROM`; five invocations, the first `verify_dali_pipeline.py --level 2`).
14. `preprocessing/tests/test_export.py` (pytest): the synthseg pyramid tests of section 14.
15. `tests/test_measure_memory.py` (new, GPU only, skipped without a GPU): `measure_level_memory.main([... --level 2 --tasks deform,rigid --n 2 --widths tiny --cascade true --steps 2 --warmup-steps 1 --val-pass true --gpu-memory-limit-mb 6000 --out <tmp>])` returns 0, every result has status `ok` or `oom`, and the deform result has `val_status` in {ok, oom} with a numeric `val_peak_mb` when ok (rigid: `val_peak_mb` null).
16. Args additions: `--cascade-from` with `--deform-residual-affine true` → parser error.
16a. `tests/test_verify_dali_level.py` (new, CPU, no DALI needed): (i) source scan of `tests/gpu/verify_dali_pipeline.py`: no literal `template_mask_4` and no literal `labels_train18_4` anywhere in the file (regex over the file text; the docstring line is rewritten per section 8); (ii) `verify_dali_pipeline._template_brain(args)` after the module globals are rebound to level 2 (the test calls the same rebind helper `_set_level(2)` that `main()` uses) loads `template_mask_2.npy` from a temp `template/` dir (written with `P.LEVEL_SHAPE[2]`), returns shape (80, 112, 96), and `SHAPE` / `SPACING` / `LEVEL` equal `P.LEVEL_SHAPE[2]`, `P.LEVEL_SPACING[2]`, 2; `_set_level(4)` restores them (tearDown). The module's DALI dependency is already optional (`data_loading/dali_loader.py` wraps `import nvidia.dali...` in try/except, `DALI_AVAILABLE`; `main()` imports `nvidia.dali` in a try block), so importing it in `tf_gpu` without DALI works; if the import fails anyway, the test is skipped with that reason.
16b. `TestCascadeBeatsFrozen` (in `tests/test_cascade.py`, GPU-tolerant, ~1–2 min at TINY widths): one synthetic level-2 group (`tests.runtime_helpers.synthetic_group(2, 4, "rigid")` and `fd.synthetic_deform_group(2, 2)`); a TINY level-4 model is trained for 100 steps (rigid, then 100 deform steps) on `cascade.pool_to_coarser` of exactly that group's images (same T, same labels pooled by mode, i.e. what the frozen stage sees in the cascade) and wrapped with `FrozenStage.from_model`. The level-4 tuples are built explicitly (DALI-format matrices are level-dependent, and `synthetic_group(4, ...)` would render a different phantom): rigid source `fd.to_dali_batch(cascade.pool_to_coarser(g2["images"]).numpy(), g2["T"], g2["T_target"], "rigid", 4)` with `g2 = synthetic_group(2, 4, "rigid")`; deform source `fd.to_deform_batch(pooled_images, pooled_labels, level=4)` with `d2 = fd.synthetic_deform_group(2, 2)`, `pooled_images = cascade.pool_to_coarser(d2["images"]).numpy()` and `pooled_labels = np.stack([preprocessing.export.block_mode(d2["labels"][i, ..., 0], 2) for i in range(2)])[..., None]` (numpy, smallest-id tie-break, as the export); then a level-2 cascade trainer (copy of the same weights) trains 100 steps per task on the level-2 group. Assertions on that group, train-mode statistics: rigid composed center RMS < frozen center RMS (the frozen baseline from `_frozen_baseline_metrics`) and deform composed template hard Dice > frozen template hard Dice; the measured margins are printed and recorded in the report. This replaces the trivial unrelated-phantom evidence of the e2e plan for AC3.5 / AC4.5 locally (the cluster `cascade_gain` lines remain the real-data evidence).
17. Regression: the goldens (`TestLinearGolden`, `TestDeformGolden`) max |diff| 0; the full suite (327 + new); the preprocessing suite (pytest, `bgr_preproc38`); Python 3.8 syntax (AC13.4): in `bgr_preproc38`, `python -m py_compile` of every new or modified module of section 17 (`models/cascade.py`, `runtime/args.py`, `runtime/run.py`, `runtime/checkpoint.py`, `main.py`, `tests/gpu/check_mixed_run.py`, `tests/gpu/verify_dali_pipeline.py`, `tests/gpu/measure_level_memory.py`, `tests/runtime_helpers.py`, every new/modified test file, `preprocessing/export.py`, `preprocessing/cli.py`, `preprocessing/tests/test_export.py`) exits 0.

Local end-to-end plan (WSL `tf_gpu`, GTX 1080, `--gpu-memory-limit-mb 6000`, `TF_CUDNN_USE_AUTOTUNE=0`), tiny widths passed as flags (`--filters 8,16,16,16,16,16 --reductions 16,8 --dense 64,32 --decoder-widths 8,8,8,8,8`):
1. Level 4: `main.py --overfit-one-group --overfit-synthetic --level 4 --task-ratio rigid:1,affine:1,deform:2 --group-size 4 --deform-group-size 2 --overfit-max-steps 150 --lambda-ncc-template 1 --lambda-ncc-pair 1 <tiny> --run-name e2e_l4`.
2. Level 2 cascade: `main.py --overfit-one-group --overfit-synthetic --level 2 --cascade-from results/e2e_l4/ckpt/last --task-ratio rigid:1,affine:1,deform:2 --group-size 4 --deform-group-size 2 --bn-recal-batches 8 --overfit-max-steps 150 --lambda-ncc-template 1 --lambda-ncc-pair 1 --task-grad-norm 12 <tiny> --run-name e2e_l2`, then `check_mixed_run.py results/e2e_l2 --tasks rigid,affine,deform --steps 150 --expect-best false --expect-init-weights full --expect-enc-grad true --expect-cascade true --expect-bn-recal true --grad-balance-rule true`. Expected: finite, one trace per task, composed train-mode errors below the frozen baselines on the overfit group (AC3.5 / AC4.5), folding < 0.1 %, `S_rel_diff_vs_level4` = null. Caveat: the level-4 overfit phantom (`smooth_phantom` at level 4) is unrelated to the level-2 phantom, so "composed < frozen" in this e2e run is weak evidence (it mostly shows that the residual learns). The non-trivial local evidence is test 16b (`TestCascadeBeatsFrozen`: frozen stage trained on the pooled SAME group); the real-data AC3.5 / AC4.5 evidence is the cluster smoke / long run's `cascade_gain` lines on identical validation groups.
3. Resume: rerun step 2 with `--overfit-max-steps 160 --resume true` on the same run name → fingerprint check passes, BN freeze re-applied from the saved record, config.json `init_weights` / `cascade` / `bn_recalibration` unchanged (diff of the three blocks), and the checker command of step 2 (with `--steps 160`) still passes.
Level-2 real data and default widths cannot run locally (GTX 1080, 6000 MB cap); those are the cluster jobs (a)–(c).

## 17. Files

New: `models/cascade.py`, `tests/gpu/measure_level_memory.py`, `slurm/m8_memory.sh`, `slurm/m8_level2_cascade.sh`, `tests/test_cascade.py`, `tests/test_slurm_scripts.py`, `tests/test_measure_memory.py`, `tests/test_verify_dali_level.py`, `.agents/tasks/milestone-8/report.md`.
Modified: `runtime/args.py` (5 flags incl. `--val-pair-chunk`, validation incl. the cascade-only recalibration rule, `train_config_from_args` → `cascade_skip_prob`, `val_pair_chunk`), `runtime/run.py` (`TrainConfig.cascade_skip_prob`, `TrainConfig.val_pair_chunk` (default 16, passed to `_deform_errors`); `Trainer(cascade=None)`; cascade branches of `_make_train_step`, `_make_deform_step`, `_val_fn`, `_val_deform_fn`; `train_step(task, batch, skip=False)`; skip draw in `fit`; `_frozen_baseline_metrics`; `recalibrate_bn`; per-task timing; `composed_mean_disp_mm`), `runtime/checkpoint.py` (`load_cascade_source`, `check_cascade_config`, `weight_fingerprint`, cross-level branch of `warm_start_model`), `main.py` (cascade wiring, frozen in `RunCheckpoint` extra, fingerprint check, recal / re-freeze, config.json keys, S diff), `tests/gpu/check_mixed_run.py`, `tests/gpu/verify_dali_pipeline.py` (`--level`, `_set_level`, `_template_brain` replacing both `template_mask_4.npy` literals, docstring), `tests/runtime_helpers.py` (`TINY`), `tests/test_runtime_infra.py`, `tests/test_check_mixed_run.py`, `preprocessing/export.py` (`LABEL_SET_LEVELS`, synthseg at levels 1, 2), `preprocessing/cli.py` (`--label-sets` help text), `preprocessing/tests/test_export.py`, `.agents/tasks/network-design/design.md` (v3.11, done in the design step). `models/group_net.py`, `models/diffeo.py`, `runtime/losses.py`, the DALI pipelines and `data_loading/params.py` need no change.

Constraints: Python 3.8 (no PEP 585/604 annotations evaluated at runtime, no 3.9+ stdlib such as `str.removeprefix`), TF 2.10 APIs only (`tf.nn.avg_pool3d`, `tf.keras.mixed_precision.global_policy/set_global_policy`, `tf.train.load_variable`), numpy 1.21, scipy only in tests (`scipy.linalg.logm` exists in 1.4). TF32 off in every new process (`configure_tf_precision()` first in `measure_level_memory.py`).

## 18. Responses to the design reviews

### v4 → v5 (review of v4: 0 HIGH, 1 MEDIUM, 6 NIT)

| # | Sev. | Response |
|---|---|---|
| 1 | MEDIUM | Addressed with option (a), the orchestrator's decision. `--bn-recal-batches K > 0` is only valid together with `--cascade-from`; otherwise `parser.error("--bn-recal-batches > 0 needs --cascade-from (BN recalibration is defined for the level-transfer cascade only)")`. This is in Q9, section 7 (scope paragraph and flags), the section 10 table, the section 15 error table and test 11 (level 4 and level 2 without cascade → error; K = 0 parses). `recalibrate_bn` also raises `RuntimeError` on a `cascade=None` trainer (test 7). Fresh vs resume semantics are now stated in one place (section 7, "When it runs"). Recalibration runs once, at step 0 of a fresh run, before any training step. On resume, the statistics come from the checkpoint (BN non-trainable variables) and recalibration is not repeated; only the freeze flag is re-applied from the saved `bn_recalibration.freeze` (section 9 step 5). A run preempted before its first checkpoint restarts as fresh. Test 8 now also asserts one `bn_recalibration` event and bitwise-restored statistics on a resume that passes K > 0. The generic (non-cascade) resume branch needs no re-freeze, because non-cascade runs can't recalibrate. |
| 2 | NIT | Addressed (section 6, test 5). `train_step(task, batch_tuple, skip=False)` passes `tf.constant(bool(skip))` only when `self.cascade is not None`. The non-cascade call `self._steps[task](*batch_tuple)` is unchanged (goldens). `fit` draws `skip` only with a cascade and p > 0. |
| 3 | NIT | Addressed (section 7 step 3, section 15, test 7). Each plan slot draws exactly one batch. A non-finite batch consumes its slot without an update or a replacement draw, increments `skipped_nonfinite[task]` (in the event), and logs a warning. A task with 0 used batches → `RuntimeError`. |
| 4 | NIT | Addressed (section 7, sections 13 and 16). `val=True` (main.py) runs the pre/post validations. `val=False` (unit tests only) skips them and leaves the gap table `{}`, and `--expect-bn-recal` accepts an empty table. |
| 5 | NIT | Addressed (Q15, section 13, test 12). The cut applies to the log record's `step` field (window end) ≥ `--grad-balance-from-step` (default 50); with `--log-every 25` the first used window is steps 26–50. The help text states this, and test 12 checks that step 25 is excluded and step 50 included. |
| 6 | NIT | Addressed with both parts of the suggested fix (sections 11, 12, 15, test 15). `measure_level_memory.py --val-pass true` runs one `_val_deform_fn` + `_deform_errors(chunk=16)` pass per deform cascade config and records `val_peak_mb` / `val_status`. The m8 memory job passes it for the 30 deform cascade configs (+~15 min, `--time` unchanged). The documented fallback is the new `--val-pair-chunk` (default 16 = today's literal, so default runs are unchanged). The smoke run's step-0 `bnrecal_pre` validation confirms on real data. |
| 7 | NIT | Addressed (section 8). The non-cascade `_val_fn` is unchanged (tensor). The cascade `_val_fn` returns `{"A", "A_prev"}` (eval) or `{"A"}` (train mode). `_val_errors` reads `out["A"]` in the cascade branch. `_val_deform_fn`'s cascade eval branch adds `U, U_inv, V`, and `_val_deform_errors` collects them into `frozen_outs`. |
| Unverified (momentum per eager call, `recompute_grad` without tape, cluster memory/time, two-level inverse tolerance, verify-DALI tolerances at 2 mm, `_needs_write` per item) | — | Unchanged. These remain mandatory test checks or provisional values. The DALI tolerances and the re-export behaviour will show in job (a) and Jay's export run, and the report records any adjustment. |

### v3 → v4 (review of v3: 0 HIGH, 1 MEDIUM, 8 NIT)

| # | Sev. | Response |
|---|---|---|
| 1 | MEDIUM | Addressed (sections 2.2, 2.3, tests 1(a), 10). Confirmed in code: `main._init_weights_lines` reads `info["ignored_source_parts"]` (main.py line 398); `warm_start_model` initialises it to `[]` and appends `deform_decoder` in the partial branch; partial `n_restored_vars = len(pairs)`. One schema is pinned in 2.2: base = today's nine keys, cross-level adds `source_level`, `source_step` (`tf.train.load_variable(prefix, C.STEP_KEY)`), `cross_level`; the cascade adds `via`. All records carry `ignored_source_parts`. Tests call `_init_weights_lines` on each record. |
| 2 | NIT | Addressed (section 3, test 3). `compose_linear` casts `R_raw` to `A_prev.dtype`; the float64 case uses `tfm.GroupCentering(iters=cfg.centering_iters, dtype="float64")`. |
| 3 | NIT | Addressed (section 9, "Fresh run"). Write config.json as today, log params, recalibrate, then rewrite with `bn_recalibration`, all before `fit` (the first `save_last`). Order confirmed against main.py lines 247–279. |
| 4 | NIT | Addressed (test 16b). Level-4 tuples come from `fd.to_dali_batch(pool_to_coarser(g2["images"]).numpy(), g2["T"], g2["T_target"], "rigid", 4)` and `fd.to_deform_batch(pooled images, block_mode labels, level=4)`. `block_mode` is called per image on the 3D array, which is its signature. |
| 5 | NIT | Addressed (section 12). The header always documents `-p dgx-a100`, the measured peak and an explicit fits / does NOT fit verdict. |
| 6 | NIT | Addressed (section 13). INFO `g_gap` lines and the report table use the existing `M.summarize` keys (`{center,pair}_{rms_mm,rot_deg,trans_mm}`, affine `{scale_pct,shear_pct,qr_rot_deg}`) for composed and `_frozen`. Keys confirmed in `runtime/metrics.py`. |
| 7 | NIT | Addressed (section 14). `label_luts.json` (`map` + `names`, from `preprocessing/luts.py`) is the label table. The core flag is derived from Jay's raw-id list. The "Jay must export / provide" list is spelled out. |
| 8 | NIT | Addressed (test 17). `bgr_preproc38` `py_compile` of every new or modified module. |
| 9 | NIT | Addressed (section 12, error table, test 13). The guard is skipped when `$RESULTS/$RUN_NAME/ckpt/last/checkpoint` exists, and a fourth stub case covers it. |
| Unverified (momentum per eager call, `recompute_grad` without tape, cluster memory/time, two-level inverse tolerance) | — | Unchanged. These are mandatory test checks or provisional values, as in v3. |

### v2 → v3 (review of v2: 0 HIGH, 2 MEDIUM, 7 NIT)

| # | Sev. | Response |
|---|---|---|
| 1 | MEDIUM | Addressed (section 8, files in 17, test 16a). Confirmed in code: `main()` loads `"template_mask_4.npy"` literally at lines 963 and 989. Both become `_template_brain(args)` = `template_mask%s.npy % P.level_suffix(LEVEL)`, called after `_set_level(args.level)`. All other level uses read the module globals at call time (checked: no default-argument capture; `check_l`, `check_e`, `manifest_files`, `brain_S_matrix` use `LEVEL`). Docstring updated. Test 16a scans for leftover `template_mask_4` / `labels_train18_4` literals and checks the level-2 mask shape. |
| 2 | MEDIUM | Addressed (section 12, test 13). `SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"`, `TMPDIR="$SCRATCH_ROOT/$JOB_ID"` in both m8 scripts. The stub test sets `PROJ/REPO/RESULTS/DATA/SIF/SCRATCH_ROOT/CASCADE_FROM` to temp paths, stubs `singularity` (plus optional `nvidia-smi`/`git`), and removes only the *recipe* env. Exact assertions are listed, along with the `N`-unset and `deform64` cases and a stub run of `m8_memory.sh`. |
| 3 | NIT | Addressed (section 8). The cascade val functions also return `A_prev` (linear) and `U, U_inv, V` (deform). `_frozen_baseline_metrics(..., frozen_outs)` uses them on a cache miss, with no extra function or forward. |
| 4 | NIT | Addressed (section 7 step 3): `self.controls[t].update(b)` only if `t in self.controls`. |
| 5 | NIT | Addressed (section 8). The rule is now `args.val_group_size or min(N, 39)`. The m8 scripts don't pass `--val-group-size`, and test 13 asserts that. |
| 6 | NIT | Addressed (section 12). T_smoke = 1.5 × (200 s_step + 4 s_val + s_recal + 600 s), floor 01:30:00, provisional 02:00:00. Each input's source is named (job (a) JSON, then the smoke-measured values in the report). |
| 7 | NIT | Addressed (section 12 header). Resubmission needs Jay's approval: partition, GPU, CPUs, memory and time are named, and the recipe is pre-agreed in message 12. The `squeue -u jn85` < 2 active-job check runs after the `scancel`. |
| 8 | NIT | Addressed with new test 16b `TestCascadeBeatsFrozen`. The frozen stage is trained on `pool_to_coarser` of the same group, and the test asserts composed center RMS < frozen and composed template Dice > frozen. The e2e caveat now points to it. |
| 9 | NIT | Addressed (section 2.3). The full and partial cross-level records are listed key by key (`mode`, `restored_parts`, `fresh_parts`, `source_level`, `cross_level`, no `via`). Test 10 asserts them. |
| Unverified: BN momentum per eager call, `recompute_grad` without tape | — | Unchanged. Test 7 makes them mandatory checks. |
| Unverified: memory / step times / `--time` | — | Unchanged. Provisional until job (a) and the smoke run report; formulas are given. |
| Unverified: two-level inverse tolerance | — | Unchanged. Test 4 prints the measured values, and the report may revise the 0.05 / 0.01 voxel bounds with them (up(u_inv4) is only an approximate inverse of up(u4)). |

### v1 → v2

| # | Sev. | Response |
|---|---|---|
| 1 | MEDIUM | Addressed. Section 9 gives the 6-step resume order (saved `cascade` required, no source re-read, `FrozenStage.from_config` + `restore_last`, fingerprint, re-freeze from the SAVED `bn_recalibration.freeze`, the three provenance blocks carried over verbatim into `write_config`). Section 10 table: source checks only on fresh runs. Test 8 asserts the blocks are unchanged after a resume with a moved source; e2e step 3 diffs them. |
| 2 | MEDIUM | Addressed. Section 12 gives the full variable block: every default assigned once, integer validation via `case`, plain variables afterwards, `RECAL_CHECK` built in the block. Test 13 greps for late `${VAR:-` uses and runs the script with a stub `singularity` to prove it reaches the checker. |
| 3 | MEDIUM | Addressed. Section 14 now uses the existing `synthseg` label set (`labels_synthseg[_2].npy`, `label_luts.json`, auto manifest columns), the code change `LABEL_SET_LEVELS` (synthseg at levels 1, 2) with pytest coverage, and Jay's exact `export_array.sh` commands (train, T1 val with `OPTIONAL_LABEL_SETS=lumir10`, merge). Core labels as raw SynthSeg ids mapped through `label_luts.json`, so no new table file. Network design section 11 addendum updated the same way. |
| 4 | MEDIUM | Addressed by the review's second option: "(B) applies unless Jay chooses (A); nothing in milestone 8 depends on it", written in Q16 and section 14; the report must flag it as open. Not paused for Jay here because no milestone-8 code or run depends on the answer, and the exporter change (useful for either choice) is done anyway. |
| 5 | MEDIUM | Addressed. Frozen stage recomputed inside the validation `tf.function`; only baseline metrics cached by (name, len(groups)). The caching sentence is removed from Q6 (design.md 4.7 never had it; it now states the recompute explicitly). |
| 6 | MEDIUM | Addressed. `--num-threads "${SLURM_CPUS_PER_TASK:-3}"` and `--data "$DATA"` in the DALI call; the four `measure_level_memory.py` invocations (49 configs) are spelled out with the time estimate. |
| 7 | NIT | Addressed: parser rule `--cascade-from` requires `--deform-residual-affine false` (section 5, table 10, test 16). |
| 8 | NIT | Addressed: construction and build inside `_float32_policy()` for `from_checkpoint`, `from_model`, `from_config`; test 9 asserts `compute_dtype == "float32"`. |
| 9 | NIT | Addressed: per-layer saved momentum restored in `finally`; test 7 checks it with a layer of different momentum. |
| 10 | NIT | Addressed: test 6 compares p = 0 vs p = 0.5 forced `skip=False` and asserts no `cascade_skip*` ops in the p = 0 graph. |
| 11 | NIT | Addressed: "ratio of the medians" (Q15, section 13); UNDECIDED → treat as FALLBACK and inspect; FALLBACK while the gradnorm long run runs → `scancel` our own job, resubmit with `GRAD_BALANCE=deform64` (section 12 header). |
| 12 | NIT | Addressed: `S_rel_diff_vs_level4` = null in overfit mode, computed from `template_mask_4.npy` only in `_build_dali_data` (section 2.1). |
| 13 | NIT | Addressed: guard accepts a directory or `<prefix>.index` (section 12). |
| Unverified: momentum per eager call | — | Test 7's exact-average check is mandatory (section 16). |
| Unverified: `recompute_grad` without a tape | — | Test 7 adds `gradient_checkpointing='both'` with update count = number of batches. |
| Unverified: direct cross-level restore | — | Not relied on; restore-then-copy stays (valid either way). |
| Unverified: memory / time estimates | — | Explicitly provisional; replaced by job (a) and the smoke run. |
| Unverified: e2e AC3.5 / AC4.5 | — | Caveat added to the local e2e plan; the evidence is the cluster `cascade_gain` lines. |
