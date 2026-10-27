# Retrain-from-scratch prerequisites: design

Status: v1.4 (2026-10-27, implementation notes for design review v1.3 = CHANGES_REQUESTED, 0 HIGH / 1 MEDIUM / 6 NIT, all resolved in the code and scripts; see section 15 v1.4; where sections 6-7 differ, the v1.4 notes and section 16 win). v1.3 (2026-10-26, revision after design review v1.2 = CHANGES_REQUESTED, 0 HIGH / 1 MEDIUM / 5 NIT: the MEDIUM and all five NITs resolved; see section 15) + section 16 addendum (Jay's confirmations of the wd grid and tent, new opt-in `--max-pairs`, `--pair-recompute` restricted to level 1). v1.2 answered review v1.1 (5 MEDIUM / 8 NIT); v1.1 was the initial self-check of v1. Workflow `wf_63bfb7f7f2cfe710`. Parent design: `.agents/tasks/network-design/design.md` (v3.15, bumped to v3.16 by this work, section 9 below).

Hard constraints for every step of this workflow: work directly in `d:\Research\Brain_Registration_Projects\brain-group-rad` (no git worktree); do NOT commit, push, ssh or sbatch (two cluster jobs are running; the orchestrator handles git and the cluster); all docs of this effort go under `.agents/tasks/retrain-prereqs/`. Cluster code must run on Python 3.8.10 / TF 2.10.0 / Keras 2.10.0 / numpy 1.21.1 (`tf10_nv.sif`); login-node helpers are stdlib-only.

## 0. Overview

Jay retrains the whole cascade (level 4 -> 2 -> 1) from scratch with one recipe: AMP at every level, the largest group size N per task and level, TF-side intensity augmentation with a shift-free low-res op, Jay's per-task / per-stage skip schedule, per-task gradient normalisation (`--task-grad-norm 12`), AdamW, and a small level-4 sweep over learning rate, weight decay and N. This document specifies the code and script prerequisites:

1. AdamW (`--optimizer adamw`), already wired; tests, a resume guard and the decision record are added (section 2).
2. A shift-free low-res method `--lowres-method nn|tent` on the TF augmentation path (section 3).
3. Per-task, per-stage cascade skip flags `--cascade-skip-linear` / `--cascade-skip-deform` with Jay's semantics (section 4).
4. A delayed task start `--task-start-step` / `--task-start-ramp-steps`, so level 4 trains from scratch in ONE run (section 5).
5. New SLURM scripts `slurm/r_level4.sh`, `slurm/r_level2.sh`, `slurm/r_level1.sh` (section 6), a sweep submit helper `slurm/sweep_level4.py` with phases A (lr x wd), B (group sizes) and C (number of iterations) (section 7) and a run summarizer `scripts/summarize_runs.py` with the ranking and the length decision (section 8).

Every new flag defaults to today's behaviour: the goldens `tests/fixtures/m7_linear_golden.json` and `tests/fixtures/lncc_deform_golden.json` stay at max |diff| 0, and the existing cascade skip identity test (`tests/test_cascade.py::TestSkip`) stays bit-exact.

Technology stack (locked): Python 3.8, TF 2.10 / Keras 2.10 (`tf.keras.optimizers.experimental`), numpy 1.21, argparse, unittest; bash for SLURM scripts; stdlib-only Python (3.6-compatible, see 7.4) for the login-node helpers. No new dependencies.

## 1. Facts verified in the code (2026-10-26)

| Fact | Where |
|---|---|
| `--optimizer` choices `sgdw, sgd, adam, adamw`, default `adam`; `--weight-decay` default 3e-5; `--decoupled` only for SGDW; `--exclude-from-weight-decay` default true | `runtime/args.py` (optimizer group) |
| `make_optimizer("adam")` builds `exp.Adam(learning_rate=lr, global_clipnorm, jit_compile)` and never reads `weight_decay`. **So today `--weight-decay 3e-5` has NO effect with Adam** (every Adam run so far had zero weight decay). It acts only on `sgdw` (coupled, or decoupled with `--decoupled true`) and `adamw`. | `runtime/optimizers.py::make_optimizer` |
| `make_optimizer("adamw")` already builds `tf.keras.optimizers.experimental.AdamW(learning_rate=lr, weight_decay=wd, global_clipnorm, jit_compile)` and calls `exclude_from_weight_decay(var_names=["gamma", "beta", "bias"])`; `amp` wraps it in `LossScaleOptimizer` | same |
| Keras 2.10.0 `optimizer_experimental/adamw.py` exists (checked in the local `tf_gpu` env, keras 2.10.0). `update_step`: if `_use_weight_decay(var)`: `var -= var * wd * lr` (lr = the optimizer's current `learning_rate`), THEN the Adam update, identical op for op to `adam.py::update_step`. `_use_weight_decay` tests `variable in self._exclude_from_weight_decay` (elementwise compare: never pass variables, only names) and `re.search(name, variable.name)` | Keras 2.10 source |
| The Trainer sets the scheduled LR every step (`O.set_learning_rate(opt, lr_at(s, schedule))`), so the decay uses the scheduled lr_t; only variables with a non-None gradient in that step are passed to `apply_gradients` (static filter), so a variable is decayed only on steps of tasks that use it | `runtime/run.py::fit`, step bodies |
| Under AMP the step calls `opt.get_unscaled_gradients` and `opt.apply_gradients` on the LSO; non-finite gradients are skipped by the LSO's own check (the Trainer's `tf.cond(in_finite if amp else finite, apply, skip)` gates only on input finiteness under AMP, `runtime/run.py` ~l.891 / 1005; in fp32 it is the gradient guard); test 11.1 d; the decay is skipped with them (it lives in `update_step`) | `runtime/run.py` |
| Low-res TF op: per axis NN source `src(l)` + linear up `c = (o + 0.5) m / n - 0.5`, edge clamp; `LOWRES_ROI = "subpixel"`; config `IntensityAugConfig.low_res_prob 0.125`, `low_res_min_factor 0.5` | `data_loading/tf_intensity_aug.py`, `data_loading/params.py` |
| Skip today: one `--cascade-skip-prob` in [0, 1); at level 2 skips the frozen level-4 stage (`skip_linear` / `skip_fields`: A_prev = I, U = 0), at level 1 the LAST frozen stage (`skip_to`: A_prev = A4, U = U4); draw `random.Random("cascade-skip-<seed>-<step>").random() < p`; one scalar bool step input; ops under name scope `cascade_skip` only when p > 0; validation never skips | `models/cascade.py`, `runtime/run.py::cascade_skip_draw`, `fit`, step bodies |
| Task draw: `TaskSchedule.draw(step)` = `random.Random(seed * 1000003 + step).random()` against the cumulative ratio (single task: no draw) | `runtime/run.py::TaskSchedule` |
| metrics.jsonl record = `{"step", "time", "data": {...}}`, flushed per record; validation keys `val_<set>/center_rms_mm`, `/pair_rms_mm`, `val/mean_pair_rms_mm`, `val_deform/{template_dice, pair_dice, neg_jac_pct, inv_consistency_mm, ncc_template, ncc_pair}`, `gpu_peak_mb`, `nonfinite_total`, `train/collapse_suspect` | `runtime/logging.py`, `tests/gpu/check_mixed_run.py` |

## 2. AdamW

### 2.1 Decision

Resolved decision 1 (effect of AdamW on the current code), with rationale: switching costs no model-code change and does not alter any past result. `make_optimizer("adam")` never passed `weight_decay`, so every m4-m9 run trained with zero weight decay despite `--weight-decay 3e-5` in its config.json; the switch to `adamw` is therefore the first time weight decay acts on this model, and AdamW at wd 0 is op-for-op the Adam all previous runs used (test 11.1 a, max |diff| 0). Consequences: (i) the sweep's wd 0 cells are the existing recipe, a true baseline; (ii) the `--optimizer` default stays `adam` and `--weight-decay` keeps its 3e-5 default, so goldens and the m7 / m8 / m9 scripts are untouched; (iii) the misleading no-op is made visible (startup NOTE, help text) rather than changed, because changing `adam` to honour `--weight-decay` would silently alter every old script's behaviour on re-run.

Use the existing `--optimizer adamw` = Keras 2.10 `experimental.AdamW` (no custom optimizer class). Its step is the decoupled decay of Loshchilov & Hutter in the PyTorch / Keras convention: `w <- w - lr_t * wd * w`, then the unchanged Adam update, with lr_t = the schedule's learning rate of that step. Rejected: a custom AdamW subclass (duplicates Keras code for no behavioural gain) and decay scaled by the schedule multiplier only (non-standard; makes wd values incomparable with the literature).

Consequence for the sweep values (binding for section 7): with lr-coupled decay the total shrink of a weight that gets no gradient is exp(-wd * sum_t lr_t). For an 8000-step run with `--schedule cosine --warmup-steps 1000 --min-lr-factor 1e-3` (the code defaults), sum_t lr_t ≈ (500 + 0.5 * 7000) * lr ≈ 0.5 * 8000 * lr = 1.2 at lr 3e-4 (0.4 at 1e-4, 4.0 at 1e-3). wd 1e-4 then shrinks by 1.2e-4 relative over the whole run, wd 1e-5 by 1.2e-5: both are indistinguishable from wd 0. The default sweep grid is therefore **wd {0, 1e-2, 1e-1}** (0.4-4 % and 4-33 % total pure-decay shrink (lr 1e-4 to 1e-3); PyTorch's AdamW default is 1e-2). The grid is a CLI argument of the sweep helper, so Jay's originally quoted {0, 1e-5, 1e-4} remains one flag away (`--wds 0,1e-5,1e-4`). This default is decided for the implementation; Jay's confirmation (section 14) only changes the `submit` command line, never the code.

### 2.2 Changes

- `runtime/optimizers.py`: no change to `make_optimizer`'s math. Docstring: state that `adam` ignores `weight_decay`, `adamw` = Keras 2.10 experimental AdamW (decay `lr_t * wd * w` before the Adam update, excluded names gamma/beta/bias: BatchNorm and LayerNorm affine parameters and every bias). Add a module constant `WEIGHT_DECAY_OPTIMIZERS = ("sgdw", "adamw")`.
- `runtime/args.py`: help text of `--weight-decay`: "used by sgdw and adamw only; adam ignores it". Parse check: `--weight-decay` finite and >= 0 (parser.error otherwise). No default change (3e-5 stays; the new scripts always pass `--weight-decay $WD`).
- `main.py`: at startup print one line `optimizer adamw | lr 3e-4 | weight decay 0.01 (decoupled, lr-scaled; excluded gamma/beta/bias: <k> of <n> variables)`, and for `--optimizer adam` with `--weight-decay` != 0 print `NOTE: --weight-decay is ignored by --optimizer adam`. Record in config.json `optimizer`: `{"name", "lr", "weight_decay", "decayed_variables": k, "excluded_variables": n - k}` (computed with `O.uses_weight_decay` after `build_optimizer`; new top-level key, written only when the optimizer is `sgdw` or `adamw`, so default (Adam) runs write the same config.json as today).
- Resume guard `main._check_optimizer(config_path, args)` (same pattern as `_check_intensity_aug_device`): on resume, if the saved `args.optimizer` differs from `--optimizer`, `SystemExit("main.py: --optimizer X differs from the run's saved Y ...; resume with the saved optimizer or start a new --run-name")`. Reason: slot structures differ (SGDW momentum vs Adam m/v) and a mismatch would restore partially or fail late. A different `--weight-decay` or `--lr` on resume is allowed (no slot change) and logged as an event `{"event": "optimizer_hparam_changed_on_resume", "old": {...}, "new": {...}}`; the config.json `optimizer` record is then REFRESHED from the current flags (not carried verbatim), so it always describes the optimizer of the latest segment.
- Checkpointing: the AdamW slots (m, v per variable) and `iterations` are tracked by the existing `RunCheckpoint` exactly like Adam's (same base class); nothing to add. Test 11.1 (e) verifies the round trip.

### 2.3 Interactions

- AMP: unchanged LSO wrapping; a non-finite step is skipped by the LSO's own check (fp32: by the Trainer's finite cond; under AMP the Trainer gates only on input finiteness; test 11.1 d), and the decay is skipped with it.
- Global clip 12 and `--task-grad-norm`: both act on the gradients before `update_step`; the decay is gradient-independent, so neither changes it.
- Variables decayed only on steps that update them: encoder every step, linear heads on their task's steps, deform decoder on deform steps. Documented as a property; no compensation.
- Frozen cascade stages are not in the optimizer and are never decayed.

## 3. Shift-free low-res (`--lowres-method nn|tent`)

### 3.1 Measured basis and decision

A 1-D numpy study (all factors f in 0.5..0.95 step 0.025, axis lengths 40/56/48 and 160/224/192, slabs of width 6-30 at ~20 positions; centroid displacement of the 0/1 slab after down + up):

| Method | max \|d\| (width 6 / 10 / 20 / 30) | mean \|d\| (width 20) |
|---|---|---|
| today: NN 'subpixel' down + linear up | 1.43 / 1.21 / 1.36 / 1.25 vox | 0.28 |
| box ("area") average down to m = round(f n) + centered linear up | 0.066 / 0.058 / 0.058 / 0.056 | 0.015 |
| **tent down (adjoint of the linear up, row-normalised) + the same linear up** (interior slabs; edge-touching slabs: up to 0.0625 at width 12, see below) | 0.032 / 0.021 / 0.0096 / 0.0070 (width 12: 0.018, 16: 0.013) | 0.0014 |

Box averaging + linear upsampling is not shift-free enough for the < 0.02 voxel acceptance: its sampled-box first moment differs from the cell center by up to ~0.06 voxel for non-integer n/m. **Decision: method `tent`.** It keeps the same footprint as the linear upsample (each low-res cell l averages the input voxels with the tent weights U[o, l] that the upsample later spreads it back with), so it is a proper anti-aliased (triangle-filtered) downsample of the kind Jay asked for. Properties: exact mass preservation (sum_o out = sum_o x, since U rows sum to 1), constants reproduced exactly up to float32 rounding, the image box stays put (both stages map the extent [0, n) onto itself), centroid displacement of interior objects (>= 2 voxels from both edges) <= 0.0181 voxel at width 12, 0.0119 at 16, 0.0076 at 20 (worst case over f 0.5..0.95 and axis lengths 40/56/48/160/224/192, reproduced by the review). Objects TOUCHING the volume border shift by up to 0.0625 voxel (width 12; 0.0375 at width 20) because the edge clamp of l0c / l1c makes the boundary rows asymmetric; practically irrelevant (skull-stripped brains are mostly background near the border; not measured here) and in any case an order of magnitude below today's NN bias (up to ~1.4 voxel), so no extra edge handling is added. The paper describes it once: "low resolution simulated by a triangle-filtered downsample to round(f n) voxels per axis and linear upsampling with the transposed weights' grid (pixel centers at 0.5), which preserves mass and position".

DALI keeps `nn` (its own op cannot do this); `tent` is TF-device only.

### 3.2 Math (per spatial axis, length n, factor f, static n)

- m = max(1, floor(f n + 0.5)) (float64, as today); o = 0..n-1; c(o) = (o + 0.5) m / n - 0.5; l0 = floor(c); w = c - l0; l0c = clip(l0, 0, m - 1); l1c = clip(l0 + 1, 0, m - 1) (identical to today's upsample).
- U [n, n] float64 (columns l >= m are zero): U = (1 - w)[:, None] * one_hot(l0c, n) + w[:, None] * one_hot(l1c, n).
- a = sum_o U[o, :] ([n]); D = transpose(U) * (1 / a)[:, None] with `tf.math.divide_no_nan` (rows l >= m are zero).
- P = U @ D ([n, n], float64), cast to float32; out = P applied along the axis: `tf.tensordot(P, x, axes=[[1], [axis]])` then `tf.transpose` back to the original axis order (helper `_apply_axis_matrix(x, P, axis)`).
- f = 1: m = n, U = I, P = I (exact).

Static shapes only (all matrices n x n, n is the static axis length); memory: one n x n matrix (<= 224^2 float64 = 0.4 MB) plus one transient volume per axis; no gather, deterministic (matmul, TF32 off as always).

### 3.3 Code changes

- `data_loading/params.py`: `LOWRES_METHODS = ("nn", "tent")`; `IntensityAugConfig.low_res_method: str = "nn"` (new dataclass field, last position). DALI never reads it.
- `data_loading/tf_intensity_aug.py`: `_lowres_axis_tent(x, axis, f)` and `lowres_resample(x, f, roi=None, method="nn")`: `method` not in `LOWRES_METHODS` -> `ValueError`; `nn` = today's code path verbatim (`roi` honoured); `tent` ignores `roi`. `augment_image` passes `getattr(cfg, "low_res_method", "nn")`. The draw pattern (`u` columns, seeds) is unchanged, so `nn` runs are bit-identical and `tent` runs draw the same factors.
- `data_loading/intensity_ref.py`: numpy reference `lowres_tent_np(img, f)` built with explicit loops (independent of the TF matrix code) for the equality test.
- `runtime/args.py`: `--lowres-method {nn,tent}` default `nn` in the augmentation group. Parse errors: `tent` with `intensity_aug_device(args) == "dali"` -> `parser.error("--lowres-method tent needs --intensity-aug-device tf (DALI only has nn)")`. With `--intensity-aug false` the flag is accepted and has no effect. `intensity_aug_tf(args)` returns `P.IntensityAugConfig(low_res_method=args.lowres_method)`.
- `main.py`: the existing "intensity aug tf" description line appends `| low-res tent`. No resume guard (the method changes no checkpoint structure).
- `tests/gpu/verify_dali_pipeline.py`: unchanged (check m compares the `nn` path with DALI).

## 4. Per-task, per-stage cascade skip

### 4.1 Semantics (Jay's decision, 2026-10-26; the only option)

Skipping a stage means that stage contributes an IDENTITY transform, and the later stages still run:

- skip level 4 -> A4 = I and u4 = u4_inv = 0; the frozen level-2 stage (at level 1) and the trainable stage run on the un-pre-warped inputs: A_prev = A2 = Center2(R2_raw), U = up(u2);
- skip level 2 alone -> level 4 only: A_prev = A4, U = up(u4) (today's level-1 skip);
- skip both -> identity for the whole frozen chain (A_prev = I, U = U_inv = 0);
- at level 2 (single frozen stage) skip level 4 -> A_prev = I, U = U_inv = 0 (today's level-2 skip).

Draws are independent per stage, per task and per step. Jay's probabilities: linear (rigid and affine) `4:0, 2:0.15`; deform `4:0.15, 2:0.15`. Validation never skips. This replaces the sentence "Skipping level 4 means A_prev = I, U = 0 for the whole frozen chain" in design.md 4.7 (section 9).

Resolved decision 2 (skip-level-4 semantics), with rationale. Rejected alternative: "skip level 4 = identity for the whole frozen chain" (the v3.15 sentence). Reasons for Jay's version: (i) a skip should simulate exactly one failed / absent upstream stage, so the trainable stage learns to absorb a larger residual from that stage while the other stages behave as at inference; dropping the whole chain conflates "level 4 failed" with "levels 4 and 2 failed", and the latter already has its own draw (skip both, probability p4 x p2); (ii) independent per-stage draws make the per-stage rates directly interpretable and loggable (`train_<task>/cascade_skip_l<lv>`); (iii) the schedule is self-consistent in the retrain: a frozen stage only sees an upstream skip it was itself trained with. Level 2 is trained with deform `4:0.15`, so at level 1 its frozen deform stage running on non-deform-pre-warped inputs is in-distribution; linear `4:0` at both levels means the frozen level-2 linear stage is never fed un-pre-warped ±45° inputs it never saw. Point (iii) is checked, not assumed: at `--level 1`, `main.py` takes the `cascade` record of the level-2 source run's config.json (that file is already loaded for the fingerprint check) and, for every task with stage-mode p[task][4] > 0, prints `WARNING: --cascade-skip-<linear|deform> skips level 4 for <task> but the level-2 source was trained with level-4 skip p = <p or 'none'> for it; the frozen level-2 stage will see out-of-distribution inputs` when the source's `cascade.skip_probs[task]["4"]` is missing or 0 (legacy `skip_prob` > 0 counts as trained for every task). A warning, not an error, so ablations stay possible; it is also written to config.json as `cascade.skip_consistency_warnings` (list of task names, empty if none).

"Un-pre-warped" is implemented as pre-warping with the identity (A4 = I through `prewarp_linear`, U4 = 0 through `prewarp_deform`), the convention of today's level-2 skip; tests bound the difference to the raw input (11.3).

### 4.2 Flags and validation (`runtime/args.py`)

- `--cascade-skip-linear SPEC` and `--cascade-skip-deform SPEC`, default None. SPEC = comma list `level:prob` (e.g. `4:0,2:0.15`); optional, string. Rules: each level an integer among the run's frozen stage levels (`--level 2`: {4}; `--level 1`: {4, 2}); prob a finite float in [0, 1); no duplicate level; unlisted levels = 0; empty string = all 0. Violations -> `parser.error` naming the flag and the bad entry. Parsed to `args.cascade_skip_stage_parsed = {"rigid": {lv: p}, "affine": {lv: p}, "deform": {lv: p}}` (rigid and affine both from the linear flag), only for the trained tasks.
- Any non-None spec (even all p = 0 or the empty string) without `--cascade-from` -> `parser.error("--cascade-skip-linear needs --cascade-from")` (resp. `-deform`). This check runs before the level check, so at `--level 4` (no frozen stages, no cascade) every spec is rejected with this one message.
- `--cascade-skip-prob > 0` together with any per-stage p > 0 -> `parser.error("--cascade-skip-prob and --cascade-skip-linear / --cascade-skip-deform are mutually exclusive")`. The legacy flag keeps working unchanged (its own draw string, its own code path).
- `TrainConfig.cascade_skip_stage: Dict[str, Dict[int, float]] = {}` (default empty = off). `Trainer.__init__` re-validates (levels must be in the cascade's frozen levels: `[4]` for `FrozenStage`, `[4, 2]` for `FrozenChain`; p in [0, 1); both modes on -> ValueError).

### 4.3 Trainer (`runtime/run.py`)

- Mode, fixed at construction: `self._skip_mode` = `"legacy"` if `cfg.cascade_skip_prob > 0`, `"stage"` if any p in `cfg.cascade_skip_stage` > 0, else `"off"`. Stage order `self._frozen_levels` = `[4]` (n_stages 1) or `[4, 2]` (n_stages 2).
- Draw (pure function): `cascade_stage_skip_draw(seed, step, task, level, p) = random.Random("cascade-skip-%s-l%d-%d-%d" % (task, level, seed, step)).random() < p` (False without a draw when p == 0). Legacy keeps `cascade_skip_draw` verbatim.
- Step signatures: legacy and off unchanged (scalar bool input for cascade runs). Stage mode: the cascade step functions take `skip` as `tf.TensorSpec([n_stages], tf.bool)` (index i = `self._frozen_levels[i]`). One trace per task regardless of the values (no retracing).
- Trace-time switches per task: `on4 = p[task][4] > 0`, `on2 = p[task].get(2, 0) > 0`; a stage with p == 0 builds no skip op (its skip tensor element is ignored).
- Level 2 (FrozenStage), stage mode: if `on4`: `A_prev = CS.skip_linear(skip[0], A_prev)` / `U, U_inv = CS.skip_fields(skip[0], [U, U_inv])`; same ops as legacy.
- Level 1 (FrozenChain), stage mode: call the new chain methods (4.4) with `skip4 = skip[0] if on4 else None`, `skip2 = skip[1] if on2 else None`. Legacy level-1 keeps today's `linear_stages` / `deform_stages` + `skip_to` branch untouched.
- `train_step(task, batch, skip=False)`: legacy / off as today; stage mode accepts a tuple / list of `n_stages` bools (a bare bool raises TypeError in stage mode).
- `fit`: stage mode draws `skips = tuple(cascade_stage_skip_draw(cfg.seed, s, task, lv, p[task].get(lv, 0)) for lv in frozen_levels)`; record `rec["cascade_skip_l<lv>"] = bool` for every lv with p > 0 for this task; window means `train_<task>/cascade_skip_l<lv>` (float) in the accumulator. Legacy key `train/cascade_skip_frac` unchanged and only in legacy mode.
- BN recalibration and validation never skip (no change: they call the cascade without skip inputs).

### 4.4 Cascade (`models/cascade.py`)

New `FrozenChain` methods (inference only, `stop_gradient` on outputs, same float32 rules):

```python
def linear_skip(self, images_l1, task, skip4=None, skip2=None):
    """A_prev [N,4,4] with per-stage skips (bool scalar tensors or None = not built)."""
    x2 = pool_to_coarser(images_l1)
    A4 = self.stage4.linear(x2, task)
    if skip4 is not None:
        A4 = skip_linear(skip4, A4)                       # A4 = I when skipped
    out = self.model2(prewarp_linear(x2, A4, P.LEVEL_SPACING[CHAIN_LEVEL]), task=task, training=False)
    A2 = compose_linear(out["A_raw"], A4, self.model2.centering)   # = Center2(R2_raw) when A4 = I
    A = A2 if skip2 is None else skip_to(skip2, A4, A2)   # skip both -> I
    return tf.stop_gradient(A)

def deform_skip(self, images_l1, skip4=None, skip2=None):
    """dict U, U_inv (level-1 voxels) with per-stage skips."""
    x2 = pool_to_coarser(images_l1)
    p4 = self.stage4.deform(x2)                           # U, U_inv at level 2
    U4, U4i = p4["U"], p4["U_inv"]
    if skip4 is not None:
        U4, U4i = skip_fields(skip4, [U4, U4i])           # u4 = u4_inv = 0 when skipped
    out = self.model2(prewarp_deform(x2, U4), task=DEFORM_TASK, training=False)
    u2, u2i = diffeo.compose_cascade(U4, U4i, out["u"], out["u_inv"])
    up = diffeo.upsample_displacement
    U, Ui = up(u2), up(u2i)
    if skip2 is not None:
        U = skip_to(skip2, up(U4), U)
        Ui = skip_to(skip2, up(U4i), Ui)
    return {"U": tf.stop_gradient(U), "U_inv": tf.stop_gradient(Ui)}
```

With `skip4=None`, the ops equal today's `linear_stages` / `deform_stages` + `skip_to` (test 11.3 c, max |diff| 0). All `tf.where` ops stay under the `cascade_skip` name scope.

### 4.5 Provenance and checker

- `main.py` cascade record: add `"skip_mode": "off"|"legacy"|"stage"` and, in stage mode, `"skip_probs": {"rigid": {"4": p, "2": p}, ...}` (string keys for JSON). `skip_prob` / `skip_stage` keys unchanged.
- Level-1 skip-consistency warning (4.1 iii): helper `main._skip_consistency(src_cascade_rec, skip_stage_parsed) -> list of task names` (pure function, unit tested); `src_cascade_rec` is the level-2 source's config.json `cascade` dict (None for a level-4 source, then nothing is checked). Missing keys in an old source config (m8 runs have only `skip_prob`) are treated as p = `skip_prob` for every task.
- Resume: the skip settings are not part of any checkpoint structure; a changed value on resume is allowed. On a cascade resume `main.py` today carries `saved_cfg["cascade"]` verbatim (~line 290); the new code refreshes `skip_mode`, `skip_probs` and `skip_consistency_warnings` in that record from the current flags (everything else stays carried), and when any of the three differs from the saved value writes an event `{"event": "cascade_skip_changed_on_resume", "old": {...}, "new": {...}}`. The checker therefore always reads the latest settings.
- `tests/gpu/check_mixed_run.py --expect-skip true`: if config.json `cascade.skip_mode == "stage"`, require every `train_<task>/cascade_skip_l<lv>` with p > 0 to be logged and finite at least once, and print `INFO skip <task> l<lv>: observed <mean> vs p <p>` (never a failure); else today's legacy check.

## 5. Delayed task start and the level-4 recommendation

### 5.1 Recommendation: ONE level-4 run

Recommended: one from-scratch level-4 run, three tasks, `--task-start-step deform:3000 --task-start-ramp-steps 500`, curriculum 15/5 -> 45/15 over 3000 steps and G 45 deg / 10 mm with `--global-start-step 3000 --global-ramp-steps 3000` (the m5 recipe for the linear tasks), 16000 steps total. Rationale:

- It reproduces the chain that worked (m5 linear warm-up, then m7 three tasks) without its artefacts: no fresh Adam at the switch (m7 smoke dipped at step 100 from the fresh optimizer), no second warm-up, one job, one cosine schedule.
- The known from-scratch risks are addressed: G from step 0 collapses the linear tasks (G starts at 3000, as m5); deform joins an encoder that is already non-collapsed (healthy `diag_feat_rel_spread`) instead of competing with the linear warm-up from step 0. The zero-init SVF head's ~100-step lag (seen in m7) happens whenever deform starts; the delay does not remove it and does not need to. Step 3000 is at once the curriculum end, the G start and the deform start; a dip in the linear val metrics there is expected and not a collapse signal (section 6). Not staggered: one change point keeps the scaled sweep schedule (2000) and the final one (3000) simple; if the smoke or sweep shows a dip that does not recover within ~1000 steps, staggering (DEFORM_START = G_START + 1000) is an env-only change.
- The deform decoder's Adam slots are zero at step 3000 (built at step 0 by `build_optimizer`, never updated) while `iterations` is already 3000, so the bias corrections are ≈ 1 and, for a steady gradient g, the k-th update is lr x (1 - 0.9^k) / sqrt(1 - 0.999^k) x sign(g): 3.2x the steady size at k = 1, peak ~6.5x at k ≈ 10-20, 3.3x at k = 100, 1.6x at k = 500. `--task-start-ramp-steps 500` linearly ramps the learning rate of the delayed task's steps from 0 over global steps [3000, 3500) (~250 deform updates at ratio 1:1:2), which keeps the product (ramp k/250 x transient) at <= 1.3x steady for k <= 100 and peaks at ~2.1x at the end of the ramp (k ≈ 250), decaying to 1.6x by k = 500, instead of the 6.5x spike without a ramp (global clip 12 / `--task-grad-norm` act on gradients, not on this slot effect); the shared encoder update on those steps is ramped too, a gentle introduction of the new task. AdamW's decay `lr_t x wd` on those steps is ramped with it (same lr variable), intended.

Resolved decision 3 (one run vs two-stage level 4), rationale in one line: one run is chosen because it removes the optimizer reset and second warm-up that the two-stage chain imposes, keeps one cosine schedule for the sweep to tune (a two-stage chain would need lr / wd / steps per stage, doubling the sweep), and costs only two small flags; the two-stage chain stays available unchanged as the fallback below.
- Fallback without model-code changes (r_level4.sh supports it: deform-only flags are gated on TASK_RATIO, section 6): the two-stage chain (r_level4.sh with `TASK_RATIO=rigid:1,affine:1 DEFORM_DECODER=true`, then `TASK_RATIO=rigid:1,affine:1,deform:2 INIT=<run>/ckpt/best DEFORM_START=0 CURRICULUM=false G_START=0 G_RAMP=0`) if the sweep shows collapse (`train/collapse_suspect`) or deform stuck at identity (final `val_deform/pair_dice` < 0.562, i.e. within 0.01 of the level-4 identity pair Dice 0.552 measured in m7). With `DEFORM_DECODER=true` the stage-1 checkpoint already holds an (untrained, zero-head) deform decoder, so stage 2 is a `full` warm start (checker `--expect-init-weights full`); an untrained decoder gets no gradient and hence, with AdamW, no decay in stage 1.

### 5.2 Flags (`runtime/args.py`)

- `--task-start-step SPEC`, default None: comma list `task:step`, task in the ratio, step integer >= 0; tasks not listed start at 0. At least one task must start at 0; each start must be < `--steps` (else parser.error: "would never train"). Duplicate task -> error.
- `--task-start-ramp-steps W`, default 0 (int >= 0): LR factor `min(1, (s - start + 1) / W)` on that task's steps s in [start, start + W); 0 = no ramp. Tasks starting at 0 never get it (the global warm-up covers them). W > 0 without `--task-start-step` (or with every listed start 0) -> `parser.error("--task-start-ramp-steps needs a --task-start-step entry > 0")`; W negative -> parser.error.
- The task names in the spec must be in `--task-ratio` (unknown task -> parser.error naming it). Parsed to `args.task_start_steps_parsed: Dict[str, int]` (only listed tasks).
- `TrainConfig.task_start_steps: Dict[str, int] = {}`, `task_start_ramp_steps: int = 0`.

### 5.3 Behaviour (`runtime/run.py`)

- `TaskSchedule(ratio, seed=0, start_steps=None)`: `draw(step)` computes `u` exactly as today; when every task is active (start <= step) it uses today's cumulative table (identical draws); otherwise a cumulative table over the active tasks (same order, renormalised), cached per active set. A single active task returns without a draw, as today. `active_tasks(step)` helper.
- `fit`: after `lr = lr_at(s, cfg.schedule)`, if the task has start > 0 and W > 0 and s < start + W: `lr *= (s - start + 1) / W` (logged `lr` is the applied value). No multiplication otherwise (bit-identical default).
- Validation runs every set at every validation step (a not-yet-started deform set shows identity-level values; `ckpt/best_deform` follows its usual rule).
- Resume: stateless, nothing stored; a changed spec on resume is allowed.
- The checker's grad-balance medians must start after the delayed task begins: the scripts pass `--grad-balance-from-step $((DEFORM_START + 100))` (section 6, BALANCE_CHECK). `check_mixed_run.py` already filters records by key presence, so windows without deform keys before that step are tolerated (verified by the review; no checker change for this).

## 6. SLURM scripts (new; m7 / m8 / m9 scripts unchanged)

Common to all three (header comments state everything below, like `slurm/m9_level1_cascade.sh`):

- `#SBATCH -A qtim -p rtx8000 --gpus=1 -N 1 --ntasks-per-node=1 --cpus-per-task=3 --mem=64G`, `--output/--error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.{out,err}`; `--time` per script below. Account `qtim` only (never `entre`). A100: `-p dgx-a100` documented, allowed only if the smoke's `INFO gpu_peak` <= 36 GiB (36864 MB); same CPUs / memory.
- Container `/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif`, binds and TMPDIR / KERAS_HOME / CUDA_CACHE_PATH exports exactly as m9; results `$PROJ/results/$RUN_NAME`.
- Recipe flags: `--amp true --intensity-aug-device tf --lowres-method $LOWRES_METHOD` (default `tent`), `--optimizer $OPTIMIZER` (default `adamw`; choices adamw|adam) `--lr $LR --weight-decay $WD --schedule cosine --warmup-steps $WARMUP --min-lr-factor 1e-3 --ckpt-every $CKPT_EVERY` (default 250 at every level), `BAL_ARGS` exactly as m9: GRAD_BALANCE=gradnorm (default) -> `--task-grad-norm 12` (no `--task-loss-weights` = none), deform64 -> `--task-loss-weights deform:64` (global clip 12 only), `--lambda-dice-template 1 --lambda-pdice 1 --lambda-ncc-template 1 --lambda-ncc-pair 1 --lambda-smooth 0.5 --p-pair 0.1 --p-pdice 0.1` (explicit even where equal to the code default, so the recipe is visible in the script), `--min-pairs $MIN_PAIRS` (default n), `--split-group-conv true`, `--log-every 25 --log-backend jsonl --val-png true`, `--num-threads ${SLURM_CPUS_PER_TASK:-3}`, `$VAL_ARGS` (`--val-group-size $VAL_GROUP_SIZE` when VAL_GROUP_SIZE is non-empty, else nothing; validated as an integer >= 2), then `$DEFORM_ARGS`.
- Task list and deform-only flags (parser rule `runtime/args.py` lines 475-480: `--gradient-checkpointing full`, `--lean-sampler`, `--lean-losses`, `--pair-recompute` need deform in the ratio): `TASKS=$(echo "$TASK_RATIO" | sed 's/:[^,]*//g')` (e.g. `rigid,affine,deform`); `HAS_DEFORM` = TASKS contains `deform` (`case ",$TASKS," in *,deform,*)`); `HAS_LINEAR` = TASKS contains rigid or affine. If HAS_DEFORM: the deform memory-knob set is assembled from the memory-knob envs below (see "Memory knobs"), and the deform loss lambdas / `--p-pdice` above move into DEFORM_ARGS too (they are accepted without deform but are meaningless; `--p-pair` stays common, it is the linear pair probability). Else `DEFORM_ARGS="--gradient-checkpointing none"`; GRAD_CKPT / PAIR_RECOMPUTE / LEAN_LOSSES env values are then ignored with an `echo NOTE: linear-only ratio, deform-only flags dropped` line. TASK_RATIO itself is validated by the pattern `^(rigid|affine|deform):[0-9]+(,(rigid|affine|deform):[0-9]+)*$` (exit 2 otherwise; further checks are main.py's).
- Memory knobs (identical rule in all three scripts; they must equal the knob set under which `scratch/m10/memory_batch.json` measured the N that fills the placeholders, `slurm/m10_batch_memory.sh` line 46 `DK`):
  - `AMP` is NOT an env: `--amp true` is hard-coded (the m10 job measured AMP only).
  - `--split-group-conv true` is hard-coded and common (also used by every m10 linear record; same variables, memory only goes down).
  - `GRAD_CKPT` default `full` at every level; validated against `none|integrator|decoder|both|full` (exit 2 otherwise). `PAIR_RECOMPUTE` default `true` (`true|false`). `MIN_PAIRS` default `n` (`1|n`; common flag, it also sets the linear pair minimum). `LEAN_LOSSES` default `false` at levels 4 and 2, `true` at level 1 (`true|false`).
  - With HAS_DEFORM and GRAD_CKPT != none: `DEFORM_ARGS="--gradient-checkpointing $GRAD_CKPT --lean-sampler true --pair-recompute $PAIR_RECOMPUTE"` plus `--lean-losses true` iff LEAN_LOSSES=true. With the defaults this is exactly m10's `DK` (+ `--lean-losses true` at level 1 = m10's "L1 deform N5 lean" record).
  - With HAS_DEFORM and GRAD_CKPT=none (speed path; meant for level 4 only, allowed but not recommended elsewhere): `DEFORM_ARGS="--gradient-checkpointing none"` only, i.e. exactly m10's "L4 deform no ckpt" record (`--gradient-checkpointing none --min-pairs n`, plus the common split conv); PAIR_RECOMPUTE / LEAN_LOSSES are then ignored with `echo NOTE: GRAD_CKPT=none uses the m10 no-ckpt knob set; PAIR_RECOMPUTE/LEAN_LOSSES dropped`. Reason: the no-ckpt record is the only measurement for that path, so the script reproduces it rather than mixing knobs that were never measured together.
  - Placeholder provenance: each `N` / `DEFORM_N` default line carries a comment `# TODO(m10 memory job): from record "<record name>" (knobs: <knob set>)`; the orchestrator fills in N from the record whose knob set equals the script's defaults (`L4 deform knobs` / `L2 deform knobs` / `L1 deform N5 lean` resp. the linear records) and rewrites the comment to `# m10 memory job <id>: record "<name>", peak <GB> at N <n>`. The r_level4 header states: `GRAD_CKPT=none` is allowed when the m10 "L4 deform no ckpt" record fits the chosen DEFORM_N (peak <= 46000 MB on the RTX 8000, <= 36864 MB for dgx-a100); otherwise keep `full`.
- `BGR_DRY_RUN=1` (test hook, default unset): after validation and argument assembly, print `MAIN: python main.py ...` and `CHECK: python tests/gpu/check_mixed_run.py ...` and exit 0 with no mkdir / write / singularity (test 11.5 h).
- Env overrides validated with `case` patterns (integers, floats via a regex `^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$`, enumerations); invalid -> `echo ERROR ... >&2; exit 2`. `MEM_ARGS` from `TF_MEM_LIMIT_MB` (0 = growth). `EXTRA_ARGS` appended last.
- After main.py: `python tests/gpu/check_mixed_run.py "$RESULTS/$RUN_NAME" --tasks "$TASKS" --steps "$STEPS" --expect-min-pairs "$MIN_PAIRS" --max-nonfinite-frac 0.06 --memory-budget-mb 36864 --project-steps $STEPS --project-val-every $VAL_EVERY --expect-enc-grad true $BALANCE_CHECK $LEVEL_CHECK_ARGS` (the checker's defaults `--tasks rigid,affine --steps 20` are wrong for these runs, so both are always passed, as in `slurm/m9_level1_cascade.sh`). `BALANCE_CHECK` = `--grad-balance-rule true --grad-balance-from-step <F>` only when HAS_DEFORM and HAS_LINEAR (F = `$((DEFORM_START + 100))` at level 4 when the task-start flags are passed, else 100), empty otherwise. `LEVEL_CHECK_ARGS` = the per-level checker flags listed below the table. Exit code: main.py's if non-zero, else the checker's.
- `N` / `DEFORM_N` defaults are PLACEHOLDERS marked `# TODO(m10 memory job)`; the orchestrator replaces them from `scratch/m10/memory_batch.json` (largest fitting N with the A100 rule noted).

| Script | Level / source | Defaults (env) | Time |
|---|---|---|---|
| `slurm/r_level4.sh` | level 4, from scratch, no `--init-weights` (`INIT` env optional, two-stage fallback only) | RUN_NAME `r4_<job>`, STEPS 16000, VAL_EVERY 500, CKPT_EVERY 250, WARMUP 1000, LR 3e-4, WD 0, N 64, DEFORM_N 32, TASK_RATIO rigid:1,affine:1,deform:2, DEFORM_START 3000, DEFORM_RAMP 500, CURRICULUM true + CURRICULUM_RAMP 3000, G_START 3000, G_RAMP 3000, G_ROT 45, G_TRANS 10, TF_MEM_LIMIT_MB 0, GRAD_CKPT full, PAIR_RECOMPUTE true, MIN_PAIRS n, LEAN_LOSSES false, VAL_PAIR_CHUNK 16, VAL_GROUP_SIZE 16 (fixed validation groups for every level-4 run, sweep and final, section 7.1), DEFORM_DECODER auto (auto\|true\|false -> `--deform-decoder`), INIT unset (set: must be an existing directory, passed as `--init-weights`, else exit 2) | 24:00:00 |
| `slurm/r_level2.sh` | level 2 cascade, `CASCADE_FROM` required (no default run; e.g. `$RESULTS/r4_final/ckpt/best`), m8 recipe | STEPS 8000, VAL_EVERY 500, WARMUP 500, LR 3e-4, WD 0, N 32, DEFORM_N 16, BN_RECAL_BATCHES 64 (freeze true), SKIP_LINEAR `4:0`, SKIP_DEFORM `4:0.15`, G from 0, curriculum off, GRAD_CKPT full, PAIR_RECOMPUTE true, MIN_PAIRS n, LEAN_LOSSES false (NOT m8's `GRAD_CKPT=none` / `AMP=false` script defaults: m8 set `both` and AMP by env; here the m10 knob set is the default), VAL_PAIR_CHUNK 4, TF_MEM_LIMIT_MB 0, VAL_GROUP_SIZE empty (main.py default min(N, n_val), as m8) | 24:00:00 |
| `slurm/r_level1.sh` | level 1 cascade, `CASCADE_FROM` required (an r_level2 run's ckpt/best), m9 recipe | STEPS 8000, VAL_EVERY 500, CKPT_EVERY 250, WARMUP 500, LR 3e-4, WD 0, N 8 (int32 limit: N >= 10 overflows the stem), DEFORM_N 4, SKIP_LINEAR `4:0,2:0.15`, SKIP_DEFORM `4:0.15,2:0.15`, GRAD_CKPT full, PAIR_RECOMPUTE true, MIN_PAIRS n, LEAN_LOSSES true (the m9 long run needed it; = m10 "L1 deform N5 lean" knob set), TF_MEM_LIMIT_MB 40000 (A100: 34000), VAL_IMAGE_CHUNK 1, VAL_PAIR_CHUNK 1, BN_RECAL_VALIDATE false, BN_RECAL_BATCHES 64, VAL_GROUP_SIZE empty (as m9) | 60:00:00 |

Level 4 passes `--curriculum $CURRICULUM --curriculum-ramp-steps $CURRICULUM_RAMP` (CURRICULUM validated `true|false`; the ramp flag only when CURRICULUM=true) `--global-transform true --global-start-step $G_START --global-ramp-steps $G_RAMP --task-start-step deform:$DEFORM_START --task-start-ramp-steps $DEFORM_RAMP` (the task-start flags only when DEFORM_START > 0 and deform is in TASK_RATIO), checker `--expect-init-weights none` (or `full` when INIT is set). A dip of the linear val metrics around step DEFORM_START is expected (curriculum end, G start and deform start coincide at 3000 in the default schedule) and is not a collapse signal; collapse is only `train/collapse_suspect` / `diag_feat_rel_spread` < 1e-3. Levels 2 / 1 pass `--cascade-skip-linear "$SKIP_LINEAR" --cascade-skip-deform "$SKIP_DEFORM"` and the checker `--expect-cascade true --expect-skip true` (+ `--expect-cascade-stages 2` at level 1, `--expect-bn-recal true` when BN_RECAL_BATCHES > 0, `--expect-init-weights full`). Both keep m8 / m9's CASCADE_FROM guard (skipped when `ckpt/last/checkpoint` exists) and their resume plan (resubmit with the same RUN_NAME).

Header extras: `r_level4.sh` also lists the two-stage fallback (section 5.1) verbatim: stage 1 `RUN_NAME=r4_lin TASK_RATIO=rigid:1,affine:1 DEFORM_DECODER=true sbatch --job-name=r4lin --time=24:00:00 --mem=64G slurm/r_level4.sh` (linear-only: DEFORM_ARGS = `--gradient-checkpointing none`, no task-start flags, BALANCE_CHECK empty), stage 2 `RUN_NAME=r4_3t TASK_RATIO=rigid:1,affine:1,deform:2 INIT=$RESULTS/r4_lin/ckpt/best DEFORM_START=0 CURRICULUM=false G_START=0 G_RAMP=0 sbatch --job-name=r4_3t --time=24:00:00 --mem=64G slurm/r_level4.sh`. `r_level1.sh` states: before the long run, compare the smoke's `INFO gpu_peak` with m9_level1_cascade's (36.7 GB at the same N 8 / DEFORM_N 4 / lean losses); the stage-mode `deform_skip` with skip2 materialises up to four extra transient [4, 160, 224, 192, 3] float32 fields (~1.3 GB, outside the tape) and m9 had ~0.6 GB headroom under 40000 MB; if the smoke peak exceeds 39000 MB, use the fallback `MIN_PAIRS=1`.

Every script header ends with the exact smoke and long commands, e.g. `RUN_NAME=r4_smoke STEPS=300 VAL_EVERY=100 WARMUP=50 DEFORM_START=100 DEFORM_RAMP=100 CURRICULUM_RAMP=100 G_START=100 G_RAMP=100 sbatch --job-name=r4smoke --time=02:00:00 --mem=64G slurm/r_level4.sh`, with "DO NOT submit without Jay's approval; at most two active jobs (`squeue -u jn85`)".

## 7. Level-4 sweep (`slurm/sweep_level4.py`)

### 7.1 Configurations

- Phase A (grid): lr {1e-4, 3e-4, 1e-3} x wd {0, 1e-2, 1e-1} at (N_max, DEFORM_N_max): 9 runs, ordered wd-major with wd 0 first (lr 3e-4, 1e-4, 1e-3, then the next wd), so stopping after the first three already gives the lr answer.
- Phase B (group sizes, at the phase-A winner lr*/wd*): (N_small, DEFORM_N_max) and (N_max, DEFORM_N_small): 2 runs. N_small / DEFORM_N_small = the next smaller measured size (default N_max / 2, DEFORM_N_max / 2). The phase-A winner itself is the (N_max, DEFORM_N_max) point of phase B.
- Phase A and B run length: STEPS 8000 with the schedule scaled: DEFORM_START 2000, DEFORM_RAMP 500, CURRICULUM_RAMP 2000, G_START 2000, G_RAMP 2000, WARMUP 1000, VAL_EVERY 500 (`--schedule cosine --min-lr-factor 1e-3 --steps 8000`: cosine from the peak to 1e-3 x peak at step 8000). Time 14:00:00 (estimate ~3.6 s mean step on the RTX 8000 at N 64 / 32 with AMP -> ~8 h + validation; corrected after the smoke's `INFO step_time` line).
- Phase C (number of iterations, at the phase-A/B winner lr*, wd*, N*, DEFORM_N*): STEPS 12000 and 16000 with the FINAL schedule (DEFORM_START 3000, DEFORM_RAMP 500, CURRICULUM_RAMP 3000, G_START 3000, G_RAMP 3000, WARMUP 1000, VAL_EVERY 500): 2 runs, time 24:00:00 each. With the 8000-step phase-A/B winner run this gives a 3-point length curve (8000 / 12000 / 16000), each point a complete cosine schedule ending at its own STEPS, which is what "number of iterations" means under a cosine schedule (the best step of a single run says little: validation usually improves until the decay ends). lr* / wd* are tuned at 8000 steps; phase C checks only the length, not a re-tuned lr / wd at 16000 (stated as a limitation in the report; the lr-scaled decay keeps wd's total effect proportional to sum lr_t, so a longer run with the same wd decays somewhat more).
- Validation groups: every sweep run (A, B, C) passes `VAL_GROUP_SIZE=16` (r_level4.sh -> `--val-group-size 16`). Reason: `main.py` lines 182 / 192 set the linear validation group to `min(N, n_val)` and the deform one to `min(DEFORM_N, n_val)` when the flag is absent, so phase B would otherwise evaluate N 32 vs 64 on 32- vs 39-image groups (template = group mean; center / pair RMS and template Dice depend on the group). 16 is the smallest DEFORM_N in any phase and was the deform validation group of m7 (fits memory there). Consequence: level-4 linear val numbers are not directly comparable with m5 / m7 linear numbers (32-image groups); deform numbers are. The summarizer records `val_group_size` per run and refuses to rank runs with different values (8).
- Wall clock: A + B = 11 runs at <= 2 concurrent ≈ 6 waves x ~10 h ≈ 60 h; C = 2 concurrent runs ≈ 18 h (the 16000 run dominates); total ≈ 78 h.
- Run names (`fmt(x) = "0" if x == 0 else ("%.0e" % x).replace("e-0", "e-").replace("e+0", "e")`, so 3e-4 -> `3e-4`, 1e-2 -> `1e-2`, 1e-1 -> `1e-1`; values whose `%.0e` form does not round-trip to the given float (e.g. 2.5e-4) use `("%.1e" % x)` with the same replacements; fmt is unit tested): phase A / B `r4sw_<P>_lr<fmt(lr)>_wd<fmt(wd)>_n<N>_d<DN>` (e.g. `r4sw_A_lr3e-4_wd1e-2_n64_d32`), phase C `r4sw_C_s<STEPS>_lr<...>_wd<...>_n<N>_d<DN>`. Job names `r4sw<k>` with k ONE global index across phases in sweep order: A 0-8, B 9-10, C 11-12 (so a phase-B job can never share a job name with a phase-A job in `squeue -o %j`). If the phase-B winner is the phase-A run, phase B's two runs are still distinct names (different n / d).
- Every submission passes the full env set: RUN_NAME, LR, WD, N, DEFORM_N, STEPS, VAL_EVERY, WARMUP, DEFORM_START, DEFORM_RAMP, CURRICULUM_RAMP, G_START, G_RAMP, VAL_GROUP_SIZE (all as strings; everything else takes r_level4.sh's defaults, so the recipe lives in one place).

### 7.2 Helper CLI (stdlib only, runs on the login node, one-shot, never a daemon)

```
python3 slurm/sweep_level4.py list   [--phase A|B|C] [grid options]   # configs + exact sbatch lines
python3 slurm/sweep_level4.py status [--phase A|B|C] [grid options]   # per config: done / failed / running / pending / not started
python3 slurm/sweep_level4.py submit [--phase A|B|C] [grid options] [--max-active 2] [--dry-run] [--a100-ok]
grid options: --lrs 1e-4,3e-4,1e-3 --wds 0,1e-2,1e-1 --n 64 --deform-n 32 --n-small 32 --deform-n-small 16
              --steps 8000 --time 14:00:00 --c-steps 12000,16000 --c-time 24:00:00 --val-group-size 16
              --partition rtx8000 --best-lr X --best-wd Y (phases B, C: required)
              --best-n N --best-deform-n DN (phase C; default --n / --deform-n)
              --results /autofs/.../GroupwiseReg/results --repo /autofs/.../brain-group-rad --user jn85
```

- `--phase` default A. Phase B / C without `--best-lr` / `--best-wd` -> exit 2 with a message. Grid values parsed as floats (lr > 0, wd >= 0) or ints (N, DN, steps >= 1; `--c-steps` strictly increasing), else exit 2.
- Status states (per config, evaluated in this order): **running** / **pending** = the job name is in `squeue -h -u <user> -o "%j %T"` (state RUNNING -> running, any other state -> pending); **done** = the run dir exists and the last metrics.jsonl record with a `data` dict has `step == config.json args.steps` and a `val/mean_pair_rms_mm` key; **failed** = the run dir exists, the job is not in squeue, and it is not done; **not started** = no run dir. `status` prints one line per config: `<k> <job name> <run> <state> [last step]`.
- `submit`: `active = len(squeue -h -u <user>)` lines (ALL of the user's jobs count, including Jay's own); `free = max_active - active`; `--max-active` must be 1 or 2 (> 2 -> exit 2). It submits at most `free` configs in sweep order whose state is **not started**; **failed** configs are never resubmitted, only printed (`FAILED, not resubmitted: <run>`; the orchestrator decides with Jay, e.g. a resume with the same RUN_NAME by hand). Each submission: `subprocess.check_output(["sbatch", "--parsable", "-A", "qtim", "-p", partition, "--job-name", name, "--time", time, "--mem", "64G", "slurm/r_level4.sh"], env=job_env(cfg_env), cwd=repo)` with `cfg_env` the full env set of 7.1 (list args, no shell). `job_env(cfg_env) = {k: v for k, v in os.environ.items() if k not in SCRIPT_ENV_KEYS}` updated with `cfg_env`, where the module constant `SCRIPT_ENV_KEYS` lists every env name `r_level4.sh` reads (RUN_NAME, STEPS, VAL_EVERY, CKPT_EVERY, WARMUP, LR, WD, OPTIMIZER, N, DEFORM_N, TASK_RATIO, DEFORM_START, DEFORM_RAMP, CURRICULUM, CURRICULUM_RAMP, G_START, G_RAMP, G_ROT, G_TRANS, TF_MEM_LIMIT_MB, GRAD_CKPT, PAIR_RECOMPUTE, MIN_PAIRS, LEAN_LOSSES, LOWRES_METHOD, GRAD_BALANCE, VAL_PAIR_CHUNK, VAL_GROUP_SIZE, DEFORM_DECODER, INIT, EXTRA_ARGS, BGR_DRY_RUN): a stale value in the login shell (e.g. `EXTRA_ARGS`) can never leak into a sweep job; the script's header defaults apply to everything not in `cfg_env`. A test asserts that SCRIPT_ENV_KEYS covers every `${NAME:-` default in r_level4.sh (parsed with a regex) so the two cannot drift.
- Code revision: before submitting anything, `submit` (not `--dry-run`) runs `git -C <repo> status --porcelain` and `git -C <repo> rev-parse HEAD` (`subprocess.check_output`, list args); a non-empty status -> stderr `ERROR: repo has uncommitted changes; commit/pull first` and exit 2 (nothing submitted); git missing / non-zero -> exit 1. Each jsonl record gets `"rev": <sha>`. Limitation (stated in the help text): a job runs the checkout present when it STARTS, so the orchestrator must not `git pull` in the cluster clone while sweep jobs are pending; the scripts print `repo rev $(git -C $REPO rev-parse HEAD)` at start (as m9) for the job log, and `summarize_runs.py` shows a `rev` column taken from the last `<results_root>/r4sw_submissions.jsonl` record with that run name (`?` if absent or the file is missing; first 8 characters), and warns (stderr, not an error) when the ranked runs have different revs.
- Appends `{time, job_id, run, rev, env}` to `<results>/r4sw_submissions.jsonl`. `--dry-run` prints the commands only (no git check).
- Partition must be `rtx8000` or `dgx-a100` (else exit 2); the account is hard-coded `qtim`. `dgx-a100` additionally needs `--a100-ok` (else exit 2 with "dgx-a100 only after a smoke with INFO gpu_peak <= 36864 MB; pass --a100-ok to confirm"); `list` / `submit --dry-run` with dgx-a100 print the same reminder line.
- Errors: `squeue` / `sbatch` missing or non-zero -> message on stderr, exit 1, nothing further submitted; a failed sbatch stops the loop (no retries).
- Approval model: Jay approves the sweep once as a batch (phases A-C); the orchestrator runs `submit` whenever a job ends (at most two active). No SLURM arrays, no background loop on the login node.

### 7.3 Selection rule and final run length

Summarizer (section 8) ranks the finished runs of one phase group. For each run: L = mean of `val/mean_pair_rms_mm` over the last two validations (lower is better), Dp = mean of `val_deform/pair_dice` over the last two validations (higher is better). Excluded (marked `X`): runs with a non-finite final metric, `nonfinite_total / steps > 0.06`, `train/collapse_suspect > 0`, or `val_deform/neg_jac_pct > 0.5`. "Final" always means the value in the LAST record that CONTAINS that key (the last record of a finished run is the step-STEPS validation record, which has no train keys): `nonfinite_total` and `train/collapse_suspect` from the last train-window record containing them, `val_deform/neg_jac_pct` and the other val keys from the last record containing them; a key never logged -> treated as missing (for the exclusion keys: not excluded, shown as `?`). Score = rank(L) + rank(Dp) (1 = best); tie-break by lower final `val_deform/neg_jac_pct`, then lower final `val_affine_g/pair_rms_mm`. Phases A and B are ranked together (all 8000 steps, same schedule, same validation groups): the winner's lr / wd come from the phase-A runs, N / DEFORM_N from the winner among phase B plus the phase-A winner (keep N_max / DEFORM_N_max unless a smaller size wins on score).

Final length (phase C), pure function `length_decision(points, tol_l=0.05, tol_dp=0.003)` in `scripts/summarize_runs.py`, points = [(steps, L, Dp)] for 8000 (the A/B winner run), 12000 and 16000:
- s* = the shortest length with L_s <= L_16000 + 0.05 mm AND Dp_s >= Dp_16000 - 0.003.
- If s* = 16000 and 16000 beats 12000 by more than the tolerance on BOTH metrics (L_12000 - L_16000 > 0.05 and Dp_16000 - Dp_12000 > 0.003), the curve is still steep: the final is a new 20000-step run with the final schedule (`RUN_NAME=r4_final_s20000`, printed as an exact sbatch command, submitted only after Jay's approval).
- Otherwise the s* run itself is the final level-4 run (no rerun; the 16000-step phase-C run doubles as the final when chosen). Its `ckpt/best` (linear rule) is the level-2 cascade source, as m8 did.
- A missing or excluded point -> the decision is `UNDECIDED (<reason>)`, never a guess.

### 7.4 Python version

The helpers must run on the login node's `python3` (version not verified; CentOS-8 hosts ship 3.6). They use only Python 3.6 syntax and stdlib (no dataclasses, no `subprocess.run(capture_output=...)`, no `text=` (decode `check_output` bytes with `.decode("utf-8")`), no `datetime.fromisoformat`, no walrus, no f-string `=`); verified with `ast.parse(src, feature_version=(3, 6))` plus the token check of test 11.5 f locally.

## 8. Run summarizer (`scripts/summarize_runs.py`)

`python3 scripts/summarize_runs.py <results_root> [--glob 'r4sw_*'] [--csv out.csv] [--last-k 2] [--rank true]` (stdlib only, Python 3.6, read-only).

- For each `<results_root>/<run>/logs/metrics.jsonl` matching the glob: parse line by line (`json.loads`; a malformed line -> skipped with a counter printed to stderr; a missing file -> run listed with `no metrics`); records with a `data` dict. Hyperparameters from `<run>/config.json` `args` (lr, weight_decay, optimizer, group_size, deform_group_size, steps, val_group_size); missing -> `?`.
- Ranking guard: if the ranked (non-excluded) runs have different `val_group_size` values (None counts as its own value), print `ERROR: runs use different validation groups (<values>); not ranked` to stderr, print the table unranked, exit 0 (the comparison would be invalid, finding 5 of the review).
- `--length-curve RUN8,RUN12,RUN16` (optional): computes L / Dp for those three runs (any order; steps from config.json) and prints `LENGTH <steps> (run <name>)`, `LENGTH 20000 (new run; command: RUN_NAME=r4_final_s20000 STEPS=20000 LR=.. WD=.. N=.. DEFORM_N=.. VAL_GROUP_SIZE=16 sbatch --job-name=r4final --time=30:00:00 --mem=64G slurm/r_level4.sh)` or `LENGTH UNDECIDED (<reason>)` via `length_decision` (7.3).
- Columns: run, lr, wd, N, DN, val group size, last step / steps, wall h (last `time` - first `time`), max `gpu_peak_mb` in GB, nonfinite_total; for each of `val_rigid`, `val_affine`, `val_rigid_g`, `val_affine_g`: center / pair RMS final and best (min) with best step; `val/mean_pair_rms_mm` final / best / best step; `val_deform/template_dice`, `pair_dice` final / best (max) / best step; `val_deform/ncc_template`, `ncc_pair`, `neg_jac_pct`, `inv_consistency_mm` final; L, Dp, score, rank, excluded flag (7.3) when `--rank true` (default).
- Output: fixed-width text table to stdout sorted by rank (excluded last), plus `--csv` (stdlib `csv`). Exit 0 even when some runs are incomplete; exit 2 only if `<results_root>` does not exist.

## 9. design.md bump (v3.15 -> v3.16)

Edit `.agents/tasks/network-design/design.md`: status line v3.16 (2026-10-26) with a one-paragraph summary pointing here; 4.7 planned skip schedule: mark "implemented (retrain-prereqs)", replace the level-4 skip sentence with section 4.1's semantics (independent draws, skip 4 keeps level 2 running on un-pre-warped inputs, skip both = identity), and also replace the v2 "Level skipping" sentence "The decision is made once for the whole group and applies to both linear and deformable cascades." by "One draw per stage, per task and per step, shared by the whole group (a group is never split); linear (rigid, affine) and deformable draws are independent."; 5.2 optimizer: AdamW decision and the fact that `--weight-decay` never acted on Adam; 3.2 / 3.3 augmentation: `--lowres-method tent` (TF only) with the measured centroid table of 3.1; 5.x: the one-run level-4 schedule (delayed deform start) and the sweep; milestones table: a "retrain from scratch" row (prerequisites done, runs pending).

## 10. Error handling summary

| Operation | Failure | Kind | Caller gets | Logged |
|---|---|---|---|---|
| Parse `--cascade-skip-*`, `--task-start-step`, `--lowres-method`, `--weight-decay` | bad syntax / level / prob / task / step / combination | fatal | argparse error, exit 2 | stderr |
| Trainer construction with invalid skip / start config (programmatic use) | as above | fatal | ValueError | - |
| `lowres_resample(method=...)` | unknown method | fatal | ValueError | - |
| Resume with a different `--optimizer` | slot mismatch | fatal | SystemExit with message | stderr |
| Resume with a different lr / wd | none | recoverable | continues | event `optimizer_hparam_changed_on_resume` |
| Level-1 stage-mode skip of level 4 for a task whose level-2 source never skipped level 4 | possible out-of-distribution frozen stage | recoverable (warning) | continues; `cascade.skip_consistency_warnings` in config.json | stdout WARNING line |
| `train_step` in stage mode with a bare bool | programming error | fatal | TypeError | - |
| Sweep `submit`: squeue / sbatch fails | cluster error | fatal for that invocation | exit 1, earlier submissions kept in the jsonl | stderr |
| Sweep `submit`: >= max-active jobs | none | normal | "NOT submitted: N active" | stdout |
| Sweep: bad grid value, phase B / C without `--best-lr` / `--best-wd`, `--max-active` > 2, unknown partition, dgx-a100 without `--a100-ok` | config error | fatal | exit 2, nothing submitted | stderr |
| Sweep `submit`: repo has uncommitted changes | mixed / unknown code version | fatal for that invocation | exit 2, nothing submitted | stderr ERROR |
| Sweep `submit`: git missing or failing | cannot record revision | fatal for that invocation | exit 1, nothing submitted | stderr |
| Summarizer: ranked runs with different `rev` | possibly mixed code versions | recoverable | ranked table, exit 0 | stderr WARNING |
| Sweep `submit`: config in state failed | earlier job died | normal (not retried) | "FAILED, not resubmitted: <run>" | stdout |
| Summarizer: ranked runs with different `val_group_size` | invalid comparison | recoverable | unranked table, exit 0 | stderr ERROR |
| Summarizer `--length-curve`: missing / excluded point | incomplete curve | recoverable | `LENGTH UNDECIDED (<reason>)` | stdout |
| Summarizer: missing / malformed metrics | partial data | recoverable | row with `no metrics` / skipped-line count | stderr |
| SLURM scripts: invalid env value (incl. GRAD_CKPT, CURRICULUM, PAIR_RECOMPUTE, LEAN_LOSSES, MIN_PAIRS outside their choices), missing CASCADE_FROM on a fresh run | config error | fatal | exit 2 before singularity | stderr (job .err) |

Invariant ownership: flag combinations are checked in `runtime/args.py` (user-facing errors) and re-checked in `Trainer.__init__` (programmatic use, tests); the identity-transform semantics of a skipped stage are owned by `models/cascade.py` (one place for both trainer paths); the "never skip in validation" invariant by the validation functions not taking a skip input; the two-job cluster limit by `sweep_level4.py submit` (and the orchestrator for any other submission).

## 11. Tests (unittest; every TF test imports `tests.tf_setup` first; TINY / narrow configs)

### 11.1 AdamW (`tests/test_runtime_optim.py`)
a. wd 0: AdamW and Adam from identical variables and 5 identical random gradient steps (with a changing lr via `set_learning_rate`) give max |diff| 0 for variables and slots.
b. Decay only on the right variables: on `GroupRegLevelModel(4, narrow_config())`, `uses_weight_decay` is False exactly for names whose leaf contains gamma / beta / bias and True for every kernel; `exclude_wd=False` decays all.
c. Magnitude: lr 0.1, wd 0.01, zero gradients for one step -> decayed variables equal (1 - 1e-3) x before (rtol 1e-6), excluded ones unchanged.
d. AMP: `make_optimizer("adamw", amp=True)` is a LossScaleOptimizer whose inner is AdamW; a step with finite gradients decays; a step with an inf gradient changes no variable (decay skipped).
e. Checkpoint round trip: save after 3 steps, restore into a fresh optimizer + model, 2 more steps equal (max |diff| 0) to an uninterrupted run.
f. `tests/test_main.py`: `--optimizer adam --weight-decay 1e-4` prints the NOTE; resume guard raises SystemExit on an optimizer change; config.json `optimizer` record present for adamw only.

### 11.2 Low-res (`tests/test_tf_intensity_aug.py`, `tests/test_main.py`)
a. `method="nn"` equals today's output (max |diff| 0) for every ROI, f in (0.5, 0.63, 0.8, 0.97, 1.0).
b. `tent` vs `intensity_ref.lowres_tent_np`: max |diff| < 1e-5 on random volumes (11, 16, 13) and (40, 56, 48).
c. Centroid test: 3-D volumes of level-4 (40, 56, 48) and level-1 (160, 224, 192) shape (level 1: one axis at a time on a [n, 4, 4] slab to keep it light), 0/1 slabs of widths w in {12, 16, 20} along the tested axis at the 5 start positions `np.linspace(2, n - w - 2, 5).round().astype(int)` (every slab starts at >= 2 and ends at <= n - 2), f in {0.5, 0.55, ..., 0.95}: |centroid displacement| < 0.02 voxel (measured worst 0.0181); the same grid with `nn` exceeds 0.1 for at least one case (documents the bias). A second, informational assertion with an edge-touching slab (start 0, w 12) bounds the tent shift by 0.07 voxel (measured 0.0625).
d. Mass and constants: sum preserved (rtol 1e-5); a constant image stays constant (max |diff| 1e-6); f = 1 returns the input (max |diff| 0).
e. Static shape and graph mode: inside `tf.function` with a tensor f, output shape == input shape, no retracing for 3 different f.
f. Parser: `--lowres-method tent` with `--intensity-aug-device dali` errors; with tf, `TrainConfig.intensity_aug_tf.low_res_method == "tent"`; default `nn`.

### 11.3 Skip (`tests/test_cascade.py`, `tests/test_trainer.py`, `tests/test_main.py`)
a. Parser: valid specs; errors for level not frozen in the run (`2:` at `--level 2`), p = 1, duplicate level, malformed entry, any spec (incl. `4:0` and `""`) without `--cascade-from` (also at `--level 4`), both legacy and stage flags. `tests/test_main.py`: a resume with a changed skip spec refreshes `cascade.skip_mode` / `skip_probs` and writes the `cascade_skip_changed_on_resume` event; a resume with a changed lr refreshes the `optimizer` record.
b. Draw: deterministic per (seed, step, task, level); independent across levels and tasks (over 20000 steps, empirical rates within 0.01 of p and the joint rate within 0.01 of p4 x p2); p = 0 never skips.
c. Level 1 chain equality: `linear_skip` / `deform_skip` with `skip4=None` and skip2 True / False equal today's `linear_stages` / `deform_stages` + `skip_to` (max |diff| 0); skip4 True: A_prev equals `compose_linear(R2_raw(model2 on prewarp(x2, I)), I)` and differs from the unskipped value; difference of `prewarp_linear(x2, I)` to x2 < 1e-4 and of the U4 = 0 composition to u2 < 1e-5; both True -> A_prev = I, U = U_inv = 0 exactly.
d. Level 2: stage mode with deform 4:p and skip True gives the same step results as legacy with skip True (max |diff| 0, same seeds).
e. Trainer: stage-mode step signature `[n_stages]` bool; one trace per task across mixed skip values; tasks with all p = 0 build no `cascade_skip` op; logged `train_<task>/cascade_skip_l<lv>` keys only for p > 0; validation functions contain no `cascade_skip` op.
f. Defaults: no skip flags -> graphs and goldens unchanged (the existing golden and `TestSkip` tests run unmodified).
g. `main._skip_consistency`: source record with `skip_probs.deform."4" = 0.15` and level-1 deform 4:0.15 -> []; linear 4:0.1 against a source with linear 0 -> ["rigid", "affine"]; source without `skip_probs` and `skip_prob` 0.1 -> []; `skip_prob` 0 -> every task with p4 > 0; None source -> [].

### 11.4 Delayed start (`tests/test_trainer.py`, `tests/test_main.py`)
a. `TaskSchedule` without starts equals today's draws for steps 0..9999 (exact).
b. With `deform:3000`: no deform before 3000; linear tasks split ~1:1 before and the ratio 1:1:2 after (within 0.02 over 20000 draws); deterministic.
c. LR ramp: applied lr on the delayed task's steps = lr_at(s) x (s - start + 1) / W in the ramp, equal after; other tasks unchanged.
d. Parser: unknown task (not in `--task-ratio`), start >= steps, no task at 0, negative values, duplicates, `--task-start-ramp-steps` > 0 without a start > 0 -> errors.
e. Short Trainer run (TINY, synthetic, 12 steps, start 6): no deform steps before 6, deform steps after, finite.

### 11.5 Helpers (`tests/test_sweep_tools.py`, no TF)
a. `sweep_level4.py list` builds 9 phase-A, 2 phase-B and 2 phase-C configs in the documented order, names and global job indices (A 0-8, B 9-10, C 11-12); phase C uses the final schedule (DEFORM_START 3000, CURRICULUM_RAMP / G_START / G_RAMP 3000, DEFORM_RAMP 500) and STEPS 12000 / 16000; every config env contains the full 7.1 key set incl. VAL_GROUP_SIZE 16; phase B / C without `--best-lr` errors; `fmt` gives `0`, `1e-4`, `3e-4`, `1e-3`, `1e-2`, `1e-1`, `2.5e-4`.
b. `submit` with a fake `squeue` / `sbatch` on PATH (temporary scripts): 2 active -> nothing submitted; 1 active -> exactly one; existing run dir skipped; a failed run (dir present, not done, not in squeue) is printed and not resubmitted; sbatch args contain `-A qtim` and never `entre`; `--max-active 3` exits 2; `--partition dgx-a100` without `--a100-ok` exits 2.
c. `status` states on a synthetic tree + fake squeue: running, pending, done (last step == args.steps with `val/mean_pair_rms_mm`), failed, not started.
d. `summarize_runs.py` on a synthetic results tree (two runs, one with a malformed line, one excluded by folding): correct best / final / steps, ranking and CSV; two runs with different `val_group_size` -> not ranked, ERROR line.
e. `length_decision`: (i) 12000 within tolerance of 16000 on both, 8000 not -> 12000; (ii) 8000 within tolerance -> 8000; (iii) 16000 better than 12000 by > tol on both -> 20000; (iv) 16000 better on only one metric -> 16000; (v) a missing point -> UNDECIDED.
f. Both helpers parse with `ast.parse(..., feature_version=(3, 6))`, and their source contains none of the 3.7+ stdlib tokens `capture_output`, `text=True`, `import dataclasses`, `from dataclasses`, `fromisoformat` (plain substring check; grammar alone does not catch them).
g. Env isolation and revision (`submit` with fake `squeue` / `sbatch` / `git` on PATH; the fake sbatch dumps its environment to a file): with `EXTRA_ARGS=x` and `TASK_RATIO=deform:1` set in the caller's env, the recorded job env has no `EXTRA_ARGS` and `TASK_RATIO` is absent (not in cfg_env), while every cfg_env key is present with the config's value; `SCRIPT_ENV_KEYS` is a superset of the names found by the regex `\$\{([A-Z_][A-Z0-9_]*):-` in `slurm/r_level4.sh`; fake `git status --porcelain` printing ` M x.py` -> exit 2 and no sbatch call; clean fake git -> the jsonl record has `rev` = the fake sha.
h. Scripts (`tests/test_sweep_tools.py`, skipped when `bash` is unavailable): each r_level script runs with `BGR_DRY_RUN=1` (new env, documented in the headers: validate env, build the argument lists, print the main.py and checker command lines, exit 0 before any mkdir, file write or singularity call; the CASCADE_FROM guard still runs and only reads paths) for: r_level4 default ratio -> main.py args contain `--gradient-checkpointing full --lean-sampler true --task-start-step deform:3000 --val-group-size 16` and the checker line `--tasks rigid,affine,deform --steps 16000 --grad-balance-rule true --grad-balance-from-step 3100`; r_level4 `TASK_RATIO=rigid:1,affine:1 DEFORM_DECODER=true` -> `--gradient-checkpointing none`, no `--lean-sampler`, `--pair-recompute`, `--task-start-step` or `--grad-balance-rule`, checker `--tasks rigid,affine`; r_level4 `TASK_RATIO=deform:1` -> no `--grad-balance-rule`; invalid `LR=abc`, `TASK_RATIO=foo`, `GRAD_CKPT=some` or `CURRICULUM=yes` -> exit 2; r_level2 / r_level1 without CASCADE_FROM on a fresh run -> exit 2. Memory knobs (finding 1 of review v1.2): every dry run's main.py line contains `--amp true --split-group-conv true --min-pairs n`; r_level4 default, r_level2 and r_level1 (CASCADE_FROM = a temporary directory, RESULTS / RUN_NAME pointing at a non-existent run so the fresh-run guard passes) contain `--gradient-checkpointing full --lean-sampler true --pair-recompute true`; only r_level1 contains `--lean-losses true`; r_level4 `GRAD_CKPT=none` -> `--gradient-checkpointing none` and no `--lean-sampler`, `--pair-recompute` or `--lean-losses`. Curriculum (finding 4): r_level4 default contains `--curriculum true --curriculum-ramp-steps 3000`; the fallback stage-2 env (`TASK_RATIO=rigid:1,affine:1,deform:2 INIT=<temp dir> DEFORM_START=0 CURRICULUM=false G_START=0 G_RAMP=0`) gives `--curriculum false`, no `--curriculum-ramp-steps`, no `--task-start-step`, `--init-weights <temp dir>` and checker `--expect-init-weights full --grad-balance-from-step 100`.

## 12. Local verification plan

1. Syntax: `py_compile` and `ast.parse(feature_version=(3, 8))` of every changed Python file in WSL env `bgr_preproc38`; `(3, 6)` for the two helpers.
2. `bash -n slurm/r_level4.sh slurm/r_level2.sh slurm/r_level1.sh`, then the `BGR_DRY_RUN=1` cases of test 11.5 h (default three-task ratio, linear-only fallback stage 1, fallback stage 2, deform-only, GRAD_CKPT=none, r_level2 / r_level1 knob sets, invalid env), run in WSL.
3. Targeted suites first: `tests.test_runtime_optim`, `tests.test_tf_intensity_aug`, `tests.test_cascade`, `tests.test_trainer`, `tests.test_main`, `tests.test_sweep_tools`.
4. Short local end-to-end at level 4 with TINY / narrow widths on synthetic data (`--overfit-one-group --overfit-synthetic` cannot use TF aug, so a second e2e uses the existing synthetic DALI-free path of the trainer tests): `--optimizer adamw --weight-decay 1e-2 --task-start-step deform:6 --task-start-ramp-steps 4 --amp true`, ~20 steps, finite, deform starts at 6.
5. Full suite once: `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/tests/run_tests_wsl.sh` in the background with a ~3.5 h timeout; baseline 558 OK + 4 skips, expected 558 + new tests, 0 failures; goldens max |diff| 0. Local GPU TF memory limit `BGR_TF_GPU_LIMIT_MB` <= 6000.
6. Not verifiable locally (stated in the report): RTX 8000 / A100 memory and step times at the new N, the login node's python3 version, real-data training behaviour of the one-run schedule and of AdamW.

## 13. Files

Modified: `runtime/optimizers.py` (docstring, constant), `runtime/args.py`, `runtime/run.py` (TaskSchedule, TrainConfig, Trainer skip modes, fit), `models/cascade.py` (`linear_skip`, `deform_skip`), `data_loading/params.py`, `data_loading/tf_intensity_aug.py`, `data_loading/intensity_ref.py`, `main.py` (optimizer line / record / resume guard, cascade skip record, lowres description), `tests/gpu/check_mixed_run.py` (stage-mode skip check only), `.agents/tasks/network-design/design.md` (v3.16), tests listed in section 11.
New: `slurm/r_level4.sh`, `slurm/r_level2.sh`, `slurm/r_level1.sh`, `slurm/sweep_level4.py`, `scripts/summarize_runs.py`, `tests/test_sweep_tools.py`, `.agents/tasks/retrain-prereqs/report.md` (at the end).

## 14. Assumptions and decisions needing confirmation

1. Weight-decay grid {0, 1e-2, 1e-1} instead of {0, 1e-5, 1e-4}. Under the standard lr-scaled AdamW decay the latter values have no measurable effect (section 2.1). The grid is a helper flag either way; the implementation uses the decided default and needs no answer to proceed. **CONFIRMED by Jay (2026-10-26, section 16).**
2. **CONFIRMED by Jay (2026-10-26, section 16).** Low-res method `tent` instead of box "area". Box + linear failed the 0.02-voxel gate (max 0.056-0.066 voxel); tent passes it for interior objects (>= 2 voxels from the edges) of width >= 12 (max 0.0181 at width 12, 0.0076 at width 20); edge-touching objects reach 0.0625 (section 3.1).
3. One level-4 run with a delayed deform start (3000 of 16000 steps, LR ramp 500) instead of the two-stage chain; the two-stage fallback needs no model code (r_level4.sh gates the deform-only flags on TASK_RATIO). The final length is decided by phase C (7.3), so 16000 is a starting point, not a fixed choice.
4. Not verified: login-node `python3` version (assumed >= 3.6); the m10 memory results (N / DEFORM_N placeholders 64 / 32 at level 4, 32 / 16 at level 2, 8 / 4 at level 1); step-time estimates for the sweep (~3.6 s mean at level 4); that validation at `--val-group-size 16` fits next to N_max training (it did at DEFORM_N 16 in m7). Verified by the review: `check_mixed_run.py` tolerates windows without deform keys (records are filtered by key presence), so no checker change is needed for the delayed start.
5. Skip semantics: Jay's decision of 2026-10-26 (section 4.1); "un-pre-warped" realised as identity pre-warp (difference < 1e-4, tested).

## 15. Revision notes

v1.1 (initial self-check, no review yet): (1) explicit rationale paragraphs for the three resolved decisions: AdamW on the current code (2.1, "resolved decision 1"), skip-level-4 semantics (4.1, "resolved decision 2"), one run vs two-stage level 4 (5.1, "resolved decision 3"); (2) Adam-slot transient for the delayed deform decoder recomputed (peak ~6.5x at k ≈ 10-20, not ~10x) and the ramp's effect quantified (<= 1.3x up to k 100, ~2.1x at the ramp end); (3) new level-1 skip-consistency WARNING + `main._skip_consistency` helper, config.json `cascade.skip_consistency_warnings`, test 11.3 g; (4) scripts: exact loss / schedule / balance flags (BAL_ARGS as m9, `--min-lr-factor 1e-3`, `--ckpt-every` 250 everywhere), r_level4 env DEFORM_DECODER / INIT validation, two-stage fallback checker mode `full` and a numeric "deform stuck" threshold; (5) `--task-start-ramp-steps` without a start > 0 is a parse error; task names must be in `--task-ratio`; (6) sweep wall-clock estimate (~60 h for 11 runs at 2 concurrent).

v1.2 (responses to design review v1.1, `design-review.md`):
- F1 MEDIUM (tent centroid gate fails at edges), resolved: test 11.2 c uses widths {12, 16, 20} at `np.linspace(2, n - w - 2, 5)` starts (worst 0.0181 < 0.02), plus an informational edge-touching bound 0.07; 3.1 and 14.2 now state the 0.0625-voxel edge-touching shift.
- F2 MEDIUM (incomplete checker call), resolved: section 6 specifies the full checker line with `--tasks "$TASKS" --steps "$STEPS" --expect-min-pairs "$MIN_PAIRS"`, TASKS derived from TASK_RATIO, BALANCE_CHECK only with deform + a linear task.
- F3 MEDIUM (fallback stage 1 fails parsing), resolved: DEFORM_ARGS gated on TASK_RATIO (`--gradient-checkpointing none` and no lean / pair-recompute flags without deform), fallback commands in the r_level4 header, new `BGR_DRY_RUN=1` script test 11.5 g incl. the linear-only case, section 12 step 2.
- F4 MEDIUM (iterations not tuned), resolved: phase C (winner at 12000 / 16000 with the final schedule), `length_decision` rule (shortest within 0.05 mm / 0.003 Dice of 16000; 20000 only if 16000 beats 12000 on both), phase-C 16000 run doubles as the final; wall clock ~78 h; tests 11.5 a / e. The old "best step >= 0.9 x STEPS" rule is removed.
- F5 MEDIUM (validation groups follow N), resolved: fixed `--val-group-size 16` for all level-4 runs via VAL_GROUP_SIZE (r_level4 default 16, helper passes it), summarizer records it and refuses to rank mixed values.
- NIT 6 resolved (5.1 rationale reworded; coincident change point at step 3000 documented, dip expected, env-only stagger fallback). NIT 7 resolved (global job index, `fmt`, full env list). NIT 8 resolved (refresh + events for the cascade and optimizer records on resume). NIT 9 resolved (any skip spec without `--cascade-from` is a parse error). NIT 10 resolved (status states; failed never resubmitted). NIT 11 resolved (v2 "decision made once" sentence replaced in the v3.16 edit). NIT 12 resolved (r_level1 header: compare smoke gpu_peak with m9's 36.7 GB, MIN_PAIRS=1 fallback above 39000 MB). NIT 13 resolved (`--a100-ok` required for dgx-a100).
- Still open for Jay (unchanged, not blocking implementation): the wd grid default {0, 1e-2, 1e-1} vs his quoted {0, 1e-5, 1e-4} (14.1).

v1.3 (responses to design review v1.2, `design-review.md`):
- F1 MEDIUM (memory-knob defaults unspecified), resolved in section 6 "Memory knobs": AMP and split group conv hard-coded; GRAD_CKPT default `full` (validated choices), PAIR_RECOMPUTE `true`, MIN_PAIRS `n`, LEAN_LOSSES `false` at levels 4 / 2 and `true` at level 1, so the defaults equal m10's `DK` knob set (+ lean losses = "L1 deform N5 lean"); GRAD_CKPT=none reproduces exactly the m10 "L4 deform no ckpt" record (no lean sampler / pair recompute / lean losses) and is allowed at level 4 when that record fits DEFORM_N; placeholder comments name the m10 record and knob set the orchestrator fills in; table rows list the knob defaults (r_level2 explicitly not m8's defaults); test 11.5 h covers all three scripts and GRAD_CKPT=none.
- F2 NIT resolved (7.3): "final" = the last record containing the key; never-logged exclusion keys -> not excluded, shown `?`.
- F3 NIT resolved (7.2): `job_env` drops `SCRIPT_ENV_KEYS` from the caller's environment before adding cfg_env; test 11.5 g (fake sbatch env dump, key-list drift check against r_level4.sh).
- F4 NIT resolved (6): `--curriculum $CURRICULUM` (validated), ramp flag only when true; dry-run case for fallback stage 2.
- F5 NIT resolved (7.2): `submit` refuses a dirty tree (exit 2), records `rev` per submission; summarizer `rev` column and mixed-rev WARNING; documented limitation that a pending job runs the checkout present at its start (no `git pull` while sweep jobs are pending).
- F6 NIT resolved (7.4, 11.5 f): substring check for `capture_output`, `text=True`, `import dataclasses`, `from dataclasses`, `fromisoformat`.
- Test letters in 11.5: new g (env isolation / revision), the script dry-run test is now h.

v1.4 (implementation of review v1.3, `design-review.md`; FEAT-001 and FEAT-005):
- F1 MEDIUM (fit rule), implemented in the `slurm/r_level4.sh` and `slurm/r_level2.sh` headers: `fits(record, partition) := record.peak_mb <= BUDGET[partition][level]`, BUDGET.rtx8000 = {4: 42000, 2: 41000}, BUDGET.dgx-a100 = 34000 at every level, unit MiB (`tests/gpu/measure_level_memory.py` writes `peak_mb` with MB = 2**20); derivation 46080 MiB − 500 context − DALI (~1000 at L4, ~2000 at L2 with TF aug) − 1024 headroom − ~1500 validation / fragmentation gap (m9). N / DEFORM_N = largest measured N that fits, from the record whose knob set equals the script defaults; `GRAD_CKPT=none` only if the no-ckpt record fits at that DEFORM_N; the smoke (>= 2 validations) must show `INFO gpu_peak` <= the same budget, else the next smaller N; growth (`TF_MEM_LIMIT_MB=0`) kept at levels 4 / 2, first fallback after a post-validation OOM `TF_MEM_LIMIT_MB=<budget>`. The section-6 "peak <= 46000 MB" wording is superseded by this rule (no script contains 46000).
- F2 NIT, corrected in place in 2.1: "0.4-4 % and 4-33 % (lr 1e-4 to 1e-3)".
- F3 NIT: r_level4 passes `--task-start-step` / `--task-start-ramp-steps` only when HAS_DEFORM && HAS_LINEAR && DEFORM_START > 0; a deform-only ratio prints `NOTE: deform-only ratio, DEFORM_START ignored` (it also drops curriculum and G, which the parser rejects without a linear task). Dry-run test asserts no `--task-start-step` / `--grad-balance-rule` for `TASK_RATIO=deform:1`.
- F4 NIT: `sweep_level4.py status` marks a config with a `r4sw_submissions.jsonl` record, no run dir and no squeue entry as **failed ("died before start")**; `submit` never resubmits it (tested).
- F5 NIT: r_level4 / r_level2 (and r_level1) headers: set `--time` from the smoke's `INFO step_time` projection x 1.3; if it exceeds the limit, resubmit with the same RUN_NAME (resumable). `--c-time` help: "derive from the phase-A INFO step_time".
- F6 NIT: `VAL_GROUP_SIZE = min(16, DEFORM_N_small)`, fixed for the whole sweep and recorded in every cfg_env (16 with the default grid; tested with `--deform-n 24` -> 12).
- F7 NIT, corrected in place in sections 1 and 2.3 (wording fixed again after the code review): fp32 = the Trainer's finite cond; AMP = the LossScaleOptimizer's own check (the Trainer gates only on input finiteness), which also skips the decay.

## 16. Addendum (2026-10-26): Jay's confirmations and scope additions

Binding for the remaining FEATs; where it conflicts with sections 6 or 14, this section wins.

1. **Weight-decay grid CONFIRMED by Jay:** AdamW sweep grid wd {0, 1e-2, 1e-1}, replacing his originally quoted {0, 1e-5, 1e-4} (section 2.1 / 14.1). The sweep helper's default grid stays as specified; `--wds` remains available but is not needed.
2. **Tent low-res CONFIRMED by Jay:** use `--lowres-method tent` (triangle filter, section 3) instead of box averaging. Measured by FEAT-002's test 11.2 c on the specified grid (widths 12/16/20, 5 interior starts, f 0.5..0.95 step 0.05, level-4 3-D and level-1 per-axis shapes): worst interior centroid shift **0.0167 voxel < 0.02** (edge-touching slab 0.0625 < 0.07; today's NN 1.55 voxel).
3. **New opt-in pair cap `--max-pairs K` (FEAT-007).**
   - Default 0 = no cap; with the default no new op is built, so goldens stay max |diff| 0.
   - Pair count k = min(K, max(floor, round(p N (N - 1)))) with floor from `--min-pairs` (1 or N), i.e. the cap is applied after the min-pairs rule (and k <= N (N - 1) as today). When the result is k == N with `--min-pairs n`, the N-cycle sampling stays.
   - Applies to the linear pairwise loss (`--p-pair`) and to the deform pairwise Dice + LNCC (`--p-pdice`; LNCC reuses the Dice pairs). `train_<task>/n_pairs` stays logged.
   - Reason: at p 0.1 the count grows as ~0.1 N^2 (N 32 -> 99, N 48 -> 226, N 64 -> 403 pairs), so large level-4 / level-2 groups make steps slow for diminishing returns.
   - The new from-scratch scripts (`slurm/r_level4.sh`, `r_level2.sh`, `r_level1.sh`) default `MAX_PAIRS=64` (the same as the validation cap `--val-deform-max-pairs 64`), validated as an integer >= 0 and passed as `--max-pairs $MAX_PAIRS`.
   - Tests: counts with and without the cap (static and tensor `num_pairs`), the cap together with `--min-pairs n` (incl. a cap that lands exactly on k == N -> cycle), CLI validation (negative / non-integer K rejected; K > 0 below the min-pairs floor: k = K wins, i.e. the cap is applied last, documented in the help).
4. **`--pair-recompute` is a level-1-only option.** It unrolls one serialized block per pair; in cluster job 10519650 a level-4 deform config with N 48 (226 pairs) sat in graph construction at 99 % CPU for ~2 h. The new scripts set `PAIR_RECOMPUTE=false` by default at levels 4 and 2 (overrides section 6's `true`); r_level1 keeps `true` (4 pairs). Consequence for the m10 placeholders: the level-4 / level-2 deform records to use are those measured without pair recompute (or the orchestrator re-measures); the header comment states this.
5. **Uncapped m10 memory facts** (job 10519893, RTX 8000, AMP, `slurm/m10_batch_memory.sh`):
   - level 4 deform (full ckpt): N 32 15.3 GB / 3.3 s (99 pairs), N 48 27.3 GB / 5.5 s (226 pairs), N 64 OOM (403 pairs); without checkpointing OOM already at N 32;
   - level 4 rigid / affine: N 128 35.3 GB / 1.86 s;
   - level 2 deform: N 16 29.9 GB / 10.5 s (24 pairs); N 20 and 24 aborted with exit -6 (likely the int32 per-tensor element limit);
   - level 2 linear: N 32 19.3 GB; N 48 aborted -6;
   - level 1 deform: N 5 aborted -6.
6. **Capped re-measurement (FEAT-007):** `tests/gpu/measure_level_memory.py` gets `--max-pairs K` (comma list; each config records `max_pairs` and the resulting `n_pairs`). New `slurm/m10_batch_memory_capped.sh` (modelled on `slurm/m10_batch_memory.sh`, commit aa0855f; account qtim only, 1 GPU, rtx8000, 3 CPUs, 64G, `--isolate true`, a timeout per step, output `scratch/m10/memory_batch_capped.json`) re-measures with `--max-pairs 64`: level 4 deform N 48 / 64 / 80 / 96 (full checkpointing, lean sampler, split group conv, NO pair recompute, LNCC 1/1, `--min-pairs n`, validation pass with `--val-pair-chunk 16`); level 4 linear N 128 / 160 / 192 after an int32 element check (stem tensor N x 40 x 56 x 48 x 32 = 3.44e6 N elements, 6.6e8 at N 192, well below 2^31 = 2.15e9; the check should still cover every per-image tensor wider than the stem, as the level-1 int32 failure came from a full-resolution tensor); level 2 deform N 16 (vs the uncapped 29.9 GB at 24 pairs); level 2 linear N 32 / 40. The orchestrator submits it after the workflow; the N placeholders of section 6 are then filled from the capped records.
7. **Implementation notes (FEAT-007, integration step):** `train_<task>/n_pairs` is logged whenever `--min-pairs n` or `--max-pairs` > 0 (so the capped count is visible with `--min-pairs 1` too; the default still adds no key). The int32 element check is `measure_level_memory.py --int32-check true`: the largest tensor of `models.memory_model.tensor_inventory` (every training-step block tensor incl. the full-resolution stem, and the frozen and validation forwards, at the measured N; the build at min(N, --build-n) is excluded because the model lists a level-1 N 2 build tensor above the limit that runs fine on the cluster) must be <= 2^31 - 1 elements, else a `limit` record ("int32 element check: ...") is written without running the configuration and the larger N are skipped. The memory-model prediction ignores the cap. `main.py` prints `NOTE: max-pairs K < N (...): the min-pairs n floor is capped` and a NOTE for `--pair-recompute true` at level 4 / 2.
