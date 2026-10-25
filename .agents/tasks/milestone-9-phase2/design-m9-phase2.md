# Milestone 9, Phase 2: technical design (3-level frozen cascade, level-1 training, minimum pair count)

Status: v3 (2026-10-24), revision after design review iteration 2 (`design-review.md`: 0 HIGH, 1 MEDIUM, 3 NIT, all resolved; responses at the end, iteration-1 responses kept below them). Inputs: Jay's messages 1-12 (workflow brief), Jay's decision on the width question (parent notification: no different-width trainable stage), `milestone-9/design-m9-phase1.md` v5 section 6 (3-level cascade, level-1 DALI / validation, retraining), `milestone-9/report.md`, `milestone-8/design.md` + report, `network-design/design.md` (on disk at **v3.14**, see section 13), the code at commit `0ddbf92`, and the cluster memory job 10516237 (`scratch/m9/memory_m9.json`, numbers transcribed in the brief).

Technology stack (locked, unchanged): Python 3.8 syntax, TF 2.10 / Keras 2.10, numpy 1.21, DALI 1.18 (existing pipelines, no new ops), unittest, bash + SLURM + `tf10_nv.sif`. No new dependencies.

## 0. Overview and decisions

Phase 2 trains the full-resolution stage (code `--level 1`, 160×224×192 at 1 mm) as the third stage of a cascade. Levels 4 and 2 are frozen and come from ONE checkpoint, the milestone-8 level-2 cascade run, which already stores its frozen level-4 stage. Every train and validation step pools the flipped, augmented level-1 group to level 2 and level 4, runs frozen level 4, pre-warps level 2, runs frozen level 2, composes, upsamples to level 1, pre-warps the level-1 images, and the trainable level-1 stage predicts a residual composed with the frozen chain inside the tape. That is exactly the frozen-chain stand-in that phase 1 measured (`tests/gpu/frozen_chain_standin.py`), now with real weights, checkpointing, provenance, resume, validation baselines and skip. Phase 2 also adds an opt-in minimum pair count for the pairwise losses (`--min-pairs n`), a level-1 SLURM script and the checker extensions.

| # | Decision | Choice and reason |
|---|---|---|
| D1 | Width of the trainable level-1 stage | **Identical ModelConfig to the frozen levels 4 and 2 (default widths). `check_cascade_config` stays strict.** Jay's decision (parent notification). No different-width, partial-copy or from-scratch path is implemented. The N = 5 route (fr50 + lean losses, measured 38.4 GB at N 5) needs a full retrain 4 → 2 → 1 with the new widths; it is documented in section 15 only. Consequence for scripts: the level-1 run must NOT pass `--filters` / `--decoder-widths` (m8 stored `decoder_widths: null`; an explicit `(128, 96, 64, 32, 16)` list is not dict-equal and would fail the check). |
| D2 | Level-2 source checkpoint | **`results/m8_level2_gradnorm/ckpt/best` (step 6000, linear criterion, mean pair 3.26 mm)** for both the frozen level-2 stage and the level-1 warm start; the frozen level-4 stage comes from the same checkpoint's `frozen_stage` object (the m7 step-11000 weights m8 was trained on). Reasons: (1) same rule as m8 Q1 (one checkpoint, one consistent shared encoder; two frozen level-2 models would double frozen compute and memory); (2) the run plateaued from ~6000, and the deform difference to `best_deform` (step 7000, 0.790) is ~0.001 pair Dice, below validation noise, while linear errors propagate into every level-1 pre-warp; (3) the frozen level-4 stage inside that checkpoint is by construction the one level 2 was trained against, so the 3-level chain is consistent. `CASCADE_FROM=.../ckpt/best_deform` remains a one-variable override. |
| D3 | How the chain reuses m8's stored frozen stage | `FrozenChain.from_checkpoint` restores `tf.train.Checkpoint(model=<level-2 model>, frozen_stage=<level-4 model>)` from the m8 prefix (`expect_partial`, `assert_existing_objects_matched`), then checks the restored level-4 weights against the m8 config.json `cascade.fingerprint` (n_vars exact, sums rel 1e-12). Mismatch is fatal. |
| D4 | Minimum pair count | One flag `--min-pairs {1,n}` (default `1` = today) for BOTH the linear pair loss (`--p-pair`) and the deform pair Dice + pair LNCC (`--p-pdice`). k = min(max(N, round(p·N(N−1))), N(N−1)). k == N: a random N-cycle (each image exactly once source, once target). k > N: today's top_k sampling. The RNG draw is identical to the default path (one `uniform([N(N−1)])` per call), so flips and later pairs follow the same stream. Section 5. |
| D5 | Recipe | Default widths, level-1 deform N 4, linear N 16, AMP, `--gradient-checkpointing full`, `--lean-sampler true`, `--split-group-conv true`, `--lean-losses false` (memory fallback), T 7, `--min-pairs n`, p 0.1 / 0.1, ratio rigid:1,affine:1,deform:2, Dice + LNCC 1/1 (window 9 = level-1 default), smoothness 0.5, `--task-grad-norm 12` (env `GRAD_BALANCE=gradnorm|deform64`), G 45° / 10 mm from step 0, curriculum off, Adam 3e-4, BN recalibration 64 at the start of a fresh run then frozen, skip 0, `--val-pair-chunk 4`. |
| D6 | Skip probability | **0 in the recipe, as in the milestone-8 runs** (`SKIP_PROB` default 0 in `slurm/m8_level2_cascade.sh`). Jay's messages do not set it, so this is a recipe choice of this design: it deviates from design.md 4.7 / phase-1 6.1 / 6.4 (0.15 for the full level), and the phase-2 report must list it as "to be confirmed with Jay". Skip is still implemented for the chain (it skips the LAST frozen stage, level 2) so that 0.15 can be tried later without code changes; with 0 no skip op is built. |
| D7 | GPU | **rtx8000 default** (48 GB RTX 8000 / A6000). A100 only if the smoke run's measured peak is ≤ 36 GiB (section 7). |
| D8 | Validation pairs | Unchanged: `--val-deform-max-pairs 64` (N 4 → all 12 ordered pairs); `--min-pairs` affects training only. |

Everything new is off by default. Level-4 and level-2 runs execute the same ops as today (goldens `m7_linear_golden.json`, `lncc_deform_golden.json` max |diff| 0); a milestone-8 level-2 cascade run builds the same graph, checkpoint keys and `cascade` / `init_weights` records as today. The only config.json change for every run (any level) is that `args` (and the PARAMETER record) gains `min_pairs: "1"`, because `args_to_dict` writes every argparse field.

## 1. Assumptions and unverified items

| ID | Item | Status / how it is checked |
|---|---|---|
| A1 | The m8 checkpoint `results/m8_level2_gradnorm/ckpt/best` holds `model` (level 2) and `frozen_stage` (level 4) with the default ModelConfig (decoder_widths null, T 7). | From the m8 code (`main.py` `ckpt_extra = dict(..., frozen_stage=...)`) and project notes; not read locally (cluster). `FrozenChain.from_checkpoint` checks the restore and the fingerprint; `check_cascade_config` checks the config. A failure exits before any training. |
| A2 | `best` is step 6000. | Project notes; recorded as `source_step` from the checkpoint itself, so a different value is visible in config.json, not assumed. |
| A3 | Extra memory of 4 instead of 1 deform pair at level 1 N 4. | Memory model (fitted coefficients, run for this design): default widths Fw 32.05 → 39.84 GiB, lean losses 36.15 GiB; f0d1 peak 34.92 → 39.23 GiB. The model over-predicts level-1 points (default N 4 predicted 45.0 GiB in D vs 28.9 GiB measured; f0d1 34.9 vs 27.7), so the scaled delta is about +4 to +7 GB: expected 35-38 GB on top of the measured 31.0 GB. Unverified until the smoke run (section 7). |
| A4 | Level-1 step time with 4 pairs. | Estimate: deform 15.6 → ~18 s (3 extra one-hot / image warps + LNCC + backward), linear ~4.5 s at default widths (3.9 / 3.7 s measured at f0d1). Mean at 1:1:2 ≈ (4.5 + 4.5 + 2·18) / 4 ≈ 11.3 s on RTX 8000. The smoke run measures it. |
| A5 | Level-1 validation time and peak. | Phase-1 estimate ~3 min per pass, first pass ×2 (identity and frozen caches); measured val peak 22.3 GB (stand-in, chunk 4). The run logs `val/wall_s` (section 3.4). |
| A6 | The frozen level-2 stage sees `pool(I_1)` instead of the DALI level-2 render it was trained on. | Same choice as m8 (Q3) one level up; the composed-vs-frozen baselines and the BN-gap lines show the effect. |

## 2. 3-level cascade (`models/cascade.py`)

### 2.1 Constants and the existing single stage

- `PREV_LEVEL = {2: 4, 1: 2}`. `FROZEN_LEVEL = 4` stays (it is the level of `FrozenStage`). The stand-in's `patched_prev_level()` becomes a no-op equivalent (it sets the same dict); keep it for the measurement tool.
- `FrozenStage` is unchanged in behaviour. New members (pure additions): `n_stages = 1`, `source_model` (= `self.model`, the model the trainable stage copies), `fingerprints` → `{4: self.fingerprint}`, `checkpoint_extra()` → `{"frozen_stage": self.model}`.

**Chain dispatch rule (resolves review finding 1).** Every Trainer code path that distinguishes a single frozen stage from a chain uses ONE helper, evaluated in Python at trace time (never a tensor):

```python
def _cascade_stages(self):
    """1 = single frozen stage (today's code verbatim), 2 = frozen chain; 0 = no cascade."""
    if self.cascade is None:
        return 0
    return int(getattr(self.cascade, "n_stages", 1))
```

It is computed once in `Trainer.__init__` and stored as `self._n_stages`; the train steps, `_cascade_val_fn`, `_cascade_val_deform_fn`, `_val_errors`, `_val_deform_errors` and `_frozen_baseline_metrics` branch on `self._n_stages == 2` only. `__init__` also checks: `self._n_stages in (0, 1, 2)`, and `self._n_stages == 2` requires `self.level == 1` (ValueError otherwise, alongside the existing `PREV_LEVEL` check). The `getattr` default keeps any object with only `level` / `linear` / `deform` (old fakes in tests) on the single-stage path. The chain interface the Trainer uses is exactly: `level`, `n_stages`, `linear`, `deform`, `linear_stages`, `deform_stages` (validation and skip > 0 only), `models` (frozen-variable test / recal invariant). `fingerprints`, `source_model`, `checkpoint_extra` and `from_*` are used only by `main.py`.

**Phase-1 stand-in (`tests/gpu/frozen_chain_standin.py`) under the new dispatch.** `FrozenChainStandIn` gets `n_stages = 2` and `linear_stages(images_l1, task)` / `deform_stages(images_l1)` implemented with the same formulas as `FrozenChain` (section 2.2): `linear_stages` returns `{"A": <today's linear result>, "A4": stop_gradient(A4)}`; `deform_stages` returns today's `deform` dict plus `U4 = up(p4["U"])`, `U4_inv = up(p4["U_inv"])`, `V4 = up(p4["V"])`, all stop_gradient. `linear` / `deform` are refactored to return the corresponding subset of `*_stages` only if that keeps the ops identical; otherwise they stay verbatim (recommended: keep verbatim, add the two methods next to them). So `measure_level_memory.py --level 1 --cascade true` (training and `--val-pass`) runs the same Trainer chain branches as a real level-1 run, including the `_frozen4` fields in the eval-mode validation forward, and records stay comparable to job 10516237 (`cascade_kind` stays `standin_chain_4_2`; no record-schema change). `patched_prev_level()` stays (now it sets the same dict as the module constant; kept so the tool and old scripts run unchanged). The measurement tool itself needs no change for the dispatch; it keeps building the stand-in. Rejected alternative: the tool builds `FrozenChain` from random models. That would also work, but it changes `cascade_kind` and the model-mapping table (`MODEL_CASCADE`) of existing records for no measurement benefit, since T-C2 proves both give identical outputs.

### 2.2 `FrozenChain`

```python
class FrozenChain:
    """Frozen level 4 -> level 2 chain in front of a trainable level-1 stage (inference only)."""
    level = 2          # the previous level of the trainable stage (PREV_LEVEL[1])
    n_stages = 2
    def __init__(self, stage4, model2)            # FrozenStage (level 4), float32 level-2 GroupRegLevelModel
    @classmethod from_checkpoint(cls, prefix, cfg, expected_fingerprint4)
    @classmethod from_config(cls, cfg)            # resume: random init, restored by RunCheckpoint
    @classmethod from_models(cls, model4, model2) # tests / tools: float32 copies via copy_weights_from
    models            -> [stage4.model, model2]
    source_model      -> model2
    fingerprints      -> {4: fp(stage4.model), 2: fp(model2)}
    fingerprint       -> fingerprints[2]          # m8 meaning: the stage directly before the trainable one
    checkpoint_extra()-> {"frozen_stage_4": stage4.model, "frozen_stage_2": model2}
    linear(images_l1, task)          -> A2 [N,4,4] float32, stop_gradient
    linear_stages(images_l1, task)   -> {"A": A2, "A4": A4}
    deform(images_l1)                -> {"U", "U_inv", "V"} level-1 voxels, stop_gradient
    deform_stages(images_l1)         -> deform(...) plus {"U4", "U4_inv", "V4"} (level 4 alone, upsampled twice)
```

Both models are constructed and built inside `_float32_policy()` with `gradient_checkpointing="none"`, `lean_sampler=False`, `split_group_conv=False`, `build_model(2)`, and are only called with `training=False` (BN statistics = the m8 values: level 2 recalibrated + frozen in m8, level 4 = m7). Validation: `stage4.level == 4`, `model2.level == 2`, identical configs (seed aside), else ValueError.

Math (x1 = level-1 group after flips / augmentation, float32; `pool` = `pool_to_coarser`; `up` = `diffeo.upsample_displacement`):

- Linear: x2 = pool(x1); A4 = stage4.linear(x2, task) (pools x2 to level 4 itself); x2' = prewarp_linear(x2, A4, 2 mm); out2 = model2(x2', task, training=False); A2 = compose_linear(out2.A_raw, A4, model2.centering) = Center₂(R2_raw · A4), i.e. exactly the level-2 run's composed output. The trainable step then does x1' = prewarp_linear(x1, A2, 1 mm) and A = compose_linear(R1_raw, A2, model.centering) = Center₁(R1_raw · A2) with ONE centering, the trainable model's (unchanged trainer code).
- Deform: p4 = stage4.deform(x2) (U4₂ = up(u4), U4₂⁻ = up(u4⁻), V4₂ = up(v4) at level 2); x2' = prewarp_deform(x2, U4₂); out2 = model2(x2', deform, training=False); (u2, u2⁻) = compose_cascade(U4₂, U4₂⁻, out2.u, out2.u_inv) at level 2; U = up(u2), U⁻ = up(u2⁻), V = up(out2.v). The trainable step does x1' = warp(x1, U) and (u, u⁻) = compose_cascade(U, U⁻, u1_res, u1_res⁻) (unchanged trainer code). `deform_stages` adds U4 = up(U4₂), U4⁻ = up(U4₂⁻), V4 = up(V4₂).
- Inverse, unrolled factor by factor: u⁻ = compose(exp(−v1), up(compose(exp(−v2), up(φ4⁻¹)))). Error = sum of stage errors + interpolation of the two upsamplings. The residual SVFs are group-mean centred per level by the models (unchanged), the linear residual by the one centering.
- The math is the stand-in's verbatim. Test T-C2 asserts FrozenChain and `FrozenChainStandIn` give identical outputs (max |diff| 0) for the same weights, so the phase-1 measurements apply.

`from_checkpoint(prefix, cfg, expected_fingerprint4)`: build both models, `tf.train.Checkpoint(model=model2, frozen_stage=model4).restore(prefix)`, `expect_partial()` (optimizer, step, RNGs are not read), `assert_existing_objects_matched()`. Then `fp4 = weight_fingerprint(model4)` compared with `expected_fingerprint4` (the m8 config.json `cascade.fingerprint`) with `main._check_cascade_fingerprint`'s rule moved to `runtime/checkpoint.fingerprints_match(saved, current, rtol=1e-12)`; mismatch → ValueError("…frozen level-4 weights in <prefix> do not match <config>.cascade.fingerprint…").

Exception contract (review iteration 2, NIT 4): `from_checkpoint` itself wraps the restore in `try: ... except (AssertionError, ValueError, tf.errors.NotFoundError, tf.errors.InvalidArgumentError) as e: raise ValueError("FrozenChain.from_checkpoint(%s): %s: %s" % (prefix, type(e).__name__, e)) from e`. So a missing `frozen_stage` / `model` object (TF raises AssertionError from `assert_existing_objects_matched`), a missing prefix (NotFoundError) and a shape mismatch (ValueError / InvalidArgumentError) all leave the function as ValueError. The fingerprint check runs after the `try` block and raises ValueError directly. `main.py` calls it inside the same `try / except ValueError as e: raise SystemExit("main.py: --cascade-from %s: %s" % (path, e))` pattern as `_load_cascade_source`. The Trainer and tests therefore only ever see ValueError from this function.

### 2.3 Skip (implemented, recipe 0)

`--cascade-skip-prob p` at level 1 skips the LAST frozen stage (level 2): A_prev = A4, U = U4, U⁻ = U4⁻ (design.md 4.7: a bypassed level passes on the output of the level before it; level 4 is never skipped). New helper `CS.skip_to(skip, coarse, fine)` = `tf.where(skip, coarse, fine)` under name scope `cascade_skip`, applied to A or to (U, U⁻). Only traced when p > 0 (then the step uses `linear_stages` / `deform_stages`); with p = 0 the chain step calls `linear` / `deform` and the graph has no `cascade_skip` ops (existing test pattern). The level-2 trainer path keeps `skip_linear` / `skip_fields` (identity / zero) unchanged.

## 3. Trainer (`runtime/run.py`)

### 3.1 Construction

- `Trainer(..., cascade=...)` accepts `FrozenStage` or `FrozenChain`. The existing check `CS.PREV_LEVEL.get(self.level) != cascade.level` is kept (chain.level = 2 = PREV_LEVEL[1]).
- `TrainConfig.min_pairs: str = "1"`; ValueError unless in `("1", "n")`. Passed to both postprocess constructors (section 5).

### 3.2 Training steps

- Linear: `if cascade is not None:` branch becomes: `self._n_stages == 1` (section 2.1 dispatch rule) → today's code verbatim. Chain: `A_prev = cascade.linear(x, task)` when not skip_on; with skip_on: `st = cascade.linear_stages(x, task); A_prev = CS.skip_to(skip, st["A4"], st["A"])`. Everything after (pre-warp at `LEVEL_SPACING[1]`, compose, losses) is unchanged.
- Deform: same pattern with `deform` / `deform_stages` and `skip_to` on U and U⁻.
- With `min_pairs == "n"` (trace-time switch) the step result gets `n_pairs = tf.size(batch["pair_i"])` (int32, cast to float for logging) for both linear and deform; `fit` window-averages it as `train_<task>/n_pairs` (key `PAIR_COUNT_KEY = "n_pairs"`, added to the accumulator only when present). With `"1"` nothing is added (logs unchanged).

### 3.3 Validation and frozen baselines

- `_cascade_val_fn` (linear), eval mode: chain → `st = cascade.linear_stages(x, task)`, returns `{"A": composed, "A_prev": st["A"], "A_prev4": st["A4"]}`; single stage unchanged. Train-mode returns `{"A"}` as today.
- `_cascade_val_deform_fn`, eval mode: chain → `deform_stages`, returns today's keys plus `U4, U4_inv, V4`. Train mode unchanged.
- `_val_errors` / `_val_deform_errors`: when collecting frozen outputs for a chain, append a dict (`{"A", "A4"}` resp. `{"U","U_inv","V","U4","U4_inv","V4"}`); single stage keeps today's array / dict.
- `_frozen_baseline_metrics` for a chain computes and caches (per set name and group count, as today) BOTH `val_<set>_frozen/...` = levels 4 + 2 composed (from A / U) and `val_<set>_frozen4/...` = level 4 alone (from A4 / U4), with the same groups, N, S, seeds and metric functions (linear `M.group_errors`, affine factor errors included; deform `_deform_errors` incl. LNCC, folding, inverse consistency). For level-2 runs nothing changes.
- Meaning of the frozen centrality keys (review NIT 2): `_frozen_baseline_metrics` passes the stored `V` as both `v` and `v_raw` to `M.deform_group_errors`. For the chain, `val_deform_frozen/{centrality_mm, raw_centrality_mm}` therefore describe the level-2 residual SVF only (V = up(v2), group-mean centred, so ≈ 0), not the composed field; `val_deform_frozen4/{centrality_mm, raw_centrality_mm}` use V4 = up(up(v4)), the full level-4 SVF (same meaning as m8's `_frozen`). No code change; the report and the checker's INFO lines label them this way.
- Host memory of the stored frozen outputs: for a chain, `_val_deform_errors` keeps the six level-1 fields `{U, U_inv, V, U4, U4_inv, V4}` as numpy for all groups until `_frozen_baseline_metrics` runs (first pass only; cached metrics afterwards, and the list is dropped when the function returns). No code change: the volume (≈ 20 GB) fits the job's host memory (section 6), and the smoke run measures it.
- `val_<set>/composed_mean_disp_mm` unchanged (composed level-1 u).
- Identity baselines (`*_identity`) unchanged.

### 3.4 Validation time

`_validate_and_log` measures the wall time of `validate()` and, only when `self.level == 1`, adds `val/wall_s` to the logged and stored record (level-4 / level-2 logs stay key-identical). Per-set `val_<set>/time_s` (eval pass only) is unchanged. BN-recal pre/post passes go through `_recal_validate` → `validate()` and are not timed separately (their `bnrecal_*/val_<set>/time_s` keys exist).

### 3.5 BN recalibration

Unchanged procedure (64 batches = 16 / 16 / 32, eager, tape-free, through `_recal_input`, which calls `cascade.linear` / `cascade.deform` and therefore works with the chain). Only the trainable level-1 model's statistics change; the frozen chain's are never touched (existing invariant, extended test T-R1).

## 4. Arguments and `main.py`

### 4.1 `runtime/args.py`

- `--cascade-from`: allowed with `--level 2` (level-4 source) or `--level 1` (level-2 cascade source); `--level 4` → `parser.error("--cascade-from needs --level 2 or --level 1 ...")`. Other rules unchanged (excludes `--init-weights`, needs `--deform-residual-affine false`). Help text updated.
- `--min-pairs {1,n}` (default `1`, choices enforced by argparse): "minimum sampled ordered pairs per training step for the pairwise losses (linear --p-pair and deform --p-pdice, incl. pair LNCC): 1 = today; n = the group size N (k = min(max(N, round(p N (N-1))), N (N-1)); k == N samples a random N-cycle so every image is once source and once target). Validation pairs are unaffected."
- New check: `--p-pair` must be in (0, 1] (`p_pdice` is already checked).
- `train_config_from_args` passes `min_pairs=args.min_pairs`. `args_to_dict` stores it in config.json automatically.

### 4.2 `runtime/checkpoint.py`

- `CASCADE_SOURCE_LEVEL = {2: 4, 1: 2}` (dict; replaces the int constant; a test asserts it equals `models.cascade.PREV_LEVEL`, checkpoint.py does not import models.cascade).
- `load_cascade_source(path, level=2)`: expected source level `CASCADE_SOURCE_LEVEL[level]`. Level 2: exactly today's checks and info. Level 1: additionally the source config.json must hold `cascade` with `source_level == 4` and a `fingerprint` (n_vars > 0), else ValueError "--cascade-from <path>: the level-1 cascade needs a level-2 CASCADE run (config.json cascade record with source_level 4); <cfg> has none". info adds `upstream_cascade` = that record verbatim.
- `fingerprints_match(saved, current, rtol=1e-12)` (moved from `main._check_cascade_fingerprint`'s `close` rule, which now calls it).
- `check_cascade_config` and `cascade_init_record` unchanged (the record gets `source_level` 2, `cross_level` [2, 1]).

### 4.3 `main.py`

Fresh vs resume is still decided once before any model is built.

Fresh, level 1:
1. `_load_cascade_source(C, path, model_cfg, level)` → prefix, src_cfg, info (strict config equality, SystemExit on any ValueError).
2. `frozen = FrozenChain.from_checkpoint(prefix, src_cfg, info["upstream_cascade"]["fingerprint"])`; `except ValueError as e: raise SystemExit("main.py: --cascade-from %s: %s" % (path, e))` (from_checkpoint already converts TF AssertionError / NotFoundError / InvalidArgumentError to ValueError, section 2.2).
3. `model.copy_weights_from(frozen.source_model)`; `init_info = C.cascade_init_record(model, info)`.
4. Cascade record (level 1 only; level-2 records unchanged):

```json
{"source_checkpoint": "<m8 prefix>", "source_config": "<m8 config.json>", "source_level": 2,
 "source_step": 6000, "fingerprint": {"n_vars": ..., "abs_sum": ..., "sq_sum": ...},
 "S_rel_diff_vs_level4": 0.0x, "S_rel_diff_vs_level2": 0.0x, "skip_prob": 0.0, "skip_stage": 2,
 "stages": [
   {"level": 4, "checkpoint_key": "frozen_stage", "fingerprint": {...},
    "origin_checkpoint": "<m7 prefix from the m8 record>", "origin_step": 11000},
   {"level": 2, "checkpoint_key": "model", "fingerprint": {...},
    "origin_checkpoint": "<m8 prefix>", "origin_step": 6000}]}
```

`fingerprint` = the level-2 stage (m8 meaning: the stage before the trainable one). The S keys: ||S1 − S_k||_F / ||S_k||_F from `template_mask_{4,2}.npy` on real data, null in overfit mode. `_cascade_lines` prints one line per stage when `stages` is present.

5. Checkpoint extras: `ckpt_extra = dict(ckpt_extra, **frozen.checkpoint_extra())` → level 2 `frozen_stage` (unchanged), level 1 `frozen_stage_4`, `frozen_stage_2`.

Resume, level 1: `saved_cfg = _saved_cascade_config(...)` (must have `cascade` with `stages`, else RuntimeError "resume of a level-1 cascade run without config.json cascade.stages"); `frozen = FrozenChain.from_config(model_cfg)`; restore through `RunCheckpoint`; `_check_cascade_fingerprints(rec, frozen, path)`: for each stage `fingerprints_match(stage["fingerprint"], frozen.fingerprints[stage["level"]])`, plus the top-level one → RuntimeError naming the level on mismatch. BN freeze from the saved record, provenance carried verbatim, source path not read (all m8 rules). Level-2 resume path unchanged (`stages` absent → today's single check).

Build N: `_build_n(args)` unchanged (split on → 2).

## 5. Minimum pair count (`data_loading/postprocess.py`)

```python
MIN_PAIRS = ("1", "n")

def num_pairs(n, p_pair=0.1, min_pairs="1"):
    """int32 tensor: min(max(m, round(p n (n - 1))), n (n - 1)), m = 1 ('1') or n ('n')."""
    n = tf.cast(n, tf.int32)
    total = n * (n - 1)
    k = tf.cast(tf.round(tf.cast(total, tf.float32) * float(p_pair)), tf.int32)
    floor = 1 if min_pairs == "1" else n        # Python switch: '1' builds today's ops exactly
    return tf.minimum(tf.maximum(k, floor), total)

def sample_pairs(generator, n, p_pair=0.1, min_pairs="1"):
    n = tf.cast(n, tf.int32)
    total = n * (n - 1)
    keys = generator.uniform(tf.reshape(total, [1]), dtype=tf.float32)   # ONE draw, as today
    k = num_pairs(n, p_pair, min_pairs)
    def topk():                                  # today's code
        _, idx = tf.math.top_k(keys, k=k, sorted=True)
        i = idx // (n - 1); r = idx % (n - 1)
        return i, r + tf.cast(r >= i, tf.int32)
    if min_pairs == "1":
        return topk()                            # identical ops and RNG use (goldens)
    def cycle():                                 # random N-cycle from the first n keys
        _, perm = tf.math.top_k(keys[:n], k=n, sorted=True)
        return perm, tf.roll(perm, shift=-1, axis=0)
    with tf.name_scope("min_pairs"):             # marker scope for T-G1
        return tf.cond(tf.equal(k, n), cycle, topk)
```

- `min_pairs` is validated in both postprocess constructors (`ValueError` unless in `MIN_PAIRS`) and stored as `self.min_pairs`; `LinearPostprocess(..., min_pairs="1")`, `DeformPostprocess(..., min_pairs="1")` pass it to `sample_pairs`. The deform pair LNCC reuses the same pairs (unchanged loss code); the linear pair targets `P_star` use the returned pairs (unchanged).
- Properties: deterministic for a given generator state; the generator advances by exactly one `uniform([N(N−1)])` per call in both modes, so with the flag on the flip / pair stream of later steps is the same as with it off (checkpointed RNG state stays compatible). No stateful op inside `tf.cond`. Graph mode: `tf.cond` with output shapes [n] / [k] gives `[None]`; every consumer uses `tf.gather` / `tf.reshape(.., [-1])` (verified in `runtime/losses.py`). `pair_i != pair_j` always (distinct permutation entries, n ≥ 2). Losses average over pairs (`reduce_mean`), so more pairs lower variance without reweighting the pair terms.
- A random N-cycle is a derangement but not uniform over all derangements; chosen because it guarantees the stated property (every image once as source and once as target) and links the whole group in one cycle. Uniform derangements (rejection sampling) are rejected: data-dependent loop length and RNG use.
- Effect at p = 0.1 (k with `1` → with `n`): N 2: 1 → 2 (both pairs); 3: 1 → 3; 4: 1 → 4 (level-1 deform); 5: 2 → 5; 6: 3 → 6; 8: 6 → 8 (level-2 deform); 10: 9 → 10; 11: 11 → 11 (count unchanged but k == N, so the cycle replaces top_k); ≥ 12: unchanged (16: 24, 32: 99). So level-4 N 16 / 32 and every linear N 16 / 32 group is unchanged.
- `models/memory_model.num_pairs(n, p, min_pairs="1")` mirrors it (Python, numpy float32 product, half-to-even); `Workload` gains `min_pairs="1"` (appended to `FIELDS_`; records without it read `"1"`).

## 6. Level-1 data, training and validation paths

- DALI: unchanged `LinearDataModule` / `DeformPipeline` at level 1 (`image.npy` 27.5 MB, `labels_train18.npy` 6.9 MB per case; DALI level-1 checks b/c/d/e/l passed in job 10516237). `--num-threads ${SLURM_CPUS_PER_TASK:-3}`, `--prefetch ${PREFETCH:-1}`. Rule (phase-1 6.2): if the smoke checker's data-wait fraction of a linear task is > 0.10, run the long job with `PREFETCH=2` (+440 MB host).
- Host memory: validation volumes ≈ 4 linear sets × 48 images × 27.5 MB ≈ 5.3 GB + deform 40 images + labels ≈ 1.4 GB + DALI buffers. During the FIRST validation pass (inside BN recal, `bnrecal_pre`) the chain's stored frozen deform outputs add 6 fields × 4 images × 160·224·192·3 × 4 B ≈ 2.0 GB per group × 10 groups ≈ 20 GB (section 3.3), plus the linear frozen A matrices (negligible). Peak host ≈ 30-35 GB with the TF runtime / DALI → `--mem 64G`. Rule: the report reads the smoke job's `sacct -j <id> --format=MaxRSS`; if MaxRSS > 56 GB, the long run uses `--mem 96G` (within the ~128 GB fair share).
- Validation sets: linear `nv` = N (16 → 3 groups of the 39 T1 cases, the last wrapped by the existing rule), deform `nvd` = DEFORM_N (4 → 10 groups, all 12 ordered pairs, `--val-deform-max-pairs 64`). `--val-group-size` is not passed (it would override both). Sets: rigid, affine, rigid_g, affine_g, deform.
- `--val-pair-chunk 4` (measured validation peak 22.3 GB at chunk 4; chunking does not change metrics).
- Validation outputs per pass (all composed at level 1 unless noted): `val_<lin>/{center,pair}_rms_mm`, `..._rot_deg`, `..._trans_mm`, affine sets `{center,pair}_{scale_pct,shear_pct,qr_rot_deg}`; `val/mean_pair_rms_mm`; `val_deform/{template_dice, pair_dice, neg_jac_pct, inv_consistency_mm, centrality_mm, ncc_template, ncc_pair, composed_mean_disp_mm, pair_dice_loss}`; identity baselines `val_deform/*_identity`; train-mode keys; frozen baselines `val_<set>_frozen/...` (levels 4 + 2) and `val_<set>_frozen4/...` (level 4 alone) for every set; `val/wall_s`.
- Cost (estimate, A5): per pass ≈ deform 10 × (eval + train-mode forward ~8 s + 12 pairs ~12 s) ≈ 3.5 min + linear 4 × 3 × ~4 s ≈ 1 min → ~5 min; the first pass adds identity + two frozen baselines ≈ +6 min. 16 validations ≈ 1.5 h of a long run.

## 7. Memory

- Measured (job 10516237): default widths, stand-in chain, AMP, full, lean + split, T 7, 1 pair: N 4 31.0 GB, 15.6 s; validation 22.3 GB.
- Predicted for 4 pairs (A3): +4 to +7 GB → 35-38 GB. RTX 8000 / A6000 48 GB: fits with margin. A100 40 GB (36 GiB budget): marginal / likely over.
- Rule encoded in the script header and the checker: the smoke checker prints `INFO gpu_peak <max gpu_peak_mb> MiB (<GiB>) vs A100 budget 36 GiB: fits | exceeds -> keep rtx8000 or --lean-losses true` (`--memory-budget-mb 36864`). If the RTX 8000 smoke itself OOMs (peak > ~46 GiB): rerun the smoke with `LEAN_LOSSES=true` (model: −3.7 GiB in Fw); if that still fails, `MIN_PAIRS=1` and report to Jay.
- `gpu_peak_mb` is the process-lifetime allocator peak (includes BN recal and validation), the number that decides "fits".
- `tests/gpu/measure_level_memory.py`: new axis `--min-pairs` (comma list of `1,n`, default `1`), in `KEY_FIELDS` (old records read `"1"`), passed to `TrainConfig(min_pairs=...)` and to the isolate child argv, and to the predicted `Workload`. Optional pre-smoke measurement (not required, Jay's approval):
  `python tests/gpu/measure_level_memory.py --level 1 --tasks deform --n 4 --widths default --cascade true --gradient-checkpointing full --amp true --lean-sampler true --split-group-conv true --lean-losses false,true --min-pairs 1,n --lambda-ncc-template 1 --lambda-ncc-pair 1 --val-pass true --val-pair-chunk 4 --isolate true --out scratch/m9/memory_m9_pairs.json`

## 8. `slurm/m9_level1_cascade.sh`

Template: `slurm/m8_level2_cascade.sh`. Header:

```bash
#SBATCH --job-name=bgr-m9-l1
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
```

Env (defaults): `PROJ`, `REPO`, `DATA`, `RESULTS`, `SIF=/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif`, `SCRATCH_ROOT=/scratch`, `RUN_NAME=m9_level1_cascade_$JOB_ID`, `STEPS=8000`, `VAL_EVERY=500`, `CKPT_EVERY=250`, `WARMUP=500`, `N=16`, `DEFORM_N=4` (measured; integer-checked, exit 2), `CASCADE_FROM=$RESULTS/m8_level2_gradnorm/ckpt/best`, `GRAD_BALANCE=gradnorm` (`deform64` → `--task-loss-weights deform:64`), `BN_RECAL_BATCHES=64`, `BN_RECAL_FREEZE=true`, `SKIP_PROB=0`, `MIN_PAIRS=n`, `LEAN_LOSSES=false`, `PREFETCH=1`, `EXTRA_ARGS=""`. The `CASCADE_FROM` guard is skipped when `$RESULTS/$RUN_NAME/ckpt/last/checkpoint` exists (m8 rule). `TMPDIR=$SCRATCH_ROOT/$JOB_ID` and the SINGULARITYENV exports as in m8.

Training call (one line per group of flags):

```bash
singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF" \
  python main.py --data "$DATA" --results "$RESULTS" --run-name "$RUN_NAME" \
  --level 1 --cascade-from "$CASCADE_FROM" --group-size "$N" --deform-group-size "$DEFORM_N" \
  --task-ratio rigid:1,affine:1,deform:2 --steps "$STEPS" --val-every "$VAL_EVERY" \
  --ckpt-every "$CKPT_EVERY" --warmup-steps "$WARMUP" \
  $BAL_ARGS --lambda-ncc-template 1 --lambda-ncc-pair 1 --lambda-smooth 0.5 \
  --p-pair 0.1 --p-pdice 0.1 --min-pairs "$MIN_PAIRS" \
  --bn-recal-batches "$BN_RECAL_BATCHES" --bn-recal-freeze "$BN_RECAL_FREEZE" --cascade-skip-prob "$SKIP_PROB" \
  --amp true --gradient-checkpointing full --lean-sampler true --split-group-conv true \
  --lean-losses "$LEAN_LOSSES" --val-pair-chunk 4 \
  --global-transform true --global-rot-max-deg 45 --global-trans-max-mm 10 \
  --global-start-step 0 --global-ramp-steps 0 --curriculum false --val-png true \
  --num-threads "${SLURM_CPUS_PER_TASK:-3}" --prefetch "$PREFETCH" --log-every 25 --log-backend jsonl \
  $EXTRA_ARGS </dev/null
```

No `--filters` / `--decoder-widths` / `--integration-steps` (D1: the defaults equal m8's stored config). Then, as in m8, the checker (exit code = main's if non-zero, else the checker's):

```bash
singularity exec -B /autofs -B /cluster -B /space "$SIF" \
  python tests/gpu/check_mixed_run.py "$RESULTS/$RUN_NAME" --tasks rigid,affine,deform --steps "$STEPS" \
  --expect-init-weights full --expect-enc-grad true --expect-cascade true --expect-cascade-stages 2 \
  --grad-balance-rule true --expect-bn-recal true (iff BN_RECAL_BATCHES > 0) --max-nonfinite-frac 0.06 \
  --expect-min-pairs "$MIN_PAIRS" --memory-budget-mb 36864 --project-steps 8000 </dev/null
```

Header comment also states: measured facts (section 7), the A100 caveat ("-p dgx-a100 only if the smoke peak ≤ 36 GiB; same 3 CPUs / 64G"), the grad-balance KEEP / FALLBACK rule from m8, the prefetch rule, the time formulas below, "DO NOT submit without Jay's approval; account qtim only", `mkdir -p .../logs`, and the cluster order (pull → smoke → long).

Time:
- Smoke: 200 steps, `VAL_EVERY=100`, `WARMUP=50`. T = 1.5 × (200 × 11.3 s + recal batches 660 s (phase-1 6.1 upper end of 5-11 min; m8 measured 515 s at level 2) + recal pre pass (first pass, identity + frozen caches) 660 s + recal post pass 300 s + 2 regular passes × 300 s + startup / trace / val-set loading 900 s) ≈ 1.5 × 5380 s ≈ 2.25 h → `--time 04:00:00` (first level-1 run; generous for unknowns).
  Accounting rule (review NIT 4): the recal event's `elapsed_s` INCLUDES its two `_recal_validate` passes (`t_start` is taken before `bnrecal_pre`, run.py `recalibrate_bn`), and those passes do not go through `_validate_and_log`, so they never produce `val/wall_s`. The projection therefore adds `recal elapsed_s` once and the `val/wall_s` passes separately, with no double counting.
  `RUN_NAME=m9_l1_smoke STEPS=200 VAL_EVERY=100 WARMUP=50 sbatch --job-name=m9smoke --time=04:00:00 --mem=64G slurm/m9_level1_cascade.sh`
- Long: T = 1.3 × (8000 × s_mean + 16 × s_val + s_recal) from the smoke run (checker `--project-steps 8000` prints it). Provisional: 1.3 × (8000 × 11.3 + 16 × 300 + 1000) ≈ 1.3 × 96 200 s ≈ 35 h → `--time 48:00:00`.
  `RUN_NAME=m9_level1_cascade sbatch --job-name=m9long --time=48:00:00 --mem=64G slurm/m9_level1_cascade.sh`
  A100 (only if the smoke peak ≤ 36 GiB; ~1.5× faster, ~24 h): `RUN_NAME=m9_level1_cascade sbatch -p dgx-a100 --job-name=m9long --time=36:00:00 --mem=64G slurm/m9_level1_cascade.sh`
- Resume plan: checkpoints every 250 steps (~47 min at 11.3 s). If the job hits `--time`, fails, or is preempted, resubmit the identical line with the same `RUN_NAME` (Jay's approval; ≤ 2 active jobs): `--resume` (default true) restores `ckpt/last`, both frozen stages and the RNGs, checks both fingerprints, re-applies the saved BN freeze and never repeats the recalibration; the cosine schedule continues from the restored step; at most 250 steps are lost. The 7-day partition maximum is far above the estimate, so a single job is the plan.

## 9. `tests/gpu/check_mixed_run.py` extensions (stdlib only)

All new options default off; existing directories check exactly as before.

- `--expect-cascade-stages {1,2}` (default 1 = m8 checks unchanged). With 2 (implies the cascade checks): `cascade.source_level == 2`; `cascade.stages` is a list with levels [4, 2], each `fingerprint.n_vars` an int > 0 and `origin_step` an int; `init_weights.via == "cascade"` and `cross_level == [2, 1]`; final-step `val_<task>_frozen4/{center,pair}_rms_mm` per linear task and `val_deform_frozen4/{template_dice,pair_dice}` finite (in addition to today's `_frozen` keys). INFO lines: `cascade_gain <set>` against `_frozen` (4 + 2) AND `_frozen4`; `g_gap <task>` for composed, `_frozen` and `_frozen4` (incl. affine scale / shear / QR-rotation; G-gap tracking vs the level-2 affine pair gap +0.92 mm); deform `INFO deform_extra`: composed / frozen / frozen4 / identity for neg_jac_pct, inv_consistency_mm, ncc_template, ncc_pair. `_frozen_sets` keeps matching only `_frozen` heads; a new `_frozen4_sets` matches `_frozen4`. The level-1 `S_rel_diff_vs_level2` is printed when present.
- `--expect-min-pairs {none,1,n}` (default none): config.json `args.min_pairs` equals the value; with `n`, every logged `train_deform/n_pairs` equals k(N_deform, p_pdice) and `train_<lin>/n_pairs` equals k(N, p_pair). Exact stdlib mirror (review NIT 7):

```python
def _f32(x):
    return struct.unpack("f", struct.pack("f", float(x)))[0]

def expected_pairs(m_floor, n, p):
    total = n * (n - 1)
    k = int(round(_f32(_f32(total) * _f32(p))))   # Python 3 round = half-to-even, as tf.round
    return min(max(m_floor, k), total)
```

with `m_floor = n` for `"n"` (1 for `"1"`), `N = args.group_size`, `N_deform = args.deform_group_size or args.group_size` (config.json stores `null` when `--deform-group-size` is unset), `p = args.p_pair` (linear) / `args.p_pdice` (deform). T-P1 asserts this function equals `postprocess.num_pairs` and `memory_model.num_pairs` for N 2-40, p ∈ {0.05, 0.1, 0.25, 1.0}.
- `--memory-budget-mb M` (default 0 = off): INFO `gpu_peak` line, max `gpu_peak_mb` vs M (never a failure).
- `--project-steps S` (default 0 = off): INFO `step_time` line: mean `train_<task>/compute_ms` + `data_ms` per task, the ratio-weighted mean step (ratio from `args.task_ratio_parsed`), the median `val/wall_s` (and the first one), and the projected wall time 1.3 × (S × mean step + (S / val_every) × median `val/wall_s` + recal `elapsed_s`), where `elapsed_s` (from the `bn_recal` event) already contains the pre / post validation passes (section 8 accounting rule); missing keys print `n/a` (INFO only).
- Cosmetic m8 NIT fixed: the grad-balance INFO label reads `KEEP current balance (deform64)` when config.json `args.task_grad_norm == 0`, else `KEEP task-grad-norm`.

## 10. Error handling

| Operation | Failure | Recoverable? | Caller receives | Logged |
|---|---|---|---|---|
| parse `--cascade-from` with `--level 4` | invalid | fatal | `parser.error` (exit 2) | stderr |
| parse `--min-pairs` other than 1 / n | invalid | fatal | argparse choices error (exit 2) | stderr |
| parse `--p-pair` outside (0, 1] | invalid | fatal | `parser.error` | stderr |
| `load_cascade_source(path, 1)` | path unresolvable / no config.json / source not level 2 / no `cascade` record with source_level 4 | fatal | ValueError → `SystemExit("main.py: --cascade-from ...")` exit 1 | stderr |
| `check_cascade_config` | any ModelConfig field differs (e.g. explicit `--decoder-widths`) | fatal | SystemExit naming the fields | stderr |
| `FrozenChain.from_checkpoint` restore | missing `model` / `frozen_stage` objects (TF AssertionError), missing prefix (NotFoundError), shape mismatch (ValueError / InvalidArgumentError) | fatal | wrapped by `from_checkpoint` into ValueError naming the prefix and the original type → main: `SystemExit("main.py: --cascade-from <path>: ...")`, exit 1 | stderr |
| same, level-4 fingerprint vs m8 record | mismatch | fatal | ValueError → SystemExit | stderr |
| `FrozenChain.__init__` | wrong levels / configs differ | fatal | ValueError | — |
| `Trainer.__init__` | cascade level not PREV_LEVEL[level]; `min_pairs` invalid | fatal | ValueError | — |
| Postprocess constructors | `min_pairs` invalid | fatal | ValueError | — |
| Resume of a level-1 cascade | config.json lacks `cascade.stages`; no checkpoint restored | fatal | RuntimeError | traceback |
| Resume fingerprint check | any stage differs | fatal | RuntimeError naming the level | traceback |
| Level-1 training step | OOM | fatal for the job | ResourceExhaustedError, job fails | SLURM log; remedy section 7 |
| Non-finite step (AMP overflow) | inf grads | recoverable | step skipped (existing), counter; checker allows 6 % | `nonfinite_step_skipped` events |
| Frozen baseline | frozen outputs missing for a group | fatal (programming error) | ValueError (existing) | — |
| Checker | missing keys / records | test outcome | FAIL line, exit 1; INFO lines never fail | stdout |
| SLURM script | N / DEFORM_N not integers; CASCADE_FROM missing on a fresh run; GRAD_BALANCE / MIN_PAIRS invalid | fatal | exit 2 with message | stderr |

Invariants and owners: identical configs across the three levels → `check_cascade_config` + `copy_weights_from` + `FrozenChain.__init__`; frozen weights never trained / recalibrated → frozen models outside `model.trainable_variables`, called only with `training=False` (FrozenChain), `recalibrate_bn` touches only `self.model`; chain consistency with m8 → `from_checkpoint` fingerprint check; resume integrity → `_check_cascade_fingerprints` in main; default-path bit-identity → Python-level switches (`min_pairs == "1"`, `n_stages == 1`, `skip_on`), enforced by the goldens.

## 11. Tests

Unit / TF (tiny widths, `tests.tf_setup` first):
- `tests/test_postprocess.py`: T-P1 `num_pairs` with `"1"` equals today's values (existing table) and with `"n"` the table of section 5 for N 2-40; mirror equality with `memory_model.num_pairs` and the checker's mirror. T-P2 `"n"`, k == N (N 2, 3, 4, 8, 11): pairs form one N-cycle (sorted i == sorted j == range(N), i != j, the successor graph is a single cycle); eager and inside `tf.function` with `input_signature` (pair shape [None]). T-P3 k > N (N 12, 16): equals the default top_k result for the same seed. T-P4 RNG stream: after `sample_pairs` with `"1"` and `"n"` from equal seeds the generator states are equal. T-P5 determinism: two generators with one seed give identical pairs. T-P6 both postprocessors pass `min_pairs` (deform N 4 → 4 pairs, linear N 4 → 4) and reject invalid values.
- Goldens: `tests/test_trainer.py` `TestLinearOnlyUnchanged`, `TestDeformUnchanged` unchanged, max |diff| 0. New T-G1 (review iteration 2, NIT 2): trace the postprocess `tf.function` and one tiny linear and deform train step with the default `TrainConfig` and with `min_pairs="n"`. The default graphs contain no op whose name contains `min_pairs/` (the `"n"` branch of `sample_pairs` is built under `tf.name_scope("min_pairs")`, section 5; other conds in the step, e.g. AMP loss scaling, are not counted) and the step results have no `n_pairs` key. The `"n"` graphs contain `min_pairs/` ops and the `n_pairs` output, and the postprocess graph's op count is strictly larger. Bit-identity of the default path stays covered by the unchanged goldens, so T-G1 does not compare the default with itself.
- `tests/test_cascade.py`: T-C1 `PREV_LEVEL == {2: 4, 1: 2} == checkpoint.CASCADE_SOURCE_LEVEL`. T-C2 FrozenChain vs `FrozenChainStandIn` same weights → max |diff| 0 for `linear` A, `linear_stages` A / A4, `deform` U / U_inv / V and `deform_stages` U4 / U4_inv / V4 (level-1 shape, N 2, tiny). Runs under `tests.test_trainer._OpDeterminism` (op determinism on, restored afterwards; review NIT 5), so GPU kernels are bit-reproducible; both objects are called in the same order on the same input tensor. T-C3 `linear_stages["A4"]` equals `FrozenStage.linear(pool(x))` and `["A"]` equals `linear`; `deform_stages` U4 = up(up(u4)). T-C4 three-level round trip on synthetic smooth fields at the real level shapes (N 1): interior |compose(u, u_inv)| below margin ceil(max |u|) + 3 voxels as `test_two_level_round_trip`. T-C5 `from_checkpoint`: write a tiny m8-style checkpoint (`tf.train.Checkpoint(model=m2, frozen_stage=m4)` + config.json with a `cascade` record), restore equals the source weights; corrupted fingerprint → ValueError; checkpoint written with only `model=m2` (no `frozen_stage`) → ValueError whose message contains "AssertionError" (TF's `assert_existing_objects_matched` error, wrapped); non-existent prefix → ValueError containing "NotFoundError". A `test_main.py` parser-level case asserts main turns that ValueError into SystemExit starting with "main.py: --cascade-from". T-C6 skip: p 0 graph has no `cascade_skip` ops; p > 0 with skip True gives A_prev = A4 and U = U4 (level-1 trainer, tiny, one step). T-C7 frozen variables unchanged after training steps and after `recalibrate_bn` (chain). T-C9 dispatch rule: `Trainer` with `cascade=None` → `_n_stages 0`; `FrozenStage` → 1; `FrozenChain` and `FrozenChainStandIn` → 2; a fake object with only `level` / `linear` / `deform` → 1 (getattr default); a chain object at `level 2` → ValueError.
- `tests/test_measure_memory.py`: T-C8 `measure_level_memory.measure()` at level 1, `--cascade true --val-pass true --tasks deform`, TINY widths, N 2, `--val-pair-chunk 4` → training `status` and `val_status` both `ok` (or `oom` with a clear record on a small GPU; never `error`), and the record's `cascade_kind == "standin_chain_4_2"`. The existing level-1 `--cascade true` test (line ~95) stays green unchanged.
- `tests/test_cascade.py` validation: T-V1 a level-1 chain trainer's `validate(max_groups=1)` has finite `val_<set>_frozen` and `_frozen4` keys and `val/wall_s` via `_validate_and_log`; level-2 trainer output keys unchanged (key-set equality with today's list).
- `tests/test_runtime_infra.py` / `tests/test_main.py` parser: `--cascade-from` with levels 4 / 2 / 1; `--min-pairs` choices and default; `--p-pair` range; `load_cascade_source(.., 1)` errors (level-4 source, level-2 source without cascade record); `_check_cascade_fingerprints` per stage (pure function).
- `tests/test_memory_model.py`: `Workload(min_pairs="n")` at level 1 N 4 predicts a larger Fw than `"1"`; old records without the field load as `"1"`.
- `tests/test_measure_memory.py`: `--min-pairs` axis in `KEY_FIELDS`, old records keyed `"1"`, isolate argv carries it.
- `tests/test_check_mixed_run.py`: fake run dirs for `--expect-cascade-stages 2` (pass, missing `_frozen4` → FAIL, wrong `stages` → FAIL), `--expect-min-pairs n` (pass / wrong count FAIL), the `gpu_peak` / `step_time` INFO lines, the deform64 label fix; existing m8 fixtures unchanged.
- `tests/test_slurm_scripts.py`: `m9_level1_cascade.sh` in `SCRIPTS` (header: `-A qtim`, no `entre`, `--gpus=1`, `-N 1`, `--ntasks-per-node=1`, `--cpus-per-task=3`, logs path, `mkdir -p` line, ASCII, `</dev/null` on both container calls); stub runs: default recipe argv contains `--level 1`, `--min-pairs n`, `--gradient-checkpointing full`, `--val-pair-chunk 4`, no `--filters` / `--decoder-widths`; `GRAD_BALANCE=deform64`; resume skips the guard; invalid `MIN_PAIRS` → exit 2.

Integration (local GPU, GTX 1080, `--gpu-memory-limit-mb 6000`):
- T-E1 `tests/test_main.py::TestLevel1CascadeMain` (pattern of `TestCascadeMain`). All three runs share: `--overfit-one-group --overfit-synthetic --task-ratio rigid:1,deform:1 --deform-decoder true --deform-residual-affine false --group-size 2 --log-every 1 --val-png false` + `TINY_ARGS` + `COMMON` (identical TINY `--filters` / `--decoder-widths` at every level, so the ModelConfigs are dict-equal and the deform decoder exists at every level; review NIT 8). Step counts are set with `--overfit-max-steps` (overfit mode overwrites `args.steps` with it, args.py; `--steps` would be ignored). Runs: (1) `--level 4 --overfit-max-steps 3`; (2) `--level 2 --cascade-from <run1>/ckpt/last --overfit-max-steps 3 --bn-recal-batches 4` (≥ 2 tasks); (3) `--level 1 --cascade-from <run2>/ckpt/last --deform-group-size 2 --min-pairs n --amp false --bn-recal-batches 4 --val-every 3 --overfit-max-steps 3 --val-png true` (the later `--val-png true` overrides the shared `false`; argparse keeps the last occurrence). Every `M.main(...)` return code is asserted `in (0, 1)` (the overfit Dice / loss threshold is not required to be reached, as in `TestCascadeMain`). Expected outcomes for run 3: config.json `cascade.source_level == 2`, `cascade.stages` levels `[4, 2]` with `checkpoint_key` `frozen_stage` / `model` and origin steps 3 / 3, `init_weights.via == "cascade"`, `cross_level == [2, 1]`, `args.min_pairs == "n"`; `tf.train.list_variables(<run3>/ckpt/last)` contains keys under `frozen_stage_4/` and `frozen_stage_2/` and none under `frozen_stage/`; logged `train_deform/n_pairs == 2` and `train_rigid/n_pairs == 2`; `val_rigid_frozen4/center_rms_mm` and `val_deform_frozen4/template_dice` finite. Then a resume of run 3 with the identical run-3 argv except `--overfit-max-steps 5` (still `--val-png true`; rc in (0, 1); fingerprints OK, exactly one `bn_recal` event in `metrics.jsonl`, no second recalibration). Run 3 validates at step 3 (`--val-every 3`) and at the stop step 5 (overfit final validation), so `ckpt/best`, `ckpt/best_deform` and `val_png/{rigid,deform}_step0000005.png` exist. Final assertion, in-process: `check_mixed_run.check(<run3>, ["rigid", "deform"], 5, expect_best=True, expect_cascade=True, expect_cascade_stages=2, expect_min_pairs="n")` returns ok (every result tuple True). `check()` gains the keyword arguments `expect_cascade_stages=1`, `expect_min_pairs=None`, `memory_budget_mb=0`, `project_steps=0` mirroring the CLI options of section 9 (defaults = today's checks); the checker gets no PNG opt-out. Expected ~8-10 min (level-1 tensors at N 2).

Added suite time (estimate): ~15 min (T-E1 ~10, level-1 unit tests ~4, rest < 1). The report states the measured value.

## 12. Files

Modified:
- `models/cascade.py`: `PREV_LEVEL`, `FrozenChain`, `skip_to`, `FrozenStage` additions (`n_stages`, `source_model`, `fingerprints`, `checkpoint_extra`), docstring table.
- `runtime/run.py`: `TrainConfig.min_pairs`, postprocess wiring, `_cascade_stages` / `self._n_stages` dispatch (section 2.1), chain branches in the train / deform steps and validation functions, frozen4 baselines, `n_pairs` logging, `val/wall_s` (level 1), docstrings.
- `runtime/args.py`: `--cascade-from` levels, `--min-pairs`, `--p-pair` check, `train_config_from_args`.
- `runtime/checkpoint.py`: `CASCADE_SOURCE_LEVEL` dict, `load_cascade_source(path, level)`, `fingerprints_match`.
- `main.py`: level-1 fresh / resume cascade path, record, extras, `_check_cascade_fingerprints`, `_cascade_lines`, docstring example.
- `data_loading/postprocess.py`: `MIN_PAIRS`, `num_pairs` / `sample_pairs` `min_pairs`, both postprocessors.
- `models/memory_model.py` (+ `scripts/memory_model.py` `--min-pairs` on `predict`): `num_pairs` / `Workload.min_pairs`.
- `tests/gpu/measure_level_memory.py`: `--min-pairs` axis (`KEY_FIELDS`, isolate child argv, `TrainConfig(min_pairs=...)`, predicted `Workload`); docstring: level-1 `--cascade true` now exercises the Trainer chain branches (dispatch via the stand-in's `n_stages = 2`), `--val-pass` includes the `_frozen4` fields. No change to `cascade_kind` / `MODEL_CASCADE` / record schema beyond `min_pairs`.
- `tests/gpu/frozen_chain_standin.py`: `n_stages = 2`, `linear_stages`, `deform_stages` (section 2.1; `linear` / `deform` verbatim); docstring (FrozenChain is the real implementation; the stand-in is kept for the measurement tool and for T-C2, reproducing job 10516237).
- `tests/gpu/check_mixed_run.py`: section 9.
- Tests listed in section 11; `tests/run_tests_wsl.sh` only if a new module needs registering (stage-1 py_compile / `bash -n` lists include the new script).
- `.agents/tasks/network-design/design.md`: v3.15 (section 13).

New:
- `slurm/m9_level1_cascade.sh`.
- `.agents/tasks/milestone-9-phase2/report.md` (after implementation: suite result, goldens, the exact submit lines).

## 13. Network design doc version

The brief says design.md is v3.13; on disk it is already **v3.14** (phase-1 implementation bump, 2026-10-23). The coder therefore bumps it to **v3.15** (not v3.14). Edits:
- Status line: prepend "v3.15 (milestone-9 phase 2 design, `.agents/tasks/milestone-9-phase2/design-m9-phase2.md`): 3-level frozen cascade `FrozenChain` (levels 4 + 2 from the milestone-8 level-2 checkpoint `m8_level2_gradnorm/ckpt/best` step 6000, its stored `frozen_stage` = level 4, fingerprint-checked), level-1 trainable stage with the identical default ModelConfig (no different-width stage; the N = 5 fr50 route needs a full 4 → 2 → 1 retrain), skip of the last frozen stage (recipe 0), `val_<set>_frozen` (4 + 2) and `val_<set>_frozen4` baselines, `val/wall_s`; opt-in `--min-pairs n` (pairwise-loss minimum = group size, random N-cycle at k = N, same RNG draw); level-1 recipe and `slurm/m9_level1_cascade.sh` (sections 4.7, 6, 9, 11)."
- 4.7: an "As specified for milestone 9 phase 2" block (sections 2-4 of this design in five bullets) and the note that the level-1 recipe uses skip 0, deviating from the 0.15 default above.
- 6.1 / 6.2: `--min-pairs` sentence.
- 9: level-1 4-pair estimate and the A100 rule.
- 11 row 9: phase-2 status.

## 14. Results to compare (reported after the long run)

| Reference | Values |
|---|---|
| Level 4 `m7_three_task_lncc` | rigid 3.82, affine 4.80 mm center; deform template / pair Dice 0.757 / 0.701 |
| Level 2 `m8_level2_gradnorm` | rigid 2.03 / 2.90, affine 2.53 / 3.67, rigid_g 2.24 / 3.22, affine_g 3.18 / 4.59 mm (center / pair); deform 0.846 / 0.789; folding 0.012 %; inverse consistency 0.19 mm; affine pair G gap +0.92 mm |

Level-1 report table per set: composed vs `_frozen` (4 + 2) vs `_frozen4`, linear center / pair RMS, rot / trans, affine scale / shear / QR rotation for the `_g` sets (G gap = `<task>_g` − `<task>`, composed and frozen); deform hard template / pair Dice, LNCC template / pair, folding %, inverse consistency, composed mean displacement, each with the identity baseline. The `_frozen` values are evaluated at level-1 resolution on pooled inputs, so they approximate but do not equal the m8 run's own level-2 numbers. Success indicators (proposed, not gates): composed better than `_frozen` on every set; folding ≤ 0.05 %; inverse consistency not worse than 2× level 2.

## 15. Backlog / documented only

- **N = 5 route (Jay: document only).** fr50 + `--lean-losses true` measured 38.4 GB, 22.2 s at level-1 N 5 (N 6 OOM). Because the trainable stage must share the frozen stages' ModelConfig, it requires retraining level 4 (m7 recipe), level 2 (m8 recipe) and level 1 with `--filters 16,32,128,192,256,384 --decoder-widths 128,96,64,16,8` passed identically at every level (phase-1 6.4 sequence), plus a level-1 re-measurement with `--min-pairs n` (more pairs at N 5: 2 → 5). No code beyond what exists.
- Skip 0.15 at level 1 (design.md 4.7 default): implemented, not in the recipe.
- Uniform random derangements instead of N-cycles: rejected (section 5).
- Label sampling (full label set), TTA: unchanged, later milestones.

## Review responses (iteration 2, `design-review.md`)

| # | Severity | Response |
|---|---|---|
| 1 | MEDIUM | **Addressed** (reviewer's recommended option). Section 11 T-E1: all step counts via `--overfit-max-steps 3` (runs 1-3) and `5` (resume of run 3); run 3 and its resume pass `--val-png true`, which overrides the shared `--val-png false`; every `M.main` rc is asserted `in (0, 1)`; the final assertion is the in-process `check_mixed_run.check(<run3>, ["rigid", "deform"], 5, expect_best=True, expect_cascade=True, expect_cascade_stages=2, expect_min_pairs="n")` → ok. `check()` gains matching keyword arguments (defaults = today); no PNG opt-out is added to the checker. |
| 2 | NIT | Addressed: T-G1 now compares the default graph with the `min_pairs="n"` graph. The `"n"` branch is built under `tf.name_scope("min_pairs")` (section 5), so the test counts only those ops and ignores other conds such as AMP loss scaling. Bit-identity stays with the goldens. |
| 3 | NIT | Addressed: D6 now reads "0 as in the milestone-8 runs, a recipe choice of this design (deviation from design.md 4.7 / phase-1 0.15), to be confirmed with Jay in the report". |
| 4 | NIT | Addressed: section 2.2 adds the exception contract. `FrozenChain.from_checkpoint` catches `(AssertionError, ValueError, tf.errors.NotFoundError, tf.errors.InvalidArgumentError)` and re-raises a ValueError that names the prefix and the original type. main wraps it into `SystemExit("main.py: --cascade-from <path>: ...")` (section 4.3 step 2). The section 10 row and T-C5 now name AssertionError / NotFoundError, and T-C5 adds the main-level SystemExit check. |

## Review responses (iteration 1, `design-review.md`)

| # | Severity | Response |
|---|---|---|
| 1 | MEDIUM | **Addressed** (reviewer's primary fix). Section 2.1: one dispatch helper, `n_stages = getattr(cascade, "n_stages", 1)`, evaluated once in `Trainer.__init__` (`self._n_stages`), used by every chain branch; `n_stages == 2` requires level 1. `FrozenChainStandIn` gets `n_stages = 2`, `linear_stages`, `deform_stages` with FrozenChain's formulas, so the measurement tool (training and `--val-pass`) runs the real level-1 chain path, including the `_frozen4` fields; record schema / `cascade_kind` unchanged. Tests: T-C2 extended to the `*_stages` outputs, new T-C8 (level-1 cascade measure with `--val-pass` ok, existing level-1 test unchanged), new T-C9 (dispatch). Section 12 lists the tool and stand-in changes. The alternative (tool builds `FrozenChain`) is rejected with a reason in 2.1. |
| 2 | NIT | Addressed: section 3.3 states that chain `_frozen` centrality keys describe the level-2 residual SVF and `_frozen4` the full level-4 SVF. No code change. |
| 3 | NIT | Addressed: sections 3.3 and 6 state the ≈ 20 GB of first-pass frozen fields, peak host ≈ 30-35 GB, and the `sacct MaxRSS > 56 GB → --mem 96G` rule. No code change. |
| 4 | NIT | Addressed: smoke formula uses 660 s recal plus the explicit pre / post passes (≈ 2.25 h, still 04:00:00). Section 8 accounting rule and section 9 `--project-steps`: recal `elapsed_s` includes its two validation passes, which never produce `val/wall_s`, so nothing is counted twice. |
| 5 | NIT | Addressed: T-C2 runs under `tests.test_trainer._OpDeterminism`. |
| 6 | NIT | Addressed: section 0 invariant reworded (`args` gains `min_pairs: "1"` in every run). |
| 7 | NIT | Addressed: section 9 gives the exact stdlib mirror (float32 via `struct`, half-to-even `round`) and `N_deform = deform_group_size or group_size`; T-P1 checks equality against it. |
| 8 | NIT | Addressed: T-E1 specifies the shared flags (`--task-ratio rigid:1,deform:1`, deform decoder, TINY widths, overfit-synthetic), the per-run flags and the expected config / checkpoint-key / log outcomes. |

Version note: the brief names design.md v3.13 → v3.14, but the file on disk is already v3.14 (phase-1 implementation bump, 2026-10-23), so section 13 keeps the bump at **v3.15** to avoid two different v3.14 entries.
