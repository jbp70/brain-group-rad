# predict-levels: pyramid-level selection for predict.py (design v1)

Status: v1, ready for review. Jay decided (via the parent session): option (A) for affine+deform
(section 5), and every combination that follows from the truncation / skip rule is allowed
(section 2.2 lists all 7 at level 1 and all 3 at level 2).

## 1. Overview

`predict.py run` and `predict.py check-streaming` get a `--levels` option that chooses which cascade
stages of the checkpoint run. Default (option absent) = all levels of the checkpoint, through the
existing code path unchanged, so outputs stay bit-identical. A selection is a strictly descending,
non-empty subset of the checkpoint's levels (`LoadedModel.levels`, `(4,)`, `(4, 2)` or `(4, 2, 1)`).
Two mechanisms follow from it:

* **Truncation**: levels finer than the last selected level are not run. The last run level's fields
  are upsampled to level 1 (160x224x192) with the cascade's own upsampling; matrices need nothing.
* **Skip**: a level coarser than the last selected level that is not selected is skipped. It
  contributes the identity, and the later stages still run on the un-pre-warped inputs, through the
  same ops the trainer uses for a skipped stage.

Before any model is built, an exposure check reads the skip probabilities from the checkpoint chain's
config.json files. A selection that skips a level for a branch task whose stage was never trained
with that skip is refused (exit 2) unless `--allow-untrained-skip` is given. The selection and the
exposure go into predict.json (and the check-streaming JSON). File names, conventions (evaluation/fields.py
canonical: xyz, mm, fwd on the atlas grid), atlas and atlas_seg are unchanged.

Technology stack (locked): Python 3.8 syntax, TF 2.10, numpy, unittest; no new dependencies. New
module `inference/levels.py` (TF-free); changes in `inference/cli.py`, `inference/runner.py`,
`inference/predict.py`, `slurm/predict_mindboggle.sh`; tests in a new `tests/test_predict_levels.py`
plus small additions to `tests/test_predict.py`.

## 2. Verified training semantics (the contract the inference must match)

### 2.1 Where the trainer skips a stage

Trainer step bodies, stage mode (`Trainer._skip_mode == 'stage'`, set in `runtime/run.py:Trainer.__init__`
from `cfg.cascade_skip_stage`, which `runtime/args.py` fills from `args.cascade_skip_stage_parsed`,
args.py lines ~719-723 and ~921-923):

| Trainable level | Task | Code (runtime/run.py) | Skip of level 4 | Skip of level 2 |
|---|---|---|---|---|
| 2 | deform | `_make_deform_train_step.body`: `prev = cascade.deform(x)`; `CS.skip_fields(skip[0], [U, U_inv])` | U = U_inv = 0 (level-2 shape) | n/a |
| 2 | rigid/affine | `_make_train_step.body`: `A_prev = cascade.linear(x)`; `CS.skip_linear(skip[0], A_prev)` | A_prev = I | n/a |
| 1 | deform | `cascade.deform_skip(x, skip4=..., skip2=...)` (`models/cascade.py:FrozenChain.deform_skip`) | U4 = U4_inv = 0 at level 2 (`skip_fields`); level-2 model runs on `prewarp_deform(x2, 0)`; `compose_cascade(0, 0, u2, u2_inv)`; upsampled | U = up(U4), U_inv = up(U4_inv) (`skip_to`) |
| 1 | rigid/affine | `cascade.linear_skip(x, task, skip4=..., skip2=...)` (`FrozenChain.linear_skip`) | A4 = I (`skip_linear`); level-2 model on `prewarp_linear(x2, I)`; A = `compose_linear(R2_raw, I, center2)` | A = A4 (`skip_to`) |

After the frozen part, every case continues identically: `x_in = CS.prewarp_deform(x, U)` or
`CS.prewarp_linear(x, A_prev, spacing)`, `out = model(x_in, task)`, then
`diffeo.compose_cascade(U, U_inv, out.u, out.u_inv)` or `CS.compose_linear(out.A_raw, A_prev,
model.centering)`. So "a skipped stage contributes identity" means: its output is replaced by exact
zeros / exact identity (`tf.where` on the computed value), and every downstream op (pre-warp,
compose, centering) still runs on that identity. Both skips -> U = U_inv = 0 (`skip_to(skip2,
up(0), ...)`) and A = I; the trainable level-1 stage runs on `prewarp(x1, 0 | I)`.

Legacy mode (`cascade_skip_prob > 0`, same file, `skip_on` branches): at level 2 the same as stage
mode skip 4 for every task; at level 1 the skip drops level 2 (`skip_to(skip, st['U4'], st['U'])`,
`skip_to(skip, st['A4'], st['A'])`), i.e. legacy p is a level-2 skip, never a level-4 skip.
Validation and eval never skip (run.py module doc).

Per-stage draws are independent per (task, level, seed, step) (`cascade_stage_skip_draw`), so the
probability that a training step of task t skipped all levels in a set K is the product of the
per-level probabilities.

### 2.2 Inference semantics per combination

Notation: x1 = level-1 images (after the affine resample for affine+deform), x2 = pool(x1),
x4 = pool(x2); "run(lv)" = the stage at level lv with the existing runner code; up = 
`diffeo.upsample_displacement` (2 x trilinear, values doubled), applied once per level step.

Level-1 checkpoint (stages 4 frozen, 2 frozen, 1 trainable):

| `--levels` | Kind | Deform branch | Linear branch |
|---|---|---|---|
| (absent) / `4,2,1` | default | existing path | existing path |
| `4,2` | truncation | U1 = up(compose(up(u4), up(u4_inv), u2, u2_inv)) (= `FrozenChain.deform(x1)['U']`) | A = A2 (= `FrozenChain.linear`) |
| `4` | truncation | U1 = up(up(u4)) (= `FrozenChain.deform_stages(x1)['U4']`, = `deform_skip(skip2)` U) | A = A4 |
| `2,1` | skip 4 | `deform_skip(skip4)` then run(1) on prewarp + compose | `linear_skip(skip4)` then run(1) |
| `1` | skip 4 and 2 | U = U_inv = 0 at level 1, run(1) on `warp(x1, 0)`, `compose_cascade(0, 0, u1, u1_inv)` | A_prev = I, run(1) on `prewarp_linear(x1, I)`, A = `compose_linear(R1_raw, I, center1)` |
| `2` | skip 4 + truncation | `deform_skip(skip4)` U (level 1) | `linear_skip(skip4)` |
| `4,1` | skip 2 | `deform_skip(skip2)` then run(1) | `linear_skip(skip2)` then run(1) |

Level-2 checkpoint: `4,2` default; `4` truncation (U1 = up(up(u4)), A = A4); `2` skip (U = U_inv = 0
at level 2, run(2), as `CS.skip_fields` / `CS.skip_linear`). Level-4 checkpoint: only `4`.

These are all non-empty descending subsets of the checkpoint levels (7 at level 1, 3 at level 2,
1 at level 4); no other value is accepted. `2`, `4,1` and the level-2 `2` are not in the brief's
required list; they fall out of the general rule with no extra code, and Jay asked for them to be
included (decided). Tests: level-1 `2` and `4,1` are covered by the skip-reference test (11.3) and
the plan / exposure unit tests; streamed == unchunked at level 1 covers the four required ones plus
`4,1` (the only selection with a two-step `_up_fn`), level 2 covers `4` and `2`.

### 2.3 Truncation upsampling

The cascade moves fields between levels only with `diffeo.upsample_displacement`: `FrozenStage.deform`
(level 4 -> 2), `FrozenChain._deform` (level 2 -> 1), `FrozenChain.deform_stages` (U4: two steps).
The runner already does the same for level-4 / level-2 checkpoints (`runner._up_to_level1`,
`StreamedRunner._to_level1_fn`, `UnchunkedRunner.run` `to_l1`): one `upsample_displacement` per
2x step, forward and inverse separately. Truncation reuses `_up_to_level1(u, last_level)`. Upsampling
fwd and inv independently is what the cascade does (the inverse is never recomputed after an
upsample), so this is exact w.r.t. training. Matrices (mm, level-independent) are passed between
levels unchanged by the cascade (`FrozenChain._linear` passes A4 to level 2 as is), so a truncated
linear branch hands its last A to `canonical.to_canonical` unchanged; the full-resolution field is
built there as today.

## 3. CLI (inference/cli.py)

Added to `_common` (both subcommands):

* `--levels LEVELS`: comma-separated levels, coarse to fine, e.g. `4`, `4,2`, `2,1`, `1`. Default
  None = all levels of the checkpoint. Parse-time validation (TF-free, argparse error, exit 2):
  every item an integer in {4, 2, 1} after `strip()`; no empty items; strictly descending (which
  also excludes duplicates); at least one level. Stored as `args.levels` = tuple of ints or None.
  Spaces around commas are accepted; `all` is not a value.
* `--allow-untrained-skip`: store_true. Proceed when the exposure check finds a skip the chain was
  never trained with (section 6); default off.

Checkpoint-dependent validation runs after `_peek_config` (no model build): every selected level
must be one of the checkpoint's levels (`loader.frozen_levels_of(level) + (level,)`); otherwise
`PredictError(EXIT_INPUT, "--levels 1: the checkpoint is a level-2 run with levels [4, 2]")`.
A selection equal to all checkpoint levels is normalised to None (the default path), so
`--levels 4,2,1` on a level-1 checkpoint is bit-identical to no flag.

## 4. Implementation

### 4.1 inference/levels.py (new, TF-free)

* `parse_levels(text) -> tuple` (used by the argparse type; raises `argparse.ArgumentTypeError`
  via a wrapper in cli.py, `ValueError` here).
* `class LevelPlan`: built by `plan(ckpt_levels, selected)` (`selected` None -> all). Fields:
  `ckpt_levels`, `selected`, `default` (bool), `last` (= selected[-1]), `run` (= selected),
  `truncated` (ckpt levels finer than last), `skipped` (ckpt levels coarser than last and not
  selected), `skipped_before(lv)` (skipped levels coarser than lv), `kind(lv)` in
  {'run', 'skipped', 'truncated'}, `to_dict()` for the JSON. Raises `ValueError` for a level not in
  the checkpoint.
* Exposure (section 6): `run_skip_probs(config) -> {task: {level: p}}` and
  `exposure(plan, tasks, configs) -> dict`, pure functions on config dicts.

### 4.2 Runner (inference/runner.py)

`_Runner.__init__(loaded, plan=None)`; `run()` unchanged in signature. Dispatch: `plan is None or
plan.default` -> the current `_linear` / `_deform` / `UnchunkedRunner._linear/_deform` verbatim (not
edited, so the default graph and numbers cannot change). Otherwise new methods `_linear_sel` /
`_deform_sel`.

**Streamed** (`StreamedRunner._linear_sel(n, x1, task, ...)`): iterate `lv` over the checkpoint
levels down to `plan.last`. Skipped `lv`: no model call; if no level has run yet, `a_prev` =
`tf.eye(4, batch_shape=[n])` float32 (bitwise the `skip_linear` value `zeros_like + eye`); if a
level has run (only `4,1`), `a_prev` is kept (`skip_to(skip2, A4, A)`). Run `lv`: `a_prev is None`
-> today's pool + centering; else today's `_prewarp_linear_fn(lv)` + `CS.compose_linear(a_raw,
a_prev, model.centering)`. Phase names and `_record` as today.

`StreamedRunner._deform_sel(...)`: `prev` is None (nothing yet), `ZERO` (identity so far) or
(store names, level). Skipped `lv`: `prev` None -> `ZERO`; otherwise kept. Run `lv`:
* `prev is None` -> pool input, `prev_fn=None` (today).
* `prev is ZERO` -> input `warp(pool(x1), zeros)` (new `_prewarp_deform_zero_fn(lv)`, same
  `diffeo.warp(..., lean=self.lean)` as today with a `tf.zeros([k] + LEVEL_SHAPE[lv] + [3])`
  field), `prev_fn` returns `(zeros, zeros)` at `lv`'s shape, so P12 runs
  `compose_cascade(0, 0, u, u_inv)` as the trainer does.
* `prev` from level `lp` -> `_prewarp_deform_fn(lv, lp)` and `_up_fn(lv, lp)` upsample `log2(lp/lv)`
  times (1 today, 2 for `4,1`; the existing one-step functions are the `lp = 2 lv` case, cache key
  `("prewarp_deform", lv, lp)`).
The last run level uses `_level_sink(plan.last, ...)` (today's `to_l1` from that level). Store
fields of a level are freed after the next run level consumed them, as today. `free_src` frees the
affine source after the last run level's P0.

**Unchunked** (`UnchunkedRunner`): uses the trainer functions literally, with
`s = tf.constant(True)` as the skip tensor, on all N in one tf.function per piece:

| Checkpoint | Selection | Frozen part (deform / linear) | Trainable stage |
|---|---|---|---|
| L1 | `4,2` | `frozen.deform(x1)` U, U_inv / `frozen.linear(x1)` | not run |
| L1 | `4` | `frozen.deform_stages(x1)` U4, U4_inv / `frozen.linear_stages(x1)['A4']` | not run |
| L1 | `2` | `frozen.deform_skip(x1, skip4=s)` / `frozen.linear_skip(x1, task, skip4=s)` | not run |
| L1 | `2,1`, `1`, `4,1` | `deform_skip` / `linear_skip` with skip4 = s for 4 not selected, skip2 = s for 2 not selected | today's trainable body (prewarp, model, compose) |
| L2 | `4` | `frozen.deform(x2)` U, U_inv upsampled once / `frozen.linear(x2)` | not run |
| L2 | `2` | `CS.skip_fields(s, [frozen.deform(x2) U, U_inv])` / `CS.skip_linear(s, frozen.linear(x2))` | today's trainable body |

`deform_stages` / `linear_stages` run level 2 too (wasted, only in the small-N unchunked check);
accepted for literal reuse of the trainer's functions.

Sink result: `A`, `U1`, `U1_inv` as today. `u_ckpt`, `u_inv_ckpt`, `v`, `v_raw` are set only
when the checkpoint's trainable stage runs (`plan.last == loaded.level`), else None (the writer does
not use them). Summary gains `levels` (`plan.to_dict()`); `chunks` lists only run levels;
`stats.max_abs_v` only run levels.

Unchanged and deliberately conservative: `store.estimate_store_gb` and `loader.check_unchunked` keep
using all checkpoint levels (an over-estimate for truncated runs; no new failure modes).

### 4.3 predict.py

In `_run_predict` and `_run_check`, right after `_peek_config`: build the plan (exit 2 on a bad
level), read the chain configs and compute the exposure (section 6), apply the refusal rule, then
continue as today (subjects, load, runner with `plan=`). `_pipeline(branches, plan)` describes the
selection, e.g. `deform cascade (L4 skipped (identity) -> L2 -> L1) ...` or `(L4 -> L2; L1 not run,
fields upsampled from level 2)`. Logging: one INFO line with the plan; WARNINGs per section 6.

## 5. affine+deform level sets (decided: option A)

Options considered: (A) one `--levels` for both cascades of affine+deform; (B) `--levels` for both
plus an optional `--linear-levels` override; (C) `--levels` for the deform cascade only.

Decision (Jay): (A). The same plan drives `_linear_sel` (affine) and `_deform_sel` (deform); the
affine resample between them is unchanged (it uses the affine A of the selection). The exposure
check covers both tasks ('affine' and 'deform'), and any untrained entry refuses the whole run.
Consequence on the retrained checkpoint: affine+deform with `2,1`, `2` or `1` is refused (exit 2)
without `--allow-untrained-skip`, because rigid/affine level 4 was never skipped in training
(`cascade_skip_linear 4:0` in both runs). `4,1` is trained for both tasks (linear 2:0.15). The
planned ablation uses the deform branch (Mindboggle is already MNI-affine-aligned), so it is
unaffected. (B) can be added later without breaking (A).

## 6. Training exposure check

### 6.1 Which run trained which stage

* Trainable stage (level = checkpoint level): the checkpoint's own config.json (`loaded` run dir;
  `_peek_config` already reads it).
* Frozen level-2 stage of a level-1 checkpoint: the level-2 source run's config.json. Candidates in
  order, first readable wins: `config.cascade.source_config`; `runtime.checkpoint.source_config_path(
  stage['origin_checkpoint'])` for the `cascade.stages` entry with level 2. Unreadable -> exposure
  `unknown` for that stage.
* Frozen level-4 stage: never has a coarser level, so no exposure is needed.

### 6.2 Skip probability of a run, `run_skip_probs(config)`

From `config['args']` (what the Trainer consumed): if `cascade_skip_prob` > 0 (legacy): level-2 run
-> p(t, 4) = cascade_skip_prob for every task; level-1 run -> p(t, 2) = cascade_skip_prob, p(t, 4) =
0 (section 2.1). Else `cascade_skip_stage_parsed[t][lv]` (JSON keys are strings: accept '4' and 4);
a task or level missing -> 0. Missing keys (runs before retrain-prereqs) -> 0, source recorded as
`absent (run predates per-stage skip)`. The `cascade.skip_probs` record is copied into the JSON for
reference but not used for the decision.

### 6.3 Exposure of a selection

For each branch task t (rigid | affine | deform; affine+deform = affine and deform) and each run level
s with `K = plan.skipped_before(s)` non-empty: `p_l = p_run(s)(t, l)` for l in K, `joint = prod(p_l)`
(independent draws, section 2.1). Status per (t, s): `trained` if joint > 0, `untrained` if some p_l
= 0, `unknown` if the run config is unreadable. Truncated and run-only levels need no exposure
(truncation is always allowed). Example, retrained chain (r1sw deform 4:0.15, 2:0.15, linear 4:0,
2:0.15; r2 deform 4:0.15, linear 4:0):

| `--levels` | deform | rigid / affine |
|---|---|---|
| `2,1` | L2: 0.15 (r2), L1: 0.15 (r1sw) | L2: 0 (r2) -> refused |
| `1` | L1: 0.15 x 0.15 = 0.0225 | L1: 0 x 0.15 = 0 -> refused |
| `2` | L2: 0.15 | refused |
| `4,1` | L1: 0.15 | L1: 0.15 |

Old chain (m9_level1_cascade / m8_level2_gradnorm, skip prob 0, no per-stage skip): every skip is
`untrained` (refused); truncation allowed.

### 6.4 Rules, logging, exit codes

* Any `untrained` or `unknown` entry and no `--allow-untrained-skip`: `PredictError(EXIT_INPUT)`
  (exit 2), message lists `level s, task t, skipped levels K, p, run config path` and names the flag.
  Raised before reading subjects or building a model; predict.json / the check JSON is still written
  (status error) with `levels` filled.
* With the flag: one WARNING per such entry ("forced: ..."), `levels.exposure.forced = true`.
* `trained` entries with joint < `LOW_EXPOSURE = 0.05`: WARNING "level s task t saw this skip in
  about x % of training steps". (0.0225 for `1` warns; 0.15 does not.)
* All WARNINGs land in predict.json `warnings` through the existing collector.

## 7. Records

predict.json: new key `levels` (added to `PREDICT_KEYS`; also in error records):

```
"levels": {"requested": "2,1" | null, "checkpoint_levels": [4, 2, 1], "selected": [2, 1],
  "default": false, "last": 1, "skipped": [4], "truncated": [],
  "per_level": {"4": "skipped (identity; later stages on un-pre-warped inputs)",
                "2": "run", "1": "run"},
  "upsampled_from_level": 1,
  "allow_untrained_skip": false,
  "exposure": {"forced": false, "low_threshold": 0.05,
    "runs": {"1": {"config": "<path>", "source": "args.cascade_skip_stage_parsed",
                   "probs": {"deform": {"4": 0.15, "2": 0.15}, ...}, "skip_mode": "stage"},
             "2": {...}},
    "entries": [{"task": "deform", "level": 2, "skipped": [4], "p": {"4": 0.15},
                 "joint": 0.15, "status": "trained", "run_config": "<path>"}, ...],
    "min_joint": {"deform": 0.15}}}
```

Default runs write the same structure (`requested` null, `default` true, empty skipped / truncated,
`entries` []). `pipeline` text reflects the plan. `chunks.bounds` lists run levels only. The
check-streaming JSON gets the same `levels` key (`CHECK_KEYS`).

## 8. check-streaming

`--levels` and `--allow-untrained-skip` come from `_common`. `_run_check` builds the plan and the
exposure as in `run`, passes `plan` to both runners, resolves chunk sizes and caps for run levels only
(`keys = ["l%d" % lv for lv in plan.run]`), and writes `levels`. Gate, bounds and exit 4 unchanged.

## 9. slurm/predict_mindboggle.sh

New env overrides: `LEVELS` (default empty = no flag) and `ALLOW_UNTRAINED_SKIP` (default empty;
`1` adds `--allow-untrained-skip`). `LEVEL_FLAGS` is built once and appended unquoted to steps 1a,
1b, 1c and 2. Header documents both. With the defaults the commands are byte-identical to today's.
Step 1c (affine+deform record, non-blocking) is refused for `LEVELS=2,1` / `1` on the retrained
checkpoint under option (A); the existing `|| echo WARNING` keeps the job going.

Prepared (not submitted) ablation, rtx8000, 1 GPU, 3 CPUs, 96G, 5 h each, deform branch, checkpoint
`.../scratch/predict_ckpts/r1sw_lr1e-4_wd1e-1_n8_d4_s3000/ckpt/best_deform`,
`CHECK_CHUNKS="--chunk-l4 3 --chunk-l2 3 --chunk-l1 3 --tol-field-max-mm 0.005"`, `LEVELS` in
{`4`, `4,2`, `2,1`, `1`}, `NAME=r1sw_lr1e-4_s3000_deform_L<levels>`; exact lines go in the report.
Submission needs Jay's approval (cluster rules).

## 10. Error handling summary

| Failure | When | Exit | Logged |
|---|---|---|---|
| bad `--levels` syntax / order / value | argparse | 2 | argparse message |
| level not in the checkpoint | after `_peek_config` | 2 | ERROR, predict.json status error |
| untrained / unknown skip without the flag | before subjects | 2 | ERROR listing entries |
| chain config unreadable with the flag | before subjects | continues | WARNING, status unknown recorded |
| everything else | as today | as today | as today |

## 11. Test plan

Unit (TF-free, `tests/test_predict_levels.py`, fast): `parse_levels` (valid, spaces, duplicates,
ascending, empty item, 3, `all` -> errors); `plan` for every checkpoint level x selection (run /
skipped / truncated / last, normalisation of the full set to default, level not in checkpoint);
`run_skip_probs` on config dicts: stage mode with string keys, legacy at level 2 and 1, absent keys;
`exposure` on the retrained-chain and old-chain fixtures reproduces the table in 6.3 (0.0225 for
`1`, refusals for linear); CLI `--levels` parse errors exit 2 without importing TF.

Runner (TF, tiny random-weight models from `tests/test_predict_runner.py` helpers `_level2`,
`_level1`; level 1 on /CPU:0 as today):
1. Default bit-identity: `plan=None` vs `plan(levels=(4,2,1))` (normalised) `array_equal` on A, U1;
   and a one-off golden check by the implementer: sha256 of the decoded output arrays of `predict.py
   run` on a tiny level-2 checkpoint (CPU) at the parent commit vs after the change, recorded in the
   report.
2. Truncation: L2 `4`: U1 == `_up_to_level1(frozen.deform(x2) U, 2)` and A == `frozen.linear`; L1 `4,2`
   / `4`: U1, U1_inv == `FrozenChain.deform(x1)` / `deform_stages` U4 (unchunked bitwise, streamed
   within the existing S_stream bounds), A == `frozen.linear` / `linear_stages['A4']`.
3. Skip, direct reference written in the test from the trainer's functions (`deform_skip`,
   `linear_skip`, `CS.skip_fields`, `CS.skip_linear` with `tf.constant(True)`, then
   `CS.prewarp_*`, `model(..., training=False)`, `compose_*`): L2 `2`; L1 `2,1`, `1`, `4,1`, for
   deform and affine. Unchunked <= B_EXACT_FALLBACK (1e-5), streamed within S_stream bounds.
4. Streamed == unchunked (`compare_modes`, chunks 1 and N) for L2 {`4`, `2`} and L1 {`4`, `4,2`,
   `2,1`, `1`, `4,1`} for deform and affine+deform; summary `chunks` has run levels only.
5. Zero-field identity sanity: L1 `1` streamed prewarp input equals `warp(x1, 0)` and the result
   differs from the default (the skip is active).

predict.py / end-to-end (`tests/test_predict.py`, tiny checkpoints written to a temp run dir with
config.json, as the existing TestLevel2Convention): `run --levels 4` and `--levels 2` on a level-2
checkpoint whose config has deform skip 4:0.15 and linear 0: outputs written with the same file
names, predict.json `levels` record, `evaluation check-convention` still selects xyz +++ mm;
refusal: `--branches affine --levels 2` exit 2 with predict.json status error and entries; same with
`--allow-untrained-skip` exit 0, `forced` true and WARNING in `warnings`; low-exposure WARNING via a
config with p 0.04; level not in checkpoint (`--levels 1` on level 2) exit 2; `check-streaming
--levels 2` passes with `levels` in the JSON.

Order: tests/test_predict_levels.py, test_predict_runner.py, test_predict.py, then the full suite
(`tests/run_tests_wsl.sh`, ~5 h, background) before the commit. GPU limit BGR_TF_GPU_LIMIT_MB <= 6000.

## 12. Out of scope

Retraining, loss changes, TTA, group sizes, evaluator metrics, changes to the store estimate or the
unchunked bound, per-branch level sets (option B / C of section 5).

## 13. Decisions log

* affine+deform: one `--levels` for both cascades (option A), Jay.
* All combinations that follow from the truncation / skip rule are allowed (`2`, `4,1` at level 1,
  `2` at level 2, besides the required ones), Jay.
* Refusal for untrained or unknown skips (exit 2), override `--allow-untrained-skip`; truncation
  always allowed (brief).
