# Design review v2: predict-levels design v2

Revision review. Prior review: v1 (CHANGES_REQUESTED, 4 MEDIUM + 5 NIT). I re-checked each v1 finding against design v2 and then reviewed the changed parts against the code again: `models/cascade.py` (FrozenStage.linear/deform, FrozenChain `_linear`, `_deform`, `deform_stages`, `linear_stages`, `linear_skip`, `deform_skip`, `skip_to`, `skip_linear`, `skip_fields`, `prewarp_*`, `compose_linear`), `runtime/run.py` (`_make_deform_train_step` body l.904-931, `_make_train_step` body l.1026-1052, `_skip_mode` l.688-691, `_stage_skip_on`, `cascade_stage_skip_draw`, stage draw in `fit` l.2022-2026), `runtime/args.py` (l.697-723, l.919-923), `main.py` (cascade record l.320-356, `_skip_record`), `runtime/checkpoint.py` (`load_cascade_source` info, `source_config_path`, `resolve_checkpoint`), `inference/runner.py` (whole file), `inference/streaming.py` (`deform` / P12 with `prev_fn`), `inference/predict.py` (`_peek_config`, `_pipeline`, `_command_wrapper`, `_run_predict`, `_run_check`, `CHECK_KEYS`), `inference/cli.py` (`_common`, `resolve_check_chunks`), `slurm/predict_mindboggle.sh`, `tests/test_predict_runner.py` (constants, `compare_modes`, `FIELD_KEYS`).

## Status of the v1 findings

| v1 | Severity | Status | Note |
|---|---|---|---|
| 1 Default bit-identity test tautological | MEDIUM | Closed | 11.1(a) runs `plan(..., normalise=False)` through `_linear_sel` / `_deform_sel` and compares it with `plan=None` using `np.array_equal`. The reason exact equality holds is stated: same cached tf.functions, same chunks, same order. The sha256 golden is kept as 11.1(b). |
| 2 "Default not edited" vs generalised helpers | MEDIUM | Closed | Existing helpers and cache keys are untouched. New `_prewarp_deform_steps_fn`, `_up_steps_fn`, `_prewarp_deform_zero_fn`, `_zero_fields_fn` and `_level_sink_sel` are used only by the `_sel` methods. |
| 3 ckpt-field keys differ between runners | MEDIUM | Closed | One sink rule for both runners (`plan.last == loaded.level`). Truncated cases compare U1 / U1_inv and assert None. `compare_modes` gets `plan`. `max_abs_v` is defined per mode and not compared. |
| 4 L1 `2` missing from tests | MEDIUM | Closed | Added to 11.3 and 11.4. Section 2.2 now matches. |
| 5 `_peek_config` return | NIT | Closed | `_peek_chain_configs` added (4.3, 6.1). |
| 6 Commas in NAME | NIT | Closed | Names are `_L4`, `_L42`, `_L21`, `_L1`. |
| 7 Resumed runs | NIT | Closed | WARNING when args and `cascade.skip_probs` disagree, plus a `runs[*].args_vs_record_mismatch` flag; test 7. |
| 8 `_pipeline` default text | NIT | Closed | Default strings stay as today; test 6 uses literal strings. |
| 9 Constructor signatures | NIT | Closed | Given in 4.2, with keyword `plan` last. |

## New findings (design v2)

### 1. NIT: "joint" exposure is the marginal "at least these levels skipped" probability, not the exact pattern
Section 6.3 sets `joint = prod(p_l for l in K)`. For `2,1` at level 1 this gives 0.15. The share of training steps where level 4 was skipped and level 2 was not is 0.15 x 0.85 = 0.1275, because draws are independent (`cascade_stage_skip_draw`). The brief's example (2.25 % for `1`) does not tell the two apart, because `1` skips every frozen level. The design's number is defensible: the stage did see "level 4 = identity" in 15 % of its steps. Still, the predict.json field name could be misread.
Fix: describe `joint` in section 7 and in the report as "P(all levels in K skipped) = product of p_l; other frozen levels may also have been skipped in those steps". Optionally add `"exact_pattern": prod(p_l) * prod(1 - p_m for run frozen levels m < s)` next to it. The refusal and low-exposure rules stay on `joint`.

### 2. NIT: content of `levels` in the error record when the plan cannot be built
Section 7 says `levels` is "also in error records", and 6.4 says it is "filled" when the run is refused. For the "level not in the checkpoint" error (section 3 / 10) no plan exists, so the content of the record is unspecified. The rule for when `rec["levels"]` is assigned relative to the exposure raise is also implicit.
Fix: state that for a refusal, `rec["levels"] = plan.to_dict()` including `exposure` is assigned before the `PredictError` is raised. For a level not in the checkpoint, `rec["levels"] = {"requested": "<text>", "checkpoint_levels": [...], "selected": null}`. Leave it null when the failure happens before `_peek_config` succeeds.

### 3. NIT: insertion point in `_run_check`
`_run_check` currently calls `_check_unchunked(n, _peek_config(args.checkpoint), args.branches)` inline (predict.py l.611). Section 8 says "builds the plan and the exposure as in `run`" but does not say where.
Fix: bind `peek = _peek_config(...)`, then build the plan and exposure (refusal), then `_check_unchunked(n, peek, ...)`, all before `_read_subjects`. This mirrors `_run_predict`.

## Verified assumptions

- Stage-mode trainer, level 2 (`run.py` l.909-914, l.1031-1035): deform `U, U_inv = CS.skip_fields(skip[0], [U, U_inv])` after `cascade.deform(x)`; linear `A_prev = CS.skip_linear(skip[0], cascade.linear(x))`. Then `prewarp_deform` / `prewarp_linear`, the model, and `compose_cascade` / `compose_linear` run on the identity. This matches design 2.1 and the L2 `2` rows.
- Stage-mode trainer, level 1 (l.915-918, l.1036-1038): `cascade.deform_skip` / `linear_skip` with `skip4` / `skip2`, built only when p > 0 (`_stage_skip_on`). In `deform_skip`, skip4 means U4 = U4_inv = 0 at level 2, the level-2 model runs on `prewarp_deform(x2, 0)`, and the result is `compose_cascade(0, 0, u2, u2i)` upsampled. skip2 gives `up(U4)` = up(up(u4)). Both give up(0) = 0. In `linear_skip`, skip4 gives A4 = `zeros_like + eye`, the level-2 model on `prewarp_linear(x2, I)`, and A = `compose_linear(R2_raw, I, center2)`. skip2 gives A = A4. All 7 level-1 rows of the 2.2 table match this code, for deform and linear.
- The streamed spec reproduces this. The ZERO prev at `lv` uses `warp(pool(x1), zeros at LEVEL_SHAPE[lv])` and P12 with `prev_fn` returning zeros at that level. `StreamedLevel.deform` composes iff `prev_fn is not None`, so `compose_cascade(0, 0, u, u_inv)` runs as in the trainer. `4,1` uses two upsamples of the level-4 field (= `up(U4)`). The linear identity `tf.eye(4, batch_shape=[n])` is bitwise `skip_linear`'s value. In `4,1` the kept `a_prev = A4` equals `skip_to(skip2, A4, A)`.
- Legacy mode (l.920-930, l.1040-1049): at level 2 it skips level 4 for every task. At level 1 it drops level 2 (`skip_to(skip, U4 / A4, U / A)`). So the 6.2 mapping (L2: p(t,4) = p; L1: p(t,2) = p, p(t,4) = 0) is correct.
- Per-stage draws are independent per (seed, step, task, level), so the product rule for "all of K skipped" holds (see NIT 1 for its meaning).
- Truncation: `FrozenStage.deform` = up(u4). `FrozenChain._deform` = up(compose_cascade(up(u4), up(u4_inv), u2, u2i)). `deform_stages` U4 = up(up(u4)). fwd and inv are upsampled independently. The runner's `_up_to_level1` / `_to_level1_fn` do the same. Matrices pass between levels unchanged (`FrozenChain._linear` hands A4 to level 2 as is), so a truncated A goes to `to_canonical` unchanged. This is the cascade's own between-level operation, as the brief requires.
- Default bit-identity: `run()` dispatches `plan is None or plan.default` to the unchanged `_linear` / `_deform` / `_level_sink` / unchunked methods. `predict.py` constructs both runners without `plan`. `--levels 4,2,1` normalises to default. The only predict.json differences for a default run are the new `levels` key; `pipeline` is unchanged.
- `args.cascade_skip_stage_parsed` is filled per task in `task_ratio_parsed` (args.py l.719-722; linear flag for rigid and affine, deform flag for deform) and lands in config.json `args` with string keys. Missing task or level -> 0 is correct.
- `cascade.source_config` is the absolute path of the level-2 source config.json (`load_cascade_source` -> `os.path.abspath(cfg_path)`). `cascade.stages[level 2].origin_checkpoint` is the resolved prefix, so `source_config_path(prefix)` works as the fallback.
- `_command_wrapper` turns `PredictError` into the exit code and still writes the JSON with status error. Exit 2 = `CLI.EXIT_INPUT`.
- `resolve_check_chunks` works per key, so restricting `keys` to run levels is fine. `StreamedRunner` only builds engines for levels it runs.
- `slurm/predict_mindboggle.sh` steps 1a/1b/1c/2 are as described. 1c is non-blocking, so a refused affine+deform record under option (A) does not stop the job. With `LEVELS` / `ALLOW_UNTRAINED_SKIP` unset, appending an empty unquoted `$LEVEL_FLAGS` leaves the commands unchanged.
- The exposure table in 6.3 matches the brief's cluster facts. Deform `1` = 0.0225 (WARNING below 0.05). Linear `2,1` / `1` / `2` are refused (level 4 has p = 0). `4,1` is trained for both tasks. The old chain is all untrained, and truncation is always allowed.
- Test tolerances: the existing `B_EXACT_FALLBACK` 1e-5 and S_stream constants already cover lean vs non-lean warp / compose between the runner and the FrozenChain functions.

## Unverified / wrong assumptions

- Unverified (cluster, not read in this review): that `cascade.source_config` in the frozen copy `scratch/predict_ckpts/r1sw_lr1e-4_wd1e-1_n8_d4_s3000/config.json` points to a readable `results/r2_s15000_lr3e-4_wd1e-1_n32_d16/config.json`. If it does not, the design reports `unknown` and refuses without the flag, which is safe. The implementer should confirm this on the cluster before preparing the ablation lines.
- Unverified: bitwise `diffeo.warp(x, 0) == x`. The design does not rely on it, because it runs the warp literally as the trainer does.
- Nothing wrong found in v2.

## Verdict

0 HIGH, 0 MEDIUM, 3 NIT -> APPROVED.
