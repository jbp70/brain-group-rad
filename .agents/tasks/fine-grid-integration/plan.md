# Implementation Plan: optional fine-grid integration of frozen cascade stages

Worktree: `d:\Research\Brain_Registration_Projects\brain-group-rad\.worktrees\fine-grid-integration`
(branch `fine-grid-integration`, base `ee8a88a`). Always use absolute paths / `git -C <worktree>`;
relative paths land in the main checkout, which another agent is editing. Never push.

## Design decisions (made here; the coder does not re-decide them)

D1. Names. Training flag `--frozen-integration-grid {native,current}` (runtime/args.py, cascade
group), argparse default `None` = "inherit" (resolved in main.py, D4). Inference flag
`--integration-grid {auto,native,fine}` (inference/cli.py, `run` and `check-streaming` only),
default `auto` = follow the checkpoint (`current` -> `fine`). `native` = today's behaviour everywhere.
Rationale: training names the frozen chain's grid relative to the level being trained; inference
names the output grid. Constants `models.cascade.INTEGRATION_GRIDS = ("native", "current")`.

D2. Where the switch lives: inside `models/cascade.py` (`FrozenStage` / `FrozenChain` get an
immutable `integration_grid` attribute, set by every factory and the constructor, kwarg default
"native"). Every trainer path (train steps incl. stage / legacy skip, validation incl.
val_deform_frozen / frozen4, BN recalibration via `Trainer._prewarp` (run.py 1790-1799), eval_run,
UnchunkedRunner) only calls `cascade.deform / deform_stages / deform_skip`, so `runtime/run.py`
needs NO change. Native branches stay verbatim (Python `if` at trace time), so native graphs and
outputs are bit-identical.

D3. Current-mode math (all float32, under `tf.stop_gradient`, frozen models `training=False`):
* New `diffeo.exp_compose_upsampled(vs, ups, steps, lean=False, return_first=False)`: per image
  (`tf.map_fn(..., parallel_iterations=1, fn_output_signature=...)`), for each SVF v_k (coarse
  first) upsample `ups[k]` times with `upsample_displacement`, `exp_and_inverse(., steps)`, and fold
  `acc = compose(acc, u_k)`, `acc_inv = compose(u_k_inv, acc_inv)` (= `compose_cascade` order).
  Returns (u, u_inv) [+ (u_0, u_0_inv) of the first SVF alone if `return_first`]. All upsampled
  shapes must match (ValueError otherwise). Per-image execution is the level-1 memory strategy
  (D6).
* New `cascade.integrate_frozen_at(svfs, level, steps, return_first=False)`: `svfs` = list of
  `(level_k, v_k)` coarse first, v_k in level_k voxels; ups_k = log2(level_k / level); empty list ->
  zero fields is NOT supported (callers handle "nothing to compose").
* `FrozenStage` (current): `out = model(pool(x), deform, False)`; `v4 = out["v"]` (already centred);
  `U, U_inv = integrate_frozen_at([(4, v4)], 2, T)`; `V = up(v4)` (unchanged). New
  `FrozenStage.svf(x)` -> stop_gradient(v4) (any mode).
* `FrozenChain` (current): `svfs(x1, skip4=None)` -> `{4: v4', 2: v2}` where v4' = skip_fields(skip4,
  v4) if skip4 else v4, x2_in = prewarp_deform(x2, integrate_frozen_at([(4, v4')], 2, T).U)
  (= what the level-2 run saw in training), v2 = model2(x2_in)["v"]. In native mode `svfs` uses the
  native `stage4.deform(x2)["U"]` for the pre-warp (so it is valid in both modes).
  `deform`: U, U_inv = integrate_frozen_at([(4, v4), (2, v2)], 1, T); V = up(v2) (unchanged).
  `deform_stages`: one call with `return_first=True` -> U, U_inv, U4, U4_inv; V4 = up(up(v4)).
  `deform_skip(skip4, skip2)`: v4' as above, v2' = skip_fields(skip2, v2) if skip2, then
  integrate_frozen_at([(4, v4'), (2, v2')], 1, T) (exp(0) = 0 and compose with 0 are exact, so a
  skipped stage is the identity, as in native semantics; no `cascade_skip` op without skips).
  T = `stage4.model.config.integration_steps` (chain configs are equal).
* The trainable level's own SVF is integrated by the model on its own grid and composed by the
  trainer exactly as today.

D4. Chain rule (pure function `runtime.args.resolve_frozen_integration_grid(requested,
source_level, source_cascade, saved_cascade=None)`, TF-free):
* without `--cascade-from`: `current` -> parser error ("needs --cascade-from"); `native`/None OK
  (recorded as native).
* fresh level-2 run (level-4 source, no cascade record): requested or "native".
* fresh level-1 run: source value = `upstream_cascade.get("frozen_integration_grid", "native")`;
  None -> inherit it; explicit value != source value -> ValueError -> SystemExit naming both values
  ("the frozen level-2 stage was trained with ..."). No override flag (a mixed chain would feed the
  frozen level-2 model inputs it never saw).
* resume: saved = `saved_cascade.get("frozen_integration_grid", "native")`; None -> saved; explicit
  != saved -> RuntimeError (same style as `_check_cascade_fingerprint`).
* main.py sets `args.frozen_integration_grid = resolved` before `write_config` (config.json args)
  and writes `cascade_rec["frozen_integration_grid"] = resolved` on every cascade run (fresh L2, fresh
  L1, carried verbatim on resume by `_refresh_skip_record`). `_cascade_lines` prints it. eval_run
  reads it from the saved record (missing = native).

D5. Inference: `loader.load_model(..., integration_grid="auto")`; checkpoint value = cascade record
key, else args key, else "native"; used = {auto: native->native, current->fine; explicit as given};
explicit != checkpoint -> `logger.warning` (experiment). Frozen built with
`integration_grid="current" if used == "fine" else "native"`. `LoadedModel(integration_grid="native")`
attribute + `knobs["integration_grid"] = {"checkpoint": <native|current>, "used": <native|fine>}`.
Runners branch on `loaded.integration_grid == "fine"` into NEW methods (`_deform_fine`); native
methods untouched. Fine semantics: each level's model input pre-warped with the frozen prior
integrated at that level's grid (as in training); level sink: `u_ckpt / u_inv_ckpt` = the
checkpoint-level composition the trainer computes (prior at ckpt grid o exp(v_ckpt)); `U1, U1_inv`
= ALL run levels integrated and composed at level 1 (probe variant C): level 1 ckpt -> d["u"]; else
`integrate_frozen_at(svfs + [(ckpt_or_last_level, v)], 1, T)`. Level selection: skipped levels are
omitted from `svfs` (identity, exact), truncated levels not run; a level preceded only by skipped
levels gets the trainer's skip input `warp(pool(x), 0)` (reuse `_prewarp_deform_zero_fn` /
`_zero_fields_fn` in streamed; `deform_skip`-style zeros in unchunked). Level-4 checkpoints have no
frozen stage: `fine` still gives U1 = exp1mm(up(up(v4))) (probe L4f). TTA: unchanged engine;
`_deform_stage` / `_deform_stage_freeing` are reused with the fine `input_fn` / `prev_fn` / sinks.
linear-eval: no flag (linear only).

D6. Memory strategy (level 1, ~2-3 GB headroom under 42500 MB): frozen work stays outside the
tape; 1 mm integrations run per image inside `tf.map_fn(parallel_iterations=1)` so only one
[1,160,224,192,3] float32 field set (~83 MB each, ~1-1.5 GB transient for exp + compose) is live at a
time; the held outputs (U, U_inv and, in validation, U4, U4_inv) have the same sizes as today; the
level-2-grid pre-warp field is small. Expected TF peak change: L2 (deform N16) <= +0.2 GB, L1
(deform N4) 0 to +1.5 GB (transient precedes the forward / backward peak; fragmentation is the risk
measured by the job). Expected time: L1 +1-3 s per deform step, L2 < +0.5 s. Not measured locally.

D7. Measurement path: real trainer (the measure tool under-predicts by 10-15 % and has no DALI):
new `slurm/fgi_memory.sh` (rtx8000, qtim, 1 GPU, 3 CPUs, 128G, 03:00:00). Configs in order
`l2_native l2_current l1_native l1_current`; each = the r_level2.sh / r_level1.sh recipe obtained
from the level script's `BGR_DRY_RUN=1` MAIN line (so the recipe cannot drift), with RESULTS =
`$OUT/runs`, RUN_NAME = config name, STEPS 16, VAL_EVERY 8, CKPT_EVERY 0, BN_RECAL_BATCHES 4,
`EXTRA_ARGS="$GROUP_ARGS --lambda-smooth $S"` (GROUP_ARGS default `--group-config B`, S 10), and
FROZEN_INTEGRATION_GRID=current for the *_current lines; L2 lines cascade from L4_SOURCE (default
`$PROJ/results/r4_cfgB_dice1_smooth2_lr3e-4_wd1e-1/ckpt/last`), l1_native from
`$OUT/runs/l2_native/ckpt/last`, l1_current from `$OUT/runs/l2_current/ckpt/last` (inherits
current). One singularity process per config under `timeout -k 60 $CONFIG_TIMEOUT` (3600), nvidia-smi
poll into `runs/<name>/gpu_poll.csv`, status.tsv in the collector's columns (dtype column = grid),
an L1 line whose L2 source failed gets rc -2; then `scripts/collect_memory_runs.py` ->
`$OUT/fgi_memory.json`. ONLY=<names> selects lines; BGR_DRY_RUN=1 prints `CONFIG <name>: <env> bash
slurm/r_level<k>.sh` lines and exits 0. Output `$PROJ/scratch/fgi/<job>/`.

D8. Scripts: `FROZEN_INTEGRATION_GRID` env in r_level2.sh / r_level1.sh (empty default = no flag;
native|current -> `--frozen-integration-grid V`, else exit 2). r_cascade.sh: add the key to
`LEVEL_ENV_KEYS` only (needed by `test_level_env_coverage`; stale values never leak); chain use =
`EXTRA_ARGS_L2="--frozen-integration-grid current"`, level 1 inherits (D4). No default changes.

D9. Tests: new `tests/test_fine_grid_integration.py` (TF; `import tests.tf_setup` first; reuses
`tests.test_cascade` cases/helpers and `tests.test_predict_runner` helpers), additions to
`tests/test_slurm_scripts.py`; `fgi_memory.sh` added to the `bash -n` list in
`tests/run_tests_wsl.sh`. Goldens (`tests/test_predict_tta_golden.py`, `tests/test_trainer.py`
m7 linear + lncc deform) must pass unchanged (0 diff).

## Items

- [ ] 1. diffeo helper `exp_compose_upsampled` (D3) with module-docstring row "Fine-grid frozen
      integration".
      Files: models/diffeo.py, tests/test_fine_grid_integration.py (class TestExpComposeUpsampled)
      Verify: WSL `python -m unittest tests.test_fine_grid_integration.TestExpComposeUpsampled -v` —
      single SVF == exp_and_inverse(up^k v) (atol 1e-5), two SVFs == manual compose order, return_first,
      shape-mismatch ValueError, map_fn result == batched result (atol 1e-6).

- [ ] 2. cascade.py: `INTEGRATION_GRIDS`, `integrate_frozen_at`, `integration_grid` kwarg on
      FrozenStage/FrozenChain + all factories (`from_checkpoint`, `from_model`, `from_config`,
      `from_models`; chain propagates to its stage4), `FrozenStage.svf`, `FrozenChain.svfs`, current
      branches of `deform`, `deform_stages`, `deform_skip` (D3); docstring table row.
      Files: models/cascade.py, tests/test_fine_grid_integration.py (TestFrozenCurrent)
      Verify: new tests: L2 FrozenStage current == manual reference; L1 chain deform / deform_stages /
      deform_skip (skip4, skip2, both) == manual (pool, v4, exp at 2 mm pre-warp, model2, v2, exp at
      1 mm, compose); V / V4 unchanged; no gradient to x (tape); native outputs bitwise == the old
      formulas and the native traced graph has no While op; ValueError for a bad grid value; then
      `python -m unittest tests.test_cascade -v` passes unchanged.

- [ ] 3. Training wiring: args flag + `resolve_frozen_integration_grid` + parse check (D4);
      main.py fresh / resume resolution, `args.frozen_integration_grid` and
      `cascade_rec["frozen_integration_grid"]`, factories called with the resolved value,
      `_cascade_lines`, module docstring; runtime/eval_run.py passes the saved value.
      Files: runtime/args.py, main.py, runtime/eval_run.py, tests/test_fine_grid_integration.py
      (TestChainRule pure; TestMainCascadeGrid: overfit-mode L4 -> L2 `current` -> config.json args +
      cascade record; L2 resume w/o flag keeps current, with `native` -> RuntimeError; a Trainer with
      a current FrozenStage: one deform step finite, frozen weights unchanged, validation frozen
      fields == frozen.deform)
      Verify: `python -m unittest tests.test_fine_grid_integration tests.test_cascade tests.test_main -v`.
      Level-1 main runs are NOT executed locally (6 GB cap); the L1 rule is covered by the pure
      resolver tests and by `_load_cascade_source` info (upstream_cascade) in a unit test with a
      fake config.json.

- [ ] 4. Inference: loader (D5), LoadedModel attribute + check frozen grid consistency, cli flag,
      predict.py `_load` passes it (run + check-streaming), StreamedRunner `_deform_fine` (default
      and level plans, TTA via `_deform_stage*`, affine+deform src freeing), UnchunkedRunner
      `_deform_fine` (default and plans) using `frozen.svfs` + `CS.integrate_frozen_at`.
      Files: inference/loader.py, inference/runner.py, inference/cli.py, inference/predict.py,
      tests/test_fine_grid_integration.py (TestInferenceFine)
      Verify: new tests on synthetic tiny chains: L2 `fine` streamed == unchunked; level-2 model input
      == training pre-warp; u_ckpt == trainer composition; U1 == manual all-at-L1; `--levels 4`
      truncation and skip `2` (allow-untrained) == manual; TTA with IDENTITY_ONLY in fine == no TTA;
      L1 chain fine streamed == unchunked (N 2, lean); loader: old config -> native, current record ->
      fine, explicit override warns; then `tests.test_predict*` all pass (TTA golden 0 diff).

- [ ] 5. Scripts + measurement: FROZEN_INTEGRATION_GRID in r_level2.sh / r_level1.sh, LEVEL_ENV_KEYS
      in r_cascade.sh, new slurm/fgi_memory.sh (D7), run_tests_wsl.sh bash -n entry, slurm tests.
      Files: slurm/r_level2.sh, slurm/r_level1.sh, slurm/r_cascade.sh, slurm/fgi_memory.sh,
      tests/run_tests_wsl.sh, tests/test_slurm_scripts.py
      Verify: TF-free `python -m unittest tests.test_slurm_scripts -v` in bgr_preproc38 (WSL): unset
      knob -> MAIN unchanged; current -> one `--frozen-integration-grid current`; bad -> exit 2;
      fgi dry run lists 4 CONFIG lines in order, L1 lines cascade from runs/l2_*/ckpt/last, missing
      L4_SOURCE exits 2, header policy lines (qtim, rtx8000, --gpus=1, 3 CPUs, 128G, approval note).

- [ ] 6. Docs + integration: design.md 4.7 addendum (rule + semantics), report
      `.agents/tasks/fine-grid-integration/report.md` (decisions, flags, chain rule, memory estimate,
      exact sbatch commands: memory job and L2 smoke, commit hashes, test results), full WSL suite
      against the worktree (background, ~3 h+), py3.8 syntax check (part 1 of the suite).
      Verify: `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.worktrees/fine-grid-integration/tests/run_tests_wsl.sh`
      -> `suite status: 0`; goldens 0 diff.

Report sbatch commands (cluster clone, after the merge + push by the orchestrator):
```
mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
cd /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad && git pull --ff-only
sbatch slurm/fgi_memory.sh
P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
RUN_NAME=r2_fgi_smoke CASCADE_FROM=$P/results/r4_cfgB_dice1_smooth2_lr3e-4_wd1e-1/ckpt/last FROZEN_INTEGRATION_GRID=current STEPS=200 VAL_EVERY=100 WARMUP=50 PROJECT_STEPS=15000 PROJECT_VAL_EVERY=500 WD=1e-1 GPU_POLL=true EXTRA_ARGS="--group-config B --lambda-smooth 10" sbatch --job-name=r2fgism --time=03:00:00 --mem=64G slurm/r_level2.sh
```
(Jay picks the smoothness; 10 = his chosen L2 value. Ask Jay before any submission; 2-job limit.)

## Digest

Verified today: worktree at ee8a88a, clean. Test runner `tests/run_tests_wsl.sh` derives REPO from
its own path (`REPO="$(cd "$(dirname "$0")/.." && pwd)"`), so it runs against the worktree when
called by the worktree path. Part 1 = py3.8 py_compile + ast(3.8) of models/ runtime/ inference/
main.py tests/ ..., `bash -n` on a fixed list of slurm scripts, TF-free tests (incl.
tests.test_slurm_scripts); part 2 = one process per tests/test_*.py module in tf_gpu with
BGR_TF_GPU_LIMIT_MB=6000. Shell quoting: `.kiro/skills/shell-and-remote-jobs/SKILL.md`.

- models/diffeo.py: `exp_and_inverse(v, steps=7, lean=False)`, `compose(u_a, u_b, lean)` (=
  u_b + warp_field(u_a, u_b)), `upsample_displacement(u)` = 2 * trilinear x2, `compose_cascade(U, Ui,
  u, ui)` = (compose(U, u), compose(ui, Ui)). Add the new function after `compose_cascade`.
- models/cascade.py: `FrozenStage.__init__(model)`, `deform(images_fine)` (model on pool(x), up of
  u/u_inv/v, stop_gradient), factories `from_checkpoint(prefix, cfg)`, `from_model(model)`,
  `from_config(cfg)`; `FrozenChain.__init__(stage4, model2)`, `from_checkpoint(prefix, cfg,
  expected_fingerprint4)` (builds `FrozenStage(model4)`), `from_config`, `from_models`, `_deform`
  (x2 = pool, p4 = stage4.deform(x2), model2 on prewarp_deform(x2, p4["U"]), compose_cascade at L2,
  up to L1), `deform`, `deform_stages` (adds U4/U4_inv/V4 = up(p4[...])), `deform_skip(images_l1,
  skip4, skip2)`; helpers `skip_fields`, `skip_to`, `prewarp_deform`. Covered by tests/test_cascade.py
  (TestDeformCascade, TestFrozenChain, TestChainTrainer, TestStageSkipChain, TestStageSkipLevel2;
  `ChainCase.chain()` / `frozen_models()` / `CascadeTrainerCase` are reusable).
- runtime/run.py: NO change. Cascade calls: deform step 950-972 (`cascade.deform`,
  `cascade.deform_skip`, `cascade.deform_stages` + `CS.skip_to` / `CS.skip_fields`), validation
  1425 (`deform_stages` / `deform`), `_prewarp` 1790-1799 (BN recal).
- runtime/args.py: cascade group 534-575 (`--cascade-from`, skips, `--bn-recal-*`); checks
  730-782 (`cascade = args.cascade_from is not None`, `parser.error`). Module TF-free (run_tests
  asserts it).
- main.py: cascade decision 263-278; frozen construction 306-370 (`FrozenStage.from_config` /
  `FrozenChain.from_config` on resume, `FrozenChain.from_checkpoint(prefix, src_cfg,
  upstream["fingerprint"])` fresh L1 with `upstream = src_info["upstream_cascade"]`,
  `FrozenStage.from_checkpoint(prefix, src_cfg)` fresh L2), `cascade_rec` dicts 333-345 / 362-366,
  resume `_refresh_skip_record` 811 (copies the saved record), `write_config(..., A.args_to_dict(args),
  extra=config_extra)` ~427, `_cascade_lines` 876. Tests: tests/test_cascade.py TestCascadeResume
  (~1370-1460, overfit-mode `MAIN.main(common + [...])` L4 -> L2 pattern).
- runtime/checkpoint.py `load_cascade_source(path, level)`: level-1 info has `upstream_cascade` =
  the L2 source's cascade record verbatim (no change needed).
- runtime/eval_run.py 169-183: `cascade = raw_cfg.get("cascade")`, frozen via from_config.
- inference/loader.py `load_model(checkpoint_path, branches, precision, device)` builds
  `CS.FrozenStage.from_config(model_cfg)` / `CS.FrozenChain.from_config(model_cfg)` (285-287),
  `_knobs(args)`, `LoadedModel.__init__(level, model, frozen=None, ..., precision=None)`.
- inference/runner.py: StreamedRunner `_deform` (424-461), `_level_sink` (492-507), `_deform_sel`
  (601-672), `_level_sink_sel`, `_prewarp_deform_zero_fn`, `_zero_fields_fn`, `_deform_stage_freeing`,
  `_deform_masks`, `run` / `_run_sel` dispatch; UnchunkedRunner `_deform` (780-803), `_deform_sel`
  (892-947), `run` / `_run_sel`. `StreamedLevel.deform(n, input_fn, prev_fn=None, sink=None)`
  (inference/streaming.py) composes with prev_fn in P12 and hands {u, u_inv, v, v_raw} to the sink.
  Plans: `inference.levels.LevelPlan` (selected, last, run, skipped, truncated, kind(lv), default).
  Tests: tests/test_predict_runner.py (RunnerCase, `_level2`, `_level4`, `_full_res`, `_collect`),
  test_predict_streaming.py, test_predict_levels_runner.py, test_predict_tta*.py (golden).
- inference/cli.py `_common(p, default_branches, chunk_defaults, branches=True)`; linear-eval passes
  `branches=False`. inference/predict.py `_load(args)` -> `LD.load_model(args.checkpoint,
  args.branches, precision=..., device=...)` (~474); predict.json `knobs=loaded.knobs`.
- slurm/r_level2.sh / r_level1.sh: env block (`EXTRA_ARGS`, `BGR_DRY_RUN` prints `MAIN: python ...`
  and exits 0 after validation incl. the CASCADE_FROM guard), `need_choice`, MAIN array. r_cascade.sh
  `LEVEL_ENV_KEYS=(...)` 159-163. tests/test_slurm_scripts.py: TestStubRuns (`run_script`,
  `train_env`), TestM11Job (dry-run line parsing with `runtime.args`), TestCascadeStatic
  `test_level_env_coverage`. scripts/collect_memory_runs.py: status.tsv columns `name level dtype
  widths n max_pairs rc elapsed_s log`; rc -2 -> skipped_source.
