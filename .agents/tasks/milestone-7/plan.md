# Milestone 7 implementation plan: three-task training at level 4

Scope: warm start of a deform model from a linear-only checkpoint, loss balance between the linear (mm^2) and deform (Dice) tasks through the one shared Adam, three-task schedule checks, a SLURM script, and docs. No commits, no pushes, no cluster access (the orchestrator does those after review).

## Ground rules for the implementer

- Repo: `d:\Research\Brain_Registration_Projects\brain-group-rad` (no worktree). Use absolute paths.
- Python 3.8 syntax only: `typing.Dict/List/Optional`, no `X | Y`, no `list[int]` at runtime, numpy 1.21 APIs.
- Linear-only runs must stay bit-identical: new behaviour is gated on non-default values with Python-level `if` at trace time, so the default graph gets no extra ops on the update path. Item 1 adds a guard test.
- Don't change Adam 3e-4, LeakyReLU 0.01, dense LayerNorm, the linear variable layout, or the checkpoint object graph.
- Don't edit `.kiro/steering/*`. The orchestrator updates project state.
- Local GPU: never set a TF memory limit above 6000 MB. For `main.py` runs in WSL, pass `--gpu-memory-limit-mb 5000`.

Commands used below (from PowerShell):

```
# targeted TF tests (tf_gpu env, same env vars as tests/run_tests_wsl.sh)
wsl -d Ubuntu bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate tf_gpu && cd /mnt/d/Research/Brain_Registration_Projects/brain-group-rad && export TF_CPP_MIN_LOG_LEVEL=3 TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 BGR_TF_GPU_LIMIT_MB=6000 TF_CUDNN_USE_AUTOTUNE=0 PYTHONDONTWRITEBYTECODE=1 && python -m unittest <module>[.<Class>[.<test>]] -v"
# TF-free tests and py3.8 checks (bgr_preproc38)
wsl -d Ubuntu bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate bgr_preproc38 && cd /mnt/d/Research/Brain_Registration_Projects/brain-group-rad && python -m py_compile <files> && python -c \"import ast,sys; [ast.parse(open(f).read(), f, feature_version=(3,8)) for f in sys.argv[1:]]; print('ast ok')\" <files>"
# full suite (about 90 min; run in the background with a generous timeout)
wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/tests/run_tests_wsl.sh
```

Baseline: 284 tests OK, 4 skipped (commit `0fb998b`).

## Decisions made in this plan

- D1. Partial warm start uses a temporary source model. Build a model with the source run's config, restore it strictly with the existing call (`assert_existing_objects_matched`), then copy per part (`encoder`, `linear_decoder`, `centering`), checking variable count, shape and scope-local name the way `GroupRegLevelModel.copy_weights_from` does. This reuses the strict restore, covers BN moving statistics and LayerNorm (all in `part.weights`), and works in both directions. The other option, a mirrored `tf.train.Checkpoint` proxy, depends on Keras object-graph attribute names, so it's rejected. Allowed config differences: `DECODER_ONLY_FIELDS = ("deform_decoder", "decoder_widths", "integration_steps")`. These are the only ModelConfig fields that change nothing outside `DeformDecoder` or the integrator (checked in `models/config.py` and `models/group_net.py`). Any other difference still raises. If either side lacks a deform decoder, or the decoder fields differ, the deform decoder is never restored: the target's freshly initialised decoder stays, and a source decoder is ignored. Identical configs keep today's single strict restore of the whole model.
- D2. Without a source `config.json`, behaviour is unchanged (strict whole-model restore). A partial warm start needs the source config. The m5 run has one.
- D3. The optimizer state is never read by `--init-weights` (already true: `tf.train.Checkpoint(model=...)` + `expect_partial`, and main.py builds the optimizer after loading). Keep that.
- D4. Loss balance knobs. Keep `--loss-scale-mm2` as the main knob: it divides the linear objective only. Add `--task-loss-weights rigid:a,affine:b,deform:c`, which multiplies each task's objective (default all 1, so nothing changes). Logged loss values stay unscaled.
- D5. Diagnostics. The global clip norm of 12 already shrinks the linear gradients (1e4) on almost every step, so what Adam's second moment sees is the post-clip gradient, not the raw one. Log both, per task, as window means:
  - `train_<task>/diag_enc_grad_norm_raw`: encoder-only gradient norm of the unscaled task loss (before loss scale, task weight, AMP scale and clipping).
  - `train_<task>/diag_enc_grad_norm_eff`: encoder-only norm actually applied (after loss scale and task weight, times the global-clip factor `min(1, clipnorm / grad_norm)`).
  - `train_<task>/grad_norm`: per-task total pre-clip grad norm. `train/grad_norm` stays as it is.
- D6. Balance target: median `diag_enc_grad_norm_eff` of each linear task within a factor of 10 of the deform median (ratio in [0.1, 10]), measured at the warm-started state. Three-task runs are always warm-started, and the linear gradients there are about 30x smaller than at init. The default comes from the local experiment (item 6). The cluster smoke run confirms it before the long run (decision point P2).
- D7. Validation group sizes. No new flag. Linear validation uses `min(--group-size, n_val)` = 32 and deform uses `min(--deform-group-size, n_val)` = 16, as long as `--val-group-size` isn't passed (it would apply to both). The m7 scripts must not pass `--val-group-size`. Deform validation pairs stay capped at 64 (`--val-deform-max-pairs`). Default `--val-groups`: 2 linear groups (ceil(39/32)) and 3 deform groups (ceil(39/16)).
- D8. Warm-started three-task recipe: curriculum OFF, global transform ON from step 0 at 45 deg / 10 mm (no start step, no ramp). Milestone 5 showed warm start + G from step 0 works, and m5's curriculum already ended at the full ranges. LR warmup 500 steps (fresh Adam moments on pretrained weights), Adam 3e-4 cosine, task ratio rigid:1,affine:1,deform:2, linear N 32, deform N 16, val PNGs on, 12000 steps (matches m5).
- D9. BN momentum stays at 0.99. The deform decoder's BN statistics start fresh, so expect `val_deform` (eval) to lag `val_deform_trainmode` early on (milestone-6 note). Report the gap; recalibration is milestone 8.

---

# Implementation Plan

- [ ] 1. Record a linear-only regression fixture BEFORE any code change.
      Add `TestLinearOnlyUnchanged` to `tests/test_trainer.py`, using `TrainerCase`. Model: `narrow_config(dropout=0.0, seed=0)`. Trainer: `config(task_ratio={"rigid": 1.0, "affine": 1.0}, seed=<a seed whose first 6 draws contain both tasks; use the existing `_seed_starting_rigid` pattern or TaskSchedule directly>)`, then `fit(6)`. Record: the per-step `loss` values from `history`, and `float(np.sum(v.numpy().astype(np.float64)))` plus the max abs value for every `model.variables` entry (by `v.name` with the top-level scope stripped, as `GroupRegLevelModel._local_name`). When env `BGR_WRITE_GOLDEN=1` is set, the test writes the fixture to `tests/data/m7_linear_golden.json` (create `tests/data/`). Otherwise it compares against it.
      Generate the fixture on the unmodified code, then run the test twice more. Decision point P0: if the two reruns match the fixture exactly, assert exact equality (`==`). Otherwise assert `max |diff| <= 10 x` the spread you observed, record that spread in the fixture as `"tolerance"`, and note it in the report.
      Files: tests/test_trainer.py, tests/data/m7_linear_golden.json
      Verify: `python -m unittest tests.test_trainer.TestLinearOnlyUnchanged -v` passes (with and without `BGR_WRITE_GOLDEN`). `git diff` shows only the new test and fixture.

- [ ] 2. Partial warm start in `runtime/checkpoint.py` (D1-D3).
      Add `DECODER_ONLY_FIELDS` and `SHARED_PARTS = ("encoder", "linear_decoder", "centering")`, and a new `warm_start_model(model, model_config, path)` that returns an info dict:
      - Resolve the prefix and the source config path (existing helpers).
      - No source config, or no diff (seed excluded): strict whole-model restore exactly as today, `mode="full"`.
      - A diff outside `DECODER_ONLY_FIELDS`: raise the existing `ValueError` message (it must still name the fields, e.g. `translation_scale`).
      - Otherwise, partial restore:
        1. Build `src = GroupRegLevelModel(model.level, src_cfg).build_model(2)`. Import inside the function to avoid a module-level cycle, and do NOT call `clear_session`.
        2. Restore it strictly: `tf.train.Checkpoint(model=src).restore(prefix)`, then `expect_partial()` and `assert_existing_objects_matched()`.
        3. For each part in `SHARED_PARTS`, zip `getattr(src, part).weights` with `getattr(model, part).weights` and raise `ValueError` on a count, shape or local-name mismatch. Only after every check passes, `assign` all of them.
        4. Coverage guard: the ids of `model.weights` must equal the union of the shared parts' weights plus `model.deform_decoder.weights` (if present). Otherwise raise `RuntimeError("unhandled model part ...")`, so a future new part can't be skipped silently.
        5. `del src`.
      - Info keys: `checkpoint`, `source_config`, `mode` ("full"/"partial"), `config_diff` ({field: [source, target]}), `restored_parts` ({part: n_vars}), `fresh_parts` ({"deform_decoder": n_vars} when the target has a decoder that wasn't restored, else {}), `ignored_source_parts` (["deform_decoder"] when the source had one that wasn't used), `n_restored_vars`, `n_fresh_vars`.
      - Keep `load_model_weights(model, model_config, path)` as a wrapper returning `(info["checkpoint"], info["source_config"])`, so existing callers and tests keep working. Update the module docstring.
      Tests, in `tests/test_runtime_infra.py` next to `test_load_model_weights_warm_start` (reuse its `train_two_steps` / `make_run_dir` / `write_config` pattern; save a narrow linear-only model, dropout 0, trained 2 steps):
        - (a) Warm start `narrow_config(dropout=0.0, seed=99, deform_decoder=True)`, built. Snapshot the deform decoder first. Check: every encoder, linear_decoder and centering variable equals the source. At least one encoder BN `moving_mean`/`moving_variance` differs from its init value 0/1, so the check means something. The deform decoder equals the snapshot (still fresh). `info["mode"] == "partial"`, `fresh_parts == {"deform_decoder": len(model.deform_decoder.weights)}`.
        - (b) Same with `decoder_widths=(8, 8, 8, 4, 4)` and `integration_steps=5`: also partial.
        - (c) `deform_decoder=True` + `translation_scale=10.0`: `ValueError` naming `translation_scale`. `deform_decoder=True` + different `filters`: `ValueError` naming `filters`.
        - (d) Reverse direction: save a deform-enabled narrow model, warm-start a linear-only one. Partial, `fresh_parts == {}`, `ignored_source_parts == ["deform_decoder"]`, encoder equal.
        - (e) An optimizer built afterwards has `iterations == 0` and all-zero slots.
        - (f) Identical configs: `mode == "full"`.
      Files: runtime/checkpoint.py, tests/test_runtime_infra.py
      Verify: `python -m unittest tests.test_runtime_infra -v`, all pass (the old `test_load_model_weights_warm_start` passes unchanged). Py3.8 compile check of runtime/checkpoint.py.

- [ ] 3. Wire the warm start into `main.py` and log it.
      In `main()`, replace `C.load_model_weights(...)` with `info = C.warm_start_model(model, model_config, args.init_weights)`. `init_info` becomes the info dict, with `checkpoint`/`source_config` as absolute paths (both keys exist today and the current test reads them), so config.json `init_weights` gains mode/restored/fresh. After `logger` is created, call `logger.log_event(0, dict(event="init_weights", **info))`. Unless `--quiet`, print:
      - `init weights (<mode>) from <prefix>: restored encoder (k vars), linear_decoder (k), centering (k)`
      - `freshly initialised: deform_decoder (k vars)` (or `none`)
      - `config diff: {...}`
      Update the `--init-weights` help in `runtime/args.py`: the deform-decoder fields (`deform_decoder`, `decoder_widths`, `integration_steps`) may differ, and only the encoder + linear heads are restored then. Add a docstring example to main.py: `--task-ratio rigid:1,affine:1,deform:2 --init-weights <linear run>/ckpt/best`.
      Extend `test_init_weights_warm_start` in `tests/test_main.py`: a linear-only `src` (dry run) warm-starts a `--task-ratio rigid:1,deform:1 --deform-group-size 2` dry run, rc 0. config.json `init_weights.mode == "partial"` and `fresh_parts` has `deform_decoder`. metrics.jsonl has an `init_weights` event. The `--translation-scale 7` mismatch still raises.
      Files: main.py, runtime/args.py, tests/test_main.py
      Verify: `python -m unittest tests.test_main.TestMain.test_init_weights_warm_start -v` passes (check the class name in the file). `python main.py --help` works TF-free in bgr_preproc38.

- [ ] 4. Per-task loss weights and encoder grad-norm diagnostics (D4, D5).
      `runtime/args.py`:
      - Add `--task-loss-weights` (default None). Parse it with a new `parse_task_loss_weights(text)`: same syntax as `parse_task_ratio`, tasks in `ALL_TASKS`, weights finite and > 0, missing tasks = 1. Bad values call `parser.error`.
      - Store it as `args.task_loss_weights_parsed` (a dict) and pass it into `TrainConfig.task_loss_weights` from `train_config_from_args`.
      - Update the module docstring defaults list.
      `runtime/run.py`:
      - Add a `TrainConfig.task_loss_weights` field: `Dict[str, float] = dataclasses.field(default_factory=dict)`, with missing = 1.0. Validate it in `Trainer.__init__` (finite, > 0).
      - Add a helper `_task_weight(task)`.
      - In `_make_train_step`: `w = self._task_weight(task)`. After the existing `loss_scale` division, `if w != 1.0: objective = objective * tf.constant(w, objective.dtype)`.
      - In `_make_deform_step`: same multiply on `terms["loss"]`.
      - Both steps, after the static None filter and the AMP unscale:
        - `enc_refs = {v.ref() for v in model.encoder.trainable_variables}` (trace time)
        - `enc_norm = tf.linalg.global_norm([g for g, v in zip(gs, vs) if v.ref() in enc_refs])`
        - `mult` = Python float: `w / loss_scale` for linear, `w` for deform
        - `diag_enc_grad_norm_raw = enc_norm / mult`
        - `clip = getattr(O.inner_optimizer(opt), "global_clipnorm", None)`
        - `diag_enc_grad_norm_eff = enc_norm * (tf.minimum(1.0, clip / tf.maximum(gnorm, 1e-12)) if clip else 1.0)`
      - Return both in `res`. Define `ENC_GRAD_KEYS = ("diag_enc_grad_norm_raw", "diag_enc_grad_norm_eff")` as a new tuple. Do NOT add them to `DIAG_KEYS`/`DEFORM_DIAG_KEYS`: `collapse_diagnostics` is tested against `set(DIAG_KEYS)`.
      - In `fit()`, for finite steps of every task: `acc.update({pre + k: res[k] for k in ENC_GRAD_KEYS})` and `acc.update({pre + "grad_norm": res["grad_norm"]})`. Keep `train/grad_norm` as is.
      - Update the module docstring (train-step pseudo-code and logged keys).
      None of this changes the applied gradients. The multiply is skipped at trace time when w == 1.
      Tests in `tests/test_trainer.py`:
        - (a) `TestLossScale` extension: linear task with `task_loss_weights={"rigid": 4.0}`. The applied gradient is 4x that of w = 1 (same pattern as `test_gradient_scales_and_logged_terms_do_not`). Logged `loss`/`center` are unchanged. `diag_enc_grad_norm_raw` is equal between w = 1 and w = 4 (rtol 1e-5), and `_eff` scales by 4 when unclipped (use `clipnorm=None` or a large clip).
        - (b) Deform step with `task_loss_weights={"deform": 3.0}`: same raw/eff relations, logged `loss` unchanged.
        - (c) Clip factor: with an optimizer clipnorm of 1e-3, `_eff ≈ enc_norm_applied * 1e-3 / grad_norm`.
        - (d) A three-task `fit` with `log_every=1` logs `train_<task>/diag_enc_grad_norm_raw`, `_eff` and `train_<task>/grad_norm`, all finite, for every task.
        - (e) `TestLinearOnlyUnchanged` from item 1 still passes.
      Args tests in `tests/test_runtime_infra.py` (where the deform args are tested, around line 146): parsing, defaults, and bad values (`deform:0`, `foo:1`, `rigid:nan`).
      Files: runtime/args.py, runtime/run.py, tests/test_trainer.py, tests/test_runtime_infra.py
      Verify: `python -m unittest tests.test_trainer tests.test_runtime_infra -v`, all pass, including `TestLinearOnlyUnchanged`. Py3.8 compile check.

- [ ] 5. Three-task schedule and validation sizes (D7), plus `check_mixed_run.py` extensions.
      - `tests/test_main.py`, the existing `_build_dali_data` test (around line 222): also assert the linear validation group size (`calls["linear_val"]` nv == 8 = `--group-size`) next to the deform nv == 4. Add a case `--group-size 32 --deform-group-size 16` with a fake export of 39 val cases (or a stub `load_level_images` returning 39). Expect linear nv 32 with 2 groups and deform nv 16 with 3 groups.
      - `tests/test_trainer.py` `TestThreeTasks`: add a test where the deform group size differs from the linear one. Use `TrainConfig(group_size=N, deform_group_size=2)` with a 2-image deform group; check `DeformCase.setUpClass` for how the deform group is built and slice it to 2. After `fit`, check:
        - `trace_counts` all 1;
        - the deform step's input signature has a leading dim of 2;
        - a validation run writes `val_deform/pair_dice` and `val/mean_pair_rms_mm`;
        - `extra_best == {"deform": DEFORM_BEST_METRIC}`;
        - with a `RunCheckpoint(extra_best=...)`, both `ckpt/best` and `ckpt/best_deform` get a file.
      - `tests/gpu/check_mixed_run.py`: add `check(..., expect_init_weights=None, expect_enc_grad=False)` and CLI flags `--expect-init-weights {none,full,partial}` (default none = not checked) and `--expect-enc-grad {true,false}` (default false).
        - init-weights check: config.json `init_weights` is not null, `mode` equals the expected value, and for `partial` `restored_parts` has `encoder` and `linear_decoder` while `fresh_parts` keys are `["deform_decoder"]` exactly when `deform` is in `--tasks`.
        - enc-grad check: every task has finite `train_<task>/diag_enc_grad_norm_raw` and `_eff`. Print the median per task and the linear/deform `_eff` ratio as an info line (never a failure).
        - Stay stdlib-only and Python 3.8. Defaults keep the old behaviour.
      - Extend `tests/test_check_mixed_run.py`: synthetic run dirs with and without the new keys/config, so both flags pass and fail correctly.
      Files: tests/test_main.py, tests/test_trainer.py, tests/gpu/check_mixed_run.py, tests/test_check_mixed_run.py
      Verify: `python -m unittest tests.test_check_mixed_run tests.test_main tests.test_trainer.TestThreeTasks -v`, all pass. Py3.8 compile of check_mixed_run.py.

- [ ] 6. Local end-to-end smoke test and the loss-balance experiment.
      (a) E2E unit test `test_three_task_warm_start_smoke` in `tests/test_main.py`:
        - Source run: `--overfit-one-group --overfit-synthetic --task-ratio rigid:1,affine:1 --group-size 4 --overfit-max-steps 6 --val-every 3 --log-every 1 --schedule constant --warmup-steps 0 --val-png false` + `NARROW_ARGS` + `COMMON`.
        - Then: `--task-ratio rigid:1,affine:1,deform:2 --group-size 4 --deform-group-size 2 --init-weights <src>/ckpt/last --overfit-max-steps 12 --log-every 1 --val-every 6 --loss-scale-mm2 100 --seed <s>`, where s is a seed whose first 12 `TaskSchedule` draws contain all three tasks (compute it in the test). Expect rc in (0, 1).
        - `check_mixed_run.check(run_dir, ["rigid", "affine", "deform"], 12, expect_best=True, expect_init_weights="partial", expect_enc_grad=True)` returns ok. This covers trace-once, both best checkpoints, PNGs, the warm-start record and the enc-grad keys.
      (b) Experiment script `.agents/tasks/milestone-7/grad_balance_probe.sh` (bash, run through WSL in tf_gpu with the run_tests_wsl.sh env vars, `--gpu-memory-limit-mb 5000`, results under `.agents/tasks/milestone-7/probe_results/`, gitignored or deleted after; keep only the summary JSON):
        1. Linear source: synthetic overfit, rigid:1,affine:1, NARROW_ARGS, `--group-size 4`, `--overfit-max-steps 300`, `--val-every 100`, Adam 3e-4 constant. Note the final center RMS so the warm-start state has a few-mm error, like the m5 checkpoint.
        2. Measurement-only probes (lr 1e-12, `--schedule constant --warmup-steps 0`, so the weights stay put): `--task-ratio rigid:1,affine:1,deform:2 --group-size 4 --deform-group-size 2 --overfit-max-steps 40 --log-every 1 --val-every 0`, with `--init-weights <source>/ckpt/last` and `--loss-scale-mm2` S in {1, 10, 100, 1000}. Add one from-scratch probe (no init) at S = 1 for comparison.
        3. A TF-free `.agents/tasks/milestone-7/summarize_grad_norms.py <run_dir>...` prints per run and task the medians of `train_<task>/diag_enc_grad_norm_raw`, `_eff` and `train_<task>/grad_norm`, plus the linear/deform `_eff` ratio. Write `.agents/tasks/milestone-7/grad_balance.json`.
      Decision point P1 (local default): `M7_LOSS_SCALE_MM2` = the smallest S in the grid whose warm-start ratio, for both linear tasks, is in [0.1, 10]. If none qualifies, pick the S with the ratio closest to 1 and also set `deform:W` in `--task-loss-weights` to cover the remaining factor (W = ratio rounded to a power of 2). Record the table and the choice in the report. Also say whether linear steps were still clipped at the chosen S (`grad_norm` > 12). Synthetic narrow N=4 numbers only show the mechanism; P2 confirms on real data.
      Files: tests/test_main.py, .agents/tasks/milestone-7/grad_balance_probe.sh, .agents/tasks/milestone-7/summarize_grad_norms.py, .agents/tasks/milestone-7/grad_balance.json
      Verify: `python -m unittest tests.test_main.<Class>.test_three_task_warm_start_smoke -v` passes. `wsl -d Ubuntu bash -n` on the probe script. The probe completes and `grad_balance.json` has finite values for every run and task. Delete `probe_results/` run dirs afterwards (keep the JSON).

- [ ] 7. SLURM script `slurm/m7_three_task.sh` (D8).
      Copy the `slurm/train_debug.sh` structure: same singularity exec, binds, TMPDIR/KERAS_HOME/CUDA_CACHE env, `</dev/null`, `set -euo pipefail`.
      Header:
      - `#SBATCH --job-name=bgr-m7-3task -A qtim -p rtx8000 --gpus=1 -N 1 --ntasks-per-node=1 --cpus-per-task=3 --mem=64G --time=12:00:00`
      - output/error under `/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.{out,err}`
      - a comment that it must NOT be submitted without Jay's approval, and never with `entre`.
      Env overrides with defaults:
      - `RUN_NAME=m7_three_task_$JOB_ID`, `STEPS=12000`, `VAL_EVERY=250`, `WARMUP=500`, `N=32`, `DEFORM_N=16`, `TASK_RATIO=rigid:1,affine:1,deform:2`
      - `INIT=$RESULTS/m5_mixed_long_curr_dG_20261019-205526/ckpt/best`
      - `LOSS_SCALE_MM2=<P1 value>`, `TASK_LOSS_WEIGHTS=` (passed only if non-empty)
      - `G_ROT=45`, `G_TRANS=10`, `AMP=false`, `EXTRA_ARGS=`
      Before running, exit 2 with a message if `$INIT` isn't a directory.
      main.py args: `--level 4 --group-size $N --deform-group-size $DEFORM_N --task-ratio $TASK_RATIO --steps $STEPS --val-every $VAL_EVERY --warmup-steps $WARMUP --init-weights $INIT --loss-scale-mm2 $LOSS_SCALE_MM2 --global-transform true --global-rot-max-deg $G_ROT --global-trans-max-mm $G_TRANS --global-start-step 0 --global-ramp-steps 0 --curriculum false --val-png true --amp $AMP --num-threads ${SLURM_CPUS_PER_TASK:-3} --log-every 25 --log-backend jsonl`. Do NOT pass `--val-group-size` (D7).
      After main.py succeeds, run `python tests/gpu/check_mixed_run.py $RESULTS/$RUN_NAME --tasks rigid,affine,deform --steps $STEPS --expect-init-weights partial --expect-enc-grad true` inside the container (as `run_overfit_glacier.sh` does). The script's exit code is main.py's if non-zero, else the checker's.
      Header comment: the smoke variant (`STEPS=200 VAL_EVERY=100 WARMUP=50`) and the time estimates from item 9.
      Add `slurm/m7_three_task.sh` to the `bash -n` loop in `tests/run_tests_wsl.sh`.
      Files: slurm/m7_three_task.sh, tests/run_tests_wsl.sh
      Verify: `wsl -d Ubuntu bash -n /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/slurm/m7_three_task.sh` and `bash -n` on tests/run_tests_wsl.sh both exit 0. Keep LF line endings (check with `git diff --stat` / an editor; no CRLF).

- [ ] 8. Docs: design v3.8 and the milestone report.
      Update `.agents/tasks/network-design/design.md` to v3.8:
      - status line: v3.8 summary;
      - 5.2: partial warm start (D1-D3), loss balance knobs and diagnostics (D4-D6) with the P1 numbers, the three-task recipe (D8), validation group sizes (D7), the BN note (D9);
      - 6.1/6.2: objective = weight x loss (/ loss_scale_mm2 for linear), logs unscaled;
      - 9: the per-task group sizes in one process (linear N 32 ~34 GB, deform N 16 ~33.8 GB on 48 GB; the peak is the larger of the two, not the sum; fallback `DEFORM_N=12`);
      - 11: milestone 7 row "implemented locally; cluster smoke + long run pending".
      Write `.agents/tasks/milestone-7/report.md`, following the milestone-6 report's structure: files changed, decisions, test results (counts), the P0 determinism finding, the P1 table, local smoke results, known gaps, and the cluster commands from item 9.
      Files: .agents/tasks/network-design/design.md, .agents/tasks/milestone-7/report.md
      Verify: the documents name the same flag names and defaults as `python main.py --help` (bgr_preproc38) and `slurm/m7_three_task.sh`.

- [ ] 9. Cluster commands and time estimates (report only; do not run).
      Step time: half linear steps (N 32, ~1.4 s) and half deform steps (N 16, ~3 s, estimated) average ~2.2 s. Validation: 4 linear sets (rigid, affine, rigid_g, affine_g; 2 groups each, eval + train mode) plus deform (3 groups, eval + train mode + identity once), estimated 1-3 min each.
      - Smoke: 200 steps ≈ 7.5 min + ~5 min startup/tracing + 2 validations ≈ 20 min → `--time=00:45:00`.
      - Long: 12000 steps ≈ 7.3 h + 48 validations × ~2 min ≈ 1.6 h ≈ 9 h → `--time=12:00:00`.
      Commands for the report (on mlsc-login after `git pull`; check `squeue -u jn85` shows fewer than 2 active jobs, get Jay's approval first):
      ```
      mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
      RUN_NAME=m7_smoke STEPS=200 VAL_EVERY=100 WARMUP=50 sbatch --job-name=m7smoke --time=00:45:00 --mem=64G slurm/m7_three_task.sh
      RUN_NAME=m7_three_task_long sbatch --job-name=m7long --time=12:00:00 --mem=64G slurm/m7_three_task.sh
      ```
      Decision point P2 (after the smoke run, before the long run): read the checker's `_eff` ratio line. If a linear/deform ratio falls outside [0.1, 10], rescale: `LOSS_SCALE_MM2_new = LOSS_SCALE_MM2 × ratio / 3` when the linear steps were unclipped (`train_<task>/grad_norm` < 12). Otherwise, when they were still clipped, set `TASK_LOSS_WEIGHTS=deform:<ratio/3 rounded to 2^k>`. Pass the new value as an env override. Also check:
      - no `collapse_suspect` events, `train_<task>/diag_feat_rel_spread` > 1e-2;
      - val_rigid/val_affine at step 100/200 not worse than the m5 checkpoint (3.95 / 4.66 mm) by more than ~1 mm. A fresh Adam with warmup may cause a small transient; a large jump means the warm start or the loss scale is wrong;
      - `val_deform/pair_dice` above `pair_dice_identity`;
      - the `val_deform` vs `val_deform_trainmode` gap (D9);
      - peak memory `gpu_peak_mb` < 46000. On OOM, use `DEFORM_N=12`.
      Files: .agents/tasks/milestone-7/report.md
      Verify: the commands in the report match the script's env names and `#SBATCH` header (no `entre`, `-A qtim`, 1 GPU).

- [ ] 10. Final verification.
      - Py3.8 compile/ast of every changed .py (runtime/checkpoint.py, runtime/run.py, runtime/args.py, main.py, tests/gpu/check_mixed_run.py, the changed tests, `.agents/tasks/milestone-7/summarize_grad_norms.py`) in bgr_preproc38.
      - `bash -n` on slurm/m7_three_task.sh and the probe script.
      - Full suite in the background with a generous timeout (~2 h): `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/tests/run_tests_wsl.sh > .agents/tasks/milestone-7/test-log.txt 2>&1`.
      - Delete temporary run dirs and `__pycache__` (the suite script removes pycache).
      Files: .agents/tasks/milestone-7/test-log.txt, .agents/tasks/milestone-7/report.md (test counts)
      Verify: stage 1 OK and stage 2 "OK (skipped=4)" with 284 + new tests. Record the count in the report. `git status` shows only the intended files: runtime/{checkpoint,run,args}.py, main.py, tests/..., tests/data/m7_linear_golden.json, tests/gpu/check_mixed_run.py, tests/run_tests_wsl.sh, slurm/m7_three_task.sh, .agents/tasks/milestone-7/*, design.md.

## Known gaps / assumptions

- The m5 checkpoint's config.json predates the milestone-6 ModelConfig fields. `ModelConfig.from_dict` fills in the defaults (`deform_decoder=False`, `decoder_widths=None`, `integration_steps=7`), so the diff against the m7 config is exactly `{deform_decoder}`. Check this on the cluster in the smoke run's `init_weights` log line.
- Deform step time at N 16 (~3 s) is extrapolated from the N 8 dryrun (1.6 s). The smoke run measures it (`time/compute_ms` split by task isn't logged; estimate it from total wall time and the task mix).
- The local loss-balance numbers (narrow, synthetic, N 4/2) only show the mechanism. The real default is confirmed at P2.
