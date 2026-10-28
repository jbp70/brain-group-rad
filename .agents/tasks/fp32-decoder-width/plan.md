# Implementation plan: float32 deform decoder (+ float32_fine), narrower widths, m11 memory job

Repo `d:\Research\Brain_Registration_Projects\brain-group-rad`, branch main, no worktree. Do NOT
commit, push or submit anything. Base commit ddb91f8 (nonfinite-fixes).

> **ORCHESTRATOR REVISION (Jay's decision, overrides D3 / item 5 and every mention of
> `--cascade-trainable-init fresh`):** no such flag. For non-default widths build random-weight
> level-4 and level-2 cascade sources at that width with a minimal main.py run (no meaningful
> training), then measure level 1 through the normal `--cascade-from` copy path. This also makes the
> frozen stages narrow, as in the planned retrain. See the revised
> `task-fp32-decoder-width/features/FEAT-002.json` and the override note in FEAT-003.json.

## Findings that shape the plan (read from the code)

- `--deform-decoder-dtype` exists with choices `policy` (default) | `float16` | `float32`
  (`runtime/args.py` ~403, `models/group_net.py` `DEFORM_DECODER_DTYPES`). `float32` builds the
  whole `DeformDecoder` inside `models.precision.policy_scope("float32")`. `policy` stays the
  default; `float16` stays an explicit alias of the policy under AMP (same graph, existing test).
- Per-block widths ALREADY exist: `--filters` (6 values: stem, L2..L6) and `--decoder-widths`
  (5 values: D5..D1), both `ModelConfig` fields stored in config.json and compared by warm start
  (`DECODER_ONLY_FIELDS`) and the cascade (`check_cascade_config`, all fields). `svf_norm_act` has
  no width of its own (it normalises the d1 output, D1 channels); `svf_head` always outputs 3.
  Named presets live in `models/width_presets.py` (`default`, `f0d1`, `fr50`, ...). => NO new width
  flags (TASK 2 = documentation). Grid widths:
  - default: no width flags (ModelConfig default; `decoder_widths=None` is NOT dict-equal to an
    explicit `(128,96,64,32,16)`, so never pass the explicit default list to a cascade run);
  - d12h (d1/d2 halved): `--decoder-widths 128,96,64,16,8`;
  - d12hs (d1/d2 halved + stem halved): `--filters 16,64,128,192,256,384 --decoder-widths 128,96,64,16,8`;
  - fr50 (= width preset fr50): `--filters 16,32,128,192,256,384 --decoder-widths 128,96,64,16,8`.
- Cascade: `main.py` -> `_load_cascade_source` -> `C.check_cascade_config` requires an identical
  ModelConfig (seed aside), then `model.copy_weights_from(frozen.source_model)`. Frozen stages are
  built from the SOURCE config (`FrozenStage.from_checkpoint(prefix, src_cfg)`,
  `FrozenChain.from_checkpoint(prefix, src_cfg, fp)`) and always in float32
  (`cascade._build_frozen_model`). So a narrower trainable stage only fails at the config check and
  the weight copy. Minimal way (decision D3 below): opt-in `--cascade-trainable-init copy|fresh`.
- Trainer metrics (`runtime/run.py` ~2100): every log record has `gpu_peak_mb` (TF allocator peak,
  MiB), `time/data_ms`, `time/compute_ms`; `--log-every 1` gives one record per step;
  `train_deform/n_pairs` with `--min-pairs n` / `--max-pairs`. config.json has
  `param_counts.trainable`. metrics.jsonl lines are `{"step","time","data"}`.
- `--val-every 0`: no validation at all (the final-step validation needs val_every > 0); validation
  sets are still built (realistic memory). `--ckpt-every 0`: only the final ckpt/last save.
- NIT 1: `fit` sets `lso_flag = amp and consecutive_task[task] >= LSO_SKIP_STREAK - 1`
  (`runtime/run.py` ~2017); `save_streak[task]` already counts hard failures only.

## Decisions

- D1 float32_fine: d2, d1, their upsample layers `up_d2` / `up_d1` (when they exist), `svf_norm_act`
  constructed under the float32 policy scope; `svf_head` is already float32. `DeformDecoder` itself,
  d5..d3 and up_d5..up_d3 keep the global policy. In `call`, cast x to each stage block's compute
  dtype BEFORE its upsample (Keras autocasts float inputs to the layer dtype, so the upsample layer
  must be float32 too), skips are cast to x.dtype (existing). `tf.cast` to the same dtype is a no-op,
  so policy/float16/float32 graphs are unchanged (goldens bit-identical). Rationale: Jay's grid; the
  probe's first inf was in d1 with d2 conv max 9640.
- D2 frozen stages: always float32 (unchanged; `deform_decoder_dtype` is a runtime attribute of the
  trainable model, never a ModelConfig field, so no flag and nothing stored per stage). Documented in
  `models/cascade.py` and the group_net docstring. Rationale: the probe ran them fp32, they never
  train, and fp32 avoids overflow in inference.
- D3 `--cascade-trainable-init copy|fresh` (default copy = today, bit-identical). `fresh`: the run's
  ModelConfig may differ from the source ONLY in `CASCADE_FRESH_FIELDS = ("filters", "decoder_widths")`;
  frozen stage(s) still come from the source checkpoint with the source config; the trainable stage
  keeps its random init (no `copy_weights_from`); `init_weights` record mode `fresh`; cascade record
  gets `trainable_init: "fresh"` and `trainable_config_diff`. Needs `--cascade-from` and
  `--resume false` (parse errors otherwise; the resume path rebuilds frozen stages from the run's
  own config). Help text: memory/time measurement only; Jay's rule stands that production level-1
  stages must match the frozen levels' widths (a width change means retraining all levels).
- D4 m11 job: real trainer (`main.py`), deform-only ratio `deform:1` (+ `--global-transform false
  --curriculum false`), 8 steps, `--log-every 1`, `--val-every 0`, `--ckpt-every 0`,
  `--bn-recal-batches 0`, `--resume false`, `--warmup-steps 0`, AdamW lr 3e-4 wd 0, `--task-grad-norm 12`,
  Dice+pDice+LNCC 1/1/1/1 + smooth 0.5, p 0.1/0.1, `--lowres-method tent`, `--intensity-aug-device tf`,
  `--val-png false`. One config per singularity process, `timeout` per config, ckpt dir deleted
  after each config, results summarised by a new stdlib collector `scripts/collect_memory_runs.py`.
- D5 NIT 1: code fix `lso_flag = amp and save_streak[task] >= LSO_SKIP_STREAK - 1` (identical for
  every existing test sequence; differs only for "gradient-only overflow then forward NaN"). NIT 2:
  documentation only.

## Items

- [ ] 1. Test helpers (local only, not part of the deliverable).
      Create `.agents/tmp/fp32w_tests.sh` (activate tf_gpu with the env of `tests/run_tests_wsl.sh`:
      TF_CPP_MIN_LOG_LEVEL=3, TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512, BGR_TF_GPU_LIMIT_MB=6000,
      TF_CUDNN_USE_AUTOTUNE=0, BGR_DEFER_MEMORY_TESTS=1; run each module given as argument in its own
      `python -m unittest <module> -v` process; exit non-zero on any failure) and
      `.agents/tmp/fp32w_py38.sh` (bgr_preproc38: py_compile + `ast.parse(feature_version=(3, 8))` of
      the changed Python files, `bash -n` of the changed scripts, the TF-free test modules
      `tests.test_slurm_scripts tests.test_sweep_tools tests.test_collect_memory_runs` with the
      "tensorflow not imported" assert). Use the `shell-and-remote-jobs` skill for invocation.
      Files: `.agents/tmp/fp32w_tests.sh`, `.agents/tmp/fp32w_py38.sh`
      Verify: `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.agents/tmp/fp32w_tests.sh tests.test_fp16_mitigations` passes on the unchanged code.

- [ ] 2. float32_fine mode (TASK 1, D1, D2).
      `models/group_net.py`: `DEFORM_DECODER_DTYPES = ("policy", "float16", "float32", "float32_fine")`,
      new `FLOAT32_FINE_STAGES = ("d2", "d1")`. `DeformDecoder.__init__(level, config=None,
      float32_stages=(), **kwargs)`: build `up_<name>` and `<name>` inside
      `precision.policy_scope("float32")` for names in float32_stages, and `svf_norm_act` too when
      float32_stages is non-empty (svf_head unchanged). `DeformDecoder.call`: per stage
      `x = tf.cast(x, block.compute_dtype)` before the upsample; `x = tf.cast(x, self.svf_norm_act.compute_dtype)`
      before the head; keep the initial `tf.cast(skips[5], self.compute_dtype)`.
      `GroupRegLevelModel.__init__`: float32_fine and global compute != float32 -> build the decoder
      with `float32_stages=FLOAT32_FINE_STAGES` (no outer scope); float32_fine under a float32 global
      policy = no-op (like float32). Docstrings: GroupRegLevelModel (modes, variables identical in all
      modes, frozen stages always float32), DeformDecoder; `models/cascade.py` "Float32" table row
      (the trainable stage's --deform-decoder-dtype never applies to frozen stages).
      `runtime/args.py`: choices + module docstring bullet + help; `float32_fine` needs 'deform' in
      --task-ratio (same rule as float32), non-policy needs --amp (existing).
      Scripts `slurm/r_level4.sh`, `r_level2.sh`, `r_level1.sh`, `slurm/m9_level1_cascade.sh`:
      accept `float32_fine` in DEFORM_DECODER_DTYPE (need_choice / case + error text), linear-only
      NOTE drop also for float32_fine, header comment lines (float32_fine = fp32 only d2/d1/svf).
      Files: models/group_net.py, models/cascade.py, runtime/args.py, slurm/r_level{4,2,1}.sh,
      slurm/m9_level1_cascade.sh
      Verify: item 3 tests.

- [ ] 3. Tests for item 2 (closes nonfinite-fixes review NIT 3).
      `tests/test_fp16_mitigations.py`: new `TestFloat32FineDecoder(MixedPolicyCase)`:
      (a) compute dtype per submodule for each mode on the level-2 repro model AND a level-1 TINY
      model (has up_d1): policy/float16 -> every decoder layer float16 except svf_head float32;
      float32 -> all float32; float32_fine -> d2, d1, up_d2, up_d1 (when not None), svf_norm_act and
      all their submodules, svf_head float32; DeformDecoder, d5..d3, up_d5..up_d3 and submodules
      float16; encoder float16; global policy restored; (b) repro overflow (`overflow_weights(k=1e5)`,
      d2 kernel x 1e5): v_raw / u / u_inv finite in float32_fine while the policy model overflows;
      without overflow v_raw within 2e-2 * max|v32| + 1e-6 of the float32-policy model;
      (c) float32_fine under a float32 global policy: same graph op types as policy (use
      `TestOffAddsNoOps.graph_ops` pattern); (d) AMP Trainer with gradient_checkpointing full + lean
      sampler + split group conv: 3 finite steps, trace count 1, decoder weights change (copy
      `test_amp_trainer_full_ckpt_lean_split`, parameterised over float32 and float32_fine).
      `TestCheckpointCompatibility`: extend to all four modes: identical variable names, shapes,
      dtypes; checkpoint write/read across every ordered pair of {policy, float32, float32_fine};
      `copy_weights_from` across modes; NEW `C.warm_start_model` round trip: save a run with
      `tests.test_cascade.save_run` (level-4, deform decoder) and warm-start level-4 models built with
      float32 and float32_fine -> `info["mode"] == "full"`, every weight equal.
      `tests/test_cascade.py` `TestFrozenStageFactories`: new test, under mixed_float16: source run
      via `save_run`, `C.load_cascade_source` + `C.check_cascade_config`, trainable level-2 model with
      deform_decoder_dtype float32 / float32_fine, `model.copy_weights_from(FrozenStage.from_checkpoint(...).model)`:
      all weights equal, frozen model compute dtype float32 everywhere (including the decoder), trainable
      d1 float32 and d3 float16 in float32_fine; plus the same for a FrozenChain level-1 copy
      (`FrozenChain.from_models`, TINY, build n 1 or 2) if it fits the 6000 MB cap (else level 2 only,
      noted in findings).
      `tests/test_runtime_infra.py` `test_nonfinite_flags`: float32_fine parses with --amp and deform,
      errors without deform / without --amp. `tests/test_sweep_tools.py` / `tests/test_slurm_scripts.py`:
      DEFORM_DECODER_DTYPE=float32_fine emits `--deform-decoder-dtype float32_fine`; linear-only r_level4
      drops it with the NOTE; invalid values still exit 2 (update expected error texts).
      Files: tests/test_fp16_mitigations.py, tests/test_cascade.py, tests/test_runtime_infra.py,
      tests/test_sweep_tools.py, tests/test_slurm_scripts.py
      Verify: `fp32w_tests.sh tests.test_fp16_mitigations tests.test_cascade tests.test_runtime_infra tests.test_group_net tests.test_trainer`
      all OK and `[golden] max |diff| 0`, `[golden deform] max |diff| 0` in the test_trainer output;
      `fp32w_py38.sh` OK.

- [ ] 4. NIT 1 + NIT 2 (TASK 4, D5).
      `runtime/run.py` `fit`: `lso_flag = amp and save_streak[task] >= LSO_SKIP_STREAK - 1`; update the
      comment above `LSO_SKIP_STREAK`, the module docstring "LSO skip" bullet and `_apply_gate`
      docstring: the flag follows the hard-failure streak, so gradient-only overflows never arm it and
      the first forward NaN of a streak always reaches the LSO. `.agents/tasks/nonfinite-fixes/report.md`
      LSO bullet: one sentence noting the change. `runtime/args.py` help of `--nonfinite-patience-task`
      + docstring bullet; `runtime/run.py` docstring "ckpt/last protection": with 0 (guard off) a task
      in a permanent hard-failure streak defers ckpt/last (incl. the final save) for the rest of the run;
      `ckpt_last_deferred` events show it; use ckpt/best* or keep the guard on.
      New test in `tests/test_trainer.py` `TestNonfiniteGuard`: AMP, LSO initial scale 2**40, ratio
      rigid+deform, `_KernelSwitchSource` with the NaN kernel on the 2nd deform step only; the 1st
      deform step is a gradient-only overflow (finite loss, not finite); assert the 2nd deform step has
      `lso_skipped` False and the loss scale halved on both steps.
      Files: runtime/run.py, runtime/args.py, tests/test_trainer.py, .agents/tasks/nonfinite-fixes/report.md
      Verify: `fp32w_tests.sh tests.test_trainer tests.test_main` OK, goldens `max |diff| 0`,
      `test_lso_not_halved_by_streaking_task` and `test_gradient_overflow_and_isolated_nan_unchanged` unchanged and passing.

- [ ] 5. `--cascade-trainable-init copy|fresh` (D3) + width documentation (TASK 2).
      `runtime/checkpoint.py`: `CASCADE_FRESH_FIELDS = ("filters", "decoder_widths")`;
      `check_cascade_config(model_config, src_cfg, allow=())` ignores fields in `allow` (message
      unchanged otherwise); `cascade_init_record(model, info, fresh=False, config_diff=None)`: fresh ->
      mode "fresh", restored_parts {}, fresh_parts = part counts, n_restored_vars 0, n_fresh_vars
      len(model.weights), config_diff {field: [source, this run]}.
      `runtime/args.py`: `--cascade-trainable-init` choices copy|fresh default copy (Cascade group);
      fresh needs --cascade-from and --resume false (parser errors); docstring bullet.
      `main.py`: `_load_cascade_source(..., allow=CASCADE_FRESH_FIELDS if fresh else ())`; in both
      cascade branches (level 2 FrozenStage, level 1 FrozenChain) skip `model.copy_weights_from` when
      fresh and use the fresh init record; cascade record adds `trainable_init: "fresh"` and
      `trainable_config_diff` only when fresh (copy records byte-identical); print a NOTE line
      "cascade: trainable stage randomly initialised (--cascade-trainable-init fresh; memory/time
      measurement only)".
      Width docs (no new flags): `runtime/args.py` module docstring + `--filters` / `--decoder-widths`
      help: per-block usage (stem = filters[0], d1 = decoder_widths[4], svf_norm_act follows D1),
      the explicit-default-list caveat, presets in `models/width_presets.py`; `models/width_presets.py`
      docstring: a short "grid widths" paragraph listing d12h / d12hs / fr50 as flag lines (no new
      PRESETS entries, so the memory-model grid tests stay unchanged).
      Tests: `tests/test_cascade.py`: `check_cascade_config(..., allow=CASCADE_FRESH_FIELDS)` accepts
      filters/decoder_widths differences and still rejects dropout; fresh init record fields.
      `tests/test_main.py` `TestCascadeMain`: new `test_cascade_trainable_init_fresh` (overfit-synthetic,
      rigid:1, `--deform-decoder true`, TINY_ARGS source at level 4; level-2 run with
      `--decoder-widths 8,8,8,4,4 --filters 4,16,16,16,16,16` (both differ from TINY):
      without the flag SystemExit naming both fields; with `--cascade-trainable-init fresh --resume false`
      rc in (0, 1), config.json init_weights mode "fresh", cascade.trainable_init "fresh",
      trainable_config_diff keys == {filters, decoder_widths}, ckpt/last has `frozen_stage/` keys;
      parse errors for fresh without --cascade-from and with --resume true).
      Files: runtime/checkpoint.py, runtime/args.py, main.py, models/width_presets.py, tests/test_cascade.py, tests/test_main.py
      Verify: `fp32w_tests.sh tests.test_cascade tests.test_main tests.test_runtime_infra tests.test_width_presets_tf`
      OK; `fp32w_py38.sh` OK (main.py --help without TF).

- [ ] 6. Collector `scripts/collect_memory_runs.py` (stdlib, py3.8, no TF) + tests.
      CLI: `--status <status.tsv> --runs <dir> --out <json> [--skip-steps 2]`. status.tsv (written by
      the job, tab-separated, header line): `name level dtype widths n max_pairs rc elapsed_s log`.
      Per config: name, level, dtype, widths (label) + filters / decoder_widths from
      `<runs>/<name>/config.json` args (null = default), n, pairs (max of train_deform/n_pairs, null if
      never logged), status (ok: rc 0 and >= 1 logged step; timeout: rc 124 or 137; oom: log contains
      `ResourceExhaustedError`, `OOM when allocating` or `out of memory`; skipped_time: rc -1 written by
      the job's time guard; abort otherwise), rc, elapsed_s, steps_logged, gpu_peak_mb (max over
      records) and gpu_peak_gib, mean_deform_step_s / mean_linear_step_s ((data_ms + compute_ms) / 1000
      over records with step > skip-steps; a record is linear iff it has a `train_rigid/` or
      `train_affine/` key, else deform; null when none), nonfinite_total (last record), param_count
      (config.json param_counts.trainable) and deform_decoder params, error (last line matching
      `Error` in the log, <= 300 chars). Writes `{"job":..., "records":[...]}` and prints an aligned
      table (name, level, dtype, widths, N, pairs, status, peak GiB, deform s, params). Exit 0 unless
      the status file is missing (2).
      Tests `tests/test_collect_memory_runs.py` (no TF, temp dirs with fake metrics.jsonl/config.json/
      logs): ok record values, oom / timeout / abort / skipped_time classification, step skipping,
      missing run dir. Add the module to the TF-free list in `tests/run_tests_wsl.sh` and the script to
      PYFILES.
      Files: scripts/collect_memory_runs.py, tests/test_collect_memory_runs.py, tests/run_tests_wsl.sh
      Verify: `fp32w_py38.sh` (runs tests.test_collect_memory_runs without TF) OK.

- [ ] 7. Job script `slurm/m11_fp32_width_memory.sh` (TASK 3) + tests.
      Header as m10_batch_memory_capped.sh: `--job-name=bgr-m11-fp32w`, `-A qtim`, `-p rtx8000`,
      `--gpus=1`, `-N 1`, `--ntasks-per-node=1`, `--cpus-per-task=3`, `--mem=64G`, `--time=04:00:00`,
      `--output/--error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.{out,err}`;
      comments: purpose, grid, recipe, time estimate, split proposal, "DO NOT submit without Jay's
      approval", the sbatch line `sbatch slurm/m11_fp32_width_memory.sh`, mkdir logs note. Env (all
      overridable): PROJ, REPO, DATA ($PROJ/data/npy_v2), SRC_RESULTS ($PROJ/results), SIF,
      SCRATCH_ROOT (/scratch), JOB_ID, L1_SOURCE ($SRC_RESULTS/m8_level2_gradnorm/ckpt/best; its stored
      frozen_stage = the m7_three_task_lncc level-4 stage), L2_SOURCE ($SRC_RESULTS/m7_three_task_lncc/ckpt/best),
      STEPS 8, CONFIG_TIMEOUT 2400, TIME_BUDGET_S 13800, ONLY (comma list of config names, empty =
      all), BGR_DRY_RUN. OUT_DIR=$PROJ/scratch/m11/$JOB_ID, RUNS=$OUT_DIR/runs, STATUS=$OUT_DIR/status.tsv;
      TMPDIR=$SCRATCH_ROOT/$JOB_ID and the SINGULARITYENV_* exports / binds of m9_level1_cascade.sh.
      Validation (exit 2): integers, ONLY names known, both sources exist (dir or <prefix>.index; also in
      dry run, read-only). Common args (D4) + per level:
      L1: `--level 1 --cascade-from $L1_SOURCE --group-size 4 --deform-group-size 4 --min-pairs n
      --pair-recompute true --lean-losses true --gpu-memory-limit-mb 40000 --val-image-chunk 1 --val-pair-chunk 1`;
      L2: `--level 2 --cascade-from $L2_SOURCE --group-size 16 --deform-group-size 16 --min-pairs n
      --max-pairs 64 --pair-recompute false --val-pair-chunk 4` (memory growth);
      L4: `--level 4 --group-size 64 --deform-group-size 64 --min-pairs n --max-pairs 64
      --pair-recompute false --val-pair-chunk 16` (from scratch, memory growth);
      all: `--amp true --gradient-checkpointing full --lean-sampler true --split-group-conv true
      --deform-decoder-dtype <dtype>`; non-default widths at L1/L2 add `--cascade-trainable-init fresh`.
      Config order (20): l1_{f32fine,f32}_{default,d12h,d12hs,fr50} interleaved per width
      (l1_f32fine_default, l1_f32_default, l1_f32fine_d12h, l1_f32_d12h, l1_f32fine_d12hs, l1_f32_d12hs,
      l1_f32fine_fr50, l1_f32_fr50), then l1_f16_{default,d12h,d12hs,fr50}, then
      l2_{f32fine,f32}_{default,d12h}, then l4_{f32fine,f32}_{default,d12h}. Per config: time guard
      (elapsed + 300 > TIME_BUDGET_S -> status row rc -1, skipped), `timeout min(CONFIG_TIMEOUT,
      remaining)` singularity exec main.py ... > $RUNS/<name>.log 2>&1 </dev/null, rc captured (no
      set -e), status row appended, tail -n 5 of the log echoed on rc != 0, `rm -rf $RUNS/<name>/ckpt`.
      End: collector in the container (no --nv) -> $OUT_DIR/fp32_width_memory.json + table; exit 0
      (OOMs are results) unless the collector fails. BGR_DRY_RUN=1: print one
      `CONFIG <name>: python main.py ...` line per selected config and the `COLLECT:` line, exit 0 before
      any mkdir/singularity. Time estimate (in the header): L1 ~9-10 min per running config (startup:
      DALI + level-1 validation sets + model builds + frozen chain ~4-5 min, first deform step trace
      ~2-3 min, 7 steps x 25-45 s), ~4 min per OOM -> 12 L1 configs ~1.8-2.0 h; L2 4 x ~5 min; L4
      4 x ~4 min -> ~2.6 h total, fits 4 h; the time guard skips the tail and records it. Split if
      needed: rerun `ONLY=<the skipped names> sbatch --time=02:00:00 slurm/m11_fp32_width_memory.sh`.
      Tests in `tests/test_slurm_scripts.py`: `M11` constant, added to the static header/forbidden-
      account/ascii/no-CR checks and `test_bash_syntax`; new env names added to `RECIPE_ENV`;
      `test_m11_header` (exact SBATCH lines incl. qtim, rtx8000, --gpus=1, 3 CPUs, 64G, 04:00:00, log
      paths), `test_m11_dry_run` (20 CONFIG lines in the stated order; every line has --task-ratio
      deform:1 --val-every 0 --ckpt-every 0 --bn-recal-batches 0 --resume false --amp true
      --gradient-checkpointing full --lean-sampler true --split-group-conv true --log-every 1; L1 lines
      have --cascade-from <L1_SOURCE> --deform-group-size 4 --min-pairs n --pair-recompute true
      --lean-losses true --gpu-memory-limit-mb 40000 --intensity-aug-device tf; L4 lines no
      --cascade-from, --max-pairs 64 --deform-group-size 64 --pair-recompute false; default-width lines
      have no --filters/--decoder-widths/--cascade-trainable-init; d12h/d12hs/fr50 lines have the exact
      lists and (L1/L2) --cascade-trainable-init fresh; dtype flags per name), `test_m11_only_and_guards`
      (ONLY subset; unknown ONLY name / missing source / bad STEPS exit 2), `test_m11_stub_run` (stub
      singularity: 20 main.py calls + 1 collector call, status.tsv has 21 lines, OUT_DIR under
      $PROJ/scratch/m11/manual). Add the script to the `bash -n` list in `tests/run_tests_wsl.sh`.
      Files: slurm/m11_fp32_width_memory.sh, tests/test_slurm_scripts.py, tests/run_tests_wsl.sh
      Verify: `fp32w_py38.sh` (bash -n + tests.test_slurm_scripts) OK.

- [ ] 8. Full integration + report.
      Run the full suite in the background: `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/tests/run_tests_wsl.sh`
      (~3 h; all OK, goldens max |diff| 0). Write `.agents/tasks/fp32-decoder-width/report.md`: files
      changed, flag table (`--deform-decoder-dtype float32_fine`, `--cascade-trainable-init`), exact
      width flag usage, FrozenStage dtype decision, NIT 1-3 resolution, the m11 grid / time estimate /
      split proposal, the cluster order (git pull after Jay commits/pushes; `sbatch slurm/m11_fp32_width_memory.sh`,
      pre-approved by Jay once this workflow lands, at most two active jobs), test counts. Delete the
      `.agents/tmp/fp32w_*` helpers at the end.
      Files: .agents/tasks/fp32-decoder-width/report.md
      Verify: suite summary line shows 0 failing invocations.

## Gaps / assumptions

- No level-2/level-4 deform main.py test with the new dtypes (local 6 GB cap OOMs at level 2 with
  deform, see tests/test_main.py T-E1 note); the cascade/warm-start NIT 3 coverage is at the function
  level (same calls main.py makes) plus the main.py rigid-only fresh-init test.
- Deform-only ratio means no linear DALI pipelines in the m11 runs: TF peaks are representative,
  process DALI memory is lower than in the real three-task run (TF limit 40000 MB keeps L1 honest).
