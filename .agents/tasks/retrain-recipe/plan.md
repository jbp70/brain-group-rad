# Implementation Plan: retrain recipe (d12hs widths, BN trainable at L4/L2, L1 float32_fine)

Scope: script defaults, tests and a report. No model or trainer code changes (the BN code paths
below are correct as they are). Do not commit, push or submit jobs.

## Decisions

- **Level-2 recalibration: KEEP `BN_RECAL_BATCHES=64`, set `BN_RECAL_FREEZE=false`.** The
  level-2 trainable stage starts as a copy of the level-4 weights (`main.py` cascade path,
  `model.copy_weights_from(frozen.model)`), so its BN moving statistics are level-4 statistics on
  different inputs (4 mm vs 2 mm, pre-warped). With Keras momentum 0.99 the moving averages need
  ~460 steps to forget them (0.99^460 ~ 1 %), so the step-0 / early eval-mode validations, the
  first best-checkpoint picks and the smoke's VAL_EVERY 100 validations would run on mismatched
  statistics. The recalibration replaces them with the exact mean over 64 level-2 batches
  (~9 min incl. the pre/post validation passes in m8, job 10505028: 515 s) and logs the
  eval/train gap. Training then updates them normally because nothing freezes them.
- **Width env:** `FILTERS="${FILTERS-16,64,128,192,256,384}"` and
  `DECODER_WIDTHS="${DECODER_WIDTHS-128,96,64,16,8}"` in all three scripts (unset -> d12hs; set
  but empty -> flag not passed = ModelConfig default; same `${VAR-}` convention as
  `VAL_GROUP_SIZE` in r_level4.sh). Format checked in bash (6 resp. 5 positive integers, comma
  separated), exit 2 otherwise. Emitted as `$WIDTH_ARGS` before `$MITIG_ARGS $EXTRA_ARGS`
  (EXTRA_ARGS stays last).
- **Sweep:** `slurm/sweep_level4.py` passes `FILTERS` / `DECODER_WIDTHS` explicitly in every
  config env (appended to `CFG_KEYS`, constants `WIDTH_FILTERS` / `WIDTH_DECODER`) and adds both
  names to `SCRIPT_ENV_KEYS`, so the width is recorded in `r4sw_submissions.jsonl` and a stale
  login-shell value never leaks into a job. No new CLI option (YAGNI).
- **No `d12hs` entry in `models.width_presets.PRESETS`** (would touch preset / memory-model
  tests); only a docstring line naming d12hs as the retrain width.
- r_level*.sh tests live in `tests/test_sweep_tools.py` (TestScriptsStatic / TestScriptsDryRun);
  `tests/test_slurm_scripts.py` covers m8/m9/m10/m11 scripts only. It gets only a hygiene edit
  (FILTERS / DECODER_WIDTHS in `RECIPE_ENV`).

## BN code trace (verify points 1 and 2): no code fix needed

Point 1, `--bn-recal-freeze false` keeps BN in training mode for the whole run:
- `models/layers.py` 471-495 `BatchNormWithSeparateAffine`: `stats_frozen=False` by default;
  `call` passes `training` through unless `stats_frozen` (line ~493
  `bn_training = False if self.stats_frozen else training`). Nothing else sets it except
  `set_bn_stats_frozen` (552-559).
- `runtime/run.py` 1753-1854 `recalibrate_bn`: momentum temporarily `c/(c+1)` and restored in
  `finally` (orig momentum 0.99 back); freezes ONLY in `if freeze:` (1844-1846:
  `set_bn_stats_frozen(True)` + `rebuild_steps()`). With freeze False the event records
  `"freeze": False` and stats_frozen stays False.
- `main.py` 429-433: recalibration only when `cascade_on and not cascade_resume and
  args.bn_recal_batches > 0` (never on resume). `main.py` 405-411 (resume): freeze re-applied
  only if the SAVED record has `freeze is True`; a freeze-false run resumes with stats_frozen
  False, same as the fresh run, and the recalibrated-then-trained statistics come back from
  ckpt/last.
- Train steps: `runtime/run.py` 933-978 (deform) and 1050-1096 (linear) call
  `model(x_in, training=True)`; the BN backup is restored only when the step is non-finite
  (`tf.cond(finite, no-op, restore)`, ~977). Train-mode validation / diagnostics
  (`_val_fn` 1195-1216, `_train_mode_forward` 1219-1240, `_val_deform_fn` 1314-1331) restore all
  `model.non_trainable_variables` after their forward, so validation never moves the statistics.
- Frozen stages: `models/cascade.py` 51 builds them float32 with `gradient_checkpointing='none'`;
  `FrozenStage.linear/deform` (116-126) and `FrozenChain` (233, 248, 274, 289) always call with
  `training=False` plus `stop_gradient`; their variables are not in `model.non_trainable_variables`
  of the trainable model nor in the optimizer.

Point 2, once-per-step update under `--gradient-checkpointing full`:
- `models/group_net.py` 404-418: deform with mode 'full' (built) -> `_call_deform_full`
  (434-456): encoder and decoder each wrapped in `checkpointing.recompute(fn, bn_layers, training)`;
  integrators in plain `tf.recompute_grad` (no BN). Linear tasks are never checkpointed (one
  forward = one update).
- `models/checkpointing.py` 31-37 `_restores_needed`: snapshot/restore built iff `training` is
  truthy and some layer has `stats_frozen False` (read at trace time, i.e. after recal, since
  freeze calls `rebuild_steps()`); 40-64 `recompute`: call 1 (forward) updates, every later call
  (gradient recompute, diagnostic gradient, 2nd tape.gradient) snapshots, runs, restores.
- Existing coverage: `tests/test_checkpointing.py` TestFullBNOnce (b) level 4 non-cascade,
  (b') persistent tape, (c) frozen, TestFullRecal (d) recal under full at level 2. Missing and
  added below: cascade level-2 TRAIN steps after recal with freeze False under 'full' (trainable
  updated once, frozen stage untouched) and the freeze-True counterpart under 'full'; main.py
  resume with freeze False.

Local memory (6000 MB cap): `tests/test_cascade.py` 522-526 records level-2 TINY deform N 2 with
T 3 under 'both' at ~2.9 GB; 'full' keeps fewer activations, and the 'integrator' reference keeps
TINY decoder activations only, so the new 'full' tests fit (run them in their own process).

## Items

- [ ] 1. Width defaults and env overrides in the three retrain scripts.
      In each of `slurm/r_level4.sh`, `slurm/r_level2.sh`, `slurm/r_level1.sh`: add
      `FILTERS="${FILTERS-16,64,128,192,256,384}"` and `DECODER_WIDTHS="${DECODER_WIDTHS-128,96,64,16,8}"`
      to the variable block (with a comment: unset -> d12hs, set-but-empty -> model default);
      after the `need_choice` checks add `WIDTH6_RE='^[1-9][0-9]*(,[1-9][0-9]*){5}$'`,
      `WIDTH5_RE='^[1-9][0-9]*(,[1-9][0-9]*){4}$'` and build `WIDTH_ARGS` (`--filters $FILTERS` iff
      non-empty and matching, `--decoder-widths $DECODER_WIDTHS` likewise; `die` with
      "FILTERS must be 6 comma-separated positive integers (stem,L2..L6) or empty, got '...'" /
      "DECODER_WIDTHS must be 5 ... (D5..D1) or empty"). Insert `$WIDTH_ARGS` in `MAIN=(...)` right
      before `$MITIG_ARGS $EXTRA_ARGS` (r_level4: after `$VAL_ARGS $DEFORM_ARGS $MEM_ARGS`). Add
      `widths: filters ${FILTERS:-default} | decoder ${DECODER_WIDTHS:-default}` to the echo block.
      Header paragraph "Width (Jay, retrain recipe)" in each script: d12hs = stem + d1/d2 halved,
      same explicit lists at every level so config.json ModelConfig is identical (the cascade check
      compares every field but seed; an explicit list is not config-equal to the None default, so
      never mix FILTERS= runs with default runs in one cascade, and resume a run with the widths it
      started with); measured fits: L4 deform N64 fp16 25.6 GiB 4.1 s (job 10528435), L2 deform
      N16 fp16 27.2 GiB 7.6 s (10528435), L1 deform N4 4 pairs float32_fine 36.95 GiB 21.1 s under
      40000 MB (10528396). Keep every string the existing header tests assert.
      Files: slurm/r_level4.sh, slurm/r_level2.sh, slurm/r_level1.sh
      Verify: `bash -n` on the three scripts (command V1 below) exits 0; dry runs (V3) show
      `--filters 16,64,128,192,256,384 --decoder-widths 128,96,64,16,8` in every MAIN line.

- [ ] 2. Level-2 BN trainable and level-1 float32_fine defaults.
      `slurm/r_level2.sh`: `BN_RECAL_FREEZE="${BN_RECAL_FREEZE:-false}"`; rewrite the header
      sentence ("BN statistics ... frozen afterwards with BN_RECAL_FREEZE=true") to: recal at step 0
      of a fresh run (BN_RECAL_BATCHES 64) then BN stays TRAINABLE (BN_RECAL_FREEZE false, Jay
      2026-10 retrain recipe: batch statistics over N 32 / 16 groups are large enough), with the
      recal reason from the Decisions section; BN_RECAL_BATCHES=0 skips it (moving stats then start
      from the level-4 copy); BN_RECAL_FREEZE=true = the old m8 recipe. Update the "Resume plan"
      line ("re-applies the BN freeze" -> "re-applies the BN freeze only if the saved record has
      freeze true; never repeats the recalibration"). Deform decoder dtype stays `policy` (fp16
      under AMP) at L4 and L2; say so in both headers' mitigation paragraph.
      `slurm/r_level1.sh`: `DEFORM_DECODER_DTYPE="${DEFORM_DECODER_DTYPE:-float32_fine}"` (policy /
      float16 / float32 still selectable); replace the paragraph "Level 1: DEFORM_DECODER_DTYPE=
      float32 is NOT expected to fit ..." with: default float32_fine (fp32 d2, d1, their upsamples
      and svf_norm_act; d5..d3 fp16) after the m9 long-run fp16 overflow in the trainable d1 split
      group conv (job 10518952, probe 10524598); m11 job 10528396 at d12hs: float32_fine 36.95 GiB
      21.1 s, float32 35.50 GiB 21.5 s, float16 36.10 GiB 20.6 s; DEFORM_DECODER_DTYPE=policy =
      the old fp16 decoder. In the "No flag is emitted while a value is at its default" sentence of
      r_level1 note that the dtype default is float32_fine (flag emitted). Keep BN_RECAL_BATCHES 64,
      BN_RECAL_FREEZE true, N 8, DEFORM_N 4, TF limit 40000, pair recompute true, lean losses true,
      val chunks 1/1. Add to the N 8 comment: with the d12hs stem (16 ch) the int32 limit moves to
      N ~19; N stays 8 (Jay: group sizes unchanged).
      Files: slurm/r_level2.sh, slurm/r_level1.sh
      Verify: V1; dry run of r_level2 shows `--bn-recal-batches 64 --bn-recal-freeze false`, of
      r_level1 shows `--bn-recal-freeze true` and `--deform-decoder-dtype float32_fine`.

- [ ] 3. Sweep helper uses the final width.
      `slurm/sweep_level4.py`: add `WIDTH_FILTERS = "16,64,128,192,256,384"` and
      `WIDTH_DECODER = "128,96,64,16,8"` (comment: the retrain width d12hs, identical to the
      r_level*.sh defaults); add `"FILTERS", "DECODER_WIDTHS"` to `SCRIPT_ENV_KEYS` and append them
      to the END of `CFG_KEYS`; `_config` sets `env["FILTERS"] = WIDTH_FILTERS` and
      `env["DECODER_WIDTHS"] = WIDTH_DECODER`. Mention in the module docstring (Configurations
      paragraph) that every config passes the d12hs widths. Python 3.6 syntax only.
      Files: slurm/sweep_level4.py
      Verify: `python3 slurm/sweep_level4.py list` (bgr_preproc38) prints lines ending with
      `... VAL_GROUP_SIZE=16 FILTERS=16,64,128,192,256,384 DECODER_WIDTHS=128,96,64,16,8 sbatch -A qtim -p rtx8000 ...`.

- [ ] 4. TF-free tests for the new defaults (depends on 1-3).
      `tests/test_sweep_tools.py`:
      - `test_phase_a` / `test_phase_b_and_c`: assert `env["FILTERS"] == "16,64,128,192,256,384"`
        and `env["DECODER_WIDTHS"] == "128,96,64,16,8"` for every config.
      - `test_script_env_keys_cover_r_level4`: regex `r"\$\{([A-Z_][A-Z0-9_]*):?-"` (also catches
        `${FILTERS-`, `${VAL_GROUP_SIZE-`), assert FILTERS / DECODER_WIDTHS in the set.
      - `TestScriptsStatic.test_headers`: every script header contains "d12hs" and
        "--filters 16,64,128,192,256,384 --decoder-widths 128,96,64,16,8"; r_level2 header contains
        "BN_RECAL_FREEZE false"; r_level1 header contains "float32_fine" and "36.95 GiB".
      - `test_r4_default`, `test_r2_default`, `test_r1_default`: add the seq
        `["--filters", "16,64,128,192,256,384", "--decoder-widths", "128,96,64,16,8"]`;
        r2: `["--bn-recal-freeze", "true"]` -> `"false"`, and `--deform-decoder-dtype` absent;
        r1: `["--deform-decoder-dtype", "float32_fine"]`, `["--bn-recal-batches", "64"]`,
        `["--bn-recal-freeze", "true"]`; r1 with `TASK_RATIO=rigid:1,affine:1`: NOTE
        "linear-only ratio, DEFORM_DECODER_DTYPE dropped" and no `--deform-decoder-dtype`.
      - `test_mitigations_default_off`: R1 now expects `["--deform-decoder-dtype", "float32_fine"]`;
        the other MITIG_FLAGS stay absent for R1; R4 / R2 unchanged (no dtype flag).
      - New `test_widths_env`: FILTERS="" DECODER_WIDTHS="" -> neither flag in MAIN for R4, R2, R1;
        FILTERS=32,64,128,192,256,384 -> passed through; invalid ("16,64", "a,b,c,d,e,f",
        "0,64,128,192,256,384", DECODER_WIDTHS "1,2,3") -> exit 2 with "ERROR" and no "MAIN: ".
      - New `test_cascade_model_configs_identical` (the AC "r_level2/r_level1 cascading from
        r_level4/r_level2 passes the check"): for R4, R2 (CASCADE_FROM=self.ckpt), R1 dry-run MAIN
        args -> `runtime.args.parse_args` -> `runtime.args.model_config_from_args` ->
        `json.loads(json.dumps(cfg.to_dict()))` -> `models.config.ModelConfig.from_dict` (the
        config.json round trip), then the `runtime.checkpoint.check_cascade_config` rule
        replicated TF-free (`{k: v for k, v in d.items() if k != "seed"}` equal for R4 vs R2 and
        R2 vs R1; comment why: runtime.checkpoint imports TF). Also assert filters ==
        (16,64,128,192,256,384), decoder_widths == (128,96,64,16,8), deform_decoder True,
        integration_steps 7 at all three levels.
      `tests/test_slurm_scripts.py`: add "FILTERS", "DECODER_WIDTHS" to `RECIPE_ENV` (hygiene only).
      Files: tests/test_sweep_tools.py, tests/test_slurm_scripts.py
      Verify: V2 — `tests.test_sweep_tools` and `tests.test_slurm_scripts` pass in bgr_preproc38
      without importing TF.

- [ ] 5. TF tests for trainable BN under full checkpointing in the level-2 cascade (verify point 3).
      `tests/test_checkpointing.py`: new class `TestFullCascadeTrainableBN(CascadeTrainerCase)`
      (imports already there: `CascadeTrainerCase`, `bn_stats`, `cascade_config`, `rel`,
      `stats_of`, `Trainer`, `L`; add `from tests.test_cascade import _vals, _all_equal` or use
      `np.array_equal` loops). Config `cascade_config(flip_prob=0.0)` (RATIO3, TINY, T 3, N 4 /
      deform N 2, fp32).
      - `test_freeze_false_updates_once`: `tr = self.make(cfg, gradient_checkpointing="full")`,
        `ref = self.make(cfg, gradient_checkpointing="integrator")` (same seed -> identical weights,
        same postprocess RNG; 'integrator' never recomputes encoder/decoder = one update by
        construction). On both: `recalibrate_bn(4, freeze=False, val=False)`; assert
        `not any(m.stats_frozen for m in Trainer._bn_layers([tr.model]))`. Snapshot
        `s0 = bn_stats(tr.model, tr._bn_parts("deform"))`, `f0 = _vals(tr.cascade.model.weights)`.
        One `train_step("deform", self.batches["deform"])` on both (finite). Assert
        `rel(full, ref) <= 1e-6`, every array of `bn_stats(tr.model, parts)` differs from s0, and
        `rel(full, s0) > 1e-4` (so a double update would exceed the tolerance; print both numbers
        to stderr like the other tests). Then `train_step("rigid", ...)` and a 2nd deform step on
        `tr`: encoder + linear_decoder BN stats change again (not array_equal to before), frozen
        stage weights `_all_equal(f0, tr.cascade.model.weights)`.
      - `test_freeze_true_full`: `tr = self.make(cfg, gradient_checkpointing="full")`,
        `recalibrate_bn(4, freeze=True, val=False)`; all layers `stats_frozen`; snapshot whole-model
        `bn_stats(tr.model)` and frozen weights; one rigid, affine, deform step (finite); stats and
        frozen weights bitwise unchanged; `tr.trace_counts == {"rigid": 1, "affine": 1, "deform": 1}`.
      Call `free_memory()` in tearDown (CascadeCase already does if it has tearDown; otherwise add).
      If the two-trainer test OOMs under the 6000 MB cap, fall back to `ref` mode "both" and compare
      the ENCODER statistics only (both = encoder not recomputed; decoder double update documented),
      plus the decoder check `rel(dec_full, s0_dec) > 0` and record the fallback in the report.
      `tests/test_cascade.py` TestCascadeResume: new `test_main_freeze_false_resume` copying the
      setup of `test_main_fresh_and_resume` (rigid-only, TINY_ARGS, `--deform-decoder true`): source
      level 4 with `--overfit-max-steps 3`; level 2 with `--bn-recal-batches 8 --bn-recal-freeze
      false --overfit-max-steps 2`: config.json `bn_recalibration.freeze is False`, one
      `bn_recalibration` event; `prefix2, bn2 = ckpt_bn_stats(d)`, `_, fz2 = ckpt_bn_stats(d,
      scope="frozen_stage/")`. Resume with the same flags and `--overfit-max-steps 3`: still one
      recalibration event, `bn_recalibration` block unchanged, ckpt prefix ends with `last-3`; the
      trainable encoder BN entries (keys of bn3 containing "encoder"; inspect the key names once)
      ALL differ from bn2 (BN keeps training after resume = fresh-run state), frozen_stage BN
      entries bitwise equal to fz2. `free_memory()` after every `MAIN.main`.
      Files: tests/test_checkpointing.py, tests/test_cascade.py
      Verify: V4 — `tests.test_checkpointing` (whole module) and
      `tests.test_cascade.TestBNRecal tests.test_cascade.TestCascadeResume` pass in tf_gpu.

- [ ] 6. Docs and goldens (verify point 5).
      `models/width_presets.py` docstring: after the d12hs bullet add "(the from-scratch retrain
      width: default of slurm/r_level{4,2,1}.sh and slurm/sweep_level4.py)". `runtime/args.py`:
      append the same note to the module-docstring per-block widths paragraph (lines 46-53); no
      code change. Run the goldens and py_compile.
      Files: models/width_presets.py, runtime/args.py
      Verify: V5 goldens pass unchanged (no fixture rewrite; never set BGR_WRITE_GOLDEN); V1
      py_compile OK; `tests.test_width_presets` passes (in V2 list).

- [ ] 7. Report.
      Write `.agents/tasks/retrain-recipe/report.md`: what changed (per file), the BN trace above
      (points 1 and 2, with the file/line references) and the test results of items 4-6 (counts,
      the printed rel numbers, whether 'full' fit the 6000 MB cap or the fallback was used), the
      level-2 recal decision, the three dry-run MAIN lines (V3 output), and the cluster commands
      (rtx8000, qtim, 1 GPU, 3 CPUs, 64G, all already in the SBATCH headers; Jay approves before
      submission; `git pull` in the cluster clone first; `mkdir -p .../GroupwiseReg/logs`):
      - L4 smoke: `RUN_NAME=r4_smoke STEPS=300 VAL_EVERY=100 WARMUP=50 DEFORM_START=100 DEFORM_RAMP=100 CURRICULUM_RAMP=100 G_START=100 G_RAMP=100 PROJECT_STEPS=16000 PROJECT_VAL_EVERY=500 sbatch --job-name=r4smoke --time=02:00:00 --mem=64G slurm/r_level4.sh`
        (defaults: N 128 / deform 64, d12hs, fp16 decoder, BN trainable; check gpu_peak <= 42000 MiB
        budget, m11 measured 25.6 GiB at deform N64).
      - L2 smoke after it (pipeline check only, source barely trained):
        `RUN_NAME=r2_smoke CASCADE_FROM=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results/r4_smoke/ckpt/best STEPS=200 VAL_EVERY=100 WARMUP=50 PROJECT_STEPS=8000 PROJECT_VAL_EVERY=500 sbatch --job-name=r2smoke --time=03:00:00 --mem=64G slurm/r_level2.sh`
        (checks: bn_recalibration event freeze false, cascade check passes = identical
        ModelConfig, gpu_peak, BN gap shrink).
      - Note the L1 smoke line from the r_level1 header for later (cascading from the r2 run).
      Files: .agents/tasks/retrain-recipe/report.md
      Verify: the report lists results for points 1-5 and the commands.

## Verification commands

All from PowerShell; repo in WSL at `/mnt/d/Research/Brain_Registration_Projects/brain-group-rad`.

- V1 (syntax, bgr_preproc38):
  `wsl -d Ubuntu bash -c "set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate bgr_preproc38; cd /mnt/d/Research/Brain_Registration_Projects/brain-group-rad && python -m py_compile slurm/sweep_level4.py models/width_presets.py runtime/args.py tests/test_sweep_tools.py tests/test_slurm_scripts.py tests/test_checkpointing.py tests/test_cascade.py && bash -n slurm/r_level4.sh && bash -n slurm/r_level2.sh && bash -n slurm/r_level1.sh && echo SYNTAX_OK"`
- V2 (TF-free tests, bgr_preproc38):
  `wsl -d Ubuntu bash -c "set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate bgr_preproc38; cd /mnt/d/Research/Brain_Registration_Projects/brain-group-rad && PYTHONDONTWRITEBYTECODE=1 python -c \"import sys, unittest; r = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(['tests.test_sweep_tools', 'tests.test_slurm_scripts', 'tests.test_width_presets'])); assert 'tensorflow' not in sys.modules; sys.exit(0 if r.wasSuccessful() else 1)\""`
  (quoting is fragile from PowerShell: if it breaks, write the snippet to a temp .py under the repo's `.agents/tasks/retrain-recipe/` and run it, or use the shell-and-remote-jobs skill.)
- V3 (dry runs, bgr_preproc38 or plain WSL bash): with `mkdir -p /tmp/rr_ckpt`:
  `BGR_DRY_RUN=1 RESULTS=/tmp/rr_results bash slurm/r_level4.sh`,
  `BGR_DRY_RUN=1 RESULTS=/tmp/rr_results CASCADE_FROM=/tmp/rr_ckpt bash slurm/r_level2.sh`,
  `BGR_DRY_RUN=1 RESULTS=/tmp/rr_results CASCADE_FROM=/tmp/rr_ckpt bash slurm/r_level1.sh`
  -> each prints one `MAIN: python main.py ...` and one `CHECK: ...` line, exit 0.
- V4 (TF tests, tf_gpu, each module in its own process, background, ~10-30 min):
  `wsl -d Ubuntu bash -c "set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate tf_gpu; cd /mnt/d/Research/Brain_Registration_Projects/brain-group-rad && export PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 BGR_TF_GPU_LIMIT_MB=6000 TF_CUDNN_USE_AUTOTUNE=0 && python -m unittest tests.test_checkpointing -v && python -m unittest tests.test_cascade.TestBNRecal tests.test_cascade.TestCascadeResume tests.test_cascade.TestFrozenStageTraining -v"`
- V5 (goldens, tf_gpu, same env exports):
  `python -m unittest tests.test_trainer.TestLinearOnlyUnchanged tests.test_trainer.TestDeformUnchanged -v` -> both pass with the existing fixtures.
- Do not run the full suite (~3 h).

## Digest

- `slurm/r_level4.sh` — variable block lines ~111-150 (`N="${N:-128}"` ... `BGR_DRY_RUN`), checks
  after `need_choice DEFORM_DECODER` (~170), `MAIN=(... --deform-decoder "$DEFORM_DECODER" $START_ARGS
  $LIN_ARGS $INIT_ARGS ... $VAL_ARGS $DEFORM_ARGS $MEM_ARGS $MITIG_ARGS $EXTRA_ARGS)`, echo block
  before `rc=0`. No width flags today. Tests: tests/test_sweep_tools.py `test_r4_default`,
  `test_r4_overrides` (asserts `main[-2:] == ["--seed","3"]` -> keep EXTRA_ARGS last),
  `test_r4_stub_run`, `test_headers` (keeps strings "NOT a collapse signal", r4 smoke line, etc.).
- `slurm/r_level2.sh` — `BN_RECAL_BATCHES="${BN_RECAL_BATCHES:-64}"`,
  `BN_RECAL_FREEZE="${BN_RECAL_FREEZE:-true}"`; MAIN has `--bn-recal-batches "$BN_RECAL_BATCHES"
  --bn-recal-freeze "$BN_RECAL_FREEZE"` and ends `$VAL_ARGS $DEFORM_ARGS $MEM_ARGS $MITIG_ARGS
  $EXTRA_ARGS`; no `--bn-recal-validate` (main.py default true). Header sentence at lines 18-20
  mentions "frozen afterwards with BN_RECAL_FREEZE=true"; Resume plan line "re-applies the BN
  freeze". Tests: `test_r2_default` (currently asserts `["--bn-recal-freeze","true"]`),
  `test_cascade_guards` (BN_RECAL_FREEZE=yes -> exit 2), `test_mitigations_*`.
- `slurm/r_level1.sh` — `DEFORM_DECODER_DTYPE="${DEFORM_DECODER_DTYPE:-policy}"` (in the
  non-finite block after HAS_LINEAR), MITIG logic emits `--deform-decoder-dtype` iff != policy and
  drops it with a NOTE for linear-only ratios; header paragraph "Level 1: DEFORM_DECODER_DTYPE=float32
  is NOT expected to fit ...". Tests: `test_r1_default`, `test_mitigations_default_off` (currently
  asserts no `--deform-decoder-dtype` for R1 -> must change), `test_headers` (strings "36.7 GB",
  "MIN_PAIRS=1", "39000 MB", "TF_MEM_LIMIT_MB=34000" must stay).
- `slurm/sweep_level4.py` — `SCRIPT_ENV_KEYS` tuple, `CFG_KEYS` tuple, `_config(k, phase, run, lr,
  wd, n, dn, steps, sched, time_limit, val_group_size)` builds env; `env_prefix` joins CFG_KEYS in
  order. Tests: `TestSweepConfigs.test_phase_a` (`set(env) == set(CFG_KEYS)`), `test_list_output`
  (substring "RUN_NAME=... LR=3e-4 WD=0 N=128 DEFORM_N=64 STEPS=8000" -> append new keys at the
  end), `TestSweepCluster.test_one_active_submits_one` (all cfg env keys reach sbatch env),
  `test_script_env_keys_cover_r_level4`, `TestPython36` (3.6 syntax).
- `tests/test_sweep_tools.py` — helpers `TestScriptsDryRun.dry(path, **env)` returns (main args
  after "python main.py", check args, stdout), `assert_seqs`, `l2(**kw)` (CASCADE_FROM=self.ckpt),
  `check_parses(main)` (runtime.args.parse_args TF-free). `STRIP_ENV = sweep.SCRIPT_ENV_KEYS +
  CASCADE_ENV` (new names stripped automatically once in SCRIPT_ENV_KEYS).
- `tests/test_slurm_scripts.py` — `RECIPE_ENV` tuple (top of file); no r_level tests here.
- `tests/test_checkpointing.py` — imports `CascadeTrainerCase, CountingSource, _GrapplerOff,
  bn_stats, cascade_config, randomize` from tests.test_cascade; `rel`, `stats_of` local;
  TestFullRecal pattern (`self.make(cfg, sources=..., gradient_checkpointing="full")`).
- `tests/test_cascade.py` — `CascadeTrainerCase.make(cfg, sources, val_sets, logger, amp,
  cascade_on, gradient_checkpointing=STAGE_CKPT)` (STAGE_CKPT "both", STAGE_KW T 3, N2 4 / ND2 2,
  `self.batches` per task); `bn_stats(model, parts)`, `_vals`, `_all_equal`; TestBNRecal
  `test_freeze_retraces` (freeze True under 'both'); TestCascadeResume `test_main_fresh_and_resume`
  (MAIN_COMMON, TINY_ARGS, `ckpt_bn_stats(run_dir, scope="model/")`, `self.events`,
  `self.read_config`, PROVENANCE).
- Code read, unchanged: `models/layers.py` 471-559, `models/checkpointing.py` (whole),
  `models/group_net.py` 404-485, `runtime/run.py` 933-990, 1180-1340, 1700-1860, `main.py`
  300-460, `models/cascade.py` 18-126, 230-292, `runtime/checkpoint.py` 150-250,
  `models/config.py` 20-125 (tuples normalised in `__post_init__`, `to_dict` = dataclasses.asdict),
  `runtime/args.py` (TF-free; `model_config_from_args` ~824; `--filters` / `--decoder-widths`
  int_list; `--bn-recal-freeze` default true).
- Goldens: `tests/test_trainer.py` `TestLinearOnlyUnchanged.test_matches_golden`
  (fixtures/m7_linear_golden.json) and `TestDeformUnchanged.test_matches_golden`
  (fixtures/lncc_deform_golden.json). No model / trainer code changes, so they must pass as is.
- Not run during planning: no tests were executed; the suite last passed at 709 tests
  (fp32-decoder-width commit 1d5df0f).
