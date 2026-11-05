# Implementation plan: synthetic-contrast training (synth-contrast)

Source of truth: `.agents/tasks/synth-contrast/design.md` v4 (APPROVED) incl. 13.1 / 13.1b, and
`review.md` (2 NITs). This plan sequences that design; it does not re-decide it. The orchestrator's
binding decisions (OD1-OD6 below) are added on top.

Work directly on branch `main` in `d:\Research\Brain_Registration_Projects\brain-group-rad`.
NO commits, NO pushes, NO cluster jobs. Every coder appends what it ran and the results to
`.agents/tasks/synth-contrast/impl-log.md`.

## Orchestrator decisions (binding, on top of the design)

- OD1 Cortical parcel deviation: `SynthConfig.hier_dev` 0.1 for non-cortical fine ids and NEW
  `SynthConfig.hier_dev_cortex` 0.25 for every fine id whose train18 class is 2 (cerebral cortex).
  Decided from `luts.TRAIN18_CLASSES`: raw ids 3, 42, 1001-1035, 2001-2035 -> contiguous synthfine ids
  {2, 20} U {33, ..., 102} (72 ids; 1004/2004 corpus callosum are in the LUT although SynthSeg never
  outputs them). Computed in code (`luts.synthfine_cortex_ids()`), documented in the luts docstring.
  CLI `--synth-hier-dev-cortex` (float, 0 <= x <= 0.5, default 0.25), recorded in config.json only
  with synthesis on (inside the `synth` record, like every other SynthConfig field). Eval presets
  `random_*` use the same value; PET-like presets keep their explicit parcel overrides.
- OD2 Review NIT 1: `n_task_steps(t)` = sum over metrics.jsonl records of `train_<t>/window_steps`;
  `--expect-synth-prob` weights the window means of `train_<t>/synth_frac` by `window_steps`;
  `--min-task-steps` compares n_task_steps with the threshold. Test: 4 records x 25 steps passes
  `deform:100`, fails `deform:101`. Review NIT 2: `export-add-labels --start/--end` = half-open
  [start, end) over the `export_manifest.csv` data rows, end clipped to the row count (last chunk
  shorter), start >= rows -> empty selection = status file with 0 cases, exit 0; negative values or
  start > end -> exit 2; `--start/--end` with `--phase manifest` -> exit 2. Status file named after the
  requested range `rows-<start>-<end>.json` (`end` when --end omitted). Tested.
- OD3 `slurm/r_level1.sh`: `#SBATCH --mem=128G`, `TF_MEM_LIMIT_MB` default 42500 (header text
  updated with the measured reasons: L1 MaxRSS ~94 GB, 64G OOM-killed in validation, 40000 MB OOM in
  d1 backward). The r_level script tests live in `tests/test_sweep_tools.py` (TestScriptsStatic,
  TestScriptsDryRun), NOT in tests/test_slurm_scripts.py; update them there.
- OD4 Config B via `EXTRA_ARGS="--group-config B ..."` in all synthesis smokes and mixed runs; no
  GROUP_CONFIG variable. Smokes keep the script default smoothness 0.5.
- OD5 Default off bit-identical: goldens `tests.test_trainer.TestLinearOnlyUnchanged` and
  `tests.test_trainer.TestDeformUnchanged` max |diff| 0; default config.json differs only by the five
  new `args.synth_*` defaults (no top-level `synth` record).
- OD6 Hard requirements kept: group sizes unchanged, no new in-training validation, memory-neutral
  synthesis (in-place image replacement, uint8 fine labels, no one-hot of fine labels, graph-def
  OneHot-depth-multiset test with negative control), one traced tf.function per task, labels
  keyword-only (cascade-skip guard tests unchanged + synth-on twin).

## Deviations decided while planning (record them in impl-report.md)

- D-a `preprocessing/slurm/add_labels_array.sh` runs through `preprocessing/run_in_container.sh`
  (`RIC=(bash "$REPO_DIR/preprocessing/run_in_container.sh")`, `export USE_GPU=${USE_GPU:-0}`) instead
  of a hand-written `singularity exec`: `preprocessing/tests/test_slurm_scripts.py` requires that
  pattern for every preprocessing SLURM script (no `python -m preprocessing` after `set -euo pipefail`).
  Effect identical (container, no `--nv`, binds incl. /autofs /cluster /space).
- D-b The synthesis-on forced-skip twin and the recal-wiring test go to `tests/test_cascade.py`
  (they need its `CascadeTrainerCase` / `ChainCase` fixtures), not tests/test_synth.py.
- D-c GPU checks s1-s3 get their own job script `slurm/verify_synth.sh` (like slurm/verify_m.sh) so the
  r_level*.sh default command lines stay byte-identical; the orchestrator submits it next to the L4
  smoke pair if wanted.
- D-d `scripts/synth_preview.py` gets a CPU-only basic-partition wrapper `slurm/synth_preview.sh`
  (compute must not run on mlsc-login).
- D-e `eval_synth.py` gets `--data` (default: the run's saved `args.data`) and `--sanity-gate on|off`
  (default on; off only for local tests whose fake run has no comparable logged validation).

## Ordered work (FEAT files hold the detailed steps)

- [ ] 1. FEAT-001 Exporter: `synthfine` label set, nested majority downsampling, `export-check`
      consistency rule, add-only `export-add-labels` (files + manifest phases), SLURM array wrapper.
      Files: preprocessing/luts.py, preprocessing/export.py, preprocessing/add_labels.py (new),
      preprocessing/cli.py, preprocessing/slurm/add_labels_array.sh (new),
      preprocessing/tests/test_add_labels.py (new), preprocessing/tests/test_export.py,
      preprocessing/tests/test_slurm_scripts.py, preprocessing/tests/run_tests_wsl.sh.
      Verify: `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/preprocessing/tests/run_tests_wsl.sh bgr_preproc38`
      -> "ALL CHECKS PASSED (bgr_preproc38)"; manual CLI run on a temp tree logged.
- [ ] 2. FEAT-002 Synthesis core: `SynthConfig` + constants, `data_loading/synth.py`, post-process
      integration (keyword-only `labels=` / `synth_override=`, `tf.cond`, linear render, deform
      fine->train18 gather), fine-label test helpers, tests/test_synth.py. Depends on 1 (luts table).
      Files: data_loading/params.py, data_loading/synth.py (new), data_loading/postprocess.py,
      tests/runtime_helpers.py, tests/test_synth.py (new), tests/test_postprocess.py.
      Verify: tf_gpu `python -m unittest tests.test_synth tests.test_postprocess tests.test_tf_intensity_aug`
      + goldens; bgr_preproc38 py_compile.
- [ ] 3. FEAT-003 Data + trainer + CLI wiring: DALI label input in LinearPipeline, data module
      `synth_label_set`, `--synth-*` flags, TrainConfig.synth, `Trainer._pp_call`, uint8 signatures and
      the four new lambdas, split_batch, synth logging, checkpoint generator, BN-recal synthetic slots,
      main.py config record + resume guard; e2e tiny trainer runs. Depends on 2.
      Files: data_loading/dali_loader.py, data_loading/data_module.py, runtime/args.py, runtime/run.py,
      main.py, tests/test_data_loading.py, tests/test_runtime_infra.py, tests/test_main.py,
      tests/test_trainer.py, tests/test_cascade.py.
      Verify: tf_gpu unittests listed in FEAT-003 + goldens + cascade-skip guards; bgr_preproc38
      `python main.py --help` (no TF) and `import runtime.args` without TF.
- [ ] 4. FEAT-004 Checker, scripts, tools: check_mixed_run `--expect-synth-prob` / `--min-task-steps`
      (window_steps) / recal synth counts; r_level{4,2,1}.sh env SYNTH_* / GPU_POLL / MIN_TASK_STEPS;
      r_level1 128G / 42500; sweep SCRIPT_ENV_KEYS; GPU checks s1-s3 + slurm/verify_synth.sh;
      scripts/synth_preview.py + slurm/synth_preview.sh; tests/run_tests_wsl.sh lists. Depends on 3.
      Files: tests/gpu/check_mixed_run.py, slurm/r_level4.sh, slurm/r_level2.sh, slurm/r_level1.sh,
      slurm/sweep_level4.py, tests/gpu/verify_dali_pipeline.py, slurm/verify_synth.sh (new),
      scripts/synth_preview.py (new), slurm/synth_preview.sh (new), tests/test_check_mixed_run.py,
      tests/test_sweep_tools.py, tests/test_slurm_scripts.py, tests/test_synth_preview.py (new),
      tests/run_tests_wsl.sh.
      Verify: bgr_preproc38 `python -m unittest tests.test_sweep_tools tests.test_slurm_scripts`;
      tf_gpu `python -m unittest tests.test_check_mixed_run tests.test_synth_preview`; bash -n and
      BGR_DRY_RUN=1 dry runs (default and SYNTH_PROB=0.5) of the three r_level scripts logged.
- [ ] 5. FEAT-005 Post-training synthetic evaluation: `group_images_fn` hooks in the validation-set
      builders, `EVAL_PRESETS` (11) + `preset_params`, runtime/eval_run.py, eval_synth.py,
      scripts/compare_synth_eval.py, slurm/eval_synth.sh, tests. Depends on 3 (and 2).
      Files: runtime/fixed_data.py, data_loading/synth.py, runtime/eval_run.py (new), eval_synth.py
      (new), scripts/compare_synth_eval.py (new), slurm/eval_synth.sh (new), tests/test_eval_synth.py
      (new), tests/test_slurm_scripts.py, tests/run_tests_wsl.sh.
      Verify: tf_gpu `python -m unittest tests.test_eval_synth` + goldens + `tests.test_trainer`
      validation tests; bgr_preproc38 TF-free tests + py_compile.
- [ ] 6. Final integration: full suite once in the background
      (`wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/tests/run_tests_wsl.sh`,
      ~3 h) plus the preprocessing suite; write `impl-report.md` (contents below).

## Test environment (all features)

Write multi-line commands into `.agents/tmp/<name>.sh` (gitignored) and run
`wsl -d Ubuntu bash -c "tr -d '\r' < /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.agents/tmp/<name>.sh | bash -s"`
(skill shell-and-remote-jobs). tf_gpu script body:
```
cd /mnt/d/Research/Brain_Registration_Projects/brain-group-rad
source ~/miniconda3/etc/profile.d/conda.sh; set +u; conda activate tf_gpu
export PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 BGR_TF_GPU_LIMIT_MB=6000 TF_CUDNN_USE_AUTOTUNE=0
python -m unittest <modules or Class names> -v
```
Never set BGR_TF_GPU_LIMIT_MB above 6000. One module per process if GPU memory runs out (the suite
does this). bgr_preproc38: same but `conda activate bgr_preproc38` (py3.8, numpy 1.21, no TF);
preprocessing tests are pytest (`python -m pytest preprocessing/tests/<file> -q -p no:cacheprovider`).
Baseline verified by the planner (2026-11, HEAD 4bfe227): preprocessing test_export + test_slurm_scripts +
test_cli 56 passed; tests.test_sweep_tools + tests.test_slurm_scripts 84 OK (bgr_preproc38).

## FINAL REPORT (impl-report.md) must contain

1. What was built (files), flags and defaults (table: `--synth-prob` 0, `--synth-jitter` 0.08,
   `--synth-device` gpu, `--synth-label-set` synthfine, `--synth-hier-dev-cortex` 0.25; checker
   `--expect-synth-prob`, `--min-task-steps`; script env SYNTH_PROB, SYNTH_JITTER, SYNTH_DEVICE,
   SYNTH_LABEL_SET, GPU_POLL, MIN_TASK_STEPS; r_level1 128G / 42500; export-add-labels arguments).
2. Test results: full-suite counts (tests / skipped / failing invocations / wall time), preprocessing
   suite counts, goldens max |diff| (must be 0), trace-once, e2e tiny run numbers (finite losses,
   trace counts) for p 0 and p 1.
3. Deviations from the design with reasons (D-a .. D-e plus anything found during implementation).
4. EXACT commands:
   (a) Jay, export (P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg; basic
   partition, -A qtim in the wrapper header, `%2` array cap = the 2-active-job rule; Jay's other jobs
   count, so check `squeue -u jn85` first):
   ```
   cd $P/brain-group-rad && git pull
   mkdir -p $P/logs
   N=$(( $(wc -l < $P/data/npy_v2/export_manifest.csv) - 1 )); echo $N    # 3423 expected
   sbatch --array=0-$(( (N + 499) / 500 - 1 ))%2 --export=ALL,CHUNK=500,LABEL_SET=synthfine preprocessing/slurm/add_labels_array.sh
   # only after the array has FINISHED (squeue -u jn85 shows none of its tasks); check
   # $P/data/npy_v2/add_labels/synthfine/rows-*.json and the logs for failed cases:
   sbatch --array=0-0%2 --export=ALL,PHASE=manifest,LABEL_SET=synthfine preprocessing/slurm/add_labels_array.sh
   # verification (all cases; synthfine consistency rule included):
   sbatch --array=0-0%2 --export=ALL,PHASE=check preprocessing/slurm/add_labels_array.sh
   ```
   (the coder of FEAT-001 adds PHASE=check = `export-check --export-root $EXPORT_ROOT --sample 0`
   to the wrapper so the check also runs on a compute node; the report must give the final form.)
   (b) Orchestrator: preview PNGs (`sbatch slurm/synth_preview.sh`, levels 4/2/1, output
   `$P/scratch/synth_preview/<job>/l<k>.png`), optional `sbatch slurm/verify_synth.sh`, then the paired
   smokes L4 -> L2 -> L1 exactly as design 7.2 (rtx8000, -A qtim in the script header, 1 GPU, 3 CPUs,
   --mem 64G L4/L2, 128G L1; --time 02:00:00 / 03:00:00 / 04:00:00; SYNTH_PROB 0 vs 0.5,
   `EXTRA_ARGS="--group-config B --lambda-smooth 0.5"`, L4 `DEFORM_START=100 DEFORM_RAMP=100
   CURRICULUM_RAMP=100 G_START=100 G_RAMP=100`, `MIN_TASK_STEPS=rigid:20,affine:20,deform:80` (L4) /
   `deform:40` (L2/L1), `GPU_POLL=true`; L2/L1 `CASCADE_FROM=$R/r<k+>syn_smoke_p$T/ckpt/last`; L1
   `TF_MEM_LIMIT_MB=42500 LR=3e-4`). State explicitly: the smokes need the synthfine export (files +
   manifest phase) to exist first; both jobs of a pair = the 2-job limit; a failing p 0 smoke stops the
   chain (design 7.2 rule).
   (c) Later: `RUNS="<Jay's real-only config B L1 run> $R/r1_cfgB_synth05" sbatch --job-name=evalsyn
   --time=08:00:00 --mem=128G slurm/eval_synth.sh` (final form from FEAT-005).

## Digest

What the planner read (current code, HEAD 4bfe227) and which tests cover it.

### preprocessing/luts.py
- `SYNTHSEG_STRUCTURES` (33 entries, raw ids 0..60), `DK_PARCELS` (35 names, ids 1001-1035 / 2001-2035),
  `_synthseg_entries()`, `_build(entries, source)` -> {"map": OrderedDict(raw -> contiguous), "names",
  "source"}; `TRAIN18_CLASSES` (cortex class 2 = [3, 42] + _DK_LH + _DK_RH); `_build_merged`;
  `LUTS = {"synthseg", "train18", "lumir10", "seg4", "seg35"}`; `apply_lut(arr, lut, dtype, what)`
  (rint float64, unknown ids -> ValueError); `lut_json`. Add `LUTS["synthfine"]`,
  `synthfine_to_train18_table()`, `synthfine_cortex_ids()`.
- Tests: preprocessing/tests/test_export.py `test_fixed_luts_are_dataset_independent`, train18 LUT test
  (~line 370).

### preprocessing/export.py
- Constants lines ~110-128: `ALL_LEVELS = (1, 2, 4)`, `SYNTHSEG_SETS = ("train18", "synthseg")`,
  `LABEL_SET_LEVELS = {"train18": (1, 2, 4), "synthseg": (1, 2)}`, `MANIFEST_COLUMNS` (BASE_COLUMNS +
  image/shape/labels/spacing/affine/world_center per level).
- `block_mode(lab, k, what)` (vectorised bincount, ties -> smallest id); `_atomic_save_npy`,
  `_atomic_write_text`, `_csv_text(rows, cols)`, `_npy_state(path, dtype, shape)` -> "missing" |
  "unreadable" | "mismatch" | "ok"; `_label_source`; in `export_npy` the per-set loop builds
  `make = full_labels if k == 1 else (lambda ...: block_mode(fl(), k, ...))` -> branch for synthfine
  to `synthfine_level(fl(), k)`.
- `merge_parts(export_root)`: reads `manifest_parts/*.csv` (+ `.luts.json`), keyed by
  (dataset, subject_dir), writes `label_luts.json` (sorted luts) and `export_manifest.csv` sorted by
  key, via `_csv_text(out, MANIFEST_COLUMNS)`. `export_check(export_root, sample, seed)` -> (n,
  problems, report); per row label files from `labels_l<k>` split on ';', dtype from `label_dtypes`.
- Tests: preprocessing/tests/test_export.py (`_subject(root, name, ..., ds="ds")` writes
  image.nii.gz + labels_synthseg.nii.gz with `phantoms.save`, SHAPE (8, 12, 16); `_manifest(exp)`).

### preprocessing/cli.py
- `build_parser()`: subparsers export-npy / export-merge / export-check; `select_rows` clamps end;
  `COMMANDS` dict; `cmd_export_merge` pattern (try/except ValueError -> log.error, return 1).
- Tests: preprocessing/tests/test_cli.py; `preprocessing/tests/run_tests_wsl.sh` loops `--help` over
  the command names (add export-add-labels).

### preprocessing/slurm/export_array.sh + preprocessing/tests/test_slurm_scripts.py
- Header: `#SBATCH --partition=basic`, `--account=qtim`, `##SBATCH --qos=PLACEHOLDER_QOS`, nodes 1,
  ntasks-per-node 1, time, `--mem=8G`, `--cpus-per-task=1`, `--array=0-0%2`, output/error under
  `$PROJ/logs/%x_%A_%a.*`; body `set -euo pipefail`, `export USE_GPU=${USE_GPU:-0}`,
  `export BGR_REPO_DIR="$REPO_DIR"`, `RIC=(bash "$REPO_DIR/preprocessing/run_in_container.sh")`,
  `cd "$REPO_DIR"`, `"${RIC[@]}" <command> ...`.
- Test `test_found_scripts` asserts the exact set of preprocessing slurm script names (add
  add_labels_array.sh); `test_slurm_placeholders` (parametrised over all) enforces the header/RIC rules
  and `%2`; `_export_argv` stubs run_in_container.sh to print argv (reuse for the new wrapper).

### data_loading/params.py
- `LEVEL_SHAPE`, `LEVEL_SPACING`, `N_TRAIN_LABELS = 19`, `IntensityAugConfig` dataclass,
  `INTENSITY_AUG_SEED_OFFSET = 7919`, `AUG_COLUMNS`. Add `SynthConfig`, `SYNTH_SEED_OFFSET = 104729`,
  `SYNTH_COLUMNS`, `SYNTH_DEVICES = ("gpu", "cpu")`. TF-free (imported by runtime/args.py).

### data_loading/tf_intensity_aug.py (pattern only)
- `draw_aug_params(generator, n)`: `generator.uniform(tf.stack([n, K]))` + `make_seeds(n)` transposed;
  `_apply_axis_matrix(x, p, axis)` (tensordot + transpose); `augment_group(images, cfg, generator)`
  `tf.map_fn(..., parallel_iterations=1, fn_output_signature=TensorSpec(images.shape[1:], float32))`.

### data_loading/postprocess.py
- `LinearPostprocess.__init__(task, level, flip_prob=0.5, p_pair=0.1, seed=0, target_iters=10,
  target_dtype=tf.float64, pixel_center=..., min_pairs="1", intensity_aug=None, max_pairs=0)`;
  `self.generator = tf.random.Generator.from_seed(seed)`, `_intensity_aug_setup` -> `aug_generator`.
  `__call__(self, images, t_dali, rt_dali=None, flip_axes=None)`: cast -> aug -> `t_mm = dali_to_mm`
  -> target -> flips -> barycentric targets -> pairs. Returns dict images/T_render/T_target/A_star/
  pair_i/pair_j/P_star/flip_axes.
- `DeformPostprocess.__init__(level, flip_prob, p_pdice, seed, n_classes=19, residual_affine, pixel_center,
  min_pairs, intensity_aug, max_pairs)`; `__call__(self, images, labels, t_dali=None, flip_axes=None)`:
  cast -> aug -> labels int32 (squeeze rank 5) -> flips -> pairs -> `tf.one_hot(labels, n_classes)`.
- Tests: tests/test_postprocess.py, tests/test_tf_intensity_aug.py.

### data_loading/dali_loader.py
- `linear_output_signature(task, level, batch_size, control=False)` -> (shapes, dtype names all
  float32); `base_output_count(task)` 3/2; `deform_output_signature(level, n, residual)`;
  `output_count(task, residual_affine=False)`; `numpy_reader_kwargs(files, file_root, shuffle,
  shuffle_mode, initial_fill, seed, name)`; `read_images(...)`; `paired_permutation(n, seed)` (=
  `RandomState(seed).permutation(n)`); `paired_reader_kwargs`; `_reader_to_gpu(kwargs, shape, dtype)`;
  `LABEL_READER_NAME`; `check_deform_inputs`. `LinearPipeline.__init__(task, level, files, batch_size,
  seed, file_root, shuffle, rotation_deg..., intensity_aug, num_threads, device_id,
  prefetch_queue_depth, initial_fill, pixel_center, exec_async, exec_pipelined, shuffle_mode,
  control_file)` permutes files with `RandomState(seed).permutation`; `define_graph`: read_images ->
  matrices (random or control) -> warp -> intensity_augment -> return `(imgs, t_fwd.gpu()[, rt.gpu()]) + extra`.
- Tests: tests/test_data_loading.py (DALI-free parts run locally; DALI parts skip).

### data_loading/data_module.py
- `manifest_files(rows, level, split, kind, label_set, contrast)` (labels: requires
  `labels_<set><sfx>.npy` in the `labels_l<k>` ';'-list, else ValueError naming the case);
  `LinearDataModule.__init__(export_root, level, batch_size, seed, split, shuffle, rigid_ranges,
  affine_ranges, intensity_aug, deform_residual_affine, num_threads, prefetch_queue_depth, device_id,
  contrast, manifest_path, shuffle_mode, control_dir, control_init, deform_batch_size, deform_ranges)`;
  `label_files` property (train18); `pipeline(task)`.

### runtime/args.py
- `build_parser()`: augmentation group (`--intensity-aug`, `--intensity-aug-device`, `--lowres-method`,
  ~line 459); `--shuffle-mode` (~549); `--group-config` (~551); `--overfit-one-group` (~587).
  `parse_args`: cross-checks via `parser.error` (e.g. lines 616-620). `train_config_from_args(args)`
  (~line 890-930) builds `TrainConfig(...)` incl. `intensity_aug_tf=intensity_aug_tf(args)`;
  `args_to_dict(args)` dumps every arg. Tests: tests/test_runtime_infra.py; TF-free import checked by
  tests/run_tests_wsl.sh.

### runtime/run.py (line numbers from grep)
- `TrainConfig` dataclass (494-597): add `synth: Optional[object] = None`.
- `bn_recal_plan(num_batches, ratio)` (313). Add module-level `synth_recal_slot(b, p)`.
- `Trainer.__init__` (638): postprocess construction loop (~752-768) -> pass `synth=cfg.synth`; set
  `self._synth_on = cfg.synth is not None` before `rebuild_steps()`.
- `checkpoint_variables` (777-785): add `postprocess_<t>_synth` when `pp.synth_generator is not None`.
- `_make_deform_step` (872-1000): `body(images, labels, t_dali=None, skip=None)` line 904,
  `batch = pp(images, labels, t_dali)` line 906 -> `self._pp_call(DEFORM, [images, labels] + ([t_dali]
  if residual else []))`; res gains `synth` when on; return lambdas 993-1000 unchanged.
- `_make_train_step` (1001-1122): `shapes, _ = dali_loader.linear_output_signature(...)` +
  `signature = [tf.TensorSpec(s, tf.float32) ...]` (1003-1004); `body(images, t_dali, rt_dali=None,
  skip=None)` line 1024, `batch = pp(images, t_dali, rt_dali)` 1026; lambdas 1111-1122 (cascade: rigid
  `lambda images, t_dali, rt_dali, skip: body(images, t_dali, rt_dali, skip)`, affine `lambda images,
  t_dali, skip: body(images, t_dali, None, skip)`; no cascade: rigid / affine without skip).
- `split_batch` (1123-1130): `k = dali_loader.output_count(task, cfg.deform_residual_affine)`.
- `train_step` (1149-1177): `res = {k: float(v.numpy()) ...}` except finite / n_vars_updated /
  lso_skipped -> `synth` arrives as float automatically.
- `_recal_input(self, task, inputs)` (1718-1729): `batch = self.postprocess[task](*inputs)`.
- `recalibrate_bn(num_batches, freeze, val=True)` (1753-1874): slot loop `for b, t in enumerate(plan)`,
  `inputs, _ = self.split_batch(t, batch_tuple)`, `x_in, finite = self._recal_input(t, inputs)`;
  event dict built at the end (num_batches, planned, batches, skipped_nonfinite, ...).
- `fit` (1979-): `win_steps[task] += 1` (2052) for every executed step; finite branch acc updates
  (~2096-2117); log window `_nonfinite_window_metrics` logs `train_<t>/window_steps`.
- Tests: tests/test_trainer.py (`config(**kw)` -> TrainConfig level 4 N 4; `TrainerCase.make`,
  `DeformCase.make_deform`; goldens `TestLinearOnlyUnchanged` / `TestDeformUnchanged` with
  `_OpDeterminism`), tests/test_cascade.py (`CascadeTrainerCase`, `TestSkip`, `TestBNRecal`,
  `TestChainTrainer`, `TestStageSkipLevel2`, `TestStageSkipChain`).

### main.py
- `_build_dali_data(args, level, tf_mods, start_step)` (154-205): builds `dm.LinearDataModule(...)`
  -> add `synth_label_set=args.synth_label_set` when synthesis is on; desc string.
- Resume branch (~line 383): `C.check_config_compatible`, `_check_intensity_aug_device`,
  `_check_optimizer` -> add `_check_synth_config(paths["config"], args)`.
- `config_extra` (~389) + `_optimizer_record` pattern -> add `config_extra["synth"] =
  dataclasses.asdict(cfg)` only when on. `C.write_config(paths["config"], model_cfg,
  A.args_to_dict(args), extra=config_extra)`; recal rewrite at ~435 uses the same extra.
- Tests: tests/test_main.py.

### runtime/fixed_data.py
- `ValidationSet.build(images, task, level, group_size, n_groups, ranges, seed, target_iters,
  global_transform)`: `rendered = render_group(images[idx], T, spacing)` -> hook
  `group_images_fn(g, idx)` replaces `images[idx]`. `DeformValidationSet.build(images, labels, level,
  group_size, n_groups, seed)`: `images[idx]` -> hook. `build_validation_sets(...)` passes through.
  `synthetic_label_phantom(shape, rng, n_labels=8)` -> labels 1..n (used as train18 classes by the new
  fine-label helper). `FixedGroupSource`, `to_dali_batch`, `to_deform_batch`.

### tests/gpu/check_mixed_run.py
- `check(run_dir, tasks, steps, ..., nonfinite_from_step=100)` (617): reads `logs/metrics.jsonl`
  records {step, data}, `fit_summary.json` trace counts (`trace_once` 679-681), `_check_bn_recal(recs)`;
  `main()` argparse (725-). `_check_nonfinite_windows` reads `train_<t>/window_steps`.
- Tests: tests/test_check_mixed_run.py (`make_run(root, tasks, ..., train_extra, train_fn, events)`).

### slurm/r_level{4,2,1}.sh, slurm/sweep_level4.py, tests/test_sweep_tools.py
- Env block (`X="${X:-default}"`), validators `need_uint`, `need_float` (FLOAT_RE), `need_choice`,
  `is_pos` (awk), `die` (exit 2); `MAIN=(main.py ... $MITIG_ARGS $EXTRA_ARGS)` (EXTRA_ARGS last);
  `CHECK=(tests/gpu/check_mixed_run.py ...)`; `BGR_DRY_RUN=1` prints `MAIN:` / `CHECK:` and exits before
  any write; main run `singularity exec --nv ... python "${MAIN[@]}" </dev/null || rc=$?`, checker
  `singularity exec ... python "${CHECK[@]}" </dev/null || crc=$?`.
- r_level1.sh: `#SBATCH --mem=64G`, `TF_MEM_LIMIT_MB="${TF_MEM_LIMIT_MB:-40000}"`, header lines
  mention "TF_MEM_LIMIT_MB 40000", "If the smoke peak exceeds 39000 MB", smoke/long lines with
  `--mem=64G`.
- sweep_level4.py `SCRIPT_ENV_KEYS` (line 61-70) must cover every `${NAME:-` of r_level4.sh
  (`test_script_env_keys_cover_r_level4`).
- tests/test_sweep_tools.py: `TestScriptsStatic.test_headers` (asserts `#SBATCH --mem=64G` for all
  three and r1 strings "36.7 GB", "MIN_PAIRS=1", "39000 MB", "TF_MEM_LIMIT_MB=34000");
  `TestScriptsDryRun.dry(path, **env)` (exactly one MAIN and one CHECK line, no file written),
  `check_parses(main)`, `test_r1_default` (asserts `--gpu-memory-limit-mb 40000`).

### tests/run_tests_wsl.sh
- `PYFILES` list (py_compile + ast 3.8), bash -n list of slurm scripts, TF-free unittest names
  (`tests.test_slurm_scripts`, `tests.test_sweep_tools`, ...), then one process per `tests/test_*.py`
  module in tf_gpu. New python files / scripts must be added to these lists.

### Verified by the planner
- bgr_preproc38: `python -m pytest preprocessing/tests/test_export.py preprocessing/tests/test_slurm_scripts.py preprocessing/tests/test_cli.py` -> 56 passed;
  `python -m unittest tests.test_sweep_tools tests.test_slurm_scripts` -> 84 OK.
- Last full suite per project state: 825 tests OK at 5b27491 (HEAD 4bfe227 adds scripts only).
