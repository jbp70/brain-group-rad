# synth-contrast implementation log

Every coder / fixer step appends: date, FEAT id, iteration, each command run (as executed) and its
result (test counts, OK/FAIL, golden max |diff|, dry-run lines). No commits, pushes or cluster jobs.

## Planner baseline (HEAD 4bfe227)
- bgr_preproc38: `python -m pytest preprocessing/tests/test_export.py preprocessing/tests/test_slurm_scripts.py preprocessing/tests/test_cli.py -q -p no:cacheprovider` -> 56 passed
- bgr_preproc38: `python -m unittest tests.test_sweep_tools tests.test_slurm_scripts` -> Ran 84 tests, OK

## FEAT-001 iteration 1 (2026-11-04): exporter, synthfine, export-add-labels, add_labels_array.sh
Files: preprocessing/luts.py, preprocessing/export.py, preprocessing/add_labels.py (new), preprocessing/cli.py,
preprocessing/slurm/add_labels_array.sh (new, LF), preprocessing/tests/test_add_labels.py (new),
preprocessing/tests/test_export.py, preprocessing/tests/test_slurm_scripts.py, preprocessing/tests/test_cli.py
(help list), preprocessing/tests/run_tests_wsl.sh (--help loop).
- bgr_preproc38 (script .agents/tmp/feat001_pytest.sh): `python -m py_compile preprocessing/add_labels.py
  preprocessing/export.py preprocessing/luts.py preprocessing/cli.py` -> OK; `bash -n preprocessing/slurm/add_labels_array.sh`
  -> OK; `python -m pytest preprocessing/tests/test_add_labels.py preprocessing/tests/test_export.py
  preprocessing/tests/test_slurm_scripts.py preprocessing/tests/test_cli.py -q -p no:cacheprovider`
  -> first run 1 failed / 74 passed (resolve_range treated `--start 1000` without `--end` as start > end;
  fixed: start > end only checked when --end is given) -> rerun 75 passed.
- `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/preprocessing/tests/run_tests_wsl.sh bgr_preproc38`
  -> 167 passed (87 s), bash -n of every preprocessing .sh OK, `--help` loop incl. export-add-labels OK,
  "ALL CHECKS PASSED (bgr_preproc38)".
- Manual CLI check (temp tree /tmp/feat001_manual.*, 3 cases lumir/LUMIRMRI_000{0,1,2}_0000 with
  image.nii.gz + labels_synthseg.nii.gz + meta.json + manifest_processed.csv, built with the test helper):
  `python -m preprocessing export-npy --out-root $T/processed --export-root $T/npy_v2 --dataset lumir` -> exit 0
  (3 written, train18 only); `export-add-labels --out-root ... --export-root ... --label-set synthfine --phase files
  --start 0 --end 2` -> "rows [0, 2) of 3: 2 written, 0 kept, 0 failed", exit 0; `... --start 2 --end 500` ->
  "rows [2, 3) of 3: 1 written", exit 0 (status rows-2-500.json); `... --phase manifest` -> backup
  backup_pre_synthfine_20261104T235624Z, "3 rows, 3 changed, columns changed
  label_dtypes,label_levels,labels,labels_l1,labels_l2,labels_l4, 1 parts changed, 0 problems", exit 0;
  `export-check --export-root $T/npy_v2 --sample 0` -> "label max synthfine=102 (0..102), train18=18",
  "checked 3 cases: OK", exit 0; manifest row 1: labels train18;synthfine, label_levels
  train18:1,2,4;synthfine:1,2,4, labels_l4 labels_train18_4.npy;labels_synthfine_4.npy; second
  `--phase manifest` -> "every row already lists synthfine; nothing to do", exit 0. Temp tree deleted.

## FEAT-002 iteration 1 (2026-11-04): synthesis core + post-process integration
Files: data_loading/params.py (SynthConfig + constants), data_loading/synth.py (new), data_loading/postprocess.py
(`synth=None`, keyword-only `labels=` / `synth_override=`, `_synth_setup`, `_synth_draw`, linear render, deform
fine->train18 gather), tests/runtime_helpers.py (`fine_labels_from_train18`, `write_fake_export(fine_labels=)`),
tests/test_synth.py (new, 25 tests), tests/test_postprocess.py (+TestSynthCallSignature, 2 tests).
- bgr_preproc38 (script .agents/tmp/feat002_py38.sh): `python -m py_compile data_loading/params.py data_loading/synth.py
  data_loading/postprocess.py tests/test_synth.py tests/runtime_helpers.py tests/test_postprocess.py` -> OK;
  `ast.parse(feature_version=(3, 8))` over data_loading/*.py + tests/*.py -> 62 files, 0 failures;
  `python -c 'import data_loading.params, sys; assert "tensorflow" not in sys.modules'` -> OK (SynthConfig defaults
  printed: prob 0.0, jitter 0.08, ..., hier_dev 0.1, hier_dev_cortex 0.25, norm_quantile 0.995, device gpu, label_set synthfine).
- tf_gpu (BGR_TF_GPU_LIMIT_MB=6000): `python -m unittest tests.test_synth -v` -> Ran 25 tests, OK (150 s; first run, no failures).
- tf_gpu: `python -m unittest tests.test_synth tests.test_postprocess tests.test_tf_intensity_aug -v` -> Ran 81 tests, OK (217 s).
  Covered: OneHot depth multiset on == off on the full graph-def (outer graph + library functions) for linear and
  deform, with and without `--lowres-method tent` aug (tent depths 40/56/48 seen in nested functions, deform depth
  19 once); negative control (fine one-hot depth 103 inside cond/map_fn) makes both comparisons differ; trace once
  with p 0.5, seed 3, 8 calls, synth True and False seen (linear rigid and deform); synth_override=False bit-equal to
  synth=None (rigid, affine, deform; with TF aug; fixed and drawn flips); render consistency (== transform_image of
  the unwarped synthetic group with T_mm^-1, atol 1e-5; targets bit-equal to off); CPU device == default on CPU and
  traced graph has CPU-placed nodes; prob 0 graph == synth=None graph; keyword-only TypeError checks.
- tf_gpu: `python -m unittest tests.test_trainer.TestLinearOnlyUnchanged tests.test_trainer.TestDeformUnchanged -v`
  -> first attempt (same script, right after the 81-test run) stalled inside TestDeformUnchanged for > 25 min at
  100 % GPU and was interrupted by the tool timeout (KeyboardInterrupt; TestLinearOnlyUnchanged had already printed
  `[golden] max |diff| 0 (tolerance 0)`); rerun alone (.agents/tmp/feat002_golden.sh) -> Ran 2 tests in 113 s, OK,
  `[golden] max |diff| 0 (tolerance 0)` and `[golden deform] max |diff| 0 (tolerance 0)`.

## FEAT-003 iteration 1 (2026-11-04): DALI labels, data module, CLI flags, Trainer / main wiring, e2e runs
Files: data_loading/dali_loader.py (`linear_output_signature(..., labels=)`, `base_output_count` /
`output_count(..., labels=)`, `LinearPipeline(label_files=)`, `check_linear_label_inputs`), data_loading/data_module.py
(`synth_label_set`, `synth_label_files`), runtime/args.py (5 `--synth-*` flags, validation, `synth_config_from_args`),
runtime/run.py (`TrainConfig.synth`, `synth_recal_slot`, `Trainer._synth_on`, `_pp_call`, uint8 linear signatures + the
four synthesis-on lambdas, split_batch, `synth` step result, synth_frac / loss_real / loss_synth window logs,
`postprocess_<t>_synth` checkpoint keys, BN-recal synthetic slots + event counts), main.py (`synth_label_set` only when on,
desc suffix, `_check_synth_config`, config.json `synth` record only when on); tests: test_data_loading (+TestSynthData 4),
test_runtime_infra (+TestSynthArgs 3), test_main (+TestSynthMain 3), test_trainer (+TestSynthTrainer 6, helpers
`fine_label_group`, `_seed_with_synth_outcomes`), test_cascade (+TestSynthSkip 1, TestSynthRecal 3, TestSynthSkipChain 1).
- bgr_preproc38 (.agents/tmp/feat003_py38.sh): `python -m py_compile data_loading/*.py runtime/*.py main.py tests/*.py` -> OK;
  `ast.parse(feature_version=(3, 8))` over data_loading/runtime/tests/main.py -> 75 files, 0 failures; `python main.py --help
  >/dev/null` -> OK (the five --synth-* flags listed); `python -c 'import runtime.args, sys; assert "tensorflow" not in
  sys.modules'` -> OK.
- tf_gpu (BGR_TF_GPU_LIMIT_MB=6000, .agents/tmp/feat003_new.sh, new tests only, first run, no failures):
  TestSynthData + TestSynthArgs + TestSynthMain -> Ran 10, OK (42 s); tests.test_trainer.TestSynthTrainer -> Ran 6, OK
  (469 s); TestSynthSkip + TestSynthRecal -> Ran 4, OK (246 s); TestSynthSkipChain -> Ran 1, OK (136 s; level-1 rigid
  forced-skip loss 2266.5112 == level-4-only chain 2266.5112 with synthesis p 1).
- tf_gpu verification (.agents/tmp/feat003_verify.sh, one process per command, logs .agents/tmp/feat003_{group,trainer,cascade}.log):
  - `python -m unittest tests.test_data_loading tests.test_runtime_infra tests.test_main tests.test_postprocess tests.test_synth -v`
    -> Ran 173 tests in 4808 s, OK (skipped=3: DALI-only tests).
  - `python -m unittest tests.test_trainer -v` -> Ran 74 tests in 4370 s, OK; `[golden] max |diff| 0 (tolerance 0)` and
    `[golden deform] max |diff| 0 (tolerance 0)`.
  - `python -m unittest tests.test_cascade -v` -> Ran 52 tests in 3702 s, OK (includes the unchanged guards TestSkip,
    TestChainTrainer, TestStageSkipLevel2, TestStageSkipChain, TestBNRecal; their bodies were not edited).
- e2e tiny runs (TestSynthTrainer.test_e2e_p1_and_p0; level 4, N 4, narrow model, rigid:1, affine:1, deform:2, 8 steps,
  flips on, seed from `_seed_with_all_tasks`): every step finite, trace_counts {rigid: 1, affine: 1, deform: 1} for both;
  final losses p 1: rigid 2219.33, affine 1198.53, deform 0.414402 (history synth == 1.0 every step; window logs
  train_<t>/synth_frac 1 and loss_synth, no loss_real); p 0 (synth None): rigid 2214.32, affine 1198.6, deform 0.415442 (no
  synth key anywhere). (The first targeted run printed 2219.32 / 1198.56 / 0.414375 and 2214.32 / 1199.25 / 0.415423: the
  run is not under op determinism, GPU float noise.) p 0.5 (12 steps): history flags == the simulated per-task generator
  draws, each task's window synth_frac in (0, 1), loss_real / loss_synth = means over the matching steps. SynthConfig(prob=0)
  vs default (4 three-task steps under `_OpDeterminism`): bit-equal. Train-step OneHot depth multisets on == off: rigid
  {4: 1}, affine {4: 1}, deform {19: 1}.

## FEAT-004 iteration 1 (2026-11-05): checker, r_level scripts, r_level1 128G / 42500, GPU check s, preview tool
Files: tests/gpu/check_mixed_run.py (`task_steps`, `parse_min_task_steps`, `--expect-synth-prob`, `--min-task-steps`,
`bn_recal_synth`), tests/test_check_mixed_run.py (+TestSynthContrast 7), slurm/r_level{4,2,1}.sh (identical SYNTH_* /
GPU_POLL / MIN_TASK_STEPS block, header paragraph + paired smoke lines of design 7.2), slurm/r_level1.sh (OD3: `--mem=128G`,
TF_MEM_LIMIT_MB 42500, header reasons, threshold 41500 MB, usage lines 128G), slurm/sweep_level4.py (SCRIPT_ENV_KEYS + the six
names + CUDA_VISIBLE_DEVICES), tests/test_sweep_tools.py (headers 128G/64G, r1 strings, 42500, +6 synth dry/stub tests),
tests/gpu/verify_dali_pipeline.py (check s1-s3, `--synth-label-set`), slurm/verify_synth.sh (new), scripts/synth_preview.py (new),
slurm/synth_preview.sh (new), tests/test_synth_preview.py (new, 2), tests/test_slurm_scripts.py (+TestSynthScripts 4),
tests/run_tests_wsl.sh (PYFILES + bash -n list).
- bgr_preproc38 (.agents/tmp/feat004_preproc.sh): py_compile of the 8 changed/new .py -> OK; ast feature_version (3, 8) on 7
  files + sweep_level4 (3, 6) -> OK; `bash -n` slurm/r_level4.sh r_level2.sh r_level1.sh verify_synth.sh synth_preview.sh
  tests/run_tests_wsl.sh -> all OK; `python -m unittest tests.test_sweep_tools tests.test_slurm_scripts -v` -> Ran 94 tests, OK
  (baseline 84 + 6 sweep + 4 slurm; first run, no failures; rerun after the preview-title edit: 94 OK).
- tf_gpu (BGR_TF_GPU_LIMIT_MB=6000, .agents/tmp/feat004_tf.sh): `python -m unittest tests.test_check_mixed_run tests.test_synth_preview -v`
  -> Ran 46 tests, OK (incl. NIT 1 test_task_steps_nit1: 4 records x 25 deform steps, deform:100 PASS / deform:101 FAIL);
  rerun after the title edit -> 46 OK.
- Default dry runs vs HEAD (.agents/tmp/feat004_bytecmp.sh, `git show HEAD:slurm/r_level<k>.sh`): r_level4 / r_level2 output
  byte-identical; r_level1 differs only by `--gpu-memory-limit-mb 40000` -> `42500` (OD3).
- Dry runs (.agents/tmp/feat004_dry.sh, PROJ/REPO/RESULTS/DATA/SIF = temp dir T, CASCADE_FROM=$T/ckpt for L2/L1; nothing written
  under T besides ckpt):
  - default (all three): MAIN/CHECK without any --synth-* / --expect-synth-prob / --min-task-steps, no POLL line.
  - L4 `BGR_DRY_RUN=1 RUN_NAME=r4syn_smoke_p05 STEPS=300 VAL_EVERY=100 WARMUP=50 DEFORM_START=100 DEFORM_RAMP=100
    CURRICULUM_RAMP=100 G_START=100 G_RAMP=100 PROJECT_STEPS=16000 PROJECT_VAL_EVERY=500 WD=1e-1 SYNTH_PROB=0.5 GPU_POLL=true
    MIN_TASK_STEPS=rigid:20,affine:20,deform:80 EXTRA_ARGS='--group-config B --lambda-smooth 0.5'`:
    MAIN ends `... --filters 16,64,128,192,256,384 --decoder-widths 128,96,64,16,8 --synth-prob 0.5 --synth-jitter 0.08
    --synth-device gpu --synth-label-set synthfine --group-config B --lambda-smooth 0.5`;
    CHECK ends `... --grad-balance-from-step 200 --expect-init-weights none --expect-synth-prob 0.5 --min-task-steps
    rigid:20,affine:20,deform:80`;
    `POLL: nvidia-smi -i 0 --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader,nounits -l 5 >
    $T/results/r4syn_smoke_p05/gpu_poll.csv`.
  - L2 (r2syn_smoke_p05, STEPS=200 ..., deform:40): MAIN ends `--decoder-widths 128,96,64,16,8 --synth-prob 0.5 --synth-jitter 0.08
    --synth-device gpu --synth-label-set synthfine --group-config B --lambda-smooth 0.5`; CHECK ends `--expect-cascade true
    --expect-skip true --expect-init-weights full --expect-bn-recal true --expect-synth-prob 0.5 --min-task-steps
    rigid:20,affine:20,deform:40`; POLL line as above with r2syn_smoke_p05.
  - L1 (r1syn_smoke_p05, LR=3e-4 TF_MEM_LIMIT_MB=42500 ...): MAIN `... --gpu-memory-limit-mb 42500 --filters ... --decoder-widths
    128,96,64,16,8 --deform-decoder-dtype float32_fine --synth-prob 0.5 --synth-jitter 0.08 --synth-device gpu
    --synth-label-set synthfine --group-config B --lambda-smooth 0.5`; CHECK ends `--expect-cascade true --expect-cascade-stages 2
    --expect-skip true --expect-init-weights full --expect-bn-recal true --expect-synth-prob 0.5 --min-task-steps
    rigid:20,affine:20,deform:40`; POLL line with r1syn_smoke_p05.
  - `BGR_DRY_RUN=1 bash slurm/synth_preview.sh` -> 3 lines `PREVIEW: singularity exec -B /autofs -B /cluster -B /space $SIF python
    scripts/synth_preview.py --data $PROJ/data/npy_v2 --level <4|2|1> --n 8 --seed 0 --groups 2 --split train --label-set
    synthfine --out $PROJ/scratch/synth_preview/manual/l<k>.png`, exit 0, nothing written.
- Local preview (.agents/tmp/feat004_preview.sh, tf_gpu, standalone process, GPU hidden by tf.config.set_visible_devices, no
  crash): fake export (write_fake_export n_train 4, labels + fine labels) -> `python scripts/synth_preview.py --data
  .agents/tmp/feat004_preview/export --level 4 --n 4 --groups 2 --seed 0 --out .agents/tmp/feat004_preview/synth_preview_l4.png`
  -> rc 0, PNG 720x569 (rows real + 2 MR-blur groups, FWHM ~2 mm, one remasked; visually checked); level 2 on that level-4 export
  -> rc 2 "no level-2 images in ...". Missing synthfine labels -> rc 2 "no 'synthfine' labels at level 4 ... (run
  export-add-labels first)" (unit test).
- Not run locally: verify_dali_pipeline check s (needs DALI + GPU; cluster only via slurm/verify_synth.sh).

## FEAT-005 post-training eval_synth (iteration 1)

Files: runtime/fixed_data.py (`group_images_fn` hooks in ValidationSet.build / build_validation_sets /
DeformValidationSet.build, default None = today's path), data_loading/synth.py (EVAL_PRESETS 11, PRESET_NAMES,
parcel_fine_ids, preset_seed, preset_mean_table_np, preset_params; preset table in the module docstring),
runtime/eval_run.py (new), eval_synth.py (new), scripts/compare_synth_eval.py (new), slurm/eval_synth.sh (new),
tests/test_eval_synth.py (new), tests/test_slurm_scripts.py (TestStatic.test_eval_synth_header, bash -n list,
TestEvalSynthScript), tests/run_tests_wsl.sh (PYFILES + eval_synth.py / scripts/compare_synth_eval.py,
`eval_synth.py --help` without TF, bash -n slurm/eval_synth.sh).

Commands (scripts .agents/tmp/feat005_preproc.sh, .agents/tmp/feat005_tf.sh; BGR_TF_GPU_LIMIT_MB=6000):
- bgr_preproc38: `python -m py_compile eval_synth.py runtime/eval_run.py scripts/compare_synth_eval.py runtime/fixed_data.py
  data_loading/synth.py tests/test_eval_synth.py tests/test_slurm_scripts.py` -> OK; ast feature_version (3, 8) -> OK.
- bgr_preproc38: `python -c "import sys, eval_synth, runtime.eval_run; assert 'tensorflow' not in sys.modules"` -> OK;
  `python eval_synth.py --help` -> rc 0 (usage printed, no TF); `python scripts/compare_synth_eval.py --help` -> rc 0.
- bgr_preproc38: `bash -n slurm/eval_synth.sh` -> first run FAILED (apostrophe inside `${DATA:-runs' own}`), fixed -> OK.
- bgr_preproc38: `BGR_DRY_RUN=1 RUNS='a b' bash slurm/eval_synth.sh` -> rc 0, one line
  `MAIN: python eval_synth.py --runs a b --level 1 --out /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results/eval_synth/manual`.
- bgr_preproc38: `python -m unittest tests.test_slurm_scripts` -> 46 tests OK (first run: 2 failures from the bash bug above).
- tf_gpu: `python -m unittest tests.test_eval_synth -v` -> first run: 1 FAIL (test compared a model_config dict, tuples vs
  lists; now ModelConfig.from_dict equality) + an error from the class attribute `run` shadowing TestCase.run (renamed
  run_dir); final: Ran 14 tests in 221.6 s, OK (incl. the tiny end-to-end eval: exit 0, both runs x 2 presets, equal
  sha256, CSV, run dir unchanged (file listing + mtimes); failed sanity gate -> exit 5, JSON written, run dir unchanged).
- tf_gpu: `python -m unittest tests.test_trainer.TestLinearOnlyUnchanged tests.test_trainer.TestDeformUnchanged
  tests.test_trainer.TestResumeAndValidation -v` -> Ran 7 tests in 324.8 s, OK; `[golden] max |diff| 0`,
  `[golden deform] max |diff| 0`.
- tf_gpu: `python -m unittest tests.test_trainer.TestMixedTasks -v` -> Ran 2 tests in 212.4 s, OK.

## Final integration (2026-11-05): full suite + preprocessing suite, no seam fixes
Working tree = HEAD 4bfe227 + FEAT-001..005 (uncommitted; 32 modified files, 14 new). All new/changed .sh files LF.
- `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/preprocessing/tests/run_tests_wsl.sh bgr_preproc38`
  (via .agents/tmp/final_preproc_suite.sh, log .agents/tmp/final_preproc_suite.log) -> 167 passed, 0 failed, 0 skipped
  (118 s), "ALL CHECKS PASSED (bgr_preproc38)", exit 0.
- `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/tests/run_tests_wsl.sh` (background,
  .agents/tmp/final_full_suite.sh, log .agents/tmp/final_full_suite.log; started 01:40:54, ended 08:25:38) -> exit 0.
  Stage 1 (bgr_preproc38): py_compile OK, ast feature_version (3, 8) OK (119 files), (3, 6) OK (2 login-node helpers),
  data_loading / runtime.args / main.py --help / eval_synth.py --help without TF OK, bash -n OK for all 20 listed
  scripts (incl. verify_synth.sh, synth_preview.sh, eval_synth.sh), TF-free tests OK without TF (181 tests).
  Stage 2 (tf_gpu, one process per module): "unittest summary: 910 tests (4 skipped; ...), 0 failing invocation(s): none",
  "suite status: 0, unittest wall time: 24221 s" (6 h 44 min). Skips: test_curriculum 1, test_data_loading 2 (DALI),
  test_runtime_infra 1 (all pre-existing / DALI; test_pair_recompute + test_val_chunk deferred memory tests ran in their
  own processes). New / touched modules: test_synth 25, test_synth_preview 2, test_eval_synth 14, test_trainer 74,
  test_cascade 52, test_postprocess 36, test_main 44, test_check_mixed_run 44, test_data_loading 30, test_runtime_infra 38,
  test_slurm_scripts 46, test_sweep_tools 52, test_tf_intensity_aug 20 - all exit 0.
  Goldens: `[golden] max |diff| 0 (tolerance 0)` and `[golden deform] max |diff| 0 (tolerance 0)`.
- Seam fixes: none needed (no failure). impl-report.md written; task.json status completed.
