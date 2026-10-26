# Verification: tf-intensity-aug (incl. addenda A pair recompute, B level-1 validation memory)

Local machine: WSL Ubuntu, GTX 1080, TF limit 6000 MB (`BGR_TF_GPU_LIMIT_MB`), envs tf_gpu (TF 2.10,
py3.9) and bgr_preproc38 (py3.8). Nothing committed; no cluster access.

## Goldens (default `--intensity-aug-device dali`, all new flags off)
- `tests.test_trainer.TestLinearOnlyUnchanged`: `[golden] max |diff| 0 (tolerance 0)` (targeted run and
  both full suites).
- `tests.test_trainer.TestDeformUnchanged`: `[golden deform] max |diff| 0 (tolerance 0)` (same).

## Python 3.8 / scripts
- bgr_preproc38: `py_compile` + `ast.parse(..., feature_version=(3, 8))` OK for every changed / new .py
  (22 files); `data_loading.intensity_ref` imports without TF; `python main.py --help` works without TF
  and lists `--intensity-aug-device`, `--pair-recompute`, `--val-image-chunk`, `--bn-recal-validate`.
- `bash -n slurm/m9_level1_cascade.sh slurm/m8_level2_cascade.sh` OK.
- TF-free suite stage (memory model, width presets, step times, SLURM scripts): 69 tests OK, no TF import.

## Targeted tests (tf_gpu)
- `tests.test_tf_intensity_aug` (15): OK. T4 low-res TF vs two-stage numpy max |diff| 1.66e-07.
  T10 KS p (mean / std / p1 / p50 / p99 / frac0 / frac1): 1, 0.111, 0.468, 0.998, 0.131, 0.994, 0.367.
  T14 transient peak: level 2 N 8 60.9 MiB (group 26.2 MiB); level 1 N 2 157.5 MiB (group 52.5 MiB).
- `tests.test_postprocess` + `tests.test_trainer` (TFIntensityAug, goldens, ResumeAndValidation): 35 OK.
- `tests.test_main.TestIntensityAugDevice`, `test_verify_dali_level`, `test_measure_memory`,
  `test_runtime_infra`, `test_postprocess.TestIntensityAugPostprocess`: 56 OK (1 skip).
- `tests.test_pair_recompute` (6): OK. Gradients max relative diff 2.4e-06. GPU peaks (64x80x64, N 4,
  pair LNCC): default sampler default 1 / 4 pairs 1740 / 2278 MiB, recompute 1889 / 1964 MiB; lean
  sampler default 1040 / 1438 MiB, recompute 970 / 968 MiB.
- `tests.test_val_chunk` (2): chunked == unchunked metrics; level 2 TINY N 2 (fixed 6000 MB): train peak
  4703 MiB, validation peak unchunked 1345 MiB -> chunk 1 / pair chunk 1 839 MiB; train -> validate ->
  train ran in both modes.
- `tests.test_memory_model` + `tests.test_slurm_scripts` (bgr_preproc38): 56 OK.
- `tests.test_main` alone: 25 OK (4010 s).

## Short training run with `--intensity-aug-device tf`
Level 4, fixed groups through the Trainer with `intensity_aug_tf` (rigid:affine:deform 1:1:2, narrow
widths, 30 steps, validation at 15 / 30): all 30 steps finite, validation finite (deform template /
pair Dice 0.865 / 0.774), aug generators advanced for all three tasks. (`main.py` rejects
`--intensity-aug-device tf` with `--overfit-one-group`, and the DALI path cannot run locally, so the
run used the Trainer directly.)

## Full suite (`tests/run_tests_wsl.sh`)
- Run 1 (`suite.log`, 12057 s): 558 tests, 1 failure + 1 error, 4 skips.
  - FAIL `test_pair_recompute.TestPairRecomputeMemory`: 4-pair growth 221 MiB > 0.35 x default. Cause:
    the serialising dependency `0 * s` was constant-folded by Grappler. Fixed (`s * z`, z a
    data-dependent zero); re-measured (numbers above), passes.
  - ERROR `test_main.TestLevel1CascadeErrors.test_level1_cascade_from_errors`: eager level-1 deform
    build pass (N 2) OOM inside the 6000 MB region; passes alone and in `tests.test_main` alone.
- Run 2 (`suite2.log`, 11621 s, test code loaded before the test fix below): 558 tests, only the same
  ERROR; goldens max |diff| 0; pair-recompute memory test passed.
  - Test fix: the test only checks the error routing after the build, and the variables do not depend
    on N, so its build pass is patched to N 1 (half the activations). Passes alone (2 OK).
- Run 3 (`suite3.log`, 11238 s, final code): stage 1 OK (69 TF-free tests, py3.8 checks, bash -n);
  `Ran 558 tests ... OK (skipped=4)`, `unittest exit code: 0`, script exit 0 (baseline was 520 OK,
  4 skips; +38 new tests). Goldens max |diff| 0.
