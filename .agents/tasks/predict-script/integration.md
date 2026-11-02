# predict-script: integration verification

## Iteration 1 (2026-11-01, HEAD 7fdc80c = FEAT-001..005)

No seam fixes were needed. No code changed in this iteration; tests/fixtures (trainer goldens) unchanged.

### Seams checked by reading
- `slurm/predict_mindboggle.sh` against `inference/cli.py`: every flag it passes exists (`--checkpoint`, `--processed-dir`, `--grid-reference`, `--precision float32|checkpoint`, `--n-subjects`, `--chunk-l4/l2/l1`, `--branches`, `--out`).
- The same script against `evaluation/evaluate.py build_parser`: check-convention (`--fwd`, `--inv`, `--moved`, `--out`) and run (`--method`, `--atlas`, `--atlas-seg`, `--components`, `--signs`, `--units`, `--pairs none`, `--workers`, `--out`) match. The printed `slurm/eval_mindboggle.sh` command uses its env names (METHOD, FWD, INV, ATLAS, ATLAS_SEG, CONV).
- `inference/predict.py` uses the FEAT-003 loader/runner API (`load_model`, `check_unchunked`, `StreamedRunner`/`UnchunkedRunner.run(images, branches, sink)`, summary keys `chunks`, `chunk_sizes`, `capped`, `stats.max_abs_v`, `fp16_max_abs`, `timing_s`, `gpu_peak_mb`) and the FEAT-002 store API (`estimate_store_gb`, `choose_store`, `peak_nbytes`, `close`) as documented in the FEAT findings.

### Quick checks on HEAD (local, before the suite)
- bgr_preproc38 (py3.8): `py_compile` and `ast.parse(feature_version=(3, 8))` over predict.py, inference/*.py, evaluation/*.py, models/layers.py, tests/predict_helpers.py, tests/test_predict*.py, tests/test_slurm_scripts.py, tests/test_evaluation.py: OK (23 files). `bash -n slurm/predict_mindboggle.sh` OK. `predict.py --help`, `run --help` and `check-streaming --help` OK without TF. `tests.test_slurm_scripts` + `tests.test_predict_store`: 50 tests OK, TensorFlow not imported.
- tf_gpu, one process per module: test_evaluation 12 OK (4 s), test_predict_store 13 OK, test_slurm_scripts 37 OK (10 s), test_predict_loader 6 OK (90 s; re-run because FEAT-004 extended tests/predict_helpers.py), test_predict_streaming 16 OK (453 s).
- test_predict (31) and test_predict_runner (7) ran inside the full suite below (their FEAT-004 runs were on 57b60c1; FEAT-005 touched no Python code they import).

### Full suite
`wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/tests/run_tests_wsl.sh` (log .agents/tmp/full_suite.log):
- Stage 1 (bgr_preproc38): py_compile OK, ast 3.8 OK (108 files), ast 3.6 OK (2 login-node helpers), bash -n OK for all scripts incl. slurm/predict_mindboggle.sh and slurm/eval_mindboggle.sh, TF-free tests OK (154 tests, incl. tests.test_predict_store).
- Stage 2 (tf_gpu, python 3.9.13, TF 2.10.0, numpy 1.24.3, scipy 1.13.0, nibabel guard OK): every module exit 0.
- **unittest summary: 802 tests (4 skipped), 0 failing invocations. suite status: 0**, unittest wall time 19236 s (5 h 21 min).
- New modules in the suite: test_predict 31, test_predict_loader 6, test_predict_runner 7, test_predict_store 13, test_predict_streaming 16 (73 tests); test_evaluation 12 and test_slurm_scripts 37 also pass.
- Regression modules named in the FEATs: test_cascade 47, test_trainer 68 (goldens unchanged), test_split_group_conv 5, test_group_layers 15, test_fp16_mitigations 17: all OK.
- Skips (4): test_curriculum 1, test_data_loading 2 (DALI not installed locally), test_runtime_infra 1; the 2 deferred GPU-memory classes ran in their own processes (OK).

Note for later steps: the suite now takes ~5.3 h locally (was ~3 h before this task); the predict modules add ~1.1 h.
