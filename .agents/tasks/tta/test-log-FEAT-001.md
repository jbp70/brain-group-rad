# Test log FEAT-001 (TTA library, pre-change golden, store estimate)

## Iteration 1 (2026-11-06)

Local machine: Windows + WSL Ubuntu, GTX 1080 8 GB. Helper `.agents/tasks/tta/run_tests.sh` (new, LF line endings; `file` reports "Bourne-Again shell script, ASCII text executable"). TF env set by the helper: `TF_CPP_MIN_LOG_LEVEL=3 TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 BGR_TF_GPU_LIMIT_MB=6000 TF_CUDNN_USE_AUTOTUNE=0` (GPU limit 6000 MB, never higher).

### Step 1: T9 golden on the UNCHANGED tree (before any source edit)

Tree state: HEAD `19b8a608ca7e3d68693d842f53746b07ee00aff5`; `git status --short` showed only untracked `.agents/...` dirs and the new `tests/test_predict_tta_golden.py` (no source file modified).

Timestamp correction (review FEAT-001 NIT 1): the clock times written below for items 1-2 (~09:39-09:45, 09:45:55) are wrong. The fixture file's mtime, 2026-11-06 05:58:55 local (Windows and WSL `stat` agree), is the reliable time of the write; run_tests.sh and the golden test were created at 05:51, and the compare ran after 05:58:55 and before the first source edit (inference/tta.py, 09:56:33). The ordering requirement (golden written and compared on the unchanged tree) holds.

1. Write (finished 05:58:55 local per the fixture mtime; the "~09:39-09:45" recorded originally is wrong):
   `wsl -d Ubuntu bash -c "BGR_WRITE_GOLDEN=1 bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.agents/tasks/tta/run_tests.sh tf tests.test_predict_tta_golden"`
   ```
   [golden] wrote .../tests/fixtures/predict_tta_off_golden.json (70 hashes, commit 19b8a608ca7e3d68693d842f53746b07ee00aff5)
   [golden tta-off] 70 hashes compared, 0 differ (fixture commit 19b8a608ca7e3d68693d842f53746b07ee00aff5)
   Ran 1 test in 404.515s
   OK
   ```
2. Compare, same unchanged tree, without the env (after 05:58:55 and before 09:56:33 local; the "09:45:55" recorded originally is not reliable):
   `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.agents/tasks/tta/run_tests.sh tf tests.test_predict_tta_golden`
   ```
   [golden tta-off] 70 hashes compared, 0 differ (fixture commit 19b8a608ca7e3d68693d842f53746b07ee00aff5)
   Ran 1 test in 406.639s
   OK
   ```
   Fixture SHA256 identical before and after the compare run (E41635A0...9798; the compare run does not rewrite it).

Determinism rule used: **'hash'** (stored sha256 hashes, tolerance 0) under `tests.test_trainer._OpDeterminism` (`enable_op_determinism`). `enable_op_determinism` raised for no op of the prediction path, so neither the TF_CUDNN_DETERMINISTIC / TF_DETERMINISTIC_OPS fallback nor the in-process 'inprocess' mode was needed. Fixture keys: `commit`, `mode` = "hash", `tolerance` = 0, `hashes` (70 = 2 checkpoints x 3 branches x (3 subjects x fwd/inv/moved + atlas + atlas_seg) = 66 NIfTI + 4 A entries for rigid / affine+deform at l4 and l2). Hash = sha256(`np.asarray(dataobj).tobytes() + affine.tobytes()`); A = sha256 of `StreamedRunner(load_model(ck, b), RamStore()).run(images, b)["A"]` as float64 bytes.

### Steps 2-6: source edits, then verification

Probe (temporary scripts, deleted afterwards): the pre-change `estimate_store_gb` values (`git show 19b8a60:inference/store.py`, N 101, default `ModelConfig()`, every level / branch / dtype) were printed and copied as literals into `tests/test_predict_store.py::HEAD_ESTIMATES_N101`. A second probe showed the TTA delta per level: at level 1 with a float16 trainable stage the frozen float32 level-2 term is the max term without TTA, so the estimate grows by 10.86 GB (default widths) / 13.38 GB (d12hs widths: 39.70 -> 53.08 GB), not by the full level-1 term 19.46 GB; with a float32 trainable stage it grows by exactly 19.46 GB. The test asserts exactly that (see Deviations).

3. `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.agents/tasks/tta/run_tests.sh tf tests.test_tta` (2026-11-06T10:02:17-05:00): `Ran 33 tests in 316.171s` / `OK` (0 skipped).
4. `... run_tests.sh py38 tests.test_predict_store` (10:07): `Ran 15 tests in 0.135s` / `OK`, `TF-free tests OK without TF (15 tests)` (0 skipped). Printed:
   ```
   [estimate] level 1 default widths N 101 deform fp16: 44.7 GB (float32 54.2, affine+deform 47.5, affine 33.0)
   [estimate] level 1 N 101 deform with TTA at all levels: float32 73.67 GB (+19.46), float16 55.60 GB (+10.86)
   ```
   (A first run of this module failed in the new test_tta_term at (1, 'deform', float16): delta 10.86 vs the expected 19.46; this is the max-term effect described above, fixed in the test, not in store.py.)
5. `... run_tests.sh compile` (2026-11-06T10:07:56-05:00): files inference/*.py (incl. tta.py, store.py, __init__.py), evaluation/*.py, predict.py, tests/test_tta.py, tests/test_predict_tta_golden.py:
   ```
   py_compile OK
   ast feature_version=(3, 8) OK (18 files)
   inference.tta imports without TF
   ```
   (`tests/test_predict_store.py` is compiled by its py38 import in step 4.)
6. FEAT verification command after all edits (2026-11-06T10:08:07-05:00):
   `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.agents/tasks/tta/run_tests.sh tf tests.test_tta tests.test_predict_tta_golden`
   ```
   [golden tta-off] 70 hashes compared, 0 differ (fixture commit 19b8a608ca7e3d68693d842f53746b07ee00aff5)
   Ran 34 tests in 689.214s
   OK
   ```
   0 skipped. `python -c "import inference.tta, sys; assert 'tensorflow' not in sys.modules"` passes in bgr_preproc38 (compile mode) and in tf_gpu (tests.test_tta.TestConfig.test_import_is_tf_free).

Existing goldens `tests/fixtures/m7_linear_golden.json`, `lncc_deform_golden.json`: untouched (no training code changed).

### Key numbers printed by tests.test_tta (iteration 1)

| Check | Value | Bound |
|---|---|---|
| C3 Jay's x / y / z vectors == derived, combined == products | exact (array_equal) | exact |
| C1 / C2 float64 max \|M(p s_f) - F M(p) F\| rigid / affine | 0.00e+00 / 0.00e+00 | < 1e-12 |
| C1 / C2 float32 | 0.00e+00 / 0.00e+00 | < 1e-5 |
| C4 center(F A F) vs F center(A) F, N 12 float64 | 0.00e+00 | < 1e-10 |
| C5 transform_image flip, level 4 float32 | 1.01e-05 | < 1e-4 |
| C6 exp(v^F) vs (exp v)^F (max \|v\| 3.83 vox) / warp | 2.21e-06 vox / 5.14e-06 | < 1e-5 / < 1e-4 |
| unflip_field involution, flip_np == tfm.flip_images | bitwise | bitwise |
| E1 round trip rigid / affine | 1.67e-16 / 4.44e-16 | < 1e-12 |
| E2 encode(F M F) vs sigma_f encode(M) | 0.00e+00 | < 1e-12 |
| C7 mean_linear of 8 identical encodings (max rel diff), rot_gap min, readout_dev max | 4.44e-16, 0.999999999999998, 1.41e-14 deg | < 1e-15, ~1 |
| eigh read-out of base * R(+-30 deg) vs base; readout_dev_deg | 3.62e-14 deg; 2.6 / 2.58 / 2.66 deg | < 1e-9 deg; > 0 |
| C7 float32 running sum of 8 identical fields / 8 (rel) | 1.49e-07 | < 2e-7 |
| E3 N 6: v1 raw averaging pair err / v2 pair err / v2 centered diff | 0.121 mm / 3.02e-14 mm / 4.00e-15 | v1 > 0.05, v2 < 1e-8 |
| E3 N 2: v1 / v2 / centered | 0.198 mm / 4.20e-14 mm / 1.78e-14 | v2 < 1e-8 |
| E4 equivariance (rigid + affine, first and later level, g = x, y, z, xyz) | 1.24e-14 | < 1e-8 |
| E5 4-iteration centering: pair err / centered translation diff | 3.20e-14 mm / 2.33e-08 mm | < 1e-8 / < 1e-5 |
| identity-only == two-line path (rigid, affine; first and later level) | max \|diff\| 0 | 0 |
| T8 rot_deg_rms / scale_pct_rms / shear_pct_rms (want 3 / 2 / 1) | 3.000000 / 2.000000 / 1.000000 | places 9 |

### Deviations recorded in this iteration

- `linear_tta_level(..., dtype=None)`: extra keyword (default = a_prev's dtype, float32 at the first level, i.e. today's behaviour). Needed so the E3/E4/E5 tests can run the whole pipeline in float64 (< 1e-8 mm bounds); with the model path's float32 tensors it is a no-op.
- `linear_spread(..., mean_diag=None)`: optional keyword so rot_gap / readout_dev_deg (from `mean_linear`) come out in the same dict.
- `decode_linear(p_bar, task, dtype=np.float32)`: dtype keyword (float64 used by the tests and by `linear_spread`).
- T11 level-1 check: the FEAT text expected "+19.46 GB at level 1 N 101 default widths"; that holds only when level 1 is the max term (float32 trainable stage). With the default float16 trainable stage the frozen float32 level-2 term dominates without TTA, so the estimate (max over levels, design 6 rule) grows by 10.86 GB (default widths) / 13.38 GB (d12hs widths, 39.7 -> 53.1 GB, vs design 6's "59.2 GB"). store.py implements the design rule as written (28 V added to the per-level deform term, then the max); the test checks +19.46 GB with float32 and 0 < delta < 19.46 GB with float16.
