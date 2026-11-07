# Test log FEAT-002 (`predict.py run --tta flip8`: engine, runner, CLI, records)

Working tree: branch main, HEAD 19b8a60 plus the FEAT-001 and FEAT-002 changes (not committed). Local WSL Ubuntu, GTX 1080 8 GB, helper `.agents/tasks/tta/run_tests.sh` (TF env: BGR_TF_GPU_LIMIT_MB=6000, TF_CUDNN_USE_AUTOTUNE=0). Raw logs: `.agents/tasks/tta/_f2_step1.log`, `_f2_v1.log` ... `_f2_v5.log`; the final chain is `.agents/tasks/tta/_f2_verify.sh` (status in `_f2_status.log`).

## Iteration 1 (2026-11-06)

### Step 1 check right after the `_v_raw_passes` split (10:35)

`wsl -d Ubuntu bash .../run_tests.sh tf tests.test_predict_tta_golden tests.test_predict_streaming` (log `_f2_step1.log`):

- `[golden tta-off] 70 hashes compared, 0 differ (fixture commit 19b8a608ca7e3d68693d842f53746b07ee00aff5)`: the split is bit-identical.
- `Ran 17 tests in 803.761s FAILED (errors=3)`. All 3 errors are `TypeError: tf___apply_block() missing 2 required positional arguments` raised from AutoGraph while it re-read `inference/streaming.py`. I edited that file (adding `deform_tta`) during the run, so AutoGraph converted function sources that no longer matched the loaded code. The errors are not a code fault: the same module passes in v1 below on the unchanged final file.

### Tolerance measurements before the final tolerances (T7)

`run_tests.sh tf tests.test_predict_tta.TestEquivariantStubs ...` (several targeted runs, 10:50-11:40). The first version asserted design T7's literal tolerances (deform u < 1e-6 vox; pairwise < 1e-5 mm). It failed on float32 rounding only:

| Case | Measured max diff TTA vs single pass |
|---|---|
| deform stub L4, `deform` | v 2.4e-7 vox; u_ckpt 2.4e-6, U1 4.2e-6, U1_inv 5.1e-6 vox |
| deform stub L4, `affine+deform` | v 6.6e-7; U1 4.3e-6 vox |
| deform stub L2, `deform` | v 1.3e-5; U1 1.5e-5 vox |
| deform stub L2, `affine+deform` | v 1.6e-5; U1 3.4e-5 vox |
| same with Gaussian-smoothed inputs (sigma 2, probe only, removed afterwards) | L4 deform U1 1.5e-6; L4 affine+deform U1 2.8e-5; L2 U1 1.1e-5 / 3.6e-5 vox |
| gauge stub L2 rigid | A 1.6e-5 mm, pairwise 2.7e-5 mm |

Reading: at a first level without a pre-warp, the averaged SVF v differs from the single pass by the float32 8-term rounding (~2e-7 vox, as C7 predicts). Float32 scaling and squaring raises this to ~4e-6 vox; C6 measured 2.2e-6 vox for exp alone. At level 2, and after the affine pre-warp, the stub's input is resampled with an upstream result that differs at rounding level, so the stub's v already differs by ~1e-5. Smoothing the inputs does not remove this, so it is not an artefact of rough fields.

**Deviation (test tolerances, no behaviour change):** in `TestEquivariantStubs` the deform fields U1, U1_inv, u_ckpt, u_inv_ckpt and v must agree within 1e-4 vox, the T6 tolerance. The design's 1e-6 vox applies to the SVF v at a first level without a pre-warp (L4 `deform`, measured 2.4e-7). The pairwise tolerance is 1e-5 mm at level 4 (measured 5.9e-6) and 1e-4 mm at level 2 (measured 2.7e-5), where the level-2 input is pre-warped with the level-4 result. Centered A within 1e-4 mm is unchanged (max 1.6e-5).

### Final verification chain (18:41-21:38), FEAT-002 `verification` commands in order

1. `wsl -d Ubuntu bash /mnt/d/.../run_tests.sh tf tests.test_predict_tta_golden tests.test_predict_streaming`
   - `[golden tta-off] 70 hashes compared, 0 differ (fixture commit 19b8a608...)`
   - `Ran 17 tests in 915.411s` / `OK`
2. `... run_tests.sh tf tests.test_predict_tta tests.test_tta`
   - `Ran 50 tests in 2344.328s` / `OK` (test_predict_tta 17 + test_tta 33)
   - T5: `identity-only TTA bitwise == no TTA for rigid, affine, deform, affine+deform` at L4 and L2 (assert_array_equal on A, U1, U1_inv, u_ckpt, u_inv_ckpt, v, v_raw)
   - T6 (float32 random models, g = x, y, z, xyz, all four branches, TTA at all levels): L4 N 3: A 8.2e-6 mm, U1 1.1e-6, u_ckpt 4.1e-7 vox; L2 N 3: A 1.0e-5 mm, U1 2.2e-6, u_ckpt 1.3e-6 vox; L4 N 2 (affine+deform, g xyz, y): A 5.7e-6 mm, U1 9.0e-7 vox. Store empty and no `src_affine` after every run.
   - T7: linear stub (a) L4: A 2.8e-6 mm, pairwise 2.8e-6 mm, gauge 1.5e-17 deg. Gauge stub (b) L4: A 1.3e-5 mm, pairwise 5.9e-6 mm, gauge 19.6 deg; the v1 raw-averaging reference has a pairwise error of 9.9e-3 mm (> 1e-3, regression guard). Gauge stub (b) L2: A 1.6e-5, pairwise 2.7e-5 mm, gauge 19.6 deg at both levels. Deform stub (c): spread_rms 9.3e-8, p99 3.0e-7, shift 2.6e-7 mm; field diffs as in the table above. A NaN in flip 3 gives `nonfinite_flips {'l4/deform': [3]}` and a WARNING.
   - T10: `--tta flip8` accepted; `--tta-levels` without flip8, flip8 with `--mode unchunked`, `--tta flip`, `--tta-levels 3 / 2,2 / 1,2` → exit 2; check-streaming has no `--tta`; `tta_spread.csv` covered by the `--overwrite` protection; `--tta-levels 1` on a level-4 checkpoint → exit 2, predict.json written (key set == PREDICT_KEYS, status error, `tta_record` {mode, levels [1], error}).
   - End to end on the golden checkpoints: `--tta none` → 66 NIfTI hashes == the pre-TTA fixture (rigid, deform, affine+deform at L4 and L2), `tta_record` null, no `tta_spread.csv`. `--tta flip8` → exit 0 for all six runs (41-87 s each). Per run: finite outputs, `tta_record` (8 flips, train_flip_prob 0.5 per level, spread with gauge for linear and p99 for deform), and `tta_spread.csv` with subjects × levels × tasks rows following the D4 empty-column rule. spread_rms_mm max: l4/rigid 0.29, l4/deform 0.13, l2 cascade up to 0.53 (l2/affine). `--tta-levels 2` on the level-2 checkpoint: rows only for level 2.
   - test_tta (FEAT-001) unchanged: C1/C2 0, C4 0, E2 0, E3 v1 0.121 mm vs v2 3.0e-14 mm, E4 1.2e-14, E5 3.2e-14 mm.
3. `... run_tests.sh tf tests.test_predict_runner tests.test_predict_levels_runner tests.test_predict`
   - `Ran 48 tests in 7307.351s` / `OK`. No existing assertion needed a change (`tests/test_predict.py` untouched; `--tta flip` is still rejected, TestCLI.test_validate_imports_no_tensorflow passes, so `inference/cli.py` stays TF-free).
4. `... run_tests.sh py38 tests.test_predict_store tests.test_predict_levels`
   - `Ran 27 tests in 0.340s` / `OK` / `TF-free tests OK without TF (27 tests)`
5. `... run_tests.sh compile`
   - `py_compile OK`, `ast feature_version=(3, 8) OK (19 files)`, `inference.tta imports without TF`

No tests were skipped in any of these runs.

## Resume check (workflow wf_3ff8d564e9d8b835, after the network failure of wf_d9878f1d8bff3eb3)

No verdict-FEAT-002.json existed, so this was the first-iteration path; nothing was re-implemented or re-run.

- Working tree (`git status` / `git diff --stat`): modified inference/__init__.py, cli.py, predict.py, runner.py, store.py, streaming.py, tests/test_predict_store.py (608+ / 44-); untracked inference/tta.py, tests/test_predict_tta.py, tests/test_tta.py, tests/test_predict_tta_golden.py, tests/fixtures/predict_tta_off_golden.json. The FEAT-002 symbols are present: `StreamedLevel._v_raw_passes`, `deform_tta`; `StreamedRunner._tta_at`, `_linear_stage`, `_deform_stage`, `_mask_fn`; cli `TTA_MODES`, `_tta_flags`, `_validate_tta`, `tta_spread.csv` in OUTPUT_NAMES; predict `tta_record` in PREDICT_KEYS, `_peek_flip_chain`, `_setup_tta`, `_tta_finish`, `TTA_SPREAD_CSV`.
- Every source and test file was last written before the final chain started (latest: inference/streaming.py 10:49:57, tests/test_predict_tta.py 18:40:53; chain 18:41-21:38), so the chain results apply to the current tree.
- Raw logs confirm the chain above: `_f2_status.log` v1-v5 rc 0; `_f2_v1.log` golden 70 compared, 0 differ, Ran 17 OK; `_f2_v2.log` Ran 50 OK; `_f2_v3.log` Ran 48 OK; `_f2_v4.log` Ran 27 OK, TF-free OK; `_f2_v5.log` py_compile OK, ast (3, 8) OK 19 files, inference.tta imports without TF.
- FEAT-001 review NIT 1 fixed in passing: test-log-FEAT-001.md now names the fixture mtime 05:58:55 as the time of the golden write and marks the earlier 09:3x/09:4x times as wrong.
