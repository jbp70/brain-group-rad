# Review FEAT-002: `predict.py run --tta flip8` (engine, runner, CLI, records)

Reviewed: uncommitted `git diff HEAD` of inference/streaming.py, runner.py, cli.py, predict.py, plus untracked tests/test_predict_tta.py, against design v2 (3.2-3.5, 5, 6, 7.1, 7.4, 7.5, 8, 12.1), plan FEAT-002 requirements and D4, D11, D13, D14. Evidence: test-log-FEAT-002.md (suites not re-run).

## Evidence

- `deform()` split (streaming.py): P0-P11 moved verbatim into `_v_raw_passes`, which returns `acc.s`; `deform()` divides by n and runs the unchanged P12. Golden: 70 hashes compared, 0 differ (test log, steps 1 and v1). End-to-end `--tta none` reproduces 66 NIfTI hashes, `tta_record` null, no CSV.
- `deform_tta`: `_put` allocates only if the name is absent (streaming.py 238-245), so flips 2-8 accumulate into the same `tta_s1`/`tta_s2` stores chunk by chunk without re-allocation. The shifted one-pass variance is algebraically right: c̄ = s1/F − m̄ = mean_f c_f, and σ² = s2/F − |c̄ − c0|² = mean_f |c_f − c̄|². P12 runs the cached `_f_p12` on s1/F and sm/F. `v_raw` is freed per flip, the TTA stores after the spread pass. fp16 maxima are the max over flips, and chunk boundaries come from flip 0. The tests assert that the store is empty after every run.
- Runner: `_tta_active` (TTA level and not `IDENTITY_ONLY`) gates every change. Without it, `_linear_stage` / `_deform_stage` call today's code (D13; T5 bitwise at L4 and L2). The free-after-last-P0 wrapper is installed only when `not _tta_active(lv)`. Otherwise `_deform_stage_freeing` frees `SRC_AFFINE` in `finally`. Both conditions use the same predicate, so `src_affine` is freed exactly once on each path. Level masks are computed before the first P0 from `src` (raw images or `src_affine`). Second moments are computed once per run from raw x1. `_linear_sel` / `_deform_sel` use the same helpers (D14).
- CLI: `--tta {none,flip8}`, `--tta-levels` parsed with `_levels`, `parser.error` for `--tta-levels` without flip8 and for flip8 with `--mode unchunked`, and `tta_spread.csv` in OUTPUT_NAMES. check-streaming has no TTA flags. The CLI stays TF-free (`test_validate_imports_no_tensorflow` passes).
- predict.py: `tta_record` is in PREDICT_KEYS, null by `_new_record` and in early error records. `_setup_tta` runs after the level peek, raises PredictError(2) for an unrun level with the record written (T10), and logs the training-flip warnings. The pipeline text and the TTA estimate are present (the estimate only applies with a deform task). `tta_spread.csv` is written atomically with the D4 empty-column rule and added to outputs, also on the non-finite path. The non-finite message names the flips. The rot_gap / readout_dev WARNINGs are present.
- Tests cover T5, T6 (g ∈ {x, y, z, xyz}, four branches, L4, L2 cascade with TTA at the last level, N 2 and 3), T7 (a)/(b)/(c) incl. gauge 19.6° for (b) vs 1.5e-17 for (a), and the v1 raw-averaging regression guard (9.9e-3 mm). They also cover a NaN flip, T10, and end-to-end none/flip8 at L4 and L2. All verification commands in the plan were run: 17 + 50 + 48 + 27 tests OK, py_compile and ast (3, 8) OK, `inference.tta` imports without TF.

## Tolerance deviation

The deviation is justified. Design T7's 1e-6 vox for u is tighter than float32 scaling and squaring allows. C6 in the design itself measured 2.2e-6 vox for exp alone. At L2 and after the affine pre-warp, the stub input is resampled with rounding-different upstream results. The measured values are 2.4e-6 to 3.4e-5 vox. The test still asserts 1e-6 for v at the first level without a pre-warp, which is where design C7's 2e-7 bound applies. Pairwise maps keep 1e-5 mm at L4 and use 1e-4 mm at L2 (measured 2.7e-5). The T6 tolerances (1e-4 mm / 1e-4 vox) are unchanged from the design, so behaviour is unchanged.

## Findings

No blockers.

- NIT: `deform_tta` leaves `tta_s1`/`tta_s2`/`tta_c0` in the store if an exception interrupts it, and the non-TTA `deform()` behaves the same way for its own stores. The process exits on that path, so this costs nothing today.
