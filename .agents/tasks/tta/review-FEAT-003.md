# Review FEAT-003: `predict.py linear-eval`

Verdict: APPROVED (no HIGH / MEDIUM findings; 1 NIT).

## Evidence checked

- `inference/linear_eval.py`, read in full against design 7.3, plan D9 and FEAT-003 step 2:
  - Subjects are read with `_read_subjects`. Labels are freed. `n < 2` and a non-level-1 grid raise PredictError(2).
  - The group mask `group_mask(images)` is computed on the un-rendered processed images (2·count >= N, i.e. mean >= 0.5) and passed to `brain_second_moment(m, 1.0)`. The mask is built before rendering, so S lives in template space as the design requires.
  - Transforms come from `random_linear_transforms(default_rng(seed), n, task)`, and the recorded ranges equal the defaults the function uses. `render_group(images[..., None], T, LEVEL_SPACING[1], chunk=2)` is a pull render I(T⁻¹y), matching design 7.3.
  - Targets are `barycentric_targets(T_target, 10)`, the same construction the training ValidationSet uses. Errors come from `group_errors(..., affine=task == 'affine')`.
  - Images are passed to `StreamedRunner.run` exactly as `_run_predict` passes them, so the array grid, the centred-mm convention and the plan/TTA wiring are unchanged from `run`. The store estimate and store choice mirror `run`, and the linear TTA store term is negligible (design 6).
  - Non-finite A records status `nonfinite` and exits 3 without writing the CSVs. In that case tta_spread.csv is still written for diagnostics, as `run` does.
  - In per_subject.csv, `pair_rms_mm_mean` averages over `i_idx == k`. That index is the moving image (`pairwise_errors` compares A_j⁻¹A_i with T_j T_i⁻¹), which matches design 7.3 / 9.3. per_pair.csv has the columns `moving`, `fixed`, `pair_*`.
  - `summary.json` is built from `LINEAR_EVAL_KEYS` and written last via `_command_wrapper`, also on failure. It carries the note, seed, ranges, tta_record (via the shared `_tta_finish` / `_write_tta_spread` from FEAT-002), timing, means and p95. Empty arrays give null.
- `inference/cli.py`:
  - `_common(..., branches=False)` means linear-eval has no `--branches` flag.
  - `validate` sets `args.branches = args.task` and `mode = 'streamed'` before the shared `_validate_out_dir`. That function applies overwrite protection over `LINEAR_EVAL_OUTPUTS`, checks `--store-dir`, and rejects `--tta-levels` without flip8.
  - The `run` checks were moved into the helper unchanged. `run`'s OUTPUT_NAMES gaining tta_spread.csv is FEAT-002 scope and was already approved. `check-streaming` is untouched. The module stays TF-free.
- `predict.py` dispatch and docstring, and the `inference/__init__.py` bullet, are as specified.
- `tests/test_linear_eval.py` covers all of the following:
  - both tasks × both arms at level 4;
  - level-2 cascade affine flip8 with spread rows at levels 4 and 2;
  - bit-identical rendered input across arms, via a sha256 recording wrapper on `FD.render_group`. The patch is effective because linear_eval looks up `FD.render_group` at call time;
  - one-subject exit 2 with the full key set;
  - CLI errors.
- Test log `test-log-FEAT-003.md` (not re-run):
  - test_linear_eval: 4 tests OK.
  - test_predict + test_predict_tta + test_predict_tta_golden: 55 tests OK; the 70-hash tta-off golden shows 0 differences, and the 66 E2E `--tta none` hashes equal the pre-TTA fixture.
  - compile / ast (3, 8): OK.
  - This covers every FEAT-003 verification command and acceptance criterion.

## Findings

- NIT (not blocking): `--overwrite` can leave stale outputs that the current summary does not describe. Two cases: a re-run with `--overwrite` that ends non-finite keeps an older run's per_subject.csv / per_pair.csv, and a `--tta none` re-run over a flip8 directory keeps the old tta_spread.csv. `summary.json` `outputs` lists only the files of the current run, so readers that follow it are safe. `run` behaves the same way with tta_spread.csv. Optional fix: delete the `LINEAR_EVAL_OUTPUTS` files that the current run will not write once `--overwrite` is accepted, or have compare_tta (FEAT-004) check `outputs` / `status` in summary.json.
