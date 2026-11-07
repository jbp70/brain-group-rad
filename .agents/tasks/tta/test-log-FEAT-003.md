# FEAT-003 test log: `predict.py linear-eval`

Date 2026-11-06/07, local WSL (GTX 1080, TF GPU limit 6000 MB via run_tests.sh). No commits (workflow rule). No verdict-FEAT-003.json existed, so this is the first implementation.

## Files changed

- `inference/linear_eval.py` (new): `LINEAR_EVAL_KEYS`, `NOTE`, `group_mask`, `run_linear_eval(args, configure_device=True)` via `predict._command_wrapper` (summary.json written last, also on failure). Body follows FEAT-003 step 2 and design 7.3.
- `inference/cli.py`: `_common(..., branches=True)`; subparser `linear-eval` (`--task {rigid,affine}` required, `--seed` int >= 0 default 0, `--tta`, `--tta-levels`, `--store`, `--store-dir`, `--out` dir required, `--overwrite`, plus every `_common` flag except `--branches`); `LINEAR_EVAL_OUTPUTS`, `LINEAR_TASKS`; `_existing_outputs(out, names, patterns)`; the `run` out-dir / store-dir / TTA checks moved into `_validate_out_dir` and reused by `linear-eval` (sets `args.branches = args.task`, `args.mode = "streamed"`). `run` and `check-streaming` behave as before.
- `predict.py`: dispatch `linear-eval`, docstring usage block.
- `inference/__init__.py`: docstring bullet `linear_eval`.
- `tests/test_linear_eval.py` (new, 4 tests).

## Verification

| Command | Result |
|---|---|
| `run_tests.sh tf tests.test_linear_eval` | 4 tests OK, 96.8 s |
| `run_tests.sh tf tests.test_predict tests.test_predict_tta tests.test_predict_tta_golden` | 55 tests OK, 6015 s; `[tta] E2E --tta none: 66 NIfTI hashes == pre-TTA fixture`; `[golden tta-off] 70 hashes compared, 0 differ (fixture commit 19b8a608)` |
| `run_tests.sh compile` | py_compile OK, ast feature_version (3, 8) OK (21 files incl. inference/linear_eval.py, tests/test_linear_eval.py), inference.tta imports without TF |
| ad-hoc py38 (bgr_preproc38) script, deleted afterwards | `cli.parse_args(['linear-eval', ..., '--tta', 'flip8', '--tta-levels', '4'])` -> branches rigid, mode streamed, tta_levels (4,), seed 0, no TF imported; `linear-eval --help` renders without `--branches`, `run --help` still has it |

The FAIL lines in the predict log are INFO messages of the tests that provoke the internal-check failure (exit 5) on purpose.

## Timings of the linear-eval runs (tiny random models, 3 phantoms 160x224x192)

| Run | wall | model phase | center / pair RMS mm |
|---|---|---|---|
| L4 rigid none | 7.7 s | 3.0 s | 19.58 / 32.92 |
| L4 rigid flip8 | 13.1 s | 8.3 s | 19.43 / 32.67 |
| L4 affine none | 6.6 s | 2.2 s | 19.86 / 33.76 |
| L4 affine flip8 | 13.5 s | 8.1 s | 19.70 / 33.49 |
| L2 cascade affine flip8 | 34.9 s | 26.8 s | (finite; spread rows at levels 4 and 2) |

Errors are large because the models are random (untrained); the tests only check finiteness and layout. Rendered-group sha256 identical in both arms for both tasks (a97cf0486beb... rigid, 8d3d13162935... affine).

## What the tests assert

- rigid and affine at level 4, `--tta none` and `flip8`: exit 0; summary.json key set == `LINEAR_EVAL_KEYS`; note text; task, seed 0, tta, ranges == `params.DEFAULT_LINEAR_RANGES[task]`; nonfinite {"A": 0}; finite means / p95 (center, pair, cycle; affine scale / shear); timing keys read, render, model, metrics, total; per_subject.csv 3 rows with the documented columns (affine adds center_scale_pct, center_shear_pct), `pair_rms_mm_mean` == mean of the subject's moving-image pairs; per_pair.csv 6 rows (all ordered pairs, moving / fixed); tta_spread.csv absent with none, 3 rows x levels with flip8, tta_record mode / levels / flips / spread keys.
- Same seed -> bit-identical rendered input in both arms (sha256 of `fixed_data.render_group`'s output, patched with a recording wrapper).
- Level-2 cascade affine flip8: exit 0, spread rows for levels 4 and 2, train_flip_prob {"4": 0.5, "2": 0.5}.
- One-subject subjects file: exit 2, summary.json status error with the full key set, no per_subject.csv.
- CLI: `--tta-levels` without flip8, missing `--task`, `--task deform`, `--branches`, `--seed -1`, existing summary.json / per_pair.csv without `--overwrite` -> SystemExit 2; `--overwrite` accepted; a `run` output (atlas.nii.gz) in --out does not block linear-eval.

## Deviations / notes

- Step 3 "factor the tta_spread.csv writer into `write_tta_spread`": FEAT-002 had already factored it as `predict._write_tta_spread(path, rows, subs)` inside `predict._tta_finish`. linear-eval calls `PR._tta_finish(rec, summary, args, subs)` (same writer, same tta_record spread / gauge / nonfinite_flips / timing_s update), so no rename was made.
- Fewer than 2 subjects: `evaluation.find_subjects` already raises for < 2 subjects, which `_read_subjects` maps to PredictError(2) ("subjects: need at least 2 subjects ..."). The explicit `n < 2` check in linear_eval stays as specified but is not reached through this path.
- With non-finite A the run exits 3 without per_subject.csv / per_pair.csv; tta_spread.csv is still written with flip8 (diagnostics, as `run` does).
- summary.json timing also has `load` and the runner's per-phase keys (e.g. `rigid:l4`) next to read / render / model / metrics; store record carries `peak_store_gb` and `peak_rss_gb`.
- `means` / `p95` cover every array of `group_errors` (incl. `cycle_rms_mm` and, for affine, `*_qr_rot_deg`); an empty array (cycles at N 2) gives null.
