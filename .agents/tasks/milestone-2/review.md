# Linear DALI pipelines, data module and post-DALI processing (review pass 2)

Milestone 2 adds the `data_loading/` package: one `LinearPipeline` for rigid and affine (one group of N images per DALI batch, an independent T_i about the image center per image), a pure-TF `postprocess` that converts the DALI matrices to centered mm, applies one flip per group and builds barycentric and pairwise targets, a manifest-driven `LinearDataModule`, and a glacier GPU verification script with its runner. This pass checks the iteration-2 fixes for the four pass-1 issues and looks for new problems in the changed code. All four are addressed: the runner redirects `KERAS_HOME` into `scratch/m2`, the reader defaults to `shuffle_after_epoch=True` (`shuffle_mode='epoch'`, with the old buffer behavior kept as `'buffer'`), check (b) asserts rotation and translation bounds, and a `unittest` runner mode puts `TMPDIR` under scratch. Watch for: design 3.2 still says `random_shuffle=True`, which no longer matches the default (confirmed, non-blocking).

**Verdict**: APPROVED

## High-level view

The matrix algebra is unchanged from pass 1 and still matches design section 2 and `models/transforms.py`. The graph builds T = C+·Tr·R·K·C−, Rt = C+·Tr·R·C− and a factor-by-factor pull T⁻¹, with C± at (n−1)/2 + `DALI_PIXEL_CENTER` in reversed (DALI) axis order. Translations are in mm and divided by the level spacing (15 mm → 3.75/7.5/15 voxels). I re-derived the K and K⁻¹ DALI-order rows from U·D and D⁻¹U⁻¹, and they match `scale_shear_rows`. T_i = Rt_i·S_i is documented in the module, the numpy mirror and the report. The rigid task emits both T (for rendering) and Rt (for targets), `LinearPostprocess` applies F·M·F to both, and the targets come only from Rt, so the network never undoes S_i.

Groups are exactly N in both shuffle modes. `pad_last_batch=False` lets the reader roll into the next epoch, the data module rejects splits smaller than N, and `initial_fill` only appears in `'buffer'` mode. `shuffle_after_epoch` exists on `fn.readers.numpy` in DALI 1.18 and excludes `random_shuffle`, which `numpy_reader_kwargs` forces to False; a local unit test pins this. DALI's per-epoch reshuffle uses its own fixed seed, and the seeded Python pre-permutation keeps the first epoch seed-dependent, so (f)'s determinism check covers both modes. The DALI ops and arguments used all exist in 1.18. `external_source` and `einops` appear nowhere, and the DALI import is guarded (1 clean skip locally).

The verification script is genuine and independent. Check (a) reads a numpy phantom through the real graph helpers and compares against TF `transform_image` on the numpy array over 8 candidates (pc 0.5/0, pull with T⁻¹/T, reversed/same axes). The candidates vary the conversion of the matrix DALI actually emitted, so a wrong centering constant shows up as a different winner rather than being masked. Check (b) loads the source `.npy` files and the template mask with numpy. Its new magnitude bounds are sound: rigid is measured on Rt (≤ 45° + 0.01°). Affine allows √3·shear rad (about 5°) on top of 45° for the polar factor of R·U·D, which is a strict upper bound; the actual excess is at most about 2.7°. The per-axis |t| ≤ 15 mm + 1e-3 also catches centering errors, since a mis-centered rotation adds mm translation. Writes stay inside `scratch/m2`: the phantom goes in `mkdtemp(dir=--scratch)`, and the runner redirects TMPDIR, the CUDA cache, KERAS_HOME and bytecode, each with its `SINGULARITYENV_` twin.

The verification evidence holds up. `test-log.txt` (14:01:45) postdates the last code edit (`tests/test_data_loading.py`, 13:59:35). It shows 61 tests, 60 passed and 1 skipped, plus a Python 3.8.20 py_compile and `ast` (3.8) check of 16 files, the no-DALI import check and `bash -n` on the runner. HEAD equals origin/main (92c85a7), and the milestone files are untracked or unstaged, so nothing was committed or pushed. Nothing in the repo points to an ssh session; that rests on the report's statement.

<details>
<summary>Issues (2)</summary>

1. **Design 3.2 shuffle wording is stale** (non-blocking, confirmed): design 3.2 says "numpy reader with `random_shuffle=True`", but the default is now `shuffle_after_epoch=True`. Update the sentence when the milestone is committed, so the design stays the single source of truth.
2. **`shuffle_after_epoch` not yet executed** (non-blocking, likely): the new default reader mode has only been checked as a kwargs dict locally. Read the glacier `f_determinism` result for `*_epoch` and the container unittest's `test_pipeline_runs_on_synthetic_files` before relying on it in milestone 4. If DALI rejects the argument combination, `shuffle_mode='buffer'` is the fallback.

</details>

<details>
<summary>Details</summary>

### Shuffle mode change and the design deviation

The deviation from design 3.2 is documented in the report and reversible per pipeline (`'buffer'`), but the design text wasn't updated (Issue 1). Because DALI's per-epoch reshuffle uses a fixed internal seed, the rigid and affine pipelines apply the same permutation sequence after epoch 1. They start from different seeded pre-permutations, though, so their groups still differ.

### Border handling in check (b)

The `direct` single-interpolation comparison in (b) (max |DALI − TF| < 2e-3) is taken over the full volume, not the mask. It therefore assumes the DALI and TF border handling only disagree where the intensity is zero. The skull-stripped LUMIR backgrounds make that likely, but if (b) fails only on `direct_max_abs` while the NCCs pass, the cause is border handling, not a convention bug.

### Test coverage

Local coverage is unchanged except for `test_reader_kwargs_shuffle_modes`, which pins the exclusivity of the two shuffle arguments and the absence of `initial_fill` in `'epoch'` mode. The DALI unittest now runs both modes across an epoch boundary (4 files, groups of 2, 3 runs), but it only executes in the container. Not tested locally: all DALI ops, intensity augmentation, `DALIDataset` wiring and `LinearDataModule.iterator`. These are deferred to glacier checks (a)–(g) and the container unittest run.

</details>

<details>
<summary>File map</summary>

- `data_loading/params.py`: level constants, `DALI_PIXEL_CENTER`, ranges (mm→voxel), `IntensityAugConfig`, DALI matrix formulas and their numpy mirror.
- `data_loading/dali_loader.py`: guarded DALI import, `numpy_reader_kwargs` and `SHUFFLE_MODES`, graph helpers, `LinearPipeline`.
- `data_loading/data_module.py`: manifest file lists, `make_dali_dataset`, `LinearDataModule` (`shuffle_mode`), memory growth.
- `data_loading/postprocess.py`: DALI→mm, group flips, pair sampling, `LinearPostprocess`, `brain_S_matrix`.
- `tests/test_data_loading.py`, `tests/test_postprocess.py`: unit tests (the DALI test skips without DALI).
- `tests/gpu/verify_dali_pipeline.py`: GPU checks (a)–(g), with magnitude bounds in (b) and both shuffle modes in (f).
- `tests/gpu/run_verify_glacier.sh`: single-GPU container runner with `verify` and `unittest` modes, confined to `scratch/m2`.
- `tests/run_tests_wsl.sh`: the syntax check now covers `data_loading/` and `tests/gpu/`; adds an import check and `bash -n`.

Full diff: `git diff` plus the untracked files (`git status --short`).

</details>
