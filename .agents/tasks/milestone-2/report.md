# Milestone 2 report: linear DALI pipelines, data module, post-DALI processing

Iteration 2. Nothing committed or pushed, no ssh. All work is local WSL. No local env has DALI, so every DALI code path is unexecuted until the glacier run (see "Glacier-only").

## Iteration 2: review fixes

1. **Keras home (blocking).** `run_verify_glacier.sh` now exports `KERAS_HOME=$SCRATCH/keras` and `SINGULARITYENV_KERAS_HOME`, so a TF/Keras import inside the container writes `keras.json` under `scratch/m2` instead of `~/.keras`.
2. **Shuffle diversity.** New `shuffle_mode` argument on `LinearPipeline`, `LinearDataModule` and `read_images`:
   - `'epoch'` (new default): `shuffle_after_epoch=True` (exists in DALI 1.18 `fn.readers.numpy`; exclusive with `random_shuffle`). The whole file list is reshuffled every epoch and there's no host shuffle buffer. The seeded Python pre-permutation still makes the first epoch seed-dependent (DALI's per-epoch reshuffle uses its own fixed seed, so it stays deterministic).
   - `'buffer'`: the iteration-1 behavior (`random_shuffle=True` with `initial_fill`, about 1 GiB of host memory).
   - The reader arguments come from a pure-Python `numpy_reader_kwargs`, unit-tested locally. GPU check (f) now checks determinism for both modes, and the DALI unittest runs both modes across an epoch boundary.
   - Deviation from the brief: the brief said `random_shuffle=True`. At levels 2 and 1 the affordable buffer is only about 312 / 39 samples, so groups would come from the same neighbors every epoch. `'buffer'` restores the brief's literal behavior.
3. **Check (b) magnitudes.** Check (b) now fails if any sampled rotation exceeds 45° (+0.01° for rigid, measured on Rt; + √3·shear rad for affine, the polar factor of R·K) or any |t| per axis exceeds 15 mm + 1e-3. This catches degree/radian, centering or combine-order errors directly.
4. **Container unittest /tmp.** `run_verify_glacier.sh` takes a third argument `unittest`. That mode runs `python -m unittest discover -s tests -t . -v` in the container with `TMPDIR`/`SINGULARITYENV_TMPDIR=$SCRATCH`, so the tests' `tempfile.mkdtemp()` folders land in `scratch/m2`. It logs to `scratch/m2/unittest.log`.

## Files

New:
- `data_loading/__init__.py`
- `data_loading/params.py`: pure Python. Level constants, `DALI_PIXEL_CENTER`, range defaults, mm→voxel conversion, `IntensityAugConfig`, and the DALI-side matrix formulas plus their numpy mirror.
- `data_loading/dali_loader.py`: `LinearPipeline` and the graph helpers. It imports without DALI.
- `data_loading/data_module.py`: manifest file lists, `DALIDataset` construction, `LinearDataModule`, `enable_memory_growth`.
- `data_loading/postprocess.py`: pure TF. Conversion, group flips, targets, pairs, `brain_S_matrix`.
- `tests/test_data_loading.py`, `tests/test_postprocess.py`
- `tests/gpu/verify_dali_pipeline.py`, `tests/gpu/run_verify_glacier.sh`. `tests/gpu` has no `__init__.py`, so unittest discovery skips it.

Modified:
- `tests/run_tests_wsl.sh`:
  - the syntax check now also covers `data_loading/` and `tests/gpu/`;
  - added an import check of `data_loading` without TF/DALI;
  - added `bash -n` on the glacier script.

Unchanged: `models/`, `preprocessing/`. The `models/transforms.py` DALI conversion is used as is (default `pixel_center=0.5`). It stays marked UNVERIFIED until check (a) passes on glacier.

## Public API

`data_loading.params`:
- `LEVELS = (4, 2, 1)`, `LEVEL_SHAPE`, `LEVEL_SPACING` (4/2/1 mm), `level_suffix(k)` ('', '_2', '_4'), `check_level`, `mm_to_voxels(mm, level)`.
- `DALI_PIXEL_CENTER = 0.5`. This is the single constant used both for centering the DALI transforms and for the TF conversion.
- `DEFAULT_LINEAR_RANGES` and `linear_ranges(task, level, rotation_deg=None, translation_mm=None, log_scale=None, shear=None) -> LinearRanges(..., translation_vox)`.
  - rigid: ±45°, ±15 mm, log ±log 1.005, shear ±0.0025;
  - affine: ±45°, ±15 mm, log ±log 1.10, shear ±0.05.
- `DEFAULT_DEFORM_RESIDUAL_RANGES` (1°, 1 mm, log 1.005, 0.0025) and `DEFAULT_DEFORM_RESIDUAL_AFFINE = False`.
- `IntensityAugConfig`, a dataclass with an enable flag and a probability per augmentation. Defaults are the reference values: noise p 0.1 (σ ≤ 0.05), brightness p 0.15 [0.75, 1.25], contrast p 0.15 [0.5, 1.5], low-res p 0.125 (factor ≥ 0.5), gamma p 0.3 [0.7, 1.5]. `IntensityAugConfig.off()` disables all of them.
- `dali_center(shape_xyz, pc)`, `scale_shear_rows(dx, dy, dz, a, b, c, zero)`, `linear_matrices_np(...)` (numpy mirror of the graph), `auto_initial_fill(n_files, N, shape)`.

`data_loading.dali_loader` (requires DALI at construction; `DALI_AVAILABLE`, `require_dali()`):
- `LinearPipeline(task, level, files, batch_size, seed=0, file_root=None, shuffle=True, rotation_deg=None, translation_mm=None, log_scale=None, shear=None, intensity_aug=None, num_threads=4, device_id=0, prefetch_queue_depth=1, initial_fill=None, pixel_center=DALI_PIXEL_CENTER, ..., shuffle_mode='epoch')`
  - `.output_signature()`;
  - outputs: rigid `(images, T_dali, Rt_dali)`, affine `(images, T_dali)`.
- `linear_output_signature(task, level, N)`: rigid `((N,X,Y,Z,1), (N,3,4), (N,3,4))`, all float32.
- `SHUFFLE_MODES = ('epoch', 'buffer')`, `numpy_reader_kwargs(files, file_root, shuffle, shuffle_mode, initial_fill, seed, name)` (pure Python).
- Graph helpers, also used by the GPU check: `read_images(..., shuffle_mode='epoch')`, `random_linear_params`, `constant_linear_params`, `build_linear_matrices -> (T, T_pull, Rt)`, `warp_linear`, `intensity_augment`.

`data_loading.data_module`:
- `enable_memory_growth()`, `read_manifest(path)`, `check_level_shape(rows, level)`.
- `manifest_files(rows, level, split='train', kind='image'|'labels', label_set='train18', contrast=None)` returns `<dataset>/<subject_dir>/<file>` paths, relative to the export root.
- `make_dali_dataset(pipeline, device_id=0)` builds a `DALIDataset` under `tf.device('/gpu:0')` with fixed shapes and dtypes.
- `LinearDataModule(export_root, level, batch_size, seed=0, split='train', shuffle=True, rigid_ranges=None, affine_ranges=None, intensity_aug=None, deform_residual_affine=False, num_threads=4, prefetch_queue_depth=1, device_id=0, contrast=None, manifest_path=None, shuffle_mode='epoch')`
  - methods: `.files`, `.template_dir`, `.pipeline(task)`, `.dataset(task)` (cached), `.iterator(task)`;
  - seeds: rigid seed+0, affine seed+1 (deformable will use seed+2).

`data_loading.postprocess` (pure TF, works in `tf.function`):
- `dali_to_mm(m_dali, level, pc)`, `sample_flip_axes(gen, flip_prob)`, `flip_matrix_dynamic(flags)`.
- `apply_flip(images, [mats], flags)`.
- `num_pairs(n, p_pair=0.1)`, `sample_pairs(gen, n, p_pair=0.1) -> (i, j)` int32.
- `LinearPostprocess(task, level, flip_prob=0.5, p_pair=0.1, seed=0, target_iters=10, target_dtype=tf.float64)`, called as `(images, T_dali[, Rt_dali], flip_axes=None)`. It returns a dict with:
  - `images`, `T_render`, `T_target`, `A_star`;
  - `pair_i`, `pair_j`, `P_star`, `flip_axes`.
- `brain_S_matrix(level, template_dir)`: loads `template_mask<suffix>.npy` and calls `transforms.brain_second_moment`.

## Matrix flow (exact formulas)

Notation: column vectors, rightmost factor applied first.

1. **DALI** (DALI axis order (W,H,D) = reversed array axes, pixel coordinates u = p + pc). Per image:
   - Build Tr (translation, voxels = mm / spacing), R (angle U(±45°), axis = normalized `fn.random.normal(shape=3)`) and K.
   - C± are translations by ±c_d, with c_d = (n − 1)/2 + pc in DALI order.
   - Forward transform: **T = C+ · Tr · R · K · C−**.
   - Rigid part: **Rt = C+ · Tr · R · C−**.
   - Pull matrix: **T⁻¹ = C+ · K⁻¹ · R⁻¹ · Tr⁻¹ · C−**, with R⁻¹ = rotation(−angle, same axis) and Tr⁻¹ = translation(−t). It is built factor by factor, since DALI 1.18 has no matrix inverse.
   - `fn.warp_affine(img, matrix=T⁻¹, inverse_map=True, fill 0, linear)` renders I_i(u) = I(T⁻¹u).
   - **Emitted: the forward T (and Rt), not the pull matrix.**
2. **K (scale/shear)**: K = U·D in TF axis order, with D = diag(e^{s_x}, e^{s_y}, e^{s_z}) and U = [[1,a,b],[0,1,c],[0,0,1]], where a = k_xy, b = k_xz, c = k_yz.
   - K⁻¹ = D⁻¹U⁻¹, with U⁻¹ = [[1,−a,ac−b],[0,1,−c],[0,0,1]].
   - In DALI order, K_dali[i][j] = K_tf[2−i][2−j] (`params.scale_shear_rows`). It is stacked into a 3×4 matrix with `fn.stack` and passed to `fn.transforms.combine`.
3. **To TF mm**: T_mm = W (P T_dali P) W⁻¹, where W maps u (TF order) to x = s (u − pc − (n−1)/2) (`models.transforms.dali_voxel_to_tf_mm`). The result satisfies I_i(y) = I(T_mm⁻¹ y). So `transform_image(I, T_mm⁻¹)` renders the DALI output, and `transform_image(I_i, T_mm)` undoes it.
4. **Flip** (one state per group, seeded `tf.random.Generator`): images get `tf.reverse` on the flipped axes, and T ← F T F, Rt ← F Rt F, with F = diag(±1, ±1, ±1, 1).
5. **Targets**:
   - A_i* = μ*⁻¹ T_tgt,i⁻¹ with μ* = barycenter{T_tgt,i⁻¹}, using 10 iterations in float64 and `stop_gradient` (`transforms.barycentric_targets`).
   - T_tgt = Rt for rigid and T for affine.
   - P_ij* = T_tgt,j T_tgt,i⁻¹ for the sampled pairs.
   - Pairs: `max(1, round(0.1 · N(N−1)))` distinct ordered pairs with i ≠ j, taken as the top-k of uniform keys.

**Nuisance composition order**: T_i = Rt_i · S_i with S_i = C+ K_i C−, both about the center.
- The image is rendered as I(S_i⁻¹ Rt_i⁻¹ y): the nuisance acts first in template space, then the rigid motion.
- The rigid target uses Rt_i only, so S_i is a well-defined residual that the network does not undo.
- In TF mm, Rt_mm⁻¹ T_mm = U·D exactly: upper triangular, with no translation. The local tests check this through the numpy mirror, and GPU check (d) checks it on the DALI-emitted matrices.

## Decisions within the spec

- **Shuffle.** Default `shuffle_mode='epoch'` (`shuffle_after_epoch=True`, full reshuffle every epoch, no host buffer; see iteration 2). With `shuffle_mode='buffer'` (`random_shuffle=True`), DALI's shuffle buffer holds decoded samples in host memory, so `initial_fill` defaults to about 1 GiB: 2,496 samples at level 4 (more than 2/3 of the train set) and 39 at level 1.
  - In both modes the file list is also permuted once in Python with the pipeline seed. This is deterministic.
  - `pad_last_batch=False`: the reader rolls into the next epoch, so a group never contains a padded duplicate. Batches are always exactly N.
- **No file identity stream.** Check (b) uses `shuffle=False`, which reads the files in order.
- **Residual-affine flag.** `deform_residual_affine` (default off) is stored on `LinearDataModule`, and its ranges are in `params`. The deformable pipeline itself is milestone 6.
- **Curriculum.** Not ported. The hook would be a numpy reader over a scale file, as in the reference `curriculum.py`, multiplying the range half-widths. No `external_source` is used.
- **Clamp.** Clamping to [0, 1] runs only when intensity aug is on. The linear warp of [0, 1] data already stays in [0, 1].
- **Spacing.** Assumed isotropic (true at every level), so voxel-space rotation equals mm-space rotation.
- **Target precision.** Targets are computed in float64 and returned as float32. Images stay float32 (the float16 cast happens later, at the model input).

## Local test results

Run with `tests/run_tests_wsl.sh` (full log in `test-log.txt`):
- `bgr_preproc38` (Python 3.8.20, numpy 1.21.1, scipy 1.4.1): py_compile and `ast` 3.8 pass on 16 files; the numpy references pass; `data_loading` imports without TF or DALI (`DALI_AVAILABLE = False`); `bash -n` passes.
- `tf_gpu` (Python 3.9.13, TF 2.10.0): **61 tests, 60 pass, 1 skipped, 0 fail** (iteration 2). That's 35 milestone-1 tests plus 26 new ones (iteration 2 added `test_reader_kwargs_shuffle_modes`). The skip is `test_pipeline_runs_on_synthetic_files`, which needs DALI and a GPU.

New coverage:
- **Ranges and levels:** level constants and spacing; mm→voxel translation per level (15 mm → 3.75/7.5/15 vox); defaults and overrides; `auto_initial_fill`; intensity defaults.
- **DALI formulas (numpy mirror + `models.transforms`):** K·K⁻¹ = I and P K P = U D; T·T⁻¹ = I. For every level and task: Rt_mm is orthonormal with translation equal to the reversed voxel offset × spacing, and Rt_mm⁻¹ T_mm equals U D with zero translation. A rotation about `dali_center(pc)` converts to zero mm translation with a matching pc; a mismatched pc gives more than 0.5 mm.
- **Reader kwargs:** `'epoch'` sets only `shuffle_after_epoch`, `'buffer'` sets `random_shuffle` + `initial_fill`, `shuffle=False` sets neither, an unknown mode raises.
- **Manifest:** split and level file lists, labels (`train18`), contrast filter, missing label/column errors, shape check, too-small split error, output signature, memory-growth helper, and the DALI import guard.
- **Flips:** for 4 flip states, the flipped render equals the render of the flipped template with (FTF)⁻¹, and undo with FTF equals the flipped undo (< 1e-5). Flip sampling covers probabilities 0, 1 and 0.5 (frequency), plus seed determinism.
- **Postprocess:** `dali_to_mm` matches models; rigid targets use Rt, are central (μ ≈ I to 1e-4), stay rigid, and are pair-consistent; the affine target is the full T; flip equivariance holds (A* → F A* F); the images follow the random flip state; one `tf.function` trace; same seed gives the same flips and pairs.
- **Pairs:** counts (N = 2/3/8/12/48 → 1/1/6/13/226), i ≠ j, uniqueness, determinism, all pairs at p = 1.
- **S:** synthetic level-4/2 masks match `brain_second_moment` and the brute-force loop; a wrong shape raises.

I also smoke-tested the TF half of GPU checks (b), (c) and (d) locally, with TF-rendered stand-ins for the DALI outputs. All passed: NCC 0.990 on the sharp synthetic phantom at 4 mm, flip deviation 3e-8, rigid centrality 5e-7 mm and pairs 1e-6 mm. The script was temporary and has been deleted.

## Glacier command

The orchestrator runs this on glacier after `git pull` in the clone and choosing a free GPU `<idx>` with `nvidia-smi`:

```bash
bash /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad/tests/gpu/run_verify_glacier.sh <idx> 1800
# container unittest run (Python 3.8.10 / DALI 1.18; the DALI test runs instead of skipping):
bash /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad/tests/gpu/run_verify_glacier.sh <idx> 1800 unittest
```

The script:
- sets `CUDA_VISIBLE_DEVICES=<idx>`, `TMPDIR`, `CUDA_CACHE_PATH` and `KERAS_HOME` under the scratch folder, and `PYTHONDONTWRITEBYTECODE=1` (also as `SINGULARITYENV_*`);
- closes stdin and runs under `timeout`;
- runs `singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast tf10_nv.sif python tests/gpu/verify_dali_pipeline.py --data .../data/npy_v2 --scratch .../scratch/m2 --out .../scratch/m2/summary.json` from the clone;
- writes only `.../GroupwiseReg/scratch/m2` (`summary.json`, `verify.log` or `unittest.log`, the CUDA cache, `keras/`, and temp folders that are removed at the end).

The exit code is non-zero if any check fails (124 means timeout).

## Glacier-only (unverified locally)

- **All DALI 1.18 API usage:**
  - `fn.readers.numpy` with `file_root`;
  - `fn.constant` and subscripts;
  - `fn.transforms.rotation` with a tensor axis;
  - `fn.transforms.combine` of `fn.stack`-built 3×4 matrices;
  - `warp_affine`;
  - the ported intensity ops.
- **Check (a), the pixel center and direction.** If DALI's real convention differs, the check reports the empirical (pc, direction, axes) candidate and FAILs. The fix is then `params.DALI_PIXEL_CENTER`, which drives both the centering and the conversion. If the axis order or direction differs, the fix is in the conversion.
- **(b) NCC > 0.99 on real 4 mm data.** It is measured inside the brain mask, excluding voxels that left the FOV. Double linear interpolation alone could land near 0.99 at 4 mm, so the summary also reports the TF-only double-interpolation NCC, the error ratio, and the single-interpolation |DALI − TF| difference. This separates a convention bug from interpolation loss.
- **(c)–(f):** flip consistency, target sanity and the nuisance structure on DALI matrices, throughput and memory at N = 48 (rigid, affine, concurrent), and determinism. Plus **(g)**, informational: float16 input to `warp_affine`.
- **`shuffle_after_epoch=True` on `fn.readers.numpy`** (documented for 1.18, not executed locally), plus the magnitude bounds in (b).
- **The container unittest run** (Python 3.8.10 / numpy 1.21.1 / scipy 1.4.1; `run_verify_glacier.sh <idx> 1800 unittest`). There, `test_pipeline_runs_on_synthetic_files` will run both shuffle modes instead of skipping.
