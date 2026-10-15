# Implementation Plan: brain MRI preprocessing pipeline (`preprocessing/`)

Goal: a manifest-driven pipeline. Its outputs are skull-stripped, in LUMIR space (template grid; LUMIR is 160x224x192 at 1 mm), normalized to [0,1], and have labels. It runs on the MGH/Martinos SLURM cluster (Linux, FreeSurfer >= 7.3, ANTs). Local verification uses WSL Ubuntu only, with fake tool executables. Do NOT git commit or push.

## 0. Ground truth from exploration

- Repo state (clean `main`, 4 commits): `.gitignore`, `.gitattributes`, `scripts/{demo_central_transform.py, check_tf_group_centering.py, run_in_tf_env.sh}`, `.agents/tasks/dataset-survey-2026-10-12/report.md`. There is no Python packaging, no tests, and no existing preprocessing code.
- Conventions to copy from `scripts/`: a module docstring at the top with a `Run:` usage line. Bash runner scripts are invoked as `wsl -d Ubuntu bash /mnt/d/.../script.sh` and start with `source ~/miniconda3/etc/profile.d/conda.sh; conda activate <env>`. The comment style is short and plain.
- `.gitignore` constraints:
  - It ignores `*.nii`, `*.nii.gz`, `*.npy`, `*.npz`, `logs/`, `outputs/`, `results/`, `build/`, `data/` and **`datasets/`**. Never name a package folder `datasets` (the builders package is `builders/`).
  - Tests must generate every NIfTI at runtime in `tmp_path`.
- `.gitattributes` forces LF only for `*.sh *.py *.yml *.yaml`. All SLURM templates therefore use the `.sh` extension, so no `.gitattributes` change is needed.
- WSL conda is at `~/miniconda3` (user `jbpatel`). Existing envs are `base`, `pt_gpu`, `tf_gpu` and `transformers`; `bgr_preproc` does not exist yet. FreeSurfer and ANTs are not installed locally.
- Dataset facts (survey report, section 2A/2B):
  - Neurite-OASIS per-subject dirs contain `orig/norm/seg35/seg4` (256^3, scanner space) plus `aligned_*`, which must not be used.
  - `norm` is FreeSurfer-normalized (WM about 110) and then scaled to [0,1].
  - UltraCortex ds005216 is BIDS, CC0, 9.4T, 0.6/0.7/0.8 mm. It has SynthStrip masks and skull-stripped images in derivatives, and manual GM/WM labels for 12 subjects. Exact derivative filenames are UNVERIFIED.
  - LUMIR is 160x224x192, [0,255], skull-stripped, affine-aligned and N4-corrected. The LUMIR25 val 10-class label naming is UNVERIFIED.

## 1. Design decisions and flagged deviations

The design in the step prompt is followed. Refinements and deviations are listed below. Items marked **FLAG** deviate from the prompt or resolve an ambiguity in it.

1. **FLAG: training on LUMIR conflicts with the survey.** The survey says not to train on LUMIR because of IXI/MPI-Leipzig leakage. The user explicitly accepted that (message 5), so this plan follows the user. The manifest gets a `split` column (train/val/test/"") so leakage-aware splits remain possible later.
2. **FLAG: `manifest_processed.csv` is written by `qc-summary`, not by `run`.** SLURM array tasks run concurrently, and appending to one CSV would race. Each task writes only its own subject dir (`meta.json`, `qc.json`). `qc-summary` aggregates them into `<out_root>/<dataset>/manifest_processed.csv` and `qc_summary.csv`.
3. **FLAG: contrast suffix is always included in the subject dir name** when `contrast` is non-empty. Builders always set it (e.g. `T1w`). Dir name: `<subject>[_ses-<session>]_<contrast>`, e.g. `OASIS_OAS1_0001_MR1_T1w`. This keeps names stable once OASIS-3 T2/FLAIR arrive.
4. **FLAG: LUMIR grid mismatch.**
   - Shape and affine match the template (max abs affine diff <= `grid_tolerance_mm`, default 1e-3): no resampling.
   - Otherwise: warn, set `grid_mismatch=true` in meta/QC, and resample onto the template grid with `antsApplyTransforms` and **no `-t` (identity)**, which is correct in physical space.
   - Flag-only is impossible when the shape differs. Silently stamping the template affine would be wrong.
5. **FLAG: `env_setup` handling.** The config's `env_setup` lines (e.g. `module load freesurfer/7.4.1`) are executed once with `subprocess.run(["bash", "-c", "\n".join(lines) + "\nenv -0"])`. The resulting environment dict is reused for every tool call.
   - Paths are never interpolated into a shell string. Only the user-authored config lines go through bash.
   - Empty `env_setup` means the current environment is used.
   - The SLURM scripts also have an env placeholder block. The user can choose either one; this is documented in both places.
6. **FLAG: requirements are split.** `requirements.txt` holds runtime pins and `requirements-test.txt` holds `-r requirements.txt` plus `pytest`, so the cluster env doesn't need pytest.
7. **FLAG: extra small subcommand `n-chunks`** (`--manifest M --chunk-size C`). It prints the number of array tasks so the user can set `#SBATCH --array=0-<N-1>`.
8. **FLAG: extra option `build-manifest --exclude <txt of subject IDs>`**, for removing Mindboggle OASIS-TRT/MICCAI-overlap IDs from Neurite-OASIS (survey section 2A leakage note). Cheap, and it doesn't change the design.
9. **FLAG: SynthStrip thread and GPU flags.** `mri_synthstrip` thread/GPU flags differ between FS versions (UNVERIFIED for 7.3 vs 7.4). The defaults pass only `-i -o -m`. `synthstrip.extra_args` in config (e.g. `["-t","8"]` or `["-g"]`) is documented as "verify with `mri_synthstrip --help` on the cluster".
10. **Normalization uses the upper percentile only** (no lower shift), as decided. An optional `lower_percentile` defaults to `null`. LUMIR gets the same per-image p99.5 normalization; it is not divided by 255.
11. **Anti-aliasing.**
    - Gaussian smoothing is done in numpy/scipy on the native (stripped, N4) image before `antsApplyTransforms`.
    - Per axis: `sigma_mm = sqrt(max(t^2 - s^2, 0)) / (2*sqrt(2*ln 2))`, with target spacing `t = 1.0` mm and input spacing `s` (FWHM-matching). `sigma_vox = sigma_mm / s`.
    - Example: 0.6 mm gives sigma 0.340 mm (0.566 vox); 0.8 mm gives 0.255 mm.
    - Smoothing applies only to axes with `s < t * resample.min_spacing_ratio` (default 0.95), and only to the intensity image. Mask and labels are never smoothed.
    - Registration uses the unsmoothed image. Scale changes inside the affine are ignored when choosing sigma (documented).
12. **qform/sform.** The validate step warns when qform and sform disagree (ITK and nibabel may then pick different affines). Every NIfTI the code writes sets both qform and sform to the same affine (code 1 for scanner, 2 for aligned/template).
    - Final outputs are written with the template's exact affine/header, after checking that the ANTs output grid matches within tolerance (otherwise the row errors).
13. **Template mask.**
    - `freq = mean_i(mask_i)`; `template_mask = largest_component(fill_holes(freq >= 0.5))`.
    - `template = mean_i(norm_i)`, set to 0 outside `template_mask` and saved as float32.
    - `template_info.json` records the seed, the selected files and any header mismatches.
14. **Per-step resumability.** Each step declares its outputs. A step is skipped when all its outputs exist, unless `--overwrite` is set. A whole row is skipped fast when `meta.json` has `status: "completed"` and the requested steps are all done.
    - Intermediates live in `<subject_dir>/work/`. `run --clean-work` deletes `work/` after a row succeeds; the default keeps it.
15. **QC "over the template" reading.** `qc.png` is a 2x3 grid:
    - Row 1: the subject image mid-slices, with the subject mask contour (red) and SynthSeg label boundaries (if present).
    - Row 2: the template mid-slices, with the subject mask contour. This row checks registration.
    - Registration similarity is cheap and numpy-only: NCC between the final image and the template inside the template mask, plus Dice(subject mask, template mask).
16. **SynthSeg.**
    - It is not in `run`'s default steps; `run --steps ...,synthseg` runs it per row. The main path is the separate `synthseg` batch command.
    - Batch mode writes `inputs.txt`, `outputs.txt`, `vols.txt` and `qc.txt` and makes one `mri_synthseg` call.
    - `--vol` and `--qc` in txt mode take a txt of per-subject CSV paths. This matches the SynthSeg README for txt inputs; verify on the cluster.
17. **Neurite-OASIS mask** = `fill_holes(largest_component((norm > 0) | (seg35 > 0)))`. `stripped = norm * mask`. SynthStrip and N4 are skipped (already FreeSurfer-processed). The intensity-distribution concern (WM-peak shape) is not solved by preprocessing; per-image p99.5 only fixes the scale.
18. **UltraCortex.** Provided SynthStrip masks are used when found; `--ignore-provided-masks` forces our own SynthStrip. N4 is always run (`bias_corrected=false`).
19. **Row failure policy.** Each row's exception is caught and logged to `<subject_dir>/logs/pipeline.log`, and `meta.json` gets `status: "failed"` and `error`. Processing continues; the exit code is 1 if any row failed. `--fail-fast` stops at the first failure.
20. **Image orientation.** No reorientation is ever done. Everything relies on header affines, and ANTs works in physical space. Exported `.npy` arrays keep nibabel array order (x, y, z) = 160x224x192, C-contiguous.

## 2. File layout (all under `preprocessing/`)

| File | Purpose |
|---|---|
| `__init__.py` | Package docstring: pipeline overview, output layout, end-to-end usage examples; `__version__ = "0.1.0"`. |
| `__main__.py` | `from .cli import main; raise SystemExit(main())`. |
| `cli.py` | argparse entry point with all subcommands (section 3); the module docstring holds usage examples. |
| `config.py` | Loads YAML, deep-merges onto `DEFAULT_CONFIG`, validates types/required keys, applies CLI overrides (`--threads`, `--synthseg-cpu`). |
| `config_example.yaml` | Example config with `# PLACEHOLDER` markers (section 5). |
| `nifti.py` | `load`, `save_like(data, affine, path, dtype, xform_code)` (sets qform=sform), `voxel_sizes`, `affines_match(a, b, tol)`, `grid_matches(img, ref, tol)`, `qform_sform_consistent`. |
| `masks.py` | `derive_mask(data, threshold=0.0, fill_holes=True, largest_component=True, extra=None)`; `largest_component`; `dice(a, b)`. |
| `intensity.py` | `normalize_intensity` (section 4); `antialias_sigma_mm(spacing, target)`; `antialias_smooth(data, spacing, target, ratio)`. |
| `labels.py` | `resample_labels_nearest(src_data, src_affine, ref_shape, ref_affine)` (vox2vox via `scipy.ndimage.map_coordinates(order=0, mode="constant", cval=0)`); `to_label_dtype(arr, dtype)` (rounds; errors on overflow). |
| `manifest.py` | `ManifestRow` dataclass; `read_manifest`/`write_manifest` (csv module); `parse_label_paths`/`format_label_paths`; `parse_bool`; `check_unique`; `subject_dir_name(row)`. |
| `builders/__init__.py` | `BUILDERS = {"lumir":..., "neurite_oasis":..., "ultracortex":..., "generic":...}`; common `apply_exclude`. |
| `builders/bids.py` | BIDS filename entity parser (`sub`, `ses`, `acq`, `run`, `desc`, suffix) plus a derivative-matching helper. |
| `builders/lumir.py` | Scans `root/**/LUMIRMRI_*_0000.nii.gz` (sorted). `subject=LUMIRMRI_XXXX`; split from the parent dir (`imagesTr`→train, `imagesVal`→val, `imagesTs`→test, else ""); optional `--label-glob` (`{id}` placeholder) → label `lumir10`. Flags: stripped/bias/in_template = true. |
| `builders/neurite_oasis.py` | Dirs under root containing `norm.nii.gz`. `subject` = dir name; labels `seg35`, `seg4` when present; `--image {norm,orig}` (orig → full pipeline: stripped/bias=false). Ignores `aligned_*`/`slice_*`. |
| `builders/ultracortex.py` | BIDS scan of `sub-*/[ses-*/]anat/*_T1w.nii.gz`, excluding `derivatives/`. Matches masks in `derivatives/**` by sub/ses (and acq/run) entities plus keyword `mask`, and manual labels by keyword `seg`/`dseg`/`manual` → label `manual`. Logs counts of matched/unmatched. `--ignore-provided-masks`. |
| `builders/generic.py` | Two modes. BIDS mode (if `dataset_description.json` exists): `--suffix T1w` (default). Folder mode: `--image-glob`, `--subject-from {parent,stem}`, `--mask-glob`, repeatable `--label NAME=GLOB` (`{subject}` placeholder), plus constant flags `--skull-stripped --bias-corrected --in-template-space`, `--name <dataset>`, `--contrast`. |
| `tools.py` | `resolve_environment(env_setup)`; `ToolRunner(env, threads, dry_run, echo)` with `.run(argv, log_path)`; `get_tool_versions(cfg)`; `check_tools(cfg) -> (ok, report)`. |
| `commands.py` | Pure argv builders: `synthstrip_cmd`, `n4_cmd`, `ants_registration_cmd`, `ants_apply_cmd`, `synthseg_cmd` (section 6). |
| `meta.py` | `load_meta`, `write_meta_atomic` (tmp + `os.replace`), `record_step(meta, name, outputs, command, params, start, end)`, `step_done(meta, outputs)`. |
| `template.py` | `check_headers(paths, tol)`; `build_template(lumir_root, out_dir, n=200, seed=0, cfg)` (section 7). |
| `pipeline.py` | `process_row(row, cfg, out_root, steps, runner, overwrite, clean_work)`; `run_rows(...)`; constants `ALL_STEPS`, `DEFAULT_STEPS`. |
| `synthseg.py` | `run_synthseg_batch(subject_dirs, cfg, runner, batch_dir, overwrite)`; `postprocess_synthseg(subject_dir, raw_seg, template)` (grid check, nearest fallback, uint16). |
| `qc.py` | `compute_qc(subject_dir, template_img, template_mask, cfg) -> dict`; `write_qc_png`; `robust_z`; `summarize(dataset_dir, mad_threshold)` → `qc_summary.csv` + `manifest_processed.csv`. |
| `export.py` | `export_npy(out_root, export_root, datasets, image_dtype, label_dtype, include_template)` → parallel `.npy` layout + `export_manifest.csv`. |
| `requirements.txt` | `numpy==1.26.4`, `scipy==1.11.4`, `nibabel==5.2.1`, `PyYAML==6.0.2`, `matplotlib==3.9.2` (all Python 3.9 compatible). |
| `requirements-test.txt` | `-r requirements.txt` plus `pytest==8.3.3`. |
| `slurm/run_array.sh` | CPU array job: chunk `[start,end)` of the manifest through strip/N4/register/resample/normalize/qc. |
| `slurm/synthseg_batch.sh` | SynthSeg batch over chunks (GPU default; CPU variant documented via commented `#SBATCH` lines plus `--synthseg-cpu`). |
| `slurm/qc_export.sh` | `qc-summary` for a dataset, then optional `export-npy`. |
| `slurm/build_template.sh` | One CPU job running `build-template` (reads about 200 LUMIR volumes). |
| `tests/__init__.py` | Empty (so `tests/fake_tools.py` is importable from conftest). |
| `tests/conftest.py` | Fixtures: `write_nifti`, `phantom` (ellipsoid brain + skull shell + smooth bias), `small_template` (20x24x22, 1 mm, LUMIR-like affine), `fake_tools` (installs fakes, sets PATH, `FAKE_TOOL_LOG`, `FREESURFER_HOME`), `config_file`. |
| `tests/fake_tools.py` | Generates fake executables in `tmp_path/bin` (section 8). |
| `tests/run_tests_wsl.sh` | Creates/updates the `bgr_preproc` env, runs `bash -n` on all `.sh`, pytest, and CLI smoke (section 9). |
| `tests/test_intensity.py` | Normalization formula, clipping fraction, outside-mask zeros, dtype, errors; antialias sigma values and axis selection. |
| `tests/test_masks.py` | `derive_mask` threshold/fill/largest-component/union with labels; `dice` edge cases (both empty → 1.0). |
| `tests/test_nifti.py` | `affines_match` tolerance, `grid_matches` shape/affine, `save_like` qform==sform, `voxel_sizes`, qform/sform mismatch detection. |
| `tests/test_labels.py` | Nearest fallback: crop/shift a known label grid; values preserved, no new labels, correct world alignment; uint16 cast; overflow error. |
| `tests/test_config.py` | Defaults merge, example config loads, invalid type error, CLI override. |
| `tests/test_manifest.py` | Round-trip; `label_paths` with paths containing `:`; `;` in a path rejected; bool parsing; duplicate keys error; `subject_dir_name`. |
| `tests/test_builders.py` | Fake trees for LUMIR (imagesTr/imagesVal plus a labels dir), Neurite-OASIS (with `aligned_*` decoys), UltraCortex BIDS (sessions, derivatives masks, manual segs for some subjects), generic BIDS and folder (Mindboggle-like); `--exclude`. |
| `tests/test_commands.py` | Exact argv lists for every tool from a fixed config (section 6). |
| `tests/test_tools.py` | `resolve_environment` with `export FOO=bar` lines; runner captures stdout/stderr to the log; nonzero exit raises `ToolError` including the log path; dry-run prints `shlex.join(argv)` and executes nothing; ITK threads env set; version parsing (`build-stamp.txt`, `ANTs Version: 2.5.0`); `check_tools` missing → not ok. |
| `tests/test_template.py` | Five synthetic LUMIR-like vols: mean equals the expected numpy mean; seed reproducible; header-mismatch volume reported and excluded; freq/mask threshold. |
| `tests/test_qc.py` | QC dict keys/values on known arrays; `robust_z` and outlier flags (MAD=0 → no flag); PNG written (non-empty file); `summarize` CSV contents. |
| `tests/test_pipeline.py` | End-to-end `run` with fake tools for LUMIR, Neurite-OASIS, UltraCortex (0.7 mm), generic-raw; dry-run; resumability; overwrite; failure handling. |
| `tests/test_synthseg.py` | Batch txt contents; single call for N subjects; fallback resample when the fake crops; meta updated; resume skip. |
| `tests/test_export.py` | NPY round trip (values, dtypes, shapes), float16 option, uint8 overflow error, template export. |
| `tests/test_cli.py` | `--help` for every subcommand; `check-tools` exit 0 with fakes and nonzero without; `build-manifest` via CLI; `n-chunks`. |
| `tests/test_slurm_scripts.py` | Every `.sh` in `preprocessing/`: first line `#!/bin/bash`, no `\r`, `bash -n` passes, contains no `~/.bashrc`/`.cshrc` sourcing; `#SBATCH` placeholders present. |

## 3. CLI (`python -m preprocessing <command>`; run from the repo root or with `PYTHONPATH=<repo>`)

Global options: `-v/--verbose` and `--log-level`.

- `check-tools --config C`
  - Resolves env_setup and checks every tool via `shutil.which` (in the resolved env PATH).
  - Prints versions: FreeSurfer from `$FREESURFER_HOME/build-stamp.txt` (regex `(\d+)\.(\d+)\.(\d+)`), requiring >= 7.3; ANTs from `antsRegistration --version` (regex `ANTs Version: v?(\d+)\.(\d+)`), warning if < 2.3 (`--random-seed`).
  - Checks that the template files from the config exist and that `FREESURFER_HOME` is set.
  - Exit 0 when all are OK, 1 otherwise, with a clear report table.
- `build-manifest --dataset {lumir,neurite_oasis,ultracortex,generic} --root R --out M.csv [--exclude ids.txt] [builder-specific opts]`. Writes the manifest and logs the row count.
- `build-template --lumir-root R --out-dir D [--n 200] [--seed 0] [--config C] [--check-all-headers/--no-check-all-headers]`. Writes `template.nii.gz`, `template_mask.nii.gz`, `template_info.json` and prints the config snippet.
- `run --config C --manifest M --out-root O [--start S --end E | --rows 3,5,9] [--steps strip,n4,register,resample,normalize,qc] [--dry-run] [--overwrite] [--threads N] [--clean-work] [--fail-fast]`
  - Validation always runs.
  - `ALL_STEPS = strip, n4, register, resample, normalize, synthseg, qc`; `DEFAULT_STEPS` = all except `synthseg`.
- `synthseg --config C --manifest M --out-root O [--start/--end|--rows] [--batch-dir D] [--overwrite] [--threads N] [--synthseg-cpu] [--dry-run]`
  - Selects rows whose `image.nii.gz` exists and whose `labels_synthseg.nii.gz` is missing (unless `--overwrite`).
  - Makes one `mri_synthseg` call, then postprocesses, updates `meta.json` and recomputes `qc.json`.
- `qc-summary --out-root O --dataset NAME [--mad-threshold 4.0]`. Writes `qc_summary.csv` and `manifest_processed.csv`; exit 0.
- `export-npy --out-root O --export-root X [--dataset NAME ...] [--image-dtype float32|float16] [--label-dtype uint16|uint8] [--include-template --config C] [--overwrite]`.
- `n-chunks --manifest M --chunk-size C`. Prints `ceil(rows/C)`.

## 4. Exact numeric definitions

Normalization (`intensity.normalize_intensity(x, mask, upper_percentile=99.5, lower_percentile=None)`):

```
M = mask > 0 ;  v = x[M].astype(float64)            # error if M.sum() == 0
hi = np.percentile(v, upper_percentile)              # numpy default 'linear' method
lo = 0.0 if lower_percentile is None else np.percentile(v, lower_percentile)
if hi <= lo: raise ValueError
y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
y[~M] = 0.0 ; y = y.astype(float32)
stats = {"p_hi": hi, "p_lo": lo, "frac_clipped_high": mean(v > hi), "frac_clipped_low": mean(v < lo)}
```

With the default `lower_percentile=None` this is exactly `clip(x / p99.5_in_mask, 0, 1) * M`.

Other definitions:
- Mask resampled with GenericLabel → `(arr > 0.5).astype(uint8)`. Labels → `np.rint(arr).astype(uint16)`.
- `mask_volume_cc = mask.sum() * prod(voxel_sizes) / 1000`.
- `dice(a,b) = 2|a∩b| / (|a|+|b|)` (1.0 if both empty).
- `ncc_to_template` = Pearson r of image vs template over template-mask voxels.
- `robust_z = (x - median) / (1.4826 * MAD)`. MAD = 0 → z = 0.
- An outlier is `|z| > mad_threshold` (default 4) for `mask_volume_cc, p50, p99, ncc_to_template, dice_mask_template, dice_mask_synthseg`. Hard flags: `grid_mismatch`, `affine_match_final == False`, `n_labels_synthseg < 1` when labels exist, `status != completed`.

## 5. Config schema (`config_example.yaml`, also `DEFAULT_CONFIG` in `config.py`)

```yaml
env_setup: []            # PLACEHOLDER e.g. ["module load freesurfer/7.4.1", "module load ants/2.5.0"]
                         # or ["export FREESURFER_HOME=/PLACEHOLDER/freesurfer", "source $FREESURFER_HOME/SetUpFreeSurfer.sh"]
tools:
  mri_synthstrip: mri_synthstrip
  mri_synthseg: mri_synthseg
  antsRegistration: antsRegistration
  antsApplyTransforms: antsApplyTransforms
  N4BiasFieldCorrection: N4BiasFieldCorrection
threads: 4               # overridden by --threads (SLURM passes $SLURM_CPUS_PER_TASK)
grid_tolerance_mm: 0.001
lumir_expected_shape: [160, 224, 192]
template:
  image: /PLACEHOLDER/templates/lumir_template.nii.gz
  mask: /PLACEHOLDER/templates/lumir_template_mask.nii.gz
mask:        {threshold: 0.0, fill_holes: true, largest_component: true}
normalization: {upper_percentile: 99.5, lower_percentile: null}
synthstrip:  {extra_args: []}          # verify flags with `mri_synthstrip --help`
n4:          {shrink_factor: 4, bspline_fitting: "[200]", convergence: "[50x50x50x50,0.0]"}
registration:
  gradient_step: 0.1
  metric_bins: 32
  sampling_strategy: Regular
  sampling_percentage: 0.25
  convergence: "[1000x500x250,1e-6,10]"
  shrink_factors: 4x2x1
  smoothing_sigmas: 2x1x0vox
  winsorize: "[0.005,0.995]"
  random_seed: 42                      # null → omit flag
  use_masks: false                     # true → add -x [template_mask,moving_mask]
resample:    {antialias: true, target_spacing_mm: 1.0, min_spacing_ratio: 0.95}
synthseg:    {parc: true, robust: true, cpu: false, threads: 8, vol: true, qc: true, extra_args: []}
qc:          {png: true, mad_threshold: 4.0}
```

## 6. Exact external command argument lists (`commands.py`)

`T` is the resolved tool path. Paths are absolute strings. Lists are passed to `subprocess.run(argv, env=env, stdout=log, stderr=STDOUT, check=False)`, and a nonzero return code raises `ToolError`.

- **SynthStrip:** `[T, "-i", in, "-o", stripped, "-m", mask] + synthstrip.extra_args`
- **N4:**
  ```
  [T, "-d", "3", "-i", stripped, "-x", mask, "-o", n4_out, "-s", str(shrink),
   "-b", bspline_fitting, "-c", convergence, "-v", "0"]
  ```
- **antsRegistration** (fixed = template image, moving = stripped/N4 native image):
  ```
  [T, "--dimensionality", "3", "--float", "1", "--collapse-output-transforms", "1",
   "--output", f"{work}/reg_", "--interpolation", "Linear",
   "--winsorize-image-intensities", winsorize, "--use-histogram-matching", "0",
   "--initial-moving-transform", f"[{fixed},{moving},1]",
   "--transform", "Rigid[0.1]", "--metric", f"MI[{fixed},{moving},1,32,Regular,0.25]",
   "--convergence", conv, "--shrink-factors", "4x2x1", "--smoothing-sigmas", "2x1x0vox",
   "--transform", "Affine[0.1]", "--metric", f"MI[{fixed},{moving},1,32,Regular,0.25]",
   "--convergence", conv, "--shrink-factors", "4x2x1", "--smoothing-sigmas", "2x1x0vox",
   ("--random-seed", "42" if set), ("-x", f"[{tmask},{mmask}]" if use_masks), "--verbose", "0"]
  ```
  - Output: `work/reg_0GenericAffine.mat`, copied to `affine_to_template.mat`. `--collapse-output-transforms 1` folds the COM initializer into that single matrix.
  - Env adds `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=<threads>` (also for N4 and antsApplyTransforms) and `OMP_NUM_THREADS=<threads>`.
- **antsApplyTransforms:**
  ```
  [T, "-d", "3", "-i", moving, "-r", template_image, "-o", out, "-n", interp, "--float", "1", "-v", "0"]
  + (["-t", affine_mat] if affine_mat else [])
  ```
  `interp` is `Linear` for the image and `GenericLabel` for masks and label maps. No `-t` means identity (LUMIR grid fix).
- **SynthSeg**, where `I/O/V/Q` are single files (single mode) or `.txt` lists (batch):
  ```
  [T, "--i", I, "--o", O] + (["--parc"] if parc) + (["--robust"] if robust)
  + (["--vol", V] if vol) + (["--qc", Q] if qc) + ["--threads", str(threads)] + (["--cpu"] if cpu) + extra_args
  ```
  - Batch dir: `<out_root>/<dataset>/_synthseg_batches/<start>_<end>/` with `inputs.txt`, `outputs.txt`, `vols.txt`, `qc.txt`.
  - One line per subject; outputs go to `<subject_dir>/work/synthseg_raw.nii.gz`, `<subject_dir>/synthseg_vols.csv` and `<subject_dir>/synthseg_qc.csv`.

## 7. Per-row pipeline (`pipeline.process_row`)

Subject dir `D = <out_root>/<dataset>/<subject_dir_name>`, with `D/work/`, `D/logs/<step>.log` and `D/logs/pipeline.log` (a per-row FileHandler).

1. **validate.**
   - Load image headers and check 3D (squeeze a trailing 4th dim of size 1, otherwise error). Record shape, spacing and the qform/sform check.
   - Check that the mask and labels exist. For `in_template_space`: check the grid against the template (sets `grid_mismatch`) and that the shape equals `lumir_expected_shape` (dataset == lumir only; warn).
2. **strip** produces `work/stripped.nii.gz` and `work/native_mask.nii.gz`:
   - `mask_path` given: mask = provided (>0) and stripped = image*mask.
   - Else if `skull_stripped`: mask = `derive_mask(image, extra=first label map if any)` and stripped = image*mask.
   - Else: run SynthStrip.
3. **n4** (skipped if `bias_corrected`): N4 on the stripped image within the native mask, producing `work/n4.nii.gz`. The input for later steps is `n4` or `stripped`.
4. **register** (skipped if `in_template_space`): antsRegistration → `affine_to_template.mat`.
5. **resample:**
   - The antialias check on spacing may write `work/smoothed.nii.gz`.
   - Not in template space: apply transforms → `work/image_resampled.nii.gz` (Linear), `work/mask_resampled.nii.gz` (GenericLabel), `work/labels_<name>_resampled.nii.gz` (GenericLabel).
   - In template space and grid matches: the image/mask/labels are used as is.
   - In template space with a mismatch: identity antsApplyTransforms.
6. **normalize:** load the resampled image and mask; assert `grid_matches(template)`; normalize.
   - Write `image.nii.gz` (float32) and `mask.nii.gz` (uint8) with the template affine.
   - Write `labels_<name>.nii.gz` as uint16 (nearest fallback via `labels.resample_labels_nearest` if a provided template-space label grid mismatches).
7. **synthseg** (optional per-row): single-subject `run_synthseg_batch`.
8. **qc:** `qc.json` plus `qc.png`.

`meta.json` contains: row (all manifest fields), `steps` {name: {status, outputs, command(s), params, started, finished, log}}, `tool_versions`, `config_snapshot`, `template` paths with sha256 of the template image, `python`/package versions, `status` and `error`.

Dry-run mode:
- Reads headers only (needed for antialias and grid decisions) and creates no files or directories.
- Prints each external command via `shlex.join(argv)`, prefixed `CMD`, and each Python step as `PY <step>: <inputs> -> <outputs>`.
- Steps whose inputs would be produced earlier are printed with their planned paths.

## 8. Fake-executable testing strategy (`tests/fake_tools.py`)

- `install_fake_tools(bin_dir, python=sys.executable)` writes executable scripts with a `#!{sys.executable}` shebang into `tmp_path/bin`. This is on the Linux FS, so chmod 0o755 works; not on `/mnt/d`. They are named `mri_synthstrip`, `mri_synthseg`, `antsRegistration`, `antsApplyTransforms` and `N4BiasFieldCorrection`.
- The fixture prepends `bin_dir` to PATH and creates `tmp_path/fs/build-stamp.txt` containing `freesurfer-linux-ubuntu22_x86_64-7.4.1-20230614-7eb8460`, with `FREESURFER_HOME` pointing at it.
- Every fake appends `{"tool": name, "argv": sys.argv[1:]}` as a JSON line to `$FAKE_TOOL_LOG`. Tests assert exact argv and call counts; resumability tests assert zero new calls.
- `FAKE_FAIL=<toolname>` makes that fake print `fake failure` to stderr and exit 3.

Behaviours, using nibabel/numpy/scipy:

| Fake | Behaviour |
|---|---|
| `mri_synthstrip` | Parses `-i -o -m`. mask = `largest_component(data >= 0.6 * data.max())` (the phantom brain is brighter than the skull shell). Writes stripped and mask on the input grid. |
| `N4BiasFieldCorrection` | Parses `-i -x -o` (supports `-o [a,b]`) and copies input → output. |
| `antsRegistration` | `--version` prints `ANTs Version: 2.5.0`. Otherwise parses `--output` (prefix or `[prefix,...]`) and writes `<prefix>0GenericAffine.mat` containing the text `FAKE_IDENTITY`. |
| `antsApplyTransforms` | Parses `-i -r -o -n [-t]`. Ignores the transform (identity) and resamples moving→reference **in world space**: `vox_m = inv(A_m) @ A_r @ vox_r`, `map_coordinates(order=1 if Linear else 0)`. Writes with the reference affine. This genuinely exercises the 0.7 mm / 256^3 → template-grid geometry. |
| `mri_synthseg` | Parses `--i --o [--vol] [--qc]`; txt or single file. Labels = `np.digitize(data, [0.05, 0.4, 0.7]) * [0,24,3,2]` mapping (0 outside); writes the vol CSV/QC CSV. `FAKE_SYNTHSEG_CROP=1` crops 2 voxels per side and shifts the affine origin accordingly, so the fallback is exercised. |

The synthetic test grid is small so tests stay fast (< 1 min total):
- Template: 20x24x22 at 1 mm, affine = LUMIR-like (diag 1 with origin offset).
- Neurite-like input: 32^3 at 1 mm, centred at the same world point.
- UltraCortex-like input: 0.7 mm, 30^3.
- Nothing hardcodes 160x224x192 except `lumir_expected_shape` (overridden in tests).

## 9. Local WSL test procedure

`preprocessing/tests/run_tests_wsl.sh` (bash, LF, `#!/bin/bash`):

```
#!/bin/bash
# Usage (PowerShell): wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/preprocessing/tests/run_tests_wsl.sh [pytest args]
set -eo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
source ~/miniconda3/etc/profile.d/conda.sh
conda env list | grep -q '^bgr_preproc ' || conda create -y -q -n bgr_preproc python=3.9
conda activate bgr_preproc
pip install -q -r "$REPO/preprocessing/requirements-test.txt"
set -u
for f in $(find "$REPO/preprocessing" -name '*.sh'); do bash -n "$f"; done
cd "$REPO"
export MPLBACKEND=Agg
python -m pytest preprocessing/tests -q -p no:cacheprovider "$@"
python -m preprocessing --help >/dev/null
for c in check-tools build-manifest build-template run synthseg qc-summary export-npy n-chunks; do python -m preprocessing "$c" --help >/dev/null; done
echo "ALL CHECKS PASSED"
```

- `-p no:cacheprovider` avoids writing `.pytest_cache` into the repo on `/mnt/d`.
- Pytest `tmp_path` lives in Linux `/tmp`.
- To run one module: `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/preprocessing/tests/run_tests_wsl.sh preprocessing/tests/test_intensity.py`. Extra args go to pytest; the CLI smoke still runs.

**Verify** below always means running this script and seeing all selected tests pass plus `ALL CHECKS PASSED`.

## 10. SLURM templates (`preprocessing/slurm/*.sh`)

All templates share these rules:
- They start with `#!/bin/bash`.
- They hold `#SBATCH` placeholders, marked `# EDIT:`, for `--partition`, `--account`, `--qos`, `--time`, `--mem`, `--cpus-per-task`, `--gres` (synthseg only), `--array` and `--output=logs/%x_%A_%a.out`.
- They contain `set -euo pipefail` and an `=== EDIT ME ===` block:
  ```
  REPO_DIR=${REPO_DIR:-/PLACEHOLDER/brain-group-rad}
  CONFIG / MANIFEST / OUT_ROOT / CHUNK_SIZE
  PY_ENV_ACTIVATE='source /PLACEHOLDER/miniconda3/etc/profile.d/conda.sh && conda activate bgr_preproc'
  ```
  plus an optional `module load` line.
- The env is activated with `eval "$PY_ENV_ACTIVATE"`. The value is user-authored config, not data paths. `set +u` is used around activation.
- They never source rc files and run `cd "$REPO_DIR"`.
- Header comments note: `mkdir -p logs` before `sbatch`; tcsh login shells are fine because sbatch honours the shebang; dependencies chain with `sbatch --dependency=afterok:<jobid>`.

Per template:
- `run_array.sh`: `START=$((SLURM_ARRAY_TASK_ID*CHUNK_SIZE)); END=$((START+CHUNK_SIZE))`, then `python -m preprocessing run --config "$CONFIG" --manifest "$MANIFEST" --out-root "$OUT_ROOT" --start "$START" --end "$END" --threads "${SLURM_CPUS_PER_TASK:-4}"`.
- `synthseg_batch.sh`: same chunking with `python -m preprocessing synthseg ...`. GPU `#SBATCH --gres=gpu:1` by default; commented CPU variant lines add `--synthseg-cpu`.
- `qc_export.sh`: `qc-summary` for `$DATASET`; `export-npy` if `DO_EXPORT=1`.
- `build_template.sh`: `build-template --lumir-root "$LUMIR_ROOT" --out-dir "$TEMPLATE_DIR" --n 200 --seed 0`.
- The scripts end with a line printing the processed range and exit status. Resumability comes from the pipeline itself.

## 11. Ordered implementation items

- [ ] 1. Scaffold the package, requirements and test runner.
      Create `__init__.py` (docstring overview), `__main__.py`, a `cli.py` skeleton with all subcommand parsers (handlers may raise `NotImplementedError` for now), `requirements.txt`, `requirements-test.txt`, `tests/__init__.py`, a minimal `tests/conftest.py`, `tests/test_cli.py` (help for every subcommand only), and `tests/run_tests_wsl.sh` exactly as in section 9.
      Files: `preprocessing/{__init__.py,__main__.py,cli.py,requirements.txt,requirements-test.txt}`, `preprocessing/tests/{__init__.py,conftest.py,test_cli.py,run_tests_wsl.sh}`
      Verify: `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/preprocessing/tests/run_tests_wsl.sh` creates `bgr_preproc`, the help tests pass and it prints `ALL CHECKS PASSED`.

- [ ] 2. Implement the pure-numpy core: `nifti.py`, `masks.py`, `intensity.py` (formula of section 4, antialias of 1.11) and `labels.py`, with tests.
      Add the `write_nifti` and `phantom` fixtures to conftest.
      Files: `preprocessing/{nifti,masks,intensity,labels}.py`, `preprocessing/tests/{test_nifti,test_masks,test_intensity,test_labels}.py`, `preprocessing/tests/conftest.py`
      Verify: the runner script passes. It must include assertions:
      - normalize of `x=[0..200]` with a full mask gives `p99.5 = 199.0` and `y = clip(x/199, 0, 1)`;
      - `antialias_sigma_mm(0.6, 1.0) ≈ 0.3397`;
      - nearest resample of a cropped grid reproduces the original labels exactly in the overlap.

- [ ] 3. Implement `config.py` and `config_example.yaml` (section 5) with tests (defaults, deep merge, validation, overrides).
      Files: `preprocessing/{config.py,config_example.yaml}`, `preprocessing/tests/test_config.py`
      Verify: the runner script passes, including a test that loads `config_example.yaml`.

- [ ] 4. Implement `manifest.py` and the four builders plus the `bids.py` helper.
      Wire up `build-manifest` (with `--exclude`) and `n-chunks` in `cli.py`. Builders write absolute paths, sorted rows, and fail on duplicate `(dataset, subject, session, contrast)`.
      Manifest columns, in order: `dataset,subject,session,contrast,split,image_path,mask_path,label_paths,skull_stripped,in_template_space,bias_corrected`.
      Files: `preprocessing/manifest.py`, `preprocessing/builders/{__init__,bids,lumir,neurite_oasis,ultracortex,generic}.py`, `preprocessing/cli.py`, `preprocessing/tests/{test_manifest,test_builders}.py`, `preprocessing/tests/test_cli.py` (add build-manifest and n-chunks CLI tests)
      Verify: the runner script passes. Builder tests check row counts, flags, label names (`seg35`, `seg4`, `manual`, `lumir10`), that `aligned_*` is ignored, that UltraCortex sessions are parsed, and the exclude list.

- [ ] 5. Implement `commands.py` (section 6) and `tools.py`, plus `tests/fake_tools.py` and the `fake_tools`/`config_file` fixtures.
      `tools.py` covers env resolution, `ToolRunner` with log capture, dry-run, threads env, versions and `check_tools`. Wire up the `check-tools` CLI.
      Files: `preprocessing/{commands,tools}.py`, `preprocessing/cli.py`, `preprocessing/tests/{fake_tools.py,conftest.py,test_commands.py,test_tools.py,test_cli.py}`
      Verify: the runner script passes. `test_commands.py` asserts the full argv lists verbatim, and `check-tools` returns exit 0 with fakes and 1 with an empty PATH.

- [ ] 6. Implement `template.py` (header check over all files, seeded subset, normalization and mean, freq mask per 1.13, `template_info.json`) and the `build-template` CLI.
      Files: `preprocessing/template.py`, `preprocessing/cli.py`, `preprocessing/tests/test_template.py`
      Verify: the runner script passes. The template equals the numpy-computed mean of the normalized selected volumes (atol 1e-6), and the mismatched-affine volume appears in `template_info.json["header_mismatches"]`.

- [ ] 7. Implement the per-subject part of `qc.py` and `meta.py`.
      `qc.py` here means `compute_qc` and `write_qc_png` (Agg; 2x3 layout per 1.15); `meta.py` is atomic and holds the step records. Both are needed by the pipeline.
      Files: `preprocessing/{qc,meta}.py`, `preprocessing/tests/test_qc.py` (per-subject parts)
      Verify: the runner script passes. QC on known arrays gives the expected Dice/NCC/volume, and the PNG exists and is > 1 kB.

- [ ] 8. Implement `pipeline.py` (section 7: steps, skip logic, resumability, dry-run, failure policy, `--clean-work`) and the `run` CLI. Depends on 2–7.
      Files: `preprocessing/pipeline.py`, `preprocessing/cli.py`, `preprocessing/tests/test_pipeline.py`
      Verify: the runner script passes. E2E tests with fakes cover the LUMIR-like row (no ANTs calls; grid-mismatch row triggers one identity antsApplyTransforms), the Neurite-like row (no synthstrip/N4; seg35/seg4 warped with GenericLabel), the UltraCortex-like 0.7 mm row (synthstrip, N4, smoothing recorded in meta) and the generic raw row. Checks:
      - outputs have the template shape/affine, image values in [0,1] as float32, and uint8 mask;
      - `--dry-run` prints `CMD` lines and creates nothing;
      - a rerun makes 0 new fake calls, while `--overwrite` reruns;
      - `FAKE_FAIL=antsRegistration` gives exit 1, `meta.status == "failed"`, the stderr text in `logs/register.log`, and the next row still processed.

- [ ] 9. Implement `synthseg.py` (batch list files, single call, postprocess with the grid check and nearest fallback, uint16, meta and QC refresh, resume) and the `synthseg` CLI; enable `synthseg` as an optional `run` step.
      Files: `preprocessing/synthseg.py`, `preprocessing/pipeline.py`, `preprocessing/cli.py`, `preprocessing/tests/test_synthseg.py`
      Verify: the runner script passes. N subjects give exactly 1 `mri_synthseg` call whose argv matches section 6. With `FAKE_SYNTHSEG_CROP=1`, the output is on the template grid and meta logs `synthseg_resampled: true`. A rerun gives 0 calls.

- [ ] 10. Implement `qc.summarize` (robust z, outlier and hard flags, `qc_summary.csv`, `manifest_processed.csv`) plus the `qc-summary` CLI, and `export.py` plus the `export-npy` CLI.
      Files: `preprocessing/{qc,export}.py`, `preprocessing/cli.py`, `preprocessing/tests/{test_qc,test_export}.py`
      Verify: the runner script passes. An injected outlier subject is flagged with a reason. The NPY round trip equals the NIfTI data, float16 export is within 1e-3, and uint8 labels > 255 raise.

- [ ] 11. Write the SLURM templates (section 10) and `tests/test_slurm_scripts.py`.
      Files: `preprocessing/slurm/{run_array,synthseg_batch,qc_export,build_template}.sh`, `preprocessing/tests/test_slurm_scripts.py`
      Verify: the runner script passes (its `bash -n` loop plus the slurm test checking shebang, no CR, and placeholders).

- [ ] 12. Final pass.
      - Make sure every module has a docstring with usage examples (no new markdown files) and that the `cli.py` epilog shows the full end-to-end sequence (check-tools → build-template → build-manifest → n-chunks → run → synthseg → qc-summary → export-npy).
      - Check Python 3.9 compatibility: no `match`, no `X | Y` unions, `from typing import Optional, List, Dict`.
      - Confirm `git status` shows only new files under `preprocessing/` (plus this task folder). Nothing is ignored unexpectedly: `git check-ignore -v preprocessing/**` prints nothing.
      - Do not commit.
      Files: any of the above.
      Verify: the full runner script passes from a clean env state, and `git status --porcelain` lists only the intended new files.

## 12. What cannot be verified locally (the final report must state this)

- Real FreeSurfer/ANTs behaviour:
  - the exact flags accepted by the cluster versions (`mri_synthstrip` threads/GPU, `mri_synthseg` txt-mode `--vol`/`--qc`, `antsRegistration --random-seed`);
  - registration quality;
  - the SynthSeg output grid on 1 mm inputs;
  - N4 results.
- Real dataset layouts: UltraCortex derivative filenames, LUMIR25 val label naming, LUMIR header affines, Neurite `norm` background exactly 0.
- SLURM submission, partitions, GPU availability, tcsh interplay, and the module names. These are syntax-checked only.

Recommended first cluster run:
1. `check-tools`.
2. `build-template`.
3. `build-manifest` for each dataset.
4. `run --dry-run --rows 0` per dataset.
5. A 5-row real run, then inspect `qc.png` before submitting arrays.

## Addendum: template/registration change (user decision, supersedes conflicting items above)

LUMIR is OpenBHB quasi-raw data (reorient, mask, N4, 1 mm, 9-DOF FLIRT to FSL MNI152_T1_1mm_brain, then N4 again, [0,255], 160x224x192 grid). Other datasets follow the same recipe instead of a mean-of-LUMIR template.

1. Registration target = MNI152 1 mm brain (T1; T2 variant per contrast via `contrast_template`), read from config (`mni_templates`, `${FSLDIR}` expanded from the resolved env). Nothing bundled in the repo.
2. `registration.backend`: `flirt` (default) or `ants`.
   - flirt: `flirt -in mov -ref MNI -omat reg.mat -dof 9 -cost corratio -searchr{x,y,z} -30 30` (optionally estimated against the 2 mm template, `flirt.estimate_at_2mm`, default true). Apply: `flirt -in X -ref template_lumirgrid -applyxfm -init M_X -out Y -interp spline|nearestneighbour`.
   - ants: Rigid then `Affine` (12) or `Similarity` (7) (`ants.transform`; ANTs has no 9-DOF); apply with `antsApplyTransforms -r template_lumirgrid -t 0GenericAffine.mat`.
   - The flirt backend reorients inputs first (step `reorient`; `registration.reorient: nibabel|fslreorient2std|none`, default nibabel = world-preserving canonical reorientation, equivalent for FLIRT because FSL coordinates flip x for neurological storage).
   - check-tools checks flirt (+fslreorient2std if selected) for flirt, antsRegistration/antsApplyTransforms for ants, and always N4/SynthStrip/SynthSeg; FSL version from `$FSLDIR/etc/fslversion`.
3. Output grid = LUMIR 160x224x192 1 mm grid. `build-target-grid` replaces `build-template`:
   - mean of a seeded subset of normalized LUMIR images (header agreement checked);
   - (a) resample MNI into the LUMIR header grid; NCC >= `--accept-ncc` (0.8) -> header accepted;
   - (b) else rigid search: 48 signed axis permutations with COM-matched translation, then Powell refinement of 3 rotations + 3 translations; if the LUMIR->MNI voxel map is a signed permutation + integer shift (rot < 0.5 deg, |frac shift| < 0.25 vox) it is snapped exactly;
   - writes `target_grid.json` (shape, affine, lumir_header_affine, method, scores, permutation/shift report), `template_lumirgrid.nii.gz` (MNI brain resampled into the grid), `template_lumirgrid_mask.nii.gz`, `lumir_mean.nii.gz`. Config `target_dir` points at this folder.
   - One interpolation per subject. flirt matrices are composed in world space (`W = A_ref S_ref^-1 M S_in A_in^-1`, then `M_X = S_T A_T^-1 W A_X S_X^-1` per input X; S = FSL scaled-voxel matrix). ANTs works in physical space, so the LUMIR-grid reference needs no extra step.
   - `affine_to_template.mat` holds the backend's native matrix (ITK for ants; FLIRT text matrix registration-input grid -> LUMIR grid for flirt). `affine_to_template_world.txt` holds the 4x4 RAS world matrix native->MNI (flirt only).
4. LUMIR rows skip reorient/strip tool/N4/registration. On-grid = shape matches and affine matches the target affine or the recorded LUMIR header affine (then the target affine is stamped, no resampling). Otherwise `grid_mismatch=true` (hard QC flag) and a numpy world-space resample (linear image, nearest mask/labels) using the row's own header. Mask = derive_mask(image>0). Same p99.5 normalization.
5. Everything else unchanged. The could-not-verify list adds: the MNI/LUMIR grid mapping, the LUMIR header check result, FSL availability/version on the cluster, and FLIRT matrix conventions on real data.

### Addendum 2: confirmed cluster facts and the revised target-grid search (supersedes item 3's (a)/(b) search)

Confirmed by the user on the cluster:
- `/autofs/space/crater_001/datasets/public/LUMIR2025/imagesVal/LUMIRMRI_3454_0000.nii.gz`: shape (160,224,192), affine = identity. The header carries no spatial information. LUMIR root: `/autofs/space/crater_001/datasets/public/LUMIR2025` (imagesVal confirmed, imagesTr expected).
- FSL: `/usr/pubsw/packages/fsl/current`. `data/standard/MNI152_T1_1mm_brain.nii.gz`: shape (182,218,182), LAS, affine [[-1,0,0,90],[0,1,0,-126],[0,0,1,-72]]. These are the example-config defaults.

Working hypothesis: LUMIR arrays are MNI152 voxel order cropped/padded per axis (x 182->160, y 218->224, z 182->192), no permutation; the identity header is wrong in x.

build-target-grid (header untrusted, no header-acceptance path; header NCC only reported):
- Mean of N (default 200, seed 0) normalized LUMIR images.
- Candidates: x flip in {no, yes} (all 48 signed permutations only with `--allow-permutations`). Per candidate, integer offsets o with `mni_idx = P*lumir_idx + o`, searched exhaustively at a coarse level (block-mean factor 4, offsets in steps of 4, range covering every crop/pad plus `--search-margin` voxels, overlap must contain >= 90 % of the mean's foreground), then refined at full resolution within +/-4 voxels. Score = Pearson NCC over the box overlap.
- Optional translation-only sub-voxel refinement (Powell) is reported; the integer mapping is used unless `--use-subvoxel`.
- Report the best offset per flip state, the NCC margin (best - best of the other flip state), and whether offsets look like a symmetric crop/pad. If margin < `--min-margin` (default 0.01): exit non-zero, write `target_grid_candidates.json` only, and tell the user to verify independently; `--force-flip {no,yes}` overrides after verification.
- Output affine = MNI_affine @ T (T = the voxel mapping). `target_grid.json` holds shape, affine, lumir_header_affine, flip/permutation, offsets, scores, margin. `template_lumirgrid.nii.gz` + mask carry the corrected affine.
- LUMIR rows in `run`: voxel arrays are never resampled. If shape matches and the header equals `lumir_header_affine` (or the corrected affine), outputs (image after normalization, mask, provided labels byte-identical) are written with the corrected affine; meta.json records original and corrected affines. Test: identity-header LUMIR row -> corrected affine, unchanged array layout.
- All other datasets are resampled once into this corrected grid.
- Final report: verifying build-target-grid on the cluster (flip margin, template_lumirgrid over a corrected LUMIR image in freeview) is the first thing to check.
