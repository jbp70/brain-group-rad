# Preprocessing pipeline: final report

The new `preprocessing/` package does the following:

- builds a manifest per dataset;
- skull-strips, N4-corrects and registers images to MNI152 1 mm (9-DOF FLIRT by default, ANTs optional);
- resamples once onto the corrected LUMIR 160x224x192 1 mm grid;
- normalizes intensities in the mask to [0, 1] (p99.5);
- adds SynthSeg labels (batch) alongside the provided labels;
- writes QC and exports `.npy`.

It is tested locally only, against fake FreeSurfer/ANTs/FSL executables. Nothing is committed. Everything under `preprocessing/` is still untracked, waiting for your review.

## Test results

- Env: WSL Ubuntu, conda env `bgr_preproc` (Python 3.9, pinned `requirements-test.txt`). It already existed for this run. An earlier implementation step created it on 2026-10-14 (`conda create --offline -n bgr_preproc python=3.9`); it did not have to be created now.
- Command: `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/preprocessing/tests/run_tests_wsl.sh`
- pytest: **91 passed, 0 failed, 0 skipped**, 14 warnings, 71.9 s. All the warnings are matplotlib/pyparsing deprecation notices.
- CLI `--help` smoke test for all 8 subcommands passed. The run ended with `ALL CHECKS PASSED`.
- `bash -n` is clean on all 5 shell files: `slurm/build_target_grid.sh`, `slurm/qc_export.sh`, `slurm/run_array.sh`, `slurm/synthseg_batch.sh`, `tests/run_tests_wsl.sh`. There are no `.sbatch`/`.slurm` files.
- No temp files are left in the repo: no `__pycache__`, no `.pytest_cache`. Test NIfTIs live in pytest `tmp_path` under Linux `/tmp`.

## File list (`preprocessing/`, all new)

| File | Purpose |
|---|---|
| `__init__.py` | Package overview, output layout, end-to-end usage; `__version__`. |
| `__main__.py` | Entry point for `python -m preprocessing`. |
| `cli.py` | argparse CLI: `check-tools`, `build-manifest`, `build-target-grid`, `run`, `synthseg`, `qc-summary`, `export-npy`, `n-chunks`. |
| `config.py` | YAML loading, merge onto defaults, validation, CLI overrides, `${VAR}` expansion, target/MNI path helpers. |
| `config_example.yaml` | Example config with `PLACEHOLDER` markers. |
| `commands.py` | Pure argv builders for SynthStrip, N4, FLIRT, fslreorient2std, antsRegistration, antsApplyTransforms, SynthSeg. |
| `tools.py` | `env_setup` resolution, `ToolRunner` (logging, dry-run, thread env), version detection, `check_tools`. |
| `grid.py` | `build-target-grid`: LUMIR mean, flip/offset search against MNI152, corrected affine, `template_lumirgrid*`; `Target` loader. |
| `nifti.py` | NIfTI load/save (qform = sform), grid/affine comparison, FLIRT ↔ world matrix conversion. |
| `masks.py` | Mask derivation (threshold, fill holes, largest component), Dice. |
| `intensity.py` | p99.5 in-mask normalization; anti-alias sigma and smoothing for sub-1 mm inputs. |
| `labels.py` | Nearest-neighbour world-space label resampling; safe uint16/uint8 casting. |
| `manifest.py` | `ManifestRow`, CSV read/write, label-path parsing, subject dir naming. |
| `meta.py` | Atomic `meta.json`, running/done step records. |
| `pipeline.py` | Per-row steps `reorient, strip, n4, register, resample, normalize, [synthseg], qc`, plus resumability, dry-run and failure policy. |
| `synthseg.py` | Batch SynthSeg (one `mri_synthseg` call), grid check with nearest fallback, QC refresh. |
| `qc.py` | Per-subject QC metrics and PNG; `qc-summary` robust-z outliers, `qc_summary.csv`, `manifest_processed.csv`. |
| `export.py` | Processed NIfTI → parallel `.npy` layout plus `export_manifest.csv`. |
| `requirements.txt` | Runtime pins: numpy 1.26.4, scipy 1.11.4, nibabel 5.2.1, PyYAML 6.0.2, matplotlib 3.9.2. |
| `requirements-test.txt` | Runtime pins plus pytest. |
| `builders/__init__.py` | `BUILDERS` registry, `--exclude` handling. |
| `builders/bids.py` | BIDS filename entity parsing and derivative matching. |
| `builders/lumir.py` | LUMIR builder (`LUMIRMRI_*_0000.nii.gz`; split from `imagesTr`/`imagesVal`/`imagesTs`; optional `--label-glob`). |
| `builders/neurite_oasis.py` | Neurite-OASIS builder (folders with `norm.nii.gz`, labels `seg35`/`seg4`, never `aligned_*`). |
| `builders/ultracortex.py` | UltraCortex ds005216 BIDS builder (derivative masks and manual labels matched by entities). |
| `builders/generic.py` | Generic BIDS/folder builder (for OASIS-2, Mindboggle-101 later). |
| `slurm/build_target_grid.sh` | SLURM job: `build-target-grid`. |
| `slurm/run_array.sh` | SLURM CPU array: manifest chunks through `run`. |
| `slurm/synthseg_batch.sh` | SLURM GPU (or CPU) array: manifest chunks through `synthseg`. |
| `slurm/qc_export.sh` | SLURM job: `qc-summary`, optionally followed by `export-npy`. |
| `tests/__init__.py` | Makes the test helpers importable. |
| `tests/conftest.py` | Fixtures: synthetic MNI/target files, fake tools on PATH, config file. |
| `tests/datasets.py` | Synthetic dataset rows (tiny NIfTIs) for end-to-end tests. |
| `tests/phantoms.py` | World-space brain phantoms that can be rendered on any grid. |
| `tests/fake_impl.py` | Behaviour of the fake external tools. |
| `tests/fake_tools.py` | Installs the fake executables into a bin dir. |
| `tests/run_tests_wsl.sh` | WSL runner: env setup, `bash -n`, pytest, CLI smoke test. |
| `tests/test_builders.py` | Builder trees for LUMIR, Neurite-OASIS, UltraCortex, generic; `--exclude`. |
| `tests/test_cli.py` | `--help` for each subcommand, check-tools, build-manifest, n-chunks via CLI. |
| `tests/test_commands.py` | Exact argv for every external tool. |
| `tests/test_config.py` | Defaults, merge, validation, overrides, example config loads. |
| `tests/test_export.py` | NPY round trip, dtypes, overflow error, template export. |
| `tests/test_grid.py` | Target-grid recovery of flip and crop/pad offsets from identity-header volumes. |
| `tests/test_intensity.py` | Normalization formula; anti-alias sigma and axis selection. |
| `tests/test_labels.py` | Nearest resampling preserves labels; dtype casting. |
| `tests/test_manifest.py` | Manifest round trip, label-path parsing, duplicates, dir names. |
| `tests/test_masks.py` | Mask derivation and Dice edge cases. |
| `tests/test_nifti.py` | Affine/grid matching, qform/sform, FLIRT matrix conversion. |
| `tests/test_pipeline.py` | End-to-end `run` with fakes: all dataset types, dry-run, resume, kill-during-step, failures. |
| `tests/test_qc.py` | QC metrics, PNG, robust z, summary CSVs. |
| `tests/test_slurm_scripts.py` | Shebang, LF, `bash -n`, `--account=qtim`, placeholder checks on the SLURM templates. |
| `tests/test_synthseg.py` | Batch txt files, single call, crop fallback, resume. |
| `tests/test_tools.py` | Env resolution, runner logs/errors/dry-run, version parsing, check_tools. |

## Cluster smoke test (run these yourself, in order)

First review the code, then commit and push it, then pull it on the cluster. Don't run anything on `mlsc-login`. Use a compute node: an interactive session for the smoke test, `sbatch` for the full run.

### 0. Session, variables, Python env, config

```bash
# On mlsc-login: one interactive GPU node covers all steps, including SynthSeg
srun -A qtim -p rtx8000 --gpus 1 --time=04:00:00 --mem=32G --cpus-per-task=3 --ntasks-per-node=1 --pty bash

PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg   # repo clone
LUMIR=/autofs/space/crater_001/datasets/public/LUMIR2025
NEURITE=/PLACEHOLDER/neurite-oasis      # <-- FILL: Neurite-OASIS root
UC=/PLACEHOLDER/ds005216                # <-- FILL: UltraCortex BIDS root
D=$PROJ/data                            # data/ is gitignored
CFG=$D/preproc_config.yaml
mkdir -p $D/manifests $D/processed $D/npy $PROJ/logs
cd $PROJ && git pull   # stay in $PROJ: python -m preprocessing needs the repo root as cwd
# freeview (steps 3-4) needs a display: run those checks wherever you have X (e.g. glacier)

# One-time Python env (PLACEHOLDER conda path)
source /PLACEHOLDER/miniconda3/etc/profile.d/conda.sh
conda create -y -n bgr_preproc python=3.9      # first time only
conda activate bgr_preproc
pip install -r $PROJ/preprocessing/requirements.txt

cp $PROJ/preprocessing/config_example.yaml $CFG
# Edit $CFG: env_setup (FreeSurfer/ANTs lines), target_dir: $D/target_grid (write the literal path)
```

### 1. Tools

```bash
python -m preprocessing check-tools --config $CFG
```

A pass looks like this:
- every tool resolves;
- FreeSurfer is 7.3 or newer, and `FREESURFER_HOME` and `FSLDIR` are set;
- the MNI T1 files are `OK`.

Two `MISSING` lines are expected at this stage and are not errors: the `target ...` lines (`build-target-grid` hasn't run yet) and the T2 template (only needed for T2/FLAIR inputs).

### 2. Manifests

```bash
# LUMIR: while imagesTr is still downloading, restrict to imagesVal so half-written files aren't picked up
python -m preprocessing build-manifest --dataset lumir --root $LUMIR --out $D/manifests/lumir.csv \
    --image-glob 'imagesVal/LUMIRMRI_*_0000.nii.gz'
python -m preprocessing build-manifest --dataset neurite_oasis --root $NEURITE --out $D/manifests/neurite_oasis.csv
python -m preprocessing build-manifest --dataset ultracortex --root $UC --out $D/manifests/ultracortex.csv
```

Check the row counts in the logs:
- LUMIR: imagesVal row count;
- Neurite-OASIS: about 414 subjects expected, and how many have `seg35`;
- UltraCortex: 86 images, 12 with manual labels, plus the number with derivative masks.

Run `head -3` on each CSV to confirm the paths look right. After the LUMIR download finishes, rebuild without `--image-glob` to include `imagesTr`.

### 3. Target grid (verify this before running anything else)

```bash
python -m preprocessing build-target-grid --config $CFG --lumir-root $LUMIR --split val --out-dir $D/target_grid
# exit 2 = ambiguous x-flip: inspect target_grid_candidates.json, verify, then rerun with --force-flip yes|no
freeview $D/target_grid/lumir_mean.nii.gz $D/target_grid/template_lumirgrid.nii.gz:colormap=heat:opacity=0.4
```

The run prints both flip candidates, their offsets, the NCC margin and the corrected affine. In freeview the MNI template must overlay the LUMIR mean exactly, including left/right. `--split val` uses only the downloaded validation images; rerun later with `--split any` (up to 200 images) if you like. Rerun `check-tools` afterwards: the `target` lines should now be `OK`.

### 4. `run` on 3 subjects per dataset (dry run first, then real)

```bash
for ds in lumir neurite_oasis ultracortex; do
  python -m preprocessing run --config $CFG --manifest $D/manifests/$ds.csv --out-root $D/processed --rows 0,1,2 --dry-run
done
for ds in lumir neurite_oasis ultracortex; do
  python -m preprocessing run --config $CFG --manifest $D/manifests/$ds.csv --out-root $D/processed --rows 0,1,2 --threads 3
done
```

The dry run prints `CMD ...` and `PY ...` lines and writes nothing.

After the real run, check the following:
- the exit status is 0;
- each `$D/processed/<ds>/<subject>_T1w/` has `image.nii.gz`, `mask.nii.gz`, `meta.json`, `qc.json` and `qc.png`;
- any failure is logged in `logs/pipeline.log` and the step logs;
- for one FLIRT row (Neurite-OASIS or UltraCortex), `freeview image.nii.gz $D/target_grid/template_lumirgrid.nii.gz:colormap=heat:opacity=0.4` shows good alignment.

### 5. SynthSeg batch (needs the GPU session)

```bash
for ds in lumir neurite_oasis ultracortex; do
  python -m preprocessing synthseg --config $CFG --manifest $D/manifests/$ds.csv --out-root $D/processed --rows 0,1,2 --dry-run
  python -m preprocessing synthseg --config $CFG --manifest $D/manifests/$ds.csv --out-root $D/processed --rows 0,1,2 --threads 3
done
```

Check that `labels_synthseg.nii.gz` exists. Then look at `meta.json` → `synthseg_resampled`: `true` means SynthSeg's output grid differed from the target and was resampled with nearest neighbour. If the run fails on `--vol`/`--qc` in txt mode, set `synthseg.vol: false` / `qc: false` in the config.

### 6. QC summary and export

```bash
for ds in lumir neurite_oasis ultracortex; do
  python -m preprocessing qc-summary --out-root $D/processed --dataset $ds
done
python -m preprocessing export-npy --out-root $D/processed --export-root $D/npy \
    --dataset lumir --dataset neurite_oasis --dataset ultracortex --include-template --config $CFG
exit   # release the allocation
```

## Full run via SLURM templates

Every variable in the `=== EDIT ME ===` block uses `${VAR:-default}`. You can therefore pass values with `--export=ALL,...` instead of editing the files.

Each `sbatch` command below needs `#SBATCH --partition` filled in first (see Placeholders). Submit from `$PROJ`, so that `logs/%x_%A_%a.out` lands in `$PROJ/logs`; stderr goes into the same file. An `--array=` on the command line overrides the `#SBATCH --array=0-0` placeholder.

How chunking works:
- Array task `i` processes manifest rows `[i*CHUNK_SIZE, (i+1)*CHUNK_SIZE)`.
- The array range is `0-(N-1)`, where `N = n-chunks(manifest, CHUNK_SIZE)`. Compute `N` separately for `run` and for `synthseg`, because their default chunk sizes differ (20 and 100).
- `%2` limits an array to 2 tasks running at once.
- Rerunning a failed chunk is safe: rows and steps that are done get skipped.

```bash
cd $PROJ
ENVV="ALL,REPO_DIR=$PROJ,CONFIG=$CFG,OUT_ROOT=$D/processed"

# (once, if not done interactively) target grid
sbatch --export=$ENVV,TARGET_DIR=$D/target_grid preprocessing/slurm/build_target_grid.sh

# per dataset (example: neurite_oasis; repeat with lumir / ultracortex)
DS=neurite_oasis; M=$D/manifests/$DS.csv
N=$(python -m preprocessing n-chunks --manifest $M --chunk-size 20)
J1=$(sbatch --parsable --array=0-$((N-1))%2 --export=$ENVV,MANIFEST=$M,CHUNK_SIZE=20 preprocessing/slurm/run_array.sh)
NS=$(python -m preprocessing n-chunks --manifest $M --chunk-size 100)
J2=$(sbatch --parsable --dependency=afterok:$J1 --array=0-$((NS-1)) --export=$ENVV,MANIFEST=$M,CHUNK_SIZE=100 \
     preprocessing/slurm/synthseg_batch.sh)
sbatch --dependency=afterok:$J2 --export=$ENVV,DATASET=$DS,DO_EXPORT=1,EXPORT_ROOT=$D/npy preprocessing/slurm/qc_export.sh
```

- CPU variant of SynthSeg: delete the `#SBATCH --gpus=1` line, switch to the commented CPU partition/mem/time lines, and add `SYNTHSEG_CPU=1` to `--export`.
- `afterok` on an array waits for every task to succeed. If any chunk fails, the dependent job never starts. Rerun that chunk with `--array=<i>`, then submit the next stage by hand.
- `python -m preprocessing n-chunks` needs the env active on the login node. If you'd rather not, compute `ceil(rows / chunk)` yourself from `wc -l` minus 1.

## Placeholders you must fill before running

- **Config `env_setup`** (`config_example.yaml`): the `module load freesurfer/7.4.1` and `module load ants/2.5.0` lines are placeholders. Martinos may not use modules. If it doesn't, use `export FREESURFER_HOME=<path>` and `source $FREESURFER_HOME/SetUpFreeSurfer.sh` (the commented lines), plus an `export PATH=<ants>/bin:$PATH`. The FSL lines are already set to the confirmed `/usr/pubsw/packages/fsl/current`.
- **FREESURFER_HOME**: it must be set by `env_setup` or by the SLURM script. check-tools reads `$FREESURFER_HOME/build-stamp.txt`.
- **Config `target_dir`**: `/PLACEHOLDER/brain-group-rad-data/target_grid`. Set it to the `--out-dir` used for build-target-grid, e.g. `$PROJ/data/target_grid` written as a literal path.
- **Config `tools`**: these are bare names resolved on PATH. Use absolute paths if PATH isn't set by `env_setup`. ANTs is only needed for `registration.backend: ants`.
- **Config `mni_templates.T2.image_1mm` / `image_2mm`**: placeholder paths. FSL doesn't ship them, and they're only needed for T2w/FLAIR/PDw inputs (none yet).
- **SLURM `#SBATCH`, all four templates**:
  - `--partition`: `PLACEHOLDER_PARTITION` / `PLACEHOLDER_GPU_PARTITION`. Suggested: `basic` for CPU jobs, `rtx8000` for SynthSeg on GPU.
  - `--time` / `--mem` / `--cpus-per-task`: starting guesses (e.g. 4 h / 16G / 4 for `run_array`). Keep them within the fair-share table: `rtx8000` is about 3 CPUs per GPU, `basic` is about 15 GB per core.
  - `--qos`: commented out (`##SBATCH`). Uncomment only if required.
  - `--account=qtim` is already set.
- **SLURM EDIT block**: `REPO_DIR`, `CONFIG`, `MANIFEST`, `OUT_ROOT`, `TARGET_DIR`, `EXPORT_ROOT` (or pass them with `--export`), and `PY_ENV_ACTIVATE` (`source /PLACEHOLDER/miniconda3/etc/profile.d/conda.sh && conda activate bgr_preproc`). The optional `module load` lines are commented out. Use them only if `env_setup` in the config is empty.
- **Dataset roots**: `$NEURITE` and `$UC` above.

## Could not verify locally

Every external tool in the tests is a fake, so these need a cluster run:

- **Real FreeSurfer/ANTs/FSL behaviour and versions on the cluster**:
  - whether the FreeSurfer install is ≥ 7.3 and what `module load` names exist;
  - which `mri_synthstrip` thread/GPU flags the installed version accepts (none are passed by default; see `synthstrip.extra_args`);
  - whether `mri_synthseg` txt-list mode accepts `--vol`/`--qc` txt files;
  - whether the installed `antsRegistration` supports `--random-seed`;
  - N4 output and registration quality.
- **FLIRT matrix convention on real data**: the fake `flirt` uses the same FSL scaled-voxel convention as the code, so a convention error would not show up in the tests. Check the freeview overlay of the first FLIRT row (step 4).
- **SynthSeg output grid**: whether SynthSeg returns the 160x224x192 input grid or something else. A mismatch is handled by nearest-neighbour resampling and logged as `synthseg_resampled`.
- **LUMIR**:
  - Header consistency across the whole dataset: only `LUMIRMRI_3454_0000` was confirmed (160x224x192, identity affine). Header mismatches are recorded in `target_grid.json` and flagged per row as `grid_mismatch`.
  - The flip/offset result of build-target-grid on real data.
  - Whether imagesTr follows the same naming.
  - The file naming of any LUMIR val labels (`--label-glob`, default label name `lumir10`).
- **Neurite-OASIS layout (assumption)**:
  - Each subject folder contains `norm.nii.gz`, `orig.nii.gz`, `seg35.nii.gz` and `seg4.nii.gz` at 256³ in scanner space.
  - The subject ID is the folder name (e.g. `OASIS_OAS1_0001_MR1`).
  - The background of `norm` is exactly 0. The mask is `(norm > 0) | (seg35 > 0)`.
  - The FreeSurfer intensity distribution is only rescaled by p99.5, not reshaped.
- **UltraCortex layout (assumption)**:
  - Raw images are at `sub-*/[ses-*/]anat/*_T1w.nii[.gz]`.
  - Derivatives are under `derivatives/**`:
    - masks have "mask" in the filename, and paths containing "synthstrip" are preferred;
    - manual labels have "manual" in the path or the BIDS suffix `dseg`.
  - The exact derivative folder and file names are unknown. Compare the logged match counts with the expected 86 images and 12 with manual labels.
- **SLURM**: the templates are syntax-checked only. Partitions, GPU availability, run times and memory use haven't been tested.

Known minor issue (non-blocking, from review): the SynthSeg batch rewrites `qc.json` without a temp file. If a job is killed in that short window, the file can be left truncated, and `qc-summary` will then fail. Regenerating that subject's QC should fix it: rerun `run --steps qc --overwrite --rows <i>`. That fix is untested.

## Container update (2026-10-14)

On Martinos machines the pipeline now runs inside the lab's Singularity image `tf10_nv.sif` (Python 3.8.10, numpy 1.21.1, scipy 1.4.1, nibabel 3.1.1, PyYAML 6.0, matplotlib 3.5.0) instead of a conda env. Nothing has to be installed on the cluster. This section supersedes the conda/`module load`/`PY_ENV_ACTIVATE` instructions and the `/usr/pubsw` paths above.

### What changed

- `run_in_container.sh` (new): runs `singularity exec [--nv] -B ... tf10_nv.sif env <tool vars> bash -c '<fixed script>' run_in_container "$@"`.
  - Your arguments are passed as separate argv entries after a fixed, single-quoted script. Nothing is interpolated into `bash -c`, so paths with spaces or shell characters are safe.
  - Defaults are the confirmed values, each overridable by an env var: `BGR_SIF`, `BGR_BINDS` (`/scratch /autofs /cluster /space /vast /usr/local/freesurfer`; a bind missing on the host is skipped with a warning), `BGR_REPO_DIR` (the folder above the script, put on `PYTHONPATH`), `BGR_TMPDIR`, `BGR_FREESURFER_HOME`, `BGR_FSLDIR`, `BGR_ANTS_BIN`, `BGR_PYTHON`, `BGR_SINGULARITY` (e.g. `apptainer`).
  - `TMPDIR` inside the container is `/scratch/$SLURM_JOB_ID` on SLURM nodes, else `$PROJ/tmp` (created). The glacier host `TMPDIR` (`/local_mount/...`, not bound) is never used.
  - `--nv` is added only with `USE_GPU=1`, or in auto mode when `nvidia-smi -L` lists a GPU. `USE_GPU=0` disables it.
  - The current directory is kept (`cd` inside the container), so relative paths work as long as you are under `/autofs` etc. The repo no longer has to be the working dir.
  - Call it as `bash .../run_in_container.sh` so it does not need the executable bit after a Windows checkout.
- Config defaults (`config.py`) and `config_example.yaml`:
  - `env_setup` exports `FREESURFER_HOME=/usr/local/freesurfer/dev`, `FSLDIR=/autofs/cluster/pubsw/2/pubsw/Linux2-2.3-x86_64/packages/fsl.64bit/5.0.7`, `FSLOUTPUTTYPE=NIFTI_GZ` and `PATH=$FREESURFER_HOME/bin:$FSLDIR/bin:/autofs/cluster/pubsw/2/pubsw/Linux2-2.3-x86_64/packages/ANTS/2.4.4/bin:$PATH`.
  - `tools` stay bare names resolved on that PATH; `check-tools` prints the absolute path each resolves to.
  - The MNI T1 1 mm / 2 mm / mask defaults are the absolute paths in that FSL's `data/standard`. The T2 templates remain PLACEHOLDERs.
  - The example's `target_dir` is now `$PROJ/data/target_grid` (literal path).
  - New key `tmp_dir` (default `null`): the folder exported to every tool as `TMPDIR`/`TMP`/`TEMP`. With `null` it is `$TMPDIR` if that exists and is writable inside the container, else `<out-root>/_tmp`. The `_` prefix keeps `export-npy` from treating it as a dataset.
- `check-tools` accepts the FreeSurfer dev build stamp (`freesurfer-linux-centos7_x86_64-dev-20260312`, no x.y.z) as OK. Before this change it reported "version unknown" and failed.
- SLURM templates:
  - All four call `bash $REPO_DIR/preprocessing/run_in_container.sh` (via `RIC`). The `PY_ENV_ACTIVATE`/conda/`module` lines are gone.
  - Defaults point at `$PROJ`: repo `$PROJ/brain-group-rad`, config `$PROJ/data/preproc_config.yaml`, outputs `$PROJ/data/...`.
  - `--nodes=1` and `--ntasks-per-node=1` were added.
  - `--output`/`--error` are absolute paths under `$PROJ/logs`. Create that folder before `sbatch`.
  - The arrays default to `%2`.
  - `synthseg_batch.sh` uses `--gpus=1` with 3 CPUs and 32G (rtx8000 fair share). It sets `USE_GPU=1`, or `USE_GPU=0` with `SYNTHSEG_CPU=1`. The CPU jobs set `USE_GPU=0`.
  - `build_target_grid.sh` gained `SPLIT` (default `any`, same as before).
  - Partitions stay PLACEHOLDER, with suggestions in comments: `basic` for CPU, `rtx8000` for SynthSeg.
  - The usage comments compute the array size with `tail`/`wc` because no Python runs on the login node.
- Python 3.8 / old-library compatibility: a grep audit plus a test run of the unchanged package on the container versions found no incompatibilities, so no source change was needed for this part. Details are in the test log.
- Requirements files:
  - `requirements.txt` now lists the container versions and says nothing needs installing on the cluster.
  - `requirements-test.txt` is those versions plus pytest 8.3.5 (env `bgr_preproc38`).
  - New `requirements-test-py39.txt` holds the old Python 3.9 pins (env `bgr_preproc`).
- `tests/run_tests_wsl.sh [env] [pytest args]`: the default env is `bgr_preproc38`.
- Tests:
  - New `tests/test_container.py`: a fake `singularity` asserts the exact argv, binds and safe argument passing, plus the `--nv` logic. A temp-dir test checks that tool temp files land in `<out-root>/_tmp` or the configured `tmp_dir`, never in an unbound `TMPDIR`.
  - The fake tools now create a temp file and log where it went.
  - `conftest.write_config` sets `env_setup: []`.
  - The SLURM, config and tools tests were updated.

### Test results

The full log is in `.agents/tasks/preprocessing-pipeline/container-test-log.txt`.

- `run_tests_wsl.sh bgr_preproc38`: Python 3.8.20 with numpy 1.21.1, scipy 1.4.1, nibabel 3.1.1, PyYAML 6.0, matplotlib 3.5.0 and pytest 8.3.5.
  - Result: **102 passed, 0 failed, 0 skipped**. `bash -n` and `--help` are OK.
  - The 140 warnings all come from inside scipy 1.4 and nibabel 3.1 (`np.int` and similar aliases).
- `run_tests_wsl.sh bgr_preproc`: Python 3.9.18 with numpy 1.26.4, scipy 1.11.4, nibabel 5.2.1 and matplotlib 3.9.2.
  - Result: **102 passed, 0 failed, 0 skipped**.
- How `bgr_preproc38` was built: conda channels were unreachable (WSL has no DNS, and repo.anaconda.com was blocked from Windows). The env is an offline-created conda env holding CPython 3.8.20 from python-build-standalone, with pip wheels downloaded on Windows. The container has Python 3.8.10; every library pin is exact.

### Cluster smoke test with the container (ONE subject)

Review first, then commit/push and `git pull` in `$PROJ/brain-group-rad`.

```bash
# (1) Get a compute node -- never run this on mlsc-login.
#   Option A, glacier: pick ONE free GPU, then pin it on every command
nvidia-smi                       # choose an index with no processes / little memory used
export CUDA_VISIBLE_DEVICES=<i>
#   Option B, cluster:
srun -A qtim -p rtx8000 --gpus 1 --time=02:00:00 --mem=32G --cpus-per-task=3 --ntasks-per-node=1 --pty bash

PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
LUMIR=/autofs/space/crater_001/datasets/public/LUMIR2025
D=$PROJ/data; CFG=$D/preproc_config.yaml
RIC="bash $PROJ/brain-group-rad/preprocessing/run_in_container.sh"
mkdir -p $D/manifests $D/processed $PROJ/logs $PROJ/tmp
cd $PROJ/brain-group-rad && git pull
cp preprocessing/config_example.yaml $CFG     # defaults are already the Martinos paths

# (2) Tools, inside the container
$RIC check-tools --config $CFG
#   expect: every tool resolves to an absolute /usr/local/freesurfer/dev/bin, .../fsl.64bit/5.0.7/bin
#   or .../ANTS/2.4.4/bin path; "FreeSurfer dev build 20260312 OK"; MNI T1 files OK.
#   Expected MISSING: the target lines (step 4 not run yet) and the T2 placeholders.

# (3) Manifest: imagesVal only while imagesTr is still being unzipped
$RIC build-manifest --dataset lumir --root $LUMIR --out $D/manifests/lumir.csv \
    --image-glob 'imagesVal/LUMIRMRI_*_0000.nii.gz'
head -3 $D/manifests/lumir.csv

# (4) Target grid from imagesVal, then check the overlay
$RIC build-target-grid --config $CFG --lumir-root $LUMIR --split val --out-dir $D/target_grid --force-flip yes
#   exit 2 = ambiguous x-flip: inspect target_grid_candidates.json, rerun with --force-flip yes|no
#   freeview needs X: run it on the host (glacier), not in the container
freeview $D/target_grid/lumir_mean.nii.gz $D/target_grid/template_lumirgrid.nii.gz:colormap=heat:opacity=0.4
$RIC check-tools --config $CFG    # target lines now OK

# (5) One row: dry run, then real
$RIC run --config $CFG --manifest $D/manifests/lumir.csv --out-root $D/processed --rows 0 --dry-run
$RIC run --config $CFG --manifest $D/manifests/lumir.csv --out-root $D/processed --rows 0 --threads 3

# (6) SynthSeg on that row (GPU: --nv is added automatically when nvidia-smi sees one; or USE_GPU=1)
$RIC synthseg --config $CFG --manifest $D/manifests/lumir.csv --out-root $D/processed --rows 0 --threads 3
#   first real mri_synthseg run inside the container: if it fails on --vol/--qc txt mode set
#   synthseg.vol/qc: false; if TF cannot see the GPU, check `USE_GPU=1` and CUDA_VISIBLE_DEVICES

# (7) QC
$RIC qc-summary --out-root $D/processed --dataset lumir
ls $D/processed/lumir/*/                       # image, mask, labels_*, meta.json, qc.json, qc.png
#   open qc.png; on glacier: freeview <subject>/image.nii.gz <subject>/labels_synthseg.nii.gz:colormap=lut
exit   # release the srun allocation / unset CUDA_VISIBLE_DEVICES on glacier
```

For a LUMIR row only SynthSeg runs a tool (no strip/N4/FLIRT). To exercise FLIRT in the container as well, repeat steps 5–7 with one Neurite-OASIS or UltraCortex row once those datasets are available.

### Could not verify (in addition to the list above)

- **`mri_synthseg` inside the container**: it has not been run there. Only `mri_synthstrip -h` was checked inside. It may need TF/GPU libraries from the FreeSurfer install that conflict with the image's TF 2.10, or a model file under `$FREESURFER_HOME/models`.
- **mlsc compute nodes**: assumed to have the same paths as glacier (`/usr/local/freesurfer/dev`, the `/autofs/cluster/pubsw/...` FSL 5.0.7 and ANTs 2.4.4 folders, `/vast`, `/space`) and `singularity` at `/usr/bin`. A missing bind source is skipped with a warning, but a missing FreeSurfer tree would make check-tools fail.
- **FSL in the `fsl.64bit/5.0.7` folder reports `FLIRT version 6.0`**. The flags used (`-dof 9 -cost corratio -searchr* -applyxfm -init -interp spline|nearestneighbour -datatype float`) exist in both. Check that `-interp spline` is accepted and look at the overlay of the first FLIRT row. check-tools shows whatever `$FSLDIR/etc/fslversion` says.
- **`python` inside the image**: the wrapper calls `python` (fact: Python 3.8.10). Set `BGR_PYTHON=python3` if `python` is not on the container PATH.
- **Wrapper on the real hosts**: only tested with a fake `singularity`. The real `singularity exec` passing the host env, and `env`/`bash` existing in the image, are assumed.
- **SLURM**: the templates are still syntax/policy-checked only.

## Storage and left/right update (2026-10-15)

### Left/right: the LUMIR grid needs the x flip

build-target-grid could not decide the x flip from image correlation on LUMIR (NCC 0.97107 no flip vs 0.97127 flip; margin 0.0002, below the 0.01 minimum), and the validation run used `--force-flip no`. Two independent checks show that choice is wrong:

- Organizer labels (`LUMIR2025/SanityCheckLabelsTr`, SLANT/Neuromorphometrics ids, e.g. 47 = Right and 48 = Left Hippocampus) put Right structures at high voxel x (~89–105) and Left at low x (~51–68) in all 5 cases. Our SynthSeg, with the no-flip header, put Left-Hippocampus (17) at x=104 and Right (53) at x=52.
- SynthSeg volume asymmetries over the 49 cases run against well-known population asymmetries. Fraction of cases where our "Left" was larger: thalamus 0.00, pallidum 0.02, lateral ventricle 0.20, hippocampus 0.76 (expected: mostly L > R for the first three, mostly R > L for hippocampus).

So in LUMIR voxel space the subject's left is at low x. The correct grid is the flipped candidate `-x+y+z` (offset [9, -2, -20]; corrected affine x-row [1, 0, 0, ...], RAS-like in x): use `--force-flip yes`. Voxel data is never changed, only the affine stamped on the outputs.

- `qc-summary` now runs this check automatically when `synthseg_vols.csv` files exist. It writes `lr_asymmetry.csv` and `qc_summary.json` (`lr_reversed_warning`), and prints `WARNING: LEFT/RIGHT LOOKS REVERSED` when more than half of the 4 structures show the unexpected pattern in a dataset with at least 20 cases (`--lr-min-cases`, `--lr-reversed-below 0.4`). It is a population heuristic, not a per-subject test.
- `build-target-grid` (ambiguity error, `--force-flip` help and the printed output when the margin is small) now recommends checking with organizer labels or this asymmetry check before choosing `--force-flip`.

### New storage defaults (config `outputs`)

| Key | Default | Effect |
|---|---|---|
| `outputs.keep_work` | `false` | `work/` is deleted once a row is fully done: status `completed` and every requested step done, including SynthSeg when it is in `run --steps`. The `synthseg` batch writes and re-reads `labels_synthseg.nii.gz`, refreshes QC, then removes `work/` (its raw output) for completed rows. `--keep-work` (run, synthseg) keeps it; `run --clean-work` forces deletion even if the config keeps it. Partial/failed rows keep `work/`. |
| `outputs.save_mask` | `false` | `mask.nii.gz` is not written (still computed and used for normalization). QC, export and qc-summary use `image > 0` instead (`qc.json` → `mask_source`). That is a subset of the saved mask: normalization clips to [0, 1], so it loses filled interior holes and every in-mask voxel normalized to 0 (intensity at or below the lower bound: <= 0 by default, the whole `lower_percentile` tail when that is set). |
| `outputs.save_qc_png` | `false` | `qc.png` is not written; `qc.json` always is. |

- Resume still works after cleanup: meta.json `work_cleaned` lets `run` skip finished rows, and the SynthSeg batch only needs `image.nii.gz`. Rerunning a single step that needs `work/` (e.g. `--steps normalize --overwrite`) fails with a hint to rerun all steps with `--overwrite`.
- The old config key `qc.png` now raises a config error that names `outputs.save_qc_png`. Re-copy `config_example.yaml` or delete that key from `$CFG`.
- LUMIR per case: ~16.6 MB before, ~7.8 MB of NIfTI masters now (image 7.2 MB, synthseg 0.55 MB, plus lumir10 0.4 MB where labels exist, and small json/csv/logs).

PNGs on demand and cleanup of existing outputs:

```bash
$RIC qc-summary --out-root $D/processed --dataset lumir --config $CFG --png flagged    # or all, sample:20 [--png-seed 0]
$RIC cleanup --out-root $D/processed --dataset lumir --work --mask --qc-png --dry-run  # prints paths + bytes
$RIC cleanup --out-root $D/processed --dataset lumir --work --mask --qc-png
```

`cleanup` only deletes `<subject>/work/`, `mask.nii.gz` and `qc.png`. It refuses rows that are not `completed` or that have any step not `done` (exit status 3), and never follows symlinks.

### export-npy defaults

- The image is float16 (`--image-dtype float32` to keep full precision).
- Labels are remapped to contiguous ids with a fixed LUT per label name (`preprocessing/luts.py`):
  - `synthseg`: the 32 structures of SynthSeg's labels table plus DK parcels 1001–1035/2001–2035, 103 ids → uint8;
  - `lumir10` 0..10, `seg4` 0..4, `seg35` 0..35 are identity LUTs whose values are unverified.
- Any value outside a LUT fails the export loudly. The LUTs go to `label_luts.json`; `export_manifest.csv` records `image_dtype`, `label_dtypes` and `label_luts`. uint16 is used when a LUT has more than 256 ids, and `--no-remap-labels` keeps the raw uint16 ids.
- No `mask.npy` by default (`--include-mask` derives it as above).
- About 21 MB/case uncompressed (image 13.8 + synthseg 6.9 MB, +6.9 MB for lumir10) vs ~41 MB before.
- `slurm/qc_export.sh` defaults to `IMAGE_DTYPE=float16`.
- Without `--overwrite`, an existing `.npy` is kept only if its dtype and shape match this run; otherwise the export fails and names the file. Kept files are counted in a warning. Their contents aren't re-checked, so after reprocessing pass `--overwrite` or delete the old export folder.

### LUMIR builder

- `LUMIRMRI_<id>_0001.nii.gz` (LUMIR25 extra validation cases, non-T1) now get contrast `unknown0001` and subject `LUMIRMRI_<id>_0001`, so their folders are `LUMIRMRI_<id>_0001_unknown0001`. `_0000` files keep contrast `T1w` and subject `LUMIRMRI_<id>`.
- The default image glob is `**/images*/LUMIRMRI_*_000[0-9].nii.gz`. It is restricted to `images*` folders because `simplelabelsVal`/`SanityCheckLabelsTr` reuse the image names. The old default `**/LUMIRMRI_*_0000.nii.gz` would have picked those files up as images; build-target-grid's default had the same problem, and `--split val` avoided it.
- `--label-glob 'simplelabelsVal/{id}*.nii.gz'` never matches another channel's label file. `{name}` (the image stem) is available too.
- LUMIR rows skip registration, so no contrast-dependent template choice is made; SynthSeg and normalization treat both channels the same. build-target-grid still averages only `_0000`.

### Rerun the validation set (on a compute node or glacier; never on mlsc-login)

```bash
srun -A qtim -p rtx8000 --gpus 1 --time=02:00:00 --mem=32G --cpus-per-task=3 --ntasks-per-node=1 --pty bash
# (glacier instead: nvidia-smi, then export CUDA_VISIBLE_DEVICES=<one free GPU>)
PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
D=$PROJ/data; LUMIR=/autofs/space/crater_001/datasets/public/LUMIR2025; CFG=$D/preproc_config.yaml
RIC="bash $PROJ/brain-group-rad/preprocessing/run_in_container.sh"
cd $PROJ/brain-group-rad && git pull
cp preprocessing/config_example.yaml $CFG      # or just delete the old `qc: {png: ...}` key from $CFG

# 1. Target grid with the x flip (prints the corrected affine: x-row should start 1, 0, 0)
$RIC build-target-grid --config $CFG --lumir-root $LUMIR --split val --out-dir $D/target_grid --force-flip yes

# 2. Old validation outputs were stamped with the wrong affine; they are regenerable.
#    This also removes any LUMIR training rows already processed (same lumir folder);
#    they carry the wrong affine too, so that is intended. The old .npy export holds
#    float32 images and raw L/R-swapped SynthSeg labels, so it goes as well. The old
#    npy/template/ (template.npy float32, template_mask.npy) was sampled on the
#    unflipped grid; a stale float32 template.npy would also abort step 5.
rm -rf $D/processed/lumir $D/processed/_synthseg_batches $D/npy/lumir $D/npy/template

# 3. Manifest with both channels and the 10-class labels (check: _0001 rows have contrast unknown0001)
$RIC build-manifest --dataset lumir --root $LUMIR --out $D/manifests/lumir_val.csv \
    --image-glob 'imagesVal/LUMIRMRI_*_000[0-9].nii.gz' --label-glob 'simplelabelsVal/{id}*.nii.gz'
tail -n +2 $D/manifests/lumir_val.csv | cut -d, -f4 | sort | uniq -c   # T1w / unknown0001 counts
grep -c lumir10 $D/manifests/lumir_val.csv                              # rows with labels

# 4. Pipeline, SynthSeg, QC with the left/right check (expected: "left/right pattern OK", no WARNING;
#    roughly thalamus/pallidum frac_left_larger near 1, lateral ventricle ~0.8, hippocampus ~0.2-0.3)
$RIC run --config $CFG --manifest $D/manifests/lumir_val.csv --out-root $D/processed --threads 3
$RIC synthseg --config $CFG --manifest $D/manifests/lumir_val.csv --out-root $D/processed --threads 3
$RIC qc-summary --out-root $D/processed --dataset lumir --config $CFG --png sample:5
du -sh $D/processed/lumir; ls $D/processed/lumir/LUMIRMRI_3454_T1w/   # no work/, mask.nii.gz; qc.png only for the 5 sampled

# 5. Export (float16 image, uint8 remapped labels)
$RIC export-npy --out-root $D/processed --export-root $D/npy --dataset lumir --include-template --config $CFG
head -3 $D/npy/export_manifest.csv; du -sh $D/npy/lumir
exit
```

If export stops with `label values [...] are not in the fixed LUT (LUMIR25 coarse labels ...)`, the lumir10 ids aren't 0..10. Send me the values; `--no-remap-labels` works in the meantime.
