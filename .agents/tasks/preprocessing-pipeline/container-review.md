# Singularity container runtime for the preprocessing pipeline

The pipeline now runs on Martinos machines inside `tf10_nv.sif` instead of a conda env. A new `run_in_container.sh` builds the `singularity exec` argv (binds, optional `--nv`, tool env vars, `TMPDIR`) and runs `python -m preprocessing "$@"` through a fixed `bash -c` script. The four SLURM templates call that wrapper, and the conda/`PY_ENV_ACTIVATE` lines are gone. Config defaults now use the real `/autofs/cluster/pubsw` FSL/ANTs paths and the FreeSurfer dev build. A new `tmp_dir` key routes tool temp files away from an unbound host `TMPDIR`. The Python code needed no changes for 3.8 or the old libraries. A 102-test suite passed in both a 3.8/container-pinned env and a 3.9/newer-libs env after the last edit.

Watch for: in `USE_GPU=auto` on glacier, `--nv` exposes all three GPUs unless `CUDA_VISIBLE_DEVICES` is set, and TF in `mri_synthseg` will claim memory on every visible GPU (likely). The templates `cd` to the repo before resolving `MANIFEST`, so the relative `M.csv` in the usage comments points at a different file than the login-node `N=` computation (confirmed). Neither blocks the change.

**Verdict**: APPROVED

## High-level view

The wrapper keeps user input out of the shell script. Every user value travels as an argv entry (`env VAR=value`, then `bash -c '<constant>' run_in_container "$@"`), and the test confirms that a manifest name full of `;`, `$(...)`, backticks and quotes reaches Python unchanged without running anything. The default binds match the rules, including `/usr/local/freesurfer`. Binds missing on the host are skipped with a warning. SIF, repo, TMPDIR, tool homes and the runtime binary can each be overridden by an env var.

Temp files are handled in two layers. The wrapper exports `TMPDIR` as `/scratch/$SLURM_JOB_ID` on SLURM nodes, or as the project's `tmp/` elsewhere. On top of that, `resolve_tmp_dir` exports `TMPDIR`/`TMP`/`TEMP` to every tool. It uses the configured `tmp_dir`, then `$TMPDIR` only if that folder exists and is writable, then `<out-root>/_tmp`. The package itself never calls `tempfile`.

The GPU gate is `USE_GPU=1/0/auto`. CPU templates force `USE_GPU=0` and SynthSeg forces it to 1, so the cluster path is deterministic. Only interactive glacier use relies on `auto`, and that is where the one-GPU rule depends on the user setting `CUDA_VISIBLE_DEVICES`.

The SLURM templates satisfy the stated limits. All use `--account=qtim`, one node and one task, and `--gpus=1` on the SynthSeg job only (3 CPUs and 32G on rtx8000). CPU jobs stay well under 15 GB per core, arrays default to `%2`, logs go to absolute project-folder paths, and partitions remain `PLACEHOLDER_*` with suggested defaults in comments.

<details>
<summary>Issues (3)</summary>

1. **Glacier multi-GPU exposure in auto mode** (likely, non-blocking): with `USE_GPU` unset and `CUDA_VISIBLE_DEVICES` unset, `--nv` exposes all GPUs and TF in `mri_synthseg` allocates on all of them. In the wrapper, refuse `--nv` (or warn loudly) when `nvidia-smi -L` lists more than one GPU and `CUDA_VISIBLE_DEVICES` is empty.
2. **Relative MANIFEST resolves against REPO_DIR** (confirmed, non-blocking): `run_array.sh` and `synthseg_batch.sh` `cd "$REPO_DIR"` before passing `--manifest "$MANIFEST"`, but the usage comments compute `N` from `M.csv` in the submit directory. Either say the path must be absolute or resolve `MANIFEST` with `realpath` before the `cd`.
3. **mri_synthseg untested inside the container** (possible, already disclosed): FreeSurfer's bundled TF/model may conflict with the image or the GPU type. The final report already flags this. Jay's one-subject smoke test is the check, and no code change is needed now.

</details>

<details>
<summary>Details</summary>

### Argument passing and argv shape in run_in_container.sh

The `inner` script is single-quoted and refers only to env vars set through `env NAME=value` argv entries, so host values such as `/fs home` or the work dir never become shell text. The test asserts the full argv against the expected list and checks that none of the user values appear in `inner`. There is no `eval`, and the defaults test asserts that too. `set -euo pipefail` and the `${binds[@]+...}` guard keep an empty `BGR_BINDS` working under `set -u`. The script exits with status 2 on missing args or a missing image before calling singularity.

`BGR_WORKDIR=$PWD` lets relative paths resolve inside the container, falling back to the repo with a warning. That interacts with the templates' `cd "$REPO_DIR"` (Issue 2):

```
login node:  cd ~/somewhere; N=$(tail -n +2 M.csv | wc -l ...)    # ~/somewhere/M.csv
job:         cd $REPO_DIR; run --manifest M.csv                     # $REPO_DIR/M.csv
```

### --nv gating and the glacier one-GPU rule

`want_gpu` adds `--nv` only when `USE_GPU` is truthy, or in auto mode when `nvidia-smi -L` lists a GPU. On glacier, running `synthseg` without `CUDA_VISIBLE_DEVICES` exposes all three A100s, and TF's default allocator maps memory on every visible device. The header comment says to pin one GPU, but nothing enforces it. A cheap guard (warn or refuse when more than one GPU is visible and `CUDA_VISIBLE_DEVICES` is empty) would turn a workspace rule into a code check. Cluster jobs are unaffected because SLURM sets `CUDA_VISIBLE_DEVICES` for `--gpus=1`.

### Python 3.8 and old-library compatibility

A grep for PEP 585/604 runtime generics, `removeprefix`/`removesuffix`, `functools.cache`, `math.lcm`, `zoneinfo`, `BooleanOptionalAction`, `importlib.resources`, `ndimage.value_indices` and `np.quantile(method=)` found nothing. All annotations use `typing.List`/`Dict`/`Optional`/`Tuple`, and `shlex.join` is 3.8+. The 3.8 env is CPython 3.8.20, not the container's 3.8.10. All library pins match the container exactly (numpy 1.21.1, scipy 1.4.1, nibabel 3.1.1, PyYAML 6.0, matplotlib 3.5.0). The 140 warnings in that run come from inside scipy/nibabel, not from `preprocessing/`.

### Test evidence

`container-test-log.txt` (written 21:31) is newer than the last source edit (`test_tools.py`, 21:23). It records 102 passed, 0 failed, 0 skipped in `bgr_preproc38`, and 102 passed in `bgr_preproc` (py3.9.18, numpy 1.26.4, nibabel 5.2.1). Both runs also passed `bash -n` on all six shell scripts and a `--help` smoke test of all 8 subcommands. The required new tests exist. `test_wrapper_passes_args_verbatim` uses a fake `singularity` on PATH, and `test_tools_temp_files_go_to_work_dir_not_tmpdir` runs `run` and `synthseg` with `TMPDIR` pointed at a nonexistent path. `test_slurm_scripts.py` statically checks qtim, no conda/eval, `--gpus=1` with ≤3 CPUs and ≤128G, ≤15 GB per core and `%2`.

Not tested: real `singularity exec` (bind visibility, `--nv` with the actual image), `mri_synthseg` inside the container, and FLIRT/ANTs at the `/autofs/cluster/pubsw` paths inside the container. These need a cluster run, which is Jay's to do under the data-handling rule.

</details>

<details>
<summary>File map</summary>

- `preprocessing/run_in_container.sh`: new; builds the singularity argv and sets binds, `--nv`, tool env and TMPDIR.
- `preprocessing/slurm/{run_array,synthseg_batch,build_target_grid,qc_export}.sh`: call the wrapper, set qtim, fair resources, `%2`, absolute log paths; conda lines removed.
- `preprocessing/config.py`, `config_example.yaml`: Martinos FSL/ANTs/FreeSurfer paths, new `tmp_dir` key and validation.
- `preprocessing/tools.py`: `ToolRunner(tmp_dir=)` exports TMPDIR/TMP/TEMP; `resolve_tmp_dir`; dev build stamp in check-tools.
- `preprocessing/cli.py`, `__init__.py`: wire `resolve_tmp_dir` into `run`/`synthseg`; docs.
- `preprocessing/requirements*.txt`: container pins; separate py3.9 test pins.
- `preprocessing/tests/test_container.py` (new), `test_config.py`, `test_tools.py`, `test_slurm_scripts.py`, `fake_impl.py`, `conftest.py`, `run_tests_wsl.sh`: wrapper argv/GPU/error tests, temp-dir tests, template checks.

The pipeline is untracked, so there is no `git diff`; files dated 2026-10-14 21:xx are this iteration's changes.

</details>
