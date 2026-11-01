# Local environment: nibabel in tf_gpu (predict-script design 10.0, FEAT-001)

On 2026-11-01 the local WSL env `tf_gpu` (python 3.9, TF 2.10) had no nibabel, and its numpy is
1.24.3, not the 1.23.4 the design assumed. nibabel 3.1.1 (the container version) does not import
under numpy 1.24: `nibabel/quaternions.py` uses `np.float` at module level, an alias numpy 1.24
removed (tested from a copy of bgr_preproc38's 3.1.1 on PYTHONPATH, nothing installed). nibabel
3.2.2, the oldest release without that module-level alias, imports and round-trips a float32
`.nii.gz` exactly under numpy 1.24.3. pip inside WSL has no network (pypi.org unreachable), so the
wheel was downloaded on Windows (`nibabel-3.2.2-py3-none-any.whl`, sha256
7df7a2733461441d3aacc61f36f5e100ec533d43ed09a191293bb4ca5a4f10f6; only dependency `packaging`,
24.2, already present). With Jay's approval it was installed with `conda activate tf_gpu && pip
install --no-deps .agents/tmp/nibwhl/nibabel-3.2.2-py3-none-any.whl`; the probe
`python -c "import nibabel, numpy; print(nibabel.__version__, numpy.__version__)"` prints
`3.2.2 1.24.3`. numpy was not changed. Deviation from the design: nibabel 3.2.2 locally vs 3.1.1
in tf10_nv.sif; the prediction code keeps to nibabel 3.1 APIs (e.g. `Nifti1Image`, `dataobj`,
`nibabel.openers.Opener.default_compresslevel`). `tests/run_tests_wsl.sh` stage 2 now prints one
clear line and sets status 1 if nibabel is missing.
