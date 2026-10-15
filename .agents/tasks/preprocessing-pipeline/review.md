# Manifest-driven MRI preprocessing pipeline onto the corrected LUMIR/MNI grid (pass 2)

This pass re-checks the untracked `preprocessing/` package against the six findings from pass 1. All six are fixed, and the fixes come with tests. Steps are now trusted only when their outputs exist and `meta.json` records them as `done`; a step is marked `running` before it starts. Every step record carries the argv lists that ran. The SLURM templates hard-code `--account=qtim` and comment out the QOS line. `use_masks` without a template mask now warns. Generic globs substitute the raw folder name. The recorded test evidence (`test-log.txt`: `bash -n OK`, 91 passed, `ALL CHECKS PASSED`, run after the last source edit) covers the new behaviour. Nothing blocking remains.

Watch for: `refresh_qc` rewrites `qc.json` in place from the SynthSeg batch path without resetting the `qc` step record, so a kill mid-write leaves a corrupt `qc.json` that `qc-summary` will fail on (likely, low severity). The fake `flirt` shares the FSL coordinate convention with the production code, so only a cluster run can check the matrix composition (carried over from pass 1, informational).

**Verdict**: APPROVED

## High-level view

The resumability gap is closed. `RowContext.is_done` requires `meta.steps[s].status == "done"` on top of the outputs existing. `process_row` writes a `running` record to disk before each step and a `done` record after it, and a failing step is marked `failed` along with the commands it attempted. The SynthSeg batch applies the same rule when it selects pending subjects. `test_pipeline` simulates a SLURM kill during N4 (a truncated gzip with the record left at `running`) and checks that the rerun redoes the step.

Each tool call is recorded through `RowContext.run_tool` into `commands`, which feeds `record_step`. A test compares the recorded argv with the fake-tool call log for strip, n4, register and resample, and checks that normalize records no commands.

The SLURM templates now encode the workspace rule (`#SBATCH --account=qtim`, with `--qos` as an inert `##SBATCH` line), and the test enforces both. The ants `use_masks` mismatch logs a warning and records `masks_used` in meta. The generic builder sanitizes only the manifest `subject` column, and a test covers a folder named `subj 01 (scan)`.

One small durability gap is left. The SynthSeg batch path rewrites `qc.json` and `qc.png` without the running/done protocol.

<details>
<summary>Issues (2)</summary>

1. **Non-atomic qc.json rewrite in the SynthSeg batch (non-blocking)**: `synthseg.refresh_qc` writes `qc.json` straight to its final path, and the batch path never resets the `qc` step record. A kill mid-write leaves a truncated `qc.json` that is still trusted, and `qc.summarize` then fails on `json.load`. Write through a temp file plus `os.replace` (as `write_meta_atomic` does).
2. **FSL convention only self-consistent in tests (informational)**: `tests/fake_impl.py` uses the same FSL scaled-voxel convention as `nifti.flirt_to_world`, so a convention error would not show up in tests. Check the first real flirt row on the cluster with a freeview overlay of `image.nii.gz` on `template_lumirgrid.nii.gz`.

</details>

<details>
<summary>Details</summary>

### Running/done protocol for resumable steps

```
step s:  record(s, running) -> write meta -> STEP_FUNCS[s] -> record(s, done, commands, params) -> write meta
rerun :  is_done(s) = applies? -> outputs declared? -> recorded done? -> outputs exist? (or work_cleaned + normalize done)
```

`synthseg` stays outside the row-level protocol because `run_synthseg_batch` records its own `done`/`failed` entries per subject. That path covers the label file, but not the QC rewrite that comes after it. `refresh_qc` writes `qc.json` with a plain `open(..., "w")`, and the batch path never marks `qc` as running. A kill there leaves `steps.qc.status == "done"` next to a truncated `qc.json`. `qc.summarize` reads it with an unguarded `json.load`, so `qc-summary` for the whole dataset fails until that subject's QC is regenerated (likely; follows from `synthseg.py` `refresh_qc` and `qc.py` `summarize`, not observed). The window is short (writing a small JSON file), which is why this is non-blocking.

### Test evidence

`test-log.txt` was written after the last source edit (the last edits were at 19:53–19:54 and the log starts at 19:56). It shows 91 passed and `ALL CHECKS PASSED`. New tests cover the kill-during-N4 rerun, the failed-step command record, matching recorded commands against the fake call log, the `use_masks` warning (`-x` absent), the raw-name generic globs, and the `qtim`/QOS template rules. Not tested: an interrupted SynthSeg-batch `qc.json` rewrite (issue 1), and the FSL convention against a real FLIRT (issue 2).

</details>

<details>
<summary>File map</summary>

- `preprocessing/pipeline.py`: running/done step protocol, `is_done` requires the recorded status, per-step `commands`, failed-step record, `use_masks` warning and `masks_used`
- `preprocessing/meta.py`: `record_step(meta, name, status, outputs, commands, params, started, log)`
- `preprocessing/synthseg.py`: pending selection also requires the recorded `synthseg: done`
- `preprocessing/builders/generic.py`: globs use the raw subject name, and only the manifest column is sanitized
- `preprocessing/slurm/*.sh`: `--account=qtim`, `##SBATCH --qos` placeholder
- `preprocessing/tests/{test_pipeline,test_builders,test_slurm_scripts}.py`: tests for the above
- Full change set: `git status` (all untracked) under `preprocessing/`

</details>
