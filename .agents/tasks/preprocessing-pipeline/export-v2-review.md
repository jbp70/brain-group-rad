# Per-level training export (npy_v2) with train18 labels, chunked SLURM export and merge

`export-npy` now writes the coarse-to-fine training layout from design doc sections 3.1 and 3.4. Each case gets full, `_2` and `_4` float32 images (exact block averages) and uint8 `train18` label maps (merged LUT first, then block majority vote). Per-level affines and world centers go into the manifest. The export can be chunked over `manifest_processed.csv` rows: each run writes its own manifest part, and a new `export-merge` step combines them. Writes are atomic and reruns resume. A new `export-check` command and a `slurm/export_array.sh` template (basic / qtim / `%2`) are added, and the CLI and `qc_export.sh` default to float32 and `$PROJ/data/npy_v2`. The test log postdates the last code edit and shows 136 passing tests in both `bgr_preproc38` (3.8 / numpy 1.21.1 / scipy 1.4.1 / nibabel 3.1.1) and `bgr_preproc`.

Watch for: the `export_array.sh` usage header queues the array and the dependent merge job together, which can exceed the two-active-jobs rule (likely). A non-chunked export started while array chunks are still running auto-merges whatever parts exist at that moment (likely). The 103-ID SynthSeg map is opt-in only and isn't part of the documented val command (confirmed; non-blocking).

**Verdict**: APPROVED

## High-level view

Pooling and geometry match the spec. Coarse images are `reshape(x,k,y,k,z,k).mean(dtype=float64)` over the full-res float32 NIfTI, and non-divisible shapes fail. Labels are merged with the fixed `train18` LUT and then reduced by a vectorized bincount mode, where `argmax` picks the smallest id on ties. The level-k affine is A_1 · [k·I, (k−1)/2], so coarse voxel q sits at fine index 2q+0.5 or 4q+1.5. The center at index (n_k−1)/2 maps to the same world point at every level. `export-check` re-verifies this on real exports.

The `train18` LUT covers all 103 SynthSeg `--parc` ids: 13 L/R pairs, 4 midline structures, background, and cortex (3/42 plus 1001–1035 and 2001–2035). Unknown ids fail with the id list and the case name. The LUT, names and classes land in `label_luts.json` through the merge. The contrast filter defaults to T1w, which drops the `unknown0001` val cases, and skip counts are logged per run.

Chunk safety comes from per-selection manifest parts with deterministic names, plus a merge that dedupes identical rows and rejects conflicting ones. `.npy` files go through temp-then-`os.replace` in the same folder. Resume skips a case only when every planned file exists with the right dtype and shape. A mismatch fails the run, and `--overwrite` rewrites everything. The remaining gaps are in how jobs are sequenced, not in the core export logic.

<details>
<summary>Issues (4)</summary>

1. **Usage header exceeds the two-job rule** — `export_array.sh` header submits the array and the `afterok` merge job back to back, giving up to 2 running tasks plus a pending merge (likely >2 active entries in `squeue -u jn85`). Change the header to the sequenced form already used in final_report.md (submit the merge after the array finishes).
2. **Auto-merge races in-progress chunks** — any `export-npy` run without a row selection (e.g. `qc_export.sh DO_EXPORT=1`) merges whatever parts exist at that moment, so running it during an array can write an incomplete `export_manifest.csv` (likely). Document "don't run an unchunked export or export-merge while the array is running", or skip auto-merge when parts are newer than N minutes.
3. **103-ID SynthSeg labels not in the val command** — design 3.1 says the full-res 103-ID labels are kept for evaluation, but `synthseg` is opt-in and the documented val export uses `train18,lumir10` (confirmed). The NIfTI stays in `processed/`. Either add `synthseg` to the val command or note in the design doc that evaluation reads the NIfTI.
4. **Stale parts are merged silently** — `merge_parts` unions every `manifest_parts/*.csv`. A leftover part from a run with different filters (e.g. `--contrast any`) would add non-T1 rows without a conflict error (possible: only after a manual rerun with other options). Log the part names and per-part row counts at merge time so this is visible.

</details>

<details>
<summary>Details</summary>

### Chunk parts, merge and job sequencing

```
array task i ──► manifest_parts/lumir_split-train_contrast-T1w_rows-<s>-<e>.csv (+ .luts.json)
val task     ──► manifest_parts/lumir_split-val_contrast-T1w_rows-0-100000.csv
export-merge ──► export_manifest.csv + label_luts.json   (union, dedupe, conflict = error)
unchunked export-npy ──► its own part, then merges ALL parts immediately
```

Concurrent chunks never write the same file. Only task 0 writes the template, and `template_info.json` is written atomically. The weak point is the last line of the diagram. `merge_parts` runs at the end of every unselected run and reads only the parts that exist at that moment. A `qc_export.sh DO_EXPORT=1` job run during the array would publish a partial manifest, and a later `export-merge` fixes it only if someone remembers to rerun it (likely). The same union semantics mean a stale part from a differently filtered run gets merged silently. The conflict check fires only when the same case differs (possible).

The `export_array.sh` header submits the array (`%2`, so up to two running tasks) and immediately submits a dependent merge. That is three active entries for `squeue -u jn85`, against the "at most two jobs active, running plus pending" rule (likely). final_report.md already sequences these steps correctly (train array → val → merge), so the fix is to align the header with it.

### Label sets for evaluation

Defaults match Jay's decisions: `train18` at every level, `lumir10` full resolution for val, and T1w only, so the export should hold 3,384 train + 39 val cases. Design 3.1 also says "Full-resolution 103-ID SynthSeg labels are kept for evaluation". The code supports this through `--label-sets ...,synthseg`, but no documented command exports it (confirmed). It isn't blocking, because `labels_synthseg.nii.gz` stays in `processed/`. Evaluation code will need to know which source to read.

### Test coverage and evidence

The log (11:50) postdates the last edit (`test_slurm_scripts.py`, 11:34). It shows both envs with 136 passed, `bash -n` on all scripts, and `--help` for the new commands. The coverage includes exact dyadic block averages, mode against brute force including ties and uint16 sparse ids, every `train18` id, the unknown-id error, affines and world centers, filters with skip counts, chunk parts with merge, dedupe and conflict handling, a simulated kill during write, the resume/mismatch/overwrite paths, the per-level template, `export-check`, and SLURM policy assertions. Not tested: concurrent merge during in-progress chunks, a real cluster run, and the real LUMIR SynthSeg id set (the report flags the last two).

</details>

<details>
<summary>File map</summary>

- `preprocessing/export.py`: per-level export, block average and mode, level affines, row selection, manifest parts, `merge_parts`, `export_check`, atomic writes and resume.
- `preprocessing/luts.py`: `train18` merged LUT, `n_classes`, `classes` in the JSON.
- `preprocessing/cli.py`: new export-npy flags, float32 default, `export-merge` and `export-check` commands.
- `preprocessing/slurm/export_array.sh` (new): chunked CPU array on basic/qtim with `%2`, plus `MERGE_ONLY`.
- `preprocessing/slurm/qc_export.sh`: partition basic, float32, `npy_v2`, `LABEL_SETS`/`SPLIT`.
- `preprocessing/__init__.py`: docstring example.
- `preprocessing/tests/test_export.py`, `test_slurm_scripts.py`, `test_cli.py`, `run_tests_wsl.sh`: new and updated tests, plus help smoke tests for the new commands.
- `.agents/tasks/preprocessing-pipeline/final_report.md`: "Training export v2" section with Jay's commands.

Full diff: `git diff` in the repo root (uncommitted), plus the untracked `preprocessing/slurm/export_array.sh`.

</details>
