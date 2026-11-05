# Review FEAT-001 (iteration 1): exporter, synthfine labels, export-add-labels, add_labels_array.sh

Verdict: APPROVED (0 HIGH, 0 MEDIUM, 3 NIT).

## Evidence checked
- Diff of preprocessing/{cli,export,luts}.py, run_tests_wsl.sh, test_cli/test_export/test_slurm_scripts.py; new
  preprocessing/add_labels.py, preprocessing/slurm/add_labels_array.sh, preprocessing/tests/test_add_labels.py.
- impl-log.md FEAT-001 entry: full preprocessing runner 167 passed + "ALL CHECKS PASSED (bgr_preproc38)", targeted
  75 passed, manual CLI run on a 3-case temp tree (files 0-2 / 2-500, manifest, export-check OK, idempotent rerun).
  No TF code touched, so no goldens apply to this feature.
- Spot-check rerun by me (WSL bgr_preproc38): `pytest test_add_labels.py test_export.py test_slurm_scripts.py
  test_cli.py` -> 75 passed.

## Requirement checks
- luts.py: `LUTS["synthfine"]` = synthseg entries (luts.py:147); `synthfine_to_train18_table()` uint8[256], 103..255 = 0
  (luts.py:157-164); `synthfine_cortex_ids()` from the table (luts.py:167-171); docstring documents OD1 ids. numpy only.
- export.py: `SYNTHSEG_SETS`/`LABEL_SET_LEVELS` (export.py:124-128); `nested_block_mode` (export.py:193-229) follows design
  3.2: int64 bincount freed after int16 cast, parent counts int32, argmax ties -> smallest class / smallest fine id,
  validates integer/non-negative/<=102/divisible. Since train18_full == table[synthfine_full] and the class vote equals
  block_mode's smallest-id tie rule, table[nested] == train18_k; tested on random, fragmented-cortex and tie cases, and
  additionally verified per case before any write in phase files. synthseg keeps plain block_mode at 1,2
  (export.py:566-571). export-check consistency rule (export.py:737-748, 802-803).
- add_labels.py: phase files verifies every level against the existing train18 file before writing
  (add_labels.py:151-158), writes only missing/unreadable files atomically, never overwrites (mismatch/different ->
  failed), per-case try/except, status file named after the requested range. Range rule matches OD2
  (resolve_range, add_labels.py:66-76; `--start` without `--end` is not "start > end", which is the sensible reading).
  Phase manifest: pre-check by merging the unmodified parts in a temp root (merge_parts is a pure function of the parts,
  export.py:686-731, so the byte comparison is valid), re-check, backup, transform rows of manifest and parts (own column
  list kept), LUT added, merge reproduction, atomic writes, diff check, restore on any later exception
  (add_labels.py:391-461). Only `labels_l<k>` names change, so `data_module.manifest_files(..., label_set='synthfine')`
  (data_module.py:58-90) resolves the appended `;labels_synthfine<sfx>.npy` entries; the test's local reimplementation
  matches that code.
- cli.py: command registered, ValueError -> exit 2 (cli.py:456-462); --help loop and test_cli list extended.
- add_labels_array.sh: basic, qtim, no GPU (USE_GPU=0), 1 CPU, 8G, %2 in header and usage lines, PHASE/CHUNK validated
  before work, argv matches the spec; usage lines identical to plan.md FINAL REPORT (a). Real case (3423 rows, CHUNK 500)
  -> array 0-6, last chunk clipped.
- Hard requirements: feature is TF-free and does not touch the model, trainer, group sizes or validation (H1-H4 not
  affected); existing synthseg tests unchanged (only import lines and the script-name sets were extended); Python 3.8
  syntax (string-quoted OrderedDict annotations, typing generics).

## Findings

NIT 1 - add_labels.py:441-446 (step 5 writes). If the manifest phase is killed between the part writes and the
manifest write (e.g. SLURM time limit or node failure, not an exception), the restore does not run; the next run then
reports "pre-existing divergence ... (not caused by export-add-labels)", which is misleading although nothing is lost
(the backup exists). Suggest mentioning in that message that a `backup_pre_<set>_*` folder, if present, holds the
pre-run state.

NIT 2 - cli.py:458-462. `cmd_export_add_labels` maps every ValueError from `run_phase` to exit 2, so a corrupt
label_luts.json (json.JSONDecodeError, add_labels.py:397) in the manifest phase exits 2 ("usage") instead of 1. Narrow
the ValueError mapping to the usage checks (`_check_set`, `resolve_range`, phase/argument checks).

NIT 3 - export.py:539-541 with export.py:740. `export-npy --label-sets synthfine --label-dtype uint16` writes uint16
synthfine files that export-check then reports as wrong dtype; consider rejecting a non-uint8 `--label-dtype` for
synthfine up front (unlikely in practice: the default `auto` gives uint8).
