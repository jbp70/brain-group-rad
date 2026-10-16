# Storage trimming, LUT-remapped export, LUMIR channel handling and left/right safety check (pass 3)

This pass checks the fix for the one pass-2 finding: the rerun recipe left a stale `$D/npy/template/`. Step 2 of the "Rerun the validation set" block now runs `rm -rf $D/processed/lumir $D/processed/_synthseg_batches $D/npy/lumir $D/npy/template`. Its comment explains that the old template arrays are float32 and sampled on the unflipped grid. With that folder gone, step 5 writes a fresh float16 `template.npy` and a matching `template_mask.npy`, followed by the manifest and `label_luts.json`. No code changed after pass 2. The latest source edit is test_export.py at 10:22:06, and final_report.md was last edited at 10:29:06. The test log is dated 10:32:35, after both. It shows 118/118 passing in the Python 3.8.20 env that matches the container (numpy 1.21.1, scipy 1.4.1, nibabel 3.1.1, PyYAML 6.0, matplotlib 3.5.0) and in the 3.9 env.

Watch for: nothing blocking. Some things are still unverified because they only exist on the cluster: the real mri_synthseg vols CSV column names, the actual lumir10 label ids, and the current contents of `$D/npy`. If the lumir10 ids turn out not to be 0..10, the export fails loudly and the report says what to do.

**Verdict**: APPROVED

## High-level view

Pass 2 covered the full diff and found it met requirements 1–4. Nothing has changed there since. Work cleanup runs only for fully done rows, after the labels are written and verified. `work_cleaned` keeps resume working. Mask and QC-PNG are optional and derived on demand. `cleanup` is limited to `<ds>/<subject>/{work,mask.nii.gz,qc.png}`. It checks realpath containment, never follows symlinks, rejects any dataset that isn't a plain folder name, and refuses rows that aren't done. The export defaults to float16 images and fixed LUTs. The synthseg LUT has 103 ids and fits uint8. Unknown values raise. The LUTs are written to `label_luts.json` and recorded in the manifest. LUMIR `_0001` rows get `unknown0001` and keep the suffix in their subject id. The L/R asymmetry check warns when the majority pattern is reversed in datasets with at least 20 cases.

The report now meets requirement 5. Both build-target-grid examples use `--force-flip yes`. The dated section has the L/R evidence, the new defaults, the cleanup and qc-png commands, the export sizes, and a complete rerun recipe with the PROJ/D/LUMIR/CFG paths. The recipe now also clears every artifact that depends on the flip. No new markdown files were added outside the task's review/report files.

A spot-check of `cleanup.py` confirmed the deletion guards again. It also confirmed there is no 3.9+ syntax: variable annotations and `typing` generics are valid in 3.8.

<details>
<summary>Issues (0)</summary>

No blocking or non-blocking action items.

</details>

<details>
<summary>File map</summary>

- `preprocessing/cleanup.py` (new): guarded deletion of work/, mask.nii.gz and qc.png, with dry run and refusal.
- `preprocessing/luts.py` (new): fixed synthseg/lumir10/seg4/seg35 LUTs; `apply_lut` fails loudly on unknown values.
- `preprocessing/export.py`: float16/LUT defaults, `--include-mask`, label_luts.json, manifest columns, dtype/shape guard on kept files.
- `preprocessing/pipeline.py`: `outputs.*` handling, optional mask, work cleanup at row end, `work_cleaned` resume.
- `preprocessing/synthseg.py`: label verify, `running` marker, per-row work cleanup, `refresh_qc` without mask.
- `preprocessing/qc.py`: `load_mask`, `render_qc_png`, L/R asymmetry check, `--png` selection, qc_summary.json.
- `preprocessing/cli.py`: `cleanup`, `--keep-work`, qc-summary `--png`/`--lr-*`, export flags.
- `preprocessing/builders/lumir.py`: channel-suffix contrast/subject, images*-only glob, `{name}` placeholder.
- `preprocessing/config.py`, `config_example.yaml`: `outputs` block, error on the legacy `qc.png` key.
- `preprocessing/grid.py`, `__init__.py`, `slurm/qc_export.sh`: docs, float16 default.
- `preprocessing/tests/*`: new and extended tests (118 total).
- `.agents/tasks/preprocessing-pipeline/final_report.md`: `--force-flip yes`, dated section, rerun step 2 now also removes `$D/npy/template`.

Full diff: `git diff` in the repo root plus untracked `cleanup.py`, `luts.py`, `tests/test_cleanup.py`.

</details>
