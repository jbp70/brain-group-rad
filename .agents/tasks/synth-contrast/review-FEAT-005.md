# Review FEAT-005 (post-training eval_synth), iteration 1

Verdict: APPROVED (0 HIGH, 0 MEDIUM, 2 NIT).

## Evidence checked
- Requirements: FEAT-005.json steps 1-7, design.md section 8, plan.md D-e / OD1.
- Hooks: `runtime/fixed_data.py` `ValidationSet.build` 195/240-241, `build_validation_sets` 260-281, `DeformValidationSet.build` 448-476. With `group_images_fn=None` the code is the old expression (`images[idx]`), and no RNG draw is added. The hook only replaces the images; labels stay train18 (472-476). This holds H2/H4: in-training validation is unchanged.
- Presets: `data_loading/synth.py` 313-336 has the 11 names. `preset_params` 369-425 returns the `draw_synth_params` keys. random_* = `draw_synth_params(prob=1)`, so hier_dev 0.1 / cortex 0.25 (OD1). PET parcel overrides go through `parcel_fine_ids` (339-347). `luts.DK_PARCELS` includes `corpuscallosum`, so index+1 matches the FreeSurfer 1000+k / 2000+k ids. Seeds are deterministic: crc32(name) and seed (349-352). The fine labels only feed `tf.gather` tables, with no one-hot.
- `runtime/eval_run.py`:
  - Config rebuild with int keys: 66-108.
  - The model and frozen stages are built as in the resume branch of `main.py` 288-313 / 384-427: same constructor kwargs, `FrozenChain` / `FrozenStage.from_config`, `checkpoint_extra`, fingerprint checks through `M._check_cascade_fingerprint(s)`, BN freeze + `rebuild_steps`.
  - Restore: `checkpoint.restore(prefix).assert_existing_objects_matched()` (eval_run 196-199), identical to `RunCheckpoint._restore`. No save is called. `Trainer.validate` (run.py 1727-) has no checkpoint/log side effects (best saving lives only in `_validate_and_log`, 1946-1961), and val PNGs are off.
  - Real sets reproduce `main._build_dali_data` 192-211: same nv rule, ranges, val_seed, G, n_groups / dry_run. Linear order is sorted, but seeding depends only on (seed, task, g).
  - Gate: reads `{"step","data"}` records of metrics.jsonl (runtime/logging.py 80). Tolerances 1e-2 mm / 1e-3 Dice; an empty comparison fails. Exit 5 writes the JSON with no preset run (eval_run 429-432).
  - Synthetic sets: built under `/cpu:0` once per spec and shared by both trainers (`trainer.val_sets = sets`). Names are `synth_<preset>_<set>`, so the `_identity_cache` / `_frozen_cache` keys (run.py 1594/1619) are per preset. Each task's `_g` set shares its plain set's contrast through the memo. sha256 is computed per set.
- `eval_synth.py`: no TF at import or `--help`. `--out` inside a run dir is rejected; exit codes 0/2/5. `slurm/eval_synth.sh`: -A qtim, rtx8000, 1 GPU, 3 CPUs, 128G, 8 h, the usage line and "DO NOT submit" note, a dry run before any write, and sstat appended. `scripts/compare_synth_eval.py` is stdlib only, and a sha mismatch names the preset and exits 1.
- Python 3.8: no 3.9+ syntax found; ast 3.8 check logged.
- Tests were run (impl-log.md "FEAT-005"):
  - tf_gpu `tests.test_eval_synth`: 14 OK, incl. e2e exit 0, shared sha, CSV, run dir unchanged (listing + mtimes), and failed gate -> exit 5.
  - TestLinearOnlyUnchanged / TestDeformUnchanged / TestResumeAndValidation: 7 OK, goldens max |diff| 0 / 0. TestMixedTasks: 2 OK.
  - bgr_preproc38: py_compile + ast 3.8, `eval_synth.py --help` without TF, bash -n, dry run, `tests.test_slurm_scripts` 46 OK.
- FEAT-005 does not touch the labels keyword-only / cascade-skip guard tests.

## Findings
- NIT 1: Only the failing gate path is tested. No local test feeds a run whose metrics.jsonl holds a comparable validation at the ckpt/last step (e.g. a tiny real-export run with `--val-every` equal to the step count). The first PASS of `logged_validation` / `compare_validation` on real logged keys will therefore happen in the cluster job. Optional: add such a tiny test before the cluster eval.
- NIT 2: The cascade restore path (level 2 / 1: `FrozenStage` / `FrozenChain.from_config`, `checkpoint_extra`, fingerprint check, BN freeze) is unexercised locally. The intended L1 job will be its first run, as the coder recorded in the FEAT findings. The code mirrors main.py's resume branch line by line, so I am not reporting it as a defect. A tiny L2 cascade eval test would catch startup errors before an 8 h rtx8000 job is spent.
