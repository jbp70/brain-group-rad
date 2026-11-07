# FEAT-004 test log (compare_tta.py, per-subject label file, SLURM wrappers, test-runner lists)

Date 2026-11-07. Working tree on main (FEAT-001..003 uncommitted, untouched). No commits, no pushes, no cluster jobs.

## Files changed

- `evaluation/evaluate.py`: additive `per_subject_label.csv` (`subject`, `template_dice_<label>` per label of the label list, from `per_label[("template", i)]`); module docstring Outputs line. No other output changes.
- `tests/test_evaluation.py`: `EndToEnd.test_method_exact_fields` asserts the file (4 rows, exact columns, same subjects, row NaN-mean over labels == per_subject.csv template_dice within 1e-6; `M.dice_summary` confirmed to be the NaN-mean over labels).
- `scripts/compare_tta.py` (new; numpy + scipy, TF-free, Python 3.8).
- `tests/test_compare_tta.py` (new; TF-free, 12 tests).
- `slurm/predict_mindboggle.sh`: env `TTA` (none), `TTA_LEVELS` (empty), `SKIP_CHECKS` (empty) + docs; `TTA_FLAGS` on step 2 only; checks 1a-1c wrapped in `if [ "$SKIP_CHECKS" != "1" ]; then ... else echo "checks 1a-1c skipped (...)"; fi` with the singularity lines unchanged at column 0; header echo shows tta.
- `slurm/tta_mindboggle.sh` (new), `slurm/compare_tta.sh` (new).
- `tests/test_slurm_scripts.py`: `test_predict_script` updated (new run-line ending, three env defaults, SKIP_CHECKS message, no TTA flags on check-streaming lines); new `TestStubRuns.test_predict_tta_flags` (stub run: default = today's calls + `--tta none` on step 2; TTA=flip8 TTA_LEVELS=1 SKIP_CHECKS=1 = no checks, `--tta flip8 --tta-levels 1`); new class `TestTTAScripts` (exact #SBATCH lines of both scripts, policy lines, both arms, linear-eval loop, compare calls, printed eval/compare lines, DRY_RUN=1 subprocess run: 2 arms + 4 linear-eval + 3 compare commands, nothing created; TTA_LEVELS / PARTS; invalid settings exit 2); `test_bash_syntax` covers the two new scripts and predict_mindboggle_multi.sh; `TTA`, `TTA_LEVELS`, `SKIP_CHECKS` added to `RECIPE_ENV`.
- `tests/run_tests_wsl.sh`: PYFILES += scripts/compare_tta.py; bash -n list += slurm/predict_mindboggle_multi.sh slurm/tta_mindboggle.sh slurm/compare_tta.sh; TF-free list += tests.test_compare_tta.

## Verification (FEAT-004.json commands)

1. `wsl -d Ubuntu bash .../.agents/tasks/tta/run_tests.sh py38 tests.test_compare_tta tests.test_evaluation tests.test_slurm_scripts`
   -> `Ran 92 tests ... OK`, `TF-free tests OK without TF (92 tests)` (bgr_preproc38, scipy 1.4.1). First run had 2 failures in my new tests (a wrong Holm expectation [0.5, 0.9] -> correct value [1.0, 1.0]; a header comment line matched the linear-eval line filter); both fixed, rerun OK.
2. `run_tests.sh bashn slurm/predict_mindboggle.sh slurm/tta_mindboggle.sh slurm/compare_tta.sh` -> all three `bash -n ... OK`, rc 0.
3. `DRY_RUN=1 NAME=x CKPT=/tmp/x bash slurm/tta_mindboggle.sh` -> rc 0, prints both arms, 4 linear-eval and 3 compare commands, the 2 all-pairs lines and the final compare line; runs nothing. Excerpt ($P = /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg; long paths shortened):
   ```
   DRY: env NAME=x_tta_none OUT=$P/results/predict_mindboggle/x_tta_none TTA=none TTA_LEVELS= SKIP_CHECKS= CKPT=/tmp/x PROCESSED=... GRID_REF=... REPO=... SIF=... bash $P/brain-group-rad/slurm/predict_mindboggle.sh
   DRY: env NAME=x_tta_flip8 OUT=$P/results/predict_mindboggle/x_tta_flip8 TTA=flip8 TTA_LEVELS= SKIP_CHECKS=1 CKPT=/tmp/x ... bash .../predict_mindboggle.sh
   DRY: singularity exec --nv ... predict.py linear-eval --checkpoint /tmp/x ... --task rigid --seed 0 --tta none --out $P/results/linear_eval_mindboggle/x_rigid_tta_none
   ... (rigid flip8, affine none, affine flip8)
   DRY: singularity exec ... scripts/compare_tta.py --a .../x_tta_none/eval_template --b .../x_tta_flip8/eval_template --out $P/results/tta_compare/x_template --kind eval --label-a none --label-b flip8
   DRY: ... compare_tta.py --a .../x_rigid_tta_none --b .../x_rigid_tta_flip8 --out $P/results/tta_compare/x_rigid_linear --kind linear ...
   DRY: ... (affine)
     METHOD=x_tta_none FWD='.../x_tta_none/fields/{subject_dir}_fwd.nii.gz' INV='...' ATLAS=... ATLAS_SEG=... CONV="--components xyz --signs +++ --units mm" sbatch slurm/eval_mindboggle.sh
     METHOD=x_tta_flip8 ... sbatch slurm/eval_mindboggle.sh
     A=$P/results/eval_mindboggle/x_tta_none B=$P/results/eval_mindboggle/x_tta_flip8 OUT=$P/results/tta_compare/x_allpairs sbatch slurm/compare_tta.sh
   finished tta_mindboggle: 0 failed part(s)
   ```
4. `run_tests.sh compile` -> `py_compile OK`, `ast feature_version=(3, 8) OK (23 files)` (incl. scripts/compare_tta.py, tests/test_compare_tta.py), `inference.tta imports without TF`, rc 0.

Extra: `python -m unittest tests.test_compare_tta` in tf_gpu (scipy 1.13.0, numpy 1.24.3; there `scipy.stats.binom_test` no longer exists, so the `binomtest` fallback ran) -> 12 tests OK. This matches stage 2 of tests/run_tests_wsl.sh, which runs every test module in tf_gpu. The full suite was not run (integration step).

## Deviations / choices (none change the design's statistics)

- Direction table: `hd95` added to the lower-is-better names (evaluation writes template_hd95 / pair_hd95 with --hd95; otherwise they would be "direction unknown").
- Decision text when the SESOI exists but neither rule applies: "practical relevance not established (SESOI x)". Full form: "measurable impact (B better|worse)" or "no measurable impact", then "; practically relevant ..." / "; no practically relevant effect (CI inside +-x)" / that third variant.
- Every metric's bootstrap uses a fresh `np.random.default_rng(seed)`, so a metric's CI does not depend on which other metrics are present.
- Subjects with a non-finite value in either arm are dropped for that metric (n is reported per metric). Pair metrics: non-finite pairs get W = 0.
- compare.csv also has the per-label rows (family `per_label`); compare.json holds per-label rows, notes, group values, versions and the scope.
- `--label-a/--label-b` default to `A` / `B`; tta_mindboggle.sh passes `none` / `flip8`.
- tta_mindboggle.sh: the arm calls use `env VAR=... bash predict_mindboggle.sh` (through a `run` wrapper so that DRY_RUN=1 can print them) and pass PROCESSED, GRID_REF, REPO and SIF explicitly. Missing CKPT/NAME, NAME with ',' or ':', or an unknown PARTS entry exit 2 before anything runs (documented in the header; that 2 can collide with "2 failed parts", which the log text tells apart). `--tta-levels` is passed only to the flip8 runs (the CLI refuses it with `--tta none`). PROJ is fixed, as in predict_mindboggle.sh.
- compare_tta.sh exits 2 when A, B or OUT is unset.
- predict_mindboggle_multi.sh was not in the bash -n list before; added.
