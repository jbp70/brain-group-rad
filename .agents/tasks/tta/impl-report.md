# Implementation report: 8-flip TTA (milestone 10)

Workflow wf_3ff8d564e9d8b835 (resume of impl-tta-flip8 / wf_d9878f1d8bff3eb3). Source of truth: `design.md` v2 (APPROVED) incl. 12.1 (Jay's answers), `review.md`, `plan.md`. Branch `main`, base HEAD `19b8a60`. Everything below is UNCOMMITTED in the working tree (no commit, push, pull or cluster job was made by the workflow). FEAT-001..004 APPROVED in their first review iteration (only NITs; see section 4).

## 1. Files

New:

| File | Content |
|---|---|
| `inference/tta.py` | TTA library (numpy at module level, TF imported inside functions): `FLIPS_8`, `IDENTITY_ONLY`, `TTAConfig`, `linear_flip_signs`, `flip_matrix_np`, `unflip_matrices`, `encode_linear` / `mean_linear` (Markley eigh read-out) / `decode_linear`, `linear_tta_level` (gauge-free residual averaging, design 2.3), `flip_np`, `unflip_field`, `linear_spread`, `gauge_size`, `train_flip_probs`, spread column names |
| `inference/linear_eval.py` | `predict.py linear-eval` (design 7.3) |
| `scripts/compare_tta.py` | paired statistics A vs B (design 9.3; numpy + scipy 1.4, TF-free) |
| `slurm/tta_mindboggle.sh` | one GPU job: arm A / arm B predict, linear-eval x 4, in-job compares; prints the basic-partition commands |
| `slurm/compare_tta.sh` | basic CPU job around `compare_tta.py` |
| `tests/test_tta.py`, `tests/test_predict_tta.py`, `tests/test_predict_tta_golden.py` + `tests/fixtures/predict_tta_off_golden.json`, `tests/test_linear_eval.py`, `tests/test_compare_tta.py` | tests (section 3) |
| `.agents/tasks/tta/run_tests.sh` | local WSL test helper (tf / py38 / compile / bashn / script) |

Modified:

| File | Change |
|---|---|
| `inference/streaming.py` | `deform()` split verbatim into `_v_raw_passes` + P12 (bit-identical, golden); new `deform_tta` with `_f_tta_acc` / `_f_tta_spread` |
| `inference/runner.py` | `tta=` argument, `_tta_at` / `_tta_active`, `_linear_stage`, `_deform_stage`, level masks, second moments, `src_affine` lifetime at a TTA last level (`try/finally`), TTA in `_linear_sel` / `_deform_sel`, `summary["tta"]` |
| `inference/cli.py` | `--tta {none,flip8}`, `--tta-levels`, validation, `tta_spread.csv` in OUTPUT_NAMES, `linear-eval` subparser, `_common(..., branches=True)`, shared `_validate_out_dir` |
| `inference/predict.py` | `tta_record` in PREDICT_KEYS, `_peek_flip_chain`, `_setup_tta`, `_tta_finish`, `_write_tta_spread`, estimate with TTA, pipeline text, warnings, non-finite flips |
| `inference/store.py` | `estimate_store_gb(..., tta_levels=())` (+28 V per image in the deform term of each TTA level) |
| `inference/__init__.py`, `predict.py` | docstrings, `linear-eval` dispatch |
| `evaluation/evaluate.py` | additive `per_subject_label.csv` (`subject`, `template_dice_<label>`); no existing output changed |
| `slurm/predict_mindboggle.sh` | env `TTA` (none), `TTA_LEVELS` (empty), `SKIP_CHECKS` (empty); TTA flags on step 2 only |
| `tests/test_predict_store.py`, `tests/test_evaluation.py`, `tests/test_slurm_scripts.py`, `tests/run_tests_wsl.sh` | new / updated tests; runner lists (py_compile, bash -n, TF-free list) |

Not touched: training code (`models/`, `runtime/`, `main.py`), `UnchunkedRunner`, `check-streaming`, `eval_synth.py`, training-time validation (Jay 12.1 item 4).

## 2. Flags and defaults

- `predict.py run`: `--tta {none,flip8}` (default `none` = the unchanged code path, bit-identical outputs); `--tta-levels LEVELS` (e.g. `1`, `2,1`; default all levels the plan runs; requires `--tta flip8`). Refused with exit 2: `--tta-levels` without flip8, `--tta flip8 --mode unchunked`, invalid / duplicate levels, a level the plan does not run (predict.json written).
- `predict.py linear-eval`: `--task {rigid,affine}` (required), `--seed` (default 0; same seed = same rendered group in both arms), `--tta`, `--tta-levels`, `--store {auto,ram,disk}`, `--store-dir`, `--out` (required), `--overwrite`, plus the common flags (`--checkpoint`, `--processed-dir`, `--subjects-file`, `--levels`, `--chunk-*`, `--precision`, `--device`, `--grid-reference`, `--gpu-memory-limit-mb`); no `--branches` (= task). Outputs `per_subject.csv`, `per_pair.csv`, `summary.json` (written last, also on failure; note "errors include the residual alignment of the processed subjects"), `tta_spread.csv` with flip8. Exit 2 input / < 2 subjects, 3 non-finite A.
- predict.json: `tta` (mode string, unchanged) and new `tta_record` (null with none; else mode, levels, flips, linear_averaging, svf_accumulation "float32", train_flip_prob per level, spread incl. `gauge_deg` / `gauge_mm`, nonfinite_flips, timing_s). `tta_spread.csv` (subject, subject_dir, level, task, spread_rms_mm, spread_max_mm [linear], spread_p99_mm [deform], shift_mm, rot_deg_rms, trans_mm_rms, scale_pct_rms, shear_pct_rms, rot_gap, readout_dev_deg) only with flip8.
- `compare_tta.py`: `--a`, `--b`, `--out` (required), `--kind {eval,linear}` (default eval), `--label-a` / `--label-b` (default A / B), `--n-boot` 10000, `--seed` 0. Outputs compare.json / compare.csv / compare.md; exit 2 for mismatched subject sets or missing files. SESOIs (Jay 12.1 item 2): Dice 0.002, folding max(0.25 x mean_A, 0.0002 pp), linear pair RMS 0.5 mm. Holm over P1-P3 (template_dice, pair_dice, fold_fwd_pct) and within secondary families.
- Plain mean of the re-encoded 10-D entries (Markley); no weighting option (12.1 item 7). All 8 flips (item 6).

## 3. Test results (integration run, 2026-11-07, local WSL, GTX 1080, TF GPU limit 6000 MB)

| Check | Result |
|---|---|
| Full suite `wsl -d Ubuntu bash .../tests/run_tests_wsl.sh` (once, background, 00:53-08:29; log `.agents/tasks/tta/_integration_full_suite.log`) | `suite status: 0`; stage 1 (bgr_preproc38): py_compile OK, ast (3, 8) OK (127 files), ast (3, 6) OK, np_reference OK, all `bash -n` OK incl. tta_mindboggle.sh / compare_tta.sh / predict_mindboggle_multi.sh, TF-free tests OK without TF (217 tests); stage 2 (tf_gpu): **1001 tests, 4 skipped, 0 failing invocations**, wall 27205 s (7.6 h). Previous full suite: 910 tests. |
| T9 golden `tests.test_predict_tta_golden` (in the suite) | `[golden tta-off] 70 hashes compared, 0 differ (fixture commit 19b8a608...)` OK |
| `tests.test_predict_tta.TestEndToEnd` (in the suite) | `--tta none: 66 NIfTI hashes == pre-TTA fixture` |
| `tests.test_trainer.TestLinearOnlyUnchanged` (in the suite) | `[golden] max |diff| 0 (tolerance 0)` OK |
| `tests.test_trainer.TestDeformUnchanged` (in the suite) | `[golden deform] max |diff| 0 (tolerance 0)` OK |
| `run_tests.sh compile` (py3.8, bgr_preproc38) | py_compile OK, ast feature_version (3, 8) OK (23 files), `inference.tta imports without TF` |
| `check_gauge_free.py` (bgr_preproc38) | ALL PASS: E1 3.6e-16, E2 0, E3 v1 0.183 mm vs v2 3.4e-14 mm, E4 5.3e-15, E5 3.0e-14 mm |
| `check_flip_signs.py` (tf_gpu, after the suite) | ALL PASS: C3 x/y/z == Jay's patterns, combined == products; C1/C2 0; C4 0; C5 7.8e-6; C6 2.2e-6 vox / 3.8e-6; C7 float64 exact, float32 1.5e-7 rel |

The goldens were not rerun separately: they ran inside the full suite on the same tree. Logs: `_integration_full_suite.log`, `_integration_cpu_checks.log`, `_integration_check_flip_signs.log` in `.agents/tasks/tta/`. **No failures, no fixes needed in this integration step.**

Per-module counts of the new modules (suite): test_tta 33, test_predict_tta 17, test_predict_tta_golden 1, test_linear_eval 4, test_compare_tta 12 (also in the TF-free stage), test_predict_store 15, test_slurm_scripts 68, test_evaluation 12.

## 4. Deviations from the design / plan

1. **Design 6 host-store estimate corrected (FEAT-001).** The design says "39.7 -> 59.2 GB for the level-1 Mindboggle run". `estimate_store_gb` takes the max over levels; with the default float16 trainable level-1 stage the frozen float32 level-2 term is the max term without TTA, so TTA adds less than the full 28 V level-1 term: **39.70 -> 53.08 GB (+13.38 GB) for d12hs widths** (+10.86 GB for default widths); with a float32 trainable stage it is +19.46 GB as in the design. Both are below the 0.7 x 103 GB = 72 GB RAM threshold of a 96G job, so the store stays in RAM and `--mem=96G` is enough. store.py implements the design rule as written; the test asserts the actual deltas.
2. **FEAT-002 test tolerances (no behaviour change).** In `TestEquivariantStubs` (T7) deform fields U1, U1_inv, u_ckpt, u_inv_ckpt and v agree within **1e-4 vox**, and A / pairwise maps within **1e-4 mm** (1e-5 mm at level 4), set from measured float32 rounding (scaling and squaring 2.4e-6 vox at L4; 1.3e-5 to 3.4e-5 vox at L2 and after the affine pre-warp; pairwise 5.9e-6 mm L4, 2.7e-5 mm L2). The design's 1e-6 vox still applies and is asserted for v at a first level without pre-warp (measured 2.4e-7). T6 tolerances unchanged (1e-4 mm / 1e-4 vox).
3. FEAT-001 API additions: `linear_tta_level(..., dtype=None)` (float64 pipeline for E3-E5 tests; no-op on the model path), `linear_spread(..., mean_diag=None)`, `decode_linear(..., dtype=np.float32)`.
4. FEAT-003: the spread writer was already factored as `predict._write_tta_spread` inside `_tta_finish` (no rename to `write_tta_spread`); with non-finite A (exit 3) `tta_spread.csv` is still written (as in `run`); summary.json has extra timing keys (`load`, per-phase) and `peak_store_gb` / `peak_rss_gb`; `means` / `p95` cover every `group_errors` array (incl. cycle, null at N 2).
5. FEAT-004: `hd95` added to the lower-is-better names; decision text variant "practical relevance not established (SESOI x)" when neither SESOI rule applies; fresh `default_rng(seed)` per metric bootstrap; subjects non-finite in either arm dropped per metric (n reported), non-finite pairs get weight 0; `compare.csv` also carries per-label rows; `compare_tta.sh` exits 2 when A, B or OUT is unset; `tta_mindboggle.sh` exit 2 also for invalid settings (log text distinguishes it from "2 failed parts"); `predict_mindboggle_multi.sh` added to the `bash -n` list.
6. FEAT-001 golden mode: 'hash' (sha256 of decoded arrays + affines + A, tolerance 0) under `enable_op_determinism`; written and compared on the unchanged tree 19b8a60 before any source edit.

Open review NITs (accepted, not fixed): FEAT-002 TTA stores not freed if `deform_tta` raises (process exits on that path); FEAT-003 with `--overwrite` stale `per_*.csv` / `tta_spread.csv` from an earlier run can stay in `--out` (summary.json `outputs` lists only the current files); FEAT-004 `read_table` drops unparsable columns silently, exit-code 2 overlap, per-label "B better" assumes Dice.

## 5. Cluster commands (for Jay; nothing was run or submitted)

Rules: account `-A qtim` only (never `entre`); one GPU per job; at most **two active jobs** (`squeue -u jn85`, Jay's own jobs count); ask before each submission. The scripts already carry `-A qtim`, rtx8000 / basic, 1 GPU, 3 CPUs (GPU job), as below.

### 5.0 Commit, push, pull (Jay)

Local (PowerShell, repo root). Leaves out `.agents/tasks/inverse-centrality-investigation/` and `.agents/tasks/smoothness-scaling/` (other agents' files) and the raw `_*.log` files:

```powershell
cd d:\Research\Brain_Registration_Projects\brain-group-rad
git add evaluation/evaluate.py inference/__init__.py inference/cli.py inference/predict.py inference/runner.py inference/store.py inference/streaming.py inference/tta.py inference/linear_eval.py predict.py scripts/compare_tta.py slurm/predict_mindboggle.sh slurm/tta_mindboggle.sh slurm/compare_tta.sh tests/run_tests_wsl.sh tests/test_evaluation.py tests/test_predict_store.py tests/test_slurm_scripts.py tests/test_tta.py tests/test_predict_tta.py tests/test_predict_tta_golden.py tests/fixtures/predict_tta_off_golden.json tests/test_linear_eval.py tests/test_compare_tta.py
git add .agents/tasks/tta/design.md .agents/tasks/tta/plan.md .agents/tasks/tta/review.md .agents/tasks/tta/review.json .agents/tasks/tta/review-FEAT-001.md .agents/tasks/tta/review-FEAT-002.md .agents/tasks/tta/review-FEAT-003.md .agents/tasks/tta/review-FEAT-004.md .agents/tasks/tta/verdict-FEAT-001.json .agents/tasks/tta/verdict-FEAT-002.json .agents/tasks/tta/verdict-FEAT-003.json .agents/tasks/tta/verdict-FEAT-004.json .agents/tasks/tta/test-log-FEAT-001.md .agents/tasks/tta/test-log-FEAT-002.md .agents/tasks/tta/test-log-FEAT-003.md .agents/tasks/tta/test-log-FEAT-004.md .agents/tasks/tta/impl-report.md .agents/tasks/tta/check_flip_signs.py .agents/tasks/tta/check_gauge_free.py .agents/tasks/tta/review_gauge_demo.py .agents/tasks/tta/run_tests.sh .agents/tasks/tta/task-tta-flip8
git commit -m "feat: 8-flip TTA (predict.py run --tta flip8, linear-eval, compare_tta.py, tta_mindboggle.sh); full suite 1001 tests OK, goldens 0 diff"
git pull --rebase
git push
```

Cluster (login node). The commit changes `inference/`, `evaluation/`, `scripts/`, `slurm/`, `tests/` only (no training code), so running training jobs are not affected; do not pull while a `predict.py` / evaluation job is running from the clone (AutoGraph re-reads changed source files):

```bash
cd /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad
squeue -u jn85
git status --short && git pull --ff-only && git log --oneline -1
```

### 5.1 Order (two-active-jobs rule)

1. GPU job (5.2): arm A + arm B predict, the four linear-eval runs and the in-job compares. 1 active job (+ Jay's).
2. After it finished: the two all-pairs CPU evaluations (5.3), together = 2 active jobs (submit one at a time if Jay has jobs queued).
3. After both finished: the all-pairs compare (5.4). Optional standalone compares (5.4) only if the in-job compare part failed or was skipped.

Before every step: `squeue -h -u jn85 | wc -l` must be < 2.

In all commands `P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg`, run from the clone `$P/brain-group-rad`, `NAME=r1_cfgB_sm10_s2500_bnfreeze_true`.

### 5.2 GPU job: Mindboggle predict arm A (`--tta none`) / arm B (`--tta flip8`) + linear-eval

rtx8000 (Quadro RTX 8000 48 GB), `-A qtim`, 1 GPU, 3 CPUs, 96G, 10 h limit (estimate ~3 h: arm A ~35 min incl. checks 1a-1c and template-only evaluation, arm B ~65 min, linear-eval 4 runs <= ~70 min). Host store estimate with TTA 53.1 GB < 72 GB (RAM store).

Recommended, one job (PARTS default `predict linear compare`):

```bash
P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
cd $P/brain-group-rad
NAME=r1_cfgB_sm10_s2500_bnfreeze_true CKPT=$P/results/r1_cfgB_sm10_s2500_bnfreeze_true/ckpt/last \
  sbatch --job-name=bgr-tta-mb slurm/tta_mindboggle.sh
```

Outputs: `results/predict_mindboggle/${NAME}_tta_none/` and `${NAME}_tta_flip8/` (fields/, predict.json, tta_spread.csv for flip8, eval_template/ = template-only evaluation), `results/linear_eval_mindboggle/${NAME}_{rigid,affine}_tta_{none,flip8}/`, `results/tta_compare/${NAME}_template`, `${NAME}_rigid_linear`, `${NAME}_affine_linear`. Log `$P/logs/bgr-tta-mb-<job>.out`. Check first (design 8): flip8 predict.json `tta_record` (spread_rms linear <~ a few mm, deform <~ 1 mm, readout_dev_deg small, gauge_deg / gauge_mm recorded, nonfinite_flips empty) and that arm A's check 1a gate passed.

Alternative, two GPU jobs in parallel (both count; only when no other job is active):

```bash
NAME=r1_cfgB_sm10_s2500_bnfreeze_true CKPT=$P/results/r1_cfgB_sm10_s2500_bnfreeze_true/ckpt/last PARTS=predict \
  sbatch --job-name=bgr-tta-pred --time=05:00:00 slurm/tta_mindboggle.sh
NAME=r1_cfgB_sm10_s2500_bnfreeze_true CKPT=$P/results/r1_cfgB_sm10_s2500_bnfreeze_true/ckpt/last PARTS=linear \
  sbatch --job-name=bgr-tta-lin --time=04:00:00 slurm/tta_mindboggle.sh
```

then the compares of 5.4 (template, rigid, affine) as basic jobs.

Dry run of the job's commands (prints only, safe on the login node): `DRY_RUN=1 NAME=r1_cfgB_sm10_s2500_bnfreeze_true CKPT=$P/results/r1_cfgB_sm10_s2500_bnfreeze_true/ckpt/last bash slurm/tta_mindboggle.sh`.

### 5.3 All-pairs CPU evaluation of both arms (basic)

basic partition, `-A qtim`, 8 CPUs, 96G, 12 h (script header; ~1.5-2 h each). `evaluation run --pairs all` writes the template metrics too (per_subject.csv, per_pair.csv, per_subject_label.csv, summary.json):

```bash
P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
cd $P/brain-group-rad
NAME=r1_cfgB_sm10_s2500_bnfreeze_true
for ARM in none flip8; do
  F=$P/results/predict_mindboggle/${NAME}_tta_$ARM/fields
  METHOD=${NAME}_tta_$ARM FWD="$F/{subject_dir}_fwd.nii.gz" INV="$F/{subject_dir}_inv.nii.gz" \
    ATLAS=$F/atlas.nii.gz ATLAS_SEG=$F/atlas_seg.nii.gz CONV="--components xyz --signs +++ --units mm" \
    sbatch slurm/eval_mindboggle.sh
done
```

Outputs `results/eval_mindboggle/${NAME}_tta_{none,flip8}/`. Fallback if basic is slow: `EVAL_PAIRS=1000` inside the predict job (design 9.4), recorded as such.

### 5.4 compare_tta.py (basic, 1 CPU, 4G, 15 min)

All-pairs (after both 5.3 jobs):

```bash
NAME=r1_cfgB_sm10_s2500_bnfreeze_true
A=$P/results/eval_mindboggle/${NAME}_tta_none B=$P/results/eval_mindboggle/${NAME}_tta_flip8 \
  OUT=$P/results/tta_compare/${NAME}_allpairs sbatch slurm/compare_tta.sh
```

Standalone versions of the in-job compares (only if needed):

```bash
A=$P/results/predict_mindboggle/${NAME}_tta_none/eval_template B=$P/results/predict_mindboggle/${NAME}_tta_flip8/eval_template \
  OUT=$P/results/tta_compare/${NAME}_template sbatch slurm/compare_tta.sh
for T in rigid affine; do
  A=$P/results/linear_eval_mindboggle/${NAME}_${T}_tta_none B=$P/results/linear_eval_mindboggle/${NAME}_${T}_tta_flip8 \
    OUT=$P/results/tta_compare/${NAME}_${T}_linear KIND=linear sbatch slurm/compare_tta.sh
done
```

Results: `compare.md` (scope statement, P1-P3 with Holm, SESOI decision), `compare.json`, `compare.csv`. Never run `compare_tta.py` on the login node.

The level-2 source (`results/r2_cfgB_s15000_smooth10_lr3e-4_wd1e-1_n32_d16/ckpt/last`, design 9.1, optional per 12.1 item 1) uses the same commands with `NAME=r2_cfgB_s15000_smooth10` and that `CKPT`.
