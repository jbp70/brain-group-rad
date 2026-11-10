# predict-recenter: post-hoc group re-centering in `predict.py run`

Branch `predict-recenter` (worktree `.worktrees/predict-recenter`, from main 3300f69). Option C1 of
`.agents/tasks/inverse-centrality-investigation/report.md` section 4. No training code changed. No
job was submitted.

## 1. What it does

`predict.py run --recenter K` (integer >= 0, default 0 = off). For K >= 1 the written forward fields
of the whole group are corrected so that their mean displacement is about zero. One common
diffeomorphism psi is applied to every subject:

- fwd_i' = phi_i o psi (atlas point x goes to psi(x), then to subject i),
- inv_i' = psi^-1 o phi_i^-1.

The subject-to-subject maps do not change, because phi_i' o phi_j'^-1 = phi_i o psi o psi^-1 o
phi_j^-1 = phi_i o phi_j^-1. In practice they change by one extra trilinear resampling and the
exp(-w) / exp(+w) discretisation (measured below: mean 0.0005 mm on the synthetic run).

Everything written after the model pass uses the corrected fields: `<subject>_fwd.nii.gz`,
`_inv.nii.gz`, `_moved.nii.gz`, `atlas.nii.gz` (mean of the moved images) and `atlas_seg.nii.gz`
(majority vote of the warped labels).

### 1.1 Algorithm (inference/recenter.py)

The fields are the canonical voxel displacements on the processed 1 mm grid, the same arrays the
evaluator reads (`evaluation/fields.py`: phi_i(x) = x + fwd_i(x), atlas to subject i).

1. Model pass (unchanged model code). The `run` sink converts each subject as before. With
   `--recenter` it does not write NIfTI files yet. It saves fwd_i and inv_i as uncompressed `.npy`
   files in a temporary directory, adds fwd_i to a float64 sum, and counts the moved image > 0 per
   voxel (for the evaluator's atlas mask). The internal consistency check and the non-finite counts
   run on these original fields, as before.
2. m_0 = sum / N (exact mean over all N subjects). w_1 = m_0.
3. Round r = 1..K: psi_r = exp(-w_r) and psi_r^-1 = exp(+w_r), by scaling and squaring at 1 mm
   (`models.diffeo.exp_and_inverse`, T = 7 steps, lean sampler). For every subject,
   fwd_i^(r) = `diffeo.compose(fwd_i, p)` = p + fwd_i(x + p), always formed from the ORIGINAL
   fwd_i. Each written field is therefore resampled once, whatever K is. The residual is
   m_r = mean_i fwd_i^(r), and the next round uses w_{r+1} = w_r + m_r. This first-order update is
   the reference probe's `center` mode (`probe_inverse_centrality.py`), with exp instead of id - m.
4. The last round also forms inv_i^(K) = `diffeo.compose(q, inv_i)` = inv_i + q(y + inv_i(y)) and
   writes the subject (moved image, labels, fwd / inv / moved files, atlas sums). The atlas files
   are written afterwards, as before.

Resampling is trilinear with border clamping, the same as `evaluation.fields.compose`. The unit test
compares it with a scipy recomputation: difference <= 1.6e-6 voxel.

### 1.2 Why psi = exp(-w)

psi = exp(-w) and psi^-1 = exp(+w) form an inverse pair by construction, the same semantics as the
model's own fields (`diffeo.exp_and_inverse`). So fwd' and inv' stay as consistent as fwd and inv
were. The alternative id -/+ m is inverse only to first order: (id + m) o (id - m) = id - (Dm) m +
O(m^3). With |m| ~ 0.4 mm and |Dm| ~ 0.1 that adds ~0.04 mm of inverse error everywhere. The cost
of exp is 2 x 7 compositions of ONE field per round, which is negligible.

### 1.3 Branches and other flags

- `deform`: corrects the written displacement u.
- `affine+deform`: corrects the full written displacement (affine plus deformable, fwd = A^-1 o
  (id + u)). The group mean of the affine part is not zero either, so psi can contain a small common
  near-linear part. The correction is the same composition, and pairwise maps still cancel.
- `rigid` / `affine`: refused at argument parsing (exit 2, "--recenter needs --branches deform or
  affine+deform"). A common non-linear psi would turn the linear outputs into non-linear fields;
  rigid / affine centering is already exact in matrix space inside the model.
- `--tta flip8`: re-centering runs on the TTA-averaged final fields (after TTA), with no other
  interaction.
- `--levels` and `--integration-grid native|fine`: re-centering works on the final 1 mm fields, so
  any level plan or integration grid works the same way. Tested: level-2 cascade with `--levels 4`
  and with `--integration-grid fine`.
- `--mode unchunked` works too (same sink).
- `check-streaming` and `linear-eval` have no `--recenter` flag.
- The internal consistency check (`--check-internal`) still compares the ORIGINAL converted fields
  with the model's own warp ops. It checks the conversion code, and the corrected fields
  intentionally differ from the model's u by psi.
- Non-finite model outputs: re-centering is skipped. The uncorrected fields are written, the atlas
  is not, exit 3 as before, and predict.json gets `recenter.status = "skipped_nonfinite"`.
- `--recenter 0` (the default) takes the old sink path unchanged. Outputs are bit-identical
  (`tests.test_predict_tta_golden`, 0 hashes differ). The only new predict.json key is
  `recenter: {"rounds": 0}`.

### 1.4 predict.json record and log

`recenter` (K >= 1): `rounds`, `status`, `method`, `integration_steps`, `mask` (definition text),
`atlas_mask_voxels`, `centrality_by_round` (round 0 = before, then after each round: `mean_mm`,
`max_mm`), `centrality_final_mask` (the final residual on the atlas mask of the CORRECTED moved
images, which is what `python -m evaluation run` will report), `max_abs_w_mm` per round, `inv_err`
(the evaluator's inv_err_fwd_mm on the atlas mask and inv_err_inv_mm on the subject mask image > 0,
with maxima, before / after, for the first 4 subjects plus per-subject values), `timing_s`
(`collect`, `round_1`..`round_K`), `tmp_dir`, `tmp_gb`. `timing_s.recenter` in the top-level
timing holds the total. `pipeline` gets "; post-hoc re-centering, K round(s): ...".

Centrality definition: the evaluator's (`evaluation/evaluate.py`), the mean / max over the atlas
mask of |mean_i fwd_i| in mm. The atlas mask is the voxels where at least half of the moved images
are > 0. Per-round values use the mask of the uncorrected moved images, because the corrected moved
images exist only in the last round. The final value is also given on the corrected mask.

Log lines: one at the start ("centrality before ... mm"), one per round ("re-centering round r / K:
centrality a -> b mm (max ...), |w| max ..., t s"), and one for the inverse consistency before /
after.

## 2. Memory, disk and time at N 101, level 1

- Host RAM: unchanged in practice. The original fields go to disk, not RAM. The extra RAM is the
  float64 sum (165 MB), the voxel count (27 MB), w / residual / psi (~0.5 GB) and one subject's
  4 fields (330 MB). Rounds run after the streaming store is closed, so they never overlap the model
  pass peak (74-93 GB today). Estimated extra < 1 GB, within the 96 GB request.
- Disk: N x 2 x 82.6 MB = 16.7 GB of `.npy` in `--store-dir`, else `$TMPDIR` (node-local
  `/scratch/$SLURM_JOB_ID` in the slurm script), else `<out>/_recenter_tmp`. The directory is
  removed at the end, also on errors. If the streaming store goes to disk (`--store disk`, or
  `auto` choosing disk), both live in `$TMPDIR` at the same time during the model pass.
- GPU: one subject at a time. exp of one 1 mm field plus two compositions, < 1.5 GB. Negligible
  next to the model pass (~11 GB at level 1 / level 2).
- Time, measured locally (GTX 1080 under WSL, full 160x224x192 grid, N 6, K 2,
  `.agents/tmp/rc_timing.py`):
  - saving the originals: 0.25 s per subject;
  - a non-final round: 0.64 s per subject, including exp of w;
  - the last round: about 1 s per subject plus the inverse diagnostics (4 subjects, about 2 s each),
    not counting the writes, which happen anyway.

  The model pass also does one extra moved-image warp per subject (for the atlas mask), about
  0.8 s on CPU. Estimate at N 101, K 2: 25 + 80 + 65 + 100 + 10 s, about 5 min, so roughly 5-10 min
  per run on the cluster. Today's run takes 23 min without TTA and 49 min with flip8.

## 3. Tests

New `tests/test_predict_recenter.py` (8 tests):

- **TestRecentererUnit** (32x36x28 grid, N 5, phi_i = exp(v_i + c) with a common uncentred c).
  - Centrality 0.869 -> 0.039 -> 0.0049 mm in 2 rounds.
  - Corrected fields match an independent scipy recomputation of the algorithm: fwd 1.6e-6,
    inv 3.6e-7 voxel.
  - Pairwise maps compose(inv_j', fwd_i') change by at most 0.025 voxel in the interior.
  - Inverse error 0.0068 -> 0.0075 voxel (tolerance +0.01).
  - Also tested: one-round case, K = 0 / missing-subject errors, the atlas-mask rule, and that the
    temporary directory is removed.
- **TestRecenterCLI**: default 0; `-1` and `x` give exit 2; rigid / affine with `--recenter 2`
  give exit 2; `check-streaming --recenter` is rejected.
- **TestRecenterRun**: a tiny level-4 checkpoint, N 3, full grid. `to_canonical` is patched to add a
  known common warp C (fwd o C, C^-1 o inv). Runs with `--recenter 0` and `--recenter 2`:
  - Records checked. Centrality 0.391 -> 0.0043 -> 0.00013 mm, the same 0.00013 on the final mask.
  - The round-0 and final-mask centralities equal the evaluator's formula recomputed from the
    written files (1e-5).
  - Moved files equal F.warp(image, written fwd) exactly. The atlas equals the mean of the moved
    images (1e-6). atlas_seg equals the MajorityVote of the labels warped with the written fwd,
    exactly.
  - Pairwise maps (rc0 vs rc2, subject masks): mean change 0.0005 mm, p99.9 0.0036 mm.
  - inv_err fwd 0.0009 -> 0.0015 mm and inv 0.0014 -> 0.0017 mm (one extra resampling; tolerance
    +0.01).
  - `--branches affine+deform --tta flip8 --recenter 1` runs (internal check passes, tta_spread.csv
    written).
  - Level-2 cascade with `--levels 4 --recenter 1` and with `--integration-grid fine --recenter 1`.
  - Non-finite outputs give exit 3, `skipped_nonfinite`, no atlas, and no temporary directory left.
- `tests/test_slurm_scripts.py`:
  - static checks of the RECENTER knob (header, default, `$RECENTER_FLAGS` only on step 2);
  - a stub run: RECENTER unset / empty gives today's calls, RECENTER=2 gives `--tta none
    --recenter 2` on step 2 only;
  - `predict_mindboggle_multi.sh` passes RECENTER (and TTA / SKIP_CHECKS) through to every entry.

Results: see section 5.

## 4. Cluster commands (NOT submitted; ask Jay first)

Pull the merged commit in the clone first (no running predict job may use the clone):

```bash
P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
cd $P/brain-group-rad && git pull --ff-only && git log --oneline -1
squeue -u jn85
```

### (a) Mindboggle check, level-2 checkpoint, RECENTER=2

rtx6000 (level-2 GPU peak ~11 GB), 3 CPUs, 96 GB, 4 h. The run takes about 40 min. Compare with
the existing no-recenter run `results/predict_mindboggle/r2_cfgB_s16000_sm10_synth05_current_deform`
(job 10627409).

```bash
P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
cd $P/brain-group-rad
NAME=r2_cfgB_s16000_sm10_synth05_current_deform_rc2 \
CKPT=$P/results/r2_cfgB_s16000_sm10_synth05_current/ckpt/last \
RECENTER=2 \
sbatch -p rtx6000 --cpus-per-task=3 --mem=96G --time=04:00:00 --job-name=bgr-predict-rc2 slurm/predict_mindboggle.sh
```

Check:

- `fields/predict.json` `recenter.centrality_by_round` should drop toward ~0;
  `recenter.inv_err.after` should be <= before + ~0.02 mm.
- `eval_template/summary.json`: `centrality_mean_mm` (was 0.266) -> ~0.01. Template Dice should be
  ~unchanged (was 0.677); fold / inv_err should be ~unchanged.

### (b) Full comparison: four level-1 runs x {no re-centering, RECENTER=2} x {TTA none, flip8}

Sixteen predictions in two GPU jobs (one per TTA arm). Each job runs `predict_mindboggle_multi.sh`
twice, without and then with re-centering. The streaming checks 1a-1c run once per checkpoint, in
the first pass of job 1. Job 2 and the re-centering passes use SKIP_CHECKS=1.

Output names are `<run>_tta_none`, `<run>_tta_none_rc2`, `<run>_tta_flip8` and
`<run>_tta_flip8_rc2` under `results/predict_mindboggle/`.

Estimated run times:

- job 1: about 4 x 45 + 4 x 40 min = 5.7 h;
- job 2: about 4 x 65 + 4 x 72 min = 9.1 h.

Memory: the level-1 predict MaxRSS was 74.5 GB (flip8 job 10622313), within the 96 GB request.

Both jobs at once is within the 2-job limit, but only if nothing else is active.

```bash
P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
cd $P/brain-group-rad
M=$P/brain-group-rad/slurm/predict_mindboggle_multi.sh
runs() { s=""; for r in r1_cfgB_s6000_sm10_real r1_cfgB_s6000_sm15_real r1_cfgB_s6000_sm10_synth05 r1_cfgB_s6000_sm15_synth05; do s="$s ${r}_$1:$P/results/$r/ckpt/last"; done; echo $s; }
# job 1: TTA none (checks 1a-1c in the first pass), then TTA none + RECENTER=2
RUNS_A="$(runs tta_none)" RUNS_B="$(runs tta_none_rc2)" sbatch -A qtim -p rtx6000 --gpus=1 -N 1 --ntasks-per-node=1 \
  --cpus-per-task=3 --mem=96G --time=12:00:00 --job-name=bgr-mb-none \
  --output=$P/logs/%x-%j.out --error=$P/logs/%x-%j.err \
  --wrap "RUNS=\"\$RUNS_A\" TTA=none bash $M; RUNS=\"\$RUNS_B\" TTA=none RECENTER=2 SKIP_CHECKS=1 bash $M"
# job 2: TTA flip8, then TTA flip8 + RECENTER=2 (checks skipped: same checkpoints as job 1)
RUNS_A="$(runs tta_flip8)" RUNS_B="$(runs tta_flip8_rc2)" sbatch -A qtim -p rtx6000 --gpus=1 -N 1 --ntasks-per-node=1 \
  --cpus-per-task=3 --mem=96G --time=16:00:00 --job-name=bgr-mb-flip8 \
  --output=$P/logs/%x-%j.out --error=$P/logs/%x-%j.err \
  --wrap "RUNS=\"\$RUNS_A\" TTA=flip8 SKIP_CHECKS=1 bash $M; RUNS=\"\$RUNS_B\" TTA=flip8 RECENTER=2 SKIP_CHECKS=1 bash $M"
```

Notes:

- sbatch exports the submitting shell's environment (`--export=ALL`), so `RUNS_A` / `RUNS_B` reach
  the job.
- The job's exit status is the failure count of its second pass. Read both "multi finished: N
  failed" lines in the log.
- If job 1's float32 gate fails for a checkpoint, job 2's entries for that checkpoint still run,
  because their checks are skipped. Check job 1's log before using job 2's numbers.
- Alternative with less wall time per job: 4 jobs, one per arm, each a plain
  `RUNS=... [TTA=flip8] [RECENTER=2] [SKIP_CHECKS=1] sbatch -p rtx6000 --cpus-per-task=3 --mem=96G
  --time=08:00:00 slurm/predict_mindboggle_multi.sh`. That needs two submission rounds under the
  2-job limit.
- Pairwise (all-pairs) evaluations go to the basic partition afterwards (`slurm/eval_mindboggle.sh`
  per NAME; printed at the end of each entry). TTA comparisons use `slurm/compare_tta.sh` with
  A/B = the `_tta_none[_rc2]` / `_tta_flip8[_rc2]` eval folders.

## 5. Verification results

- Targeted runs (WSL, tf_gpu, BGR_TF_GPU_LIMIT_MB 6000):
  - `tests.test_predict_recenter`: 8 tests OK (unit class 21 s, run class 527 s).
  - `tests.test_slurm_scripts`: 78 tests, 77 OK. `TestCascadeRuns.test_dry_run` fails in the
    worktree only: WSL git cannot read the worktree's Windows `gitdir:` path. It does not touch the
    changed code, and it passes in a normal clone.
- Full suite `tests/run_tests_wsl.sh` against the worktree (commit c4cf573 plus the log-message
  edit of the next commit; 28 153 s unittest wall time, 28 227 s in total):
  - Step 1 passed: py3.8 `py_compile` and `ast feature_version=(3, 8)` OK on 132 files,
    py3.6 OK on the login-node helpers, `bash -n` OK on all slurm scripts.
  - 1061 tests (5 skipped). 1059 passed and 2 failed: `tests.test_slurm_scripts.TestCascadeRuns.test_dry_run`
    (it also fails in the TF-free step 1 subset) and
    `tests.test_predict.TestGitCommit.test_real_repository`. Both fail because WSL git cannot
    resolve the worktree's `.git` file (`gitdir: D:/...`, a Windows path). `git rev-parse` fails,
    and the `.git/HEAD` fallback finds a file, not a directory. Neither test touches the changed
    code. Both are re-run in the main checkout after the fast-forward merge (section 6).
  - Golden: `tests.test_predict_tta_golden` reports 70 hashes compared, 0 differ (fixture commit
    19b8a60). Every other golden-based test passed.
  - New module `tests.test_predict_recenter`: 8 tests OK.

## 6. Commits

- c4cf573 feat: predict.py run --recenter K (code, slurm knob, tests)
- 9fb0b36 docs: this report; clearer non-finite log message in the re-centering write
- Main had not moved (3300f69), so no rebase was needed. `git merge --ff-only predict-recenter` on
  local main: 3300f69 -> 9fb0b36. Not pushed.
- Re-run in the main checkout at 9fb0b36 (a normal `.git` directory):
  `tests.test_slurm_scripts.TestCascadeRuns.test_dry_run` and
  `tests.test_predict.TestGitCommit.test_real_repository` both OK (tf_gpu), and `test_dry_run` OK
  in bgr_preproc38. So the full suite result is 1061 tests, 0 real failures.
