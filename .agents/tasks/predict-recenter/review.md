# Review: predict-recenter (`predict.py run --recenter K`)

Reviewed: `3300f69..e33e1f7` (c4cf573 feat, 9fb0b36 + e33e1f7 docs). The branch was already
fast-forwarded into local main (main == predict-recenter == e33e1f7, not pushed), so
`git diff main...HEAD` is empty and the review used the commit range. Tests were not re-run; the
commit messages and report section 5 hold the evidence.

## Evidence

- **K = 0 is bit-identical, with no extra passes.** `_Writer.sink` (inference/predict.py ~660-680)
  takes `self._emit` when `recenter is None`. `_emit` contains the old write code line for line.
  The Recenterer is built only when `rounds` is non-zero. Store close and `model_and_write` timing
  are unchanged. The only new predict.json content is `recenter: {"rounds": 0}`. The full suite
  reports `tests.test_predict_tta_golden` with 70 hashes and 0 differ.
- **Composition order matches the repo conventions.**
  - In `models/diffeo.py`, `compose(u_a, u_b)` is phi_a o phi_b with b applied first, and
    `exp_and_inverse(v)` returns (exp v, exp -v).
  - `exp_pair` calls `exp_and_inverse(-w)` and gets p = exp(-w), q = exp(+w).
  - `correct_fwd` = `compose(fwd, p)` = phi_i o psi.
  - `correct_inv` = `compose(q, inv)` = psi^-1 o phi_i^-1.
  - The diagnostic `compose(i, f)` equals the evaluator's `F.compose(fwd, inv)` (fwd first, atlas
    mask), as in evaluation/evaluate.py 244-246.
  - The unit test cross-checks these against scipy `F.compose` (b o a semantics) to 1.6e-6 vox.
- **Rounds.** Every round recomputes the residual from fields corrected from the original fwd, with
  w <- w + m_r (the probe's first-order update, with exp). Inverse quality is recorded before and
  after.
  - Unit test: centrality 0.869 -> 0.039 -> 0.0049 mm; pairwise change <= 0.025 vox.
  - Run test: centrality 0.391 -> 0.0043 -> 0.00013 mm; inv_err +0.0006 mm.
- **Derived outputs use the corrected fields.**
  - The last round calls `writer.emit_corrected` -> `_emit` with (fwd', inv'). So moved, labels,
    the fwd/inv/moved files and `atlas.add` all use the corrected fields. `atlas.write` runs
    afterwards as before.
  - The test recomputes moved, atlas and atlas_seg (MajorityVote) from the written files and gets
    exact or 1e-6 agreement.
  - The internal check stays on the original conversion (documented, correct intent).
- **Memory.**
  - Originals are saved as .npy in `--store-dir`, else `$TMPDIR`, else `<out>/_recenter_tmp`
    (16.7 GB of disk at N 101).
  - The extra RAM is a float64 sum, a count, w, and one subject's fields.
  - The rounds run after `store.close()`, so they do not overlap the model-pass peak.
  - The tmp dir is removed in `finally`.
  - The estimate of < 1 GB extra host RAM is plausible within 96 GB.
- **Branches and flags.**
  - rigid / affine are refused in `cli.validate` (exit 2, tested).
  - affine+deform with flip8, the level-2 cascade with `--levels 4`, and `--integration-grid fine`
    are exercised end to end.
  - The non-finite path writes the originals and returns exit 3 with `skipped_nonfinite`.
- **predict.json.** The record has rounds, status, method, `centrality_by_round` (mean/max),
  `centrality_final_mask`, `inv_err` before/after, per-round timing and tmp_gb. A log line is
  printed per round.
- **Slurm.**
  - `RECENTER` empty adds no flag. `RECENTER=K` adds `--recenter K` on step 2 only. The knob is
    documented in the header.
  - Multi passes it through via the inherited environment; the header comment was updated and the
    stub test covers it.
  - No defaults changed.
- **Py3.8 / numpy 1.21.** py_compile and ast(3.8) are OK on 132 files. Only standard numpy and TF
  APIs are used (`default_rng` exists in 1.21).
- **Report.** It contains the (a) sbatch command (rtx6000, RECENTER=2, r2 synth05_current) and the
  (b) commands: 2 jobs x 8 predictions with distinct `_tta_{none,flip8}[_rc2]` NAMEs. Nothing was
  submitted. The `--wrap` quoting expands correctly (`$M` at submission, `$RUNS_A/B` in the job via
  `--export=ALL`).
- **Tests.** Full suite: 1061 tests, 2 failures from WSL git and the worktree gitdir. Both passed
  when re-run in the main checkout. The new module ran 8 tests OK.

## Non-blocking NITs

1. The per-round centralities use the atlas mask of the uncorrected moved images. Only the final
   value is on the corrected mask. This is documented, but readers comparing with the evaluator
   should use `centrality_final_mask`.
2. `inv_err` before/after covers only the first 4 subjects (`DIAG_SUBJECTS`). That is fine as a
   sanity check; the evaluator gives the full numbers.
3. In report (b), job 2 runs with SKIP_CHECKS=1 throughout. The report already says to read job 1's
   gate before using job 2's numbers.
4. With `--store disk` (or `auto` choosing disk), the streaming store and the re-centering originals
   share `$TMPDIR` during the model pass. That needs about 16.7 GB plus the store on node-local
   /scratch. This is documented.

## Verdict

APPROVED. No blockers.
