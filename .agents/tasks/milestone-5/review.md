# Mixed rigid + affine training, affine metrics, G sign fix and curriculum follow-ups

Milestone 5 makes rigid and affine train interleaved with one shared Adam optimizer. It adds QR-based affine validation metrics (scale, shear, rotation factor) and a combined best-checkpoint metric `val/mean_pair_rms_mm` for multi-task runs. It also closes review-curriculum issues 2-5: the G angle sign, check i's tautology, check j's ungated barycenter, and step-0 control values on resume. On top of that it adds G warm-up (`--global-start-step` / `--global-ramp-steps`) and warm start (`--init-weights`).

The recorded evidence is complete. `test-log.txt` was written at 10:36:47, after the last code edit at 04:13. It shows 194 tests OK with 3 DALI/dllogger skips. The affine overfit reached 0.606 mm (eval) and the mixed overfit reached 0.318 / 0.605 mm (eval), so both meet the < 1 mm target.

Watch for: the DALI side of the G sign fix is derived analytically and has not run on DALI yet. Glacier check i (now strict) has to pass before G is used in a real run (likely). Also, resuming a run that started before this change with more than one task silently switches its best metric (likely).

**Verdict**: APPROVED

## High-level view

The mixed schedule needed almost no new machinery. The optimizer is built on all trainable variables in `Trainer.__init__`, and rigid steps drop the scale/shear head's None gradients when the step is traced. So the first affine step after a run of rigid steps creates no slots and does not retrace. The new trainer test covers this directly. It starts with 3 rigid steps and runs 30 steps, with flips off and with flips, curriculum and G on. In both cases it asserts one trace per task, a constant optimizer-variable count of 2 × trainable, the head unchanged until the first affine step, and 2 fewer updated variables on rigid steps.

The affine metrics factor each linear part as Q·U·D, with the diagonal sign-normalized. In TF order, the augmentation's DALI-order lower-triangular K reverses to exactly that upper-triangular U·D, so the reported shear entries are the augmentation's (a, b, c). The tests build matrices from known factors and check them against those factors, and they brute-force the pair loop. `val/mean_pair_rms_mm` is the mean over the G = I `val_<task>` sets and is emitted only when every trained task was validated. It becomes the default `--best-metric` only when more than one task is trained. Single-task runs resolve to `val_<task>/pair_rms_mm` exactly as before.

The G sign derivation is correct. P (the x↔z swap) is improper, and Q R(u,θ) Qᵀ = R(det Q · Q u, θ), so P R(Pa,θ) P = R(a,−θ). Negating the angle in `control_global_params` makes the emitted G equal `global_matrix_mm` of the logged values. Check i now requires a positive signed axis dot, the angle within 0.01°, and an independent comparison against `global_matrix_mm`. The tautological recompose sub-check is gone, and check j gates the barycenter equivariance at 1e-6 against a measured 1.7e-14.

On resume, `main` reads the step from the checkpoint before building the DALI data, so the control files start at the resumed step. Warm start loads only the model variables, refuses a ModelConfig mismatch, and is not allowed when the run dir already has a checkpoint and `--resume` is on.

<details>
<summary>Issues (4)</summary>

1. **G sign not yet confirmed on DALI** (likely) — the fix depends on DALI 1.18's rotation being right-handed in its own axis order. Run `run_verify_glacier.sh <gpu> 1800 verify h,i,j,k` before any run uses `--global-transform`. If only the sign fails, flip `G_DALI_ANGLE_SIGN`.
2. **Best metric switches on resume of an older multi-task run** (likely) — the restored `best_metric` value was tracked on `val_rigid/pair_rms_mm`, but new validations are compared on `val/mean_pair_rms_mm`. Pass `--best-metric val_rigid/pair_rms_mm` when resuming such a run, or note it in the design.
3. **Overfit window now skips non-finite steps** (confirmed) — the single-task stop criterion used to need the last 10 steps all finite. It now averages the last 10 finite steps. This is minor; note it, or keep the old strictness for the single-task case.
4. **G changes every step during the ramp** (confirmed) — during [S, S+R] the block's angle is scaled by f(step), so the control file is rewritten every step and G is not constant within a K-block. Document it, or scale by f at the block's first step.

</details>

<details>
<summary>Details</summary>

### Optimizer slots and None-gradient filtering across interleaved tasks

The core mechanism (`O.build_optimizer` on `model.trainable_variables` in `__init__`, and the static `g is not None` filter in the step) predates this diff. Milestone 5 relies on it, documents it in the module docstring and design 5.2, and finally tests the case that motivated it. `TestMixedTasks.test_interleaving_starting_rigid` seeds a schedule that opens with 3 rigid steps. It checks `opt.variables()` before and after (count unchanged), the head's kernel before and after the rigid prefix, `n_vars_updated = len(tvars) − 2` on rigid steps, and `trace_counts == {rigid: 1, affine: 1}`. It does all of this with controls and flips both off and on. The step order is asserted to equal `TaskSchedule.draw(s)`, which also covers resume determinism of the task sequence.

The full-width mixed overfit shows interference at a constant LR. With constant Adam 3e-4, rigid eval RMS at the stopping step was 1.14 mm, and both tasks oscillated between about 0.5 and 2.4 mm from step 1250 on. The suite test uses cosine decay to pass. The report discloses this, and real runs use the cosine default. Still, the per-task val curves of the first real mixed run are where task interference would show.

### Affine metrics

`qr_factors` loops per matrix, which is correct for numpy 1.21 (no batched QR). The sign normalization keeps diag(K) > 0, so Q is a rotation whenever det L > 0. That holds for every augmented and predicted matrix here: the model's K = exp(...) diagonal is positive. The shear convention matches the augmentation. `scale_shear_rows` builds K_dali = [[dz,0,0],[c·dz,dy,0],[b·dz,a·dy,dx]], and reversing the axes gives K_tf = [[dx,a·dy,b·dz],[0,dy,c·dz],[0,0,dz]] = U·D with U₀₁ = a, U₀₂ = b, U₁₂ = c. These match `qr_factors`' K01/K11, K02/K22 and K12/K22. The pair errors use the same pairing as `pairwise_errors` (inv(A_j)·A_i vs T_j·inv(T_i)), and the test cross-checks them by brute force. The affine errors are computed in eval mode only, which is documented.

### Combined best metric and resume

`resolve_best_metric` and `parse_args` agree, and resolving twice is idempotent. When a run resumes, the restored `RunCheckpoint.best_metric` value is compared against whatever key the new code resolves. A multi-task run started before this change therefore compares mean-pair values against a stored `val_rigid/pair_rms_mm` best. Mean pair RMS is usually higher than rigid alone, because affine is harder, so the best checkpoint may stop updating for a while. This only affects such resumes. No such run is documented, so it is not blocking.

### G sign derivation and the strict check i

```
file (TF order): axis a, angle θ       DALI: axis P a, angle s·θ
M_tf = P · R(P a, s·θ) · P = R(det P · P P a, s·θ) = R(−a, s·θ) = R(a, −s·θ)
s = G_DALI_ANGLE_SIGN = −1  ⇒  M_tf = R(a, +θ) = global_matrix_mm(a, θ, t)
```

`test_g_angle_sign` confirms that the numpy mirror plus `dali_voxel_to_tf_mm` gives R(a, −θ) without the fix and `global_matrix_mm` with it. The numpy mirror encodes the claim being tested, though: `_rotation_np` is assumed to be DALI's matrix. The only independent evidence is check i on glacier. Its `G_vs_global_matrix_mm_max_abs` and the signed axis dot `rotation_axis(g_mm)·G_AXIS` (unambiguous for 0 < θ = 30° < 180°) are the real test, and the mirror comparison is a second, consistent path. The tautological `recompose_max_abs` is gone. That leaves the render equality in j as the independent check on T. The logged `control_g_angle_deg` is the file value, so after the fix it matches the emitted G.

### Check j barycenter gate

`bary_equiv < args.tol_bary` (1e-6) is now part of `task_ok`. `measure_bary.py` reproduces check j's computation: float32-rounded inputs, float64 `tfm.barycenter` (since `to_float` keeps float64), and 10 iterations. It reports a maximum of 1.7e-14, so the tolerance leaves about eight orders of magnitude of margin. The gate still catches a real equivariance break, which would be orders of magnitude larger.

### G warm-up semantics

`params(step)` scales the block's draw by `fraction(step)`, not by the fraction at the block start. During the ramp, the angle and translation change every step while the axis is held for K steps. `ControlWriter` writes whenever the values change, so the file is rewritten (and its version bumped) every step until S+R. It also means that "resampled every K steps" holds only for the axis during the ramp. Defaults reproduce the old sampler exactly (tested). Before S, `val_<task>_g` still uses full-range G, so its numbers will look poor by design until the ramp completes.

### Overfit window change

`window_center_rms_by_task` walks the history backwards and collects the last `overfit_window` finite steps of each task. In the single-task case this differs from the old `history[-w:]` filter. A non-finite step inside the window used to block the criterion (it returned None), but now it is skipped. The non-finite patience guard still aborts runs that go bad. The overfit criterion is slightly more permissive than before for rigid-only runs.

### Glacier tooling and compatibility

`dryrun_mixed` exists, writes under `scratch/m5`, and runs `check_mixed_run.py` from the repo root (the script already `cd`s to `$REPO`). The checker is stdlib-only and is exercised locally against the mixed overfit run dir. Check names a-k are unchanged. The report states the glacier commands were not run, and nothing in the task folder references SSH or the cluster. HEAD equals origin/main (the milestone-4 commit), and all changes are uncommitted. The added code uses no 3.9+ syntax or stdlib, and `default_rng([seed, k])` and the per-matrix `np.linalg.qr` exist in numpy 1.21.

</details>

<details>
<summary>File map</summary>

`runtime/metrics.py` adds the QR factors, affine errors and `mean_pair_rms`. `runtime/run.py` adds best-metric resolution, affine validation errors, the combined metric, per-task overfit windows and G warm-up logging. `runtime/args.py` adds the auto best-metric default, the G warm-up flags and `--init-weights`. `runtime/curriculum.py` adds the sampler warm-up. `runtime/checkpoint.py` adds `peek_last_step`, `resolve_checkpoint` and `load_model_weights`. `main.py` adds per-task overfit groups, resume control values, warm start, `fit_summary.json` and per-task overfit summaries. `data_loading/params.py` and `data_loading/dali_loader.py` add the G angle sign. `tests/gpu/verify_dali_pipeline.py` makes check i strict and gates check j. `tests/gpu/run_overfit_glacier.sh` adds `dryrun_mixed`. `tests/gpu/check_mixed_run.py` is new. The tests for metrics, trainer, infra, curriculum and main were extended, and the design doc moved to v3.6. Full diff: `git diff` against `a5da557`.

</details>
