# Augmentation curriculum and shared global transform G via a re-read control file

The change adds two opt-in augmentation features for the linear DALI pipelines. The first is a linear ramp of the per-image rotation/translation/scale/shear half-widths from start values to the existing `--<task>-*` values, which become the end values. The second is one rigid G per group, resampled every K steps and composed last, so the group barycenter moves away from MNI. Both reach DALI through one float32[12] control file per task (`<run_dir>/control/<task>.npy`). The trainer rewrites the file atomically each step, and a second `fn.readers.numpy` re-reads it every batch with the reference project's scale-reader settings. The pipeline emits G_dali and a version counter so lag and mixed batches are observable. Targets need no post-processing change because DALI already emits G·Rt_i (rigid) and G·Rt_i·S_i (affine). Watch for: the DALI graph (re-read behaviour, G composition) has not run anywhere yet, because checks h-k and `dryrun_ctl` are pending on glacier. The meaning of the file's G angle sign is unresolved (check i accepts either sign). Two sub-checks in h-k prove nothing or gate nothing.

**Verdict**: APPROVED

## High-level view

Composition matches the requested contract. `build_global_linear_matrices` produces T = C+·Tg·Rg·C-·C+·Tr·R·K·C- (G applied last, about the same center), the rigid target G·Rt_i, and a pull matrix that inverts in the right order, all in one `warp_affine`. The numpy mirror in `params.linear_matrices_np` matches. Flips stay F·T·F on the emitted T and target, because `LinearPostprocess` receives only the base outputs.

The control reader uses the same arguments as the reference `_make_scale_reader`: a single absolute file, unsharded, `random_shuffle=False`, `shuffle_after_epoch=False`, `read_ahead=False`, `dont_use_mmap=True`, `pad_last_batch=True`. Writes go to a `mkstemp` file in the same folder, followed by fsync and `os.replace`, with a retry on Windows. The reference only relied on a re-read per epoch. Per-batch re-reading of a one-record reader is plausible but unverified until check h runs.

Defaults OFF leave the old path byte-for-byte. With `control_dir=None` there are no writers, `control_file=None` selects the old `build_linear_matrices` graph and 3/2-output signature, the trainer's `controls` is `{}`, and `split_batch` returns the full tuple. Existing range flags keep their meaning, and start ≤ end is only enforced with `--curriculum`. Curriculum and G are pure functions of the step, and the sampler uses `default_rng([seed, step // K])`. Which control state a given batch sees still depends on prefetch timing, though.

`val_<task>_g` sets exist only with `--global-transform`. They are seeded by (val_seed, task, group, 303) and built TF-side with `global_matrix_mm`, so they don't depend on the DALI angle sign. Python 3.8 / numpy 1.21 compatibility holds (`typing` annotations only; `default_rng` with a list seed and `np.ptp` exist in 1.21). The local log `test-log-curriculum.txt` started 22:32 after the newest source edit (20:35), and records 178 tests OK (3 DALI/GPU skips) plus the py3.8 AST check.

<details>
<summary>Issues (6)</summary>

1. **DALI graph unexercised** — no DALI run has executed the control reader, the scalar DataNode arithmetic or G composition yet. Run `run_verify_glacier.sh <gpu> 1800 verify h,i,j,k` and `run_overfit_glacier.sh <gpu> 1800 dryrun_ctl` before any training run uses the flags.
2. **G angle sign ambiguity** — reversing the TF axis to DALI order is a reflection, so the emitted G is likely R(axis, −angle) in TF space. Check i accepts both signs. Once glacier reports which mirror sign matched, negate the angle in `control_global_params` if needed and make check i strict on the signed axis and angle.
3. **Tautological recompose check** — in check i, `g_mm @ (g_inv @ tgt) @ (inv(tgt) @ t_mm) == t_mm` holds for any output. Drop it, or replace it with an independent reconstruction.
4. **Barycenter shift not gated** — check j computes `barycenter_equivariance_max_abs` but leaves it out of `task_ok`. Add a tolerance so the motivating property (μ* → μ*·G⁻¹) is actually tested.
5. **Resume starts from step-0 control values** — `main` writes `values_at(0)` before the checkpoint restore, so the first prefetched batch(es) after a resume use the curriculum start ranges. Write the resumed step's values before the pipelines start, or accept and keep documenting it.
6. **Batch-level non-reproducibility with controls on** — which file version a batch reads depends on the writer/prefetch race, so two runs with the same seed can differ. Keep `control_version_lag` logged, and note in the design that bitwise reproducibility only holds with both features off.

</details>

<details><summary>Details</summary>

### DALI composition and the G angle sign

The per-image and G factors are combined in one `fn.transforms.combine(c_minus, k, rot, tr, g_rot, g_tr, c_plus)`, which (first = rightmost) gives C+·Tg·Rg·Tr·R·K·C-. Because C-·C+ = I, that equals G·Rt_i·S_i with G = C+·Tg·Rg·C-. The rigid target drops `k`, and the pull matrix reverses every factor with negated parameters. This is consistent with the existing `build_linear_matrices` convention.

`control_global_params` reverses the TF-order axis `[x, y, z]` into DALI order and passes the file's angle unchanged. Reversing x and z is an improper permutation P, and P·R(a, θ)·P⁻¹ = R(P·a, −θ). So if DALI's `transforms.rotation` has the same handedness as `global_matrix_mm`, the emitted G in TF mm is R(axis, −angle) (likely). Check i deliberately sidesteps this: it compares `|dot(axis)|`, an unsigned angle, and the numpy mirror under both signs. Training is unaffected, since the emitted matrix is what the targets use and the angle distribution is symmetric. The file's documented semantics and the `control_g_angle_deg` log can still disagree with the actual G, and the training G would then differ from `val_<task>_g`'s TF-side `global_matrix_mm` convention. The `G_vs_numpy_mirror_max_abs` dict from the first glacier run will settle it.

### Re-read behaviour versus the reference

The kwargs match the reference scale reader exactly (confirmed against `reference/.../dali_loader.py` lines 201-219), with `os.path.abspath` on the path. The reference refreshed per epoch. Here the reader has one record, so each batch is a new epoch and should reopen the file. That is the behaviour check h measures: the version must change within three batches of a rewrite, the v1 batches must stay within 5°, and the v2 batches must exceed 10° and stay within 40°. This check is genuine, because it uses the emitted version and the measured rotation of the emitted Rt, not the pipeline's own parameters. 

### Verification checks h-k

Check i is mostly a genuine decomposition. G_dali is converted to TF mm and compared against the written translation, angle magnitude and axis. Rt_i = G⁻¹·target must be orthonormal within 20° / 5 mm. S_i = target⁻¹·T must have zero translation, a zero lower triangle and a diagonal within the scale range. That confirms the rigid target excludes exactly the nuisance. One sub-check is circular (confirmed):

```python
rt  = g_inv @ tgt
s_i = np.linalg.inv(tgt) @ t_mm
recompose = np.abs(g_mm[None] @ rt @ s_i - t_mm).max()   # identically ~0
```

In check j, centrality, pair consistency and the DALI-vs-TF render equality are gated. `bary_equiv`, the property that motivates G in the first place, is computed and reported but not part of `task_ok` (confirmed). Check k tests the curriculum start magnitudes independently via QR on the emitted matrices, with a non-collapse lower bound.

### Resume and determinism

`TaskControl.values_at(step)` and `GlobalTransformSampler.params(step)` are pure, so a resumed trainer writes the same values as an uninterrupted one from its first step on. `main._build_dali_data` writes `values_at(0)` before the checkpoint is restored, though. The pipeline's first prefetched batch can therefore use the curriculum start ranges and identity-step G (confirmed; documented in the report). More generally, the trainer writes before `next()`, but the prefetched batch was produced during the previous step. The version a batch reads depends on thread timing, so batches aren't bitwise reproducible across runs when either feature is on (likely). Targets stay correct because they come from the emitted matrices.

### Test coverage

Covered locally: curriculum endpoints and validation, sampler determinism and the K interval, the control-file layout round trip, atomic writes and the no-op rule, the numpy mirror with G, post-processing targets with G against an independent reference, the determinism of `val_<task>_g` and its shifted barycenter, CLI defaults and validation, and the data-module initial files and signature. Not tested anywhere yet: the DALI control reader re-read, DataNode scalar arithmetic in 1.18, G composition inside DALI, and the end-to-end DALIDataset + trainer run with both flags on. These all wait on glacier (checks h-k, `test_pipeline_with_control_file`, `dryrun_ctl`).

</details>

<details>
<summary>File map</summary>

- `runtime/curriculum.py` (new): LinearCurriculum, GlobalTransformSampler, atomic writer, ControlWriter, TaskControl.
- `data_loading/params.py`: control-file layout, `control_vector` / `parse_control`, G in the numpy mirror.
- `data_loading/dali_loader.py`: control reader, control-driven params, `build_global_linear_matrices`, extra outputs.
- `data_loading/data_module.py`: `control_dir` / `control_init`, initial file writes, `control_file(task)`.
- `data_loading/postprocess.py`: docstring for targets with G.
- `runtime/args.py`: curriculum / global-transform flags, validation, `task_controls_from_args`.
- `runtime/fixed_data.py`: `val_<task>_g` sets.
- `runtime/run.py`: per-step control writes, `split_batch`, control metrics, val sets keyed by name.
- `main.py`: wiring of controls and G validation sets.
- `tests/test_curriculum.py` (new), `tests/test_trainer.py`: local tests.
- `tests/gpu/verify_dali_pipeline.py`: checks h-k; `run_verify_glacier.sh` / `run_overfit_glacier.sh`: check subset and `dryrun_ctl` mode.
- `.agents/...` design v3.5 and report section.

Full diff: `git diff` in the workspace (uncommitted).

</details>
