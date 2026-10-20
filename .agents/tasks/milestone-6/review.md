# Deformable task for the level-4 groupwise model (milestone 6)

The change adds deformable registration as a third task next to rigid and affine. A shared voxel sampler (zero or border boundary) and a half-pixel trilinear ×2 upsample sit under the new `models/diffeo.py`: warps, composition, scaling and squaring with the inverse as exp(−v), group-mean centering, Jacobian and folding. On top of that sit a U-Net `DeformDecoder` with a zero-init float32 SVF head, a two-reader DALI `DeformPipeline` (images + train18 labels), `DeformPostprocess`, masked soft-Dice template and pairwise losses, hard-Dice metrics, a third traced train step, `val_deform` validation with PNGs, and `ckpt/best_deform`. Linear defaults are unchanged: `deform_decoder` defaults to off, `P.TASKS` is still linear, checkpoints without `extra_best` hold the milestone-5 objects, and the only removed test assertions are the two planned NotImplementedError checks.

Watch for: the pairwise composition order phi_i ∘ phi_j⁻¹ is correct, but no unit test would catch it if someone swapped it. The default deform group size inherits `--group-size` (32 at level 4), which the milestone's own memory estimate puts above 48 GB. The DALI deform pipeline and the full suite (plan item 21) have not run yet.

**Verdict**: APPROVED

## High-level view

The diffeo math matches plan section 0, confirmed by reading the code. Fields use border clamping and images/one-hots zero fill. `compose(u_a, u_b) = u_b + warp_field(u_a, u_b)`. Scaling and squaring divides by 2^T and then composes the field with itself T times. The inverse is `integrate_svf(−v)`. Centering subtracts the group mean over axis 0. The Jacobian is a central-difference det(I + ∇u) on the interior. Each of these has a numpy/scipy reference test, and the round-trip error (0.010 voxel max) reproduced in this review. `compose(u_i, u_inv_j)` maps j's coordinates into i's. An oracle run in this review confirmed that direction (pair hard Dice 0.936 vs 0.879 with the order reversed), but the unit tests only use constant or zero fields, where both orders give the same result.

None-gradient routing reuses the m5 static filter. Deform steps leave the whole linear decoder alone, rigid and affine steps leave the deform decoder alone (BN statistics included), and the single optimizer has slots for every variable from `__init__`. The three-task test confirms one trace per task and no slots created later. The only weak spot in this area is `diag_dead_frac`, which is computed on raw conv outputs.

The linear paths are guarded: `resolve_best_metric`, the overfit stop (`overfit_status()` gives the same result for linear-only runs), the overfit `curve`/`val_curve` layout, logged keys and control files all iterate the linear tasks only. Old `config.json` files load through `from_dict` defaults. `fit_summary.json` gains two keys that are `None` without deform.

The deform data path and memory are the operational risks. Reader alignment across epoch reshuffles and the NN residual label warp exist only as code plus an unrun glacier check (l). At default widths a deform step costs about 2 GB per image at level 4, and nothing caps the deform N, which defaults to the linear N.

<details>
<summary>Issues (8)</summary>

1. **No oracle test of the pairwise composition order** — the tests only use constant or zero fields, so a swapped compose order would still pass. Add a test on `synthetic_deform_group` with u = exp(−v_true) and require that `compose(u_i, u_inv_j)` clearly beats the reversed order (0.936 vs 0.879 hard Dice at max_vox 6 here).
2. **Deform group size defaults to the linear N** — `--deform-group-size` falls back to `--group-size` (32 at level 4, estimated 70+ GB). Use a level-dependent default (e.g. 16 at level 4) or warn when the deform N is above 16 at level 4.
3. **DALI DeformPipeline unverified on GPU** — image/label reader alignment and the NN residual warp have never run. Run glacier check (l) and `dryrun_deform` before any deform SLURM job.
4. **Full suite not run, report placeholders open** — `test-log.txt` is missing and report.md still has two `[FILLED IN BY THE FINAL STEP]` markers. Run `tests/run_tests_wsl.sh` after the last edit and fill them in.
5. **Pre-existing flaky linear test** — `TestGroupBehaviour.test_permutation_equivariance` fails about 1 in 8 runs, at HEAD as well (the translation tolerance floor assumes ~10 mm). Expect a possible failure in the full suite: rerun the module, or loosen the floor (e.g. 3e-5) with approval.
6. **`diag_dead_frac` on deform features is near-meaningless** — `dec_feat` is the signed conv output of the D1 pre-activation block, so "≤ 0 everywhere" almost never fires. Compute it after `act(norm(dec_feat))` or drop it from `DEFORM_DIAG_KEYS`.
7. **Eval-mode BN lag in short deform runs** — at BN momentum 0.99 the synthetic overfit stopped with eval template Dice 0.873 vs 0.966 in train mode; the test works around it with `--bn-momentum 0.9`. Watch `val_deform` against `val_deform_trainmode` in the first real debug runs.
8. **test_diffeo needs scipy ≥ 1.6** — `map_coordinates(mode='grid-constant')` does not exist in the container's scipy 1.4.1. Add a version skip as `test_upsample` does.

</details>

<details>
<summary>Details</summary>

### Composition direction has no discriminating test

```
phi_j^-1 : image j -> central        (u_inv_j = exp(-v_j))
phi_i    : central -> image i        (u_i = exp(v_i))
phi_i o phi_j^-1 : image j -> image i  = compose(u_i, u_inv_j) = u_inv_j + u_i(x + u_inv_j(x))
warp(onehot_i, that) = onehot_i in j's space  -> compared with onehot_j
```

`pairwise_displacement`, `pair_dice_loss` and `_deform_errors` all use this order (confirmed), and `test_compose_matches_scipy` checks the `compose` formula itself. What is missing (confirmed) is a test that fails if the order is swapped. `test_perfect_alignment_zero_and_shift` uses u_i = 0, where `compose(0, w) = compose(w, 0) = w`. The synthetic overfit's pair Dice rise (0.774 → 0.950) doesn't settle it either, because for small smooth fields the two orders differ only at second order. A throw-away script for this review (since deleted) used `synthetic_deform_group(max_vox=6)` with u = exp(−v_true). Pair hard Dice was 0.691 at identity, 0.936 with the code's order and 0.879 with the order reversed (fields differ by up to 6.2 voxels); template Dice was 0.978. That setup is ready to become the test.

### Deform group size and memory

`parse_args` sets `deform_group_size = group_size` when the flag is absent (confirmed). At level 4 that is N = 32, so `--task-ratio rigid:1,affine:1,deform:2` with no other flags builds a 32-image deform step with 99 sampled 19-channel pair warps. memory.json measures about 2.0 GB per image at default widths, which extrapolates to 70+ GB (likely OOM on the 48 GB RTX 8000). The report recommends 12–16, but nothing in the code warns or caps it. `main.py` already builds the model with `min(group_size, deform_group_size)`, so a smaller deform default would not affect the linear side.

### DALI alignment is argued, not shown

Two numpy readers get the same pre-permuted lists, the same seed and identical kwargs apart from `files`/`name`. `'buffer'` mode is rejected, and the label residual warp reuses the image's pull matrix with `INTERP_NN` and fill 0. Alignment therefore depends on DALI shuffling identically for identical reader settings (possible risk until shown). Check (l1) tests exactly this: 16 samples over 6 cases across epoch boundaries, with exact label equality per identified case. Check (l2) re-warps the images and labels in TF with the emitted T and compares. Both are written but have not run.

### Deform diagnostics and eval-mode BN

`deform_diagnostics` keeps the linear key `diag_feat_rel_spread`, so collapse detection works unchanged (`test_collapse_diag_deform`). `diag_dead_frac`, though, runs on the D1 `GroupConvBlock` output, which is a signed conv result rather than an activation (confirmed in `models/layers.py`), so it adds no information. The BN finding comes from FEAT-003 (confirmed by its stored overfit summary). With momentum 0.99 the eval-mode moving statistics trailed the quickly adapting decoder: eval template Dice 0.873 vs 0.966 in train mode at step 189. The overfit test passes only with `--bn-momentum 0.9`, while the default stays 0.99. In short real runs, eval-mode `val_deform` may understate progress.

### Test evidence

This review reran the modules the integration logs had not covered, plus the core math: `test_group_layers`, `test_transforms`, `test_curriculum`, `test_precision`, `test_runtime_optim`, `test_upsample`, `test_diffeo`, `test_spatial_transformer` and `test_check_mixed_run`. Result: 116 tests OK, 1 DALI skip. The round-trip (0.0103/0.0108 max), cascade and upsampling numbers match FEAT-001. The integration logs cover everything else: `integ_fast` ran 130 tests with only the pre-existing flaky failure, `integ_slow` ran 70 OK, and `integ_overfit` passed (template/pair 0.971/0.949, neg J 0 %). On compatibility: stage-1 logs show the ast feature_version=(3, 8) check passing, there are no runtime PEP 585/604 annotations, and the library code calls only numpy 1.21 APIs. scipy appears only in tests, where one mode needs ≥ 1.6 (issue 8). On repository state: `git log origin/main..HEAD` is empty, nothing is staged, and no file under `reference/` changed after the milestone started.

Not tested: `DeformPipeline` on GPU, `dryrun_deform` on real data, AMP deform training steps (only the model-level mixed-precision forward is tested), deform training at levels 2/1, and the full suite in a single run.

</details>

<details>
<summary>File map</summary>

- `models/layers.py`: `sample_voxel_coords` (zero/border), `upsample_trilinear2x`, `TrilinearUpsample2x`; `transform_image` delegates to the sampler.
- `models/diffeo.py` (new): warp, compose, SVF integration and inverse, centering, displacement upsampling, Jacobian, folding, pairwise and cascade composition.
- `models/config.py`: `deform_decoder`, `decoder_widths`, `integration_steps`.
- `models/group_net.py`: `DeformDecoder`, `deform` task, gradient checkpointing, parameter counts.
- `data_loading/params.py`, `dali_loader.py`, `data_module.py`, `postprocess.py`: deform constants, `DeformPipeline`, lazy label files, `DeformPostprocess`.
- `runtime/losses.py`, `metrics.py`: soft Dice template/pair losses, smoothness, hard Dice metrics.
- `runtime/fixed_data.py`: label phantom, random SVF, synthetic deform group, `DeformValidationSet`.
- `runtime/run.py`, `args.py`, `checkpoint.py`, `val_png.py`, `main.py`: deform step, validation, overfit status, CLI flags, `ckpt/best_deform`, deform PNG, wiring.
- `tests/*`: new `test_upsample`, `test_diffeo`, `test_deform_data`, `test_deform_overfit`, `test_check_mixed_run`; extended module tests.
- `tests/gpu/*`: check (l), `dryrun_deform`, deform criteria in `check_mixed_run.py`, `measure_deform_memory.py`.
- `.agents/tasks/network-design/design.md`: v3.7.

Full diff: `git diff` plus the untracked files in `git status` (base `cb161f1`).

</details>
