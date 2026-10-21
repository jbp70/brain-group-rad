# LNCC loss for the deformable task: report

Status: implemented and verified locally (not committed, nothing submitted). Plan: `plan.md` (items 0-10). Design: `.agents/tasks/network-design/design.md` v3.9 (6.2 "LNCC", 7, 9).

## What changed

| File | Change |
|---|---|
| `runtime/losses.py` | LNCC primitives: `check_ncc_window`, `box_sum` (separable `tf.nn.conv3d`, ones kernels), `window_count` (numpy constant), `local_ncc` (squared / signed), `foreground`, `ncc_roi`, `masked_mean`, `template_ncc_loss` (LOO template, gradients through it), `pair_ncc_loss`; `NCC_EPS = 1e-7`, `NCC_ROI_THRESHOLD = 1e-3`. `pair_dice_loss(..., disp=None, return_moved=False)`. `deform_loss(..., lambda_dice_tmpl=1, lambda_ncc_tmpl=0, lambda_ncc_pair=0, ncc_window=5)`: original three-term expression first, the template-Dice factor only when != 1, each LNCC term (and its keys) only when its weight > 0; the pair term reuses the pairwise displacement and moved one-hot of the pairwise Dice. Module docstring has the formulas. |
| `data_loading/params.py` | `NCC_WINDOW_DEFAULT = {4: 5, 2: 7, 1: 9}`, `ncc_window_for_level`. |
| `runtime/run.py` | `TrainConfig.lambda_dice_tmpl / lambda_ncc_tmpl / lambda_ncc_pair / ncc_window`; `Trainer` validates the weights (finite, >= 0) and resolves `self.ncc_window`; deform step passes them; `DEFORM_NCC_KEYS` logged as `train_deform/*` when present. Validation: `_deform_errors(..., images=)` adds `ncc_template` [N] / `ncc_pair` [P]; identity cache adds `ncc_template_identity` / `ncc_pair_identity`. Docstrings. |
| `runtime/metrics.py` | docstring only (new per-item keys). |
| `runtime/args.py` | `--lambda-dice-template` (1), `--lambda-ncc-template` (0), `--lambda-ncc-pair` (0), `--ncc-window` (None = per level); validation (finite >= 0; window odd, >= 3, <= min level dim; deform trained with all 4 similarity weights 0 -> error); passed to `TrainConfig`; docstring. |
| `tests/gpu/measure_deform_memory.py` | `--lambda-ncc-template / --lambda-ncc-pair` (default 0), recorded per result and in the config key. |
| `slurm/m7_three_task.sh` | comment only: the LNCC A/B submit line (uses the existing `EXTRA_ARGS`). |
| Tests | `tests/test_runtime_losses_metrics.py` (`TestLNCC` 9 tests, `TestDeformLosses` +3), `tests/test_trainer.py` (`TestDeformUnchanged` golden, `TestDeformNCC` 3 tests, `TestDeformValidation` extended), `tests/test_runtime_infra.py` (`test_ncc_flags`), `tests/test_main.py` (key list, LNCC-off checks, `test_synthetic_deform_ncc_dry_run`). New fixture `tests/fixtures/lncc_deform_golden.json`. |
| Task folder | `run_some.sh`, `py38_check.sh`, `measure_memory.sh` + `memory.json`, `grad_probe.py/.sh` + `grad_probe.json`, `overfit_compare.sh`, `overfit_table.py`, `overfit_{A,B,C}.json`, `overfit_table.json`, `test-log.txt`. |

Defaults keep every existing run bit-identical (LNCC weights 0, template-Dice weight 1, terms not built). Validation LNCC metrics are always computed (cheap; new log keys only).

## Reference adaptation (`reference/.../runtime/losses.py`, local only, nothing copied)

Reference: `fast_nlcc_metric` / `Joint_NLCC_Grad_Loss`: 9³ ones Keras `Conv3D` (groups 5) on [I, J, I², J², IJ], sums normalised by k³, `cc = cross² / (I_var J_var + 1e-7)`, mean over voxels with I != 0 or J != 0, loss −cc, template = inclusive mean of the moved images with gradients, bending-energy smoothness.

- Kept: squared cc (bounded [0, 1], no sqrt), ε = 1e-7, one filter pass over the 5 stacked quantities, mean template with gradients flowing through it.
- Changed, and why:
  - Local MEANS with the true per-voxel window count instead of raw sums / k³. Border windows are normalised correctly, and ε has a fixed meaning in intensity² units. The reference's 1e-7 on sums is ≈ 2e-13 in means units, effectively no guard. Variances are clamped at 0 (float32 cancellation).
  - Separable functional `tf.nn.conv3d` instead of a dense Keras Conv3D: 15 vs 125 MACs per voxel at k = 5, and no layer variables, so checkpoints, warm start and `n_vars_updated` are unchanged.
  - Window per level (5 / 7 / 9 at levels 4 / 2 / 1) instead of a fixed 9. Nine voxels at 4 mm would be 36 mm on a 40×56×48 grid, nearly global.
  - Leave-one-out template instead of the inclusive mean. The self term gives a floor of ≈ 1/N even for unrelated images (measured: N = 4 independent noise has inclusive mean > 0.15 vs LOO < 0.05, `test_leave_one_out`).
  - Label ROI (foreground dilated by k/2) instead of nonzero voxels. The DALI intensity augmentation adds noise to the zero background, so "nonzero" selects almost the whole volume.
  - Loss 1 − cc instead of −cc: O(1) and >= 0, like 1 − Dice. The logged value is the cc itself.
  - Added a pairwise term on the pairwise-Dice pairs (shared displacements). The smoothness term stays ours (first order by default).

## Verification

### Golden guard (made before any runtime change)

`TestDeformUnchanged`: 8 three-task steps (rigid:1, affine:1, deform:2, flips on, SVF head N(0, 1e-3)). It records tasks, losses, deform terms, and every variable's (sum, max |v|).

- First attempt without determinism: reruns differed (max |diff| 114 and 276 in variable sums). GPU segment-sum accumulation order in the warp-gather gradients is nondeterministic, and Adam amplifies the noise.
- Fix: the run is wrapped in `_OpDeterminism` (enable / restore `tf_config.*_op_determinism`, local to the test), tolerance 0. With determinism, two separate processes gave bit-identical JSON.
- The fixture was generated on the unmodified runtime code (b35bb54, `git status` clean for runtime/models/data_loading). It then passed twice with max |diff| 0 before any change, and again (max |diff| 0) after all changes. `TestLinearOnlyUnchanged` also stays at 0.

### Unit tests (targeted runs, all OK)

- `tests.test_runtime_losses_metrics` TestLNCC + TestDeformLosses: 17 OK.
  - box sums and local means match `scipy.ndimage.uniform_filter` (float64, true-count normalisation) to rtol 1e-5 at k = 3 and 5, borders included, multi-channel too.
  - `local_ncc` squared and signed match the numpy reference to rtol 1e-4 / atol 1e-5.
  - a·I + b and −I give cc = 1 (signed ±1) on var > 1e-4 voxels; self LNCC > 1 − 1e-4; independent noise < 0.05.
  - Flat blocks and zero background: finite map, cc ≈ 0 where both are flat, finite non-zero gradients w.r.t. images and u through `diffeo.warp`. All-zero images give loss 1 with finite gradients.
  - Noise outside the 2·(k/2)-dilated foreground leaves the loss equal (rtol 1e-6). Foreground touching the border is finite.
  - LOO vs inclusive (see above). Identical images give LOO cc = 1.
  - The template gradient differs from the stop-gradient variant.
  - Pair shift by one voxel: the aligned u_inv gives > 0.999, identity < 0.99.
  - float16 in gives float32 out. Windows 4 / 1 / 11 (> dim) / 2.5 raise ValueError.
  - Deform-loss wiring:
    - Defaults are bit-identical to the explicit 1/0/0 call, with the same key set.
    - A batch without `images` works when LNCC is off and raises when it is on.
    - Total = λ-weighted sum (rtol 1e-6), and `ncc_*_loss = 1 − ncc_*`.
    - The pair / template LNCC with the shared displacement equals a separate recomputation.
    - λ_dt = 0 gives the gradient of pdice + 0.5 smooth (rtol 1e-5).
- `tests.test_trainer` TestDeformNCC (3), TestDeformUnchanged, TestLinearOnlyUnchanged, TestDeformStep, TestThreeTasks, TestDeformValidation, TestTaskLossWeightDeform, plus `tests.test_runtime_infra` and `tests.test_main.TestDeformMain`: 80 tests OK, 1 skip (dllogger).
  - LNCC step: finite, keys in [0, 1], same `n_vars_updated` and variable count as Dice-only, trace once, and a different grad norm from the same weights.
  - LNCC-only config is finite and still reports `soft_dice_tmpl`. Invalid λ (−1, nan, inf) and windows 4 / 41 raise.
  - The 8-step three-task fit traces once per task and logs `train_deform/ncc_*`.
  - Validation: the 4 new keys are finite, the identity model equals the identity baseline (places 10), and the baselines are unchanged after `fit(2)`.
  - CLI: defaults 1/0/0/5, windows 7 / 9 at levels 2 / 1, round trip into TrainConfig, bad values exit.
  - main.py dry run with 1/1 logs `train_deform/ncc_tmpl`, `ncc_pair`, `ncc_tmpl_loss` and `val_deform/ncc_template`, `ncc_pair`, `ncc_template_identity`. Dice-only runs log no `train_deform/ncc_*` but still log `val_deform/ncc_template`.

### Python 3.8 compatibility

`py38_check.sh` (bgr_preproc38, Python 3.8.20) passes:
- py_compile and `ast.parse(feature_version=(3, 8))` on the changed .py files;
- `runtime.args` imports without TF;
- `main.py --help` lists the 4 new flags;
- `bash -n` on `slurm/m7_three_task.sh`, `tests/run_tests_wsl.sh` and the task scripts.

### Full suite

`tests/run_tests_wsl.sh`, run once at the end (`test-log.txt`):
- Stage 1 (py3.8) OK: py_compile OK, ast (3, 8) OK on 51 files, TF-free imports, `--help`, `bash -n`.
- Stage 2: **Ran 313 tests in 6524 s, OK (skipped=4)**, `unittest exit code: 0`. The baseline was 295 OK / 4 skips, so this adds 18 new tests.
- Both golden tests ran inside the full process at max |diff| 0 (`[golden deform]` and `[golden]`).

## Memory (item 7, `memory.json`)

Local GTX 1080, 6000 MB TF limit, level 4, 5 measured steps, TF allocator peak:

| Widths | N | Peak LNCC off | Peak LNCC 1/1 | Δ | Step off → on |
|---|---|---|---|---|---|
| narrow | 2 | 2095 MB | 2160 MB | +65 | 710 → 754 ms (+6 %) |
| narrow | 3 | 3188 MB | 3171 MB | −17 | 948 → 1026 ms (+8 %) |
| narrow | 4 | 4087 MB | 4047 MB | −41 | 1136 → 1374 ms (+21 %) |
| default | 2 | 4668 MB | 4619 MB | −49 | 1873 → 1932 ms (+3 %) |

There is no measurable per-image slope: the differences are within run-to-run noise. The peak is the SVF-integration backward, and the LNCC activations are small. Estimate, not measured: at default widths N = 16 (24 pairs) the LNCC stores ≈ 3 tensors of [5N] volumes (≈ 0.1 GB) for the template term and ≈ 3 of [5P] (≈ 0.15 GB) for the pairs, ≤ 0.5 GB worst case on top of 33.8 GB. That is well under the 1 GB target, so N = 16 fits the 40 GB A100 and the 48 GB cards. The step time grows by 3-21 % locally, most at N = 4 narrow, where the deform step is cheapest relative to the 5-quantity filter. On the cluster, expect roughly +10 % deform step time; the smoke run measures it.

## Gradient magnitude (item 8a, `grad_probe.json`)

Synthetic group N = 4, narrow widths, after 50 Dice-only steps (soft template Dice 0.856, u_max 3.1 vox). Means over 8 postprocessed batches (flips on, 1 pair):

| Term / objective | value | total grad norm | encoder grad norm |
|---|---|---|---|
| tmpl (Dice) | 0.146 | 0.206 | 0.126 |
| pdice | 0.142 | 0.287 | 0.187 |
| smooth (unweighted) | 0.0053 | 0.070 | 0.035 |
| ncc_tmpl_loss | 0.242 | 0.368 | 0.202 |
| ncc_pair_loss | 0.266 | 0.625 | 0.340 |
| Dice-only (tmpl + pdice + 0.5 smooth) | 0.290 | 0.453 | 0.294 |
| Dice + LNCC 1/1 | 0.799 | 1.252 | 0.687 |
| LNCC-only (0.5 smooth + LNCC) | 0.511 | 0.949 | 0.518 |

- Each LNCC term has ≈ 1.5-2× the gradient of a Dice term. Dice+LNCC raises the deform total norm ≈ 2.8× and the encoder norm ≈ 2.3×.
- Milestone-7 context: the deform step total norm on real data was ~0.08-0.5. With LNCC 1/1 expect roughly 0.2-1.4, still far below the global clip of 12, so deform steps stay unclipped.
- The applied deform encoder gradient (`train_deform/diag_enc_grad_norm_eff`, 0.002-0.008 in the m7 smoke) should grow ≈ 2-3×. That is still 2-3 orders below the clipped linear steps (rigid ≈ 3.4), so LNCC alone does not fix the linear/deform imbalance.
- Synthetic phantom only. The real-T1 ratio is unknown and should be read from the cluster smoke run.

## Synthetic overfit A/B/C (item 8b, `overfit_{A,B,C}.json`, `overfit_table.json`)

Flags of `tests/test_deform_overfit.py` (N = 4, narrow, Adam 3e-4 constant, BN momentum 0.9, seed 0, no flips in overfit mode) with a fixed 600-step budget (`--overfit-dice-threshold 1.01`, never stops early). Eval mode, final step:

| Run | template hard Dice | pair hard Dice | val ncc_template | val ncc_pair | neg J % | train grad_norm mean / max | median step | GPU peak |
|---|---|---|---|---|---|---|---|---|
| identity | 0.8635 | 0.7742 | 0.5108 | 0.4408 | 0 | | | |
| A Dice-only | 0.9768 | 0.9607 | 0.7843 | 0.7554 | 0 | 0.53 / 0.63 | 1222 ms | 4191 MB |
| B Dice + LNCC 1/1 | 0.9771 | 0.9537 | 0.8280 | 0.8235 | 0 | 1.00 / 1.77 | 1362 ms | 4222 MB |
| C LNCC-only | 0.9628 | 0.9343 | 0.8281 | 0.8246 | 0 | 0.66 / 1.49 | 1272 ms | 4264 MB |

Validation curves (step: template Dice / pair Dice / ncc_template):
- A: 100: 0.955 / 0.925 / 0.785, then 300: 0.970 / 0.950 / 0.793, then 600: 0.977 / 0.961 / 0.784. A's LNCC peaks at step 200 and drifts down while Dice keeps rising.
- B: 100: 0.954 / 0.923 / 0.803, then 300: 0.972 / 0.946 / 0.822, then 600: 0.977 / 0.954 / 0.828.
- C: 100: 0.945 / 0.912 / 0.801, then 300: 0.957 / 0.926 / 0.820, then 600: 0.963 / 0.934 / 0.828.

Against the expectations:
- A is close to the milestone-6 reference (0.971 / 0.950 at its stop step 306). It is slightly higher here because the run continues to 600 steps.
- B's Dice is within 0.02 of A (template +0.0003, pair −0.007) with higher LNCC (+0.044 template, +0.068 pair).
- C beats identity by 0.10 template / 0.16 pair Dice using intensities only, with no folding anywhere.
- The train-mode window soft template Dice at step 600 was A 0.922, B 0.896, C 0.871. With LNCC the optimiser trades a little soft label overlap for intensity alignment; the eval hard Dice barely moves.

How meaningful LNCC is on these phantoms: the image is 0.7 × a per-label mean intensity + 0.3 × smooth texture, normalised. So intensity edges coincide with label boundaries, and LNCC is close to a smoothed label-boundary similarity plus texture. That is why LNCC-only recovers most of the Dice gain. On real T1, LNCC also sees intra-label structure (cortical folding, WM/GM contrast within merged labels), so the effect there can differ in either direction. No real level-4 data exists locally (searched `d:\Research\Brain_Registration_Projects` for `image_4.npy`: none; the export is on the cluster), so the real-data check is the cluster A/B below.

## Recommended cluster A/B (not submitted)

Both arms use the m7 recipe of `slurm/m7_three_task.sh`: warm start from the m5 checkpoint, rigid:1, affine:1, deform:2, N 32 / deform N 16, G 45°/10 mm, 12k steps. The script already has `EXTRA_ARGS`, so no script change was needed (only a comment line).

- Smoke first (~20-25 min; check finite steps, `train_deform/ncc_*`, `val_deform/ncc_*` + identity, `gpu_peak_mb` < 46 GB, deform step time vs the m7 smoke's ~2.6 s mean):
  `RUN_NAME=m7_lncc_smoke STEPS=200 VAL_EVERY=100 WARMUP=50 EXTRA_ARGS="--lambda-ncc-template 1 --lambda-ncc-pair 1" sbatch --job-name=m7lnccsmoke --time=00:45:00 --mem=64G slurm/m7_three_task.sh`
- Dice+LNCC arm (14 h to allow for the ~10 % slower deform step):
  `RUN_NAME=m7_three_task_lncc EXTRA_ARGS="--lambda-ncc-template 1 --lambda-ncc-pair 1" sbatch --job-name=m7lncc --time=14:00:00 --mem=64G slurm/m7_three_task.sh`
- Dice-only arm: the running job 10493712 (`m7_three_task_long`, same recipe, no weights) is the Dice control for the Dice / folding / linear metrics. It runs b35bb54, so it logs no `val_deform/ncc_*`. For an LNCC-metric comparison either re-validate its `ckpt/best_deform` with the new code, or run a fresh control: `RUN_NAME=m7_three_task_dice sbatch --job-name=m7dice --time=14:00:00 --mem=64G slurm/m7_three_task.sh`.
- Compare `val_deform/{template_dice, pair_dice, ncc_template, ncc_pair, neg_jac_pct}` vs their `_identity` values, `val_rigid/affine` (task interference), and `train_deform/grad_norm` / `diag_enc_grad_norm_eff`.
- Memory: N = 16 + LNCC ≈ 34.3 GB (estimate), so it fits RTX 8000 / A6000 / A100 40 GB.
- Two-job limit: two jobs are active now (10493712, 10493723), so submit only after one finishes and with Jay's approval.
