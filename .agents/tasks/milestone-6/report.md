# Milestone 6 report: deformable pipeline, U-Net decoder, diffeo, deformable losses (level 4)

Outcome: the deformable task is in. The model has a U-Net decoder that predicts a group-centered SVF. Scaling and squaring (T = 7) gives the deformation, and exp(−v) gives its inverse. Training uses the template + pairwise soft Dice and smoothness losses, a DALI image+label pipeline, deform validation with PNGs, and a third train step that runs alongside rigid and affine with one optimizer. Locally the diffeo round trip is accurate to 0.01 voxel. The synthetic overfit reaches eval template / pair hard Dice 0.971 / 0.950 (identity 0.864 / 0.774) with no folding, and every task traces once in a rigid:1,affine:1,deform:2 run. Linear-only defaults, variables and checkpoints are unchanged. Local full suite: **284 tests OK, 4 skips** (3 DALI, 1 dllogger), unittest wall time 5366 s (about 89 min), `test-log.txt`. Nothing was committed or pushed, nothing touched Martinos, and the glacier and SLURM commands below were not run.

Memory is the main open issue: a default-width deform step at level 4 needs about 2 GB per image, so N = 32 does not fit a 48 GB card (estimate). Plan the deform group size around 12-16 (section "Memory").

## Files changed

| File | Change |
|---|---|
| `models/layers.py` | `sample_voxel_coords` (shared sampler, `boundary='zero'|'border'`); `transform_image` uses it (bit-identical); `upsample_trilinear2x`, `TrilinearUpsample2x` |
| `models/diffeo.py` (new) | `warp`, `warp_field`, `compose`, `integrate_svf`, `exp_and_inverse`, `center_velocities`, `upsample_displacement`, `jacobian_determinant`, `folding_fraction`, `pairwise_displacement`, `compose_cascade`, `to_mm` |
| `models/config.py` | `deform_decoder=False`, `decoder_widths=None`, `integration_steps=7`, `resolved_decoder_widths()` |
| `models/group_net.py` | `DeformDecoder` (D5..D1, SVF head), `GroupRegLevelModel(..., gradient_checkpointing=)`, task `deform`, `ALL_TASKS`, `DEFORM_TASK`, param counts by decoder stage |
| `data_loading/params.py` | `DEFORM_TASK`, `ALL_TASKS`, `N_TRAIN_LABELS = 19`, `DEFAULT_DEFORM_RESIDUAL_RANGES`, `deform_residual_ranges` |
| `data_loading/dali_loader.py` | `DeformPipeline` (paired image/label readers, optional residual affine with NN label warp), `deform_output_signature`, `output_count`, `paired_reader_kwargs`, `paired_permutation`, `read_labels`, `check_deform_inputs` |
| `data_loading/data_module.py` | `deform_batch_size`, `deform_ranges`, lazy `label_files`, `pipeline('deform')`, `DataModule` alias |
| `data_loading/postprocess.py` | `DeformPostprocess` (flips, one-hot, pairs, optional `T_render`) |
| `runtime/fixed_data.py` | `synthetic_label_phantom`, `random_svf`, `synthetic_deform_group`, `to_deform_batch`, `load_level_images_and_labels`, `DeformValidationSet` |
| `runtime/losses.py` | `soft_dice` (masked over present labels), `template_dice_loss`, `pair_dice_loss`, `smoothness_loss` (order 1/2), `deform_loss` |
| `runtime/metrics.py` | `hard_dice_per_label`, `mean_fg_dice`, `deform_group_errors`, `deform_pair_dice_loss` |
| `runtime/run.py` | Deform train step, `DEFORM_LOSS_KEYS`, `DEFORM_DIAG_KEYS`, `deform_diagnostics`, deform validation (eval + train mode, identity baseline), `DEFORM_BEST_METRIC`, `extra_best`, `overfit_status`, `window_deform_by_task` |
| `runtime/checkpoint.py` | `RunCheckpoint(extra_best=)`, `maybe_save_best(value, name)`, `latest_best_named` |
| `runtime/val_png.py` | `build_deform_panels`, `write_deform_png`, `deform_figure_title` |
| `runtime/args.py` | Deform flags (below), `LINEAR_TASKS`, `linear_tasks()`, `default_best_metric()`, `deform_decoder_from_args()`; controls and curriculum restricted to linear tasks |
| `main.py` | Deform data / overfit / validation paths, `ckpt/best_deform`, `gradient_checkpointing`, overfit summary deform keys |
| `tests/gpu/verify_dali_pipeline.py` | Check l (l1 alignment, l2 residual re-warp); `--checks` default now a-l |
| `tests/gpu/run_verify_glacier.sh` | Accepts check letter l |
| `tests/gpu/run_overfit_glacier.sh` | Mode `dryrun_deform` (two runs, writes `scratch/m6`) |
| `tests/gpu/check_mixed_run.py` | Deform criteria; `val/mean_pair_rms_mm` over linear tasks only; `ckpt/best_deform` with deform + linear |
| `tests/gpu/measure_deform_memory.py` (new) | Peak memory / step time of a deform step per N, widths, checkpointing |
| `tests/test_group_net.py` (`TestGradients._check`) | Two forward/backward passes, the second checked (local GTX 1080 first-call cuDNN NaN) |
| Tests | New `test_upsample`, `test_diffeo`, `test_deform_data`, `test_deform_overfit`, `test_check_mixed_run`; extended `test_group_net`, `test_data_loading`, `test_postprocess`, `test_runtime_losses_metrics`, `test_runtime_infra`, `test_trainer`, `test_val_png`, `test_main`; `runtime_helpers.write_fake_export(labels=)` |
| `.agents/tasks/network-design/design.md` | v3.7: 3.3, 4.5, 4.6, 5.2, 6.2, 9, 11 |
| `.agents/tasks/milestone-6/` | This report, `plan.md`, `memory.json` (+ `memory_none.log`, `memory_extra.log`), overfit summaries and logs, `run_some.sh`, `test-log.txt` |

## API

- **Model.** `ModelConfig(deform_decoder=True)` adds `DeformDecoder` with widths (128, 96, 64, 32, 16) for D5..D1. `model(images, task="deform", training=..., return_warped=False)` returns `v_raw`, `v` (centered), `u`, `u_inv` (float32 [N, X, Y, Z, 3], level voxels), `dec_feat`, `bottleneck`, `skips`, plus `warped` if requested. `task="deform"` without the decoder raises ValueError. `GroupRegLevelModel(level, config, gradient_checkpointing="none"|"integrator"|"decoder"|"both")`.
- **Data.** `DeformPipeline` outputs `(images float32, labels uint8[, T_dali])`. Shuffle mode is 'epoch' only ('buffer' raises, because two random shuffle buffers are not guaranteed to stay aligned). `LinearDataModule.pipeline("deform")`. `DeformPostprocess` returns `images`, `labels` (int32), `onehot` (19 ch float32), `pair_i/pair_j`, `flip_axes` (+ `T_render`).
- **CLI** (all saved in config.json): `--lambda-pdice 1.0`, `--p-pdice 0.1`, `--lambda-smooth 0.5`, `--smooth-order {1,2}`, `--integration-steps 7`, `--decoder-widths`, `--deform-decoder {auto,true,false}`, `--deform-residual-affine`, `--deform-group-size`, `--gradient-checkpointing`, `--overfit-dice-threshold 0.9`, `--overfit-fold-max 0.001`, `--val-deform-max-pairs 64`. `--task-ratio` accepts `deform`. `--curriculum` and `--global-transform` without a linear task are parser errors.
- **Logged keys.** `train_deform/{loss,tmpl,pdice,smooth,soft_dice_tmpl,soft_dice_pair}` plus `train_deform/diag_{feat_spread,feat_rel_spread,dead_frac,v_rms_vox,v_raw_mean_rms_vox,u_max_vox,fold_frac}`. Validation: `val_deform/{template_dice,pair_dice,neg_jac_pct,inv_consistency_mm,inv_consistency_max_mm,centrality_mm,raw_centrality_mm,pair_dice_loss,pair_dice_identity,template_dice_identity,groups,time_s}` and `val_deform_trainmode/{template_dice,pair_dice}`. PNG `val_png/deform_step<step>.png` has 4 rows: input, warped, labels, det J.
- **Best files.** Deform only: `ckpt/best` on `val_deform/pair_dice_loss`. Deform + linear: `ckpt/best` keeps the linear rule and `ckpt/best_deform` tracks the deform metric.

## Conventions (plan.md section 0, design 4.6)

- Fields are in voxels of their level. Warps are pulls: warp(vol, u)(x) = vol(x + u(x)).
- compose(u_a, u_b) = φ_a ∘ φ_b = u_b + warp_field(u_a, u_b).
- Fields are resampled with border clamping. Images and one-hots use zero fill.
- Centering is v − mean_group v. The inverse is exp(−v). The pairwise map j → i is compose(u_i, u_inv_j).
- Trilinear ×2 uses half-pixel semantics (coarse q at fine 2q + 0.5).
- The Jacobian is computed on interior voxels only.
- float32 everywhere for grids, fields and losses.

## Conflicts with linear defaults (plan.md section 1): all kept

`P.TASKS` stays ("rigid", "affine"). `deform_decoder` defaults to False, so linear-only models, parameter counts and checkpoints are unchanged, and old config.json files load. `transform_image` is bit-identical, and `test_spatial_transformer` is unchanged and passes. Linear best metrics, `DIAG_KEYS`, the linear overfit layout and the control files are unchanged; control files and the curriculum iterate only the linear tasks. The only deliberate test updates are the two former deform `NotImplementedError` assertions in `test_runtime_infra` and `test_trainer`.

## Test results (local GTX 1080, WSL)

- **Diffeo** (`test_diffeo`, 40×56×48, smooth v with max 2 voxels, T = 7, 3-voxel margin):
  - |exp(v) ∘ exp(−v)| max 0.0103 / mean 0.00186 voxels; the reverse order gives max 0.0108 / mean 0.00186.
  - The two-level cascade gives max 0.0046-0.0105 voxels.
  - A constant v gives the exact translation (1e-6).
  - Upsampled-field warp vs upsampled coarse warp: max 0.033, mean 0.010 (image range 3).
- **Synthetic deform data:** identity pairwise hard Dice 0.75-0.77 and template 0.85-0.86 (seeds 0-2), with max |u| 3.8 voxels (`synthetic_deform_group` default max_vox 4.0).
- **Module runs:** FEAT-001 60 tests OK. FEAT-002 86 tests OK (2 DALI skips). FEAT-003 68 + 1 tests OK. FEAT-004 `test_check_mixed_run` 10 tests OK (stdlib only, bgr_preproc38). Python 3.8 ast and py_compile of `tests/gpu/*.py` pass, and `bash -n` passes on the three scripts.
- **Full suite** (`tests/run_tests_wsl.sh`, after the last edit, `test-log.txt`): stage 1 (py3.8 ast/py_compile of 51 files, numpy references, TF-free imports, `main.py --help`, `bash -n` on 3 scripts) OK; stage 2 ran **284 tests, OK (skipped=4)**: three need nvidia.dali + GPU (`test_pipeline_with_control_file`, `test_deform_pipeline_aligned_on_synthetic_files`, `test_pipeline_runs_on_synthetic_files`) and one needs dllogger. Unittest wall time 5366 s (about 89 min); runner exit 0.
- **Fix during the final run:** the first full-suite attempt failed 1 of 284: `TestGradients.test_every_variable_gets_gradient` (cfg B, affine), with 512 NaN entries in the encoder l3 conv-kernel gradient. No model code had changed since it last passed. Reproduced in isolation on both local GTX 1080s, while the CPU stayed finite. The NaN appears only on the first backward pass of a new conv shape, so it is the same local cuDNN issue as milestone-4 report item 5, here with autotune off. The fix is test-only: `_check` now runs the forward/backward twice and checks the second gradient. Model and training code are unchanged; in training, the NaN guard skips such a step. The previously flaky `test_permutation_equivariance` passed in this run.

### Synthetic deform overfit (narrow, level 4, N = 4, Adam 3e-4 constant, no flips/dropout, BN momentum 0.9, stop at 10-step soft template Dice ≥ 0.91 and folding < 0.1 %)

| Quantity | Identity (step 0) | Final (step 306), eval mode | Train mode |
|---|---|---|---|
| Template hard Dice | 0.8635 | **0.9712** | 0.9709 |
| Pair hard Dice | 0.7742 | **0.9503** | 0.9505 |
| neg. Jacobian | 0 % | 0.0 % | |
| Inverse consistency | 0 | 0.050 mm (max 0.95) | |
| Centrality (raw) | | 9e-8 mm (2.66 mm) | |

The training soft Dice curve (template / pair) runs 0.831 / 0.775 at step 1, 0.871 / 0.882 at 50, 0.887 / 0.902 at 100, 0.902 / 0.912 at 200 and 0.910 / 0.916 at 306. The median step is 1.24 s, the GPU peak 4.2 GB and the wall time 456 s (`overfit_summary_deform_narrow.json`).

With the default BN momentum 0.99, the run stops at step 189 with eval 0.873 / 0.799 against train mode 0.966 / 0.941: the BN moving statistics lag behind the fixed group (`overfit_summary_deform_narrow_bn099.json`). Watch `val_deform` vs `val_deform_trainmode` early in real runs.

**Three tasks** (`TestThreeTasks`, rigid:1,affine:1,deform:2, flips on, 16 steps): trace counts {rigid 1, affine 1, deform 1}. The optimizer keeps 2 × trainable variables (no late slots). Deform steps leave the linear decoder unchanged, and rigid steps leave the deform decoder (including BN stats) unchanged.

## Memory (level 4 deform step; `memory.json`, `tests/gpu/measure_deform_memory.py`)

Measured on the GTX 1080 with a 6000 MB TF limit, `TF_CUDNN_USE_AUTOTUNE=0`, a 512 MB cuDNN workspace limit, fp32, Adam and default losses on the synthetic group. Values are the TF allocator peak over 3 measured steps after one warm-up step.

| Widths | Checkpointing | N = 2 | N = 3 | N = 4 | N = 6 | N = 8 | N = 16 | Median step |
|---|---|---|---|---|---|---|---|---|
| narrow (4.03 M) | none | 2090 MB | 3008 MB | 4010 MB | OOM | OOM | skipped | 0.71 / 0.95 / 1.08 s |
| narrow | both | | | 2510 MB | | 4781 MB | OOM | 1.38 / 2.32 s |
| default (29.63 M) | none | 4437 MB | OOM | OOM | | skipped | skipped | 1.92 s |
| default | both | 3909 MB | | OOM | | skipped | | 2.29 s |

- Weights, Adam slots and the batch take 0.07-0.15 GB (narrow) and 0.41-0.42 GB (default).
- **Narrow:** about 0.96 GB per image without checkpointing and about 0.57 GB with `both`, which saves 40 % at 1.3-2.2× the step time.
- **Default:** about 2.0 GB per image, from one point after subtracting the resident 0.41 GB. `both` saves only about 12 %, since the encoder is not recomputed.
- A larger local limit (7000 MB) crashed the WSL VM, so no second default-width point exists.
- **N = 32 extrapolation (ESTIMATE, not measured):**
  - Default widths: 32 × 2.0 GB + pair warps (p_pdice 0.1 → 99 pairs, roughly 0.1 GB each at level 4) ≈ **70+ GB**. This does not fit an RTX 8000 (48 GB) or an A100 (40 GB).
  - Default widths at N = 16 (24 pairs): ≈ 35 GB, expected to fit 48 GB.
  - Narrow at N = 32: ≈ 31 GB + pairs ≈ 40 GB.
  - The level-4 linear step at N = 32 measured 26-34 GB, so the deform step is about 2× that per image.
- **Recommendation:**
  - Use `--deform-group-size 12` to `16` for deform at level 4; the linear tasks can keep N = 32.
  - Or measure `--amp true` and `--gradient-checkpointing both` on the cluster first. The script runs there unchanged with memory growth (omit `--gpu-memory-limit-mb`).
  - cuDNN autotune on the cluster may add workspace.

## Not verified

- The DALI `DeformPipeline` on a GPU: reader alignment across epochs, the NN label warp with the residual, and labels as uint8 through `DALIDataset`. Check l covers these on glacier.
- Real-data deform training (`dryrun_deform`), the real-data deform val set (39 T1 val cases with train18 labels) and the PNGs on real labels.
- Memory and step time on the RTX 8000 or A100, and every value at N > 8. The N = 32 numbers above are extrapolations.
- Python 3.8 / numpy 1.21 / scipy 1.4.1 at runtime: only the syntax checks ran. Library code avoids newer APIs, and scipy appears only in the tests.
- Warm-starting a deform model from a linear-only checkpoint (milestone 7): `load_model_weights` refuses because `deform_decoder` differs. The follow-up is to allow that field to differ and assert only the encoder and linear variables.
- Residual-affine range flags are not exposed (design defaults; `LinearDataModule(deform_ranges=)` in code).
- Open minor review findings (`review.md`, verdict APPROVED), not changed here:
  - There is no oracle test of the pairwise composition order.
  - `--deform-group-size` defaults to `--group-size`, so it is 32 at level 4. Pass `--deform-group-size 12-16` for three-task runs.
  - `diag_dead_frac` runs on the pre-activation `dec_feat`.
  - `test_diffeo` uses scipy ≥ 1.6 `grid-constant` and does not skip on older scipy.

## Glacier and SLURM commands for the orchestrator (NOT run)

Prerequisite: the milestone-6 changes are still uncommitted locally (base `cb161f1`), so commit and push them first, then `git pull` in the clone. On glacier, pick a free GPU with `nvidia-smi` and pass its index as `<gpu>`.

```bash
# check l: DeformPipeline image/label alignment (l1, residual off, 4 shuffled batches over 6 val cases)
# and residual re-warp (l2); writes scratch/m2 (summary.json, verify.log)
bash /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad/tests/gpu/run_verify_glacier.sh <gpu> 1800 verify l
# two 20-step real-data runs (deform:1, then rigid:1,affine:1,deform:2), N = 8, level 4, val + PNGs at
# 10 and 20, each checked by tests/gpu/check_mixed_run.py; writes scratch/m6; exit = first failure
bash /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad/tests/gpu/run_overfit_glacier.sh <gpu> 1800 dryrun_deform
# optional: memory on an A100 (inside the container, one GPU)
cd /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad && \
CUDA_VISIBLE_DEVICES=<gpu> SINGULARITYENV_CUDA_VISIBLE_DEVICES=<gpu> SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1 singularity exec --nv -B /scratch -B /autofs -B /space \
  /autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif \
  python tests/gpu/measure_deform_memory.py --n 4,8,16,24,32 --widths default --gradient-checkpointing none,both \
  --out /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/scratch/m6/memory_a100.json </dev/null
```

Expected results:
- **l1:** PASS with every sample's image max |diff| ≤ 1e-6 and its labels equal. The case order changes between epochs. INFO mask Dice is DALI = numpy > mismatched.
- **l2:** PASS with direct max |diff| < 2e-3, label agreement ≥ 0.99, rotation ≤ about 1.25°, |t| ≤ 1 mm and max |L − I| (linear part) of order 0.02.
- **`dryrun_deform`:** `check_mixed_run: PASS` twice. Expect train_deform/loss finite, `val_deform/*` finite, `deform_step0000020.png`, trace counts 1 per task, `ckpt/best` in both runs and `ckpt/best_deform` in the mixed3 run.

SLURM debug run (`slurm/train_debug.sh` reads `RUN_NAME`, `STEPS`, `N`, `VAL_EVERY`, `WARMUP`, `TASK_RATIO`, `LEVEL`, `GROUP_CONFIG`, `AMP`, `EXTRA_ARGS`; it uses `-A qtim -p rtx8000`, 1 GPU and 3 CPUs). Needs Jay's approval and fewer than two active jobs (`squeue -u jn85`):

```bash
cd /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad && git pull
mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
RUN_NAME=m6_deform_debug STEPS=500 N=16 VAL_EVERY=100 WARMUP=50 TASK_RATIO=deform:1 \
  sbatch --job-name=m6dbg --time=01:00:00 --mem=64G slurm/train_debug.sh
```

N = 16 is about 35 GB by the estimate above (48 GB RTX 8000). If it OOMs, rerun with `N=12` or `EXTRA_ARGS="--gradient-checkpointing both"`. The validation group size defaults to min(N, 39) deform val cases.
