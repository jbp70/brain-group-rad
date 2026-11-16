# ablation-oom-fixes: report

Branch `ablation-oom-fixes` off main 4fe21f7 (worktree `.worktrees/ablation-oom-fixes`), workflow
wf_063ae32b8181bed4. Brief: `brief.md` (same folder). No cluster job was submitted and nothing was
pulled on the cluster (cluster logs and results were only read). Every command in section 6 needs
Jay's approval, and the branch must be merged, pushed and pulled in the clone first (r_cascade.sh
freezes the clone's HEAD; r_level1_scratch.sh runs the clone directly).

## 1. Summary

| Item | Change | Default behaviour |
|---|---|---|
| Failure 1: IN OOM at L4 N 128 | `InstanceNormWithAffine` under float16 uses the fused BatchNorm kernel per (image, channel) (`models.layers.instance_norm_fused`); float32 statistics inside the kernel, every full-size tensor float16 | float32 models and frozen cascade stages: unchanged ops (bit-identical) |
| Failure 2: scratch L1 OOM at first deform step | `--deform-warmup-step true\|false` (default false): one deform train step at startup, everything it changed restored, its batch replayed; on by default in `slurm/r_level1_scratch.sh` | flag off: no extra op, no extra call |
| Checker short window | `nonfinite_window_<task>` judges a bucket only with >= max(20, ceil(2 / F)) task steps (40 at F 0.05) | full 500-step windows of long runs judged as before |
| Step rule | r_level1_scratch.sh header: STEPS = (229860 - val_time) / (3/16 t_lin + 13/16 t_mix) | header comment only |

## 2. Failure 1: instance norm

### Cause

At level 4 the encoder's `l4` block normalises the `l3` output, [128, 40, 56, 48, 128] = 1.76e9
elements (3.5 GB in float16). The old layer cast it to float32 (7 GB), computed `tf.nn.moments`
and `(xf - mean) * rsqrt(var + eps)` in float32 and cast back. The autodiff graph keeps `xf` and
`xf - mean` (two full float32 copies, 4x the float16 activation) for the backward pass, and the
forward holds a third one transiently. The linear tasks are not checkpointed, so these copies of
every IN layer stay alive until the backward pass: at N 128 that does not fit 48 GB (the OOM node
`encoder/l4/norm/sub` is the `xf - mean` of that layer).

### Fix

`instance_norm_fused(x, eps, data_format=None)` (models/layers.py):

* x [N, s.., C] -> transpose [N, C, s..] -> reshape [1, N*C, S, 1] (data format NCHW, chosen when a
  GPU is visible; on CPU, whose kernel has no NCHW: transpose [s.., N, C] -> [1, S, 1, N*C], NHWC).
* `tf.compat.v1.nn.fused_batch_norm(..., is_training=True)` with scale 1, offset 0: the per-channel
  batch statistics of a one-sample batch whose channels are the (image, channel) pairs are exactly
  the instance-norm statistics. The kernel (cuDNN on the GPU) reads float16, accumulates mean and
  biased variance in float32 (no float16 x^2, variance >= 0), and writes y in float16. Its gradient
  is the full batch-norm gradient (= the instance-norm gradient) and needs only the saved input
  plus float32 [N*C] statistics.
* reshape / transpose back; then the layer's `AffineParams` in float16 as before.
* Epsilon below 1.001e-5 is raised to 1.001e-5, as `fused_batch_norm` and the fused Keras BN do
  (config 1e-5 -> 1.001e-5).

Why this and not a hand-written float16 formula: the float16 alternatives either need a full
float32 cast for the moments (the materialised copy we want to avoid; TF without XLA materialises
`tf.cast` outputs) or compute x^2 / (x - mean)^2 in float16, which overflows above |x| ~ 256 unless
rescaled per channel. The fused kernel is the same one the batch-norm path uses (Keras BN runs
FusedBatchNormV3 on 5-D NDHWC float16, which on the GPU transposes to NCDHW internally unless
TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT is set), so the full-size tensors per IN layer are the
same as per BN layer: the saved input (here the transposed copy instead of the original) and the
layout transposes. No custom gradient was needed.

When it is used (`InstanceNormWithAffine.call`): float16 input, or float32 input with the new
Python attribute `fused_float32` True. `GroupRegLevelModel` sets `fused_float32` on every IN layer
(`layers.set_instance_norm_fused`) when the global policy is not float32 and `norm == 'instance'`.
That covers the float32 decoder stages of `--deform-decoder-dtype float32_fine` (d2, d1 and
svf_norm_act at level 1 are float32): with the old path their IN layers would have kept one extra
float32 copy each, ~5 GB at level 1 N 4 (d1 input [4, 160, 224, 192, 32] f32 = 3.5 GB, d2 input
1.76 GB), which would likely not fit the 42500 MB limit. Float32 models (float32 global policy) and
the frozen cascade stages (built under a float32 policy in models/cascade.py) keep the old
float32-moment ops bit for bit.

Gradient checkpointing `full`: IN has no state, the recompute runs the same kernel; gradients with
`full` equal `none` (rel 0.00e+00, under op determinism with Grappler off, mixed_float16, TINY
level 4, test `TestInstanceNormFused.test_checkpointing_full_gradients`).

Numerics: vs a float64 reference (|x| up to 1.5e4, so a float16 x^2 would overflow): float16 path
max |y error| 9.6e-4 (float16 rounding of y ~ 3), max |dx error| 4.3e-5 (|dx| max 0.082); float32
fused path 2.4e-7 / 7e-9. Under AMP the IN results change slightly vs main (kernel rounding,
epsilon 1.001e-5). No IN model has been trained yet, so nothing to keep compatible; IN checkpoints
load unchanged (same variables).

### Memory (local GTX 1080, 6000 MB cap, one traced forward + backward, d12hs config B, AMP)

Script `measure_in_memory.py` (this folder; one child process per configuration; no optimizer,
data pipeline or losses, so absolute numbers are below the trainer's, the ratios are what counts).
Variants: `batch` = production BN, `instance` = this branch, `in_old` = main's IN (monkeypatched).
JSON: `in_memory_l4.json`, `in_memory_l2.json`, `in_memory_l4_f32fine.json`, `in_time_l4.json`.

| Level / task | N | batch MiB | instance MiB | in_old MiB |
|---|---|---|---|---|
| L4 rigid (no ckpt) | 4 / 8 / 16 | 1266 / 2180 / 4255 | 1176 / 2208 / 4300 | 1708 / 3371 / OOM |
| L4 rigid, slope per image | 8 -> 16 | 259 | 261 | 416 (4 -> 8) |
| L4 deform (ckpt full, lean, split) | 4 / 8 | 2048 / 3585 | 1904 / 3575 | 3811 / OOM |
| L4 rigid / deform, rerun with timing | 8 | 2159 / 3595 | 2217 / 3655 | - |
| L2 rigid | 2 / 3 | 1160 / 1605 | 1035 / 1557 | 1286 / 1900 |
| L2 deform | 2 / 3 | 3700 / 5296-5361 | 3651 / 5485 (one OOM at 6000) | OOM at 2 |
| L4 deform, float32_fine decoder | 4 / 8 | 2504 / 4785 | 2529 / 4802 | 3695 / OOM |
| L2 deform, float32_fine decoder | 2 | error (no memory for cuDNN scratch at 6000 MB) | same | same |

Step time (median of 3, L4 N 8): rigid 1.69 s BN vs 1.67 s IN, deform 4.66 s vs 4.78 s (+3 %).
The GTX 1080 (Pascal) has no tensor cores, so Grappler's layout choices differ from the RTX 8000;
the smoke gives the real step times.

Estimate on the RTX 8000 (IN = BN + 0-3 %, the spread of the local reruns; BN reference peaks are
the trainer's `gpu_peak_mb` of the config-B chains):

| Level | Groups (linear / deform) | BN peak | IN estimate | Old IN |
|---|---|---|---|---|
| 4 | 128 / 64 | 33.2 GiB (34000 MiB) | 33.2-34.2 GiB | OOM at step 0 (~53 GB from the 416 MiB/image slope) |
| 2 | 32 / 16 | 27.6-27.9 GiB | 27.6-28.8 GiB | not run |
| 1 (cascade, TF limit 42500 MB) | 8 / 4 | 38.3-39.2k MiB (37.4-38.3 GiB) | 38.3-40.4k MiB | not run (float32_fine stages +~5 GB) |

So the fused path should fit at every level. Level 1 has the least headroom (~1-4 GB under the
42500 MB limit, which is ~41500 MiB of usable pool). What the IN smoke must confirm: L4 `gpu_peak`
<= ~35 GiB, L2 <= ~30 GiB, L1 <= ~41500 MiB with no OOM; finite IN steps after the AMP warm-up.
Fallbacks if L1 does not fit (Jay decides; none changes N): `MIN_PAIRS=1` at level 1 (r_cascade.sh
has no knob for it, open decision 3 of ablation-prereqs), or `--deform-decoder-dtype float16` at
level 1 for the IN chain only (a recipe difference to report).

## 3. Failure 2: deform warm-up step

`--deform-warmup-step true` (runtime/args.py; parser error without 'deform' in --task-ratio).
main.py calls `Trainer.deform_warmup_step()` once per process, after the checkpoint restore, the
resume BN-freeze rebuild and the BN recalibration (both call `rebuild_steps()`, which would
re-trace), and before the dry-run / `fit()`. A resumed run starts a fresh process with a fresh
allocator, so it runs the warm-up again.

`Trainer.deform_warmup_step()` (runtime/run.py):

1. draws the next deform batch and keeps a host (numpy) copy;
2. snapshots to host every variable of `_warmup_state_variables()`: all model variables (weights,
   BN moving statistics), the optimizer's slots, `iterations` and learning-rate variable, the
   LossScaleOptimizer's `current_loss_scale` and growth counter, `trainer.step`, the
   post-processing RNG states (flip / pair, TF intensity augmentation, synthesis generators) and
   the LSO skip flags;
3. runs `train_step('deform', batch)` = the same traced tf.function the real deform steps use (so
   this is the deform trace; trace once per task holds: `trace_counts` deform 1 after the warm-up
   and after fit);
4. restores every snapshotted variable (also if the step raised), and puts the batch back in front
   of the deform source (`_ReplayFirst`), so the first real deform step trains on the batch it
   would have had without the warm-up.

Nothing is appended to `history`, nothing is logged to metrics.jsonl or events, no non-finite step
is counted (the warm-up step's own result is printed in the .out line `deform warm-up step
(discarded, state restored): finite .., loss .., .. s, gpu peak .. MB`). What stays changed: the GPU
allocator (intended) and the process's peak statistic, so `gpu_peak_mb` includes the warm-up step
from the first log window on (it is a process maximum anyway; for the scratch smoke it is the deform
peak the run needs). The DALI deform pipeline produces one batch earlier than without the warm-up;
its sequence is unchanged and the consumer sees the same order (replay). The linear sources are not
touched.

Test `tests.test_trainer.TestDeformWarmupStep` (narrow level 4, rigid:affine:deform 1:1:2, deform
delayed to step 4, flips on, three distinct deform batches cycled, non-zero SVF head, op
determinism): right after the warm-up every snapshotted variable equals its value before it, the
source was drawn once, history empty; after `fit(10)` the run with the warm-up equals the run
without it bit for bit (all model / optimizer / LSO / step / RNG variables, task and loss
sequence, non-finite count, iterations, trace counts {rigid 1, affine 1, deform 1}), fp32 and AMP.

Validation is not warmed up: diagnostic (a) 10639614 (deform from 0, validation at 100) ran deform
steps after the validation pass at 38119 MiB, so a deform-first layout survives validation.
Applicability: the flag is generic (any level, cascade or not); it is only enabled in
r_level1_scratch.sh. The L4 from-scratch recipe (DEFORM_START 3000) has worked without it, so the
level-4 / level-2 scripts are unchanged; if a delayed-deform run OOMs at its first deform step,
`EXTRA_ARGS='--deform-warmup-step true'` is the knob.

## 4. Checker window rule

`tests/gpu/check_mixed_run.py`: `min_window_task_steps(F) = max(20, ceil(2 / F))`; a bucket is
judged only with at least that many steps of the task (40 at the scripts' F 0.05; 20 at F >= 0.1).
Two isolated AMP loss-scale skips in a short smoke bucket can then never fail it (run (a): 2 of 21
affine steps, 9.5 %), while every full 500-step window of a long run (>= 125 steps per task at
1:1:2) is judged as before. A 200-step smoke's deform bucket (48 steps) is still judged (2 of 48
pass, 3 of 48 fail; tests in `TestNonfiniteWindows`).

## 5. Step rule for the from-scratch run (A1)

The run is linear-only for the first 3/16 of STEPS (deform starts at DEFORM_START = 3 STEPS / 16),
then mixed at 1:1:2. With t_lin and t_mix the mean step times of the two phases and v one
validation pass (the run validates STEPS / VAL_EVERY times):

    STEPS = (229860 - val_time) / (3/16 t_lin + 13/16 t_mix),  val_time = STEPS / VAL_EVERY x v
          = floor(229860 / (3/16 t_lin + 13/16 t_mix + v / VAL_EVERY))   (minus ~1 % for startup)

From the diagnostic smokes on rtx-04 (metrics.jsonl `time/compute_ms + time/data_ms`, log windows
of 25 steps): t_lin 0.34 s (10639615, steps 26-100), t_mix ~9.0 s (10639614 windows 150-200: 9.2,
9.3, 8.6 s), v 285-414 s (val/wall_s). With VAL_EVERY 500: 229860 / (0.064 + 7.31 + 0.8) ~ 28 000
steps (the old one-phase rule with a 9 s mean step gave ~23 500; the ablation-prereqs example of
19 300 assumed 11 s). Re-measure on the smoke (the warm-up moves the deform trace out of the first
deform window), with the stdlib-only helper in this folder (login node is fine, it only reads
metrics.jsonl):

    python3 .agents/tasks/ablation-oom-fixes/step_rule.py /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results/r1_scratch_smoke2 100 500

It averages `time/compute_ms + time/data_ms` over the log windows ending in (25, DEFORM_START]
(t_lin; the first window holds the linear traces) and after DEFORM_START + 25 (t_mix; 25-step
windows hold a random task mix, so all mixed windows are averaged), takes v as the mean
`val/wall_s` without the first pass (tracing), and prints STEPS = 0.99 x 229860 / (3/16 t_lin +
13/16 t_mix + v / VAL_EVERY). On the diagnostic smokes: 10639615 gives t_lin, 10639614 t_mix.

## 6. Smoke commands (Jay's approval needed; after merge + `git pull` in the clone)

From the clone `/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad`.
Before any sbatch: `squeue -u jn85` (two-job limit; the four v3 chains count), and the log folder
exists. Both use the new MNI-affine data and NEW run names: the old `r4_abl_in_smoke` chain state
file pins commit 94eb7ed (a resubmit with the old names would rerun the old code), and the old
`r1_scratch_smoke` run dir would be resumed.

    D=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/data/npy_v3

### 6.1 Instance-norm chain smoke (r_cascade.sh NORM=instance; same recipe as 10639200)

    env DATA=$D S4=2.5 S2=10 S1=15 SYNTH_PROB=0 STEPS_L4=300 STEPS_L2=200 STEPS_L1=200 VAL_EVERY=100 WARMUP_L4=50 DEFORM_START_L4=100 DEFORM_RAMP_L4=100 CURRICULUM_RAMP_L4=100 G_START_L4=100 G_RAMP_L4=100 EXTRA_ARGS_L2='--frozen-integration-grid current --warmup-steps 50' EXTRA_ARGS_L1='--warmup-steps 50' GROUP_ARGS='--group-config B' NORM=instance RUN_NAME_L4=r4_abl_in_smoke2 RUN_NAME_L2=r2_abl_in_smoke2 RUN_NAME_L1=r1_abl_in_smoke2 sbatch --job-name=abl_in_smk2 --time=06:00:00 slurm/r_cascade.sh

(rtx8000, 1 GPU, 3 CPUs, 128G from the script header; ~2.5 h expected like the gnone / gE smokes.
Dry run first: prefix `BGR_DRY_RUN=1` and replace `sbatch ...` by `bash slurm/r_cascade.sh`; check
`norm instance` in the header line and `BN_RECAL_BATCHES=0` at L2 / L1.)

Read (logs/abl_in_smk2-<job>.out, results/r{4,2,1}_abl_in_smoke2):
* each level ends with checker PASS (`=== level k end ... rc 0`);
* `INFO gpu_peak` per level: L4 <= ~35 GiB (BN 33.2), L2 <= ~30 GiB (BN 27.6-27.9), L1 <= ~41500
  MiB (BN 38.3-39.2k MiB) under the 42500 MB limit; no RESOURCE_EXHAUSTED in the .err;
* non-finite steps only early (AMP warm-up; loss scale not at the floor), `nonfinite_window_*` PASS;
* `train_<task>/diag_feat_rel_spread` > 1e-3 (no collapse);
* no `bn_recalibration` event at L2 / L1; config.json `model_config.norm` = instance,
  `group_config` = B;
* step times vs the batch-norm smokes (gnone / gE: L4 33 GiB, L1 deform 24-26 s): the fused IN adds
  transposes; a large slowdown (> ~10 %) would matter for the 64 h chain.

### 6.2 From-scratch level-1 smoke (reproduces 10639201: deform from 100, validation at 100)

    env DATA=$D RUN_NAME=r1_scratch_smoke2 STEPS=300 VAL_EVERY=100 WARMUP=50 DEFORM_START=100 DEFORM_RAMP=50 CURRICULUM_RAMP=100 G_START=100 G_RAMP=100 PROJECT_STEPS=28000 PROJECT_VAL_EVERY=500 sbatch --job-name=r1scr_smk2 --time=05:00:00 --mem=128G slurm/r_level1_scratch.sh

(rtx8000, 1 GPU, 3 CPUs, 128G; DEFORM_WARMUP_STEP defaults to true; ~1 h expected.)

Read (logs/r1scr_smk2-<job>.out, results/r1_scratch_smoke2):
* header line `... | deform warm-up step true` and, before the first train step, the line
  `deform warm-up step (discarded, state restored): finite .., loss .., .. s, gpu peak .. MB`
  (peak ~38-40k MiB);
* training passes step 101 (the first deform step after the step-100 validation, where 10639201
  died) and reaches 300 with the validations at 100 / 200 / 300; no RESOURCE_EXHAUSTED;
* checker PASS, incl. `trace_once` (deform 1: the warm-up was the trace) and `nonfinite_window_*`;
* `INFO gpu_peak` <= ~41500 MiB (the first windows already show the warm-up's deform peak);
* t_lin, t_mix, v with the snippet of section 5 -> STEPS for the long run
  (`env DATA=$D RUN_NAME=r1_scratch STEPS=<..> sbatch --job-name=r1scr --time=48:00:00 --mem=128G
  slurm/r_level1_scratch.sh`, resubmitted identically after each 48 h limit).

## 7. Verification

* Targeted (WSL tf_gpu, GTX 1080, BGR_TF_GPU_LIMIT_MB 6000): `tests.test_ablation_configs.TestInstanceNormFused`,
  `tests.test_trainer.TestDeformWarmupStep`, `tests.test_check_mixed_run`, `tests.test_group_layers`:
  67 tests OK. `tests.test_ablation_configs` (incl. the existing IN training at L4 / L2 / L1 chain,
  streaming and predict tests, now on the fused path) + `tests.test_checkpointing.TestFullGradients`:
  28 tests OK (BN ckpt full vs none rel 0.00e+00 fp32 and AMP). `tests.test_slurm_scripts.TestLevel1Scratch`: 5 OK.
* Full suite (`tests/run_tests_wsl.sh` in the worktree): see section 8.

## 8. Full suite and goldens

`tests/run_tests_wsl.sh` of the worktree (resolves the repo from its own location; worktree `.git`
uses a relative gitdir) on commit fe3852b: stage 1 (py3.8 syntax, bash -n, TF-free tests) OK,
240 TF-free tests without TF; stage 2: suite status 0, 1107 tests (5 skipped), 0 failing module
invocations, unittest wall time 27 462 s (7.6 h). Goldens: `[golden] max |diff| 0 (tolerance 0)`,
`[golden deform] max |diff| 0 (tolerance 0)`, `[golden tta-off] 70 hashes compared, 0 differ`.
Modules touched: test_trainer 78, test_slurm_scripts 88, test_check_mixed_run 44,
test_ablation_configs 21, test_checkpointing 10, test_group_layers 15 (all exit 0).

`step_rule.py` was run read-only on the login node against the diagnostic smoke
r1_scratch_diag_noval (DEFORM_START 100): t_lin 0.335 s, t_mix 9.083 s, v 407 s -> STEPS ~ 27 564
(its mixed phase is only 100 steps; use the new smoke's numbers).
