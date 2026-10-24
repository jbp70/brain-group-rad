# Milestone 9, Phase 1: requirements (level-1 memory design and model sizing)

Status: draft for design (2026-10-22). Sources: `network-design/design.md` v3.12 (sections 4, 9, 11), `milestone-6/report.md`, `milestone-8/report.md`, the code listed in section 2, and Jay's messages (milestone 9 request, message 12).

## 1. Summary

Milestone 9 adds the full-resolution level 1 (160×224×192, 1 mm, 6.88 M voxels, 8× level 2) as the third cascade stage. At today's widths a level-1 deformable training step does not fit even with AMP and checkpointing `both` (level 2 costs ~3.4 GB per image, so level 1 would need ~27 GB per image, i.e. N = 1 at best). Phase 1 finds out where the memory goes, adds memory knobs that are off by default, builds a calibrated analytic memory model, and uses it to recommend one model configuration (widths, checkpointing, AMP, integration steps). That configuration must give a deformable group size of at least N = 4 at level 1 with AMP and full gradient checkpointing on a 48 GB card. The chosen configuration then applies to every level: levels 4 and 2 are retrained with it later. Phase 1 plans that retraining and writes the measurement job for level 1, but runs nothing on the cluster. Phase 1 also designs, without implementing, the rest of milestone 9: the 3-level cascade, the level-1 DALI pipeline, level-1 validation and the retraining sequence.

## 2. Context and constraints taken from the code and measurements

- Widths are already configurable: `--filters` (6 encoder widths, stem..L6), `--decoder-widths` (D5..D1, default half the matching encoder widths), `--reductions`, `--dense`. `--integration-steps` (default 7) exists. `--gradient-checkpointing none|integrator|decoder|both` wraps only the decoder and/or the integrator in `tf.recompute_grad`; the encoder is never recomputed. In training mode, recomputing the decoder updates its BatchNorm moving statistics twice per step (design 4.5).
- Full-resolution tensors at level 1 come from the stem (f0 = 32 channels), D1 (input = up(D2) + stem skip = 64 channels, doubled to 128 by the group-mean concat), the SVF fields and the integration (float32 even under AMP), the image and one-hot warps, and the losses. L2/D2 run at 80×112×96 (D2 input 128 channels, 256 after the group concat). L3 and coarser are the same size as at level 2.
- Per-tensor size limit (hypothesis to confirm, R1.3): the measured level-2 linear fp32 N = 12 failure in layer `d2` came from `tests/gpu/measure_level_memory.py` calling `build_model(n)`, which also runs a deform forward at the linear N. `main.py` already caps the build N at the deform group size. That forward's D2 group-concat tensor [12, 80, 112, 96, 256] has 2.64·10⁹ elements, above 2³¹ − 1 ≈ 2.15·10⁹, while N = 8 (1.76·10⁹) worked at 11 GB peak. At level 1 with default widths, the D1 group-concat input [4, 160, 224, 192, 128] has 3.5·10⁹ elements at N = 4. If the 2³¹ limit holds, that fails no matter how much memory is free. A 32-channel full-resolution tensor exceeds it from N = 10 (relevant for linear N up to 16).
- Measured memory (deform numbers include Dice + LNCC 1/1):
  - Level 4 (RTX A6000, default widths, fp32, no checkpointing): deform peak 16.6 / 24.8 / 33.8 GB at N 8 / 12 / 16 (~2.15 GB per image + 0.44 GB resident). Linear N = 32 ~26–34 GB. GTX 1080, narrow widths: ~0.96 GB per image with `none`, ~0.57 GB with `both`. Default widths: `both` saves only ~12 %.
  - Level 2 (RTX 8000, default widths, frozen level-4 cascade, `scratch/m8/memory_level2.json` on the cluster, not in the repo):
    - deform fp32: none N2 19.0 / N3 27.5 / N4 36.4 GB (N6 OOM); integrator N2 15.6 / N4 30.7 / N6 45.3; both N2 12.2 / N3 17.9 / N4 23.9 / N6 35.2.
    - deform AMP: none N2 13.9 / N4 27.0 / N6 39.9; integrator N2 10.9 / N4 20.8 / N6 30.8 / N8 40.9; both N2 7.6 / N4 14.4 / N6 21.2 / N8 27.8 (7.0 s per step on RTX 8000, ~4.9 s on A100).
    - linear fp32: N4 5.9 / N8 11.0 (N12 failed in `d2`). Linear AMP: N4 3.2 / N8 5.8 / N12 8.5 / N16 10.9.
    - OOM messages showed single activations [N, 256, 80, 112, 96], [N, 80, 112, 96, 19] and [N, 3, 80, 112, 96].
- Sizing arithmetic: with a 44 GB budget on a 48 GB card, N = 4 leaves ≤ ~10 GB per image after resident weights, Adam slots and the frozen stages. That is a ~2.7–3× per-image reduction against the naive 27–28 GB at level 1. N = 8 needs ~5 GB per image (~5.5×). (The brief's "8–10× reduction" estimate is taken as the margin for N = 8 plus the per-tensor limit, not as a requirement for N = 4.)
- Results to compare against after retraining:
  - Level 4: `m7_three_task_lncc` (rigid 3.82, affine 4.80 mm center; deform template / pair Dice 0.757 / 0.701).
  - Level 2: `m8_level2_gradnorm` (rigid 2.03 / 2.90, affine 2.53 / 3.67, affine_g 3.18 / 4.59 mm center / pair; deform 0.846 / 0.789; folding 0.012 %).

## 3. Assumptions (the design review may revise them)

- A1. Budgets: "fits a 48 GB card" means TF allocator peak ≤ 44 GB (RTX 8000 / RTX A6000). "Fits the A100 (40 GB)" means peak ≤ 36 GB, the rule used so far.
- A2. "Full gradient checkpointing" means the encoder, the deform decoder and the integrator are all recomputed in the backward pass. It is a new `--gradient-checkpointing` value. The existing values keep their meaning.
- A3. Reduced widths apply to all levels, because weights copy across levels and every level must have identical variables. The linear-decoder widths (`--reductions`, `--dense`) and the bottleneck shape 5×7×6×24 stay unchanged unless the root cause shows they matter at level 1 (they run at 5×7×6, so they should not).
- A4. Phase 1 cannot measure real-width level-1 steps locally (GTX 1080, 6000 MB cap). Level-1 numbers are predictions until the m9 memory job runs. That job is submitted later, after Jay approves.
- A5. Thresholds for T = 6 (R3.4) and for retraining parity (R6.4) are proposed values for Jay to confirm.
- A6. The level-2 cluster JSON is not in the repo. Calibration uses the numbers transcribed in section 2 plus the local `milestone-6/memory.json` and `lncc-loss/memory.json`.

## 4. Functional requirements

### R1. Root cause of per-image memory (deliverable 1)

- R1.1 Produce a per-component breakdown of the level-2 deformable per-image memory slope (GB per image) for fp32 and AMP × `none`, `integrator`, `both`. Components: encoder full- and half-resolution activations, the encoder at 40×56×48 and coarser, decoder D2/D1, the SVF head, centering, integration (forward and inverse, per squaring step), the image warp for LNCC, the one-hot creation and warp, template and pair Dice, LNCC, smoothness and Jacobian, cuDNN layout transposes and workspace, and the frozen-stage inputs (U, U⁻).
- R1.2 The breakdown must say for each component whether it is taped, recomputed or freed before the peak under each checkpointing mode, and which phase sets the peak (forward, decoder recompute, integrator recompute, loss backward). It must explain the measured slopes: AMP both ≈ 3.4, AMP integrator ≈ 5.0, AMP none ≈ 6.5, fp32 both ≈ 5.8 GB per image.
- R1.3 Confirm or refute the 2³¹-element per-tensor hypothesis behind the linear fp32 N = 12 `d2` failure. Name the op or kernel and the exact limit (elements or bytes) by a reproducer that needs no 48 GB GPU, for example a single op on a tensor just above and just below the limit, or an analysis of the error text. Fix the measurement tool so linear measurements no longer run the deform build forward at the linear N (match `main.py`'s `n_build` rule).
- R1.4 Scale the breakdown to level 1 per component (8× for full- and half-resolution parts, 1× for L3 and coarser). List the largest single tensors and their element counts for the default and candidate configurations.

### R2. Memory knobs, all off by default (deliverable 2)

- R2.1 Width scaling: any candidate width configuration must be expressible from the CLI and in `ModelConfig`, either through the existing `--filters` / `--decoder-widths` or through one added option such as a width multiplier or named preset. It must keep the cross-level invariants: identical variables at levels 4, 2 and 1, `copy_weights_from` working, and the 5×7×6×24 bottleneck. Default = today's widths.
- R2.2 Encoder gradient checkpointing, and a `--gradient-checkpointing` mode that recomputes encoder, decoder and integrator (A2), available in `main.py`, the `Trainer`, the cascade paths and the measurement tool. Requirements:
  - (a) Gradients equal the non-checkpointed gradients within the existing recompute tolerance (1e-4 relative, as for the decoder), in fp32 and under AMP.
  - (b) In training mode with unfrozen BN statistics, every BatchNorm moving statistic is updated exactly once per step. Not twice, not from the recompute.
  - (c) Works with BN statistics frozen (cascade after recalibration) and with `recalibrate_bn`, which must still make exactly one update per layer and batch.
  - (d) One trace per task stays one trace.
  If the design also changes the double BN update of today's `decoder` / `both` modes, the change must keep the goldens at max |diff| 0 and be documented.
- R2.3 Integration steps: `--integration-steps 6` must be usable at level 1 while the frozen stages keep their own T. Today the cascade config check requires equal ModelConfigs, so either it must allow `integration_steps` to differ between the frozen stages and the trainable stage, or the design must state that T is changed for all levels in the retraining. The design picks one.
- R2.4 Cheap wins found by R1 (candidates to evaluate, not mandates):
  - chunked one-hot creation and label warp with template / pair Dice in label or image chunks;
  - per-squaring-step or per-image checkpointing of the integrator;
  - a memory-lean trilinear sampler gradient that does not keep the 8 corner weight and value tensors;
  - the group-mean concat computed as conv_a(x) + conv_b(mean), which never materialises the doubled-channel tensor. The existing kernel would be split, so weights and checkpoints stay compatible.
  - splitting full-resolution convs over the batch axis to stay under the per-tensor limit.
  Each knob adopted in Phase 1 is a flag, off by default. With the knob on, its numerics must match the reference within a stated tolerance: bit-identical where the op order is unchanged, else ≤ 1e-6 relative in fp32 for losses and ≤ 1e-4 relative for gradients.
- R2.5 With every new option at its default, levels 4 and 2 are bit-identical to today: both goldens (`tests/fixtures/m7_linear_golden.json`, `tests/fixtures/lncc_deform_golden.json`) pass with max |diff| 0. Existing checkpoints and `config.json` files (`m7_three_task_lncc`, `m8_level2_gradnorm`) still restore and pass the cascade config check without change.
- R2.6 Every new option is validated at parse time:
  - unknown values fail via `parser.error` with the option named;
  - widths must be positive integers, with the same entry counts as today;
  - integration steps ≥ 0;
  - chunk sizes ≥ 1;
  - a knob that is incompatible with the current level or task fails with a clear message rather than being silently ignored.

### R3. Calibrated analytic memory model and level-1 prediction (deliverable 3)

- R3.1 A pure-Python memory model with no TF import (runs in `bgr_preproc38` and in the container). Inputs: level, widths (the full ModelConfig), task, N, checkpointing mode, AMP, T, LNCC on/off, pair fraction, number of label classes, the adopted R2.4 knobs, and the cascade stages. Outputs: predicted peak GB, a per-component breakdown, resident GB, the largest single tensor (elements and bytes), and a feasibility flag that also fails on the per-tensor limit from R1.3.
- R3.2 The model is built from tensor inventories derived from the architecture: channels × voxels × bytes × liveness per checkpointing mode. A small number of fitted coefficients (for example allocator overhead, cuDNN workspace, transposes) is allowed, and every one of them is listed with its fitted value. The calibration points are stored in a checked-in JSON transcribed from section 2 and the local memory JSONs.
- R3.3 Accuracy: every calibration point is reproduced within ±15 %: level-4 deform N 8/12/16; level-2 deform fp32 and AMP × none / integrator / both at each measured N; level-2 linear fp32 and AMP; the GTX 1080 narrow and default points. A leave-one-mode-out check is also reported: refit without one checkpointing × AMP group, then predict it. The OOM and failed points must be predicted as infeasible: level-2 fp32 none N6, the linear fp32 N12 `d2` failure, and the level-4 GTX 1080 OOMs under the 6000 MB cap.
- R3.4 Level-1 prediction grid:
  - width configurations: default, at least three uniform reductions (for example ×0.75, ×0.5, ×0.375), and at least two targeted reductions of only the full- and half-resolution stages (f0, f1, D2, D1);
  - checkpointing: none / both / full; AMP: false / true; T: 6 / 7;
  - deform N 2–8; linear N 4–16.
  It also reports the frozen-stage cost of the 3-level cascade at level 1: level-4 and level-2 inference only, weights, and the persistent U and U⁻ fields (measured or bounded).
- R3.5 T = 6 vs T = 7:
  - Quantify the memory saving (GB per image at level 1, from the model and measured at a locally feasible size).
  - Quantify the accuracy cost on synthetic fields: max and mean |exp(v) ∘ exp(−v) − id| in voxels (both orders, interior margin as in the existing tests) and the folding fraction. Use the level-4 grid (comparable to the documented 0.010 / 0.0019 at T = 7) and the level-1 grid, with velocity magnitudes of at least max |v| = 2, 4 and 8 voxels.
  - Proposed acceptance for adopting T = 6 (A5): mean inverse error ≤ 2× the T = 7 value and ≤ 0.05 voxel, and folding fraction ≤ 0.1 %, at the largest magnitude expected at level 1. T = 6 is recommended only if it raises the feasible deform N or gives ≥ 10 % headroom.
- R3.6 Recommendation: exactly one configuration (widths, checkpointing mode, AMP, T, adopted knobs, deform N, linear N) chosen by a stated rule:
  - maximise deform N at level 1 (≥ 4 required) within 44 GB, including frozen stages and resident memory;
  - tie-breaks in this order: the wider model (more parameters), T = 7, fewer non-default knobs.
  It must state:
  - the predicted level-1 peak for deform at the chosen N and for linear at the largest linear N ≤ 16, with linear N ≥ 8 targeted;
  - the A100 verdict (≤ 36 GB) for both, and the largest deform N that fits the A100;
  - predicted peaks and group sizes at levels 4 and 2 under the same configuration;
  - predicted step times per task at levels 4, 2 and 1 on RTX 8000 and A100. These scale from the measured step times and recompute overhead and are labelled as estimates with a stated uncertainty.
  - predicted wall time of a long level-1 run, flagged if it exceeds the 7-day partition maximum or needs requeue / resume.
- R3.7 If no width configuration reaches deform N ≥ 4 within 44 GB without changing the architecture beyond widths and the R2.4 knobs, the report says so, names the smallest extra change that reaches it (for example per-image chunking of a stage, or a different D1 design), and stops for Jay's decision. Dropping below N = 4 is not chosen silently.

### R4. Level-1 measurement tooling (deliverable 4)

- R4.1 Extend `tests/gpu/measure_level_memory.py`:
  - `--level 1`;
  - arbitrary width configurations: the R3.4 grid by name or by explicit `--filters` / `--decoder-widths` values, all part of the configuration key;
  - the new checkpointing mode and the adopted R2.4 knobs, each in the configuration key;
  - `--integration-steps 6,7` as a grid axis.
  It records the predicted peak from R3 next to each measured peak (predicted / measured ratio) and the largest-tensor size. The linear build pass follows `main.py`'s `n_build` rule (R1.3). Existing level-4 / level-2 invocations and the JSON schema keep working: new fields only add to the schema and old files still merge.
- R4.2 At level 1 without an implemented 3-level cascade, the tool measures the trainable step with `--cascade false`. It separately measures, or records as a bound, the frozen-stage inference peak at level-1 input sizes using the existing level-4 stage plus a stand-in level-2 inference forward. The JSON labels this as a stand-in.
- R4.3 New `slurm/m9_memory.sh`, following `slurm/m8_memory.sh`:
  - `-A qtim` only (never `entre`), `--gpus=1`, `-N 1`, `--ntasks-per-node=1`, `--cpus-per-task=3`, `--mem=64G` (header note: up to ~128G allowed), `-p rtx8000` by default, with the `dgx-a100` variant documented in the header;
  - logs to `/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out|.err`, results under `.../GroupwiseReg/scratch/m9/`;
  - `tf10_nv.sif` via `singularity exec --nv` with the standard binds, `TMPDIR` under `SCRATCH_ROOT/$SLURM_JOB_ID`, `</dev/null` on every call, AMP grid;
  - a measured grid built around the R3.6 recommendation and its neighbours: one width step wider and narrower, T 6 / 7, deform N 2–8, linear N 4–16, with validation passes for deform;
  - an estimated `--time` derived in the header;
  - the "DO NOT submit without Jay's approval" note and the exact `sbatch` command.
  The script is added to the `bash -n` loop in `tests/run_tests_wsl.sh` and covered by a stub test like the m8 scripts (`tests/test_slurm_scripts.py`).
- R4.4 A local level-1 run of the tool with tiny widths at N = 2 under the 6000 MB cap either completes or records `oom` without crashing. Status values, OOM skipping and the JSON merge behave as at level 2.

### R5. Design of the rest of milestone 9, not implemented (deliverable 5)

The Phase-1 design document must specify, concretely enough to implement later:
- R5.1 3-level cascade: frozen level 4 and level 2, trainable level 1.
  - Generalise `FrozenStage` / `PREV_LEVEL` to a chain.
  - Pooling inputs 4× and 2×; linear composition A = Center(R · A2 · A4); deformable composition over three levels with the exact inverse unrolled factor by factor.
  - Checkpoint and fingerprint storage of both frozen stages; resume rules.
  - Level skipping per cascade position (design 4.7: p_skip for level 2 = 0.15 while training level 1, default 0 in code).
  - BN recalibration at level 1.
  - Where the frozen stages' memory and time go.
- R5.2 Level-1 DALI pipeline:
  - full-resolution `image.npy` / labels (27.5 + 6.9 MB per case);
  - read throughput for N = 4–8 with 3 CPUs vs the predicted step time;
  - the `warp_affine` rules (no float16; inverse map);
  - the verification checks to run (`verify_dali_pipeline.py --level 1`).
- R5.3 Level-1 validation:
  - validation group size(s);
  - `--val-pair-chunk` and any label or image chunking needed so the validation peak stays ≤ 44 GB (and the A100 verdict);
  - frozen baselines (`val_<set>_frozen`) for both frozen stages;
  - expected validation wall time.
- R5.4 Retraining sequence with the recommended configuration:
  - level 4: linear-only warm-up with curriculum and delayed G as in m5, then three-task with LNCC as in m7, with the gradient-balance arm;
  - level 2: cascade as in m8;
  - level 1: cascade.
  For each step: script, group sizes, steps, estimated wall time and GPU type. Order and concurrency must respect the two-active-jobs limit, and each submission needs Jay's approval.
- R5.5 Comparison criteria for the retrained narrower model, proposed in A5: center RMS no worse than +10 % against `m7_three_task_lncc` (level 4) and `m8_level2_gradnorm` (level 2), deform template / pair Dice no worse than −0.01, folding ≤ 0.05 %. If the criteria fail, the decision returns to Jay.

### R6. Reporting

- R6.1 A Phase-1 report (`.agents/tasks/milestone-9/report.md`) gives the R1 breakdown, the knobs and their verification, the memory-model accuracy table (R3.3), the level-1 grid and the recommendation (R3.4–R3.6), the T = 6 vs T = 7 numbers, the tooling, the exact `m9_memory.sh` command for Jay, and the R5 design pointers.
- R6.2 After implementation, `design.md` sections 4.5 (checkpointing modes), 4.6 (T at level 1), 9 (level-1 memory) and 11 (milestone-9 row) are updated, with a version bump.

## 5. Non-functional requirements

- N1. Cluster compatibility: Python 3.8 syntax, TF 2.10, numpy 1.21, scipy 1.4 (container `tf10_nv.sif`). Checked with `py_compile` and `ast` `feature_version=(3, 8)` in `bgr_preproc38`.
- N2. TF32 stays off: `configure_tf_precision()` first in every GPU entry point.
- N3. Tests follow the repo pattern: unittest, `tests.tf_setup` imported first, tiny or narrow widths, levels 4 / 2 or tiny level 1. New tests should add no more than ~20 minutes to the ~3 h suite, and the added time is stated in the report.
- N4. No cluster access (no ssh, sbatch or git pull) and no commit or push in Phase 1.

## 6. Acceptance criteria

1. The full local suite (`tests/run_tests_wsl.sh`) passes, and both goldens report `max |diff| 0` (R2.5).
2. A test restores a config.json / ModelConfig dict of today's format without the new keys and gets today's defaults. `check_cascade_config` accepts today's level-4 → level-2 pair unchanged (R2.5).
3. The full-checkpointing mode passes:
   - (a) a gradient-equality test against `none` within 1e-4 relative, in fp32 and AMP;
   - (b) a test that each BN moving mean / variance changes exactly as one update would, in training mode with unfrozen statistics;
   - (c) a frozen-stats test;
   - (d) a `recalibrate_bn` exactly-one-update test;
   - (e) a one-trace-per-task test (R2.2).
4. Each adopted R2.4 knob has an on/off equivalence test within its stated tolerance, and its default-off path is bit-identical (R2.4).
5. Parser tests cover every new option's invalid values (R2.6).
6. The per-tensor-limit reproducer runs locally and prints the op and limit. With the fix, the measurement tool's linear configs build at `min(N, deform N)` (R1.3).
7. Memory-model unit tests (no TF) reproduce every calibration point within ±15 %, flag every recorded OOM or failure as infeasible, and print the leave-one-mode-out errors (R3.3).
8. The model prints the level-1 grid of R3.4 and a recommendation with predicted deform N ≥ 4 at ≤ 44 GB, including frozen stages. If none exists, it prints the R3.7 verdict instead (R3.6, R3.7).
9. The T = 6 vs T = 7 test prints inverse-consistency (max, mean, both orders) and folding at the level-4 and level-1 grids for max |v| 2 / 4 / 8 voxels, and the report states the R3.5 verdict.
10. `measure_level_memory.py --level 1` with tiny widths at N = 2 runs locally per R4.4. The existing level-2 test (`tests/test_measure_memory.py`) still passes. The tool's JSON contains `predicted_peak_mb` and `largest_tensor_elems` (R4.1).
11. `bash -n slurm/m9_memory.sh` passes. The stub test asserts `-A qtim`, no `entre`, `--gpus=1`, `-N 1`, `--ntasks-per-node=1`, `--cpus-per-task=3`, `--mem=64G`, `-p rtx8000`, the log paths, the sif path and `</dev/null` on every container call (R4.3).
12. The report contains every item of R6.1. The design document covers R5.1–R5.5, each with concrete file, flag and number choices.

## 7. Out of scope (Phase 1)

- Per-step label sampling and the full SynthSeg label set (design section 11 addendum): deferred until the model is done.
- 8-flip test-time augmentation (rigid quaternion-matrix averaging, deform SVF averaging): later, with Jay.
- Implementing the 3-level cascade, the level-1 DALI pipeline, level-1 validation changes and the retraining runs (Phase 1 designs them only, R5).
- Any cluster action: ssh, `git pull`, `sbatch`, `scancel`, or running the m9 memory job or any training.
- Any git commit or push.
- Encoder-only gradient normalisation and the broader grad-norm debug (after all milestones).
- The affine G-gap follow-up (tabled; revisit with level-1 results).
- Changing the linear-decoder widths, dropout, losses or the task schedule, except where R1 shows a level-1 memory need.
