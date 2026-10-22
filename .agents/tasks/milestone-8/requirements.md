# Milestone 8 requirements: level 2 (1/2 resolution) with a frozen level-4 cascade

Status: requirements (input to the design phase). Date 2026-10-21. Design doc baseline: `.agents/tasks/network-design/design.md` v3.10.

## Summary

Move training from level 4 (40×56×48, 4 mm) to level 2 (80×112×96, 2 mm). The level-2 model is initialised from the best level-4 three-task model ("weight copy"). The level-4 model stays frozen (inference mode, BN in eval) as the first cascade stage. The level-2 stage predicts residuals: a residual linear transform on images pre-warped by the level-4 estimate, and a residual SVF on top of the upsampled level-4 deformation. Milestone 8 also covers BN recalibration of the copied model, level skipping (or a documented deferral), the level-2 data path and validation, level-2 memory measurement and group sizes, new SLURM recipes with the gradient-balancing switch, checker extensions, and a decision point for full-label sampling. All new behaviour is off by default, so level-4 runs stay bit-identical.

Design-doc sources: section 4.3 (encoder per level, parameter-free pools), 4.5 (decoder per level, parameter-free upsamples), 4.6 conventions table (Cascade row), 4.7 (multi-level composition, frozen levels, level skipping), 5.1 (coarse-to-fine procedure: copy weights, recalibrate BN then freeze, 1/4 model frozen as the first stage, N 8–12 at 1/2), 5.2 (BN momentum, deform BN gap), 9 (memory), 11 (milestone 8 row and the label-sampling addendum), 12 (open question 3).

## What already exists (verified by reading the code; reuse, do not rebuild)

- `--level {4,2,1}` in `runtime/args.py`, with per-level default group size (32 / 8 / 2) and LNCC window (`P.NCC_WINDOW_DEFAULT` 5 / 7 / 9).
- `data_loading/params.py`: `LEVEL_SHAPE`, `LEVEL_SPACING`, `mm_to_voxels`; DALI pipelines, postprocess, `fixed_data` validation sets and PNG writers all take a `level` argument. The export has `image_2.npy` and `labels_train18_2.npy`.
- `models/group_net.py`: `GroupRegLevelModel(level, cfg)` has identical variables at levels 4 / 2 / 1 (parameter-free pools and upsamples); `copy_weights_from(other)` copies all variables when the ModelConfigs match (seed excepted), tested in `tests/test_group_net.py`. Linear heads output translations in mm (`translation_scale` 1), so they are level-independent; SVFs are in voxels of their level.
- `models/diffeo.py`: `upsample_displacement` (2 × trilinear ×2, half-pixel) and `compose_cascade(u_prev_up, u_inv_prev_up, u_new, u_inv_new)`, tested in `tests/test_diffeo.py` (two-level round trip, zero previous level = single level).
- `models/layers.py`: `BatchNormWithSeparateAffine(stats_frozen=...)` and `set_bn_stats_frozen` (frozen = moving statistics even with `training=True`; re-trace needed, `Trainer.rebuild_steps()`).
- `runtime/checkpoint.warm_start_model`: strict whole-model restore for identical configs, partial restore for `DECODER_ONLY_FIELDS`. It builds the source model at `model.level`, so a level-4 checkpoint into a level-2 model with the same config is expected to restore strictly (not yet exercised).
- Validation reports `val_<task>_trainmode/*` (train-mode BN, no statistics update, dropout off) next to eval mode.
- `--task-grad-norm`, `--task-grad-norm-min`, `--task-loss-weights`, `diag_enc_grad_norm_{raw,eff}`; `tests/gpu/check_mixed_run.py` prints `enc_grad_ratio` as INFO only.
- `tests/gpu/measure_deform_memory.py` (deform only, level 4 only, no AMP or cascade option).

## Fixed constraints (decided; not open in the design phase)

- C1. Environment: cluster code runs in `tf10_nv.sif` (Python 3.8.10, TF 2.10.0, DALI 1.18.0, numpy 1.21.1, scipy 1.4.1). Python 3.8 syntax only. TF32 off in every process (`models.precision.configure_tf_precision()`).
- C2. Model and optimiser: Adam 3e-4 (cosine, warmup), LeakyReLU 0.01, dense-path LayerNorm, conv-path BatchNorm (momentum 0.99), pre-head norm-act kept, no group concat at the first dense layer, global clipnorm 12, one group per batch, fixed N per task.
- C3. Deform loss default at level 2: template soft Dice 1 + pairwise soft Dice 1 + LNCC template 1 + LNCC pair 1 + first-order smoothness 0.5; LNCC window 7 (level-2 default). Deform-step inputs use LUMIR's existing affine alignment (A = I in deform steps; design 6.2 v3).
- C4. Gradient balancing: try per-task gradient normalisation (`--task-grad-norm 12 --task-loss-weights none`, min 1e-3). Known caveat: it normalises each task's total gradient, not its encoder share. Decision rule (Jay): if the level-2 smoke run's checker `enc_grad_ratio` is outside [0.1, 10] (after any zero-head warm-up), fall back to clip 12 + `--task-loss-weights deform:64`. No deeper grad-norm redesign in this milestone.
- C5. Augmentation: group flips on all 3 axes (p 0.5 each), training only. G at 45° / 10 mm; curriculum off for warm-started runs. Task ratio rigid:1, affine:1, deform:2.
- C6. Warm-start source: `results/m7_three_task_lncc` (Dice + LNCC 1/1, deform:64), `ckpt/best` = step 11000 (linear criterion), `ckpt/best_deform` = step 10500 (pair Dice). Level-4 val at 12k: rigid 3.82, affine 4.80, rigid_g 3.93, affine_g 6.55 mm; deform template / pair Dice 0.757 / 0.701 (identity 0.666 / 0.552).
- C7. Labels: Jay has exported only `labels_train18*` (optional `lumir10`). Level 4 keeps train18 in any case.
- C8. Cluster process: no commits, pushes or cluster actions in this milestone's implementation steps; the orchestrator commits, pushes and submits after Jay approves. Cluster order: level-2 memory job first, then smoke, then long. At most two active jobs; account `qtim` only.

## Functional requirements

Each requirement has a rationale (R) and numbered acceptance criteria (AC). "Locally" means the WSL `tf_gpu` test environment (GTX 1080, small configs); "cluster" means a job run by the orchestrator after Jay's approval.

### FR1. Level-2 model initialised from level-4 weights

R: Every trainable parameter was trained at level 4 with large groups (design 4.3, 5.1). The level-2 trainable stage starts from those weights.

- AC1.1 A level-2 run can be initialised from a level-4 checkpoint directory (prefix or `ckpt/best`, `ckpt/best_deform`, `ckpt/last`) with the same ModelConfig. Every variable of the level-2 model is either copied from the source (array-equal) or explicitly listed as re-initialised by a documented design decision (Q2). A unit test asserts the copied count and array equality.
- AC1.2 The transfer rejects a source whose ModelConfig differs in any field other than those the design explicitly allows, with a ValueError naming the fields (same rule style as `warm_start_model`).
- AC1.3 config.json and an `init_weights` event in metrics.jsonl record the source path, the resolved checkpoint (step), the source level, the mode, and the restored / fresh parts with variable counts.
- AC1.4 The design document states, per model part (stem, encoder L2–L6, 1×1 reductions, dense layers, rotation/translation head, scale/shear head, centering, deform decoder D5–D1, SVF head), whether its shapes or output units depend on the level, with the evidence (code reference or test). In particular: the flattened bottleneck is 5×7×6×24 at both levels; translations are in mm; the SVF head output is in voxels of its level (half the mm per unit at level 2).
- AC1.5 The brain second-moment matrix S used by the linear losses at level 2 is computed for level 2; a test reports its difference to the level-4 S (expected small, since both are in mm about the same center).

### FR2. Frozen level-4 stage

R: Design 4.7 / 5.1: prior levels run outside the GradientTape in inference mode, and their outputs are composed inside the tape.

- AC2.1 The level-4 stage is a separate `GroupRegLevelModel(4, cfg)` restored from the warm-start source (which checkpoint(s): Q1). It runs with `training=False` (BN moving statistics, dropout off), outside the GradientTape. A test asserts that none of its variables receive gradients or change after K level-2 training steps (array-equal), and that it is not in the optimiser's variable list.
- AC2.2 Its inputs are the level-4 versions of the same group (how they are produced: Q3), so both stages see the same group, flips and augmentation.
- AC2.3 Resume (`--resume`) of a level-2 run reproduces the same frozen stage: either its weights are in the level-2 checkpoint or they are re-loaded from the recorded source and verified (Q14). A test resumes a short local run and checks the frozen stage's outputs on a fixed group are identical before and after.
- AC2.4 All cascade behaviour is behind a flag that is off by default; a level-4 run never builds a frozen stage.

### FR3. Linear cascade (rigid and affine)

R: Design 4.7: warp image i into the current central estimate (pull with A_i^(4)⁻¹), predict a residual R_i, A_i^(2) = Center(R_i · A_i^(4)). Losses and targets apply to the final composed and re-centered A_i (design 6.1).

- AC3.1 Linear composition unit test (numpy reference): for random invertible A_i^(4) and residuals R_i, the composed and centered result matches the reference to 1e-5 (float64) / 1e-4 (float32), and its group barycenter is the identity (rotation < 1e-4 deg, translation < 1e-4 mm).
- AC3.2 Identity residual test: with R_i = I for all i, the composed result equals Center(A_i^(4)) to 1e-5.
- AC3.3 Pre-warp consistency test: warping an image by (A^(4))⁻¹ and then by R⁻¹ matches one warp by (R A^(4))⁻¹ (interior NCC > 0.99 on a smooth phantom at level 2), confirming composition order and mm conventions.
- AC3.4 Pairwise maps from the composed A_i satisfy A_j⁻¹ A_i, and the pairwise loss uses the composed matrices; barycentric targets A_i* and pair targets T_j T_i⁻¹ are unchanged from level 4 (computed from T_i, as now).
- AC3.5 A local synthetic overfit at level 2 (narrow widths, small N) with the cascade on reduces the train-mode center RMS below the frozen stage's own error on the same group.
- AC3.6 Each linear task stays one traced `tf.function` over the run (trace counts logged and asserted in a test), with cascade, flips and G on.

### FR4. Deformable cascade

R: Design 4.6 Cascade row and 4.7: φ^(2) = up(φ^(4)) ∘ exp(v^(2)) → `compose_cascade(up(u_4), up(u_inv_4), u_2, u_inv_2)`; inverse compose(u_inv_2, up(u_inv_4)); every v stored; exact unrolled inverse.

- AC4.1 The level-2 deform stage sees the level-2 image warped by up(u_4) (the frozen stage's forward field) and predicts a residual SVF v^(2) in level-2 voxels, group-mean centered (mean_i v_i^(2) = 0 to 1e-5 at every voxel, tested).
- AC4.2 Zero-residual test: with v^(2) = 0 the composed field equals up(u_4) and its inverse equals up(u_inv_4) to 1e-6.
- AC4.3 Unrolled inverse over 2 levels at model level: on a synthetic group with smooth fields (max 2 voxels per level), interior |Ψ ∘ Ψ⁻¹ − id| max ≤ 0.05 level-2 voxels and mean ≤ 0.01, both orders (tolerances may be revised in the design with measured values; the single-level reference is 0.010 / 0.0019).
- AC4.4 Template Dice, pairwise Dice, both LNCC terms, the Jacobian / folding metric and inverse consistency are computed on the composed field (Ψ_i = composed φ_i; pairs compose(u_i, u_inv_j) of the composed fields). Which field the smoothness term penalises (composed or residual) is a design decision (Q6) and is tested either way.
- AC4.5 A local synthetic overfit at level 2 with the cascade on raises template and pair hard Dice above the frozen stage's values on the same group, with folding < 0.1 %.
- AC4.6 One traced deform `tf.function` over the run (trace count asserted).

### FR5. Level skipping

R: Design 4.7 "Level skipping" (v2): per step and per prior level, bypass that level with probability p_skip (decision once per group, both cascades, never at inference). The defaults (p_skip ≈ 0 for the first cascade level, 0.15 for the 1/2 level while training the full level) use cascade-position names ("level 1" = 1/4, "level 2" = 1/2), not the code's level numbers.

- AC5.1 The design states what skipping means at milestone 8 (only prior level: the 1/4 stage; design default ≈ 0) and either (a) implements it behind a flag with default 0, or (b) documents the deferral to milestone 9 with the reason. Either way the design doc records the naming clarification.
- AC5.2 If implemented: a bypassed stage gives A^(4) = I for linear and u_4 = u_inv_4 = 0 for deform; the decision is a seeded stateless draw per step (reproducible on resume, like the task draw); it causes no retracing (trace counts asserted); the per-window skip fraction is logged; validation never skips; with p_skip = 0 the train step is identical to the cascade without the feature (test).

### FR6. BN recalibration

R: Design 5.1: recalibrate BN statistics on level-2 inputs with forward-only passes, then freeze them. The copied BN moving statistics were estimated at level 4 (different activation statistics at the 2× finer stem and L2), and milestone 7 showed an eval vs train-mode gap for the deform decoder.

- AC6.1 A CLI flag (off by default) runs the recalibration before the first level-2 training step. Its parameters (number of batches, which tasks' inputs, reset vs momentum update, freeze vs continue afterwards) are fixed in the design (Q9, Q10) and recorded in config.json and a `bn_recalibration` event (batches used, layers updated, elapsed time).
- AC6.2 Recalibration changes BN moving statistics only: a test asserts that every trainable variable, the optimiser state and the step counter are bitwise unchanged, and that the moving mean/variance of every recalibrated BN layer changed.
- AC6.3 With the freeze option, every level-2 BN layer has `stats_frozen=True` afterwards and the train steps are (re)traced after the change (`rebuild_steps`); a test asserts the moving statistics do not change over K training steps.
- AC6.4 A validation pass is logged before and after recalibration (e.g. step 0 pre and post) with eval-mode and train-mode-statistics metrics for every task, so the gap can be compared. The gap definition must stay meaningful when statistics are frozen (with frozen stats `val_<task>_trainmode` equals eval mode by construction; Q9).
- AC6.5 Cluster: the smoke run reports the gap |val_<task> − batch-statistics metric| before and after recalibration for rigid, affine (center RMS) and deform (template / pair Dice). Expected: the gap after recalibration is not larger than before; the report states the measured values.
- AC6.6 The frozen level-4 stage is never recalibrated (it keeps its level-4 statistics, eval mode).

### FR7. Level-2 data, validation and PNGs

R: The pipelines are level-parameterised but have only run at level 4.

- AC7.1 DALI linear (rigid, affine) and deform pipelines run at level 2 with 3 CPU threads, emitting [N, 80, 112, 96, 1] images (and uint8 labels for deform); the GPU check script (`tests/gpu/verify_dali_pipeline.py` or a level-2 variant) passes its re-warp / pixel-center / label checks at level 2 (cluster, part of the memory or smoke job).
- AC7.2 Translations are specified in mm and converted with the level-2 spacing (existing rule); a test asserts level-4 and level-2 renderings of the same T agree in world coordinates (block-averaged level-2 rendering vs level-4 rendering, interior NCC > 0.98).
- AC7.3 Validation sets at level 2 from the 39 T1 val cases: linear (G = I and `_g`) and deform sets with group sizes derived from the measured memory (FR9), fixed composition and seeds; the frozen stage's inputs for validation are produced exactly as in training (Q3).
- AC7.4 PNGs at level 2 for every task (`val_png/<task>_step*.png`, `<task>_g_step*.png`, deform figures) with the level-2 spacing and mid-slice.
- AC7.5 Throughput: the smoke run logs the data-wait fraction per task; the report states whether 3 CPUs keep the GPU fed at level 2.

### FR8. Level-2 metrics, comparison with level 4, best checkpoints

R: The milestone-8 verification row requires "1/2 error < 1/4 error", and Jay tabled the G gap until level-2 results exist.

- AC8.1 Every level-2 validation pass also reports the frozen stage's own metrics on the same groups (linear center / pair RMS, rotation, translation, affine scale / shear / QR-rotation errors; deform template / pair Dice, folding, inverse consistency), under distinct keys, so the comparison uses identical groups and N. Because the stage is frozen these may be computed once and cached.
- AC8.2 Cluster long run acceptance: composed level-2 val errors < frozen-stage errors on the same groups for `val_rigid` and `val_affine` (center and pair RMS), and composed template and pair Dice > frozen-stage Dice, with folding < 0.1 % and inverse consistency reported in mm.
- AC8.3 The report gives the G gap at level 2 (val_affine vs val_affine_g, val_rigid vs val_rigid_g, pair RMS) with the rotation / shear / scale decomposition, next to the level-4 numbers (affine pair 6.92 vs 9.45 mm in the LNCC run), to answer Jay's tabled question.
- AC8.4 Best checkpoints keep the existing rules (`ckpt/best` on `val/mean_pair_rms_mm` of the composed linear results, `ckpt/best_deform` on `val_deform/pair_dice_loss` of the composed field).

### FR9. Level-2 memory measurement and group sizes

R: Level 2 has 8× the voxels at the stem, L2, D2, D1 and the integrator. Design 5.1 suggests N 8–12, but level-4 deform already needs ~2.15 GB per image, so N must come from measurement before the smoke run.

- AC9.1 A measurement script (extending `tests/gpu/measure_deform_memory.py` or a new `tests/gpu/measure_level_memory.py`) measures one training step at level 2 for `--task rigid|affine|deform`, with options: cascade on/off (frozen level-4 stage included), `--gradient-checkpointing none|integrator|decoder|both`, `--amp true|false`, an N grid, LNCC on/off, default and narrow widths. Output JSON per configuration: TF allocator peak, mean and median step time over ≥ 10 measured steps after warm-up (the 3-step level-4 timings were unreliable), OOM recorded as a result rather than a crash.
- AC9.2 The script runs locally at a small configuration (narrow widths, N 2) as a test that the code paths work; the cluster job produces the real table.
- AC9.3 The report gives the level-2 group-size recommendation per task (linear N, deform N) and the validation group sizes, each with its measured peak; a recipe "fits 48 GB" only if its peak is ≤ 44 GB on an RTX 8000 / A6000, and "fits the 40 GB A100" only if ≤ 36 GB; the report states both explicitly. If no deform N ≥ 4 fits without AMP or checkpointing, the report states which option is used and its measured step-time cost.
- AC9.4 The three-task process peak (the larger of the linear and deform step peaks, plus resident weights of both stages and Adam slots) is confirmed by the smoke run's `gpu_peak_mb`.

### FR10. SLURM recipes

R: Jay submits approved jobs; scripts must follow the cluster rules and the m7 template.

- AC10.1 New scripts under `slurm/m8_*.sh` provide (a) the level-2 memory measurement job, (b) the level-2 smoke run (~200 steps) and (c) the level-2 long run ((b) and (c) may be one script with env overrides, as in m7).
- AC10.2 Every script has `#SBATCH -A qtim` and no occurrence of `entre`; `--gpus=1`, `-N 1`, `--ntasks-per-node=1`, `--cpus-per-task=3`, `-p rtx8000` by default, `--mem=64G` (or a justified value ≤ 128G), `--time` set; `--output` / `--error` under `/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs`; results under `.../GroupwiseReg/results`; container `/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif`; `TMPDIR=/scratch/$SLURM_JOB_ID`; stdin closed. A test (or `bash -n` plus grep checks in the test suite) verifies these properties.
- AC10.3 The header documents the `-p dgx-a100` override and whether the recipe fits the A100 (FR9 numbers), and the exact `sbatch` lines with `--job-name`, `--time` and `--mem` for (a), (b) and (c); `--time` estimates are derived from measured step times (stated in the report).
- AC10.4 The train script has one env switch between the two gradient-balancing recipes: grad-norm (`--task-grad-norm 12 --task-loss-weights none`) and fallback (clip 12 + `--task-loss-weights deform:64`), with the decision rule (C4) in the header comment. The default is the grad-norm recipe (Jay: try it first).
- AC10.5 The train script fails fast (exit 2, message on stderr) if the warm-start / frozen-stage checkpoint path does not exist, passes `--level 2` and the cascade / recalibration flags, and runs the checker after `main.py` (exit code: main.py's if non-zero, else the checker's).

### FR11. Checker for level-2 / cascade runs

R: `tests/gpu/check_mixed_run.py` validated the m7 runs; level-2 runs need cascade-specific checks and the decision rule.

- AC11.1 New optional checks: cascade config and frozen-stage provenance in config.json (source path, step, level 4); frozen-stage baseline keys present in the validation records; BN recalibration event present when enabled; skip fraction keys when p_skip > 0.
- AC11.2 The checker evaluates the C4 rule explicitly: it prints the median applied encoder-gradient ratio linear/deform per linear task from windows after the warm-up cut (defined in the design, Q15) and a verdict line `grad_balance KEEP task-grad-norm` or `grad_balance FALLBACK deform:64`. It does not fail the run on FALLBACK (informational), so the long run is not marked failed by it.
- AC11.3 It reports (INFO) composed vs frozen-stage metrics at the final validation; the long-run acceptance AC8.2 is evaluated from these lines.
- AC11.4 Unit tests in `tests/test_check_mixed_run.py` cover the new checks on synthetic run directories (pass, fail and FALLBACK cases), and level-4 run directories still pass unchanged.

### FR12. Full label set with per-step label sampling (decision point)

R: Design section 11 addendum (Jay, 2026-10-20). It needs data Jay has not exported yet.

- AC12.1 The design phase records Jay's choice: (A) implement now, with `--dice-labels train18` as the default and tests on synthetic multi-label phantoms, or (B) defer to after the export. The requirements do not presume either.
- AC12.2 If (A): with `--dice-labels train18` the deform step is bit-identical to the current one (golden); with sampling on, the subset S = core ∪ K random non-core labels is the same for every image of the group, seeded per step, reproducible on resume, with no retracing; one-hot is built in TF after DALI with only |S| channels; validation scores every label (hard Dice in label chunks, no gradients) and reports core and all-label means separately; a test shows that the sampled loss averaged over ≥ 200 seeds matches the full-label loss on a fixed synthetic group within 2 % (or the reweighted variant within 1 %).
- AC12.3 Exports Jay would have to run (listed in the report and the design, whichever option): per level `_2` (and later full level) the full SynthSeg label map remapped to contiguous uint8 ids (e.g. `labels_full_2.npy`, `labels_full.npy`), produced with the same mode pooling as train18; a label table (contiguous id, SynthSeg id, name, core flag); manifest columns or a file-naming rule for the new label set in `export_manifest.csv`; the 39 T1 val cases included; Jay's list of core labels and the value of K. Level 4 keeps train18 (no `_4` full-label file required).
- AC12.4 If (B): the design doc marks the addendum as deferred with the reason, and milestone-8 runs use train18 at level 2.

### FR13. Backward compatibility (bit-identical level 4)

R: Every change is off by default where it alters existing behaviour.

- AC13.1 Golden tests `tests/fixtures/m7_linear_golden.json` and `tests/fixtures/lncc_deform_golden.json` pass with max |diff| 0.
- AC13.2 The full unit-test suite passes (`tests/run_tests_wsl.sh`; currently 327 tests), plus the new tests.
- AC13.3 Existing level-4 checkpoints (m5, m7) still restore via `--resume` and `--init-weights` unchanged; new config.json keys appear only with off defaults.
- AC13.4 Every modified or new module compiles under Python 3.8 (`bgr_preproc38` py_compile) and uses only APIs available in TF 2.10, DALI 1.18, numpy 1.21, scipy 1.4.

### FR14. Documentation and report

- AC14.1 `design.md` is bumped (v3.11) with the milestone-8 decisions: weight transfer, frozen stage, linear and deform cascade equations as implemented, skip, BN recalibration, level-2 memory table, group sizes, label-sampling status, and the answers to the open questions below.
- AC14.2 `.agents/tasks/milestone-8/report.md` lists files changed, tests run with counts, local verification results, the memory recommendation (or "pending cluster job"), and the exact sbatch lines with time estimates for the memory, smoke and long jobs, in the cluster order of C8.

## Non-functional requirements

- NF1. Step time: the frozen stage adds no backward pass; the report states its measured share of the level-2 step time.
- NF2. Numerics: grids, fields, compositions and losses in float32 (never float16 grids), including under AMP; TF32 off.
- NF3. Determinism: all new random decisions (skip, label subsets, recalibration batch order) are seeded and stateless per step so a resumed run reproduces them.
- NF4. Logging: every new behaviour logs its effective configuration once (config.json + stdout) and its per-window statistics in metrics.jsonl, flushed per record as now.

## Open design questions (to resolve in the design phase, not here)

- Q1. Warm-start checkpoint(s). `ckpt/best` (step 11000) vs `ckpt/best_deform` (step 10500) vs `ckpt/last` (12000). The encoder is shared, so a linear decoder from 11000 and a deform decoder from 10500 cannot be combined in one consistent model. Options: one checkpoint for everything, or two frozen level-4 models (linear stage from `best`, deform stage from `best_deform`). Also: is the trainable level-2 model initialised from the same checkpoint as the frozen stage?
- Q2. What the trainable level-2 stage copies. Copy every part (heads trained to predict full poses, applied to pre-aligned inputs) vs re-initialise the heads to identity residuals (zero kernel / identity bias, zero SVF head). The SVF unit change (level-2 voxels = half the mm per unit) makes a copied SVF head predict half-size displacements on already-warped inputs. A zero SVF head reintroduces the zero-encoder-gradient warm-up seen at level 4.
- Q3. How the frozen stage gets level-4 inputs: 2×2×2 average pooling of the level-2 DALI output in TF (consistent with the export's block averaging and cheap) vs a second level-4 DALI output with the same T_i (two reads, exact level-4 rendering). Same choice for validation sets.
- Q4. Linear composition details. Center only the composition, Center(R_raw · A^(4)), or center R first and then compose and center again; whether `GroupCentering` runs once or twice; whether the residual uses the same heads (10D rotation + t, R·K) and ranges.
- Q5. Pre-warp interpolation. The level-2 linear stage sees the DALI-augmented image warped again in TF by (A^(4))⁻¹ (a second linear interpolation and a second zero-fill boundary). Accept the extra blur, or another route.
- Q6. Deform cascade details. Smoothness on the composed field (design 6.2: "u_i = φ_i − id, composite deformable part at the current level") vs on the residual only; centering only the residual v^(2) (design) and tracking composed centrality as a metric; whether the frozen stage's u_4 is integrated at level 4 and then upsampled (design) and how its fields are cached for validation.
- Q7. Gradient flow. Confirm that composition with frozen outputs happens inside the tape and that no gradient reaches the frozen stage (stop_gradient vs running outside the tape), including through pre-warped images and up(u_4).
- Q8. Skip at milestone 8. Implement the mechanism now with default p_skip 0 for the 1/4 stage, or defer to milestone 9. If implemented: whether skipped linear steps (full ±45° residual from identity) need any range or curriculum consideration.
- Q9. BN recalibration procedure. Number of batches and which tasks' inputs (linear, deform, or both in the task ratio; with G and flips on?); on cascade pre-warped inputs (what the stage sees in training); reset statistics and accumulate an exact average vs momentum updates; how the "gap" is measured once statistics are frozen (batch-statistics mode vs moving statistics on the same groups); interaction with `--gradient-checkpointing decoder|both` (double statistic updates).
- Q10. After recalibration: freeze BN statistics for the rest of the run (design 5.1) or keep updating them. Small level-2 N makes per-step batch statistics noisier.
- Q11. Group sizes. Linear N and deform N at level 2 from FR9; validation group sizes and pair caps; how N affects comparability with level-4 metrics (the frozen stage was trained at N 32 / 16 and will run at the level-2 N, eval mode).
- Q12. Memory fallback order: smaller N vs gradient checkpointing vs AMP (`mixed_float16` is a flag but has not been validated on the deform task or the cascade).
- Q13. Level-2 schedule: total steps, warmup, LR (fresh Adam), val cadence, `ckpt_every`, task ratio (1:1:2 assumed), time budget per job.
- Q14. Checkpointing the frozen stage: store its weights in the level-2 checkpoint or reload them from the recorded source path on resume (and how a missing / changed source is detected).
- Q15. Decision-rule mechanics for C4: which windows count (if the SVF head is copied there is no zero-head warm-up; if re-initialised, the cut after ~100 steps); what happens if one linear task is inside [0.1, 10] and the other outside; whether the rule uses `diag_enc_grad_norm_eff` medians as at level 4.
- Q16. Label sampling now vs later (FR12, Jay's choice), and if now, the defaults for K and the reweighting option.
- Q17. G at level 2. G is applied to the group before both stages (the frozen stage must handle G as at level 4); confirm the `_g` validation sets render G at level 2 and pool consistently for the frozen stage.
- Q18. Whether a no-cascade level-2 baseline (single-level level-2 model from level-4 weights, existing `--level 2` path) is supported and run as an ablation, or left out.

## Out of scope

- Level 1 (full resolution) training, AMP and checkpointing as defaults for level 1, full-res memory profiling (milestone 9).
- Test-time flip augmentation (8-flip TTA), large-N chunked inference, OASIS / Mindboggle evaluation.
- A deeper gradient-normalisation redesign (encoder-only normalisation, capped gain); only the C4 switch.
- G-range changes and the affine-G investigation beyond reporting the level-2 numbers.
- Running preprocessing or exports on the cluster (Jay runs them), and any commit, push or job submission by the implementation steps.
- Config A vs B and other ablations.

## Assumptions

- The level-2 export (`image_2.npy`, `labels_train18_2.npy`, manifest entries for 3384 train + 39 T1 val) is complete and matches the level-4 export case for case.
- Linear tasks at level 2 use the same augmentation ranges as level 4 (rigid ±45° / 15 mm, affine with the m5 scale / shear ranges) since translations are in mm.
- The frozen stage runs at the level-2 group size of the task (linear N or deform N), not at its training N.
