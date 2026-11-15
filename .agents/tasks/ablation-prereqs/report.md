# Ablation prerequisites: report

Branch `ablation-prereqs` off main 8e12596 (worktree `.worktrees/ablation-prereqs`). Plan with decisions D1-D9: `plan.md`; linear-heads finding: `linear-heads.md` (same folder). No cluster job was submitted. Every command below needs Jay's approval before it runs.

## 1. What the three ablations need and what is in place

| Ablation | Launch path | Code status |
|---|---|---|
| A2b group-block placement: none / E / A vs B (final) | `slurm/r_cascade.sh` with `GROUP_ARGS='--group-config <X>'` | new `--group-config none` and `E` (FEAT-001) |
| A2 batch norm vs instance norm | `slurm/r_cascade.sh` with `NORM=instance` | IN already worked end to end; scripts gained `NORM` (FEAT-003) |
| A1 full resolution from scratch at matched GPU-hours | new `slurm/r_level1_scratch.sh` | main.py already trained level 1 without `--cascade-from`; new script (FEAT-003) |

### Group-config mapping (`--group-config`, ModelConfig field `group_config`)

| value | encoder (stem, L2-L6) | linear decoder (4 reductions + dense_1) | deform decoder (D5-D1) | paper name |
|---|---|---|---|---|
| `none` | no | no | no | no group blocks |
| `E` | yes | no | no | encoder only |
| `B` | no | yes | yes | decoders only (final model) |
| `A` | yes | yes | yes | everywhere (MultiMorph-style) |

How the linear heads got group information before this change (details in `linear-heads.md`): in both A and B the rigid / affine heads see the group through the linear decoder (the four 1x1 reduction blocks and `dense_1` concatenate the group mean); A adds the encoder. The literal encoder-only variant was chosen for E (Jay via the orchestrator, option 2), because "linear heads as in the existing configs" would have made E's linear branch identical to A. `--dense-group-concat-first true` with `none` or `E` raises a ValueError (it would add a group concat to the linear decoder).

Cross-image coupling that stays in every variant: group centering of A (barycenter) and of the SVFs, the template / pair losses, and BatchNorm in training mode, whose statistics are computed over the N images of the one group on the batch axis. BN is left unchanged for the group-config variants; the BN-vs-IN ablation covers it separately. For `none` with `--norm instance`, the network itself has no cross-image path at all (check-streaming differences are exactly 0 for `none`).

## 2. What changed

FEAT-001 (commit e23bffd), group configs:
- `models/config.py`: `GROUP_CONFIGS = ("A", "B", "none", "E")`; methods `group_encoder()` (A, E) and `group_decoders()` (A, B). They are methods, so `to_dict()` and config.json are unchanged and every existing config.json loads.
- `models/group_net.py`: encoder, linear decoder (reductions, dense blocks) and deform decoder stages use those methods instead of `group_config == "A"` / hard-coded `group=True`. A and B build the same graph as before.
- `runtime/args.py`: `--group-config` choices from `GROUP_CONFIGS`, help text with the mapping.
- `models/width_presets.py`, `models/memory_model.py`: the same flags in the TF-free mirrors. A/B memory predictions are bit-equal to main 8e12596 (pinned in tests).
- `inference/streaming.py`: ungrouped decoder stages skip the group sums; the element bound drops the decoder "group concat" item when the decoders are ungrouped.
- Tests: tests/test_group_net.py, test_split_group_conv.py, test_width_presets(_tf).py, test_memory_model.py, test_predict_streaming.py, test_runtime_infra.py, new tests/test_ablation_configs.py.

FEAT-002 (commit 95bf844), instance norm and from-scratch level 1: tests only, no production change was needed (section 3).

FEAT-003 (this commit), scripts:
- `slurm/r_cascade.sh`: new env `NORM` (batch | instance, default batch). `instance` appends `--norm instance` to every level's EXTRA_ARGS (after `GROUP_ARGS --lambda-smooth S`, before `EXTRA_ARGS_L<k>`) and sets `BN_RECAL_BATCHES=0` at levels 2 and 1. `group_args` and `norm` are written to the chain state file and compared on resume (exit 2 on a change unless `FORCE_RECIPE_CHANGE=1`). A state file written before this change has neither key and counts as `--group-config B` / batch, so chains started earlier resume unchanged. The header line prints `group args '...' | norm ...`. With `NORM` unset the level env and commands are identical to before.
- New `slurm/r_level1_scratch.sh` (section 5).
- `tests/run_tests_wsl.sh`: `bash -n slurm/r_level1_scratch.sh`.
- `tests/test_slurm_scripts.py`: classes `TestLevel1Scratch` and `TestCascadeAblationKnobs` (dry runs parse with `runtime.args.parse_args`; NORM / GROUP_ARGS reach every level; IN level-2/1 commands parse without the recal error; old state files resume; recorded values are enforced).

## 3. Already fine before this work (pinned by FEAT-002 tests)

- `InstanceNormWithAffine` computes moments and normalisation in float32 and casts back, so IN trains under mixed_float16 (L4 three tasks with config B, checkpointing full, lean sampler, split group conv; L2 cascade from an IN L4 source).
- IN models have no BN layers: no BN layers in the Trainer, the resume freeze re-apply is a no-op, gradient checkpointing builds no BN snapshot/restore ops.
- `--bn-recal-batches > 0` with `--norm instance` exits in the parser with a message naming `--norm batch`; `--bn-recal-batches 0` (also with `--bn-recal-freeze true`) parses.
- Streaming inference is exact for IN (per image, no running statistics): float32 max error 2e-6 vox; check-streaming on an IN L2 cascade passes; `predict.py run --tta flip8 --recenter 1` works on IN.
- Mixed group configs or norms across cascade levels are rejected by the existing cascade checks (`check_cascade_config`, `FrozenChain`, main.py exit naming the field); D3 keeps it that way. Each ablation is a cascade with one config at all levels.
- main.py trains level 1 without `--cascade-from` (no cascade record, no BN recalibration) and resumes with the same run name.

## 4. Decisions (plan.md D1-D9, short)

- D1 names: A / B unchanged, new `none` and `E`.
- D2 `none` removes every group concat; `E` = grouped encoder, ungrouped linear and deform decoders.
- D3 mixed group configs / norms across levels: rejected (no per-stage config).
- D4 IN under AMP needs no dtype change. Its float32 temporaries are not in the memory model; the L1 smoke measures them.
- D5 streaming changes only for ungrouped decoders; IN needed none.
- D6 from-scratch L1 schedule = the L4 phase-C schedule as fractions of STEPS (section 5).
- D7 GPU-hours budget 63.85 h (section 5).
- D8 `NORM` in r_cascade.sh, recorded with `group_args`.
- D9 ablations = r_cascade.sh with the final recipe plus GROUP_ARGS / NORM.

## 5. Ablation A1: GPU-hours and the step rule

sacct on mlsc (2026-11-12), RTX 8000: L4 job 10623341 13:54:14 + L2 job 10624169 27:20:44 + L1 job 10627751 22:36:02 = 229 860 s = **63.85 h**. Caveat for the paper: the L2 job shared its node with another job part of the time, so the budget is a little generous.

Step rule, from the smoke's `INFO step_time` mean step (s) and the validation pass time v (~462 s per pass at level 1, the first pass is slower because of tracing):

    STEPS = floor(229860 / (mean_step_s + v / VAL_EVERY))   (minus a few hundred steps for startup and the slower first validation)

Example: 11 s mean step, VAL_EVERY 500 -> ~19 300 steps. The final cascade's level 1 ran ~13 s mean step with 3 tasks at 1:1:2; from scratch there are no frozen stages, so the step should be somewhat faster.

`slurm/r_level1_scratch.sh` recipe: `--level 1`, no `--cascade-from`, no `--bn-recal-*` (BN trains normally as at level 4), d12hs widths, `--deform-decoder-dtype float32_fine`, `GROUP_CONFIG` B, `SMOOTH` 15, Dice + LNCC 1/1, p 0.1 / 0.1, ratio 1:1:2, N 8 / DEFORM_N 4, AdamW lr 3e-4 wd 1e-1 cosine with min-lr factor 1e-3, `--task-grad-norm 12`, AMP, split group conv, TF intensity aug, tent low-res, TF_MEM_LIMIT_MB 42500, checkpointing full + lean sampler, pair recompute, lean losses, min pairs n, max pairs 64, val image / pair chunk 1 / 1, val groups = N / DEFORM_N, curriculum on, G 45 deg / 10 mm. STEPS is required (no default). Schedule defaults from STEPS (integer bash arithmetic, each overridable): WARMUP STEPS/16, DEFORM_START 3*STEPS/16 with DEFORM_RAMP STEPS/32, CURRICULUM_RAMP 3*STEPS/16, G_START 3*STEPS/16, G_RAMP 3*STEPS/16; STEPS 16000 gives 1000 / 3000 / 500 / 3000 / 3000 / 3000, the L4 recipe. NORM=instance and GROUP_CONFIG=A|none|E work too (not needed for A1). Checker: `--expect-init-weights none`, grad-balance verdict from DEFORM_START + 100, memory budget 36864, non-finite limits as r_level1.sh. `#SBATCH --mem=128G --time=48:00:00`.

Resume: the run needs ~64 h, more than the 48 h job limit. Resubmit the identical line (same RUN_NAME and same STEPS). main.py restores ckpt/last (every 250 steps) and the RNGs; at most 250 steps are lost per resubmission. Changing STEPS on resume would change the cosine schedule and the derived fractions, so keep it fixed.

## 6. Memory and host notes

- Group configs: `none` and `E` have fewer and smaller group concats than A / B, so they need less memory than B at the same N (memory model, d12hs L2 N16 deform: A 41.95, B 41.85, E 41.57, none 41.47 GiB; the model over-predicts, these are relative numbers). Config A at L1 ran before (r1sw, config A, 37.8-38.6 GiB under 42500 MB).
- IN at L1: the float32 IN temporaries at full resolution are not modelled and not measured. The final recipe peaks 38-39 GB under `TF_MEM_LIMIT_MB=42500` on the RTX 8000. Check the IN L1 smoke's `INFO gpu_peak`: it must stay <= ~41500 MB. Fallback is `MIN_PAIRS=1` (a float32 decoder would need more memory, not less). The cascade script has no MIN_PAIRS knob (open decision 3).
- From-scratch L1: no frozen stages, so memory is at most the final L1's (38-39 GB). The local GTX 1080 could not execute an L1 deform step even at tiny widths (traced only), so the smoke is the first real run of this path.
- Host RAM: MaxRSS ~94 GB at level 1 -> `--mem=128G` at L1 (r_cascade.sh already requests 128G for the whole chain; r_level1_scratch.sh too). L4 / L2 smokes: 64G.
- int32 per-tensor limit: linear level-1 N <= 9 at default widths (job 10517266); the d12hs stem (16 ch) moves it to ~19. N stays 8.
- Wall time: each ablation chain is expected at ~64 h (L4 ~14 h, L2 ~21-27 h, L1 ~23 h), under r_cascade.sh's 72 h limit. If a chain hits the limit, resubmit the identical line: it continues at the running level from its ckpt/last with the recorded commit.

## 7. Commands (from the clone `/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad`, after `git pull`)

Before any sbatch: `mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs`, and `squeue -u jn85` (at most two active jobs). `R=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results`.

### 7.1 Dry runs (print everything, create nothing, no sbatch)

    env BGR_DRY_RUN=1 S4=2.5 S2=10 S1=15 SYNTH_PROB=0 STEPS_L2=16000 STEPS_L1=6000 EXTRA_ARGS_L2='--frozen-integration-grid current' GROUP_ARGS='--group-config none' RUN_NAME_L4=r4_abl_gnone RUN_NAME_L2=r2_abl_gnone RUN_NAME_L1=r1_abl_gnone bash slurm/r_cascade.sh
    env BGR_DRY_RUN=1 S4=2.5 S2=10 S1=15 SYNTH_PROB=0 STEPS_L2=16000 STEPS_L1=6000 EXTRA_ARGS_L2='--frozen-integration-grid current' GROUP_ARGS='--group-config B' NORM=instance RUN_NAME_L4=r4_abl_in RUN_NAME_L2=r2_abl_in RUN_NAME_L1=r1_abl_in bash slurm/r_cascade.sh
    env BGR_DRY_RUN=1 STEPS=19000 RUN_NAME=r1_scratch bash slurm/r_level1_scratch.sh

Check in the cascade output: `group args '...' | norm ...` in the header line, each level's `EXTRA_ARGS` (`--group-config X --lambda-smooth S [--norm instance]`), `BN_RECAL_BATCHES=0` in `ENV L2` / `ENV L1` for IN only, `STEPS=16000` at L2 and `STEPS=6000` at L1, and `dirty check: clean`.

### 7.2 Smokes (one job each; per new path)

Group config X in {none, E, A} (name tag `gnone`, `gE`, `gA`); each level from the previous smoke's ckpt/last. A at L4/L2/L1 has run before in other recipes, so its smokes are optional.

    # L4, 300 steps (~20 min on the RTX 8000)
    env RUN_NAME=r4_abl_gnone_smoke STEPS=300 VAL_EVERY=100 WARMUP=50 DEFORM_START=100 DEFORM_RAMP=100 CURRICULUM_RAMP=100 G_START=100 G_RAMP=100 PROJECT_STEPS=16000 PROJECT_VAL_EVERY=500 WD=1e-1 EXTRA_ARGS="--group-config none --lambda-smooth 2.5" sbatch --job-name=r4gn_smk --time=02:00:00 --mem=64G slurm/r_level4.sh
    # L2, 200 steps (~35 min)
    env RUN_NAME=r2_abl_gnone_smoke CASCADE_FROM=$R/r4_abl_gnone_smoke/ckpt/last STEPS=200 VAL_EVERY=100 WARMUP=50 PROJECT_STEPS=16000 PROJECT_VAL_EVERY=500 WD=1e-1 FROZEN_INTEGRATION_GRID=current EXTRA_ARGS="--group-config none --lambda-smooth 10" sbatch --job-name=r2gn_smk --time=03:00:00 --mem=64G slurm/r_level2.sh
    # L1, 200 steps (~1.5 h)
    env RUN_NAME=r1_abl_gnone_smoke CASCADE_FROM=$R/r2_abl_gnone_smoke/ckpt/last STEPS=200 VAL_EVERY=100 WARMUP=50 PROJECT_STEPS=6000 PROJECT_VAL_EVERY=500 WD=1e-1 TF_MEM_LIMIT_MB=42500 EXTRA_ARGS="--group-config none --lambda-smooth 15" sbatch --job-name=r1gn_smk --time=04:00:00 --mem=128G slurm/r_level1.sh

For E: replace `none` by `E` and `gnone` / `gn` by `gE` / `gE`. Instance norm (config B; BN_RECAL_BATCHES=0 at L2 and L1, as the cascade sets it):

    env RUN_NAME=r4_abl_in_smoke STEPS=300 VAL_EVERY=100 WARMUP=50 DEFORM_START=100 DEFORM_RAMP=100 CURRICULUM_RAMP=100 G_START=100 G_RAMP=100 PROJECT_STEPS=16000 PROJECT_VAL_EVERY=500 WD=1e-1 EXTRA_ARGS="--group-config B --lambda-smooth 2.5 --norm instance" sbatch --job-name=r4in_smk --time=02:00:00 --mem=64G slurm/r_level4.sh
    env RUN_NAME=r2_abl_in_smoke CASCADE_FROM=$R/r4_abl_in_smoke/ckpt/last STEPS=200 VAL_EVERY=100 WARMUP=50 PROJECT_STEPS=16000 PROJECT_VAL_EVERY=500 WD=1e-1 BN_RECAL_BATCHES=0 FROZEN_INTEGRATION_GRID=current EXTRA_ARGS="--group-config B --lambda-smooth 10 --norm instance" sbatch --job-name=r2in_smk --time=03:00:00 --mem=64G slurm/r_level2.sh
    env RUN_NAME=r1_abl_in_smoke CASCADE_FROM=$R/r2_abl_in_smoke/ckpt/last STEPS=200 VAL_EVERY=100 WARMUP=50 PROJECT_STEPS=6000 PROJECT_VAL_EVERY=500 WD=1e-1 TF_MEM_LIMIT_MB=42500 BN_RECAL_BATCHES=0 EXTRA_ARGS="--group-config B --lambda-smooth 15 --norm instance" sbatch --job-name=r1in_smk --time=04:00:00 --mem=128G slurm/r_level1.sh

From-scratch level 1 (300 steps, deform and G from 100, three validations; projection printed for 19000 steps):

    env RUN_NAME=r1_scratch_smoke STEPS=300 VAL_EVERY=100 WARMUP=50 DEFORM_START=100 DEFORM_RAMP=50 CURRICULUM_RAMP=100 G_START=100 G_RAMP=100 PROJECT_STEPS=19000 PROJECT_VAL_EVERY=500 sbatch --job-name=r1scr_smk --time=05:00:00 --mem=128G slurm/r_level1_scratch.sh

What to read from each smoke: checker PASS; `INFO gpu_peak` (L1 <= ~41500 MB under the 42500 limit; L4 / L2 as before, ~33 / ~28 GiB for B); non-finite steps only early (AMP warm-up); no collapse (`train_<task>/diag_feat_rel_spread` > 1e-3); `INFO step_time` (for the scratch STEPS and the chain wall time); for IN no `bn_recalibration` event; config.json `model_config.group_config` / `norm` as intended.

Priority if GPU time is short: the IN L1 smoke (unmeasured memory) and the scratch L1 smoke (first real run of that path, and it sets STEPS) matter most. none / E need less memory than B and their code paths are covered by local tests, so their L1 smokes are the least informative.

### 7.3 Long runs

Group-block chains (X = none | E | A), ~64 h each:

    env S4=2.5 S2=10 S1=15 SYNTH_PROB=0 STEPS_L2=16000 STEPS_L1=6000 EXTRA_ARGS_L2='--frozen-integration-grid current' GROUP_ARGS='--group-config none' RUN_NAME_L4=r4_abl_gnone RUN_NAME_L2=r2_abl_gnone RUN_NAME_L1=r1_abl_gnone sbatch --job-name=bgr-abl-gnone slurm/r_cascade.sh
    env S4=2.5 S2=10 S1=15 SYNTH_PROB=0 STEPS_L2=16000 STEPS_L1=6000 EXTRA_ARGS_L2='--frozen-integration-grid current' GROUP_ARGS='--group-config E' RUN_NAME_L4=r4_abl_gE RUN_NAME_L2=r2_abl_gE RUN_NAME_L1=r1_abl_gE sbatch --job-name=bgr-abl-gE slurm/r_cascade.sh
    env S4=2.5 S2=10 S1=15 SYNTH_PROB=0 STEPS_L2=16000 STEPS_L1=6000 EXTRA_ARGS_L2='--frozen-integration-grid current' GROUP_ARGS='--group-config A' RUN_NAME_L4=r4_abl_gA RUN_NAME_L2=r2_abl_gA RUN_NAME_L1=r1_abl_gA sbatch --job-name=bgr-abl-gA slurm/r_cascade.sh

Instance norm chain (config B):

    env S4=2.5 S2=10 S1=15 SYNTH_PROB=0 STEPS_L2=16000 STEPS_L1=6000 EXTRA_ARGS_L2='--frozen-integration-grid current' GROUP_ARGS='--group-config B' NORM=instance RUN_NAME_L4=r4_abl_in RUN_NAME_L2=r2_abl_in RUN_NAME_L1=r1_abl_in sbatch --job-name=bgr-abl-in slurm/r_cascade.sh

From-scratch level 1 (STEPS from the smoke with the step rule of section 5; resubmit the identical line after each 48 h limit):

    env RUN_NAME=r1_scratch STEPS=<from the smoke> sbatch --job-name=r1scr --time=48:00:00 --mem=128G slurm/r_level1_scratch.sh

The reference (B, batch norm) is the existing final chain: L4 `r4_cfgB_smooth2.5_real`, L2 `r2_cfgB_s16000_sm10_real_current`, L1 `r1_cfgB_s6000_sm15_real` (all last checkpoints). The chains above use the same recipe: r_cascade.sh's level-4 schedule equals r_level4.sh's defaults, lr 3e-4, wd 1e-1, d12hs, SYNTH_PROB 0, fine-grid integration at L2 (L1 inherits), the level-2/1 BN recal 64 with freeze false / true (batch norm only). Evaluation (Jay): LUMIR training-time validation of each run's last checkpoint (`metrics.jsonl` val records at the final step, e.g. via scripts/summarize_runs.py).

## 8. Verification

- FEAT-001 / FEAT-002: see their commits (targeted suites incl. goldens `[golden] max |diff| 0`, `[golden deform] max |diff| 0`, `[golden tta-off] 70 hashes compared, 0 differ`).
- FEAT-003: `bash -n slurm/r_cascade.sh slurm/r_level1_scratch.sh` OK; `python -m unittest tests.test_slurm_scripts` in bgr_preproc38 (no TF import): 88 tests, all new tests pass. `TestCascadeRuns.test_dry_run` fails ONLY inside the worktree under WSL because the worktree's `.git` file holds a Windows path (`git rev-parse HEAD` fails in WSL, "REPO ... is not a git clone"); it passes in a real clone (re-run in a WSL clone of the branch, see the commit message).
- Full suite (integration, commit 2410ac2, `tests/run_tests_wsl.sh` in the worktree): suite status 0, 1099 tests (5 skipped), 0 failing module invocations; TF-free stage 240 tests OK without TF; unittest wall time 29 829 s (8.3 h). Goldens: `[golden] max |diff| 0 (tolerance 0)`, `[golden deform] max |diff| 0 (tolerance 0)`, `[golden tta-off] 70 hashes compared, 0 differ`. Two fixes on the way: the worktree's `.git` file now uses a relative gitdir, so WSL git works there and `TestCascadeRuns.test_dry_run` passes in the worktree (local metadata, not committed); and the first full run's only failure, a float32 rounding miss in `test_permutation_equivariance` (config A rigid translation 2.03e-5 mm vs a 1.97e-5 mm bound), led to a 1e-4 mm translation floor for every config (test-only change, commit 2410ac2).

## 9. Open decisions for Jay

1. STEPS of the from-scratch run: from the scratch smoke's `INFO step_time` and the step rule (section 5).
2. Order and GPU budget: five long jobs of ~64 h each (none, E, A, IN chains and the scratch run) against the 2-job limit, about 7 days of wall time with two jobs running at once. Which first? Suggested: IN and none first (the two cleanest ablation claims), then E, A and the scratch run.
3. If the IN L1 smoke exceeds ~41500 MB: the fallback MIN_PAIRS=1 is a level-script env that r_cascade.sh unsets; the chain would need a small change (or level 1 run by hand with r_level1.sh). Decide after the smoke.
4. Whether to run all per-level smokes (section 7.2) or only the IN L1 and the scratch L1 smokes before the long chains.
