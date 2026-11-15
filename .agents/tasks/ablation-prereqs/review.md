# Group-block placement variants, instance-norm path and from-scratch level-1 launcher for the paper ablations

The change adds `--group-config none` (no group blocks) and `E` (stem/encoder only) next to A and B. Two `ModelConfig` methods, `group_encoder()` and `group_decoders()`, replace every `group_config == "A"` check and every hard-coded `group=True` in the model build, the TF-free param/memory mirrors and the streaming engine. Instance norm needed no production change, only tests that pin it through AMP, the cascade, L1 float32_fine, streaming and predict. Scripts gain `NORM` in `r_cascade.sh` (with `group_args`/`norm` recorded in the chain state) and a new `slurm/r_level1_scratch.sh` for the full-resolution-from-scratch run. Recorded verification: full suite status 0, 1099 tests, goldens `[golden] 0`, `[golden deform] 0`, `[golden tta-off] 70 hashes, 0 differ`.

Watch for: the GPU-hours step rule uses the steady-state ratio-weighted step and ignores the linear-only phase before DEFORM_START, so the scratch run will come in under the 63.85 h budget (likely). The IN cascade still passes `--bn-recal-*` flags with 0 batches at L2/L1 (confirmed; a no-op, but the brief asked for no recal flags). Nothing at L1 (IN, or trainable BN from scratch) has executed a real deform step; the smokes are the first runs.

**Verdict**: APPROVED

## High-level view

The group mapping follows linear-heads.md exactly: A everything, B both decoders (linear reductions + dense_1, deform D5-D1), E stem + L2-L6 only, none nothing. Because the selectors are methods and A/B map to the same booleans as before, config.json is unchanged, A/B graphs are identical and the goldens are 0 diff. `dense_group_concat_first` with none/E is rejected in `ModelConfig.validate`. Train-mode BN coupling stays and is documented.

Mixed configs across cascade levels are rejected by the existing `check_cascade_config` / `FrozenChain` equality checks. `group_config` and `norm` are ModelConfig fields, and new tests pin the rejection for the new values. No new code was needed for this.

The streaming engine now skips group sums for ungrouped decoder stages and drops the decoder "group concat" item from the element bound. For `none` the streamed result equals the model exactly. The linear decoder still runs once on the gathered bottleneck, so it needed no change.

IN was already safe under AMP: moments and normalisation run in float32. IN models have no BN layers, so freeze re-apply and the checkpointing BN snapshot are no-ops, and the parser rejects `--bn-recal-batches > 0` with `--norm instance`. `r_cascade.sh NORM=instance` adds `--norm instance` at every level and sets `BN_RECAL_BATCHES=0` at L2/L1. The level scripts still emit `--bn-recal-batches 0 --bn-recal-freeze ...`.

`r_level1_scratch.sh` reproduces the final L1 recipe without a cascade source. It uses the L4 phase-C schedule as fractions of a required STEPS, trainable BN, curriculum and delayed deform/G, 48 h / 128G, and resubmission with the same RUN_NAME. The GPU-hours budget (63.85 h from sacct) and the step rule are in the script header and the report.

<details>
<summary>Issues (4)</summary>

1. **Step rule under-fills the GPU-hours budget**: the rule divides 229 860 s by the checker's ratio-weighted (all tasks active) mean step, but the first 3/16 of the run is linear-only (~2.3 s/step vs ~11 s). STEPS will come out roughly 10-15 % low. Fold the two phases into the formula, e.g. S = (B - val) / (3/16·t_lin + 13/16·t_mix), or state the bias in the report.
2. **IN chain still passes recal flags**: with `NORM=instance`, r_level2/1.sh still emit `--bn-recal-batches 0 --bn-recal-freeze true|false` (and `--bn-recal-validate`). This is a tested no-op, but the brief said scripts must not pass recal flags. Either drop the flags when BN_RECAL_BATCHES=0, or record the deviation in the report.
3. **Scratch-L1 memory claim not established**: the report says from-scratch L1 needs "at most the final L1's" memory. The final L1 had frozen BN, while this run trains BN at full resolution with 4 pairs, a combination that never ran. Soften the claim and list `gpu_peak <= ~41500 MB` explicitly for the scratch smoke, as is done for the IN smoke.
4. **Old chain state assumed '--group-config B'**: a pre-change chain started with a non-default GROUP_ARGS will now hit exit 2 on resume and need FORCE_RECIPE_CHANGE=1. No such chain exists (chains were never submitted), so this only needs a line in the report.

</details>

<details><summary>Details</summary>

### GPU-hours matching for ablation A1

The budget is read correctly from sacct: 13:54:14 + 27:20:44 + 22:36:02 = 229 860 s. The header and report give `STEPS = floor(229860 / (mean_step_s + v / VAL_EVERY))` with `mean_step_s` from the checker's `INFO step_time`. `check_mixed_run._step_time` computes a ratio-weighted mean of per-task means, which is the step cost once all three tasks run. In this run deform starts at 3·STEPS/16, so the first 18.75 % of steps are rigid/affine only. With L1 numbers (linear ~2.3 s, deform ~20 s, 1:1:2 → ~11 s mixed), the true mean over the run is ~9.4 s. Using 11 s therefore under-fills the budget by ~15 % (**likely**; the exact gap depends on the measured deform time). For a matched-compute claim this biases against the scratch baseline. The fix is a two-phase formula in the report and header; no code change is needed.

### Instance norm through the level scripts

The `r_cascade.sh` change is minimal and ordered well (`--norm instance` after `GROUP_ARGS --lambda-smooth`, before `EXTRA_ARGS_L<k>`), and the checkers drop `--expect-bn-recal` when batches are 0. r_level1.sh and r_level2.sh still put `--bn-recal-batches 0 --bn-recal-freeze <x>` on the command line, and r_level1.sh also `--bn-recal-validate false` (**confirmed**, r_level1.sh line ~371, r_level2.sh ~374). FEAT-002 shows the parse succeeds and the freeze count is 0, so behaviour is correct. It contradicts only the brief's wording. The new scratch script emits no recal flags.

### Memory notes for the first real L1 runs

The local GTX 1080 traced but never executed an L1 deform train step for IN or for the scratch Trainer. The report states this. For IN, the float32 temporaries are unmodelled and the smoke check is spelled out. For the scratch run, the report reasons that no frozen stages means less memory. That saving is small (frozen L4/L2 forwards at coarse grids plus fine-grid integration, ~0.7 GB measured). It is offset by train-mode BN in the trainable stage, which the final L1 never had (freeze true), so the claim is **possible**, not established. The smoke reads gpu_peak anyway; the claim in the memory section should match.

### Chain-state compatibility

`check_rec` treats a missing `group_args` key as `--group-config B` and a missing `norm` as batch. That is right for chains launched with the old default. A chain started with an explicit different GROUP_ARGS before this change would fail on resume with exit 2. This fails closed, and no such chain exists (project state: chains not submitted), so it only needs documenting.

### Test coverage

Each new variant has build, gradient-flow, eval-mode independence (none: A_raw / v_raw unchanged under group replacement), param-count and split-count tests, a tiny L2 cascade Trainer step, mixed-config rejection, a main.py L4→L2 run, and predict (`default`, `--levels 2`, `flip8 + --recenter 1`, `check-streaming`). IN has the same coverage plus streaming at L4/L2 in both precisions. Not tested: an executed L1 deform train step for IN, none/E or the scratch Trainer (local GPU memory). The `r_level1_scratch.sh` recipe with curriculum/G is covered only at parse level, because those options need DALI. The permutation-equivariance tolerance floor (1e-4 mm) now applies to A/B as well. It is test-only and well below any practical threshold.

</details>

<details>
<summary>File map</summary>

- `models/config.py`: GROUP_CONFIGS + none/E, `group_encoder()` / `group_decoders()`, dense_group_concat_first validation.
- `models/group_net.py`: encoder, linear decoder, deform decoder group flags from the methods; docstring table.
- `models/width_presets.py`, `models/memory_model.py`: same flags in the TF-free mirrors; norm-agnostic note.
- `inference/streaming.py`: skip group sums for ungrouped decoder stages; element bound.
- `runtime/args.py`: `--group-config` choices from GROUP_CONFIGS, help text.
- `slurm/r_cascade.sh`: NORM env, group_args/norm in chain state and resume check, ablation usage lines.
- `slurm/r_level1_scratch.sh`: new from-scratch level-1 launcher.
- `tests/test_ablation_configs.py` (new) and updates to test_group_net, test_memory_model, test_predict_streaming, test_runtime_infra, test_slurm_scripts, test_split_group_conv, test_width_presets(_tf), run_tests_wsl.sh.
- `.agents/tasks/ablation-prereqs/{plan,linear-heads,report}.md`: plan, linear-heads finding, report with commands.

Full diff: `git -C .worktrees/ablation-prereqs diff 8e12596..HEAD`.
</details>
