# Retrain-from-scratch prerequisites: AdamW, tent low-res, per-stage cascade skip, delayed task start, pair cap, retrain scripts and sweep tooling (review pass 2)

This pass re-reviews the whole uncommitted diff after the fixes for the first review (1 MEDIUM, 4 NIT). The change adds what the level 4 -> 2 -> 1 retrain needs: an AdamW decision record with a resume guard, a shift-free `tent` low-res op on the TF augmentation path, Jay's per-task and per-stage cascade skip, a delayed task start with an LR ramp, an opt-in training pair cap, three `r_level{4,2,1}.sh` scripts, a capped memory re-measurement job, a login-node sweep helper and a run summarizer. Every new flag defaults to the old ops. The goldens hold at max |diff| 0, `tests/fixtures/` is untouched, `TestSkip` (test_cascade.py:634) is unmodified, and the full suite passed: 647 tests, 0 failures, 4 skips, suite status 0. It ran 13:18-16:53, after the last trainer/model edit. The edits made after the suite are the review fixes: a docstring in `runtime/optimizers.py` plus the TF-free helpers and scripts. For those I re-ran py_compile, ast (3, 6) on both helpers, `bash -n` on the four scripts (LF only) and `tests.test_sweep_tools` + `tests.test_slurm_scripts`: 60 OK, TensorFlow not imported.

All five first-pass findings are fixed. The summarizer now excludes incomplete runs, and its length curve returns UNDECIDED for them (two new tests). The AMP guard docstring is corrected. The capped linear measurements pass `--min-pairs n`. The smoke line sets `DEFORM_RAMP=100`. The sweep help documents the job-index rule. No new HIGH or MEDIUM issue.

Watch for: two hand-off details. The summarizer's suggested 20000-step command hard-codes `--time=30:00:00` instead of the smoke step-time rule (NIT). The scripts' TODO comments name memory records "L4 linear" / "L4 deform knobs", but the capped job labels its steps "... capped", and the records themselves carry no label (NIT).

**Verdict**: APPROVED

## High-level view

Defaults are bit-identical, and the code shows how. Each new path sits behind a trace-time Python switch: `max_pairs > 0`, `method == "tent"`, `_skip_mode == "stage"`, or a task start > 0. With the defaults no op is built and no result key is added. `TaskSchedule` uses the old cumulative table and the same `random.Random(seed * 1000003 + step)` draw whenever all tasks are active. A hard-coded replica of the old loop reproduces 10000 draws for starts None, {} and all-zero.

The skip semantics match Jay's decision. A skipped level 4 is the identity (A_prev = I, U = U_inv = 0), and level 2 still runs on the identity-pre-warped x2. Skipping level 2 gives level 4 alone, and skipping both gives the identity. The draws are stateless and seeded by (task, level, seed, step), separately from the legacy draw. In stage mode the step input is `bool[n_stages]`, and a task with p = 0 at a level builds no op. Validation and BN recalibration call the cascade without skip inputs. Tests check stage vs legacy level-2 skip (max |diff| 0), `skip4=None` vs `linear_stages` / `deform_stages` + `skip_to` (exact), skip-both = identity, draw rates and joint rates, and the absence of skip ops in validation graphs.

The AdamW, tent and pair-cap pieces are small and well tested. AdamW at wd 0 equals Adam (variables and slots max |diff| 0). Decay hits kernels only, works under the LSO, and survives a checkpoint round trip. The resume guard refuses an optimizer change and logs an event when lr or wd changes. The tent operator is the static `P = U @ D` per axis. Its rows and columns sum to 1, f = 1 gives P = I, and a numpy loop reference matches it to 1.8e-7. The centroid gate passes on the specified grid (worst interior shift 0.0167 voxel). The pair cap is applied after the min-pairs floor in all four mirrors: postprocess, static count, memory model and checker.

The scripts and helpers keep to the cluster rules. They hard-code qtim, cap `--max-active` at 2 and count every squeue line of the user. `submit` refuses a dirty tree. A failed run, including "died before start", is never resubmitted. Every dry-run main.py line in the tests goes through `runtime.args.parse_args`, and every checker flag the scripts emit exists in `check_mixed_run.py`. The two remaining findings are cosmetic hand-off text.

<details>
<summary>Issues (2)</summary>

1. **NIT: hard-coded wall time in the 20000-step suggestion** (scripts/summarize_runs.py:331). Confirmed. The "LENGTH 20000 (new run; command: ...)" line always prints `--time=30:00:00`, but the binding F5 rule sets `--time` from the smoke's INFO step_time projection x 1.3. Fix: print `--time=<from INFO step_time x 1.3>` as a placeholder, or scale the phase-C `--c-time` by 20000/16000 and say so in the line.
2. **NIT: TODO record names don't match the capped job's labels** (slurm/r_level4.sh:111, :113; slurm/r_level2.sh:95, :97 vs slurm/m10_batch_memory_capped.sh:78-82). Confirmed. The TODOs say "from record "L4 linear" / "L4 deform knobs" / "L2 linear" / "L2 deform knobs"". The capped job's step labels are "... capped", and they appear only in the job log, because measure_level_memory records have no label field. The knob set is the real selector. Fix: reword the TODOs to "the memory_batch_capped.json records with level/task and knob set ... (job step "L4 linear capped")".

</details>

<details>
<summary>Details</summary>

### Hand-off text for the length decision and the memory fill-in

The 20000-step suggestion fires only when the 16000-step run beats 12000 by more than the tolerance on both metrics. That is the case where the run is longest, and also where the wall time matters most. At the current placeholders (N 64 / DEFORM_N 32, ~2 s mean step from the uncapped m10 tool numbers plus the trainer overhead) 30 h is generous. With other sizes it could also be short. The line reads as a ready command, so it should either carry a placeholder or follow the documented rule.

The memory fill-in rule in the r_level4 / r_level2 headers already says to pick "the record whose knob set equals this script's defaults". So the record names in the TODO comments are labels a reader has to map to step labels in the job's stdout. After the first-pass fix, the capped linear steps pass `--min-pairs n`, so the knob sets match literally. Only the names differ.

### Test coverage

Not tested (unchanged from the first pass; recorded in the report as unverified):

- the level-1 stage-mode deform step run end to end: traced only, because of local memory. The r_level1 header flags the extra ~1.3 GB of transient fields from `deform_skip` with skip2 and gates the long run on the smoke `INFO gpu_peak`;
- `--int32-check` on the level-2 cascade configurations of the capped job;
- AMP from scratch at level 4 on real data, and the real sbatch/squeue behaviour of the sweep helper (stubs only).

</details>

<details>
<summary>File map</summary>

- `runtime/optimizers.py`: weight-decay / AMP-guard docstring, `WEIGHT_DECAY_OPTIMIZERS`.
- `runtime/args.py`: `--lowres-method`, `--cascade-skip-linear/-deform`, `--task-start-step/-ramp-steps`, `--max-pairs`, `--weight-decay` >= 0, help text.
- `runtime/run.py`: `cascade_stage_skip_draw`, `TaskSchedule(start_steps)`, `TrainConfig` fields, Trainer skip modes and `[n_stages]` signature, LR ramp in `fit`, pair-cap plumbing.
- `models/cascade.py`: `FrozenChain.linear_skip` / `deform_skip`.
- `data_loading/{params,tf_intensity_aug,intensity_ref,postprocess}.py`: `low_res_method`, tent operator and numpy reference, `max_pairs` in `num_pairs` / `sample_pairs` / post-processes.
- `models/memory_model.py`: `num_pairs(max_pairs)`.
- `main.py`: optimizer line, record and resume guard, skip record / refresh / consistency warning, pair NOTE lines, tent description.
- `tests/gpu/check_mixed_run.py`: stage-mode skip check, capped `expected_pairs`.
- `tests/gpu/measure_level_memory.py`: `--max-pairs` grid axis, `--int32-check`.
- `slurm/r_level{4,2,1}.sh`, `slurm/m10_batch_memory_capped.sh`, `slurm/sweep_level4.py`, `scripts/summarize_runs.py`: new.
- `tests/*`: new tests per design section 11 and the review fixes; `tests/tf_setup.py` + `tests/run_tests_wsl.sh`: per-module processes, deferred memory tests.
- `.agents/tasks/network-design/design.md`: v3.16.

Full diff: `git -C d:\Research\Brain_Registration_Projects\brain-group-rad diff` plus the untracked files listed by `git status`.

</details>
