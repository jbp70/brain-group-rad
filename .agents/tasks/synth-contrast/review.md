# Review: synth-contrast design v4

Reviewer step, workflow wf_54f1dd7ba6107c74. This is the fourth review. The review v3 findings (0 HIGH /
1 MEDIUM / 7 NIT) are re-checked first, then the v4 changes: exact smoke command lines, the new checker
option `--min-task-steps` / env `MIN_TASK_STEPS`, the manifest pre-check, the `%2` array caps, the
synthetic set naming, and Jay's answer 13.1b (no L1 reference model, smoothness passed as parameters).

Code read for this round (on top of the v3 list, which still holds; nothing relevant changed in the
repo since v3):
- `runtime/run.py`: `TaskSchedule` (421; stateless seeded draw, delayed start = never drawn before the
  start step), the delayed-task LR ramp in `fit` (2009-2014; `DEFORM_RAMP` is an LR ramp, not a draw
  ramp), `_nonfinite_window_metrics` (1876) and the `win_steps` counter (1994-2128) that logs
  `train_<task>/window_steps` per log window.
- `runtime/args.py`: `--task-start-step`, `--task-start-ramp-steps`, `parse_task_start_steps` (149).
- `tests/gpu/check_mixed_run.py`: `check()` (614 onward), `trace_once` (678-681), the per-task
  non-finite window check.
- `slurm/r_level{4,2,1}.sh`: env names `DEFORM_START`, `DEFORM_RAMP`, `CURRICULUM_RAMP`, `G_START`,
  `G_RAMP`, `PROJECT_STEPS`, `PROJECT_VAL_EVERY`, `LR`, `WD`, `EXTRA_ARGS` (appended last after the
  hard-coded `DEFORM_LOSSES ... --lambda-smooth 0.5`, so `--lambda-smooth $S` in `EXTRA_ARGS` wins, as in
  Jay's config B runs).
- `preprocessing/export.py` `ALL_LEVELS = (1, 2, 4)`; `preprocessing/slurm/export_array.sh` header
  `#SBATCH --array=0-0%2`; `preprocessing/tests/test_slurm_scripts.py` (array lines must end in `%2`,
  basic-partition memory <= 15 GB per CPU; the new 1-CPU / 8G wrapper satisfies both);
  `preprocessing/cli.py` `select_rows` (clamps `--end` to the row count).

Verdict: **APPROVED** (0 HIGH, 0 MEDIUM, 2 NIT).

## Review v3 findings

| # | Severity | Topic | Status |
|---|---|---|---|
| 1 | MEDIUM | L4 smoke under the section 10 recipe runs no deform step | Closed. 7.2 gives exact command lines for all three pairs. The L4 pair sets `DEFORM_START=100 DEFORM_RAMP=100 CURRICULUM_RAMP=100 G_START=100 G_RAMP=100`, which are all real `r_level4.sh` env names. `TaskSchedule` never draws deform before step 100 and draws it at its full 2/4 share afterwards (the ramp scales only the LR, run.py 2013-2014), so ~100 deform steps run in 300 steps. The draw is seeded and deterministic, so both runs of a pair get the same task sequence and the `deform:80` threshold is not flaky. `--min-task-steps` (5.6) turns a task that never ran into a FAIL. L2/L1 thresholds 20/20/40 against ~50/50/100 expected are fine. |
| 2 | NIT | five vs six label columns | Closed (3.3 step 6 defines `LABEL_COLUMNS` from `export.ALL_LEVELS`, which exists). |
| 3 | NIT | array concurrency cap | Closed (`--array=0-..%2`, header `0-0%2` as in `export_array.sh`; the preprocessing slurm test requires `%2`). |
| 4 | NIT | synthetic set names vs Trainer caches | Closed (8 step 4: `synth_<preset>_<real set name>`, test in `tests/test_eval_synth.py`). |
| 5 | NIT | preset count | Closed (11 presets, time recomputed). |
| 6 | NIT | `$R=$RESULTS` | Closed (`R=/autofs/.../results` at the top of both blocks). |
| 7 | NIT | pre-existing parts/manifest divergence | Closed (manifest step 0 read-only pre-check, error table row, test with a hand-edited `npy_dir`). |
| 8 | NIT | config B never run at L2/L1 | Closed (6.2 and 7.2: a failing p 0 smoke stops the chain and is reported as a config B / recipe issue). |

## Hard requirements: coverage

| Requirement | Covered? | Evidence checked |
|---|---|---|
| H1 same model / group sizes / RTX 8000 limits | Yes | No model or group-size change. Smoke and long commands use the script defaults (128/64, 32/16, 8/4). L1 commands pass `TF_MEM_LIMIT_MB=42500` and `--mem=128G`. Config B is passed via `EXTRA_ARGS` exactly as in Jay's `r4_cfgB_*` runs. The p 0 / p 0.5 pairing makes the memory gate independent of config B's unmeasured L2/L1 peaks. |
| H2 no new in-training validation | Yes | Validation sets and passes unchanged; `eval_synth.py` is a separate process; the `group_images_fn` hook defaults to today's `images[idx]`. |
| H3 memory-neutral, one graph per task, quantified, fallback | Yes | Per-level table in 6.1 (re-checked in v3, unchanged); DALI and host RAM in 6.2/6.3; `tf.cond` on a drawn tensor keeps `trace_counts` at 1 and the checker's `trace_once` applies to every task in `tasks`; fallbacks 6.2 (1)-(3) ordered by where the pressure shows. |
| H4 default off is bit-identical | Yes | Unchanged from v3: `_pp_call` off path is `pp(*inputs)`, no generator / checkpoint key / DALI reader when off, `--synth-*` flags and `MIN_TASK_STEPS` only appended when set (asserted in `test_slurm_scripts.py`). |
| No one-hot of fine labels | Yes | Two gathers of a [256] table; graph-def OneHot depth-multiset test with negative control. |
| Exporter label downsampling | Yes | Nested majority vote reproduces `block_mode(train18_full, k)` exactly and is verified voxel for voxel against the existing train18 files before any write; no interpolation. |
| Py 3.8 / TF 2.10 / DALI 1.18 | Yes | No new op beyond those verified in v3; the new smoke/checker pieces are bash and stdlib Python. |

## Findings

### 1. NIT: "executed training steps" in `--min-task-steps` and `n_task_steps` should name `window_steps`

Where: 5.6 (`--min-task-steps`: "the number of executed training steps of that task in `metrics.jsonl`
(records of `train_<task>/...`, the same count used as `n_task_steps` above)"; `--expect-synth-prob`:
"step-weighted mean").

Problem: `metrics.jsonl` has one record per log window (`--log-every 25`), not one per step. A coder who
counts records gets ~4 deform records in the L4 smoke instead of ~100 and fails `deform:80` every time;
the same miscount would also make the `--expect-synth-prob` bound 4 sqrt(p(1-p)/n) almost vacuous. The
trainer already logs the right quantity: `train_<task>/window_steps` (run.py `_nonfinite_window_metrics`,
fed by `win_steps[task] += 1` per executed step).

Fix: in 5.6 write "n_task_steps(t) = sum over records of `train_<t>/window_steps`; the synth_frac mean is
weighted by `window_steps`; `--min-task-steps` compares n_task_steps(t) with the threshold". Add a checker
test where a run has 4 records x 25 steps (passes `deform:80` at 100, fails at 101).

### 2. NIT: `--start/--end` range rule for `export-add-labels` is unstated

Where: 3.3 arguments and 3.4 (the last array task gets `--start 3000 --end 3500` for 3423 rows).

Problem: the design does not say whether an `--end` beyond the row count is clamped or an error, or what
`start >= rows` does. The existing CLI clamps (`preprocessing/cli.py select_rows`: `e = min(len(rows),
end)`), and the array command relies on clamping for the last chunk.

Fix: state "`--start/--end` are clamped like `cli.select_rows` (end > rows -> rows; an empty selection
writes a status file with 0 cases and exits 0); negative values or start > end exit 2", and keep the
status file name on the requested range. Add the last-chunk case to `test_add_labels.py`.

## Verified assumptions

- `r_level4.sh` exposes `DEFORM_START` (default 3000), `DEFORM_RAMP` (500, LR ramp via
  `--task-start-ramp-steps`), `CURRICULUM_RAMP`, `G_START`, `G_RAMP`; `r_level{2,1}.sh` expose
  `PROJECT_STEPS` / `PROJECT_VAL_EVERY` (validated as uints) and `LR` (default 3e-4).
- `EXTRA_ARGS` is the last element of the main command in all three scripts, after the hard-coded
  `--lambda-smooth 0.5`, so argparse takes the `EXTRA_ARGS` value.
- `TaskSchedule.draw` depends only on (seed, step, ratio, start steps); a delayed task is never drawn
  before its start and at its full share after it.
- `check_mixed_run.trace_once` requires `trace_counts[t] == 1` for every task in `tasks`, read from
  `fit_summary.json` / `overfit_summary.json`.
- `train_<task>/window_steps` is logged per log window for every task with a step in the window.
- `export.ALL_LEVELS = (1, 2, 4)`; `export_array.sh` header `#SBATCH --array=0-0%2`; the preprocessing
  slurm test checks `%2` and <= 15 GB per CPU.
- Bash `${P/./}` yields `0` and `05` for `P=0` and `P=0.5`, matching the run names in 7.2.
- All v3 verified assumptions (post-process signatures, `_recal_input`, `split_batch`, DALI readers,
  exporter `block_mode` / `merge_parts` / `export_check`, Trainer caches) still hold.

## Unverified / wrong assumptions

- Unverified: DALI seed streams unchanged when a seeded label reader is appended (GPU check s1 measures
  it; not a correctness issue).
- Unverified: config B GPU peaks at L2/L1 (measured by the p 0 smokes; the gate is relative to them).
- Unverified: the current `npy_v2` parts reproduce the current manifest byte for byte (now detected by
  manifest step 0 and reported as a pre-existing divergence).
- Unverified: CPU fallback time of 1-3 s per L1 linear synthetic step and the 6.5 overhead estimates
  (measured in the smokes).
- Ambiguous: the step count behind `--min-task-steps` / `n_task_steps` (finding 1).
