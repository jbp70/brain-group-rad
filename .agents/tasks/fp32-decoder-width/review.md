# float32_fine deform decoder, width docs, NIT fixes and the m11 fp32/width memory job

This change adds `--deform-decoder-dtype float32_fine`, which runs only the decoder stages d2 and d1, their upsamples and `svf_norm_act` in float32, while d5..d3 and the encoder stay float16. It also fixes nonfinite-fixes review NIT 1 in code (the LSO-skip flag now follows the hard-failure streak) and documents NIT 2. Following Jay's override, there is no `--cascade-trainable-init`: the per-block widths go through the existing `--filters` / `--decoder-widths` flags, now documented. Non-default widths get random-weight level-4 and level-2 cascade sources, built in-job by `slurm/m11_random_sources.sh`. The new SLURM job `slurm/m11_fp32_width_memory.sh` runs the 20-config grid through the real trainer, one process per config, and a stdlib collector writes the JSON and a table. Nothing is committed (HEAD ddb91f8) and nothing is submitted.

Watch for: source builds have no time guard of their own, so a bad source phase could eat up to 2 h of the 4 h budget (NIT). The time estimate (~3.0 h) is unmeasured. Plan item 8 (report.md, deleting the `.agents/tmp/fp32w_*` helpers) is still open.

**Verdict**: APPROVED

## High-level view

`float32_fine` builds the chosen stages under a float32 policy scope inside `DeformDecoder`. `call` casts x to each block's compute dtype before its upsample and before `svf_norm_act`. A cast to the same dtype is a no-op, so policy, float16 and float32 graphs are unchanged, and the verification log shows both goldens at max |diff| 0. Variables are float32 with identical names, shapes and order in all four modes. Tests cover checkpoint read/write and `copy_weights_from` across every ordered pair of modes, `warm_start_model` into float32 and float32_fine models, and a cascade copy into float32 / float32_fine trainable stages at level 2 and through a level-1 FrozenChain. That closes NIT 3. The cascade docs state that frozen stages are always float32.

The NIT 1 fix replaces `consecutive_task[task]` with `save_streak[task]` in the LSO flag. `save_streak` only counts forward NaNs and is reset by a finite step of the same task, so gradient-only overflows can no longer arm the skip. A new trainer test covers "gradient-only overflow, then forward NaN". NIT 2 (guard off defers ckpt/last for good) is documented in the `--nonfinite-patience-task` help and the run.py docstring.

Widths need no new flags. The grid lines (d12h, d12hs, fr50) match the existing `fr50` / `f0d1` stem conventions. Each random source is a one-step rigid main.py run with `--deform-decoder true`. Level-2 grid lines cascade from `l4_W/ckpt/last` and level-1 lines from `l2_W/ckpt/last` through the normal copy path, so the frozen stages are narrow too. A local three-level test (`test_random_width_sources_chain`) checks this.

The m11 header matches the requested SBATCH lines exactly. Each config runs in its own `timeout -k 60` singularity process with rc captured and a status row written, and its ckpt dir is deleted afterwards. A width whose source failed is recorded as rc -2. The dry run prints 6 SOURCE lines, 20 CONFIG lines in the requested order and a COLLECT line, and every argv parses with `runtime.args`. I re-ran the py3.8 gate: 86 TF-free tests OK, bash -n OK, `main.py --help` OK without TF.

<details>
<summary>Issues (4)</summary>

1. **No time guard on the source phase** (NIT, confirmed): `m11_build_sources` runs up to 6 x `SOURCE_TIMEOUT` 1200 s before the grid starts, with no check against `TIME_BUDGET_S`. Realistically this phase takes ~20-25 min, but a stalling source could cost up to 2 h of grid time. Optionally lower SOURCE_TIMEOUT to ~600 s or apply the budget check there too.
2. **Unmeasured time estimate** (NIT, likely acceptable): the header's ~3.0 h of 4 h has no measurements behind it, and float32 L1 lines may take longer per step than the fp16 23 s. The `skipped_time` guard and the ONLY/SOURCES_DIR split rule cover overruns. Check the per-config times in status.tsv after the first job.
3. **Plan item 8 outstanding** (NIT, confirmed): `.agents/tasks/fp32-decoder-width/report.md` does not exist yet, and the `.agents/tmp/fp32w_*` helpers are still present (verification.md keeps them on purpose for review iterations). Finish both before handing over.
4. **Unrelated test-tolerance change** (NIT, confirmed): `tests/test_group_net.py` raises the noise floor of `test_recompute_matches` from 1e-6 to 1e-5 of g_max to fix the known GPU-nondeterminism flake. The change is justified, but it is not part of this task. Mention it in the commit message or report.

</details>

<details><summary>Details</summary>

### Per-stage dtype in DeformDecoder

The scope is applied per stage name in `__init__`. `svf_norm_act` is float32 whenever any stage is. `svf_head` was already float32. The only new graph ops are the `tf.cast` calls in `call`, and they return their input when the dtypes match, which the bit-identical goldens and `test_noop_under_float32_policy` confirm. Under float32_fine, `dec_feat` (the d1 output) is now float32. The diagnostics cast it to float32 anyway, so the only effect is memory: with full checkpointing, a float32 d1 output at level 1 ([4,160,224,192,16]) is about 0.9 GB more than fp16. The m11 job measures exactly this. `test_dtypes_per_submodule` checks every Keras sublayer's compute dtype for all four modes, at level 2 and on a level-1 TINY model (where `up_d1` exists). It also checks that the global policy is restored afterwards.

### LSO flag after NIT 1

```
lso_flag = amp and save_streak[task] >= LSO_SKIP_STREAK - 1
```

`save_streak` increments only when the loss is non-finite (or without AMP) and resets on any finite step of the task. A gradient-only overflow followed by a forward NaN now reaches the LSO. A second consecutive forward NaN is still skipped. For every sequence in the existing tests the old and new flags agree, and those tests are unchanged and passing.

### m11 job and random sources

With `set -uo pipefail` and no `-e`, a failing config cannot stop the loop. The grid loop's guard (`elapsed + 300 > TIME_BUDGET_S` -> rc -1) together with `t = min(CONFIG_TIMEOUT, remaining)` keeps the grid within 13800 s + 60 s kill + collector, inside the 14400 s limit. The source phase before it has no such check (Issue 1). Default-width L1/L2 lines use the real m8/m7 `ckpt/best`. Their `--global-transform false` does not conflict with the G-trained sources, because G is not a ModelConfig field (as verification.md notes). L2 and L4 lines use memory growth. L1 lines use the 40000 MB limit with lean losses and pair recompute, matching the last long run, so the L1 f16 rows are comparable to the 36.7 GB first-window peak of job 10518952. In the dry run, the selected sources are validated read-only and nothing is created. I checked this locally: only the pre-made ckpt dir existed afterwards.

### Collector

The collector reads `<runs>/<name>/logs/metrics.jsonl` and `config.json` and skips malformed lines. Status is assigned in this order: skipped_time, skipped_source, timeout, oom, ok, abort. OOM text is only checked when rc != 0, which avoids misclassifying the benign BFC "ran out of memory ... not a failure" warning. Deform/linear step times exclude steps <= 2 (trace and autotune). Because every grid line is deform-only, `mean_linear_step_s` is always null here. Tests cover each status, step skipping, a missing run dir and a missing status file (exit 2).

</details>

<details>
<summary>File map</summary>

- models/group_net.py: `float32_fine` mode, `FLOAT32_FINE_STAGES`, per-stage scopes and casts, docstrings.
- models/cascade.py: Float32 row says frozen stages ignore the trainable dtype flags.
- models/width_presets.py: docstring paragraph with the m11 grid width flag lines.
- runtime/args.py: float32_fine choice and validation, per-block width docs, NIT 2 help.
- runtime/run.py: NIT 1 `save_streak` LSO flag, docstrings, NIT 2 docs.
- slurm/r_level{4,2,1}.sh, slurm/m9_level1_cascade.sh: accept float32_fine; linear-only drop.
- slurm/m11_fp32_width_memory.sh (new): 20-config real-trainer memory grid.
- slurm/m11_random_sources.sh (new): random-weight l4_W / l2_W sources, sourced by the job.
- scripts/collect_memory_runs.py (new): status.tsv + run dirs -> JSON + table.
- tests/test_fp16_mitigations.py, test_cascade.py, test_trainer.py, test_main.py, test_runtime_infra.py, test_slurm_scripts.py, test_sweep_tools.py, test_collect_memory_runs.py (new): coverage above.
- tests/test_group_net.py: tolerance floor of the recompute test (flake).
- tests/run_tests_wsl.sh: new script/module in PYFILES, bash -n and TF-free lists.
- .agents/tasks/nonfinite-fixes/report.md: NIT 1 note.

Full diff: `git diff` plus untracked files from `git status` against ddb91f8.

</details>
