# fp32-decoder-width: verification

## Iteration 1 (integration, no review.md yet)

Working tree on main at base ddb91f8 with FEAT-001..003 uncommitted. Nothing committed, pushed,
submitted, no ssh.

### Cross-FEAT seam check (read-only)

- m11 grid argv (slurm/m11_fp32_width_memory.sh) vs runtime/args.py: every flag exists; all 20
  CONFIG argv and the 6 SOURCE argv parse with `runtime.args.parse_args` (tests
  `TestM11Job.test_m11_dry_run_argv_parses`, `TestM11Sources`), non-default lines have the same
  ModelConfig (seed aside) as their random source, default lines the default config.
- No `--cascade-trainable-init` anywhere (orchestrator override; static test asserts it).
- `--deform-decoder-dtype float32_fine` (FEAT-001) is what the m11 f32fine lines pass; needs
  `--amp true` + deform in the ratio, both present.
- Collector (scripts/collect_memory_runs.py) vs trainer output: run dir `<results>/<run_name>/`
  (runtime/logging.make_run_dir), `logs/metrics.jsonl` records `{step, time, data}` with
  `time/data_ms`, `time/compute_ms`, `gpu_peak_mb`, `nonfinite_total`, `train_deform/n_pairs`
  (runtime/run.py ~2051-2122); config.json `args.filters` / `args.decoder_widths`,
  `param_counts.trainable` / `param_counts.deform_decoder` (GroupRegLevelModel.count_params_by_part).
  status.tsv columns written by the job's `row()` = collector `COLUMNS`; rc -1 / -2 map to
  skipped_time / skipped_source.
- Source layout: m11_random_sources.sh `m11_source_ok` checks `<dir>/config.json` +
  `<dir>/ckpt/last/checkpoint`; grid lines cascade from `<dir>/ckpt/last` (accepted by
  `C.load_cascade_source`, covered by `tests.test_main.TestCascadeMain.test_random_width_sources_chain`).
- ModelConfig has no task / global-transform / dtype fields, so deform-only grid lines with
  `--global-transform false` cascade from the real m7 / m8 sources (G on) without a config mismatch.

No seam defects found.

### Commands and results

1. py3.8 gate:
   `wsl -d Ubuntu bash -c "tr -d '\r' < .../.agents/tmp/fp32w_py38.sh > /tmp/fp32w_py38.sh; bash /tmp/fp32w_py38.sh"`
   -> py_compile OK, `ast feature_version=(3, 8) OK (89 files)`, bash -n OK for all slurm/*.sh
   (incl. m11_fp32_width_memory.sh, m11_random_sources.sh), `main.py --help OK (no TF)`,
   `TF-free tests OK without TF (86 tests)`, status 0. Rerun after the test fix below: same result.
2. Full suite (background, log .agents/tmp/fp32w_full.log):
   `wsl -d Ubuntu bash -c "bash /mnt/d/.../tests/run_tests_wsl.sh > /mnt/d/.../.agents/tmp/fp32w_full.log 2>&1"`
   - Part 1 (bgr_preproc38): all OK, `TF-free tests OK without TF (136 tests)`.
   - Part 2: `unittest summary: 709 tests (4 skipped ...), 1 failing invocation(s): tests.test_group_net`,
     wall time 14055 s (3 h 54 min).
   - Goldens: `[golden deform] max |diff| 0 (tolerance 0)` and `[golden] max |diff| 0 (tolerance 0)`;
     tests.test_trainer 68 tests OK.
   - Per module (FEAT-relevant): test_cascade 46 OK, test_fp16_mitigations 17 OK, test_main 41 OK,
     test_runtime_infra 35 OK (1 skip dllogger), test_slurm_scripts 35 OK, test_sweep_tools 44 OK,
     test_collect_memory_runs 7 OK, test_width_presets_tf 2 OK; both deferred memory tests OK.
3. The one failure: `tests.test_group_net.TestGradientCheckpointing.test_recompute_matches`
   (`2.48e-09 not less than or equal to 1.10e-09 : integrator .../encoder/stem/conv/bias:0`).
   This is the known pre-existing GPU-nondeterminism flake (milestone-9 report: HEAD 8f850cd failed
   1 of 8 isolated runs; milestone-8 FEAT-001): the stem conv bias sits before a BN, its gradient is
   analytically ~0 and carries only rounding noise. The fp32-decoder-width changes do not touch
   this graph (float32 policy, `float32_stages=()`; `tf.cast` to the same dtype returns its input).
   Fix (test only, tests/test_group_net.py): noise floor of the per-variable tolerance
   1e-6 -> 1e-5 of the largest gradient (1.1e-8 here; a real recompute bug gives O(1) relative
   differences), with a comment.
   Rerun: `bash /tmp/fp32w_tests.sh tests.test_group_net tests.test_group_net.TestGradientCheckpointing x4`
   -> test_group_net 27 OK, TestGradientCheckpointing 4/4 OK (log .agents/tmp/fp32w_gn.log).

### Files changed in this iteration

- tests/test_group_net.py (tolerance floor of test_recompute_matches, comment).

### Notes

- Helpers `.agents/tmp/fp32w_tests.sh`, `fp32w_py38.sh`, `fp32w_poll.sh` kept for the review-fix
  iterations; plan item 8 deletes them at the end.
- A `nohup ... &` started inside a one-shot `wsl -d Ubuntu bash -c` dies when the call returns;
  start the full suite with the terminal tool's background mode instead.
