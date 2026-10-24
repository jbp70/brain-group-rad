# Implementation Plan: Milestone 9 Phase 1 (level-1 memory design)

Source of truth: `requirements.md` (R1–R6, acceptance criteria AC1–AC12, out-of-scope list) and the approved
`design-m9-phase1.md` v5 (sections cited as §x.y). This plan sequences the design. It does not re-decide it.
Code base: commit `8f850cd` plus the uncommitted `network-design/design.md` v3.13 edits.

## Hard constraints (every item)

- Every new knob is OFF by default. With the defaults, levels 4 and 2 must stay bit-identical. Both goldens must report
  `max |diff| 0`: `tests/test_trainer.py` `TestLinearOnlyUnchanged` (`tests/fixtures/m7_linear_golden.json`) and
  `TestDeformUnchanged` (`tests/fixtures/lncc_deform_golden.json`). Never rewrite the fixtures (no `BGR_WRITE_GOLDEN=1`).
- New paths are Python-level (trace-time) switches. With the defaults, no new op is created (§3.9).
- None of the new options is a `ModelConfig` field (§3). Old `config.json` files restore unchanged, and
  `runtime.checkpoint.check_cascade_config` accepts today's level-4 → level-2 pair unchanged.
- Cluster compatibility: Python 3.8.10 syntax, TF 2.10, DALI 1.18, numpy 1.21, scipy 1.4. Verify with `py_compile` and
  `ast.parse(..., feature_version=(3, 8))` in the WSL conda env `bgr_preproc38`, and with `bash -n` on every new or
  changed shell script. No PEP 585/604 annotations evaluated at runtime. No 3.9+ stdlib.
- The local GPU is a WSL GTX 1080. The TF memory limit is at most 6000 MB, never higher (`BGR_TF_GPU_LIMIT_MB=6000`,
  `--gpu-memory-limit-mb 6000`). Use `TF_CUDNN_USE_AUTOTUNE=0` and `TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512`, as
  `tests/run_tests_wsl.sh` does.
- TF32 stays off. `models.precision.configure_tf_precision()` is the first TF call in every GPU entry point. New TF tests
  import `tests.tf_setup` first.
- No commits, no push, no cluster access (no ssh, sbatch or git pull). Do not touch files outside the repo.
- Phase 1 does NOT implement the 3-level cascade, level-1 DALI training, level-1 validation changes or the retraining
  sequence (R5, design-only in §6). `--cascade-from` stays level-2-only in `runtime/args.py`.

## Shell mechanics (PowerShell 5.1 mangles quotes)

- Put every multi-line or quoted command in a script under
  `d:\Research\Brain_Registration_Projects\brain-group-rad\.agents\tmp\<name>.sh`. Run it with
  `wsl -d Ubuntu bash -c "tr -d '\r' < /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.agents/tmp/<name>.sh | bash -s"`.
  Delete the script afterwards.
- Inside the scripts: `source ~/miniconda3/etc/profile.d/conda.sh`, then `set +u` before `conda activate <env>`, then
  `cd /mnt/d/Research/Brain_Registration_Projects/brain-group-rad`.
- Targeted TF tests (env `tf_gpu`, with the exports of `run_tests_wsl.sh` stage 2):
  `python -m unittest tests.test_x -v`.
- The syntax check (env `bgr_preproc38`): `python -m py_compile <files>`, plus the `ast` loop of `run_tests_wsl.sh`
  stage 1.
- The full suite runs only in the background with a generous timeout (≥ 4 h; baseline 381 OK, 4 skips, ~3.5 h):
  `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/tests/run_tests_wsl.sh`.

## Design-review NITs carried forward (apply during implementation)

- N1. In `memory_model.predict`, `skip_grad_bound=False, fwd_transient_bound=False` are the defaults. Setting either to
  True selects its conservative variant. Word the docstring and the report that way (§4.1 has it inverted).
- N2. U4 is stale for the lean sampler. In the report and in the `_sample_linear_lean` comment, state that the lean
  sampler computes its indices in int64 on the GPU, that this transient is modelled explicitly, and why int64 was
  chosen: int32 elementwise kernels are host-memory on GPU, and int32 would also be correct
  (N·V1 ≤ 5.5·10⁷ < 2³¹). List "int32 indices pinned to the GPU" under the Fw remedies next to forward corner
  sequencing.
- N3. Name the `fwd_transient_bound` quantity an "order-independent estimate (one overlapping sampler forward)" in code,
  printouts and the report. It omits a concurrently running second sampler forward (~+1.5 GiB at `f0d1` N4). Keep the
  4.6 pin at 41.0 ± 0.2.
- N4. `recommendation.json` `level1` also carries `deform_peak_fwd_bound_gib`, `deform_peak_fwd_bound_margin_gib` and
  `bound_alt_knobs` (e.g. `{"lean_losses": true}`). `level1.a100` also carries `deform_n_no_margin_fwd_bound`.
- N5. `m9_memory.sh` step 2c comment: "fr50 + lean losses (tests the Fw cap: predicted N5 39.6 GiB, N6 47.3 GiB → OOM
  expected), N 5,6,7".

## Ordered items

### A. Model knobs (TF code; §3)

- [ ] 1. Width presets and positivity validation (R2.1, R2.6).
      Create TF-free `models/width_presets.py` with `PRESETS`, `uniform(m)`, `preset_args`, `preset_kwargs` and
      `count_params(model_config_dict)` per §3.1. Use the table values exactly. `default` and `f0d1` list explicit
      decoder widths. Unknown names raise a ValueError that lists the names. `count_params` returns per-part counts
      (encoder / linear_decoder / deform_decoder / total), because the memory model needs them per part (U16).
      `ModelConfig.validate()` gains: every entry of `filters`, `reductions`, `dense` is an int ≥ 1 (ValueError naming
      the field). `runtime/args.py` `parse_args` gets the matching `parser.error` checks for `--filters` (6 entries),
      `--reductions` and `--dense` (≥ 1 entries).
      Note: an explicit `default` preset (`decoder_widths=(128,96,64,32,16)`) is not dict-equal to `ModelConfig()`
      (`decoder_widths=None`). Presets are only for new runs; old configs are untouched. Say this in the module docstring.
      Files: `models/width_presets.py`, `models/config.py`, `runtime/args.py`, new `tests/test_width_presets.py`
      (no TF: table reproduced by `uniform`, the `uniform` rounding rule, unknown name, `preset_args` strings, ModelConfig
      positivity errors), new `tests/test_width_presets_tf.py` (`tests.tf_setup` first: `count_params` within 1 % of the
      built level-4 model's trainable count for every preset; `default` 29.63 M, `f0d1` 29.54 M).
      Verify: `python -m unittest tests.test_width_presets tests.test_width_presets_tf -v` in `tf_gpu` passes;
      `python -m unittest tests.test_width_presets` also passes in `bgr_preproc38` (no TF import).

- [ ] 2. Move `smooth_field` to `tests/field_helpers.py` (needed by items 4 and 8).
      Move it out of `tests/test_diffeo.py` unchanged. `test_diffeo.py` imports it from there.
      Files: `tests/field_helpers.py`, `tests/test_diffeo.py`.
      Verify: `python -m unittest tests.test_diffeo -v` passes.

- [ ] 3. `--gradient-checkpointing full` with BN updated exactly once (R2.2, §3.2).
      Create `models/checkpointing.py` `recompute(fn, bn_layers=())`. Call 1 is the forward. Every later call
      snapshots the BN moving statistics, runs `fn`, and restores them under a control dependency on the outputs. No
      snapshot or restore is built when `training` is a Python False/None, or when every layer has
      `stats_frozen=True`.
      Add `layers.batch_norm_layers(layer)` (the logic of `Trainer._bn_layers`, which now delegates to it; keep the
      staticmethod name for callers).
      `GRADIENT_CHECKPOINTING` gains `"full"`. `GroupRegLevelModel._call_deform` under `full` with
      `encoder.built and deform_decoder.built` runs three pieces: the recomputed encoder returning the 6 skips, the
      recomputed decoder, and two separate `tf.recompute_grad` integrator blocks for exp(v) and exp(−v). Otherwise it
      runs today's direct path. Linear tasks are unchanged under `full`.
      `runtime/args.py` choices gain `full`. `full` without `deform` in `--task-ratio` gives
      `parser.error("--gradient-checkpointing full needs 'deform' in --task-ratio")`.
      `none / integrator / decoder / both` are unchanged, including their double BN update.
      Update the docstrings of `GroupRegLevelModel` and `models/layers.py`.
      Files: `models/checkpointing.py`, `models/layers.py`, `models/group_net.py`, `runtime/run.py`, `runtime/args.py`,
      new `tests/test_checkpointing.py` with cases (a)–(f) of §8:
      - (a) `full` vs `none` gradients within 1e-4 relative, fp32 and AMP, level 4 narrow N 3;
      - (b) BN moving mean / variance after one `full` step equal the `none` step (≤ 1e-7 relative), and `both` still
        shows the double update;
      - (b') two `tape.gradient` calls (persistent tape) under `full` still give one update;
      - (c) frozen statistics unchanged;
      - (d) `recalibrate_bn` under `full` makes one update per layer and batch (follow the existing T7 pattern in
        `tests/test_cascade.py`);
      - (e) one trace per task over 6 steps of rigid:1,affine:1,deform:2;
      - (f) `build_model` under `full` creates the same variables (names and shapes) as under `none`.
      Verify: `python -m unittest tests.test_checkpointing -v` passes. `tests.test_group_net` and the goldens
      `tests.test_trainer.TestLinearOnlyUnchanged tests.test_trainer.TestDeformUnchanged` report `max |diff| 0`.

- [ ] 4. Lean trilinear sampler (R2.4, §3.3).
      In `models/layers.py`, `sample_voxel_coords(..., lean=False, lean_images_grad=True)`. `lean=False` executes
      today's code byte-for-byte (keep it as is). `lean=True` with linear interpolation calls `_sample_linear_lean`
      (`tf.custom_gradient`). Its forward uses today's float op order with int64 index math, so the values are bitwise
      identical. Only `images` and `coords` are captured. Its backward is the module-level function
      `_lean_linear_backward(images, coords, dy, fill_value, boundary, images_grad)`:
      - the `dy`-dependent zero added to `coords` only (module constant `_DY_DEP = True`; tests may switch it off);
      - `_axis_terms` once;
      - the 8 corners in forward order x→y→z, each corner's ops under `tf.control_dependencies` on the previous
        corner's accumulators (d_coords, plus d_flat only when `images_grad`);
      - the fill-value and border-mask corrections;
      - images gradient `None` when `images_grad=False`.
      `nearest` ignores `lean`. Specified fallback: if memory-test case C fails, switch the corner loop to
      `tf.while_loop(..., parallel_iterations=1)` per §3.3 and record that in the report.
      Plumbing (`lean=False` default everywhere): `diffeo.warp` (+ `lean_images_grad=True`), `warp_field`, `compose`,
      `integrate_svf`, `exp_and_inverse`, `pairwise_displacement`, `compose_cascade`.
      `GroupRegLevelModel(..., lean_sampler=False)` is a constructor attribute passed to the integrator in every
      checkpointing mode.
      Files: `models/layers.py`, `models/diffeo.py`, `models/group_net.py`, new `tests/test_lean_sampler.py`:
      - forward max |diff| 0 vs default, eagerly, for zero / border, fill 0 / 0.3, C 1/3/19, float32 / float64; once more
        inside `tf.function` under `_GrapplerOff` + `_OpDeterminism` (import them from `tests/test_cascade.py` /
        `tests/test_trainer.py`, or move them to a helper module without changing behaviour);
      - gradients w.r.t. coords and volume ≤ 1e-4 relative (float64 ≤ 1e-12);
      - `lean_images_grad=False`: images gradient `None`, d coords identical;
      - `exp_and_inverse(lean=True)` vs default;
      - `test_graph_mode_memory` cases A / B / C with the asserted bounds of §3.3 (A ≤ 2.25 GiB; B lean ≤ 1.24 GiB and
        default − lean ≥ 0.85 GiB; C1 ≤ 0.58 GiB, C2 ≤ 0.41 GiB). Every modelled value is computed in the test from
        the §2.1 formulas, and M and the peaks are printed. Case B without `_DY_DEP` is printed only. Skip without a GPU.
      Verify: `python -m unittest tests.test_lean_sampler -v` passes on the GTX 1080 (6000 MB cap) and prints the
      case A/B/C numbers (copy them into the report later). `tests.test_diffeo`, `tests.test_spatial_transformer` and
      both goldens pass with `max |diff| 0`.

- [ ] 5. Lean losses (R2.4, §3.4).
      In `runtime/losses.py`, add `_soft_dice_lean` (no `broadcast_to`, no materialised slices). `deform_loss(...,
      lean_sampler=False, lean_losses=False)`, `pair_dice_loss(..., lean_sampler=False, lean_losses=False)`,
      `template_ncc_loss(..., lean_sampler=False, lean=False)` and `pair_ncc_loss(..., lean_sampler=False, lean=False)`
      (LNCC under `tf.recompute_grad` when lean). The data-volume warps pass `lean=lean_sampler, lean_images_grad=False`.
      Today's code paths stay exactly as they are when both flags are False.
      Files: `runtime/losses.py`, new `tests/test_lean_losses.py`: losses ≤ 1e-6 relative and gradients ≤ 1e-4 relative
      (fp32) with lean on vs off, default path bit-identical, combined with `lean_sampler` on/off.
      Verify: `python -m unittest tests.test_lean_losses tests.test_runtime_losses_metrics -v` passes.

- [ ] 6. Split group-mean conv (R2.4, §3.5).
      `GroupConvBlock.split_group = False` (Python attribute). When True, with `group=True` and `self.conv.built`, the
      block computes norm → act → `conv3d(x, k[..., :C, :]) + conv3d(mean, k[..., C:, :])` + bias, in the compute
      dtype, with the layer's strides, 'SAME' padding, NDHWC.
      `layers.set_split_group_conv(layer, on)` sets it only on `GroupConvBlock`s with `group=True` and
      `pre_activation=True` (the stem is excluded).
      `GroupRegLevelModel(..., split_group_conv=False)` calls the setter after construction. `FrozenStage` keeps False.
      Files: `models/layers.py`, `models/group_net.py`, new `tests/test_split_group_conv.py`:
      - outputs ≤ 1e-6 relative fp32 (≤ 1e-3 fp16) and gradients ≤ 1e-4 relative;
      - default path bit-identical;
      - variable names and count unchanged;
      - the stem keeps `split_group=False`;
      - `copy_weights_from` works between split and non-split models;
      - strided blocks (l4–l6) included.
      Verify: `python -m unittest tests.test_split_group_conv tests.test_group_layers tests.test_group_net -v` passes.

- [ ] 7. Flags, TrainConfig, Trainer and `main.py` wiring (R2.5, R2.6, §3.8).
      - `runtime/args.py`: add `--lean-sampler`, `--lean-losses` and `--split-group-conv` (`str2bool`, default false).
        `--lean-sampler` and `--lean-losses` without `deform` in `--task-ratio` give `parser.error` naming the option.
      - `TrainConfig`: `lean_sampler: bool = False`, `lean_losses: bool = False`, filled by `train_config_from_args`.
      - `Trainer.__init__` raises `ValueError("lean_sampler: model %s vs TrainConfig %s")` on a mismatch.
        `_make_deform_step` passes `lean=cfg.lean_sampler` to `compose_cascade`, and
        `lean_sampler=cfg.lean_sampler, lean_losses=cfg.lean_losses` to `deform_loss`. Every validation path keeps the
        defaults.
      - `main.py`: pass `lean_sampler` / `split_group_conv` to the `GroupRegLevelModel` constructor. Factor
        `_build_n(args)` = today's rule, then `min(n_build, 2)` when `args.split_group_conv`.
      - Update the module docstrings (args defaults list, Trainer).
      Files: `runtime/args.py`, `runtime/run.py`, `main.py`; tests added to `tests/test_main.py` and
      `tests/test_runtime_infra.py`:
      - invalid values of every new flag (bool parse errors, task incompatibility, non-positive widths);
      - `_build_n` (split, group 16 / deform 8 → 2; without the flag → 8);
      - a `ModelConfig` dict and an `args` dict of today's format (no new keys) restore with today's defaults;
      - `check_cascade_config` accepts today's pair;
      - the Trainer mismatch error.
      Then one e2e local run (script in `.agents/tmp`):
      `python main.py --overfit-one-group --overfit-synthetic --level 4 --task-ratio deform:1 <NARROW_ARGS>
      --gradient-checkpointing full --lean-sampler true --lean-losses true --split-group-conv true
      --lambda-ncc-template 1 --lambda-ncc-pair 1 --gpu-memory-limit-mb 6000 --steps …`,
      followed by `tests/gpu/check_mixed_run.py` on its run dir (take the options from the m6 / m7 usage in
      `tests/test_check_mixed_run.py`). Optionally add it to `tests/test_main.py` if it runs in < 2 min.
      Verify: `python -m unittest tests.test_main tests.test_runtime_infra -v` passes; the e2e run and its checker exit 0;
      both goldens `max |diff| 0`.

- [ ] 8. T = 6 vs T = 7 accuracy test and verdict (R2.3 Option B, R3.5, §3.6).
      New `tests/test_integration_steps.py` (`tests.tf_setup` first), also runnable as
      `python -m tests.test_integration_steps --max-v 8 --out <json>`:
      - grids level 4 (sigma 3) and level 1 (sigma 6), T 6/7, max |v| 2/4/8;
      - max and mean inverse error in both orders, with interior margin ceil(max |u|) + 3;
      - `folding_fraction`;
      - assertion only on level 4, max 2, T 7: max < 0.1 and mean < 0.01;
      - the `t6_accuracy_ok` rule of §3.6; writes `{"t6_accuracy_ok", "cases", "criteria"}`.
      Default `--out` is `$BGR_SCRATCH/m9/integration_steps.json`, else the repo's `scratch/m9/` (the code default from
      the design). `scratch/` is NOT git-ignored (checked during planning), so every LOCAL run in this phase passes
      `--out .agents/tasks/milestone-9/<name>.json` explicitly. Never leave files in the repo's `scratch/`.
      Files: `tests/test_integration_steps.py`.
      Verify: `python -m unittest tests.test_integration_steps -v` passes, prints the table, and its runtime is
      recorded (budget ~3 min). The script mode writes the JSON.

### B. Calibrated memory model (TF-free; §4)

- [ ] 9. Calibration data `models/memory_calibration.json` (R3.2, A6, §4.3).
      Transcribe every point:
      - requirements §2: level-2 deform fp32 / AMP × none / integrator / both at each N, including the fp32 none N6
        `oom`; level-2 linear fp32 N4/8 + N12 `failed_d2`, AMP N4/8/12/16 (with `build_n = n`);
      - level-4 A6000 N8/12/16 (`lncc: false`);
      - the GTX 1080 default and narrow points, including OOMs, from `.agents/tasks/milestone-6/memory.json`;
      - the LNCC on/off points from `.agents/tasks/lncc-loss/memory.json`;
      - the level-4 linear N32 `range` point.
      Every record carries `source, gpu, cap_gib, status, measure: "peak_mb", lncc, cascade, build_n` plus the
      workload fields. Read the two local JSONs to get exact MB values and their configs (widths, ckpt, T).
      Files: `models/memory_calibration.json`.
      Verify: the loader test in item 11 parses every record and validates the fields.

- [ ] 10. `models/memory_model.py` and the `scripts/memory_model.py` CLI (R3.1–R3.7, §2.1, §4.1–§4.5).
      Implement in stdlib + numpy (scipy only inside `fit`):
      - `Workload`, the inventory built from the architecture (stages, resolutions, C_in, group flags,
        5·C_in / 4·C_in + C_in/N, head 4·C, skip gradients `B_sg`);
      - the sampler-call records (name, C, share, images_grad, term, order) and the phases F / Fw / Fb / I / D / E / Z / V;
      - the coefficient table of §4.3, resident per touched part (U16), `limit_flags` (training-step tensors and the
        build forward at `build_n`, `--tensor-limits` override);
      - `predict` (returns the §4.1 fields), `fit` (`least_squares` soft_l1, f_scale 0.05, OOM hinge residual; exit 1
        rules), the leave-one-mode-out refits, the hold-out gate;
      - `grid` (the full R3.4 grid), `recommend` (§4.4 steps 1–6, per-N table under the timed rule and both bounds, the
        T rule with `--t6-verdict`, exit codes 3 / 4, `--n-target`, `--margin`, `--print-env`, `recommendation.json`
        with the N4 fields);
      - the step-time model with the m7 consistency check (§4.5).
      Python-3.8 syntax. CLI: `scripts/memory_model.py {fit, predict, grid, recommend}`. `fit` writes
      `models/memory_coefficients.json`, which every other command and the measurement tool read by default.
      Files: `models/memory_model.py`, `scripts/memory_model.py`, `models/memory_coefficients.json` (generated by
      `python scripts/memory_model.py fit`).
      Verify: `python scripts/memory_model.py fit` exits 0 in `bgr_preproc38` and `tf_gpu`, and prints the accuracy
      table (every `ok` point within ±15 %, OOM / failed points infeasible) and `warmup_excess_gib`.
      `python scripts/memory_model.py recommend --out <tmp>/recommendation.json` prints the per-N table. Expected
      pre-measurement: `f0d1` LS, N 4, ~33.5 GiB. The generated table is authoritative.

- [ ] 11. Memory-model tests `tests/test_memory_model.py` (no TF; AC7, AC8, §4.6).
      Implement every bullet of §4.6, including:
      - the pre-fit pins (the six level-2 deform rows within 0.1 GiB including `B_sg`; `f0d1` LS N4 33.5 ± 0.1 phase D,
        Fw 32.0 ± 0.2, Fb 25.5 ± 0.2; bounds 41.0 ± 0.2 / LLS 35.0 ± 0.3 / skip 34.9 ± 0.1);
      - the Fw / Fb rules; the limit-flag cases (`f0d1` N4 unsplit flagged, split passes; `build_n` 6 vs 2; m8 linear
        fp32 N12);
      - "the build never changes `peak_gib`"; the level-2 linear table; the m8 recipe within 15 % of 27.8;
      - LNCC on/off for c_loss ∈ [0, 1]; resident per touched part; the fake-OOM `fit` exit 1; the hinge;
      - T6 exclusion and selection branches; the `recommendation.json` fields (including N4); the step-time chain;
      - the leave-one-mode-out errors and the hold-out table printed (hold-out records may be absent until item 18; the
        gate test then skips with a message, and is required to run after item 18);
      - a calibration-file schema check.
      Files: `tests/test_memory_model.py`.
      Verify: `python -m unittest tests.test_memory_model -v` passes in both `bgr_preproc38` and `tf_gpu`, with
      `'tensorflow' not in sys.modules` asserted.

- [ ] 12. Step-time extractor (R3.6, §4.5).
      New TF-free `scripts/step_times_from_metrics.py` (`--runs`, `--min-step`, `--extra-keys`, `--args-keys`, `--out`;
      median / IQR / window counts, `per_task`, max of extra-key window means, `missing` handling, exit 0).
      Files: `scripts/step_times_from_metrics.py`, new `tests/test_step_times.py` (two fake run dirs: cascade-style and
      non-cascade, fake `config.json`, a missing run).
      Verify: `python -m unittest tests.test_step_times -v` passes in both envs.

### C. Measurement tooling (§5)

- [ ] 13. Per-tensor-limit probe (R1.3, AC6, §2.4, §5.3).
      New `tests/gpu/probe_tensor_limit.py`, modes `graph | run | m8-errors`:
      - `configure_tf_precision` is the first TF call in `graph` / `run`; `m8-errors` imports no TF;
      - `graph` builds the level-2 N 12 default model with a fixed-shape input, without executing it, and lists sorted op
        outputs with more than 2³¹ − 1 elements;
      - `run` covers the forward ops and the GradientTape backward ops of §2.4 at E = 2³¹ ± 2²⁴ fp32 / fp16 and fp16
        3.52·10⁹ / 4.4·10⁹. A byte estimate gives `skipped_mem`. Writes `verified_max_elements`;
      - `m8-errors` prints the failing records and the `peak_mb` / `warmup_peak_mb` columns.
      Files: `tests/gpu/probe_tensor_limit.py`, new `tests/test_probe_tensor_limit.py` (graph mode lists the D2 concat
      first; run mode with tiny `--sizes` has a status for every op and direction and the summary; m8-errors on a fake
      JSON).
      Verify: `python -m unittest tests.test_probe_tensor_limit -v` passes. Running
      `python tests/gpu/probe_tensor_limit.py --mode graph --level 2 --n 12 --widths default --task rigid` prints the
      op, its shape and `hypothesis limit: 2147483647 elements (int32 index); run mode decides`. Keep the output for
      the report.

- [ ] 14. Measurement tool for level 1, plus the frozen-chain stand-in (R1.3 fix, R4.1, R4.2, R4.4, §5.1, §5.2).
      Extend `tests/gpu/measure_level_memory.py`:
      - `--level` (4, 2, 1); `--widths` adds presets and `custom` (with `--filters` / `--decoder-widths`, key
        `custom:f=…;d=…`); `--gradient-checkpointing` accepts `full`;
      - bool-list axes `--lean-sampler / --lean-losses / --split-group-conv`; an int-list `--integration-steps` axis
        (0 resolves to the ModelConfig default before the record is written);
      - `--build-n` (default 2, < 1 → `ap.error`): every model builds at `min(n, build_n)`, which fixes R1.3;
      - `--val-pair-chunk` (default 16, < 1 → `ap.error`, used by `_val_pass`);
      - `--cascade true` at level 1 uses `tests/gpu/frozen_chain_standin.py` (`FrozenChainStandIn`, `build_standin`,
        `patched_prev_level()`, restored in `finally`);
      - `frozen_peak_mb` / `frozen_status`;
      - `predicted_peak_mb`, `predicted_phase`, `pred_over_meas`, `largest_tensor_elems`, `largest_tensor_desc` from
        `memory_model.predict` (null with a warning without coefficients);
      - the `limit` status classification;
      - `KEY_FIELDS` and old-record defaults (§5.1); the per-configuration atomic write (`<out>.tmp.<pid>`, fsync,
        `os.replace`, OSError fatal, temp removed in `finally`).
      Pass `lean_sampler` / `lean_losses` into `TrainConfig`, and `lean_sampler` / `split_group_conv` into the model
      constructor. Update the docstring and the JSON schema description.
      Files: `tests/gpu/measure_level_memory.py`, `tests/gpu/frozen_chain_standin.py`, `tests/test_measure_memory.py`
      (keep `test_tiny_level2_cascade`; add `test_tiny_level1`, `test_merge_old_record` with the two cases,
      `test_incremental_write`).
      Verify: `python -m unittest tests.test_measure_memory -v` passes on the GTX 1080 (level-1 tiny N 2 records `ok` or
      `oom`, never crashes; the JSON has `predicted_peak_mb`, `largest_tensor_elems`,
      `cascade_kind: "standin_chain_4_2"` and `frozen_peak_mb` or `frozen_status`). Record the added test time.

- [ ] 15. `verify_dali_pipeline.py --level 1` (§5.4; pure-Python parts only).
      Allow `--level` choices (4, 2, 1). `_set_level(1)` and `_template_brain` read the level-1 template (suffix "").
      Extend `tests/test_verify_dali_level.py` with a level-1 `_set_level` / template case.
      Files: `tests/gpu/verify_dali_pipeline.py`, `tests/test_verify_dali_level.py`.
      Verify: `python -m unittest tests.test_verify_dali_level -v` passes. No DALI run locally (DALI is not installed).

- [ ] 16. `slurm/m9_memory.sh`, its tests, and the runner script (R4.3, AC11, §5.5).
      Write the script from §5.5 verbatim, with the N5 comment fix. Keep `-A qtim` only, `--gpus=1`, `-N 1`,
      `--ntasks-per-node=1`, `--cpus-per-task=3`, `--mem=64G`, `-p rtx8000`, `--time=10:00:00`, and the log paths under
      `/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out|.err`. Outputs go under
      `$PROJ/scratch/m9`, using `tf10_nv.sif` and `</dev/null` on every container call. Include the dgx-a100 variant,
      the DO-NOT-SUBMIT note and the exact line `sbatch --job-name=m9mem --time=10:00:00 --mem=64G slurm/m9_memory.sh`.
      LF line endings only.
      `tests/test_slurm_scripts.py`:
      - add the script to the static header checks;
      - `test_m9_memory_script`: the `"${RUN[@]}"` line count equals the `</dev/null || rc_x=$?` count, plus
        `-p dgx-a100` and the exact sbatch line;
      - a stub run with the call order and arguments: step 2b `--cascade false`, `$KNOBS` split into arguments, step 5a
        / 5b arguments, exit-code propagation.
      `tests/run_tests_wsl.sh`:
      - add `slurm/m9_memory.sh` to the `bash -n` loop;
      - add `scripts/memory_model.py scripts/step_times_from_metrics.py` to the py_compile / ast file list (only these
        two new files, not every file in `scripts/`);
      - add a TF-free unittest line in stage 1 (`python -m unittest tests.test_memory_model tests.test_width_presets
        tests.test_step_times tests.test_slurm_scripts`) and assert that TF was not imported.
      Files: `slurm/m9_memory.sh`, `tests/test_slurm_scripts.py`, `tests/run_tests_wsl.sh`.
      Verify: `bash -n slurm/m9_memory.sh`; `python -m unittest tests.test_slurm_scripts -v` passes in `bgr_preproc38`.

### D. Local measurements, recommendation, report

- [ ] 17. Syntax and compatibility sweep.
      `py_compile` + `ast feature_version=(3, 8)` over `models/*.py runtime/*.py data_loading/*.py main.py tests/*.py
      tests/gpu/*.py scripts/memory_model.py scripts/step_times_from_metrics.py` in `bgr_preproc38`, and `bash -n` on
      all slurm and runner scripts. Check every new numpy / scipy call against numpy 1.21 / scipy 1.4 (e.g. no
      `np.random.Generator` methods newer than 1.21, no `scipy.optimize` options newer than 1.4).
      Verify: the stage-1 portion of `run_tests_wsl.sh` passes (run it via a `.agents/tmp` script that executes only
      stage 1, or the full runner in item 21).

- [ ] 18. Local hold-out and knob-slope measurements (§5.6, §3.6, §4.3; "local level-4 / level-2 memory slopes for the
      new knobs").
      On the GTX 1080 with `TF_CUDNN_USE_AUTOTUNE=0` and `--gpu-memory-limit-mb 6000`, run the three hold-out
      invocations of §5.6 into `.agents/tasks/milestone-9/memory_holdout.json`:
      - level 4 narrow N 2/3/4;
      - level 4 default N 2;
      - level 2 tiny cascade N 2 (T 7 if it fits, else T 3).
      Also run the 8-record T6 / T7 invocation of §3.6. Transcribe the gated knob combinations into
      `models/memory_calibration.json` with `status: "holdout"`.
      If any `ok` hold-out point is outside ±20 %, or a hold-out OOM is predicted feasible, correct the inventory of
      that knob (do not widen the gate) and document it.
      Re-run `python scripts/memory_model.py fit`.
      Files: `.agents/tasks/milestone-9/memory_holdout.json`, `models/memory_calibration.json`,
      `models/memory_coefficients.json`, `models/memory_model.py` (only if a correction is needed).
      Verify: `python -m unittest tests.test_memory_model -v` passes with the hold-out gate active (not skipped). The
      printed hold-out table goes into the report.

- [ ] 19. T6 verdict, level-1 grid and recommendation (R3.4–R3.7, AC8, AC9).
      Local outputs go to `.agents/tasks/milestone-9/` (the repo's `scratch/` is not git-ignored):
      - Run `python -m tests.test_integration_steps --max-v 8 --out .agents/tasks/milestone-9/integration_steps.json`.
      - Run `python scripts/memory_model.py grid --out .agents/tasks/milestone-9/grid.json`.
      - Run `python scripts/memory_model.py recommend --t6-verdict .agents/tasks/milestone-9/integration_steps.json
        --out .agents/tasks/milestone-9/recommendation.json --print-env`.
      If the recommendation differs from `f0d1` LS, update the `WIDTHS` / `KNOBS` defaults in `slurm/m9_memory.sh` to
      the printed env, and keep the stub test in line. If `recommend` exits 3 (R3.7) or 4, stop and message Jay with
      the verdict (send_message warning). Do not choose N < 4 silently.
      Files: possibly `slurm/m9_memory.sh`, `tests/test_slurm_scripts.py`; outputs in `.agents/tasks/milestone-9/` and copied
      into the report.
      Verify: `recommend` exits 0 with deform N* ≥ 4 at ≤ 44 GiB including frozen stages and validation, or the R3.7
      verdict is reported to Jay. `tests.test_slurm_scripts` still passes.

- [ ] 20. Report and design-doc update (R6.1, R6.2, AC12).
      Write `.agents/tasks/milestone-9/report.md` with:
      - the R1 breakdown (level-2 inventory and phases, cross-check table, level-1 scaling and the largest tensors;
        reference the design tables and state the measured / fitted deltas);
      - the per-tensor-limit result (graph-mode output, run mode pending on the cluster);
      - the knobs with their verification numbers (checkpointing tests, lean-sampler graph-mode A/B/C numbers and
        whether `_DY_DEP` mattered, the chain or `while_loop` form used, split / lean-loss tolerances);
      - the memory-model accuracy table (R3.3) including leave-one-mode-out, the hold-out table and
        `warmup_excess_gib`, plus the fitted coefficients;
      - the level-1 grid summary, the per-N table under the timed rule and both bounds, and the recommendation (R3.6
        fields, A100 verdicts, levels 4 / 2 peaks, step times labelled ±50 %, long-run wall time);
      - the T6 vs T7 numbers and verdict;
      - the tooling and the exact `m9_memory.sh` command for Jay (DO NOT submit without approval);
      - the R5 pointers to design §6;
      - the added suite time (N3);
      - the requirements correction F16 (level-4 / GTX points are Dice only);
      - the open decision for Jay (`--n-target 4`, or N ≥ 5 via no margin / forward corner sequencing).
      Apply the N1–N4 wording.
      Update `.agents/tasks/network-design/design.md` sections 4.5, 4.6, 9 and 11 with the implemented and measured
      outcomes, and bump the version to v3.14 with a status-line entry. v3.13 already holds the design content, so the
      bump records the implementation.
      Files: `.agents/tasks/milestone-9/report.md`, `.agents/tasks/network-design/design.md`.
      Verify: every R6.1 item is present (self-check against the list above). The numbers match the JSON outputs from
      items 13, 18 and 19.

- [ ] 21. Full local suite and cleanup (AC1).
      Run `tests/run_tests_wsl.sh` in the background (timeout ≥ 4 h). Expected: stage 1 OK (py3.8 checks, bash -n
      including m9, TF-free tests). Stage 2 OK with 381 + new tests, 4 + new GPU/DALI skips, and both goldens printing
      `[golden] max |diff| 0` / `[golden deform] max |diff| 0`. Record the counts, the wall time and the added time
      versus ~3 h in the report.
      Delete the `.agents/tmp` scripts and any `__pycache__` / temp outputs created during the work, including anything
      written to the repo's `scratch/` (it is not git-ignored). Keep the JSON outputs in `.agents/tasks/milestone-9/`.
      Verify: runner exit status 0; `git status` shows only the intended new and modified files.

## Acceptance-criteria map

| AC | Items |
|---|---|
| 1 suite + goldens | 3–7, 21 |
| 2 old config restore, cascade check | 7 |
| 3 full checkpointing (a)–(e) | 3 |
| 4 knob equivalence + default bit-identity | 4, 5, 6 |
| 5 parser invalid values | 1, 3, 7 |
| 6 tensor-limit reproducer + build_n fix | 13, 14 |
| 7 memory-model tests | 9–11, 18 |
| 8 grid + recommendation | 10, 19 |
| 9 T6 vs T7 | 8, 19, 20 |
| 10 tool level 1, JSON fields, level-2 test | 14 |
| 11 m9 script + stub test | 16 |
| 12 report + design R5 | 20 (R5 content already in design §6; the report points to it) |

## Gaps and assumptions

- Checked during planning: `.gitignore` ignores `.agents/tmp/`, `results/` and `logs/`, but NOT `scratch/`. All local
  JSON outputs therefore go to `.agents/tasks/milestone-9/` via explicit `--out` arguments.
- `case A or B fails, stop` also means: send_message severity warning to Jay with the measured numbers.
- Where `_GrapplerOff` / `_OpDeterminism` live (test_cascade / test_trainer): importing them across test modules is
  acceptable. A shared helper is allowed if behaviour is unchanged.
- `synthetic_group(1, n, task)` and `fd.synthetic_deform_group(1, 2)` at level 1 are untested. Their runtime and memory
  are checked in item 14. If the numpy/scipy phantom is too slow, the tool may build the level-1 group by upsampling the
  level-2 phantom. Document it.
- If any assertion of the §3.3 graph-mode memory test fails, follow the design's fallback (case C → `while_loop`). If
  case A or B fails, stop and report it (it would invalidate the lean-sampler inventory) before continuing to items
  18–19.
