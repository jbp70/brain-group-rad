# Level-1 memory knobs, calibrated memory model and level-1 measurement job (milestone 9, phase 1)

Phase 1 adds four memory knobs, all off by default: `--gradient-checkpointing full`, `--lean-sampler`, `--lean-losses` and `--split-group-conv`. It also adds named width presets, a TF-free memory model fitted on 45 measured points, level-1 support in the measurement tool (`--isolate true`, one configuration per process) and `slurm/m9_memory.sh`. The recommended level-1 configuration is `f0d1` + lean sampler + split, AMP, `full`, T 7, deform N 4, predicted at 34.99 GiB. With the defaults, the existing level-4 / level-2 paths are unchanged and both goldens report max |diff| 0 (coder evidence: 477 tests OK, 4 skips). The review found no correctness problem in the checkpointing, the lean sampler or the split conv. The memory-model gate is one-sided and the tool leak is fixed with `--isolate`; both are orchestrator decisions, documented in report.md, and are not findings here.

Watch for: step 2 of `m9_memory.sh` measures the three priority configurations of step 2p again under the same record key (confirmed). Report.md mentions the retraining plan only in a one-line pointer (confirmed). The recommended configuration's forward bound with the margin is 44.39 GiB, just above the 44 GiB budget; this is documented, and the job measures it.

**Verdict**: APPROVED

## High-level view

`full` checkpointing wraps the encoder (returning the 6 skips), the deform decoder and the two integrator blocks in `tf.recompute_grad`. `models/checkpointing.recompute` counts calls of the wrapped function: call 1 is the forward, and every later call snapshots the BN moving statistics and restores them after the recompute. So the statistics change only in the forward pass, however many gradient calls follow. The restore depends on resource ordering in tf.function's automatic control dependencies. Tests (b), (b'), (c), (d) and (e) check this, and they also show that `both` still applies its documented double update. Linear tasks never take the `full` path, and the build pass runs unwrapped, so the variables match `none`.

The lean sampler is a `tf.custom_gradient` that keeps only `images` and `coords`. Its forward uses the same float ops in the same order, in int64 index math, so it is bitwise equal to today's sampler. The backward recomputes the corner math from `coords + dep(dy)`, which blocks Grappler CSE, and accumulates the 8 corners sequentially under a control-dependency chain. The `_DY_DEP` off case in the graph-mode memory test shows that the dy guard is what keeps the memory down with Grappler on.

The split group conv computes conv(x, K[:C]) + conv(mean, K[C:]) + bias. The first call still goes through the Keras conv, so variables and checkpoints are unchanged. `main.py` and the tool build at `min(N, 2)` when split is on, because the unsplit build pass would otherwise create the large group concat.

With the knobs off, the new paths are not traced (Python switches read at trace time), and the `_call_deform` refactor keeps the op order of the existing modes. Python 3.8 compatibility was re-checked in this review in `bgr_preproc38` (Python 3.8.20): `py_compile` and `ast` 3.8 over 75 files, `bash -n slurm/m9_memory.sh`, and the 58 TF-free tests all pass.

`slurm/m9_memory.sh` meets every cluster rule in the brief and passes `--isolate true` on every measurement call. Step 2p measures the orchestrator's configurations first. Report.md gives the exact sbatch line and the A100 verdict (N 4 only without the margin, so level 1 trains on 48 GB cards). Step 2 of the script measures part of 2p again, and the report covers the retraining plan with a pointer only.

<details>
<summary>Issues (3)</summary>

1. **Step 2 re-measures the 2p priority configurations** (NIT, confirmed). Step 2 runs `f0d1` T 7 at N 3, 4 and 5 again with the same knobs and the same record key, so it adds about 18–30 min to the job, and a crashed re-run child overwrites a good 2p record with an `error` record. Drop N 3–5 at T 7 for the center widths from step 2, or skip configurations already recorded with status `ok`.
2. **Retraining plan is only a pointer in report.md** (NIT, confirmed). The report's R5 section names design 6.4 in one line. Add the three-step summary (level 4 → 2 → 1: script, group sizes, steps, GPU, wall time) so Jay can review the plan from the report alone.
3. **Recommended config at the budget edge under the forward bound** (NIT, confirmed from recommendation.json / report.md). `f0d1` LS N 4 is predicted at 44.39 GiB with the margin under the forward bound, above the 44 GiB budget. The report names `lean_losses: true` as the fallback, and step 3 measures it. When the cluster numbers come back, check this case before fixing N 4.

</details>

<details>
<summary>Details</summary>

### BN-once restore under `full` and where it depends on ordering

The guarantee that the recompute's own moving-average assigns run before the restore comes from tf.function's automatic control dependencies, which order ops on the same resource variable by creation order. The code has no explicit edge for it (confirmed; informational, no action needed). The docstring states the dependency (design U8), and test (b) checks it at the Trainer level: `full` stays ≤ 1e-7 from `none`, while `both` differs by 7.8·10⁻³. A future change that reads the statistics through a different resource handle, or that moves the restore outside the traced function, would break this silently, apart from test (b).

### Lean sampler backward

```
coords_b = coords + where(equal(b, b), 0, nan)     # b = bool(dy[0]); data dependency only
lo_i, frac = axis_terms(clip(coords_b))            # once, outside the chain
for k in 0..7:
    with control_dependencies([d_coords, d_flat]):  # corner k-1 accumulated
        idx, w, dw = corner(k); g = gather(flat, idx)
        d_coords += stack((sum(dy*g) - fill*sum(dy)) * dw_a)
        d_flat   += unsorted_segment_sum(dy*w, idx)  # only if images_grad
```

The dy guard depends on Grappler not folding `Equal(x, x)` on bool to a constant (confirmed dependency; informational). TF 2.10 has no such rewrite, and the measured `_DY_DEP`-off case (1.14 GiB vs 0.52 GiB) shows the guard takes effect. A TF upgrade could add the fold. Keep the graph-mode memory test (case B) as the regression check.

### Measurement job: duplicate priority configurations

`KEY_FIELDS` does not include `val_pass`, and step 2 uses the same `$KNOBS`, `--val-pair-chunk 4` and AMP as step 2p. So `f0d1` T 7 N 3 / 4 / 5 has the same key in both steps (confirmed). With `--isolate true`, `measure_isolated` runs the child again. If that child exits non-zero without a fresh record, the parent writes `error` under the key, which `write_merged` uses to replace the good 2p record. OOMs and limit errors are recorded by the child with exit 0, so only hard crashes (process abort, host OOM kill) trigger this. The duplicates also cost about 3 × 6–10 min of the 12 h limit. The header's 7 h estimate already counts them (49 level-1 configurations), so the margin holds.

### Test coverage

Covered: `full` gradients vs `none` (bitwise equal with Grappler off, fp32 and AMP), BN-once in Trainer steps, persistent tape, frozen statistics, recalibration counts, one trace per task, identical variables; lean sampler forward bitwise equality eagerly and in graph mode, gradients in fp32 and fp64, `images_grad=False`, the integrator, and graph-mode memory with Grappler on; split at block level (strided, fp16) and model level (TINY); lean-loss equivalence; parser errors; the slurm stub (`--isolate`, 2p order, exit codes); the TF-free memory model with the one-sided gate and the symmetric statistics printed.

Not tested: the lean sampler and split at level-1 shapes (cannot run locally; deferred to the cluster job); the parent of `--isolate` holding no GPU memory on a 48 GB card (local `resident_mb` check only, which measures the child's allocator, not other processes); `verify_dali_pipeline.py --level 1` (unverified until job step 1, as the docstring says).

</details>

<details>
<summary>File map</summary>

- `models/checkpointing.py` (new): `recompute` with BN snapshot/restore after call 1.
- `models/group_net.py`: `full` mode (`_call_deform_full`), `lean_sampler` / `split_group_conv` attributes, `_deform_outputs` split out.
- `models/layers.py`: lean trilinear sampler (custom gradient, sequenced backward), `batch_norm_layers`, `set_split_group_conv`, split path in `GroupConvBlock`.
- `models/diffeo.py`: `lean` passed through warp / compose / integrate / pairwise / cascade.
- `models/config.py`: integer ≥ 1 validation of filters / reductions / dense.
- `models/width_presets.py` (new): 7 presets and a TF-free parameter count.
- `models/memory_model.py`, `memory_calibration.json`, `memory_coefficients.json` (new): analytic model, calibration and hold-out data, fitted coefficients, one-sided gate.
- `runtime/args.py`, `runtime/run.py`, `main.py`: new flags, validation, TrainConfig fields, Trainer lean check, `_build_n`.
- `runtime/losses.py`: lean Dice, LNCC under recompute, lean-sampler warps for data volumes.
- `tests/gpu/measure_level_memory.py`: level 1, widths, knob axes, predictions, `--build-n`, `--isolate`.
- `tests/gpu/probe_tensor_limit.py`, `tests/gpu/frozen_chain_standin.py`, `scripts/memory_model.py`, `scripts/step_times_from_metrics.py` (new): tooling.
- `tests/gpu/verify_dali_pipeline.py`: `--level 1`.
- `slurm/m9_memory.sh` (new), `tests/run_tests_wsl.sh`: memory job, stage-1 additions.
- `tests/test_*` (new and modified): checkpointing, lean sampler / losses, split, presets, memory model, step times, integration steps, slurm scripts, measure tool, parser.
- `.agents/tasks/network-design/design.md`: v3.14 updates (4.5, 4.6, 9, 11).

Full diff: `git diff HEAD` plus the untracked files in `git status`.

</details>
