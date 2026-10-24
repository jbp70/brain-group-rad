# Milestone 9, Phase 1 report: level-1 memory root cause, knobs, calibrated model, tooling

Scope: requirements R1–R6 (`requirements.md`), design v5 (`design-m9-phase1.md`), plan items 1–21 (`plan.md`).
Everything here was done locally (WSL GTX 1080, TF limit 6000 MB, `TF_CUDNN_USE_AUTOTUNE=0`). Nothing was committed,
pushed or run on the cluster. Every level-1 number is a **prediction** until `slurm/m9_memory.sh` runs (command at the
end, needs Jay's approval).

## Summary

- **Recommendation (provisional):** widths `f0d1` (filters 16, 64, 128, 192, 256, 384; decoder 128, 96, 64, 32, 8;
  29.54 M parameters vs 29.63 M today), AMP, `--gradient-checkpointing full`, `--lean-sampler true`,
  `--split-group-conv true`, `--lean-losses false`, T = 7.
  - Level 1: deform N = 4 at a predicted 34.99 GiB (40.24 GiB with the 15 % margin, phase D); linear N = 16 at 13.51 GiB.
  - Level-1 training needs the 48 GB cards. The A100 fits N = 4 only without the margin.
- **New knobs**, all off by default. With the defaults both goldens report `max |diff| 0`. The knobs are
  `--gradient-checkpointing full` (BN statistics updated exactly once per step), `--lean-sampler`, `--lean-losses` and
  `--split-group-conv`, plus the width presets.
- **Memory model** (`models/memory_model.py`, TF-free):
  - Calibration: 45 fitted points within −3.7 … +2.1 %; every OOM and failed point is predicted infeasible.
  - Hold-out: 40 local points pass a **one-sided** gate (worst under-prediction −13.0 %). This is a documented deviation
    from design 4.3, covered below.
- **Two findings from the hold-out**, both handled:
  - (1) Under `full`, the two integrator blocks are recomputed **concurrently**. The inventory now counts both.
  - (2) The measurement tool leaked memory across configurations. Fixed with `--isolate true` (one configuration per
    process), which `m9_memory.sh` now uses.
- **T6 vs T7:** `t6_accuracy_ok = true`, but T = 6 brings no memory benefit in the recommended configuration, so T = 7
  stays.

## R1. Memory root cause

The full derivation is in design sections 2.1–2.5. Here are the measured and fitted deltas.

- **Level-2 inventory and phases** (design 2.2, 2.3): which phase sets the peak depends on the checkpointing mode:
  - `none`: end of the forward pass (F);
  - `integrator`: the integrator backward (I);
  - `both`: the decoder backward (D), in both precisions.
- **Cross-check against the 21 level-2 deform points (RTX 8000):**
  - pre-fit (all coefficients 1, `c_bwd` 2): −9.5 … +13.7 % over all 45 ok points; the level-2 deform rows −5 … +1 % as
    the design predicted;
  - fitted: −3.7 … +2.1 % over all 45 ok points; level-2 deform rows −2.4 … +1.8 %.
  - The m8 recipe (AMP `both` N 8) is predicted at 27.88 GiB against 27.80 measured.
- **Level-1 scaling** (design 2.5): at level 1 these run at full resolution (8× level 2):
  - stem, D1, SVF head, integration, every warp, the one-hot, the losses.
  - L2 / D2 stay at level-2 size.
- **Level-1 drivers, largest first:**
  - the sampler tapes: 14 compositions plus the 19-channel one-hot warps, about 0.7 KB per voxel per image for each
    one-hot warp with the default sampler;
  - D1 at full resolution with its backward transient;
  - the encoder matters little, because only the stem runs at full resolution.
- **Largest tensors:**
  - the D1 group concat [N, 160, 224, 192, 2·C1]: unsplit `f0d1` N 4 is above 2³¹ elements, which is why split is needed;
  - the one-hot [N, 160, 224, 192, 19];
  - the fields [N, 160, 224, 192, 3];
  - at level 2: the D2 concat [N, 80, 112, 96, 256].

## Per-tensor limit (R1.3)

Graph mode (`probe_graph_level2_n12.txt`): level 2, N 12, default widths, rigid (with the deform forward), 7023 ops.
Exactly one op output is above 2³¹ − 1 elements:

```
ConcatV2  group_reg_level_model_1/deform_decoder/d2/concat  [12, 80, 112, 96, 256]  float32  2642411520 elements  10.57 GB
hypothesis limit: 2147483647 elements (int32 index); run mode decides
```

- This explains the m8 linear fp32 N 12 `d2` failure: the old tool ran `build_model(n)` with the deform decoder at N 12.
  The tool now builds at `min(N, --build-n 2)`.
- The memory model flags the record infeasible ("build/d2 group concat … > 2147483647 (fwd)").
- **Run mode** (forward and backward kernels at 2³¹ ± 2²⁴ elements) is pending: job step 0a writes `tensor_limit.json`.
- Until then, backward kernels are bounded at 2³¹ − 1 elements in both precisions.

## R2. Knobs and their verification

- **`--gradient-checkpointing full`** (`models/checkpointing.recompute`): call 1 is the forward; every later call
  snapshots and restores the BN statistics.
  - (a) gradients `full` vs `none` are bitwise equal (rel 0) in fp32 and AMP, with Grappler off and op determinism on.
    With Grappler on, AMP alone differed by 4.7·10⁻⁴ rel.
  - (b) BN statistics after one step are equal to `none` (rel 0); `both` still shows the double update (decoder
    7.8·10⁻³).
  - (b') a persistent tape with 2 gradient calls still gives one update.
  - (c) frozen statistics are unchanged.
  - (d) `recalibrate_bn` makes exactly 2 updates for 2 batches.
  - (e) one trace per task over 6 steps.
  - (f) the same variables are created.
- **`--lean-sampler`:** custom-gradient trilinear sampler; its backward is a control-dependency chain over the 8
  corners. No `while_loop` fallback was needed.
  - Forward: max |diff| 0 eagerly and in `tf.function`.
  - Gradient error: fp32 coordinates 8.6·10⁻⁸ rel, images 7.4·10⁻⁸ rel; fp64 1.6·10⁻¹⁶.
  - Graph-mode memory test (level-2 field, N 2, Grappler on, GiB):

    | Case | Lean | Default | Model | Bound |
    |---|---|---|---|---|
    | A one-hot warp C19 | 0.368 / 0.375 | 0.413 | 1.500 | 2.249 |
    | B integrator T7 | 0.516 / 0.516 | 2.439 / 2.458 | 0.827 (saving 1.65) | 1.240, saving ≥ 0.85 (measured 1.92 / 1.94) |
    | B lean, `_DY_DEP` off | 1.137 / 1.118 | | | printed only |
    | C1 backward C19, zero fill | 0.256 / 0.250 | | 0.385 | 0.58 |
    | C2 backward C3, border | 0.211 / 0.218 | | 0.276 | 0.41 |

    `_DY_DEP` matters: without the dy dependency, Grappler's dedup keeps about 0.6 GiB more.
  - N2 wording: the lean sampler computes its indices in **int64 on the GPU**. This transient (168 B per output voxel
    per image in the forward) is modelled explicitly. int64 was chosen because int32 elementwise kernels are
    host-memory on the GPU. int32 would also be correct, since N·V1 ≤ 5.5·10⁷ < 2³¹. "int32 indices pinned to the GPU"
    is listed as an Fw remedy next to forward corner sequencing.
- **`--lean-losses`:** loss terms rel 0.0, gradients 2.7·10⁻⁷ rel, for every combination with the lean sampler on and
  off.
- **`--split-group-conv`:**
  - block level: fp32 outputs ≤ 1·10⁻⁶ rel, gradients ≤ 1·10⁻⁴; fp16 4.6·10⁻⁴.
  - model level (TINY): fp32 7.0·10⁻⁷, AMP 1.05·10⁻³, gradients 1.7·10⁻⁶.
  - With NARROW widths the model-level fp32 gradient differs by 1.43·10⁻⁴. The algebra is exact; the difference comes
    from training-mode normalisation over N 3, which is ill-conditioned. That is why the model-level test uses TINY.
- **Width presets:** `count_params` matches the built models exactly for all 7 presets.
- **End-to-end:** `main.py` level 4 NARROW N 4 with every knob plus LNCC 1/1 reaches Dice 0.88 at step 143 (eval
  template / pair 0.961 / 0.933, folding 0 %). `check_mixed_run.py` passes.

## R3. Memory-model accuracy (R3.3)

Fitted coefficients (`models/memory_coefficients.json`). None is at a bound.

| c_net | c_enc32 | c_enc16 | c_loss | c_rec | c_warp | c_bwd | c0_cluster | c0_gtx1080 |
|---|---|---|---|---|---|---|---|---|
| 0.999 | 1.064 | 0.749 | 0.531 | 0.912 | 1.197 | 2.694 | 0.103 GiB | 0.090 GiB |

- `c_enc16` (0.75) is below the design's expected 0.89 but inside its bounds [0.6, 1.6].
- Calibration accuracy: 45 ok points fitted within −3.7 … +2.1 % (pre-fit −9.5 … +13.7 %).
- Infeasibility: all 8 OOM points are predicted infeasible. The marginal one is GTX narrow `none` N 6 at 5.94 against a
  5.86 GiB cap. The `failed_d2` point carries the limit flag. The level-4 linear N 32 range point is predicted at
  22.1 GiB; the upper end of its 26–34 range is respected, the lower end is not (the range itself is unverified, U14).
- `warmup_excess_gib` = 0.207.
- Leave-one-mode-out errors:

  | Group | Errors |
  |---|---|
  | fp32_none | −0.2 … +0.9 % |
  | fp32_integrator | +1.6 … +2.6 % |
  | fp32_both | +1.5 … +2.5 % |
  | amp_none | −2.8 … +0.6 % |
  | amp_integrator | −2.8 … −0.9 % |
  | amp_both | −6.5 … −4.6 % |

### Hold-out (design 4.3, 5.6) and its deviation

Measurements, all saved under `.agents/tasks/milestone-9/`:
- The three design 5.6 invocations plus the 8-record T6/T7 invocation (design 3.6) were run first as multi-configuration
  processes. The raw output is kept as `memory_holdout_multiconfig.json`.
- The 40 gated configurations were then re-measured **one configuration (one N) per process** and merged into
  `memory_holdout.json`. The information-only combinations in that file stay multi-configuration.
- The four gated knob sets (`full`; +L; +L+LL; +L+LL+S; fp32 and AMP) are transcribed into `models/memory_calibration.json`
  with `status: "holdout"`.

**Correction 1: inventory (applied).** Under `full`, the exp(v) and exp(−v) integrator blocks are independent in the
backward graph, so TF recomputes them **concurrently**. Phase I holds two block tapes, not one as design 3.2 / 4.2
assumed. Evidence:
- level-2 tiny `full` with the default sampler measured 5.28 – 5.48 GB; the model predicted 3.67 – 3.79 before the fix
  and 6.18 – 6.30 after;
- T 6 → 7 adds 721 MB at N 2, which is 2 compositions per step, against a predicted 705 MB.

The fix is in `models/memory_model.py` (`_Inventory._deform`, docstring) and `test_lean_integrator_moments`. No
calibration record uses `full`, so the fit is unchanged. At level 1 the recommended row binds in D, so the recommendation
is unchanged as well. The only change in the per-N table is the N 2 row: default with no knobs becomes default + L.

**Correction 2: tool memory leak (fixed).** In one process, the allocator kept memory across configurations even after
`clear_session()` + `gc.collect()`:
- `resident_mb` grew from 73 to 329 MB over 48 narrow configurations, and from 424 to 1602 MB over 16 default ones;
- measured peaks of later configurations were inflated by up to about 1.2 GB.

The fix is `measure_level_memory.py --isolate true`: every configuration runs in its own child process, merged into the
same JSON with the same atomic write. A real check on 4 isolated level-4 narrow configurations gives `resident_mb`
73.4 / 73.4 / 73.6 / 73.6 MB. The new tests are `tests/test_measure_memory.TestIsolate`: one child per configuration,
skip after a child's OOM, an error record when a child crashes, and the custom-widths argv. `slurm/m9_memory.sh` passes
`--isolate true` on every measurement call.

**Gate verdict (deviation from design 4.3, orchestrator decision).**
- **Symmetric gate (as designed): FAILED.**
  - 18 of the 40 ok points are outside ±20 %, all of them **over-predictions**: level-4 narrow AMP +21 … +28 %, narrow
    AMP L+LL+S +43 … +52 %, narrow fp32 L+LL+S +27 … +30 %, default AMP +23 / +25 / +36 %, default fp32 L+LL+S +20.1 %.
  - The prompt and the FEAT-004 blocked message said 21 failing points; the correct count is 18.
  - Every level-2 tiny point is within −13.0 … +17.5 %.
- **One-sided gate (adopted): PASSED.** It fails only on an under-prediction beyond −20 % or on a hold-out OOM predicted
  feasible. Over-predictions are conservative, so the margin comes from the worst under-prediction: −13.0 %, so
  m = max(0.15, 0.13) = **0.15**.
  - Implemented in `holdout_gate()`: `passed`, `max_under_err`, `n_over`, `symmetric_passed`.
  - `recommend` exit 4 now means an under-prediction failure.
  - Tests: `test_holdout_gate` (runs, not skipped) and `test_holdout_gate_one_sided_rule`.
- **Open model-form question:** the model over-predicts small level-4 points with AMP and with split by 20–52 %. Likely
  causes are fixed-size transients (cuDNN workspace, `c_bwd` at tiny channel counts) and split's per-half conv inputs
  being freed earlier than modelled.
  - It does not affect the level-1 selection: the margin is unchanged, and at margins 0.20 / 0.30 the N 4 pick would be
    `f0d1` LS / `fr50` L.
  - The cluster job measures both of those configurations first (step 2p).

Hold-out table (fitted coefficients, one configuration per process; GiB, error = (pred − meas) / meas):

| Point | Meas | Pred | Err |
|---|---|---|---|
| L2 tiny full AMP / +L / +L+LL / +L+LL+S | 5.16 / 2.36 / 2.33 / 2.32 | 6.03 / 2.05 / 2.05 / 2.05 | +17.0 / −13.0 / −12.0 / −11.7 % |
| L2 tiny full fp32, same four | 5.24 / 2.52 / 2.50 / 2.47 | 6.15 / 2.44 / 2.44 / 2.31 | +17.5 / −3.1 / −2.5 / −6.3 % |
| L4 default N2 AMP, same four | 1.64 / 1.71 / 1.62 / 1.42 | 2.02 / 2.02 / 2.02 / 1.93 | +23.1 / +18.2 / +24.6 / +36.2 % |
| L4 default N2 fp32, same four | 2.69 / 2.68 / 2.69 / 2.39 | 3.04 / 3.04 / 3.04 / 2.87 | +12.8 / +13.2 / +13.0 / +20.1 % |
| L4 narrow AMP full N2/3/4 | 0.72 / 1.03 / 1.38 | 0.90 / 1.28 / 1.65 | +24.6 / +23.5 / +19.7 % |
| L4 narrow AMP +L N2/3/4 | 0.49 / 0.68 / 0.86 | 0.62 / 0.86 / 1.10 | +26.0 / +26.6 / +27.9 % |
| L4 narrow AMP +L+LL N2/3/4 | 0.48 / 0.69 / 0.91 | 0.62 / 0.86 / 1.10 | +28.3 / +24.1 / +21.1 % |
| L4 narrow AMP +L+LL+S N2/3/4 | 0.39 / 0.56 / 0.71 | 0.59 / 0.80 / 1.02 | +52.0 / +42.8 / +43.0 % |
| L4 narrow fp32 full N2/3/4 | 0.87 / 1.30 / 1.64 | 0.97 / 1.38 / 1.79 | +10.5 / +6.3 / +9.5 % |
| L4 narrow fp32 +L N2/3/4 | 0.84 / 1.24 / 1.58 | 0.97 / 1.38 / 1.79 | +15.0 / +11.4 / +13.5 % |
| L4 narrow fp32 +L+LL N2/3/4 | 0.85 / 1.23 / 1.64 | 0.97 / 1.38 / 1.79 | +13.6 / +12.2 / +9.4 % |
| L4 narrow fp32 +L+LL+S N2/3/4 | 0.70 / 1.00 / 1.27 | 0.91 / 1.27 / 1.63 | +30.1 / +27.3 / +28.0 % |

Level-2 tiny `full` runs at T 7, since it fits under the 6000 MB cap. There were no hold-out OOMs among the gated
points. The only OOM was `integrator` + default sampler at T 7 (information, predicted infeasible at 6.70 GB).

## R3.4–R3.7. Level-1 grid and recommendation

Fitted coefficients, margin 0.15, budget 44 GiB, AMP + `full` + `chain42` + LNCC 1/1. Outputs: `grid.json`,
`recommendation.json`. `recommend` exits 0.

Grid summary (largest feasible deform N per width preset, timed rule):

| Widths | Params | N | Knobs / T | Peak |
|---|---|---|---|---|
| default | 29.63 M | 3 | LS, T 7 | 34.26 GiB, D |
| u075 | 22.17 M | 4 | LS, T 7 | 35.88, D |
| u050 | 16.82 M | 4 | L, T 7 | 32.23, I |
| u0375 | 15.00 M | 4 | L, T 7 | 32.13, I |
| **f0d1** | **29.54 M** | **4** | **LS, T 7** | **34.99, D** |
| fr50 | 29.14 M | 4 | L, T 7 | 32.57, I |
| fr25 | 28.96 M | 4 | L, T 7 | 32.18, I |

Per-N tables (peak / with margin, GiB; A100 verdict = ≤ 36 GiB with / without the margin):

| N | Timed rule (selection) | Skip-gradient bound | Order-independent estimate (one overlapping sampler forward) |
|---|---|---|---|
| 2 | default L 24.43 / 28.09 D, A100 yes / yes | default L 25.52 / 29.34 | default L 24.43 / 28.09 |
| 3 | default LS 34.26 / 39.40 D, A100 no / yes | default LS 35.89 / 41.27 | default LS 34.26 / 39.40 |
| 4 | **f0d1 LS 34.99 / 40.24 D**, A100 no / yes | f0d1 LS 36.34 / 41.80 (A100 no / no) | f0d1 **LLS** 34.99 / 40.24 |
| 5–8 | none feasible | none | none |

N1 wording: the bounds are the conservative variants (`skip_grad_bound=True`, `fwd_transient_bound=True`). The defaults
(both False) are the timed rule.

Recommendation, all R3.6 fields (`recommendation.json`):
- Configuration: `f0d1`, AMP, `full`, lean sampler + split, lean losses off, T 7.
- Level 1:
  - deform N* = 4 (`n_target` 4; 5 without the margin), 34.99 GiB, 40.24 with the margin, phase D;
  - `deform_peak_fwd_bound_gib` 38.60, `deform_peak_fwd_bound_margin_gib` 44.39; `bound_alt_knobs` `{"lean_losses": true}`;
  - linear N 16 at 13.51 GiB;
  - `val_pair_chunk` 16.
- A100 (≤ 36 GiB): deform N 3 with the margin, 4 without, 3 under the forward bound (`deform_n_no_margin_fwd_bound`);
  linear N 16. So level 1 trains on the 48 GB cards.
- Level 2, m8 recipe at `f0d1` widths: deform N 8 at 25.57 GiB, linear N 16 at 7.26 GiB. Largest A100 deform N:
  `both` 9, `full` 13.
- Level 4, m7 recipe: deform N 16 at 33.08 GiB, linear N 32 at 21.12 GiB. Largest A100 deform N with `none`: 17.
  Neither recipe is flagged.
- Step times (FLOP model, **±50 %**, pre-measurement):

  | Step | RTX 8000 | A100 |
  |---|---|---|
  | L1 deform | 11.4–34.3 s | 8.0–24.0 s |
  | L1 linear | 0.5–1.6 s | 0.3–1.0 s |
  | L2 deform | 3.4–10.2 s | 2.4–7.1 s |
  | L4 deform | 1.8–5.4 s | 1.3–3.8 s |

  The m7 consistency ratio is 1.12, so `r_task` is kept.
- Level-1 long run (8000 steps, 1:1:2, 16 validations): RTX 8000 13.8–41.3 h, A100 9.7–29.2 h. That is under 72 h and
  under the 7-day partition maximum.
- `--print-env`:
  `EXTRA_ARGS="--filters 16,64,128,192,256,384 --decoder-widths 128,96,64,32,8 --integration-steps 7 --amp true --gradient-checkpointing full --lean-sampler true --lean-losses false --split-group-conv true" N=16 DEFORM_N=4`,
  `WIDTHS="f0d1,default,fr50"`, `KNOBS="--lean-sampler true --lean-losses false --split-group-conv true"`. These are the
  existing `m9_memory.sh` defaults, so the defaults did not change.
- PROVISIONAL until `slurm/m9_memory.sh` measures it.

## R3.5. T = 6 vs T = 7

`integration_steps.json` (max |v| 8 voxels, level-1 grid, sigma 6): `t6_accuracy_ok = true`.
- The criterion is e = the mean interior inverse error, maximum over the two orders. e(6) = 0.00207 voxel and
  e(7) = 0.00202: e6 / e7 = 1.02 ≤ 2, and e6 ≤ 0.05.
- Folding is 0 / 0.
- The documented level-4 case (max 2, T 7): max 0.0119 / 0.0126, mean 0.0021.
- The m8 `diag_u_res_max_vox` window-mean maximum (the 2× rerun check) is pending from job step 0c.

Memory, measured T6 − T7 (level 2 tiny, AMP, N 2, multi-configuration run):

| Checkpointing | Default sampler | Lean sampler |
|---|---|---|
| `integrator` | T6 5429 MB, T7 OOM | +2 MB |
| `full` | +721 MB (360 MB per image; predicted 705) | −57 MB (noise) |

At level 1 `f0d1` LS N 4 the model predicts no difference, because phase D binds and I is not the peak.

**Verdict:** T = 6 is accurate enough but neither raises N nor restores headroom, so the T rule keeps **T = 7**.

## Tooling

- `tests/gpu/measure_level_memory.py`:
  - level 1, presets and custom widths, `full`, bool-list knob axes, a T axis;
  - `--build-n`, `--val-pair-chunk`, the frozen-chain stand-in;
  - predicted fields, the `limit` status, the per-configuration atomic write;
  - new: `--isolate`.
- `tests/gpu/probe_tensor_limit.py`: modes `graph | run | m8-errors`.
- `tests/gpu/frozen_chain_standin.py`.
- `scripts/memory_model.py` (`fit | predict | grid | recommend`) and `scripts/step_times_from_metrics.py`.
- `tests/gpu/verify_dali_pipeline.py --level 1`.
- `slurm/m9_memory.sh` changes in this step:
  - `--isolate true` on every measurement;
  - new **priority step 2p**, measured first: the recommendation (`f0d1`, `$KNOBS`, T 7) at N 3, 4, 5 with validation,
    and the fallback `fr50` + lean sampler only at N 4;
  - `--time` raised from 10 h to 12 h. Per-process start costs about 30 s × about 75 configurations, which is about
    0.6 h. With 2p the estimate is about 7 h; 12 h keeps a 1.7× margin.
- The step-2 grid (WIDTHS × T 6, 7 × N 2–8) also covers `f0d1` / `fr50` / default with the lean sampler + split at every
  N.

**Command for Jay. DO NOT submit without Jay's approval.** RTX 8000, account qtim, 1 GPU, 3 CPUs, 64G, 12 h. Run after
pushing and `git pull` in the cluster clone:

```bash
mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
sbatch --job-name=m9mem --time=12:00:00 --mem=64G slurm/m9_memory.sh
```

The A100 variant is described in the script header (`-p dgx-a100`).

## R5. Rest of milestone 9 (design only)

Not implemented in Phase 1. Design `design-m9-phase1.md` section 6:
- 6.1: the 3-level cascade, `PREV_LEVEL = {2: 4, 1: 2}`, `FrozenChain`;
- 6.2: the level-1 DALI pipeline;
- 6.3: level-1 validation, `--val-pair-chunk`;
- 6.4: the retraining sequence levels 4 → 2 → 1 with the recommended widths;
- 6.5: the comparison criteria. Against `m7_three_task_lncc` / `m8_level2_gradnorm`: center RMS ≤ +10 %, Dice ≥ −0.01,
  folding ≤ 0.05 %.

`--cascade-from` is still level-2-only.

## Suite and time (AC1, N3)

`tests/run_tests_wsl.sh` passes: stage 1 OK, stage 2 OK.
- Stage 1: `py_compile` and `ast` (3.8) over 75 files OK; `bash -n` on every script including `m9_memory.sh` OK; 58
  TF-free tests OK in `bgr_preproc38` without importing TF.
- Stage 2: **477 tests OK, 4 skips** (DALI / GPU-only), in 10669 s (2.96 h). Baseline was 381 tests + 4 skips in ~3 h.
  (Final run after the integration check, which added `TestCliErrors` for the measurement tool's option errors; the
  run before it had 476 tests in 10637 s.)
- Both goldens: `[golden] max |diff| 0 (tolerance 0)` and `[golden deform] max |diff| 0 (tolerance 0)`.
- Added suite time (N3): the 95 new tests fall within run-to-run noise. The two full runs took 3.19 h and 2.95 h
  against ~3 h before; the design estimated +16 min. Milestone-9 modules: checkpointing ~6 min, measure_memory ~6 min,
  lean sampler / losses / split ~2.5 min, integration steps ~1 min, the rest < 1 min.
- The first full run failed only in `test_no_tensorflow` of `test_memory_model` / `test_step_times`. Under full-suite
  discovery, other modules import TF first. Those tests now check the modules' own imports in that case, as
  `test_width_presets` already does. The second run passed.
- The known `test_recompute_matches` flake did not appear in either run.

## Corrections and notes

- **Requirements correction F16:** the level-4 A6000 and GTX calibration points are Dice-only (`lncc: false`). Only the
  `lncc-loss/memory.json` points and the level-2 points carry LNCC.
- **Design deviations recorded in this step:**
  - both integrator blocks are alive in I under `full`;
  - the hold-out gate is one-sided;
  - `m9_memory.sh` has step 2p, `--isolate true` and `--time` 12 h.

  Earlier deviations are in the FEAT-001..003 findings: model-level split tests on TINY; `recompute(..., training)`;
  resident memory counts optimizer slots for all trainable variables; "center" spelling in the script; the f0d1 pin is
  33.6 instead of 33.5.
- Pre-existing flake (not caused by this work): `tests.test_group_net.TestGradientCheckpointing.test_recompute_matches`
  intermittently fails on the stem conv bias. The gradients there are at noise level (about 1·10⁻⁹ against a tolerance
  of about 2·10⁻⁹), from GPU nondeterminism.

## Open decision for Jay

- **`--n-target 4` (recommended):** keeps today's half-resolution widths and halves only f0 and D1 (`f0d1`). The model
  stays at 29.5 M parameters.
- **N ≥ 5 at level 1** needs one of:
  - (a) running without the margin: `f0d1` LS N 5 is predicted feasible without it, and the m9 job measures N 5 in step
    2p;
  - (b) forward corner sequencing, which is backlogged (design 3.7) and would restore the `fr50` lean-losses
    alternatives at N 6;
  - (c) "int32 indices pinned to the GPU", the second Fw remedy.
- **Also open:** the over-prediction of small level-4 AMP and split points (model-form question above). It is
  conservative for selection, and the cluster job will show whether it carries over to level 1.
