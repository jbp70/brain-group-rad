# nonfinite-fixes: report

The per-task non-finite guard, ckpt/last protection, LSO skip, window logs, checker window check and the three opt-in float16 mitigations (a1 clip, a2 float32 deform decoder, a3 BN epsilon override) are implemented as in `plan.md` (decisions D1-D12). All new behaviour is off or inert for finite runs. Both goldens print `max |diff| 0`. Nothing was committed or pushed, and the cluster was not touched.

## Files changed

| File | Change |
|---|---|
| `runtime/run.py` | `TrainConfig.nonfinite_patience_task` (20). Constants `NONFINITE_WINDOW_KEYS`, `LOSS_SCALE_KEY`, `NONFINITE_WARN_FRAC`, `LSO_SKIP_STREAK`, `SAVE_LAST_MAX_STREAK`. `Trainer._lso_skip` (AMP only), `_apply_gate` used by both step builders, `train_step(..., lso_skip=False)` and the `lso_skipped` result. `fit` has per-task streaks, both guards with the `nonfinite_abort` event and flush, deferred ckpt/last saves, window counters, the new `nonfinite_step_skipped` fields, `_nonfinite_window_metrics` and `_nonfinite_abort`. The module docstring has a "Non-finite guard" section (incl. the future restore mode, not implemented). |
| `runtime/args.py` | `--nonfinite-patience-task`, `--fp16-act-clip`, `--fp16-act-clip-scope`, `--deform-decoder-dtype`, `--bn-epsilon-override`, validation, the `TrainConfig` mapping and a docstring bullet. |
| `main.py` | Passes the four model kwargs to `GroupRegLevelModel`. |
| `models/precision.py` | `policy_scope(name)` context manager. |
| `models/cascade.py` | `_float32_policy()` delegates to `precision.policy_scope("float32")`. The unused `contextlib` import was removed. |
| `models/layers.py` | `clip_fp16`, `set_fp16_act_clip` and `set_bn_epsilon`. A `fp16_act_clip` attribute (0) on `GroupConvBlock` (standard path, both split-conv terms and the sum) and on `BatchNormWithSeparateAffine` (after the affine). |
| `models/group_net.py` | `FP16_ACT_CLIP_SCOPES`, `FP16_ACT_CLIP_MAX` 32000, `DEFORM_DECODER_DTYPES`. `GroupRegLevelModel(..., fp16_act_clip, fp16_act_clip_scope, deform_decoder_dtype, bn_epsilon_override)` with validation; the decoder is built inside the float32 policy scope for 'float32'. `DeformDecoder.call` casts `skips[5]` to its compute dtype (a no-op when the dtypes match). Docstring updated. |
| `tests/gpu/check_mixed_run.py` | `--max-window-nonfinite-frac` (0 = off), `--nonfinite-window-steps` 500, `--nonfinite-from-step` 100, `_check_nonfinite_windows`, the `_loss_scale_info` INFO line, docstring. |
| `slurm/r_level4.sh`, `r_level2.sh`, `r_level1.sh`, `m9_level1_cascade.sh` | Env `FP16_ACT_CLIP`, `FP16_ACT_CLIP_SCOPE`, `DEFORM_DECODER_DTYPE`, `BN_EPSILON_OVERRIDE` (validated, no flag emitted at the default, placed before `$EXTRA_ARGS`), an echo line, a header block, and the checker call `--max-window-nonfinite-frac 0.05`. r_level4/2/1 drop deform-scoped settings with a NOTE for linear-only ratios. |
| `slurm/sweep_level4.py` | The four env names added to `SCRIPT_ENV_KEYS`. |
| Tests | New `tests/test_fp16_mitigations.py`. New `TestNonfiniteGuard` in `tests/test_trainer.py`. New `TestNonfiniteWindows` and a `make_run(train_fn=...)` hook in `tests/test_check_mixed_run.py`. `test_nonfinite_flags` in `tests/test_runtime_infra.py`. `test_fp16_mitigation_flags_reach_model` in `tests/test_main.py`. Four mitigation tests in `tests/test_sweep_tools.py`. In `tests/test_slurm_scripts.py`: an m9 override test, the default-recipe assertions, and `RECIPE_ENV`. |

## Guard semantics and defaults

- **Global guard (unchanged):** `--nonfinite-patience` (20) counts consecutive non-finite steps of any task and is checked first, so single-task runs keep the old message.
- **Per-task guard:** `--nonfinite-patience-task K` (default 20, 0 = off). Each task has its own streak, which only a finite step of that same task resets. The guard raises on the (K+1)-th consecutive non-finite step of one task. Message: `task deform: 21 consecutive non-finite steps of this task (first at step s0, last at step s1; finite steps of other tasks do not reset this count); stopping. ckpt/last was not written during the streak (last save at step S); restart from ckpt/last or ckpt/best*, e.g. with a lower --lr, --fp16-act-clip or --deform-decoder-dtype float32`.
- **Abort logging:** before either raise, a `nonfinite_abort` event is logged (`task`, `streak`, `first_step`, `last_step`, `last_ckpt_step`) and the logs are flushed.
- **ckpt/last protection:** `save_last` is not called while any task has a streak >= `SAVE_LAST_MAX_STREAK` = 2 of hard failures. A single non-finite step changes no weights. AMP loss-scale warm-up overflows (finite loss, non-finite gradient norm) do not count towards this save streak at all; they still count for the abort guards, as before. Without this exemption the full suite failed `tests.test_main` T-E1: a 3-step level-1 AMP run had a warm-up streak at its last step, so the final ckpt/last was skipped and the resume repeated the BN recalibration. The deviation from the plan's D3 is deliberate: the warm-up never delays ckpt/last.
  - A blocked due save logs `ckpt_last_deferred` (with the open streaks) and runs at the first later step that has no such streak.
  - A blocked final save logs `ckpt_last_skipped_final` instead.
  - Caveat: "pre-failure" means only that no streak of 2 or more was open at save time. If failures start intermittently, the weights just before a streak may already be close to failing, so `ckpt/best_deform` is the safer restart point. `ckpt/best*` saving is unchanged.
- **LSO skip (AMP only):** a step is not passed to the LossScaleOptimizer when the previous step of the same task was non-finite AND this step's forward loss is non-finite. Skipped means no scale halving, no counter reset and no `iterations` increment. Two cases still reach the LSO exactly as today:
  - gradient-only overflows (finite loss), which are the normal loss-scale warm-up;
  - isolated forward NaNs.

  The flag is a captured `tf.Variable(bool)` per task, `Trainer._lso_skip`. It is not part of any checkpoint or of the BN backup, and nothing re-traces (trace counts stay at 1).
- **Streak scope:** streaks and the pending-save flag live inside one `fit()` call. A resume starts with empty streaks, which is fine because the ckpt/last it resumes from is pre-streak.
- **Window logs:** each log record carries `train_<task>/{window_steps, nonfinite_steps, nonfinite_frac}` for every task with a step in the window, plus `optimizer/loss_scale` under an LSO. A stderr `WARNING step N: task T non-finite in b of n steps of the last log window (p %)` is printed when the fraction is > 0.1. `nonfinite_step_skipped` gains `task_streak`, and `lso_skipped` under AMP.
- **Not implemented (documented):** an optional "restore last good + lower lr" mode. It would keep a `ckpt/good` manager, saved only when no task had a non-finite step in the last window, restore model + optimizer + LSO + RNG on a trip, halve the failing task's lr, and allow at most 2 recoveries.

## New flags and env overrides (all default off)

| Flag | Default | Rules | Env (scripts) |
|---|---|---|---|
| `--nonfinite-patience-task K` | 20 | K >= 0 | not exposed (`EXTRA_ARGS`) |
| `--fp16-act-clip C` | 0 | 0 or (0, 32000]; > 0 needs `--amp true`; with scope deform_decoder needs 'deform' in the ratio | `FP16_ACT_CLIP` (0) |
| `--fp16-act-clip-scope` | deform_decoder | deform_decoder (d5..d1, svf_norm_act and their BN; linear tasks unchanged) or all (also the encoder and linear decoder) | `FP16_ACT_CLIP_SCOPE` |
| `--deform-decoder-dtype` | policy | policy / float16 / float32; non-policy needs `--amp true`; float32 needs 'deform' | `DEFORM_DECODER_DTYPE` (policy) |
| `--bn-epsilon-override E` | 0 | finite >= 0; BN of the trainable model only; pass it again on resume | `BN_EPSILON_OVERRIDE` (0) |
| checker `--max-window-nonfinite-frac F` | 0 (off) | scripts pass 0.05, with `--nonfinite-window-steps` 500 and `--nonfinite-from-step` 100 | in all four scripts |

Implementation notes:

- **a1 (clip):** clips only float16 tensors (a trace-time check, so nothing is added under float32). Clipped voxels get zero gradient, so this treats the symptom. Two clipped split-conv terms sum to at most 64000 < 65504, hence the 32000 cap.
- **a2 (float32 decoder):** constructs `DeformDecoder` and all its sublayers under a float32 policy; the encoder stays float16. Variables are float32 with the same names and order in every mode. Checkpoint round trip both ways, `copy_weights_from`, and an AMP Trainer step with `--gradient-checkpointing full` + lean sampler + split group conv are tested. Warm start and cascade init use the same variable list.
- **a3 (BN epsilon):** done as a runtime override, not as a `ModelConfig` change. This deviates from the brief's "config field" wording. `ModelConfig.bn_epsilon` (1e-5) is shared with LayerNorm and InstanceNorm and is compared by warm start and `--cascade-from`, so changing it would block exactly the level-2/1 runs where a near-zero recalibrated variance occurs. Frozen cascade stages are never touched.

## Verification

Python 3.8 (`bgr_preproc38`, Python 3.8.20) via `.agents/tmp/nf_py38.sh`:
- py_compile OK.
- ast feature_version (3, 8) OK on 16 files.
- sweep_level4.py py3.6 OK.
- runtime.args stays TF-free.
- `main.py --help` OK.
- `bash -n` OK on r_level4.sh, r_level2.sh, r_level1.sh and m9_level1_cascade.sh.
- `tests.test_slurm_scripts` + `tests.test_sweep_tools`: 65 OK.

Targeted TF runs (`tf_gpu`, WSL GTX 1080, `BGR_TF_GPU_LIMIT_MB=6000`) via `.agents/tmp/nf_tf.sh`:

| Run | Result |
|---|---|
| `tests.test_fp16_mitigations` | 10 OK (after replacing a wrong "no clip op anywhere" assertion: the sampler already clips coordinates, so the test now counts clip ops inside `deform_decoder` and compares op-type lists) |
| `tests.test_check_mixed_run` | 37 OK (6 new) |
| `tests.test_trainer.TestNonfiniteGuard` | 6 OK (the gradient-overflow twin test needed `nonfinite_patience` 60: initial scale 2^40 takes > 20 consecutive halvings) |
| `tests.test_runtime_infra.TestArgs` | 17 OK |
| `tests.test_main.TestDeformMain.test_fp16_mitigation_flags_reach_model` | OK |
| `tests.test_trainer.TestTrainStep`, `TestResumeAndValidation` | 4 + 5 OK. Existing NaN-guard and AMP tests unchanged: `test_nan_guard_skips_update`, `test_mixed_precision_smoke` (iterations 21), `test_mixed_precision_overflow_skips` |
| Goldens | `TestLinearOnlyUnchanged`: `[golden] max |diff| 0 (tolerance 0)`. `TestDeformUnchanged`: `[golden deform] max |diff| 0 (tolerance 0)` |

What the new tests show:

- **Per-task guard:** in a rigid:1,deform:1 run with NaN deform inputs and finite rigid steps inside the streak, `nonfinite_patience_task=3` raises on the 4th deform failure. The message gives the first and last step, and the `nonfinite_abort` event is written. With K = 0, `fit(30)` does not raise.
- **ckpt/last:**
  - No `save_last` happens at a step with an open streak >= 2.
  - The `ckpt_last_deferred` events appear and their save runs at the first unblocked, off-cadence step.
  - The regular cadence is unchanged.
  - In the raising variant, the last save is before the 2nd failing deform step, and `latest_last` matches it.
- **LSO:** with an initial scale of 16 and 5 forward-NaN deform steps (NaN in a d1 kernel), the scale is exactly 8 (halved once).
  - `lso_skipped` reads [F, T, T, T, T] across the 5 failures, and F on the next, finite deform step.
  - Rigid steps are finite and applied.
  - `iterations` = rigid steps + 2.
  - Trace counts are 1.
  - Control: the same step with `lso_skip=False` halves every time; with `True` it never does.
- **Unchanged behaviour:** gradient-only overflows give the same per-step loss-scale sequence, finite flags, `iterations` and bit-identical weights in `fit` as in plain `train_step` calls (op determinism on). An isolated forward-NaN deform step halves the scale once.
- **Window logs:** the counts are exact per 4-step window, the stderr WARNINGs match the windows above 10 %, there is no `optimizer/loss_scale` without AMP, and an all-finite run logs 0 with no warning.
- **Checker:** 33 % non-finite in window 501-1000 FAILS, naming the first step 525.
  - PASS cases: the same rate before step 100; an isolated 0.4 %; a trailing bucket of 12 steps (not judged); old runs without the keys ("skipped").
  - With F = 0 no window results appear.
  - The loss-scale INFO / WARN text is printed and is never a failure.
  - CLI exit codes 0/1 and the flag validation work.
- **a1:**
  - Precondition: the repro model (level 2 tiny, frozen BN, d2 kernel x 1e5) gives non-finite `v_raw` and loss.
  - With `fp16_act_clip=30000`, `v_raw`, `v`, `u`, `u_inv` and the loss are finite, with and without split group conv.
  - Without overflow, the clip changes nothing (bit-equal `v_raw`).
- **a2:**
  - Every decoder sublayer computes in float32 and the encoder in float16.
  - `v_raw` matches the float32-policy model within 2e-2 * max|v32|.
  - The x 1e5 case stays finite.
  - 3 finite AMP deform steps run with full checkpointing + lean sampler + split group conv, traced once.
- **a3:** the override sets every BN epsilon, leaves LayerNorm and `config.to_dict()` unchanged, and changes the output when a moving variance is 1e-6.

Full suite: see the section "Full suite" below. The gradient-overflow twin test also asserts that warm-up overflows never block ckpt/last (`ckpt_every=1`: saves at every step 1..45).

## Memory cost of a2 (estimate, not measured)

a2 doubles the bytes of the activations held by the deform decoder. The extra memory is about the decoder's float16 activation footprint, dominated by the d1 stage at full level resolution. For scale: the level-1 OOM tensor `[4,160,224,192,64]` half (d1 backward) is 3.3 GiB, and the d1 block holds roughly 3-5 tensors of that size. That puts level-1 decoder activations at ~10-16 GiB in float16, so a2 adds ~10-16 GiB at level 1. Other levels scale with the deform voxels per step:

| Level | Deform batch | Voxels vs L1 | Measured peak (fp16 decoder) | Est. a2 extra | Est. a2 peak | Fits? |
|---|---|---|---|---|---|---|
| 4 | N 64 | 0.25x | 27.6 GB | ~2.5-4 GiB | ~30-32 GB | RTX 8000 budget 42000 MiB yes; A100 34000 MiB borderline |
| 2 | N 16 | 0.5x | 29.9 GB | ~5-8 GiB | ~35-38 GB | RTX 8000 (41000 MiB) probably; A100 no |
| 1 | N 4, 4 pairs | 1x | ~37-38 GB under the 40000 MB TF limit | ~10-16 GiB | > 47 GB | no |

These are upper-side estimates: they assume the step peak falls inside the decoder recompute/backward. A memory job (`measure_level_memory.py` with `--deform-decoder-dtype float32` support, not added here) or a short smoke would be needed before relying on them.

## Which mitigation to enable per level (pending the probe, job 10524598)

- **Probe part A:** no non-finite weights; the loss scale stuck at 1 (LSO starvation, which the LSO skip fixes); the minimum BN variance is 0.0027-0.0033, i.e. a frozen-BN gain of ~17-19. That gain is not extreme, so a3 is unlikely to be the fix unless parts B/C name a BN output.
- **Level 1:**
  - first choice: `FP16_ACT_CLIP=30000` (scope deform_decoder) together with the lower lr / AdamW sweep;
  - a2: does not fit;
  - a3 only if the probe's first inf is a BN output;
  - scope all only if the probe names an encoder layer.
- **Levels 2 and 4 (retrain):** a2 (`DEFORM_DECODER_DTYPE=float32`) removes the overflow class and probably fits the RTX 8000 budgets (unverified), so it is the cleanest option there if the probe shows a decoder overflow. a1 is the zero-cost alternative. Level 4 trains BN normally (no frozen recalibrated statistics), so it is the least exposed.
- **Defaults:** keep everything off until the probe's parts B/C name the first non-finite op. The per-task guard and checker window check are on in all scripts already.
- **Expected AMP noise:** with the delayed deform start at 3000 (r_level4.sh), the deform task's first steps may trigger a few gradient overflows after step 100. These are short, consecutive, finite-loss halvings, normally well under 5 % of a 500-step window (~250 deform steps). If the checker flags it, raise `--nonfinite-from-step` in that call.

## Full suite

Command: `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/tests/run_tests_wsl.sh`. Log: `.agents/tmp/nf_suite.log`, deleted after reading.

**Stage 1 (py3.8):** all OK, TF-free tests 115 OK.

**Stage 2 (tf_gpu):** 677 tests (4 skipped) in 15171 s. Three invocations failed:

1. **`tests.test_main` T-E1** (`test_level1_chain_resume_and_checker`): a real bug in my D3, where the AMP warm-up blocked the final ckpt/last. Fixed with the save-streak exemption above.
2. **`tests.test_trainer`** (8 TaskGradNorm errors, `tf___task_weight() takes 2 positional arguments`): an artifact. I edited `runtime/run.py` while that module was running, so autograph re-read the shifted source and converted the wrong function. Two `test_trainer` processes also shared the GPU for ~15 min because my rerun overlapped the suite (stopped).
3. **`tests.test_pair_recompute.TestPairRecomputeMemory`** (peak growth 40 MB vs bound 2.9 MB): it passes when rerun alone on the final code (below). I count it as an environment flake in the suite run; it does not touch any changed code path.

**Reruns on the final code**, sequential, nothing else on the GPU, no edits while running (`.agents/tmp/nf_rerun.sh`, same env as the suite):

| Invocation | Result |
|---|---|
| `tests.test_pair_recompute.TestPairRecomputeMemory` | 1 OK |
| `tests.test_trainer` (`BGR_DEFER_MEMORY_TESTS=1`) | 67 OK; `[golden] max |diff| 0`, `[golden deform] max |diff| 0`, grad-norm equivalence A0 vs B max |diff| 0 |
| `tests.test_main` | 40 OK |
| `nf_py38.sh` (after the last edit) | all OK |

Every other module passed in the suite run with the final code.

Totals: 677 tests / 4 skipped, against a baseline of 647 / 4 skipped. That is 30 more; the new tests I wrote account for 29 (fp16 10, TestNonfiniteGuard 6, checker 6, args 1, main 1, slurm 1, sweep 4). All pass after the fix and the reruns. I did not rerun the whole suite a second time, since only the three modules above were affected.

## Probe note (job 10524598, received during the work)

The first non-finite op at level-1 ckpt/last is the float16 AddV2 of the two split-conv terms of d1 (per-image term + group-mean term). This is exactly the sum that a1 clips: each term is clipped to [-C, C] before the add, and the sum is clipped again. With C <= 32000 the add stays <= 64000 < 65504, and the repro test covers `split_group_conv=True`.

The probe also measured d2 conv max 9640 and d1 bn_out 6932 at step 8000, growing from step 5000. a1 therefore keeps the forward finite, but saturated voxels get zero gradient. a2 (float32 decoder) removes the overflow class, but does not fit level 1 at 4 pairs per the estimate above. Jay's later scope addition (float32 only for d1/d2, `float32_fine`, per-block widths, `slurm/m11_fp32_width_memory.sh`) is NOT part of this step and needs its own plan item.
