# m9 level-1 deform NaN (job 10518952): investigation

## 1. Summary

The NaN starts in the forward pass of the trainable stage's deform network (encoder on the pre-warped deform input -> deform decoder -> svf_norm_act -> SVF head), before any loss, sampler or integrator code runs. The most likely cause is a float16 overflow (|x| > 65504 -> inf, then inf - inf / inf * 0 = NaN) in a deform-decoder activation under mixed_float16 with frozen BN statistics. The weights are probably still finite. I can't name the exact layer from code alone. The probe job in section 6 names the first op with inf/NaN and reads the checkpoints.

How each observed fact fits this explanation:

- Validation also gives NaN, with no augmentation, no gradients and no grad-norm in the path. `val_deform/centrality_mm` is a mean over `|mean_i v_i|` (`runtime/metrics.py` `deform_group_errors`), so the centred SVF `v` itself contains NaN. Everything downstream of `v` is therefore ruled out as the source: LNCC, Dice, smoothness, `pair_terms_recompute`, the lean sampler, scaling-and-squaring, cascade composition and the task-grad-norm.
- Hard Dice stays finite (0.88/0.82) while LNCC and inverse consistency are NaN. That means the NaN is local (few voxels), not global. A NaN weight would make every voxel NaN (every conv mixes all channels everywhere) and hard Dice would collapse. A local fp16 overflow at extreme-activation voxels gives exactly this pattern. I reproduced it locally (section 4).
- The NaN is deform-only. The deform decoder runs only for the deform task. The shared encoder sees linear inputs fine.
- It never recovers. A skipped step leaves the weights unchanged, so the same weights overflow on every later batch. The onset was intermittent (135 of ~250 deform steps in 5500-6000), and the finite steps kept moving the weights further.
- Onset came while `v_rms` grew 0.44 -> 0.66 with the Dice already flat. The deform stage was gaining magnitude without improving. With frozen BN, nothing renormalises activations when weights grow (`BatchNormWithSeparateAffine.stats_frozen`, recalibrated once at step 0, then frozen).

Two structural findings make the failure worse and permanent:

- The existing guard can't trip in a mixed-task run. `Trainer.fit` (`runtime/run.py`) keeps one `consecutive` counter that every finite step resets. Linear steps interleave with deform (1:1:2), so `--nonfinite-patience 20` never fires, even with 100 % deform failure.
- Under AMP, NaN deform steps are still passed to `LossScaleOptimizer.apply_gradients` (`tf.cond(in_finite if amp else finite, apply, ...)`). The LSO skips them but halves the shared dynamic loss scale each time. After 5500 the loss scale must have sat at its floor (1). With a growth interval of 2000 consecutive finite steps it can never recover while deform keeps failing. So linear tasks trained with loss scale 1 for 2500 steps (fp16 gradient underflow risk). This is not verified: the probe's part A prints `optimizer/loss_scale/current_loss_scale`.

Recommended fixes, ranked:

1. Per-task non-finite guard plus early per-task logging.
2. A float16 activation clamp (opt-in) or float32 deform decoder (opt-in).
3. Lower lr plus AdamW weight decay at level 1.

All three can keep the defaults bit-identical (section 5).

What I couldn't check: I did not read the cluster logs, metrics or checkpoints. The SSH call to the login node was declined, so all run numbers come from the brief. No GPU job was submitted.

## 2. Code-path audit (fp16 vs fp32 under mixed_float16)

| Path | Dtype / guard | Can it create NaN from finite inputs? |
|---|---|---|
| Encoder / decoder `GroupConvBlock` (`models/layers.py`) | float16 compute: norm -> LeakyReLU -> [group mean concat] -> Conv3D (or the split-conv path `_split_call`, also float16) | **Yes: an fp16 conv output saturates to ±inf above 65504.** The next conv mixes ±inf into NaN. |
| `BatchNormWithSeparateAffine` | Keras BN computes internally in fp32 with moving stats frozen. Output fp16 = (x-mu)/sqrt(var+1e-5) * gamma + beta. Gain per channel up to 1/sqrt(1e-5) = 316 when the recalibrated var is ~0. | **Yes: fp16 output overflow** if a frozen-stat channel now varies much more than at recalibration. |
| `group_mean_concat` / split conv mean | fp16 `reduce_mean` over N=4 | It spreads an inf/NaN at one image's voxel to all group members (confirmed in the repro: the NaN voxel count is identical for all 3 images). |
| `DeformDecoder` SVF head (`models/group_net.py`) | `svf_norm_act` fp16; `svf_head` Conv3D `dtype="float32"` (autocasts its fp16 input); `tf.cast(v_raw, tf.float32)` | No: the head is fp32, but it passes on inf from its fp16 input (inf*w + -inf*w' = NaN). |
| `center_velocities`, `integrate_svf`, `exp_and_inverse`, `compose`, `warp` (`models/diffeo.py`) | Fields float32 via `_as_field`. `v / 2^T` then T compositions. The warp is a convex combination with clipped coordinates (`sample_voxel_coords`, `_lean_linear_forward`: `clip_by_value(coords, -2, max+1)`). | No: bounded for finite v. NaN in v spreads spatially through the compositions (repro: v_raw 199k NaN voxels -> u ~350k). |
| Lean sampler custom gradient (`_lean_linear_backward`) | `dep = tf.where(always_true, 0.0, NaN)` | Never selects NaN (it is a data-dependency trick, also used by TF's own `recompute_grad`), and it is backward-only. |
| LNCC (`runtime/losses.py` `local_ncc`) | float32. var clamped >= 0, cc = cov^2/(var_a var_b + 1e-7), division by true window counts, `masked_mean` divides by max(sum w, 1) | No. A NaN input makes the whole masked mean NaN, because NaN*0 = NaN in the box sums. |
| Soft Dice (`soft_dice`, `_soft_dice_lean`) | eps 1e-5. With NaN inputs the "present" test (total > 1e-3) is False -> count 0 -> returns **1.0** | No. Side effect: soft Dice reads 1.0 on NaN batches (repro: `soft dice tmpl 1`). |
| Smoothness | squared differences, mean | No. |
| `pair_terms_recompute` | float32. The `s * z` trick uses `tf.where(s == s, 0, 1)` | No. |
| Cascade (`diffeo.compose_cascade`, `CS.prewarp_deform`) | Frozen stages unchanged (fingerprinted) and finite before 5500. Composition is bounded. | No. |
| TF intensity augmentation (`data_loading/tf_intensity_aug.py` `gamma`) | `pow((x-lo)/max(rng,1e-7), g)`; `s0/max(s1,1e-7)`; clip to [0, 1] | No. In any case the Trainer zeroes non-finite inputs (`in_finite`) and the step would log a finite loss. Validation has no augmentation. |
| Task-grad-norm (`task_grad_norm_scale`) | scale = 12/max(norm, 1e-12) only if the norm is finite and >= 1e-3, else 1 | No. It acts after the forward, and non-finite norms are never rescaled. Under Adam a per-step scalar on the gradient barely changes the update (Adam is scale-invariant up to eps). |

Ranking of the candidates against the observed pattern (deform-only, forward NaN every step after onset, val NaN with finite hard Dice, onset while v grows):

1. **fp16 overflow in the trainable deform decoder / svf_norm_act** (frozen BN, grown weights): explains everything.
2. **fp16 overflow in the encoder on deform inputs only**: possible but less likely. The linear tasks run the same encoder on similar intensities with no NaN. The probe distinguishes these.
3. **A non-finite weight in the deform decoder**: contradicted by the finite hard Dice of 0.88. Part A of the probe checks it.
4. Losses, sampler, integration, cascade, augmentation, grad-norm: excluded (table above, and validation `centrality_mm` NaN means `v` is NaN).

## 3. Why level 1 and why now (lr question, 4d)

Facts from the brief: `v_rms` 0.44 (4500) -> 0.54 (5000) -> 0.66 (5500). That is +0.10 and then +0.12 per 500 steps, accelerating, while the lr fell from 1.3e-4 to 7.5e-5 (-40 %). Over the same span the best deform validation was at 5000 (tD/pD 0.888/0.823) with only a small gain over the frozen 4+2 stages (0.875/0.799). Inverse consistency rose 0.156 -> 0.180 mm.

So the residual stage was putting on magnitude and roughness, not accuracy. Adam moves each weight by about lr per step whenever the gradient sign is consistent, regardless of gradient size. With the task-grad-norm, every deform gradient is also forced to norm 12 (scale 16-40x), so even a pure-noise plateau gradient produces a full-size Adam step.

For a rough scale, compare with the He-init std of a decoder kernel (sqrt(2/(27 * C_in)) ~ 0.02-0.03 at C_in 96-192). Integrated lr over 4500-5500 is ~1e-4 * 1000 = 0.1. That is several init-stds of possible drift per weight in 1000 steps.

With frozen BN statistics, nothing renormalises the resulting activation growth, so a multiplicative chain through d5 -> d1 can eventually cross 65504 at the most extreme voxels. Level 1 is the stage where the target residual is smallest, so most of the gradient is noise. This matches Jay's rigid-project experience: level 1 needs a smaller lr and weight decay. The numbers here make lr 3e-4 peak (still 1e-4 at the onset) a plausible trigger. They are not proof. The probe's part A relative-change table (deform_decoder vs encoder, last vs best_deform vs the L2 source) shows whether the decoder weights grew disproportionately.

Note: the `train_deform/*` window metrics after ~5500 average only the finite steps (`fit` accumulates only when `res["finite"]`). They are not representative after the onset.

## 4. Local reproduction (WSL GTX 1080, 6000 MB limit)

Tool: `.agents/tasks/m9-level1-nan/repro_fp16_overflow.py`. Setup: tiny level-2 deform model, mixed_float16, frozen BN, 3 synthetic label phantoms. The `d2` conv kernel is scaled by k (the SVF head kernel is set to N(0, 0.02) so that v ≠ 0). The same weights are also run in a float32 model.

| k | d2 max (fp16) | first non-finite | NaN voxels v_raw / v / u (per image, of 860k) | loss | ncc_tmpl | hard Dice | inv-cons | centrality | folding |
|---|---|---|---|---|---|---|---|---|---|
| 1-3e4 | 1.6 ... 4.8e4 | none | 0 | finite | finite | finite | finite | finite | finite |
| 1e5 | 6.55e4 (saturated) | **d2 conv (fp16)** | 199473 / 199473 (same in all 3 images: group mean) / ~350k | **NaN** | **NaN** | finite (0.0 here: 23 % NaN) | **NaN** | **NaN** | finite (0.007-0.016) |
| 1e5, float32 model | 1.59e5 | none | 0 | 29.9 | 0.007 | 0.06 | 9.1 | finite | finite |

Results:

- fp16 and fp32 agree to 3-4 digits up to the overflow threshold.
- Past it, the first non-finite tensor is the fp16 conv output. It then spreads through d1, svf_norm_act and v_raw, and the group-mean concat and centering spread it to every group member.
- The metric signature is the same as job 10518952: loss NaN, LNCC / inverse consistency / centrality NaN, folding finite, hard Dice finite. Hard Dice stays finite because NaN coordinates in the nearest warp sample the fill label. Folding stays finite because `jac <= 0` is False for NaN.
- The real run's hard Dice of 0.88 means its NaN region is far smaller than this synthetic 23 %.
- The float32 model never produces NaN.

What the repro shows is the mechanism and the signature. It does not show the real trigger layer (k = 1e5 is artificial).

## 5. Fixes (ranked), risk, and golden bit-identity

Golden tests (`tests/fixtures/m7_linear_golden.json`, `lncc_deform_golden.json`, fp32, finite) must stay at max |diff| 0. Every item below keeps its defaults off, or touches only code paths that run on non-finite steps, so goldens are unaffected.

**(b) Per-task non-finite guard. Do it first; low risk, prevents 30 h of wasted GPU time.**

- In `Trainer.fit`, replace the single `consecutive` counter with `consecutive[task]`, reset only by a finite step of the same task.
- Add `--nonfinite-patience-task K` (default e.g. 20 deform steps). When K is exceeded, raise `RuntimeError("task deform: K consecutive non-finite steps (first at step s0) ...")` and do not save `ckpt/last`. Also guard `save_last` so it never writes while any task is in a non-finite streak > a small threshold, so the last checkpoint on disk is pre-failure.
- Recommended behaviour: abort, not auto-recover. Automatic restore + lr reduction needs a "last good" snapshot of model + optimizer + LSO + RNG states. Today `ckpt/last` is overwritten every 250 steps, so the 8000 checkpoint already holds the bad weights. Restore also interacts with the shared optimizer of the other tasks. If wanted later: keep a `ckpt/good` manager saved only when no task had a non-finite step in the last window, restore it on trip, multiply that task's lr by 0.5, and allow at most 2 recoveries.
- Bit-identical: yes for every finite run (the guard only acts on non-finite steps). Behaviour change: mixed-task runs like 10518952 abort at ~step 5520 instead of running to 8000.
- Also stop calling `apply_gradients` with non-finite gradients for repeated failures of one task. Otherwise the LSO keeps halving the shared loss scale and starves the healthy linear tasks. For example, for a task already in a streak (>= 2 consecutive), skip the LSO call and skip the scale update. Bit-identical for finite runs, since early isolated AMP overflows still reach the LSO as today.

**(c) Early per-task non-finite logging / checker. Low risk.**

- In the periodic log, add `train_<task>/nonfinite_steps` (count in the window) and `nonfinite_frac`, plus a stderr WARNING when a task's window fraction is > 0.1.
- In `tests/gpu/check_mixed_run.py`, check the rate per task (e.g. deform non-finite in any 500-step window > 5 % -> FAIL with the first step) instead of only `--max-nonfinite-frac` over the whole run.
- Log the LSO `current_loss_scale` every window (that is how the starvation above would show).
- Bit-identical: adds log keys only. Golden tests don't compare log keys; check `test_trainer` expectations of exact key sets.

**(a) Numerical root-cause fix. Choose after the probe (section 6).**

- **a1 `--fp16-act-clip C`** (default 0 = off). In `GroupConvBlock.call` / `_split_call` and `BatchNormWithSeparateAffine.call`, when the compute dtype is float16: `y = tf.clip_by_value(y, -C, C)` with C = 3e4 (scope: deform decoder + svf_norm_act; optionally the encoder).
  - Cost: negligible memory and time (elementwise; under the recompute blocks).
  - Risk: saturated voxels get zero gradient and wrong but finite features, so it treats the symptom. Combine with a2/(d).
  - Bit-identical when off: no op is added.
- **a2 `--deform-decoder-dtype float32`** (Keras `dtype="float32"` policy on `DeformDecoder` and its blocks; skips are cast).
  - This removes the overflow class entirely.
  - Cost: about 2x decoder activation memory. At level 1 that is a few GB on top of 36.7-38.5 GB, so it will not fit the 40000 MB limit with 4 pairs. It is fine at levels 4/2 for the retrain.
  - Bit-identical when off.
- **a3** If the probe shows the first inf at a BN output with a near-zero recalibrated variance: larger `bn_epsilon` for the recalibration (e.g. 1e-3, a ModelConfig field, so it changes the config and is for new runs only), or more recalibration batches. Not bit-identical for runs that opt in.

**(d) Level-1 learning rate / weight decay. Needed anyway (Jay); medium risk only in tuning cost.**

- Short warm-started level-1 runs (from `ckpt/best_deform` 5000 or a fresh level-1 cascade start) with AdamW, lr in {1e-4, 3e-5, 1e-5} x wd in {0, 1e-2}, 2000 steps, comparing deform pD and `v_rms` growth.
- Expected: `v_rms` flat or slowly growing and no NaN at lr <= 3e-5.
- Weight decay bounds kernel growth, which matters precisely because BN is frozen.
- Defaults unchanged (flags exist: `--optimizer adamw`, `--weight-decay`, `--lr`).

Order: (b)+(c) code change -> probe job (section 6) -> a1 or a3 per the probe -> (d) sweep with the guard on.

## 6. Diagnostic job (for the orchestrator, after Jay's approval)

Files (new, under this directory):

- `ckpt_stats.py`: numpy + `tf.train.load_checkpoint` only (no model build, CPU).
  - Per-variable non-finite count, max|w|, rms, and relative change vs reference checkpoints.
  - BN moving-variance minima with the implied frozen gain.
  - Largest Adam slots and non-model scalars (incl. `optimizer/loss_scale/current_loss_scale`).
  - Tested locally on a saved tiny LSO checkpoint with an injected NaN bias (found it).
- `probe_activations.py`: wraps `main.main` unchanged.
  - `PROBE_ACT=1` prints `ACT <scope> conv|bn_in|bn_out|v_raw max .. min ..` per block for every forward / recompute (reduce_max/min only, no fp32 copies).
  - `PROBE_NUMERICS=1` calls `tf.debugging.enable_check_numerics()` at `Trainer.fit` start, so the first inf/NaN op raises with its name, dtype and creation stack.
  - It patches the two always-unselected NaN constants (lean-sampler backward and TF's `recompute_grad` `dresult_dep`) to 1.0 so they don't trip the check, and it disables `save_last`.
  - Tested locally on the tiny overfit path (AMP, full checkpointing, lean sampler, split conv): 2 deform steps passed under the numerics check, and the ACT lines were printed.
- `m9_nan_probe.sh`: SLURM job.
  - Extracts the run's code (`git archive e6815d9`) into `scratch/m9_nan/src_e6815d9`.
  - Runs part A, then resumes scratch copies of the run dir (config.json + control/ + one checkpoint as ckpt/last) for 3 deform-only steps:
    - B: step 8000 with ACT + numerics.
    - C: `ckpt/best_deform` step 5000 with ACT only, which gives the fp16 headroom before the onset.
  - Overrides: `--task-ratio deform:1 --min-pairs 1 --lean-losses true --val-every 0 --ckpt-every 0`.
  - It writes only under `scratch/m9_nan/<job>/`. The real run dir is only read.

Resources: rtx8000, qtim, 1 GPU, 3 CPUs, 64G, 1 h. The estimate is ~3 min for A plus 2 x ~12-15 min (L1 startup, trace, 3 deform steps of ~23 s). It is not verified on the cluster.

Commands (after commit + push of this directory; check `squeue -u jn85` shows < 2 active jobs):

```bash
cd /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad && git pull --ff-only
mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
sbatch .agents/tasks/m9-level1-nan/m9_nan_probe.sh
# if B/C stop at a resume guard about the task ratio:
TASK_RATIO=rigid:1,affine:1,deform:2 sbatch .agents/tasks/m9-level1-nan/m9_nan_probe.sh
```

Outputs: `logs/m9nanprobe-<job>.out` (summary: part A head, the first ACT lines with inf/nan, deform-decoder ACT lines, the numerics error block) and `scratch/m9_nan/<job>/{ckpt_stats.json,ckpt_stats.txt,last.log,bd5000.log}`.

How to read it:

- First ACT line with `inf` in the forward order (stem, l2..l6, d5..d1, svf_norm_act, v_raw) plus the `Detected Infinity or NaN in output 0 of graph op "<type>"` line: the overflow layer and dtype (`half` confirms fp16).
- Part A `non-finite variables in last: 0` confirms finite weights.
- The `[deform_decoder]` relative-change table vs `best_deform` / `l2_source` and the BN gain list show which weights grew and whether a near-zero recalibrated variance is involved (-> a3).
- `bd5000` ACT maxima vs 65504 give the headroom that a1's clip value C must respect.

Known risks of the probe:

- `--task-ratio deform:1` may hit a resume guard (fallback above).
- The numerics check adds one CheckNumerics op per tensor. With 1 pair and lean losses there should be ~9 GB of headroom under the 40000 MB limit.
- The resumed LSO loss scale is whatever the checkpoint holds (probably 1), so a legitimate backward fp16 overflow is unlikely to raise first. If the raise names a `gradients/...` or `Backprop` op, the forward was finite on that batch: read the ACT lines instead.
