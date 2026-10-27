# Design review: retrain-prereqs design v1.3

Reviewer: fresh-context design review (workflow `wf_63bfb7f7f2cfe710`), 2026-10-26, third round.
Verdict: **CHANGES_REQUESTED** (0 HIGH, 1 MEDIUM, 6 NIT).

The v1.2 MEDIUM (memory-knob defaults) and NITs 2-6 are resolved as described in section 15. I re-checked the five focus areas against the source:

- AdamW: `runtime/optimizers.py`, plus the Keras 2.10 `adamw.py` in the local `tf_gpu` env.
- Tent low-res: re-derived in numpy.
- Skip: `models/cascade.py`, `runtime/run.py`, `main.py`.
- Delayed start: `TaskSchedule`, `fit`.
- Scripts: `runtime/args.py`, `tests/gpu/check_mixed_run.py`, `slurm/m9_level1_cascade.sh`, `slurm/m10_batch_memory.sh`.

One gap remains. The rule that turns the m10 memory records into the N / DEFORM_N defaults and the `GRAD_CKPT=none` permission isn't defined well enough, and the one number it gives ignores the trainer-versus-tool memory gap that milestone 9 measured.

## Findings

### 1. MEDIUM: the "largest fitting N" rule is undefined, and the only threshold has no margin for the trainer-vs-tool gap (section 6 memory knobs, placeholders, r_level4 header; section 14.4)

**What the design says.** The orchestrator fills N / DEFORM_N with the "largest fitting N" from `scratch/m10/memory_batch.json`. "Fitting" is defined only for the `GRAD_CKPT=none` path: "peak <= 46000 MB on the RTX 8000, <= 36864 MB for dgx-a100". It isn't defined at all for the default `full` (DK) records, which are the ones that pick the level-4 and level-2 N for the whole sweep.

**Why that isn't safe.**
- 46000 is roughly the RTX 8000's entire TF budget. `slurm/m9_level1_cascade.sh` derives it as 46080 MiB usable - ~500 context - DALI - >= 1024 headroom. The m10 header itself says "~44 GiB TF (minus DALI ...)".
- The unit is not stated. `measure_level_memory` / `gpu_peak_mb` report in one unit, the TF limit is in MiB, and the project report mixes GB and GiB.
- Milestone 9 showed the real Trainer process needs much more than `measure_level_memory` reports. 1-pair level 1 measured 31 GB but OOMed in the trainer. A later OOM after the first validation pass came from fragmentation.
- At levels 4 and 2 the scripts default to `TF_MEM_LIMIT_MB 0` (memory growth), the mode in which that fragmentation OOM happened.

**Consequence.** Choosing N at ~46 GB would very likely fail only after the first validation (step > 100 in the smoke), or mid-sweep in all 13 runs.

**Fix (binding text for section 6 and the r_level4 / r_level2 headers).**
- Define for every record (DK, no-ckpt, linear): `fits(record, partition) := record.peak_mb <= BUDGET[partition][level]`.
  - `BUDGET.rtx8000 = {4: 42000, 2: 41000}` MB, in the unit of `memory_batch.json`'s peak field. The coder states that unit in the comment after checking `tests/gpu/measure_level_memory.py`.
  - These numbers are 46080 MiB - 500 context - DALI (~1000 at level 4, ~2000 at level 2 with TF-side aug) - 1024 headroom - ~1500 for the validation / fragmentation gap seen in m9.
  - `BUDGET.dgx-a100 = 34000` at every level.
- N / DEFORM_N = the largest measured N with `fits`, using the record whose knob set equals the script defaults.
- `GRAD_CKPT=none` is allowed only if the no-ckpt record `fits` at the chosen DEFORM_N.
- The smoke (which crosses at least two validations) must show `INFO gpu_peak` <= the same budget. Otherwise drop to the next measured N.
- State in the header that `TF_MEM_LIMIT_MB=0` (growth) is kept at levels 4 / 2 because DALI is small there. If the smoke OOMs after a validation, the first fallback is `TF_MEM_LIMIT_MB=<budget>`, as at level 1.
- Replace "peak <= 46000 MB" everywhere with this rule.

### 2. NIT: weight-decay shrink percentages misquoted (section 2.1)
Using sum_t lr_t ≈ 4000 lr, the pure-decay shrink is exp(-wd · 4000 lr):
- wd 1e-2 gives 0.4 % (lr 1e-4), 1.2 % (lr 3e-4) and 3.9 % (lr 1e-3).
- wd 1e-1 gives 3.9 %, 11.3 % and 33 %.

The text says "1-4 % and 11-33 %". Fix: "0.4-4 % and 4-33 % (lr 1e-4 to 1e-3)". The grid decision doesn't change.

### 3. NIT: r_level4 with a deform-only ratio emits arguments main.py rejects (sections 5.2, 6, test 11.5 h)
The task-start flags are added "when DEFORM_START > 0 and deform is in TASK_RATIO". With `TASK_RATIO=deform:1` and the default DEFORM_START 3000, the only task starts at 3000, and 5.2 makes that a parse error ("at least one task must start at 0"). Test 11.5 h runs exactly this case in dry-run mode and only asserts "no `--grad-balance-rule`", so the bug would ship with a passing test. Fix: add the task-start flags only when `HAS_DEFORM && HAS_LINEAR && DEFORM_START > 0`, and print `NOTE: deform-only ratio, DEFORM_START ignored` otherwise. Test 11.5 h then asserts no `--task-start-step` for `TASK_RATIO=deform:1`.

### 4. NIT: a sweep job that dies before creating its run dir is resubmitted on every `submit` (section 7.2)
State "not started" means "no run dir". A job that exits in r_level4.sh validation, a singularity bind error, or a node failure before `main.py` creates `<results>/<run>` leaves no run dir. It drops out of squeue and is submitted again by the next `submit` call, and so on indefinitely. Fix: a config with a `r4sw_submissions.jsonl` record for its run name that is neither in squeue nor has a run dir is **failed** ("died before start"), and is never resubmitted automatically. Add a test 11.5 c case for it.

### 5. NIT: wall-time limits for level 2 and phase C rest on unverified step times (sections 6, 7.1)
- Level 2 at N 32 / DEFORM_N 16 on an RTX 8000 has no step-time estimate. m8 needed ~2.8 s mean on an A100 at N 16 / 8, so a ~2x larger group on a ~1.5x slower card suggests ~8 s × 8000 ≈ 18 h + validation, close to the 24 h limit.
- The phase-C 16000-step run is budgeted at ~3.6 s × 16000 = 16 h + 32 validations within 24 h.

Both runs are resumable, but the sweep helper never resumes (it marks the run failed). Fix:
- r_level2 / r_level4 headers state "set --time from the smoke's `INFO step_time` projection × 1.3; if it exceeds the limit, resubmit with the same RUN_NAME".
- The helper's `--c-time` is documented as "derive from the phase-A INFO step_time".

### 6. NIT: VAL_GROUP_SIZE 16 rationale depends on the m10 result (section 7.1)
"16 is the smallest DEFORM_N in any phase" holds only if DEFORM_N_max = 32. If m10 yields 24, DEFORM_N_small = 12. Fix: VAL_GROUP_SIZE = min(16, DEFORM_N_small), fixed for the whole sweep and recorded. Alternatively, keep 16 and state that validation (no tape) fits at 16 even when training at a smaller DEFORM_N.

### 7. NIT: who skips non-finite steps (sections 1, 2.3)
"non-finite steps are skipped by the LSO (decay included)" isn't how the code works. The Trainer itself wraps `apply_gradients` in `tf.cond(finite, apply, skip)` (`runtime/run.py` ~742-755), and the LSO's own check is a second guard. The conclusion (the decay is skipped too) still holds. Fix the wording: "skipped by the Trainer's finite check (and by the LSO when called directly, test 11.1 d)".

## Verified assumptions
- **(a) AdamW and the no-op claim.**
  - `make_optimizer("adam")` builds `exp.Adam(learning_rate=lr, **common)` and never reads `weight_decay`. `--optimizer` defaults to `adam` and `--weight-decay` to 3e-5 (`runtime/args.py` 193-204). So every Adam run so far had zero weight decay: **verified**.
  - `adamw` builds `exp.AdamW(learning_rate, weight_decay, global_clipnorm, jit_compile)`, calls `exclude_from_weight_decay(var_names=["gamma","beta","bias"])`, and gets wrapped in an LSO under AMP.
  - Keras 2.10 `adamw.py` defines `exclude_from_weight_decay(var_list, var_names)` and `_use_weight_decay` (`variable in list` then `re.search(name, variable.name)`). `update_step` decays `var * wd * lr` when `_use_weight_decay`. At wd 0 that subtracts an exact 0, so it is bit-identical to Adam (11.1 a is achievable).
  - No model layer name contains gamma / beta / bias except the LayerNorm weights themselves (`models/layers.py` 462-463), so test 11.1 b's "leaf" criterion matches the full-name regex.
  - `uses_weight_decay` already supports the config record (returns False at wd 0).
  - The Trainer assigns the scheduled lr every step (`fit` ~1719-1721). Only non-None gradients reach `apply_gradients` (static filter, `run.py` 735 / 837), so an inactive deform decoder gets neither updates nor decay.
- **Slot transient (5.1).** (1-0.9^k)/sqrt(1-0.999^k) = 3.16 at k 1, 6.5 at k 10, 3.24 at k 100, 2.13 at k 250 and 1.59 at k 500. With the ramp k/250 the product is 1.3 at k 100 and 2.1 at k 250. All match the design.
- **(b) Tent math.** I reproduced 3.2 in numpy (float64 P, f32 const check):
  - Interior worst centroid shift over the exact test-11.2 c grid (n ∈ {40,56,48,160,224,192}, widths {12,16,20}, starts `linspace(2, n-w-2, 5)`, f 0.5..0.95 step 0.05) is **0.0167 voxel**, under 0.02, so the gate can pass.
  - Edge-touching (start 0, w 12) worst is 0.0625, under the informational 0.07.
  - P·1 = 1 (float32 error 6e-8) and mass is preserved, because U rows and D rows each sum to 1.
  - Today's op (`_lowres_axis`, `m = max(floor(f n + 0.5), 1)`, `c = (o+0.5) m/n - 0.5`, edge clamps) matches 3.2's U. `augment_image(x, u, seed, cfg, roi)` already computes a float64 tensor f, so a traced tent path with static n × n matrices is feasible.
  - `intensity_aug_tf` returns a bare `P.IntensityAugConfig()` today, so passing `low_res_method` changes nothing else.
- **(c) Skip semantics.**
  - Every helper the 4.4 sketches use exists with the stated semantics: `skip_linear`, `skip_fields`, `skip_to`, `prewarp_linear`, `prewarp_deform`, `compose_linear`, `pool_to_coarser`, `FrozenChain._linear` / `_deform`, `linear_stages` / `deform_stages`, `CHAIN_LEVEL`, `diffeo.upsample_displacement` / `compose_cascade`.
  - With `skip4=None` the sketches equal today's legacy level-1 branch (`run.py` 709-713, 815-818: `skip_to(skip, st["U4"], st["U"])` with `U4 = up(p4["U"])`).
  - The training deform step uses only U / U_inv, so dropping V in `deform_skip` is fine. Validation never skips.
  - Jay's 2026-10-26 semantics are already in network-design 4.7 (line 282) and match v2 "Level skipping" (line 252).
  - "Each stage usable alone" holds where the schedule needs it: the level-2 deform stage is trained with level-4 skip 0.15, and linear p4 = 0 everywhere. `main.py` already has the level-2 source's cascade record at level 1 (`src_info["upstream_cascade"]`), so `_skip_consistency` needs no extra file read.
- **(d) One run vs two-stage.**
  - `TaskSchedule.draw` is stateless (`random.Random(seed*1000003 + step)`), so the all-active draws stay identical.
  - The rationale (no optimizer reset, one cosine schedule, delayed G because G from step 0 collapsed from scratch, delayed deform as a stated precaution) is consistent with the project record.
  - The fallback is parseable, because the deform-only flags are gated on TASK_RATIO (`args.py` 475-480) and `--deform-decoder true` exists.
- **(e) Scripts.**
  - Every main.py flag used exists in `runtime/args.py`: `--num-threads`, `--val-png`, `--log-backend`, `--ckpt-every`, `--min-lr-factor`, `--gpu-memory-limit-mb`, `--intensity-aug-device`, `--lean-losses`, `--split-group-conv`, `--task-grad-norm`, `--lambda-dice-template`, `--val-image-chunk`, `--bn-recal-validate`, ...
  - Every checker flag exists in `check_mixed_run.py` (619-659), including `--grad-balance-from-step`, `--memory-budget-mb`, `--project-steps` / `--project-val-every`, `--expect-min-pairs`.
  - The accounting rules hold: account `qtim` only (hard-coded in `#SBATCH` and the helper's list-form sbatch call), 1 GPU, one node, 3 CPUs / 64G within the rtx8000 fair share, at most two active jobs counting all of `squeue -u jn85`, and `--max-active` > 2 rejected.
  - Env isolation via `SCRIPT_ENV_KEYS`, the dirty-tree refusal and `rev` recording are in.
  - TF_MEM_LIMIT_MB defaults (0 / 0 / 40000, A100 34000) match m9. The DK knob set equals `m10_batch_memory.sh` line 46.
  - New-flag defaults are bit-identical: `adam` untouched, `nn` path verbatim, skip mode off/legacy unchanged, no lr factor without starts, and the config.json `optimizer` key only for sgdw / adamw.

## Unverified / wrong assumptions
- **Wrong or missing:** the N-selection / `GRAD_CKPT=none` fit rule (finding 1).
- **Minor wording errors:**
  - wd shrink percentages (finding 2);
  - who skips non-finite steps (finding 7);
  - the deform-only task-start case (finding 3).
- **Unverified (the design states this itself):**
  - the login-node python3 version;
  - the m10 memory results and the units of their peak field;
  - step times at the new N (finding 5);
  - real-data behaviour of AdamW at wd 1e-2 / 1e-1, of the coincident step-3000 change point, and of AMP from scratch at level 4 (m8 / m9 AMP runs were all warm-started).
