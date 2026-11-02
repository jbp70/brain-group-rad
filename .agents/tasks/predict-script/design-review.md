# Design review v5: standalone groupwise prediction script (design v5, 2026-11-01)

Reviewer: design-review step (fourth revision). Code re-read for this round:
`models/layers.py` (`_PreActivation._pre`, `group_mean_concat`, `GroupConvBlock.call` /
`_split_call`, `set_split_group_conv`), `models/cascade.py` (`pool_to_coarser`,
`FrozenStage.linear` / `deform`, `FrozenChain._linear` / `_deform`, `prewarp_linear`,
`prewarp_deform`, `compose_linear`), `preprocessing/grid.py` (`build_target_grid` output keys,
`Target`, `load_target_grid`), `preprocessing/config.py` (`target_paths`, `grid_tolerance_mm`),
`preprocessing/config_mindboggle.yaml` (`target_dir`). The v4 review's verified items (branch
chaining, encoder / decoder wiring, canonical mapping, BN / precision, checkpoint and evaluator
APIs, int32 bounds) were not invalidated by the v5 edits; I re-checked the parts the edits touch.

Verdict: **APPROVED** (0 HIGH, 0 MEDIUM, 2 NIT).

## Status of the v4 findings

| v4 | Topic | Status |
|---|---|---|
| 1 MEDIUM | cluster gate streams only at level 1 | Closed. 4.6 adds check-streaming's own defaults `min(cap_l, max(1, n - 1))` (pure `resolve_check_chunks`, unit-tested: N 4 -> (3, 3, 3)); section 9 passes `--chunk-l4 3 --chunk-l2 3 --chunk-l1 3` explicitly; 3 + 1 at every level exercises full and remainder traces; the JSON records `chunks` with source; an end-to-end test asserts the chunk boundaries [(0, 2), (2, 3)] at N 3. The OOM fallback (N_CHECK 3 with chunks 2 / 2 / 2) keeps two chunks per level. |
| 2 MEDIUM | only gate uses provisional fp16 bounds | Closed. Step 1a gates at `--precision float32` with the float32 bound set and `|| exit $?`; 1b (fp16) and 1c (affine+deform, float32) are non-blocking with `|| echo "WARNING ... rc $?"` (in bash `$?` there is the failed command's status, so the message is right); section 8 states the fp16 bounds never gate the cluster job; `gpu_peak_mb` of both sides is recorded; time limit 5 h; AC 7 updated; slurm static test asserts all of it. |
| 3 NIT | x_in stored vs recomputed | Closed. 4.3 states one rule (computed once in P0, stored float32, read by P1-P11, freed after P11 / P6); the section 6 total includes it (~40 GB deform only, ~43 GB affine+deform, ~63 GB float32 trainable stage, still RAM at 0.7 x 96 GB). |
| 4 NIT | which fields test (b) compares | Closed. AC 2 and 10 (b): level 2 compares the internal level-2 u / u_inv and A; level 1 additionally compares the written `_fwd` / `_inv` files (mm / spacing) with the trainer's u / u_inv; linear branches compare A; `run_predict(args, configure_device=False)` for in-process calls. |
| 5 NIT | LR-mirrored grid accepted | Closed. Optional `--grid-reference` (json via `json.load`, key `affine` required, `shape` optional; or a NIfTI), 1e-3 tolerance, exit 2 with both affines, INFO when absent, recorded in both JSONs, passed by the SLURM script on every call; unit tests for json, NIfTI, mirrored affine, absent flag. |

## Findings

### 1. NIT: which subjects check-streaming uses is not stated

Where: F6, section 7 (`check-streaming ... [--n-subjects 4] [--subjects-file F]`), section 9
step 1.

The design says "a small group of real subjects" but not which. With 101 subjects the coder could
take the first n, a random draw, or require a subjects file; a random draw would make the cluster
numbers irreproducible. Fix: "check-streaming uses the first `--n-subjects` subjects in
`find_subjects` order (after `--subjects-file`, if given); fewer available -> exit 2." The JSON
already records `subjects`.

### 2. NIT: say what to do when the float32 gate fails on real weights

Where: section 8 (float32 bounds: field max 1e-3 mm, moved 1e-3, dA_lin 1e-5, dA_t 1e-4 mm),
section 9 step 1a (`|| exit $?`).

The float32 bounds are reasoned but not yet measured on the level-1 checkpoint (three levels,
split vs concat rounding in the frozen stages, cuDNN algorithm choice per chunk size, 7
squarings). The gate is now correct as a gate. If it trips, the deform-only prediction does not
run in that job. Fix: one sentence in the script header and the report template: "on exit 4,
read `check_streaming_f32.json`. Differences that are localised or grow with field magnitude
point to an engine bug (fix the code). Uniform differences slightly above the bound point to
rounding: resubmit with `--tol-*` overrides chosen from the measured numbers and record them in
the report. Never drop the gate." No code change.

## Verified assumptions

- Branch chaining, deform only included (re-read for this round). `FrozenStage.deform` =
  pool -> level-4 model (`training=False`) -> `upsample_displacement` of u, u_inv, v.
  `FrozenChain._deform` = pool -> `stage4.deform(x2)` -> `prewarp_deform(x2, U)` (=
  `diffeo.warp(float32 x, U)`) -> model2 -> `compose_cascade` at level 2 -> upsample. The linear
  chain (`_linear`: `stage4.linear` -> `prewarp_linear(x2, A4, 2 mm)` -> model2 `A_raw` ->
  `compose_linear(A_raw, A4, model2.centering)`) never feeds the deform path. This matches 2.1,
  3.4, 4.3 (x_in rule) and 4.4. Deform only is the trained configuration. affine+deform is the
  only sequential combination, and it gives the deform branch affine-aligned inputs, as in
  training.
- The split form used by the engine applies to unsplit frozen blocks as well. `_split_call` slices
  the single `conv.kernel` at `channels` (`k[..., :C, :]`, `k[..., C:, :]`) with the layer's strides
  and bias. The concat order in `group_mean_concat` is [x, mean], so `conv_with_group_mean` on a
  block built without `split_group` computes the same function. The difference is rounding,
  which 2.3 source (ii) lists. The mean enters as a [1, ...] tensor, so a per-chunk call with the
  full-group mean is the model's op.
- Group interactions: `_pre(concat=True)` adds the group mean only through `group_mean_concat`
  (`reduce_mean` over axis 0). Norm and act are per image in inference. The 2.3 table stays
  complete for `block='single'`.
- Grid reference: `build_target_grid` writes `target_grid.json` with top-level `shape` and
  `affine`. `config_mindboggle.yaml` `target_dir` = `$PROJ/data/target_grid`, which matches the
  SLURM default `GRID_REF`. `preprocessing.grid.Target` refuses to run without that file, so the
  Mindboggle preprocessing that succeeded implies it exists. The 1e-3 tolerance equals
  `grid_tolerance_mm`. float32 sform rounding (~1e-5 mm at 100 mm offsets) is far below it.
- Chunk arithmetic: N 4 with chunk 3 gives (0, 3), (3, 4) at each level. The level-1 streamed cap
  is 4 (440 M x 4 < 2^31). N_CHECK 4 equals the unchunked n_max 4 for deform and is below 9 for
  linear, so steps 1a-1c pass the loader bound.
- SLURM step semantics: `cmd || exit $?` propagates cmd's exit code (4 for the gate). In
  `|| echo "... rc $? ..."`, `$?` is the failed command's code.
- Memory: the section 6 arithmetic is consistent. Activation stores are ~21 GB fp16 and ~42 GB
  float32, v_raw is float32 in both, so ~40 -> ~61-63 GB.

## Unverified / wrong assumptions

- Unverified: that the float32 bounds (1e-3 mm field max) hold on the real level-1 checkpoint.
  NIT 2 covers the procedure if they do not.
- Unverified: peak GPU memory of the float32 unchunked level-1 reference at N 4. Estimated well
  below 48 GB (forward only; largest tensor 7 GB). Measured by step 1a.
- Unverified (carried over): `tf.config.set_visible_devices([], 'GPU')` avoiding the local tf_gpu
  CPU-only crash. The design probes this first and limits the skip to a signal death.
- Unverified (carried over): nibabel 3.1.1 importing under numpy 1.23 in tf_gpu. The install
  needs Jay's OK, and a fallback is named.
- Unverified (carried over): host RAM, GPU memory and time estimates of section 6. Measured into
  predict.json.
- No wrong assumptions found in this round.
