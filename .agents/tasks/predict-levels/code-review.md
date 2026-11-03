# predict-levels code review (commit 5b27491 vs origin/main)

Verdict: APPROVED, no blockers.

## What I checked

**Default path stays bit-identical.** With `plan is None` or a normalised full plan (`plan.default`), `StreamedRunner.run` and `UnchunkedRunner.run` take the unchanged default body. The default methods (`_linear`, `_deform`, `_level_sink`, `_pool_fn`, `_prewarp_*_fn`, `_up_fn`, `_to_level1_fn`) and their cache keys were not edited. The only change on the default path is the new summary key `levels`. `_pipeline` returns the same text when the plan is the default. In `slurm/predict_mindboggle.sh`, `LEVEL_FLAGS` is empty by default and is expanded unquoted, so the default commands are unchanged. The commit records evidence for both levels of the check: golden sha256 of all NIfTI outputs, parent vs HEAD, is identical for deform, affine+deform and rigid in streamed and unchunked mode; and the general path with a full plan matches the default path exactly (`np.array_equal`) at L2 chunks 1/3 and L1 GPU chunk 1.

**Skip semantics match the trainer.** I compared the runner against `runtime/run.py` stage mode (deform and linear bodies) and `models/cascade.py` (`FrozenChain.deform_skip`, `linear_skip`, `skip_fields`, `skip_linear`, `skip_to`).
- Streamed linear: a skip before any level has run gives `tf.eye` (the identity), which equals the `skip_linear` value. Later stages then use `prewarp_linear(pool(x), I)` and `compose_linear`. For `4,1`, A4 is kept, which matches `skip_to(skip2, A4, A)`.
- Streamed deform: a skip with nothing run yet (`_ZERO`) gives the input `warp(pool(x), 0)` and `prev_fn = (0, 0)`. The engine's P12 step then computes `compose_cascade(0, 0, u, u_inv)`, the same as `deform_skip(skip4)`. For `4,1`, the input is two upsamples of u4, which equals `up(U4)` in `deform_skip(skip2)`.
- Unchunked mode calls the trainer functions literally, with `tf.constant(True)` as the skip tensor.
- Legacy `cascade_skip_prob` maps to a level-4 skip for a level-2 run and a level-2 skip for a level-1 run. That matches the trainer's `skip_on` branches.

**Truncation upsampling matches the cascade.** Truncated levels are not run. The last run level goes through `_to_level1_fn` / `_up_to_level1`, which applies one `upsample_displacement` per 2x step, forward and inverse separately. That is what `FrozenStage.deform`, `FrozenChain._deform` and `deform_stages` do. Matrices pass through unchanged. The tests show `4` equals `deform_skip(skip2)` within 1e-5.

**Exposure, refusal and records.**
- `run_skip_probs` reads `args.cascade_skip_stage_parsed`. Its task keys are the trainer's task names, and both int and string level keys are accepted. Missing keys count as 0; an unreadable config is reported as `unknown`.
- `exposure` computes joint probabilities as products of the per-level probabilities. `_setup_levels` runs before subjects are read or models are built, fills `rec['levels']` (also on refusal), exits 2 without `--allow-untrained-skip`, and sets `forced` plus a WARNING when the flag is given.
- A WARNING is logged when the joint probability is below 0.05 (0.0225 for `1`).
- `levels` was added to `PREDICT_KEYS` and `CHECK_KEYS`.
- The level-2 source config is found through `cascade.source_config`, then the `origin_checkpoint` of the level-2 stage.

**check-streaming coverage.** `--levels` is available in `_common`, and chunk keys and caps are limited to the levels that run. Streamed == unchunked is tested at chunk 2 for L1 `4,2`, `4`, `2,1`, `1`, `2` and `4,1` (deform and affine+deform), and at chunks 1/2/N for L2 `4` and `2` (rigid, deform and affine+deform). Each case is also checked against a direct reference built from the trainer functions. End-to-end runs of check-streaming with `--levels 2` and `--levels 4` passed.

**Python 3.8 and tests.** The full suite's py3.8 ast check passed (111 files). The commit message records 825 tests with 0 failures (21303 s) and the module tests passing first. check-convention still selects xyz +++ mm for outputs from `--levels 2` and `--levels 4`.

## Non-blocking notes (not findings)
- At L1, the rigid-only and affine-only branches are covered through the affine part of affine+deform rather than as separate branches. Both use the same `_linear_sel` code.
- In `_run_check`, `_setup_levels` now runs before `_check_unchunked`. This only changes which exit-2 message appears first.
