# Implementation Plan: standalone groupwise prediction script

Source: approved design `.agents/tasks/predict-script/design.md` v5 (review v5 APPROVED, 2 NITs,
both folded in below). Repo `d:\Research\Brain_Registration_Projects\brain-group-rad`, branch
`main`, no worktree. Commit locally after each item group; push is a later workflow step.
The same work is split into FEAT files under `.agents/tasks/predict-script/features/`
(FEAT-001..005 = items 1-2, 3, 4-5, 6-8, 9); `context.json` holds the commands and patterns.

Commands (WSL, from PowerShell; write multi-line commands to `.agents/tmp/*.sh`, see the
`shell-and-remote-jobs` skill):
- One TF module: `cd /mnt/d/Research/Brain_Registration_Projects/brain-group-rad && source ~/miniconda3/etc/profile.d/conda.sh && conda activate tf_gpu && export TF_CPP_MIN_LOG_LEVEL=3 BGR_TF_GPU_LIMIT_MB=6000 TF_CUDNN_USE_AUTOTUNE=0 TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 PYTHONDONTWRITEBYTECODE=1 && python -m unittest tests.<module> -v`
- py3.8 syntax: `conda activate bgr_preproc38` + `ast.parse(src, f, feature_version=(3, 8))`.
- Full suite (~3 h, background): `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/tests/run_tests_wsl.sh`, must end with `suite status: 0`.

Design decisions carried over (not re-decided): branch set {rigid, affine, deform,
affine+deform} (Jay D1); deform-only for the Mindboggle cluster run (Jay's clarification);
lean sampler always on at inference; engine always uses the split conv form; check-streaming
uses the FIRST `--n-subjects` subjects in `find_subjects` order (review NIT 1); the SLURM header
documents the exit-4 procedure (review NIT 2).

- [ ] 1. Environment prerequisite: nibabel in tf_gpu (design 10.0).
      `import nibabel` fails in tf_gpu today (probed 2026-11-01). Pause and ask Jay to approve
      `conda activate tf_gpu && pip install nibabel==3.1.1`; install after approval, re-probe,
      record in `.agents/tasks/predict-script/findings-env.md`. Add the stage-2 guard
      `python -c "import nibabel" || { echo "tf_gpu needs nibabel (pip install nibabel==3.1.1)"; status=1; }`
      after the version print in `tests/run_tests_wsl.sh`.
      Files: tests/run_tests_wsl.sh, .agents/tasks/predict-script/findings-env.md
      Verify: `python -c "import nibabel, numpy; print(nibabel.__version__, numpy.__version__)"`
      in tf_gpu prints 3.1.1; `python -m unittest tests.test_evaluation` passes (it fails at import today).

- [ ] 2. Bit-identical GroupConvBlock refactor + guard (design 4.2).
      First copy today's `_split_call` body verbatim into `tests/test_predict_streaming.py` as
      `_legacy_split_call(block, x, training)` (comment: HEAD 95e58c8). Then add
      `GroupConvBlock.pre(x, training=None)` and
      `GroupConvBlock.conv_with_group_mean(x, mean, inner_clip=True)`; `_split_call` = `_pre`
      (concat False) -> `reduce_mean(axis 0, keepdims)` -> `conv_with_group_mean`. Tests:
      bitwise legacy vs refactored (float32 / mixed_float16, clip 0 / 100 active, strides 1 / 2),
      engine primitive vs `block(x, training=False)` split on (bitwise) and off (allclose).
      Files: models/layers.py, tests/test_predict_streaming.py
      Verify: tests.test_predict_streaming, tests.test_split_group_conv, tests.test_group_layers,
      tests.test_fp16_mitigations, tests.test_trainer (goldens) all OK.

- [ ] 3. Stores and the streamed single-level engine (design 4.1, 4.3, 4.6, section 6).
      `inference/__init__.py`, `inference/store.py` (RamStore, DiskStore via open_memmap with
      cleanup on error, `estimate_store_gb`, `choose_store` auto = RAM iff estimate <= 0.7 x
      budget from SLURM_MEM_PER_NODE or MemAvailable), `inference/streaming.py`
      (`max_elements_per_image`, `chunk_cap`, `StreamedLevel.deform(n, input_fn, prev_fn,
      sink)` with passes P0-P12, `StreamedLevel.linear(n, input_fn, task)` -> A_raw from the
      gathered bottleneck; float32 accumulation of group means; one tf.function per (pass,
      chunk length); fp16 max-abs per block; block='residual' -> ValueError).
      Files: inference/__init__.py, inference/store.py, inference/streaming.py,
      tests/test_predict_store.py (TF-free), tests/test_predict_streaming.py
      Verify: tests.test_predict_store in bgr_preproc38 without TF; tests.test_predict_streaming
      in tf_gpu: streamed == model call at levels 4 / 2 (chunks 1, 2, N; float32 1e-4 vox / A
      1e-5; mixed_float16 2e-2 max / 1e-3 mean; split on/off; config A/B; clip; float32_fine;
      lean on/off bitwise) and level 1 TINY on `/CPU:0`; hand counts 440401920 / 220200960 ->
      caps 4 / 9.

- [ ] 4. Checkpoint loader (design 3.2, 3.3). Depends on 3 (`max_elements_per_image`).
      `inference/loader.py`: `load_model(path, branches, precision, device)` -> LoadedModel;
      config.json defaults for absent knob keys; mixed_float16 trainable stage iff amp and
      precision 'checkpoint' and device gpu; 'float16' decoder dtype under a float32 build ->
      'policy' (recorded); `lean_sampler=True` always; FrozenStage / FrozenChain `from_config`;
      restore with `frozen_stage` / `frozen_stage_4` + `frozen_stage_2`; fingerprint checks;
      step from STEP_KEY; `unchunked_n_max` / `check_unchunked` from the config alone.
      `LoaderError(ValueError)` -> exit 2 in the CLI.
      Files: inference/loader.py, tests/predict_helpers.py (save_level4_run, save_cascade_run,
      save_chain_run, writing config.json records like main.py 322-354), tests/test_predict_loader.py
      Verify: tests.test_predict_loader OK (precision mapping, old config, fingerprint mismatch,
      missing config, n_max 4 / 9 with the exit-2 message).

- [ ] 5. Runners: branch orchestration over the cascade (design 3.4, 4.4, 4.5). Depends on 3, 4.
      `inference/runner.py`: `BRANCHES`; `StreamedRunner` (level inputs pooled per chunk from
      the level-1 source; linear chain A4 -> A2 -> A1 with prewarp_linear / compose_linear;
      deform chain with diffeo.warp(lean) pre-warps, upsample + compose_cascade; affine+deform
      resamples the full-res images once with A); `UnchunkedRunner` (one tf.function per level
      stage, the trainer's own cascade ops, `check_unchunked` first); common `run(images,
      branches, sink)` with per-chunk level-1 results plus the checkpoint-level fields for
      test (b); summary with A, chunks, caps, stats, timing, gpu_peak_mb.
      Files: inference/runner.py, tests/test_predict_runner.py
      Verify: tests.test_predict_runner OK: (a) streamed == unchunked at levels 4 / 2, all four
      branches, chunks 1, 2, N, float32 and fp16 bound sets, chunk boundaries [(0, 2), (2, 3)];
      level 1 (FrozenChain + float32 level-1 model) on `/CPU:0` for deform and affine+deform;
      (b) unchunked vs Trainer `_val_deform_fn` / `_val_fn` under `_OpDeterminism` within 1e-6
      (fallback 1e-5) and identical Dice; streamed within S_stream; maxima and peak RSS printed.
      tests.test_cascade still OK.

- [ ] 6. Canonical conversion, writer, atlas, internal check, grid reference (design 5).
      `inference/canonical.py`: `to_canonical` (design 5.1 table, float64 V / Mv, border-clamped
      u_inv sampling for affine+deform, x spacing for mm), `internal_moved` /
      `internal_pull_inv` / interior mask / `internal_check` (coord 1e-3 vox, moved 1e-3 except
      affine+deform), `load_reference_affine` / `check_grid`, atomic `write_nifti` with
      compress level, `AtlasAccumulator` (float64 mean; `evaluation.metrics.MajorityVote`).
      Files: inference/canonical.py (tests in item 8)
      Verify: py3.8 ast check; unit tests of item 8.

- [ ] 7. CLI, run / check-streaming drivers, entry point (design 3.3, 7, 8). Depends on 4-6.
      `inference/cli.py` (parser per design 7, `validate` before any TF import,
      `resolve_check_chunks`), `inference/predict.py` (`setup_device`, `run_predict(args,
      configure_device=True)`, `run_check_streaming`, predict.json / check JSON per design 8,
      `git_commit` with git -> .git -> none fallback, exit codes 0/1/2/3/4/5 with 3 over 5),
      `predict.py` (root entry).
      Files: inference/cli.py, inference/predict.py, predict.py
      Verify: `python predict.py run --help` and `python predict.py check-streaming --help` in
      bgr_preproc38 (no TF needed for argparse); item 8 tests.

- [ ] 8. End-to-end and unit tests for items 6-7 (AC 2 file part, AC 3, AC 4, AC 5).
      `tests/predict_helpers.py` gains `write_subjects` (NIfTI subjects, textured option).
      `tests/test_predict.py`: level-1 written files vs trainer u / u_inv (CPU scope, unchunked
      and streamed); convention checks per branch with forced magnitudes and the wrong-conversion
      power test; check-convention RESULT line + ranking margins; `evaluation run --pairs 2`;
      branch combinations and rejections; conversion unit tests at 1 mm and 2 mm; chunk resolver;
      grid reference (json, NIfTI, mirrored, absent); predict.json keys; nonfinite -> 3;
      internal mismatch -> 5; git fallbacks; CLI validation without TF; setup_device call order;
      `--device cpu` subprocess (skip only on a signal death); check-streaming e2e and the
      n_max exit 2.
      Files: tests/predict_helpers.py, tests/test_predict.py
      Verify: tests.test_predict OK; tests.test_evaluation OK; earlier predict modules still OK.

- [ ] 9. SLURM job + suite wiring (design 9, AC 6). Independent of items 3-8 except the CLI flags.
      `slurm/predict_mindboggle.sh` (rtx8000, -A qtim, 1 GPU, 3 CPUs, 96G, 5 h; steps 1a gate
      float32 `|| exit $?`, 1b fp16 and 1c affine+deform non-blocking, 2 predict deform-only, 3
      check-convention + template-only evaluation; prints the basic-partition full evaluation
      command; header: never submit without Jay, OOM fallback, exit-4 procedure). Static test in
      `tests/test_slurm_scripts.py`. `tests/run_tests_wsl.sh`: PYFILES += `predict.py
      inference/*.py evaluation/*.py`, `bash -n` += predict / eval scripts, TF-free list +=
      `tests.test_predict_store`.
      Files: slurm/predict_mindboggle.sh, tests/test_slurm_scripts.py, tests/run_tests_wsl.sh
      Verify: bgr_preproc38 `python -m unittest tests.test_slurm_scripts tests.test_predict_store`
      without TF; `bash -n slurm/predict_mindboggle.sh`.

- [ ] 10. Integration: full suite and docs.
      Run `tests/run_tests_wsl.sh` in the background (~3 h); fix any failure at the seams.
      Files: as needed
      Verify: log ends with `suite status: 0`; record test counts and the printed maxima in the
      FEAT findings (the report step uses them).

## Digest

Baseline verified for this plan (2026-11-01, HEAD 95e58c8, clean tree):
`python -m unittest tests.test_split_group_conv` in tf_gpu: 5 tests OK (38 s).
`python -m unittest tests.test_evaluation` in tf_gpu: import error (no nibabel), expected until item 1.
`import nibabel` in tf_gpu: ModuleNotFoundError.

- `models/layers.py` `GroupConvBlock` (lines 653-711). `call(self, x, training=None)` takes
  `_split_call` iff `self.split_group and self.group and self.conv.built`, else
  `clip_fp16(self.conv(self._pre(x, training)), self.fp16_act_clip)`. `_split_call(self, x,
  training)` lines 691-704: `x = self._pre(x, training, concat=False)`; `channels =
  int(x.shape[-1])`; `mean = tf.reduce_mean(x, axis=0, keepdims=True)`; `k =
  tf.cast(self.conv.kernel, x.dtype)`; `strides = [1, s, s, s, 1]`; `c = self.fp16_act_clip`;
  `y = clip_fp16(conv3d(x, k[..., :channels, :]), c) + clip_fp16(conv3d(mean, k[...,
  channels:, :]), c)`; `return clip_fp16(tf.nn.bias_add(y, tf.cast(self.conv.bias, x.dtype)),
  c)`. `_PreActivation._pre(x, training, concat=True)` (636-644): norm -> act -> group_mean_concat
  if `self.group and concat`. The stem has `pre_activation=False` (norm / act None) and is never
  split (`set_split_group_conv` skips it). Covered by tests/test_split_group_conv.py (5 tests),
  tests/test_group_layers.py, tests/test_fp16_mitigations.py, trainer goldens in tests/test_trainer.py.
- `models/group_net.py` (read only): `GroupEncoder.call` (104-117): stem -> skips[0]; pool2
  (level 1); l2; pool3 (levels 2/1); l3..l6 (l4-l6 stride 2). `DeformDecoder.call` (245-263):
  cast(skips[5], compute_dtype); per stage cast(x, block.compute_dtype), up (d5-d3 always, d2 at
  levels 2/1, d1 at level 1), `block(concat([x, cast(skip, x.dtype)]))`; head
  `svf_head(svf_norm_act(cast(x, svf_norm_act.compute_dtype)))` cast float32.
  `GroupRegLevelModel.__init__(level, config, gradient_checkpointing, lean_sampler,
  split_group_conv, fp16_act_clip, fp16_act_clip_scope, deform_decoder_dtype,
  bn_epsilon_override)`: 'float16' decoder dtype raises under a float32 global policy.
  `_call_deform`: `v = diffeo.center_velocities(v_raw)`, `exp_and_inverse(v, steps, lean)`.
  Linear `call`: `linear_decoder(bottleneck, task, training)` -> (A_raw, params, feats), `A =
  centering(A_raw)`.
- `models/cascade.py` (read only): `pool_to_coarser`, `prewarp_linear(images, A_prev,
  spacing)`, `compose_linear(R_raw, A_prev, centering)`, `prewarp_deform(images, U)` =
  `diffeo.warp(float32, U)` (no lean flag), `FrozenStage.from_config/linear/deform/
  checkpoint_extra/fingerprint`, `FrozenChain.from_config/_linear/_deform/linear_stages/
  deform_stages/checkpoint_extra/fingerprints`. `diffeo.compose_cascade(U, U_inv, u, u_inv,
  lean)` = (compose(U, u), compose(u_inv, U_inv)).
- `runtime/checkpoint.py` (read only): `resolve_checkpoint`, `source_config_path`,
  `STEP_KEY`, `fingerprints_match(saved, current, rtol=1e-12)`, `weight_fingerprint`,
  `RunCheckpoint(ckpt_last, ckpt_best, model, opt, step, extra=...)`, `write_config(path, cfg,
  args, extra)`. main.py 322-354 writes `cascade` {fingerprint, stages[{level, checkpoint_key,
  fingerprint, origin_checkpoint, origin_step}]}; main.py 217-223 GPU setup pattern.
- `runtime/run.py` (read only, test references): `Trainer._val_fn` 1179, `_cascade_val_fn` 1241,
  `_val_deform_fn` 1295, `_cascade_val_deform_fn` 1336, `_deform_errors` 1378.
- `evaluation/evaluate.py` (read only): `find_subjects(processed_dir, label_name,
  subjects_file)` (sorted dirs with image.nii.gz; subject from meta.json row.subject or dir minus
  `_T1w`; >= 2 subjects), `load_grid_data(subs)` -> images, labels, masks, shape, affine
  (ValueError off-grid), `spacing_of(aff)`, `cmd_check_convention` (RESULT line, `--out` JSON with
  ranking / identity_ncc), `main(argv)`. `evaluation/fields.py`: `warp(vol, disp, order)`,
  `compose(a, b)`. `evaluation/metrics.py`: `MajorityVote(warped).full()`.
- Test helpers (read only): `tests/test_cascade.py` `randomize` (45), `save_run` (57),
  `CascadeTrainerCase.stages/make` (604-640), `ChainCase.frozen_models/chain/level1_model/make1`
  (1394-1435), `chain_config` (1353), `N_CLASSES = 9`, `STAGE_KW`; `tests/test_trainer.py`
  `_OpDeterminism` (1330); `tests/runtime_helpers.py` `tiny_config`, `free_memory`, `synthetic_group`;
  `runtime/fixed_data.py` `synthetic_deform_group` (358).
- `tests/run_tests_wsl.sh`: `PYFILES=` line (stage 1), the `for s in ... ; do bash -n` loop, the
  TF-free `names = [...]` list, stage-2 version print line (guard goes right after it).
- `tests/test_slurm_scripts.py`: module constants (`SCRIPTS`, `M10C`, `M11` ...), `TestStatic`,
  `test_bash_syntax` iterates `SCRIPTS + (M10C, M11)`; stdlib only.
- `slurm/eval_mindboggle.sh`: header / container-call pattern to copy (`singularity exec -B
  /autofs -B /scratch "$SIF" bash -c 'cd "$1" && shift && python -m evaluation "$@"' ...
  </dev/null`).
