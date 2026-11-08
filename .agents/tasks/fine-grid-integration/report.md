# Report: optional fine-grid integration of frozen cascade stages

Branch `fine-grid-integration` (worktree `.worktrees/fine-grid-integration`, base `ee8a88a`).
Plan: `plan.md` (decisions D1-D9). Design addendum: `.agents/tasks/network-design/design.md`
section 4.7, "Optional fine-grid frozen integration".

## Why

The inverse-centrality investigation traced 89 % of the inverse error to integrating the level-4
SVF on the 4 mm grid. The 1 mm probe (job 10620870) showed that integrating level 4 at 1 mm makes
level 4 alone 7x more invertible and slightly more accurate. Doing it only at inference breaks
the cascade: Dice -0.006, LNCC -0.02, folding 80-120x higher, because L2 / L1 were trained on the
4 mm-integrated prior. This branch adds a train/test-consistent option.

## Decisions (short)

- D1 Names: training `--frozen-integration-grid native|current` (argparse default None = inherit);
  inference `--integration-grid auto|native|fine` (`run`, `check-streaming`), default `auto`.
- D2 The switch lives in `models/cascade.py` (`FrozenStage` / `FrozenChain` attribute
  `integration_grid`, set by every factory). Every trainer path goes through the cascade methods,
  so `runtime/run.py` is unchanged. Native branches are verbatim (bit-identical graphs/outputs).
- D3 Current-mode math: `diffeo.exp_compose_upsampled` (per image, `tf.map_fn`,
  parallel_iterations 1) and `cascade.integrate_frozen_at([(level_k, v_k)], level, T)`. Level 2:
  U = exp at 2 mm of up(v4). Level 1: level-2 pre-warp with exp at 2 mm of up(v4), then
  U = exp(up^2 v4) o exp(up v2) at 1 mm. Skipped stages = zero SVF (identity, exact).
- D4 Chain rule: fresh L2 = flag or native; fresh L1 inherits the L2 source's value (explicit
  mismatch -> main.py exits); resume keeps the saved value (explicit mismatch -> RuntimeError).
  config.json: `args.frozen_integration_grid` and `cascade.frozen_integration_grid`.
- D5 Inference follows the checkpoint (`auto`: current -> fine). `fine` = each level pre-warped
  as in training, all run levels integrated and composed at 1 mm (probe variant C). `--levels`
  plans, TTA flip8 and affine+deform are supported. An explicit override logs a warning.
  predict.json `knobs.integration_grid = {checkpoint, used}`.
- D6 Memory strategy: frozen work outside the tape, 1 mm integrations one image at a time.
- D7 Measurement with the real trainer: `slurm/fgi_memory.sh` (below).
- D8 Script knob `FROZEN_INTEGRATION_GRID` in `r_level2.sh` / `r_level1.sh` (empty = no flag);
  `r_cascade.sh` only lists the key in `LEVEL_ENV_KEYS`. No default changes.
- D9 Tests: `tests/test_fine_grid_integration.py` (TF) and `tests/test_slurm_scripts.py`
  (`TestFrozenGridKnob`, `TestFgiMemory`); goldens unchanged.

## How to use

- Level 2: `FROZEN_INTEGRATION_GRID=current` with `slurm/r_level2.sh` (or
  `--frozen-integration-grid current` in EXTRA_ARGS).
- Level 1: nothing to set; it inherits `current` from its level-2 source. Setting
  `FROZEN_INTEGRATION_GRID=native` on a current source fails in main.py (by design).
- Chain: `EXTRA_ARGS_L2='--frozen-integration-grid current'` with `slurm/r_cascade.sh`.
- Level 4 is unaffected (no frozen stage). A level-4 checkpoint predicted with
  `--integration-grid fine` gives U1 = exp at 1 mm of up(up(v4)) (probe L4f).
- Prediction: `predict.py run ...` follows the checkpoint. `--integration-grid fine` on a native
  checkpoint reproduces the inference-only probe (variant C), `native` on a current checkpoint
  the reverse; both warn.

## Memory and time estimate (plan D6, NOT measured)

| Level | Config | Expected TF peak change | Expected deform step change |
|---|---|---|---|
| 2 | deform N 16, 24 pairs, d12hs fp16 | <= +0.2 GB (ref 27.2-27.8 GiB) | < +0.5 s |
| 1 | deform N 4, 4 pairs, float32_fine, 42500 MB limit | 0 to +1.5 GB (ref 37.3-38.9 GB) | +1-3 s |

The level-1 transient (~1-1.5 GB for one image's exp + compose at 1 mm) comes before the forward /
backward peak; fragmentation is the risk the job measures.

## Commands (cluster clone, after merge + push; ask Jay before any submission, 2-job limit)

```
mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
cd /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad && git pull --ff-only
sbatch slurm/fgi_memory.sh
P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
RUN_NAME=r2_fgi_smoke CASCADE_FROM=$P/results/r4_cfgB_dice1_smooth2_lr3e-4_wd1e-1/ckpt/last FROZEN_INTEGRATION_GRID=current STEPS=200 VAL_EVERY=100 WARMUP=50 PROJECT_STEPS=15000 PROJECT_VAL_EVERY=500 WD=1e-1 GPU_POLL=true EXTRA_ARGS="--group-config B --lambda-smooth 10" sbatch --job-name=r2fgism --time=03:00:00 --mem=64G slurm/r_level2.sh
```

Jay picks the smoothness (10 = his chosen L2 value). Check first without submitting:
`BGR_DRY_RUN=1 bash slurm/fgi_memory.sh` prints the four configs.

`slurm/fgi_memory.sh` (rtx8000, qtim, 1 GPU, 3 CPUs, 128G, 3 h, ~2 h expected): configs
`l2_native l2_current l1_native l1_current`, each the real `r_level2.sh` / `r_level1.sh` recipe
(taken from the level script's dry-run MAIN line) with STEPS 16, VAL_EVERY 8, CKPT_EVERY 0,
BN_RECAL_BATCHES 4, `--group-config B --lambda-smooth 10`; L2 lines cascade from
`L4_SOURCE` (default `results/r4_cfgB_dice1_smooth2_lr3e-4_wd1e-1/ckpt/last`), L1 lines from the
job's own `runs/l2_<grid>/ckpt/last`. Env overrides: L4_SOURCE, GROUP_ARGS, SMOOTH, STEPS,
VAL_EVERY, BN_RECAL_BATCHES, CONFIG_TIMEOUT (3600), ONLY. Output `$PROJ/scratch/fgi/<job>/`.

## How to read the results

- `fgi_memory.json` / the printed table (`scripts/collect_memory_runs.py`): `gpu_peak_gib` and
  `mean_deform_step_s` per config; compare current vs native at each level against the table above.
- `runs/<name>/gpu_poll.csv` (nvidia-smi every 5 s): max `memory.used` must stay <= `memory.total`
  - 512 MiB (device level, includes DALI and the CUDA context).
- Level 1 current must fit the 42500 MB TF limit with >= 1 GB headroom (TF peak <= ~41.5 GB).
  If it does not: fall back to `MIN_PAIRS=1` or a smaller deform group only after asking Jay.
- status `skipped_source` (rc -2) on an L1 line = its L2 config failed; read `runs/l2_<grid>.log`.
- L2 smoke: checker PASS, `cascade.frozen_integration_grid == "current"` in config.json, deform
  non-finite only during AMP warm-up, `gpu_peak` within the L2 budget.

## Commits and test results

| Feature | Commit | Tests |
|---|---|---|
| FEAT-001 training path | e7c0851 | test_fine_grid_integration 14 OK, test_cascade 52, test_main 44, test_trainer 74 (goldens 0 diff) |
| FEAT-002 inference | 90c3d41 | test_fine_grid_integration 22 OK, test_predict* OK (TTA golden 0 diff); test_predict TestGitCommit fails only in the worktree (WSL git cannot read the worktree .git file) |
| FEAT-003 scripts + docs | (this commit, see git log) | test_slurm_scripts (see FEAT-003 findings) |
| Full suite | TODO (integration step) | TODO: `tests/run_tests_wsl.sh` -> `suite status: ...` |
