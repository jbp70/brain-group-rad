# fp32-decoder-width: report

Status: done, review APPROVED (4 NITs). Nothing committed, pushed or submitted; base commit ddb91f8.
Jay commits/pushes; the m11 job is pre-approved for submission once this lands.

## What changed

- `--deform-decoder-dtype float32_fine`: only decoder stages d2, d1, their upsamples `up_d2` /
  `up_d1` (when present) and `svf_norm_act` run in float32 (`svf_head` was already float32);
  d5..d3 and the encoder stay float16. Under a float32 global policy it is a no-op.
- NIT 1 fixed in code, NIT 2 documented, NIT 3 covered by tests (see below).
- Width flags documented; no new width flags. Per Jay's override there is NO
  `--cascade-trainable-init`: narrow widths get random-weight cascade sources built inside the job.
- New m11 memory job, random-source helper and a stdlib collector.

### Flags

| Flag | Values | Notes |
|---|---|---|
| `--deform-decoder-dtype` | `policy` (default) / `float16` / `float32` / `float32_fine` (new) | `float32` and `float32_fine` need `--amp true` and `deform` in `--task-ratio`. Variables are float32 with identical names/shapes/order in all four modes, so checkpoints, `copy_weights_from`, warm start and cascade work across modes. Accepted as `DEFORM_DECODER_DTYPE=float32_fine` by `slurm/r_level{4,2,1}.sh` and `slurm/m9_level1_cascade.sh` (dropped with a NOTE for linear-only r_level4). |
| `--cascade-trainable-init copy\|fresh` | not added | Removed by Jay's decision. A narrower level needs matching narrow frozen levels; the m11 job builds random-weight level-4 / level-2 sources per width (`slurm/m11_random_sources.sh`) and uses the normal `--cascade-from` copy path. A static test asserts the flag does not exist. |

### Width flag lines (existing `--filters` stem,L2..L6 / `--decoder-widths` D5..D1)

- default: no width flags. Never pass the explicit default list: `decoder_widths=None` is not equal
  to `(128,96,64,32,16)` in the cascade config check.
- d12h (d1/d2 halved): `--decoder-widths 128,96,64,16,8`
- d12hs (+ stem halved): `--filters 16,64,128,192,256,384 --decoder-widths 128,96,64,16,8`
- fr50 (= preset fr50): `--filters 16,32,128,192,256,384 --decoder-widths 128,96,64,16,8`

Documented in the `runtime/args.py` docstring and help, and in the `models/width_presets.py`
docstring. No PRESETS entries were added.

### FrozenStage dtype decision

Frozen stages (FrozenStage / FrozenChain) always run in float32, whatever the trainable stage's
`--deform-decoder-dtype`. The dtype is a runtime attribute of the trainable model, not a ModelConfig
field, so nothing is stored per stage. Rationale: the probe ran them in fp32, they never train, and
fp32 avoids inference overflow. Documented in `models/cascade.py` and the group_net docstring.

### Nonfinite-fixes NITs

- NIT 1: `runtime/run.py` now uses `lso_flag = amp and save_streak[task] >= LSO_SKIP_STREAK - 1`.
  Gradient-only overflows no longer arm the LSO skip, so the first forward NaN after them reaches the
  LSO. New test `TestNonfiniteGuard.test_lso_flag_ignores_gradient_only_overflow` fails with the old
  flag and passes with the fix; existing guard tests unchanged and passing. One sentence added to
  `.agents/tasks/nonfinite-fixes/report.md`.
- NIT 2 (docs only): with `--nonfinite-patience-task 0` a task in a permanent hard-failure streak
  defers ckpt/last, including the final save, for the rest of the run (`ckpt_last_deferred`
  events). Use ckpt/best* or keep the guard on. Documented in the flag help and the run.py docstring.
- NIT 3: tests for checkpoint read/write and `copy_weights_from` across all ordered mode pairs,
  `C.warm_start_model` into float32 / float32_fine models (mode full, all weights equal), and cascade
  copies into float32 / float32_fine trainable stages at level 2 and through a level-1 FrozenChain.

## m11 job: `slurm/m11_fp32_width_memory.sh`

- SBATCH: `bgr-m11-fp32w`, `-A qtim`, `-p rtx8000`, 1 GPU, 3 CPUs, 64G, 4 h; logs
  `.../GroupwiseReg/logs/bgr-m11-fp32w-<job>.{out,err}`.
- Phase 1, random-weight sources (`slurm/m11_random_sources.sh`, one rigid step each with
  `--deform-decoder true`): l4_d12h, l2_d12h, l4_d12hs, l2_d12hs, l4_fr50, l2_fr50 in
  `$OUT_DIR/sources`. Resumable; a failed source marks its width's grid lines rc -2 (skipped_source).
- Phase 2, grid (20 configs, real trainer, one singularity process each, in this order):
  `l1_f32fine_default l1_f32_default l1_f32fine_d12h l1_f32_d12h l1_f32fine_d12hs l1_f32_d12hs
  l1_f32fine_fr50 l1_f32_fr50 l1_f16_default l1_f16_d12h l1_f16_d12hs l1_f16_fr50
  l2_f32fine_default l2_f32_default l2_f32fine_d12h l2_f32_d12h
  l4_f32fine_default l4_f32_default l4_f32fine_d12h l4_f32_d12h`.
  Default-width lines cascade from the real sources (L1: `m8_level2_gradnorm/ckpt/best`, L2:
  `m7_three_task_lncc/ckpt/best`); width lines from `sources/l2_W/ckpt/last` (L1) or
  `sources/l4_W/ckpt/last` (L2); L4 from scratch.
- Recipe: AMP, deform-only (`--task-ratio deform:1 --global-transform false --curriculum false`),
  8 steps, `--log-every 1`, `--val-every 0`, `--ckpt-every 0` (ckpt deleted after each config),
  `--bn-recal-batches 0`, `--resume false`, AdamW lr 3e-4 wd 0, `--task-grad-norm 12`,
  Dice+pDice+LNCC 1/1/1/1 + smooth 0.5, p 0.1/0.1, `--min-pairs n`, `--gradient-checkpointing full
  --lean-sampler true --split-group-conv true --intensity-aug-device tf --lowres-method tent`.
  L1: N 4 (4 pairs), pair recompute, lean losses, `--gpu-memory-limit-mb 40000`, val chunks 1/1.
  L2: N 16, `--max-pairs 64`, memory growth. L4: N 64, `--max-pairs 64`, memory growth.
- Time estimate (RTX 8000, not measured): sources ~20-25 min; 12 L1 configs ~1.8-2.0 h (~9-10 min
  per running config, ~4 min per OOM); L2 4 x ~5 min; L4 4 x ~4 min; total ~3.0 h of 4 h.
  Guard: a config that would start after 13800 - 300 s is recorded rc -1 (skipped_time);
  each config is capped at 2400 s.
- Split rule: rerun skipped configs in a new job, reusing the sources:
  `ONLY=<names, comma-separated> SOURCES_DIR=<first job>/sources sbatch --time=02:00:00 slurm/m11_fp32_width_memory.sh`
- Output: `.../GroupwiseReg/scratch/m11/<job>/` with `runs/<name>/` (config.json,
  logs/metrics.jsonl), `runs/<name>.log`, `status.tsv`, `sources/`, and `fp32_width_memory.json`
  plus a printed table from `scripts/collect_memory_runs.py` (status ok / oom / timeout / abort /
  skipped_time / skipped_source, peak GiB, mean deform step s, params). OOMs are results; the job's
  exit code is the collector's.
- Submit (cluster, after Jay commits and pushes; check `squeue -u jn85`, at most two active jobs):

  ```bash
  mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
  cd /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad && git pull --ff-only
  sbatch slurm/m11_fp32_width_memory.sh
  ```

## Tests

- Full suite (`tests/run_tests_wsl.sh`, 3 h 54 min): part 1 TF-free 136 tests OK; part 2
  709 tests (4 skipped), 1 failing invocation: `tests.test_group_net` (the known GPU-nondeterminism
  flake in `TestGradientCheckpointing.test_recompute_matches`, 2.48e-09 vs 1.10e-09 on a stem conv
  bias whose gradient is ~0). Fixed in the test only (noise floor 1e-6 -> 1e-5 of the largest
  gradient); rerun `tests.test_group_net` 27 OK and `TestGradientCheckpointing` 4/4 OK. The full
  suite was not rerun after that test-only change.
- Goldens: `[golden] max |diff| 0` and `[golden deform] max |diff| 0` (tolerance 0); policy /
  float16 / float32 graphs unchanged.
- Per module: test_trainer 68, test_cascade 46, test_fp16_mitigations 17, test_main 41,
  test_runtime_infra 35 (1 skip), test_slurm_scripts 35, test_sweep_tools 44,
  test_collect_memory_runs 7, test_width_presets_tf 2, all OK. py3.8 gate OK (89 files, bash -n of
  all slurm scripts, `main.py --help` without TF).

## Open NITs from the review

1. The source phase has no time-budget guard: up to 6 x 1200 s before the grid if sources stall
   (realistically ~20-25 min). Optional: lower `SOURCE_TIMEOUT` to ~600 s.
2. The ~3.0 h estimate is unmeasured; float32 L1 lines may be slower per step than fp16 (23 s).
   Check per-config times in status.tsv after the first job and use the split rule if needed.
3. (Resolved here) report.md written and `.agents/tmp/fp32w_*` helpers deleted.
4. `tests/test_group_net.py` tolerance change is unrelated to the task (flake fix); mention it in
   the commit message.

## Files (`git status --short`)

Modified: `.agents/tasks/nonfinite-fixes/report.md`, `models/cascade.py`, `models/group_net.py`,
`models/width_presets.py`, `runtime/args.py`, `runtime/run.py`, `slurm/m9_level1_cascade.sh`,
`slurm/r_level1.sh`, `slurm/r_level2.sh`, `slurm/r_level4.sh`, `tests/run_tests_wsl.sh`,
`tests/test_cascade.py`, `tests/test_fp16_mitigations.py`, `tests/test_group_net.py`,
`tests/test_main.py`, `tests/test_runtime_infra.py`, `tests/test_slurm_scripts.py`,
`tests/test_sweep_tools.py`, `tests/test_trainer.py`.

New: `.agents/tasks/fp32-decoder-width/` (plan, features, verification, review, this report),
`scripts/collect_memory_runs.py`, `slurm/m11_fp32_width_memory.sh`, `slurm/m11_random_sources.sh`,
`tests/test_collect_memory_runs.py`.

`main.py` and `runtime/checkpoint.py` are unchanged.

## Follow-up L4/L2 job
After jobs 10527941 / 10528396 (every float32 / float32_fine line at L4 N 64 and L2 N 16 OOMed at
default and d12h widths), `slurm/m11_fp32_width_memory.sh` has 28 more configs, appended after the
existing 20 (names and order of those unchanged; SBATCH header unchanged):
- 12 main lines, recipe as the existing l4_* / l2_* lines: L4 deform N 64 from scratch and L2 deform
  N 16 cascading from the random `l4_W` source, `--max-pairs 64`, {f32fine, f32, f16} x {d12hs, fr50}.
- 16 fallbacks, float32_fine / float32 only, d12hs / fr50, named `..._n<N>`: L4 N 48, N 32, then
  L2 N 12, N 8. The suffix sets `--group-size N --deform-group-size N`; status.tsv column `n` (and
  so the collector's `n`) is the real N.
- Sources: a width used only by level-2 lines now builds only `l4_W` (new `SOURCE_L4_ONLY` in
  `m11_random_sources.sh`, filled by the job); `l2_W` is built only if a level-1 line needs it. Both
  ONLY lists below build `l4_d12hs` and `l4_fr50` only.

Dry run (`ONLY=<12 main names> BGR_DRY_RUN=1 PROJ=/tmp/m11dry bash slurm/m11_fp32_width_memory.sh`,
common recipe flags cut):
```
SOURCE l4_d12hs: python main.py ... --run-name l4_d12hs --level 4 ... --filters 16,64,128,192,256,384 --decoder-widths 128,96,64,16,8
SOURCE l4_fr50:  python main.py ... --run-name l4_fr50 --level 4 ... --filters 16,32,128,192,256,384 --decoder-widths 128,96,64,16,8
CONFIG l4_f32fine_d12hs: ... --level 4 --group-size 64 --deform-group-size 64 --max-pairs 64 --pair-recompute false --val-pair-chunk 16 --filters 16,64,... --run-name l4_f32fine_d12hs --deform-decoder-dtype float32_fine
CONFIG l4_f32_d12hs / l4_f16_d12hs / l4_f32fine_fr50 / l4_f32_fr50 / l4_f16_fr50: same, dtype float32 / float16, fr50 = --filters 16,32,...
CONFIG l2_f32fine_d12hs: ... --level 2 --cascade-from .../sources/l4_d12hs/ckpt/last --group-size 16 --deform-group-size 16 --max-pairs 64 --pair-recompute false --val-pair-chunk 4 --filters 16,64,... --deform-decoder-dtype float32_fine
CONFIG l2_f32_d12hs / l2_f16_d12hs: same; l2_*_fr50: --cascade-from .../sources/l4_fr50/ckpt/last
COLLECT: python scripts/collect_memory_runs.py --status .../status.tsv --runs .../runs --out .../fp32_width_memory.json --job manual
```
Fallback dry run: `--deform-group-size` 48 / 32 (L4) and 12 / 8 (L2), L2 lines cascade from
`sources/l4_d12hs` or `sources/l4_fr50`, sources again `l4_d12hs`, `l4_fr50` only.

Time estimate for the 12 main lines (RTX 8000): 2 level-4 sources ~1-2 min each; L4 N 64 line
~2 min if it OOMs (measured 100-130 s), ~4-6 min if it fits; L2 N 16 line ~2 min OOM (120-130 s),
~4-6 min if it fits. Expected (f16 fits, part of the fp32 lines fit) ~45-55 min; worst case (all
12 fit at 6 min) ~75-78 min. So 1 h 15 min is enough in the expected case but not in the worst
case. The job's time guard defaults to 13800 s and does not see the shorter `--time`, so pass
`TIME_BUDGET_S=4200` (70 min): a config that would start after 65 min is recorded as
skipped_time, the remaining ones are capped at 70 min, and the collector still runs before SLURM
kills the job. Skipped lines are rerun with the split rule. (`--time=01:30:00` with the default
guard also removes the risk.)

(a) 12 main configs:
```
ONLY=l4_f32fine_d12hs,l4_f32_d12hs,l4_f16_d12hs,l4_f32fine_fr50,l4_f32_fr50,l4_f16_fr50,l2_f32fine_d12hs,l2_f32_d12hs,l2_f16_d12hs,l2_f32fine_fr50,l2_f32_fr50,l2_f16_fr50 SOURCE_TIMEOUT=600 TIME_BUDGET_S=4200 sbatch --time=01:15:00 slurm/m11_fp32_width_memory.sh
```
(b) fallbacks (only if needed; reuse the sources of job (a); ~16 x 2-5 min ~45-75 min):
```
ONLY=l4_f32fine_d12hs_n48,l4_f32_d12hs_n48,l4_f32fine_fr50_n48,l4_f32_fr50_n48,l4_f32fine_d12hs_n32,l4_f32_d12hs_n32,l4_f32fine_fr50_n32,l4_f32_fr50_n32,l2_f32fine_d12hs_n12,l2_f32_d12hs_n12,l2_f32fine_fr50_n12,l2_f32_fr50_n12,l2_f32fine_d12hs_n8,l2_f32_d12hs_n8,l2_f32fine_fr50_n8,l2_f32_fr50_n8 SOURCES_DIR=<job (a) dir>/sources SOURCE_TIMEOUT=600 TIME_BUDGET_S=6000 sbatch --time=01:45:00 slurm/m11_fp32_width_memory.sh
```
A subset of (b) is fine (e.g. only the N steps below the largest fitting main config).
rtx8000, account qtim, 1 GPU, 3 CPUs, 64G (SBATCH header). Check `squeue -u jn85` (2-job limit).

Tests (WSL, bgr_preproc38): `python -m unittest tests.test_slurm_scripts tests.test_collect_memory_runs`
-> 44 tests OK; `bash -n` on both scripts OK.
