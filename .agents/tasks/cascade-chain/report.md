# cascade-chain: one SLURM job for the full level 4 -> 2 -> 1 cascade

New file `slurm/r_cascade.sh`, tests in `tests/test_slurm_scripts.py` (`TestCascadeStatic`, `TestCascadeRuns`),
and `slurm/r_cascade.sh` added to the `bash -n` list in `tests/run_tests_wsl.sh`. Nothing committed, pushed or
submitted.

## What the script does

It runs `slurm/r_level4.sh`, `slurm/r_level2.sh` and `slurm/r_level1.sh` one after another with `bash` inside
one allocation (rtx8000, 1 GPU, 3 CPUs, 128G, 72 h, account qtim). The level scripts' `#SBATCH` lines are then
plain comments. Each level gets an explicit environment. Before each level, every recipe variable the level
scripts read (`LEVEL_ENV_KEYS`: `N`, `DEFORM_N`, `INIT`, `FILTERS`, `DECODER_WIDTHS`, `SKIP_*`, `BN_RECAL_*`,
`MIN_TASK_STEPS`, `EXTRA_ARGS`, ...) is unset in a subshell. A stale value in the submitting shell therefore
never reaches a level. This matters for `FILTERS=` (set but empty), which would silently drop the d12hs widths.
Only `PROJ REPO DATA RESULTS SIF SCRATCH_ROOT SLURM_JOB_ID SLURM_CPUS_PER_TASK CUDA_VISIBLE_DEVICES` pass
through. A test checks that every `${NAME:-` / `${NAME-` in the three level scripts is either unset or passed
through.

Wiring follows Jay's last-checkpoint rule: level 2 `CASCADE_FROM=$RESULTS/$RUN_NAME_L4/ckpt/last` and level 1
`CASCADE_FROM=$RESULTS/$RUN_NAME_L2/ckpt/last`.

## Env table

| Variable | Default | Effect |
|---|---|---|
| `S4`, `S2`, `S1` | required | `--lambda-smooth` per level. A missing value, or one that is not a positive number, gives exit 2 |
| `SYNTH_PROB` | 0.5 | passed to every level (0 = real-only chain, no synth flags) |
| `RUN_NAME_L4` / `_L2` / `_L1` | `r4_cfgB_synth05` / `r2_cfgB_synth05` / `r1_cfgB_synth05` | run names (must differ) |
| `GROUP_ARGS` | `--group-config B` | start of every level's `EXTRA_ARGS`, followed by `--lambda-smooth $Sk` (set but empty = no group flag) |
| `EXTRA_ARGS_L4` / `_L2` / `_L1` | empty | appended last to that level's `EXTRA_ARGS` (main.py flags) |
| `STEPS_L4` / `_L2` / `_L1` | 16000 / 15000 / 4000 | `STEPS` and `PROJECT_STEPS` of the level |
| `LR`, `WD`, `VAL_EVERY` | 3e-4, 1e-1, 500 | all levels |
| `WARMUP_L4`, `DEFORM_START_L4`, `DEFORM_RAMP_L4`, `CURRICULUM_RAMP_L4`, `G_START_L4`, `G_RAMP_L4`, `VAL_GROUP_SIZE_L4` | 1000, 3000, 500, 3000, 3000, 3000, 16 | level-4 schedule (phase C) |
| `TF_MEM_LIMIT_MB_L1` | 42500 | level 1 `TF_MEM_LIMIT_MB` |
| `GPU_POLL` | false | all levels |
| `START_LEVEL` | 4 | 4, 2 or 1. With 2 or 1, `CASCADE_FROM_START` (the source checkpoint dir) is required and must exist if that level starts fresh |
| `L4_SCRIPT` / `L2_SCRIPT` / `L1_SCRIPT` | `$REPO/slurm/r_level{4,2,1}.sh` | test hook |
| `BGR_DRY_RUN=1` | off | validates, prints each level's decision, env and command, then exits 0. Creates nothing and calls no level script, singularity or sbatch |

Levels 2 and 1 otherwise use their script defaults: L2 BN recal 64 with freeze false, fp16 decoder, skip
deform 4:0.15. L1 BN recal 64 with freeze true, float32_fine decoder, pair recompute, lean losses, skips
4:0,2:0.15 / 4:0.15,2:0.15. All levels use d12hs widths, gradnorm 12, Dice + LNCC 1/1/1/1 and G 45/10. Group
sizes are 128/64, 32/16 and 8/4.

## Phase-C schedule vs r_level4.sh defaults

`slurm/sweep_level4.py` `SCHEDULE_C` = `DEFORM_START 3000, DEFORM_RAMP 500, CURRICULUM_RAMP 3000, G_START 3000,
G_RAMP 3000, WARMUP 1000, VAL_EVERY 500`. Every config also gets `VAL_GROUP_SIZE = min(16, DEFORM_N_small = 32)
= 16`. The r_level4.sh defaults are identical (`WARMUP 1000`, the four 3000s, `DEFORM_RAMP 500`,
`VAL_GROUP_SIZE 16`, `VAL_EVERY 500`, `STEPS 16000`, N 128 / deform 64, d12hs). The only difference is
`WD`: the r_level4.sh default is 0, the sweep passed 1e-1, and the cascade passes 1e-1. The cascade sets the
schedule values explicitly anyway, so a later change of the r_level4.sh defaults does not alter the chain. The
level-4 MAIN line built from the cascade env (level-script dry run) has `--weight-decay 1e-1 --warmup-steps
1000 --task-start-step deform:3000 --task-start-ramp-steps 500 --curriculum-ramp-steps 3000
--global-start-step 3000 --global-ramp-steps 3000 --val-group-size 16`, and ends with `--group-config B
--lambda-smooth <S4>` after the script's own `--lambda-smooth 0.5`. argparse keeps the last value.

## Exit code and checker gate

Each level script runs main.py, then `tests/gpu/check_mixed_run.py`. It exits with main.py's code if that is
non-zero, otherwise with the checker's code (end of r_level{4,2,1}.sh: `exit "$rc"` / `exit "$crc"`). So exit
0 means training finished and the checker passed, and the cascade does not run the checker again. After exit 0,
the cascade also requires `$RESULTS/<run>/ckpt/last/checkpoint` with `model_checkpoint_path` `last-<STEPS>`
(exit 3 otherwise). It then writes `$RESULTS/<run>/cascade_done.json` atomically (tmp + mv). The marker holds
level, run name, steps, checkpoint, job id, date, git rev, elapsed seconds, synth p, lambda_smooth and
cascade_from.

On a non-zero level exit the cascade prints `CASCADE FAILED at level k (...)` to stderr and exits with that
code. The next level never starts. Each level is wrapped in `=== level k start <date>` and
`=== level k end <date> rc <rc> elapsed <h:mm> (<s> s)` lines. The last line is
`CASCADE DONE <date>: <per-level summary>`.

## Resume and skip

- Marker present and ckpt/last at `last-<STEPS>`: the level is skipped and `=== level k skip` is printed.
- ckpt/last present but no marker (interrupted run, or the checker failed after main.py finished): the level
  script is rerun with the same RUN_NAME and env. It resumes from ckpt/last, and its CASCADE_FROM guard is
  skipped on resume. A marker with a ckpt/last that differs from `last-<STEPS>` (e.g. `STEPS_*` raised) is
  also rerun.
- Nothing present: the level starts fresh.

Resubmitting the identical sbatch line therefore continues the chain. Do not `git pull` in the cluster clone
while a chain runs, because later levels would run the pulled level scripts (each marker records its git rev).
Expected RTX 8000 wall time from earlier runs: L4 ~14.5-15 h (r4cfgB0 14h35), L2 ~21-25 h (config B smoothness
sweep ~24.5 h), L1 ~15 h. That totals ~51-55 h of the 72 h limit.

## Dry run (`BGR_DRY_RUN=1 S4=0.5 S2=0.5 S1=0.5`, local WSL, `PROJ` = the cluster project path, `REPO` = local checkout)

```
cascade | host Jay | job manual | repo rev 3bc4bafcd1d81cb5ea12857433671b764a2a7dd7 | 2026-11-05T23:55:18-05:00
levels 4 2 1 | S4 0.5 S2 0.5 S1 0.5 | synth p 0.5 | lr 3e-4 wd 1e-1 | group args '--group-config B' | results /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results
unset before each level: RUN_NAME STEPS VAL_EVERY CKPT_EVERY WARMUP LR WD OPTIMIZER N DEFORM_N TASK_RATIO DEFORM_START DEFORM_RAMP CURRICULUM CURRICULUM_RAMP G_START G_RAMP G_ROT G_TRANS TF_MEM_LIMIT_MB GRAD_CKPT PAIR_RECOMPUTE MIN_PAIRS MAX_PAIRS LEAN_LOSSES LOWRES_METHOD GRAD_BALANCE VAL_PAIR_CHUNK VAL_IMAGE_CHUNK VAL_GROUP_SIZE DEFORM_DECODER FILTERS DECODER_WIDTHS INIT CASCADE_FROM BN_RECAL_BATCHES BN_RECAL_FREEZE BN_RECAL_VALIDATE SKIP_LINEAR SKIP_DEFORM PROJECT_STEPS PROJECT_VAL_EVERY EXTRA_ARGS BGR_DRY_RUN FP16_ACT_CLIP FP16_ACT_CLIP_SCOPE DEFORM_DECODER_DTYPE BN_EPSILON_OVERRIDE SYNTH_PROB SYNTH_JITTER SYNTH_DEVICE SYNTH_LABEL_SET GPU_POLL MIN_TASK_STEPS
LEVEL 4: fresh (no ckpt/last in /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results/r4_cfgB_synth05)
ENV L4: PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg RESULTS=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results RUN_NAME=r4_cfgB_synth05 STEPS=16000 PROJECT_STEPS=16000 LR=3e-4 WD=1e-1 VAL_EVERY=500 WARMUP=1000 DEFORM_START=3000 DEFORM_RAMP=500 CURRICULUM_RAMP=3000 G_START=3000 G_RAMP=3000 VAL_GROUP_SIZE=16 SYNTH_PROB=0.5 GPU_POLL=false EXTRA_ARGS=--group-config B --lambda-smooth 0.5
CMD L4: PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg RESULTS=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results RUN_NAME=r4_cfgB_synth05 STEPS=16000 PROJECT_STEPS=16000 LR=3e-4 WD=1e-1 VAL_EVERY=500 WARMUP=1000 DEFORM_START=3000 DEFORM_RAMP=500 CURRICULUM_RAMP=3000 G_START=3000 G_RAMP=3000 VAL_GROUP_SIZE=16 SYNTH_PROB=0.5 GPU_POLL=false EXTRA_ARGS='--group-config B --lambda-smooth 0.5' bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/slurm/r_level4.sh
LEVEL 2: fresh (no ckpt/last in /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results/r2_cfgB_synth05)
ENV L2: PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg RESULTS=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results RUN_NAME=r2_cfgB_synth05 STEPS=15000 PROJECT_STEPS=15000 LR=3e-4 WD=1e-1 VAL_EVERY=500 CASCADE_FROM=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results/r4_cfgB_synth05/ckpt/last SYNTH_PROB=0.5 GPU_POLL=false EXTRA_ARGS=--group-config B --lambda-smooth 0.5
CMD L2: PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg RESULTS=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results RUN_NAME=r2_cfgB_synth05 STEPS=15000 PROJECT_STEPS=15000 LR=3e-4 WD=1e-1 VAL_EVERY=500 CASCADE_FROM=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results/r4_cfgB_synth05/ckpt/last SYNTH_PROB=0.5 GPU_POLL=false EXTRA_ARGS='--group-config B --lambda-smooth 0.5' bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/slurm/r_level2.sh
LEVEL 1: fresh (no ckpt/last in /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results/r1_cfgB_synth05)
ENV L1: PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg RESULTS=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results RUN_NAME=r1_cfgB_synth05 STEPS=4000 PROJECT_STEPS=4000 LR=3e-4 WD=1e-1 VAL_EVERY=500 CASCADE_FROM=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results/r2_cfgB_synth05/ckpt/last TF_MEM_LIMIT_MB=42500 SYNTH_PROB=0.5 GPU_POLL=false EXTRA_ARGS=--group-config B --lambda-smooth 0.5
CMD L1: PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg RESULTS=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results RUN_NAME=r1_cfgB_synth05 STEPS=4000 PROJECT_STEPS=4000 LR=3e-4 WD=1e-1 VAL_EVERY=500 CASCADE_FROM=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results/r2_cfgB_synth05/ckpt/last TF_MEM_LIMIT_MB=42500 SYNTH_PROB=0.5 GPU_POLL=false EXTRA_ARGS='--group-config B --lambda-smooth 0.5' bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/slurm/r_level1.sh
DRY RUN: nothing created, no level script called
rc 0
```

On the cluster the script paths are `.../GroupwiseReg/brain-group-rad/slurm/r_level{4,2,1}.sh`. Missing S1 gives
`ERROR: set S1 (--lambda-smooth of level 1); S4, S2 and S1 have no default (...)` and rc 2.

Extra check: the three level scripts were run with `BGR_DRY_RUN=1` and the cascade's per-level env. All exit 0.
The MAIN lines contain the expected flags (`--synth-prob 0.5`, `--weight-decay 1e-1`, `--group-config B
--lambda-smooth <S>` last, L1 `--gpu-memory-limit-mb 42500 --deform-decoder-dtype float32_fine`). The CHECK lines
contain `--expect-synth-prob 0.5` and `--project-steps <STEPS>`.

## Tests

- `python -m unittest tests.test_slurm_scripts` in WSL `bgr_preproc38` (conda at `~/miniconda3`): **Ran 55
  tests, OK** (7.8 s). That includes 9 new tests:
  - Static: header, policy lines and exactly 11 `#SBATCH` lines; level-env coverage; `bash -n`.
  - Missing or bad S4/S2/S1, bad START_LEVEL / SYNTH_PROB, or duplicate run names give exit 2.
  - Dry run: three levels with the right RUN_NAME, CASCADE_FROM, STEPS / PROJECT_STEPS, SYNTH_PROB,
    `--group-config B --lambda-smooth <S>` and the level-4 schedule vars. No level-4 schedule vars at
    level 2. Nothing is created. The real-only variant is also checked.
  - START_LEVEL 2/1 without, or with a non-existent, CASCADE_FROM_START gives exit 2. CASCADE_FROM_START
    without START_LEVEL gives exit 2. START_LEVEL 2 runs only L2 then L1.
  - Stub chain: three markers in order 4 -> 2 -> 1 with the right content. A stale `N`, `FILTERS=`, `INIT`
    or `EXTRA_ARGS` never reaches a level. An identical rerun skips all three levels.
  - A failing level 2 stops before level 1 with its exit code (7). The resubmission continues at level 2.
    An exit 0 with a short checkpoint gives exit 3 and no marker.
  - A ckpt/last without a marker (L4 at last-16000, L2 at last-750) is rerun.
- `bash -n slurm/r_cascade.sh`: OK. There are no CR characters in the three changed files.
- The full suite was not run, because only shell scripts and the TF-free slurm test module changed.

## Submit lines (after commit, push and `git pull` in the clone; ask Jay first; at most two active jobs)

`env VAR=... sbatch` works from both bash and tcsh login shells. Before submitting:

```
cd /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad
mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
squeue -u jn85
```

Optional check first (prints everything, runs nothing): prefix the same line with `BGR_DRY_RUN=1` and replace
`sbatch ...` with `bash slurm/r_cascade.sh`.

(a) Mixed model (SYNTH_PROB 0.5, run names r4/r2/r1_cfgB_synth05):

```
env S4=<S4> S2=<S2> S1=<S1> sbatch --job-name=bgr-cascade-syn slurm/r_cascade.sh
```

(b) Jay's real-only config B chain (names are examples; Jay picks them):

```
env S4=<S4> S2=<S2> S1=<S1> SYNTH_PROB=0 RUN_NAME_L4=r4_cfgB_final RUN_NAME_L2=r2_cfgB_final RUN_NAME_L1=r1_cfgB_final sbatch --job-name=bgr-cascade-real slurm/r_cascade.sh
```

Logs go to `.../GroupwiseReg/logs/<job-name>-<job>.out/.err`. Resume after a time-out or failure: resubmit the
identical line.
