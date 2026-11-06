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

Resubmitting the identical sbatch line therefore continues the chain. Since v2 the chain runs from a frozen copy
of the code, so `git pull` in the cluster clone during a chain is safe (section "Frozen code (v2)").
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

## Frozen code (v2)

Changed: `slurm/r_cascade.sh` and `tests/test_slurm_scripts.py`. `tests/run_tests_wsl.sh` already runs `bash -n`
on `slurm/r_cascade.sh`, so it is unchanged. Nothing committed, pushed or submitted.

### Behaviour

The chain no longer runs the code in the clone. It freezes one commit into scratch and runs every level from
there, following the pattern of `.agents/tasks/m9-level1-nan/m9_nan_probe.sh`. Jay can therefore commit and
`git pull` in the cluster clone while a 72 h chain runs.

- **Fresh chain.** A chain is fresh when it has no state file and none of its levels has a `cascade_done.json`
  or a `ckpt/last`. The script records `git -C $REPO rev-parse HEAD`. If
  `git status --porcelain --untracked-files=no` is non-empty, it exits 2 and lists up to 10 changed files.
  This check covers the whole tracked tree, the simplest rule the task allowed. `ALLOW_DIRTY=1` freezes HEAD
  anyway, without the uncommitted changes, and prints `WARNING: ALLOW_DIRTY=1 ...`. The status call runs with
  `GIT_OPTIONAL_LOCKS=0`, so it does not rewrite the clone's index.
- **Freeze.** The commit is extracted to `CODE_ROOT=$PROJ/scratch/cascade_code/<RUN_NAME_L4>/<first 12 hex of
  the sha>` with `git -C $REPO archive <sha> | tar -x -C $CODE_ROOT`, the same command the probe uses. After a
  successful extract, the script writes the stamp file `$CODE_ROOT/.cascade_extract_ok`, which holds the full
  sha. A `CODE_ROOT` with a matching stamp is reused. A `CODE_ROOT` without a matching stamp is a partial
  extract: it is removed and extracted again. The `rm -rf` only runs on paths matching
  `*/scratch/cascade_code/*/*`.
- **Levels.** Each level runs as `bash $CODE_ROOT/slurm/r_level{4,2,1}.sh` with `REPO=$CODE_ROOT` in its env.
  `REPO` is now part of the printed `ENV`/`CMD` lines. The level scripts `cd "$REPO"` and call `main.py` and
  `tests/gpu/check_mixed_run.py` by relative path, so both come from the frozen copy. `PROJ`, `DATA`,
  `RESULTS`, `SIF` and `SCRATCH_ROOT` are unchanged, so data, results, logs and the container image stay where
  they are. The test hooks `L4_SCRIPT`/`L2_SCRIPT`/`L1_SCRIPT`, when set, override the frozen script path.
  `REPO` is still `$CODE_ROOT` in that case.
- **State file.** The state file is `$RESULTS/cascade_chains/<RUN_NAME_L4>.json`. I chose a dedicated directory
  over `$RESULTS/<RUN_NAME_L4>/` because with `START_LEVEL=2`/`1` the level-4 run dir belongs to another run.
  The file holds one string field per line: `chain_id`, `commit`, `code_root`, `code_mode` (fresh | pin),
  `clone_head` (at creation), `run_name_l4/l2/l1`, `s4`, `s2`, `s1`, `synth_prob`, `start_level`,
  `allow_dirty`, `created_date`, `created_job_id`, `last_job_id` and `last_start_date`. It is written atomically
  (tmp + mv) after the extract and before the first level. On every resume, only `last_job_id` and
  `last_start_date` change.
- **Resume.** When the state file exists, the recorded commit and code root are reused, never the clone's new
  HEAD, and there is no dirty check. If the code root is missing or has no matching stamp (for example, scratch
  was cleaned), the recorded commit is extracted again from the clone. That needs the commit to exist in the
  clone; otherwise the script exits 2.
- **Recipe guard.** On resume, if `S4`/`S2`/`S1`/`SYNTH_PROB`/`RUN_NAME_L4/L2/L1` differ from the recorded
  strings, the script exits 2 and names each field with its old and new value. `FORCE_RECIPE_CHANGE=1`
  continues with a WARNING, keeps the old state as `<RUN_NAME_L4>.json.prev-<YYYYmmddTHHMMSS>` and rewrites
  the state with the new values. The comparison is exact on the strings, so `0.5` and `.5` count as different.
- **Runs without a state file.** If no state file exists but runs of this chain do (a marker or `ckpt/last`),
  the commit they ran is unknown, so the script exits 2 and asks for `CODE_COMMIT`. The message points to
  `git_rev` in the markers and `git_commit` in config.json.
- **`CODE_COMMIT=<sha or ref>`.** On a fresh chain, this pins the commit. It must resolve in the clone
  (`rev-parse --verify <x>^{commit}`), and the dirty check is skipped. On resume, it must equal the recorded
  commit. With `FORCE_RECIPE_CHANGE=1`, the remaining levels switch to the new commit and its own code root.
- **Markers.** `cascade_done.json` records `git_rev` as the frozen commit and also gains `code_root`.
- **Dry run.** `BGR_DRY_RUN=1` prints `code: frozen ...`, `code source:` (fresh | pin | resume, plus the state
  file path), `dirty check:` (clean | DIRTY (files) | skipped (...)) and `extract:` (present | needed: git
  archive ... | partial copy ...). A dirty clone is reported as `DRY RUN NOTE: a real run would exit 2 ...` and
  the dry run still exits 0, so the full plan stays visible. The dry run creates nothing and calls neither
  `git archive` nor a level script. When the copy does not exist yet, it checks the level scripts with
  `git cat-file -e <sha>:slurm/r_level<k>.sh`.
- **Start line.** Every run prints `code: frozen <sha> at <CODE_ROOT> (clone HEAD <sha2>)`.
- **Cleanup.** After the chain finishes, `$PROJ/scratch/cascade_code/<RUN_NAME_L4>/` can be deleted. The
  markers and the state file keep the sha.

### Caveats

- In the frozen copy, the level scripts' `repo rev` line prints `?`, and config.json's `git_commit` is `null`.
  This is because `runtime/checkpoint._git_commit` runs `git rev-parse` next to the code, and the copy is not a
  git checkout. The sha is in `cascade_done.json` and the state file. Nothing reads `git_commit`: I checked
  `check_mixed_run.py`, `main.py` and `runtime/`.
- `git archive` writes LFS pointers, not the LFS content. The only LFS-tracked file is
  `preprocessing/tests/__init__.py`, and training does not use it.

### Env (new)

| Variable | Default | Effect |
|---|---|---|
| `CODE_COMMIT` | empty | pin the frozen commit (fresh chain). On resume it must equal the recorded commit unless `FORCE_RECIPE_CHANGE=1` |
| `ALLOW_DIRTY` | empty | `1` = freeze HEAD of a dirty clone, with a WARNING |
| `FORCE_RECIPE_CHANGE` | empty | `1` = allow a different S4/S2/S1/SYNTH_PROB/run names (or CODE_COMMIT) on resume |
| `L4_SCRIPT` / `L2_SCRIPT` / `L1_SCRIPT` | `$CODE_ROOT/slurm/r_level{4,2,1}.sh` | test hook (was `$REPO/...`) |

### Tests

`python -m unittest tests.test_slurm_scripts` in WSL `bgr_preproc38` (`source ~/miniconda3/etc/profile.d/conda.sh`):
**Ran 62 tests, OK** (26 s). `bash -n slurm/r_cascade.sh`: OK.

`TestCascadeRuns` now builds a temp git repo in `setUp`. It holds committed stub `slurm/r_level{4,2,1}.sh` that
log `REPO`, `$0` and a `TAG`, plus a `main.py`. The class is skipped without bash or git. The existing tests
were adapted in two places. `test_dry_run` expects the frozen script paths and `REPO=<code root>`.
`test_ckpt_without_marker_is_rerun` now first expects exit 2 without a state file, then passes `CODE_COMMIT=HEAD`.
`test_header` checks the new header text and that the old `git pull` warning is gone. New tests:

- `test_frozen_fresh_chain_runs_from_code_root`: the stubs see `REPO=CODE_ROOT` and run from
  `CODE_ROOT/slurm/`. The test also checks the stamp, the state-file fields and the markers' `git_rev` /
  `code_root`. With the hook stubs set, `REPO` is still the code root.
- `test_resume_ignores_new_clone_commit`: level 2 fails, the test makes a new commit, then resubmits. Levels 2
  and 1 run the v1 copy, no copy of the new sha is made, and `last_job_id` is updated.
- `test_dirty_clone`: a modified tracked file gives exit 2 and creates nothing. The dry run reports DIRTY and
  exits 0. `ALLOW_DIRTY=1` runs with a WARNING, and the uncommitted change is absent from the copy.
- `test_code_commit_pin`: a bad sha gives exit 2. A short sha of an older commit runs that commit despite a
  dirty tree.
- `test_recipe_change_on_resume`: changing S2, SYNTH_PROB, RUN_NAME_L1 or CODE_COMMIT gives exit 2.
  `FORCE_RECIPE_CHANGE=1` proceeds with the new S2 on the recorded commit and writes one `.prev-` backup.
- `test_missing_or_partial_code_root_is_reextracted`: the dry run reports `needed` or `partial copy`, then the
  real run extracts the recorded sha again.
- `test_frozen_dry_run_creates_nothing`: the dry run prints the commit, code root, `clean` and `needed`, and
  creates nothing (the temp repo stays clean). After a run plus a new commit, it reports
  `resume / present / skipped`.

### Local dry run (`BGR_DRY_RUN=1 S4=0.5 S2=0.5 S1=0.5 REPO=<local checkout>`, WSL, PROJ = cluster path)

The clone shows DIRTY only because of this step's uncommitted edits. The per-level ENV/CMD lines are the v1
lines plus `REPO=<code root>`, with the script paths under the code root. Only the L4 lines are shown here:

```
cascade | host Jay | job manual | 2026-11-06T00:41:04-05:00
code: frozen 6445bb27fd65ed7b6ce98f6d6e8a908a26688b6e at /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/scratch/cascade_code/r4_cfgB_synth05/6445bb27fd65 (clone HEAD 6445bb27fd65ed7b6ce98f6d6e8a908a26688b6e)
code source: fresh (fresh chain: clone HEAD) | chain state /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results/cascade_chains/r4_cfgB_synth05.json
dirty check: DIRTY (2 tracked file(s) with uncommitted changes: slurm/r_cascade.sh tests/test_slurm_scripts.py )
extract: needed: git archive 6445bb27fd65ed7b6ce98f6d6e8a908a26688b6e | tar -x -C /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/scratch/cascade_code/r4_cfgB_synth05/6445bb27fd65
levels 4 2 1 | S4 0.5 S2 0.5 S1 0.5 | synth p 0.5 | lr 3e-4 wd 1e-1 | group args '--group-config B' | results /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results
unset before each level: RUN_NAME STEPS ... MIN_TASK_STEPS
LEVEL 4: fresh (no ckpt/last in /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results/r4_cfgB_synth05)
ENV L4: PROJ=/autofs/.../GroupwiseReg REPO=/autofs/.../GroupwiseReg/scratch/cascade_code/r4_cfgB_synth05/6445bb27fd65 RESULTS=/autofs/.../GroupwiseReg/results RUN_NAME=r4_cfgB_synth05 STEPS=16000 PROJECT_STEPS=16000 LR=3e-4 WD=1e-1 VAL_EVERY=500 WARMUP=1000 DEFORM_START=3000 DEFORM_RAMP=500 CURRICULUM_RAMP=3000 G_START=3000 G_RAMP=3000 VAL_GROUP_SIZE=16 SYNTH_PROB=0.5 GPU_POLL=false EXTRA_ARGS=--group-config B --lambda-smooth 0.5
CMD L4: ... EXTRA_ARGS='--group-config B --lambda-smooth 0.5' bash /autofs/.../GroupwiseReg/scratch/cascade_code/r4_cfgB_synth05/6445bb27fd65/slurm/r_level4.sh
LEVEL 2: fresh (...)   [CASCADE_FROM=.../results/r4_cfgB_synth05/ckpt/last, .../6445bb27fd65/slurm/r_level2.sh]
LEVEL 1: fresh (...)   [CASCADE_FROM=.../results/r2_cfgB_synth05/ckpt/last, TF_MEM_LIMIT_MB=42500, .../slurm/r_level1.sh]
DRY RUN NOTE: a real run would exit 2 (dirty clone; commit first or set ALLOW_DIRTY=1)
DRY RUN: nothing created, no level script called
rc 0
```

### Submit lines (unchanged)

These are run from the clone after commit, push and `git pull`. Ask Jay first, and keep to at most two active
jobs.

```
env S4=<S4> S2=<S2> S1=<S1> sbatch --job-name=bgr-cascade-syn slurm/r_cascade.sh
env S4=<S4> S2=<S2> S1=<S1> SYNTH_PROB=0 RUN_NAME_L4=r4_cfgB_final RUN_NAME_L2=r2_cfgB_final RUN_NAME_L1=r1_cfgB_final sbatch --job-name=bgr-cascade-real slurm/r_cascade.sh
```

The commit is frozen when the job STARTS, not when it is submitted. While the job is pending, the clone must
stay clean, and HEAD will be whatever it is at start. To freeze exactly the commit you submitted from, add
`CODE_COMMIT=$(git rev-parse HEAD)` to the `env` line; in tcsh, use backticks instead of `$(...)`. A pull in
the clone after the job has started is harmless. To resume, resubmit the identical line: the recorded commit is reused, and a
different S or SYNTH_PROB gives exit 2. The SBATCH header is unchanged (rtx8000, 1 GPU, 3 CPUs, 128G, 72 h,
qtim).
