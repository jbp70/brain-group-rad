#!/bin/bash
#SBATCH --job-name=bgr-cascade
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Full 3-level cascade (level 4 -> level 2 -> level 1) in ONE allocation: runs slurm/r_level4.sh,
# slurm/r_level2.sh and slurm/r_level1.sh one after another with `bash` (their #SBATCH lines are
# then plain comments), so the chain waits in the queue once instead of three times. Report:
# .agents/tasks/cascade-chain/report.md.
#
# Recipe = the final real-only config B recipe (Jay): lr 3e-4, AdamW wd 1e-1, d12hs widths (the
# level-script defaults; FILTERS / DECODER_WIDTHS are never passed), '--group-config B' and a
# per-level smoothness weight --lambda-smooth S4 / S2 / S1 (appended after the scripts'
# --lambda-smooth 0.5, so argparse takes ours). Level 4: 16000 steps with the phase-C schedule of
# r4sw_C_s16000_lr3e-4_wd1e-1_n128_d64 (slurm/sweep_level4.py SCHEDULE_C: WARMUP 1000, deform from
# 3000 with a 500-step ramp, curriculum ramp 3000, G from 3000 ramped over 3000, VAL_GROUP_SIZE 16,
# VAL_EVERY 500; identical to the r_level4.sh defaults, set explicitly here). Level 2: 15000 steps,
# level 1: 4000 steps, both with their script defaults (BN recal 64, freeze false / true, level-1
# float32_fine decoder, pair recompute, lean losses) plus LR / WD / VAL_EVERY 500; level 1 with
# TF_MEM_LIMIT_MB 42500. PROJECT_STEPS = the level's steps. Wiring (Jay: always the LAST
# checkpoint): level 2 CASCADE_FROM = $RESULTS/$RUN_NAME_L4/ckpt/last, level 1 CASCADE_FROM =
# $RESULTS/$RUN_NAME_L2/ckpt/last.
#
# Frozen code: the chain never runs the clone directly. At the start of a FRESH chain (no chain
# state file, and no cascade_done.json / ckpt/last for any level of this chain) it records the
# clone's HEAD and extracts that commit with `git archive` into
#   CODE_ROOT = $PROJ/scratch/cascade_code/<RUN_NAME_L4>/<first 12 hex of the sha>
# (stamp file .cascade_extract_ok with the full sha written after a successful extract; a CODE_ROOT
# without a matching stamp is removed and extracted again). All three level scripts run as
# `bash $CODE_ROOT/slurm/r_level<k>.sh` with REPO=$CODE_ROOT, so main.py and
# tests/gpu/check_mixed_run.py come from the frozen copy; data, results, logs and the container
# image stay where they are. Jay can therefore commit and `git pull` in the clone while a chain runs.
# A fresh chain refuses to start (exit 2) if `git status --porcelain --untracked-files=no` of the
# clone is non-empty (uncommitted changes to tracked files would silently not be part of the
# frozen copy); ALLOW_DIRTY=1 freezes HEAD anyway (without those changes) and prints a WARNING.
# Chain state file: $RESULTS/cascade_chains/<RUN_NAME_L4>.json (commit, code root, run names,
# S4/S2/S1, SYNTH_PROB, start level, created date / job, last job).
# Resume (state file present): the RECORDED commit and code root are reused, never the clone's new
# HEAD; a missing or partial code root (scratch cleaned) is re-extracted from the clone with
# `git archive <recorded sha>` (needs that commit in the clone). If S4 / S2 / S1 / SYNTH_PROB /
# RUN_NAME_L4/L2/L1 differ from the recorded values -> exit 2 (no recipe mixing mid-chain), unless
# FORCE_RECIPE_CHANGE=1 (then the state file is rewritten with the new values; the old one is kept
# as <RUN_NAME_L4>.json.prev-<date>). No state file but runs of this chain exist -> exit 2 unless
# CODE_COMMIT is set.
# CODE_COMMIT=<sha or ref>: pin the commit of a fresh chain (must exist in the clone; the dirty
# check is skipped because the working tree is not used). On resume it must equal the recorded
# commit, unless FORCE_RECIPE_CHANGE=1 (switches the remaining levels to that commit).
# After the chain finished, $PROJ/scratch/cascade_code/<RUN_NAME_L4>/ can be deleted (the markers
# and the state file keep the sha). Note: config.json's git_commit is null for runs of the frozen
# copy (not a git checkout); the sha is in cascade_done.json and the chain state file.
#
# Env (defaults in brackets):
#   S4, S2, S1      REQUIRED: --lambda-smooth per level (positive numbers; exit 2 if missing)
#   SYNTH_PROB      [0.5] synthetic-contrast probability at every level (0 = real-only chain)
#   RUN_NAME_L4 / RUN_NAME_L2 / RUN_NAME_L1  [r4_cfgB_synth05 / r2_cfgB_synth05 / r1_cfgB_synth05]
#                   (RUN_NAME_L4 is also the chain id: state file and code root)
#   GROUP_ARGS      ['--group-config B'] appended to every level's EXTRA_ARGS (set but empty = none)
#   EXTRA_ARGS_L4 / EXTRA_ARGS_L2 / EXTRA_ARGS_L1  [empty] appended after GROUP_ARGS --lambda-smooth
#                   (fine-grid chain: EXTRA_ARGS_L2='--frozen-integration-grid current'; level 1 inherits it)
#   STEPS_L4 / STEPS_L2 / STEPS_L1  [16000 / 15000 / 4000]
#   LR [3e-4], WD [1e-1], VAL_EVERY [500]  (all levels)
#   WARMUP_L4 [1000], DEFORM_START_L4 [3000], DEFORM_RAMP_L4 [500], CURRICULUM_RAMP_L4 [3000],
#   G_START_L4 [3000], G_RAMP_L4 [3000], VAL_GROUP_SIZE_L4 [16]  (level-4 schedule)
#   TF_MEM_LIMIT_MB_L1 [42500], GPU_POLL [false] (all levels)
#   START_LEVEL     [4] 4 | 2 | 1; with 2 or 1, CASCADE_FROM_START (the source checkpoint dir of
#                   that level, e.g. $RESULTS/<L4 run>/ckpt/last) is required
#   CODE_COMMIT     [empty] pin the frozen commit (see above)
#   ALLOW_DIRTY     [empty] 1 = freeze HEAD of a dirty clone with a WARNING
#   FORCE_RECIPE_CHANGE [empty] 1 = allow a recipe / commit change on resume
#   L4_SCRIPT / L2_SCRIPT / L1_SCRIPT  [$CODE_ROOT/slurm/r_level{4,2,1}.sh] (test hook; when set it
#                   overrides the frozen path, REPO is still $CODE_ROOT)
#   PROJ, REPO (the clone), DATA, RESULTS, SIF, SCRATCH_ROOT: PROJ / DATA / RESULTS / SIF /
#                   SCRATCH_ROOT pass through to the level scripts; they get REPO=$CODE_ROOT.
#   BGR_DRY_RUN=1   validate, print the commit that would be frozen / reused, the code root, the
#                   dirty-check result, whether extraction would happen, and every level's env,
#                   command and skip / resume decision, exit 0 (a dirty clone is reported, not
#                   fatal); creates nothing, calls no level script / git archive / singularity.
# Every other recipe variable of the level scripts (LEVEL_ENV_KEYS below: N, DEFORM_N, INIT,
# FILTERS, SKIP_*, BN_RECAL_*, MIN_TASK_STEPS, ...) is UNSET before each level, so a stale value in
# the submitting shell never leaks into a level; change a level's recipe with EXTRA_ARGS_L<k>
# (main.py flags, appended last).
#
# Gate: a level script exits with main.py's code if non-zero, else with the code of
# tests/gpu/check_mixed_run.py, so exit 0 = training finished AND checker PASS. After exit 0 this
# script also requires $RESULTS/<run>/ckpt/last/checkpoint at last-<STEPS> and then writes
# $RESULTS/<run>/cascade_done.json (level, steps, checkpoint, job id, date, git rev = the frozen
# commit, code root, elapsed). On failure it prints which level failed and exits with that level's
# code (3 if the checkpoint check fails); the next level never starts.
# Resume rule: a level with cascade_done.json AND ckpt/last at last-<STEPS> is skipped; a level
# with a ckpt/last but no marker is rerun with the same RUN_NAME and env (the level scripts resume
# from ckpt/last, the CASCADE_FROM guard is skipped on resume); otherwise it starts fresh. If the
# 72 h limit hits, resubmit the IDENTICAL line: the chain continues where it stopped, with the
# recorded commit (at most CKPT_EVERY 250 steps of the running level are lost).
# Expected wall time on the RTX 8000: L4 ~14.5-15 h, L2 ~21-25 h, L1 ~15 h -> ~51-55 h.
#
# Cluster rules: DO NOT submit without Jay's approval; at most two active jobs (squeue -u jn85,
# Jay's own jobs count); account qtim only; one GPU. Before sbatch, the log folder must exist:
#   mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
# Usage (from the clone; `env VAR=... sbatch` works in bash and tcsh login shells):
#   mixed model (SYNTH_PROB 0.5, default run names):
#     env S4=<S4> S2=<S2> S1=<S1> sbatch slurm/r_cascade.sh
#   real-only config B chain:
#     env S4=<S4> S2=<S2> S1=<S1> SYNTH_PROB=0 RUN_NAME_L4=r4_cfgB_final RUN_NAME_L2=r2_cfgB_final RUN_NAME_L1=r1_cfgB_final sbatch --job-name=bgr-cascade-real slurm/r_cascade.sh
#   start at level 2 from an existing level-4 run:
#     env S4=<S4> S2=<S2> S1=<S1> START_LEVEL=2 CASCADE_FROM_START=<results>/<L4 run>/ckpt/last sbatch slurm/r_cascade.sh
#   check first (prints everything, runs nothing): add BGR_DRY_RUN=1 and run with bash instead of sbatch.
set -uo pipefail

PROJ="${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}"
REPO="${REPO:-$PROJ/brain-group-rad}"
RESULTS="${RESULTS:-$PROJ/results}"
JOB_ID="${SLURM_JOB_ID:-manual}"
S4="${S4:-}"
S2="${S2:-}"
S1="${S1:-}"
SYNTH_PROB="${SYNTH_PROB:-0.5}"
RUN_NAME_L4="${RUN_NAME_L4:-r4_cfgB_synth05}"
RUN_NAME_L2="${RUN_NAME_L2:-r2_cfgB_synth05}"
RUN_NAME_L1="${RUN_NAME_L1:-r1_cfgB_synth05}"
GROUP_ARGS="${GROUP_ARGS---group-config B}"
EXTRA_ARGS_L4="${EXTRA_ARGS_L4:-}"
EXTRA_ARGS_L2="${EXTRA_ARGS_L2:-}"
EXTRA_ARGS_L1="${EXTRA_ARGS_L1:-}"
STEPS_L4="${STEPS_L4:-16000}"
STEPS_L2="${STEPS_L2:-15000}"
STEPS_L1="${STEPS_L1:-4000}"
LR="${LR:-3e-4}"
WD="${WD:-1e-1}"
VAL_EVERY="${VAL_EVERY:-500}"
WARMUP_L4="${WARMUP_L4:-1000}"
DEFORM_START_L4="${DEFORM_START_L4:-3000}"
DEFORM_RAMP_L4="${DEFORM_RAMP_L4:-500}"
CURRICULUM_RAMP_L4="${CURRICULUM_RAMP_L4:-3000}"
G_START_L4="${G_START_L4:-3000}"
G_RAMP_L4="${G_RAMP_L4:-3000}"
VAL_GROUP_SIZE_L4="${VAL_GROUP_SIZE_L4:-16}"
TF_MEM_LIMIT_MB_L1="${TF_MEM_LIMIT_MB_L1:-42500}"
GPU_POLL="${GPU_POLL:-false}"
START_LEVEL="${START_LEVEL:-4}"
CASCADE_FROM_START="${CASCADE_FROM_START:-}"
CODE_COMMIT="${CODE_COMMIT:-}"
ALLOW_DIRTY="${ALLOW_DIRTY:-}"
FORCE_RECIPE_CHANGE="${FORCE_RECIPE_CHANGE:-}"
L4_SCRIPT="${L4_SCRIPT:-}"      # empty -> $CODE_ROOT/slurm/r_level4.sh (resolved after the freeze)
L2_SCRIPT="${L2_SCRIPT:-}"
L1_SCRIPT="${L1_SCRIPT:-}"
BGR_DRY_RUN="${BGR_DRY_RUN:-}"

# Recipe env of r_level{4,2,1}.sh, unset before every level (only the values set below reach a
# level). Not listed (passed through): PROJ REPO DATA RESULTS SIF SCRATCH_ROOT SLURM_JOB_ID
# SLURM_CPUS_PER_TASK CUDA_VISIBLE_DEVICES (REPO is set to $CODE_ROOT in every level's env).
LEVEL_ENV_KEYS=(RUN_NAME STEPS VAL_EVERY CKPT_EVERY WARMUP LR WD OPTIMIZER N DEFORM_N TASK_RATIO
    DEFORM_START DEFORM_RAMP CURRICULUM CURRICULUM_RAMP G_START G_RAMP G_ROT G_TRANS TF_MEM_LIMIT_MB
    GRAD_CKPT PAIR_RECOMPUTE MIN_PAIRS MAX_PAIRS LEAN_LOSSES LOWRES_METHOD GRAD_BALANCE VAL_PAIR_CHUNK
    VAL_IMAGE_CHUNK VAL_GROUP_SIZE DEFORM_DECODER FILTERS DECODER_WIDTHS INIT CASCADE_FROM
    BN_RECAL_BATCHES BN_RECAL_FREEZE BN_RECAL_VALIDATE SKIP_LINEAR SKIP_DEFORM PROJECT_STEPS
    PROJECT_VAL_EVERY EXTRA_ARGS BGR_DRY_RUN FP16_ACT_CLIP FP16_ACT_CLIP_SCOPE DEFORM_DECODER_DTYPE
    BN_EPSILON_OVERRIDE SYNTH_PROB SYNTH_JITTER SYNTH_DEVICE SYNTH_LABEL_SET GPU_POLL MIN_TASK_STEPS
    FROZEN_INTEGRATION_GRID)

die() { echo "ERROR: $*" >&2; exit 2; }
# non-negative decimal integer without leading zeros ($(( )) would read 08 as octal)
need_uint() { case "$2" in ''|*[!0-9]*|0?*) die "$1 must be a non-negative integer, got '$2'";; esac; }
need_pos_int() { need_uint "$1" "$2"; [ "$2" -gt 0 ] || die "$1 must be > 0, got '$2'"; }
FLOAT_RE='^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
need_float() { [[ "$2" =~ $FLOAT_RE ]] || die "$1 must be a non-negative number, got '$2'"; }
is_pos() { awk -v x="$1" 'BEGIN { exit !(x + 0 > 0) }'; }
NAME_RE='^[A-Za-z0-9._-]+$'

for v in S4 S2 S1; do
    [ -n "${!v}" ] || die "set $v (--lambda-smooth of level ${v#S}); S4, S2 and S1 have no default (e.g. env S4=10 S2=10 S1=10 sbatch slurm/r_cascade.sh)"
    [[ "${!v}" =~ $FLOAT_RE ]] && is_pos "${!v}" || die "$v must be a positive number, got '${!v}'"
done
need_float SYNTH_PROB "$SYNTH_PROB"
awk -v x="$SYNTH_PROB" 'BEGIN { exit !(x + 0 <= 1) }' || die "SYNTH_PROB must be <= 1, got '$SYNTH_PROB'"
for v in LR WD; do need_float "$v" "${!v}"; done
for v in STEPS_L4 STEPS_L2 STEPS_L1 VAL_EVERY VAL_GROUP_SIZE_L4 TF_MEM_LIMIT_MB_L1; do need_pos_int "$v" "${!v}"; done
for v in WARMUP_L4 DEFORM_START_L4 DEFORM_RAMP_L4 CURRICULUM_RAMP_L4 G_START_L4 G_RAMP_L4; do need_uint "$v" "${!v}"; done
case "$GPU_POLL" in true|false) ;; *) die "GPU_POLL must be true or false, got '$GPU_POLL'" ;; esac
case "$START_LEVEL" in 4|2|1) ;; *) die "START_LEVEL must be 4, 2 or 1, got '$START_LEVEL'" ;; esac
for v in ALLOW_DIRTY FORCE_RECIPE_CHANGE; do
    case "${!v}" in ''|0|1) ;; *) die "$v must be empty, 0 or 1, got '${!v}'" ;; esac
done
for v in RUN_NAME_L4 RUN_NAME_L2 RUN_NAME_L1; do
    [[ "${!v}" =~ $NAME_RE ]] || die "$v must match [A-Za-z0-9._-]+, got '${!v}'"
done
if [ "$RUN_NAME_L4" = "$RUN_NAME_L2" ] || [ "$RUN_NAME_L4" = "$RUN_NAME_L1" ] || [ "$RUN_NAME_L2" = "$RUN_NAME_L1" ]; then
    die "RUN_NAME_L4, RUN_NAME_L2 and RUN_NAME_L1 must differ ($RUN_NAME_L4 / $RUN_NAME_L2 / $RUN_NAME_L1)"
fi
if [ "$START_LEVEL" != 4 ]; then
    [ -n "$CASCADE_FROM_START" ] || die "START_LEVEL=$START_LEVEL needs CASCADE_FROM_START (the source checkpoint dir, e.g. \$RESULTS/<level-$((START_LEVEL * 2)) run>/ckpt/last)"
elif [ -n "$CASCADE_FROM_START" ]; then
    die "CASCADE_FROM_START is only used with START_LEVEL 2 or 1 (level 4 trains from scratch)"
fi

LEVELS=()
for k in 4 2 1; do
    if [ "$k" -le "$START_LEVEL" ]; then LEVELS+=("$k"); fi
done

run_name_of() { local v="RUN_NAME_L$1"; echo "${!v}"; }
steps_of() { local v="STEPS_L$1"; echo "${!v}"; }
script_of() { local v="L$1_SCRIPT"; if [ -n "${!v}" ]; then echo "${!v}"; else echo "$CODE_ROOT/slurm/r_level$1.sh"; fi; }
smooth_of() { local v="S$1"; echo "${!v}"; }
run_dir_of() { echo "$RESULTS/$(run_name_of "$1")"; }
# CASCADE_FROM of level k (empty at level 4)
source_of() {
    if [ "$1" = 4 ]; then echo ""
    elif [ "$1" = "$START_LEVEL" ]; then echo "$CASCADE_FROM_START"
    elif [ "$1" = 2 ]; then echo "$(run_dir_of 4)/ckpt/last"
    else echo "$(run_dir_of 2)/ckpt/last"
    fi
}
# checkpoint name in <run>/ckpt/last/checkpoint (e.g. last-16000), empty if none
last_ckpt_of() {
    local f
    f="$(run_dir_of "$1")/ckpt/last/checkpoint"
    [ -f "$f" ] || return 0
    sed -n 's/^model_checkpoint_path: *"\(.*\)"[[:space:]]*$/\1/p' "$f" | head -n 1 | sed 's#.*/##'
}
# skip | resume | fresh (+ reason in DECISION_WHY)
decide() {
    local k="$1" dir ck
    dir="$(run_dir_of "$k")"
    ck="$(last_ckpt_of "$k")"
    if [ -f "$dir/cascade_done.json" ] && [ "$ck" = "last-$(steps_of "$k")" ]; then
        DECISION=skip; DECISION_WHY="cascade_done.json present and ckpt/last at $ck"
    elif [ -n "$ck" ]; then
        DECISION=resume
        if [ -f "$dir/cascade_done.json" ]; then
            DECISION_WHY="cascade_done.json present but ckpt/last at $ck != last-$(steps_of "$k"); rerun with the same RUN_NAME"
        else
            DECISION_WHY="ckpt/last at $ck, no cascade_done.json; rerun with the same RUN_NAME (the level script resumes)"
        fi
    else
        DECISION=fresh; DECISION_WHY="no ckpt/last in $dir"
    fi
}
# LEVEL_ENV = the KEY=VALUE list of level k
level_env() {
    local k="$1" steps extra
    steps="$(steps_of "$k")"
    extra="$GROUP_ARGS --lambda-smooth $(smooth_of "$k")"
    local ev="EXTRA_ARGS_L$k"
    if [ -n "${!ev}" ]; then extra="$extra ${!ev}"; fi
    extra="${extra# }"
    LEVEL_ENV=("PROJ=$PROJ" "REPO=$CODE_ROOT" "RESULTS=$RESULTS" "RUN_NAME=$(run_name_of "$k")" "STEPS=$steps"
               "PROJECT_STEPS=$steps" "LR=$LR" "WD=$WD" "VAL_EVERY=$VAL_EVERY")
    case "$k" in
        4) LEVEL_ENV+=("WARMUP=$WARMUP_L4" "DEFORM_START=$DEFORM_START_L4" "DEFORM_RAMP=$DEFORM_RAMP_L4"
                       "CURRICULUM_RAMP=$CURRICULUM_RAMP_L4" "G_START=$G_START_L4" "G_RAMP=$G_RAMP_L4"
                       "VAL_GROUP_SIZE=$VAL_GROUP_SIZE_L4") ;;
        2) LEVEL_ENV+=("CASCADE_FROM=$(source_of 2)") ;;
        1) LEVEL_ENV+=("CASCADE_FROM=$(source_of 1)" "TF_MEM_LIMIT_MB=$TF_MEM_LIMIT_MB_L1") ;;
    esac
    LEVEL_ENV+=("SYNTH_PROB=$SYNTH_PROB" "GPU_POLL=$GPU_POLL" "EXTRA_ARGS=$extra")
}
SAFE_RE='^[A-Za-z0-9._/:,+=-]+$'
shq() { if [[ "$1" =~ $SAFE_RE ]]; then printf '%s' "$1"; else printf "'%s'" "${1//\'/\'\\\'\'}"; fi; }
print_cmd() {   # a copy-paste line equivalent to what runs (plus the unset of LEVEL_ENV_KEYS)
    local e out=""
    for e in "${LEVEL_ENV[@]}"; do out="$out ${e%%=*}=$(shq "${e#*=}")"; done
    echo "CMD L$1:${out} bash $(shq "$(script_of "$1")")"
}

# ---- frozen code: which commit, where, extract or not (no side effects here)
# GIT_OPTIONAL_LOCKS=0: `git status` must not rewrite the clone's index (dry run creates nothing).
export GIT_OPTIONAL_LOCKS=0
CHAIN_ID="$RUN_NAME_L4"
CHAIN_DIR="$RESULTS/cascade_chains"
CHAIN_STATE="$CHAIN_DIR/$CHAIN_ID.json"
CODE_BASE="$PROJ/scratch/cascade_code/$CHAIN_ID"
STAMP=".cascade_extract_ok"
SHA_RE='^[0-9a-f]{40}$'
resolve_commit() { git -C "$REPO" rev-parse --verify --quiet "$1^{commit}" 2>/dev/null; }
state_get() { sed -n 's/^  "'"$1"'": "\(.*\)",\{0,1\}$/\1/p' "$CHAIN_STATE" | head -n 1; }
CLONE_HEAD="$(git -C "$REPO" rev-parse --verify --quiet 'HEAD^{commit}' 2>/dev/null || echo unknown)"
WARNINGS=()
DIRTY_FAIL=""
STATE_REWRITE=""        # 1 = the state file is rewritten with a backup (forced recipe / commit change)
REC_CREATED_DATE=""
REC_CREATED_JOB=""
REC_MODE=""
REC_CLONE_HEAD=""
REC_ALLOW_DIRTY=""
if [ -f "$CHAIN_STATE" ]; then
    CODE_MODE=resume
    COMMIT="$(state_get commit)"
    CODE_ROOT="$(state_get code_root)"
    [[ "$COMMIT" =~ $SHA_RE ]] || die "chain state $CHAIN_STATE has no valid commit ('$COMMIT')"
    [ -n "$CODE_ROOT" ] || die "chain state $CHAIN_STATE has no code_root"
    REC_CREATED_DATE="$(state_get created_date)"
    REC_CREATED_JOB="$(state_get created_job_id)"
    REC_MODE="$(state_get code_mode)"
    REC_CLONE_HEAD="$(state_get clone_head)"
    REC_ALLOW_DIRTY="$(state_get allow_dirty)"
    MISMATCH=""
    for kv in "s4=$S4" "s2=$S2" "s1=$S1" "synth_prob=$SYNTH_PROB" "run_name_l4=$RUN_NAME_L4" \
              "run_name_l2=$RUN_NAME_L2" "run_name_l1=$RUN_NAME_L1"; do
        key="${kv%%=*}"; val="${kv#*=}"; rec="$(state_get "$key")"
        if [ "$rec" != "$val" ]; then MISMATCH="$MISMATCH $key recorded '$rec' now '$val';"; fi
    done
    if [ -n "$MISMATCH" ]; then
        if [ "$FORCE_RECIPE_CHANGE" = 1 ]; then
            WARNINGS+=("FORCE_RECIPE_CHANGE=1: recipe differs from $CHAIN_STATE:$MISMATCH continuing with the new values")
            STATE_REWRITE=1
        else
            die "recipe differs from the chain state $CHAIN_STATE:$MISMATCH resubmit with the recorded values or set FORCE_RECIPE_CHANGE=1"
        fi
    fi
    if [ -n "$CODE_COMMIT" ]; then
        c="$(resolve_commit "$CODE_COMMIT")" || die "CODE_COMMIT '$CODE_COMMIT' is not a commit of the clone $REPO"
        if [ "$c" != "$COMMIT" ]; then
            if [ "$FORCE_RECIPE_CHANGE" = 1 ]; then
                WARNINGS+=("FORCE_RECIPE_CHANGE=1: switching the chain from the recorded commit $COMMIT to CODE_COMMIT $c")
                COMMIT="$c"
                CODE_ROOT="$CODE_BASE/${COMMIT:0:12}"
                STATE_REWRITE=1
                REC_MODE="pin (forced change on resume)"
                REC_CLONE_HEAD="$CLONE_HEAD"
            else
                die "CODE_COMMIT $c differs from the commit $COMMIT recorded in $CHAIN_STATE; drop CODE_COMMIT or set FORCE_RECIPE_CHANGE=1"
            fi
        fi
    fi
    CODE_WHY="resume: commit and code root from $CHAIN_STATE"
    DIRTY_MSG="skipped (resume: the clone's working tree is not used)"
else
    HAS_RUNS=""
    for k in "${LEVELS[@]}"; do
        d="$(run_dir_of "$k")"
        if [ -f "$d/cascade_done.json" ] || [ -f "$d/ckpt/last/checkpoint" ]; then HAS_RUNS="$HAS_RUNS $(run_name_of "$k")"; fi
    done
    if [ -n "$HAS_RUNS" ] && [ -z "$CODE_COMMIT" ]; then
        die "no chain state $CHAIN_STATE but runs of this chain exist ($HAS_RUNS ); the commit they ran is unknown: set CODE_COMMIT=<sha> (e.g. git_rev in their cascade_done.json or git_commit in config.json)"
    fi
    if [ -n "$CODE_COMMIT" ]; then
        CODE_MODE=pin
        COMMIT="$(resolve_commit "$CODE_COMMIT")" || die "CODE_COMMIT '$CODE_COMMIT' is not a commit of the clone $REPO"
        CODE_WHY="CODE_COMMIT pin '$CODE_COMMIT'"
        DIRTY_MSG="skipped (CODE_COMMIT pin: the working tree is not used)"
    else
        CODE_MODE=fresh
        [ "$CLONE_HEAD" != unknown ] || die "REPO $REPO is not a git clone (git rev-parse HEAD failed)"
        COMMIT="$CLONE_HEAD"
        CODE_WHY="fresh chain: clone HEAD"
        DIRTY_LIST="$(git -C "$REPO" status --porcelain --untracked-files=no)" || die "git status failed in $REPO"
        if [ -z "$DIRTY_LIST" ]; then
            DIRTY_MSG="clean"
        else
            n_dirty="$(printf '%s\n' "$DIRTY_LIST" | wc -l | tr -d ' ')"
            DIRTY_MSG="DIRTY ($n_dirty tracked file(s) with uncommitted changes: $(printf '%s\n' "$DIRTY_LIST" | head -n 10 | sed 's/^...//' | tr '\n' ' '))"
            if [ "$ALLOW_DIRTY" = 1 ]; then
                WARNINGS+=("ALLOW_DIRTY=1: the clone has uncommitted changes; freezing HEAD $COMMIT WITHOUT them")
            else
                DIRTY_FAIL=1
            fi
        fi
    fi
    CODE_ROOT="$CODE_BASE/${COMMIT:0:12}"
fi
if [ -f "$CODE_ROOT/$STAMP" ] && [ "$(head -n 1 "$CODE_ROOT/$STAMP")" = "$COMMIT" ]; then
    EXTRACT=present; EXTRACT_MSG="present ($CODE_ROOT/$STAMP matches)"
else
    if [ -e "$CODE_ROOT" ]; then
        EXTRACT=redo; EXTRACT_MSG="partial copy without a matching stamp: would remove it and git archive $COMMIT again"
    else
        EXTRACT=needed; EXTRACT_MSG="needed: git archive $COMMIT | tar -x -C $CODE_ROOT"
    fi
    resolve_commit "$COMMIT" >/dev/null || die "commit $COMMIT is not in the clone $REPO, cannot extract $CODE_ROOT"
fi

for k in "${LEVELS[@]}"; do
    hv="L${k}_SCRIPT"
    if [ -n "${!hv}" ]; then
        [ -f "${!hv}" ] || die "level-$k script ${!hv} not found (L${k}_SCRIPT)"
    elif [ "$EXTRACT" = present ]; then
        [ -f "$CODE_ROOT/slurm/r_level$k.sh" ] || die "level-$k script $CODE_ROOT/slurm/r_level$k.sh not found in the frozen copy"
    else
        git -C "$REPO" cat-file -e "$COMMIT:slurm/r_level$k.sh" 2>/dev/null \
            || die "commit $COMMIT has no slurm/r_level$k.sh"
    fi
done
# A fresh first level needs its source now (later levels get theirs from the chain).
if [ "$START_LEVEL" != 4 ]; then
    decide "$START_LEVEL"
    if [ "$DECISION" = fresh ] && [ ! -d "$CASCADE_FROM_START" ] && [ ! -f "$CASCADE_FROM_START.index" ]; then
        die "CASCADE_FROM_START $CASCADE_FROM_START is neither a checkpoint directory nor a prefix"
    fi
fi
if [ -n "$DIRTY_FAIL" ] && [ "$BGR_DRY_RUN" != "1" ]; then
    die "the clone $REPO has uncommitted changes to tracked files; commit them (or stash them) first, or set ALLOW_DIRTY=1 to freeze HEAD $COMMIT without them. $DIRTY_MSG"
fi

echo "cascade | host $(hostname) | job $JOB_ID | $(date -Is)"
echo "code: frozen $COMMIT at $CODE_ROOT (clone HEAD $CLONE_HEAD)"
echo "code source: $CODE_MODE ($CODE_WHY) | chain state $CHAIN_STATE"
echo "dirty check: $DIRTY_MSG"
echo "extract: $EXTRACT_MSG"
for w in "${WARNINGS[@]+"${WARNINGS[@]}"}"; do echo "WARNING: $w" >&2; echo "WARNING: $w"; done
echo "levels ${LEVELS[*]} | S4 $S4 S2 $S2 S1 $S1 | synth p $SYNTH_PROB | lr $LR wd $WD | group args '$GROUP_ARGS' | results $RESULTS"
echo "unset before each level: ${LEVEL_ENV_KEYS[*]}"

if [ "$BGR_DRY_RUN" = "1" ]; then
    for k in 4 2 1; do
        if [ "$k" -gt "$START_LEVEL" ]; then
            echo "LEVEL $k: not run (START_LEVEL $START_LEVEL)"
            continue
        fi
        decide "$k"
        level_env "$k"
        echo "LEVEL $k: $DECISION ($DECISION_WHY)"
        echo "ENV L$k: ${LEVEL_ENV[*]}"
        print_cmd "$k"
    done
    if [ -n "$DIRTY_FAIL" ]; then
        echo "DRY RUN NOTE: a real run would exit 2 (dirty clone; commit first or set ALLOW_DIRTY=1)"
    fi
    echo "DRY RUN: nothing created, no level script called"
    exit 0
fi

# ---- freeze (extract) and record the chain state
extract_code() {
    if [ "$EXTRACT" = redo ]; then
        case "$CODE_ROOT" in
            */scratch/cascade_code/*/*) rm -rf "$CODE_ROOT" || return 1 ;;
            *) echo "refusing to remove $CODE_ROOT (not under */scratch/cascade_code/)" >&2; return 1 ;;
        esac
    fi
    mkdir -p "$CODE_ROOT" || return 1
    git -C "$REPO" archive "$COMMIT" | tar -x -C "$CODE_ROOT" || return 1
    printf '%s\n' "$COMMIT" > "$CODE_ROOT/$STAMP"
}
if [ "$EXTRACT" != present ]; then
    echo "extracting $COMMIT into $CODE_ROOT $(date -Is)"
    extract_code || { echo "ERROR: git archive of $COMMIT into $CODE_ROOT failed" >&2; exit 2; }
    for k in "${LEVELS[@]}"; do
        [ -f "$(script_of "$k")" ] || { echo "ERROR: $(script_of "$k") missing after the extract" >&2; exit 2; }
    done
fi
write_state() {
    local tmp="$CHAIN_STATE.tmp.$$" now
    now="$(date -Is)"
    mkdir -p "$CHAIN_DIR" || return 1
    if [ -n "$STATE_REWRITE" ] && [ -f "$CHAIN_STATE" ]; then
        cp -p "$CHAIN_STATE" "$CHAIN_STATE.prev-$(date +%Y%m%dT%H%M%S)" || return 1
    fi
    {
        echo "{"
        printf '  "chain_id": "%s",\n' "$CHAIN_ID"
        printf '  "commit": "%s",\n' "$COMMIT"
        printf '  "code_root": "%s",\n' "$CODE_ROOT"
        printf '  "code_mode": "%s",\n' "${REC_MODE:-$CODE_MODE}"
        printf '  "clone_head": "%s",\n' "${REC_CLONE_HEAD:-$CLONE_HEAD}"
        printf '  "run_name_l4": "%s",\n' "$RUN_NAME_L4"
        printf '  "run_name_l2": "%s",\n' "$RUN_NAME_L2"
        printf '  "run_name_l1": "%s",\n' "$RUN_NAME_L1"
        printf '  "s4": "%s",\n' "$S4"
        printf '  "s2": "%s",\n' "$S2"
        printf '  "s1": "%s",\n' "$S1"
        printf '  "synth_prob": "%s",\n' "$SYNTH_PROB"
        printf '  "start_level": "%s",\n' "$START_LEVEL"
        printf '  "allow_dirty": "%s",\n' "${REC_ALLOW_DIRTY:-${ALLOW_DIRTY:-0}}"
        printf '  "created_date": "%s",\n' "${REC_CREATED_DATE:-$now}"
        printf '  "created_job_id": "%s",\n' "${REC_CREATED_JOB:-$JOB_ID}"
        printf '  "last_job_id": "%s",\n' "$JOB_ID"
        printf '  "last_start_date": "%s"\n' "$now"
        echo "}"
    } > "$tmp" && mv -f "$tmp" "$CHAIN_STATE"
}
write_state || { echo "ERROR: cannot write the chain state $CHAIN_STATE" >&2; exit 2; }
echo "chain state: $CHAIN_STATE"

write_marker() {   # write_marker <level> <checkpoint> <elapsed s>
    local k="$1" dir tmp
    dir="$(run_dir_of "$k")"
    tmp="$dir/.cascade_done.json.tmp"
    printf '{"level": %s, "run_name": "%s", "steps": %s, "checkpoint": "%s", "job_id": "%s", "date": "%s", "git_rev": "%s", "code_root": "%s", "elapsed_s": %s, "synth_prob": "%s", "lambda_smooth": "%s", "cascade_from": "%s"}\n' \
        "$k" "$(run_name_of "$k")" "$(steps_of "$k")" "$2" "$JOB_ID" "$(date -Is)" "$COMMIT" "$CODE_ROOT" "$3" \
        "$SYNTH_PROB" "$(smooth_of "$k")" "$(source_of "$k")" > "$tmp" && mv -f "$tmp" "$dir/cascade_done.json"
}
fmt_time() { printf '%dh%02dm' $(( $1 / 3600 )) $(( $1 % 3600 / 60 )); }

SUMMARY=()
for k in "${LEVELS[@]}"; do
    decide "$k"
    if [ "$DECISION" = skip ]; then
        echo "=== level $k skip $(date -Is): $(run_name_of "$k") ($DECISION_WHY)"
        SUMMARY+=("L$k $(run_name_of "$k") skipped")
        continue
    fi
    level_env "$k"
    echo "=== level $k start $(date -Is): $(run_name_of "$k") ($DECISION: $DECISION_WHY)"
    print_cmd "$k"
    t0=$SECONDS
    rc=0
    (
        unset "${LEVEL_ENV_KEYS[@]}"
        export "${LEVEL_ENV[@]}"
        exec bash "$(script_of "$k")"
    ) </dev/null || rc=$?
    el=$(( SECONDS - t0 ))
    echo "=== level $k end $(date -Is) rc $rc elapsed $(fmt_time "$el") ($el s)"
    if [ "$rc" -ne 0 ]; then
        echo "CASCADE FAILED at level $k ($(run_name_of "$k")): level script exit code $rc; later levels not started" >&2
        exit "$rc"
    fi
    ck="$(last_ckpt_of "$k")"
    if [ "$ck" != "last-$(steps_of "$k")" ]; then
        echo "CASCADE FAILED at level $k ($(run_name_of "$k")): exit 0 but ckpt/last is '${ck:-missing}', expected last-$(steps_of "$k"); later levels not started" >&2
        exit 3
    fi
    write_marker "$k" "$ck" "$el" || { echo "CASCADE FAILED at level $k: cannot write $(run_dir_of "$k")/cascade_done.json" >&2; exit 3; }
    echo "level $k done: $(run_dir_of "$k")/cascade_done.json ($ck)"
    SUMMARY+=("L$k $(run_name_of "$k") $ck $(fmt_time "$el")")
done
echo "CASCADE DONE $(date -Is): ${SUMMARY[*]}"
exit 0
