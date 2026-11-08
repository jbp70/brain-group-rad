#!/bin/bash
#SBATCH --job-name=bgr-fgi-mem
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Fine-grid frozen integration: real-trainer GPU memory / step-time check of
# --frozen-integration-grid current against native (plan .agents/tasks/fine-grid-integration/plan.md
# D6 / D7). The measure tool under-predicts the trainer by 10-15 % and has no DALI, so every config
# is the REAL r_level2.sh / r_level1.sh recipe: the main.py argv is taken from the level script's
# BGR_DRY_RUN=1 'MAIN: python ...' line (the recipe cannot drift), then run here in one singularity
# process per config.
#
# Configs, in this order (dtype column of status.tsv = the grid):
#   l2_native   r_level2.sh, CASCADE_FROM=$L4_SOURCE
#   l2_current  r_level2.sh, CASCADE_FROM=$L4_SOURCE, FROZEN_INTEGRATION_GRID=current
#   l1_native   r_level1.sh, CASCADE_FROM=$OUT/runs/l2_native/ckpt/last
#   l1_current  r_level1.sh, CASCADE_FROM=$OUT/runs/l2_current/ckpt/last, FROZEN_INTEGRATION_GRID=current
#               (level 1 would inherit current from its source anyway; the explicit value is checked
#               against it by main.py)
# Per config env: RESULTS=$OUT/runs RUN_NAME=<name> STEPS ($STEPS, 16) VAL_EVERY ($VAL_EVERY, 8:
# two validations) CKPT_EVERY=0 (one final ckpt/last) BN_RECAL_BATCHES ($BN_RECAL_BATCHES, 4)
# PROJECT_STEPS=$STEPS EXTRA_ARGS="$GROUP_ARGS --lambda-smooth $SMOOTH"; everything else = the level
# script defaults (d12hs widths, L2 N 32 / deform 16 / max pairs 64 with memory growth, L1 N 8 /
# deform 4 under TF_MEM_LIMIT_MB 42500, float32_fine decoder, BN recal + freeze). The level scripts
# get a clean environment (env -i), so a stale recipe variable of the submitting shell never leaks.
# An L1 line whose L2 source run left no ckpt/last/checkpoint gets rc -2 (skipped_source) and is
# not run. L1 checkpoints are deleted after their config; L2 checkpoints are kept (L1 sources).
# A per-config nvidia-smi poll (every 5 s, host side) writes $OUT/runs/<name>/gpu_poll.csv.
#
# Expected (plan D6, NOT measured): TF peak L2 (deform N 16) current <= native + 0.2 GB; L1
# (deform N 4) current = native + 0 to 1.5 GB (the 1 mm integrations run per image in a
# tf.map_fn, ~1-1.5 GB transient outside the tape); deform step time L2 < +0.5 s, L1 +1-3 s.
# References: L2 deform N 16 d12hs fp16 27.2 GiB (m11 job 10528435), r2 runs 27.8 GiB; L1 r1 runs
# 37.3-38.9 GB under the 42500 MB limit (L1 lr sweep). Pass rule: L1 current within 42500 MB with
# >= 1 GB headroom, gpu_poll.csv max memory.used <= memory.total - 512 MiB.
# Time estimate (RTX 8000, not measured): L2 config ~10 min (startup, recal 4 batches, 16 steps,
# two validations), L1 config ~30-35 min (startup ~5 min, two level-1 validations ~8 min each)
# -> ~2 h of the 3 h limit. CONFIG_TIMEOUT (3600 s) caps every config.
#
# Env (all overridable): PROJ, REPO ($PROJ/brain-group-rad), DATA ($PROJ/data/npy_v2), SIF,
# SCRATCH_ROOT (/scratch), L4_SOURCE ($PROJ/results/r4_cfgB_dice1_smooth2_lr3e-4_wd1e-1/ckpt/last;
# a checkpoint directory or prefix), GROUP_ARGS ('--group-config B'; set but empty = none), SMOOTH
# (10), STEPS (16), VAL_EVERY (8), BN_RECAL_BATCHES (4), CONFIG_TIMEOUT (3600 s), ONLY (comma list
# of config names, empty = all; unknown names exit 2), BGR_DRY_RUN (1 = validate, print one
# 'CONFIG <name>: <env> bash slurm/r_level<k>.sh' line per selected config and exit 0; nothing is
# created or run).
# Output: $PROJ/scratch/fgi/<job>/ with runs/<name>/ (config.json, logs/metrics.jsonl, gpu_poll.csv),
# runs/<name>.log, status.tsv (name level dtype widths n max_pairs rc elapsed_s log) and
# fgi_memory.json + a table from scripts/collect_memory_runs.py (gpu_peak_gib, mean deform step s).
# OOM / timeout configs are results, not job failures: the exit code is the collector's.
#
# DO NOT submit without Jay's approval; at most two active jobs (squeue -u jn85). Account qtim only.
# Before sbatch, the log folder must exist (SLURM does not create it):
#   mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
#   cd /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad && git pull --ff-only
#   sbatch slurm/fgi_memory.sh
set -uo pipefail

PROJ="${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}"
REPO="${REPO:-$PROJ/brain-group-rad}"
DATA="${DATA:-$PROJ/data/npy_v2}"
SIF="${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}"
JOB_ID="${SLURM_JOB_ID:-manual}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"
L4_SOURCE="${L4_SOURCE:-$PROJ/results/r4_cfgB_dice1_smooth2_lr3e-4_wd1e-1/ckpt/last}"
GROUP_ARGS="${GROUP_ARGS---group-config B}"
SMOOTH="${SMOOTH:-10}"
STEPS="${STEPS:-16}"
VAL_EVERY="${VAL_EVERY:-8}"
BN_RECAL_BATCHES="${BN_RECAL_BATCHES:-4}"
CONFIG_TIMEOUT="${CONFIG_TIMEOUT:-3600}"
ONLY="${ONLY:-}"
BGR_DRY_RUN="${BGR_DRY_RUN:-}"
OUT="$PROJ/scratch/fgi/$JOB_ID"
RUNS="$OUT/runs"
STATUS="$OUT/status.tsv"

die() { echo "ERROR: $*" >&2; exit 2; }
need_uint() {   # need_uint <var name> <minimum>
    local v="${!1}"
    case "$v" in ''|*[!0-9]*|0?*) die "$1 must be an integer >= $2, got '$v'";; esac
    [ "$v" -ge "$2" ] || die "$1 must be an integer >= $2, got '$v'"
}
need_uint STEPS 2
need_uint VAL_EVERY 0
need_uint BN_RECAL_BATCHES 0
need_uint CONFIG_TIMEOUT 1
FLOAT_RE='^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
[[ "$SMOOTH" =~ $FLOAT_RE ]] || die "SMOOTH must be a non-negative number, got '$SMOOTH'"
[ -d "$L4_SOURCE" ] || [ -f "$L4_SOURCE.index" ] || \
    die "L4_SOURCE $L4_SOURCE is neither a checkpoint directory nor a prefix"
[ -f "$REPO/slurm/r_level2.sh" ] && [ -f "$REPO/slurm/r_level1.sh" ] || \
    die "$REPO/slurm/r_level{2,1}.sh not found (REPO=$REPO)"

ALL_CONFIGS="l2_native l2_current l1_native l1_current"
CONFIGS="$ALL_CONFIGS"
if [ -n "$ONLY" ]; then
    only_list=" ${ONLY//,/ } "
    for o in $only_list; do
        case " $ALL_CONFIGS " in *" $o "*) ;; *) die "unknown ONLY name '$o' (known: $ALL_CONFIGS)";; esac
    done
    CONFIGS=""
    for c in $ALL_CONFIGS; do
        case "$only_list" in *" $c "*) CONFIGS="$CONFIGS $c";; esac
    done
    CONFIGS="${CONFIGS# }"
fi

# config_env <config>: sets LVL (2 | 1), GRID (native | current), SRC (the CASCADE_FROM path) and
# CENV (the level script's env assignments, one array element each).
config_env() {
    local c="$1"
    LVL="${c%%_*}"
    LVL="${LVL#l}"
    GRID="${c#*_}"
    if [ "$LVL" = 2 ]; then SRC="$L4_SOURCE"; else SRC="$RUNS/l2_$GRID/ckpt/last"; fi
    CENV=("RESULTS=$RUNS" "RUN_NAME=$c" "STEPS=$STEPS" "VAL_EVERY=$VAL_EVERY" "CKPT_EVERY=0"
          "BN_RECAL_BATCHES=$BN_RECAL_BATCHES" "PROJECT_STEPS=$STEPS"
          "EXTRA_ARGS=${GROUP_ARGS:+$GROUP_ARGS }--lambda-smooth $SMOOTH" "CASCADE_FROM=$SRC")
    if [ "$GRID" = current ]; then CENV+=("FROZEN_INTEGRATION_GRID=current"); fi
    if [ "$LVL" = 2 ]; then CN=16; else CN=4; fi
}

SAFE_RE='^[A-Za-z0-9._/:,+=-]+$'
shq() { if [[ "$1" =~ $SAFE_RE ]]; then printf '%s' "$1"; else printf "'%s'" "${1//\'/\'\\\'\'}"; fi; }
env_text() {    # the CENV assignments as a copy-paste prefix
    local e out=""
    for e in "${CENV[@]}"; do out="$out ${e%%=*}=$(shq "${e#*=}")"; done
    printf '%s' "${out# }"
}

if [ "$BGR_DRY_RUN" = "1" ]; then
    for c in $CONFIGS; do
        config_env "$c"
        echo "CONFIG $c: $(env_text) bash slurm/r_level$LVL.sh"
    done
    exit 0
fi

mkdir -p "$RUNS" "$PROJ/scratch/keras" "$PROJ/scratch/cuda_cache"
export TMPDIR="$SCRATCH_ROOT/$JOB_ID"
mkdir -p "$TMPDIR"
export KERAS_HOME="$PROJ/scratch/keras"
export CUDA_CACHE_PATH="$PROJ/scratch/cuda_cache"
export PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_TMPDIR="$TMPDIR"
export SINGULARITYENV_KERAS_HOME="$KERAS_HOME"
export SINGULARITYENV_CUDA_CACHE_PATH="$CUDA_CACHE_PATH"
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1

cd "$REPO" || die "cannot cd to $REPO"
echo "host $(hostname) | job $JOB_ID | repo $(git rev-parse --short HEAD 2>/dev/null || echo '?') | $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
echo "configs: $CONFIGS"
echo "L4 source $L4_SOURCE | steps $STEPS | val every $VAL_EVERY | bn recal $BN_RECAL_BATCHES | smooth $SMOOTH | group args '$GROUP_ARGS' | config timeout $CONFIG_TIMEOUT s | out $OUT"

POLL_DEV="${CUDA_VISIBLE_DEVICES:-0}"
POLL_DEV="${POLL_DEV%%,*}"
POLL_PID=""
stop_poll() {
    if [ -n "$POLL_PID" ]; then
        kill "$POLL_PID" 2>/dev/null || true
        wait "$POLL_PID" 2>/dev/null || true
        POLL_PID=""
    fi
}
trap stop_poll EXIT

printf 'name\tlevel\tdtype\twidths\tn\tmax_pairs\trc\telapsed_s\tlog\n' > "$STATUS"
row() {   # row <name> <rc> <elapsed_s> (uses LVL GRID CN)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$LVL" "$GRID" "d12hs" "$CN" "64" "$2" "$3" \
        "$RUNS/$1.log" >> "$STATUS"
}

for c in $CONFIGS; do
    config_env "$c"
    log="$RUNS/$c.log"
    if [ "$LVL" = 1 ] && [ ! -f "$SRC/checkpoint" ]; then
        echo "=== $c skipped: level-2 source $SRC has no checkpoint"
        row "$c" -2 0
        continue
    fi
    # The recipe: the level script's dry-run MAIN line, in a clean environment.
    dry="$(env -i PATH="$PATH" HOME="${HOME:-/tmp}" PROJ="$PROJ" REPO="$REPO" DATA="$DATA" SIF="$SIF" \
        SCRATCH_ROOT="$SCRATCH_ROOT" SLURM_JOB_ID="$JOB_ID" SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-3}" \
        "${CENV[@]}" BGR_DRY_RUN=1 bash "$REPO/slurm/r_level$LVL.sh" 2>&1)"
    drc=$?
    main_line="$(printf '%s\n' "$dry" | grep '^MAIN: python ' | head -n 1)"
    if [ "$drc" -ne 0 ] || [ -z "$main_line" ]; then
        printf '%s\n' "$dry" > "$log"
        echo "=== $c: level-script dry run failed (rc $drc)"
        tail -n 5 "$log" || true
        if [ "$drc" -eq 0 ]; then drc=2; fi   # no MAIN line
        row "$c" "$drc" 0
        continue
    fi
    read -r -a ARGV <<< "${main_line#MAIN: python }"
    mkdir -p "$RUNS/$c"
    nvidia-smi -i "$POLL_DEV" --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader,nounits \
        -l 5 > "$RUNS/$c/gpu_poll.csv" 2>/dev/null &
    POLL_PID=$!
    t0=$SECONDS
    echo "=== $c start $(date -Is) (timeout $CONFIG_TIMEOUT s): $(env_text) bash slurm/r_level$LVL.sh"
    timeout -k 60 "$CONFIG_TIMEOUT" singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast \
        "$SIF" python "${ARGV[@]}" > "$log" 2>&1 </dev/null
    rc=$?
    stop_poll
    secs=$((SECONDS - t0))
    row "$c" "$rc" "$secs"
    echo "=== $c rc $rc $secs s"
    if [ "$rc" -ne 0 ]; then tail -n 5 "$log" || true; fi
    if [ "$LVL" = 1 ]; then rm -rf "$RUNS/$c/ckpt"; fi
done

crc=0
singularity exec -B /autofs -B /cluster -B /space "$SIF" python scripts/collect_memory_runs.py \
    --status "$STATUS" --runs "$RUNS" --out "$OUT/fgi_memory.json" --skip-steps 2 --job "$JOB_ID" \
    </dev/null || crc=$?
echo "finished fgi_memory collector exit code $crc after $SECONDS s $(date -Is)"
exit "$crc"
