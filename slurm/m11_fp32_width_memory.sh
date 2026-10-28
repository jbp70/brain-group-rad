#!/bin/bash
#SBATCH --job-name=bgr-m11-fp32w
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# m11: real-trainer memory / step-time grid for a float32 deform decoder (fp32-decoder-width
# FEAT-003, plan .agents/tasks/fp32-decoder-width/plan.md items 6-7, decision D4).
# Jay's decision after the m9 NaN probe (job 10524598: first inf = fp16 AddV2 in the trainable d1
# split group conv, d2 conv max 9640): run the deform decoder in float32 at all levels and, to
# make room, use narrower full-resolution widths; the from-scratch retrain then uses the chosen
# width at levels 4, 2 and 1. This job measures which (dtype, width) fits.
#
# Deform decoder dtype (--deform-decoder-dtype, name tag in parentheses):
#   float16      (f16)     = the AMP policy (today's fp16 decoder; reference)
#   float32      (f32)     = the whole deform decoder in float32 (encoder stays float16)
#   float32_fine (f32fine) = only d2, d1 (+ their upsamples) and the SVF norm/head in float32;
#                            d5..d3 stay float16
# Widths (existing --filters stem,L2..L6 / --decoder-widths D5..D1 flags; models/width_presets.py):
#   default  (no width flags; the ModelConfig default, as the real m7 / m8 sources)
#   d12h     --decoder-widths 128,96,64,16,8                                   (d1 / d2 halved)
#   d12hs    --filters 16,64,128,192,256,384 --decoder-widths 128,96,64,16,8   (+ stem halved)
#   fr50     --filters 16,32,128,192,256,384 --decoder-widths 128,96,64,16,8   (= preset fr50)
# Non-default widths: --cascade-from needs an identical ModelConfig, so the job first builds
# RANDOM-WEIGHT cascade sources at those widths (slurm/m11_random_sources.sh: per width W a
# level-4 run l4_W and a level-2 run l2_W cascading from it, one rigid step each, weights
# meaningless) in $SOURCES_DIR. Level-1 W lines cascade from l2_W/ckpt/last, level-2 W lines from
# l4_W/ckpt/last, through the normal copy path; the frozen stages are then narrow too, as in the
# planned retrain. Memory / time only. If a width's source build failed, its grid lines are
# recorded with rc -2 (status skipped_source) and not run. Default-width lines use the real
# sources L1_SOURCE (m8_level2_gradnorm, holding the m7 level-4 frozen stage) / L2_SOURCE
# (m7_three_task_lncc). Frozen stages always run in float32, whatever the trainable dtype.
#
# Recipe (main.py, one config per singularity process, `timeout` per config, deform-only): AMP,
# --task-ratio deform:1, --global-transform false, --curriculum false, STEPS (8) steps,
# --log-every 1, --val-every 0 (no validation pass; the validation sets are still built),
# --ckpt-every 0 (one final ckpt/last, deleted after the config), --bn-recal-batches 0,
# --resume false, --warmup-steps 0, AdamW lr 3e-4 wd 0, --task-grad-norm 12, Dice + pDice +
# LNCC 1/1/1/1, smoothness 0.5, p 0.1 / 0.1, --min-pairs n, --gradient-checkpointing full,
# --lean-sampler true, --split-group-conv true, --intensity-aug-device tf, --lowres-method tent.
#   L1: --level 1, deform N 4 (4 pairs), --pair-recompute true, --lean-losses true,
#       --gpu-memory-limit-mb 40000, --val-image-chunk 1 --val-pair-chunk 1
#   L2: --level 2, deform N 16, --max-pairs 64, --pair-recompute false, --val-pair-chunk 4
#   L4: --level 4 (from scratch), deform N 64, --max-pairs 64, --pair-recompute false,
#       --val-pair-chunk 16
#   (L2 / L4 use memory growth.) The 20 configs, in this order (<level>_<dtype tag>_<width>):
#   l1_f32fine_default l1_f32_default l1_f32fine_d12h l1_f32_d12h l1_f32fine_d12hs l1_f32_d12hs
#   l1_f32fine_fr50 l1_f32_fr50 l1_f16_default l1_f16_d12h l1_f16_d12hs l1_f16_fr50
#   l2_f32fine_default l2_f32_default l2_f32fine_d12h l2_f32_d12h
#   l4_f32fine_default l4_f32_default l4_f32fine_d12h l4_f32_d12h
# Deform-only means no linear DALI pipelines: TF peaks are representative, the process DALI
# memory is lower than in the three-task run (the 40000 MB TF limit keeps level 1 honest).
#
# Previous numbers: m9 phase 1 tool (job 10516237, stand-in chain, fp16, ONE pair): L1 deform N 4
# default 31.0 GB, f0d1 29.7 GB, fr50 29.3 GB. Trainer smoke 10518446 (fp16, 4 pairs, no lean
# losses): 38.5 GB. Long run 10518952 (fp16, 4 pairs, lean losses): 36.7 GB in the first window.
# Capped job 10524450: L4 deform N 64 27.6 GB / 5.8 s, L2 deform N 16 29.9 GB / 10.6 s (tool).
#
# Time estimate (RTX 8000, not measured): random sources l4 ~2-3 min + l2 ~4-6 min per width ->
# ~20-25 min for d12h, d12hs, fr50. A level-1 config that runs ~9-10 min (startup with DALI,
# level-1 validation sets, model builds and the frozen chain restore ~4-5 min; first deform step
# trace + autotune ~2-3 min; 7 more steps at ~25-45 s), an OOM ~4 min -> 12 L1 configs ~1.8-2.0 h;
# L2 N 16 ~5 min x 4; L4 N 64 ~4 min x 4 -> ~3.0 h of 4 h in total.
# Guard: a config that would start after TIME_BUDGET_S - 300 s (default 13800 - 300 = 3h45m) is
# recorded with rc -1 (status skipped_time) and not run. CONFIG_TIMEOUT (2400 s) caps every config.
# Split rule: if any configs are skipped, rerun them in a new job (results go to a new job
# directory; SOURCES_DIR may point at the first job's sources to skip the rebuild):
#   ONLY=<names, comma-separated> SOURCES_DIR=<first job>/sources sbatch --time=02:00:00 slurm/m11_fp32_width_memory.sh
#
# Env (all overridable): PROJ, REPO ($PROJ/brain-group-rad), DATA ($PROJ/data/npy_v2),
# SRC_RESULTS ($PROJ/results), SIF, SCRATCH_ROOT (/scratch), L1_SOURCE
# ($SRC_RESULTS/m8_level2_gradnorm/ckpt/best), L2_SOURCE ($SRC_RESULTS/m7_three_task_lncc/ckpt/best),
# STEPS (8, >= 3), CONFIG_TIMEOUT (2400 s), TIME_BUDGET_S (13800 s), ONLY (comma list of config
# names, empty = all; unknown names exit 2), SOURCES_DIR ($OUT_DIR/sources), SOURCE_N (4),
# SOURCE_TIMEOUT (1200 s per source run), BGR_DRY_RUN (1 = print the SOURCE / CONFIG / COLLECT
# command lines after validation and exit 0; nothing is created or run). L1_SOURCE / L2_SOURCE
# must be a checkpoint directory or prefix (<prefix>.index) when a selected default-width line
# needs it (exit 2 otherwise, also in a dry run).
# Output: $PROJ/scratch/m11/<job>/ with runs/<name>/ (config.json, logs/metrics.jsonl) and
# runs/<name>.log per config, status.tsv (name level dtype widths n max_pairs rc elapsed_s log),
# sources/, and fp32_width_memory.json + a table from scripts/collect_memory_runs.py (status ok /
# oom / timeout / abort / skipped_time / skipped_source, peak GiB, mean deform step s, params).
# OOM / abort / timeout configs are results, not job failures: the exit code is the collector's.
#
# DO NOT submit without Jay's approval (pre-approved once fp32-decoder-width lands); at most two
# active jobs (squeue -u jn85). Account qtim only.
# Before sbatch, the log folder must exist (SLURM does not create it):
#   mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
#   cd /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad && git pull --ff-only
#   sbatch slurm/m11_fp32_width_memory.sh
set -uo pipefail

PROJ="${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}"
REPO="${REPO:-$PROJ/brain-group-rad}"
DATA="${DATA:-$PROJ/data/npy_v2}"
SRC_RESULTS="${SRC_RESULTS:-$PROJ/results}"
SIF="${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}"
JOB_ID="${SLURM_JOB_ID:-manual}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"
L1_SOURCE="${L1_SOURCE:-$SRC_RESULTS/m8_level2_gradnorm/ckpt/best}"
L2_SOURCE="${L2_SOURCE:-$SRC_RESULTS/m7_three_task_lncc/ckpt/best}"
STEPS="${STEPS:-8}"
CONFIG_TIMEOUT="${CONFIG_TIMEOUT:-2400}"
TIME_BUDGET_S="${TIME_BUDGET_S:-13800}"
ONLY="${ONLY:-}"
BGR_DRY_RUN="${BGR_DRY_RUN:-}"
OUT_DIR="$PROJ/scratch/m11/$JOB_ID"
RUNS="$OUT_DIR/runs"
STATUS="$OUT_DIR/status.tsv"
SOURCES_DIR="${SOURCES_DIR:-$OUT_DIR/sources}"
SOURCE_N="${SOURCE_N:-4}"
SOURCE_TIMEOUT="${SOURCE_TIMEOUT:-1200}"

die() { echo "ERROR: $*" >&2; exit 2; }
need_uint() {   # need_uint <var name> <minimum>
    local v="${!1}"
    case "$v" in ''|*[!0-9]*) die "$1 must be an integer >= $2, got '$v'";; esac
    [ "$v" -ge "$2" ] || die "$1 must be an integer >= $2, got '$v'"
}
need_uint STEPS 3
need_uint CONFIG_TIMEOUT 1
need_uint TIME_BUDGET_S 1
need_uint SOURCE_N 2
need_uint SOURCE_TIMEOUT 1

M11_LIB="$REPO/slurm/m11_random_sources.sh"
[ -f "$M11_LIB" ] || die "$M11_LIB not found (REPO=$REPO)"
# shellcheck source=slurm/m11_random_sources.sh
source "$M11_LIB"    # m11_width_args, m11_source_ok, m11_build_sources (sets no shell options)

ALL_CONFIGS="l1_f32fine_default l1_f32_default l1_f32fine_d12h l1_f32_d12h l1_f32fine_d12hs l1_f32_d12hs
l1_f32fine_fr50 l1_f32_fr50 l1_f16_default l1_f16_d12h l1_f16_d12hs l1_f16_fr50
l2_f32fine_default l2_f32_default l2_f32fine_d12h l2_f32_d12h
l4_f32fine_default l4_f32_default l4_f32fine_d12h l4_f32_d12h"
ALL_CONFIGS="$(echo $ALL_CONFIGS)"   # one line, single spaces

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

# split_name <config>: sets lv (l1/l2/l4), tag (f16/f32/f32fine) and w (width) of <level>_<tag>_<width>.
split_name() {
    lv="${1%%_*}"
    w="${1##*_}"
    tag="${1#*_}"
    tag="${tag%_*}"
}

dtype_of() {
    case "$1" in
        f16) echo float16 ;;
        f32) echo float32 ;;
        f32fine) echo float32_fine ;;
        *) return 2 ;;
    esac
}

# cascade_src <config>: the --cascade-from path of a config (empty at level 4).
cascade_src() {
    local lv tag w
    split_name "$1"
    case "$lv:$w" in
        l4:*) echo "" ;;
        l1:default) echo "$L1_SOURCE" ;;
        l2:default) echo "$L2_SOURCE" ;;
        l1:*) echo "$SOURCES_DIR/l2_$w/ckpt/last" ;;
        l2:*) echo "$SOURCES_DIR/l4_$w/ckpt/last" ;;
    esac
}

# Sources needed by the selected configs.
NEED_L1=0; NEED_L2=0; SOURCE_WIDTHS=""
for c in $CONFIGS; do
    split_name "$c"
    case "$lv:$w" in
        l4:*) ;;
        l1:default) NEED_L1=1 ;;
        l2:default) NEED_L2=1 ;;
        *) case " $SOURCE_WIDTHS " in *" $w "*) ;; *) SOURCE_WIDTHS="${SOURCE_WIDTHS:+$SOURCE_WIDTHS }$w";; esac ;;
    esac
done
ckpt_ok() { [ -d "$1" ] || [ -f "$1.index" ]; }
if [ "$NEED_L1" = 1 ] && ! ckpt_ok "$L1_SOURCE"; then
    die "L1_SOURCE $L1_SOURCE is neither a checkpoint directory nor a prefix"
fi
if [ "$NEED_L2" = 1 ] && ! ckpt_ok "$L2_SOURCE"; then
    die "L2_SOURCE $L2_SOURCE is neither a checkpoint directory nor a prefix"
fi

COMMON=(--data "$DATA" --results "$RUNS" --resume false --task-ratio deform:1 --steps "$STEPS"
        --val-every 0 --ckpt-every 0 --warmup-steps 0 --log-every 1 --log-backend jsonl --val-png false
        --global-transform false --curriculum false --optimizer adamw --lr 3e-4 --weight-decay 0
        --task-grad-norm 12 --lambda-dice-template 1 --lambda-pdice 1 --lambda-ncc-template 1
        --lambda-ncc-pair 1 --lambda-smooth 0.5 --p-pair 0.1 --p-pdice 0.1 --min-pairs n
        --bn-recal-batches 0 --amp true --gradient-checkpointing full --lean-sampler true
        --split-group-conv true --intensity-aug-device tf --lowres-method tent
        --num-threads "${SLURM_CPUS_PER_TASK:-3}" --prefetch 1)

# config_cmd <config>: sets CMD=(python main.py ...), LVL, DTYPE, WIDTH, CN (deform N), CMP (max pairs).
config_cmd() {
    local c="$1" lv tag w wargs src
    split_name "$c"
    WIDTH="$w"
    DTYPE="$(dtype_of "$tag")" || die "bad dtype tag in $c"
    wargs="$(m11_width_args "$WIDTH")" || die "bad width in $c"
    src="$(cascade_src "$c")"
    CMD=(python main.py "${COMMON[@]}")
    case "$lv" in
        l1) LVL=1; CN=4; CMP=0
            CMD+=(--level 1 --cascade-from "$src" --group-size 4 --deform-group-size 4 --pair-recompute true
                  --lean-losses true --gpu-memory-limit-mb 40000 --val-image-chunk 1 --val-pair-chunk 1) ;;
        l2) LVL=2; CN=16; CMP=64
            CMD+=(--level 2 --cascade-from "$src" --group-size 16 --deform-group-size 16 --max-pairs 64
                  --pair-recompute false --val-pair-chunk 4) ;;
        l4) LVL=4; CN=64; CMP=64
            CMD+=(--level 4 --group-size 64 --deform-group-size 64 --max-pairs 64 --pair-recompute false
                  --val-pair-chunk 16) ;;
        *) die "bad level in $c" ;;
    esac
    # shellcheck disable=SC2206  # width flags are split on purpose
    CMD+=($wargs --run-name "$c" --deform-decoder-dtype "$DTYPE")
}

# required_source <config>: the random source run dir a non-default L1 / L2 line needs (else empty).
required_source() {
    local lv tag w
    split_name "$1"
    case "$lv:$w" in
        l4:*|*:default) echo "" ;;
        l1:*) echo "$SOURCES_DIR/l2_$w" ;;
        l2:*) echo "$SOURCES_DIR/l4_$w" ;;
    esac
}

COLLECT=(python scripts/collect_memory_runs.py --status "$STATUS" --runs "$RUNS"
         --out "$OUT_DIR/fp32_width_memory.json" --job "$JOB_ID")

if [ -n "$BGR_DRY_RUN" ]; then
    if [ -n "$SOURCE_WIDTHS" ]; then m11_build_sources || die "source list failed"; fi
    for c in $CONFIGS; do
        config_cmd "$c"
        echo "CONFIG $c: ${CMD[*]}"
    done
    echo "COLLECT: ${COLLECT[*]}"
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
echo "steps $STEPS | config timeout $CONFIG_TIMEOUT s | time budget $TIME_BUDGET_S s | out $OUT_DIR"

printf 'name\tlevel\tdtype\twidths\tn\tmax_pairs\trc\telapsed_s\tlog\n' > "$STATUS"
row() {   # row <name> <rc> <elapsed_s> (uses LVL DTYPE WIDTH CN CMP)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$LVL" "$DTYPE" "$WIDTH" "$CN" "$CMP" "$2" "$3" \
        "$RUNS/$1.log" >> "$STATUS"
}

if [ -n "$SOURCE_WIDTHS" ]; then
    echo "=== random-weight sources ($SOURCE_WIDTHS) -> $SOURCES_DIR $(date -Is)"
    m11_build_sources
    echo "=== sources done after $SECONDS s"
fi

for c in $CONFIGS; do
    config_cmd "$c"
    log="$RUNS/$c.log"
    need="$(required_source "$c")"
    if [ -n "$need" ] && ! m11_source_ok "$need"; then
        echo "=== $c skipped: random source $need missing (see $SOURCES_DIR/status.tsv)"
        row "$c" -2 0
        continue
    fi
    elapsed=$SECONDS
    if [ $((elapsed + 300)) -gt "$TIME_BUDGET_S" ]; then
        echo "=== $c skipped: time budget ($elapsed s elapsed of $TIME_BUDGET_S s)"
        row "$c" -1 0
        continue
    fi
    t=$((TIME_BUDGET_S - elapsed))
    if [ "$t" -gt "$CONFIG_TIMEOUT" ]; then t=$CONFIG_TIMEOUT; fi
    t0=$SECONDS
    echo "=== $c start $(date -Is) (timeout $t s)"
    timeout -k 60 "$t" singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF" \
        "${CMD[@]}" > "$log" 2>&1 </dev/null
    rc=$?
    secs=$((SECONDS - t0))
    row "$c" "$rc" "$secs"
    echo "=== $c rc $rc $secs s"
    if [ "$rc" -ne 0 ]; then tail -n 5 "$log" || true; fi
    rm -rf "$RUNS/$c/ckpt"
done

crc=0
singularity exec -B /autofs -B /cluster -B /space "$SIF" "${COLLECT[@]}" </dev/null || crc=$?
echo "finished m11_fp32_width_memory collector exit code $crc after $SECONDS s $(date -Is)"
exit "$crc"
