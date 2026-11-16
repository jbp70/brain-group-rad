#!/bin/bash
#SBATCH --job-name=bgr-r1scr
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Paper ablation A1 (.agents/tasks/ablation-prereqs/plan.md D6 / D7, report.md): ONE level-1 model
# (full resolution, 160x224x192 at 1 mm) trained FROM SCRATCH with no cascade source (no
# --cascade-from, no frozen stages, no --init-weights), at the GPU-hours of the final 3-level
# cascade. Compared on the LUMIR training-time validation, last checkpoint (Jay's rule).
#
# Recipe = the final level-1 recipe (slurm/r_level1.sh + r_cascade.sh, run r1_cfgB_s6000_sm15_real)
# wherever it applies: d12hs widths (--filters 16,64,128,192,256,384 --decoder-widths
# 128,96,64,16,8), --deform-decoder-dtype float32_fine, GROUP_CONFIG B, SMOOTH 15 (--lambda-smooth),
# Dice + LNCC 1/1 (template and pair), p 0.1 / 0.1, task ratio rigid:affine:deform 1:1:2, N 8 /
# DEFORM_N 4, AdamW lr 3e-4 wd 1e-1 (cosine, --min-lr-factor 1e-3), GRAD_BALANCE gradnorm
# (--task-grad-norm 12), AMP, --split-group-conv true, TF-side intensity augmentation, LOWRES_METHOD
# tent, TF_MEM_LIMIT_MB 42500, GRAD_CKPT full + lean sampler, PAIR_RECOMPUTE true, LEAN_LOSSES true,
# MIN_PAIRS n, MAX_PAIRS 64, VAL_IMAGE_CHUNK 1 / VAL_PAIR_CHUNK 1, validation groups = N / DEFORM_N
# (VAL_GROUP_SIZE empty), G 45 deg / 10 mm. Differences forced by "from scratch" (the rules that
# worked at level 4, slurm/r_level4.sh): no BN recalibration and no freeze (no source statistics;
# BN trains normally, as at level 4), no cascade skip flags, curriculum ON, deform and G DELAYED.
# Schedule = the level-4 phase-C schedule (16000 steps) as fractions of STEPS (integer division):
#   WARMUP STEPS/16, DEFORM_START 3*STEPS/16 (LR ramp DEFORM_RAMP STEPS/32), CURRICULUM_RAMP
#   3*STEPS/16 (15/5 -> 45/15), G_START 3*STEPS/16 ramped over G_RAMP 3*STEPS/16
#   (STEPS 16000 -> 1000 / 3000 / 500 / 3000 / 3000 / 3000, = r_level4.sh). Each is overridable.
# GROUP_CONFIG (A | B | none | E, default B) -> --group-config (A everywhere, B decoders only, E
# stem / encoder only, none no group blocks). NORM (batch | instance, default batch) -> --norm
# instance only when instance (batch = no flag). SMOOTH (default 15) -> --lambda-smooth.
#
# STEPS is REQUIRED (no default): the cosine schedule and the fractions above depend on it, so pick
# it once from the smoke and keep it for every resubmission. GPU-hours budget (sacct, 2026-11-12):
# L4 10623341 13:54:14 + L2 10624169 27:20:44 + L1 10627751 22:36:02 = 63.85 h = 229860 s on the
# RTX 8000. Step rule (two phases: the first 3/16 of the run is linear-only, rigid:affine 1:1, the
# rest the 1:1:2 mix): with t_lin = mean step s of the linear-only phase and t_mix = mean step s of
# the mixed phase (from the smoke's log windows, time/compute_ms + time/data_ms, report
# .agents/tasks/ablation-oom-fixes/report.md section 5) and v = one validation pass s (~462 s at
# level 1; the run validates STEPS / VAL_EVERY times, val_time = STEPS / VAL_EVERY x v):
#   STEPS = (229860 - val_time) / (3/16 t_lin + 13/16 t_mix)
#         = floor(229860 / (3/16 t_lin + 13/16 t_mix + v / VAL_EVERY))   (minus ~1 % for startup)
#   e.g. diagnostic smokes 10639615 / 10639614 (rtx-04): t_lin 0.34 s, t_mix ~9.0 s, v ~300-410 s,
#   VAL_EVERY 500 -> 229860 / (0.06 + 7.31 + 0.8) -> ~28000 steps (to be re-measured in the smoke).
# The run needs ~64 h > the 48 h limit: resubmit the IDENTICAL line (same RUN_NAME and STEPS);
# --resume (default true) restores ckpt/last (every CKPT_EVERY 250 steps) and the RNGs, so at most
# 250 steps are lost per resubmission.
#
# Memory / host notes: the d12hs stem allows linear N up to ~19 under the TF int32 per-tensor limit
# (default widths: N <= 9, job 10517266); N stays 8. Final cascade level 1 peaks 38-39 GB under
# TF_MEM_LIMIT_MB 42500; from scratch there are no frozen stages (less memory). If the smoke's
# 'INFO gpu_peak' exceeds 41500 MB, fall back to MIN_PAIRS=1. Host RAM: MaxRSS ~94 GB at level 1,
# hence --mem=128G (64G was OOM-killed during validation, jobs 10541478 / 10541479).
# DEFORM_WARMUP_STEP (true | false, default TRUE here) -> --deform-warmup-step true: one discarded
# deform train step at startup (and on every resume) before the linear-only phase, so the allocator
# is laid out for the deform step first. Without it the smoke 10639201 OOMed at its first deform
# step after the step-100 validation (one-hot [4,19,160,224,192] float32), while deform from step 0
# (10639614, 38119 MiB) and deform from 100 without a validation before it (10639615) fit
# (.agents/tasks/ablation-oom-fixes/report.md).
#
# After main.py, tests/gpu/check_mixed_run.py checks the run dir (--expect-init-weights none, the
# grad-balance verdict from DEFORM_START + 100) and prints the INFO gpu_peak / step_time lines
# (projection over PROJECT_STEPS, default STEPS, at PROJECT_VAL_EVERY, default VAL_EVERY). Exit
# code: main.py's if non-zero, else the checker's.
# BGR_DRY_RUN=1 (test hook): validate the env, print the 'MAIN: python main.py ...' and 'CHECK:
# python tests/gpu/check_mixed_run.py ...' lines and exit 0 before any mkdir, file write or
# singularity call.
#
# DO NOT submit without Jay's approval; at most two active jobs (squeue -u jn85). Account qtim only.
# Before sbatch, the log folder must exist (SLURM does not create it):
#   mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
# Smoke (300 steps, deform and G from 100, three validations; projection for ~19000 steps):
#   RUN_NAME=r1_scratch_smoke STEPS=300 VAL_EVERY=100 WARMUP=50 DEFORM_START=100 DEFORM_RAMP=50 CURRICULUM_RAMP=100 G_START=100 G_RAMP=100 PROJECT_STEPS=19000 PROJECT_VAL_EVERY=500 sbatch --job-name=r1scr_smk --time=05:00:00 --mem=128G slurm/r_level1_scratch.sh
# Long (STEPS from the step rule; resubmit the identical line after the 48 h limit):
#   RUN_NAME=r1_scratch STEPS=<from the smoke> sbatch --job-name=r1scr --time=48:00:00 --mem=128G slurm/r_level1_scratch.sh
#
# Non-finite mitigations as in r_level1.sh (no flag while at the default, except the deform decoder
# dtype, default float32_fine here, whose flag is emitted): FP16_ACT_CLIP / FP16_ACT_CLIP_SCOPE,
# DEFORM_DECODER_DTYPE (policy | float16 | float32 | float32_fine), BN_EPSILON_OVERRIDE.
# GPU_POLL (true | false, default false): nvidia-smi device-memory poll every 5 s into
# $RESULTS/$RUN_NAME/gpu_poll.csv while main.py runs. MIN_TASK_STEPS (default empty) -> checker
# --min-task-steps. Synthetic contrast is not offered here (real-only ablation).
set -euo pipefail

PROJ="${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}"
REPO="${REPO:-$PROJ/brain-group-rad}"
DATA="${DATA:-$PROJ/data/npy_v2}"
RESULTS="${RESULTS:-$PROJ/results}"
SIF="${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_NAME="${RUN_NAME:-r1scr_$JOB_ID}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"
STEPS="${STEPS:-}"
BGR_DRY_RUN="${BGR_DRY_RUN:-}"

die() { echo "ERROR: $*" >&2; exit 2; }
# non-negative decimal integer without leading zeros ($(( )) would read 08 as octal)
need_uint() { case "$2" in ''|*[!0-9]*|0?*) die "$1 must be a non-negative integer, got '$2'";; esac; }
FLOAT_RE='^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
need_float() { [[ "$2" =~ $FLOAT_RE ]] || die "$1 must be a non-negative number, got '$2'"; }
need_choice() {
    local name="$1" val="$2" c
    shift 2
    for c in "$@"; do [ "$val" = "$c" ] && return 0; done
    die "$name must be one of: $* (got '$val')"
}
is_pos() { awk -v x="$1" 'BEGIN { exit !(x + 0 > 0) }'; }

[ -n "$STEPS" ] || die "set STEPS (no default: the cosine schedule and the deform / G / curriculum fractions depend on it; pick it from the smoke with the step rule in the header and keep it on every resubmission)"
need_uint STEPS "$STEPS"
[ "$STEPS" -gt 0 ] || die "STEPS must be > 0, got '$STEPS'"

VAL_EVERY="${VAL_EVERY:-500}"
CKPT_EVERY="${CKPT_EVERY:-250}"
# Schedule: the level-4 phase-C fractions of STEPS (D6).
WARMUP="${WARMUP:-$((STEPS / 16))}"
DEFORM_START="${DEFORM_START:-$((3 * STEPS / 16))}"
DEFORM_RAMP="${DEFORM_RAMP:-$((STEPS / 32))}"
CURRICULUM="${CURRICULUM:-true}"
CURRICULUM_RAMP="${CURRICULUM_RAMP:-$((3 * STEPS / 16))}"
G_START="${G_START:-$((3 * STEPS / 16))}"
G_RAMP="${G_RAMP:-$((3 * STEPS / 16))}"
G_ROT="${G_ROT:-45}"
G_TRANS="${G_TRANS:-10}"
LR="${LR:-3e-4}"
WD="${WD:-1e-1}"
OPTIMIZER="${OPTIMIZER:-adamw}"
N="${N:-8}"
DEFORM_N="${DEFORM_N:-4}"
TASK_RATIO="${TASK_RATIO:-rigid:1,affine:1,deform:2}"
GROUP_CONFIG="${GROUP_CONFIG:-B}"
NORM="${NORM:-batch}"
SMOOTH="${SMOOTH:-15}"
# Width (d12hs): unset -> d12hs; set but empty -> flag not passed (ModelConfig default widths).
FILTERS="${FILTERS-16,64,128,192,256,384}"
DECODER_WIDTHS="${DECODER_WIDTHS-128,96,64,16,8}"
TF_MEM_LIMIT_MB="${TF_MEM_LIMIT_MB:-42500}"
GRAD_CKPT="${GRAD_CKPT:-full}"
PAIR_RECOMPUTE="${PAIR_RECOMPUTE:-true}"
MIN_PAIRS="${MIN_PAIRS:-n}"
MAX_PAIRS="${MAX_PAIRS:-64}"
LEAN_LOSSES="${LEAN_LOSSES:-true}"
LOWRES_METHOD="${LOWRES_METHOD:-tent}"
GRAD_BALANCE="${GRAD_BALANCE:-gradnorm}"
VAL_IMAGE_CHUNK="${VAL_IMAGE_CHUNK:-1}"
VAL_PAIR_CHUNK="${VAL_PAIR_CHUNK:-1}"
VAL_GROUP_SIZE="${VAL_GROUP_SIZE:-}"
PROJECT_STEPS="${PROJECT_STEPS:-$STEPS}"
PROJECT_VAL_EVERY="${PROJECT_VAL_EVERY:-$VAL_EVERY}"
DEFORM_WARMUP_STEP="${DEFORM_WARMUP_STEP:-true}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

RATIO_RE='^(rigid|affine|deform):[0-9]+(,(rigid|affine|deform):[0-9]+)*$'
[[ "$TASK_RATIO" =~ $RATIO_RE ]] || die "TASK_RATIO must look like rigid:1,affine:1,deform:2, got '$TASK_RATIO'"
for v in VAL_EVERY CKPT_EVERY WARMUP N DEFORM_N DEFORM_START DEFORM_RAMP CURRICULUM_RAMP G_START G_RAMP \
         TF_MEM_LIMIT_MB MAX_PAIRS VAL_IMAGE_CHUNK VAL_PAIR_CHUNK PROJECT_STEPS PROJECT_VAL_EVERY; do
    need_uint "$v" "${!v}"
done
for v in LR WD G_ROT G_TRANS SMOOTH; do need_float "$v" "${!v}"; done
need_choice OPTIMIZER "$OPTIMIZER" adamw adam
need_choice LOWRES_METHOD "$LOWRES_METHOD" tent nn
need_choice GRAD_CKPT "$GRAD_CKPT" none integrator decoder both full
need_choice PAIR_RECOMPUTE "$PAIR_RECOMPUTE" true false
need_choice LEAN_LOSSES "$LEAN_LOSSES" true false
need_choice MIN_PAIRS "$MIN_PAIRS" 1 n
need_choice CURRICULUM "$CURRICULUM" true false
need_choice GROUP_CONFIG "$GROUP_CONFIG" A B none E
need_choice NORM "$NORM" batch instance
need_choice DEFORM_WARMUP_STEP "$DEFORM_WARMUP_STEP" true false
WIDTH6_RE='^[1-9][0-9]*(,[1-9][0-9]*){5}$'
WIDTH5_RE='^[1-9][0-9]*(,[1-9][0-9]*){4}$'
WIDTH_ARGS=""
if [ -n "$FILTERS" ]; then
    [[ "$FILTERS" =~ $WIDTH6_RE ]] || die "FILTERS must be 6 comma-separated positive integers (stem,L2..L6) or empty, got '$FILTERS'"
    WIDTH_ARGS="--filters $FILTERS"
fi
if [ -n "$DECODER_WIDTHS" ]; then
    [[ "$DECODER_WIDTHS" =~ $WIDTH5_RE ]] || die "DECODER_WIDTHS must be 5 comma-separated positive integers (D5..D1) or empty, got '$DECODER_WIDTHS'"
    WIDTH_ARGS="$WIDTH_ARGS --decoder-widths $DECODER_WIDTHS"
fi
[ "$VAL_PAIR_CHUNK" -ge 1 ] || die "VAL_PAIR_CHUNK must be >= 1, got '$VAL_PAIR_CHUNK'"
VAL_ARGS=""
if [ -n "$VAL_GROUP_SIZE" ]; then
    need_uint VAL_GROUP_SIZE "$VAL_GROUP_SIZE"
    [ "$VAL_GROUP_SIZE" -ge 2 ] || die "VAL_GROUP_SIZE must be an integer >= 2, got '$VAL_GROUP_SIZE'"
    VAL_ARGS="--val-group-size $VAL_GROUP_SIZE"
fi
case "$GRAD_BALANCE" in
    gradnorm) BAL_ARGS="--task-grad-norm 12" ;;            # --task-loss-weights not passed (= none)
    deform64) BAL_ARGS="--task-loss-weights deform:64" ;;  # global clip 12 only
    *) die "GRAD_BALANCE must be gradnorm or deform64, got '$GRAD_BALANCE'" ;;
esac
MEM_ARGS=""
if [ "$TF_MEM_LIMIT_MB" -gt 0 ]; then MEM_ARGS="--gpu-memory-limit-mb $TF_MEM_LIMIT_MB"; fi
# Architecture ablation knobs: group-block placement always explicit, --norm only for instance.
MODEL_ARGS="--group-config $GROUP_CONFIG"
if [ "$NORM" = instance ]; then MODEL_ARGS="$MODEL_ARGS --norm instance"; fi

TASKS=$(echo "$TASK_RATIO" | sed 's/:[^,]*//g')
HAS_DEFORM=false
HAS_LINEAR=false
case ",$TASKS," in *,deform,*) HAS_DEFORM=true ;; esac
case ",$TASKS," in *,rigid,*|*,affine,*) HAS_LINEAR=true ;; esac

# Non-finite mitigations (as r_level1.sh; the deform decoder dtype default float32_fine is emitted).
FP16_ACT_CLIP="${FP16_ACT_CLIP:-0}"
FP16_ACT_CLIP_SCOPE="${FP16_ACT_CLIP_SCOPE:-deform_decoder}"
DEFORM_DECODER_DTYPE="${DEFORM_DECODER_DTYPE:-float32_fine}"
BN_EPSILON_OVERRIDE="${BN_EPSILON_OVERRIDE:-0}"
need_float FP16_ACT_CLIP "$FP16_ACT_CLIP"
need_float BN_EPSILON_OVERRIDE "$BN_EPSILON_OVERRIDE"
need_choice FP16_ACT_CLIP_SCOPE "$FP16_ACT_CLIP_SCOPE" deform_decoder all
need_choice DEFORM_DECODER_DTYPE "$DEFORM_DECODER_DTYPE" policy float16 float32 float32_fine
MITIG_ARGS=""
if is_pos "$FP16_ACT_CLIP"; then
    awk -v x="$FP16_ACT_CLIP" 'BEGIN { exit !(x + 0 <= 32000) }' || die "FP16_ACT_CLIP must be <= 32000, got '$FP16_ACT_CLIP'"
    if [ "$FP16_ACT_CLIP_SCOPE" = deform_decoder ] && [ "$HAS_DEFORM" = false ]; then
        echo "NOTE: linear-only ratio, deform-scoped FP16_ACT_CLIP dropped"
    else
        MITIG_ARGS="--fp16-act-clip $FP16_ACT_CLIP --fp16-act-clip-scope $FP16_ACT_CLIP_SCOPE"
    fi
fi
if [ "$DEFORM_DECODER_DTYPE" != policy ]; then
    if { [ "$DEFORM_DECODER_DTYPE" = float32 ] || [ "$DEFORM_DECODER_DTYPE" = float32_fine ]; } \
            && [ "$HAS_DEFORM" = false ]; then
        echo "NOTE: linear-only ratio, DEFORM_DECODER_DTYPE dropped"
    else
        MITIG_ARGS="$MITIG_ARGS --deform-decoder-dtype $DEFORM_DECODER_DTYPE"
    fi
fi
if is_pos "$BN_EPSILON_OVERRIDE"; then MITIG_ARGS="$MITIG_ARGS --bn-epsilon-override $BN_EPSILON_OVERRIDE"; fi

GPU_POLL="${GPU_POLL:-false}"
MIN_TASK_STEPS="${MIN_TASK_STEPS:-}"
need_choice GPU_POLL "$GPU_POLL" true false
if [ -n "$MIN_TASK_STEPS" ] && ! [[ "$MIN_TASK_STEPS" =~ $RATIO_RE ]]; then
    die "MIN_TASK_STEPS must look like rigid:20,affine:20,deform:80 or be empty, got '$MIN_TASK_STEPS'"
fi
TASK_CHECK=""
if [ -n "$MIN_TASK_STEPS" ]; then TASK_CHECK="--min-task-steps $MIN_TASK_STEPS"; fi
# GPU_POLL: device-level memory poll on the host (outside the container, where CUDA_VISIBLE_DEVICES
# names the allocated GPU; unset -> device 0), every 5 s into $RESULTS/$RUN_NAME/gpu_poll.csv.
POLL_DEV="${CUDA_VISIBLE_DEVICES:-0}"
POLL_DEV="${POLL_DEV%%,*}"
POLL_CMD=(nvidia-smi -i "$POLL_DEV" --query-gpu=timestamp,memory.used,memory.total
          --format=csv,noheader,nounits -l 5)

DEFORM_LOSSES="--lambda-dice-template 1 --lambda-pdice 1 --lambda-ncc-template 1 --lambda-ncc-pair 1 --lambda-smooth $SMOOTH --p-pdice 0.1"
if [ "$HAS_DEFORM" = true ]; then
    if [ "$GRAD_CKPT" = none ]; then
        DEFORM_ARGS="--gradient-checkpointing none $DEFORM_LOSSES"
        echo "NOTE: GRAD_CKPT=none uses the m10 no-ckpt knob set; PAIR_RECOMPUTE/LEAN_LOSSES dropped"
    else
        DEFORM_ARGS="--gradient-checkpointing $GRAD_CKPT --lean-sampler true --pair-recompute $PAIR_RECOMPUTE"
        if [ "$LEAN_LOSSES" = true ]; then DEFORM_ARGS="$DEFORM_ARGS --lean-losses true"; fi
        DEFORM_ARGS="$DEFORM_ARGS $DEFORM_LOSSES"
    fi
    DEFORM_ARGS="$DEFORM_ARGS --val-image-chunk $VAL_IMAGE_CHUNK"
    if [ "$DEFORM_WARMUP_STEP" = true ]; then DEFORM_ARGS="$DEFORM_ARGS --deform-warmup-step true"; fi
else
    DEFORM_ARGS="--gradient-checkpointing none"
    echo "NOTE: linear-only ratio, deform-only flags dropped"
fi
START_ARGS=""
if [ "$HAS_DEFORM" = true ] && [ "$HAS_LINEAR" = true ] && [ "$DEFORM_START" -gt 0 ]; then
    START_ARGS="--task-start-step deform:$DEFORM_START --task-start-ramp-steps $DEFORM_RAMP"
elif [ "$HAS_DEFORM" = true ] && [ "$HAS_LINEAR" = false ] && [ "$DEFORM_START" -gt 0 ]; then
    echo "NOTE: deform-only ratio, DEFORM_START ignored"
fi
if [ "$HAS_LINEAR" = true ]; then
    LIN_ARGS="--curriculum $CURRICULUM"
    if [ "$CURRICULUM" = true ]; then LIN_ARGS="$LIN_ARGS --curriculum-ramp-steps $CURRICULUM_RAMP"; fi
    LIN_ARGS="$LIN_ARGS --global-transform true --global-rot-max-deg $G_ROT --global-trans-max-mm $G_TRANS"
    LIN_ARGS="$LIN_ARGS --global-start-step $G_START --global-ramp-steps $G_RAMP"
else
    LIN_ARGS="--curriculum false --global-transform false"
    echo "NOTE: deform-only ratio, curriculum and G dropped (they act on the linear tasks only)"
fi
BALANCE_CHECK=""
if [ "$HAS_DEFORM" = true ] && [ "$HAS_LINEAR" = true ]; then
    if [ -n "$START_ARGS" ]; then
        BALANCE_CHECK="--grad-balance-rule true --grad-balance-from-step $((DEFORM_START + 100))"
    else
        BALANCE_CHECK="--grad-balance-rule true --grad-balance-from-step 100"
    fi
fi

# shellcheck disable=SC2206  # the *_ARGS strings are split on purpose
MAIN=(main.py --data "$DATA" --results "$RESULTS" --run-name "$RUN_NAME" --level 1
      --group-size "$N" --deform-group-size "$DEFORM_N" --task-ratio "$TASK_RATIO"
      --steps "$STEPS" --val-every "$VAL_EVERY" --ckpt-every "$CKPT_EVERY"
      --optimizer "$OPTIMIZER" --lr "$LR" --weight-decay "$WD" --schedule cosine
      --warmup-steps "$WARMUP" --min-lr-factor 1e-3
      $BAL_ARGS --p-pair 0.1 --min-pairs "$MIN_PAIRS" --max-pairs "$MAX_PAIRS"
      --amp true --split-group-conv true --intensity-aug-device tf --lowres-method "$LOWRES_METHOD"
      $START_ARGS $LIN_ARGS
      --val-pair-chunk "$VAL_PAIR_CHUNK" --val-png true
      --num-threads "${SLURM_CPUS_PER_TASK:-3}" --log-every 25 --log-backend jsonl
      $VAL_ARGS $DEFORM_ARGS $MEM_ARGS $WIDTH_ARGS $MITIG_ARGS $MODEL_ARGS $EXTRA_ARGS)
# AMP: the dynamic loss scale overflows a few times early on (m9 smoke: 8 of 200), hence 0.06.
# Per task: FAIL if > 5 % of a task's steps in any 500-step window after step 100 are non-finite.
# shellcheck disable=SC2206
CHECK=(tests/gpu/check_mixed_run.py "$RESULTS/$RUN_NAME" --tasks "$TASKS" --steps "$STEPS"
       --expect-min-pairs "$MIN_PAIRS" --max-nonfinite-frac 0.06 --max-window-nonfinite-frac 0.05
       --memory-budget-mb 36864
       --project-steps "$PROJECT_STEPS" --project-val-every "$PROJECT_VAL_EVERY" --expect-enc-grad true
       $BALANCE_CHECK --expect-init-weights none $TASK_CHECK)

if [ "$BGR_DRY_RUN" = "1" ]; then
    echo "MAIN: python ${MAIN[*]}"
    echo "CHECK: python ${CHECK[*]}"
    if [ "$GPU_POLL" = true ]; then echo "POLL: ${POLL_CMD[*]} > $RESULTS/$RUN_NAME/gpu_poll.csv"; fi
    exit 0
fi

SCRATCH="$PROJ/scratch"
mkdir -p "$RESULTS" "$SCRATCH/keras" "$SCRATCH/cuda_cache"
export TMPDIR="$SCRATCH_ROOT/$JOB_ID"
mkdir -p "$TMPDIR"
export KERAS_HOME="$SCRATCH/keras"
export CUDA_CACHE_PATH="$SCRATCH/cuda_cache"
export PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_TMPDIR="$TMPDIR"
export SINGULARITYENV_KERAS_HOME="$KERAS_HOME"
export SINGULARITYENV_CUDA_CACHE_PATH="$CUDA_CACHE_PATH"
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1

cd "$REPO"
echo "host $(hostname) | job $JOB_ID | repo rev $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '?') | $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
echo "run $RUN_NAME | level 1 FROM SCRATCH (no cascade) | ratio $TASK_RATIO | steps $STEPS | val every $VAL_EVERY | ckpt every $CKPT_EVERY | warmup $WARMUP | N $N | deform N $DEFORM_N"
echo "model: group config $GROUP_CONFIG | norm $NORM | smoothness $SMOOTH | widths: filters ${FILTERS:-default} | decoder ${DECODER_WIDTHS:-default}"
echo "optimizer $OPTIMIZER | lr $LR | wd $WD | grad balance $GRAD_BALANCE ($BAL_ARGS) | lowres $LOWRES_METHOD | min pairs $MIN_PAIRS | max pairs $MAX_PAIRS"
echo "deform start $DEFORM_START (ramp $DEFORM_RAMP) | curriculum $CURRICULUM ($CURRICULUM_RAMP) | G $G_ROT deg / $G_TRANS mm from $G_START (ramp $G_RAMP)"
echo "memory: grad ckpt $GRAD_CKPT | pair recompute $PAIR_RECOMPUTE | lean losses $LEAN_LOSSES | tf memory limit ${TF_MEM_LIMIT_MB} MB (0 = growth) | val image chunk $VAL_IMAGE_CHUNK | val pair chunk $VAL_PAIR_CHUNK | deform warm-up step $DEFORM_WARMUP_STEP"
echo "fp16 act clip $FP16_ACT_CLIP ($FP16_ACT_CLIP_SCOPE) | deform decoder dtype $DEFORM_DECODER_DTYPE | bn epsilon override $BN_EPSILON_OVERRIDE (0 = off) | gpu poll $GPU_POLL"

POLL_PID=""
stop_poll() {
    if [ -n "$POLL_PID" ]; then
        kill "$POLL_PID" 2>/dev/null || true
        wait "$POLL_PID" 2>/dev/null || true
        POLL_PID=""
    fi
}
if [ "$GPU_POLL" = true ]; then
    mkdir -p "$RESULTS/$RUN_NAME"
    "${POLL_CMD[@]}" > "$RESULTS/$RUN_NAME/gpu_poll.csv" 2>/dev/null &
    POLL_PID=$!
    trap stop_poll EXIT
    echo "gpu poll: pid $POLL_PID -> $RESULTS/$RUN_NAME/gpu_poll.csv (device $POLL_DEV)"
fi

rc=0
singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF" \
    python "${MAIN[@]}" </dev/null || rc=$?
stop_poll
echo "main.py exit code $rc $(date -Is)"
if [ "$rc" -ne 0 ]; then
    exit "$rc"
fi
crc=0
singularity exec -B /autofs -B /cluster -B /space "$SIF" \
    python "${CHECK[@]}" </dev/null || crc=$?
echo "finished check_mixed_run exit code $crc $(date -Is)"
exit "$crc"
