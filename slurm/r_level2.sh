#!/bin/bash
#SBATCH --job-name=bgr-r2
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Retrain from scratch, level 2 (80x112x96 at 2 mm): three tasks as the second stage of a FROZEN
# level-4 cascade (the m8 recipe; design .agents/tasks/retrain-prereqs/design.md sections 4, 6, 16).
# CASCADE_FROM is REQUIRED on a fresh run (no default): the final level-4 run's ckpt/best, e.g.
# CASCADE_FROM=$RESULTS/r4_final/ckpt/best. BN statistics of the trainable stage are recalibrated at
# step 0 of a fresh run (BN_RECAL_BATCHES 64; never repeated on resume) and BN then stays TRAINABLE
# for the whole run (BN_RECAL_FREEZE false; Jay, 2026-10 retrain recipe: batch statistics over
# N 32 / DEFORM_N 16 groups are large enough). Why recal although BN trains: the trainable stage
# starts as a copy of the level-4 weights, so its moving statistics are level-4 statistics on
# different inputs (4 mm vs 2 mm, pre-warped); with momentum 0.99 they need ~460 steps to fade
# (0.99^460 ~ 1 %), so the early eval-mode validations and best-checkpoint picks would use
# mismatched statistics. The recal replaces them with the mean over 64 level-2 batches (~9 min incl.
# its pre / post validation passes, m8 job 10505028: 515 s) and logs the eval/train gap; training
# then updates them normally. BN_RECAL_BATCHES=0 skips it (moving stats start from the level-4
# copy); BN_RECAL_FREEZE=true = the old m8 recipe (BN frozen after the recal).
# Deform decoder dtype: the AMP policy (float16; DEFORM_DECODER_DTYPE default policy), as level 4.
# Recipe (the same at every level of the retrain): AMP (hard-coded),
# --split-group-conv true, TF-side intensity augmentation with LOWRES_METHOD (default tent),
# AdamW (OPTIMIZER adamw|adam, LR, WD; cosine, WARMUP 500, --min-lr-factor 1e-3), GRAD_BALANCE
# gradnorm = --task-grad-norm 12 (deform64 = --task-loss-weights deform:64), Dice + LNCC 1/1 +
# smoothness 0.5, p 0.1 / 0.1, --min-pairs $MIN_PAIRS, --max-pairs $MAX_PAIRS (default 64), G 45 deg
# / 10 mm from step 0, curriculum off, checkpoints every CKPT_EVERY 250 steps.
# Per-stage skip (Jay, design 4.1): SKIP_LINEAR 4:0 (rigid and affine), SKIP_DEFORM 4:0.15 -> a
# skipped level-4 stage contributes the identity (A_prev = I, U = 0); independent draws per task and
# step; validation never skips. Validation groups follow N / DEFORM_N (VAL_GROUP_SIZE empty =
# main.py's min(N, n_val), as m8); set VAL_GROUP_SIZE to fix them.
#
# Width (Jay, retrain recipe): d12hs = default widths with the full-resolution stem and the d1/d2
# decoder stages halved: --filters 16,64,128,192,256,384 --decoder-widths 128,96,64,16,8.
# FILTERS / DECODER_WIDTHS (unset -> d12hs; set but empty -> flag not passed = ModelConfig default).
# The same explicit lists at every level (r_level4/2/1.sh, slurm/sweep_level4.py) so config.json's
# ModelConfig is identical across the cascade (the cascade check compares every field but seed; an
# explicit list is NOT config-equal to the None default, so never mix FILTERS= runs with default
# runs in one cascade, and resume a run with the widths it started with). Measured fits (m11):
# L4 deform N64 fp16 25.6 GiB 4.1 s (job 10528435), L2 deform N16 fp16 27.2 GiB 7.6 s (10528435),
# L1 deform N4 4 pairs float32_fine 36.95 GiB 21.1 s under 40000 MB (10528396).
#
# Memory knobs (NOT m8's GRAD_CKPT=none / AMP=false script defaults: here the m10 knob set is the
# default): GRAD_CKPT (default full; none|integrator|decoder|both|full), PAIR_RECOMPUTE (default
# false), MIN_PAIRS (default n; 1|n), LEAN_LOSSES (default false). With deform and GRAD_CKPT != none:
# --gradient-checkpointing $GRAD_CKPT --lean-sampler true --pair-recompute $PAIR_RECOMPUTE
# (+ --lean-losses true iff LEAN_LOSSES=true) = m10's DK set; GRAD_CKPT=none: only
# --gradient-checkpointing none (allowed, not recommended at level 2). PAIR_RECOMPUTE stays false:
# --pair-recompute unrolls one serialized block per pair, and a level-4 N 48 config (226 pairs) sat
# in graph construction ~2 h in job 10519650; it is a level-1 option.
#
# Batch sizes (N / DEFORM_N; review v1.3 finding 1). For every m10 record define
#   fits(record, partition) := record.peak_mb <= BUDGET[partition][level]
#   BUDGET.rtx8000  = {4: 42000, 2: 41000}, BUDGET.dgx-a100 = 34000 at every level.
#   Unit: MiB, as written by tests/gpu/measure_level_memory.py (MB = 2**20) into the records'
#   peak_mb field of scratch/m10/memory_batch*.json. Derivation: 46080 MiB usable - ~500 CUDA
#   context - DALI (~1000 at level 4, ~2000 at level 2 with TF-side aug) - 1024 headroom - ~1500
#   for the validation / fragmentation gap seen in m9 (trainer vs measure tool).
# N / DEFORM_N = the largest measured N with fits(), from the record whose knob set equals this
# script's defaults (measured without pair recompute, with --max-pairs 64). The smoke (>= 2
# validations) must show 'INFO gpu_peak' <= the same budget, else drop to the next smaller measured
# N. TF_MEM_LIMIT_MB=0 (memory growth) is kept at levels 4 / 2 because DALI is small there; if the
# smoke OOMs after a validation, the first fallback is TF_MEM_LIMIT_MB=<budget> (as at level 1).
# A100: -p dgx-a100 only if the smoke's 'INFO gpu_peak' <= 34000 MiB; same 3 CPUs / --mem=64G.
#
# CASCADE_FROM guard: on a fresh run CASCADE_FROM must be a checkpoint directory or a checkpoint
# prefix (<prefix>.index exists). The guard is skipped when $RESULTS/$RUN_NAME/ckpt/last/checkpoint
# exists (resume / requeue): --cascade-from is still passed (it selects the cascade code path) but
# not read (the frozen stage comes from the run checkpoint).
# After main.py, tests/gpu/check_mixed_run.py checks the run dir (cascade provenance, frozen
# baselines, BN recalibration, stage-mode skip keys, pair counts) and prints the grad-balance verdict
# and the INFO gpu_peak / step_time lines (projection over PROJECT_STEPS, default STEPS, at
# PROJECT_VAL_EVERY, default VAL_EVERY). Exit code: main.py's if non-zero, else the checker's.
# BGR_DRY_RUN=1 (test hook): validate the env, run the CASCADE_FROM guard (read-only), print the
# 'MAIN: python main.py ...' and 'CHECK: python tests/gpu/check_mixed_run.py ...' lines and exit 0
# before any mkdir, file write or singularity call.
#
# DO NOT submit without Jay's approval; at most two active jobs (squeue -u jn85). Account qtim only.
# Before sbatch, the log folder must exist (SLURM does not create it):
#   mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
# Smoke (200 steps, two validations, BN recal 64 batches ~9 min as in m8):
#   RUN_NAME=r2_smoke CASCADE_FROM=$RESULTS/r4_final/ckpt/best STEPS=200 VAL_EVERY=100 WARMUP=50 PROJECT_STEPS=8000 PROJECT_VAL_EVERY=500 sbatch --job-name=r2smoke --time=03:00:00 --mem=64G slurm/r_level2.sh
# Long:
#   RUN_NAME=r2_long CASCADE_FROM=$RESULTS/r4_final/ckpt/best sbatch --job-name=r2long --time=24:00:00 --mem=64G slurm/r_level2.sh
# Wall time (review v1.3 NIT 5; level 2 at N 32 / DEFORM_N 16 on the RTX 8000 has no measured step
# time, ~8 s x 8000 ~ 18 h + validation is plausible):
#   set --time from the smoke's 'INFO step_time' projection x 1.3; if it exceeds the limit,
#   resubmit with the same RUN_NAME (resumable).
# Resume plan: --resume (default true) restores ckpt/last (every CKPT_EVERY steps), the frozen stage
# and the RNGs, checks the fingerprint, re-applies the BN freeze only if the saved record has
# freeze true (default false: BN keeps training after resume, as in the fresh run) and never
# repeats the recalibration.
#
# Non-finite mitigations (nonfinite-fixes, default off; decide after the m9 NaN probe; report
# .agents/tasks/nonfinite-fixes/report.md). No flag is emitted while a value is at its default.
#   FP16_ACT_CLIP (default 0 = off; 0 < C <= 32000, e.g. 30000) -> --fp16-act-clip C with
#     FP16_ACT_CLIP_SCOPE (deform_decoder | all, default deform_decoder) -> --fp16-act-clip-scope.
#   DEFORM_DECODER_DTYPE (policy | float16 | float32 | float32_fine, default policy) ->
#     --deform-decoder-dtype (float32 = deform decoder in float32 under AMP, ~2x decoder
#     activation memory; float32_fine = fp32 only d2/d1/svf, d5..d3 stay float16).
#   BN_EPSILON_OVERRIDE (default 0 = off) -> --bn-epsilon-override (pass it again on resume: the
#     same env line does).
#   Linear-only ratio: the deform-scoped clip and the decoder dtype are dropped with a NOTE.
# The per-task non-finite guard (--nonfinite-patience-task, default 20) is always on; override
# with EXTRA_ARGS. The checker also fails a task with > 5 % non-finite steps in any 500-step
# window after step 100 (--max-window-nonfinite-frac 0.05).
set -euo pipefail

PROJ="${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}"
REPO="${REPO:-$PROJ/brain-group-rad}"
DATA="${DATA:-$PROJ/data/npy_v2}"
RESULTS="${RESULTS:-$PROJ/results}"
SIF="${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_NAME="${RUN_NAME:-r2_$JOB_ID}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"
STEPS="${STEPS:-8000}"
VAL_EVERY="${VAL_EVERY:-500}"
CKPT_EVERY="${CKPT_EVERY:-250}"
WARMUP="${WARMUP:-500}"
LR="${LR:-3e-4}"
WD="${WD:-0}"
OPTIMIZER="${OPTIMIZER:-adamw}"
# Group sizes (Jay, 2026-10-27): rigid/affine N = 2 x DEFORM_N at every level: 128/64 (level 4), 32/16 (level 2), 8/4 (level 1). Measured with --max-pairs 64 (job 10524450, scratch/m10/memory_batch_capped.json, RTX 8000 tool peaks): linear N 32 19.4 GB 2.1-2.5 s (N 36 21.4 GB; N 40 int32 limit).
N="${N:-32}"
# Deform N 16: 29.9 GB, 10.6 s (24 pairs, cap inactive; N 20 / 24 aborted, job 10519893).
DEFORM_N="${DEFORM_N:-16}"
TASK_RATIO="${TASK_RATIO:-rigid:1,affine:1,deform:2}"
CASCADE_FROM="${CASCADE_FROM:-}"
BN_RECAL_BATCHES="${BN_RECAL_BATCHES:-64}"
BN_RECAL_FREEZE="${BN_RECAL_FREEZE:-false}"
# Width (d12hs): unset -> d12hs; set but empty -> flag not passed (ModelConfig default widths).
FILTERS="${FILTERS-16,64,128,192,256,384}"
DECODER_WIDTHS="${DECODER_WIDTHS-128,96,64,16,8}"
SKIP_LINEAR="${SKIP_LINEAR-4:0}"          # unset -> 4:0; set but empty = all 0
SKIP_DEFORM="${SKIP_DEFORM-4:0.15}"       # unset -> 4:0.15; set but empty = all 0
G_ROT="${G_ROT:-45}"
G_TRANS="${G_TRANS:-10}"
TF_MEM_LIMIT_MB="${TF_MEM_LIMIT_MB:-0}"
GRAD_CKPT="${GRAD_CKPT:-full}"
PAIR_RECOMPUTE="${PAIR_RECOMPUTE:-false}"
MIN_PAIRS="${MIN_PAIRS:-n}"
MAX_PAIRS="${MAX_PAIRS:-64}"
LEAN_LOSSES="${LEAN_LOSSES:-false}"
LOWRES_METHOD="${LOWRES_METHOD:-tent}"
GRAD_BALANCE="${GRAD_BALANCE:-gradnorm}"
VAL_PAIR_CHUNK="${VAL_PAIR_CHUNK:-4}"
VAL_GROUP_SIZE="${VAL_GROUP_SIZE:-}"
PROJECT_STEPS="${PROJECT_STEPS:-$STEPS}"
PROJECT_VAL_EVERY="${PROJECT_VAL_EVERY:-$VAL_EVERY}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
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
SKIP_RE='^([0-9]+:[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?(,[0-9]+:[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)*)?$'
need_skip() { [[ "$2" =~ $SKIP_RE ]] || die "$1 must be a comma list level:prob (e.g. 4:0.15) or empty, got '$2'"; }
# 0 if a level:prob spec has an entry with p > 0 (mantissa with a non-zero digit)
skip_positive() {
    local e p
    for e in ${1//,/ }; do
        p="${e#*:}"
        p="${p%%[eE]*}"
        case "$p" in *[1-9]*) return 0 ;; esac
    done
    return 1
}
RATIO_RE='^(rigid|affine|deform):[0-9]+(,(rigid|affine|deform):[0-9]+)*$'
[[ "$TASK_RATIO" =~ $RATIO_RE ]] || die "TASK_RATIO must look like rigid:1,affine:1,deform:2, got '$TASK_RATIO'"
for v in STEPS VAL_EVERY CKPT_EVERY WARMUP N DEFORM_N BN_RECAL_BATCHES TF_MEM_LIMIT_MB MAX_PAIRS \
         VAL_PAIR_CHUNK PROJECT_STEPS PROJECT_VAL_EVERY; do
    need_uint "$v" "${!v}"
done
for v in LR WD G_ROT G_TRANS; do need_float "$v" "${!v}"; done
need_skip SKIP_LINEAR "$SKIP_LINEAR"
need_skip SKIP_DEFORM "$SKIP_DEFORM"
need_choice OPTIMIZER "$OPTIMIZER" adamw adam
need_choice LOWRES_METHOD "$LOWRES_METHOD" tent nn
need_choice GRAD_CKPT "$GRAD_CKPT" none integrator decoder both full
need_choice PAIR_RECOMPUTE "$PAIR_RECOMPUTE" true false
need_choice LEAN_LOSSES "$LEAN_LOSSES" true false
need_choice MIN_PAIRS "$MIN_PAIRS" 1 n
need_choice BN_RECAL_FREEZE "$BN_RECAL_FREEZE" true false
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
[ -n "$CASCADE_FROM" ] || die "set CASCADE_FROM (the level-4 run's ckpt/best, e.g. \$RESULTS/r4_final/ckpt/best; also on a resume, it selects the cascade code path)"
# Guard only for fresh runs: a resume / requeue (ckpt/last exists) does not read the source.
if [ ! -f "$RESULTS/$RUN_NAME/ckpt/last/checkpoint" ]; then
    if [ ! -d "$CASCADE_FROM" ] && [ ! -f "$CASCADE_FROM.index" ]; then
        die "CASCADE_FROM $CASCADE_FROM is neither a checkpoint directory nor a prefix"
    fi
fi
MEM_ARGS=""
if [ "$TF_MEM_LIMIT_MB" -gt 0 ]; then MEM_ARGS="--gpu-memory-limit-mb $TF_MEM_LIMIT_MB"; fi

TASKS=$(echo "$TASK_RATIO" | sed 's/:[^,]*//g')
HAS_DEFORM=false
HAS_LINEAR=false
case ",$TASKS," in *,deform,*) HAS_DEFORM=true ;; esac
case ",$TASKS," in *,rigid,*|*,affine,*) HAS_LINEAR=true ;; esac

# Non-finite mitigations (nonfinite-fixes; all default off, no flag emitted when off).
FP16_ACT_CLIP="${FP16_ACT_CLIP:-0}"
FP16_ACT_CLIP_SCOPE="${FP16_ACT_CLIP_SCOPE:-deform_decoder}"
DEFORM_DECODER_DTYPE="${DEFORM_DECODER_DTYPE:-policy}"
BN_EPSILON_OVERRIDE="${BN_EPSILON_OVERRIDE:-0}"
need_float FP16_ACT_CLIP "$FP16_ACT_CLIP"
need_float BN_EPSILON_OVERRIDE "$BN_EPSILON_OVERRIDE"
need_choice FP16_ACT_CLIP_SCOPE "$FP16_ACT_CLIP_SCOPE" deform_decoder all
need_choice DEFORM_DECODER_DTYPE "$DEFORM_DECODER_DTYPE" policy float16 float32 float32_fine
is_pos() { awk -v x="$1" 'BEGIN { exit !(x + 0 > 0) }'; }
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

DEFORM_LOSSES="--lambda-dice-template 1 --lambda-pdice 1 --lambda-ncc-template 1 --lambda-ncc-pair 1 --lambda-smooth 0.5 --p-pdice 0.1"
if [ "$HAS_DEFORM" = true ]; then
    if [ "$GRAD_CKPT" = none ]; then
        DEFORM_ARGS="--gradient-checkpointing none $DEFORM_LOSSES"
        echo "NOTE: GRAD_CKPT=none uses the m10 no-ckpt knob set; PAIR_RECOMPUTE/LEAN_LOSSES dropped"
    else
        DEFORM_ARGS="--gradient-checkpointing $GRAD_CKPT --lean-sampler true --pair-recompute $PAIR_RECOMPUTE"
        if [ "$LEAN_LOSSES" = true ]; then DEFORM_ARGS="$DEFORM_ARGS --lean-losses true"; fi
        DEFORM_ARGS="$DEFORM_ARGS $DEFORM_LOSSES"
    fi
else
    DEFORM_ARGS="--gradient-checkpointing none"
    echo "NOTE: linear-only ratio, deform-only flags dropped"
fi
if [ "$HAS_LINEAR" = true ]; then
    LIN_ARGS="--global-transform true --global-rot-max-deg $G_ROT --global-trans-max-mm $G_TRANS"
    LIN_ARGS="$LIN_ARGS --global-start-step 0 --global-ramp-steps 0 --curriculum false"
else
    LIN_ARGS="--global-transform false --curriculum false"
    echo "NOTE: deform-only ratio, G dropped (it acts on the linear tasks only)"
fi
BALANCE_CHECK=""
if [ "$HAS_DEFORM" = true ] && [ "$HAS_LINEAR" = true ]; then
    BALANCE_CHECK="--grad-balance-rule true --grad-balance-from-step 100"
fi
# --expect-skip only when a trained task has a skip p > 0 (else no skip key is ever logged)
SKIP_CHECK=""
if { [ "$HAS_LINEAR" = true ] && skip_positive "$SKIP_LINEAR"; } || \
   { [ "$HAS_DEFORM" = true ] && skip_positive "$SKIP_DEFORM"; }; then
    SKIP_CHECK="--expect-skip true"
fi
RECAL_CHECK=""
if [ "$BN_RECAL_BATCHES" -gt 0 ]; then RECAL_CHECK="--expect-bn-recal true"; fi
LEVEL_CHECK_ARGS="--expect-cascade true $SKIP_CHECK --expect-init-weights full $RECAL_CHECK"

# shellcheck disable=SC2206  # the *_ARGS strings are split on purpose
MAIN=(main.py --data "$DATA" --results "$RESULTS" --run-name "$RUN_NAME" --level 2
      --cascade-from "$CASCADE_FROM" --group-size "$N" --deform-group-size "$DEFORM_N"
      --task-ratio "$TASK_RATIO" --steps "$STEPS" --val-every "$VAL_EVERY" --ckpt-every "$CKPT_EVERY"
      --optimizer "$OPTIMIZER" --lr "$LR" --weight-decay "$WD" --schedule cosine
      --warmup-steps "$WARMUP" --min-lr-factor 1e-3
      $BAL_ARGS --p-pair 0.1 --min-pairs "$MIN_PAIRS" --max-pairs "$MAX_PAIRS"
      --amp true --split-group-conv true --intensity-aug-device tf --lowres-method "$LOWRES_METHOD"
      --bn-recal-batches "$BN_RECAL_BATCHES" --bn-recal-freeze "$BN_RECAL_FREEZE"
      --cascade-skip-linear "$SKIP_LINEAR" --cascade-skip-deform "$SKIP_DEFORM" $LIN_ARGS
      --val-pair-chunk "$VAL_PAIR_CHUNK" --val-png true
      --num-threads "${SLURM_CPUS_PER_TASK:-3}" --log-every 25 --log-backend jsonl
      $VAL_ARGS $DEFORM_ARGS $MEM_ARGS $WIDTH_ARGS $MITIG_ARGS $EXTRA_ARGS)
# AMP: the dynamic loss scale overflows a few times early on (m8: 10-11 of 8000), hence 0.06.
# Per task: FAIL if > 5 % of a task's steps in any 500-step window after step 100 are non-finite.
# shellcheck disable=SC2206
CHECK=(tests/gpu/check_mixed_run.py "$RESULTS/$RUN_NAME" --tasks "$TASKS" --steps "$STEPS"
       --expect-min-pairs "$MIN_PAIRS" --max-nonfinite-frac 0.06 --max-window-nonfinite-frac 0.05
       --memory-budget-mb 36864
       --project-steps "$PROJECT_STEPS" --project-val-every "$PROJECT_VAL_EVERY" --expect-enc-grad true
       $BALANCE_CHECK $LEVEL_CHECK_ARGS)

if [ "$BGR_DRY_RUN" = "1" ]; then
    echo "MAIN: python ${MAIN[*]}"
    echo "CHECK: python ${CHECK[*]}"
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
echo "run $RUN_NAME | level 2 | ratio $TASK_RATIO | steps $STEPS | val every $VAL_EVERY | ckpt every $CKPT_EVERY | warmup $WARMUP | N $N | deform N $DEFORM_N"
echo "cascade from $CASCADE_FROM | bn recal $BN_RECAL_BATCHES (freeze $BN_RECAL_FREEZE) | skip linear '$SKIP_LINEAR' deform '$SKIP_DEFORM'"
echo "optimizer $OPTIMIZER | lr $LR | wd $WD | grad balance $GRAD_BALANCE ($BAL_ARGS) | lowres $LOWRES_METHOD | min pairs $MIN_PAIRS | max pairs $MAX_PAIRS"
echo "memory: grad ckpt $GRAD_CKPT | pair recompute $PAIR_RECOMPUTE | lean losses $LEAN_LOSSES | tf memory limit ${TF_MEM_LIMIT_MB} MB (0 = growth) | val group ${VAL_GROUP_SIZE:-default} | val pair chunk $VAL_PAIR_CHUNK"
echo "widths: filters ${FILTERS:-default} | decoder ${DECODER_WIDTHS:-default}"
echo "fp16 act clip $FP16_ACT_CLIP ($FP16_ACT_CLIP_SCOPE) | deform decoder dtype $DEFORM_DECODER_DTYPE | bn epsilon override $BN_EPSILON_OVERRIDE (0 = off)"

rc=0
singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF" \
    python "${MAIN[@]}" </dev/null || rc=$?
echo "main.py exit code $rc $(date -Is)"
if [ "$rc" -ne 0 ]; then
    exit "$rc"
fi
crc=0
singularity exec -B /autofs -B /cluster -B /space "$SIF" \
    python "${CHECK[@]}" </dev/null || crc=$?
echo "finished check_mixed_run exit code $crc $(date -Is)"
exit "$crc"
