#!/bin/bash
#SBATCH --job-name=bgr-m7-3task
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Milestone 7: three-task training at level 4 (rigid:affine:deform 1:1:2, one shared Adam),
# warm-started from the milestone-5 mixed rigid+affine checkpoint. The deform decoder is new, so
# the warm start is PARTIAL (encoder + linear heads restored, deform decoder fresh; logged as the
# 'init_weights' event and in config.json). Linear groups N = 32, deform groups DEFORM_N = 16
# (validation: 2 linear groups of 32, 3 deform groups of 16, deform pairs capped at 64; do NOT
# pass --val-group-size, it would apply to both). G (group-shared global transform) at
# 45 deg / 10 mm from step 0, curriculum off (warm start, milestone-5 result). Linear losses are
# divided by LOSS_SCALE_MM2 (default 1) and TASK_LOSS_WEIGHTS (default deform:64; 'none' = all 1) are the
# balance knobs against the deform Dice loss (milestone-7 report: train_<task>/diag_enc_grad_norm_eff). After main.py, check_mixed_run.py
# checks the run dir (validation, PNGs, both best checkpoints, warm start, grad-norm keys) and
# prints the linear/deform encoder-gradient ratio. Exit code: main.py's if non-zero, else the
# checker's.
#
# DO NOT submit without Jay's approval (cluster-access rules). Account qtim only, never entre.
# Before sbatch, the log folder must exist (SLURM does not create it):
#   mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
# Smoke variant first (~20 min; step time and memory measured there):
#   RUN_NAME=m7_smoke STEPS=200 VAL_EVERY=100 WARMUP=50 sbatch --job-name=m7smoke --time=00:45:00 --mem=64G slurm/m7_three_task.sh
# Long run (12000 steps, ~2.2 s/step average estimated + 48 validations, ~9 h):
#   RUN_NAME=m7_three_task_long sbatch --job-name=m7long --time=12:00:00 --mem=64G slurm/m7_three_task.sh
# LNCC A/B (design 6.2 v3.9; LNCC weights default 0 = this recipe unchanged), Dice+LNCC arm:
#   RUN_NAME=m7_three_task_lncc EXTRA_ARGS="--lambda-ncc-template 1 --lambda-ncc-pair 1" sbatch --job-name=m7lncc --time=14:00:00 --mem=64G slurm/m7_three_task.sh
# On an A100 (fits: 34.3 GB peak; ~1.6 s/step measured): add -p dgx-a100 to the sbatch line.
# Time estimates (not measured for this recipe): linear step ~1.4 s at N = 32 (m5, RTX 8000),
# deform step ~3 s at N = 16 (extrapolated from 1.6 s at N = 8). Memory: linear N 32 ~34 GB,
# deform N 16 ~33.8 GB on the 48 GB cards; the peak is the larger of the two, not the sum. On
# OOM use DEFORM_N=12.
set -euo pipefail

PROJ="${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}"
REPO="${REPO:-$PROJ/brain-group-rad}"
DATA="${DATA:-$PROJ/data/npy_v2}"
RESULTS="${RESULTS:-$PROJ/results}"
SIF="${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_NAME="${RUN_NAME:-m7_three_task_$JOB_ID}"
STEPS="${STEPS:-12000}"
VAL_EVERY="${VAL_EVERY:-250}"
WARMUP="${WARMUP:-500}"
N="${N:-32}"
DEFORM_N="${DEFORM_N:-16}"
TASK_RATIO="${TASK_RATIO:-rigid:1,affine:1,deform:2}"
INIT="${INIT:-$RESULTS/m5_mixed_long_curr_dG_20261019-205526/ckpt/best}"
# 1 = unscaled (milestone-7 probe: the global clip of 12 already caps every linear step, so the
# applied encoder-gradient ratio linear/deform was ~1.4-1.7 for every S in 1..1000; confirm with
# the checker's enc_grad_ratio line after the smoke run, P2 in the milestone-7 report).
LOSS_SCALE_MM2="${LOSS_SCALE_MM2:-1}"
# deform:64 (2026-10-21 A/B, jobs 10493712 vs 10493723): the unweighted deform encoder gradient
# was ~70x below the clipped linear ones; with deform:64 the deform steps are clipped too (per-task
# normalisation) and val deform template / pair Dice at step 8000 rose 0.704 / 0.608 -> 0.723 /
# 0.638 with linear errors unchanged. Set TASK_LOSS_WEIGHTS=none for the unweighted recipe.
TASK_LOSS_WEIGHTS="${TASK_LOSS_WEIGHTS:-deform:64}"
if [ "$TASK_LOSS_WEIGHTS" = "none" ]; then TASK_LOSS_WEIGHTS=""; fi
G_ROT="${G_ROT:-45}"
G_TRANS="${G_TRANS:-10}"
AMP="${AMP:-false}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ ! -d "$INIT" ]; then
    echo "ERROR: --init-weights directory $INIT does not exist (set INIT=...)" >&2
    exit 2
fi
WEIGHT_ARGS=""
if [ -n "$TASK_LOSS_WEIGHTS" ]; then
    WEIGHT_ARGS="--task-loss-weights $TASK_LOSS_WEIGHTS"
fi

SCRATCH="$PROJ/scratch"
mkdir -p "$RESULTS" "$SCRATCH/keras" "$SCRATCH/cuda_cache"
export TMPDIR="/scratch/$JOB_ID"
mkdir -p "$TMPDIR"
export KERAS_HOME="$SCRATCH/keras"
export CUDA_CACHE_PATH="$SCRATCH/cuda_cache"
export PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_TMPDIR="$TMPDIR"
export SINGULARITYENV_KERAS_HOME="$KERAS_HOME"
export SINGULARITYENV_CUDA_CACHE_PATH="$CUDA_CACHE_PATH"
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1

cd "$REPO"
echo "host $(hostname) | job $JOB_ID | repo $(git rev-parse --short HEAD 2>/dev/null || echo '?') | $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
echo "run $RUN_NAME | steps $STEPS | val every $VAL_EVERY | warmup $WARMUP | N $N | deform N $DEFORM_N | ratio $TASK_RATIO"
echo "init $INIT | loss scale mm2 $LOSS_SCALE_MM2 | task loss weights ${TASK_LOSS_WEIGHTS:-default} | G $G_ROT deg / $G_TRANS mm | amp $AMP"

rc=0
# shellcheck disable=SC2086  # WEIGHT_ARGS / EXTRA_ARGS are split on purpose
singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF" \
    python main.py --data "$DATA" --results "$RESULTS" --run-name "$RUN_NAME" \
    --level 4 --group-size "$N" --deform-group-size "$DEFORM_N" --task-ratio "$TASK_RATIO" \
    --steps "$STEPS" --val-every "$VAL_EVERY" --warmup-steps "$WARMUP" \
    --init-weights "$INIT" --loss-scale-mm2 "$LOSS_SCALE_MM2" $WEIGHT_ARGS \
    --global-transform true --global-rot-max-deg "$G_ROT" --global-trans-max-mm "$G_TRANS" \
    --global-start-step 0 --global-ramp-steps 0 --curriculum false --val-png true \
    --amp "$AMP" --num-threads "${SLURM_CPUS_PER_TASK:-3}" --log-every 25 --log-backend jsonl \
    $EXTRA_ARGS </dev/null || rc=$?
echo "main.py exit code $rc $(date -Is)"
if [ "$rc" -ne 0 ]; then
    exit "$rc"
fi
crc=0
singularity exec -B /autofs -B /cluster -B /space "$SIF" \
    python tests/gpu/check_mixed_run.py "$RESULTS/$RUN_NAME" --tasks rigid,affine,deform --steps "$STEPS" \
    --expect-init-weights partial --expect-enc-grad true </dev/null || crc=$?
echo "finished check_mixed_run exit code $crc $(date -Is)"
exit "$crc"
