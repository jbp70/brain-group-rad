#!/bin/bash
#SBATCH --job-name=bgr-m4-debug
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
# Alternatives (A100 40 GB, full-resolution work later; costs more fairshare on dgx-a100):
##SBATCH -p dgx-a100
##SBATCH --cpus-per-task=16
##SBATCH -p pubgpu
##SBATCH --cpus-per-task=16
#
# Milestone-4 debug training job: rigid task only, level 4 (40x56x48 at 4 mm), N = 32, 2000
# steps (warmup 200, cosine), validation every 250 steps on the 39 T1w val cases (fixed groups
# and transforms), 'last' checkpoint every 250 steps, 'best' by val_rigid/pair_rms_mm.
# DO NOT submit without Jay's approval (cluster-access rules).
#
# Before sbatch, the log folder must exist (SLURM does not create it):
#   mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
# Submit (from the clone, after git pull):
#   sbatch slurm/train_debug.sh
#   STEPS=1000 N=16 sbatch slurm/train_debug.sh      # environment overrides (exported by sbatch)
#
# Expected runtime (ESTIMATE, not measured on an RTX 8000): on the local GTX 1080 a full-width
# rigid step at N = 4 takes ~1.0 s (encoder fwd+bwd 0.66 s, post-processing targets ~0.09 s,
# the rest group centering, loss and the update); peak TF memory ~3.8 GB. Scaling the conv
# work linearly in N and by the fp32 throughput ratio (RTX 8000 ~16 vs GTX 1080 ~9 TFLOPS) gives
# ~2-3 s per step at N = 32, i.e. ~1.1-1.7 h for 2000 steps plus ~1-2 min startup (cuDNN
# autotune) and ~10 s per validation. With --time=01:00:00 the job will probably stop at a
# timeout after ~1200-1800 steps; 'last' is saved every 250 steps, so results survive. For a
# complete run either pass STEPS=1000 or raise --time to 02:00:00. Memory: ~0.9 GB of
# activations per image at level 4 -> ~30 GB at N = 32 (extrapolated); on OOM rerun with N=16
# or AMP=true.
set -euo pipefail

PROJ="${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}"
REPO="${REPO:-$PROJ/brain-group-rad}"
DATA="${DATA:-$PROJ/data/npy_v2}"
RESULTS="${RESULTS:-$PROJ/results}"
SIF="${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_NAME="${RUN_NAME:-m4_debug_$JOB_ID}"
LEVEL="${LEVEL:-4}"
N="${N:-32}"
STEPS="${STEPS:-2000}"
VAL_EVERY="${VAL_EVERY:-250}"
WARMUP="${WARMUP:-200}"
TASK_RATIO="${TASK_RATIO:-rigid:1}"
GROUP_CONFIG="${GROUP_CONFIG:-A}"
AMP="${AMP:-false}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

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
echo "run $RUN_NAME | level $LEVEL | N $N | steps $STEPS | val every $VAL_EVERY | ratio $TASK_RATIO | config $GROUP_CONFIG | amp $AMP"

# shellcheck disable=SC2086  # EXTRA_ARGS is split on purpose
singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF" \
    python main.py --data "$DATA" --results "$RESULTS" --run-name "$RUN_NAME" \
    --level "$LEVEL" --group-size "$N" --steps "$STEPS" --val-every "$VAL_EVERY" \
    --warmup-steps "$WARMUP" --task-ratio "$TASK_RATIO" --group-config "$GROUP_CONFIG" \
    --amp "$AMP" --num-threads "${SLURM_CPUS_PER_TASK:-3}" --log-every 25 --log-backend jsonl \
    $EXTRA_ARGS </dev/null
echo "finished $(date -Is)"
