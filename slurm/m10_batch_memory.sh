#!/bin/bash
#SBATCH --job-name=bgr-m10-mem
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
# Retrain-from-scratch batch sizes (2026-10-26, Jay): the largest rigid/affine N and deform N that
# fit at each cascade level with AMP and the milestone-9 memory knobs (default widths), for the
# from-scratch retrain and its hyperparameter sweep. One configuration per process (--isolate true).
# Deform: Dice + LNCC 1/1, --min-pairs n, AMP, --gradient-checkpointing full, lean sampler,
# split group conv (pair recompute only at level 1), with a validation pass; level 4 also without
# checkpointing (speed). Linear: AMP + split group conv. The tool records OOMs as results.
# Budgets: 48 GB cards ~44 GiB TF (minus DALI: ~1 GiB at level 4, ~3.5 GiB level 2 with DALI aug,
# less with TF-side aug); A100 rule peak <= 36 GiB. Level 1 is known (deform N 4, linear N 8 by the
# int32 element limit), so only deform N 5 with lean losses is re-tested there.
# Output: $PROJ/scratch/m10/memory_batch.json.
# DO NOT submit without Jay's approval. Account qtim only.
#   sbatch slurm/m10_batch_memory.sh
# Each step is bounded by `timeout` so one stuck configuration cannot hold the GPU for hours.
set -uo pipefail

PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
SIF=/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT_DIR="$PROJ/scratch/m10"
mkdir -p "$OUT_DIR" "$PROJ/scratch/keras" "$PROJ/scratch/cuda_cache"
export TMPDIR="/scratch/$JOB_ID"; mkdir -p "$TMPDIR"
export SINGULARITYENV_TMPDIR="$TMPDIR"
export SINGULARITYENV_KERAS_HOME="$PROJ/scratch/keras"
export SINGULARITYENV_CUDA_CACHE_PATH="$PROJ/scratch/cuda_cache"
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1

cd "$PROJ/brain-group-rad"
echo "host $(hostname) | job $JOB_ID | repo $(git rev-parse --short HEAD) | $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

RUN=(singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF")
M=(python tests/gpu/measure_level_memory.py --isolate true --amp true --widths default
   --lambda-ncc-template 1 --lambda-ncc-pair 1 --warmup-steps 2 --steps 4 --out "$OUT_DIR/memory_batch.json")
# No --pair-recompute at levels 4 / 2: it unrolls one serialized block per pair, and at large N the
# pair count is large (p 0.1: N 32 -> 99, N 48 -> 226, N 64 -> 403 pairs). In job 10519650 the N 48
# config sat in graph construction at 99 % CPU for ~2 h. Pair recompute stays for level 1 (4 pairs).
DK=(--gradient-checkpointing full --lean-sampler true --split-group-conv true --min-pairs n)
DK1=("${DK[@]}" --pair-recompute true)

rc=0
step() {   # step <label> <args...>
    local label="$1"; shift
    echo "=== $label start $(date -Is)"
    timeout 2400 "${RUN[@]}" "${M[@]}" "$@" </dev/null || { echo "$label exit code $?"; rc=1; }
    echo "=== $label end $(date -Is)"
}

# level 4 (no cascade)
step "L4 deform knobs"    --level 4 --cascade false --tasks deform --n 32,48,64,96 "${DK[@]}" --val-pass true --val-pair-chunk 16
step "L4 deform no ckpt"  --level 4 --cascade false --tasks deform --n 32,48,64 --gradient-checkpointing none --min-pairs n
step "L4 linear"          --level 4 --cascade false --tasks rigid,affine --n 32,64,96,128 --split-group-conv true
# level 2 (frozen level-4 stage)
step "L2 deform knobs"    --level 2 --cascade true --tasks deform --n 8,12,16,20,24 "${DK[@]}" --val-pass true --val-pair-chunk 4
step "L2 linear"          --level 2 --cascade true --tasks rigid,affine --n 16,24,32,48 --split-group-conv true
# level 1: deform N 5 with lean losses (N 4 fits; N 5 OOMed without lean losses)
step "L1 deform N5 lean"  --level 1 --cascade true --tasks deform --n 5 "${DK1[@]}" --lean-losses true --val-pass true --val-pair-chunk 1

echo "finished rc=$rc $(date -Is) | $OUT_DIR/memory_batch.json"
exit "$rc"
