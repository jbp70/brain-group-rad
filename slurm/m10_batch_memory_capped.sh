#!/bin/bash
#SBATCH --job-name=bgr-m10-memcap
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
# Retrain-from-scratch batch sizes, CAPPED re-measurement (retrain-prereqs design 16.6): the same
# tool and knobs as slurm/m10_batch_memory.sh, but every configuration with --max-pairs 64 (the
# training pair cap of the from-scratch scripts r_level4/2/1.sh, MAX_PAIRS=64). Each record stores
# max_pairs and the resulting n_pairs. One configuration per process (--isolate true), AMP, default
# widths, a `timeout` per step. The N / DEFORM_N placeholders of slurm/r_level4.sh and
# slurm/r_level2.sh are filled from these records (fits() rule in their headers).
# Measured:
#   L4 deform N 48/64/80/96: --gradient-checkpointing full, lean sampler, split group conv, NO pair
#     recompute (level-1 option only), Dice + LNCC 1/1, --min-pairs n, validation pass with
#     --val-pair-chunk 16 (int32 check on: the widest training tensor, the d3 conv_a input
#     N x 40 x 56 x 48 x 224, is 1.93e9 elements at N 80 and 2.31e9 at N 96 -> N 96 'limit' expected;
#     the same rule matches the uncapped L2 deform N 20 abort: d2 conv_a input 2.20e9);
#   L4 linear (rigid, affine) N 128/144/160/192 with split group conv and --min-pairs n (the knob
#     set of r_level4/2.sh; under the cap k = 64 for both floors), after the int32 element check
#     (--int32-check true: the largest per-group tensor of every training block, incl. the stem,
#     must stay <= 2^31 - 1 elements; violations are recorded as 'limit', not run). At level 4 the
#     widest tensor is the l3 output N x 40 x 56 x 48 x 128 (not the stem N x 40 x 56 x 48 x 32):
#     1.76e9 at N 128, 1.98e9 at N 144, 2.20e9 at N 160 and 2.64e9 at N 192, so the model expects
#     N 160 / 192 to be recorded as 'limit' (N 144 added as the largest N below the limit);
#   L2 deform N 16 (same deform knobs, --val-pair-chunk 4; compare with the uncapped 29.9 GB at 24
#     pairs, i.e. the cap has no effect at N 16: 0.1 x 240 = 24 < 64);
#   L2 linear N 32/36/40 with split group conv and --min-pairs n (int32 check on; l2 output N x 80 x 112 x 96 x 64:
#     1.98e9 at N 36, 2.20e9 at N 40 -> 'limit' expected; the uncapped N 48 aborted with -6).
# Uncapped reference (job 10519893, RTX 8000, AMP, scratch/m10/memory_batch.json): L4 deform (full
# ckpt) N 32 15.3 GB / 3.3 s (99 pairs), N 48 27.3 GB / 5.5 s (226 pairs), N 64 OOM (403 pairs),
# without checkpointing OOM at N 32; L4 rigid/affine N 128 35.3 GB / 1.86 s; L2 deform N 16
# 29.9 GB / 10.5 s (24 pairs), N 20/24 abort -6 (int32?); L2 linear N 32 19.3 GB, N 48 abort -6;
# L1 deform N 5 abort -6.
# Output: $PROJ/scratch/m10/memory_batch_capped.json.
# DO NOT submit without Jay's approval; at most two active jobs (squeue -u jn85). Account qtim only.
#   sbatch slurm/m10_batch_memory_capped.sh
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
   --lambda-ncc-template 1 --lambda-ncc-pair 1 --warmup-steps 2 --steps 4 --max-pairs 64
   --int32-check true --out "$OUT_DIR/memory_batch_capped.json")
# No pair recompute at levels 4 / 2 (it unrolls one serialized block per pair; job 10519650).
DK=(--gradient-checkpointing full --lean-sampler true --split-group-conv true --min-pairs n)

rc=0
step() {   # step <label> <args...>
    local label="$1"; shift
    echo "=== $label start $(date -Is)"
    timeout 2400 "${RUN[@]}" "${M[@]}" "$@" </dev/null || { echo "$label exit code $?"; rc=1; }
    echo "=== $label end $(date -Is)"
}

# level 4 (no cascade)
step "L4 deform knobs capped" --level 4 --cascade false --tasks deform --n 48,64,80,96 "${DK[@]}" --val-pass true --val-pair-chunk 16
step "L4 linear capped"       --level 4 --cascade false --tasks rigid,affine --n 128,144,160,192 --split-group-conv true --min-pairs n
# level 2 (frozen level-4 stage)
step "L2 deform knobs capped" --level 2 --cascade true --tasks deform --n 16 "${DK[@]}" --val-pass true --val-pair-chunk 4
step "L2 linear capped"       --level 2 --cascade true --tasks rigid,affine --n 32,36,40 --split-group-conv true --min-pairs n

echo "finished rc=$rc $(date -Is) | $OUT_DIR/memory_batch_capped.json"
exit "$rc"
