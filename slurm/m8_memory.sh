#!/bin/bash
#SBATCH --job-name=bgr-m8-mem
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
# Milestone 8, job (a): level-2 DALI checks and the level-2 memory / step-time grid (design
# .agents/tasks/milestone-8/design.md v5, sections 8, 11, 12). Runs first, before the smoke and
# long runs of slurm/m8_level2_cascade.sh, whose N / DEFORM_N / GRAD_CKPT / AMP come from it.
#   1. verify_dali_pipeline.py --level 2, checks b, c, d, e (N 12 throughput with the 3 requested
#      CPUs) and l -> $OUT_DIR/verify_level2.json
#   2. measure_level_memory.py, four invocations merged into $OUT_DIR/memory_level2.json:
#      deform cascade N 2,3,4,6,8 x checkpointing none,integrator,both x AMP (30 configs, each
#      with a validation pass), deform no-cascade N 2,4 (frozen-stage share), rigid/affine cascade
#      N 4,8,12,16 x AMP (16), rigid no-cascade N 8 (1).
# Every call runs; the exit code is the first non-zero one (the memory script records OOMs as
# results and returns 0 unless it crashes). Time: ~49 configs x ~2 min + 30 validation passes x
# ~0.5 min + ~15 min DALI checks = ~2.1 h (level-2 step times are unmeasured; ~2x margin).
#
# DO NOT submit without Jay's approval (cluster-access rules). Account qtim only.
# Before sbatch, the log folder must exist (SLURM does not create it):
#   mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
# Submit (RTX 8000 / A6000 48 GB):
#   sbatch --job-name=m8mem --time=04:00:00 --mem=64G slurm/m8_memory.sh
set -euo pipefail

PROJ="${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}"
REPO="${REPO:-$PROJ/brain-group-rad}"
DATA="${DATA:-$PROJ/data/npy_v2}"
SIF="${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}"
JOB_ID="${SLURM_JOB_ID:-manual}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"
OUT_DIR="$PROJ/scratch/m8"

SCRATCH="$PROJ/scratch"
mkdir -p "$OUT_DIR/dali" "$SCRATCH/keras" "$SCRATCH/cuda_cache"
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
echo "host $(hostname) | job $JOB_ID | repo $(git rev-parse --short HEAD 2>/dev/null || echo '?') | $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
echo "data $DATA | out $OUT_DIR"

RUN=(singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF")
rc=0
rc_x=0
"${RUN[@]}" python tests/gpu/verify_dali_pipeline.py --data "$DATA" --level 2 --checks b,c,d,e,l \
    --num-threads "${SLURM_CPUS_PER_TASK:-3}" --n-throughput 12 \
    --scratch "$OUT_DIR/dali" --out "$OUT_DIR/verify_level2.json" </dev/null || rc_x=$?
echo "verify_dali_pipeline exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi

M=(python tests/gpu/measure_level_memory.py --level 2 --widths default --lambda-ncc-template 1
   --lambda-ncc-pair 1 --warmup-steps 2 --steps 10 --out "$OUT_DIR/memory_level2.json")
rc_x=0   # deform, cascade: 30 configs, each with a validation pass
"${RUN[@]}" "${M[@]}" --tasks deform --n 2,3,4,6,8 --cascade true --gradient-checkpointing none,integrator,both \
    --amp false,true --val-pass true </dev/null || rc_x=$?
echo "measure_level_memory (deform, cascade) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # deform, no cascade: 2 configs (frozen-stage share)
"${RUN[@]}" "${M[@]}" --tasks deform --n 2,4 --cascade false --gradient-checkpointing none \
    --amp false </dev/null || rc_x=$?
echo "measure_level_memory (deform, no cascade) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # rigid + affine, cascade: 16 configs
"${RUN[@]}" "${M[@]}" --tasks rigid,affine --n 4,8,12,16 --cascade true --gradient-checkpointing none \
    --amp false,true </dev/null || rc_x=$?
echo "measure_level_memory (linear, cascade) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # rigid, no cascade: 1 config
"${RUN[@]}" "${M[@]}" --tasks rigid --n 8 --cascade false --gradient-checkpointing none \
    --amp false </dev/null || rc_x=$?
echo "measure_level_memory (rigid, no cascade) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi

echo "finished m8_memory exit code $rc $(date -Is)"
exit "$rc"
