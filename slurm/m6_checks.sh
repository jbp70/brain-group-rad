#!/bin/bash
#SBATCH --job-name=bgr-m6-checks
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Milestone-6 GPU checks on one cluster GPU:
#   1. DALI check l (DeformPipeline image/label alignment, residual re-warp; writes scratch/m2)
#   2. dryrun_deform: 20 deform-only steps + 20 mixed rigid:affine:deform 1:1:2 steps on real
#      level-4 data, N = 8, validation + PNGs, each checked by check_mixed_run.py (scratch/m6)
#   3. deform-step memory / step time on this GPU for N = 8, 12, 16 (default widths, no
#      checkpointing), to size the deform group (tests/gpu/measure_deform_memory.py)
# Submit from the clone on mlsc-login:  sbatch slurm/m6_checks.sh
set -uo pipefail

PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
SIF=/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif
cd "$PROJ/brain-group-rad"
GPU="${CUDA_VISIBLE_DEVICES:-0}"; GPU="${GPU%%,*}"
echo "host $(hostname) | job ${SLURM_JOB_ID:-?} | GPU $GPU | repo $(git rev-parse --short HEAD) | $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

rc=0
echo "=== check l"
bash tests/gpu/run_verify_glacier.sh "$GPU" 1500 verify l || rc=1
echo "=== dryrun_deform"
bash tests/gpu/run_overfit_glacier.sh "$GPU" 1800 dryrun_deform || rc=1
echo "=== deform memory"
S="$PROJ/scratch/m6"; mkdir -p "$S/tmp" "$S/keras" "$S/cuda_cache"
export TMPDIR="$S/tmp" KERAS_HOME="$S/keras" CUDA_CACHE_PATH="$S/cuda_cache" PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_TMPDIR="$TMPDIR" SINGULARITYENV_KERAS_HOME="$KERAS_HOME" \
       SINGULARITYENV_CUDA_CACHE_PATH="$CUDA_CACHE_PATH" SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
# one process per N so an OOM at a large N cannot poison the smaller ones; the script merges
# results into the same JSON
for n in 8 12 16; do
    timeout 900 singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF" \
        python tests/gpu/measure_deform_memory.py --n "$n" --widths default \
        --gradient-checkpointing none --out "$S/memory_cluster.json" </dev/null \
        || echo "memory N=$n failed (likely OOM)"
done
cat "$S/memory_cluster.json" 2>/dev/null || echo "no memory JSON written"
echo "finished rc=$rc $(date -Is)"
exit "$rc"
