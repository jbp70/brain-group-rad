#!/bin/bash
#SBATCH --job-name=bgr-verify-synth
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
#
# Synthetic-contrast DALI checks (check s of tests/gpu/verify_dali_pipeline.py, design
# .agents/tasks/synth-contrast/design.md 7.1, plan deviation D-c) at levels 4 and 1.
#   s1 LinearPipeline with the synthfine label reader: image / matrix streams unchanged (INFO if
#      not), labels uint8 aligned with the unwarped image (>= 99 % of the voxels; must PASS)
#   s2 TF render with T_mm^-1 vs the DALI warp, interior mean |diff| < 1e-3 (must PASS)
#   s3 DALI process memory with / without the label reader (INFO)
# Needs the synthfine export (export-add-labels files + manifest phase) to exist first.
# Output: $PROJ/scratch/verify_synth/verify_synth_l<k>.json
# DO NOT submit without Jay's approval; at most two active jobs (squeue -u jn85). Account qtim only.
#   sbatch slurm/verify_synth.sh
set -uo pipefail

PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
SIF=/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="$PROJ/scratch/verify_synth"
mkdir -p "$OUT/tmp" "$PROJ/scratch/keras" "$PROJ/scratch/cuda_cache"
export TMPDIR="/scratch/$JOB_ID"; mkdir -p "$TMPDIR"
export SINGULARITYENV_TMPDIR="$TMPDIR"
export SINGULARITYENV_KERAS_HOME="$PROJ/scratch/keras"
export SINGULARITYENV_CUDA_CACHE_PATH="$PROJ/scratch/cuda_cache"
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1

cd "$PROJ/brain-group-rad"
echo "host $(hostname) | job $JOB_ID | repo $(git rev-parse --short HEAD) | $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

rc=0
for lvl in 4 1; do
    echo "=== check s, level $lvl $(date -Is)"
    timeout 1500 singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF" \
        python tests/gpu/verify_dali_pipeline.py --data "$PROJ/data/npy_v2" --scratch "$OUT/tmp" \
        --out "$OUT/verify_synth_l$lvl.json" --checks s --level "$lvl" --num-threads "${SLURM_CPUS_PER_TASK:-3}" \
        </dev/null || rc=1
done
echo "finished rc=$rc $(date -Is)"
exit "$rc"
