#!/bin/bash
#SBATCH --job-name=bgr-verify-m
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
# DALI vs TF intensity-augmentation equivalence (check m of tests/gpu/verify_dali_pipeline.py,
# .agents/tasks/tf-intensity-aug/report.md) at levels 4 and 1.
#   m1 low-res ROI convention (must PASS; else flip LOWRES_ROI to the best mode reported)
#   m2 distribution statistics (must PASS)
#   m3 DALI device memory per pipeline with / without aug (INFO)
# DO NOT submit without Jay's approval. Account qtim only.
#   sbatch slurm/verify_m.sh
set -uo pipefail

PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
SIF=/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="$PROJ/scratch/verify_m"
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
    echo "=== check m, level $lvl $(date -Is)"
    timeout 1500 singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF" \
        python tests/gpu/verify_dali_pipeline.py --data "$PROJ/data/npy_v2" --scratch "$OUT/tmp" \
        --out "$OUT/verify_m_l$lvl.json" --checks m --level "$lvl" --num-threads "${SLURM_CPUS_PER_TASK:-3}" \
        </dev/null || rc=1
done
echo "finished rc=$rc $(date -Is)"
exit "$rc"
