#!/bin/bash
#SBATCH --job-name=bgr-m5-checks
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Milestone-5 GPU checks on one cluster GPU (same as the glacier commands):
#   1. tests/gpu/verify_dali_pipeline.py checks h,i,j,k (strict G sign; writes scratch/m2)
#   2. main.py dryrun_mixed: 20 alternating rigid/affine steps on real level-4 data, N = 8,
#      both tasks validated, PNGs written (writes scratch/m5)
# The glacier wrappers pin CUDA_VISIBLE_DEVICES to the index they are given; here that is the
# device SLURM allocated (0 under cgroup device isolation). TF32 is off via main.py / tests.
# Submit from the clone on mlsc-login:  sbatch slurm/m5_checks.sh
set -uo pipefail

PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
cd "$PROJ/brain-group-rad"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
GPU="${GPU%%,*}"
echo "host $(hostname) | job ${SLURM_JOB_ID:-?} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} -> $GPU | repo $(git rev-parse --short HEAD) | $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

rc=0
echo "=== checks h,i,j,k"
bash tests/gpu/run_verify_glacier.sh "$GPU" 1500 verify h,i,j,k || rc=1
echo "=== dryrun_mixed"
bash tests/gpu/run_overfit_glacier.sh "$GPU" 1500 dryrun_mixed || rc=1
R=$(ls -td "$PROJ"/scratch/m5/results/* 2>/dev/null | head -n1)
if [ -n "$R" ] && [ -f tests/gpu/check_mixed_run.py ]; then
    echo "=== check_mixed_run $R"
    python3 tests/gpu/check_mixed_run.py "$R" || rc=1
fi
echo "finished rc=$rc $(date -Is)"
exit "$rc"
