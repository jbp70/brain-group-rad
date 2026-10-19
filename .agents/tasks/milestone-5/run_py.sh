#!/usr/bin/env bash
# Run a Python script in tf_gpu from the repo root with the local GPU settings of
# tests/run_tests_wsl.sh. Usage: bash run_py.sh <script.py> [args...]
REPO=/mnt/d/Research/Brain_Registration_Projects/brain-group-rad
cd "$REPO"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONDONTWRITEBYTECODE=1
export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 BGR_TF_GPU_LIMIT_MB=6000 TF_CUDNN_USE_AUTOTUNE=0
start=$(date +%s)
python -u "$@" 2>&1 | grep --line-buffered -v "cuBLAS factory"
echo "exit: ${PIPESTATUS[0]}  wall $(( $(date +%s) - start )) s"
