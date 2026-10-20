#!/usr/bin/env bash
# Run selected unittest modules / cases in tf_gpu (milestone 6 development), with the local GPU
# settings of tests/run_tests_wsl.sh. Usage: bash run_some.sh tests.test_x[.Class[.test]] ...
REPO=/mnt/d/Research/Brain_Registration_Projects/brain-group-rad
cd "$REPO"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONDONTWRITEBYTECODE=1
export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 BGR_TF_GPU_LIMIT_MB=${BGR_TF_GPU_LIMIT_MB:-6000} TF_CUDNN_USE_AUTOTUNE=0
start=$(date +%s)
python -m unittest -v "$@" 2>&1 | grep --line-buffered -v "cuBLAS factory"
echo "exit: ${PIPESTATUS[0]}  wall $(( $(date +%s) - start )) s"
