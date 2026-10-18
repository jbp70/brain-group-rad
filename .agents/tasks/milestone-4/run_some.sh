#!/usr/bin/env bash
# Run selected unittest modules in tf_gpu (milestone 4 development). Usage: bash run_some.sh tests.test_x ...
# Same local GPU settings as tests/run_tests_wsl.sh.
REPO=/mnt/d/Research/Brain_Registration_Projects/brain-group-rad
cd "$REPO"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONDONTWRITEBYTECODE=1
export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 BGR_TF_GPU_LIMIT_MB=6000
start=$(date +%s)
python -m unittest -v "$@" 2>&1 | grep --line-buffered -v "cuBLAS factory"
echo "exit: ${PIPESTATUS[0]}  wall $(( $(date +%s) - start )) s"
