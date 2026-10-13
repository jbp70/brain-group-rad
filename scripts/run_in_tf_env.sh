#!/usr/bin/env bash
# Run a script from this folder inside the local WSL `tf_gpu` conda env.
#
# Usage (from Windows PowerShell):
#   wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/scripts/run_in_tf_env.sh [script.py]
#
# Defaults to check_tf_group_centering.py. Calling through this file avoids
# PowerShell -> WSL -> bash quoting problems.
# Note: the local tf_gpu env crashes in CPU-only mode (CUDA_VISIBLE_DEVICES=""), so keep the GPU visible.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
export TF_CPP_MIN_LOG_LEVEL=3
python -u "$(dirname "$0")/${1:-check_tf_group_centering.py}" 2>&1 | grep -v "cuBLAS factory"
