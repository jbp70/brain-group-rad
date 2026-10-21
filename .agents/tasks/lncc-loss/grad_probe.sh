#!/usr/bin/env bash
# LNCC loss item 8a: per-term deform gradient norms (grad_probe.py -> grad_probe.json).
set -o pipefail
REPO=/mnt/d/Research/Brain_Registration_Projects/brain-group-rad
cd "$REPO"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONDONTWRITEBYTECODE=1
export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 BGR_TF_GPU_LIMIT_MB=6000 TF_CUDNN_USE_AUTOTUNE=0
python .agents/tasks/lncc-loss/grad_probe.py "$@" 2>&1 | grep -v "cuBLAS factory"
echo "exit ${PIPESTATUS[0]}"
