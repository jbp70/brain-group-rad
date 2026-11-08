#!/usr/bin/env bash
# FEAT-001 verification (fine-grid-integration). Usage: bash run_feat001_tests.sh [modules...]
set -o pipefail
W=/mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.worktrees/fine-grid-integration
cd "$W" || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
export TF_CPP_MIN_LOG_LEVEL=3 TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 BGR_TF_GPU_LIMIT_MB=6000 TF_CUDNN_USE_AUTOTUNE=0 PYTHONDONTWRITEBYTECODE=1
MODS="${*:-tests.test_fine_grid_integration}"
rc=0
for m in $MODS; do
    echo "=== $m"
    python -m unittest "$m" -v 2>&1 | tail -n 400
    r=${PIPESTATUS[0]}
    echo "=== $m rc $r"
    [ "$r" -ne 0 ] && rc=1
done
echo "ALL rc $rc"
