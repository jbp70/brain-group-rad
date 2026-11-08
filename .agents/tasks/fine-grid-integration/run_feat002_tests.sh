#!/usr/bin/env bash
# FEAT-002 verification (fine-grid-integration). Usage: bash run_feat002_tests.sh <tag> [modules...]
# Each module's full log goes to .agents/tasks/fine-grid-integration/logs/<tag>_<module>.log.
set -o pipefail
W=/mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.worktrees/fine-grid-integration
cd "$W" || exit 1
L="$W/.agents/tasks/fine-grid-integration/logs"
mkdir -p "$L"
TAG="$1"
shift
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
export TF_CPP_MIN_LOG_LEVEL=3 TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 BGR_TF_GPU_LIMIT_MB=6000 TF_CUDNN_USE_AUTOTUNE=0 PYTHONDONTWRITEBYTECODE=1
rc=0
for m in "$@"; do
    t0=$(date +%s)
    python -m unittest "$m" -v > "$L/${TAG}_${m}.log" 2>&1
    r=$?
    echo "=== $m rc $r $(( $(date +%s) - t0 )) s :: $(grep -E '^(Ran|OK|FAILED)' "$L/${TAG}_${m}.log" | tr '\n' ' ')"
    [ "$r" -ne 0 ] && rc=1
done
echo "ALL rc $rc"
