#!/usr/bin/env bash
# Full-width synthetic overfit runs of main.py on the local GTX 1080 (milestone 4, item 9).
# Usage: bash overfit_full.sh <name> <extra main.py args...>
# Writes results/m4_local/<name>/ (git-ignored) and copies overfit_summary.json to this folder.
REPO=/mnt/d/Research/Brain_Registration_Projects/brain-group-rad
TASK="$REPO/.agents/tasks/milestone-4"
cd "$REPO"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONDONTWRITEBYTECODE=1
# WSL / 8 GB GTX 1080: cap TF memory and the cuDNN workspace, otherwise the GPU pages to host RAM
# and a full-width step takes minutes instead of ~0.8 s.
export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512
NAME="$1"; shift
python -u main.py --overfit-one-group --overfit-synthetic --level 4 --group-size 4 \
    --results results/m4_local --run-name "$NAME" --resume false --log-backend jsonl \
    --log-every 50 --val-every 250 --gpu-memory-limit-mb 6000 "$@" 2>&1 | grep -v "cuBLAS factory" \
    | tee "$TASK/overfit_$NAME.log" | grep -E "^\[|overfit|run dir|Error|error"
rc=${PIPESTATUS[0]}
cp "results/m4_local/$NAME/overfit_summary.json" "$TASK/overfit_summary_$NAME.json" 2>/dev/null
echo "exit: $rc"
