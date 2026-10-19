#!/usr/bin/env bash
# Synthetic overfit runs of main.py on the local GTX 1080 (milestone 5: affine and mixed).
# Usage: bash overfit_run.sh <name> <extra main.py args...>
# Writes results/m5_local/<name>/ (git-ignored), the log and a copy of overfit_summary.json here.
REPO=/mnt/d/Research/Brain_Registration_Projects/brain-group-rad
TASK="$REPO/.agents/tasks/milestone-5"
cd "$REPO"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONDONTWRITEBYTECODE=1
export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 TF_CUDNN_USE_AUTOTUNE=0
NAME="$1"; shift
python -u main.py --overfit-one-group --overfit-synthetic --level 4 --group-size 4 \
    --optimizer adam --lr 3e-4 --schedule constant --warmup-steps 0 \
    --results results/m5_local --run-name "$NAME" --resume false --log-backend jsonl \
    --log-every 50 --val-every 250 --gpu-memory-limit-mb 6000 "$@" 2>&1 | grep -v "cuBLAS factory" \
    > "$TASK/overfit_$NAME.log"
rc=${PIPESTATUS[0]}
cp "results/m5_local/$NAME/overfit_summary.json" "$TASK/overfit_summary_$NAME.json" 2>/dev/null
grep -E "overfit:|Error|error" "$TASK/overfit_$NAME.log" | tail -5
echo "exit: $rc"
