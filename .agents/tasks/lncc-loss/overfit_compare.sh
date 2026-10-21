#!/usr/bin/env bash
# LNCC loss item 8b: synthetic deform overfit (fixed 600-step budget) A Dice-only, B Dice+LNCC,
# C LNCC-only, through main.py with the flags of tests/test_deform_overfit.py. Results under
# .agents/tmp/lncc-overfit/; summaries copied to overfit_{A,B,C}.json; table by overfit_table.py.
set -o pipefail
REPO=/mnt/d/Research/Brain_Registration_Projects/brain-group-rad
HERE=$REPO/.agents/tasks/lncc-loss
RES=$REPO/.agents/tmp/lncc-overfit
cd "$REPO"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONDONTWRITEBYTECODE=1
export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 BGR_TF_GPU_LIMIT_MB=6000 TF_CUDNN_USE_AUTOTUNE=0
NARROW="--filters 16,32,32,48,64,96 --reductions 64,48,24,24 --dense 512,256"
FLAGS="--overfit-one-group --overfit-synthetic --task-ratio deform:1 --level 4 --group-size 4
 --optimizer adam --lr 3e-4 --schedule constant --warmup-steps 0 --overfit-max-steps 600
 --overfit-dice-threshold 1.01 --val-every 100 --log-every 50 --log-backend jsonl --quiet true
 --seed 0 --bn-momentum 0.9 --gpu-memory-limit-mb 6000 --results $RES $NARROW"
declare -A EXTRA
EXTRA[A]=""
EXTRA[B]="--lambda-ncc-template 1 --lambda-ncc-pair 1"
EXTRA[C]="--lambda-dice-template 0 --lambda-pdice 0 --lambda-ncc-template 1 --lambda-ncc-pair 1"
rm -rf "$RES"
status=0
for run in A B C; do
    start=$(date +%s)
    # shellcheck disable=SC2086
    python main.py $FLAGS ${EXTRA[$run]} --run-name "run_$run" 2>&1 | grep -v "cuBLAS factory" | tail -5
    rc=${PIPESTATUS[0]}
    echo "run $run exit $rc wall $(( $(date +%s) - start )) s"
    # exit 1 = overfit threshold not reached (expected: 1.01 is unreachable)
    [ "$rc" -le 1 ] || status=1
    cp "$RES/run_$run/overfit_summary.json" "$HERE/overfit_$run.json" || status=1
done
python "$HERE/overfit_table.py" "$RES" || status=1
echo "status $status"
exit $status
