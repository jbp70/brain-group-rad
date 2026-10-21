#!/usr/bin/env bash
# LNCC loss item 7: local deform-step peak memory with LNCC off (0/0) and on (1/1), level 4,
# narrow N = 2, 3, 4 and default widths N = 2 (GTX 1080, 6000 MB TF limit). Writes memory.json.
set -o pipefail
REPO=/mnt/d/Research/Brain_Registration_Projects/brain-group-rad
OUT=$REPO/.agents/tasks/lncc-loss/memory.json
cd "$REPO"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONDONTWRITEBYTECODE=1
export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 TF_CUDNN_USE_AUTOTUNE=0
rm -f "$OUT"
status=0
for ncc in 0 1; do
    for spec in "narrow 2,3,4" "default 2"; do
        set -- $spec
        python tests/gpu/measure_deform_memory.py --n "$2" --widths "$1" --steps 5 \
            --gpu-memory-limit-mb 6000 --lambda-ncc-template $ncc --lambda-ncc-pair $ncc \
            --out "$OUT" 2>&1 | grep -v "cuBLAS factory" || status=1
    done
done
echo "status $status"
exit $status
