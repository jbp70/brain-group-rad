#!/bin/bash
#SBATCH --job-name=m9oomdiag
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# m9 level-1 OOM diagnostic (.agents/tasks/m9-level1-oom/report.md section 5). Copy to
# slurm/m9_oom_diag.sh and the Python tool to tests/gpu/diag_l1_memory.py, commit, push, pull.
# Runs (each under `timeout`, records in $OUT_DIR/*.jsonl, stdout/stderr in $OUT_DIR/*.log):
#   R1 dali        DALI + validation sets only, intensity aug on   -> DALI device footprint
#   R2 dali_noaug  same, --intensity-aug false                      -> share of the augmentation ops
#   R3 growth      real main.py = D1 config (memory growth), 2 steps, TF_CPP_MIN_LOG_LEVEL=0 (BFC dump)
#   R4 limit       R3 + --gpu-memory-limit-mb $LIMIT_MB (one BFC region, no growth), 6 steps
#   R5 async       R3 + TF_GPU_ALLOCATOR=cuda_malloc_async, 4 steps, 10 min cap (the D2 stall)
# Submit (after Jay's approval; <= 2 active jobs):
#   sbatch slurm/m9_oom_diag.sh            (optional env: LIMIT_MB=38000 RUN_ASYNC=1)
set -uo pipefail

PROJ="${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}"
REPO="${REPO:-$PROJ/brain-group-rad}"
DATA="${DATA:-$PROJ/data/npy_v2}"
RESULTS="${RESULTS:-$PROJ/results}"
SIF="${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}"
JOB_ID="${SLURM_JOB_ID:-manual}"
CASCADE_FROM="${CASCADE_FROM:-$RESULTS/m8_level2_gradnorm/ckpt/best}"
LIMIT_MB="${LIMIT_MB:-38000}"
RUN_ASYNC="${RUN_ASYNC:-1}"
OUT_DIR="$PROJ/scratch/m9_oom_diag/$JOB_ID"
mkdir -p "$OUT_DIR" "$PROJ/scratch/keras" "$PROJ/scratch/cuda_cache"
export TMPDIR="/scratch/$JOB_ID"; mkdir -p "$TMPDIR"
export SINGULARITYENV_TMPDIR="$TMPDIR"
export SINGULARITYENV_KERAS_HOME="$PROJ/scratch/keras"
export SINGULARITYENV_CUDA_CACHE_PATH="$PROJ/scratch/cuda_cache"
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1

cd "$REPO"
echo "host $(hostname) | job $JOB_ID | repo $(git rev-parse --short HEAD) | $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

# Same recipe as slurm/m9_level1_cascade.sh with the D1 overrides (1 pair, no BN recal, no
# validation, no PNGs) and log-every 1.
BASE=(--data "$DATA" --results "$RESULTS" --level 1 --cascade-from "$CASCADE_FROM"
      --group-size 8 --deform-group-size 4 --task-ratio rigid:1,affine:1,deform:2
      --val-every 1000 --ckpt-every 1000 --warmup-steps 50 --task-grad-norm 12
      --lambda-ncc-template 1 --lambda-ncc-pair 1 --lambda-smooth 0.5 --p-pair 0.1 --p-pdice 0.1
      --min-pairs 1 --bn-recal-batches 0 --cascade-skip-prob 0 --amp true
      --gradient-checkpointing full --lean-sampler true --split-group-conv true --lean-losses false
      --val-pair-chunk 4 --global-transform true --global-rot-max-deg 45 --global-trans-max-mm 10
      --global-start-step 0 --global-ramp-steps 0 --curriculum false --val-png false
      --num-threads "${SLURM_CPUS_PER_TASK:-3}" --prefetch 1 --log-every 1 --log-backend jsonl)
RUNC=(singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF")
TOOL=tests/gpu/diag_l1_memory.py

run() {   # run <name> <timeout s> <env assignments...> -- <tool args...>
    local name="$1" tmo="$2"; shift 2
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    echo "=== $name start $(date -Is)"
    env "${envs[@]}" SINGULARITYENV_DIAG_OUT="$OUT_DIR/$name.jsonl" DIAG_OUT="$OUT_DIR/$name.jsonl" \
        timeout "$tmo" "${RUNC[@]}" python "$TOOL" "$@" </dev/null >"$OUT_DIR/$name.log" 2>&1
    echo "=== $name exit $? $(date -Is)"
    grep -E '"phase": "(after_dali|fetch|idle|step_|exit|main_returned)' "$OUT_DIR/$name.jsonl" | tail -n 25 || true
    grep -E 'Limit:|InUse:|MaxInUse:|Reserved:|PeakReserved:|LargestFreeBlock:|Sum Total of in-use chunks|total_region_allocated_bytes|ran out of memory|Created device' \
        "$OUT_DIR/$name.log" | head -n 20 || true
}

run dali 900 SINGULARITYENV_TF_CPP_MIN_LOG_LEVEL=1 -- dali "${BASE[@]}" --run-name "m9_oomdiag_${JOB_ID}_dali"
run dali_noaug 900 SINGULARITYENV_TF_CPP_MIN_LOG_LEVEL=1 -- dali "${BASE[@]}" --intensity-aug false \
    --run-name "m9_oomdiag_${JOB_ID}_dali_noaug"
run growth 900 SINGULARITYENV_TF_CPP_MIN_LOG_LEVEL=0 -- main "${BASE[@]}" --steps 2 \
    --run-name "m9_oomdiag_${JOB_ID}_growth"
run limit 1200 SINGULARITYENV_TF_CPP_MIN_LOG_LEVEL=1 -- main "${BASE[@]}" --steps 6 \
    --gpu-memory-limit-mb "$LIMIT_MB" --run-name "m9_oomdiag_${JOB_ID}_limit"
if [ "$RUN_ASYNC" = "1" ]; then
    run async 600 SINGULARITYENV_TF_CPP_MIN_LOG_LEVEL=1 SINGULARITYENV_TF_GPU_ALLOCATOR=cuda_malloc_async -- \
        main "${BASE[@]}" --steps 4 --run-name "m9_oomdiag_${JOB_ID}_async"
fi
echo "done $(date -Is) | records in $OUT_DIR"
