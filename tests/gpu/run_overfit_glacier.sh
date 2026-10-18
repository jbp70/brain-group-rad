#!/usr/bin/env bash
# Milestone-4 checks on glacier, inside tf10_nv.sif, on ONE GPU:
#   overfit  main.py --overfit-one-group on real level-4 training images (N = 8, fixed rigid
#            transforms, TF-side warp; must reach a 10-step mean centered RMS < 1 mm)
#   dryrun   main.py --dry-run with the real DALI pipeline (N = 8): builds the DALI datasets,
#            the validation set (1 group), runs 2 steps + 1 validation, writes a checkpoint
#
# Usage (on glacier, after `git pull` in the clone; pick a free GPU with nvidia-smi first):
#   bash /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad/tests/gpu/run_overfit_glacier.sh <gpu_index> [timeout_seconds] [overfit|dryrun]
#
# Writes only inside $PROJ/scratch/m4: results/<run>/ (config.json, logs, tb, ckpt,
# overfit_summary.json), <mode>.log, the CUDA JIT cache, the Keras config folder and TMPDIR.
# TF32 is turned off by main.py (models.precision.configure_tf_precision). Bytecode writing is
# disabled, so the repo clone stays clean. Exit code: that of main.py (overfit: 1 = threshold
# not reached; 124 = timeout).
set -euo pipefail
GPU="${1:?usage: run_overfit_glacier.sh <gpu_index> [timeout_seconds] [overfit|dryrun]}"
TIMEOUT_S="${2:-3600}"
MODE="${3:-overfit}"
if ! [[ "$GPU" =~ ^[0-9]+$ ]]; then
    echo "gpu_index must be a single integer, got '$GPU'" >&2
    exit 2
fi
if ! [[ "$TIMEOUT_S" =~ ^[0-9]+$ ]]; then
    echo "timeout_seconds must be an integer, got '$TIMEOUT_S'" >&2
    exit 2
fi
if [[ "$MODE" != "overfit" && "$MODE" != "dryrun" ]]; then
    echo "mode must be 'overfit' or 'dryrun', got '$MODE'" >&2
    exit 2
fi
PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
REPO="$PROJ/brain-group-rad"
DATA="$PROJ/data/npy_v2"
SCRATCH="$PROJ/scratch/m4"
SIF=/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$SCRATCH/results" "$SCRATCH/tmp"
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
export TMPDIR="$SCRATCH/tmp"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_CACHE_PATH="$SCRATCH/cuda_cache"
export KERAS_HOME="$SCRATCH/keras"
export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU"
export SINGULARITYENV_TMPDIR="$SCRATCH/tmp"
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_CUDA_CACHE_PATH="$SCRATCH/cuda_cache"
export SINGULARITYENV_KERAS_HOME="$SCRATCH/keras"
echo "host $(hostname) | GPU $GPU | mode $MODE | timeout ${TIMEOUT_S}s | repo $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
nvidia-smi -i "$GPU" --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
if [[ "$MODE" == "overfit" ]]; then
    CMD=(python main.py --overfit-one-group --data "$DATA" --level 4 --group-size 8
         --overfit-max-steps 3000 --warmup-steps 100 --results "$SCRATCH/results" --run-name "overfit_$STAMP"
         --log-every 50 --val-every 250 --log-backend jsonl)
else
    CMD=(python main.py --dry-run --data "$DATA" --level 4 --group-size 8
         --results "$SCRATCH/results" --run-name "dryrun_$STAMP" --num-threads 4 --log-backend jsonl)
fi
LOG="$SCRATCH/$MODE.log"
set +e
timeout "$TIMEOUT_S" singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF" \
    "${CMD[@]}" </dev/null 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "$MODE exit code: $rc (log: $LOG, results: $SCRATCH/results)"
exit "$rc"
