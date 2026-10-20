#!/usr/bin/env bash
# Run the milestone-2 DALI verification (or the unittest suite) on glacier, inside tf10_nv.sif,
# on ONE GPU.
#
# Usage (on glacier, after `git pull` in the clone; pick a free GPU with nvidia-smi first):
#   bash /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad/tests/gpu/run_verify_glacier.sh <gpu_index> [timeout_seconds] [verify|unittest] [checks]
#
# Modes:
#   verify   (default) tests/gpu/verify_dali_pipeline.py -> summary.json, verify.log
#            [checks]: comma-separated subset of a-l (default: all; e.g. h,i,j,k for the
#            curriculum / global-transform checks, l for the deformable image/label pipeline)
#   unittest python -m unittest discover -s tests -t . -v -> unittest.log (the DALI test runs
#            here instead of skipping; the tests' tempfile folders go to TMPDIR = scratch)
#
# Writes only inside $PROJ/scratch/m2: summary.json, verify.log / unittest.log, the CUDA JIT
# cache, the Keras config folder and temporary folders (removed at the end; the control files
# of checks h-k live there too). Bytecode writing is disabled, so the repo clone stays clean.
# Exit code: that of the Python command (1 = a check / test failed, 124 = timeout).
set -euo pipefail

GPU="${1:?usage: run_verify_glacier.sh <gpu_index> [timeout_seconds] [verify|unittest] [checks]}"
TIMEOUT_S="${2:-1800}"
MODE="${3:-verify}"
CHECKS="${4:-}"
if ! [[ "$GPU" =~ ^[0-9]+$ ]]; then
    echo "gpu_index must be a single integer, got '$GPU'" >&2
    exit 2
fi
if ! [[ "$TIMEOUT_S" =~ ^[0-9]+$ ]]; then
    echo "timeout_seconds must be an integer, got '$TIMEOUT_S'" >&2
    exit 2
fi
if [[ "$MODE" != "verify" && "$MODE" != "unittest" ]]; then
    echo "mode must be 'verify' or 'unittest', got '$MODE'" >&2
    exit 2
fi
if [[ -n "$CHECKS" ]] && ! [[ "$CHECKS" =~ ^[a-l](,[a-l])*$ ]]; then
    echo "checks must be a comma-separated list of letters a-l, got '$CHECKS'" >&2
    exit 2
fi

PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
REPO="$PROJ/brain-group-rad"
DATA="$PROJ/data/npy_v2"
SCRATCH="$PROJ/scratch/m2"
SIF=/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif

mkdir -p "$SCRATCH"
cd "$REPO"

# One GPU only; temp files, CUDA cache, Keras config (~/.keras otherwise; Singularity binds
# $HOME) and bytecode kept out of shared / unbound / unapproved locations.
export CUDA_VISIBLE_DEVICES="$GPU"
export TMPDIR="$SCRATCH"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_CACHE_PATH="$SCRATCH/cuda_cache"
export KERAS_HOME="$SCRATCH/keras"
export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU"
export SINGULARITYENV_TMPDIR="$SCRATCH"
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_CUDA_CACHE_PATH="$SCRATCH/cuda_cache"
export SINGULARITYENV_KERAS_HOME="$SCRATCH/keras"

echo "host $(hostname) | GPU $GPU | mode $MODE | checks ${CHECKS:-default} | timeout ${TIMEOUT_S}s | repo $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
nvidia-smi -i "$GPU" --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

if [[ "$MODE" == "verify" ]]; then
    CMD=(python tests/gpu/verify_dali_pipeline.py --data "$DATA" --scratch "$SCRATCH"
         --out "$SCRATCH/summary.json")
    if [[ -n "$CHECKS" ]]; then
        CMD+=(--checks "$CHECKS")
    fi
    LOG="$SCRATCH/verify.log"
else
    CMD=(python -m unittest discover -s tests -t . -v)
    LOG="$SCRATCH/unittest.log"
fi

set +e
timeout "$TIMEOUT_S" singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF" \
    "${CMD[@]}" </dev/null 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "$MODE exit code: $rc (log: $LOG)"
exit "$rc"
