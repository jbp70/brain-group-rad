#!/bin/bash
# SLURM array job: export-npy (training export v2: per-level float32 images,
# uint8 train18 labels) over rows [START, END) of
# <OUT_ROOT>/<DATASET>/manifest_processed.csv, CPU only, inside the tf10_nv.sif
# container via preprocessing/run_in_container.sh. Each task writes its own
# manifest part (race-free); merge them afterwards with MERGE_ONLY=1.
#
# Usage (on mlsc-login, from anywhere; the logs folder must exist):
#   P=$PROJ/data/processed/lumir/manifest_processed.csv
#   N=$(( ($(tail -n +2 $P | wc -l) + 199) / 200 ))     # chunks of 200 rows
#   sbatch --array=0-$((N-1))%2 --export=ALL,SPLIT=train preprocessing/slurm/export_array.sh
#   # Only once that array has FINISHED (squeue -u jn85 empty; cluster rule: at most two of your
#   # jobs active, so don't queue the merge as a pending dependency next to a running array):
#   sbatch --array=0-0%2 --export=ALL,MERGE_ONLY=1 preprocessing/slurm/export_array.sh
#   Don't run an unchunked export-npy or export-merge while array tasks are still running: it
#   merges only the manifest parts that exist at that moment.
#
# Works from a tcsh login shell (bash shebang); no rc files are sourced.
# Rerunning is safe: complete cases are skipped, writes are atomic.
#
#SBATCH --job-name=bgr_export
# EDIT: CPU partition (suggested: basic)
#SBATCH --partition=basic
# Martinos MLSC rule: always account qtim (never entre)
#SBATCH --account=qtim
# EDIT: uncomment (single #) only if your cluster requires a QOS
##SBATCH --qos=PLACEHOLDER_QOS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# EDIT: per chunk (200 cases, a few seconds each)
#SBATCH --time=02:00:00
# EDIT: basic fair share is ~15 GB per core
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
# EDIT: 0-(n-chunks - 1); %2 = at most two tasks at once (cluster rule)
#SBATCH --array=0-0%2
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x_%A_%a.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x_%A_%a.err

set -euo pipefail

# === EDIT ME ===
PROJ=${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}
REPO_DIR=${REPO_DIR:-$PROJ/brain-group-rad}
CONFIG=${CONFIG:-$PROJ/data/preproc_config.yaml}
OUT_ROOT=${OUT_ROOT:-$PROJ/data/processed}
DATASET=${DATASET:-lumir}
EXPORT_ROOT=${EXPORT_ROOT:-$PROJ/data/npy_v2}
CHUNK_SIZE=${CHUNK_SIZE:-200}
SPLIT=${SPLIT:-any}
CONTRAST=${CONTRAST:-T1w}
LABEL_SETS=${LABEL_SETS:-train18}
LEVELS=${LEVELS:-1,2,4}
IMAGE_DTYPE=${IMAGE_DTYPE:-float32}
# 1 = task 0 also exports the per-level template (needs CONFIG)
INCLUDE_TEMPLATE=${INCLUDE_TEMPLATE:-1}
# 1 = only merge the manifest parts (submit after the array with --dependency=afterok)
MERGE_ONLY=${MERGE_ONLY:-0}
# Container settings (BGR_SIF, BGR_BINDS, BGR_TMPDIR, ...) are read by
# run_in_container.sh; defaults are the Martinos values, TMPDIR=/scratch/$SLURM_JOB_ID.
export USE_GPU=${USE_GPU:-0}
# === END EDIT ===

export BGR_REPO_DIR="$REPO_DIR"
RIC=(bash "$REPO_DIR/preprocessing/run_in_container.sh")
cd "$REPO_DIR"

status=0
if [ "$MERGE_ONLY" = "1" ]; then
    "${RIC[@]}" export-merge --export-root "$EXPORT_ROOT" || status=$?
    echo "export-merge into ${EXPORT_ROOT} finished with exit status ${status}"
    exit "$status"
fi

TASK=${SLURM_ARRAY_TASK_ID:-0}
START=$((TASK * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))
extra=()
if [ "$INCLUDE_TEMPLATE" = "1" ] && [ "$TASK" = "0" ]; then
    extra=(--include-template --config "$CONFIG")
fi
echo "task ${TASK}: rows [${START}, ${END}) of ${OUT_ROOT}/${DATASET}/manifest_processed.csv split=${SPLIT} on $(hostname)"
"${RIC[@]}" export-npy --out-root "$OUT_ROOT" --export-root "$EXPORT_ROOT" --dataset "$DATASET" \
    --start "$START" --end "$END" --split "$SPLIT" --contrast "$CONTRAST" --label-sets "$LABEL_SETS" \
    --levels "$LEVELS" --image-dtype "$IMAGE_DTYPE" ${extra[@]+"${extra[@]}"} || status=$?
echo "task ${TASK}: rows [${START}, ${END}) finished with exit status ${status}"
exit "$status"
