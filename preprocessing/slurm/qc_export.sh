#!/bin/bash
# SLURM job: qc-summary for one dataset, then (optionally) export-npy, run
# inside the tf10_nv.sif container via preprocessing/run_in_container.sh.
#
# Usage (on mlsc-login; the logs folder must exist):
#   sbatch --dependency=afterok:<synthseg jobid> --export=ALL,DATASET=lumir,DO_EXPORT=1 preprocessing/slurm/qc_export.sh
# The export here is one unchunked export-npy run (it merges the manifest parts
# itself). For the full LUMIR training set use slurm/export_array.sh (chunked).
#
# Works from a tcsh login shell (bash shebang); no rc files are sourced.
#
#SBATCH --job-name=bgr_qc_export
# EDIT: CPU partition (suggested: basic)
#SBATCH --partition=basic
# Martinos MLSC rule: always account qtim (never entre)
#SBATCH --account=qtim
# EDIT: uncomment (single #) only if your cluster requires a QOS
##SBATCH --qos=PLACEHOLDER_QOS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# EDIT
#SBATCH --time=02:00:00
# EDIT: basic fair share is ~15 GB per core
#SBATCH --mem=16G
# EDIT
#SBATCH --cpus-per-task=2
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x_%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x_%j.err

set -euo pipefail

# === EDIT ME ===
PROJ=${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}
REPO_DIR=${REPO_DIR:-$PROJ/brain-group-rad}
CONFIG=${CONFIG:-$PROJ/data/preproc_config.yaml}
OUT_ROOT=${OUT_ROOT:-$PROJ/data/processed}
DATASET=${DATASET:-lumir}
DO_EXPORT=${DO_EXPORT:-0}
# New root for the v2 layout (per-level files); never the old $PROJ/data/npy
EXPORT_ROOT=${EXPORT_ROOT:-$PROJ/data/npy_v2}
# float32: augmentation runs in float32 in DALI; float16 cast happens at the model input
IMAGE_DTYPE=${IMAGE_DTYPE:-float32}
LABEL_SETS=${LABEL_SETS:-train18}
SPLIT=${SPLIT:-any}
# Container settings (BGR_SIF, BGR_BINDS, BGR_TMPDIR, ...) are read by
# run_in_container.sh; defaults are the Martinos values, TMPDIR=/scratch/$SLURM_JOB_ID.
export USE_GPU=${USE_GPU:-0}
# === END EDIT ===

export BGR_REPO_DIR="$REPO_DIR"
RIC=(bash "$REPO_DIR/preprocessing/run_in_container.sh")
cd "$REPO_DIR"

status=0
"${RIC[@]}" qc-summary --out-root "$OUT_ROOT" --dataset "$DATASET" || status=$?
if [ "$status" -eq 0 ] && [ "$DO_EXPORT" = "1" ]; then
    "${RIC[@]}" export-npy --out-root "$OUT_ROOT" --export-root "$EXPORT_ROOT" \
        --dataset "$DATASET" --image-dtype "$IMAGE_DTYPE" --label-sets "$LABEL_SETS" --split "$SPLIT" \
        --include-template --config "$CONFIG" || status=$?
fi
echo "qc/export for ${DATASET} finished with exit status ${status}"
exit "$status"
