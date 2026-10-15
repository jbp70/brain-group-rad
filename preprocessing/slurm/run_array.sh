#!/bin/bash
# SLURM array job: rows [START, END) of a manifest through
# reorient/strip/N4/register/resample/normalize/qc (CPU only), run inside the
# tf10_nv.sif container via preprocessing/run_in_container.sh.
#
# Usage (on mlsc-login, from anywhere; the logs folder must exist):
#   mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
#   N=$(( ($(tail -n +2 M.csv | wc -l) + 19) / 20 ))     # chunks of 20 rows (no python on the login node)
#   sbatch --array=0-$((N-1))%2 --export=ALL,MANIFEST=M.csv preprocessing/slurm/run_array.sh
#   (chain SynthSeg with: sbatch --dependency=afterok:<jobid> preprocessing/slurm/synthseg_batch.sh)
#
# Works from a tcsh login shell: sbatch runs this file with bash (shebang).
# No rc files are sourced and no Python env is activated: Python, numpy,
# nibabel etc. come from the container. Rerunning is safe: finished steps/rows are skipped.
#
#SBATCH --job-name=bgr_preproc
# EDIT: CPU partition (suggested: basic)
#SBATCH --partition=basic
# Martinos MLSC rule: always account qtim (never entre)
#SBATCH --account=qtim
# EDIT: uncomment (single #) only if your cluster requires a QOS
##SBATCH --qos=PLACEHOLDER_QOS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# EDIT: per chunk
#SBATCH --time=04:00:00
# EDIT: basic fair share is ~15 GB per core
#SBATCH --mem=16G
# EDIT: passed as --threads
#SBATCH --cpus-per-task=4
# EDIT: 0-(n-chunks - 1); %2 = at most two tasks at once (cluster rule)
#SBATCH --array=0-0%2
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x_%A_%a.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x_%A_%a.err

set -euo pipefail

# === EDIT ME ===
PROJ=${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}
REPO_DIR=${REPO_DIR:-$PROJ/brain-group-rad}
CONFIG=${CONFIG:-$PROJ/data/preproc_config.yaml}
MANIFEST=${MANIFEST:-$PROJ/data/manifests/lumir.csv}
OUT_ROOT=${OUT_ROOT:-$PROJ/data/processed}
CHUNK_SIZE=${CHUNK_SIZE:-20}
# Container settings (BGR_SIF, BGR_BINDS, BGR_TMPDIR, ...) are read by
# run_in_container.sh; defaults are the Martinos values, TMPDIR=/scratch/$SLURM_JOB_ID.
export USE_GPU=${USE_GPU:-0}
# === END EDIT ===

export BGR_REPO_DIR="$REPO_DIR"
RIC=(bash "$REPO_DIR/preprocessing/run_in_container.sh")
cd "$REPO_DIR"

TASK=${SLURM_ARRAY_TASK_ID:-0}
START=$((TASK * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))
echo "task ${TASK}: rows [${START}, ${END}) of ${MANIFEST} on $(hostname)"

status=0
"${RIC[@]}" run --config "$CONFIG" --manifest "$MANIFEST" --out-root "$OUT_ROOT" \
    --start "$START" --end "$END" --threads "${SLURM_CPUS_PER_TASK:-4}" || status=$?
echo "task ${TASK}: rows [${START}, ${END}) finished with exit status ${status}"
exit "$status"
