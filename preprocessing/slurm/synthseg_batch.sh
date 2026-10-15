#!/bin/bash
# SLURM array job: SynthSeg in batch mode (one mri_synthseg call per chunk,
# model loaded once) over rows [START, END) that already have image.nii.gz,
# run inside the tf10_nv.sif container (with --nv) via run_in_container.sh.
#
# Usage (on mlsc-login; the logs folder must exist):
#   N=$(( ($(tail -n +2 M.csv | wc -l) + 99) / 100 ))    # chunks of 100 rows
#   sbatch --array=0-$((N-1))%2 --export=ALL,MANIFEST=M.csv preprocessing/slurm/synthseg_batch.sh
#   sbatch --dependency=afterok:<run_array jobid> --array=0-$((N-1))%2 preprocessing/slurm/synthseg_batch.sh
#
# GPU by default (one GPU per job). CPU variant: delete the --gpus line, switch
# to the commented CPU lines below and set SYNTHSEG_CPU=1 (adds --synthseg-cpu,
# and the container is started without --nv).
# Works from a tcsh login shell (bash shebang); no rc files are sourced.
#
#SBATCH --job-name=bgr_synthseg
# EDIT: GPU partition (suggested: rtx8000; avoid qtim/pubgpu-req Blackwell and L40S nodes with TF 2.10)
#SBATCH --partition=PLACEHOLDER_GPU_PARTITION
# Martinos MLSC rule: always account qtim (never entre)
#SBATCH --account=qtim
# EDIT: uncomment (single #) only if your cluster requires a QOS
##SBATCH --qos=PLACEHOLDER_QOS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# EDIT: delete for the CPU variant (never more than one GPU)
#SBATCH --gpus=1
# EDIT
#SBATCH --time=04:00:00
# EDIT: rtx8000 fair share is ~128 GB per GPU
#SBATCH --mem=32G
# EDIT: passed as --threads; rtx8000 fair share is ~3 CPUs per GPU
#SBATCH --cpus-per-task=3
# EDIT: 0-(n-chunks - 1); %2 = at most two tasks at once (cluster rule)
#SBATCH --array=0-0%2
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x_%A_%a.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x_%A_%a.err
# CPU variant (suggested partition: basic, ~15 GB per core):
##SBATCH --partition=PLACEHOLDER_CPU_PARTITION
##SBATCH --cpus-per-task=4
##SBATCH --mem=48G
##SBATCH --time=12:00:00

set -euo pipefail

# === EDIT ME ===
PROJ=${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}
REPO_DIR=${REPO_DIR:-$PROJ/brain-group-rad}
CONFIG=${CONFIG:-$PROJ/data/preproc_config.yaml}
MANIFEST=${MANIFEST:-$PROJ/data/manifests/lumir.csv}
OUT_ROOT=${OUT_ROOT:-$PROJ/data/processed}
CHUNK_SIZE=${CHUNK_SIZE:-100}
SYNTHSEG_CPU=${SYNTHSEG_CPU:-0}
# Container settings (BGR_SIF, BGR_BINDS, BGR_TMPDIR, ...) are read by
# run_in_container.sh; defaults are the Martinos values, TMPDIR=/scratch/$SLURM_JOB_ID.
# === END EDIT ===

EXTRA=()
if [ "$SYNTHSEG_CPU" = "1" ]; then
    EXTRA+=(--synthseg-cpu)
    export USE_GPU=0
else
    export USE_GPU=1
fi
export BGR_REPO_DIR="$REPO_DIR"
RIC=(bash "$REPO_DIR/preprocessing/run_in_container.sh")
cd "$REPO_DIR"

TASK=${SLURM_ARRAY_TASK_ID:-0}
START=$((TASK * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))
echo "task ${TASK}: SynthSeg rows [${START}, ${END}) of ${MANIFEST} on $(hostname)"

status=0
"${RIC[@]}" synthseg --config "$CONFIG" --manifest "$MANIFEST" --out-root "$OUT_ROOT" \
    --start "$START" --end "$END" --threads "${SLURM_CPUS_PER_TASK:-3}" "${EXTRA[@]+"${EXTRA[@]}"}" || status=$?
echo "task ${TASK}: SynthSeg rows [${START}, ${END}) finished with exit status ${status}"
exit "$status"
