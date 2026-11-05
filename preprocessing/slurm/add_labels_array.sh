#!/bin/bash
# SLURM array job: export-add-labels (add-only; synth-contrast design 3.3/3.4). Adds the label set
# LABEL_SET (default synthfine: the 103-id SynthSeg --parc remap, nested majority vote at levels
# 2/4, consistent with labels_train18 voxel for voxel) to the EXISTING export EXPORT_ROOT, CPU
# only, inside the tf10_nv.sif container via preprocessing/run_in_container.sh.
#   PHASE=files     (default) task i handles export_manifest.csv data rows [i*CHUNK, (i+1)*CHUNK)
#                   (the last chunk is clipped); writes only labels_<set>{,_2,_4}.npy per case and
#                   EXPORT_ROOT/add_labels/<set>/rows-<start>-<end>.json; never modifies a file.
#   PHASE=manifest  once, after ALL file tasks finished: appends the set to the manifest, parts and
#                   LUTs (pre-check, backup in EXPORT_ROOT/backup_pre_<set>_<UTC>/, diff check,
#                   restore on failure). The array index is ignored.
#   PHASE=check     export-check over all cases (includes the synthfine consistency rule).
#
# Usage (on mlsc-login, from the repo clone; the logs folder must exist; check `squeue -u jn85`
# first: at most two of your jobs active, Jay's other jobs count):
#   P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
#   cd $P/brain-group-rad && git pull
#   mkdir -p $P/logs
#   N=$(( $(wc -l < $P/data/npy_v2/export_manifest.csv) - 1 )); echo $N    # 3423 expected
#   sbatch --array=0-$(( (N + 499) / 500 - 1 ))%2 --export=ALL,CHUNK=500,LABEL_SET=synthfine preprocessing/slurm/add_labels_array.sh
#   # only after the array has FINISHED (squeue -u jn85 shows none of its tasks); check
#   # $P/data/npy_v2/add_labels/synthfine/rows-*.json and the logs for failed cases:
#   sbatch --array=0-0%2 --export=ALL,PHASE=manifest,LABEL_SET=synthfine preprocessing/slurm/add_labels_array.sh
#   # verification (all cases; synthfine consistency rule included):
#   sbatch --array=0-0%2 --export=ALL,PHASE=check preprocessing/slurm/add_labels_array.sh
#
# Works from a tcsh login shell (bash shebang); no rc files are sourced.
# Rerunning phase files is safe: existing identical files are kept, writes are atomic.
#
#SBATCH --job-name=bgr_add_labels
# EDIT: CPU partition (suggested: basic)
#SBATCH --partition=basic
# Martinos MLSC rule: always account qtim (never entre)
#SBATCH --account=qtim
# EDIT: uncomment (single #) only if your cluster requires a QOS
##SBATCH --qos=PLACEHOLDER_QOS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# EDIT: per chunk (500 cases, ~3-6 s each; phase manifest re-checks all cases, ~10-20 min)
#SBATCH --time=02:00:00
# EDIT: basic fair share is ~15 GB per core (peak < 1.5 GB per case)
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
OUT_ROOT=${OUT_ROOT:-$PROJ/data/processed}
EXPORT_ROOT=${EXPORT_ROOT:-$PROJ/data/npy_v2}
# files | manifest | check
PHASE=${PHASE:-files}
# manifest rows per array task (phase files)
CHUNK=${CHUNK:-500}
LABEL_SET=${LABEL_SET:-synthfine}
# Container settings (BGR_SIF, BGR_BINDS, BGR_TMPDIR, ...) are read by
# run_in_container.sh; defaults are the Martinos values, TMPDIR=/scratch/$SLURM_JOB_ID.
export USE_GPU=${USE_GPU:-0}
# === END EDIT ===

case "$PHASE" in
    files|manifest|check) ;;
    *) echo "ERROR: PHASE must be files, manifest or check, got '${PHASE}'" >&2; exit 2 ;;
esac
if ! [[ "$CHUNK" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: CHUNK must be a positive integer, got '${CHUNK}'" >&2
    exit 2
fi

export BGR_REPO_DIR="$REPO_DIR"
RIC=(bash "$REPO_DIR/preprocessing/run_in_container.sh")
cd "$REPO_DIR"

status=0
if [ "$PHASE" = "manifest" ]; then
    echo "phase manifest: ${LABEL_SET} into ${EXPORT_ROOT} on $(hostname)"
    "${RIC[@]}" export-add-labels --export-root "$EXPORT_ROOT" --label-set "$LABEL_SET" --phase manifest \
        || status=$?
    echo "phase manifest finished with exit status ${status}"
    exit "$status"
fi
if [ "$PHASE" = "check" ]; then
    echo "phase check: export-check of all cases in ${EXPORT_ROOT} on $(hostname)"
    "${RIC[@]}" export-check --export-root "$EXPORT_ROOT" --sample 0 || status=$?
    echo "phase check finished with exit status ${status}"
    exit "$status"
fi

TASK=${SLURM_ARRAY_TASK_ID:-0}
START=$((TASK * CHUNK))
END=$((START + CHUNK))
echo "task ${TASK}: phase files, ${LABEL_SET}, manifest rows [${START}, ${END}) of ${EXPORT_ROOT}/export_manifest.csv on $(hostname)"
"${RIC[@]}" export-add-labels --out-root "$OUT_ROOT" --export-root "$EXPORT_ROOT" --label-set "$LABEL_SET" \
    --phase files --start "$START" --end "$END" || status=$?
echo "task ${TASK}: rows [${START}, ${END}) finished with exit status ${status}"
exit "$status"
