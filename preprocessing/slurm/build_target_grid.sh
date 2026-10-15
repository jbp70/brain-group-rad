#!/bin/bash
# SLURM job: build-target-grid (mean of ~200 LUMIR images, flip/offset search
# against MNI152_T1_1mm_brain, writes target_grid.json + template_lumirgrid*),
# run inside the tf10_nv.sif container via preprocessing/run_in_container.sh.
#
# Usage (on mlsc-login; the logs folder must exist):
#   mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
#   sbatch preprocessing/slurm/build_target_grid.sh
#   sbatch --export=ALL,SPLIT=val preprocessing/slurm/build_target_grid.sh   # only imagesVal
# Exit status 2 = ambiguous flip (see the log and target_grid_candidates.json);
# verify independently and resubmit with FORCE_FLIP=yes or FORCE_FLIP=no.
# Works from a tcsh login shell (bash shebang); no rc files are sourced.
#
#SBATCH --job-name=bgr_target_grid
# EDIT: CPU partition (suggested: basic)
#SBATCH --partition=PLACEHOLDER_PARTITION
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
LUMIR_ROOT=${LUMIR_ROOT:-/autofs/space/crater_001/datasets/public/LUMIR2025}
TARGET_DIR=${TARGET_DIR:-$PROJ/data/target_grid}
SPLIT=${SPLIT:-any}
N_IMAGES=${N_IMAGES:-200}
SEED=${SEED:-0}
FORCE_FLIP=${FORCE_FLIP:-}
# Container settings (BGR_SIF, BGR_BINDS, BGR_TMPDIR, ...) are read by
# run_in_container.sh; defaults are the Martinos values, TMPDIR=/scratch/$SLURM_JOB_ID.
export USE_GPU=${USE_GPU:-0}
# === END EDIT ===

export BGR_REPO_DIR="$REPO_DIR"
RIC=(bash "$REPO_DIR/preprocessing/run_in_container.sh")
cd "$REPO_DIR"

EXTRA=()
if [ -n "$FORCE_FLIP" ]; then
    EXTRA+=(--force-flip "$FORCE_FLIP")
fi
status=0
"${RIC[@]}" build-target-grid --config "$CONFIG" --lumir-root "$LUMIR_ROOT" --split "$SPLIT" \
    --out-dir "$TARGET_DIR" --n "$N_IMAGES" --seed "$SEED" "${EXTRA[@]+"${EXTRA[@]}"}" || status=$?
echo "build-target-grid finished with exit status ${status}"
exit "$status"
