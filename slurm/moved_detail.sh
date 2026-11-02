#!/bin/bash
#SBATCH --job-name=bgr-detail
#SBATCH -A qtim
#SBATCH -p basic
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Per-image cortical detail check on Mindboggle (scripts/moved_detail.py, CPU only, tf10_nv.sif):
# native vs moved cortical gradient, sulcal-CSF fraction/volume and log-Jacobian in the cortex, for
# our retrained sweep checkpoint (step 3000) and MultiMorph. Output $OUT/{per_subject.csv,summary.json}.
# DO NOT submit without Jay's approval (cluster-access rules).
set -uo pipefail
PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
REPO=${REPO:-$PROJ/brain-group-rad}
SIF=${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}
OURS=$PROJ/results/predict_mindboggle/r1sw_lr1e-4_s3000_deform/fields
MM=$PROJ/baselines/multimorph/mindboggle_results
OUT=${OUT:-$PROJ/results/atlas_sharpness/mindboggle_per_image}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}
ARGS=(--processed-dir "$PROJ/data/processed/mindboggle" --out "$OUT" --workers "$WORKERS"
      --method "ours_r1sw_s3000=$OURS/{subject_dir}_moved.nii.gz,$OURS/{subject_dir}_fwd.nii.gz"
      --method "multimorph=$MM/{subject_dir}_moved.nii.gz,$MM/{subject_dir}_fwd.nii.gz")
mkdir -p "$OUT"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  repo $(git -C "$REPO" log --oneline -1)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
singularity exec -B /autofs "$SIF" bash -c 'cd "$1" && shift && python scripts/moved_detail.py "$@"' \
    detail "$REPO" "${ARGS[@]}" </dev/null
rc=$?
echo "exit status $rc"
exit $rc
