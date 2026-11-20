#!/bin/bash
#SBATCH --job-name=bgr-perturb
#SBATCH -A qtim
#SBATCH -p basic
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Data handling (Jay runs it): copy processed subjects with a known random affine per subject
# (scripts/perturb_processed.py; CPU, inside tf10_nv.sif). Default = the 39 LUMIR validation cases of
# npy_v3 (processed lumir_ants) at the affine TRAINING ranges (45 deg, 15 mm, scale +-10 %, shear
# 0.05), plus ONE shared rotation of exactly 30 deg for every subject (axis from the seed, about the
# grid centre, no shared translation; moves the group's central space off MNI), seed 2026
# -> data/processed/lumir_ants_val_affpert45_rot30 (paper figure 2). ~1-2 min per subject.
#
# Environment overrides: SRC, OUT, MANIFEST, SPLIT, LABEL_NAME, SEED, ROT, TRANS, LOGSCALE, SHEAR,
# SHARED_ROT (0 = none), DATASET, OVERWRITE=1.
#
#   sbatch slurm/perturb_processed.sh
set -uo pipefail
PROJ=${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}
REPO=${REPO:-$PROJ/brain-group-rad}
SIF=${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}
SRC=${SRC:-$PROJ/data/processed/lumir_ants}
OUT=${OUT:-$PROJ/data/processed/lumir_ants_val_affpert45_rot30}
MANIFEST=${MANIFEST:-$PROJ/data/npy_v3/export_manifest.csv}
SPLIT=${SPLIT:-val}
LABEL_NAME=${LABEL_NAME:-synthseg}
SEED=${SEED:-2026}
ROT=${ROT:-45}
TRANS=${TRANS:-15}
LOGSCALE=${LOGSCALE:-0.0953101798}   # ln 1.10
SHEAR=${SHEAR:-0.05}
SHARED_ROT=${SHARED_ROT:-30}
DATASET=${DATASET:-lumir_ants_val_affpert45_rot30}
OVERWRITE=${OVERWRITE:-0}
WORKERS=${SLURM_CPUS_PER_TASK:-8}
FLAGS=""
if [ "$OVERWRITE" = "1" ]; then FLAGS="--overwrite"; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  repo $(git -C "$REPO" log --oneline -1)"
# shellcheck disable=SC2086
singularity exec -B /autofs -B /scratch "$SIF" bash -c 'cd "$1" && shift && python "$@"' perturb "$REPO" \
    scripts/perturb_processed.py --src "$SRC" --out "$OUT" --manifest "$MANIFEST" --split "$SPLIT" \
    --label-name "$LABEL_NAME" --dataset "$DATASET" --seed "$SEED" --rotation-deg "$ROT" \
    --translation-mm "$TRANS" --log-scale "$LOGSCALE" --shear "$SHEAR" --shared-rotation-deg "$SHARED_ROT" \
    --workers "$WORKERS" $FLAGS </dev/null
rc=$?
echo "finished perturb_processed exit code $rc"
exit $rc
