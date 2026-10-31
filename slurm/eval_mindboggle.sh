#!/bin/bash
#SBATCH --job-name=bgr-eval
#SBATCH -A qtim
#SBATCH -p basic
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# CPU evaluation of one method on the processed Mindboggle grid (python -m evaluation run, inside
# tf10_nv.sif; numpy/scipy only). See evaluation/evaluate.py for the metrics and outputs.
# DO NOT submit without Jay's approval (cluster-access rules).
#
# Environment overrides (exported by sbatch):
#   METHOD      name for summary.json and the output folder (required)
#   FWD, INV    field path templates with {subject} / {subject_dir} (not needed with IDENTITY=1)
#   ATLAS, ATLAS_SEG   method's atlas image / segmentation (optional: mean image / majority vote)
#   CONV        convention flags from `check-convention`, e.g. "--components zyx --signs +++ --units mm"
#   IDENTITY=1  no-registration reference
#   PAIRS       all (default) | none | number of seeded random pairs
#   EXTRA       more flags, e.g. "--hd95"
#   OUT         default $PROJ/results/eval_mindboggle/$METHOD
#
# Cost (measured locally, 160x224x192, one core): ~4 s per pair (compose 1.6 s, LNCC 0.9 s,
# warps 0.8 s, Jacobian 0.4 s) -> 101 subjects = 10100 pairs ~11 CPU-h, ~1.5-2 h on 8 workers;
# --hd95 adds ~7 s per pair (~100 labels). Memory: ~11 GB of subject arrays + ~8.4 GB preloaded
# forward fields + ~1-2 GB per worker.
#
#   METHOD=baseline FWD='.../{subject}_fwd.nii.gz' INV='.../{subject}_inv.nii.gz' \
#     ATLAS=.../atlas.nii.gz ATLAS_SEG=.../atlas_seg.nii.gz CONV="--components xyz --signs +++ --units mm" \
#     sbatch slurm/eval_mindboggle.sh
#   METHOD=identity IDENTITY=1 sbatch slurm/eval_mindboggle.sh
set -uo pipefail
PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
REPO=${REPO:-$PROJ/brain-group-rad}
SIF=${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}
PROCESSED=${PROCESSED:-$PROJ/data/processed/mindboggle}
METHOD=${METHOD:?set METHOD}
OUT=${OUT:-$PROJ/results/eval_mindboggle/$METHOD}
PAIRS=${PAIRS:-all}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}
ARGS=(run --processed-dir "$PROCESSED" --method "$METHOD" --out "$OUT" --pairs "$PAIRS" --workers "$WORKERS")
if [ "${IDENTITY:-0}" = "1" ]; then
    ARGS+=(--identity)
else
    ARGS+=(--fwd "${FWD:?set FWD}" --inv "${INV:?set INV}")
    [ -n "${ATLAS:-}" ] && ARGS+=(--atlas "$ATLAS")
    [ -n "${ATLAS_SEG:-}" ] && ARGS+=(--atlas-seg "$ATLAS_SEG")
fi
# shellcheck disable=SC2206
ARGS+=(${CONV:-} ${EXTRA:-})
mkdir -p "$OUT"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  repo $(git -C "$REPO" log --oneline -1)"
echo "python -m evaluation ${ARGS[*]}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
singularity exec -B /autofs -B /scratch "$SIF" bash -c 'cd "$1" && shift && python -m evaluation "$@"' \
    eval "$REPO" "${ARGS[@]}" </dev/null
rc=$?
echo "exit status $rc"
exit $rc
