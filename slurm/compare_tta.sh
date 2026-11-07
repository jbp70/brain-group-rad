#!/bin/bash
#SBATCH --job-name=bgr-tta-compare
#SBATCH -A qtim
#SBATCH -p basic
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Paired TTA comparison (scripts/compare_tta.py, TTA design 9.3) of two arms on the same subjects,
# CPU only (numpy + scipy inside tf10_nv.sif; seconds). Never run it on the login node.
# DO NOT submit without Jay's approval (cluster-access rules; at most two active jobs).
#
# Environment:
#   A      arm A directory (reference, --tta none)          required
#   B      arm B directory (--tta flip8)                     required
#   OUT    output directory (compare.json / .csv / .md)      required
#   KIND   eval (python -m evaluation run outputs, default) | linear (predict.py linear-eval outputs)
#   REPO   repo clone (default $PROJ/brain-group-rad)
#   SIF    container (default tf10_nv.sif)
#
# Examples (P = /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg):
#   all-pairs evaluations of both arms (eval kind):
#     A=$P/results/eval_mindboggle/<NAME>_tta_none B=$P/results/eval_mindboggle/<NAME>_tta_flip8 \
#       OUT=$P/results/tta_compare/<NAME>_allpairs sbatch slurm/compare_tta.sh
#   linear-eval runs (linear kind):
#     A=$P/results/linear_eval_mindboggle/<NAME>_rigid_tta_none B=$P/results/linear_eval_mindboggle/<NAME>_rigid_tta_flip8 \
#       OUT=$P/results/tta_compare/<NAME>_rigid_linear KIND=linear sbatch slurm/compare_tta.sh
set -uo pipefail
PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
REPO=${REPO:-$PROJ/brain-group-rad}
SIF=${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}
A=${A:-}
B=${B:-}
OUT=${OUT:-}
KIND=${KIND:-eval}
if [ -z "$A" ] || [ -z "$B" ] || [ -z "$OUT" ]; then
    echo "ERROR: set A, B and OUT" >&2
    exit 2
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  kind $KIND  A $A  B $B  out $OUT"
singularity exec -B /autofs -B /scratch "$SIF" bash -c 'cd "$1" && shift && python "$@"' cmp "$REPO" scripts/compare_tta.py --a "$A" --b "$B" --out "$OUT" --kind "$KIND" </dev/null
rc=$?
echo "finished compare_tta exit code $rc"
exit $rc
