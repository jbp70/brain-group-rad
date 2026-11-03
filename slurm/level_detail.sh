#!/bin/bash
#SBATCH --job-name=bgr-lvl-detail
#SBATCH -A qtim
#SBATCH -p basic
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Atlas sharpness (scripts/atlas_sharpness.py) + per-image cortical detail (scripts/moved_detail.py)
# for several prediction runs under results/predict_mindboggle plus MultiMorph, CPU only, tf10_nv.sif.
# RUNS: space-separated run names (default: the level ablation of the retrained step-3000 checkpoint).
# Output: $OUT/sharpness/ and $OUT/per_image/.
# DO NOT submit without Jay's approval (cluster-access rules).
set -uo pipefail
PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
REPO=${REPO:-$PROJ/brain-group-rad}
SIF=${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}
PR=$PROJ/results/predict_mindboggle
MM=$PROJ/baselines/multimorph/mindboggle_results
B=r1sw_lr1e-4_s3000_deform
RUNS=${RUNS:-"${B}_L4 ${B}_L42 ${B} ${B}_L21 ${B}_L1"}
OUT=${OUT:-$PROJ/results/atlas_sharpness/mindboggle_levels}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}
SH=(--out "$OUT/sharpness" --seg "$PR/$B/fields/atlas_seg.nii.gz" --seg "$MM/atlas_seg.nii.gz")
PI=(--processed-dir "$PROJ/data/processed/mindboggle" --out "$OUT/per_image" --workers "$WORKERS")
for r in $RUNS; do
    F=$PR/$r/fields
    [ -f "$F/atlas.nii.gz" ] || { echo "missing $F/atlas.nii.gz"; exit 2; }
    SH+=(--method "$r=$F/atlas.nii.gz,$F/*_T1w_moved.nii.gz")
    PI+=(--method "$r=$F/{subject_dir}_moved.nii.gz,$F/{subject_dir}_fwd.nii.gz")
done
SH+=(--method "multimorph=$MM/atlas.nii.gz,$MM/*_T1w_moved.nii.gz")
PI+=(--method "multimorph=$MM/{subject_dir}_moved.nii.gz,$MM/{subject_dir}_fwd.nii.gz")
mkdir -p "$OUT"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  repo $(git -C "$REPO" log --oneline -1)  runs: $RUNS"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
run() { singularity exec -B /autofs "$SIF" bash -c 'cd "$1" && shift && python "$@"' x "$REPO" "$@" </dev/null; }
run scripts/atlas_sharpness.py "${SH[@]}"; rc1=$?
run scripts/moved_detail.py "${PI[@]}"; rc2=$?
echo "exit status sharpness $rc1 per_image $rc2"
[ "$rc1" = 0 ] && [ "$rc2" = 0 ]
