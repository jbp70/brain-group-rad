#!/bin/bash
#SBATCH --job-name=bgr-sharp
#SBATCH -A qtim
#SBATCH -p basic
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Atlas sharpness comparison on Mindboggle (scripts/atlas_sharpness.py, CPU only, inside tf10_nv.sif).
# Default methods: our two 101-subject atlases (retrain sweep step 3000; m9 step 5000 float32),
# MultiMorph and BrainMorph tps_0. Output: $OUT/{sharpness.json, atlases.png, spread.png}.
# DO NOT submit without Jay's approval (cluster-access rules).
#   sbatch slurm/atlas_sharpness.sh            (OUT, METHODS, SEGS env overrides)
set -uo pipefail
PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
REPO=${REPO:-$PROJ/brain-group-rad}
SIF=${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}
PR=$PROJ/results/predict_mindboggle
BL=$PROJ/baselines
OUT=${OUT:-$PROJ/results/atlas_sharpness/mindboggle}
ARGS=(--out "$OUT"
      --seg "$PR/r1sw_lr1e-4_s3000_deform/fields/atlas_seg.nii.gz"
      --seg "$BL/multimorph/mindboggle_results/atlas_seg.nii.gz"
      --method "ours_r1sw_s3000=$PR/r1sw_lr1e-4_s3000_deform/fields/atlas.nii.gz,$PR/r1sw_lr1e-4_s3000_deform/fields/*_T1w_moved.nii.gz"
      --method "ours_m9_s5000_f32=$PR/m9_best_deform_5000_deform_f32/fields/atlas.nii.gz,$PR/m9_best_deform_5000_deform_f32/fields/*_T1w_moved.nii.gz"
      --method "multimorph=$BL/multimorph/mindboggle_results/atlas.nii.gz,$BL/multimorph/mindboggle_results/*_T1w_moved.nii.gz"
      --method "brainmorph_tps0=$BL/brainmorph/mindboggle_results/tps_0/atlas.nii.gz,$BL/brainmorph/mindboggle_results/tps_0/*_T1w_moved.nii.gz")
mkdir -p "$OUT"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  repo $(git -C "$REPO" log --oneline -1)"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
singularity exec -B /autofs "$SIF" bash -c 'cd "$1" && shift && python scripts/atlas_sharpness.py "$@"' \
    sharp "$REPO" "${ARGS[@]}" </dev/null
rc=$?
echo "exit status $rc"
exit $rc
