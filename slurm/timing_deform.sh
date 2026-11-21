#!/bin/bash
#SBATCH --job-name=bgr-time-dfm
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Registration time of the Mindboggle-101 group (N = 101, paper Table 2), all GPU methods on the
# SAME GPU in one job: ours deform + re-centering 2 (TTA none and flip8), MultiMorph, BrainMorph
# (TPS lambda 0). Each method registers the whole group REPS times in a row (rep 0 cold, later reps
# warm); images are read before the clock starts and no NIfTI is written (scripts/timing/time_*.py).
# Ours' re-centering keeps the original fields in $TMPDIR (.npy, node-local), as predict.py does.
# MultiMorph falls back to one CPU pass if CUDA runs out of memory. ANTs SyN is not re-run (CPU,
# ~62 h on 24 CPUs; recorded time used).
# DO NOT submit without Jay's approval (cluster-access rules).
#
# Environment overrides:
#   REPS   runs per method (default 2)
#   STEPS  subset of: ours ours_tta multimorph bm_tps (default all)
#   CKPT   default $PROJ/results/r1_v3_gB/ckpt/last    PROCESSED  default $PROJ/data/processed/mindboggle
#   OUT    default $PROJ/results/timing/deform_<job id>
#   FIELDS_ONLY=1  time only the generation of the full-resolution fwd / inv fields (ours: + re-centering
#          rounds on the fields; no image / label warping, no atlas) for every method
#   MM_CPU_FALLBACK  1 (default) = one CPU pass of MultiMorph if CUDA runs out of memory; 0 = none
set -uo pipefail
PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
REPO=${REPO:-$PROJ/brain-group-rad}
CKPT=${CKPT:-$PROJ/results/r1_v3_gB/ckpt/last}
PROCESSED=${PROCESSED:-$PROJ/data/processed/mindboggle}
GRID_REF=${GRID_REF:-$PROJ/data/target_grid/target_grid.json}
REPS=${REPS:-2}
STEPS=${STEPS:-"ours ours_tta multimorph bm_tps"}
FIELDS_ONLY=${FIELDS_ONLY:-}
MM_CPU_FALLBACK=${MM_CPU_FALLBACK:-1}
FO=""; [ "$FIELDS_ONLY" = 1 ] && FO=--fields-only
MMFB=""; [ "$MM_CPU_FALLBACK" = 1 ] && MMFB=--cpu-fallback
OUT=${OUT:-$PROJ/results/timing/deform_${SLURM_JOB_ID:-manual}}
TF_SIF=/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif
PT_SIF=/autofs/space/crater_001/users/jn85/Singularity_Files/pt_nv.sif
BM=$PROJ/baselines/brainmorph
MM=$PROJ/baselines/multimorph
export TMPDIR=/scratch/${SLURM_JOB_ID:-manual}
mkdir -p "$TMPDIR" "$OUT"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  repo $(git -C "$REPO" log --oneline -1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# the evaluated group: every subject folder with image + DKT labels (must be 101)
SUBJ=$OUT/mindboggle101.txt
for d in "$PROCESSED"/*/; do
    [ -f "$d/image.nii.gz" ] && [ -f "$d/labels_dkt31aseg.nii.gz" ] && basename "$d"
done > "$SUBJ"
N=$(grep -c . "$SUBJ")
echo "$N subjects in $SUBJ"
[ "$N" = 101 ] || { echo "expected 101 subjects"; exit 2; }

TF=(singularity exec --nv -B /autofs -B /scratch "$TF_SIF" bash -c 'cd "$1" && shift && python -u "$@"' t "$REPO")
PT=(singularity exec --nv -B /autofs -B /scratch "$PT_SIF" bash -c 'cd "$1" && shift && python -u "$@"' t "$REPO")
fails=0
for s in $STEPS; do
    echo "################ $s  $(date)"
    case $s in
        ours|ours_tta)
            tta=none; [ "$s" = ours_tta ] && tta=flip8
            "${TF[@]}" scripts/timing/time_ours.py --checkpoint "$CKPT" --processed-dir "$PROCESSED" \
                --label-name dkt31aseg --grid-reference "$GRID_REF" --branches deform --tta "$tta" \
                --recenter 2 --subjects-file "$SUBJ" --reps "$REPS" --label "$s" $FO --out "$OUT/$s.json" </dev/null ;;
        multimorph)
            "${PT[@]}" scripts/timing/time_multimorph.py --multimorph-dir "$MM" --csv "$MM/mindboggle_data.csv" \
                --model "$MM/trained_models/model_cvpr.pt" --reps "$REPS" $MMFB $FO \
                --out "$OUT/multimorph.json" </dev/null ;;
        bm_tps)
            "${PT[@]}" scripts/timing/time_brainmorph.py --keymorph-dir "$BM/keymorph" \
                --weights "$BM/weights/foundation-numkey512-numlevels6.pth.tar" --processed-dir "$PROCESSED" \
                --label-name dkt31aseg --align tps_0 --subjects-file "$SUBJ" --reps "$REPS" $FO \
                --out "$OUT/bm_tps.json" </dev/null ;;
        *) echo "unknown step $s"; false ;;
    esac
    rc=$?
    echo "################ $s finished rc $rc  $(date)"
    [ "$rc" = 0 ] || fails=$((fails + 1))
done
echo "timing finished: $fails failed -> $OUT"
exit $fails
