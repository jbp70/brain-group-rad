#!/bin/bash
#SBATCH --job-name=bgr-tta-mb
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Mindboggle TTA comparison of one checkpoint in one GPU job (TTA design 9.1-9.4, milestone 10):
# arm A = --tta none, arm B = --tta flip8 (all levels unless TTA_LEVELS), same subjects, same group.
# DO NOT submit without Jay's approval (cluster-access rules).
#
# Order (PARTS, default 'predict linear compare'):
#   predict  slurm/predict_mindboggle.sh twice with bash inside this allocation (its #SBATCH lines
#            are ignored; both runs share TMPDIR, the store frees its entries after each run):
#            ${NAME}_tta_none (TTA=none, checks 1a-1c + run + template-only evaluation), then
#            ${NAME}_tta_flip8 (TTA=flip8, SKIP_CHECKS=1: the checks are TTA-free and ran in arm A).
#            Stops at the first failing arm.
#   linear   predict.py linear-eval for task rigid and affine x arm none and flip8 (same --seed, so
#            both arms see the identical rendered group) -> results/linear_eval_mindboggle/
#            ${NAME}_<task>_tta_<arm>; a failing run is counted, the others still run.
#   compare  scripts/compare_tta.py (CPU, inside the container) on the two template-only
#            evaluations -> results/tta_compare/${NAME}_template, and per task on the linear-eval
#            outputs (--kind linear) -> results/tta_compare/${NAME}_<task>_linear.
# Then it PRINTS (never submits) the two all-pairs evaluation commands (basic partition, one per
# arm) and the final all-pairs compare command (slurm/compare_tta.sh). Submit them only after
# Jay's approval and within the two-active-jobs rule (count Jay's own jobs in squeue -u jn85):
# the two eval_mindboggle.sh jobs first, compare_tta.sh after both finished.
# Exit status = number of failed parts (0 = all ok); 2 also for invalid settings (nothing run).
#
# Environment:
#   CKPT        checkpoint prefix or directory (required)
#   NAME        method name / output prefix (required; no commas or colons)
#   PROCESSED   processed subjects (default $PROJ/data/processed/mindboggle)
#   GRID_REF    training-grid reference (default $PROJ/data/target_grid/target_grid.json)
#   PARTS       subset of 'predict linear compare' (default all three, always run in this order)
#   SEED        linear-eval seed (default 0)
#   TTA_LEVELS  --tta-levels of the flip8 arm (default empty: all levels)
#   DRY_RUN     1 = print every command, run nothing (default empty)
#   REPO, SIF, SCRATCH_ROOT  as in slurm/predict_mindboggle.sh
#
# Estimated wall time, level-1 checkpoint, N 101: arm A ~35 min, arm B ~65 min, linear-eval
# 4 runs <= ~70 min -> ~3 h (10 h limit for safety).
#
#   NAME=r1_cfgB_sm10_s2500_bnF CKPT=$PROJ/results/r1_cfgB_sm10_s2500_bnfreeze_true/ckpt/last \
#     sbatch slurm/tta_mindboggle.sh
#   DRY_RUN=1 NAME=x CKPT=/tmp/x bash slurm/tta_mindboggle.sh
set -uo pipefail
PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
REPO=${REPO:-$PROJ/brain-group-rad}
SIF=${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}
CKPT=${CKPT:-}
NAME=${NAME:-}
PROCESSED=${PROCESSED:-$PROJ/data/processed/mindboggle}
GRID_REF=${GRID_REF:-$PROJ/data/target_grid/target_grid.json}
PARTS=${PARTS:-predict linear compare}
SEED=${SEED:-0}
TTA_LEVELS=${TTA_LEVELS:-}
DRY_RUN=${DRY_RUN:-}
if [ -z "$CKPT" ] || [ -z "$NAME" ]; then
    echo "ERROR: set CKPT and NAME" >&2
    exit 2
fi
case "$NAME" in
    *,*|*:*) echo "ERROR: NAME must not contain commas or colons: $NAME" >&2; exit 2;;
esac
for p in $PARTS; do
    case "$p" in
        predict|linear|compare) ;;
        *) echo "ERROR: unknown part '$p' in PARTS (predict linear compare)" >&2; exit 2;;
    esac
done
has_part() { case " $PARTS " in *" $1 "*) return 0;; *) return 1;; esac; }
# run <command...>: execute, or print it with DRY_RUN=1
run() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY: $*"
        return 0
    fi
    "$@"
}
export TMPDIR=${SCRATCH_ROOT:-/scratch}/${SLURM_JOB_ID:-manual}
if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$TMPDIR"
    echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  repo $(git -C "$REPO" log --oneline -1)"
fi
echo "checkpoint $CKPT  name $NAME  parts '$PARTS'  seed $SEED  tta levels ${TTA_LEVELS:-all}  dry run ${DRY_RUN:-0}"
PRED=$PROJ/results/predict_mindboggle
LIN=$PROJ/results/linear_eval_mindboggle
CMP=$PROJ/results/tta_compare
failed=0

if has_part predict; then
    echo "=== predict: arm A (tta none) ==="
    run env NAME="${NAME}_tta_none" OUT="$PRED/${NAME}_tta_none" TTA=none TTA_LEVELS= SKIP_CHECKS= CKPT="$CKPT" PROCESSED="$PROCESSED" GRID_REF="$GRID_REF" REPO="$REPO" SIF="$SIF" bash "$REPO/slurm/predict_mindboggle.sh"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "=== predict: arm B (tta flip8) ==="
        run env NAME="${NAME}_tta_flip8" OUT="$PRED/${NAME}_tta_flip8" TTA=flip8 TTA_LEVELS="$TTA_LEVELS" SKIP_CHECKS=1 CKPT="$CKPT" PROCESSED="$PROCESSED" GRID_REF="$GRID_REF" REPO="$REPO" SIF="$SIF" bash "$REPO/slurm/predict_mindboggle.sh"
        rc=$?
    fi
    if [ "$rc" -ne 0 ]; then
        echo "WARNING: predict part failed (rc $rc)"
        failed=$((failed + 1))
    fi
fi

if has_part linear; then
    nbad=0
    for task in rigid affine; do
        for arm in none flip8; do
            echo "=== linear-eval $task, tta $arm ==="
            LEVEL_ARGS=()
            if [ "$arm" = "flip8" ] && [ -n "$TTA_LEVELS" ]; then LEVEL_ARGS=(--tta-levels "$TTA_LEVELS"); fi
            run singularity exec --nv -B /autofs -B /scratch "$SIF" bash -c 'cd "$1" && shift && python "$@"' predict "$REPO" predict.py linear-eval --checkpoint "$CKPT" --processed-dir "$PROCESSED" --grid-reference "$GRID_REF" --task "$task" --seed "$SEED" --tta "$arm" ${LEVEL_ARGS[@]+"${LEVEL_ARGS[@]}"} --out "$LIN/${NAME}_${task}_tta_${arm}" </dev/null
            rc=$?
            if [ "$rc" -ne 0 ]; then
                echo "WARNING: linear-eval $task tta $arm rc $rc"
                nbad=$((nbad + 1))
            fi
        done
    done
    if [ "$nbad" -ne 0 ]; then failed=$((failed + 1)); fi
fi

if has_part compare; then
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    nbad=0
    echo "=== compare: template-only evaluations ==="
    run singularity exec -B /autofs -B /scratch "$SIF" bash -c 'cd "$1" && shift && python "$@"' cmp "$REPO" scripts/compare_tta.py --a "$PRED/${NAME}_tta_none/eval_template" --b "$PRED/${NAME}_tta_flip8/eval_template" --out "$CMP/${NAME}_template" --kind eval --label-a none --label-b flip8 </dev/null || nbad=$((nbad + 1))
    for task in rigid affine; do
        echo "=== compare: linear-eval $task ==="
        run singularity exec -B /autofs -B /scratch "$SIF" bash -c 'cd "$1" && shift && python "$@"' cmp "$REPO" scripts/compare_tta.py --a "$LIN/${NAME}_${task}_tta_none" --b "$LIN/${NAME}_${task}_tta_flip8" --out "$CMP/${NAME}_${task}_linear" --kind linear --label-a none --label-b flip8 </dev/null || nbad=$((nbad + 1))
    done
    if [ "$nbad" -ne 0 ]; then
        echo "WARNING: $nbad compare call(s) failed"
        failed=$((failed + 1))
    fi
fi

echo "All-pairs evaluation of both arms (basic partition; ask Jay before submitting, two-active-jobs rule):"
for ARM in none flip8; do
    F="$PRED/${NAME}_tta_$ARM/fields"
    echo "  METHOD=${NAME}_tta_$ARM FWD='$F/{subject_dir}_fwd.nii.gz' INV='$F/{subject_dir}_inv.nii.gz' ATLAS=$F/atlas.nii.gz ATLAS_SEG=$F/atlas_seg.nii.gz CONV=\"--components xyz --signs +++ --units mm\" sbatch slurm/eval_mindboggle.sh"
done
echo "Then, after both finished, the all-pairs comparison:"
echo "  A=$PROJ/results/eval_mindboggle/${NAME}_tta_none B=$PROJ/results/eval_mindboggle/${NAME}_tta_flip8 OUT=$CMP/${NAME}_allpairs sbatch slurm/compare_tta.sh"
echo "finished tta_mindboggle: $failed failed part(s)"
exit "$failed"
