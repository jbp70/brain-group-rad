#!/bin/bash
#SBATCH --job-name=bgr-fig2
#SBATCH -A qtim
#SBATCH -p rtx6000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Paper figure 2 (off-MNI qualitative examples): groupwise rigid, affine and affine+deform
# registration of ONE group = all subjects of $PROCESSED (default: the 39 LUMIR validation cases with
# a known random affine at the training ranges, slurm/perturb_processed.sh), template-only
# evaluation of every row (identity included), then the slice extraction for the figure
# (scripts/figure_offmni.py extract). Plot locally afterwards (scripts/figure_offmni.py plot).
# DO NOT submit without Jay's approval (cluster-access rules).
#
# Steps:
#   1  check-streaming float32 gate per branch (first $N_CHECK subjects; a failure stops the job)
#   2  predict.py run per branch -> $OUT/<row>/fields (row = rigid | affine | affine_deform);
#      --tta $TTA on every branch, --recenter $RECENTER on affine+deform only (linear outputs
#      cannot be re-centred by predict.py). A row whose predict.json reports status ok is skipped.
#   3  evaluation run --pairs $EVAL_PAIRS -> $OUT/eval/<row> (identity = no registration)
#   4  scripts/figure_offmni.py extract -> $OUT/fig2_data.npz (+ fig2_data.json)
#
# Environment overrides (exported by sbatch):
#   CKPT        default $PROJ/results/r1_v3_gB/ckpt/last (final model, v3 config B)
#   PROCESSED   default $PROJ/data/processed/lumir_ants_val_affpert45_rot30
#   OUT         default $PROJ/results/fig2_offmni/v3_gB_lumirval_affpert45_rot30
#   LABEL_NAME  default synthseg
#   TTA         default flip8 (final inference setting)
#   RECENTER    default 2 (affine+deform only); empty = off
#   EVAL_PAIRS  default none (template metrics only); all = also pair metrics (1482 pairs at N 39)
#   ROWS        default "rigid affine affine+deform"
#   N_CHECK     default 4;  GRID_REF default $PROJ/data/target_grid/target_grid.json
#   EXTRACT_ARGS  more flags for the extract step (e.g. "--n-candidates 10")
#   CHECK_EXTRA   more flags for the check-streaming gates, e.g. "--tol-field-max-mm 0.002" when the
#               measured float32 differences are rounding-sized (tiny mean, max just over the bound);
#               record the reason. Never drop the gate.
#
#   sbatch slurm/fig2_offmni_predict.sh
set -uo pipefail
PROJ=${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}
REPO=${REPO:-$PROJ/brain-group-rad}
SIF=${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}
CKPT=${CKPT:-$PROJ/results/r1_v3_gB/ckpt/last}
PROCESSED=${PROCESSED:-$PROJ/data/processed/lumir_ants_val_affpert45_rot30}
OUT=${OUT:-$PROJ/results/fig2_offmni/v3_gB_lumirval_affpert45_rot30}
LABEL_NAME=${LABEL_NAME:-synthseg}
TTA=${TTA:-flip8}
RECENTER=${RECENTER-2}
EVAL_PAIRS=${EVAL_PAIRS:-none}
ROWS=${ROWS:-"rigid affine affine+deform"}
N_CHECK=${N_CHECK:-4}
GRID_REF=${GRID_REF:-$PROJ/data/target_grid/target_grid.json}
EXTRACT_ARGS=${EXTRACT_ARGS:-}
CHECK_EXTRA=${CHECK_EXTRA:-}
CHECK_CHUNKS="--chunk-l4 3 --chunk-l2 3 --chunk-l1 3"
export TMPDIR=${SCRATCH_ROOT:-/scratch}/${SLURM_JOB_ID:-manual}
mkdir -p "$TMPDIR" "$OUT/eval"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  repo $(git -C "$REPO" log --oneline -1)"
echo "checkpoint $CKPT  processed $PROCESSED  out $OUT  tta $TTA  recenter ${RECENTER:-off}  rows $ROWS"
if [ ! -f "$PROCESSED/perturbation_summary.csv" ]; then
    echo "WARNING: $PROCESSED/perturbation_summary.csv missing (not made by perturb_processed.py?)"
fi
GPU=(singularity exec --nv -B /autofs -B /scratch "$SIF" bash -c 'cd "$1" && shift && python "$@"' fig2 "$REPO")
CPU=(singularity exec -B /autofs -B /scratch -B /usr/local/freesurfer "$SIF" bash -c 'cd "$1" && shift && python "$@"' fig2 "$REPO")
row_name() { echo "${1//+/_}"; }

for B in $ROWS; do
    R=$(row_name "$B")
    F="$OUT/$R/fields"
    if [ -f "$F/predict.json" ] && grep -q '"status": "ok"' "$F/predict.json"; then
        echo "=== $R: predict.json ok, skipped"; continue
    fi
    echo "=== 1 check-streaming float32 ($B, gate) $(date +%T) ==="
    # shellcheck disable=SC2086
    "${GPU[@]}" predict.py check-streaming --checkpoint "$CKPT" --processed-dir "$PROCESSED" --label-name "$LABEL_NAME" --grid-reference "$GRID_REF" --precision float32 --n-subjects "$N_CHECK" $CHECK_CHUNKS $CHECK_EXTRA --branches "$B" --out "$OUT/$R/check_streaming_f32.json" </dev/null || exit $?
    FLAGS="--tta $TTA"
    if [ "$B" = "affine+deform" ] && [ -n "$RECENTER" ]; then FLAGS="$FLAGS --recenter $RECENTER"; fi
    echo "=== 2 predict.py run ($B, $FLAGS) $(date +%T) ==="
    # shellcheck disable=SC2086
    "${GPU[@]}" predict.py run --checkpoint "$CKPT" --processed-dir "$PROCESSED" --label-name "$LABEL_NAME" --grid-reference "$GRID_REF" --branches "$B" --out "$F" --overwrite $FLAGS </dev/null || exit $?
done

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKERS=${SLURM_CPUS_PER_TASK:-4}
echo "=== 3 evaluation (pairs $EVAL_PAIRS) $(date +%T) ==="
if [ ! -f "$OUT/eval/identity/summary.json" ]; then
    "${CPU[@]}" -m evaluation run --processed-dir "$PROCESSED" --label-name "$LABEL_NAME" --method identity --identity --pairs "$EVAL_PAIRS" --workers "$WORKERS" --out "$OUT/eval/identity" </dev/null || exit $?
fi
for B in $ROWS; do
    R=$(row_name "$B")
    F="$OUT/$R/fields"
    [ -f "$OUT/eval/$R/summary.json" ] && continue
    "${CPU[@]}" -m evaluation run --processed-dir "$PROCESSED" --label-name "$LABEL_NAME" --method "$R" --fwd "$F/{subject_dir}_fwd.nii.gz" --inv "$F/{subject_dir}_inv.nii.gz" --atlas "$F/atlas.nii.gz" --atlas-seg "$F/atlas_seg.nii.gz" --components xyz --signs +++ --units mm --pairs "$EVAL_PAIRS" --workers "$WORKERS" --out "$OUT/eval/$R" </dev/null || exit $?
done

echo "=== 4 figure data extract $(date +%T) ==="
# shellcheck disable=SC2086
"${CPU[@]}" scripts/figure_offmni.py extract --processed-dir "$PROCESSED" --label-name "$LABEL_NAME" --root "$OUT" --out "$OUT/fig2_data.npz" $EXTRACT_ARGS </dev/null || exit $?
echo "copy $OUT/fig2_data.npz to the local machine and run: python scripts/figure_offmni.py plot --data fig2_data.npz --out fig2.png"
echo "finished fig2_offmni_predict exit code 0"
exit 0
