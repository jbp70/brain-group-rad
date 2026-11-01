#!/bin/bash
#SBATCH --job-name=bgr-predict
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Groupwise prediction on the processed Mindboggle grid with a trained cascade checkpoint
# (predict.py, inside tf10_nv.sif), then a template-only evaluation (predict-script design 9).
# DO NOT submit without Jay's approval (cluster-access rules).
#
# Steps:
#   1a  check-streaming, float32 trainable stage, $BRANCHES: streamed == unchunked on the first
#       $N_CHECK subjects (float32 bounds). GATE: a failure stops the job (|| exit $?).
#   1b  the same with --precision checkpoint (fp16 record, provisional bounds; non-blocking).
#   1c  the step-1a command with --branches affine+deform (linear streaming record; non-blocking).
#   2   predict.py run over all subjects -> $OUT/fields (exit 5 = internal check failed, stops here).
#   3   evaluation check-convention + template-only evaluation (--pairs $EVAL_PAIRS) on the fields.
#   The full pairwise evaluation runs separately on the basic partition; the command is printed
#   at the end (never submitted from here).
#
# Environment overrides (exported by sbatch):
#   CKPT          checkpoint prefix or directory (default $PROJ/results/m9_level1_cascade/ckpt/best_deform)
#   NAME          method name / output folder (default m9_best_deform_5000_deform)
#   BRANCHES      rigid | affine | deform | affine+deform ... (default deform)
#   PROCESSED     processed subjects (default $PROJ/data/processed/mindboggle)
#   OUT           output root (default $PROJ/results/predict_mindboggle/$NAME)
#   N_CHECK       subjects of the streaming checks (default 4 = n_max at level 1)
#   CHECK_CHUNKS  chunk flags of the streaming checks (default '--chunk-l4 3 --chunk-l2 3 --chunk-l1 3':
#                 3 + 1 at every level, so every level streams a full chunk and a remainder)
#   GRID_REF      training-grid reference (default $PROJ/data/target_grid/target_grid.json)
#   EVAL_PAIRS    --pairs of the in-job evaluation (default none: template metrics only)
#   EXTRA         more flags for `predict.py run`, e.g. "--check-internal 4"
#
# OOM fallback of step 1a (not automatic; the gate is never skipped): resubmit with
#   N_CHECK=3 CHECK_CHUNKS='--chunk-l4 2 --chunk-l2 2 --chunk-l1 2'
# (2 + 1 at every level; chunk 3 at N = 3 would be a single chunk).
#
# Exit 4 of step 1a (a float32 bound exceeded): read $OUT/check_streaming_f32.json. Localised
# differences or differences growing with field magnitude point to an engine bug (fix the code);
# uniform differences slightly above the bound point to rounding: resubmit with --tol-* overrides
# chosen from the measured numbers (e.g. in CHECK_CHUNKS) and record them in the report; never
# drop the gate.
#
#   sbatch slurm/predict_mindboggle.sh
#   NAME=m9_best_deform_5000_affine_deform BRANCHES=affine+deform sbatch slurm/predict_mindboggle.sh
set -uo pipefail
PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
REPO=${REPO:-$PROJ/brain-group-rad}
SIF=${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}
CKPT=${CKPT:-$PROJ/results/m9_level1_cascade/ckpt/best_deform}
NAME=${NAME:-m9_best_deform_5000_deform}
BRANCHES=${BRANCHES:-deform}
PROCESSED=${PROCESSED:-$PROJ/data/processed/mindboggle}
OUT=${OUT:-$PROJ/results/predict_mindboggle/$NAME}
N_CHECK=${N_CHECK:-4}
CHECK_CHUNKS=${CHECK_CHUNKS:-"--chunk-l4 3 --chunk-l2 3 --chunk-l1 3"}
GRID_REF=${GRID_REF:-$PROJ/data/target_grid/target_grid.json}
EVAL_PAIRS=${EVAL_PAIRS:-none}
EXTRA=${EXTRA:-}
export TMPDIR=/scratch/${SLURM_JOB_ID:-manual}
mkdir -p "$TMPDIR" "$OUT"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  repo $(git -C "$REPO" log --oneline -1)"
echo "checkpoint $CKPT  branches $BRANCHES  out $OUT"
FIELDS="$OUT/fields"
FWD="$FIELDS/{subject_dir}_fwd.nii.gz"
INV="$FIELDS/{subject_dir}_inv.nii.gz"
MOVED="$FIELDS/{subject_dir}_moved.nii.gz"
CONV="--components xyz --signs +++ --units mm"

# $CHECK_CHUNKS (and $EXTRA, $CONV) are expanded unquoted on purpose: word splitting turns them into flags.
echo "=== 1a check-streaming float32 ($BRANCHES, gate) ==="
# shellcheck disable=SC2086
singularity exec --nv -B /autofs -B /scratch "$SIF" bash -c 'cd "$1" && shift && python "$@"' predict "$REPO" predict.py check-streaming --checkpoint "$CKPT" --processed-dir "$PROCESSED" --grid-reference "$GRID_REF" --precision float32 --n-subjects "$N_CHECK" $CHECK_CHUNKS --branches "$BRANCHES" --out "$OUT/check_streaming_f32.json" </dev/null || exit $?
echo "=== 1b check-streaming checkpoint precision ($BRANCHES, record) ==="
# shellcheck disable=SC2086
singularity exec --nv -B /autofs -B /scratch "$SIF" bash -c 'cd "$1" && shift && python "$@"' predict "$REPO" predict.py check-streaming --checkpoint "$CKPT" --processed-dir "$PROCESSED" --grid-reference "$GRID_REF" --precision checkpoint --n-subjects "$N_CHECK" $CHECK_CHUNKS --branches "$BRANCHES" --out "$OUT/check_streaming_fp16.json" </dev/null || echo "WARNING: fp16 streaming check rc $? (non-blocking), see $OUT/check_streaming_fp16.json"
echo "=== 1c check-streaming float32 (affine+deform, record) ==="
# shellcheck disable=SC2086
singularity exec --nv -B /autofs -B /scratch "$SIF" bash -c 'cd "$1" && shift && python "$@"' predict "$REPO" predict.py check-streaming --checkpoint "$CKPT" --processed-dir "$PROCESSED" --grid-reference "$GRID_REF" --precision float32 --n-subjects "$N_CHECK" $CHECK_CHUNKS --branches affine+deform --out "$OUT/check_streaming_affine_deform_f32.json" </dev/null || echo "WARNING: affine+deform streaming check rc $? (non-blocking), see $OUT/check_streaming_affine_deform_f32.json"
echo "=== 2 predict.py run ($BRANCHES) ==="
# shellcheck disable=SC2086
singularity exec --nv -B /autofs -B /scratch "$SIF" bash -c 'cd "$1" && shift && python "$@"' predict "$REPO" predict.py run --checkpoint "$CKPT" --processed-dir "$PROCESSED" --grid-reference "$GRID_REF" --branches "$BRANCHES" --out "$FIELDS" $EXTRA </dev/null || exit $?
echo "=== 3 evaluation check-convention + template metrics ==="
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
singularity exec -B /autofs -B /scratch "$SIF" bash -c 'cd "$1" && shift && python "$@"' predict "$REPO" -m evaluation check-convention --processed-dir "$PROCESSED" --fwd "$FWD" --inv "$INV" --moved "$MOVED" --out "$OUT/convention.json" </dev/null || exit $?
# shellcheck disable=SC2086
singularity exec -B /autofs -B /scratch "$SIF" bash -c 'cd "$1" && shift && python "$@"' predict "$REPO" -m evaluation run --processed-dir "$PROCESSED" --method "$NAME" --fwd "$FWD" --inv "$INV" --atlas "$FIELDS/atlas.nii.gz" --atlas-seg "$FIELDS/atlas_seg.nii.gz" $CONV --pairs "$EVAL_PAIRS" --workers 3 --out "$OUT/eval_template" </dev/null || exit $?
echo "Full pairwise evaluation (basic partition; ask Jay before submitting):"
echo "  METHOD=$NAME FWD='$FWD' INV='$INV' ATLAS=$FIELDS/atlas.nii.gz ATLAS_SEG=$FIELDS/atlas_seg.nii.gz CONV=\"$CONV\" sbatch slurm/eval_mindboggle.sh"
echo "finished predict_mindboggle exit code 0"
exit 0
