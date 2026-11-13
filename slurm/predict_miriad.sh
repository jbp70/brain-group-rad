#!/bin/bash
#SBATCH --job-name=bgr-pred-miriad
#SBATCH -A qtim
#SBATCH -p rtx6000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Patient-level groupwise RIGID or AFFINE registration on processed MIRIAD with a trained cascade
# checkpoint: one `predict.py run --branches $TASK` per patient (all scans of that patient = one
# group, ~5-12 scans). Evaluate afterwards with slurm/eval_miriad.sh (CPU, basic partition).
# DO NOT submit without Jay's approval (cluster-access rules).
#
# Input: $PROCESSED/<scan>_T1w/{image,labels_synthseg}.nii.gz (preprocessing/config_miriad.yaml:
# SynthStrip + N4 + brain-centroid translation only, so the scans are NOT pre-aligned).
#
# Steps:
#   0  python -m evaluation.groups write-groups -> $OUT/groups/<patient>.txt (+ groups.csv)
#   1  check-streaming float32, --branches $TASK, first $N_CHECK scans of the first patient with at
#      least $N_CHECK scans: streamed == unchunked. GATE: a failure stops the job.
#   2  per patient: predict.py run -> $OUT/<patient>/fields, then python -m evaluation.groups
#      fields-to-matrices: the dense linear fields are fitted exactly by 4x4 matrices (check
#      <= 0.01 vox in matrices_check.json) and deleted, so fields/ keeps matrices.npz, atlas.nii.gz,
#      atlas_seg.nii.gz, predict.json (KEEP_DENSE=1 keeps the per-scan fwd/inv/moved NIfTIs).
#      A patient with predict.json (and matrices.npz) is skipped (resubmit the same command to
#      resume). A failed patient is logged in $OUT/failed.txt and the loop goes on; the job exits 1
#      if any patient failed.
#
# Environment overrides (exported by sbatch):
#   CKPT        checkpoint prefix or directory (required, e.g. <run>/ckpt/last)
#   TASK        rigid | affine (required)
#   NAME        method name / output folder (default ours_$TASK)
#   PROCESSED   processed MIRIAD (default $PROJ/data/processed/miriad)
#   OUT         output root (default $PROJ/results/predict_miriad/$NAME)
#   PATIENTS    space-separated patient ids to run (default: all), e.g. PATIENTS=miriad_188_AD_M
#   LABEL_NAME  labels_<name>.nii.gz (default synthseg)
#   N_CHECK     scans of the streaming gate (default 4)
#   CHECK_CHUNKS  chunk flags of the gate (default '--chunk-l4 3 --chunk-l2 3 --chunk-l1 3')
#   GRID_REF    training-grid reference (default $PROJ/data/target_grid/target_grid.json)
#   LEVELS      cascade levels (default empty: all levels of the checkpoint)
#   EXTRA       more flags for `predict.py run`, e.g. "--precision float32"
#   SCRATCH_ROOT  parent of TMPDIR (default /scratch)
#
#   CKPT=$PROJ/results/<L1 run>/ckpt/last TASK=rigid sbatch slurm/predict_miriad.sh
#   CKPT=... TASK=affine PATIENTS=miriad_188_AD_M sbatch --time=01:00:00 slurm/predict_miriad.sh   # smoke
set -uo pipefail
PROJ=${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}
REPO=${REPO:-$PROJ/brain-group-rad}
SIF=${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}
CKPT=${CKPT:?set CKPT (checkpoint prefix or directory)}
TASK=${TASK:?set TASK (rigid or affine)}
case "$TASK" in rigid|affine) ;; *) echo "TASK must be rigid or affine, got $TASK" >&2; exit 2 ;; esac
NAME=${NAME:-ours_$TASK}
PROCESSED=${PROCESSED:-$PROJ/data/processed/miriad}
OUT=${OUT:-$PROJ/results/predict_miriad/$NAME}
PATIENTS=${PATIENTS:-}
LABEL_NAME=${LABEL_NAME:-synthseg}
N_CHECK=${N_CHECK:-4}
CHECK_CHUNKS=${CHECK_CHUNKS:-"--chunk-l4 3 --chunk-l2 3 --chunk-l1 3"}
GRID_REF=${GRID_REF:-$PROJ/data/target_grid/target_grid.json}
LEVELS=${LEVELS:-}
EXTRA=${EXTRA:-}
KEEP_DENSE=${KEEP_DENSE:-0}
LEVEL_FLAGS=""
if [ -n "$LEVELS" ]; then LEVEL_FLAGS="--levels $LEVELS"; fi
export TMPDIR=${SCRATCH_ROOT:-/scratch}/${SLURM_JOB_ID:-manual}
mkdir -p "$TMPDIR" "$OUT"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  repo $(git -C "$REPO" log --oneline -1)"
echo "checkpoint $CKPT  task $TASK  name $NAME  out $OUT  levels ${LEVELS:-all}"
RUN=(singularity exec --nv -B /autofs -B /scratch "$SIF" bash -c 'cd "$1" && shift && python "$@"' predict "$REPO")

echo "=== 0 groups ==="
"${RUN[@]}" -m evaluation.groups write-groups --processed-dir "$PROCESSED" --out-dir "$OUT/groups" </dev/null || exit $?
if [ -n "$PATIENTS" ]; then
    LIST="$PATIENTS"
else
    LIST=$(tail -n +2 "$OUT/groups/groups.csv" | awk -F, '$3 == 1 {print $1}')
fi
N_GROUPS=$(echo "$LIST" | wc -w)
echo "$N_GROUPS patients"

echo "=== 1 check-streaming float32 ($TASK, gate) ==="
GATE_GROUP=""
for g in $LIST; do
    if [ "$(grep -c . "$OUT/groups/$g.txt")" -ge "$N_CHECK" ]; then GATE_GROUP=$g; break; fi
done
if [ -z "$GATE_GROUP" ]; then echo "no patient with >= $N_CHECK scans for the gate" >&2; exit 2; fi
head -n "$N_CHECK" "$OUT/groups/$GATE_GROUP.txt" > "$OUT/groups/_gate.txt"
# $CHECK_CHUNKS, $LEVEL_FLAGS and $EXTRA are expanded unquoted on purpose (word splitting -> flags).
# shellcheck disable=SC2086
"${RUN[@]}" predict.py check-streaming --checkpoint "$CKPT" --processed-dir "$PROCESSED" --subjects-file "$OUT/groups/_gate.txt" --label-name "$LABEL_NAME" --grid-reference "$GRID_REF" --precision float32 --n-subjects "$N_CHECK" $CHECK_CHUNKS $LEVEL_FLAGS --branches "$TASK" --out "$OUT/check_streaming_f32.json" </dev/null || exit $?

echo "=== 2 predict.py run per patient ($TASK) ==="
rm -f "$OUT/failed.txt"
n_done=0; n_skip=0; n_fail=0
for g in $LIST; do
    if [ ! -f "$OUT/groups/$g.txt" ]; then
        echo "$g: no subjects file (unknown patient or fewer than 2 scans)" | tee -a "$OUT/failed.txt"
        n_fail=$((n_fail + 1)); continue
    fi
    F="$OUT/$g/fields"
    if [ -f "$F/predict.json" ] && { [ "$KEEP_DENSE" = "1" ] || [ -f "$F/matrices.npz" ]; }; then
        n_skip=$((n_skip + 1)); continue
    fi
    rc=0
    if [ ! -f "$F/predict.json" ]; then
        echo "--- $g ($(grep -c . "$OUT/groups/$g.txt") scans) $(date +%T)"
        # shellcheck disable=SC2086
        "${RUN[@]}" predict.py run --checkpoint "$CKPT" --processed-dir "$PROCESSED" --subjects-file "$OUT/groups/$g.txt" --label-name "$LABEL_NAME" --grid-reference "$GRID_REF" --branches "$TASK" --out "$F" --overwrite $LEVEL_FLAGS $EXTRA </dev/null
        rc=$?
    fi
    if [ $rc -eq 0 ] && [ "$KEEP_DENSE" != "1" ]; then
        # Linear outputs: keep only the 4x4 matrices (exact fit of the dense fields, checked).
        "${RUN[@]}" -m evaluation.groups fields-to-matrices --fields-dir "$F" --subjects-file "$OUT/groups/$g.txt" --delete-dense </dev/null
        rc=$?
    fi
    if [ $rc -eq 0 ]; then
        n_done=$((n_done + 1))
    else
        echo "$g: predict.py rc $rc" | tee -a "$OUT/failed.txt"
        n_fail=$((n_fail + 1))
    fi
done
echo "patients: $n_done predicted, $n_skip already done, $n_fail failed (of $N_GROUPS)"
echo "Evaluation (basic partition; ask Jay before submitting):"
echo "  METHOD=$NAME PRED=$OUT sbatch slurm/eval_miriad.sh"
if [ $n_fail -gt 0 ]; then exit 1; fi
echo "finished predict_miriad exit code 0"
exit 0
