#!/bin/bash
#SBATCH --job-name=bgr-smgroup
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# SynthMorph keypoints (FreeSurfer's trained rigid / affine Detector models) + BrainMorph-style
# groupwise mean-keypoint registration, written in the MultiMorph / BrainMorph file layout for
# `python -m evaluation` (convention --components xyz --signs +++ --units mm).
# Runs baselines/synthmorph/run_synthmorph_groupwise.py with FreeSurfer's fspython on the host
# (no container): first a synthetic self-test, then the real run, which aborts if the extracted
# keypoints/weights do not reproduce the model's own pairwise matrix (pair_check.json).
#
# Environment overrides (exported by sbatch):
#   PROCESSED      processed dataset: <subject>_T1w/image.nii.gz on ONE common grid (required)
#   OUT            output folder (required); results in $OUT/rigid and $OUT/affine
#   ALIGNS         default "rigid affine"
#   LABEL_NAME     labels_<name>.nii.gz for moved_seg / atlas_seg (default dkt31aseg; optional)
#   MAX_SUBJECTS, SUBJECTS_FILE, EXTRA (e.g. "--unweighted" or "--extent 192")
#   FREESURFER_HOME  default /usr/local/freesurfer/dev
#
#   PROCESSED=$PROJ/data/processed/<dataset> OUT=$PROJ/baselines/synthmorph/<dataset>_results \
#     sbatch slurm/synthmorph_groupwise.sh
#   Smoke: add MAX_SUBJECTS=4 and --time=00:45:00
set -uo pipefail
PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
REPO=${REPO:-$PROJ/brain-group-rad}
export FREESURFER_HOME=${FREESURFER_HOME:-/usr/local/freesurfer/dev}
FSPY=$FREESURFER_HOME/bin/fspython
PROCESSED=${PROCESSED:?set PROCESSED (processed dataset folder)}
OUT=${OUT:?set OUT (output folder)}
ALIGNS=${ALIGNS:-rigid affine}
ARGS=(--processed-dir "$PROCESSED" --out "$OUT" --label-name "${LABEL_NAME:-dkt31aseg}")
# shellcheck disable=SC2206
ARGS+=(--aligns $ALIGNS)
[ -n "${MAX_SUBJECTS:-}" ] && ARGS+=(--max-subjects "$MAX_SUBJECTS")
[ -n "${SUBJECTS_FILE:-}" ] && ARGS+=(--subjects-file "$SUBJECTS_FILE")
# shellcheck disable=SC2206
ARGS+=(${EXTRA:-})
[ -x "$FSPY" ] || { echo "missing $FSPY (FreeSurfer)"; exit 2; }
for m in synthmorph.rigid.1.h5 synthmorph.affine.2.h5; do
    [ -f "$FREESURFER_HOME/models/$m" ] || echo "WARNING: missing $FREESURFER_HOME/models/$m"
done
mkdir -p "$OUT"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  repo $(git -C "$REPO" log --oneline -1)  FS $FREESURFER_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export NEURITE_BACKEND=tensorflow VXM_BACKEND=tensorflow TF_CPP_MIN_LOG_LEVEL=2
cd "$REPO" || exit 2
echo "== self-test"
"$FSPY" baselines/synthmorph/run_synthmorph_groupwise.py --self-test </dev/null || { echo "self-test FAILED"; exit 3; }
echo "== run: ${ARGS[*]}"
"$FSPY" baselines/synthmorph/run_synthmorph_groupwise.py "${ARGS[@]}" </dev/null
rc=$?
echo "exit status $rc"
exit $rc
