#!/bin/bash
#SBATCH --job-name=bgr-sm-miriad
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# SynthMorph baseline on MIRIAD, patient level: one groupwise RIGID registration (FreeSurfer's
# synthmorph.rigid.1.h5 keypoints) and one groupwise AFFINE registration (synthmorph.affine.2.h5)
# per patient (all scans of a patient = one group), BrainMorph-style mean-keypoint loop with
# SynthMorph's power-weighted symmetric fit (5 iterations), as for Mindboggle
# (slurm/synthmorph_groupwise.sh). Runs baselines/synthmorph/run_synthmorph_groupwise.py
# --groups-dir with FreeSurfer's fspython on the host (no container): first the synthetic
# self-test, then the real run. Each patient's pair check (keypoint refit == the model's own
# pairwise matrix) must pass, else that patient fails. Evaluate with slurm/eval_miriad.sh.
# DO NOT submit without Jay's approval (cluster-access rules).
#
# Outputs ($OUT = $PROJ/results/predict_miriad):
#   $OUT/synthmorph_rigid/<patient>/fields/{matrices.npz, atlas.nii.gz, atlas_seg.nii.gz}
#       (same for synthmorph_affine; no dense fields, eval_miriad.sh uses --matrices;
#       EXTRA="--save-fields dense" adds the per-scan fwd/inv/moved NIfTIs)
#   $OUT/synthmorph_<align>/groups/<patient>.txt              (read by eval_miriad.sh)
#   $OUT/_synthmorph_keypoints/<patient>/{<align>_keypoints.npz,<align>_pair_check.json,run_info.json}
#       (run_info.json written last; a patient that has it is skipped, so resubmitting resumes)
#
# Environment overrides (exported by sbatch):
#   PATIENTS    space-separated patient ids (default: all patients with >= 2 scans)
#   ALIGNS      default "rigid affine"
#   PROCESSED   default $PROJ/data/processed/miriad
#   OUT         default $PROJ/results/predict_miriad
#   METHOD_PREFIX  output folder prefix (default synthmorph; e.g. synthmorph_uw with EXTRA=--unweighted)
#   EXTRA       more flags for run_synthmorph_groupwise.py (e.g. "--unweighted", "--extent 192")
#   FREESURFER_HOME  default /usr/local/freesurfer/dev
#
# Smoke (one patient):  PATIENTS=miriad_188_AD_M sbatch --time=00:45:00 slurm/synthmorph_miriad.sh
# Full:                 sbatch slurm/synthmorph_miriad.sh
# Then:                 METHOD=synthmorph_rigid sbatch slurm/eval_miriad.sh
#                       METHOD=synthmorph_affine sbatch slurm/eval_miriad.sh
set -uo pipefail
PROJ=${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}
REPO=${REPO:-$PROJ/brain-group-rad}
export FREESURFER_HOME=${FREESURFER_HOME:-/usr/local/freesurfer/dev}
FSPY=$FREESURFER_HOME/bin/fspython
PROCESSED=${PROCESSED:-$PROJ/data/processed/miriad}
OUT=${OUT:-$PROJ/results/predict_miriad}
ALIGNS=${ALIGNS:-"rigid affine"}
METHOD_PREFIX=${METHOD_PREFIX:-synthmorph}
PATIENTS=${PATIENTS:-}
EXTRA=${EXTRA:-}
GROUPS_DIR="$OUT/_${METHOD_PREFIX}_keypoints/groups"
[ -x "$FSPY" ] || { echo "missing $FSPY (FreeSurfer)"; exit 2; }
for m in synthmorph.rigid.1.h5 synthmorph.affine.2.h5; do
    [ -f "$FREESURFER_HOME/models/$m" ] || { echo "missing $FREESURFER_HOME/models/$m"; exit 2; }
done
mkdir -p "$OUT"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  repo $(git -C "$REPO" log --oneline -1)  FS $FREESURFER_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export NEURITE_BACKEND=tensorflow VXM_BACKEND=tensorflow TF_CPP_MIN_LOG_LEVEL=2
cd "$REPO" || exit 2
echo "== groups"
"$FSPY" -u -m evaluation.groups write-groups --processed-dir "$PROCESSED" --out-dir "$GROUPS_DIR" </dev/null || exit $?
echo "== self-test"
"$FSPY" -u baselines/synthmorph/run_synthmorph_groupwise.py --self-test </dev/null || { echo "self-test FAILED"; exit 3; }
ARGS=(--processed-dir "$PROCESSED" --out "$OUT" --label-name synthseg --num-iters 5 --groups-dir "$GROUPS_DIR"
      --method-prefix "$METHOD_PREFIX")
# shellcheck disable=SC2206
ARGS+=(--aligns $ALIGNS)
# shellcheck disable=SC2206
[ -n "$PATIENTS" ] && ARGS+=(--groups $PATIENTS)
# shellcheck disable=SC2206
ARGS+=($EXTRA)
echo "== run: ${ARGS[*]}"
"$FSPY" -u baselines/synthmorph/run_synthmorph_groupwise.py "${ARGS[@]}" </dev/null
rc=$?
echo "exit status $rc"
for al in $ALIGNS; do
    echo "Evaluation (basic partition; ask Jay before submitting): METHOD=${METHOD_PREFIX}_$al sbatch slurm/eval_miriad.sh"
done
exit $rc
