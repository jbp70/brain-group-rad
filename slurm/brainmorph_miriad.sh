#!/bin/bash
#SBATCH --job-name=bgr-bm-miriad
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
# BrainMorph baseline on MIRIAD, patient level: one groupwise RIGID and one groupwise AFFINE
# registration per patient (all scans of a patient = one group), BrainMorph variant L, 512
# keypoints, 5 unweighted groupwise iterations (keymorph groupwise_register). The keypoints of a
# patient are extracted once and shared by both alignments. Runs
# baselines/brainmorph/run_brainmorph.py --groups-dir in pt_nv.sif (PyTorch); first the synthetic
# self-test on CPU (~1 min), then the real run. Evaluate with slurm/eval_miriad.sh.
# DO NOT submit without Jay's approval (cluster-access rules).
#
# Uses the keymorph clone + weights of the Mindboggle run (one-time setup in
# slurm/brainmorph_mindboggle.sh).
#
# Outputs ($OUT = $PROJ/results/predict_miriad):
#   $OUT/brainmorph_rigid/<patient>/fields/{<scan>_T1w_fwd,_inv,_moved,_moved_seg}.nii.gz,
#       atlas.nii.gz, atlas_seg.nii.gz, matrices.npz        (same for brainmorph_affine)
#   $OUT/brainmorph_<align>/groups/<patient>.txt              (read by eval_miriad.sh)
#   $OUT/_brainmorph_keypoints/<patient>/{keypoints.npz,run_info.json}  (run_info.json written
#       last; a patient that has it is skipped, so resubmitting the same command resumes;
#       run_info.json records gpu_peak_allocated_gb)
#
# Environment overrides (exported by sbatch):
#   PATIENTS    space-separated patient ids (default: all patients with >= 2 scans)
#   ALIGNS      default "rigid affine"
#   PROCESSED   default $PROJ/data/processed/miriad
#   OUT         default $PROJ/results/predict_miriad
#   KEYMORPH_DIR, WEIGHTS, SIF, REPO, EXTRA (more flags for run_brainmorph.py, e.g. --weighted
#               with a different METHOD_PREFIX)
#   METHOD_PREFIX  output folder prefix (default brainmorph)
#
# Smoke (one patient):  PATIENTS=miriad_188_AD_M sbatch --time=00:45:00 slurm/brainmorph_miriad.sh
# Full:                 sbatch slurm/brainmorph_miriad.sh
# Then:                 METHOD=brainmorph_rigid sbatch slurm/eval_miriad.sh
#                       METHOD=brainmorph_affine sbatch slurm/eval_miriad.sh
set -uo pipefail
PROJ=${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}
REPO=${REPO:-$PROJ/brain-group-rad}
BM=$PROJ/baselines/brainmorph
SIF=${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/pt_nv.sif}
PROCESSED=${PROCESSED:-$PROJ/data/processed/miriad}
KEYMORPH_DIR=${KEYMORPH_DIR:-$BM/keymorph}
WEIGHTS=${WEIGHTS:-$BM/weights/foundation-numkey512-numlevels6.pth.tar}
OUT=${OUT:-$PROJ/results/predict_miriad}
ALIGNS=${ALIGNS:-"rigid affine"}
METHOD_PREFIX=${METHOD_PREFIX:-brainmorph}
PATIENTS=${PATIENTS:-}
EXTRA=${EXTRA:-}
GROUPS_DIR="$OUT/_${METHOD_PREFIX}_keypoints/groups"
for f in "$KEYMORPH_DIR/keymorph/model.py" "$WEIGHTS"; do
    [ -f "$f" ] || { echo "missing $f (see one-time setup in slurm/brainmorph_mindboggle.sh)"; exit 2; }
done
mkdir -p "$OUT"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  repo $(git -C "$REPO" log --oneline -1)"
echo "keymorph $(git -C "$KEYMORPH_DIR" log --oneline -1 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
RUN=(singularity exec --nv -B /autofs -B /scratch "$SIF" bash -c 'cd "$1" && shift && python "$@"' bm "$REPO")

echo "== groups"
"${RUN[@]}" -m evaluation.groups write-groups --processed-dir "$PROCESSED" --out-dir "$GROUPS_DIR" </dev/null || exit $?
echo "== self-test"
"${RUN[@]}" baselines/brainmorph/run_brainmorph.py --self-test --keymorph-dir "$KEYMORPH_DIR" --device cpu </dev/null \
    || { echo "self-test FAILED"; exit 3; }
ARGS=(--keymorph-dir "$KEYMORPH_DIR" --processed-dir "$PROCESSED" --label-name synthseg --out "$OUT"
      --weights "$WEIGHTS" --num-keypoints 512 --variant L --num-iters 5 --groups-dir "$GROUPS_DIR"
      --method-prefix "$METHOD_PREFIX")
# shellcheck disable=SC2206
ARGS+=(--aligns $ALIGNS)
# shellcheck disable=SC2206
[ -n "$PATIENTS" ] && ARGS+=(--groups $PATIENTS)
# shellcheck disable=SC2206
ARGS+=($EXTRA)
echo "== run: ${ARGS[*]}"
"${RUN[@]}" baselines/brainmorph/run_brainmorph.py "${ARGS[@]}" </dev/null
rc=$?
echo "exit status $rc"
for al in $ALIGNS; do
    echo "Evaluation (basic partition; ask Jay before submitting): METHOD=${METHOD_PREFIX}_$al sbatch slurm/eval_miriad.sh"
done
exit $rc
