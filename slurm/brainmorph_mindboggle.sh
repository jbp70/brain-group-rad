#!/bin/bash
#SBATCH --job-name=bgr-brainmorph
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# BrainMorph groupwise baseline (deformable only: TPS lambda 0; variant L, 512 keypoints) on the
# processed Mindboggle grid, written in the MultiMorph file layout for `python -m evaluation`.
# The TPS is fitted directly to the raw keypoints (its affine part is included), so it does not
# need the rigid/affine runs; add them with ALIGNS="rigid affine tps_0" if ever needed.
# Runs baselines/brainmorph/run_brainmorph.py in pt_nv.sif (PyTorch); first a ~1 min synthetic
# self-test on CPU, then the real run. See the docstring of run_brainmorph.py for conventions.
#
# One-time setup (login node; code + weights only):
#   BM=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/baselines/brainmorph
#   SIF=/autofs/space/crater_001/users/jn85/Singularity_Files/pt_nv.sif
#   mkdir -p $BM/weights && git clone https://github.com/alanqrwang/keymorph.git $BM/keymorph
#   singularity exec -B /autofs $SIF gdown 16015bMeH7aG9jMknmtwpp2zxrreNq6_X \
#       -O $BM/weights/foundation-numkey512-numlevels6.pth.tar
#   (id from brainmorph scripts/register.py URL_DICT; manual download: README "Model Weights")
#
# Environment overrides (exported by sbatch):
#   OUT            default $PROJ/baselines/brainmorph/mindboggle_results (one subfolder per alignment)
#   ALIGNS         default "tps_0" (also accepts rigid, affine, tps_<lambda>)
#   MAX_SUBJECTS   first N subjects only (smoke test; default all)
#   SUBJECTS_FILE  optional list of subject folder names
#   KEYMORPH_DIR, WEIGHTS, PROCESSED, SIF, REPO, EXTRA (more flags for run_brainmorph.py)
#
# Smoke (4 subjects, ~15 min):  MAX_SUBJECTS=4 OUT=$PROJ/baselines/brainmorph/smoke4 \
#                                 sbatch --time=01:00:00 slurm/brainmorph_mindboggle.sh
# Full (101 subjects, est. 1-2 h, mostly writing .nii.gz):  sbatch slurm/brainmorph_mindboggle.sh
#   -> $OUT/tps_0/{<subject_dir>_fwd,_inv,_moved,_moved_seg}.nii.gz, atlas.nii.gz, atlas_seg.nii.gz
set -uo pipefail
PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
REPO=${REPO:-$PROJ/brain-group-rad}
BM=$PROJ/baselines/brainmorph
SIF=${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/pt_nv.sif}
PROCESSED=${PROCESSED:-$PROJ/data/processed/mindboggle}
KEYMORPH_DIR=${KEYMORPH_DIR:-$BM/keymorph}
WEIGHTS=${WEIGHTS:-$BM/weights/foundation-numkey512-numlevels6.pth.tar}
OUT=${OUT:-$BM/mindboggle_results}
ALIGNS=${ALIGNS:-tps_0}
ARGS=(--keymorph-dir "$KEYMORPH_DIR" --processed-dir "$PROCESSED" --out "$OUT" --weights "$WEIGHTS"
      --num-keypoints 512 --variant L)
# shellcheck disable=SC2206
ARGS+=(--aligns $ALIGNS)
[ -n "${MAX_SUBJECTS:-}" ] && ARGS+=(--max-subjects "$MAX_SUBJECTS")
[ -n "${SUBJECTS_FILE:-}" ] && ARGS+=(--subjects-file "$SUBJECTS_FILE")
# shellcheck disable=SC2206
ARGS+=(${EXTRA:-})
for f in "$KEYMORPH_DIR/keymorph/model.py" "$WEIGHTS"; do
    [ -f "$f" ] || { echo "missing $f (see one-time setup in this script)"; exit 2; }
done
mkdir -p "$OUT"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  repo $(git -C "$REPO" log --oneline -1)"
echo "keymorph $(git -C "$KEYMORPH_DIR" log --oneline -1 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
run() {
    singularity exec --nv -B /autofs -B /scratch "$SIF" bash -c 'cd "$1" && shift && python "$@"' \
        bm "$REPO" baselines/brainmorph/run_brainmorph.py "$@" </dev/null
}
echo "== self-test"
run --self-test --keymorph-dir "$KEYMORPH_DIR" --device cpu || { echo "self-test FAILED"; exit 3; }
echo "== run: ${ARGS[*]}"
run "${ARGS[@]}"
rc=$?
echo "exit status $rc"
exit $rc
