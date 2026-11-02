#!/bin/bash
#SBATCH --job-name=bgr-antsgw
#SBATCH -A qtim
#SBATCH -p basic
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=120:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# ANTs groupwise template construction (antsMultivariateTemplateConstruction2.sh, SyGN) for one
# registration type, converted to the MultiMorph / BrainMorph evaluation format
# (--components xyz --signs +++ --units mm). CPU only: runs on `basic`, 24 registrations (one per CPU) in
# parallel inside this ONE job (-c 2 -j $SLURM_CPUS_PER_TASK; never -c 5, which would submit one
# SLURM job per registration). Python side runs in tf10_nv.sif (numpy/scipy/nibabel); ANTs 2.4.4
# from pubsw through the /autofs bind. First a synthetic self-test (~5 min), then the run.
# Measured (job 10542601, 4 Mindboggle subjects, 1 iteration): one affine registration ~40 min on
# one core. Job 10542618 (4 subjects, 1 iteration, NO_LINEAR=1): one SyN registration (100x100x70x20,
# CC, 1 mm) ~3 h on one core (~6 min per full-res iteration). Est. for 101 subjects x 4 iterations on
# 24 CPUs (5 rounds of 24 per iteration): affine ~11-15 h, SyN-only ~60 h, rigid+affine+SyN longer.
#
# Environment overrides (exported by sbatch):
#   TYPE        rigid | affine | deformable (required)
#   PROCESSED   processed dataset: <subject>_T1w/image.nii.gz on ONE common grid (required)
#   OUT         output folder (required); results in $OUT/$TYPE (ANTs work files in $OUT/$TYPE/work)
#   LABEL_NAME  labels_<name>.nii.gz for moved_seg / atlas_seg (default dkt31aseg; optional)
#   ITERATIONS  template iterations (default 4)
#   ANTS_ARGS   extra antsMultivariateTemplateConstruction2.sh flags, appended last (override presets)
#   NO_LINEAR=1 deformable only: skip ALL rigid / affine alignment (-r 0 -l 0) for pre-aligned inputs;
#               results in $OUT/deformable_nolinear
#   MAX_SUBJECTS, SUBJECTS_FILE, SKIP_TEMPLATE=1 (only redo the conversion), SKIP_SELFTEST=1
#   ANTSPATH    default /autofs/cluster/pubsw/2/pubsw/Linux2-2.3-x86_64/packages/ANTS/2.4.4/bin
#
# Timing: the log ends with "TIMING ..." (ANTs template construction alone = the number for the
# paper, its CPU-hours, our conversion) and "JOB TIMING ..." (whole job incl. self-test);
# also $OUT/<name>/timing.json.
#
# Mindboggle deformable, already affinely aligned -> SyN only (resubmit with SKIP_TEMPLATE=1 if only
# the conversion failed):
#   TYPE=deformable NO_LINEAR=1 PROCESSED=$PROJ/data/processed/mindboggle \
#     OUT=$PROJ/baselines/ants/mindboggle_results sbatch slurm/ants_groupwise.sh
# Rigid / affine on another dataset (est. 11-15 h):
#   TYPE=affine PROCESSED=$PROJ/data/processed/<dataset> OUT=$PROJ/baselines/ants/<dataset>_results \
#     sbatch --time=24:00:00 slurm/ants_groupwise.sh
# Smoke: add MAX_SUBJECTS=4 ITERATIONS=1 and --time=02:00:00
set -uo pipefail
PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
REPO=${REPO:-$PROJ/brain-group-rad}
SIF=${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}
export ANTSPATH=${ANTSPATH:-/autofs/cluster/pubsw/2/pubsw/Linux2-2.3-x86_64/packages/ANTS/2.4.4/bin}
TYPE=${TYPE:?set TYPE (rigid, affine or deformable)}
PROCESSED=${PROCESSED:?set PROCESSED (processed dataset folder)}
OUT=${OUT:?set OUT (output folder)}
CPUS=${SLURM_CPUS_PER_TASK:-24}
ARGS=(--type "$TYPE" --processed-dir "$PROCESSED" --out "$OUT" --label-name "${LABEL_NAME:-dkt31aseg}"
      --iterations "${ITERATIONS:-4}" --cpus "$CPUS" --ants-path "$ANTSPATH")
[ -n "${ANTS_ARGS:-}" ] && ARGS+=(--ants-args "$ANTS_ARGS")
[ -n "${MAX_SUBJECTS:-}" ] && ARGS+=(--max-subjects "$MAX_SUBJECTS")
[ -n "${SUBJECTS_FILE:-}" ] && ARGS+=(--subjects-file "$SUBJECTS_FILE")
[ "${SKIP_TEMPLATE:-0}" = "1" ] && ARGS+=(--skip-template)
[ "${NO_LINEAR:-0}" = "1" ] && ARGS+=(--no-linear)
JOB_START=$(date +%s)
echo "start $(date '+%F %T')"
mkdir -p "$OUT"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  cpus $CPUS  repo $(git -C "$REPO" log --oneline -1)"
echo "ANTs $ANTSPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=${TMPDIR:-/scratch/${SLURM_JOB_ID:-$$}}
run() {
    singularity exec -B /autofs -B /scratch "$SIF" bash -c 'cd "$1" && shift && python -u "$@"' \
        ants "$REPO" baselines/ants/run_ants_groupwise.py "$@" </dev/null
}
if [ "${SKIP_SELFTEST:-0}" != "1" ]; then
    echo "== self-test"
    run --self-test --ants-path "$ANTSPATH" || { echo "self-test FAILED"; exit 3; }
fi
echo "== run: ${ARGS[*]}"
run "${ARGS[@]}"
rc=$?
el=$(( $(date +%s) - JOB_START ))
printf 'JOB TIMING: end %s, whole job %dh %02dm %02ds wall on %s CPUs (incl. self-test)\n' \
    "$(date '+%F %T')" $((el / 3600)) $((el % 3600 / 60)) $((el % 60)) "$CPUS"
echo "exit status $rc"
exit $rc
