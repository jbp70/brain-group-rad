#!/bin/bash
#SBATCH --job-name=bgr-ants-miriad
#SBATCH -A qtim
#SBATCH -p basic
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# ANTs baseline on MIRIAD, patient level: one groupwise template construction
# (antsMultivariateTemplateConstruction2.sh, same presets as slurm/ants_groupwise.sh) per patient
# (all scans of a patient = one group) for ONE linear type, TYPE=rigid or TYPE=affine (one job each).
# CPU only (basic): PARALLEL_GROUPS patients at a time, each with 24 / PARALLEL_GROUPS parallel
# single-thread registrations (-c 2 -j, local; never -c 5). Python side in tf10_nv.sif, ANTs 2.4.4
# from pubsw. First a linear-only synthetic self-test (a few min), then the run. Evaluate with
# slurm/eval_miriad.sh.
# DO NOT submit without Jay's approval (cluster-access rules).
#
# Cost (from job 10542601 on Mindboggle: one affine registration ~40 min on one core at 1 mm):
# 708 scans x ITERATIONS 4 = 2832 registrations ~1900 CPU-h for affine -> ~80 h wall on 24 CPUs;
# rigid has the same schedule but fewer parameters (expect somewhat less). The run resumes: a
# patient whose work folder has run_info.json is skipped, so resubmit the same command after a
# time-out (an interrupted patient restarts from scratch). ITERATIONS=2 or 3 halves / quarters it.
#
# Outputs ($OUT = $PROJ/results/predict_miriad):
#   $OUT/ants_<type>/<patient>/fields/{matrices.npz, atlas.nii.gz, atlas_seg.nii.gz} (no dense
#       fields; eval_miriad.sh evaluates with --matrices)
#   $OUT/ants_<type>/groups/<patient>.txt                     (read by eval_miriad.sh)
#   $OUT/_ants_work/<type>/<patient>/  ANTs .mat transforms, ants_template_construction.log,
#       ants_check.json (our moved image vs ANTs' WarpedToTemplate, matrix vs ANTs' dense field),
#       run_info.json (done marker, timing); ANTs' images are deleted (KEEP_WORK=1 keeps them)
#
# Environment overrides (exported by sbatch):
#   TYPE             rigid | affine (required)
#   PATIENTS         space-separated patient ids (default: all patients with >= 2 scans)
#   ITERATIONS       template iterations (default 4, as for Mindboggle)
#   PARALLEL_GROUPS  patients at the same time (default 2 -> 12 registrations each; MIRIAD patients
#                    have ~5-12 scans)
#   ANTS_ARGS        extra antsMultivariateTemplateConstruction2.sh flags (appended last)
#   PROCESSED        default $PROJ/data/processed/miriad
#   OUT              default $PROJ/results/predict_miriad
#   KEEP_WORK=1, SKIP_SELFTEST=1, ANTSPATH, SIF, REPO
#
# Smoke (one patient, 1 iteration):
#   TYPE=affine PATIENTS=miriad_188_AD_M ITERATIONS=1 sbatch --time=03:00:00 slurm/ants_miriad.sh
# Full:   TYPE=rigid sbatch slurm/ants_miriad.sh ;  TYPE=affine sbatch slurm/ants_miriad.sh
# Then:   METHOD=ants_rigid sbatch slurm/eval_miriad.sh ;  METHOD=ants_affine sbatch slurm/eval_miriad.sh
set -uo pipefail
PROJ=${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}
REPO=${REPO:-$PROJ/brain-group-rad}
SIF=${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}
export ANTSPATH=${ANTSPATH:-/autofs/cluster/pubsw/2/pubsw/Linux2-2.3-x86_64/packages/ANTS/2.4.4/bin}
TYPE=${TYPE:?set TYPE (rigid or affine)}
case "$TYPE" in rigid|affine) ;; *) echo "TYPE must be rigid or affine, got $TYPE" >&2; exit 2 ;; esac
PROCESSED=${PROCESSED:-$PROJ/data/processed/miriad}
OUT=${OUT:-$PROJ/results/predict_miriad}
ITERATIONS=${ITERATIONS:-4}
PARALLEL_GROUPS=${PARALLEL_GROUPS:-2}
PATIENTS=${PATIENTS:-}
CPUS=${SLURM_CPUS_PER_TASK:-24}
GROUPS_DIR="$OUT/_ants_work/groups"
JOB_START=$(date +%s)
mkdir -p "$OUT"
echo "start $(date '+%F %T')  host $(hostname)  job ${SLURM_JOB_ID:-none}  cpus $CPUS  repo $(git -C "$REPO" log --oneline -1)"
echo "ANTs $ANTSPATH  type $TYPE  iterations $ITERATIONS  parallel groups $PARALLEL_GROUPS"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=${TMPDIR:-/scratch/${SLURM_JOB_ID:-$$}}
mkdir -p "$TMPDIR"
RUN=(singularity exec -B /autofs -B /scratch "$SIF" bash -c 'cd "$1" && shift && python -u "$@"' ants "$REPO")

echo "== groups"
"${RUN[@]}" -m evaluation.groups write-groups --processed-dir "$PROCESSED" --out-dir "$GROUPS_DIR" </dev/null || exit $?
if [ "${SKIP_SELFTEST:-0}" != "1" ]; then
    echo "== self-test (linear only)"
    "${RUN[@]}" baselines/ants/run_ants_groupwise.py --self-test --self-test-linear-only --ants-path "$ANTSPATH" </dev/null \
        || { echo "self-test FAILED"; exit 3; }
fi
ARGS=(--type "$TYPE" --processed-dir "$PROCESSED" --out "$OUT" --label-name synthseg --iterations "$ITERATIONS"
      --cpus "$CPUS" --ants-path "$ANTSPATH" --groups-dir "$GROUPS_DIR" --parallel-groups "$PARALLEL_GROUPS")
[ -n "${ANTS_ARGS:-}" ] && ARGS+=(--ants-args "$ANTS_ARGS")
# shellcheck disable=SC2206
[ -n "$PATIENTS" ] && ARGS+=(--groups $PATIENTS)
[ "${KEEP_WORK:-0}" = "1" ] && ARGS+=(--keep-work)
echo "== run: ${ARGS[*]}"
"${RUN[@]}" baselines/ants/run_ants_groupwise.py "${ARGS[@]}" </dev/null
rc=$?
el=$(( $(date +%s) - JOB_START ))
printf 'JOB TIMING: end %s, whole job %dh %02dm %02ds wall on %s CPUs\n' \
    "$(date '+%F %T')" $((el / 3600)) $((el % 3600 / 60)) $((el % 60)) "$CPUS"
echo "exit status $rc"
echo "Evaluation (basic partition; ask Jay before submitting): METHOD=ants_$TYPE sbatch slurm/eval_miriad.sh"
exit $rc
