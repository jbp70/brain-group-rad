#!/bin/bash
#SBATCH --job-name=bgr-eval-synth
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Post-training synthetic-contrast evaluation of the real-only and the mixed run in ONE job
# (eval_synth.py; design .agents/tasks/synth-contrast/design.md section 8): both models restored
# read-only from <run>/ckpt/last, sanity gate on the real validation sets (exit 5 if the restore
# does not reproduce the logged validation), then the 11 presets (or PRESETS) on the same
# synthetic sets. Nothing is written into a run directory. --mem=128G: the level-1 training
# process reached ~94 GB MaxRSS (64G jobs were killed during validation). Expected ~3.5-4.5 h.
# Needs the synthfine export (export-add-labels files + manifest phase).
# Env: RUNS (required, space-separated run dirs; first = real-only, second = mixed), LEVEL (1),
#   OUT (default $PROJ/results/eval_synth/<job>), PRESETS (comma list, empty = all),
#   DATA (empty = each run's saved args.data).
# Output: $OUT/eval_synth.json, $OUT/eval_synth.csv, $OUT/sstat.txt (sstat MaxRSS at the end;
#   record the post-job `sacct -j <job> --format=MaxRSS` in the eval report).
# BGR_DRY_RUN=1: print the MAIN command and exit 0 before any mkdir or singularity call.
# DO NOT submit without Jay's approval; at most two active jobs (squeue -u jn85). Account qtim only.
# Before sbatch, the log folder must exist (SLURM does not create it):
#   mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
#   R=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results
#   RUNS="<real-only config B L1 run> $R/r1_cfgB_synth05" sbatch --job-name=evalsyn --time=08:00:00 --mem=128G slurm/eval_synth.sh
set -euo pipefail

PROJ="${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}"
REPO="${REPO:-$PROJ/brain-group-rad}"
SIF="${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RUNS="${RUNS:-}"
LEVEL="${LEVEL:-1}"
OUT="${OUT:-$PROJ/results/eval_synth/$JOB_ID}"
PRESETS="${PRESETS:-}"
DATA="${DATA:-}"
BGR_DRY_RUN="${BGR_DRY_RUN:-}"

die() { echo "ERROR: $*" >&2; exit 2; }
[ -n "$RUNS" ] || die "set RUNS (1 or 2 space-separated run directories)"
read -r -a RUN_ARR <<< "$RUNS"
[ "${#RUN_ARR[@]}" -ge 1 ] && [ "${#RUN_ARR[@]}" -le 2 ] || die "RUNS takes 1 or 2 run directories, got ${#RUN_ARR[@]}"
case "$LEVEL" in 1|2|4) ;; *) die "LEVEL must be 1, 2 or 4, got '$LEVEL'";; esac
PRESET_RE='^[A-Za-z0-9_,]*$'
[[ "$PRESETS" =~ $PRESET_RE ]] || die "PRESETS must be a comma list of preset names, got '$PRESETS'"
[ -n "$OUT" ] || die "OUT must not be empty"

MAIN=(eval_synth.py --runs "${RUN_ARR[@]}" --level "$LEVEL" --out "$OUT")
if [ -n "$PRESETS" ]; then MAIN+=(--presets "$PRESETS"); fi
if [ -n "$DATA" ]; then MAIN+=(--data "$DATA"); fi

if [ "$BGR_DRY_RUN" = "1" ]; then
    echo "MAIN: python ${MAIN[*]}"
    exit 0
fi

mkdir -p "$OUT"
export TMPDIR="$SCRATCH_ROOT/$JOB_ID"
mkdir -p "$TMPDIR"
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_TMPDIR="$TMPDIR"
cd "$REPO"
echo "host $(hostname) | job $JOB_ID | repo rev $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '?') | $(date -Is)"
echo "runs ${RUN_ARR[*]} | level $LEVEL | out $OUT | presets ${PRESETS:-all} | data ${DATA:-saved args.data}"
rc=0
singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space "$SIF" python "${MAIN[@]}" </dev/null || rc=$?
if [ -n "${SLURM_JOB_ID:-}" ] && command -v sstat >/dev/null 2>&1; then
    sstat -j "$SLURM_JOB_ID.batch" --format=MaxRSS >> "$OUT/sstat.txt" 2>&1 || true
fi
echo "finished eval_synth rc=$rc $(date -Is) -> $OUT"
exit "$rc"
