#!/bin/bash
#SBATCH --job-name=bgr-synth-preview
#SBATCH -A qtim
#SBATCH -p basic
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Synthetic-contrast preview PNGs for Jay (scripts/synth_preview.py; design
# .agents/tasks/synth-contrast/design.md 7.2, plan deviation D-d): levels 4, 2 and 1, CPU only
# (basic partition, container without GPU access; compute never runs on mlsc-login).
# Needs the synthfine export (export-add-labels files + manifest phase) to exist first.
# Output: $PROJ/scratch/synth_preview/<job>/l<k>.png (rows: real + N_GROUPS synthetic groups,
# columns: N cases, middle axial slice).
# Env: N (8), SEED (0), N_GROUPS (2), SPLIT (train), LABEL_SET (synthfine).
# BGR_DRY_RUN=1: print the commands and exit 0 before any mkdir or singularity call.
# DO NOT submit without Jay's approval; at most two active jobs (squeue -u jn85). Account qtim only.
# Before sbatch, the log folder must exist (SLURM does not create it):
#   mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
#   sbatch slurm/synth_preview.sh
set -euo pipefail

PROJ="${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}"
REPO="${REPO:-$PROJ/brain-group-rad}"
DATA="${DATA:-$PROJ/data/npy_v2}"
SIF="${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}"
JOB_ID="${SLURM_JOB_ID:-manual}"
N="${N:-8}"
SEED="${SEED:-0}"
N_GROUPS="${N_GROUPS:-2}"
SPLIT="${SPLIT:-train}"
LABEL_SET="${LABEL_SET:-synthfine}"
BGR_DRY_RUN="${BGR_DRY_RUN:-}"

die() { echo "ERROR: $*" >&2; exit 2; }
for v in N SEED N_GROUPS; do
    case "${!v}" in ''|*[!0-9]*) die "$v must be a non-negative integer, got '${!v}'";; esac
done
NAME_RE='^[A-Za-z0-9_-]+$'
[[ "$SPLIT" =~ $NAME_RE ]] || die "SPLIT must match [A-Za-z0-9_-]+, got '$SPLIT'"
[[ "$LABEL_SET" =~ $NAME_RE ]] || die "LABEL_SET must match [A-Za-z0-9_-]+, got '$LABEL_SET'"
OUT="$PROJ/scratch/synth_preview/$JOB_ID"

cmd() {
    echo singularity exec -B /autofs -B /cluster -B /space "$SIF" python scripts/synth_preview.py \
        --data "$DATA" --level "$1" --n "$N" --seed "$SEED" --groups "$N_GROUPS" --split "$SPLIT" \
        --label-set "$LABEL_SET" --out "$OUT/l$1.png"
}

if [ "$BGR_DRY_RUN" = "1" ]; then
    for lvl in 4 2 1; do echo "PREVIEW: $(cmd "$lvl")"; done
    exit 0
fi

mkdir -p "$OUT"
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_MPLBACKEND=Agg
cd "$REPO"
echo "host $(hostname) | job $JOB_ID | repo rev $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '?') | $(date -Is)"
rc=0
for lvl in 4 2 1; do
    echo "=== preview level $lvl $(date -Is)"
    singularity exec -B /autofs -B /cluster -B /space "$SIF" python scripts/synth_preview.py \
        --data "$DATA" --level "$lvl" --n "$N" --seed "$SEED" --groups "$N_GROUPS" --split "$SPLIT" \
        --label-set "$LABEL_SET" --out "$OUT/l$lvl.png" </dev/null || rc=1
done
echo "finished rc=$rc $(date -Is) -> $OUT"
exit "$rc"
