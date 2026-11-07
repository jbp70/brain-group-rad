#!/bin/bash
#SBATCH --job-name=bgr-eval-miriad
#SBATCH -A qtim
#SBATCH -p basic
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Patient-level evaluation on processed MIRIAD (CPU; numpy/scipy only, inside tf10_nv.sif): one
# `python -m evaluation run` per patient (all pairs within the patient), then
# `python -m evaluation.groups aggregate` (mean per patient, then over patients; also AD / HC).
# Works for any method that writes per-scan fwd/inv displacement fields on the processed grid
# (our model via slurm/predict_miriad.sh; baselines once converted to the same layout), and for
# the identity reference (IDENTITY=1).
# DO NOT submit without Jay's approval (cluster-access rules).
#
# Metrics kept by the aggregate: template / pair LNCC (and NCC), template / pair Dice (all,
# cortical, subcortical; SynthSeg labels present in every scan of the patient), centrality
# (|mean_i fwd_i| over the atlas mask, mm) and atlas sharpness (atlas_intensity_std = mean
# voxelwise std of the warped scans). The per-patient summary.json also holds the folding /
# Jacobian numbers, which are not used for linear maps.
#
# Environment overrides (exported by sbatch):
#   METHOD      method name = output folder (required), e.g. ours_rigid, identity, ants_affine
#   PRED        prediction root with <patient>/fields (default $PROJ/results/predict_miriad/$METHOD)
#   FWD, INV    field templates ({group}, {subject}, {subject_dir}); default our layout
#               $PRED/{group}/fields/{subject_dir}_fwd.nii.gz / _inv.nii.gz
#   ATLAS, ATLAS_SEG  atlas templates ({group}); default $PRED/{group}/fields/atlas.nii.gz /
#               atlas_seg.nii.gz; set to "none" for the mean image / majority vote
#   CONV        convention flags (default "--components xyz --signs +++ --units mm")
#   IDENTITY=1  no-registration reference (zero fields; mean image / majority vote atlas)
#   GROUPS_DIR  subjects files <patient>.txt (default $PRED/groups; identity: $OUT/groups, written here)
#   PATIENTS    space-separated patient ids (default: every <patient>.txt in GROUPS_DIR)
#   PROCESSED   processed MIRIAD (default $PROJ/data/processed/miriad)
#   OUT         default $PROJ/results/eval_miriad/$METHOD (-> <patient>/summary.json, _aggregate/)
#   PAIRS       all (default) | none | number of seeded random pairs per patient
#   OVERWRITE=1 re-evaluate patients that already have summary.json (default: skip them)
#   EXTRA       more flags for `evaluation run`, e.g. "--hd95"
#
#   METHOD=ours_rigid sbatch slurm/eval_miriad.sh
#   METHOD=identity IDENTITY=1 sbatch slurm/eval_miriad.sh
set -uo pipefail
PROJ=${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}
REPO=${REPO:-$PROJ/brain-group-rad}
SIF=${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}
METHOD=${METHOD:?set METHOD}
PROCESSED=${PROCESSED:-$PROJ/data/processed/miriad}
PRED=${PRED:-$PROJ/results/predict_miriad/$METHOD}
OUT=${OUT:-$PROJ/results/eval_miriad/$METHOD}
# Defaults with literal {group} / {subject_dir} set in if blocks: a "}" inside ${VAR:-...} would
# end the expansion early.
if [ -z "${FWD:-}" ]; then FWD="$PRED/{group}/fields/{subject_dir}_fwd.nii.gz"; fi
if [ -z "${INV:-}" ]; then INV="$PRED/{group}/fields/{subject_dir}_inv.nii.gz"; fi
if [ -z "${ATLAS:-}" ]; then ATLAS="$PRED/{group}/fields/atlas.nii.gz"; fi
if [ -z "${ATLAS_SEG:-}" ]; then ATLAS_SEG="$PRED/{group}/fields/atlas_seg.nii.gz"; fi
CONV=${CONV:-"--components xyz --signs +++ --units mm"}
IDENTITY=${IDENTITY:-0}
PAIRS=${PAIRS:-all}
OVERWRITE=${OVERWRITE:-0}
EXTRA=${EXTRA:-}
PATIENTS=${PATIENTS:-}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}
mkdir -p "$OUT"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  repo $(git -C "$REPO" log --oneline -1)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN=(singularity exec -B /autofs -B /scratch "$SIF" bash -c 'cd "$1" && shift && python "$@"' eval "$REPO")

if [ "$IDENTITY" = "1" ]; then
    GROUPS_DIR=${GROUPS_DIR:-$OUT/groups}
    "${RUN[@]}" -m evaluation.groups write-groups --processed-dir "$PROCESSED" --out-dir "$GROUPS_DIR" </dev/null || exit $?
else
    GROUPS_DIR=${GROUPS_DIR:-$PRED/groups}
fi
if [ -n "$PATIENTS" ]; then
    LIST="$PATIENTS"
else
    LIST=$(cd "$GROUPS_DIR" && ls -- *.txt 2>/dev/null | grep -v '^_' | sed 's/\.txt$//')
fi
N_GROUPS=$(echo "$LIST" | wc -w)
if [ "$N_GROUPS" -eq 0 ]; then echo "no subjects files in $GROUPS_DIR" >&2; exit 2; fi
echo "method $METHOD: $N_GROUPS patients, groups $GROUPS_DIR, identity $IDENTITY"

rm -f "$OUT/failed.txt"
n_done=0; n_skip=0; n_fail=0
for g in $LIST; do
    if [ "$OVERWRITE" != "1" ] && [ -f "$OUT/$g/summary.json" ]; then n_skip=$((n_skip + 1)); continue; fi
    ARGS=(run --processed-dir "$PROCESSED" --label-name synthseg --subjects-file "$GROUPS_DIR/$g.txt"
          --method "$METHOD" --out "$OUT/$g" --pairs "$PAIRS" --workers "$WORKERS")
    if [ "$IDENTITY" = "1" ]; then
        ARGS+=(--identity)
    else
        ARGS+=(--fwd "${FWD//\{group\}/$g}" --inv "${INV//\{group\}/$g}")
        [ "$ATLAS" != "none" ] && ARGS+=(--atlas "${ATLAS//\{group\}/$g}")
        [ "$ATLAS_SEG" != "none" ] && ARGS+=(--atlas-seg "${ATLAS_SEG//\{group\}/$g}")
        # shellcheck disable=SC2206
        ARGS+=($CONV)
    fi
    # shellcheck disable=SC2206
    ARGS+=($EXTRA)
    echo "--- $g $(date +%T)"
    "${RUN[@]}" -m evaluation "${ARGS[@]}" </dev/null > "$OUT/$g.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        n_done=$((n_done + 1))
    else
        echo "$g: evaluation rc $rc (log $OUT/$g.log)" | tee -a "$OUT/failed.txt"
        n_fail=$((n_fail + 1))
    fi
done
echo "patients: $n_done evaluated, $n_skip already done, $n_fail failed (of $N_GROUPS)"

echo "=== aggregate ==="
"${RUN[@]}" -m evaluation.groups aggregate --root "$OUT" --method "$METHOD" --out "$OUT/_aggregate" --expected "$N_GROUPS" </dev/null
rc=$?
if [ $n_fail -gt 0 ]; then exit 1; fi
exit $rc
