#!/bin/bash
#SBATCH --job-name=bgr-adni-suvr
#SBATCH -A qtim
#SBATCH -p basic
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# ADNI tau-PET experiment (paper Figure 5): SUVR in the four arms per patient (scripts/adni_tau_suvr.py
# run: ANTs rigid T1 -> central PET mean, then native / forward / backward / central SUVR at
# interpolation orders $ORDERS), then the paired statistics (stats). CPU only. Resumable: patients with
# suvr.csv are skipped. Needs the rigid predictions of slurm/predict_miriad.sh on the processed PETs:
#   CKPT=$PROJ/results/r1_v3_gB_synth05/ckpt/last TASK=rigid NAME=v3_gB_synth05_rigid_tta \
#   PROCESSED=$PROJ/data/processed/adni_tau_pet OUT=$PROJ/results/predict_adni_tau/v3_gB_synth05_rigid_tta \
#   GROUP_REGEX='^(\d{3}_S_\d{4})_(?:bl|fu)_PET$' LABEL_NAME=mask N_CHECK=2 EXTRA='--tta flip8' \
#   sbatch slurm/predict_miriad.sh
# DO NOT submit without Jay's approval (cluster-access rules).
#
#   NAME=v3_gB_synth05_rigid_tta sbatch slurm/adni_tau_suvr.sh
#   NAME=... PATIENTS='168_S_6051' sbatch --time=00:30:00 slurm/adni_tau_suvr.sh      # smoke
set -uo pipefail
PROJ=${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}
REPO=${REPO:-$PROJ/brain-group-rad}
SIF=${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}
NAME=${NAME:?set NAME (prediction name under results/predict_adni_tau)}
ADNI=${ADNI:-$PROJ/data/adni_tau}
PRED=${PRED:-$PROJ/results/predict_adni_tau/$NAME}
OUT=${OUT:-$PROJ/results/adni_tau_suvr/$NAME}
PATIENTS=${PATIENTS:-}
ORDERS=${ORDERS:-1,3}
SUPERSAMPLE=${SUPERSAMPLE:-3}
REF_ERODE=${REF_ERODE:-0}
SAVE_IMAGES=${SAVE_IMAGES:-1}
EXTRA=${EXTRA:-}
export TMPDIR=${SCRATCH_ROOT:-/scratch}/${SLURM_JOB_ID:-manual}
mkdir -p "$TMPDIR" "$OUT"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  repo $(git -C "$REPO" log --oneline -1)"
echo "pred $PRED  out $OUT  orders $ORDERS  supersample $SUPERSAMPLE  ref erode $REF_ERODE"
PY=(singularity exec -B /autofs -B /scratch "$SIF" python)
FLAGS=()
if [ -n "$PATIENTS" ]; then FLAGS+=(--patients "$PATIENTS"); fi
if [ "$SAVE_IMAGES" = "1" ]; then FLAGS+=(--save-images); fi
rc=0
# shellcheck disable=SC2086
"${PY[@]}" "$REPO/scripts/adni_tau_suvr.py" run --inventory "$ADNI/inventory.csv" --static-dir "$ADNI/static" \
    --pet-processed "$PROJ/data/processed/adni_tau_pet" --t1-processed "$PROJ/data/processed/adni_tau_t1" \
    --pred "$PRED" --out "$OUT" --orders "$ORDERS" --supersample "$SUPERSAMPLE" --ref-erode "$REF_ERODE" \
    --threads "${SLURM_CPUS_PER_TASK:-4}" "${FLAGS[@]}" $EXTRA </dev/null || rc=$?
"${PY[@]}" "$REPO/scripts/adni_tau_suvr.py" stats --root "$OUT" </dev/null || rc=$?
echo "finished adni_tau_suvr exit code $rc"
exit $rc
