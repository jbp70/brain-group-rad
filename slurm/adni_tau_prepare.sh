#!/bin/bash
#SBATCH --job-name=bgr-adni-prep
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
# ADNI tau-PET experiment (paper Figure 5), data preparation (CPU). Data handling = Jay runs it.
# STEPS (comma list, default all four, in this order):
#   unzip      $ZIPS/*.zip -> $RAW (unzip -n: existing files are kept, so rerunning after more zips
#              arrive only adds the new patients)
#   inventory  $RAW -> $ADNI/inventory.csv (+ printed summary of PET grids and follow-up intervals)
#   average    frame alignment (ANTs rigid to the frame mean) + mean -> $ADNI/static/<patient>/pet_{bl,fu}.nii.gz
#   manifests  $MANIFESTS/adni_tau_pet.csv and adni_tau_t1.csv for `python -m preprocessing run`
# Then (Jay): preprocessing/slurm/run_array.sh with CONFIG=preprocessing/config_adni_tau_pet.yaml on
# adni_tau_pet.csv and CONFIG=preprocessing/config_adni_tau_t1.yaml on adni_tau_t1.csv, SynthSeg on the
# T1 rows (preprocessing/slurm/synthseg_batch.sh), then STEPS=pet-labels with this script.
#
#   sbatch slurm/adni_tau_prepare.sh
#   STEPS=pet-labels sbatch --time=00:30:00 slurm/adni_tau_prepare.sh
set -uo pipefail
PROJ=${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}
REPO=${REPO:-$PROJ/brain-group-rad}
SIF=${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}
ZIPS=${ZIPS:-$PROJ/data_adni}
RAW=${RAW:-$PROJ/data_adni/raw}
ADNI=${ADNI:-$PROJ/data/adni_tau}
MANIFESTS=${MANIFESTS:-$PROJ/data/manifests}
PROCESSED=${PROCESSED:-$PROJ/data/processed}
STEPS=${STEPS:-unzip,inventory,average,manifests}
ALIGN=${ALIGN:-ants}
FOLLOWUP=${FOLLOWUP:-first}
THREADS=${SLURM_CPUS_PER_TASK:-4}
export TMPDIR=${SCRATCH_ROOT:-/scratch}/${SLURM_JOB_ID:-manual}
mkdir -p "$TMPDIR" "$ADNI" "$MANIFESTS"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  repo $(git -C "$REPO" log --oneline -1)  steps $STEPS"
PY=(singularity exec -B /autofs -B /scratch "$SIF" python)
has() { case ",$STEPS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

if has unzip; then
    echo "=== unzip $ZIPS/*.zip -> $RAW ==="
    mkdir -p "$RAW"
    for z in "$ZIPS"/*.zip; do
        unzip -tq "$z" >/dev/null 2>&1 || { echo "skip $(basename "$z"): not a complete zip (still copying?)"; continue; }
        unzip -n -q "$z" -d "$RAW" || { echo "unzip failed: $z" >&2; exit 1; }
    done
    echo "$(ls "$RAW" | wc -l) patient folders in $RAW"
fi
if has inventory; then
    echo "=== inventory ==="
    "${PY[@]}" "$REPO/scripts/adni_tau_prepare.py" inventory --raw "$RAW" --out "$ADNI/inventory.csv" \
        --followup "$FOLLOWUP" </dev/null || exit $?
fi
if has average; then
    echo "=== average (align $ALIGN, $THREADS threads) ==="
    "${PY[@]}" "$REPO/scripts/adni_tau_prepare.py" average --inventory "$ADNI/inventory.csv" \
        --static-dir "$ADNI/static" --align "$ALIGN" --threads "$THREADS" </dev/null || exit $?
fi
if has manifests; then
    echo "=== manifests ==="
    "${PY[@]}" "$REPO/scripts/adni_tau_prepare.py" manifests --inventory "$ADNI/inventory.csv" \
        --static-dir "$ADNI/static" --out-dir "$MANIFESTS" </dev/null || exit $?
fi
if has pet-labels; then
    echo "=== pet-labels ==="
    "${PY[@]}" "$REPO/scripts/adni_tau_prepare.py" pet-labels --processed-dir "$PROCESSED/adni_tau_pet" </dev/null || exit $?
fi
echo "finished adni_tau_prepare exit code 0"
