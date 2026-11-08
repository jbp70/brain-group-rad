#!/bin/bash
#SBATCH --job-name=bgr-probe1mm
#SBATCH -A qtim
#SBATCH -p rtx6000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
# 1 mm integration probe (read-only for the repo; outputs under scratch/agent/invcen1mm).
set -uo pipefail
P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
SIF=/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif
CKPT=${CKPT:-$P/results/r1_cfgB_sm20_s2500_bnfreeze_true/ckpt/last}
OUT=${OUT:-$P/scratch/agent/invcen1mm/r1_cfgB_sm20}
export TMPDIR=/scratch/${SLURM_JOB_ID:-manual}
mkdir -p "$TMPDIR" "$OUT"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} repo $(git -C $P/brain-group-rad log --oneline -1)"
singularity exec --nv -B /autofs -B /scratch "$SIF" python $P/scratch/agent/invcen1mm/probe_integrate_1mm.py \
  --repo $P/brain-group-rad --checkpoint "$CKPT" --processed-dir $P/data/processed/mindboggle \
  --grid-reference $P/data/target_grid/target_grid.json --out "$OUT" --workers 3 </dev/null
echo "finished rc $?"
