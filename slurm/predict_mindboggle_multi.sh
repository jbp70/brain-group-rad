#!/bin/bash
#SBATCH --job-name=bgr-predict-multi
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Runs slurm/predict_mindboggle.sh once per entry of RUNS, sequentially on one GPU (one job instead of
# one per checkpoint). RUNS: space-separated NAME:CKPT entries (NAME without commas or colons).
# Every other predict_mindboggle.sh setting (BRANCHES, LEVELS, CHECK_CHUNKS, EXTRA, ...) applies to
# all entries (inherited environment; e.g. TTA=flip8, RECENTER=2, SKIP_CHECKS=1). A failing entry is reported and the loop continues; exit status = number of failures.
# DO NOT submit without Jay's approval (cluster-access rules).
#   RUNS="a:/path/ckpt/last b:/path2/ckpt/last" sbatch slurm/predict_mindboggle_multi.sh
set -uo pipefail
PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
REPO=${REPO:-$PROJ/brain-group-rad}
RUNS=${RUNS:?set RUNS to NAME:CKPT entries}
fails=0
for e in $RUNS; do
    n=${e%%:*}; c=${e#*:}
    echo "################ $n  ($c)  $(date)"
    NAME=$n CKPT=$c OUT=$PROJ/results/predict_mindboggle/$n bash "$REPO/slurm/predict_mindboggle.sh"
    rc=$?
    echo "################ $n finished rc $rc  $(date)"
    [ "$rc" = 0 ] || fails=$((fails + 1))
done
echo "multi finished: $fails failed"
exit $fails
