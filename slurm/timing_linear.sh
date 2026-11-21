#!/bin/bash
#SBATCH --job-name=bgr-time-lin
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Registration time per group for the MIRIAD rigid / affine table (paper Table 1), all GPU methods
# on the SAME GPU in one job: ours (TTA none and flip8), BrainMorph, SynthMorph, rigid and affine.
# Same groups as the MIRIAD evaluation (one patient = one group, 4-12 scans): N_GROUPS patients
# evenly spread over the group sizes. Each group is registered REPS times in a row (rep 0 cold,
# later reps warm); images are read before the clock starts and nothing is written to disk
# (scripts/timing/time_*.py). ANTs is not re-run (CPU, ~2.5 h per group; its recorded times are used).
# DO NOT submit without Jay's approval (cluster-access rules).
#
# Environment overrides:
#   N_GROUPS   patients to time (default 10)       REPS  runs per group (default 2)
#   STEPS      subset of: ours_rigid ours_rigid_tta ours_affine ours_affine_tta bm_rigid bm_affine
#              sm_rigid sm_affine (default all)
#   CKPT       default $PROJ/results/r1_v3_gB/ckpt/last     PROCESSED  default $PROJ/data/processed/miriad
#   GROUPS_DIR default $PROJ/results/predict_miriad/v3_gB_rigid/groups
#   OUT        default $PROJ/results/timing/linear_<job id>
set -uo pipefail
PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
REPO=${REPO:-$PROJ/brain-group-rad}
CKPT=${CKPT:-$PROJ/results/r1_v3_gB/ckpt/last}
PROCESSED=${PROCESSED:-$PROJ/data/processed/miriad}
GROUPS_DIR=${GROUPS_DIR:-$PROJ/results/predict_miriad/v3_gB_rigid/groups}
GRID_REF=${GRID_REF:-$PROJ/data/target_grid/target_grid.json}
N_GROUPS=${N_GROUPS:-10}
REPS=${REPS:-2}
STEPS=${STEPS:-"ours_rigid ours_rigid_tta ours_affine ours_affine_tta bm_rigid bm_affine sm_rigid sm_affine"}
OUT=${OUT:-$PROJ/results/timing/linear_${SLURM_JOB_ID:-manual}}
TF_SIF=/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif
PT_SIF=/autofs/space/crater_001/users/jn85/Singularity_Files/pt_nv.sif
BM=$PROJ/baselines/brainmorph
export FREESURFER_HOME=${FREESURFER_HOME:-/usr/local/freesurfer/dev}
FSPY=$FREESURFER_HOME/bin/fspython
export TMPDIR=/scratch/${SLURM_JOB_ID:-manual}
mkdir -p "$TMPDIR" "$OUT"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  repo $(git -C "$REPO" log --oneline -1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# N_GROUPS patients evenly spread over (group size, name)
SEL=$(for f in "$GROUPS_DIR"/*.txt; do b=$(basename "$f" .txt); case $b in _*) continue;; esac
      echo "$(grep -c . "$f") $b"; done | sort -k1,1n -k2,2 | awk -v k="$N_GROUPS" '{s[NR]=$0} END {
      for (i = 0; i < k; i++) { j = (k == 1) ? 1 : int(1 + i * (NR - 1) / (k - 1) + 0.5); print s[j] } }')
echo "$SEL" > "$OUT/groups_selected.txt"
SEL_GROUPS=$(echo "$SEL" | awk '{print $2}' | tr '\n' ' ')
echo "groups (size name):"; cat "$OUT/groups_selected.txt"

TF=(singularity exec --nv -B /autofs -B /scratch "$TF_SIF" bash -c 'cd "$1" && shift && python -u "$@"' t "$REPO")
PT=(singularity exec --nv -B /autofs -B /scratch "$PT_SIF" bash -c 'cd "$1" && shift && python -u "$@"' t "$REPO")
fails=0
step() { echo "################ $1  $(date)"; }
for s in $STEPS; do
    step "$s"
    case $s in
        ours_rigid|ours_affine|ours_rigid_tta|ours_affine_tta)
            task=${s#ours_}; task=${task%_tta}; tta=none; [ "${s%_tta}" != "$s" ] && tta=flip8
            # shellcheck disable=SC2086
            "${TF[@]}" scripts/timing/time_ours.py --checkpoint "$CKPT" --processed-dir "$PROCESSED" \
                --label-name synthseg --grid-reference "$GRID_REF" --branches "$task" --tta "$tta" \
                --groups-dir "$GROUPS_DIR" --groups $SEL_GROUPS --reps "$REPS" --label "$s" --out "$OUT/$s.json" </dev/null ;;
        bm_rigid|bm_affine)
            # shellcheck disable=SC2086
            "${PT[@]}" scripts/timing/time_brainmorph.py --keymorph-dir "$BM/keymorph" \
                --weights "$BM/weights/foundation-numkey512-numlevels6.pth.tar" --processed-dir "$PROCESSED" \
                --label-name synthseg --align "${s#bm_}" --groups-dir "$GROUPS_DIR" --groups $SEL_GROUPS \
                --reps "$REPS" --out "$OUT/$s.json" </dev/null ;;
        sm_rigid|sm_affine)
            # shellcheck disable=SC2086
            (cd "$REPO" && NEURITE_BACKEND=tensorflow VXM_BACKEND=tensorflow TF_CPP_MIN_LOG_LEVEL=2 \
                "$FSPY" -u scripts/timing/time_synthmorph.py --processed-dir "$PROCESSED" --label-name synthseg \
                --align "${s#sm_}" --groups-dir "$GROUPS_DIR" --groups $SEL_GROUPS --reps "$REPS" \
                --out "$OUT/$s.json" </dev/null) ;;
        *) echo "unknown step $s"; false ;;
    esac
    rc=$?
    echo "################ $s finished rc $rc  $(date)"
    [ "$rc" = 0 ] || fails=$((fails + 1))
done
echo "timing finished: $fails failed -> $OUT"
exit $fails
