#!/bin/bash
#SBATCH --job-name=m9nanprobe
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# m9 level-1 deform NaN probe (.agents/tasks/m9-level1-nan/report.md section 6). READ-ONLY on the
# real run: results/m9_level1_cascade is only read (copied into scratch); everything is written to
# $PROJ/scratch/m9_nan/<job>/. The code under test is the run's own commit (RUN_COMMIT, default
# e6815d9), extracted with `git archive` into scratch (the clone is not touched); the probe tools
# come from this directory of the pulled clone.
#   A  ckpt_stats.py (CPU): per-variable finite / max|w| / relative change of ckpt/last vs
#      ckpt/best_deform (5000), ckpt/best (6500) and the level-2 source; BN variance minima; loss scale.
#   B  'last' (step 8000): resume a scratch copy for 3 deform-only steps with PROBE_ACT=1 (per-block
#      max/min of every conv / BN / v_raw) and PROBE_NUMERICS=1 (first op with inf/NaN output raises).
#   C  'bd5000' (ckpt/best_deform, step 5000, before the onset): same, PROBE_ACT=1 only -> the
#      float16 headroom (max |activation| vs 65504) per block before the failure.
# Overrides vs the run (forward-identical): --task-ratio deform:1 (TASK_RATIO), --min-pairs 1,
# --lean-losses true, --val-every 0, --ckpt-every 0, --val-png false, --log-every 1, --steps S+3.
# Submit (after Jay's approval; <= 2 active jobs; commit + push + git pull first):
#   sbatch .agents/tasks/m9-level1-nan/m9_nan_probe.sh
# If B/C stop at a resume guard about the task ratio, resubmit with TASK_RATIO=rigid:1,affine:1,deform:2.
set -uo pipefail

PROJ="${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}"
REPO="${REPO:-$PROJ/brain-group-rad}"
DATA="${DATA:-$PROJ/data/npy_v2}"
RESULTS="${RESULTS:-$PROJ/results}"
RUN="${RUN:-$RESULTS/m9_level1_cascade}"
SRC_CKPT_L2="${SRC_CKPT_L2:-$RESULTS/m8_level2_gradnorm/ckpt/best}"
SIF="${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}"
RUN_COMMIT="${RUN_COMMIT:-e6815d9}"
TASK_RATIO="${TASK_RATIO:-deform:1}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="$PROJ/scratch/m9_nan/$JOB_ID"
SRC="$PROJ/scratch/m9_nan/src_$RUN_COMMIT"
TOOLS="$REPO/.agents/tasks/m9-level1-nan"
mkdir -p "$OUT" "$PROJ/scratch/keras" "$PROJ/scratch/cuda_cache"
export TMPDIR="/scratch/$JOB_ID"; mkdir -p "$TMPDIR"
export SINGULARITYENV_TMPDIR="$TMPDIR"
export SINGULARITYENV_KERAS_HOME="$PROJ/scratch/keras"
export SINGULARITYENV_CUDA_CACHE_PATH="$PROJ/scratch/cuda_cache"
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
RUNC=(singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF")

echo "host $(hostname) | job $JOB_ID | clone $(git -C "$REPO" rev-parse --short HEAD) | code $RUN_COMMIT | $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
if [ ! -f "$SRC/main.py" ]; then
    mkdir -p "$SRC" && git -C "$REPO" archive "$RUN_COMMIT" | tar -x -C "$SRC" || { echo "git archive failed"; exit 2; }
fi

# ---- A: checkpoint statistics (CPU)
echo "=== A ckpt_stats $(date -Is)"
SINGULARITYENV_CUDA_VISIBLE_DEVICES="" timeout 900 "${RUNC[@]}" python "$TOOLS/ckpt_stats.py" \
    --out "$OUT/ckpt_stats.json" last="$RUN/ckpt/last" best_deform="$RUN/ckpt/best_deform" \
    best="$RUN/ckpt/best" l2_source="$SRC_CKPT_L2" </dev/null >"$OUT/ckpt_stats.txt" 2>&1
echo "=== A exit $?"; head -n 120 "$OUT/ckpt_stats.txt"

# ---- scratch copies of the run dir (config + control + one checkpoint as ckpt/last)
make_copy() {   # make_copy <name> <ckpt prefix>  -> prints the copy's results root
    local name="$1" prefix="$2" root="$OUT/res_$1"
    local d="$root/m9_level1_cascade"
    mkdir -p "$d/ckpt/last"
    cp "$RUN/config.json" "$d/"
    [ -d "$RUN/control" ] && cp -r "$RUN/control" "$d/"
    cp "$prefix".* "$d/ckpt/last/"
    local base; base="$(basename "$prefix")"
    printf 'model_checkpoint_path: "%s"\nall_model_checkpoint_paths: "%s"\n' "$base" "$base" >"$d/ckpt/last/checkpoint"
    echo "$root"
}
latest() { sed -n 's/^model_checkpoint_path: "\(.*\)"/\1/p' "$1/checkpoint"; }
P_LAST="$RUN/ckpt/last/$(latest "$RUN/ckpt/last")"
P_BD="$RUN/ckpt/best_deform/$(latest "$RUN/ckpt/best_deform")"
echo "last = $P_LAST | best_deform = $P_BD"

probe() {   # probe <name> <prefix> <PROBE_NUMERICS 0|1>
    local name="$1" prefix="$2" num="$3"
    local root; root="$(make_copy "$name" "$prefix")"
    local s; s="${prefix##*-}"
    local steps=$((s + 3))
    echo "=== $name start (resume step $s -> $steps, numerics $num) $(date -Is)"
    (cd "$SRC" && SINGULARITYENV_PROBE_ACT=1 SINGULARITYENV_PROBE_NUMERICS="$num" timeout 1500 "${RUNC[@]}" \
        python "$TOOLS/probe_activations.py" --data "$DATA" --results "$root" --run-name m9_level1_cascade \
        --level 1 --cascade-from "$SRC_CKPT_L2" --group-size 8 --deform-group-size 4 \
        --task-ratio "$TASK_RATIO" --steps "$steps" --val-every 0 --ckpt-every 0 --warmup-steps 500 \
        --task-grad-norm 12 --lambda-ncc-template 1 --lambda-ncc-pair 1 --lambda-smooth 0.5 \
        --p-pair 0.1 --p-pdice 0.1 --min-pairs 1 --bn-recal-batches 64 --bn-recal-freeze true \
        --cascade-skip-prob 0 --amp true --gradient-checkpointing full --lean-sampler true \
        --split-group-conv true --lean-losses true --val-pair-chunk 1 --val-image-chunk 1 \
        --bn-recal-validate false --global-transform true --global-rot-max-deg 45 \
        --global-trans-max-mm 10 --global-start-step 0 --global-ramp-steps 0 --curriculum false \
        --val-png false --num-threads "${SLURM_CPUS_PER_TASK:-3}" --prefetch 1 --log-every 1 \
        --log-backend jsonl --intensity-aug-device tf --pair-recompute true \
        --gpu-memory-limit-mb 40000 </dev/null >"$OUT/$name.log" 2>&1)
    echo "=== $name exit $? $(date -Is)"
    grep -c '^ACT' "$OUT/$name.log" | sed 's/^/ACT lines: /'
    echo "--- first ACT lines with inf / nan:"
    grep '^ACT' "$OUT/$name.log" | grep -n -iE 'inf|nan' | head -n 15
    echo "--- deform_decoder / svf ACT lines, all three stages (first 120; the trainable stage's"
    echo "    model scope is the one whose lines differ between 'last' and 'bd5000'):"
    grep '^ACT' "$OUT/$name.log" | grep -E '/deform_decoder/' | head -n 120
    echo "--- numerics / errors:"
    grep -n -A 12 -E 'Detected Infinity or NaN|Traceback|Error' "$OUT/$name.log" | head -n 60
    grep -E '"(loss|grad_norm|finite)"' "$root/m9_level1_cascade/logs/metrics.jsonl" 2>/dev/null | tail -n 5
}
probe last "$P_LAST" 1
probe bd5000 "$P_BD" 0
echo "done $(date -Is) | outputs in $OUT"
