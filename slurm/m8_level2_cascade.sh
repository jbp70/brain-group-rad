#!/bin/bash
#SBATCH --job-name=bgr-m8-l2
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Milestone 8: three-task training at level 2 (rigid:affine:deform 1:1:2) on top of a FROZEN
# level-4 stage (design .agents/tasks/milestone-8/design.md v5, sections 2-9, 12). The frozen
# stage and the trainable level-2 stage both start from CASCADE_FROM (default: the milestone-7
# Dice+LNCC run's ckpt/best, step 11000); the trainable stage learns the residual. BN statistics
# of the trainable stage are recalibrated on level-2 data before step 0 (BN_RECAL_BATCHES, default
# 64; frozen afterwards with BN_RECAL_FREEZE=true). Deform loss Dice + LNCC 1/1, G 45 deg / 10 mm
# from step 0, curriculum off. Group sizes N (linear) and DEFORM_N have NO default: take them
# from the memory job (slurm/m8_memory.sh -> scratch/m8/memory_level2.json). Validation group
# sizes follow them (nv = N, nvd = DEFORM_N; the validation group-size flag is deliberately not
# passed, it would override both).
# After main.py, check_mixed_run.py checks the run dir (cascade provenance, frozen baselines,
# BN recalibration, encoder-gradient keys) and prints the grad-balance verdict. Exit code:
# main.py's if non-zero, else the checker's.
#
# Gradient balance (GRAD_BALANCE): gradnorm (default) = --task-grad-norm 12 (per-task gradient
# normalisation, task loss weights none); deform64 = --task-loss-weights deform:64 with the
# global clip 12 only (the milestone-7 recipe).
# C4 decision rule (read from the SMOKE run's checker output):
#   KEEP      iff the checker prints 'grad_balance KEEP task-grad-norm' (both linear/deform
#             encoder-gradient ratios in [0.1, 10]): keep GRAD_BALANCE=gradnorm for the long run.
#   FALLBACK  ('grad_balance FALLBACK deform:64') -> use GRAD_BALANCE=deform64.
#   UNDECIDED ('grad_balance UNDECIDED ...') -> treat as FALLBACK and inspect the run.
# On FALLBACK while the gradnorm long run is already running (submitted early, Jay's message 5):
#   scancel <our own m8long job id>   (never a job we did not submit)
#   then resubmit the long run with GRAD_BALANCE=deform64. The resubmission is a new job and
#   needs Jay's approval (state partition, GPU type, CPUs, memory, --time). Before resubmitting,
#   `squeue -u jn85` must show fewer than two active jobs of ours (running plus pending),
#   counted after the scancel has taken effect.
#
# CASCADE_FROM guard: on a fresh run CASCADE_FROM must be a checkpoint directory or a
# checkpoint prefix (<prefix>.index exists). The guard is skipped when
# $RESULTS/$RUN_NAME/ckpt/last/checkpoint exists (resume / requeue): --cascade-from is still
# passed to select the cascade code path, but it is not read (the frozen stage comes from the
# run checkpoint).
#
# DO NOT submit without Jay's approval (cluster-access rules). Account qtim only.
# Before sbatch, the log folder must exist (SLURM does not create it):
#   mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
# Cluster order: (a) memory job first, then (b) smoke, then (c) long.
# (a) memory job (slurm/m8_memory.sh):
#   sbatch --job-name=m8mem --time=04:00:00 --mem=64G slurm/m8_memory.sh
# (b) smoke (provisional --time 02:00:00 until job (a) reports):
#   RUN_NAME=m8_smoke N=<n> DEFORM_N=<nd> STEPS=200 VAL_EVERY=100 WARMUP=50 sbatch --job-name=m8smoke --time=02:00:00 --mem=64G slurm/m8_level2_cascade.sh
#   T_smoke = 1.5 x (200 x s_step + 4 x s_val + s_recal + 600 s), floor 01:30:00, rounded up to
#   15 min; s_step = (rigid + affine + 2 x deform) / 4 of the median step_ms at the chosen N /
#   options; s_val = sum over the validation sets (rigid, affine, rigid_g, affine_g, deform) of
#   n_groups x 2 (eval + trainmode) x 0.5 x median step_ms of that set's task; four passes
#   (bnrecal_pre, bnrecal_post, step 100, step 200); s_recal = 64 x 0.5 x task step.
# (c) long (provisional --time 24:00:00 until the smoke run is measured):
#   RUN_NAME=m8_level2_cascade N=<n> DEFORM_N=<nd> sbatch --job-name=m8long --time=24:00:00 --mem=64G slurm/m8_level2_cascade.sh
#   T = 1.3 x (8000 x mean step s + 16 x val s), from the smoke run.
# A100 override (same CPUs 3, --mem=64G): add -p dgx-a100 to the sbatch line, e.g.
#   RUN_NAME=m8_level2_cascade N=<n> DEFORM_N=<nd> sbatch -p dgx-a100 --job-name=m8long --time=<T> --mem=64G slurm/m8_level2_cascade.sh
#   peak (job 10504614, RTX 8000, scratch/m8/memory_level2.json): recipe N=16 DEFORM_N=8 AMP=true
#   GRAD_CKPT=both -> deform 27.8 GB (6.97 s/step), linear 10.9-11.0 GB (1.16 s/step).
#   verdict: fits the 40 GB A100 (peak 27.8 GB <= 36 GB).
#   Without AMP: deform N 8 OOMs with every checkpointing option (N 6 'both' 35.2 GB, 7.8 s);
#   linear N 12 fails in layer d2 (N 8 is 11.0 GB). AMP + 'integrator' at deform N 8: 40.9 GB, 6.5 s.
# Intensity augmentation stays in DALI here (default). To move it to the TF post-process (~3.5 GB
# less DALI memory at level 2), pass EXTRA_ARGS="--intensity-aug-device tf".
set -euo pipefail

PROJ="${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}"
REPO="${REPO:-$PROJ/brain-group-rad}"
DATA="${DATA:-$PROJ/data/npy_v2}"
RESULTS="${RESULTS:-$PROJ/results}"
SIF="${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_NAME="${RUN_NAME:-m8_level2_cascade_$JOB_ID}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"
STEPS="${STEPS:-8000}"
VAL_EVERY="${VAL_EVERY:-500}"
WARMUP="${WARMUP:-500}"
N="${N:-}"
DEFORM_N="${DEFORM_N:-}"
CASCADE_FROM="${CASCADE_FROM:-$RESULTS/m7_three_task_lncc/ckpt/best}"
GRAD_BALANCE="${GRAD_BALANCE:-gradnorm}"
BN_RECAL_BATCHES="${BN_RECAL_BATCHES:-64}"
BN_RECAL_FREEZE="${BN_RECAL_FREEZE:-true}"
SKIP_PROB="${SKIP_PROB:-0}"
GRAD_CKPT="${GRAD_CKPT:-none}"
AMP="${AMP:-false}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

case "$N" in ''|*[!0-9]*) echo "ERROR: set N (linear group size) from the memory job" >&2; exit 2;; esac
case "$DEFORM_N" in ''|*[!0-9]*) echo "ERROR: set DEFORM_N from the memory job" >&2; exit 2;; esac
case "$BN_RECAL_BATCHES" in ''|*[!0-9]*) echo "ERROR: BN_RECAL_BATCHES must be an integer" >&2; exit 2;; esac
# Guard only for fresh runs: a resume / requeue (ckpt/last exists) does not read the source (section 9).
if [ ! -f "$RESULTS/$RUN_NAME/ckpt/last/checkpoint" ]; then
    if [ ! -d "$CASCADE_FROM" ] && [ ! -f "$CASCADE_FROM.index" ]; then
        echo "ERROR: CASCADE_FROM $CASCADE_FROM is neither a checkpoint directory nor a prefix" >&2
        exit 2
    fi
fi
case "$GRAD_BALANCE" in
    gradnorm) BAL_ARGS="--task-grad-norm 12" ;;            # --task-loss-weights not passed (= none)
    deform64) BAL_ARGS="--task-loss-weights deform:64" ;;  # global clip 12 only
    *) echo "ERROR: GRAD_BALANCE must be gradnorm or deform64" >&2; exit 2 ;;
esac
RECAL_CHECK=""
if [ "$BN_RECAL_BATCHES" -gt 0 ]; then RECAL_CHECK="--expect-bn-recal true"; fi
# AMP: the dynamic loss scale overflows (inf gradients, step skipped, scale halved) a few times
# early on; the m8 smoke (job 10505028) skipped 10 of 200 steps, all linear, all by step 51.
if [ "$AMP" = "true" ]; then RECAL_CHECK="$RECAL_CHECK --max-nonfinite-frac 0.06"; fi

SCRATCH="$PROJ/scratch"
mkdir -p "$RESULTS" "$SCRATCH/keras" "$SCRATCH/cuda_cache"
export TMPDIR="$SCRATCH_ROOT/$JOB_ID"
mkdir -p "$TMPDIR"
export KERAS_HOME="$SCRATCH/keras"
export CUDA_CACHE_PATH="$SCRATCH/cuda_cache"
export PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_TMPDIR="$TMPDIR"
export SINGULARITYENV_KERAS_HOME="$KERAS_HOME"
export SINGULARITYENV_CUDA_CACHE_PATH="$CUDA_CACHE_PATH"
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1

cd "$REPO"
echo "host $(hostname) | job $JOB_ID | repo $(git rev-parse --short HEAD 2>/dev/null || echo '?') | $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
echo "run $RUN_NAME | steps $STEPS | val every $VAL_EVERY | warmup $WARMUP | N $N | deform N $DEFORM_N"
echo "cascade from $CASCADE_FROM | grad balance $GRAD_BALANCE ($BAL_ARGS) | bn recal $BN_RECAL_BATCHES (freeze $BN_RECAL_FREEZE) | skip $SKIP_PROB | grad ckpt $GRAD_CKPT | amp $AMP"

rc=0
# shellcheck disable=SC2086  # BAL_ARGS / EXTRA_ARGS are split on purpose
singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF" \
    python main.py --data "$DATA" --results "$RESULTS" --run-name "$RUN_NAME" \
    --level 2 --cascade-from "$CASCADE_FROM" --group-size "$N" --deform-group-size "$DEFORM_N" \
    --task-ratio rigid:1,affine:1,deform:2 --steps "$STEPS" --val-every "$VAL_EVERY" --warmup-steps "$WARMUP" \
    $BAL_ARGS --lambda-ncc-template 1 --lambda-ncc-pair 1 \
    --bn-recal-batches "$BN_RECAL_BATCHES" --bn-recal-freeze "$BN_RECAL_FREEZE" \
    --cascade-skip-prob "$SKIP_PROB" --gradient-checkpointing "$GRAD_CKPT" --amp "$AMP" \
    --global-transform true --global-rot-max-deg 45 --global-trans-max-mm 10 \
    --global-start-step 0 --global-ramp-steps 0 --curriculum false --val-png true \
    --num-threads "${SLURM_CPUS_PER_TASK:-3}" --log-every 25 --log-backend jsonl \
    $EXTRA_ARGS </dev/null || rc=$?
echo "main.py exit code $rc $(date -Is)"
if [ "$rc" -ne 0 ]; then
    exit "$rc"
fi
crc=0
# shellcheck disable=SC2086  # RECAL_CHECK is split on purpose
singularity exec -B /autofs -B /cluster -B /space "$SIF" \
    python tests/gpu/check_mixed_run.py "$RESULTS/$RUN_NAME" --tasks rigid,affine,deform --steps "$STEPS" \
    --expect-init-weights full --expect-enc-grad true --expect-cascade true --grad-balance-rule true \
    $RECAL_CHECK </dev/null || crc=$?
echo "finished check_mixed_run exit code $crc $(date -Is)"
exit "$crc"
