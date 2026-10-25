#!/bin/bash
#SBATCH --job-name=bgr-m9-l1
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Milestone 9 phase 2: three-task training at level 1 (full resolution, 160x224x192 at 1 mm;
# rigid:affine:deform 1:1:2) as the third stage of a FROZEN level 4 -> level 2 cascade (design
# .agents/tasks/milestone-9-phase2/design-m9-phase2.md v3, sections 0, 6-9). Both frozen stages
# come from ONE checkpoint, CASCADE_FROM (default: the milestone-8 level-2 cascade run's
# ckpt/best, step 6000; its stored frozen_stage object is the level-4 stage, fingerprint-checked
# against the m8 config.json cascade record). The trainable level-1 stage starts from the level-2
# weights, has the IDENTICAL default ModelConfig (no --filters / --decoder-widths /
# --integration-steps here: an explicit list would not be dict-equal to m8's stored config) and
# learns the residual. BN statistics of the trainable stage are recalibrated on level-1 data at
# step 0 of a fresh run (BN_RECAL_BATCHES, default 64; frozen afterwards with BN_RECAL_FREEZE=true;
# never repeated on resume). Recipe (design D5): linear N 16, deform N 4, AMP,
# --gradient-checkpointing full, --lean-sampler true, --split-group-conv true, T 7, --min-pairs n
# (deform N 4 -> 4 pairs as a random 4-cycle; linear N 16 -> 24 pairs, unchanged), p 0.1 / 0.1,
# Dice + LNCC 1/1, smoothness 0.5, --task-grad-norm 12, G 45 deg / 10 mm from step 0, curriculum
# off, skip 0 (SKIP_PROB; design D6, to be confirmed with Jay), --val-pair-chunk 4.
# Validation group sizes follow N / DEFORM_N (the validation group-size flag is deliberately not
# passed, it would override both).
# After main.py, check_mixed_run.py checks the run dir (3-stage cascade provenance, _frozen and
# _frozen4 baselines, BN recalibration, encoder-gradient keys, pair counts) and prints the
# grad-balance verdict plus the INFO gpu_peak / step_time lines. Exit code: main.py's if
# non-zero, else the checker's.
#
# Measured facts (job 10516237, RTX 8000, scratch/m9/memory_m9.json; stand-in chain, AMP, full
# checkpointing, lean sampler + split, T 7, ONE deform pair): level-1 deform N 4 default widths
# 31.0 GB, 15.6 s/step; validation peak 22.3 GB (chunk 4); N 5 OOM except fr50 + lean losses.
# Predicted with 4 pairs (--min-pairs n): +4 to +7 GB -> 35-38 GB (unverified until the smoke).
# Estimated mean step at 1:1:2 ~11.3 s on the RTX 8000 (deform ~18 s, linear ~4.5 s).
# If the RTX 8000 smoke OOMs (peak > ~46 GiB): rerun the smoke with LEAN_LOSSES=true; if that
# still fails, MIN_PAIRS=1 and report to Jay.
#
# A100 rule: -p dgx-a100 ONLY if the smoke's 'INFO gpu_peak' line says it fits the 36 GiB
# budget (peak <= 36 GiB); same 3 CPUs / --mem=64G. Otherwise keep rtx8000.
#
# Gradient balance (GRAD_BALANCE): gradnorm (default) = --task-grad-norm 12 (per-task gradient
# normalisation, task loss weights none); deform64 = --task-loss-weights deform:64 with the
# global clip 12 only (the milestone-7 recipe).
# Decision rule (read from the SMOKE run's checker output):
#   KEEP      iff the checker prints 'grad_balance KEEP task-grad-norm' (both linear/deform
#             encoder-gradient ratios in [0.1, 10]): keep GRAD_BALANCE=gradnorm for the long run.
#   FALLBACK  ('grad_balance FALLBACK deform:64') -> use GRAD_BALANCE=deform64.
#   UNDECIDED ('grad_balance UNDECIDED ...') -> treat as FALLBACK and inspect the run.
# Prefetch rule: if the smoke checker's 'INFO data wait' fraction of a linear task is > 0.10,
# run the long job with PREFETCH=2 (+440 MB host).
# Host memory rule: read `sacct -j <smoke id> --format=MaxRSS`; if MaxRSS > 56 GB, submit the
# long run with --mem=96G (within the ~128 GB fair share).
#
# CASCADE_FROM guard: on a fresh run CASCADE_FROM must be a checkpoint directory or a
# checkpoint prefix (<prefix>.index exists). The guard is skipped when
# $RESULTS/$RUN_NAME/ckpt/last/checkpoint exists (resume / requeue): --cascade-from is still
# passed to select the cascade code path, but it is not read (both frozen stages come from the
# run checkpoint, keys frozen_stage_4 / frozen_stage_2).
#
# DO NOT submit without Jay's approval (cluster-access rules). Account qtim only.
# Before sbatch, the log folder must exist (SLURM does not create it):
#   mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
# Cluster order: (a) git pull, then (b) smoke, then (c) long.
# (a) cd /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad && git pull --ff-only
# (b) smoke: T_smoke = 1.5 x (200 x 11.3 s + recal batches 660 s + recal pre pass 660 s + recal
#     post pass 300 s + 2 regular passes x 300 s + startup / trace / val-set loading 900 s)
#     ~ 1.5 x 5380 s ~ 2.25 h -> --time 04:00:00 (first level-1 run, generous for unknowns).
#     The recal event's elapsed_s includes its pre / post validation passes; those passes never
#     log val/wall_s, so the projection adds recal elapsed_s once and the val/wall_s passes
#     separately.
#   RUN_NAME=m9_l1_smoke STEPS=200 VAL_EVERY=100 WARMUP=50 sbatch --job-name=m9smoke --time=04:00:00 --mem=64G slurm/m9_level1_cascade.sh
# (c) long: T = 1.3 x (8000 x s_mean + 16 x s_val + s_recal), printed by the smoke checker's
#     'INFO step_time' line (--project-steps 8000 --project-val-every 500, so the 16 passes use
#     the long run's cadence, not the smoke's VAL_EVERY 100). Provisional:
#     1.3 x (8000 x 11.3 + 16 x 300 + 1000) ~ 1.3 x 96200 s ~ 35 h -> --time 48:00:00.
#   RUN_NAME=m9_level1_cascade sbatch --job-name=m9long --time=48:00:00 --mem=64G slurm/m9_level1_cascade.sh
#   A100 (only if the smoke peak <= 36 GiB; ~1.5x faster, ~24 h):
#   RUN_NAME=m9_level1_cascade sbatch -p dgx-a100 --job-name=m9long --time=36:00:00 --mem=64G slurm/m9_level1_cascade.sh
# Resume plan: checkpoints every 250 steps (CKPT_EVERY, ~47 min at 11.3 s). If the job hits
# --time, fails or is preempted, resubmit the identical line with the same RUN_NAME (Jay's
# approval; at most two active jobs): --resume (default true) restores ckpt/last, both frozen
# stages and the RNGs, checks both fingerprints, re-applies the saved BN freeze and never repeats
# the recalibration; at most 250 steps are lost.
set -euo pipefail

PROJ="${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}"
REPO="${REPO:-$PROJ/brain-group-rad}"
DATA="${DATA:-$PROJ/data/npy_v2}"
RESULTS="${RESULTS:-$PROJ/results}"
SIF="${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_NAME="${RUN_NAME:-m9_level1_cascade_$JOB_ID}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"
STEPS="${STEPS:-8000}"
VAL_EVERY="${VAL_EVERY:-500}"
CKPT_EVERY="${CKPT_EVERY:-250}"
WARMUP="${WARMUP:-500}"
N="${N:-16}"
DEFORM_N="${DEFORM_N:-4}"
CASCADE_FROM="${CASCADE_FROM:-$RESULTS/m8_level2_gradnorm/ckpt/best}"
GRAD_BALANCE="${GRAD_BALANCE:-gradnorm}"
BN_RECAL_BATCHES="${BN_RECAL_BATCHES:-64}"
BN_RECAL_FREEZE="${BN_RECAL_FREEZE:-true}"
SKIP_PROB="${SKIP_PROB:-0}"
MIN_PAIRS="${MIN_PAIRS:-n}"
LEAN_LOSSES="${LEAN_LOSSES:-false}"
PREFETCH="${PREFETCH:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

case "$N" in ''|*[!0-9]*) echo "ERROR: N (linear group size) must be an integer, got '$N'" >&2; exit 2;; esac
case "$DEFORM_N" in ''|*[!0-9]*) echo "ERROR: DEFORM_N must be an integer, got '$DEFORM_N'" >&2; exit 2;; esac
case "$BN_RECAL_BATCHES" in ''|*[!0-9]*) echo "ERROR: BN_RECAL_BATCHES must be an integer" >&2; exit 2;; esac
case "$MIN_PAIRS" in
    1|n) ;;
    *) echo "ERROR: MIN_PAIRS must be 1 or n, got '$MIN_PAIRS'" >&2; exit 2 ;;
esac
# Guard only for fresh runs: a resume / requeue (ckpt/last exists) does not read the source.
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
echo "run $RUN_NAME | level 1 | steps $STEPS | val every $VAL_EVERY | ckpt every $CKPT_EVERY | warmup $WARMUP | N $N | deform N $DEFORM_N"
echo "cascade from $CASCADE_FROM | grad balance $GRAD_BALANCE ($BAL_ARGS) | bn recal $BN_RECAL_BATCHES (freeze $BN_RECAL_FREEZE) | skip $SKIP_PROB | min pairs $MIN_PAIRS | lean losses $LEAN_LOSSES | prefetch $PREFETCH"

rc=0
# shellcheck disable=SC2086  # BAL_ARGS / EXTRA_ARGS are split on purpose
singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF" \
    python main.py --data "$DATA" --results "$RESULTS" --run-name "$RUN_NAME" \
    --level 1 --cascade-from "$CASCADE_FROM" --group-size "$N" --deform-group-size "$DEFORM_N" \
    --task-ratio rigid:1,affine:1,deform:2 --steps "$STEPS" --val-every "$VAL_EVERY" \
    --ckpt-every "$CKPT_EVERY" --warmup-steps "$WARMUP" \
    $BAL_ARGS --lambda-ncc-template 1 --lambda-ncc-pair 1 --lambda-smooth 0.5 \
    --p-pair 0.1 --p-pdice 0.1 --min-pairs "$MIN_PAIRS" \
    --bn-recal-batches "$BN_RECAL_BATCHES" --bn-recal-freeze "$BN_RECAL_FREEZE" --cascade-skip-prob "$SKIP_PROB" \
    --amp true --gradient-checkpointing full --lean-sampler true --split-group-conv true \
    --lean-losses "$LEAN_LOSSES" --val-pair-chunk 4 \
    --global-transform true --global-rot-max-deg 45 --global-trans-max-mm 10 \
    --global-start-step 0 --global-ramp-steps 0 --curriculum false --val-png true \
    --num-threads "${SLURM_CPUS_PER_TASK:-3}" --prefetch "$PREFETCH" --log-every 25 --log-backend jsonl \
    $EXTRA_ARGS </dev/null || rc=$?
echo "main.py exit code $rc $(date -Is)"
if [ "$rc" -ne 0 ]; then
    exit "$rc"
fi
crc=0
# AMP: the dynamic loss scale overflows a few times early on (m8: 10-11 skipped steps of 200 /
# 8000, all linear), hence --max-nonfinite-frac 0.06.
# shellcheck disable=SC2086  # RECAL_CHECK is split on purpose
singularity exec -B /autofs -B /cluster -B /space "$SIF" \
    python tests/gpu/check_mixed_run.py "$RESULTS/$RUN_NAME" --tasks rigid,affine,deform --steps "$STEPS" \
    --expect-init-weights full --expect-enc-grad true --expect-cascade true --expect-cascade-stages 2 \
    --grad-balance-rule true $RECAL_CHECK --max-nonfinite-frac 0.06 \
    --expect-min-pairs "$MIN_PAIRS" --memory-budget-mb 36864 --project-steps 8000 \
    --project-val-every 500 </dev/null || crc=$?
echo "finished check_mixed_run exit code $crc $(date -Is)"
exit "$crc"
