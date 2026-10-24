#!/bin/bash
#SBATCH --job-name=bgr-m9-mem
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.out
#SBATCH --error=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs/%x-%j.err
#
# Milestone 9 phase 1, memory job (design .agents/tasks/milestone-9/design-m9-phase1.md section 5.5).
# Untrained weights; synthetic groups (data needed only for step 1, logs only for step 0c).
#   0a. probe_tensor_limit.py --mode run (fp32/fp16 single ops at 2^31 +- 2^24 elements, forward AND
#       backward via GradientTape) -> tensor_limit.json
#   0b. probe_tensor_limit.py --mode m8-errors: failed records + peak_mb / warmup_peak_mb of
#       scratch/m8/memory_level2.json
#   0c. step_times_from_metrics.py on results/m7_three_task_lncc, results/m8_level2_gradnorm and
#       the m5 run -> step_times.json (window-mean compute_ms: per task for the m8 cascade run,
#       task-mixed time/compute_ms for m7; median / max of the diag_u_res_max_vox window means for
#       the T = 6 magnitude check, design 3.6; the m5 G-schedule args for retraining step 1, 6.4)
#   1.  verify_dali_pipeline.py --level 1, checks b,c,d,e,l (--n-throughput 16) -> verify_level1.json
#   2p. PRIORITY (measured first, so a timeout cannot lose it): the recommendation (center widths,
#       $KNOBS, T 7) at N 3,4,5 with validation, and the fallback fr50 + lean sampler only (no split,
#       the margin-0.30 pick) at N 4: 4 configs (FEAT-004 orchestrator decision)
#   2.  level 1 deform, cascade (frozen-chain stand-in), WIDTHS x T 6,7 x N 2,3,4,5,6,8, ckpt full,
#       AMP, $KNOBS, validation pass (pair chunk 4): <= 36 configs (larger N skipped after an OOM)
#   2b. level 1 deform, --cascade false, center widths, N 2,4, T 7, $KNOBS: 2 configs
#   2c. level 1 deform, cascade, fr50 + lean losses (tests the Fw cap: predicted N5 39.6 GiB,
#       N6 47.3 GiB -> OOM expected), N 5,6,7, T 7: 3 configs
#   3.  level 1 deform ablations at the center widths, N 4, T 7: lean sampler off / split off /
#       lean losses on / checkpointing both: 4 configs
#   4.  level 1 rigid + affine, cascade, center widths, AMP, N 4,8,12,16: 8 configs
#   5.  level 2 deform (frozen level-4 cascade), center widths, AMP: the pinned m8 retraining recipe
#       (ckpt both, all knobs off, N 8) + ckpt full with $KNOBS (N 8,12,16): 4 configs
#   6.  level 4 deform, default + center widths, fp32, none, N 16,24,32: <= 6 configs (level-4
#       per-task step times for the m7 consistency check, design 4.5)
#   6b. level 4 rigid, default + center widths, fp32, N 16,32: 4 configs (replaces the unverified
#       "26-34 GB" linear N 32 range, design U14)
# Level 1 is measured with AMP only (selection rule, design 4.4 step 1); fp32 level-1 rows are
# model-only (scripts/memory_model.py grid).
# Every configuration builds at min(N, 2) (--build-n default); the build never sets a peak.
# Every measure call passes --isolate true: ONE configuration per Python process (the allocator keeps
# memory across configurations in one process; local hold-out resident_mb grew 73 -> 329 MB over 48
# configurations), so resident_mb is flat and no peak is inflated by an earlier configuration.
# All memory results merge into $OUT_DIR/memory_m9.json. Every call runs; the exit code is the first
# non-zero one (the tools record OOM / limit results and return 0 unless they crash).
# Time: level-1 deform config ~6 min (trace + cuDNN autotune ~3 min, 1 warm-up + 3 measured steps at
# ~25-50 s, validation pass ~1 min) x 49 (worst case incl. 2p, no OOM skips) = 4.9 h; linear 8 x 4 min
# = 0.5 h; level 2: 4 x 3 min; level 4: 10 x 1.5 min; probe, DALI, step times ~0.5 h; per-process
# start (interpreter + TF + CUDA, ~30 s) x ~75 configurations = ~0.6 h -> ~7 h;
# --time 12:00:00 (1.7x margin). Peak memory is the point of the job; the 48 GB RTX 8000 is the
# default. A100 variant (40 GB; the level-1 N >= 4 configs are expected to OOM or exceed 36 GiB there,
# which is recorded): add -p dgx-a100 (same CPUs and memory).
# Memory: --mem 64G (host: synthetic level-1 groups ~0.5 GB, DALI buffers); up to ~128G is a fair
# share on rtx8000 / dgx-a100 if host OOMs appear.
# WIDTHS = recommendation, one step wider, one step narrower; KNOBS = the recommendation's knobs.
# `python scripts/memory_model.py recommend --print-env` prints both lines; replace the defaults below
# if the calibrated model picks another configuration.
# (Design 5.5 text, with the British spelling of "center" replaced: the static test rejects the
# forbidden account name anywhere in the script, also as a substring.)
#
# DO NOT submit without Jay's approval (cluster-access rules). Account qtim only.
# Before sbatch, the log folder must exist (SLURM does not create it):
#   mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
# Submit (RTX 8000 / A6000 48 GB):
#   sbatch --job-name=m9mem --time=12:00:00 --mem=64G slurm/m9_memory.sh
set -euo pipefail

PROJ="${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}"
REPO="${REPO:-$PROJ/brain-group-rad}"
DATA="${DATA:-$PROJ/data/npy_v2}"
SIF="${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}"
JOB_ID="${SLURM_JOB_ID:-manual}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"
OUT_DIR="$PROJ/scratch/m9"
WIDTHS="${WIDTHS:-f0d1,default,fr50}"
CENTER="${WIDTHS%%,*}"
KNOBS="${KNOBS:---lean-sampler true --lean-losses false --split-group-conv true}"

SCRATCH="$PROJ/scratch"
mkdir -p "$OUT_DIR/dali" "$SCRATCH/keras" "$SCRATCH/cuda_cache"
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
echo "data $DATA | out $OUT_DIR | widths $WIDTHS (center $CENTER) | knobs $KNOBS"

RUN=(singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF")
rc=0
rc_x=0   # 0a: per-tensor limit probe (forward + backward)
"${RUN[@]}" python tests/gpu/probe_tensor_limit.py --mode run --out "$OUT_DIR/tensor_limit.json" </dev/null || rc_x=$?
echo "probe_tensor_limit (run) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 0b: the m8 failure texts and peak columns (read only)
"${RUN[@]}" python tests/gpu/probe_tensor_limit.py --mode m8-errors \
    --m8-json "$PROJ/scratch/m8/memory_level2.json" </dev/null || rc_x=$?
echo "probe_tensor_limit (m8-errors) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 0c: per-task step times from the m7 / m8 training logs (read only)
"${RUN[@]}" python scripts/step_times_from_metrics.py --min-step 1000 --extra-keys diag_u_res_max_vox \
    --args-keys global_transform,global_start_step,global_ramp_steps,global_rot_max_deg,global_trans_max_mm,curriculum,curriculum_ramp_steps,task_ratio,group_size,steps,lr,warmup_steps \
    --runs "$PROJ/results/m7_three_task_lncc" "$PROJ/results/m8_level2_gradnorm" \
           "$PROJ/results/m5_mixed_long_curr_dG_20261019-205526" \
    --out "$OUT_DIR/step_times.json" </dev/null || rc_x=$?
echo "step_times_from_metrics exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 1: level-1 DALI checks
"${RUN[@]}" python tests/gpu/verify_dali_pipeline.py --data "$DATA" --level 1 --checks b,c,d,e,l \
    --num-threads "${SLURM_CPUS_PER_TASK:-3}" --n-throughput 16 \
    --scratch "$OUT_DIR/dali" --out "$OUT_DIR/verify_level1.json" </dev/null || rc_x=$?
echo "verify_dali_pipeline (level 1) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi

M=(python tests/gpu/measure_level_memory.py --isolate true --lambda-ncc-template 1 --lambda-ncc-pair 1
   --out "$OUT_DIR/memory_m9.json")
L1=(--level 1 --amp true --warmup-steps 1 --steps 3)
rc_x=0   # 2p: PRIORITY - the recommendation at N 3,4,5 and the fr50 lean-sampler fallback at N 4
# shellcheck disable=SC2086
"${RUN[@]}" "${M[@]}" "${L1[@]}" --cascade true --tasks deform --widths "$CENTER" --integration-steps 7 \
    --n 3,4,5 --gradient-checkpointing full $KNOBS --val-pass true --val-pair-chunk 4 </dev/null || rc_x=$?
"${RUN[@]}" "${M[@]}" "${L1[@]}" --cascade true --tasks deform --widths fr50 --integration-steps 7 \
    --n 4 --gradient-checkpointing full --lean-sampler true --lean-losses false --split-group-conv false \
    --val-pass true --val-pair-chunk 4 </dev/null || rc_x=$?
echo "measure_level_memory (level 1 priority) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 2: level-1 deform grid (<= 36 configs, validation pass each)
# shellcheck disable=SC2086
"${RUN[@]}" "${M[@]}" "${L1[@]}" --cascade true --tasks deform --widths "$WIDTHS" --integration-steps 6,7 \
    --n 2,3,4,5,6,8 --gradient-checkpointing full $KNOBS --val-pass true --val-pair-chunk 4 </dev/null || rc_x=$?
echo "measure_level_memory (level 1 deform grid) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 2b: level-1 trainable step without the frozen chain (2 configs)
# shellcheck disable=SC2086
"${RUN[@]}" "${M[@]}" "${L1[@]}" --cascade false --tasks deform --widths "$CENTER" --integration-steps 7 \
    --n 2,4 --gradient-checkpointing full $KNOBS --val-pass false </dev/null || rc_x=$?
echo "measure_level_memory (level 1 no cascade) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 2c: fr50 + lean losses (tests the Fw cap: predicted N5 39.6 GiB, N6 47.3 GiB -> OOM expected), N 5,6,7
"${RUN[@]}" "${M[@]}" "${L1[@]}" --cascade true --tasks deform --widths fr50 --integration-steps 7 \
    --n 5,6,7 --gradient-checkpointing full --lean-sampler true --lean-losses true --split-group-conv true \
    --val-pass true --val-pair-chunk 4 </dev/null || rc_x=$?
echo "measure_level_memory (level 1 fr50 lean losses) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 3: level-1 ablations at the center widths (4 configs)
for ab in "--lean-sampler false --lean-losses false --split-group-conv true --gradient-checkpointing full" \
          "--lean-sampler true --lean-losses false --split-group-conv false --gradient-checkpointing full" \
          "--lean-sampler true --lean-losses true --split-group-conv true --gradient-checkpointing full" \
          "--lean-sampler true --lean-losses false --split-group-conv true --gradient-checkpointing both"; do
    # shellcheck disable=SC2086
    "${RUN[@]}" "${M[@]}" "${L1[@]}" --cascade true --tasks deform --widths "$CENTER" --integration-steps 7 \
        --n 4 $ab --val-pass false </dev/null || rc_x=$?
done
echo "measure_level_memory (level 1 ablations) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 4: level-1 linear (8 configs)
"${RUN[@]}" "${M[@]}" "${L1[@]}" --cascade true --tasks rigid,affine --widths "$CENTER" --n 4,8,12,16 \
    --split-group-conv true </dev/null || rc_x=$?
echo "measure_level_memory (level 1 linear) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 5: level-2 deform at the center widths (4 configs)
# 5a: the pinned level-2 retraining recipe (m8: both, no knobs) at the reduced widths, N 8
"${RUN[@]}" "${M[@]}" --level 2 --cascade true --amp true --warmup-steps 2 --steps 5 --tasks deform \
    --widths "$CENTER" --n 8 --gradient-checkpointing both --lean-sampler false --lean-losses false \
    --split-group-conv false --val-pass true </dev/null || rc_x=$?
# 5b: full with the recommendation's knobs, N 8,12,16
# shellcheck disable=SC2086
"${RUN[@]}" "${M[@]}" --level 2 --cascade true --amp true --warmup-steps 2 --steps 5 --tasks deform \
    --widths "$CENTER" --n 8,12,16 --gradient-checkpointing full $KNOBS --val-pass true </dev/null || rc_x=$?
echo "measure_level_memory (level 2 deform) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 6: level-4 deform, default and center widths (<= 6 configs; default N 24 OOM -> 32 skipped)
"${RUN[@]}" "${M[@]}" --level 4 --cascade false --amp false --warmup-steps 2 --steps 5 --tasks deform \
    --widths "default,$CENTER" --n 16,24,32 --gradient-checkpointing none </dev/null || rc_x=$?
echo "measure_level_memory (level 4 deform) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi
rc_x=0   # 6b: level-4 linear, default and center widths (4 configs)
"${RUN[@]}" "${M[@]}" --level 4 --cascade false --amp false --warmup-steps 2 --steps 5 --tasks rigid \
    --widths "default,$CENTER" --n 16,32 </dev/null || rc_x=$?
echo "measure_level_memory (level 4 linear) exit code $rc_x $(date -Is)"
if [ "$rc" -eq 0 ]; then rc=$rc_x; fi

echo "finished m9_memory exit code $rc $(date -Is)"
exit "$rc"
