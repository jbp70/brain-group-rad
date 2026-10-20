#!/usr/bin/env bash
# Milestone-4 checks on glacier, inside tf10_nv.sif, on ONE GPU:
#   overfit  main.py --overfit-one-group on real level-4 training images (N = 8, fixed rigid
#            transforms, TF-side warp; must reach a 10-step mean centered RMS < 1 mm)
#   dryrun   main.py --dry-run with the real DALI pipeline (N = 8): builds the DALI datasets,
#            the validation set (1 group), runs 2 steps + 1 validation, writes a checkpoint
#   dryrun_ctl  the same dry run with rigid:1,affine:1, --curriculum true --global-transform true
#            (control files under results/<run>/control, val_<task>_g sets)
#   dryrun_mixed  milestone 5: a short REAL-data mixed run, rigid:1,affine:1 (seeded alternation,
#            one shared Adam), N = 8, level 4, 20 steps, validation (1 group of 8 val cases per
#            task) and PNGs at steps 10 and 20, best checkpoint on val/mean_pair_rms_mm. Then
#            tests/gpu/check_mixed_run.py checks the run directory (both tasks trained and
#            validated with finite values, affine errors present, one trace per task, PNGs
#            rigid_/affine_step0000020.png, a best checkpoint); its exit code is the mode's.
#   dryrun_deform  milestone 6: two short REAL-data runs with the DALI DeformPipeline (images +
#            train18 labels), N = 8, level 4, 20 steps, validation (1 group of 8 val cases) and
#            PNGs at steps 10 and 20: (1) deform_<stamp> --task-ratio deform:1 (best checkpoint
#            on val_deform/pair_dice_loss), (2) mixed3_<stamp> --task-ratio
#            rigid:1,affine:1,deform:2 (ckpt/best on val/mean_pair_rms_mm, ckpt/best_deform).
#            Each run is checked by tests/gpu/check_mixed_run.py (deform: train_deform/loss,
#            val_deform/* finite, deform_step0000020.png, trace once per task); the exit code
#            is the first failure (main.py or checker), else 0.
#
# Usage (on glacier, after `git pull` in the clone; pick a free GPU with nvidia-smi first):
#   bash /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad/tests/gpu/run_overfit_glacier.sh <gpu_index> [timeout_seconds] [overfit|dryrun|dryrun_ctl|dryrun_mixed|dryrun_deform]
#
# Writes only inside $PROJ/scratch/m4 (dryrun_mixed: $PROJ/scratch/m5, dryrun_deform:
# $PROJ/scratch/m6; the timeout applies to each main.py run separately): results/<run>/
# (config.json, logs, tb, ckpt, val_png, overfit_summary.json / fit_summary.json), <mode>.log,
# the CUDA JIT cache, the Keras config folder and TMPDIR. TF32 is turned off by main.py
# (models.precision.configure_tf_precision). Bytecode writing is disabled, so the repo clone
# stays clean. Exit code: that of main.py (overfit: 1 = threshold not reached; 124 = timeout),
# for dryrun_mixed that of main.py if non-zero, else that of check_mixed_run.py.
set -euo pipefail
GPU="${1:?usage: run_overfit_glacier.sh <gpu_index> [timeout_seconds] [overfit|dryrun|dryrun_ctl|dryrun_mixed|dryrun_deform]}"
TIMEOUT_S="${2:-3600}"
MODE="${3:-overfit}"
if ! [[ "$GPU" =~ ^[0-9]+$ ]]; then
    echo "gpu_index must be a single integer, got '$GPU'" >&2
    exit 2
fi
if ! [[ "$TIMEOUT_S" =~ ^[0-9]+$ ]]; then
    echo "timeout_seconds must be an integer, got '$TIMEOUT_S'" >&2
    exit 2
fi
if [[ "$MODE" != "overfit" && "$MODE" != "dryrun" && "$MODE" != "dryrun_ctl" && "$MODE" != "dryrun_mixed" && "$MODE" != "dryrun_deform" ]]; then
    echo "mode must be 'overfit', 'dryrun', 'dryrun_ctl', 'dryrun_mixed' or 'dryrun_deform', got '$MODE'" >&2
    exit 2
fi
PROJ=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
REPO="$PROJ/brain-group-rad"
DATA="$PROJ/data/npy_v2"
SCRATCH="$PROJ/scratch/m4"
if [[ "$MODE" == "dryrun_mixed" ]]; then
    SCRATCH="$PROJ/scratch/m5"
elif [[ "$MODE" == "dryrun_deform" ]]; then
    SCRATCH="$PROJ/scratch/m6"
fi
SIF=/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$SCRATCH/results" "$SCRATCH/tmp"
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
export TMPDIR="$SCRATCH/tmp"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_CACHE_PATH="$SCRATCH/cuda_cache"
export KERAS_HOME="$SCRATCH/keras"
export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU"
export SINGULARITYENV_TMPDIR="$SCRATCH/tmp"
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_CUDA_CACHE_PATH="$SCRATCH/cuda_cache"
export SINGULARITYENV_KERAS_HOME="$SCRATCH/keras"
echo "host $(hostname) | GPU $GPU | mode $MODE | timeout ${TIMEOUT_S}s | repo $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
nvidia-smi -i "$GPU" --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
if [[ "$MODE" == "dryrun_deform" ]]; then
    LOG="$SCRATCH/$MODE.log"
    : > "$LOG"
    rc=0
    for spec in "deform_$STAMP|deform:1|deform" "mixed3_$STAMP|rigid:1,affine:1,deform:2|rigid,affine,deform"; do
        IFS='|' read -r RUN RATIO TASKS <<< "$spec"
        echo "=== run $RUN | task ratio $RATIO" | tee -a "$LOG"
        CMD=(python main.py --data "$DATA" --level 4 --group-size 8 --task-ratio "$RATIO"
             --steps 20 --warmup-steps 5 --log-every 5 --val-every 10 --ckpt-every 10
             --val-groups 1 --val-group-size 8 --results "$SCRATCH/results" --run-name "$RUN"
             --resume false --num-threads 4 --log-backend jsonl)
        set +e
        timeout "$TIMEOUT_S" singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF" \
            "${CMD[@]}" </dev/null 2>&1 | tee -a "$LOG"
        run_rc=${PIPESTATUS[0]}
        if [[ "$run_rc" -eq 0 ]]; then
            singularity exec -B /autofs -B /space "$SIF" \
                python tests/gpu/check_mixed_run.py "$SCRATCH/results/$RUN" --tasks "$TASKS" --steps 20 \
                </dev/null 2>&1 | tee -a "$LOG"
            run_rc=${PIPESTATUS[0]}
        fi
        set -e
        echo "run $RUN exit code: $run_rc" | tee -a "$LOG"
        if [[ "$rc" -eq 0 && "$run_rc" -ne 0 ]]; then
            rc=$run_rc
        fi
    done
    echo "$MODE exit code: $rc (log: $LOG, results: $SCRATCH/results)"
    exit "$rc"
fi
if [[ "$MODE" == "overfit" ]]; then
    CMD=(python main.py --overfit-one-group --data "$DATA" --level 4 --group-size 8
         --overfit-max-steps 3000 --warmup-steps 100 --results "$SCRATCH/results" --run-name "overfit_$STAMP"
         --log-every 50 --val-every 250 --log-backend jsonl)
elif [[ "$MODE" == "dryrun" ]]; then
    CMD=(python main.py --dry-run --data "$DATA" --level 4 --group-size 8
         --results "$SCRATCH/results" --run-name "dryrun_$STAMP" --num-threads 4 --log-backend jsonl)
elif [[ "$MODE" == "dryrun_ctl" ]]; then
    CMD=(python main.py --dry-run --data "$DATA" --level 4 --group-size 8
         --results "$SCRATCH/results" --run-name "dryrun_ctl_$STAMP" --num-threads 4 --log-backend jsonl
         --task-ratio rigid:1,affine:1 --curriculum true --global-transform true)
else
    RUN="dryrun_mixed_$STAMP"
    CMD=(python main.py --data "$DATA" --level 4 --group-size 8 --task-ratio rigid:1,affine:1
         --steps 20 --warmup-steps 5 --log-every 5 --val-every 10 --ckpt-every 10
         --val-groups 1 --val-group-size 8 --results "$SCRATCH/results" --run-name "$RUN"
         --resume false --num-threads 4 --log-backend jsonl)
fi
LOG="$SCRATCH/$MODE.log"
set +e
timeout "$TIMEOUT_S" singularity exec --nv -B /scratch -B /autofs -B /cluster -B /space -B /vast "$SIF" \
    "${CMD[@]}" </dev/null 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
if [[ "$MODE" == "dryrun_mixed" && "$rc" -eq 0 ]]; then
    singularity exec -B /autofs -B /space "$SIF" \
        python tests/gpu/check_mixed_run.py "$SCRATCH/results/$RUN" --tasks rigid,affine --steps 20 \
        </dev/null 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
fi
echo "$MODE exit code: $rc (log: $LOG, results: $SCRATCH/results)"
exit "$rc"
