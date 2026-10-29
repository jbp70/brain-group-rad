#!/bin/bash
# Random-weight cascade sources at non-default widths for the m11 memory job (fp32-decoder-width,
# FEAT-002; Jay's decision: no random-trainable-stage cascade flag). Not a SLURM job of its own (no
# SBATCH header): slurm/m11_fp32_width_memory.sh sources this file and calls m11_build_sources
# inside its allocation. Account qtim only; nothing here submits anything.
#
# Why: --cascade-from needs an identical ModelConfig (filters, decoder_widths, ...) in the source
# and the trainable stage. For a width W in d12h / d12hs / fr50 we therefore build, per width,
#   l4_W = level-4 main.py run from scratch, W widths, deform decoder built;
#   l2_W = level-2 main.py run with --cascade-from l4_W/ckpt/last (normal copy path), so its
#          checkpoint holds the frozen level-4 stage ('frozen_stage/') and config.json the
#          cascade record that a level-1 cascade needs.
# Each run trains ONE rigid step (--steps 1 --ckpt-every 0 -> one final ckpt/last save,
# --val-every 0 -> no validation pass); the weights are meaningless (memory / time measurements
# only). The level-1 grid configs then use --cascade-from l2_W/ckpt/last unchanged, and the
# frozen level-4 / level-2 stages are narrow too, as in the planned from-scratch retrain.
# Default-width grid lines keep the real m7_three_task_lncc / m8_level2_gradnorm sources.
#
# Not part of the ModelConfig (so they need not match the grid lines): task ratio, AMP,
# --deform-decoder-dtype, --gradient-checkpointing, --lean-sampler, --split-group-conv, group
# sizes, seed. Frozen stages are always rebuilt in float32 from the source config.json. The
# sources run rigid-only in float32 with the deform decoder built (--deform-decoder true), N 4.
#
# Width flag lines (models/width_presets.py docstring; no presets / width flags added):
#   default  (none)
#   d12h     --decoder-widths 128,96,64,16,8
#   d12hs    --filters 16,64,128,192,256,384 --decoder-widths 128,96,64,16,8
#   fr50     --filters 16,32,128,192,256,384 --decoder-widths 128,96,64,16,8
#
# Interface (sourced):  source slurm/m11_random_sources.sh
#   m11_width_args W         -> prints W's width flags (default: nothing); exit 2 if unknown
#   m11_source_cmd LEVEL W   -> sets the array M11_CMD=(python main.py ...) of source l<LEVEL>_W
#   m11_source_ok DIR        -> 0 iff DIR/config.json and DIR/ckpt/last/checkpoint exist
#   m11_build_sources        -> builds the sources of $SOURCE_WIDTHS (resumable: present sources
#                               are kept); BGR_DRY_RUN=1 prints 'SOURCE <name>: python main.py ...'
#                               lines and touches nothing. Returns 2 on an unknown width, else 0
#                               (failed builds are results): the caller checks
#                               m11_source_ok "$SOURCES_DIR/l2_W" per width.
#   Globals read: DATA, SIF, SOURCES_DIR, SOURCE_WIDTHS (default "d12h d12hs fr50"),
#   SOURCE_L4_ONLY (widths of SOURCE_WIDTHS for which only l4_W is built, e.g. when only level-2
#   grid lines need them; default empty), SOURCE_N (4), SOURCE_TIMEOUT (1200 s per run),
#   SLURM_CPUS_PER_TASK (3), BGR_DRY_RUN.
#   Status rows (tab-separated, header written once) in $SOURCES_DIR/status.tsv:
#     name  level  widths  rc  ok  elapsed_s  log
#   rc -2 = not run because its level-4 source is missing; ok = 1 iff m11_source_ok after the run.
# Standalone (tests / manual use): bash slurm/m11_random_sources.sh with the same env variables
# (PROJ, REPO, DATA, SIF, SOURCES_DIR default $PROJ/scratch/m11/<job>/sources); exit 0 iff every
# requested width has its level-2 source (level-4 source for SOURCE_L4_ONLY widths), 2 on invalid
# settings.
# Time estimate (RTX 8000, not measured yet): level-4 source ~2-3 min (DALI + validation sets +
# model build + 1 step), level-2 source ~4-6 min (level-2 validation sets + frozen stage) ->
# ~6-9 min per width, ~20-25 min for the three widths.

M11_SOURCE_WIDTHS_ALL="d12h d12hs fr50"

m11_width_args() {
    case "$1" in
        default) ;;
        d12h) echo "--decoder-widths 128,96,64,16,8" ;;
        d12hs) echo "--filters 16,64,128,192,256,384 --decoder-widths 128,96,64,16,8" ;;
        fr50) echo "--filters 16,32,128,192,256,384 --decoder-widths 128,96,64,16,8" ;;
        *) echo "ERROR: unknown width '$1' (default, d12h, d12hs, fr50)" >&2; return 2 ;;
    esac
}

m11_source_ok() {
    [ -f "$1/config.json" ] && [ -f "$1/ckpt/last/checkpoint" ]
}

m11_source_cmd() {   # m11_source_cmd <4|2> <width>
    local level="$1" width="$2" wargs
    wargs="$(m11_width_args "$width")" || return 2
    # shellcheck disable=SC2206
    M11_CMD=(python main.py --data "$DATA" --results "$SOURCES_DIR" --run-name "l${level}_$width"
             --resume false --level "$level" --task-ratio rigid:1 --deform-decoder true
             --group-size "${SOURCE_N:-4}" --steps 1 --warmup-steps 0 --val-every 0 --ckpt-every 0
             --val-png false --log-every 1 --log-backend jsonl
             --num-threads "${SLURM_CPUS_PER_TASK:-3}" --prefetch 1 $wargs)
    if [ "$level" = 2 ]; then
        M11_CMD+=(--cascade-from "$SOURCES_DIR/l4_$width/ckpt/last")
    elif [ "$level" != 4 ]; then
        echo "ERROR: source level must be 4 or 2, got '$level'" >&2
        return 2
    fi
}

m11_build_sources() {
    local widths="${SOURCE_WIDTHS:-$M11_SOURCE_WIDTHS_ALL}" w level levels name dir t0 rc ok status
    status="$SOURCES_DIR/status.tsv"
    if [ -z "${BGR_DRY_RUN:-}" ]; then
        mkdir -p "$SOURCES_DIR"
        [ -f "$status" ] || printf 'name\tlevel\twidths\trc\tok\telapsed_s\tlog\n' > "$status"
    fi
    for w in $widths; do
        case " ${SOURCE_L4_ONLY:-} " in *" $w "*) levels="4" ;; *) levels="4 2" ;; esac
        for level in $levels; do
            name="l${level}_$w"
            dir="$SOURCES_DIR/$name"
            m11_source_cmd "$level" "$w" || return 2
            if [ -n "${BGR_DRY_RUN:-}" ]; then
                echo "SOURCE $name: ${M11_CMD[*]}"
                continue
            fi
            if m11_source_ok "$dir"; then
                echo "=== source $name present, kept"
                continue
            fi
            if [ "$level" = 2 ] && ! m11_source_ok "$SOURCES_DIR/l4_$w"; then
                echo "=== source $name not built: level-4 source l4_$w missing"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$level" "$w" -2 0 0 "$SOURCES_DIR/$name.log" >> "$status"
                continue
            fi
            rm -rf "$dir"                       # partial run from an earlier attempt
            t0=$SECONDS
            echo "=== source $name start $(date -Is)"
            timeout "${SOURCE_TIMEOUT:-1200}" singularity exec --nv -B /scratch -B /autofs -B /cluster \
                -B /space -B /vast "$SIF" "${M11_CMD[@]}" > "$SOURCES_DIR/$name.log" 2>&1 </dev/null
            rc=$?
            ok=0
            if m11_source_ok "$dir"; then ok=1; fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$level" "$w" "$rc" "$ok" "$((SECONDS - t0))" \
                "$SOURCES_DIR/$name.log" >> "$status"
            echo "=== source $name rc $rc ok $ok $((SECONDS - t0)) s"
            if [ "$ok" != 1 ]; then tail -n 5 "$SOURCES_DIR/$name.log" || true; fi
        done
    done
    return 0
}

# Standalone run (not when sourced).
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    set -uo pipefail
    PROJ="${PROJ:-/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg}"
    REPO="${REPO:-$PROJ/brain-group-rad}"
    DATA="${DATA:-$PROJ/data/npy_v2}"
    SIF="${SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}"
    JOB_ID="${SLURM_JOB_ID:-manual}"
    SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"
    SOURCES_DIR="${SOURCES_DIR:-$PROJ/scratch/m11/$JOB_ID/sources}"
    SOURCE_WIDTHS="${SOURCE_WIDTHS:-$M11_SOURCE_WIDTHS_ALL}"
    SOURCE_N="${SOURCE_N:-4}"
    SOURCE_TIMEOUT="${SOURCE_TIMEOUT:-1200}"
    for w in $SOURCE_WIDTHS; do
        case "$w" in
            d12h|d12hs|fr50) ;;
            *) echo "ERROR: SOURCE_WIDTHS entries must be d12h, d12hs or fr50, got '$w'" >&2; exit 2 ;;
        esac
    done
    case "$SOURCE_N" in ''|*[!0-9]*|0|1) echo "ERROR: SOURCE_N must be an integer >= 2, got '$SOURCE_N'" >&2; exit 2;; esac
    case "$SOURCE_TIMEOUT" in ''|*[!0-9]*|0) echo "ERROR: SOURCE_TIMEOUT must be an integer >= 1, got '$SOURCE_TIMEOUT'" >&2; exit 2;; esac
    if [ -n "${BGR_DRY_RUN:-}" ]; then
        m11_build_sources
        exit 0
    fi
    mkdir -p "$PROJ/scratch/keras" "$PROJ/scratch/cuda_cache"
    export TMPDIR="$SCRATCH_ROOT/$JOB_ID"
    mkdir -p "$TMPDIR"
    export SINGULARITYENV_TMPDIR="$TMPDIR"
    export SINGULARITYENV_KERAS_HOME="$PROJ/scratch/keras"
    export SINGULARITYENV_CUDA_CACHE_PATH="$PROJ/scratch/cuda_cache"
    export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
    cd "$REPO" || exit 2
    m11_build_sources
    rc=0
    for w in $SOURCE_WIDTHS; do
        case " ${SOURCE_L4_ONLY:-} " in *" $w "*) top="l4_$w" ;; *) top="l2_$w" ;; esac
        m11_source_ok "$SOURCES_DIR/$top" || { echo "source $top missing"; rc=1; }
    done
    exit "$rc"
fi
