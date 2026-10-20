#!/usr/bin/env bash
# Milestone 7 loss-balance probe (local, WSL tf_gpu, synthetic data, narrow model, N 4 / deform 2).
#
#   wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.agents/tasks/milestone-7/grad_balance_probe.sh
#
# 1. lin_src: linear-only source (rigid:1,affine:1), 300 steps, Adam 3e-4 constant (stands in for
#    the m5 checkpoint).
# 2. warm3: three-task run warm-started from lin_src (partial restore, fresh deform decoder),
#    100 steps at Adam 3e-4 with --loss-scale-mm2 100, so the deform decoder is trained: with a
#    fresh decoder the SVF head kernel is zero and the deform task sends EXACTLY zero gradient
#    into the encoder, so measuring there says nothing about the balance.
# 3. Measurement probes (lr 1e-12: the weights stay put, the logged norms are those of one fixed
#    state), 40 steps rigid:1,affine:1,deform:2, from warm3 (full restore) with --loss-scale-mm2
#    S in {1, 10, 100, 1000}; plus two comparisons at S = 1: from lin_src (fresh decoder) and
#    from scratch (no init).
# 4. summarize_grad_norms.py -> grad_balance.json (medians per run and task, eff ratio).
# Run dirs go to probe_results/ (delete afterwards, keep grad_balance.json).
set -o pipefail      # no -u: conda's activate hooks read unset variables
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
OUT="$HERE/probe_results"
cd "$REPO"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
export TF_CPP_MIN_LOG_LEVEL=3 TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 TF_CUDNN_USE_AUTOTUNE=0 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT"
NARROW="--filters 16,32,32,48,64,96 --reductions 64,48,24,24 --dense 512,256"
COMMON="--overfit-one-group --overfit-synthetic --level 4 --group-size 4 --schedule constant --warmup-steps 0 \
 --log-every 1 --results $OUT --log-backend jsonl --quiet true --gpu-memory-limit-mb 5000 --resume false $NARROW"
THREE="--task-ratio rigid:1,affine:1,deform:2 --deform-group-size 2 --seed 0"

run() {   # name, args...
    local name="$1"; shift
    echo "=== $name: $*"
    # shellcheck disable=SC2086
    python main.py $COMMON --run-name "$name" "$@" 2>&1 | grep -v "cuBLAS factory" | tail -3
    echo "rc ${PIPESTATUS[0]} (1 = overfit threshold not reached, expected)"
}

# shellcheck disable=SC2086
run lin_src --task-ratio rigid:1,affine:1 --overfit-max-steps 300 --val-every 100 --lr 3e-4 --val-png false
# shellcheck disable=SC2086
run warm3 $THREE --init-weights "$OUT/lin_src/ckpt/last" --overfit-max-steps 100 --val-every 0 --lr 3e-4 \
    --loss-scale-mm2 100 --val-png false
for S in 1 10 100 1000; do
    # shellcheck disable=SC2086
    run "probe_S$S" $THREE --init-weights "$OUT/warm3/ckpt/last" --overfit-max-steps 40 --val-every 0 \
        --lr 1e-12 --loss-scale-mm2 "$S" --val-png false
done
# shellcheck disable=SC2086
run probe_fresh_dec_S1 $THREE --init-weights "$OUT/lin_src/ckpt/last" --overfit-max-steps 40 --val-every 0 \
    --lr 1e-12 --loss-scale-mm2 1 --val-png false
# shellcheck disable=SC2086
run probe_scratch_S1 $THREE --overfit-max-steps 40 --val-every 0 --lr 1e-12 --loss-scale-mm2 1 --val-png false

conda activate bgr_preproc38
python "$HERE/summarize_grad_norms.py" "$OUT/warm3" "$OUT"/probe_S1 "$OUT"/probe_S10 "$OUT"/probe_S100 \
    "$OUT"/probe_S1000 "$OUT/probe_fresh_dec_S1" "$OUT/probe_scratch_S1" --out "$HERE/grad_balance.json"
python "$HERE/summarize_grad_norms.py" "$OUT/warm3" --min-step 50
python - "$OUT" <<'EOF'
import json, math, os, sys
out = sys.argv[1]
for name in ("lin_src", "warm3"):
    p = os.path.join(out, name, "overfit_summary.json")
    if os.path.exists(p):
        s = json.load(open(p))
        print(name, "steps", s["steps_run"], "final val", {k: round(v, 3) for k, v in (s.get("final_val") or {}).items()
              if k.endswith("center_rms_mm") or k.endswith("template_dice") or k.endswith("pair_dice")})
EOF
