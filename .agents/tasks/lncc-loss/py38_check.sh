#!/usr/bin/env bash
# LNCC loss: Python 3.8 compile / ast check of the changed .py files, main.py --help (TF-free)
# and bash -n of the changed shell scripts (bgr_preproc38).
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bgr_preproc38
export PYTHONDONTWRITEBYTECODE=1
F="data_loading/params.py runtime/losses.py runtime/run.py runtime/args.py runtime/metrics.py main.py
tests/gpu/measure_deform_memory.py tests/test_trainer.py tests/test_runtime_losses_metrics.py
tests/test_runtime_infra.py tests/test_main.py .agents/tasks/lncc-loss/grad_probe.py
.agents/tasks/lncc-loss/overfit_table.py"
status=0
python --version
# shellcheck disable=SC2086
python -m py_compile $F && echo "py_compile OK" || status=1
# shellcheck disable=SC2086
python - $F <<'EOF' || status=1
import ast, sys
for f in sys.argv[1:]:
    ast.parse(open(f).read(), f, feature_version=(3, 8))
print("ast feature_version=(3, 8) OK (%d files)" % (len(sys.argv) - 1))
EOF
python -c "import runtime.args, sys; assert 'tensorflow' not in sys.modules; print('runtime.args imports without TF')" || status=1
python main.py --help > /tmp/lncc_help.txt && echo "main.py --help OK" || status=1
for flag in --lambda-ncc-template --lambda-ncc-pair --ncc-window --lambda-dice-template; do
    grep -q -- "$flag" /tmp/lncc_help.txt && echo "help has $flag" || { echo "help MISSING $flag"; status=1; }
done
for s in slurm/m7_three_task.sh tests/run_tests_wsl.sh .agents/tasks/lncc-loss/run_some.sh \
         .agents/tasks/lncc-loss/grad_probe.sh .agents/tasks/lncc-loss/overfit_compare.sh \
         .agents/tasks/lncc-loss/measure_memory.sh; do
    bash -n "$s" && echo "bash -n $s OK" || status=1
done
echo "status $status"
exit $status
