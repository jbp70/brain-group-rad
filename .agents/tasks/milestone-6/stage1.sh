#!/usr/bin/env bash
# Stage 1 of tests/run_tests_wsl.sh only (py3.8 syntax, TF-free imports, main.py --help, bash -n).
REPO=/mnt/d/Research/Brain_Registration_Projects/brain-group-rad
cd "$REPO"
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONDONTWRITEBYTECODE=1
PYFILES="models/*.py data_loading/*.py runtime/*.py main.py tests/*.py tests/gpu/*.py"
status=0
conda activate bgr_preproc38
python --version
python -m py_compile $PYFILES && echo "py_compile OK" || status=1
python - $PYFILES <<'EOF' || status=1
import ast, sys
for f in sorted(sys.argv[1:]):
    ast.parse(open(f).read(), f, feature_version=(3, 8))
print("ast feature_version=(3, 8) OK (%d files)" % len(sys.argv[1:]))
EOF
python tests/np_reference.py || status=1
python -c "import data_loading.params, data_loading.dali_loader as d; print('data_loading imports without TF/DALI; DALI_AVAILABLE =', d.DALI_AVAILABLE)" || status=1
python -c "import runtime.args, runtime.schedulers, sys; assert 'tensorflow' not in sys.modules; print('runtime.args / runtime.schedulers import without TF')" || status=1
python main.py --help >/dev/null && echo "main.py --help OK (no TF)" || status=1
for s in tests/gpu/run_verify_glacier.sh tests/gpu/run_overfit_glacier.sh slurm/train_debug.sh; do
    bash -n "$s" && echo "bash -n $s OK" || status=1
done
python -m unittest tests.test_check_mixed_run 2>&1 | tail -n 3
echo "stage1 status: $status"
exit $status
