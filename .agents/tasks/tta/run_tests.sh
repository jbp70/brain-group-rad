#!/usr/bin/env bash
# Local test helper of the TTA implementation (plan D15). Avoids PowerShell -> WSL quoting.
#
#   wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.agents/tasks/tta/run_tests.sh tf tests.test_tta
#   ... run_tests.sh py38 tests.test_predict_store      (TF-free; asserts TF was not imported)
#   ... run_tests.sh compile                            (py_compile + ast feature_version 3.8)
#   ... run_tests.sh bashn slurm/a.sh slurm/b.sh        (bash -n)
#   ... run_tests.sh script path/to/file.py [args]      (python file in tf_gpu with the TF env)
#
# BGR_WRITE_GOLDEN / BGR_DEFER_MEMORY_TESTS are inherited when exported by the caller, e.g.
#   wsl -d Ubuntu bash -c "BGR_WRITE_GOLDEN=1 bash /mnt/d/.../run_tests.sh tf tests.test_predict_tta_golden"
REPO=/mnt/d/Research/Brain_Registration_Projects/brain-group-rad
cd "$REPO" || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
set +u
export PYTHONDONTWRITEBYTECODE=1

_tf_env() {
    conda activate tf_gpu
    export TF_CPP_MIN_LOG_LEVEL=3 TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 BGR_TF_GPU_LIMIT_MB=6000
    export TF_CUDNN_USE_AUTOTUNE=0
}

mode="$1"
shift
case "$mode" in
    tf)
        _tf_env
        echo "[run_tests] tf: $* ($(date -Is))"
        python -m unittest "$@" -v 2>&1 | grep -v "cuBLAS factory"
        exit "${PIPESTATUS[0]}"
        ;;
    py38)
        conda activate bgr_preproc38
        echo "[run_tests] py38: $* ($(date -Is))"
        python - "$@" <<'EOF'
import sys, unittest
names = sys.argv[1:]
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
assert "tensorflow" not in sys.modules, "tensorflow was imported by the TF-free tests"
print("TF-free tests OK without TF (%d tests)" % result.testsRun if result.wasSuccessful() else "TF-free tests FAILED")
sys.exit(0 if result.wasSuccessful() else 1)
EOF
        exit $?
        ;;
    compile)
        conda activate bgr_preproc38
        files=""
        for f in inference/*.py evaluation/*.py scripts/compare_tta.py predict.py tests/test_tta*.py \
                 tests/test_predict_tta*.py tests/test_linear_eval.py tests/test_compare_tta.py; do
            [ -f "$f" ] && files="$files $f"
        done
        echo "[run_tests] compile ($(date -Is)):$files"
        python -m py_compile $files || exit 1
        echo "py_compile OK"
        python - $files <<'EOF' || exit 1
import ast, sys
for f in sorted(sys.argv[1:]):
    ast.parse(open(f).read(), f, feature_version=(3, 8))
print("ast feature_version=(3, 8) OK (%d files)" % len(sys.argv[1:]))
EOF
        python -c "import inference.tta, sys; assert 'tensorflow' not in sys.modules; print('inference.tta imports without TF')" 2>/dev/null \
            || { [ -f inference/tta.py ] && { echo "inference.tta imported TF (or failed)"; exit 1; }; }
        exit 0
        ;;
    bashn)
        rc=0
        for s in "$@"; do
            bash -n "$s" && echo "bash -n $s OK" || rc=1
        done
        exit $rc
        ;;
    script)
        _tf_env
        echo "[run_tests] script: $* ($(date -Is))"
        python "$@" 2>&1 | grep -v "cuBLAS factory"
        exit "${PIPESTATUS[0]}"
        ;;
    *)
        echo "usage: run_tests.sh tf|py38|compile|bashn|script ..." >&2
        exit 2
        ;;
esac
