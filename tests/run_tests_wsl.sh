#!/usr/bin/env bash
# Run the unit tests locally in WSL.
#
# Usage (from Windows PowerShell):
#   wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/tests/run_tests_wsl.sh
#
# 1. Python 3.8 syntax check of models/ and tests/ plus the TF-free numpy references, in the
#    bgr_preproc38 env (Python 3.8, numpy 1.21.1, scipy 1.4.1, no TF), as a stand-in for the
#    container's Python 3.8.10.
# 2. The full unittest suite in the tf_gpu env (TF 2.10, Python 3.9). That env crashes when the
#    GPU is hidden, so CUDA_VISIBLE_DEVICES is left alone.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/miniconda3/etc/profile.d/conda.sh

status=0
echo "=== [1/2] py3.8 syntax check + numpy references (bgr_preproc38) ==="
conda activate bgr_preproc38
python --version
python -c "import numpy, scipy; print('numpy', numpy.__version__, '| scipy', scipy.__version__)"
python -m py_compile models/*.py tests/*.py && echo "py_compile OK" || status=1
python - <<'EOF' || status=1
import ast, glob
for f in sorted(glob.glob("models/*.py") + glob.glob("tests/*.py")):
    ast.parse(open(f).read(), f, feature_version=(3, 8))
print("ast feature_version=(3, 8) OK")
EOF
python tests/np_reference.py || status=1
conda deactivate

echo "=== [2/2] unittest suite (tf_gpu) ==="
conda activate tf_gpu
export TF_CPP_MIN_LOG_LEVEL=3
python -c "import sys, tensorflow as tf, numpy, scipy; print('python', sys.version.split()[0], '| tf', tf.__version__, '| numpy', numpy.__version__, '| scipy', scipy.__version__)" 2>&1 | grep -v "cuBLAS factory"
start=$(date +%s)
python -m unittest discover -s tests -t . -v 2>&1 | grep -v "cuBLAS factory"
rc=${PIPESTATUS[0]}
echo "unittest exit code: $rc, wall time: $(( $(date +%s) - start )) s"
[ "$rc" -eq 0 ] || status=1
find "$REPO/models" "$REPO/tests" -name "__pycache__" -type d -prune -exec rm -rf {} +
exit $status
