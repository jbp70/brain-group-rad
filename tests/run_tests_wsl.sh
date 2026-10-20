#!/usr/bin/env bash
# Run the unit tests locally in WSL.
#
# Usage (from Windows PowerShell):
#   wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/tests/run_tests_wsl.sh
#
# 1. Python 3.8 syntax check of models/, data_loading/, runtime/, main.py, tests/ and tests/gpu/
#    plus the TF-free numpy references, in the bgr_preproc38 env (Python 3.8, numpy 1.21.1,
#    scipy 1.4.1, no TF), as a stand-in for the container's Python 3.8.10. TF-free imports of
#    runtime.args / runtime.schedulers and `main.py --help`. `bash -n` on the GPU runner scripts
#    and the SLURM template.
# 2. The full unittest suite in the tf_gpu env (TF 2.10, Python 3.9). That env crashes when the
#    GPU is hidden, so CUDA_VISIBLE_DEVICES is left alone. Neither env has nvidia.dali, so the
#    DALI-dependent tests skip. tests/gpu/ (DALI GPU verification) is not part of discovery.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONDONTWRITEBYTECODE=1

PYFILES="models/*.py data_loading/*.py runtime/*.py main.py tests/*.py tests/gpu/*.py"
status=0
echo "=== [1/2] py3.8 syntax check + numpy references (bgr_preproc38) ==="
conda activate bgr_preproc38
python --version
python -c "import numpy, scipy; print('numpy', numpy.__version__, '| scipy', scipy.__version__)"
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
for s in tests/gpu/run_verify_glacier.sh tests/gpu/run_overfit_glacier.sh slurm/train_debug.sh slurm/m7_three_task.sh; do
    bash -n "$s" && echo "bash -n $s OK" || status=1
done
conda deactivate

echo "=== [2/2] unittest suite (tf_gpu) ==="
conda activate tf_gpu
export TF_CPP_MIN_LOG_LEVEL=3
# 8 GB GTX 1080 under WSL: with memory growth, cuDNN autotune workspaces push TF to the last GB,
# where WSL pages GPU memory to host RAM and full-width steps take minutes. A 512 MB workspace cap
# keeps them at ~1 s, and a fixed 6 GB TF memory limit (tests/tf_setup.py) keeps TF out of the
# last GB (milestone-4 report). Local only; the cluster GPUs have 40-48 GB.
export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512
export BGR_TF_GPU_LIMIT_MB=6000
# cuDNN autotune off: on this GTX 1080 the first full-width backward pass of a process with
# autotune on returned 512 NaN entries in the encoder l3 conv-kernel gradient (all later calls
# and the CPU were finite; .agents/tasks/milestone-4/diag_nan_grad.py), which made
# TestGradients flaky. Local only.
export TF_CUDNN_USE_AUTOTUNE=0
python -c "import sys, tensorflow as tf, numpy, scipy; print('python', sys.version.split()[0], '| tf', tf.__version__, '| numpy', numpy.__version__, '| scipy', scipy.__version__)" 2>&1 | grep -v "cuBLAS factory"
start=$(date +%s)
python -m unittest discover -s tests -t . -v 2>&1 | grep --line-buffered -v "cuBLAS factory"
rc=${PIPESTATUS[0]}
echo "unittest exit code: $rc, wall time: $(( $(date +%s) - start )) s"
[ "$rc" -eq 0 ] || status=1
find "$REPO/models" "$REPO/data_loading" "$REPO/runtime" "$REPO/tests" -name "__pycache__" -type d -prune -exec rm -rf {} +
exit $status
