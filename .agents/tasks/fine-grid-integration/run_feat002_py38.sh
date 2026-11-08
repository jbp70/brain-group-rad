#!/usr/bin/env bash
# FEAT-002 py3.8 checks (bgr_preproc38): TF-free tests + py_compile of inference/ and predict.py.
set -o pipefail
W=/mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.worktrees/fine-grid-integration
cd "$W" || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bgr_preproc38
export PYTHONDONTWRITEBYTECODE=1
python --version
python -m unittest tests.test_predict_levels tests.test_predict_store -v 2>&1 | tail -n 5
echo "unittest rc ${PIPESTATUS[0]}"
python -m py_compile inference/*.py predict.py tests/test_fine_grid_integration.py
echo "py_compile rc $?"
