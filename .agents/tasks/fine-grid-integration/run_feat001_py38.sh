#!/usr/bin/env bash
# FEAT-001 py3.8 syntax check + runtime.args TF-free check (bgr_preproc38).
set -o pipefail
W=/mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.worktrees/fine-grid-integration
cd "$W" || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bgr_preproc38
export PYTHONDONTWRITEBYTECODE=1
python --version
python -m py_compile models/diffeo.py models/cascade.py runtime/args.py main.py runtime/eval_run.py tests/test_fine_grid_integration.py \
  && python -c "import runtime.args, sys; assert 'tensorflow' not in sys.modules; print('args TF-free OK')"
echo "rc $?"
