#!/usr/bin/env bash
# FEAT-003: run tests.test_slurm_scripts in a normal WSL clone of branch fine-grid-integration
# (the worktree's .git file holds a Windows path that WSL git cannot resolve; TestCascadeRuns.test_dry_run
# needs `git rev-parse HEAD` in REPO). Read-only on the main repository (clone from its .git dir).
set -o pipefail
T=/mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.worktrees/fine-grid-integration/.agents/tasks/fine-grid-integration
C=$(mktemp -d)/clone
git clone -q --branch fine-grid-integration /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.git "$C" || exit 1
cd "$C" || exit 1
echo "clone HEAD $(git rev-parse --short HEAD)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bgr_preproc38
export PYTHONDONTWRITEBYTECODE=1
python -c "
import sys, unittest
suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_slurm_scripts')
r = unittest.TextTestRunner(verbosity=2).run(suite)
assert 'tensorflow' not in sys.modules, 'tensorflow imported'
print('tensorflow not imported')
sys.exit(0 if r.wasSuccessful() else 1)
" > "$T/logs/feat003_slurm_clone.log" 2>&1
rc=$?
tail -n 6 "$T/logs/feat003_slurm_clone.log"
grep -E "^(FAIL|ERROR)" "$T/logs/feat003_slurm_clone.log" | head -n 20
grep -cE "\.\.\. ok$" "$T/logs/feat003_slurm_clone.log"
echo "test_slurm_scripts (clone) rc $rc"
rm -rf "$(dirname "$C")"
