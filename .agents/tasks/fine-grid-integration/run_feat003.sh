#!/usr/bin/env bash
# FEAT-003 checks (bgr_preproc38, TF-free): tests.test_slurm_scripts, bash -n of the four scripts,
# and the unset-knob dry-run MAIN / CHECK lines of r_level2.sh / r_level1.sh vs the base commit copies
# in base/ (written from `git show ee8a88a:slurm/<script>` on the Windows side).
set -o pipefail
W=/mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.worktrees/fine-grid-integration
T=$W/.agents/tasks/fine-grid-integration
cd "$W" || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bgr_preproc38
export PYTHONDONTWRITEBYTECODE=1
python --version
echo "=== bash -n"
for s in slurm/fgi_memory.sh slurm/r_level2.sh slurm/r_level1.sh slurm/r_cascade.sh; do
    bash -n "$s" && echo "bash -n $s OK" || echo "bash -n $s FAILED"
done
echo "=== base comparison (unset FROZEN_INTEGRATION_GRID)"
tmp=$(mktemp -d)
mkdir -p "$tmp/ckpt"
for s in r_level2.sh r_level1.sh; do
    for v in base new; do
        if [ $v = base ]; then p=$T/base/$s; else p=slurm/$s; fi
        env -i PATH="$PATH" HOME="$HOME" PROJ="$tmp" RESULTS="$tmp/results" CASCADE_FROM="$tmp/ckpt" \
            EXTRA_ARGS="--group-config B --lambda-smooth 10" BGR_DRY_RUN=1 bash "$p" > "$tmp/$s.$v" 2>&1
        echo "$s $v rc $?"
    done
    if cmp -s "$tmp/$s.base" "$tmp/$s.new"; then echo "$s dry run IDENTICAL to base"; else echo "$s dry run DIFFERS"; diff "$tmp/$s.base" "$tmp/$s.new"; fi
done
rm -rf "$tmp"
echo "=== tests.test_slurm_scripts"
python -c "
import sys, unittest
suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_slurm_scripts')
r = unittest.TextTestRunner(verbosity=2).run(suite)
assert 'tensorflow' not in sys.modules, 'tensorflow imported'
print('tensorflow not imported')
sys.exit(0 if r.wasSuccessful() else 1)
" > "$T/logs/feat003_slurm.log" 2>&1
rc=$?
tail -n 8 "$T/logs/feat003_slurm.log"
grep -E "FAIL|ERROR" "$T/logs/feat003_slurm.log" | head -n 30
echo "test_slurm_scripts rc $rc"
