#!/usr/bin/env bash
# Run tests/run_tests_wsl.sh and record the full output in this folder's test-log.txt.
REPO=/mnt/d/Research/Brain_Registration_Projects/brain-group-rad
LOG="$REPO/.agents/tasks/milestone-4/test-log.txt"
cd "$REPO"
{
  echo "# Milestone 4 test log. Command: wsl -d Ubuntu bash .agents/tasks/milestone-4/record_tests.sh"
  echo "# (runs tests/run_tests_wsl.sh). Started: $(date -Is)  host: $(hostname)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
  echo "# Newest source edit before this run:"
  find models data_loading runtime tests slurm main.py -type f \( -name "*.py" -o -name "*.sh" \) -printf "%TY-%Tm-%Td %TH:%TM:%TS %p\n" | sort | tail -n 3
  bash tests/run_tests_wsl.sh
  echo "# runner exit status: $?"
  echo "# Finished: $(date -Is)"
} > "$LOG" 2>&1
grep -E "^(Ran|OK|FAILED|# runner|# Finished|py_compile|ast |bash -n|main.py|runtime\.args)" "$LOG"
grep -A 18 "overfit curve" "$LOG"
grep -A 8 "overfit val" "$LOG"
grep "amp overflow" "$LOG"
