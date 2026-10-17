#!/usr/bin/env bash
# Run tests/run_tests_wsl.sh and record the full output in this folder's test-log.txt.
REPO=/mnt/d/Research/Brain_Registration_Projects/brain-group-rad
LOG="$REPO/.agents/tasks/milestone-3/test-log.txt"
cd "$REPO"
{
  echo "# Milestone 3 test log. Command: wsl -d Ubuntu bash tests/run_tests_wsl.sh"
  echo "# Started: $(date -Is)  host: $(hostname)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
  bash tests/run_tests_wsl.sh
  echo "# runner exit status: $?"
  echo "# Finished: $(date -Is)"
} > "$LOG" 2>&1
tail -n 30 "$LOG"
