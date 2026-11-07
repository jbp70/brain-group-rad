#!/usr/bin/env bash
# FEAT-002 verification commands, run one after another (logs in .agents/tasks/tta/_f2_v<k>.log).
H=/mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.agents/tasks/tta
cd "$H" || exit 1
bash run_tests.sh tf tests.test_predict_tta_golden tests.test_predict_streaming > _f2_v1.log 2>&1; echo "v1 rc $?" >> _f2_status.log
bash run_tests.sh tf tests.test_predict_tta tests.test_tta > _f2_v2.log 2>&1; echo "v2 rc $?" >> _f2_status.log
bash run_tests.sh tf tests.test_predict_runner tests.test_predict_levels_runner tests.test_predict > _f2_v3.log 2>&1; echo "v3 rc $?" >> _f2_status.log
bash run_tests.sh py38 tests.test_predict_store tests.test_predict_levels > _f2_v4.log 2>&1; echo "v4 rc $?" >> _f2_status.log
bash run_tests.sh compile > _f2_v5.log 2>&1; echo "v5 rc $?" >> _f2_status.log
echo "done $(date -Is)" >> _f2_status.log
