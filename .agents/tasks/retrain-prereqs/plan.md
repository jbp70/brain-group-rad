# Implementation Plan: retrain-from-scratch prerequisites

Source: approved design `.agents/tasks/retrain-prereqs/design.md` (v1.3) plus the round-3 review `design-review.md` (1 MEDIUM, 6 NIT), whose fixes are folded in below as binding requirements. Detailed per-item steps live in `task-retrain-prereqs/features/FEAT-00N.json`; shared context (commands, patterns, constraints) in `task-retrain-prereqs/context.json`.

Hard rules for every item: work in `d:\Research\Brain_Registration_Projects\brain-group-rad` with absolute paths (no worktree); no git commit/push, no ssh/sbatch; Python 3.8 syntax (helpers: 3.6 + stdlib); local TF memory limit 6000 MB (`BGR_TF_GPU_LIMIT_MB=6000`); every TF test imports `tests.tf_setup` first; new flag defaults are bit-identical (goldens `tests.test_trainer.TestLinearOnlyUnchanged`, `tests.test_trainer.TestDeformUnchanged`, and `tests.test_cascade.TestSkip` pass unmodified). Activate the `shell-and-remote-jobs` skill before composing WSL commands.

Two deviations from the step prompt wording, both decided in the approved design and both CONFIRMED by Jay on 2026-10-26 (design section 16): the low-res flag is `--lowres-method nn|tent`, not `nn|area` (box + linear failed the < 0.02 voxel gate, 0.06 voxel max; tent passes, measured 0.0167 voxel worst interior shift in FEAT-002's test 11.2 c), and the weight-decay sweep grid is {0, 1e-2, 1e-1}, replacing {0, 1e-5, 1e-4}.

Scope addition (design section 16, binding): opt-in `--max-pairs K` (item 4b, FEAT-007; default 0 = no cap, goldens unchanged; scripts default MAX_PAIRS=64), and `--pair-recompute` is a level-1-only option (scripts: PAIR_RECOMPUTE=false at levels 4 and 2; a level-4 N 48 config sat ~2 h in graph construction in job 10519650).

Test command template (WSL, tf_gpu):
`wsl -d Ubuntu bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate tf_gpu && cd /mnt/d/Research/Brain_Registration_Projects/brain-group-rad && export PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 BGR_TF_GPU_LIMIT_MB=6000 TF_CUDNN_USE_AUTOTUNE=0 && python -m unittest <modules> -v'` (TF-free tests: `conda activate bgr_preproc38`).

- [ ] 1. AdamW decision record and guards (FEAT-001).
      Docstring + `WEIGHT_DECAY_OPTIMIZERS` in optimizers; `--weight-decay` help and >= 0 check; main.py startup line, Adam NOTE, config.json `optimizer` record (sgdw/adamw only), `_check_optimizer` resume guard + `optimizer_hparam_changed_on_resume` event; tests 11.1 a-f.
      Files: runtime/optimizers.py, runtime/args.py, main.py, tests/test_runtime_optim.py, tests/test_main.py
      Verify: `python -m unittest tests.test_runtime_optim tests.test_main tests.test_trainer.TestLinearOnlyUnchanged tests.test_trainer.TestDeformUnchanged -v` OK; AdamW wd 0 vs Adam max |diff| 0.

- [x] 2. Tent low-res on the TF aug path (FEAT-002).
      `LOWRES_METHODS`, `IntensityAugConfig.low_res_method="nn"`; `_tent_matrix`, `_apply_axis_matrix`, `lowres_resample(..., method="nn")`, `augment_image` passes the method; numpy `lowres_tent_np`; `--lowres-method` with the DALI-device error; main.py description; tests 11.2 a-f (nn bit-identical, tent vs numpy < 1e-5, centroid < 0.02 voxel on the specified grid, mass/constants, static shape / one trace).
      Files: data_loading/params.py, data_loading/tf_intensity_aug.py, data_loading/intensity_ref.py, runtime/args.py, main.py, tests/test_tf_intensity_aug.py, tests/test_main.py
      Verify: `python -m unittest tests.test_tf_intensity_aug tests.test_trainer.TestTFIntensityAug tests.test_main -v` OK + goldens OK.

- [ ] 3. Per-task, per-stage cascade skip (FEAT-003).
      `--cascade-skip-linear/-deform` specs and parse rules (spec without `--cascade-from` errors first; mutually exclusive with legacy p > 0); `TrainConfig.cascade_skip_stage`, `cascade_stage_skip_draw`, Trainer skip modes off/legacy/stage with a `[n_stages]` bool input; `FrozenChain.linear_skip` / `deform_skip` (design 4.4); fit logging `train_<task>/cascade_skip_l<lv>`; main.py `skip_mode`/`skip_probs`/`skip_consistency_warnings`, `_skip_consistency`, resume refresh + event; check_mixed_run stage-mode check; tests 11.3 a-g.
      Files: runtime/args.py, runtime/run.py, models/cascade.py, main.py, tests/gpu/check_mixed_run.py, tests/test_cascade.py, tests/test_trainer.py, tests/test_main.py, tests/test_check_mixed_run.py
      Verify: `python -m unittest tests.test_cascade tests.test_check_mixed_run tests.test_trainer tests.test_main -v` OK; `TestSkip` unmodified; skip4=None chain path max |diff| 0.

- [ ] 4. Delayed task start + local level-4 e2e (FEAT-004; after items 1 and 3).
      `--task-start-step`, `--task-start-ramp-steps` (validated after `steps` is final); `TaskSchedule(start_steps=...)` with identical draws when all tasks are active; fit LR ramp; tests 11.4 a-e; local e2e `main.py --level 4 --overfit-one-group --overfit-synthetic` tiny widths, three tasks, `--optimizer adamw --weight-decay 1e-2 --task-start-step deform:6 --task-start-ramp-steps 4 --amp true`, ~20 steps, evidence in `.agents/tasks/retrain-prereqs/e2e_level4.txt`.
      Files: runtime/args.py, runtime/run.py, tests/test_trainer.py, tests/test_main.py
      Verify: `python -m unittest tests.test_trainer tests.test_main -v` OK; e2e finite, no deform step before 6.

- [ ] 4b. Training pair cap `--max-pairs K` (FEAT-007; before item 5).
      k = min(K, max(floor, round(p N (N - 1)))) applied after `--min-pairs` (N-cycle kept when k == N with `--min-pairs n`); linear pairwise loss and deform pairwise Dice + LNCC; `train_<task>/n_pairs` logged; default 0 builds no new op; mirrors in `models/memory_model.py` and `check_mixed_run.expected_pairs`; `--pair-recompute` help: level-1 only. Tests: counts with / without cap, cap + min-pairs n, CLI validation, max_pairs 0 bit-exact.
      Also: `tests/gpu/measure_level_memory.py --max-pairs K` (comma list; records max_pairs + resulting n_pairs per config) and new `slurm/m10_batch_memory_capped.sh` (as `m10_batch_memory.sh`, qtim / rtx8000 / 1 GPU / 3 CPUs / 64G, `--isolate true`, per-step timeout, output `scratch/m10/memory_batch_capped.json`; all with `--max-pairs 64`: L4 deform N 48/64/80/96 with full ckpt + lean sampler + split conv, no pair recompute, LNCC 1/1, min-pairs n, validation with val-pair-chunk 16; L4 linear N 128/160/192 after an int32 element check; L2 deform N 16; L2 linear N 32/40). The orchestrator submits it after the workflow.
      Files: data_loading/postprocess.py, runtime/run.py, runtime/args.py, main.py, models/memory_model.py, tests/gpu/check_mixed_run.py, tests/gpu/measure_level_memory.py, slurm/m10_batch_memory_capped.sh, tests/test_postprocess.py, tests/test_main.py, tests/test_check_mixed_run.py, tests/run_tests_wsl.sh
      Verify: `python -m unittest tests.test_postprocess tests.test_main tests.test_check_mixed_run -v` OK + goldens OK.

- [ ] 5. SLURM scripts, sweep helper, summarizer (FEAT-005; after items 1-4 and 4b). Section 16 overrides: MAX_PAIRS=64 in all three scripts, PAIR_RECOMPUTE=false at levels 4 and 2.
      `slurm/r_level4.sh`, `r_level2.sh`, `r_level1.sh` per design 6 (qtim only, AMP/split conv hard-coded, TF aug + tent, AdamW, gradnorm 12, skip schedule, memory knobs, BGR_DRY_RUN, checker line); `slurm/sweep_level4.py` per 7.1-7.4; `scripts/summarize_runs.py` per 7.3/8; `tests/test_sweep_tools.py` (11.5 a-h); run_tests_wsl.sh hooks. Binding review-v1.3 fixes: F1 `fits()` / BUDGET rule (MiB; rtx8000 L4 42000, L2 41000; dgx-a100 34000) in the r_level4/r_level2 headers replacing "peak <= 46000 MB"; F3 task-start flags only with deform AND a linear task (deform-only NOTE); F4 jsonl-recorded run with no dir and not in squeue = failed ("died before start"), never resubmitted; F5 `--time` from smoke step_time x 1.3 + `--c-time` help; F6 VAL_GROUP_SIZE = min(16, DEFORM_N_small).
      Files: slurm/r_level4.sh, slurm/r_level2.sh, slurm/r_level1.sh, slurm/sweep_level4.py, scripts/summarize_runs.py, tests/test_sweep_tools.py, tests/run_tests_wsl.sh
      Verify: `bash -n` on the three scripts; `python -m unittest tests.test_sweep_tools tests.test_slurm_scripts -v` OK in bgr_preproc38 with TF not imported; ast feature_version (3, 6) for the helpers.

- [ ] 6. Docs and report (FEAT-006; after items 1-5).
      network-design design.md v3.16 edits (design 9); retrain-prereqs design.md v1.4 with the review-v1.3 implementation notes and F2/F7 wording fixes; `report.md` with deliverables, evidence, unverified items, hand-off commands, open questions for Jay.
      Files: .agents/tasks/network-design/design.md, .agents/tasks/retrain-prereqs/design.md, .agents/tasks/retrain-prereqs/report.md
      Verify: `python main.py --help` (bgr_preproc38) lists every flag named in the docs.

- [ ] 7. Integration: full local suite (convergence-loop fixer, iteration 1).
      Run `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/tests/run_tests_wsl.sh` in the background (~3-3.5 h, log under `.agents/tasks/retrain-prereqs/`), fix any seam failures, fill the report's "Full suite" section (expected 558 baseline + new tests, 0 failures, 4+ skips; goldens max |diff| 0).
      Verify: suite exit status 0.
