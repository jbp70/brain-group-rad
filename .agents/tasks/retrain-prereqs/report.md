# Retrain-from-scratch prerequisites: report (draft)

Workflow `wf_63bfb7f7f2cfe710`. Binding design: `.agents/tasks/retrain-prereqs/design.md` v1.4 (section 16 addendum wins over sections 6 / 14). Parent design bumped to `.agents/tasks/network-design/design.md` v3.16. Nothing was committed, pushed or submitted by the workflow steps; no cluster access.

## What was built

| Deliverable | Flags / files | Default |
|---|---|---|
| AdamW (FEAT-001) | `--optimizer adamw` (Keras 2.10 `experimental.AdamW`, decoupled lr-scaled decay, gamma/beta/bias excluded); `runtime/optimizers.py` decision docstring + `WEIGHT_DECAY_OPTIMIZERS`; `--weight-decay` finite and >= 0; startup line / `NOTE: --weight-decay is ignored by --optimizer adam`; config.json `optimizer` record (sgdw / adamw only); resume guard `_check_optimizer` (different optimizer -> SystemExit, changed lr / wd -> event `optimizer_hparam_changed_on_resume`) | `adam`, unchanged |
| Shift-free low-res (FEAT-002) | `--lowres-method nn\|tent` (TF device only; `tent` + DALI device -> parser error); `IntensityAugConfig.low_res_method`; numpy loop reference `intensity_ref.lowres_tent_np` | `nn`, bit-identical |
| Per-task, per-stage skip (FEAT-003) | `--cascade-skip-linear SPEC`, `--cascade-skip-deform SPEC` (`level:prob`), stage mode in the Trainer (`[n_stages]` bool input), `FrozenChain.linear_skip` / `deform_skip`, logs `train_<task>/cascade_skip_l<lv>`, config.json `cascade.skip_mode` / `skip_probs` / `skip_consistency_warnings`, level-1 consistency WARNING, checker stage-mode check | off, no skip op built |
| Delayed task start (FEAT-004) | `--task-start-step task:step`, `--task-start-ramp-steps W`; `TaskSchedule` renormalises over active tasks; lr x (s - start + 1) / W on the delayed task's ramp steps | all tasks at 0, identical draws |
| Training pair cap (FEAT-007) | `--max-pairs K`: k = min(K, max(floor, round(p N (N - 1)))) for the linear pairwise loss and the deform pairwise Dice + LNCC (cap applied after `--min-pairs`; a cap landing on k == N keeps the N-cycle); `train_<task>/n_pairs` logged with `--min-pairs n` or K > 0; NOTE lines for K < N with `--min-pairs n` and for `--pair-recompute true` at level 4 / 2 (help: level-1 option); mirrors in `models/memory_model.num_pairs` and `check_mixed_run.expected_pairs` (reads config.json `args.max_pairs`); `measure_level_memory.py --max-pairs` (records `max_pairs`, `n_pairs`) and `--int32-check true`; `slurm/m10_batch_memory_capped.sh` | 0 = no cap, no new op |
| Scripts and helpers (FEAT-005) | `slurm/r_level4.sh` (one run, deform from 3000, two-stage fallback in the header), `slurm/r_level2.sh`, `slurm/r_level1.sh` (skip schedules 4:0 / 4:0.15 and 4:0,2:0.15 / 4:0.15,2:0.15), `slurm/sweep_level4.py` (phases A lr x wd, B group sizes, C length; <= 2 active jobs, qtim only, dirty-tree refusal, failed incl. "died before start" never resubmitted), `scripts/summarize_runs.py` (ranking, `length_decision`), `tests/test_sweep_tools.py`, `tests/run_tests_wsl.sh` hooks | new files; m7 / m8 / m9 scripts unchanged |

Script defaults per design section 16: `MAX_PAIRS=64` (-> `--max-pairs 64`) in all three, `PAIR_RECOMPUTE=false` at levels 4 / 2, `true` at level 1.

## Local verification evidence

All in WSL, GTX 1080 under `BGR_TF_GPU_LIMIT_MB=6000`, TINY / narrow widths; syntax gate in `bgr_preproc38` (Python 3.8.20).

- FEAT-001: `tests.test_runtime_optim` 16 OK (new `TestAdamW` a-e: AdamW wd 0 == Adam, variables and slots max |diff| 0; decay only on kernels; decay magnitude; AMP LSO; checkpoint round trip max |diff| 0); `tests.test_main` 28 OK (incl. `TestOptimizerMain`). +9 tests.
- FEAT-002: `TestTentLowres` 5 OK; `tests.test_tf_intensity_aug` + `TestTFIntensityAug` 23 OK; `tests.test_main` OK. `nn` path max |diff| 0 vs before for every ROI; tent TF vs numpy loop max |diff| 1.8e-7; centroid gate worst interior 0.0167 voxel (< 0.02), edge slab 0.0625 (< 0.07), `nn` worst 1.55 voxel.
- FEAT-003: `tests.test_cascade` + `tests.test_check_mixed_run` 74 OK (unmodified `TestSkip` 4/4); `tests.test_trainer` + `tests.test_main` 90 OK. Stage vs legacy level-2 skip max |diff| 0 (6 losses, 94 variable sums); `skip4=None` == `linear_stages` / `deform_stages` + `skip_to` exactly; skip both == identity exactly; draw rates within 0.01 of p over 20000 steps.
- FEAT-004: targeted 7 OK; `tests.test_trainer` + `tests.test_main` 96 OK; `TestSkip` 4 OK. `TaskSchedule` without starts reproduces the old draws for steps 0..9999 exactly; ramp lr == lr_at(s) x (s - start + 1) / W exactly.
- FEAT-005: `tests.test_sweep_tools` + `tests.test_slurm_scripts` 58 OK (at the time 1 skipped, `TestMaxPairsFlag`; it runs since FEAT-007); `bash -n` on the three scripts OK; helpers parse with `ast` feature_version (3, 6); importing `tests.test_sweep_tools` loads no TensorFlow; every dry-run main.py line is accepted by `parse_args` (incl. `--max-pairs 64` since FEAT-007).
- FEAT-007 (implemented in the integration step; it was still pending while the scripts already passed `--max-pairs`): `feat007_targeted.log` (`test_postprocess`, `test_check_mixed_run`, `test_pair_recompute`, `TestMinPairsTrainer`, `TestMaxPairsArgs`, goldens incl. `test_cascade.TestSkip`) 84 tests, 83 OK; the one failure, `TestPairRecomputeMemory.test_peak_1_vs_4_pairs` (recompute growth 128 MiB vs bound 95 MiB), depends on allocator state from earlier tests in the same process and passes alone (96 s); it exercises `runtime/losses.py`, which no FEAT changed. TF-free `test_memory_model` + `test_sweep_tools` + `test_slurm_scripts` 96 OK, 0 skips (`TestMaxPairsFlag` now runs; the dry-run lines parse with `--max-pairs` kept). `max_pairs` 0: same pairs and generator state as before (bit-exact); counts k = min(K, ...) for N 4-64, K 0/16/64, both floors; postprocess, memory model and checker mirrors agree for N 2-40, 48, 64, 96.
- Goldens (`TestLinearOnlyUnchanged`, `TestDeformUnchanged`) after every FEAT: max |diff| 0, fixtures unmodified.
- Level-4 end-to-end (`e2e_level4.txt`): `main.py --level 4 --overfit-synthetic`, three tasks, AdamW wd 1e-2, AMP, `--task-start-step deform:6 --task-start-ramp-steps 4`, 20 steps: first deform step s = 6, none before; all logged losses finite; 2 early affine steps skipped as AMP loss-scale overflow (as on the cluster); logged lr == schedule x ramp on 20/20 steps; config.json optimizer record 18 decayed / 50 excluded variables; exit 1 only because the overfit threshold 0 cannot be reached (accepted).

## Full suite

`tests/run_tests_wsl.sh`, final run (log `full_suite.log`): **647 tests, 0 failures, 0 errors, 4 skips** (the same 4 as the 558-test baseline: 3 DALI-only tests and the dllogger backend), suite status 0. Stage 1: py_compile + `ast` (3, 8) on 84 files, `ast` (3, 6) on the 2 login-node helpers, `bash -n` on 15 scripts (now incl. both m10 scripts), 109 TF-free tests OK without TF. Stage 2 wall time 12918 s (3.6 h). Goldens: `[golden] max |diff| 0`, `[golden deform] max |diff| 0`, `test_cascade.TestSkip` 4/4 OK; `tests/fixtures/` unmodified.

Seams fixed by the integration step:
- `--max-pairs` (FEAT-007) was not implemented, so every r_level script line would have been rejected by main.py; implemented here (see the FEAT-007 bullet above).
- Run 1 (one `discover` process, `full_suite_run1.log`): 644 tests, 1 failure + 1 error, both GPU-memory tests late in the process: `test_val_chunk.TestValidationMemory.test_peaks` OOM at the 6000 MB limit and `test_pair_recompute.TestPairRecomputeMemory.test_peak_1_vs_4_pairs` (growth bound). Both pass in a fresh process (pair recompute 3/3, `fresh_memory_tests_run1.log`). Fix: `tests/tf_setup.py` `DEFER_MEMORY_TESTS` (`BGR_DEFER_MEMORY_TESTS=1`) skips these two classes inside their module; the runner runs them last, each in its own process.
- Run 2 (still one `discover` process, `full_suite_run2.log`): 647 tests, 2 errors, OOM on a 30 MB tensor in `test_group_net` (`TestDeformGradients`, `TestGradientCheckpointing`), which passed in run 1. GPU memory held over from earlier modules fills the 6 GB limit once the suite has ~640 tests (558 before this work). Fix: `tests/run_tests_wsl.sh` runs every test module in its own process and sums the counts (`--- <module>: exit code, N tests, K skipped` lines + one `unittest summary` line); extra cost ~0 (stage 2 took 12918 s vs 13165 s in run 2). The leak itself was not traced; on the cluster the GPU tests are not run this way.
- `measure_level_memory.py --int32-check` initially included the build tensors and flagged level-1 N 2; it now checks the training step and the frozen / validation forwards at N. With that rule the capped job expects 'limit' records for L4 deform N 96, L4 linear N 160 / 192 and L2 linear N 40 (the same rule matches the uncapped L2 deform N 20 and L2 linear N 48 aborts); N 144 (L4 linear) and N 36 (L2 linear) were added to `slurm/m10_batch_memory_capped.sh` as the largest N below the limit.

## Code-review fixes (review 2026-10-27, 1 MEDIUM + 4 NIT)

- MEDIUM, `scripts/summarize_runs.py`: `exclusion()` now returns `incomplete (last validation <s> of <steps>)` when the last record holding `val/mean_pair_rms_mm` is not at config.json `args.steps` (same rule as `sweep_level4.run_progress`; unknown steps -> `of ?`). Such runs stay in the table (X column) but are not ranked, and `--length-curve` prints `LENGTH UNDECIDED (run <name> excluded: incomplete ...)`. New tests: `TestSummarize.test_incomplete_run_not_ranked` and an incomplete 16000-step point in `test_length_curve_cli`.
- NIT, `runtime/optimizers.py` docstring and design sections 1 / 2.3 / F7: fp32 = the Trainer's finite cond; AMP = the LossScaleOptimizer's own check (the Trainer gates only on input finiteness); the decay is skipped with the step.
- NIT, `slurm/m10_batch_memory_capped.sh`: both linear steps pass `--min-pairs n`, so the records carry the r_level4 / r_level2 knob set (pair count unchanged under the cap).
- NIT, `slurm/r_level4.sh` smoke line (and hand-off step 4 / design): `DEFORM_RAMP=100`, so the deform LR ramp ends at step 200 and the post-ramp path runs.
- NIT, `slurm/sweep_level4.py`: docstring and `--lrs` / `--wds` help say to pass the same grid options in every phase (B / C indices start after the phase-A grid); indices unchanged.
- Verification: `ast` (3, 6) on both helpers, (3, 8) on the edited Python files; `bash -n` on both edited scripts (LF only); `tests.test_sweep_tools` + `tests.test_slurm_scripts` 60 OK; `tests.test_runtime_optim` 16 OK. No trainer / model code changed (docstring only), so the full suite was not re-run.

## Not verified locally

- RTX 8000 / A100 memory and step times at the new N; the m10 placeholders `N` / `DEFORM_N` (L4 64 / 32, L2 32 / 16) are unmeasured with `--max-pairs 64`.
- Login-node `python3` version (only `ast` feature_version (3, 6) + token checks).
- Real-data behaviour of AdamW at wd 1e-2 / 1e-1, of the coincident step-3000 change point (curriculum end, G start, deform start), and of AMP from scratch at level 4 (m8 / m9 AMP runs were warm-started).
- Real sbatch / squeue behaviour of the sweep helper (stubs only).
- `--max-pairs` was exercised locally only at TINY / narrow widths (N <= 8 in the Trainer, N <= 96 in the pair-count functions); `slurm/m10_batch_memory_capped.sh` has only passed `bash -n` and a static header test. Its `--int32-check` uses the memory model's tensor inventory, which may miss tensors that the model does not list.

## Hand-off (orchestrator)

1. After the full suite: commit, push, `git pull` in the cluster clone (not while sweep jobs are pending). Files of this work: the modified files in `git status` plus new `slurm/r_level{4,2,1}.sh`, `slurm/sweep_level4.py`, `slurm/m10_batch_memory_capped.sh`, `scripts/summarize_runs.py`, `tests/test_sweep_tools.py` (the 1-line pre-existing change in `.agents/tasks/network-design/design.md` is folded into the v3.16 edit).
2. With Jay's approval and < 2 active jobs: `sbatch slurm/m10_batch_memory_capped.sh` -> `scratch/m10/memory_batch_capped.json` (the uncapped `scratch/m10/memory_batch.json` is reference only).
3. Fill `N` / `DEFORM_N` in `slurm/r_level4.sh` and `slurm/r_level2.sh`: the largest measured N with `fits(record, partition) := peak_mb <= BUDGET[partition][level]` (MiB; rtx8000 L4 42000, L2 41000; dgx-a100 34000), from the record whose knob set equals the script defaults. Rewrite each `# TODO(m10 memory job)` comment to `# m10 memory job <id>: record "<name>", peak <GB> at N <n>`. Commit, push, pull.
4. Smokes (each needs Jay's approval; check `squeue -u jn85` first):
   - `RUN_NAME=r4_smoke STEPS=300 VAL_EVERY=100 WARMUP=50 DEFORM_START=100 DEFORM_RAMP=100 CURRICULUM_RAMP=100 G_START=100 G_RAMP=100 PROJECT_STEPS=16000 PROJECT_VAL_EVERY=500 sbatch --job-name=r4smoke --time=02:00:00 --mem=64G slurm/r_level4.sh`
   - level 2 (after the level-4 final): `RUN_NAME=r2_smoke CASCADE_FROM=$RESULTS/r4_final/ckpt/best STEPS=200 VAL_EVERY=100 WARMUP=50 PROJECT_STEPS=8000 PROJECT_VAL_EVERY=500 sbatch --job-name=r2smoke --time=03:00:00 --mem=64G slurm/r_level2.sh`
   - level 1 (after level 2): `RUN_NAME=r1_smoke CASCADE_FROM=$RESULTS/r2_long/ckpt/best STEPS=200 VAL_EVERY=100 WARMUP=50 PROJECT_STEPS=8000 PROJECT_VAL_EVERY=500 sbatch --job-name=r1smoke --time=04:00:00 --mem=64G slurm/r_level1.sh` (compare `INFO gpu_peak` with m9's 36.7 GB; above 39000 MB use `MIN_PAIRS=1`).
   - Smoke gate: `INFO gpu_peak` <= the budget after >= 2 validations, else the next smaller N; `--time` of the long runs = smoke `INFO step_time` projection x 1.3.
5. Sweep (Jay approves phases A-C once; run `submit` whenever a job ends; pass the filled sizes if they differ from 64 / 32):
   - `python3 slurm/sweep_level4.py list --phase A --n <N> --deform-n <DN> --n-small <N/2> --deform-n-small <DN/2>`
   - `python3 slurm/sweep_level4.py submit --phase A --max-active 2 <same size options>` (`--dry-run` first), `status` to follow.
   - `python3 scripts/summarize_runs.py /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results --glob 'r4sw_*'` -> phase B (`--best-lr X --best-wd Y`, with the same `--lrs` / `--wds` and size options as phase A: the B / C job indices start after the phase-A grid), then phase C (`--c-time` from the phase-A `INFO step_time`), then `--length-curve RUN8,RUN12,RUN16` for the final length.

## Decisions for Jay

Both were confirmed by Jay on 2026-10-26 (design section 16) and need no further answer:
- Weight-decay grid {0, 1e-2, 1e-1} instead of his {0, 1e-5, 1e-4}. Under lr-scaled AdamW the latter shrink weights by < 0.05 % over a run; his grid is one flag away (`--wds 0,1e-5,1e-4`).
- `tent` instead of `area`: box averaging + linear upsampling still shifts objects by up to ~0.06 voxel and fails the 0.02 gate; `tent` stays within 0.0167 voxel at interior positions.
