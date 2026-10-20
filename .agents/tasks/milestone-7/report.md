# Milestone 7 report: three-task training at level 4 (local implementation)

Status: implemented and verified locally. Nothing committed or pushed, no cluster access. The cluster smoke run and the long run are pending Jay's approval (commands at the end).

## Files changed

| File | Change |
|---|---|
| `runtime/checkpoint.py` | `DECODER_ONLY_FIELDS`, `SHARED_PARTS`, new `warm_start_model` (full / partial warm start, info dict); `load_model_weights` is now a wrapper with the old return value |
| `main.py` | uses `warm_start_model`; `init_weights` in config.json gains mode / restored / fresh / config diff; `init_weights` event in metrics.jsonl (after the PARAMETER line); readable stdout summary; docstring example |
| `runtime/args.py` | `--task-loss-weights` + `parse_task_loss_weights`; passed to `TrainConfig.task_loss_weights`; `--init-weights` help; docstring |
| `runtime/run.py` | `TrainConfig.task_loss_weights` (validated), `_task_weight`, objective × w in both steps (skipped at trace time when w = 1), `_enc_grad_diagnostics`, `ENC_GRAD_KEYS`, per-task `train_<task>/{diag_enc_grad_norm_raw, diag_enc_grad_norm_eff, grad_norm}` logs; docstring |
| `tests/gpu/check_mixed_run.py` | `--expect-init-weights none|full|partial`, `--expect-enc-grad true|false` (defaults keep the old checks); INFO line with the linear/deform `_eff` ratio |
| `slurm/m7_three_task.sh` | new: level-4 three-task run on rtx8000 / qtim, env overrides, runs the checker afterwards |
| `tests/run_tests_wsl.sh` | `bash -n slurm/m7_three_task.sh` added |
| `tests/test_trainer.py` | `TestLinearOnlyUnchanged` (golden), task-weight / enc-grad tests (linear and deform, clipped and unclipped), enc-grad logging in a three-task fit, different linear / deform group sizes with both best files |
| `tests/test_runtime_infra.py` | partial warm start both directions, decoder widths / integration steps, mismatch errors, full mode, fresh optimizer; `--task-loss-weights` parsing |
| `tests/test_main.py` | warm start of a deform run through main.py; validation group sizes 32 / 16 on 39 val cases; end-to-end three-task warm-start smoke |
| `tests/test_check_mixed_run.py` | the two new checks pass and fail correctly |
| `tests/fixtures/m7_linear_golden.json` | golden values of a 6-step linear-only run (written on the unmodified milestone-6 code) |
| `.agents/tasks/network-design/design.md` | v3.8 (5.2, 6.1, 6.2, 9, 11) |
| `.agents/tasks/milestone-7/` | this report, `grad_balance_probe.sh`, `summarize_grad_norms.py`, `grad_balance.json`, `probe.log`, `py38_check.sh`, `test-log.txt` |

The golden fixture lives in `tests/fixtures/`, not `tests/data/` as planned: the `.gitignore` rule `data/` ignores every `data/` folder, so the fixture would never be committed there.

## Decisions

1. **Partial warm start (D1-D3).** Configs that differ only in `deform_decoder`, `decoder_widths` or `integration_steps` restore through a temporary model built with the source config (strict restore), then encoder, linear decoder + heads and centering are copied after a count / shape / scope-local-name check of every variable. The target's deform decoder stays fresh; a source deform decoder is ignored (reverse direction). A coverage guard raises `RuntimeError` if a model part is neither restored nor the fresh decoder. Any other difference raises `ValueError` naming the fields (e.g. `in ['translation_scale'] (only [...] may differ)`). No source config or identical configs: the old strict whole-model restore (`mode: full`). The optimizer is never read: main.py builds it after the warm start, and the tests check `iterations == 0` and all-zero slots.
2. **Loss balance knobs (D4).** `--loss-scale-mm2` (existing) and new `--task-loss-weights` (multiplies each task's objective; default 1 for all). Logged losses stay unscaled and unweighted.
3. **Diagnostics (D5).** `train_<task>/diag_enc_grad_norm_raw` = encoder-only gradient norm of the unscaled task loss; `_eff` = encoder norm actually applied (after loss scale, task weight and the global-clip factor `min(1, 12 / grad_norm)`); `train_<task>/grad_norm` = the task's pre-clip total norm. They're read-only, so the applied update is unchanged.
4. **Loss-balance recommendation: keep `--loss-scale-mm2 1`, no task weights.** This was measured, see the next section. The global clip norm 12 already caps every linear step, so the applied encoder gradients of the linear tasks are only 1.25-1.66× the deform one, for any S from 1 to 1000. `slurm/m7_three_task.sh` defaults to `LOSS_SCALE_MM2=1`. The smoke run confirms this on real data (P2).
5. **Three-task recipe (D8): curriculum OFF, G at 45° / 10 mm from step 0** (`--global-start-step 0 --global-ramp-steps 0`). Reasons: the warm start comes from a model trained on the full ranges (m5's curriculum had ended at ±45° / 15 mm), and milestone 5 showed warm start + G from step 0 works (val_rigid_g 4.59 mm, no collapse). Delayed G and the curriculum exist to escape the identity fixed point from scratch, which a warm-started encoder is not at. Also warmup 500 steps (fresh Adam moments), Adam 3e-4 cosine, 12000 steps, val every 250, PNGs on.
6. **Validation group sizes (D7): no new flag.** Linear sets use min(`--group-size`, 39) = 32 (2 groups); the deform set uses min(`--deform-group-size`, 39) = 16 (3 groups); deform pairs capped at 64 per group (16 × 15 = 240 ordered pairs → 64). Tested on 39 stub val cases through `_build_dali_data`. `--val-group-size` would apply to both, so the script does not pass it. Deform validation at N = 16 runs the same forward as a deform training step without the backward pass, so it fits whenever training at N = 16 fits (33.8 GB measured).
7. **BN (D9).** Momentum stays 0.99. Expect eval-mode `val_deform` to lag `val_deform_trainmode` early on (local probe after 100 steps: eval template / pair Dice 0.872 / 0.753 vs train mode 0.947 / 0.898). Recalibration is milestone 8.

## Loss-balance measurement (P1)

Setup (`grad_balance_probe.sh`, log `probe.log`, numbers `grad_balance.json`): synthetic phantom, narrow widths, level 4, linear N 4, deform N 2, global clip 12.
- Source: 300 linear steps (rigid:1,affine:1, Adam 3e-4). Train-mode window center RMS 6.5 mm (eval 19 / 23 mm: BN eval lag at N 4 this early).
- Warm-started three-task run `warm3`: 100 steps with S = 100, so the fresh deform decoder is trained. Afterwards rigid / affine eval 5.9 / 4.1 mm, deform eval template Dice 0.872 (identity 0.835).
- Measurement probes from `warm3` (full restore), 40 steps at lr 1e-12, so the weights stay put. Medians:

| Run | S | rigid raw | rigid eff | affine raw | affine eff | deform raw = eff | rigid grad_norm | ratio eff rigid / affine vs deform |
|---|---|---|---|---|---|---|---|---|
| probe_S1 | 1 | 306 | 0.239 | 257 | 0.202 | 0.144 | 14 780 (clipped 100 %) | 1.66 / 1.40 |
| probe_S10 | 10 | 306 | 0.239 | 257 | 0.202 | 0.144 | 1 478 (100 %) | 1.66 / 1.40 |
| probe_S100 | 100 | 306 | 0.239 | 257 | 0.202 | 0.144 | 148 (100 %) | 1.66 / 1.40 |
| probe_S1000 | 1000 | 306 | 0.239 | 257 | 0.179 | 0.144 | 14.8 (rigid 91 %, affine 50 % clipped) | 1.66 / 1.25 |
| probe_fresh_dec_S1 (from the linear source, fresh decoder) | 1 | 791 | 0.405 | 523 | 0.414 | 3.8e-10 | 25 440 (100 %) | ~1e9 |
| probe_scratch_S1 (no init) | 1 | 1.9e-5 | 5.5e-8 | 2.4e-5 | 3.5e-8 | 3.6e-10 | 8 197 (100 %) | 151 / 96 |
| warm3 (training, S = 100, steps > 50) | 100 | 517 | 0.396 | 613 | 0.535 | 0.310 | 131 (100 %) | 1.27 / 1.72 |

Findings:
- In loss units the imbalance is real: the raw encoder gradient of the mm² loss is ~2000× the Dice one. But the total linear gradient norm (10³-10⁴ at S = 1) is far above the clip norm 12, so every linear step is rescaled to norm 12, and the encoder share that is applied is O(0.2-0.4). The deform steps (total norm ~0.5, never clipped) apply 0.14-0.31. The applied ratio is 1.3-1.7 at the warm-started state, inside the [0.1, 10] target, for every S in the grid.
- S has no effect while the linear steps stay clipped (S ≤ 100 here). It starts to matter only near S ≈ 1000, where the linear norm falls to the clip threshold. The smallest S in the grid that meets the target is S = 1, so **P1 → `LOSS_SCALE_MM2=1`, no `TASK_LOSS_WEIGHTS`**.
- Right after the warm start, the deform task sends essentially zero gradient into the encoder (3.8e-10), because the zero-initialised SVF head blocks it. The encoder is moved by the linear tasks only until the head kernel grows (it did within 100 steps here). This is expected and harmless. Early `_eff` ratios in the smoke run will look huge for that reason, so judge the ratio after a few hundred steps (the long run's logs), or from the smoke run's later window.
- The from-scratch row shows why three-task runs should be warm-started. At init the head kernels are zero, so the encoder gradient is ~0 for all tasks.
- Caveat: narrow widths, synthetic data, N 4 / 2. These numbers show the mechanism only. The cluster smoke run checks the ratio at real widths and N 32 / 16 (P2 below).

## Test results

- **P0 (determinism):** the golden fixture was generated on the unmodified code (BGR_WRITE_GOLDEN=1) and the test was rerun twice: max |diff| 0 both times. So `TestLinearOnlyUnchanged` asserts exact equality (tolerance 0) of the 6 per-step losses and every variable's float64 sum and max |v|. It still passes exactly after all changes (in the targeted runs and in the full suite).
- **Targeted runs during development (tf_gpu):**
  - `tests.test_runtime_infra.TestCheckpoint` + `tests.test_main...test_init_weights_warm_start`: 9 OK.
  - `TestLinearOnlyUnchanged`, `TestLossScale`, `TestThreeTasks`, `TestTaskLossWeightDeform`, `TestArgs`: 17 tests, 1 failure in my own new test (the clipped trainer copied weights after the reference trainer had already stepped). Fixed, then OK in the next runs.
  - `test_three_task_validation_group_sizes`, `test_dali_data_wiring_with_deform`, `test_three_task_warm_start_smoke`, `TestLossScale`: 5 OK.
  - `tests.test_check_mixed_run` (bgr_preproc38): 11 OK.
- **Py3.8 checks** (`py38_check.sh`, bgr_preproc38, Python 3.8.20): py_compile and `ast.parse(feature_version=(3, 8))` OK for all 10 changed .py files; `runtime.args` imports without TF; `main.py --help` OK and lists `--task-loss-weights`; `bash -n` OK for `slurm/m7_three_task.sh`, `tests/run_tests_wsl.sh` and `grad_balance_probe.sh`. All shell scripts are LF (`.gitattributes` `*.sh eol=lf`).
- **Full suite** (`tests/run_tests_wsl.sh`, log `test-log.txt`):
  - Stage 1 (bgr_preproc38): py_compile and ast (3, 8) OK on 51 files; np_reference OK; TF-free imports and `main.py --help` OK; `bash -n` OK including `slurm/m7_three_task.sh`.
  - Stage 2 (tf_gpu): **Ran 295 tests in 6127 s, OK (skipped=4)**. That is the 284 baseline plus 11 new tests. The 4 skips are the same as before: 3 need DALI, 1 needs dllogger. `TestLinearOnlyUnchanged` max |diff| 0. The suite's run of the three-task smoke passed again.
- `.gitignore` has an uncommitted change (adds `.kiro/skills/` and `.agents/tmp/`). It is not from this milestone; I left it untouched.

## Local end-to-end smoke (`tests.test_main...test_three_task_warm_start_smoke`)

Source: 6 linear steps (rigid:1,affine:1, N 4). Then a rigid:1,affine:1,deform:2 run with `--group-size 4 --deform-group-size 2 --init-weights <src>/ckpt/last --loss-scale-mm2 100`, 12 steps (seed 0: 4 rigid, 3 affine, 5 deform), validation every 6 steps. `check_mixed_run.check(..., expect_init_weights="partial", expect_enc_grad=True)` passed all 18 lines:
- trace counts 1 / 1 / 1, no non-finite step, PNGs for all three tasks at step 12;
- `ckpt/best` and `ckpt/best_deform` both written;
- `init_weights` mode partial: restored centering / encoder / linear_decoder, fresh deform_decoder, config diff `{deform_decoder: [False, True]}`;
- encoder grad medians: rigid raw 3819 / eff 4.42 (grad_norm 103), affine raw 12 140 / eff 7.48 (188), deform raw = eff 0.0076 (0.18). The INFO ratio was 585 / 990: this is the first-steps regime with a near-zero SVF head kernel described above, on a 6-step source.
- Final validation: deform template Dice 0.844 vs identity 0.835, pair Dice 0.695 (= identity), neg J 0 %; rigid / affine eval center RMS 22.1 / 26.4 mm (barely trained source, as intended).

## Known gaps / notes

- The m5 checkpoint's config.json predates the milestone-6 fields. `ModelConfig.from_dict` fills in the defaults, so the expected diff is exactly `{deform_decoder}`. Check the `init weights (partial)` lines in the smoke `.out` log.
- The deform step time at N 16 (~3 s) is extrapolated from N 8. The smoke run measures it: the jsonl `time/compute_ms` is per window and mixes tasks; estimate it from wall time and the task mix.
- During the probes, lr 1e-12 still lets BN moving statistics update (eval metrics drift slightly between probes). The gradient norms use training-mode batch statistics, so the measurement is unaffected.
- Not implemented (not needed): a separate deform validation group-size flag.

## Cluster commands (for the orchestrator; get Jay's approval first)

On mlsc-login, after commit + push and `git pull` in `/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad`. Check that `squeue -u jn85` shows fewer than 2 active jobs. Both jobs: partition rtx8000, account qtim, 1 GPU (RTX 8000 or A6000, 48 GB), 3 CPUs, 64 GB RAM, `-N 1 --ntasks-per-node=1` (all in the `#SBATCH` header). Logs go to `.../GroupwiseReg/logs/<job-name>-<id>.{out,err}`, results to `.../GroupwiseReg/results/<RUN_NAME>`.

Time estimates: linear step ~1.4 s at N 32 (m5, measured), deform step ~3 s at N 16 (estimated). With the 1:1:2 ratio, half the steps are linear and half deform, so ~2.2 s per step on average. Each validation covers 4 linear sets (rigid, affine, rigid_g, affine_g; 2 groups each, eval + train mode) plus deform (3 groups, eval + train mode, identity once), ~1-3 min.
- Smoke: 200 × 2.2 s ≈ 7.5 min + ~5 min startup / tracing + 2 validations ≈ 20 min → `--time=00:45:00`.
- Long: 12000 × 2.2 s ≈ 7.3 h + 48 validations × ~2 min ≈ 1.6 h ≈ 9 h → `--time=12:00:00`.

```
mkdir -p /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/logs
RUN_NAME=m7_smoke STEPS=200 VAL_EVERY=100 WARMUP=50 sbatch --job-name=m7smoke --time=00:45:00 --mem=64G slurm/m7_three_task.sh
# after reviewing the smoke run (P2):
RUN_NAME=m7_three_task_long sbatch --job-name=m7long --time=12:00:00 --mem=64G slurm/m7_three_task.sh
```

**P2 (after the smoke run, before the long run).** Read the checker's `enc_grad_ratio` INFO line and the jsonl. Because the SVF head starts at zero, the 200-step smoke ratio is biased high, so use the windows after step ~100.
- If a linear/deform `_eff` ratio is outside [0.1, 10] and the linear steps were unclipped (`train_<task>/grad_norm` < 12): `LOSS_SCALE_MM2=<1 × ratio / 3>`.
- If they were still clipped: `TASK_LOSS_WEIGHTS=deform:<ratio/3 rounded to 2^k>`. Pass either as an env override on the long-run sbatch line.

Also check:
- no `collapse_suspect` events and `train_<task>/diag_feat_rel_spread` > 1e-2;
- val_rigid / val_affine at steps 100 / 200 within ~1 mm of the m5 checkpoint (3.95 / 4.66 mm);
- `val_deform/pair_dice` above `pair_dice_identity`;
- the `val_deform` vs `val_deform_trainmode` gap;
- `gpu_peak_mb` < 46000 (on OOM: `DEFORM_N=12`).
