# predict-levels: pyramid-level selection for predict.py (report)

Design v2 (`design.md`, review APPROVED with 3 NITs, `design-review.md`). Code commit `5b27491`
(`feat: predict.py --levels ...`); code review APPROVED, no findings (`code-review.md`).
Cluster: see section 6.

## 1. CLI

Both subcommands (`run`, `check-streaming`) get:

```
--levels LEVELS          cascade levels to run, coarse to fine, comma-separated:
                         4 | 4,2 | 4,2,1 | 2,1 | 1 | 2 | 4,1 (level-1 checkpoint),
                         4 | 4,2 | 2 (level-2 checkpoint), 4 (level-4 checkpoint).
                         Default: all levels of the checkpoint (today's code path, bit-identical).
--allow-untrained-skip   run a selection whose skip the checkpoint chain never saw in training
                         (refused with exit 2 otherwise)
```

Parse-time checks (argparse, exit 2, no TF import): items in {4, 2, 1}, no empty item, strictly
descending (no duplicates), at least one level; spaces around commas are fine. After the
checkpoint's config.json is read (no model build): a level that is not a level of the checkpoint
exits 2 (`--levels 1: the checkpoint is a level-2 run with levels [4, 2]`). A selection equal to all
checkpoint levels (`--levels 4,2,1` on a level-1 run) is the default path.

`affine+deform` uses the same `--levels` for the affine and the deform cascade (Jay's option A).

SLURM: `slurm/predict_mindboggle.sh` takes `LEVELS` (default empty = no flag) and
`ALLOW_UNTRAINED_SKIP=1`; both reach steps 1a, 1b, 1c and 2 through `$LEVEL_FLAGS`. With both
unset the predict.py command lines are unchanged (stub-run test).

## 2. Semantics per combination (level-1 checkpoint: stages 4 frozen, 2 frozen, 1 trainable)

x1 = level-1 images (affine-resampled for affine+deform), x2 = pool(x1); up =
`diffeo.upsample_displacement` (one per 2x step, forward and inverse separately).

| `--levels` | Kind | Deform | Linear (rigid / affine) | Same as the trainer's |
|---|---|---|---|---|
| default / `4,2,1` | all | existing path | existing path | validation cascade |
| `4,2` | truncation | U1 = up(compose(up(u4), up(u4_inv), u2, u2_inv)) | A = A2 | `FrozenChain.deform` / `linear` |
| `4` | truncation | U1 = up(up(u4)) | A = A4 | `FrozenChain.deform_stages` U4 / `linear_stages` A4 |
| `2,1` | skip 4 | U4 = 0; level 2 on warp(x2, 0), compose(0, 0, u2, u2_inv), up; level 1 on warp(x1, U), compose | A4 = I; level 2 on x2 o I^-1, A2 = Center2(R2 I); level 1 as usual | `deform_skip(skip4)` / `linear_skip(skip4)`, then the level-1 body |
| `1` | skip 4 and 2 | U = U_inv = 0; level 1 on warp(x1, 0), compose(0, 0, u1, u1_inv) | A_prev = I; level 1 on x1 o I^-1, A = Center1(R1 I) | `deform_skip(skip4, skip2)` / `linear_skip(skip4, skip2)` |
| `2` | skip 4 + truncation | `deform_skip(skip4)` U | `linear_skip(skip4)` | as `2,1` without level 1 |
| `4,1` | skip 2 | U = up(up(u4)); level 1 on warp(x1, U), compose | A = A4; level 1 on x1 o A4^-1 | `deform_skip(skip2)` / `linear_skip(skip2)` |

Level-2 checkpoint: `4` = truncation (U1 = up(up(u4)), A = A4); `2` = skip
(`skip_fields` / `skip_linear`: U = U_inv = 0, A_prev = I; level 2 on the identity-pre-warped x2,
composed with the identity). Level-4 checkpoint: only `4`.

How it matches training (runtime/run.py, stage mode): a skipped stage's output is replaced by
exact zeros / the exact identity, and every downstream op (pre-warp, `compose_cascade`,
`compose_linear` with centering) still runs on that identity. The streamed runner reproduces this
with the same ops (`diffeo.warp(pool(x), 0)`, the P12 composition with (0, 0), `tf.eye(4)` = the
bitwise `skip_linear` value); the unchunked runner calls the trainer's functions literally with
`tf.constant(True)` as the skip tensor. Truncation uses the cascade's own between-level
upsampling; matrices pass between levels unchanged, as in `FrozenChain`.

predict.json / check-streaming JSON key `levels`: requested, checkpoint_levels, selected, default,
last, skipped, truncated, per_level text, upsampled_from_level, allow_untrained_skip, exposure
(runs, entries, min_joint, forced, low_threshold). `pipeline` names each level's kind for a
non-default plan (e.g. `deform cascade (L4 skipped (identity) -> L2 -> L1) ...`,
`(L4 -> L2; L1 not run (truncated after level 2))`); default runs keep today's text.
`chunks.bounds` lists the run levels only. u_ckpt / v fields are only produced when the
checkpoint's own stage runs (not written to files in any case).

## 3. Training exposure

Read before any model is built from config.json `args` of the trainable run and (level 1) its
level-2 source (`cascade.source_config`, else the config of the level-2 `cascade.stages` origin
checkpoint). Stage mode: `cascade_skip_stage_parsed[task][level]` (missing = 0); legacy
`cascade_skip_prob` p: level-2 run skips level 4 with p, level-1 run skips level 2 with p. A
resume whose args differ from `cascade.skip_probs` logs a WARNING and sets
`args_vs_record_mismatch` (decision from args).

For every run level s with skipped coarser levels K and every branch task t:
`joint` = P(all levels in K skipped in a training step of s, task t) = product of p_l (independent
draws, `cascade_stage_skip_draw`); other frozen levels may also have been skipped in those steps.
`exact_pattern` = joint x product of (1 - p_m) over the run frozen levels m of that run (exactly
this pattern). `trained` if joint > 0, `untrained` if some p_l = 0, `unknown` if the config is not
readable. Untrained or unknown without `--allow-untrained-skip` -> exit 2 before subjects are read
(predict.json written with status error and the entries); with the flag: WARNING per entry and
`forced` true. A trained entry with joint < 0.05 logs a WARNING. Truncation needs no exposure.

Retrained chain (r1sw_lr1e-4_wd1e-1_n8_d4: deform 4:0.15, 2:0.15, linear 4:0, 2:0.15; its source
r2_s15000_lr3e-4_wd1e-1_n32_d16: deform 4:0.15, linear 4:0):

| `--levels` | deform | rigid / affine | deform+refusal result |
|---|---|---|---|
| `4`, `4,2` | truncation | truncation | runs |
| `2,1` | L2: 0.15 (r2), L1: 0.15 (r1sw; exact pattern 0.1275) | L2: 0 (r2) | deform runs; linear / affine+deform refused |
| `1` | L1: 0.15 x 0.15 = 0.0225 (WARNING < 5 %) | L1: 0 x 0.15 = 0 | deform runs with a WARNING; linear refused |
| `2` | L2: 0.15 | L2: 0 | deform runs; linear refused |
| `4,1` | L1: 0.15 (exact 0.1275) | L1: 0.15 | all branches run |

Old chain (m9_level1_cascade, m8_level2_gradnorm: `cascade_skip_prob` 0, no per-stage keys): every
skip is untrained (refused without the flag); truncation `4`, `4,2` allowed.

Not verified on the cluster in this step: that the frozen copy
`scratch/predict_ckpts/r1sw_lr1e-4_wd1e-1_n8_d4_s3000/config.json` has a readable
`cascade.source_config` (results/r2_s15000_.../config.json). If not, `2,1` / `2` report the
level-2 entry `unknown` and are refused; the `levels.exposure.runs["2"]` record shows which path
was tried.

## 4. Tests and equivalence numbers

Local WSL (tf_gpu TF 2.10 / py3.9, GTX 1080, BGR_TF_GPU_LIMIT_MB 6000; py3.8 syntax via
bgr_preproc38). Full suite `tests/run_tests_wsl.sh`: py_compile + ast 3.8 OK (111 files), TF-free
tests 167 OK without TF, unittest 825 tests (4 skipped), 0 failing invocations, status 0, 21303 s.

New / changed tests:
* `tests/test_predict_levels.py` (TF-free, 12 tests): parse_levels, plans for every checkpoint
  level x selection, normalisation, run_skip_probs (stage / legacy L2 and L1 / absent),
  exposure tables of section 3 (0.0225 for `1`, exact pattern 0.1275), unknown source, resume
  mismatch WARNING, CLI parse errors exit 2 without TF.
* `tests/test_predict_levels_runner.py` (4 tests, ~45 min, L1 on /CPU:0): items 1-5 below.
* `tests/test_predict.py`: `TestLevelsEndToEnd` (6 tests) + default-record assertions (literal
  `pipeline` strings, `levels.default`) in `TestLevel2Convention`.
* `tests/test_predict_runner.py`: `compare_modes(plan=, on_streamed=, keep_ref=)`.
* `tests/test_slurm_scripts.py`: LEVELS / ALLOW_UNTRAINED_SKIP static checks + a stub run
  (LEVELS unset gives the same predict.py calls as a LEVELS run with the flags removed).

1. Default bit-identity
   * (a) general `_sel` path with a non-normalised full plan vs the default path, same streamed
     runner, op determinism: L2 (GPU) chunks 1 and 3, L1 (GPU) chunk 1: max |diff| 0 on A, U1,
     U1_inv (deform, affine+deform). L1 chunk 2/3 do not fit the 6 GB local GPU. On /CPU:0 the
     default path is not run-to-run reproducible itself (two runs of the same default runner
     differ by up to 1.4e-6 vox), so bit identity is only testable on the GPU.
   * (b) one-off golden: `predict.py run` on a tiny level-2 checkpoint (N 3, SVF heads x64) at the
     parent commit f4e017f (git archive) vs this commit, `--device gpu`, TF_DETERMINISTIC_OPS=1;
     sha256 over the decoded arrays of all 11 NIfTI outputs:
     | branch | mode | parent | new (no flag) | new `--levels 4,2` |
     |---|---|---|---|---|
     | deform | streamed | 1ecf2e21... | 1ecf2e21... | 1ecf2e21... |
     | affine+deform | streamed | a968f91a... | a968f91a... | a968f91a... |
     | rigid | streamed | 8279ce26... | 8279ce26... | 8279ce26... |
     | deform | unchunked | 8cfb581a... | 8cfb581a... | - |
     | affine+deform | unchunked | f5ffb99f... | f5ffb99f... | - |
     | rigid | unchunked | 3b12d3ae... | 3b12d3ae... | - |
     On `--device cpu` even two parent runs differ (max 9.4e-5 in the fwd files of this tiny
     model), so the CPU cannot give a golden.
2-4. Truncation / skip vs the direct reference written from the trainer's functions, and streamed
   vs unchunked (max |diff|, vox for fields; exact = unchunked vs reference, bound 1e-5; stream =
   streamed vs reference, bounds 1e-4 fields / 1e-5 A; s-vs-u = streamed vs unchunked, same
   bounds); L1 streamed chunk 2 (N 3), L2 chunks 1, 2, 3:
   | ckpt | `--levels` | exact U1 / A | stream U1 / A | s-vs-u U1 / U1_inv / A | min |U1 - default| |
   |---|---|---|---|---|---|
   | L2 | `4` | 3.6e-6 / 2.4e-7 | 2.8e-6 / 1.5e-7 | 3.0e-6 / 3.0e-6 / 1.8e-7 | 0.70 |
   | L2 | `2` | 4.3e-6 / 1.8e-7 | 5.0e-6 / 1.2e-7 | 3.4e-6 / 3.5e-6 / 1.8e-7 (u_ckpt 2.2e-6) | 0.58 |
   | L1 | `4,2` | 3.1e-6 / 2.4e-7 | 4.2e-6 / 3.6e-7 | 3.7e-6 / 3.7e-6 / 3.6e-7 | 0.17 |
   | L1 | `4` | 1.2e-6 / 1.2e-7 | 2.1e-6 / 1.8e-7 | 2.1e-6 / 2.1e-6 / 2.4e-7 | 0.24 |
   | L1 | `2,1` | 5.6e-6 / 1.8e-7 | 6.4e-6 / 2.4e-7 | 7.0e-6 / 7.0e-6 / 2.7e-7 | 0.19 |
   | L1 | `1` | 4.1e-6 / 1.8e-7 | 4.3e-6 / 2.4e-7 | 4.0e-6 / 4.0e-6 / 1.8e-7 | 0.22 |
   | L1 | `2` | 1.6e-6 / 1.2e-7 | 1.5e-6 / 1.5e-7 | 2.5e-6 / 2.6e-6 / 1.5e-7 | 0.21 |
   | L1 | `4,1` | 4.7e-6 / 2.4e-7 | 5.2e-6 / 1.5e-7 | 4.9e-6 / 5.0e-6 / 2.4e-7 | 0.11 |
   L2 branches rigid, deform, affine+deform; L1 deform, affine+deform (A = the affine cascade).
   The unchunked results are not bitwise equal to the eager reference (lean vs non-lean warp /
   compose, tf.function vs eager, CPU nondeterminism ~1e-6). `4` truncation vs
   `deform_skip(skip2)`: 0 (bitwise). The last column shows every selection changes the result.
   Truncated selections return u_ckpt / v None in both modes (asserted).
5. Zero-field sanity of L1 `1`: the streamed pre-warp equals `diffeo.warp(x1, 0, lean=True)`
   bitwise and equals x1 (max |diff| 0).

End to end (`TestLevelsEndToEnd`, tiny level-2 checkpoint with deform 4:0.15, linear 0, SVF heads
doubled until max |U1| >= 4 vox for both selections): `run --levels 4` and `--levels 2` write the
same files, internal check passes, written fields == the streamed runner under the plan (0
vox), check-convention RESULT `--components xyz --signs +++ --units mm` for both (best NCC 1.0000);
`--branches affine --levels 2` exit 2 before reading subjects (predict.json status error with the
untrained entry); with `--allow-untrained-skip` exit 0, `forced` true, WARNING recorded;
`--levels 1` on level 2 exit 2; p 0.04 logs the "about 4.00 %" WARNING; `check-streaming --levels
2` and `--levels 4` pass (fwd max 3.5e-6 / 8.1e-6 mm), chunks for the run level only.

Local runtime note: one level-1 streamed run on /CPU:0 takes ~1.5 min; the level-1 tests peak at
~22-26 GB host RSS (existing test_predict_runner) and the first version of the new L1 test, which
kept all results, drove WSL into swapping (since fixed: one branch at a time).

## 5. Prepared ablation (NOT submitted; needs Jay's approval)

rtx8000, qtim, 1 GPU, 3 CPUs, 96G, 5 h each (script header), deform branch, one job per selection
(`4,2,1` exists: results/predict_mindboggle/r1sw_lr1e-4_s3000_deform). Each job runs the float32
streaming gate (1a), the fp16 record (1b), the affine+deform record (1c; refused for `2,1` and `1`
under option A, non-blocking), the prediction and the template-only evaluation.

```bash
cd /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad
git pull --ff-only
P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
CK=$P/scratch/predict_ckpts/r1sw_lr1e-4_wd1e-1_n8_d4_s3000/ckpt/best_deform
CC="--chunk-l4 3 --chunk-l2 3 --chunk-l1 3 --tol-field-max-mm 0.005"
for L in 4 4,2 2,1 1; do
  T=${L//,/}
  CKPT=$CK NAME=r1sw_lr1e-4_s3000_deform_L$T BRANCHES=deform LEVELS=$L CHECK_CHUNKS="$CC" \
    sbatch --job-name=bgr-predict-L$T slurm/predict_mindboggle.sh
done
```

Names: r1sw_lr1e-4_s3000_deform_L4, _L42, _L21, _L1 (no commas in directories or evaluation
`--method`). Four jobs exceed the 2-active-job rule: they need Jay's per-job exception (or submit
two at a time).

The same four submissions written out (run from the clone after `git pull`):

```bash
P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
CK=$P/scratch/predict_ckpts/r1sw_lr1e-4_wd1e-1_n8_d4_s3000/ckpt/best_deform
CC="--chunk-l4 3 --chunk-l2 3 --chunk-l1 3 --tol-field-max-mm 0.005"
CKPT=$CK NAME=r1sw_lr1e-4_s3000_deform_L4  BRANCHES=deform LEVELS=4   CHECK_CHUNKS="$CC" sbatch --job-name=bgr-predict-L4  slurm/predict_mindboggle.sh
CKPT=$CK NAME=r1sw_lr1e-4_s3000_deform_L42 BRANCHES=deform LEVELS=4,2 CHECK_CHUNKS="$CC" sbatch --job-name=bgr-predict-L42 slurm/predict_mindboggle.sh
CKPT=$CK NAME=r1sw_lr1e-4_s3000_deform_L21 BRANCHES=deform LEVELS=2,1 CHECK_CHUNKS="$CC" sbatch --job-name=bgr-predict-L21 slurm/predict_mindboggle.sh
CKPT=$CK NAME=r1sw_lr1e-4_s3000_deform_L1  BRANCHES=deform LEVELS=1   CHECK_CHUNKS="$CC" sbatch --job-name=bgr-predict-L1  slurm/predict_mindboggle.sh
```

Outputs: `$P/results/predict_mindboggle/r1sw_lr1e-4_s3000_deform_L{4,42,21,1}/` (fields,
check_streaming_*.json, convention.json, eval_template), logs `$P/logs/bgr-predict-L*-<job>.out`.
Expected: `L1` logs the low-exposure WARNING (2.25 %); step 1c (affine+deform record) exits 2 for
`L21` and `L1` (linear level 4 never skipped in training) and the job continues.

## 6. Cluster state (2026-11-03)

Cluster clone pulled to `d20ddf3` (the report commit on top of code commit `5b27491`).

Configs of the chain behind `scratch/predict_ckpts/r1sw_lr1e-4_wd1e-1_n8_d4_s3000` (read with
grep on the login node):
* `config.json` (level 1): `cascade_skip_deform` 4:0.15,2:0.15, `cascade_skip_linear`
  4:0,2:0.15, `cascade_skip_prob` 0; `cascade_skip_stage_parsed` deform {4: 0.15, 2: 0.15},
  rigid / affine {4: 0.0, 2: 0.15}; `cascade.source_config` =
  `results/r2_s15000_lr3e-4_wd1e-1_n32_d16/config.json` (exists, readable; stage origins
  r4sw_C_s16000 last-16000 and r2_s15000 last-15000).
* Source (level 2): `cascade_skip_stage_parsed` deform {4: 0.15}, rigid / affine {4: 0.0}.

So the source config is found and no entry is `unknown`. Exposure the new code computes for the
deform branch:

| `--levels` | kind | level-2 stage (r2 run) | level-1 stage (r1sw run) | result |
|---|---|---|---|---|
| `4` | truncation after 4 | - | - | runs |
| `4,2` | truncation after 2 | - | - | runs |
| `2,1` | skip 4 | joint 0.15 | joint 0.15 (exact pattern 0.1275) | runs |
| `1` | skip 4 and 2 | - | joint 0.15 x 0.15 = 0.0225 | runs, WARNING (< 5 %) |

squeue at that time: 2 jobs active (10542695 bgr-antsgw, basic, running 1 d; 10543287 r4cfgB0,
rtx8000, running 13 h), so every submission above needs Jay's exception.

SUBMITTED (Jay approved all four as an exception, 6 active jobs; clone at 8028b32): 10548270
`L4`, 10548271 `L42`, 10548272 `L21`, 10548273 `L1` (results/predict_mindboggle/
r1sw_lr1e-4_s3000_deform_L{4,42,21,1}, logs/bgr-predict-L*-<job>.out).

## 7. Ablation results (all 4 jobs COMPLETED, 33-38 min each, exit 0)

Gates: float32 streaming check (1a) PASS for all four; check-convention xyz +++ mm for all four.
predict.json `levels`: L4 selected [4] truncated [2, 1]; L42 [4, 2] truncated [1]; L21 [2, 1]
skipped [4] min_joint deform 0.15; L1 [1] skipped [4, 2] min_joint deform 0.0225. Non-blocking
records: fp16 check (1b) exceeded its provisional bounds (rc 4) for L21, L1 and also for the
existing 4,2,1 run; it passed for L4 and L42. The affine+deform record (1c) was refused (rc 2) for
L21 and L1, as expected.

Template-only metrics, mean over 101 Mindboggle subjects (4,2,1 = existing run
r1sw_lr1e-4_s3000_deform):

| metric | `4` | `4,2` | `4,2,1` | `2,1` | `1` |
|---|---|---|---|---|---|
| template Dice | 0.613 | 0.680 | **0.699** | 0.689 | 0.652 |
| Dice cortical / subcortical | 0.578 / 0.665 | 0.659 / 0.711 | 0.684 / 0.722 | 0.669 / 0.719 | 0.630 / 0.685 |
| template LNCC | 0.498 | 0.656 | **0.781** | 0.755 | 0.679 |
| template NCC | 0.906 | 0.955 | 0.981 | 0.980 | 0.968 |
| folding fwd / inv % | 0.0002 / 0.0002 | 0.015 / 0.048 | 0.211 / 0.491 | 0.280 / 0.642 | 0.110 / 0.083 |
| log-Jacobian std fwd / inv | 0.319 / 0.351 | 0.459 / 0.570 | 0.684 / 0.852 | 0.743 / 1.041 | 0.852 / 1.121 |
| inverse error fwd / inv mm | 0.40 / 0.42 | 0.62 / 0.64 | 0.76 / 0.75 | 0.63 / 0.77 | 0.32 / 0.42 |
| centrality mean / max mm | 0.30 / 2.00 | 0.51 / 3.70 | 0.63 / 4.49 | 0.51 / 3.55 | 0.28 / 1.72 |
| atlas_intensity_std | 0.108 | 0.080 | **0.059** | 0.060 | 0.069 |
| mean |fwd| mm | 2.03 | 2.53 | 2.79 | 2.33 | 1.74 |
