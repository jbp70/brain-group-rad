# Predict-script: final report

Date 2026-11-02. Design v5 (`design.md`), plan `plan.md`, code review `code-review.md` (APPROVED,
4 NITs). Code on main: 7a91cd0 (FEAT-002 stores + streaming engine), f8d6590 (FEAT-003 loader +
runners), 57b60c1 (FEAT-004 canonical fields, atlas, internal check, CLI), 7fdc80c (FEAT-005 SLURM
script + suite wiring), 835659b (integration: full suite 802 tests, 4 skipped, status 0),
1c8c525 (review). Cluster validation: job 10542610.

## 1. What it does

`predict.py` loads a trained run checkpoint (level 4, level-2 cascade or level-1 cascade; model
config, knobs and frozen stages restored from the run's `config.json`, fingerprints checked). It
runs the selected branch over the WHOLE group of processed subjects in one model evaluation (all
group means, the SVF / matrix centering and the atlas use all N subjects). It writes per subject
`<subject_dir>_fwd.nii.gz`, `_inv.nii.gz` (displacements in mm, [X,Y,Z,3] float32, evaluator
convention `--components xyz --signs +++ --units mm`, no role swap) and `_moved.nii.gz`, plus
`atlas.nii.gz` (mean of the moved images), `atlas_seg.nii.gz` (majority vote of the warped
labels, `evaluation.metrics.MajorityVote`) and `predict.json` (config, precision, chunks, timing,
GPU peaks, nonfinite counts, field statistics, fp16 headroom, internal check). The outputs go
straight into `python -m evaluation`. Test-time flips are not implemented (`--tta none` is the only
choice; milestone 10).

## 2. CLI

```
python predict.py run --checkpoint CKPT --processed-dir DIR --out OUT
    [--branches rigid|affine|deform|affine+deform]   (default affine+deform)
    [--subjects-file F] [--label-name dkt31aseg] [--grid-reference target_grid.json]
    [--mode streamed|unchunked] [--device gpu|cpu] [--precision checkpoint|float32]
    [--chunk-l1 2] [--chunk-l2 8] [--chunk-l4 32]  (0 = whole group)
    [--store auto|ram|disk] [--store-dir D] [--gpu-memory-limit-mb M]
    [--check-internal 2] [--compress-level 1] [--tta none] [--overwrite] [-v]

python predict.py check-streaming --checkpoint CKPT --processed-dir DIR --out report.json
    [--n-subjects 4] [--branches ...] [--precision checkpoint|float32] [--device gpu|cpu]
    [--chunk-l1/-l2/-l4 C]   (default per level min(cap, N-1): every level streams a full chunk
                              and a remainder)
    [--grid-reference R] [--tol-field-max-mm X] [--tol-field-mean-mm Y] [--tol-moved W]
    [--tol-matrix Z] [--tol-translation-mm T]
```

- `--checkpoint` takes a prefix (`.../ckpt/best_deform/best_deform-5000`) or a checkpoint
  directory (latest prefix in it).
- `--precision checkpoint` runs the trainable stage in the checkpoint's policy (mixed_float16 for
  AMP runs, as in training validation). `float32` evaluates the same weights in float32. Frozen
  stages are always float32.
- `--grid-reference` rejects a processed grid whose affine differs from the training grid by
  more than 1e-3 (catches an LR-mirrored grid with the right shape).
- `--store auto` keeps intermediate activations in RAM when the estimate is at most 0.7 x the
  memory budget (`SLURM_MEM_PER_NODE` or MemAvailable), else in memmapped files under
  `--store-dir` (default `$TMPDIR`).
- Exit codes: 0 ok, 2 invalid input / config / checkpoint, 3 non-finite outputs (files written,
  atlas not), 4 check-streaming bound exceeded, 5 internal consistency check failed, 1 unexpected.

Cluster job: `sbatch slurm/predict_mindboggle.sh` (rtx8000, qtim, 1 GPU, 3 CPUs, 96G, 5 h). Env
overrides CKPT, NAME, BRANCHES (default `deform`), PROCESSED, OUT, N_CHECK, CHECK_CHUNKS, GRID_REF,
EVAL_PAIRS, EXTRA. Steps: 1a float32 streaming check (gate), 1b fp16 check (record), 1c float32
affine+deform check (record), 2 prediction, 3 check-convention + template-only evaluation. It
prints the `slurm/eval_mindboggle.sh` command for the full pairwise evaluation on `basic` and does
not submit it.

## 3. Streaming design

Problem: at level 1 the group cannot be one tensor. A single default-width activation
(d1 skip concat, 64 ch x 6.88 M voxels = 440 M elements per image) limits an unchunked level-1
deform pass to N <= 4 by the int32 per-tensor limit, and linear to N <= 9 (32-channel stem).

Observation (design 2.3): in inference mode (`training=False`) BatchNorm uses moving statistics,
so every group interaction of the network is either a group MEAN (stem and encoder
`GroupConvBlock`s, deform decoder blocks, SVF centering, atlas) or a computation on N small
tensors (linear decoder on the [N,5,7,6,C] bottleneck, matrix centering, cascade
`compose_linear`). Means decompose exactly into chunk sums.

Engine (`inference/streaming.py`, `StreamedLevel`): one level model runs as passes P0-P12, one
block per pass. Each pass reads the previous block's output per chunk from a host store, applies
the block with the group mean accumulated in the previous pass, stores the output and
accumulates the next block's mean (float32 sums, divided by N once). The block math is the
model's own: a pure refactor in `models/layers.py` exposes `GroupConvBlock.pre` and
`conv_with_group_mean` (conv(concat(x, mean)) as conv_a(x) + conv_b(mean) + bias); a bitwise
guard test and the unchanged trainer goldens show that training is unaffected. The full-res stem
output and d1 output are never stored (the stem is recomputed from the stored level input;
d1 is fused with the SVF head). The linear decoder, centering and `compose_linear` run on all N
gathered at once with the model's own code.

Runner (`inference/runner.py`): chains the levels as the trainer does (`FrozenStage`,
`FrozenChain`, `_cascade_val_*_fn`): pooled level-4/2 inputs, linear chain A4 -> A2 -> A1 with
pre-warps, deform chain with pre-warps and `compose_cascade`. The last level hands per-chunk
fields to a sink that converts them to canonical mm fields (`inference/canonical.py`), writes the
files, warps image and labels with `evaluation.fields.warp` and accumulates the atlas.
`UnchunkedRunner` runs each level stage as one direct call on all N and is the reference of the
equivalence checks.

Internal consistency check (`--check-internal K`, default 2): for the first K subjects the
written fields are compared with the model's own TF warp ops applied to a coordinate volume
(exact for any branch) and to the image. This ties the files to the computed warp independently
of how the moved image was produced.

Chunk defaults: 2 / 8 / 32 at levels 1 / 2 / 4, capped below the int32 limit (cap 4 at level 1).
`--device cpu` runs the same code without a GPU (trainable stage then float32).

## 4. Branch combinations

| `--branches` | Pipeline |
|---|---|
| `rigid` | rigid cascade on the raw images; linear fwd / inv fields of A_rigid |
| `affine` | affine cascade on the raw images; linear fields of A_affine |
| `deform` | deform cascade on the raw images; fwd = u, inv = u_inv |
| `affine+deform` | affine cascade, images resampled once with A^-1, deform cascade on them; one composed field A^-1 o (id + u) and its inverse |

Why only these four (Jay's decision D1): rigid and affine are two alternative heads on the same
encoder, trained on the same input distribution in separate steps; no step ever fed one linear
head's output into the other, so `rigid+affine` or "all three" has no trained meaning. The deform
branch was trained on MNI-affine-aligned images (residual affine off), never on rigidly aligned
inputs that keep per-subject scale and shear, so `rigid+deform` is excluded. Any other value
(`all`, `rigid+deform`, `rigid+affine`, `deform+affine`, ...) is an argparse error (exit 2) whose
message lists the four values.

Deform only works and is the Mindboggle choice: the processed Mindboggle images are already
affine-aligned to the training grid, which is the input distribution the deform branch was
trained on. The script cannot check that the inputs are affine-aligned; the `deform` help text
states the requirement.

## 5. Equivalence numbers

Local (GTX 1080, random-weight models; commit messages of FEAT-002..004):

| Comparison | Max difference |
|---|---|
| streamed vs model, float32, levels 4 / 2 / 1 (L1 TINY widths on CPU, N 3) | fields <= 1.34e-5 vox (cascade compose), <= 1.3e-6 otherwise; A <= 1.8e-7; L1 u 5.1e-7 |
| streamed vs unchunked, float32, cascade, all branches, chunks 1 / 2 / N | fields <= 8.94e-6 vox, A <= 3.58e-7 |
| streamed vs unchunked, mixed_float16 trainable stage | fields max <= 4.6e-3 vox (mean 3.45e-4) |
| unchunked vs Trainer validation | level 2 bitwise; level 1 u 1.90e-7, Dice identical; streamed vs Trainer <= 8.94e-7 |
| written level-1 files vs Trainer | unchunked 4.67e-7 vox, streamed <= 1.39e-6 vox |
| level-2 internal check | coordinate <= 2.44e-4 vox, moved <= 2.30e-5 |

`check-convention` on the test outputs picked the canonical flags (margins 0.141 / 0.918) and
`evaluation run --pairs 2` finished with finite metrics. Full suite: 802 tests, 4 skipped,
status 0.

Real checkpoint (cluster job 10542610, 4 subjects, chunks 3 + 1 at every level, streamed vs
unchunked):

| Check | fwd max (mean) mm | inv max (mean) mm | moved max | Bound | Result |
|---|---|---|---|---|---|
| 1a float32, deform (gate) | 3.49e-4 (7.97e-6) | 2.39e-4 (6.95e-6) | 5.72e-5 | 1e-3 | PASS |
| 1b fp16 (checkpoint precision), deform | 0.666 (3.94e-3) | 0.645 (3.44e-3) | 2.58e-3 | 0.1 max, 2e-3 mean (provisional) | FAIL (record only) |
| 1c float32, affine+deform | 3.89e-3 (1.54e-5) | 4.99e-4 | 3.03e-4 | 1e-3 | FAIL (record only) |

- 1c: the affine matrices differ by 1.19e-7 (linear part) and 9.87e-6 mm (translation). The
  differences are uniform and tiny, consistent with rounding; the affine branch is not used on
  Mindboggle.
- 1b: mean differences are small, the max of 0.67 mm is plausibly fp16 rounding of the group
  means in a different summation order. The fp16 bounds were provisional by design and never a
  gate. The 101-subject prediction ran in fp16 (checkpoint precision), so this is the open
  question; a `--precision float32` rerun (peak ~15 GB streamed in 1a) removes it.

## 6. Cluster run (job 10542610)

rtx-07, Quadro RTX 8000 48 GB, 45 min, exit 0:0. Checkpoint
`results/m9_level1_cascade/ckpt/best_deform/best_deform-5000` (frozen L4 m7_three_task_lncc
step 11000, L2 m8_level2_gradnorm, fingerprints matched), BRANCHES=deform, trainable stage
mixed_float16. Streaming checks ran at 1c8c525; the clone was pulled to 5152bb1 (ANTs commits
only) during the job, so predict.json records 5152bb1. Jay approved the submission as a one-time
exception to the 2-job limit.

| Phase | Wall | TF GPU peak |
|---|---|---|
| 1a float32 check, deform, 4 subjects | 177 s | unchunked 17.6 GB / streamed 15.0 GB |
| 1b fp16 check | 130 s | 12.1 / 10.6 GB |
| 1c float32 check, affine+deform | 198 s | 18.0 / 15.3 GB |
| 2 prediction, 101 subjects, streamed | 1356 s | L4 10.9, L2 11.8, L1 9.9 GB |
| 3 check-convention + template evaluation | ~14 min | CPU |

- Phase 2: L4 61 s, L2 139 s, L1 1076 s, atlas 31 s; writing 898 s, overlapping L1. Chunks
  32 / 8 / 2.
- Host memory: sacct MaxRSS 74.4 GB; predict.json peak RSS 48.1 GB, of which 36.7 GB is the RAM
  store (design estimate ~40 GB).
- The check-streaming GPU peaks are those of the last runner phase (level-1 deform), not of the
  whole side (review NIT 1); for these runs that phase is the largest.
- Nonfinite voxels: 0 in fwd, inv and moved for all 101 subjects. Max |fwd| 17.5 mm, max |inv|
  21.9 mm; max |v| L4 7.0, L2 9.8, L1 7.7. Largest fp16 activation 4300 (L1 d1), 6.6 % of 65504.
- Internal check (2 subjects): files reproduce the internal coordinates to 4.6e-5 vox and the
  moved image to 2.4e-7.
- check-convention: `--components xyz --signs +++ --units mm`, no role swap, NCC 1.0000
  (runner-up 0.892).

Outputs: `.../GroupwiseReg/results/predict_mindboggle/m9_best_deform_5000_deform/` (`fields/`
with the per-subject files, atlas, atlas_seg and predict.json; `check_streaming_*.json`;
`convention.json`; `eval_template/`). Logs `.../GroupwiseReg/logs/bgr-predict-10542610.{out,err}`.

## 7. Mindboggle metrics (101 subjects, 106 labels, template-only)

Our model, deformable branch only (the Mindboggle images are already affine-aligned). Identity
and baselines from the existing `results/eval_mindboggle/*`.

| Metric | Ours (deform) | Identity | MultiMorph | BrainMorph tps0 |
|---|---|---|---|---|
| template_ncc | 0.984 | 0.748 | 0.953 | 0.803 |
| template_lncc | 0.794 | 0.309 | 0.728 | 0.356 |
| template_dice | 0.700 +- 0.022 | 0.495 | 0.688 | 0.505 |
| cortical / subcortical | 0.684 / 0.724 | 0.440 / 0.580 | 0.658 / 0.735 | 0.443 / 0.598 |
| transfer_dice | 0.691 | 0.495 | 0.680 | 0.505 |
| template_dice_loo | 0.696 | 0.486 | 0.684 | 0.496 |
| fold_fwd / inv % | 0.22 / 0.40 | 0 / 0 | 0.0002 / 0.0011 | 0 / 0 |
| logjac_std fwd / inv | 0.70 / 0.87 | 0 | 0.47 / 0.61 | 0.04 |
| inv_err fwd / inv mm (max) | 0.74 / 0.75 (13.8 / 13.7) | 0 | 0.23 / 0.28 (4.2 / 5.3) | 0.02 |
| disp_fwd_mean mm | 2.62 | 0 | 2.22 | 1.95 |
| centrality mean / max mm | 0.58 / 4.18 | 0 | 0.19 / 1.18 | ~0 |
| atlas_intensity_std | 0.057 | 0.168 | 0.082 | 0.156 |

- Accuracy: template Dice 0.700 vs identity 0.495 (+0.205) and MultiMorph 0.688; higher on
  cortex (0.684 vs 0.658), slightly lower on subcortical labels (0.724 vs 0.735).
- Regularity is worse than MultiMorph (folding 0.22 / 0.40 %, Jacobian spread, inverse error,
  centrality) and worse than our own level-1 training validation (folding 0.012 %, inverse
  error 0.14 mm, groups of 4). Candidate causes: N = 101 vs training group sizes of 4-16
  (design 2.4), or fp16. A float32 rerun separates the two.
- Pairwise metrics are not computed yet (template-only in the GPU job).

## 8. Open points and follow-ups (not submitted; need Jay's approval)

1. Rerun the prediction with `--precision float32` (e.g. `EXTRA="--precision float32"
   NAME=m9_best_deform_5000_deform_f32 sbatch slurm/predict_mindboggle.sh`): settles the fp16
   streaming gap and whether the high folding comes from fp16 or the group size.
2. Full pairwise evaluation on `basic`: `METHOD=m9_best_deform_5000_deform FWD=<fields>/{subject_dir}_fwd.nii.gz
   INV=<fields>/{subject_dir}_inv.nii.gz ATLAS=<fields>/atlas.nii.gz ATLAS_SEG=<fields>/atlas_seg.nii.gz
   CONV="--components xyz --signs +++ --units mm" sbatch slurm/eval_mindboggle.sh`.
3. Review NITs, not fixed: check-streaming `gpu_peak_mb` covers only the last phase; NaN can hide
   in the `max_abs_v` / `fp16_max_abs` records (fields and exit codes unaffected); the store
   estimate ignores float32 decoder stages (~5 GB low at N 101 for the float32_fine recipe);
   residual-block checkpoints exit 1 instead of 2.
4. Group-size ablation (N = 4 groups vs N = 101) and test-time flips (milestone 10) later.
