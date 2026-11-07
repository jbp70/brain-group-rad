# Review of `.agents/tasks/tta/design.md` (version 2)

Reviewer, 2026-11-06. Second review: the earlier findings are re-checked against version 2, then the changes are reviewed. Code read for this round: `models/transforms.py` (`decode_10d`, `encode_10d`, `_matrix_to_quaternion`, `rigid_matrix`, `scale_shear_matrix`, `affine_matrix`, `barycenter`, `center_group`, `GroupCentering`, `to_float`), `models/cascade.py` (`compose_linear`, `prewarp_linear`), `inference/runner.py` (`_linear`, `_resample`, `_deform`, `run`, `_linear_sel`, `_deform_sel` wrapper), `inference/streaming.py` (P0 `np.asarray(input_fn(...))`, P11 sum, `_f_p12`, `deform`, `linear`), `inference/store.py` (`estimate_store_gb`, `choose_store`), `inference/predict.py` (`PREDICT_KEYS`, `_peek_chain_configs`), `inference/cli.py` (`--tta` placeholder), `runtime/checkpoint.py` (`source_config_path`, `source_config`), `runtime/metrics.py` / `runtime/fixed_data.py` signatures, `slurm/predict_mindboggle.sh`, `slurm/eval_mindboggle.sh`.

Verdict: **APPROVED** (0 HIGH, 0 MEDIUM, 3 NIT).

## Earlier findings (review v1)

| # | Severity | Status | Check |
|---|---|---|---|
| 1 | HIGH, raw gauge | **Closed** | 2.3 now averages R_f = A_f A_prev⁻¹ with A_f = Center(F A^F_raw F · A_prev). In the code `compose_linear(R_raw, A_prev, c) = c(R_raw @ A_prev)` and `center_group = mu⁻¹ A`, so R_f = mu_f⁻¹ M_f and any common left factor of M_f is removed (up to the 4-iteration fixed point, which `barycenter`'s docstring puts at ~1e-5 mm). The re-encoding is consistent with `affine_matrix` = [R K | t] (R @ K, K upper-triangular, raw strict-upper entries, not the `qr_factors` ratios). QR of F L F = (F Q F)(F K F) and quadratic `encode_10d`, so encode(F M F) = σ_f ⊙ encode(M) exactly. Claim 1 is rewritten for centered-equivariant models, with tolerances. The identity-only path is kept bit-identical. T7(b) is a regression guard. I hand-checked the T7(a) stub signs against σ_x and σ_y (10-D entries and shears): consistent. |
| 2 | MEDIUM, `src_affine` freed after flip 0 | **Closed** | 3.4 installs the wrapper only when `is_last and free_src and not _tta_at(lv)`, in both `_deform` and `_deform_sel` (both wrappers exist at runner.py ~319 / ~497). With TTA, `SRC_AFFINE` is freed after `_deform_stage` in `try/finally`. The level mask is computed before the first P0. `estimate_store_gb` already adds `4 * full_vox` at every deform level when `resampled` (verified). T6, T10 and T11 have cases for this. The previous level's fields (`prev`) are freed after `eng.deform` returns today, so they survive the 8 flips too. |
| 3 | MEDIUM, statistics | **Closed** | The scope statement (conditional on the fixed 101-subject group) is in 9.3 and in `compare.md`. The subject-cluster bootstrap cᵀ(W ⊙ D)c / cᵀWc has self-pairs excluded and supports the seeded subset. The folding SESOI is fixed (OQ 2). Replication is optional (OQ 8). |
| 4 | NIT | Closed | `SKIP_CHECKS` added. Section 9.4 notes that the nested `bash` call ignores `#SBATCH` and that TMPDIR is shared. |
| 5 | NIT | Closed | `tta_record` added to `PREDICT_KEYS` (null when off and in early error records). |
| 6 | NIT | Closed | T9 now hashes the decoded arrays, the affines and A. |
| 7 | NIT | Closed | T6 and T7 run in float32. |
| 8 | NIT | Closed | `train_flip_probs` follows the `_peek_chain_configs` chain rule (`cascade.source_config`, then the origin checkpoint's `source_config_path`; both exist). Warn-only, with the reason stated. |
| 9 | NIT | Closed | The residual-alignment note is in 7.3 and in `summary.json`. |
| 10 | NIT | Closed | The `ValueError` is caught: p = 1.0 and `note = "all differences zero"`; the sign test is handled the same way. |

## New findings

### 1. NIT: T9 depends on `enable_op_determinism` but gives no rule for when it fails

Where: T9.
Problem. `tf.config.experimental.enable_op_determinism()` (TF 2.10) raises at run time for GPU ops that have no deterministic kernel. Without it, cuDNN autotuning can change the conv algorithm between runs and break bit-exact hashes, even with unchanged code. The design does not say what the implementer does if the prediction path hits such an op, or how to tell nondeterminism from a real regression.
Fix. Add to T9: "Step 0: run the fixture twice on the pre-change commit. The hashes must match. If `enable_op_determinism` raises for an op of the prediction path, set `TF_CUDNN_DETERMINISTIC=1` and `TF_DETERMINISTIC_OPS=1` instead and repeat step 0. If the two pre-change runs still differ, T9 compares with max |diff| == 0 on the decoded arrays of two in-process runs (pre- and post-refactor `deform()` in the same process), not with stored hashes."

### 2. NIT: 9.4 lacks the exact submission lines for the compare jobs

Where: 9.4. The brief asks for exact SLURM commands. The `compare_tta.py` step appears as a bare `python` line with the note "inside a basic job, 1 CPU, 4G, 15 min". The linear-eval comparison has no command at all.
Fix. Give the full line, for example:
```bash
sbatch -A qtim -p basic -N 1 --ntasks-per-node=1 --cpus-per-task=1 --mem=4G --time=00:15:00 \
  --output=$PROJ/logs/tta-compare-%j.out --error=$PROJ/logs/tta-compare-%j.err \
  --wrap "singularity exec -B /autofs -B /scratch $SIF bash -c 'cd $REPO && python scripts/compare_tta.py --a ... --b ... --out ...' </dev/null"
```
Add the matching `--kind linear` (or equivalent) invocation for `results/linear_eval_mindboggle/${NAME}_{rigid,affine}_tta_{none,flip8}`, and state `compare_tta.py`'s flags for the per-subject / per-pair CSV layouts of `evaluation run` and `linear-eval`.

### 3. NIT: `spread_max_mm` means "max" for linear but "p99" for deform

Where: section 5 table. For linear the column is max_f point_rms. For deform it is the p99 of σ over the mask. One column name with two meanings will be misread in `tta_spread.csv`.
Fix. Name the deform column `spread_p99_mm` and leave it empty for linear rows. Or rename both: linear `spread_max_mm`, deform `spread_p99_mm`.

## Checklist (reviewer brief)

| Check | Result |
|---|---|
| Sign / unflip derivation vs model code | Correct. 10-D layout a00 a01 a02 a03 a11 a12 a13 a22 a23 a33 over (w, x, y, z), Hamilton `_quaternion_to_matrix`. With S_q = diag(1, f_y f_z, f_x f_z, f_x f_y), the entry signs equal Jay's patterns. Translation sign f_a; log-scales +1; shears f_a f_b on raw K entries (`scale_shear_matrix`). R K ordering (`affine_matrix`) matches the QR re-encoding. |
| Rotation averaging and sign ambiguity | Re-encoding as I − q qᵀ is quadratic in q, so there is no hemisphere problem. Q̄ = I − W; the smallest eigenvector of Q̄ is the dominant eigenvector of W (Markley mean). Projecting back to I − q̄ q̄ᵀ makes `decode_10d` exact (adj of a rank-3 I − qqᵀ ∝ qqᵀ). `rot_gap` / `readout_dev_deg` cover ambiguous cases. eigh is covariant under S_q conjugation, so Claim 2 holds. |
| Centering preserved | The final Center(R̄ A_prev) is at the same place as the non-TTA path. Conjugation by F commutes with `logm_real` / `expm` and the mu = I start (F I F = I). Pairwise maps are gauge-free. |
| SVF unflip | v^F(p) = F v(F p) in voxel units about (n − 1)/2, consistent with `diffeo` pull warps. The common-field gauge cancels exactly because centering is linear (3.2). Integration happens once after averaging, with the checkpoint's T, and P12 is unchanged. The spread σ² = S2/8 − \|c̄\|² is correct. |
| TTA-off bit identity | When off, the call sites are unchanged. The `_v_raw_passes` split is verbatim. `estimate_store_gb(..., tta_levels=())` is unchanged by default. predict.json gains one null key (`tta_record`), as intended. T5 and T9 cover this (NIT 1). |
| Memory at level 1 | GPU: unchanged per-chunk peak (sequential flips). `_f_tta_acc` at chunk 2 is ~1 GB. Host: 28 V B per image = +19.5 GB at N 101 (arithmetic checked), plus the measured 31.7 GB store / 42.8 GB RSS → ~63 GB RSS under 96G. The `affine+deform` source is already counted. P0 converts `input_fn` output with `np.asarray`, so a host-side `flip_np` fits. |
| Time | 8 × per-level model time; the 62 / 35 min totals are consistent with the measured phases (upper bound, P12 counted 8×). |
| Statistics | Conditional scope stated. Paired Wilcoxon + bootstrap CI + Holm over 3 primaries + SESOI. The cluster bootstrap for pairs is adequate for "measurable impact" on this group. |
| Python 3.8 / TF 2.10 | No 3.9+ syntax. numpy `eigh`, `default_rng`; scipy 1.4 `wilcoxon(zero_method="pratt")`, `binom_test`; TF float64 `barycenter` / `encode_10d` work (`to_float` keeps float64). |
| DESIGN MUST COVER 1-9 | All covered: 1 (2.1-2.7; G is not a model output), 2 (3), 3 (4), 4 (6), 5 (5), 6 (7), 7 (9), 8 (8), 9 (12). eval_run / eval_synth TTA moved to OQ 4 (acceptable given the Mindboggle focus). |

## Verified assumptions

- `compose_linear(R_raw, A_prev, c) = c(R_raw @ A_prev)`, so residual = A_f A_prev⁻¹ = mu_f⁻¹ M_f.
- `affine_matrix` = [R @ K | t], with K raw upper-triangular entries and exp diagonal; `encode_10d` = I − q qᵀ via Shepperd.
- `barycenter` starts at mu = I with unrolled iterations; `GroupCentering` is float32 by default, and float64 is possible.
- `StreamedLevel.deform`: P11 stores v_raw and its float32 sum; P12 computes `v = v_raw − mean_v`, `exp_and_inverse`, an optional `compose_cascade`, and frees v_raw afterwards. P0 calls `input_fn` once per chunk and converts it with `np.asarray`.
- Runner `_deform` / `_deform_sel` wrap the last level's `input_fn` to free `SRC_AFFINE` when `i1 == n`. `prev` fields are freed after the level's `eng.deform`.
- `estimate_store_gb` adds `4 * full_vox` for `affine+deform` at every deform level.
- `PREDICT_KEYS` contains `tta`; the CLI has `--tta {none}`; `_peek_chain_configs` uses `cascade.source_config` and then the stage origin's `source_config_path`.
- `eval_mindboggle.sh` takes METHOD / FWD / INV / ATLAS / ATLAS_SEG / CONV; `predict_mindboggle.sh` takes CKPT / NAME / BRANCHES / OUT / EVAL_PAIRS / LEVELS / EXTRA and runs steps 1a-1c, 2 and 3 as described.

## Unverified / wrong assumptions

- Unverified: the outputs of `check_flip_signs.py` and `check_gauge_free.py` (not re-run; the derivations were checked by hand against the code).
- Unverified: the measured timings, peaks and the 39.7 GB baseline estimate from the cluster `predict.json` (not read).
- Unverified: whether every GPU op of the prediction path has a deterministic kernel under `enable_op_determinism` (NIT 1).
- Unverified: the gauge size of real checkpoints. It is irrelevant to correctness now and is measured by `gauge_deg` / `gauge_mm`.
