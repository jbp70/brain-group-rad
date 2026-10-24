# Design review: Milestone 9 Phase 1 (design-m9-phase1.md v5)

Reviewer pass 5 (fresh read; document review only, no test suite or cluster job run). Inputs: `requirements.md`, `design-m9-phase1.md` v5, `network-design/design.md` v3.13, the v4 review, and the code at HEAD (`models/layers.py` `sample_voxel_coords`).

Verdict: **APPROVED** (0 HIGH, 0 MEDIUM, 5 NIT).

The v4 MEDIUM (F1) is resolved:
- **Backward sequencing.** Every op of corner k is created under `tf.control_dependencies` on corner k − 1's accumulators (d_coords, plus d_flat when built). Ops whose inputs come from inside the block inherit the edge transitively. So corner k's gather cannot be scheduled before corner k − 1's products and segment sum have been consumed. A second effect: the accumulators drive control edges, so Grappler's AddOpsRewrite cannot fold the 8 accumulations into one AddN, which would have kept all 8 contributions alive. The `tf.while_loop(parallel_iterations=1)` fallback is fully specified: bit selection, loop vars, and the d_coords-only carry when `lean_images_grad=False`. It is triggered by a test that discriminates 4–6×.
- **CSE guard.** The `dy` dependency is kept unchanged and matches TF 2.10's `recompute_grad` construction. The `dep` mechanism note is now correct: `cast(NaN, bool)` is True, so `dep` is always 0, and non-finite values propagate through dy · g_k / dy · w_k.
- **Inventory.** The forward transient (84 + 32·C float + 168 B int64, forward moment 252 + 36·C) is in sub-phase Fw at its call site. The sequenced backward transient (88 + 8·C, +16·C with the input gradient) is in Fb and I. The level-1 tables were re-checked against both.
- **Tests.** The graph-mode test asserts lean peak minus inputs ≤ 1.5× the modelled transient in cases A, B and C. It also asserts a measured lean-vs-default saving floor (≥ 0.85 GiB against a predicted 1.65 GiB) on a level-2-size field.

All seven v4 NITs (F2–F8) are resolved. Details are below.

## Findings

### N1 (NIT): inverted wording of the conservative-variant switches in 4.1

Where: 4.1 `predict(...)` bullet, "`predict(..., skip_grad_bound=False, fwd_transient_bound=False)` selects the conservative variants (2.1)".

Problem: the conservative variants are the `True` settings. That is how 2.1, 2.5, 4.2 and 4.6 use them (`fwd_transient_bound=True` gives 41.0 GiB). As written, 4.1 reads as the opposite.

Fix: "`predict(..., skip_grad_bound=False, fwd_transient_bound=False)`; setting either to True selects its conservative variant (2.1)."

### N2 (NIT): U4 is stale for the lean sampler, and the int64 choice has no stated rationale

Where: U4 ("does not affect the lean sampler, which keeps no index tensors"); 2.1 Forward transient (168 B of int64 GPU indices); 3.3 Forward ("computed in int64, which stays on the GPU (U4)").

Problem:
- In v5 the lean sampler does hold GPU index tensors: 168 of the 936 B one-hot forward moment, and 24 + 8 B in the backward. U4 contradicts that.
- The design never says why int64 is used instead of the default path's int32. At `f0d1` N4 the int64 indices are about 4.3 GiB of Fw. Under the order-independent bound they are the difference between 41.0 and about 36.7 GiB.

Fix:
- Reword U4: "...does not count in the GPU peak for the default sampler. The lean sampler computes its indices in int64 on the GPU (2.1), and that transient is modelled explicitly."
- Add one sentence of rationale in 3.3. For example: "int64 because TF registers int32 elementwise GPU kernels with HostMemory. int32 index math would run on the CPU over N·V1 = 2.75·10⁷ rows per call and copy 8 index tensors per call to the device. int32 would also be correct (max N·V1 = 5.5·10⁷ < 2³¹)."
- Optional: list "int32 indices pinned to the GPU" under 3.7 as a further Fw remedy next to forward corner sequencing.

### N3 (NIT): the order-independent forward bound leaves out overlapping transients of other sampler calls

Where: 2.1 Fw "order-independent bound" (counts every other loss term's retained tensors as alive, plus c's own forward moment); 2.5 / 4.4 (`f0d1` LS N4 41.0 GiB).

Problem: U19 says independent branches may overlap. So the template one-hot warp's forward can run concurrently with another lean call's forward, for example the pair one-hot warp at share P/N. That transient (936 · P/N ≈ 234 B at N4, about +1.5 GiB) is not in the "bound". The quantity is therefore an estimate, not a strict bound. It does not change any decision: about 42.5 GiB is still ≤ 44 without the margin, and the measured-N override uses real peaks.

Fix: either add `max over other calls c' of share_c' · forward_moment(c')` to the bound, or rename it "order-independent estimate (one overlapping sampler forward)" and say what it leaves out. Update the 4.6 pin (41.0 ± 0.2) to match whichever is chosen.

### N4 (NIT): `recommendation.json` hides the bound's verdict that step 4 depends on

Where: 4.4 step 6 JSON (`level1.deform_peak_gib` 33.5, `a100.deform_n_no_margin` 4); 6.4 step 4.

Problem: selection uses the timed rule. Under the forward bound, `f0d1` LS N4 is 41.0 / 47.2 GiB: it fails the margin, and the A100 fails even without the margin. 4.4 prints this, but the JSON that the retraining step consumes records only the timed values.

Fix: add `"deform_peak_fwd_bound_gib": 41.0, "deform_peak_fwd_bound_margin_gib": 47.2, "bound_alt_knobs": {"lean_losses": true}` to `level1`, and `"deform_n_no_margin_fwd_bound"` to `a100`, so Jay sees from the JSON alone that the m9 measurement decides between LS and LLS.

### N5 (NIT): stale comment for step 2c in `m9_memory.sh`

Where: 5.5 header, "2c. level 1 deform, cascade, fr50 + lean losses (the N 5-6 alternative), N 5,6,7".

Problem: v5 withdraws the `fr50` N 5–6 alternative, because Fw caps every lean row at N 5 without the margin. Step 2c is still useful: it measures whether Fw really caps `fr50` LLS at N 5 and how the timed rule compares with the bound. The label is out of date.

Fix: "2c. ... fr50 + lean losses (tests the Fw cap: predicted N5 39.6 GiB, N6 47.3 GiB → OOM expected), N 5,6,7". The stub test is unaffected.

## Status of the v4 findings

| v4 finding | Status |
|---|---|
| F1 (MEDIUM) corners unsequenced, forward transient missing | Resolved. Control-dependency chain on the accumulators with a specified `while_loop` fallback (3.3, U19). Fw / Fb sub-phases at the call sites (2.1, 4.2). Tables re-checked (2.5, 4.4). Test cases A / B / C with 1.5× assertions and a saving floor (3.3, 8). |
| F2 (NIT) `dep` NaN mechanism | Resolved (3.3, section 7). |
| F3 (NIT) stale "optional coefficient", unreachable exit 4 | Resolved. The section 7 `fit` row is corrected. Exit 4 is redefined as the gate-bypass check, and the m > 0.30 clause is dropped (4.3, section 7). |
| F4 (NIT) `level1.deform_n` ambiguous | Resolved: `deform_n` = N*, and `n_target` is added to `level1` (4.4 step 5, JSON, 6.4). |
| F5 (NIT) split on the stem | Resolved by exclusion (`pre_activation=False` blocks skipped), with a test (3.5, 8). |
| F6 (NIT) step 5 knobs | Resolved: 5a is the pinned m8 recipe at the reduced widths, 5b is `full` + `$KNOBS`. The stub test checks both calls (5.5). |
| F7 (NIT) JSON written only at the end | Resolved: per-configuration atomic write (`<out>.tmp.<pid>`, fsync, `os.replace`), fatal `OSError`, `test_incremental_write` (5.1, 7, 8). |
| F8 (NIT) deform pre-fit table not pinned | Resolved: six level-2 deform rows within 0.1 GiB, with `B_sg` added consistently (checked: fp32 both 12.3 → 12.5, …, 35.2 → 35.8; AMP both 7.6 → 7.7, …, 27.8 → 28.2), plus the level-1 pins (33.5 D, Fw 32.0, Fb 25.5, bounds 41.0 / 35.0) (4.6). |

## Verified assumptions

- **Corner order and integer math (code read).** Today's linear branch in `layers.sample_voxel_coords` loops x → y → z over (offset, weight) pairs. It accumulates `out` and `wsum` left to right and uses int32 `lo_i`, `base` and offsets. The lean bit mapping k = 4·bx + 2·by + bz (fallback `k // 4 % 2`, `k // 2 % 2`, `k % 2`) reproduces that order. With exact integer index math in int64 the same rows are gathered, so the eager forward max |diff| 0 claim holds. Border mode clips coordinates before `lo` / `frac`, as the backward mask rule assumes. Nearest mode has no gradient, so ignoring `lean` there is correct.
- **dy guard.** The construction (`reduce_max(reshape(dy, [-1])[:1])`, cast to bool, `where(equal(b, b), 0, nan)`) matches TF 2.10 `recompute_grad`'s `dresult_dep`. Only `coords` is perturbed, and every recomputed node (lo, frac, weights, int64 indices, and through `idx_k` every gather) descends from `coords_b`. So DedupComputations has no identical node pair to merge.
- **Sequencing mechanism.** All per-corner ops (`_corner`, the gather, both products, `reduce_sum`, the stack, the segment sum, the accumulator adds) are created inside the control block, and k = 0 depends on the `zeros_like` accumulators. With `images_grad=False` no d_flat op exists, so no unprunable branch is created. This is the reason for `lean_images_grad`, which defaults to True (always correct). The 1.5× tolerance in case C covers the omitted per-axis validity terms (≈ 24 B) and the small per-corner temporaries: 240 → ≈ 265 B against a 360 B ceiling.
- **Transient arithmetic.**
  - T_fwd(19) = 84 + 608 + 168 = 860, plus the 76 B output gives 936 B.
  - T_bwd: 88 + 152 = 240 (one-hot), 88 + 24 + 48 = 160 (field).
  - `f0d1` Fw = 172 + 55 + 48 + 936 = 1211 B, i.e. 7.76 GiB per image, i.e. 32.0 GiB at N4.
  - Fb = F − 54 B gives 25.5 GiB.
  - D = 1270 B gives 33.5 GiB, so D binds.
  - Under the bound: 1562 B gives 41.0 / 47.2 GiB, and `f0d1` LLS 1326 B gives 35.0 / 40.3 GiB.
- **Saving claim.** Case B: lean 516 B (6 × 24 retained + 12 + the 360 B forward moment) against default 1546 B (7 × 180 + 12 + 12 + 12 + 250). That is a saving of 1030 B at V2 = 860 160 and N = 2, i.e. 1.65 GiB. The ≥ 0.85 GiB floor subtracts both allowances, so it is a fair discriminator against the v3 F1 failure mode.
- **Recommendation.** It is unchanged by v5 and robust to the open ordering question. `f0d1` LS N4 is ≤ 44 GiB under the timed rule (33.5) and under the bound (41.0). The margin case under the bound has a named fallback (`f0d1` LLS, 40.3 GiB), measured by job step 3. The measured-N override in 6.4 step 4 uses real peaks.
- **design.md v3.13.** It matches: status line, section 4.5 (`full`, the lean sampler with int64 indices, the control-dependency chain, `lean_images_grad`), section 9 (Fw / Fb, the withdrawn `fr50` / `fr25` alternatives, the replaced graph-mode test numbers 2.25 / 0.58 / 0.41 / 0.85 / 1.65 GiB) and the section 11 milestone-9 row.
- **Requirements coverage.**
  - R2.4 tolerances: the forward is bit-identical, gradients ≤ 1e-4 (target 1e-6), and the default path is bit-identical.
  - R2.5: the goldens are untouched by construction.
  - R3.6 / R3.7: the target-N rule change is justified by Jay's message 12, and exit 3 gives the R3.7 verdict.
  - R4.3: script header and stub test.
  - N3: ≈ 16 min added.
  - Out of scope: label sampling, TTA and cluster actions stay excluded.

## Unverified or wrong assumptions

- **Unverified, as the design states (U19):** that the TF 2.10 executor and Grappler's dependency optimizer keep the control chain. This is reasonable: the edges are not implied by data paths, and AddOpsRewrite skips control-driven nodes. Case C settles it, with the `while_loop` fallback.
- **Unverified (U18):** that DedupComputations would merge the backward with the forward without the guard. The design prints the case-B numbers without `dep` for information.
- **Unverified (U4):** that int32 elementwise kernels are host-resident. U4 is stale for the lean sampler (N2).
- **Approximation:** the order-independent forward bound leaves out an overlapping second sampler forward (N3). No decision changes.
- **Not re-checked in this pass:** the TF 2.10 `recompute_grad` source, which was relied on from earlier reviews and matches the construction quoted. Also U1, U2, U7, U10, U11, U14–U17 as listed in the design, each with a planned check.
