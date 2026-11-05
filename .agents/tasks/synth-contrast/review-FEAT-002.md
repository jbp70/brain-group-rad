# Review FEAT-002 (synthesis core + post-process), iteration 1

Verdict: APPROVED (0 HIGH, 0 MEDIUM, 1 NIT).

## Evidence checked
- Spec: FEAT-002.json steps 1-6, design 4.1-4.4, 5.2, 6.4, OD1/OD5/OD6.
- `data_loading/params.py`: `SYNTH_SEED_OFFSET 104729`, `SYNTH_DEVICES`, `SYNTH_LABEL_SET_DEFAULT`, `SYNTH_COLUMNS`
  (5 columns), `N_SYNTH_GROUP_UNIFORMS 5`, `SynthConfig` with all design 4.1 defaults plus `hier_dev_cortex 0.25`
  (OD1). No TF import (impl-log TF-free check OK).
- `data_loading/synth.py`:
  - Draw pattern (`draw_synth_params`) is fixed and independent of p and the flag. `test_draw_pattern_fixed` compares
    generator states for p 0 / 0.3 / 1.
  - Mean table is `clip(gather(cls_mean, g) + 2 h_f (dev - 0.5)) * fg`, std table is `gather(std_max cls_std, g) * fg`,
    so id 0 and ids >= 103 are 0.
  - Step 1 uses gathers only. The blur operator uses sigma with the s^2/12 term and is applied with
    `tfa._apply_axis_matrix`. Remask is exactly 1 or exactly the mask (`keep`) and is applied after the blur and again
    after the noise.
  - Bias uses static numpy interpolation matrices with control points at fixed fractions of the extent.
  - `robust_normalise`: stride 4, boolean_mask, dynamic top_k k = max(1, round((1-q) size)), and the
    `tf.cond(size > 0)` guard; `test_empty_mask_finite` covers it.
  - The chain order (pow, noise, clip, remask) matches design 4.2.
  - `maybe_synthesize`: `tf.cond(flag, true_fn, lambda: images)`. With device cpu, the body runs under
    `/cpu:0` inside the true branch. No one_hot anywhere.
- `data_loading/postprocess.py`:
  - `_synth_setup` returns (None, None) for None or prob 0 and validates the device when on.
  - Off paths of both `__call__`s are today's op order: linear `dali_to_mm` stays after the aug when off.
  - `labels` / `synth_override` are keyword-only. A linear call with synthesis on and no labels raises ValueError.
  - Render convention verified against `models/layers.py:45-60`, where `out_i(y) = in_i(M_i y)`. Passing
    `M = affine_inverse(t_mm)` gives I_i(y) = S_i(T_i^-1 y), the DALI pull relation of module step 2. t_mm is float32
    (DALI matrices), so the render grid stays float32.
  - Deform on path maps fine ids to train18 with `tf.gather(fine_to_train18_table(), ...)` before the existing
    cast/squeeze/one_hot(19). The batch dict gains `synth` only when on.
- `tests/runtime_helpers.py`: `fine_labels_from_train18` splits cortex into 4 parcels and hemispheres at the x
  mid-plane. `write_fake_export(fine_labels=False)` default is unchanged because `fine_rng` is None and the image rng
  is untouched.
- Tests (`tests/test_synth.py` with 25 tests, `TestSynthCallSignature` with 2 tests) cover every item of FEAT-002
  step 5:
  - OneHot multiset on the full graph-def including library functions, with and without the tent aug.
  - Negative control (depth 103 in cond/map_fn).
  - Trace once at p 0.5 with both outcomes seen (linear and deform).
  - `synth_override=False` bit-equal to off (rigid, affine, deform, with aug, fixed and drawn flips).
  - Render consistency, with targets bit-equal to the off path.
  - CPU device equals the GPU path, and the traced graph has CPU-placed nodes.
  - Default off builds no generator, and the prob-0 graph equals the synth=None graph.
  - Keyword-only TypeError checks.
- Hard requirements:
  - H1: no model or group-size change.
  - H2: no validation added.
  - H3: one traced graph per task through tf.cond, and one image at a time in map_fn.
  - H4: the off path is unchanged; golden diffs are 0 and the prob-0 graph is identical.
  - No fine one-hot. Python 3.8 ast check passes (62 files).
- Run evidence (impl-log.md, FEAT-002 iteration 1):
  - bgr_preproc38: py_compile, ast (3, 8) and the TF-free import all pass.
  - tf_gpu: `tests.test_synth tests.test_postprocess tests.test_tf_intensity_aug` ran 81 tests, OK.
  - Goldens: `TestLinearOnlyUnchanged` and `TestDeformUnchanged` both report `max |diff| 0`. This is from a separate
    rerun (113 s) after the first combined attempt stalled. The stall happened in the goldens-only script after the
    81-test run in the same process and is not a test failure.
  - I did not rerun the tests: the evidence is specific and consistent with the code.

## Findings

### NIT-1: normalisation test is looser than the spec wording
`tests/test_synth.py` `test_normalisation` asserts `max == 1` and the full-mask p99.5 in (0.9, 1]. The spec says
"p99.5 of the pre-noise image == 1 within subsample tolerance". The test is a reasonable proxy, because the stride-4
subsample at level 4 has only ~1.7 k voxels, and the code path is correct. A tighter check would compare against the
p99.5 of the stride-4 subsample of the pre-clip image. Optional.
