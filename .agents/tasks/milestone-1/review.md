# Milestone 1: transform conventions, group centering and spatial transformer

The change adds `models/transforms.py` and `models/layers.py`, which implement design section 2: centered-mm 4x4 matrices, the batched 10D rotation decoder, R·K affine builders, `logm_real`/`expm`, the barycenter and `GroupCentering`, barycentric and pairwise targets, flips, mm↔voxel and DALI↔TF conversion, the point loss, and a pull-style `transform_image`. It also adds a 35-test unittest suite under `tests/`, checked against scipy `map_coordinates`, `scripts/demo_central_transform.py`, the reference `rotation_representations.py` and brute-force numpy loops. Every convention matches section 2 as written, and the out-of-bounds handling uses a genuine mask. The test log (12:29:13) postdates the last source edit (12:25:56) and shows 35/35 passing, plus a Python 3.8 syntax check.

Watch for:
- The displacement argument composes as `M y + u(y)`, not as `A⁻¹ ∘ φ`. This is harmless while deformable steps use A = I, but it has to be settled before milestone 6 combines the two.
- The TF suite has not yet run in the actual container (Python 3.8 / numpy 1.21 / scipy 1.4).

**Verdict**: APPROVED

## High-level view

All transform math lives in `models/transforms.py`, one function per convention, batched over arbitrary leading axes with the group on axis −3. T_i maps template→image, A_i maps image→central, pairwise is A_j⁻¹A_i ↔ T_j T_i⁻¹, the center is (n−1)/2 and world coordinates are (p−c)·s. Each is implemented exactly as section 2 states and is checked against an independent numpy reference.

Precision is handled at the entry point. `to_float` promotes float16, bfloat16 and integer inputs to float32 and leaves float64 alone. `GroupCentering` pins `dtype="float32"`. A mixed_float16 test confirms that the centering layer's compute dtype stays float32, that float16 head outputs give float32 matrices, and that a float16 image warp matches float32 at a tolerance tight enough to catch a float16 grid. The grid follows the matrix dtype (float64 when the matrices are float64), which slightly relaxes the design's "always float32" wording.

`transform_image` clips every trilinear corner index and multiplies its weight by an explicit validity mask. The missing weight is filled with `fill_value`, so out-of-range behaviour doesn't depend on the device, and a CPU-pinned test exercises it. The displacement is added in input-voxel space after the affine. That diverges from the design's Ψ_i = A_i⁻¹ ∘ φ_i whenever A ≠ I.

Gradients have finite-difference checks in float64 for decode_10d, logm_real, GroupCentering plus the point loss, and transform_image w.r.t. the matrix. They also have non-None checks for the displacement. Only the DALI conversions remain unverified, by design, and are flagged for milestone 2.

<details>
<summary>Issues (4)</summary>

1. **Displacement composition order** — `transform_image` samples at `M y + u(y)`, which equals A⁻¹∘φ only when A = I. Before milestone 6 uses it with a non-identity A, either document that callers must pass an identity matrix (or a field already mapped by L⁻¹), or apply the displacement before the affine. Non-blocking.
2. **Container run outstanding** — the TF suite ran on Python 3.9 / numpy 1.24 / scipy 1.13. Re-run `python -m unittest discover -s tests -t .` inside `tf10_nv.sif` before building on this. Non-blocking (the orchestrator plans this).
3. **Tensor `fill_value` unsupported** — `if fill_value != 0:` is a Python comparison and fails if a tensor is passed inside `tf.function`. Either always add the fill term or document that fill must be a Python scalar. Nit.
4. **float64 grid vs design wording** — the grid runs in float64 when the matrices are float64, while design §2 says "always float32". Either update the design row to "float32 (float64 if matrices are float64), never float16" or force float32. Nit.

</details>

<details>
<summary>Details</summary>

### Convention fidelity against design section 2

Each section 2 row maps to one code path, and none deviates:

- **Index to mm:** `index_to_mm_matrix` builds V with translation −s(n−1)/2, and `mm_to_voxel_matrix` is V⁻¹ M V. The test re-derives V independently in `np_reference.index_to_mm` and point-checks a voxel round trip, including anisotropic spacing.
- **Targets:** `barycentric_targets` computes μ* = barycenter{T_i⁻¹} and returns μ*⁻¹ T_i⁻¹ under stop_gradient (section 3.3).
- **Pairwise:** `pair_targets` is T_j T_i⁻¹ and `pair_predictions` is A_j⁻¹ A_i, both documented as image i → image j, which matches the "Linear-only" clause of the pairwise row.
- **Pull warp:** `transform_image` is out(y) = in(M y), so rendering with T_i⁻¹ and resampling into central space with A_i⁻¹ are both correct uses.
- **Flips:** `flip_images` is `tf.reverse`, which with c = (n−1)/2 is exactly x → −x, and `flip_conjugate` is F M F. The image-level test (render with T then flip, versus flip then render with FTF) passes to 1e-5 on the interior for three flip combinations. That makes it a real check of the center convention, not just the algebra.
- **Barycentric targets test:** this test re-derives centrality using the demo's numpy barycenter. It checks pair consistency against `t_j @ inv(t_i)` computed in numpy, not against the module itself.
- **`pair_predictions` vs `pair_targets`:** these two are compared to each other, which on its own would be circular. It's backed by an independent numpy assertion on `pt[1, 2]` and by the numpy pairwise check on the targets.

### Precision under mixed_float16

`to_float` is the single gate and is called at the top of every matrix builder, `logm_real`, `expm`, `barycenter` and the conversions. `GroupCentering` passes `dtype="float32"` to the Layer constructor, so Keras autocasts float16 inputs up. As a side effect, it also autocasts float64 inputs down unless `dtype="float64"` is passed. That is documented, and the float64 tests do pass it.

`test_h_mixed_float16_policy` sets the global policy and asserts these properties:

- the layer's `compute_dtype` is float32;
- the centered output is float32 and within 1e-3 of the float64 reference;
- `affine_matrix` on float16 params returns float32;
- a float16-image, float32-matrix warp matches float32 to 2e-3.

On a uniform-random volume with coordinates up to about 14, a float16 grid would carry an error of roughly 1e-2, so 2e-3 is discriminating.

`transform_image` sets the compute dtype from the matrices (layers.py:78). It is float64 if the matrices are float64, otherwise float32. That's never float16, but the design table says "always float32" (issue 4).

### Out-of-bounds masking in transform_image

The trilinear path (layers.py:115–140) clips coordinates to [−2, max(n)+1] before the int cast, which avoids int32 overflow for huge coordinates. Each axis then gets two corners, whose indices are clipped to [0, n−1] and whose weights are multiplied by a 0/1 validity mask (layers.py:123–125). The masked weight sum is used to add `fill_value·(1 − wsum)`. Nearest mode uses `tf.where(valid, sampled, fill)` (layers.py:109). Nothing depends on gather returning 0 for out-of-range indices.

`test_e_cpu_out_of_bounds` runs all three modes (linear, nearest, fill = −1) under `tf.device("/CPU:0")` with a 1e7 mm translation, and checks that the output is finite and exactly the fill. The resulting edge behaviour is "pad with fill", which is a soft fall-off over the last voxel. That intentionally differs from scipy 1.4's `mode='constant'`. The scipy comparison accounts for this by checking only in-range samples and samples more than one voxel out, and that comparison is robust across the scipy 1.4 → 1.6 change in constant-mode semantics.

### Displacement composition versus Ψ_i = A_i⁻¹ ∘ φ_i

layers.py:86 adds the displacement after the affine: `coords = M_vox y + u(y)`. The design's full map is Ψ_i = A_i⁻¹ ∘ φ_i with φ_i = id + u_i in central space, which gives Ψ(y) = A⁻¹(y + u(y)) = A⁻¹y + L⁻¹u(y). The two agree only when the linear part L is the identity (**confirmed** by reading the code). Section 6.2 fixes A_i = I in deformable training steps, and section 8 resamples into central space before the deformable cascade, so no current or planned training path is affected. Still, a later caller that passes `matrices=A_i⁻¹` together with `displacement=u_i` to save one resample would silently get the wrong map. The docstring states the formula, but it doesn't flag the difference from Ψ (issue 1).

### Gradients

The float64 finite-difference checks cover:

- `decode_10d` (rel < 1e-6);
- `logm_real` through its custom Fréchet-adjoint gradient (rel < 1e-6);
- `GroupCentering` + `point_loss` over 4 unrolled iterations (rel < 1e-5);
- `transform_image` w.r.t. the matrix (rel < 1e-4).

The displacement gradient is checked for non-None and nonzero, but there is no finite-difference check. It shares the `frac` path with the matrix gradient, so the matrix finite-difference check covers it indirectly. The 10D decoder's piecewise argmax and identity fallback are inherited from the reference and documented. `test_gradients_finite` exercises the identity encoding, where the singular minors would break a `tf.linalg.det`-based gradient.

### Test independence and evidence

The references are independent of the implementation:

- scipy `map_coordinates` for trilinear sampling;
- `demo_central_transform.barycenter` (numpy, 20 iterations) for centering;
- `np_reference.decode_10d_np` (numpy determinants) and the reference TF module for 10D;
- brute-force voxel loops for S and the point loss;
- `scipy.linalg.logm`/`expm`.

The DALI tests are algebraic only and are labelled as such. Python 3.8 syntax was checked with `py_compile` and `ast.parse(feature_version=(3, 8))` under Python 3.8.20, numpy 1.21.1 and scipy 1.4.1, and the TF-free references also ran there. No 3.9+ syntax, einops, or post-1.21 numpy APIs appear in `models/`.

The TF-dependent tests ran under Python 3.9 with numpy 1.24 and scipy 1.13, so the container run is still outstanding (issue 2). `test_matches_reference_implementation` will skip on the cluster with a recorded reason, because `reference/` is gitignored.

The demo module draws from its own module-level seeded RNG, so the transforms a test sees depend on which tests ran before it. The full suite is deterministic, but a single test run in isolation sees different draws. This is a minor reproducibility quirk, not a correctness issue.

</details>

<details>
<summary>File map</summary>

- `models/__init__.py` — package docstring.
- `models/transforms.py` — conventions docstring, dtype gate, 10D rotation encode/decode, rigid/affine builders, inverse/compose, flips, mm↔voxel, DALI↔TF (unverified), `logm_real`/`expm`, barycenter, `GroupCentering`, targets, second moment and point loss.
- `models/layers.py` — `index_grid`, `world_grid`, `transform_image` (masked trilinear/nearest pull warp with optional voxel displacement).
- `tests/np_reference.py` — TF-free numpy references and phantom generators.
- `tests/test_transforms.py` — rotation, builders, helpers, barycenter, targets, flips, point loss, DALI algebra.
- `tests/test_spatial_transformer.py` — transform_image tests (a)–(i) and flip commutation.
- `tests/tf_setup.py`, `tests/__init__.py`, `tests/run_tests_wsl.sh` — test harness.

Full diff: the files are untracked; `git status` and `git diff --no-index` show the new files.

</details>
