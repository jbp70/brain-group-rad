# Milestone 1 report: transforms, conventions, spatial transformer

Iteration 1. Nothing committed (the orchestrator commits after review). No cluster or glacier use.

## Files

- `models/__init__.py`, `models/transforms.py`, `models/layers.py`
- `tests/__init__.py`, `tests/tf_setup.py` (log level, GPU memory growth), `tests/np_reference.py` (TF-free numpy/scipy references), `tests/test_transforms.py`, `tests/test_spatial_transformer.py`
- `tests/run_tests_wsl.sh` (py3.8 syntax check + numpy references in `bgr_preproc38`, then the suite in `tf_gpu`)

Run: `python -m unittest discover -s tests -t .` from the repo root.

## Public API

`models/transforms.py` (all batched over leading axes; group axis = -3 of `[..., N, 4, 4]`):
- `to_float(x)`: tensor in float32/float64 (float16, bfloat16, int → float32).
- `identity_rotation_10d() -> np[10]`; `decode_10d(v[..., 10], eps=1e-8) -> [..., 3, 3]`; `encode_10d(R[..., 3, 3]) -> [..., 10]`.
- `identity_rigid_params() -> np[13]`; `identity_scale_shear_params() -> np[6]` (zeros).
- `rigid_matrix(p[..., 13]) -> [..., 4, 4]` = [R | t], p = [10D, t_mm].
- `scale_shear_matrix(p[..., 6]) -> K[..., 3, 3]`, p = [s_x, s_y, s_z, k_xy, k_xz, k_yz], diag exp(s), shears in the strict upper triangle.
- `affine_matrix(p[..., 19]) -> [..., 4, 4]` = [R K | t].
- `to_homogeneous(m)`, `to_3x4(m)`, `assemble_affine(L, t)`, `translation_matrix(t)`.
- `affine_inverse(m, rigid=False)`: block inverse, keeps the 3x4/4x4 layout.
- `compose(*mats) -> [..., 4, 4]` (mats[0] @ mats[1] @ ...).
- `flip_matrix(flip_axes, dtype)`, `flip_conjugate(m, F)` = F m F, `flip_images(images, flip_axes)`.
- `index_to_mm_matrix(shape, spacing)`, `mm_to_index_matrix(shape, spacing)` (numpy 4x4).
- `mm_to_voxel_matrix(m, shape, spacing)` = V⁻¹ M V; `voxel_to_mm_matrix(m, shape, spacing)` = V M V⁻¹.
- `axis_reversal_matrix(dtype)`; `dali_to_tf_matrix(m)` = P M P (alias `tf_to_dali_matrix`).
- `dali_voxel_to_tf_mm(m, shape, spacing, pixel_center=0.5)`, `tf_mm_to_dali_voxel(...)` (UNVERIFIED against DALI).
- `logm_real(a)` (custom gradient, ported), `expm(x)`.
- `barycenter(mats, iters=4) -> [..., 4, 4]`; `center_group(mats, iters=4)` = μ⁻¹ A_i.
- `class GroupCentering(tf.keras.layers.Layer)(iters=4, dtype="float32")`: `call(a_raw[..., N, 4, 4])`.
- `barycentric_targets(T[..., N, 4, 4], iters=10)` = stop_gradient(μ*⁻¹ T_i⁻¹), μ* = barycenter{T_i⁻¹}.
- `pair_targets(T, i_idx, j_idx)` = T_j T_i⁻¹ (stop_gradient); `pair_predictions(A, i_idx, j_idx)` = A_j⁻¹ A_i.
- `brain_second_moment(mask, spacing) -> np[4, 4]` (numpy, float64); `point_loss(dA, S) -> [...]` = tr(dA S dAᵀ).

`models/layers.py`:
- `index_grid(shape, dtype)`, `world_grid(shape, spacing, dtype)`: `[X, Y, Z, 3]`.
- `transform_image(images[N,X,Y,Z,C], matrices[N,4|3,4], spacing, displacement=None, interpolation="linear"|"nearest", fill_value=0.0)`: out_i(y) = in_i(M_i y [+ u_i(y)]).

## Conventions and decisions

Design section 2 is implemented as written, with no deviations: channel-last arrays, batch = group axis, world x = (p − c)·s with c = (n − 1)/2, 4x4 matrices on centered mm coordinates, T_i template → image i (pull with T_i⁻¹), A_i image i → central (pull with A_i⁻¹), pairwise A_j⁻¹ A_i ↔ T_j T_i⁻¹, flips F T F, and M_tf = P M_dali P. All of this is in the module docstrings.

Decisions within the spec:
- **10D representation:** ported with the reference ordering `[a00, a01, a02, a03, a11, a12, a13, a22, a23, a33]` and identity `diag(0, 1, 1, 1)`. Translation is the trailing 3 of the 13, as in the reference `HEAD_WIDTH`. Batched by replacing `[:, i, j]` / `gather(batch_dims=1)` with `[..., i, j]` / one-hot selection. The forward pass is identical to the reference to 1e-6, and `encode_10d` is ported too.
  - Gradient: the decoder uses the adjugate with no eigendecomposition, so there is no eigen-gap degeneracy. The eps = 1e-8 guard is kept and gradients are finite.
  - Two inherited caveats are documented: the argmax column choice is piecewise, and the identity fallback applies at a degenerate adjugate. No extra epsilon was needed.
- **Shear ordering:** `[k_xy, k_xz, k_yz]` = K[0,1], K[0,2], K[1,2]. The shears are added raw (not multiplied by the scales), matching `demo_central_transform.random_affine`.
- **Precision:**
  - float16/bfloat16 matrices or params are cast to float32. float64 stays float64 everywhere, including the sampling grid in `transform_image`. The spec allows float32 or float64, and float64 makes the finite-difference checks precise.
  - `GroupCentering` defaults to `dtype="float32"`. Like any Keras layer it autocasts float64 inputs to its dtype, so pass `dtype="float64"` for float64 work.
- **Out of bounds:** each trilinear corner is clipped and its weight masked, and the missing weight is filled with `fill_value`. This equals padding the volume with the fill value: a soft fall-off over the last voxel that is differentiable at borders and identical on CPU and GPU.
  - scipy 1.4's `mode='constant'` instead returns cval for any sample outside [0, n−1]. The scipy test therefore compares only in-range samples, plus samples more than one voxel outside, which must be exactly 0.
  - Coordinates are clipped to [−2, n+1] before the int cast. That changes nothing numerically, because all corners there are invalid, and it avoids int32 overflow for huge coordinates.
- **Nearest mode:** floor(q + 0.5). Integer labels are gathered as float and cast back (exact for uint8), and the output keeps the input dtype.
- **Displacement:** in voxels of the current level, added to the input-voxel sampling coordinate after the affine.
- **`pair_targets`** is wrapped in stop_gradient, like `barycentric_targets`.
- **DALI pixel center:** assumed at i + 0.5, so the DALI image center is n/2, which is consistent with the reference review's table C. It's exposed as `pixel_center` and marked UNVERIFIED.

Memory, documented in `transform_image`: about 12 + 3·(8 + 4C) B per output voxel per image in the forward pass, and about 12 + 8·(8 + 4C) B under a tape. For a full-res image with C = 1 that's ~0.1 GB forward and ~0.75 GB taped. For 19-channel one-hot labels it's about 4.7 GB taped per image. If that's too much in milestone 6, a recompute-in-backward custom gradient (or `tf.recompute_grad`) is the fix.

## Test results

35 tests, 35 pass, 0 skipped, ~35 s locally (WSL `tf_gpu`: Python 3.9.13, TF 2.10.0, numpy 1.24.3, scipy 1.13.0, GTX 1080). Python 3.8 syntax (py_compile + `ast` feature_version 3.8) and the TF-free numpy references pass in `bgr_preproc38` (Python 3.8.20, numpy 1.21.1, scipy 1.4.1). Full log: `test-log.txt`.

Coverage per spec item:
- **10D:** identity, orthonormality and det, the numpy re-derivation, the reference module (to 1e-6), encode/decode round trip, finite gradients, and an FD gradient check.
- **Builders:** identity (exact), the R·K form, and a QR round trip that recovers positive scales, shears and R.
- **Barycenter vs `demo_central_transform.py`** (rigid and affine, ±45°, ±30 mm):
  - float64 to 1e-9;
  - float32 4-iteration layer: linear part < 1e-4, translations < 1e-3 mm;
  - centered barycenter < 1e-3 mm from I;
  - pairwise maps unchanged (< 5e-6 linear, < 1e-4 mm translation in float32);
  - invariance to common left-multiplication;
  - batched leading axes;
  - FD gradient of GroupCentering + point_loss in float64 (rel < 1e-5).
- **Targets:** central to 1e-8 mm, pair-consistent to 1e-9, no gradient, and `pair_predictions` agrees with `pair_targets`.
- **Flips:** the algebra test, plus the image test (flip ∘ warp_T = warp_{FTF} ∘ flip, < 1e-5, interior).
- **point_loss / S:** match brute-force voxel loops at isotropic and anisotropic spacing.
- **transform_image (a)–(i):**
  - (a) scipy `map_coordinates` to 2e-5;
  - (b) exact integer shift, both as a matrix and as a displacement;
  - (c) round-trip RMS below the RMS change of a 0.5-voxel shift, and the composed matrix ≈ I to 1e-5;
  - (d) nearest mode preserves the label set and dtype;
  - (e) CPU out-of-bounds gives zeros or the fill value, with no error;
  - (f) FD gradient w.r.t. the matrix (rel < 1e-4), and non-None gradients for the matrix and the displacement;
  - (g) N = 2, 3, 5 in `tf.function` with a `None` batch: one trace;
  - (h) mixed_float16 policy: centering stays float32, a float16 warp matches float32, and the policy is restored;
  - (i) multi-level: exact for the identity; for a random affine, RMS < 1% of the range and < 0.25× the error of a 0.5 mm-shifted coarse matrix.
- **DALI:** involution, composition, reversed-axis point action, a voxel ↔ mm round trip, translation and center-rotation meaning (algebraic only).

## Unverified

- **DALI pixel-center convention** (`pixel_center=0.5`, DALI center n/2) and the direction of the emitted DALI matrices. These are algebraic only here; milestone 2 checks them against real `fn.warp_affine` output.
- **Container run:** not run in `tf10_nv.sif` (Python 3.8.10, numpy 1.21.1, scipy 1.4.1). Locally the TF tests ran on Python 3.9 with a newer numpy/scipy; only syntax and the numpy references were checked under 3.8 / numpy 1.21 / scipy 1.4. The orchestrator re-runs the suite in the container.
- **Reference test on the cluster:** `test_matches_reference_implementation` skips there with a recorded reason, because `reference/` is gitignored and absent. Locally it ran and passed.
- **Memory figures** in the `transform_image` docstring are estimates, not measured.
