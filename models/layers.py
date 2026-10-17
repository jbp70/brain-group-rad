"""Spatial transformer and grid utilities (group blocks arrive in milestone 3).

Conventions follow `models.transforms` (design section 2): arrays ``[N, X, Y, Z, C]``,
channel-last, batch axis = group axis; voxel index p sits at centered mm coordinate
``(p - c) * s`` with ``c = (n - 1) / 2``; matrices act on centered mm coordinates.
Sampling grids are float32 (float64 if the matrices are float64), never float16.
"""
import numpy as np
import tensorflow as tf

from models import transforms as tfm


def index_grid(shape, dtype=tf.float32):
    """Voxel-index grid [X, Y, Z, 3] ('ij' order: [..., a] is the index along array axis a)."""
    axes = [tf.range(int(n), dtype=dtype) for n in shape]
    return tf.stack(tf.meshgrid(*axes, indexing="ij"), axis=-1)


def world_grid(shape, spacing, dtype=tf.float32):
    """Centered mm grid [X, Y, Z, 3]: (p - (n - 1) / 2) * s."""
    n = np.asarray([int(v) for v in shape], np.float64)
    s = tfm._spacing3(spacing)
    return (index_grid(shape, dtype) - tf.constant((n - 1.0) / 2.0, dtype)) * tf.constant(s, dtype)


def _static_spatial_shape(images):
    shape = images.shape
    if shape.rank != 5:
        raise ValueError("images must be [N, X, Y, Z, C], got {}".format(shape))
    spatial = shape[1:4].as_list()
    if any(v is None for v in spatial):
        raise ValueError("transform_image needs static spatial dims, got {}".format(shape))
    return spatial


def transform_image(images, matrices, spacing, displacement=None, interpolation="linear",
                    fill_value=0.0):
    """Pull-warp images with per-image mm-space affines and an optional displacement field.

    out_i(y) = in_i(M_i y [+ u_i(y)]): for every output voxel y (centered mm coordinates of this
    level), sample input image i at M_i y. Output shape = input shape.

    Args:
      images: [N, X, Y, Z, C], any float dtype (float16 is fine) or integer labels (nearest).
        N may be dynamic; spatial dims must be static.
      matrices: [N, 4, 4] or [N, 3, 4] mm-space matrices (e.g. A_i^-1 to resample image i into
        central space). float16 is cast to float32; float64 keeps the whole grid in float64.
      spacing: this level's voxel spacing in mm (scalar or 3 values).
      displacement: optional [N, X, Y, Z, 3] field in voxels of this level, added to the sampling
        position AFTER the affine: coords = M y + u(y) (in input voxel units). This is NOT the
        design's full map Psi = A^-1 o phi = A^-1 (y + u(y)) unless the linear part of M is the
        identity. For a deformable step with a non-identity affine, either resample into central
        space first and pass an identity matrix here (what the design does), or pass the field
        already mapped by the inverse linear part (L^-1 u).
      interpolation: "linear" (trilinear) or "nearest" (labels; no gradient).
      fill_value: constant for samples outside the volume. Must be a Python number (not a
        tensor): it selects code paths at trace time.

    Out-of-bounds handling uses an explicit validity mask per trilinear corner: indices are
    clipped, then their weights are zeroed, and the missing weight is filled with
    ``fill_value``. This is identical on CPU and GPU (no reliance on GPU gather_nd returning 0)
    and equals padding the volume with ``fill_value`` (soft fall-off over the last voxel).
    Gradients flow to ``matrices``, ``displacement`` and ``images`` (linear mode).

    Memory per call, per image and output voxel (V = X*Y*Z, compute dtype float32): the grid
    and coordinates 12 B (float32 [N, X, Y, Z, 3], plus the displacement if given), per corner a
    flat int32 index (4 B), a weight (4 B) and the gathered values (4*C B). The 8 corners are
    accumulated one at a time, so the forward peak is about 12 + 3*(8 + 4*C) B; under a
    GradientTape the 8 weights and gathered value tensors are kept, about 12 + 8*(8 + 4*C) B.
    Example: one full-res image (6.88 M voxels), C = 1: ~0.1 GB forward, ~0.75 GB taped.
    float16 images are cast to float32 for blending; the output is cast back to the input
    dtype (integer labels in nearest mode come back as their integer dtype).
    """
    images = tf.convert_to_tensor(images)
    spatial = _static_spatial_shape(images)
    in_dtype = images.dtype
    if interpolation not in ("linear", "nearest"):
        raise ValueError("interpolation must be 'linear' or 'nearest', got {!r}".format(interpolation))

    mats = tf.convert_to_tensor(matrices)
    cdt = tf.float64 if mats.dtype == tf.float64 else tf.float32
    mats = tf.cast(mats, cdt)
    m_vox = tfm.mm_to_voxel_matrix(mats, spatial, spacing)          # [N, 3|4, 4], voxel index space

    # Sampling coordinates in input voxel indices: [N, X, Y, Z, 3]
    grid = index_grid(spatial, cdt)
    # NOTE: on Ampere GPUs (A100) TF 2.10 runs float32 matmul/einsum in TensorFloat-32 by
    # default (10-bit mantissa), which puts ~1e-3 errors into these coordinates and into all
    # 4x4 matrix math. Processes must call models.precision.configure_tf_precision() (TF32 off)
    # at startup; mixed_float16 keeps convolutions fast. Verified on glacier (A100).
    coords = tf.einsum("nij,xyzj->nxyzi", m_vox[:, :3, :3], grid) + m_vox[:, None, None, None, :3, 3]
    if displacement is not None:
        coords = coords + tf.cast(displacement, cdt)

    n_img = tf.shape(images)[0]
    channels = images.shape[-1] if images.shape[-1] is not None else tf.shape(images)[-1]
    values = tf.cast(images, cdt)
    flat = tf.reshape(values, [-1, channels])                        # [N*V, C]
    sx = spatial[1] * spatial[2]
    sy = spatial[2]
    vox = spatial[0] * sx
    base = tf.reshape(tf.range(n_img, dtype=tf.int32) * vox, [-1, 1, 1, 1])  # [N, 1, 1, 1]
    strides = (sx, sy, 1)
    fill = tf.cast(fill_value, cdt)

    if interpolation == "nearest":
        rounded = tf.floor(tf.clip_by_value(coords, -2.0, max(spatial) + 1.0) + 0.5)
        idx = base
        valid = None
        for a in range(3):
            ia = tf.cast(rounded[..., a], tf.int32)
            va = tf.logical_and(ia >= 0, ia <= spatial[a] - 1)
            valid = va if valid is None else tf.logical_and(valid, va)
            idx = idx + tf.clip_by_value(ia, 0, spatial[a] - 1) * strides[a]
        sampled = tf.gather(flat, idx)                               # [N, X, Y, Z, C]
        out = tf.where(valid[..., None], sampled, fill)
        if in_dtype.is_floating or in_dtype.is_integer or in_dtype == tf.bool:
            out = tf.cast(out, in_dtype)
        return out

    # Trilinear. Per axis: two corner indices (clipped) and two masked weights.
    lo = tf.floor(tf.clip_by_value(coords, -2.0, max(spatial) + 1.0))
    frac = coords - lo                                               # carries the gradient
    lo_i = tf.cast(lo, tf.int32)
    corners = []                                                     # per axis: [(offset, weight)] x2
    for a in range(3):
        i0 = lo_i[..., a]
        i1 = i0 + 1
        f = frac[..., a]
        v0 = tf.cast(tf.logical_and(i0 >= 0, i0 <= spatial[a] - 1), cdt)
        v1 = tf.cast(tf.logical_and(i1 >= 0, i1 <= spatial[a] - 1), cdt)
        corners.append(((tf.clip_by_value(i0, 0, spatial[a] - 1) * strides[a], (1.0 - f) * v0),
                        (tf.clip_by_value(i1, 0, spatial[a] - 1) * strides[a], f * v1)))

    out = None
    wsum = None
    for ox, wx in corners[0]:
        for oy, wy in corners[1]:
            wxy = wx * wy
            oxy = base + ox + oy
            for oz, wz in corners[2]:
                w = wxy * wz
                term = w[..., None] * tf.gather(flat, oxy + oz)
                out = term if out is None else out + term
                wsum = w if wsum is None else wsum + w
    if fill_value != 0:
        out = out + fill * (1.0 - wsum)[..., None]
    if in_dtype.is_floating:
        out = tf.cast(out, in_dtype)
    return out
