"""Spatial transformer, grid utilities and group blocks.

Conventions follow `models.transforms` (design section 2): arrays ``[N, X, Y, Z, C]``,
channel-last, batch axis = group axis; voxel index p sits at centered mm coordinate
``(p - c) * s`` with ``c = (n - 1) / 2``; matrices act on centered mm coordinates.
Sampling grids are float32 (float64 if the matrices are float64), never float16.
`sample_voxel_coords` is the shared sampler (zero or border boundary) of `transform_image` and
`models.diffeo`; `upsample_trilinear2x` / `TrilinearUpsample2x` is the half-pixel x2 upsampling.

Group blocks (design 4.1, milestone 3): ``norm -> act -> [concat(x, group mean of x)] -> conv``
(or ``-> Dense`` for vectors). The group is the batch axis: the mean is over axis 0 (one group
per batch), broadcast back to N, so N may be dynamic. Every variable is created by a Keras
sublayer or in build() and used whole in call() (no variable slicing in build(): that was the
groupwise reference's `fast_group_conv_3d` bug, which left every conv kernel without gradient).
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
    return sample_voxel_coords(images, coords, interpolation=interpolation, fill_value=fill_value,
                               boundary="zero")


BOUNDARIES = ("zero", "border")


def sample_voxel_coords(images, coords, interpolation="linear", fill_value=0.0, boundary="zero"):
    """Sample images at per-voxel input coordinates (the shared core of every pull warp).

    out_i[y] = images_i(coords_i[y]) for every output voxel y.

    Args:
      images: [N, X, Y, Z, C] input volumes (static spatial dims; N may be dynamic). Any float
        dtype or integer labels (nearest).
      coords: [N, X', Y', Z', 3] sampling positions in input voxel indices ('ij' order), float32
        or float64 (the compute dtype). The output has the spatial shape of ``coords``.
      interpolation: "linear" (trilinear) or "nearest" (no gradient).
      fill_value: Python number used outside the volume with ``boundary="zero"``.
      boundary: "zero" = the `transform_image` semantics (each out-of-range corner gets weight 0
        and the missing weight is filled with ``fill_value``: soft fall-off over the last voxel).
        "border" = coordinates clipped to [0, n - 1] per axis before interpolation (edge
        replication; a constant volume stays exactly constant). Used for displacement fields.

    Output dtype: floating inputs keep their dtype (blending in the coords dtype); integer and
    bool inputs come back in their dtype in nearest mode. See `transform_image` for memory.
    """
    images = tf.convert_to_tensor(images)
    spatial = _static_spatial_shape(images)
    in_dtype = images.dtype
    if interpolation not in ("linear", "nearest"):
        raise ValueError("interpolation must be 'linear' or 'nearest', got {!r}".format(interpolation))
    if boundary not in BOUNDARIES:
        raise ValueError("boundary must be one of {}, got {!r}".format(BOUNDARIES, boundary))
    coords = tf.convert_to_tensor(coords)
    cdt = coords.dtype
    if cdt not in (tf.float32, tf.float64):
        raise ValueError("coords must be float32 or float64, got {}".format(cdt))
    border = boundary == "border"
    if border:
        upper = tf.constant([float(n - 1) for n in spatial], cdt)
        coords = tf.clip_by_value(coords, tf.zeros_like(upper), upper)

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
        out = sampled if border else tf.where(valid[..., None], sampled, fill)
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
        if border:
            # coords are in [0, n - 1]: i0 is valid; i1 = n only when f == 0 (clipped index).
            w0, w1 = 1.0 - f, f
        else:
            v0 = tf.cast(tf.logical_and(i0 >= 0, i0 <= spatial[a] - 1), cdt)
            v1 = tf.cast(tf.logical_and(i1 >= 0, i1 <= spatial[a] - 1), cdt)
            w0, w1 = (1.0 - f) * v0, f * v1
        corners.append(((tf.clip_by_value(i0, 0, spatial[a] - 1) * strides[a], w0),
                        (tf.clip_by_value(i1, 0, spatial[a] - 1) * strides[a], w1)))

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
    if fill_value != 0 and not border:
        out = out + fill * (1.0 - wsum)[..., None]
    if in_dtype.is_floating:
        out = tf.cast(out, in_dtype)
    return out


# ============================================================================ trilinear x2
def _upsample_axis2x(x, axis):
    """x2 linear upsampling along one spatial axis (half-pixel, edge replicated)."""
    n = int(x.shape[axis])

    def sl(start, stop):
        return x[(slice(None),) * axis + (slice(start, stop),)]

    prev = tf.concat([sl(0, 1), sl(0, n - 1)], axis=axis)          # x[k-1], x[-1] := x[0]
    nxt = tf.concat([sl(1, n), sl(n - 1, n)], axis=axis)            # x[k+1], x[n] := x[n-1]
    even = 0.75 * x + 0.25 * prev                                   # fine 2k:   coarse k - 1/4
    odd = 0.75 * x + 0.25 * nxt                                     # fine 2k+1: coarse k + 1/4
    y = tf.stack([even, odd], axis=axis + 1)                        # [..., n, 2, ...]
    dims = [tf.shape(x)[0]] + [int(v) for v in x.shape[1:]]
    dims[axis] = 2 * n
    return tf.reshape(y, dims)


def upsample_trilinear2x(x):
    """Separable trilinear x2 upsampling of [N, X, Y, Z, C] -> [N, 2X, 2Y, 2Z, C].

    World-consistent "half-pixel" semantics (milestone-6 plan section 0): coarse voxel q sits at
    fine coordinate 2q + 0.5, so fine voxel p samples coarse coordinate (p - 0.5) / 2. Per axis
    out[2k] = 0.75 x[k] + 0.25 x[k-1] and out[2k+1] = 0.75 x[k] + 0.25 x[k+1], with edge
    replication (x[-1] := x[0], x[n] := x[n-1]). This equals
    ``scipy.ndimage.map_coordinates(x, (p - 0.5) / 2, order=1, mode='nearest')`` and
    ``scipy.ndimage.zoom(x, 2, order=1, mode='nearest', grid_mode=True)`` (scipy >= 1.6); it is
    NOT align_corners. Same as tf.image bilinear with half_pixel_centers=True.

    Reshape/stack interleave per axis (no tf.image). Spatial and channel dims must be static; N
    may be dynamic. Computes in the input dtype (float16 features stay float16).
    """
    x = tf.convert_to_tensor(x)
    if x.shape.rank != 5 or any(v is None for v in x.shape[1:].as_list()):
        raise ValueError("upsample_trilinear2x needs [N, X, Y, Z, C] with static X, Y, Z, C, "
                         "got {}".format(x.shape))
    for axis in (1, 2, 3):
        x = _upsample_axis2x(x, axis)
    return x


class TrilinearUpsample2x(tf.keras.layers.Layer):
    """Parameter-free Keras layer wrapping `upsample_trilinear2x` (half-pixel semantics)."""

    def call(self, x):
        return upsample_trilinear2x(x)

    def compute_output_shape(self, input_shape):
        s = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([s[0]] + [None if v is None else 2 * v for v in s[1:4]] + [s[4]])


# ============================================================================ group blocks
NORMS = ("batch", "instance", "layer", "none")
DENSE_BLOCK_NORMS = ("layer", "batch", "none")


class KaimingNormal(tf.keras.initializers.VarianceScaling):
    """He-normal for (leaky) ReLU: Var = 2 / ((1 + slope^2) fan_in), untruncated (rigid project)."""

    def __init__(self, negative_slope=0.0, seed=None):
        super().__init__(scale=2.0 / (1.0 + float(negative_slope) ** 2), mode="fan_in",
                         distribution="untruncated_normal", seed=seed)
        self.negative_slope = float(negative_slope)

    def get_config(self):
        return {"negative_slope": self.negative_slope, "seed": self.seed}


class AffineParams(tf.keras.layers.Layer):
    """Per-channel trainable scale gamma (ones) and shift beta (zeros) on the last axis."""

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.gamma = self.add_weight(name="gamma", shape=(channels,), initializer="ones")
        self.beta = self.add_weight(name="beta", shape=(channels,), initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        return x * self.gamma + self.beta


class BatchNormWithSeparateAffine(tf.keras.layers.Layer):
    """Keras BatchNormalization without its own affine (center=scale=False) + `AffineParams`.

    Rigid-project semantics: the statistics and gamma/beta are separate, so the statistics can be
    frozen (recalibrate, then freeze; design 5.1) while gamma/beta keep training. With
    ``stats_frozen=True`` the BN normalizes with its moving statistics and does not update them,
    even when called with ``training=True``; gamma/beta still get gradients. ``stats_frozen`` is a
    Python attribute read at trace time: re-trace tf.functions after changing it.
    Statistics are over every axis except the last, i.e. over the N images of the one group
    (and their voxels). Under mixed_float16 Keras computes BN in float32 internally.
    """

    def __init__(self, momentum=0.99, epsilon=1e-5, stats_frozen=False, **kwargs):
        super().__init__(**kwargs)
        self.bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=momentum, epsilon=epsilon,
                                                     center=False, scale=False, name="bn")
        self.affine = AffineParams(name="affine")
        self.stats_frozen = bool(stats_frozen)

    def call(self, x, training=None):
        bn_training = False if self.stats_frozen else training
        return self.affine(self.bn(x, training=bn_training))


class InstanceNormWithAffine(tf.keras.layers.Layer):
    """Per-image, per-channel normalization over the spatial axes (float32 math) + `AffineParams`.

    Needs rank >= 3 (spatial axes); no statistics to freeze. tensorflow_addons is not required.
    """

    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)
        self.affine = AffineParams(name="affine")

    def call(self, x, training=None):
        rank = x.shape.rank
        if rank is None or rank < 3:
            raise ValueError("instance norm needs spatial axes, got {}".format(x.shape))
        xf = tf.cast(x, tf.float32)
        mean, var = tf.nn.moments(xf, axes=list(range(1, rank - 1)), keepdims=True)
        y = (xf - mean) * tf.math.rsqrt(var + self.epsilon)
        return self.affine(tf.cast(y, x.dtype))


class LayerNormWithAffine(tf.keras.layers.Layer):
    """Per-sample normalization over the last (feature) axis (float32 math) + `AffineParams`.

    Each vector is normalized by its own mean and variance over its D features, so the output
    of image i does not depend on the other group members, and training and inference compute
    the same function (no statistics). Used on the dense path (design 4.4, v3.4).
    """

    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)
        self.affine = AffineParams(name="affine")

    def call(self, x, training=None):
        xf = tf.cast(x, tf.float32)
        mean, var = tf.nn.moments(xf, axes=[-1], keepdims=True)
        y = (xf - mean) * tf.math.rsqrt(var + self.epsilon)
        return self.affine(tf.cast(y, x.dtype))


def make_norm(norm, bn_momentum=0.99, bn_epsilon=1e-5, name="norm"):
    """'batch' -> BatchNormWithSeparateAffine, 'instance' -> InstanceNormWithAffine,
    'layer' -> LayerNormWithAffine, 'none' -> None."""
    if norm == "batch":
        return BatchNormWithSeparateAffine(momentum=bn_momentum, epsilon=bn_epsilon, name=name)
    if norm == "instance":
        return InstanceNormWithAffine(epsilon=bn_epsilon, name=name)
    if norm == "layer":
        return LayerNormWithAffine(epsilon=bn_epsilon, name=name)
    if norm == "none":
        return None
    raise ValueError("norm must be one of {}, got {!r}".format(NORMS, norm))


def set_bn_stats_frozen(layer, frozen=True):
    """Set `stats_frozen` on every BatchNormWithSeparateAffine inside `layer`. Returns the count."""
    count = 0
    for m in [layer] + list(layer.submodules):
        if isinstance(m, BatchNormWithSeparateAffine):
            m.stats_frozen = bool(frozen)
            count += 1
    return count


def group_mean_concat(x):
    """concat(x_i, mean_j x_j) on the last axis; mean over the group (batch) axis 0, N dynamic."""
    mean = tf.reduce_mean(x, axis=0, keepdims=True)
    return tf.concat([x, tf.broadcast_to(mean, tf.shape(x))], axis=-1)


class _PreActivation(tf.keras.layers.Layer):
    """Shared norm -> act (-> group concat) front end of the group blocks."""

    def _init_pre(self, norm, negative_slope, bn_momentum, bn_epsilon, group, pre_activation):
        self.group = bool(group)
        self.pre_activation = bool(pre_activation)
        self.negative_slope = float(negative_slope)
        self.norm = make_norm(norm, bn_momentum, bn_epsilon) if self.pre_activation else None
        self.act = (tf.keras.layers.ReLU(negative_slope=self.negative_slope, name="act")
                    if self.pre_activation else None)

    def _pre(self, x, training):
        if self.norm is not None:
            x = self.norm(x, training=training)
        if self.act is not None:
            x = self.act(x)
        if self.group:
            x = group_mean_concat(x)
        return x

    def _overhead(self, kernel):
        """Parameters of the kernel half that multiplies the group mean (0 without group)."""
        if not self.group:
            return 0
        shape = [int(v) for v in kernel.shape]
        return int(np.prod(shape[:-2])) * (shape[-2] // 2) * shape[-1]


class GroupConvBlock(_PreActivation):
    """norm -> act -> [concat(x, group mean)] -> Conv3D (design 4.1).

    Args:
      filters, kernel_size, strides: of the Conv3D ('same' padding, bias, Kaiming-normal kernel).
      group: concat the group mean (doubles the conv's input channels). False = plain block.
      norm: 'batch' (BatchNormWithSeparateAffine), 'instance' or 'none'.
      pre_activation: False skips norm and act (the encoder stem: a plain conv on the image).
      negative_slope: LeakyReLU slope (0 = ReLU, the rigid project's default).
    """

    def __init__(self, filters, kernel_size=3, strides=1, group=True, norm="batch",
                 pre_activation=True, negative_slope=0.0, bn_momentum=0.99, bn_epsilon=1e-5,
                 seed=None, **kwargs):
        super().__init__(**kwargs)
        self._init_pre(norm, negative_slope, bn_momentum, bn_epsilon, group, pre_activation)
        self.conv = tf.keras.layers.Conv3D(
            filters, kernel_size, strides=strides, padding="same", use_bias=True,
            kernel_initializer=KaimingNormal(negative_slope, seed), name="conv")

    def call(self, x, training=None):
        return self.conv(self._pre(x, training))

    def group_overhead_params(self):
        return self._overhead(self.conv.kernel)


class GroupResBlock(tf.keras.layers.Layer):
    """Residual variant: two GroupConvBlocks plus a skip (design 4.1; flag, default off).

    out = block2(block1(x)) + skip(x). block1 carries the stride. The skip is the identity when
    channels and stride are unchanged, else a 1x1 Conv3D with the block's stride on the raw input.
    """

    def __init__(self, filters, kernel_size=3, strides=1, group=True, norm="batch",
                 negative_slope=0.0, bn_momentum=0.99, bn_epsilon=1e-5, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.strides = int(strides)
        self.seed = seed
        self.negative_slope = float(negative_slope)
        common = dict(kernel_size=kernel_size, group=group, norm=norm,
                      negative_slope=negative_slope, bn_momentum=bn_momentum,
                      bn_epsilon=bn_epsilon, seed=seed)
        self.block1 = GroupConvBlock(filters, strides=strides, name="block1", **common)
        self.block2 = GroupConvBlock(filters, strides=1, name="block2", **common)
        self.projection = None

    def build(self, input_shape):
        if int(input_shape[-1]) != self.filters or self.strides != 1:
            self.projection = tf.keras.layers.Conv3D(
                self.filters, 1, strides=self.strides, padding="same", use_bias=True,
                kernel_initializer=KaimingNormal(self.negative_slope, self.seed), name="projection")
        super().build(input_shape)

    def call(self, x, training=None):
        skip = x if self.projection is None else self.projection(x)
        return self.block2(self.block1(x, training=training), training=training) + skip

    def group_overhead_params(self):
        return self.block1.group_overhead_params() + self.block2.group_overhead_params()


class GroupDenseBlock(_PreActivation):
    """norm -> act -> [concat(h_i, mean_j h_j)] -> Dense on [N, D] vectors (design 4.4, v3.2).

    The concat makes the Dense W_a h_i + W_b mean(h) + b, i.e. a per-group bias. norm is 'layer'
    (per-vector LayerNormWithAffine, the default), 'batch' (BatchNormWithSeparateAffine over the
    N vectors of the group; ablation only, see `models.config.ModelConfig.dense_norm`) or 'none'
    (activation only).
    """

    def __init__(self, units, group=True, norm="layer", negative_slope=0.0, bn_momentum=0.99,
                 bn_epsilon=1e-5, seed=None, **kwargs):
        if norm not in DENSE_BLOCK_NORMS:
            raise ValueError("dense blocks support norm {}, got {!r}".format(DENSE_BLOCK_NORMS, norm))
        super().__init__(**kwargs)
        self._init_pre(norm, negative_slope, bn_momentum, bn_epsilon, group, True)
        self.dense = tf.keras.layers.Dense(
            units, use_bias=True, kernel_initializer=KaimingNormal(negative_slope, seed),
            name="dense")

    def call(self, x, training=None):
        return self.dense(self._pre(x, training))

    def group_overhead_params(self):
        return self._overhead(self.dense.kernel)


class NormAct(_PreActivation):
    """norm -> act only (no group concat, no weights besides the norm's). norm: any of `NORMS`.

    ``activation=False`` drops the activation (norm only; identity if norm is 'none').
    """

    def __init__(self, norm="layer", negative_slope=0.0, bn_momentum=0.99, bn_epsilon=1e-5,
                 activation=True, **kwargs):
        super().__init__(**kwargs)
        self._init_pre(norm, negative_slope, bn_momentum, bn_epsilon, False, True)
        if not activation:
            self.act = None

    def call(self, x, training=None):
        return self._pre(x, training)
