"""Transform conventions and linear-transform math for groupwise registration.

This module is the code form of design section 2 (`.agents/tasks/network-design/design.md`).

Conventions
-----------
* Arrays are ``[..., X, Y, Z, C]``, channel-last. Inside a train step the batch axis is the
  group axis N (one group per batch), so a group of transforms is ``[N, 4, 4]``. Every
  function here is batched over arbitrary leading axes; group reductions use axis -3 of a
  ``[..., N, 4, 4]`` tensor.
* World / mm coordinates: for a level with voxel spacing ``s`` (4, 2 or 1 mm) and shape ``n``,
  voxel index ``p`` has world coordinate ``x = (p - c) * s`` with ``c = (n - 1) / 2`` per axis.
  The origin is the image center, which is also the rotation center. Block averaging by 2 or 4
  maps coarse voxel centers onto the same world points, so one mm matrix applies at every level.
* Matrices are 4x4 homogeneous and act on centered mm coordinates (column vectors,
  ``x' = M @ [x, 1]``). They are sliced to 3x4 only for warping.
  `mm_to_voxel_matrix` / `voxel_to_mm_matrix` convert to and from voxel-index space.
* Augmentation ``T_i`` maps template points into image i. Images are rendered by pull:
  ``I_i(y) = I(T_i^-1 y)``.
* Network output ``A_i`` maps image-i coordinates to central space. Image i resampled into
  central space is a pull with ``A_i^-1``.
* Pairwise: ``A_j^-1 A_i`` maps image i to image j. Its true value is ``T_j T_i^-1``.
* Flip ``F = diag(f_x, f_y, f_z, 1)`` with ``f = -1`` on flipped axes (centered mm coordinates).
  Flipping every image of a group (``tf.reverse`` on those array axes) turns each ``T_i`` into
  ``F T_i F``.
* DALI uses the reversed axis order: ``M_tf = P M_dali P`` with P the axis-reversal permutation.

Precision: everything that builds matrices, logm/expm or barycenters runs in float32 or
float64. float16/bfloat16 inputs are cast to float32 first, so these functions are safe under a
``mixed_float16`` Keras policy. `GroupCentering` sets ``dtype='float32'`` explicitly.

Dependencies: tensorflow and numpy only (Python 3.8 / TF 2.10 compatible).
"""
import numpy as np
import tensorflow as tf

ROTATION_DIMS = 10       # 10D rotation representation
RIGID_PARAMS = 13        # 10D rotation + 3 translations (mm)
SCALE_SHEAR_PARAMS = 6   # 3 log-scales + 3 shears
AFFINE_PARAMS = RIGID_PARAMS + SCALE_SHEAR_PARAMS


# ---------------------------------------------------------------------------- dtype helpers
def to_float(x):
    """Convert to a float32/float64 tensor. float16, bfloat16 and integer inputs become float32."""
    x = tf.convert_to_tensor(x)
    if x.dtype in (tf.float32, tf.float64):
        return x
    return tf.cast(x, tf.float32)


def _spacing3(spacing):
    """Scalar or length-3 spacing -> float64 numpy array of shape (3,)."""
    s = np.asarray(spacing, dtype=np.float64).reshape(-1)
    if s.size == 1:
        s = np.repeat(s, 3)
    if s.size != 3:
        raise ValueError("spacing must be a scalar or have 3 entries, got {!r}".format(spacing))
    return s


def _shape3(shape):
    s = [int(v) for v in shape]
    if len(s) != 3:
        raise ValueError("shape must have 3 spatial entries, got {!r}".format(shape))
    return s


# ---------------------------------------------------------------------------- 10D rotation
# Ported from the rigid project's models/rotation_representations.py (Lin/Hanson/Hanson,
# ICCV 2023), generalized from [B, ...] to arbitrary leading axes.
#
# The 10 values are the upper-triangular entries, row-major, of a symmetric 4x4 matrix A:
#     [a00, a01, a02, a03, a11, a12, a13, a22, a23, a33]
# The quaternion q = (w, x, y, z) is the eigenvector of A's smallest eigenvalue. It is read
# sign-free from the adjugate adj(A) (rank 1 and proportional to q q^T when that eigenvalue is
# ~0): take the adjugate column with the largest diagonal entry and normalize it.
# Identity encoding: diag(0, 1, 1, 1) -> [0, 0, 0, 0, 1, 0, 0, 1, 0, 1] -> q = (1, 0, 0, 0).
#
# Gradient behaviour: there is no eigendecomposition, so no eigen-gap degeneracy. Minors use an
# explicit cofactor polynomial (tf.linalg.det's gradient needs an inverse and fails on the
# singular minors of the identity encoding). The gradient is finite everywhere except that
# (i) the column choice (argmax) is piecewise, so the output can jump where two adjugate
# diagonals tie for a non-rank-1 A, and (ii) when the selected column's norm is <= eps the decode
# falls back to the identity quaternion with zero gradient. Both are inherited unchanged from
# the validated reference; the eps = 1e-8 guard is kept.

def identity_rotation_10d():
    """10D encoding of the identity rotation (float32, shape [10])."""
    return np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)


def _build_symmetric_4x4(v):
    a00, a01, a02, a03, a11, a12, a13, a22, a23, a33 = tf.unstack(v, axis=-1)
    row0 = tf.stack([a00, a01, a02, a03], axis=-1)
    row1 = tf.stack([a01, a11, a12, a13], axis=-1)
    row2 = tf.stack([a02, a12, a22, a23], axis=-1)
    row3 = tf.stack([a03, a13, a23, a33], axis=-1)
    return tf.stack([row0, row1, row2, row3], axis=-2)


def _det_3x3(m):
    """Explicit cofactor determinant (polynomial gradient, no matrix inverse)."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    return (m00 * (m11 * m22 - m12 * m21)
            - m01 * (m10 * m22 - m12 * m20)
            + m02 * (m10 * m21 - m11 * m20))


def _adjugate_4x4(a):
    """Classical adjugate of [..., 4, 4] matrices from explicit cofactors (finite when singular)."""
    idx = [0, 1, 2, 3]
    rows = []
    for i in idx:
        sub_rows = tf.gather(a, [r for r in idx if r != i], axis=-2)  # [..., 3, 4]
        row = []
        for j in idx:
            minor = _det_3x3(tf.gather(sub_rows, [c for c in idx if c != j], axis=-1))
            row.append(minor if (i + j) % 2 == 0 else -minor)
        rows.append(tf.stack(row, axis=-1))
    # cofactor matrix C_ij; A symmetric -> adj = C^T = C
    return tf.stack(rows, axis=-2)


def _quaternion_to_matrix(q):
    """Unit quaternions (w, x, y, z), Hamilton convention -> [..., 3, 3] rotations."""
    w, x, y, z = tf.unstack(q, axis=-1)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - w * x)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return tf.stack([tf.stack([r00, r01, r02], axis=-1),
                     tf.stack([r10, r11, r12], axis=-1),
                     tf.stack([r20, r21, r22], axis=-1)], axis=-2)


def decode_10d(v, eps=1e-8):
    """10D representation [..., 10] -> rotation matrices [..., 3, 3] (float32/64).

    See the block comment above for the entry ordering, identity encoding and gradient notes.
    """
    v = to_float(v)
    adj = _adjugate_4x4(_build_symmetric_4x4(v))
    k = tf.argmax(tf.linalg.diag_part(adj), axis=-1)                    # [...]
    sel = tf.one_hot(k, 4, dtype=v.dtype)                               # [..., 4]
    q = tf.reduce_sum(adj * sel[..., None, :], axis=-1)                 # column k: [..., 4]
    norm = tf.norm(q, axis=-1, keepdims=True)
    identity_q = tf.concat([tf.ones_like(norm), tf.zeros_like(norm),
                            tf.zeros_like(norm), tf.zeros_like(norm)], axis=-1)
    q = tf.where(norm > eps, q, identity_q)
    q = q / tf.maximum(tf.norm(q, axis=-1, keepdims=True), eps)
    return _quaternion_to_matrix(q)


def _matrix_to_quaternion(r):
    """Rotations [..., 3, 3] -> unit quaternions (w, x, y, z) (branchless Shepperd method)."""
    r00, r01, r02 = r[..., 0, 0], r[..., 0, 1], r[..., 0, 2]
    r10, r11, r12 = r[..., 1, 0], r[..., 1, 1], r[..., 1, 2]
    r20, r21, r22 = r[..., 2, 0], r[..., 2, 1], r[..., 2, 2]
    zero = tf.zeros_like(r00)
    trace = r00 + r11 + r22
    s_w = tf.sqrt(tf.maximum(zero, 1.0 + trace))
    s_x = tf.sqrt(tf.maximum(zero, 1.0 + r00 - r11 - r22))
    s_y = tf.sqrt(tf.maximum(zero, 1.0 - r00 + r11 - r22))
    s_z = tf.sqrt(tf.maximum(zero, 1.0 - r00 - r11 + r22))
    eps = tf.constant(1e-12, dtype=r.dtype)
    inv_w = 0.5 / tf.maximum(s_w, eps)
    inv_x = 0.5 / tf.maximum(s_x, eps)
    inv_y = 0.5 / tf.maximum(s_y, eps)
    inv_z = 0.5 / tf.maximum(s_z, eps)
    q_w = tf.stack([0.5 * s_w, (r21 - r12) * inv_w, (r02 - r20) * inv_w, (r10 - r01) * inv_w], -1)
    q_x = tf.stack([(r21 - r12) * inv_x, 0.5 * s_x, (r01 + r10) * inv_x, (r02 + r20) * inv_x], -1)
    q_y = tf.stack([(r02 - r20) * inv_y, (r01 + r10) * inv_y, 0.5 * s_y, (r12 + r21) * inv_y], -1)
    q_z = tf.stack([(r10 - r01) * inv_z, (r02 + r20) * inv_z, (r12 + r21) * inv_z, 0.5 * s_z], -1)
    branch = tf.argmax(tf.stack([trace, r00, r11, r22], axis=-1), axis=-1)
    sel = tf.one_hot(branch, 4, dtype=r.dtype)                          # [..., 4]
    branches = tf.stack([q_w, q_x, q_y, q_z], axis=-2)                  # [..., 4(branch), 4]
    return tf.reduce_sum(branches * sel[..., :, None], axis=-2)


def encode_10d(r):
    """Rotations [..., 3, 3] -> 10D encoding of A = I - q q^T (so decode_10d(encode_10d(R)) = R)."""
    r = to_float(r)
    q = _matrix_to_quaternion(r)
    a = tf.eye(4, dtype=r.dtype) - q[..., :, None] * q[..., None, :]
    return tf.stack([a[..., 0, 0], a[..., 0, 1], a[..., 0, 2], a[..., 0, 3], a[..., 1, 1],
                     a[..., 1, 2], a[..., 1, 3], a[..., 2, 2], a[..., 2, 3], a[..., 3, 3]], axis=-1)


# ---------------------------------------------------------------------------- matrix helpers
def to_homogeneous(m):
    """[..., 3, 4] -> [..., 4, 4] by appending [0, 0, 0, 1]. 4x4 inputs are returned unchanged."""
    m = to_float(m)
    if m.shape[-2] == 4:
        return m
    bottom = tf.concat([tf.zeros_like(m[..., :1, :3]), tf.ones_like(m[..., :1, :1])], axis=-1)
    return tf.concat([m, bottom], axis=-2)


def to_3x4(m):
    """[..., 4, 4] (or [..., 3, 4]) -> [..., 3, 4]."""
    m = to_float(m)
    return m[..., :3, :]


def assemble_affine(linear, translation):
    """Linear part [..., 3, 3] and translation [..., 3] -> [..., 4, 4]."""
    return to_homogeneous(tf.concat([linear, translation[..., None]], axis=-1))


def affine_inverse(m, rigid=False):
    """Inverse of homogeneous affines via the block structure: [L | t]^-1 = [L^-1 | -L^-1 t].

    Accepts [..., 4, 4] or [..., 3, 4] and returns the same layout. Only the 3x3 block is inverted
    (transposed when ``rigid=True``), which is cheaper and better conditioned than a 4x4 inverse.
    """
    m = to_float(m)
    lin = m[..., :3, :3]
    t = m[..., :3, 3:]
    lin_inv = tf.linalg.matrix_transpose(lin) if rigid else tf.linalg.inv(lin)
    out = tf.concat([lin_inv, -tf.matmul(lin_inv, t)], axis=-1)
    return to_homogeneous(out) if m.shape[-2] == 4 else out


def compose(*mats):
    """Matrix product mats[0] @ mats[1] @ ... (applied right to left). Returns [..., 4, 4]."""
    if not mats:
        raise ValueError("compose needs at least one matrix")
    out = to_homogeneous(mats[0])
    for m in mats[1:]:
        out = tf.matmul(out, to_homogeneous(m))
    return out


def translation_matrix(t):
    """Translations [..., 3] (mm) -> [..., 4, 4]."""
    t = to_float(t)
    lin = tf.zeros_like(t[..., :, None] * t[..., None, :]) + tf.eye(3, dtype=t.dtype)
    return assemble_affine(lin, t)


# ---------------------------------------------------------------------------- builders
def identity_rigid_params():
    """Head bias that decodes to the identity: 10D identity + zero translation (float32, [13])."""
    return np.concatenate([identity_rotation_10d(), np.zeros(3, np.float32)])


def identity_scale_shear_params():
    """Scale/shear branch bias that decodes to K = I: zeros (float32, [6])."""
    return np.zeros(SCALE_SHEAR_PARAMS, np.float32)


def rigid_matrix(params):
    """Rigid head output [..., 13] = [10D rotation, t_x, t_y, t_z (mm)] -> [R | t] as [..., 4, 4]."""
    params = to_float(params)
    return assemble_affine(decode_10d(params[..., :ROTATION_DIMS]), params[..., ROTATION_DIMS:])


def scale_shear_matrix(params):
    """Scale/shear branch output [..., 6] -> upper-triangular K [..., 3, 3].

    Ordering: ``[s_x, s_y, s_z, k_xy, k_xz, k_yz]``. The diagonal is ``exp(s)`` (log-scales) and
    the shears fill the strict upper triangle: ``K = [[e^s_x, k_xy, k_xz], [0, e^s_y, k_yz],
    [0, 0, e^s_z]]``. Zeros give exactly K = I.
    """
    params = to_float(params)
    sx, sy, sz, kxy, kxz, kyz = tf.unstack(params, axis=-1)
    zero = tf.zeros_like(sx)
    return tf.stack([tf.stack([tf.exp(sx), kxy, kxz], axis=-1),
                     tf.stack([zero, tf.exp(sy), kyz], axis=-1),
                     tf.stack([zero, zero, tf.exp(sz)], axis=-1)], axis=-2)


def affine_matrix(params):
    """Affine params [..., 19] = [rigid 13, scale/shear 6] -> [R K | t] as [..., 4, 4].

    Covers every orientation-preserving affine (QR factorization with positive diagonal), is
    always invertible, and the identity params (`identity_rigid_params` + zeros) give exactly I.
    """
    params = to_float(params)
    rot = decode_10d(params[..., :ROTATION_DIMS])
    t = params[..., ROTATION_DIMS:RIGID_PARAMS]
    k = scale_shear_matrix(params[..., RIGID_PARAMS:AFFINE_PARAMS])
    return assemble_affine(tf.matmul(rot, k), t)


# ---------------------------------------------------------------------------- flips
def flip_matrix(flip_axes, dtype=tf.float32):
    """Booleans for (x, y, z) -> F = diag(+-1, +-1, +-1, 1) with -1 on flipped axes."""
    flags = [bool(f) for f in flip_axes]
    if len(flags) != 3:
        raise ValueError("flip_axes needs 3 entries")
    return tf.constant(np.diag([-1.0 if f else 1.0 for f in flags] + [1.0]), dtype=dtype)


def flip_conjugate(m, flip):
    """F m F for a flip matrix F (its own inverse). m: [..., 4, 4] or [..., 3, 4] -> [..., 4, 4]."""
    m = to_homogeneous(m)
    flip = tf.cast(flip, m.dtype)
    return tf.matmul(tf.matmul(flip, m), flip)


def flip_images(images, flip_axes):
    """Reverse the flagged spatial axes of [N, X, Y, Z, C] images (array axes 1, 2, 3).

    With the center at (n - 1) / 2 this is exactly x -> -x in centered mm coordinates.
    """
    axes = [i + 1 for i, f in enumerate(flip_axes) if f]
    return tf.reverse(images, axis=axes) if axes else images


# ---------------------------------------------------------------------------- mm <-> voxel
def index_to_mm_matrix(shape, spacing, dtype=np.float64):
    """V with x_mm = V @ [p, 1]: x = (p - c) * s, c = (n - 1) / 2 (numpy, [4, 4])."""
    n = np.asarray(_shape3(shape), np.float64)
    s = _spacing3(spacing)
    v = np.eye(4)
    v[:3, :3] = np.diag(s)
    v[:3, 3] = -s * (n - 1.0) / 2.0
    return v.astype(dtype)


def mm_to_index_matrix(shape, spacing, dtype=np.float64):
    """V^-1 with p = V^-1 @ [x_mm, 1] (numpy, [4, 4])."""
    n = np.asarray(_shape3(shape), np.float64)
    s = _spacing3(spacing)
    v = np.eye(4)
    v[:3, :3] = np.diag(1.0 / s)
    v[:3, 3] = (n - 1.0) / 2.0
    return v.astype(dtype)


def _keep_layout(out, like):
    return out if like.shape[-2] == 4 else out[..., :3, :]


def mm_to_voxel_matrix(m_mm, shape, spacing):
    """Centered-mm matrix -> voxel-index matrix V^-1 M V (same layout as the input)."""
    m = to_float(m_mm)
    v = tf.constant(index_to_mm_matrix(shape, spacing), m.dtype)
    v_inv = tf.constant(mm_to_index_matrix(shape, spacing), m.dtype)
    return _keep_layout(tf.matmul(tf.matmul(v_inv, to_homogeneous(m)), v), m)


def voxel_to_mm_matrix(m_vox, shape, spacing):
    """Voxel-index matrix -> centered-mm matrix V M V^-1 (same layout as the input)."""
    m = to_float(m_vox)
    v = tf.constant(index_to_mm_matrix(shape, spacing), m.dtype)
    v_inv = tf.constant(mm_to_index_matrix(shape, spacing), m.dtype)
    return _keep_layout(tf.matmul(tf.matmul(v, to_homogeneous(m)), v_inv), m)


# ---------------------------------------------------------------------------- DALI <-> TF
# UNVERIFIED against DALI: these are algebraic conversions only. P M P matches the rigid
# project's construction (reference review A1), but DALI's pixel-center convention (assumed:
# pixel i covers [i, i + 1), center at i + 0.5, so the image center is at n / 2) is checked
# against real fn.warp_affine output in milestone 2.

def axis_reversal_matrix(dtype=tf.float32):
    """P: homogeneous 4x4 permutation that reverses the 3 spatial axes."""
    p = np.zeros((4, 4))
    p[0, 2] = p[1, 1] = p[2, 0] = p[3, 3] = 1.0
    return tf.constant(p, dtype=dtype)


def dali_to_tf_matrix(m):
    """M_tf = P M_dali P (axis order only; units and origin unchanged). Same layout as input.

    P is its own inverse, so this is also the TF -> DALI conversion (`tf_to_dali_matrix`).
    UNVERIFIED against DALI (algebraic consistency only, see the note above).
    """
    m = to_float(m)
    p = axis_reversal_matrix(m.dtype)
    return _keep_layout(tf.matmul(tf.matmul(p, to_homogeneous(m)), p), m)


tf_to_dali_matrix = dali_to_tf_matrix


def _dali_to_mm_np(shape, spacing, pixel_center):
    """W with x_mm = W @ [u, 1], u = DALI continuous voxel coordinate in TF axis order.

    u = p + pixel_center for voxel index p, so x = s * (u - pixel_center - (n - 1) / 2).
    """
    n = np.asarray(_shape3(shape), np.float64)
    s = _spacing3(spacing)
    w = np.eye(4)
    w[:3, :3] = np.diag(s)
    w[:3, 3] = -s * (pixel_center + (n - 1.0) / 2.0)
    return w


def dali_voxel_to_tf_mm(m_dali, shape, spacing, pixel_center=0.5):
    """DALI voxel matrix (reversed axes, DALI pixel coordinates) -> TF centered-mm matrix.

    ``shape`` is the TF-order spatial shape (X, Y, Z). The matrix's meaning (e.g. the
    destination -> source map T_i^-1 used with ``warp_affine(inverse_map=True)``) is unchanged;
    only coordinates are converted: M_mm = W (P M_dali P) W^-1.
    UNVERIFIED against DALI: ``pixel_center=0.5`` is an assumption checked in milestone 2.
    """
    m = to_float(m_dali)
    w_np = _dali_to_mm_np(shape, spacing, pixel_center)
    w = tf.constant(w_np, m.dtype)
    w_inv = tf.constant(np.linalg.inv(w_np), m.dtype)
    out = tf.matmul(tf.matmul(w, to_homogeneous(dali_to_tf_matrix(m))), w_inv)
    return _keep_layout(out, m)


def tf_mm_to_dali_voxel(m_mm, shape, spacing, pixel_center=0.5):
    """Inverse of `dali_voxel_to_tf_mm`. UNVERIFIED against DALI."""
    m = to_float(m_mm)
    w_np = _dali_to_mm_np(shape, spacing, pixel_center)
    w = tf.constant(w_np, m.dtype)
    w_inv = tf.constant(np.linalg.inv(w_np), m.dtype)
    out = dali_to_tf_matrix(tf.matmul(tf.matmul(w_inv, to_homogeneous(m)), w))
    return _keep_layout(out, m)


# ---------------------------------------------------------------------------- logm / expm
@tf.custom_gradient
def _logm_real(a):
    out = tf.cast(tf.math.real(tf.linalg.logm(tf.cast(a, tf.complex128))), a.dtype)

    def grad(g):
        n = tf.shape(a)[-1]
        at = tf.linalg.matrix_transpose(a)
        big = tf.concat([tf.concat([at, g], axis=-1),
                         tf.concat([tf.zeros_like(a), at], axis=-1)], axis=-2)
        lbig = tf.math.real(tf.linalg.logm(tf.cast(big, tf.complex128)))
        return tf.cast(lbig[..., :n, n:], a.dtype)

    return out, grad


def logm_real(a):
    """Differentiable real matrix logarithm for real matrices without negative eigenvalues.

    Ported from scripts/check_tf_group_centering.py. TF 2.10's tf.linalg.logm is complex-only and
    has no gradient. Forward: complex128 logm, real part. Backward: the adjoint Frechet
    derivative, i.e. the upper-right block of log([[A^T, G], [0, A^T]]) (Higham). Computed in
    complex128 internally; returns float32/float64 (float16 inputs are cast to float32).
    """
    return _logm_real(to_float(a))


def expm(x):
    """Matrix exponential (tf.linalg.expm, differentiable) in float32/float64."""
    return tf.linalg.expm(to_float(x))


# ---------------------------------------------------------------------------- barycenter
def barycenter(mats, iters=4):
    """Group exponential barycenter of [..., N, 4, 4] affines over the group axis (-3).

    Unrolled fixed point mu <- mu @ expm(mean_i logm(mu^-1 A_i)), starting at mu = I (iteration
    1 is the log-Euclidean mean). Differentiable through every iteration. Returns [..., 4, 4].
    4 iterations leave the center ~1e-5 mm off at +-45 deg / +-30 mm; use ~10 to converge.
    """
    mats = to_float(mats)
    mu = tf.zeros_like(mats[..., 0, :, :]) + tf.eye(4, dtype=mats.dtype)
    for _ in range(int(iters)):
        x = tf.reduce_mean(logm_real(tf.matmul(affine_inverse(mu)[..., None, :, :], mats)), axis=-3)
        mu = tf.matmul(mu, expm(x))
    return mu


def center_group(mats, iters=4):
    """mu^-1 A_i with mu = barycenter(A): the centered group (pairwise maps unchanged)."""
    mats = to_float(mats)
    return tf.matmul(affine_inverse(barycenter(mats, iters))[..., None, :, :], mats)


class GroupCentering(tf.keras.layers.Layer):
    """Left-multiply each group of transforms by the inverse of its barycenter.

    Input ``[..., N, 4, 4]`` raw predictions A_raw; output A_i = mu^-1 A_raw_i, with gradients
    through mu. Ported from scripts/check_tf_group_centering.py. Defaults to ``dtype='float32'``
    so a ``mixed_float16`` policy cannot push the logm/expm math into float16 (float16 inputs are
    autocast to float32). Pass ``dtype='float64'`` for float64 work.
    """

    def __init__(self, iters=4, dtype="float32", **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.iters = int(iters)

    def call(self, a_raw):
        return center_group(a_raw, self.iters)

    def get_config(self):
        config = super().get_config()
        config.update({"iters": self.iters})
        return config


# ---------------------------------------------------------------------------- targets
def barycentric_targets(t_aug, iters=10):
    """Central targets A_i* = mu*^-1 T_i^-1 with mu* = barycenter{T_i^-1}, from T_i [..., N, 4, 4].

    Run to convergence (default 10 iterations) and wrapped in tf.stop_gradient: these are
    targets, computed outside the tape.
    """
    t_inv = affine_inverse(t_aug)
    return tf.stop_gradient(center_group(t_inv, iters))


def _gather_group(mats, idx):
    mats = to_homogeneous(mats)
    return tf.gather(mats, idx, axis=mats.shape.rank - 3)


def pair_targets(t_aug, i_idx, j_idx):
    """True pairwise maps image i -> image j: T_j T_i^-1 for index pairs. Returns [..., P, 4, 4]."""
    t_i = _gather_group(t_aug, i_idx)
    t_j = _gather_group(t_aug, j_idx)
    return tf.stop_gradient(tf.matmul(t_j, affine_inverse(t_i)))


def pair_predictions(a_pred, i_idx, j_idx):
    """Predicted pairwise maps image i -> image j: A_j^-1 A_i. Returns [..., P, 4, 4]."""
    a_i = _gather_group(a_pred, i_idx)
    a_j = _gather_group(a_pred, j_idx)
    return tf.matmul(affine_inverse(a_j), a_i)


# ---------------------------------------------------------------------------- point loss
def brain_second_moment(mask, spacing):
    """S = mean over mask voxels of h h^T, h = [x_mm, 1] in centered mm coordinates (numpy).

    ``mask``: [X, Y, Z] or [X, Y, Z, 1], nonzero = brain. Returns float64 [4, 4]. Precomputed
    once (e.g. from the template brain mask) and passed to `point_loss`.
    """
    mask = np.asarray(mask)
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 3:
        raise ValueError("mask must be [X, Y, Z] or [X, Y, Z, 1], got {}".format(mask.shape))
    idx = np.argwhere(mask != 0).astype(np.float64)
    if idx.shape[0] == 0:
        raise ValueError("mask is empty")
    s = _spacing3(spacing)
    c = (np.asarray(mask.shape, np.float64) - 1.0) / 2.0
    h = np.concatenate([(idx - c) * s, np.ones((idx.shape[0], 1))], axis=1)
    return h.T @ h / h.shape[0]


def point_loss(d_a, s_moment):
    """tr(dA S dA^T): mean squared displacement (mm^2) over the points S was computed from.

    ``d_a``: [..., 4, 4] or [..., 3, 4] matrix differences (e.g. A - A*); ``s_moment``: [4, 4].
    Returns [...] (one value per matrix); average it for the loss, sqrt for RMS mm.
    """
    d_a = to_float(d_a)
    s = tf.cast(s_moment, d_a.dtype)
    return tf.reduce_sum(tf.matmul(d_a, s) * d_a, axis=[-2, -1])
