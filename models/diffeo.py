"""Displacement fields, composition and SVF integration (design 4.6, milestone-6 plan section 0).

Conventions (binding):

| Item | Convention |
|---|---|
| Field units | Displacements and velocities are in voxel-index units of the level they live at. mm = voxels * `P.LEVEL_SPACING[level]` (`to_mm`). |
| Grid | Voxel index p (world (p - (n-1)/2) * s, `models.layers.index_grid`). Fields act on voxel indices directly, so the center does not enter them. |
| Pull warps | ``warp(vol, u)(x) = vol(x + u(x))``. phi(x) = x + u(x) maps central-space points to image-i points; warped image = I_i o phi_i. |
| Composition | ``compose(u_a, u_b)`` = displacement of phi_a o phi_b (phi_b first) = ``u_b + warp_field(u_a, u_b)``. |
| Boundaries | Fields are resampled with border clamping (coords clipped to [0, n-1], edge replication: a constant field stays constant). Images and one-hot labels use zero fill (`transform_image` semantics). |
| SVF | v_i in level voxels, central space. Centering: v_i <- v_i - mean_j v_j at every voxel (group axis 0, N dynamic). Exp: scaling and squaring, T steps (default 7): u <- v / 2^T, then T times u <- compose(u, u); T = 0 returns v. Inverse: exp(-v), never a numerical inversion. |
| Pairwise | image j coords -> image i coords: phi_i o phi_j^-1 = ``compose(u_i, u_inv_j)``; onehot_i in j's space = ``warp(onehot_i, compose(u_i, u_inv_j))``. |
| Cascade | phi^(k) = phi^(k-1) o exp(v^(k)) -> ``compose(up(u_prev), u_k)``; inverse exp(-v^(k)) o phi^(k-1)^-1 -> ``compose(u_inv_k, up(u_inv_prev))``, up = `upsample_displacement`. |
| Trilinear x2 | Coarse voxel q at fine 2q + 0.5 (half-pixel, edge replicate); `upsample_displacement(u) = 2 * upsample_trilinear2x(u)`. |
| Jacobian | det(I + grad u), central differences, interior voxels only: [N, X-2, Y-2, Z-2]. Folding fraction = mean(J <= 0). |
| Dtypes | Fields and integration in float32 (float64 kept); float16 inputs are cast to float32; never float16 grids. |
"""
import tensorflow as tf

from models import layers as L
from models import transforms as tfm


def _field_dtype(*tensors):
    return tf.float64 if any(t.dtype == tf.float64 for t in tensors) else tf.float32


def _as_field(u, dtype=None):
    u = tf.convert_to_tensor(u)
    return tf.cast(u, dtype or _field_dtype(u))


def warp(vol, u, interpolation="linear", boundary="zero", fill_value=0.0, lean=False,
         lean_images_grad=True):
    """Pull warp: out(x) = vol(x + u(x)). vol [N, X, Y, Z, C], u [N, X', Y', Z', 3] (voxels).

    The output has u's spatial shape. float16 volumes are warped in float32 and returned as
    float32; other float dtypes are kept; integer labels in nearest mode stay integer.
    ``boundary`` 'zero' (images, one-hots; ``fill_value`` outside) or 'border' (fields).
    ``lean`` / ``lean_images_grad``: the lean trilinear sampler (milestone 9 design 3.3; same
    forward values, no corner tensors kept for the backward; False = today's sampler) and, with
    it, whether ``vol`` gets a gradient (False for data volumes). The other functions of this
    module pass ``lean`` through (default False everywhere).
    """
    vol = tf.convert_to_tensor(vol)
    if vol.dtype == tf.float16:
        vol = tf.cast(vol, tf.float32)
    u = _as_field(u)
    spatial = u.shape[1:4].as_list()
    if u.shape.rank != 5 or u.shape[-1] != 3 or any(v is None for v in spatial):
        raise ValueError("u must be [N, X, Y, Z, 3] with static spatial dims, got {}".format(u.shape))
    coords = L.index_grid(spatial, u.dtype)[None] + u
    if lean:
        return L.sample_voxel_coords(vol, coords, interpolation=interpolation, fill_value=fill_value,
                                     boundary=boundary, lean=True, lean_images_grad=lean_images_grad)
    return L.sample_voxel_coords(vol, coords, interpolation=interpolation, fill_value=fill_value,
                                 boundary=boundary)


def warp_field(field, u, lean=False):
    """Resample a field (any channel count) at x + u(x) with border clamping."""
    field = tf.convert_to_tensor(field)
    u = tf.convert_to_tensor(u)
    dt = _field_dtype(field, u)
    return warp(tf.cast(field, dt), tf.cast(u, dt), boundary="border", lean=lean)


def compose(u_a, u_b, lean=False):
    """Displacement of phi_a o phi_b (phi_b applied first): u_b + u_a(x + u_b(x))."""
    u_a = tf.convert_to_tensor(u_a)
    u_b = tf.convert_to_tensor(u_b)
    dt = _field_dtype(u_a, u_b)
    u_a, u_b = tf.cast(u_a, dt), tf.cast(u_b, dt)
    return u_b + warp_field(u_a, u_b, lean=lean)


def integrate_svf(v, steps=7, lean=False):
    """exp(v) by scaling and squaring: u = v / 2^T, then T times u <- compose(u, u)."""
    steps = int(steps)
    if steps < 0:
        raise ValueError("steps must be >= 0, got {}".format(steps))
    v = _as_field(v)
    if steps == 0:
        return v
    u = v / (2.0 ** steps)
    for _ in range(steps):
        u = compose(u, u, lean=lean)
    return u


def exp_and_inverse(v, steps=7, lean=False):
    """(exp(v), exp(-v)): forward displacement and its exact-by-construction inverse."""
    v = _as_field(v)
    return integrate_svf(v, steps, lean=lean), integrate_svf(-v, steps, lean=lean)


def center_velocities(v):
    """v_i - mean_j v_j at every voxel (group = batch axis 0, N dynamic)."""
    v = _as_field(v)
    return v - tf.reduce_mean(v, axis=0, keepdims=True)


def upsample_displacement(u):
    """Coarse-level field -> next finer level: 2 * trilinear x2 (values doubled: voxels halve)."""
    u = _as_field(u)
    return 2.0 * L.upsample_trilinear2x(u)


def jacobian_determinant(u):
    """det(I + grad u) by central differences on the interior voxels: [N, X-2, Y-2, Z-2]."""
    u = _as_field(u)
    if u.shape.rank != 5 or u.shape[-1] != 3:
        raise ValueError("u must be [N, X, Y, Z, 3], got {}".format(u.shape))
    grads = []                                     # grads[a][..., c] = d u_c / d x_a
    for a in range(3):
        hi = [slice(None), slice(1, -1), slice(1, -1), slice(1, -1)]
        lo = list(hi)
        hi[a + 1] = slice(2, None)
        lo[a + 1] = slice(None, -2)
        grads.append((u[tuple(hi)] - u[tuple(lo)]) * 0.5)

    def j(c, a):                                   # J[c, a] = delta_ca + d u_c / d x_a
        g = grads[a][..., c]
        return g + 1.0 if c == a else g

    return (j(0, 0) * (j(1, 1) * j(2, 2) - j(1, 2) * j(2, 1))
            - j(0, 1) * (j(1, 0) * j(2, 2) - j(1, 2) * j(2, 0))
            + j(0, 2) * (j(1, 0) * j(2, 1) - j(1, 1) * j(2, 0)))


def folding_fraction(u, mask=None):
    """Fraction of interior voxels with J <= 0, per image: [N].

    mask (optional): [N | 1, X, Y, Z(, 1)] (full size, cropped to the interior here) or
    [N | 1, X-2, Y-2, Z-2(, 1)]; the fraction is taken over mask > 0.
    """
    jac = jacobian_determinant(u)
    fold = tf.cast(jac <= 0.0, jac.dtype)
    if mask is None:
        return tf.reduce_mean(fold, axis=[1, 2, 3])
    m = tf.convert_to_tensor(mask)
    if m.shape.rank == 5:
        m = m[..., 0]
    if m.shape[1:4].as_list() == [int(d) + 2 for d in jac.shape[1:4]]:
        m = m[:, 1:-1, 1:-1, 1:-1]
    m = tf.cast(tf.cast(m, jac.dtype) > 0, jac.dtype)
    m = tf.broadcast_to(m, tf.shape(fold))
    return tf.reduce_sum(fold * m, axis=[1, 2, 3]) / tf.maximum(tf.reduce_sum(m, axis=[1, 2, 3]), 1.0)


def pairwise_displacement(u, u_inv, i, j, lean=False):
    """Image j coords -> image i coords: compose(u[i], u_inv[j]) -> [P, X, Y, Z, 3].

    i, j: scalar indices or index vectors of equal length P (a scalar gives P = 1).
    warp(onehot[i], pairwise_displacement(u, u_inv, i, j)) is image i's one-hot in j's space.
    """
    i = tf.reshape(tf.convert_to_tensor(i), [-1])
    j = tf.reshape(tf.convert_to_tensor(j), [-1])
    return compose(tf.gather(u, i), tf.gather(u_inv, j), lean=lean)


def compose_cascade(u_prev_up, u_inv_prev_up, u_new, u_inv_new, lean=False):
    """Milestone-8 cascade: (phi_prev o exp(v_new), exp(-v_new) o phi_prev^-1).

    ``u_prev_up``/``u_inv_prev_up`` are the previous level's fields already upsampled with
    `upsample_displacement`. Returns (u, u_inv).
    """
    return compose(u_prev_up, u_new, lean=lean), compose(u_inv_new, u_inv_prev_up, lean=lean)


def to_mm(u, spacing):
    """Voxel displacements -> mm (per-axis spacing, scalar or 3 values)."""
    u = _as_field(u)
    return u * tf.constant(tfm._spacing3(spacing), u.dtype)
