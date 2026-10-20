"""Linear-task losses (design 6.1), in mm^2.

* Barycentric (center): L_center = mean_i tr(dA_i S dA_i^T), dA_i = A_i - A_i*.
* Pairwise: L_pair = mean over sampled ordered pairs (i, j) of tr(dP S dP^T),
  dP = A_j^-1 A_i - T_j T_i^-1.
* Total: L = L_center + lambda_pair * L_pair (default lambda_pair = 1, p_pair = 0.1 sampled in
  `data_loading.postprocess`).

S = E[h h^T] over the template brain mask (homogeneous centered mm), so tr(dA S dA^T) is the
mean squared displacement over brain voxels; RMS mm (sqrt of the means) is reported. Everything
runs in float32 (or float64 if A is float64); A is float32 even under mixed_float16.

Deformable losses (design 6.2, milestone 6), float32, on one-hot train18 labels (float32
[N, X, Y, Z, C], `DeformPostprocess`), fields in level voxels (`models.diffeo` conventions):
* soft Dice per image: per channel (2 sum ab + eps) / (sum a + sum b + eps) over voxels, then a
  masked mean over the PRESENT foreground channels (sum a + sum b > 1e-3); channels absent in
  both are excluded so they cannot inflate the mean (mean over no channel = 1).
* Template: warped_i = warp(onehot_i, u_i) (linear, zero fill) in central space;
  template P = mean_i warped_i over ALL N members, with gradients through P (no stop_gradient);
  L_tmpl = mean_i (1 - dice(warped_i, P)).
* Pairwise: for sampled ordered pairs (i, j), onehot_i in j's space =
  warp(onehot_i, compose(u_i, u_inv_j)) vs onehot_j; L_pdice = mean_p (1 - dice).
* Smoothness of u: order 1 (default) = forward differences, (1/3) sum_axes mean((u[x + e_a] -
  u[x])^2) (VoxelMorph Grad-l2); order 2 = bending energy (central second differences with
  2x cross terms on the interior, the reference `deformation_field_gradient_loss`).
* Total: L = L_tmpl + lambda_pdice L_pdice + lambda_smooth L_smooth (defaults 1, 0.5).
"""
import tensorflow as tf

from models import diffeo
from models import transforms as tfm


def _cast_like(x, ref):
    return tf.cast(tf.convert_to_tensor(x), ref.dtype)


def linear_center_loss(A, A_star, S):
    """mean_i point_loss(A_i - A*_i, S) (mm^2). A, A_star: [N, 4, 4]; S: [4, 4]."""
    A = tfm.to_float(A)
    d = A - _cast_like(A_star, A)
    return tf.reduce_mean(tfm.point_loss(d, _cast_like(S, A)))


def linear_pair_loss(A, T_target, pair_i, pair_j, S, P_star=None):
    """mean_p point_loss(A_j^-1 A_i - T_j T_i^-1, S) over the sampled pairs (mm^2).

    ``P_star`` (the pair targets from `LinearPostprocess`, computed in float64) is used when
    given; otherwise the targets are formed here from ``T_target`` with `pair_targets`.
    """
    A = tfm.to_float(A)
    pred = tfm.pair_predictions(A, pair_i, pair_j)
    if P_star is None:
        P_star = tfm.pair_targets(_cast_like(T_target, A), pair_i, pair_j)
    d = pred - _cast_like(P_star, A)
    return tf.reduce_mean(tfm.point_loss(d, _cast_like(S, A)))


def linear_loss(A, batch, S, lambda_pair=1.0):
    """Total linear loss from a `LinearPostprocess` batch dict.

    Returns dict(loss, center, pair (mm^2), center_rms_mm, pair_rms_mm).
    """
    center = linear_center_loss(A, batch["A_star"], S)
    pair = linear_pair_loss(A, batch["T_target"], batch["pair_i"], batch["pair_j"], S,
                            P_star=batch.get("P_star"))
    loss = center + tf.constant(float(lambda_pair), center.dtype) * pair
    return {"loss": loss, "center": center, "pair": pair,
            "center_rms_mm": tf.sqrt(tf.maximum(center, 0.0)),
            "pair_rms_mm": tf.sqrt(tf.maximum(pair, 0.0))}


# ============================================================================ deformable
PRESENT_THRESHOLD = 1e-3


def _float(x):
    x = tf.convert_to_tensor(x)
    return x if x.dtype in (tf.float32, tf.float64) else tf.cast(x, tf.float32)


def soft_dice(a, b, eps=1e-5, exclude_background=True):
    """Soft Dice per image [N] of [N, X, Y, Z, C] maps: per-channel Dice, then the mean over the
    present (sum a + sum b > 1e-3) channels (channel 0 excluded by default)."""
    a = _float(a)
    b = tf.cast(b, a.dtype)
    if exclude_background:
        a, b = a[..., 1:], b[..., 1:]
    axes = [1, 2, 3]
    inter = tf.reduce_sum(a * b, axis=axes)
    total = tf.reduce_sum(a, axis=axes) + tf.reduce_sum(b, axis=axes)
    dice = (2.0 * inter + eps) / (total + eps)
    present = tf.cast(total > PRESENT_THRESHOLD, a.dtype)
    count = tf.reduce_sum(present, axis=-1)
    return tf.where(count > 0, tf.reduce_sum(dice * present, axis=-1) / tf.maximum(count, 1.0),
                    tf.ones_like(count))


def template_dice_loss(warped, eps=1e-5):
    """(loss, dice [N], template) with template = mean_i warped_i (gradients flow through it)."""
    warped = _float(warped)
    template = tf.reduce_mean(warped, axis=0, keepdims=True)
    dice = soft_dice(warped, tf.broadcast_to(template, tf.shape(warped)), eps)
    return tf.reduce_mean(1.0 - dice), dice, template


def pair_dice_loss(onehot, u, u_inv, pair_i, pair_j, eps=1e-5):
    """(loss, dice [P]): warp(onehot_i, compose(u_i, u_inv_j)) vs onehot_j over the pairs."""
    onehot = _float(onehot)
    disp = diffeo.pairwise_displacement(u, u_inv, pair_i, pair_j)
    moved = diffeo.warp(tf.gather(onehot, tf.reshape(pair_i, [-1])), disp)
    target = tf.gather(onehot, tf.reshape(pair_j, [-1]))
    dice = soft_dice(moved, target, eps)
    return tf.reduce_mean(1.0 - dice), dice


def smoothness_loss(u, order=1):
    """Smoothness of a displacement field [N, X, Y, Z, 3] (module docstring): order 1 = mean
    squared forward differences averaged over the 3 axes; order 2 = bending energy."""
    u = _float(u)
    order = int(order)
    if order == 1:
        terms = []
        for a in range(3):
            hi = [slice(None)] * 5
            lo = [slice(None)] * 5
            hi[a + 1] = slice(1, None)
            lo[a + 1] = slice(None, -1)
            terms.append(tf.reduce_mean(tf.square(u[tuple(hi)] - u[tuple(lo)])))
        return tf.add_n(terms) / 3.0
    if order != 2:
        raise ValueError("smoothness order must be 1 or 2, got {}".format(order))
    # Port of the reference deformation_field_gradient_loss (weights [0, 1], ndims 3).
    d_xx = u[:, :-2] - 2.0 * u[:, 1:-1] + u[:, 2:]
    d_yy = u[:, :, :-2] - 2.0 * u[:, :, 1:-1] + u[:, :, 2:]
    d_zz = u[:, :, :, :-2] - 2.0 * u[:, :, :, 1:-1] + u[:, :, :, 2:]
    d_xy = (u[:, :-2, :-2] - u[:, 2:, :-2] - u[:, :-2, 2:] + u[:, 2:, 2:]) / 4.0
    d_xz = (u[:, :-2, :, :-2] - u[:, 2:, :, :-2] - u[:, :-2, :, 2:] + u[:, 2:, :, 2:]) / 4.0
    d_yz = (u[:, :, :-2, :-2] - u[:, :, 2:, :-2] - u[:, :, :-2, 2:] + u[:, :, 2:, 2:]) / 4.0
    be = (tf.square(d_xx)[:, :, 1:-1, 1:-1] + tf.square(d_yy)[:, 1:-1, :, 1:-1]
          + tf.square(d_zz)[:, 1:-1, 1:-1] + 2.0 * tf.square(d_xy)[:, :, :, 1:-1]
          + 2.0 * tf.square(d_xz)[:, :, 1:-1] + 2.0 * tf.square(d_yz)[:, 1:-1])
    return tf.reduce_mean(be)


def deform_loss(out, batch, lambda_pdice=1.0, lambda_smooth=0.5, smooth_order=1):
    """Total deformable loss from a model output (``u``, ``u_inv``) and a `DeformPostprocess`
    batch (``onehot``, ``pair_i``, ``pair_j``).

    Returns dict(loss, tmpl, pdice, smooth, soft_dice_tmpl (mean), soft_dice_pair (mean)).
    """
    u = _float(out["u"])
    u_inv = tf.cast(out["u_inv"], u.dtype)
    onehot = tf.cast(batch["onehot"], u.dtype)
    warped = diffeo.warp(onehot, u)
    tmpl, dice_t, _ = template_dice_loss(warped)
    pdice, dice_p = pair_dice_loss(onehot, u, u_inv, batch["pair_i"], batch["pair_j"])
    smooth = smoothness_loss(u, smooth_order)
    loss = (tmpl + tf.constant(float(lambda_pdice), u.dtype) * pdice
            + tf.constant(float(lambda_smooth), u.dtype) * smooth)
    return {"loss": loss, "tmpl": tmpl, "pdice": pdice, "smooth": smooth,
            "soft_dice_tmpl": tf.reduce_mean(dice_t), "soft_dice_pair": tf.reduce_mean(dice_p)}
