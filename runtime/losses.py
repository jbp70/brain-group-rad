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

LNCC (design 6.2 v3.9), float32, on intensities [N, X, Y, Z, 1], window k (odd, level voxels):
* Local means over a k^3 box (zero padded, divided by the true per-voxel window count):
  mu_a, mu_b, cov = <ab> - mu_a mu_b, var_a = max(<a^2> - mu_a^2, 0) (same for b).
  Squared LNCC (default) cc = cov^2 / (var_a var_b + eps), in [0, 1]; signed (tests only)
  cov / sqrt(var_a var_b + eps). eps = `NCC_EPS` (local std product ~3e-4 for [0, 1] intensities).
  Box sums: separable `tf.nn.conv3d` with ones kernels (no variables).
* ROI: fg = sum_{c >= 1} onehot (zero fill gives 0 outside the FOV), ROI = box_sum(fg) > 1e-3
  (the foreground dilated by k // 2, no gradient); per-image masked mean, then mean over images.
* Template: W_i = warp(image_i, u_i); leave-one-out template T_-i = (sum_j W_j - W_i) / (N - 1)
  with gradients through it; ROI from the warped one-hot template's fg;
  L_ncc_tmpl = mean_i (1 - lncc(W_i, T_-i)).
* Pairwise, on the pairwise-Dice pairs and displacements: warp(image_i, disp_p) vs image_j, ROI
  from max(fg(moved one-hot_i), fg(one-hot_j)); L_ncc_pair = mean_p (1 - lncc).
* Total (each extra term only when its weight is > 0, so the default graph is unchanged):
  L = lambda_dice_tmpl L_tmpl + lambda_pdice L_pdice + lambda_smooth L_smooth
      + lambda_ncc_tmpl L_ncc_tmpl + lambda_ncc_pair L_ncc_pair.
"""
import numpy as np
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


def _soft_dice_lean(a, b, eps=1e-5, exclude_background=True):
    """`soft_dice` without materialised channel slices or broadcast copies (milestone 9 design
    3.4): the sums run over all channels (b, e.g. a [1, X, Y, Z, C] template, broadcasts
    implicitly), then the background column of the [N, C] results is dropped."""
    a = _float(a)
    b = tf.cast(b, a.dtype)
    axes = [1, 2, 3]
    inter = tf.reduce_sum(a * b, axis=axes)
    total = tf.reduce_sum(a, axis=axes) + tf.reduce_sum(b, axis=axes)
    if exclude_background:
        inter, total = inter[..., 1:], total[..., 1:]
    dice = (2.0 * inter + eps) / (total + eps)
    present = tf.cast(total > PRESENT_THRESHOLD, a.dtype)
    count = tf.reduce_sum(present, axis=-1)
    return tf.where(count > 0, tf.reduce_sum(dice * present, axis=-1) / tf.maximum(count, 1.0),
                    tf.ones_like(count))


def template_dice_loss(warped, eps=1e-5, lean=False):
    """(loss, dice [N], template) with template = mean_i warped_i (gradients flow through it).
    ``lean``: `_soft_dice_lean` with the [1, ...] template broadcast implicitly."""
    warped = _float(warped)
    template = tf.reduce_mean(warped, axis=0, keepdims=True)
    if lean:
        dice = _soft_dice_lean(warped, template, eps)
    else:
        dice = soft_dice(warped, tf.broadcast_to(template, tf.shape(warped)), eps)
    return tf.reduce_mean(1.0 - dice), dice, template


def pair_dice_loss(onehot, u, u_inv, pair_i, pair_j, eps=1e-5, disp=None, return_moved=False,
                   lean_sampler=False, lean_losses=False):
    """(loss, dice [P]): warp(onehot_i, compose(u_i, u_inv_j)) vs onehot_j over the pairs.

    ``disp`` (optional): the pairwise displacements [P, X, Y, Z, 3] if already computed
    (`diffeo.pairwise_displacement`, shared with the pairwise LNCC). ``return_moved``: also
    return the moved and target one-hots, (loss, dice, moved, target). ``lean_sampler``: the
    warps use the lean sampler (the one-hot without an input gradient); ``lean_losses``:
    `_soft_dice_lean` (milestone 9 design 3.3 / 3.4; both default False = today's ops).
    """
    onehot = _float(onehot)
    if disp is None:
        disp = diffeo.pairwise_displacement(u, u_inv, pair_i, pair_j, lean=lean_sampler)
    if lean_sampler:
        moved = diffeo.warp(tf.gather(onehot, tf.reshape(pair_i, [-1])), disp, lean=True,
                            lean_images_grad=False)
    else:
        moved = diffeo.warp(tf.gather(onehot, tf.reshape(pair_i, [-1])), disp)
    target = tf.gather(onehot, tf.reshape(pair_j, [-1]))
    dice = _soft_dice_lean(moved, target, eps) if lean_losses else soft_dice(moved, target, eps)
    if return_moved:
        return tf.reduce_mean(1.0 - dice), dice, moved, target
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


# ============================================================================ LNCC
NCC_EPS = 1e-7
NCC_ROI_THRESHOLD = 1e-3


def check_ncc_window(window, spatial=None):
    """``window`` as an int: odd, >= 3 and (with ``spatial``) <= every spatial dim; ValueError
    otherwise."""
    if isinstance(window, bool) or int(window) != window:
        raise ValueError("LNCC window must be an integer, got {!r}".format(window))
    window = int(window)
    if window < 3 or window % 2 == 0:
        raise ValueError("LNCC window must be odd and >= 3, got {}".format(window))
    if spatial is not None and any(window > int(n) for n in spatial):
        raise ValueError("LNCC window {} larger than the volume {}".format(window, list(spatial)))
    return window


def _spatial(x):
    dims = x.shape[1:4].as_list()
    if x.shape.rank != 5 or any(d is None for d in dims):
        raise ValueError("expected [B, X, Y, Z, C] with static spatial dims, got {}".format(x.shape))
    return dims


def box_sum(x, window):
    """k^3 box sum (zero padded, same size) of x [B, X, Y, Z, C], per channel: separable
    `tf.nn.conv3d` with constant ones kernels (channels are folded into the batch axis)."""
    x = _float(x)
    spatial = _spatial(x)
    window = check_ncc_window(window, spatial)
    c = x.shape[-1]
    flat = x
    if c != 1:
        flat = tf.reshape(tf.transpose(x, [4, 0, 1, 2, 3]), [-1] + spatial + [1])
    for axis in range(3):
        k = [1, 1, 1, 1, 1]
        k[axis] = window
        flat = tf.nn.conv3d(flat, tf.ones(k, flat.dtype), strides=[1, 1, 1, 1, 1], padding="SAME")
    if c != 1:
        flat = tf.transpose(tf.reshape(flat, [c, -1] + spatial), [1, 2, 3, 4, 0])
    return flat


def window_count(spatial, window):
    """Number of in-volume voxels of the k^3 window at every voxel: float64 numpy [X, Y, Z]
    (= box_sum of ones; a constant of the shape and window)."""
    r = int(window) // 2
    axes = []
    for n in spatial:
        i = np.arange(int(n))
        axes.append((np.minimum(i + r, int(n) - 1) - np.maximum(i - r, 0) + 1).astype(np.float64))
    return axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]


def _as_volume(x):
    x = _float(x)
    return x[..., None] if x.shape.rank == 4 else x


def local_ncc(a, b, window, eps=NCC_EPS, signed=False):
    """LNCC map [B, X, Y, Z] of single-channel volumes a, b ([B, X, Y, Z] or [B, X, Y, Z, 1];
    broadcast over B). Local means over the k^3 window (true count at the borders); squared
    cc = cov^2 / (var_a var_b + eps) (default) or signed cov / sqrt(var_a var_b + eps)."""
    a = _as_volume(a)
    b = tf.cast(_as_volume(b), a.dtype)
    if a.shape[-1] != 1 or b.shape[-1] != 1:
        raise ValueError("local_ncc expects single-channel volumes, got {} / {}".format(a.shape, b.shape))
    if a.shape.as_list() != b.shape.as_list():
        a, b = a + tf.zeros_like(b), b + tf.zeros_like(a)     # broadcast (static shapes kept)
    spatial = _spatial(a)
    window = check_ncc_window(window, spatial)
    # The five quantities stacked along the batch axis: one separable box filter pass.
    s = box_sum(tf.concat([a, b, a * a, b * b, a * b], axis=0), window)
    inv = tf.constant(1.0 / window_count(spatial, window)[None, ..., None], a.dtype)
    s_a, s_b, s_aa, s_bb, s_ab = tf.split(s * inv, 5, axis=0)
    cov = s_ab - s_a * s_b
    var_a = tf.maximum(s_aa - s_a * s_a, 0.0)
    var_b = tf.maximum(s_bb - s_b * s_b, 0.0)
    eps = tf.constant(float(eps), a.dtype)
    if signed:
        cc = cov / tf.sqrt(var_a * var_b + eps)
    else:
        cc = cov * cov / (var_a * var_b + eps)
    return cc[..., 0]


def foreground(onehot):
    """Foreground probability sum_{c >= 1} onehot: [B, X, Y, Z]. Zero fill makes every channel 0
    outside the FOV, so out-of-FOV voxels read as background (unlike 1 - onehot[..., 0])."""
    return tf.reduce_sum(_float(onehot)[..., 1:], axis=-1)


def ncc_roi(fg, window):
    """LNCC region: the foreground dilated by the window radius, box_sum(fg) > 1e-3, as a float
    mask [B, X, Y, Z] without gradient."""
    fg = _as_volume(fg)
    return tf.stop_gradient(tf.cast(box_sum(fg, window)[..., 0] > NCC_ROI_THRESHOLD, fg.dtype))


def masked_mean(m, w):
    """Per-image mean of m [B, X, Y, Z] over the mask w (broadcast): [B]."""
    m = _float(m)
    w = tf.broadcast_to(tf.cast(w, m.dtype), tf.shape(m))
    axes = [1, 2, 3]
    return tf.reduce_sum(m * w, axis=axes) / tf.maximum(tf.reduce_sum(w, axis=axes), 1.0)


def _local_ncc_maybe_lean(a, b, window, eps, lean):
    """`local_ncc`, under tf.recompute_grad when ``lean`` (milestone 9 design 3.4: its box
    filters and moments are recomputed in the backward instead of kept)."""
    if not lean:
        return local_ncc(a, b, window, eps)
    return tf.recompute_grad(lambda x, y: local_ncc(x, y, window, eps))(a, b)


def template_ncc_loss(warped_img, fg_tmpl, window, leave_one_out=True, eps=NCC_EPS,
                      lean_sampler=False, lean=False):
    """(loss, ncc [N]) of the warped images [N, X, Y, Z(, 1)] vs the intensity template, with
    gradients through the template. ``leave_one_out``: T_-i = (sum_j W_j - W_i) / (N - 1) (the
    inclusive mean's self term gives a ~1/N floor even for unrelated images); else the mean of
    all N. ROI from ``fg_tmpl`` [1 | N, X, Y, Z] (the warped one-hot template's foreground).
    ``lean``: `local_ncc` under tf.recompute_grad. ``lean_sampler`` is accepted for symmetry with
    `pair_ncc_loss`; the template image warp is done by the caller (`deform_loss`)."""
    del lean_sampler
    w = _as_volume(warped_img)
    total = tf.reduce_sum(w, axis=0, keepdims=True)
    n = tf.cast(tf.shape(w)[0], w.dtype)
    if leave_one_out:
        template = (total - w) / tf.maximum(n - 1.0, 1.0)
    else:
        template = total / n
    roi = ncc_roi(tf.cast(_float(fg_tmpl), w.dtype), window)
    ncc = masked_mean(_local_ncc_maybe_lean(w, template, window, eps, lean), roi)
    return tf.reduce_mean(1.0 - ncc), ncc


def pair_ncc_loss(images, disp, pair_i, pair_j, fg_moved, fg_target, window, eps=NCC_EPS,
                  lean_sampler=False, lean=False):
    """(loss, ncc [P]): warp(image_i, disp_p) vs image_j over the pairs (``disp`` from
    `diffeo.pairwise_displacement`, shared with the pairwise Dice). ROI from
    max(fg_moved, fg_target) [P, X, Y, Z] (foreground of the moved one-hot_i and of one-hot_j).
    ``lean_sampler``: the image warp uses the lean sampler without an image gradient; ``lean``:
    `local_ncc` under tf.recompute_grad (both default False = today's ops)."""
    images = _as_volume(images)
    if lean_sampler:
        moved = diffeo.warp(tf.gather(images, tf.reshape(pair_i, [-1])), tf.cast(disp, images.dtype),
                            lean=True, lean_images_grad=False)
    else:
        moved = diffeo.warp(tf.gather(images, tf.reshape(pair_i, [-1])), tf.cast(disp, images.dtype))
    target = tf.gather(images, tf.reshape(pair_j, [-1]))
    fg = tf.maximum(tf.cast(_float(fg_moved), images.dtype), tf.cast(_float(fg_target), images.dtype))
    roi = ncc_roi(fg, window)
    ncc = masked_mean(_local_ncc_maybe_lean(moved, target, window, eps, lean), roi)
    return tf.reduce_mean(1.0 - ncc), ncc


def deform_loss(out, batch, lambda_pdice=1.0, lambda_smooth=0.5, smooth_order=1,
                lambda_dice_tmpl=1.0, lambda_ncc_tmpl=0.0, lambda_ncc_pair=0.0, ncc_window=5,
                lean_sampler=False, lean_losses=False):
    """Total deformable loss from a model output (``u``, ``u_inv``) and a `DeformPostprocess`
    batch (``onehot``, ``pair_i``, ``pair_j``; ``images`` only when an LNCC weight is > 0).

    Returns dict(loss, tmpl, pdice, smooth, soft_dice_tmpl (mean), soft_dice_pair (mean)) and,
    per active LNCC term, ncc_tmpl / ncc_pair (mean LNCC) and ncc_tmpl_loss / ncc_pair_loss.
    With the default weights the graph is the milestone-6 one (extra terms are static Python
    choices). Memory knobs (milestone 9 design 3.3 / 3.4, default False = today's ops):
    ``lean_sampler`` = every warp uses the lean sampler (data volumes without an input
    gradient, the pairwise field composition with it); ``lean_losses`` = Dice without slices /
    broadcast copies and LNCC under tf.recompute_grad.
    """
    lam_dt, lam_nt, lam_np = float(lambda_dice_tmpl), float(lambda_ncc_tmpl), float(lambda_ncc_pair)
    if (lam_nt > 0 or lam_np > 0) and "images" not in batch:
        raise ValueError("deform_loss: an LNCC weight is > 0 but the batch has no 'images'")
    lean_kw = {"lean_sampler": True} if lean_sampler else {}
    if lean_losses:
        lean_kw["lean_losses"] = True
    u = _float(out["u"])
    u_inv = tf.cast(out["u_inv"], u.dtype)
    onehot = tf.cast(batch["onehot"], u.dtype)
    if lean_sampler:
        warped = diffeo.warp(onehot, u, lean=True, lean_images_grad=False)
    else:
        warped = diffeo.warp(onehot, u)
    tmpl, dice_t, template = template_dice_loss(warped, lean=bool(lean_losses))
    if lam_np > 0:
        disp = diffeo.pairwise_displacement(u, u_inv, batch["pair_i"], batch["pair_j"], lean=bool(lean_sampler))
        pdice, dice_p, moved_oh, target_oh = pair_dice_loss(onehot, u, u_inv, batch["pair_i"], batch["pair_j"],
                                                            disp=disp, return_moved=True, **lean_kw)
    else:
        pdice, dice_p = pair_dice_loss(onehot, u, u_inv, batch["pair_i"], batch["pair_j"], **lean_kw)
    smooth = smoothness_loss(u, smooth_order)
    tmpl_term = tmpl if lam_dt == 1.0 else tf.constant(lam_dt, u.dtype) * tmpl
    loss = (tmpl_term + tf.constant(float(lambda_pdice), u.dtype) * pdice
            + tf.constant(float(lambda_smooth), u.dtype) * smooth)
    res = {"tmpl": tmpl, "pdice": pdice, "smooth": smooth,
           "soft_dice_tmpl": tf.reduce_mean(dice_t), "soft_dice_pair": tf.reduce_mean(dice_p)}
    if lam_nt > 0 or lam_np > 0:
        images = tf.cast(_float(batch["images"]), u.dtype)
    ncc_kw = {}
    if lean_sampler:
        ncc_kw["lean_sampler"] = True
    if lean_losses:
        ncc_kw["lean"] = True
    if lam_nt > 0:
        if lean_sampler:
            warped_img = diffeo.warp(images, u, lean=True, lean_images_grad=False)
        else:
            warped_img = diffeo.warp(images, u)
        nt_loss, ncc_t = template_ncc_loss(warped_img, foreground(template), ncc_window, **ncc_kw)
        loss = loss + tf.constant(lam_nt, u.dtype) * nt_loss
        res.update(ncc_tmpl=tf.reduce_mean(ncc_t), ncc_tmpl_loss=nt_loss)
    if lam_np > 0:
        np_loss, ncc_p = pair_ncc_loss(images, disp, batch["pair_i"], batch["pair_j"], foreground(moved_oh),
                                       foreground(target_oh), ncc_window, **ncc_kw)
        loss = loss + tf.constant(lam_np, u.dtype) * np_loss
        res.update(ncc_pair=tf.reduce_mean(ncc_p), ncc_pair_loss=np_loss)
    res["loss"] = loss
    return res
