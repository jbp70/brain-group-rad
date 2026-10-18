"""Linear-task losses (design 6.1), in mm^2.

* Barycentric (center): L_center = mean_i tr(dA_i S dA_i^T), dA_i = A_i - A_i*.
* Pairwise: L_pair = mean over sampled ordered pairs (i, j) of tr(dP S dP^T),
  dP = A_j^-1 A_i - T_j T_i^-1.
* Total: L = L_center + lambda_pair * L_pair (default lambda_pair = 1, p_pair = 0.1 sampled in
  `data_loading.postprocess`).

S = E[h h^T] over the template brain mask (homogeneous centered mm), so tr(dA S dA^T) is the
mean squared displacement over brain voxels; RMS mm (sqrt of the means) is reported. Everything
runs in float32 (or float64 if A is float64); A is float32 even under mixed_float16.
"""
import tensorflow as tf

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
