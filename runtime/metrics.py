"""Rigid/affine metrics (design 7), numpy float64.

Per matrix M against its truth M_true (both [4, 4], centered mm):
* rms_mm: sqrt(tr(dM S dM^T)), the RMS displacement over the template brain voxels;
* rot_deg: angle of polar(L_M) polar(L_true)^T, polar(L) = U V^T from the SVD of the 3x3 part;
* trans_mm: |t_M - t_true|, the displacement of the rotation center (the image center).

Centered errors compare A_i with A_i*. Pairwise errors compare A_j^-1 A_i with T_j T_i^-1 over ALL
ordered pairs i != j. The cycle error (A numerics check, ~0 by construction) is the RMS of
P_ki P_jk P_ij - I over random distinct triplets.

Affine-specific errors (``group_errors(..., affine=True)``, milestone 5). Each 3x3 linear part is
factored as L = Q K (QR, the model's own parametrization A = [R K | t], design 4.4) with Q a
rotation and K = U D upper triangular: D = diag(exp(s)) the per-axis scales, U unit upper
triangular with the shears (k_xy, k_xz, k_yz) = (K01 / K11, K02 / K22, K12 / K22). Per matrix:
* scale_pct: 100 * mean over the 3 axes of |s - s_true| (log-scale difference, ~ percent);
* shear_pct: 100 * mean over the 3 shears of |k - k_true|;
* qr_rot_deg: angle of Q Q_true^T (the rotation factor; `rot_deg` uses the polar rotation).

`group_errors` returns per-matrix arrays (so validation can pool them over groups);
`summarize` turns them into means (+ p95 of the RMS values) under a key prefix.

Deformable metrics (design 7, milestone 6), from hard label maps (argmax of warped one-hots):
* hard Dice per foreground label, NaN where the label is absent in both maps; the mean
  foreground Dice is the nanmean (absent labels skipped; 1 when no label is present at all);
* template_dice [N]: warped seg_i vs the label template (argmax of the mean warped one-hot);
* pair_dice [P]: seg_i moved into j's space vs seg_j;
* neg_jac_pct [N]: 100 * fraction of interior voxels with det J <= 0;
* inv_consistency_mm / _max_mm [N]: mean / max over voxels of |compose(u, u_inv)| * spacing;
* centrality_mm / raw_centrality_mm [1]: mean over voxels of |mean_i v_i| * spacing for the
  centred v (~0 by construction) and for the raw head output v_raw.
* ncc_template [N] / ncc_pair [P] (v3.9, added by the Trainer from the images, not by
  `deform_group_errors`): masked mean squared LNCC of each warped image vs the leave-one-out
  intensity template, and of image i moved into j's space vs image j (`runtime.losses`).
`deform_group_errors` returns per-item arrays (poolable with `concat_errors`, `summarize`).
"""
import numpy as np


def _np(x):
    return np.asarray(x, np.float64)


def qr_factors(L):
    """L = Q K for [..., 3, 3] -> (Q [..., 3, 3], log_scale [..., 3], shear [..., 3]).

    K has a positive diagonal (so Q is a rotation when det L > 0). log_scale = log diag(K);
    shear = (K01 / K11, K02 / K22, K12 / K22), the entries of U in K = U diag(K). Loops over the
    leading dimensions (numpy 1.21 has no batched QR)."""
    L = _np(L)
    lead = L.shape[:-2]
    flat = L.reshape((-1, 3, 3))
    q_out = np.empty_like(flat)
    ls = np.empty((flat.shape[0], 3))
    sh = np.empty((flat.shape[0], 3))
    for n in range(flat.shape[0]):
        q, k = np.linalg.qr(flat[n])
        sgn = np.where(np.diag(k) < 0, -1.0, 1.0)
        q, k = q * sgn[None, :], sgn[:, None] * k
        d = np.diag(k)
        q_out[n] = q
        ls[n] = np.log(d)
        sh[n] = (k[0, 1] / d[1], k[0, 2] / d[2], k[1, 2] / d[2])
    return q_out.reshape(lead + (3, 3)), ls.reshape(lead + (3,)), sh.reshape(lead + (3,))


def polar_rotation(L):
    """Nearest rotation U V^T of [..., 3, 3] matrices (det fixed to +1)."""
    L = _np(L)
    u, _, vt = np.linalg.svd(L)
    d = np.sign(np.linalg.det(u @ vt))
    u = u.copy()
    u[..., :, -1] *= d[..., None]
    return u @ vt


def rotation_angle_deg(L):
    """Rotation angle (degrees) of the polar rotation of [..., 3, 3] matrices (atan2 form)."""
    r = polar_rotation(L)
    c = (np.trace(r, axis1=-2, axis2=-1) - 1.0) / 2.0
    v = np.stack([r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0],
                  r[..., 1, 0] - r[..., 0, 1]], axis=-1)
    s = 0.5 * np.linalg.norm(v, axis=-1)
    return np.degrees(np.arctan2(s, c))


def point_rms(dM, S):
    """sqrt(tr(dM S dM^T)) for [..., 4, 4] (or [..., 3, 4]) differences."""
    dM = _np(dM)
    ms = np.einsum("...ij,jk,...ik->...", dM, _np(S), dM)
    return np.sqrt(np.maximum(ms, 0.0))


def matrix_errors(M, M_true, S):
    """dict(rms_mm, rot_deg, trans_mm) per matrix ([...] arrays)."""
    M, M_true = _np(M), _np(M_true)
    rel = polar_rotation(M[..., :3, :3]) @ np.swapaxes(polar_rotation(M_true[..., :3, :3]), -1, -2)
    return {"rms_mm": point_rms(M - M_true, S),
            "rot_deg": rotation_angle_deg(rel),
            "trans_mm": np.linalg.norm(M[..., :3, 3] - M_true[..., :3, 3], axis=-1)}


def affine_matrix_errors(M, M_true):
    """dict(scale_pct, shear_pct, qr_rot_deg) per matrix from the QR factors (module docstring)."""
    q, ls, sh = qr_factors(_np(M)[..., :3, :3])
    q0, ls0, sh0 = qr_factors(_np(M_true)[..., :3, :3])
    return {"scale_pct": 100.0 * np.mean(np.abs(ls - ls0), axis=-1),
            "shear_pct": 100.0 * np.mean(np.abs(sh - sh0), axis=-1),
            "qr_rot_deg": rotation_angle_deg(q @ np.swapaxes(q0, -1, -2))}


def centered_errors(A, A_star, S):
    return matrix_errors(A, A_star, S)


def all_ordered_pairs(n):
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    keep = i != j
    return i[keep], j[keep]


def pairwise_errors(A, T_target, S):
    """Errors of A_j^-1 A_i vs T_j T_i^-1 over all N (N - 1) ordered pairs."""
    A, T = _np(A), _np(T_target)
    i, j = all_ordered_pairs(A.shape[0])
    pred = np.linalg.inv(A[j]) @ A[i]
    true = T[j] @ np.linalg.inv(T[i])
    return matrix_errors(pred, true, S)


def sample_triplets(n, n_triplets=64, seed=0):
    """[K, 3] random triplets of distinct indices (K = n_triplets; empty if n < 3)."""
    if n < 3 or n_triplets <= 0:
        return np.zeros((0, 3), np.int64)
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(n, 3, replace=False) for _ in range(int(n_triplets))])


def cycle_errors(A, S, n_triplets=64, seed=0):
    """RMS (mm) of C - I with C = P_ki P_jk P_ij (i -> j -> k -> i), P_ab = A_b^-1 A_a."""
    A = _np(A)
    tri = sample_triplets(A.shape[0], n_triplets, seed)
    if tri.shape[0] == 0:
        return {"rms_mm": np.zeros((0,)), "triplets": tri}
    inv = np.linalg.inv(A)
    i, j, k = tri[:, 0], tri[:, 1], tri[:, 2]
    c = (inv[i] @ A[k]) @ (inv[k] @ A[j]) @ (inv[j] @ A[i])
    return {"rms_mm": point_rms(c - np.eye(4), S), "triplets": tri}


def group_errors(A, A_star, T_target, S, n_triplets=64, seed=0, affine=False):
    """Per-matrix error arrays of one group (unprefixed keys). ``affine=True`` adds
    center_/pair_{scale_pct, shear_pct, qr_rot_deg} (`affine_matrix_errors`)."""
    out = {}
    for k, v in centered_errors(A, A_star, S).items():
        out["center_" + k] = v
    for k, v in pairwise_errors(A, T_target, S).items():
        out["pair_" + k] = v
    out["cycle_rms_mm"] = cycle_errors(A, S, n_triplets, seed)["rms_mm"]
    if affine:
        A, T = _np(A), _np(T_target)
        for k, v in affine_matrix_errors(A, A_star).items():
            out["center_" + k] = v
        i, j = all_ordered_pairs(A.shape[0])
        pred = np.linalg.inv(A[j]) @ A[i]
        true = T[j] @ np.linalg.inv(T[i])
        for k, v in affine_matrix_errors(pred, true).items():
            out["pair_" + k] = v
    return out


def mean_pair_rms(metrics, tasks):
    """Combined best-checkpoint metric: mean of ``val_<task>/pair_rms_mm`` over `tasks`, or None
    if any of them is missing."""
    vals = [metrics.get("val_%s/pair_rms_mm" % t) for t in tasks]
    if not vals or any(v is None for v in vals):
        return None
    return float(np.mean(vals))


def hard_dice_per_label(seg_a, seg_b, n_classes):
    """Dice of every foreground label 1..n_classes-1 between two integer maps -> [n_classes - 1]
    (float64), NaN where the label is absent in both."""
    a = np.asarray(seg_a).reshape(-1)
    b = np.asarray(seg_b).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("label maps differ in size (%d vs %d)" % (a.size, b.size))
    n = int(n_classes)
    a, b = a.astype(np.int64), b.astype(np.int64)
    count_a = np.bincount(a, minlength=n)[:n].astype(np.float64)
    count_b = np.bincount(b, minlength=n)[:n].astype(np.float64)
    inter = np.bincount(a[a == b], minlength=n)[:n].astype(np.float64)
    total = count_a + count_b
    with np.errstate(invalid="ignore", divide="ignore"):
        dice = np.where(total > 0, 2.0 * inter / np.where(total > 0, total, 1.0), np.nan)
    return dice[1:]


def mean_fg_dice(seg_a, seg_b, n_classes):
    """Mean foreground hard Dice over the labels present in either map (1.0 if none is)."""
    d = hard_dice_per_label(seg_a, seg_b, n_classes)
    if not np.any(np.isfinite(d)):
        return 1.0
    return float(np.nanmean(d))


def deform_group_errors(seg_warped, seg_template, seg_pair_pred, seg_pair_true, jac, inv_err_vox,
                        v, v_raw, spacing, n_classes):
    """Per-item deformable error arrays of one group (unprefixed keys, module docstring).

    seg_warped [N, X, Y, Z] (int), seg_template [X, Y, Z] (or [1, X, Y, Z]), seg_pair_pred /
    seg_pair_true [P, X, Y, Z], jac [N, ...] (det J, interior), inv_err_vox [N, X, Y, Z, 3]
    (the displacement compose(u, u_inv), voxels) or its norm [N, X, Y, Z], v / v_raw
    [N, X, Y, Z, 3] (voxels), spacing: scalar mm per voxel.
    """
    sw = np.asarray(seg_warped)
    st = np.asarray(seg_template)
    if st.ndim == sw.ndim:
        st = st[0]
    sp_pred, sp_true = np.asarray(seg_pair_pred), np.asarray(seg_pair_true)
    s = float(spacing)
    jac = _np(jac)
    err = _np(inv_err_vox)
    if err.ndim == 5:
        err = np.linalg.norm(err, axis=-1)
    err = err.reshape((err.shape[0], -1)) * s

    def centrality(field):
        f = _np(field)
        return np.asarray([float(np.mean(np.linalg.norm(f.mean(axis=0), axis=-1))) * s])

    return {"template_dice": np.asarray([mean_fg_dice(sw[i], st, n_classes)
                                         for i in range(sw.shape[0])]),
            "pair_dice": np.asarray([mean_fg_dice(sp_pred[p], sp_true[p], n_classes)
                                     for p in range(sp_pred.shape[0])]),
            "neg_jac_pct": 100.0 * np.mean((jac <= 0).reshape((jac.shape[0], -1)), axis=1),
            "inv_consistency_mm": err.mean(axis=1),
            "inv_consistency_max_mm": err.max(axis=1),
            "centrality_mm": centrality(v),
            "raw_centrality_mm": centrality(v_raw)}


def deform_pair_dice_loss(metrics, prefix):
    """Deformable best-checkpoint metric 1 - ``<prefix>/pair_dice`` (lower is better), or None if
    missing."""
    v = metrics.get("%s/pair_dice" % prefix)
    return None if v is None else 1.0 - float(v)


def summarize(errors, prefix):
    """Means of per-matrix error arrays, plus p95 of the RMS values, keyed '<prefix>/<name>'."""
    out = {}
    for k, v in errors.items():
        v = _np(v).reshape(-1)
        if v.size == 0:
            continue
        out["%s/%s" % (prefix, k)] = float(np.mean(v))
        if k.endswith("rms_mm") and not k.startswith("cycle"):
            out["%s/%s_p95" % (prefix, k)] = float(np.percentile(v, 95))
    return out


def group_metrics(A, A_star, T_target, S, prefix, n_triplets=64, seed=0, affine=False):
    """`summarize(group_errors(...), prefix)`: center/pair rms, rot, trans and cycle rms (plus
    the affine errors with ``affine=True``)."""
    return summarize(group_errors(A, A_star, T_target, S, n_triplets, seed, affine), prefix)


def concat_errors(list_of_errors):
    """Pool per-matrix error dicts (e.g. of several validation groups)."""
    keys = list_of_errors[0].keys()
    return {k: np.concatenate([_np(e[k]).reshape(-1) for e in list_of_errors]) for k in keys}


class MeanAccumulator:
    """Running means of scalar dicts (train-step losses between log points)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = {}
        self._count = {}

    def update(self, values):
        for k, v in values.items():
            v = float(v)
            self._sum[k] = self._sum.get(k, 0.0) + v
            self._count[k] = self._count.get(k, 0) + 1

    def result(self):
        return {k: self._sum[k] / self._count[k] for k in self._sum}

    def __len__(self):
        return len(self._sum)
