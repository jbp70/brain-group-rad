"""Rigid/affine metrics (design 7), numpy float64.

Per matrix M against its truth M_true (both [4, 4], centered mm):
* rms_mm: sqrt(tr(dM S dM^T)), the RMS displacement over the template brain voxels;
* rot_deg: angle of polar(L_M) polar(L_true)^T, polar(L) = U V^T from the SVD of the 3x3 part;
* trans_mm: |t_M - t_true|, the displacement of the rotation center (the image center).

Centered errors compare A_i with A_i*. Pairwise errors compare A_j^-1 A_i with T_j T_i^-1 over ALL
ordered pairs i != j. The cycle error (A numerics check, ~0 by construction) is the RMS of
P_ki P_jk P_ij - I over random distinct triplets.

`group_errors` returns per-matrix arrays (so validation can pool them over groups);
`summarize` turns them into means (+ p95 of the RMS values) under a key prefix.
"""
import numpy as np


def _np(x):
    return np.asarray(x, np.float64)


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


def group_errors(A, A_star, T_target, S, n_triplets=64, seed=0):
    """Per-matrix error arrays of one group (unprefixed keys)."""
    out = {}
    for k, v in centered_errors(A, A_star, S).items():
        out["center_" + k] = v
    for k, v in pairwise_errors(A, T_target, S).items():
        out["pair_" + k] = v
    out["cycle_rms_mm"] = cycle_errors(A, S, n_triplets, seed)["rms_mm"]
    return out


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


def group_metrics(A, A_star, T_target, S, prefix, n_triplets=64, seed=0):
    """`summarize(group_errors(...), prefix)`: center/pair rms, rot, trans and cycle rms."""
    return summarize(group_errors(A, A_star, T_target, S, n_triplets, seed), prefix)


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
