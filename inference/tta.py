"""8-flip test-time augmentation (TTA) library (TTA design `.agents/tasks/tta/design.md` 2-5, 7.2).

TTA runs a cascade level once per group-level flip F = diag(+-1, +-1, +-1) (`FLIPS_8`, identity
first, (x, y, z) order), maps every pass back to the unflipped frame and averages:

* linear (design 2.3): per flip the unflipped raw matrices M_f = F A^F_raw F are centered /
  composed exactly as without TTA (A_f), divided by the previous level's result (gauge-free
  residual R_f = A_f A_prev^-1, float64), re-encoded in the model's own coordinates (10-D entries
  of I - q q^T, translation, log-scales, raw K entries; `encode_linear`), averaged (plain float64
  mean, eigh / Markley read-out of the 10-D matrix; `mean_linear`), decoded (`decode_linear`) and
  centered / composed once (`linear_tta_level`). The identity-only flip list (`IDENTITY_ONLY`)
  runs today's two lines unchanged (bit-identical to no TTA).
* deform (design 3): `unflip_field` (v -> F v(F p), an involution) for the SVF accumulation of
  `inference.streaming.StreamedLevel.deform_tta`; `flip_np` flips host level inputs.
* sign vectors of the head outputs (design 2.1, `linear_flip_signs`; equal to Jay's patterns),
  flip-consistency diagnostics (design 5: `linear_spread`, `gauge_barycenters`, `gauge_size`,
  `summarize_spread`) and the training-flip check (design 7.4: `train_flip_probs`,
  `flip_prob_warnings`).

Module-level imports are numpy-only (the CLI and TF-free tests import this module); functions
that need TensorFlow import it in their body.
"""
import itertools
import logging

import numpy as np

logger = logging.getLogger(__name__)

FLIPS_8 = tuple(itertools.product((False, True), repeat=3))     # identity first, (x, y, z)
IDENTITY_ONLY = ((False, False, False),)
MODES = ("none", "flip8")
LINEAR_AVERAGING = "gauge-free residual, float64, eigh (Markley) read-out"
SVF_ACCUMULATION = "float32"
# tta_spread.csv / tta_record.spread columns (design 5; review v2 NIT 3: spread_max_mm linear
# only, spread_p99_mm deform only).
SPREAD_COLUMNS = ("spread_rms_mm", "spread_max_mm", "spread_p99_mm", "shift_mm", "rot_deg_rms",
                  "trans_mm_rms", "scale_pct_rms", "shear_pct_rms", "rot_gap", "readout_dev_deg")
# Upper triangle (row-major) of the symmetric 4x4 over the quaternion (w, x, y, z): the 10-D order.
IU10 = ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
_IU_ROWS = np.array([i for i, _ in IU10])
_IU_COLS = np.array([j for _, j in IU10])
_TASK_PARAMS = {"rigid": 13, "affine": 19}
# design 7.4 warning texts
_WARN_PARTLY = "flips partly out of training distribution"
_WARN_NEVER = "flips never seen in training; TTA averages out-of-distribution predictions"
_WARN_UNKNOWN = "training flip probability unknown"


def _flip3(flip):
    flip = tuple(bool(b) for b in flip)
    if len(flip) != 3:
        raise ValueError("a flip needs 3 booleans (x, y, z), got %r" % (flip,))
    return flip


def _check_task(task):
    if task not in _TASK_PARAMS:
        raise ValueError("task must be 'rigid' or 'affine', got %r" % (task,))
    return _TASK_PARAMS[task]


class TTAConfig(object):
    """TTA settings of a run: ``mode`` ('none' | 'flip8'), the cascade ``levels`` it applies to
    and the ``flips`` (default `FLIPS_8`; `IDENTITY_ONLY` in tests)."""

    def __init__(self, mode, levels, flips=FLIPS_8):
        if mode not in MODES:
            raise ValueError("TTA mode must be one of %s, got %r" % (MODES, mode))
        self.mode = mode
        self.levels = tuple(int(lv) for lv in levels)
        self.flips = tuple(_flip3(f) for f in flips)
        if not self.flips:
            raise ValueError("TTA needs at least one flip")

    def to_dict(self):
        return {"mode": self.mode, "levels": list(self.levels),
                "flips": [[bool(b) for b in f] for f in self.flips]}


# ============================================================================ signs (design 2.1)
def flip_signs3(flip):
    """(f_x, f_y, f_z) float64, -1 on the flipped axes."""
    return np.array([-1.0 if b else 1.0 for b in _flip3(flip)], np.float64)


def linear_flip_signs(flip, task):
    """Sign vector sigma_f (float64 [13] rigid / [19] affine) with matrix(sigma_f * p) = F matrix(p) F.

    Derived from F (not a table): quaternion signs S_q = (1, f_y f_z, f_x f_z, f_x f_y), 10-D
    entry a_ij -> s_i s_j over `IU10`; translation f; log-scales +1; shears (k_xy, k_xz, k_yz)
    -> (f_x f_y, f_x f_z, f_y f_z)."""
    n = _check_task(task)
    fx, fy, fz = flip_signs3(flip)
    sq = np.array([1.0, fy * fz, fx * fz, fx * fy])
    rot = sq[_IU_ROWS] * sq[_IU_COLS]
    out = np.concatenate([rot, [fx, fy, fz]])
    if n == 19:
        out = np.concatenate([out, np.ones(3), [fx * fy, fx * fz, fy * fz]])
    return out


def flip_matrix_np(flip):
    """F = diag(f_x, f_y, f_z, 1) float64 [4, 4]."""
    return np.diag(np.concatenate([flip_signs3(flip), [1.0]]))


def _conj_signs(flip):
    f = np.concatenate([flip_signs3(flip), [1.0]])
    return np.outer(f, f)


def unflip_matrices(M, flip):
    """F M F for [..., 4, 4] matrices (numpy: dtype kept; TF tensors: tensor of M's dtype), as an
    exact sign change of entries. The identity flip returns ``M`` itself."""
    flip = _flip3(flip)
    if not any(flip):
        return M
    signs = _conj_signs(flip)
    if isinstance(M, np.ndarray):
        return M * signs.astype(M.dtype)
    import tensorflow as tf
    M = tf.convert_to_tensor(M)
    return M * tf.constant(signs, M.dtype)


# ============================================================================ host / field flips
def flip_np(x, flip):
    """Reverse the flipped spatial axes (1, 2, 3) of host arrays [k, X, Y, Z, C]: C-contiguous float32."""
    flip = _flip3(flip)
    x = np.asarray(x, np.float32)
    idx = (slice(None),) + tuple(slice(None, None, -1) if b else slice(None) for b in flip)
    return np.ascontiguousarray(x[idx])


def unflip_field(v, flip):
    """v^F(p) = F v(F p) for fields [k, X, Y, Z, 3] in voxel units (TF): reverse the flipped spatial
    axes and negate the flipped components. An involution; the identity flip returns ``v``."""
    flip = _flip3(flip)
    if not any(flip):
        return v
    import tensorflow as tf
    v = tf.convert_to_tensor(v)
    axes = [i + 1 for i, b in enumerate(flip) if b]
    return tf.reverse(v, axis=axes) * tf.constant(flip_signs3(flip), v.dtype)


# ============================================================================ encoding (2.3, 2.4)
def _sym4(entries):
    """10-D entries [..., 10] -> symmetric [..., 4, 4] (float64)."""
    e = np.asarray(entries, np.float64)
    out = np.zeros(e.shape[:-1] + (4, 4))
    out[..., _IU_ROWS, _IU_COLS] = e
    out[..., _IU_COLS, _IU_ROWS] = e
    return out


def encode_linear(R, task):
    """Residuals [N, 4, 4] = [L | t] -> float64 params [N, 13] (rigid) / [N, 19] (affine).

    L = Q K by numpy QR with the sign fix of `runtime.metrics.qr_factors` (positive diagonal of K);
    10-D entries of I - q q^T for Q (`tfm.encode_10d`, float64); t; affine adds log diag K and
    the RAW entries (K01, K02, K12) (the `scale_shear_matrix` convention, so
    `affine_matrix(encode_linear(R)) == R`). Rigid drops K (max |K - I| logged at DEBUG). Rows
    with a non-finite R, det L <= 0 or a failing QR are NaN (the caller marks the flip)."""
    import tensorflow as tf
    from models import transforms as tfm
    n_par = _check_task(task)
    R = np.asarray(R, np.float64)
    if R.ndim != 3 or R.shape[1:] != (4, 4):
        raise ValueError("encode_linear needs [N, 4, 4] matrices, got %s" % (R.shape,))
    n = R.shape[0]
    out = np.full((n, n_par), np.nan)
    qs = np.zeros((n, 3, 3))
    ks = np.zeros((n, 3, 3))
    ok = np.zeros(n, bool)
    for i in range(n):
        L = R[i, :3, :3]
        if not np.all(np.isfinite(R[i])):
            continue
        try:
            if not np.linalg.det(L) > 0.0:
                continue
            q, k = np.linalg.qr(L)
        except np.linalg.LinAlgError:
            continue
        sgn = np.where(np.diag(k) < 0, -1.0, 1.0)
        qs[i], ks[i] = q * sgn[None, :], sgn[:, None] * k
        ok[i] = bool(np.all(np.isfinite(qs[i])) and np.all(np.diag(ks[i]) > 0.0))
    if not np.any(ok):
        return out
    rot10 = tfm.encode_10d(tf.constant(qs[ok], tf.float64)).numpy()
    out[ok, :10] = rot10
    out[ok, 10:13] = R[ok, :3, 3]
    if n_par == 19:
        k = ks[ok]
        out[ok, 13:16] = np.log(np.diagonal(k, axis1=-2, axis2=-1))
        out[ok, 16] = k[:, 0, 1]
        out[ok, 17] = k[:, 0, 2]
        out[ok, 18] = k[:, 1, 2]
    elif logger.isEnabledFor(logging.DEBUG):
        logger.debug("rigid residual max |K - I| = %.3g (K dropped)",
                     float(np.max(np.abs(ks[ok] - np.eye(3)))))
    return out


def mean_linear(params_list):
    """Plain float64 mean of re-encoded params (list over flips of [N, P], summed in the given
    order) with the eigh (Markley) rotation read-out (design 2.4): the 10-D entries of the mean are
    replaced by those of I - q q^T, q = eigenvector of the smallest eigenvalue of the averaged
    symmetric 4x4. Returns (p_bar float64 [N, P], {'rot_gap': [N], 'readout_dev_deg': [N]});
    rot_gap = (lambda_2 - lambda_1) / lambda_max, readout_dev_deg = angle between
    `tfm.decode_10d` of the averaged matrix and the eigh rotation. Non-finite rows stay NaN."""
    import tensorflow as tf
    from models import transforms as tfm
    from runtime import metrics as RM
    params = [np.asarray(p, np.float64) for p in params_list]
    if not params:
        raise ValueError("mean_linear needs at least one parameter set")
    acc = np.zeros_like(params[0])
    for p in params:
        acc = acc + p
    p_bar = acc / float(len(params))
    n = p_bar.shape[0]
    gap = np.full(n, np.nan)
    dev = np.full(n, np.nan)
    ok = np.all(np.isfinite(p_bar), axis=1)
    p_bar[~ok] = np.nan
    if not np.any(ok):
        return p_bar, {"rot_gap": gap, "readout_dev_deg": dev}
    mean10 = p_bar[ok, :10].copy()
    w, V = np.linalg.eigh(_sym4(mean10))                 # ascending eigenvalues
    q = V[:, :, 0]
    proj = np.eye(4)[None] - q[:, :, None] * q[:, None, :]
    p_bar[np.flatnonzero(ok)[:, None], np.arange(10)[None, :]] = proj[:, _IU_ROWS, _IU_COLS]
    gap[ok] = (w[:, 1] - w[:, 0]) / np.maximum(np.abs(w[:, -1]), 1e-300)
    r_adj = tfm.decode_10d(tf.constant(mean10, tf.float64)).numpy()
    r_eig = tfm.decode_10d(tf.constant(p_bar[ok, :10], tf.float64)).numpy()
    dev[ok] = RM.rotation_angle_deg(r_adj @ np.swapaxes(r_eig, -1, -2))
    return p_bar, {"rot_gap": gap, "readout_dev_deg": dev}


def decode_linear(p_bar, task, dtype=np.float32):
    """Params [N, 13|19] -> matrices [N, 4, 4] via `tfm.rigid_matrix` / `tfm.affine_matrix` in
    float64, cast to ``dtype`` (float32 by default)."""
    import tensorflow as tf
    from models import transforms as tfm
    _check_task(task)
    p = tf.constant(np.asarray(p_bar, np.float64), tf.float64)
    m = tfm.rigid_matrix(p) if task == "rigid" else tfm.affine_matrix(p)
    return m.numpy().astype(dtype)


def _inv64(a):
    """Batched float64 inverse; images whose inverse fails are NaN."""
    try:
        return np.linalg.inv(a)
    except np.linalg.LinAlgError:
        out = np.full_like(a, np.nan)
        for i in range(a.shape[0]):
            try:
                out[i] = np.linalg.inv(a[i])
            except np.linalg.LinAlgError:
                pass
        return out


def linear_tta_level(M_flips, a_prev, centering, task, flips=FLIPS_8, dtype=None):
    """Linear TTA of one level and task (design 2.3 steps 3-9).

    M_flips: list (flip order) of UNFLIPPED raw [N, 4, 4] matrices (tensors or arrays); a_prev:
    None (first run level) or the previous level's result [N, 4, 4]; centering: the model's
    `centering` (any `GroupCentering`). ``dtype``: dtype of the result (default: a_prev's dtype,
    float32 at the first level, as without TTA; float64 in the gauge tests).
    The identity-only flip list returns exactly ``tf.cast(centering(M), float32)`` /
    ``CS.compose_linear(M, a_prev, centering)`` (no re-encoding). Returns dict(A tensor, A_flips:
    list of float64 [N, 4, 4] per-flip composed and centered results (None for identity-only),
    params: list of float64 residual encodings [N, P] (None), p_bar (None), mean_diag (None),
    nonfinite: indices of flips with any non-finite image)."""
    import tensorflow as tf
    from models import cascade as CS
    flips = tuple(_flip3(f) for f in flips)
    if len(M_flips) != len(flips):
        raise ValueError("M_flips has %d entries for %d flips" % (len(M_flips), len(flips)))
    if a_prev is not None:
        a_prev = tf.convert_to_tensor(a_prev)
    if dtype is None:
        dtype = tf.float32 if a_prev is None else a_prev.dtype

    def compose(M):
        if a_prev is None:
            return tf.cast(centering(M), dtype)
        return CS.compose_linear(M, a_prev, centering)

    if flips == IDENTITY_ONLY:
        A = compose(M_flips[0])
        bad = [] if bool(np.all(np.isfinite(np.asarray(A)))) else [0]
        return {"A": A, "A_flips": None, "params": None, "p_bar": None, "mean_diag": None,
                "nonfinite": bad}
    inv_prev = None if a_prev is None else _inv64(np.asarray(a_prev, np.float64))
    A_flips, params, nonfinite = [], [], []
    for k, M in enumerate(M_flips):
        a_f = np.asarray(compose(M), np.float64)
        r_f = a_f if inv_prev is None else a_f @ inv_prev
        with np.errstate(invalid="ignore", divide="ignore"):
            p_f = encode_linear(r_f, task)
        if not (np.all(np.isfinite(a_f)) and np.all(np.isfinite(p_f))):
            nonfinite.append(k)
        A_flips.append(a_f)
        params.append(p_f)
    p_bar, diag = mean_linear(params)
    np_dtype = tf.as_dtype(dtype).as_numpy_dtype
    r_bar = decode_linear(p_bar, task, np_dtype)
    r_bar[~np.all(np.isfinite(p_bar), axis=1)] = np.nan
    A = compose(tf.constant(r_bar))
    return {"A": A, "A_flips": A_flips, "params": params, "p_bar": p_bar, "mean_diag": diag,
            "nonfinite": nonfinite}


# ============================================================================ diagnostics (5)
def _point_rms(dM, S):
    """sqrt(tr(dM S dM^T)) for [..., 4, 4] differences; S [4, 4] or per image [N, 4, 4]."""
    dM = np.asarray(dM, np.float64)
    S = np.asarray(S, np.float64)
    ms = np.einsum("...ij,...jk,...ik->...", dM, S, dM)
    return np.sqrt(np.maximum(ms, 0.0))


def linear_spread(A_flips, A_bar, S, params_flips, p_bar, task, mean_diag=None):
    """Per-image flip-consistency diagnostics of one linear level / task (design 5), float64 [N]:
    spread_rms_mm, spread_max_mm (max over flips), shift_mm (identity flip = A_flips[0] vs the
    TTA result), rot_deg_rms (RMS over flips of the angle of Q_f Q_bar^T, QR rotation factors of
    the decoded residuals), trans_mm_rms, and for affine scale_pct_rms / shear_pct_rms (|delta
    log-scale| x100, |delta K_ab| x100 of the residual params); with ``mean_diag`` (from
    `mean_linear`) also rot_gap and readout_dev_deg. S: [4, 4] or per-image [N, 4, 4]."""
    from runtime import metrics as RM
    n_par = _check_task(task)
    Af = np.stack([np.asarray(a, np.float64) for a in A_flips])          # [F, N, 4, 4]
    Ab = np.asarray(A_bar, np.float64)
    rms = _point_rms(Af - Ab[None], S)                                     # [F, N]
    out = {"spread_rms_mm": np.sqrt(np.mean(rms ** 2, axis=0)),
           "spread_max_mm": np.max(rms, axis=0),
           "shift_mm": rms[0]}
    P = np.stack([np.asarray(p, np.float64) for p in params_flips])      # [F, N, P]
    pb = np.asarray(p_bar, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        Rf = np.stack([decode_linear(p, task, np.float64) for p in P])
        Rb = decode_linear(pb, task, np.float64)
        qf, _, _ = RM.qr_factors(Rf[..., :3, :3])
        qb, _, _ = RM.qr_factors(Rb[..., :3, :3])
        ang = RM.rotation_angle_deg(qf @ np.swapaxes(qb, -1, -2)[None])
    out["rot_deg_rms"] = np.sqrt(np.mean(ang ** 2, axis=0))
    dt = np.linalg.norm(P[..., 10:13] - pb[None, :, 10:13], axis=-1)
    out["trans_mm_rms"] = np.sqrt(np.mean(dt ** 2, axis=0))
    if n_par == 19:
        ds = 100.0 * np.linalg.norm(P[..., 13:16] - pb[None, :, 13:16], axis=-1)
        dk = 100.0 * np.linalg.norm(P[..., 16:19] - pb[None, :, 16:19], axis=-1)
        out["scale_pct_rms"] = np.sqrt(np.mean(ds ** 2, axis=0))
        out["shear_pct_rms"] = np.sqrt(np.mean(dk ** 2, axis=0))
    if mean_diag is not None:
        out["rot_gap"] = np.asarray(mean_diag["rot_gap"], np.float64)
        out["readout_dev_deg"] = np.asarray(mean_diag["readout_dev_deg"], np.float64)
    return out


def gauge_barycenters(M_flips, a_prev, iters):
    """mu_f = `tfm.barycenter`(M_f A_prev) in float64 with ``iters`` iterations (the model's
    `centering_iters`), per flip: the left factor that the per-flip centering removes."""
    import tensorflow as tf
    from models import transforms as tfm
    prev = None if a_prev is None else np.asarray(a_prev, np.float64)
    out = []
    for M in M_flips:
        m = np.asarray(M, np.float64)
        if prev is not None:
            m = m @ prev
        out.append(tfm.barycenter(tf.constant(m, tf.float64), int(iters)).numpy())
    return out


def gauge_size(mu_flips):
    """(deg, mm): max over flips of the rotation angle / translation norm of mu_0^-1 mu_f
    (mu_0 = the identity flip's barycenter)."""
    from runtime import metrics as RM
    mus = [np.asarray(m, np.float64) for m in mu_flips]
    inv0 = np.linalg.inv(mus[0])
    deg, mm = 0.0, 0.0
    for mu in mus:
        g = inv0 @ mu
        deg = max(deg, float(RM.rotation_angle_deg(g[:3, :3])))
        mm = max(mm, float(np.linalg.norm(g[:3, 3])))
    return deg, mm


def summarize_spread(rows):
    """{'l<level>/<task>': {column: {mean, median, p95, max}}} over the rows (dicts with 'level',
    'task' and `SPREAD_COLUMNS` values; None / '' / NaN skipped, columns without values omitted)."""
    groups = {}
    for r in rows:
        groups.setdefault("l%d/%s" % (int(r["level"]), r["task"]), []).append(r)
    out = {}
    for key in sorted(groups):
        cols = {}
        for c in SPREAD_COLUMNS:
            vals = []
            for r in groups[key]:
                v = r.get(c)
                if v is None or v == "":
                    continue
                v = float(v)
                if np.isfinite(v):
                    vals.append(v)
            if vals:
                a = np.asarray(vals, np.float64)
                cols[c] = {"mean": float(a.mean()), "median": float(np.median(a)),
                           "p95": float(np.percentile(a, 95)), "max": float(a.max())}
        out[key] = cols
    return out


# ============================================================================ training flips (7.4)
def train_flip_probs(chain):
    """{level: effective training flip probability or None} from {level: (config path or None,
    config dict or None)}: args.flip_prob, 0.0 with args.overfit_one_group, None when the config
    or the key is missing (TF-free)."""
    out = {}
    for lv, entry in chain.items():
        config = entry[1] if entry is not None else None
        args = (config or {}).get("args") if isinstance(config, dict) else None
        p = None
        if isinstance(args, dict):
            if args.get("overfit_one_group"):
                p = 0.0
            elif args.get("flip_prob") is not None:
                try:
                    p = float(args["flip_prob"])
                except (TypeError, ValueError):
                    p = None
        out[int(lv)] = p
    return out


def flip_prob_warnings(probs, tta_levels):
    """WARNING texts (design 7.4) for the TTA levels whose stage was trained with flip p < 0.5,
    p == 0 or an unknown p; [] when every TTA level saw p >= 0.5."""
    out = []
    for lv in tta_levels:
        p = probs.get(int(lv))
        if p is None:
            out.append("TTA level %d: %s" % (int(lv), _WARN_UNKNOWN))
        elif p == 0.0:
            out.append("TTA level %d: training flip probability 0: %s" % (int(lv), _WARN_NEVER))
        elif p < 0.5:
            out.append("TTA level %d: training flip probability %g < 0.5: %s" % (int(lv), p, _WARN_PARTLY))
    return out
