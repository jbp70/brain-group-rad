"""TF-free numpy reference of the intensity augmentation (`params.IntensityAugConfig`).

A second, independent implementation of the chain in `data_loading.tf_intensity_aug` (and of
`dali_loader.intensity_augment`), used by the unit tests and the GPU check (m) of
tests/gpu/verify_dali_pipeline.py. Everything is per image ([X, Y, Z] or [X, Y, Z, C]) in
float64; parameters come from one row of uniforms (`params.AUG_COLUMNS`).

The low-res op is written as a real two-stage resize (build the low-res array with NN index
arrays, then a per-axis pixel-center linear upsample with edge clamp), not as the fused gather
of the TF module, so the tests compare two different derivations.
"""
import numpy as np

from data_loading import params as P

C = P.AUG_COLUMNS
LOWRES_ROIS = ("subpixel", "integer", "subpixel_centered")


def _factor(u, prefix, prob, low, high):
    """(applied, factor) of `dali_loader._intensity_factor` from a row of uniforms."""
    applied = bool(u[C[prefix + "_apply"]] < prob)
    if not applied:
        return False, 1.0
    if u[C[prefix + "_low"]] < 0.5:
        return True, float(low + u[C[prefix + "_lo_val"]] * (1.0 - low))
    return True, float(1.0 + u[C[prefix + "_hi_val"]] * (high - 1.0))


def factors_from_uniforms(u, cfg):
    """Row of N_AUG_UNIFORMS uniforms -> dict of applied flags and values (D5 table)."""
    u = np.asarray(u, np.float64).reshape(-1)
    noise_on = bool(u[C["noise_apply"]] < cfg.noise_prob)
    b_on, b = _factor(u, "brightness", cfg.brightness_prob, *cfg.brightness_range)
    c_on, c = _factor(u, "contrast", cfg.contrast_prob, *cfg.contrast_range)
    lr_on = bool(u[C["low_res_apply"]] < cfg.low_res_prob)
    f = (cfg.low_res_min_factor + u[C["low_res_factor"]] * (1.0 - cfg.low_res_min_factor)) if lr_on else 1.0
    g_on, g = _factor(u, "gamma", cfg.gamma_prob, *cfg.gamma_range)
    return {"noise": noise_on, "noise_std": float(u[C["noise_std"]] * cfg.noise_std_max) if noise_on else 0.0,
            "brightness": b_on, "brightness_factor": b,
            "contrast": c_on, "contrast_factor": c,
            "low_res": lr_on, "low_res_factor": float(f),
            "gamma": g_on, "gamma_factor": g}


def sample_uniforms_np(rng, n):
    """[n, N_AUG_UNIFORMS] uniforms from a numpy RandomState / Generator (statistical tests)."""
    return np.asarray(rng.uniform(0.0, 1.0, size=(int(n), P.N_AUG_UNIFORMS)), np.float64)


def lowres_size(n, f):
    """Low-res length m = max(1, round(f n)), rounding half up (DALI rounds half away from 0)."""
    return max(1, int(np.floor(float(f) * float(n) + 0.5)))


def nn_source_index(n, f, roi="subpixel"):
    """NN source voxel of every low-res voxel l in [0, m) for one axis of length n.

    'subpixel': scale 1/f from voxel 0 (DALI subpixel_scale=True keeps the fractional scale);
    'integer': scale n/m (the rounded size); 'subpixel_centered': scale 1/f with the input ROI
    of width m/f centered in the axis. Source coordinate of output pixel center l + 0.5, floored,
    clamped to [0, n - 1].
    """
    if roi not in LOWRES_ROIS:
        raise ValueError("roi must be one of {}, got {!r}".format(LOWRES_ROIS, roi))
    m = lowres_size(n, f)
    if roi == "integer":
        start, scale = 0.0, float(n) / float(m)
    elif roi == "subpixel":
        start, scale = 0.0, 1.0 / float(f)
    else:
        start, scale = 0.5 * (float(n) - float(m) / float(f)), 1.0 / float(f)
    src = np.floor(start + (np.arange(m, dtype=np.float64) + 0.5) * scale).astype(np.int64)
    return np.clip(src, 0, n - 1)


def _linear_up_axis(x, axis, n):
    """Pixel-center linear resize of `axis` (length m) to length n, edge clamp, no antialias."""
    m = x.shape[axis]
    out_shape = list(x.shape)
    out_shape[axis] = n
    out = np.empty(out_shape, np.float64)
    for o in range(n):
        c = (o + 0.5) * float(m) / float(n) - 0.5
        lo = int(np.floor(c))
        w = c - lo
        a = min(max(lo, 0), m - 1)
        b = min(max(lo + 1, 0), m - 1)
        sl = [slice(None)] * x.ndim
        sl[axis] = o
        out[tuple(sl)] = (1.0 - w) * np.take(x, a, axis=axis) + w * np.take(x, b, axis=axis)
    return out


def lowres_np(img, f, roi="subpixel"):
    """Simulated low resolution: NN downsample to round(f n) per spatial axis (one isotropic f),
    then linear upsample back to n. Two-stage reference (spatial axes 0, 1, 2)."""
    x = np.asarray(img, np.float64)
    shape = x.shape[:3]
    idx = [nn_source_index(n, f, roi) for n in shape]
    low = x[np.ix_(*idx)]           # [m_x, m_y, m_z(, C)]: trailing channel axis kept
    for a, n in enumerate(shape):
        low = _linear_up_axis(low, a, n)
    return low


def _tent_down_axis(x, axis, m):
    """Triangle-filtered downsample of `axis` (length n) to m cells: cell l = sum_o U[o, l] x[o]
    / sum_o U[o, l], U the pixel-center linear-upsample weights m -> n (explicit loops)."""
    n = x.shape[axis]
    weights = [[] for _ in range(m)]                 # per low-res cell l: [(o, U[o, l])]
    for o in range(n):
        c = (o + 0.5) * float(m) / float(n) - 0.5
        lo = int(np.floor(c))
        w = c - lo
        weights[min(max(lo, 0), m - 1)].append((o, 1.0 - w))
        weights[min(max(lo + 1, 0), m - 1)].append((o, w))
    out_shape = list(x.shape)
    out_shape[axis] = m
    out = np.zeros(out_shape, np.float64)
    for l in range(m):
        sl = [slice(None)] * x.ndim
        sl[axis] = l
        total = sum(wt for _, wt in weights[l])
        if total <= 0.0:
            continue
        acc = np.zeros(out[tuple(sl)].shape, np.float64)
        for o, wt in weights[l]:
            acc += wt * np.take(x, o, axis=axis)
        out[tuple(sl)] = acc / total
    return out


def lowres_tent_np(img, f):
    """'tent' low-res reference (retrain-prereqs design 3.2): triangle-filtered downsample to
    round(f n) per spatial axis (0, 1, 2), then the pixel-center linear upsample back to n."""
    x = np.asarray(img, np.float64)
    shape = x.shape[:3]
    low = x
    for a, n in enumerate(shape):
        low = _tent_down_axis(low, a, lowres_size(n, f))
    for a, n in enumerate(shape):
        low = _linear_up_axis(low, a, n)
    return low


def contrast_np(img, factor):
    """(img - mean) factor + mean, clamped to the PRE-contrast [min, max]."""
    x = np.asarray(img, np.float64)
    mean = x.mean()
    return np.clip((x - mean) * factor + mean, x.min(), x.max())


def gamma_np(img, g):
    """`dali_loader._gamma`: pow on the [min, max]-normalised image, then mean / std restored
    (population std, ddof 0)."""
    x = np.asarray(img, np.float64)
    m0, s0, lo = x.mean(), x.std(), x.min()
    rng = x.max() - lo
    out = np.power((x - lo) / max(rng, 1e-7), g) * rng + lo
    m1, s1 = out.mean(), out.std()
    return (out - m1) * (s0 / max(s1, 1e-7)) + m0


def augment_np(img, u_row, noise, cfg, roi="subpixel"):
    """Full chain for one image: noise -> brightness -> contrast -> low-res -> gamma -> clamp.

    ``noise``: unit-normal field of the image's shape (scaled by the drawn std when applied).
    ``cfg.enabled`` False returns the input unchanged (no clamp), as DALI. Returns float64.
    """
    x = np.asarray(img, np.float64)
    if cfg is None or not cfg.enabled:
        return x.copy()
    p = factors_from_uniforms(u_row, cfg)
    if cfg.noise and p["noise"]:
        x = x + p["noise_std"] * np.asarray(noise, np.float64)
    if cfg.brightness:
        x = x * p["brightness_factor"]
    if cfg.contrast and p["contrast"]:
        x = contrast_np(x, p["contrast_factor"])
    if cfg.low_res and p["low_res"]:
        x = lowres_np(x, p["low_res_factor"], roi)
    if cfg.gamma:
        x = gamma_np(x, p["gamma_factor"])
    return np.clip(x, 0.0, 1.0)
