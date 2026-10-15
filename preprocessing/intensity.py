"""Intensity normalization and anti-alias smoothing (pure numpy/scipy).

Normalization (applied to every dataset, LUMIR included):
    y = clip((x - lo) / (hi - lo), 0, 1) inside the mask, 0 outside, float32
    hi = in-mask upper percentile (default 99.5), lo = 0 unless lower_percentile is set.

Anti-aliasing before downsampling to the 1 mm grid uses FWHM matching:
    sigma_mm = sqrt(max(t^2 - s^2, 0)) / (2 sqrt(2 ln 2)),  sigma_vox = sigma_mm / s
applied only on axes with spacing s < t * min_spacing_ratio. Scale changes
inside the affine are ignored when choosing sigma.

Example:
    y, stats = normalize_intensity(x, mask, upper_percentile=99.5)
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def normalize_intensity(x: np.ndarray, mask: np.ndarray, upper_percentile: float = 99.5,
                        lower_percentile: Optional[float] = None
                        ) -> Tuple[np.ndarray, Dict[str, float]]:
    """Scale ``x`` to [0, 1] using in-mask percentiles; outside-mask voxels are 0."""
    m = np.asarray(mask) > 0
    if m.shape != x.shape:
        raise ValueError("mask shape %s != image shape %s" % (m.shape, x.shape))
    if not m.any():
        raise ValueError("empty mask: cannot normalize")
    v = x[m].astype(np.float64)
    hi = float(np.percentile(v, upper_percentile))
    lo = 0.0 if lower_percentile is None else float(np.percentile(v, lower_percentile))
    if hi <= lo:
        raise ValueError("degenerate intensities: p_hi=%g <= p_lo=%g" % (hi, lo))
    y = np.clip((x.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)
    y[~m] = 0.0
    stats = {
        "p_hi": hi,
        "p_lo": lo,
        "upper_percentile": float(upper_percentile),
        "lower_percentile": None if lower_percentile is None else float(lower_percentile),
        "frac_clipped_high": float(np.mean(v > hi)),
        "frac_clipped_low": float(np.mean(v < lo)),
    }
    return y.astype(np.float32), stats


def antialias_sigma_mm(spacing, target: float = 1.0) -> np.ndarray:
    """Gaussian sigma (mm) per axis that brings spacing ``s`` to FWHM ``target``."""
    s = np.atleast_1d(np.asarray(spacing, dtype=float))
    return np.sqrt(np.maximum(target ** 2 - s ** 2, 0.0)) * FWHM_TO_SIGMA


def antialias_sigma_vox(spacing: Sequence[float], target: float = 1.0,
                        min_spacing_ratio: float = 0.95) -> np.ndarray:
    """Per-axis sigma in voxels; 0 on axes not finer than target * ratio."""
    s = np.asarray(spacing, dtype=float)
    sig = antialias_sigma_mm(s, target) / s
    sig[s >= target * min_spacing_ratio] = 0.0
    return sig


def antialias_smooth(data: np.ndarray, spacing: Sequence[float], target: float = 1.0,
                     min_spacing_ratio: float = 0.95) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Smooth sub-target-resolution axes. Returns (data, sigma_vox, applied)."""
    sig = antialias_sigma_vox(spacing, target, min_spacing_ratio)
    if not np.any(sig > 0):
        return data, sig, False
    out = ndimage.gaussian_filter(data.astype(np.float32), sigma=sig, mode="nearest")
    return out, sig, True
