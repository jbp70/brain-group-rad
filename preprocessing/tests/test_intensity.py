import numpy as np
import pytest

from preprocessing.intensity import (antialias_sigma_mm, antialias_sigma_vox, antialias_smooth,
                                     normalize_intensity)


def test_normalize_formula_full_mask():
    x = np.arange(201, dtype=np.float32).reshape(201, 1, 1)
    m = np.ones_like(x)
    y, st = normalize_intensity(x, m, 99.5)
    assert st["p_hi"] == pytest.approx(199.0)
    np.testing.assert_allclose(y, np.clip(x / 199.0, 0, 1), rtol=0, atol=1e-7)
    assert y.dtype == np.float32
    assert st["frac_clipped_high"] == pytest.approx(1 / 201)
    assert st["frac_clipped_low"] == 0.0


def test_normalize_outside_mask_zero_and_inmask_percentile():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 100, (10, 10, 10)).astype(np.float32)
    m = np.zeros_like(x, bool)
    m[2:8, 2:8, 2:8] = True
    x[~m] = 1e6  # outside values must not affect the percentile
    y, st = normalize_intensity(x, m)
    assert st["p_hi"] == pytest.approx(np.percentile(x[m], 99.5))
    assert np.all(y[~m] == 0)
    assert y.max() == 1.0 and y.min() >= 0


def test_normalize_lower_percentile_and_errors():
    x = np.linspace(10, 110, 1000).reshape(10, 10, 10)
    y, st = normalize_intensity(x, np.ones_like(x), 99.0, 1.0)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    np.testing.assert_allclose(y, np.clip((x - lo) / (hi - lo), 0, 1), atol=1e-6)
    with pytest.raises(ValueError):
        normalize_intensity(x, np.zeros_like(x))
    with pytest.raises(ValueError):
        normalize_intensity(np.zeros((3, 3, 3)), np.ones((3, 3, 3)))
    with pytest.raises(ValueError):
        normalize_intensity(x, np.ones((2, 2, 2)))


def test_antialias_sigma():
    assert antialias_sigma_mm(0.6, 1.0)[0] == pytest.approx(0.3397, abs=1e-4)
    assert antialias_sigma_mm(0.8, 1.0)[0] == pytest.approx(0.2548, abs=1e-4)
    assert antialias_sigma_mm(1.0, 1.0)[0] == 0.0
    sv = antialias_sigma_vox([0.6, 0.97, 1.0])
    assert sv[0] == pytest.approx(0.3397 / 0.6, abs=1e-3)
    assert sv[1] == 0.0 and sv[2] == 0.0


def test_antialias_smooth_applies_only_when_needed():
    d = np.zeros((9, 9, 9), np.float32)
    d[4, 4, 4] = 1
    out, sig, applied = antialias_smooth(d, [1.0, 1.0, 1.0])
    assert not applied and out is d
    out, sig, applied = antialias_smooth(d, [0.7, 0.7, 0.7])
    assert applied and out[4, 4, 4] < 1 and out.sum() == pytest.approx(1.0, abs=1e-4)
