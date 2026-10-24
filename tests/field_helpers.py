"""Shared displacement-field helpers of the TF tests (not a test module; numpy / scipy only).

`smooth_field` moved here unchanged from tests/test_diffeo.py (milestone 9 plan item 2) so
tests/test_lean_sampler.py and tests/test_integration_steps.py can use it too.
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def smooth_field(shape, rng, n=1, sigma=3.0, max_vox=2.0):
    """[n, X, Y, Z, 3] Gaussian-filtered noise scaled so max_voxel |v| = max_vox (float64)."""
    v = np.stack([np.stack([gaussian_filter(rng.normal(size=shape), sigma, mode="nearest")
                            for _ in range(3)], axis=-1) for _ in range(n)])
    return v * (max_vox / np.linalg.norm(v, axis=-1).max())
