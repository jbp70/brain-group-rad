"""Binary mask helpers: mask derivation from skull-stripped images, connected
components and Dice.

Example:
    from preprocessing.masks import derive_mask, dice
    mask = derive_mask(norm, extra=seg35)   # (norm > 0) | (seg35 > 0), largest CC, holes filled
"""
from typing import Optional

import numpy as np
from scipy import ndimage


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 26-connected component (empty masks are returned as is)."""
    m = np.asarray(mask) > 0
    lab, n = ndimage.label(m, structure=np.ones((3, 3, 3), dtype=bool))
    if n <= 1:
        return m
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == int(np.argmax(counts))


def derive_mask(data: np.ndarray, threshold: float = 0.0, fill_holes: bool = True,
                largest_component_only: bool = True, extra: Optional[np.ndarray] = None
                ) -> np.ndarray:
    """Brain mask for an already skull-stripped image.

    mask = (data > threshold) | (extra > 0), then largest component, then
    hole filling. Returns a bool array.
    """
    m = np.asarray(data) > threshold
    if extra is not None:
        if extra.shape != m.shape:
            raise ValueError("extra mask shape %s != image shape %s" % (extra.shape, m.shape))
        m = m | (np.asarray(extra) > 0)
    if largest_component_only:
        m = largest_component(m)
    if fill_holes:
        m = ndimage.binary_fill_holes(m)
    return m.astype(bool)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice overlap of two binary masks; 1.0 when both are empty."""
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0
    return 2.0 * float(np.logical_and(a, b).sum()) / denom
