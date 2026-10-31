"""Image, label and field metrics (numpy/scipy only; definitions in evaluation/evaluate.py)."""
from typing import Dict, Sequence

import numpy as np
from scipy import ndimage


def is_cortical(labels: Sequence[int]) -> np.ndarray:
    """DKT31 cortical labels (1000-2999); everything else (aseg) counts as subcortical."""
    lab = np.asarray(labels)
    return (lab >= 1000) & (lab < 3000)


def dice_per_label(a: np.ndarray, b: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """Hard Dice of every label in ``labels`` (float64), NaN where absent in both maps."""
    if a.shape != b.shape:
        raise ValueError("label maps differ in shape: %s vs %s" % (a.shape, b.shape))
    labels = np.asarray(labels, np.int64)
    a = a.ravel()
    b = b.ravel()
    m = int(max(int(a.max()), int(b.max()), int(labels.max()))) + 1
    ca = np.bincount(a, minlength=m).astype(np.float64)
    cb = np.bincount(b, minlength=m).astype(np.float64)
    inter = np.bincount(a[a == b], minlength=m).astype(np.float64)
    tot = ca[labels] + cb[labels]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, 2.0 * inter[labels] / np.where(tot > 0, tot, 1.0), np.nan)


def dice_summary(d: np.ndarray, labels: Sequence[int], prefix: str) -> Dict[str, float]:
    """Mean over all / cortical / subcortical labels (NaN-aware) -> {prefix, prefix_cortical, ...}."""
    cort = is_cortical(labels)

    def m(x):
        x = x[np.isfinite(x)]
        return float(x.mean()) if x.size else float("nan")

    return {prefix: m(d), prefix + "_cortical": m(d[cort]), prefix + "_subcortical": m(d[~cort])}


def hd95_per_label(a: np.ndarray, b: np.ndarray, labels: Sequence[int], spacing: Sequence[float]) -> np.ndarray:
    """Symmetric 95th-percentile surface distance (mm) per label: max(p95(d_ab), p95(d_ba)).

    NaN when the label is missing in either map."""
    out = np.full(len(labels), np.nan)
    for k, lab in enumerate(labels):
        ma, mb = a == lab, b == lab
        if not ma.any() or not mb.any():
            continue
        idx = np.nonzero(ma | mb)
        sl = tuple(slice(max(int(i.min()) - 1, 0), int(i.max()) + 2) for i in idx)
        ca, cb = np.pad(ma[sl], 1), np.pad(mb[sl], 1)
        sa = ca & ~ndimage.binary_erosion(ca)
        sb = cb & ~ndimage.binary_erosion(cb)
        da = ndimage.distance_transform_edt(~sa, sampling=spacing)
        db = ndimage.distance_transform_edt(~sb, sampling=spacing)
        out[k] = max(float(np.percentile(db[sa], 95)), float(np.percentile(da[sb], 95)))
    return out


def ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Global Pearson correlation over ``mask``."""
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    if x.size < 2:
        return float("nan")
    x -= x.mean()
    y -= y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


def lncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray, window: int = 9, eps: float = 1e-5) -> float:
    """Mean squared local correlation (VoxelMorph LNCC, cubic window, zero padding) over ``mask``."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    def f(x):
        return ndimage.uniform_filter(x, size=window, mode="constant", cval=0.0)

    mu_a, mu_b = f(a), f(b)
    cross = f(a * b) - mu_a * mu_b
    va = f(a * a) - mu_a * mu_a
    vb = f(b * b) - mu_b * mu_b
    cc = cross * cross / (np.maximum(va, 0) * np.maximum(vb, 0) + eps)
    return float(cc[mask].mean()) if mask.any() else float("nan")


def folding_pct(det: np.ndarray, mask: np.ndarray) -> float:
    """Percent of mask voxels with det J <= 0."""
    return 100.0 * float(np.mean(det[mask] <= 0)) if mask.any() else float("nan")


def logjac_std(det: np.ndarray, mask: np.ndarray) -> float:
    """Std of log det J over mask voxels with det J > 0."""
    v = det[mask & (det > 0)]
    return float(np.std(np.log(v))) if v.size else float("nan")


class MajorityVote:
    """Per-voxel label votes of N warped label maps [N, X, Y, Z]: full and leave-one-out majority.

    Ties go to the smaller label value (background 0 first)."""

    def __init__(self, warped_labels: np.ndarray):
        n = warped_labels.shape[0]
        if n > 32767:
            raise ValueError("too many subjects for int16 votes")
        self.labels = np.unique(warped_labels)
        lut = np.zeros(int(self.labels.max()) + 1, np.int64)
        lut[self.labels] = np.arange(self.labels.size)
        self.lut = lut
        self.shape = warped_labels.shape[1:]
        v = int(np.prod(self.shape))
        self.ar = np.arange(v)
        self.counts = np.zeros((self.labels.size, v), np.int16)
        self.warped = warped_labels
        for i in range(n):
            self.counts[lut[warped_labels[i].ravel()], self.ar] += 1

    def _arg(self) -> np.ndarray:
        return self.labels[np.argmax(self.counts, axis=0)].reshape(self.shape).astype(self.warped.dtype)

    def full(self) -> np.ndarray:
        return self._arg()

    def loo(self, i: int) -> np.ndarray:
        idx = self.lut[self.warped[i].ravel()]
        self.counts[idx, self.ar] -= 1
        try:
            return self._arg()
        finally:
            self.counts[idx, self.ar] += 1
