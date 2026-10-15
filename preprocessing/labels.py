"""Label-map helpers: nearest-neighbour fallback resampling and dtype casting.

Example:
    from preprocessing.labels import resample_labels_nearest, to_label_dtype
    lab = resample_labels_nearest(seg, seg_affine, template_shape, template_affine)
    lab = to_label_dtype(lab, "uint16")
"""
from typing import Sequence

import numpy as np

from .nifti import resample_world


def resample_labels_nearest(src_data: np.ndarray, src_affine, ref_shape: Sequence[int],
                            ref_affine) -> np.ndarray:
    """Nearest-neighbour resample of a label map onto a reference grid (world space).

    Voxels mapping outside the source get label 0. No new label values appear.
    """
    return resample_world(np.asarray(src_data), src_affine, ref_shape, ref_affine, order=0, cval=0)


def to_label_dtype(arr: np.ndarray, dtype="uint16") -> np.ndarray:
    """Round to integers and cast; raises ValueError if values do not fit."""
    dt = np.dtype(dtype)
    r = np.rint(np.asarray(arr, dtype=np.float64))
    info = np.iinfo(dt)
    if r.size and (r.min() < info.min or r.max() > info.max):
        raise ValueError("label values [%d, %d] do not fit in %s" % (r.min(), r.max(), dt.name))
    return r.astype(dt)
