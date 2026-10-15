import numpy as np
import pytest

from preprocessing.labels import resample_labels_nearest, to_label_dtype


def test_nearest_resample_of_cropped_grid_reproduces_labels():
    rng = np.random.default_rng(1)
    full = rng.integers(0, 2036, (12, 14, 10)).astype(np.float64)
    a_full = np.array([[1.0, 0, 0, -6], [0, 1, 0, -7], [0, 0, 1, -5], [0, 0, 0, 1]])
    # A cropped copy (2 voxels per side) with the origin shifted accordingly.
    crop = full[2:-2, 2:-2, 2:-2]
    a_crop = a_full.copy()
    a_crop[:3, 3] += 2
    out = resample_labels_nearest(crop, a_crop, full.shape, a_full)
    np.testing.assert_array_equal(out[2:-2, 2:-2, 2:-2], crop)
    assert np.all(out[:2] == 0) and np.all(out[-2:] == 0)
    assert set(np.unique(out)) <= set(np.unique(full)) | {0}


def test_nearest_resample_flipped_reference():
    lab = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    a = np.eye(4)
    ref = np.array([[-1.0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    np.testing.assert_array_equal(resample_labels_nearest(lab, a, (2, 3, 4), ref), lab[::-1])


def test_to_label_dtype():
    out = to_label_dtype(np.array([0.0, 1.0000001, 2034.9999]), "uint16")
    assert out.dtype == np.uint16 and list(out) == [0, 1, 2035]
    with pytest.raises(ValueError):
        to_label_dtype(np.array([300.0]), "uint8")
    with pytest.raises(ValueError):
        to_label_dtype(np.array([-1.0]), "uint16")
