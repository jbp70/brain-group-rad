import numpy as np

from preprocessing.masks import derive_mask, dice, largest_component


def test_derive_mask_threshold_fill_and_largest_component():
    d = np.zeros((20, 20, 20))
    d[4:14, 4:14, 4:14] = 1.0
    d[8:10, 8:10, 8:10] = 0.0          # internal hole
    d[17, 17, 17] = 1.0                # isolated speck
    m = derive_mask(d)
    assert m.dtype == bool
    assert m[9, 9, 9]                  # hole filled
    assert not m[17, 17, 17]           # speck removed
    assert m.sum() == 1000
    m2 = derive_mask(d, fill_holes=False, largest_component_only=False)
    assert m2[17, 17, 17] and not m2[9, 9, 9]
    assert derive_mask(d, threshold=2.0).sum() == 0


def test_derive_mask_union_with_labels():
    d = np.zeros((10, 10, 10))
    d[2:6, 2:6, 2:6] = 1
    lab = np.zeros_like(d)
    lab[5:8, 2:6, 2:6] = 3   # touches the image component
    m = derive_mask(d, extra=lab)
    assert m.sum() == 6 * 4 * 4


def test_dice_and_largest_component():
    a = np.zeros((5, 5, 5), bool)
    b = np.zeros((5, 5, 5), bool)
    assert dice(a, b) == 1.0
    a[0:2] = True
    b[1:3] = True
    assert dice(a, b) == 0.5
    assert dice(a, ~a) == 0.0
    assert largest_component(np.zeros((3, 3, 3))).sum() == 0
