import nibabel as nib
import numpy as np
import pytest

from preprocessing import nifti

from . import phantoms as ph


def test_affines_and_grid_match():
    a = np.eye(4)
    b = a.copy()
    b[0, 3] = 5e-4
    assert nifti.affines_match(a, b, 1e-3)
    b[0, 3] = 2e-3
    assert not nifti.affines_match(a, b, 1e-3)
    assert nifti.grid_matches((2, 3, 4), a, (2, 3, 4), a)
    assert not nifti.grid_matches((2, 3, 4), a, (2, 3, 5), a)


def test_save_like_sets_qform_equal_sform(tmp_path):
    aff = ph.target_affine()
    p = str(tmp_path / "x.nii.gz")
    nifti.save_like(np.zeros((3, 4, 5)), aff, p, np.uint8, xform_code=4)
    img = nib.load(p)
    np.testing.assert_allclose(img.header.get_qform(), aff, atol=1e-6)
    np.testing.assert_allclose(img.header.get_sform(), aff, atol=1e-6)
    assert int(img.header["sform_code"]) == 4 and img.get_data_dtype() == np.uint8
    assert nifti.qform_sform_consistent(img)
    np.testing.assert_allclose(nifti.voxel_sizes(np.diag([0.7, 0.8, 2.0, 1])), [0.7, 0.8, 2.0])


def test_qform_sform_mismatch_detected(tmp_path):
    img = nib.Nifti1Image(np.zeros((3, 3, 3), np.float32), np.eye(4))
    q = np.eye(4)
    q[0, 3] = 10
    img.set_qform(q, code=1)
    img.set_sform(np.eye(4), code=1)
    assert not nifti.qform_sform_consistent(img)


def test_load_data_squeezes_singleton_4d(tmp_path):
    p = str(tmp_path / "x4.nii.gz")
    nib.save(nib.Nifti1Image(np.ones((3, 4, 5, 1), np.float32), np.eye(4)), p)
    d, img = nifti.load_data(p)
    assert d.shape == (3, 4, 5)
    nib.save(nib.Nifti1Image(np.ones((3, 4, 5, 2), np.float32), np.eye(4)), p)
    with pytest.raises(ValueError):
        nifti.load_data(p)


def test_resample_world_exact_for_flip_and_shift():
    rng = np.random.default_rng(0)
    src = rng.integers(0, 50, (6, 7, 8)).astype(np.float32)
    a_src = np.eye(4)
    # Reference: x flipped and shifted by 1 voxel in y -> exact permutation of voxels.
    a_ref = np.array([[-1.0, 0, 0, 5], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
    out0 = nifti.resample_world(src, a_src, (6, 6, 8), a_ref, order=0)
    out1 = nifti.resample_world(src, a_src, (6, 6, 8), a_ref, order=1)
    exp = src[::-1, 1:7, :]
    np.testing.assert_array_equal(out0, exp)
    np.testing.assert_allclose(out1, exp, atol=1e-4)


def test_fsl_matrix_convention():
    # Neurological (det > 0): x flipped in FSL coordinates.
    s = nifti.fsl_scaled_voxel_matrix((10, 5, 5), np.diag([2.0, 1, 1, 1]))
    np.testing.assert_allclose(s @ [0, 0, 0, 1], [18, 0, 0, 1])
    np.testing.assert_allclose(s @ [9, 0, 0, 1], [0, 0, 0, 1])
    # Radiological (det < 0): plain scaling.
    s = nifti.fsl_scaled_voxel_matrix((10, 5, 5), np.diag([-2.0, 1, 1, 1]))
    np.testing.assert_allclose(s @ [3, 1, 2, 1], [6, 1, 2, 1])


def test_flirt_world_roundtrip_and_ras_las_equivalence():
    w = np.eye(4)
    w[:3, 3] = [1.5, -2.0, 3.0]
    in_aff = ph.ras_affine((20, 22, 18), 0.7)
    ref_aff = ph.MNI_AFFINE
    m = nifti.world_to_flirt(w, (20, 22, 18), in_aff, ph.MNI_SHAPE, ref_aff)
    back = nifti.flirt_to_world(m, (20, 22, 18), in_aff, ph.MNI_SHAPE, ref_aff)
    np.testing.assert_allclose(back, w, atol=1e-9)
    # Identity world mapping between an RAS image and its LAS-stored copy is
    # the identity in FSL coordinates (FSL flips x for neurological storage).
    shape = (10, 6, 6)
    ras = ph.ras_affine(shape)
    las = ras @ np.array([[-1.0, 0, 0, shape[0] - 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    m = nifti.world_to_flirt(np.eye(4), shape, ras, shape, las)
    np.testing.assert_allclose(m, np.eye(4), atol=1e-9)


def test_matrix_txt_roundtrip(tmp_path):
    m = np.arange(16, dtype=float).reshape(4, 4)
    p = str(tmp_path / "m.mat")
    nifti.write_matrix_txt(m, p)
    np.testing.assert_allclose(nifti.read_matrix_txt(p), m)
