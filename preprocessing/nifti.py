"""NIfTI helpers: loading, saving with consistent qform/sform, grid comparisons,
world-space resampling and FSL scaled-voxel matrices.

All geometry relies on header affines (nibabel ``img.affine``, i.e. the sform
when its code is > 0, otherwise the qform). No implicit reorientation happens.

Example:
    from preprocessing import nifti
    img = nifti.load("sub.nii.gz")
    print(nifti.voxel_sizes(img.affine), nifti.qform_sform_consistent(img))
"""
import logging
import os
from typing import Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage

log = logging.getLogger(__name__)


def load(path) -> nib.Nifti1Image:
    """Load a NIfTI image (header only until data is requested)."""
    return nib.load(str(path))


def spatial_shape(img) -> Tuple[int, int, int]:
    """Return the 3D shape, allowing a trailing singleton 4th dimension."""
    shape = tuple(int(s) for s in img.shape)
    if len(shape) == 4 and shape[3] == 1:
        shape = shape[:3]
    if len(shape) != 3:
        raise ValueError("expected a 3D image, got shape %s" % (shape,))
    return shape  # type: ignore[return-value]


def load_data(path, dtype=np.float32) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """Load voxel data as ``dtype`` (squeezing a trailing singleton dim)."""
    img = load(path)
    shape = spatial_shape(img)
    data = np.asanyarray(img.dataobj).reshape(shape)
    return data.astype(dtype, copy=False), img


def voxel_sizes(affine) -> np.ndarray:
    """Voxel spacing (mm) from the column norms of the affine."""
    a = np.asarray(affine, dtype=float)
    return np.sqrt((a[:3, :3] ** 2).sum(axis=0))


def affines_match(a, b, tol: float = 1e-3) -> bool:
    """True when the max absolute element difference is <= tol."""
    return bool(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))) <= tol)


def grid_matches(shape_a: Sequence[int], affine_a, shape_b: Sequence[int], affine_b,
                 tol: float = 1e-3) -> bool:
    """True when both grids have the same shape and (within tol) the same affine."""
    return tuple(shape_a) == tuple(shape_b) and affines_match(affine_a, affine_b, tol)


def img_grid_matches(img, ref, tol: float = 1e-3) -> bool:
    """grid_matches for two nibabel images."""
    return grid_matches(spatial_shape(img), img.affine, spatial_shape(ref), ref.affine, tol)


def qform_sform_consistent(img, tol: float = 1e-3) -> bool:
    """False when both qform and sform are set but disagree (tools may then differ)."""
    hdr = img.header
    if int(hdr["qform_code"]) > 0 and int(hdr["sform_code"]) > 0:
        return affines_match(hdr.get_qform(), hdr.get_sform(), tol)
    return True


def save_like(data: np.ndarray, affine, path, dtype=np.float32, xform_code: int = 1) -> None:
    """Write ``data`` as ``dtype`` with qform == sform == ``affine``."""
    arr = np.asarray(data).astype(dtype, copy=False)
    img = nib.Nifti1Image(arr, np.asarray(affine, dtype=float))
    img.set_qform(affine, code=int(xform_code))
    img.set_sform(affine, code=int(xform_code))
    img.header.set_data_dtype(dtype)
    os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
    nib.save(img, str(path))


def resample_world(src: np.ndarray, src_affine, ref_shape: Sequence[int], ref_affine,
                   order: int = 1, cval: float = 0.0) -> np.ndarray:
    """Resample ``src`` onto a reference grid in world space.

    order=0 uses exact nearest-voxel lookup (np.rint of the mapped coordinates),
    which is exact for grids related by axis permutations/flips/integer shifts.
    order>=1 uses scipy.ndimage.map_coordinates. Processed one x-slab at a time
    to keep memory bounded on 160x224x192 grids.
    """
    vox2vox = np.linalg.inv(np.asarray(src_affine, float)) @ np.asarray(ref_affine, float)
    ref_shape = tuple(int(s) for s in ref_shape)
    out = np.zeros(ref_shape, dtype=np.float64 if order > 0 else src.dtype)
    jj, kk = np.meshgrid(np.arange(ref_shape[1]), np.arange(ref_shape[2]), indexing="ij")
    jj = jj.ravel()
    kk = kk.ravel()
    for i in range(ref_shape[0]):
        pts = np.stack([np.full_like(jj, i), jj, kk, np.ones_like(jj)]).astype(float)
        c = (vox2vox @ pts)[:3]
        if order == 0:
            idx = np.rint(c).astype(np.int64)
            ok = np.all((idx >= 0) & (idx < np.asarray(src.shape)[:, None]), axis=0)
            vals = np.full(jj.shape, cval, dtype=src.dtype)
            vals[ok] = src[idx[0, ok], idx[1, ok], idx[2, ok]]
        else:
            vals = ndimage.map_coordinates(src, c, order=order, mode="constant", cval=cval)
        out[i] = vals.reshape(ref_shape[1], ref_shape[2])
    return out


def fsl_scaled_voxel_matrix(shape: Sequence[int], affine, zooms: Optional[Sequence[float]] = None
                            ) -> np.ndarray:
    """Voxel -> FSL scaled-voxel ("FSL coordinates") matrix used by FLIRT .mat files.

    FSL coordinates are voxel indices times pixdim; when the voxel->world
    affine has a positive determinant (neurological storage) the x axis is
    flipped: x_fsl = (nx - 1 - i) * dx.
    """
    if zooms is None:
        zooms = voxel_sizes(affine)
    s = np.diag([float(zooms[0]), float(zooms[1]), float(zooms[2]), 1.0])
    if np.linalg.det(np.asarray(affine, float)[:3, :3]) > 0:
        flip = np.eye(4)
        flip[0, 0] = -1.0
        flip[0, 3] = int(shape[0]) - 1
        s = s @ flip
    return s


def img_fsl_matrix(img) -> np.ndarray:
    """fsl_scaled_voxel_matrix for a nibabel image (uses header pixdims)."""
    return fsl_scaled_voxel_matrix(spatial_shape(img), img.affine, img.header.get_zooms()[:3])


def flirt_to_world(flirt_mat, in_shape, in_affine, ref_shape, ref_affine,
                   in_zooms=None, ref_zooms=None) -> np.ndarray:
    """Convert a FLIRT matrix (input FSL coords -> reference FSL coords) to world->world."""
    s_in = fsl_scaled_voxel_matrix(in_shape, in_affine, in_zooms)
    s_ref = fsl_scaled_voxel_matrix(ref_shape, ref_affine, ref_zooms)
    return (np.asarray(ref_affine, float) @ np.linalg.inv(s_ref) @ np.asarray(flirt_mat, float)
            @ s_in @ np.linalg.inv(np.asarray(in_affine, float)))


def world_to_flirt(world, in_shape, in_affine, ref_shape, ref_affine,
                   in_zooms=None, ref_zooms=None) -> np.ndarray:
    """Inverse of flirt_to_world: a world->world matrix as a FLIRT matrix for (in, ref)."""
    s_in = fsl_scaled_voxel_matrix(in_shape, in_affine, in_zooms)
    s_ref = fsl_scaled_voxel_matrix(ref_shape, ref_affine, ref_zooms)
    return (s_ref @ np.linalg.inv(np.asarray(ref_affine, float)) @ np.asarray(world, float)
            @ np.asarray(in_affine, float) @ np.linalg.inv(s_in))


def read_matrix_txt(path) -> np.ndarray:
    """Read a whitespace-separated 4x4 matrix (FLIRT .mat format)."""
    m = np.loadtxt(str(path), dtype=float)
    if m.shape != (4, 4):
        raise ValueError("%s: expected a 4x4 matrix, got %s" % (path, m.shape))
    return m


def write_matrix_txt(m, path) -> None:
    """Write a 4x4 matrix in FLIRT text format."""
    np.savetxt(str(path), np.asarray(m, float), fmt="%.10f")
