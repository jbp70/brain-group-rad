"""Synthetic brain phantoms defined in world (mm) coordinates.

The same anatomy can be rendered on any grid (affine), so tests can check that
outputs line up in world space. The phantom is left/right asymmetric so flip
searches have a unique answer; ``symmetric=True`` gives a mirror-symmetric one.
"""
import os
from typing import Optional, Sequence

import nibabel as nib
import numpy as np

# MNI-like LAS grid (like FSL's MNI152 but small).
MNI_SHAPE = (30, 36, 30)
MNI_AFFINE = np.array([[-1.0, 0, 0, 15], [0, 1.0, 0, -18], [0, 0, 1.0, -15], [0, 0, 0, 1]])
# LUMIR-like target: crop x 30->24, pad y 36->40, pad z 30->34, no flip.
TARGET_SHAPE = (24, 40, 34)
TARGET_OFFSET = (3, -2, -2)


def voxel_map(offset=TARGET_OFFSET, flip_x: bool = False, shape=TARGET_SHAPE) -> np.ndarray:
    t = np.eye(4)
    if flip_x:
        t[0, 0] = -1
        t[0, 3] = shape[0] - 1
    t[:3, 3] += np.asarray(offset, float)
    return t


def target_affine(offset=TARGET_OFFSET, flip_x: bool = False) -> np.ndarray:
    return MNI_AFFINE @ voxel_map(offset, flip_x)


def world_coords(shape: Sequence[int], affine) -> np.ndarray:
    ijk = np.indices(shape).reshape(3, -1)
    xyz = affine[:3, :3] @ ijk + affine[:3, 3:4]
    return xyz.reshape((3,) + tuple(shape))


def _soft(r, width=0.12):
    return 1.0 / (1.0 + np.exp((r - 1.0) / width))


def _radius(xyz: np.ndarray, symmetric: bool = False) -> np.ndarray:
    """Normalized ellipsoid radius; egg-shaped (x radius 9 on +x, 7 on -x) unless symmetric."""
    x, y, z = xyz
    rx = 9.0 if symmetric else np.where(x > 0, 9.0, 7.0)
    return np.sqrt((x / rx) ** 2 + ((y + 1) / 12.0) ** 2 + (z / 9.0) ** 2)


def brain(xyz: np.ndarray, symmetric: bool = False) -> np.ndarray:
    x, y, z = xyz
    r = _radius(xyz, symmetric)
    v = _soft(r)
    blobs = [((4.0, 4.0, 2.0), 3.0, -0.5), ((-3.0, -6.0, -3.0), 2.5, -0.3), ((5.0, -5.0, 0.0), 2.0, -0.25)]
    if symmetric:
        blobs = [((0.0, 4.0, 2.0), 3.0, -0.5), ((0.0, -6.0, -3.0), 2.5, -0.3)]
    for (bx, by, bz), rad, amp in blobs:
        rb = np.sqrt((x - bx) ** 2 + (y - by) ** 2 + (z - bz) ** 2) / rad
        v = v + amp * _soft(rb, 0.2) * (r < 1.0)
    return np.clip(v, 0, None)


def brain_mask(xyz: np.ndarray, symmetric: bool = False) -> np.ndarray:
    return _radius(xyz, symmetric) < 1.0


def head(xyz: np.ndarray, bias: bool = True) -> np.ndarray:
    """Brain + dim skull shell + smooth bias field (a raw-like image)."""
    x = xyz[0]
    r = _radius(xyz)
    v = brain(xyz) + 0.4 * ((r > 1.15) & (r < 1.3))
    if bias:
        v = v * (1.0 + 0.05 * x / 10.0)
    return v * 100.0


def seg(xyz: np.ndarray) -> np.ndarray:
    """Label map: 2 (left-ish WM) / 41 (right-ish WM) / 4 (blob) inside the brain."""
    x = xyz[0]
    m = brain_mask(xyz)
    lab = np.where(x < 0, 2, 41) * m
    b = brain(xyz)
    lab[(b < 0.75) & m] = 4
    return lab.astype(np.int16)


def save(path: str, data: np.ndarray, affine, dtype=np.float32, code: int = 1) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = nib.Nifti1Image(np.asarray(data).astype(dtype), np.asarray(affine, float))
    img.set_qform(affine, code=code)
    img.set_sform(affine, code=code)
    nib.save(img, path)
    return path


def ras_affine(shape: Sequence[int], spacing: float = 1.0, center: Optional[Sequence[float]] = None) -> np.ndarray:
    """RAS affine whose grid centre sits at world ``center`` (default origin)."""
    c = np.zeros(3) if center is None else np.asarray(center, float)
    a = np.diag([spacing, spacing, spacing, 1.0])
    a[:3, 3] = c - spacing * (np.asarray(shape) - 1) / 2.0
    return a
