"""Synthetic dataset rows (tiny NIfTIs written at test time) for end-to-end tests."""
import os
from collections import OrderedDict
from typing import Dict, List

import numpy as np

from preprocessing.manifest import ManifestRow, write_manifest

from . import phantoms as ph


def lumir_row(root: str, subject: str = "LUMIRMRI_0001", shape=ph.TARGET_SHAPE, header=None) -> ManifestRow:
    """LUMIR-like: data already on the (corrected) target voxel grid but an identity header.

    A different ``shape`` with ``header=ph.target_affine()`` gives an off-grid
    but world-consistent row (grid mismatch path).
    """
    header = np.eye(4) if header is None else header
    xyz = ph.world_coords(ph.TARGET_SHAPE, ph.target_affine())
    data = (ph.brain(xyz) * ph.brain_mask(xyz) * 255.0)[: shape[0], : shape[1], : shape[2]]
    lab = ph.seg(xyz)[: shape[0], : shape[1], : shape[2]]
    img = ph.save(os.path.join(root, "lumir", "imagesVal", subject + "_0000.nii.gz"), data, header)
    lp = ph.save(os.path.join(root, "lumir", "labelsVal", subject + ".nii.gz"), lab, header, np.int16)
    return ManifestRow(dataset="lumir", subject=subject, contrast="T1w", split="val", image_path=img,
                       label_paths=OrderedDict([("lumir10", lp)]), skull_stripped=True,
                       in_template_space=True, bias_corrected=True)


def neurite_row(root: str, subject: str = "OASIS_OAS1_0001_MR1") -> ManifestRow:
    shape = (32, 32, 32)
    aff = ph.ras_affine(shape, 1.0)
    xyz = ph.world_coords(shape, aff)
    d = os.path.join(root, "neurite", subject)
    norm = ph.save(os.path.join(d, "norm.nii.gz"), ph.brain(xyz) * ph.brain_mask(xyz), aff)
    seg = ph.save(os.path.join(d, "seg35.nii.gz"), ph.seg(xyz), aff, np.int16)
    return ManifestRow(dataset="neurite_oasis", subject=subject, contrast="T1w", image_path=norm,
                       label_paths=OrderedDict([("seg35", seg)]), skull_stripped=True, bias_corrected=True)


def ultracortex_row(root: str, subject: str = "sub-01", session: str = "1") -> ManifestRow:
    """0.7 mm, LPS-like storage, raw head with skull and bias."""
    shape = (40, 48, 40)
    aff = ph.ras_affine(shape, 0.7)
    aff[0, :] *= -1
    aff[1, :] *= -1
    xyz = ph.world_coords(shape, aff)
    img = ph.save(os.path.join(root, "uc", subject, "ses-" + session, "anat",
                               "%s_ses-%s_T1w.nii.gz" % (subject, session)), ph.head(xyz), aff, np.int16)
    return ManifestRow(dataset="ultracortex", subject=subject, session=session, contrast="T1w", image_path=img)


def generic_row(root: str, subject: str = "OAS2_0001") -> ManifestRow:
    shape = (32, 40, 32)
    aff = ph.ras_affine(shape, 1.0, center=(0.5, -0.3, 0.2))
    xyz = ph.world_coords(shape, aff)
    img = ph.save(os.path.join(root, "generic", subject, "raw.nii.gz"), ph.head(xyz), aff)
    lab = ph.save(os.path.join(root, "generic", subject, "manual.nii.gz"), ph.seg(xyz), aff, np.int16)
    return ManifestRow(dataset="oasis2", subject=subject, contrast="T1w", image_path=img,
                       label_paths=OrderedDict([("manual", lab)]))


def mindboggle_row(root: str, subject: str = "OASIS-TRT-20-1") -> ManifestRow:
    """Mindboggle-like: raw head (skull, bias) already on the MNI grid (LAS header), float32 labels.

    Off the target grid -> world-space resampling with its own header; needs strip + N4.
    """
    xyz = ph.world_coords(ph.MNI_SHAPE, ph.MNI_AFFINE)
    d = os.path.join(root, "mindboggle", "OASIS-TRT-20_volumes", subject)
    img = ph.save(os.path.join(d, "t1weighted.MNI152.nii.gz"), ph.head(xyz), ph.MNI_AFFINE)
    lab = ph.save(os.path.join(d, "labels.DKT31.manual+aseg.MNI152.nii.gz"), ph.seg(xyz), ph.MNI_AFFINE)
    return ManifestRow(dataset="mindboggle", subject=subject, contrast="T1w", image_path=img,
                       label_paths=OrderedDict([("dkt31aseg", lab)]), in_template_space=True)


def write_rows(path: str, rows: List[ManifestRow]) -> str:
    write_manifest(rows, path)
    return path


def calls_by_tool(calls: List[Dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in calls:
        out[c["tool"]] = out.get(c["tool"], 0) + 1
    return out
