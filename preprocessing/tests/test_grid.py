"""build-target-grid: recover flip + crop/pad offsets from identity-header volumes."""
import json
import os

import numpy as np
import pytest

from preprocessing import nifti
from preprocessing.cli import main
from preprocessing.config import load_config
from preprocessing.grid import (GridAmbiguityError, Target, apply_candidate, build_target_grid, candidate_matrix,
                                load_target_grid)

from . import phantoms as ph


def _lumir_from_mni(mni, t, shape=ph.TARGET_SHAPE):
    """lumir[i] = mni[T i] (zero outside), i.e. a crop/pad (+flip) of the MNI array."""
    idx = np.indices(shape).reshape(3, -1)
    m = (t[:3, :3] @ idx + t[:3, 3:4]).round().astype(int)
    ok = np.all((m >= 0) & (m < np.array(mni.shape)[:, None]), axis=0)
    out = np.zeros(idx.shape[1])
    out[ok] = mni[m[0, ok], m[1, ok], m[2, ok]]
    return out.reshape(shape)


def _make_lumir(tmp_path, mni, t, n=3):
    rng = np.random.default_rng(0)
    paths = []
    base = _lumir_from_mni(mni, t)
    for i in range(n):
        d = base * 255.0 * (0.8 + 0.1 * i) + rng.normal(0, 1.0, base.shape) * (base > 0)
        paths.append(ph.save(str(tmp_path / "LUMIR" / "imagesTr" / ("LUMIRMRI_%04d_0000.nii.gz" % i)),
                             np.clip(d, 0, None), np.eye(4)))
    return paths


def test_candidate_matrix_matches_apply_candidate():
    a = np.random.default_rng(0).random((4, 5, 6))
    for perm, flips in (((0, 1, 2), (True, False, False)), ((2, 0, 1), (False, True, True))):
        arr = apply_candidate(a, perm, flips)
        p = candidate_matrix(a.shape, perm, flips)
        for i in np.ndindex(a.shape):
            j = (p @ np.array(list(i) + [1]))[:3].astype(int)
            assert arr[tuple(j)] == a[i]


@pytest.mark.parametrize("flip", [False, True])
def test_recovers_flip_and_offsets(tmp_path, mni_files, flip):
    mni, mimg = nifti.load_data(mni_files.image_1mm)
    t_true = ph.voxel_map(ph.TARGET_OFFSET, flip_x=flip)
    paths = _make_lumir(tmp_path, mni, t_true)
    cfg = load_config(None)
    cfg["lumir_expected_shape"] = list(ph.TARGET_SHAPE)
    out = str(tmp_path / "tg")
    info = build_target_grid(paths, mni_files.image_1mm, out, cfg, mni_files.mask_1mm, n=3, seed=0)
    np.testing.assert_array_equal(np.asarray(info["voxel_map"]), t_true)
    assert info["flip_x"] is flip and info["ncc_margin"] > 0.02 and info["ncc"] > 0.99
    np.testing.assert_allclose(info["affine"], mimg.affine @ t_true)
    sv = info["candidates"][0]["subvoxel_shift"]
    assert np.all(np.abs(sv) < 0.25)
    g = load_target_grid(os.path.join(out, "target_grid.json"))
    np.testing.assert_array_equal(g["lumir_header_affine"], np.eye(4))
    tmpl, timg = nifti.load_data(os.path.join(out, "template_lumirgrid.nii.gz"))
    np.testing.assert_allclose(timg.affine, mimg.affine @ t_true, atol=1e-5)
    # The template on the corrected grid is the same crop of MNI (exact integer mapping).
    np.testing.assert_allclose(tmpl, _lumir_from_mni(mni, t_true) * (tmpl > 0), atol=1e-4)
    # Target loader accepts both the raw LUMIR header and the corrected affine.
    cfg["target_dir"] = out
    tg = Target(cfg, {})
    assert tg.state(ph.TARGET_SHAPE, np.eye(4)) == "header"
    assert tg.state(ph.TARGET_SHAPE, timg.affine) == "target"
    assert tg.state((1, 2, 3), np.eye(4)) == "mismatch"


def test_symmetric_volume_refused_then_forced(tmp_path):
    xyz = ph.world_coords(ph.MNI_SHAPE, ph.MNI_AFFINE)
    sym = ph.brain(xyz, symmetric=True) * ph.brain_mask(xyz, symmetric=True)
    # Symmetric about the array centre so a flip is indistinguishable.
    sym = 0.5 * (sym + sym[::-1])
    mni_path = ph.save(str(tmp_path / "mni_sym.nii.gz"), sym, ph.MNI_AFFINE)
    paths = _make_lumir(tmp_path, sym, ph.voxel_map(ph.TARGET_OFFSET))
    cfg = load_config(None)
    out = str(tmp_path / "tg")
    with pytest.raises(GridAmbiguityError) as ei:
        build_target_grid(paths, mni_path, out, cfg, None, n=3, min_margin=0.01)
    assert "force-flip" in str(ei.value)
    assert os.path.isfile(os.path.join(out, "target_grid_candidates.json"))
    assert not os.path.isfile(os.path.join(out, "target_grid.json"))
    # CLI: non-zero exit with the same refusal.
    rc = main(["build-target-grid", "--lumir-images"] + paths + ["--mni", mni_path, "--out-dir", out, "--n", "3"])
    assert rc == 2
    rc = main(["build-target-grid", "--lumir-images"] + paths + ["--mni", mni_path, "--out-dir", out, "--n", "3",
                                                                 "--force-flip", "no"])
    assert rc == 0
    with open(os.path.join(out, "target_grid.json")) as f:
        g = json.load(f)
    assert g["forced"] == "no" and g["flip_x"] is False
