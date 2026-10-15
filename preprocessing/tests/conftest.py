"""Shared fixtures: synthetic MNI/target files, fake tools on PATH, config file."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from . import phantoms as ph
from .fake_tools import install_fake_tools, read_calls


@pytest.fixture
def mni_files(tmp_path):
    """Small MNI-like 1 mm / 2 mm templates and mask (LAS)."""
    d = tmp_path / "mni"
    xyz = ph.world_coords(ph.MNI_SHAPE, ph.MNI_AFFINE)
    p1 = ph.save(str(d / "MNI_1mm_brain.nii.gz"), ph.brain(xyz) * ph.brain_mask(xyz), ph.MNI_AFFINE)
    pm = ph.save(str(d / "MNI_1mm_brain_mask.nii.gz"), ph.brain_mask(xyz), ph.MNI_AFFINE, np.uint8)
    a2 = ph.MNI_AFFINE.copy()
    a2[:3, :3] *= 2
    s2 = (15, 18, 15)
    xyz2 = ph.world_coords(s2, a2)
    p2 = ph.save(str(d / "MNI_2mm_brain.nii.gz"), ph.brain(xyz2) * ph.brain_mask(xyz2), a2)
    return SimpleNamespace(image_1mm=p1, mask_1mm=pm, image_2mm=p2)


@pytest.fixture
def target_dir(tmp_path, mni_files):
    """target_grid.json + template_lumirgrid files for the synthetic target grid."""
    d = tmp_path / "target"
    aff = ph.target_affine()
    xyz = ph.world_coords(ph.TARGET_SHAPE, aff)
    m = ph.brain_mask(xyz)
    ph.save(str(d / "template_lumirgrid.nii.gz"), ph.brain(xyz) * m, aff, code=4)
    ph.save(str(d / "template_lumirgrid_mask.nii.gz"), m, aff, np.uint8, code=4)
    grid = {"shape": list(ph.TARGET_SHAPE), "affine": aff.tolist(), "lumir_header_affine": np.eye(4).tolist(),
            "outputs": {"grid": str(d / "target_grid.json")}}
    with open(d / "target_grid.json", "w") as f:
        json.dump(grid, f)
    return str(d)


@pytest.fixture
def fake_tools(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    install_fake_tools(str(bin_dir))
    fs = tmp_path / "fs"
    fs.mkdir()
    (fs / "build-stamp.txt").write_text("freesurfer-linux-ubuntu22_x86_64-7.4.1-20230614-7eb8460\n")
    fsl = tmp_path / "fsl"
    (fsl / "etc").mkdir(parents=True)
    (fsl / "etc" / "fslversion").write_text("6.0.7.9\n")
    log = tmp_path / "fake_calls.jsonl"
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("FAKE_TOOL_LOG", str(log))
    monkeypatch.setenv("FREESURFER_HOME", str(fs))
    monkeypatch.setenv("FSLDIR", str(fsl))
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    monkeypatch.delenv("FAKE_SYNTHSEG_CROP", raising=False)
    monkeypatch.delenv("FAKE_FLIRT_SHIFT", raising=False)
    def calls():
        # Version probes (antsRegistration --version for meta.json) are not pipeline work.
        return [c for c in read_calls(str(log)) if c["argv"] != ["--version"]]

    return SimpleNamespace(bin=str(bin_dir), log=str(log), calls=calls)


def write_config(path, target_dir, mni_files, **overrides):
    cfg = {
        "env_setup": [],  # use the fake tool env, not the Martinos defaults
        "threads": 2,
        "target_dir": target_dir,
        "lumir_expected_shape": list(ph.TARGET_SHAPE),
        "mni_templates": {"T1": {"image_1mm": mni_files.image_1mm, "mask_1mm": mni_files.mask_1mm,
                                 "image_2mm": mni_files.image_2mm}},
        "synthseg": {"threads": 2},
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return str(path)


@pytest.fixture
def config_file(tmp_path, target_dir, mni_files):
    return write_config(tmp_path / "config.yaml", target_dir, mni_files)
