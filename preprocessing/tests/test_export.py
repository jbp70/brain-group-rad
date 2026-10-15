import csv
import json
import os

import numpy as np
import pytest

from preprocessing import nifti
from preprocessing.cli import main
from preprocessing.export import export_npy

from . import phantoms as ph


def _subject(root, name, labels_max=40, status="completed"):
    d = os.path.join(root, "ds", name)
    rng = np.random.default_rng(0)
    img = rng.random(ph.TARGET_SHAPE).astype(np.float32)
    mask = (img > 0.3).astype(np.uint8)
    lab = rng.integers(0, labels_max + 1, ph.TARGET_SHAPE).astype(np.uint16)
    aff = ph.target_affine()
    ph.save(os.path.join(d, "image.nii.gz"), img, aff)
    ph.save(os.path.join(d, "mask.nii.gz"), mask, aff, np.uint8)
    ph.save(os.path.join(d, "labels_synthseg.nii.gz"), lab, aff, np.uint16)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"status": status, "row": {"subject": name, "split": "train"}}, f)
    return d, img, mask, lab


def test_roundtrip_float32_and_float16(tmp_path, config_file):
    out = str(tmp_path / "out")
    d, img, mask, lab = _subject(out, "s1")
    _subject(out, "s_failed", status="failed")
    exp = str(tmp_path / "npy")
    assert main(["export-npy", "--out-root", out, "--export-root", exp, "--include-template",
                 "--config", config_file]) == 0
    e = os.path.join(exp, "ds", "s1")
    a = np.load(os.path.join(e, "image.npy"))
    assert a.dtype == np.float32 and a.shape == ph.TARGET_SHAPE and a.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(a, nifti.load_data(os.path.join(d, "image.nii.gz"))[0])
    np.testing.assert_array_equal(np.load(os.path.join(e, "mask.npy")), mask)
    assert np.load(os.path.join(e, "mask.npy")).dtype == np.uint8
    ll = np.load(os.path.join(e, "labels_synthseg.npy"))
    assert ll.dtype == np.uint16
    np.testing.assert_array_equal(ll, lab)
    assert not os.path.exists(os.path.join(exp, "ds", "s_failed"))
    assert os.path.isfile(os.path.join(exp, "template", "template.npy"))
    rows = list(csv.DictReader(open(os.path.join(exp, "export_manifest.csv"))))
    assert len(rows) == 1 and rows[0]["labels"] == "synthseg" and rows[0]["split"] == "train"
    exp16 = str(tmp_path / "npy16")
    export_npy(out, exp16, image_dtype="float16", label_dtype="uint8")
    a16 = np.load(os.path.join(exp16, "ds", "s1", "image.npy"))
    assert a16.dtype == np.float16
    np.testing.assert_allclose(a16.astype(np.float32), img, atol=1e-3)
    assert np.load(os.path.join(exp16, "ds", "s1", "labels_synthseg.npy")).dtype == np.uint8


def test_uint8_overflow_raises(tmp_path):
    out = str(tmp_path / "out")
    _subject(out, "s1", labels_max=2035)
    with pytest.raises(ValueError):
        export_npy(out, str(tmp_path / "npy"), label_dtype="uint8")
