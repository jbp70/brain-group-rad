import csv
import json
import os

import numpy as np
import pytest

from preprocessing import nifti
from preprocessing.cli import main
from preprocessing.export import export_npy
from preprocessing.luts import LUTS, SYNTHSEG_STRUCTURES, _build, _identity, apply_lut

from . import phantoms as ph

SYNTHSEG_IDS = np.array([i for i, _ in SYNTHSEG_STRUCTURES] + list(range(1001, 1036)) + list(range(2001, 2036)))


def _subject(root, name, status="completed", synthseg_ids=SYNTHSEG_IDS, save_mask=True, extra_label=None):
    d = os.path.join(root, "ds", name)
    rng = np.random.default_rng(0)
    img = rng.random(ph.TARGET_SHAPE).astype(np.float32)
    img[img < 0.3] = 0.0
    mask = (img > 0).astype(np.uint8)
    lab = rng.choice(synthseg_ids, ph.TARGET_SHAPE).astype(np.uint16)
    aff = ph.target_affine()
    ph.save(os.path.join(d, "image.nii.gz"), img, aff)
    if save_mask:
        ph.save(os.path.join(d, "mask.nii.gz"), mask, aff, np.uint8)
    ph.save(os.path.join(d, "labels_synthseg.nii.gz"), lab, aff, np.uint16)
    if extra_label is not None:
        ph.save(os.path.join(d, "labels_%s.nii.gz" % extra_label[0]), extra_label[1], aff, np.uint16)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"status": status, "row": {"subject": name, "split": "train"}}, f)
    return d, img, mask, lab


def test_defaults_float16_and_uint8_lut(tmp_path, config_file):
    out = str(tmp_path / "out")
    lumir = np.random.default_rng(1).integers(0, 11, ph.TARGET_SHAPE)
    d, img, mask, lab = _subject(out, "s1", save_mask=False, extra_label=("lumir10", lumir))
    _subject(out, "s_failed", status="failed")
    exp = str(tmp_path / "npy")
    assert main(["export-npy", "--out-root", out, "--export-root", exp, "--include-template",
                 "--config", config_file]) == 0
    e = os.path.join(exp, "ds", "s1")
    a = np.load(os.path.join(e, "image.npy"))
    assert a.dtype == np.float16 and a.shape == ph.TARGET_SHAPE and a.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(a.astype(np.float32), img, atol=5e-4)
    assert not os.path.exists(os.path.join(e, "mask.npy"))  # not exported by default
    ll = np.load(os.path.join(e, "labels_synthseg.npy"))
    assert ll.dtype == np.uint8 and ll.max() == len(SYNTHSEG_IDS) - 1
    luts = json.load(open(os.path.join(exp, "label_luts.json")))
    m = {int(k): v for k, v in luts["luts"]["synthseg"]["map"].items()}
    assert m[0] == 0 and m[2] == 1 and m[2035] == 102 and luts["luts"]["synthseg"]["dtype"] == "uint8"
    assert luts["luts"]["synthseg"]["names"]["17"] == "left hippocampus"
    assert luts["luts"]["synthseg"]["names"]["1035"] == "ctx-lh-insula"
    np.testing.assert_array_equal(np.array(sorted(m)), np.sort(SYNTHSEG_IDS))
    np.testing.assert_array_equal(np.array([k for k, _ in sorted(m.items(), key=lambda x: x[1])])[ll], lab)
    l10 = np.load(os.path.join(e, "labels_lumir10.npy"))
    assert l10.dtype == np.uint8
    np.testing.assert_array_equal(l10, lumir)
    assert not os.path.exists(os.path.join(exp, "ds", "s_failed"))
    assert os.path.isfile(os.path.join(exp, "template", "template.npy"))
    rows = list(csv.DictReader(open(os.path.join(exp, "export_manifest.csv"))))
    assert len(rows) == 1 and rows[0]["labels"] == "lumir10;synthseg" and rows[0]["split"] == "train"
    assert rows[0]["label_dtypes"] == "lumir10:uint8;synthseg:uint8" and rows[0]["image_dtype"] == "float16"
    assert rows[0]["label_luts"].endswith("label_luts.json") and rows[0]["mask"] == ""


def test_float32_raw_labels_and_mask_option(tmp_path):
    out = str(tmp_path / "out")
    d, img, mask, lab = _subject(out, "s1", save_mask=False)
    exp = str(tmp_path / "npy32")
    export_npy(out, exp, image_dtype="float32", remap_labels=False, include_mask=True)
    e = os.path.join(exp, "ds", "s1")
    np.testing.assert_array_equal(np.load(os.path.join(e, "image.npy")), img)
    ll = np.load(os.path.join(e, "labels_synthseg.npy"))
    assert ll.dtype == np.uint16
    np.testing.assert_array_equal(ll, lab)
    mk = np.load(os.path.join(e, "mask.npy"))
    assert mk.dtype == np.uint8
    np.testing.assert_array_equal(mk, (img > 0).astype(np.uint8))  # derived from image > 0
    rows = list(csv.DictReader(open(os.path.join(exp, "export_manifest.csv"))))
    assert rows[0]["mask"] == "image>0" and rows[0]["label_dtypes"] == "synthseg:uint16:raw"
    assert json.load(open(os.path.join(exp, "label_luts.json"))) == {"remap_labels": False, "luts": {}}


def test_unknown_label_fails_loudly(tmp_path):
    out = str(tmp_path / "out")
    _subject(out, "s1", synthseg_ids=np.array([0, 2, 999]))
    with pytest.raises(ValueError) as ei:
        export_npy(out, str(tmp_path / "npy"))
    assert "999" in str(ei.value) and "not in the fixed LUT" in str(ei.value)
    # Explicit uint8 without remapping still refuses values that do not fit.
    out2 = str(tmp_path / "out2")
    _subject(out2, "s1")
    with pytest.raises(ValueError):
        export_npy(out2, str(tmp_path / "npy2"), label_dtype="uint8", remap_labels=False)


def test_uint16_fallback_for_large_lut(tmp_path):
    out = str(tmp_path / "out")
    big = _build(_identity(300), "test")
    _subject(out, "s1", synthseg_ids=np.array([0, 5, 299, 300]))
    exp = str(tmp_path / "npy")
    export_npy(out, exp, luts={"synthseg": big})
    ll = np.load(os.path.join(exp, "ds", "s1", "labels_synthseg.npy"))
    assert ll.dtype == np.uint16 and ll.max() == 300
    assert json.load(open(os.path.join(exp, "label_luts.json")))["luts"]["synthseg"]["dtype"] == "uint16"


def test_reexport_into_existing_root(tmp_path, caplog):
    out = str(tmp_path / "out")
    _subject(out, "s1")
    exp = str(tmp_path / "npy")
    # An earlier export with float32 images and raw uint16 labels.
    export_npy(out, exp, image_dtype="float32", remap_labels=False)
    e = os.path.join(exp, "ds", "s1")
    with pytest.raises(ValueError) as ei:
        export_npy(out, exp)
    assert "image.npy" in str(ei.value) and "float32" in str(ei.value) and "--overwrite" in str(ei.value)
    # Image matches now, but the old raw uint16 labels still do not.
    np.save(os.path.join(e, "image.npy"), np.zeros(ph.TARGET_SHAPE, np.float16))
    with pytest.raises(ValueError) as ei:
        export_npy(out, exp)
    assert "labels_synthseg.npy" in str(ei.value) and "uint16" in str(ei.value)
    # --overwrite rewrites everything with the new defaults.
    export_npy(out, exp, overwrite=True)
    assert np.load(os.path.join(e, "image.npy")).dtype == np.float16
    assert np.load(os.path.join(e, "labels_synthseg.npy")).dtype == np.uint8
    # Matching dtype/shape: files are kept and the skip is reported.
    before = os.path.getmtime(os.path.join(e, "image.npy"))
    with caplog.at_level("WARNING"):
        assert export_npy(out, exp) == 1
    assert os.path.getmtime(os.path.join(e, "image.npy")) == before
    assert "kept 2 existing .npy files" in caplog.text


def test_fixed_luts_are_dataset_independent():
    assert len(LUTS["synthseg"]["map"]) == 103 and list(LUTS["synthseg"]["map"].values()) == list(range(103))
    a = apply_lut(np.array([0, 2, 60, 1001, 2035]), LUTS["synthseg"], "uint8")
    b = apply_lut(np.array([2035, 0]), LUTS["synthseg"], "uint8")
    assert a[-1] == b[0] == 102 and a[0] == b[1] == 0
    for name in ("lumir10", "seg4", "seg35"):
        lut = LUTS[name]["map"]
        assert all(k == v for k, v in lut.items())
