import csv
import glob
import json
import logging
import os

import numpy as np
import pytest

from preprocessing import export as ex
from preprocessing.cli import main
from preprocessing.export import (MANIFEST_COLUMNS, block_average, block_mode, export_check, export_npy,
                                  level_affine, merge_parts, world_center)
from preprocessing.luts import LUTS, SYNTHSEG_STRUCTURES, TRAIN18_CLASSES, _build, _identity, apply_lut
from preprocessing.qc import summarize

from . import phantoms as ph

SYNTHSEG_IDS = np.array([i for i, _ in SYNTHSEG_STRUCTURES] + list(range(1001, 1036)) + list(range(2001, 2036)))
SHAPE = (8, 12, 16)  # divisible by 4 (LUMIR: 160x224x192)
# Non-trivial corrected affine: x flip, y/z axes swapped, fractional offsets.
AFFINE = np.array([[-1.0, 0, 0, 40.5], [0, 0.0, 1.0, -60.25], [0, 1.0, 0.0, -7.0], [0, 0, 0, 1]])


def _manifest(exp):
    return list(csv.DictReader(open(os.path.join(exp, "export_manifest.csv"))))


def _subject(root, name, status="completed", synthseg_ids=SYNTHSEG_IDS, save_mask=False, extra_label=None,
             contrast="T1w", split="train", ds="ds", seed=0, shape=SHAPE):
    d = os.path.join(root, ds, name)
    rng = np.random.default_rng(seed)
    img = rng.random(shape).astype(np.float32)
    img[img < 0.3] = 0.0
    lab = rng.choice(synthseg_ids, shape).astype(np.uint16)
    ph.save(os.path.join(d, "image.nii.gz"), img, AFFINE)
    if save_mask:
        ph.save(os.path.join(d, "mask.nii.gz"), (img > 0).astype(np.uint8), AFFINE, np.uint8)
    ph.save(os.path.join(d, "labels_synthseg.nii.gz"), lab, AFFINE, np.uint16)
    if extra_label is not None:
        ph.save(os.path.join(d, "labels_%s.nii.gz" % extra_label[0]), extra_label[1], AFFINE, np.uint16)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"status": status, "row": {"subject": name, "split": split, "contrast": contrast}}, f)
    return d, img, lab


def _train18(lab):
    table = {o: k for k, _, ids in TRAIN18_CLASSES for o in ids}
    return np.vectorize(table.__getitem__)(lab).astype(np.uint8)


def _mean_ref(a, k):
    out = np.zeros(tuple(s // k for s in a.shape))
    for i, j, l in np.ndindex(*out.shape):
        out[i, j, l] = a[i * k:(i + 1) * k, j * k:(j + 1) * k, l * k:(l + 1) * k].astype(np.float64).mean()
    return out


def _mode_ref(a, k):
    out = np.zeros(tuple(s // k for s in a.shape), a.dtype)
    for i, j, l in np.ndindex(*out.shape):
        v, c = np.unique(a[i * k:(i + 1) * k, j * k:(j + 1) * k, l * k:(l + 1) * k], return_counts=True)
        out[i, j, l] = v[c == c.max()].min()
    return out


# ----------------------------------------------------------------- defaults / layout

def test_defaults_float32_levels_train18(tmp_path):
    out = str(tmp_path / "out")
    d, img, lab = _subject(out, "s1")
    _subject(out, "s_failed", status="failed")
    exp = str(tmp_path / "npy")
    assert main(["export-npy", "--out-root", out, "--export-root", exp]) == 0
    e = os.path.join(exp, "ds", "s1")
    assert sorted(os.listdir(e)) == ["image.npy", "image_2.npy", "image_4.npy", "labels_train18.npy",
                                     "labels_train18_2.npy", "labels_train18_4.npy"]
    a = np.load(os.path.join(e, "image.npy"))
    assert a.dtype == np.float32 and a.shape == SHAPE and a.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(a, img)
    for k in (2, 4):
        ak = np.load(os.path.join(e, "image_%d.npy" % k))
        assert ak.dtype == np.float32 and ak.shape == tuple(s // k for s in SHAPE)
        np.testing.assert_allclose(ak, _mean_ref(img, k), rtol=0, atol=1e-6)  # float32 rounding only
    t18 = _train18(lab)
    l1 = np.load(os.path.join(e, "labels_train18.npy"))
    assert l1.dtype == np.uint8
    np.testing.assert_array_equal(l1, t18)
    for k in (2, 4):
        np.testing.assert_array_equal(np.load(os.path.join(e, "labels_train18_%d.npy" % k)), _mode_ref(t18, k))
    assert not os.path.exists(os.path.join(exp, "ds", "s_failed"))
    rows = _manifest(exp)
    assert len(rows) == 1 and list(rows[0]) == MANIFEST_COLUMNS
    r = rows[0]
    assert r["image_dtype"] == "float32" and r["labels"] == "train18" and r["label_dtypes"] == "train18:uint8"
    assert r["levels"] == "1;2;4" and r["label_levels"] == "train18:1,2,4" and r["shape"] == "8x12x16"
    assert r["image_l2"] == "image_2.npy" and r["shape_l4"] == "2x3x4" and r["labels_l4"] == "labels_train18_4.npy"
    assert r["spacing_l1"] == "1x1x1" and r["spacing_l2"] == "2x2x2" and r["spacing_l4"] == "4x4x4"
    luts = json.load(open(os.path.join(exp, "label_luts.json")))
    t = luts["luts"]["train18"]
    assert list(luts["luts"]) == ["train18"] and t["n_labels"] == 19 and t["dtype"] == "uint8"
    assert t["classes"]["0"] == "background" and t["classes"]["18"] == "ventral DC"
    assert t["names"]["1035"] == "cerebral cortex" and t["map"]["2035"] == 2
    assert os.path.isfile(os.path.join(exp, "manifest_parts", "all_split-any_contrast-T1w_rows-all.csv"))


def test_float16_option_raw_labels_mask_and_synthseg_set(tmp_path):
    out = str(tmp_path / "out")
    d, img, lab = _subject(out, "s1")
    exp = str(tmp_path / "npy16")
    export_npy(out, exp, image_dtype="float16", include_mask=True, label_sets=["train18,synthseg"])
    e = os.path.join(exp, "ds", "s1")
    a2 = np.load(os.path.join(e, "image_2.npy"))
    assert a2.dtype == np.float16
    # Coarse levels come from the float32 image, cast once at the end.
    np.testing.assert_allclose(a2.astype(np.float64), _mean_ref(img, 2), rtol=0, atol=5e-4)
    ll = np.load(os.path.join(e, "labels_synthseg.npy"))
    assert ll.dtype == np.uint8 and ll.max() == len(SYNTHSEG_IDS) - 1
    assert not os.path.exists(os.path.join(e, "labels_synthseg_2.npy"))  # full resolution only
    m = {int(k): v for k, v in json.load(open(os.path.join(exp, "label_luts.json")))["luts"]["synthseg"]["map"].items()}
    np.testing.assert_array_equal(np.array([k for k, _ in sorted(m.items(), key=lambda x: x[1])])[ll], lab)
    np.testing.assert_array_equal(np.load(os.path.join(e, "mask.npy")), (img > 0).astype(np.uint8))
    r = _manifest(exp)[0]
    assert r["mask"] == "image>0" and r["label_levels"] == "train18:1,2,4;synthseg:1"
    assert r["labels_l1"] == "labels_train18.npy;labels_synthseg.npy" and r["labels_l2"] == "labels_train18_2.npy"
    # Raw ids (no remap) for synthseg; train18 cannot be un-remapped.
    exp2 = str(tmp_path / "raw")
    export_npy(out, exp2, remap_labels=False, label_sets=["synthseg"])
    raw = np.load(os.path.join(exp2, "ds", "s1", "labels_synthseg.npy"))
    assert raw.dtype == np.uint16
    np.testing.assert_array_equal(raw, lab)
    assert _manifest(exp2)[0]["label_dtypes"] == "synthseg:uint16:raw"
    assert json.load(open(os.path.join(exp2, "label_luts.json"))) == {"remap_labels": False, "luts": {}}
    with pytest.raises(ValueError, match="train18"):
        export_npy(out, str(tmp_path / "x"), remap_labels=False)


def test_val_labels_full_resolution_only_and_missing_set_fails(tmp_path):
    out = str(tmp_path / "out")
    lumir = np.random.default_rng(1).integers(0, 11, SHAPE)
    _subject(out, "v1", split="val", extra_label=("lumir10", lumir))
    _subject(out, "t1", split="train")
    exp = str(tmp_path / "npy")
    assert main(["export-npy", "--out-root", out, "--export-root", exp, "--split", "val",
                 "--label-sets", "train18,lumir10"]) == 0
    e = os.path.join(exp, "ds", "v1")
    l10 = np.load(os.path.join(e, "labels_lumir10.npy"))
    assert l10.dtype == np.uint8
    np.testing.assert_array_equal(l10, lumir)
    assert not glob.glob(os.path.join(e, "labels_lumir10_*.npy"))
    r = _manifest(exp)
    assert [x["subject_dir"] for x in r] == ["v1"] and r[0]["labels"] == "train18;lumir10"
    # The train case has no lumir10 map: requesting it fails loudly.
    with pytest.raises(ValueError, match="labels_lumir10.nii.gz"):
        export_npy(out, str(tmp_path / "npy2"), label_sets=["train18,lumir10"])
    with pytest.raises(ValueError, match="full resolution only"):
        export_npy(out, str(tmp_path / "npy3"), levels=[2, 4], label_sets=["train18,lumir10"], splits=["val"])


# ----------------------------------------------------------------- pooling

def test_block_average_exact():
    # Dyadic values: every sum and mean is exact in floating point.
    a = (np.arange(4 * 4 * 8, dtype=np.float32).reshape(4, 4, 8) ** 2) / 64.0
    b2 = block_average(a, 2)
    assert b2.dtype == np.float64 and b2.shape == (2, 2, 4)
    np.testing.assert_array_equal(b2, _mean_ref(a, 2))
    # Block (0, 0, 0) = full indices {0,1} x {0,1} x {0,1}; flat index 32i + 8j + k.
    assert b2[0, 0, 0] == sum((32 * i + 8 * j + k) ** 2 for i in (0, 1) for j in (0, 1) for k in (0, 1)) / 64.0 / 8
    b4 = block_average(a, 4)
    np.testing.assert_array_equal(b4, _mean_ref(a, 4))
    assert b4.shape == (1, 1, 2) and b4[0, 0, 1] == a[:, :, 4:].astype(np.float64).sum() / 64
    with pytest.raises(ValueError, match="not divisible by 4"):
        block_average(np.zeros((4, 4, 6)), 4)


def test_block_mode_majority_and_ties():
    lab = np.zeros((2, 2, 4), np.uint8)
    lab[:, :, 0:2] = 5
    lab[0, 0, 0] = 3                       # block 0: 7x5, 1x3 -> 5
    lab[:, :, 2:4] = np.array([[[7, 2], [7, 2]], [[2, 7], [7, 2]]], np.uint8)  # block 1: 4x7, 4x2 -> tie -> 2
    m = block_mode(lab, 2)
    assert m.dtype == np.uint8 and m.shape == (1, 1, 2) and m.tolist() == [[[5, 2]]]
    # 3-way tie in a 2x2x2 block with background: smallest id wins.
    t = np.array([0, 0, 4, 4, 9, 9, 1, 2], np.uint8).reshape(2, 2, 2)
    assert block_mode(t, 2).item() == 0
    rng = np.random.default_rng(3)
    r = rng.integers(0, 19, (8, 12, 16)).astype(np.uint8)
    for k in (2, 4):
        np.testing.assert_array_equal(block_mode(r, k), _mode_ref(r, k))
    big = rng.choice(np.array([0, 2, 1001, 2035], np.uint16), (4, 4, 4))
    out = block_mode(big, 2)
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, _mode_ref(big, 2))
    with pytest.raises(ValueError, match="not divisible"):
        block_mode(np.zeros((2, 2, 3), np.uint8), 2)
    with pytest.raises(ValueError, match="integer"):
        block_mode(np.zeros((2, 2, 2), np.float32), 2)


# ----------------------------------------------------------------- train18 LUT

def test_train18_lut_every_id():
    lut = LUTS["train18"]
    expected = {0: [0], 1: [2, 41], 2: [3, 42] + list(range(1001, 1036)) + list(range(2001, 2036)),
                3: [4, 43], 4: [5, 44], 5: [7, 46], 6: [8, 47], 7: [10, 49], 8: [11, 50], 9: [12, 51],
                10: [13, 52], 11: [14], 12: [15], 13: [16], 14: [17, 53], 15: [18, 54], 16: [24],
                17: [26, 58], 18: [28, 60]}
    names = ["background", "cerebral white matter", "cerebral cortex", "lateral ventricle",
             "inferior lateral ventricle", "cerebellum white matter", "cerebellum cortex", "thalamus", "caudate",
             "putamen", "pallidum", "3rd ventricle", "4th ventricle", "brainstem", "hippocampus", "amygdala", "CSF",
             "accumbens", "ventral DC"]
    assert list(lut["classes"].values()) == names and list(lut["classes"]) == list(range(19))
    flat = {o: k for k, ids in expected.items() for o in ids}
    assert dict(lut["map"]) == flat
    # Covers exactly the 103 SynthSeg --parc ids.
    assert sorted(lut["map"]) == sorted(SYNTHSEG_IDS.tolist())
    ids = np.array(sorted(flat))
    np.testing.assert_array_equal(apply_lut(ids, lut, "uint8"), np.array([flat[i] for i in ids], np.uint8))
    with pytest.raises(ValueError, match=r"\[1, 999\]"):
        apply_lut(np.array([0, 1, 2, 999]), lut, "uint8")


def test_unknown_synthseg_id_fails_with_ids_and_case(tmp_path):
    out = str(tmp_path / "out")
    _subject(out, "s_bad", synthseg_ids=np.array([0, 2, 77, 999]))
    with pytest.raises(ValueError) as ei:
        export_npy(out, str(tmp_path / "npy"))
    msg = str(ei.value)
    assert "[77, 999]" in msg and "s_bad" in msg and "not in the fixed LUT" in msg and "train18" in msg
    # Explicit uint8 without remapping still refuses values that do not fit.
    out2 = str(tmp_path / "out2")
    _subject(out2, "s1")
    with pytest.raises(ValueError):
        export_npy(out2, str(tmp_path / "npy2"), label_dtype="uint8", remap_labels=False, label_sets=["synthseg"])


def test_uint16_fallback_for_large_lut(tmp_path):
    out = str(tmp_path / "out")
    big = _build(_identity(300), "test")
    _subject(out, "s1", synthseg_ids=np.array([0, 5, 299, 300]))
    exp = str(tmp_path / "npy")
    export_npy(out, exp, luts={"synthseg": big}, label_sets=["synthseg"])
    ll = np.load(os.path.join(exp, "ds", "s1", "labels_synthseg.npy"))
    assert ll.dtype == np.uint16 and ll.max() == 300
    assert json.load(open(os.path.join(exp, "label_luts.json")))["luts"]["synthseg"]["dtype"] == "uint16"


def test_fixed_luts_are_dataset_independent():
    assert len(LUTS["synthseg"]["map"]) == 103 and list(LUTS["synthseg"]["map"].values()) == list(range(103))
    a = apply_lut(np.array([0, 2, 60, 1001, 2035]), LUTS["synthseg"], "uint8")
    b = apply_lut(np.array([2035, 0]), LUTS["synthseg"], "uint8")
    assert a[-1] == b[0] == 102 and a[0] == b[1] == 0
    for name in ("lumir10", "seg4", "seg35"):
        lut = LUTS[name]["map"]
        assert all(k == v for k, v in lut.items())


# ----------------------------------------------------------------- geometry

def test_level_affines_and_world_center():
    full = (160, 224, 192)
    a1 = np.array([[1.0, 0, 0, -78], [0, 1.0, 0, -112], [0, 0, 1.0, -70], [0, 0, 0, 1]])
    for aff in (a1, AFFINE):
        c1 = world_center(full, aff)
        for k in (2, 4):
            ak = level_affine(aff, k)
            sk = tuple(n // k for n in full)
            np.testing.assert_allclose(world_center(sk, ak), c1, atol=1e-9)
            np.testing.assert_allclose(np.sqrt((ak[:3, :3] ** 2).sum(0)), [k] * 3)
            # Coarse voxel q sits at the mean of its block's fine voxel centers (k*q + (k-1)/2).
            q = np.array([3, 5, 7])
            fine = np.array([[k * q[0] + i, k * q[1] + j, k * q[2] + l] for i in range(k) for j in range(k)
                             for l in range(k)], float)
            w_fine = (aff[:3, :3] @ fine.T + aff[:3, 3:4]).mean(axis=1)
            np.testing.assert_allclose(ak[:3, :3] @ q + ak[:3, 3], w_fine, atol=1e-9)
    np.testing.assert_array_equal(level_affine(AFFINE, 1), AFFINE)


def test_manifest_world_centers_consistent(tmp_path):
    out = str(tmp_path / "out")
    _subject(out, "s1")
    exp = str(tmp_path / "npy")
    export_npy(out, exp)
    r = _manifest(exp)[0]
    c = [np.array([float(x) for x in r["world_center_l%d" % k].split()]) for k in (1, 2, 4)]
    np.testing.assert_allclose(c[1], c[0], atol=1e-9)
    np.testing.assert_allclose(c[2], c[0], atol=1e-9)
    np.testing.assert_allclose(np.array([float(x) for x in r["affine_l1"].split()]).reshape(4, 4), AFFINE)
    np.testing.assert_allclose(np.array([float(x) for x in r["affine_l4"].split()]).reshape(4, 4),
                               level_affine(AFFINE, 4))
    np.testing.assert_allclose(c[0], world_center(SHAPE, AFFINE))


def test_non_divisible_shape_fails(tmp_path):
    out = str(tmp_path / "out")
    _subject(out, "s1", shape=(8, 12, 6))
    with pytest.raises(ValueError, match="not divisible by 4"):
        export_npy(out, str(tmp_path / "npy"))
    export_npy(out, str(tmp_path / "npy2"), levels=[1, 2])  # 6 is divisible by 2


# ----------------------------------------------------------------- filters

def test_contrast_and_split_filters(tmp_path, caplog):
    out = str(tmp_path / "out")
    _subject(out, "a_T1w", split="train")
    _subject(out, "b_unknown0001", contrast="unknown0001", split="val")
    _subject(out, "c_T1w", split="val")
    _subject(out, "d_acq-x_T1w", contrast="acq-x_T1w", split="train")
    exp = str(tmp_path / "npy")
    with caplog.at_level(logging.INFO, logger="preprocessing"):
        assert export_npy(out, exp) == 3
    assert [r["subject_dir"] for r in _manifest(exp)] == ["a_T1w", "c_T1w", "d_acq-x_T1w"]
    assert not os.path.exists(os.path.join(exp, "ds", "b_unknown0001"))
    assert "skipped 1 by --contrast T1w" in caplog.text
    assert export_npy(out, str(tmp_path / "any"), contrasts=["any"]) == 4
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="preprocessing"):
        assert export_npy(out, str(tmp_path / "val"), splits=["val"]) == 1
    assert "skipped 1 by --contrast T1w, 2 by --split val" in caplog.text
    assert export_npy(out, str(tmp_path / "both"), contrasts=["T1w,unknown0001"], splits=["val", "train"]) == 4


# ----------------------------------------------------------------- chunks / merge

def _processed(out, n=5):
    for i in range(n):
        _subject(out, "s%d" % i, seed=i)
    summarize(os.path.join(out, "ds"))
    return os.path.join(out, "ds", "manifest_processed.csv")


def test_chunks_write_parts_and_merge(tmp_path, capsys):
    out = str(tmp_path / "out")
    _processed(out)
    exp = str(tmp_path / "npy")
    for s in (0, 2, 4):
        assert main(["export-npy", "--out-root", out, "--export-root", exp, "--dataset", "ds",
                     "--start", str(s), "--end", str(s + 2)]) == 0
    assert not os.path.exists(os.path.join(exp, "export_manifest.csv"))  # chunks never write the shared file
    parts = sorted(os.path.basename(p) for p in glob.glob(os.path.join(exp, "manifest_parts", "*.csv")))
    assert parts == ["ds_split-any_contrast-T1w_rows-%d-%d.csv" % (s, s + 2) for s in (0, 2, 4)]
    assert len(list(csv.DictReader(open(os.path.join(exp, "manifest_parts", parts[2]))))) == 1  # rows [4, 5)
    capsys.readouterr()
    assert main(["export-merge", "--export-root", exp]) == 0
    assert "5 cases" in capsys.readouterr().out
    assert [r["subject_dir"] for r in _manifest(exp)] == ["s%d" % i for i in range(5)]
    assert list(json.load(open(os.path.join(exp, "label_luts.json")))["luts"]) == ["train18"]
    # Overlapping chunk with identical content merges cleanly; explicit rows too.
    export_npy(out, exp, datasets=["ds"], start=1, end=3)
    export_npy(out, exp, datasets=["ds"], rows=[4, 0])
    assert merge_parts(exp) == 5
    # Different options for the same case in another part -> conflict.
    export_npy(out, exp, datasets=["ds"], rows=[0], label_sets=["train18,synthseg"])
    with pytest.raises(ValueError, match="differs between"):
        merge_parts(exp)
    assert main(["export-merge", "--export-root", exp]) == 1
    with pytest.raises(ValueError, match="no manifest parts"):
        merge_parts(str(tmp_path / "empty"))


def test_row_selection_needs_processed_manifest(tmp_path):
    out = str(tmp_path / "out")
    _subject(out, "s1")
    with pytest.raises(ValueError, match="manifest_processed.csv"):
        export_npy(out, str(tmp_path / "npy"), start=0, end=1)
    _processed(out, 1)  # manifest_processed.csv with s0, s1
    with pytest.raises(ValueError, match="either --rows"):
        export_npy(out, str(tmp_path / "npy"), start=0, rows=[0])
    assert export_npy(out, str(tmp_path / "npy"), start=1, end=10) == 1
    with pytest.raises(ValueError, match="out of range"):
        export_npy(out, str(tmp_path / "npy"), rows=[3])


# ----------------------------------------------------------------- atomic / resume

def test_atomic_write_leaves_no_partial_file(tmp_path, monkeypatch):
    out = str(tmp_path / "out")
    _subject(out, "s1")
    exp = str(tmp_path / "npy")
    real = np.save
    calls = {"n": 0}

    def flaky(f, arr, *a, **k):
        calls["n"] += 1
        if calls["n"] == 3:  # third file of the case (image_4.npy)
            f.write(b"\x93NUMPY partial")
            raise KeyboardInterrupt("killed")
        return real(f, arr, *a, **k)

    monkeypatch.setattr(ex.np, "save", flaky)
    with pytest.raises(KeyboardInterrupt):
        export_npy(out, exp)
    monkeypatch.setattr(ex.np, "save", real)
    e = os.path.join(exp, "ds", "s1")
    assert sorted(os.listdir(e)) == ["image.npy", "image_2.npy"]  # no truncated image_4.npy, no temp file
    # Resume completes the case.
    assert export_npy(out, exp) == 1
    assert len(os.listdir(e)) == 6


def test_resume_skip_rewrite_and_overwrite(tmp_path, caplog):
    out = str(tmp_path / "out")
    _subject(out, "s1")
    exp = str(tmp_path / "npy")
    export_npy(out, exp)
    e = os.path.join(exp, "ds", "s1")
    paths = sorted(glob.glob(os.path.join(e, "*.npy")))
    mt = {p: os.stat(p).st_mtime_ns for p in paths}
    os.utime(paths[0], ns=(1, 1))  # mark: kept files keep this mtime
    with caplog.at_level(logging.INFO, logger="preprocessing"):
        assert export_npy(out, exp) == 1
    assert os.stat(paths[0]).st_mtime_ns == 1 and "1 already complete and kept" in caplog.text
    # A missing file -> the case is rewritten.
    os.remove(os.path.join(e, "labels_train18_4.npy"))
    export_npy(out, exp)
    assert os.path.isfile(os.path.join(e, "labels_train18_4.npy")) and os.stat(paths[0]).st_mtime_ns != 1
    # A truncated file (e.g. copied by hand) is unreadable -> rewritten.
    with open(os.path.join(e, "image_2.npy"), "wb") as f:
        f.write(b"\x93NUMPY")
    export_npy(out, exp)
    assert np.load(os.path.join(e, "image_2.npy")).shape == (4, 6, 8)
    # Wrong dtype (another export configuration) -> error without --overwrite.
    np.save(os.path.join(e, "image.npy"), np.zeros(SHAPE, np.float16))
    with pytest.raises(ValueError) as ei:
        export_npy(out, exp)
    assert "image.npy" in str(ei.value) and "float16" in str(ei.value) and "--overwrite" in str(ei.value)
    assert main(["export-npy", "--out-root", out, "--export-root", exp, "--overwrite"]) == 0
    assert np.load(os.path.join(e, "image.npy")).dtype == np.float32
    assert set(mt) == set(glob.glob(os.path.join(e, "*.npy")))


# ----------------------------------------------------------------- template / check

def test_template_per_level(tmp_path):
    rng = np.random.default_rng(5)
    t = rng.random(SHAPE).astype(np.float32)
    m = (rng.random(SHAPE) > 0.5).astype(np.uint8)
    m[0:2, 0:2, 0:2] = np.array([1, 1, 1, 1, 0, 0, 0, 0]).reshape(2, 2, 2)  # 4/4 tie -> 0
    m[2:4, 0:2, 0:2] = np.array([1, 1, 1, 1, 1, 0, 0, 0]).reshape(2, 2, 2)  # 5/8 -> 1
    tp = {"template": ph.save(str(tmp_path / "t" / "template.nii.gz"), t, AFFINE),
          "template_mask": ph.save(str(tmp_path / "t" / "mask.nii.gz"), m, AFFINE, np.uint8)}
    out = str(tmp_path / "out")
    _subject(out, "s1")
    exp = str(tmp_path / "npy")
    export_npy(out, exp, template_paths=tp)
    td = os.path.join(exp, "template")
    assert sorted(os.listdir(td)) == ["template.npy", "template_2.npy", "template_4.npy", "template_info.json",
                                      "template_mask.npy", "template_mask_2.npy", "template_mask_4.npy"]
    np.testing.assert_array_equal(np.load(os.path.join(td, "template.npy")), t)
    for k in (2, 4):
        np.testing.assert_allclose(np.load(os.path.join(td, "template_%d.npy" % k)), _mean_ref(t, k),
                                   rtol=0, atol=1e-6)
        mk = np.load(os.path.join(td, "template_mask_%d.npy" % k))
        assert mk.dtype == np.uint8
        np.testing.assert_array_equal(mk, _mode_ref(m, k))
    mk2 = np.load(os.path.join(td, "template_mask_2.npy"))
    assert mk2[0, 0, 0] == 0 and mk2[1, 0, 0] == 1
    info = json.load(open(os.path.join(td, "template_info.json")))
    assert info["mask_rule"].startswith("block majority vote")
    c = [np.array(info["levels"][str(k)]["world_center"]) for k in (1, 2, 4)]
    np.testing.assert_allclose(c[1], c[0], atol=1e-9)
    np.testing.assert_allclose(c[2], c[0], atol=1e-9)
    assert info["levels"]["4"]["shape"] == [2, 3, 4] and info["levels"]["2"]["spacing"] == [2.0, 2.0, 2.0]


def test_cli_include_template_from_config(tmp_path, config_file):
    out = str(tmp_path / "out")
    _subject(out, "s1", shape=(24, 40, 34))  # conftest target grid: 34 is not divisible by 4
    exp = str(tmp_path / "npy")
    assert main(["export-npy", "--out-root", out, "--export-root", exp, "--include-template", "--config",
                 config_file]) == 1  # fails loudly on --levels 4
    assert main(["export-npy", "--out-root", out, "--export-root", exp, "--include-template", "--config",
                 config_file, "--levels", "1,2"]) == 0
    assert np.load(os.path.join(exp, "template", "template_2.npy")).shape == (12, 20, 17)


def test_export_check(tmp_path, capsys):
    out = str(tmp_path / "out")
    _processed(out, 3)
    exp = str(tmp_path / "npy")
    export_npy(out, exp)
    n, problems, report = export_check(exp, sample=0)
    assert n == 3 and problems == [] and any("max world-center difference" in x for x in report)
    assert any("train18=" in x for x in report)
    capsys.readouterr()
    assert main(["export-check", "--export-root", exp, "--sample", "2"]) == 0
    assert "checked 2 cases: OK" in capsys.readouterr().out
    e = os.path.join(exp, "ds", "s1")
    np.save(os.path.join(e, "labels_train18_2.npy"), np.full((4, 6, 8), 19, np.uint8))  # out of range
    np.save(os.path.join(e, "image_4.npy"), np.zeros((2, 3, 4), np.float16))  # wrong dtype
    n, problems, _ = export_check(exp, sample=0)
    assert len(problems) == 2 and any("max label 19 outside 0..18" in p for p in problems)
    assert any("image_4.npy mismatch" in p for p in problems)
    assert main(["export-check", "--export-root", exp, "--sample", "0"]) == 1
