import csv
import json
import os

import numpy as np
import pytest

from preprocessing.cli import main
from preprocessing.qc import compute_qc, robust_z, summarize, write_qc_png

from . import phantoms as ph


def _subject(root, name, img, mask, labels=None, status="completed", meta_extra=None):
    d = os.path.join(root, name)
    aff = ph.target_affine()
    ph.save(os.path.join(d, "image.nii.gz"), img, aff)
    ph.save(os.path.join(d, "mask.nii.gz"), mask, aff, np.uint8)
    if labels is not None:
        ph.save(os.path.join(d, "labels_synthseg.nii.gz"), labels, aff, np.uint16)
    meta = {"status": status, "row": {"dataset": "ds", "subject": name, "session": "", "contrast": "T1w",
                                      "label_paths": {}, "image_path": "/x"}}
    meta.update(meta_extra or {})
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    return d, meta


def test_compute_qc_known_values(tmp_path):
    shape = ph.TARGET_SHAPE
    mask = np.zeros(shape, np.uint8)
    mask[5:15, 10:30, 10:20] = 1
    img = mask * np.linspace(0, 1, shape[2])[None, None, :]
    lab = mask * 3
    lab[5:10] = 0
    d, meta = _subject(str(tmp_path), "s1", img, mask, lab)
    tmask = np.zeros(shape, bool)
    tmask[5:15, 10:30, 10:25] = True
    q = compute_qc(d, img.astype(np.float32), ph.target_affine(), tmask, meta=meta)
    assert q["affine_match_final"] is True
    assert q["mask_volume_cc"] == pytest.approx(2000 / 1000.0)
    assert q["dice_mask_synthseg"] == pytest.approx(2 * 1000 / 3000)
    assert q["dice_mask_template"] == pytest.approx(2 * 2000 / (2000 + 3000))
    assert q["n_labels_synthseg"] == 1
    assert q["ncc_to_template"] == pytest.approx(1.0)
    assert q["p50"] == pytest.approx(np.percentile(img[mask > 0], 50))
    png = os.path.join(d, "qc.png")
    write_qc_png(png, img, mask, img, lab, title="s1")
    assert os.path.getsize(png) > 1000


def test_robust_z():
    z = robust_z([1.0, 1.1, 0.9, 1.0, 10.0, None])
    assert abs(z[4]) > 4 and z[5] == 0
    assert np.all(robust_z([2.0, 2.0, 2.0, 2.0]) == 0)  # MAD == 0 -> no flags


def test_summarize_flags_outlier(tmp_path):
    root = str(tmp_path / "out" / "ds")
    xyz = ph.world_coords(ph.TARGET_SHAPE, ph.target_affine())
    base = ph.brain_mask(xyz).astype(np.uint8)
    inten = 0.3 + 0.7 * ph.brain(xyz)
    for i in range(6):  # evenly spread normal subjects (slightly scaled brains)
        m = ph.brain_mask(xyz / (0.96 + 0.02 * i)).astype(np.uint8)
        _subject(root, "s%d" % i, m * inten, m)
    big = np.zeros_like(base)
    big[1:-1, 1:-1, 1:-1] = 1   # mask far too large
    _subject(root, "s_big", big * inten, big)
    _subject(root, "s_fail", base * inten, base, status="failed")
    tmpl = ph.brain(xyz) * base
    for d in sorted(os.listdir(root)):
        q = compute_qc(os.path.join(root, d), tmpl, ph.target_affine(), base > 0)
        with open(os.path.join(root, d, "qc.json"), "w") as f:
            json.dump(q, f)
    assert main(["qc-summary", "--out-root", str(tmp_path / "out"), "--dataset", "ds"]) == 0
    rows = {r["subject_dir"]: r for r in csv.DictReader(open(os.path.join(root, "qc_summary.csv")))}
    assert rows["s_big"]["flagged"] == "True" and "mask_volume_cc_z" in rows["s_big"]["reasons"]
    assert rows["s_fail"]["flagged"] == "True" and "status=failed" in rows["s_fail"]["reasons"]
    assert rows["s0"]["flagged"] == "False"
    man = list(csv.DictReader(open(os.path.join(root, "manifest_processed.csv"))))
    assert len(man) == 8 and man[0]["image"].endswith("image.nii.gz")
    assert man[0]["mask"].endswith("mask.nii.gz")
    assert summarize(root, 4.0) == (8, 2)
    summary = json.load(open(os.path.join(root, "qc_summary.json")))
    assert summary["n_flagged"] == 2


def test_png_on_demand(tmp_path, config_file, capsys):
    root = str(tmp_path / "out" / "ds")
    xyz = ph.world_coords(ph.TARGET_SHAPE, ph.target_affine())
    m = ph.brain_mask(xyz).astype(np.uint8)
    for name in ["s%d" % i for i in range(5)] + ["s_fail"]:
        d, _ = _subject(root, name, m * 0.5, m, status="failed" if name == "s_fail" else "completed")
        os.remove(os.path.join(d, "mask.nii.gz"))  # outputs.save_mask false
        with open(os.path.join(d, "qc.json"), "w") as f:
            json.dump(compute_qc(d, m * 0.5, ph.target_affine(), m > 0), f)
    base = ["qc-summary", "--out-root", str(tmp_path / "out"), "--dataset", "ds"]
    with pytest.raises(SystemExit):
        main(base + ["--png", "flagged"])          # needs --config
    with pytest.raises(SystemExit):
        main(base + ["--config", config_file, "--png", "sample:0"])
    pngs = lambda: sorted(p for p in os.listdir(root) if os.path.isfile(os.path.join(root, p, "qc.png")))  # noqa
    assert main(base) == 0 and pngs() == []        # qc.json/summary only, no PNGs by default
    assert main(base + ["--config", config_file, "--png", "flagged"]) == 0
    assert pngs() == ["s_fail"]                    # status=failed / missing_qc rows are flagged
    assert main(base + ["--config", config_file, "--png", "sample:2", "--png-seed", "1"]) == 0
    assert len(pngs()) in (2, 3)
    assert main(base + ["--config", config_file, "--png", "all"]) == 0
    assert len(pngs()) == 6
    assert os.path.getsize(os.path.join(root, "s0", "qc.png")) > 1000
    assert "rendered 6 qc.png" in capsys.readouterr().out


def test_compute_qc_without_saved_mask(tmp_path):
    shape = ph.TARGET_SHAPE
    mask = np.zeros(shape, np.uint8)
    mask[5:15, 10:30, 10:20] = 1
    img = mask * 0.5
    img[10, 20, 15] = 0.0  # interior hole: kept by the saved mask, lost by image > 0
    d, meta = _subject(str(tmp_path), "s1", img, mask)
    q = compute_qc(d, img.astype(np.float32), ph.target_affine(), mask > 0, meta=meta)
    assert q["mask_source"] == "mask.nii.gz" and q["mask_volume_cc"] == pytest.approx(2.0)
    os.remove(os.path.join(d, "mask.nii.gz"))
    q = compute_qc(d, img.astype(np.float32), ph.target_affine(), mask > 0, meta=meta)
    assert q["mask_source"] == "image>0" and q["mask_volume_cc"] == pytest.approx(1.999)
