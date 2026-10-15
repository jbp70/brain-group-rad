"""SynthSeg batch mode with the fake mri_synthseg."""
import json
import os

import numpy as np

from preprocessing import nifti
from preprocessing.cli import main
from preprocessing.commands import synthseg_cmd
from preprocessing.config import load_config
from preprocessing.manifest import subject_dir_name

from . import datasets as ds
from . import phantoms as ph


def _prepare(tmp_path, config_file):
    rows = [ds.lumir_row(str(tmp_path / "in")), ds.lumir_row(str(tmp_path / "in"), "LUMIRMRI_0002"),
            ds.neurite_row(str(tmp_path / "in"))]
    man = ds.write_rows(str(tmp_path / "m.csv"), rows)
    out = str(tmp_path / "out")
    assert main(["run", "--config", config_file, "--manifest", man, "--out-root", out]) == 0
    return rows, man, out


def test_batch_single_call(tmp_path, fake_tools, config_file):
    rows, man, out = _prepare(tmp_path, config_file)
    n = len(fake_tools.calls())
    assert main(["synthseg", "--config", config_file, "--manifest", man, "--out-root", out]) == 0
    calls = [c for c in fake_tools.calls()[n:] if c["tool"] == "mri_synthseg"]
    assert len(calls) == 1
    batch = os.path.join(out, "_synthseg_batches", "m_0_3")
    cfg = load_config(config_file)
    lists = [os.path.join(batch, x) for x in ("inputs.txt", "outputs.txt", "vols.txt", "qc.txt")]
    assert calls[0]["argv"] == synthseg_cmd("x", lists[0], lists[1], cfg["synthseg"], lists[2], lists[3])[1:]
    dirs = [os.path.join(out, r.dataset, subject_dir_name(r)) for r in rows]
    assert open(lists[0]).read().split() == [os.path.join(d, "image.nii.gz") for d in dirs]
    for d in dirs:
        lab, li = nifti.load_data(os.path.join(d, "labels_synthseg.nii.gz"), np.int64)
        assert li.get_data_dtype() == np.uint16 and lab.shape == ph.TARGET_SHAPE
        np.testing.assert_allclose(li.affine, ph.target_affine(), atol=1e-4)
        meta = json.load(open(os.path.join(d, "meta.json")))
        assert meta["steps"]["synthseg"]["status"] == "done" and meta["synthseg_resampled"] is False
        q = json.load(open(os.path.join(d, "qc.json")))
        assert q["n_labels_synthseg"] >= 1 and q["dice_mask_synthseg"] > 0.9
    # Resume: nothing left to do -> no call.
    n = len(fake_tools.calls())
    assert main(["synthseg", "--config", config_file, "--manifest", man, "--out-root", out]) == 0
    assert len(fake_tools.calls()) == n


def test_crop_fallback_and_cpu_flag(tmp_path, fake_tools, config_file, monkeypatch):
    rows, man, out = _prepare(tmp_path, config_file)
    monkeypatch.setenv("FAKE_SYNTHSEG_CROP", "1")
    assert main(["synthseg", "--config", config_file, "--manifest", man, "--out-root", out, "--rows", "2",
                 "--synthseg-cpu", "--threads", "3"]) == 0
    call = [c for c in fake_tools.calls() if c["tool"] == "mri_synthseg"][-1]
    assert call["argv"][-3:] == ["--threads", "3", "--cpu"]
    d = os.path.join(out, rows[2].dataset, subject_dir_name(rows[2]))
    lab, li = nifti.load_data(os.path.join(d, "labels_synthseg.nii.gz"), np.int64)
    assert lab.shape == ph.TARGET_SHAPE and json.load(open(os.path.join(d, "meta.json")))["synthseg_resampled"]
    # Cropped border is label 0, interior preserved (nearest fallback, no new labels).
    assert np.all(lab[:2] == 0) and set(np.unique(lab)) <= {0, 2, 3, 24}


def test_run_with_synthseg_step_and_dry_run(tmp_path, fake_tools, config_file, capsys):
    row = ds.lumir_row(str(tmp_path / "in"))
    man = ds.write_rows(str(tmp_path / "m.csv"), [row])
    out = str(tmp_path / "out")
    assert main(["synthseg", "--config", config_file, "--manifest", man, "--out-root", out, "--dry-run"]) == 0
    assert "CMD " in capsys.readouterr().out and not os.path.exists(out)
    steps = "strip,resample,normalize,synthseg,qc"
    assert main(["run", "--config", config_file, "--manifest", man, "--out-root", out, "--steps", steps]) == 0
    assert [c["tool"] for c in fake_tools.calls()] == ["mri_synthseg"]
    d = os.path.join(out, row.dataset, subject_dir_name(row))
    meta = json.load(open(os.path.join(d, "meta.json")))
    assert meta["status"] == "completed" and meta["steps"]["synthseg"]["status"] == "done"
    assert os.path.isfile(os.path.join(d, "labels_synthseg.nii.gz"))
