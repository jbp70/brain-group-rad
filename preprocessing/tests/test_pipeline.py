"""End-to-end `run` with fake tools on PATH."""
import json
import os

import numpy as np
import pytest

from preprocessing import nifti
from preprocessing.cli import main
from preprocessing.commands import ants_apply_cmd, flirt_register_cmd
from preprocessing.config import load_config
from preprocessing.intensity import normalize_intensity
from preprocessing.manifest import subject_dir_name
from preprocessing.masks import derive_mask, dice

from . import datasets as ds
from . import phantoms as ph
from .conftest import write_config


def _run(cfg, manifest, out, *extra):
    return main(["run", "--config", cfg, "--manifest", manifest, "--out-root", out] + list(extra))


def _sd(out, row):
    return os.path.join(out, row.dataset, subject_dir_name(row))


def _meta(d):
    with open(os.path.join(d, "meta.json")) as f:
        return json.load(f)


def _check_outputs(d, labels=(), saved=False):
    """Check the row outputs; saved=True: mask.nii.gz/qc.png/work expected (outputs.* true)."""
    img, im = nifti.load_data(os.path.join(d, "image.nii.gz"))
    assert img.shape == ph.TARGET_SHAPE and im.get_data_dtype() == np.float32
    np.testing.assert_allclose(im.affine, ph.target_affine(), atol=1e-4)
    if saved:
        mask, mm = nifti.load_data(os.path.join(d, "mask.nii.gz"), np.uint8)
        assert mm.get_data_dtype() == np.uint8 and set(np.unique(mask)) <= {0, 1}
        np.testing.assert_allclose(mm.affine, ph.target_affine(), atol=1e-4)
    else:
        mask = (img > 0).astype(np.uint8)
        for f in ("mask.nii.gz", "qc.png"):
            assert not os.path.exists(os.path.join(d, f)), f
    assert img.min() >= 0 and img.max() == pytest.approx(1.0) and np.all(img[mask == 0] == 0)
    for name in labels:
        lab, li = nifti.load_data(os.path.join(d, "labels_%s.nii.gz" % name), np.int64)
        assert li.get_data_dtype() == np.uint16 and lab.shape == ph.TARGET_SHAPE
    for f in ("meta.json", "qc.json") + (("qc.png",) if saved else ()):
        assert os.path.isfile(os.path.join(d, f)), f
    return img, mask


def _template_mask():
    xyz = ph.world_coords(ph.TARGET_SHAPE, ph.target_affine())
    return ph.brain_mask(xyz)


def test_lumir_row_keeps_array_and_gets_corrected_affine(tmp_path, fake_tools, config_file):
    row = ds.lumir_row(str(tmp_path / "in"))
    man = ds.write_rows(str(tmp_path / "m.csv"), [row])
    out = str(tmp_path / "out")
    assert _run(config_file, man, out) == 0
    assert fake_tools.calls() == []  # no strip/N4/registration/resampling tools
    d = _sd(out, row)
    img, mask = _check_outputs(d, ["lumir10"])
    orig = nifti.load_data(row.image_path)[0]
    m = derive_mask(orig)
    exp, _ = normalize_intensity(orig * m, m)
    np.testing.assert_array_equal(img, exp)          # same voxel array layout, only normalized
    lab_out = nifti.load_data(os.path.join(d, "labels_lumir10.nii.gz"), np.int64)[0]
    lab_in = nifti.load_data(row.label_paths["lumir10"], np.int64)[0]
    np.testing.assert_array_equal(lab_out, lab_in)   # byte-identical labels
    meta = _meta(d)
    assert meta["status"] == "completed" and meta["grid_state"] == "header" and not meta["grid_mismatch"]
    np.testing.assert_allclose(meta["original_affine"], np.eye(4))
    np.testing.assert_allclose(meta["corrected_affine"], ph.target_affine())
    q = json.load(open(os.path.join(d, "qc.json")))
    assert q["affine_match_final"] and q["dice_mask_template"] > 0.95


def test_lumir_off_grid_row_flagged_and_resampled(tmp_path, fake_tools, config_file):
    row = ds.lumir_row(str(tmp_path / "in"), shape=(22, 40, 34), header=ph.target_affine())
    man = ds.write_rows(str(tmp_path / "m.csv"), [row])
    out = str(tmp_path / "out")
    assert _run(config_file, man, out) == 0
    assert fake_tools.calls() == []
    d = _sd(out, row)
    _check_outputs(d, ["lumir10"])
    meta = _meta(d)
    assert meta["grid_mismatch"] is True and meta["grid_state"] == "mismatch"
    assert json.load(open(os.path.join(d, "qc.json")))["grid_mismatch"] is True


def test_neurite_row_flirt(tmp_path, fake_tools, config_file, mni_files):
    row = ds.neurite_row(str(tmp_path / "in"))
    man = ds.write_rows(str(tmp_path / "m.csv"), [row])
    out = str(tmp_path / "out")
    assert _run(config_file, man, out) == 0
    calls = fake_tools.calls()
    assert ds.calls_by_tool(calls) == {"flirt": 4}  # 1 registration + image/mask/seg35
    d = _sd(out, row)
    cfg = load_config(config_file)
    assert calls[0]["argv"] == flirt_register_cmd(
        "x", os.path.join(d, "work", "stripped.nii.gz"), mni_files.image_2mm,
        os.path.join(d, "work", "flirt_reg.mat"), cfg["registration"]["flirt"])[1:]
    applies = calls[1:]
    assert [c["argv"][c["argv"].index("-interp") + 1] for c in applies] == ["spline", "nearestneighbour",
                                                                            "nearestneighbour"]
    assert applies[0]["argv"][-2:] == ["-datatype", "float"]
    assert all(c["argv"][c["argv"].index("-ref") + 1].endswith("template_lumirgrid.nii.gz") for c in applies)
    assert applies[2]["argv"][1] == row.label_paths["seg35"]
    _, mask = _check_outputs(d, ["seg35"])
    np.testing.assert_allclose(nifti.read_matrix_txt(os.path.join(d, "affine_to_template_world.txt")),
                               np.eye(4), atol=1e-6)
    assert dice(mask, _template_mask()) > 0.9
    lab = nifti.load_data(os.path.join(d, "labels_seg35.nii.gz"), np.int64)[0]
    assert set(np.unique(lab)) <= {0, 2, 4, 41} and (lab == 41).any()
    meta = _meta(d)
    assert meta["mask_source"] == "derived" and meta["registration"]["backend"] == "flirt"
    assert "n4" not in meta["steps"] and meta["steps"]["register"]["status"] == "done"


def test_flirt_world_shift_is_composed_once(tmp_path, fake_tools, target_dir, mni_files, monkeypatch):
    monkeypatch.setenv("FAKE_FLIRT_SHIFT", "2,0,0")
    config_file = write_config(tmp_path / "c.yaml", target_dir, mni_files, outputs={"save_mask": True})
    row = ds.neurite_row(str(tmp_path / "in"))
    man = ds.write_rows(str(tmp_path / "m.csv"), [row])
    out = str(tmp_path / "out")
    assert _run(config_file, man, out) == 0
    d = _sd(out, row)
    w = nifti.read_matrix_txt(os.path.join(d, "affine_to_template_world.txt"))
    np.testing.assert_allclose(w[:3, 3], [2, 0, 0], atol=1e-6)
    mask = nifti.load_data(os.path.join(d, "mask.nii.gz"))[0]
    com = lambda m: (ph.target_affine() @ np.r_[np.argwhere(m > 0).mean(0), 1])[:3]  # noqa: E731
    np.testing.assert_allclose(com(mask) - com(_template_mask()), [2, 0, 0], atol=0.3)


def test_ultracortex_row_strip_n4_antialias(tmp_path, fake_tools, config_file):
    row = ds.ultracortex_row(str(tmp_path / "in"))
    man = ds.write_rows(str(tmp_path / "m.csv"), [row])
    out = str(tmp_path / "out")
    assert _run(config_file, man, out, "--keep-work") == 0
    assert ds.calls_by_tool(fake_tools.calls()) == {"mri_synthstrip": 1, "N4BiasFieldCorrection": 1, "flirt": 3}
    d = _sd(out, row)
    assert os.path.basename(d) == "sub-01_ses-1_T1w"
    _, mask = _check_outputs(d)
    assert dice(mask, _template_mask()) > 0.85
    meta = _meta(d)
    sig = meta["antialias"]["sigma_vox"]
    assert sig == pytest.approx([0.3033 / 0.7] * 3, abs=2e-3)
    assert os.path.isfile(os.path.join(d, "work", "smoothed.nii.gz"))
    assert os.path.isfile(os.path.join(d, "work", "reoriented.nii.gz"))
    assert meta["mask_source"] == "synthstrip"


def test_mindboggle_row_template_space_strip_n4_union_labels(tmp_path, fake_tools, target_dir, mni_files, capsys):
    cfg = write_config(tmp_path / "mb.yaml", target_dir, mni_files, mask={"union_labels": True})
    row = ds.mindboggle_row(str(tmp_path / "in"))
    man = ds.write_rows(str(tmp_path / "m.csv"), [row])
    out = str(tmp_path / "out")
    assert _run(cfg, man, out, "--dry-run") == 0
    text = capsys.readouterr().out
    d = _sd(out, row)
    # Template-space resampling reads the N4 output, not the stripped image.
    assert "PY resample(world, own header): %s" % os.path.join(d, "work", "n4.nii.gz") in text
    assert "PY strip(union SynthStrip mask with labels > 0)" in text
    assert _run(cfg, man, out) == 0
    # No registration, no reorientation, no FLIRT resampling: SynthStrip + N4 only.
    assert ds.calls_by_tool(fake_tools.calls()) == {"mri_synthstrip": 1, "N4BiasFieldCorrection": 1}
    img, mask = _check_outputs(d, ["dkt31aseg"])
    meta = _meta(d)
    assert meta["status"] == "completed" and meta["grid_state"] == "mismatch"
    assert meta["mask_source"] == "synthstrip+labels"
    assert meta["mask_union_labels"]["added_label_voxels"] > 0  # fake SynthStrip mask is tighter than seg
    assert meta["label_voxels_lost_resample"] == {"dkt31aseg": 0}
    assert "register" not in meta["steps"] and "reorient" not in meta["steps"]
    assert not os.path.exists(os.path.join(d, "work"))  # intermediates deleted
    lab = nifti.load_data(os.path.join(d, "labels_dkt31aseg.nii.gz"), np.int64)[0]
    xyz = ph.world_coords(ph.TARGET_SHAPE, ph.target_affine())
    np.testing.assert_array_equal(lab, ph.seg(xyz))  # world-consistent, label IDs unchanged
    assert np.all(img[lab > 0] > 0)                  # no labelled voxel stripped away
    assert dice(mask, _template_mask()) > 0.9


def test_mindboggle_native_hln_row_registered(tmp_path, fake_tools, target_dir, mni_files):
    cfg = write_config(tmp_path / "mb.yaml", target_dir, mni_files, mask={"union_labels": True})
    row = ds.mindboggle_native_row(str(tmp_path / "in"))
    man = ds.write_rows(str(tmp_path / "m.csv"), [row])
    out = str(tmp_path / "out")
    assert _run(cfg, man, out) == 0
    calls = fake_tools.calls()
    # SynthStrip + N4 + FLIRT (1 registration + image/mask/labels applied with the same matrix).
    assert ds.calls_by_tool(calls) == {"mri_synthstrip": 1, "N4BiasFieldCorrection": 1, "flirt": 4}
    d = _sd(out, row)
    reg = [c["argv"] for c in calls if c["tool"] == "flirt"][0]
    assert reg[reg.index("-in") + 1] == os.path.join(d, "work", "n4.nii.gz")
    applies = [c["argv"] for c in calls if c["tool"] == "flirt"][1:]
    assert [a[a.index("-interp") + 1] for a in applies][1:] == ["nearestneighbour", "nearestneighbour"]
    assert applies[2][1] == row.label_paths["dkt31aseg"]
    img, mask = _check_outputs(d, ["dkt31aseg"])
    meta = _meta(d)
    assert meta["status"] == "completed" and meta["mask_source"] == "synthstrip+labels"
    assert meta["mask_union_labels"]["added_label_voxels"] > 0
    assert meta["steps"]["reorient"]["status"] == "done" and meta["steps"]["register"]["status"] == "done"
    # Fake FLIRT finds the true (identity world) mapping: labels land on the right anatomy, IDs unchanged.
    lab = nifti.load_data(os.path.join(d, "labels_dkt31aseg.nii.gz"), np.int64)[0]
    exp = ph.seg(ph.world_coords(ph.TARGET_SHAPE, ph.target_affine()))
    assert set(np.unique(lab)) <= {0, 2, 4, 41}
    assert dice(lab == 41, exp == 41) > 0.99 and dice(lab == 2, exp == 2) > 0.99  # left/right preserved
    assert dice(lab == 41, exp == 2) < 0.1                                         # not mirrored
    assert dice(mask, _template_mask()) > 0.85


def test_union_labels_off_keeps_synthstrip_mask(tmp_path, fake_tools, config_file):
    row = ds.mindboggle_row(str(tmp_path / "in"))
    man = ds.write_rows(str(tmp_path / "m.csv"), [row])
    out = str(tmp_path / "out")
    assert _run(config_file, man, out) == 0
    meta = _meta(_sd(out, row))
    assert meta["mask_source"] == "synthstrip" and "mask_union_labels" not in meta


def test_generic_row_ants_backend(tmp_path, fake_tools, target_dir, mni_files):
    cfg = write_config(tmp_path / "ants.yaml", target_dir, mni_files, registration={"backend": "ants"})
    row = ds.generic_row(str(tmp_path / "in"))
    man = ds.write_rows(str(tmp_path / "m.csv"), [row])
    out = str(tmp_path / "out")
    assert _run(cfg, man, out) == 0
    calls = fake_tools.calls()
    assert ds.calls_by_tool(calls) == {"mri_synthstrip": 1, "N4BiasFieldCorrection": 1, "antsRegistration": 1,
                                       "antsApplyTransforms": 3}
    d = _sd(out, row)
    tpl = os.path.join(target_dir, "template_lumirgrid.nii.gz")
    mat = os.path.join(d, "affine_to_template.mat")
    apply_calls = [c["argv"] for c in calls if c["tool"] == "antsApplyTransforms"]
    assert apply_calls[0] == ants_apply_cmd("x", os.path.join(d, "work", "n4.nii.gz"), tpl,
                                            os.path.join(d, "work", "image_resampled.nii.gz"), "Linear", mat)[1:]
    assert apply_calls[2][apply_calls[2].index("-n") + 1] == "GenericLabel"
    reg = [c["argv"] for c in calls if c["tool"] == "antsRegistration"][0]
    assert "[%s,%s,1]" % (mni_files.image_1mm, os.path.join(d, "work", "n4.nii.gz")) in reg
    _, mask = _check_outputs(d, ["manual"])
    assert dice(mask, _template_mask()) > 0.85
    assert not os.path.isfile(os.path.join(d, "work", "reoriented.nii.gz"))  # ants: no reorient step


def test_dry_run_prints_and_creates_nothing(tmp_path, fake_tools, config_file, capsys):
    rows = [ds.lumir_row(str(tmp_path / "in")), ds.neurite_row(str(tmp_path / "in")),
            ds.ultracortex_row(str(tmp_path / "in"))]
    man = ds.write_rows(str(tmp_path / "m.csv"), rows)
    out = str(tmp_path / "out")
    assert _run(config_file, man, out, "--dry-run") == 0
    text = capsys.readouterr().out
    assert not os.path.exists(out) and fake_tools.calls() == []
    cmds = [x for x in text.splitlines() if x.startswith("CMD ")]
    assert any("/mri_synthstrip -i " in x for x in cmds)
    assert any("/N4BiasFieldCorrection -d 3" in x for x in cmds)
    assert sum(" -omat " in x for x in cmds) == 2 and sum(" -applyxfm " in x for x in cmds) == 5
    assert "PY strip(derived mask)" in text and "PY normalize" in text and "PY resample(antialias" in text


def test_resume_overwrite_and_clean_work(tmp_path, fake_tools, target_dir, mni_files):
    rows = [ds.neurite_row(str(tmp_path / "in")), ds.ultracortex_row(str(tmp_path / "in"))]
    man = ds.write_rows(str(tmp_path / "m.csv"), rows)
    out = str(tmp_path / "out")
    config_file = write_config(tmp_path / "c.yaml", target_dir, mni_files)
    assert _run(config_file, man, out, "--keep-work") == 0
    n = len(fake_tools.calls())
    assert _run(config_file, man, out, "--keep-work") == 0
    assert len(fake_tools.calls()) == n  # everything skipped
    # Only one step missing -> only that step reruns.
    os.remove(os.path.join(_sd(out, rows[1]), "work", "n4.nii.gz"))
    assert _run(config_file, man, out, "--rows", "1", "--keep-work") == 0
    new = fake_tools.calls()[n:]
    assert [c["tool"] for c in new] == ["N4BiasFieldCorrection"]
    n = len(fake_tools.calls())
    assert _run(config_file, man, out, "--rows", "0", "--overwrite", "--keep-work") == 0
    assert len(fake_tools.calls()) == n + 4
    assert os.path.isdir(os.path.join(_sd(out, rows[0]), "work"))
    # --clean-work forces deletion even when the config keeps work/.
    keep_cfg = write_config(tmp_path / "k.yaml", target_dir, mni_files, outputs={"keep_work": True})
    assert _run(keep_cfg, man, out, "--rows", "0", "--overwrite", "--clean-work") == 0
    d0 = _sd(out, rows[0])
    assert not os.path.isdir(os.path.join(d0, "work")) and _meta(d0)["work_cleaned"]
    n = len(fake_tools.calls())
    assert _run(config_file, man, out, "--rows", "0") == 0
    assert len(fake_tools.calls()) == n
    with pytest.raises(SystemExit):
        _run(config_file, man, out, "--keep-work", "--clean-work")


def test_default_storage_cleans_work_and_resumes(tmp_path, fake_tools, config_file):
    rows = [ds.lumir_row(str(tmp_path / "in")), ds.neurite_row(str(tmp_path / "in"))]
    man = ds.write_rows(str(tmp_path / "m.csv"), rows)
    out = str(tmp_path / "out")
    assert _run(config_file, man, out) == 0
    for r in rows:
        d = _sd(out, r)
        _check_outputs(d, list(r.label_paths))   # no mask.nii.gz, no qc.png
        assert not os.path.exists(os.path.join(d, "work"))
        meta = _meta(d)
        assert meta["work_cleaned"] and meta["mask_saved"] is False and meta["status"] == "completed"
        q = json.load(open(os.path.join(d, "qc.json")))
        assert q["mask_source"] == "image>0" and q["dice_mask_template"] > 0.85
    n = len(fake_tools.calls())
    assert _run(config_file, man, out) == 0
    assert len(fake_tools.calls()) == n  # resume after cleanup: nothing redone
    # SynthSeg batch after work/ is gone: needs only image.nii.gz, cleans its own raw output.
    assert main(["synthseg", "--config", config_file, "--manifest", man, "--out-root", out]) == 0
    for r in rows:
        d = _sd(out, r)
        assert os.path.isfile(os.path.join(d, "labels_synthseg.nii.gz"))
        assert not os.path.exists(os.path.join(d, "work"))
        meta = _meta(d)
        assert meta["steps"]["synthseg"]["status"] == "done" and meta["work_cleaned"]
        assert json.load(open(os.path.join(d, "qc.json")))["dice_mask_synthseg"] > 0.9
    n = len(fake_tools.calls())
    assert _run(config_file, man, out) == 0
    assert main(["synthseg", "--config", config_file, "--manifest", man, "--out-root", out]) == 0
    assert len(fake_tools.calls()) == n
    # Rerunning a single step that needs work/ explains how to recover.
    assert _run(config_file, man, out, "--rows", "1", "--steps", "normalize", "--overwrite") == 1
    assert "work/ was deleted" in _meta(_sd(out, rows[1]))["error"]


def test_saved_mask_png_and_kept_work(tmp_path, fake_tools, target_dir, mni_files):
    cfg = write_config(tmp_path / "c.yaml", target_dir, mni_files,
                       outputs={"keep_work": True, "save_mask": True, "save_qc_png": True})
    row = ds.lumir_row(str(tmp_path / "in"))
    man = ds.write_rows(str(tmp_path / "m.csv"), [row])
    out = str(tmp_path / "out")
    assert _run(cfg, man, out) == 0
    d = _sd(out, row)
    _check_outputs(d, ["lumir10"], saved=True)
    assert os.path.isfile(os.path.join(d, "work", "stripped.nii.gz"))
    assert json.load(open(os.path.join(d, "qc.json")))["mask_source"] == "mask.nii.gz"
    # Switching save_mask off and rerunning normalize removes the now-stale mask.
    cfg2 = write_config(tmp_path / "c2.yaml", target_dir, mni_files, outputs={"keep_work": True})
    assert _run(cfg2, man, out, "--steps", "normalize,qc", "--overwrite") == 0
    assert not os.path.exists(os.path.join(d, "mask.nii.gz"))
    assert json.load(open(os.path.join(d, "qc.json")))["mask_source"] == "image>0"


def _write_meta(d, meta):
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)


def test_interrupted_step_with_outputs_reruns(tmp_path, fake_tools, target_dir, mni_files):
    config_file = write_config(tmp_path / "keep.yaml", target_dir, mni_files, outputs={"keep_work": True})
    row = ds.ultracortex_row(str(tmp_path / "in"))
    man = ds.write_rows(str(tmp_path / "m.csv"), [row])
    out = str(tmp_path / "out")
    assert _run(config_file, man, out) == 0
    d = _sd(out, row)
    n = len(fake_tools.calls())
    # Simulate a SLURM kill during N4: truncated output, step still marked "running".
    with open(os.path.join(d, "work", "n4.nii.gz"), "wb") as f:
        f.write(b"\x1f\x8b trunc")
    meta = _meta(d)
    meta["steps"]["n4"]["status"] = "running"
    meta["status"] = "running"
    _write_meta(d, meta)
    assert _run(config_file, man, out) == 0
    assert [c["tool"] for c in fake_tools.calls()[n:]] == ["N4BiasFieldCorrection"]
    assert _meta(d)["steps"]["n4"]["status"] == "done" and _meta(d)["status"] == "completed"
    nifti.load_data(os.path.join(d, "work", "n4.nii.gz"))  # rewritten, readable
    # Outputs present but no step record at all (e.g. meta.json lost the entry) -> rerun too.
    n = len(fake_tools.calls())
    meta = _meta(d)
    del meta["steps"]["strip"]
    _write_meta(d, meta)
    assert _run(config_file, man, out) == 0
    assert [c["tool"] for c in fake_tools.calls()[n:]] == ["mri_synthstrip"]


def test_failed_step_marked_failed_in_meta(tmp_path, fake_tools, config_file, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "N4BiasFieldCorrection")
    row = ds.ultracortex_row(str(tmp_path / "in"))
    man = ds.write_rows(str(tmp_path / "m.csv"), [row])
    out = str(tmp_path / "out")
    assert _run(config_file, man, out) == 1
    rec = _meta(_sd(out, row))["steps"]["n4"]
    assert rec["status"] == "failed" and rec["commands"][0][0].endswith("N4BiasFieldCorrection")


def test_meta_step_records_commands(tmp_path, fake_tools, config_file):
    row = ds.ultracortex_row(str(tmp_path / "in"))
    man = ds.write_rows(str(tmp_path / "m.csv"), [row])
    out = str(tmp_path / "out")
    assert _run(config_file, man, out) == 0
    steps = _meta(_sd(out, row))["steps"]
    calls = fake_tools.calls()
    by_tool = {}
    for c in calls:
        by_tool.setdefault(c["tool"], []).append(c["argv"])
    assert [cmd[1:] for cmd in steps["strip"]["commands"]] == by_tool["mri_synthstrip"]
    assert [cmd[1:] for cmd in steps["n4"]["commands"]] == by_tool["N4BiasFieldCorrection"]
    assert len(steps["register"]["commands"]) == 1 and len(steps["resample"]["commands"]) == 2
    assert all(os.path.basename(cmd[0]) == "flirt" for cmd in steps["resample"]["commands"])
    assert steps["normalize"]["commands"] == []  # pure numpy


def test_ants_use_masks_without_template_mask_warns(tmp_path, fake_tools, target_dir, mni_files, caplog):
    cfg = write_config(tmp_path / "ants.yaml", target_dir, mni_files,
                       registration={"backend": "ants", "ants": {"use_masks": True}},
                       mni_templates={"T1": {"image_1mm": mni_files.image_1mm, "mask_1mm": None,
                                             "image_2mm": mni_files.image_2mm}})
    row = ds.generic_row(str(tmp_path / "in"))
    man = ds.write_rows(str(tmp_path / "m.csv"), [row])
    out = str(tmp_path / "out")
    with caplog.at_level("WARNING", logger="preprocessing"):
        assert _run(cfg, man, out) == 0
    assert "use_masks is true" in caplog.text
    reg = [c["argv"] for c in fake_tools.calls() if c["tool"] == "antsRegistration"][0]
    assert "-x" not in reg
    assert _meta(_sd(out, row))["registration"]["masks_used"] is False


def test_failure_is_logged_and_next_row_runs(tmp_path, fake_tools, config_file, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "mri_synthstrip")
    rows = [ds.ultracortex_row(str(tmp_path / "in")), ds.neurite_row(str(tmp_path / "in"))]
    man = ds.write_rows(str(tmp_path / "m.csv"), rows)
    out = str(tmp_path / "out")
    assert _run(config_file, man, out) == 1
    d0, d1 = _sd(out, rows[0]), _sd(out, rows[1])
    m0 = _meta(d0)
    assert m0["status"] == "failed" and "exited with 3" in m0["error"]
    assert "fake failure" in open(os.path.join(d0, "logs", "strip.log")).read()
    assert os.path.isfile(os.path.join(d0, "logs", "pipeline.log"))
    assert _meta(d1)["status"] == "completed"
    # --fail-fast stops after the first failure.
    out2 = str(tmp_path / "out2")
    assert _run(config_file, man, out2, "--fail-fast") == 1
    assert not os.path.exists(_sd(out2, rows[1]))


def test_steps_subset_and_missing_inputs(tmp_path, fake_tools, config_file):
    row = ds.ultracortex_row(str(tmp_path / "in"))
    man = ds.write_rows(str(tmp_path / "m.csv"), [row])
    out = str(tmp_path / "out")
    assert _run(config_file, man, out, "--steps", "reorient,strip") == 0
    assert ds.calls_by_tool(fake_tools.calls()) == {"mri_synthstrip": 1}
    d = _sd(out, row)
    assert not os.path.isfile(os.path.join(d, "image.nii.gz"))
    assert _meta(d)["status"] == "partial"
    assert os.path.isfile(os.path.join(d, "work", "stripped.nii.gz"))  # partial rows keep work/
    assert _run(config_file, man, out, "--steps", "register") == 1  # n4 output missing
    assert "missing inputs" in _meta(d)["error"]
    with pytest.raises(SystemExit):
        _run(config_file, man, out, "--steps", "bogus")
