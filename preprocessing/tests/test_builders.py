import argparse

import numpy as np

from preprocessing.builders import BUILDERS, apply_exclude
from preprocessing.builders.bids import parse_bids_name

from . import phantoms as ph


def _nii(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ph.save(str(path), np.zeros((2, 2, 2)), np.eye(4))


def _args(dataset, argv=()):
    p = argparse.ArgumentParser(conflict_handler="resolve")
    for mod in BUILDERS.values():
        mod.add_arguments(p.add_argument_group())
    return p.parse_args(list(argv))


def test_lumir(tmp_path):
    for d, ids in (("imagesTr", ("0001", "0002")), ("imagesVal", ("3454",))):
        for i in ids:
            _nii(tmp_path / d / ("LUMIRMRI_%s_0000.nii.gz" % i))
    _nii(tmp_path / "labelsVal" / "LUMIRMRI_3454.nii.gz")
    rows = BUILDERS["lumir"].build(str(tmp_path), _args("lumir", ["--label-glob", "labelsVal/{id}.nii.gz"]))
    assert [r.subject for r in rows] == ["LUMIRMRI_0001", "LUMIRMRI_0002", "LUMIRMRI_3454"]
    assert [r.split for r in rows] == ["train", "train", "val"]
    assert all(r.skull_stripped and r.in_template_space and r.bias_corrected for r in rows)
    assert list(rows[2].label_paths) == ["lumir10"] and not rows[0].label_paths
    assert all(r.contrast == "T1w" for r in rows)


def test_lumir_extra_channel_contrast_subject_and_labels(tmp_path):
    from preprocessing.config import contrast_key, load_config
    from preprocessing.manifest import subject_dir_name
    for f in ("LUMIRMRI_3454_0000", "LUMIRMRI_4033_0001", "LUMIRMRI_4033_0000", "LUMIRMRI_4040_0001"):
        _nii(tmp_path / "imagesVal" / ("%s.nii.gz" % f))
    _nii(tmp_path / "imagesVal" / "other_0001.nii.gz")          # not LUMIR naming
    (tmp_path / "imagesVal" / "LUMIRMRI_5000_0000.json").write_text("{}")  # not NIfTI
    for f in ("LUMIRMRI_3454_0000", "LUMIRMRI_4033_0001", "LUMIRMRI_4040_0001"):
        _nii(tmp_path / "simplelabelsVal" / ("%s.nii.gz" % f))
    rows = BUILDERS["lumir"].build(str(tmp_path), _args("lumir", ["--label-glob", "simplelabelsVal/{id}*.nii.gz"]))
    by = {r.subject: r for r in rows}
    assert len(rows) == 4 and all("/imagesVal/" in r.image_path for r in rows)  # label folder not scanned
    assert sorted(by) == ["LUMIRMRI_3454", "LUMIRMRI_4033", "LUMIRMRI_4033_0001", "LUMIRMRI_4040_0001"]
    assert by["LUMIRMRI_3454"].contrast == "T1w" and by["LUMIRMRI_4033"].contrast == "T1w"
    assert by["LUMIRMRI_4033_0001"].contrast == "unknown0001"
    assert subject_dir_name(by["LUMIRMRI_4033_0001"]) == "LUMIRMRI_4033_0001_unknown0001"
    assert by["LUMIRMRI_3454"].label_paths["lumir10"].endswith("simplelabelsVal/LUMIRMRI_3454_0000.nii.gz")
    assert by["LUMIRMRI_4033_0001"].label_paths["lumir10"].endswith("simplelabelsVal/LUMIRMRI_4033_0001.nii.gz")
    assert not by["LUMIRMRI_4033"].label_paths       # never the _0001 channel's label
    assert all(r.in_template_space and r.split == "val" for r in rows)
    # {name} = image stem placeholder works as well.
    rows = BUILDERS["lumir"].build(str(tmp_path), _args("lumir", ["--label-glob", "simplelabelsVal/{name}.nii.gz"]))
    assert sum(1 for r in rows if r.label_paths) == 3
    # The neutral contrast tag maps to the default template key (never used: LUMIR skips registration).
    assert contrast_key(load_config(None), "unknown0001") == "T1"


def test_neurite_oasis_ignores_aligned(tmp_path):
    for s in ("OASIS_OAS1_0001_MR1", "OASIS_OAS1_0002_MR1"):
        for f in ("orig", "norm", "seg35", "seg4", "aligned_norm", "aligned_seg35", "slice_norm"):
            _nii(tmp_path / s / ("%s.nii.gz" % f))
    (tmp_path / "OASIS_OAS1_0002_MR1" / "seg4.nii.gz").unlink()
    rows = BUILDERS["neurite_oasis"].build(str(tmp_path), _args("neurite_oasis"))
    assert [r.subject for r in rows] == ["OASIS_OAS1_0001_MR1", "OASIS_OAS1_0002_MR1"]
    assert rows[0].image_path.endswith("/OASIS_OAS1_0001_MR1/norm.nii.gz")
    assert list(rows[0].label_paths) == ["seg35", "seg4"] and list(rows[1].label_paths) == ["seg35"]
    assert all("aligned" not in p for r in rows for p in [r.image_path] + list(r.label_paths.values()))
    assert rows[0].skull_stripped and rows[0].bias_corrected and not rows[0].in_template_space
    orig = BUILDERS["neurite_oasis"].build(str(tmp_path), _args("neurite_oasis", ["--image", "orig"]))
    assert orig[0].image_path.endswith("orig.nii.gz") and not orig[0].skull_stripped
    # exclude list
    ex = tmp_path / "ex.txt"
    ex.write_text("# leakage\nOASIS_OAS1_0002_MR1\nNOT_THERE\n")
    assert [r.subject for r in apply_exclude(rows, str(ex))] == ["OASIS_OAS1_0001_MR1"]


def test_ultracortex_bids(tmp_path):
    _nii(tmp_path / "sub-01" / "anat" / "sub-01_T1w.nii.gz")
    _nii(tmp_path / "sub-02" / "ses-1" / "anat" / "sub-02_ses-1_T1w.nii.gz")
    _nii(tmp_path / "sub-02" / "ses-2" / "anat" / "sub-02_ses-2_T1w.nii.gz")
    _nii(tmp_path / "sub-03" / "anat" / "sub-03_acq-mp2rage_T1w.nii.gz")
    d = tmp_path / "derivatives"
    _nii(d / "synthstrip" / "sub-01" / "anat" / "sub-01_desc-brain_mask.nii.gz")
    _nii(d / "synthstrip" / "sub-01" / "anat" / "sub-01_desc-brain_T1w.nii.gz")      # stripped image, ignored
    _nii(d / "synthstrip" / "sub-02" / "ses-2" / "anat" / "sub-02_ses-2_desc-brain_mask.nii.gz")
    _nii(d / "manual_segmentations" / "sub-02" / "ses-1" / "anat" / "sub-02_ses-1_desc-manual_dseg.nii.gz")
    _nii(d / "freesurfer" / "sub-01" / "ribbon.nii.gz")                              # no BIDS entities
    rows = BUILDERS["ultracortex"].build(str(tmp_path), _args("ultracortex"))
    key = {(r.subject, r.session): r for r in rows}
    assert len(rows) == 4
    assert key[("sub-01", "")].mask_path.endswith("sub-01_desc-brain_mask.nii.gz")
    assert key[("sub-02", "1")].mask_path == "" and list(key[("sub-02", "1")].label_paths) == ["manual"]
    assert key[("sub-02", "2")].mask_path.endswith("ses-2_desc-brain_mask.nii.gz")
    assert key[("sub-03", "")].contrast == "acq-mp2rage_T1w"
    assert all(not r.skull_stripped and not r.bias_corrected for r in rows)
    rows = BUILDERS["ultracortex"].build(str(tmp_path), _args("ultracortex", ["--ignore-provided-masks"]))
    assert all(r.mask_path == "" for r in rows)


def test_generic_bids_and_folder(tmp_path):
    b = tmp_path / "bids"
    b.mkdir()
    (b / "dataset_description.json").write_text("{}")
    _nii(b / "sub-0001" / "ses-d0100" / "anat" / "sub-0001_ses-d0100_T2w.nii.gz")
    _nii(b / "sub-0001" / "ses-d0100" / "anat" / "sub-0001_ses-d0100_T1w.nii.gz")
    rows = BUILDERS["generic"].build(str(b), _args("generic", ["--name", "oasis3", "--suffix", "T2w"]))
    assert len(rows) == 1 and rows[0].dataset == "oasis3" and rows[0].contrast == "T2w"
    assert rows[0].session == "d0100" and rows[0].subject == "sub-0001"
    m = tmp_path / "mb"
    for s in ("OASIS-TRT-20-1", "OASIS-TRT-20-2"):
        _nii(m / "OASIS-TRT-20_volumes" / s / "t1weighted.nii.gz")
        _nii(m / "OASIS-TRT-20_volumes" / s / "labels.DKT31.manual+aseg.nii.gz")
    rows = BUILDERS["generic"].build(str(m), _args("generic", [
        "--name", "mindboggle", "--image-glob", "*/*/t1weighted.nii.gz",
        "--label", "manual=*/{subject}/labels.DKT31.manual+aseg.nii.gz"]))
    assert [r.subject for r in rows] == ["OASIS-TRT-20-1", "OASIS-TRT-20-2"]
    assert rows[0].label_paths["manual"].endswith("OASIS-TRT-20-1/labels.DKT31.manual+aseg.nii.gz")
    assert not rows[0].skull_stripped and rows[0].contrast == "T1w"


def test_generic_mindboggle_mni152_files(tmp_path):
    """The exact Mindboggle-101 MNI152 command: the globs must not pick the sibling files."""
    m = tmp_path / "data_mindboggle"
    names = ("t1weighted.nii.gz", "t1weighted_brain.nii.gz", "t1weighted.MNI152.nii.gz",
             "t1weighted_brain.MNI152.nii.gz", "labels.DKT31.manual.nii.gz", "labels.DKT31.manual.MNI152.nii.gz",
             "labels.DKT31.manual+aseg.nii.gz", "labels.DKT31.manual+aseg.MNI152.nii.gz")
    for grp, s in (("Extra-18_volumes", "Colin27-1"), ("MMRR-21_volumes", "MMRR-21-1"),
                   ("OASIS-TRT-20_volumes", "OASIS-TRT-20-1")):
        for n in names:
            _nii(m / grp / s / n)
    rows = BUILDERS["generic"].build(str(m), _args("generic", [
        "--name", "mindboggle", "--image-glob", "*/*/t1weighted.MNI152.nii.gz",
        "--label", "dkt31aseg=*/{subject}/labels.DKT31.manual+aseg.MNI152.nii.gz", "--in-template-space"]))
    assert [r.subject for r in rows] == ["Colin27-1", "MMRR-21-1", "OASIS-TRT-20-1"]
    for r in rows:
        assert r.image_path.endswith(r.subject + "/t1weighted.MNI152.nii.gz")
        assert list(r.label_paths) == ["dkt31aseg"]
        assert r.label_paths["dkt31aseg"].endswith(r.subject + "/labels.DKT31.manual+aseg.MNI152.nii.gz")
        assert r.in_template_space and not r.skull_stripped and not r.bias_corrected and r.dataset == "mindboggle"


def _mindboggle_tree(m):
    names = ("t1weighted.nii.gz", "t1weighted_brain.nii.gz", "t1weighted.MNI152.nii.gz",
             "labels.DKT31.manual.MNI152.nii.gz", "labels.DKT31.manual+aseg.nii.gz",
             "labels.DKT31.manual+aseg.MNI152.nii.gz")
    for grp, s in (("Extra-18_volumes", "Colin27-1"), ("Extra-18_volumes", "HLN-12-1"),
                   ("Extra-18_volumes", "HLN-12-10"), ("OASIS-TRT-20_volumes", "OASIS-TRT-20-1")):
        for n in names:
            _nii(m / grp / s / n)
    (m / ".DS_Store").write_text("")


def test_mindboggle_builder_hln_native(tmp_path):
    m = tmp_path / "data_mindboggle"
    _mindboggle_tree(m)
    rows = BUILDERS["mindboggle"].build(str(m), _args("mindboggle"))
    by = {r.subject: r for r in rows}
    assert sorted(by) == ["Colin27-1", "HLN-12-1", "HLN-12-10", "OASIS-TRT-20-1"]
    for s in ("HLN-12-1", "HLN-12-10"):
        r = by[s]
        assert not r.in_template_space and r.image_path.endswith(s + "/t1weighted.nii.gz")
        assert r.label_paths["dkt31aseg"].endswith(s + "/labels.DKT31.manual+aseg.nii.gz")
    for s in ("Colin27-1", "OASIS-TRT-20-1"):
        r = by[s]
        assert r.in_template_space and r.image_path.endswith(s + "/t1weighted.MNI152.nii.gz")
        assert r.label_paths["dkt31aseg"].endswith(s + "/labels.DKT31.manual+aseg.MNI152.nii.gz")
    assert all(r.dataset == "mindboggle" and not r.skull_stripped and not r.bias_corrected for r in rows)
    native = BUILDERS["mindboggle"].build(str(m), _args("mindboggle", ["--mb-select", "native"]))
    assert [r.subject for r in native] == ["HLN-12-1", "HLN-12-10"]
    mni = BUILDERS["mindboggle"].build(str(m), _args("mindboggle", ["--mb-select", "mni"]))
    assert [r.subject for r in mni] == ["Colin27-1", "OASIS-TRT-20-1"]
    custom = BUILDERS["mindboggle"].build(str(m), _args("mindboggle", ["--mb-native", "Colin27-*",
                                                                       "--mb-select", "native"]))
    assert [r.subject for r in custom] == ["Colin27-1"]


def test_generic_globs_use_raw_folder_name(tmp_path):
    m = tmp_path / "raw"
    _nii(m / "subj 01 (scan)" / "t1.nii.gz")
    _nii(m / "subj 01 (scan)" / "brainmask.nii.gz")
    _nii(m / "subj 01 (scan)" / "aseg.nii.gz")
    rows = BUILDERS["generic"].build(str(m), _args("generic", [
        "--image-glob", "*/t1.nii.gz", "--mask-glob", "{subject}/brainmask.nii.gz",
        "--label", "aseg={subject}/aseg.nii.gz"]))
    assert len(rows) == 1 and rows[0].subject == "subj_01__scan_"
    assert rows[0].mask_path.endswith("brainmask.nii.gz")
    assert rows[0].label_paths["aseg"].endswith("aseg.nii.gz")


def test_parse_bids_name():
    assert parse_bids_name("sub-01_ses-2_acq-x_run-1_T1w.nii.gz") == (
        {"sub": "01", "ses": "2", "acq": "x", "run": "1"}, "T1w")
