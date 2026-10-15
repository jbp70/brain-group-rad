from collections import OrderedDict

import pytest

from preprocessing.manifest import (ManifestRow, check_unique, format_label_paths, parse_bool,
                                    parse_label_paths, read_manifest, subject_dir_name, write_manifest)


def _row(**kw):
    base = dict(dataset="ds", subject="s1", image_path="/a/b.nii.gz")
    base.update(kw)
    return ManifestRow(**base)


def test_roundtrip(tmp_path):
    rows = [_row(session="2", contrast="T1w", split="val", mask_path="/m.nii.gz",
                 label_paths=OrderedDict([("manual", "/x/C:y.nii.gz"), ("seg35", "/s.nii.gz")]),
                 skull_stripped=True, bias_corrected=True),
            _row(subject="s2")]
    p = str(tmp_path / "m.csv")
    write_manifest(rows, p)
    back = read_manifest(p)
    assert [r.to_dict() for r in back] == [r.to_dict() for r in rows]
    header = open(p).readline().strip().split(",")
    assert header[:6] == ["dataset", "subject", "session", "contrast", "split", "image_path"]


def test_label_paths_parsing():
    d = parse_label_paths("manual:/p/a:b.nii.gz; seg4:/q.nii.gz")
    assert d == OrderedDict([("manual", "/p/a:b.nii.gz"), ("seg4", "/q.nii.gz")])
    assert parse_label_paths("") == OrderedDict()
    with pytest.raises(ValueError):
        format_label_paths({"x": "/a;b"})
    with pytest.raises(ValueError):
        parse_label_paths("bad name:/x")
    with pytest.raises(ValueError):
        parse_label_paths("a:/x;a:/y")


def test_bool_and_unique_and_dirname():
    assert parse_bool("TRUE") and parse_bool("1") and not parse_bool("") and not parse_bool("false")
    with pytest.raises(ValueError):
        parse_bool("maybe")
    with pytest.raises(ValueError):
        check_unique([_row(), _row()])
    assert subject_dir_name(_row(session="01", contrast="T1w")) == "s1_ses-01_T1w"
    assert subject_dir_name(_row(contrast="T1w")) == "s1_T1w"
    assert subject_dir_name(_row()) == "s1"
    with pytest.raises(ValueError):
        write_manifest([_row(subject="a/b")], "/tmp/never.csv")
