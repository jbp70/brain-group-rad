"""export-add-labels (synth-contrast design 3.3, 7.1; review NIT 2 range rule = OD2)."""
import csv
import glob
import json
import os

import numpy as np
import pytest

from preprocessing import add_labels as al
from preprocessing.cli import main
from preprocessing.export import block_mode, export_npy, merge_parts
from preprocessing.luts import synthfine_to_train18_table
from preprocessing.qc import summarize

from .test_export import _subject

DS = "lumir"
TABLE = synthfine_to_train18_table()
SFX = {1: "", 2: "_2", 4: "_4"}
SF_FILES = ["labels_synthfine.npy", "labels_synthfine_2.npy", "labels_synthfine_4.npy"]


def _tree(tmp_path, n=3):
    """Processed tree with n LUMIR-like cases (all 103 SynthSeg ids, cortex-heavy), exported train18-only
    in TWO chunks ([0, 2) and [2, n)) -> two parts -> merged manifest."""
    out = str(tmp_path / "processed")
    for i in range(n):
        _subject(out, "s%d" % i, ds=DS, seed=i)
    summarize(os.path.join(out, DS))
    exp = str(tmp_path / "npy_v2")
    export_npy(out, exp, datasets=[DS], start=0, end=2)
    export_npy(out, exp, datasets=[DS], start=2, end=n)
    merge_parts(exp)
    return out, exp


def _case_dir(exp, i):
    return os.path.join(exp, DS, "s%d" % i)


def _snapshot(root):
    """{relative path: (bytes, mtime_ns)} of every file under root."""
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if os.path.isfile(p):
            with open(p, "rb") as f:
                out[os.path.relpath(p, root)] = (f.read(), os.stat(p).st_mtime_ns)
    return out


def _bytes(snap):
    return {k: v[0] for k, v in snap.items()}


def _manifest_rows(exp):
    with open(os.path.join(exp, "export_manifest.csv"), newline="") as f:
        return list(csv.DictReader(f))


def _status(exp, name):
    with open(os.path.join(exp, "add_labels", "synthfine", name)) as f:
        return json.load(f)


def _files(exp, start=None, end=None, out=None):
    return al.add_labels_files(out, exp, "synthfine", start, end)


def _manifest_lookup(rows, k, label_set="train18"):
    """Same rule as data_loading.data_module.manifest_files(kind='labels') (that module imports TF)."""
    out = []
    for r in rows:
        want = "labels_%s%s.npy" % (label_set, SFX[k])
        if want not in [v for v in r["labels_l%d" % k].split(";") if v]:
            raise ValueError("case %s/%s has no %s" % (r["dataset"], r["subject_dir"], want))
        out.append("/".join([r["dataset"], r["subject_dir"], want]))
    return out


# ----------------------------------------------------------------- phase files

def test_files_phase_writes_only_synthfine_consistent_and_rerun_kept(tmp_path):
    out, exp = _tree(tmp_path)
    before = _snapshot(exp)
    st, rc = _files(exp, out=out)
    assert rc == 0 and st["counts"] == {"written": 3, "kept": 0, "failed": 0}
    after = _snapshot(exp)
    # Every pre-existing file keeps its bytes AND mtime; the only new files are the synthfine npys + status.
    for k, v in before.items():
        assert after[k] == v, k
    new = sorted(set(after) - set(before))
    assert new == sorted([os.path.join("add_labels", "synthfine", "rows-0-end.json")]
                         + [os.path.join(DS, "s%d" % i, f) for i in range(3) for f in SF_FILES])
    for i in range(3):
        d = _case_dir(exp, i)
        for k in (1, 2, 4):
            f = np.load(os.path.join(d, "labels_synthfine%s.npy" % SFX[k]))
            t = np.load(os.path.join(d, "labels_train18%s.npy" % SFX[k]))
            assert f.dtype == np.uint8 and f.shape == t.shape and f.max() <= 102
            np.testing.assert_array_equal(TABLE[f], t)
        # The nested vote really differs from the plain fine block mode (cortex split over parcels).
        full = np.load(os.path.join(d, "labels_synthfine.npy"))
        assert (np.load(os.path.join(d, "labels_synthfine_2.npy")) != block_mode(full, 2)).any()
    snap = _snapshot(exp)
    st, rc = _files(exp, out=out)
    assert rc == 0 and st["counts"] == {"written": 0, "kept": 3, "failed": 0}
    assert all(c["status"] == "kept" and c["levels"] == [1, 2, 4] for c in st["cases"])
    after2 = _snapshot(exp)
    for k, v in snap.items():
        if not k.startswith("add_labels"):
            assert after2[k] == v, k


def test_files_phase_failures_per_case(tmp_path, capsys):
    out, exp = _tree(tmp_path, 4)
    # s0: SynthSeg map differs from the one used for train18 (reprocessed case).
    _subject(out, "s0", ds=DS, seed=99)
    # s1: pre-existing wrong labels_synthfine_2.npy (right dtype/shape, wrong content).
    wrong = np.zeros((4, 6, 8), np.uint8)
    np.save(os.path.join(_case_dir(exp, 1), "labels_synthfine_2.npy"), wrong)
    w_bytes = open(os.path.join(_case_dir(exp, 1), "labels_synthfine_2.npy"), "rb").read()
    # s2: wrong dtype file.
    np.save(os.path.join(_case_dir(exp, 2), "labels_synthfine_4.npy"), np.zeros((2, 3, 4), np.uint16))
    # s3: missing source.
    os.remove(os.path.join(out, DS, "s3", "labels_synthseg.nii.gz"))
    capsys.readouterr()
    st, rc = _files(exp, out=out)
    assert rc == 1 and st["counts"]["failed"] == 4
    by = {c["case"]: c for c in st["cases"]}
    assert "voxels differ" in by["lumir/s0"]["reason"] and "level 1" in by["lumir/s0"]["reason"]
    assert "differs from the recomputed" in by["lumir/s1"]["reason"]
    assert "uint16" in by["lumir/s2"]["reason"]
    assert "no source" in by["lumir/s3"]["reason"]
    printed = capsys.readouterr().out
    assert all("FAILED lumir/s%d" % i in printed for i in range(4))
    # Nothing written for any failed case; the wrong file is left alone.
    assert not glob.glob(os.path.join(_case_dir(exp, 0), "labels_synthfine*"))
    assert sorted(os.path.basename(p) for p in glob.glob(os.path.join(_case_dir(exp, 1), "labels_synthfine*"))) == [
        "labels_synthfine_2.npy"]
    assert open(os.path.join(_case_dir(exp, 1), "labels_synthfine_2.npy"), "rb").read() == w_bytes
    assert sorted(os.path.basename(p) for p in glob.glob(os.path.join(_case_dir(exp, 2), "labels_synthfine*"))) == [
        "labels_synthfine_4.npy"]
    # Other cases continue: fix s3's source -> only s3 is written in a rerun of its row.
    _subject(out, "s3", ds=DS, seed=3)
    st, rc = _files(exp, 3, 4, out=out)
    assert rc == 0 and st["counts"]["written"] == 1
    assert main(["export-add-labels", "--out-root", out, "--export-root", exp, "--phase", "files"]) == 1


def test_range_rule(tmp_path, capsys):
    out, exp = _tree(tmp_path)
    # End beyond the rows is clipped (last chunk shorter); file named after the requested range.
    assert main(["export-add-labels", "--out-root", out, "--export-root", exp, "--phase", "files",
                 "--start", "2", "--end", "500"]) == 0
    s = _status(exp, "rows-2-500.json")
    assert s["rows"] == [2, 3] and [c["case"] for c in s["cases"]] == ["lumir/s2"]
    assert main(["export-add-labels", "--out-root", out, "--export-root", exp, "--phase", "files",
                 "--start", "0", "--end", "2"]) == 0
    assert [c["case"] for c in _status(exp, "rows-0-2.json")["cases"]] == ["lumir/s0", "lumir/s1"]
    # start >= rows: empty selection, 0-case status file, exit 0.
    assert main(["export-add-labels", "--out-root", out, "--export-root", exp, "--phase", "files",
                 "--start", "3", "--end", "500"]) == 0
    s = _status(exp, "rows-3-500.json")
    assert s["cases"] == [] and s["counts"] == {"written": 0, "kept": 0, "failed": 0}
    assert main(["export-add-labels", "--out-root", out, "--export-root", exp, "--phase", "files",
                 "--start", "1000"]) == 0
    assert _status(exp, "rows-1000-end.json")["cases"] == []
    # Negative values or start > end: exit 2, no status file.
    for args in (["--start", "-1"], ["--end", "-1"], ["--start", "2", "--end", "1"]):
        assert main(["export-add-labels", "--out-root", out, "--export-root", exp, "--phase", "files"] + args) == 2
    assert not os.path.exists(os.path.join(exp, "add_labels", "synthfine", "rows-2-1.json"))
    assert al.resolve_range(3423, 3000, 3500) == (3000, 3423)
    assert al.resolve_range(3423, None, None) == (0, 3423)
    assert al.resolve_range(10, 20, 30) == (10, 10) and al.resolve_range(10, 20, None) == (10, 10)
    with pytest.raises(ValueError):
        al.resolve_range(10, 5, 4)


def test_usage_errors_exit_2(tmp_path):
    out, exp = _tree(tmp_path, 1)
    before = _bytes(_snapshot(exp))
    assert main(["export-add-labels", "--out-root", out, "--export-root", exp, "--phase", "files",
                 "--label-set", "synthseg"]) == 2
    assert main(["export-add-labels", "--export-root", exp, "--phase", "manifest", "--start", "0"]) == 2
    assert main(["export-add-labels", "--export-root", exp, "--phase", "manifest", "--label-set", "x"]) == 2
    assert main(["export-add-labels", "--export-root", exp, "--phase", "files"]) == 2  # no --out-root
    assert _bytes(_snapshot(exp)) == before


# ----------------------------------------------------------------- phase manifest

def test_manifest_phase(tmp_path, capsys):
    out, exp = _tree(tmp_path)
    assert _files(exp, out=out)[1] == 0
    old_rows = _manifest_rows(exp)
    old_text = open(os.path.join(exp, "export_manifest.csv"), "rb").read()
    old_luts = json.load(open(os.path.join(exp, "label_luts.json")))
    assert main(["export-add-labels", "--export-root", exp, "--phase", "manifest"]) == 0
    new_rows = _manifest_rows(exp)
    assert [(r["dataset"], r["subject_dir"]) for r in new_rows] == [(r["dataset"], r["subject_dir"]) for r in old_rows]
    assert list(new_rows[0]) == list(old_rows[0])
    for o, n in zip(old_rows, new_rows):
        for c in o:
            if c in al.LABEL_COLUMNS:
                assert n[c].startswith(o[c] + ";") and n[c] != o[c], c
            else:
                assert n[c] == o[c], c
        assert n["labels"] == "train18;synthfine" and n["label_dtypes"] == "train18:uint8;synthfine:uint8"
        assert n["label_levels"] == "train18:1,2,4;synthfine:1,2,4"
        assert n["labels_l4"] == "labels_train18_4.npy;labels_synthfine_4.npy"
    luts = json.load(open(os.path.join(exp, "label_luts.json")))
    assert sorted(luts["luts"]) == ["synthfine", "train18"] and luts["luts"]["train18"] == old_luts["luts"]["train18"]
    # Backup with the three items, byte-identical to the old state.
    b = glob.glob(os.path.join(exp, "backup_pre_synthfine_*"))
    assert len(b) == 1 and sorted(os.listdir(b[0])) == ["export_manifest.csv", "label_luts.json", "manifest_parts"]
    assert open(os.path.join(b[0], "export_manifest.csv"), "rb").read() == old_text
    assert len(glob.glob(os.path.join(b[0], "manifest_parts", "*.csv"))) == 2
    diff = json.load(open(os.path.join(exp, "add_labels", "synthfine", "manifest_diff.json")))
    assert diff["problems"] == [] and diff["rows_changed"] == 3 and diff["columns_changed"] == sorted(al.LABEL_COLUMNS)
    # Name lookups: synthfine and the default train18 both resolve.
    assert _manifest_lookup(new_rows, 4, "synthfine") == ["lumir/s%d/labels_synthfine_4.npy" % i for i in range(3)]
    assert _manifest_lookup(new_rows, 2) == ["lumir/s%d/labels_train18_2.npy" % i for i in range(3)]
    with pytest.raises(ValueError):
        _manifest_lookup(old_rows, 1, "synthfine")
    # export-merge afterwards reproduces the new manifest byte for byte.
    new_text = open(os.path.join(exp, "export_manifest.csv"), "rb").read()
    assert main(["export-merge", "--export-root", exp]) == 0
    assert open(os.path.join(exp, "export_manifest.csv"), "rb").read() == new_text
    # export-check passes with the consistency rule.
    capsys.readouterr()
    assert main(["export-check", "--export-root", exp, "--sample", "0"]) == 0
    assert "checked 3 cases: OK" in capsys.readouterr().out
    # Idempotent: a second run changes nothing and makes no new backup.
    snap = _bytes(_snapshot(exp))
    assert main(["export-add-labels", "--export-root", exp, "--phase", "manifest"]) == 0
    assert "nothing to do" in capsys.readouterr().out
    assert _bytes(_snapshot(exp)) == snap
    assert al.add_set(new_rows[0], "synthfine") == new_rows[0]
    assert al.add_set({"labels": "", "label_dtypes": "", "label_levels": "", "levels": "1;2",
                       "labels_l1": "", "labels_l2": ""}, "synthfine") == {
        "labels": "synthfine", "label_dtypes": "synthfine:uint8", "label_levels": "synthfine:1,2", "levels": "1;2",
        "labels_l1": "labels_synthfine.npy", "labels_l2": "labels_synthfine_2.npy"}


def test_manifest_refuses_while_a_case_lacks_a_file(tmp_path, capsys):
    out, exp = _tree(tmp_path)
    assert _files(exp, out=out)[1] == 0
    os.remove(os.path.join(_case_dir(exp, 1), "labels_synthfine_4.npy"))
    before = _bytes(_snapshot(exp))
    capsys.readouterr()
    assert main(["export-add-labels", "--export-root", exp, "--phase", "manifest"]) == 1
    assert "lumir/s1: labels_synthfine_4.npy missing" in capsys.readouterr().out
    assert _bytes(_snapshot(exp)) == before
    assert not glob.glob(os.path.join(exp, "backup_pre_*"))


def test_manifest_refuses_on_pre_existing_divergence(tmp_path, capsys):
    out, exp = _tree(tmp_path)
    assert _files(exp, out=out)[1] == 0
    mp = os.path.join(exp, "export_manifest.csv")
    text = open(mp, newline="").read()
    old_dir = os.path.join(exp, DS, "s1")
    assert text.count(old_dir + ",") == 1
    with open(mp, "w", newline="") as f:
        f.write(text.replace(old_dir + ",", "/moved/elsewhere/s1,"))
    before = _bytes(_snapshot(exp))
    capsys.readouterr()
    assert main(["export-add-labels", "--export-root", exp, "--phase", "manifest"]) == 1
    printed = capsys.readouterr().out
    assert "pre-existing divergence between manifest_parts and export_manifest.csv" in printed
    assert "DIVERGED lumir/s1" in printed and "lumir/s0" not in printed
    assert _bytes(_snapshot(exp)) == before
    assert not glob.glob(os.path.join(exp, "backup_pre_*"))


def test_manifest_restores_backup_on_late_failure(tmp_path, monkeypatch, capsys):
    out, exp = _tree(tmp_path)
    assert _files(exp, out=out)[1] == 0
    before = _bytes(_snapshot(exp))
    real = al.add_set

    def corrupt(row, label_set):
        new = real(row, label_set)
        new["subject"] = "corrupted"  # a non-label column: parts and manifest agree, the diff check catches it
        return new

    monkeypatch.setattr(al, "add_set", corrupt)
    capsys.readouterr()
    assert main(["export-add-labels", "--export-root", exp, "--phase", "manifest"]) == 1
    printed = capsys.readouterr().out
    assert "non-label column subject changed" in printed and "restored" in printed
    after = _bytes(_snapshot(exp))
    b = glob.glob(os.path.join(exp, "backup_pre_synthfine_*"))
    assert len(b) == 1
    # Every original file restored byte-identical; only the backup and the diff report are new.
    for k, v in before.items():
        assert after[k] == v, k
    extra = sorted(k for k in after if k not in before)
    assert all(k.startswith(os.path.basename(b[0])) or k == os.path.join("add_labels", "synthfine",
                                                                          "manifest_diff.json") for k in extra), extra


def test_old_column_part_kept_in_its_layout(tmp_path):
    """A part written before missing_optional_labels existed keeps its own column list."""
    from preprocessing.export import MANIFEST_COLUMNS, OLD_MANIFEST_COLUMNS
    out, exp = _tree(tmp_path)
    assert _files(exp, out=out)[1] == 0
    part = sorted(glob.glob(os.path.join(exp, "manifest_parts", "*.csv")))[0]
    rows = list(csv.DictReader(open(part, newline="")))
    with open(part, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OLD_MANIFEST_COLUMNS, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    merge_parts(exp)
    assert main(["export-add-labels", "--export-root", exp, "--phase", "manifest"]) == 0
    with open(part, newline="") as f:
        reader = csv.DictReader(f)
        prow = list(reader)
        assert list(reader.fieldnames) == OLD_MANIFEST_COLUMNS
    assert all(r["labels"] == "train18;synthfine" for r in prow)
    assert list(_manifest_rows(exp)[0]) == MANIFEST_COLUMNS
    text = open(os.path.join(exp, "export_manifest.csv"), "rb").read()
    merge_parts(exp)
    assert open(os.path.join(exp, "export_manifest.csv"), "rb").read() == text


def test_constants():
    assert al.LABEL_COLUMNS == ("labels", "label_dtypes", "label_levels", "labels_l1", "labels_l2", "labels_l4")
    assert al.SUPPORTED_SETS == ("synthfine",) and al.STATUS_DIR == "add_labels"
