import csv
import os

import numpy as np
import pytest

from preprocessing.cli import main, select_rows
from preprocessing.manifest import ManifestRow

from . import phantoms as ph

COMMANDS = ["check-tools", "build-manifest", "build-target-grid", "run", "synthseg", "qc-summary",
            "export-npy", "cleanup", "n-chunks"]


@pytest.mark.parametrize("cmd", COMMANDS)
def test_help(cmd, capsys):
    with pytest.raises(SystemExit) as ei:
        main([cmd, "--help"])
    assert ei.value.code == 0
    assert "usage" in capsys.readouterr().out


def test_check_tools_cli(fake_tools, config_file, monkeypatch, capsys):
    assert main(["check-tools", "--config", config_file]) == 0
    assert "ALL OK" in capsys.readouterr().out
    monkeypatch.setenv("PATH", "/nonexistent")
    assert main(["check-tools", "--config", config_file]) == 1


def test_build_manifest_and_n_chunks(tmp_path, capsys):
    for i in range(5):
        ph.save(str(tmp_path / "L" / "imagesTr" / ("LUMIRMRI_%04d_0000.nii.gz" % i)), np.zeros((2, 2, 2)), np.eye(4))
    ex = tmp_path / "ex.txt"
    ex.write_text("LUMIRMRI_0004\n")
    out = str(tmp_path / "lumir.csv")
    assert main(["build-manifest", "--dataset", "lumir", "--root", str(tmp_path / "L"), "--out", out,
                 "--exclude", str(ex)]) == 0
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 4 and rows[0]["split"] == "train" and rows[0]["in_template_space"] == "true"
    capsys.readouterr()
    assert main(["n-chunks", "--manifest", out, "--chunk-size", "3"]) == 0
    assert capsys.readouterr().out.strip() == "2"
    assert main(["build-manifest", "--dataset", "lumir", "--root", str(tmp_path / "empty"), "--out",
                 str(tmp_path / "e.csv")]) == 1


def test_select_rows():
    rows = [ManifestRow(dataset="d", subject="s%d" % i, image_path="/x") for i in range(5)]
    assert [i for i, _ in select_rows(rows, 3, 10, None)] == [3, 4]
    assert [i for i, _ in select_rows(rows, None, None, [4, 0])] == [4, 0]
    assert select_rows(rows, 7, 9, None) == []
    with pytest.raises(SystemExit):
        select_rows(rows, None, None, [9])
    with pytest.raises(SystemExit):
        select_rows(rows, 0, 1, [1])
