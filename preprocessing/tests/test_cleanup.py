"""`cleanup` command: deletes only work/, mask.nii.gz, qc.png of completed rows."""
import json
import os

from preprocessing.cli import main
from preprocessing.cleanup import cleanup


def _row(root, name, status="completed", steps=None):
    d = os.path.join(root, "ds", name)
    os.makedirs(os.path.join(d, "work", "synthseg_batch"))
    for rel, n in (("work/stripped.nii.gz", 1000), ("work/synthseg_batch/inputs.txt", 10),
                   ("mask.nii.gz", 100), ("qc.png", 50), ("image.nii.gz", 500), ("qc.json", 5)):
        with open(os.path.join(d, rel), "wb") as f:
            f.write(b"x" * n)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"status": status, "steps": steps or {"normalize": {"status": "done"}}}, f)
    return d


def _tree(root):
    out = []
    for r, dirs, files in os.walk(root):
        out += [os.path.relpath(os.path.join(r, f), root) for f in files]
    return sorted(out)


def test_dry_run_then_delete_and_refusals(tmp_path, capsys):
    out = str(tmp_path / "out")
    ok = _row(out, "s_ok")
    _row(out, "s_partial", status="partial")
    _row(out, "s_running", steps={"normalize": {"status": "done"}, "synthseg": {"status": "running"}})
    keep = os.path.join(out, "ds", "notes.txt")
    open(keep, "w").write("keep")
    before = _tree(out)
    args = ["cleanup", "--out-root", out, "--dataset", "ds", "--work", "--mask", "--qc-png"]
    assert main(args + ["--dry-run"]) == 3          # refused rows -> exit 3
    text = capsys.readouterr().out
    assert _tree(out) == before                     # dry run changes nothing
    assert "WOULD DELETE %s (1010 bytes)" % os.path.join(ok, "work") in text
    assert "REFUSE" in text and "status=partial" in text and "synthseg=running" in text
    assert "would delete 3 items, 0.0 MB" in text
    assert main(args) == 3
    after = _tree(out)
    assert not os.path.exists(os.path.join(ok, "work"))
    assert not os.path.exists(os.path.join(ok, "mask.nii.gz")) and not os.path.exists(os.path.join(ok, "qc.png"))
    assert os.path.isfile(os.path.join(ok, "image.nii.gz")) and os.path.isfile(os.path.join(ok, "qc.json"))
    assert os.path.isfile(keep)
    for name in ("s_partial", "s_running"):          # untouched
        assert [p for p in before if p.startswith(name)] == [p for p in after if p.startswith(name)]
    meta = json.load(open(os.path.join(ok, "meta.json")))
    assert meta["work_cleaned"] is True and meta["mask_saved"] is False


def test_selected_items_only_and_bad_args(tmp_path):
    out = str(tmp_path / "out")
    d = _row(out, "s1")
    res = cleanup(out, ["ds"], qc_png=True, echo=lambda s: None)
    assert [os.path.basename(p) for p, _ in res["deleted"]] == ["qc.png"] and res["bytes"] == 50
    assert os.path.isdir(os.path.join(d, "work")) and os.path.isfile(os.path.join(d, "mask.nii.gz"))
    assert "work_cleaned" not in json.load(open(os.path.join(d, "meta.json")))
    assert main(["cleanup", "--out-root", out, "--dataset", "ds"]) == 1           # nothing selected
    assert main(["cleanup", "--out-root", out, "--dataset", "..", "--work"]) == 1  # not a dataset name
    assert main(["cleanup", "--out-root", out, "--dataset", "nope", "--work"]) == 1
    # A symlinked work/ is never followed or deleted.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("x")
    d2 = os.path.join(out, "ds", "s2")
    os.makedirs(d2)
    with open(os.path.join(d2, "meta.json"), "w") as f:
        json.dump({"status": "completed", "steps": {}}, f)
    os.symlink(str(outside), os.path.join(d2, "work"))
    assert main(["cleanup", "--out-root", out, "--dataset", "ds", "--work"]) == 0
    assert (outside / "precious.txt").is_file() and os.path.islink(os.path.join(d2, "work"))
    assert not os.path.exists(os.path.join(d, "work"))
