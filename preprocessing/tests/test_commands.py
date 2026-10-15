"""Exact argv lists for every external tool."""
from preprocessing.commands import (ants_apply_cmd, ants_registration_cmd, flirt_apply_cmd, flirt_register_cmd,
                                    fslreorient2std_cmd, n4_cmd, synthseg_cmd, synthstrip_cmd)
from preprocessing.config import load_config


def test_synthstrip():
    assert synthstrip_cmd("/b/mri_synthstrip", "/i.nii.gz", "/o.nii.gz", "/m.nii.gz", ["-t", 8]) == [
        "/b/mri_synthstrip", "-i", "/i.nii.gz", "-o", "/o.nii.gz", "-m", "/m.nii.gz", "-t", "8"]


def test_n4():
    cfg = load_config(None)
    assert n4_cmd("N4", "/s.nii.gz", "/m.nii.gz", "/n4.nii.gz", cfg["n4"]) == [
        "N4", "-d", "3", "-i", "/s.nii.gz", "-x", "/m.nii.gz", "-o", "/n4.nii.gz", "-s", "4",
        "-b", "[200]", "-c", "[50x50x50x50,0.0]", "-v", "0"]


def test_fslreorient2std():
    assert fslreorient2std_cmd("fslreorient2std", "/a.nii.gz", "/b.nii.gz") == [
        "fslreorient2std", "/a.nii.gz", "/b.nii.gz"]


def test_flirt():
    cfg = load_config(None)
    assert flirt_register_cmd("flirt", "/mov.nii.gz", "/MNI_2mm.nii.gz", "/w/reg.mat",
                              cfg["registration"]["flirt"]) == [
        "flirt", "-in", "/mov.nii.gz", "-ref", "/MNI_2mm.nii.gz", "-omat", "/w/reg.mat", "-dof", "9",
        "-cost", "corratio", "-searchrx", "-30", "30", "-searchry", "-30", "30", "-searchrz", "-30", "30"]
    assert flirt_apply_cmd("flirt", "/x.nii.gz", "/t.nii.gz", "/x.mat", "/o.nii.gz", "spline", True) == [
        "flirt", "-in", "/x.nii.gz", "-ref", "/t.nii.gz", "-applyxfm", "-init", "/x.mat", "-out", "/o.nii.gz",
        "-interp", "spline", "-datatype", "float"]
    assert flirt_apply_cmd("flirt", "/l.nii.gz", "/t.nii.gz", "/l.mat", "/o.nii.gz", "nearestneighbour") == [
        "flirt", "-in", "/l.nii.gz", "-ref", "/t.nii.gz", "-applyxfm", "-init", "/l.mat", "-out", "/o.nii.gz",
        "-interp", "nearestneighbour"]


def test_ants_registration():
    cfg = load_config(None)
    a = dict(cfg["registration"]["ants"])
    mi = "MI[/F.nii.gz,/M.nii.gz,1,32,Regular,0.25]"
    stage = ["--convergence", "[1000x500x250,1e-6,10]", "--shrink-factors", "4x2x1",
             "--smoothing-sigmas", "2x1x0vox"]
    expected = ["antsRegistration", "--dimensionality", "3", "--float", "1", "--collapse-output-transforms", "1",
                "--output", "/w/reg_", "--interpolation", "Linear", "--winsorize-image-intensities",
                "[0.005,0.995]", "--use-histogram-matching", "0", "--initial-moving-transform",
                "[/F.nii.gz,/M.nii.gz,1]", "--transform", "Rigid[0.1]", "--metric", mi] + stage + [
                "--transform", "Affine[0.1]", "--metric", mi] + stage + ["--random-seed", "42", "--verbose", "0"]
    assert ants_registration_cmd("antsRegistration", "/F.nii.gz", "/M.nii.gz", "/w/reg_", a,
                                 "/fm.nii.gz", "/mm.nii.gz") == expected
    a.update(transform="Similarity", random_seed=None, use_masks=True)
    got = ants_registration_cmd("antsRegistration", "/F.nii.gz", "/M.nii.gz", "/w/reg_", a, "/fm.nii.gz", "/mm.nii.gz")
    assert "Similarity[0.1]" in got and "--random-seed" not in got
    assert got[-4:] == ["-x", "[/fm.nii.gz,/mm.nii.gz]", "--verbose", "0"]


def test_ants_apply():
    assert ants_apply_cmd("aAT", "/i.nii.gz", "/r.nii.gz", "/o.nii.gz", "GenericLabel", "/a.mat") == [
        "aAT", "-d", "3", "-i", "/i.nii.gz", "-r", "/r.nii.gz", "-o", "/o.nii.gz", "-n", "GenericLabel",
        "--float", "1", "-v", "0", "-t", "/a.mat"]
    assert "-t" not in ants_apply_cmd("aAT", "/i", "/r", "/o", "Linear", None)


def test_synthseg():
    cfg = load_config(None)
    s = dict(cfg["synthseg"])
    assert synthseg_cmd("mri_synthseg", "/b/in.txt", "/b/out.txt", s, "/b/vols.txt", "/b/qc.txt") == [
        "mri_synthseg", "--i", "/b/in.txt", "--o", "/b/out.txt", "--parc", "--robust", "--vol", "/b/vols.txt",
        "--qc", "/b/qc.txt", "--threads", "8"]
    s.update(parc=False, robust=False, vol=False, qc=False, cpu=True, extra_args=["--fast"])
    assert synthseg_cmd("mri_synthseg", "/i", "/o", s, "/v", "/q") == [
        "mri_synthseg", "--i", "/i", "--o", "/o", "--threads", "8", "--cpu", "--fast"]
