import json
import os

import pytest
import yaml

from preprocessing.config import (DEFAULT_CONFIG, MARTINOS_ANTS_BIN, MARTINOS_FREESURFER_HOME, MARTINOS_FSLDIR,
                                  ConfigError, contrast_key, expand_path, load_config, mni_template,
                                  target_paths)
from preprocessing.tools import resolve_environment

EXAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_example.yaml")


def test_defaults_and_overrides():
    cfg = load_config(None, threads=6, synthseg_cpu=True)
    assert cfg["threads"] == 6 and cfg["synthseg"]["threads"] == 6 and cfg["synthseg"]["cpu"]
    assert cfg["registration"]["backend"] == "flirt"
    assert DEFAULT_CONFIG["threads"] == 4  # defaults not mutated


def test_example_config_loads():
    cfg = load_config(EXAMPLE)
    assert cfg["registration"]["flirt"]["dof"] == 9
    assert cfg["mni_templates"]["T1"]["image_1mm"] == MARTINOS_FSLDIR + "/data/standard/MNI152_T1_1mm_brain.nii.gz"
    assert "PLACEHOLDER" in cfg["mni_templates"]["T2"]["image_1mm"]  # still unknown
    assert cfg["env_setup"] == DEFAULT_CONFIG["env_setup"] and cfg["tmp_dir"] is None
    text = open(EXAMPLE).read()
    for gone in ("module load", "conda", "/usr/pubsw/packages"):
        assert gone not in text, gone


def test_default_env_setup_is_martinos_container_env():
    env = resolve_environment(load_config(None)["env_setup"], base_env={"PATH": "/usr/bin:/bin"})
    assert env["FREESURFER_HOME"] == MARTINOS_FREESURFER_HOME == "/usr/local/freesurfer/dev"
    assert env["FSLDIR"] == MARTINOS_FSLDIR and env["FSLOUTPUTTYPE"] == "NIFTI_GZ"
    assert env["PATH"].split(":")[:3] == [MARTINOS_FREESURFER_HOME + "/bin", MARTINOS_FSLDIR + "/bin",
                                          MARTINOS_ANTS_BIN]
    assert env["PATH"].endswith(":/usr/bin:/bin")
    assert "/usr/pubsw" not in json.dumps(DEFAULT_CONFIG)


def test_deep_merge_and_validation(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"registration": {"backend": "ants", "ants": {"transform": "Similarity"}},
                                 "contrast_template": {"T1map": "T1"}}))
    cfg = load_config(str(p))
    assert cfg["registration"]["ants"]["transform"] == "Similarity"
    assert cfg["registration"]["ants"]["metric_bins"] == 32
    assert cfg["contrast_template"]["T1map"] == "T1" and cfg["contrast_template"]["T2w"] == "T2"
    for bad in ({"threads": 0}, {"registration": {"backend": "spm"}}, {"unknown_key": 1},
                {"registration": {"ants": {"transform": "Rigid9"}}}, {"normalization": {"upper_percentile": 150}},
                {"env_setup": "module load x"}, {"tmp_dir": ""}, {"tmp_dir": 3}):
        p.write_text(yaml.safe_dump(bad))
        with pytest.raises(ConfigError):
            load_config(str(p))


def test_paths_and_contrast_templates():
    env = {"FSLDIR": "/opt/fsl"}
    assert expand_path("${FSLDIR}/data/x.nii.gz", env) == "/opt/fsl/data/x.nii.gz"
    with pytest.raises(ConfigError):
        expand_path("${NOPE}/x", env)
    cfg = load_config(None)
    assert contrast_key(cfg, "acq-mprage_T1w") == "T1"
    assert contrast_key(cfg, "FLAIR") == "T2"
    assert contrast_key(cfg, "weird") == "T1"
    t = mni_template(cfg, "T1w", env)
    assert t["image_1mm"] == MARTINOS_FSLDIR + "/data/standard/MNI152_T1_1mm_brain.nii.gz"
    assert t["image_2mm"] == MARTINOS_FSLDIR + "/data/standard/MNI152_T1_2mm_brain.nii.gz"
    cfg["mni_templates"]["T1"]["image_1mm"] = "${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz"
    assert mni_template(cfg, "T1w", env)["image_1mm"] == "/opt/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"
    with pytest.raises(ConfigError):
        mni_template(cfg, "T2w", env)  # T2 template not set by default
    with pytest.raises(ConfigError):
        target_paths(cfg, env)
    cfg["target_dir"] = "/t"
    assert target_paths(cfg, env)["template"] == os.path.join("/t", "template_lumirgrid.nii.gz")
