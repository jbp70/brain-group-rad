"""Pipeline configuration: defaults, YAML/JSON loading, validation, overrides.

The user config is deep-merged onto DEFAULT_CONFIG, so it only needs the keys
that differ (see config_example.yaml). Unknown keys are rejected to catch typos.
Path values may contain ``${VAR}`` (e.g. ``${FSLDIR}``); they are expanded with
the environment resolved from ``env_setup`` at use time (see expand_path).
The defaults (env_setup, MNI templates) are the confirmed Martinos paths for
running inside tf10_nv.sif (see run_in_container.sh).

Example:
    from preprocessing.config import load_config
    cfg = load_config("my_config.yaml", threads=8)
"""
import copy
import json
import os
import re
from typing import Any, Dict, Mapping, Optional

import yaml

# Martinos install paths, verified inside tf10_nv.sif on glacier (2026-10-14).
# /usr/pubsw is a host symlink that a -B bind does not expose inside the
# container, so FSL and ANTs use their resolved /autofs/cluster/pubsw paths.
MARTINOS_FREESURFER_HOME = "/usr/local/freesurfer/dev"
MARTINOS_FSLDIR = "/autofs/cluster/pubsw/2/pubsw/Linux2-2.3-x86_64/packages/fsl.64bit/5.0.7"
MARTINOS_ANTS_BIN = "/autofs/cluster/pubsw/2/pubsw/Linux2-2.3-x86_64/packages/ANTS/2.4.4/bin"

DEFAULT_CONFIG: Dict[str, Any] = {
    # Lines run once with bash -c (inside the container on the cluster).
    "env_setup": [
        "export FREESURFER_HOME=%s" % MARTINOS_FREESURFER_HOME,
        "export FSLDIR=%s" % MARTINOS_FSLDIR,
        "export FSLOUTPUTTYPE=NIFTI_GZ",
        "export PATH=$FREESURFER_HOME/bin:$FSLDIR/bin:%s:$PATH" % MARTINOS_ANTS_BIN,
    ],
    # Folder for temporary files of the external tools (exported as TMPDIR).
    # null: $TMPDIR if it exists and is writable (e.g. /scratch/$SLURM_JOBID),
    # else <out-root>/_tmp. Never relies on an unbound host TMPDIR.
    "tmp_dir": None,
    "tools": {
        "mri_synthstrip": "mri_synthstrip",
        "mri_synthseg": "mri_synthseg",
        "antsRegistration": "antsRegistration",
        "antsApplyTransforms": "antsApplyTransforms",
        "N4BiasFieldCorrection": "N4BiasFieldCorrection",
        "flirt": "flirt",
        "fslreorient2std": "fslreorient2std",
    },
    "threads": 4,
    "grid_tolerance_mm": 0.001,
    "lumir_expected_shape": [160, 224, 192],
    # Folder written by `build-target-grid`: target_grid.json,
    # template_lumirgrid.nii.gz, template_lumirgrid_mask.nii.gz.
    "target_dir": None,
    "mni_templates": {
        "T1": {
            "image_1mm": MARTINOS_FSLDIR + "/data/standard/MNI152_T1_1mm_brain.nii.gz",
            "mask_1mm": MARTINOS_FSLDIR + "/data/standard/MNI152_T1_1mm_brain_mask.nii.gz",
            "image_2mm": MARTINOS_FSLDIR + "/data/standard/MNI152_T1_2mm_brain.nii.gz",
        },
        "T2": {
            "image_1mm": None,
            "mask_1mm": MARTINOS_FSLDIR + "/data/standard/MNI152_T1_1mm_brain_mask.nii.gz",
            "image_2mm": None,
        },
    },
    # Contrast suffix -> MNI template key. Unknown contrasts use "T1".
    "contrast_template": {"T1w": "T1", "T2w": "T2", "FLAIR": "T2", "PDw": "T2"},
    # union_labels: rows skull-stripped by SynthStrip get mask = SynthStrip mask | (labels > 0)
    # (every provided label file), so no labelled voxel is stripped away (e.g. Mindboggle).
    # zero_labels_outside: written labels_*.nii.gz are set to 0 outside the final brain mask
    # (on the target grid, the same mask used for intensity normalization).
    "mask": {"threshold": 0.0, "fill_holes": True, "largest_component": True, "union_labels": False,
             "zero_labels_outside": False},
    "normalization": {"upper_percentile": 99.5, "lower_percentile": None},
    "synthstrip": {"extra_args": []},
    "n4": {"shrink_factor": 4, "bspline_fitting": "[200]", "convergence": "[50x50x50x50,0.0]"},
    "registration": {
        "backend": "flirt",
        "reorient": "nibabel",
        "flirt": {
            "dof": 9,
            "cost": "corratio",
            "search_range": 30,
            "estimate_at_2mm": True,
            "image_interp": "spline",
            "extra_args": [],
        },
        "ants": {
            "transform": "Affine",
            "gradient_step": 0.1,
            "metric_bins": 32,
            "sampling_strategy": "Regular",
            "sampling_percentage": 0.25,
            "convergence": "[1000x500x250,1e-6,10]",
            "shrink_factors": "4x2x1",
            "smoothing_sigmas": "2x1x0vox",
            "winsorize": "[0.005,0.995]",
            "random_seed": 42,
            "use_masks": False,
            "image_interp": "Linear",
        },
    },
    "resample": {"antialias": True, "target_spacing_mm": 1.0, "min_spacing_ratio": 0.95},
    "synthseg": {"parc": True, "robust": True, "cpu": False, "threads": 8, "vol": True,
                 "qc": True, "extra_args": []},
    # What stays on disk per subject (storage). work/ (intermediates, ~8 MB for LUMIR)
    # is deleted once the row is fully done; mask.nii.gz and qc.png are optional
    # because QC/export/qc-summary re-derive the mask as image > 0 (a subset of the
    # saved mask: no filled holes, and in-mask voxels normalized to 0, i.e. at or
    # below the normalization lower bound, drop out) and `qc-summary --png ...`
    # renders PNGs on demand. qc.json is always written.
    "outputs": {"keep_work": False, "save_mask": False, "save_qc_png": False},
    "qc": {"mad_threshold": 4.0},
}

_BACKENDS = ("flirt", "ants")
_REORIENT = ("nibabel", "fslreorient2std", "none")
_ANTS_TRANSFORMS = ("Affine", "Similarity")


class ConfigError(ValueError):
    """Invalid configuration."""


def deep_merge(base: Dict[str, Any], override: Mapping[str, Any], path: str = "") -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``; unknown keys raise."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        where = path + key
        if key not in out:
            # Free-form mappings: contrast_template and mni_templates accept new keys.
            if path in ("contrast_template.", "mni_templates."):
                out[key] = copy.deepcopy(val)
                continue
            raise ConfigError("unknown config key: %s" % where)
        if isinstance(out[key], dict) and isinstance(val, Mapping):
            out[key] = deep_merge(out[key], val, where + ".")
        else:
            out[key] = copy.deepcopy(val)
    return out


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def validate(cfg: Dict[str, Any]) -> None:
    """Type/range checks for the merged config."""
    _require(isinstance(cfg["env_setup"], list) and all(isinstance(s, str) for s in cfg["env_setup"]),
             "env_setup must be a list of strings")
    _require(all(isinstance(v, str) and v for v in cfg["tools"].values()),
             "tools.* must be non-empty strings")
    _require(isinstance(cfg["threads"], int) and cfg["threads"] > 0, "threads must be a positive int")
    _require(isinstance(cfg["grid_tolerance_mm"], (int, float)) and cfg["grid_tolerance_mm"] >= 0,
             "grid_tolerance_mm must be >= 0")
    _require(isinstance(cfg["lumir_expected_shape"], list) and len(cfg["lumir_expected_shape"]) == 3,
             "lumir_expected_shape must be a list of 3 ints")
    _require(cfg["target_dir"] is None or isinstance(cfg["target_dir"], str), "target_dir must be a path")
    _require(cfg["tmp_dir"] is None or (isinstance(cfg["tmp_dir"], str) and cfg["tmp_dir"]),
             "tmp_dir must be null or a path")
    up = cfg["normalization"]["upper_percentile"]
    lo = cfg["normalization"]["lower_percentile"]
    _require(isinstance(up, (int, float)) and 0 < up <= 100, "normalization.upper_percentile in (0,100]")
    _require(lo is None or (isinstance(lo, (int, float)) and 0 <= lo < up),
             "normalization.lower_percentile must be null or in [0, upper)")
    reg = cfg["registration"]
    _require(reg["backend"] in _BACKENDS, "registration.backend must be one of %s" % (_BACKENDS,))
    _require(reg["reorient"] in _REORIENT, "registration.reorient must be one of %s" % (_REORIENT,))
    _require(reg["ants"]["transform"] in _ANTS_TRANSFORMS,
             "registration.ants.transform must be one of %s (ANTs has no 9-DOF)" % (_ANTS_TRANSFORMS,))
    _require(isinstance(reg["flirt"]["dof"], int), "registration.flirt.dof must be an int")
    for key in ("synthstrip", "synthseg"):
        _require(isinstance(cfg[key]["extra_args"], list), "%s.extra_args must be a list" % key)
    _require(isinstance(reg["flirt"]["extra_args"], list), "registration.flirt.extra_args must be a list")
    _require(isinstance(cfg["synthseg"]["threads"], int) and cfg["synthseg"]["threads"] > 0,
             "synthseg.threads must be a positive int")
    for name, t in cfg["mni_templates"].items():
        _require(isinstance(t, dict) and "image_1mm" in t, "mni_templates.%s needs image_1mm" % name)
    for key in ("fill_holes", "largest_component", "union_labels", "zero_labels_outside"):
        _require(isinstance(cfg["mask"][key], bool), "mask.%s must be true or false" % key)
    for key, val in cfg["outputs"].items():
        _require(isinstance(val, bool), "outputs.%s must be true or false" % key)


def load_config(path: Optional[str] = None, threads: Optional[int] = None,
                synthseg_cpu: bool = False) -> Dict[str, Any]:
    """Load YAML/JSON config (or defaults when path is None), merge, override, validate."""
    user: Dict[str, Any] = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        user = json.loads(text) if str(path).endswith(".json") else (yaml.safe_load(text) or {})
        if not isinstance(user, dict):
            raise ConfigError("%s: top level must be a mapping" % path)
    if isinstance(user.get("qc"), Mapping) and "png" in user["qc"]:
        raise ConfigError("%s: qc.png was replaced by outputs.save_qc_png (default false; render PNGs on "
                          "demand with `qc-summary --png ...`); remove qc.png from the config" % path)
    cfg = deep_merge(DEFAULT_CONFIG, user)
    apply_overrides(cfg, threads=threads, synthseg_cpu=synthseg_cpu)
    validate(cfg)
    cfg["_config_path"] = os.path.abspath(path) if path else None
    return cfg


def apply_overrides(cfg: Dict[str, Any], threads: Optional[int] = None, synthseg_cpu: bool = False) -> None:
    """CLI overrides: --threads sets threads and synthseg.threads; --synthseg-cpu sets cpu."""
    if threads is not None:
        cfg["threads"] = int(threads)
        cfg["synthseg"]["threads"] = int(threads)
    if synthseg_cpu:
        cfg["synthseg"]["cpu"] = True


_VAR = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def expand_path(value: Optional[str], env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Expand ${VAR}/$VAR with ``env`` (default os.environ); raises if a variable is unset."""
    if value is None:
        return None
    env = os.environ if env is None else env

    def sub(m):
        name = m.group(1) or m.group(2)
        if name not in env:
            raise ConfigError("path %r uses $%s, which is not set (env_setup?)" % (value, name))
        return env[name]

    return os.path.expanduser(_VAR.sub(sub, value))


def contrast_key(cfg: Dict[str, Any], contrast: str) -> str:
    """MNI template key for a contrast string; uses the last '_' token (e.g. acq-x_T2w -> T2w)."""
    suffix = (contrast or "").split("_")[-1]
    return cfg["contrast_template"].get(suffix, "T1")


def mni_template(cfg: Dict[str, Any], contrast: str, env: Optional[Mapping[str, str]] = None
                 ) -> Dict[str, Optional[str]]:
    """Expanded MNI template paths (image_1mm, mask_1mm, image_2mm) for a contrast."""
    key = contrast_key(cfg, contrast)
    if key not in cfg["mni_templates"]:
        raise ConfigError("contrast_template maps %r to unknown mni_templates key %r" % (contrast, key))
    t = cfg["mni_templates"][key]
    out = {k: expand_path(t.get(k), env) for k in ("image_1mm", "mask_1mm", "image_2mm")}
    if not out["image_1mm"]:
        raise ConfigError("mni_templates.%s.image_1mm is not set (needed for contrast %r)" % (key, contrast))
    return out


def target_paths(cfg: Dict[str, Any], env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Paths of the build-target-grid outputs inside target_dir."""
    d = expand_path(cfg.get("target_dir"), env)
    if not d:
        raise ConfigError("target_dir is not set; run build-target-grid and set it in the config")
    return {
        "dir": d,
        "grid": os.path.join(d, "target_grid.json"),
        "template": os.path.join(d, "template_lumirgrid.nii.gz"),
        "template_mask": os.path.join(d, "template_lumirgrid_mask.nii.gz"),
    }
