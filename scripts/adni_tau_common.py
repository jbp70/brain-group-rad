"""Shared helpers for the ADNI tau-PET SUVR experiment (paper Figure 5).

scripts/adni_tau_prepare.py  raw ADNI folders -> inventory, frame-averaged native static PETs, manifests
scripts/adni_tau_suvr.py     per-patient SUVR in four arms (native / forward / backward / central) + stats

Geometry: every matrix here is a 4x4 POINT map in world RAS mm (nibabel convention). A NIfTI affine
maps voxel indices to world RAS. ITK / ANTs transforms are LPS and map FIXED points to MOVING points.

Python 3.8, numpy 1.21 / scipy 1.4 / nibabel 3.1 (tf10_nv.sif).
"""
import datetime as _dt
import os
import re
import subprocess
from typing import Dict, List, Optional, Sequence

import numpy as np

# Raw file names, e.g.
#   Baseline.3D_T1_weighted.I829296.101.2017-03-13.nii.gz
#   Baseline.tauPET.I829443.101.2017-03-13.nii.gz
#   Follow_up_1.tauPET.I1265214.103.2019-12-11.nii.gz
# -> role (Baseline | Follow_up_<k>), description, image id, series number, date.
RAW_NAME = re.compile(r"^(?P<role>Baseline|Follow_up_(?P<k>\d+))\.(?P<desc>.+)\.(?P<iid>I\d+)\.(?P<series>[^.]+)\."
                      r"(?P<date>\d{4}-\d{2}-\d{2})\.nii(?:\.gz)?$")
PATIENT_DIR = re.compile(r"^\d{3}_S_\d{4}$")
# Processed PET folder names written by the manifests: <patient>_<bl|fu>_PET.
GROUP_REGEX = r"^(\d{3}_S_\d{4})_(?:bl|fu)_PET$"

LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


def parse_raw_name(fname: str) -> Optional[Dict[str, object]]:
    """Role, modality and date of one raw ADNI file name (None if the name does not match)."""
    m = RAW_NAME.match(os.path.basename(fname))
    if not m:
        return None
    desc = m.group("desc")
    low = desc.lower()
    if "pet" in low or "tau" in low or "av1451" in low or "ftp" in low:
        modality = "PET"
    elif "t1" in low or "mprage" in low or "spgr" in low:
        modality = "T1"
    else:
        modality = "unknown"
    k = m.group("k")
    return {"role": "bl" if k is None else "fu%d" % int(k), "followup_index": 0 if k is None else int(k),
            "modality": modality, "description": desc, "image_id": m.group("iid"),
            "series": m.group("series"), "date": m.group("date")}


def years_between(d0: str, d1: str) -> float:
    a = _dt.date.fromisoformat(d0)
    b = _dt.date.fromisoformat(d1)
    return (b - a).days / 365.25


# ------------------------------------------------------------------ ITK / ANTs transforms
def read_itk_affine(path: str) -> np.ndarray:
    """ANTs/ITK .mat (MatrixOffsetTransformBase, LPS, fixed -> moving point map) as a 4x4 RAS
    world point map (fixed RAS -> moving RAS)."""
    from scipy.io import loadmat
    z = loadmat(path)
    key = [k for k in z if k.startswith("AffineTransform") or k.startswith("MatrixOffsetTransform")]
    if not key:
        raise ValueError("%s: no affine transform parameters (keys %s)" % (path, sorted(z)))
    p = np.asarray(z[key[0]], float).ravel()
    c = np.asarray(z.get("fixed", np.zeros(3)), float).ravel()
    if p.size != 12 or c.size != 3:
        raise ValueError("%s: expected 12 parameters and a 3-vector centre, got %d / %d" % (path, p.size, c.size))
    A = p[:9].reshape(3, 3)
    t = p[9:]
    m = np.eye(4)
    m[:3, :3] = A
    m[:3, 3] = t + c - A @ c          # y = A (x - c) + c + t
    return LPS @ m @ LPS


def write_itk_affine(ras_point_map: np.ndarray, path: str) -> None:
    """Inverse of read_itk_affine (centre 0); used by the self-checks."""
    from scipy.io import savemat
    m = LPS @ np.asarray(ras_point_map, float) @ LPS
    params = np.r_[m[:3, :3].ravel(), m[:3, 3]].reshape(12, 1)
    savemat(path, {"AffineTransform_double_3_3": params, "fixed": np.zeros((3, 1))}, format="4")


def rigid_deviation(m: np.ndarray) -> float:
    """max |R^T R - I| of the linear part (0 for a rigid map)."""
    r = np.asarray(m, float)[:3, :3]
    return float(np.abs(r.T @ r - np.eye(3)).max())


def displacement_stats(m: np.ndarray, points_world: np.ndarray) -> Dict[str, float]:
    """Mean / max displacement (mm) of world points [K, 3] under the point map m."""
    p = np.c_[points_world, np.ones(len(points_world))]
    d = np.linalg.norm((p @ (np.asarray(m, float) - np.eye(4)).T)[:, :3], axis=1)
    return {"mean_mm": float(d.mean()) if d.size else 0.0, "max_mm": float(d.max()) if d.size else 0.0}


# ------------------------------------------------------------------ ANTs command lines
def ants_rigid_cmd(ants_bin: str, fixed: str, moving: str, prefix: str, init: int, levels: str = "4x2x1",
                   sigmas: str = "2x1x0vox", iters: str = "1000x500x250", threads: int = 1,
                   fixed_mask: Optional[str] = None) -> List[str]:
    """antsRegistration rigid, Mattes MI. init: 0 = image centres, 1 = centres of mass."""
    argv = [os.path.join(ants_bin, "antsRegistration") if ants_bin else "antsRegistration",
            "--dimensionality", "3", "--float", "1", "--output", "[%s]" % prefix,
            "--interpolation", "Linear", "--winsorize-image-intensities", "[0.005,0.995]",
            "--initial-moving-transform", "[%s,%s,%d]" % (fixed, moving, init),
            "--transform", "Rigid[0.1]",
            "--metric", "MI[%s,%s,1,32,Regular,0.25]" % (fixed, moving),
            "--convergence", "[%s,1e-6,10]" % iters, "--shrink-factors", levels,
            "--smoothing-sigmas", sigmas, "--random-seed", "42", "--verbose", "0"]
    if fixed_mask:
        argv += ["--masks", "[%s,NULL]" % fixed_mask]
    return argv


def run(argv: Sequence[str], log_path: Optional[str] = None, threads: int = 1) -> None:
    env = dict(os.environ)
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(int(threads))
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "a") as f:
            f.write("$ " + " ".join(argv) + "\n")
            f.flush()
            r = subprocess.run(list(argv), stdout=f, stderr=subprocess.STDOUT, env=env, stdin=subprocess.DEVNULL)
    else:
        r = subprocess.run(list(argv), env=env, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError("command failed (rc %d): %s%s" % (r.returncode, " ".join(argv[:3]),
                                                             (" (log %s)" % log_path) if log_path else ""))


def default_ants_bin() -> str:
    """ANTs 2.4.4 on Martinos (not on any PATH), or '' to use PATH."""
    p = "/autofs/cluster/pubsw/2/pubsw/Linux2-2.3-x86_64/packages/ANTS/2.4.4/bin"
    return os.environ.get("ANTS_BIN", p if os.path.isdir(p) else "")
