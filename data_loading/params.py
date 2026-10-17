"""Pure-Python data constants, augmentation defaults and the DALI-side matrix formulas.

No DALI and no TF imports, so everything here is unit-tested locally. The DALI pipeline
(`data_loading.dali_loader`) evaluates the same formulas on DALI DataNodes.

Levels
------
``level`` is the downsampling factor of the precomputed pyramid (design 3.1): 4 -> 40x56x48 at
4 mm, 2 -> 80x112x96 at 2 mm, 1 -> 160x224x192 at 1 mm. File suffixes are ``_4``, ``_2`` and ``''``.

Augmentation ranges (design 3.2, v3)
-----------------------------------
All uniform and independent per image. Rotation: angle U(-r, r) degrees about a uniformly random
axis. Translation: U(-t, t) mm per axis, the same in mm at every level, converted to voxels with
the level's spacing. Scale: log-scale U(-l, l) per axis. Shear: the three strictly-upper-triangular
entries of K (TF axis order), U(-h, h).

DALI coordinates
----------------
DALI matrices act on (W, H, D) = reversed array axes, in DALI's continuous pixel coordinates
u = p + DALI_PIXEL_CENTER (p = voxel index). The image center is therefore at
u_c = (n - 1) / 2 + DALI_PIXEL_CENTER per axis. `DALI_PIXEL_CENTER` is the one constant used both
to center the DALI transforms and to convert DALI matrices to TF mm (`data_loading.postprocess`).
It is checked empirically by tests/gpu/verify_dali_pipeline.py check (a).
"""
import collections
import math
from dataclasses import dataclass

import numpy as np

LEVELS = (4, 2, 1)
LEVEL_SHAPE = {4: (40, 56, 48), 2: (80, 112, 96), 1: (160, 224, 192)}
LEVEL_SPACING = {4: 4.0, 2: 2.0, 1: 1.0}

# DALI warp_affine pixel-center convention (pixel i covers [i, i + 1), center i + 0.5).
# Single source for the DALI pipeline and the TF conversion; verified on GPU by check (a).
DALI_PIXEL_CENTER = 0.5

TASKS = ("rigid", "affine")

# Design 3.2 (v3) defaults. log_scale and shear are half-widths of uniform ranges.
DEFAULT_LINEAR_RANGES = {
    # Rigid: the network undoes rotation + translation only; scale/shear are a small nuisance S_i.
    "rigid": {"rotation_deg": 45.0, "translation_mm": 15.0,
              "log_scale": math.log(1.005), "shear": 0.0025},
    # Affine: the network undoes everything.
    "affine": {"rotation_deg": 45.0, "translation_mm": 15.0,
               "log_scale": math.log(1.10), "shear": 0.05},
}
# Deformable residual affine (milestone 6; used only when deform_residual_affine is True).
DEFAULT_DEFORM_RESIDUAL_RANGES = {"rotation_deg": 1.0, "translation_mm": 1.0,
                                  "log_scale": math.log(1.005), "shear": 0.0025}
DEFAULT_DEFORM_RESIDUAL_AFFINE = False

LinearRanges = collections.namedtuple(
    "LinearRanges", ["task", "level", "rotation_deg", "translation_mm", "translation_vox",
                     "log_scale", "shear"])


def check_level(level):
    level = int(level)
    if level not in LEVEL_SHAPE:
        raise ValueError("level must be one of {}, got {!r}".format(LEVELS, level))
    return level


def level_suffix(level):
    """File suffix of a level: '' (1), '_2', '_4'."""
    level = check_level(level)
    return "" if level == 1 else "_%d" % level


def mm_to_voxels(value_mm, level):
    """Convert a length in mm to voxels of `level` (isotropic spacing)."""
    return float(value_mm) / LEVEL_SPACING[check_level(level)]


def linear_ranges(task, level, rotation_deg=None, translation_mm=None, log_scale=None, shear=None):
    """Resolve augmentation ranges for a linear task and level (None -> design default).

    Returns `LinearRanges`; ``translation_vox`` is the translation half-width in voxels of the
    level (``translation_mm / spacing``). All values must be finite and >= 0.
    """
    if task not in TASKS:
        raise ValueError("task must be one of {}, got {!r}".format(TASKS, task))
    level = check_level(level)
    d = DEFAULT_LINEAR_RANGES[task]
    vals = {
        "rotation_deg": d["rotation_deg"] if rotation_deg is None else float(rotation_deg),
        "translation_mm": d["translation_mm"] if translation_mm is None else float(translation_mm),
        "log_scale": d["log_scale"] if log_scale is None else float(log_scale),
        "shear": d["shear"] if shear is None else float(shear),
    }
    for k, v in vals.items():
        if not math.isfinite(v) or v < 0:
            raise ValueError("{} must be finite and >= 0, got {!r}".format(k, v))
    return LinearRanges(task=task, level=level, translation_vox=mm_to_voxels(vals["translation_mm"], level),
                        **vals)


@dataclass
class IntensityAugConfig:
    """Per-image intensity augmentation (ported from the rigid project's DALI pipeline).

    Order: noise, brightness, contrast, simulated low resolution, gamma, then clamp to [0, 1].
    Each augmentation has an enable flag and a probability; ``enabled=False`` turns all off.
    Defaults are the rigid project's values.
    """
    enabled: bool = True
    noise: bool = True
    noise_prob: float = 0.1
    noise_std_max: float = 0.05          # stddev ~ U(0, noise_std_max)
    brightness: bool = True
    brightness_prob: float = 0.15
    brightness_range: tuple = (0.75, 1.25)
    contrast: bool = True
    contrast_prob: float = 0.15
    contrast_range: tuple = (0.5, 1.5)
    low_res: bool = True
    low_res_prob: float = 0.125
    low_res_min_factor: float = 0.5      # size factor ~ U(min, 1), NN down then linear up
    gamma: bool = True
    gamma_prob: float = 0.3
    gamma_range: tuple = (0.7, 1.5)

    @classmethod
    def off(cls):
        return cls(enabled=False)


# ---------------------------------------------------------------------------- DALI formulas
def dali_center(shape_xyz, pixel_center=DALI_PIXEL_CENTER):
    """Image center in DALI coordinates (DALI axis order W, H, D = reversed array axes)."""
    return [(float(n) - 1.0) / 2.0 + float(pixel_center) for n in reversed(list(shape_xyz))]


def scale_shear_rows(dx, dy, dz, a, b, c, zero=0.0):
    """Rows of K and K^-1 as 3x4 DALI-order matrices (centered coordinates, no translation).

    K is defined in TF axis order (x, y, z = array axes) as K = U D with D = diag(dx, dy, dz)
    (the per-axis scales) and U = [[1, a, b], [0, 1, c], [0, 0, 1]] (a = k_xy, b = k_xz,
    c = k_yz). K^-1 = D^-1 U^-1 with U^-1 = [[1, -a, a c - b], [0, 1, -c], [0, 0, 1]].
    In DALI order (axis reversal P) K_dali[i][j] = K_tf[2 - i][2 - j].

    Works for Python/numpy floats and for DALI DataNodes (only +, -, *, / are used); ``zero`` is
    the zero element (0.0 or a DALI constant).
    """
    fwd = [[dz, zero, zero, zero],
           [c * dz, dy, zero, zero],
           [b * dz, a * dy, dx, zero]]
    inv = [[1.0 / dz, zero, zero, zero],
           [-c / dy, 1.0 / dy, zero, zero],
           [(a * c - b) / dx, -a / dx, 1.0 / dx, zero]]
    return fwd, inv


def auto_initial_fill(n_files, batch_size, shape_xyz, budget_bytes=1 << 30, bytes_per_voxel=4):
    """Reader shuffle-buffer size: as many samples as fit in `budget_bytes`, in [batch, n_files].

    DALI's random_shuffle buffer holds decoded samples in host memory, so a full-dataset buffer
    is only affordable at level 4 (~0.4 MB per image); at level 1 (27.5 MB) it is ~39 samples.
    """
    per = int(np.prod([int(v) for v in shape_xyz])) * int(bytes_per_voxel)
    fill = max(int(batch_size), int(budget_bytes // max(per, 1)))
    return int(max(1, min(fill, int(n_files))))


# ---------------------------------------------------------------------------- numpy mirror
def _h(m34):
    m = np.eye(4)
    m[:3, :] = np.asarray(m34, np.float64)
    return m


def _translation_np(t):
    m = np.eye(4)
    m[:3, 3] = np.asarray(t, np.float64)
    return m


def _rotation_np(angle_deg, axis):
    """Right-handed rotation about `axis` (Rodrigues). DALI's sign convention is not assumed by
    anything downstream: the pipeline emits whatever matrix DALI built."""
    ax = np.asarray(axis, np.float64)
    ax = ax / np.linalg.norm(ax)
    th = np.deg2rad(angle_deg)
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    m = np.eye(4)
    m[:3, :3] = np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * (k @ k)
    return m


def linear_matrices_np(angle_deg, axis_dali, t_vox_dali, log_scale_xyz, shear_xyz, shape_xyz,
                       pixel_center=DALI_PIXEL_CENTER):
    """Numpy mirror of the DALI graph in `dali_loader.build_linear_matrices` (float64, 4x4).

    Composition (centered coordinates, column vectors, rightmost applied first):
        T  = C+ . Tr . R . K . C-        (rendered transform, template -> image i)
        Rt = C+ . Tr . R . C-            (rigid part; T = Rt . S with S = C+ K C-)
        T^-1 = C+ . K^-1 . R^-1 . Tr^-1 . C-   (pull matrix passed to warp_affine)
    with C+- translations by +-dali_center. Returns dict(T, T_inv, Rt), DALI order.
    """
    c = dali_center(shape_xyz, pixel_center)
    cp, cm = _translation_np(c), _translation_np(-np.asarray(c))
    ls = np.asarray(log_scale_xyz, np.float64)
    a, b, cc = [float(v) for v in shear_xyz]
    fwd, inv = scale_shear_rows(math.exp(ls[0]), math.exp(ls[1]), math.exp(ls[2]), a, b, cc)
    k, k_inv = _h(fwd), _h(inv)
    r = _rotation_np(angle_deg, axis_dali)
    tr = _translation_np(t_vox_dali)
    tr_inv = _translation_np(-np.asarray(t_vox_dali, np.float64))
    r_inv = _rotation_np(-angle_deg, axis_dali)
    return {"T": cp @ tr @ r @ k @ cm,
            "T_inv": cp @ k_inv @ r_inv @ tr_inv @ cm,
            "Rt": cp @ tr @ r @ cm}
