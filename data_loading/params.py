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
# Deformable task (milestone 6). Deliberately NOT part of TASKS: TASKS are the linear tasks
# (LinearPostprocess, control files, curriculum, ValidationSet iterate it).
DEFORM_TASK = "deform"
ALL_TASKS = TASKS + (DEFORM_TASK,)
# train18 label set: 18 foreground classes + background (preprocessing.luts TRAIN18_CLASSES).
N_TRAIN_LABELS = 19

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
# LNCC box-window side in voxels per level (design 6.2 v3.9): 20 mm at level 4, 14 mm at
# level 2, 9 mm at level 1 (the VoxelMorph / reference value at 1 mm). --ncc-window overrides.
NCC_WINDOW_DEFAULT = {4: 5, 2: 7, 1: 9}

LinearRanges = collections.namedtuple(
    "LinearRanges", ["task", "level", "rotation_deg", "translation_mm", "translation_vox",
                     "log_scale", "shear"])


def check_level(level):
    level = int(level)
    if level not in LEVEL_SHAPE:
        raise ValueError("level must be one of {}, got {!r}".format(LEVELS, level))
    return level


def ncc_window_for_level(level):
    """Default LNCC window (odd voxel count) of a level (`NCC_WINDOW_DEFAULT`)."""
    return NCC_WINDOW_DEFAULT[check_level(level)]


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
    return _resolve_ranges(task, level, DEFAULT_LINEAR_RANGES[task], rotation_deg, translation_mm,
                           log_scale, shear)


def deform_residual_ranges(level, rotation_deg=None, translation_mm=None, log_scale=None, shear=None):
    """Ranges of the deformable pipeline's optional residual affine (None -> design default,
    `DEFAULT_DEFORM_RESIDUAL_RANGES`). Same validation and fields as `linear_ranges`, with
    ``task='deform'``."""
    return _resolve_ranges(DEFORM_TASK, level, DEFAULT_DEFORM_RESIDUAL_RANGES, rotation_deg,
                           translation_mm, log_scale, shear)


def _resolve_ranges(task, level, d, rotation_deg, translation_mm, log_scale, shear):
    level = check_level(level)
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


# ---------------------------------------------------------------------------- control file
# Curriculum + global transform G (design 3.2, v3.5). One control file per pipeline,
# ``<run_dir>/control/<task>.npy``: a float32 vector of CONTROL_SIZE values, written atomically by
# `runtime.curriculum.ControlWriter` and re-read by the DALI pipeline for every batch
# (`dali_loader.read_control`). Layout (the single definition; indices below):
#   0 rot_deg      per-image rotation half-width (degrees)          U(-1, 1) * rot_deg
#   1 trans_mm     per-image translation half-width (mm, per axis)  U(-1, 1) * trans_mm
#   2 log_scale    per-image log-scale half-width (per axis)        U(-1, 1) * log_scale
#   3 shear        per-image shear half-width (k_xy, k_xz, k_yz)    U(-1, 1) * shear
#   4-6 g_axis     G rotation axis, TF axis order (x, y, z), unit norm
#   7 g_angle_deg  G rotation angle (degrees); 0 = no rotation
#   8-10 g_t_mm    G translation (mm, TF axis order x, y, z); 0 = no translation
#   11 version     monotonically increasing write counter (emitted per sample by the pipeline)
# The same G applies to every sample of a batch (= group). G is about the image center, like the
# per-image transforms, and is composed last: T_i = G Rt_i S_i.
CONTROL_FIELDS = ("rot_deg", "trans_mm", "log_scale", "shear", "g_axis_x", "g_axis_y", "g_axis_z",
                  "g_angle_deg", "g_tx_mm", "g_ty_mm", "g_tz_mm", "version")
CONTROL_SIZE = len(CONTROL_FIELDS)
CONTROL_INDEX = {k: i for i, k in enumerate(CONTROL_FIELDS)}
CONTROL_RANGE_KEYS = ("rotation_deg", "translation_mm", "log_scale", "shear")
DEFAULT_G_AXIS = (0.0, 0.0, 1.0)


def control_vector(rotation_deg, translation_mm, log_scale, shear, g_axis=DEFAULT_G_AXIS,
                   g_angle_deg=0.0, g_translation_mm=(0.0, 0.0, 0.0), version=0):
    """Control values -> float32 vector [CONTROL_SIZE] (layout above). ``g_axis`` is normalized;
    the ranges must be finite and >= 0 and the G values finite."""
    ranges = [float(v) for v in (rotation_deg, translation_mm, log_scale, shear)]
    for k, v in zip(CONTROL_RANGE_KEYS, ranges):
        if not math.isfinite(v) or v < 0:
            raise ValueError("control {} must be finite and >= 0, got {!r}".format(k, v))
    axis = np.asarray(g_axis, np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if not (math.isfinite(norm) and norm > 0):
        raise ValueError("g_axis must be a finite non-zero vector, got {!r}".format(g_axis))
    gt = np.asarray(g_translation_mm, np.float64).reshape(3)
    rest = [float(g_angle_deg), float(version)] + gt.tolist()
    if not all(math.isfinite(v) for v in rest):
        raise ValueError("G angle, translation and version must be finite")
    vec = ranges + (axis / norm).tolist() + [float(g_angle_deg)] + gt.tolist() + [float(version)]
    return np.asarray(vec, np.float32)


def parse_control(vec):
    """Inverse of `control_vector`: float32 [CONTROL_SIZE] -> dict(rotation_deg, translation_mm,
    log_scale, shear, g_axis (3,), g_angle_deg, g_translation_mm (3,), version)."""
    v = np.asarray(vec, np.float64).reshape(-1)
    if v.size != CONTROL_SIZE:
        raise ValueError("control vector must have %d values, got %d" % (CONTROL_SIZE, v.size))
    return {"rotation_deg": float(v[0]), "translation_mm": float(v[1]), "log_scale": float(v[2]),
            "shear": float(v[3]), "g_axis": v[4:7].copy(), "g_angle_deg": float(v[7]),
            "g_translation_mm": v[8:11].copy(), "version": int(round(v[11]))}


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
    """Right-handed rotation about `axis` (Rodrigues), the same matrix as DALI 1.18's
    ``fn.transforms.rotation`` (``rotation3D(axis, deg2rad(angle))`` in
    include/dali/core/geom/transform.h), in the coordinates the axis is given in."""
    ax = np.asarray(axis, np.float64)
    ax = ax / np.linalg.norm(ax)
    th = np.deg2rad(angle_deg)
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    m = np.eye(4)
    m[:3, :3] = np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * (k @ k)
    return m


# Sign applied to the control file's G angle before it reaches DALI (milestone 5, review-curriculum
# issue 2). DALI builds the right-handed rotation R_d(a_d, th) in DALI (reversed) axis order, with
# a_d = P a for the file's TF-order axis a, and M_tf = P M_dali P (P = axis reversal; the
# isotropic voxel scaling and the centering commute with it). P swaps x and z, an improper
# orthogonal matrix (det -1), and for any orthogonal Q: Q R(u, th) Q^T = R(det(Q) Q u, th). So
# P R(P a, th) P = R(-P P a, th) = R(-a, th) = R(a, -th): passing the file's angle unchanged
# would emit G = R(axis, -angle) in TF space. DALI therefore gets -angle, and the emitted G is
# R(axis, +angle) in TF space (`runtime.curriculum.global_matrix_mm`, the val_<task>_g sets and
# the logged control_g_angle_deg). The per-image rotations are reflected the same way, which
# needs no fix: their angle distribution is symmetric and the emitted matrices are the truth.
G_DALI_ANGLE_SIGN = -1.0


def control_global_dali_angle(g_angle_deg):
    """The angle (degrees) DALI receives for the control file's G angle (`G_DALI_ANGLE_SIGN`);
    mirrors `dali_loader.control_global_params`."""
    return G_DALI_ANGLE_SIGN * float(g_angle_deg)


def control_global_dali(g_axis_xyz, g_t_mm_xyz, level):
    """G axis and translation of the control file (TF order, mm) -> DALI order (axis, voxels).

    Mirrors `dali_loader.control_global_params`: the axis and the translation are reversed to
    DALI axis order and the translation is divided by the level's spacing (as the per-image
    translation, `mm_to_voxels`)."""
    s = LEVEL_SPACING[check_level(level)]
    axis = np.asarray(g_axis_xyz, np.float64).reshape(3)[::-1]
    t = np.asarray(g_t_mm_xyz, np.float64).reshape(3)[::-1] / s
    return axis, t


def linear_matrices_np(angle_deg, axis_dali, t_vox_dali, log_scale_xyz, shear_xyz, shape_xyz,
                       pixel_center=DALI_PIXEL_CENTER, g_angle_deg=0.0, g_axis_dali=DEFAULT_G_AXIS,
                       g_t_vox_dali=(0.0, 0.0, 0.0)):
    """Numpy mirror of the DALI graph in `dali_loader.build_linear_matrices` and
    `dali_loader.build_global_linear_matrices` (float64, 4x4).

    Composition (centered coordinates, column vectors, rightmost applied first):
        T  = G . C+ . Tr . R . K . C-    (rendered transform, template -> image i)
        Rt = G . C+ . Tr . R . C-        (rigid target part; T = Rt . S with S = C+ K C-)
        T^-1 = C+ . K^-1 . R^-1 . Tr^-1 . C- . G^-1   (pull matrix passed to warp_affine)
        G  = C+ . Tg . Rg . C-           (shared global transform; identity by default)
    with C+- translations by +-dali_center. Returns dict(T, T_inv, Rt, G), DALI order.
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
    g = cp @ _translation_np(g_t_vox_dali) @ _rotation_np(g_angle_deg, g_axis_dali) @ cm
    g_inv = (cp @ _rotation_np(-g_angle_deg, g_axis_dali)
             @ _translation_np(-np.asarray(g_t_vox_dali, np.float64)) @ cm)
    return {"T": g @ cp @ tr @ r @ k @ cm,
            "T_inv": cp @ k_inv @ r_inv @ tr_inv @ cm @ g_inv,
            "Rt": g @ cp @ tr @ r @ cm,
            "G": g}
