"""Canonical output fields, NIfTI writer, atlas and the internal consistency check
(predict-script design 5.1, 5.2, 5.3).

Internal conventions (design 5.1): arrays keep the nibabel (i, j, k) order, so internal channel a
is grid axis a; A_i maps image i -> central space in centered mm and images are pulled with
A_i^-1; u maps central -> image coordinates (warped = I o (id + u)), u_inv is its inverse. With
V = `models.transforms.index_to_mm_matrix(shape, spacing)` and Mv(B) = V^-1 B V, for an atlas voxel
p and a subject voxel y:

  branches        fwd_vox(p)                         inv_vox(y)
  rigid / affine  Mv(A^-1) [p; 1] - p                Mv(A) [y; 1] - y
  deform          u(p)                               u_inv(y)
  affine+deform   Mv(A^-1) [p + u(p); 1] - p         z + u_inv(z) - y, z = Mv(A) [y; 1]
                                                     (u_inv trilinear, border clamping)

and fwd_mm[..., a] = fwd_vox[..., a] * spacing[a]. In evaluator terms (evaluation/fields.py):
warped_i(x) = I_i(x + fwd_i(x)), convention ``--components xyz --signs +++ --units mm``.

The internal check (design 5.3) ties the converted fields to the model's own TF warp ops
(`layers.transform_image`, `diffeo.warp`, `diffeo.compose`), independent of the formulas above.
"""
import contextlib
import json
import logging
import os

import numpy as np
import tensorflow as tf

from data_loading import params as P
from evaluation import fields as F
from evaluation import metrics as EM
from models import diffeo
from models import layers as L
from models import transforms as tfm

logger = logging.getLogger(__name__)

LINEAR_BRANCHES = ("rigid", "affine")
DEFORM_BRANCHES = ("deform", "affine+deform")
BRANCH_VALUES = LINEAR_BRANCHES + DEFORM_BRANCHES
GRID_SHAPE = tuple(P.LEVEL_SHAPE[1])
GRID_SPACING = (1.0, 1.0, 1.0)
GRID_TOL = 1e-3                       # shape / spacing / reference affine (preprocessing grid_tolerance_mm)
GATE_COORD_VOX = 1e-3                 # design 5.3 gates
GATE_MOVED = 1e-3
INTERIOR_MARGIN = 2
# Pulled ones must equal 1 within 1e-6 (design 5.3 says 1e-5): a corner weight w missing at the
# border pulls the coordinate volume C (values up to n - 1 = 223) by w * |C|, so 1e-5 admits errors
# of ~2e-3 vox at isolated border voxels (measured 1.5e-3); 1e-6 bounds them by ~2e-4 per stage.
INTERIOR_ONES_TOL = 1e-6
MIN_INTERIOR_FRACTION = 0.5


class GridError(ValueError):
    """Processed grid / --grid-reference problem (CLI exit 2)."""


def _check_branches(branches):
    if branches not in BRANCH_VALUES:
        raise ValueError("branches must be one of %s, got %r" % (", ".join(BRANCH_VALUES), branches))


# ============================================================================ conversion (5.1)
def voxel_matrices(A, shape, spacing):
    """(Mv(A^-1), Mv(A)) as float64 numpy [k, 4, 4] for centered-mm matrices A [k, 4, 4]."""
    a = np.asarray(A.numpy() if hasattr(A, "numpy") else A, np.float64)
    if a.ndim == 2:
        a = a[None]
    v = tfm.index_to_mm_matrix(shape, spacing)
    v_inv = tfm.mm_to_index_matrix(shape, spacing)
    return v_inv[None] @ np.linalg.inv(a) @ v[None], v_inv[None] @ a @ v[None]


def _linear_disp(m, grid):
    """Displacement M [p; 1] - p of voxel matrices m [k, 4, 4] (float64) on grid [X, Y, Z, 3]:
    (L - I) p + t, formed in float64 and evaluated in float32."""
    d = np.asarray(m, np.float64)[:, :3, :].copy()
    d[:, :, :3] -= np.eye(3)[None]
    d = tf.constant(d, tf.float32)
    return tf.einsum("nij,xyzj->nxyzi", d[:, :, :3], grid) + d[:, None, None, None, :, 3]


def to_canonical(branches, A_chunk, U1, U1_inv, shape, spacing):
    """Internal results of one chunk -> (fwd_vox, inv_vox), TF float32 [k, X, Y, Z, 3] (design 5.1).

    branches: a ``--branches`` value. A_chunk: [k, 4, 4] centered-mm matrices (linear branches,
    affine+deform), else None. U1 / U1_inv: level-1 deform fields [k, X, Y, Z, 3] in voxels
    (deform branches), else None. shape / spacing: the processed (level-1) grid.
    """
    _check_branches(branches)
    shape = tuple(int(s) for s in shape)
    if branches == "deform":
        return tf.cast(U1, tf.float32), tf.cast(U1_inv, tf.float32)
    grid = L.index_grid(shape, tf.float32)
    m_inv, m = voxel_matrices(A_chunk, shape, spacing)
    lin_fwd = _linear_disp(m_inv, grid)
    lin_inv = _linear_disp(m, grid)
    if branches in LINEAR_BRANCHES:
        return lin_fwd, lin_inv
    # affine+deform: fwd(p) = (L - I) p + t + L u(p); inv(y) = (z - y) + u_inv(z), z = Mv(A) y
    u = tf.cast(U1, tf.float32)
    u_inv = tf.cast(U1_inv, tf.float32)
    lin = tf.constant(m_inv[:, :3, :3], tf.float32)
    fwd = lin_fwd + tf.einsum("nij,nxyzj->nxyzi", lin, u)
    z = grid[None] + lin_inv
    inv = lin_inv + L.sample_voxel_coords(u_inv, z, boundary="border", lean=True,
                                          lean_images_grad=False)
    return fwd, inv


def to_mm(field_vox, spacing):
    """fwd_mm[..., a] = fwd_vox[..., a] * spacing[a] (numpy float32 or TF float32)."""
    s = np.asarray(spacing, np.float32)
    if isinstance(field_vox, np.ndarray):
        return (field_vox * s).astype(np.float32, copy=False)
    return field_vox * tf.constant(s)


# ============================================================================ internal ops (5.3)
def _batch(x, channels_last=True):
    t = tf.convert_to_tensor(x)
    if t.dtype != tf.float32:
        t = tf.cast(t, tf.float32)
    if t.shape.rank == 3:
        t = t[None, ..., None]
    elif t.shape.rank == 4 and channels_last:
        t = t[None]
    return t


def _a_batch(A):
    a = tf.cast(tf.convert_to_tensor(A), tf.float32)
    return a[None] if a.shape.rank == 2 else a


def internal_moved(vol, branches, A=None, U1=None, spacing=GRID_SPACING, lean=True):
    """The model's own pull of ``vol`` [k, X, Y, Z, C] onto the atlas grid (design 5.3 table):
    rigid / affine transform_image(V, A^-1); deform warp(V, U1); affine+deform both in that order."""
    _check_branches(branches)
    v = _batch(vol)
    if branches in LINEAR_BRANCHES or branches == "affine+deform":
        v = L.transform_image(v, tfm.affine_inverse(_a_batch(A)), spacing)
    if branches in DEFORM_BRANCHES:
        v = diffeo.warp(v, U1, lean=lean, lean_images_grad=False)
    return v


def internal_pull_inv(vol, branches, A=None, U1_inv=None, spacing=GRID_SPACING, lean=True):
    """The model's own pull of an atlas-space ``vol`` onto the subject grid: rigid / affine
    transform_image(J, A); deform warp(J, U1_inv); affine+deform transform_image(warp(J, U1_inv), A)."""
    _check_branches(branches)
    v = _batch(vol)
    if branches in DEFORM_BRANCHES:
        v = diffeo.warp(v, U1_inv, lean=lean, lean_images_grad=False)
    if branches in LINEAR_BRANCHES or branches == "affine+deform":
        v = L.transform_image(v, _a_batch(A), spacing)
    return v


def interior_mask(ones_pulled, margin=INTERIOR_MARGIN, tol=INTERIOR_ONES_TOL):
    """Valid interior Omega (design 5.3): voxels with margin <= p_a <= n_a - 1 - margin on every
    axis where the pulled ones volume equals 1 within ``tol``. ``ones_pulled`` [X, Y, Z] or
    [1, X, Y, Z, 1] -> bool numpy [X, Y, Z]."""
    o = np.asarray(ones_pulled.numpy() if hasattr(ones_pulled, "numpy") else ones_pulled)
    o = o.reshape(o.shape[-4:-1] if o.ndim == 5 else o.shape[:3])
    mask = np.abs(o.astype(np.float64) - 1.0) <= tol
    inner = np.zeros(mask.shape, bool)
    m = int(margin)
    inner[m:mask.shape[0] - m, m:mask.shape[1] - m, m:mask.shape[2] - m] = True
    return mask & inner


def _max_on(x, mask):
    if not mask.any():
        return float("nan")
    return float(np.max(x[mask]))


def _le(x, bound):
    return x is not None and np.isfinite(x) and x <= bound


def internal_check(image, moved, fwd_vox, inv_vox, branches, A=None, U1=None, U1_inv=None,
                   spacing=GRID_SPACING, lean=True):
    """Design 5.3 checks of one subject. image / moved: [X, Y, Z] (moved = the written moved
    image, F.warp(image, fwd_vox)); fwd_vox / inv_vox: [X, Y, Z, 3] voxels (numpy or TF);
    A [1, 4, 4] or [4, 4]; U1 / U1_inv [1, X, Y, Z, 3] (deform branches). Returns the record
    {coord_fwd_max_vox, coord_inv_max_vox, moved_vs_internal_max_abs, moved_gated,
    inv_consistency_max_mm, inv_consistency_raw_max_mm, interior_voxels, interior_voxels_inv,
    interior_fraction, passed}."""
    _check_branches(branches)
    fwd = np.asarray(fwd_vox.numpy() if hasattr(fwd_vox, "numpy") else fwd_vox, np.float32)
    inv = np.asarray(inv_vox.numpy() if hasattr(inv_vox, "numpy") else inv_vox, np.float32)
    fwd, inv = fwd.reshape(fwd.shape[-4:]), inv.reshape(inv.shape[-4:])
    shape = fwd.shape[:3]
    grid = np.moveaxis(F.identity_grid(shape), 0, -1)
    kw = dict(spacing=spacing, lean=lean)
    c = tf.constant(grid[None])
    ones = tf.ones((1,) + tuple(shape) + (1,), tf.float32)
    om_f = interior_mask(internal_moved(ones, branches, A, U1, **kw))
    om_i = interior_mask(internal_pull_inv(ones, branches, A, U1_inv, **kw))
    mf = internal_moved(c, branches, A, U1, **kw).numpy()[0]
    coord_fwd = _max_on(np.max(np.abs(mf - (grid + fwd)), axis=-1), om_f)
    del mf
    mi = internal_pull_inv(c, branches, A, U1_inv, **kw).numpy()[0]
    coord_inv = _max_on(np.max(np.abs(mi - (grid + inv)), axis=-1), om_i)
    del mi
    img = np.asarray(image, np.float32).reshape(shape)
    mov_int = internal_moved(img[None, ..., None], branches, A, U1, **kw).numpy()[0, ..., 0]
    moved_diff = _max_on(np.abs(np.asarray(moved, np.float32).reshape(shape) - mov_int), om_f)
    e = F.compose(fwd, inv)
    raw = _max_on(F.norm_mm(e, spacing), om_f)
    if branches in DEFORM_BRANCHES:
        ref = diffeo.compose(U1_inv, U1, lean=lean).numpy()[0]
        inv_cons = _max_on(F.norm_mm(e - ref, spacing), om_f)
    else:
        inv_cons = raw
    total = int(np.prod(shape))
    frac = min(int(om_f.sum()), int(om_i.sum())) / float(total)
    moved_gated = branches != "affine+deform"
    passed = (_le(coord_fwd, GATE_COORD_VOX) and _le(coord_inv, GATE_COORD_VOX)
              and (not moved_gated or _le(moved_diff, GATE_MOVED)))
    if frac < MIN_INTERIOR_FRACTION:
        logger.warning("internal check: valid interior is only %.1f %% of the volume (the check "
                       "says little)", 100.0 * frac)
    return {"coord_fwd_max_vox": coord_fwd, "coord_inv_max_vox": coord_inv,
            "moved_vs_internal_max_abs": moved_diff, "moved_gated": moved_gated,
            "inv_consistency_max_mm": inv_cons, "inv_consistency_raw_max_mm": raw,
            "interior_voxels": int(om_f.sum()), "interior_voxels_inv": int(om_i.sum()),
            "interior_fraction": frac, "passed": bool(passed)}


# ============================================================================ grid (5.1)
def load_reference_affine(path):
    """(affine float64 [4, 4], shape tuple or None) of a --grid-reference: ``.json`` (key
    ``affine`` required, ``shape`` optional; target_grid.json) or any NIfTI nibabel reads."""
    path = str(path)
    if path.endswith(".json"):
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, ValueError) as e:
            raise GridError("cannot read grid reference %s: %s" % (path, e))
        try:
            aff = np.asarray(d["affine"], np.float64)
        except (KeyError, TypeError, ValueError):
            aff = None
        if aff is None or aff.shape != (4, 4) or not np.all(np.isfinite(aff)):
            raise GridError("grid reference %s has no 4x4 affine" % path)
        shape = d.get("shape")
        return aff, None if shape is None else tuple(int(s) for s in shape)[:3]
    import nibabel as nib
    try:
        img = nib.load(path)
    except Exception as e:  # noqa: BLE001  (nibabel raises several types)
        raise GridError("cannot read grid reference %s: %s" % (path, e))
    return np.asarray(img.affine, np.float64), tuple(int(s) for s in img.shape[:3])


def check_grid(processed_affine, shape, spacing, reference=None):
    """Grid record {shape, spacing, affine, reference_path, reference_max_abs_diff, checked}.
    GridError if shape != (160, 224, 192), spacing != 1 mm (within 1e-3) or, with a reference,
    its shape differs or max |affine - reference| > 1e-3."""
    shape = tuple(int(s) for s in shape)
    spacing = tuple(float(s) for s in spacing)
    aff = np.asarray(processed_affine, np.float64)
    if shape != GRID_SHAPE:
        raise GridError("processed grid shape %s != %s (the model's level-1 grid)" % (shape, GRID_SHAPE))
    if np.max(np.abs(np.asarray(spacing) - np.asarray(GRID_SPACING))) > GRID_TOL:
        raise GridError("processed grid spacing %s != (1, 1, 1) mm (the model was trained at 1 mm)"
                        % (spacing,))
    rec = {"shape": list(shape), "spacing": list(spacing), "affine": aff.tolist(),
           "reference_path": None, "reference_max_abs_diff": None, "checked": False}
    if reference is None:
        logger.info("no --grid-reference; affine not checked against the training grid")
        return rec
    ref, ref_shape = load_reference_affine(reference)
    rec["reference_path"] = os.path.abspath(str(reference))
    if ref_shape is not None and tuple(ref_shape) != shape:
        raise GridError("grid reference %s has shape %s, the processed grid %s"
                        % (reference, tuple(ref_shape), shape))
    diff = float(np.max(np.abs(aff - ref)))
    rec["reference_max_abs_diff"] = diff
    if diff > GRID_TOL:
        logger.error("processed affine\n%s\ndiffers from the grid reference %s\n%s\n(max |diff| %.4g)",
                     np.array2string(aff, precision=4), reference, np.array2string(ref, precision=4), diff)
        raise GridError("processed affine differs from the grid reference %s by %.4g (> %g): "
                        "processed %s, reference %s" % (reference, diff, GRID_TOL, aff.tolist(), ref.tolist()))
    rec["checked"] = True
    logger.info("grid reference %s: max |affine diff| %.3g", reference, diff)
    return rec


# ============================================================================ writer (5.2)
@contextlib.contextmanager
def compress_level(level):
    """nibabel gzip level for the writes inside the block (Opener.default_compresslevel)."""
    from nibabel.openers import Opener
    old = Opener.default_compresslevel
    Opener.default_compresslevel = int(level)
    try:
        yield
    finally:
        Opener.default_compresslevel = old


def write_nifti(path, array, affine):
    """Atomic NIfTI write: a temporary name in the same directory, then os.replace."""
    import nibabel as nib
    path = str(path)
    d, base = os.path.split(path)
    stem = base[:-7] if base.endswith(".nii.gz") else os.path.splitext(base)[0]
    ext = ".nii.gz" if base.endswith(".nii.gz") else os.path.splitext(base)[1]
    tmp = os.path.join(d, ".%s.tmp%d%s" % (stem, os.getpid(), ext))
    try:
        nib.save(nib.Nifti1Image(np.asarray(array), np.asarray(affine, np.float64)), tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def moved_and_labels(image, labels, fwd_vox):
    """(moved float32, warped labels in the labels' dtype): the evaluator's own warps."""
    return F.warp(image, fwd_vox, order=1), F.warp(labels, fwd_vox, order=0)


class AtlasAccumulator(object):
    """atlas = mean of the moved images (float64 sum / N, written float32); atlas_seg = majority
    vote of the warped labels (`evaluation.metrics.MajorityVote`, ties to the smaller label)."""

    def __init__(self, n, shape, label_dtype):
        self.n = int(n)
        self.sum = np.zeros(tuple(shape), np.float64)
        self.labels = np.zeros((self.n,) + tuple(shape), label_dtype)
        self.seen = np.zeros(self.n, bool)

    def add(self, i, moved, warped_labels):
        self.sum += moved
        self.labels[i] = warped_labels
        self.seen[i] = True

    def write(self, out_dir, affine):
        if not self.seen.all():
            raise RuntimeError("atlas: %d of %d subjects missing" % (int((~self.seen).sum()), self.n))
        atlas = (self.sum / float(self.n)).astype(np.float32)
        seg = EM.MajorityVote(self.labels).full()
        paths = {"atlas": write_nifti(os.path.join(out_dir, "atlas.nii.gz"), atlas, affine),
                 "atlas_seg": write_nifti(os.path.join(out_dir, "atlas_seg.nii.gz"), seg, affine)}
        return paths
