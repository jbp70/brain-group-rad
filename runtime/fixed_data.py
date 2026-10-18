"""TF-side data that needs no DALI: phantoms, fixed transforms, fixed groups, the validation set.

Every source here emits the same tuple format as the DALI pipeline
(`data_loading.dali_loader.linear_output_signature`): ``(images, T_dali[, Rt_dali])``, float32,
with 3x4 DALI-voxel matrices. The train step therefore runs the identical post-processing path
(`data_loading.postprocess.LinearPostprocess`) for DALI, overfit and synthetic batches.

Transforms (numpy float64, TF centered mm, 4x4), matching the DALI semantics
T = C+ Tr R K C- = [R K | t] about the image center (`dali_loader` module docstring):
rotation U(-r, r) degrees about a uniformly random axis, translation U(-t, t) mm per axis,
K = U D with D = diag(exp(U(-l, l))) and the strictly upper shears U(-h, h). Rendering is a pull
warp I_i(y) = I(T_i^-1 y) with `models.layers.transform_image`.
"""
import math
import os

import numpy as np
import tensorflow as tf

from data_loading import data_module as dm
from data_loading import params as P
from models import layers as L
from models import transforms as tfm


# ---------------------------------------------------------------------------- phantom
def smooth_phantom(shape, rng, n_blobs=8, sigma=4.0):
    """Asymmetric Gaussian blobs inside an ellipsoid envelope (voxel units).

    Returns (volume float32 in [0, 1], mask bool = the ellipsoid). The blobs and an off-center
    ramp break every symmetry, so orientation is identifiable.
    """
    shape = [int(v) for v in shape]
    g = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    c = [(n - 1) / 2.0 for n in shape]
    semi = [0.36 * n for n in shape]
    r2 = sum(((gi - ci) / si) ** 2 for gi, ci, si in zip(g, c, semi))
    mask = r2 <= 1.0
    envelope = 1.0 / (1.0 + np.exp((r2 - 1.0) / 0.05))
    vol = 0.25 + 0.15 * (g[0] / shape[0]) + 0.1 * (g[1] / shape[1]) ** 2
    for _ in range(int(n_blobs)):
        ctr = [ci + rng.uniform(-0.6, 0.6) * si for ci, si in zip(c, semi)]
        sg = [sigma * rng.uniform(0.7, 1.6) for _ in range(3)]
        d2 = sum(((gi - ci) / s) ** 2 for gi, ci, s in zip(g, ctr, sg))
        vol = vol + rng.uniform(0.3, 1.0) * np.exp(-0.5 * d2)
    vol = vol * envelope
    vol = vol / max(float(vol.max()), 1e-8)
    return vol.astype(np.float32), mask


# ---------------------------------------------------------------------------- transforms
def _rotation(axis, angle_rad):
    ax = np.asarray(axis, np.float64)
    ax = ax / np.linalg.norm(ax)
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(angle_rad) * k + (1 - math.cos(angle_rad)) * (k @ k)


def random_linear_transforms(rng, n, task, ranges=None, nuisance=True):
    """n random transforms -> (T, T_target), float64 [n, 4, 4] (TF centered mm).

    ``ranges``: dict(rotation_deg, translation_mm, log_scale, shear) (None ->
    `params.DEFAULT_LINEAR_RANGES[task]`, missing keys too). T = [R K | t]; T_target = [R | t]
    for 'rigid' (K is a nuisance) or T for 'affine'. ``nuisance=False`` sets K = I for 'rigid'.
    """
    if task not in P.TASKS:
        raise ValueError("task must be one of {}, got {!r}".format(P.TASKS, task))
    r = dict(P.DEFAULT_LINEAR_RANGES[task])
    r.update(ranges or {})
    use_k = task == "affine" or nuisance
    T = np.tile(np.eye(4), (int(n), 1, 1))
    tgt = T.copy()
    for i in range(int(n)):
        axis = rng.normal(size=3)
        ang = math.radians(rng.uniform(-r["rotation_deg"], r["rotation_deg"]))
        rot = _rotation(axis, ang)
        t = rng.uniform(-r["translation_mm"], r["translation_mm"], size=3)
        ls = rng.uniform(-r["log_scale"], r["log_scale"], size=3)
        a, b, c = rng.uniform(-r["shear"], r["shear"], size=3)
        k = np.array([[1.0, a, b], [0.0, 1.0, c], [0.0, 0.0, 1.0]]) @ np.diag(np.exp(ls))
        if not use_k:
            k = np.eye(3)
        T[i, :3, :3] = rot @ k
        T[i, :3, 3] = t
        tgt[i, :3, :3] = rot if task == "rigid" else rot @ k
        tgt[i, :3, 3] = t
    return T, tgt


def render_group(images, T, spacing, chunk=8):
    """Pull-render I_i(y) = I_i(T_i^-1 y): images [n, X, Y, Z, 1] (or one [X, Y, Z] volume used for
    every T) -> float32 numpy [n, X, Y, Z, 1]. float32 matrices (TF32 must be off), in chunks."""
    T = np.asarray(T, np.float64)
    x = np.asarray(images, np.float32)
    if x.ndim == 3:
        x = np.broadcast_to(x[None, ..., None], (T.shape[0],) + x.shape + (1,))
    inv = np.linalg.inv(T).astype(np.float32)
    out = []
    for s in range(0, T.shape[0], int(chunk)):
        out.append(L.transform_image(tf.constant(x[s:s + chunk]), tf.constant(inv[s:s + chunk]),
                                     spacing).numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def to_dali_batch(images, T, T_target, task, level):
    """(images, T_dali[, Rt_dali]) float32 tensors in the DALI output format (3x4 DALI voxel)."""
    level = P.check_level(level)
    shape, spacing = P.LEVEL_SHAPE[level], P.LEVEL_SPACING[level]

    def conv(m):
        m = tf.constant(np.asarray(m, np.float64)[:, :3, :])
        return tf.cast(tfm.tf_mm_to_dali_voxel(m, shape, spacing, P.DALI_PIXEL_CENTER), tf.float32)

    out = [tf.constant(np.asarray(images, np.float32)), conv(T)]
    if task == "rigid":
        out.append(conv(T_target))
    return tuple(out)


class FixedGroupSource:
    """Iterator that yields the same batch tuple forever (overfit mode, tests)."""

    def __init__(self, batch_tuple):
        self.batch = tuple(batch_tuple)

    def __iter__(self):
        return self

    def __next__(self):
        return self.batch


# ---------------------------------------------------------------------------- real data
def load_level_images(export_root, level, split, n=None, seed=0, contrast=None, manifest_path=None):
    """Load `split` images of `level` from the export with numpy -> (float32 [n, X, Y, Z, 1], files).

    ``n``: None = all, else a seeded random selection (without replacement) of n cases.
    """
    level = P.check_level(level)
    rows = dm.read_manifest(manifest_path or os.path.join(export_root, dm.MANIFEST_FILE))
    files = dm.manifest_files(rows, level, split=split, contrast=contrast)
    if not files:
        raise ValueError("no %r images at level %d in %s" % (split, level, export_root))
    if n is not None:
        if n > len(files):
            raise ValueError("asked for %d %r images, the export has %d" % (n, split, len(files)))
        sel = np.random.default_rng(int(seed)).choice(len(files), int(n), replace=False)
        files = [files[i] for i in sel]
    shape = tuple(P.LEVEL_SHAPE[level])
    vols = []
    for f in files:
        v = np.load(os.path.join(export_root, f)).astype(np.float32)
        if tuple(v.shape[:3]) != shape:
            raise ValueError("%s has shape %s, expected %s" % (f, v.shape, shape))
        vols.append(v.reshape(shape + (1,)))
    return np.stack(vols), files


# ---------------------------------------------------------------------------- validation
# Fixed per-task seed component of the validation sets (not the task's position in the ratio).
VAL_TASK_SEED = {"rigid": 101, "affine": 202}


class ValidationSet:
    """Fixed validation groups: fixed composition, fixed seeded transforms, rendered once.

    No flips, no intensity augmentation. Each group dict holds ``images`` (float32
    [Nv, X, Y, Z, 1]), ``T``, ``T_target`` and ``A_star`` (float64 [Nv, 4, 4]) and ``index``.
    """

    def __init__(self, task, level, groups):
        self.task = task
        self.level = P.check_level(level)
        self.groups = list(groups)
        self.group_size = int(self.groups[0]["images"].shape[0]) if self.groups else 0

    @staticmethod
    def _targets(T_target, iters):
        t = tf.constant(np.asarray(T_target, np.float64))
        return tfm.barycentric_targets(t, iters).numpy()

    @classmethod
    def build(cls, images, task, level, group_size, n_groups=None, ranges=None, seed=1234,
              target_iters=10):
        """Groups g = images[perm[(g Nv + k) % n_val]], k < Nv; default n_groups = ceil(n_val / Nv).

        Seeding depends only on (seed, task, g): the permutation uses (seed, task id) and the
        transforms of group g use (seed, task id, g), so a task's set does not change with the
        order (or presence) of other tasks in the task ratio.
        """
        level = P.check_level(level)
        if task not in VAL_TASK_SEED:
            raise ValueError("task must be one of {}, got {!r}".format(tuple(VAL_TASK_SEED), task))
        images = np.asarray(images, np.float32)
        n_val = images.shape[0]
        nv = int(group_size)
        if not 2 <= nv <= n_val:
            raise ValueError("validation group size must be in [2, n_val=%d], got %d" % (n_val, nv))
        if n_groups is None:
            n_groups = int(math.ceil(n_val / float(nv)))
        tid = VAL_TASK_SEED[task]
        perm = np.random.default_rng([int(seed), tid]).permutation(n_val)
        groups = []
        for g in range(int(n_groups)):
            idx = np.asarray([perm[(g * nv + k) % n_val] for k in range(nv)])
            rng = np.random.default_rng([int(seed), tid, g])
            T, tgt = random_linear_transforms(rng, nv, task, ranges, nuisance=True)
            rendered = render_group(images[idx], T, P.LEVEL_SPACING[level])
            groups.append({"images": rendered, "T": T, "T_target": tgt,
                           "A_star": cls._targets(tgt, target_iters), "index": idx})
        return cls(task, level, groups)

    @classmethod
    def from_group(cls, rendered, T, T_target, task, level, target_iters=10):
        """A one-group set from an already rendered group (overfit mode)."""
        g = {"images": np.asarray(rendered, np.float32), "T": np.asarray(T, np.float64),
             "T_target": np.asarray(T_target, np.float64),
             "A_star": cls._targets(T_target, target_iters),
             "index": np.arange(np.asarray(rendered).shape[0])}
        return cls(task, level, [g])

    def subset(self, n_groups):
        return ValidationSet(self.task, self.level, self.groups[:int(n_groups)])


def build_validation_sets(images, tasks, level, group_size, n_groups=None, ranges=None, seed=1234,
                          target_iters=10):
    """{task: ValidationSet.build(...)} for each task, all with the same base ``seed``.

    ``ranges``: {task: ranges dict} or None. Each set depends only on (seed, task, group index),
    so e.g. the rigid set is the same for task ratios 'rigid:1' and 'affine:1,rigid:2'.
    """
    ranges = ranges or {}
    return {t: ValidationSet.build(images, t, level, group_size, n_groups, ranges.get(t), seed=seed,
                                   target_iters=target_iters)
            for t in tasks}
