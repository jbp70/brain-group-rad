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

Deformable data (milestone 6) uses the DALI deform format
(`dali_loader.deform_output_signature`): ``(images float32, labels uint8[, T_dali])``, so the
same `DeformPostprocess` path runs for DALI, overfit and synthetic batches. Synthetic groups are
a label phantom (`synthetic_label_phantom`) pulled through exp(v_i) of smooth random SVFs
(`random_svf`, `models.diffeo`); `DeformValidationSet` holds fixed real (image, label) groups.
"""
import math
import os

import numpy as np
import tensorflow as tf

from data_loading import data_module as dm
from data_loading import params as P
from models import diffeo
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
# Extra seed component of the per-group global transform of the val_<task>_g sets.
VAL_G_SEED = 303
# Name suffix of the validation sets with a global transform: val_<task>_g.
VAL_G_SUFFIX = "_g"


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
              target_iters=10, global_transform=None):
        """Groups g = images[perm[(g Nv + k) % n_val]], k < Nv; default n_groups = ceil(n_val / Nv).

        Seeding depends only on (seed, task, g): the permutation uses (seed, task id) and the
        transforms of group g use (seed, task id, g), so a task's set does not change with the
        order (or presence) of other tasks in the task ratio.

        ``global_transform``: optional dict(rotation_deg, translation_mm) (design 3.2 v3.5). Each
        group then also gets one fixed G (`runtime.curriculum.sample_global_params`, seeded by
        (seed, task id, g, VAL_G_SEED)) composed last, T_i <- G T_i and T_target_i <- G T_target_i,
        with the same groups and per-image transforms as without it. ``G`` and ``G_params`` are
        stored in each group.
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
            extra = {}
            if global_transform is not None:
                from runtime.curriculum import global_matrix_mm, sample_global_params
                gp = sample_global_params(np.random.default_rng([int(seed), tid, g, VAL_G_SEED]),
                                          float(global_transform["rotation_deg"]),
                                          float(global_transform["translation_mm"]))
                G = global_matrix_mm(gp["axis"], gp["angle_deg"], gp["translation_mm"])
                T, tgt = G @ T, G @ tgt
                extra = {"G": G, "G_params": gp}
            rendered = render_group(images[idx], T, P.LEVEL_SPACING[level])
            groups.append(dict({"images": rendered, "T": T, "T_target": tgt,
                                "A_star": cls._targets(tgt, target_iters), "index": idx}, **extra))
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
                          target_iters=10, global_transform=None):
    """{task: ValidationSet.build(...)} for each task, all with the same base ``seed``.

    ``ranges``: {task: ranges dict} or None. Each set depends only on (seed, task, group index),
    so e.g. the rigid set is the same for task ratios 'rigid:1' and 'affine:1,rigid:2'.
    ``global_transform``: optional dict(rotation_deg, translation_mm); adds a set
    ``<task>_g`` per task (same groups and per-image transforms plus one fixed G per group).
    """
    ranges = ranges or {}
    out = {}
    for t in tasks:
        out[t] = ValidationSet.build(images, t, level, group_size, n_groups, ranges.get(t), seed=seed,
                                     target_iters=target_iters)
    if global_transform is not None:
        for t in tasks:
            out[t + VAL_G_SUFFIX] = ValidationSet.build(
                images, t, level, group_size, n_groups, ranges.get(t), seed=seed,
                target_iters=target_iters, global_transform=global_transform)
    return out


# ============================================================================ deformable (m6)
def _smooth_noise(shape, rng, sigma, channels=1):
    """Gaussian-smoothed white noise [X, Y, Z, channels] (float64, numpy FFT, periodic)."""
    shape = [int(v) for v in shape]
    noise = rng.normal(size=[channels] + shape)
    freqs = np.meshgrid(np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1]),
                        np.fft.rfftfreq(shape[2]), indexing="ij")
    transfer = np.exp(-2.0 * (math.pi * float(sigma)) ** 2 * sum(f ** 2 for f in freqs))
    out = np.stack([np.fft.irfftn(np.fft.rfftn(c) * transfer, s=shape) for c in noise], axis=-1)
    return out


def synthetic_label_phantom(shape, rng, n_labels=8):
    """Phantom with a label map -> (image float32 [X, Y, Z] in [0, 1], labels uint8 [X, Y, Z]).

    Foreground = the ellipsoid of `smooth_phantom`, split into ``n_labels`` Voronoi-like cells
    (seeded centres at least ~7 voxels apart at level 4, cell boundaries bent by a smooth noise
    term), labels 1..n_labels, background 0. Every label is present. Image = per-label mean
    intensity (U(0.2, 1)) * 0.7 inside the ellipsoid + 0.3 * the `smooth_phantom` texture.
    """
    shape = [int(v) for v in shape]
    texture, mask = smooth_phantom(shape, rng)
    n_labels = int(n_labels)
    if not 1 <= n_labels <= 254:
        raise ValueError("n_labels must be in [1, 254], got %d" % n_labels)
    g = np.stack(np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij"), -1)
    c = np.asarray([(n - 1) / 2.0 for n in shape])
    semi = np.asarray([0.36 * n for n in shape])
    min_dist = 0.5 * float(semi.min())
    centres = []
    for trial in range(100000):
        if trial % 2000 == 1999:
            min_dist *= 0.9                     # relax if the cells do not fit
        p = c + rng.uniform(-0.8, 0.8, 3) * semi
        if np.sum(((p - c) / semi) ** 2) > 0.6:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in centres):
            centres.append(p)
            if len(centres) == n_labels:
                break
    centres = np.asarray(centres)
    if centres.shape[0] != n_labels:
        raise RuntimeError("could not place %d label centres" % n_labels)
    bend = _smooth_noise(shape, rng, sigma=3.0, channels=n_labels)
    bend = bend / max(float(np.abs(bend).max()), 1e-8) * (0.3 * min_dist)
    dist = np.stack([np.linalg.norm(g - ck, axis=-1) for ck in centres], axis=-1)
    labels = np.where(mask, np.argmin(dist + bend, axis=-1) + 1, 0).astype(np.uint8)
    if np.unique(labels).size != n_labels + 1:  # a cell swallowed by its neighbours: plain Voronoi
        labels = np.where(mask, np.argmin(dist, axis=-1) + 1, 0).astype(np.uint8)
    means = np.concatenate([[0.0], rng.uniform(0.2, 1.0, n_labels)])
    image = 0.7 * means[labels] + 0.3 * texture
    image = np.clip(image / max(float(image.max()), 1e-8), 0.0, 1.0)
    return image.astype(np.float32), labels


def random_svf(shape, rng, max_vox=2.0, sigma_vox=4.0):
    """Smooth random velocity field [X, Y, Z, 3] float32 (voxels): Gaussian-smoothed noise
    (sigma ``sigma_vox``) scaled so that max_x |v(x)| = ``max_vox``."""
    v = _smooth_noise(shape, rng, sigma_vox, channels=3)
    v = v / max(float(np.linalg.norm(v, axis=-1).max()), 1e-12) * float(max_vox)
    return v.astype(np.float32)


def to_deform_batch(images, labels, T=None, level=4):
    """(images float32 [n, X, Y, Z, 1], labels uint8 [n, X, Y, Z, 1][, T_dali float32 [n, 3, 4]])
    tensors in the DALI deform format; ``T``: optional residual affines [n, 4, 4] TF centered mm."""
    level = P.check_level(level)
    shape = tuple(P.LEVEL_SHAPE[level])
    images = np.asarray(images, np.float32).reshape((-1,) + shape + (1,))
    labels = np.asarray(labels).reshape((-1,) + shape + (1,))
    if labels.dtype != np.uint8:
        if labels.min() < 0 or labels.max() > 255:
            raise ValueError("labels must fit in uint8")
        labels = labels.astype(np.uint8)
    if images.shape[0] != labels.shape[0]:
        raise ValueError("images and labels differ in group size")
    out = [tf.constant(images), tf.constant(labels)]
    if T is not None:
        m = tf.constant(np.asarray(T, np.float64)[:, :3, :])
        out.append(tf.cast(tfm.tf_mm_to_dali_voxel(m, shape, P.LEVEL_SPACING[level],
                                                   P.DALI_PIXEL_CENTER), tf.float32))
    return tuple(out)


def synthetic_deform_group(level=4, n=4, seed=0, max_vox=4.0, sigma_vox=4.0, n_labels=8,
                           integration_steps=7):
    """A label phantom deformed by n diffeomorphisms (overfit mode and tests).

    Default max_vox = 4 (level-4 voxels, 16 mm; RMS |u| ~1.5 vox): the identity pairwise hard
    Dice is ~0.75-0.77 and the identity template Dice ~0.85-0.86 (seeds 0-2). max_vox = 2 gives a
    pairwise Dice of ~0.88, too close to aligned for a meaningful overfit check.

    v_i = `random_svf` per image, centred over the group (so the base phantom is, to first
    order, the central space) and rescaled together so that max_i,x |v_i(x)| = ``max_vox``;
    u_i = exp(v_i) (`diffeo.integrate_svf`); image_i = base o phi_i (linear), labels_i =
    base labels o phi_i (nearest). Returns dict(images float32 [n, X, Y, Z, 1], labels uint8
    [n, X, Y, Z, 1], u_true, v_true float32 [n, X, Y, Z, 3], base_image, base_labels,
    batch = `to_deform_batch(images, labels, level=level)`).
    """
    level = P.check_level(level)
    shape = P.LEVEL_SHAPE[level]
    rng = np.random.default_rng(int(seed))
    base_img, base_lab = synthetic_label_phantom(shape, rng, n_labels)
    v = np.stack([random_svf(shape, rng, max_vox, sigma_vox) for _ in range(int(n))]).astype(np.float64)
    v = v - v.mean(axis=0, keepdims=True)
    v = v / max(float(np.linalg.norm(v, axis=-1).max()), 1e-12) * float(max_vox)
    v = v.astype(np.float32)
    u = diffeo.integrate_svf(tf.constant(v), integration_steps)
    imgs = np.broadcast_to(base_img[None, ..., None], (int(n),) + tuple(shape) + (1,))
    labs = np.broadcast_to(base_lab[None, ..., None], (int(n),) + tuple(shape) + (1,))
    images = diffeo.warp(tf.constant(imgs), u).numpy().astype(np.float32)
    labels = diffeo.warp(tf.constant(labs), u, interpolation="nearest").numpy().astype(np.uint8)
    return {"images": images, "labels": labels, "u_true": u.numpy(), "v_true": v,
            "base_image": base_img, "base_labels": base_lab,
            "batch": to_deform_batch(images, labels, level=level)}


def load_level_images_and_labels(export_root, level, split, n=None, seed=0, contrast=None,
                                 label_set=dm.DEFAULT_LABEL_SET, manifest_path=None):
    """Load `split` images and their `label_set` label maps with numpy (same seeded selection
    as `load_level_images`) -> (images float32 [n, X, Y, Z, 1], labels uint8 [n, X, Y, Z, 1],
    image_files, label_files)."""
    level = P.check_level(level)
    rows = dm.read_manifest(manifest_path or os.path.join(export_root, dm.MANIFEST_FILE))
    files = dm.manifest_files(rows, level, split=split, contrast=contrast)
    label_files = dm.manifest_files(rows, level, split=split, kind="labels", label_set=label_set,
                                    contrast=contrast)
    if not files:
        raise ValueError("no %r images at level %d in %s" % (split, level, export_root))
    if n is not None:
        if n > len(files):
            raise ValueError("asked for %d %r images, the export has %d" % (n, split, len(files)))
        sel = np.random.default_rng(int(seed)).choice(len(files), int(n), replace=False)
        files = [files[i] for i in sel]
        label_files = [label_files[i] for i in sel]
    shape = tuple(P.LEVEL_SHAPE[level])
    vols, labs = [], []
    for f, lf in zip(files, label_files):
        v = np.load(os.path.join(export_root, f)).astype(np.float32)
        lab = np.load(os.path.join(export_root, lf))
        for name, arr in ((f, v), (lf, lab)):
            if tuple(arr.shape[:3]) != shape:
                raise ValueError("%s has shape %s, expected %s" % (name, arr.shape, shape))
        vols.append(v.reshape(shape + (1,)))
        labs.append(lab.astype(np.uint8).reshape(shape + (1,)))
    return np.stack(vols), np.stack(labs), files, label_files


# Fixed seed component of the deformable validation groups.
VAL_DEFORM_SEED = 404


class DeformValidationSet:
    """Fixed deformable validation groups (real images with their labels; no augmentation, no
    flips, no residual affine). Each group dict holds ``images`` (float32 [Nv, X, Y, Z, 1]),
    ``labels`` (uint8 [Nv, X, Y, Z, 1]) and ``index``. ``task`` is 'deform'."""

    def __init__(self, level, groups, task=P.DEFORM_TASK):
        self.task = task
        self.level = P.check_level(level)
        self.groups = list(groups)
        self.group_size = int(self.groups[0]["images"].shape[0]) if self.groups else 0

    @classmethod
    def build(cls, images, labels, level, group_size, n_groups=None, seed=1234):
        """Groups g = cases perm[(g Nv + k) % n_val], k < Nv, perm from
        default_rng([seed, VAL_DEFORM_SEED]); default n_groups = ceil(n_val / Nv)."""
        level = P.check_level(level)
        shape = tuple(P.LEVEL_SHAPE[level])
        images = np.asarray(images, np.float32).reshape((-1,) + shape + (1,))
        labels = np.asarray(labels).astype(np.uint8).reshape((-1,) + shape + (1,))
        n_val = images.shape[0]
        if labels.shape[0] != n_val:
            raise ValueError("images and labels differ in count (%d vs %d)" % (n_val, labels.shape[0]))
        nv = int(group_size)
        if not 2 <= nv <= n_val:
            raise ValueError("validation group size must be in [2, n_val=%d], got %d" % (n_val, nv))
        if n_groups is None:
            n_groups = int(math.ceil(n_val / float(nv)))
        perm = np.random.default_rng([int(seed), VAL_DEFORM_SEED]).permutation(n_val)
        groups = []
        for g in range(int(n_groups)):
            idx = np.asarray([perm[(g * nv + k) % n_val] for k in range(nv)])
            groups.append({"images": images[idx], "labels": labels[idx], "index": idx})
        return cls(level, groups)

    @classmethod
    def from_group(cls, images, labels, level):
        """A one-group set from a fixed group (overfit mode)."""
        level = P.check_level(level)
        shape = tuple(P.LEVEL_SHAPE[level])
        images = np.asarray(images, np.float32).reshape((-1,) + shape + (1,))
        labels = np.asarray(labels).astype(np.uint8).reshape((-1,) + shape + (1,))
        return cls(level, [{"images": images, "labels": labels,
                            "index": np.arange(images.shape[0])}])

    def subset(self, n_groups):
        return DeformValidationSet(self.level, self.groups[:int(n_groups)], self.task)
