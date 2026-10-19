"""Post-DALI processing in TF (design 3.3), run in the train step outside the GradientTape.

Matrix flow for one linear batch (group of N images):
1. DALI emits forward matrices T_dali (and Rt_dali for rigid): 3x4, DALI axis order, DALI
   voxel coordinates (`data_loading.dali_loader`).
2. Conversion to TF centered mm (`models.transforms.dali_voxel_to_tf_mm`):
       T_mm = W (P T_dali P) W^-1,  W: DALI coordinate u (TF order) -> x = s (u - pc - (n - 1) / 2)
   so that image i satisfies I_i(y) = I(T_mm^-1 y) (pull), i.e.
   `transform_image(I, T_mm^-1)` renders what DALI rendered and `transform_image(I_i, T_mm)`
   undoes it.
3. One flip state F = diag(f_x, f_y, f_z, 1) per group: images <- tf.reverse on the flipped
   axes, T <- F T F (render and target matrices alike).
4. Targets: A_i* = barycentric_targets(T_target) = mu*^-1 T_i^-1 with
   mu* = barycenter{T_i^-1}; T_target = Rt (rigid: excludes the nuisance S_i) or T (affine).
   Pair targets P_ij* = T_target_j T_target_i^-1 for sampled ordered pairs (i, j), i != j.
5. Global transform G (design 3.2 v3.5, only with a control file): DALI already emits
   T_dali = G Rt_i S_i and Rt_dali = G Rt_i, so with the steps above
       T_target = F (G Rt_i) F (rigid) or F (G Rt_i S_i) F (affine),
       A_i* = mu*^-1 T_target_i^-1,  mu* = barycenter{T_target_i^-1},  P_ij* = T_target_j T_target_i^-1.
   The rigid pair targets become G Rt_j Rt_i^-1 G^-1 (the relative pose, conjugated by G), and
   the barycenter moves with G (mu* = mu*(G = I) G^-1 for a right-equivariant barycenter), so
   the central space is no longer close to the template pose. Nothing in this module changes
   for G: the formulas take whatever DALI emitted. The trainer passes only the first 3 (rigid)
   / 2 (affine) DALI outputs here; G_dali and control_version are for logging.

Everything is batched over the group axis and works inside tf.function. Randomness comes from
a seeded `tf.random.Generator` (deterministic for a given seed and call sequence).
"""
import os

import numpy as np
import tensorflow as tf

from data_loading import params as P
from models import transforms as tfm


def dali_to_mm(m_dali, level, pixel_center=P.DALI_PIXEL_CENTER):
    """DALI voxel matrices [..., 3|4, 4] -> TF centered-mm matrices [..., 4, 4]."""
    level = P.check_level(level)
    out = tfm.dali_voxel_to_tf_mm(m_dali, P.LEVEL_SHAPE[level], P.LEVEL_SPACING[level],
                                  pixel_center=pixel_center)
    return tfm.to_homogeneous(out)


def sample_flip_axes(generator, flip_prob=(0.5, 0.5, 0.5)):
    """One flip state per group: bool [3], axis a flipped with probability flip_prob[a]."""
    probs = np.broadcast_to(np.asarray(flip_prob, np.float32), (3,))
    return generator.uniform([3], dtype=tf.float32) < tf.constant(probs)


def flip_matrix_dynamic(flip_axes, dtype=tf.float32):
    """F = diag(+-1, +-1, +-1, 1) from a bool tensor [3] (tensor version of tfm.flip_matrix)."""
    signs = tf.where(tf.convert_to_tensor(flip_axes, tf.bool), tf.constant(-1.0, dtype),
                     tf.constant(1.0, dtype))
    return tf.linalg.diag(tf.concat([signs, tf.ones([1], dtype)], axis=0))


def apply_flip(images, matrices, flip_axes):
    """Flip a group: images [N, X, Y, Z, C] reversed on the flagged axes, each matrix -> F M F.

    ``flip_axes``: bool [3] (Python or tensor). ``matrices``: list of [N, 3|4, 4] mm matrices.
    Returns (images, [F M F, ...]) with 4x4 matrices.
    """
    flags = tf.convert_to_tensor(flip_axes, tf.bool)
    for a in range(3):
        images = tf.cond(flags[a], lambda a=a, x=images: tf.reverse(x, axis=[a + 1]),
                         lambda x=images: x)
    out = []
    for m in matrices:
        m = tfm.to_homogeneous(m)
        out.append(tfm.flip_conjugate(m, flip_matrix_dynamic(flags, m.dtype)))
    return images, out


def num_pairs(n, p_pair=0.1):
    """Number of sampled ordered pairs: max(1, round(p_pair * n (n - 1))) (int32 tensor)."""
    n = tf.cast(n, tf.int32)
    total = n * (n - 1)
    k = tf.cast(tf.round(tf.cast(total, tf.float32) * float(p_pair)), tf.int32)
    return tf.minimum(tf.maximum(k, 1), total)


def sample_pairs(generator, n, p_pair=0.1):
    """Sample distinct ordered pairs (i, j), i != j, without replacement. Returns int32 (i, j).

    Draws a uniform random key per ordered pair (flat index k = i (n - 1) + r) and keeps the
    `num_pairs` largest keys (top_k), so the result depends only on the generator state.
    Needs n >= 2.
    """
    n = tf.cast(n, tf.int32)
    total = n * (n - 1)
    keys = generator.uniform(tf.reshape(total, [1]), dtype=tf.float32)
    _, k = tf.math.top_k(keys, k=num_pairs(n, p_pair), sorted=True)
    i = k // (n - 1)
    r = k % (n - 1)
    j = r + tf.cast(r >= i, tf.int32)
    return i, j


class LinearPostprocess:
    """Post-DALI processing of one rigid/affine batch.

    Args:
      task: 'rigid' or 'affine'.
      level: 4, 2 or 1.
      flip_prob: per-axis flip probability (scalar or 3 values); 0 disables flips.
      p_pair: fraction of ordered pairs i != j sampled for the pairwise loss (at least 1).
      seed: seed of the internal tf.random.Generator (flips and pairs).
      target_iters: barycenter iterations for the targets (run to convergence).
      target_dtype: dtype of the target computation (float64 default; results are float32).
      pixel_center: DALI pixel-center convention (`params.DALI_PIXEL_CENTER`).

    Call with the DALI outputs ``(images, T_dali[, Rt_dali])``; the rigid task requires Rt_dali.
    ``flip_axes`` overrides the random flip state (bool [3]). Returns a dict:
      images   float32 [N, X, Y, Z, C] (flipped)
      T_render float32 [N, 4, 4] mm, the full T_i (flipped): I_i(y) = I(T_render_i^-1 y)
      T_target float32 [N, 4, 4] mm, Rt_i (rigid) or T_i (affine) (flipped)
      A_star   float32 [N, 4, 4] barycentric targets of T_target (stop_gradient)
      pair_i, pair_j int32 [P] sampled ordered pairs
      P_star   float32 [P, 4, 4] pair targets T_target_j T_target_i^-1
      flip_axes bool [3]
    """

    def __init__(self, task, level, flip_prob=0.5, p_pair=0.1, seed=0, target_iters=10,
                 target_dtype=tf.float64, pixel_center=P.DALI_PIXEL_CENTER):
        if task not in P.TASKS:
            raise ValueError("task must be one of {}, got {!r}".format(P.TASKS, task))
        self.task = task
        self.level = P.check_level(level)
        self.flip_prob = np.broadcast_to(np.asarray(flip_prob, np.float32), (3,)).copy()
        self.p_pair = float(p_pair)
        self.target_iters = int(target_iters)
        self.target_dtype = target_dtype
        self.pixel_center = float(pixel_center)
        self.generator = tf.random.Generator.from_seed(int(seed))

    def __call__(self, images, t_dali, rt_dali=None, flip_axes=None):
        images = tf.convert_to_tensor(images)
        if images.dtype != tf.float32:
            images = tf.cast(images, tf.float32)
        t_mm = dali_to_mm(t_dali, self.level, self.pixel_center)
        if self.task == "rigid":
            if rt_dali is None:
                raise ValueError("the rigid task needs Rt_dali")
            target_mm = dali_to_mm(rt_dali, self.level, self.pixel_center)
        else:
            target_mm = t_mm
        if flip_axes is None:
            flip_axes = sample_flip_axes(self.generator, self.flip_prob)
        else:
            flip_axes = tf.convert_to_tensor(flip_axes, tf.bool)
        images, (t_mm, target_mm) = apply_flip(images, [t_mm, target_mm], flip_axes)
        tgt = tf.cast(target_mm, self.target_dtype)
        a_star = tfm.barycentric_targets(tgt, self.target_iters)
        n = tf.shape(images)[0]
        pair_i, pair_j = sample_pairs(self.generator, n, self.p_pair)
        p_star = tfm.pair_targets(tgt, pair_i, pair_j)
        return {"images": images,
                "T_render": tf.cast(t_mm, tf.float32),
                "T_target": tf.cast(target_mm, tf.float32),
                "A_star": tf.cast(a_star, tf.float32),
                "pair_i": pair_i, "pair_j": pair_j,
                "P_star": tf.cast(p_star, tf.float32),
                "flip_axes": flip_axes}


def brain_S_matrix(level, template_dir):
    """S = E[h h^T] over the template brain mask of `level` (homogeneous centered mm, float64).

    Loads ``<template_dir>/template_mask<suffix>.npy`` (suffix '', '_2', '_4') and calls
    `models.transforms.brain_second_moment` with the level's spacing.
    """
    level = P.check_level(level)
    path = os.path.join(template_dir, "template_mask%s.npy" % P.level_suffix(level))
    mask = np.load(path)
    if tuple(mask.shape[:3]) != tuple(P.LEVEL_SHAPE[level]):
        raise ValueError("%s has shape %s, expected %s for level %d"
                         % (path, mask.shape, P.LEVEL_SHAPE[level], level))
    return tfm.brain_second_moment(mask, P.LEVEL_SPACING[level])
