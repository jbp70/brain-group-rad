"""Intensity augmentation in TF (--intensity-aug-device tf), the same chain as the DALI
`dali_loader.intensity_augment` (`params.IntensityAugConfig`).

Runs inside `LinearPostprocess` / `DeformPostprocess` right after the float32 cast and before
the flips (intensity ops commute with flips; mean / std / min / max are flip-invariant), images
only. Moving it out of DALI frees ~9.5 GB of DALI device memory at level 1 (job 10517362).

Order per image: noise -> brightness -> contrast -> low-res -> gamma -> clamp to [0, 1].

Randomness (fixed draw pattern, independent of the probabilities and enable flags): per call
``u = generator.uniform([N, 16])`` (columns `params.AUG_COLUMNS`) and one stateless noise seed
per image (``generator.make_seeds(N)``). The generator is the post-process's ``aug_generator``
(seed = post-process seed + `params.INTENSITY_AUG_SEED_OFFSET`), checkpointed with the run.
Semantics follow DALI: "apply" = u < prob (`_multiplex`); a factor is with probability 0.5 from
the low side U(low, 1), else from the high side U(1, high) (`_intensity_factor`); the noise std
is U(0, noise_std_max); contrast is clamped to the PRE-contrast min / max; gamma renormalises
to the input mean / population std. Ops with ``cfg.<op> = False`` are not built (as DALI);
their columns are still drawn. Noise, contrast and low-res are skipped with tf.cond when not
applied (bit-exact identity); brightness multiplies by exactly 1; gamma always runs with g = 1
when not applied, mirroring DALI.

Low-res factor: ONE isotropic factor f ~ U(min_factor, 1) per image (in DALI the scalar
``fn.random.uniform`` multiplies the shape vector).

Low-res method (`lowres_resample`): a fused, separable, static-shape NN-down + linear-up. Per
spatial axis of length n: m = max(1, floor(f n + 0.5)); output voxel o maps to the low-res
coordinate c = (o + 0.5) m / n - 0.5 (pixel center 0.5, edge clamp, no antialias, as DALI's
linear upsample); low-res voxel l takes the NN source src(l) = min(floor(start + (l + 0.5) s),
n - 1); out[o] = (1 - w) x[src(l0)] + w x[src(l1)]. NN and linear resizes are tensor-product
operators, so this equals the two-stage 3-D resize (`intensity_ref.lowres_np`) while keeping
the output shape n (traceable, no dynamic shapes). `LOWRES_ROI` selects the NN source mapping:
'subpixel' (s = 1/f, start 0; DALI 1.18 subpixel_scale=True keeps the fractional scale),
'integer' (s = n / m) or 'subpixel_centered' (s = 1/f, ROI of width m/f centered). Which one DALI
uses is measured by GPU check (m1); possible differences from DALI otherwise: the low-res grid
spans the full extent n although its true footprint m/f differs by < 1 voxel (DALI's own
behaviour, mirrored), and float32 vs DALI rounding of the index arithmetic.

dtype: always float32 (explicit dtypes; the mixed_float16 Keras policy does not affect raw ops).
Memory: `augment_group` maps over the images with ``parallel_iterations=1``, so only one
image's temporaries (~4 volumes) are alive, plus the map's output (~2 group tensors).
"""
import tensorflow as tf

from data_loading import params as P

LOWRES_ROI = "subpixel"
LOWRES_ROIS = ("subpixel", "integer", "subpixel_centered")
_C = P.AUG_COLUMNS


def draw_aug_params(generator, n):
    """Per-call draws: (u float32 [n, N_AUG_UNIFORMS], seeds int64 [n, 2])."""
    u = generator.uniform(tf.stack([n, P.N_AUG_UNIFORMS]), dtype=tf.float32)
    seeds = tf.transpose(generator.make_seeds(n))
    return u, seeds


def _factor(u, prefix, prob, low, high):
    """(applied bool, factor float32) from one row of uniforms (`_intensity_factor`)."""
    applied = u[_C[prefix + "_apply"]] < float(prob)
    lo = float(low) + u[_C[prefix + "_lo_val"]] * (1.0 - float(low))
    hi = 1.0 + u[_C[prefix + "_hi_val"]] * (float(high) - 1.0)
    f = tf.where(u[_C[prefix + "_low"]] < 0.5, lo, hi)
    return applied, tf.where(applied, f, tf.constant(1.0, tf.float32))


def _lowres_axis(x, axis, f, roi):
    """One spatial axis of the fused NN-down / linear-up resample (static length n)."""
    n = int(x.shape[axis])
    # A Python float goes straight to float64 (tf.cast would round it through float32 first).
    f64 = tf.cast(f, tf.float64) if tf.is_tensor(f) else tf.constant(float(f), tf.float64)
    m = tf.maximum(tf.floor(f64 * n + 0.5), 1.0)          # float64 holding an integer
    o = tf.range(n, dtype=tf.float64)
    c = (o + 0.5) * m / float(n) - 0.5
    l0 = tf.floor(c)
    w = c - l0
    l0c = tf.clip_by_value(l0, 0.0, m - 1.0)
    l1c = tf.clip_by_value(l0 + 1.0, 0.0, m - 1.0)
    if roi == "integer":
        start, scale = tf.constant(0.0, tf.float64), float(n) / m
    elif roi == "subpixel":
        start, scale = tf.constant(0.0, tf.float64), 1.0 / f64
    else:
        start, scale = 0.5 * (float(n) - m / f64), 1.0 / f64

    def src(l):
        s = tf.floor(start + (l + 0.5) * scale)
        return tf.cast(tf.clip_by_value(s, 0.0, float(n - 1)), tf.int32)

    a = tf.gather(x, src(l0c), axis=axis)
    b = tf.gather(x, src(l1c), axis=axis)
    shape = [1] * x.shape.rank
    shape[axis] = n
    w = tf.reshape(tf.cast(w, x.dtype), shape)
    return (1.0 - w) * a + w * b


def lowres_resample(x, f, roi=None):
    """Simulated low resolution of one image x [X, Y, Z(, C)] with isotropic factor f (scalar):
    NN down to round(f n) per axis, linear back to n (module docstring). Output shape == input."""
    roi = LOWRES_ROI if roi is None else roi
    if roi not in LOWRES_ROIS:
        raise ValueError("roi must be one of {}, got {!r}".format(LOWRES_ROIS, roi))
    x = tf.convert_to_tensor(x, tf.float32)
    for axis in range(3):
        x = _lowres_axis(x, axis, f, roi)
    return x


def contrast(x, factor):
    """(x - mean) factor + mean, clamped to the PRE-contrast [min, max] (DALI)."""
    mean = tf.reduce_mean(x)
    return tf.clip_by_value((x - mean) * factor + mean, tf.reduce_min(x), tf.reduce_max(x))


def gamma(x, g):
    """`dali_loader._gamma`: pow on the [min, max]-normalised image, then the input mean and
    population std restored."""
    m0 = tf.reduce_mean(x)
    s0 = tf.sqrt(tf.reduce_mean(tf.square(x - m0)))
    lo = tf.reduce_min(x)
    rng = tf.reduce_max(x) - lo
    out = tf.pow((x - lo) / tf.maximum(rng, 1e-7), g) * rng + lo
    m1 = tf.reduce_mean(out)
    s1 = tf.sqrt(tf.reduce_mean(tf.square(out - m1)))
    return (out - m1) * (s0 / tf.maximum(s1, 1e-7)) + m0


def augment_image(x, u, seed, cfg, roi=None):
    """Full chain for one image x float32 [X, Y, Z(, C)]; u float32 [N_AUG_UNIFORMS] (one row of
    `draw_aug_params`); seed int64 [2] (stateless noise seed). Returns float32, same shape."""
    x = tf.convert_to_tensor(x, tf.float32)
    if cfg is None or not cfg.enabled:
        return x
    if cfg.noise:
        on = u[_C["noise_apply"]] < float(cfg.noise_prob)
        std = u[_C["noise_std"]] * float(cfg.noise_std_max)
        x = tf.cond(on, lambda x=x: x + std * tf.random.stateless_normal(tf.shape(x), seed, dtype=tf.float32),
                    lambda x=x: x)
    if cfg.brightness:
        _, b = _factor(u, "brightness", cfg.brightness_prob, *cfg.brightness_range)
        x = x * b
    if cfg.contrast:
        on, c = _factor(u, "contrast", cfg.contrast_prob, *cfg.contrast_range)
        x = tf.cond(on, lambda x=x: contrast(x, c), lambda x=x: x)
    if cfg.low_res:
        on = u[_C["low_res_apply"]] < float(cfg.low_res_prob)
        # float64 factor: the index arithmetic is float64 (identical to the numpy reference).
        f = float(cfg.low_res_min_factor) + tf.cast(u[_C["low_res_factor"]], tf.float64) * (
            1.0 - float(cfg.low_res_min_factor))
        x = tf.cond(on, lambda x=x: lowres_resample(x, f, roi), lambda x=x: x)
    if cfg.gamma:
        _, g = _factor(u, "gamma", cfg.gamma_prob, *cfg.gamma_range)
        x = gamma(x, g)
    return tf.clip_by_value(x, 0.0, 1.0)


def augment_group(images, cfg, generator, roi=None):
    """Augment a group images [N, X, Y, Z, C] (float32), independent draws per image.

    Returns the images unchanged (no draw, no op) if ``cfg`` is None or disabled. One image at a
    time (tf.map_fn, parallel_iterations=1) bounds the transient memory.
    """
    if cfg is None or not cfg.enabled:
        return images
    with tf.name_scope("intensity_aug"):
        images = tf.convert_to_tensor(images, tf.float32)
        u, seeds = draw_aug_params(generator, tf.shape(images)[0])
        return tf.map_fn(lambda e: augment_image(e[0], e[1], e[2], cfg, roi), (images, u, seeds),
                         parallel_iterations=1,
                         fn_output_signature=tf.TensorSpec(images.shape[1:], tf.float32))
