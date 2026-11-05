"""SynthMorph-style synthetic-contrast images from fine label maps (synth-contrast design 4.1-4.4).

Opt-in (``--synth-prob p > 0``): `data_loading.postprocess` calls `draw_synth_params` and
`maybe_synthesize` right after the float32 cast, BEFORE the TF intensity augmentation and the
flips. With probability p per call (= per group) every image of the group is replaced in place
by an image synthesised from its fine label map (`synthfine`, 103 contiguous SynthSeg --parc ids,
uint8) with ONE contrast shared by the group plus a small per-image jitter. The switch is a
boolean tensor consumed by one `tf.cond`, so real and synthetic groups go through the same traced
graph; the false branch returns the input tensor (real groups are bit-exact). Synthesis off
(`SynthConfig` None or prob 0) builds no op and no generator (`postprocess._synth_setup`).

Group contrast (design 4.3, hierarchical only; orchestrator decision OD1):
  class mean mu_c ~ U(0, 1) and class std sd_c ~ U(0, std_max) per train18 class c (left/right
  and all cortex parcels share a class), fine-id mean
  mu_f = clip(mu_c[g(f)] + U(-h_f, h_f), 0, 1) with g = `fine_to_train18_table` and the
  half-width h_f = hier_dev_cortex (0.25) for the 72 cortical fine ids
  (`preprocessing.luts.synthfine_cortex_ids`: train18 class 2) and hier_dev (0.1) for the other
  fine ids; sd_f = sd_c[g(f)]. Background (id 0) and ids >= 103 have mu = sd = 0.

Per-image chain (design 4.2; `synthesize_image`, one image at a time in `tf.map_fn` with
parallel_iterations 1, like `tf_intensity_aug.augment_group`):
  1. x = mu[l] (1 + d_i[l]) + sd[l] (1 + d_i[l]) n(x), d_i ~ U(-jitter, jitter) per fine id
     (two gathers of 256-entry tables; NO one-hot of the fine labels anywhere).
  2. Separable Gaussian blur: three in-graph [n, n] operators with rows
     exp(-(o - j)^2 / (2 sigma^2)) normalised to sum 1 (renormalised truncation at the borders),
     sigma_vox = sqrt((FWHM / 2.3548)^2 + s^2 / 12) / s (s = level spacing; the s^2 / 12 term is
     the box partial-volume blur of the block-averaged real images), applied by tensordot
     (`tf_intensity_aug._apply_axis_matrix`). FWHM per group from the MR family U(0, 3) mm or,
     with probability pet_family_prob, the PET family U(4, 8) mm, then x U(0.9, 1.1) per image
     and axis.
  3. Re-mask (group draw < remask_prob): x *= (l > 0).
  4. Bias: x *= exp(field), field = per-image normals on a bias_grid^3 grid times the group std
     U(0, bias_std_max), upsampled by three static numpy-built [n, g] linear-interpolation
     matrices whose control points sit at fixed fractions of the physical extent (identical in
     mm at every level).
  5. Robust normalisation like the real images (in-mask p99.5 then clip): hi = in-mask
     norm_quantile quantile of the stride-4 subsample (`tf.boolean_mask` + `tf.math.top_k`,
     dynamic k = max(1, round((1 - q) size))), guarded by tf.cond(size > 0, ..., 1.0);
     x = clip(x / max(hi, 1e-6), 0, 1).
  6. Gamma: x = x^gamma, gamma = exp(N(0, gamma_log_std)) per group.
  7. Noise: x += U(0, noise_std_max) n(x) per image; clip [0, 1]; then the re-mask of step 3
     again (noise stays zero outside the mask on re-masked groups).

Linear groups are synthesised in template space and rendered by the caller (``render`` of
`maybe_synthesize`: `transform_image` with T_mm^-1, the pull relation of DALI's warp), deform
groups are synthesised from the (unwarped or residual-warped) fine labels directly.

Randomness (design 4.4): `draw_synth_params` consumes a FIXED draw pattern per call from the
post-process's ``synth_generator`` (seed + `params.SYNTH_SEED_OFFSET`), independent of p and of
the flag value: group uniform([N_SYNTH_GROUP_UNIFORMS]) (columns `params.SYNTH_COLUMNS`), one
group normal([1]) (log-gamma), group tables uniform([3, 256]) (row 0 class means and row 1 class
stds indexed by train18 id, row 2 parcel deviations indexed by fine id), per image
uniform([N, 256]) (jitter), uniform([N, 3]) (FWHM jitter), uniform([N, 1]) (noise std),
normal([N, g^3]) (bias grid) and make_seeds(N) twice (stateless normals of steps 1 and 7).

Memory (design 6.1): one image's temporaries at a time (~6 float32 volumes: L1 ~165 MB, L4
~2.6 MB); the synthesised group replaces the input images in place; the linear render adds one
image's `transform_image` forward (~48 B/voxel) and the unwarped group until it is rendered. All
of it is freed before the model forward. ``cfg.device == 'cpu'`` places the true branch on
/cpu:0 (fallback, design 6.4); the flag and the tf.cond stay on the default device.

dtype: float32 everywhere (explicit; the mixed_float16 Keras policy does not touch raw ops).

Evaluation presets (post-training synthetic-contrast evaluation, design section 8, OD1;
`EVAL_PRESETS`, `preset_params`): 11 named contrasts, each turned into the params dict that
`synthesize_group` consumes, deterministic from (name, seed) (`preset_seed` ->
tf.random.Generator.from_seed). ``random_0..3``: the training distribution (`draw_synth_params`
with the given SynthConfig and prob 1, i.e. hier_dev 0.1 / hier_dev_cortex 0.25 with the defaults);
the index only changes the seed. The other 7 are fixed train18 class means (index = train18 id,
background 0) plus optional DK-parcel overrides (both hemispheres, resolved to synthfine ids by
`parcel_fine_ids`; no random parcel deviation), the within-class std (U(0, std_max) per class),
per-image jitter, per-image FWHM jitter, bias field and noise drawn as in training, re-mask on,
gamma 1. Blur: MR presets draw the group FWHM from the MR family U(fwhm_mr_mm), PET presets use a
fixed FWHM. Class means (train18 ids 1..18: WM, cortex, lat vent, inf lat vent, cbl WM, cbl
cortex, thalamus, caudate, putamen, pallidum, 3rd vent, 4th vent, brainstem, hippocampus,
amygdala, CSF, accumbens, ventral DC):

  t1_like     .85 .55 .15 .20 .80 .55 .75 .65 .70 .80 .15 .15 .75 .55 .55 .15 .60 .75  MR
  t2_like     .30 .55 .95 .90 .30 .55 .45 .50 .45 .35 .95 .95 .40 .60 .55 .95 .50 .40  MR
  flair_like  .45 .65 .05 .10 .45 .60 .55 .60 .55 .50 .05 .05 .50 .70 .65 .05 .60 .50  MR
  fdg_like    .35 .90 .05 .10 .35 .75 .80 .80 .90 .60 .05 .05 .50 .60 .60 .05 .80 .50  PET 5 mm
  amyloid_pos .85 .80 .05 .10 .75 .30 .60 .75 .75 .65 .05 .05 .70 .45 .50 .05 .70 .60  PET 6 mm
              parcels precuneus, posteriorcingulate, medialorbitofrontal .90
  amyloid_neg .85 .35 .05 .10 .75 .30 .45 .40 .45 .55 .05 .05 .70 .35 .35 .05 .40 .55  PET 6 mm
  tau_like    .25 .30 .02 .10 .25 .20 .40 .30 .45 .50 .02 .02 .30 .50 .60 .02 .35 .40  PET 6 mm
              parcels entorhinal .90, parahippocampal .80, fusiform .70,
              inferiortemporal .70, middletemporal .60
"""
import dataclasses
import zlib

import numpy as np
import tensorflow as tf

from data_loading import params as P
from data_loading import tf_intensity_aug as tfa
from preprocessing import luts

TABLE_SIZE = 256                                     # uint8 label ids index 256-entry tables
N_FINE = len(luts.LUTS["synthfine"]["map"])          # 103 contiguous synthfine ids
FWHM_PER_SIGMA = 2.3548                              # 2 sqrt(2 ln 2)
NORM_STRIDE = 4                                      # subsample stride of step 5
_C = P.SYNTH_COLUMNS


def fine_to_train18_table():
    """int32 [256]: train18 id of every synthfine id (`luts.synthfine_to_train18_table`);
    entries 103..255 = 0."""
    return tf.constant(luts.synthfine_to_train18_table().astype(np.int32), tf.int32)


def dev_halfwidth_np(cfg):
    """float32 [256] numpy: deviation half-width per fine id (OD1): cfg.hier_dev_cortex for the
    cortical ids (`luts.synthfine_cortex_ids`), cfg.hier_dev for the other ids 1..102, 0 for id 0
    and ids >= 103."""
    out = np.zeros(TABLE_SIZE, np.float32)
    out[1:N_FINE] = float(cfg.hier_dev)
    out[luts.synthfine_cortex_ids()] = float(cfg.hier_dev_cortex)
    return out


def dev_halfwidth_table(cfg):
    """`dev_halfwidth_np` as a float32 [256] tensor."""
    return tf.constant(dev_halfwidth_np(cfg), tf.float32)


def draw_synth_params(generator, n, cfg):
    """One call's draws (fixed pattern, module docstring) -> dict of tensors:

      flag      bool []          u_flag < cfg.prob (synthetic group)
      mean      float32 [256]    fine-id mean table (entry 0 and ids >= 103: 0)
      std       float32 [256]    fine-id within-label std table (entry 0 and ids >= 103: 0)
      cls_mean  float32 [256]    class means indexed by train18 id (row 0 of the tables)
      jitter    float32 [n, 256] per-image relative deviation d_i ~ U(-jitter, jitter)
      fwhm_mm   float32 [n, 3]   per-image, per-axis blur FWHM (mm)
      pet_family bool []         PET blur family drawn
      remask    bool []          re-mask the group
      gamma     float32 []       group gamma
      noise_std float32 [n]      per-image noise std
      bias      float32 [n, g^3] per-image log bias-field control values (group std applied)
      seeds     int64 [n, 2, 2]  per-image stateless seeds (step 1, step 7)

    ``n`` is the group size (Python int or int32 scalar tensor).
    """
    f32 = tf.float32
    g3 = int(cfg.bias_grid) ** 3
    u = generator.uniform([P.N_SYNTH_GROUP_UNIFORMS], dtype=f32)
    z_gamma = generator.normal([1], dtype=f32)
    tables = generator.uniform([3, TABLE_SIZE], dtype=f32)
    u_jit = generator.uniform(tf.stack([n, TABLE_SIZE]), dtype=f32)
    u_fwhm = generator.uniform(tf.stack([n, 3]), dtype=f32)
    u_noise = generator.uniform(tf.stack([n, 1]), dtype=f32)
    z_bias = generator.normal(tf.stack([n, g3]), dtype=f32)
    seeds_x = tf.transpose(generator.make_seeds(n))
    seeds_n = tf.transpose(generator.make_seeds(n))

    g = fine_to_train18_table()
    fg = tf.cast(g > 0, f32)
    cls_mean, cls_std, dev = tables[0], tables[1], tables[2]
    half = dev_halfwidth_table(cfg)
    mean = tf.clip_by_value(tf.gather(cls_mean, g) + 2.0 * half * (dev - 0.5), 0.0, 1.0) * fg
    std = tf.gather(float(cfg.std_max) * cls_std, g) * fg

    pet = u[_C["blur_family"]] < float(cfg.pet_family_prob)
    ub = u[_C["blur_base"]]
    mr_lo, mr_hi = [float(v) for v in cfg.fwhm_mr_mm]
    pet_lo, pet_hi = [float(v) for v in cfg.fwhm_pet_mm]
    base = tf.where(pet, pet_lo + ub * (pet_hi - pet_lo), mr_lo + ub * (mr_hi - mr_lo))
    fwhm = base * (1.0 + float(cfg.fwhm_image_jitter) * (2.0 * u_fwhm - 1.0))
    bias_std = u[_C["bias_std"]] * float(cfg.bias_std_max)
    return {
        "flag": u[_C["flag"]] < float(cfg.prob),
        "mean": mean,
        "std": std,
        "cls_mean": cls_mean,
        "jitter": float(cfg.jitter) * (2.0 * u_jit - 1.0),
        "fwhm_mm": fwhm,
        "pet_family": pet,
        "remask": u[_C["remask"]] < float(cfg.remask_prob),
        "gamma": tf.exp(float(cfg.gamma_log_std) * z_gamma[0]),
        "noise_std": u_noise[:, 0] * float(cfg.noise_std_max),
        "bias": bias_std * z_bias,
        "seeds": tf.stack([seeds_x, seeds_n], axis=1),
    }


def gaussian_operator(n, fwhm_mm, spacing):
    """[n, n] float32 blur operator of one axis (step 2): rows exp(-(o - j)^2 / (2 sigma^2))
    normalised to sum 1, sigma_vox = sqrt((FWHM / 2.3548)^2 + s^2 / 12) / s."""
    s = float(spacing)
    fwhm = tf.cast(fwhm_mm, tf.float32)
    sigma = tf.sqrt(tf.square(fwhm / FWHM_PER_SIGMA) + s * s / 12.0) / s
    o = tf.range(int(n), dtype=tf.float32)
    k = tf.exp(-tf.square(o[:, None] - o[None, :]) / (2.0 * tf.square(sigma)))
    return k / tf.reduce_sum(k, axis=1, keepdims=True)


def bias_interp_matrix(n, g):
    """Static float32 numpy [n, g] linear interpolation from g control points to n voxels.

    Control point k sits at the fraction k / (g - 1) of the physical extent [-n s / 2, n s / 2],
    voxel p (centre (p + 0.5) / n of the extent) gets t = (p + 0.5) (g - 1) / n, so the control
    points are at the same mm positions at every level (step 4)."""
    n, g = int(n), int(g)
    m = np.zeros((n, g), np.float64)
    if g == 1:
        m[:, 0] = 1.0
        return m.astype(np.float32)
    t = (np.arange(n, dtype=np.float64) + 0.5) * (g - 1) / float(n)
    l0 = np.minimum(np.floor(t).astype(np.int64), g - 2)
    w = t - l0
    rows = np.arange(n)
    m[rows, l0] = 1.0 - w
    m[rows, l0 + 1] = w
    return m.astype(np.float32)


def bias_field(bias_i, shape, g):
    """Log bias field float32 [X, Y, Z] from control values bias_i [g^3] (step 4)."""
    c = tf.reshape(tf.cast(bias_i, tf.float32), [int(g)] * 3)
    for a in range(3):
        c = tfa._apply_axis_matrix(c, tf.constant(bias_interp_matrix(shape[a], g)), a)
    return c


def robust_normalise(x, idx, quantile):
    """Step 5: x / max(hi, 1e-6) clipped to [0, 1], hi = in-mask `quantile` of the stride-4
    subsample (top_k with dynamic k = max(1, round((1 - quantile) size))); empty mask -> hi 1."""
    st = NORM_STRIDE
    v = tf.boolean_mask(x[::st, ::st, ::st], idx[::st, ::st, ::st] > 0)
    size = tf.size(v)
    k = tf.maximum(1, tf.cast(tf.round(tf.cast(size, tf.float32) * (1.0 - float(quantile))), tf.int32))
    hi = tf.cond(size > 0, lambda: tf.reduce_min(tf.math.top_k(v, k=k).values),
                 lambda: tf.constant(1.0, tf.float32))
    return tf.clip_by_value(x / tf.maximum(hi, 1e-6), 0.0, 1.0)


def contrast_image(idx, mean, std, d_i, seed):
    """Step 1: x = mean[l] (1 + d_i[l]) + std[l] (1 + d_i[l]) n(x) (gathers only)."""
    scale = 1.0 + d_i
    m = tf.gather(mean * scale, idx)
    s = tf.gather(std * scale, idx)
    return m + s * tf.random.stateless_normal(tf.shape(idx), seed, dtype=tf.float32)


def synthesize_image(labels_u8, d_i, fwhm_i, noise_std_i, bias_i, seeds_i, prm, cfg, level):
    """Design 4.2 steps 1-7 for one image: labels uint8 [X, Y, Z] -> float32 [X, Y, Z, 1] in
    [0, 1]. Per-image draws: d_i [256], fwhm_i [3] mm, noise_std_i [], bias_i [g^3], seeds_i
    [2, 2]; group draws from ``prm`` (`draw_synth_params`)."""
    spacing = P.LEVEL_SPACING[P.check_level(level)]
    idx = tf.cast(labels_u8, tf.int32)
    shape = idx.shape.as_list()
    mask = tf.cast(idx > 0, tf.float32)
    x = contrast_image(idx, prm["mean"], prm["std"], d_i, seeds_i[0])
    for a in range(3):
        x = tfa._apply_axis_matrix(x, gaussian_operator(shape[a], fwhm_i[a], spacing), a)
    rm = tf.cast(prm["remask"], tf.float32)
    keep = 1.0 - rm + rm * mask                       # exactly 1 or exactly the mask
    x = x * keep
    x = x * tf.exp(bias_field(bias_i, shape, cfg.bias_grid))
    x = robust_normalise(x, idx, cfg.norm_quantile)
    x = tf.pow(x, prm["gamma"])
    x = x + noise_std_i * tf.random.stateless_normal(tf.shape(idx), seeds_i[1], dtype=tf.float32)
    x = tf.clip_by_value(x, 0.0, 1.0) * keep
    return x[..., None]


def synthesize_group(labels_u8, prm, cfg, level):
    """Synthesise a group: labels uint8 [N, X, Y, Z(, 1)] -> float32 [N, X, Y, Z, 1], one image
    at a time (tf.map_fn, parallel_iterations=1)."""
    with tf.name_scope("synth"):
        lab = tf.convert_to_tensor(labels_u8)
        if lab.shape.rank == 5:
            lab = lab[..., 0]
        spatial = lab.shape[1:4].as_list()
        elems = (lab, prm["jitter"], prm["fwhm_mm"], prm["noise_std"], prm["bias"], prm["seeds"])
        return tf.map_fn(
            lambda e: synthesize_image(e[0], e[1], e[2], e[3], e[4], e[5], prm, cfg, level),
            elems, parallel_iterations=1,
            fn_output_signature=tf.TensorSpec(spatial + [1], tf.float32))


def maybe_synthesize(images, labels_u8, flag, prm, cfg, level, render=None):
    """tf.cond(flag, synthetic group, images): the false branch returns ``images`` unchanged.

    ``render`` (optional callable float32 [N, X, Y, Z, 1] -> same) is applied to the synthetic
    group inside the true branch (the linear post-process renders it with T_mm^-1). With
    ``cfg.device == 'cpu'`` the whole true-branch body runs under tf.device('/cpu:0')."""
    images = tf.convert_to_tensor(images)

    def synth_branch():
        out = synthesize_group(labels_u8, prm, cfg, level)
        return render(out) if render is not None else out

    def true_fn():
        if cfg.device == "cpu":
            with tf.device("/cpu:0"):
                return synth_branch()
        return synth_branch()

    return tf.cond(flag, true_fn, lambda: images)


# ============================================================================ evaluation presets
def _named(family, fwhm_mm, class_means, parcels=None):
    if len(class_means) != P.N_TRAIN_LABELS - 1:
        raise ValueError("a preset needs %d class means, got %d" % (P.N_TRAIN_LABELS - 1, len(class_means)))
    return {"kind": "named", "family": family, "fwhm_mm": fwhm_mm,
            "class_means": [0.0] + [float(v) for v in class_means], "parcels": dict(parcels or {})}


# train18 class means (ids 1..18, module docstring); "class_means" holds 19 entries (id 0 = 0).
EVAL_PRESETS = {
    "random_0": {"kind": "random", "index": 0},
    "random_1": {"kind": "random", "index": 1},
    "random_2": {"kind": "random", "index": 2},
    "random_3": {"kind": "random", "index": 3},
    "t2_like": _named("mr", None, [.30, .55, .95, .90, .30, .55, .45, .50, .45, .35, .95, .95, .40,
                                   .60, .55, .95, .50, .40]),
    "flair_like": _named("mr", None, [.45, .65, .05, .10, .45, .60, .55, .60, .55, .50, .05, .05,
                                      .50, .70, .65, .05, .60, .50]),
    "fdg_like": _named("pet", 5.0, [.35, .90, .05, .10, .35, .75, .80, .80, .90, .60, .05, .05, .50,
                                    .60, .60, .05, .80, .50]),
    "amyloid_pos": _named("pet", 6.0, [.85, .80, .05, .10, .75, .30, .60, .75, .75, .65, .05, .05,
                                       .70, .45, .50, .05, .70, .60],
                          {"precuneus": .90, "posteriorcingulate": .90, "medialorbitofrontal": .90}),
    "amyloid_neg": _named("pet", 6.0, [.85, .35, .05, .10, .75, .30, .45, .40, .45, .55, .05, .05,
                                       .70, .35, .35, .05, .40, .55]),
    "tau_like": _named("pet", 6.0, [.25, .30, .02, .10, .25, .20, .40, .30, .45, .50, .02, .02, .30,
                                    .50, .60, .02, .35, .40],
                       {"entorhinal": .90, "parahippocampal": .80, "fusiform": .70,
                        "inferiortemporal": .70, "middletemporal": .60}),
    "t1_like": _named("mr", None, [.85, .55, .15, .20, .80, .55, .75, .65, .70, .80, .15, .15, .75,
                                   .55, .55, .15, .60, .75]),
}
PRESET_NAMES = tuple(EVAL_PRESETS)


def parcel_fine_ids(name):
    """Synthfine ids (sorted int list) of the DK parcel ``name`` in both hemispheres
    (FreeSurfer ids 1000 + k and 2000 + k, k = 1-based index in `luts.DK_PARCELS`)."""
    if name not in luts.DK_PARCELS:
        raise ValueError("unknown DK parcel %r" % (name,))
    k = luts.DK_PARCELS.index(name) + 1
    fine = luts.LUTS["synthfine"]["map"]
    return sorted(int(fine[base + k]) for base in (1000, 2000))


def preset_seed(name, seed):
    """Deterministic generator seed of (preset name, seed): crc32(name) * 1000003 + seed."""
    return (zlib.crc32(name.encode("ascii")) * 1000003 + int(seed)) % (2 ** 62)


def preset_mean_table_np(name):
    """float32 [256] numpy fine-id mean table of a named preset: the class mean of every fine id
    (`fine_to_train18_table`) with the parcel overrides; id 0 and ids >= 103 are 0."""
    spec = EVAL_PRESETS[name]
    if spec["kind"] != "named":
        raise ValueError("preset %r has no fixed mean table" % name)
    g = luts.synthfine_to_train18_table().astype(np.int64)
    cls = np.asarray(spec["class_means"], np.float32)
    mean = cls[g].astype(np.float32)
    for parcel, v in spec["parcels"].items():
        mean[parcel_fine_ids(parcel)] = float(v)
    mean[g == 0] = 0.0
    return mean


def preset_params(name, n, seed, cfg, level):
    """Params dict of `draw_synth_params` (same keys, shapes and dtypes; ``flag`` True) for the
    evaluation preset ``name`` and a group of ``n`` images, deterministic from (name, seed).

    ``cfg``: the SynthConfig whose ranges apply (std_max, jitter, FWHM families and jitter, bias,
    noise; ``prob`` is ignored). random_*: `draw_synth_params` with prob 1. Named presets: fixed
    class means + parcel overrides (module docstring), re-mask on, gamma 1, the rest drawn as in
    training from the preset's generator. Call under tf.device('/cpu:0') for CPU synthesis."""
    P.check_level(level)
    if name not in EVAL_PRESETS:
        raise ValueError("unknown preset %r (known: %s)" % (name, ", ".join(PRESET_NAMES)))
    spec = EVAL_PRESETS[name]
    n = int(n)
    gen = tf.random.Generator.from_seed(preset_seed(name, seed))
    if spec["kind"] == "random":
        return draw_synth_params(gen, n, dataclasses.replace(cfg, prob=1.0))
    f32 = tf.float32
    g3 = int(cfg.bias_grid) ** 3
    u_mr = gen.uniform([1], dtype=f32)
    u_bias = gen.uniform([1], dtype=f32)
    cls_std = gen.uniform([TABLE_SIZE], dtype=f32)
    u_jit = gen.uniform([n, TABLE_SIZE], dtype=f32)
    u_fwhm = gen.uniform([n, 3], dtype=f32)
    u_noise = gen.uniform([n, 1], dtype=f32)
    z_bias = gen.normal([n, g3], dtype=f32)
    seeds_x = tf.transpose(gen.make_seeds(n))
    seeds_n = tf.transpose(gen.make_seeds(n))
    g = fine_to_train18_table()
    fg = tf.cast(g > 0, f32)
    std = tf.gather(float(cfg.std_max) * cls_std, g) * fg
    cls_mean = np.zeros(TABLE_SIZE, np.float32)
    cls_mean[:P.N_TRAIN_LABELS] = spec["class_means"]
    pet = spec["family"] == "pet"
    if spec["fwhm_mm"] is not None:
        base = tf.constant(float(spec["fwhm_mm"]), f32)
    else:
        lo, hi = [float(v) for v in cfg.fwhm_mr_mm]
        base = lo + u_mr[0] * (hi - lo)
    return {
        "flag": tf.constant(True),
        "mean": tf.constant(preset_mean_table_np(name)),
        "std": std,
        "cls_mean": tf.constant(cls_mean),
        "jitter": float(cfg.jitter) * (2.0 * u_jit - 1.0),
        "fwhm_mm": base * (1.0 + float(cfg.fwhm_image_jitter) * (2.0 * u_fwhm - 1.0)),
        "pet_family": tf.constant(pet),
        "remask": tf.constant(True),
        "gamma": tf.constant(1.0, f32),
        "noise_std": u_noise[:, 0] * float(cfg.noise_std_max),
        "bias": u_bias[0] * float(cfg.bias_std_max) * z_bias,
        "seeds": tf.stack([seeds_x, seeds_n], axis=1),
    }
