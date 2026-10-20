"""DALI 1.18 pipelines for the linear (rigid / affine) tasks and the deformable task (milestone 6,
`DeformPipeline`, see below). Adapted from the rigid project's
``data_loading/dali_loader.py`` (reader, random transforms, warp, intensity augmentation).

One group per batch (design 3.2): the DALI batch size is the group size N, every sample is one
image with its own independent random transform about the image center.

Matrices (DALI order and DALI pixel coordinates, 3x4, float32), per image i:
    T_i    = C+ . Tr_i . R_i . K_i . C-      rendered transform (template -> image i)
    Rt_i   = C+ . Tr_i . R_i . C-            rigid part (rigid task target)
    T_i^-1 = C+ . K_i^-1 . R_i^-1 . Tr_i^-1 . C-   pull matrix given to warp_affine
C+- translate by +-the image center (`params.dali_center`). K_i = U_i D_i is the scale/shear
(`params.scale_shear_rows`), the nuisance S_i of the rigid task and part of the target for affine.
So T_i = Rt_i S_i with S_i = C+ K_i C- about the center: the image is rendered as
I_i(y) = I(S_i^-1 Rt_i^-1 y), and the rigid target uses Rt_i only, so the network never has to
undo S_i. The emitted matrices are the FORWARD T_i (and Rt_i), not the pull matrix.

Outputs (all float32, all on GPU):
    rigid:  images [N, X, Y, Z, 1], T_dali [N, 3, 4], Rt_dali [N, 3, 4]
    affine: images [N, X, Y, Z, 1], T_dali [N, 3, 4]
Images are channel-last with the numpy array axis order (x, y, z) (DALI layout "DHWC").
`data_loading.postprocess` converts the matrices to TF centered-mm 4x4.

Control file (curriculum and global transform G, design 3.2 v3.5; only when `control_file` is
given, default off). A second numpy reader re-reads the control vector
(`params.CONTROL_FIELDS`) every batch (`read_control`). The per-image half-widths become
U(-1, 1) * control value instead of constants, and one shared G = C+ . Tg . Rg . C- (about the
image center, from the file's axis / angle / translation) is composed last:
    T_i    = G . Rt_i . S_i         rendered transform (one combined matrix, one interpolation)
    Rt_dali = G . Rt_i              rigid target part (the network undoes G and Rt_i, not S_i)
    T_i^-1 = S_i^-1 . Rt_i^-1 . G^-1   pull matrix
and two outputs are appended: G_dali [N, 3, 4] (per sample, so a batch mixing two control states
is detectable) and control_version [N] (float32). With G = I and the end ranges in the file the
distribution equals the constant-range pipeline.

No flips in DALI (they are per group, in TF). No external_source (it failed under model.fit in
the reference project; the control file is read by a numpy reader as in its curriculum.py).

Deformable pipeline (milestone 6, `DeformPipeline`). Images are already in MNI space (A = I), so
by default nothing is warped. Outputs (all on GPU):
    residual off (default): images float32 [N, X, Y, Z, 1], labels uint8 [N, X, Y, Z, 1]
    residual on:            images, labels, T_dali float32 [N, 3, 4]
With the residual affine (small, `params.DEFAULT_DEFORM_RESIDUAL_RANGES`) one pull matrix per
sample warps the image (linear) and its labels (nearest, fill 0), and the forward T_dali is
emitted as for the linear tasks. Intensity augmentation touches the images only.
Alignment of images and labels: two numpy readers (`paired_reader_kwargs`, names ImageReader /
LabelReader) get the same pre-permuted file order (`paired_permutation`), the same seed, and
identical settings (`shuffle_after_epoch`, `pad_last_batch=False`), so they produce identical
sample sequences, also across DALI's per-epoch reshuffle. 'buffer' shuffling (two independent
random_shuffle buffers) is refused because nothing guarantees their alignment. Verified on GPU
by tests/gpu/verify_dali_pipeline.py check (l).

This module imports without DALI; building a pipeline needs nvidia.dali (`require_dali`).
"""
import os

import numpy as np

from data_loading import params as P

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.math as dmath
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    DALI_AVAILABLE = True
    _DALI_IMPORT_ERROR = None
except ImportError as _e:  # local WSL envs have no DALI
    fn = dmath = types = None
    Pipeline = object
    DALI_AVAILABLE = False
    _DALI_IMPORT_ERROR = _e

READER_NAME = "ImageReader"
LABEL_READER_NAME = "LabelReader"
CONTROL_READER_NAME = "ControlReader"


def require_dali():
    if not DALI_AVAILABLE:
        raise ImportError("nvidia.dali is required for the DALI pipelines (%s)" % _DALI_IMPORT_ERROR)


def linear_output_signature(task, level, batch_size, control=False):
    """Static output shapes and dtype names of `LinearPipeline` (no DALI needed).

    ``control=True`` (pipeline with a control file) appends G_dali [N, 3, 4] and
    control_version [N]."""
    if task not in P.TASKS:
        raise ValueError("task must be one of {}, got {!r}".format(P.TASKS, task))
    n = int(batch_size)
    shape = P.LEVEL_SHAPE[P.check_level(level)]
    shapes = [(n,) + tuple(shape) + (1,), (n, 3, 4)]
    if task == "rigid":
        shapes.append((n, 3, 4))
    if control:
        shapes += [(n, 3, 4), (n,)]
    return tuple(shapes), tuple(["float32"] * len(shapes))


def base_output_count(task):
    """Number of outputs without the control extras: 3 (rigid) or 2 (affine)."""
    if task not in P.TASKS:
        raise ValueError("task must be one of {}, got {!r}".format(P.TASKS, task))
    return 3 if task == "rigid" else 2


def deform_output_signature(level, batch_size, residual_affine=False):
    """Static output shapes and dtype names of `DeformPipeline` (no DALI needed):
    images float32 [N, X, Y, Z, 1], labels uint8 [N, X, Y, Z, 1] (+ T_dali float32 [N, 3, 4]
    with the residual affine)."""
    n = int(batch_size)
    vol = (n,) + tuple(P.LEVEL_SHAPE[P.check_level(level)]) + (1,)
    shapes, dtypes = [vol, vol], ["float32", "uint8"]
    if residual_affine:
        shapes.append((n, 3, 4))
        dtypes.append("float32")
    return tuple(shapes), tuple(dtypes)


def output_count(task, residual_affine=False):
    """Number of base outputs of a task's pipeline: linear = `base_output_count` (without the
    control extras); deform = 2 (images, labels) or 3 with the residual affine (+ T_dali)."""
    if task == P.DEFORM_TASK:
        return 3 if residual_affine else 2
    if task not in P.TASKS:
        raise ValueError("task must be one of {}, got {!r}".format(P.ALL_TASKS, task))
    return base_output_count(task)


# ---------------------------------------------------------------------------- graph helpers
def _zero():
    return types.Constant(0.0, dtype=types.FLOAT)


def _const_vec(values):
    vals = [float(v) for v in values]
    return fn.constant(fdata=vals, shape=[len(vals)], dtype=types.FLOAT)


def _const_scalar(value):
    return _const_vec([value])[0]


def _uniform_scalar(half):
    return fn.random.uniform(range=(-float(half), float(half))) if half > 0 else _const_scalar(0.0)


def _uniform_vec(half, n=3):
    return (fn.random.uniform(range=(-float(half), float(half)), shape=[n]) if half > 0
            else _const_vec([0.0] * n))


def _multiplex(probability, augmented, original):
    """Per-sample: `augmented` with the given probability, else `original` (reference code)."""
    condition = fn.cast(fn.random.coin_flip(probability=probability), dtype=types.DALIDataType.BOOL)
    neg_condition = condition ^ True
    return condition * augmented + neg_condition * original


def _stack_matrix(rows):
    return fn.stack(*[fn.stack(*row) for row in rows])


def random_linear_params(ranges):
    """Per-sample random parameters (DataNodes) for `build_linear_matrices`.

    angle: degrees; axis: unnormalized standard normal (normalized later, so uniform on the
    sphere); translation: voxels, DALI axis order; log_scale and shear: TF axis order
    (x, y, z) and (k_xy, k_xz, k_yz).
    """
    return {"angle": _uniform_scalar(ranges.rotation_deg),
            "axis": fn.random.normal(shape=[3]),
            "translation": _uniform_vec(ranges.translation_vox),
            "log_scale": _uniform_vec(ranges.log_scale),
            "shear": _uniform_vec(ranges.shear)}


def _scaled_uniform(scale):
    """U(-1, 1) * scale for a scalar DataNode `scale` (per sample)."""
    return fn.random.uniform(range=(-1.0, 1.0)) * scale


def _scaled_uniform_vec(scale, n=3):
    # Independent scalar draws stacked, so only scalar x scalar arithmetic is needed.
    return fn.stack(*[_scaled_uniform(scale) for _ in range(n)])


def control_linear_params(ctl, level):
    """Per-sample random parameters (same meaning as `random_linear_params`) whose half-widths
    come from the control vector `ctl` (`params.CONTROL_FIELDS`): U(-1, 1) * value. The
    translation is converted mm -> voxels of `level` (1 / spacing, as `params.mm_to_voxels`)."""
    i = P.CONTROL_INDEX
    inv_s = 1.0 / P.LEVEL_SPACING[P.check_level(level)]
    return {"angle": _scaled_uniform(ctl[i["rot_deg"]]),
            "axis": fn.random.normal(shape=[3]),
            "translation": _scaled_uniform_vec(ctl[i["trans_mm"]] * inv_s),
            "log_scale": _scaled_uniform_vec(ctl[i["log_scale"]]),
            "shear": _scaled_uniform_vec(ctl[i["shear"]])}


def control_global_params(ctl, level):
    """G parameters from the control vector: angle (degrees), axis and translation (voxels) in
    DALI axis order (the file stores TF order x, y, z; mirrored by `params.control_global_dali`).

    The angle is multiplied by `params.G_DALI_ANGLE_SIGN` (-1): reversing the axis order is a
    reflection, so DALI's right-handed rotation about the reversed axis by +angle is
    R(axis, -angle) in TF space (derivation at `params.G_DALI_ANGLE_SIGN`). With the sign flip
    the emitted G is R(axis, +angle) in TF space, i.e. `runtime.curriculum.global_matrix_mm` of
    the file's values, as logged (control_g_angle_deg) and as used by the val_<task>_g sets."""
    i = P.CONTROL_INDEX
    inv_s = 1.0 / P.LEVEL_SPACING[P.check_level(level)]
    ax = [ctl[i["g_axis_x"]], ctl[i["g_axis_y"]], ctl[i["g_axis_z"]]]
    t = [ctl[i["g_tx_mm"]] * inv_s, ctl[i["g_ty_mm"]] * inv_s, ctl[i["g_tz_mm"]] * inv_s]
    return {"angle": ctl[i["g_angle_deg"]] * P.G_DALI_ANGLE_SIGN, "axis": fn.stack(*ax[::-1]),
            "translation": fn.stack(*t[::-1])}


def control_reader_kwargs(path, name=CONTROL_READER_NAME):
    """Keyword arguments of the control-file reader (pure Python, unit-tested without DALI).

    Mirrors the reference rigid project's curriculum scale reader (``_make_scale_reader``), which
    worked under DALI 1.18 where ``fn.external_source`` did not: one file, unsharded, unshuffled,
    ``dont_use_mmap=True`` and ``read_ahead=False`` so the file is opened and read again for every
    batch (with mmap or read-ahead the reader could keep the first contents), and
    ``pad_last_batch=True`` so the single record fills the batch (every sample of a batch sees
    the same values). The path must be absolute. The trainer's atomic ``os.replace`` makes every
    read see either the old or the new file in full. Because of the pipeline prefetch, a rewrite
    shows up after about 1-2 batches; the emitted ``control_version`` says which state a batch
    used (measured by tests/gpu/verify_dali_pipeline.py check (h)).
    """
    return {"device": "cpu", "files": [os.path.abspath(str(path))], "num_shards": 1,
            "shard_id": 0, "random_shuffle": False, "shuffle_after_epoch": False,
            "read_ahead": False, "dont_use_mmap": True, "pad_last_batch": True, "name": name}


def read_control(path, name=CONTROL_READER_NAME):
    """The control vector [CONTROL_SIZE] float32 per sample (CPU DataNode), re-read every batch."""
    return fn.readers.numpy(**control_reader_kwargs(path, name))


def constant_linear_params(angle_deg, axis, translation_vox, log_scale=(0.0, 0.0, 0.0),
                           shear=(0.0, 0.0, 0.0)):
    """Fixed parameters (same meaning as `random_linear_params`), e.g. for convention tests."""
    return {"angle": _const_scalar(angle_deg), "axis": _const_vec(axis),
            "translation": _const_vec(translation_vox), "log_scale": _const_vec(log_scale),
            "shear": _const_vec(shear)}


def build_linear_matrices(prm, shape_xyz, pixel_center=P.DALI_PIXEL_CENTER):
    """(T, T_inv, Rt) as 3x4 DALI-order matrices (CPU DataNodes). See the module docstring.

    Mirrored in numpy by `params.linear_matrices_np` (unit-tested locally).
    """
    c = P.dali_center(shape_xyz, pixel_center)
    c_minus = fn.transforms.translation(offset=[-v for v in c])
    c_plus = fn.transforms.translation(offset=c)
    ax = prm["axis"]
    norm = dmath.max(dmath.sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]), 1e-6)
    axis = fn.stack(ax[0] / norm, ax[1] / norm, ax[2] / norm)
    angle = prm["angle"]
    rot = fn.transforms.rotation(angle=angle, axis=axis)
    rot_inv = fn.transforms.rotation(angle=-angle, axis=axis)
    t = prm["translation"]
    tr = fn.transforms.translation(offset=t)
    tr_inv = fn.transforms.translation(offset=-t)
    ls, sh = prm["log_scale"], prm["shear"]
    fwd, inv = P.scale_shear_rows(dmath.exp(ls[0]), dmath.exp(ls[1]), dmath.exp(ls[2]),
                                  sh[0], sh[1], sh[2], zero=_zero())
    k, k_inv = _stack_matrix(fwd), _stack_matrix(inv)
    # fn.transforms.combine applies its inputs in the listed order (first = rightmost factor).
    t_fwd = fn.transforms.combine(c_minus, k, rot, tr, c_plus)
    t_pull = fn.transforms.combine(c_minus, tr_inv, rot_inv, k_inv, c_plus)
    rt = fn.transforms.combine(c_minus, rot, tr, c_plus)
    return t_fwd, t_pull, rt


def _rotation_pair(prm):
    ax = prm["axis"]
    norm = dmath.max(dmath.sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]), 1e-6)
    axis = fn.stack(ax[0] / norm, ax[1] / norm, ax[2] / norm)
    angle = prm["angle"]
    return (fn.transforms.rotation(angle=angle, axis=axis),
            fn.transforms.rotation(angle=-angle, axis=axis))


def build_global_linear_matrices(prm, gprm, shape_xyz, pixel_center=P.DALI_PIXEL_CENTER):
    """(T, T_inv, Rt, G) with the shared global transform G composed last (module docstring).

    ``prm``: per-image parameters (`control_linear_params` / `random_linear_params`); ``gprm``:
    G parameters (`control_global_params`). G = C+ . Tg . Rg . C- about the same center as the
    per-image transforms. Mirrored in numpy by `params.linear_matrices_np` (g_* arguments).
    """
    c = P.dali_center(shape_xyz, pixel_center)
    c_minus = fn.transforms.translation(offset=[-v for v in c])
    c_plus = fn.transforms.translation(offset=c)
    rot, rot_inv = _rotation_pair(prm)
    t = prm["translation"]
    tr = fn.transforms.translation(offset=t)
    tr_inv = fn.transforms.translation(offset=-t)
    ls, sh = prm["log_scale"], prm["shear"]
    fwd, inv = P.scale_shear_rows(dmath.exp(ls[0]), dmath.exp(ls[1]), dmath.exp(ls[2]),
                                  sh[0], sh[1], sh[2], zero=_zero())
    k, k_inv = _stack_matrix(fwd), _stack_matrix(inv)
    g_rot, g_rot_inv = _rotation_pair(gprm)
    gt = gprm["translation"]
    g_tr = fn.transforms.translation(offset=gt)
    g_tr_inv = fn.transforms.translation(offset=-gt)
    # fn.transforms.combine applies its inputs in the listed order (first = rightmost factor).
    t_fwd = fn.transforms.combine(c_minus, k, rot, tr, g_rot, g_tr, c_plus)
    t_pull = fn.transforms.combine(c_minus, g_tr_inv, g_rot_inv, tr_inv, rot_inv, k_inv, c_plus)
    rt = fn.transforms.combine(c_minus, rot, tr, g_rot, g_tr, c_plus)
    g = fn.transforms.combine(c_minus, g_rot, g_tr, c_plus)
    return t_fwd, t_pull, rt, g


SHUFFLE_MODES = ("epoch", "buffer")


def numpy_reader_kwargs(files, file_root=None, shuffle=True, shuffle_mode="epoch",
                        initial_fill=1024, seed=0, name=READER_NAME):
    """Keyword arguments of `fn.readers.numpy` (pure Python, unit-tested without DALI).

    shuffle_mode (only used when shuffle=True):
      'epoch'  (default): shuffle_after_epoch=True, i.e. the whole file list is reshuffled at
               every epoch, with no host shuffle buffer. In DALI 1.18 it excludes random_shuffle.
      'buffer': random_shuffle=True with an `initial_fill` sample buffer in host memory. At
               levels 2 and 1 the affordable buffer is small (about 312 / 39 samples), so groups
               come from a sliding window over the file list.
    pad_last_batch=False: at an epoch boundary the reader continues into the next epoch, so a
    group never contains a padded duplicate. Batches always have exactly batch_size samples.
    """
    if shuffle_mode not in SHUFFLE_MODES:
        raise ValueError("shuffle_mode must be one of {}, got {!r}".format(SHUFFLE_MODES, shuffle_mode))
    kwargs = {"device": "cpu", "files": list(files), "random_shuffle": False,
              "pad_last_batch": False, "seed": int(seed), "dont_use_mmap": True, "name": name}
    if shuffle and shuffle_mode == "epoch":
        kwargs["shuffle_after_epoch"] = True
    elif shuffle:
        kwargs["random_shuffle"] = True
        kwargs["initial_fill"] = int(initial_fill)
    if file_root:
        kwargs["file_root"] = file_root
    return kwargs


def read_images(files, shape_xyz, file_root=None, shuffle=True, initial_fill=1024, seed=0,
                name=READER_NAME, shuffle_mode="epoch"):
    """numpy reader -> GPU images [X, Y, Z, 1] per sample (layout DHWC).

    Reader settings: `numpy_reader_kwargs`.
    """
    imgs = fn.readers.numpy(**numpy_reader_kwargs(files, file_root, shuffle, shuffle_mode,
                                                  initial_fill, seed, name))
    imgs = fn.reshape(imgs, shape=[int(v) for v in shape_xyz] + [1], layout="DHWC")
    return imgs.gpu()


def paired_permutation(n, seed):
    """The one index permutation applied to BOTH the image and the label list (the same
    RandomState permutation `LinearPipeline` applies to its file list)."""
    return np.random.RandomState(int(seed)).permutation(int(n))


def paired_reader_kwargs(image_files, label_files, file_root=None, shuffle=True, seed=0):
    """(image kwargs, label kwargs) of the two aligned numpy readers of `DeformPipeline`.

    Both are `numpy_reader_kwargs` with shuffle_mode 'epoch' and identical settings; they differ
    only in ``files`` and ``name`` (`READER_NAME`, `LABEL_READER_NAME`). The lists must already
    be in the same (pre-permuted) order and have equal length.
    """
    image_files, label_files = [str(f) for f in image_files], [str(f) for f in label_files]
    if len(image_files) != len(label_files):
        raise ValueError("image and label lists differ in length (%d vs %d)"
                         % (len(image_files), len(label_files)))
    img = numpy_reader_kwargs(image_files, file_root, shuffle, "epoch", seed=seed, name=READER_NAME)
    lbl = numpy_reader_kwargs(label_files, file_root, shuffle, "epoch", seed=seed,
                              name=LABEL_READER_NAME)
    return img, lbl


def _reader_to_gpu(kwargs, shape_xyz, dtype=None):
    out = fn.readers.numpy(**kwargs)
    if dtype is not None:
        out = fn.cast(out, dtype=dtype)
    out = fn.reshape(out, shape=[int(v) for v in shape_xyz] + [1], layout="DHWC")
    return out.gpu()


def read_labels(files, shape_xyz, file_root=None, shuffle=True, seed=0, name=LABEL_READER_NAME):
    """numpy reader -> GPU uint8 label maps [X, Y, Z, 1] per sample (layout DHWC), 'epoch' mode."""
    kwargs = numpy_reader_kwargs(files, file_root, shuffle, "epoch", seed=seed, name=name)
    return _reader_to_gpu(kwargs, shape_xyz, types.UINT8)


def warp_linear(images, pull_matrix):
    """Pull warp on GPU: out(u) = in(pull_matrix u), linear interpolation, fill 0, same size."""
    return fn.warp_affine(images, matrix=pull_matrix, fill_value=0.0, inverse_map=True,
                          interp_type=types.DALIInterpType.INTERP_LINEAR)


def _intensity_factor(prob, low, high):
    factor = _multiplex(0.5, fn.random.uniform(range=(low, 1.0)), fn.random.uniform(range=(1.0, high)))
    return _multiplex(prob, factor, 1.0)


def _gamma(img, prob, low, high):
    orig_mean = fn.reductions.mean(img)
    orig_std = fn.reductions.std_dev(img, orig_mean)
    orig_min = fn.reductions.min(img)
    orig_range = fn.reductions.max(img) - orig_min
    out = dmath.pow((img - orig_min) / dmath.max(orig_range, 1e-7),
                    _intensity_factor(prob, low, high)) * orig_range + orig_min
    cur_mean = fn.reductions.mean(out)
    cur_std = fn.reductions.std_dev(out, cur_mean)
    return ((out - cur_mean) * (orig_std / dmath.max(cur_std, 1e-7))) + orig_mean


def intensity_augment(img, cfg, shape_xyz):
    """Rigid-project intensity augmentation on GPU images (see `params.IntensityAugConfig`)."""
    if cfg is None or not cfg.enabled:
        return img
    if cfg.noise:
        img = fn.noise.gaussian(img, stddev=_multiplex(
            cfg.noise_prob, fn.random.uniform(range=(0.0, cfg.noise_std_max)), 0.0))
    if cfg.brightness:
        img = img * _intensity_factor(cfg.brightness_prob, *cfg.brightness_range)
    if cfg.contrast:
        mean = fn.reductions.mean(img)
        img = dmath.clamp((img - mean) * _intensity_factor(cfg.contrast_prob, *cfg.contrast_range) + mean,
                          fn.reductions.min(img), fn.reductions.max(img))
    if cfg.low_res:
        full = types.Constant(np.asarray(shape_xyz, np.float32), dtype=types.FLOAT)
        size = _multiplex(cfg.low_res_prob, fn.random.uniform(range=(cfg.low_res_min_factor, 1.0)), 1.0) * full
        img = fn.resize(img, size=size, interp_type=types.DALIInterpType.INTERP_NN)
        # Upsampling back to the full grid: antialiasing only matters when shrinking, so turn it
        # off explicitly (DALI >= 1.18 warns that LINEAR now antialiases by default).
        img = fn.resize(img, size=full, interp_type=types.DALIInterpType.INTERP_LINEAR, antialias=False)
    if cfg.gamma:
        img = _gamma(img, cfg.gamma_prob, *cfg.gamma_range)
    return dmath.clamp(img, 0.0, 1.0)


# ---------------------------------------------------------------------------- pipeline
class LinearPipeline(Pipeline):
    """Rigid or affine augmentation pipeline: one group of N images per batch.

    Args:
      task: 'rigid' or 'affine'.
      level: 4, 2 or 1 (downsampling factor; sets shape and spacing).
      files: image .npy paths (relative to `file_root` if given), e.g. from
        `data_module.manifest_files`.
      batch_size: group size N.
      seed: pipeline seed (random ops, reader). With shuffle=True the file list is also
        permuted once with this seed before DALI's shuffle, so the first epoch's groups are
        random and depend on the seed (DALI's per-epoch reshuffle uses its own fixed seed).
      shuffle: shuffle the reader. False reads `files` in order (used by the GPU check).
      shuffle_mode: 'epoch' (default; full reshuffle every epoch, no host buffer) or 'buffer'
        (random_shuffle with `initial_fill`). See `numpy_reader_kwargs`.
      rotation_deg, translation_mm, log_scale, shear: half-widths of the uniform ranges
        (None -> `params.DEFAULT_LINEAR_RANGES[task]`). Translations are in mm and converted
        to voxels of `level`.
      intensity_aug: `params.IntensityAugConfig` (None -> defaults, i.e. ON). Use
        `IntensityAugConfig.off()` to disable.
      initial_fill: shuffle buffer size for shuffle_mode='buffer' (None ->
        `params.auto_initial_fill`, ~1 GiB host).
      prefetch_queue_depth: DALI prefetch depth (default 1, design 3.2).
      control_file: path of the task's control .npy (`params.CONTROL_FIELDS`), or None (default:
        constant ranges, no extra outputs). When given, the per-image half-widths and G come from
        the file, which must exist before the pipeline runs, and the outputs gain G_dali and
        control_version; the rotation_deg ... shear arguments are then unused.
    """

    def __init__(self, task, level, files, batch_size, seed=0, file_root=None, shuffle=True,
                 rotation_deg=None, translation_mm=None, log_scale=None, shear=None,
                 intensity_aug=None, num_threads=4, device_id=0, prefetch_queue_depth=1,
                 initial_fill=None, pixel_center=P.DALI_PIXEL_CENTER, exec_async=True,
                 exec_pipelined=True, shuffle_mode="epoch", control_file=None):
        require_dali()
        if shuffle_mode not in SHUFFLE_MODES:
            raise ValueError("shuffle_mode must be one of {}, got {!r}".format(SHUFFLE_MODES, shuffle_mode))
        super().__init__(batch_size=int(batch_size), num_threads=int(num_threads),
                         device_id=int(device_id), seed=int(seed),
                         prefetch_queue_depth=int(prefetch_queue_depth),
                         exec_async=exec_async, exec_pipelined=exec_pipelined)
        self.task = task
        self.level = P.check_level(level)
        self.ranges = P.linear_ranges(task, self.level, rotation_deg, translation_mm, log_scale, shear)
        self.volume_shape = P.LEVEL_SHAPE[self.level]
        self.group_size = int(batch_size)
        files = [str(f) for f in files]
        if not files:
            raise ValueError("LinearPipeline needs at least one file")
        self.shuffle = bool(shuffle)
        self.shuffle_mode = shuffle_mode
        if self.shuffle:
            files = [files[i] for i in np.random.RandomState(int(seed)).permutation(len(files))]
        self.files = files
        self.file_root = file_root
        self.reader_seed = int(seed)
        self.initial_fill = (P.auto_initial_fill(len(files), batch_size, self.volume_shape)
                             if initial_fill is None else int(initial_fill))
        self.intensity_aug = P.IntensityAugConfig() if intensity_aug is None else intensity_aug
        self.pixel_center = float(pixel_center)
        self.control_file = None if control_file is None else os.path.abspath(str(control_file))

    def output_signature(self):
        return linear_output_signature(self.task, self.level, self.group_size,
                                       control=self.control_file is not None)

    def define_graph(self):
        imgs = read_images(self.files, self.volume_shape, self.file_root, self.shuffle, self.initial_fill,
                           self.reader_seed, shuffle_mode=self.shuffle_mode)
        extra = ()
        if self.control_file is None:
            t_fwd, t_pull, rt = build_linear_matrices(random_linear_params(self.ranges), self.volume_shape,
                                                      self.pixel_center)
        else:
            ctl = read_control(self.control_file)
            t_fwd, t_pull, rt, g = build_global_linear_matrices(
                control_linear_params(ctl, self.level), control_global_params(ctl, self.level),
                self.volume_shape, self.pixel_center)
            extra = (g.gpu(), ctl[P.CONTROL_INDEX["version"]].gpu())
        imgs = warp_linear(imgs, t_pull)
        imgs = intensity_augment(imgs, self.intensity_aug, self.volume_shape)
        if self.task == "rigid":
            return (imgs, t_fwd.gpu(), rt.gpu()) + extra
        return (imgs, t_fwd.gpu()) + extra


def check_deform_inputs(image_files, label_files, shuffle_mode="epoch"):
    """Pure-Python validation of `DeformPipeline` inputs (runs before `require_dali`)."""
    if shuffle_mode != "epoch":
        raise ValueError("DeformPipeline supports shuffle_mode 'epoch' only (two random_shuffle "
                         "buffers are not guaranteed to stay aligned), got {!r}".format(shuffle_mode))
    if len(image_files) != len(label_files):
        raise ValueError("image and label lists differ in length (%d vs %d)"
                         % (len(image_files), len(label_files)))
    if not image_files:
        raise ValueError("DeformPipeline needs at least one file")


class DeformPipeline(Pipeline):
    """Deformable-task pipeline: one group of N (image, train18 labels) pairs per batch.

    Args:
      level: 4, 2 or 1.
      image_files, label_files: aligned .npy lists (same cases, same order; relative to
        `file_root` if given), e.g. `LinearDataModule.files` / `.label_files`.
      batch_size: group size N.
      seed: pipeline and reader seed. With shuffle=True both lists are permuted once with
        `paired_permutation(len, seed)` before DALI's per-epoch reshuffle.
      shuffle: shuffle the readers ('epoch' mode). False reads the lists in order.
      residual_affine: apply a small random residual affine (default False: no warp at all).
      residual_ranges: dict of `params.deform_residual_ranges` overrides (rotation_deg,
        translation_mm, log_scale, shear); None -> design defaults.
      intensity_aug: `params.IntensityAugConfig` (None -> defaults ON), images only.
      shuffle_mode: 'epoch' only ('buffer' raises ValueError, module docstring).
      Other arguments as `LinearPipeline`.
    Outputs: (images, labels[, T_dali]) (module docstring, `deform_output_signature`).
    """

    def __init__(self, level, image_files, label_files, batch_size, seed=0, file_root=None,
                 shuffle=True, residual_affine=False, residual_ranges=None, intensity_aug=None,
                 num_threads=4, device_id=0, prefetch_queue_depth=1,
                 pixel_center=P.DALI_PIXEL_CENTER, shuffle_mode="epoch", exec_async=True,
                 exec_pipelined=True):
        image_files, label_files = [str(f) for f in image_files], [str(f) for f in label_files]
        check_deform_inputs(image_files, label_files, shuffle_mode)
        level = P.check_level(level)
        ranges = P.deform_residual_ranges(level, **dict(residual_ranges or {}))
        require_dali()
        super().__init__(batch_size=int(batch_size), num_threads=int(num_threads),
                         device_id=int(device_id), seed=int(seed),
                         prefetch_queue_depth=int(prefetch_queue_depth),
                         exec_async=exec_async, exec_pipelined=exec_pipelined)
        self.task = P.DEFORM_TASK
        self.level = level
        self.volume_shape = P.LEVEL_SHAPE[level]
        self.group_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.shuffle_mode = shuffle_mode
        if self.shuffle:
            perm = paired_permutation(len(image_files), seed)
            image_files = [image_files[i] for i in perm]
            label_files = [label_files[i] for i in perm]
        self.files = image_files
        self.label_files = label_files
        self.file_root = file_root
        self.reader_seed = int(seed)
        self.residual_affine = bool(residual_affine)
        self.ranges = ranges
        self.intensity_aug = P.IntensityAugConfig() if intensity_aug is None else intensity_aug
        self.pixel_center = float(pixel_center)

    def output_signature(self):
        return deform_output_signature(self.level, self.group_size, self.residual_affine)

    def define_graph(self):
        img_kw, lbl_kw = paired_reader_kwargs(self.files, self.label_files, self.file_root,
                                              self.shuffle, self.reader_seed)
        imgs = _reader_to_gpu(img_kw, self.volume_shape)
        labels = _reader_to_gpu(lbl_kw, self.volume_shape, types.UINT8)
        extra = ()
        if self.residual_affine:
            t_fwd, t_pull, _ = build_linear_matrices(random_linear_params(self.ranges),
                                                     self.volume_shape, self.pixel_center)
            imgs = warp_linear(imgs, t_pull)
            # Same pull matrix for the labels (nearest, background fill): image and labels stay
            # aligned voxel by voxel.
            labels = fn.warp_affine(labels, matrix=t_pull, fill_value=0, inverse_map=True,
                                    interp_type=types.DALIInterpType.INTERP_NN)
            extra = (t_fwd.gpu(),)
        imgs = intensity_augment(imgs, self.intensity_aug, self.volume_shape)
        return (imgs, labels) + extra
