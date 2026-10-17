"""DALI 1.18 pipelines for the linear (rigid / affine) tasks. Adapted from the rigid project's
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

No flips in DALI (they are per group, in TF). No external_source (the curriculum hook, if added
later, is a numpy reader over a scale file as in the reference curriculum.py).

This module imports without DALI; building a pipeline needs nvidia.dali (`require_dali`).
"""
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


def require_dali():
    if not DALI_AVAILABLE:
        raise ImportError("nvidia.dali is required for the DALI pipelines (%s)" % _DALI_IMPORT_ERROR)


def linear_output_signature(task, level, batch_size):
    """Static output shapes and dtype names of `LinearPipeline` (no DALI needed)."""
    if task not in P.TASKS:
        raise ValueError("task must be one of {}, got {!r}".format(P.TASKS, task))
    n = int(batch_size)
    shape = P.LEVEL_SHAPE[P.check_level(level)]
    shapes = [(n,) + tuple(shape) + (1,), (n, 3, 4)]
    if task == "rigid":
        shapes.append((n, 3, 4))
    return tuple(shapes), tuple(["float32"] * len(shapes))


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
    """

    def __init__(self, task, level, files, batch_size, seed=0, file_root=None, shuffle=True,
                 rotation_deg=None, translation_mm=None, log_scale=None, shear=None,
                 intensity_aug=None, num_threads=4, device_id=0, prefetch_queue_depth=1,
                 initial_fill=None, pixel_center=P.DALI_PIXEL_CENTER, exec_async=True,
                 exec_pipelined=True, shuffle_mode="epoch"):
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

    def output_signature(self):
        return linear_output_signature(self.task, self.level, self.group_size)

    def define_graph(self):
        imgs = read_images(self.files, self.volume_shape, self.file_root, self.shuffle, self.initial_fill,
                           self.reader_seed, shuffle_mode=self.shuffle_mode)
        t_fwd, t_pull, rt = build_linear_matrices(random_linear_params(self.ranges), self.volume_shape,
                                                  self.pixel_center)
        imgs = warp_linear(imgs, t_pull)
        imgs = intensity_augment(imgs, self.intensity_aug, self.volume_shape)
        if self.task == "rigid":
            return imgs, t_fwd.gpu(), rt.gpu()
        return imgs, t_fwd.gpu()
