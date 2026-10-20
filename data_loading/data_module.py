"""Per-level file lists from export_manifest.csv and the per-task DALIDatasets.

Data layout (preprocessing `export-npy`, design 3.1): ``<export_root>/export_manifest.csv`` with
one row per case; files live in ``<export_root>/<dataset>/<subject_dir>/`` and are named by the
columns ``image_l<k>`` and ``labels_l<k>`` (``;``-separated label files). The template is in
``<export_root>/template``. On Martinos the export root is
``/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/data/npy_v2`` (pass it in;
nothing here hardcodes it).

Usage::

    from data_loading import data_module as dm
    dm.enable_memory_growth()                 # before any TF GPU work
    data = dm.LinearDataModule(export_root, level=4, batch_size=48, seed=0)
    it = data.iterator("rigid")               # yields (images, T_dali, Rt_dali)
    it_aff = data.iterator("affine")          # yields (images, T_dali)

    it_def = data.iterator("deform")          # yields (images, labels[, T_dali]) (milestone 6)

With ``control_dir`` (curriculum / global transform) every linear tuple also ends with G_dali
[N, 3, 4] and control_version [N] (`dali_loader.linear_output_signature(..., control=True)`).
The deformable dataset (`dali_loader.DeformPipeline`) reads the image and the train18 label file
of each case (`label_files`, resolved lazily so linear-only runs never need label columns); its
group size is ``deform_batch_size`` (default: ``batch_size``). Control files are linear-only.

Device placement: the DALIDatasets are created and iterated under ``tf.device('/gpu:0')`` with
DALI ``device_id=0``, i.e. the single GPU the process sees (set CUDA_VISIBLE_DEVICES outside).
"""
import csv
import os

import tensorflow as tf

from data_loading import params as P
from data_loading import dali_loader

MANIFEST_FILE = "export_manifest.csv"
TEMPLATE_DIR = "template"
DEFAULT_LABEL_SET = "train18"
TASK_SEED_OFFSET = {"rigid": 0, "affine": 1, "deform": 2}


def enable_memory_growth():
    """Turn on TF memory growth for every visible GPU and turn TF32 off (see
    models/precision.py: TF32 on Ampere GPUs corrupts the float32 transform/grid math).
    Call before building datasets or models."""
    from models.precision import configure_tf_precision
    configure_tf_precision(tf32=False, memory_growth=True)
    return tf.config.list_physical_devices("GPU")


def read_manifest(path):
    """export_manifest.csv -> list of row dicts (csv module; no pandas)."""
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def manifest_files(rows, level, split="train", kind="image", label_set=DEFAULT_LABEL_SET,
                   contrast=None):
    """Relative file paths (``<dataset>/<subject_dir>/<file>``) for one level and split.

    Args:
      rows: from `read_manifest`.
      level: 4, 2 or 1 (columns ``image_l<k>`` / ``labels_l<k>``).
      split: manifest ``split`` value ('train', 'val'); None keeps every row.
      kind: 'image' or 'labels' (the file of `label_set`, e.g. labels_train18_4.npy).
      contrast: optional contrast filter (e.g. 'T1w'); None keeps every row.
    Rows are returned in manifest order. Paths are relative to the export root, so the reader
    uses ``file_root=export_root`` and the export can move without editing the manifest.
    """
    level = P.check_level(level)
    if kind not in ("image", "labels"):
        raise ValueError("kind must be 'image' or 'labels', got {!r}".format(kind))
    col = "%s_l%d" % (kind, level)
    out = []
    for r in rows:
        if split is not None and r.get("split") != split:
            continue
        if contrast is not None and r.get("contrast") != contrast:
            continue
        if col not in r:
            raise KeyError("manifest has no column %s" % col)
        value = (r[col] or "").strip()
        if kind == "labels":
            want = "labels_%s%s.npy" % (label_set, P.level_suffix(level))
            names = [v for v in value.split(";") if v]
            if want not in names:
                raise ValueError("case %s/%s has no %s (labels_l%d = %r)"
                                 % (r.get("dataset"), r.get("subject_dir"), want, level, value))
            value = want
        if not value:
            raise ValueError("case %s/%s has no %s" % (r.get("dataset"), r.get("subject_dir"), col))
        out.append("/".join([r["dataset"], r["subject_dir"], value]))
    return out


def check_level_shape(rows, level):
    """Raise if any row's ``shape_l<k>`` differs from `params.LEVEL_SHAPE[level]`."""
    level = P.check_level(level)
    want = "x".join(str(v) for v in P.LEVEL_SHAPE[level])
    bad = [r.get("subject_dir") for r in rows if r.get("shape_l%d" % level) != want]
    if bad:
        raise ValueError("%d case(s) have shape_l%d != %s, e.g. %s" % (len(bad), level, want, bad[:3]))


def make_dali_dataset(pipeline, device_id=0):
    """Wrap a `LinearPipeline` in a DALIDataset with fixed output shapes/dtypes, on the GPU."""
    import nvidia.dali.plugin.tf as dali_tf
    shapes, dtypes = pipeline.output_signature()
    with tf.device("/gpu:%d" % int(device_id)):
        return dali_tf.DALIDataset(pipeline=pipeline, batch_size=pipeline.group_size,
                                   output_shapes=shapes,
                                   output_dtypes=tuple(tf.as_dtype(d) for d in dtypes),
                                   device_id=int(device_id))


class LinearDataModule:
    """Rigid, affine and deformable DALIDatasets for one level (alias `DataModule`).

    Args:
      export_root: npy export folder (holds export_manifest.csv and template/).
      level: 4, 2 or 1; batch_size: group size N for this level.
      seed: base seed; rigid uses seed + 0, affine seed + 1, deform seed + 2.
      split: manifest split for the file list ('train' or 'val').
      shuffle: reader shuffle (training: True).
      shuffle_mode: 'epoch' (default, full reshuffle every epoch) or 'buffer'
        (see `dali_loader.numpy_reader_kwargs`).
      rigid_ranges / affine_ranges: dicts of `LinearPipeline` range overrides
        (rotation_deg, translation_mm, log_scale, shear); empty -> design defaults.
      intensity_aug: `params.IntensityAugConfig` (None -> defaults ON).
      deform_residual_affine: residual affine in the deformable pipeline (default OFF).
      deform_batch_size: group size of the deformable pipeline (None -> batch_size).
      deform_ranges: dict of residual-affine range overrides (`params.deform_residual_ranges`);
        empty -> design defaults. Only used with deform_residual_affine.
      num_threads, prefetch_queue_depth (default 1), device_id (default 0): DALI settings.
      contrast: optional manifest contrast filter.
      control_dir: folder of the per-task control files (curriculum / global transform, design
        3.2 v3.5), e.g. ``<run_dir>/control``; None (default) = constant ranges, no control file.
        When given, ``<control_dir>/<task>.npy`` is written here (version 1) for every task of
        `control_init` (default: both linear tasks), so the pipelines can start before the
        trainer's first update; `control_writers[task]` is the `runtime.curriculum.ControlWriter`
        the trainer keeps writing through.
      control_init: {task: `params.control_vector` keyword dict} initial values (None -> the
        task's ranges with G = identity).
    """

    def __init__(self, export_root, level, batch_size, seed=0, split="train", shuffle=True,
                 rigid_ranges=None, affine_ranges=None, intensity_aug=None,
                 deform_residual_affine=P.DEFAULT_DEFORM_RESIDUAL_AFFINE, num_threads=4,
                 prefetch_queue_depth=1, device_id=0, contrast=None, manifest_path=None,
                 shuffle_mode="epoch", control_dir=None, control_init=None, deform_batch_size=None,
                 deform_ranges=None):
        self.export_root = export_root
        self.level = P.check_level(level)
        self.batch_size = int(batch_size)
        self.deform_batch_size = self.batch_size if deform_batch_size is None else int(deform_batch_size)
        self.deform_ranges = dict(deform_ranges or {})
        self.seed = int(seed)
        self.split = split
        self.shuffle = bool(shuffle)
        self.shuffle_mode = shuffle_mode
        self.ranges = {"rigid": dict(rigid_ranges or {}), "affine": dict(affine_ranges or {})}
        self.intensity_aug = intensity_aug
        self.deform_residual_affine = bool(deform_residual_affine)
        self.num_threads = int(num_threads)
        self.prefetch_queue_depth = int(prefetch_queue_depth)
        self.device_id = int(device_id)
        self.manifest_path = manifest_path or os.path.join(export_root, MANIFEST_FILE)
        rows = read_manifest(self.manifest_path)
        self.rows = [r for r in rows if split is None or r.get("split") == split]
        check_level_shape(self.rows, self.level)
        self.contrast = contrast
        self.files = manifest_files(self.rows, self.level, split=None, contrast=contrast)
        self._label_files = None
        if len(self.files) < self.batch_size:
            raise ValueError("split %r has %d images at level %d, fewer than the group size %d"
                             % (split, len(self.files), self.level, self.batch_size))
        self._datasets = {}
        self.control_dir = control_dir
        self.control_writers = {}
        if control_dir:
            from runtime.curriculum import ControlWriter, identity_global
            init = dict(control_init) if control_init is not None else {t: None for t in P.TASKS}
            for task, vals in init.items():
                if task not in P.TASKS:
                    raise ValueError("control_init task must be one of {}, got {!r}".format(P.TASKS, task))
                if vals is None:
                    r = P.linear_ranges(task, self.level, **self.ranges[task])
                    vals = dict({k: getattr(r, k) for k in P.CONTROL_RANGE_KEYS}, **identity_global())
                w = ControlWriter(os.path.join(control_dir, "%s.npy" % task))
                w.write(vals)
                self.control_writers[task] = w

    @property
    def template_dir(self):
        return os.path.join(self.export_root, TEMPLATE_DIR)

    @property
    def label_files(self):
        """train18 label files of the same rows, in the same order as `files` (resolved on first
        use; raises KeyError / ValueError if the manifest lacks them)."""
        if self._label_files is None:
            self._label_files = manifest_files(self.rows, self.level, split=None, kind="labels",
                                               label_set=DEFAULT_LABEL_SET, contrast=self.contrast)
        return self._label_files

    def pipeline(self, task):
        if task == P.DEFORM_TASK:
            if len(self.files) < self.deform_batch_size:
                raise ValueError("split %r has %d images at level %d, fewer than the deform group "
                                 "size %d" % (self.split, len(self.files), self.level,
                                              self.deform_batch_size))
            return dali_loader.DeformPipeline(
                self.level, self.files, self.label_files, self.deform_batch_size,
                seed=self.seed + TASK_SEED_OFFSET[task], file_root=self.export_root,
                shuffle=self.shuffle, residual_affine=self.deform_residual_affine,
                residual_ranges=self.deform_ranges, intensity_aug=self.intensity_aug,
                num_threads=self.num_threads, device_id=self.device_id,
                prefetch_queue_depth=self.prefetch_queue_depth, shuffle_mode=self.shuffle_mode)
        return dali_loader.LinearPipeline(
            task, self.level, self.files, self.batch_size, seed=self.seed + TASK_SEED_OFFSET[task],
            file_root=self.export_root, shuffle=self.shuffle, intensity_aug=self.intensity_aug,
            num_threads=self.num_threads, device_id=self.device_id,
            prefetch_queue_depth=self.prefetch_queue_depth, shuffle_mode=self.shuffle_mode,
            control_file=self.control_file(task), **self.ranges[task])

    def control_file(self, task):
        """Path of the task's control file, or None without one."""
        w = self.control_writers.get(task)
        return None if w is None else w.path

    def dataset(self, task):
        """The task's DALIDataset (built once, cached)."""
        if task not in P.ALL_TASKS:
            raise ValueError("task must be one of {}, got {!r}".format(P.ALL_TASKS, task))
        if task not in self._datasets:
            self._datasets[task] = make_dali_dataset(self.pipeline(task), self.device_id)
        return self._datasets[task]

    def iterator(self, task):
        """A Python iterator over the task's dataset (created on the DALI GPU device)."""
        ds = self.dataset(task)
        with tf.device("/gpu:%d" % self.device_id):
            return iter(ds)


# The module now serves all three tasks; the old name stays for existing callers.
DataModule = LinearDataModule
