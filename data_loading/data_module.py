"""Per-level file lists from export_manifest.csv and the linear-task DALIDatasets.

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
    """Turn on TF memory growth for every visible GPU. Call before building datasets."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # GPU already initialized; growth must be set before first use
    return gpus


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
    """Rigid and affine DALIDatasets for one level.

    Args:
      export_root: npy export folder (holds export_manifest.csv and template/).
      level: 4, 2 or 1; batch_size: group size N for this level.
      seed: base seed; rigid uses seed + 0, affine seed + 1 (deformable will use seed + 2).
      split: manifest split for the file list ('train' or 'val').
      shuffle: reader shuffle (training: True).
      shuffle_mode: 'epoch' (default, full reshuffle every epoch) or 'buffer'
        (see `dali_loader.numpy_reader_kwargs`).
      rigid_ranges / affine_ranges: dicts of `LinearPipeline` range overrides
        (rotation_deg, translation_mm, log_scale, shear); empty -> design defaults.
      intensity_aug: `params.IntensityAugConfig` (None -> defaults ON).
      deform_residual_affine: residual affine in the deformable pipeline (default OFF). Stored
        for milestone 6 (the deformable pipeline is not implemented yet).
      num_threads, prefetch_queue_depth (default 1), device_id (default 0): DALI settings.
      contrast: optional manifest contrast filter.
    """

    def __init__(self, export_root, level, batch_size, seed=0, split="train", shuffle=True,
                 rigid_ranges=None, affine_ranges=None, intensity_aug=None,
                 deform_residual_affine=P.DEFAULT_DEFORM_RESIDUAL_AFFINE, num_threads=4,
                 prefetch_queue_depth=1, device_id=0, contrast=None, manifest_path=None,
                 shuffle_mode="epoch"):
        self.export_root = export_root
        self.level = P.check_level(level)
        self.batch_size = int(batch_size)
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
        self.files = manifest_files(self.rows, self.level, split=None, contrast=contrast)
        if len(self.files) < self.batch_size:
            raise ValueError("split %r has %d images at level %d, fewer than the group size %d"
                             % (split, len(self.files), self.level, self.batch_size))
        self._datasets = {}

    @property
    def template_dir(self):
        return os.path.join(self.export_root, TEMPLATE_DIR)

    def pipeline(self, task):
        return dali_loader.LinearPipeline(
            task, self.level, self.files, self.batch_size, seed=self.seed + TASK_SEED_OFFSET[task],
            file_root=self.export_root, shuffle=self.shuffle, intensity_aug=self.intensity_aug,
            num_threads=self.num_threads, device_id=self.device_id,
            prefetch_queue_depth=self.prefetch_queue_depth, shuffle_mode=self.shuffle_mode,
            **self.ranges[task])

    def dataset(self, task):
        """The task's DALIDataset (built once, cached)."""
        if task not in P.TASKS:
            raise ValueError("task must be one of {}, got {!r}".format(P.TASKS, task))
        if task not in self._datasets:
            self._datasets[task] = make_dali_dataset(self.pipeline(task), self.device_id)
        return self._datasets[task]

    def iterator(self, task):
        """A Python iterator over the task's dataset (created on the DALI GPU device)."""
        ds = self.dataset(task)
        with tf.device("/gpu:%d" % self.device_id):
            return iter(ds)
