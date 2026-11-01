"""Host stores of per-image arrays for the streaming engine (predict-script design 4.1, 6).

A store maps a name to one host array [N, ...]; the engine writes it chunk by chunk (`put`) and
reads chunks back (`get`). The dtype is kept (float16 activations stay float16).

* `RamStore`: `np.empty` preallocated in RAM.
* `DiskStore`: `numpy.lib.format.open_memmap(<dir>/<name>.npy, 'w+')` in a private
  subdirectory of the given store directory; `free` deletes the file and `close` removes the
  subdirectory (and the store directory itself if the store created it and it is empty). Use it
  as a context manager so the files are removed also on error.

`estimate_store_gb` is the host-memory arithmetic of design section 6 and `choose_store` the
`--store auto` rule of design 4.1. TF-free (numpy, `data_loading.params` and the model config
only), so the CLI and its tests run without TensorFlow.
"""
import collections
import logging
import os
import re
import shutil
import tempfile

import numpy as np

from data_loading import params as P

logger = logging.getLogger(__name__)

STORE_KINDS = ("auto", "ram", "disk")
BRANCHES = ("rigid", "affine", "deform", "affine+deform")
RAM_FRACTION = 0.7          # design 4.1: RAM iff estimate <= 0.7 x budget
CASCADE_LEVELS = (4, 2, 1)  # coarse -> fine; a level-L checkpoint runs the levels >= L
_NAME_RE = re.compile(r"[A-Za-z0-9_.\-]+")
# Host inputs per full-resolution voxel (design 6): image float32 + labels 2 B + mask 1 B.
_INPUT_BYTES_PER_VOXEL = 4 + 2 + 1
_DTYPE_BYTES = {"float16": 2, "float32": 4}


# ============================================================================ stores
class _Store(object):
    """Name -> host array [N, ...]. API: alloc, put, get, free, nbytes, close (design 4.1)."""

    kind = None

    def __init__(self):
        self._arrays = collections.OrderedDict()
        self._peak = 0
        self.choice = None  # set by `choose_store` (kind, estimate, budget)

    def alloc(self, name, shape, dtype):
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise ValueError("store names must match [A-Za-z0-9_.-]+, got {!r}".format(name))
        if name in self._arrays:
            raise ValueError("store array {!r} already allocated".format(name))
        shape = tuple(int(s) for s in shape)
        self._arrays[name] = self._new(name, shape, np.dtype(dtype))
        self._peak = max(self._peak, self.nbytes())

    def put(self, name, i0, chunk):
        arr = self._arrays[name]
        chunk = np.asarray(chunk)
        i0 = int(i0)
        i1 = i0 + int(chunk.shape[0])
        if i0 < 0 or i1 > arr.shape[0]:
            raise IndexError("chunk [{}, {}) outside {!r} of length {}".format(
                i0, i1, name, arr.shape[0]))
        arr[i0:i1] = chunk

    def get(self, name, i0, i1):
        return self._arrays[name][int(i0):int(i1)]

    def free(self, name):
        self._release(name, self._arrays.pop(name))

    def __contains__(self, name):
        return name in self._arrays

    def names(self):
        return list(self._arrays)

    def nbytes(self):
        return int(sum(a.nbytes for a in self._arrays.values()))

    def peak_nbytes(self):
        return int(self._peak)

    def close(self):
        for name in list(self._arrays):
            self.free(name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _new(self, name, shape, dtype):
        raise NotImplementedError

    def _release(self, name, arr):
        pass


class RamStore(_Store):
    """`np.empty` arrays in RAM; `get` returns a view."""

    kind = "ram"

    def _new(self, name, shape, dtype):
        return np.empty(shape, dtype)


class DiskStore(_Store):
    """`.npy` memmaps in a private subdirectory of ``store_dir``; `get` returns a copy."""

    kind = "disk"

    def __init__(self, store_dir):
        super().__init__()
        parent = os.path.abspath(str(store_dir))
        self._created_parent = parent if not os.path.isdir(parent) else None
        os.makedirs(parent, exist_ok=True)
        self.directory = tempfile.mkdtemp(prefix="bgr_stream_", dir=parent)

    def _path(self, name):
        return os.path.join(self.directory, name + ".npy")

    def _new(self, name, shape, dtype):
        return np.lib.format.open_memmap(self._path(name), mode="w+", dtype=dtype, shape=shape)

    def get(self, name, i0, i1):
        return np.array(self._arrays[name][int(i0):int(i1)])

    def _release(self, name, arr):
        del arr
        try:
            os.remove(self._path(name))
        except OSError:
            pass

    def close(self):
        try:
            super().close()
        finally:
            shutil.rmtree(self.directory, ignore_errors=True)
            if self._created_parent is not None:
                try:
                    os.rmdir(self._created_parent)
                except OSError:
                    pass  # not empty (other files) or already gone


# ============================================================================ geometry
def _pooled(shape):
    return tuple(int(s) // 2 for s in shape)   # AveragePooling3D(2), 'valid'


def _strided(shape):
    return tuple((int(s) + 1) // 2 for s in shape)  # Conv3D stride 2, 'same'


def level_geometry(level, model_cfg):
    """Per-image (spatial shape, channels) of every block output of a `GroupRegLevelModel` at
    ``level``: stem, l2..l6 (encoder; pool2 before l2 at level 1, pool3 before l3 at levels 2 / 1,
    l4-l6 stride 2) and d5..d1 (deform decoder, at the resolution of their skips L5, L4, L3, L2,
    stem). OrderedDict in execution order."""
    level = P.check_level(level)
    s = tuple(P.LEVEL_SHAPE[level])
    f = model_cfg.filters
    w = model_cfg.resolved_decoder_widths()
    s2 = _pooled(s) if level == 1 else s
    s3 = _pooled(s2) if level in (2, 1) else s2
    s4 = _strided(s3)
    s5 = _strided(s4)
    s6 = _strided(s5)
    return collections.OrderedDict([
        ("stem", (s, f[0])), ("l2", (s2, f[1])), ("l3", (s3, f[2])), ("l4", (s4, f[3])),
        ("l5", (s5, f[4])), ("l6", (s6, f[5])),
        ("d5", (s5, w[0])), ("d4", (s4, w[1])), ("d3", (s3, w[2])), ("d2", (s2, w[3])),
        ("d1", (s, w[4]))])


def _voxels(shape):
    return int(np.prod([int(v) for v in shape]))


# Store lifetimes of the pass schedule (design 4.3): (store written, stores freed after it).
_DEFORM_SCHEDULE = (("l2", ()), ("l3", ()), ("l4", ()), ("l5", ()), ("l6", ()),
                    ("d5", ("l6", "l5")), ("d4", ("d5", "l4")), ("d3", ("d4", "l3")),
                    ("d2", ("l2", "d3")), (None, ("d2",)))
_LINEAR_SCHEDULE = (("l2", ()), ("l3", ("l2",)), ("l4", ("l3",)), ("l5", ("l4",)),
                    ("l6", ("l5",)))


def _peak_activation_bytes(geo, act_bytes, schedule):
    live, peak = set(), 0
    for written, freed in schedule:
        if written is not None:
            live.add(written)
        peak = max(peak, sum(_voxels(geo[n][0]) * geo[n][1] * act_bytes for n in live))
        live -= set(freed)
    return peak


def _branch_set(branches):
    if isinstance(branches, str):
        branches = (branches,)
    branches = tuple(branches)
    bad = [b for b in branches if b not in BRANCHES]
    if bad or not branches:
        raise ValueError("branches must be a non-empty subset of {}, got {!r}".format(
            BRANCHES, branches))
    return branches


def estimate_store_gb(level, model_cfg, n, branches, trainable_dtype="float16"):
    """Host memory (GB, 1e9 bytes) of a prediction run, design section 6.

    Host inputs (n full-resolution images float32, labels 2 B, masks 1 B per voxel) plus the
    largest level of the cascade (levels 4 .. ``level``; frozen stages float32, the trainable
    stage at ``level`` in ``trainable_dtype`` 'float16' | 'float32'). Per level and branch:
      deform: largest simultaneous activation stores of the pass schedule (design 4.3) + x_in
        (float32, 1 ch) + v_raw (float32, 3 ch) + the previous level's fields (u, u_inv) + this
        level's fields when another level follows;
      rigid / affine: x_in + the two encoder stores alive at once;
      affine+deform: the larger of both, plus the affine-resampled full-resolution source
        (float32) during the deform branch.
    """
    level = P.check_level(level)
    if trainable_dtype not in _DTYPE_BYTES:
        raise ValueError("trainable_dtype must be one of {}, got {!r}".format(
            tuple(_DTYPE_BYTES), trainable_dtype))
    branches = _branch_set(branches)
    n = int(n)
    deform = any(b in ("deform", "affine+deform") for b in branches)
    linear = any(b != "deform" for b in branches)
    resampled = "affine+deform" in branches
    full_vox = _voxels(P.LEVEL_SHAPE[1])
    levels = [l for l in CASCADE_LEVELS if l >= level]
    worst, prev_field_vox = 0, 0
    for i, lv in enumerate(levels):
        geo = level_geometry(lv, model_cfg)
        vox = _voxels(P.LEVEL_SHAPE[lv])
        act = _DTYPE_BYTES[trainable_dtype] if lv == level else 4
        x_in = 4 * vox
        per_image = 0
        if linear:
            per_image = x_in + _peak_activation_bytes(geo, act, _LINEAR_SCHEDULE)
        if deform:
            d = (_peak_activation_bytes(geo, act, _DEFORM_SCHEDULE) + x_in + 12 * vox
                 + 24 * prev_field_vox + (24 * vox if i + 1 < len(levels) else 0))
            if resampled:
                d += 4 * full_vox
            per_image = max(per_image, d)
        worst = max(worst, per_image)
        prev_field_vox = vox
    total = n * (_INPUT_BYTES_PER_VOXEL * full_vox + worst)
    return total / 1e9


# ============================================================================ choice
def memory_budget_gb(environ=None, meminfo="/proc/meminfo"):
    """(budget GB, source): ``SLURM_MEM_PER_NODE`` (MB) when set, else ``MemAvailable`` of
    /proc/meminfo, else (None, None)."""
    env = os.environ if environ is None else environ
    value = env.get("SLURM_MEM_PER_NODE")
    if value:
        try:
            return int(value) * 2 ** 20 / 1e9, "SLURM_MEM_PER_NODE"
        except ValueError:
            logger.warning("SLURM_MEM_PER_NODE=%r is not an integer (MB); ignored", value)
    try:
        with open(meminfo) as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / 1e9, "MemAvailable"
    except (OSError, ValueError, IndexError):
        pass
    return None, None


def choose_store(kind, estimate_gb, store_dir=None):
    """Store for ``kind`` 'ram' | 'disk' | 'auto' (design 4.1). 'auto' = RAM iff ``estimate_gb``
    <= 0.7 x the budget of `memory_budget_gb`, else disk; RAM with a WARNING when no budget is
    readable. ``store_dir`` (needed for disk; the caller passes $TMPDIR or <out>/_stream_tmp).
    The decision is recorded in ``store.choice``."""
    if kind not in STORE_KINDS:
        raise ValueError("store kind must be one of {}, got {!r}".format(STORE_KINDS, kind))
    budget, source = None, None
    chosen = kind
    if kind == "auto":
        budget, source = memory_budget_gb()
        if budget is None:
            logger.warning("no memory budget readable (SLURM_MEM_PER_NODE, /proc/meminfo); "
                           "using the RAM store for an estimated %.1f GB", estimate_gb)
            chosen = "ram"
        else:
            chosen = "ram" if estimate_gb <= RAM_FRACTION * budget else "disk"
            logger.info("store auto: estimate %.1f GB vs %.2f x budget %.1f GB (%s) -> %s",
                        estimate_gb, RAM_FRACTION, budget, source, chosen)
    if chosen == "ram":
        store = RamStore()
    else:
        if store_dir is None:
            raise ValueError("the disk store needs store_dir")
        store = DiskStore(store_dir)
    store.choice = {"requested": kind, "kind": chosen, "estimate_gb": float(estimate_gb),
                    "budget_gb": budget, "budget_source": source,
                    "directory": getattr(store, "directory", None)}
    return store
