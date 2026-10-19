"""Augmentation curriculum and the shared global transform G (design 3.2, v3.5). numpy only.

Both features feed the DALI pipelines through one control file per pipeline,
``<run_dir>/control/<task>.npy`` (layout: `data_loading.params.CONTROL_FIELDS`), which the
pipeline re-reads every batch (`data_loading.dali_loader.read_control`). Both default OFF; with
both off no control file exists and the pipelines use their constant ranges.

* `LinearCurriculum`: per-image half-widths ramp linearly from the start to the end values over
  ``ramp_steps`` steps, then stay at the end values.
* `GlobalTransformSampler`: G = rotation by U(-rot_max, rot_max) degrees about an axis uniform on
  the sphere, plus a translation U(-trans_max, trans_max) mm per axis. One G per group (all
  images of a batch), resampled every ``resample_every`` steps. Deterministic per
  (seed, step // resample_every).
* `ControlWriter`: atomic writes (temp file in the same folder + os.replace), a version counter,
  no write when the values did not change.
* `TaskControl`: the control values of one task at a step (curriculum or the end ranges, plus G
  or the identity) and the write through its `ControlWriter`.

Everything is a pure function of the step, so a resumed run reproduces the curriculum and G.
"""
import math
import os
import tempfile
import time

import numpy as np

from data_loading import params as P

RANGE_KEYS = P.CONTROL_RANGE_KEYS


def _check_ranges(d, name):
    out = {}
    for k in RANGE_KEYS:
        if k not in d:
            raise ValueError("%s is missing %r" % (name, k))
        v = float(d[k])
        if not math.isfinite(v) or v < 0:
            raise ValueError("%s %s must be finite and >= 0, got %r" % (name, k, d[k]))
        out[k] = v
    return out


class LinearCurriculum:
    """Linear ramp of the per-image half-widths.

    Args:
      start, end: dicts with rotation_deg, translation_mm, log_scale, shear (start <= end).
      ramp_steps: steps of the ramp; values(step) = start + min(step / ramp_steps, 1) (end - start).
        ramp_steps = 0 gives the end values at every step.
    """

    def __init__(self, start, end, ramp_steps):
        self.start = _check_ranges(start, "curriculum start")
        self.end = _check_ranges(end, "curriculum end")
        for k in RANGE_KEYS:
            if self.start[k] > self.end[k]:
                raise ValueError("curriculum start %s = %g exceeds the end value %g"
                                 % (k, self.start[k], self.end[k]))
        self.ramp_steps = int(ramp_steps)
        if self.ramp_steps < 0:
            raise ValueError("ramp_steps must be >= 0, got %d" % self.ramp_steps)

    def fraction(self, step):
        if self.ramp_steps == 0:
            return 1.0
        return min(max(float(step) / self.ramp_steps, 0.0), 1.0)

    def values(self, step):
        f = self.fraction(step)
        return {k: self.start[k] + f * (self.end[k] - self.start[k]) for k in RANGE_KEYS}


def sample_global_params(rng, rot_max_deg, trans_max_mm):
    """One G: dict(axis (3,) unit, angle_deg, translation_mm (3,)) from a numpy Generator.

    Axis: normalized standard normal (uniform on the sphere); angle U(-rot_max, rot_max);
    translation U(-trans_max, trans_max) per axis (TF axis order x, y, z)."""
    axis = rng.normal(size=3)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    angle = float(rng.uniform(-rot_max_deg, rot_max_deg)) if rot_max_deg > 0 else 0.0
    t = (rng.uniform(-trans_max_mm, trans_max_mm, size=3) if trans_max_mm > 0
         else np.zeros(3))
    return {"axis": axis, "angle_deg": angle, "translation_mm": np.asarray(t, np.float64)}


def global_matrix_mm(axis, angle_deg, translation_mm):
    """G as a TF centered-mm 4x4 (float64): [R(axis, angle) | t], right-handed Rodrigues."""
    ax = np.asarray(axis, np.float64)
    ax = ax / np.linalg.norm(ax)
    th = math.radians(float(angle_deg))
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    g = np.eye(4)
    g[:3, :3] = np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * (k @ k)
    g[:3, 3] = np.asarray(translation_mm, np.float64)
    return g


class GlobalTransformSampler:
    """G parameters at a step, constant within blocks of ``resample_every`` steps.

    Args:
      rot_max_deg: angle half-width (degrees); trans_max_mm: translation half-width (mm).
      resample_every: K >= 1; block b = step // K uses np.random.default_rng([seed, b]).
      seed: base seed (the trainer uses a different seed per task).
    """

    def __init__(self, rot_max_deg, trans_max_mm, resample_every=5, seed=0):
        self.rot_max_deg = float(rot_max_deg)
        self.trans_max_mm = float(trans_max_mm)
        for name, v in (("rot_max_deg", self.rot_max_deg), ("trans_max_mm", self.trans_max_mm)):
            if not math.isfinite(v) or v < 0:
                raise ValueError("%s must be finite and >= 0, got %r" % (name, v))
        self.resample_every = int(resample_every)
        if self.resample_every < 1:
            raise ValueError("resample_every must be >= 1, got %d" % self.resample_every)
        self.seed = int(seed)

    def block(self, step):
        return int(step) // self.resample_every

    def params(self, step):
        b = self.block(step)
        out = sample_global_params(np.random.default_rng([self.seed, b]), self.rot_max_deg,
                                   self.trans_max_mm)
        out["block"] = b
        return out


def atomic_save_npy(path, array):
    """np.save to a temp file in the same folder, fsync, then os.replace (atomic on one
    filesystem: a concurrent reader sees the old or the new file in full). On Windows,
    os.replace can briefly fail while a scanner holds the new file; it is retried (as in the
    reference project's CurriculumState)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".control_", suffix=".npy", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, np.asarray(array))
            f.flush()
            os.fsync(f.fileno())
        last = None
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                last = None
                break
            except PermissionError as e:  # pragma: no cover  (Windows-only race)
                last = e
                time.sleep(0.02 * (attempt + 1))
        if last is not None:  # pragma: no cover
            raise last
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


class ControlWriter:
    """Writer of one control file. ``write(values)`` stores `params.control_vector(**values,
    version=v + 1)` atomically and returns True, or does nothing and returns False when the
    values equal the last written ones. ``version`` is the version of the file on disk (0 before
    the first write)."""

    def __init__(self, path):
        self.path = os.path.abspath(str(path))
        self.version = 0
        self._last = None

    def write(self, values):
        vec = P.control_vector(version=0, **values)
        if self._last is not None and np.array_equal(vec[:-1], self._last[:-1]):
            return False
        vec[-1] = float(self.version + 1)
        atomic_save_npy(self.path, vec)
        self.version += 1
        self._last = vec
        return True

    def read(self):
        return P.parse_control(np.load(self.path))


def identity_global():
    return {"g_axis": P.DEFAULT_G_AXIS, "g_angle_deg": 0.0, "g_translation_mm": (0.0, 0.0, 0.0)}


class TaskControl:
    """Control values of one task as a function of the step.

    Args:
      end_ranges: dict of the per-image half-widths used when there is no curriculum (and the
        curriculum's end values).
      curriculum: optional `LinearCurriculum`; sampler: optional `GlobalTransformSampler`
        (None -> G = identity).
      writer: `ControlWriter` (attach before `update`).
    """

    def __init__(self, end_ranges, curriculum=None, sampler=None, writer=None):
        self.end_ranges = _check_ranges(end_ranges, "end ranges")
        self.curriculum = curriculum
        self.sampler = sampler
        self.writer = writer

    def values_at(self, step):
        """Keyword arguments of `params.control_vector` (without version) at `step`."""
        out = dict(self.curriculum.values(step) if self.curriculum is not None else self.end_ranges)
        if self.sampler is None:
            out.update(identity_global())
        else:
            g = self.sampler.params(step)
            out.update(g_axis=tuple(g["axis"]), g_angle_deg=g["angle_deg"],
                       g_translation_mm=tuple(g["translation_mm"]))
        return out

    def update(self, step):
        """Write the values of `step` (no-op if unchanged). Returns (values, written)."""
        if self.writer is None:
            raise RuntimeError("TaskControl.update needs a writer")
        vals = self.values_at(step)
        return vals, self.writer.write(vals)
