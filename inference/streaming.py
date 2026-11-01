"""Exact chunked execution of one `GroupRegLevelModel` (predict-script design 2.3, 4.2, 4.3, 4.6).

In inference mode (BatchNorm with moving statistics, no dropout) the group couples the images
only through group means (the group-concat convolutions, the SVF centering) and through the
linear decoder on the tiny bottleneck. `StreamedLevel` therefore runs the model block by block
("passes") over chunks of the group: each pass reads the previous block's output per chunk from a
host store (`inference.store`), applies the block with the group mean accumulated over ALL chunks
in the previous pass (float32 sums divided by N once), writes its output and accumulates the
next block's group mean. Results equal the model's own call up to float rounding (design 2.3).

Per-block primitive (design 4.2): a group block on a chunk is
``block.conv_with_group_mean(block.pre(x, False), mean, inner_clip=block.split_group)`` (the stem
has an identity ``pre`` and is never split, so only the outer fp16 clip applies, as in its concat
path); a non-group block (config B encoder) is ``clip_fp16(block.conv(block.pre(x, False)), C)``.
Casts are the model's: ``tf.cast(images, model.compute_dtype)``, the decoder's
``tf.cast(x, block.compute_dtype)`` / ``tf.cast(skip, x.dtype)`` per stage and
``svf_norm_act.compute_dtype`` before the head.

Pass schedule (deform task, design 4.3), store names prefixed per level (``L<level>_``):

  P0  x_in = input_fn(chunk) (float32 model input)        store x_in        S_x (stem)
  P1  stem(x, mean_x) -> pool2 (level 1)                  -                 S_l2
  P2  l2 -> pool3 (levels 2 / 1)                          store l2          S_l3
  P3-P5  l3, l4, l5                                       store l3, l4, l5  S of the next block
  P6  l6                                                  store l6          S_d5 (deform) or -
  P7-P9  d5, d4, d3 on concat(up(prev), skip)             store; free the used skip and prev
  P10 d2                                                  store d2; free l2, d3   S_d1 (stem recomputed)
  P11 d1 (stem recomputed) -> SVF head -> v_raw float32   store v_raw; free d2, x_in   S_v
  P12 v = v_raw - S_v / N; exp_and_inverse; optional compose_cascade with the previous level;
      store u, u_inv or hand them to a sink per chunk.

The stem output is never stored (recomputed from x_in in P1, P2, P10, P11); the d1 output is
never stored. Linear tasks run P0-P6 (only the latest encoder store kept; x_in freed after P2),
gather the bottleneck [N, 5, 7, 6, C] and call ``model.linear_decoder`` once on all N.

One tf.function per pass (it traces once per chunk length: at most two, the full chunk and the
remainder). Only ``block='single'`` models are supported (design 2.3).
"""
import logging

import numpy as np
import tensorflow as tf

from data_loading import params as P
from inference.store import level_geometry
from models import diffeo
from models import layers as L

logger = logging.getLogger(__name__)

INT32_LIMIT = 2 ** 31          # TF per-tensor element limit (int32 work_element_count)
FP16_MAX = 65504.0
LINEAR_TASKS = ("rigid", "affine")
DEFORM_TASK = "deform"
ENCODER_BLOCKS = ("stem", "l2", "l3", "l4", "l5", "l6")
DECODER_STAGES = ("d5", "d4", "d3", "d2", "d1")
DECODER_SKIPS = {"d5": "l5", "d4": "l4", "d3": "l3", "d2": "l2", "d1": "stem"}


# ============================================================================ element bound
def _voxels(shape):
    return int(np.prod([int(v) for v in shape]))


def _task_tuple(tasks):
    if isinstance(tasks, str):
        tasks = (tasks,)
    tasks = tuple(tasks)
    bad = [t for t in tasks if t not in LINEAR_TASKS + (DEFORM_TASK,)]
    if bad or not tasks:
        raise ValueError("tasks must be a non-empty subset of {}, got {!r}".format(
            LINEAR_TASKS + (DEFORM_TASK,), tasks))
    return tasks


def max_elements_per_image(level, model_cfg, tasks, split_group_conv):
    """(E_max, tensor name): the largest per-image element count of the level model's tensors.

    Counted at ``data_loading.params.LEVEL_SHAPE[level]``: every block input after pooling, the
    group concat (2 x input channels; only when not split, except the stem, which always
    concatenates [x, mean] = 2 channels in config A), every block output; for the deform task also
    each decoder stage's concat(up(x), skip), its group concat when not split, its output, and the
    3-channel deform fields. Encoder in config A has group blocks, config B none; the decoder
    always has them. Ties keep the first tensor in execution order.
    """
    level = P.check_level(level)
    tasks = _task_tuple(tasks)
    geo = level_geometry(level, model_cfg)
    group_enc = model_cfg.group_config == "A"
    split = bool(split_group_conv)
    s0, c0 = geo["stem"]
    v0 = _voxels(s0)
    items = [("input", v0)]
    if group_enc:
        items.append(("stem group concat", 2 * v0))
    items.append(("stem output", c0 * v0))
    prev = "stem"
    for name in ENCODER_BLOCKS[1:]:
        spatial, channels = geo[name]
        in_channels = geo[prev][1]
        if name in ("l2", "l3"):          # stride 1: the input is at the output resolution
            in_spatial = spatial
            pooled = (name == "l2" and level == 1) or (name == "l3" and level in (2, 1))
            if pooled:
                items.append(("pool%d output" % (2 if name == "l2" else 3),
                              in_channels * _voxels(in_spatial)))
        else:                             # stride 2: the input is the previous output
            in_spatial = geo[prev][0]
        if group_enc and not split:
            items.append((name + " group concat", 2 * in_channels * _voxels(in_spatial)))
        items.append((name + " output", channels * _voxels(spatial)))
        prev = name
    if DEFORM_TASK in tasks:
        prev = "l6"
        for name in DECODER_STAGES:
            spatial, channels = geo[name]
            concat = (geo[prev][1] + geo[DECODER_SKIPS[name]][1]) * _voxels(spatial)
            items.append((name + " skip concat", concat))
            if not split:
                items.append((name + " group concat", 2 * concat))
            items.append((name + " output", channels * _voxels(spatial)))
            prev = name
        items.append(("deform fields", 3 * v0))
    best = items[0]
    for item in items[1:]:
        if item[1] > best[1]:
            best = item
    return int(best[1]), best[0]


def chunk_cap(level, model_cfg, tasks):
    """Largest chunk with chunk x E_max < 2^31, E_max of the split form (the engine's)."""
    e_max, _ = max_elements_per_image(level, model_cfg, tasks, True)
    return int((INT32_LIMIT - 1) // e_max)


# ============================================================================ primitives
def _apply_block(block, x, mean):
    """One encoder / decoder block on a chunk with the group mean of ``block.pre`` over all N."""
    if block.group:
        return block.conv_with_group_mean(block.pre(x, training=False), mean,
                                          inner_clip=block.split_group)
    return L.clip_fp16(block.conv(block.pre(x, training=False)), block.fp16_act_clip)


def _pre_sum(block, x):
    """Chunk part of the group mean of ``block.pre(x)``: float32 sum over the chunk axis."""
    return tf.reduce_sum(tf.cast(block.pre(x, training=False), tf.float32), axis=0, keepdims=True)


def _max_abs(y):
    return tf.reduce_max(tf.abs(tf.cast(y, tf.float32)))


class _Sum(object):
    """Float32 group sum accumulated chunk by chunk; the mean is S / N cast once."""

    def __init__(self):
        self.s = None

    def add(self, t):
        self.s = t if self.s is None else self.s + t

    def mean(self, n, dtype):
        return tf.cast(self.s / float(n), dtype)


# ============================================================================ engine
class StreamedLevel(object):
    """Chunked inference of one `GroupRegLevelModel` (design 4.3).

    Args:
      model: a built GroupRegLevelModel (``config.block == 'single'``).
      store: an `inference.store` store (RAM or disk) for the per-image intermediates.
      chunk: images per chunk (None or <= 0 = the whole group); capped per task so that
        chunk x E_max < 2^31 (WARNING, recorded in ``self.capped``).
      lean: the ``lean`` flag of every diffeo call (integration, compose_cascade).
      record_fp16: record max |output| of every float16 block (``stats['fp16_max_abs']``).
      prefix: store-name prefix (default ``'L<level>_'``, so the levels of a cascade can share
        one store).
    """

    def __init__(self, model, store, chunk, lean=True, record_fp16=True, prefix=None):
        if model.config.block != "single":
            raise ValueError("the streaming engine supports block='single' only, got {!r}".format(
                model.config.block))
        self.model = model
        self.store = store
        self.chunk = chunk
        self.lean = bool(lean)
        self.record_fp16 = bool(record_fp16)
        self.level = model.level
        self.config = model.config
        self.prefix = "L%d_" % self.level if prefix is None else str(prefix)
        self.capped = {}
        self.last_stats = None
        self._enc = {name: getattr(model.encoder, name) for name in ENCODER_BLOCKS}
        self._dec = model.deform_decoder
        self._fns = {}

    # ------------------------------------------------------------------ helpers
    def _bounds(self, n, task):
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1, got {}".format(n))
        k = n if self.chunk is None or int(self.chunk) <= 0 else min(int(self.chunk), n)
        e_max, tensor = max_elements_per_image(self.level, self.config, (task,), True)
        if k * e_max >= INT32_LIMIT:
            cap = chunk_cap(self.level, self.config, (task,))
            if cap < 1:
                raise ValueError("one image of level {} has {} elements in '{}' (>= 2^31)".format(
                    self.level, e_max, tensor))
            logger.warning("level %d %s: chunk %d x %d elements (%s) >= 2^31; capped to %d",
                           self.level, task, k, e_max, tensor, cap)
            self.capped[task] = {"requested": k, "used": cap, "e_max": e_max, "tensor": tensor}
            k = cap
        return k, [(i0, min(i0 + k, n)) for i0 in range(0, n, k)]

    def _fn(self, key, body):
        fn = self._fns.get(key)
        if fn is None:
            fn = self._fns[key] = tf.function(body)
        return fn

    def _rec(self, block):
        return self.record_fp16 and block.compute_dtype == "float16"

    def _note(self, stats, name, value):
        m = stats["fp16_max_abs"]
        m[name] = max(m.get(name, 0.0), float(value))

    def _put(self, name, n, i0, t):
        """Copy a chunk (tensor or array) to the host store; allocated on the first chunk."""
        arr = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
        full = self.prefix + name
        if full not in self.store:
            self.store.alloc(full, (n,) + arr.shape[1:], arr.dtype)
        self.store.put(full, i0, arr)

    def _get(self, name, i0, i1):
        return self.store.get(self.prefix + name, i0, i1)

    def _free(self, *names):
        for name in names:
            self.store.free(self.prefix + name)

    def _new_stats(self, task, n, k):
        return {"task": task, "n": int(n), "chunk": int(k), "chunks": {}, "fp16_max_abs": {}}

    @staticmethod
    def _finish_stats(stats):
        stats["fp16_max_abs"] = {name: {"max_abs": m, "ratio": m / FP16_MAX}
                                 for name, m in stats["fp16_max_abs"].items()}
        return stats

    # ------------------------------------------------------------------ graph pieces
    def _x(self, x_in):
        return tf.cast(x_in, self.model.compute_dtype)

    def _stem(self, x_in, mean_x):
        return _apply_block(self._enc["stem"], self._x(x_in), mean_x)

    def _after_stem(self, y0):
        pool2 = self.model.encoder.pool2
        return pool2(y0) if pool2 is not None else y0

    def _stage_in(self, name, prev, skip):
        """concat(up(x), skip) of decoder stage ``name`` with the DeformDecoder's casts."""
        dec = self._dec
        block = getattr(dec, name)
        x = prev
        if name == "d5":
            x = tf.cast(x, dec.compute_dtype)
        x = tf.cast(x, block.compute_dtype)
        up = getattr(dec, "up_" + name)
        if up is not None:
            x = up(x)
        return tf.concat([x, tf.cast(skip, x.dtype)], axis=-1)

    def _head(self, f):
        dec = self._dec
        x_head = tf.cast(f, dec.svf_norm_act.compute_dtype)
        return tf.cast(dec.svf_head(dec.svf_norm_act(x_head, training=False)), tf.float32)

    def _next_sum(self, nxt, y, prev):
        """Group-sum part of the block after an encoder block (None if it has no group)."""
        if nxt == "d5":
            return _pre_sum(self._dec.d5, self._stage_in("d5", y, prev))
        if nxt is not None and self._enc[nxt].group:
            return _pre_sum(self._enc[nxt], y)
        return None

    # ------------------------------------------------------------------ pass bodies
    def _f_xsum(self):
        def body(x_in):
            return _pre_sum(self._enc["stem"], self._x(x_in))
        return self._fn("xsum", body)

    def _f_p1(self):
        enc = self._enc

        def body(x_in, mean_x):
            y0 = self._stem(x_in, mean_x)
            out = {}
            if enc["l2"].group:
                out["sum"] = _pre_sum(enc["l2"], self._after_stem(y0))
            if self._rec(enc["stem"]):
                out["max"] = _max_abs(y0)
            return out
        return self._fn("P1", body)

    def _f_p2(self):
        enc = self._enc
        pool3 = self.model.encoder.pool3

        def body(x_in, mean_x, mean):
            a = self._after_stem(self._stem(x_in, mean_x))
            y = _apply_block(enc["l2"], a, mean)
            out = {"y": y}
            if enc["l3"].group:
                out["sum"] = _pre_sum(enc["l3"], pool3(y) if pool3 is not None else y)
            if self._rec(enc["l2"]):
                out["max"] = _max_abs(y)
            return out
        return self._fn("P2", body)

    def _f_enc(self, name, nxt):
        block = self._enc[name]
        pool3 = self.model.encoder.pool3 if name == "l3" else None

        def body(prev, mean):
            b = pool3(prev) if pool3 is not None else prev
            y = _apply_block(block, b, mean)
            out = {"y": y}
            s = self._next_sum(nxt, y, prev)
            if s is not None:
                out["sum"] = s
            if self._rec(block):
                out["max"] = _max_abs(y)
            return out
        return self._fn(("enc", name, nxt), body)

    def _f_dec(self, name, nxt):
        block = getattr(self._dec, name)
        nblock = getattr(self._dec, nxt)

        def body(prev, skip, mean, nskip):
            y = _apply_block(block, self._stage_in(name, prev, skip), mean)
            out = {"y": y, "sum": _pre_sum(nblock, self._stage_in(nxt, y, nskip))}
            if self._rec(block):
                out["max"] = _max_abs(y)
            return out
        return self._fn(("dec", name), body)

    def _f_d2(self):
        block, d1 = self._dec.d2, self._dec.d1

        def body(prev, skip, mean, x_in, mean_x):
            y = _apply_block(block, self._stage_in("d2", prev, skip), mean)
            stem = self._stem(x_in, mean_x)
            out = {"y": y, "sum": _pre_sum(d1, self._stage_in("d1", y, stem))}
            if self._rec(block):
                out["max"] = _max_abs(y)
            return out
        return self._fn("P10", body)

    def _f_d1(self):
        d1 = self._dec.d1

        def body(d2, x_in, mean_x, mean):
            stem = self._stem(x_in, mean_x)
            f = _apply_block(d1, self._stage_in("d1", d2, stem), mean)
            v_raw = self._head(f)
            out = {"v_raw": v_raw, "sum": tf.reduce_sum(v_raw, axis=0, keepdims=True)}
            if self._rec(d1):
                out["max"] = _max_abs(f)
            return out
        return self._fn("P11", body)

    def _f_p12(self, cascade):
        steps = self.config.integration_steps
        lean = self.lean

        def body(v_raw, mean_v, u_prev=None, u_inv_prev=None):
            v = v_raw - mean_v
            u, u_inv = diffeo.exp_and_inverse(v, steps, lean=lean)
            if cascade:
                u, u_inv = diffeo.compose_cascade(u_prev, u_inv_prev, u, u_inv, lean=lean)
            return {"u": u, "u_inv": u_inv, "v": v, "v_raw": tf.identity(v_raw),
                    "max_v": tf.reduce_max(tf.abs(v))}
        return self._fn(("P12", bool(cascade)), body)

    def _f_linear(self, task):
        decoder = self.model.linear_decoder

        def body(bottleneck):
            return decoder(bottleneck, task=task, training=False)[0]
        return self._fn(("linear", task), body)

    # ------------------------------------------------------------------ passes
    def _p0(self, n, bounds, stats, input_fn):
        stem = self._enc["stem"]
        shape = tuple(self.model.input_shape_level)
        acc = _Sum()
        for i0, i1 in bounds:
            x = np.asarray(input_fn(i0, i1), np.float32)
            if x.shape != (i1 - i0,) + shape + (1,):
                raise ValueError("input_fn({}, {}) returned shape {}, expected {}".format(
                    i0, i1, x.shape, (i1 - i0,) + shape + (1,)))
            self._put("x_in", n, i0, x)
            if stem.group:
                acc.add(self._f_xsum()(x))
        stats["chunks"]["P0"] = list(bounds)
        return acc.mean(n, stem.compute_dtype) if stem.group else None

    def _encoder(self, n, bounds, stats, mean_x, deform):
        """P1-P6. Returns the group mean for d5 (deform) or None."""
        enc = self._enc
        mean = None
        if enc["l2"].group or self._rec(enc["stem"]):
            acc, f = _Sum(), self._f_p1()
            for i0, i1 in bounds:
                out = f(self._get("x_in", i0, i1), mean_x)
                if "sum" in out:
                    acc.add(out["sum"])
                if "max" in out:
                    self._note(stats, "stem", out["max"])
            stats["chunks"]["P1"] = list(bounds)
            if enc["l2"].group:
                mean = acc.mean(n, enc["l2"].compute_dtype)
        acc, f = _Sum(), self._f_p2()
        for i0, i1 in bounds:
            out = f(self._get("x_in", i0, i1), mean_x, mean)
            self._put("l2", n, i0, out["y"])
            if "sum" in out:
                acc.add(out["sum"])
            if "max" in out:
                self._note(stats, "l2", out["max"])
        stats["chunks"]["P2"] = list(bounds)
        if not deform:
            self._free("x_in")
        mean = acc.mean(n, enc["l3"].compute_dtype) if enc["l3"].group else None
        prev = "l2"
        nexts = (("l3", "l4"), ("l4", "l5"), ("l5", "l6"), ("l6", "d5" if deform else None))
        for p, (name, nxt) in enumerate(nexts, start=3):
            acc, f = _Sum(), self._f_enc(name, nxt)
            for i0, i1 in bounds:
                out = f(self._get(prev, i0, i1), mean)
                self._put(name, n, i0, out["y"])
                if "sum" in out:
                    acc.add(out["sum"])
                if "max" in out:
                    self._note(stats, name, out["max"])
            stats["chunks"]["P%d" % p] = list(bounds)
            if not deform:
                self._free(prev)
            if nxt is None:
                mean = None
            else:
                nblock = self._dec.d5 if nxt == "d5" else enc[nxt]
                mean = acc.mean(n, nblock.compute_dtype) if acc.s is not None else None
            prev = name
        return mean

    # ------------------------------------------------------------------ public
    def deform(self, n, input_fn, prev_fn=None, sink=None):
        """Deform task on a group of ``n`` images (passes P0-P12).

        input_fn(i0, i1) -> float32 [k, X, Y, Z, 1] model input of images [i0, i1) (the caller's
        pooling / pre-warp). prev_fn(i0, i1) -> (U, U_inv) previous-level fields at this level
        (voxels) for the cascade composition, or None. sink(i0, i1, dict(u, u_inv, v, v_raw))
        receives the GPU tensors per chunk (nothing stored); without a sink 'u' / 'u_inv' are
        stored (float32) and their store names returned.

        Returns {'u': name | None, 'u_inv': name | None, 'stats': {task, n, chunk, chunks{pass:
        boundaries}, fp16_max_abs{block: {max_abs, ratio}}, max_abs_v}}.
        """
        if self._dec is None:
            raise ValueError("task 'deform' needs ModelConfig(deform_decoder=True)")
        n = int(n)
        k, bounds = self._bounds(n, DEFORM_TASK)
        stats = self._new_stats(DEFORM_TASK, n, k)
        dec = self._dec
        mean_x = self._p0(n, bounds, stats, input_fn)
        mean = self._encoder(n, bounds, stats, mean_x, deform=True)
        # P7-P9: d5, d4, d3
        stages = (("d5", "l6", "d4", "l4", ("l6", "l5")),
                  ("d4", "d5", "d3", "l3", ("d5", "l4")),
                  ("d3", "d4", "d2", "l2", ("d4", "l3")))
        for p, (name, prev, nxt, nskip, freed) in enumerate(stages, start=7):
            acc, f = _Sum(), self._f_dec(name, nxt)
            skip = DECODER_SKIPS[name]
            for i0, i1 in bounds:
                out = f(self._get(prev, i0, i1), self._get(skip, i0, i1), mean,
                        self._get(nskip, i0, i1))
                self._put(name, n, i0, out["y"])
                acc.add(out["sum"])
                if "max" in out:
                    self._note(stats, name, out["max"])
            stats["chunks"]["P%d" % p] = list(bounds)
            self._free(*freed)
            mean = acc.mean(n, getattr(dec, nxt).compute_dtype)
        # P10: d2 (+ S_d1 with the stem recomputed)
        acc, f = _Sum(), self._f_d2()
        for i0, i1 in bounds:
            out = f(self._get("d3", i0, i1), self._get("l2", i0, i1), mean,
                    self._get("x_in", i0, i1), mean_x)
            self._put("d2", n, i0, out["y"])
            acc.add(out["sum"])
            if "max" in out:
                self._note(stats, "d2", out["max"])
        stats["chunks"]["P10"] = list(bounds)
        self._free("l2", "d3")
        mean = acc.mean(n, dec.d1.compute_dtype)
        # P11: d1 (stem recomputed) -> v_raw
        acc, f = _Sum(), self._f_d1()
        for i0, i1 in bounds:
            out = f(self._get("d2", i0, i1), self._get("x_in", i0, i1), mean_x, mean)
            self._put("v_raw", n, i0, out["v_raw"])
            acc.add(out["sum"])
            if "max" in out:
                self._note(stats, "d1", out["max"])
        stats["chunks"]["P11"] = list(bounds)
        self._free("d2", "x_in")
        mean_v = acc.s / float(n)
        # P12: centre, integrate, compose; store or sink
        f = self._f_p12(prev_fn is not None)
        max_v = 0.0
        for i0, i1 in bounds:
            v_raw = self._get("v_raw", i0, i1)
            if prev_fn is not None:
                u_prev, u_inv_prev = prev_fn(i0, i1)
                out = f(v_raw, mean_v, u_prev, u_inv_prev)
            else:
                out = f(v_raw, mean_v)
            max_v = max(max_v, float(out["max_v"]))
            if sink is None:
                self._put("u", n, i0, out["u"])
                self._put("u_inv", n, i0, out["u_inv"])
            else:
                sink(i0, i1, {"u": out["u"], "u_inv": out["u_inv"], "v": out["v"],
                              "v_raw": out["v_raw"]})
        stats["chunks"]["P12"] = list(bounds)
        self._free("v_raw")
        stats["max_abs_v"] = max_v
        self.last_stats = self._finish_stats(stats)
        names = (None, None) if sink is not None else (self.prefix + "u", self.prefix + "u_inv")
        return {"u": names[0], "u_inv": names[1], "stats": self.last_stats}

    def linear(self, n, input_fn, task):
        """Rigid / affine task: P0-P6, then the model's linear decoder once on the gathered
        bottleneck of all ``n`` images. Returns A_raw [n, 4, 4] float32 (before centering; the
        caller applies ``model.centering`` or ``cascade.compose_linear``). Stats in
        ``self.last_stats``."""
        if task not in LINEAR_TASKS:
            raise ValueError("task must be one of {}, got {!r}".format(LINEAR_TASKS, task))
        n = int(n)
        k, bounds = self._bounds(n, task)
        stats = self._new_stats(task, n, k)
        mean_x = self._p0(n, bounds, stats, input_fn)
        self._encoder(n, bounds, stats, mean_x, deform=False)
        bottleneck = self._get("l6", 0, n)
        a_raw = self._f_linear(task)(bottleneck)
        self._free("l6")
        self.last_stats = self._finish_stats(stats)
        return a_raw
