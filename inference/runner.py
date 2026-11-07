"""Branch orchestration over the cascade levels (predict-script design 3.4, 4.4, 4.5).

Both runners take the full-resolution images I [N, 160, 224, 192] float32 (level-1 grid, host
numpy) and a ``--branches`` value, and hand the results per chunk to a sink:

    sink(i0, i1, res)   res = {'A': [k, 4, 4] float32 tensor or None,
                               'U1', 'U1_inv': the composed deform fields at level 1 (voxels) or None,
                               'u_ckpt', 'u_inv_ckpt': the same fields at the checkpoint level
                                   (after compose_cascade, before upsampling) or None,
                               'v', 'v_raw': the checkpoint level's residual SVF (centered / raw)}

`run` returns a summary dict: A for all N (numpy or None), chunk boundaries per level and task,
capped chunks, max |v| per level, fp16 headroom per block, timing per level stage and the TF GPU
peak per stage (None without a GPU).

Pipelines (design 3.4): ``rigid`` / ``affine`` = the linear cascade on the raw images;
``deform`` = the deform cascade on the raw images; ``affine+deform`` = the affine cascade, then the
deform cascade on the images resampled into the affine central space (I o A^-1, trilinear, zero
fill, at level 1); the sink gets A and the deform fields (composed later, design 5).

Cascade (design 4.4), level inputs derived from the level-1 source: x2 = pool(x1), x4 = pool(x2).
Linear: A4 = center4(A_raw4(x4)); A2 = compose_linear(A_raw2(x2 o A4^-1), A4, center2);
A = compose_linear(A_raw1(x1 o A2^-1), A2, center1). Deform: (u4, u4_inv) on x4; level 2 on
warp(x2, up(u4)), composed with compose_cascade(up(u4), up(u4_inv), u2, u2_inv); level 1 likewise
with the composed level-2 fields. A level-2 / level-4 checkpoint runs the levels up to its own.

`StreamedRunner`: every level stage is a `streaming.StreamedLevel` call (exact chunked
execution, frozen stages included); per-image results between levels live in a host store and
are freed as soon as they are consumed. `UnchunkedRunner`: each level stage is ONE tf.function on
all N with the trainer's own cascade ops (`Trainer._cascade_val_fn` / `_cascade_val_deform_fn`
without the baseline extras); limited by the int32 element bound (`loader.check_unchunked`).

Level selection (predict-levels design 4.2): both runners take ``plan`` (`inference.levels.LevelPlan`,
None = all levels). None or the normalised full plan runs the default path above, unchanged. Other
plans run `_linear_sel` / `_deform_sel`: truncated levels are not run (the last run level's fields
are upsampled to level 1, A passes unchanged); a skipped level contributes the identity (A = I,
U = U_inv = 0) and later stages run on warp(x, 0) / x o I^-1 and compose with it, as the trainer's
stage skip. u_ckpt / u_inv_ckpt / v / v_raw are None unless the checkpoint's own stage runs.

8-flip TTA (TTA design `.agents/tasks/tta/design.md` 3.4, 3.5, 4, 5): `StreamedRunner(...,
tta=inference.tta.TTAConfig)` runs every level in ``tta.levels`` once per flip (sequential per
level: the next level's input is pre-warped with the averaged result). Linear: `_linear_stage`
(gauge-free residual averaging, `tta.linear_tta_level`); deform: `_deform_stage`
(`StreamedLevel.deform_tta`). Without TTA at a level (or with `tta.IDENTITY_ONLY`) the unchanged
calls run. summary['tta'] = None without TTA, else {rows (per image, level, task: the
`tta.SPREAD_COLUMNS`), gauge {'l<lv>/<task>': [deg, mm]}, nonfinite_flips, timing_s, flips}.
`UnchunkedRunner` has no TTA.
"""
import contextlib
import logging
import time

import numpy as np
import tensorflow as tf

from data_loading import params as P
from inference import levels as LVS
from inference import loader as LD
from inference import streaming as ST
from inference import tta as TTA
from models import cascade as CS
from models import diffeo
from models import transforms as tfm

logger = logging.getLogger(__name__)

# --branches value -> (linear task or None, deform task or None) (design 3.4, decision D1).
BRANCHES = {"rigid": ("rigid", None), "affine": ("affine", None), "deform": (None, "deform"),
            "affine+deform": ("affine", "deform")}
DEFAULT_CHUNKS = {"l1": 2, "l2": 8, "l4": 32}
SRC_AFFINE = "src_affine"
_POOLS = {1: 0, 2: 1, 4: 2}          # 2x2x2 poolings from level 1 to the level
_ZERO = object()                     # level selection: skipped levels so far = identity fields


def branch_tasks(branches):
    """(linear task or None, deform task or None) of a ``--branches`` value."""
    if branches not in BRANCHES:
        raise ValueError("branches must be one of %s, got %r (rigid and affine are alternatives; "
                         "the deform branch was trained on affine-aligned inputs)"
                         % (", ".join(BRANCHES), branches))
    return BRANCHES[branches]


def _as_level1(images):
    x = np.asarray(images, np.float32)
    if x.ndim == 4:
        x = x[..., None]
    shape = tuple(P.LEVEL_SHAPE[1])
    if x.ndim != 5 or tuple(x.shape[1:4]) != shape or x.shape[-1] != 1 or x.shape[0] < 1:
        raise ValueError("images must be float32 [N, %d, %d, %d] (level-1 grid), got %s"
                         % (shape + (tuple(np.shape(images)),)))
    return x


def _pool_to(x1, level):
    x = tf.cast(x1, tf.float32)
    for _ in range(_POOLS[level]):
        x = CS.pool_to_coarser(x)
    return x


def _up_to_level1(u, level):
    for _ in range(_POOLS[level]):
        u = diffeo.upsample_displacement(u)
    return u


def _sync(t):
    """Wait for ``t`` (a tensor or a dict / tuple of tensors) so stage timings are real."""
    first = t
    if isinstance(t, dict):
        first = next(iter(t.values()))
    elif isinstance(t, (tuple, list)):
        first = t[0]
    tf.reshape(first, [-1])[:1].numpy()
    return t


def _bounds(n, chunk):
    k = n if chunk is None or int(chunk) <= 0 else min(int(chunk), n)
    return [(i0, min(i0 + k, n)) for i0 in range(0, n, k)]


class _Phases(object):
    """Wall time and TF GPU peak (MiB) per named stage (``reset_memory_stats`` before each)."""

    def __init__(self):
        self.timing = {}
        self.gpu_peak_mb = {}
        self.gpu = bool(tf.config.list_logical_devices("GPU"))

    @contextlib.contextmanager
    def __call__(self, name):
        if self.gpu:
            try:
                tf.config.experimental.reset_memory_stats("GPU:0")
            except (ValueError, RuntimeError):
                self.gpu = False
        t0 = time.time()
        try:
            yield
        finally:
            self.timing[name] = self.timing.get(name, 0.0) + time.time() - t0
            if self.gpu:
                try:
                    peak = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2.0 ** 20
                    self.gpu_peak_mb[name] = max(self.gpu_peak_mb.get(name, 0.0), float(peak))
                except (ValueError, RuntimeError):
                    pass


class _Runner(object):
    mode = None

    def __init__(self, loaded, plan=None):
        self.loaded = loaded
        self.level = loaded.level
        self.levels = loaded.levels
        self.models = loaded.stage_models()
        self._fns = {}
        if plan is not None and tuple(plan.ckpt_levels) != tuple(self.levels):
            raise ValueError("level plan for levels %s, checkpoint levels %s"
                             % (list(plan.ckpt_levels), list(self.levels)))
        self.plan = plan

    def _is_default(self):
        """True iff the unchanged default path runs (no plan or the normalised full plan)."""
        return self.plan is None or self.plan.default

    def _plan_dict(self):
        if self.plan is not None:
            return self.plan.to_dict()
        return LVS.plan(self.levels, None).to_dict()

    def _fn(self, key, body, signature=None):
        fn = self._fns.get(key)
        if fn is None:
            fn = self._fns[key] = tf.function(body, input_signature=signature)
        return fn

    def _new_summary(self, n, branches, chunk_sizes):
        return {"mode": self.mode, "level": self.level, "n": int(n), "branches": branches,
                "A": None, "chunks": {"l%d" % lv: {} for lv in self.levels},
                "chunk_sizes": chunk_sizes, "capped": {}, "stats": {"max_abs_v": {}},
                "fp16_max_abs": {}, "timing_s": None, "gpu_peak_mb": None,
                "levels": self._plan_dict()}

    def _new_summary_sel(self, n, branches, chunk_sizes):
        """Summary of a non-default plan: chunk bounds for the run levels only."""
        summary = self._new_summary(n, branches, chunk_sizes)
        summary["chunks"] = {"l%d" % lv: {} for lv in self.plan.run}
        return summary

    def _check(self, images, branches):
        x1 = _as_level1(images)
        lin, dfm = branch_tasks(branches)
        if dfm is not None and self.models[self.level].deform_decoder is None:
            raise ValueError("branches %r need a deform decoder; this model has none" % branches)
        return x1, lin, dfm

    @staticmethod
    def _res(A, fields=None):
        res = {"A": A, "U1": None, "U1_inv": None, "u_ckpt": None, "u_inv_ckpt": None,
               "v": None, "v_raw": None}
        if fields is not None:
            res.update(fields)
        return res


# ============================================================================ streamed
class StreamedRunner(_Runner):
    """Chunked cascade (design 4.4, 4.5). ``chunks`` {'l1', 'l2', 'l4'}: images per chunk at each
    level (0 / None = whole group; capped by the int32 bound inside `StreamedLevel`). ``store``: an
    `inference.store` store shared by all levels (per-level name prefixes)."""

    mode = "streamed"

    def __init__(self, loaded, store, chunks=None, lean=True, record_fp16=True, plan=None, tta=None):
        super(StreamedRunner, self).__init__(loaded, plan)
        self.store = store
        self.chunks = dict(DEFAULT_CHUNKS)
        self.chunks.update(chunks or {})
        self.lean = bool(lean)
        self.record_fp16 = bool(record_fp16)
        self._engines = {}
        self.tta = tta
        self._tta_reset()

    # ---------------------------------------------------------------- TTA (TTA design 3.4, 3.5, 5)
    def _tta_at(self, lv):
        """True iff ``lv`` is a TTA level of this runner."""
        return self.tta is not None and int(lv) in self.tta.levels

    def _tta_active(self, lv):
        """TTA at ``lv`` with more than the identity flip (`IDENTITY_ONLY` runs today's code)."""
        return self._tta_at(lv) and tuple(self.tta.flips) != TTA.IDENTITY_ONLY

    def _tta_reset(self):
        self._tta_rows = []
        self._tta_gauge = {}
        self._tta_nonfinite = {}
        self._tta_timing = {}
        self._tta_S = None
        self._tta_masks = None

    def _tta_summary(self):
        if self.tta is None:
            return None
        return {"rows": list(self._tta_rows), "gauge": dict(self._tta_gauge),
                "nonfinite_flips": dict(self._tta_nonfinite), "timing_s": dict(self._tta_timing),
                "flips": [list(f) for f in self.tta.flips]}

    def _second_moments(self, x1):
        """Per-image S_i = brain_second_moment(x1_i > 0, 1.0) [N, 4, 4] (linear spread), once per run."""
        if self._tta_S is None:
            self._tta_S = np.stack([tfm.brain_second_moment(np.asarray(x1[i, ..., 0]) > 0, 1.0)
                                    for i in range(x1.shape[0])])
        return self._tta_S

    @staticmethod
    def _merge_stats(acc, st):
        """fp16 maxima: max over passes; chunk boundaries of the first pass."""
        if acc is None:
            return {k: (dict(v) if isinstance(v, dict) else v) for k, v in st.items()}
        fp = acc["fp16_max_abs"]
        for block, rec in st["fp16_max_abs"].items():
            if block not in fp or rec["max_abs"] > fp[block]["max_abs"]:
                fp[block] = dict(rec)
        return acc

    def _linear_stage(self, lv, eng, model, n, input_fn, task, a_prev, x1):
        """One linear level: today's two lines without TTA at ``lv``; else design 2.3 (8 passes
        on the flipped level input, unflip, gauge-free residual averaging) plus the spread rows."""
        if not self._tta_active(lv):
            a_raw = eng.linear(n, input_fn, task)
            if a_prev is None:
                return tf.cast(model.centering(a_raw), tf.float32)
            return CS.compose_linear(a_raw, a_prev, model.centering)
        flips = self.tta.flips
        M_flips, stats, timing = [], None, []
        for flip in flips:
            t0 = time.time()
            a_raw_f = eng.linear(n, lambda i0, i1, flip=flip: TTA.flip_np(input_fn(i0, i1), flip), task)
            M_flips.append(TTA.unflip_matrices(a_raw_f, flip))
            stats = self._merge_stats(stats, eng.last_stats)
            timing.append(time.time() - t0)
        eng.last_stats = stats
        res = TTA.linear_tta_level(M_flips, a_prev, model.centering, task, flips)
        key = "l%d/%s" % (lv, task)
        self._tta_timing[key] = timing
        self._tta_nonfinite[key] = list(res["nonfinite"])
        for f_idx in res["nonfinite"]:
            logger.warning("level %d %s TTA flip %d %s: non-finite raw parameters / A_f / residual",
                           lv, task, f_idx, list(flips[f_idx]))
        A_bar = np.asarray(res["A"], np.float64)
        nan = float("nan")
        with np.errstate(invalid="ignore", divide="ignore"):
            try:
                sp = TTA.linear_spread(res["A_flips"], A_bar, self._second_moments(x1), res["params"],
                                       res["p_bar"], task, res["mean_diag"])
            except (ValueError, np.linalg.LinAlgError) as e:
                logger.warning("level %d %s TTA spread not computed: %s", lv, task, e)
                sp = {c: np.full(n, nan) for c in TTA.SPREAD_COLUMNS if c != "spread_p99_mm"}
            try:
                gauge = TTA.gauge_size(TTA.gauge_barycenters(M_flips, a_prev, model.config.centering_iters))
            except (ValueError, np.linalg.LinAlgError, tf.errors.InvalidArgumentError) as e:
                logger.warning("level %d %s TTA gauge not computed: %s", lv, task, e)
                gauge = (nan, nan)
        self._tta_gauge[key] = [float(gauge[0]), float(gauge[1])]
        for i in range(n):
            row = {"image": i, "level": int(lv), "task": task}
            for c in TTA.SPREAD_COLUMNS:
                row[c] = float(sp[c][i]) if c in sp else None
            self._tta_rows.append(row)
            dev, gap = row["readout_dev_deg"], row["rot_gap"]
            if (dev is not None and dev > 1.0) or (gap is not None and gap < 1e-3):
                logger.warning("level %d %s TTA image %d: rotation average ambiguous (readout_dev_deg "
                               "%s, rot_gap %s)", lv, task, i, dev, gap)
        return res["A"]

    def _deform_stage(self, lv, eng, n, input_fn, prev_fn, sink):
        """One deform level: `StreamedLevel.deform` without TTA at ``lv``, else `deform_tta` with
        the level mask, plus the spread rows."""
        if not self._tta_active(lv):
            return eng.deform(n, input_fn, prev_fn=prev_fn, sink=sink)
        res = eng.deform_tta(n, input_fn, self.tta.flips, prev_fn=prev_fn, sink=sink,
                             mask_fn=self._mask_fn(lv))
        st = res["stats"]["tta"]
        key = "l%d/deform" % lv
        self._tta_timing[key] = list(st["timing_s"])
        self._tta_nonfinite[key] = list(st["nonfinite_flips"])
        for i in range(n):
            row = {"image": i, "level": int(lv), "task": "deform"}
            for c in TTA.SPREAD_COLUMNS:
                row[c] = float(st["spread"][c][i]) if c in st["spread"] else None
            self._tta_rows.append(row)
        return res

    def _deform_stage_freeing(self, lv, eng, n, input_fn, prev_fn, sink, last_with_src):
        """`_deform_stage`; at a TTA last level of affine+deform (``last_with_src``) 'src_affine'
        lives through all flip passes and is freed afterwards, also on errors (design 3.4)."""
        if not (last_with_src and self._tta_active(lv)):
            return self._deform_stage(lv, eng, n, input_fn, prev_fn, sink)
        try:
            return self._deform_stage(lv, eng, n, input_fn, prev_fn, sink)
        finally:
            if SRC_AFFINE in self.store:
                self.store.free(SRC_AFFINE)

    def _deform_masks(self, n, src, levels):
        """Level masks of the deform spread (design 3.4) when a deform level is a TTA level:
        m1 = mean_i 1[src_i > 0] >= 0.5 (one chunked pass over src, chunk l1), pooled to each
        level and thresholded at 0.5; uint8 [X, Y, Z] per level."""
        self._tta_masks = None
        if not any(self._tta_active(lv) for lv in levels):
            return
        acc = None
        for i0, i1 in _bounds(n, self.chunks["l1"]):
            s = np.sum(np.asarray(src(i0, i1), np.float32) > 0, axis=0, dtype=np.float64)
            acc = s if acc is None else acc + s
        m = tf.constant(((acc / float(n)) >= 0.5).astype(np.float32)[None])      # [1, X, Y, Z, 1]
        masks = {}
        for lv in sorted(_POOLS):
            x = m
            for _ in range(_POOLS[lv]):
                x = CS.pool_to_coarser(x)
            masks[lv] = (x.numpy()[0, ..., 0] >= 0.5).astype(np.uint8)
        self._tta_masks = masks

    def _mask_fn(self, lv):
        mask = None if self._tta_masks is None else self._tta_masks.get(int(lv))
        if mask is None:
            return None
        m = mask.astype(np.float32)[None, ..., None]

        def mask_fn(i0, i1):
            return np.broadcast_to(m, (i1 - i0,) + m.shape[1:])
        return mask_fn

    def _engine(self, level):
        eng = self._engines.get(level)
        if eng is None:
            eng = self._engines[level] = ST.StreamedLevel(
                self.models[level], self.store, self.chunks["l%d" % level], lean=self.lean,
                record_fp16=self.record_fp16)
        return eng

    # ---------------------------------------------------------------- per-chunk graph pieces
    def _pool_fn(self, level):
        return self._fn(("pool", level), lambda x1: _pool_to(x1, level))

    def _prewarp_linear_fn(self, level):
        sp = P.LEVEL_SPACING[level]
        return self._fn(("prewarp_linear", level),
                        lambda x1, a_prev: CS.prewarp_linear(_pool_to(x1, level), a_prev, sp))

    def _prewarp_deform_fn(self, level):
        lean = self.lean

        def body(x1, u_prev):
            u_up = diffeo.upsample_displacement(u_prev)
            return diffeo.warp(_pool_to(x1, level), u_up, lean=lean)
        return self._fn(("prewarp_deform", level), body)

    def _up_fn(self):
        return self._fn("up", lambda u, u_inv: (diffeo.upsample_displacement(u),
                                                diffeo.upsample_displacement(u_inv)))

    def _to_level1_fn(self, level):
        return self._fn(("to_l1", level), lambda u, u_inv: (_up_to_level1(u, level),
                                                            _up_to_level1(u_inv, level)))

    def _record(self, summary, level, task, eng):
        st = eng.last_stats
        key = "l%d" % level
        summary["chunks"][key][task] = list(st["chunks"]["P0"])
        if eng.capped:
            summary["capped"][key] = dict(eng.capped)
        fp = summary["fp16_max_abs"]
        for block, rec in st["fp16_max_abs"].items():
            cur = fp.setdefault(key, {}).get(block)
            if cur is None or rec["max_abs"] > cur["max_abs"]:
                fp[key][block] = dict(rec)
        if "max_abs_v" in st:
            summary["stats"]["max_abs_v"][key] = float(st["max_abs_v"])

    # ---------------------------------------------------------------- branches
    def _linear(self, n, x1, task, summary, phases):
        a_prev = None
        for lv in self.levels:
            model, eng = self.models[lv], self._engine(lv)
            if a_prev is None:
                pool = self._pool_fn(lv)

                def input_fn(i0, i1, pool=pool):
                    return pool(x1[i0:i1])
            else:
                prewarp = self._prewarp_linear_fn(lv)

                def input_fn(i0, i1, prewarp=prewarp, a_prev=a_prev):
                    return prewarp(x1[i0:i1], a_prev[i0:i1])
            with phases("%s:l%d" % (task, lv)):
                a = self._linear_stage(lv, eng, model, n, input_fn, task, a_prev, x1)
            self._record(summary, lv, task, eng)
            a_prev = a
        return a_prev

    def _resample(self, n, x1, A, phases):
        """Store 'src_affine' = I o A^-1 at level 1 (trilinear, zero fill), per l1 chunk."""
        fn = self._fn("resample", lambda x, a: CS.prewarp_linear(x, a, P.LEVEL_SPACING[1]))
        with phases("affine+deform:resample"):
            self.store.alloc(SRC_AFFINE, x1.shape, np.float32)
            for i0, i1 in _bounds(n, self.chunks["l1"]):
                self.store.put(SRC_AFFINE, i0, fn(x1[i0:i1], A[i0:i1]).numpy())

    def _deform(self, n, src, A, sink, summary, phases, free_src):
        last = self.levels[-1]
        prev = None                                     # store names of the previous level's fields
        self._deform_masks(n, src, self.levels)
        for lv in self.levels:
            eng = self._engine(lv)
            is_last = lv == last
            if prev is None:
                pool = self._pool_fn(lv)

                def input_fn(i0, i1, pool=pool):
                    return pool(src(i0, i1))
                prev_fn = None
            else:
                prewarp, up = self._prewarp_deform_fn(lv), self._up_fn()

                def input_fn(i0, i1, prewarp=prewarp, name=prev[0]):
                    return prewarp(src(i0, i1), self.store.get(name, i0, i1))

                def prev_fn(i0, i1, up=up, names=prev):
                    return up(self.store.get(names[0], i0, i1), self.store.get(names[1], i0, i1))
            if is_last and free_src and not self._tta_active(lv):
                inner = input_fn

                def input_fn(i0, i1, inner=inner):     # free the source after the last P0 chunk
                    x = inner(i0, i1)
                    if i1 == n:
                        self.store.free(SRC_AFFINE)
                    return x
            level_sink = self._level_sink(lv, A, sink) if is_last else None
            with phases("deform:l%d" % lv):
                res = self._deform_stage_freeing(lv, eng, n, input_fn, prev_fn, level_sink,
                                                 is_last and free_src)
            self._record(summary, lv, "deform", eng)
            if prev is not None:
                for name in prev:
                    self.store.free(name)
            prev = None if is_last else (res["u"], res["u_inv"])

    def _level_sink(self, level, A, sink):
        to_l1 = self._to_level1_fn(level)

        def level_sink(i0, i1, d):
            if sink is None:
                return
            U1, U1_inv = (d["u"], d["u_inv"]) if level == 1 else to_l1(d["u"], d["u_inv"])
            sink(i0, i1, self._res(None if A is None else A[i0:i1],
                                   {"U1": U1, "U1_inv": U1_inv, "u_ckpt": d["u"],
                                    "u_inv_ckpt": d["u_inv"], "v": d["v"], "v_raw": d["v_raw"]}))
        return level_sink

    def run(self, images, branches, sink=None):
        """Run ``branches`` on images [N, 160, 224, 192] (float32); results go to ``sink`` per
        chunk (see the module docstring). Returns the summary dict."""
        if not self._is_default():
            return self._run_sel(images, branches, sink)
        x1, lin, dfm = self._check(images, branches)
        n = x1.shape[0]
        sizes = {"l%d" % lv: self.chunks["l%d" % lv] for lv in self.levels}
        summary = self._new_summary(n, branches, sizes)
        phases = _Phases()
        self._tta_reset()
        A = None
        if lin is not None:
            A = self._linear(n, x1, lin, summary, phases)
        if dfm is not None:
            if A is not None:
                self._resample(n, x1, A, phases)

                def src(i0, i1):
                    return self.store.get(SRC_AFFINE, i0, i1)
            else:
                def src(i0, i1):
                    return x1[i0:i1]
            self._deform(n, src, A, sink, summary, phases, free_src=A is not None)
        elif sink is not None:
            for i0, i1 in _bounds(n, self.chunks["l1"]):
                sink(i0, i1, self._res(A[i0:i1]))
        summary["A"] = None if A is None else A.numpy()
        summary["timing_s"] = phases.timing
        summary["gpu_peak_mb"] = phases.gpu_peak_mb if phases.gpu else None
        summary["tta"] = self._tta_summary()
        return summary

    # ---------------------------------------------------------------- level selection (predict-levels)
    # Used only by a non-default plan; the default path above is not touched (design 4.2).
    def _prewarp_deform_steps_fn(self, level, steps):
        lean = self.lean

        def body(x1, u_prev):
            u_up = u_prev
            for _ in range(steps):
                u_up = diffeo.upsample_displacement(u_up)
            return diffeo.warp(_pool_to(x1, level), u_up, lean=lean)
        return self._fn(("prewarp_deform_steps", level, steps), body)

    def _up_steps_fn(self, steps):
        def body(u, u_inv):
            for _ in range(steps):
                u, u_inv = diffeo.upsample_displacement(u), diffeo.upsample_displacement(u_inv)
            return u, u_inv
        return self._fn(("up_steps", steps), body)

    def _prewarp_deform_zero_fn(self, level):
        """warp(pool(x1), 0) at ``level``: the trainer's pre-warp of a skipped-identity field."""
        lean, shape = self.lean, list(P.LEVEL_SHAPE[level])

        def body(x1):
            x = _pool_to(x1, level)
            zero = tf.zeros([tf.shape(x)[0]] + shape + [3], tf.float32)
            return diffeo.warp(x, zero, lean=lean)
        return self._fn(("prewarp_deform_zero", level), body)

    def _zero_fields_fn(self, level):
        """(0, 0) fields [k, X, Y, Z, 3] at ``level`` (k a Python int)."""
        shape = list(P.LEVEL_SHAPE[level])

        def body(k):
            z = tf.zeros([k] + shape + [3], tf.float32)
            return z, tf.zeros_like(z)
        return self._fn(("zero_fields", level), body)

    def _linear_sel(self, n, x1, task, summary, phases):
        """Linear cascade of a non-default plan: a skipped level gives A = I (bitwise
        `cascade.skip_linear`) before any level ran, or keeps the last A (`skip_to`, '4,1')."""
        plan = self.plan
        a_prev = None
        ran = False
        for lv in self.levels:
            kind = plan.kind(lv)
            if kind == "truncated":
                break
            if kind == "skipped":
                if not ran:
                    a_prev = tf.eye(4, batch_shape=[n], dtype=tf.float32)
                continue
            model, eng = self.models[lv], self._engine(lv)
            if a_prev is None:
                pool = self._pool_fn(lv)

                def input_fn(i0, i1, pool=pool):
                    return pool(x1[i0:i1])
            else:
                prewarp = self._prewarp_linear_fn(lv)

                def input_fn(i0, i1, prewarp=prewarp, a_prev=a_prev):
                    return prewarp(x1[i0:i1], a_prev[i0:i1])
            with phases("%s:l%d" % (task, lv)):
                a = self._linear_stage(lv, eng, model, n, input_fn, task, a_prev, x1)
            self._record(summary, lv, task, eng)
            a_prev = a
            ran = True
        return a_prev

    def _deform_sel(self, n, src, A, sink, summary, phases, free_src):
        """Deform cascade of a non-default plan. ``prev``: None (nothing yet), _ZERO (identity so
        far: input warp(pool(x), 0), composition with (0, 0) as the trainer's skip) or (store
        names, level) of the last run level's fields."""
        plan = self.plan
        last = plan.last
        prev = None
        self._deform_masks(n, src, plan.run)
        for lv in self.levels:
            kind = plan.kind(lv)
            if kind == "truncated":
                break
            if kind == "skipped":
                if prev is None:
                    prev = _ZERO
                continue
            eng = self._engine(lv)
            is_last = lv == last
            if prev is None:
                pool = self._pool_fn(lv)

                def input_fn(i0, i1, pool=pool):
                    return pool(src(i0, i1))
                prev_fn = None
            elif prev is _ZERO:
                prewarp0, zeros = self._prewarp_deform_zero_fn(lv), self._zero_fields_fn(lv)

                def input_fn(i0, i1, prewarp0=prewarp0):
                    return prewarp0(src(i0, i1))

                def prev_fn(i0, i1, zeros=zeros):
                    return zeros(i1 - i0)
            else:
                names, lp = prev
                steps = _POOLS[lp] - _POOLS[lv]
                if steps == 1:
                    prewarp, up = self._prewarp_deform_fn(lv), self._up_fn()
                else:
                    prewarp, up = self._prewarp_deform_steps_fn(lv, steps), self._up_steps_fn(steps)

                def input_fn(i0, i1, prewarp=prewarp, name=names[0]):
                    return prewarp(src(i0, i1), self.store.get(name, i0, i1))

                def prev_fn(i0, i1, up=up, names=names):
                    return up(self.store.get(names[0], i0, i1), self.store.get(names[1], i0, i1))
            if is_last and free_src and not self._tta_active(lv):
                inner = input_fn

                def input_fn(i0, i1, inner=inner):     # free the source after the last P0 chunk
                    x = inner(i0, i1)
                    if i1 == n:
                        self.store.free(SRC_AFFINE)
                    return x
            level_sink = (self._level_sink_sel(lv, A, sink, ckpt_fields=lv == self.level)
                          if is_last else None)
            with phases("deform:l%d" % lv):
                res = self._deform_stage_freeing(lv, eng, n, input_fn, prev_fn, level_sink,
                                                 is_last and free_src)
            self._record(summary, lv, "deform", eng)
            if prev is not None and prev is not _ZERO:
                for name in prev[0]:
                    self.store.free(name)
            prev = None if is_last else ((res["u"], res["u_inv"]), lv)

    def _level_sink_sel(self, level, A, sink, ckpt_fields):
        """`_level_sink` of a non-default plan: u_ckpt, u_inv_ckpt, v, v_raw only when the
        checkpoint's trainable stage ran (``ckpt_fields``), else None (design 4.2 sink rule)."""
        to_l1 = None if level == 1 else self._to_level1_fn(level)

        def level_sink(i0, i1, d):
            if sink is None:
                return
            U1, U1_inv = (d["u"], d["u_inv"]) if level == 1 else to_l1(d["u"], d["u_inv"])
            fields = {"U1": U1, "U1_inv": U1_inv}
            if ckpt_fields:
                fields.update(u_ckpt=d["u"], u_inv_ckpt=d["u_inv"], v=d["v"], v_raw=d["v_raw"])
            sink(i0, i1, self._res(None if A is None else A[i0:i1], fields))
        return level_sink

    def _run_sel(self, images, branches, sink):
        x1, lin, dfm = self._check(images, branches)
        n = x1.shape[0]
        sizes = {"l%d" % lv: self.chunks["l%d" % lv] for lv in self.plan.run}
        summary = self._new_summary_sel(n, branches, sizes)
        phases = _Phases()
        self._tta_reset()
        A = None
        if lin is not None:
            A = self._linear_sel(n, x1, lin, summary, phases)
        if dfm is not None:
            if A is not None:
                self._resample(n, x1, A, phases)

                def src(i0, i1):
                    return self.store.get(SRC_AFFINE, i0, i1)
            else:
                def src(i0, i1):
                    return x1[i0:i1]
            self._deform_sel(n, src, A, sink, summary, phases, free_src=A is not None)
        elif sink is not None:
            for i0, i1 in _bounds(n, self.chunks["l1"]):
                sink(i0, i1, self._res(A[i0:i1]))
        summary["A"] = None if A is None else A.numpy()
        summary["timing_s"] = phases.timing
        summary["gpu_peak_mb"] = phases.gpu_peak_mb if phases.gpu else None
        summary["tta"] = self._tta_summary()
        return summary


# ============================================================================ unchunked
class UnchunkedRunner(_Runner):
    """Every level stage as ONE tf.function on all N (design 4.4 unchunked mode): the trainer's
    cascade ops (frozen ``stage.linear`` / ``chain.linear_stages``, ``CS.prewarp_linear``,
    ``model(x_in, task, training=False)['A_raw']``, ``CS.compose_linear``; frozen
    ``stage.deform`` / ``chain.deform``, warp(x, U), ``model(x_in, 'deform')``,
    ``diffeo.compose_cascade``). The group size is bounded by `loader.check_unchunked`."""

    mode = "unchunked"

    def __init__(self, loaded, lean=True, plan=None):
        super(UnchunkedRunner, self).__init__(loaded, plan)
        self.lean = bool(lean)

    def _spec(self, n, level, channels=1):
        return tf.TensorSpec((n,) + tuple(P.LEVEL_SHAPE[level]) + (channels,), tf.float32)

    def _pool(self, x1, level, n, phases):
        if level == 1:
            return x1
        fn = self._fn(("pool", level, n), lambda x: _pool_to(x, level), [self._spec(n, 1)])
        with phases("pool"):
            return _sync(fn(x1))

    def _linear(self, x1, task, n, phases):
        model, frozen, lv = self.loaded.model, self.loaded.frozen, self.level
        x = self._pool(x1, lv, n, phases)
        spec, aspec = self._spec(n, lv), tf.TensorSpec((n, 4, 4), tf.float32)
        if frozen is None:
            fn = self._fn(("linear", task, n), lambda x: tf.cast(
                model(x, task=task, training=False)["A"], tf.float32), [spec])
            with phases("%s:l%d" % (task, lv)):
                return _sync(fn(x))
        chain = lv == 1                                 # FrozenChain (level 1) or FrozenStage

        def frozen_body(x):
            if chain:
                return frozen.linear_stages(x, task)["A"]
            return frozen.linear(x, task)
        sp = P.LEVEL_SPACING[lv]

        def body(x, a_prev):
            x_in = CS.prewarp_linear(x, a_prev, sp)
            out = model(x_in, task=task, training=False)
            return CS.compose_linear(out["A_raw"], a_prev, model.centering)
        f_frozen = self._fn(("linear_frozen", task, n), frozen_body, [spec])
        f_train = self._fn(("linear", task, n), body, [spec, aspec])
        with phases("%s:frozen" % task):
            a_prev = _sync(f_frozen(x))
        with phases("%s:l%d" % (task, lv)):
            return _sync(f_train(x, a_prev))

    def _deform(self, x1, n, phases):
        model, frozen, lv, lean = self.loaded.model, self.loaded.frozen, self.level, self.lean
        x = self._pool(x1, lv, n, phases)
        spec, fspec = self._spec(n, lv), self._spec(n, lv, 3)
        names = ("u", "u_inv", "v", "v_raw")
        if frozen is None:
            def body(x):
                out = model(x, task="deform", training=False)
                return {k: tf.cast(out[k], tf.float32) for k in names}
            fn = self._fn(("deform", n), body, [spec])
            with phases("deform:l%d" % lv):
                return _sync(fn(x))

        def frozen_body(x):
            prev = frozen.deform(x)
            return prev["U"], prev["U_inv"]

        def body(x, U, U_inv):
            x_in = diffeo.warp(tf.cast(x, tf.float32), U, lean=lean)
            out = model(x_in, task="deform", training=False)
            u, u_inv = diffeo.compose_cascade(U, U_inv, out["u"], out["u_inv"], lean=lean)
            return {"u": tf.cast(u, tf.float32), "u_inv": tf.cast(u_inv, tf.float32),
                    "v": tf.cast(out["v"], tf.float32), "v_raw": tf.cast(out["v_raw"], tf.float32)}
        f_frozen = self._fn(("deform_frozen", n), frozen_body, [spec])
        f_train = self._fn(("deform", n), body, [spec, fspec, fspec])
        with phases("deform:frozen"):
            U, U_inv = _sync(f_frozen(x))
        with phases("deform:l%d" % lv):
            return _sync(f_train(x, U, U_inv))

    def run(self, images, branches, sink=None):
        """As `StreamedRunner.run`, one chunk [0, N) (``check_unchunked`` first)."""
        if not self._is_default():
            return self._run_sel(images, branches, sink)
        x1, lin, dfm = self._check(images, branches)
        n = x1.shape[0]
        LD.check_unchunked(n, self.level, self.loaded.model_cfg, branches,
                           self.loaded.knobs["split_group_conv"])
        summary = self._new_summary(n, branches, {"l%d" % lv: 0 for lv in self.levels})
        phases = _Phases()
        x1_t = tf.constant(x1)
        A, fields = None, None
        if lin is not None:
            A = self._linear(x1_t, lin, n, phases)
            for lv in self.levels:
                summary["chunks"]["l%d" % lv][lin] = [(0, n)]
        if dfm is not None:
            src = x1_t
            if A is not None:
                fn = self._fn(("resample", n), lambda x, a: CS.prewarp_linear(
                    x, a, P.LEVEL_SPACING[1]), [self._spec(n, 1), tf.TensorSpec((n, 4, 4), tf.float32)])
                with phases("affine+deform:resample"):
                    src = _sync(fn(x1_t, A))
            out = self._deform(src, n, phases)
            del src
            to_l1 = self._fn(("to_l1", n), lambda u, u_inv: (
                _up_to_level1(u, self.level), _up_to_level1(u_inv, self.level)))
            U1, U1_inv = (out["u"], out["u_inv"]) if self.level == 1 else to_l1(out["u"], out["u_inv"])
            fields = {"U1": U1, "U1_inv": U1_inv, "u_ckpt": out["u"], "u_inv_ckpt": out["u_inv"],
                      "v": out["v"], "v_raw": out["v_raw"]}
            summary["stats"]["max_abs_v"]["l%d" % self.level] = float(
                tf.reduce_max(tf.abs(out["v"])))
            for lv in self.levels:
                summary["chunks"]["l%d" % lv][dfm] = [(0, n)]
        if sink is not None:
            sink(0, n, self._res(A, fields))
        summary["A"] = None if A is None else A.numpy()
        summary["timing_s"] = phases.timing
        summary["gpu_peak_mb"] = phases.gpu_peak_mb if phases.gpu else None
        return summary

    # ---------------------------------------------------------------- level selection (predict-levels)
    # The trainer's own functions with tf.constant(True) as the skip tensor (design 4.2 table).
    def _frozen_skips(self):
        skipped = self.plan.skipped
        return 4 in skipped, 2 in skipped

    def _linear_sel(self, x1, task, n, phases):
        """A of a non-default plan (trainable stage only when ``plan.last`` is the checkpoint level)."""
        model, frozen, lv, plan = self.loaded.model, self.loaded.frozen, self.level, self.plan
        if frozen is None:                              # level 4: the full plan only
            return self._linear(x1, task, n, phases)
        x = self._pool(x1, lv, n, phases)
        spec, aspec = self._spec(n, lv), tf.TensorSpec((n, 4, 4), tf.float32)
        chain = lv == 1
        skip4, skip2 = self._frozen_skips()
        last = plan.last
        sel = plan.selected

        def frozen_body(x):
            s = tf.constant(True)
            if chain:
                if skip4 or skip2:
                    return frozen.linear_skip(x, task, skip4=s if skip4 else None,
                                              skip2=s if skip2 else None)
                if last == 4:
                    return frozen.linear_stages(x, task)["A4"]
                if last == 2:
                    return frozen.linear(x, task)
                return frozen.linear_stages(x, task)["A"]
            a = frozen.linear(x, task)
            if skip4:
                a = CS.skip_linear(s, a)
            return a
        f_frozen = self._fn(("linear_frozen_sel", task, n, sel), frozen_body, [spec])
        with phases("%s:frozen" % task):
            a_prev = _sync(f_frozen(x))
        if last != lv:
            return a_prev
        sp = P.LEVEL_SPACING[lv]

        def body(x, a_prev):
            x_in = CS.prewarp_linear(x, a_prev, sp)
            out = model(x_in, task=task, training=False)
            return CS.compose_linear(out["A_raw"], a_prev, model.centering)
        f_train = self._fn(("linear_sel", task, n), body, [spec, aspec])
        with phases("%s:l%d" % (task, lv)):
            return _sync(f_train(x, a_prev))

    def _deform_sel(self, x1, n, phases):
        """Fields of a non-default plan: {'U1', 'U1_inv'} plus 'u_ckpt', 'u_inv_ckpt', 'v',
        'v_raw' when the trainable stage runs (else None)."""
        model, frozen, lv, lean, plan = (self.loaded.model, self.loaded.frozen, self.level,
                                         self.lean, self.plan)
        to_l1 = self._fn(("to_l1_sel", n), lambda u, u_inv: (
            _up_to_level1(u, self.level), _up_to_level1(u_inv, self.level)))

        def level1(u, u_inv):
            return (u, u_inv) if lv == 1 else to_l1(u, u_inv)
        if frozen is None:                              # level 4: the full plan only
            out = self._deform(x1, n, phases)
        else:
            x = self._pool(x1, lv, n, phases)
            spec, fspec = self._spec(n, lv), self._spec(n, lv, 3)
            chain = lv == 1
            skip4, skip2 = self._frozen_skips()
            last = plan.last

            def frozen_body(x):
                s = tf.constant(True)
                if chain:
                    if skip4 or skip2:
                        prev = frozen.deform_skip(x, skip4=s if skip4 else None,
                                                  skip2=s if skip2 else None)
                    elif last == 4:
                        st = frozen.deform_stages(x)
                        return st["U4"], st["U4_inv"]
                    else:
                        prev = frozen.deform(x)
                    return prev["U"], prev["U_inv"]
                prev = frozen.deform(x)
                U, U_inv = prev["U"], prev["U_inv"]
                if skip4:
                    U, U_inv = CS.skip_fields(s, [U, U_inv])
                return U, U_inv
            f_frozen = self._fn(("deform_frozen_sel", n, plan.selected), frozen_body, [spec])
            with phases("deform:frozen"):
                U, U_inv = _sync(f_frozen(x))
            if last != lv:
                U1, U1_inv = level1(U, U_inv)
                return {"U1": U1, "U1_inv": U1_inv, "u_ckpt": None, "u_inv_ckpt": None,
                        "v": None, "v_raw": None}

            def body(x, U, U_inv):
                x_in = diffeo.warp(tf.cast(x, tf.float32), U, lean=lean)
                out = model(x_in, task="deform", training=False)
                u, u_inv = diffeo.compose_cascade(U, U_inv, out["u"], out["u_inv"], lean=lean)
                return {"u": tf.cast(u, tf.float32), "u_inv": tf.cast(u_inv, tf.float32),
                        "v": tf.cast(out["v"], tf.float32), "v_raw": tf.cast(out["v_raw"], tf.float32)}
            f_train = self._fn(("deform_sel", n), body, [spec, fspec, fspec])
            with phases("deform:l%d" % lv):
                out = _sync(f_train(x, U, U_inv))
        U1, U1_inv = level1(out["u"], out["u_inv"])
        return {"U1": U1, "U1_inv": U1_inv, "u_ckpt": out["u"], "u_inv_ckpt": out["u_inv"],
                "v": out["v"], "v_raw": out["v_raw"]}

    def _run_sel(self, images, branches, sink):
        x1, lin, dfm = self._check(images, branches)
        n = x1.shape[0]
        LD.check_unchunked(n, self.level, self.loaded.model_cfg, branches,
                           self.loaded.knobs["split_group_conv"])
        plan = self.plan
        summary = self._new_summary_sel(n, branches, {"l%d" % lv: 0 for lv in plan.run})
        phases = _Phases()
        x1_t = tf.constant(x1)
        A, fields = None, None
        if lin is not None:
            A = self._linear_sel(x1_t, lin, n, phases)
            for lv in plan.run:
                summary["chunks"]["l%d" % lv][lin] = [(0, n)]
        if dfm is not None:
            src = x1_t
            if A is not None:
                fn = self._fn(("resample_sel", n), lambda x, a: CS.prewarp_linear(
                    x, a, P.LEVEL_SPACING[1]), [self._spec(n, 1), tf.TensorSpec((n, 4, 4), tf.float32)])
                with phases("affine+deform:resample"):
                    src = _sync(fn(x1_t, A))
            fields = self._deform_sel(src, n, phases)
            del src
            if plan.last == self.level:
                summary["stats"]["max_abs_v"]["l%d" % self.level] = float(
                    tf.reduce_max(tf.abs(fields["v"])))
            for lv in plan.run:
                summary["chunks"]["l%d" % lv][dfm] = [(0, n)]
        if sink is not None:
            sink(0, n, self._res(A, fields))
        summary["A"] = None if A is None else A.numpy()
        summary["timing_s"] = phases.timing
        summary["gpu_peak_mb"] = phases.gpu_peak_mb if phases.gpu else None
        return summary
