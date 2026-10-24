#!/usr/bin/env python
"""Per-tensor element-limit probe (milestone 9 phase 1, R1.3).

Design: `.agents/tasks/milestone-9/design-m9-phase1.md` sections 2.4 and 5.3. Not part of
unittest discovery (tests/test_probe_tensor_limit.py loads it with importlib). Three modes:

  graph     Builds the level-`--level` model (default: level 2, N 12, default widths) on a
            fixed-shape input WITHOUT executing it, exactly like `GroupRegLevelModel.build_model`
            does it (one `--task` linear forward, then one deform forward), in a TF1-style
            `tf.Graph` with a placeholder. Fallback if Keras refuses: the same two calls traced
            once by `tf.function(...).get_concrete_function(TensorSpec)` (the variables are
            created during that single trace). Lists every op output with more than 2^31 - 1
            elements, sorted by size (op type, scope, shape, dtype, elements, bytes), then prints
            ``hypothesis limit: 2147483647 elements (int32 index); run mode decides``. CPU-safe:
            nothing is executed.
  run       (GPU) single ops at E elements per dtype (``--sizes``, default fp32 / fp16 at
            2^31 -+ 2^24 plus fp16 3.52e9 and 4.4e9). Forward: conv3d (NDHWC, 8 -> 8 channels),
            transpose NDHWC -> NCDHW, `models.layers.group_mean_concat`, Keras BatchNormalization
            (training call), LeakyReLU, reduce_mean(axis 0), gather (rows of [E/8, 8]),
            unsorted_segment_sum. Backward (GradientTape, output gradients = ones): conv3d (input
            AND kernel: Conv3DBackpropInputV2 / FilterV2), BN (FusedBatchNormGradV3), LeakyReLU
            (LeakyReluGrad). Before each op a byte estimate ((inputs + outputs) x 1.5 forward,
            x 3 backward) is compared with the device limit (``--gpu-memory-limit-mb``, else
            nvidia-smi memory.total); above it the op is recorded as ``skipped_mem``. Status per
            op and direction: ok | oom | error | skipped_mem (with the first error line).
            Summary ``verified_max_elements = {fp32: {fwd, bwd}, fp16: {fwd, bwd}}``: the largest E
            at which EVERY op of that direction is ok (null if none); `models.memory_model`
            (`--tensor-limits`) reads it. Exit 2 without a GPU.
  m8-errors Prints every non-ok record of a milestone-8 memory JSON (`--m8-json`, e.g.
            scratch/m8/memory_level2.json: configuration key and error text) and the peak_mb /
            warmup_peak_mb columns of all records (design U11). Imports no TensorFlow.

graph and run call `models.precision.configure_tf_precision(tf32=False, memory_growth=not
--gpu-memory-limit-mb)` before any other TF call (TF32 off). Exit code 0 unless it crashes
(run without a GPU: 2; m8-errors with a missing file: 1).

    python tests/gpu/probe_tensor_limit.py --mode graph --level 2 --n 12 --widths default --task rigid
    python tests/gpu/probe_tensor_limit.py --mode run --out scratch/m9/tensor_limit.json
    python tests/gpu/probe_tensor_limit.py --mode m8-errors --m8-json scratch/m8/memory_level2.json
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

INT32_MAX = 2 ** 31 - 1
HYPOTHESIS = "hypothesis limit: 2147483647 elements (int32 index); run mode decides"
DEFAULT_SIZES = ("fp32:%d" % (2 ** 31 - 2 ** 24), "fp32:%d" % (2 ** 31 + 2 ** 24),
                 "fp16:%d" % (2 ** 31 - 2 ** 24), "fp16:%d" % (2 ** 31 + 2 ** 24),
                 "fp16:%d" % (16 * 80 * 112 * 96 * 256), "fp16:%d" % (20 * 80 * 112 * 96 * 256))
DTYPE_BYTES = {"fp32": 4, "fp16": 2}
CHANNELS = 8
FWD_OPS = ("conv3d", "transpose_ncdhw", "group_mean_concat", "batch_norm", "leaky_relu",
           "reduce_mean_axis0", "gather", "unsorted_segment_sum")
BWD_OPS = ("conv3d", "batch_norm", "leaky_relu")
M8_KEY = ("level", "task", "n", "widths", "cascade", "gradient_checkpointing", "amp", "integration_steps")


# ----------------------------------------------------------------------------- common
def _first_line(e, limit=300):
    s = str(e).strip()
    return (s.splitlines()[0] if s else repr(e))[:limit]


def _configure(limit_mb):
    """TF32 off FIRST; then a fixed logical-device limit or memory growth. Returns tf."""
    import tensorflow as tf
    from models.precision import configure_tf_precision
    configure_tf_precision(tf32=False, memory_growth=not (limit_mb and limit_mb > 0))
    if limit_mb and limit_mb > 0:
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.set_logical_device_configuration(
                    gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=int(limit_mb))])
            except RuntimeError:
                pass  # already initialised (e.g. by tests.tf_setup)
    return tf


def model_config(widths):
    """Three-task ModelConfig (deform decoder on, dropout 0) of narrow / tiny / a width preset."""
    from models.config import ModelConfig
    from models import width_presets as WP
    kw = {"deform_decoder": True, "dropout": 0.0}
    if widths == "narrow":
        from tests.runtime_helpers import NARROW
        kw.update(NARROW)
    elif widths == "tiny":
        from tests.runtime_helpers import TINY
        kw.update(TINY)
        kw["dropout"] = 0.0
    elif widths != "default":
        kw.update(WP.preset_kwargs(widths))
    return ModelConfig(**kw)


def _write_json(path, obj):
    if not path:
        return
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print("written %s" % path, flush=True)


# ----------------------------------------------------------------------------- graph
def _list_large(ops, limit):
    rows = []
    for idx, op in enumerate(ops):
        for t in op.outputs:
            shape = t.shape
            if shape.rank is None or not shape.is_fully_defined():
                continue
            dims = [int(d) for d in shape.as_list()]
            elems = 1
            for d in dims:
                elems *= d
            if elems > limit:
                size = t.dtype.size if t.dtype.size else 0
                rows.append({"op_type": op.type, "scope": op.name, "shape": dims, "dtype": t.dtype.name,
                             "elements": int(elems), "bytes": int(elems * size), "order": idx})
    rows.sort(key=lambda r: (-r["elements"], r["order"]))
    return rows


def graph_mode(a):
    tf = _configure(a.gpu_memory_limit_mb)
    from data_loading import params as P
    from models.group_net import GroupRegLevelModel
    shape = (int(a.n),) + tuple(P.LEVEL_SHAPE[a.level]) + (1,)
    cfg = model_config(a.widths)
    mp = tf.keras.mixed_precision
    if a.amp:
        mp.set_global_policy("mixed_float16")
    method = "tf.Graph + placeholder"
    try:
        try:
            g = tf.Graph()
            with g.as_default():
                x = tf.compat.v1.placeholder(tf.float32, shape, name="images")
                model = GroupRegLevelModel(a.level, cfg)
                model(x, task=a.task, training=False)          # = build_model: linear forward,
                if model.deform_decoder is not None:          # then one deform forward
                    model(x, task="deform", training=False)
            ops = g.get_operations()
        except Exception as e:  # noqa: BLE001  (Keras refused the TF1 graph)
            method = "tf.function concrete function (tf.Graph failed: %s)" % _first_line(e, 120)
            model = GroupRegLevelModel(a.level, cfg)

            def fwd(x):
                out = model(x, task=a.task, training=False)
                res = [out["A"]]
                if model.deform_decoder is not None:
                    res.append(model(x, task="deform", training=False)["u"])
                return res

            cf = tf.function(fwd).get_concrete_function(tf.TensorSpec(shape, tf.float32))
            ops = cf.graph.get_operations()
    finally:
        mp.set_global_policy("float32")
    rows = _list_large(ops, INT32_MAX)
    print("graph mode: level %d, N %d, widths %s, task %s (+ deform forward), amp %s; %s; %d ops"
          % (a.level, a.n, a.widths, a.task, a.amp, method, len(ops)), flush=True)
    print("op outputs with more than %d elements: %d" % (INT32_MAX, len(rows)))
    for i, r in enumerate(rows):
        print("  %2d. %-22s %-70s %-28s %-8s %12d elements %8.2f GB"
              % (i + 1, r["op_type"], r["scope"], str(r["shape"]), r["dtype"], r["elements"], r["bytes"] / 1e9))
    print(HYPOTHESIS, flush=True)
    _write_json(a.out, {"mode": "graph", "level": a.level, "n": a.n, "widths": a.widths, "task": a.task,
                        "amp": bool(a.amp), "method": method, "n_ops": len(ops), "limit": INT32_MAX,
                        "large_outputs": rows, "hypothesis": HYPOTHESIS})
    return 0


# ----------------------------------------------------------------------------- run
def parse_sizes(text):
    """'fp32:E,fp16:E,...' -> [(dtype, E)] with E rounded down to a multiple of 8 (channels)."""
    out = []
    for item in [x.strip() for x in str(text).split(",") if x.strip()]:
        if ":" not in item:
            raise ValueError("--sizes entries are dtype:elements, got %r" % item)
        dt, e = item.split(":", 1)
        dt = dt.strip().lower()
        if dt not in DTYPE_BYTES:
            raise ValueError("--sizes dtype must be fp32 or fp16, got %r" % dt)
        e = int(float(e))
        e -= e % CHANNELS
        if e < CHANNELS:
            raise ValueError("--sizes elements must be >= %d, got %r" % (CHANNELS, item))
        out.append((dt, e))
    return out


def shape5(elements, channels=CHANNELS):
    """[1, d0, d1, d2, channels] with d0 d1 d2 channels == elements (d1, d2 powers of two <= 128
    dividing the voxel count, d0 the rest)."""
    v = int(elements) // channels
    dims = []
    for _ in range(2):
        d = 128
        while d > 1 and v % d:
            d //= 2
        dims.append(d)
        v //= d
    return [1, v, dims[1], dims[0], channels]


def _device_limit_mb(limit_mb):
    if limit_mb and limit_mb > 0:
        return float(limit_mb)
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                           timeout=20, stdin=subprocess.DEVNULL)
        vals = [float(x) for x in r.stdout.split() if x.strip()]
        return vals[0] if vals else None
    except Exception:  # noqa: BLE001
        return None


def _estimate_bytes(op, direction, e, s):
    """(inputs + outputs) bytes x 1.5 (forward) / x 3 (backward)."""
    rows = e // CHANNELS
    io = {"conv3d": 2 * e * s, "transpose_ncdhw": 2 * e * s, "group_mean_concat": 3 * e * s,
          "batch_norm": 2 * e * s, "leaky_relu": 2 * e * s, "reduce_mean_axis0": e * s + rows * s,
          "gather": 2 * e * s + 4 * rows, "unsorted_segment_sum": 2 * e * s + 4 * rows}[op]
    return io * (1.5 if direction == "fwd" else 3.0)


def _run_op(tf, op, direction, dt, e):
    from models import layers as L
    dtype = tf.float32 if dt == "fp32" else tf.float16
    shp = shape5(e)
    rows = e // CHANNELS

    def sync(*ts):
        for t in ts:
            tf.reshape(t, [-1])[-1:].numpy()

    with tf.device("/GPU:0"):
        x = tf.ones(shp, dtype)
        if op == "conv3d":
            k = tf.ones([3, 3, 3, CHANNELS, CHANNELS], dtype) * tf.constant(0.01, dtype)
            if direction == "fwd":
                sync(tf.nn.conv3d(x, k, [1, 1, 1, 1, 1], "SAME"))
            else:
                with tf.GradientTape() as tape:
                    tape.watch([x, k])
                    y = tf.nn.conv3d(x, k, [1, 1, 1, 1, 1], "SAME")
                gx, gk = tape.gradient(y, [x, k], output_gradients=tf.ones_like(y))
                sync(gx, gk)
        elif op == "transpose_ncdhw":
            sync(tf.transpose(x, [0, 4, 1, 2, 3]))
        elif op == "group_mean_concat":
            sync(L.group_mean_concat(x))
        elif op == "batch_norm":
            bn = tf.keras.layers.BatchNormalization(dtype="float32" if dt == "fp32" else "mixed_float16")
            if direction == "fwd":
                sync(bn(x, training=True))
            else:
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    y = bn(x, training=True)
                sync(tape.gradient(y, x, output_gradients=tf.ones_like(y)))
        elif op == "leaky_relu":
            if direction == "fwd":
                sync(tf.nn.leaky_relu(x, 0.01))
            else:
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    y = tf.nn.leaky_relu(x, 0.01)
                sync(tape.gradient(y, x, output_gradients=tf.ones_like(y)))
        elif op == "reduce_mean_axis0":
            sync(tf.reduce_mean(tf.reshape(x, [CHANNELS, rows]), axis=0))
        elif op == "gather":
            idx = tf.range(rows - 1, -1, -1, dtype=tf.int32)
            sync(tf.gather(tf.reshape(x, [rows, CHANNELS]), idx))
        elif op == "unsorted_segment_sum":
            ids = tf.range(rows - 1, -1, -1, dtype=tf.int32)
            sync(tf.math.unsorted_segment_sum(tf.reshape(x, [rows, CHANNELS]), ids, rows))
        else:
            raise ValueError("unknown op %r" % op)


def verified_max(records):
    """{dtype: {fwd, bwd}}: the largest E at which every op of that direction is ok (None if none)."""
    out = {}
    for dt in ("fp32", "fp16"):
        out[dt] = {}
        for direction in ("fwd", "bwd"):
            sizes = sorted({r["elements"] for r in records if r["dtype"] == dt and r["direction"] == direction})
            best = None
            for e in sizes:
                recs = [r for r in records if r["dtype"] == dt and r["direction"] == direction and r["elements"] == e]
                if recs and all(r["status"] == "ok" for r in recs):
                    best = e
            out[dt][direction] = best
    return out


def run_mode(a):
    tf = _configure(a.gpu_memory_limit_mb)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("no GPU visible", file=sys.stderr)
        return 2
    try:
        sizes = parse_sizes(a.sizes)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    limit_mb = _device_limit_mb(a.gpu_memory_limit_mb)
    env = {"python": sys.version.split()[0], "tensorflow": tf.__version__,
           "gpu": tf.config.experimental.get_device_details(gpus[0]).get("device_name"),
           "device_limit_mb": limit_mb, "gpu_memory_limit_mb": a.gpu_memory_limit_mb,
           "tf_cudnn_use_autotune": os.environ.get("TF_CUDNN_USE_AUTOTUNE")}
    print("environment: %s" % env, flush=True)
    records = []
    for dt, e in sizes:
        for direction, ops in (("fwd", FWD_OPS), ("bwd", BWD_OPS)):
            for op in ops:
                est = _estimate_bytes(op, direction, e, DTYPE_BYTES[dt])
                rec = {"op": op, "dtype": dt, "elements": int(e), "shape": shape5(e), "direction": direction,
                       "est_bytes": int(est), "status": None, "error": None, "ms": None}
                if limit_mb is not None and est > limit_mb * 2.0 ** 20:
                    rec["status"] = "skipped_mem"
                else:
                    t0 = time.time()
                    try:
                        _run_op(tf, op, direction, dt, e)
                        rec["status"] = "ok"
                    except tf.errors.ResourceExhaustedError as ex:
                        rec["status"], rec["error"] = "oom", _first_line(ex)
                    except Exception as ex:  # noqa: BLE001
                        rec["status"], rec["error"] = "error", _first_line(ex)
                    rec["ms"] = (time.time() - t0) * 1e3
                    gc.collect()
                records.append(rec)
                print("%-4s %-3s %-22s E %12d %-28s est %7.2f GB  %s%s"
                      % (dt, direction, op, e, str(rec["shape"]), est / 1e9, rec["status"],
                         "" if not rec["error"] else ": " + rec["error"]), flush=True)
    summary = verified_max(records)
    print("verified_max_elements: %s" % json.dumps(summary), flush=True)
    _write_json(a.out, {"mode": "run", "environment": env, "sizes": [list(s) for s in sizes],
                        "records": records, "verified_max_elements": summary})
    return 0


# ----------------------------------------------------------------------------- m8-errors
def _fmt(v, f="%.1f"):
    return "-" if v is None else (f % v if isinstance(v, float) else str(v))


def m8_errors_mode(a):
    if not a.m8_json or not os.path.exists(a.m8_json):
        print("m8 memory JSON not found: %r" % (a.m8_json,), file=sys.stderr)
        return 1
    with open(a.m8_json) as f:
        res = json.load(f).get("results", [])
    bad = [r for r in res if r.get("status") != "ok"]
    print("m8 memory JSON %s: %d records, %d not ok" % (a.m8_json, len(res), len(bad)))
    for r in bad:
        key = " ".join("%s=%s" % (k, r.get(k)) for k in M8_KEY)
        print("  [%s] %s\n      error: %s" % (r.get("status"), key, r.get("error")))
        if r.get("val_error"):
            print("      val_error: %s" % r.get("val_error"))
    print("peak columns (MB):")
    print("  %-6s %-3s %-8s %-5s %-10s %-5s %-6s %10s %14s" % ("task", "N", "widths", "casc", "ckpt", "amp",
                                                            "status", "peak_mb", "warmup_peak_mb"))
    for r in res:
        print("  %-6s %-3s %-8s %-5s %-10s %-5s %-6s %10s %14s" % (
            r.get("task"), r.get("n"), r.get("widths"), r.get("cascade"), r.get("gradient_checkpointing"),
            r.get("amp"), r.get("status"), _fmt(r.get("peak_mb")), _fmt(r.get("warmup_peak_mb"))))
    if a.out:
        _write_json(a.out, {"mode": "m8-errors", "m8_json": a.m8_json, "not_ok": bad,
                            "peaks": [{k: r.get(k) for k in M8_KEY + ("status", "peak_mb", "warmup_peak_mb")}
                                      for r in res]})
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("graph", "run", "m8-errors"), required=True)
    ap.add_argument("--level", type=int, choices=(4, 2, 1), default=2, help="graph: pyramid level")
    ap.add_argument("--n", type=int, default=12, help="graph: group size of the fixed-shape input")
    ap.add_argument("--widths", default="default", help="graph: narrow, tiny or a models.width_presets name")
    ap.add_argument("--task", choices=("rigid", "affine"), default="rigid",
                    help="graph: linear task of the build forward (a deform forward follows)")
    ap.add_argument("--amp", default="false", help="graph: true = mixed_float16 policy")
    ap.add_argument("--gpu-memory-limit-mb", type=int, default=0,
                    help="fixed TF logical-device limit (0: memory growth); run: also the skipped_mem bound")
    ap.add_argument("--sizes", default=",".join(DEFAULT_SIZES),
                    help="run: comma list of dtype:elements (dtype fp32 | fp16)")
    ap.add_argument("--m8-json", default=None, help="m8-errors: milestone-8 memory JSON")
    ap.add_argument("--out", default=None, help="JSON output (optional for graph / m8-errors)")
    a = ap.parse_args(argv)
    v = str(a.amp).strip().lower()
    if v not in ("true", "false", "1", "0", "yes", "no"):
        ap.error("--amp must be true or false, got %r" % a.amp)
    a.amp = v in ("true", "1", "yes")
    if a.n < 1:
        ap.error("--n must be >= 1")
    if a.mode == "m8-errors":
        return m8_errors_mode(a)                  # no TensorFlow import
    if a.mode == "graph":
        return graph_mode(a)
    return run_mode(a)


if __name__ == "__main__":
    sys.exit(main())
