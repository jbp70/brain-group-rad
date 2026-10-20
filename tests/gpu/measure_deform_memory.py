#!/usr/bin/env python
"""GPU memory and step time of one level-4 deformable training step (milestone 6).

Not part of unittest discovery. Runs locally (GTX 1080, WSL tf_gpu) and on the cluster (inside
tf10_nv.sif) on ONE GPU:

    python tests/gpu/measure_deform_memory.py --n 4,8,16 --widths narrow,default \
        --gradient-checkpointing none --gpu-memory-limit-mb 6000 --out memory.json

For every configuration (N x widths x gradient checkpointing) it builds the level-4 model with
the deform decoder (`ModelConfig(deform_decoder=True)`, default = full widths, narrow =
`tests.runtime_helpers.NARROW`), a `Trainer` with ratio deform:1 (Adam, default deform losses:
template soft Dice + 1.0 x pairwise soft Dice at p_pdice 0.1 + 0.5 x first-order smoothness)
on the synthetic deform group (`fd.synthetic_deform_group`, labels 0..8, 19 one-hot channels),
runs one warm-up step (trace, cuDNN setup) and then --steps measured steps. TF's allocator
statistics (`tf.config.experimental.get_memory_info('GPU:0')`, reset before the measured
steps) give the peak; DALI is not involved. An OOM (ResourceExhaustedError) is caught and
recorded per configuration; larger N of the same widths / checkpointing are then skipped.

JSON (``--out``): {"environment": {...}, "results": [{n, widths, gradient_checkpointing,
params, status ('ok' | 'oom' | 'error' | 'skipped'), peak_mb (measured steps), warmup_peak_mb,
resident_mb (allocated before the measured steps: weights, optimizer slots, batch), step_ms
(median), step_ms_all, loss, error}]}. With an existing --out file the new results are merged
(same configuration replaced), so several invocations build one table.
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from models.precision import configure_tf_precision  # noqa: E402

LEVEL = 4
MB = float(2 ** 20)


def configure_gpu(limit_mb):
    """TF32 off FIRST (A100); then either a fixed logical-device limit or memory growth."""
    if limit_mb and limit_mb > 0:
        configure_tf_precision(tf32=False, memory_growth=False)
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.set_logical_device_configuration(
                gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=int(limit_mb))])
    else:
        configure_tf_precision(tf32=False, memory_growth=True)


def model_config(widths):
    from models.config import ModelConfig
    from tests.runtime_helpers import NARROW
    if widths == "narrow":
        return ModelConfig(deform_decoder=True, dropout=0.0, **NARROW)
    if widths == "default":
        return ModelConfig(deform_decoder=True, dropout=0.0)
    raise ValueError("widths must be 'narrow' or 'default', got %r" % (widths,))


def measure(n, widths, ckpt, steps, seed=0):
    from data_loading import params as P
    from models import transforms as tfm
    from models.group_net import GroupRegLevelModel
    from runtime import fixed_data as fd
    from runtime import optimizers as O
    from runtime.run import Trainer, TrainConfig
    from runtime.schedulers import ScheduleConfig

    res = {"n": int(n), "widths": widths, "gradient_checkpointing": ckpt}
    group = fd.synthetic_deform_group(LEVEL, int(n), seed=seed)
    mask = np.asarray(group["base_labels"]).reshape(P.LEVEL_SHAPE[LEVEL]) > 0
    S = tfm.brain_second_moment(mask, P.LEVEL_SPACING[LEVEL])
    cfg = TrainConfig(level=LEVEL, group_size=int(n), steps=steps + 1, val_every=0, log_every=1,
                      task_ratio={"deform": 1.0}, seed=seed,
                      schedule=ScheduleConfig("constant", base_lr=3e-4, total_steps=steps + 1,
                                              warmup_steps=0))
    model = GroupRegLevelModel(LEVEL, model_config(widths), gradient_checkpointing=ckpt).build_model(int(n))
    res["params"] = int(sum(int(np.prod(v.shape)) for v in model.trainable_variables))
    trainer = Trainer(model, O.make_optimizer("adam", lr=3e-4), {"deform": fd.FixedGroupSource(group["batch"])},
                      S, cfg)
    batch = tuple(tf.constant(x) for x in group["batch"])
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
        out = trainer.train_step("deform", batch)                 # warm-up: trace + cuDNN setup
        res["warmup_peak_mb"] = tf.config.experimental.get_memory_info("GPU:0")["peak"] / MB
        res["resident_mb"] = tf.config.experimental.get_memory_info("GPU:0")["current"] / MB
        tf.config.experimental.reset_memory_stats("GPU:0")
        times = []
        for _ in range(steps):
            t0 = time.time()
            out = trainer.train_step("deform", batch)              # returns Python scalars (synced)
            times.append((time.time() - t0) * 1e3)
        res["peak_mb"] = tf.config.experimental.get_memory_info("GPU:0")["peak"] / MB
        res["step_ms"] = float(np.median(times))
        res["step_ms_all"] = [float(t) for t in times]
        res["loss"] = float(out["loss"])
        res["finite"] = bool(out["finite"])
        res["trace_counts"] = dict(trainer.trace_counts)
        res["status"] = "ok"
    finally:
        del trainer, model, batch, group
    return res


def free_memory():
    tf.keras.backend.clear_session()
    gc.collect()


def config_key(r):
    return (r["n"], r["widths"], r["gradient_checkpointing"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", default="4,8,16", help="comma-separated group sizes")
    ap.add_argument("--widths", default="narrow,default", help="comma-separated: narrow, default")
    ap.add_argument("--gradient-checkpointing", default="none",
                    help="comma-separated: none, integrator, decoder, both")
    ap.add_argument("--steps", type=int, default=3, help="measured steps after the warm-up step")
    ap.add_argument("--gpu-memory-limit-mb", type=int, default=0,
                    help="fixed TF logical-device limit (0: memory growth, e.g. on the cluster)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="JSON output (merged with an existing file)")
    a = ap.parse_args(argv)
    configure_gpu(a.gpu_memory_limit_mb)
    from models.group_net import GRADIENT_CHECKPOINTING

    ns = sorted(int(x) for x in a.n.split(",") if x.strip())
    widths = [w.strip() for w in a.widths.split(",") if w.strip()]
    ckpts = [c.strip() for c in a.gradient_checkpointing.split(",") if c.strip()]
    for c in ckpts:
        if c not in GRADIENT_CHECKPOINTING:
            ap.error("--gradient-checkpointing %r not in %s" % (c, GRADIENT_CHECKPOINTING))
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("no GPU visible", file=sys.stderr)
        return 2
    env = {"python": sys.version.split()[0], "tensorflow": tf.__version__, "numpy": np.__version__,
           "gpu": tf.config.experimental.get_device_details(gpus[0]).get("device_name"),
           "gpu_memory_limit_mb": a.gpu_memory_limit_mb, "level": LEVEL, "measured_steps": a.steps,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "tf_cudnn_use_autotune": os.environ.get("TF_CUDNN_USE_AUTOTUNE")}
    print("environment: %s" % env, flush=True)

    results = []
    for w in widths:
        for c in ckpts:
            oom_at = None
            for n in ns:
                if oom_at is not None:
                    results.append({"n": n, "widths": w, "gradient_checkpointing": c, "status": "skipped",
                                    "error": "OOM at N = %d" % oom_at})
                    continue
                t0 = time.time()
                try:
                    r = measure(n, w, c, a.steps, a.seed)
                except tf.errors.ResourceExhaustedError as e:
                    r = {"n": n, "widths": w, "gradient_checkpointing": c, "status": "oom",
                         "error": str(e).strip().splitlines()[0][:300]}
                    oom_at = n
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    r = {"n": n, "widths": w, "gradient_checkpointing": c, "status": "error",
                         "error": repr(e)[:300]}
                r["wall_s"] = time.time() - t0
                free_memory()
                results.append(r)
                if r["status"] == "ok":
                    print("N=%-3d %-7s ckpt=%-10s peak %8.1f MB (warm-up %8.1f, resident %7.1f)  step %7.1f ms  "
                          "loss %.4f" % (n, w, c, r["peak_mb"], r["warmup_peak_mb"], r["resident_mb"],
                                         r["step_ms"], r["loss"]), flush=True)
                else:
                    print("N=%-3d %-7s ckpt=%-10s %s: %s" % (n, w, c, r["status"].upper(), r.get("error")),
                          flush=True)

    merged = {}
    if os.path.exists(a.out):
        with open(a.out) as f:
            old = json.load(f)
        for r in old.get("results", []):
            merged[config_key(r)] = r
    for r in results:
        merged[config_key(r)] = r
    out = {"environment": env,
           "results": sorted(merged.values(), key=lambda r: (r["widths"], r["gradient_checkpointing"], r["n"]))}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("written %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
