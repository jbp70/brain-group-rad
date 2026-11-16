#!/usr/bin/env python
"""Local GPU-memory comparison of the conv norms under mixed_float16 (ablation-oom-fixes, failure 1).

Not a unittest. For every (norm variant, task, N) it builds the d12hs config-B model of one level
(three-task model, deform decoder on) under mixed_float16 and measures the TF allocator peak of
ONE traced forward + backward (tape.gradient of a scalar of the outputs; no optimizer, no data
pipeline, no losses), after one unmeasured warm-up call (trace, cuDNN setup), then the median
time of 3 more calls (step_s_median). Deform runs with
--gradient-checkpointing full, lean sampler and split group conv (the r_level*.sh recipe); linear
tasks are not checkpointed (as in the trainer). Variants:
  batch     BatchNormWithSeparateAffine (Keras fused BN, the production model)
  instance  InstanceNormWithAffine as in this branch (float16 input -> fused kernel)
  in_old    the float32-moment implementation of main 4fe21f7 (monkeypatched call)
Each configuration runs in its own child process (fresh allocator), so peaks are comparable.

    python .agents/tasks/ablation-oom-fixes/measure_in_memory.py --level 4 --tasks rigid --n 2,4,8 \
        --norms batch,instance,in_old --out in_memory_l4.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FILTERS = (16, 64, 128, 192, 256, 384)
DECODER_WIDTHS = (128, 96, 64, 16, 8)


def child(level, task, n, norm, ddtype="policy"):
    sys.path.insert(0, REPO)
    import tests.tf_setup  # noqa: F401  (TF32 off, GPU limit)
    import numpy as np
    import tensorflow as tf
    from models import layers as L
    from data_loading import params as P
    from models.config import ModelConfig
    from models.group_net import GroupRegLevelModel

    if norm == "in_old":
        def old_call(self, x, training=None):
            rank = x.shape.rank
            xf = tf.cast(x, tf.float32)
            mean, var = tf.nn.moments(xf, axes=list(range(1, rank - 1)), keepdims=True)
            y = (xf - mean) * tf.math.rsqrt(var + self.epsilon)
            return self.affine(tf.cast(y, x.dtype))
        L.InstanceNormWithAffine.call = old_call
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    cfg = ModelConfig(filters=FILTERS, decoder_widths=DECODER_WIDTHS, group_config="B",
                      norm="batch" if norm == "batch" else "instance", deform_decoder=True)
    deform = task == "deform"
    model = GroupRegLevelModel(level, cfg, gradient_checkpointing="full" if deform else "none",
                               lean_sampler=deform, split_group_conv=True,
                               deform_decoder_dtype=ddtype)
    model.build_model(2)
    shape = (n,) + tuple(P.LEVEL_SHAPE[level]) + (1,)
    x = tf.constant(np.random.default_rng(0).normal(0.5, 0.2, shape).astype(np.float32))

    @tf.function
    def step(x):
        with tf.GradientTape() as tape:
            out = model(x, task=task, training=True)
            if deform:
                obj = tf.reduce_mean(tf.square(out["u"])) + tf.reduce_mean(tf.square(out["u_inv"]))
            else:
                obj = tf.reduce_mean(tf.square(out["A"])) + tf.reduce_mean(tf.square(out["params"]))
        gs = tape.gradient(obj, model.trainable_variables)
        return obj, tf.linalg.global_norm([g for g in gs if g is not None])

    rec = {"level": level, "task": task, "n": n, "norm": norm, "deform_decoder_dtype": ddtype}
    try:
        o, g = step(x)
        o.numpy()
        tf.config.experimental.reset_memory_stats("GPU:0")
        base = tf.config.experimental.get_memory_info("GPU:0")["current"]
        o, g = step(x)
        rec.update(status="ok", obj=float(o.numpy()), grad_norm=float(g.numpy()),
                   resident_mb=base / 2.0 ** 20,
                   peak_mb=tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2.0 ** 20)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            step(x)[1].numpy()
            times.append(time.perf_counter() - t0)
        rec["step_s_median"] = sorted(times)[1]
    except tf.errors.ResourceExhaustedError:
        rec["status"] = "oom"
    print("RESULT " + json.dumps(rec), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--tasks", default="rigid,deform")
    ap.add_argument("--n", default="2,4")
    ap.add_argument("--norms", default="batch,instance,in_old")
    ap.add_argument("--deform-decoder-dtype", default="policy",
                    help="GroupRegLevelModel deform_decoder_dtype (L1 recipe: float32_fine)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--child", default=None)
    a = ap.parse_args()
    if a.child:
        task, n, norm = a.child.split(":")
        child(a.level, task, int(n), norm, a.deform_decoder_dtype)
        return 0
    recs = []
    for task in a.tasks.split(","):
        for norm in a.norms.split(","):
            for n in [int(v) for v in a.n.split(",")]:
                p = subprocess.run([sys.executable, os.path.abspath(__file__), "--level", str(a.level),
                                    "--deform-decoder-dtype", a.deform_decoder_dtype,
                                    "--out", a.out, "--child", "%s:%d:%s" % (task, n, norm)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
                rec = (json.loads(lines[-1][7:]) if lines else
                       {"level": a.level, "task": task, "n": n, "norm": norm, "status": "error",
                        "rc": p.returncode, "stderr_tail": p.stderr[-800:]})
                print(json.dumps(rec), flush=True)
                recs.append(rec)
    with open(a.out, "w") as f:
        json.dump(recs, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
