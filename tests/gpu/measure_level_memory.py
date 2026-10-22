#!/usr/bin/env python
"""GPU memory and step time of one training step per task at a pyramid level (milestone 8).

Not part of unittest discovery. Runs locally (GTX 1080, WSL tf_gpu) and on the cluster (inside
tf10_nv.sif, job slurm/m8_memory.sh) on ONE GPU, e.g.:

    python tests/gpu/measure_level_memory.py --level 2 --tasks deform --n 2,4,8 --widths default \
        --cascade true --gradient-checkpointing none,both --amp false,true --val-pass true \
        --lambda-ncc-template 1 --lambda-ncc-pair 1 --out memory_level2.json

Same structure as the milestone-6 `measure_deform_memory.py` (kept as is for reproducibility).
For every configuration (task x widths x cascade x gradient checkpointing x AMP x N) it builds
the trainable level-`--level` model (deform decoder always on, i.e. the three-task model; widths
default = full, narrow = `tests.runtime_helpers.NARROW`, tiny = `tests.runtime_helpers.TINY`),
with `--cascade true` also a `models.cascade.FrozenStage.from_model` of a random level-4 model
of the same config (float32 always), and a `Trainer` with ratio <task>:1 (Adam 3e-4, global
clip 12) on synthetic data (no DALI): linear `tests.runtime_helpers.synthetic_group(level, n,
task)`, deform `fd.synthetic_deform_group(level, n)` (labels 0..8, 19 one-hot channels). It runs
--warmup-steps steps (trace, cuDNN setup), then --steps measured steps; TF's allocator statistics
(`tf.config.experimental.get_memory_info('GPU:0')`, reset before the measured steps) give the
peak. Gradient checkpointing applies to deform only (linear configs use 'none'). `--amp true`
sets `mixed_float16` before building the trainable model and uses a LossScaleOptimizer (the
frozen stage stays float32); the policy is reset to float32 after each configuration.

An OOM (ResourceExhaustedError) is recorded per configuration (status 'oom'); larger N of the
same configuration are then 'skipped'. Any other exception gives status 'error' (repr; the
traceback goes to stderr) and the next configuration runs.

`--val-pass true` (deform only): after the measured steps (also after a training OOM),
`reset_memory_stats`, then one validation pass on the synthetic group as `_val_deform_errors`
does it: the eval-mode `trainer._val_deform_fn(n)` forward (cascade branch with --cascade true)
followed by `trainer._deform_errors(labels, u, u_inv, v, v_raw, seed=0, chunk=16, images=images)`
(eager warps, pair chunks of 16). Records val_peak_mb, val_status ('ok' | 'oom' | 'error') and
val_ms (wall time of that first pass, incl. the validation function's trace); an OOM there does not mark the training configuration as OOM. null for linear tasks or
without --val-pass.

`--integration-steps` (default 0 = the ModelConfig default, 7) is a local-testing knob (a
level-2 TINY deform step with T = 7 needs more than the 6000 MB GTX 1080 cap); it is part of the
configuration key.

JSON (``--out``): {"environment": {...}, "results": [{level, task, n, widths, cascade,
gradient_checkpointing, amp, lambda_ncc_tmpl, lambda_ncc_pair, integration_steps, params,
frozen_params, status ('ok' | 'oom' | 'error' | 'skipped'), peak_mb (measured steps),
warmup_peak_mb, resident_mb (allocated after the warm-up: weights of both stages, optimizer
slots, batch), step_ms (median), step_ms_mean, step_ms_all, frozen_ms (median of the frozen
forward alone, cascade only, else null), trace_counts, finite, loss, val_peak_mb, val_status,
val_ms, error}]}. With an existing --out file the new results are merged (same configuration key
replaced), so several invocations build one table. main(argv) returns 0 unless it crashes (2
without a GPU).
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

MB = float(2 ** 20)
TASKS = ("rigid", "affine", "deform")
WIDTHS = ("narrow", "default", "tiny")
DEFORM = "deform"


def configure_gpu(limit_mb):
    """TF32 off FIRST (A100); then either a fixed logical-device limit or memory growth. If the
    GPU is already initialised in this process (e.g. by tests.tf_setup), the limit cannot change
    any more; that RuntimeError is ignored."""
    if limit_mb and limit_mb > 0:
        configure_tf_precision(tf32=False, memory_growth=False)
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.set_logical_device_configuration(
                    gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=int(limit_mb))])
            except RuntimeError:
                pass  # already initialised
    else:
        configure_tf_precision(tf32=False, memory_growth=True)


def str2bool(s):
    v = str(s).strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    raise ValueError("not a bool: %r" % (s,))


def csv_list(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def model_config(widths, integration_steps=0):
    from models.config import ModelConfig
    from tests.runtime_helpers import NARROW, TINY
    kw = {"deform_decoder": True, "dropout": 0.0}
    if integration_steps:
        kw["integration_steps"] = int(integration_steps)
    if widths == "narrow":
        kw.update(NARROW)
    elif widths == "tiny":
        kw.update(TINY)
        kw["dropout"] = 0.0
    elif widths != "default":
        raise ValueError("widths must be one of %s, got %r" % (WIDTHS, widths))
    return ModelConfig(**kw)


def _peak_mb():
    return tf.config.experimental.get_memory_info("GPU:0")["peak"] / MB


def _current_mb():
    return tf.config.experimental.get_memory_info("GPU:0")["current"] / MB


def _val_pass(trainer, n, images, labels):
    """One deform validation pass (eval forward + eager metrics) -> (status, peak_mb, ms, error)."""
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
        t0 = time.time()
        fn = trainer._val_deform_fn(int(n))
        out = fn(tf.constant(images))
        trainer._deform_errors(labels, out["u"], out["u_inv"], out["v"].numpy(), out["v_raw"].numpy(),
                               seed=0, chunk=16, images=images)
        return "ok", _peak_mb(), (time.time() - t0) * 1e3, None
    except tf.errors.ResourceExhaustedError as e:
        return "oom", None, None, str(e).strip().splitlines()[0][:300]
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return "error", None, None, repr(e)[:300]


def measure(base, warmup_steps, steps, val_pass, seed=0):
    """One configuration -> result dict (status 'ok' or 'oom'; other exceptions propagate)."""
    from data_loading import params as P
    from models import cascade as CS
    from models import transforms as tfm
    from models.group_net import GroupRegLevelModel
    from runtime import fixed_data as fd
    from runtime import optimizers as O
    from runtime.run import Trainer, TrainConfig
    from runtime.schedulers import ScheduleConfig
    from tests.runtime_helpers import synthetic_group

    res = dict(base)
    level, task, n = int(base["level"]), base["task"], int(base["n"])
    shape = tuple(P.LEVEL_SHAPE[level])
    for k in ("frozen_ms", "val_peak_mb", "val_status", "val_ms"):
        res[k] = None
    if task == DEFORM:
        group = fd.synthetic_deform_group(level, n, seed=seed)
        mask = np.asarray(group["base_labels"]).reshape(shape) > 0
        S = tfm.brain_second_moment(mask, P.LEVEL_SPACING[level])
        labels = group["labels"]
    else:
        group = synthetic_group(level, n, task, seed=seed)
        S = group["S"]
        labels = None
    images = np.asarray(group["images"], np.float32).reshape((n,) + shape + (1,))
    cfg_m = model_config(base["widths"], base["integration_steps"])
    total = int(warmup_steps) + int(steps)
    cfg = TrainConfig(level=level, group_size=n, deform_group_size=n, steps=total + 1, val_every=0,
                      log_every=1, task_ratio={task: 1.0}, seed=seed, amp=bool(base["amp"]),
                      lambda_ncc_tmpl=float(base["lambda_ncc_tmpl"]),
                      lambda_ncc_pair=float(base["lambda_ncc_pair"]),
                      schedule=ScheduleConfig("constant", base_lr=3e-4, total_steps=total + 1,
                                              warmup_steps=0))
    mp = tf.keras.mixed_precision
    frozen = None
    trainer = model = batch = None
    try:
        if base["cascade"]:
            m4 = GroupRegLevelModel(CS.FROZEN_LEVEL, cfg_m).build_model(2)   # random level-4 weights
            frozen = CS.FrozenStage.from_model(m4)
            del m4
            res["frozen_params"] = int(sum(int(np.prod(v.shape)) for v in frozen.model.weights))
        else:
            res["frozen_params"] = None
        if base["amp"]:
            mp.set_global_policy("mixed_float16")
        model = GroupRegLevelModel(level, cfg_m, gradient_checkpointing=base["gradient_checkpointing"]
                                   ).build_model(n)
        res["params"] = int(sum(int(np.prod(v.shape)) for v in model.trainable_variables))
        trainer = Trainer(model, O.make_optimizer("adam", lr=3e-4, amp=bool(base["amp"])),
                          {task: fd.FixedGroupSource(group["batch"])}, S, cfg, cascade=frozen)
        batch = tuple(tf.constant(x) for x in group["batch"])
        try:
            tf.config.experimental.reset_memory_stats("GPU:0")
            for _ in range(max(int(warmup_steps), 1)):
                out = trainer.train_step(task, batch)              # trace + cuDNN setup
            res["warmup_peak_mb"] = _peak_mb()
            res["resident_mb"] = _current_mb()
            tf.config.experimental.reset_memory_stats("GPU:0")
            times = []
            for _ in range(int(steps)):
                t0 = time.time()
                out = trainer.train_step(task, batch)              # returns Python scalars (synced)
                times.append((time.time() - t0) * 1e3)
            res["peak_mb"] = _peak_mb()
            res["step_ms"] = float(np.median(times)) if times else None
            res["step_ms_mean"] = float(np.mean(times)) if times else None
            res["step_ms_all"] = [float(t) for t in times]
            res["loss"] = float(out["loss"])
            res["finite"] = bool(out["finite"])
            res["trace_counts"] = dict(trainer.trace_counts)
            res["status"] = "ok"
        except tf.errors.ResourceExhaustedError as e:
            res["status"] = "oom"
            res["error"] = str(e).strip().splitlines()[0][:300]
        if res["status"] == "ok" and frozen is not None:
            x = tf.constant(images)
            def frozen_fwd(t):
                if task == DEFORM:
                    return frozen.deform(t)["U"]
                return frozen.linear(t, task)

            ffn = tf.function(frozen_fwd)
            ffn(x).numpy()                                         # trace
            ft = []
            for _ in range(max(int(steps), 1)):
                t0 = time.time()
                ffn(x).numpy()
                ft.append((time.time() - t0) * 1e3)
            res["frozen_ms"] = float(np.median(ft))
            del ffn, x
        if val_pass and task == DEFORM:
            st, pk, ms, err = _val_pass(trainer, n, images, labels)
            res["val_status"], res["val_peak_mb"], res["val_ms"] = st, pk, ms
            if err:
                res["val_error"] = err
    finally:
        mp.set_global_policy("float32")
        del trainer, model, batch, group, frozen
    return res


def free_memory():
    tf.keras.backend.clear_session()
    gc.collect()


KEY_FIELDS = ("level", "task", "n", "widths", "cascade", "gradient_checkpointing", "amp",
              "lambda_ncc_tmpl", "lambda_ncc_pair", "integration_steps")


def config_key(r):
    return tuple(str(r.get(k)) for k in KEY_FIELDS)


def _sort_key(r):
    return tuple(str(r.get(k)) for k in KEY_FIELDS if k != "n") + (int(r.get("n", 0)),)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", type=int, choices=(4, 2), default=2)
    ap.add_argument("--tasks", default="rigid,affine,deform", help="comma-separated: rigid, affine, deform")
    ap.add_argument("--n", default="2,4", help="comma-separated group sizes")
    ap.add_argument("--widths", default="default", help="comma-separated: narrow, default, tiny")
    ap.add_argument("--cascade", default="true",
                    help="comma-separated true/false (true: frozen level-4 stage, --level 2 only)")
    ap.add_argument("--gradient-checkpointing", default="none",
                    help="comma-separated: none, integrator, decoder, both (deform only; linear uses none)")
    ap.add_argument("--amp", default="false", help="comma-separated true/false (mixed_float16)")
    ap.add_argument("--lambda-ncc-template", type=float, default=0.0,
                    help="LNCC template weight of the measured deform loss (0 = off; m8 recipe 1)")
    ap.add_argument("--lambda-ncc-pair", type=float, default=0.0, help="LNCC pairwise weight (0 = off)")
    ap.add_argument("--val-pass", default="false",
                    help="true: also measure one deform validation pass (deform configs only)")
    ap.add_argument("--integration-steps", type=int, default=0,
                    help="scaling-and-squaring steps T (0 = ModelConfig default; local-testing knob)")
    ap.add_argument("--warmup-steps", type=int, default=2, help="unmeasured steps first (trace, cuDNN)")
    ap.add_argument("--steps", type=int, default=10, help="measured steps after the warm-up")
    ap.add_argument("--gpu-memory-limit-mb", type=int, default=0,
                    help="fixed TF logical-device limit (0: memory growth, e.g. on the cluster)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="JSON output (merged with an existing file)")
    a = ap.parse_args(argv)
    configure_gpu(a.gpu_memory_limit_mb)
    from models.group_net import GRADIENT_CHECKPOINTING

    tasks = csv_list(a.tasks)
    for t in tasks:
        if t not in TASKS:
            ap.error("--tasks %r not in %s" % (t, TASKS))
    ns = sorted(int(x) for x in csv_list(a.n))
    if not ns or min(ns) < 2:
        ap.error("--n needs group sizes >= 2")
    widths = csv_list(a.widths)
    for w in widths:
        if w not in WIDTHS:
            ap.error("--widths %r not in %s" % (w, WIDTHS))
    ckpts = csv_list(a.gradient_checkpointing)
    for c in ckpts:
        if c not in GRADIENT_CHECKPOINTING:
            ap.error("--gradient-checkpointing %r not in %s" % (c, GRADIENT_CHECKPOINTING))
    try:
        cascades = [str2bool(x) for x in csv_list(a.cascade)]
        amps = [str2bool(x) for x in csv_list(a.amp)]
        val_pass = str2bool(a.val_pass)
    except ValueError as e:
        ap.error(str(e))
    if a.level != 2 and any(cascades):
        ap.error("--cascade true needs --level 2 (the frozen stage is level 4)")
    if a.steps < 1 or a.warmup_steps < 1:
        ap.error("--steps and --warmup-steps must be >= 1")
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("no GPU visible", file=sys.stderr)
        return 2
    env = {"python": sys.version.split()[0], "tensorflow": tf.__version__, "numpy": np.__version__,
           "gpu": tf.config.experimental.get_device_details(gpus[0]).get("device_name"),
           "gpu_memory_limit_mb": a.gpu_memory_limit_mb, "level": a.level, "warmup_steps": a.warmup_steps,
           "measured_steps": a.steps, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "tf_cudnn_use_autotune": os.environ.get("TF_CUDNN_USE_AUTOTUNE")}
    print("environment: %s" % env, flush=True)

    results = []
    for task in tasks:
        task_ckpts = ckpts if task == DEFORM else ["none"]
        for w in widths:
            for casc in cascades:
                for c in task_ckpts:
                    for amp in amps:
                        oom_at = None
                        for n in ns:
                            base = {"level": a.level, "task": task, "n": n, "widths": w, "cascade": casc,
                                    "gradient_checkpointing": c, "amp": amp,
                                    "lambda_ncc_tmpl": a.lambda_ncc_template,
                                    "lambda_ncc_pair": a.lambda_ncc_pair,
                                    "integration_steps": a.integration_steps}
                            if oom_at is not None:
                                results.append(dict(base, status="skipped", error="OOM at N = %d" % oom_at,
                                                    val_peak_mb=None, val_status=None, val_ms=None))
                                continue
                            t0 = time.time()
                            try:
                                r = measure(base, a.warmup_steps, a.steps, val_pass, a.seed)
                            except tf.errors.ResourceExhaustedError as e:
                                r = dict(base, status="oom", error=str(e).strip().splitlines()[0][:300],
                                         val_peak_mb=None, val_status=None, val_ms=None)
                            except Exception as e:  # noqa: BLE001
                                traceback.print_exc()
                                r = dict(base, status="error", error=repr(e)[:300],
                                         val_peak_mb=None, val_status=None, val_ms=None)
                            if r["status"] == "oom":
                                oom_at = n
                            r["wall_s"] = time.time() - t0
                            free_memory()
                            results.append(r)
                            tag = "%-6s N=%-3d %-7s casc=%-5s ckpt=%-10s amp=%-5s" % (task, n, w, casc, c, amp)
                            if r["status"] == "ok":
                                fz = "" if r["frozen_ms"] is None else "  frozen %6.1f ms" % r["frozen_ms"]
                                val = ""
                                if r.get("val_status") is not None:
                                    val = "  val %s" % r["val_status"]
                                    if r["val_status"] == "ok":
                                        val += " peak %8.1f MB %7.1f ms" % (r["val_peak_mb"], r["val_ms"])
                                print("%s peak %8.1f MB (warm-up %8.1f, resident %7.1f)  step %7.1f ms%s%s"
                                      % (tag, r["peak_mb"], r["warmup_peak_mb"], r["resident_mb"],
                                         r["step_ms"], fz, val), flush=True)
                            else:
                                print("%s %s: %s (val %s)" % (tag, r["status"].upper(), r.get("error"),
                                                              r.get("val_status")), flush=True)

    merged = {}
    if os.path.exists(a.out):
        with open(a.out) as f:
            old = json.load(f)
        for r in old.get("results", []):
            merged[config_key(r)] = r
    for r in results:
        merged[config_key(r)] = r
    out = {"environment": env, "results": sorted(merged.values(), key=_sort_key)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("written %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
