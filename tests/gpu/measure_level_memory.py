#!/usr/bin/env python
"""GPU memory and step time of one training step per task at a pyramid level (milestones 8, 9).

Not part of unittest discovery. Runs locally (GTX 1080, WSL tf_gpu) and on the cluster (inside
tf10_nv.sif, jobs slurm/m8_memory.sh and slurm/m9_memory.sh) on ONE GPU, e.g.:

    python tests/gpu/measure_level_memory.py --level 2 --tasks deform --n 2,4,8 --widths default \
        --cascade true --gradient-checkpointing none,both --amp false,true --val-pass true \
        --lambda-ncc-template 1 --lambda-ncc-pair 1 --out memory_level2.json
    python tests/gpu/measure_level_memory.py --level 1 --tasks deform --n 2,3,4 --widths f0d1 \
        --cascade true --gradient-checkpointing full --amp true --lean-sampler true \
        --split-group-conv true --integration-steps 6,7 --val-pass true --val-pair-chunk 4 \
        --lambda-ncc-template 1 --lambda-ncc-pair 1 --out memory_m9.json

Same structure as the milestone-6 `measure_deform_memory.py` (kept as is for reproducibility);
milestone 9 extensions: `.agents/tasks/milestone-9/design-m9-phase1.md` sections 5.1 / 5.2.
For every configuration (task x widths x cascade x gradient checkpointing x AMP x T x lean
sampler x lean losses x split group conv x N) it builds the trainable level-`--level` model
(deform decoder always on, i.e. the three-task model) and a `Trainer` with ratio <task>:1 (Adam
3e-4, global clip 12) on synthetic data (no DALI): linear `tests.runtime_helpers.synthetic_group
(level, n, task)`, deform `fd.synthetic_deform_group(level, n)` (labels 0..8, 19 one-hot
channels). It runs --warmup-steps steps (trace, cuDNN setup), then --steps measured steps; TF's
allocator statistics (`tf.config.experimental.get_memory_info('GPU:0')`, reset before the
measured steps) give the peak. Gradient checkpointing and the lean sampler / lean losses apply to
deform only (linear configs use 'none' / false); split group conv applies to every task. `--amp
true` sets `mixed_float16` before building the trainable model and uses a LossScaleOptimizer (the
frozen stages stay float32); the policy is reset to float32 after each configuration.

Widths: `narrow` = `tests.runtime_helpers.NARROW`, `tiny` = `tests.runtime_helpers.TINY`,
`default` = `ModelConfig()` (decoder widths None = half the encoder widths; the same model as
the `default` preset), any other `models.width_presets` name, or `custom` (needs --filters and
--decoder-widths; the widths key is ``custom:f=16-32-...;d=...``).

Build (R1.3 fix, design 2.4): EVERY model (trainable, frozen level 4, stand-in level 4 / 2) is
built with `build_model(min(N, --build-n))` (default 2). The variables do not depend on N and the
build runs with training=False, so it has no numerical effect on the measured steps; a build
forward can no longer be recorded as a training-configuration failure. Records store `build_n`.

Cascade: `--cascade true` at level 2 = a `models.cascade.FrozenStage` of a random float32
level-4 model; at level 1 = `tests/gpu/frozen_chain_standin.FrozenChainStandIn` (random float32
level-4 and level-2 models, `PREV_LEVEL` patched to {2: 4, 1: 2} while the configuration runs;
design 5.2; since phase 2 the patch is a no-op and the stand-in's ``n_stages`` = 2 makes the
Trainer run its frozen-chain branches, the same as a real level-1 run with
`models.cascade.FrozenChain`; the `--val-pass` eval forward then also returns the level-4-alone
fields U4, U4_inv, V4 (the inputs of the ``val_<set>_frozen4`` baselines);
no record-schema change, so records stay schema-compatible with job 10516237, but
``val_peak_mb`` now includes the ``_frozen4`` fields, ~+1 GB at level-1 N 4); invalid at level 4. `cascade_kind` = none | frozen4 | standin_chain_4_2. For cascade
configurations with training status 'ok', `frozen_ms` = median of the frozen forward alone, and
`frozen_peak_mb` = the allocator peak of one frozen forward with the trainer, model, optimizer
slots and batch still alive (frozen_status ok | oom | error; an OOM there leaves the training
status unchanged).

Status per configuration: 'ok' | 'oom' (ResourceExhaustedError) | 'limit' (an
InvalidArgumentError / InternalError / UnknownError / ValueError whose message names a 32-bit
size limit: "2^31", "2**31", "INT_MAX", "2147483647" or "num_elements") | 'error' (anything
else; repr, the traceback goes to stderr) | 'skipped' (larger N of a configuration after an
'oom' or 'limit').

`--val-pass true` (deform only): after the measured steps (also after a training OOM),
`reset_memory_stats`, then one validation pass on the synthetic group as `_val_deform_errors`
does it: the eval-mode `trainer._val_deform_fn(n)` forward (cascade branch with --cascade true)
followed by `trainer._deform_errors(labels, u, u_inv, v, v_raw, seed=0, chunk=--val-pair-chunk,
images=images)` (eager warps). Records val_peak_mb, val_status ('ok' | 'oom' | 'error') and
val_ms (wall time of that first pass, incl. the validation function's trace); an OOM there does
not mark the training configuration as OOM. null for linear tasks or without --val-pass.

`--integration-steps` (comma list; 0 = the ModelConfig default, resolved before the record is
written, tool-only convention: main.py's 0 means T = 0) is a grid axis.

`--min-pairs` (comma list of 1 / n, default 1; milestone 9 phase 2) is a grid axis: the training
pair-count floor (`TrainConfig.min_pairs`), also passed to the memory-model Workload.
`--pair-recompute` (true / false, default false; tf-intensity-aug addendum) sets
`TrainConfig.pair_recompute` for the deform configurations (record field pair_recompute; old
records read as false). The memory model has no term for it (its prediction stays the default path).
`--max-pairs` (comma list of integers >= 0, default 0 = no cap; retrain-prereqs design 16.6) is a
grid axis: `TrainConfig.max_pairs`; each record stores max_pairs and the resulting training pair
count n_pairs (`models.memory_model.num_pairs` with p 0.1, the TrainConfig default of p_pair and
p_pdice); old records read max_pairs 0. The memory-model prediction ignores the cap.
`--int32-check true` (default false): before running a configuration, its largest per-group tensor
at N (`int32_violation`: training step + frozen / validation forwards of the memory model's tensor
inventory, build excluded) is compared with 2^31 - 1 elements; a violation is written as a 'limit' record
("int32 element check: ...", not run) and the larger N of that configuration are skipped.

Predictions: `predicted_peak_mb`, `predicted_phase`, `pred_over_meas` (= predicted / measured
peak), `largest_tensor_elems`, `largest_tensor_desc` from `models.memory_model.predict` (margin
0) with the stored coefficients (--coefficients, default models/memory_coefficients.json); null
with a stderr warning if the file is missing.

JSON (``--out``): {"environment": {...}, "results": [{level, task, n, widths, cascade,
gradient_checkpointing, amp, lambda_ncc_tmpl, lambda_ncc_pair, integration_steps, lean_sampler,
lean_losses, split_group_conv, val_pair_chunk, cascade_kind, build_n, min_pairs, max_pairs, n_pairs,
pair_recompute, params, frozen_params,
status, peak_mb (measured steps), warmup_peak_mb, resident_mb (allocated after the warm-up:
weights of all stages, optimizer slots, batch), step_ms (median), step_ms_mean, step_ms_all,
frozen_ms, frozen_peak_mb, frozen_status, trace_counts, finite, loss, val_peak_mb, val_status,
val_ms, predicted_peak_mb, predicted_phase, pred_over_meas, largest_tensor_elems,
largest_tensor_desc, wall_s, error}]}. The output is merged and written atomically AFTER EVERY
configuration (`<out>.tmp.<pid>`, flush + fsync, `os.replace`; an OSError is fatal): records
with the same configuration key (KEY_FIELDS; old records without the milestone-9 fields read as
lean/split false, val_pair_chunk 16, min_pairs '1', cascade_kind frozen4/none from `cascade`, integration_steps
0 = the resolved T) are replaced, new ones appended, so several invocations build one table and
a killed job loses at most the configuration being measured. main(argv) returns 0 unless it
crashes (2 without a GPU).

`--isolate true` (milestone-9 FEAT-004 fix; default false = the in-process loop above): every
configuration (one N) runs in its OWN Python child process (`sys.executable` this script, single
values on every axis, `--isolate false`, the same --out). Reason: in one process the allocator
keeps memory across configurations even after `clear_session()` + `gc.collect()` (local GTX hold-out:
`resident_mb` 73 -> 329 MB over 48 narrow configurations, 424 -> 1602 MB over 16 default ones), which
inflates later peaks. The child measures and writes its record (atomic merge as above); the parent
reads it back by configuration key for the skip-after-OOM rule, and writes an 'error' record
("child exit code <rc>") if the child crashed without writing one. The parent never touches the GPU
allocator. Cost: one interpreter + TF + CUDA start per configuration (~20-40 s).
"""
import argparse
import contextlib
import gc
import importlib.util
import json
import os
import subprocess
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
CUSTOM = "custom"
DEFORM = "deform"
COEFFICIENTS = os.path.join(REPO, "models", "memory_coefficients.json")
STANDIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_chain_standin.py")
CASCADE_KINDS = {(2, True): "frozen4", (1, True): "standin_chain_4_2"}
MODEL_CASCADE = {"none": "none", "frozen4": "frozen4", "standin_chain_4_2": "chain42"}
LIMIT_SUBSTRINGS = ("2^31", "2**31", "INT_MAX", "2147483647", "num_elements")


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


def width_names():
    from models import width_presets as WP
    return tuple(WIDTHS) + tuple(p for p in WP.PRESETS if p not in WIDTHS)


def custom_key(filters, decoder_widths):
    return "custom:f=%s;d=%s" % ("-".join(str(int(x)) for x in filters),
                                 "-".join(str(int(x)) for x in decoder_widths))


def _parse_custom(widths):
    """'custom:f=a-b-...;d=a-b-...' -> (filters tuple, decoder_widths tuple)."""
    body = widths[len(CUSTOM) + 1:]
    parts = dict(p.split("=", 1) for p in body.split(";") if "=" in p)
    if "f" not in parts or "d" not in parts:
        raise ValueError("custom widths key needs f= and d=, got %r" % widths)
    return (tuple(int(x) for x in parts["f"].split("-")), tuple(int(x) for x in parts["d"].split("-")))


def model_config(widths, integration_steps=0):
    """Three-task ModelConfig (deform decoder on, dropout 0) of a widths key; integration_steps
    0 = the ModelConfig default."""
    from models.config import ModelConfig
    from models import width_presets as WP
    from tests.runtime_helpers import NARROW, TINY
    kw = {"deform_decoder": True, "dropout": 0.0}
    if integration_steps:
        kw["integration_steps"] = int(integration_steps)
    if widths == "narrow":
        kw.update(NARROW)
    elif widths == "tiny":
        kw.update(TINY)
        kw["dropout"] = 0.0
    elif str(widths).startswith(CUSTOM + ":"):
        f, d = _parse_custom(widths)
        kw.update(filters=f, decoder_widths=d)
    elif widths != "default":
        if widths not in WP.PRESETS:
            raise ValueError("widths must be one of %s or %s:f=..;d=.., got %r" % (width_names(), CUSTOM, widths))
        kw.update(WP.preset_kwargs(widths))
    return ModelConfig(**kw)


def classify_error(e):
    """'oom' | 'limit' | 'error' (design 5.1, review F14)."""
    if isinstance(e, tf.errors.ResourceExhaustedError):
        return "oom"
    limit_types = (tf.errors.InvalidArgumentError, tf.errors.InternalError, tf.errors.UnknownError, ValueError)
    if isinstance(e, limit_types) and any(s in str(e) for s in LIMIT_SUBSTRINGS):
        return "limit"
    return "error"


def gpu_family(name):
    """memory_model GPU name of a TF device name."""
    s = str(name or "").upper()
    if "1080" in s:
        return "gtx1080"
    if "A100" in s:
        return "a100"
    if "A6000" in s:
        return "a6000"
    return "rtx8000"


def _peak_mb():
    return tf.config.experimental.get_memory_info("GPU:0")["peak"] / MB


def _current_mb():
    return tf.config.experimental.get_memory_info("GPU:0")["current"] / MB


def _first_line(e):
    s = str(e).strip()
    return (s.splitlines()[0] if s else repr(e))[:300]


def _standin_module():
    spec = importlib.util.spec_from_file_location("frozen_chain_standin", STANDIN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _val_pass(trainer, n, images, labels, chunk=16):
    """One deform validation pass (eval forward + eager metrics) -> (status, peak_mb, ms, error)."""
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
        t0 = time.time()
        fn = trainer._val_deform_fn(int(n))
        out = fn(tf.constant(images))
        trainer._deform_errors(labels, out["u"], out["u_inv"], out["v"].numpy(), out["v_raw"].numpy(),
                               seed=0, chunk=int(chunk), images=images)
        return "ok", _peak_mb(), (time.time() - t0) * 1e3, None
    except tf.errors.ResourceExhaustedError as e:
        return "oom", None, None, _first_line(e)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return "error", None, None, repr(e)[:300]


def measure(base, warmup_steps, steps, val_pass, seed=0):
    """One configuration -> result dict (status 'ok', 'oom' or 'limit'; other exceptions propagate)."""
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
    build_n = int(base["build_n"])
    shape = tuple(P.LEVEL_SHAPE[level])
    for k in ("frozen_ms", "frozen_peak_mb", "frozen_status", "val_peak_mb", "val_status", "val_ms"):
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
                      val_pair_chunk=int(base["val_pair_chunk"]), min_pairs=str(base.get("min_pairs", "1")),
                      max_pairs=int(base.get("max_pairs", 0)),
                      pair_recompute=bool(base.get("pair_recompute", False)),
                      lean_sampler=bool(base["lean_sampler"]), lean_losses=bool(base["lean_losses"]),
                      schedule=ScheduleConfig("constant", base_lr=3e-4, total_steps=total + 1,
                                              warmup_steps=0))
    mp = tf.keras.mixed_precision
    frozen = None
    trainer = model = batch = None
    stack = contextlib.ExitStack()
    try:
        if base["cascade"] and level == 1:
            sc = _standin_module()
            frozen = sc.build_standin(cfg_m, build_n=build_n)
            stack.enter_context(sc.patched_prev_level())
            res["frozen_params"] = frozen.n_params
        elif base["cascade"]:
            with CS._float32_policy():                                 # random level-4 weights
                m4 = GroupRegLevelModel(CS.FROZEN_LEVEL, cfg_m, gradient_checkpointing="none"
                                        ).build_model(build_n)
            frozen = CS.FrozenStage(m4)
            del m4
            res["frozen_params"] = int(sum(int(np.prod(v.shape)) for v in frozen.model.weights))
        else:
            res["frozen_params"] = None
        if base["amp"]:
            mp.set_global_policy("mixed_float16")
        model = GroupRegLevelModel(level, cfg_m, gradient_checkpointing=base["gradient_checkpointing"],
                                   lean_sampler=bool(base["lean_sampler"]),
                                   split_group_conv=bool(base["split_group_conv"])).build_model(build_n)
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
        except Exception as e:  # noqa: BLE001
            st = classify_error(e)
            if st == "error":
                raise
            res["status"] = st
            res["error"] = _first_line(e)
        if res["status"] == "ok" and frozen is not None:
            x = tf.constant(images)

            def frozen_fwd(t):
                if task == DEFORM:
                    return frozen.deform(t)["U"]
                return frozen.linear(t, task)

            ffn = tf.function(frozen_fwd)
            try:
                ffn(x).numpy()                                     # trace
                ft = []
                for _ in range(max(int(steps), 1)):
                    t0 = time.time()
                    ffn(x).numpy()
                    ft.append((time.time() - t0) * 1e3)
                res["frozen_ms"] = float(np.median(ft))
                tf.config.experimental.reset_memory_stats("GPU:0")
                ffn(x).numpy()
                res["frozen_peak_mb"] = _peak_mb()
                res["frozen_status"] = "ok"
            except tf.errors.ResourceExhaustedError as e:
                res["frozen_status"] = "oom"
                res["frozen_error"] = _first_line(e)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                res["frozen_status"] = "error"
                res["frozen_error"] = repr(e)[:300]
            del ffn, x
        if val_pass and task == DEFORM:
            st, pk, ms, err = _val_pass(trainer, n, images, labels, chunk=base["val_pair_chunk"])
            res["val_status"], res["val_peak_mb"], res["val_ms"] = st, pk, ms
            if err:
                res["val_error"] = err
    finally:
        stack.close()
        mp.set_global_policy("float32")
        del trainer, model, batch, group, frozen
    return res


def free_memory():
    tf.keras.backend.clear_session()
    gc.collect()


# ----------------------------------------------------------------------------- predictions
def load_coefficients(path):
    """Stored memory-model coefficients, or None (stderr warning) if the file is missing."""
    from models import memory_model as MM
    if not path or not os.path.exists(path):
        print("WARNING: memory-model coefficients %r not found; predicted_* fields are null" % (path,),
              file=sys.stderr)
        return None
    return MM.load_coefficients(path)


def _workload(base, gpu):
    """`models.memory_model.Workload` of a configuration record."""
    from models import memory_model as MM
    return MM.Workload(base["level"], base["task"], base["n"],
                       model_config=model_config(base["widths"], base["integration_steps"]),
                       checkpointing=base["gradient_checkpointing"], amp=bool(base["amp"]),
                       integration_steps=int(base["integration_steps"]),
                       lncc=bool(float(base["lambda_ncc_tmpl"]) > 0 or float(base["lambda_ncc_pair"]) > 0),
                       lean_sampler=bool(base["lean_sampler"]), lean_losses=bool(base["lean_losses"]),
                       split_group_conv=bool(base["split_group_conv"]),
                       cascade=MODEL_CASCADE[base["cascade_kind"]],
                       val_pair_chunk=int(base["val_pair_chunk"]), build_n=int(base["build_n"]), gpu=gpu,
                       min_pairs=str(base.get("min_pairs", "1")))


def int32_violation(base, gpu):
    """--int32-check: the largest per-group tensor of the configuration at its N (training step and
    frozen / validation forwards of `models.memory_model.tensor_inventory`: every block tensor incl.
    the full-resolution stem) -> (elements, description); the configuration violates the int32
    per-tensor element limit iff elements > 2^31 - 1 (the level-1 job 10517266 failure). The build
    tensors are excluded: the build runs at min(N, --build-n), not at N, and the model's level-1
    build list (full-resolution decoder concat at N 2) is above the limit although level-1 builds
    run fine on the cluster."""
    from models import memory_model as MM
    train, fwd, _build = MM.tensor_inventory(_workload(base, gpu))
    t = max(train + fwd, key=lambda x: x["elements"])
    return int(t["elements"]), "%s %s" % (t["desc"], "x".join(str(s) for s in t["shape"]))


def predicted_fields(base, gpu, coeffs):
    """predicted_peak_mb, predicted_phase, largest_tensor_elems, largest_tensor_desc (null without
    coefficients or if the model cannot predict the configuration)."""
    out = {"predicted_peak_mb": None, "predicted_phase": None, "largest_tensor_elems": None,
           "largest_tensor_desc": None}
    if coeffs is None:
        return out
    try:
        from models import memory_model as MM
        w = _workload(base, gpu)
        p = MM.predict(w, coeffs, margin=0.0)
        out["predicted_peak_mb"] = float(p["peak_gib"] * 1024.0)
        out["predicted_phase"] = p["peak_phase"]
        out["largest_tensor_elems"] = int(p["largest_tensor"]["elements"])
        out["largest_tensor_desc"] = p["largest_tensor"].get("desc")
    except Exception as e:  # noqa: BLE001
        print("WARNING: memory-model prediction failed for %s: %r" % (base, e), file=sys.stderr)
    return out


# ----------------------------------------------------------------------------- merge + write
KEY_FIELDS = ("level", "task", "n", "widths", "cascade", "gradient_checkpointing", "amp",
              "lambda_ncc_tmpl", "lambda_ncc_pair", "integration_steps", "lean_sampler", "lean_losses",
              "split_group_conv", "val_pair_chunk", "cascade_kind", "min_pairs", "pair_recompute",
              "max_pairs")


def _resolved_t(r):
    t = r.get("integration_steps", 0)
    if not t:
        try:
            t = model_config(r.get("widths", "default"), 0).integration_steps
        except Exception:  # noqa: BLE001  (unknown widths key: keep the stored value)
            pass
    return t


def _key_value(r, k):
    if k in ("lean_sampler", "lean_losses", "split_group_conv"):
        return r.get(k, False)
    if k == "val_pair_chunk":
        return r.get(k, 16)
    if k == "pair_recompute":
        return bool(r.get(k, False))
    if k == "min_pairs":
        return r.get(k, "1")
    if k == "max_pairs":
        return int(r.get(k, 0) or 0)
    if k == "cascade_kind":
        return r.get(k, "frozen4" if r.get("cascade") else "none")
    if k == "integration_steps":
        return _resolved_t(r)
    return r.get(k)


def config_key(r):
    """Configuration key; missing milestone-9 fields of old records read with their defaults."""
    return tuple(str(_key_value(r, k)) for k in KEY_FIELDS)


def _sort_key(r):
    return tuple(str(_key_value(r, k)) for k in KEY_FIELDS if k != "n") + (int(r.get("n", 0)),)


def write_merged(path, env, records):
    """Merge ``records`` into the JSON at ``path`` (same config_key replaced) and write it
    atomically: <path>.tmp.<pid> in the same folder, flush + fsync, os.replace. OSError is fatal
    (propagates); the temp file is removed in ``finally``."""
    merged = {}
    order = []
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        for r in old.get("results", []):
            k = config_key(r)
            if k not in merged:
                order.append(k)
            merged[k] = r
    for r in records:
        k = config_key(r)
        if k not in merged:
            order.append(k)
        merged[k] = r
    out = {"environment": env, "results": sorted((merged[k] for k in order), key=_sort_key)}
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = "%s.tmp.%d" % (os.path.abspath(path), os.getpid())
    try:
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _null_record(base, status, error):
    r = dict(base, status=status, error=error)
    for k in ("frozen_ms", "frozen_peak_mb", "frozen_status", "val_peak_mb", "val_status", "val_ms"):
        r.setdefault(k, None)
    return r


def _b(x):
    return "true" if x else "false"


def child_argv(a, base, val_pass):
    """argv of the `--isolate true` child that measures exactly the configuration ``base``."""
    w = base["widths"]
    if str(w).startswith(CUSTOM + ":"):
        f, d = _parse_custom(w)
        widths = [CUSTOM, "--filters", ",".join(str(x) for x in f),
                  "--decoder-widths", ",".join(str(x) for x in d)]
    else:
        widths = [w]
    return ([sys.executable, os.path.abspath(__file__), "--isolate", "false",
             "--level", str(base["level"]), "--tasks", base["task"], "--n", str(base["n"]), "--widths"]
            + widths
            + ["--cascade", _b(base["cascade"]), "--gradient-checkpointing", base["gradient_checkpointing"],
               "--amp", _b(base["amp"]), "--lean-sampler", _b(base["lean_sampler"]),
               "--lean-losses", _b(base["lean_losses"]), "--split-group-conv", _b(base["split_group_conv"]),
               "--lambda-ncc-template", repr(float(base["lambda_ncc_tmpl"])),
               "--lambda-ncc-pair", repr(float(base["lambda_ncc_pair"])),
               "--val-pass", _b(val_pass), "--val-pair-chunk", str(base["val_pair_chunk"]),
               "--integration-steps", str(base["integration_steps"]), "--build-n", str(a.build_n),
               "--warmup-steps", str(a.warmup_steps), "--steps", str(a.steps),
               "--gpu-memory-limit-mb", str(a.gpu_memory_limit_mb), "--coefficients", str(a.coefficients),
               "--min-pairs", str(base.get("min_pairs", "1")),
               "--max-pairs", str(int(base.get("max_pairs", 0))),
               "--pair-recompute", _b(base.get("pair_recompute", False)),
               "--seed", str(a.seed), "--out", a.out])


def _read_record(path, base):
    """The record of configuration ``base`` in the JSON at ``path`` (None if absent)."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    k = config_key(base)
    for r in d.get("results", []):
        if config_key(r) == k:
            return r
    return None


def measure_isolated(a, base, val_pass):
    """Run ``base`` in a child process -> (record, written_by_child). A crashed child (non-zero
    exit or no record) gives an 'error' record that the caller writes."""
    old = _read_record(a.out, base)
    t0 = time.time()
    rc = subprocess.run(child_argv(a, base, val_pass), stdin=subprocess.DEVNULL).returncode
    r = _read_record(a.out, base)
    fresh = r is not None and (old is None or r != old)
    if rc == 0 and fresh:
        return r, True
    r = _null_record(base, "error", "child exit code %d%s" % (rc, "" if fresh else ", no record written"))
    r["wall_s"] = time.time() - t0
    return r, False


def _parse_widths(ap, a):
    names = width_names()
    out = []
    for w in csv_list(a.widths):
        if w == CUSTOM:
            if not a.filters or not a.decoder_widths:
                ap.error("--widths custom needs --filters and --decoder-widths")
            try:
                f = [int(x) for x in csv_list(a.filters)]
                d = [int(x) for x in csv_list(a.decoder_widths)]
            except ValueError:
                ap.error("--filters / --decoder-widths must be comma lists of integers")
            if len(f) != 6 or len(d) != 5 or min(f + d) < 1:
                ap.error("--filters needs 6 and --decoder-widths 5 integers >= 1")
            out.append(custom_key(f, d))
        elif w in names:
            out.append(w)
        else:
            ap.error("--widths %r not in %s or %s" % (w, names, CUSTOM))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", type=int, choices=(4, 2, 1), default=2)
    ap.add_argument("--tasks", default="rigid,affine,deform", help="comma-separated: rigid, affine, deform")
    ap.add_argument("--n", default="2,4", help="comma-separated group sizes")
    ap.add_argument("--widths", default="default",
                    help="comma-separated: narrow, default, tiny, a models.width_presets name, or custom")
    ap.add_argument("--filters", default=None, help="--widths custom: 6 encoder widths (comma list)")
    ap.add_argument("--decoder-widths", default=None, help="--widths custom: 5 decoder widths (comma list)")
    ap.add_argument("--cascade", default="true",
                    help="comma-separated true/false (true: frozen level-4 stage at --level 2, the "
                         "frozen-chain stand-in at --level 1; invalid at level 4)")
    ap.add_argument("--gradient-checkpointing", default="none",
                    help="comma-separated: none, integrator, decoder, both, full (deform only; linear uses none)")
    ap.add_argument("--amp", default="false", help="comma-separated true/false (mixed_float16)")
    ap.add_argument("--lean-sampler", default="false", help="comma-separated true/false (deform only)")
    ap.add_argument("--lean-losses", default="false", help="comma-separated true/false (deform only)")
    ap.add_argument("--split-group-conv", default="false", help="comma-separated true/false")
    ap.add_argument("--lambda-ncc-template", type=float, default=0.0,
                    help="LNCC template weight of the measured deform loss (0 = off; m8 recipe 1)")
    ap.add_argument("--lambda-ncc-pair", type=float, default=0.0, help="LNCC pairwise weight (0 = off)")
    ap.add_argument("--val-pass", default="false",
                    help="true: also measure one deform validation pass (deform configs only)")
    ap.add_argument("--val-pair-chunk", type=int, default=16, help="pair chunk of the validation pass (>= 1)")
    ap.add_argument("--pair-recompute", default="false",
                    help="true: --pair-recompute in the deform training step (one pair at a time under "
                    "tf.recompute_grad; recorded as pair_recompute, deform configs only)")
    ap.add_argument("--min-pairs", default="1",
                    help="comma-separated 1/n: training pair-count floor (main.py --min-pairs; default 1)")
    ap.add_argument("--max-pairs", default="0",
                    help="comma-separated integers >= 0: training pair cap (main.py --max-pairs; "
                    "0 = no cap, default); recorded as max_pairs with the resulting n_pairs")
    ap.add_argument("--integration-steps", default="0",
                    help="comma-separated scaling-and-squaring steps T (0 = the ModelConfig default, "
                         "tool-only convention; resolved before the record is written)")
    ap.add_argument("--build-n", type=int, default=2,
                    help="every model is built with build_model(min(N, build_n)) (R1.3 fix)")
    ap.add_argument("--warmup-steps", type=int, default=2, help="unmeasured steps first (trace, cuDNN)")
    ap.add_argument("--steps", type=int, default=10, help="measured steps after the warm-up")
    ap.add_argument("--gpu-memory-limit-mb", type=int, default=0,
                    help="fixed TF logical-device limit (0: memory growth, e.g. on the cluster)")
    ap.add_argument("--coefficients", default=COEFFICIENTS, help="memory-model coefficients JSON")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="JSON output (merged with an existing file)")
    ap.add_argument("--int32-check", default="false",
                    help="true: before each configuration, compute its largest per-group tensor at N "
                    "(models.memory_model.tensor_inventory: training step, frozen / validation "
                    "forwards; build excluded); above 2^31 - 1 elements write a 'limit' record "
                    "without running it (larger N skipped)")
    ap.add_argument("--isolate", default="false",
                    help="true: every configuration (one N) in its own child process (no memory carried "
                         "over between configurations; slurm/m9_memory.sh)")
    a = ap.parse_args(argv)
    try:
        a.isolate = str2bool(a.isolate)
    except ValueError as e:
        ap.error("--isolate: %s" % e)
    try:
        a.int32_check = str2bool(a.int32_check)
    except ValueError as e:
        ap.error("--int32-check: %s" % e)
    if not a.isolate:
        configure_gpu(a.gpu_memory_limit_mb)
    from models.group_net import GRADIENT_CHECKPOINTING

    tasks = csv_list(a.tasks)
    for t in tasks:
        if t not in TASKS:
            ap.error("--tasks %r not in %s" % (t, TASKS))
    ns = sorted(int(x) for x in csv_list(a.n))
    if not ns or min(ns) < 2:
        ap.error("--n needs group sizes >= 2")
    widths = _parse_widths(ap, a)
    ckpts = csv_list(a.gradient_checkpointing)
    for c in ckpts:
        if c not in GRADIENT_CHECKPOINTING:
            ap.error("--gradient-checkpointing %r not in %s" % (c, GRADIENT_CHECKPOINTING))

    def bool_axis(flag, value):         # comma list of true/false; errors name the option (R2.6)
        try:
            return [str2bool(x) for x in csv_list(value)]
        except ValueError as e:
            ap.error("%s: %s" % (flag, e))

    cascades = bool_axis("--cascade", a.cascade)
    amps = bool_axis("--amp", a.amp)
    lean_samplers = bool_axis("--lean-sampler", a.lean_sampler)
    lean_losses = bool_axis("--lean-losses", a.lean_losses)
    splits = bool_axis("--split-group-conv", a.split_group_conv)
    min_pairs = csv_list(a.min_pairs)
    if not min_pairs or any(m not in ("1", "n") for m in min_pairs):
        ap.error("--min-pairs needs a comma list of 1 / n, got %r" % a.min_pairs)
    try:
        max_pairs = [int(x) for x in csv_list(a.max_pairs)]
    except ValueError:
        ap.error("--max-pairs needs a comma list of integers >= 0, got %r" % a.max_pairs)
    if not max_pairs or min(max_pairs) < 0:
        ap.error("--max-pairs needs a comma list of integers >= 0, got %r" % a.max_pairs)
    try:
        val_pass = str2bool(a.val_pass)
    except ValueError as e:
        ap.error("--val-pass: %s" % e)
    try:
        str2bool(a.pair_recompute)
    except ValueError as e:
        ap.error("--pair-recompute: %s" % e)
    try:
        t_list = [int(x) for x in csv_list(a.integration_steps)]
    except ValueError:
        ap.error("--integration-steps must be a comma list of integers")
    if not t_list or min(t_list) < 0:
        ap.error("--integration-steps needs integers >= 0")
    if a.level == 4 and any(cascades):
        ap.error("--cascade true needs --level 2 or 1 (level 4 has no previous stage)")
    if a.steps < 1 or a.warmup_steps < 1:
        ap.error("--steps and --warmup-steps must be >= 1")
    if a.build_n < 1:
        ap.error("--build-n must be >= 1")
    if a.val_pair_chunk < 1:
        ap.error("--val-pair-chunk must be >= 1")
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("no GPU visible", file=sys.stderr)
        return 2
    gpu_name = tf.config.experimental.get_device_details(gpus[0]).get("device_name")
    env = {"python": sys.version.split()[0], "tensorflow": tf.__version__, "numpy": np.__version__,
           "gpu": gpu_name, "gpu_memory_limit_mb": a.gpu_memory_limit_mb, "level": a.level,
           "warmup_steps": a.warmup_steps, "measured_steps": a.steps, "build_n": a.build_n,
           "val_pair_chunk": a.val_pair_chunk, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "tf_cudnn_use_autotune": os.environ.get("TF_CUDNN_USE_AUTOTUNE")}
    print("environment: %s" % env, flush=True)
    coeffs = load_coefficients(a.coefficients)
    family = gpu_family(gpu_name)

    for task in tasks:
        is_deform = task == DEFORM
        task_ckpts = ckpts if is_deform else ["none"]
        task_ls = lean_samplers if is_deform else [False]
        task_ll = lean_losses if is_deform else [False]
        for w in widths:
            t_axis = []
            for t in t_list:
                rt = int(t) if t else int(model_config(w, 0).integration_steps)
                if rt not in t_axis:
                    t_axis.append(rt)
            for casc in cascades:
                for c in task_ckpts:
                    for amp in amps:
                        for T in t_axis:
                            for ls in sorted(set(task_ls), key=task_ls.index):
                                for ll in sorted(set(task_ll), key=task_ll.index):
                                    for sp in sorted(set(splits), key=splits.index):
                                        for mp in sorted(set(min_pairs), key=min_pairs.index):
                                            for k in sorted(set(max_pairs), key=max_pairs.index):
                                                _run_ns(a, env, coeffs, family, task, w, casc, c, amp, T,
                                                        ls, ll, sp, ns, val_pass, mp, k)
    print("written %s" % a.out)
    return 0


def _run_ns(a, env, coeffs, family, task, w, casc, c, amp, T, ls, ll, sp, ns, val_pass, mp="1",
            max_pairs=0):
    """All N of one configuration (larger N skipped after 'oom' / 'limit'); writes after each."""
    from models import memory_model as MM
    stop_at = None
    for n in ns:
        base = {"level": a.level, "task": task, "n": n, "widths": w, "cascade": casc,
                "gradient_checkpointing": c, "amp": amp, "lambda_ncc_tmpl": a.lambda_ncc_template,
                "lambda_ncc_pair": a.lambda_ncc_pair, "integration_steps": T, "lean_sampler": ls,
                "lean_losses": ll, "split_group_conv": sp, "val_pair_chunk": a.val_pair_chunk,
                "cascade_kind": CASCADE_KINDS.get((a.level, bool(casc)), "none"),
                "build_n": min(n, a.build_n), "min_pairs": mp,
                "pair_recompute": bool(task == "deform" and str2bool(a.pair_recompute)),
                # --max-pairs (retrain-prereqs design 16.6); TrainConfig p_pair = p_pdice = 0.1
                "max_pairs": int(max_pairs), "n_pairs": MM.num_pairs(n, 0.1, mp, int(max_pairs))}
        pred = predicted_fields(base, family, coeffs)
        big = int32_violation(base, family) if a.int32_check else None
        if stop_at is not None:
            r = _null_record(base, "skipped", "%s at N = %d" % (stop_at[0].upper(), stop_at[1]))
        elif big is not None and big[0] > MM.INT32_MAX:
            r = _null_record(base, "limit", "int32 element check: %s has %d elements > 2^31 - 1 "
                             "(not run)" % (big[1], big[0]))
            r["wall_s"] = 0.0
            stop_at = ("limit", n)
        elif a.isolate:
            r, by_child = measure_isolated(a, base, val_pass)
            if r["status"] in ("oom", "limit"):
                stop_at = (r["status"], n)
            if by_child:
                continue                     # the child wrote and printed its record
        else:
            t0 = time.time()
            try:
                r = measure(base, a.warmup_steps, a.steps, val_pass, a.seed)
            except Exception as e:  # noqa: BLE001
                st = classify_error(e)
                if st == "error":
                    traceback.print_exc()
                    r = _null_record(base, "error", repr(e)[:300])
                else:
                    r = _null_record(base, st, _first_line(e))
            if r["status"] in ("oom", "limit"):
                stop_at = (r["status"], n)
            r["wall_s"] = time.time() - t0
            free_memory()
        r.update(pred)
        meas = r.get("peak_mb")
        r["pred_over_meas"] = (float(pred["predicted_peak_mb"]) / float(meas)
                               if pred["predicted_peak_mb"] is not None and meas else None)
        write_merged(a.out, env, [r])
        tag = "%-6s N=%-3d %-8s casc=%-5s ckpt=%-10s amp=%-5s T=%d ls=%d ll=%d sp=%d" % (
            task, n, w, casc, c, amp, T, ls, ll, sp)
        pr = "" if pred["predicted_peak_mb"] is None else "  pred %8.1f MB (%s)" % (
            pred["predicted_peak_mb"], pred["predicted_phase"])
        if r["status"] == "ok":
            fz = "" if r["frozen_ms"] is None else "  frozen %6.1f ms" % r["frozen_ms"]
            if r.get("frozen_status") not in (None, "ok"):
                fz += "  frozen %s" % r["frozen_status"]
            elif r.get("frozen_peak_mb") is not None:
                fz += " peak %8.1f MB" % r["frozen_peak_mb"]
            val = ""
            if r.get("val_status") is not None:
                val = "  val %s" % r["val_status"]
                if r["val_status"] == "ok":
                    val += " peak %8.1f MB %7.1f ms" % (r["val_peak_mb"], r["val_ms"])
            print("%s peak %8.1f MB (warm-up %8.1f, resident %7.1f)  step %7.1f ms%s%s%s"
                  % (tag, r["peak_mb"], r["warmup_peak_mb"], r["resident_mb"], r["step_ms"], fz, val, pr),
                  flush=True)
        else:
            print("%s %s: %s (val %s)%s" % (tag, r["status"].upper(), r.get("error"), r.get("val_status"), pr),
                  flush=True)


if __name__ == "__main__":
    sys.exit(main())
