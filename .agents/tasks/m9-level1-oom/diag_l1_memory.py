#!/usr/bin/env python
"""Level-1 OOM diagnostic (m9): the REAL main.py path with GPU memory logged per phase.

Copy to tests/gpu/diag_l1_memory.py (not part of unittest discovery). Python 3.8 / TF 2.10 /
DALI 1.18 (tf10_nv.sif). Two modes:

  python tests/gpu/diag_l1_memory.py dali <main.py args...>
      main._build_dali_data exactly as main.py (same args -> same pipelines, validation sets,
      control files), then 3 fetches per task. Isolates the DALI device footprint (memory OUTSIDE
      TF's allocator) from TF.
  python tests/gpu/diag_l1_memory.py main <main.py args...>
      main.main(argv) unchanged, with logging wrappers around: _load_cascade_source,
      GroupRegLevelModel.build_model, FrozenChain.from_checkpoint, _build_dali_data, every
      source fetch (first 3 per task) and every Trainer.train_step (TF peak reset before each
      step; a ResourceExhaustedError is logged and re-raised).

Every record (one JSON line in $DIAG_OUT, also echoed to stdout) holds: phase, t (s since start),
tf_current_mb / tf_peak_mb (tf.config.experimental.get_memory_info: bytes IN USE in TF's
allocator, not reserved), proc_mb (nvidia-smi used memory of THIS pid = TF reserved arena + DALI +
CUDA context/handles), gpu_used_mb / gpu_total_mb (whole device). A background poller samples
nvidia-smi every second and writes a 'poll_max' record whenever the process maximum rises by
>= 256 MB, plus the GPU utilisation (to see a stall like job 10517337).
"""
import atexit
import json
import os
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.exists(os.path.join(REPO, "main.py")):
    REPO = os.getcwd()
if REPO not in sys.path:
    sys.path.insert(0, REPO)

T0 = time.time()
OUT = os.environ.get("DIAG_OUT", "diag_l1_memory.jsonl")
_lock = threading.Lock()
_poll = {"max_proc_mb": 0, "max_gpu_mb": 0, "last_util": None}


def _smi():
    """(proc_mb, gpu_used_mb, gpu_total_mb, util_pct) from nvidia-smi (None on failure)."""
    proc = used = total = util = None
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits"], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, universal_newlines=True, timeout=20).stdout
        pid = os.getpid()
        vals = []
        for line in q.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
                vals.append(int(parts[1]))
        proc = sum(vals) if vals else None
        g = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, universal_newlines=True, timeout=20).stdout
        parts = [p.strip() for p in g.strip().splitlines()[0].split(",")]
        used, total, util = int(parts[0]), int(parts[1]), int(parts[2])
    except Exception:  # noqa: BLE001
        pass
    return proc, used, total, util


def _tf_mem():
    tf = sys.modules.get("tensorflow")
    if tf is None:
        return None, None
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        return info["current"] / 2.0 ** 20, info["peak"] / 2.0 ** 20
    except Exception:  # noqa: BLE001  (GPU not initialised yet / allocator without stats)
        return None, None


def log(phase, **extra):
    cur, peak = _tf_mem()
    proc, used, total, util = _smi()
    rec = {"phase": phase, "t": round(time.time() - T0, 1), "tf_current_mb": cur, "tf_peak_mb": peak,
           "proc_mb": proc, "gpu_used_mb": used, "gpu_total_mb": total, "gpu_util_pct": util,
           "poll_max_proc_mb": _poll["max_proc_mb"]}
    rec.update(extra)
    line = json.dumps(rec)
    with _lock:
        with open(OUT, "a") as f:
            f.write(line + "\n")
            f.flush()
    print("DIAG " + line, flush=True)


def _poller():
    last_write = 0.0
    while True:
        proc, used, _, util = _smi()
        bump = False
        if proc is not None and proc >= _poll["max_proc_mb"] + 256:
            _poll["max_proc_mb"] = proc
            bump = True
        if used is not None:
            _poll["max_gpu_mb"] = max(_poll["max_gpu_mb"], used)
        _poll["last_util"] = util
        if bump or time.time() - last_write > 60:
            last_write = time.time()
            rec = {"phase": "poll_max" if bump else "poll", "t": round(time.time() - T0, 1),
                   "proc_mb": proc, "max_proc_mb": _poll["max_proc_mb"], "gpu_used_mb": used,
                   "max_gpu_mb": _poll["max_gpu_mb"], "gpu_util_pct": util}
            with _lock:
                with open(OUT, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        time.sleep(1.0)


def _start_poller():
    th = threading.Thread(target=_poller, daemon=True)
    th.start()
    atexit.register(lambda: log("exit", max_proc_mb=_poll["max_proc_mb"], max_gpu_mb=_poll["max_gpu_mb"]))


class LoggedIter(object):
    """Wraps a task source; logs after each of the first `k` fetches (DALI first runs)."""

    def __init__(self, task, it, k=3):
        self.task, self.it, self.k, self.n = task, it, int(k), 0

    def __iter__(self):
        return self

    def __next__(self):
        b = next(self.it)
        self.n += 1
        if self.n <= self.k:
            log("fetch", task=self.task, fetch=self.n)
        return b


def _patch_main_path():
    import tensorflow as tf
    import main as M
    from models import cascade as CS
    from models.group_net import GroupRegLevelModel
    from runtime import run as R

    orig_src = M._load_cascade_source

    def load_src(*a, **k):
        out = orig_src(*a, **k)
        log("cascade_source_checked")
        return out
    M._load_cascade_source = load_src

    orig_data = M._build_dali_data

    def build_data(*a, **k):
        log("before_dali_data")
        out = orig_data(*a, **k)
        log("after_dali_data_and_val_sets")
        return out
    M._build_dali_data = build_data

    orig_build = GroupRegLevelModel.build_model

    def build_model(self, n=1):
        out = orig_build(self, n)
        log("model_built", level=int(self.level), n=int(n), dtype=str(self.compute_dtype))
        return out
    GroupRegLevelModel.build_model = build_model

    orig_chain = CS.FrozenChain.from_checkpoint.__func__

    def chain_from_ckpt(cls, *a, **k):
        out = orig_chain(cls, *a, **k)
        log("frozen_chain_restored")
        return out
    CS.FrozenChain.from_checkpoint = classmethod(chain_from_ckpt)

    orig_init = R.Trainer.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        self.sources = {t: LoggedIter(t, it) for t, it in self.sources.items()}
        log("trainer_built", tasks=sorted(self.sources))
    R.Trainer.__init__ = init

    orig_step = R.Trainer.train_step

    def train_step(self, task, batch_tuple, skip=False):
        s = int(self.step.numpy())
        try:
            tf.config.experimental.reset_memory_stats("GPU:0")
        except Exception:  # noqa: BLE001
            pass
        log("step_begin", task=task, step=s)
        t = time.time()
        try:
            res = orig_step(self, task, batch_tuple, skip=skip)
        except tf.errors.ResourceExhaustedError as e:
            log("step_oom", task=task, step=s, seconds=round(time.time() - t, 1),
                error=str(e).strip().splitlines()[0][:400])
            raise
        log("step_end", task=task, step=s, seconds=round(time.time() - t, 1), finite=bool(res.get("finite")))
        return res
    R.Trainer.train_step = train_step


def run_main(argv):
    _start_poller()
    log("start", mode="main", argv=argv, tf_gpu_allocator=os.environ.get("TF_GPU_ALLOCATOR"))
    import tensorflow  # noqa: F401  (imported first so the wrappers can read memory stats)
    _patch_main_path()
    import main as M
    rc = 1
    try:
        rc = M.main(argv)
    finally:
        log("main_returned", rc=rc)
    return rc


def run_dali(argv):
    _start_poller()
    log("start", mode="dali", argv=argv)
    from runtime import args as A
    args = A.parse_args(argv)
    import tensorflow as tf
    from models.precision import configure_tf_precision
    configure_tf_precision(tf32=False, memory_growth=not args.gpu_memory_limit_mb)
    import main as M
    from data_loading import data_module as dm
    from data_loading import params as P
    from data_loading import postprocess
    from models import transforms as tfm
    from runtime import fixed_data as fd
    tf.zeros([1]).numpy()                           # CUDA context + TF device, nothing else
    log("tf_initialised")
    sources, val_sets, S, desc, controls = M._build_dali_data(args, args.level, (fd, P, tfm, postprocess, dm), 0)
    log("after_dali_data_and_val_sets", desc=desc[:120])
    for k in range(3):
        for t, it in sources.items():
            b = next(it)
            float(tf.reduce_sum(tf.cast(b[0], tf.float32)).numpy())    # sync on the output
            log("fetch", task=t, fetch=k + 1)
    time.sleep(10.0)                                # DALI prefetches the next iteration meanwhile
    log("idle_10s")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("dali", "main"):
        raise SystemExit("usage: diag_l1_memory.py dali|main <main.py args...>")
    mode, rest = sys.argv[1], sys.argv[2:]
    sys.exit(run_dali(rest) if mode == "dali" else run_main(rest))
