"""Run logging: dllogger (when importable) or a JSON-lines file, plus TensorBoard scalars.

Run directory layout (`make_run_dir`)::

    <results>/<run_name>/
        config.json          model config, run args, versions (runtime.checkpoint.write_config)
        logs/                dllogger.json (dllogger) or metrics.jsonl (fallback)
        tb/                  TensorBoard event files
        ckpt/last, ckpt/best checkpoints (runtime.checkpoint)

Each log line is ``{"step": <int or "PARAMETER">, "time": <unix s>, "data": {...}}``. Values are
converted to plain JSON (numpy / TF scalars -> float or int, non-finite floats -> strings).
"""
import json
import math
import os
import sys
import time

try:  # the container has dllogger; local envs do not
    import dllogger as _dllogger
    DLLOGGER_AVAILABLE = True
except ImportError:
    _dllogger = None
    DLLOGGER_AVAILABLE = False

BACKENDS = ("auto", "dllogger", "jsonl")


def make_run_dir(results, run_name):
    root = os.path.join(results, run_name)
    paths = {"root": root, "logs": os.path.join(root, "logs"), "tb": os.path.join(root, "tb"),
             "ckpt_last": os.path.join(root, "ckpt", "last"),
             "ckpt_best": os.path.join(root, "ckpt", "best"),
             "config": os.path.join(root, "config.json")}
    for k in ("logs", "tb", "ckpt_last", "ckpt_best"):
        os.makedirs(paths[k], exist_ok=True)
    return paths


def to_jsonable(v):
    """numpy / TF / Python values -> JSON-safe values (non-finite floats as strings)."""
    if hasattr(v, "numpy"):
        v = v.numpy()
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        v = v.tolist()
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else str(v)
    if isinstance(v, dict):
        return {str(k): to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_jsonable(x) for x in v]
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)


def _is_number(v):
    if isinstance(v, bool):
        return False
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


class JsonLinesBackend:
    """Append one JSON object per log call to `path`."""

    def __init__(self, path):
        self.path = path
        self._f = open(path, "a")

    def log(self, step, data):
        rec = {"step": step, "time": time.time(), "data": to_jsonable(dict(data))}
        self._f.write(json.dumps(rec) + "\n")

    def flush(self):
        self._f.flush()

    def close(self):
        if not self._f.closed:
            self._f.close()


class _DLLoggerBackend:
    def __init__(self, path, quiet):
        from dllogger import JSONStreamBackend, StdOutBackend, Verbosity
        backends = [JSONStreamBackend(Verbosity.VERBOSE, path)]
        if not quiet:
            backends.append(StdOutBackend(Verbosity.DEFAULT))
        _dllogger.init(backends=backends)
        self.path = path

    def log(self, step, data):
        _dllogger.log(step=step, data=to_jsonable(dict(data)))

    def flush(self):
        _dllogger.flush()

    def close(self):
        _dllogger.flush()


class RunLogger:
    """Metrics logger: one backend (dllogger or JSON lines) plus TensorBoard scalars.

    Args:
      run_dir: the run root (or a `make_run_dir` dict); logs go to <run_dir>/logs, TB to /tb.
      backend: 'auto' (dllogger if importable, else jsonl), 'dllogger' or 'jsonl'.
      tensorboard: write `tf.summary.scalar` for every numeric value.
      quiet: no stdout echo (the jsonl backend prints a compact line per log call otherwise).
    """

    def __init__(self, run_dir, backend="auto", tensorboard=True, quiet=False):
        if backend not in BACKENDS:
            raise ValueError("backend must be one of {}".format(BACKENDS))
        root = run_dir["root"] if isinstance(run_dir, dict) else run_dir
        self.logs_dir = os.path.join(root, "logs")
        self.tb_dir = os.path.join(root, "tb")
        os.makedirs(self.logs_dir, exist_ok=True)
        self.quiet = bool(quiet)
        use_dll = backend == "dllogger" or (backend == "auto" and DLLOGGER_AVAILABLE)
        if use_dll:
            if not DLLOGGER_AVAILABLE:
                raise ImportError("dllogger is not importable; use backend='jsonl'")
            self.metrics_path = os.path.join(self.logs_dir, "dllogger.json")
            self._backend = _DLLoggerBackend(self.metrics_path, self.quiet)
            self.backend_name = "dllogger"
        else:
            self.metrics_path = os.path.join(self.logs_dir, "metrics.jsonl")
            self._backend = JsonLinesBackend(self.metrics_path)
            self.backend_name = "jsonl"
        self._writer = None
        if tensorboard:
            import tensorflow as tf
            os.makedirs(self.tb_dir, exist_ok=True)
            self._writer = tf.summary.create_file_writer(self.tb_dir)

    def _echo(self, step, data):
        if self.quiet or self.backend_name == "dllogger":
            return
        parts = []
        for k, v in data.items():
            parts.append("%s %.5g" % (k, float(v)) if _is_number(v) else "%s %s" % (k, v))
        print("[%s] %s" % (step, " | ".join(parts)), file=sys.stdout, flush=True)

    def log_params(self, params):
        self._backend.log("PARAMETER", params)
        self.flush()

    def log_metrics(self, step, metrics):
        step = int(step)
        self._backend.log(step, metrics)
        self._backend.flush()   # so metrics.jsonl can be read while the run is going
        self._echo(step, metrics)
        if self._writer is not None:
            import tensorflow as tf
            with self._writer.as_default():
                for k, v in metrics.items():
                    if _is_number(v) and math.isfinite(float(v)):
                        tf.summary.scalar(k, float(v), step=step)

    def log_event(self, step, data):
        """Non-scalar records (warnings such as skipped non-finite steps); no TensorBoard."""
        self._backend.log(int(step), data)
        self._backend.flush()
        self._echo(step, data)

    def flush(self):
        self._backend.flush()
        if self._writer is not None:
            self._writer.flush()

    def close(self):
        self.flush()
        self._backend.close()
        if self._writer is not None:
            self._writer.close()
