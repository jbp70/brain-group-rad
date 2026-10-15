"""Per-subject provenance (meta.json): atomic writes and step records.

Example:
    meta = load_meta(path)
    record_step(meta, "n4", "done", outputs=[...], commands=[argv], params={}, started=t0, log="logs/n4.log")
    write_meta_atomic(meta, path)
"""
import datetime
import hashlib
import json
import os
from typing import Any, Dict, List, Optional


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def load_meta(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_meta_atomic(meta: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=_default)
    os.replace(tmp, path)


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
    except ImportError:  # pragma: no cover
        pass
    return str(o)


def record_step(meta: Dict[str, Any], name: str, status: str, outputs: Optional[List[str]] = None,
                commands: Optional[List[List[str]]] = None, params: Optional[Dict[str, Any]] = None,
                started: Optional[str] = None, log: Optional[str] = None) -> None:
    meta.setdefault("steps", {})[name] = {
        "status": status,
        "outputs": list(outputs or []),
        "commands": [list(map(str, c)) for c in (commands or [])],
        "params": params or {},
        "started": started,
        "finished": now(),
        "log": log,
    }


def outputs_exist(paths: List[str]) -> bool:
    return bool(paths) and all(os.path.isfile(p) for p in paths)


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
