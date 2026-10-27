"""Shared TF test setup: quiet logs, TF32 off and GPU memory growth (so tests don't grab a
shared GPU). TF32 must be off on Ampere GPUs (A100): see models/precision.py.

``BGR_TF_GPU_LIMIT_MB`` (optional, set by tests/run_tests_wsl.sh for the local 8 GB GTX 1080
under WSL): use a fixed logical-device memory limit instead of memory growth. Under WSL, letting
TF grow into the last GB of the card pages GPU memory to host RAM and full-width steps slow down
from ~1 s to minutes (milestone-4 report). Unset on the cluster / glacier.

``BGR_DEFER_MEMORY_TESTS=1`` (set by tests/run_tests_wsl.sh for its per-module runs): the GPU
memory-measurement classes (``DEFER_MEMORY_TESTS`` skip) are skipped there and run afterwards,
each in a fresh process. Their peaks depend on the allocator state left by earlier tests in the
process: in one ~640-test process the level-2 validation-memory test OOMed at the 6000 MB limit
and the pair-recompute growth bound failed (also after only test_postprocess in the same
process), while both pass in a fresh process.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402,F401
from models.precision import configure_tf_precision  # noqa: E402

_LIMIT_MB = int(os.environ.get("BGR_TF_GPU_LIMIT_MB", "0") or 0)
if _LIMIT_MB > 0:
    configure_tf_precision(tf32=False, memory_growth=False)
    for _gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.set_logical_device_configuration(
                _gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=_LIMIT_MB)])
        except RuntimeError:
            pass  # already initialized
else:
    configure_tf_precision(tf32=False, memory_growth=True)

DEFER_MEMORY_TESTS = os.environ.get("BGR_DEFER_MEMORY_TESTS", "") == "1"
DEFER_MEMORY_REASON = "GPU memory measurement: deferred to a fresh process (BGR_DEFER_MEMORY_TESTS=1)"
