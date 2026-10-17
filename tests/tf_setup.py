"""Shared TF test setup: quiet logs and GPU memory growth (so tests don't grab a shared GPU)."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402

for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass  # already initialized
