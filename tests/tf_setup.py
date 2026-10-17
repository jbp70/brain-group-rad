"""Shared TF test setup: quiet logs, TF32 off and GPU memory growth (so tests don't grab a
shared GPU). TF32 must be off on Ampere GPUs (A100): see models/precision.py."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402,F401

from models.precision import configure_tf_precision  # noqa: E402

configure_tf_precision(tf32=False, memory_growth=True)
