"""Process-wide TF numerics setup. Call `configure_tf_precision()` once, before building models.

Why TF32 must be off: on Ampere GPUs (A100, RTX A6000/A40), TF 2.10 runs float32 matmul,
einsum and convolutions in TensorFloat-32 by default, i.e. with a 10-bit mantissa (about 3
decimal digits). That silently degrades every float32 computation that must be exact:
sampling-grid coordinates in `models.layers.transform_image`, 4x4 transform composition and
inversion, the group barycenter (`logm_real`/`expm`), targets and the point loss. On glacier
(A100) this produced ~1e-3 errors in warped intensities and failed the spatial-transformer
tests, which all pass with TF32 disabled.

Turning TF32 off costs little for training because the model uses the `mixed_float16`
policy: convolutions then run in float16 on tensor cores regardless of the TF32 setting.
Pre-Ampere GPUs (Quadro RTX 8000/6000, V100, GTX 1080) have no TF32, so this is a no-op there.
"""
import tensorflow as tf


def configure_tf_precision(tf32=False, memory_growth=True):
    """Disable TF32 (default) and enable GPU memory growth for every visible GPU.

    Must run before the first GPU op (memory growth cannot change after initialization; a
    RuntimeError there is ignored). Returns the resulting TF32 state.
    """
    tf.config.experimental.enable_tensor_float_32_execution(bool(tf32))
    if memory_growth:
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass  # already initialized
    return tf.config.experimental.tensor_float_32_execution_enabled()
