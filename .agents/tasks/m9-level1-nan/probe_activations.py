"""main.py wrapper for the m9 level-1 NaN probe (.agents/tasks/m9-level1-nan/report.md).

Runs ``main.main(argv)`` unchanged, with two optional instruments (env vars):
  PROBE_ACT=1      tf.print per block, to stderr, every forward (and recompute) of every model:
                   'ACT <variable scope> <kind> max <x> min <y>' for the GroupConvBlock conv
                   output ('conv'), the BatchNorm input / output ('bn_in' / 'bn_out') and the SVF
                   head output ('v_raw'). reduce_max / reduce_min only (no float32 copy of the big
                   level-1 tensors); an overflowed float16 tensor shows max inf / min -inf.
  PROBE_NUMERICS=1 tf.debugging.enable_check_numerics() at the start of Trainer.fit (after the
                   model build, checkpoint restore and BN freeze; before the first step's trace):
                   the first op whose float output holds inf / NaN raises InvalidArgumentError
                   naming the op, its dtype and the Python stack that created it.
Usage: python probe_activations.py <main.py args>   (cwd = the source tree to import)
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import tensorflow as tf  # noqa: E402

from models import layers as L  # noqa: E402
from models import group_net as G  # noqa: E402
from runtime import run as R  # noqa: E402


def _label(var):
    name = var.name
    return name.rsplit("/", 1)[0] if "/" in name else name


def _p(tag, kind, y):
    tf.print("ACT", tag, kind, "max", tf.reduce_max(y), "min", tf.reduce_min(y), output_stream=sys.stderr)


if os.environ.get("PROBE_ACT", "0") == "1":
    _conv_call = L.GroupConvBlock.call

    def conv_call(self, x, training=None):
        y = _conv_call(self, x, training=training)
        _p(_label(self.conv.kernel), "conv", y)
        return y

    L.GroupConvBlock.call = conv_call

    _bn_call = L.BatchNormWithSeparateAffine.call

    def bn_call(self, x, training=None):
        y = _bn_call(self, x, training=training)       # builds the BN on the first call
        tag = _label(self.bn.moving_variance)
        _p(tag, "bn_in", x)
        _p(tag, "bn_out", y)
        return y

    L.BatchNormWithSeparateAffine.call = bn_call

    _dec_call = G.DeformDecoder.call

    def dec_call(self, skips, training=None):
        v_raw, dec_feat = _dec_call(self, skips, training=training)
        _p(_label(self.svf_head.kernel), "v_raw", v_raw)
        return v_raw, dec_feat

    G.DeformDecoder.call = dec_call

if os.environ.get("PROBE_NUMERICS", "0") == "1":
    _fit = R.Trainer.fit

    def fit(self, *a, **k):
        print("PROBE: enable_check_numerics at step %d" % int(self.step.numpy()), file=sys.stderr, flush=True)
        tf.debugging.enable_check_numerics(stack_height_limit=60, path_length_limit=120)
        return _fit(self, *a, **k)

    R.Trainer.fit = fit

    # The lean sampler's backward builds a data-dependency constant tf.where(always_true, 0, NaN)
    # (models/layers.py, _DY_DEP): its NaN Const would trip the numerics check although it is never
    # selected. Same function with 1.0 instead of NaN (identical values and op structure).
    # TF 2.10's tf.recompute_grad builds the same kind of constant (custom_gradient.py,
    # 'dresult_dep'), so it is patched the same way and re-exported as tf.recompute_grad.
    import inspect  # noqa: E402
    from tensorflow.python.ops import custom_gradient as _CG  # noqa: E402

    def _unnan(module, name):
        src = inspect.getsource(getattr(module, name))
        if 'float("nan")' not in src:
            raise SystemExit("probe: %s.%s has no NaN constant; update the patch" % (module.__name__, name))
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("@tf_export"))
        exec(compile(src.replace('float("nan")', "1.0"), module.__file__, "exec"), module.__dict__)

    _unnan(L, "_lean_linear_backward")
    _unnan(_CG, "recompute_grad")
    tf.recompute_grad = _CG.recompute_grad

# The probe never writes checkpoints (its run dir is a scratch copy; under the numerics check the
# save would also trip on the best_metric = inf variable).
from runtime import checkpoint as CK  # noqa: E402
CK.RunCheckpoint.save_last = lambda self: None

import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main.main(sys.argv[1:]))
