"""Named width presets and a TF-free trainable-parameter count (milestone 9, design 3.1).

`.agents/tasks/milestone-9/design-m9-phase1.md` section 3.1. A preset expands to explicit
``--filters`` / ``--decoder-widths`` values; it is not a `ModelConfig` field, so ``config.json``
stores the explicit lists and the cascade config check applies unchanged. Reductions
(192, 96, 48, 24) and dense (2048, 512) stay at their defaults, so the bottleneck stays
5x7x6x24 and the parameter count stays dominated by the dense layer and the coarse stages.

| Name | filters (stem, L2..L6) | decoder (D5..D1) |
|---|---|---|
| default | (32, 64, 128, 192, 256, 384) | (128, 96, 64, 32, 16) |
| u075 / u050 / u0375 | `uniform(m)` of the default widths | `uniform(m)` of the default decoder |
| f0d1 | (16, 64, 128, 192, 256, 384) | (128, 96, 64, 32, 8) |
| fr50 | (16, 32, 128, 192, 256, 384) | (128, 96, 64, 16, 8) |
| fr25 | (8, 16, 128, 192, 256, 384) | (128, 96, 64, 8, 8) |

Every preset lists BOTH widths explicitly. Note: the explicit ``default`` preset
(``decoder_widths=(128, 96, 64, 32, 16)``) builds the same model as ``ModelConfig()`` but is
not dict-equal to it (``ModelConfig().decoder_widths`` is None = half the encoder widths).
Presets are for new runs only; old configs are untouched.

m11 grid widths (fp32-decoder-width memory job; flag lines, not presets, so `PRESETS` and the
memory-model grid tests are unchanged):

* default: no width flags (``ModelConfig`` default, decoder_widths None);
* d12h (d1/d2 halved): ``--decoder-widths 128,96,64,16,8``;
* d12hs (d1/d2 + stem halved): ``--filters 16,64,128,192,256,384 --decoder-widths 128,96,64,16,8``;
* fr50: the ``fr50`` preset, ``--filters 16,32,128,192,256,384 --decoder-widths 128,96,64,16,8``.

A non-default width at levels 2 / 1 cascades through the normal ``--cascade-from`` path from
random-weight level-4 / level-2 sources built at the same width (slurm/m11_random_sources.sh),
so the frozen stages are narrow too.

`count_params` uses the architecture of `models.group_net` (same rules as the memory model,
design 4.2) and is checked against built models by `tests/test_width_presets_tf.py` (1 %).
No TensorFlow import.
"""
import collections
import math

from models.config import ModelConfig

DEFAULT_FILTERS = (32, 64, 128, 192, 256, 384)
DEFAULT_DECODER = (128, 96, 64, 32, 16)
BOTTLENECK_VOXELS = 5 * 7 * 6
RIGID_HEAD_UNITS = 13
SCALE_SHEAR_HEAD_UNITS = 6


def uniform_width(w, m):
    """One width scaled by m, rounded to a multiple of 8 (fp16 tensor cores), at least 8."""
    return max(8, 8 * int(math.floor(float(w) * float(m) / 8.0 + 0.5)))


def uniform(m):
    """Uniform preset dict(filters, decoder_widths): every default width w ->
    max(8, 8 * floor(w * m / 8 + 0.5))."""
    return {"filters": tuple(uniform_width(w, m) for w in DEFAULT_FILTERS),
            "decoder_widths": tuple(uniform_width(w, m) for w in DEFAULT_DECODER)}


PRESETS = collections.OrderedDict([
    ("default", {"filters": DEFAULT_FILTERS, "decoder_widths": DEFAULT_DECODER}),
    ("u075", {"filters": (24, 48, 96, 144, 192, 288), "decoder_widths": (96, 72, 48, 24, 16)}),
    ("u050", {"filters": (16, 32, 64, 96, 128, 192), "decoder_widths": (64, 48, 32, 16, 8)}),
    ("u0375", {"filters": (16, 24, 48, 72, 96, 144), "decoder_widths": (48, 40, 24, 16, 8)}),
    ("f0d1", {"filters": (16, 64, 128, 192, 256, 384), "decoder_widths": (128, 96, 64, 32, 8)}),
    ("fr50", {"filters": (16, 32, 128, 192, 256, 384), "decoder_widths": (128, 96, 64, 16, 8)}),
    ("fr25", {"filters": (8, 16, 128, 192, 256, 384), "decoder_widths": (128, 96, 64, 8, 8)}),
])
UNIFORM_FACTORS = {"u075": 0.75, "u050": 0.5, "u0375": 0.375}


def _preset(name):
    if name not in PRESETS:
        raise ValueError("unknown width preset {!r} (presets: {})".format(name, ", ".join(PRESETS)))
    return PRESETS[name]


def preset_kwargs(name):
    """ModelConfig kwargs dict(filters=..., decoder_widths=...) of a preset (ValueError if unknown)."""
    p = _preset(name)
    return {"filters": tuple(p["filters"]), "decoder_widths": tuple(p["decoder_widths"])}


def preset_args(name):
    """CLI arguments of a preset: ['--filters', 'a,b,...', '--decoder-widths', 'a,b,...']."""
    p = _preset(name)
    return ["--filters", ",".join(str(w) for w in p["filters"]),
            "--decoder-widths", ",".join(str(w) for w in p["decoder_widths"])]


def _as_config(model_config):
    if isinstance(model_config, ModelConfig):
        return model_config
    if isinstance(model_config, str):
        return ModelConfig(**preset_kwargs(model_config))
    return ModelConfig(**dict(model_config))


def _norm_params(norm, channels):
    """Trainable parameters of a pre-activation norm (gamma, beta per channel; 'none' 0)."""
    return 0 if norm == "none" else 2 * int(channels)


def _conv(cin, cout, k):
    return int(k) ** 3 * int(cin) * int(cout) + int(cout)


def _group_block(cin, cout, k, group, norm, pre_activation=True):
    """GroupConvBlock: [norm affine on cin] + Conv3D(k) on cin (x2 with the group concat)."""
    n = _norm_params(norm, cin) if pre_activation else 0
    return n + _conv(cin * (2 if group else 1), cout, k)


def _encoder_block(cfg, cin, cout, group):
    if cfg.block == "residual":
        n = (_group_block(cin, cout, cfg.kernel_size, group, cfg.norm)
             + _group_block(cout, cout, cfg.kernel_size, group, cfg.norm))
        return n  # projection added by the caller (needs the stride)
    return _group_block(cin, cout, cfg.kernel_size, group, cfg.norm)


def count_params(model_config, deform_decoder=None):
    """Trainable parameter counts per part of a `GroupRegLevelModel` (level-independent).

    Args:
      model_config: a ModelConfig, a ModelConfig kwargs dict (missing keys = defaults) or a
        preset name.
      deform_decoder: count the deform decoder (None = the config's ``deform_decoder`` flag).

    Returns OrderedDict(encoder, linear_decoder, deform_decoder, total) (deform_decoder 0 when
    not counted).
    """
    cfg = _as_config(model_config)
    with_deform = cfg.deform_decoder if deform_decoder is None else bool(deform_decoder)
    g = cfg.group_config == "A"
    f = cfg.filters
    k = cfg.kernel_size
    # Encoder: stem (no pre-activation) then L2..L6 (L4-L6 stride 2).
    enc = _group_block(1, f[0], k, g, cfg.norm, pre_activation=False)
    strides = (1, 1, 2, 2, 2)
    for i in range(1, 6):
        enc += _encoder_block(cfg, f[i - 1], f[i], g)
        if cfg.block == "residual" and (f[i - 1] != f[i] or strides[i - 1] != 1):
            enc += _conv(f[i - 1], f[i], 1)
    # Linear decoder: 1x1 group reductions, flatten, dense blocks, pre-head norm, two heads.
    lin = 0
    cin = f[5]
    for r in cfg.reductions:
        lin += _group_block(cin, r, 1, True, cfg.norm)
        cin = r
    d_in = BOTTLENECK_VOXELS * cin
    for i, units in enumerate(cfg.dense):
        group = i > 0 or cfg.dense_group_concat_first
        lin += _norm_params(cfg.dense_norm, d_in) + d_in * (2 if group else 1) * units + units
        d_in = units
    lin += _norm_params(cfg.dense_norm, d_in)
    lin += d_in * RIGID_HEAD_UNITS + RIGID_HEAD_UNITS + d_in * SCALE_SHEAR_HEAD_UNITS + SCALE_SHEAR_HEAD_UNITS
    # Deform decoder: D5..D1 group blocks on concat(upsampled, skip), SVF head.
    dfm = 0
    if with_deform:
        widths = cfg.resolved_decoder_widths()
        skips = (f[4], f[3], f[2], f[1], f[0])
        x = f[5]
        for w, s in zip(widths, skips):
            dfm += _group_block(x + s, w, k, True, cfg.norm)
            x = w
        dfm += _norm_params(cfg.norm, x) + _conv(x, 3, 3)
    out = collections.OrderedDict()
    out["encoder"] = int(enc)
    out["linear_decoder"] = int(lin)
    out["deform_decoder"] = int(dfm)
    out["total"] = int(enc + lin + dfm)
    return out
