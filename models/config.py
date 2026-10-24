"""Model hyperparameters (design sections 4.1-4.4) with documented defaults.

`ModelConfig` is a plain dataclass (Python 3.8 compatible, no TF import) so it can be logged and
rebuilt from a dict (`to_dict` / `from_dict`). Training-loop settings (N per level, task ratio,
loss weights, p_skip, ...) live in the runtime config, not here.
"""
import dataclasses
from typing import Optional, Tuple

GROUP_CONFIGS = ("A", "B")
BLOCKS = ("single", "residual")
CONV_NORMS = ("batch", "instance", "none")
DENSE_NORMS = ("layer", "batch", "none")
HEAD_ACTIVATIONS = ("relu", "none")


@dataclasses.dataclass
class ModelConfig:
    # Encoder widths: stem (L1), L2, L3, L4, L5, L6 (rigid project `init_small`).
    filters: Tuple[int, ...] = (32, 64, 128, 192, 256, 384)
    # 1x1 reductions after L6 (group blocks in both configs; design 4.4).
    reductions: Tuple[int, ...] = (192, 96, 48, 24)
    # Dense widths after flattening the 5x7x6x24 bottleneck.
    dense: Tuple[int, ...] = (2048, 512)
    # Standard dropout before the heads (training only).
    dropout: float = 0.15
    # 'A': group blocks everywhere (stem, encoder, decoders). 'B': only after the bottleneck.
    group_config: str = "A"
    # Group concat at the first dense layer too (doubles it: 10.3 M -> 20.6 M; design v3.2).
    dense_group_concat_first: bool = False
    # Norm of the conv blocks (encoder and 1x1 reductions): 'batch' | 'instance' | 'none'.
    norm: str = "batch"
    # Norm of the dense blocks and the pre-head norm-act: 'layer' | 'batch' | 'none' (design 4.4,
    # v3.4). 'layer' (default): per-vector LayerNorm over the features, independent of the other
    # group members and identical in training and inference. 'none': activation only (the rigid
    # project had no norm between its dense layers). 'batch' is for ablations only: with one
    # group per batch it normalizes over the N group members, so in training mode it is
    # group-relative (subtracts the group mean, divides by the group spread), which the eval-mode
    # moving statistics (pooling between- and within-group variation) cannot reproduce. On the
    # milestone-4 overfit group that gave 0.36 mm RMS in training mode but ~20 mm in eval mode.
    dense_norm: str = "layer"
    # Activation of the pre-head norm-act (after the last dense layer): 'relu' (default; LeakyReLU
    # with `negative_slope`) | 'none' (norm only, so no head-input channel can die; collapse
    # follow-up of milestone 4). The dense blocks always keep their activation.
    head_activation: str = "relu"
    # Encoder L2-L6 block type: 'single' conv or 'residual' (two blocks + skip).
    block: str = "single"
    # LeakyReLU slope. 0.01 (not ReLU) so units cannot die permanently; part of the setup that
    # escaped the identity collapse on real level-4 data (milestone 4, with Adam 3e-4).
    negative_slope: float = 0.01
    kernel_size: int = 3
    bn_momentum: float = 0.99
    bn_epsilon: float = 1e-5
    # Unrolled barycenter iterations of the in-graph GroupCentering.
    centering_iters: int = 4
    # Fixed multiplier on the head's 3 translation outputs (mm per unit output). 1 = raw mm.
    translation_scale: float = 1.0
    # Optional seed of the Kaiming initializers (None = TF global seed).
    seed: Optional[int] = None
    # Build the deformable U-Net decoder + SVF head (design 4.5-4.6, milestone 6). Off by default
    # so linear-only models keep exactly the same variables and checkpoints.
    deform_decoder: bool = False
    # Deform decoder widths D5, D4, D3, D2, D1. None = half the matching encoder widths
    # (f4, f3, f2, f1, f0) // 2 = (128, 96, 64, 32, 16) by default.
    decoder_widths: Optional[Tuple[int, ...]] = None
    # Scaling-and-squaring steps of the SVF integration (0 = use v as the displacement).
    integration_steps: int = 7

    def __post_init__(self):
        self.filters = tuple(int(v) for v in self.filters)
        self.reductions = tuple(int(v) for v in self.reductions)
        self.dense = tuple(int(v) for v in self.dense)
        if self.decoder_widths is not None:
            self.decoder_widths = tuple(int(v) for v in self.decoder_widths)
        self.deform_decoder = bool(self.deform_decoder)
        self.integration_steps = int(self.integration_steps)
        self.validate()

    def resolved_decoder_widths(self):
        """Deform decoder widths D5..D1 (explicit, or half the matching encoder widths)."""
        if self.decoder_widths is not None:
            return tuple(self.decoder_widths)
        f = self.filters
        return (f[4] // 2, f[3] // 2, f[2] // 2, f[1] // 2, f[0] // 2)

    def validate(self):
        if len(self.filters) != 6:
            raise ValueError("filters needs 6 entries (stem, L2..L6), got {}".format(self.filters))
        if not self.reductions or not self.dense:
            raise ValueError("reductions and dense must be non-empty")
        for name in ("filters", "reductions", "dense"):
            vals = getattr(self, name)
            if any(isinstance(v, bool) or int(v) != v or v < 1 for v in vals):
                raise ValueError("{} entries must be integers >= 1, got {}".format(name, vals))
        if self.group_config not in GROUP_CONFIGS:
            raise ValueError("group_config must be one of {}".format(GROUP_CONFIGS))
        if self.block not in BLOCKS:
            raise ValueError("block must be one of {}".format(BLOCKS))
        if self.norm not in CONV_NORMS:
            raise ValueError("norm must be one of {}".format(CONV_NORMS))
        if self.dense_norm not in DENSE_NORMS:
            raise ValueError("dense_norm must be one of {}".format(DENSE_NORMS))
        if self.head_activation not in HEAD_ACTIVATIONS:
            raise ValueError("head_activation must be one of {}".format(HEAD_ACTIVATIONS))
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.centering_iters < 1:
            raise ValueError("centering_iters must be >= 1")
        if self.decoder_widths is not None and (
                len(self.decoder_widths) != 5 or any(w < 1 for w in self.decoder_widths)):
            raise ValueError("decoder_widths needs 5 positive entries (D5..D1), got {}".format(
                self.decoder_widths))
        if self.integration_steps < 0:
            raise ValueError("integration_steps must be >= 0")
        return self

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError("unknown ModelConfig keys: {}".format(sorted(unknown)))
        return cls(**d)

    def replace(self, **changes):
        return dataclasses.replace(self, **changes)
