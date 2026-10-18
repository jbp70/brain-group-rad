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
    # Encoder L2-L6 block type: 'single' conv or 'residual' (two blocks + skip).
    block: str = "single"
    # LeakyReLU slope (0 = ReLU, the rigid project's default).
    negative_slope: float = 0.0
    kernel_size: int = 3
    bn_momentum: float = 0.99
    bn_epsilon: float = 1e-5
    # Unrolled barycenter iterations of the in-graph GroupCentering.
    centering_iters: int = 4
    # Fixed multiplier on the head's 3 translation outputs (mm per unit output). 1 = raw mm.
    translation_scale: float = 1.0
    # Optional seed of the Kaiming initializers (None = TF global seed).
    seed: Optional[int] = None

    def __post_init__(self):
        self.filters = tuple(int(v) for v in self.filters)
        self.reductions = tuple(int(v) for v in self.reductions)
        self.dense = tuple(int(v) for v in self.dense)
        self.validate()

    def validate(self):
        if len(self.filters) != 6:
            raise ValueError("filters needs 6 entries (stem, L2..L6), got {}".format(self.filters))
        if not self.reductions or not self.dense:
            raise ValueError("reductions and dense must be non-empty")
        if self.group_config not in GROUP_CONFIGS:
            raise ValueError("group_config must be one of {}".format(GROUP_CONFIGS))
        if self.block not in BLOCKS:
            raise ValueError("block must be one of {}".format(BLOCKS))
        if self.norm not in CONV_NORMS:
            raise ValueError("norm must be one of {}".format(CONV_NORMS))
        if self.dense_norm not in DENSE_NORMS:
            raise ValueError("dense_norm must be one of {}".format(DENSE_NORMS))
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.centering_iters < 1:
            raise ValueError("centering_iters must be >= 1")
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
