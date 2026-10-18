"""Learning-rate schedules as pure functions of the global step (no TF, no state).

Ported from the rigid project's ``runtime/schedulers.py`` (``CustomLearningRateSchedules`` and
``PolynomialDecayWithWarmup``), rewritten statelessly: ``lr_at(step, cfg)`` depends only on the
step, so a resumed run continues the schedule exactly and the train step never retraces (the LR
is assigned to the optimizer's variable from Python every step).

Shape (``warmup_steps`` W, ``total_steps`` T, base LR b, floor m = b * min_lr_factor):
* warmup (step < W): linear from b / 1e8 (the reference's warmup floor) to b at step W;
* then, with progress p = clip((step - W) / (T - W), 0, 1):
  - ``cosine``: m + 0.5 (b - m) (1 + cos(pi p)), reaching m at step T;
  - ``poly``: m + (b - m) (1 - p)^power (reference power 0.9 in args), reaching m at step T;
  - ``cycle``: the reference cyclic cosine: the first cycle lasts ``cycle_steps`` steps, every
    next cycle is ``cycle_mult`` (1.5) times longer, and both peak and floor drop by
    ``cycle_decay`` (10) per cycle (restart at each boundary);
  - ``constant``: b.
"""
import math
from dataclasses import dataclass
from typing import Optional

SCHEDULES = ("cosine", "poly", "cycle", "constant")
WARMUP_FLOOR_FACTOR = 1e-8


@dataclass
class ScheduleConfig:
    name: str = "cosine"
    base_lr: float = 0.01
    total_steps: int = 50000
    warmup_steps: int = 1000
    min_lr_factor: float = 1e-3
    power: float = 0.9
    cycle_steps: Optional[int] = None     # None -> (total_steps - warmup_steps) / 4
    cycle_mult: float = 1.5
    cycle_decay: float = 10.0

    def __post_init__(self):
        if self.name not in SCHEDULES:
            raise ValueError("schedule must be one of {}, got {!r}".format(SCHEDULES, self.name))
        if self.base_lr < 0 or self.warmup_steps < 0 or self.total_steps < 1:
            raise ValueError("need base_lr >= 0, warmup_steps >= 0 and total_steps >= 1")


def _cosine(p, peak, floor):
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * p))


def lr_at(step, cfg):
    """Learning rate at global step `step` (0-based) for a `ScheduleConfig`."""
    step = float(max(0, int(step)))
    base = float(cfg.base_lr)
    warm = int(cfg.warmup_steps)
    if warm > 0 and step < warm:
        lo = base * WARMUP_FLOOR_FACTOR
        return lo + (base - lo) * step / warm
    if cfg.name == "constant":
        return base
    floor = base * float(cfg.min_lr_factor)
    span = max(1.0, float(cfg.total_steps - warm))
    s = step - warm
    if cfg.name == "cosine":
        return _cosine(min(max(s / span, 0.0), 1.0), base, floor)
    if cfg.name == "poly":
        p = min(max(s / span, 0.0), 1.0)
        return floor + (base - floor) * (1.0 - p) ** float(cfg.power)
    # cycle
    length = float(cfg.cycle_steps) if cfg.cycle_steps else max(1.0, span / 4.0)
    peak = base
    while s >= length:
        s -= length
        length *= float(cfg.cycle_mult)
        peak /= float(cfg.cycle_decay)
        floor /= float(cfg.cycle_decay)
    return _cosine(s / length, peak, floor)
