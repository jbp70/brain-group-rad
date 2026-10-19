"""Training runtime: arguments, losses, metrics, optimizers, LR schedules, checkpoints, logging
and the custom training loop (design sections 5-7 and 10).

Modules:
* `args`: argparse surface of main.py (TF-free, so ``python main.py --help`` needs no TF).
* `schedulers`: LR warmup + cosine / poly / cycle / constant, a pure function of the step.
* `losses`: barycentric (center) and pairwise linear losses in mm^2 (TF).
* `metrics`: rigid/affine errors in mm and degrees, pairwise and cycle errors (numpy float64).
* `optimizers`: SGDW port, optimizer factory, LR get/set helpers (TF).
* `checkpoint`: last/best checkpoint managers and the run config JSON (TF).
* `logging`: dllogger (if importable) or JSON lines, plus TensorBoard scalars.
* `fixed_data`: TF-side data without DALI (phantoms, fixed groups, the validation set).
* `curriculum`: augmentation curriculum, shared global transform G and the control-file writer
  (numpy only).
* `run`: task schedule, `TrainConfig` and the `Trainer` loop (TF).

This package imports nothing on its own; import the submodules you need.
"""
