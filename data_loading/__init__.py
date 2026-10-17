"""Data loading: DALI pipelines (rigid / affine), the TF data module and post-DALI processing.

Modules:
* `params`: pure-Python level constants, augmentation defaults, mm -> voxel range conversion and
  the DALI-side matrix formulas (no DALI, no TF; unit-tested locally).
* `dali_loader`: `LinearPipeline` (DALI 1.18). Imports without DALI; using it needs DALI.
* `data_module`: manifest file lists, `DALIDataset` construction, TF memory growth.
* `postprocess`: TF post-DALI processing (matrix conversion, group flips, targets, pairs).

"Level" everywhere means the downsampling factor of the precomputed .npy pyramid: 4 (40x56x48,
4 mm), 2 (80x112x96, 2 mm) or 1 (160x224x192, 1 mm).
"""
