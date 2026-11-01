"""Standalone groupwise prediction (predict-script; design `.agents/tasks/predict-script/design.md`).

Modules:
* `store`: host stores of per-image arrays (`RamStore`, `DiskStore`), the store-size estimate of
  design section 6 and the RAM / disk choice (TF-free).
* `streaming`: `StreamedLevel`, the exact chunked execution of one `GroupRegLevelModel` (design
  4.2, 4.3, 4.6), plus the per-image element bound and chunk cap.
"""
