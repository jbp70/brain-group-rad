"""Standalone groupwise prediction (predict-script; design `.agents/tasks/predict-script/design.md`).

Modules:
* `store`: host stores of per-image arrays (`RamStore`, `DiskStore`), the store-size estimate of
  design section 6 and the RAM / disk choice (TF-free).
* `streaming`: `StreamedLevel`, the exact chunked execution of one `GroupRegLevelModel` (design
  4.2, 4.3, 4.6), plus the per-image element bound and chunk cap.
* `loader`: checkpoint + config.json -> `LoadedModel` (trainable stage under the right dtype
  policy, float32 frozen stages, fingerprint checks; design 3.2, 3.3) and the unchunked
  group-size bound; `LoaderError` (CLI exit 2).
* `runner`: branch orchestration over the cascade levels (design 3.4, 4.4, 4.5):
  `StreamedRunner` (chunked, host store) and `UnchunkedRunner` (one tf.function per level stage).
* `canonical`: internal (A, u, u_inv) -> canonical fwd / inv fields (mm), NIfTI writer, atlas and
  atlas segmentation, the internal consistency check, the --grid-reference check (design 5).
* `cli`: argparse and validation of predict.py (TF-free; design 7), check-streaming chunk defaults.
* `predict`: device setup, `run_predict`, `run_check_streaming`, predict.json, exit codes (design 3.3, 8).
"""
