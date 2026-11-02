# Standalone groupwise prediction script with exact group streaming

The change adds `predict.py` and a new `inference/` package. They restore a trained cascade checkpoint from its run's `config.json` and run one of four branch pipelines (rigid, affine, deform, affine+deform) over the whole subject group at once. Per subject they write canonical mm displacement fields (`--components xyz --signs +++ --units mm`) and a moved image, plus `atlas.nii.gz`, `atlas_seg.nii.gz` and `predict.json`. A full-resolution N = 101 group fits one GPU because a pass-by-pass engine streams every group-mean interaction of the network over chunks: chunk sums are accumulated in float32 and divided by N once. The group-wide matrix steps (linear decoder, centering) run on all N at once. An unchunked mode calls the trainer's own cascade ops and is the reference of a `check-streaming` command and of the SLURM job's gate. The only model change is a refactor that exposes `GroupConvBlock.pre` / `conv_with_group_mean`; a bitwise guard against a verbatim copy of the old `_split_call`, plus unchanged trainer goldens, show it changes no behaviour.

Watch for: the check-streaming JSON `gpu_peak_mb` only covers the last runner phase (confirmed), and NaN can be hidden in the `max_abs_v` / `fp16_max_abs` records (confirmed). Neither affects the fields, the gates or the exit codes. Everything else matches the approved design v5, with documented and justified deviations.

**Verdict**: APPROVED

## High-level view

The streaming engine (`inference/streaming.py`) reproduces `GroupRegLevelModel` as passes P0-P12. Each pass reads the previous block's output per chunk from a host store, applies the block with the group mean collected in the previous pass, and accumulates the next block's mean. Reading it against `models/group_net.py` shows that the stem, the pool2 / pool3 placement, the skip wiring (l2 skip taken before pool3, stem skip before pool2), the decoder's per-stage casts and the SVF centering all follow the model's own call. Every group interaction in inference mode is either a mean that decomposes exactly or a computation on N small tensors. The engine runs the latter on the whole group: the bottleneck goes through `model.linear_decoder`, and `centering` / `compose_linear` run on all N matrices. FEAT-002/003 measured float32 streamed-vs-model differences of ~1e-6 vox at levels 4, 2 and 1 (level 1 on CPU). The level-2 and level-1 unchunked runs match the `Trainer` validation functions bitwise or to 3e-7.

The runner (`inference/runner.py`) chains the levels per branch, with the same op order as `FrozenStage` / `FrozenChain` / `_cascade_val_*_fn`. For affine+deform it resamples the full-res images once with A^-1 and hands both A and the level-1 deform fields to a per-chunk sink. `BRANCHES` and the argparse `type=` validator together give exactly the four supported values. Deform-only works, and any other spelling exits 2 with the required message.

`inference/canonical.py` implements the design 5.1 table: Mv matrices in float64, the affine+deform inverse through border-clamped trilinear sampling of u_inv at Mv(A) y, and the mm scaling. The moved image and the atlas use `evaluation.fields.warp`, and the atlas segmentation uses `evaluation.metrics.MajorityVote`. The internal check ties the files to the model's own TF warp ops through a coordinate volume. Tests showed it detects a wrong conversion (6-57 vox) and a 0.01 vox shift (exit 5). It uses a tighter interior tolerance than the design (1e-6 instead of 1e-5). That deviation is measured, documented in code and leaves the gates unchanged.

The loader restores everything from `config.json`: ModelConfig, level, knobs with defaults for keys absent in old runs, frozen stages built from the same config, a restore with `assert_existing_objects_matched` and fingerprint checks. TF32 is off in `setup_device`, which runs before the first TF op. `predict.py` and `inference/cli.py` import no TF before argument parsing, and the py3.8 AST check covers all new files.

The SLURM script follows design 9. Step 1a is the float32 gate, steps 1b and 1c are non-blocking records, step 2 is `run`, and step 3 runs check-convention and the template-only evaluation. The script never submits jobs itself.

Two further NITs: the store-size estimate ignores the float32 decoder modes, and residual-block checkpoints get exit 1 instead of 2. Neither touches the fields.

<details>
<summary>Issues (4)</summary>

1. **check-streaming GPU peak covers only the last phase** (NIT, confirmed): `_collect_side` resets the peak, but `runner.run`'s `_Phases` resets it again at every stage, so `gpu_peak_mb {unchunked, streamed}` is the peak of the last stage only. Report the max over `summary["gpu_peak_mb"]` values (or drop the inner resets for that call).
2. **NaN hidden in max records** (NIT, confirmed): `max(max_v, float(out["max_v"]))` in `StreamedLevel.deform` and `max(m.get(name, 0.0), float(value))` in `_note` keep the old value when the new one is NaN, so `stats.max_abs_v` and `fp16_max_abs` can read finite for a NaN-producing block. Use a NaN-propagating max (`float(np.maximum(a, b))`, not `np.fmax`) or an explicit `math.isnan` branch.
3. **Store estimate ignores float32 decoder stages** (NIT, confirmed): `estimate_store_gb` gets one trainable dtype ('float16' under AMP), while `deform_decoder_dtype` 'float32' / 'float32_fine' checkpoints (the planned L1 recipe) store d2 in float32. The estimate is low by ~5 GB at N 101. Pass the decoder dtype or the per-stage dtypes into the estimate.
4. **Residual-block checkpoints exit 1, not 2** (NIT, confirmed): `StreamedLevel` raises a plain ValueError for `block != 'single'` inside `runner.run`, which `_command_wrapper` maps to exit 1 with a traceback. Check `model_cfg.block` in the loader / `_run_predict` and raise `PredictError(EXIT_INPUT)` (no current checkpoint uses residual blocks).

</details>

<details><summary>Details</summary>

## Pass schedule against the model's call

The stem uses the split form with only the outer clip (`inner_clip=block.split_group`, False for the stem). That matches the stem's concat path, because `set_split_group_conv` never splits pre-activation-free blocks. Unsplit frozen-stage blocks are evaluated in split form, so streamed and unchunked differ by rounding source (ii) of design 2.3, as intended.

Two deviations are documented and harmless. In the linear task, `x_in` is freed after P2 (design: after P6) because nothing reads it later. Store names carry an `L<level>_` prefix so that the levels can share one store.

P12 integrates with `diffeo.exp_and_inverse`, while the m9 trainer ran `gradient_checkpointing full` (`integrate_svf` on v and -v). `exp_and_inverse` is literally `(integrate_svf(v), integrate_svf(-v))`, so building with `'none'` gives the same forward values.

## Reporting gaps in check-streaming and the stats records

```
_collect_side:  _gpu_reset()            # peak := 0
                runner.run(...)
                  _Phases("pool")       # reset_memory_stats again
                  _Phases("deform:frozen")  # reset
                  _Phases("deform:l1")      # reset  <- only this one survives
                _gpu_peak_mb(active)
```

For deform-only and affine+deform the last phase is `deform:l1`, which is also the largest. The AC-7 number for the level-1 unchunked reference is therefore close in practice, but the field is mislabelled as a whole-side peak (confirmed).

The `max_abs_v` and `fp16_max_abs` maxima are kept with Python `max`, which drops a NaN in the second argument. Because the m9 run died from fp16 overflow, these records exist to diagnose exactly that, so they should not read finite when a block produced NaN. Inf survives `max`. Non-finite fields still trigger exit 3, so outputs are never silently accepted (confirmed).

## Store estimate and residual blocks

`_run_predict` passes `tdt = "float16"` whenever the trainable stage is mixed_float16. `StreamedLevel._put` stores tensors in their real dtype, so d2 (float32 under 'float32_fine') takes twice the estimated bytes. For m9 (`policy`) the estimate is exact. For the retrain recipe (L1 float32_fine) the shortfall is a few GB, well below the 0.7 x 96 GB RAM threshold, so it matters only near the boundary (confirmed).

A `block='residual'` config passes the loader and fails inside the runner as an unmapped ValueError (exit 1). The design wanted it to fail fast; the CLI contract maps config problems to exit 2 (confirmed).

## Test coverage

The tests cover all of AC 1-6: refactor guard, per-level streamed vs model, cascade streamed vs unchunked for every branch at levels 4 and 2 with chunks 1/2/N, level 1 on CPU, trainer equivalence at levels 2 and 1, the convention evidence with check-convention and `evaluation run --pairs 2`, the branch compositions, the CLI, grid-reference, nonfinite and internal-mismatch exits, and the CPU subprocess (which ran, not skipped). The full suite passed (802 tests, status 0).

Not tested: the fp16 bound set on large fields (FEAT-002 showed S_fp16 fails for |u| ~4 vox on TINY; the real-checkpoint fp16 check 1b is non-blocking by design), `--store disk` through the CLI end to end (DiskStore is unit-tested and used by one engine test), and the real default-width level-1 path, which only the cluster job (step 1a) exercises.

## Unrelated commits in the range

`95e58c8..HEAD` also contains baseline work (BrainMorph, SynthMorph, ANTs wrappers and their SLURM scripts: f41593a, 4a2cac2, 58bc538, a056241, d67a791, 68c42e8, c872f77, f84b0cf). Those commits are separate tasks and were not reviewed here. The predict-script commits touch none of their files. There are no uncommitted changes.

</details>

<details>
<summary>File map</summary>

- `predict.py`: thin CLI entry, parses before importing TF.
- `inference/cli.py`: argparse, validation, exit codes, `resolve_check_chunks`.
- `inference/loader.py`: config.json -> built and restored cascade, precision mapping, fingerprints, unchunked n_max.
- `inference/store.py`: RamStore / DiskStore, level geometry, store estimate, auto choice.
- `inference/streaming.py`: `StreamedLevel` pass engine, element bound, chunk cap.
- `inference/runner.py`: `BRANCHES`, `StreamedRunner`, `UnchunkedRunner`, phase timing / GPU peaks.
- `inference/canonical.py`: canonical conversion, internal check, grid reference, NIfTI writer, atlas.
- `inference/predict.py`: device setup, `run` / `check-streaming`, predict.json, git commit.
- `models/layers.py`: `GroupConvBlock.pre` / `conv_with_group_mean` refactor.
- `slurm/predict_mindboggle.sh`: cluster job (gate, records, run, template eval).
- `tests/test_predict*.py`, `tests/predict_helpers.py`, `tests/test_slurm_scripts.py`, `tests/run_tests_wsl.sh`: tests and suite wiring.

Full diff: `git diff 95e58c8..HEAD -- predict.py inference models tests slurm/predict_mindboggle.sh`.

</details>
