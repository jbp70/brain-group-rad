# m9 level-1 OOM: why the real deform step OOMs when the memory tool measured 31 GB

Read-only investigation of commit `abc9d96`. No code changed. No cluster access. No local GPU run (reason in section 4).
New files, for the orchestrator to copy into the repo: `diag_l1_memory.py` and `m9_oom_diag.sh`, both in this folder.

## 1. Summary

- **The training graph is the same.** The deform step that runs in the real Trainer is the same traced graph that `measure_level_memory.py` measured. The tool builds a real `Trainer` and calls `Trainer.train_step`, so it shares the postprocess, frozen-chain branch, pre-warp, `compose_cascade`, losses, diagnostics, AMP and optimizer. `FrozenChain` and the stand-in run identical math. Every memory flag reaches the trained model. The few TF-side extras in the real run (task-grad-norm scaling, global clip, two more traced linear steps, optimizer slots) add up to less than 0.5 GB. So nothing inside the step explains a gap of 13+ GB.
- **The difference is process-level and the tool cannot see it.**
  1. **DALI GPU memory.** DALI allocates its own GPU memory, outside TF's allocator, and nothing measured or modelled it. The memory tool uses no DALI, and `models/memory_model.py` budgets 44 GiB with no DALI term. At level 1, the three pipelines (rigid N 8, affine N 8, deform N 4) with the default intensity augmentation are estimated at **6-8 GiB**. About 5 GiB of that comes from the augmentation ops: each GPU op keeps a full batch-sized output buffer.
  2. **Fragmentation under memory growth.** With `set_memory_growth`, TF's allocator grows in many separate regions of doubling size. Building the real run creates transient GPU tensors first (validation-set rendering at level 1, model builds, frozen-chain restore). A chunk cannot span two regions, so a new 3.3 GiB request (`[4,160,224,192,64]` fp16) needs fresh device memory, which DALI now shares. The tool's 31.0 GB is the in-use peak (`get_memory_info` peak). It is not the reserved arena and it excludes the first, autotuning step.
- **Budget arithmetic for RTX 8000** (45.0 GiB):
  - CUDA context and handles ~0.5 GiB, TF in-use 31 GiB and DALI 6-8 GiB together leave only ~5-7 GiB for fragmentation and first-step extras. That is not enough.
  - With 4 pairs (+4-7 GiB predicted) it is clearly over.
  - D1 failed at step 0, before any linear step. That rules out leftovers from BN recalibration, validation and linear steps.
- **D2 is the strongest evidence.** With seed 0 and task ratio 1:1:2, the task draw is `D r D r r D D a ...`, so step 0 is deform. D2 logged the non-finite rigid step as step 2 (`done = s + 1`, s = 1), so **its step-0 deform step completed** under `cuda_malloc_async`. That allocator has no region fragmentation. So the step's real in-use memory fits beside DALI, and the D1 OOM comes from how the allocator lays out memory, not from what the step needs.
- **Likely D2 stall.** TF's async pool keeps freed memory (its release threshold is the whole TF limit), so DALI's next prefetch can't get device memory. DALI's worker or `GetNext` then blocks, and GPU utilisation drops. An alternative is extreme slowness from allocation retries with stream syncs. This is unverified; run R5 of the diagnostic tells the two apart.
- **Recommended fix order:**
  1. A fixed TF memory limit (one BFC region, no growth) via the existing `--gpu-memory-limit-mb`, sized from the measured DALI footprint.
  2. A smaller DALI footprint (move or fuse the intensity augmentation).
  3. Only then the in-graph savings (fp16 one-hot).

## 2. Evidence: the real Trainer step vs the measurement tool

### 2.1 Same step function

- `tests/gpu/measure_level_memory.py::measure` builds `Trainer(model, O.make_optimizer("adam", lr=3e-4, amp=...), {task: fd.FixedGroupSource(...)}, S, cfg, cascade=frozen)` and calls `trainer.train_step(task, batch)`. That is the same `Trainer._make_deform_step` body (`runtime/run.py`) used by `fit`.
- Inside that body, the following all run inside ONE `tf.function`, in both paths:
  - postprocess `pp(images, labels)`: flips plus `tf.one_hot(labels, 19, float32)`;
  - the frozen-chain branch (`chain = self._n_stages == 2`, `cascade.deform(...)`);
  - `CS.prewarp_deform`;
  - the tape: `model(x_in, task="deform", training=True)`, `diffeo.compose_cascade(..., lean=True)` and `losses.deform_loss(..., lean_sampler=True[, lean_losses])`;
  - `deform_diagnostics`, `_enc_grad_diagnostics`, LSO unscale and `tf.cond(apply)`.
- So the postprocess outputs are not "kept alive" across steps. Inside the step, the one-hot must stay alive through the backward pass anyway, because the Dice gradient needs it. That holds in both paths.

### 2.2 Frozen chain: real vs stand-in

`models/cascade.py::FrozenChain._deform` / `_linear` and `tests/gpu/frozen_chain_standin.py::FrozenChainStandIn.deform` / `linear` match line by line:

- pool to level 2;
- `stage4.deform(x2)`;
- `model2(prewarp_deform(x2, U4))` with `training=False`;
- `compose_cascade` at level 2;
- `upsample_displacement` to level 1;
- `stop_gradient`.

Both build their models with `_float32_policy()`, checkpointing `'none'`, `build_model(2)`. The only differences are where the weights come from (checkpoint vs random) and the fingerprint checks. `FrozenChain.from_checkpoint` restores model variables only (`expect_partial`); no optimizer slots are created.

### 2.3 Flag plumbing (main.py -> trained model)

- `main.py` builds `GroupRegLevelModel(level, model_cfg, gradient_checkpointing=args.gradient_checkpointing, lean_sampler=args.lean_sampler, split_group_conv=args.split_group_conv).build_model(_build_n(args))`. `_build_n` is 2 with split.
- `runtime/args.py::train_config_from_args` passes `lean_sampler`, `lean_losses` and `min_pairs` to `TrainConfig`. `Trainer.__init__` raises if the model's `lean_sampler` differs from the config's.
- `models/group_net.py::_call_deform` takes `_call_deform_full` when `mode == "full" and encoder.built and decoder.built`. `build_model` runs a rigid and a deform forward, so both parts are built and the full recompute path is active in training.
- `split_group_conv` is applied in `__init__` (`L.set_split_group_conv(self, True)`), so it is active in every later call.
- The frozen models deliberately use checkpointing `'none'` and the default sampler. They are inference-only, outside the tape.

### 2.4 TF-side extras of the real run (all small)

| Item | Real run | Tool | Size at level 1 |
|---|---|---|---|
| Trainable params (29.6 M), fp32 | yes | yes | 0.12 GB |
| Adam m, v for ALL variables (`O.build_optimizer` in `__init__`) | yes | yes | 0.24 GB |
| `--task-grad-norm 12`: `scale_gradients` copy | yes | no (tool: `task_grad_norm` 0) | 0.12 GB transient |
| Global clip 12 (`make_optimizer(clipnorm=...)`) | yes | no (tool: no clipnorm) | 0.12 GB transient |
| BN backup / restore (`tf.identity` of BN statistics) | yes | yes | < 0.01 GB |
| Frozen chain weights (2 x 29.6 M fp32) | yes | yes | 0.24 GB |
| Rigid / affine traced steps | yes | no | no persistent tensors (in D1 not even traced before the OOM) |
| LossScaleOptimizer | yes | yes | scalars |
| Validation sets | numpy on the host | none | 0 resident, but rendered on GPU at startup (`fixed_data.render_group`, chunk 8 at level 1) -> transient arena growth |

Total persistent difference: under 0.5 GB.

### 2.5 What the 31.0 GB number is

- `peak_mb` = `get_memory_info('GPU:0')['peak']`, i.e. the BFC bytes in use. It is measured after `reset_memory_stats` over the 3 measured steps (job 10516237 ran `--warmup-steps 1 --steps 3`).
- It excludes the allocator's reserved-but-free space (fragmentation), DALI, and the CUDA context.
- It also excludes the first step (trace plus cuDNN autotuning, with up to a 4 GiB workspace per conv by default). That first step is stored separately as `warmup_peak_mb`.
- Every real failure happened at the FIRST deform step. Check `warmup_peak_mb` of the level-1 default N 4 record in `scratch/m9/memory_m9.json` (command in section 5.1).

### 2.6 DALI device memory (question 2)

`data_loading/data_module.py::make_dali_dataset` wraps each pipeline in a `DALIDataset` on `/gpu:0`. DALI allocates its operator buffers from its own device pool (cudaMalloc), outside TF's BFC allocator. That pool grows and is not returned during the run. The output tensors are copied into TF-allocated tensors (inside BFC).

The pipelines run `exec_async=True` and `exec_pipelined=True` with `prefetch_queue_depth` 1 (`--prefetch 1`). Each pipeline therefore computes its next batch in the background, so its buffers stay allocated all the time.

Estimate per pipeline. At level 1 one float32 volume is 6,881,280 voxels = 26.25 MiB, so an N 8 batch is 210 MiB and an N 4 batch is 105 MiB.

- **`LinearPipeline.define_graph`** (rigid, affine), with `IntensityAugConfig` all on:
  - `.gpu()` copy;
  - `warp_affine`;
  - `noise.gaussian`;
  - brightness multiply;
  - the contrast expression (one fused arithmetic op);
  - resize NN plus resize LINEAR (plus resampling scratch);
  - the two gamma expressions;
  - the final clamp.

  That is about 11-12 batch-sized GPU buffers, ~2.4-2.6 GiB per linear pipeline.
- **`DeformPipeline`** (no residual affine): images `.gpu()` plus the same augmentation chain is ~10 x 105 MiB, plus labels 26 MiB, ~1.1 GiB.
- **Total:** ~6 GiB, or 6-8 GiB with pool rounding and scratch. With `--intensity-aug false` it falls to ~1 GiB (copy plus warp only). At level 2 the same pipelines are 8x smaller per volume (~1.5-2 GiB in total), which is why level 2 never showed the problem.

Interaction with TF:

- With memory growth, TF's BFC limit is fixed when the device is created. On cc 7.5 that is roughly the free memory minus 1050 MiB, ~43.6 GiB. TF then grows by `cudaMalloc` as needed.
- DALI competes for the same physical memory. TF fails as soon as a region `Extend` can't get device memory, even though its own limit isn't reached. The OOM message's `Limit` vs `InUse` shows this directly.
- With a fixed limit and no growth, BFC allocates ONE region of the limit at the first allocation, and DALI gets whatever is left. If that is too little, DALI fails with a clear error.

### 2.7 Fragmentation under growth

The BFC with growth starts with a 2 MiB region and doubles the region size on each `Extend`. Near the end of free memory it backs off by 0.9x per attempt until an allocation succeeds.

Before the first train step, the real run has already made a series of differently sized transient allocations:

- `render_group` of 4 linear validation sets (`rigid`, `affine`, `rigid_g`, `affine_g`) at level 1, in chunks of 8 (images, grid, output, ~1 GiB each);
- three model builds;
- the checkpoint restore.

Small persistent tensors (weights, Adam slots, BN statistics, the step variable, captured constants) sit in those early regions. They stop the regions from being freed by the allocator's garbage collection, which only frees regions that are completely free. The holes left inside them can't hold 3.3 GiB chunks.

The memory tool's process does far less before its first step (synthetic group, deform-only trainer). It also runs without DALI.

### 2.8 D1 / D2 timeline (seed 0, ratio rigid:1,affine:1,deform:2)

`TaskSchedule.draw(s) = Random(0 * 1000003 + s).random()` with cumulative thresholds 0.25 (rigid), 0.5 (affine), 1.0 (deform). That gives, for s = 0..9: `D r D r r D D a r a`.

- **D1:** OOM at s = 0 (the first train step, ~3 min after start). The process had no BN recalibration, no validation and no linear steps before it.
- **D2:** the `nonfinite_step_skipped` event at step 2 is the rigid step s = 1. So the deform step s = 0 completed with `cuda_malloc_async`. Then nothing appeared for 38 min. With `--log-every 25`, a quiet stdout for the first ~5-8 min is normal (24 steps at ~16 s deform / ~4 s linear, plus 2 traces). 38 min is not.

## 3. Why D2 (`cuda_malloc_async`) may stall (question 3, hypotheses)

1. **Memory contention between TF's async pool and DALI (most likely).** TF 2.10's `GpuCudaMallocAsyncAllocator` sets the pool release threshold to its memory limit, so memory freed after step 0 stays in TF's pool. DALI's pool allocates with plain `cudaMalloc`. It can't use that cached memory, and its next background run (the deform prefetch for s = 2, scheduled right after fetch 1) can fail or wait. The trainer then blocks in `next(source)` or the step with ~0 % GPU utilisation. This fits "GPU utilisation looks off".
2. **Extreme slowness near full memory.** On allocation failure the async allocator synchronises the stream and retries. cuDNN autotuning of the not-yet-traced deform-step convs (s = 2 reuses the trace, but the rigid step at s = 1 was just traced and autotuned) and multi-GB allocations then serialise the GPU. Possible, but 38 min with no event is long.
3. Less likely: the DALI TF plugin writes its outputs into TF-allocated memory on DALI's own stream. With a stream-ordered allocator, that adds cross-stream synchronisation; it can slow things down but should not hang.

Run R5 in section 5 records the phase (`fetch` vs `step_begin` without `step_end`) and the GPU utilisation every second, which tells 1 from 2.

## 4. Local reproduction: not done

The tool already drives the real `Trainer.train_step`, so a local tiny-width Trainer-vs-tool comparison would compare a function with itself. The gap comes from DALI (not installed in the local `tf_gpu` env) and from level-1 allocator layout on a 48 GB card. Neither can be reproduced under the 6000 MB local limit. The cluster diagnostic below measures both directly.

## 5. Diagnostic plan

### 5.1 Free checks first (login node: reading logs and JSON only)

```bash
P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
# BFC stats at the D1 OOM: Limit vs InUse vs Reserved vs LargestFreeBlock (fragmentation)
grep -hE 'Limit:|InUse:|MaxInUse:|Reserved:|PeakReserved:|LargestFreeBlock:|Sum Total of in-use|total_region_allocated|Created device|ran out of memory' $P/logs/*-10517336.err | head -40
# first-step (warm-up) peak of the measured level-1 config
python3 -c "import json;d=json.load(open('$P/scratch/m9/memory_m9.json'));[print(r['widths'],r['n'],r['min_pairs'],r['lean_losses'],r.get('warmup_peak_mb'),r.get('peak_mb'),r.get('resident_mb')) for r in d['results'] if r['level']==1 and r['task']=='deform' and r['status']=='ok']"
# D2: what was the last thing logged?
tail -n 5 $P/results/m9_l1_diag2/metrics.jsonl; tail -n 20 $P/logs/*-10517337.err
```

How to read the results:

- **Fragmentation:** `InUse` well below `Limit` with `Reserved >> InUse`, or a small `LargestFreeBlock`.
- **External memory (DALI):** `InUse + Reserved` well below the device's 45 GiB.
- **Genuine in-use need:** `InUse` close to `Limit`.

### 5.2 One cluster job (needs Jay's approval)

The job runs on rtx8000, 1 GPU, 3 CPUs, 64G, 1 h, account qtim. It runs everything inside `tf10_nv.sif`.

Files to copy:

- `.agents/tasks/m9-level1-oom/diag_l1_memory.py` -> `tests/gpu/diag_l1_memory.py`
- `.agents/tasks/m9-level1-oom/m9_oom_diag.sh` -> `slurm/m9_oom_diag.sh`

Then commit, push, `git pull --ff-only` on the cluster and run:

```bash
cd /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad
sbatch slurm/m9_oom_diag.sh             # optional env: LIMIT_MB=38000 RUN_ASYNC=1
```

What the tool does:

- **`main` mode:** runs `main.main(argv)` unchanged. It wraps `_load_cascade_source`, `GroupRegLevelModel.build_model`, `FrozenChain.from_checkpoint`, `_build_dali_data`, the first 3 fetches per task and every `Trainer.train_step`. The TF peak is reset before each step. An OOM is logged and re-raised.
- **`dali` mode:** runs `_build_dali_data` with the same args, then 3 fetches per task.
- **Every record:** TF in-use current/peak, nvidia-smi memory of this PID (TF reserved + DALI + context), device used/total and utilisation. A 1 s poller logs the process maximum.

The runs (records go to `scratch/m9_oom_diag/<job>/<run>.jsonl` and `.log`; the job's `.out` shows a grep summary):

| Run | What | Answers |
|---|---|---|
| R1 `dali` | DALI + validation sets, augmentation on | DALI footprint = growth of `proc_mb` across the fetches (TF in-use barely moves) |
| R2 `dali_noaug` | same, `--intensity-aug false` | share of the augmentation ops |
| R3 `growth` | D1 config, 2 steps, `TF_CPP_MIN_LOG_LEVEL=0` | `proc_mb` vs `tf_peak_mb` right before / at the step-0 OOM, plus the full BFC dump |
| R4 `limit` | R3 + `--gpu-memory-limit-mb $LIMIT_MB` (default 38000), 6 steps (D r D r r D) | does one BFC region fix it; measured `tf_peak_mb` per task; or a DALI OOM if the limit is too high |
| R5 `async` | R3 + `TF_GPU_ALLOCATOR=cuda_malloc_async`, 4 steps, 10 min cap | where D2 stalls (fetch vs step) and the utilisation trace |

Estimated wall time is ~40 min. Each run is capped by `timeout`; the D1 settings (1 pair, no recalibration, no validation, no PNGs) keep it short.

If R1 shows DALI below ~6 GiB, raise `LIMIT_MB` for a re-test: set it to 46080 minus the R3 pre-step context/TF baseline minus the R1 DALI footprint minus 512.

## 6. Recommendations (ranked by expected saving / risk at level 1, N 4)

| # | Fix | Expected effect | Risk |
|---|---|---|---|
| 1 | Run level 1 with a fixed TF limit, no growth: `EXTRA_ARGS="--gpu-memory-limit-mb <L>"` (the flag already exists in `main.py` / `runtime/args.py`). L = 46080 - context (~500) - DALI (R1) - 512 MiB, about 37-38.5k on the RTX 8000. | Removes cross-region fragmentation: the 3-8 GiB effective loss behind the D1 OOM (estimate; R4 verifies). Enough for 1 pair (31 GiB in use + first step); 4 pairs (+4-7) likely still too tight without #2. | Low. No code change. If L is too high, DALI fails with a clear error at the first fetch. |
| 2 | Shrink the DALI footprint. Move the intensity augmentation (noise, brightness, contrast, low-res, gamma, clamp) out of DALI into the TF postprocess, where it is transient and shares BFC memory with the step. Alternatively, fuse it into fewer DALI ops; with fewer ops, `low_res` and gamma are the expensive part. DALI then keeps only copy + warp (~1 GiB). | ~4-5 GiB returned to TF (R2 verifies). Makes 4 pairs plausible together with #1. | Medium. Needs augmentation-equivalence tests (distribution, flips, seeds) and a CPU/GPU time check. |
| 3 | fp16 (AMP) one-hot and one-hot warp in the deform losses, instead of float32 `[4,160,224,192,19]`. That tensor is 2.1 GB, and so is the warped copy in `GatherV2`; this is the OOM site of jobs 10517270 and 10517322, even with `--lean-losses`. | ~2 GiB at the loss / backward peak. | Medium. Dice numerics in fp16 (sums over 6.9 M voxels must stay in fp32); needs golden tests. |
| 4 | Lower the transient startup footprint under growth: render the validation sets on the CPU or in chunks of 2, before the model is built. | Less fragmentation, only if growth is kept; redundant with #1. | Low. |
| 5 | `TF_GPU_ALLOCATOR=cuda_malloc_async` | Avoids fragmentation (D2 passed step 0), but stalled. | High until R5 explains the stall; would also need `cudaMemPoolAttrReleaseThreshold` / DALI pool tuning. Not recommended now. |
| 6 | DALI `exec_async=False` / `exec_pipelined=False` | 0-1 GiB (no concurrent next-batch buffers); loses data/compute overlap. | Low-medium (step time). |

Not worth doing, because the saving is near zero at the peak:

- Computing the frozen chain outside the train-step `tf.function`. Its outputs (U, U_inv, V, 0.33 GB each) are needed in the step anyway, and its level-2 activations are freed before the backward pass.
- Running the frozen chain in fp16 (0.1-0.5 GB, and float32 is a design rule in `models/cascade.py`).
- Trimming the diagnostics, Adam slots or the LSO (< 0.5 GB together).

### Follow-ups for the tools

- `models/memory_model.py`'s `BUDGET_GIB = 44.0` ignores DALI. At level 1 the usable TF budget is about 44 minus the DALI footprint (~6-8 GiB with augmentation).
- `measure_level_memory.py` should report `warmup_peak_mb` next to `peak_mb` in the recommendation, because training fails on the first step.
- Validate by checking the cascade smoke's `gpu_peak_mb` against nvidia-smi per-process memory, not TF in-use alone.

## 7. What was checked and what was not

**Checked by reading the code:**

- `runtime/run.py` (`Trainer.__init__`, `_make_deform_step`, `_make_train_step`, `train_step`, `fit`);
- `tests/gpu/measure_level_memory.py::measure`;
- `tests/gpu/frozen_chain_standin.py`;
- `models/cascade.py` (`FrozenChain`, `FrozenStage`);
- `models/group_net.py` (`_call_deform`, `_call_deform_full`, `build_model`);
- `data_loading/postprocess.py::DeformPostprocess`;
- `data_loading/dali_loader.py` (`LinearPipeline`, `DeformPipeline`, `intensity_augment`);
- `data_loading/data_module.py`;
- `main.py` (startup order, `_build_dali_data`, memory configuration);
- `runtime/args.py` (flag plumbing, defaults: seed 0, prefetch 1, intensity aug on);
- `slurm/m9_level1_cascade.sh` and `slurm/m9_memory.sh`;
- the task draw sequence for seed 0, computed locally.

**Not verified:**

- The DALI footprint numbers. They are op-count estimates; R1 and R2 measure them.
- The BFC region layout and fragmentation size. The D1 `.err` dump and R3 show them.
- The cause of the D2 stall (R5).
- `warmup_peak_mb` of job 10516237 (it is in `memory_m9.json` on the cluster).

The diagnostic files pass `python -m py_compile` and `bash -n` locally; they have not been run.
