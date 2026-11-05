# Design: SynthMorph-style synthetic-contrast training (synth-contrast), v4

Status: DESIGN ONLY (no code, no commits, no jobs). Workflow wf_54f1dd7ba6107c74.
Author step: design author. Reviewer verdict goes to `review.json` / `review.md` in this folder.

## 0. Overview

We add an opt-in synthetic-contrast path to the training pipeline. With `--synth-prob p > 0`, each
training group (one group per step, any task) becomes, with probability p, a fully synthetic group:
every image of the group is replaced in place by an image synthesised from its fine SynthSeg label
map with ONE random contrast shared by the group plus a small per-image jitter, then blurred, bias-
corrupted, gamma-transformed and noised, and finally passed through the EXISTING TF intensity
augmentation (`data_loading/tf_intensity_aug.py`) and the existing flips. The group contrast is
hierarchical: one mean per train18 class (left/right shared) plus a small per-parcel deviation
(Jay's answer 9). Real and synthetic groups
go through the same traced `tf.function` per task; the switch is a boolean tensor drawn inside the
graph and consumed by one `tf.cond`. Losses do not change. A new label set `synthfine` (the
103-id SynthSeg `--parc` map) exists at levels 1, 2 and 4, downsampled by a NESTED majority vote
that guarantees `LUT(synthfine_k) == train18_k` voxel for voxel. It is added to the existing
`npy_v2` export by a new ADD-ONLY command `export-add-labels` that is driven by
`export_manifest.csv` (no original filters or chunk boundaries needed), never rewrites an existing
file, and verifies every new file against the existing train18 file. The mixed model is trained
with model config B (`--group-config B`) at all three levels (Jay's answer 4). In-training
validation stays real-T1-only and unchanged; synthetic-contrast evaluation is a separate
post-training script that runs both models on identical, seeded synthetic validation groups.
Default `--synth-prob 0` builds no new op, reads no new file and creates no new generator, so
every existing golden stays bit-identical.

Technology stack (locked): Python 3.8.10, TensorFlow 2.10 (raw `tf.*` ops, float32, no
tensorflow-probability), DALI 1.18 (only ops already used in `data_loading/dali_loader.py`:
`fn.readers.numpy`, `fn.cast`, `fn.reshape`, `.gpu()`), numpy 1.21 for the exporter and tests.
No new third-party dependency.

## 1. Findings from the code (evidence)

| Topic | What the code does today | Where |
|---|---|---|
| Linear pipelines | Read images only (no labels), warp them in DALI with the pull matrix (`warp_linear`), emit `(images, T_dali[, Rt_dali][, G_dali, control_version])`, all float32 | `dali_loader.LinearPipeline.define_graph`, `linear_output_signature` |
| Deform pipeline | Two aligned numpy readers (image, train18 labels; same pre-permutation `paired_permutation`, same seed, `shuffle_mode='epoch'` only); labels uint8; optional residual affine warps labels NN | `dali_loader.DeformPipeline`, `paired_reader_kwargs`, `check_deform_inputs` |
| Label files | `labels_train18{,_2,_4}.npy` (uint8, 19 ids); `manifest_files(kind='labels', label_set=...)` picks `labels_<set><suffix>.npy` from the `labels_l<k>` `;`-list | `data_module.manifest_files`, `LinearDataModule.label_files` (hard-codes `train18`) |
| Exporter | `synthseg` label set (103 contiguous ids, uint8) is OPT-IN and exported at levels 1 and 2 only (`LABEL_SET_LEVELS = {"train18": (1, 2, 4), "synthseg": (1, 2)}`); train18 and synthseg are both read from the raw `labels_synthseg.nii.gz`, remapped by a fixed LUT, THEN block-mode downsampled (`block_mode`, ties -> smallest id) | `preprocessing/export.py` lines 112-120, 489-510; `preprocessing/luts.py` |
| Exporter resume | If ANY requested file of a case is missing, ALL items of the case are rewritten (`_needs_write` returns one bool) | `export._needs_write`, `_write_items` |
| TF post-process | Runs INSIDE each task's traced step body, outside the tape: cast -> TF intensity aug (own generator `aug_generator`, seed + 7919) -> flips -> targets/pairs (linear) or LUT-free int32 labels + `tf.one_hot(labels, 19)` (deform) | `postprocess.LinearPostprocess.__call__`, `DeformPostprocess.__call__`; `run.Trainer._make_train_step` / `_make_deform_step` (`batch = pp(...)` in `body`) |
| Intensity aug | `augment_group`: `tf.map_fn(..., parallel_iterations=1)`, fixed draw pattern, float32, images only; all levels run `--intensity-aug-device tf` | `tf_intensity_aug.py`; `slurm/r_level{4,2,1}.sh` (`--intensity-aug-device tf --lowres-method tent`) |
| Losses | Linear losses use only matrices (`linear_loss(A, batch, S)`, mm^2). Deform: soft Dice on `batch["onehot"]` (train18), LNCC on `batch["images"]` (the post-processed image) with the LOO template and the train18-foreground ROI | `runtime/losses.py` `linear_loss`, `deform_loss`, `template_ncc_loss`, `ncc_roi` |
| Trace-once | `Trainer.trace_counts[task]` incremented in each `body`; `check_mixed_run.py` asserts 1 per task | `run.py` 904 / 1024; `tests/gpu/check_mixed_run.py` 678-681 |
| Linear step signature | `signature = [tf.TensorSpec(s, tf.float32) for s in shapes]` (dtypes ignored) | `run.Trainer._make_train_step` |
| BN recalibration | Eager loop over `bn_recal_plan`; each slot calls the TRAINING post-process `self.postprocess[t](*inputs)` | `run.Trainer.recalibrate_bn`, `_recal_input` |
| Checkpointed RNGs | `checkpoint_variables()` adds `postprocess_<task>` and, only if TF aug is on, `postprocess_<task>_aug` | `run.Trainer.checkpoint_variables` |
| GPU limit | With `--gpu-memory-limit-mb`, memory growth is OFF and TF reserves the fixed arena; DALI memory is outside it | `main.main` (`configure_tf_precision(memory_growth=not limit)`) |

Consequences used below: (i) the linear pipelines need a new label input; (ii) a label set at
level 4 does not exist yet; (iii) a plain block mode of the 103-id map does NOT aggregate to the
train18 block mode (section 3.2); (iv) BN recalibration calls the same post-process object, but
positionally (`self.postprocess[task](*inputs)` in `Trainer._recal_input`, run.py ~1722), so the
new label input must be passed by keyword through one shared helper (`Trainer._pp_call`, 5.3);
appending a positional `labels` parameter would bind the labels to `rt_dali` (affine) or
`flip_axes` (rigid).

## 2. Requirements (summary)

Functional:
- F1 `--synth-prob p` (default 0): per group, synthetic with probability p, all three tasks.
- F2 One contrast per group with per-image jitter; then blur, bias field, gamma, noise; then the
  existing TF intensity augmentation and flips.
- F3 Fine labels (`synthfine`, 103 ids, uint8) exported at levels 1, 2, 4 with label-safe
  downsampling; fine -> train18 in TF by a 256-entry table and `tf.gather`, never a one-hot of
  fine labels.
- F4 Linear pipelines load the fine label map when synthesis is on.
- F5 BN recalibration (level 1) uses synthetic groups at the same rate p.
- F6 Separate post-training synthetic-contrast evaluation script (fixed seeds, both models).
- F7 Flags + env overrides in `slurm/r_level{4,2,1}.sh`, checker support, tests, smoke plan,
  PET evaluation outline, retraining plan.

Hard requirements (Jay): H1 same model and group sizes, fits RTX 8000 with current limits;
H2 no new in-training validation; H3 memory-neutral, single traced graph per task, quantified
memory, fallback; H4 default off is bit-identical.

## 3. Labels and the exporter

### 3.1 New label set `synthfine`

`preprocessing/luts.py`: `LUTS["synthfine"]` = the same map and names as `LUTS["synthseg"]`
(103 ids, uint8), source text "SynthSeg 2.0 --parc ids, contiguous (synthfine: train18-consistent
nested downsampling)". New helper `synthfine_to_train18_table() -> np.ndarray uint8 [256]`: entry
k (contiguous synthfine id) = train18 id of raw SynthSeg id `o` with `synthseg_map[o] == k`;
entries 103..255 = 0 (background; never present because the exporter rejects unknown ids). Built
from the two fixed LUTs only (pure numpy), so it is the single source of truth for both the
exporter and TF.

`preprocessing/export.py`:
- `SYNTHSEG_SETS = ("train18", "synthseg", "synthfine")` (source `labels_synthseg.nii.gz`).
- `LABEL_SET_LEVELS["synthfine"] = (1, 2, 4)`. `synthseg` keeps (1, 2) and its plain block mode
  (existing files and tests unchanged).
- File names: `labels_synthfine.npy`, `labels_synthfine_2.npy`, `labels_synthfine_4.npy`
  (uint8, shapes 160x224x192 / 80x112x96 / 40x56x48), listed in the manifest `labels_l<k>`
  column like every set.
- Level 1: `apply_lut(raw, synthfine LUT)` (identical content to `labels_synthseg.npy`).
- Levels 2 and 4: `synthfine_level(fine_full, k) = nested_block_mode(fine_full, k,
  synthfine_to_train18_table())` (3.2), one function used by `export-npy` and by
  `export-add-labels` (3.3).

Why a new set instead of changing `synthseg`: `synthseg` keeps its documented plain-block-mode
meaning and its tests; nothing silently changes. Jay's answer 1: the cluster `npy_v2` tree holds
only `image{,_2,_4}.npy` and `labels_train18{,_2,_4}.npy` per case, so `synthseg` was never
exported there (milestone 8 only added the exporter option).

`export-npy` itself learns `synthfine` (both constants above, levels 2/4 through the same
`synthfine_level` function, 3.2), so a FUTURE fresh export can write it directly. It is NOT used
for the existing `npy_v2` tree: `_needs_write` returns one bool per case and `_write_items` rewrites
all items of an incomplete case, so adding a set via `export-npy` would rewrite every image and
train18 file, and the manifest part names depend on filters Jay no longer has. The existing tree
gets the set through the add-only command of 3.3. `_needs_write` / `_write_items` stay unchanged
(v2's `_items_to_write` change is dropped as no longer needed).

### 3.2 Downsampling: nested (hierarchical) majority vote

Plain `block_mode` of the fine map differs from the train18 file at coarse levels. Example
(level 2, 8 voxels): 3 white matter (fine id of 2), 5 cortex split over 3 DK parcels (2+2+1).
train18 block mode = cortex (5 > 3); fine block mode = white matter (3 > 2), and LUT(white
matter) = white matter. Cortex is the only class split over many fine ids (70 DK parcels + 3/42),
so "downsample then aggregate" systematically thins the cortex at levels 2 and 4 and moves the
cortex/WM boundary. The reverse order ("aggregate then downsample", today's train18) is what the
real-only model was trained on.

Chosen rule, per k^3 block (vectorised exactly like `block_mode`):
1. `counts[b, f]` = `np.bincount(b * 103 + f, minlength=B * 103).reshape(B, 103)`, f in 0..102.
   `np.bincount` always returns int64: L1 -> L2 860,160 x 103 x 8 B = 0.71 GB, L1 -> L4
   107,520 x 103 x 8 B = 0.09 GB. The result is cast to int16 right away (counts <= k^3 = 64;
   0.18 GB / 0.02 GB) and the int64 array is deleted, so the steady footprint is small; the
   transient peak is ~0.9 GB per case process at k = 2 (int64 + int16 copy), which is harmless on
   the basic partition (15 GB per core). The masked argmax in step 3 works on the int16 array
   (`np.where(mask, counts, -1)`, another int16 array of the same size).
2. Parent counts `pc[b, c] = sum_{f: g(f) = c} counts[b, f]` (g = `synthfine_to_train18_table`),
   `c* = argmax_c pc` (ties -> smallest c). This equals `block_mode(train18_full, k)` exactly,
   because train18_full = g(fine_full) and the tie rule is the same.
3. `f* = argmax_f counts[b, f] * [g(f) == c*]` (ties -> smallest f). c* has >= 1 voxel, so f* is a
   voxel of class c*, and `g(f*) == c*`.

Property (tested, and checked on the real export by `export-check`): `g(synthfine_k) ==
train18_k` for every voxel, every case, every level. Nearest-neighbour or majority votes only;
never interpolation.

### 3.3 Add-only export into the existing tree: `export-add-labels`

Choice. Three ways to add one label set to `npy_v2` were considered: (i) rerun `export-npy` with
the original filters (rejected: Jay does not have the original command, and `export-npy`
rewrites incomplete cases, 3.1); (ii) reconstruct the original filters from
`manifest_parts/*.csv` names and rerun (rejected: still rewrites images, and a wrong guess makes
`merge_parts` fail or drop rows); (iii) a new add-only command whose case list is
`export_manifest.csv` itself (chosen). With (iii) the case set, order and every non-label column
come from the file the trainer already reads, no filters are needed, and the only files created
are `labels_synthfine{,_2,_4}.npy`.

New module `preprocessing/add_labels.py`, CLI subcommand `export-add-labels` in
`preprocessing/cli.py` (registered like `export-merge`, function `cmd_export_add_labels`), SLURM
wrapper `preprocessing/slurm/add_labels_array.sh` (basic partition, `-A qtim`, 1 CPU, 8G,
2 h per task, array over row chunks; same header style as `export_array.sh`). Arguments:
`--out-root <processed>` (required), `--export-root <npy_v2>` (required), `--label-set synthfine`
(only `synthfine` accepted in v1; anything else exits 2), `--phase files|manifest` (required),
`--start/--end` (row indices into `export_manifest.csv`, files phase only; default all rows).

Phase `files` (array-safe: writes only per-case npy files and one status file per chunk):
for each manifest row r in `[start, end)`:
1. Source `src = <out_root>/<r.dataset>/<r.subject_dir>/labels_synthseg.nii.gz`; npy dir
   `dst = <export_root>/<r.dataset>/<r.subject_dir>` (the trainer's rule in
   `data_module.manifest_files`; `r.npy_dir` is only compared and a difference is logged as a
   WARNING, because the tree may have been moved). Missing `src` -> case FAILED ("no source").
2. `fine = apply_lut(load_data(src, float64), LUTS["synthfine"], "uint8")` (the same load and
   remap as `export_npy.full_labels`; an unknown raw id raises ValueError -> case FAILED with the
   message). Its shape must equal `r.shape_l1`, else FAILED.
3. For each level k in `r.levels` (all LUMIR rows: 1;2;4): `f_k = fine if k == 1 else
   synthfine_level(fine, k)` (3.2). Verification BEFORE any write: load the existing
   `labels_train18<sfx>.npy` (named in `r.labels_l<k>`) and require `table[f_k] == train18_k`
   for every voxel (`table = synthfine_to_train18_table()`). At k = 1 this also proves that the
   processed SynthSeg map is the one the original export used (a reprocessed case fails here).
   Any mismatch -> case FAILED with the number of differing voxels; nothing written for it.
4. Write `labels_synthfine<sfx>.npy` with `export._atomic_save_npy` when its `_npy_state` is
   "missing" or "unreadable". State "ok": load it and require bit equality with `f_k` (rerun
   after a partial chunk; mismatch -> FAILED, the file is left alone). State "mismatch"
   (wrong dtype/shape) -> FAILED. Existing image / train18 / other files are only read.
5. Status: `<export_root>/add_labels/synthfine/rows-<start>-<end>.json` (atomic write) with
   per-case `{case, status: written|kept|failed, reason, levels}` and counts. Exit 0 if no case
   failed, 1 otherwise (listing the failures on stdout).
Memory per process: raw float64 55 MB + fine 6.9 MB + `synthfine_level` transient ~0.9 GB at
k = 2 (3.2) -> < 1.5 GB; 8G is ample. Time: ~3-6 s per case (gzip read dominates) -> ~3-6 h for
3423 cases in one task; 7 array tasks of 500 rows ~0.5-1 h each, run at most two at a time
(`%2`, cluster rule, 3.4) -> ~2-4 h wall time for the files phase.

Phase `manifest` (single process, run once after all file chunks):
0. Pre-check (before anything else, read-only): run `merge_parts` on a temporary copy of the
   UNMODIFIED `manifest_parts/` (into `$TMPDIR` or `<export_root>/add_labels/tmp_merge_pre/`) and
   require that its `export_manifest.csv` equals the current `export_manifest.csv` byte for byte.
   On failure: exit 1 with "pre-existing divergence between manifest_parts and
   export_manifest.csv (not caused by export-add-labels)" and the list of differing
   (dataset, subject_dir) keys (rows only in one file, or rows whose columns differ); nothing is
   changed. Jay then decides (e.g. regenerate the manifest with `export-merge`, or restore the
   parts); the command does not repair it. This separates an old divergence from a bug in the
   transform checked in step 5.
1. Re-check every manifest row: the three `labels_synthfine*` files exist with uint8 and the
   level shape (`_npy_state`), and `table[f_k] == train18_k` for every level (re-loads ~14 MB per
   case, ~10 min for all). Any problem -> exit 1, nothing changed.
2. Backup (copy, never move) `export_manifest.csv`, `label_luts.json` and `manifest_parts/` to
   `<export_root>/backup_pre_synthfine_<UTC timestamp>/`.
3. Row transform `add_set(row)`, applied to every row of every `manifest_parts/*.csv` and of
   `export_manifest.csv`: append `;synthfine` to `labels`, `;synthfine:uint8` to `label_dtypes`,
   `;synthfine:<levels of the row, comma list>` to `label_levels`, and
   `;labels_synthfine<sfx>.npy` to each `labels_l<k>` (k in the row's levels). Rows whose
   `labels` already contain `synthfine` are left unchanged (idempotent). Every other column, the
   row order and the column order stay byte-identical. Only rows that are also in
   `export_manifest.csv` are transformed in the parts (other part rows unchanged); a part row in
   the manifest is identified by (dataset, subject_dir) as in `merge_parts`.
4. Add `"synthfine": lut_json(LUTS["synthfine"])` to `luts` of every transformed
   `<part>.luts.json` and of `label_luts.json` (sorted keys, as `merge_parts` writes them).
5. Consistency: run `merge_parts` into a temporary copy of the export root's manifest files
   (`$TMPDIR` or `<export_root>/add_labels/tmp_merge/`) and require that its
   `export_manifest.csv` equals the directly transformed one byte for byte; then write the
   transformed parts, luts and manifest with `_atomic_write_text`. A later `export-merge` thus
   reproduces the new manifest (the parts and the manifest never disagree).
6. Diff check (always printed, and written to `add_labels/synthfine/manifest_diff.json`): for
   every row, old vs new differ only in the six label columns, defined in code as the allowed-diff
   set `LABEL_COLUMNS = ("labels", "label_dtypes", "label_levels") + tuple("labels_l%d" % k for k
   in export.ALL_LEVELS)` (ALL_LEVELS = (1, 2, 4) -> `labels_l1`, `labels_l2`, `labels_l4`), and
   the new values are the old ones plus the appended entries; the row count and
   (dataset, subject_dir) order are unchanged. Any other difference -> restore from the backup
   and exit 1.

Training jobs read the manifest only at startup and every write is an atomic rename, so running
jobs are unaffected; consumers select label files by name (`manifest_files(...,
label_set=...)`, `export_check`), so the extra `;`-entries do not change what any existing code
loads (`train18` remains the default `label_set`). Jay should still run phase `manifest` between
job submissions so no job starts mid-write.

### 3.4 Jay's export commands (agents write and test; Jay runs)

Exact commands (copy-paste; `P=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg`):

```
cd $P/brain-group-rad && git pull
# rows: number of data rows in the manifest (3423 expected: 3384 train + 39 T1 val)
N=$(( $(wc -l < $P/data/npy_v2/export_manifest.csv) - 1 )); echo $N
# phase files, 500 rows per task (7 tasks for 3423 rows), at most two tasks at once (%2, cluster
# rule; ~2-4 h wall time), Jay's own basic-partition array
sbatch --array=0-$(( (N + 499) / 500 - 1 ))%2 --export=ALL,CHUNK=500,LABEL_SET=synthfine \
    preprocessing/slurm/add_labels_array.sh
# only after the array has FINISHED (squeue -u jn85 shows none of its tasks; check the logs /
# add_labels/synthfine/*.json for failures):
sbatch --array=0-0%2 --export=ALL,PHASE=manifest,LABEL_SET=synthfine preprocessing/slurm/add_labels_array.sh
# final check (all cases, includes the new synthfine consistency rule below)
singularity exec -B /autofs -B /cluster -B /space /autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif \
    bash -c "cd $P/brain-group-rad && python -m preprocessing export-check --export-root $P/data/npy_v2 --sample 0" </dev/null
```

The wrapper runs, inside the container,
`python -m preprocessing export-add-labels --out-root $P/data/processed --export-root
$P/data/npy_v2 --label-set $LABEL_SET --phase files --start $((SLURM_ARRAY_TASK_ID * CHUNK))
--end $(((SLURM_ARRAY_TASK_ID + 1) * CHUNK))` (or `--phase manifest` with `PHASE=manifest`, which
ignores the array index; the header default `#SBATCH --array=0-0%2`, as in `export_array.sh`,
makes a plain `sbatch` a single task). It uses `singularity exec` without `--nv` (CPU only, binds `-B /autofs
-B /cluster -B /space`), `PHASE` defaults to `files`, `CHUNK` to 500, and an invalid value exits
2 before any work. Extra disk: (6.9 + 0.86 + 0.11) MB x 3423 cases = 27 GB (approved).

`export_check` gains: for every checked case that lists `synthfine` at a level, verify
`table[synthfine_k] == train18_k` (exact), `max(synthfine_k) <= 102`, dtype uint8; failures are
listed like the existing checks (exit 1).

### 3.5 Decision: training labels from the fine map (option a)

With synthesis on, the deform pipeline reads ONLY `labels_synthfine_<k>.npy` (one uint8 volume
per image) and the post-process computes train18 as `tf.gather(table, fine)`. Because of 3.2
these labels are identical to the `labels_train18` files, so option (a) gives exactly the labels
of option (b) while loading one volume instead of two (DALI host and device memory unchanged vs
today). Validation (in training and in the synthetic evaluation) always uses the existing train18
files for both models.

Residual affine (`--deform-residual-affine`, default off): DALI warps the fine labels NN with the
same pull matrix (existing code path, label dtype unchanged), so the LUT still applies.

## 4. Synthesis

### 4.1 Module and API

New module `data_loading/synth.py` (TF, float32 raw ops; imports `data_loading.params` and
`preprocessing.luts` only for the table; `preprocessing/__init__.py` has no imports, `luts.py`
only numpy, so this works in the container):

```python
@dataclass
class SynthConfig:                    # data_loading/params.py, recorded in config.json
    prob: float = 0.0                 # --synth-prob
    jitter: float = 0.08              # --synth-jitter, per-image per-label relative half-width
    std_max: float = 0.08             # within-label voxel std ~ U(0, std_max) (group)
    fwhm_mr_mm: tuple = (0.0, 3.0)    # MR-like blur family
    fwhm_pet_mm: tuple = (4.0, 8.0)   # PET-like blur family
    pet_family_prob: float = 0.5
    fwhm_image_jitter: float = 0.10   # per image and axis, relative
    bias_grid: int = 4                # control points per axis (same physical spacing at all levels)
    bias_std_max: float = 0.3         # log-field std ~ U(0, max) (group), field per image
    gamma_log_std: float = 0.25       # gamma = exp(N(0, s)) (group)
    noise_std_max: float = 0.03       # white noise std ~ U(0, max) (per image)
    remask_prob: float = 0.5          # multiply by (label > 0) after blur (skull-stripped look)
    hier_dev: float = 0.1             # half-width of the parcel deviation around the class mean (4.3)
    norm_quantile: float = 0.995      # robust in-mask normalisation quantile (4.2 step 5)
    device: str = "gpu"               # --synth-device gpu | cpu
    label_set: str = "synthfine"      # --synth-label-set

def fine_to_train18_table() -> tf.Tensor            # int32 [256], from preprocessing.luts
def draw_synth_params(generator, n) -> dict          # fixed draw pattern (4.4)
def synthesize_group(labels_u8, params, cfg, level) -> tf.Tensor   # float32 [N, X, Y, Z, 1]
def maybe_synthesize(images, labels_u8, flag, params, cfg, level, render=None) -> tf.Tensor
```

`maybe_synthesize` = `tf.cond(flag, lambda: synth_branch, lambda: images)`; the false branch
returns the input tensor unchanged (no op on the data, so real groups are bit-exact). With
`cfg.device == "cpu"` the true branch body is wrapped in `tf.device("/cpu:0")` (fallback, 6.4).

### 4.2 Per-image chain (inside `tf.map_fn(..., parallel_iterations=1)`, as `augment_group`)

For image i with fine labels `l` (uint8 [X, Y, Z]), group tables `mu[256]`, `sd[256]`, per-image
jitter `d_i[256]`:
1. `idx = tf.cast(l, tf.int32)`; `m = tf.gather(mu * (1 + d_i), idx)`;
   `s = tf.gather(sd * (1 + d_i), idx)`; `x = m + s * stateless_normal(seed_i1)`. Background
   (id 0) has mu = sd = 0. No one-hot anywhere: two gathers of a 256-entry table.
2. Blur: separable Gaussian as three `[n, n]` operators applied by `tf.tensordot` (the pattern of
   `tf_intensity_aug._apply_axis_matrix`), operator rows `exp(-(o - j)^2 / (2 sigma^2))`
   normalised to sum 1 (renormalised truncation at the borders, shape unchanged, no cuDNN
   workspace). `sigma_vox = sqrt((FWHM_i,a / 2.3548)^2 + s^2 / 12) / s` with s the level spacing
   (the s^2/12 term is the box partial-volume blur that the real block-averaged images have).
3. Optional re-mask (group draw < remask_prob): `x *= cast(idx > 0)`.
4. Bias: `B = exp(field)`, field = per-image normals on a `bias_grid^3` grid times the group std,
   upsampled by three static `[n, g]` linear-interpolation matrices (numpy-built at trace time).
   `x *= B`.
5. Normalise like the real images (in-mask p99.5 then clip, `preprocessing/intensity.py
   normalize_intensity`), approximated cheaply in-graph: take a fixed strided subsample
   `xs = x[::4, ::4, ::4]` and its foreground `fs = idx[::4, ::4, ::4] > 0` (L1: 107,520 voxels,
   ~60 k in the mask; L2 13,440; L4 1,680 sampled voxels). `v = tf.boolean_mask(xs, fs)`,
   `k = max(1, round(0.005 * size(v)))` (dynamic), `hi = reduce_min(tf.math.top_k(v, k).values)`
   (= the in-mask 99.5th percentile of the subsample), `x = clip(x / max(hi, 1e-6), 0, 1)`.
   `tf.math.top_k` with a dynamic k and `tf.boolean_mask` exist in TF 2.10 on GPU; the temporaries
   are one subsample (<= 0.43 MB at L1). A plain max would let one noisy voxel set the scale and
   make synthetic images darker on average than real ones (real images clip ~0.5 % of the mask to
   1). An empty foreground (size 0) gives k = 1 on an empty tensor, so the code guards with
   `tf.cond(size(v) > 0, ..., lambda: 1.0)`; the image is then all zero and stays finite.
6. Gamma: `x = pow(x, gamma_g)` (x in [0, 1], so gamma keeps the range).
7. Noise: `x += noise_std_i * stateless_normal(seed_i2)`; `x = clip(x, 0, 1)`; then
   `x *= cast(idx > 0)` if re-masked in step 3 (noise stays zero outside the mask, as on real
   skull-stripped images).

Then (unchanged code) TF intensity augmentation and flips run on the result.

### 4.3 Parameter ranges and justification

SynthSeg/SynthMorph values below are from memory and approximate: SynthSeg samples label means
U(0, 255) and stds U(0, 25) on a 0-255 scale, blurs to a random resolution up to several mm,
uses a smooth multiplicative bias field (log-field std on the order of 0.5 on a coarse grid of
~4-5 control points per axis), gamma exp(N(0, ~0.25-0.5)) and min-max normalisation; SynthMorph
uses similar label-driven images with smaller blur for brain registration.

| Parameter | Range | Why |
|---|---|---|
| Group mean per label | hierarchical ONLY (Jay's answer 9): `mu_c ~ U(0, 1)` per train18 class (left/right and all cortex parcels share a class, `luts.TRAIN18_CLASSES`), `mu_f = clip(mu_c[g(f)] + U(-0.1, 0.1), 0, 1)` per fine id. Background 0 | MR-like uniform tissue classes and PET-like smooth regional variation (amyloid/tau differ by region, not by DK parcel boundary noise), L/R symmetric up to the small deviation. The fully independent SynthMorph-style per-label means (patchwork cortex, L/R-asymmetric WM) are dropped for now (backlog B2). Background 0 matches the skull-stripped training images (`preprocessing/intensity.py`: 0 outside the mask) |
| Within-label std | `sd_c ~ U(0, 0.08)` per train18 class (group), `sd_f = sd_c[g(f)]` | SynthSeg ~0-0.1; blurred, it becomes correlated texture; shared per class like the mean |
| Per-image jitter | relative U(-0.08, +0.08) per (image, label), applied to mean and std | Jay's 5-10 %; regional SUVR changes over 1-3 years are of this order; 103 labels incl. 70 DK parcels gives regional (not global) changes; global gain is removed by step 5 anyway |
| Blur FWHM | per group: MR family U(0, 3) mm with p 0.5, else PET family U(4, 8) mm; per image and axis x U(0.9, 1.1) | Covers MR and PET (amyloid/tau PET ~4-8 mm after reconstruction); small per-image change mimics scanner/reconstruction differences between visits |
| Re-mask after blur | p 0.5 (group) | PET used with an MR- or SynthStrip-derived mask (hard edge) vs unmasked blur spill (soft edge) |
| Bias field | grid 4^3, log-std U(0, 0.3) (group), field per image | Smaller than SynthSeg's MR-oriented value: PET has little coil bias, but attenuation/scatter errors give smooth gain changes |
| Gamma | exp(N(0, 0.25)) per group | SynthMorph-like |
| Noise | std U(0, 0.03) per image | Mild; the existing TF aug adds its own noise (p 0.1, std <= 0.05) per image |

Optional random sub-splitting of large labels (WM/GM) for texture: evaluated and NOT included in
v1. A cheap version (a second low-frequency random field with per-label amplitude, +1 volume of
temporaries) is possible, but the within-label noise plus blur, the 70 cortical parcels and the
bias field already give intra-class variation, and the PET targets are smooth. Backlog item B1,
only if the synthetic evaluation shows texture-related failures.

### 4.4 Randomness

Each post-process with synthesis on owns a third generator `synth_generator =
tf.random.Generator.from_seed(seed + P.SYNTH_SEED_OFFSET)` (`SYNTH_SEED_OFFSET = 104729`), so the
flip/pair stream (`generator`) and the TF-aug stream (`aug_generator`) are unchanged. Fixed draw
pattern per call, independent of the flag value and of p: group uniforms `[G_COLS]` (flag, blur
family, blur base, remask, bias std) and one group normal (log-gamma), group tables
`uniform([3, 256])` (class means and class stds indexed by train18 id 0..18 in the first 19
entries, parcel deviations indexed by fine id); with `g = fine_to_train18_table()` the mean table
is `clip(tf.gather(cls_mean, g) + 2 hier_dev (dev - 0.5), 0, 1)` and the std table
`tf.gather(std_max * cls_std, g)`, then entry 0 of both is set to 0; per-image `uniform([N, 256])` (jitter), `uniform([N, 3])`
(FWHM jitter), `uniform([N, 1])` (noise std), `normal([N, g^3])` (bias grid), `make_seeds(N)` x 2
(stateless normals). Column layout constants in `params.SYNTH_COLUMNS`. The flag is
`u_flag < p`. The generator is added to `Trainer.checkpoint_variables()` as
`postprocess_<task>_synth` only when synthesis is on (existing checkpoints unchanged).

## 5. Integration

### 5.1 DALI pipelines (`data_loading/dali_loader.py`, `data_module.py`)

- `LinearPipeline(..., label_files=None)`: when given, the image and label lists are permuted with
  the same `paired_permutation(len, seed)` (identical to today's `RandomState(seed).permutation`)
  and read by two readers built with `paired_reader_kwargs` (shuffle_mode must be 'epoch'; 'buffer'
  raises ValueError as in `check_deform_inputs`). Labels are read with `_reader_to_gpu(..., UINT8)`
  and NOT warped (synthesis happens in the template space, 5.2). Output order:
  `(images, T_dali[, Rt_dali], labels_u8[, G_dali, control_version])`. The label reader op is
  created after all random ops in `define_graph` (and has an explicit seed), so the image and
  matrix streams are as without labels; GPU check s1 confirms it.
- `linear_output_signature(task, level, n, control=False, labels=False)` returns per-output dtypes
  (`uint8` for labels); `base_output_count(task, labels=False)` and `output_count(task,
  residual_affine=False, labels=False)` count the label output.
- `LinearDataModule(..., synth_label_set=None)`: `synth_label_files` property (same rows/order as
  `files`, `manifest_files(kind='labels', label_set=synth_label_set)`); with a set given, linear
  pipelines get `label_files=synth_label_files` and the deform pipeline gets `synth_label_files`
  instead of `label_files`. None (default) = today's code path exactly.

### 5.2 Post-process (`data_loading/postprocess.py`)

`LinearPostprocess(..., synth=None)` and `DeformPostprocess(..., synth=None)` take a
`SynthConfig` (None or prob 0 -> no generator, no op). New KEYWORD-ONLY call arguments:
`LinearPostprocess.__call__(self, images, t_dali, rt_dali=None, flip_axes=None, *, labels=None,
synth_override=None)` and `DeformPostprocess.__call__(self, images, labels, t_dali=None, ..., *,
synth_override=None)` (deform labels stay the existing second positional argument; with synthesis
on they are the fine ids). `synth_override` is a Python bool or None: None draws the flag, a bool
forces it (BN recalibration); the draws are consumed either way. With synthesis on, a linear call
without `labels` raises ValueError at trace time ("synthesis on: labels= required"), so a wiring
mistake fails loudly instead of silently training real-only.

Linear, with synthesis on:
1. cast images to float32; `t_mm = dali_to_mm(t_dali)` (moved up; pure function, no RNG).
2. `flag, prm = draw(...)`; `images = tf.cond(flag, synth_render, identity)` where
   `synth_render = transform_image(synthesize_group(labels), affine_inverse(t_mm), spacing)` in
   chunks of one image (`tf.map_fn`, parallel_iterations 1; `models.layers.transform_image`,
   linear, fill 0). This renders I_i(y) = S_i(T_i^-1 y), the same pull relation DALI uses for the
   real image (postprocess module docstring step 2), so the targets from `t_mm` are exact for the
   synthetic image by construction. Synthesising in template space and rendering linearly mirrors
   how the real image is produced (smooth image, one linear interpolation) and avoids NN-warped
   label staircases.
3. existing TF intensity aug, flips, targets, pairs (unchanged code).
The labels are not used after step 2 (dead before the model forward).

Deform, with synthesis on: labels input = fine uint8; `images = tf.cond(flag, synth, identity)`
(no render: the deform pipeline does not warp unless the residual affine is on, and then the
labels are already warped), then `labels = tf.gather(fine_to_train18_table(), int32(fine))`, then
the existing aug, flips, `tf.one_hot(labels, 19)` (train18 only, as today), pairs.

The batch dict gains `synth` (bool scalar) only when synthesis is on.

### 5.3 Trainer (`runtime/run.py`)

- `TrainConfig.synth` (SynthConfig or None), passed to both post-process constructors.
- One helper routes every post-process call (training `body` of both step builders and
  `_recal_input`), so the label mapping lives in one place:

  ```python
  def _pp_call(self, task, inputs, synth_override=None):
      """Post-process one split batch. inputs = split_batch(...) order without skip/G/version."""
      pp = self.postprocess[task]
      kw = {} if synth_override is None else {"synth_override": synth_override}
      if task == DEFORM:
          return pp(*inputs, **kw)        # (images, labels[, t_dali]); labels = fine ids when on
      if not self._synth_on:
          return pp(*inputs)              # today's call, unchanged (default-off path)
      n = 3 if task == "rigid" else 2     # (images, T[, Rt]) then labels
      return pp(*inputs[:n], labels=inputs[n], **kw)
  ```
  `self._synth_on = cfg.synth is not None` (Python bool, trace-time constant).
- `_make_train_step`: the signature uses the dtypes from `linear_output_signature(...,
  labels=on)` (uint8 for labels; float32 for the rest as today).
  `body(images, t_dali, rt_dali=None, skip=None, *, labels=None)` (positional order of today's
  `body` in `_make_train_step`, run.py 1023, kept; `labels` keyword-only and last) calls
  `self._pp_call(task, [images, t_dali] + ([rt_dali] if rigid else []) + ([labels] if on else []))`.
  Today's four lambdas (run.py 1113-1121) pass `skip` positionally as the 4th argument
  (`body(images, t_dali, rt_dali, skip)` / `body(images, t_dali, None, skip)`); with the order
  kept they bind exactly as today and stay byte-identical on the default-off path, and a
  positional misuse of `labels` is impossible. With synthesis on, four NEW lambdas are used:
  - rigid + cascade: `lambda images, t_dali, rt_dali, labels, skip: body(images, t_dali, rt_dali, skip, labels=labels)`
  - affine + cascade: `lambda images, t_dali, labels, skip: body(images, t_dali, None, skip, labels=labels)`
  - rigid, no cascade: `lambda images, t_dali, rt_dali, labels: body(images, t_dali, rt_dali, labels=labels)`
  - affine, no cascade: `lambda images, t_dali, labels: body(images, t_dali, None, labels=labels)`
  The lambda parameter order follows the DALI/`split_batch` order (labels before skip).
  with `input_signature = signature (+ [skip spec])`, signature now containing the label spec at
  position 3 (rigid) / 2 (affine), matching the DALI output order of 5.1.
  `_make_deform_step` keeps its three lambdas; only `body` switches to `self._pp_call`.
  The extra input exists only when on, so the default signature and graph are unchanged.
- `split_batch` uses `output_count(..., labels=on)`; the returned tuple keeps the DALI order
  `(images, T[, Rt], labels)` followed by the skip draw as today.
- Step results gain `synth` (float 0/1) only when on; `fit` logs window means
  `train_<task>/synth_frac`, `train_<task>/loss_real`, `train_<task>/loss_synth` (Python side,
  split by the flag; no graph change).
- Single traced graph per task: the flag is a tensor drawn inside `body`; `tf.cond` traces both
  branches once. Nothing Python-side varies between real and synthetic steps, so
  `trace_counts[task]` stays 1 (existing checker `trace_once`).
- `recalibrate_bn`: with synthesis on, slot b_t (the b_t-th recalibration batch of task t, K_t
  batches) is synthetic iff `floor((b_t + 1) p) > floor(b_t p)` (exact count floor(K_t p),
  interleaved, no RNG). `_recal_input(task, inputs, synth_override)` replaces its
  `self.postprocess[task](*inputs)` with `self._pp_call(task, inputs, synth_override)`; with
  synthesis off it passes None and the call is today's. The returned record gains
  `synth_batches: {task: planned count}` (= `floor(K_t p)`, the slots assigned synthetic by the
  rule, independent of what happens to them) and `synth_batches_used: {task: count}` (planned
  synthetic slots whose batch was actually used for the statistics; a slot that
  `recalibrate_bn` skips as non-finite, `skipped_nonfinite[t]`, is not counted). At K = 64 and
  ratio 1:1:2 (`bn_recal_plan` 16 / 16 / 32) with p 0.5: planned 8 / 8 / 16.
  Levels 2 and 1 recalibrate at the start of a fresh cascade run (`r_level2.sh`: 64 batches,
  freeze false, checker `--expect-bn-recal true`; `r_level1.sh`: 64 batches, freeze true); both
  use the same synthetic-slot rule. Level 4 has no recalibration; at L4 and L2 BN stays trainable
  and also sees the real/synthetic mixture through training.
- Validation: unchanged (no new sets, no new passes).

### 5.4 Arguments (`runtime/args.py`)

| Flag | Type / limits | Default | Validation failure |
|---|---|---|---|
| `--synth-prob` | float, 0 <= p <= 1, finite | 0.0 (off) | parser.error |
| `--synth-jitter` | float, 0 <= j <= 0.5 | 0.08 | parser.error |
| `--synth-device` | gpu \| cpu | gpu | argparse choices |
| `--synth-label-set` | name matching `^[A-Za-z0-9_-]+$` | synthfine | parser.error |

Cross-checks (parser.error, exit 2): p > 0 needs `--intensity-aug-device tf` or `--intensity-aug
false` (DALI-side augmentation would be overwritten by synthesis); p > 0 is refused with
`--overfit-one-group`; p > 0 with `--shuffle-mode buffer` is refused (paired readers). Missing
label files are a startup ValueError from `manifest_files` naming the case (fatal, before any
GPU work). `A.synth_config_from_args(args)` -> SynthConfig or None (None when p == 0, which keeps
the default path free of new objects). `synth_config_from_args` reads every field with
`getattr(args, "synth_prob", 0.0)` etc., so a Namespace from an older config.json (no synth
fields) yields None. config.json records `synth = asdict(SynthConfig)` ONLY when p > 0 (same
pattern as `_optimizer_record`, which returns None for default runs); default runs write no
top-level `synth` record. config.json `args` (`runtime/args.args_to_dict`, which dumps every
parsed argument) gains the four `synth_*` defaults in every run, like every earlier new flag;
training outputs and goldens are unaffected, and no test compares config.json bytes.

Resume: `main._check_synth_config(config_path, args)` (like `_check_intensity_aug_device`, where a
missing key means 'dali'): a MISSING `synth` key means off. Saved off + requested off -> OK
(covers in-flight runs such as the L2 smoothness sweep, resumed with the same RUN_NAME); any other
difference (off vs on, or different field values) -> SystemExit with both values.
A synth run resumed without the `postprocess_<task>_synth` checkpoint entry (impossible unless
config changed) is caught by the same check.

### 5.5 Scripts (`slurm/r_level{4,2,1}.sh`)

New env overrides: `SYNTH_PROB` (default 0), `SYNTH_JITTER` (0.08), `SYNTH_DEVICE` (gpu),
`SYNTH_LABEL_SET` (synthfine), `GPU_POLL` (false), `MIN_TASK_STEPS` (empty; checker
`--min-task-steps`, 5.6; validated against `^(rigid|affine|deform):[0-9]+(,(rigid|affine|deform):[0-9]+)*$`,
else `die`). The `--synth-*` flags are appended ONLY when
`SYNTH_PROB` is not 0, so the default command line is byte-identical (asserted in
`tests/test_slurm_scripts.py`). `need_choice SYNTH_DEVICE gpu cpu`; a float regex for
`SYNTH_PROB` / `SYNTH_JITTER`. With `SYNTH_PROB` != 0 the checker gets `--expect-synth-prob
$SYNTH_PROB`. `GPU_POLL=true` starts, on the host side of the job (outside the container, where
`CUDA_VISIBLE_DEVICES` names the allocated GPU),
`nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=timestamp,memory.used,memory.total
--format=csv,noheader,nounits -l 5 > $RESULTS/$RUN_NAME/gpu_poll.csv &` before main.py and kills
it after (trap on EXIT). This is device-level memory (TF arena + DALI + CUDA contexts), which
avoids the PID-namespace problem of per-process queries inside Singularity. One GPU per job, so
the device total is this job's usage. The smoke criterion is
`max(memory.used) <= memory.total - 512` (MiB). Level-1 synth runs pass
`--mem=128G` on sbatch and `TF_MEM_LIMIT_MB=42500` (the values the real-only L1 sweep used; the
script defaults are still 64G / 40000, a pending TODO outside this design).

### 5.6 Checker (`tests/gpu/check_mixed_run.py`)

`--expect-synth-prob p`: per task, the step-weighted mean of `train_<task>/synth_frac` is within
`p +- 4 sqrt(p (1 - p) / n_task_steps)` (FAIL otherwise), and `loss_real` / `loss_synth` are
printed as INFO. With `--expect-bn-recal true` and p > 0: planned
`bn_recalibration.synth_batches[t] == floor(K_t p)` (FAIL otherwise); `synth_batches_used[t]` is
printed as INFO (it may be lower after a non-finite slot, which the existing recal check already
reports). Without the flag nothing changes.

`--min-task-steps rigid:20,affine:20,deform:80` (new, optional; default none = no check): for
every listed task, FAIL unless `trace_counts[task] == 1` (the existing `trace_once` record) AND
the number of executed training steps of that task in `metrics.jsonl` (records of
`train_<task>/...`, the same count used as `n_task_steps` above) is at least the given number.
A task that never traced or never ran therefore fails instead of passing `trace_once` and
`--expect-synth-prob` vacuously. The scripts pass it from the new env `MIN_TASK_STEPS` (default
empty -> flag not passed, default checker line unchanged; asserted in `test_slurm_scripts.py`).

## 6. Memory (H1, H3)

Voxels per image V: L4 107,520; L2 860,160; L1 6,881,280. A float32 volume is 4V bytes (L1
27.5 MB, L2 3.4 MB, L4 0.43 MB); a uint8 volume is V bytes.

### 6.1 GPU, TF allocator

| Item | L4 (lin 128 / def 64) | L2 (32 / 16) | L1 (8 / 4) |
|---|---|---|---|
| New step input: fine labels, linear (N x V uint8), alive for the whole step (held by the caller) | 13.8 MB | 27.5 MB | 55 MB |
| Deform labels (fine replaces train18, same size) | 0 | 0 | 0 |
| Synthesised group (replaces the TF-aug output, which is a new tensor today too) | 0 | 0 | 0 |
| Per-image synthesis temporaries (~6 float32 volumes: int32 index, mean, std*noise, 2 tensordot/transposes, bias), one image at a time | 2.6 MB | 21 MB | 165 MB |
| Per-image render (`transform_image` forward, ~48 B/voxel), linear only | 5 MB | 41 MB | 0.33 GB |
| Unwarped synthetic group before render (linear only, freed after) | 55 MB | 110 MB | 220 MB |

All temporaries live in the post-process, before the frozen-stage forward and the model forward
(they are data dependencies of both), and are freed before the activation peak of the backward
pass, where `gpu_peak_mb` is reached. Expected change of the logged TF peak: + the label input
only, i.e. <= 0.06 GB at every level, plus allocator-placement noise. Measured peaks today: L4
33.2 GiB, L2 27.8 GiB, L1 38.2-38.9 GB (limit 42500 MB, ~3.6 GB TF headroom). These are config A
runs except L4, where config B (`r4_cfgB_dice1_smooth05_lr3e-4_wd1e-1`) also peaked at ~33.2 GiB.
Config B at L2 / L1 has no measured peak yet (the L2 smoothness sweep runs config B on the
cluster; its peak is the L2 reference once read). Config B removes the group blocks from the stem
and encoder (`models/group_net.py`: `g = cfg.group_config == "A"` gates `GroupConvBlock(...,
group=g)`), which removes group-mean concatenations, so its activation memory is expected to be
at most config A's; the paired p 0 smoke at each level (7.2) is the config B baseline that the
p 0.5 run is compared with, so the synthesis criterion does not depend on this expectation.

Risk: the transient post-process allocations (<= ~0.8 GB at L1 for the linear task) change BFC
placement and could fragment the arena (the level-1 validation-fragmentation OOM history). They
are smaller than one d1 activation and freed in LIFO order, so the risk is low; the L1 smoke
measures it (pass criterion 7.2).

### 6.2 GPU, DALI (outside the TF limit)

Two linear pipelines gain a uint8 label reader + host-to-device copy. Per pipeline about 2-3
batches of labels are in flight (prefetch 1 + pipelined stage): L1 2-3 x 55 MB = 0.11-0.17 GB,
x 2 pipelines = 0.22-0.33 GB; L2 0.11-0.17 GB; L4 0.06-0.08 GB. The deform pipeline is unchanged
(fine replaces train18). Pessimistic L1 bound: applying the measured ratio of DALI memory to
per-batch output bytes (~4.5 GB for ~0.58 GB of outputs at L1, job 10517362) gives ~0.9 GB.
The device headroom left next to the 42500 MB TF arena is not derivable from the available
numbers (the `r_level1.sh` header budgets from 46080 MiB, which would give a negative remainder,
yet the 42500 MB runs succeeded). It is therefore MEASURED: the p 0 L1 smoke logs
`memory.total - max(memory.used)` in `gpu_poll.csv` (5.5), and the p 0.5 criterion is relative
to that measurement (7.2). If that headroom is below the pessimistic 0.9 GB, mitigation (2)
below is applied before the long run.

Mitigations apply ONLY when the paired p 0 smoke of the level passed and the p 0.5 smoke failed.
A failing p 0 smoke is a config B (or recipe) problem, not a synthesis problem: config B has
never run at L1 and is only in Jay's in-progress sweep at L2. Rule 7.2 (stop and report) covers
that case. Mitigations, chosen by WHERE the smoke shows the pressure:
- TF-side failure (OOM inside the 42500 MB arena, or logged `gpu_peak_mb` more than 0.5 GB above
  the paired p 0 run): (1) `--synth-device cpu` (removes the TF-side temporaries, 6.4); rerun the
  L1 smoke.
- DALI/device-side failure (DALI allocation error, cuBLAS/cuDNN handle failure, or
  `gpu_poll.csv` `max(memory.used) > memory.total - 512` while the TF peak is normal): (2)
  `TF_MEM_LIMIT_MB=42000`. This is an UNTESTED setting: at 40000 MB the L1 runs OOMed with logged
  peaks of 38.3-38.6 GB (fragmentation), so TF headroom below ~1.5 GB already failed once, and
  42000 leaves ~3.1 GB above the 38.9 GB peak, which is probably enough but unmeasured. Jay
  approved lowering the limit on measured evidence (answer 8). It is
  used only after the device-memory log shows DALI pressure, and it must pass its own L1 smoke
  (same criteria, both p 0 and p 0.5 at 42000 so the pair stays comparable) before any long run.
  If that smoke fails: (1) as well.
- Last resort, not designed in detail: (3) synthesis on the CPU inside DALI via
  `fn.python_function`, which in DALI 1.18 requires `exec_async=False, exec_pipelined=False` and
  would slow every step.

### 6.3 Host RAM

DALI CPU readers for the two new label streams: 2-3 batches + pinned staging per pipeline:
L1 ~0.3-0.5 GB, L2 ~0.2 GB, L4 < 0.1 GB. Synthesis on GPU uses no host memory; with
`--synth-device cpu` the per-image temporaries (~0.5 GB at L1) are on the host. Validation is
unchanged. Expected L1 MaxRSS 94-95 GB (< 128 GB).

### 6.4 Fallback: `--synth-device cpu`

Same code, the true branch placed on `/cpu:0` inside the same `tf.function` (TF inserts the
GPU<->CPU copies: L1 linear 55 MB labels down, 220 MB images up per synthetic step). The flag and
`tf.cond` stay on the default device. CPU time with 3 threads: estimated 1-3 s per L1 linear
synthetic step (blur tensordots ~8 GFLOP per image, gathers, render); acceptable as a fallback
only. Tested locally (CPU TF).

### 6.5 Compute time (GPU path)

Per synthetic L1 linear step: blur 3 tensordots ~8 GFLOP per image (fp32, TF32 off), gathers and
render ~10 ms per image: < 0.3 s per step vs 2.24 s; deform 4 images < 0.1 s vs 23 s. Only
synthetic steps pay. Estimated wall-time overhead at p 0.5: L4 <= 5 %, L2 <= 5 %, L1 <= 7 %.

## 7. Verification

### 7.1 Tests (local, unittest, `tests.tf_setup` first for TF tests)

New `tests/test_synth.py`:
- shapes/dtypes: uint8 [N, X, Y, Z, 1] in -> float32 [N, X, Y, Z, 1] in [0, 1]; finite; seeded
  determinism (same seed -> bit-equal on CPU).
- one contrast per group: with blur/bias/gamma/noise/jitter off, per-label mean ratios between two
  images of a group are exactly 1; with jitter j, within [1 - j, 1 + j] / [1 + j, 1 - j] bounds.
- background 0 with remask; normalisation max == 1 before noise.
- `fine_to_train18_table()` == mapping derived from `preprocessing.luts` for all 103 ids;
  entries >= 103 are 0.
- no one-hot of fine labels, checked on the FULL graph: the synthesis runs inside `tf.cond` and
  `tf.map_fn` bodies, which are nested FuncGraphs invisible to `graph.get_operations()`. The test
  traces the post-process (and, in `test_trainer.py`, the deform and linear train steps) with
  synthesis on, takes `gd = concrete_fn.graph.as_graph_def()`, and scans `gd.node` plus every
  `gd.library.function[*].node_def`, collecting every `OneHot` node with its depth (the `depth`
  input is a Const, resolved in the same function's node list). Assertion: the multiset of
  OneHot depths with synthesis ON equals the one with synthesis OFF for the same step (same
  level, same flags otherwise), i.e. synthesis adds no one-hot at all. This avoids depending on
  scope names inside nested functions, and it is correct with `--lowres-method tent`, whose
  `tf_intensity_aug._tent_matrix` builds `tf.one_hot(..., n)` with n = 160/224/192 in both graphs
  (the deform label one-hot, depth 19, is also in both). A negative control (a test-only
  synthesis variant that one-hots the fine labels with depth 103 inside a `tf.cond` /
  `tf.map_fn` body) must make the assertion fail, so the scan is proven to see nested functions.
- trace once, drawn path only: the post-process wrapped in a `tf.function` with an
  `input_signature` (images, T[, Rt], labels; no Python arguments), synthesis p = 0.5 and a seed
  known (found once, then fixed in the test) to give both outcomes within 8 calls; call it 8
  times; assert one trace and that `synth` was both True and False. The forced overrides
  (`synth_override=True/False`, used only by the eager recal path) are tested eagerly for
  correctness (next bullets), not for tracing: a Python-bool argument retraces per value by
  design.
- cascade skip on the default-off path (finding v2.1): the existing
  `tests/test_cascade.py::TestSkip::test_graph_and_forced_skip` (L2 cascade: forced
  `skip=True` train steps for rigid/affine/deform equal a no-cascade step, alternating skip never
  retraces) and `test_skip_and_frozen_unchanged` (L1 frozen chain, forced skip) are the guard
  that `body`'s positional `skip` still binds (`TestChainTrainer::test_skip_and_frozen_unchanged`;
  plus `TestStageSkipLevel2` / `TestStageSkipChain` for stage mode); they must pass unchanged. New synth-on twin in
  `tests/test_synth.py`: the same forced-skip comparison with synthesis p = 1 for both trainers
  (same seed), which exercises the new lambdas (`skip` positional, `labels=` keyword).
- real branch identity: `synth_override=False` output is bit-equal to the synthesis-off
  post-process for the same inputs and flip state.
- recal Bresenham counts for K in {1, 16, 64}, p in {0, 0.25, 0.5, 1}.
- recal wiring (in `tests/test_trainer.py`, fixed-source cascade Trainer at L4, tiny group): run
  `recalibrate_bn` with tasks rigid AND affine (and deform) at p = 1; assert every slot's batch
  dict has `synth == True` and that the images differ from the real inputs, and at p 1 with
  `synth_override=False` forced that they equal the synthesis-off post-process. This catches the
  positional-binding bug (affine labels landing in `rt_dali`, rigid labels in `flip_axes`).
- `_pp_call` unit test: with synthesis on, linear inputs `(images, T[, Rt], labels)` reach
  `labels=`; with synthesis off the call is `pp(*inputs)`; a linear call with synthesis on and no
  labels raises ValueError.
- linear render consistency: a synthetic group's image equals `transform_image` of the unwarped
  synthetic image with `T_mm^-1` (tests the wiring), and targets are unchanged by synthesis.
- CPU device path gives the same values as the default path on CPU (in the CPU-only test env both
  run on CPU; equality checks the wiring).
- Default off: the existing golden tests pass unchanged; a new test asserts that
  `LinearPostprocess` / `DeformPostprocess` with `synth=None` create no `synth_generator` and that
  `checkpoint_variables()` keys are unchanged.

Extended tests: `tests/test_postprocess.py` (labels argument), `tests/test_data_loading.py`
(signature/counts with labels, `synth_label_files`, buffer mode refused), `tests/test_trainer.py`
(a 3-task fixed-source run with p 0.5: trace counts 1, `synth_frac` logged, checkpoint round
trip of the synth generator, resume config mismatch exits), `tests/test_runtime_infra.py` / args
(validation table 5.4), `tests/test_slurm_scripts.py` (default command line unchanged; synth
flags present only with SYNTH_PROB), `tests/test_check_mixed_run.py` (synth_frac rule, recal
counts), `preprocessing/tests/test_export.py` (nested mode property on random and adversarial
fragmented-cortex arrays, tie rules, `synthfine` at levels 1/2/4 via `export-npy`, export-check
consistency failure), new `preprocessing/tests/test_add_labels.py` on a tiny fake processed +
export tree built with `export_npy` (train18 only, two chunks -> two parts, merged): phase files
writes exactly the three new files per case and leaves every other file's bytes and mtime
unchanged; rerun = all "kept"; a case whose processed SynthSeg map differs from the one used
for train18 FAILS with nothing written; a pre-existing wrong `labels_synthfine_2.npy` FAILS and
is left alone; phase manifest: only the six `LABEL_COLUMNS` change (old + appended entries), row
and column order unchanged, idempotent on a second run, backup created, `export-merge`
afterwards reproduces the new manifest byte for byte, `manifest_files(rows, k,
label_set="synthfine")` and the default train18 lookup both work; phase manifest refuses (exit 1,
no change) while a case lacks a file; phase manifest with a hand-edited `export_manifest.csv`
(one row's `npy_dir` changed so the unmodified parts no longer merge to it) exits 1 with the
"pre-existing divergence" message naming that (dataset, subject_dir), and every file (parts,
manifest, luts) is byte-identical afterwards and no backup is created; unknown `--label-set`
exits 2. `tests/test_check_mixed_run.py` also covers `--min-task-steps` (a task with 0 traced
or too few executed steps FAILs; enough steps PASS). `tests/test_eval_synth.py` (new): two
presets give distinct set names `synth_<preset>_<set>` and distinct
`Trainer._identity_cache` keys, and their `template_dice_identity` / `ncc_template_identity`
entries differ (8 step 4).

GPU checks (new cases in `tests/gpu/verify_dali_pipeline.py`, run inside the smoke job):
s1 linear pipeline with labels: images and T_dali equal the no-label pipeline for the same seed
(else report; not a correctness issue); labels uint8 aligned with the unwarped image (label > 0
<=> image > 0 on >= 99 % of voxels). s2 TF render of the unwarped real image with `T_mm^-1` vs the
DALI-warped image: interior mean abs diff < 1e-3. s3 DALI process-memory delta with labels at L1
(recorded).

### 7.2 Smokes (cluster; ask Jay first; 2-job limit)

Per level, two paired smokes on the same commit, submitted together (2 jobs = the limit), the
two command lines identical apart from `SYNTH_PROB` and `RUN_NAME`. Each L2 / L1 smoke cascades
from the `ckpt/last` of the smoke of the same p one level up. The L4 smoke overrides the delayed
schedule exactly like the `r_level4.sh` header smoke (deform, curriculum ramp and G from step
100), so all three tasks and G run in 300 steps; without these overrides the defaults
(`DEFORM_START` / `G_START` 3000) would run no deform step. Smoothness values S4 / S2 / S1: the
values Jay fixes for the comparison models (13.1b); if the smokes run earlier, 0.5 (memory and
trace checks do not depend on the weight); both runs of a pair always use the same value.

```
R=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results
S4=0.5; S2=0.5; S1=0.5      # or Jay's final values
cd /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad
# L4 pair (~20-30 min each; deform >= 80 steps expected ~100 = 200 post-start steps x 2/4)
for P in 0 0.5; do T=${P/./}
  RUN_NAME=r4syn_smoke_p$T STEPS=300 VAL_EVERY=100 WARMUP=50 DEFORM_START=100 DEFORM_RAMP=100 \
    CURRICULUM_RAMP=100 G_START=100 G_RAMP=100 PROJECT_STEPS=16000 PROJECT_VAL_EVERY=500 WD=1e-1 \
    SYNTH_PROB=$P GPU_POLL=true MIN_TASK_STEPS=rigid:20,affine:20,deform:80 \
    EXTRA_ARGS="--group-config B --lambda-smooth $S4" \
    sbatch --job-name=r4synsm$T --time=02:00:00 --mem=64G slurm/r_level4.sh
done
# L2 pair, after both L4 smokes finished (~40 min each incl. BN recal ~9 min)
for P in 0 0.5; do T=${P/./}
  RUN_NAME=r2syn_smoke_p$T CASCADE_FROM=$R/r4syn_smoke_p$T/ckpt/last STEPS=200 VAL_EVERY=100 \
    WARMUP=50 PROJECT_STEPS=15000 PROJECT_VAL_EVERY=500 WD=1e-1 \
    SYNTH_PROB=$P GPU_POLL=true MIN_TASK_STEPS=rigid:20,affine:20,deform:40 \
    EXTRA_ARGS="--group-config B --lambda-smooth $S2" \
    sbatch --job-name=r2synsm$T --time=03:00:00 --mem=64G slurm/r_level2.sh
done
# L1 pair, after both L2 smokes finished (~1.5 h each)
for P in 0 0.5; do T=${P/./}
  RUN_NAME=r1syn_smoke_p$T CASCADE_FROM=$R/r2syn_smoke_p$T/ckpt/last STEPS=200 VAL_EVERY=100 \
    WARMUP=50 PROJECT_STEPS=4000 PROJECT_VAL_EVERY=500 LR=3e-4 WD=1e-1 TF_MEM_LIMIT_MB=42500 \
    SYNTH_PROB=$P GPU_POLL=true MIN_TASK_STEPS=rigid:20,affine:20,deform:40 \
    EXTRA_ARGS="--group-config B --lambda-smooth $S1" \
    sbatch --job-name=r1synsm$T --time=04:00:00 --mem=128G slurm/r_level1.sh
done
```

(`T` is `0` or `05`, so the run names are `r<k>syn_smoke_p0` / `r<k>syn_smoke_p05`.) The L2 / L1
thresholds: 200 steps at ratio 1:1:2 give ~50 / 50 / 100 task steps, so 20 / 20 / 40 leaves a
wide margin while still failing a task that never ran.

Rule for a failing p 0 smoke (config B has never run at L1, and at L2 only in Jay's in-progress
sweep): if the p 0 run of a pair fails (OOM, int32 limit, checker FAIL), STOP and report to Jay
as a config B / recipe issue. No synthesis mitigation (6.2) and no further level is run; the
p 0.5 result of that pair is only informative. Mitigations 6.2 (1)-(3) apply only when p 0
passes and p 0.5 fails.

Pass criteria per level: checker PASS incl. `trace_once`, `--min-task-steps` (every task traced
once and executed at least the listed steps) and `--expect-synth-prob` (p 0.5 run); logged
`gpu_peak_mb` of p 0.5 within 0.5 GB of the paired p 0 run; `max(memory.used) <=
memory.total - 512` MiB in `gpu_poll.csv` (device-level, 5.5), and its maximum within 0.5 GB of
the paired p 0 run; sacct MaxRSS < 128 GB (L1; L2 < 64 GB as its script requests); at L2 AND
L1 the planned `bn_recalibration.synth_batches` == {rigid 8, affine 8, deform 16}
(`synth_batches_used` printed); non-finite only in the AMP warm-up as today; step-time overhead
within 6.5. Plus a visual QC: `scripts/synth_preview.py --data <export> --level k --n 8 --seed s
--out <png>` (CPU TF, no training) writes synthetic examples per level for Jay before the smokes.

## 8. Synthetic-contrast evaluation (separate process, H2)

`eval_synth.py` (top level, like `predict.py`) + `runtime/eval_run.py`, SLURM wrapper
`slurm/eval_synth.sh` (rtx8000, 1 GPU, 3 CPUs, `--mem=128G` (fair share; the L1 training process
reached ~94 GB MaxRSS and the 64G jobs were killed during validation), 8 h). ONE job evaluates
BOTH models: `eval_synth.py --runs <real_only_run> <mixed_run> --level 1 --out <dir>`, where
`<real_only_run>` is the L1 run of Jay's final config B real-only cascade (13.1b) and
`<mixed_run>` is `r1_cfgB_synth05`. Both
Trainers stay resident; each preset's synthetic sets are built ONCE in memory and handed to both
(`trainer.val_sets = sets` before `validate()`), so the two models see the same array objects and
cross-node / cross-process determinism of CPU synthesis does not matter (review v2 finding 5).
The val functions are keyed by (task, group size) (`Trainer._val_fn`, `_cascade_val_fn`), so a
new preset with the same group sizes reuses the traced functions (no retrace per preset).
The synthetic sets are built once per distinct validation spec (level, val group sizes, val
seeds, taken from each run's TrainConfig); with the r_level1.sh defaults both runs share one spec,
and a run with a different spec gets its own build (recorded in the JSON). sha256 of every set
is still written (audit). `sacct MaxRSS` is
not final while the job runs, so the script records the process peak itself
(`resource.getrusage(RUSAGE_SELF).ru_maxrss`) in `eval_synth.json` after every preset, the
wrapper appends `sstat -j $SLURM_JOB_ID.batch --format=MaxRSS` at the end, and the post-job
`sacct MaxRSS` is recorded in the eval report.
Expected host RAM: one real validation set (as in training) + one synthetic set at a time, so
about the training process's validation share (< 94 GB) and well below 128 GB.

Steps:
1. Read `<run>/config.json` with the same path rule as `inference.loader._read_config`
   (`C.source_config_path`). Rebuild the Namespace from `args`, normalising every `*_parsed`
   value recursively: JSON turns int dict keys into strings (e.g. `cascade_skip_stage_parsed
   {task: {"4": p}}`), so keys that are decimal strings are converted back to int; then
   `argparse.Namespace(**args)`. Fields missing from older configs (all `synth_*` for the
   real-only model) are filled by `getattr` defaults in `synth_config_from_args` (-> None = off).
   `A.train_config_from_args(ns)` with val PNGs off. A unit test rebuilds the TrainConfig of a
   saved config with `cascade_skip_stage_parsed` and checks equality with the original.
2. Per run (steps 1-3 run for each of the two runs): build the model and the frozen stages exactly like the resume branch of `main.main`
   (`GroupRegLevelModel(...).build_model`, `FrozenChain.from_config` / `FrozenStage.from_config`),
   a Trainer with placeholder sources (`next()` raises), and restore `<run>/ckpt/last` (Jay's
   rule: last checkpoint) read-only through `C.RunCheckpoint`; re-apply BN freeze from the saved
   record. Nothing is written into the run directory; output goes to `--out`.
3. Sanity gate: `trainer.validate()` on the REAL val sets (built as in `main._build_dali_data`)
   must reproduce the run's logged validation at the checkpoint step (from `metrics.jsonl`)
   within 1e-2 mm (every mm metric) / 1e-3 Dice; the measured max diffs are always written to
   `eval_synth.json` (`sanity_gate: {max_mm_diff, max_dice_diff, pass}`). Larger diffs -> exit 5
   (restore or config mismatch). The tolerances allow cross-process cuDNN autotuning and fp16
   frozen-stage differences; a restore or config error moves metrics by much more (mm-scale).
4. For each preset (one at a time, freed before the next, host RAM ~ one real validation): build
   synthetic val sets from the val `synthfine` labels on the CPU (deterministic) with a fixed seed
   per (preset, group, image). Group composition is decided inside `ValidationSet.build` /
   `DeformValidationSet.build` (seeded permutation), so one contrast per group needs a hook: both
   builders get an optional `group_images_fn(g, idx) -> float32 [Nv, X, Y, Z, 1]` (default None =
   `images[idx]`, today's code, so in-training validation is unchanged). The eval script passes a
   function that synthesises the group's images in template space from `fine_labels[idx]` with
   the contrast of (preset, g); the linear builder then renders them with its seeded transforms
   (incl. the G sets); the deform builder pairs them with the train18 val labels. Run
   `trainer.validate()` with these sets. Every synthetic set is NAMED `synth_<preset>_<real set
   name>` (e.g. `synth_t2_like_deform`, `synth_t2_like_affine_g`, `synth_random_0_rigid`), so the
   metric keys are `val_synth_<preset>_<set>/...` and, because `Trainer._identity_cache` and
   `_frozen_cache` are keyed by (set name, n_groups) (run.py ~1526 / 1551 / 1577), the identity
   LNCC/Dice and frozen-stage baselines are computed per preset and never reused from the real
   gate or from another preset. Unit test in `tests/test_eval_synth.py` (7.1).
5. Write `eval_synth.json` (all metrics per run and preset, the shared-set flag per preset,
   sha256 of every synthetic set) and a CSV with the paired table (real-only vs mixed per preset
   and metric). `scripts/compare_synth_eval.py eval_synth.json` re-prints the table and, when
   given two JSONs from separate jobs (fallback only), asserts equal sha256 and reports a
   mismatch as an error naming the preset.
GPU memory: two L1 cascades' weights (< 1 GB together) plus one validation pass at a time
(L1 val peak ~22 GB in training jobs), no optimizer state, no training activations: well under
48 GB. Host RAM: real val sets once per spec + one preset's synthetic sets at a time + two models
(< 128 GB). Time: real-set gate ~2 x 10-20 min (incl. the first trace), then ~2 x 8 min per
preset x 11 presets -> ~3.5-4.5 h; limit 8 h.

Presets (11; `data_loading/synth.py EVAL_PRESETS`; train18-class means plus DK-parcel overrides,
per-image jitter as in training): `random_0..3` (training distribution, 4 seeds), `t2_like` (CSF
bright, WM dark), `flair_like`, `fdg_like` (GM > WM, FWHM 5 mm), `amyloid_pos` (cortex ~ WM high,
cerebellar cortex low, FWHM 6 mm), `amyloid_neg`, `tau_like` (low cortex, medial/inferior temporal
parcels high, FWHM 6 mm), `t1_like` (sanity, close to the real contrast). Both models (real-only
and mixed) at level 1 (the full cascade); L4 and L2 runs optional (cheap).

## 9. PET evaluation outline (paper experiment, later workflow)

Goal: show that fixed-baseline rigid registration biases longitudinal SUVR because only follow-ups
are resampled, and that groupwise registration (every scan resampled once into the central space)
removes the asymmetry.

1. Simulation (pure interpolation bias): one PET P0 (native space). EVERY copy, the baseline
   included, gets one random non-identity rigid transform: P_k = resample(P0_src, T_k) for
   k = 1..K (rotations ~2-10 deg, translations ~2-10 mm, each with a non-zero sub-voxel part), so
   every simulated scan has exactly one prior interpolation, as native acquisitions do relative
   to each other. If copy 1 were P0 itself, the simulation would create a smoothing difference
   between time points that mixes with the method's resampling asymmetry. P0_src: default P0
   upsampled 2x (cubic) and blurred by the scanner PSF residual (Gaussian ~2 mm FWHM), then
   resampled to the native grid by T_k, so the copies are not resampled from the exact grid used
   for the truth; the plain-P0 variant is reported as a sensitivity check. Truth: no change;
   reference SUVR per ROI computed on P0 with the ROIs, and SUVR_k^truth = that value for every k
   (ROIs carried to each copy by the known T_k for the error analysis). Methods: (a) fixed baseline = FLIRT 6 DOF and ANTs
   rigid to copy 1, follow-ups resampled, baseline not; (b) FreeSurfer `mri_robust_template`
   (rigid, unbiased midspace, all resampled); (c) ours (`predict.py --branches rigid`, all images
   resampled once from native space by composing the preprocessing affine with the predicted
   transform). Metrics: rigid error vs the known T_k (rotation deg, translation mm), SUVR per ROI
   (cortical composite / Braak regions; reference cerebellar grey) per time point; error =
   SUVR_k - SUVR_k^truth per method and the apparent change SUVR_k - SUVR_baseline (fixed
   baseline expected: baseline exact-ish, follow-ups biased, so a spurious change; symmetric
   methods ~equal across k). Order test: rerun the fixed-baseline methods with baseline = first
   and baseline = last copy; the spurious change should flip sign/location with the order, while
   groupwise methods are order-invariant (input permutation).
2. Real longitudinal data (Jay's answer 7; data not ready yet): a small ADNI subset, each
   subject with one structural T1, a baseline PET and a follow-up PET (amyloid and/or tau). The
   PET pair is the group (N = 2, single modality); the T1 is NOT in the group. ROIs: SynthSeg on
   the T1, brought to PET by one T1 -> PET rigid coregistration per subject (FLIRT 6 DOF to the
   baseline PET for the fixed-baseline methods; to the groupwise midspace for the groupwise
   methods, so every method's ROIs reach its PET space with one transform). Reference region:
   cerebellar grey matter (SynthSeg cerebellar cortex labels). Per method: SUVR per ROI at both
   visits and the change; order dependence (baseline = first vs = follow-up) for the
   fixed-baseline methods vs order invariance of the groupwise methods; inverse consistency of
   the pairwise map. The simulation of step 1 is run on these subjects' baseline PETs too.
3. Preprocessing for our model: skull-strip (SynthStrip on PET or the T1-derived mask), rough
   MNI alignment, one resampling to the LUMIR grid for prediction only; final resampling from
   native space once. ROIs from the subject's T1 SynthSeg (step 2) or carried by the known
   transforms in the simulation.
4. Both models (real-only and mixed), no TTA. 8-flip TTA (milestone 10) is future work in a
   separate workflow and out of scope here.

## 10. Retraining plan and cost

The mixed model follows CONFIG B (Jay's answer 4). Where config B is defined: `--group-config
B` (`runtime/args.py` model group, `models/config.py` `GROUP_CONFIGS`; 'B' = group blocks only
after the bottleneck, gated in `models/group_net.py`). On real data it was run only via
`EXTRA_ARGS` (the r_level*.sh scripts have no GROUP_CONFIG variable): L4
`results/r4_cfgB_dice1_smooth05_lr3e-4_wd1e-1` (job 10543287, `EXTRA_ARGS='--group-config B'`,
r_level4.sh defaults + `WD=1e-1`; best L4 so far) and the in-progress L2 smoothness sweep
`results/r2_cfgB_s15000_smooth{S}_lr3e-4_wd1e-1_n32_d16` (cascading from that L4). No config B
L1 exists. `C.check_cascade_config` requires identical ModelConfigs across the cascade, so all
three mixed levels must use config B. Everything else equals the recipe of Jay's final config B
real-only cascade (13.1b), including its smoothness weight per level, which is passed as a
parameter (`--lambda-smooth $S4 / $S2 / $S1`). The mixed model is trained only after Jay has
fixed these values.

Mixed-model commands (each after Jay's approval; 2-job rule; `R` and `S4/S2/S1` defined in the
first lines, the S values are Jay's final real-only values):

```
R=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results
S4=<Jay's L4 value>; S2=<Jay's L2 value>; S1=<Jay's L1 value>
cd /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/brain-group-rad
# L4 (16000 steps, ~14.5-15 h)
RUN_NAME=r4_cfgB_synth05 WD=1e-1 SYNTH_PROB=0.5 EXTRA_ARGS="--group-config B --lambda-smooth $S4" \
  sbatch --job-name=r4syn --time=24:00:00 --mem=64G slurm/r_level4.sh
# L2 (15000 steps, ~21-22 h)
RUN_NAME=r2_cfgB_synth05 STEPS=15000 WD=1e-1 SYNTH_PROB=0.5 \
  CASCADE_FROM=$R/r4_cfgB_synth05/ckpt/last EXTRA_ARGS="--group-config B --lambda-smooth $S2" \
  sbatch --job-name=r2syn --time=36:00:00 --mem=64G slurm/r_level2.sh
# L1 (4000 steps, ~15 h)
RUN_NAME=r1_cfgB_synth05 STEPS=4000 LR=3e-4 WD=1e-1 TF_MEM_LIMIT_MB=42500 SYNTH_PROB=0.5 \
  CASCADE_FROM=$R/r2_cfgB_synth05/ckpt/last EXTRA_ARGS="--group-config B --lambda-smooth $S1" \
  sbatch --job-name=r1syn --time=36:00:00 --mem=128G slurm/r_level1.sh
```

If Jay's real-only recipe differs in other settings (steps, Dice/LNCC weights), the mixed
commands copy them; the only intended difference between the two cascades is `SYNTH_PROB`.

The script defaults supply the rest: lr 3e-4, AdamW, d12hs widths, gradnorm 12, Dice + LNCC
1/1/1/1, smoothness 0.5, TF aug + tent, G 45 deg / 10 mm, group sizes 128/64, 32/16, 8/4; L4
16000 steps with deform from 3000 and the 3000-step curriculum / G ramps, fp16 decoder;
L2 fp16 decoder, BN recal 64 + freeze false, skip deform 4:0.15; L1 float32_fine decoder, BN
recal 64 + freeze true, pair recompute, lean losses, skips 4:0,2:0.15 / 4:0.15,2:0.15.
Smoothness: per level from Jay's final real-only config B recipe (13.1b).
Sequential (each level is the next level's `CASCADE_FROM`, last checkpoint):

| Step | Wall time (RTX 8000) |
|---|---|
| `export-add-labels` (Jay, basic-partition array, 7 x 500 rows two at a time + manifest phase) | ~2.5-4.5 h |
| Preview PNGs + paired smokes L4 / L2 / L1 | ~0.5 h + ~1 h + ~1.5 h (+ queue) |
| L4 16000 steps | ~14.5-15 h (r4cfgB0 14h35 real-only + <= 5 %) |
| L2 15000 steps | ~21-22 h |
| L1 4000 steps | ~15 h |
| Synthetic eval, both models in one job, L1 | ~4-5 h |
| Total | ~2.2 days of sequential training + ~0.5 day smokes/eval |

Comparison model: Jay's own final config B real-only cascade (13.1b), trained by Jay with the
same recipe and smoothness. Real-only and mixed then differ only in `SYNTH_PROB`; no level-1
reference model is part of this design (the memory gates use the paired p 0 / p 0.5 smokes).

## 11. Error handling summary

| Operation | Failure | Recoverable? | Caller gets | Logged |
|---|---|---|---|---|
| Flag parsing | out of range, incompatible flags | no | parser.error, exit 2 | stderr |
| Manifest lookup | case lacks `labels_synthfine*` | no (export first) | ValueError at startup naming the case | traceback |
| Exporter nested mode | non-integer map, negative ids, shape not divisible | no | ValueError (existing messages) | log.error |
| Exporter unknown raw id | id not in fixed LUT | no | ValueError (existing `apply_lut`) | log.error |
| export-check consistency | `g(synthfine_k) != train18_k` or id > 102 | no | listed failure, exit 1 | stdout |
| export-add-labels files: missing source, unknown raw id, shape differs from `shape_l1`, `table[f_k] != train18_k`, existing synthfine file differs or has wrong dtype/shape | per case | no for that case (others continue) | case FAILED in the chunk status JSON, nothing written for it, exit 1 at the end | log.error per case + stdout summary |
| export-add-labels manifest: a case lacks a file or fails the consistency re-check | data incomplete | no | exit 1 before any manifest change | stdout list |
| export-add-labels manifest pre-check: unmodified parts do not merge to the current manifest | pre-existing divergence (not this command) | no (Jay decides) | exit 1 with the differing (dataset, subject_dir) keys, nothing changed, no backup | stdout list + log.error |
| export-add-labels manifest: merge reproduction or diff check fails (after a passing pre-check) | bug | no | originals restored from the backup, exit 1 | log.error |
| Smoke: a task never traced / too few executed steps | schedule or wiring | no | checker FAIL (`--min-task-steps`) | checker output |
| Smoke: the p 0 run of a pair fails | config B / recipe issue | no | stop, report to Jay, no mitigation, no next level | SLURM log + checker |
| export-add-labels: `--label-set` other than synthfine, `--start/--end` with phase manifest | usage | no | exit 2 | stderr |
| Synthesis numeric | empty foreground / quantile 0 (empty labels) | yes | `tf.cond` on the subsample size + `max(hi, 1e-6)` -> zero image; step stays finite | none |
| Linear post-process called without `labels=` while synthesis is on | wiring bug | no | ValueError at trace time | traceback |
| Non-finite synthetic image | should not happen | yes | existing `in_finite` guard zeroes images, step skipped and counted (per-task guard) | existing WARNING |
| DALI label reader misaligned | readers drift | no | caught by GPU check s1 / l (smoke) | check output |
| Resume with different SynthConfig | config mismatch | no | SystemExit with both configs | stderr |
| Eval sanity gate | real-val metrics differ from the log | no | exit 5 | eval json + stderr |
| GPU OOM in smoke | memory | via fallbacks 6.2 | job failure | SLURM log |

Invariant ownership: label consistency (`LUT(fine) == train18`) is owned by the exporter
(construction) and verified by export-check, because it is a data property; one contrast per group
is owned by `draw_synth_params` (group tables drawn once per call); default-off identity is owned
by Python trace-time switches (`synth is None` builds nothing), the same pattern as the TF aug.

## 12. Compatibility notes

Python 3.8: no PEP 585/604 annotations, no `dict |`, no `str.removeprefix`; dataclasses OK.
TF 2.10: `tf.gather`, `tf.tensordot`, `tf.map_fn(fn_output_signature=...)`, `tf.cond`,
`tf.random.Generator` (`uniform`, `normal`, `make_seeds`), `tf.random.stateless_normal`,
`tf.math.pow`, `tf.math.top_k` (dynamic k), `tf.boolean_mask`, `tf.device` inside `tf.function`
all exist; keyword-only arguments (`*,`) are Python 3 syntax; explicit float32 (the mixed_float16
Keras policy does not affect raw ops). GPU `tf.gather` with a 256-entry table and uint8-derived
indices can never go out of range. DALI 1.18: no new operator. numpy 1.21: `bincount`, `argmax`,
fancy indexing only.

## 13. Decisions and open questions

### 13.1 Jay's decisions (2026-11, binding; v2 questions 1-10 closed)

1. Export: the cluster `npy_v2` tree has per case only the three images and the three train18
   label files; the original `export-npy` command is unknown. -> add-only `export-add-labels`
   driven by `export_manifest.csv`, verified against the train18 files (3.3, 3.4). ~27 GB extra
   disk approved.
2. Per-image jitter half-width 0.08: approved (`--synth-jitter 0.08`).
3. Blur families MR U(0, 3) mm / PET U(4, 8) mm, 50/50 within synthetic groups: approved.
4. The mixed model uses config B at all levels: recipe and commands in section 10.
5. (SUPERSEDED by 13.1b) Real-only L1 reference `r1sw_lr3e-4_wd1e-1_n8_d4`: no longer used.
6. 8-flip TTA: out of scope, future workflow (removed from section 9 except as future).
7. PET data: small ADNI subset (T1 + baseline PET + follow-up PET per subject), reference region
   cerebellar grey matter (section 9 step 2).
8. Lowering the L1 TF limit (e.g. 42000 MB) on measured RTX 8000 evidence: approved, smoke gate
   kept (6.2).
9. Contrast structure: hierarchical only (class mean + parcel deviation, L/R shared); the
   independent branch and `hier_prob` are removed (4.1, 4.3, 4.4).
10. Normalisation: robust in-mask p99.5 (4.2 step 5).

### 13.1b Jay's later answer (added by the orchestrator; supersedes answer 5 and closes open questions A and B)

The design needs NO level-1 reference model. Memory gates use only the paired SYNTH_PROB 0 vs 0.5 smokes
(7.2). The real-only comparison model for the paper is Jay's own final config B real-only cascade, trained
with the smoothness weight he picks from his ongoing sweep (he is raising it to reduce cortical folding). The
mixed model uses the same config B recipe and the same smoothness weight(s) at every level, passed as
parameters (`--lambda-smooth`), and is trained only after Jay fixes them. Remove `r1sw_lr3e-4_wd1e-1_n8_d4`
as a comparison model and drop the config A vs B confound caveat.

### 13.2 Open questions for Jay

A, B: closed by 13.1b (comparison model = Jay's final config B real-only cascade; smoothness per
   level from that recipe, passed as `--lambda-smooth $S4/$S2/$S1`).
C. Run names in section 10 (`r4_cfgB_synth05`, `r2_cfgB_synth05`, `r1_cfgB_synth05`) and the
   smoke names `r<k>syn_smoke_p{0,05}` (7.2): OK?
D. The final values S4 / S2 / S1 and the run directory of your real-only config B L1 (the
   first `--runs` argument of `eval_synth.py`, section 8): please send them when fixed. The
   smokes may run before that with 0.5 (7.2).
E. Smoke timing: run the paired smokes now (with S = 0.5, to find config B / synthesis memory
   issues early), or only after the smoothness values are fixed?

## 14. Out of scope

Running any job; per-image different contrasts within a group; one-hot of fine labels; new
in-training validation; synthetic eval inside training; label sub-splitting (B1); independent
per-label (non-hierarchical) contrasts (B2); TTA (future workflow); the PET pipeline
implementation; changing the existing `synthseg` set; changing `export-npy` resume semantics;
r_level1.sh default `--mem` / TF limit change (separate TODO); a GROUP_CONFIG variable in the
r_level*.sh scripts (config B is passed via `EXTRA_ARGS`, as in Jay's runs).

## 15. Changelog

### v2 (response to review v1, CHANGES_REQUESTED, 0 HIGH / 2 MEDIUM / 9 NIT)

1. MEDIUM, positional labels break BN recal / linear wiring: addressed. `labels` and
   `synth_override` are keyword-only (5.2); one `Trainer._pp_call` helper used by both step
   bodies and `_recal_input` (5.3); the four linear lambdas are listed explicitly; a linear call
   without `labels=` while synthesis is on raises at trace time; new tests: recal with rigid and
   affine at p = 1 and a `_pp_call` unit test (7.1). Section 1 consequence (iv) corrected.
2. MEDIUM, asymmetric PET simulation: addressed (9 step 1). Every copy incl. the baseline gets
   one non-identity rigid transform from P0_src (default 2x-upsampled + PSF-blurred P0; plain
   P0 as sensitivity check); truth SUVR from P0 with ROIs carried by the known T_k; order test
   kept.
3. NIT, config/resume/eval for runs without synth keys: addressed (5.4, 8 step 1). `synth`
   written only when p > 0; missing key = off; `getattr` defaults; eval normalises int keys of
   `*_parsed` dicts and uses the `inference.loader` config path rule; unit test added.
4. NIT, vacuous one-hot test: addressed (7.1). Scan of `as_graph_def()` incl. the function
   library; assertion = OneHot depth multiset is identical with synthesis on and off (robust to
   the tent `intensity_aug` one-hots and to scope names in nested functions); negative control
   proves the scan sees nested functions. (Chosen over the reviewer's scope-based rule because
   name scopes are not reliably carried into cond/map_fn function names.)
5. NIT, bincount memory: addressed (3.2 step 1). int64 figures stated (0.71 / 0.09 GB), immediate
   cast to int16, transient ~0.9 GB at k = 2.
6. NIT, export rerun must match the original: addressed (3.4). Exact original filters/chunks/sets
   plus `synthfine`; delete stale parts if boundaries change; pre-flight check in `export-npy`;
   open question 1 asks for the original command lines.
7. NIT, GPU_POLL: addressed (5.5, 7.2). Device-level `--query-gpu=timestamp,memory.used,
   memory.total` for the allocated GPU, run on the host side; criterion
   `max(memory.used) <= memory.total - 512` MiB.
8. NIT, untested 42000 MB fallback: addressed (6.2, open question 8). Mitigations ordered by where
   the pressure shows; 42000 used only after device-memory evidence of DALI pressure and only
   after its own paired L1 smoke passes.
9. NIT, eval gate tolerance / host RAM: addressed (8). Gate 1e-2 mm / 1e-3 Dice with recorded
   diffs; `--mem=128G`; in-process ru_maxrss per preset + sstat/sacct MaxRSS recorded.
10. NIT, max normalisation: addressed (4.2 step 5). Robust in-mask 99.5 % quantile via `top_k` on a
   stride-4 foreground subsample, then clip, matching `normalize_intensity`; open question 10.
11. NIT, independent L/R and parcel contrasts: addressed (4.3, 4.4, SynthConfig `hier_prob`,
   `hier_dev`). Per-group 50/50 mixture of independent and hierarchical (train18 class mean,
   L/R shared, + U(-0.1, 0.1) parcel deviation) means; open question 9.

### v3 (response to review v2, CHANGES_REQUESTED, 0 HIGH / 1 MEDIUM / 6 NIT; plus Jay's binding answers 13.1)

Review v2 findings:
1. MEDIUM, new linear `body` signature breaks the off-path cascade lambdas: addressed (5.3).
   `body(images, t_dali, rt_dali=None, skip=None, *, labels=None)` keeps today's positional
   order (run.py 1023), so today's four lambdas (run.py 1113-1121) bind `skip` exactly as now;
   `labels` is keyword-only and last; the four synthesis-on lambdas pass `labels=labels` by keyword.
   Guards named in 7.1: `TestSkip::test_graph_and_forced_skip`,
   `TestChainTrainer::test_skip_and_frozen_unchanged`, `TestStageSkipLevel2`, `TestStageSkipChain`
   (unchanged, must pass), plus a new synthesis-on forced-skip twin.
2. NIT, L2 also recalibrates: addressed (5.3, 7.2, 10). Text now says L2 (64, freeze false) and
   L1 (64, freeze true) both recalibrate with the same slot rule; the L2 smoke checks planned
   `synth_batches` {8, 8, 16}; section 10 lists L2 "BN recal 64 + freeze false".
3. NIT, config.json not byte-identical: addressed (5.4). Claim replaced: no top-level `synth`
   record by default; `args` gains the four `synth_*` defaults like every new flag.
4. NIT, trace-once test retraces with Python-bool overrides: addressed (7.1). Trace-once is
   tested on the drawn path only (input_signature, p 0.5, fixed seed with both outcomes, 8 calls);
   forced overrides are tested eagerly for correctness.
5. NIT, cross-process sha256 fragility: addressed (8). One job evaluates both models with both
   Trainers resident; each preset's sets are built once in memory and given to both; sha256 kept
   for audit; the two-JSON comparison remains only as a fallback.
6. NIT, `synth_batches` planned vs used: addressed (5.3, 5.6, 7.2). `synth_batches` = planned
   `floor(K_t p)` (checked), `synth_batches_used` = used slots (INFO).
7. NIT, headroom arithmetic in 6.2: addressed. The derived "~1 GB left" was removed; headroom is
   measured by the p 0 L1 smoke's `gpu_poll.csv` and the p 0.5 criterion is relative to it.

Jay's answers (13.1):
- 1 (export): v2's rerun-with-original-filters plan, `_items_to_write` and the `export-npy`
  pre-flight are dropped. New add-only `export-add-labels` (3.3) driven by `export_manifest.csv`:
  files phase (array-safe, writes only `labels_synthfine{,_2,_4}.npy`, verifies
  `table[f_k] == train18_k` against the existing files before writing) and manifest phase (backup,
  append-only edit of the six label columns (corrected in v4) in parts + manifest + luts, `merge_parts`
  reproduction check, diff check with restore). Exact commands in 3.4; tests in 7.1.
- 2, 3, 8, 10: recorded as approved (13.1); 6.2 notes the approval for 42000 MB.
- 4 (config B): located (`--group-config B`, `models/config.py`, `models/group_net.py`; runs
  r4cfgB0 and the config B L2 sweep), explicit commands for all three levels in section 10;
  memory note for config B in 6.1; smokes use config B (7.2); config A vs B confound noted and
  asked (open question A); smoothness choice asked (B).
- 5: real-only L1 reference recorded (13.1, section 10 caveat).
- 6: TTA removed from section 9 (future only) and listed out of scope.
- 7: section 9 step 2 rewritten for the ADNI subset (T1 + two PETs, PET pair = group, T1 for
  SynthSeg ROIs, cerebellar grey reference).
- 9: hierarchical contrasts only; `hier_prob` and the independent branch removed (4.1, 4.3,
  4.4); class-shared stds; independent means backlogged as B2.

### v4 (response to review v3, CHANGES_REQUESTED, 0 HIGH / 1 MEDIUM / 7 NIT; plus Jay's answer 13.1b)

Review v3 findings:
1. MEDIUM, L4 smoke under the section 10 recipe runs no deform step: addressed (7.2). Exact
   command lines for the L4, L2 and L1 pairs; the L4 pair uses the `r_level4.sh` header smoke
   overrides (`DEFORM_START=100 DEFORM_RAMP=100 CURRICULUM_RAMP=100 G_START=100 G_RAMP=100`).
   New checker option `--min-task-steps` (5.6, env `MIN_TASK_STEPS` in 5.5): every listed task
   must be traced once and executed at least rigid 20 / affine 20 / deform 80 steps (L4; expected
   ~100) or deform 40 (L2/L1). Tests in `test_check_mixed_run.py`; error table rows added.
2. NIT, five vs six label columns: addressed (3.3 step 6, 7.1, v3 changelog). The allowed-diff
   set is defined as `LABEL_COLUMNS = ("labels", "label_dtypes", "label_levels") +
   labels_l<k> for k in ALL_LEVELS` (six names).
3. NIT, no concurrency cap on the export array: addressed (3.3, 3.4, 10). `--array=0-..%2`,
   header default `#SBATCH --array=0-0%2`, manifest phase also `--array=0-0%2` and only after the
   array finished; time ~2-4 h for the files phase, table updated.
4. NIT, synthetic set names vs Trainer caches: addressed (8 step 4). Sets are named
   `synth_<preset>_<real set name>`, so `_identity_cache` / `_frozen_cache` keys are per preset;
   unit test in `tests/test_eval_synth.py` (7.1).
5. NIT, preset count: addressed (8). 11 presets; time estimate recomputed (~3.5-4.5 h).
6. NIT, `$R=$RESULTS`: addressed (7.2, 10). Command blocks start with
   `R=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/results`.
7. NIT, pre-existing parts/manifest divergence: addressed (3.3 manifest step 0, error table, 7.1).
   A read-only pre-check merges the unmodified parts and requires byte equality with the current
   manifest; failure exits 1 with "pre-existing divergence ..." and the differing keys, nothing
   changed; test case added.
8. NIT, config B never run at L2/L1: addressed (6.2, 7.2, error table). A failing p 0 smoke means
   stop and report to Jay (config B / recipe issue); mitigations 6.2 (1)-(3) apply only when p 0
   passes and p 0.5 fails.

Jay's answer 13.1b (no L1 reference model; comparison = Jay's final config B real-only cascade;
same smoothness passed as parameters): section 10 commands pass `--lambda-smooth $S4/$S2/$S1` at
every level and state that only `SYNTH_PROB` differs from Jay's real-only recipe; the config A vs
B caveat and `r1sw_lr3e-4_wd1e-1_n8_d4` as comparison model are removed (13.1 item 5 marked
superseded); section 8 names the real-only run as Jay's final config B L1; open questions A and B
closed, D (final S values and real-only run dir) and E (smoke timing) added.