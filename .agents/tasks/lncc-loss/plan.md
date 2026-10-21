# Implementation Plan: LNCC image-similarity loss for the deformable task

Goal: the deform task can train on LNCC (intensities) + soft Dice (labels) + smoothness. All new
weights default to 0, so deform-only, three-task and linear-only runs stay bit-identical.

Ground rules for the coder
- Work directly in `d:\Research\Brain_Registration_Projects\brain-group-rad` (no worktree). Do NOT
  commit, push, ssh or sbatch: two cluster jobs run from `b35bb54`; the orchestrator owns git and
  the cluster. Never copy or commit anything from `reference/` (gitignored, local-only).
- Activate the `shell-and-remote-jobs` skill before running WSL commands. Put multi-line commands
  in small scripts under `.agents/tasks/lncc-loss/` and run them as
  `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/.agents/tasks/lncc-loss/<script>.sh ...`.
- Python 3.8 syntax only (no PEP 585/604 at runtime, no 3.9 stdlib), numpy 1.21 / scipy 1.4 APIs.
  `runtime/args.py` must stay TF-free. Every new TF test module imports `tests.tf_setup` first.
- Helper scripts (create in item 0):
  - `.agents/tasks/lncc-loss/run_some.sh`: a copy of `.agents/tasks/milestone-6/run_some.sh`
    (tf_gpu, `TF_CUDNN_USE_AUTOTUNE=0`, workspace 512 MB, `BGR_TF_GPU_LIMIT_MB=6000`); usage
    `run_some.sh tests.test_x[.Class[.test]] ...`.
  - `.agents/tasks/lncc-loss/py38_check.sh`: adapted from `.agents/tasks/milestone-7/py38_check.sh`
    (bgr_preproc38: py_compile + `ast.parse(feature_version=(3, 8))` on every changed .py file,
    `import runtime.args` without TF, `main.py --help` contains `--lambda-ncc-template`,
    `bash -n` on the new .sh scripts).

## Design decisions (record these in design.md v3.9, item 10)

Reference (`reference/2_deformable_groupwise_registration_src/runtime/losses.py`,
`fast_nlcc_metric` / `Joint_NLCC_Grad_Loss`): window 9 (module constant), box filter = a Keras
`Conv3D` with a constant-ones 9x9x9 kernel (groups=5) on the stacked [I, J, I², J², IJ], sums
normalised by k³, `cc = cross² / (I_var J_var + 1e-7)` on sums (squared NCC, in [0, 1]), mean over
voxels where I != 0 or J != 0 (boolean gather), loss = −cc, template = inclusive mean of all
moved images with gradients, paired with a bending-energy-only smoothness. What we keep and change:

- D1 Formula: KEEP the squared form cc = cov² / (var_a var_b + ε) (VoxelMorph / reference; bounded
  [0, 1], no sqrt, so gradients stay finite on flat regions). CHANGE: compute local MEANS (box sum /
  per-voxel window count), not raw sums / k³, so border windows are normalised by the true count and
  ε has a fixed meaning in intensity² units; clamp var ≥ 0 (float32 cancellation). A `signed=False`
  function argument gives cov / sqrt(var_a var_b + ε) for tests/ablations only (no CLI flag).
- D2 ε = `NCC_EPS = 1e-7` (means scale, intensities in [0, 1]): equals a local std product of 3e-4,
  well below brain-tissue contrast at 4 mm, and bounds |∂cc/∂cov| ≤ 1/sqrt(ε) ≈ 3e3. (The
  reference's 1e-7 on k³-scaled sums is ≈ 2e-13 in means units: effectively no guard.)
- D3 Box filter: separable `tf.nn.conv3d` with constant ones kernels [k,1,1,1,1], [1,k,1,1,1],
  [1,1,k,1,1] (SAME zero padding), quantities stacked along the BATCH axis as [5B, X, Y, Z, 1].
  Chosen over cumsum (same-size intermediates plus pad/shift bookkeeping) and over a dense k³
  kernel (125 vs 15 MACs per voxel at k = 5). No Keras layer: the model gains NO variables
  (checkpoints, warm start, n_vars_updated unchanged). Window counts = box_sum of ones, a constant
  per (shape, k). Memory at level 4, N = 16: one [5N, 40, 56, 48, 1] float32 tensor is 34 MB, so a
  handful of intermediates plus backward is ~0.3-0.5 GB total (~20-30 MB per image vs ~2.15 GB per
  image today, ~1 %). Item 7 measures it.
- D4 Window: CLI `--ncc-window` (odd, ≥ 3), default None → per level `NCC_WINDOW_DEFAULT = {4: 5,
  2: 7, 1: 9}` in `data_loading/params.py` (TF-free). Level 4: 5 voxels = 20 mm, 125 samples per
  window (enough for a stable local variance); the reference's 9 voxels would be 36 mm on a 40x56x48
  grid, nearly global. Level 2: 7 (14 mm); level 1: 9 (9 mm, the VoxelMorph/reference value at 1 mm).
- D5 Groupwise structure, mirroring the Dice terms:
  - Template term: W_i = warp(image_i, u_i) (linear, zero fill). Template = LEAVE-ONE-OUT mean
    T_{-i} = (Σ_j W_j − W_i) / (N − 1), gradients through it (no stop_gradient, like the Dice
    template). Reason: with the inclusive mean the self term gives a floor of ≈ 1/N in squared LNCC
    even for unrelated images (cov(W_i, T) ≈ var/N, var(T) ≈ var/N): 0.25 at N = 4 (all local tests
    and the overfit), 0.06 at N = 16, and it rewards each image for staying put. LOO costs one
    subtraction. Function arg `leave_one_out=True`; no CLI flag.
  - Pair term: on the SAME sampled pairs as the pairwise Dice, moved_p = warp(image_i, disp_p) with
    disp_p = `diffeo.pairwise_displacement(u, u_inv, i, j)` computed ONCE and shared with the pair
    Dice (no extra integration); vs image_j.
  - Both terms are kept: the template term is the groupwise objective (all N images every step), the
    pair term gives sharp single-subject targets early, when the mean template is blurred.
    Recommended experiment values: `--lambda-ncc-template 1 --lambda-ncc-pair 1` (same 1:1 as the
    Dice terms); CLI defaults stay 0.
  - Degenerate solutions: none found. Warping everything into a flat/empty region drives cc → 0
    (loss → 1, the maximum), and the per-level SVF centring (mean_i v_i = 0) rules out a global
    shrink of the whole group; smoothness and Dice anchor the rest. The template sharpens as
    alignment improves (same positive feedback as the Dice template).
- D6 Masking / background: images are skull-stripped with zero background, but the DALI intensity
  augmentation adds Gaussian noise everywhere, so the reference's "nonzero voxels" mask selects
  almost the whole volume during training. Instead use a label ROI: fg = Σ_{c ≥ 1} onehot (NOT
  1 − onehot[..., 0]: zero fill makes every channel 0 outside the FOV, which would read as
  foreground), ROI = stop_gradient(box_sum(fg) > 1e-3), i.e. the foreground dilated by the window
  radius so the brain boundary (most of the signal at 4 mm) is included. Template term: one ROI from
  the warped one-hot template fg (shared by all i). Pair term: ROI from max(fg of moved one-hot_i,
  fg of one-hot_j). The masked mean per image is Σ(cc·roi) / max(Σ roi, 1); then the mean over
  images (each image weighs equally, like Dice). No warped ones-mask for the FOV: zero fill equals
  the skull-stripped background value and the ROI excludes far background, so out-of-FOV voxels only
  enter if the brain is within k/2 voxels of the border (a unit test covers the border case).
- D7 Loss convention: terms minimised as `ncc_tmpl_loss = mean_i (1 − ncc_i)` and
  `ncc_pair_loss = mean_p (1 − ncc_p)` (O(1), in [0, 1], like 1 − Dice); the logged values
  `ncc_tmpl` / `ncc_pair` are the mean LNCC itself (higher is better).
- D8 Total deform loss (static Python checks at trace time, so the default graph is unchanged):
  `L = [λ_dt·]L_tmpl + λ_pdice L_pdice + λ_smooth L_smooth [+ λ_nt ncc_tmpl_loss] [+ λ_np ncc_pair_loss]`.
  The original three-term expression is built first, unchanged; the factor λ_dt multiplies L_tmpl
  only when λ_dt != 1; an NCC term is computed (and its keys returned) only when its λ > 0. New
  `--lambda-dice-template` (default 1) allows the LNCC-only ablation (`--lambda-dice-template 0
  --lambda-pdice 0`); soft Dice values are still computed and logged (the overfit stop uses them).
  Parser error when 'deform' is trained and all four similarity weights are 0.
- D9 Numerics: everything in float32 (`tf.cast` images and fields; never float16, AMP-safe since no
  Keras layer and explicit casts); finite gradients on constant regions are tested.
- D10 Logging: train `train_deform/{ncc_tmpl, ncc_pair, ncc_tmpl_loss, ncc_pair_loss}` (only the
  active terms), via a new `DEFORM_NCC_KEYS` tuple; `DEFORM_LOSS_KEYS` is unchanged. Validation
  (always computed, cheap, eval mode, same function / window / LOO / ROI as training):
  `val_deform/ncc_template`, `val_deform/ncc_pair` and the cached identity baselines
  `val_deform/ncc_template_identity`, `val_deform/ncc_pair_identity`.

## Items

- [ ] 0. Create the helper scripts `run_some.sh` and `py38_check.sh` (see ground rules).
      Files: .agents/tasks/lncc-loss/run_some.sh, .agents/tasks/lncc-loss/py38_check.sh
      Verify: `wsl -d Ubuntu bash .../lncc-loss/run_some.sh tests.test_runtime_losses_metrics.TestDeformLosses`
      passes on the unmodified code.

- [ ] 1. Deform golden guard, written on the UNMODIFIED code before any runtime change. In
      `tests/test_trainer.py` add `_deform_golden_run()` modelled on `_linear_golden_run()`: ratio
      {rigid: 1, affine: 1, deform: 2}, `seed = _seed_with_all_tasks(ratio, 8)`, 8 steps, flips on
      (flip_prob 0.5), narrow model with `deform_decoder=True`, dropout 0, seed 1, SVF head kernel set
      to N(0, 1e-3) with rng seed 0 (as `TestTaskLossWeightDeform.make`, so deform gradients reach
      the encoder), Adam 1e-3, linear groups = `synthetic_group(4, N, t, seed=i)`, deform group =
      `fd.synthetic_deform_group(4, N, seed=0)`. Record tasks, per-step loss, per-step deform terms
      (tmpl, pdice, smooth) and every variable's (float64 sum, max |v|). Add class
      `TestDeformUnchanged` with the same write/compare logic as `TestLinearOnlyUnchanged`
      (`BGR_WRITE_GOLDEN=1` writes `tests/fixtures/lncc_deform_golden.json`). Generate it once, then
      run the test twice more without the env var. If the reruns are not bit-identical (GPU scatter
      in the warp gradients may be nondeterministic), set the fixture's `tolerance` to 10x the
      largest observed run-to-run |diff| and document the measured value in the class docstring.
      Files: tests/test_trainer.py, tests/fixtures/lncc_deform_golden.json
      Verify: `run_some.sh tests.test_trainer.TestDeformUnchanged tests.test_trainer.TestLinearOnlyUnchanged`
      passes (twice) on the unmodified runtime code.

- [ ] 2. LNCC primitives in `runtime/losses.py` and the per-level window table. Add
      `NCC_WINDOW_DEFAULT = {4: 5, 2: 7, 1: 9}` and `def ncc_window_for_level(level)` to
      `data_loading/params.py`. In `runtime/losses.py` add `NCC_EPS = 1e-7`, `NCC_ROI_THRESHOLD = 1e-3` and:
      `box_sum(x, window)` ([B, X, Y, Z, C] → same, separable conv3d, D3; validate odd window ≥ 3
      and ≤ every spatial dim, ValueError otherwise); `local_ncc(a, b, window, eps=NCC_EPS,
      signed=False)` → map [B, X, Y, Z] (D1); `foreground(onehot)` = Σ_{c≥1} (D6);
      `ncc_roi(fg, window)` → stop-gradient float mask; `masked_mean(m, w)` → [B];
      `template_ncc_loss(warped_img, fg_tmpl, window, leave_one_out=True, eps=NCC_EPS)` →
      (loss, ncc [N]); `pair_ncc_loss(images, disp, pair_i, pair_j, fg_moved, fg_target, window,
      eps=NCC_EPS)` → (loss, ncc [P]). All cast inputs to float32 (`_float`). Extend the module
      docstring with the LNCC formulas. Add `class TestLNCC` to `tests/test_runtime_losses_metrics.py`
      with a float64 numpy reference (local means via `scipy.ndimage.uniform_filter(x, k,
      mode="constant")` divided by the same filter of ones):
      (a) `box_sum`/count means match the reference for k = 3, 5 incl. borders (rtol 1e-5);
      (b) `local_ncc` squared and signed match the reference on random volumes (rtol 1e-4);
      (c) b = 2a + 0.3 → cc = 1 (signed +1) and b = −a → squared 1, signed −1, on voxels with
          var > 1e-4;
      (d) constant blocks + zero background: map finite, cc = 0 where both are flat, gradients of the
          template and pair losses w.r.t. images and u (through `diffeo.warp`) finite;
      (e) ROI: background noise outside the dilated foreground leaves the loss bit-identical; a
          foreground touching the volume border gives a finite loss;
      (f) LOO: N = 4 independent noise images in a common ROI → LOO mean ncc < 0.05, inclusive
          mean > 0.15; N identical images → ncc = 1 (atol 1e-5) with LOO;
      (g) template gradient flows through the template (differs from the stop_gradient variant, as
          `test_template_gradient_flows_through_template`);
      (h) pair term: image 1 = image 0 shifted by one voxel; u_inv_1 = +1 voxel along x gives
          ncc_p > 0.999 on the interior, identity gives < 0.99;
      (i) float16 inputs return float32 outputs; window 4 / 1 / larger than the volume → ValueError.
      Files: data_loading/params.py, runtime/losses.py, tests/test_runtime_losses_metrics.py
      Verify: `run_some.sh tests.test_runtime_losses_metrics` all pass.

- [ ] 3. Wire LNCC into `deform_loss` (D8). New signature
      `deform_loss(out, batch, lambda_pdice=1.0, lambda_smooth=0.5, smooth_order=1,
      lambda_dice_tmpl=1.0, lambda_ncc_tmpl=0.0, lambda_ncc_pair=0.0, ncc_window=5)`. Give
      `pair_dice_loss` optional `disp=None` and `return_moved=False` (default path = the current ops
      in the current order). When λ_np > 0 compute `disp` once, pass it to `pair_dice_loss` and reuse
      it (and the moved one-hot for the ROI) in `pair_ncc_loss`. Template NCC uses
      `foreground(template)` from `template_dice_loss`'s third output. Needs `batch["images"]` only
      when an NCC weight is > 0 (ValueError naming the key otherwise). Extend `TestDeformLosses`:
      default call vs explicit zeros/1.0 give bit-identical terms and the same key set (no ncc keys);
      batch without "images" works when off; total = λ_dt tmpl + λ_p pdice + λ_s smooth + λ_nt
      ncc_tmpl_loss + λ_np ncc_pair_loss (rtol 1e-6) with all keys float32 and ncc in [0, 1];
      `ncc_*_loss == 1 − ncc_*`; λ_dt = 0 → gradient w.r.t. u equals that of the loss without the
      template Dice term (rtol 1e-5); pair NCC with shared disp equals a separate recomputation.
      Files: runtime/losses.py, tests/test_runtime_losses_metrics.py
      Verify: `run_some.sh tests.test_runtime_losses_metrics` passes; `run_some.sh
      tests.test_trainer.TestDeformUnchanged` still passes.

- [ ] 4. Trainer wiring. `TrainConfig` gains `lambda_dice_tmpl: float = 1.0`,
      `lambda_ncc_tmpl: float = 0.0`, `lambda_ncc_pair: float = 0.0`, `ncc_window: Optional[int] =
      None` (None → `P.ncc_window_for_level(level)`). `Trainer.__init__` validates the lambdas
      (finite, ≥ 0) and resolves `self.ncc_window` (odd ≥ 3). `_make_deform_step` passes them to
      `losses.deform_loss`. Add `DEFORM_NCC_KEYS = ("ncc_tmpl", "ncc_pair", "ncc_tmpl_loss",
      "ncc_pair_loss")`; the window logging (`acc.update` at the DEFORM branch, ~line 1007) adds
      `pre + k` for the NCC keys present in `res`. Update the run.py module docstring (deform step,
      logged keys). New `class TestDeformNCC(DeformCase)` in `tests/test_trainer.py`: with λ_nt =
      λ_np = 1: two finite deform steps, NCC keys finite and in [0, 1], `n_vars_updated` and
      `len(model.variables)` equal to a Dice-only trainer, `trace_counts == {"deform": 1}`;
      `grad_norm` differs from the Dice-only step from the same weights; a 8-step three-task fit
      traces once per task and logs `train_deform/ncc_tmpl` (jsonl, as
      `test_enc_grad_diagnostics_logged`); LNCC-only config (λ_dt = λ_pdice = 0) is finite and still
      reports soft_dice_tmpl; invalid λ (−1, nan) and window 4 raise ValueError.
      Files: runtime/run.py, tests/test_trainer.py
      Verify: `run_some.sh tests.test_trainer.TestDeformNCC tests.test_trainer.TestDeformUnchanged
      tests.test_trainer.TestLinearOnlyUnchanged tests.test_trainer.TestDeformStep
      tests.test_trainer.TestThreeTasks` all pass.

- [ ] 5. Validation metrics. `Trainer._deform_errors(..., images=None)`: when images are given,
      compute (eager, float32) warped images, the template ROI from the warped one-hot BEFORE its
      `del`, `template_ncc_loss` per image, and per chunk of pairs `pair_ncc_loss` with the same
      `disp`; add per-item arrays `ncc_template` [N] and `ncc_pair` [P] to the errors dict (pooled by
      `M.concat_errors`, averaged by `M.summarize`). Pass `images=grp["images"]` from
      `_val_deform_errors` and `_identity_deform_metrics`; the identity cache also stores
      `val_<name>/ncc_template_identity` / `ncc_pair_identity`. Update the docstrings (run.py
      deformable-validation paragraph, metrics.py deformable list). Extend
      `TestDeformValidation.test_deform_validation_and_pngs`: the four new keys finite, identity model
      at init equals the identity baseline (places=10), identity values unchanged after `fit(2)`,
      `val_trace_counts` unchanged.
      Files: runtime/run.py, runtime/metrics.py (docstring only), tests/test_trainer.py
      Verify: `run_some.sh tests.test_trainer.TestDeformValidation tests.test_trainer.TestThreeTasks` pass.

- [ ] 6. CLI. In the deform group of `runtime/args.py`: `--lambda-dice-template` (float, 1.0),
      `--lambda-ncc-template` (float, 0.0), `--lambda-ncc-pair` (float, 0.0), `--ncc-window` (int,
      None = per-level default, resolved in `parse_args` from `P.ncc_window_for_level(args.level)`).
      Validation: lambdas finite ≥ 0; window odd, ≥ 3, ≤ min(LEVEL_SHAPE[level]); 'deform' trained
      with all four similarity weights 0 → parser.error. Pass to `TrainConfig` in
      `train_config_from_args`; update the module docstring. Tests: extend
      `tests/test_runtime_infra.py::test_deform_flag_defaults` (defaults 1/0/0/5 at level 4, 7 at
      level 2, 9 at level 1; round trip of explicit values into TrainConfig; bad values → SystemExit)
      and the key list in `tests/test_main.py` (~line 125) with `lambda_dice_template,
      lambda_ncc_template, lambda_ncc_pair, ncc_window`. Add a deform dry run through `main.py` with
      `--lambda-ncc-template 1 --lambda-ncc-pair 1` to the existing deform dry-run test in
      `tests/test_main.py` (or a sibling test) asserting `train_deform/ncc_tmpl` and
      `val_deform/ncc_template` in metrics.jsonl.
      Files: runtime/args.py, tests/test_runtime_infra.py, tests/test_main.py
      Verify: `run_some.sh tests.test_runtime_infra tests.test_main` pass; `py38_check.sh` OK.

- [ ] 7. Memory measurement. Add `--lambda-ncc-template` / `--lambda-ncc-pair` (default 0) to
      `tests/gpu/measure_deform_memory.py` (passed into its `TrainConfig`, recorded per result and
      included in `config_key`). Run locally (`--gpu-memory-limit-mb 6000`): narrow N = 2, 3, 4 and
      default widths N = 2, each with NCC 0/0 and 1/1, `--steps 5`, output
      `.agents/tasks/lncc-loss/memory.json`. Report the extra peak MB per image (slope difference) and
      extrapolate to default widths N = 16 at level 4 (baseline 33.8 GB on the cluster); the target is
      < 1 GB extra at N = 16 (fits the 40 GB A100 with margin). If above, record the cause.
      Files: tests/gpu/measure_deform_memory.py, .agents/tasks/lncc-loss/memory.json
      Verify: the script exits 0 and every row has status 'ok' (or a recorded 'oom' at the same N for
      both loss settings).

- [ ] 8. Gradient-balance probe and local synthetic overfit experiment.
      (a) `.agents/tasks/lncc-loss/grad_probe.py` (+ `.sh` wrapper): synthetic deform group N = 4,
      narrow model, after 50 Dice-only training steps (non-zero SVF head), compute with separate
      tapes the total and encoder-only gradient norms of each term (tmpl Dice, pdice, smooth,
      ncc_tmpl_loss, ncc_pair_loss) and of the Dice-only vs Dice+LNCC (1/1) objectives; write
      `grad_probe.json`. Milestone-7 context: deform step total norm ~0.08-0.5, never clipped (clip
      12); report whether Dice+LNCC stays far below 12 and the new deform/linear `_eff` implication.
      (b) `.agents/tasks/lncc-loss/overfit_compare.sh`: three `main.py` runs with the flags of
      `tests/test_deform_overfit.py` (`--overfit-one-group --overfit-synthetic --task-ratio deform:1
      --group-size 4`, Adam 3e-4 constant, no warmup, `--bn-momentum 0.9`, seed 0, NARROW_ARGS,
      jsonl, quiet) but a FIXED budget: `--overfit-max-steps 600 --overfit-dice-threshold 1.01`
      (never stops early), `--val-every 100`, results under `.agents/tmp/lncc-overfit/`:
      A Dice-only (defaults); B Dice+LNCC (`--lambda-ncc-template 1 --lambda-ncc-pair 1`);
      C LNCC-only (`--lambda-dice-template 0 --lambda-pdice 0 --lambda-ncc-template 1
      --lambda-ncc-pair 1`). Run in the background (~40 min). Copy the three overfit_summary.json
      files to `.agents/tasks/lncc-loss/overfit_{A,B,C}.json` and tabulate: final eval template /
      pair hard Dice vs identity, val ncc_template / ncc_pair vs identity, neg_jac_pct, train
      grad_norm window means, median step ms, gpu peak. Expectations (report, not hard gates): A
      close to the milestone-6 reference (0.971 / 0.950); B Dice within 0.02 of A with higher NCC;
      C Dice > identity + 0.05, no folding. Explain any miss.
      Files: .agents/tasks/lncc-loss/grad_probe.py, grad_probe.sh, grad_probe.json,
      overfit_compare.sh, overfit_{A,B,C}.json
      Verify: all three runs exit 0 with finite losses; grad_probe.json written with finite values.

- [ ] 9. Full verification. Run `py38_check.sh` (all changed .py files incl. tests and
      `.agents/tasks/lncc-loss/*.py`, `bash -n` on the new .sh). Then the full suite in the
      background: `wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/tests/run_tests_wsl.sh`
      with output to `.agents/tasks/lncc-loss/test-log.txt` (~90+ min). Fix any failure and rerun the
      affected modules.
      Files: .agents/tasks/lncc-loss/test-log.txt
      Verify: test-log.txt shows py3.8 stage OK and `unittest exit code: 0` with the test count
      (previously 295) increased by the new tests, both golden tests passing.

- [ ] 10. Documentation. `.agents/tasks/network-design/design.md` → v3.9: status line; 6.2 new
      "LNCC (v3.9)" paragraph (D1-D9, formulas, defaults 0, recommended 1/1, window table, LOO, ROI,
      the reference keep/change list); total-loss line; section 7 deformable row adds
      val ncc_template / ncc_pair (+ identity); section 9 adds the measured LNCC memory per image.
      `.agents/tasks/lncc-loss/report.md`: files changed, decisions, memory numbers (item 7), gradient
      norms (item 8a), overfit table (item 8b), test results, suggested cluster command for Jay (e.g.
      `TASK_RATIO`/`EXTRA_ARGS="--lambda-ncc-template 1 --lambda-ncc-pair 1"` with
      `slurm/m7_three_task.sh`; not submitted).
      Files: .agents/tasks/network-design/design.md, .agents/tasks/lncc-loss/report.md
      Verify: design.md status reads v3.9 and the numbers in report.md match memory.json /
      grad_probe.json / overfit_*.json.

## Assumptions and gaps
- Validation NCC is always computed (cheap; the new keys only add log entries), so Dice-only runs
  also report it for comparison. Training NCC is computed only when its weight is > 0.
- Real-data identity LNCC values at level 4 are unknown locally (no local export); the cluster run
  reports them via `val_deform/ncc_*_identity`.
- The intensity augmentation (noise, gamma, contrast) stays on for deform training; LNCC is
  invariant to local affine intensity changes, and noise only lowers the attainable cc.
- `tests/gpu/check_mixed_run.py` is not changed (its deform keys still exist).
