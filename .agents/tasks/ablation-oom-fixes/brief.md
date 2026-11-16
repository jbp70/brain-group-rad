# ablation-oom-fixes: brief (workflow wf_063ae32b8181bed4)

Fix two GPU out-of-memory failures that block two paper ablations (TF 2.10). Base `main` 4fe21f7,
worktree `.worktrees/ablation-oom-fixes`. No cluster submissions, no `git pull` on the cluster.

1. Instance norm (A2, `--norm instance`, config B): smoke 10639200 (r_cascade.sh NORM=instance) OOM at
   level 4 in the first training step, node `encoder/l4/norm/sub`. `InstanceNormWithAffine` computed
   moments and normalisation in float32, so every normalised activation had full-size float32 copies
   under mixed_float16. Wanted: an IN whose full-size tensors stay float16 with float32 moments, safe
   numerics (variance >= 0, epsilon, no float16 x^2), correct under `--gradient-checkpointing full`;
   memory measured / estimated per level vs batch norm (L4 N 128 / 64, L2 32 / 16, L1 8 / 4 under
   TF_MEM_LIMIT_MB 42500); float32 paths bit-identical (goldens 0 diff); report instead of changing
   N if it cannot fit.
2. Full resolution from scratch (A1, `slurm/r_level1_scratch.sh`): smoke 10639201 OOM at the first
   deform step right after the step-100 validation (diagnosis done: allocator layout after a
   linear-only phase plus a validation pass; 10639614 deform from 0 and 10639615 no validation before
   deform both fit). Fix (Jay): opt-in `--deform-warmup-step` (default false, bit-identical) = one
   deform train step at startup whose update is fully discarded (weights, BN statistics, optimizer
   slots and iterations, loss scale, RNG / data positions, non-finite counters, metrics / logs), same
   traced function (trace once), also on resume; enabled in r_level1_scratch.sh; test that the state
   after N steps is identical with and without the warm-up.
3. Small: checker `nonfinite_window_<task>` failed a 200-step smoke on one short window (2 of 21
   affine steps); skip windows with too few task steps, long-run behaviour unchanged. Step rule of
   r_level1_scratch.sh: STEPS = (229860 - val_time) / (3/16 t_lin + 13/16 t_mix).

Done when: both fixes + tests, full suite passes, goldens 0 diff, review approved, report with smoke
commands (IN chain via r_cascade.sh NORM=instance; scratch L1 smoke 300 steps, deform from 100,
validation at 100), both with DATA=.../GroupwiseReg/data/npy_v3, plus what to read from each smoke.
