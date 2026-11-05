# Review FEAT-003 (iteration 1): DALI labels, data module, CLI flags, Trainer / main wiring, e2e runs

Verdict: APPROVED (0 HIGH, 0 MEDIUM, 3 NIT).

## Evidence checked
- Diff of data_loading/dali_loader.py, data_loading/data_module.py, runtime/args.py, runtime/run.py, main.py and the
  FEAT-003 test additions (test_data_loading, test_runtime_infra, test_main, test_trainer, test_cascade).
- impl-log.md "FEAT-003 iteration 1": tf_gpu group (5 modules) 173 OK (3 DALI skips); tests.test_trainer 74 OK with
  `[golden] max |diff| 0` and `[golden deform] max |diff| 0`; tests.test_cascade 52 OK; bgr_preproc38 py_compile,
  ast 3.8 (75 files), `main.py --help`, TF-free `import runtime.args`. e2e numbers for p 1 / p 0 recorded with trace
  counts {rigid 1, affine 1, deform 1}. The evidence matches the acceptance criteria, so I did not rerun suites.
- Cascade-skip guards unchanged: `git diff -U0 tests/test_cascade.py` removes only two import lines; TestSkip,
  TestChainTrainer, TestStageSkipLevel2, TestStageSkipChain, TestBNRecal bodies are untouched and pass in the 52-test run.

## Requirement checks
- DALI order (design 5.1): `linear_output_signature` puts uint8 labels after T[, Rt] and before G / version
  (dali_loader.py:89-110); `base_output_count` / `output_count` +1 for linear only (:113-143). `LinearPipeline` validates
  with `check_linear_label_inputs` before `require_dali` (:499-501, :571-580), permutes both lists with
  `paired_permutation` (= the old RandomState permutation, :374-377, :520-526), and creates the label reader after
  every random op with the same 'epoch' kwargs and seed as the image reader (:557-567); `numpy_reader_kwargs` adds
  `initial_fill` only in buffer mode (:350-356), so the two readers differ only in files / name, as in DeformPipeline.
  Labels are not warped.
- Data module: `synth_label_set` resolved eagerly (data_module.py:184-187, :217-228); default path is the old call
  byte-for-byte (:254-259); synth deform uses the fine labels (:237-239).
- CLI (5.4): five flags, validation with parser.error (args.py:649-665), `synth_config_from_args` returns None at p 0 and
  for Namespaces without the fields (:980-990), `train_config_from_args` passes it. TF-free import confirmed in the log.
- Trainer (5.3): `_pp_call` (run.py:806-823) is the only post-process entry (grep: no remaining direct
  `self.postprocess[t](...)` calls); off path keeps `pp(*inputs)`; linear signatures float32-only when off
  (:1045-1052); the four synthesis-on lambdas pass `labels=` keyword-only and `skip` positional (:1160-1176), the
  existing off lambdas are untouched; `split_batch` counts labels only for linear (:1194-1197); `synth` result,
  window logs `synth_frac` every executed step and `loss_synth` / `loss_real` for finite steps (:2139-2143);
  `rec.update(res)` keeps `synth` in history; checkpoint key `postprocess_<t>_synth` only with a generator (:798-804);
  BN recal slots via `synth_recal_slot` and event counts only when on (:1878-1936).
- main.py: `synth_label_set` passed only when on (main.py:181-188), desc suffix, `_check_synth_config` in the resume
  branch (:400), `config_extra['synth']` only when on (:404-407) before the first write.
- H1/H2/H3/H4: no model or group-size change, no validation-path change, trace counts 1 per task with p 0.5 / p 1
  (TestSynthTrainer, TestSynthSkip, TestSynthSkipChain), OneHot depth multiset on == off ({4:1},{4:1},{19:1}), goldens 0,
  `SynthConfig(prob=0)` bit-equal to default under op determinism.
- Deviation `_synth_on = any(pp.synth is not None ...)` (run.py:770): consistent with FEAT-002 treating prob 0 as off;
  `recalibrate_bn` only dereferences `cfg.synth.prob` when `_synth_on`, so it is safe. Accepted.

## Findings
- NIT 1 (no DALI execution of the linear label reader yet): the 3 DALI tests are skipped locally, so the aligned
  image / label readers of `LinearPipeline` (dali_loader.py:557-567) have only been unit-tested through the pure-Python
  helpers. FEAT-004's verify_dali / cluster smokes should include an image-label alignment check (e.g. labels > 0 where
  the unwarped image is non-zero at T = I) before the paired smokes are trusted.
- NIT 2 (tests/test_cascade.py:19-20): two separate `from tests.runtime_helpers import ...` lines; could be merged.
- NIT 3 (data_module.py:245-253): the synthesis branch duplicates the whole `LinearPipeline(...)` call to keep the
  default call literally unchanged; a future argument added to only one branch would diverge silently. Acceptable for
  now; keep both branches in sync when editing.
