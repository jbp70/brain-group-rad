"""Brain MRI preprocessing for groupwise registration (brain-group-rad).

Outputs for every dataset: skull-stripped, on the LUMIR 160x224x192 1 mm grid
(with a corrected, MNI152-world affine), intensities in [0, 1], with labels.

Pipeline (manifest-driven; each step resumable):
  validate -> reorient (flirt backend) -> strip (SynthStrip / provided / derived mask)
  -> N4 -> 9-DOF FLIRT (or ANTs) to MNI152 1 mm -> ONE resampling into the target grid
  -> in-mask p99.5 normalization -> SynthSeg (batch) -> QC

Output layout:
  <out_root>/<dataset>/<subject>[_ses-<session>][_<contrast>]/
      image.nii.gz (float32 [0,1]), labels_synthseg.nii.gz (uint16), labels_<name>.nii.gz,
      affine_to_template.mat, [affine_to_template_world.txt], meta.json, qc.json,
      synthseg_vols.csv, synthseg_qc.csv, logs/
      optional (config outputs): mask.nii.gz (save_mask), qc.png (save_qc_png; or on demand
      with qc-summary --png), work/ (keep_work; otherwise deleted once the row is done)
      Without mask.nii.gz the mask is image > 0 (loses filled interior holes and
      in-mask voxels normalized to 0, i.e. at/below the normalization lower bound).
  <out_root>/<dataset>/{manifest_processed.csv, qc_summary.csv, qc_summary.json,
      lr_asymmetry.csv}   (written by qc-summary)
  <out_root>/_tmp/   tool temp files when $TMPDIR is unusable (config tmp_dir)

Runs with Python >= 3.8 (the cluster container tf10_nv.sif has Python 3.8.10,
numpy 1.21, scipy 1.4, nibabel 3.1, matplotlib 3.5). On Martinos machines
replace ``python -m preprocessing`` with ``preprocessing/run_in_container.sh``.

End-to-end (from the repo root):
  python -m preprocessing check-tools --config cfg.yaml
  python -m preprocessing build-target-grid --config cfg.yaml --lumir-root /data/LUMIR2025
  python -m preprocessing build-manifest --dataset neurite_oasis --root /data/neurite-oasis --out oasis.csv
  python -m preprocessing n-chunks --manifest oasis.csv --chunk-size 20
  python -m preprocessing run --config cfg.yaml --manifest oasis.csv --out-root /data/processed --rows 0 --dry-run
  python -m preprocessing run --config cfg.yaml --manifest oasis.csv --out-root /data/processed --start 0 --end 20
  python -m preprocessing synthseg --config cfg.yaml --manifest oasis.csv --out-root /data/processed
  python -m preprocessing qc-summary --out-root /data/processed --dataset neurite_oasis
  python -m preprocessing export-npy --out-root /data/processed --export-root /data/npy

SLURM templates live in preprocessing/slurm/.
"""
__version__ = "0.1.0"
