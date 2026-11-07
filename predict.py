"""Standalone groupwise prediction (predict-script design, .agents/tasks/predict-script/design.md).

    python predict.py run --checkpoint <run>/ckpt/best_deform --processed-dir <processed> --out <dir>
        [--branches rigid|affine|deform|affine+deform] [--mode streamed|unchunked] [--device gpu|cpu] ...
    python predict.py check-streaming --checkpoint ... --processed-dir ... --out check.json [--n-subjects 4]
    python predict.py linear-eval --checkpoint ... --processed-dir ... --task rigid|affine --out <dir>
        [--seed 0] [--tta none|flip8] [--tta-levels LEVELS] ...

`run` (and `linear-eval`) accept --tta none|flip8 (8-flip test-time augmentation, TTA design
.agents/tasks/tta/design.md). `linear-eval` renders the subjects with seeded synthetic rigid /
affine misalignments and writes per_subject.csv, per_pair.csv, summary.json (+ tta_spread.csv with
flip8) into --out (TTA design 7.3).

`run` writes per subject <subject_dir>_fwd.nii.gz / _inv.nii.gz (mm, evaluator convention
--components xyz --signs +++ --units mm) and _moved.nii.gz, plus atlas.nii.gz, atlas_seg.nii.gz and
predict.json. Arguments are parsed and validated before TensorFlow is imported. Exit codes: 0 ok,
1 unexpected error, 2 invalid input, 3 non-finite outputs, 4 check-streaming gate exceeded,
5 internal consistency check failed.
"""
import logging
import sys


def main(argv=None):
    from inference import cli
    args = cli.parse_args(argv)                     # argparse errors exit 2, no TF imported yet
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    from inference import predict as PR
    if args.command == "run":
        return PR.run_predict(args, configure_device=True)
    if args.command == "linear-eval":
        from inference import linear_eval as LE
        return LE.run_linear_eval(args, configure_device=True)
    return PR.run_check_streaming(args, configure_device=True)


if __name__ == "__main__":
    sys.exit(main())
