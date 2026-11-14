"""Manifest for re-registering the PROCESSED LUMIR cases affinely to MNI152 with ANTs.

Input: <processed>/lumir/manifest_processed.csv (qc-summary output). Each row becomes a row of
dataset ``lumir_ants`` whose image is the processed LUMIR image (corrected MNI-world affine,
skull-stripped, bias-corrected, [0, 1]), with in_template_space false so the pipeline runs
``register`` (ANTs Rigid + Affine to MNI152 1 mm, config preprocessing/config_lumir_ants.yaml)
and resamples ONCE onto the LUMIR grid. Labels are carried along (nearest / GenericLabel):
labels_synthseg -> label name ``synthsegsrc`` ('synthseg' is reserved for the SynthSeg step;
rename the outputs afterwards, see the config header) and labels_lumir10 where present.
Split, contrast and subject ids are kept.

    python scripts/lumir_ants_manifest.py --processed-manifest $PROJ/data/processed/lumir/manifest_processed.csv \\
        --out $PROJ/data/manifests/lumir_ants.csv
"""
import argparse
import csv
import os
import sys
from collections import OrderedDict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from preprocessing.manifest import ManifestRow, parse_bool, write_manifest  # noqa: E402

LABEL_MAP = (("synthseg", "synthsegsrc"), ("lumir10", "lumir10"))


def build(processed_manifest, dataset="lumir_ants", only_completed=True):
    rows, skipped = [], []
    with open(processed_manifest, newline="", encoding="utf-8") as f:
        for d in csv.DictReader(f):
            if only_completed and d.get("status") != "completed":
                skipped.append(d["subject"])
                continue
            out_dir = d["out_dir"]
            image = os.path.join(out_dir, "image.nii.gz")
            labels = OrderedDict()
            for src, name in LABEL_MAP:
                p = os.path.join(out_dir, "labels_%s.nii.gz" % src)
                if os.path.isfile(p):
                    labels[name] = p
            rows.append(ManifestRow(dataset=dataset, subject=d["subject"], session=d.get("session", ""),
                                    contrast=d.get("contrast", ""), split=d.get("split", ""), image_path=image,
                                    label_paths=labels, skull_stripped=True, in_template_space=False,
                                    bias_corrected=True))
            assert parse_bool(d.get("skull_stripped", "true"))
    return rows, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--processed-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="lumir_ants")
    a = ap.parse_args(argv)
    rows, skipped = build(a.processed_manifest, a.dataset)
    write_manifest(rows, a.out)
    n_ss = sum("synthsegsrc" in r.label_paths for r in rows)
    n_l10 = sum("lumir10" in r.label_paths for r in rows)
    print("%d rows -> %s (%d with synthseg labels, %d with lumir10; %d not completed, skipped)"
          % (len(rows), a.out, n_ss, n_l10, len(skipped)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
