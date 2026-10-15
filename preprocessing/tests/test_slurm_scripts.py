import glob
import os
import re
import subprocess

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = sorted(glob.glob(os.path.join(PKG, "**", "*.sh"), recursive=True))
SLURM = [s for s in SCRIPTS if os.sep + "slurm" + os.sep in s]
PROJ = "/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg"


def test_found_scripts():
    names = {os.path.basename(s) for s in SLURM}
    assert names == {"run_array.sh", "synthseg_batch.sh", "qc_export.sh", "build_target_grid.sh"}
    assert os.path.join(PKG, "run_in_container.sh") in SCRIPTS


@pytest.mark.parametrize("path", SCRIPTS, ids=os.path.basename)
def test_shebang_lf_syntax_no_rc(path):
    data = open(path, "rb").read()
    assert data.startswith(b"#!/bin/bash\n")
    assert b"\r" not in data
    text = data.decode()
    for rc in (".bashrc", ".cshrc", ".tcshrc", ".bash_profile"):
        assert "source ~/" + rc not in text and ". ~/" + rc not in text
    assert subprocess.run(["bash", "-n", path]).returncode == 0


def _sbatch(text):
    return [line for line in text.splitlines() if line.startswith("#SBATCH ")]


def _opt(lines, key):
    vals = [line.split("=", 1)[1] for line in lines if line.startswith("#SBATCH --%s=" % key)]
    assert len(vals) == 1, (key, vals)
    return vals[0]


def _mem_gb(v):
    m = re.fullmatch(r"(\d+)([GM])", v)
    assert m, v
    return int(m.group(1)) / (1024.0 if m.group(2) == "M" else 1.0)


@pytest.mark.parametrize("path", SLURM, ids=os.path.basename)
def test_slurm_placeholders(path):
    text = open(path).read()
    for key in ("--partition=PLACEHOLDER", "--account=qtim", "--time=", "--mem=", "--cpus-per-task=",
                "--nodes=1", "--ntasks-per-node=1", "--output=%s/logs/" % PROJ, "--error=%s/logs/" % PROJ):
        assert "#SBATCH " + key in text, key
    # Workspace cluster rules: never the entre account; an unedited QOS placeholder must stay inactive.
    assert "entre" not in text.replace("(never entre)", "")
    assert "\n#SBATCH --qos" not in text and "##SBATCH --qos=PLACEHOLDER" in text
    assert "set -euo pipefail" in text and 'cd "$REPO_DIR"' in text
    # Pipeline runs in the container through the wrapper; no conda/PY_ENV_ACTIVATE left.
    assert 'RIC=(bash "$REPO_DIR/preprocessing/run_in_container.sh")' in text
    assert '"${RIC[@]}" ' in text and 'export BGR_REPO_DIR="$REPO_DIR"' in text
    assert "python -m preprocessing" not in text.split("set -euo pipefail", 1)[1]
    for gone in ("PY_ENV_ACTIVATE", "conda", "eval "):
        assert gone not in text, gone
    # No inline comments after #SBATCH options (sbatch may not strip them).
    lines = _sbatch(text)
    for line in lines:
        assert "#" not in line[len("#SBATCH"):], line
    # Fair share: one GPU per job on rtx8000 (<= 3 CPUs, <= 128 GB); CPU jobs ~15 GB per core.
    cpus = int(_opt(lines, "cpus-per-task"))
    mem = _mem_gb(_opt(lines, "mem"))
    gpus = [line for line in lines if "--gpus" in line or "--gres" in line]
    if gpus:
        assert gpus == ["#SBATCH --gpus=1"]
        assert cpus <= 3 and mem <= 128
    else:
        assert mem <= 15 * cpus
    array = [line for line in lines if line.startswith("#SBATCH --array=")]
    for line in array:
        assert line.endswith("%2"), line


def test_synthseg_gpu_toggle():
    text = open(os.path.join(PKG, "slurm", "synthseg_batch.sh")).read()
    assert "export USE_GPU=1" in text and "export USE_GPU=0" in text and "--synthseg-cpu" in text
    for name in ("run_array.sh", "build_target_grid.sh", "qc_export.sh"):
        assert "export USE_GPU=${USE_GPU:-0}" in open(os.path.join(PKG, "slurm", name)).read()
