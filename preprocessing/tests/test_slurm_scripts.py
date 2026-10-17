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
    assert names == {"run_array.sh", "synthseg_batch.sh", "qc_export.sh", "build_target_grid.sh",
                     "export_array.sh"}
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
    for key in ("--account=qtim", "--time=", "--mem=", "--cpus-per-task=",
                "--nodes=1", "--ntasks-per-node=1", "--output=%s/logs/" % PROJ, "--error=%s/logs/" % PROJ):
        assert "#SBATCH " + key in text, key
    # Partition: a placeholder or one of the partitions allowed by the cluster rules (TF 2.10-capable GPUs,
    # no Blackwell qtim/pubgpu-req, no L40S, no lab-private lcn*/west).
    part = re.search(r"^#SBATCH --partition=(\S+)$", text, re.M)
    assert part, "missing #SBATCH --partition"
    assert part.group(1).startswith("PLACEHOLDER") or set(part.group(1).split(",")) <= {
        "basic", "rtx8000", "rtx6000", "dgx-a100", "pubgpu"}, part.group(1)
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
    for name in ("run_array.sh", "build_target_grid.sh", "qc_export.sh", "export_array.sh"):
        assert "export USE_GPU=${USE_GPU:-0}" in open(os.path.join(PKG, "slurm", name)).read()


def test_export_array_policy():
    text = open(os.path.join(PKG, "slurm", "export_array.sh")).read()
    lines = _sbatch(text)
    # CPU partition basic, account qtim, no GPU, at most two tasks at once, fair share (~15 GB/core).
    assert _opt(lines, "partition") == "basic" and _opt(lines, "account") == "qtim"
    assert not [line for line in lines if "--gpus" in line or "--gres" in line]
    assert _opt(lines, "array").endswith("%2")
    assert _mem_gb(_opt(lines, "mem")) <= 15 * int(_opt(lines, "cpus-per-task"))
    for key in ("output", "error"):
        assert _opt(lines, key).startswith(PROJ + "/logs/")
    # Usage lines keep %2 on every sbatch --array, including the merge job.
    usage = [line for line in text.splitlines() if line.startswith("#") and "sbatch" in line and "--array" in line]
    assert usage and all(re.search(r"--array=\S+%2\b", line) for line in usage), usage
    # Chunked export-npy with race-free parts, a separate merge step, the new root and float32.
    for s in ('--start "$START" --end "$END"', 'export-merge --export-root "$EXPORT_ROOT"',
              "EXPORT_ROOT=${EXPORT_ROOT:-$PROJ/data/npy_v2}", "IMAGE_DTYPE=${IMAGE_DTYPE:-float32}",
              "LABEL_SETS=${LABEL_SETS:-train18}", "CONTRAST=${CONTRAST:-T1w}", 'if [ "$MERGE_ONLY" = "1" ]'):
        assert s in text, s
    # Only task 0 exports the template (no concurrent template writes).
    assert '[ "$TASK" = "0" ]' in text
    # Optional (evaluation-only) label sets: default empty, passed only when set; documented via env.
    for s in ("OPTIONAL_LABEL_SETS=${OPTIONAL_LABEL_SETS:-}", 'if [ -n "$OPTIONAL_LABEL_SETS" ]; then',
              'extra+=(--optional-label-sets "$OPTIONAL_LABEL_SETS")',
              "env LABEL_SETS=train18 OPTIONAL_LABEL_SETS=lumir10 sbatch --array=0-0%2"):
        assert s in text, s


def _export_argv(tmp_path, env_extra):
    """Run export_array.sh with a stub run_in_container.sh that prints its argv; return that argv."""
    repo = tmp_path / "repo"
    (repo / "preprocessing").mkdir(parents=True, exist_ok=True)
    (repo / "preprocessing" / "run_in_container.sh").write_text('printf "%s\\n" "$@"\n')
    env = {k: v for k, v in os.environ.items() if k not in ("OPTIONAL_LABEL_SETS", "LABEL_SETS")}
    env.update(REPO_DIR=str(repo), SLURM_ARRAY_TASK_ID="1", **env_extra)
    res = subprocess.run(["bash", os.path.join(PKG, "slurm", "export_array.sh")], env=env,
                         stdout=subprocess.PIPE, universal_newlines=True)
    assert res.returncode == 0
    return res.stdout.splitlines()


def test_export_array_optional_label_sets(tmp_path):
    argv = _export_argv(tmp_path, {})
    assert "--optional-label-sets" not in argv and argv[argv.index("--label-sets") + 1] == "train18"
    argv = _export_argv(tmp_path, {"OPTIONAL_LABEL_SETS": "lumir10,seg4", "LABEL_SETS": "train18"})
    assert argv[argv.index("--optional-label-sets") + 1] == "lumir10,seg4"


def test_qc_export_defaults():
    text = open(os.path.join(PKG, "slurm", "qc_export.sh")).read()
    assert "IMAGE_DTYPE=${IMAGE_DTYPE:-float32}" in text and "IMAGE_DTYPE:-float16" not in text
    assert "EXPORT_ROOT=${EXPORT_ROOT:-$PROJ/data/npy_v2}" in text
