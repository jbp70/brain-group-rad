#!/bin/bash
# Run the preprocessing test suite inside a local WSL conda env.
#
# Usage (from Windows PowerShell):
#   wsl -d Ubuntu bash /mnt/d/Research/Brain_Registration_Projects/brain-group-rad/preprocessing/tests/run_tests_wsl.sh [env] [pytest args]
#
# env (optional first argument):
#   bgr_preproc38  (default) Python 3.8 + the versions inside the cluster container
#                  (tf10_nv.sif): requirements-test.txt
#   bgr_preproc    Python 3.9 + newer pins: requirements-test-py39.txt
#   any other name uses Python 3.9 + requirements-test-py39.txt
# The env is created on first use (conda create, offline fallback). Extra args go
# to pytest (e.g. a single test module). The bash -n syntax check of every .sh
# file and the CLI --help smoke test always run.
set -eo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_NAME=bgr_preproc38
if [ $# -gt 0 ] && [[ "$1" != -* ]] && [[ "$1" != */* ]] && [[ "$1" != *.py ]]; then
    ENV_NAME=$1
    shift
fi
case "$ENV_NAME" in
    bgr_preproc38) PY_VER=3.8; REQ="$REPO/preprocessing/requirements-test.txt" ;;
    *)             PY_VER=3.9; REQ="$REPO/preprocessing/requirements-test-py39.txt" ;;
esac
source ~/miniconda3/etc/profile.d/conda.sh
conda env list | grep -q "^${ENV_NAME} " || conda create -y -q -n "$ENV_NAME" "python=$PY_VER" \
    || conda create -y -q --offline -n "$ENV_NAME" "python=$PY_VER"
conda activate "$ENV_NAME"
# Offline fallback (e.g. WSL DNS broken under VPN): set BGR_WHEELS to a folder of linux
# wheels made with `pip download --platform manylinux2014_x86_64 --python-version 38 ...`.
if ! pip install -q -r "$REQ"; then
    if [ -n "${BGR_WHEELS:-}" ]; then
        pip install -q --no-index --find-links "$BGR_WHEELS" -r "$REQ"
    else
        exit 1
    fi
fi
set -u
echo "env ${ENV_NAME}: $(python -c 'import sys, numpy, scipy, nibabel, yaml, matplotlib, pytest; print("python", sys.version.split()[0], "numpy", numpy.__version__, "scipy", scipy.__version__, "nibabel", nibabel.__version__, "pyyaml", yaml.__version__, "matplotlib", matplotlib.__version__, "pytest", pytest.__version__)')"
while IFS= read -r f; do
    bash -n "$f"
done < <(find "$REPO/preprocessing" -name '*.sh')
echo "bash -n OK"
cd "$REPO"
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
python -m pytest preprocessing/tests -q -p no:cacheprovider "$@"
python -m preprocessing --help >/dev/null
for c in check-tools build-manifest build-target-grid run synthseg qc-summary export-npy export-merge export-check cleanup n-chunks; do
    python -m preprocessing "$c" --help >/dev/null
done
echo "ALL CHECKS PASSED (${ENV_NAME})"
