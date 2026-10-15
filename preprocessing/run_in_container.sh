#!/bin/bash
# Run `python -m preprocessing <args>` inside the lab's TF 2.10 Singularity
# image (tf10_nv.sif) with FreeSurfer, FSL and ANTs on PATH and a TMPDIR that
# is visible inside the container. Use it on glacier or an MLSC compute node
# (never on mlsc-login).
#
# Usage (same arguments as python -m preprocessing; `bash` avoids needing +x):
#   bash preprocessing/run_in_container.sh check-tools --config "$CFG"
#   bash preprocessing/run_in_container.sh run --config "$CFG" --manifest "$M" --out-root "$D/processed" --rows 0
#   USE_GPU=1 bash preprocessing/run_in_container.sh synthseg --config "$CFG" ...
#
# Settings (environment variables; defaults are the confirmed Martinos values):
#   BGR_SIF              image (/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif)
#   BGR_BINDS            space-separated host paths bound with -B (a path that does
#                        not exist on this host is skipped with a warning)
#                        (/scratch /autofs /cluster /space /vast /usr/local/freesurfer)
#   BGR_REPO_DIR         repo checkout put on PYTHONPATH (default: the folder above this script)
#   BGR_TMPDIR           TMPDIR inside the container (default: /scratch/$SLURM_JOB_ID when it
#                        exists, else /autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg/tmp)
#   USE_GPU              1 = always add --nv, 0 = never, unset/auto = add --nv only when
#                        nvidia-smi lists a GPU (pin one with CUDA_VISIBLE_DEVICES on glacier)
#   BGR_SINGULARITY      singularity or apptainer (default: singularity)
#   BGR_PYTHON           python inside the container (default: python)
#   BGR_FREESURFER_HOME  (/usr/local/freesurfer/dev; needs the /usr/local/freesurfer bind)
#   BGR_FSLDIR           (/autofs/cluster/pubsw/2/pubsw/Linux2-2.3-x86_64/packages/fsl.64bit/5.0.7)
#   BGR_ANTS_BIN         (/autofs/cluster/pubsw/2/pubsw/Linux2-2.3-x86_64/packages/ANTS/2.4.4/bin)
#
# Arguments are handed to python as separate argv entries ("$@" after the
# fixed bash -c script); nothing is interpolated into the script, so paths
# with spaces or shell characters are passed through unchanged. Relative
# paths resolve against the current directory, which must be visible in the
# container (e.g. under /autofs); otherwise the repo dir is used.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DEFAULT=/autofs/space/crater_001/users/jn85/BrainRegProjects/GroupwiseReg
BGR_SIF=${BGR_SIF:-/autofs/space/crater_001/users/jn85/Singularity_Files/tf10_nv.sif}
BGR_BINDS=${BGR_BINDS:-/scratch /autofs /cluster /space /vast /usr/local/freesurfer}
BGR_REPO_DIR=${BGR_REPO_DIR:-$(dirname "$SCRIPT_DIR")}
BGR_SINGULARITY=${BGR_SINGULARITY:-singularity}
BGR_PYTHON=${BGR_PYTHON:-python}
BGR_FREESURFER_HOME=${BGR_FREESURFER_HOME:-/usr/local/freesurfer/dev}
BGR_FSLDIR=${BGR_FSLDIR:-/autofs/cluster/pubsw/2/pubsw/Linux2-2.3-x86_64/packages/fsl.64bit/5.0.7}
BGR_ANTS_BIN=${BGR_ANTS_BIN:-/autofs/cluster/pubsw/2/pubsw/Linux2-2.3-x86_64/packages/ANTS/2.4.4/bin}
if [ -z "${BGR_TMPDIR:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ] && [ -d "/scratch/${SLURM_JOB_ID}" ]; then
        BGR_TMPDIR="/scratch/${SLURM_JOB_ID}"
    else
        BGR_TMPDIR="${PROJ_DEFAULT}/tmp"
    fi
fi

if [ $# -eq 0 ]; then
    echo "usage: $0 <preprocessing command> [options]   (see python -m preprocessing --help)" >&2
    exit 2
fi
if [ ! -f "$BGR_SIF" ]; then
    echo "run_in_container: image not found: $BGR_SIF (set BGR_SIF)" >&2
    exit 2
fi
mkdir -p "$BGR_TMPDIR"

want_gpu() {
    case "${USE_GPU:-auto}" in
        1|yes|true) return 0 ;;
        0|no|false) return 1 ;;
    esac
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU'
}

cmd=("$BGR_SINGULARITY" exec)
if want_gpu; then
    cmd+=(--nv)
fi
read -r -a binds <<< "$BGR_BINDS"
for b in ${binds[@]+"${binds[@]}"}; do
    if [ -e "${b%%:*}" ]; then
        cmd+=(-B "$b")
    else
        echo "run_in_container: skipping bind $b (not found on $(hostname))" >&2
    fi
done

# Fixed script (single-quoted: expanded only inside the container). The user's
# arguments arrive as "$@" (after $0 = run_in_container).
inner='set -e
export PATH="$FREESURFER_HOME/bin:$FSLDIR/bin:$BGR_ANTS_BIN:$PATH"
export PYTHONPATH="$BGR_REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
if ! cd "$BGR_WORKDIR" 2>/dev/null; then
    echo "run_in_container: $BGR_WORKDIR is not visible in the container; running from $BGR_REPO_DIR" >&2
    cd "$BGR_REPO_DIR"
fi
exec "$BGR_PYTHON" -m preprocessing "$@"'

cmd+=("$BGR_SIF" env
      "FREESURFER_HOME=$BGR_FREESURFER_HOME"
      "FSLDIR=$BGR_FSLDIR"
      "FSLOUTPUTTYPE=NIFTI_GZ"
      "BGR_ANTS_BIN=$BGR_ANTS_BIN"
      "BGR_REPO_DIR=$BGR_REPO_DIR"
      "BGR_WORKDIR=$PWD"
      "BGR_PYTHON=$BGR_PYTHON"
      "TMPDIR=$BGR_TMPDIR"
      bash -c "$inner" run_in_container "$@")
exec "${cmd[@]}"
