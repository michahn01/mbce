#!/usr/bin/env bash
# Provision Python 3.11 and the pinned JAX runtime in a project-local environment.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME=cuda12
case "${1:-}" in
    '') ;;
    --cpu) RUNTIME=cpu ;;
    --cuda=12) RUNTIME=cuda12 ;;
    --cuda=13) RUNTIME=cuda13 ;;
    -h|--help) echo 'Usage: ./setup_vm.sh [--cpu | --cuda=12 | --cuda=13]'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
esac
[[ $# -le 1 ]] || { echo 'Expected at most one runtime option.' >&2; exit 1; }
if [[ "$RUNTIME" != cpu ]]; then
    command -v nvidia-smi >/dev/null || { echo 'Install an NVIDIA driver before GPU setup.' >&2; exit 1; }
    nvidia-smi
fi
if command -v apt-get >/dev/null; then
    SUDO=()
    if [[ "$(id -u)" != 0 ]]; then SUDO=(sudo); fi
    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" apt-get install -y python3-venv libegl1 libgl1 libopengl0
fi
python3 -m venv "$ROOT/.bootstrap"
"$ROOT/.bootstrap/bin/python" -m pip install --upgrade uv
export UV_PYTHON_INSTALL_DIR="$ROOT/.python"
UV="$ROOT/.bootstrap/bin/uv"
"$UV" python install 3.11
if [[ ! -d "$ROOT/.venv" ]]; then
    "$UV" venv --seed --python 3.11 "$ROOT/.venv"
fi
PYTHON="$ROOT/.venv/bin/python"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "The project environment must use Python 3.11"'
"$PYTHON" -m pip install -r "$ROOT/requirements.txt"
"$PYTHON" -m pip uninstall -y jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt
if [[ "$RUNTIME" != cpu ]]; then
    "$PYTHON" -m pip install "jax[$RUNTIME]==0.7.2"
fi
# Use the CUDA libraries installed with JAX instead of system toolkit overrides.
unset LD_LIBRARY_PATH
if [[ "$RUNTIME" == cpu ]]; then export JAX_PLATFORMS=cpu; else export JAX_PLATFORMS=cuda; fi
"$PYTHON" -c 'import jax; print("JAX devices:", jax.devices())'
echo 'Ready. Run ./launch.sh run --dry-run to inspect the suite.'
