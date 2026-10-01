#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${MBCE_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo 'Run ./setup_vm.sh first, or set MBCE_PYTHON to a Python executable.' >&2
    exit 1
fi
exec "$PYTHON" "$ROOT/launch.py" "$@"
