#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="${1:-$SCRIPT_DIR/requirements.txt}"
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "requirements.txt not found: $REQUIREMENTS_FILE" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON:-python}"
"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install torch==2.6.0 torchvision==0.21.0
"$PYTHON_BIN" -m pip install -r "$REQUIREMENTS_FILE"

echo "Done."
