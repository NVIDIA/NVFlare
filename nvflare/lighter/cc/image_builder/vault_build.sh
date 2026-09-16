#!/bin/sh
# Build a fresh application vault and a self-contained CVM delivery for every release/site.
set -eu
builder_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$builder_dir${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${CVM_BUILDER_PYTHON:-}" ]; then
    python=$CVM_BUILDER_PYTHON
elif [ -x "$builder_dir/.venv/bin/python" ]; then
    python=$builder_dir/.venv/bin/python
else
    python=python3
fi
exec "$python" -m builder.vault "$@"
