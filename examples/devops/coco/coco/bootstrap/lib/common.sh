#!/usr/bin/env bash
# Preserve role-specific configuration and state while using the shared implementation.
COCO_BOOTSTRAP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" \
    source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)/shared/bootstrap/lib/common.sh"
