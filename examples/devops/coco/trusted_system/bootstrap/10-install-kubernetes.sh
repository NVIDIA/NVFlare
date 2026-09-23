#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export COCO_BOOTSTRAP_DIR="${SCRIPT_DIR}"
exec "${SCRIPT_DIR}/../../shared/bootstrap/10-install-kubernetes.sh" "$@"
