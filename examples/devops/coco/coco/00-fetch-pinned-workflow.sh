#!/usr/bin/env bash
# Historical entry-point name retained; nothing is cloned.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
for stage in 00-verify-host.sh 10-install-kubernetes.sh 20-install-coco-gpu.sh; do
    need_file "$COCO_WORKFLOW/$stage"
    bash -n "$COCO_WORKFLOW/$stage"
done
printf 'Vendored cluster bootstrap is present: %s\n' "$COCO_WORKFLOW"
