#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

stages=(
  00-verify-host.sh
  10-install-kubernetes.sh
  20-install-coco-gpu.sh
  30-deploy-security-services.sh
  40-sign-image.sh
  50-launch-and-attest.sh
)

for stage in "${stages[@]}"; do
  printf '\n========== %s ==========\n' "$stage"
  "$SCRIPT_DIR/$stage"
done

echo
echo "End-to-end CoCo deployment and attestation completed."
echo "Reports: $SCRIPT_DIR/state/latest-attestation-reports"
