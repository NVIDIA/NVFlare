#!/bin/bash
set -e

echo "[Attestation] Starting CPU attestation..."
if ! /opt/scripts/cpu_attestation.sh; then
    echo "[Attestation] CPU attestation failed. Powering off."
    poweroff
fi


# Run GPU attestation if script is present and executable
if [ -f /vault/scripts/gpu_attestation.py ]; then
    echo "[Attestation] Starting GPU attestation..."
    if ! python3 /vault/scripts/gpu_attestation.py; then
        echo "[Attestation] GPU attestation failed. Powering off."
        poweroff
    fi
fi

echo "[Attestation] All attestation checks passed."
exit 0

