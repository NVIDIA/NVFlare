#!/bin/bash
set -e

echo "[Attestation] Starting CPU attestation..."
if ! /opt/scripts/cpu_attestation.sh; then
    echo "[Attestation] CPU attestation failed. Powering off."
    poweroff
fi

echo "[Attestation] Starting GPU attestation..."
if ! python3 /opt/scripts/gpu_attestation.py; then
    echo "[Attestation] GPU attestation failed. Powering off."
    poweroff
fi

echo "[Attestation] All attestation checks passed."
exit 0

