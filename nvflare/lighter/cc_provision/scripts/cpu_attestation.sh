#!/bin/bash
set -e

CERT_DIR=/opt/certs
REPORT=report.bin
REQUEST=request.bin

echo "==> Generating attestation report..."
snpguest report "$REPORT" "$REQUEST" --random

echo "==> Fetching AMD CA certs..."
snpguest fetch ca pem "$CERT_DIR" milan

echo "==> Fetching VCEK certificate..."
snpguest fetch vcek pem "$CERT_DIR" "$REPORT"

echo "==> Verifying attestation..."
snpguest verify attestation "$CERT_DIR" "$REPORT"

if [ $? -eq 0 ]; then
    echo "Attestation PASSED"
else
    echo "Attestation FAILED"
    exit 1
fi

