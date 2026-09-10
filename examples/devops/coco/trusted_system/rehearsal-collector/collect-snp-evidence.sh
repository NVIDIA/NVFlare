#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

CHALLENGE=/challenge/request-data.bin
WORK=/run/snp-evidence
[[ -r "${CHALLENGE}" ]] || { printf 'missing trusted challenge\n' >&2; exit 1; }
[[ "$(wc -c < "${CHALLENGE}")" -eq 64 ]] || { printf 'challenge is not 64 bytes\n' >&2; exit 1; }
install -d -m 0700 "${WORK}"

# Some Kata guest images expose the SNP misc device through sysfs without
# populating its /dev node. Create that node inside this short-lived trusted
# collector; never accept operator-supplied major/minor values.
if [[ ! -e /dev/sev-guest ]]; then
    SNP_DEVICE_SYSFS=/sys/class/misc/sev-guest/dev
    [[ -r "${SNP_DEVICE_SYSFS}" ]] || {
        printf 'missing /dev/sev-guest and %s\n' "${SNP_DEVICE_SYSFS}" >&2
        exit 1
    }
    IFS=: read -r SNP_MAJOR SNP_MINOR < "${SNP_DEVICE_SYSFS}"
    [[ "${SNP_MAJOR}" =~ ^[0-9]+$ && "${SNP_MINOR}" =~ ^[0-9]+$ ]] || {
        printf 'invalid SNP device identity in %s\n' "${SNP_DEVICE_SYSFS}" >&2
        exit 1
    }
    mknod /dev/sev-guest c "${SNP_MAJOR}" "${SNP_MINOR}"
    chmod 0600 /dev/sev-guest
fi
[[ -c /dev/sev-guest ]] || { printf '/dev/sev-guest is not a character device\n' >&2; exit 1; }

# The caller supplied the challenge; --random is intentionally forbidden.
snpguest report "${WORK}/attestation-report.bin" "${CHALLENGE}"

snpguest display report "${WORK}/attestation-report.bin" > "${WORK}/attestation-report.txt"
cp "${CHALLENGE}" "${WORK}/request-data.bin"
(
    cd "${WORK}"
    find . -type f ! -name evidence.tar.gz ! -name SHA256SUMS -print0 \
        | sort -z | xargs -0 sha256sum > SHA256SUMS
    tar -czf evidence.tar.gz attestation-report.bin attestation-report.txt request-data.bin SHA256SUMS
)

# A single framed line makes collection deterministic and avoids kubectl cp/exec.
printf 'COCO_SNP_EVIDENCE_V1='
base64 -w0 "${WORK}/evidence.tar.gz"
printf '\n'

# Keep this trusted diagnostic guest alive briefly so the host can capture its
# actual QEMU process and correlate it to this Pod's CRI sandbox. No listener.
sleep 45
