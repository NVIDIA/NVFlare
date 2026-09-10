#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

[[ $# -eq 4 ]] || {
    printf 'Usage: %s BASE-CONFIG APPROVAL-ENV ATTESTATION-REPORT CERTS-DIR\n' "$0" >&2
    exit 2
}
BASE_CONFIG="$(realpath -- "$1")"
APPROVAL_ENV="$(realpath -- "$2")"
REPORT="$(realpath -- "$3")"
CERTS_DIR="$(realpath -- "$4")"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }
for command in python3 realpath sha256sum snpguest; do need "${command}"; done
[[ -s "${BASE_CONFIG}" ]] || die "missing base configuration: ${BASE_CONFIG}"
[[ -s "${APPROVAL_ENV}" ]] || die "missing approval configuration: ${APPROVAL_ENV}"
[[ -s "${REPORT}" ]] || die "missing attestation report: ${REPORT}"
[[ -d "${CERTS_DIR}" ]] || die "missing certificate directory: ${CERTS_DIR}"

# shellcheck source=/dev/null
source "${BASE_CONFIG}"
[[ "${PLATFORM_PROFILE-}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || die 'invalid PLATFORM_PROFILE'
PROFILE_DIR="${PLATFORM_WORK_ROOT:?PLATFORM_WORK_ROOT is required}/${PLATFORM_PROFILE}"
[[ "$(dirname -- "${APPROVAL_ENV}")" == "${PROFILE_DIR}" ]] \
    || die 'APPROVAL-ENV must be inside the selected platform profile'

# Nothing is recorded unless both the AMD certificate chain and complete report verify.
snpguest verify certs "${CERTS_DIR}"
snpguest verify attestation "${CERTS_DIR}" "${REPORT}"
snpguest verify attestation "${CERTS_DIR}" "${REPORT}" --tcb

EVIDENCE_DIR="${PROFILE_DIR}/reported-tcb-evidence"
[[ ! -e "${EVIDENCE_DIR}" ]] || die "refusing to overwrite: ${EVIDENCE_DIR}"
install -d -m 0700 "${EVIDENCE_DIR}/certs"
install -m 0600 "${REPORT}" "${EVIDENCE_DIR}/attestation-report.bin"
find "${CERTS_DIR}" -maxdepth 1 -type f -exec install -m 0600 -- {} "${EVIDENCE_DIR}/certs/" \;
snpguest display report "${REPORT}" > "${EVIDENCE_DIR}/attestation-report.txt"
(
    cd "${EVIDENCE_DIR}"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum
) > "${EVIDENCE_DIR}/SHA256SUMS"

# AMD SEV-SNP ABI: REPORTED_TCB is the little-endian 64-bit field at offset
# 0x180. Its component bytes are bootloader[7:0], TEE[15:8], reserved[47:16],
# SNP firmware[55:48], and microcode[63:56]. Parse the signed binary report,
# rather than depending on snpguest's human-readable formatting.
read -r bootloader tee snp microcode < <(
    python3 - "${EVIDENCE_DIR}/attestation-report.bin" <<'PY'
import sys
from pathlib import Path

report = Path(sys.argv[1]).read_bytes()
if len(report) < 0x188:
    raise SystemExit(f"attestation report is too short: {len(report)} bytes")
tcb = report[0x180:0x188]
if any(tcb[2:6]):
    raise SystemExit(f"reserved REPORTED_TCB bytes are nonzero: {tcb[2:6].hex()}")
print(tcb[0], tcb[1], tcb[6], tcb[7])
PY
)

python3 - "${APPROVAL_ENV}" "${bootloader}" "${tee}" "${snp}" "${microcode}" <<'PY'
import os
import re
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
updates = {
    "TCB_EVIDENCE_FILE": "reported-tcb-evidence/attestation-report.txt",
    "SNP_MIN_REPORTED_TCB_BOOTLOADER": sys.argv[2],
    "SNP_MIN_REPORTED_TCB_TEE": sys.argv[3],
    "SNP_MIN_REPORTED_TCB_SNP": sys.argv[4],
    "SNP_MIN_REPORTED_TCB_MICROCODE": sys.argv[5],
}
text = path.read_text()
for key, value in updates.items():
    pattern = rf"(?m)^{re.escape(key)}=.*$"
    replacement = f'{key}="{value}"'
    text, count = re.subn(pattern, replacement, text)
    if count != 1:
        raise SystemExit(f"expected exactly one {key} assignment; found {count}")
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
try:
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
except BaseException:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
    raise
PY

printf 'Verified and recorded reported-TCB floors in %s:\n' "${APPROVAL_ENV}"
printf '  bootloader=%s tee=%s snp=%s microcode=%s\n' \
    "${bootloader}" "${tee}" "${snp}" "${microcode}"
printf 'Evidence retained at %s\n' "${EVIDENCE_DIR}"
