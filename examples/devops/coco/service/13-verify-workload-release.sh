#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

[[ $# -ge 1 && $# -le 2 ]] || {
    printf 'Usage: %s RELEASE_NAME [DOCKER_LOG_SINCE]\n' "$0" >&2
    exit 2
}
RELEASE_NAME="$1"
LOG_SINCE="${2:-30m}"
[[ "${RELEASE_NAME}" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]] \
    || die 'release name is not a lowercase DNS label'
[[ "${LOG_SINCE}" =~ ^[1-9][0-9]*[smhd]$ ]] \
    || die 'DOCKER_LOG_SINCE must look like 30m, 2h, or 1d'

require_sudo
for command in curl docker grep mktemp python3; do need_cmd "${command}"; done
need_file "${TRUSTEE_PUBLIC_CERT}"

RAW_LOG="$(mktemp)"
CLEAN_LOG="$(mktemp)"
cleanup() {
    rm -f -- "${RAW_LOG}" "${CLEAN_LOG}"
}
trap cleanup EXIT

sudo docker compose -p "${TRUSTEE_PROJECT}" -f "${TRUSTEE_COMPOSE}" \
    logs --no-color --since "${LOG_SINCE}" > "${RAW_LOG}" 2>&1
python3 - "${RAW_LOG}" "${CLEAN_LOG}" <<'PY'
from pathlib import Path
import re
import sys

ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
raw = Path(sys.argv[1]).read_text(errors="replace")
Path(sys.argv[2]).write_text(ansi.sub("", raw))
PY

grep -Eq 'Verifier/endorsement check passed\..*tee=Snp.*tee_class="cpu"' \
    "${CLEAN_LOG}" || die 'no successful AMD SEV-SNP CPU appraisal found'
grep -Eq 'Verifier/endorsement check passed\..*tee=Nvidia.*tee_class="gpu"' \
    "${CLEAN_LOG}" || die 'no successful NVIDIA GPU appraisal found'
grep -Fq 'AttestationEvaluate succeeded.' "${CLEAN_LOG}" \
    || die 'no successful composite attestation evaluation found'

for resource in security-policy sig-public-key image-key; do
    grep -Eq \
        "GET /kbs/v0/resource/default/${resource}/${RELEASE_NAME} HTTP/1\\.1\" 200 .*\"attestation-agent-kbs-client/" \
        "${CLEAN_LOG}" \
        || die "no successful attested release of ${resource} found"
done

curl --fail --silent --show-error --cacert "${TRUSTEE_PUBLIC_CERT}" \
    --output /dev/null "${KBS_URL}/healthz"

printf 'AMD SEV-SNP CPU appraisal: passed\n'
printf 'NVIDIA GPU appraisal: passed\n'
printf 'Composite AttestationEvaluate: succeeded\n'
printf 'Attested KBS releases: security-policy, sig-public-key, image-key\n'
printf 'Workload release verification passed for %s (logs since %s).\n' \
    "${RELEASE_NAME}" "${LOG_SINCE}"
