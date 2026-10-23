#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

[[ $# -eq 2 && "$2" == "--remove-incomplete-policy-build" ]] || {
    printf 'Usage: %s WORKLOAD.env --remove-incomplete-policy-build\n' "$0" >&2
    exit 2
}
export OWNER_CONFIG="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/release.sh
source "${SCRIPT_DIR}/lib/release.sh"

[[ ! -e "${OUTPUT_DIR}/policy-SHA256SUMS" ]] \
    || die "policy generation is complete; refusing recovery cleanup"
[[ ! -e "${HANDOFF_DIR}" ]] \
    || die "a handoff exists; refusing recovery cleanup"

INCOMPLETE_FILES=(
    approved-workload-launch-profile.json
    rules.rego
    genpolicy-settings.json
    image-security-policy.json
    base-initdata.toml
    pod.yaml
    final-initdata.toml
    generated-policy.rego
    expected-initdata-sha256.hex
    resource-policy-fragment.rego
    release-authorization.json
    kata-policy-security-fix.txt
)

printf 'Removing only these incomplete policy-generation files, if present:\n'
for file in "${INCOMPLETE_FILES[@]}"; do
    printf '  %s\n' "${OUTPUT_DIR}/${file}"
done

for file in "${INCOMPLETE_FILES[@]}"; do
    rm -f -- "${OUTPUT_DIR}/${file}"
done
find "${RELEASE_DIR}" -maxdepth 1 -type d -name '.policy-build.*' \
    -empty -delete

printf 'Incomplete policy-generation output removed. Published-image evidence was preserved.\n'
