#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

[[ $# -eq 1 && ( "$1" == "--approve-pinned-platform" || "$1" == "--approve-pinned-snp-platform" ) ]] || {
    printf 'Usage: %s --approve-pinned-platform\n' "$0" >&2
    exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
require_sudo
need_file "${KBS_CLIENT}"
need_file "${TRUSTEE_PUBLIC_CERT}"
need_file "${ADMIN_TOKEN}"
need_file "${SCRIPT_DIR}/policies/default_cpu.rego"
need_file "${SCRIPT_DIR}/policies/default_gpu.rego"
need_file "${KBS_POLICY_DIR}/resource-policy.rego"
GPU_POLICY_PATH="${AS_STORAGE_DIR}/attestation_service_policy/default_gpu.rego"
need_file "${GPU_POLICY_PATH}"

lock_platform_reference_update
# Only the pinned Trustee v0.22 default (blob 0cd04bb6) or this reviewed
# three-claim policy may be replaced. An unknown active policy needs a
# separate security review; never overwrite it during CPU setup.
GPU_ACTIVE_SHA256=$(sha256sum "$GPU_POLICY_PATH" | awk '{print $1}')
GPU_REVIEWED_SHA256=$(sha256sum "${SCRIPT_DIR}/policies/default_gpu.rego" | awk '{print $1}')
[[ "$GPU_ACTIVE_SHA256" == a6a9ff392be66d75f8f80bd094af4872deba7440efaad1aff141980da81dbe9f \
    || "$GPU_ACTIVE_SHA256" == "$GPU_REVIEWED_SHA256" ]] || \
    die 'Active GPU policy is neither pinned Trustee v0.22 nor the reviewed CoCo policy'
PLATFORM_VALUES_JSON="$(configured_platform_values)"
VALUES_FILE="$(mktemp)"
trap 'rm -f -- "$VALUES_FILE"' EXIT
printf '%s\n' "$PLATFORM_VALUES_JSON" > "$VALUES_FILE"
APPROVED_TEE=$(python3 "$SCRIPT_DIR/lib/platform-reference-values.py" tee "$VALUES_FILE")
[[ "$1" != --approve-pinned-snp-platform || "$APPROVED_TEE" == snp ]] || \
    die 'Use --approve-pinned-platform for a TDX reference set'
[[ "$(stat -c '%a' "${ADMIN_TOKEN}")" == 600 ]] \
    || die "KBS admin token must be mode 0600"

printf 'Installing independently reviewed SNP/TDX CPU policy and approved %s references:\n' "$APPROVED_TEE"
python3 "$SCRIPT_DIR/lib/platform-reference-values.py" validate "$VALUES_FILE"

# Initial policy setup is distinct from stage 02 reference-only updates.
# Existing approved references stay approved; unconfigured TEE branches deny.
# Stage 09 leaves the workload resource policy unchanged and upgrades only
# the pinned upstream GPU default to the reviewed exact-vector equivalent.
kbs_admin set-attestation-policy --type rego --id default_cpu \
    --policy-file "${SCRIPT_DIR}/policies/default_cpu.rego" >/dev/null
kbs_admin set-attestation-policy --type rego --id default_gpu \
    --policy-file "${SCRIPT_DIR}/policies/default_gpu.rego" >/dev/null

install_platform_references "$VALUES_FILE"

need_file "${AS_STORAGE_DIR}/attestation_service_policy/default_cpu.rego"
need_file "${AS_STORAGE_DIR}/attestation_service_policy/default_gpu.rego"
cmp --silent "${SCRIPT_DIR}/policies/default_cpu.rego" \
    "${AS_STORAGE_DIR}/attestation_service_policy/default_cpu.rego" \
    || die "persisted CPU policy differs from the reviewed policy"
cmp --silent "${SCRIPT_DIR}/policies/default_gpu.rego" "$GPU_POLICY_PATH" \
    || die "persisted GPU policy differs from the reviewed policy"

bash "$SCRIPT_DIR/10-verify-platform-reference-values.sh" "$VALUES_FILE"

curl --fail --silent --show-error --cacert "${TRUSTEE_PUBLIC_CERT}" \
    --output /dev/null "${KBS_URL}/healthz"
printf 'Reviewed CPU policy and complete approved %s references installed.\n' "$APPROVED_TEE"
printf 'The reviewed pinned-derived default GPU policy is active with exact three-claim output.\n'
printf 'The KBS resource policy is unchanged; on a fresh installation it remains default-deny.\n'
