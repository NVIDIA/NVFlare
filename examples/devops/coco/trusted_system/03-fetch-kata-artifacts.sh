#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

[[ $# -eq 1 ]] || {
    printf 'Usage: %s /path/to/platform-reference.env\n' "$0" >&2
    exit 2
}
CONFIG_FILE="$(realpath -- "$1")"
[[ -s "${CONFIG_FILE}" ]] || { printf 'Missing configuration: %s\n' "${CONFIG_FILE}" >&2; exit 1; }
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }
for command in docker helm python3 sha256sum; do need "${command}"; done

[[ "${PLATFORM_PROFILE-}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || die 'invalid PLATFORM_PROFILE'
[[ "${KATA_VERSION-}" == "3.29.0" ]] || die 'this kit is pinned to Kata 3.29.0'
[[ "${KATA_CHART_OCI_DIGEST-}" =~ ^sha256:[0-9a-f]{64}$ ]] || die 'invalid chart OCI digest'
[[ "${KATA_CHART_TGZ_SHA256-}" =~ ^[0-9a-f]{64}$ ]] || die 'invalid chart archive SHA-256'
[[ "${KATA_DEPLOY_AMD64-}" =~ @sha256:[0-9a-f]{64}$ ]] || die 'Kata deploy image must use an amd64 manifest digest'

PROFILE_DIR="${PLATFORM_WORK_ROOT:?PLATFORM_WORK_ROOT is required}/${PLATFORM_PROFILE}"
ARTIFACT_DIR="${PROFILE_DIR}/artifacts"
KATA_ROOT="${ARTIFACT_DIR}/opt/kata"
CHART_FILE="${ARTIFACT_DIR}/kata-deploy-${KATA_VERSION}.tgz"
[[ ! -e "${PROFILE_DIR}" ]] || die "profile directory already exists; preserve it and choose a new PLATFORM_PROFILE: ${PROFILE_DIR}"
install -d -m 0700 "${ARTIFACT_DIR}"

printf 'Fetching pinned Kata chart %s\n' "${KATA_VERSION}"
PULL_OUTPUT="$(helm pull "${KATA_CHART_OCI}" --version "${KATA_VERSION}" --destination "${ARTIFACT_DIR}" 2>&1)"
printf '%s\n' "${PULL_OUTPUT}"
grep -Fq "Digest: ${KATA_CHART_OCI_DIGEST}" <<<"${PULL_OUTPUT}" \
    || die 'Helm reported a different chart OCI digest'
printf '%s  %s\n' "${KATA_CHART_TGZ_SHA256}" "${CHART_FILE}" | sha256sum --check

printf 'Pulling pinned amd64 Kata deploy image\n'
docker pull "${KATA_DEPLOY_AMD64}"
docker image inspect "${KATA_DEPLOY_AMD64}" >/dev/null

CONTAINER_NAME="platform-reference-${PLATFORM_PROFILE//[^a-zA-Z0-9_.-]/-}-$$"
cleanup() { docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true; }
trap cleanup EXIT
docker create --name "${CONTAINER_NAME}" "${KATA_DEPLOY_AMD64}" /bin/true >/dev/null
install -d -m 0700 "${KATA_ROOT}"
# Kata deploy 3.29.0 is an artifact image. Its host /opt/kata payload is
# staged beneath /opt/kata-artifacts inside the image.
docker cp "${CONTAINER_NAME}:/opt/kata-artifacts/opt/kata/." "${KATA_ROOT}"
docker rm "${CONTAINER_NAME}" >/dev/null
trap - EXIT

helm template kata-deploy "${CHART_FILE}" \
    --namespace kata-system \
    --set node-feature-discovery.enabled=false \
    --set-string "image.reference=${KATA_DEPLOY_AMD64}" \
    > "${PROFILE_DIR}/rendered-kata-deploy.yaml"
grep -Fq "${KATA_DEPLOY_AMD64}" "${PROFILE_DIR}/rendered-kata-deploy.yaml" \
    || die 'rendered chart does not use the pinned Kata deploy image'

KATA_CONFIG="$(find "${KATA_ROOT}" -type f -name configuration-qemu-nvidia-gpu-snp.toml -print -quit)"
[[ -n "${KATA_CONFIG}" ]] || die 'GPU-SNP Kata configuration not found in the pinned image'
printf '%s\n' "${KATA_CONFIG#${PROFILE_DIR}/}" > "${PROFILE_DIR}/kata-config-relative-path.txt"

python3 - "${KATA_CONFIG}" <<'PY' | tee "${PROFILE_DIR}/kata-measurement-input-review.txt"
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    config = tomllib.load(stream)
qemu = config["hypervisor"]["qemu"]
for field in (
    "path", "firmware", "firmware_volume", "kernel", "initrd", "image",
    "kernel_params", "machine_type", "cpu_features", "default_vcpus",
):
    print(f"{field}={qemu.get(field)!r}")
PY

{
    printf 'platform_profile=%s\n' "${PLATFORM_PROFILE}"
    printf 'kata_version=%s\n' "${KATA_VERSION}"
    printf 'chart_oci_digest=%s\n' "${KATA_CHART_OCI_DIGEST}"
    printf 'chart_archive_sha256=%s\n' "${KATA_CHART_TGZ_SHA256}"
    printf 'kata_deploy_amd64=%s\n' "${KATA_DEPLOY_AMD64}"
    printf 'helm=%s\n' "$(helm version --short)"
    printf 'docker=%s\n' "$(docker version --format '{{.Client.Version}}')"
    printf 'python=%s\n' "$(python3 --version 2>&1)"
} > "${PROFILE_DIR}/collection-metadata.txt"

find "${KATA_ROOT}" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${PROFILE_DIR}/kata-artifacts.sha256"

printf '\nCollected immutable inputs in %s\n' "${PROFILE_DIR}"
printf 'Next: install runtime (04), define profile (05), prepare (06), rehearse (07),\n'
printf 'repeat (08), finalize (09), then export with stage 10. See SEC-SYS-LAUNCH-PROFILE.md for workload-profile capture and repeat verification.\n'
