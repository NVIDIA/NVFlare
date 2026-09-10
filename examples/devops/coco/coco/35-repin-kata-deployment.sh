#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

[[ $# -eq 1 ]] || {
    printf 'Usage: %s /path/to/kata-deploy-3.29.0.tgz\n' "$0" >&2
    exit 2
}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=public/kata-platform.env
source "${SCRIPT_DIR}/public/kata-platform.env"
CHART_FILE="$(realpath -- "$1")"

require_sudo
for command in helm kubectl sha256sum; do need_cmd "${command}"; done
need_file "${CHART_FILE}"
printf '%s  %s\n' "${KATA_CHART_TGZ_SHA256}" "${CHART_FILE}" | sha256sum --check

helmctl upgrade --install kata-deploy "${CHART_FILE}" \
    --namespace kata-system \
    --create-namespace \
    --reset-values \
    --set node-feature-discovery.enabled=false \
    --set-string "image.reference=${KATA_DEPLOY_AMD64}" \
    --wait \
    --timeout 15m

kctl -n kata-system rollout status daemonset/kata-deploy --timeout=15m
LIVE_IMAGE="$(kctl -n kata-system get daemonset kata-deploy \
    -o jsonpath='{.spec.template.spec.containers[?(@.name=="kube-kata")].image}')"
[[ "${LIVE_IMAGE}" == "${KATA_DEPLOY_AMD64}" ]] || \
    die "Kata DaemonSet image is ${LIVE_IMAGE}, expected ${KATA_DEPLOY_AMD64}"
kctl get runtimeclass "${RUNTIME_CLASS}" >/dev/null
sudo test -x /opt/kata/bin/containerd-shim-kata-v2 \
    || die 'Kata shim was not installed on the host'

printf 'Kata chart archive and amd64 deployment image are digest pinned.\n'
printf 'Chart SHA-256: %s\nImage: %s\n' "${KATA_CHART_TGZ_SHA256}" "${KATA_DEPLOY_AMD64}"
printf 'Trustee still decides whether the launched guest measurement is authorized.\n'
