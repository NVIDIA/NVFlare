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
selected_runtime="$RUNTIME_CLASS"
source "$SCRIPT_DIR/bootstrap/lib/common.sh"
load_config
[[ "$RUNTIME_CLASS" == "$selected_runtime" ]] || die 'platform.env and config.env select different runtimes'
# shellcheck source=public/kata-platform.env
source "${SCRIPT_DIR}/public/kata-platform.env"
# Legacy artifact pin files must not override the selected RuntimeClass.
RUNTIME_CLASS="$selected_runtime"
CHART_FILE="$(realpath -- "$1")"

require_sudo
for command in helm kubectl sha256sum python3; do need_cmd "${command}"; done
config_name="$(python3 "$SCRIPT_DIR/lib/kata-runtime-profile.py" target "$RUNTIME_CLASS" --field config_name)"
need_file "${CHART_FILE}"
printf '%s  %s\n' "${KATA_CHART_TGZ_SHA256}" "${CHART_FILE}" | sha256sum --check
validate_runtime_prerequisites "$RUNTIME_CLASS"

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
as_root test -x /opt/kata/bin/containerd-shim-kata-v2 \
    || die 'Kata shim was not installed on the host'
RUNTIME_CONFIG="/opt/kata/share/defaults/kata-containers/$config_name"
as_root python3 "$SCRIPT_DIR/lib/kata-runtime-profile.py" enable "$RUNTIME_CONFIG"
as_root python3 "$SCRIPT_DIR/lib/kata-runtime-profile.py" check-target "$RUNTIME_CLASS" "$RUNTIME_CONFIG"
as_root python3 "$SCRIPT_DIR/lib/kata-runtime-profile.py" check "$RUNTIME_CONFIG" --runtime /opt/kata/bin/kata-runtime
validate_runtime_prerequisites "$RUNTIME_CLASS" "$RUNTIME_CONFIG"

printf 'Kata chart archive and amd64 deployment image are digest pinned.\n'
printf 'Chart SHA-256: %s\nImage: %s\n' "${KATA_CHART_TGZ_SHA256}" "${KATA_DEPLOY_AMD64}"
printf 'Trustee still decides whether the launched guest measurement is authorized.\n'
