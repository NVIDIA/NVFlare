#!/usr/bin/env bash
# Shared primitives only: no configuration loading or external side effects.
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command is missing: $1"; }
need() { need_cmd "$@"; }
need_file() { [[ -s "$1" ]] || die "required file is missing or empty: $1"; }
as_root() { if ((EUID == 0)); then "$@"; else sudo "$@"; fi; }
require_sudo() {
    if ((EUID != 0)); then
        need_cmd sudo
        sudo -n true || die 'passwordless non-interactive sudo is required'
    fi
}
kctl() {
    if ((EUID == 0)) || [[ -r "$KUBECONFIG_PATH" ]]; then kubectl --kubeconfig "$KUBECONFIG_PATH" "$@"
    else sudo kubectl --kubeconfig "$KUBECONFIG_PATH" "$@"; fi
}
helmctl() {
    if ((EUID == 0)) || [[ -r "$KUBECONFIG_PATH" ]]; then helm --kubeconfig "$KUBECONFIG_PATH" "$@"
    else sudo helm --kubeconfig "$KUBECONFIG_PATH" "$@"; fi
}
ensure_setup_image() {
    as_root docker image inspect "$1" >/dev/null 2>&1 || as_root docker pull "$1"
}
check_registry_trust() {
    local host="$1"
    need_cmd curl
    need_file "/etc/containerd/certs.d/${host}/hosts.toml"
    need_file "/etc/containerd/certs.d/${host}/ca.crt"
    curl --fail --silent --show-error --connect-timeout 5 --max-time 15 \
        --cacert "/etc/containerd/certs.d/${host}/ca.crt" \
        --output /dev/null "https://${host}/v2/" ||
        die 'Registry TLS/access check failed; complete stage 40 before launch'
}
confirm_action() {
    local expected="$1" prompt="$2" timeout="${3:-120}" answer=''
    read -r -t "$timeout" -p "$prompt" answer || return 1
    [[ "$answer" == "$expected" ]]
}
