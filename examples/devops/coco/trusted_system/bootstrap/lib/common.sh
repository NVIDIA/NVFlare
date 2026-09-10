#!/usr/bin/env bash
# Cluster-only bootstrap helpers; no secure-services deployment.
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${COCO_CONFIG:-${SUITE_DIR}/config.env}"
STATE_DIR="${COCO_STATE_DIR:-${HOME}/.local/state/coco-bootstrap}"
load_config() {
  [[ -s "$CONFIG_FILE" ]] || die "Copy config.env.example to $CONFIG_FILE, chmod 600, and review it first"
  source "$CONFIG_FILE"
  source "$SUITE_DIR/../lib/validate-config.sh"
  validate_target_host
  [[ ${TEE_PLATFORM:-} == snp && ${RUNTIME_CLASS:-} == kata-qemu-nvidia-gpu-snp ]] || die "This package supports AMD SNP plus NVIDIA confidential GPU only"
  [[ ${IGNORE_CHECKSUM_MISMATCH:-0} == 0 ]] || die "Checksum bypass is not supported"
  IGNORE_CHECKSUM_MISMATCH=0
  [[ ${ALLOW_PACKAGE_DOWNGRADES:-0} =~ ^[01]$ ]] || die "Invalid downgrade option"
  ALLOW_PACKAGE_DOWNGRADES=${ALLOW_PACKAGE_DOWNGRADES:-0}
  TEE_NAME="AMD SEV-SNP"
  TEE_NODE_LABEL_KEY=amd.feature.node.kubernetes.io/snp
  if [[ -z ${NODE_IP:-} ]]; then
    NODE_IP="$(ip -4 route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')"
  fi
  python3 - "$NODE_IP" "$POD_CIDR" "$SERVICE_CIDR" "$CLUSTER_DNS" <<'CHECK'
import ipaddress
import sys
ipaddress.IPv4Address(sys.argv[1])
pod = ipaddress.IPv4Network(sys.argv[2])
service = ipaddress.IPv4Network(sys.argv[3])
assert not pod.overlaps(service), "Pod and service CIDRs overlap"
assert ipaddress.IPv4Address(sys.argv[4]) in service, "Cluster DNS must be in the service subnet"
CHECK
  validate_private_root "$STATE_DIR"
  export NODE_IP TEE_PLATFORM TEE_NAME TEE_NODE_LABEL_KEY RUNTIME_CLASS
  export IGNORE_CHECKSUM_MISMATCH ALLOW_PACKAGE_DOWNGRADES
  mkdir -p "$STATE_DIR"
}

log() { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"; }
as_root() { if ((EUID == 0)); then "$@"; else sudo "$@"; fi; }

prepare_download_dir() {
  local owner_uid owner_gid
  owner_uid="${SUDO_UID:-$(id -u)}"
  owner_gid="${SUDO_GID:-$(id -g)}"
  as_root install -d -m 0775 "$STATE_DIR/downloads"
  as_root chown "$owner_uid:$owner_gid" "$STATE_DIR/downloads"
}

kctl() {
  if ((EUID == 0)); then kubectl --kubeconfig "$KUBECONFIG_PATH" "$@"
  elif [[ -r "$KUBECONFIG_PATH" ]]; then kubectl --kubeconfig "$KUBECONFIG_PATH" "$@"
  else sudo kubectl --kubeconfig "$KUBECONFIG_PATH" "$@"
  fi
}

helmctl() {
  if ((EUID == 0)); then helm --kubeconfig "$KUBECONFIG_PATH" "$@"
  elif [[ -r "$KUBECONFIG_PATH" ]]; then helm --kubeconfig "$KUBECONFIG_PATH" "$@"
  else sudo helm --kubeconfig "$KUBECONFIG_PATH" "$@"
  fi
}

# Populate a persistent cache atomically. A missing/empty artifact is downloaded;
# a verified artifact is also redownloaded whenever its checksum is wrong.
ensure_download() {
  local url=$1 out=$2 partial
  [[ -s "$out" ]] && return 0
  mkdir -p "$(dirname "$out")"
  partial="${out}.partial.$$"
  if ! curl -L --fail --silent --show-error "$url" -o "$partial"; then
    rm -f "$partial"
    return 1
  fi
  mv "$partial" "$out"
}

ensure_download_verified() {
  local url=$1 sha=$2 out=$3 partial
  if [[ -s "$out" ]]; then
    if echo "$sha  $out" | sha256sum --check --status; then
      return 0
    fi
    if [[ "${IGNORE_CHECKSUM_MISMATCH:-0}" == 1 ]]; then
      echo "WARNING: ignoring checksum mismatch for cached artifact $out (expected SHA-256: $sha)" >&2
      return 0
    fi
  fi
  mkdir -p "$(dirname "$out")"
  partial="${out}.partial.$$"
  if ! curl -L --fail --silent --show-error "$url" -o "$partial"; then
    rm -f "$partial"
    return 1
  fi
  if ! echo "$sha  $partial" | sha256sum -c -; then
    if [[ "${IGNORE_CHECKSUM_MISMATCH:-0}" == 1 ]]; then
      echo "WARNING: ignoring checksum mismatch for downloaded artifact $url (expected SHA-256: $sha)" >&2
      mv "$partial" "$out"
      return 0
    fi
    rm -f "$partial"
    return 1
  fi
  mv "$partial" "$out"
}

wait_for() {
  local description=$1 timeout=$2; shift 2
  local deadline=$((SECONDS + timeout))
  until "$@"; do
    ((SECONDS < deadline)) || die "Timed out waiting for $description"
    sleep 10
  done
}

require_root_or_sudo() {
  if ((EUID != 0)); then
    need sudo
    sudo -n true ||
      die "Passwordless non-interactive sudo is required; run as root or configure NOPASSWD for this workflow"
  fi
}
