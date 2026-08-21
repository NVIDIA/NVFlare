#!/usr/bin/env bash

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${COCO_CONFIG:-${SUITE_DIR}/config.env}"
STATE_DIR="${COCO_STATE_DIR:-${SUITE_DIR}/state}"
readonly COCO_MANAGED_NAMESPACE_LABEL="nvflare.nvidia.com/coco-managed"

detect_tee_platform() {
  if grep -q AuthenticAMD /proc/cpuinfo 2>/dev/null &&
    grep -qw sev_snp /proc/cpuinfo 2>/dev/null; then
    printf 'snp\n'
  elif grep -q GenuineIntel /proc/cpuinfo 2>/dev/null &&
    grep -qw tdx_host_platform /proc/cpuinfo 2>/dev/null; then
    printf 'tdx\n'
  else
    printf 'unknown\n'
  fi
}

load_config() {
  local node_ip_override="${NODE_IP:-}"
  local checksum_override="${IGNORE_CHECKSUM_MISMATCH:-}"
  local downgrade_override="${ALLOW_PACKAGE_DOWNGRADES:-}"
  local tlog_mode_override="${COSIGN_TLOG_MODE:-}"
  local tee_platform_override="${TEE_PLATFORM:-}"
  local runtime_class_override="${RUNTIME_CLASS:-}"
  local nvflare_server_runtime_class_override="${NVFLARE_SERVER_RUNTIME_CLASS:-}"
  local nvflare_client_runtime_class_override="${NVFLARE_CLIENT_RUNTIME_CLASS:-}"
  local snp_measurement_override="${EXPECTED_SNP_LAUNCH_MEASUREMENT:-}"
  local tdx_mrtd_override="${EXPECTED_TDX_MRTD:-}"
  local tdx_rtmr0_override="${EXPECTED_TDX_RTMR0:-}"
  local tdx_rtmr1_override="${EXPECTED_TDX_RTMR1:-}"
  local tdx_rtmr2_override="${EXPECTED_TDX_RTMR2:-}"
  local tdx_rtmr3_override="${EXPECTED_TDX_RTMR3:-}"
  local intel_pcs_api_key_override="${INTEL_PCS_API_KEY:-}"
  local intel_pcs_api_key_file_override="${INTEL_PCS_API_KEY_FILE:-}"
  local confidential_volume_size_override="${CONFIDENTIAL_VOLUME_SIZE:-}"
  local keep_attestation_pod_override="${KEEP_ATTESTATION_POD:-}"
  local nvflare_tdx_config_override="${NVFLARE_TDX_ATTESTATION_CONFIG_FILE:-}"
  local nvflare_gpu_key_file_override="${NVFLARE_GPU_ATTESTATION_SERVICE_KEY_FILE:-}"
  local nvflare_gpu_verifier_override="${NVFLARE_GPU_VERIFIER:-}"
  local nvflare_gpu_nras_url_override="${NVFLARE_GPU_NRAS_URL:-}"
  if [[ ! -f "$CONFIG_FILE" ]]; then
    cp "${SUITE_DIR}/config.env.example" "$CONFIG_FILE"
    echo "Created $CONFIG_FILE from the example configuration." >&2
  fi
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
  [[ -z "$node_ip_override" ]] || NODE_IP="$node_ip_override"
  [[ -z "$checksum_override" ]] || IGNORE_CHECKSUM_MISMATCH="$checksum_override"
  [[ -z "$downgrade_override" ]] || ALLOW_PACKAGE_DOWNGRADES="$downgrade_override"
  [[ -z "$tlog_mode_override" ]] || COSIGN_TLOG_MODE="$tlog_mode_override"
  [[ -z "$tee_platform_override" ]] || TEE_PLATFORM="$tee_platform_override"
  [[ -z "$runtime_class_override" ]] || RUNTIME_CLASS="$runtime_class_override"
  [[ -z "$nvflare_server_runtime_class_override" ]] ||
    NVFLARE_SERVER_RUNTIME_CLASS="$nvflare_server_runtime_class_override"
  [[ -z "$nvflare_client_runtime_class_override" ]] ||
    NVFLARE_CLIENT_RUNTIME_CLASS="$nvflare_client_runtime_class_override"
  [[ -z "$snp_measurement_override" ]] ||
    EXPECTED_SNP_LAUNCH_MEASUREMENT="$snp_measurement_override"
  [[ -z "$tdx_mrtd_override" ]] || EXPECTED_TDX_MRTD="$tdx_mrtd_override"
  [[ -z "$tdx_rtmr0_override" ]] || EXPECTED_TDX_RTMR0="$tdx_rtmr0_override"
  [[ -z "$tdx_rtmr1_override" ]] || EXPECTED_TDX_RTMR1="$tdx_rtmr1_override"
  [[ -z "$tdx_rtmr2_override" ]] || EXPECTED_TDX_RTMR2="$tdx_rtmr2_override"
  [[ -z "$tdx_rtmr3_override" ]] || EXPECTED_TDX_RTMR3="$tdx_rtmr3_override"
  [[ -z "$intel_pcs_api_key_override" ]] || INTEL_PCS_API_KEY="$intel_pcs_api_key_override"
  [[ -z "$intel_pcs_api_key_file_override" ]] ||
    INTEL_PCS_API_KEY_FILE="$intel_pcs_api_key_file_override"
  [[ -z "$confidential_volume_size_override" ]] ||
    CONFIDENTIAL_VOLUME_SIZE="$confidential_volume_size_override"
  [[ -z "$keep_attestation_pod_override" ]] ||
    KEEP_ATTESTATION_POD="$keep_attestation_pod_override"
  [[ -z "$nvflare_tdx_config_override" ]] ||
    NVFLARE_TDX_ATTESTATION_CONFIG_FILE="$nvflare_tdx_config_override"
  [[ -z "$nvflare_gpu_key_file_override" ]] ||
    NVFLARE_GPU_ATTESTATION_SERVICE_KEY_FILE="$nvflare_gpu_key_file_override"
  [[ -z "$nvflare_gpu_verifier_override" ]] || NVFLARE_GPU_VERIFIER="$nvflare_gpu_verifier_override"
  [[ -z "$nvflare_gpu_nras_url_override" ]] || NVFLARE_GPU_NRAS_URL="$nvflare_gpu_nras_url_override"
  TEE_PLATFORM="${TEE_PLATFORM:-auto}"
  if [[ "$TEE_PLATFORM" == auto ]]; then
    TEE_PLATFORM="$(detect_tee_platform)"
  fi
  case "$TEE_PLATFORM" in
    snp)
      TEE_NAME="AMD SEV-SNP"
      TEE_NODE_LABEL_KEY="amd.feature.node.kubernetes.io/snp"
      platform_runtime_class="kata-qemu-nvidia-gpu-snp"
      platform_nvflare_server_runtime_class="kata-qemu-snp"
      ;;
    tdx)
      TEE_NAME="Intel TDX"
      TEE_NODE_LABEL_KEY="intel.feature.node.kubernetes.io/tdx"
      platform_runtime_class="kata-qemu-nvidia-gpu-tdx"
      platform_nvflare_server_runtime_class="kata-qemu-tdx"
      ;;
    *)
      die "Could not detect AMD SEV-SNP or Intel TDX; set TEE_PLATFORM=snp or tdx only on a matching host"
      ;;
  esac
  RUNTIME_CLASS="${RUNTIME_CLASS:-auto}"
  [[ "$RUNTIME_CLASS" != auto ]] || RUNTIME_CLASS="$platform_runtime_class"
  NVFLARE_SERVER_RUNTIME_CLASS="${NVFLARE_SERVER_RUNTIME_CLASS:-auto}"
  [[ "$NVFLARE_SERVER_RUNTIME_CLASS" != auto ]] ||
    NVFLARE_SERVER_RUNTIME_CLASS="$platform_nvflare_server_runtime_class"
  NVFLARE_CLIENT_RUNTIME_CLASS="${NVFLARE_CLIENT_RUNTIME_CLASS:-auto}"
  [[ "$NVFLARE_CLIENT_RUNTIME_CLASS" != auto ]] ||
    NVFLARE_CLIENT_RUNTIME_CLASS="$RUNTIME_CLASS"
  IGNORE_CHECKSUM_MISMATCH="${IGNORE_CHECKSUM_MISMATCH:-0}"
  case "$IGNORE_CHECKSUM_MISMATCH" in
    0|1) ;;
    *) die "IGNORE_CHECKSUM_MISMATCH must be 0 or 1" ;;
  esac
  ALLOW_PACKAGE_DOWNGRADES="${ALLOW_PACKAGE_DOWNGRADES:-0}"
  case "$ALLOW_PACKAGE_DOWNGRADES" in
    0|1) ;;
    *) die "ALLOW_PACKAGE_DOWNGRADES must be 0 or 1" ;;
  esac
  COSIGN_TLOG_MODE="${COSIGN_TLOG_MODE:-disabled}"
  case "$COSIGN_TLOG_MODE" in
    disabled|rekor) ;;
    *) die "COSIGN_TLOG_MODE must be 'disabled' or 'rekor'" ;;
  esac
  KEEP_ATTESTATION_POD="${KEEP_ATTESTATION_POD:-0}"
  case "$KEEP_ATTESTATION_POD" in
    0|1) ;;
    *) die "KEEP_ATTESTATION_POD must be 0 or 1" ;;
  esac
  CALICO_SHA256="${CALICO_SHA256:-a1df919d9721cf667accdc3e72848911b0cb25cfab7d2478ad0c996302c95744}"
  REGISTRY_IMAGE="${REGISTRY_IMAGE:-docker.io/library/registry@sha256:a3d8aaa63ed8681a604f1dea0aa03f100d5895b6a58ace528858a7b332415373}"
  TRUSTEE_IMAGE="${TRUSTEE_IMAGE:-ghcr.io/confidential-containers/staged-images/kbs@sha256:c128232d271c3bc6cdfbd57b1e585b4aaa0c8de6dd987dbf2731786f60405e25}"
  SNPGUEST_VERSION="${SNPGUEST_VERSION:-0.10.0}"
  NVFLARE_COCO_PYTHON_BUILD_BASE="${NVFLARE_COCO_PYTHON_BUILD_BASE:-python:3.12.12-slim-trixie@sha256:f3fa41d74a768c2fce8016b98c191ae8c1bacd8f1152870a3f9f87d350920b7c}"
  NVFLARE_COCO_PYTHON_RUNTIME_BASE="${NVFLARE_COCO_PYTHON_RUNTIME_BASE:-nvcr.io/nvidia/distroless/python:3.12-v4.0.7@sha256:9e7012ce1816b720123ae11fc0edd005ff1373c9eb562ded2b3e9da869ed7bc9}"
  NVFLARE_COCO_PYTHON_MINOR="${NVFLARE_COCO_PYTHON_MINOR:-3.12}"
  SNPGUEST_SHA256="${SNPGUEST_SHA256:-70e700465e3523e67dd5104583dc36cd11eef630c6f04c5b9ccafd6ba2e76ca0}"
  TRUSTAUTHORITY_CLI_VERSION="${TRUSTAUTHORITY_CLI_VERSION:-v1.10.1}"
  TRUSTAUTHORITY_CLI_SHA256="${TRUSTAUTHORITY_CLI_SHA256:-d3875adbee96268471c82dd54f012b726fa8d6eefdd8f3243c0e7650fb55ff4e}"
  NVAT_VERSION="${NVAT_VERSION:-1.2.2}"
  TENSORBOARD_VERSION="${TENSORBOARD_VERSION:-2.20.0}"
  NVAT_REPO_SHA256="${NVAT_REPO_SHA256:-31b0a1646f2bbc08ee599d10dbae106124ef2903f39e37095b96493913b37657}"
  NVAT_REPO_URL="${NVAT_REPO_URL:-https://developer.download.nvidia.com/compute/nvat/1.2.2/local_installers/nvat-local-repo-ubuntu2404-1-2-local_1.0-1_amd64.deb}"
  NVAT_BUILD_BASE="${NVAT_BUILD_BASE:-ubuntu:24.04@sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea}"
  NVFLARE_TDX_ATTESTATION_CONFIG_FILE="${NVFLARE_TDX_ATTESTATION_CONFIG_FILE:-}"
  NVFLARE_GPU_ATTESTATION_SERVICE_KEY_FILE="${NVFLARE_GPU_ATTESTATION_SERVICE_KEY_FILE:-}"
  NVFLARE_GPU_VERIFIER="${NVFLARE_GPU_VERIFIER:-local}"
  case "$NVFLARE_GPU_VERIFIER" in
    local|remote) ;;
    *) die "NVFLARE_GPU_VERIFIER must be 'local' or 'remote'" ;;
  esac
  NVFLARE_GPU_NRAS_URL="${NVFLARE_GPU_NRAS_URL:-https://nras.attestation.nvidia.com}"
  if [[ "$NVFLARE_GPU_VERIFIER" == remote && ! "$NVFLARE_GPU_NRAS_URL" =~ ^https:// ]]; then
    die "NVFLARE_GPU_NRAS_URL must be HTTPS for remote GPU verification"
  fi
  if [[ -z "${NODE_IP:-}" ]]; then
    NODE_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}' || true)"
  fi
  if [[ -z "$NODE_IP" ]]; then
    NODE_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  fi
  [[ -n "$NODE_IP" ]] || die "Could not derive NODE_IP; set it in $CONFIG_FILE"
  REGISTRY_BIND_ADDRESS="${REGISTRY_BIND_ADDRESS:-$NODE_IP}"
  REGISTRY_HOST="${REGISTRY_HOST:-${NODE_IP}:${REGISTRY_PORT}}"
  TARGET_REPOSITORY="${REGISTRY_HOST}/${TARGET_IMAGE_NAME}"
  export NODE_IP REGISTRY_HOST TARGET_REPOSITORY IGNORE_CHECKSUM_MISMATCH ALLOW_PACKAGE_DOWNGRADES COSIGN_TLOG_MODE
  export TEE_PLATFORM TEE_NAME TEE_NODE_LABEL_KEY RUNTIME_CLASS
  export NVFLARE_SERVER_RUNTIME_CLASS NVFLARE_CLIENT_RUNTIME_CLASS
  export EXPECTED_SNP_LAUNCH_MEASUREMENT EXPECTED_TDX_MRTD
  export EXPECTED_TDX_RTMR0 EXPECTED_TDX_RTMR1 EXPECTED_TDX_RTMR2 EXPECTED_TDX_RTMR3
  export INTEL_PCS_API_KEY INTEL_PCS_API_KEY_FILE
  export CONFIDENTIAL_VOLUME_SIZE KEEP_ATTESTATION_POD
  export CALICO_SHA256 REGISTRY_IMAGE TRUSTEE_IMAGE REGISTRY_BIND_ADDRESS
  export SNPGUEST_VERSION SNPGUEST_SHA256 TRUSTAUTHORITY_CLI_VERSION TRUSTAUTHORITY_CLI_SHA256
  export NVFLARE_COCO_PYTHON_BUILD_BASE NVFLARE_COCO_PYTHON_RUNTIME_BASE NVFLARE_COCO_PYTHON_MINOR
  export NVAT_VERSION NVAT_REPO_SHA256 NVAT_REPO_URL NVAT_BUILD_BASE
  export TENSORBOARD_VERSION
  export NVFLARE_TDX_ATTESTATION_CONFIG_FILE NVFLARE_GPU_ATTESTATION_SERVICE_KEY_FILE
  export NVFLARE_GPU_VERIFIER NVFLARE_GPU_NRAS_URL
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

ensure_coco_managed_namespace() {
  local namespace=$1 managed
  [[ "$namespace" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && ${#namespace} -le 63 ]] ||
    die "Namespace must be a lowercase RFC 1123 name: $namespace"

  if kctl get namespace "$namespace" >/dev/null 2>&1; then
    managed="$(
      kctl get namespace "$namespace" \
        -o 'jsonpath={.metadata.labels.nvflare\.nvidia\.com/coco-managed}'
    )"
    [[ "$managed" == true ]] ||
      die "Namespace $namespace already exists without $COCO_MANAGED_NAMESPACE_LABEL=true; refusing to adopt a potentially shared namespace"
    return
  fi

  log "Creating suite-owned namespace $namespace"
  kctl create -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: $namespace
  labels:
    $COCO_MANAGED_NAMESPACE_LABEL: "true"
EOF
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
