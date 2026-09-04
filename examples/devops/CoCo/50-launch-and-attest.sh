#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
load_config
require_root_or_sudo
prepare_download_dir

signed_env="$STATE_DIR/signed-image.env"
[[ -r "$signed_env" ]] || die "Run 40-sign-image.sh first: $signed_env is missing"
# shellcheck disable=SC1090
source "$signed_env"
case "${COSIGN_TLOG_MODE:-disabled}" in
  disabled|rekor) ;;
  *) die "Signed-image state contains an invalid COSIGN_TLOG_MODE" ;;
esac
case "$TEE_PLATFORM" in
  snp)
    EXPECTED_SNP_LAUNCH_MEASUREMENT="${EXPECTED_SNP_LAUNCH_MEASUREMENT,,}"
    [[ "$EXPECTED_SNP_LAUNCH_MEASUREMENT" =~ ^[0-9a-f]{96}$ ]] ||
      die "EXPECTED_SNP_LAUNCH_MEASUREMENT must be the approved 48-byte SNP launch measurement (96 lowercase hex characters)"
    ;;
  tdx)
    for reference_name in EXPECTED_TDX_MRTD EXPECTED_TDX_RTMR0 EXPECTED_TDX_RTMR1 EXPECTED_TDX_RTMR2 EXPECTED_TDX_RTMR3; do
      printf -v "$reference_name" '%s' "${!reference_name,,}"
      [[ "${!reference_name}" =~ ^[0-9a-f]{96}$ ]] ||
        die "$reference_name must be an independently approved 48-byte TDX reference value (96 lowercase hex characters)"
    done
    ;;
esac
CONFIDENTIAL_VOLUME_SIZE="${CONFIDENTIAL_VOLUME_SIZE:-8Gi}"
[[ "$CONFIDENTIAL_VOLUME_SIZE" =~ ^[1-9][0-9]*(Ki|Mi|Gi|Ti)$ ]] ||
  die "CONFIDENTIAL_VOLUME_SIZE must be a positive binary Kubernetes quantity such as 8Gi"

report_dir="${1:-$STATE_DIR/attestation-reports-$(date -u +%Y%m%dT%H%M%SZ)}"
driver="$SCRIPT_DIR/lib/deploy-coco-attest.sh"
[[ -x "$driver" ]] || die "Attestation driver is missing or not executable: $driver"

if ! as_root containerd config dump |
  awk -v needle="runtimes.${RUNTIME_CLASS}]" 'index($0, needle) {found=1} END {exit !found}'; then
  die "containerd has no runtime handler for ${RUNTIME_CLASS}; rerun 20-install-coco-gpu.sh"
fi
if ! as_root ctr plugins ls |
  awk '$1 == "io.containerd.snapshotter.v1" && $2 == "nydus-for-kata-tee" && $4 == "ok" {found=1} END {exit !found}'; then
  die "containerd snapshotter nydus-for-kata-tee is not ready; rerun 20-install-coco-gpu.sh"
fi

log "Launching a signed-image CoCo pod and collecting CPU/GPU evidence"
driver_args=(--kubeconfig "$KUBECONFIG_PATH" --output-dir "$report_dir")
[[ "$KEEP_ATTESTATION_POD" == 1 ]] || driver_args+=(--cleanup)
as_root env \
  KUBECONFIG="$KUBECONFIG_PATH" \
  TEE_PLATFORM="$TEE_PLATFORM" \
  POD_IMAGE="$SIGNED_IMAGE" \
  POLICY_REPOSITORY="$POLICY_REPOSITORY" \
  REGISTRY_HOST="$REGISTRY_HOST" \
  COSIGN_PUBLIC_KEY="$STATE_DIR/security/cosign.pub" \
  REGISTRY_CA_CERT="$STATE_DIR/registry-tls/ca.crt" \
  RUNTIME_CLASS="$RUNTIME_CLASS" \
  GPU_RESOURCE="$GPU_RESOURCE" \
  GPU_COUNT="$GPU_COUNT" \
  POD_MEMORY="$POD_MEMORY" \
  CONFIDENTIAL_VOLUME_SIZE="$CONFIDENTIAL_VOLUME_SIZE" \
  AMD_KDS_PRODUCT="$AMD_KDS_PRODUCT" \
  EXPECTED_SNP_LAUNCH_MEASUREMENT="$EXPECTED_SNP_LAUNCH_MEASUREMENT" \
  EXPECTED_TDX_MRTD="${EXPECTED_TDX_MRTD:-}" \
  EXPECTED_TDX_RTMR0="${EXPECTED_TDX_RTMR0:-}" \
  EXPECTED_TDX_RTMR1="${EXPECTED_TDX_RTMR1:-}" \
  EXPECTED_TDX_RTMR2="${EXPECTED_TDX_RTMR2:-}" \
  EXPECTED_TDX_RTMR3="${EXPECTED_TDX_RTMR3:-}" \
  KBS_NAMESPACE="$SECURITY_NAMESPACE" \
  KBS_SERVICE="$KBS_SERVICE" \
  ARTIFACT_CACHE_DIR="$STATE_DIR/downloads" \
  IGNORE_CHECKSUM_MISMATCH="$IGNORE_CHECKSUM_MISMATCH" \
  COSIGN_TLOG_MODE="${COSIGN_TLOG_MODE:-disabled}" \
  "$driver" "${driver_args[@]}"

as_root chown -R "$(id -u):$(id -g)" "$report_dir"
ln -sfn "$report_dir" "$STATE_DIR/latest-attestation-reports"
log "Human-readable reports are in $report_dir"
sed -n '/Overall CPU attestation result:/p' "$report_dir/cpu-attestation-report.txt"
sed -n '/Overall GPU attestation result:/p' "$report_dir/gpu-attestation-report.txt"
sed -n '/Overall confidential storage result:/p' "$report_dir/storage-verification-report.txt"
