#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
load_config
require_root_or_sudo
need kubectl
need openssl
need jq

[[ "$REGISTRY_IMAGE" =~ @sha256:[0-9a-f]{64}$ ]] ||
  die "REGISTRY_IMAGE must be pinned by sha256 digest"
[[ "$TRUSTEE_IMAGE" =~ @sha256:[0-9a-f]{64}$ ]] ||
  die "TRUSTEE_IMAGE must be pinned by sha256 digest"
ip -4 -o addr show | awk '{print $4}' | cut -d/ -f1 | grep -Fxq "$REGISTRY_BIND_ADDRESS" ||
  die "REGISTRY_BIND_ADDRESS is not assigned to this host: $REGISTRY_BIND_ADDRESS"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
download_dir="$STATE_DIR/downloads"
prepare_download_dir
security_dir="$STATE_DIR/security"
tls_dir="$STATE_DIR/registry-tls"
mkdir -p "$security_dir" "$tls_dir"
chmod 0700 "$security_dir" "$tls_dir"

if ! command -v cosign >/dev/null || [[ "$(cosign version 2>/dev/null)" != *"${COSIGN_VERSION#v}"* ]]; then
  log "Installing Cosign ${COSIGN_VERSION}"
  cosign_download="$download_dir/cosign-${COSIGN_VERSION}-linux-amd64"
  ensure_download_verified "https://github.com/sigstore/cosign/releases/download/${COSIGN_VERSION}/cosign-linux-amd64" "$COSIGN_SHA256" "$cosign_download"
  as_root install -m 0755 "$cosign_download" /usr/local/bin/cosign
fi

if [[ "${ROTATE_SECURITY_MATERIAL}" == 1 ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  for item in "$security_dir/cosign.key" "$security_dir/cosign.pub" "$tls_dir/ca.key" "$tls_dir/ca.crt" "$tls_dir/ca.srl" "$tls_dir/server.key" "$tls_dir/server.crt"; do
    [[ ! -e "$item" ]] || mv "$item" "$item.$stamp.bak"
  done
fi

if [[ ! -s "$security_dir/cosign.key" || ! -s "$security_dir/cosign.pub" ]]; then
  log "Generating the image-signing key pair"
  COSIGN_PASSWORD="${COSIGN_PASSWORD:-}" cosign generate-key-pair --output-key-prefix "$security_dir/cosign"
fi
chmod 0600 "$security_dir/cosign.key"
chmod 0644 "$security_dir/cosign.pub"

public_key_digest_from_cert() {
  openssl x509 -in "$1" -pubkey -noout 2>/dev/null |
    openssl pkey -pubin -outform DER 2>/dev/null |
    sha256sum | awk '{print $1}'
}

public_key_digest_from_key() {
  openssl pkey -in "$1" -pubout -outform DER 2>/dev/null |
    sha256sum | awk '{print $1}'
}

certificate_key_match() {
  [[ -s "$1" && -s "$2" ]] || return 1
  [[ "$(public_key_digest_from_cert "$1")" == "$(public_key_digest_from_key "$2")" ]]
}

backup_tls_file() {
  local item=$1 stamp=$2
  [[ ! -e "$item" ]] || mv "$item" "$item.replaced-$stamp"
}

tls_repair_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
ca_needs_generation=0
if [[ ! -s "$tls_dir/ca.crt" || ! -s "$tls_dir/ca.key" ]]; then
  ca_needs_generation=1
elif ! openssl x509 -in "$tls_dir/ca.crt" -noout -checkend 86400 >/dev/null 2>&1; then
  ca_needs_generation=1
elif ! certificate_key_match "$tls_dir/ca.crt" "$tls_dir/ca.key"; then
  ca_needs_generation=1
fi

if ((ca_needs_generation == 1)); then
  log "Generating a private registry CA"
  for item in "$tls_dir/ca.key" "$tls_dir/ca.crt" "$tls_dir/ca.srl" "$tls_dir/server.key" "$tls_dir/server.crt"; do
    backup_tls_file "$item" "$tls_repair_stamp"
  done
  openssl genrsa -out "$tls_dir/ca.key" 4096
  openssl req -x509 -new -sha256 -days 3650 -key "$tls_dir/ca.key" \
    -subj "/CN=CoCo Local Registry CA" -out "$tls_dir/ca.crt"
fi

registry_tls_name="${REGISTRY_HOST%:*}"
if [[ "$registry_tls_name" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
  registry_san="IP:${registry_tls_name}"
  registry_check=(-checkip "$registry_tls_name")
else
  registry_san="DNS:${registry_tls_name}"
  registry_check=(-checkhost "$registry_tls_name")
fi

server_needs_generation=0
if [[ ! -s "$tls_dir/server.crt" || ! -s "$tls_dir/server.key" ]]; then
  server_needs_generation=1
elif ! openssl x509 -in "$tls_dir/server.crt" -noout -checkend 86400 >/dev/null 2>&1; then
  server_needs_generation=1
elif ! certificate_key_match "$tls_dir/server.crt" "$tls_dir/server.key"; then
  server_needs_generation=1
elif ! openssl verify -CAfile "$tls_dir/ca.crt" "$tls_dir/server.crt" >/dev/null 2>&1; then
  server_needs_generation=1
elif ! openssl x509 -in "$tls_dir/server.crt" -noout "${registry_check[@]}" >/dev/null 2>&1; then
  server_needs_generation=1
fi

if ((server_needs_generation == 1)); then
  log "Issuing a registry server certificate for ${registry_san}"
  backup_tls_file "$tls_dir/server.key" "$tls_repair_stamp"
  backup_tls_file "$tls_dir/server.crt" "$tls_repair_stamp"
  openssl genrsa -out "$tls_dir/server.key" 3072
  openssl req -new -key "$tls_dir/server.key" -subj "/CN=$registry_tls_name" -out "$tmp_dir/server.csr"
  printf 'subjectAltName=%s\nextendedKeyUsage=serverAuth\n' "$registry_san" >"$tmp_dir/server.ext"
  openssl x509 -req -sha256 -days 825 -in "$tmp_dir/server.csr" \
    -CA "$tls_dir/ca.crt" -CAkey "$tls_dir/ca.key" \
    -CAserial "$tls_dir/ca.srl" -CAcreateserial \
    -extfile "$tmp_dir/server.ext" -out "$tls_dir/server.crt"
fi

certificate_key_match "$tls_dir/server.crt" "$tls_dir/server.key" ||
  die "Registry server certificate and private key do not match"
openssl verify -CAfile "$tls_dir/ca.crt" "$tls_dir/server.crt" >/dev/null ||
  die "Registry server certificate is not signed by the configured CA"
openssl x509 -in "$tls_dir/server.crt" -noout "${registry_check[@]}" >/dev/null ||
  die "Registry certificate SAN does not match $registry_tls_name"
chmod 0600 "$tls_dir/ca.key" "$tls_dir/server.key"
chmod 0644 "$tls_dir/ca.crt" "$tls_dir/server.crt"

log "Deploying the TLS registry"
kctl create namespace "$REGISTRY_NAMESPACE" --dry-run=client -o yaml | kctl apply -f -
kctl -n "$REGISTRY_NAMESPACE" create secret tls coco-registry-tls \
  --cert="$tls_dir/server.crt" --key="$tls_dir/server.key" --dry-run=client -o yaml | kctl apply -f -
registry_manifest="$tmp_dir/registry.yaml"
sed -e "s|@@NAMESPACE@@|$REGISTRY_NAMESPACE|g" \
  -e "s|@@PORT@@|$REGISTRY_PORT|g" \
  -e "s|@@HOST_IP@@|$REGISTRY_BIND_ADDRESS|g" \
  -e "s|@@REGISTRY_IMAGE@@|$REGISTRY_IMAGE|g" \
  -e "s|@@DATA_DIR@@|$REGISTRY_DATA_DIR|g" \
  "$SCRIPT_DIR/templates/registry.yaml.in" >"$registry_manifest"
kctl apply -f "$registry_manifest"
kctl -n "$REGISTRY_NAMESPACE" rollout restart deployment/coco-registry
kctl -n "$REGISTRY_NAMESPACE" rollout status deployment/coco-registry --timeout=10m

log "Trusting the private registry from containerd"
as_root install -d -m 0755 "/etc/containerd/certs.d/$REGISTRY_HOST"
as_root install -m 0644 "$tls_dir/ca.crt" "/etc/containerd/certs.d/$REGISTRY_HOST/ca.crt"
printf 'server = "https://%s"\n\n[host."https://%s"]\n  capabilities = ["pull", "resolve", "push"]\n  ca = "/etc/containerd/certs.d/%s/ca.crt"\n' \
  "$REGISTRY_HOST" "$REGISTRY_HOST" "$REGISTRY_HOST" | as_root tee "/etc/containerd/certs.d/$REGISTRY_HOST/hosts.toml" >/dev/null
as_root systemctl restart containerd

log "Deploying Trustee KBS and its image-verification resources"
policy="$(jq -cn --arg repo "$TARGET_REPOSITORY" --arg key "$(<"$security_dir/cosign.pub")" '{default:[{type:"reject"}],transports:{docker:{($repo):[{type:"sigstoreSigned",keyData:$key,signedIdentity:{type:"matchRepository"}}]}}}')"
kctl create namespace "$SECURITY_NAMESPACE" --dry-run=client -o yaml | kctl apply -f -
kctl -n "$SECURITY_NAMESPACE" create secret generic coco-image-verification-resources \
  --from-literal=security-policy="$policy" \
  --from-file=cosign-public-key="$security_dir/cosign.pub" \
  --from-file=registry-ca="$tls_dir/ca.crt" \
  --dry-run=client -o yaml | kctl apply -f -
trustee_manifest="$tmp_dir/trustee.yaml"
sed -e "s|@@NAMESPACE@@|$SECURITY_NAMESPACE|g" \
  -e "s|@@SERVICE@@|$KBS_SERVICE|g" \
  -e "s|@@TRUSTEE_IMAGE@@|$TRUSTEE_IMAGE|g" \
  "$SCRIPT_DIR/templates/trustee-kbs.yaml.in" >"$trustee_manifest"
kctl apply -f "$trustee_manifest"
kctl -n "$SECURITY_NAMESPACE" rollout restart "deployment/$KBS_SERVICE"
kctl -n "$SECURITY_NAMESPACE" rollout status "deployment/$KBS_SERVICE" --timeout=10m

log "Staging NVFlare hardware-attestation service configuration"
if [[ "$TEE_PLATFORM" == tdx && -n "$NVFLARE_TDX_ATTESTATION_CONFIG_FILE" ]]; then
  [[ -r "$NVFLARE_TDX_ATTESTATION_CONFIG_FILE" ]] ||
    die "Cannot read NVFLARE_TDX_ATTESTATION_CONFIG_FILE: $NVFLARE_TDX_ATTESTATION_CONFIG_FILE"
  jq -e '
    type == "object" and
    (.trustauthority_url | type == "string" and length > 0) and
    (.trustauthority_api_url | type == "string" and test("^https://")) and
    (.trustauthority_api_key | type == "string" and length > 0)
  ' "$NVFLARE_TDX_ATTESTATION_CONFIG_FILE" >/dev/null ||
    die "TDX config.json must contain trustauthority_url, an HTTPS trustauthority_api_url, and trustauthority_api_key"
  ensure_coco_managed_namespace "$NVFLARE_NAMESPACE"
  kctl -n "$NVFLARE_NAMESPACE" create secret generic nvflare-tdx-attestation \
    --from-file="config.json=$NVFLARE_TDX_ATTESTATION_CONFIG_FILE" \
    --dry-run=client -o yaml | kctl apply -f -
elif [[ "$TEE_PLATFORM" == tdx ]]; then
  log "No TDX Trust Authority config staged; the generic workflow is unaffected, but NVFlare Stage 50 will fail closed until Stage 30 is rerun with NVFLARE_TDX_ATTESTATION_CONFIG_FILE"
fi

if [[ -n "$NVFLARE_GPU_ATTESTATION_SERVICE_KEY_FILE" ]]; then
  [[ -r "$NVFLARE_GPU_ATTESTATION_SERVICE_KEY_FILE" ]] ||
    die "Cannot read NVFLARE_GPU_ATTESTATION_SERVICE_KEY_FILE: $NVFLARE_GPU_ATTESTATION_SERVICE_KEY_FILE"
  gpu_service_key="$(<"$NVFLARE_GPU_ATTESTATION_SERVICE_KEY_FILE")"
  [[ -n "$gpu_service_key" && "$gpu_service_key" != *$'\n'* && ${#gpu_service_key} -le 4096 ]] ||
    die "The NVIDIA attestation service-key file must contain one non-empty line of at most 4096 characters"
  install -m 0600 /dev/null "$tmp_dir/nvidia-attestation-service-key"
  printf '%s' "$gpu_service_key" >"$tmp_dir/nvidia-attestation-service-key"
  ensure_coco_managed_namespace "$NVFLARE_NAMESPACE"
  kctl -n "$NVFLARE_NAMESPACE" create secret generic nvflare-gpu-attestation \
    --from-file="service-key=$tmp_dir/nvidia-attestation-service-key" \
    --dry-run=client -o yaml | kctl apply -f -
else
  log "No NVIDIA service-key file configured; local NVAT verification needs none, while authenticated remote NRAS verification requires one"
fi

curl --fail --silent --show-error --cacert "$tls_dir/ca.crt" "https://${REGISTRY_HOST}/v2/" >/dev/null
log "Registry and Trustee KBS are ready"
echo "Signing key: $security_dir/cosign.key"
echo "Registry CA: $tls_dir/ca.crt"
