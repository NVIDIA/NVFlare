#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
load_config
need skopeo
need cosign
need jq

source_image="${1:-$SOURCE_IMAGE}"
target_tag="${2:-signed}"
target_image="${TARGET_REPOSITORY}:${target_tag}"
security_dir="$STATE_DIR/security"
tls_dir="$STATE_DIR/registry-tls"
cert_dir="$STATE_DIR/containers-certs.d/$REGISTRY_HOST"
mkdir -p "$cert_dir"
[[ -r "$security_dir/cosign.key" ]] || die "Run 30-deploy-security-services.sh first"
[[ -r "$tls_dir/ca.crt" ]] || die "Registry CA is missing; run stage 30"
install -m 0644 "$tls_dir/ca.crt" "$cert_dir/ca.crt"

log "Copying $source_image to $target_image"
skopeo copy --retry-times 5 --dest-cert-dir "$cert_dir" \
  "docker://$source_image" "docker://$target_image"
digest="$(skopeo inspect --retry-times 5 --cert-dir "$cert_dir" "docker://$target_image" | jq -er .Digest)"
signed_image="${TARGET_REPOSITORY}@${digest}"

log "Signing the digest-pinned image"
if [[ "$COSIGN_TLOG_MODE" == rekor ]]; then
  COSIGN_PASSWORD="${COSIGN_PASSWORD:-}" cosign sign --yes --tlog-upload=true \
    --allow-insecure-registry --key "$security_dir/cosign.key" "$signed_image"
else
  COSIGN_PASSWORD="${COSIGN_PASSWORD:-}" cosign sign --yes --tlog-upload=false \
    --allow-insecure-registry --key "$security_dir/cosign.key" "$signed_image"
fi

log "Verifying the signature before producing deployment state"
cosign_stderr="$STATE_DIR/.cosign-verify-stderr.$$"
if [[ "$COSIGN_TLOG_MODE" == rekor ]]; then
  if ! cosign verify --allow-insecure-registry \
    --key "$security_dir/cosign.pub" "$signed_image" \
    >"$STATE_DIR/cosign-verification.json" 2>"$cosign_stderr"; then
    cat "$cosign_stderr" >&2
    rm -f "$cosign_stderr"
    die "Cosign signature or Rekor transparency-log verification failed"
  fi
else
  echo "NOTICE: COSIGN_TLOG_MODE=disabled; verifying the private-key signature without public transparency-log inclusion."
  if ! cosign verify --allow-insecure-registry --insecure-ignore-tlog \
    --key "$security_dir/cosign.pub" "$signed_image" \
    >"$STATE_DIR/cosign-verification.json" 2>"$cosign_stderr"; then
    cat "$cosign_stderr" >&2
    rm -f "$cosign_stderr"
    die "Cosign private-key signature verification failed"
  fi
  sed '/^WARNING: Skipping tlog verification is an insecure practice/d' "$cosign_stderr" >&2
fi
rm -f "$cosign_stderr"

{
  printf 'SIGNED_IMAGE=%q\n' "$signed_image"
  printf 'POLICY_REPOSITORY=%q\n' "$TARGET_REPOSITORY"
  printf 'REGISTRY_HOST=%q\n' "$REGISTRY_HOST"
  printf 'SOURCE_IMAGE=%q\n' "$source_image"
  printf 'COSIGN_TLOG_MODE=%q\n' "$COSIGN_TLOG_MODE"
  printf 'SIGNED_AT=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$STATE_DIR/signed-image.env"

cat >"$STATE_DIR/image-signing-report.txt" <<EOF
CONTAINER IMAGE SIGNING REPORT
==============================
Source image: $source_image
Local tagged image: $target_image
Signed digest reference: $signed_image
Signing key: $security_dir/cosign.key
Verification key: $security_dir/cosign.pub
Transparency-log mode: $COSIGN_TLOG_MODE
Cosign verification: PASS
EOF
log "Signed image ready: $signed_image"
