#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
load_config
require_root_or_sudo
need jq
need skopeo
need openssl
need kubectl

server_dockerfile="$REPO_ROOT/docker/Dockerfile.coco"
client_dockerfile="$REPO_ROOT/docker/Dockerfile.coco"
[[ -s "$server_dockerfile" ]] || die "NVFlare server Dockerfile is missing: $server_dockerfile"
[[ -s "$client_dockerfile" ]] || die "NVFlare client Dockerfile is missing: $client_dockerfile"

for value_name in NVFLARE_SERVER_IMAGE_NAME NVFLARE_CLIENT_IMAGE_NAME; do
  value="${!value_name:-}"
  [[ "$value" =~ ^[a-z0-9]+([._/-][a-z0-9]+)*$ ]] ||
    die "$value_name must be a lowercase OCI repository path"
done
for value_name in NVFLARE_SERVER_IMAGE_TAG NVFLARE_CLIENT_IMAGE_TAG; do
  value="${!value_name:-}"
  [[ "$value" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]] ||
    die "$value_name is not a valid OCI tag"
done
for value_name in SNPGUEST_SHA256 TRUSTAUTHORITY_CLI_SHA256 NVAT_REPO_SHA256; do
  [[ "${!value_name:-}" =~ ^[0-9a-f]{64}$ ]] || die "$value_name must be a lowercase SHA-256 digest"
done
[[ "$SNPGUEST_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] ||
  die "SNPGUEST_VERSION must use numeric X.Y.Z form"
[[ "$TRUSTAUTHORITY_CLI_VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] ||
  die "TRUSTAUTHORITY_CLI_VERSION must use vX.Y.Z form"
[[ "$NVAT_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "NVAT_VERSION must use numeric X.Y.Z form"
[[ "$TENSORBOARD_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] ||
  die "TENSORBOARD_VERSION must use numeric X.Y.Z form"
[[ "$NVAT_REPO_URL" == https://developer.download.nvidia.com/compute/nvat/${NVAT_VERSION}/local_installers/nvat-local-repo-ubuntu2404-*.deb ]] ||
  die "NVAT_REPO_URL must be the pinned NVIDIA NVAT repository package for NVAT_VERSION"
for value_name in NVFLARE_COCO_PYTHON_BUILD_BASE NVFLARE_COCO_PYTHON_RUNTIME_BASE NVAT_BUILD_BASE; do
  [[ "${!value_name:-}" =~ ^[^[:space:]]+@sha256:[0-9a-f]{64}$ ]] ||
    die "$value_name must be an immutable digest-pinned image"
done
[[ "$NVFLARE_COCO_PYTHON_MINOR" =~ ^[0-9]+\.[0-9]+$ ]] ||
  die "NVFLARE_COCO_PYTHON_MINOR must use numeric X.Y form"

# Stage 30 is an explicit prerequisite. Do not hide infrastructure deployment
# inside this image-build stage: operators must run the numbered stages in
# order and can inspect the registry, signing material, and Trustee separately.
[[ -r "$STATE_DIR/security/cosign.key" && -r "$STATE_DIR/security/cosign.pub" ]] ||
  die "Run 30-deploy-security-services.sh first: Cosign signing material is missing"
[[ -r "$STATE_DIR/registry-tls/ca.crt" ]] ||
  die "Run 30-deploy-security-services.sh first: registry CA is missing"
kctl -n "$REGISTRY_NAMESPACE" get deployment coco-registry >/dev/null 2>&1 ||
  die "Run 30-deploy-security-services.sh first: registry deployment is missing"
kctl -n "$SECURITY_NAMESPACE" get deployment "$KBS_SERVICE" >/dev/null 2>&1 ||
  die "Run 30-deploy-security-services.sh first: Trustee KBS deployment is missing"

if ! command -v podman >/dev/null 2>&1; then
  log "Installing Podman for daemonless NVFlare image builds"
  as_root apt-get update
  as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y podman
fi
need podman
need cosign

security_dir="$STATE_DIR/security"
tls_dir="$STATE_DIR/registry-tls"
cert_dir="$STATE_DIR/nvflare-build-certs"
podman_root="$STATE_DIR/podman-storage"
podman_runroot="$STATE_DIR/podman-runroot"
mkdir -p "$cert_dir"
as_root install -d -m 0755 "$podman_root" "$podman_runroot"
install -m 0644 "$tls_dir/ca.crt" "$cert_dir/ca.crt"
[[ -r "$security_dir/cosign.key" ]] || die "Cosign private key is missing after security-service deployment"

server_repository="${REGISTRY_HOST}/${NVFLARE_SERVER_IMAGE_NAME}"
client_repository="${REGISTRY_HOST}/${NVFLARE_CLIENT_IMAGE_NAME}"
[[ "$server_repository" != "$client_repository" ]] ||
  die "Server and client image repositories must be different"
server_tagged="${server_repository}:${NVFLARE_SERVER_IMAGE_TAG}"
client_tagged="${client_repository}:${NVFLARE_CLIENT_IMAGE_TAG}"

source_revision="unknown"
if command -v git >/dev/null 2>&1 && git -C "$REPO_ROOT" rev-parse --verify HEAD >/dev/null 2>&1; then
  source_revision="$(git -C "$REPO_ROOT" rev-parse HEAD)"
fi

build_and_push() {
  local role=$1 dockerfile=$2 destination=$3 local_image
  local_image="localhost/nvflare-coco-${role}:${destination##*:}"
  log "Building the NVFlare ${role} image from $dockerfile"
  # Use the host's already-qualified egress path. Rootful Podman's private
  # build network cannot reach package mirrors on some TDX host firewalls.
  as_root podman --root "$podman_root" --runroot "$podman_runroot" build \
    --network=host --pull=always \
    --build-arg "PYTHON_BUILD_BASE=$NVFLARE_COCO_PYTHON_BUILD_BASE" \
    --build-arg "PYTHON_RUNTIME_BASE=$NVFLARE_COCO_PYTHON_RUNTIME_BASE" \
    --build-arg "PYTHON_MINOR=$NVFLARE_COCO_PYTHON_MINOR" \
    --build-arg "SNPGUEST_VERSION=$SNPGUEST_VERSION" \
    --build-arg "SNPGUEST_SHA256=$SNPGUEST_SHA256" \
    --build-arg "TRUSTAUTHORITY_CLI_VERSION=$TRUSTAUTHORITY_CLI_VERSION" \
    --build-arg "TRUSTAUTHORITY_CLI_SHA256=$TRUSTAUTHORITY_CLI_SHA256" \
    --build-arg "NVAT_VERSION=$NVAT_VERSION" \
    --build-arg "NVAT_REPO_SHA256=$NVAT_REPO_SHA256" \
    --build-arg "NVAT_REPO_URL=$NVAT_REPO_URL" \
    --build-arg "NVAT_BUILD_BASE=$NVAT_BUILD_BASE" \
    --build-arg "TENSORBOARD_VERSION=$TENSORBOARD_VERSION" \
    --label "org.opencontainers.image.source=NVFlare" \
    --label "org.opencontainers.image.revision=${source_revision}" \
    --label "org.opencontainers.image.role=${role}" \
    --file "$dockerfile" --tag "$local_image" "$REPO_ROOT"

  log "Checking attestation and hello-numpy dependencies in the ${role} runtime image"
  as_root podman --root "$podman_root" --runroot "$podman_runroot" run --rm \
    --entrypoint /opt/attestation/bin/snpguest "$local_image" --version 2>&1 |
    grep -F "snpguest $SNPGUEST_VERSION" >/dev/null ||
    die "The ${role} image does not contain snpguest $SNPGUEST_VERSION"
  as_root podman --root "$podman_root" --runroot "$podman_runroot" run --rm \
    --entrypoint /opt/attestation/bin/trustauthority-cli "$local_image" version 2>&1 |
    grep -F "Version: $TRUSTAUTHORITY_CLI_VERSION" >/dev/null ||
    die "The ${role} image does not contain Intel Trust Authority CLI $TRUSTAUTHORITY_CLI_VERSION"
  as_root podman --root "$podman_root" --runroot "$podman_runroot" run --rm \
    --entrypoint /opt/attestation/bin/nvattest "$local_image" version 2>&1 |
    grep -F "nvattest $NVAT_VERSION" >/dev/null ||
    die "The ${role} image does not contain NVIDIA NVAT $NVAT_VERSION"
  as_root podman --root "$podman_root" --runroot "$podman_runroot" run --rm \
    --entrypoint python "$local_image" -c \
    "import importlib.util, os, tensorboard; from nvflare.app_opt.confidential_computing.coco_hello_numpy_executor import CoCoHelloNumpyExecutor; from nvflare.app_opt.confidential_computing.snp_authorizer import SNPAuthorizer; from nvflare.app_opt.confidential_computing.tdx_authorizer import TDXAuthorizer; from nvflare.app_opt.confidential_computing.gpu_authorizer import GPUAuthorizer; assert tensorboard.__version__ == '$TENSORBOARD_VERSION'; assert importlib.util.find_spec('kubernetes') is None; assert os.path.isfile(CoCoHelloNumpyExecutor.TRAINER_PATH)"
  log "Pushing $destination to the private TLS registry"
  as_root podman --root "$podman_root" --runroot "$podman_runroot" push \
    --cert-dir "$cert_dir" --tls-verify=true "$local_image" "docker://$destination"
}

build_and_push server "$server_dockerfile" "$server_tagged"
build_and_push client "$client_dockerfile" "$client_tagged"

sign_and_verify() {
  local role=$1 tagged=$2 repository=$3 digest signed_ref stderr_file
  digest="$(skopeo inspect --retry-times 5 --cert-dir "$cert_dir" "docker://$tagged" | jq -er .Digest)"
  [[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]] || die "Registry returned an invalid digest for $tagged"
  signed_ref="${repository}@${digest}"
  log "Signing the ${role} image digest $signed_ref"
  if [[ "$COSIGN_TLOG_MODE" == rekor ]]; then
    COSIGN_PASSWORD="${COSIGN_PASSWORD:-}" cosign sign --yes --tlog-upload=true \
      --allow-insecure-registry --key "$security_dir/cosign.key" "$signed_ref"
  else
    COSIGN_PASSWORD="${COSIGN_PASSWORD:-}" cosign sign --yes --tlog-upload=false \
      --allow-insecure-registry --key "$security_dir/cosign.key" "$signed_ref"
  fi

  stderr_file="$STATE_DIR/.nvflare-${role}-cosign-stderr.$$"
  if [[ "$COSIGN_TLOG_MODE" == rekor ]]; then
    if ! cosign verify --allow-insecure-registry \
      --key "$security_dir/cosign.pub" "$signed_ref" \
      >"$STATE_DIR/nvflare-${role}-cosign-verification.json" 2>"$stderr_file"; then
      cat "$stderr_file" >&2
      rm -f "$stderr_file"
      die "Cosign/Rekor verification failed for the NVFlare ${role} image"
    fi
  else
    if ! cosign verify --allow-insecure-registry --insecure-ignore-tlog \
      --key "$security_dir/cosign.pub" "$signed_ref" \
      >"$STATE_DIR/nvflare-${role}-cosign-verification.json" 2>"$stderr_file"; then
      cat "$stderr_file" >&2
      rm -f "$stderr_file"
      die "Cosign verification failed for the NVFlare ${role} image"
    fi
    sed '/^WARNING: Skipping tlog verification is an insecure practice/d' "$stderr_file" >&2
  fi
  rm -f "$stderr_file"
  SIGNED_REF="$signed_ref"
}

sign_and_verify server "$server_tagged" "$server_repository"
server_image="$SIGNED_REF"
sign_and_verify client "$client_tagged" "$client_repository"
client_image="$SIGNED_REF"

policy_file="$STATE_DIR/nvflare-image-security-policy.json"
jq -cn \
  --arg server "$server_repository" \
  --arg client "$client_repository" \
  --arg key "$(<"$security_dir/cosign.pub")" \
  '{default:[{type:"reject"}],transports:{docker:{
    ($server):[{type:"sigstoreSigned",keyData:$key,signedIdentity:{type:"matchRepository"}}],
    ($client):[{type:"sigstoreSigned",keyData:$key,signedIdentity:{type:"matchRepository"}}]
  }}}' >"$policy_file"

# Keep Trustee's public verification resources aligned with the measured
# init-data policy used by the NVFlare deployments.
kctl -n "$SECURITY_NAMESPACE" create secret generic coco-image-verification-resources \
  --from-file=security-policy="$policy_file" \
  --from-file=cosign-public-key="$security_dir/cosign.pub" \
  --from-file=registry-ca="$tls_dir/ca.crt" \
  --dry-run=client -o yaml | kctl apply -f -
kctl -n "$SECURITY_NAMESPACE" rollout restart "deployment/$KBS_SERVICE"
kctl -n "$SECURITY_NAMESPACE" rollout status "deployment/$KBS_SERVICE" --timeout=10m

state_tmp="$STATE_DIR/.nvflare-images.env.$$"
{
  printf 'NVFLARE_SERVER_IMAGE=%q\n' "$server_image"
  printf 'NVFLARE_CLIENT_IMAGE=%q\n' "$client_image"
  printf 'NVFLARE_SERVER_REPOSITORY=%q\n' "$server_repository"
  printf 'NVFLARE_CLIENT_REPOSITORY=%q\n' "$client_repository"
  printf 'NVFLARE_SOURCE_REVISION=%q\n' "$source_revision"
  printf 'NVFLARE_IMAGE_PYTHON_BUILD_BASE=%q\n' "$NVFLARE_COCO_PYTHON_BUILD_BASE"
  printf 'NVFLARE_IMAGE_PYTHON_RUNTIME_BASE=%q\n' "$NVFLARE_COCO_PYTHON_RUNTIME_BASE"
  printf 'NVFLARE_IMAGE_PYTHON_MINOR=%q\n' "$NVFLARE_COCO_PYTHON_MINOR"
  printf 'NVFLARE_IMAGE_NVAT_BUILD_BASE=%q\n' "$NVAT_BUILD_BASE"
  printf 'NVFLARE_SNPGUEST_VERSION=%q\n' "$SNPGUEST_VERSION"
  printf 'NVFLARE_TRUSTAUTHORITY_CLI_VERSION=%q\n' "$TRUSTAUTHORITY_CLI_VERSION"
  printf 'NVFLARE_NVAT_VERSION=%q\n' "$NVAT_VERSION"
  printf 'NVFLARE_TENSORBOARD_VERSION=%q\n' "$TENSORBOARD_VERSION"
  printf 'COSIGN_TLOG_MODE=%q\n' "$COSIGN_TLOG_MODE"
  printf 'SIGNED_AT=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$state_tmp"
mv "$state_tmp" "$STATE_DIR/nvflare-images.env"

cat >"$STATE_DIR/nvflare-image-signing-report.txt" <<EOF
NVFLARE COCO IMAGE BUILD AND SIGNING REPORT
===========================================
Source checkout: $REPO_ROOT
Source revision: $source_revision
Server Dockerfile: docker/Dockerfile.coco
Server image: $server_image
Client Dockerfile: docker/Dockerfile.coco (client role)
Client image: $client_image
CoCo Python builder: $NVFLARE_COCO_PYTHON_BUILD_BASE
CoCo Python runtime: $NVFLARE_COCO_PYTHON_RUNTIME_BASE
NVAT dependency builder: $NVAT_BUILD_BASE
SNP guest tool: snpguest $SNPGUEST_VERSION (runtime check PASS)
TDX tool: Intel Trust Authority CLI $TRUSTAUTHORITY_CLI_VERSION (runtime check PASS)
GPU tool: NVIDIA NVAT $NVAT_VERSION (runtime CLI check PASS)
hello-numpy signed trainer: /opt/nvflare/examples/hello-numpy/client.py (runtime file check PASS)
hello-numpy tracking dependency: TensorBoard $TENSORBOARD_VERSION (runtime import PASS)
Kubernetes Python package: ABSENT (ProcessJobLauncher workflow)
Signature policy: $policy_file
Transparency-log mode: $COSIGN_TLOG_MODE
Server Cosign verification: PASS
Client Cosign verification: PASS
EOF

log "Signed NVFlare images are ready"
echo "Server: $server_image"
echo "Client: $client_image"
