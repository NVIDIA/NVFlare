#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
load_config
require_root_or_sudo
need kubectl
need jq
need gzip
need base64
need sha256sum
need cosign

work_dir="$STATE_DIR/nvflare-deployment"
current_deployment="$work_dir/current.env"
mkdir -p "$work_dir"
# Any Stage-50 attempt invalidates the old active pointer before preflight.
# It is recreated only after the new deployment passes every check.
rm -f "$current_deployment"

images_env="$STATE_DIR/nvflare-images.env"
policy_file="$STATE_DIR/nvflare-image-security-policy.json"
[[ -r "$images_env" ]] || die "Run 40-nvflare-build-sign-images.sh first: $images_env is missing"
[[ -r "$policy_file" ]] || die "NVFlare image policy is missing: $policy_file"
# shellcheck disable=SC1090
source "$images_env"

for value_name in NVFLARE_SERVER_IMAGE NVFLARE_CLIENT_IMAGE; do
  [[ "${!value_name:-}" =~ ^[^[:space:]]+@sha256:[0-9a-f]{64}$ ]] ||
    die "$value_name must be a digest-pinned image"
done
[[ "${NVFLARE_SNPGUEST_VERSION:-}" == "$SNPGUEST_VERSION" ]] ||
  die "Signed images do not record snpguest $SNPGUEST_VERSION; rerun 40-nvflare-build-sign-images.sh"
[[ "${NVFLARE_TRUSTAUTHORITY_CLI_VERSION:-}" == "$TRUSTAUTHORITY_CLI_VERSION" ]] ||
  die "Signed images do not record Trust Authority CLI $TRUSTAUTHORITY_CLI_VERSION; rerun Stage 40"
[[ "${NVFLARE_NVAT_VERSION:-}" == "$NVAT_VERSION" ]] ||
  die "Signed images do not record NVIDIA NVAT $NVAT_VERSION; rerun Stage 40"
[[ "${NVFLARE_TENSORBOARD_VERSION:-}" == "$TENSORBOARD_VERSION" ]] ||
  die "Signed images do not record TensorBoard $TENSORBOARD_VERSION; rerun Stage 40"
[[ "${NVFLARE_IMAGE_PYTHON_BUILD_BASE:-}" == "$NVFLARE_COCO_PYTHON_BUILD_BASE" &&
  "${NVFLARE_IMAGE_PYTHON_RUNTIME_BASE:-}" == "$NVFLARE_COCO_PYTHON_RUNTIME_BASE" &&
  "${NVFLARE_IMAGE_PYTHON_MINOR:-}" == "$NVFLARE_COCO_PYTHON_MINOR" &&
  "${NVFLARE_IMAGE_NVAT_BUILD_BASE:-}" == "$NVAT_BUILD_BASE" ]] ||
  die "Signed images do not match the configured CoCo Python/NVAT base images; rerun Stage 40"
for value_name in NVFLARE_NAMESPACE NVFLARE_SERVER_NAME NVFLARE_CLIENT_NAME; do
  value="${!value_name:-}"
  [[ "$value" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && ${#value} -le 63 ]] ||
    die "$value_name must be a lowercase RFC 1123 name"
done
[[ "${NVFLARE_ADMIN_NAME:-}" =~ ^[A-Za-z0-9@_.-]+$ ]] ||
  die "NVFLARE_ADMIN_NAME contains unsupported characters"
[[ "${NVFLARE_PROJECT_NAME:-}" =~ ^[A-Za-z0-9_.-]+$ ]] ||
  die "NVFLARE_PROJECT_NAME contains unsupported characters"
[[ "${NVFLARE_ORG:-}" =~ ^[A-Za-z0-9_.-]+$ ]] ||
  die "NVFLARE_ORG contains unsupported characters"
[[ "${NVFLARE_CLIENT_GPU_COUNT:-}" =~ ^[1-9][0-9]*$ ]] ||
  die "NVFLARE_CLIENT_GPU_COUNT must be a positive integer"
[[ "${NVFLARE_FS_GROUP:-}" =~ ^[1-9][0-9]*$ && "$NVFLARE_FS_GROUP" -le 2147483647 ]] ||
  die "NVFLARE_FS_GROUP must be a positive 32-bit integer"
for value_name in NVFLARE_SERVER_RUNTIME_CLASS NVFLARE_CLIENT_RUNTIME_CLASS; do
  value="${!value_name:-}"
  [[ "$value" =~ ^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$ && ${#value} -le 253 ]] ||
    die "$value_name must be a lowercase DNS subdomain"
  kctl get runtimeclass "$value" >/dev/null 2>&1 ||
    die "$value_name does not exist in the cluster: $value"
done
[[ "${GPU_RESOURCE:-}" =~ ^[a-z0-9]([-a-z0-9.]*[a-z0-9])?/[A-Za-z0-9]([-A-Za-z0-9_.]*[A-Za-z0-9])?$ ]] ||
  die "GPU_RESOURCE must be a Kubernetes extended-resource name"
for value_name in NVFLARE_SERVER_MEMORY NVFLARE_CLIENT_MEMORY NVFLARE_WORKSPACE_SIZE; do
  [[ "${!value_name:-}" =~ ^[1-9][0-9]*(Ki|Mi|Gi|Ti)$ ]] ||
    die "$value_name must be a positive binary Kubernetes quantity"
done
[[ "${AMD_KDS_PRODUCT:-}" =~ ^[A-Za-z0-9_-]+$ ]] || die "AMD_KDS_PRODUCT contains unsupported characters"
ensure_coco_managed_namespace "$NVFLARE_NAMESPACE"

case "$TEE_PLATFORM" in
  snp)
    cpu_attestation_device="sev-guest"
    cpu_mechanism="amd_sev_snp"
    cpu_authorizer_id="snp_authorizer"
    cpu_authorizer_path="nvflare.app_opt.confidential_computing.snp_authorizer.SNPAuthorizer"
    ;;
  tdx)
    cpu_attestation_device="tdx_guest"
    cpu_mechanism="intel_tdx"
    cpu_authorizer_id="tdx_authorizer"
    cpu_authorizer_path="nvflare.app_opt.confidential_computing.tdx_authorizer.TDXAuthorizer"
    kctl -n "$NVFLARE_NAMESPACE" get secret nvflare-tdx-attestation \
      -o jsonpath='{.data.config\.json}' | grep -q . ||
      die "TDX attestation Secret is missing; rerun 30-deploy-security-services.sh with NVFLARE_TDX_ATTESTATION_CONFIG_FILE"
    ;;
  *) die "Unsupported TEE_PLATFORM: $TEE_PLATFORM" ;;
esac

security_dir="$STATE_DIR/security"
tls_dir="$STATE_DIR/registry-tls"
[[ -r "$security_dir/cosign.pub" ]] || die "Cosign public key is missing"
[[ -r "$tls_dir/ca.crt" ]] || die "Registry CA is missing"

verify_signed_image() {
  local role=$1 image=$2 stderr_file
  stderr_file="$STATE_DIR/.nvflare-deploy-${role}-cosign-stderr.$$"
  if [[ "${COSIGN_TLOG_MODE:-disabled}" == rekor ]]; then
    if ! cosign verify --allow-insecure-registry --key "$security_dir/cosign.pub" \
      "$image" >/dev/null 2>"$stderr_file"; then
      cat "$stderr_file" >&2
      rm -f "$stderr_file"
      die "Cosign/Rekor verification failed for $role image $image"
    fi
  else
    if ! cosign verify --allow-insecure-registry --insecure-ignore-tlog \
      --key "$security_dir/cosign.pub" "$image" >/dev/null 2>"$stderr_file"; then
      cat "$stderr_file" >&2
      rm -f "$stderr_file"
      die "Cosign verification failed for $role image $image"
    fi
    sed '/^WARNING: Skipping tlog verification is an insecure practice/d' "$stderr_file" >&2
  fi
  rm -f "$stderr_file"
}

log "Re-verifying both digest-pinned NVFlare image signatures"
verify_signed_image server "$NVFLARE_SERVER_IMAGE"
verify_signed_image client "$NVFLARE_CLIENT_IMAGE"

jq -e --arg server "$NVFLARE_SERVER_REPOSITORY" --arg client "$NVFLARE_CLIENT_REPOSITORY" \
  --arg key "$(<"$security_dir/cosign.pub")" '
  . as $policy |
  ($policy.default == [{type:"reject"}]) and
  all([$server, $client][];
    . as $repository |
    ($policy.transports.docker[$repository][0].type == "sigstoreSigned") and
    ($policy.transports.docker[$repository][0].keyData == $key) and
    ($policy.transports.docker[$repository][0].signedIdentity.type == "matchRepository"))
' "$policy_file" >/dev/null || die "NVFlare image policy is not deny-by-default for both repositories"

nvflare_cli="${NVFLARE_CLI:-}"
if [[ -z "$nvflare_cli" ]] && command -v nvflare >/dev/null 2>&1; then
  nvflare_cli="$(command -v nvflare)"
fi
if [[ -z "$nvflare_cli" && -x "$REPO_ROOT/.venv/bin/nvflare" ]]; then
  nvflare_cli="$REPO_ROOT/.venv/bin/nvflare"
fi
[[ -x "$nvflare_cli" ]] ||
  die "NVFlare CLI not found; install this checkout in a virtual environment or set NVFLARE_CLI"

project_file="$work_dir/project.yml"
manifest="$work_dir/nvflare-coco.yaml"
initdata_file="$work_dir/initdata.toml"
server_cc_config="$work_dir/cc-server.yml"
client_cc_config="$work_dir/cc-client.yml"

write_cc_config() {
  local role=$1 destination=$2
  cat >"$destination" <<EOF
compute_env: onprem_cvm
cc_cpu_mechanism: $cpu_mechanism
role: $role
cc_issuers:
  - id: $cpu_authorizer_id
    path: $cpu_authorizer_path
    token_expiration: 100
    args:
EOF
  if [[ "$TEE_PLATFORM" == snp ]]; then
    cat >>"$destination" <<EOF
      snpguest_binary: /opt/attestation/bin/snpguest
      amd_certs_dir: /var/tmp/nvflare/workspace/attestation/snp-certs
      cpu_model: ${AMD_KDS_PRODUCT,,}
EOF
  else
    cat >>"$destination" <<EOF
      tdx_cli_command: /opt/attestation/bin/trustauthority-cli
      config_dir: /etc/nvflare/tdx-attestation
      use_sudo: false
EOF
  fi
  if [[ "$role" == client ]]; then
    cat >>"$destination" <<EOF
  - id: gpu_authorizer
    path: nvflare.app_opt.confidential_computing.gpu_authorizer.GPUAuthorizer
    token_expiration: 100
    args:
      nvat_command: /opt/attestation/bin/nvattest
      verifier: $NVFLARE_GPU_VERIFIER
EOF
    if [[ "$NVFLARE_GPU_VERIFIER" == remote ]]; then
      printf '      nras_url: %s\n' "$NVFLARE_GPU_NRAS_URL" >>"$destination"
    fi
  fi
  cat >>"$destination" <<'EOF'
cc_attestation:
  check_frequency: 120
  # NVAT permits a 180-second subprocess timeout. Keep the peer request
  # deadline above it so a slow appraisal cannot trigger federation shutdown.
  get_token_request_timeout: 200
class_allow_list:
  - nvflare.app_opt.confidential_computing.cc_manager.CCManager
  - nvflare.app_opt.confidential_computing.coco_hello_numpy_executor.CoCoHelloNumpyExecutor
  - nvflare.app_opt.confidential_computing.snp_authorizer.SNPAuthorizer
  - nvflare.app_opt.confidential_computing.tdx_authorizer.TDXAuthorizer
  - nvflare.app_opt.confidential_computing.gpu_authorizer.GPUAuthorizer
EOF
}

write_cc_config server "$server_cc_config"
write_cc_config client "$client_cc_config"
# NVFlare increments prod_00 to prod_01, prod_02, ... when a workspace already
# exists.  Provision into a fresh directory so prod_00 always belongs to this
# run and an interrupted rerun cannot deploy stale participant credentials.
workspace_dir="$(mktemp -d "$work_dir/workspace.XXXXXX")"

sed \
  -e "s|@@PROJECT_NAME@@|$NVFLARE_PROJECT_NAME|g" \
  -e "s|@@SERVER_NAME@@|$NVFLARE_SERVER_NAME|g" \
  -e "s|@@CLIENT_NAME@@|$NVFLARE_CLIENT_NAME|g" \
  -e "s|@@ADMIN_NAME@@|$NVFLARE_ADMIN_NAME|g" \
  -e "s|@@ORG@@|$NVFLARE_ORG|g" \
  -e "s|@@NAMESPACE@@|$NVFLARE_NAMESPACE|g" \
  -e "s|@@CLIENT_GPU_COUNT@@|$NVFLARE_CLIENT_GPU_COUNT|g" \
  -e "s|@@SERVER_CC_CONFIG@@|$server_cc_config|g" \
  -e "s|@@CLIENT_CC_CONFIG@@|$client_cc_config|g" \
  "$SCRIPT_DIR/templates/nvflare-project.yml.in" >"$project_file"

log "Provisioning NVFlare identities with CPU and client-GPU hardware authorizers"
provision_log="$work_dir/provision.log"
if ! "$nvflare_cli" provision -p "$project_file" -w "$workspace_dir" --force 2>&1 | tee "$provision_log"; then
  die "NVFlare provisioning failed; see $provision_log"
fi
if grep -Fq "CC is not enabled for" "$provision_log"; then
  die "CCBuilder rejected a participant configuration; see $provision_log"
fi
prod_dir="$workspace_dir/$NVFLARE_PROJECT_NAME/prod_00"
server_kit="$prod_dir/$NVFLARE_SERVER_NAME"
client_kit="$prod_dir/$NVFLARE_CLIENT_NAME"
admin_kit="$prod_dir/$NVFLARE_ADMIN_NAME"
for kit in "$server_kit" "$client_kit" "$admin_kit"; do
  [[ -d "$kit/startup" ]] || die "Provisioned startup kit is missing: $kit/startup"
done
[[ -d "$server_kit/local" ]] || die "Provisioned server local configuration is missing"
[[ -d "$client_kit/local" ]] || die "Provisioned client local configuration is missing"

for kit in "$server_kit" "$client_kit"; do
  for resource in cc_manager__p_resources.json "${cpu_authorizer_id}__p_resources.json" gpu_authorizer__p_resources.json; do
    [[ -s "$kit/local/$resource" ]] ||
      die "CCBuilder did not generate $resource for ${kit##*/}"
  done
done
jq -e --arg issuer "$cpu_authorizer_id" '
  .components[0].path == "nvflare.app_opt.confidential_computing.cc_manager.CCManager" and
  .components[0].args.get_token_request_timeout == 200 and
  (.components[0].args.cc_issuers_conf | map(.issuer_id) | index($issuer) != null) and
  (.components[0].args.cc_verifier_ids | index($issuer) != null) and
  (.components[0].args.cc_verifier_ids | index("gpu_authorizer") != null)
' "$server_kit/local/cc_manager__p_resources.json" >/dev/null ||
  die "Server CCManager is not configured for CPU issuance and CPU/GPU verification"
jq -e --arg issuer "$cpu_authorizer_id" '
  .components[0].path == "nvflare.app_opt.confidential_computing.cc_manager.CCManager" and
  .components[0].args.get_token_request_timeout == 200 and
  (.components[0].args.cc_issuers_conf | length == 2) and
  (.components[0].args.cc_issuers_conf | map(.issuer_id) | index($issuer) != null) and
  (.components[0].args.cc_issuers_conf | map(.issuer_id) | index("gpu_authorizer") != null) and
  (.components[0].args.cc_verifier_ids | index($issuer) != null) and
  (.components[0].args.cc_verifier_ids | index("gpu_authorizer") != null)
' "$client_kit/local/cc_manager__p_resources.json" >/dev/null ||
  die "Client CCManager is not configured for CPU/GPU issuance and verification"
for kit in "$server_kit" "$client_kit"; do
  jq -e --arg verifier "$NVFLARE_GPU_VERIFIER" '
    .components[0].path == "nvflare.app_opt.confidential_computing.gpu_authorizer.GPUAuthorizer" and
    .components[0].args.verifier == $verifier
  ' "$kit/local/gpu_authorizer__p_resources.json" >/dev/null ||
    die "GPUAuthorizer for ${kit##*/} does not use verifier mode $NVFLARE_GPU_VERIFIER"
done

# The stock provisioned server uses /tmp for job and snapshot storage. Keep
# those files on the released CoCo block-encrypted emptyDir instead.
server_resources="$server_kit/local/resources.json.default"
[[ -r "$server_resources" ]] || die "Provisioned server resources are missing: $server_resources"
server_resources_tmp="${server_resources}.tmp.$$"
jq --arg workspace "/var/tmp/nvflare/workspace" '
  (.snapshot_persistor.args.storage.args.root_dir) = ($workspace + "/snapshot-storage") |
  (.components[] | select(.id == "job_manager") | .args.uri_root) = ($workspace + "/jobs-storage")
' "$server_resources" >"$server_resources_tmp"
mv "$server_resources_tmp" "$server_resources"
jq -e '
  .snapshot_persistor.args.storage.args.root_dir == "/var/tmp/nvflare/workspace/snapshot-storage" and
  any(.components[]; .id == "job_manager" and .args.uri_root == "/var/tmp/nvflare/workspace/jobs-storage")
' "$server_resources" >/dev/null || die "Could not relocate NVFlare server storage to the encrypted workspace"

image_policy="$(<"$policy_file")"
registry_ca="$(<"$tls_dir/ca.crt")"
cat >"$initdata_file" <<EOF
version = "0.1.0"
algorithm = "sha256"

[data]
"cdh.toml" = '''
[kbc]
name = "offline_fs_kbc"
url = ""

[image]
image_security_policy = '${image_policy}'
extra_root_certificates = ["""${registry_ca}"""]

[image.registry_config]
unqualified-search-registries = ["docker.io"]

[[image.registry_config.registry]]
location = "${REGISTRY_HOST}"
insecure = false
'''

"policy.rego" = '''
package agent_policy
default AddARPNeighborsRequest := true
default AddSwapRequest := true
default CloseStdinRequest := true
default CopyFileRequest := true
default CreateContainerRequest := true
default CreateSandboxRequest := true
default DestroySandboxRequest := true
default ExecProcessRequest := true
default GetMetricsRequest := true
default GetOOMEventRequest := true
default GuestDetailsRequest := true
default ListInterfacesRequest := true
default ListRoutesRequest := true
default MemHotplugByProbeRequest := true
default OnlineCPUMemRequest := true
default PauseContainerRequest := true
default PullImageRequest := true
default ReadStreamRequest := true
default RemoveContainerRequest := true
default RemoveStaleVirtiofsShareMountsRequest := true
default ReseedRandomDevRequest := true
default ResumeContainerRequest := true
default SetGuestDateTimeRequest := true
default SetPolicyRequest := false
default SignalProcessRequest := true
default StartContainerRequest := true
default StartTracingRequest := true
default StatsContainerRequest := true
default StopTracingRequest := true
default TtyWinResizeRequest := true
default UpdateContainerRequest := true
default UpdateEphemeralMountsRequest := true
default UpdateInterfaceRequest := true
default UpdateRoutesRequest := true
default WaitProcessRequest := true
default WriteStreamRequest := true
'''
EOF

init_data="$(gzip -c "$initdata_file" | base64 -w 0)"
initdata_sha256="$(sha256sum "$initdata_file" | awk '{print $1}')"

log "Staging provisioned startup kits as Kubernetes Secrets and local configuration as ConfigMaps"
log "Allowing privileged Kata workloads in the dedicated NVFlare namespace"
kctl label namespace "$NVFLARE_NAMESPACE" \
  pod-security.kubernetes.io/enforce=privileged --overwrite
for participant in "$NVFLARE_SERVER_NAME" "$NVFLARE_CLIENT_NAME"; do
  if [[ "$participant" == "$NVFLARE_SERVER_NAME" ]]; then
    kit="$server_kit"
  else
    kit="$client_kit"
  fi
  kctl -n "$NVFLARE_NAMESPACE" create secret generic "${participant}-startup" \
    --from-file="$kit/startup" --dry-run=client -o yaml | kctl apply -f -
  kctl -n "$NVFLARE_NAMESPACE" create configmap "${participant}-local" \
    --from-file="$kit/local" --dry-run=client -o yaml | kctl apply -f -
done

sed \
  -e "s|@@NAMESPACE@@|$NVFLARE_NAMESPACE|g" \
  -e "s|@@SERVER_NAME@@|$NVFLARE_SERVER_NAME|g" \
  -e "s|@@CLIENT_NAME@@|$NVFLARE_CLIENT_NAME|g" \
  -e "s|@@ORG@@|$NVFLARE_ORG|g" \
  -e "s|@@SERVER_RUNTIME_CLASS@@|$NVFLARE_SERVER_RUNTIME_CLASS|g" \
  -e "s|@@CLIENT_RUNTIME_CLASS@@|$NVFLARE_CLIENT_RUNTIME_CLASS|g" \
  -e "s|@@SERVER_IMAGE@@|$NVFLARE_SERVER_IMAGE|g" \
  -e "s|@@CLIENT_IMAGE@@|$NVFLARE_CLIENT_IMAGE|g" \
  -e "s|@@SERVER_MEMORY@@|$NVFLARE_SERVER_MEMORY|g" \
  -e "s|@@CLIENT_MEMORY@@|$NVFLARE_CLIENT_MEMORY|g" \
  -e "s|@@GPU_RESOURCE@@|$GPU_RESOURCE|g" \
  -e "s|@@CLIENT_GPU_COUNT@@|$NVFLARE_CLIENT_GPU_COUNT|g" \
  -e "s|@@WORKSPACE_SIZE@@|$NVFLARE_WORKSPACE_SIZE|g" \
  -e "s|@@FS_GROUP@@|$NVFLARE_FS_GROUP|g" \
  -e "s|@@CPU_ATTESTATION_DEVICE@@|$cpu_attestation_device|g" \
  -e "s|@@INIT_DATA@@|$init_data|g" \
  "$SCRIPT_DIR/templates/nvflare-coco.yaml.in" >"$manifest"
if grep -Eq '@@[A-Z0-9_]+@@' "$manifest"; then
  die "Unresolved placeholder remains in $manifest"
fi

log "Deploying a CPU-only NVFlare server and GPU NVFlare client as Kata confidential VMs"
kctl apply -f "$manifest"
kctl -n "$NVFLARE_NAMESPACE" rollout restart "deployment/$NVFLARE_SERVER_NAME" "deployment/$NVFLARE_CLIENT_NAME"
kctl -n "$NVFLARE_NAMESPACE" rollout status "deployment/$NVFLARE_SERVER_NAME" --timeout=20m
kctl -n "$NVFLARE_NAMESPACE" rollout status "deployment/$NVFLARE_CLIENT_NAME" --timeout=20m

verify_deployment() {
  local participant=$1 expected_image=$2 expected_runtime=$3
  local pod runtime annotation pod_image image_id expected_digest privileged
  pod="$(kctl -n "$NVFLARE_NAMESPACE" get pod -l "app=$participant" \
    -o jsonpath='{.items[0].metadata.name}')"
  [[ -n "$pod" ]] || die "No pod found for $participant"
  kctl -n "$NVFLARE_NAMESPACE" wait --for=condition=Ready "pod/$pod" --timeout=20m
  runtime="$(kctl -n "$NVFLARE_NAMESPACE" get pod "$pod" -o jsonpath='{.spec.runtimeClassName}')"
  [[ "$runtime" == "$expected_runtime" ]] ||
    die "$participant used $runtime instead of $expected_runtime"
  annotation="$(kctl -n "$NVFLARE_NAMESPACE" get pod "$pod" \
    -o 'jsonpath={.metadata.annotations.io\.katacontainers\.config\.hypervisor\.cc_init_data}')"
  [[ "$annotation" == "$init_data" ]] || die "$participant measured init-data annotation changed"
  pod_image="$(kctl -n "$NVFLARE_NAMESPACE" get pod "$pod" -o jsonpath='{.spec.containers[0].image}')"
  [[ "$pod_image" == "$expected_image" ]] || die "$participant pod image is not the signed digest"
  image_id="$(kctl -n "$NVFLARE_NAMESPACE" get pod "$pod" -o jsonpath='{.status.containerStatuses[0].imageID}')"
  expected_digest="${expected_image##*@}"
  [[ "$image_id" == *"$expected_digest" ]] ||
    die "$participant runtime image ID does not match the signed digest: $image_id"
  # This is a Pod-spec pre-flight check, not proof of the runtime's effective
  # security context. Opening the guest TEE device below is authoritative.
  privileged="$(kctl -n "$NVFLARE_NAMESPACE" get pod "$pod" -o jsonpath='{.spec.containers[0].securityContext.privileged}')"
  [[ "$privileged" == true ]] ||
    die "$participant is not privileged inside its Kata VM and cannot access the guest TEE interface"
  kctl -n "$NVFLARE_NAMESPACE" exec "$pod" -- python -c '
import os, stat, subprocess, sys
import tensorboard

name, snp_version, tdx_version, nvat_version, tensorboard_version = sys.argv[1:]
path = f"/dev/{name}"
info = os.stat(path)
assert stat.S_ISCHR(info.st_mode), f"not a character device: {path}"
fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
os.close(fd)
assert snp_version in subprocess.check_output(["/opt/attestation/bin/snpguest", "--version"], text=True)
assert tdx_version in subprocess.check_output(["/opt/attestation/bin/trustauthority-cli", "version"], text=True)
nvat = subprocess.check_output(
    ["/opt/attestation/bin/nvattest", "version"], stderr=subprocess.STDOUT, text=True
)
assert f"nvattest {nvat_version}" in nvat
assert tensorboard.__version__ == tensorboard_version
' "$cpu_attestation_device" "$SNPGUEST_VERSION" "$TRUSTAUTHORITY_CLI_VERSION" "$NVAT_VERSION" \
    "$TENSORBOARD_VERSION" ||
    die "$participant failed its in-guest TEE-device or attestation-tool checks"
  VERIFIED_POD="$pod"
  VERIFIED_IMAGE_ID="$image_id"
}

verify_deployment "$NVFLARE_SERVER_NAME" "$NVFLARE_SERVER_IMAGE" "$NVFLARE_SERVER_RUNTIME_CLASS"
server_pod="$VERIFIED_POD"
server_image_id="$VERIFIED_IMAGE_ID"
verify_deployment "$NVFLARE_CLIENT_NAME" "$NVFLARE_CLIENT_IMAGE" "$NVFLARE_CLIENT_RUNTIME_CLASS"
client_pod="$VERIFIED_POD"
client_image_id="$VERIFIED_IMAGE_ID"

server_gpu_request="$(kctl -n "$NVFLARE_NAMESPACE" get pod "$server_pod" -o json |
  jq -r --arg resource "$GPU_RESOURCE" '.spec.containers[0].resources.requests[$resource] // "0"')"
server_gpu_limit="$(kctl -n "$NVFLARE_NAMESPACE" get pod "$server_pod" -o json |
  jq -r --arg resource "$GPU_RESOURCE" '.spec.containers[0].resources.limits[$resource] // "0"')"
client_gpu_request="$(kctl -n "$NVFLARE_NAMESPACE" get pod "$client_pod" -o json |
  jq -r --arg resource "$GPU_RESOURCE" '.spec.containers[0].resources.requests[$resource] // "0"')"
client_gpu_limit="$(kctl -n "$NVFLARE_NAMESPACE" get pod "$client_pod" -o json |
  jq -r --arg resource "$GPU_RESOURCE" '.spec.containers[0].resources.limits[$resource] // "0"')"
[[ "$server_gpu_request" == 0 ]] ||
  die "NVFlare server unexpectedly requests $server_gpu_request $GPU_RESOURCE"
[[ "$server_gpu_limit" == 0 ]] ||
  die "NVFlare server unexpectedly limits $server_gpu_limit $GPU_RESOURCE"
[[ "$client_gpu_request" == "$NVFLARE_CLIENT_GPU_COUNT" ]] ||
  die "NVFlare client requests $client_gpu_request $GPU_RESOURCE instead of $NVFLARE_CLIENT_GPU_COUNT"
[[ "$client_gpu_limit" == "$NVFLARE_CLIENT_GPU_COUNT" ]] ||
  die "NVFlare client limit is $client_gpu_limit $GPU_RESOURCE instead of $NVFLARE_CLIENT_GPU_COUNT"

cc_peer_validation_passed() {
  local pod=$1
  # "Validated CC info for" is logged by CCManager in
  # nvflare/app_opt/confidential_computing/cc_manager.py. If that string changes,
  # update this check and retest Stage 50.
  kctl -n "$NVFLARE_NAMESPACE" logs "$pod" 2>/dev/null | grep -Fq "Validated CC info for"
}
log "Waiting for the NVFlare CPU/GPU authorization handshake"
wait_for "server validation of the client's CPU/GPU evidence" 600 cc_peer_validation_passed "$server_pod"
wait_for "client validation of the server's CPU token" 600 cc_peer_validation_passed "$client_pod"

report="$STATE_DIR/nvflare-deployment-integrity-report.txt"
cat >"$report" <<EOF
NVFLARE ON COCO DEPLOYMENT AND IMAGE-INTEGRITY REPORT
====================================================
Namespace: $NVFLARE_NAMESPACE
Server pod: $server_pod
Server runtime class: $NVFLARE_SERVER_RUNTIME_CLASS
Server GPU request: none
Server signed image: $NVFLARE_SERVER_IMAGE
Server runtime image ID: $server_image_id
Client pod: $client_pod
Client runtime class: $NVFLARE_CLIENT_RUNTIME_CLASS
Client GPU request: $NVFLARE_CLIENT_GPU_COUNT $GPU_RESOURCE
Client signed image: $NVFLARE_CLIENT_IMAGE
Client runtime image ID: $client_image_id
Image policy: $policy_file
Image policy default: reject
Measured init-data SHA-256: $initdata_sha256
Host Cosign verification: PASS
Digest-pinned pod specifications: PASS
Measured init-data annotation match: PASS
Kata RuntimeClass matches: PASS
Exclusive client GPU assignment: PASS
Guest image-rs enforcement: PASS (both signed-image pods reached Ready)
SNP tool in both signed images: snpguest $SNPGUEST_VERSION (PASS)
TDX tool in both signed images: Trust Authority CLI $TRUSTAUTHORITY_CLI_VERSION (PASS)
GPU verifier in both signed images: NVIDIA NVAT $NVAT_VERSION (PASS)
GPU verification mode: $NVFLARE_GPU_VERIFIER (PASS)
hello-numpy tracking dependency in both signed images: TensorBoard $TENSORBOARD_VERSION (PASS)
Guest CPU device: /dev/$cpu_attestation_device (character-device open PASS)
NVFlare CCBuilder resources: CPU issuer plus CPU/GPU peer verifiers (PASS)
Kata-VM-only privilege for guest attestation device: PASS
NVFlare peer hardware-authorization handshake: PASS
Server/client deployment health: PASS

VM launch-reference appraisal: NOT PERFORMED BY THIS STAGE
Use remote attestation with independently trusted SNP/TDX reference values and
verify that HOST_DATA/MRCONFIGID binds this init-data hash before releasing
production credentials or data.
EOF

current_deployment_tmp="$work_dir/.current.env.$$"
install -m 0600 /dev/null "$current_deployment_tmp"
{
  printf 'DEPLOYED_NVFLARE_NAMESPACE=%q\n' "$NVFLARE_NAMESPACE"
  printf 'DEPLOYED_NVFLARE_SERVER_NAME=%q\n' "$NVFLARE_SERVER_NAME"
  printf 'DEPLOYED_NVFLARE_CLIENT_NAME=%q\n' "$NVFLARE_CLIENT_NAME"
  printf 'DEPLOYED_NVFLARE_ADMIN_STARTUP_KIT=%q\n' "$admin_kit/startup"
  printf 'DEPLOYED_NVFLARE_SERVER_IMAGE=%q\n' "$NVFLARE_SERVER_IMAGE"
  printf 'DEPLOYED_NVFLARE_CLIENT_IMAGE=%q\n' "$NVFLARE_CLIENT_IMAGE"
  printf 'DEPLOYED_AT=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$current_deployment_tmp"
mv "$current_deployment_tmp" "$current_deployment"

log "NVFlare server and client are running in CoCo"
echo "Namespace: $NVFLARE_NAMESPACE"
echo "Server service: $NVFLARE_SERVER_NAME:8002"
echo "Server runtime: $NVFLARE_SERVER_RUNTIME_CLASS (no GPU)"
echo "Client: $NVFLARE_CLIENT_NAME"
echo "Client runtime: $NVFLARE_CLIENT_RUNTIME_CLASS ($NVFLARE_CLIENT_GPU_COUNT $GPU_RESOURCE)"
echo "Administrator startup kit: $admin_kit/startup"
echo "Active deployment state: $current_deployment"
echo "Integrity report: $report"
