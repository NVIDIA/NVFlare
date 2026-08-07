#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
load_config
require_root_or_sudo
need kubectl
need helm
need containerd
need ctr
need lspci

kctl get nodes >/dev/null
node_name="$(kctl get nodes -o jsonpath='{.items[0].metadata.name}')"
[[ -n "$node_name" ]] || die "No Kubernetes node found"

install_tdx_quote_service() {
  local suite base key_url key_fingerprint tmp_dir observed_fingerprint
  local pccs_config pcs_key configured_key pcs_client
  [[ "$TEE_PLATFORM" == tdx ]] || return 0

  pccs_config="/opt/intel/sgx-dcap-pccs/config/default.json"
  configured_key=""
  if as_root test -s "$pccs_config"; then
    configured_key="$(as_root jq -r '.ApiKey // ""' "$pccs_config")"
  fi
  pcs_key="${INTEL_PCS_API_KEY:-}"
  if [[ -n "${INTEL_PCS_API_KEY_FILE:-}" ]]; then
    [[ -z "$pcs_key" ]] ||
      die "Set only one of INTEL_PCS_API_KEY or INTEL_PCS_API_KEY_FILE"
    [[ -r "$INTEL_PCS_API_KEY_FILE" ]] ||
      die "Intel PCS API-key file is not readable: $INTEL_PCS_API_KEY_FILE"
    IFS= read -r pcs_key <"$INTEL_PCS_API_KEY_FILE" || true
  fi
  [[ -n "$pcs_key" ]] || pcs_key="$configured_key"
  if [[ -n "$pcs_key" && ! "$pcs_key" =~ ^[[:alnum:]]{32}$ ]]; then
    die "Intel PCS subscription key must be empty or exactly 32 alphanumeric characters"
  fi

  if ! systemctl is-active --quiet qgsd 2>/dev/null; then
    # Canonical's TDX project installs these DCAP/QGS packages from its release
    # PPA. Keep the suite override explicit because mixing an older PPA suite
    # onto a newer Ubuntu host is a compatibility decision, not a safe default.
    # shellcheck disable=SC1091
    source /etc/os-release
    suite="${TDX_ATTESTATION_PPA_SUITE:-${VERSION_CODENAME:-}}"
    [[ -n "$suite" ]] || die "Could not determine the Ubuntu suite for the TDX attestation PPA"
    base="https://ppa.launchpadcontent.net/kobuk-team/tdx-attestation-release/ubuntu"
    if ! curl -L --fail --silent --show-error --head "$base/dists/$suite/Release" >/dev/null; then
      die "Canonical's TDX attestation PPA has no '$suite' suite; set TDX_ATTESTATION_PPA_SUITE only after approving a compatible published suite"
    fi

    log "Installing Intel TDX QGS/DCAP packages from Canonical's ${suite} release PPA"
    tmp_dir="$(mktemp -d)"
    key_fingerprint="0C0E6AF955CE463C03FC51574D098D70AFBE5E1F"
    key_url="https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x${key_fingerprint}"
    curl -L --fail --silent --show-error "$key_url" -o "$tmp_dir/ppa.asc"
    observed_fingerprint="$(
      gpg --show-keys --with-colons "$tmp_dir/ppa.asc" |
        awk -F: '$1 == "fpr" {print $10; exit}'
    )"
    [[ "$observed_fingerprint" == "$key_fingerprint" ]] ||
      die "Unexpected Canonical TDX PPA signing-key fingerprint: $observed_fingerprint"
    gpg --dearmor <"$tmp_dir/ppa.asc" >"$tmp_dir/ppa.gpg"
    as_root install -m 0644 "$tmp_dir/ppa.gpg" /etc/apt/keyrings/canonical-tdx-attestation.gpg
    printf 'deb [signed-by=/etc/apt/keyrings/canonical-tdx-attestation.gpg] %s %s main\n' \
      "$base" "$suite" |
      as_root tee /etc/apt/sources.list.d/canonical-tdx-attestation.list >/dev/null
    as_root apt-get update
    as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades \
      sgx-dcap-pccs tdx-qgs libsgx-dcap-default-qpl \
      sgx-ra-service sgx-pck-id-retrieval-tool
    rm -rf "$tmp_dir"
  else
    log "Intel TDX Quote Generation Service is already installed"
  fi

  pccs_config="/opt/intel/sgx-dcap-pccs/config/default.json"
  as_root test -s "$pccs_config" ||
    die "PCCS configuration is missing after installing the TDX attestation packages"

  if [[ -n "$pcs_key" && "$configured_key" != "$pcs_key" ]]; then
    log "Installing the Intel PCS subscription key into the root-owned PCCS configuration"
    tmp_dir="$(mktemp -d)"
    chmod 0700 "$tmp_dir"
    printf '%s' "$pcs_key" >"$tmp_dir/pcs-api-key"
    chmod 0600 "$tmp_dir/pcs-api-key"
    as_root cat "$pccs_config" >"$tmp_dir/pccs.json"
    jq --rawfile api_key "$tmp_dir/pcs-api-key" \
      '.ApiKey = ($api_key | rtrimstr("\n"))' \
      "$tmp_dir/pccs.json" >"$tmp_dir/pccs-updated.json"
    # PCCS runs as the unprivileged pccs user and must be able to read this
    # file. Keep the subscription key root-owned and expose it only to the
    # service account through its private group.
    as_root install -o root -g pccs -m 0640 "$tmp_dir/pccs-updated.json" "$pccs_config"
    rm -rf "$tmp_dir"
  fi

  if [[ -z "$pcs_key" ]]; then
    # Canonical PCCS 1.21 constructs an empty Ocp-Apim-Subscription-Key
    # header even though Intel's production PCS accepts an omitted header.
    # Apply a guarded compatibility edit at the common request boundary.
    pcs_client="/opt/intel/sgx-dcap-pccs/pcs_client/pcs_client.js"
    as_root test -s "$pcs_client" ||
      die "PCCS client source is missing: $pcs_client"
    if ! as_root grep -Fq 'bare-metal-coco: omit an empty Intel PCS subscription header' "$pcs_client"; then
      log "Patching PCCS to omit its empty Intel PCS subscription header"
      tmp_dir="$(mktemp -d)"
      chmod 0700 "$tmp_dir"
      as_root cat "$pcs_client" >"$tmp_dir/pcs_client.js"
      awk '
        { print }
        $0 == "async function do_request(url, options) {" {
          print "  // bare-metal-coco: omit an empty Intel PCS subscription header"
          print "  if (!Config.get('\''ApiKey'\'') && options.headers) {"
          print "    delete options.headers['\''Ocp-Apim-Subscription-Key'\''];"
          print "  }"
          inserted++
        }
        END { if (inserted != 1) exit 42 }
      ' "$tmp_dir/pcs_client.js" >"$tmp_dir/pcs_client-updated.js" ||
        die "Installed PCCS client does not contain the expected request-function anchor"
      node --check "$tmp_dir/pcs_client-updated.js" >/dev/null ||
        die "Refusing a keyless PCCS compatibility edit that does not pass node --check"
      as_root install -o root -g root -m 0644 "$tmp_dir/pcs_client-updated.js" "$pcs_client"
      rm -rf "$tmp_dir"
    fi
  fi

  as_root runuser -u pccs -- test -r "$pccs_config" ||
    die "PCCS configuration is not readable by the pccs service account"
  as_root systemctl enable --now pccs qgsd
  as_root systemctl restart pccs qgsd
  pccs_endpoint_ready() {
    systemctl is-active --quiet pccs &&
      curl --insecure --silent --show-error --max-time 2 \
        --output /dev/null https://127.0.0.1:8081/
  }
  wait_for "Intel PCCS HTTPS endpoint" 60 pccs_endpoint_ready
  as_root systemctl is-active --quiet qgsd ||
    die "Intel TDX Quote Generation Service failed to start"
  if [[ -n "$pcs_key" ]]; then
    log "Intel PCCS and QGS are active with a configured PCS subscription key"
  else
    log "Intel PCCS and QGS are active in production-PCS keyless mode"
  fi
}

install_tdx_quote_service

log "Labelling $node_name for ${TEE_NAME} and NVIDIA VM passthrough"
kctl label node "$node_name" "${TEE_NODE_LABEL_KEY}=true" --overwrite
kctl label node "$node_name" nvidia.com/gpu.workload.config=vm-passthrough --overwrite

log "Installing Kata Containers ${KATA_VERSION}"
helmctl upgrade --install kata-deploy oci://ghcr.io/kata-containers/kata-deploy-charts/kata-deploy \
  --namespace kata-system --create-namespace --version "$KATA_VERSION" \
  --set nfd.enabled=false --wait --timeout 15m

log "Waiting for Kata host installation and containerd runtime fragment"
kata_host_files_ready() {
  as_root test -x /opt/kata/bin/containerd-shim-kata-v2 &&
    as_root test -s /opt/kata/containerd/config.d/kata-deploy.toml &&
    as_root grep -Fq "runtimes.${RUNTIME_CLASS}]" /opt/kata/containerd/config.d/kata-deploy.toml
}
wait_for "Kata host files for ${RUNTIME_CLASS}" 900 kata_host_files_ready

# Older executions of stage 10 did not predeclare the Kata import. kata-deploy
# normally adds it, but repair the suite-owned base configuration if it did not.
if ! as_root grep -Fq "/opt/kata/containerd/config.d/" /etc/containerd/config.toml; then
  log "Adding the Kata runtime fragment import to containerd"
  as_root sed -i \
    "s|^imports = \\['/etc/containerd/conf.d/\\*.toml'\\]$|imports = ['/etc/containerd/conf.d/*.toml', '/opt/kata/containerd/config.d/*.toml']|" \
    /etc/containerd/config.toml
fi
as_root grep -Fq "/opt/kata/containerd/config.d/" /etc/containerd/config.toml ||
  die "Could not add the Kata import to /etc/containerd/config.toml"

log "Restarting containerd with the Kata runtime configuration"
as_root systemctl restart containerd
as_root systemctl is-active --quiet containerd ||
  die "containerd did not restart successfully with the Kata configuration"
# Kubernetes 1.34 can cache the CRI runtime's cgroup-driver response. After the
# Kata fragment changes and restarts containerd, restart kubelet so new pod
# sandboxes use the effective systemd-cgroup runtime configuration.
as_root systemctl restart kubelet
kubernetes_api_ready() {
  kctl get --raw=/readyz >/dev/null 2>&1
}
wait_for "Kubernetes API after runtime restart" 180 kubernetes_api_ready

containerd_runtime_ready() {
  as_root containerd config dump |
    awk -v needle="runtimes.${RUNTIME_CLASS}]" 'index($0, needle) {found=1} END {exit !found}'
}
wait_for "effective containerd runtime ${RUNTIME_CLASS}" 300 containerd_runtime_ready

nydus_plugin_ready() {
  as_root ctr plugins ls |
    awk '$1 == "io.containerd.snapshotter.v1" && $2 == "nydus-for-kata-tee" && $4 == "ok" {found=1} END {exit !found}'
}
wait_for "nydus-for-kata-tee containerd snapshotter" 300 nydus_plugin_ready

runtime_config="$(
  as_root awk -F' = ' -v runtime="$RUNTIME_CLASS" '
    $0 ~ "runtimes\\." runtime "\\.options\\]" {in_runtime=1; next}
    in_runtime && /^\[/ {exit}
    in_runtime && $1 == "ConfigPath" {
      gsub(/"/, "", $2)
      print $2
      exit
    }
  ' /opt/kata/containerd/config.d/kata-deploy.toml
)"
[[ -n "$runtime_config" ]] ||
  die "Could not resolve the Kata configuration for ${RUNTIME_CLASS}"
as_root grep -Eq '^emptydir_mode = "block-encrypted"$' "$runtime_config" ||
  die "${RUNTIME_CLASS} does not enable released CoCo block-encrypted emptyDir volumes"
log "Released CoCo LUKS2/dm-crypt emptyDir support is enabled"

log "Installing NVIDIA GPU Operator ${GPU_OPERATOR_VERSION} for Kata passthrough"
helmctl repo add nvidia https://helm.ngc.nvidia.com/nvidia --force-update
helmctl repo update nvidia
helmctl upgrade --install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator --create-namespace --version "$GPU_OPERATOR_VERSION" \
  --set sandboxWorkloads.enabled=true \
  --set sandboxWorkloads.mode=kata \
  --set sandboxWorkloads.defaultWorkload=container \
  --set nfd.enabled=true \
  --set nfd.nodefeaturerules=true \
  --set ccManager.enabled=true \
  --set ccManager.defaultMode=on \
  --set vfioManager.enabled=true \
  --set kataSandboxDevicePlugin.enabled=true \
  --set kataManager.enabled=false \
  --wait --timeout 25m

log "Waiting for CoCo GPU readiness"
kctl wait --for=condition=Ready nodes "$node_name" --timeout=10m
runtime_ready() { kctl get runtimeclass "$RUNTIME_CLASS" >/dev/null 2>&1; }
cc_ready() { [[ "$(kctl get node "$node_name" -o jsonpath='{.metadata.labels.nvidia\.com/cc\.ready\.state}' 2>/dev/null)" == true ]]; }
gpu_allocatable() {
  local escaped_resource value
  escaped_resource="${GPU_RESOURCE//./\\.}"
  value="$(kctl get node "$node_name" -o "jsonpath={.status.allocatable.${escaped_resource}}" 2>/dev/null || true)"
  [[ "${value:-0}" != 0 ]]
}
wait_for "RuntimeClass ${RUNTIME_CLASS}" 900 runtime_ready
wait_for "NVIDIA confidential-computing readiness label" 1800 cc_ready
wait_for "$GPU_RESOURCE allocatable resource" 1800 gpu_allocatable

mapfile -t gpu_bdfs < <(lspci -Dnd 10de: | awk '$2 ~ /^03(00|02):$/{print $1}')
((${#gpu_bdfs[@]} > 0)) || die "No NVIDIA VGA/3D GPU PCI device was detected"
for gpu_bdf in "${gpu_bdfs[@]}"; do
  gpu_driver="$(basename "$(readlink -f "/sys/bus/pci/devices/$gpu_bdf/driver" 2>/dev/null)" 2>/dev/null || true)"
  [[ "$gpu_driver" == vfio-pci ]] || die "GPU $gpu_bdf is bound to '${gpu_driver:-no driver}', not vfio-pci, after GPU Operator deployment"
  log "Post-install validation: GPU $gpu_bdf is bound to vfio-pci"
done
log "Kata CoCo runtime and GPU passthrough are ready"
