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

log "Labelling $node_name for ${TEE_NAME} and NVIDIA VM passthrough"
kctl label node "$node_name" "${TEE_NODE_LABEL_KEY}=true" --overwrite
kctl label node "$node_name" nvidia.com/gpu.workload.config=vm-passthrough --overwrite

log "Installing Kata Containers ${KATA_VERSION}"
source "$SCRIPT_DIR/../public/kata-platform.env"
chart="$SCRIPT_DIR/../public/kata-deploy-${KATA_VERSION}.tgz"
[[ -s $chart ]] || die "Receive the public Kata chart before installing the runtime"
printf '%s  %s\n' "$KATA_CHART_TGZ_SHA256" "$chart" | sha256sum --check --strict
helmctl upgrade --install kata-deploy "$chart" \
  --namespace kata-system --create-namespace --reset-values \
  --set node-feature-discovery.enabled=false \
  --set-string "image.reference=$KATA_DEPLOY_AMD64" --wait --timeout 15m

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
