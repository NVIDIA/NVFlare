#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
[[ $# == 1 ]] || { echo "Usage: $0 PLATFORM_ENV"; exit 2; }
source "$(realpath -e "$1")"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/validate-config.sh"
validate_target_host
sudo -n true
export KUBECONFIG=/etc/kubernetes/admin.conf
kctl() { sudo kubectl --kubeconfig "$KUBECONFIG" "$@"; }
hctl() { sudo helm --kubeconfig "$KUBECONFIG" "$@"; }
PROFILE_DIR="$PLATFORM_WORK_ROOT/$PLATFORM_PROFILE"
CHART="$PROFILE_DIR/artifacts/kata-deploy-$KATA_VERSION.tgz"
printf '%s  %s\n' "$KATA_CHART_TGZ_SHA256" "$CHART" | sha256sum --check --strict
node=$(kctl get nodes -o jsonpath='{.items[0].metadata.name}')
[[ -n $node && $(kctl get nodes -o name | wc -l) == 1 ]] || { echo 'Expected one trusted node'; exit 1; }
[[ $(sudo cat /sys/module/kvm_amd/parameters/sev_snp) == Y ]] || { echo 'SNP is disabled'; exit 1; }
kctl label node "$node" amd.feature.node.kubernetes.io/snp=true nvidia.com/gpu.workload.config=vm-passthrough --overwrite
hctl upgrade --install kata-deploy "$CHART" --namespace kata-system --create-namespace \
  --reset-values --set node-feature-discovery.enabled=false \
  --set-string "image.reference=$KATA_DEPLOY_AMD64" --wait --timeout 15m
kctl -n kata-system rollout status daemonset/kata-deploy --timeout=15m
[[ $(kctl -n kata-system get daemonset kata-deploy -o jsonpath='{.spec.template.spec.containers[?(@.name=="kube-kata")].image}') == "$KATA_DEPLOY_AMD64" ]]
sudo test -x /opt/kata/bin/containerd-shim-kata-v2
sudo grep -Fq '/opt/kata/containerd/config.d/' /etc/containerd/config.toml
sudo systemctl restart containerd kubelet
for attempt in {1..60}; do
  if kctl get --raw=/readyz >/dev/null 2>&1; then break; fi
  sleep 3
done
kctl get --raw=/readyz
hctl repo add nvidia https://helm.ngc.nvidia.com/nvidia --force-update
hctl repo update nvidia
hctl upgrade --install gpu-operator nvidia/gpu-operator --version v26.3.1 \
  --namespace gpu-operator --create-namespace \
  --set sandboxWorkloads.enabled=true --set sandboxWorkloads.mode=kata \
  --set sandboxWorkloads.defaultWorkload=container \
  --set nfd.enabled=true --set nfd.nodefeaturerules=true \
  --set ccManager.enabled=true --set ccManager.defaultMode=on \
  --set vfioManager.enabled=true --set kataSandboxDevicePlugin.enabled=true \
  --set kataManager.enabled=false --wait --timeout 25m
for attempt in {1..180}; do
  ready=$(kctl get node "$node" -o jsonpath='{.metadata.labels.nvidia\.com/cc\.ready\.state}')
  gpu=$(kctl get node "$node" -o jsonpath='{.status.allocatable.nvidia\.com/pgpu}')
  if [[ $ready == true && ${gpu:-0} == 1 ]]; then break; fi
  sleep 5
done
[[ $ready == true && ${gpu:-0} == 1 ]] || { echo 'CC GPU is not ready'; exit 1; }
kctl get runtimeclass "$RUNTIME_CLASS"
sudo ctr plugins ls | awk '$1 == "io.containerd.snapshotter.v1" && $2 == "nydus-for-kata-tee" && $4 == "ok" {ok=1} END {exit !ok}'
mapfile -t gpu_bdfs < <(lspci -Dnd 10de: | awk '$2 ~ /^03(00|02):$/ {print $1}')
[[ ${#gpu_bdfs[@]} == 1 ]] || { echo 'This approved profile requires exactly one NVIDIA GPU'; exit 1; }
[[ $(basename "$(readlink -f "/sys/bus/pci/devices/${gpu_bdfs[0]}/driver")") == vfio-pci ]]
kctl get nodes -o json > "$PROFILE_DIR/kubernetes-node-profile.json"
hctl get values kata-deploy -n kata-system -a > "$PROFILE_DIR/kata-helm-values.yaml"
hctl get values gpu-operator -n gpu-operator -a > "$PROFILE_DIR/gpu-operator-helm-values.yaml"
printf 'Trusted SNP runtime and one CC GPU are ready.\n'
