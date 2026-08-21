#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
load_config
require_root_or_sudo

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
download_dir="$STATE_DIR/downloads"
prepare_download_dir

if [[ "$ALLOW_PACKAGE_DOWNGRADES" != 1 ]]; then
  for package in kubelet kubeadm kubectl; do
    installed_version="$(dpkg-query -W -f='${Version}' "$package" 2>/dev/null || true)"
    if [[ -n "$installed_version" ]] &&
      dpkg --compare-versions "$installed_version" gt "$KUBERNETES_VERSION"; then
      die "$package $installed_version is newer than pinned $KUBERNETES_VERSION; set ALLOW_PACKAGE_DOWNGRADES=1 only after approving the downgrade"
    fi
  done
fi

log "Installing host packages and Kubernetes prerequisites"
as_root apt-get update
host_packages=(ca-certificates curl gpg jq openssl gcc make git python3-venv skopeo pciutils uidmap xxd)
if [[ "$TEE_PLATFORM" == tdx ]] && ! command -v go >/dev/null 2>&1; then
  host_packages+=(golang-go)
fi
[[ -x /usr/bin/runc ]] || host_packages+=(runc)
as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y "${host_packages[@]}"
[[ -x /usr/bin/runc ]] || die "A trusted runc binary is required at /usr/bin/runc"
printf 'overlay\nbr_netfilter\n' | as_root tee /etc/modules-load.d/k8s.conf >/dev/null
as_root modprobe overlay
as_root modprobe br_netfilter
printf 'net.bridge.bridge-nf-call-iptables = 1\nnet.bridge.bridge-nf-call-ip6tables = 1\nnet.ipv4.ip_forward = 1\n' | as_root tee /etc/sysctl.d/99-kubernetes-cri.conf >/dev/null
as_root sysctl --system >/dev/null
as_root swapoff -a
as_root sed -ri '/^[^#].+[[:space:]]swap[[:space:]]/s/^/# disabled-by-bare-metal-coco: /' /etc/fstab

if ! command -v containerd >/dev/null || [[ "$(containerd --version 2>/dev/null)" != *"v${CONTAINERD_VERSION}"* ]]; then
  log "Installing containerd ${CONTAINERD_VERSION}"
  archive="$download_dir/containerd-${CONTAINERD_VERSION}-linux-amd64.tar.gz"
  ensure_download_verified "https://github.com/containerd/containerd/releases/download/v${CONTAINERD_VERSION}/containerd-${CONTAINERD_VERSION}-linux-amd64.tar.gz" "$CONTAINERD_SHA256" "$archive"
  as_root tar -C /usr/local -xzf "$archive"
fi

log "Installing CNI plugins ${CNI_PLUGINS_VERSION}"
cni_name="cni-plugins-linux-amd64-${CNI_PLUGINS_VERSION}.tgz"
cni_archive="$download_dir/$cni_name"
ensure_download "https://github.com/containernetworking/plugins/releases/download/${CNI_PLUGINS_VERSION}/$cni_name" "$cni_archive"
ensure_download "https://github.com/containernetworking/plugins/releases/download/${CNI_PLUGINS_VERSION}/$cni_name.sha256" "$cni_archive.sha256"
if ! (cd "$download_dir" && sha256sum -c "$(basename "$cni_archive.sha256")"); then
  if [[ "$IGNORE_CHECKSUM_MISMATCH" == 1 ]]; then
    echo "WARNING: ignoring CNI plugin checksum mismatch for $cni_archive" >&2
  else
    rm -f "$cni_archive" "$cni_archive.sha256"
    ensure_download "https://github.com/containernetworking/plugins/releases/download/${CNI_PLUGINS_VERSION}/$cni_name" "$cni_archive"
    ensure_download "https://github.com/containernetworking/plugins/releases/download/${CNI_PLUGINS_VERSION}/$cni_name.sha256" "$cni_archive.sha256"
    (cd "$download_dir" && sha256sum -c "$(basename "$cni_archive.sha256")")
  fi
fi
as_root install -d -m 0755 /opt/cni/bin
as_root tar -C /opt/cni/bin -xzf "$cni_archive"

as_root install -d -m 0755 /etc/containerd /etc/containerd/conf.d
as_root tee /etc/containerd/config.toml >/dev/null <<'EOF'
version = 3
root = '/var/lib/containerd'
state = '/run/containerd'
imports = ['/etc/containerd/conf.d/*.toml', '/opt/kata/containerd/config.d/*.toml']
[grpc]
  address = '/run/containerd/containerd.sock'
[plugins]
  [plugins.'io.containerd.cri.v1.images']
    snapshotter = 'overlayfs'
    [plugins.'io.containerd.cri.v1.images'.registry]
      config_path = '/etc/containerd/certs.d'
    [plugins.'io.containerd.cri.v1.images'.pinned_images]
      sandbox = 'registry.k8s.io/pause:3.10.1'
  [plugins.'io.containerd.cri.v1.runtime']
    enable_cdi = true
    cdi_spec_dirs = ['/etc/cdi', '/var/run/cdi']
    [plugins.'io.containerd.cri.v1.runtime'.containerd]
      default_runtime_name = 'runc'
      [plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.runc]
        runtime_type = 'io.containerd.runc.v2'
        [plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.runc.options]
          BinaryName = '/usr/bin/runc'
          SystemdCgroup = true
    [plugins.'io.containerd.cri.v1.runtime'.cni]
      bin_dirs = ['/opt/cni/bin']
      conf_dir = '/etc/cni/net.d'
      max_conf_num = 1
EOF
as_root tee /etc/systemd/system/containerd.service >/dev/null <<'EOF'
[Unit]
Description=containerd container runtime
Documentation=https://containerd.io
After=network.target local-fs.target
[Service]
ExecStartPre=-/sbin/modprobe overlay
ExecStart=/usr/local/bin/containerd
Type=notify
Delegate=yes
KillMode=process
Restart=always
RestartSec=5
LimitNPROC=infinity
LimitCORE=infinity
LimitNOFILE=infinity
TasksMax=infinity
OOMScoreAdjust=-999
[Install]
WantedBy=multi-user.target
EOF
as_root systemctl daemon-reload
as_root systemctl enable containerd
# Apply the generated configuration on both first install and idempotent reruns.
as_root systemctl restart containerd
as_root systemctl is-active --quiet containerd ||
  die "containerd did not start with the generated configuration"

log "Installing Kubernetes ${KUBERNETES_VERSION}"
as_root install -d -m 0755 /etc/apt/keyrings
ensure_download "https://pkgs.k8s.io/core:/stable:/${KUBERNETES_MINOR}/deb/Release.key" "$download_dir/kubernetes-${KUBERNETES_MINOR}-Release.key"
gpg --dearmor <"$download_dir/kubernetes-${KUBERNETES_MINOR}-Release.key" >"$tmp_dir/kubernetes.gpg"
as_root install -m 0644 "$tmp_dir/kubernetes.gpg" /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/${KUBERNETES_MINOR}/deb/ /" | as_root tee /etc/apt/sources.list.d/kubernetes.list >/dev/null
as_root apt-get update
kubernetes_apt_options=(-y)
[[ "$ALLOW_PACKAGE_DOWNGRADES" != 1 ]] ||
  kubernetes_apt_options+=(--allow-downgrades --allow-change-held-packages)
as_root env DEBIAN_FRONTEND=noninteractive apt-get install "${kubernetes_apt_options[@]}" \
  "kubelet=${KUBERNETES_VERSION}" "kubeadm=${KUBERNETES_VERSION}" "kubectl=${KUBERNETES_VERSION}"
as_root apt-mark hold kubelet kubeadm kubectl
# An existing package may not re-enable its unit. Ensure the cluster remains
# available after the next host reboot.
as_root systemctl enable kubelet
if [[ ! -x /usr/local/bin/helm ]] || [[ "$(/usr/local/bin/helm version --short 2>/dev/null)" != *"v${HELM_VERSION}"* ]]; then
  log "Installing Helm ${HELM_VERSION}"
  helm_archive="$download_dir/helm-${HELM_VERSION}-linux-amd64.tar.gz"
  ensure_download_verified "https://get.helm.sh/helm-${HELM_VERSION}-linux-amd64.tar.gz" "$HELM_SHA256" "$helm_archive"
  tar -C "$tmp_dir" -xzf "$helm_archive"
  as_root install -m 0755 "$tmp_dir/linux-amd64/helm" /usr/local/bin/helm
fi

if [[ ! -s "$KUBECONFIG_PATH" ]]; then
  log "Initializing the single-node Kubernetes control plane"
  kubeadm_config="$tmp_dir/kubeadm.yaml"
  sed -e "s|@@NODE_IP@@|$NODE_IP|g" -e "s|@@K8S_SEMVER@@|$KUBERNETES_SEMVER|g" -e "s|@@POD_CIDR@@|$POD_CIDR|g" -e "s|@@SERVICE_CIDR@@|$SERVICE_CIDR|g" -e "s|@@CLUSTER_DNS@@|$CLUSTER_DNS|g" "$SCRIPT_DIR/templates/kubeadm.yaml.in" >"$kubeadm_config"
  as_root kubeadm init --config "$kubeadm_config"
else
  log "Existing cluster found at $KUBECONFIG_PATH; kubeadm init skipped"
fi

log "Installing Calico ${CALICO_VERSION} and enabling single-node scheduling"
calico="$download_dir/calico-${CALICO_VERSION}.yaml"
ensure_download_verified \
  "https://raw.githubusercontent.com/projectcalico/calico/${CALICO_VERSION}/manifests/calico.yaml" \
  "$CALICO_SHA256" "$calico"
kctl apply -f "$calico"
kctl taint nodes --all node-role.kubernetes.io/control-plane- >/dev/null 2>&1 || true
kctl wait --for=condition=Ready nodes --all --timeout=15m

login_user="${SUDO_USER:-${USER:-root}}"
if [[ "$login_user" != root ]]; then
  login_home="$(getent passwd "$login_user" | cut -d: -f6)"
  as_root install -d -o "$login_user" -g "$login_user" -m 0700 "$login_home/.kube"
  as_root install -o "$login_user" -g "$login_user" -m 0600 "$KUBECONFIG_PATH" "$login_home/.kube/config"
fi
log "Kubernetes is ready"
