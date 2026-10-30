# Prepare the GPU host and Kubernetes

This is the Host B portion of the [beginner prerequisite appendix](../PREREQUISITES.md).
This optional appendix helps beginners bring up the prerequisite systems for the
NVFlare CoCo workflow. Consult the official documentation for each component and
the version you deploy. If there is any discrepancy, the component's official
documentation takes precedence.

Trustee and the registry run separately on Host A. Run the commands here on the
physical GPU machine as the same administrator account, using `sudo` where shown.
Load the input file again after reconnecting or rebooting:

~~~bash
source "$HOME/coco-private/inputs.env"
mkdir -p "$HOME/coco-private/downloads"
cd "$HOME/coco-private/downloads"
~~~

This reference uses one physical GPU and one Kubernetes node. It does not cover
HA control planes, nested virtualization, shared GPUs, or multi-GPU/NVSwitch
configuration. Use the component documentation for those configurations.
Use the [component versions](../PREREQUISITES.md#component-versions) specified in
the parent appendix.

## 1. Prepare firmware and CPU virtualization before Kubernetes

Consult the server manufacturer's BIOS and firmware instructions together with
the [NVIDIA hardware matrix](https://docs.nvidia.com/confidential-computing/containers/v1.0.0/supported-platforms)
and the matching [SNP](https://docs.nvidia.com/cc-deployment-guide-snp.pdf) or
[TDX](https://docs.nvidia.com/cc-deployment-guide-tdx.pdf) hardware preparation
guide. Use their hardware/firmware sections; the Kata chart later supplies its
own QEMU, firmware, and guest artifacts. Do not install a second libvirt/QEMU
stack from the standalone-VM examples.

Install Ubuntu updates and inspection utilities:

~~~bash
sudo apt-get update
sudo apt-get install -y linux-generic linux-firmware pciutils cpu-checker \
  iproute2 jq curl ca-certificates openssl python3 python3-yaml
sudo reboot
~~~

After reconnecting, inspect the actual platform:

~~~bash
source "$HOME/coco-private/inputs.env"
cat /etc/os-release
uname -r
lscpu
lspci -nn -d 10de:
timedatectl show -p NTPSynchronized
~~~

Expected: Ubuntu 26.04 on x86_64, a CPU/GPU combination covered by the applicable
hardware guidance, and `NTPSynchronized=yes`. Confirm firmware and GPU VBIOS
versions against that platform's requirements. Installing a newer kernel cannot
make an unsupported CPU or GPU confidential-computing capable.

### AMD SEV-SNP branch (`CPU_TEE=snp`)

In the server BIOS, enable AMD virtualization/SVM, IOMMU, SEV, SEV-ES, and
SEV-SNP as required by the server's firmware. Apply the vendor's current
compatible BIOS, AMD security processor firmware, and CPU microcode updates.
Record the four independent minimum TCB values from the platform security
baseline for later capture; do not derive acceptable minimums by copying the
running worker's reported values.

Use `sudoedit /etc/default/grub` to add the following to the existing
`GRUB_CMDLINE_LINUX_DEFAULT` value without removing other required options:

~~~text
amd_iommu=on iommu=pt kvm_amd.sev=1 kvm_amd.sev_es=1 kvm_amd.sev_snp=1
~~~

~~~bash
sudo update-grub
sudo reboot
~~~

After reconnecting, check the firmware/KVM state:

~~~bash
sudo modprobe kvm_amd
test -c /dev/kvm && test -c /dev/sev
cat /sys/module/kvm_amd/parameters/sev
cat /sys/module/kvm_amd/parameters/sev_es
cat /sys/module/kvm_amd/parameters/sev_snp
sudo journalctl -k -b | grep -Ei 'SEV|SNP|AMD-Vi|IOMMU'
~~~

Expected: the three parameters are enabled (`Y` or `1`), the devices exist, and
the current boot log reports usable SNP without initialization failures. Missing
parameters, devices, or SNP firmware support mean this branch is not ready;
resolve the BIOS/kernel/firmware mismatch before installing Kubernetes. See the
[Linux SEV documentation](https://docs.kernel.org/virt/kvm/x86/amd-memory-encryption.html)
for the kernel interface.

### Intel TDX branch (`CPU_TEE=tdx`)

In the server BIOS, enable VT-x, VT-d, total memory encryption, and TDX, with
private TDX KeyIDs and the compatible TDX module configured according to the
server documentation. Install the required BIOS/microcode/TDX-module updates.
Do not assume that the `vmx` CPU flag demonstrates a usable TDX module.

Use `sudoedit /etc/default/grub` to add these options to the existing
`GRUB_CMDLINE_LINUX_DEFAULT` value:

~~~text
intel_iommu=on iommu=pt kvm_intel.tdx=1 nohibernate
~~~

~~~bash
sudo update-grub
sudo reboot
~~~

After reconnecting:

~~~bash
sudo modprobe kvm_intel
test -c /dev/kvm
cat /sys/module/kvm_intel/parameters/tdx
sudo journalctl -k -b | grep -Ei 'TDX|SEAM|DMAR|IOMMU'
~~~

Expected: KVM TDX is enabled and the log reports BIOS-provided TDX resources
without initialization failures. Some kernels initialize the module on the
first TDX VM creation: its successful initialization must then be confirmed
again during the sample in step 7. A missing module, zero private KeyIDs,
initialization error, or unsupported CPU is a failure, not a skipped check.
See [Linux TDX host support](https://docs.kernel.org/arch/x86/tdx.html).

TDX attestation also requires a host Quote Generation Service (QGS). Follow
Intel's official [QGS installation and configuration instructions](https://cc-enabling.trustedservices.intel.com/intel-tdx-enabling-guide/05/host_os_setup/#set-up-quote-generation-service-qgs),
including platform certificate provisioning and PCK certificate retrieval.
For the Kata 3.29.0 TDX profile used here, explicitly configure QGS to listen on
vsock port `4050` to match Kata's quote-generation endpoint, then restart
`qgsd.service`. Leaving the QGS port setting commented selects a Unix socket
instead; a running service alone does not demonstrate a usable quote path.

Once a TDX guest is available, follow Intel's official
[TD quote-generation check](https://cc-enabling.trustedservices.intel.com/intel-tdx-enabling-guide/07/trust_domain_at_runtime/#td-quote-generation)
inside the guest and confirm that it produces a nonempty `quote.dat` before
[measurement capture](../MEASUREMENTS.md). Resolve quote-generation failures
before proceeding with TDX attestation.

### Check GPU isolation for either branch

~~~bash
find /sys/kernel/iommu_groups -type l | head
lspci -nnk -d 10de:
sudo kvm-ok
~~~

Expected: populated IOMMU groups and usable KVM. Identify the GPU's PCI address
and IOMMU group; all devices required by its passthrough group must be suitable
for assignment. The GPU must not be the host's active display device.
An NVIDIA host driver or `nouveau` must not own this passthrough GPU. If Ubuntu
installed either automatically, remove its use of this GPU using the vendor's
host preparation instructions and reboot before proceeding. Do not unload a
driver from a GPU currently used by another workload. VFIO Manager will bind
the GPU to `vfio-pci` later; check that final binding in step 6.

These checks establish host prerequisites. Signed evidence verification during
[measurement capture](../MEASUREMENTS.md) establishes the attestation baseline.

## 2. Install containerd and its dependencies

Follow the [containerd binary installation guide](https://github.com/containerd/containerd/blob/v2.3.7/docs/getting-started.md)
and [CRI configuration guide](https://github.com/containerd/containerd/blob/v2.3.7/docs/cri/config.md).
This fresh-host recipe installs containerd independently from Docker. Install
Docker only on Host A; Docker's packages can replace the containerd used here.

Disable swap now and remove or comment its entries in `/etc/fstab` with
`sudoedit /etc/fstab`, so it stays disabled after reboot. If the host uses a
separate swap service, disable that service as well.

~~~bash
sudo swapoff -a
sudo modprobe overlay
sudo modprobe br_netfilter
printf 'overlay\nbr_netfilter\n' | sudo tee /etc/modules-load.d/coco-k8s.conf
cat <<'EOF' | sudo tee /etc/sysctl.d/90-coco-k8s.conf
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward = 1
EOF
sudo sysctl --system
sudo apt-get install -y runc

cd "$HOME/coco-private/downloads"
CONTAINERD_ASSET=containerd-2.3.7-linux-amd64.tar.gz
curl -fLO "https://github.com/containerd/containerd/releases/download/v2.3.7/$CONTAINERD_ASSET"
curl -fLO "https://github.com/containerd/containerd/releases/download/v2.3.7/$CONTAINERD_ASSET.sha256sum"
sha256sum -c "$CONTAINERD_ASSET.sha256sum"
sudo tar -C /usr/local -xzf "$CONTAINERD_ASSET"
curl -fLo containerd.service \
  https://raw.githubusercontent.com/containerd/containerd/v2.3.7/containerd.service
sudo install -D -m 0644 containerd.service /usr/local/lib/systemd/system/containerd.service
sudo mkdir -p /etc/containerd
containerd config default | sudo tee /etc/containerd/config.toml >/dev/null
~~~

Use `sudoedit /etc/containerd/config.toml` and change the existing runc options
section to `SystemdCgroup = true`. Ensure CRI is not listed in
`disabled_plugins`. The relevant containerd 2.x section is:

~~~toml
[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.runc.options]
  SystemdCgroup = true
~~~

Install CNI binaries, including Flannel's required `portmap`, and the CRI client:

~~~bash
CNI_ASSET=cni-plugins-linux-amd64-v1.7.1.tgz
curl -fLO "https://github.com/containernetworking/plugins/releases/download/v1.7.1/$CNI_ASSET"
curl -fLO "https://github.com/containernetworking/plugins/releases/download/v1.7.1/$CNI_ASSET.sha256"
printf '%s  %s\n' "$(awk '{print $1}' "$CNI_ASSET.sha256")" "$CNI_ASSET" | sha256sum -c -
sudo mkdir -p /opt/cni/bin
sudo tar -C /opt/cni/bin -xzf "$CNI_ASSET"
CRI_ASSET=crictl-v1.36.0-linux-amd64.tar.gz
curl -fLO "https://github.com/kubernetes-sigs/cri-tools/releases/download/v1.36.0/$CRI_ASSET"
curl -fLO "https://github.com/kubernetes-sigs/cri-tools/releases/download/v1.36.0/$CRI_ASSET.sha256"
printf '%s  %s\n' "$(awk '{print $1}' "$CRI_ASSET.sha256")" "$CRI_ASSET" | sha256sum -c -
sudo tar -C /usr/local/bin -xzf "$CRI_ASSET"
cat <<'EOF' | sudo tee /etc/crictl.yaml
runtime-endpoint: unix:///run/containerd/containerd.sock
image-endpoint: unix:///run/containerd/containerd.sock
timeout: 120
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now containerd
~~~

Check before installing Kubernetes:

~~~bash
containerd --version
runc --version
sudo systemctl is-active containerd
sudo crictl info | jq -e '.status.conditions[] | select(.type == "RuntimeReady") | .status == true'
test -x /opt/cni/bin/portmap
swapon --show
~~~

Expected: containerd `2.3.7`, an active service, CRI `RuntimeReady=true`, an
executable portmap plugin, and no active swap. `NetworkReady=false` is expected
until Flannel is installed. If CRI fails, inspect `sudo journalctl -u containerd`
and correct the configuration/socket; `ctr` working alone does not prove CRI.
Keep the downloaded checksums and installed runc package version in the setup
record. For artifact authenticity, also consult each project's release signing
instructions; a downloaded checksum checks bytes against that release's list.

## 3. Create a single-node Kubernetes cluster

Use the official [kubeadm installation](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/),
[cluster creation](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/create-cluster-kubeadm/),
and [ports](https://kubernetes.io/docs/reference/networking/ports-and-protocols/)
references. Restrict API access to the administrator network. Keep required
control-plane/node traffic available; do not expose etcd to Host A or the Internet.

~~~bash
sudo apt-get install -y apt-transport-https gpg
sudo install -d -m 0755 /etc/apt/keyrings
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.36/deb/Release.key \
  | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
printf '%s\n' 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.36/deb/ /' \
  | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
K8S_PACKAGE_VERSION=1.36.5-1.1
sudo apt-get install -y kubeadm="$K8S_PACKAGE_VERSION" \
  kubelet="$K8S_PACKAGE_VERSION" kubectl="$K8S_PACKAGE_VERSION"
sudo apt-mark hold kubeadm kubelet kubectl
sudo systemctl enable kubelet
~~~

Check the installed version before initializing:

~~~bash
kubeadm version -o short
kubectl version --client
~~~

Expected: `v1.36.5`. If that exact package is unavailable, stop and consult the
release repository rather than substituting another minor version. Kubelet may
restart repeatedly before `kubeadm init`; this is expected at this stage.

For the sample networks below, first inspect `ip route`: the machine/site routes
must not overlap `10.244.0.0/16` or `10.96.0.0/12`. If they overlap, choose unused
Pod/Service CIDRs and also update Flannel and the provisioning service address.

~~~bash
export NODE_NAME=coco-worker
export POD_CIDR=10.244.0.0/16
export SERVICE_CIDR=10.96.0.0/12
cat > kubeadm.yaml <<EOF
apiVersion: kubeadm.k8s.io/v1beta4
kind: InitConfiguration
localAPIEndpoint:
  advertiseAddress: "$WORKER_IP"
nodeRegistration:
  name: "$NODE_NAME"
  criSocket: unix:///run/containerd/containerd.sock
---
apiVersion: kubeadm.k8s.io/v1beta4
kind: ClusterConfiguration
kubernetesVersion: v1.36.5
networking:
  podSubnet: "$POD_CIDR"
  serviceSubnet: "$SERVICE_CIDR"
---
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
cgroupDriver: systemd
resolvConf: /run/systemd/resolve/resolv.conf
runtimeRequestTimeout: 20m
featureGates:
  RuntimeClassInImageCriApi: true
EOF
sudo kubeadm init --config kubeadm.yaml
mkdir -p "$HOME/.kube"
sudo install -o "$(id -u)" -g "$(id -g)" -m 0600 /etc/kubernetes/admin.conf "$HOME/.kube/config"
kubectl taint nodes "$NODE_NAME" node-role.kubernetes.io/control-plane:NoSchedule-
~~~

The taint removal intentionally permits workloads on this example's sole
control-plane node. Do not use that scheduling layout as a production HA design.

## 4. Install Flannel and configure guest DNS

The [Flannel release instructions](https://github.com/flannel-io/flannel/blob/v0.28.10/README.md)
describe its network prerequisites. Download the versioned manifest, inspect it,
and keep its checksum with the setup record:

~~~bash
curl -fLo kube-flannel.yml \
  https://github.com/flannel-io/flannel/releases/download/v0.28.10/kube-flannel.yml
sha256sum kube-flannel.yml
editor kube-flannel.yml
kubectl apply -f kube-flannel.yml
kubectl -n kube-flannel rollout status daemonset/kube-flannel-ds --timeout=5m
kubectl wait --for=condition=Ready node/"$NODE_NAME" --timeout=5m
kubectl -n kube-system rollout status deployment/coredns --timeout=5m
~~~

Before applying, verify the manifest's image versions and `Network` match the
release and `$POD_CIDR`. If your firewall restricts internal traffic, allow the
selected Flannel backend traffic (UDP 8472 for its default VXLAN backend) only
between cluster nodes. A single-node cluster still needs the bridge module and
CNI plugins configured in step 2.

For a laboratory deployment without site DNS, the host `/etc/hosts` entries in
the main appendix do not populate guest DNS. Use
`kubectl -n kube-system edit configmap coredns` and add a single `hosts` block
inside the existing `.:53` server block, substituting the actual values:

~~~text
hosts {
    <SERVICES_IP> registry.coco.lab trustee.coco.lab
    fallthrough
}
~~~

Use the chosen `$REGISTRY_HOST` and `$TRUSTEE_HOST` instead of those sample names
if changed. Preserve the existing Kubernetes/forwarding plugins. With working
site DNS, configure its forwarding normally and omit the `hosts` block. See
[customizing cluster DNS](https://kubernetes.io/docs/tasks/administer-cluster/dns-custom-nameservers/).

~~~bash
kubectl -n kube-system rollout restart deployment/coredns
kubectl -n kube-system rollout status deployment/coredns --timeout=5m
sudo crictl info | jq -e '[.status.conditions[] | select(.type == "RuntimeReady" or .type == "NetworkReady") | .status] | (length == 2) and all(. == true)'
kubectl get nodes -o wide
kubectl -n kube-system get pods
~~~

Expected: RuntimeReady and NetworkReady are true, the node is Ready, and CoreDNS
is running. If not, inspect the Flannel/CoreDNS logs, Pod events, subnet settings,
and `/etc/cni/net.d`. Do not continue to Kata while ordinary networking is broken.
The confidential sample later exercises the selected runtime; measurement
capture and launch also need guests to resolve and reach Host A with valid TLS.

## 5. Install Kata 3.29.0

Use the [versioned NVIDIA deployment guide](https://docs.nvidia.com/confidential-computing/containers/v1.0.0/confidential-containers-deploy)
and the [Kata 3.29 chart values](https://github.com/kata-containers/kata-containers/blob/3.29.0/tools/packaging/kata-deploy/helm-chart/kata-deploy/values.yaml).
These commands require Helm and Skopeo from the main appendix's tool preparation.

~~~bash
helm pull oci://ghcr.io/kata-containers/kata-deploy-charts/kata-deploy --version 3.29.0
sha256sum kata-deploy-3.29.0.tgz
KATA_DEPLOY_DIGEST=$(skopeo inspect --format '{{.Digest}}' docker://quay.io/kata-containers/kata-deploy:3.29.0)
KATA_KUBECTL_DIGEST=$(skopeo inspect --format '{{.Digest}}' docker://quay.io/kata-containers/kubectl:latest)
printf '%s\n' "kata-deploy=$KATA_DEPLOY_DIGEST" "kata-kubectl=$KATA_KUBECTL_DIGEST" \
  > kata-images.txt
cat > kata-values.yaml <<EOF
image:
  reference: quay.io/kata-containers/kata-deploy@$KATA_DEPLOY_DIGEST
kubectlImage:
  reference: quay.io/kata-containers/kubectl@$KATA_KUBECTL_DIGEST
node-feature-discovery:
  enabled: false
shims:
  disableAll: true
  qemu-nvidia-gpu-snp:
    enabled: true
  qemu-nvidia-gpu-tdx:
    enabled: true
defaultShim:
  amd64: qemu-nvidia-gpu-snp
EOF
helm template kata-deploy ./kata-deploy-3.29.0.tgz -n kata-system -f kata-values.yaml \
  > kata-rendered.yaml
editor kata-rendered.yaml
helm install kata-deploy ./kata-deploy-3.29.0.tgz \
  -n kata-system --create-namespace -f kata-values.yaml --wait --timeout 10m
kubectl -n kata-system rollout status daemonset/kata-deploy --timeout=10m
kubectl get runtimeclass kata-qemu-nvidia-gpu-snp kata-qemu-nvidia-gpu-tdx
~~~

For TDX change `defaultShim.amd64` to `qemu-nvidia-gpu-tdx` before installation.
Only the selected TEE can run on this worker; having both RuntimeClass objects
does not make an AMD machine support TDX, or vice versa. Keep the chart checksum,
Helm-reported OCI digest, resolved image digests, and rendered values. The helper
image has no versioned default tag: resolve it once as above, then reuse that
recorded digest for repeat installations instead of resolving `latest` again.
Verify the downloaded chart against the official release information. A checksum
you compute yourself records the download; it does not independently authenticate
the publisher.

Expected: Kata's DaemonSet finishes its rollout and both RuntimeClasses exist.
On a single node Helm may finish before initialization; use the rollout check.
If installation fails, inspect `kubectl -n kata-system logs daemonset/kata-deploy`
and `sudo journalctl -u containerd`; do not substitute a different Kata release.

Select and inspect the actual installed configuration:

~~~bash
source "$HOME/coco-private/inputs.env"
case "$CPU_TEE" in
  snp) export RUNTIME_CLASS=kata-qemu-nvidia-gpu-snp ;;
  tdx) export RUNTIME_CLASS=kata-qemu-nvidia-gpu-tdx ;;
  *) printf 'CPU_TEE must be snp or tdx\n' >&2; exit 1 ;;
esac
sudo containerd config dump | grep -A12 "$RUNTIME_CLASS"
sudo find /opt/kata -name "configuration-qemu-nvidia-gpu-$CPU_TEE.toml"
~~~

Use the configuration path actually referenced by that handler. Set
`INSTALLED_KATA_CONFIG` to it in your input file and edit it with `sudoedit`.
Preserve its existing kernel parameters and add exactly one
`agent.guest_components_rest_api=all` parameter in `hypervisor.qemu.kernel_params`.
This enables the guest-local AA token API used by NVFlare. Do not enable a host
listener for port 8006. Inspect any configuration drop-ins as well: conflicting
overrides must be resolved before capture.

~~~bash
source "$HOME/coco-private/inputs.env"
sudo python3 - "$INSTALLED_KATA_CONFIG" <<'PY'
import shlex
import sys
import tomllib
with open(sys.argv[1], 'rb') as stream:
    config = tomllib.load(stream)
options = shlex.split(config['hypervisor']['qemu'].get('kernel_params', ''))
api_options = [item for item in options if item.startswith('agent.guest_components_rest_api=')]
assert api_options == ['agent.guest_components_rest_api=all'], api_options
print('AA option: PASS')
PY
~~~

Expected: `AA option: PASS`. Repeat this check after Kata upgrades or operator
changes that rewrite configuration; these edits apply to newly created guests.

## 6. Install GPU Operator for confidential passthrough

Use the [GPU Operator deployment instructions](https://docs.nvidia.com/confidential-computing/containers/v1.0.0/install-quickstart)
and [CC mode semantics](https://docs.nvidia.com/confidential-computing/containers/v1.0.0/configure-cc-mode).
This dedicated worker uses one GPU in CC mode `on`.

~~~bash
kubectl label node "$NODE_NAME" nvidia.com/gpu.workload.config=vm-passthrough --overwrite
kubectl label node "$NODE_NAME" nvidia.com/cc.mode=on --overwrite
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update nvidia
helm pull nvidia/gpu-operator --version v26.3.1
sha256sum gpu-operator-v26.3.1.tgz
helm show values ./gpu-operator-v26.3.1.tgz > gpu-operator-defaults.yaml
helm template gpu-operator ./gpu-operator-v26.3.1.tgz -n gpu-operator \
  --set sandboxWorkloads.enabled=true \
  --set sandboxWorkloads.defaultWorkload=vm-passthrough \
  --set sandboxWorkloads.mode=kata \
  --set ccManager.enabled=true --set vfioManager.enabled=true \
  --set kataSandboxDevicePlugin.enabled=true \
  --set nfd.enabled=true --set nfd.nodefeaturerules=true > gpu-operator-rendered.yaml
editor gpu-operator-rendered.yaml
helm install gpu-operator ./gpu-operator-v26.3.1.tgz \
  -n gpu-operator --create-namespace --wait --timeout 10m \
  --set sandboxWorkloads.enabled=true \
  --set sandboxWorkloads.defaultWorkload=vm-passthrough \
  --set sandboxWorkloads.mode=kata \
  --set ccManager.enabled=true --set vfioManager.enabled=true \
  --set kataSandboxDevicePlugin.enabled=true \
  --set nfd.enabled=true --set nfd.nodefeaturerules=true
kubectl -n gpu-operator get pods -o wide
kubectl get node "$NODE_NAME" \
  -L nvidia.com/cc.mode.state,nvidia.com/cc.ready.state
kubectl get node "$NODE_NAME" -o jsonpath='{.status.allocatable.nvidia\.com/pgpu}{"\n"}'
lspci -nnk -d 10de:
~~~

Expected: the CC Manager, VFIO Manager, and Kata sandbox device plugin are
healthy; the worker reports `cc.mode.state=on`, `cc.ready.state=true`, one
allocatable `nvidia.com/pgpu`, and its GPU is bound to `vfio-pci`. Check the
selected worker's actual operands and ClusterPolicy using the
[README readiness checks](../README.md#check-gpu-operator-readiness), including
its driver-management guidance. The ordinary container GPU device plugin and
`nvidia.com/gpu` do not satisfy this profile.

If the mode is off/failed, operands fail, or the GPU is still host-driver bound,
inspect the relevant operand logs and the vendor firmware requirements. Do not
manually set readiness labels to make scheduling succeed. NFD must also discover
the actual SNP/TDX feature used by the RuntimeClass selector.

## 7. Run a confidential GPU workload

Use the NVIDIA [single-GPU sample](https://docs.nvidia.com/datacenter/cloud-native/confidential-containers/1.0.0/run-sample-workload.html)
as a diagnostic, with the image resolved to a digest. This is a public CUDA test,
not the encrypted NVFlare workload or proof of Trustee key release.

~~~bash
CUDA_IMAGE=nvcr.io/nvidia/k8s/cuda-sample
CUDA_DIGEST=$(skopeo inspect --format '{{.Digest}}' \
  docker://$CUDA_IMAGE:vectoradd-cuda12.5.0-ubuntu22.04)
cat > confidential-gpu-check.yaml <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: confidential-gpu-check
spec:
  runtimeClassName: "$RUNTIME_CLASS"
  nodeSelector:
    kubernetes.io/hostname: "$NODE_NAME"
  restartPolicy: Never
  containers:
    - name: vectoradd
      image: "$CUDA_IMAGE@$CUDA_DIGEST"
      resources:
        limits:
          nvidia.com/pgpu: "1"
          memory: 16Gi
EOF
kubectl apply -f confidential-gpu-check.yaml
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded \
  pod/confidential-gpu-check --timeout=10m
kubectl logs confidential-gpu-check | tee confidential-gpu-check.log
grep -F 'Test PASSED' confidential-gpu-check.log
kubectl get pod confidential-gpu-check \
  -o jsonpath='{.spec.nodeName}{" "}{.spec.runtimeClassName}{" "}{.status.containerStatuses[0].state.terminated.exitCode}{"\n"}'
sudo journalctl -k -b | grep -Ei 'SNP|TDX|SEAM'
~~~

Expected: phase `Succeeded`, `Test PASSED` in the logs, exit code `0`, and the
selected node/runtime. For TDX, confirm successful TDX module initialization in
the current boot log. If the Pod fails or times out, inspect
`kubectl describe pod confidential-gpu-check`, its logs, and
`sudo journalctl -u kubelet -u containerd`. A missing `pgpu` or runtime-selector
label is a scheduling failure; a guest boot/driver failure needs host/runtime
repair. Do not replace the confidential RuntimeClass with `runc` to pass the test.

Delete the diagnostic Pod to release the sole GPU before capture:

~~~bash
kubectl delete pod confidential-gpu-check --wait=true
~~~

## 8. Check Host A DNS and TLS from a confidential guest

Host TLS checks do not establish guest trust. With the GPU released, run a short
diagnostic using only the public CA from Host A. This reference uses the same
lab CA for both services; use two public bundles and corresponding curl paths
when the services have different issuing CAs. No publisher or Trustee
administration credentials are involved.

~~~bash
test -s "$COCO_DIR/public/ca.pem"
CURL_IMAGE=curlimages/curl
CURL_DIGEST=$(skopeo inspect --format '{{.Digest}}' docker://$CURL_IMAGE:8.23.0)
kubectl create configmap coco-public-ca --from-file=ca.pem="$COCO_DIR/public/ca.pem"
cat > confidential-service-check.yaml <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: confidential-service-check
spec:
  runtimeClassName: "$RUNTIME_CLASS"
  nodeSelector:
    kubernetes.io/hostname: "$NODE_NAME"
  restartPolicy: Never
  containers:
    - name: check
      image: "$CURL_IMAGE@$CURL_DIGEST"
      command: ["/bin/sh", "-ec"]
      args:
        - |
          curl --fail --silent --show-error --cacert /trust/ca.pem --connect-timeout 15 \\
            https://$TRUSTEE_HOST:8443/healthz >/dev/null
          status=\$(curl --silent --show-error --cacert /trust/ca.pem --connect-timeout 15 \\
            --output /dev/null --write-out '%{http_code}' https://$REGISTRY_HOST/v2/)
          test "\$status" = 401
          echo 'Guest DNS and TLS: PASS'
          aa_status=\$(curl --silent --show-error --connect-timeout 15 \\
            --output /dev/null --write-out '%{http_code}' http://127.0.0.1:8006/aa/token)
          test "\$aa_status" = 400
          echo 'Guest AA token route: PASS'
      resources:
        limits:
          nvidia.com/pgpu: "1"
          memory: 16Gi
      volumeMounts:
        - name: public-ca
          mountPath: /trust
          readOnly: true
  volumes:
    - name: public-ca
      configMap:
        name: coco-public-ca
EOF
kubectl apply -f confidential-service-check.yaml
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded \
  pod/confidential-service-check --timeout=10m
kubectl logs confidential-service-check
kubectl delete pod confidential-service-check --wait=true
kubectl delete configmap coco-public-ca
~~~

Expected: `Guest DNS and TLS: PASS`, `Guest AA token route: PASS`, and successful
Pod completion. The AA request deliberately omits `token_type`: the
[AA REST router](https://github.com/confidential-containers/guest-components/blob/v0.21.0/api-server-rest/src/router.rs)
returns `400` for that incomplete request when its token route is enabled.
Neither this route check nor the CUDA sample in step 7 exercises TDX quote
generation. For TDX, complete the Intel quote-generation check linked above.
A connection failure or `404` does not pass. No token is requested or logged;
successful token issuance is checked after platform references are registered
in the four-step workflow. Harbor's
unauthenticated `/v2/` returns its normal authentication challenge (`401`);
that is different from a TLS/DNS failure. On failure inspect the Pod events/logs,
CoreDNS records, Host A firewall, and certificate SAN/CA bundle. Keep the public
CA ConfigMap only for this diagnostic; provisioning supplies the real workload's
trust inputs through its existing flow.

Continue with [workflow tool preparation](../PREREQUISITES.md#6-prepare-capture-and-provisioning-inputs--both-hosts).
The final workload must still demonstrate successful measurement verification,
Trustee key release, NVFlare peer-proof verification, and participant registration.
