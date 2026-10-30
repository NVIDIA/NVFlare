# Appendix: bring up the prerequisite systems

This optional appendix helps beginners bring up the prerequisite systems for the
NVFlare CoCo workflow. Consult the official documentation for each component and
the version you deploy. If there is any discrepancy, the component's official
documentation takes precedence.

Start with fresh **Ubuntu 26.04 LTS, x86-64** installations. This is a small,
non-HA learning deployment. It uses two different machines:

| Machine | Responsibilities |
| --- | --- |
| **Host A: trusted services** | Trustee KBS/AS/RVPS, HTTPS registry, image building, and NVFlare provisioning. A VM or physical machine is suitable. |
| **Host B: confidential GPU worker** | Kubernetes control plane and worker, containerd, Kata, and GPU Operator. Use a physical machine with a supported CPU and one supported confidential GPU. |

Trustee runs on **Host A, outside Kubernetes**. Do not install it on Host B.
Larger deployments can separate the registry, builder, Trustee, control plane,
and workers across more machines; they need corresponding network, credential,
and availability configuration beyond this example.

## Compatibility status

**This Ubuntu 26.04/Kata 3.29 combination has not been qualified as a complete
fresh-host reference deployment.** The instructions are a preparation recipe,
not a claim of successful SNP/TDX hardware acceptance. Stop at a failed check;
do not silently substitute versions to make a command pass. TDX remains
experimental in the NVFlare 2.9 CoCo workflow.

The [NVIDIA v1.0 supported-platform matrix](https://docs.nvidia.com/confidential-computing/containers/v1.0.0/supported-platforms)
lists Kata 3.29 with Ubuntu 25.10. The [v1.1 matrix](https://docs.nvidia.com/confidential-computing/containers/v1.1.0/supported-platforms)
adds Ubuntu 26.04 with Kata 4.0.0. Neither matrix validates the combination in
this appendix. NVFlare 2.9 requires the matching Kata/genpolicy 3.29.0 profile;
support for newer Kata/genpolicy is intended for a future NVFlare upgrade.

| Component | Reference version |
| --- | --- |
| Kata runtime, genpolicy, rules, settings | 3.29.0 |
| NVIDIA GPU Operator | 26.3.1 |
| Trustee | v0.22.0 |
| Kubernetes | 1.36.5 |
| containerd | 2.3.7 |
| Flannel | v0.28.10 |
| Harbor | v2.14.6 |
| Helm | v3.19.0 |
| Cosign | v2.6.1 |

Use versioned artifacts and verify their published checksums before installing
them. The commands pin standalone images by digest and chart-managed components
by their chart release. Keep a private record of the resolved artifacts and
rendered chart configuration. Existing compatible services can be reused when
the checks below pass.

## 1. Gather inputs — both hosts

Obtain these inputs before installing Kubernetes:

- The supported CPU/GPU models and the server vendor's BIOS, firmware, and
  CPU-TEE enablement instructions. The simplified worker has **one GPU**;
  multi-GPU/NVSwitch/PPCIE deployments need their own qualified configuration.
- Stable, mutually reachable host addresses, SSH access, and an operator account
  with sudo privileges on each host. The trusted operator must be permitted to
  inspect runtime artifacts and QEMU processes and launch diagnostic collectors.
- DNS names for the registry and Trustee. The example uses `registry.coco.lab`
  and `trustee.coco.lab`; replace them with names you control. Do not use IP
  literals as registry endpoints. Use the same `host[:port]` spelling throughout
  registry authentication, image references, and policy configuration.
- For SNP, all four minimum reported-TCB values from an independently maintained
  vendor/platform security baseline: bootloader, TEE, SNP firmware, and microcode.
  Reading the worker's current firmware does not establish an independent
  baseline. Do not fill unknown minima with zeros just to pass capture.
- Internet or approved proxy/cache access for package/image downloads and AMD
  KDS, Intel PCS, and NVIDIA NRAS/RIM collateral. Both machines and confidential
  guests need the applicable HTTPS access.

On **each host**, create an input file, replace both address placeholders, and
select `snp` or `tdx`. Use the same values on both machines; the paths below are
local to each machine and contain no shared credentials.

~~~bash
umask 077
mkdir -p "$HOME/coco-private"
cat > "$HOME/coco-private/inputs.env" <<'EOF'
export SERVICES_IP='REPLACE_WITH_HOST_A_ADDRESS'
export WORKER_IP='REPLACE_WITH_HOST_B_ADDRESS'
export NODE_NAME='coco-worker'
export REGISTRY_HOST='registry.coco.lab'
export TRUSTEE_HOST='trustee.coco.lab'
export REGISTRY="$REGISTRY_HOST"
export KBS_URL="https://$TRUSTEE_HOST:8443"
export CPU_TEE='snp'
export COCO_DIR="$HOME/coco-prerequisites"
export COCO_ROOT="$HOME/NVFlare/examples/devops/coco"
export KATA_VERSION='3.29.0'
EOF
editor "$HOME/coco-private/inputs.env"
source "$HOME/coco-private/inputs.env"
mkdir -p "$COCO_DIR/public" "$COCO_DIR/private" "$COCO_DIR/downloads"
~~~

Source that file again in every new terminal. Commands in the linked sections
use these variables. Paths beginning with `$HOME` refer to the operator's home
on the machine identified by that section, not to a network-mounted directory.

Reserve Pod CIDR `10.244.0.0/16` and service CIDR `10.96.0.0/12`; compare them
with your LAN, VPN, and routed networks. If they overlap, choose different
nonoverlapping CIDRs and update the kubeadm/Flannel configuration and later
NVFlare Kubernetes service input together.

Allow SSH for administration, Host A TCP **443** for Harbor and **8443** for
Trustee from the worker and guest networks, and the Kubernetes/CNI traffic
specified in the [official kubeadm port requirements](https://kubernetes.io/docs/reference/networking/ports-and-protocols/).
Keep registry databases, internal registry listeners, and Trustee backing
storage private. Docker-published ports need the
[Docker firewall rules](https://docs.docker.com/engine/network/packet-filtering-firewalls/);
host UFW rules alone do not restrict them.

**Check — both hosts:**

~~~bash
. /etc/os-release
test "$ID:$VERSION_ID" = ubuntu:26.04
test "$(uname -m)" = x86_64
case "$CPU_TEE" in snp|tdx) ;; *) exit 1 ;; esac
case "$SERVICES_IP:$WORKER_IP" in *REPLACE*) exit 1 ;; esac
test "$SERVICES_IP" != "$WORKER_IP"
ip -brief address
ip route
timedatectl status
~~~

Expected: all tests succeed, each address belongs to the intended machine,
routes do not overlap the chosen cluster ranges, and the clock is synchronized.
Fix addresses, routing, or time service before continuing. Hardware support is
checked separately; an OS/version check is not confidential-computing acceptance.

## 2. Install the common tools — both hosts

Follow [common tools and workflow inputs](appendix/TOOLS.md#common-tools--both-hosts).
This installs the source checkout and Python environment, Skopeo, and the tools
needed by later checks. Helm is installed on Host B and Cosign on Host A.

**Check:** complete that section's version and dependency checks before using
the following commands. Keep Docker installation confined to Host A.

## 3. Prepare confidential hardware — Host B

Follow [host qualification and TEE preparation](appendix/HOST_CLUSTER.md).
Complete its hardware/firmware section, reboot, and pass its post-reboot checks
**before** installing Kubernetes. Use the branch for the actual CPU vendor.

**Check:** TEE initialization, KVM access, IOMMU isolation, and the GPU binding
requirements must all pass. CPU feature flags alone are insufficient.

## 4. Bring up the registry and Trustee — Host A

Follow [DNS, TLS, registry, and Trustee](appendix/SERVICES.md). It creates public
trust inputs separately from private service keys and administrator/publisher
credentials. Only the public inputs are copied to Host B.

**Check:** require registry push/pull/signature checks, authenticated Trustee
administration, denial without credentials, active policy checks, and persistent
state across restart. A successful health endpoint alone is insufficient.

## 5. Install Kubernetes and the confidential runtime — Host B

Continue [the cluster and runtime instructions](appendix/HOST_CLUSTER.md) after
the completed firmware section. Configure guest-visible DNS as well as host
DNS. An `/etc/hosts` entry on the worker alone does not resolve service names
inside Kata guests.

**Check:** complete each containerd, Kubernetes, CNI, Kata, and GPU Operator check
in order, then run the confidential GPU sample. Require successful guest
execution and GPU access, not just a Ready Pod.

## 6. Prepare capture and provisioning inputs — both hosts

Continue [workflow tools and configuration mapping](appendix/TOOLS.md#prepare-the-workflow-inputs).
Host A receives compatible genpolicy artifacts and publisher credentials; Host B
receives collector metadata, public trust, and its evidence verifier. Set the
capture configuration from the inspected worker, including the independent SNP
minimum-TCB values when applicable.

**Check:** run the toolchain and configuration checks there. Do not copy the
Trustee administrator token, TLS/AS signing private keys, registry credentials,
or application-signing private keys to Host B.

## 7. Return to the four-step workflow

Before starting, confirm the following:

- Both machines and guests resolve and reach the intended registry and Trustee
  endpoints and verify TLS. Existing system trust may suffice for host tools;
  the workflow still needs actual public bundle files to embed guest trust.
- Host A can publish images and signatures; Host B can pull the same images
  anonymously by digest. Publishing credentials stay on Host A.
- Trustee has the CPU/GPU appraisal policies, authenticated administration,
  persistent AS public key, and a current resource-policy export. It need not
  have platform references or workload keys yet.
- The selected Host B RuntimeClass, AA token API, confidential GPU sample, and
  capture/verifier tool checks pass. The worker is under trusted operator
  control during measurement capture.
- The builder has the matching Kata/genpolicy 3.29.0 artifacts. Public trust
  files and private credential files are mapped to the configuration inputs.

Now follow the [four-step workflow](README.md#workflow):

1. [Capture platform measurements](MEASUREMENTS.md) on Host B.
2. [Register platform references](TRUSTEE.md) from Host A.
3. [Provision NVFlare images and Pod YAML](provision/README.md) on Host A.
4. Register each generated Trustee handoff on Host A, then
   [launch the generated Pod](LAUNCH.md) on Host B.

`nvflare provision` does not register its Trustee handoff automatically. After
launch, successful key release, NVFlare proof verification, and participant
registration establish that the workflow completed; Pod readiness alone does
not.
