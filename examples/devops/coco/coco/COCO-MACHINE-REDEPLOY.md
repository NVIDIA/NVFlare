# CoCo machine: rebuild the adversarial Kubernetes confidential cluster

First complete [public-package configuration](../CONFIGURATION.md), including
`platform.env`. The scripts are for a fresh, dedicated host.

Run this procedure as `operator` on Ubuntu 26.04 host `coco-cluster-node`.
This operator is deliberately outside the workload trust boundary. CoCo IT
receives only the registry CA and one Pod YAML plus its independently
authenticated SHA-256 as separately delivered artifacts. The measured Pod
contains the public Trustee certificate as its guest trust anchor; public trust
material is not a secret. CoCo must never receive the registry publisher
credential, service policy fragment, image-decryption key, signing private key,
build input, plaintext image, Trustee private key, or KBS admin token.

## 1. Hardware and host prerequisites

- The selected AMD SEV-SNP or Intel TDX host stack enabled, with `/dev/kvm`
  accessible to the runtime. SNP needs `/dev/sev`; TDX needs the reviewed,
  pinned Intel QGS/SGX/collateral setup in [RUNTIME-VARIANTS.md](../RUNTIME-VARIANTS.md).
- GPU targets: one supported NVIDIA confidential GPU isolated from host drivers
  before passthrough. CPU-only targets do not need GPU/IOMMU/VFIO setup.
- Ubuntu 26.04 x86_64, swap disabled, passwordless noninteractive sudo, ample
  memory/disk, DNS, and outbound access to pinned artifact registries.
- No existing Kubernetes/containerd state when performing a clean rebuild.
- Hash-verified kit at `/home/operator/coco-it`.

Initial inspection:

```bash
cd /home/operator/coco-it
sudo apt-get update
sudo apt-get install -y ca-certificates curl pciutils python3
find . -type f -print0 | sort -z | xargs -0 sha256sum
cat /etc/os-release
uname -r
swapon --show
lscpu
ls -l /dev/kvm
# SNP only: ls -l /dev/sev
# GPU targets only: lspci -Dnnk | grep -A3 -i NVIDIA
```

Stop if an NVIDIA/Nouveau driver owns the required GPU, swap is active, the expected TEE
devices are absent, or the host/lease differs from the reviewed target.

## 2. Check the vendored bootstrap

```bash
./00-fetch-pinned-workflow.sh
```

The historical filename is retained, but this command only checks the three
included bootstrap stages. It does not clone or fetch a repository. Provenance
and local changes are recorded in `bootstrap/README.md`. Secure services are
never installed in the adversarial cluster.

## 3. Create, review, and run host preflight

Create the private configuration explicitly from the reviewed kit template:

```bash
install -d -m 0700 "$HOME/.config/coco-platform"
install -m 0600 config.env.example \
  "$HOME/.config/coco-platform/config.env"
editor "$HOME/.config/coco-platform/config.env"
```

Review `$HOME/.config/coco-platform/config.env` against the kit template and
host. Required values include Kubernetes `1.34.9-1.1`, containerd `2.2.2`,
Kata `3.29.0`, GPU Operator `v26.3.1` for GPU targets, Calico `v3.32.1`, the
selected `TEE_PLATFORM`/`RUNTIME_CLASS` pair, and
`EXPECTED_HOSTNAME` matching the intended machine. No measurement is needed
in this untrusted installer configuration. An empty `NODE_IP` means derive the default-route IPv4 address.
Do not learn the approved launch measurement from this adversarial node.
Set the same `RUNTIME_CLASS` in `platform.env`. For TDX also pin
`QGS_PACKAGE_VERSION`, `QGS_CONFIG_SHA256` and `QGS_QCNL_CONFIG_SHA256` in
bootstrap configuration after reviewing the preinstalled host stack.

After review, run preflight:

```bash
./10-run-host-preflight.sh
```

It must report zero failures. Review every warning before continuing.

## 4. Install Kubernetes and containerd

```bash
./20-install-kubernetes.sh
```

This invokes only the pinned upstream Kubernetes stage: installs and verifies
containerd/CNI/Kubernetes/Helm artifacts, configures a single-node kubeadm
control plane, installs Calico, removes the control-plane scheduling taint, and
waits for readiness. After it returns:

```bash
sudo systemctl status containerd kubelet --no-pager
sudo kubectl --kubeconfig /etc/kubernetes/admin.conf get nodes -o wide
sudo kubectl --kubeconfig /etc/kubernetes/admin.conf get pods -A -o wide
```

## 5. Receive the public runtime inputs

The provisioning node delivers `kata-deploy-3.29.0.tgz`; place it at
`$HOME/coco-it/public/kata-deploy-3.29.0.tgz`. The kit already contains
`public/kata-platform.env`, which provides the approved chart hash and runtime
image digest. No platform archive, detached signature, platform signing public
key or key-fingerprint ceremony is required for this public handoff.

The pins used by stage 35 are:

```text
Kata version: 3.29.0
Chart SHA-256: dfa752945f35e2fd2d81e5293e214b7d513ae05a5edb0b0f58b3ef67eed854c2
Runtime image: quay.io/kata-containers/kata-deploy@sha256:1e80246bbecd4fdfde2281a1ebefcd77da3f3722eca57856260d2b270de0b6ff
RuntimeClass: selected approved SNP/TDX CPU-only or NVIDIA GPU runtime
```

CoCo is the untrusted cluster operator, not the trusted system. Do not generate
or approve platform measurements here, run the trusted-system rehearsal stages,
or install CoCo-provided measurements into RVPS. Local hash checks make the
installation reproducible; they are not a security boundary against CoCo IT.
The independent service evaluates hardware attestation against its approved
references and CPU/GPU and workload policies before releasing keys.

## 6. Install the selected Kata confidential runtime

```bash
./30-install-coco-gpu.sh
./35-repin-kata-deployment.sh public/kata-deploy-3.29.0.tgz
```

This invokes only the included cluster bootstrap stage 20. The first Kata
installation already uses the hash-checked chart and immutable image digest. It installs the Kata deployment and
NVIDIA GPU Operator for confidential passthrough only on GPU targets, waits for the host/runtime
files, verifies released confidential-volume settings, and waits for the
runtime class and, for GPU targets, `nvidia.com/pgpu` capacity. CPU-only uses the
same historical stage-30 filename without installing GPU Operator. TDX runtime
setup verifies the configured QGS path. Do not install a host NVIDIA
driver for this passthrough design.

## 7. Trust the independent registry for pulls only

Authenticate `public/registry-ca.crt` with the service administrator. It is the
only standalone service artifact CoCo receives before the owner-approved Pod.
Then run:

```bash
./40-configure-registry-trust.sh
./60-verify-platform.sh
```

The script installs containerd `hosts.toml` with only `pull` and `resolve`
capabilities and no publisher credential. The final verifier must show a Ready
node, the selected runtime class, one allocatable `nvidia.com/pgpu` for GPU
targets, healthy system pods, pinned client/runtime versions, and the
registry trust files.

## 8. Receive and launch one immutable Pod handoff

Receive exactly one `RELEASE-pod.yaml` and obtain its 64-hex SHA-256 over a
separate authenticated channel. Do not accept a directory containing service
policy, keys, credentials, or build material. The service administrator must
confirm that the matching trusted-service handoff was installed first.

From an interactive terminal:

```bash
./50-launch-handoff.sh /path/to/RELEASE-pod.yaml EXPECTED_SHA256
```

The launcher checks the hash, enforces the one-container/no-volume/no-port
shape, exact registry digest, selected Kata runtime, target-specific GPU
allocation, non-root and explicitly approved read-only or writable security
context, disabled service-account token and service links,
embedded init-data, server-side dry run, and an explicit typed confirmation. It
also rejects embedded Kata policy that lacks the four official
CVE-2026-77176 workaround checks. It applies the unchanged bytes and waits for
readiness.

## 9. Verification and adversarial limits

Record only nonsecret status:

```bash
./70-verify-running-workload.sh /path/to/RELEASE-pod.yaml EXPECTED_SHA256
```

The verifier compares the live image, command, init-data, and runtime with the
authenticated handoff; requires Running, Ready, and zero restarts; and samples
application output with `kubectl logs --limit-bytes=1`. Any visible byte fails
the silent demo or silent provisioned NVFlare startup check; an explicit guest
`ReadStreamRequest` denial also passes. A negative `kubectl exec` test must
return `PermissionDenied` with `ExecProcessRequest is blocked by policy`.

NVFlare startup is redirected before kit signing on the trusted provisioning
node. Do not work around visible output by skipping this check or changing the
approved command. Rebuild and authorize a fresh release for older noisy kits.
The verifier never prints application logs and does not mistake an unrelated
Kubernetes log-access error for policy enforcement. An empty sample is not
proof that no earlier or future sensitive output reaches host-visible logs.

Do not use SSH, attach, copy, debug containers, host mounts, or altered YAML as
a success path. The operator can delete or withhold the Pod and observe
metadata, timing, resource usage, the ciphertext manifest/layers, the public
embedded policy, and Kubernetes logs. It must not learn plaintext image layers
or receive decryption/signing keys. Do not interpret these requirements as a
guarantee that every YAML change fails attestation or prevents KBS release:

- Changing the decoded, approved InitData changes its attested digest and must
  fail release authorization. KBS checks policy-derived image/command claims
  and the InitData binding, not a measurement of the actual decrypted image or
  the host's live container-creation request.
- With unchanged InitData, an effective command outside the approved guest
  rules must be rejected by the guest agent. Attestation and KBS authorization
  may already have passed for the original policy; guest-request denial does
  not imply that no resources were released inside the confidential guest.
- Image signature, digest/integrity, decryption and guest-policy checks still
  apply. However, exact-image-digest authorization has an unresolved coverage
  limitation: another image signed by the accepted key in the permitted
  repository is not shown to be rejected solely because its digest differs.
  It must still satisfy the other checks. This is not a demonstrated successful
  substitution or key-extraction attack.
- A different RuntimeClass name that produces the same approved measured guest
  launch can pass. Independent attestation authorizes covered guest inputs,
  not the hostile host's runtime name or every literal YAML field.

Retain the independent digest, signature and release checks. Trusted deployment
acceptance must test changed InitData, forbidden effective commands and
unsigned/corrupt images at the enforcing service/guest boundary. Before claiming
all image substitutions are blocked, close the exact-image binding gap and test
an actual accepted-key/same-repository replacement with unchanged InitData.
Changing only policy claims, checking the handoff hash, or passing stage 70 does
not establish that binding. See the canonical
[image authorization analysis](https://nvflare.readthedocs.io/en/2.9/user_guide/confidential_computing/coco_security_architecture.html#image-changes-and-the-current-authorization-limitation).

The service administrator, not CoCo IT, confirms CPU/GPU appraisal and KBS
resource release. The workload owner confirms application success over its own
mTLS channel using the [trusted federation verification procedure](../provision/VERIFY-RUNNING-FEDERATION.md).
CoCo-controlled Pod status and the stage-70 result alone do not establish
NVFlare registration, peer attestation, or application readiness.

## 10. Maintenance

See [maintenance boundaries](../trusted_system/TEARDOWN.md). Destructive
lab-specific teardown scripts are not shipped in this public package.
