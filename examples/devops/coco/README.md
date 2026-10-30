# NVFlare on an existing CoCo-ready cluster

**Complete and verify these prerequisites before starting the four-step workflow:**

This optional [beginner prerequisite appendix](PREREQUISITES.md) helps beginners
bring up the prerequisite systems for the NVFlare CoCo workflow. Consult the
official documentation for each component and the version you deploy. If there
is any discrepancy, the component's official documentation takes precedence.

- A running, configured **Trustee service (KBS/AS/RVPS)**. Follow the
  [official Trustee installation guide](https://confidentialcontainers.org/docs/attestation/installation/).
- A **CoCo-ready Kubernetes cluster with the supported Kata confidential runtime**.
  GPU workloads require the Kata confidential GPU runtime and NVIDIA GPU Operator;
  follow the [official NVIDIA CoCo/Kata GPU deployment guide (v1.0)](https://docs.nvidia.com/confidential-computing/containers/v1.0.0/confidential-containers-deploy).
  CPU-only profiles do not require the GPU components.

Use the component versions required by the supported NVFlare profile. This
example then covers measurements, Trustee values, NVFlare images, and Pod launch
using those existing services.

Run commands from `examples/devops/coco` in an NVFlare source checkout,
except where a guide explicitly changes directory. The helper modules use that
checkout or an installed NVFlare package.

## Existing inputs

- The supported Kata 3.29.0 runtime profile, access to the selected worker,
  and permission to launch a measurement collector.
- Trustee HTTPS endpoint, its TLS CA, AS signing public key, and administration
  credentials. Its configured appraisal policies must support the selected TEE
  and the reference values described in [Trustee configuration](TRUSTEE.md).
- An existing image registry, a CA certificate/trust bundle for guest/tool TLS,
  and private publisher credentials.
- NVFlare provisioning and the image/policy tools listed in
  [provisioning](provision/README.md).

Before starting, check these operational dependencies:

- Synchronize clocks on workers, Trustee, and NVFlare participants; certificate
  validity, collateral expiry, and short-lived attestation proofs depend on time.
- Ensure hosts and confidential guests can resolve the configured registry and
  Trustee DNS names, reach their HTTPS endpoints, and verify their certificates.
- Allow the configured verifiers to retrieve AMD KDS/VCEK material for SNP,
  Intel PCS collateral for TDX, and NVIDIA NRAS/RIM material for GPU appraisal,
  through the deployment's existing network/cache configuration.
- Confirm the existing CPU/GPU appraisal policies consume the selected reference
  values, and the guest-local AA token API supports the NVFlare proof flow.

These public trust inputs serve different purposes:

| Input | Purpose |
| --- | --- |
| Registry CA/bundle | Verify registry TLS for image publication and pulls. |
| Trustee TLS CA/bundle | Verify the Trustee HTTPS endpoint. |
| AS signing public key | Verify AS-signed attestation results (EARs) carried by NVFlare peer proofs. |

## Select the runtime

| CPU | GPU | Participant `cpu_tee` | Participant `gpu_tee` | RuntimeClass | Allocation |
| --- | --- | --- | --- | --- | --- |
| SNP | None | `amd_sev_snp` | `none` | `kata-qemu-snp` | None |
| SNP | NVIDIA | `amd_sev_snp` | `nvidia_cc` | `kata-qemu-nvidia-gpu-snp` | 1 `nvidia.com/pgpu` |
| TDX | None | `intel_tdx` | `none` | `kata-qemu-tdx` | None |
| TDX | NVIDIA | `intel_tdx` | `nvidia_cc` | `kata-qemu-nvidia-gpu-tdx` | 1 `nvidia.com/pgpu` |

The table uses participant configuration values. For constraints on signed EAR
evidence, `cpu_tee` instead uses `snp` or `tdx`.

The guest must provide its local AA token API at
`http://127.0.0.1:8006/aa/token`. GPU profiles allocate one `nvidia.com/pgpu`
per Pod and require successful CPU and GPU appraisal. CPU-only profiles need
no GPU Operator.

## Check GPU Operator readiness

Skip this section for CPU-only profiles. The platform owner runs these read-only
checks before capture and repeats them after GPU Operator upgrades or changes to
its operands. Replace the names below with the existing policy, namespace, and
selected worker; access must permit reading ClusterPolicies, Nodes, and operands.

~~~bash
NODE_NAME=existing-worker
CLUSTER_POLICY=existing-policy
GPU_OPERATOR_NAMESPACE=existing-namespace
kubectl get clusterpolicies.nvidia.com "$CLUSTER_POLICY" -o yaml
kubectl -n "$GPU_OPERATOR_NAMESPACE" get deployments,daemonsets -o wide
kubectl -n "$GPU_OPERATOR_NAMESPACE" get pods -o wide \
  --field-selector "spec.nodeName=$NODE_NAME"
kubectl get node "$NODE_NAME" \
  -L nvidia.com/gpu.workload.config,nvidia.com/cc.mode,nvidia.com/cc.mode.state,nvidia.com/cc.ready.state
kubectl get node "$NODE_NAME" -o jsonpath='{.status.allocatable.nvidia\.com/pgpu}{"\n"}'
~~~

Check the effective configuration and selected worker against the
[GPU Operator ClusterPolicy fields](https://github.com/NVIDIA/gpu-operator/blob/v26.3.1/api/nvidia/v1/clusterpolicy_types.go):

| Check | Required state |
| --- | --- |
| `spec.ccManager.enabled`, `spec.vfioManager.enabled`, `spec.kataSandboxDevicePlugin.enabled` | Enabled, with the corresponding selected-worker operands ready. Check the Kata sandbox device plugin, not the ordinary container GPU device plugin. |
| `spec.sandboxWorkloads` | `enabled: true`, `mode: kata`; the selected worker's `nvidia.com/gpu.workload.config` override or `defaultWorkload` must select `vm-passthrough`. |
| Worker and CC labels | Node is Ready; `nvidia.com/cc.ready.state=true`; `nvidia.com/cc.mode.state` matches the intended enabled mode from the node override or `ccManager.defaultMode`. |
| GPU allocation | At least one allocatable `nvidia.com/pgpu` for this example's one-GPU Pod. |

For CC mode values, use the installed release's
[mode-label semantics](https://docs.nvidia.com/confidential-computing/containers/configure-cc-mode):
`on` is the normal single-GPU mode; `ppcie` requires a Hopper PPCIE
configuration. Missing/false readiness, `off`, `failed`, or an
incomplete mode transition requires platform-owner repair before capture.
These labels indicate operational readiness, not attestation.

On the selected worker, inspect PCI binding with `lspci -nnk -d 10de:`
(`lspci` comes from pciutils). The GPUs intended for passthrough must use
`vfio-pci`; an NVIDIA host driver must not own them. Review the existing
`spec.driver.enabled` setting against the platform's driver-management plan.
The [preinstalled-driver option](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/26.3/getting-started.html#pre-installed-nvidia-gpu-drivers)
uses `driver.enabled=false` when driver installation is managed outside the
Operator, including an applicable Ubuntu 26.04 host-managed-driver deployment.
This is conditional, not an instruction to install NVIDIA host drivers on
passthrough GPUs or to disable driver management solely because of the OS.

Keep the supported version combination explicit:
[NVIDIA CoCo v1.0](https://docs.nvidia.com/confidential-computing/containers/v1.0.0/supported-platforms)
lists Kata 3.29 and Ubuntu 25.10;
[v1.1](https://docs.nvidia.com/confidential-computing/containers/v1.1.0/supported-platforms)
lists Kata 4.0 and adds Ubuntu 26.04. This example uses the pinned
Kata/genpolicy 3.29.0 profile.

For new-host BIOS, CPU TEE, firmware, and GPU preparation, follow the optional
[beginner prerequisite appendix](PREREQUISITES.md). It helps beginners bring up
the prerequisite systems; the official documentation for each component and
version takes precedence if there is any discrepancy. Complete host preparation
before starting this workflow.

## Workflow

1. [Capture measurements](MEASUREMENTS.md) from the installed runtime to obtain
   reference values and the NVFlare launch profile.
2. [Set Trustee values](TRUSTEE.md) and read them back.
3. [Provision NVFlare](provision/README.md) to build encrypted images and Pod YAML.
4. Register each generated release with Trustee, then [launch its Pod](LAUNCH.md).

| Operator | Input | Output/destination |
| --- | --- | --- |
| Platform owner | Existing worker and capture configuration | Reference values to Trustee admin; launch profile to provisioner. |
| Trustee admin | Reference values and per-release handoff | Configured references, image keys, and release policies before launch. |
| NVFlare provisioner | Project/application inputs and launch profile | Encrypted images to registry; private handoff to Trustee admin; Pod YAML to cluster operator. |
| Cluster operator | Generated Pod YAML | Running confidential Pod. |

`nvflare provision` generates the Trustee release handoff; it does not install
that handoff automatically. The received files are used directly. Keep image
keys and credentials with their intended trusted recipients. The direct handoff
workflow assumes recipients supply the intended files; bundled checksums check
consistency, not sender identity. See the [Trustee handoff checks](TRUSTEE.md#optional-handoff-digest-check)
and [Pod digest check](LAUNCH.md#optional-pod-digest-check) for optional inputs.

For the security model and peer-proof protocol, see the
[CoCo security architecture](https://nvflare.readthedocs.io/en/2.9/user_guide/confidential_computing/coco_security_architecture.html).

For upstream platform, service, and tool documentation, see the CC user guide's
[official component references](https://nvflare.readthedocs.io/en/2.9/user_guide/confidential_computing/getting_started.html#cc-official-references).

## Developer package check

`python3 validate-package.py` checks public source files in a Git checkout using
Python 3.11+ and Git. It is a developer check, not a deployment step, and does not
support archive or non-Git copies. Ignored deployment inputs remain outside it.
