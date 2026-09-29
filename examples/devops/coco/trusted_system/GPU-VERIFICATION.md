# Verify GPU availability inside a confidential Pod

GPU availability and driver-reported CC mode do not prove successful GPU
cryptographic appraisal. Run this optional diagnostic only on the trusted
system, never on an adversarial cluster or with a production workload image.

| Runtime | GPU diagnostic |
|---|---|
| `kata-qemu-snp` | Skip: CPU-only deployment |
| `kata-qemu-tdx` | Skip: CPU-only deployment |
| `kata-qemu-nvidia-gpu-snp` | Supported: one passthrough GPU |
| `kata-qemu-nvidia-gpu-tdx` | Supported: one passthrough GPU |

CPU-only deployments do not need NVIDIA hardware, drivers, device plugins or
this diagnostic. Calling the helper with a CPU-only or unknown runtime returns
an error before creating an output directory or running GCC, Docker or kubectl.

Run on the trusted-system host after the rehearsal has succeeded:

```bash
cd /home/operator/coco_deployment
source /home/operator/private-platform-reference/platform-reference.env
PROFILE="$PLATFORM_WORK_ROOT/$PLATFORM_PROFILE"
case "$RUNTIME_CLASS" in
  kata-qemu-nvidia-gpu-snp|kata-qemu-nvidia-gpu-tdx)
    python3 trusted_system/verify-gpu-in-pod.py "$PROFILE" ;;
  kata-qemu-snp|kata-qemu-tdx)
    echo 'CPU-only profile: skip the optional GPU diagnostic' ;;
  *) echo "Unsupported RuntimeClass: $RUNTIME_CLASS" >&2; exit 1 ;;
esac
```

The optional helper uses the retained collector image and temporary-registry
data from stage 07. Its temporary TLS certificate must still be valid (the
rehearsal generates a two-day certificate). Both SNP and TDX stage 07 retain
`rehearsal-collector-build/collector-pod.yaml`, `registry-tls/` and
`registry-data/`. Keep their TLS files, local containerd image cache and registry
contents on the same trusted host. The helper preserves the collector's image
reference, `imagePullPolicy: Never`, runtime, resources and TLS InitData; in
particular, the TDX collector's image remains digest-pinned.

It needs host GCC, Docker, curl, kubectl and Python/PyYAML, installed by the
bootstrap procedure, plus access to the administrative kubeconfig (using sudo
when needed). It compiles
`gpu-probe.c` without CUDA toolkit headers; the Pod loads its injected CUDA
driver library at runtime. No production workload image is changed.

Both retained collector images use an Ubuntu base that provides `/bin/bash`.
The diagnostic overrides the collector entrypoint: it does not execute
`snpguest` or the TDX evidence-collection Python program. The selected GPU
Kata runtime must inject `nvidia-smi` and `libcuda.so.1` into the container;
missing tools or driver libraries cause the diagnostic to fail. It does not
install a different runtime, guest driver or workload dependency to make the
test pass.

The Pod retains the profile's selected GPU runtime and requests one `nvidia.com/pgpu`.
It runs as root inside the container, but is explicitly non-privileged, has
privilege escalation disabled and uses no hostPath or host namespace.
The helper mounts its diagnostic executable read-only using a ConfigMap.
Its trusted collector InitData uses the temporary registry CA and an offline
KBC, not production KBS resources or the workload's restrictive agent policy.
This permits diagnostic command/log access only for this disposable trusted
rehearsal. Do not loosen a production Pod policy to run this helper.

Inside that Pod the test:

1. Lists NVIDIA devices and runs `nvidia-smi -L`, a device query and `nvidia-smi -q`.
2. Loads `libcuda.so.1` and initializes the CUDA driver.
3. Requires exactly one CUDA device and creates a context.
4. Allocates device memory, fills it, copies it to the host buffer inside the
   container, and checks every returned byte before releasing the allocation.

A successful test demonstrates in-Pod driver/device access and basic CUDA
memory operations, not application training performance or GPU attestation.
The helper does not request NRAS appraisal or Trustee decryption-key release.
It does not generate or approve platform references, prove CPU TCB compliance,
or replace AS GPU appraisal; GPU targets still require both `cpu0` and `gpu0`
to pass. CPU-only targets require the CPU appraisal without a `gpu0` submodule.
This optional test never changes `platform-reference-values.json`, the
approved launch profile, or any attestation/release policy.

## Private evidence

Output is retained beneath `$PROFILE/gpu-verification-<random-id>/`:
`gpu-check.log`, `pod.yaml`, `pod-result.json`, `pod-describe.txt` and
`cuda-probe`. The helper removes only its own temporary namespace and registry
container. Failure is an error, not approval of GPU availability.
