# Verify GPU availability inside a confidential Pod

GPU availability and driver-reported CC mode do not prove successful GPU
cryptographic appraisal. Run this diagnostic only on the trusted system.

Run on the trusted trusted_system host after the rehearsal has succeeded:

```bash
cd /home/operator/coco_deployment
source /home/operator/private-platform-reference/platform-reference.env
PROFILE="$PLATFORM_WORK_ROOT/$PLATFORM_PROFILE"
python3 trusted_system/verify-gpu-in-pod.py "$PROFILE"
```

The optional helper uses the retained collector image and temporary-registry
data from stage 07. Its temporary TLS certificate must still be valid (the
rehearsal generates a two-day certificate). It needs host GCC, Docker,
kubectl and Python/PyYAML, installed by the bootstrap procedure. It compiles
`gpu-probe.c` without CUDA toolkit headers; the Pod loads its injected CUDA
driver library at runtime. No production workload image is changed.

The Pod uses `kata-qemu-nvidia-gpu-snp` and requests one `nvidia.com/pgpu`.
It runs as root inside the container, but is explicitly non-privileged, has
privilege escalation disabled and uses no hostPath or host namespace.
The helper mounts its diagnostic executable read-only using a ConfigMap.

Inside that Pod the test:

1. Lists NVIDIA devices and runs `nvidia-smi -L`, a device query and `nvidia-smi -q`.
2. Loads `libcuda.so.1` and initializes the CUDA driver.
3. Requires exactly one CUDA device and creates a context.
4. Allocates device memory, fills it, copies it to the host buffer inside the
   container, and checks every returned byte before releasing the allocation.

A successful test demonstrates in-Pod driver/device access and basic CUDA
memory operations, not application training performance or GPU attestation.
No NRAS request or Trustee decryption-key release is performed. This optional
test never changes `platform-reference-values.json`.

## Private evidence

Output is retained beneath `$PROFILE/gpu-verification-<random-id>/`:
`gpu-check.log`, `pod.yaml`, `pod-result.json`, `pod-describe.txt` and
`cuda-probe`. The helper removes only its own temporary namespace and registry
container. Failure is an error, not approval of GPU availability.
