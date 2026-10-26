# NVFlare on an existing CoCo-ready cluster

This example starts with Kubernetes, Kata confidential runtimes, NVIDIA GPU
Operator for GPU workloads, and configured Trustee KBS/AS/RVPS already running.
It covers measurements, Trustee values, NVFlare images, and Pod launch.

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

## Select the runtime

| CPU | GPU | `cpu_tee` | `gpu_tee` | RuntimeClass | Allocation |
| --- | --- | --- | --- | --- | --- |
| SNP | None | `amd_sev_snp` | `none` | `kata-qemu-snp` | None |
| SNP | NVIDIA | `amd_sev_snp` | `nvidia_cc` | `kata-qemu-nvidia-gpu-snp` | 1 `nvidia.com/pgpu` |
| TDX | None | `intel_tdx` | `none` | `kata-qemu-tdx` | None |
| TDX | NVIDIA | `intel_tdx` | `nvidia_cc` | `kata-qemu-nvidia-gpu-tdx` | 1 `nvidia.com/pgpu` |

The guest must provide its local AA token API at
`http://127.0.0.1:8006/aa/token`. GPU profiles allocate one `nvidia.com/pgpu`
per Pod and require successful CPU and GPU appraisal. CPU-only profiles need
no GPU Operator. TDX remains experimental pending complete hardware acceptance.

## Workflow

1. [Capture measurements](MEASUREMENTS.md) from the installed runtime to obtain
   reference values and the NVFlare launch profile.
2. [Set Trustee values](TRUSTEE.md) and read them back.
3. [Provision NVFlare](provision/README.md) to build encrypted images and Pod YAML.
4. Register each generated release with Trustee, then [launch its Pod](LAUNCH.md).

`nvflare provision` generates the Trustee release handoff; it does not install
that handoff automatically. The received files are used directly. Keep image
keys and credentials with their intended trusted recipients.

For the security model and peer-proof protocol, see the
[CoCo security architecture](https://nvflare.readthedocs.io/en/2.9/user_guide/confidential_computing/coco_security_architecture.html).
