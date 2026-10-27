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
no GPU Operator. TDX remains experimental pending complete hardware acceptance.

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
