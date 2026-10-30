# Trusted-system Kubernetes bootstrap

The bootstrap implementation originates from NVFlare commit
`54452740e68bd776343c4d0b8883459e0ffa9cd7`, `examples/devops/CoCo/`.
The implementation is maintained in [shared/](../../shared/README.md), with thin
role entry points here. Assemble a role kit before standalone transfer; packaging
materializes the helper, Kubernetes installer and template. Only the Kubernetes
installer is run. No NVFlare checkout, workload deployment,
Trustee, or persistent registry installation is performed here.

The installer has local corrections: check the exact `/usr/local/bin/containerd`
used by its systemd unit (not a distribution binary elsewhere on PATH), and
match the Helm version without accidentally adding a second `v` prefix.
It also suppresses printing the cluster bootstrap token during initialization.
It installs `cri-tools` so the rehearsal can associate the actual QEMU process
with the correct Kubernetes Pod sandbox through `crictl`.

Prepare `config.env` from the example, set the intended `EXPECTED_HOSTNAME`,
and select the approved `RUNTIME_CLASS` using the
[four-runtime guide](../../RUNTIME-VARIANTS.md). Set `TEE_PLATFORM=snp` for
either SNP runtime or `TEE_PLATFORM=tdx` for either TDX runtime. GPU count is
derived from the runtime: zero for CPU-only, one for a GPU target. TDX also
requires the reviewed QGS installation and configuration pins described there.
Invoke from this bootstrap directory:

```bash
bash ../02-install-kubernetes.sh
```

Kata installation is performed separately using the trusted profile's pinned
artifacts. GPU Operator is installed only for GPU targets; CPU-only rehearsal
does not require NVIDIA hardware or GPU passthrough. Make bootstrap changes in
the shared source.

The shared configuration loader is reduced to cluster-only inputs and rejects
checksum bypass. The host-specific live configuration is not packaged.
