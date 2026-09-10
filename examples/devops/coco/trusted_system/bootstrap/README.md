# Trusted-system Kubernetes bootstrap

The bootstrap implementation originates from NVFlare commit
`54452740e68bd776343c4d0b8883459e0ffa9cd7`, `examples/devops/CoCo/`.
Only the Kubernetes installer is run. No NVFlare checkout, workload deployment,
Trustee, or persistent registry installation is performed here.

The installer has local corrections: check the exact `/usr/local/bin/containerd`
used by its systemd unit (not a distribution binary elsewhere on PATH), and
match the Helm version without accidentally adding a second `v` prefix.
It also suppresses printing the cluster bootstrap token during initialization.
It installs `cri-tools` so the rehearsal can associate the actual QEMU process
with the correct Kubernetes Pod sandbox through `crictl`.

Prepare `config.env` from the example, set the intended `EXPECTED_HOSTNAME`,
keep `TEE_PLATFORM=snp`, and invoke from this bootstrap directory:

```bash
bash ../02-install-kubernetes.sh
```

Kata and GPU Operator installation are performed separately using the trusted
profile's pinned artifacts. Preserve the source commit when changing this copy.

The shared configuration loader is reduced to cluster-only inputs and rejects
checksum bypass. The host-specific live configuration is not packaged.
