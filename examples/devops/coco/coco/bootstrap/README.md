# Vendored cluster bootstrap

Stages 00, 10 and 20 and their helpers originate from NVFlare commit
`54452740e68bd776343c4d0b8883459e0ffa9cd7`, `examples/devops/CoCo/`.
The shared implementation is maintained in [shared/](../../shared/README.md).
This source directory contains role entry points. Assemble role kits using the
main README before transferring this role alone; packaging materializes its
helpers, Kubernetes installer and template. No separate repository checkout is
needed on the destination.

The Kubernetes implementation includes the trusted-system bootstrap fixes:
explicit containerd binary location, corrected Helm version comparison,
cri-tools installation and private kubeadm initialization logs. Shared helpers
are reduced to cluster installation, require an explicit target hostname,
validate network configuration and prohibit checksum bypass. This package
selects AMD SNP or Intel TDX, with optional NVIDIA confidential GPU passthrough.
TDX firmware, a compatible host kernel, Intel platform provisioning and a pinned
QGS installation are external prerequisites; preflight checks them instead of
silently installing or trusting an unreviewed quote service. See the
[target prerequisites](../../RUNTIME-VARIANTS.md).

The first Kata installation uses the separately received, hash-checked public
chart and the immutable amd64 image digest from `../public/kata-platform.env`.
It does not briefly install a mutable runtime image before stage 35. No secure
services, workload signing or reference approval occurs in these stages.

Invoke through the role's 10/20/30 wrappers, not directly. Cluster configuration
is copied from `../config.env.example` and reviewed on the intended host.
