# Vendored cluster bootstrap

Stages 00, 10 and 20 and their helpers originate from NVFlare commit
`54452740e68bd776343c4d0b8883459e0ffa9cd7`, `examples/devops/CoCo/`.
They are included here so no separate repository checkout is needed.

The Kubernetes implementation includes the trusted-system bootstrap fixes:
explicit containerd binary location, corrected Helm version comparison,
cri-tools installation and private kubeadm initialization logs. Shared helpers
are reduced to cluster installation, require an explicit target hostname,
validate network configuration and prohibit checksum bypass. This package
supports AMD SNP plus NVIDIA confidential GPU; TDX setup is not included.

The first Kata installation uses the separately received, hash-checked public
chart and the immutable amd64 image digest from `../public/kata-platform.env`.
It does not briefly install a mutable runtime image before stage 35. No secure
services, workload signing or reference approval occurs in these stages.

Invoke through the role's 10/20/30 wrappers, not directly. Cluster configuration
is copied from `../config.env.example` and reviewed on the intended host.
