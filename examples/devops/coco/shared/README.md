# Authoritative shared deployment code

Maintain these implementations here, not in role copies:

- `validate-config.sh`: hostname, endpoint and private-directory checks for all four roles.
- `bootstrap/lib/common.sh`: cluster-only configuration, download and bootstrap helpers.
- `bootstrap/10-install-kubernetes.sh`: Kubernetes/containerd/CNI installation for CoCo and the trusted platform.
- `bootstrap/templates/kubeadm.yaml.in`: their common kubeadm template.

The bootstrap originates from NVFlare commit
`54452740e68bd776343c4d0b8883459e0ffa9cd7`, `examples/devops/CoCo/`, with the
local hardening described in the role bootstrap guides. Role entry points set
their own bootstrap directory; configuration, state and trust roles remain
separate. No shared helper installs secure services or approves references.

Run role entry points, not shared scripts directly. Source wrappers resolve
this directory. From a clean Git checkout, `python3 role_kits.py /path/to/new-output` replaces wrappers
with copies from these sources and adds the template to each generated kit.
`validate-package.py --assembled` checks every generated dependency against
its authoritative source. Do not edit generated copies: change shared source,
validate, commit, and assemble a new output directory.
