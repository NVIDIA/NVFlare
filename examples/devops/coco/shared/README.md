# Authoritative shared deployment code

Maintain these implementations here, not in role copies:

- `validate-config.sh`: hostname, endpoint and private-directory checks for all four roles.
- `kata-runtime-profile.py`: reviewed token-API configuration derivation and verification for trusted-system and CoCo kits.
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

## Application security-context contract

`workload-security-context.py` is the shared source for strict application-context
approval and structural checks of the generated guest OCI policy. Assembly places
it in the trusted-system, admin and CoCo role libraries; do not maintain separate copies.
See the [v3 contract](../admin/APPROVED-LAUNCH-PROFILE.md#approved-application-security-context-v3)
for migration, the collector exception and the pinned runtime's seccomp limitation.

The admin generator, final approved-profile validation and CoCo launch preflight
use the same effective-request checker. `ReadStreamRequest` and
`WriteStreamRequest` must be explicit JSON `false` values; global exec command
and regex allowlists and every container's `exec_commands` must be empty.
Policy replacement and fail-open settings remain disabled. A constant Rego
`default ... := false` is not evidence that its conditional allow rule is disabled.
These checks require the reviewed pinned rules; they do not prove arbitrary Rego
equivalence. The CoCo launch preflight parses InitData with Python 3.11+ `tomllib`,
provided by the documented Ubuntu host. No TOML backport is required.

## Private bootstrap download cache

The shared bootstrap creates `COCO_STATE_DIR/downloads` with mode 0700, uses
unpredictable mode-0600 temporary files, and rejects symlinks, hardlinked cached
files, untrusted ownership and group/world-writable cache paths or ancestors.
Root-owned sticky temporary ancestors are allowed, but the state directory and
cache themselves must be private. Artifact checksum failures cannot be bypassed.
Archive extraction copies the artifact to a fresh root-private temporary
directory, verifies that snapshot, and extracts those same bytes; the staging
directory is removed on success or failure.

Legacy mode-0775 caches are refused without modifying their contents. Set
`COCO_STATE_DIR` in the reviewed role configuration to a **new dedicated directory
under the operator's private home**, then rerun bootstrap to download and verify
fresh artifacts. Do not copy old cache contents or merely chmod a potentially
tampered cache. Existing services and recovery records are not deleted by this
migration. This does not add independent publisher authentication for CNI or the
Kubernetes repository signing key; those are separate supply-chain review items.
