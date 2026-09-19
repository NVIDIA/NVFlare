# Authoritative shared deployment code

Shared entry points and shell implementations (do not edit generated role copies):

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

The canonical Python helpers live in the importable `nvflare/lighter/cc_provision/`
package: `workload_security.py`, `workload_launch_profile.py`,
`kata_runtime_profile.py`, and `kbs_audience.py`. Source-tree scripts are adapters.
Role-kit assembly vendors the reviewed modules so remote machines need no NVFlare
checkout or installation. It also vendors the canonical successful EAR vector
from `nvflare/app_opt/confidential_computing/trustee_claims.py`; the authorizer,
authorization manifest and KBS policy generation all consume that definition.
Changes to these package sources must be committed before assembly as well.

`workload_security.py` supplies strict application-context approval, handoff
snapshot authentication, and generated guest OCI validation. Assembly places
it in all four role libraries; do not maintain separate implementations.
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

The complete reviewed Kata 3.29 rule preamble is SHA-256 pinned after the
AdditionalGids/CVE-2026-77176 derivation. Comments, dead branches, additional
allow rules, and any other preamble edits are rejected; searching for a guard
substring is not sufficient. Exactly one application OCI and one approved
non-root `/pause` sandbox OCI are allowed. Both privilege/capability profiles
are checked, including the pause capability expansion. A Kata rules upgrade
requires review and an explicit pin/fixture update, not automatic acceptance.

`lib/common-base.sh` owns shared shell primitives and bounded approval prompts.
Tests call package functions and source callable shell helpers instead of
extracting executable substrings from installation scripts. Wiring/entry-point
tests still check that the public adapters call the reviewed implementations.

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
migration.

### Pinned CNI release

Both `coco/config.env.example` and `trusted_system/bootstrap/config.env.example`
pin the linux-amd64 CNI archive using `CNI_PLUGINS_VERSION` and
`CNI_PLUGINS_SHA256`. Stage 10 requires a 64-hex digest before installing packages,
checks downloads against that configured pin, and checks the private extraction
snapshot against the **same pin**. It does not fetch or trust a checksum sidecar
at installation time, derive the expected hash from downloaded bytes, or allow
a checksum-mismatch bypass.

The v1.8.0 pin was checked against the
[official release](https://github.com/containernetworking/plugins/releases/tag/v1.8.0)
checksum, GitHub release-asset digest, and the downloaded archive's SHA-256.
These checks agree but are not independent publisher-signature authentication;
the reviewed configuration is the installation trust anchor. For upgrades,
review the new artifact and its provenance, update version and digest together
in **both templates**, and propagate them to each private `config.env`. Existing
configurations must add the reviewed pin; missing or malformed pins fail closed.
Never replace a pin merely to silence an unexpected checksum mismatch.

### Pinned Kubernetes apt signing key

Both role templates set `KUBERNETES_APT_KEY_FINGERPRINT` to the approved primary
fingerprint `DE15B14486CD377B9E876E1A234654DA9A296436` for the Kubernetes OBS
repository key. Stage 10 rejects a missing or malformed pin before installing
packages. The key is downloaded from the
[Kubernetes package repository](https://pkgs.k8s.io/core:/stable:/v1.34/deb/Release.key),
but its contents do not define the expected fingerprint.

`install_verified_apt_key` converts the key in a fresh root-private directory,
uses `gpg --with-colons --show-keys --fingerprint` with an isolated GPG home,
requires exactly one approved primary key, rejects expired/revoked keys and
unexpected additional primary keys, and installs the same checked binary keyring.
Validation failures leave the previous installed keyring unchanged. The key is scoped to
the Kubernetes source using apt's `signed-by`, not global apt trust.

The pin was checked against the current v1.34 public key and the fingerprint
recorded in the [upstream Kubernetes repository report](https://github.com/kubernetes/kubernetes/issues/133735).
This pins the reviewed key identity; it is not independent proof of publisher
ownership. Existing private `config.env` files must add the pin. For a legitimate
key rotation, authenticate the replacement through a trusted publisher channel,
review it, and update both templates and private configurations. A mismatch is
not permission to adopt the fingerprint just downloaded. If an old cached key
has expired, retain it for investigation and use a fresh private
`COCO_STATE_DIR` to fetch the publisher's renewed key; do not bypass verification.
