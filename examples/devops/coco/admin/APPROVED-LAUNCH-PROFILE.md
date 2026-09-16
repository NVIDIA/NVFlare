# Approved workload launch profile: trusted_system to provisioning_node

The Pod generator requires an authenticated launch contract from the trusted
platform owner. It no longer relies only on manually matching the workload's
resource fields to the measurement rehearsal.

Only `coco-approved-workload-launch/v3` with
`guest_token_api: guest-local-aa-token/v1` and an explicit
`workload_security_context` is accepted. Follow the
[runtime-profile migration](../RUNTIME-PROFILE.md) and security-context migration
below; v1/v2 profiles must be replaced by newly reviewed, rehearsed and authenticated
contracts, not edited to add fields or change their schema identifier.

## Two separate handoffs

| Recipient | File | Purpose |
|---|---|---|
| secure services | `platform-reference-values.json` | Exactly five RVPS values: one SNP measurement and four TCB floors |
| provisioning_node | `approved-workload-launch-profile.json` | Approved Pod constraints and runtime/configuration provenance |

The admin file contains no private keys, certificate chains, hardware reports,
host paths or full runtime configuration. The two files have different schemas
and must not be substituted for each other. CoCo IT still receives only the
final workload Pod YAML from admin for each release; it receives neither
trusted-side evidence nor workload secrets. For separate cluster setup, provisioning_node
also supplies the public Kata chart and digest pins, without a signed bundle.

## 1. Export on trusted_system

Run trusted-system stages 01 through 09 as documented. Stage 09 now stores
SHA-256 bindings for the approved profile and actual QEMU launch capture in
`platform-reference.final.env`. Stage 10 verifies those bindings when an admin
output is requested. A modified or unbound source profile fails closed.

```bash
cd /home/operator/coco_deployment
source /home/operator/private-platform-reference/platform-reference.env
PROFILE="$PLATFORM_WORK_ROOT/$PLATFORM_PROFILE"
install -d -m 0700 "$PROFILE/handoff-with-admin/secure-services" "$PROFILE/handoff-with-admin/admin"
bash trusted_system/10-export-platform-reference-values.sh \
  "$PROFILE/platform-reference.final.env" \
  "$PROFILE/handoff-with-admin/secure-services/platform-reference-values.json" \
  "$PROFILE/handoff-with-admin/admin/approved-workload-launch-profile.json"
sha256sum "$PROFILE/handoff-with-admin/admin/approved-workload-launch-profile.json"
```

Outputs must not already exist. The existing two-argument stage-10 command
still emits only secure services' five-value JSON. No extra numbered stage is needed.
For older profiles without the token API, preserve their evidence and create a
new profile through stages 03–10. Do not insert approval hashes or capability
fields by hand to bypass finalization.

## 2. Authenticate and install on provisioning_node

The trusted coordinator authenticates the trusted_system transfer and independently
obtains the expected SHA-256. Do not trust a hash supplied only by CoCo IT or
recalculate a pin from an untrusted incoming file and treat that as approval.

Copy the public contract through authenticated SSH/SCP to a private incoming
directory on provisioning_node. Then, as the trusted coordinator:

```bash
cd /home/operator/coco-admin
INCOMING=/path/to/authenticated/approved-workload-launch-profile.json
# Set from the authenticated trusted-system channel, NOT from CoCo IT.
EXPECTED_SHA256=APPROVED_LAUNCH_PROFILE_SHA256
python3 lib/workload-launch-profile.py "$INCOMING" "$EXPECTED_SHA256" \
  kata-qemu-nvidia-gpu-snp 3.29.0
install -m 0644 "$INCOMING" public/approved-workload-launch-profile.json
```

Set `WORKLOAD_LAUNCH_PROFILE_SHA256` in this kit's trusted `platform.env` to
that authenticated value. The checked-in kit contains no deployment-specific
approved profile or pin. Protect the platform configuration, validator and contract
from untrusted modification. A workload `.env` cannot override these readonly
settings after they are loaded. The pin is a trusted configuration check,
not a security boundary against an administrator who can edit both code and pin.

Use a separate admin kit directory for a different approved profile; do not
silently change the profile used by an existing release. Future profile
changes need trusted review/rehearsal and a new authenticated contract.

## 3. Generate and package as usual

```bash
./30-generate-pod-and-policies.sh /path/to/workload.env
./40-create-handoffs.sh /path/to/workload.env
```

Stage 30 validates the pinned contract before running genpolicy. It snapshots
the contract into the release, validates the initial Pod before genpolicy and
validates the final Pod afterward. Its checksum manifest includes the contract.
Stage 40 validates the release snapshot against the trusted pin and checks
the Pod again before creating any handoff directories. A changed kit pin,
missing snapshot or incompatible Pod causes an error; an old release is not
silently grandfathered in. Existing delivered handoffs are not modified or revoked.

The tested contract requires one container, one `nvidia.com/pgpu`, runtime
`kata-qemu-nvidia-gpu-snp`, omitted CPU/memory requests and limits, no host
namespaces, and no unreviewed annotations or additional Pod launch fields.
An omitted GPU request and a request of one are equivalent here, matching
Kubernetes limit-to-request defaulting. CPU/memory omissions refer to the
rehearsed runtime defaults of 1 vCPU and 8192 MiB; the generator must not
replace them with nominally equivalent resource requests without a new review.

The validator is deliberately limited to this reviewed profile shape. It
rejects sidecars/init/ephemeral containers, volumes, Pod overhead, runtime
overrides, privileged containers, resource substitutions and extra GPUs.
It does not restrict application image/command changes beyond the existing
workload policy-generation checks; new releases may use different applications.

## Security boundary and verification limits

This contract prevents trusted-side configuration drift. It does not remotely
prove what an adversarial CoCo host installed. Its runtime image/configuration
hashes identify the approved installation for coordination; the admin generator
does not inspect the target host. Trustee must still independently verify CPU
measurement/TCB, GPU attestation and workload init-data before releasing keys.
The contract itself is not a new attestation claim and is not sent as an AS policy.

The generator does not yet support arbitrary resource profiles. If the workload
needs different VM sizing, GPU count or container layout, review/rehearse the
new profile and extend the validator explicitly instead of weakening it.

## Approved application security context (v3)

The trusted platform owner approves these exact application settings in the source
Pod before stage 05. The shared `workload-security-context.py` validator is used
by trusted-system approval/finalization/export and admin Pod/policy validation.
Role-kit assembly materializes this shared implementation in both kits.

| Field | Requirement |
| --- | --- |
| `privileged` | Explicit `false` |
| `allowPrivilegeEscalation` | Explicit `false` |
| `runAsNonRoot` | Explicit `true` |
| `runAsUser`, `runAsGroup` | Approved integer IDs, greater than zero and less than 4294967295 |
| `capabilities` | Drop exactly `ALL`; `add` absent or empty |
| `seccompProfile` | Exactly `{type: RuntimeDefault}`; see guest limitation below |
| `readOnlyRootFilesystem` | Explicit approved boolean, not a universal `true` requirement |

Missing/unknown fields, wrong types (including booleans used as numeric IDs),
Pod-level security-context overrides, and safe-but-unapproved ID/rootfs changes
are rejected. Empty `capabilities.add` is canonicalized to absence.

For the static example approve `readOnlyRootFilesystem: true`. For provisioned
NVFlare clients approve `false` with UID/GID 65532: the current packager needs
writable guest-local logs and job state. Set this in the reviewed source YAML
before stage 05; do not silently widen an existing read-only approval to support
NVFlare. Stage 30 compares the generated context to the authenticated contract,
before running genpolicy and after constructing the final Pod. Stage 40 repeats
that check before packaging the handoff.

The rehearsal collector deliberately uses its own privileged diagnostic context.
Stages 05/09 approve and revalidate the **application source**, not the collector's
context. The exported constraint is a trusted approval, not a claim that the
collector ran with these application settings. The five-value RVPS JSON stays
unchanged in format; the context is not a new SNP launch-measurement field.

### Guest enforcement and its limits

The generator checks the embedded policy's application OCI UID/GID, empty Linux
capability sets, `NoNewPrivileges: true`, and exact `Root.Readonly` value. It also
checks that the corresponding pinned Kata rule guards are present. These are
structural consistency checks of known generated output, not a general Rego
verifier. Use only the authenticated pinned toolchain and reviewed rules.

Kata 3.29 genpolicy does not parse `runAsNonRoot`. Stage 30 validates the complete
context first, removes only that Kubernetes-only field for genpolicy, restores
it afterward, and checks the nonzero UID in the resulting guest policy.
The pinned [Kata 3.29 rules](https://github.com/kata-containers/kata-containers/blob/3.29.0/src/tools/genpolicy/rules.rego)
require null guest OCI `Seccomp`. Therefore `RuntimeDefault` is a YAML approval
requirement, **not proof of guest seccomp filtering**. Do not promise equivalent
guest protection for every Kubernetes security-context field. Guest seccomp
enforcement needs a separately reviewed runtime/policy change and hardware tests.

An adversarial cluster owner can bypass local launcher checks. The security
boundary remains guest agent-policy enforcement plus Trustee's check of the
approved init-data digest: changing the embedded policy changes that digest and
must fail workload key authorization. The security-context contract alone does
not protect against a modified runtime or prove guest enforcement.

### Migration and verification

1. Preserve old profiles/releases. Select a new profile directory, review all
   explicit source-YAML security fields, and run stages 05–10 with both rehearsals
   (03/04 first if the runtime changes). Stage 09 revalidates the source context
   against the approved profile; stage 10 requires the hash-bound context.
2. Authenticate and install the new v3 admin contract and its hash as above.
   Review the newly collected platform references on secure services as usual;
   changing only application security settings does not itself imply a different
   CPU launch measurement.
3. Create a new workload release using stages 30/40. Secure services must approve
   its resulting init-data digest/resource policy before delivery to CoCo IT.
   Previously installed workload authorizations are not automatically revoked.
4. Offline negative tests must reject root/changed IDs, privilege escalation,
   added capabilities, altered seccomp, missing fields, old contracts and flipped
   rootfs mode. Both explicitly approved read-only and writable releases must pass.
5. On an authorized test cluster, launch the approved encrypted workload, then
   test separate tampered copies: root UID, added capabilities, privilege
   escalation and flipped rootfs mode, **without regenerating init-data**. Confirm
   guest policy denial, not merely a launcher/admission rejection. Separately
   regenerate a changed policy but leave service authorization unchanged; confirm
   KBS key denial for its changed init-data digest. Do not approve the negative
   test digest or weaken production policy to make the test pass.

The offline checks do not establish these hardware outcomes. Record positive
and negative guest/KBS evidence before claiming an end-to-end security test.
