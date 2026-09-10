# Approved workload launch profile: trusted_system to provisioning_node

The Pod generator requires an authenticated launch contract from the trusted
platform owner. It no longer relies only on manually matching the workload's
resource fields to the measurement rehearsal.

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
For older finalizations without the new evidence bindings, preserve the old
final environment and rerun stage 09 against the retained verified evidence;
do not insert approval hashes by hand to bypass finalization.

## 2. Authenticate and install on provisioning_node

The trusted coordinator authenticates the trusted_system transfer and independently
obtains the expected SHA-256. Do not trust a hash supplied only by CoCo IT or
recalculate a pin from an untrusted incoming file and treat that as approval.

Copy the public contract through authenticated SSH/SCP to a private incoming
directory on provisioning_node. Then, as the trusted coordinator:

```bash
cd /home/operator/coco-admin
INCOMING=/path/to/authenticated/approved-workload-launch-profile.json
# Set from the trusted trusted_system channel, NOT from CoCo IT.
EXPECTED_SHA256=APPROVED_LAUNCH_PROFILE_SHA256
python3 lib/workload-launch-profile.py "$INCOMING" "$EXPECTED_SHA256" \
  kata-qemu-nvidia-gpu-snp 3.29.0
install -m 0644 "$INCOMING" public/approved-workload-launch-profile.json
```

Set `WORKLOAD_LAUNCH_PROFILE_SHA256` in this kit's trusted `platform.env` to
that authenticated value. The checked-in kit contains the tested profile and
the above pin. Protect the platform configuration, validator and contract
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
