# Installing a workload-owner handoff

This is the confidential per-workload handoff, not the trusted system's
SNP/TDX platform-reference handoff or CoCo's public runtime kit. Complete secure-services
installation before running the installer below.

The input is the `trusted-service/` directory created on `admin`. It contains
exactly:

```text
image_key
cosign.pub
image-security-policy.json
release-authorization.json
resource-policy-fragment.rego
SHA256SUMS
```

It must arrive through an authenticated confidential channel and must never
pass through the CoCo cluster.

## Receive and authenticate

```bash
install -d -m 0700 "$HOME/incoming"
chmod 0700 "$HOME/incoming/RELEASE"
chmod 0600 "$HOME/incoming/RELEASE/image_key"
cd "$HOME/incoming/RELEASE"
sha256sum --check --strict SHA256SUMS
```

Authenticate the checksum independently with the workload owner. A checksum
inside the same directory detects corruption but does not authenticate who sent
it. Review `release-authorization.json`, including its immutable image digest,
complete command/argument vector, CPU type, GPU requirement, canonical
`init_data_sha256`, hardware `init_data_claim`, and exact resource paths.
New handoffs use authorization v2. TDX's claim is the 32-byte digest plus
16 zero bytes for MRCONFIGID; SNP uses the unpadded 32-byte digest. A CPU-only
release must not accidentally authorize a GPU-required workload or another TEE.

## Install

```bash
cd "$HOME/coco-service-admin"
EXPECTED_MANIFEST_SHA256='<pin received from the workload owner over an authenticated independent channel>'
./12-install-trusted-service-handoff.sh "$HOME/incoming/RELEASE" "$EXPECTED_MANIFEST_SHA256"
```

The installer requires the independently authenticated SHA-256 of `SHA256SUMS`.
A checksum carried only inside the handoff does not authenticate its sender.
Stage 40 on the provisioning node prints the pin for that separate exchange.
Use matching reviewed role-kit revisions on both machines. Authenticated legacy
v1 handoffs are accepted only as SNP+GPU and upgraded to the independently
generated CPU/GPU-type-bound fragment. Prefer regenerating older not-yet-installed
handoffs; do not weaken the service template to accept an unknown fragment.
Already installed immutable releases
are not rewritten by updating these scripts.
The installer snapshots all six regular files into a private temporary directory,
validates the pin and every payload digest, then uses only those checked bytes.
It requires exactly the six expected files and validates their
cross-references. It generates executable Rego from validated authorization data
using secure services' own `policies/workload-resource-policy.rego.template`.
The received fragment must match that generated policy (apart from surrounding
whitespace); extra rules, removed checks, and other changes are rejected. Only
the locally generated fragment is merged into the active global resource policy.
Do not edit the service template to accommodate an unreviewed owner fragment.

OPA is required: stage 01 installs the checksum-pinned OPA 1.8.0 CLI. It checks
syntax and exercises positive/negative authorization cases as defense in depth;
sampled tests are not a substitute for the trusted template. No OPA server runs.
The installer holds the shared update lock, shows the complete diff, and requires
the release name as confirmation. It backs up current state, installs all three
resources (including the image key), verifies their persisted bytes, and commits
the authorization policy last. A resource upload failure can be retried before
the release becomes immutable.

Do not upload the fragment directly: `set-resource-policy` replaces the entire
global policy. Do not accept a KBS admin token, service private key, registry
publisher secret, Cosign private key, or plaintext image from the workload
owner. Do not send any handoff file or backup to CoCo IT.

After successful installation, tell the workload owner only that the named
immutable release is authorized. The owner then manually gives CoCo IT the
separate one-file Pod YAML handoff.

## Inspect a fresh workload release

After CoCo IT launches a fresh Pod, use a time window containing that launch:

```bash
./13-verify-workload-release.sh RELEASE 5m \
  "$HOME/incoming/RELEASE/release-authorization.json"
```

Use the reviewed authorization file retained with the installed handoff, not a
file from CoCo IT. Supplying it selects SNP/TDX and CPU-only/GPU expectations.
Omitting the third argument retains the historical SNP+GPU expectation; do not
omit it for TDX or CPU-only releases. The check inspects appraisals and the three
resource paths. It is diagnostic: separate success lines in a log window are
not cryptographic correlation of one attestation session to one key request.
Confirm authenticated application execution and peer-proof verification too.
Never expose guest workload logs or relax policy for this diagnostic.
