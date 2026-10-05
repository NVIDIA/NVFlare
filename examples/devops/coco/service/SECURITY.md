# Service security boundary

Read the [CoCo + NVFlare security architecture](https://nvflare.readthedocs.io/en/2.9/user_guide/confidential_computing/coco_security_architecture.html)
for the complete trust model, attestation/key-release protocol, threat matrix and
limitations. This page lists the secure-services administrator's responsibilities
for SNP or TDX releases, with or without an NVIDIA confidential GPU.

## Keep administration independent

Treat CoCo IT as adversarial. Keep KBS/AS/RVPS policy administration, image keys,
AS signing keys, TLS private keys and their backups outside its control. Separate
registry publisher authorization from anonymous ciphertext pulls and KBS
administration. Protect [administrative credentials](TRUSTEE-ADMIN-CREDENTIAL-SECURITY.md)
and authenticate public certificate/key distribution.

Registry write access permits delivery disruption, deletion and replacement
attempts. Retain independent digest, signature and release checks. Exact binding
of the authorized image digest to the image actually decrypted and mounted
remains an unresolved coverage limitation. Each independently provisioned
release has its own encryption key and exact resource authorization: one
release's key cannot decrypt another release's encryption material, and the
other key must be denied under the first release's InitData. Repository-scoped
signature acceptance alone does not bypass those checks. Any replacement would
still need an accepted signature, authorized decryption resources, and all
guest-policy checks. A successful substitution has not been demonstrated;
retain the exact-digest assurance limitation without treating it as a proven
per-release key bypass.

## Approve platform references and workload releases separately

- Authenticate platform-owner approvals using the
  [reference handoff](PLATFORM-REFERENCE-VALUES-HANDOFF.md). For SNP, follow the
  [numeric TCB-floor policy](SNP-TCB-POLICY.md) and require an approved launch
  measurement extracted from the trusted rehearsal system's challenge-bound
  AMD-signed report, debug and migration disabled, and all four signed reported-TCB
  SVNs at or above independently approved numeric floors. Missing or nonnumeric
  floor data must fail; do not learn the floors solely from CoCo IT.
- For TDX, require one complete approved MRTD/RTMR0–3/XFAM/kernel-event tuple in
  `coco_tdx_profiles_v2`, verified quote/collateral and measured-boot evidence,
  acceptable TCB, and non-debug configuration. Independent per-field allowlists
  must not approve mixed profiles. Quote verification and event-log replay
  establish integrity; all four RTMR values also require explicit approval in
  that tuple. See [TDX approval](TDX-REFERENCE-VALUES.md).
- Require the approved CPU TEE and exactly `cpu0` for a CPU-only release, or
  exactly `cpu0` and `gpu0` for a GPU-required release, with each complete signed
  EAR trust vector matching its platform-approved vector. A valid CPU certificate
  chain alone does not establish an acceptable TCB. Installing the GPU verifier
  does not make a GPU mandatory for every workload. Neither the cluster owner
  nor a received token can downgrade a GPU-required release to CPU-only.
- Use the [reviewed workload handoff installer](TRUSTED-HANDOFF-RUNBOOK.md) for
  each immutable release. Review the image digest, complete arguments, InitData
  binding, CPU TEE, GPU requirement and resource paths. An accepted appraisal
  does not authorize arbitrary KBS resources.
- Install only the service's validated policy template, merged with existing
  reviewed releases. Do not upload a received Rego fragment as the global policy
  or give the workload owner KBS administrative credentials.

InitData is public configuration whose digest is bound into hardware evidence,
not a separately signed manifest. Its workload policy is enforced in the approved
guest. For the pinned Trustee SNP deployment, resource authorization compares
`ear.veraison.annotated-evidence.init_data` with the 32-byte SHA-256 digest encoded
as 64 lowercase hexadecimal characters. Base64 is used for the compressed Pod
annotation transport, not this EAR claim.
For TDX the claim is 96 lowercase hex characters: the same 32-byte SHA-256 digest
followed by 16 zero bytes, matching the quoted MRCONFIGID. Each workload's KBS
rule pins this exact target-specific InitData representation; arbitrary
truncation or nonzero padding must fail.

The encrypted image remains ciphertext in the registry and on CoCo storage;
decryption occurs inside the confidential guest after resource authorization.
The workload's generated policy must allow only its intended command, mounts,
environment and I/O. Do not place SSH credentials, an SSH daemon, debug shells,
host mounts, privileged mode, host namespaces or Kubernetes API credentials in
the workload. Application-level mTLS is needed to authenticate successful
execution; Kubernetes status does not establish it.

## Verify changes and maintain trust

Follow [installation and verification order](SERVICE-INSTALLATION.md), retaining
private approvals, version pins and verification records. Rerunning deployment
can replace the active resource policy with default-deny; follow the documented
reinstallation order before accepting workload launches again.

After authorization changes, check persisted references and policies and exercise
positive and negative release cases. Service logs and Kubernetes status alone
do not prove authenticated NVFlare operation; complete the
[federation verification](../provision/VERIFY-RUNNING-FEDERATION.md). Do not weaken
policy to diagnose a failed release.

The pinned service includes compatibility changes for the large CPU+GPU EAR path:
RVPS pool handling, a larger KBS HTTP head limit and matching Nginx header buffers.
Revalidate them before upgrading Trustee. Keep collateral, reference approvals
and minimum TCB levels current, and review runtime or firmware changes before
updating approvals.

The CoCo owner can refuse to schedule, kill the Pod, block the network or submit
a modified manifest that fails attestation. Revocation cannot recall keys already
released or instantly invalidate every cached attestation token. Availability,
persistent-state rollback and compromised
trusted components require separate controls described in the canonical
architecture.
