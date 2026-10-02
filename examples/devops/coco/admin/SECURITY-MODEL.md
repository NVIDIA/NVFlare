# Workload-owner security requirements

Read the [CoCo + NVFlare security architecture](https://nvflare.readthedocs.io/en/2.9/user_guide/confidential_computing/coco_security_architecture.html)
for the complete trust model, enforcement flow, threat matrix and limitations.
This page is the workload owner's operational checklist for SNP or TDX releases,
with or without an NVIDIA confidential GPU.

## Review and authorize each release

Treat the host, Kubernetes, storage, network and delivered Pod YAML as
adversary-controlled. Keep plaintext builds, startup-kit private credentials,
image keys and signing keys on trusted systems. Do not place secrets in public
manifests, OCI metadata, environment variables or arguments.

Follow the [build, review, encrypt, sign and handoff procedure](README.md), or
[NVFlare provisioning](../provision/README.md) for protected servers and clients.
Before publication:

- Authenticate the [approved launch profile](APPROVED-LAUNCH-PROFILE.md). New
  launch contracts use the reviewed v4 profile with an explicit CPU TEE, GPU requirement
  and matching runtime class. Legacy v3 is accepted only for its original SNP+GPU
  shape. Follow the [runtime-variant guide](../RUNTIME-VARIANTS.md) for the target.
- Review the image, complete command, application security context, generated
  agent policy, image signature policy and final InitData.
- Use a new immutable release and image key for each changed workload or
  authorization. Do not broaden an existing release to make a failed launch pass.
- Send the confidential service handoff only to the independent secure-services
  administrator. Send the final Pod YAML and independently authenticated checksum
  to CoCo IT only after authorization is installed.

InitData is public configuration whose digest is bound into hardware evidence;
it is not a separately signed manifest. An authenticated Pod checksum detects
transfer errors and records the release, but an adversarial operator can ignore
it. Independent KBS authorization and guest policy enforcement remain necessary.

Exact binding of the authorized image digest to the image actually decrypted and
mounted remains an unresolved coverage limitation. In particular, do not claim
that the current checks reject every same-key/same-repository image substitution.
Digest pins, signatures and new release keys remain required controls; they do
not by themselves close that limitation.

## Enforce the approved target

| Threat | Enforcement |
|---|---|
| Host downloads image | Registry stores and anonymously serves ciphertext only |
| Host changes image | Immutable digest, guest signature policy, agent policy and KBS authorization constrain the release, subject to the image-binding limitation above |
| Host changes command/OCI properties | Kata agent policy rejects `CreateContainerRequest` values outside its enforced rules; not every Pod/OCI field is constrained |
| Host replaces guest policy or KBS endpoint | SHA-256 of exact init-data changes; attested SNP HOST_DATA or TDX MRCONFIGID no longer matches service authorization |
| Host runs `kubectl exec`/attach/cp | No exec commands are authorized; exec and stream requests default-deny |
| Host tries SSH | Owner review must exclude SSH servers and unauthorized access paths; the packager does not scan custom images for them. Application network services require owner mTLS |
| Host corrupts ciphertext | OCI digest/signature and authenticated encryption fail |
| CPU or GPU evidence is unacceptable | Service requires the approved CPU type and exactly `cpu0`, plus `gpu0` for a GPU release; every vector must equal its approved target vector |

The service-owned CPU appraisal also requires all four signed SNP reported-TCB
SVNs (bootloader, TEE, SNP firmware, and microcode) to meet independently
approved minimums. These floors are platform policy, not workload input, and
must never be learned solely from `coco`.
TDX instead requires a complete approved reference tuple, verified boot events,
accepted quote/TCB/collateral status and non-debug configuration. Related fields
must match one approved profile, never a mixture of independent allowlists.

For this pinned post-v0.21 Trustee SNP deployment, the resource policy compares the
`ear.veraison.annotated-evidence.init_data` claim with the 32-byte SHA-256 value
encoded as 64 lowercase hexadecimal characters. Base64 is retained only inside
the Pod transport encoding for the compressed init-data annotation; it is not the
format of this EAR claim.
For TDX the claim is 96 lowercase hex characters: that same 32-byte SHA-256
digest followed by 16 zero bytes. Its quoted MRCONFIGID must agree. The service
authorizes the exact target-specific representation; arbitrary truncation or
nonzero padding is not accepted.

## Verify operation and protect application data

A successful build or Kubernetes `Running` status does not establish the security
claim. Complete hardware-backed positive and negative tests, then verify
[authenticated federation operation](../provision/VERIFY-RUNNING-FEDERATION.md).

Do not log secrets or sensitive payloads. Denying guest exec or stream requests
does not erase output already emitted to the host. Review stdout/stderr, file
logs, custom output paths and support procedures. See the
[IT verification limits](../coco/COCO-IT-RUNBOOK.md).
In the validated demo, host-side `kubectl logs` returned zero bytes because the
process emitted none, not because Kubernetes logging was disabled.

Use owner-controlled mutually authenticated TLS (mTLS) for application connections.
Host-controlled DNS, routes and source addresses do not authenticate a federation
peer. Keep registry publisher credentials separate from CoCo IT and follow the
[secure-services handoff requirements](../service/TRUSTED-HANDOFF-RUNBOOK.md).
Workload owners propose release constraints; secure services independently review
and install them. Workload owners do not receive KBS administrative credentials.

## Trusted service boundary

Trustee/KBS/AS/RVPS and registry write authorization must remain independent of
`coco`. A cluster owner who can change KBS resource policy, AS appraisal policy,
RVPS reference values or Trustee TLS identity can defeat the design. Registry
write access permits disruption and replacement attempts; retain the independent
digest, signature and release controls and account for the image-binding
limitation above.

Workload users supply release-specific resources and constraints. The service
administrator owns and reviews the global merge, CPU/GPU appraisal policies, and
reference-value provenance. Workload users never receive a KBS admin token.

The current service has compatibility changes for the large CPU+GPU EAR path
(RVPS pool handling, a larger KBS HTTP head limit, and matching Nginx header
buffers). Treat these as part of the pinned service deployment and revalidate them
before upgrading Trustee.

## Explicit non-goals

- availability, anti-rollback, exactly-once execution, and anti-replay;
- hiding image/POD metadata, sizes, timing, or network metadata;
- protecting secrets baked into an image before the owner audits it;
- protecting data sent to untrusted peers or written to logs;
- providing FL algorithm or data-privacy guarantees;
- defending against a compromised workload-owner signing host or trusted service;
- proving that every Kata/Trustee/kernel/GPU component is vulnerability-free.

Exactly-once semantics require an owner-controlled, attestation-bound one-time
lease. Persistent data needs a separate attested encryption and rollback design.

The operator can deny service and observe public metadata. Attestation does not
provide persistent-data rollback protection, singleton or exactly-once execution,
or FL algorithm/data-privacy guarantees. Review the canonical architecture's
application, peer-trust, storage, logging and upgrade assumptions before approving
a workload.
