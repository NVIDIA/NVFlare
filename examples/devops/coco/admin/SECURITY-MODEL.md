# Security model: workload owner versus adversarial `coco`

## Defensible claim

When the independent service policies are correct and all negative tests pass,
the `coco` administrator can schedule the approved encrypted workload but cannot
obtain its plaintext image layers or make a changed image, process specification,
or guest policy receive the protected KBS resources.

This is not a claim that the cluster cannot affect the workload. It controls
availability and observes public metadata.

## Adversary control

Assume `coco` controls Kubernetes, the node OS, containerd, Kata shim, hypervisor
arguments, disks, storage, DNS, routing, Pod YAML, and all Kubernetes APIs. It can
kill, pause, starve, replay, duplicate, delay, or disconnect the workload.

The adversary sees the Pod, image reference, manifest, signatures, encrypted-layer
sizes, access timing, public certificates, public signing key, resource requests,
and the complete embedded init-data/agent policy.

## Enforcement chain

| Threat | Enforcement |
|---|---|
| Host downloads image | Registry stores and anonymously serves ciphertext only |
| Host changes image | Immutable digest, guest signature policy, agent policy, and KBS authorization disagree |
| Host changes command/OCI properties | Kata agent policy rejects `CreateContainerRequest` |
| Host replaces guest policy or KBS endpoint | SHA-256 of exact init-data changes; attested SNP `init_data` no longer matches the service authorization |
| Host runs `kubectl exec`/attach/cp | No exec commands are authorized; exec and stream requests default-deny |
| Host tries SSH | Image has no SSH server, credential, or port; network services require owner mTLS |
| Host corrupts ciphertext | OCI digest/signature and authenticated encryption fail |
| CPU or GPU evidence is unacceptable | Service requires exactly `cpu0` and `gpu0`; each complete signed EAR trust vector must equal the platform-approved vector |

The service-owned CPU appraisal also requires all four signed SNP reported-TCB
SVNs (bootloader, TEE, SNP firmware, and microcode) to meet independently
approved minimums. These floors are platform policy, not workload input, and
must never be learned solely from `coco`.

For this pinned post-v0.21 Trustee SNP deployment, the resource policy compares the
`ear.veraison.annotated-evidence.init_data` claim with the 32-byte SHA-256 value
encoded as 64 lowercase hexadecimal characters. Base64 is retained only inside
the Pod transport encoding for the compressed init-data annotation; it is not the
format of this EAR claim.

## Why the Pod checksum is not the trust anchor

An authenticated out-of-band SHA-256 lets IT detect delivery corruption and gives
the owner an exact release record. A malicious IT operator can ignore it. The
trust anchor is the independently administered service policy: only attestation
of the exact embedded agent policy, image digest, process argv, trusted CPU, and
trusted GPU can release the image key and image-verification material.

## Logging and network limits

The guest policy can deny interactive process execution and agent stream
operations. It cannot erase bytes that the application voluntarily writes to
stdout/stderr, nor can it promise that the host Kubernetes log API is unavailable.
Therefore the application must never log secrets or sensitive payloads. In the
validated demo, host-side `kubectl logs` returned zero bytes because the process
emitted none, not because Kubernetes logging was disabled.

The host controls networking and traffic metadata. Applications must authenticate
peers with owner-controlled mTLS and must treat DNS, routes, source addresses, and
all unauthenticated inbound connections as hostile.

## Trusted service boundary

Trustee/KBS/AS/RVPS and registry write authorization must remain independent of
`coco`. A cluster owner who can change KBS resource policy, AS appraisal policy,
RVPS reference values, registry ciphertext, or Trustee TLS identity can defeat the
design.

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
- defending against a compromised workload-owner signing host or trusted service;
- proving that every Kata/Trustee/kernel/GPU component is vulnerability-free.

Exactly-once semantics require an owner-controlled, attestation-bound one-time
lease. Persistent data needs a separate attested encryption and rollback design.
