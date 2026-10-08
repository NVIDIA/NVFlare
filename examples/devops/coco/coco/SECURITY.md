# CoCo adversary model

Read the [CoCo + NVFlare security architecture](https://nvflare.readthedocs.io/en/2.9/user_guide/confidential_computing/coco_security_architecture.html)
for the complete adversary model, enforcement flow and threat matrix. This page
summarizes the CoCo IT operating boundary; the architecture does not rely on a
hostile operator following these instructions.

## Operator responsibilities

Follow the [cluster installation guide](README.md) and
[manual release runbook](COCO-IT-RUNBOOK.md):

- Install the public pinned SNP+GPU runtime approved for the delivered workload.
- Authenticate the Pod checksum and launch the unchanged manifest. Request a new
  workload-owner release if its image, command, policy or resources must change.
- Run the documented bounded readiness and expected-denial checks. Do not use
  interactive guest access, policy replacement or workload log collection for
  troubleshooting.
- Receive no plaintext image, build context, private startup kit, signing key,
  image key, registry publisher credential or secure-services administrative
  credential. Fetch only the encrypted published artifact.

## What cluster checks do not establish

CoCo IT controls the host, Kubernetes, networking and manifests. The delivered
checksum and local launcher cannot constrain a malicious operator, and changing
YAML does not automatically change the embedded InitData. Independent release
authorization and approved guest enforcement check specific attested claims
and effective requests. InitData is public and attestation-bound, not
confidential or separately signed.

Exact-image-digest authorization has an unresolved coverage limitation. KBS
checks the image claim in the attested policy, not the host's actual pull
request. Another image signed by the accepted key in the permitted repository
is not shown to be rejected solely because its digest differs. The other
signature, decryption and guest-policy checks still apply; do not claim that
every image substitution fails. See the architecture's
[image authorization analysis](https://nvflare.readthedocs.io/en/2.9/user_guide/confidential_computing/coco_security_architecture.html#image-changes-and-the-current-authorization-limitation).

Cluster-side `Running`/`Ready` reports do not prove secure application execution.
The trusted federation operator performs
[authenticated NVFlare verification](../provision/VERIFY-RUNNING-FEDERATION.md).
The cluster operator can always prevent execution and observe public metadata,
resource use and traffic timing.

The guest policy's restrictions do not erase information that an application
emits to host-visible logs or unauthenticated network connections. Preserve the
runbook's silent-output contract, report unexpected output without collecting
its contents, and do not treat a zero-byte sample as proof that no earlier or
future leak is possible. Persistent storage, rollback, duplicate execution and
application data leakage have separate limits in the canonical threat matrix.
