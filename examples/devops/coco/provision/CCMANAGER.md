# CoCo clients with an ordinary NVFlare server

There are two separate enforcement points. KBS releases an image key only when
its resource policy permits it. NVFlare's CCManager on every participant then
periodically verifies tokens from all protected participants. CCManager cannot replace
KBS policy, genpolicy, image signing/encryption, or approved platform references.

The server can verify a CoCo client's proof on an ordinary host or in an ordinary
container. Verification requires the trusted AS public key and project audience,
not a CoCo Pod, Kata runtime, confidential GPU, guest Attestation Agent, or a
network connection to Trustee. Token generation has different prerequisites:

| Operation | Where it runs | What it accesses |
| --- | --- | --- |
| `generate()` | Protected client inside the CoCo guest | Guest-local AA token API; AA contacts Trustee/KBS when a new attestation token is needed |
| `verify_for_site(token, authenticated_site)` | Ordinary server or another participant | Local pinned AS public key, signed subject matching the authenticated peer, token claims, and in-memory replay cache; no Trustee/RVPS query |

## Before provisioning

### Secure-services owner: export the AS signing public key

The authorizer pins the ES256/P-256 key that signs Trustee EAR tokens. It must
not trust a key supplied by CoCo IT or by an incoming token. This key differs
from both the KBS HTTPS certificate and KBS administration authentication key.

On the trusted secure-services machine, inspect the deployed AS configuration:

```bash
cd /path/to/trustee
jq '.attestation_token_broker.signer' kbs/config/docker-compose/as-config.json
docker compose config
```

Resolve `signer.key_path` through the AS container's volume mounts to its host
file. For the standard compose layout, the configured container path
`/opt/confidential-containers/kbs/user-keys/token.key` corresponds to
`kbs/config/docker-compose/token.key`. Confirm the running AS actually uses
this configuration; do not guess a key from a different service or stale kit.
Export only the public key:

```bash
openssl pkey -in kbs/config/docker-compose/token.key -pubout -out trustee-as-public.pem
openssl pkey -pubin -in trustee-as-public.pem -text -noout
sha256sum trustee-as-public.pem
```

Use `sudo` for key-file access if required. If there is no persistent configured
signer, stop: configure and retain a P-256 AS signer using the reviewed Trustee
configuration procedure first. An ephemeral signer changes after restart and
cannot provide a stable pin. Never copy its private key to the provisioning
node, client image, or CoCo IT. Authentically deliver the public PEM and confirm
its SHA-256 through an independently authenticated channel with the provisioning
owner. Put it at `provision/trustee-as-public.pem` (not in the public example).
Changing this key requires reprovisioning clients/server, new images and
corresponding workload authorization; there is no trust-on-first-use fallback.

### Trusted platform owner: rehearse the guest API

The approved guest runtime must expose the [guest-components token API](https://github.com/confidential-containers/guest-components/blob/main/api-server-rest/README.md)
at `http://127.0.0.1:8006/aa/token?token_type=kbs` **inside the container's guest
network**, with KBS configured and both CPU/GPU evidence enabled. Its response
contains a signed token and a TEE private key. Never print it, save it, put it in
Pod logs, or expose this API through a Kubernetes Service, host port, proxy, or
ingress. No Kubernetes volume or hostPath is needed by the authorizer.

A runtime supporting GPU/SNP does not by itself prove this API is built,
enabled, reachable from the workload, or returns the required claims. Verify
those properties in a trusted rehearsal of the actual NVFlare image. If enabling
the API requires changes to the guest image or kernel command line, those are
measured launch inputs: repeat the trusted measurement workflow and approve the
new platform references before using it. These provisioning changes deliberately
do not alter a cluster runtime or reuse an old measurement after such changes.

In the tested pinned Kata 3.29.0 guest, the default REST feature exposed resource
routes but not `/aa/token`. The historical diagnostic used a per-Pod kernel
override; that is not supported by the packaged workload's approved profile.
The packaged workflow now derives and installs a runtime-level configuration
with `agent.guest_components_rest_api=all`, preserving all other parameters,
including repeated `pci=` options. Follow the [runtime-profile procedure](../RUNTIME-PROFILE.md)
to collect and approve a new measurement and v2 admin contract before provisioning.
Do not add a kernel-parameter annotation to the generated Pod or reuse the old
measurement. Configure `aa.toml` in InitData with the intended KBS URL and
authenticated TLS certificate as well. Do not expose the loopback API outside
the guest or print its private-key-bearing response.

## Provisioning and generated configuration

Use [cc_site-1.yml](cc_site-1.yml) and run:

```bash
nvflare provision -p project.yaml
```

The signed, encrypted client kit contains these additional files:

| File under `local/` | Client configuration | Ordinary server configuration |
| --- | --- | --- |
| `coco_authorizer__p_resources.json` | Pinned public key, project audience, site name, loopback API, EAR age limit | Same pinned public key, audience, and EAR age limit |
| `cc_manager__p_resources.json` | `coco_authorizer` is both issuer and verifier; protected client list | No issuers; verifier `coco_authorizer`; protected client list |

NVFlare loads these component fragments alongside `resources.json.default`.
The ordinary server is not marked CC-enabled and does not request a guest token.
It continues authenticating with the usual FL certificates. Ordinary clients
can coexist; all configured protected participants generate proofs and verify
one another, while the ordinary server verifies every protected client.

`token_expiration` is the maximum accepted EAR age (1–300 seconds), not a
request to change AS token lifetime. `check_frequency` must be positive and
smaller than that limit (defaults: 300/120 seconds). The outer proof lifetime is
configured separately by the authorizer constructor argument `proof_lifetime_seconds`
(positive integer, default 300). Generation sets `exp = iat + proof_lifetime_seconds`;
verification limits both proof age and declared lifetime to its locally configured
value. Configure matching values on the issuer and verifiers; a stricter verifier
rejects longer-lived proofs. Existing generated kits use the 300-second default.
This option does not extend EAR validity or add clock-skew tolerance or retries.
AA may return a cached EAR; a new NVFlare proof does not imply
a new hardware attestation at every poll. Too-old or expired EAR fails closed.

## Direct client/server API

The constructor does **not** accept `expected_workloads`. Use `generate()` on
the client and `verify_for_site(token, authenticated_site)` at a participant
authorization boundary. Obtain the expected site from authenticated FL/mTLS
identity, never from the submitted token or its envelope. Both verification
methods return `True` or `False`; reject on `False`. The compatible
`verify(token)` API checks proof validity only, without expected-peer binding.

For a standalone integration, load the public key authenticated by the
secure-services owner. Use the same project audience on both sides; provisioned
kits use `nvflare-coco:` followed by the project name. For example, inside the
CoCo client:

```python
from pathlib import Path

from nvflare.app_opt.confidential_computing.coco_authorizer import CoCoAuthorizer

client = CoCoAuthorizer(
    trustee_public_key=Path("trustee-as-public.pem").read_text(),
    audience="nvflare-coco:example-project",
    site_name="site-1",
    proof_lifetime_seconds=300,
)
proof = client.generate()
# Send only proof to the server over an authenticated, encrypted connection.
# Never send the raw AA response or its tee_keypair field.
```

On the ordinary server, instantiate the verifier once and reuse it across
requests so its replay cache remains effective:

```python
from pathlib import Path

from nvflare.app_opt.confidential_computing.coco_authorizer import CoCoAuthorizer

verifier = CoCoAuthorizer(
    trustee_public_key=Path("trustee-as-public.pem").read_text(),
    audience="nvflare-coco:example-project",
    proof_lifetime_seconds=300,
)

def accept_attestation(received_proof: str, authenticated_site: str) -> bool:
    return verifier.verify_for_site(received_proof, authenticated_site)
```

Omitting `site_name` creates a verifier-only instance; calling `generate()` on
it fails. The constructor's default loopback `token_url` is not contacted by
`verify()`. Generated NVFlare kits configure these components through resource
fragments; the snippets illustrate the API, not additional components to add
alongside the generated ones.

The supplied token must be the signed proof returned by `generate()`, which
contains the Trustee EAR and is signed by the guest-held TEE key. Passing the
raw EAR alone does not satisfy `verify()`.

## Verification protocol

1. The guest fetches the token response without environment proxies or redirects.
   It validates the EAR signature with the pinned AS public key and checks its
   time claims, profile, and successful CPU/GPU trust vectors.
2. It checks that the returned private key matches the public key in the signed
   CPU `runtime_data_claims.tee-pubkey`, then signs a short-lived proof containing
   the EAR, FL site, project-specific audience, issue/expiry times, and random
   single-use identifier. The private key stays in process memory.
3. Every participant verifies the EAR signature and then verifies the proof
   using the public key authenticated by that EAR. The proof must have the
   project-specific audience, the expected authenticated site as subject, a short lifetime, and a
   previously unseen identifier.
4. Both `cpu0` and `gpu0`, and no other submods, must carry this exact vector:
   `executables=3`, `hardware=2`, `configuration=3`, and `file-system`,
   `instance-identity`, `runtime-opaque`, `storage-opaque`, `sourced-data` all `0`.
   These checks match this kit's KBS resource policy; they are not universal
   success thresholds for every Trustee policy.
5. Each verifier rejects an expired proof or an already-seen proof identifier.
   Its bounded in-memory replay cache resets on restart; this is not a durable
   replay ledger or a server-challenge protocol. Keep FL mTLS enabled and
   credentials inside the encrypted image.
6. Registration failure rejects that client. Periodic validation failure invokes
   CCManager's existing shutdown behavior; do not treat attestation outages as
   harmless warnings. The cluster owner can still deny service or stop a Pod.

The signed-claim contract follows [Trustee's EAR documentation](https://github.com/confidential-containers/trustee/blob/main/attestation-service/docs/attestation_token.md).
An incompatible token format or policy fails closed rather than silently
accepting missing CPU/GPU appraisal claims. Proof of possession prevents forwarding
an EAR alone from satisfying NVFlare; it does not make arbitrary trusted
application code safe or replace the guest policy's isolation protections.

### What verification does not authorize

CCManager binds a protected client's registration envelope to `CLIENT_NAME`,
the same asserted name that ClientManager must authenticate against its
certificate and registration nonce before accepting registration. It requires
all configured attestation namespaces and calls `verify_for_site()` with that
name. A different envelope name, missing token, or different signed subject
rejects registration without shutting down healthy clients. Ordinary clients
outside `cc_enabled_sites` do not require CC tokens. Periodic responses must
name the site requested through the FL transport. Keep FL authentication enabled.

The authorizer still does not compare expected image, command, or InitData
values. The added peer binding is not workload authorization. Compatibility
`verify(token)` alone also does not bind a peer. Legacy non-CoCo authorizers
inherit their existing token-verification semantics unless they implement
site-aware verification themselves.

KBS workload/resource-path policies and protected guest policies remain separate
controls. Successful proof verification does not itself release an image key
or prove that a particular application is authorized for the FL project.

Because verification is local, it does not immediately discover an RVPS update,
reference removal, or policy change. Previously issued tokens can remain
acceptable until their expiration or the verifier's freshness limit. A fresh
proof may contain a cached EAR. Do not treat an RVPS update as immediate
revocation of every existing token. Replay state is local to one verifier
instance; replicas and process restarts require separate consideration.

## Verification status

Offline tests cover real cryptographic signatures, RSA/EC TEE proof keys,
expiry/replay failures, CPU/GPU appraisal failures, and actual startup-kit
provisioning with a mocked image runner. These tests alone do not establish
guest REST availability or change a remote runtime.

A separate live cross-node test on September 14, 2026 used the authorizer from
revision `45b5e50a80a1a141249672b2f1cba5876fa53abf` on both sides. Client A ran
inside a pinned Kata 3.29.0 SNP/GPU Pod; the verifier ran directly as a Python
process on an ordinary host outside CoCo. The client sent its generated proof
directly over TLS to that host. Only public AS/TLS trust material was distributed;
the guest private key remained in guest memory, and tokens were not logged or
saved. The updated authorizer's 40 targeted unit tests also passed.

| Live check | Result |
| --- | --- |
| Client `generate()` using the guest AA API | Passed on the second attempt |
| Ordinary-host `verify(proof)` | `True` |
| Replay against the same verifier instance | `False` |
| Altered proof signature against a fresh verifier | `False` |
| Wrong audience against a fresh verifier | `False` |

The first generation attempt returned `CCTokenGenerateError`. The test harness
waited one second and retried without modifying the authorizer or relaxing its
checks; the error's underlying cause was not established. This is not evidence
of reliable first-attempt generation or built-in retry handling. Handle transient
failures with bounded retries where appropriate and fail closed on exhaustion.

This historical live result establishes the revision above's cross-node `generate()`/`verify()`
path, including AS-signature and CPU/GPU-appraisal validation. It used a
plaintext diagnostic client image, not the protected production application.
It does not constitute a full NVFlare registration/periodic-CCManager run,
workload decryption-key release test, or validation of the removed workload
identity checks. Rehearse the actual protected application and its complete
authorization path before deployment. Deployment-specific certificates,
measurements, node identities and private operational artifacts are not
included in this public example.

The later peer-binding and mixed-client registration fixes are covered by
offline regression tests, not that historical live test. Re-run a complete
protected-client/ordinary-server registration rehearsal before deployment.
