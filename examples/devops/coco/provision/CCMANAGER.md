# CoCo clients with an ordinary NVFlare server

There are two separate enforcement points. KBS releases an image key only when
its resource policy permits it. NVFlare's CCManager on every participant then
periodically verifies tokens from all protected participants. CCManager cannot replace
KBS policy, genpolicy, image signing/encryption, or approved platform references.

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
smaller than that limit (defaults: 300/120 seconds). The outer proof lasts at
most 60 seconds. AA may return a cached EAR; a new NVFlare proof does not imply
a new hardware attestation at every poll. Too-old or expired EAR fails closed.

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
   project-specific audience, a non-empty subject, a short lifetime, and a
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

## Verification status

Offline tests cover real cryptographic signatures, RSA/EC TEE proof keys,
expiry/replay failures, CPU/GPU appraisal failures, and actual startup-kit
provisioning with a mocked image runner.
They do not establish REST availability in the pinned guest image or constitute
an end-to-end hardware-attested FL run. No remote runtime is changed by these tests.
