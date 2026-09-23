# CoCo client and server attestation

There are two separate enforcement points. KBS releases an image key only when
its resource policy permits it. Each generated NVFlare CCManager then
periodically verifies tokens from all protected participants. CCManager cannot replace
KBS policy, genpolicy, image signing/encryption, or approved platform references.

The server can run as a protected CoCo participant or verify CoCo client proofs
on an ordinary host or in an ordinary container. Verification requires the
trusted AS public key and project audience,
not a CoCo Pod, Kata runtime, confidential GPU, guest Attestation Agent, or a
network connection to Trustee. Token generation has different prerequisites:

| Operation | Where it runs | What it accesses |
| --- | --- | --- |
| `generate()` | Protected server or client inside its CoCo guest | Guest-local AA token API; AA contacts Trustee/KBS when a new attestation token is needed |
| `verify_for_site(token, authenticated_site)` | Ordinary server or another participant | Local pinned AS public key, signed subject matching the authenticated peer, token claims, and in-memory replay cache; no Trustee/RVPS query |

### Required attestation coverage

`cc_enabled_sites` is the locally provisioned **required set**, not a hint that a
peer can override. If it includes `server`, registration requires an envelope
containing exactly NVFlare's authenticated logical root-server identity `server`
and verifies its token against that identity. This logical name is not the
certificate DNS name. Secure FL authentication must remain enabled. Adding
`cc_config: cc_server.yml` to the server participant includes `server` in every
generated required set; the protected server's authorizer uses
`site_name: server`. Clients reject a missing, wrong-identity, or invalid server
proof during registration. Ordinary clients receive verifier-only components
when the server is protected, so they enforce the same server requirement.
Without a server `cc_config`, generated CoCo client-only deployments omit `server`
from this set, and the ordinary server needs no TEE or generated token.

Periodic and pre-job validation require verified tokens covering every locally
configured protected participant. Server discovery supplies routes only: missing
required sites, duplicate names/routes or a substituted root-server route fail
closed. A complete set with an invalid token still fails. Ordinary participants
outside the required set do not acquire an attestation requirement.

The first periodic round waits one configured verification interval plus
0–20% jitter for coordinated startup; pre-job validation does not wait and cannot
pass with missing attestations. All required sites must be connected and able to
attest by that first round and remain available thereafter. An omitted/offline site
is a validation failure, not an implicit membership removal, and follows the
existing federation-shutdown policy. Coordinate startup and review the configured
validation interval; changing federation membership requires trusted
reprovisioning rather than accepting a shorter server-provided list.

### Per-participant attestation namespaces

Provisioning now writes `required_site_verifier_ids` into each generated
CCManager configuration. It maps every protected site's logical identity to
the verifier component IDs corresponding to that site's configured issuers.
At startup, CCManager resolves those IDs to namespaces using the local trusted
verifier components. The received namespace set must match that site's set
exactly, and every proof must verify; missing, duplicate, extra or unknown
namespaces fail closed. Server identity is `server`, not its certificate DNS name.

For example, a CoCo server and one CoCo client require the combined CoCo
authorizer namespace for each participant:

```json
"cc_verifier_ids": ["coco_authorizer"],
"cc_enabled_sites": ["server", "site-1"],
"required_site_verifier_ids": {
  "server": ["coco_authorizer"],
  "site-1": ["coco_authorizer"]
}
```

With an ordinary server, the required map instead contains only protected
clients. CPU/GPU appraisal inside each combined proof is still enforced by
CoCoAuthorizer. No extra GPU token namespace is required. The protected server
uses the same AMD SEV-SNP and NVIDIA confidential-GPU runtime/profile as the
clients; CPU-only CoCo provisioning and mixed CC compute environments are not
supported by this example.

Re-run provisioning and distribute the regenerated kits to fix heterogeneous
deployments. Existing hand-written/previously generated configurations without
the mapping retain the strict all-verifier requirement; they do **not** silently
accept arbitrary subsets. Direct configurations can add the mapping explicitly.
It must cover exactly `cc_enabled_sites`, with a non-empty, duplicate-free list
of known verifier IDs for each site. Only trusted provisioning may change it;
peer tokens and discovery responses cannot reduce the requirements.

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
node, workload image, or CoCo IT. Authentically deliver the public PEM and confirm
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
to collect and approve a new measurement and v3 admin contract before provisioning.
The contract must explicitly approve UID/GID 65532 and
`readOnlyRootFilesystem: false` for the current NVFlare packager, along with the
[remaining application security settings](../admin/APPROVED-LAUNCH-PROFILE.md#approved-application-security-context-v3).
A read-only example profile cannot silently authorize a writable NVFlare Pod.
Do not add a kernel-parameter annotation to the generated Pod or reuse the old
measurement. Configure `aa.toml` in InitData with the intended KBS URL and
authenticated TLS certificate as well. Do not expose the loopback API outside
the guest or print its private-key-bearing response.

## Provisioning and generated configuration

Use [cc_site-1.yml](cc_site-1.yml), optionally enable the server's
[cc_server.yml](cc_server.yml) reference in [project.yaml](project.yaml), and run:

```bash
nvflare provision -p project.yaml
```

Each signed kit receives the configuration appropriate to its role. Protected
participants' kits are packaged separately into encrypted images:

| File under `local/` | Protected client | Protected server | Ordinary verifier participant |
| --- | --- | --- | --- |
| `coco_authorizer__p_resources.json` | Pinned public key, audience, client site name, loopback API, EAR age limit | Same trust, audience and limits; site name `server`; guest loopback API | Same trust, audience and limits; no issuing site name |
| `cc_manager__p_resources.json` | `coco_authorizer` is issuer and verifier; required protected-site set | `coco_authorizer` is issuer and verifier; same required set | No issuers; verifier `coco_authorizer`; same required set |

NVFlare loads these component fragments alongside `resources.json.default`.
An ordinary server is not marked CC-enabled and does not request a guest token.
It continues authenticating with the usual FL certificates. Ordinary clients
can coexist. When the server is protected, they receive the verifier-only
configuration above and do not request guest tokens; when the server is ordinary,
their existing configuration is preserved. All protected participants generate
proofs and verify one another; each generated verifier checks the required
protected-site set, including the server when enabled.

`token_expiration` is the maximum accepted EAR age (1–300 seconds), not a
request to change AS token lifetime. `check_frequency` must be positive and
smaller than that limit (defaults: 300/120 seconds). The outer proof lifetime is
configured separately by the authorizer constructor argument `proof_lifetime_seconds`
(positive integer, default 300). Generation sets `exp = iat + proof_lifetime_seconds`;
verification limits both proof age and declared lifetime to its locally configured
value. Configure matching values on the issuer and verifiers; a stricter verifier
rejects longer-lived proofs. Existing generated kits use the 300-second default.
This lifetime option does not extend EAR validity or add clock-skew tolerance or retries.
AA may return a cached EAR; a new NVFlare proof does not imply
a new hardware attestation at every poll. EAR older than the maximum age or
expired beyond the clock-skew allowance fails closed.

### Separate EAR and outer-proof clock-skew checks

The separate `ear_leeway_seconds` constructor argument defaults to 180 seconds
and accepts integers from 0 to 180. It is passed as PyJWT's `leeway` to the EAR
decode during both generation and verification, matching the tested cold-boot
clock-skew patch. It allows `iat` and `nbf` (if present) up to that many seconds
ahead of the local clock, and accepts expiration less than that many seconds in
the past. Set it to 0 for strict time checks. The maximum EAR age is not increased:
an EAR older than `max_token_age_seconds` still fails even within expiration leeway.
Signature, required integer timestamps, `exp > iat`, CPU/GPU appraisals, and
outer-proof checks remain enforced.

The outer peer proof has a distinct `proof_iat_leeway_seconds` constructor
argument, also an integer from 0 to 180 with default 180. It permits the signed
proof's `iat` to be at most that many seconds ahead of the verifier's clock.
After verifying the JWT signature, the authorizer explicitly checks
`-proof_iat_leeway_seconds <= now - iat <= proof_lifetime_seconds`.
Set it to 0 to reject future-issued proofs. Unlike EAR leeway, this is **not**
generic JWT leeway: outer `exp` and optional `nbf` remain strict. The maximum
proof age and declared lifetime remain limited by `proof_lifetime_seconds`;
signature, audience, authenticated peer binding, and CPU/GPU checks are unchanged.
An accepted proof's identifier remains in the replay cache until its strict
expiration; no expired-proof acceptance window is introduced.

For example, a protected server with a clock 90 seconds ahead of a lagging
client can issue a proof that the client now accepts under the default
future-`iat` allowance, provided every other check passes. In the reverse
direction, the lagging client's proof appears 90 seconds old to the server;
it must still be unexpired and within the normal proof-age/lifetime limits.
This change does not extend those limits. Increasing the proof lifetime alone
would not fix rejection of a future-issued proof.

Configure the new option under `cc_issuers[].args` in every protected
participant's `cc_config` YAML:

```yaml
cc_issuers:
  - id: coco_authorizer
    path: nvflare.app_opt.confidential_computing.coco_authorizer.CoCoAuthorizer
    token_expiration: 300
    args:
      trustee_public_key_file: ./trustee-as-public.pem
      token_url: http://127.0.0.1:8006/aa/token
      proof_iat_leeway_seconds: 180
```

Use the same value on all protected participants in the project. Provisioning
checks this agreement and also writes the value into ordinary participants'
generated verifier configurations. Omitting it selects 180. This provisioning
option controls only the outer proof's future issue-time allowance, not the
EAR leeway or proof lifetime described above.

To deploy the updated authorizer or change the allowance, update the trusted
provisioning inputs, run `nvflare provision -p project.yaml`, and rebuild, sign,
encrypt and publish new protected workload images and their approved handoffs.
Deploy the regenerated ordinary participants' kits with the updated NVFlare
code as well. Do not edit signed startup kits or an already approved Pod YAML
to change these settings. This is not clock synchronization; the allowances add
no retries. Clock lag above the allowance, shorter outer-proof lifetimes than
the lag, and transient attestation failures can still prevent registration.
Retry behavior is described below; the shutdown policy is unchanged.

## Bounded token-generation retries

CCManager calls `CoCoAuthorizer.generate_with_retry(timeout, cancel_event)` before
client registration and when obtaining tokens for periodic cross-validation or a
peer refresh request. It retries only connection errors, HTTP timeouts, and
guest API HTTP 429/502/503/504 responses. Every successful response still goes
through the existing signature, freshness, CPU/GPU appraisal and TEE-key checks.
Malformed responses, invalid proofs, failed appraisals, and other HTTP statuses
(including 401/403/500) fail immediately. An upstream KDS failure hidden behind
an undifferentiated 401 or 500 is **not** automatically retried: that response
cannot safely be distinguished from a permanent attestation/policy failure.

Defaults (constructor arguments):

| Component | Argument | Default |
| --- | --- | --- |
| CCManager | `registration_token_timeout` | 300 seconds |
| CCManager | `refresh_token_timeout` | Derived: `min(30, get_token_request_timeout / 2)`; 22.5 seconds with the default 45-second request timeout |
| CCManager | `get_token_request_timeout` | 45 seconds |
| CoCoAuthorizer | `retry_max_attempts` | 10 attempts, including the first |
| CoCoAuthorizer | `retry_initial_delay` | 1 second |
| CoCoAuthorizer | `retry_max_delay` | 15 seconds |
| CoCoAuthorizer | `retry_backoff_multiplier` | 2 |
| CoCoAuthorizer | `retry_jitter_ratio` | 0.5 |

The first attempt is immediate. With the defaults, subsequent attempts use exponential backoff
with jitter: 0.5–1 seconds, 1–2, 2–4, 4–8, then 7.5–15 seconds. Generation stops
when either the attempt limit or the monotonic time budget is exhausted.
Queueing behind another generation request consumes the same budget. The
manager shares the generation budget across its issuers; legacy authorizers
retain their existing single-attempt implementation and are not given CoCo's
bounded worker or retry classification.

Set the authorizer retry options under `cc_issuers[].args` and the CCManager
timeouts under `cc_attestation` in each protected participant's `cc_config` YAML
referenced by `project.yaml` (see `cc_site-1.yml` and `cc_server.yml`). For example,
these optional fields explicitly
select a 30-second refresh budget; omit it to use the derived default:

```yaml
cc_issuers:
  - id: coco_authorizer
    path: nvflare.app_opt.confidential_computing.coco_authorizer.CoCoAuthorizer
    token_expiration: 300
    args:
      trustee_public_key_file: ./trustee-as-public.pem
      token_url: http://127.0.0.1:8006/aa/token
      retry_max_attempts: 10
      retry_initial_delay: 1.0
      retry_max_delay: 15.0
      retry_backoff_multiplier: 2.0
      retry_jitter_ratio: 0.5
cc_attestation:
  check_frequency: 120
  registration_token_timeout: 300
  refresh_token_timeout: 30
  get_token_request_timeout: 45
```

Run `nvflare provision -p project.yaml` again. Provisioning writes the authorizer
settings into each protected participant's resources and the manager timeouts
into every generated manager. All CoCo participants in the project must
share the manager timeouts; their authorizer backoff settings may differ.
The same names are constructor arguments when configuring components directly.

Delays must satisfy `0 < retry_initial_delay <= retry_max_delay`; the multiplier
must be at least 1, and the jitter ratio must be between 0 and 1. Each delay is
sampled uniformly from `delay * (1 - retry_jitter_ratio)` through `delay`, then
the delay grows by the multiplier up to the maximum. Ratio 0 disables jitter;
ratio 1 enables full jitter. Numeric settings must be finite, and attempts must
be an integer from 1 through 100. Invalid settings fail before kit publication.

Each peer's `get_token_request_timeout` must exceed the remote
peer's `refresh_token_timeout`, with room for network transit (the constructor
also checks this against its own refresh budget). Rebuild protected images and
deploy updated ordinary participants' code. Omitting the optional settings retains
the defaults.
Existing resource configurations that set the old 10-second peer timeout but
omit `refresh_token_timeout` now receive a 5-second refresh budget and can start
without changing that request timeout. An explicitly configured refresh budget
is preserved, not clamped: a 30-second budget with a 10-second request timeout
still fails validation. Use matching timeouts across peers; a local derived
budget does not change a remote participant's configuration. Provisioning and
CCManager share the same default-resolution and finite-positive validation.

Shutdown cancels outstanding retry waits. A bounded caller wait and one guarded
worker per CoCo authorizer prevent stalled HTTP from blocking registration
indefinitely or accumulating retry workers. Python cannot forcibly terminate
an in-flight HTTP call: if it outlives the budget, its result is discarded and
the worker retains the guard until it exits. Subsequent requests cannot start
another worker in the meantime. HTTP connect/read timeouts are capped by the
remaining budget; they are not treated as an absolute wall-clock guarantee.

Exhaustion returns no usable token and follows the existing registration or
periodic-validation failure path (including federation shutdown where already
configured). This change does not cache VCEKs, quarantine individual sites,
extend token validity, or reuse proofs. Direct `generate()` remains a single
attempt; standalone callers can explicitly use `generate_with_retry()` with
a timeout in seconds and a `threading.Event` for cancellation.

## Direct client/server API

The constructor does **not** accept `expected_workloads`. Use `generate()` on
each protected server or client and `verify_for_site(token, authenticated_site)` at a participant
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
    ear_leeway_seconds=180,
    proof_iat_leeway_seconds=180,
)
proof = client.generate()
# Send only proof to the server over an authenticated, encrypted connection.
# Never send the raw AA response or its tee_keypair field.
```

For a protected server, construct its issuing authorizer inside its own guest
with `site_name="server"`. Keep that logical identity even when its TLS
certificate and project participant name are `server.example.com`. A client
checking the authenticated root server uses `verify_for_site(proof, "server")`.
Generated kits set these values automatically.

On the ordinary server, instantiate the verifier once and reuse it across
requests so its replay cache remains effective:

```python
from pathlib import Path

from nvflare.app_opt.confidential_computing.coco_authorizer import CoCoAuthorizer

verifier = CoCoAuthorizer(
    trustee_public_key=Path("trustee-as-public.pem").read_text(),
    audience="nvflare-coco:example-project",
    proof_lifetime_seconds=300,
    ear_leeway_seconds=180,
    proof_iat_leeway_seconds=180,
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

### Optional local EAR constraints

The default trust boundary delegates platform and workload approval to the pinned
AS signer and secure services' AS/RVPS/KBS policies. The FL proof's project-specific
`audience` is always checked; it is distinct from the inner EAR audience. A deployment
that needs additional FL-side restrictions can configure a verifier directly:

```python
verifier = CoCoAuthorizer(
    trustee_public_key=as_public_key_pem,
    audience="nvflare-coco:my_project",
    ear_audience="my-reviewed-as-audience",  # only if AS actually emits this aud
    workload_constraints={
        "site-1": {
            "init_data": approved_init_data_sha256,  # 64 lowercase hex characters
            "measurement": approved_snp_measurement,  # 96 lowercase hex characters
        },
    },
)
```

Both options default to `None` for the existing Trustee flow. With constraints
configured, every verified subject must have an entry; all configured claims
must match signed CPU evidence. An entry can pin either or both fields. Obtain
values from the trusted platform and workload owner, never from the CoCo host.
An absent/mismatched configured EAR audience or workload claim fails closed.
The generated CC YAML schema does not infer these optional verifier pins:
configure them on the ordinary server's verifier after approving the release.
Do not bake a workload's own final InitData digest into that same image, which
would create a circular image/policy dependency.

Replay IDs are intentionally local to each verifier process. A still-valid proof
may be accepted by a different verifier or after restart. They do not prove a
fresh response to a verifier-issued nonce. Deployments requiring that stronger
property need a separate challenge/response protocol; no such guarantee is made
here. Keep secure FL authentication, site binding and protected participant keys.

Generated CoCo managers set `require_site_binding: true`. A custom authorizer
must declare `supports_site_binding = True` and implement `verify_for_site`;
otherwise startup fails. Legacy non-CoCo managers default to `false`. The base
`generate_with_retry` adapter cannot interrupt a legacy blocking `generate()`;
CoCo's implementation enforces its bounded request/retry budget explicitly.

### What verification does not authorize

CCManager binds a protected client's registration envelope to `CLIENT_NAME`,
the same asserted name that ClientManager must authenticate against its
certificate and registration nonce before accepting registration. It requires
all configured attestation namespaces and calls `verify_for_site()` with that
name. A different envelope name, missing token, or different signed subject
rejects registration without shutting down healthy clients. Ordinary clients
outside `cc_enabled_sites` do not require CC tokens. Periodic responses must
name the site requested through the FL transport. Keep FL authentication enabled.
When `server` is required, clients similarly bind the server registration proof
to the authenticated root-server identity `server` and reject an absent or invalid
proof before completing registration. The certificate's DNS name still controls
TLS endpoint authentication; it does not replace that logical attestation identity.

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
of reliable first-attempt generation or a live validation of the bounded retry
path added later. The current retry behavior is documented above and covered by
unit tests and the separate live retry check below; fail closed on exhaustion.

This historical live result establishes the revision above's cross-node `generate()`/`verify()`
path, including AS-signature and CPU/GPU-appraisal validation. It used a
plaintext diagnostic client image, not the protected production application.
It does not constitute a full NVFlare registration/periodic-CCManager run,
workload decryption-key release test, or validation of the removed workload
identity checks. Rehearse the actual protected application and its complete
authorization path before deployment. Deployment-specific certificates,
measurements, node identities and private operational artifacts are not
included in this public example.

The later peer-binding, mixed-client registration, and protected-server
provisioning changes are covered by offline regression tests, not that historical
live test. Re-run a complete registration and periodic-validation rehearsal for
the selected ordinary-server or protected-server deployment before use. Protected
server testing must also cover its own encrypted image-key authorization,
stable network endpoint, and rejection of missing or invalid server proofs.

### Live bounded-retry check (2026-09-16)

The updated, unmodified authorizer ran inside an SNP/NVIDIA diagnostic CoCo Pod.
Direct generation used the actual guest AA API and returned a verified proof.
A guest-local HTTP relay then injected two 503 responses before forwarding the
third request to that same real API: the retry path recovered in 2.577 seconds,
and signature, CPU/GPU appraisal, peer-binding, and replay checks passed.
Additional real-HTTP cases rejected a 401 after one request, exhausted a
three-attempt limit after three 503 responses, and enforced a 0.2-second budget
in 0.202 seconds. The Pod completed successfully with zero restarts.

These were controlled transport failures, not a naturally occurring KDS outage.
No authorizer methods were mocked and no attestation policy was weakened.
The test used a plaintext diagnostic image and an in-guest verifier; it did not
exercise full NVFlare registration, CCManager event/F3 integration, or workload
decryption-key release. No raw token or guest private key was persisted.

A second run correlated the diagnostic with secure-services logs and TCP-header
observations inside AS's network namespace. It confirmed live HTTPS exchanges
with both AMD KDS and NVIDIA NRAS, plus successful SNP/NVIDIA verification and
KBS attestation responses. No cache was cleared and no service restarted.
An intermediate VCEK-fetch failure produced a KBS HTTP 401; the guest attestation
stack retried internally and subsequently obtained HTTP 200. The Python
authorizer only saw the relay's two injected 503 responses and final 200: this
does **not** establish that it retries generic 401 responses. Vendor HTTP bodies
were not decrypted or retained, and connection counts are not request counts.
