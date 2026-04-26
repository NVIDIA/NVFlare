# FLARE Distributed Provisioning

Created: 2026-03-27
Updated: 2026-04-25

---

## Problem Statement

Before distributed provisioning, FLARE provisioning is centralized: the Project Admin must
collect all participant details upfront, generate each participant's **private key**
centrally, and distribute full startup kits over a secure channel. This creates two
problems:

1. **Private keys in transit** - keys are generated centrally and must be sent to each site
2. **Centralized gathering** - all participant information must be collected before any kit
   can be generated

Distributed provisioning keeps private-key custody with each participant while making the
normal flow an approval workflow:

1. Requester creates an identity request.
2. Project Admin approves the request.
3. Requester packages the approval into a startup kit.

The private key is generated locally and must never be sent to the Project Admin.

---

## Approach: Request and Approval Workflow

The distributed provisioning workflow eliminates private-key transfer without requiring
new infrastructure. The requester generates its own private key locally. The only
portable artifacts exchanged are zip files:

- **Requester -> Project Admin**: request zip containing CSR and metadata, no private key
- **Project Admin -> Requester**: signed zip containing signed certificate, `rootCA.pem`,
  and approval metadata, no private key

The resulting startup kits are structurally identical to those produced by
`nvflare provision` and are fully compatible with FLARE runtime components.

### Canonical Step-by-Step Workflow

The public workflow always uses zip artifacts. Remote production and local automation use
the same command shape; the only difference is whether the zip files physically move
between machines.

| Step | Who | Action |
|------|-----|--------|
| 1 | Project Admin | `nvflare cert init --project example_project --org nvidia -o ./ca` *(one-time per federation)* |
| 2 | Requester | `nvflare cert request site site-3 --org nvidia --project example_project` |
| 3 | Requester | Send generated `site-3/site-3.request.zip` to Project Admin |
| 4 | Project Admin | `nvflare cert approve site-3.request.zip --ca-dir ./ca` |
| 5 | Project Admin | Return generated `site-3.signed.zip` to requester |
| 6 | Requester | `nvflare package site-3.signed.zip -e grpc://server1:8002` |
| 7 | Requester | `cd workspace/example_project/prod_00/site-3 && ./startup/start.sh` |

Step 1 is done once. Each new participant repeats steps 2-7 independently.

Remote examples assume the generated zip is copied into the receiver's current working
directory before running the next command. Local automation examples use the generated
nested path directly, such as `site-3/site-3.request.zip`.

### User Certificate Workflow

User certificates use a positional certificate role so the command shape remains stable
as roles evolve:

```bash
nvflare cert request user org-admin alice@nvidia.com --org nvidia --project example_project
nvflare cert approve alice@nvidia.com.request.zip --ca-dir ./ca
nvflare package alice@nvidia.com.signed.zip -e grpc://server1:8002
```

The positional certificate role is not a study role. Future study-specific roles are
assigned by study commands after the identity is provisioned.

### Local Automation Workflow

Local automation uses the same zip artifacts:

```bash
nvflare cert init --project example_project --org nvidia -o ./ca
nvflare cert request site site-3 --org nvidia --project example_project
nvflare cert approve site-3/site-3.request.zip --ca-dir ./ca
nvflare package site-3/site-3.signed.zip -e grpc://server1:8002
```

This is useful for tests and scripted local deployments. It exercises the same request,
approval, and package semantics as remote production, but skips the physical zip transfer.

### Multiple Participants

The canonical unit is one identity request and one signed response. This keeps remote
approval simple and makes private-key custody unambiguous.

Automation can repeat the same zip workflow for many participants:

```bash
nvflare cert request site site-1 --org nvidia --project example_project
nvflare cert request site site-2 --org nvidia --project example_project
nvflare cert request user org-admin admin@nvidia.com --org nvidia --project example_project

nvflare cert approve site-1/site-1.request.zip --ca-dir ./ca
nvflare cert approve site-2/site-2.request.zip --ca-dir ./ca
nvflare cert approve admin@nvidia.com/admin@nvidia.com.request.zip --ca-dir ./ca

nvflare package site-1/site-1.signed.zip -e grpc://server1:8002
nvflare package site-2/site-2.signed.zip -e grpc://server1:8002
nvflare package admin@nvidia.com/admin@nvidia.com.signed.zip -e grpc://server1:8002
```

A future bulk command may read a participant file and produce multiple request zips, but
it should still preserve the same artifact boundary: request zip goes to Project Admin,
signed zip comes back, `package` consumes signed zip.

### Custom Builders

Custom builders belong to startup-kit generation, not identity approval. Therefore
`cert request` and `cert approve` do not read or validate custom builders. The requester
passes the project file to `package`:

```bash
nvflare package site-3.signed.zip \
    -e grpc://server1:8002 \
    --project-file ./project.yml
```

In this mode:

- `site-3.signed.zip` supplies identity artifacts: signed certificate, `rootCA.pem`, and
  approval metadata.
- The local request folder or request state supplies the private key.
- `--project-file ./project.yml` supplies custom builders and project-level package
  configuration.
- `-e grpc://server1:8002` supplies the endpoint and overrides any endpoint-like builder
  configuration for the generated startup kit.
- The package step generates exactly one startup kit: the participant identified by the
  signed zip. It must not re-provision every participant listed in `project.yml`, and it
  must not overwrite existing startup kits for other participants.

If the project file contains participant entries, they are used as optional lookup and
configuration data. The signed zip remains the source of truth for identity. If the
signed identity is not found in the project file, `package` should continue with the
signed identity and project builders, but warn:

```text
Warning: site-3 was not found in project.yml participants; using signed zip identity and project builders.
```

If the signed identity is found in the project file, `package` may merge only
non-identity packaging configuration from the matching participant entry. The signed zip
is authoritative for identity fields: participant name, organization, kind, project,
certificate type, and certificate role. If any of those fields conflict between the signed
zip and `project.yml`, `package` must fail with a clear error instead of silently choosing
one side.

---

## Data Flow

### Artifact Ownership

| Location | Owns | Must not own |
|----------|------|--------------|
| Requester machine | Private key, request folder, returned signed zip, generated startup kit | Project CA private key |
| Project Admin machine | Project CA, received request zip, signed zip, approval audit | Requester private key |
| Out-of-band channel | Request zip and signed zip | Any `*.key` file |
| Runtime host | Generated startup kit | Request zip internals unless retained for audit |

### Step 0: Project Admin Initializes the CA

```bash
nvflare cert init --project example_project --org nvidia -o ./ca
```

Project Admin machine writes:

```text
ca/
  rootCA.key
  rootCA.pem
  ca.json
```

Data flow:

- `rootCA.key` stays only with the Project Admin.
- `rootCA.pem` is copied into signed responses.
- `ca.json` records CA metadata used by signing.

### Step 1: Requester Creates a Request

For a site:

```bash
nvflare cert request site site-3 --org nvidia --project example_project
```

For a user:

```bash
nvflare cert request user org-admin alice@nvidia.com --org nvidia --project example_project
```

Requester machine writes one request folder named after the identity. For the site example:

```text
site-3/
  site.yaml
  request.json
  site-3.key
  site-3.csr
  site-3.request.zip
```

For the user example, the same structure is written under `alice@nvidia.com/` with
`alice@nvidia.com.key`, `alice@nvidia.com.csr`, and
`alice@nvidia.com.request.zip`.

Data flow:

- `site-3.key` is generated locally and never leaves the requester machine.
- `site-3.csr` contains public key, subject identity, organization, and optional
  certificate-role hint.
- `site.yaml` contains identity metadata used later by `package`.
- `request.json` is generated in the request folder and placed into
  `site-3.request.zip`.
- The CLI records request metadata under centralized local state for audit and key
  lookup, for example `~/.nvflare/cert_requests/<request_id>/`.

The request audit record stores:

- a full request metadata snapshot
- request folder path
- request zip path
- private key path
- CSR path
- `site.yaml` path
- CSR, `site.yaml`, request zip, and public-key hashes

The request audit record must not copy or embed the private key. It stores only the
private-key path and a key/public-key hash so `package` can locate and validate the local
key later.

Output should make the generated artifact explicit:

```text
request_zip: site-3/site-3.request.zip
private_key: site-3/site-3.key
next_step: Send site-3.request.zip to the Project Admin.
```

The user does not manually assemble this zip. `cert request` creates it and excludes the
private key by construction.

### Step 2: Requester Sends the Request Zip

The requester transfers only:

```text
site-3.request.zip
```

The transport mechanism is outside NVFLARE: email, ticket system, secure file share, or
other operational channel. The design assumes the request zip can arrive on a different
machine with no access to the requester's filesystem.

### Step 3: Project Admin Approves the Request

On the Project Admin machine:

```bash
nvflare cert approve site-3.request.zip --ca-dir ./ca
```

The command reads:

```text
site-3.request.zip
ca/rootCA.key
ca/rootCA.pem
ca/ca.json
```

The command validates:

- The zip does not contain absolute paths or `..` path traversal entries.
- `request.json`, `site.yaml`, and exactly one CSR are present.
- No private key is present in the request zip; `approve` rejects the request if any
  `*.key` file is present.
- Hashes in `request.json` match the CSR and metadata files.
- CSR subject fields match `request.json` and `site.yaml`.
- The request project matches the CA metadata and root CA subject.
- Requested `kind` and certificate role map to an allowed certificate type.
- The Project Admin explicitly chose to approve this request.

The command writes:

```text
site-3.signed.zip
```

For audit, it writes an approval record under centralized local state, for example:

```text
~/.nvflare/cert_approves/<request_id>.json
```

The approval audit record stores:

- a full approval metadata snapshot
- request zip path
- signed zip path
- signer CA metadata, excluding `rootCA.key`
- certificate serial number
- certificate validity timestamps
- CSR, `site.yaml`, signed certificate, `rootCA.pem`, signed zip, and public-key hashes

The approval audit record must not copy or embed CA private-key material.

Output should show what was signed:

```text
name: site-3
org: nvidia
kind: site
cert_role: null
cert_type: client
project: example_project
csr_sha256: ...
signed_zip: site-3.signed.zip
next_step: Return site-3.signed.zip to the requester.
```

### Step 4: Project Admin Returns the Signed Zip

The Project Admin transfers only:

```text
site-3.signed.zip
```

The signed zip contains the signed certificate and root CA, but no private key.

### Step 5: Requester Packages the Startup Kit

On the requester machine:

```bash
nvflare package site-3.signed.zip -e grpc://server1:8002
```

Endpoint is package-time configuration. It does not belong in `cert request` because the
server endpoint is not identity. If the endpoint changes, a new startup kit can be
packaged without generating a new CSR or signed certificate.

The command reads:

```text
site-3.signed.zip
site-3/site-3.key
```

The command finds the private key by:

1. Explicit override: `--request-dir <dir>`.
2. Centralized local request state created by `cert request`, for example
   `~/.nvflare/cert_requests/<request_id>/`.
3. A request folder next to the signed zip, such as `./site-3/`.

Each discovered request directory must contain both `<name>.key` and `request.json`.
Incomplete candidates are skipped so a stray private-key file does not block lookup of the
real request folder.

The command validates:

- The signed zip does not contain absolute paths or `..` path traversal entries.
- `signed.json`, `site.yaml`, the signed certificate, and `rootCA.pem` are present.
- The signed zip metadata matches the local request metadata by `request_id`.
- The local private key matches the public key in the signed certificate.
- The certificate chains to `rootCA.pem`.
- Certificate CN, organization, project, and certificate type match the approved metadata.
- The endpoint scheme and host are valid for the generated kit.

`package` may materialize returned approval files into the request folder:

```text
site-3/
  site.yaml
  request.json
  site-3.key
  site-3.csr
  site-3.request.zip
  signed.json
  site-3.crt
  rootCA.pem
  site-3.signed.zip
```

Then it writes the startup kit:

```text
workspace/example_project/prod_NN/site-3/
  startup/
    start.sh
    fed_client.json
    client.key
    client.crt
    rootCA.pem
  local/
  transfer/
```

For a user startup kit:

```text
workspace/example_project/prod_NN/alice@nvidia.com/
  startup/
    fl_admin.sh
    fed_admin.json
    client.key
    client.crt
    rootCA.pem
  local/
  transfer/
```

### Step 6: Requester Runs the Startup Kit

For a site:

```bash
cd workspace/example_project/prod_00/site-3
./startup/start.sh
```

For a user:

```bash
cd workspace/example_project/prod_00/alice@nvidia.com
./startup/fl_admin.sh
```

---

## Bundle Model

### Local Request Folder

`cert request` creates a local request folder. The folder is not a startup kit. It is the
local identity material used to request and later package a startup kit.

For `site-3`:

```text
site-3/
  site.yaml
  request.json
  site-3.key
  site-3.csr
  site-3.request.zip
```

For `alice@nvidia.com`:

```text
alice@nvidia.com/
  site.yaml
  request.json
  alice@nvidia.com.key
  alice@nvidia.com.csr
  alice@nvidia.com.request.zip
```

The private key stays in this folder and is never included in the request zip.

### Request Zip

The request zip is what the requester sends to the Project Admin.

Example `site-3.request.zip`:

```text
request.json
site.yaml
site-3.csr
```

`site.yaml` is included in both the request zip and the signed zip. It is identity
metadata only; it does not contain secrets.

For a site:

```yaml
name: site-3
org: nvidia
type: client
project: example_project
kind: site
```

For a server:

```yaml
name: server1
org: nvidia
type: server
project: example_project
kind: server
```

For a user:

```yaml
name: org_admin@nvidia.com
org: nvidia
type: org_admin
project: example_project
kind: user
cert_role: org-admin
```

`request.json` contains metadata needed for review, matching, and error messages:

```json
{
  "artifact_type": "nvflare.cert.request",
  "schema_version": "1",
  "request_id": "8ff2d9e7-6f89-4acb-96e2-fc2d2fb8c6f7",
  "created_at": "2026-04-24T00:00:00Z",
  "project": "example_project",
  "name": "site-3",
  "org": "nvidia",
  "kind": "site",
  "cert_type": "client",
  "cert_role": null,
  "csr_sha256": "<hex>",
  "public_key_sha256": "<hex>"
}
```

The request zip must not contain:

```text
*.key
rootCA.pem
*.crt
```

`cert request` enforces this when creating the zip. `cert approve` also rejects request
zips that contain a private key, even if the zip was assembled manually.

### Signed Zip

The signed zip is what the Project Admin returns.

Example `site-3.signed.zip`:

```text
signed.json
site.yaml
site-3.crt      # signed participant certificate
rootCA.pem      # project root CA certificate
```

`signed.json` contains enough information for `nvflare package` to match the approval
to the local request:

```json
{
  "artifact_type": "nvflare.cert.signed",
  "schema_version": "1",
  "request_id": "8ff2d9e7-6f89-4acb-96e2-fc2d2fb8c6f7",
  "approved_at": "2026-04-24T00:00:00Z",
  "project": "example_project",
  "name": "site-3",
  "org": "nvidia",
  "kind": "site",
  "cert_type": "client",
  "cert_role": null,
  "certificate": {
    "serial": "<hex>",
    "valid_until": "2029-04-24T00:00:00Z"
  },
  "cert_file": "site-3.crt",
  "rootca_file": "rootCA.pem",
  "hashes": {
    "csr_sha256": "<hex>",
    "site_yaml_sha256": "<hex>",
    "certificate_sha256": "<hex>",
    "rootca_sha256": "<hex>",
    "public_key_sha256": "<hex>"
  }
}
```

The signed zip must not contain:

```text
*.key
```

The signed zip is not a startup kit. It is an approval response used to create a startup
kit on the machine that holds the private key.

### Filename Policy

Generated request and signed zip filenames preserve the participant name, including email
addresses:

```text
alice@nvidia.com.request.zip
alice@nvidia.com.signed.zip
```

The real identity is also stored in `request.json` and `signed.json`. If a future
platform-specific issue requires normalized filenames, metadata remains the source of
truth and can preserve the original identity.

---

## Comparison: Centralized vs. Distributed Provisioning

| | Centralized (`nvflare provision`) | Distributed (`request` / `approve` / `package`) |
|---|---|---|
| **Private key custody** | Project Admin generates and distributes | Requester generates locally; never leaves the machine |
| **Data distributed to site** | Full startup kit (keys, certs, config, scripts) | Signed zip with cert + `rootCA.pem`, no key |
| **Data sent from site** | Nothing | Request zip with CSR + metadata, no key |
| **Steps for Project Admin** | One command provisions all sites | Approve one request zip per participant |
| **Steps for requester** | Unzip and run | Request -> send zip -> receive zip -> package -> run |
| **Participant info required upfront** | All participants before any kit is generated | Each participant joins independently, on demand |
| **Adding a new site** | Dynamic provisioning with existing root CA | Same workflow; no impact on existing sites |
| **Endpoint changes** | Rebuild or edit kit configuration | Re-run `nvflare package` with new `-e`; no new certificate needed |
| **CC deployments** | Supported | Not supported |
| **HE deployments** | Supported | Not supported (future) |
| **Trust required in Project Admin** | Must trust Project Admin with your private key | Project Admin never sees private keys |

---

## Assumptions and Non-Goals

### Trust Model

This document covers standard (non-Confidential Computing) deployments.

- **mTLS is the trust anchor.** A participant is authorized if and only if its certificate
  chains to the project root CA. No startup kit integrity signature (`signature.json`) is
  needed for this check.
- **Trusted local requester.** The requester who runs `nvflare package` and the startup
  script is trusted not to modify their own startup kit maliciously. Startup kit integrity
  signatures, when present, protect managed startup content; they are not the general
  identity/authentication boundary for standard PKI deployments.
- **Local configuration customization is allowed.** Site admins may adjust local config
  (resource limits, log levels) without triggering any integrity failure.

### Non-Goals

- **Confidential Computing** - CC deployments are centralized by design; see
  [CC Deployments: Out of Scope](#cc-deployments-out-of-scope).
- **HE with distributed provisioning** - current HE requires a shared symmetric key
  generated centrally; per-site asymmetric HE is a future release item; see
  [HE Deployments](#he-deployments).
- **Hierarchical FL (relay nodes)** - relay node ownership is ambiguous in a distributed
  model. Use `nvflare provision` for relay topologies.
- **HUB / Hierarchy Unification Bridge** - HUB is deprecated and not enhanced by this
  design. Distributed provisioning does not add HUB-specific enrollment, approval, or
  packaging behavior.
- **Multi-root-CA federations** - all participants share one root CA per project.
- **Key rotation** - out of scope for this iteration.
- **Online approval service** - the exchange is intentionally out-of-band.
- **Study-specific role authorization** - this design keeps certificate provisioning
  from blocking that future model, but does not implement it.

---

## Signature Handling

FLARE has two independent signature systems.

### System 1: Startup Kit Integrity (`signature.json`)

**Purpose:** Detect tampering with managed startup kit content when startup-content
integrity metadata is present and startup-kit immutability is part of the deployment
security model.

`signature.json` is not the normal identity or authentication boundary for standard
non-CC, non-HE PKI deployments. In those deployments, the participant identity is
protected by mTLS and the certificate chain. Site operators may need to customize
`startup/` scripts and `local/` resources. Those local operational changes should not be
treated as runtime tampering unless the changed files are intentionally covered by startup
integrity metadata and checked by the runtime.

**Decision:** standard signed-zip distributed provisioning does not generate a
rootCA-signed `signature.json`, and standard non-CC, non-HE kits do not require one. If a
startup kit does contain valid startup integrity metadata, the existing runtime integrity
checks enforce it according to the runtime's secure-startup rules. If metadata is absent,
standard PKI deployments proceed with mTLS as the trust boundary.

`signature.json` is still required and enforced for:

- **CC deployments** - part of the CVM attestation chain; required before mTLS exists
- **HE deployments** - protects the shared TenSEAL context files

### System 2: Job Submission Signature (`__nvfl_sig.json`)

**Purpose:** Authenticate the job submitter and prove job content was not modified in
transit. Generated by `push_folder` in `file_transfer.py` using the admin user's private
key at submission time. Verified by the server (`job_runner.py`) and each client
(`training_cmds.py`) before executing the job.

**Decision:** Job signing is fully preserved in the distributed workflow. After the
workflow, the submitter holds a private key and a certificate signed by the project root
CA - exactly what `sign_folders()` requires. No changes are needed to the job signature
format or cryptographic verification primitive.

This is the same trust model as centralized provisioning: the submitted job carries the
submitter certificate, and `verify_folder_signature()` validates that certificate's chain
against the server/client `rootCA.pem` before verifying the folder signatures.

The server policy `require_signed_jobs` (default: `true` when `rootCA.pem` is present)
controls whether unsigned jobs are rejected. This policy applies uniformly regardless of
how participants were provisioned.

### HUB Job Forwarding

HUB is deprecated and outside the distributed provisioning scope. Distributed provisioning
must not introduce new HUB behavior.

Existing HUB job forwarding has a separate signature-handling concern: T1 receives an
originator-signed job, rewrites it into a T2-style job definition, and injects that
rewritten job into the T2 job store. If HUB remains supported, the HUB path must verify
the original T1 payload before any rewrite and remove the stale originator
`__nvfl_sig.json` before packaging the T2 payload. A rewritten T2 payload must not carry a
signature that was produced for the pre-rewrite T1 files.

If HUB remains deprecated and unused, this should be handled as a separate HUB
deprecation/cleanup decision, not as part of distributed provisioning.

---

## Code Changes

The runtime changes support startup kits assembled from requester-owned private keys. The
public CLI workflow is `request` / `approve` / positional `package`.

### Runtime and Security Changes

| # | File | Change |
|---|------|--------|
| 1 | `lighter/impl/signature.py` | Do not require `signature.json` for standard non-CC, non-HE signed-zip/distributed kits. Continue to generate it for modes that require startup-kit immutability, such as CC or HE. |
| 2 | `fed_utils.py` (`security_init`, `security_init_for_job`) | Do not treat absent startup integrity metadata as a failure for standard PKI kits. In secure runtime paths, run startup integrity checks only when valid startup integrity metadata exists. `secure_train` remains the PKI/mTLS switch and secure-startup trigger. |
| 3 | `file_transfer.py` (`push_folder`) | Guard `load_private_key_file` on key file existence before loading. Prevents crash in simulator (no key file); PKI runtime behavior unchanged. |
| 4 | `job_runner.py`, `training_cmds.py` | Replace `secure_train` gate on job sig verification with: verify if `__nvfl_sig.json` present; reject if absent and `require_signed_jobs=true`. |
| 5 | `fuel/hci/client/config.py` (`secure_load_admin_config`) | Gate strict `LoadResult.OK` check on `mgr.valid_config`. Without `signature.json` (non-CC, distributed workflow), `fed_admin.json` returns `NOT_SIGNED`; the strict check causes admin login to fail. |

After these changes, `secure_train` remains the secure-runtime switch. It must not be used
as proof that startup integrity metadata exists. Startup integrity enforcement depends on
the presence of valid startup integrity metadata in the startup kit.

### CLI Changes

| Command | Purpose |
|---------|---------|
| `nvflare cert request ...` | Create local key, CSR, metadata, and request zip. |
| `nvflare cert approve <request.zip>` | Project Admin signs a request zip and creates a signed zip. |
| `nvflare package <signed.zip> -e <endpoint>` | Requester combines signed approval with local private key and creates startup kit. |

The implementation should reuse existing CSR generation, certificate signing, and package
builder logic behind these commands. Public help and customer docs should expose this final
workflow: `cert request`, `cert approve`, and signed-zip `package`.

---

## Existing Deployment Compatibility

Distributed provisioning is additive. Runtime and security changes are designed to leave
existing deployments intact:

| Deployment | Behavior after changes |
|------------|----------------------|
| Centralized provisioning (non-CC, non-HE) | `nvflare provision` remains the centralized provisioning workflow. Standard PKI kits may omit `signature.json`; runtime behavior is unchanged because mTLS was already the trust anchor. |
| Centralized provisioning (CC) | `signature.json` still generated; startup check still runs. No change. |
| Centralized provisioning (HE) | `signature.json` still generated for TenSEAL context files. No change. |
| Distributed provisioning | Standard signed-zip packaging does not generate `signature.json`; startup integrity checks are skipped when metadata is absent. If valid metadata is present, existing runtime enforcement rules apply. |
| Mixed federation (some centralized, some distributed) | Each site resolves independently based on its own kit. Both connect to the same server over the same root CA. |
| Simulator | No PKI; job signing skipped (no key file). No change. |

`nvflare provision` remains the centralized provisioning workflow. The distributed
provisioning CLI is the public workflow for requester-owned private keys.

---

## Roles and Authorization

### Certificate Type Chain

The certificate type flows through three steps:

1. **`cert request`** - the requester chooses identity kind and, for user certificates,
   a positional certificate role:

   ```bash
   nvflare cert request site site-3 --org nvidia --project example_project
   nvflare cert request user org-admin alice@nvidia.com --org nvidia --project example_project
   ```

2. **`cert approve`** - the Project Admin approves the request zip and signs the
   certificate with the certificate type represented by the approved request metadata.

3. **`nvflare package`** - reads the signed zip and signed certificate. No role or type
   argument is needed; the signed certificate is the source of truth.

The design makes the Project Admin's trust decision explicit while keeping the normal
flow short and zip-based.

### Kind and Certificate Role Mapping

| User-facing input | Certificate type | Notes |
|-------------------|------------------|-------|
| `kind=site` | `client` | Federated runtime client/site. |
| `kind=server` | `server` | FL server identity. |
| `kind=user, cert-role=org-admin` | `org_admin` | User certificate role. Name must be email. |
| `kind=user, cert-role=lead` | `lead` | User certificate role. Name must be email. |
| `kind=user, cert-role=member` | `member` | User certificate role. Name must be email. |

The signed certificate's embedded certificate type remains authoritative at runtime. The
user-facing kind is CLI vocabulary. The positional certificate role must not be treated
as study authorization.

### Study Role Boundary

Study roles must not be part of `cert request`, request zip metadata, or signed zip
metadata. Provisioning answers "who is this identity and which organization does it
belong to?" Study authorization answers "what can this identity do in a specific study?"

This boundary keeps future study-specific authorization flexible. The same user identity
can be assigned different roles in different studies without generating a new
certificate:

```bash
nvflare study add-user cancer-research alice@nvidia.com --role lead
nvflare study add-user fraud-detection alice@nvidia.com --role org-admin
```

The positional certificate role should not be reused as the study authorization model.

### Default Authorization Policy

The role embedded in the certificate is enforced at runtime against
`local/authorization.json.default` on each site:

| Permission | `lead` | `org_admin` | `member` |
|-----------|--------|------------|---------|
| `submit_job` | any site | none | - |
| `clone_job` | own jobs | none | - |
| `manage_job` (abort/delete) | own jobs | jobs from own org | - |
| `download_job` | own jobs | jobs from own org | - |
| `view` | any | any | any |
| `operate` | own site | own site | - |
| `byoc` | any | none | - |

Sites may customize `local/authorization.json.default` to tighten or loosen these rules.

---

## Joining a Running Federation

Once a site runs `start.sh`, two things happen automatically:

1. **mTLS connection** - the server accepts any client whose certificate chains to
   `rootCA.pem`. No server-side allowlist, restart, or reconfiguration is required.
   The Project Admin approving the request zip is the act of authorization.

2. **Certificate role assignment** - for user certificates, the certificate type is read
   at login time and enforced against the site's authorization policy. This is not the
   future study-specific role model.

Adding a new site to an existing centrally provisioned federation follows the same
workflow: the new participant's request is approved by the same root CA. No
re-provisioning of existing sites is required.

---

## CLI Commands

### `nvflare cert init` - Project Admin, once per federation

```bash
nvflare cert init --project <name> --org <org> -o <ca-dir>
```

Produces:

```text
<ca-dir>/rootCA.key
<ca-dir>/rootCA.pem
<ca-dir>/ca.json
```

`rootCA.key` must stay with the Project Admin.

### `nvflare cert request` - Requester

```bash
nvflare cert request site <name> --org <org> --project <project> [--out <dir>]
nvflare cert request server <name> --org <org> --project <project> [--out <dir>]
nvflare cert request user <cert-role> <email> --org <org> --project <project> [--out <dir>]
```

Examples:

```bash
nvflare cert request site site-3 --org nvidia --project example_project
nvflare cert request server server1 --org nvidia --project example_project
nvflare cert request user org-admin alice@nvidia.com --org nvidia --project example_project
```

Produces:

```text
<name>/<name>.key
<name>/<name>.csr
<name>/request.json
<name>/site.yaml
<name>/<name>.request.zip
```

The request zip is generated by the command and is the only artifact sent to the Project
Admin. It does not include the private key.

Validation:

- `site` and `server` names must be valid participant names.
- User certificates require a positional `<cert-role>` before `<email>`.
- `<cert-role>` must map to a known certificate type, such as `org-admin`, `lead`,
  or `member`.
- User `<email>` must be an email address.
- Project names must be path-safe identifiers matching `[A-Za-z0-9][A-Za-z0-9._-]*`.

### `nvflare cert approve` - Project Admin

```bash
nvflare cert approve <request-zip> [--ca-dir ./ca] [--out <signed-zip>]
```

Example:

```bash
nvflare cert approve site-3.request.zip --ca-dir ./ca
```

Produces:

```text
<name>.signed.zip
```

The signed zip contains:

```text
signed.json
site.yaml
<name>.crt      # signed participant certificate
rootCA.pem      # project root CA certificate
```

It does not include the private key.

### `nvflare package` - Requester

```bash
nvflare package <signed-zip> -e <server-endpoint> [-w <workspace>] [--project-file <project.yml>] [--request-dir <dir>]
```

Example:

```bash
nvflare package site-3.signed.zip -e grpc://server1:8002
nvflare package site-3.signed.zip -e grpc://server1:8002 --project-file ./project.yml
```

`package` reads the signed zip, finds the matching local private key, validates the
certificate and metadata, and writes:

```text
<workspace>/<project>/prod_NN/<name>/
```

The server endpoint belongs to `package`, not `cert request`, because endpoint is
startup-kit connection config, not identity.

Custom builders also belong to `package`. When `--project-file` is provided, package
loads custom builders and project-level package configuration from that file while using
the signed zip as the source of truth for participant identity. Signed-zip mode is
selected-participant provisioning: only the signed participant is built, even if the
project file lists all servers, sites, and users.

If the signed participant appears in the project file, only non-identity packaging
configuration is merged from that entry. Conflicting identity fields must fail the command.

---

## CC Deployments: Out of Scope

Confidential Computing deployments are incompatible with distributed provisioning by
design. CC requires the CVM image to be built and attested centrally. If site admins
assembled their own kits, they would control what runs inside the enclave, defeating the
IP protection guarantee.

CC deployments continue to use centralized `nvflare provision` unchanged.

---

## HE Deployments

HE deployments are not supported in this release. FLARE's current HE implementation uses
a symmetric TenSEAL context where all clients share the same secret key, which requires
centralized generation via `nvflare provision`. Use centralized provisioning for any
federation that requires HE.

Asymmetric (per-site) HE key support is planned for a future release, at which point HE
will be fully compatible with the distributed provisioning workflow.

---

## Decisions

- `~/.nvflare/cert_requests` and `~/.nvflare/cert_approves` do not get retention or
  cleanup commands in this iteration. Audit records remain until the user deletes them.
- Request and signed zip filenames preserve participant names as-is, including email
  characters. Metadata remains the source of truth if filename normalization is needed
  later.
