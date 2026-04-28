# FLARE Distributed Provisioning

Created: 2026-03-27
Updated: 2026-04-27

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

### Key Design Decisions

- **Distributed provisioning is request/approve/package.**
  Requesters generate their own private key and CSR locally. The Project Admin
  approves a request zip and returns a signed zip. The requester packages the signed
  zip into a normal FLARE startup kit.

- **Private keys stay with the participant.**
  Request zips and signed zips never contain private keys. Only CSR, signed
  certificate, root CA, and approval metadata are exchanged.

- **Participant definition files are the request input.**
  `nvflare cert request -p/--participant <participant.yaml>` reads identity and
  package-time metadata from one participant definition file. The file uses a
  project-style structure: top-level project `name` and one `participants` entry.
  This keeps request input auditable, reusable, and consistent with existing
  provisioning concepts.

- **Project Admin approves against a project profile.**
  `project_profile.yaml` defines project `name`, communication `scheme`, and default
  `connection_security`. `nvflare cert approve --profile <project_profile.yaml>`
  verifies project identity and records approved communication defaults in the
  signed metadata.

- **Communication scheme is Project Admin-owned and consistent.**
  The communication `scheme` is defined once in `project_profile.yaml`. Requester
  participant files do not choose or override `scheme`. Approval signs the approved
  `scheme` into `signed.json`, and packaging uses the signed `scheme` when
  generating startup kits. This ensures all participants in the approved project use
  the same protocol, such as `grpc` or `http`.

- **Connection configuration is split by ownership.**
  Project-wide defaults come from the Project Admin profile. Client and user
  startup-kit server endpoints come from the requester's local participant
  definition. Server-side `connection_security` may be resolved locally during
  packaging for deployments behind proxies, load balancers, or ingress. Server-local
  connection overrides are excluded from sanitized approval metadata because they are
  local package-time behavior, not Project Admin policy.

- **Generated startup kits remain compatible with existing FLARE runtime.**
  The output layout and runtime startup behavior match centrally provisioned kits.
  Certificate generation reuses centralized provisioning logic to keep certificate
  contents and validation behavior consistent. Server certificate SAN/default-host
  behavior follows centralized provisioning conventions, including `localhost`
  support for local/demo workflows.

- **Root CA trust is explicit.**
  Interactive packaging can use `--confirm-rootca`. Automation can use
  `--expected-rootca-fingerprint`. If neither is provided, packaging warns that the
  root CA fingerprint was not verified out-of-band.

- **Custom startup-kit builders are supported.**
  Participant definition files may include a `builders:` section. Packaging honors
  those builders for the signed participant while preserving one-participant-at-a-time
  distributed provisioning.

### Canonical Step-by-Step Workflow

The public workflow always uses zip artifacts. Remote production and local automation use
the same command shape; the only difference is whether the zip files physically move
between machines. The table below uses `example_project` as a generic placeholder; the
Data Flow section uses a concrete hospital federation example.

| Step | Who | Action |
|------|-----|--------|
| 1 | Project Admin | Create `project_profile.yaml` with `name`, `scheme`, and `connection_security` *(one-time per federation)* |
| 1a | Project Admin | `nvflare cert init --profile project_profile.yaml -o ./ca` *(one-time per federation)* |
| 2 | Requester | Create participant definition file (e.g. `hospital-a.yaml`) with identity and connection fields |
| 3 | Requester | `nvflare cert request --participant hospital-a.yaml` |
| 4 | Requester | Send generated `hospital-a/hospital-a.request.zip` to Project Admin |
| 5 | Project Admin | `nvflare cert approve hospital-a.request.zip --ca-dir ./ca --profile project_profile.yaml` |
| 6 | Project Admin | Return generated `hospital-a.signed.zip` and share `rootca_fingerprint_sha256` out-of-band |
| 7 | Requester | `nvflare package hospital-a.signed.zip --confirm-rootca` |
| 8 | Requester | `cd workspace/example_project/prod_00/hospital-a && ./startup/start.sh` |

Step 1 and 1a are done once. Each new participant repeats steps 2-8 independently.

Remote examples assume the generated zip is copied into the receiver's current working
directory before running the next command. Local automation examples use the generated
nested path directly, such as `hospital-a/hospital-a.request.zip`.

### User Certificate Workflow

User certificates use a participant definition file that contains the certificate role:

```bash
nvflare cert request --participant alice.yaml
nvflare cert approve alice@hospital-alpha.org.request.zip --ca-dir ./ca --profile project_profile.yaml
nvflare package alice@hospital-alpha.org.signed.zip --confirm-rootca
```

The certificate role (`org_admin`, `lead`, or `member`) is declared in the participant
definition file and is not a study role. Future study-specific roles are assigned by study
commands after the identity is provisioned.

### Local Automation Workflow

Local automation uses the same zip artifacts:

```bash
# create project_profile.yaml and participant definition files
nvflare cert init --profile project_profile.yaml -o ./ca
nvflare cert request --participant site-3.yaml
nvflare cert approve site-3/site-3.request.zip --ca-dir ./ca --profile project_profile.yaml
nvflare package site-3/site-3.signed.zip --confirm-rootca
```

This is useful for tests and scripted local deployments. It exercises the same request,
approval, and package semantics as remote production, but skips the physical zip transfer.

### Multiple Participants

The canonical unit is one identity request and one signed response. This keeps remote
approval simple and makes private-key custody unambiguous.

Automation can repeat the same zip workflow for many participants:

```bash
nvflare cert request --participant site-1.yaml
nvflare cert request --participant site-2.yaml
nvflare cert request --participant admin.yaml

nvflare cert approve site-1/site-1.request.zip --ca-dir ./ca --profile project_profile.yaml
nvflare cert approve site-2/site-2.request.zip --ca-dir ./ca --profile project_profile.yaml
nvflare cert approve admin@nvidia.com/admin@nvidia.com.request.zip --ca-dir ./ca --profile project_profile.yaml

nvflare package site-1/site-1.signed.zip --confirm-rootca
nvflare package site-2/site-2.signed.zip --confirm-rootca
nvflare package admin@nvidia.com/admin@nvidia.com.signed.zip --confirm-rootca
```

A future bulk command may read a participant definition file and produce multiple request
zips, but it should still preserve the same artifact boundary: request zip goes to Project
Admin, signed zip comes back, `package` consumes signed zip.

### Custom Builders

Distributed provisioning separates identity approval from startup-kit assembly. `cert
request` and `cert approve` do not read custom builders.

The participant definition file uses the same top-level structure as `project.yaml` —
`name`, `description`, `participants`, and an optional `builders:` section — with the
addition of distributed-provisioning endpoint fields (the `server:` block in client and
user definitions) that `nvflare package` uses to generate startup configs. When a participant definition contains custom builders,
`nvflare package` honors them when building the startup kit. The assumption is that custom
builders operate on a single participant's artifacts and do not require information about
other participants — in signed-zip mode, only the signed participant is built.

```yaml
name: hospital_federation

participants:
  - name: hospital-a
    type: client
    org: hospital_alpha
    server:
      host: server1.hospital-central.org
      fed_learn_port: 8002
      admin_port: 8003

builders:
  - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
    args:
      template_file: master_template.yml
  - path: com.example.CustomAuditBuilder
```

If no `builders:` section is present, `package` uses the default FLARE builders.

---

## Project Profile File

**Owner:** Project Admin — kept exclusively by the Project Admin. It is never distributed
to site admins. Site admins receive `scheme` and the project default
`connection_security` through the signed zip returned after approval, avoiding an extra
out-of-band distribution step.

**Purpose:** Defines the project-wide identity and default connection parameters. This is a
lightweight file — not the full `project.yaml` used in centralized provisioning. `name`
uses the same field name and convention as `project.yaml`. `connection_security` follows
the existing project-level default semantics. `scheme` is a new top-level field introduced
for distributed provisioning; in centralized provisioning, scheme is a `StaticFileBuilder`
arg or a `listening_host` sub-field, not a top-level project field.

**Fields:**

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Project name. All requests must match this exactly. |
| `scheme` | Yes | FLARE communication driver. Allowed values: `grpc`, `tcp`, `http`. Matches the supported driver names in FLARE's communication layer. |
| `connection_security` | Yes | Default connection security. e.g. `tls`, `mtls`, `clear` |

**Example (`project_profile.yaml`):**

```yaml
name: hospital_federation
scheme: grpc
connection_security: tls
```

`scheme` and `connection_security` are independent. For example, `scheme: grpc` may be
paired with `connection_security: tls`, `mtls`, or `clear`; the selected
`connection_security` value in the project profile is the authority for generated startup
configs.

**Rationale for name `project_profile.yaml`:** Avoids confusion with the full `project.yaml`
used in centralized provisioning, while making clear it is project-scoped, not
site-scoped. Existing fields keep the same names and allowed values where they overlap
with `project.yaml`; distributed-only fields are limited to the project-wide connection
defaults needed by this workflow.

---

## Participant Definition Files

> **Design note:** Participant definition files reuse existing `project.yaml` participant
> fields for identity and participant-local properties wherever those fields already exist.
> Each file is scoped to a single participant so the Project Admin does not need to collect
> a full federation `project.yaml` before approval.
>
> Two categories of additions are made for distributed provisioning:
>
> 1. **`connection_security`** (server definition only, optional override) — a site-local
>    packaging value resolved by `nvflare package`. It is not approved or resolved by the
>    Project Admin.
> 2. **`server` block** (client and user definitions only) — distributed-provisioning
>    endpoint data used by `nvflare package` to generate client/admin startup configs.
>    Centralized provisioning does not need this block because the server participant is
>    available in the same full `project.yaml`.
>
> `scheme` in `project_profile.yaml` is also new as a top-level field. In centralized
> provisioning, scheme is a `StaticFileBuilder` arg or a `listening_host` sub-field — not a
> top-level project field. It is placed at the top level of `project_profile.yaml` because
> it is a project-wide decision that must be communicated back to all sites through the
> signed zip.

### Server Participant Definition (e.g. `server.yaml`)

**Owner:** Server site admin

**Purpose:** Describes the server participant only. Uses the same field names and value
conventions as `project.yaml` (participants section), but contains only the server entry.
No `scheme` field — scheme is authoritative from `project_profile.yaml`.
`connection_security` is optional and overrides the project default if provided; the server
admin is responsible for ensuring compatibility with clients.

**Structure mirrors `project.yaml` participants entry for `type: server`, minus builders
and scheme.**

**Project-level fields:**

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Project name; must exactly match `project_profile.yaml` `name` |
| `description` | No | Optional human-readable description of the server or federation |

**Participant fields (server entry):**

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Server name using the same validation convention as centralized `project.yaml` server participants. A DNS name is recommended for production, and `localhost` remains valid for local/demo workflows. Used as the certificate CN, the primary SAN, and the default hostname clients connect to unless `default_host` is set. Also used as the request folder name and startup kit directory name, so it must be a valid directory name on the host OS. |
| `type` | Yes | Must be `server` |
| `org` | Yes | Organization name |
| `fed_learn_port` | Yes | Port for federated learning traffic (e.g. `8002`) |
| `admin_port` | Yes | Port for admin console connections (e.g. `8003`) |
| `host_names` | No | Additional hostnames or IP addresses added as Subject Alternative Names (SANs) in the server certificate. Allows clients to connect via multiple addresses (e.g. internal IP, load-balancer hostname, localhost). Each entry is auto-detected as a DNS name or IP address. |
| `default_host` | No | Override the default hostname used in the server certificate. Useful when clients should connect through a hostname different from the participant `name`. |
| `connection_security` | No | Server-side override of the project default. The server deployment may be behind a proxy, load balancer, or ingress where TLS terminates before traffic reaches the FLARE server. The server admin owns this local deployment decision. |

**Example:**

```yaml
name: hospital_federation           # project name — must match project_profile.yaml
description: Central FL server for hospital network

participants:
  - name: server1.hospital-central.org   # server name — primary hostname and cert CN
    type: server
    org: hospital_central
    fed_learn_port: 8002
    admin_port: 8003
    host_names:                          # optional: additional SANs in the server cert
      - 10.0.1.50                        #   IP address SAN
      - fl-server.internal               #   internal DNS SAN
    # default_host: server-public.hospital-central.org  # optional connection hostname override
    connection_security: mtls   # optional: overrides project_profile.yaml default
```

### Client Participant Definition (e.g. `site-1.yaml`, `hospital-a.yaml`)

**Owner:** Client site admin

**Purpose:** Describes a single client participant. Uses the same field names and value
conventions as `project.yaml` participants entry for `type: client`. No `scheme` field and
no `connection_security` field — both are determined by the project admin's
`project_profile.yaml` and delivered through the signed zip.

Client connection security is intentionally not overridable. The client initiates the
federation connection and must use the project admin-approved `scheme` and
`connection_security` from `project_profile.yaml` / `signed.json`. The client has no
deployment-layer flexibility comparable to the server (no proxy, LB, or ingress sits in
front of a client). `tcp` is not typical over the internet.

**Project-level fields:**

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Project name; must exactly match `project_profile.yaml` `name` |
| `description` | No | Optional human-readable description |

**Participant fields (client entry):**

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Participant name — any meaningful identifier for this client site (e.g. `site-1`, `hospital-a`, `org-west`). Not required to be a hostname. Used as the participant identity in the certificate and startup kit, and as the request folder and startup kit directory name, so it must be a valid directory name on the host OS. |
| `type` | Yes | Must be `client` |
| `org` | Yes | Organization name. Identifies which organization this client belongs to. |
| `server.host` | Yes | Hostname/address of the FL server this client connects to, using the same host-name validation convention as centralized provisioning. Used by `nvflare package` to generate the client startup config. Not present in centralized `project.yaml` because the server participant is in the same file. |
| `server.fed_learn_port` | Yes | Federated learning port on the server. |
| `server.admin_port` | Yes | Admin port on the server. |
| `listening_host` | No | Not supported by the v2 distributed request/approve/package flow yet. The field is rejected because v2 signs one participant CSR and does not yet return the extra server-side listener certificate/key pair needed for 3rd-party integration. Use centralized provisioning for participants that need listener certificates. |

**Example (`hospital-a.yaml`):**

```yaml
name: hospital_federation           # project name — must match project_profile.yaml
description: Site A — Hospital Alpha

participants:
  - name: hospital-a                # organization participant name — unique site identity
    type: client
    org: hospital_alpha
    server:
      host: server1.hospital-central.org
      fed_learn_port: 8002
      admin_port: 8003
    # listening_host is intentionally omitted. Use centralized provisioning if this
    # client needs a 3rd-party listener certificate/key pair.
```

### User Participant Definition (e.g. `alice.yaml`)

**Owner:** Individual user (admin, lead, or member)

**Purpose:** Describes a single user identity. Uses the same field names and value
conventions as the `admin` participant entry in `project.yaml`. No `scheme` or
`connection_security` fields — those are connection-layer concerns, not user identity.

**Project-level fields:**

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Project name; must exactly match `project_profile.yaml` `name` |
| `description` | No | Optional human-readable description |

**Participant fields (user entry):**

| Field | Required | Description |
|---|---|---|
| `name` | Yes | User email address (e.g. `alice@hospital.org`). Must be a valid email. |
| `type` | Yes | Must be `admin` (same as `project.yaml` convention) |
| `org` | Yes | Organization name |
| `role` | Yes | User certificate role. Allowed values: `org_admin`, `lead`, `member` |
| `server.host` | Yes | Hostname/address of the FL server this user connects to, using the same host-name validation convention as centralized provisioning. Used by `nvflare package` to generate the admin startup config. Not present in centralized `project.yaml` admin entries because the server participant is in the same file. |
| `server.fed_learn_port` | Yes | Federated learning port on the server. |
| `server.admin_port` | Yes | Admin port on the server. |

**Role mapping:**

| `role` | Certificate type | Notes |
|---|---|---|
| `org_admin` | `org_admin` | Organization-level administration |
| `lead` | `lead` | Can submit and manage own jobs |
| `member` | `member` | View-only access |

**Example (`alice.yaml`):**

```yaml
name: hospital_federation           # project name — must match project_profile.yaml

participants:
  - name: alice@hospital-alpha.org  # must be an email address
    type: admin
    org: hospital_alpha
    role: lead
    server:
      host: server1.hospital-central.org
      fed_learn_port: 8002
      admin_port: 8003
```

---

## Data Flow

### Artifact Ownership

| Location | Owns | Must not own |
|----------|------|--------------|
| Requester machine | Private key, request folder, returned signed zip, generated startup kit | Project CA private key |
| Project Admin machine | Project CA, `project_profile.yaml`, received request zip, signed zip, approval audit | Requester private key |
| Out-of-band channel | Request zip and signed zip | Any `*.key` file |
| Runtime host | Generated startup kit | Request zip internals unless retained for audit |

### Step 0: Project Admin Creates the Project Profile and Initializes the CA

```yaml
name: hospital_federation
scheme: grpc
connection_security: tls
```

```bash
nvflare cert init --profile project_profile.yaml -o ./ca
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
- `project_profile.yaml` stays exclusively with the Project Admin; `scheme` and the project
  default `connection_security` reach participants only through the signed zip.

### Step 1: Requester Creates a Request

Each requester creates a participant definition file describing their identity and
connection fields, then runs:

```bash
nvflare cert request --participant hospital-a.yaml
```

For a server:

```bash
nvflare cert request --participant server.yaml
```

For a user:

```bash
nvflare cert request --participant alice.yaml
```

Requester machine writes one request folder named after the identity. For the client
example (`hospital-a`):

```text
hospital-a/
  site.yaml
  request.json
  hospital-a.key
  hospital-a.csr
  hospital-a.request.zip
```

For the user example, the same structure is written under `alice@hospital-alpha.org/` with
`alice@hospital-alpha.org.key`, `alice@hospital-alpha.org.csr`, and
`alice@hospital-alpha.org.request.zip`.

**Input mapping from participant definition file to request fields:**

| Request metadata | Source in participant definition file |
|---|---|
| project | top-level `name` field |
| organization | `participants[0].org` |
| `kind` (site/server/user) | derived from `participants[0].type` |
| `name` (participant name) | `participants[0].name` |
| `role` (user only) | `participants[0].role` |

Data flow:

- `hospital-a.key` is generated locally and never leaves the requester machine.
- `hospital-a.csr` contains public key, subject identity, organization, and optional
  certificate-role hint.
- `site.yaml` in the local request folder is the full normalized participant definition
  used later by `package`; it may include non-secret package-time connection fields.
- `site.yaml` inside the request zip is a sanitized approval copy generated from the local
  definition. It contains identity, certificate-approval fields, and endpoint fields needed
  by other participants to connect. **Server-side `connection_security` overrides are
  local deployment decisions and must not appear in the sanitized approval copy — they are
  never approved by the Project Admin and are applied at package time from the local
  participant definition only.**
- `request.json` is generated in the request folder and placed into
  `hospital-a.request.zip`.
- The CLI records request metadata under centralized local state for audit and key
  lookup, for example `~/.nvflare/cert_requests/<request_id>/`.

The request audit record stores:

- a full request metadata snapshot
- request folder path
- request zip path
- private key path
- CSR path
- local `site.yaml` path
- CSR, sanitized request `site.yaml`, request zip, and public-key hashes

The request audit record must not copy or embed the private key. It stores only the
private-key path and a key/public-key hash so `package` can locate and validate the local
key later.

Output should make the generated artifact explicit:

```text
request_zip: hospital-a/hospital-a.request.zip
private_key: hospital-a/hospital-a.key
next_step: Send hospital-a.request.zip to the Project Admin.
```

The user does not manually assemble this zip. `cert request` creates it and excludes the
private key by construction.

### Step 2: Requester Sends the Request Zip

The requester transfers only:

```text
hospital-a.request.zip
```

The transport mechanism is outside NVFLARE: email, ticket system, secure file share, or
other operational channel. The design assumes the request zip can arrive on a different
machine with no access to the requester's filesystem.

### Step 3: Project Admin Approves the Request

On the Project Admin machine:

```bash
nvflare cert approve hospital-a.request.zip --ca-dir ./ca --profile project_profile.yaml
```

The command reads:

```text
hospital-a.request.zip
ca/rootCA.key
ca/rootCA.pem
ca/ca.json
project_profile.yaml
```

The command validates:

- The zip does not contain absolute paths or `..` path traversal entries.
- `request.json`, `site.yaml`, and exactly one CSR are present.
- No private key is present in the request zip; `approve` rejects the request if any
  `*.key` file is present.
- Hashes in `request.json` match the CSR and metadata files.
- CSR subject fields match `request.json` and `site.yaml`.
- The request project matches the CA metadata and root CA subject.
- `project` in `request.json` exactly matches `name` in `project_profile.yaml` —
  fails with error code `PROJECT_PROFILE_MISMATCH` if mismatched.
- Requested `kind` and certificate role map to an allowed certificate type.
- The Project Admin explicitly chose to approve this request.

The command writes:

```text
hospital-a.signed.zip
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
name: hospital-a
org: hospital_alpha
kind: site
cert_role: null
cert_type: client
project: hospital_federation
csr_sha256: ...
signed_zip: hospital-a.signed.zip
rootca_fingerprint_sha256: SHA256:...
next_step: Return hospital-a.signed.zip to the requester.
```

### Step 4: Project Admin Returns the Signed Zip

The Project Admin transfers only:

```text
hospital-a.signed.zip
```

The transport mechanism is the same as for the request zip: email, ticket system, secure
file share, or other operational channel. The signed zip does not contain the private key
and may be sent over a non-secure channel. Its integrity is protected by the certificate
chain back to the root CA and by the hashes in `signed.json`.

The signed zip contains:

```text
signed.json        # approval metadata: scheme, default_connection_security, hashes
site.yaml          # sanitized approval metadata copied from the request
hospital-a.crt     # signed participant certificate
rootCA.pem         # project root CA certificate
```

No private key is included. `project_profile.yaml` is never distributed — its
`scheme` and `default_connection_security` values are copied into `signed.json` at
approval time so the requester receives them without needing the profile file.

The Project Admin also shares the `rootca_fingerprint_sha256` value from the
`cert approve` output through a separate trusted channel (email, ticket, verbal
confirmation, etc.) so the requester can independently verify the root CA identity
before packaging.

### Root CA Fingerprint Verification

`rootca_fingerprint_sha256` is the OpenSSL-style SHA256 certificate fingerprint
of the `rootCA.pem` certificate:

```text
SHA256:AA:BB:...
```

The fingerprint is useful only when compared against a value received outside
the signed zip transfer path. The signed zip already contains `rootCA.pem`, so
`nvflare package` can always compute and display the same fingerprint, but that
does not independently prove who sent the zip.

`nvflare package` therefore supports two explicit verification modes:

- `--expected-rootca-fingerprint <fingerprint>` is the automation path. The
  command fails if the signed zip's `rootCA.pem` fingerprint does not match the
  expected value.
- `--confirm-rootca` is the interactive path. The command prints the signed
  zip's `rootCA.pem` fingerprint and asks the requester to confirm they compared
  it with the Project Admin's out-of-band value.

If neither option is supplied, `nvflare package` still validates signed zip
structure, hashes, identity metadata, certificate chain, and local private-key
match, then includes `rootca_fingerprint_sha256` in the output. It does not
prompt by default because the CLI has no independent value to compare against.

### Step 5: Requester Packages the Startup Kit

On the requester machine:

```bash
nvflare package hospital-a.signed.zip --confirm-rootca
```

Connection info — scheme, security, and server address — comes from `signed.json` and the
full participant definition stored in the local request folder. All connection
configuration is resolved from these two sources.

The command reads:

```text
hospital-a.signed.zip
hospital-a/hospital-a.key
hospital-a/site.yaml               (full local participant definition, for server host/port)
```

The command finds the private key by:

1. Explicit override: `--request-dir <dir>`.
2. Centralized local request state created by `cert request`, for example
   `~/.nvflare/cert_requests/<request_id>/`.
3. A request folder next to the signed zip, such as `./hospital-a/`.

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
- Identity fields in the local participant definition match the approved metadata.
- Local endpoint fields that are present in the signed approval `site.yaml` match the local
  request-folder `site.yaml`. If host or port values changed after approval, packaging fails
  and the requester must regenerate the request and signed zip.
- Package-time fields intentionally excluded from the signed zip, such as custom builders and
  the server-side `connection_security` override, remain local site configuration.

**`connection_security` resolution is role-specific:**

- **Client and user:** `connection_security` = `default_connection_security` from
  `signed.json`. No local override is applied. The client must match the
  federation-approved security.
- **Server:** resolved in priority order:
  1. `connection_security` from the local server definition in the request folder
  2. `default_connection_security` from `signed.json` (project admin's default)
  3. NVFlare built-in default

`scheme` is read exclusively from `signed.json` — the project admin's value is
authoritative for all participants.

Server host and port for client and user kits come from the `server` block in the
participant definition.

`package` may materialize returned approval files into the request folder:

```text
hospital-a/
  site.yaml
  request.json
  hospital-a.key
  hospital-a.csr
  hospital-a.request.zip
  signed.json
  hospital-a.crt
  rootCA.pem
  hospital-a.signed.zip
```

Then it writes the startup kit:

```text
workspace/hospital_federation/prod_NN/hospital-a/
  startup/
    start.sh
    fed_client.json
    client.key
    client.crt
    rootCA.pem
  local/
  transfer/
```

For a server startup kit:

```text
workspace/hospital_federation/prod_NN/server1.hospital-central.org/
  startup/
    start.sh
    fed_server.json      # server config: scheme, host, fed_learn_port, admin_port, host_names
    server.crt
    server.key           # permissions: 0600
    rootCA.pem
  local/
  transfer/
```

For a user startup kit:

```text
workspace/hospital_federation/prod_NN/alice@hospital-alpha.org/
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
cd workspace/hospital_federation/prod_00/hospital-a
./startup/start.sh
```

For a user:

```bash
cd workspace/hospital_federation/prod_00/alice@hospital-alpha.org
./startup/fl_admin.sh
```

---

## Bundle Model

### Local Request Folder

`cert request` creates a local request folder. The folder is not a startup kit. It is the
local identity material used to request and later package a startup kit.

For `hospital-a`:

```text
hospital-a/
  site.yaml
  request.json
  hospital-a.key
  hospital-a.csr
  hospital-a.request.zip
```

For `alice@hospital-alpha.org`:

```text
alice@hospital-alpha.org/
  site.yaml
  request.json
  alice@hospital-alpha.org.key
  alice@hospital-alpha.org.csr
  alice@hospital-alpha.org.request.zip
```

The private key stays in this folder and is never included in the request zip.

### Request Zip

The request zip is what the requester sends to the Project Admin.

Example `hospital-a.request.zip`:

```text
request.json
site.yaml
hospital-a.csr
```

`site.yaml` in the request zip is a sanitized approval copy generated from the local
participant definition. It contains identity, certificate-approval fields, and the server
endpoint information needed for participants to connect. The full local participant
definition remains in the request folder and is used later by `package`. Server-side
`connection_security` overrides are excluded from this copy — they are local deployment
decisions, not Project Admin-approved fields, and are applied at package time from the
local participant definition.

For a site (client):

```yaml
name: hospital_federation

participants:
  - name: hospital-a
    type: client
    org: hospital_alpha
    server:
      host: server1.hospital-central.org
      fed_learn_port: 8002
      admin_port: 8003
```

For a server:

```yaml
name: hospital_federation

participants:
  - name: server1.hospital-central.org
    type: server
    org: hospital_central
    fed_learn_port: 8002
    admin_port: 8003
    host_names:
      - 10.0.1.50
      - fl-server.internal
```

For a user:

```yaml
name: hospital_federation

participants:
  - name: alice@hospital-alpha.org
    type: admin
    org: hospital_alpha
    role: org_admin
    server:
      host: server1.hospital-central.org
      fed_learn_port: 8002
      admin_port: 8003
```

`request.json` contains metadata needed for review, matching, and error messages:

```json
{
  "artifact_type": "nvflare.cert.request",
  "schema_version": "1",
  "request_id": "8ff2d9e7-6f89-4acb-96e2-fc2d2fb8c6f7",
  "created_at": "2026-04-24T00:00:00Z",
  "project": "hospital_federation",
  "name": "hospital-a",
  "org": "hospital_alpha",
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

Example `hospital-a.signed.zip`:

```text
signed.json
site.yaml
hospital-a.crt      # signed participant certificate
rootCA.pem          # project root CA certificate
```

`signed.json` contains enough information for `nvflare package` to match the approval
to the local request. The Project Admin injects `scheme` and `default_connection_security`
from `project_profile.yaml` into `signed.json`. These are project-wide defaults only. The
Project Admin does not validate or resolve local package-time fields — `signed.json` does
not carry the final server-resolved value.

```json
{
  "artifact_type": "nvflare.cert.signed",
  "schema_version": "1",
  "request_id": "8ff2d9e7-6f89-4acb-96e2-fc2d2fb8c6f7",
  "approved_at": "2026-04-24T00:00:00Z",
  "project": "hospital_federation",
  "name": "hospital-a",
  "org": "hospital_alpha",
  "kind": "site",
  "cert_type": "client",
  "cert_role": null,
  "scheme": "grpc",
  "default_connection_security": "tls",
  "certificate": {
    "serial": "<hex>",
    "valid_until": "2029-04-24T00:00:00Z"
  },
  "cert_file": "hospital-a.crt",
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

Field sources:
- `scheme` — from project admin's `project_profile.yaml`; authoritative for all participants
- `default_connection_security` — from project admin's `project_profile.yaml`; used as
  fallback by `nvflare package` if the participant definition provides no override (server only;
  clients and users always use this value)
- `site_yaml_sha256` — hash of the sanitized approval `site.yaml` in the request/signed zip.
  Local-only package fields excluded from this copy are validated from the local request
  folder during packaging.

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
alice@hospital-alpha.org.request.zip
alice@hospital-alpha.org.signed.zip
```

The real identity is also stored in `request.json` and `signed.json`. If a future
platform-specific issue requires normalized filenames, metadata remains the source of
truth and can preserve the original identity.

---

## Comparison: Centralized vs. Distributed Provisioning

| | Centralized (`nvflare provision`) | Distributed (`request` / `approve` / `package`) |
|---|---|---|
| **Private key custody** | Project Admin generates and distributes | Requester generates locally; never leaves the machine |
| **Data distributed to site** | Full startup kit (keys, certs, config, scripts) | Signed zip with `signed.json`, signed cert, `rootCA.pem`, and sanitized participant metadata; no private key |
| **Data sent from site** | Nothing | Request zip with CSR + metadata, no key |
| **Steps for Project Admin** | One command provisions all sites | Approve one request zip per participant |
| **Steps for requester** | Unzip and run | Request -> send zip -> receive zip -> package -> run |
| **Participant info required upfront** | All participants before any kit is generated | Each participant joins independently, on demand |
| **Adding a new site** | Dynamic provisioning with existing root CA | Same workflow; no impact on existing sites |
| **Endpoint changes** | Rebuild or edit kit configuration | Coordinated participant configuration update. Re-package only if existing certificate identities remain valid; otherwise re-request/re-approve affected certificates. |
| **CC deployments** | Supported | Not supported |
| **HE deployments** | Supported | Not supported (future) |
| **Trust required in Project Admin** | Must trust Project Admin with your private key | Project Admin never sees private keys |

---

## Assumptions and Non-Goals

### Trust Model

This document covers standard (non-Confidential Computing) deployments.

- **Certificate chain and configured connection security are the trust boundary.** In mTLS
  mode, a participant is authorized if and only if its certificate chains to the project
  root CA. In TLS or clear server-side deployment modes, the project default and site-local
  security settings must be consistent with the deployment topology. No startup kit
  integrity signature (`signature.json`) is needed for the certificate-chain check.
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
standard PKI deployments proceed with the configured connection security as the trust boundary.

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
public CLI workflow is `request` / `approve` / `package`.

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
| `nvflare cert init ...` | Project Admin initializes the CA (one-time per federation). |
| `nvflare cert request -p/--participant <path>` | Create local key, CSR, metadata, and request zip from a participant definition file. |
| `nvflare cert approve <request.zip> --ca-dir <dir> --profile <project_profile.yaml>` | Project Admin signs a request zip and creates a signed zip. Validates project name against profile. |
| `nvflare package <signed.zip> [-w <workspace>] [--request-dir <dir>] [--expected-rootca-fingerprint <fp> | --confirm-rootca]` | Requester combines signed approval with local private key and creates startup kit. Connection info comes from the signed zip and local participant definition. |

The implementation should reuse existing CSR generation, certificate signing, and package
builder logic behind these commands. Public help and customer docs should expose this final
workflow: `cert request`, `cert approve`, and signed-zip `package`.

---

## Existing Deployment Compatibility

Distributed provisioning is additive. Runtime and security changes are designed to leave
existing deployments intact:

| Deployment | Behavior after changes |
|------------|----------------------|
| Centralized provisioning (non-CC, non-HE) | `nvflare provision` remains the centralized provisioning workflow. Standard PKI kits may omit `signature.json`; runtime behavior is unchanged because the certificate-based connection security was already the trust anchor. |
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

1. **`cert request`** - the requester creates a participant definition declaring identity kind
   and, for user certificates, a certificate role:

   ```bash
   nvflare cert request --participant hospital-a.yaml
   nvflare cert request --participant alice.yaml    # alice.yaml has role: lead
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
| `type=client` | `client` | Federated runtime client/site. |
| `type=server` | `server` | FL server identity. |
| `type=admin, role=org_admin` | `org_admin` | User certificate role. Name must be email. |
| `type=admin, role=lead` | `lead` | User certificate role. Name must be email. |
| `type=admin, role=member` | `member` | User certificate role. Name must be email. |

The signed certificate's embedded certificate type remains authoritative at runtime. The
user-facing type is CLI vocabulary derived from the participant definition. The certificate role
must not be treated as study authorization.

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

The certificate role should not be reused as the study authorization model.

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

1. **Configured secure connection** - in mTLS mode, the server accepts any client whose
   certificate chains to `rootCA.pem`. No server-side allowlist, restart, or reconfiguration
   is required. The Project Admin approving the request zip is the act of authorization.

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
nvflare cert init --profile <project_profile.yaml> -o <ca-dir>
```

Produces:

```text
<ca-dir>/rootCA.key
<ca-dir>/rootCA.pem
<ca-dir>/ca.json
```

`rootCA.key` must stay with the Project Admin. `cert init` reads the explicit
project profile path and uses only the profile `name` as the root CA subject.
It does not discover or search for profile files automatically. `cert approve`
validates the profile `scheme` and `connection_security` fields later.

### `nvflare cert request` - Requester

```bash
nvflare cert request -p <path> [--out <dir>]
nvflare cert request --participant <path> [--out <dir>]
```

`-p/--participant` takes a path to a single-participant definition file. It does not take
the participant name directly.

Examples:

```bash
nvflare cert request --participant server.yaml
nvflare cert request --participant hospital-a.yaml
nvflare cert request --participant alice.yaml
```

All identity fields — project name, org, participant name, type, and role — are read from
the participant definition file. The command produces:

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

- Participant definition must be a valid server, client, or user participant definition.
- Server `name` follows the same validation convention as centralized provisioning and must be a valid directory name.
- Client `name` must be a valid directory name on the host OS.
- User `name` must be an email address and a valid directory name.
- `role` in user yamls must be one of `org_admin`, `lead`, `member`. `project_admin` is
  not a valid user cert role.
- Project names must be path-safe identifiers matching `[A-Za-z0-9][A-Za-z0-9._-]*`.

### `nvflare cert approve` - Project Admin

```bash
nvflare cert approve <request-zip> --ca-dir <ca-dir> --profile <project_profile.yaml> [--out <signed-zip>] [--valid-days <days>]
```

Example:

```bash
nvflare cert approve hospital-a.request.zip --ca-dir ./ca --profile project_profile.yaml
```

| Argument | Required | Description |
|---|---|---|
| `<request.zip>` | Yes | Request zip received from the site admin |
| `--ca-dir` | Yes | Directory containing `rootCA.key`, `rootCA.pem`, and `ca.json` |
| `--profile` | Yes | `project_profile.yaml` — authoritative source for project name, scheme, and default `connection_security` |
| `--out` | No | Output path for the signed zip. Default: same directory as request zip, named `<name>.signed.zip` |
| `--valid-days` | No | Certificate validity in days. Default: 1095 (3 years) |

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

It does not include the private key. `signed.json` includes `scheme` and
`default_connection_security` injected from `project_profile.yaml`.

The command output includes `rootca_fingerprint_sha256`; the Project Admin
shares this value through a trusted out-of-band channel.

### `nvflare package` - Requester

```bash
nvflare package <signed-zip> [-w <workspace>] [--request-dir <dir>] [--expected-rootca-fingerprint <fingerprint> | --confirm-rootca]
```

Examples:

```bash
nvflare package hospital-a.signed.zip --confirm-rootca
nvflare package hospital-a.signed.zip --expected-rootca-fingerprint SHA256:AA:BB:...
nvflare package server1.hospital-central.org.signed.zip --confirm-rootca
```

Connection info comes from two sources:
- `signed.json`: `scheme`, `default_connection_security` (project admin defaults)
- Participant definition in the request folder: server gets host/ports from its own
  definition; client/user get server host/ports from the `server` block in their definition

Server endpoint changes are coordinated configuration changes, not casual package-time
overrides. Re-packaging is sufficient only when the existing certificates remain valid for
the configured hostnames. If the server certificate identity or SANs must change, affected
certificates must be requested and approved again.

`package` reads the signed zip, finds the matching local private key, validates the
certificate and metadata, and writes:

```text
<workspace>/<project>/prod_NN/<name>/
```

`package` is selected-participant provisioning: only the signed participant is built.
It does not require the requester to provide the full federation `project.yml`, and it
must not re-provision other participants.

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
- **Project profile is kept exclusively by the Project Admin.** It is never distributed to
  site admins. Site admins receive `scheme` and the project default
  `connection_security` through the signed zip returned after approval, avoiding an extra
  out-of-band distribution step.
- **Clients cannot override `connection_security`.** The client initiates the federation
  connection and must match the federation-approved protocol. Only the server may override
  the project default to accommodate proxy, load balancer, or ingress deployments.
- **Server `connection_security` overrides are local and never Project Admin-approved.**
  The server-side override stays in the local participant definition and is applied by
  `nvflare package`. It is excluded from the sanitized approval `site.yaml` placed in the
  request zip and signed zip.
- **`project_admin` is not a valid user cert role.** Valid roles in participant definitions are
  `org_admin`, `lead`, and `member`. `project_admin` is an operational concept, not a
  certificate role.
- **Project name mismatch error: `PROJECT_PROFILE_MISMATCH`.** Follows the existing
  `PROJECT_CA_MISMATCH` convention. Both values are shown so the Project Admin can
  immediately see which side is wrong:
  ```
  Request project 'hospital_federation' does not match profile project 'cancer_research'.
  Hint: Use a request zip generated for project 'cancer_research', or approve with the correct project_profile.yaml.
  ```
- **`scheme` is a new top-level field in `project_profile.yaml`.** In centralized
  provisioning, scheme is a `StaticFileBuilder` arg or a `listening_host` sub-field, not a
  top-level project field. It is placed at the top level of `project_profile.yaml` because
  it is a project-wide decision that must be communicated back to all sites through the
  signed zip.
