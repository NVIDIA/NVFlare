# Distributed Provisioning — Revised Design (Draft)

> **Status:** Temporary design draft. To be merged into `distributed_provisioning.md` once reviewed.

---

## Overview of Changes

This revision introduces consistency in connection protocols and project identity across all participants
by separating project-wide settings (owned by the project admin) from site-specific settings (owned by
each site admin). It also tightens the approval workflow to enforce project name matching.

---

## 1. Project Profile File (`project_profile.yaml`)

**Owner:** Project admin — kept exclusively by the project admin. It is never distributed to site
admins. Site admins receive `scheme` and `connection_security` only through the signed zip returned
after approval, avoiding an extra out-of-band distribution step.

**Purpose:** Defines the project-wide identity and default connection parameters. This is a lightweight
file — not the full `project.yaml` used in centralized provisioning. `name` uses the same field name
and convention as `project.yaml`. `scheme` and `connection_security` are new top-level fields
introduced for distributed provisioning — in centralized provisioning, scheme is a `StaticFileBuilder`
arg or a `listening_host` sub-field, not a top-level project field.

**Fields:**

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Project name. All requests must match this exactly. |
| `scheme` | Yes | FLARE communication driver. Allowed values: `grpc`, `tcp`, `http`. Matches the supported driver names in FLARE's communication layer. |
| `connection_security` | Yes | Default connection security. e.g. `tls`, `mtls`, `clear` |

**Example:**

```yaml
name: hospital_federation
scheme: grpc
connection_security: tls
```

**Rationale for name `project_profile.yaml`:** Avoids confusion with the full `project.yaml` used in
centralized provisioning, while making clear it is project-scoped (not site-scoped). The fields are a
strict subset of `project.yaml` using the same names and allowed values.

---

## 2. Participant YAML Files

> **Design note:** Participant fields in the participant YAML files (`server.yaml`, client yamls, user yamls)
> must be identical to the corresponding participant fields in `project.yaml` used by centralized
> provisioning — same field names, same allowed values, same semantics. The participant yamls are a structural
> subset of `project.yaml`, scoped to a single participant.
>
> Two categories of additions are made for distributed provisioning:
>
> 1. **`connection_security`** (server yaml only, optional override) — not in `project.yaml`; introduced
>    here as a site-local field resolved at packaging time.
> 2. **`server` block** (client and user yamls) — provides server host and ports; not in `project.yaml`
>    client/admin entries because centralized provisioning reads the server participant from the same file.
>
> `scheme` in `project_profile.yaml` is also new as a top-level field. In centralized provisioning,
> scheme is a `StaticFileBuilder` arg or a `listening_host` sub-field — not a top-level project field.
> It is placed at the top level of `project_profile.yaml` because it is a project-wide decision that
> must be communicated back to all sites through the signed zip.

### 2a. Server Participant YAML (e.g. `server.yaml`)

**Owner:** Server site admin

**Purpose:** Describes the server participant only. Uses the same field names and value conventions as
`project.yaml` (participants section), but contains only the server entry. No `scheme` field — scheme
is authoritative from `project_profile.yaml`. `connection_security` is optional and overrides the
project default if provided; the server admin is responsible for ensuring compatibility with clients.

**Structure mirrors `project.yaml` participants entry for `type: server`, minus builders and scheme.**

**Project-level fields:**

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Project name; must exactly match `project_profile.yaml` `name` |
| `description` | No | Optional human-readable description of the server or federation |

**Participant fields (server entry):**

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Server FQDN (e.g. `server1.example.com`). Must be a fully qualified domain name — not an IP address. Used as the certificate CN, the primary SAN, and the hostname clients connect to. Also used as the request folder name and startup kit directory name, so it must be a valid directory name on the host OS. IP addresses go in `host_names`. |
| `type` | Yes | Must be `server` |
| `org` | Yes | Organization name |
| `fed_learn_port` | Yes | Port for federated learning traffic (e.g. `8002`) |
| `admin_port` | Yes | Port for admin console connections (e.g. `8003`) |
| `host_names` | No | Additional hostnames or IP addresses added as Subject Alternative Names (SANs) in the server certificate. Allows clients to connect via multiple addresses (e.g. internal IP, load-balancer hostname, localhost). Each entry is auto-detected as a DNS name or IP address. |
| `default_host` | No | Override the default hostname used in the server certificate. Useful when `name` exceeds the 64-character CN limit — set `name` to the truncated value and `default_host` to the full FQDN. |
| `connection_security` | No | Server-side override of the project default. The server deployment may be behind a proxy, load balancer, or ingress where TLS terminates before traffic reaches the FLARE server (e.g. HTTP + TLS to LB, clear from LB to FLARE; or gRPC with TCP forwarding and mTLS at the FLARE server). The server admin owns this local deployment decision. |

**Example:**

```yaml
name: hospital_federation           # project name — must match project_profile.yaml
description: Central FL server for hospital network

participants:
  - name: server1.hospital-central.org   # FQDN — primary hostname and cert CN
    type: server
    org: hospital_central
    fed_learn_port: 8002
    admin_port: 8003
    host_names:                          # optional: additional SANs in the server cert
      - 10.0.1.50                        #   IP address SAN
      - fl-server.internal               #   internal DNS SAN
    # default_host: server1.hospital-central.org  # only needed if name is truncated
    connection_security: mtls   # optional: overrides project_profile.yaml default
```

### 2b. Client Participant YAML (e.g. `site-1.yaml`, `hospital-a.yaml`)

**Owner:** Client site admin

**Purpose:** Describes a single client participant. Uses the same field names and value conventions as
`project.yaml` participants entry for `type: client`. No `scheme` field and no `connection_security`
field — both are determined by the project admin's `project_profile.yaml` and delivered through the
signed zip.

Client connection security is intentionally not overridable. The client initiates the federation
connection and must match the federation-approved protocol. For `http`, the client uses TLS to the
reachable endpoint. For `grpc`, the client uses mTLS. The client has no deployment-layer flexibility
comparable to the server (no proxy, LB, or ingress sits in front of a client). `tcp` is not typical
over the internet.

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
| `server.host` | Yes | FQDN of the FL server this client connects to. |
| `server.fed_learn_port` | Yes | Federated learning port on the server. |
| `server.admin_port` | Yes | Admin port on the server. |
| `listening_host` | No | Hostname for 3rd-party integration. If set, a server-side cert/key pair is generated for this client so it can accept incoming connections. Must be reachable by the external trainer. |
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
    # listening_host: hospital-a.internal   # optional: for 3rd-party integration
```

---

### 2c. User Participant YAML (e.g. `alice.yaml`, `lead.yaml`)

**Owner:** Individual user (admin, lead, or member)

**Purpose:** Describes a single user identity. Uses the same field names and value conventions as the
`admin` participant entry in `project.yaml`. No `scheme` or `connection_security` fields — those are
connection-layer concerns, not user identity.

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
| `server.host` | Yes | FQDN of the FL server this user connects to. |
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

## 3. Request Generation (Server, Client, and User)

> **Implementation note:** The request process is identical to the existing `nvflare cert request`
> implementation. The **only change** is how inputs are supplied: instead of reading `--org`,
> `--project`, kind, and name from CLI arguments, these values are read from the participant yaml file.
> Everything downstream — CSR generation, `request.json` and `site.yaml` creation, zip assembly,
> audit record, and output — is unchanged.

**Input mapping from participant yaml to existing request fields:**

| Existing CLI arg | Source in participant yaml |
|---|---|
| `--project` | top-level `name` field |
| `--org` | `participants[0].org` |
| `kind` (site/server/user) | derived from `participants[0].type` |
| `name` (participant name) | `participants[0].name` |
| `role` (user only) | `participants[0].role` |

### Examples

**Server** — run by the server site admin:

```bash
nvflare cert request --participant-yaml server.yaml
```

`server.yaml` provides: project name (`hospital_federation`), server FQDN
(`server1.hospital-central.org`), org, kind (`server`), and connection fields
(`fed_learn_port`, `admin_port`, `host_names`, `connection_security`).
The connection fields are stored in the request folder for use during packaging.

**Client** — run by each client site admin:

```bash
nvflare cert request --participant-yaml hospital-a.yaml
```

`hospital-a.yaml` provides: project name, participant name (`hospital-a`), org, kind (`client`).

**User** — run by each individual user:

```bash
nvflare cert request --participant-yaml alice.yaml
```

`alice.yaml` provides: project name, participant name (`alice@hospital-alpha.org`), org,
kind (`user`), and role (`lead`).

### Generated Output

The command creates a **request folder** named after the participant, in the current working directory
by default. The folder location can be overridden with `--out <dir>`.

**Request folder layout** (example for server `server1.hospital-central.org`):

```text
server1.hospital-central.org/
  server1.hospital-central.org.key      # private key — never leaves this machine
  server1.hospital-central.org.csr      # certificate signing request
  request.json                          # request metadata (project, name, org, kind, hashes)
  site.yaml                             # site identity metadata included in the request zip
  server1.hospital-central.org.request.zip   # the only file sent to the project admin
```

The same layout applies to client and user requests, with the participant name as the folder and
file prefix (e.g. `hospital-a/hospital-a.key`, `alice@hospital.org/alice@hospital.org.key`).

**Files sent to the project admin** — request zip only:

```text
server1.hospital-central.org.request.zip
  request.json
  site.yaml
  server1.hospital-central.org.csr
```

The private key (`*.key`) is never included in the zip.

**Audit record** — written locally, not shared:

```text
~/.nvflare/cert_requests/<request-id>/audit.json
```

Contains a full metadata snapshot, file paths, and hashes. Does not contain or copy the private key.

---

## 4. Approval Workflow

### Examples

**Approve a server request:**

```bash
nvflare cert approve server1.hospital-central.org.request.zip \
    --ca-dir ./ca \
    --profile project_profile.yaml
```

**Approve a client request:**

```bash
nvflare cert approve hospital-a.request.zip \
    --ca-dir ./ca \
    --profile project_profile.yaml
```

**Approve with a custom output path:**

```bash
nvflare cert approve hospital-a.request.zip \
    --ca-dir ./ca \
    --profile project_profile.yaml \
    --out ./signed/hospital-a.signed.zip
```

| Argument | Required | Description |
|---|---|---|
| `<request.zip>` | Yes | Request zip received from the site admin |
| `--ca-dir` | Yes | Directory containing `rootCA.key`, `rootCA.pem`, and `ca.json` |
| `--profile` | Yes | **(New)** `project_profile.yaml` — authoritative source for project name, scheme, and default connection_security |
| `--out` | No | Output path for the signed zip. Default: same directory as request zip, named `<name>.signed.zip` |
| `--valid-days` | No | Certificate validity in days. Default: 1095 (3 years) |

### Validations Performed

All existing validations are unchanged. The new project-name check is added:

1. Zip security: no absolute paths, no `..` traversal, no `*.key` files present
2. Required members present: `request.json`, `site.yaml`, exactly one `*.csr`
3. Hashes in `request.json` match the CSR and `site.yaml`
4. CSR subject fields match `request.json` and `site.yaml`
5. **(New)** `project` in `request.json` exactly matches `name` in `project_profile.yaml` — fails with explicit error if mismatched
6. Project matches CA metadata (`ca.json`)
7. `kind` and `cert_role` map to an allowed certificate type

### Generated Output

**Signed zip** (sent back to the site admin):

Default location: same directory as the input request zip.

```text
<name>.signed.zip
  signed.json        # approval metadata
  site.yaml          # copy of site identity from the request
  <name>.crt         # signed participant certificate
  rootCA.pem         # project root CA certificate
```

The signed zip does not contain any private key.

**`signed.json` contents:**

The project admin injects `scheme` and `default_connection_security` from `project_profile.yaml`
into `signed.json`. These are project-wide defaults only. The project admin never sees or resolves
site-specific overrides — `signed.json` does not carry the final server-resolved value.

For clients and users, `default_connection_security` is the final value used at packaging time.
For the server, `nvflare package` may override it with the local `connection_security` from
`server.yaml` to accommodate proxy, load balancer, or ingress deployments.

```json
{
  "artifact_type": "nvflare.cert.signed",
  "schema_version": "1",
  "request_id": "<uuid>",
  "approved_at": "<ISO-8601 UTC>",
  "project": "<project-name>",
  "name": "<participant-name>",
  "org": "<org>",
  "kind": "server | site | user",
  "cert_type": "<cert-type>",
  "cert_role": "<role or null>",
  "scheme": "grpc",
  "default_connection_security": "tls",
  "certificate": {
    "serial": "<hex>",
    "valid_until": "<ISO-8601 UTC>"
  },
  "cert_file": "<name>.crt",
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
- `default_connection_security` — from project admin's `project_profile.yaml`; used as fallback by `nvflare package` if the participant yaml provides no override

**Audit record** — written locally on the project admin machine, not shared:

```text
~/.nvflare/cert_approves/<request-id>.json
```

Contains full approval metadata, request and signed zip paths, CA metadata (excluding `rootCA.key`),
certificate serial, validity timestamps, and all file hashes.

**CLI output** shown to the project admin on success:

```json
{
  "name": "<participant-name>",
  "project": "<project-name>",
  "signed_zip": "<path/to/name.signed.zip>",
  "request_id": "<uuid>",
  "rootca_fingerprint_sha256": "SHA256:AA:BB:...",
  "audit": "<path/to/audit.json>",
  "next_step": "Return <name>.signed.zip to the requester."
}
```

The `rootca_fingerprint_sha256` must be shared with the requester through a trusted out-of-band
channel so they can verify the root CA when running `nvflare package`.

---

## 5. Startup Kit Package Generation

### Examples

**Package a server startup kit** (interactive root CA confirmation):

```bash
nvflare package server1.hospital-central.org.signed.zip --confirm-rootca
```

**Package a client startup kit:**

```bash
nvflare package hospital-a.signed.zip --confirm-rootca
```

**Package a user startup kit:**

```bash
nvflare package alice@hospital-alpha.org.signed.zip --confirm-rootca
```

**Non-interactive (automation), with explicit root CA fingerprint:**

```bash
nvflare package hospital-a.signed.zip \
    --expected-rootca-fingerprint SHA256:AA:BB:CC:...
```

**With explicit request folder and custom workspace:**

```bash
nvflare package hospital-a.signed.zip \
    --request-dir ./hospital-a \
    -w /opt/nvflare/workspace \
    --confirm-rootca
```

| Argument | Required | Description |
|---|---|---|
| `<name>.signed.zip` | Yes | Signed zip returned by the project admin |
| `--request-dir` | No | Local request folder containing the private key. Default: auto-discovered by `request_id` from `~/.nvflare/cert_requests/` or a folder named `<name>/` next to the signed zip |
| `-w / --workspace` | No | Workspace root. Output goes to `<workspace>/<project>/prod_NN/<name>/`. Default: `workspace` |
| `--confirm-rootca` | No | Interactive prompt to confirm the root CA fingerprint was verified out-of-band |
| `--expected-rootca-fingerprint` | No | Non-interactive root CA verification for automation |

### Input Sources

The package command assembles the startup kit from three sources:

| Source | Provides |
|---|---|
| `<name>.signed.zip` | Signed certificate, `rootCA.pem`, `scheme`, `default_connection_security` (project admin defaults) |
| Request folder (`<name>/`) | Private key, original full participant yaml |
| Site yaml — server | `name` (FQDN), `fed_learn_port`, `admin_port`, `host_names`, `default_host`, optional `connection_security` override |
| Site yaml — client / user | `server.host`, `server.fed_learn_port`, `server.admin_port` |

`scheme` is read exclusively from `signed.json` — the project admin's value is authoritative for all participants.

`connection_security` resolution is role-specific:

- **Client and user:** `connection_security` = `default_connection_security` from `signed.json`. No local override is applied. The client must match the federation-approved security.
- **Server:** resolved in priority order:
  1. `connection_security` from `server.yaml` (local deployment override — project admin never sees it)
  2. `default_connection_security` from `signed.json` (project admin's default)
  3. NVFlare built-in default

Server host and port for client and user kits come from the `server` block in the participant yaml —
there is no `-e` endpoint argument.

### Generated Output

Output location: `<workspace>/<project-name>/prod_NN/<name>/`

A new `prod_NN` directory is created for each `nvflare package` run (or when `--force` is used
to re-package an existing participant).

**Server startup kit:**

```text
<workspace>/<project>/prod_00/server1.hospital-central.org/
  startup/
    start.sh
    fed_server.json      # server config: scheme, host, fed_learn_port, admin_port, host_names
    server.crt           # signed server certificate
    server.key           # private key  (permissions: 0600)
    rootCA.pem
  local/
  transfer/
```

**Client startup kit:**

```text
<workspace>/<project>/prod_00/hospital-a/
  startup/
    start.sh
    fed_client.json      # client config: scheme, server host and port
    client.crt
    client.key           # (permissions: 0600)
    rootCA.pem
  local/
  transfer/
```

**User (admin) startup kit:**

```text
<workspace>/<project>/prod_00/alice@hospital-alpha.org/
  startup/
    fl_admin.sh
    fed_admin.json       # admin config: scheme, server host and port
    client.crt
    client.key           # (permissions: 0600)
    rootCA.pem
  local/
  transfer/
```

**CLI output** shown to the site admin on success:

```json
{
  "output_dir": "<workspace>/<project>/prod_NN/<name>",
  "name": "<participant-name>",
  "type": "server | client | org_admin | lead | member",
  "endpoint": "<scheme>://<host>:<port>",
  "cert": "<path>/startup/<server|client>.crt",
  "key": "<path>/startup/<server|client>.key  (permissions: 0600)",
  "rootca": "<path>/startup/rootCA.pem",
  "next_step": "cd <output_dir> && ./startup/<start.sh|fl_admin.sh>"
}
```

---

## Decisions

**Project name mismatch at approval** — error code `PROJECT_PROFILE_MISMATCH`, following the
existing `PROJECT_CA_MISMATCH` convention:

```
Request project 'hospital_federation' does not match profile project 'cancer_research'.
Hint: Use a request zip generated for project 'cancer_research', or approve with the correct project_profile.yaml.
```

Both values are shown so the project admin can immediately see which side is wrong.

## Open Questions

None.
