# FLARE Simplified Enrollment System

## Executive Summary

This document describes a simplified enrollment system for NVIDIA FLARE that reduces the complexity of participant onboarding while maintaining strong security.

The current provisioning workflow requires the Project Admin to generate complete startup kits for every participant, securely distribute them (including private keys), and manually coordinate when adding new participants. Two new workflows eliminate centralized startup kit generation. Both are based on the same CSR (Certificate Signing Request) model: private keys are always generated locally and never transmitted. The difference is how the CSR signing is orchestrated.

### 1. Manual Workflow (5-10 participants)

No additional infrastructure required. The Project Admin signs CSRs locally using the root CA private key. CSRs and signed certificates are exchanged out-of-band (email, USB, secure file transfer). There are no tokens, no Certificate Service, and no automated enrollment.

| Step | Who | Action |
|------|-----|--------|
| 1 | **Site Admin** | Generate private key + CSR locally (`nvflare cert csr`) |
| 2 | **Site Admin** | Send CSR to Project Admin via email/secure channel (private key **never** leaves the site) |
| 3 | **Project Admin** | Initialize root CA if not yet done (`nvflare cert init`), sign the CSR locally (`nvflare cert sign`), return signed certificate + rootCA.pem + Server URI back via email/secure channel |
| 4 | **Site Admin** | Generate startup kit locally (`nvflare package --cert ... --rootca ...`) with the signed certificate |
| 5 | **Site Admin** | Start (`./startup/start.sh`) |

**What makes this different:** No infrastructure to deploy. No tokens. No service. The Project Admin holds the root CA key on their own machine and signs CSRs manually. The exchange is human-to-human.

### 2. Auto-Scale Workflow (10+ participants or dynamic scaling)

Same CSR-based security model, but fully automated via a **Certificate Service** + **enrollment tokens**. The root CA private key lives on the Certificate Service (not with the Project Admin). Sites never interact with humans for certificate signing -- they present a token to the service and get a signed certificate back programmatically.

| Step | Who | Action |
|------|-----|--------|
| 1 | **Project Admin** | Initialize root CA and deploy Certificate Service -- one-time setup |
| 2 | **Project Admin** | Generate enrollment token for the invited participant (`nvflare token generate`) |
| 3 | **Project Admin** | Send the token to the Site Admin (token only, no keys or certs) |
| 4 | **Site Admin** | Generate startup kit locally (`nvflare package --token ...`) |
| 5 | **Site Admin** | Start (`./startup/start.sh`) -- the client automatically: |
|   |                |  a. Generates private key + CSR locally |
|   |                |  b. Submits CSR to Certificate Service over HTTPS (token authorizes signing) |
|   |                |  c. Receives signed certificate and connects to FL server |

**What makes this different:** Requires deploying a Certificate Service, but eliminates all manual CSR signing. New participants are added by simply issuing tokens. Supports dynamic K8s scaling where pods auto-enroll at startup.

**Key Benefits (both workflows):**

1. **Private keys generated locally**: Never transmitted over the network
2. **No centralized participant gathering**: Project Admin no longer needs to collect all participants' information before provisioning
3. **Lightweight distribution**: Only a signed certificate (Manual) or a short token (Auto-Scale) needs to be delivered

**Complexity Comparison:**

| Scenario | Current Provisioning | Manual Workflow | Auto-Scale Workflow |
| --- | --- | --- | --- |
| Add 1 new client | 4 steps | 3 steps (CSR + sign + package) | **1 command** |
| Add 100 clients (K8s) | 100 kits + distribution | 100 CSRs + sign + distribute certs | **1 batch token + deploy** |
| Site operator setup | Receive kit, extract, start | Generate CSR, receive signed cert, run package, start | **1 command + start** |
| Project admin effort | Manage all kits | Sign CSRs, return cert + rootCA + Server URI | **Generate tokens only** |
| Infrastructure needed | None | None | Certificate Service |
| Root CA key location | Project Admin machine | Project Admin machine | Certificate Service |
| Human interaction | Distribute full kits | Exchange CSR + signed cert (email/USB) | Send token string |

### Quick Start

**Manual Workflow**

```bash
# Site Admin: generate private key and CSR locally
nvflare cert csr -n hospital-1 -t client -o ./csr
# Email hospital-1.csr to Project Admin (private key stays local)
# Project Admin saves the received file as ./incoming/hospital-1.csr

# Project Admin: sign the CSR locally
nvflare cert init -n "Project" -o ./ca    # one-time
nvflare cert sign -r ./incoming/hospital-1.csr -c ./ca -o ./signed
# Email back: signed client.crt + rootCA.pem + Server URI (grpc://server:8002)

# Site Admin: generate startup kit with the signed certificate
# Current compatibility command:
nvflare package -n hospital-1 -e grpc://server:8002 -t client \
    --cert ./signed/client.crt --rootca ./signed/rootCA.pem

# UX simplification target (design intent):
# nvflare package -n hospital-1 -e grpc://server:8002 -t client --cert-dir ./signed
# (single folder input; auto-discover cert/key/rootCA with conventional filenames)

# Ideal target with defaults:
# nvflare package
# (when standard files and environment defaults are present)
cd hospital-1 && ./startup/start.sh
```

**Auto-Scale Workflow**

```bash
# Project Admin: Deploy Certificate Service (one-time), then generate tokens
nvflare token generate -n hospital-1 \
    --cert-service https://cert-service.example.com:8443 \
    --api-key $API_KEY
# Send token + Certificate Service URL to Site Admin

# Site Admin: Package with embedded enrollment info (1 command)
nvflare package -n hospital-1 -e grpc://server:8002 -t client \
    --cert-service https://cert-service.example.com:8443 \
    --token "eyJhbGciOiJSUzI1NiIs..."
cd hospital-1 && ./startup/start.sh  # Auto-enrolls!
```

For K8s deployments, inject `NVFLARE_CERT_SERVICE_URL` and `NVFLARE_ENROLLMENT_TOKEN` via a K8s Secret (`envFrom: secretRef`) instead of embedding at package time. Environment variables take priority over embedded files.

## Background

### The Provisioning Problem

Today, NVIDIA FLARE uses `nvflare provision` to create startup kits containing certificates and private keys for all participants. This requires a `project.yml` defining all participants upfront, running `nvflare provision`, and distributing each kit (containing private keys) via a secure channel.

| Limitation | Description |
| --- | --- |
| **Private keys in transit** | Keys are generated centrally and must be distributed securely |
| **Manual distribution** | Each startup kit must be sent via secure channel |
| **Static pre-shared trust** | All participants must be known upfront; adding new ones requires manual dynamic-provisioning |
| **Scalability** | Difficult to manage 100+ participants with manual distribution |
| **DevOps expertise required** | Configuration errors lead to cryptic failures |

### How the New Design Addresses These Concerns

| Concern | Manual Workflow | Auto-Scale Workflow |
| --- | --- | --- |
| **Adding client mid-project** | Site Admin generates CSR locally, emails it to Project Admin who signs and returns cert. No re-provisioning. | Project Admin generates a token. Site auto-enrolls at startup with the Certificate Service. |
| **Private keys in transit** | Keys generated locally, never transmitted (only CSR/signed cert exchanged via email) | Keys generated locally, never transmitted |
| **Manual distribution** | Exchange CSR and signed cert + rootCA only (smaller than full kits) | **Distribute tokens only** (short strings, any channel) |
| **Static pre-shared trust** | New participants added by signing their CSR on demand | **Dynamic enrollment** -- sites join on-demand with tokens |
| **Scalability** | Practical for 5-10 participants | **Scales to 100+ participants** with automated enrollment |
| **DevOps expertise** | Simple CLI commands, no infrastructure | Simple CLI + one-time Certificate Service setup |

### PKI Concepts

- **Root CA**: `rootCA.pem` (public, distributed to all) + `rootCA.key` (private, signs certificates)
- **CSR**: Certificate Signing Request containing a public key; private key never leaves the requestor
- **JWT**: JSON Web Token with signed claims; used for enrollment tokens with embedded approval policies (Auto-Scale workflow only)

## Workflows

Both workflows share the same CSR-based security model: private keys are always generated locally and never leave the site. The difference is how the CSR signing is orchestrated -- manually by the Project Admin (Manual) or automatically by the Certificate Service using a token sent only to invited participants (Auto-Scale).

### Workflow 1: Manual (Small Scale)

For 5-10 participants. No Certificate Service needed. The Project Admin holds the root CA key locally and signs CSRs by hand. All exchanges happen out-of-band (email, USB, etc.).

```
Step 1: Site Admin              Step 2: Exchange             Step 3: Project Admin          Step 4: Site Admin
──────────────────              ────────────────             ─────────────────────          ──────────────────

Generate private key            Send CSR only                Receive CSR                    Receive signed cert
and CSR locally:                (via email/USB;              (no private key):              + rootCA.pem
                                private key stays                                           + Server URI
nvflare cert csr \              local):                      nvflare cert sign \            (via email/USB)
   -n hospital-1 \                                             -r hospital-1.csr \
   -t client \               hospital-1.csr                    -c ./ca \                   Generate startup kit:
   -o ./csr                  ════════════════►                  -o ./signed
                                                                   │                       nvflare package \
Output:                         ◄════════════════              signed/                        -n hospital-1 \
  ./csr/                        Send back:                     client.crt                     -e grpc://server:8002 \
  hospital-1.key                • client.crt                   rootCA.pem                     -t client \
  (STAYS LOCAL)                 • rootCA.pem                                                  --cert client.crt \
  hospital-1.csr                • Server URI                                                  --rootca rootCA.pem
                                  (host + ports)
                                                                                            cd hospital-1 && \
                                                                                              ./startup/start.sh
```

**Key characteristics:**
- No tokens, no Certificate Service, no automated enrollment
- Root CA key stays on the Project Admin's machine
- Human-to-human exchange of CSR and signed certificate (email, USB, etc.)
- Site Admin runs `nvflare package` with the received certificate files
- Startup is immediate -- certificates already in place, no enrollment at runtime

### Workflow 2: Auto-Scale (Large Scale)

For 10+ participants or dynamic environments. Requires a Certificate Service. The root CA private key lives on the Certificate Service. Token-based authorization replaces human CSR exchange.

```
Phase 1: Setup (Project Admin, one-time)
────────────────────────────────────────
1. nvflare cert init -n "Project" -o ./ca
2. Deploy Certificate Service with rootCA.pem + rootCA.key

Phase 2: Generate Tokens (Project Admin, per site)
──────────────────────────────────────────────────
nvflare token generate -n hospital-1 \
    --cert-service https://cert-service.example.com \
    --api-key $API_KEY

Distribute to each site: token string + Certificate Service URL

Phase 3: Site Enrollment (Site Admin)
─────────────────────────────────────
nvflare package -n hospital-1 -e grpc://server:8002 -t client \
    --cert-service https://cert-service.example.com \
    --token "eyJhbGciOiJSUzI1NiIs..."

cd hospital-1 && ./startup/start.sh
(Auto-enrollment happens at startup:
  1. Client generates key + CSR locally
  2. Submits CSR to Certificate Service with token
  3. Receives signed cert, connects to FL server)
```

**Key characteristics:**
- Token-based: Project Admin sends a short token string, not certificates
- Fully automated: no human in the CSR signing loop
- Root CA key lives on the Certificate Service, not with Project Admin
- `nvflare package` embeds enrollment info; `start.sh` auto-enrolls
- K8s-friendly: inject token via Secret, pods auto-enroll at startup

**K8s deployment:** Generate package without enrollment info, inject `NVFLARE_CERT_SERVICE_URL` and `NVFLARE_ENROLLMENT_TOKEN` from a K8s Secret at runtime (`envFrom: secretRef`). For production, use an External Secrets Operator to manage token lifecycle.

## Design Overview (Auto-Scale Workflow)

The following sections describe the components specific to the Auto-Scale workflow. The Manual workflow requires only the `nvflare cert` and `nvflare package` CLI commands -- no service, no tokens, no store.

### Architecture

```
                    +-------------------+
                    |   PROJECT ADMIN   |
                    | nvflare token     |
                    | generate/batch    |
                    +---------+---------+
                              | HTTPS API
                              v
+-------------------------------------------------------------------------------+
|                     CERTIFICATE SERVICE                                       |
|  POST /api/v1/token   - Token generation                                     |
|  POST /api/v1/enroll  - CSR signing                                          |
|  GET  /api/v1/ca-cert - Public root CA                                       |
|  GET  /api/v1/pending - List pending requests (admin)                        |
|                                                                               |
|  CertService (core)  |  TokenService (JWT)  |  rootCA.key                    |
+-------------------------------------------------------------------------------+
                              | HTTPS (TLS)
              +---------------+---------------+
              v               v               v
      +-------------+ +-------------+ +-------------+
      |  FL Server  | |  FL Client  | |  FL Client  |
      +-------------+ +-------------+ +-------------+
       1. Gen keys locally
       2. Create CSR
       3. Submit to Cert Service (with token)
       4. Receive signed cert
```

### Key Components

| Component | Location | Responsibility |
| --- | --- | --- |
| **TokenService** | nvflare/tool/enrollment/token_service.py | Generate JWT enrollment tokens with embedded policies |
| **CertService** | nvflare/cert_service/cert_service.py | Validate tokens, evaluate policies, sign CSRs |
| **CertServiceApp** | nvflare/cert_service/app.py | HTTP wrapper exposing CertService via REST API |
| **CertRequestor** | nvflare/security/enrollment/cert_requestor.py | Client-side: generate keys, create CSR, submit for signing |
| **EnrollmentStore** | nvflare/cert_service/store.py | Persistence for enrolled entities and pending requests |

## Enrollment Token (Auto-Scale Only)

Enrollment tokens are JWTs signed with the root CA private key. They are used **only** in the Auto-Scale workflow. The Manual workflow does not use tokens.

### Token Structure

```json
{
  "header": { "alg": "RS256", "typ": "JWT" },
  "payload": {
    "jti": "550e8400-e29b-41d4-a716-446655440000",
    "sub": "hospital-1",
    "subject_type": "client",
    "iss": "MyProjectRootCA",
    "iat": 1704067200,
    "exp": 1704672000,
    "policy": {
      "site": { "name_pattern": "^hospital-.*" },
      "approval": {
        "method": "policy",
        "rules": [{ "name": "auto-approve", "match": {}, "action": "approve" }]
      }
    }
  },
  "signature": "..."
}
```

| Claim | Description |
| --- | --- |
| jti | Unique token identifier (UUID) |
| sub | Participant name or pattern |
| subject_type | client, server, relay, or user |
| iss | Issuer (from root CA certificate) |
| iat / exp | Issued-at and expiration timestamps |
| policy | Embedded approval policy |
| roles | Roles for user/admin identities (when applicable) |

Example (user token excerpt):

```json
{
  "sub": "admin@org.com",
  "subject_type": "user",
  "roles": ["lead"]
}
```

### Token Security

| Property | How It's Achieved |
| --- | --- |
| Cannot be forged | Signed with root CA private key (RS256) |
| Cannot be tampered | Signature verification detects modification |
| Time-limited | Expiration time (exp claim) |
| Single-use | Token is invalid after one enrollment submission; no token re-use |
| Scoped | Subject (sub) specifies who can use it |

| Attack | Mitigation |
| --- | --- |
| Token theft | Short expiry + name binding + single-use |
| Token forgery | RS256 signature verification |
| Replay attack | Single-use enforcement + expiration |
| Brute force | UUID-based JTI + rate limiting |

### Token Signing Key

By default, the root CA private key signs both participant certificates and JWT tokens. For advanced deployments, a separate JWT signing key pair can be used for key rotation or security isolation. Both TokenService and CertService must share the same JWT key pair.

## Approval Policy (Auto-Scale Only)

The approval policy is embedded in the token and evaluated by the Certificate Service during enrollment. Rules are evaluated in order (first match wins).

```yaml
metadata:
  project: "my-fl-project"
  version: "1.0"

token:
  validity: "7d"

site:
  name_pattern: "^hospital-[0-9]+$"

user:
  allowed_roles: [lead, member, org_admin]
  default_role: lead

approval:
  method: policy
  rules:
    - name: auto_approve_hospitals
      match:
        site_name_pattern: "^hospital-.*"
      action: approve
      log: true

    - name: ip_restricted_sites
      match:
        site_name_pattern: "^datacenter-.*"
        source_ips: ["10.0.0.0/8", "192.168.1.0/24"]
      action: approve

    - name: reject_unknown
      action: reject
      message: "Site name not recognized"
```

| Action | Description |
| --- | --- |
| approve | Automatically approve and sign the certificate |
| reject | Reject the enrollment request |
| pending | Queue for manual admin approval via CLI |

### Policy Evaluation Flow

```
Enrollment Request
      │
      ▼
┌─────────────────────┐
│ Validate JWT Token  │ ──► Invalid? ──► Reject (401)
└──────────┬──────────┘
           │ Valid
           ▼
┌─────────────────────┐
│ Check subject_type  │ ──► Mismatch? ──► Reject
│ matches enrollment  │
└──────────┬──────────┘
           │ Match
           ▼
┌─────────────────────┐
│ Check name matches  │ ──► Mismatch? ──► Reject
│ token subject       │
└──────────┬──────────┘
           │ Match
           ▼
┌─────────────────────┐
│ Evaluate policy     │
│ rules (first match) │
└──────────┬──────────┘
           │
   ┌───────┴───────┐
   │               │
approve         reject
   │               │
   ▼               ▼
Sign CSR      Return error
```

## CSR Signing Process

Both workflows use the same CSR-based model, but the signing step differs fundamentally:

**Manual Workflow:** Project Admin runs `nvflare cert sign` locally with the root CA key. No network involved. The CSR arrives via email/USB.

**Auto-Scale Workflow:** CertRequestor submits CSR to Certificate Service over HTTPS with a token. CertService validates the token, evaluates the approval policy, verifies the CSR signature, and signs the certificate with the root CA key.

**Certificate Contents (both workflows):**

| Field | Value |
| --- | --- |
| Common Name (CN) | Participant name (e.g., "hospital-1") |
| Organization (O) | Organization name (optional) |
| Organizational Unit (OU) | Participant type (client, server, relay, user) -- new in enrollment cert profile |
| Unstructured Name | Role for user/admin tokens (legacy-compatible fallback) |

Backward-compatibility note: existing provisioned certificates may not carry OU in the same way. Enrollment consumers should remain tolerant by falling back to existing identity fields (including unstructured name where applicable).

## Certificate Service (Auto-Scale Only)

### Overview

The Certificate Service is a standalone HTTP service deployed separately from the FL Server. It is used **only** in the Auto-Scale workflow. The Manual workflow does not require any service.

Reasons for separation from FL Server:

1. **Security isolation**: Root CA private key is not on FL Server
2. **Scalability**: Can handle many concurrent enrollments
3. **Audit**: Centralized logging of all certificate issuance
4. **Blast radius**: If FL Server is compromised, attacker cannot issue certs

### HTTP API

| Endpoint | Auth | Description |
| --- | --- | --- |
| `POST /api/v1/enroll` | Token | Submit CSR for signing. Returns certificate (200), pending (202), or error |
| `GET /api/v1/enroll/{request_id}` | Token | Optional polling endpoint for external automation; standard client flow does not poll |
| `POST /api/v1/token` | API Key | Generate enrollment tokens (single or batch) |
| `GET /api/v1/pending` | API Key | List pending requests (filter by `?type=`) |
| `POST /api/v1/pending/{name}/approve` | API Key | Approve single request |
| `POST /api/v1/pending/approve_batch` | API Key | Bulk approve by pattern |
| `POST /api/v1/pending/{name}/reject` | API Key | Reject single request |
| `POST /api/v1/pending/reject_batch` | API Key | Bulk reject by pattern |
| `GET /api/v1/enrolled` | API Key | List enrolled entities |
| `GET /api/v1/ca-cert` | None | Download public root CA certificate |
| `GET /api/v1/ca-info` | None | Root CA information |
| `GET /health` | None | Health check |

Admin API key: generate with `nvflare cert api-key`, configure via `NVFLARE_API_KEY` env var or config file. Sent as `Authorization: Bearer <api-key>`.

### Configuration

```yaml
server:
  host: 0.0.0.0
  port: 8443
  tls:
    cert: /path/to/service.crt    # Or use reverse proxy for TLS termination
    key: /path/to/service.key

api_key: "your-api-key-here"       # Or NVFLARE_API_KEY env var
data_dir: /var/lib/cert_service    # Root CA auto-generated here on first start
project_name: "NVFlare"

ca:
  cert: /var/lib/cert_service/rootCA.pem   # FLARE root CA (separate from service TLS)
  key: /var/lib/cert_service/rootCA.key

policy:
  file: /path/to/approval_policy.yaml

storage:
  type: sqlite                     # sqlite (default) | postgresql (production/HA)
  path: /var/lib/cert_service/enrollment.db

pending:
  timeout: 604800                  # 7 days
  cleanup_interval: 3600

audit:
  enabled: true
  log_file: /var/log/cert_service/audit.log
```

The service TLS certificate (`server.tls`) is for HTTPS and is separate from the FLARE root CA (`ca`), which is used only for signing FL participant certificates. On first start, if rootCA files don't exist in `data_dir`, they are auto-generated.

### Enrollment Store

Tracks enrolled entities and pending requests via a pluggable storage backend.

**Entity Types:** client, relay, server, user (uniqueness by `(name, entity_type)` pair).

The `EnrollmentStore` ABC provides: `is_enrolled`, `add_enrolled`, `get_enrolled`, `is_pending`, `add_pending`, `get_pending`, `get_all_pending`, `approve_pending`, `reject_pending`, `cleanup_expired`.

| Feature | SQLite (Default) | PostgreSQL |
| --- | --- | --- |
| Deployment | Single node | Multi-node / HA |
| Dependencies | Built-in | Requires psycopg2 |
| Concurrency | Limited | Full ACID |
| Use Case | Dev, small production | Large production |

## Client Enrollment (Auto-Scale Only)

### CertRequestor

Generates a key pair locally, creates a CSR, submits it to the Certificate Service with the enrollment token, and saves the returned certificate. This component is used only in the Auto-Scale workflow. In the Manual workflow, the Site Admin generates keys and CSR via `nvflare cert csr` and receives a signed certificate from the Project Admin out-of-band.

```python
from nvflare.security.enrollment import CertRequestor, EnrollmentIdentity, EnrollmentOptions

requestor = CertRequestor(
    cert_service_url="https://cert-service.example.com",
    enrollment_token="eyJhbGciOiJSUzI1NiIs...",
    identity=EnrollmentIdentity.for_client("hospital-1", org="Hospital A"),
    options=EnrollmentOptions(output_dir="/workspace/startup"),
)
result = requestor.request_certificate()
# result.cert_path, result.key_path, result.ca_path
```

### Runtime Source Resolution

This section separates two different concerns:

1. **Enrollment endpoint inputs** (where to enroll and what credential to use)
2. **Enrollment behavior options** (timeouts/retries)

#### What Is `enrollment.json`?

`enrollment.json` is an optional startup-time config file at:

- `startup/enrollment.json`

It is typically created by `nvflare package --cert-service <URL>` and is used by auto-enrollment to resolve Certificate Service connection details.

Typical content:

```json
{
  "cert_service_url": "https://cert-service.example.com:8443",
  "timeout": 30.0,
  "max_retries": 3,
  "retry_delay": 5.0
}
```

Notes:

- `cert_service_url` belongs to deployment/runtime setup, not part of existing runtime client config.
- `timeout`, `max_retries`, and `retry_delay` can be sourced from multiple places (table below).
- If environment variables are present, they override file values.

#### Enrollment Endpoint Inputs

These endpoint inputs are deployment-time values and often vary by environment. They are resolved only from runtime sources (environment variables and startup files).

| Input | Priority 1 (highest) | Priority 2 |
| --- | --- | --- |
| cert_service_url | `NVFLARE_CERT_SERVICE_URL` env | `startup/enrollment.json` (`cert_service_url`) |
| enrollment_token | `NVFLARE_ENROLLMENT_TOKEN` env | `startup/enrollment_token` file |

#### Enrollment Behavior Options

These behavior options are resolved from runtime enrollment sources only and remain isolated from existing runtime client configuration.

| Option | Priority 1 (highest) | Priority 2 | Priority 3 |
| --- | --- | --- | --- |
| timeout, max_retries, retry_delay | Env vars | `startup/enrollment.json` | Defaults |

### Enrollment Flow

**Auto-Approved:**

```
CertRequestor                          Certificate Service
     │  1. Generate RSA key pair (locally)
     │  2. Create CSR
     │  3. POST /api/v1/enroll
     │────────────────────────────────────>│
     │                                     │  4. Validate token
     │                                     │  5. Policy → approve
     │                                     │  6. Sign certificate
     │  200: { certificate, ca_cert }      │
     │<────────────────────────────────────│
     │  7. Save client.crt, client.key, rootCA.pem
```

**Pending (Manual Approval via Certificate Service):**

When the approval policy evaluates to `pending`, the Certificate Service stores the request and returns 202. The standard client exits with `EnrollmentPending` (no polling loop). Because tokens are single-use, the original token is not re-used after this submission. After admin approval via `nvflare enrollment approve`, the Site Admin restarts with a newly issued token and receives the signed certificate. The `GET /api/v1/enroll/{request_id}` endpoint is provided for optional external automation/observability, not for the default client startup flow.

Key decisions: no polling loop (client exits immediately), strict single-use tokens (no re-use), server tracks by `(name, entity_type)`, pending requests expire after 7 days.

## Server Enrollment

**Manual Workflow:** Same as client -- Site Admin generates CSR (`nvflare cert csr -t server`), sends to Project Admin for signing via email/USB, receives signed `server.crt` + `rootCA.pem`, runs `nvflare package -t server --cert ... --rootca ...`.

**Auto-Scale Workflow:** Server enrolls the same way as clients, using `EnrollmentIdentity.for_server()`. Output files are `server.crt` and `server.key`.

```bash
# Auto-Scale
nvflare package -n server1 -e grpc://0.0.0.0:8002:8003 -t server \
    --cert-service https://cert-service.example.com:8443 \
    --token eyJhbGciOiJSUzI1NiIs...
cd server1 && ./startup/start.sh  # Auto-enrolls
```

## Security Analysis

### Trust Model

```
             Root CA (rootCA.pem + rootCA.key)
             - Standalone Manual: private key held by Project Admin
             - Standalone Auto-Scale: private key held by Certificate Service
             - Mixed deployment: choose one signing authority model
               for a given root key (or use separate intermediates)
             - Public cert distributed to all participants
                              │
               Signs all certificates
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
  Server Cert           Client Cert            Client Cert
  (signed by Root CA)   (signed by Root CA)    (signed by Root CA)
```

### Key Security Properties

| Property | Manual Workflow | Auto-Scale Workflow |
| --- | --- | --- |
| Private keys never transit | Yes (CSR/cert exchanged via email) | Yes (CSR submitted over HTTPS) |
| Root CA key protected | On Project Admin machine | On Certificate Service (not FL Server) |
| Token security | N/A (no tokens) | RS256 signed, single-use, time-limited |
| mTLS between participants | All certs signed by same root CA | All certs signed by same root CA |
| Audit trail | Manual tracking | Automated logging by Certificate Service |

### Threat Analysis

| Threat | Impact | Mitigation |
| --- | --- | --- |
| Compromised FL Server | Cannot issue new certs (no root CA key in either workflow) | Revoke server cert |
| Stolen token (Auto-Scale) | Attacker could enroll as participant | Short expiry + name binding + policy checks |
| Compromised Cert Service (Auto-Scale) | Could issue arbitrary certs | Network isolation, audit, access controls |
| MITM | Intercept enrollment | TLS (Auto-Scale); secure channel for CSR/cert exchange (Manual) |

## CLI Commands

### nvflare cert (Both Workflows)

Certificate generation and management. Used in both workflows.

| Subcommand | Description |
| --- | --- |
| `init` | Initialize root CA (`rootCA.pem` + `rootCA.key`) |
| `csr` | Generate private key + CSR locally (Site Admin) |
| `sign` | Sign a CSR with root CA locally (Project Admin, Manual workflow) |
| `site` | *(Legacy)* Generate cert directly; prefer `csr` + `sign` |
| `api-key` | Generate API key for Certificate Service (Auto-Scale workflow) |

```bash
# Manual Workflow
nvflare cert init -n "My Project" -o ./ca                      # Project Admin, one-time
nvflare cert csr -n hospital-1 -t client -o ./csr              # Site Admin
nvflare cert sign -r ./csr/hospital-1.csr -c ./ca -o ./signed  # Project Admin
nvflare cert api-key                                           # Auto-Scale only
```

### nvflare token (Auto-Scale Only)

Generate enrollment tokens via the Certificate Service API. Not used in the Manual workflow.

| Subcommand | Description |
| --- | --- |
| `generate` | Generate a single enrollment token |
| `batch` | Generate multiple tokens at once |
| `info` | Inspect and decode a token |

```bash
nvflare token generate -n hospital-1 --type client \
    --cert-service https://cert-service:8443 --api-key "$NVFLARE_API_KEY"

nvflare token batch --pattern "site-{001..100}" --type client \
    --cert-service https://cert-service:8443 --api-key $NVFLARE_API_KEY -o ./tokens/

nvflare token info -t eyJhbGciOiJSUzI1NiIs...
```

### nvflare package (Both Workflows)

Generate startup kits. Usage differs fundamentally between workflows.

For Manual workflow UX, the design direction is to reduce arguments:

1. Prefer a single-folder input (cert/key/rootCA bundle) over multiple certificate flags.
2. Support sensible defaults so `nvflare package` can work with minimal or no extra arguments when conventions are met.

Current compatibility commands:

```bash
# Manual Workflow: package with pre-signed certificates (received from Project Admin)
nvflare package -n hospital-1 -e grpc://server:8002 -t client \
    --cert ./signed/client.crt --rootca ./signed/rootCA.pem

# Proposed simplified Manual workflow UX (design target)
# nvflare package -n hospital-1 -e grpc://server:8002 -t client --cert-dir ./signed

# Auto-Scale Workflow: package with enrollment info (auto-enrolls at startup)
nvflare package -n hospital-1 -e grpc://server:8002 -t client \
    --cert-service https://cert-service:8443 --token eyJhbGciOiJSUzI1NiIs...

# Server package (either workflow)
nvflare package -n server1 -e grpc://0.0.0.0:8002:8003 -t server

# From project file
nvflare package -p project.yml -w ./packages
```

Endpoint formats: `grpc://host:port` (admin port = fl_port + 1) or `grpc://host:fl_port:admin_port`.

### nvflare enrollment (Auto-Scale Only)

Manage pending enrollment requests. Admin CLI for the Certificate Service. Not used in the Manual workflow.

```bash
nvflare enrollment list [--type client|user]
nvflare enrollment info hospital-1 --type client
nvflare enrollment approve hospital-1 --type client
nvflare enrollment approve --pattern "hospital-*" --type client
nvflare enrollment reject hospital-2 --type client --reason "Not authorized"
nvflare enrollment enrolled [--type client|user]
```

Requires `NVFLARE_CERT_SERVICE_URL` and `NVFLARE_API_KEY` environment variables.

### CLI Summary

| Command | Manual Workflow Usage | Auto-Scale Workflow Usage |
| --- | --- | --- |
| `nvflare cert init` | Required (Project Admin) | Required (one-time service setup) |
| `nvflare cert csr` | Required (Site Admin generates CSR) | Not in standard flow (CertRequestor generates CSR automatically) |
| `nvflare cert sign` | Required (Project Admin signs CSR locally) | Not in standard flow (Certificate Service signs CSR) |
| `nvflare cert site` | Optional legacy path (prefer csr + sign) | Not in standard flow |
| `nvflare cert api-key` | Not required | Required (admin auth for service operations) |
| `nvflare token` | Not used | Required (generate/batch/info) |
| `nvflare package` | Required (current: `--cert` + `--rootca`; target: single-folder/default UX) | Required (with `--cert-service` and `--token`) |
| `nvflare enrollment` | Not used | Admin operation (used when pending approvals are enabled) |

## FLARE Integration (Auto-Scale Only)

Auto-enrollment hooks into three components at startup via `_auto_enroll_if_needed()`:

1. **FederatedClientBase** (`nvflare/private/fed/client/fed_client_base.py`)
2. **FederatedServer** (`nvflare/private/fed/server/fed_server.py`)
3. **AdminAPI** (`nvflare/fuel/hci/client/api.py`)

Logic: check for existing certs → look for token (env or file) → look for cert service URL (env or file) → enroll via CertRequestor → update runtime certificate paths.

In the Manual workflow, certificates are already in place before startup, so auto-enrollment is not triggered.

**Key Design Notes:**

- Uses standard HTTP/REST, not CellNet
- No authentication bypass: valid certificates required before connecting to FL Server
- Token grants eligibility to *request* a certificate; the certificate is used for mTLS
- **Additive**: does not modify existing authentication, CellNet, FL training, or job execution

## Backward Compatibility

1. **Existing provisioned deployments** continue to work unchanged
2. **No migration required** -- can adopt gradually
3. **Coexistence** -- some sites provisioned, some enrolled via Manual, some via Auto-Scale

### Key Custody in Coexistence Deployments

To avoid ambiguity in mixed fleets, define key custody explicitly:

- For a given `rootCA.key`, choose one signing authority model and document it.
- If both workflows must operate in the same federation, either:
  - use a controlled shared-signing process (audited key custody), or
  - use separate intermediate signing CAs under a common trust root.
- Do not run independent, uncontrolled copies of `rootCA.key` in multiple systems.

## Future Enhancements

1. **Certificate rotation**: Automatic renewal before expiry
2. **Revocation**: CRL or OCSP support
3. **Dashboard integration**: Web UI for token and enrollment management
4. **Notification webhooks**: Alert admin when requests are pending
