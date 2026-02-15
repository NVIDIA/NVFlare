# FLARE Simplified Enrollment System

## Executive Summary

This document describes a simplified enrollment system for NVIDIA FLARE that reduces the complexity of participant onboarding while maintaining strong security.

The current provisioning workflow requires the Project Admin to generate complete startup kits for every participant, securely distribute them (including private keys), and manually coordinate when adding new participants. Two new workflows eliminate centralized startup kit generation. Both are based on the same CSR (Certificate Signing Request) model: private keys are always generated locally and never transmitted. The difference is how the CSR signing is orchestrated.

### 1. Manual Workflow (5–10 participants)

No additional infrastructure required.

| Step | Who | Action |
|------|-----|--------|
| 1 | **Site Admin** | Generate private key + CSR locally (`nvflare cert csr`) |
| 2 | **Site Admin** | Send CSR to Project Admin (private key **never** leaves the site) |
| 3 | **Project Admin** | Initialize root CA if not yet done (`nvflare cert init`), sign the CSR (`nvflare cert sign`), return signed certificate + rootCA.pem + Server URI (host + port) |
| 4 | **Site Admin** | Generate startup kit locally (`nvflare package`) with the signed certificate |
| 5 | **Site Admin** | Start (`./startup/start.sh`) |

### 2. Auto-Scale Workflow (10+ participants or dynamic scaling)

Same CSR-based model, but automated via a Certificate Service + enrollment tokens.

| Step | Who | Action |
|------|-----|--------|
| 1 | **Project Admin** | Initialize root CA and deploy Certificate Service — one-time setup |
| 2 | **Project Admin** | Generate enrollment token for the invited participant (`nvflare token generate`) |
| 3 | **Project Admin** | Send the token to the Site Admin (token only, no keys or certs) |
| 4 | **Site Admin** | Generate startup kit locally (`nvflare package --token ...`) |
| 5 | **Site Admin** | Start (`./startup/start.sh`) — the client automatically: |
|   |                |  a. Generates private key + CSR locally |
|   |                |  b. Submits CSR to Certificate Service over HTTPS (token authorizes signing) |
|   |                |  c. Receives signed certificate and connects to FL server |

**Trade-off:** Requires deploying and maintaining a Certificate Service, but eliminates manual CSR signing and enables dynamic scaling. New participants are added by simply issuing new tokens.

**Key Benefits:**

1. **Private keys generated locally**: Never transmitted over the network
2. **No centralized participant gathering**: Project Admin no longer needs to collect all participants' information before provisioning
3. **Lightweight distribution**: Startup kits can be large; now only a signed certificate (Manual) or a short token (Auto-Scale) needs to be delivered instead of the full kit

**Design goals addressed:**

This design was created with the following goals in mind:

| Goal | How Addressed |
| --- | ----- |
| **5-minute setup** | Single command for Site Admins: nvflare package --cert-service --token |
| **Easier than provisioning** | No project.yml needed; no distribution of startup kits |
| **Add clients mid-project** | Generate token + send to site. Zero touch for existing participants |
| **Dynamic K8s scaling** | Batch tokens + StatefulSet with auto-enrollment at startup |
| **No DevOps expertise** | Simple CLI commands with sensible defaults |
| **Private keys stay local** | Keys generated on each site, never transmitted |

**Complexity Comparison**

| Scenario | Current Provisioning | Manual Workflow | Auto-Scale Workflow |
| --- | ----- | --- | ----- |
| Add 1 new client | 4 steps | 3 steps (CSR + sign + package) | **1 command** |
| Add 100 clients (K8s) | 100 kits + distribution | 100 CSRs + sign + distribute certs | **1 batch token + deploy** |
| Site operator setup | Receive kit, extract, start | Generate CSR, receive signed cert, run package, start | **1 command + start** |
| Project admin effort | Manage all kits | Sign CSRs, return cert + rootCA + Server URI | **Generate tokens only** |

### Quick Start

**Auto-Scale Workflow (Recommended for 10+ sites)**

```bash
# Project Admin: Deploy Certificate Service, then generate tokens
nvflare token generate -n hospital-1 \
    --cert-service https://cert-service.example.com:8443 \
    --api-key $API_KEY

# Send token to Site Admin

# Site Admin: Package with embedded enrollment info
nvflare package -n hospital-1 -e grpc://server:8002 -t client \
    --cert-service https://cert-service.example.com:8443 \
    --token "eyJhbGciOiJSUzI1NiIs..."

cd hospital-1 && ./startup/start.sh  # Auto-enrolls!
```

**Alternative: K8s / Container Deployment**

```bash
# Site Admin: Generate package without enrollment info
nvflare package -n hospital-1 -e grpc://server:8002 -t client
```

```yaml
# K8s: Store token and cert-service URL in a K8s Secret
apiVersion: v1
kind: Secret
metadata:
  name: hospital-1-enrollment
  namespace: flare
type: Opaque
stringData:
  NVFLARE_CERT_SERVICE_URL: "https://cert-service.example.com:8443"
  NVFLARE_ENROLLMENT_TOKEN: "eyJhbGciOiJSUzI1NiIs..."
```

```yaml
# Pod spec: inject from Secret (env or volume mount)
envFrom:
  - secretRef:
      name: hospital-1-enrollment
```

For production, use an External Secrets Operator or Vault CSI driver to manage the Secret lifecycle (rotation, audit, access control) rather than creating Secrets manually.

**Note**

Environment variables and Secret-mounted files take priority over embedded files. This allows the same package to work across dev/staging/prod with different tokens and Certificate Service URLs.

## Background

### Current Provisioning Workflow

> **Deprecated.** This workflow is replaced by the Manual (CSR + sign) and Auto-Scale workflows described below, where private keys are generated locally and never transmitted.

Today, NVIDIA FLARE uses nvflare provision to create startup kits:

```
Project Admin                    Distribution              Org Admin
─────────────                    ────────────              ─────────────

1. Create project.yml
   (defines all participants)
       │
       ▼
2. nvflare provision
       │
       ▼
3. Generated startup kits:
   ├── server1/
   │   ├── startup/
   │   │   ├── rootCA.pem
   │   │   ├── server.crt      ──────► Email/USB/     ──────► Receive kit
   │   │   ├── server.key              Secure Channel         Start server
   │   │   └── fed_server.json
   │   └── ...
   │
   ├── site-1/
   │   ├── startup/
   │   │   ├── rootCA.pem
   │   │   ├── client.crt      ──────► Email/USB/     ──────► Receive kit
   │   │   ├── client.key              Secure Channel         Start client
   │   │   └── fed_client.json
   │   └── ...
```

### The Provisioning Problem

The provisioning workflow has been consistently identified as a major source of friction for NVFLARE users:

**Rigidity**

In NVFLARE, you cannot simply “connect” a client to a server. You must “provision” the entire network upfront. A project admin defines a project.yml file listing every participant, then generates startup kits that must be securely distributed out-of-band.

If a new client wants to join mid-experiment, the admin must perform manual dynamic-provisioning with the same rootCA - a process that is not familiar to DevOps.

**Opacity**

Errors in the project.yml (e.g., mismatching port definitions or domain names) often result in cryptic “Connection Refused” or “Cannot find path to server” errors, making debugging difficult.

**DevOps Burden**

Compared with frameworks like OpenFL or Flower, which support dynamic joining of clients via simple token authentication, NVFLARE’s provisioning feels archaic to researchers used to cloud-native flexibility. This architecture impedes rapid prototyping and requires substantial DevOps expertise.

**FLARE Dashboard**

Flare Dashboard simplifies the startup kit distribution, but still requires users to sign up the site, get approval (manual) and download the startup kit. The process is still pretty rigid.

**Summary of Limitations**

| Limitation | Description |
| --- | ----- |
| **Private keys in transit** | Keys are generated centrally and must be distributed securely |
| **Manual distribution** | Each startup kit must be sent to the recipient via secure channel |
| **Static pre-shared trust** | All participants must be known upfront; adding new ones requires manual dynamic-provisioning with the same rootCA |
| **Scalability** | Difficult to manage 100+ participants with manual distribution |
| **DevOps expertise required** | Configuration errors lead to cryptic failures |

### How the New Design Addresses These Concerns

The following table shows how each limitation is addressed by the new workflows:

| Concern | Manual Workflow | Auto-Scale Workflow |
| --- | ----- | --- |
| **Adding client mid-project** | Site Admin generates private key and CSR locally, sends CSR to Project Admin for signing. Project Admin signs and returns the signed certificate + rootCA.pem + Server URI. | Same CSR model, automated: Project Admin generates a token for the invited participant. Site generates CSR locally and submits it to the Certificate Service with the token for automatic signing. |
| **Private keys in transit** | **Keys generated locally, never transmitted** (only CSR/signed cert exchanged) | **Keys generated locally, never transmitted** |
| **Manual distribution** | Exchange CSR and signed certificate + rootCA only (smaller, simpler) | **Distribute tokens only** (short strings, can be sent via any channel) |
| **Static pre-shared trust** | New participants added by signing their CSR on demand | **Dynamic enrollment** - sites join on-demand with tokens |
| **Scalability** | Better for 5-10 participants  | **Scales to 100+ participants** with automated enrollment |
| **DevOps expertise** | Simple CLI commands | Simple CLI + one-time Certificate Service setup |

**Example: Adding a Client Mid-Project**

**Current provisioning**

1. Update project.yml with new participant.
2. Run `nvflare provision` (or dynamic-provision with same rootCA).
3. Locate the new participant's startup kit.
4. Securely distribute the entire startup kit.

**Manual workflow**

```bash
# Site Admin: generate private key and CSR locally
nvflare cert csr -n new-hospital -t client -o ./csr
# Send new-hospital.csr to Project Admin (private key stays local)

# Project Admin: sign the CSR and return signed certificate + Server URI
nvflare cert sign -r ./csr/new-hospital.csr -c ./ca -o ./signed
# Send signed client.crt, rootCA.pem, and Server URI (grpc://server:8002) to site

# Site Admin: generate startup kit locally with signed certificate
nvflare package -n new-hospital -e grpc://server:8002 -t client \
    --cert ./signed/client.crt --rootca ./signed/rootCA.pem
cd new-hospital && ./startup/start.sh
```

**Auto-Scale workflow**

```bash
# Project Admin (30 seconds)
nvflare token generate -n new-hospital \
    --cert-service https://cert-service:8443 \
    --api-key $API_KEY
# Send token + Certificate Service URL to site

# Site Admin (1 command)
nvflare package -n new-hospital -e grpc://server:8002 -t client \
    --cert-service https://cert-service:8443 \
    --token "eyJhbGciOiJSUzI1NiIs..."
cd new-hospital && ./startup/start.sh
# Auto-enrollment happens, site joins the federation
```

**Key Improvement**: Neither workflow requires touching existing participants or re-provisioning the network.

**Example: Kubernetes Dynamic Scaling to 100 Clients**

This example shows how to use the Auto-Scale workflow to dynamically scale a federated learning deployment to 100 clients in Kubernetes.

**Step 1: Project Admin — batch generate tokens (one-time)**

```bash
# Generate 100 tokens via Certificate Service API
nvflare token batch \
    --pattern "site-{001..100}" \
    --cert-service https://cert-service.example.com:8443 \
    --api-key $NVFLARE_API_KEY \
    -o ./tokens/

# Result: tokens/site-001.token, tokens/site-002.token, ..., tokens/site-100.token
```

**Step 2: Store tokens in K8s Secret Store**

Use an External Secrets Operator (ESO) to sync tokens from your cloud secret manager into K8s Secrets. Each site's token is stored as a separate key.

```yaml
# ExternalSecret: sync batch tokens from cloud secret manager into a K8s Secret
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: flare-tokens
  namespace: flare
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: cloud-secret-store    # Your SecretStore (AWS, Azure, GCP, Vault)
    kind: SecretStore
  target:
    name: flare-tokens          # K8s Secret created by ESO
  dataFrom:
    - extract:
        key: flare/enrollment-tokens   # Path in your secret manager
```

Alternatively, for dev/test you can create the Secret directly:

```bash
kubectl create secret generic flare-tokens \
    --from-file=./tokens/ \
    --namespace=flare
```

Each key in the Secret becomes a file when volume-mounted (e.g. `site-001.token` → `/tokens/site-001.token`).

**Step 3: Kubernetes StatefulSet**

Pods are named `flare-client-0`, `flare-client-1`, etc. Each pod derives its site name from its ordinal. Tokens are mounted from the K8s Secret as files; the Certificate Service URL is stored in a ConfigMap.

```yaml
# ConfigMap for non-secret configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: flare-config
  namespace: flare
data:
  NVFLARE_CERT_SERVICE_URL: "https://cert-service.flare.svc:8443"
  FL_SERVER_ENDPOINT: "grpc://flare-server.flare.svc:8002"
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: flare-client
  namespace: flare
spec:
  replicas: 100
  serviceName: flare-client
  selector:
    matchLabels:
      app: flare-client
  template:
    metadata:
      labels:
        app: flare-client
    spec:
      initContainers:
        - name: init-package
          image: nvflare/nvflare:latest
          command: ["/bin/sh", "-c"]
          args:
            - |
              ORDINAL=$${HOSTNAME##*-}
              SITE_NAME=$$(printf "site-%03d" $$((ORDINAL + 1)))
              echo "Generating package for: $$SITE_NAME"

              nvflare package -n $$SITE_NAME \
                  -e $${FL_SERVER_ENDPOINT} \
                  -t client \
                  -w /workspace

              echo $$SITE_NAME > /workspace/site_name.txt
          envFrom:
            - configMapRef:
                name: flare-config
          volumeMounts:
            - name: workspace
              mountPath: /workspace
      containers:
        - name: flare-client
          image: nvflare/nvflare:latest
          command: ["/bin/sh", "-c"]
          args:
            - |
              SITE_NAME=$$(cat /workspace/site_name.txt)

              # Token is read from file (mounted from K8s Secret)
              export NVFLARE_ENROLLMENT_TOKEN=$$(cat /tokens/$${SITE_NAME}.token)

              echo "Starting $$SITE_NAME"
              cd /workspace && ./startup/start.sh
          envFrom:
            - configMapRef:
                name: flare-config
          volumeMounts:
            - name: workspace
              mountPath: /workspace
            - name: tokens
              mountPath: /tokens
              readOnly: true
      volumes:
        - name: workspace
          emptyDir: {}
        - name: tokens
          secret:
            secretName: flare-tokens   # Managed by ExternalSecret or created manually
```

**Step 4: Deploy and scale**

```bash
kubectl apply -f flare-client-statefulset.yaml
kubectl scale statefulset flare-client --replicas=50 -n flare
kubectl scale statefulset flare-client --replicas=100 -n flare
# Each pod: 1) generates startup kit (init container), 2) auto-enrolls, 3) joins federation
```

**Key benefits for Kubernetes:** no pre-built images (clients generate packages at startup), dynamic scaling with `kubectl scale`, token-based identity per pod, auto-enrollment, and stateless pods that can be replaced without re-provisioning.

### PKI and Certificate Concepts

Before diving into the design, here are the key PKI concepts:

**Root CA (Certificate Authority)**

- **rootCA.pem**: Public certificate - distributed to all participants for verification
- **rootCA.key**: Private key - used to sign other certificates, must be protected

**Participant Certificates**

- **client.crt / server.crt**: Public certificate signed by root CA
- **client.key / server.key**: Private key for the participant

**CSR (Certificate Signing Request)**

A request containing a public key and identity information, submitted to a CA for signing. The private key never leaves the requestor’s machine.

**JWT (JSON Web Token)**

A signed token containing claims. Used for enrollment tokens that embed approval policies.

## Design Overview

### Architecture

The token-based enrollment system consists of:


```
                    +-------------------+
                    |   PROJECT ADMIN   |
                    | nvflare token     |
                    | generate/batch/   |
                    | info              |
                    +---------+---------+
                              | HTTPS API (tokens)
                              v
+-------------------------------------------------------------------------------+
|                     CERTIFICATE SERVICE                                       |
|  +-------------------------------------------------------------------------+  |
|  | CertServiceApp (HTTP)                                                    |  |
|  |   POST /api/v1/token   Token generation (nvflare token CLI)              |  |
|  |   GET  /api/v1/ca-cert Public root CA                                    |  |
|  |   POST /api/v1/enroll  CSR signing (CertRequestor)                       |  |
|  |   GET  /api/v1/pending List pending requests (admin)                     |  |
|  +-------------------------------------------------------------------------+  |
|  | CertService (core)  TokenService (JWT+policy)  rootCA.key (here)             |  |
|  +-------------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------+
                              | HTTPS (TLS)
              +---------------+---------------+
              v               v               v
      +-------------+ +-------------+ +-------------+
      |  FL Server  | |  FL Client  | |  FL Client  |
      |  CertReqs   | <--mTLS--> |  CertReqs   | |  CertReqs   |
      | 1-4 flow    | | 1-4 flow    | | 1-4 flow    |
      +-------------+ +-------------+ +-------------+
       1.Gen keys    1.Gen keys    1.Gen keys
       2.CSR 3.Submit 4.Get cert  (same for each)
```

### Key Components

| Component | Location | Responsibility |
| --- | ----- | --- |
| **TokenService** | nvflare/tool/enrollment/token_service.py | Generate JWT enrollment tokens with embedded policies |
| **CertService** | nvflare/cert_service/cert_service.py | Validate tokens, evaluate policies, sign CSRs |
| **CertServiceApp** | nvflare/cert_service/app.py | HTTP wrapper exposing CertService via REST API |
| **CertRequestor** | nvflare/security/enrollment/cert_requestor.py | Client-side: generate keys, create CSR, submit for signing via HTTP |
| **EnrollmentStore** | nvflare/cert_service/store.py | Persistence layer for enrolled entities and pending requests (SQLite default) |

## Enrollment Token

### Token Structure

Enrollment tokens are JWTs (JSON Web Tokens) signed with the root CA private key. They are tamper-proof and contain all information needed for enrollment.

**JWT structure**

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

**Token Claims:**

| Claim | Example | Description |
| --- | ----- | --- |
| jti | UUID | Unique token identifier (for audit/tracking) |
| sub | “hospital-1” | Subject - the participant name or pattern |
| subject_type | “client” | Participant type: client, server, relay, admin |
| iss | “MyProjectCA” | Issuer - extracted from root CA certificate |
| iat | timestamp | Issued at time |
| exp | timestamp | Expiration time |
| policy | {…} | Embedded approval policy (see below) |
| roles | [“lead”] | Roles for admin tokens |

### Token Security

**Why JWT with RS256?**

1. **Tamper-proof**: Tokens are signed with the root CA private key
2. **Self-contained**: No database lookup needed to validate
3. **Decentralized validation**: Any service with the public key can verify
4. **Expiration**: Built-in expiry prevents indefinite use

**Security Properties:**

| Property | How It’s Achieved |
| --- | ----- |
| Cannot be forged | Signed with root CA private key (RS256) |
| Cannot be tampered | Signature verification detects any modification |
| Cannot be reused indefinitely | Expiration time (exp claim) |
| Single-use enforcement | Certificate Service tracks used tokens |
| Scoped to participant | Subject (sub) specifies who can use it |

**Attack Mitigations:**

| Attack | Risk | Mitigation |
| --- | ----- | --- |
| Token theft | Attacker uses stolen token | Short expiry + single-use + name binding |
| Token forgery | Attacker creates fake token | RS256 signature verification |
| Token tampering | Attacker modifies claims | JWT signature detects changes |
| Replay attack | Reuse of valid token | Single-use tracking + expiration |
| Brute force | Guess valid tokens | UUID-based JTI + rate limiting |

### Token Signing Key

By default, the **root CA private key** serves two purposes:

1. **Sign participant certificates** (CSRs from clients, servers, relays)
2. **Sign JWT enrollment tokens**

This simplifies key management - one key pair for all cryptographic operations.

```
Root CA Key Pair
────────────────
rootCA.key (private)
    │
    ├──> Sign certificates (CSRs)
    │
    └──> Sign JWT tokens

rootCA.pem (public)
    │
    ├──> Verify certificates
    │
    └──> Verify JWT tokens
```

**Optional: Separate JWT Signing Key**

For advanced deployments, you can use a **separate key pair** for JWT signing:

| Scenario | Configuration | Use Case |
| --- | ----- | --- |
| Default (recommended) | Single root CA key | Simple, sufficient for most deployments |
| Separate JWT key | Both services must use same key pair | Key rotation, security isolation, compliance |

**Important: Key Consistency**

If using separate JWT keys, both services MUST use the same key pair:

```
TokenService                        CertService
────────────                        ───────────
jwt_key.key (private)    ----->    jwt_key.pub (public)
     │                                   │
     v                                   v
Sign tokens                         Verify tokens
```

Configuration for separate keys:

```python
# TokenService - signs with JWT private key
token_service = TokenService(
    root_ca_path="/path/to/ca",
    jwt_signing_key_path="/path/to/jwt_key.key"  # Optional
)

# CertService - verifies with JWT public key
cert_service = CertService(
    root_ca_cert=cert,
    root_ca_key=key,
    verification_key_path="/path/to/jwt_key.pub"  # Must match!
```

When to use separate keys:

- **Key rotation**: Rotate JWT signing key without changing root CA
- **Security isolation**: Limit blast radius if JWT key is compromised
- **Compliance**: Some security policies require key separation
- **Distributed systems**: Different services hold different keys

For most deployments, the **default single-key approach is recommended**.

## Approval Policy

### Policy Structure

The approval policy is embedded in the token and evaluated during enrollment.

**Policy schema**

```yaml
# approval_policy.yaml

metadata:
  project: "my-fl-project"
  description: "Enrollment policy for hospital network"
  version: "1.0"

token:
  validity: "7d"              # Token expiration

site:
  name_pattern: "^hospital-[0-9]+$"   # Allowed site names

user:
  allowed_roles:              # For admin tokens
    - lead
    - member
    - org_admin
  default_role: lead

approval:
  method: policy              # "policy" or "manual"
  rules:                      # First-match-wins
    - name: auto_approve_hospitals
      description: "Auto-approve hospital sites"
      match:
        site_name_pattern: "^hospital-.*"
      action: approve
      log: true

    - name: ip_restricted_sites
      description: "Sites from known IP ranges"
      match:
        site_name_pattern: "^datacenter-.*"
        source_ips:
          - "10.0.0.0/8"
          - "192.168.1.0/24"
      action: approve

    - name: reject_unknown
      description: "Reject all other requests"
      action: reject
      message: "Site name not recognized"
```

### Policy Elements

**Metadata**

Identifies the policy scope and version.

- **metadata**:
  - **project**: "my-fl-project"
  - **description**: "Policy description"
  - **version**: "1.0"

**Token Configuration**

Controls token lifetime.

- **token**:
  - **validity**: "7d" (supports: 30m, 2h, 7d, etc.)

**Site Constraints**

Restricts which site names are allowed.

- **site**:
  - **name_pattern**: "^hospital-[0-9]+$" (regex pattern)

**User Constraints**

For admin tokens, controls allowed roles.

- **user**:
  - **allowed_roles**: lead, member
  - **default_role**: lead

**Approval Rules**

Rules are evaluated in order — first match wins.

- **approval**:
  - **method**: policy
  - **rules** (list of rule objects):
    - **name**: rule_name
    - **description**: Human-readable description
    - **match** (optional):
      - **site_name_pattern**: "^pattern-.*"
      - **source_ips**: e.g. "10.0.0.0/8"
    - **action**: approve | reject | pending
    - **message**: Reason message (for reject/pending)
    - **log**: true

**Match Conditions**

| Condition | Description |
| --- | ----- |
| site_name_pattern | Wildcard pattern for site name (e.g., “hospital-\*”) |
| source_ips | CIDR ranges for client IP (optional, for static environments) |

**Actions**

| Action | Description |
| --- | ----- |
| approve | Automatically approve and sign the certificate |
| reject | Reject the enrollment request |
| pending | Queue for manual approval; admin must approve/reject via CLI |

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

### CSR Generation (Client-Side)

The CertRequestor generates a key pair and CSR locally:

```python
# Client-side: CertRequestor

def create_csr(self) -> bytes:
    # 1. Generate RSA key pair (2048-bit)
    private_key, public_key = generate_keys()

    # 2. Build CSR with identity attributes
    name_attributes = [
        x509.NameAttribute(NameOID.COMMON_NAME, "hospital-1"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Hospital A"),
    ]

    # 3. Sign CSR with private key
    csr = CertificateSigningRequestBuilder()
        .subject_name(Name(name_attributes))
        .sign(private_key, hashes.SHA256())

    # 4. Return PEM-encoded CSR
    return csr.public_bytes(Encoding.PEM)
```

**Key point:** The private key is generated locally and never transmitted.

### CSR Signing (Certificate Service)

The CertService validates the request and signs the CSR:

```python
# Server-side: CertService

def sign_csr(self, csr_data: bytes, token: str, context: EnrollmentContext) -> bytes:
    # 1. Validate JWT token
    token_payload = self.validate_token(token, context)

    # 2. Evaluate approval policy
    result = self.evaluate_policy(token_payload, context)

    if result.action == REJECT:
        raise ValueError(f"Rejected: {result.message}")

    # 3. Parse and verify CSR signature
    csr = load_pem_x509_csr(csr_data)
    if not csr.is_signature_valid:
        raise ValueError("Invalid CSR signature")

    # 4. Build certificate
    cert = generate_cert(
        subject=Identity(context.name, org=context.org, role=context.role),
        issuer=Identity(self.issuer),
        signing_pri_key=self.root_key,
        subject_pub_key=csr.public_key(),
        valid_days=365,
    )

    # 5. Return PEM-encoded certificate
    return serialize_cert(cert)
```

### Certificate Contents

The signed certificate includes:

| Field | OID | Value |
| --- | ----- | --- |
| Common Name (CN) | 2.5.4.3 | Participant name (e.g., “hospital-1”) |
| Organization (O) | 2.5.4.10 | Organization name (optional) |
| Organizational Unit (OU) | 2.5.4.11 | Participant type (client, server, relay, admin) |
| Unstructured Name | 1.2.840.113549.1.9.2 | Role for admin tokens (lead, member, etc.) |

## Certificate Service

### Overview

The Certificate Service is a standalone HTTP service that handles enrollment. It is deployed separately from the FL Server.

**Why Separate?**

1. **Security isolation**: Root CA private key is not on FL Server
2. **Scalability**: Can handle many concurrent enrollments
3. **Audit**: Centralized logging of all certificate issuance
4. **Blast radius**: If FL Server is compromised, attacker cannot issue certs

### Architecture

```
nvflare/cert_service/
├── __init__.py
├── app.py                  # CertServiceApp (HTTP wrapper)
├── cert_service.py         # CertService (core logic)
├── approval_policy.yaml    # Default policy
└── examples/
    ├── auto_approve.yaml
    ├── hospital_network.yaml
    ├── ip_whitelist.yaml
    └── manual_approval.yaml
```

### HTTP API

#### POST /api/v1/enroll

Enrollment endpoint. Returns signed certificate and root CA.

**Request**

```json
{
  "token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "csr": "-----BEGIN CERTIFICATE REQUEST-----\n...",
  "metadata": {
    "name": "hospital-1",
    "type": "client",
    "org": "Hospital A"
  }
}
```

**Response (200 OK)**

```json
{
  "certificate": "-----BEGIN CERTIFICATE-----\n...",
  "ca_cert": "-----BEGIN CERTIFICATE-----\n..."
}
```

**Response (202 Accepted — Pending Manual Approval)**

```json
{
  "status": "pending",
  "request_id": "abc123-def456-789",
  "message": "Enrollment request queued for manual approval",
  "poll_url": "/api/v1/enroll/abc123-def456-789"
}
```

**Error responses**

- **401 Unauthorized**: Invalid or expired token
- **403 Forbidden**: Policy rejection
- **400 Bad Request**: Invalid request format

---

#### GET /api/v1/enroll/{request_id}

Poll for pending request status.

**Response (200 — Still Pending)**

```json
{
  "status": "pending",
  "submitted_at": "2025-01-04T10:15:00Z"
}
```

**Response (200 — Approved)**

```json
{
  "status": "approved",
  "certificate": "-----BEGIN CERTIFICATE-----\n...",
  "ca_cert": "-----BEGIN CERTIFICATE-----\n..."
}
```

**Response (200 — Rejected)**

```json
{
  "status": "rejected",
  "reason": "Site not authorized for this project"
}
```

**Response (404)** — Request ID not found or expired.

---

#### GET /api/v1/pending (Admin Only)

List all pending enrollment requests. Requires admin authentication.

**Query parameters**

- `type` (optional): Filter by entity type (client, relay, user)

**Response**

```json
{
  "pending": [
    {
      "name": "hospital-1",
      "entity_type": "client",
      "org": "Hospital A",
      "submitted_at": "2025-01-04T10:15:00Z",
      "expires_at": "2025-01-11T10:15:00Z",
      "token_subject": "hospital-*",
      "source_ip": "10.2.3.4"
    },
    {
      "name": "admin@org.com",
      "entity_type": "user",
      "org": "Org Inc",
      "role": "lead",
      "submitted_at": "2025-01-04T11:00:00Z",
      "expires_at": "2025-01-11T11:00:00Z",
      "token_subject": "*@org.com",
      "source_ip": null
    }
  ]
}
```

---

#### POST /api/v1/pending/{name}/approve (Admin Only)

Approve a single pending enrollment request.

**Query parameters**

- `type` (required): Entity type (client, relay, user)

**Response (200)**

```json
{
  "status": "approved",
  "name": "hospital-1",
  "entity_type": "client",
  "certificate_issued": true
}
```

---

#### POST /api/v1/pending/approve_batch (Admin Only)

Approve multiple pending requests by pattern.

**Request**

```json
{
  "pattern": "hospital-*",
  "type": "client"
}
```

**Response (200)**

```json
{
  "approved": ["hospital-1", "hospital-2", "hospital-3"],
  "count": 3
}
```

---

#### POST /api/v1/pending/{name}/reject (Admin Only)

Reject a single pending enrollment request.

**Query parameters**

- `type` (required): Entity type (client, relay, user)

**Request**

```json
{
  "reason": "Not authorized for this project"
}
```

**Response (200)**

```json
{
  "status": "rejected",
  "name": "hospital-1",
  "entity_type": "client"
}
```

---

#### POST /api/v1/pending/reject_batch (Admin Only)

Reject multiple pending requests by pattern.

**Request**

```json
{
  "pattern": "temp-*",
  "type": "client",
  "reason": "Batch cleanup"
}
```

**Response (200)**

```json
{
  "rejected": ["temp-1", "temp-2"],
  "count": 2
}
```

---

#### GET /api/v1/enrolled (Admin Only)

List all enrolled entities.

**Query parameters**

- `type` (optional): Filter by entity type

**Response**

```json
{
  "enrolled": [
    {
      "name": "hospital-1",
      "entity_type": "client",
      "org": "Hospital A",
      "enrolled_at": "2025-01-04T12:00:00Z"
    },
    {
      "name": "researcher@uni.edu",
      "entity_type": "user",
      "org": "University",
      "role": "member",
      "enrolled_at": "2025-01-04T12:30:00Z"
    }
  ]
}
```

---

#### POST /api/v1/token (Admin Only)

Generate an enrollment token. Used by `nvflare token generate` when the root CA private key is on the Certificate Service.

**Request**

```json
{
  "name": "hospital-1",
  "entity_type": "client",
  "valid_days": 7,
  "policy_override": {}
}
```

**Response (200 OK)**

```json
{
  "token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "subject": "hospital-1",
  "expires_at": "2025-01-11T10:00:00Z"
}
```

**Request (batch)**

```json
{
  "names": ["site-001", "site-002", "site-003"],
  "entity_type": "client",
  "valid_days": 7
}
```

**Response (200 OK — batch)**

```json
{
  "tokens": [
    {"name": "site-001", "token": "eyJhbGci..."},
    {"name": "site-002", "token": "eyJhbGci..."},
    {"name": "site-003", "token": "eyJhbGci..."}
  ]
}
```

---

#### GET /api/v1/ca-cert

Download public root CA certificate. No authentication required.

**Response**

PEM-encoded root CA certificate (`application/x-pem-file`).

---

#### GET /api/v1/ca-info

Get root CA information. No authentication required.

**Response**

```json
{
  "subject": "CN=NVFlare",
  "issuer": "CN=NVFlare",
  "not_valid_before": "2025-01-01T00:00:00Z",
  "not_valid_after": "2035-01-01T00:00:00Z",
  "serial_number": "1234567890"
}
```

---

#### GET /health

Health check endpoint.

**Response**

```json
{
  "status": "healthy"
}
```

### API Authentication

**Admin Operations** require an API key:

- POST /api/v1/token — Token generation
- GET /api/v1/pending — List pending requests
- GET /api/v1/pending/{name} — Get pending request details
- GET /api/v1/pending/{name}/approve — Approve single request
- GET /api/v1/pending/{name}/reject — Reject single request
- POST /api/v1/pending/approve_batch — Bulk approve by pattern
- POST /api/v1/pending/reject_batch — Bulk reject by pattern
- GET /api/v1/enrolled — List enrolled entities

**No API Key Required** for:

- POST /api/v1/enroll — Enrollment (token is authentication)
- GET /api/v1/ca-cert — Public CA certificate
- GET /api/v1/ca-info — CA information
- GET /health — Health check

**API Key Setup:**

1. **Generate key with CLI:**

   ```bash
   nvflare cert api-key > api_key.txt
   ```

2. **Configure Certificate Service** (choose one):

   - **Option A - Environment variable:** Set `NVFLARE_API_KEY` to the generated key (e.g. in systemd, Docker, or K8s Secret).
   - **Option B - Config file:** In `cert_service_config.yaml`, set `api_key: "<the-key>"`.

3. **Use with admin CLI commands:**

   ```bash
   export NVFLARE_API_KEY="<the-key>"
   nvflare token generate -n site-1 --cert-service https://...
   ```

**HTTP Header Format:** `Authorization: Bearer <api-key>`

### Pending Request Storage

Pending requests are stored by the Certificate Service until:

- Approved by admin (certificate issued)
- Rejected by admin
- Expired (default: 7 days timeout)

Storage options:

1. **File-based** (simple deployments)
2. **SQLite** (single-node service)
3. **PostgreSQL** (production, multi-node)

### Admin CLI for Pending Requests

Administrators manage pending requests via CLI.

**List all pending enrollment requests:**

```bash
nvflare enrollment list
```

Output:

```
Name           Type      Org          Submitted             Token Subject    Status
──────────────────────────────────────────────────────────────────────────────────────
hospital-1     client    Hospital A   2025-01-04 10:15:00   hospital-*       pending
hospital-2     client    Hospital B   2025-01-04 10:20:00   hospital-*       pending
admin@org.com  user      Org Inc      2025-01-04 11:00:00   *@org.com        pending
```

**Filter by type:**

```bash
nvflare enrollment list --type client    # Sites only
nvflare enrollment list --type user      # Users only
```

**View details of a specific request:**

```bash
nvflare enrollment info hospital-1 --type client
```

Output:

```
Name:            hospital-1
Type:            client
Organization:    Hospital A
Submitted:       2025-01-04 10:15:00 UTC
Expires:         2025-01-11 10:15:00 UTC
Token Subject:   hospital-*
Source IP:       10.2.3.4
CSR Subject:     CN=hospital-1, O=Hospital A
```

**View user request:**

```bash
nvflare enrollment info admin@org.com --type user
```

Output:

```
Name:            admin@org.com
Type:            user
Organization:    Org Inc
Role:            lead
Submitted:       2025-01-04 11:00:00 UTC
...
```

**Approve a pending request (specify type):**

```bash
nvflare enrollment approve hospital-1 --type client
nvflare enrollment approve admin@org.com --type user
```

**Reject a pending request with reason:**

```bash
nvflare enrollment reject hospital-2 --type client --reason "Not authorized"
```

**Bulk approve matching pattern:**

```bash
nvflare enrollment approve --pattern "hospital-*" --type client
```

**List enrolled entities:**

```bash
nvflare enrollment enrolled                    # All
nvflare enrollment enrolled --type client      # Sites only
nvflare enrollment enrolled --type user       # Users only
```

**Configuration (environment variables):** `NVFLARE_CERT_SERVICE_URL` (Certificate Service URL), `NVFLARE_API_KEY` (admin authentication token).

### Configuration

The Certificate Service is configured via a YAML file. This configuration is loaded at startup and controls all aspects of the service.

**Configuration File:**

```yaml
# cert_service_config.yaml

server:
  host: 0.0.0.0
  port: 8443
  tls:
    cert: /path/to/service.crt   # Public TLS cert (Let's Encrypt)
    key: /path/to/service.key

# API key for admin authentication (token gen, approvals)
# Generate with: nvflare cert api-key
# Or set via environment: NVFLARE_API_KEY
api_key: "your-api-key-here"

# Data directory for auto-initialization
# On first start, root CA is generated here if not exists
data_dir: /var/lib/cert_service

# Project name (used as CA CN)
project_name: "NVFlare"

ca:
  # If not specified, derived from data_dir
  cert: /var/lib/cert_service/rootCA.pem   # FLARE root CA (public)
  key: /var/lib/cert_service/rootCA.key    # FLARE root CA (private - auto-generated)

policy:
  file: /path/to/approval_policy.yaml  # Enrollment policy
  # If not specified, uses built-in default (auto-approve all)

storage:
  type: sqlite                   # sqlite | postgresql
  path: /var/lib/cert_service/enrollment.db
  # For PostgreSQL:
  # type: postgresql
  # connection: postgresql://user:pass@host:5432/certdb

pending:
  timeout: 604800                # 7 days in seconds
  cleanup_interval: 3600         # Clean expired requests every hour

audit:
  enabled: true
  log_file: /var/log/cert_service/audit.log
```

**Configuration Sections Explained:**

| Section | Purpose |
| --- | ----- |
| api_key | API key for admin authentication (token gen, approvals). Generate with nvflare cert api-key |
| data_dir | Data directory for auto-initialization. Root CA generated here on first start if not exists |
| project_name | Project name used as CA Common Name (CN) |
| server | HTTP server binding and TLS configuration for the service endpoint |
| ca | FLARE root CA paths. If not specified, derived from data_dir |
| policy | Approval policy for enrollment requests |
| storage | Backend for tracking enrolled entities and pending requests |
| pending | Pending request timeout and cleanup settings |
| audit | Audit logging for compliance |

**How CertServiceApp Uses Configuration:**

```python
# nvflare/cert_service/app.py
from nvflare.cert_service.app import CertServiceApp

# Option 1: From config file
app = CertServiceApp("/path/to/config.yaml")

# Option 2: With arguments (auto-initializes root CA on first start)
app = CertServiceApp(
    data_dir="/var/lib/cert_service",   # Root CA generated here if not exists
    project_name="MyProject",           # CA Common Name
    api_key="your-api-key",             # For admin operations
)

# Start the server
app.run(
    host="0.0.0.0",
    port=8443,
    ssl_context=("tls.crt", "tls.key"),  # Service TLS (e.g., Let's Encrypt)
)
```

**Key Behaviors:**

1. **Auto-initialization**: On first start, if rootCA.pem and rootCA.key don’t exist in data_dir, they are automatically generated. The private key only exists on the service.
2. **API Key**: Required for admin operations (token generation, approvals). Generate with nvflare cert api-key. Configure via NVFLARE_API_KEY env var or config file.
3. **Token-based enrollment**: The POST /api/v1/enroll endpoint uses the enrollment token for authentication - no API key needed for sites/users.

**Auto-Initialization on First Start:**

On first start, if the root CA does not exist, the Certificate Service will:

1. Generate root CA certificate and private key
2. Log the generated files location
3. The root CA private key **only exists on the service** - never distributed

This ensures the root CA private key is generated locally and never transmitted.

**Starting the Certificate Service:**

**Option 1 — Direct Python (with config file):**

```bash
python -c "
from nvflare.cert_service.app import CertServiceApp
app = CertServiceApp('/path/to/config.yaml')
app.run()
"
```

**Option 2 — Direct Python (with arguments):**

```bash
python -c "
from nvflare.cert_service.app import CertServiceApp
app = CertServiceApp(
    data_dir='/var/lib/cert_service',
    project_name='MyProject',
    api_key='your-api-key',
)
app.run(host='0.0.0.0', port=8443, ssl_context=('tls.crt', 'tls.key'))
"
```

**Option 3 — Docker:**

```bash
docker run -d  \
    -v /var/lib/cert_service:/data  \
    -e NVFLARE_API_KEY="your-api-key"  \
    -p 8443:8443  \
    nvflare/cert-service:latest
```

**Option 4 — Production with gunicorn:**

```bash
gunicorn "nvflare.cert_service.app:create_app('/path/to/config.yaml')"
```

**Configuration Flow:**


```
┌─────────────────────────────────────────────────────────────────────┐
│                    cert_service_config.yaml                         │
│                      (or constructor args)                          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CertServiceApp.__init__()                      │
│                                                                     │
│   1. Auto-init: If rootCA not exists, generate it (first start)    │
│                                                                     │
│   api_key: ────► Admin authentication for protected endpoints      │
│                                                                     │
│   data_dir: ───► Root CA files, database location                  │
│                                                                     │
│   server: ─────► Flask app binding (host, port, TLS)               │
│                                                                     │
│   ca: ─────────► CertService (root CA for signing)                 │
│                                                                     │
│   policy: ─────► CertService (approval rules)                      │
│                                                                     │
│   storage: ────► EnrollmentStore (SQLite or PostgreSQL)            │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        HTTPS Server Running                         │
│                                                                     │
│   No API Key Required:                                              │
│   POST /api/v1/enroll          ─► Token validation + CSR signing   │
│   GET  /api/v1/ca-cert         ─► Download public rootCA.pem       │
│   GET  /health                 ─► Health check                     │
│                                                                     │
│   API Key Required (Admin):                                         │
│   POST /api/v1/token           ─► Generate enrollment tokens       │
│   GET  /api/v1/pending         ─► List pending requests            │
│   POST /api/v1/pending/{n}/approve ─► Approve single request       │
│   POST /api/v1/pending/approve_batch ─► Bulk approve               │
│   POST /api/v1/pending/reject_batch  ─► Bulk reject                │
│   GET  /api/v1/enrolled        ─► List enrolled entities           │
└─────────────────────────────────────────────────────────────────────┘

```

**Two Different Certificates Explained:**

The Certificate Service uses two completely separate certificates:

| Config Key | Purpose | Example |
| --- | ----- | --- |
| server.tls.cert | HTTPS endpoint TLS (public trust) | Let’s Encrypt, DigiCert, or handled by reverse proxy |
| ca.cert | FLARE root CA (signs FL participant certs) | Auto-generated on first start, project-specific |

**Service TLS (server.tls) - Deployment Options:**

| Scenario | tls_cert / tls_key | Notes |
| --- | ----- | --- |
| Local development | None / None | HTTP only, no TLS needed |
| Behind reverse proxy (nginx, traefik) | None / None | HTTP internally, proxy handles TLS |
| Kubernetes with Ingress | None / None | HTTP internally, Ingress handles TLS |
| Cloud (GCP, AWS, Azure) | None / None | HTTP internally, Load Balancer handles TLS |
| Direct HTTPS (Let’s Encrypt) | Paths to Let’s Encrypt certs | For simple deployments without proxy |
| Direct HTTPS (self-signed) | Paths to self-signed certs | Dev/testing only, clients must trust cert |

**Recommended approach:** For most production deployments, leave tls_cert and tls_key as None. Run the service with HTTP internally and let infrastructure (reverse proxy, Ingress controller, or cloud load balancer) handle TLS termination. This is standard practice for web services.

```python
# Development (HTTP)
app.run(host="127.0.0.1", port=8080)
# Access: http://localhost:8080

# Production behind nginx/traefik (HTTP internally)
app.run(host="127.0.0.1", port=8080)
# Proxy handles: https://cert-service.example.com → http://127.0.0.1:8080

# Simple production with Let's Encrypt (direct HTTPS)
app.run(
    host="0.0.0.0",
    port=443,
    ssl_context=(
        "/etc/letsencrypt/live/example.com/fullchain.pem",
        "/etc/letsencrypt/live/example.com/privkey.pem",
    ),
)
```

**FLARE Root CA (ca.cert/ca.key):**

This is completely separate from service TLS. The FLARE root CA:

- Is auto-generated on first start (if not exists)
- Signs FL participant certificates (clients, users, relays)
- Private key **only exists on the Certificate Service**
- Is NOT used for HTTPS — only for FL PKI

### Enrollment Store

The Certificate Service tracks enrolled sites and pending requests using a pluggable storage backend. Default is SQLite.

**Entity Types**

Enrollment applies to both sites and users:

| Entity Type | Examples | Description |
| --- | ----- | --- |
| client | hospital-1, site-001 | FL client sites |
| relay | relay-east, relay-1 | Hierarchical FL relay nodes |
| server | server1 | FL server (Manual workflow only) |
| user | admin@org.com, researcher-1 | FLARE Console users (with roles) |

**Abstract Interface**


```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Set

@dataclass
class EnrolledEntity:
    """An enrolled site or user."""
    name: str                    # Entity name (site-1, admin@org.com)
    entity_type: str             # client | relay | server | user
    enrolled_at: datetime
    org: Optional[str] = None    # Organization
    role: Optional[str] = None   # Role (for users only)

@dataclass
class PendingRequest:
    """Pending enrollment request awaiting admin approval."""
    name: str                    # Entity name
    entity_type: str             # client | relay | server | user
    org: str
    csr_pem: str
    submitted_at: datetime
    expires_at: datetime
    token_subject: str
    role: Optional[str] = None   # Role (for users only)
    source_ip: Optional[str] = None
    signed_cert: Optional[str] = None
    approved: bool = False
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None

class EnrollmentStore(ABC):
    """Abstract interface for enrollment state storage.

    Tracks enrolled entities (sites and users) and pending requests.
    Entity uniqueness is determined by (name, entity_type) pair.
    """

    # ─────────────────────────────────────────────────────
    # Enrolled Entities (Sites and Users)
    # ─────────────────────────────────────────────────────

    @abstractmethod
    def is_enrolled(self, name: str, entity_type: str) -> bool:
        """Check if an entity is already enrolled."""
        pass

    @abstractmethod
    def add_enrolled(self, entity: EnrolledEntity) -> None:
        """Mark an entity as enrolled. Also removes from pending."""
        pass

    @abstractmethod
    def get_enrolled(
        self, entity_type: Optional[str] = None
    ) -> List[EnrolledEntity]:
        """Get enrolled entities, optionally filtered by type."""
        pass

    # ─────────────────────────────────────────────────────
    # Pending Requests
    # ─────────────────────────────────────────────────────

    @abstractmethod
    def is_pending(self, name: str, entity_type: str) -> bool:
        """Check if an entity has a pending request."""
        pass

    @abstractmethod
    def add_pending(self, request: PendingRequest) -> None:
        """Add a new pending request."""
        pass

    @abstractmethod
    def get_pending(
        self, name: str, entity_type: str
    ) -> Optional[PendingRequest]:
        """Get pending request for an entity."""
        pass

    @abstractmethod
    def get_all_pending(
        self, entity_type: Optional[str] = None
    ) -> List[PendingRequest]:
        """Get all pending requests, optionally filtered by type."""
        pass

    @abstractmethod
    def approve_pending(
        self, name: str, entity_type: str,
        signed_cert: str, approved_by: str
    ) -> None:
        """Approve a pending request and store signed certificate."""
        pass

    @abstractmethod
    def reject_pending(
        self, name: str, entity_type: str, reason: str
    ) -> None:
        """Reject and remove a pending request."""
        pass

    @abstractmethod
    def cleanup_expired(self) -> int:
        """Remove expired pending requests. Returns count removed."""
        pass

**SQLite Implementation (Default)**
```python
import sqlite3
from pathlib import Path

class SQLiteEnrollmentStore(EnrollmentStore):
    """SQLite-based enrollment store. Default for single-node deployments."""

    def __init__(self, db_path: str = "/var/lib/cert_service/enrollment.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript('''
                -- Enrolled entities (sites and users)
                CREATE TABLE IF NOT EXISTS enrolled_entities (
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    enrolled_at TEXT NOT NULL,
                    org TEXT,
                    role TEXT,
                    PRIMARY KEY (name, entity_type)
                );

                -- Pending enrollment requests
                CREATE TABLE IF NOT EXISTS pending_requests (
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    org TEXT NOT NULL,
                    csr_pem TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    token_subject TEXT NOT NULL,
                    role TEXT,
                    source_ip TEXT,
                    signed_cert TEXT,
                    approved INTEGER DEFAULT 0,
                    approved_at TEXT,
                    approved_by TEXT,
                    PRIMARY KEY (name, entity_type)
                );

                CREATE INDEX IF NOT EXISTS idx_pending_type
                    ON pending_requests(entity_type);
                CREATE INDEX IF NOT EXISTS idx_expires
                    ON pending_requests(expires_at);
            ''')

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ─────────────────────────────────────────────────────
    # Enrolled Entities
    # ─────────────────────────────────────────────────────

    def is_enrolled(self, name: str, entity_type: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM enrolled_entities "
                "WHERE name = ? AND entity_type = ?",
                (name, entity_type)
            ).fetchone()
        return row is not None

    def add_enrolled(self, entity: EnrolledEntity) -> None:
        with self._connect() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO enrolled_entities
                (name, entity_type, enrolled_at, org, role)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                entity.name, entity.entity_type,
                entity.enrolled_at.isoformat(),
                entity.org, entity.role
            ))
            # Remove from pending
            conn.execute(
                "DELETE FROM pending_requests "
                "WHERE name = ? AND entity_type = ?",
                (entity.name, entity.entity_type)
            )

    def get_enrolled(
        self, entity_type: Optional[str] = None
    ) -> List[EnrolledEntity]:
        with self._connect() as conn:
            if entity_type:
                rows = conn.execute(
                    "SELECT * FROM enrolled_entities WHERE entity_type = ?",
                    (entity_type,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM enrolled_entities"
                ).fetchall()
        return [self._row_to_entity(row) for row in rows]

    # ─────────────────────────────────────────────────────
    # Pending Requests
    # ─────────────────────────────────────────────────────

    def is_pending(self, name: str, entity_type: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM pending_requests "
                "WHERE name = ? AND entity_type = ?",
                (name, entity_type)
            ).fetchone()
        return row is not None

    def add_pending(self, request: PendingRequest) -> None:
        with self._connect() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO pending_requests
                (name, entity_type, org, csr_pem, submitted_at,
                 expires_at, token_subject, role, source_ip)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                request.name, request.entity_type,
                request.org, request.csr_pem,
                request.submitted_at.isoformat(),
                request.expires_at.isoformat(),
                request.token_subject, request.role,
                request.source_ip,
            ))

    def get_pending(
        self, name: str, entity_type: str
    ) -> Optional[PendingRequest]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pending_requests "
                "WHERE name = ? AND entity_type = ?",
                (name, entity_type)
            ).fetchone()
        if not row:
            return None
        return self._row_to_request(row)

    def get_all_pending(
        self, entity_type: Optional[str] = None
    ) -> List[PendingRequest]:
        with self._connect() as conn:
            if entity_type:
                rows = conn.execute(
                    "SELECT * FROM pending_requests "
                    "WHERE approved = 0 AND entity_type = ?",
                    (entity_type,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM pending_requests WHERE approved = 0"
                ).fetchall()
        return [self._row_to_request(row) for row in rows]

    def approve_pending(
        self, name: str, entity_type: str,
        signed_cert: str, approved_by: str
    ) -> None:
        with self._connect() as conn:
            conn.execute('''
                UPDATE pending_requests
                SET signed_cert = ?, approved = 1,
                    approved_at = ?, approved_by = ?
                WHERE name = ? AND entity_type = ?
            ''', (
                signed_cert, datetime.utcnow().isoformat(),
                approved_by, name, entity_type
            ))

    def reject_pending(
        self, name: str, entity_type: str, reason: str
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM pending_requests "
                "WHERE name = ? AND entity_type = ?",
                (name, entity_type)
            )

    def cleanup_expired(self) -> int:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM pending_requests WHERE expires_at \< ?",
                (now,)
            )
        return cursor.rowcount

    # ─────────────────────────────────────────────────────
    # Row Converters
    # ─────────────────────────────────────────────────────

    def _row_to_entity(self, row) -> EnrolledEntity:
        return EnrolledEntity(
            name=row["name"],
            entity_type=row["entity_type"],
            enrolled_at=datetime.fromisoformat(row["enrolled_at"]),
            org=row["org"],
            role=row["role"],
        )

    def _row_to_request(self, row) -> PendingRequest:
        return PendingRequest(
            name=row["name"],
            entity_type=row["entity_type"],
            org=row["org"],
            csr_pem=row["csr_pem"],
            submitted_at=datetime.fromisoformat(row["submitted_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            token_subject=row["token_subject"],
            role=row["role"],
            source_ip=row["source_ip"],
            signed_cert=row["signed_cert"],
            approved=bool(row["approved"]),
            approved_at=(datetime.fromisoformat(row["approved_at"])
                        if row["approved_at"] else None),
            approved_by=row["approved_by"],
        )

```

**PostgreSQL Implementation (Production)**

Located in nvflare/app_opt/cert_service/postgres_store.py:

```python
import psycopg2
from psycopg2.extras import RealDictCursor

class PostgreSQLEnrollmentStore(EnrollmentStore):
    """PostgreSQL-based enrollment store for production/HA deployments."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._init_db()

    def _connect(self):
        return psycopg2.connect(
            self.connection_string,
            cursor_factory=RealDictCursor
        )

    def _init_db(self):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS enrolled_entities (
                        name TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        enrolled_at TIMESTAMP NOT NULL,
                        PRIMARY KEY (name, entity_type)
                    );

                    CREATE TABLE IF NOT EXISTS pending_requests (
                        name TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        org TEXT NOT NULL,
                        csr_pem TEXT NOT NULL,
                        submitted_at TIMESTAMP NOT NULL,
                        expires_at TIMESTAMP NOT NULL,
                        token_subject TEXT NOT NULL,
                        source_ip TEXT,
                        signed_cert TEXT,
                        approved BOOLEAN DEFAULT FALSE,
                        approved_at TIMESTAMP,
                        approved_by TEXT
                    );

                    CREATE INDEX IF NOT EXISTS idx_expires
                        ON pending_requests(expires_at);
                ''')

    # ... same interface methods as SQLiteEnrollmentStore ...
    # (uses psycopg2 instead of sqlite3)

```


**Factory Function**

```python
def create_enrollment_store(config: dict) -> EnrollmentStore:
    """Create enrollment store based on configuration."""

    storage_type = config.get("type", "sqlite")

    if storage_type == "sqlite":
        path = config.get("path", "/var/lib/cert_service/enrollment.db")
        return SQLiteEnrollmentStore(db_path=path)

    elif storage_type == "postgresql":
        conn_string = config["connection"]
        # Import only when needed (optional dependency)
        from nvflare.app_opt.cert_service.postgres_store import (
            PostgreSQLEnrollmentStore
        )
        return PostgreSQLEnrollmentStore(connection_string=conn_string)

    else:
        raise ValueError(f"Unknown storage type: {storage_type}")

```

**Storage Comparison**

| Feature | SQLite (Default) | PostgreSQL |
| --- | ----- | --- |
| Deployment | Single node | Multi-node / HA |
| Dependencies | Built-in (Python stdlib) | Requires psycopg2 |
| Concurrency | Limited (file locks) | Full ACID |
| Backup | Copy file | pg_dump / replication |
| Use Case | Dev, small production | Large production, HA |

### Deployment Options

1. **Standalone container** (Docker/K8s)
2. **Cloud-managed** (AWS, Azure, GCP)

## Client Enrollment

### CertRequestor

The CertRequestor handles client-side enrollment:

**Location:** nvflare/security/enrollment/cert_requestor.py

from nvflare.security.enrollment import (
    CertRequestor,
    EnrollmentIdentity,
    EnrollmentOptions,
    EnrollmentResult,
)

# Create identity
identity = EnrollmentIdentity.for_client("hospital-1", org="Hospital A")

# Configure options
options = EnrollmentOptions(
    timeout=30.0,                    # HTTP request timeout
    output_dir="/workspace/startup",
)

# Create requestor
requestor = CertRequestor(
    cert_service_url="https://cert-service.example.com",
    enrollment_token="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
    identity=identity,
    options=options,
)

# Perform enrollment
result = requestor.request_certificate()

print(f"Certificate: {result.cert_path}")
print(f"Private key: {result.key_path}")
print(f"Root CA: {result.ca_path}")

### EnrollmentOptions Configuration

EnrollmentOptions can be configured via:

1. Direct instantiation (for testing/scripts)
2. FLARE client configuration (fed_client.json)
3. Environment variables (for containerized deployments)

Option 1: Direct Instantiation

options = EnrollmentOptions(
    timeout=30.0,
    output_dir="/workspace/startup",
)

Option 2: From FLARE Client Configuration

Enrollment settings in fed_client.json (timeouts only, NOT the URL):

{
    "format_version": 2,
    "client": {
        "name": "hospital-1"
    },
    "enrollment": {
        "timeout": 30.0,
        "max_retries": 3,
        "retry_delay": 5.0
    }
}

Note

The cert_service_url is intentionally NOT in fed_client.json because:

1. The Certificate Service URL is not known when the startup kit is generated
2. The URL may change between environments (dev, staging, prod)
3. The URL should be provided at deployment time, not package time

Use environment variables or a separate file for the URL (see below).

Then load from client_args:

@staticmethod
def from_client_args(client_args: dict, output_dir: str) -> "EnrollmentOptions":
    """Create EnrollmentOptions from FLARE client configuration.

    *Args:
        *client_args: Client configuration dictionary (from fed_client.json)
        *output_dir: Directory to save certificates

    *Returns:
        *EnrollmentOptions configured from client_args
    """
    enrollment_config = client_args.get("enrollment", {})

    return EnrollmentOptions(
        timeout=enrollment_config.get("timeout", 30.0),
        output_dir=output_dir,
        max_retries=enrollment_config.get("max_retries", 3),
        retry_delay=enrollment_config.get("retry_delay", 5.0),
    )

Option 3: From Environment Variables (Recommended for URL)

The Certificate Service URL and token are provided via environment variables, which are set at deployment time (not package time):

# Required for enrollment
NVFLARE_CERT_SERVICE_URL=https://cert-service.example.com:8443
NVFLARE_ENROLLMENT_TOKEN=eyJhbGciOiJSUzI1NiIs...

# Optional (have sensible defaults)
NVFLARE_ENROLLMENT_TIMEOUT=30.0
NVFLARE_ENROLLMENT_MAX_RETRIES=3
NVFLARE_ENROLLMENT_RETRY_DELAY=5.0

@staticmethod
def from_env(output_dir: str) -> "EnrollmentOptions":
    """Create EnrollmentOptions from environment variables.

    *Environment Variables:
        *NVFLARE_ENROLLMENT_TIMEOUT: HTTP request timeout (default: 30.0)
        *NVFLARE_ENROLLMENT_MAX_RETRIES: Max retry attempts (default: 3\)
        *NVFLARE_ENROLLMENT_RETRY_DELAY: Delay between retries (default: 5.0)

    *Returns:
        *EnrollmentOptions configured from environment
    """
    import os

    return EnrollmentOptions(
        timeout=float(os.getenv("NVFLARE_ENROLLMENT_TIMEOUT", "30.0")),
        output_dir=output_dir,
        max_retries=int(os.getenv("NVFLARE_ENROLLMENT_MAX_RETRIES", "3")),
        retry_delay=float(os.getenv("NVFLARE_ENROLLMENT_RETRY_DELAY", "5.0")),
    )

Option 4: From Separate Enrollment File

For non-containerized deployments, a separate enrollment.json can be placed in the startup directory at deployment time.

Consolidated Package Command (Recommended)

The nvflare package command can include the Certificate Service URL and token, creating everything in one step:

# Site operator runs single command with all info from Project Admin
nvflare package  \
    -n hospital-1  \
    -e grpc://server.example.com:8002  \
    -t client  \
    --cert-service https://cert-service.example.com:8443  \
    --token eyJhbGciOiJSUzI1NiIs...  \
    -o ./

This generates:

./hospital-1/
├── startup/
│   ├── fed_client.json
│   ├── enrollment.json         \# Created with cert_service_url
│   ├── enrollment_token        \# Created with token
│   ├── start.sh
│   └── ...
└── ...

Manual File Creation (Alternative)

If the package was already generated without enrollment info:

# Create enrollment.json
cat > ./hospital-1/startup/enrollment.json \<\< EOF
{
    "cert_service_url": "https://cert-service.example.com:8443"
}
EOF

# Place token
echo "eyJhbGciOiJSUzI1..." > ./hospital-1/startup/enrollment_token

File contents:

{
    "cert_service_url": "https://cert-service.example.com:8443",
    "timeout": 30.0,
    "max_retries": 3,
    "retry_delay": 5.0
}

Note

For containerized/K8s deployments, inject NVFLARE_CERT_SERVICE_URL and NVFLARE_ENROLLMENT_TOKEN from a K8s Secret (via `envFrom: secretRef` or volume mount). For production, manage the Secret through an External Secrets Operator or Vault CSI driver for rotation and audit.

EnrollmentOptions Dataclass

from dataclasses import dataclass, field
from typing import Optional
import os

@dataclass
class EnrollmentOptions:
    """Configuration options for certificate enrollment.

    *Args:
        *timeout: HTTP request timeout in seconds
        *output_dir: Directory to save certificates
        *max_retries: Maximum number of retry attempts on failure
        *retry_delay: Delay between retries in seconds
    """
    timeout: float = 30.0
    output_dir: str = "./startup"
    max_retries: int = 3
    retry_delay: float = 5.0

    @classmethod
    def from_client_args(
        cls, client_args: dict, output_dir: str
    ) -> "EnrollmentOptions":
        """Create from FLARE client configuration."""
        enrollment_config = client_args.get("enrollment", {})
        return cls(
            timeout=enrollment_config.get("timeout", 30.0),
            output_dir=output_dir,
            max_retries=enrollment_config.get("max_retries", 3),
            retry_delay=enrollment_config.get("retry_delay", 5.0),
        )

    @classmethod
    def from_env(cls, output_dir: str) -> "EnrollmentOptions":
        """Create from environment variables."""
        return cls(
            timeout=float(os.getenv("NVFLARE_ENROLLMENT_TIMEOUT", "30.0")),
            output_dir=output_dir,
            max_retries=int(os.getenv("NVFLARE_ENROLLMENT_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("NVFLARE_ENROLLMENT_RETRY_DELAY", "5.0")),
        )

Usage in FederatedClientBase

When integrating with FLARE client startup:

# In fed_client_base.py or auto-enrollment logic

def _perform_auto_enrollment(self, client_args: dict, startup_dir: str):
    """Perform automatic enrollment if certificates are missing."""

    # 1. Get Certificate Service URL
    #    Priority: Environment > enrollment.json
    #    NOT from fed_client.json (not known at package time)
    cert_service_url = os.getenv("NVFLARE_CERT_SERVICE_URL")

    if not cert_service_url:
        enrollment_file = os.path.join(startup_dir, "enrollment.json")
        if os.path.exists(enrollment_file):
            with open(enrollment_file) as f:
                enrollment_config = json.load(f)
            cert_service_url = enrollment_config.get("cert_service_url")

    if not cert_service_url:
        raise ValueError(
            "No Certificate Service URL found. Set NVFLARE_CERT_SERVICE_URL "
            "or create startup/enrollment.json"
        )

    # 2. Get enrollment token
    #    Priority: Environment > token file
    token = os.getenv("NVFLARE_ENROLLMENT_TOKEN")

    if not token:
        token_file = os.path.join(startup_dir, "enrollment_token")
        if os.path.exists(token_file):
            with open(token_file) as f:
                token = f.read().strip()

    if not token:
        raise ValueError(
            "No enrollment token found. Set NVFLARE_ENROLLMENT_TOKEN "
            "or place token in startup/enrollment_token"
        )

    # 3. Get timeout options
    #    Priority: Environment > enrollment.json > fed_client.json > defaults
    options = self._get_enrollment_options(client_args, startup_dir)

    # 4. Create identity from client_args
    identity = EnrollmentIdentity.for_client(
        name=client_args.get("client_name"),
        org=client_args.get("org"),
    )

    # 5. Perform enrollment
    requestor = CertRequestor(
        cert_service_url=cert_service_url,
        enrollment_token=token,
        identity=identity,
        options=options,
    )

    return requestor.request_certificate()

def _get_enrollment_options(
    self, client_args: dict, startup_dir: str
) -> EnrollmentOptions:
    """Get enrollment options from multiple sources."""

    # Start with defaults
    timeout = 30.0
    max_retries = 3
    retry_delay = 5.0

    # Override from fed_client.json
    enrollment_config = client_args.get("enrollment", {})
    timeout = enrollment_config.get("timeout", timeout)
    max_retries = enrollment_config.get("max_retries", max_retries)
    retry_delay = enrollment_config.get("retry_delay", retry_delay)

    # Override from enrollment.json
    enrollment_file = os.path.join(startup_dir, "enrollment.json")
    if os.path.exists(enrollment_file):
        with open(enrollment_file) as f:
            file_config = json.load(f)
        timeout = file_config.get("timeout", timeout)
        max_retries = file_config.get("max_retries", max_retries)
        retry_delay = file_config.get("retry_delay", retry_delay)

    # Override from environment (highest priority)
    timeout = float(os.getenv("NVFLARE_ENROLLMENT_TIMEOUT", str(timeout)))
    max_retries = int(os.getenv("NVFLARE_ENROLLMENT_MAX_RETRIES", str(max_retries)))
    retry_delay = float(os.getenv("NVFLARE_ENROLLMENT_RETRY_DELAY", str(retry_delay)))

    return EnrollmentOptions(
        timeout=timeout,
        output_dir=startup_dir,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )

What Goes Where

| Configuration | When Known | Where Specified |
| --- | ----- | --- |
| cert_service_url | After Certificate Service is deployed | Environment variable or enrollment.json |
| enrollment_token | After token is generated by Project Admin | Environment variable or enrollment_token file |
| timeout, max_retries | At package generation time (optional) | fed_client.json or environment |
| client_name, org | At package generation time | fed_client.json |

Configuration Precedence

- cert_service_url:
    1. `NVFLARE_CERT_SERVICE_URL` (env)
    2. `enrollment.json`
    3. (NOT in fed_client.json)

- enrollment_token:
    1. `NVFLARE_ENROLLMENT_TOKEN` (env)
    2. `enrollment_token` (file)

- timeout, max_retries, retry_delay:
    1. Environment variables (highest)
    2. `enrollment.json`
    3. `fed_client.json`
    4. Default values (lowest)

Deployment Workflow

Option A: Consolidated (Recommended)
─────────────────────────────────────

Site operator receives from Project Admin:
    - Token string
    - Cert Service URL
    - Server endpoint

Single command:
    nvflare package \\
        -n hospital-1 \\
        -e grpc://server:8002 \\
        -t client \\
        --cert-service https://cert-service:8443 \\
        --token eyJhbGciOiJSUzI1...

Start client:
    cd hospital-1 && ./startup/start.sh

─────────────────────────────────────

Option B: K8s Secret / Environment (K8s)
─────────────────────────────────────────

Package generated without enrollment info:
    nvflare package -n hospital-1 -e grpc://server:8002 -t client

Token and URL injected from K8s Secret (via secretRef or volume mount):
    # K8s Secret "hospital-1-enrollment" contains:
    #   NVFLARE_CERT_SERVICE_URL: https://cert-service:8443
    #   NVFLARE_ENROLLMENT_TOKEN: eyJhbGciOiJSUzI1...
    cd hospital-1 && ./startup/start.sh

### Enrollment Flow

Auto-Approved Flow


```
CertRequestor                          Certificate Service
─────────────                          ───────────────────
     │
     │  1. Generate RSA key pair
     │     (locally - never transmitted)
     │
     │  2. Create CSR
     │     (signed with private key)
     │
     │  3. POST /api/v1/enroll
     │     { token, csr, metadata }
     │────────────────────────────────────>│
     │                                     │
     │                                     │  4. Validate token
     │                                     │
     │                                     │  5. Check policy → approve
     │                                     │
     │                                     │  6. Verify CSR signature
     │                                     │
     │                                     │  7. Sign certificate
     │                                     │
     │  200: { certificate, ca_cert }      │
     │<────────────────────────────────────│
     │
     │  8. Save files:
     │     - client.crt (certificate)
     │     - client.key (private key)
     │     - rootCA.pem (root CA)
     │
     │  9. Return EnrollmentResult
     │     (in-memory + file paths)
     ▼

Pending (Manual Approval) Flow

When the approval policy evaluates to pending, the Certificate Service stores the request and returns 202 Accepted (no certificate yet). The client exits with `EnrollmentPending`; there is no polling loop. After an admin approves the request, the Site Admin restarts the client; the client submits again and receives the signed certificate.

Sequence:

1. Client: Same as Auto-Approved (generate key, CSR, POST /api/v1/enroll).
2. Certificate Service: Validates token, evaluates policy → pending, stores request under (name, entity_type), returns 202 with `status: "pending"` and `request_id`.
3. Client: Returns `EnrollmentPending`, startup fails. Site operator waits.
4. Admin: Calls GET /api/v1/pending to list requests, then POST /api/v1/pending/{name}/approve (with type) to approve.
5. Certificate Service: On approve, signs the stored CSR, marks entity enrolled.
6. Site operator: Restarts the client (e.g. `./startup/start.sh` again).
7. Client: POST /api/v1/enroll again (same or new token). Certificate Service looks up (name, entity_type), finds already enrolled, returns 200 with the signed certificate and ca_cert.
8. Client: Saves cert/key/rootCA, returns `EnrollmentResult`, startup continues.

```
CertRequestor              Certificate Service                    Admin
─────────────              ──────────────────                    ─────
     │
     │  1. Keys + CSR, POST /api/v1/enroll
     │────────────────────────────────────>│
     │                                     │  2. Validate token, policy → pending
     │                                     │     Store (name, entity_type)
     │  3. 202 { status: pending }         │
     │<────────────────────────────────────│
     │  EnrollmentPending, exit            │
     ▼                                     │
                                           │  4. GET /api/v1/pending
     [Site operator waits]                 │<─────────────────────────────────>
     [Admin lists, then approves]          │  POST /api/v1/pending/{name}/approve
                                           │  5. Sign CSR, mark enrolled
     [Site operator restarts]              │
     │                                     │
     │  7. POST /api/v1/enroll (retry)     │  7. Lookup (name, entity_type),
     │────────────────────────────────────>│     already enrolled → return cert
     │  8. 200 { certificate, ca_cert }    │
     │<────────────────────────────────────│
     │  Save files, EnrollmentResult
     ▼
```

Key Design Decisions:

1. No polling loop: Client returns immediately with EnrollmentPending error
2. Server tracks by site name: Uses (name, entity_type) as unique key, not token
3. Re-submission on restart: Site operator restarts process after admin approval
4. Server matches by site: Re-submitted request for same site finds existing approved record
5. Timeout: Pending requests expire after 7 days if not approved/rejected
6. No token state tracking: Only enrolled sites and pending requests are tracked (O(sites))

Return Values:

- `status: approved` → Return EnrollmentResult (success)
- `status: pending` → Raise EnrollmentPending(request_id, message)
- `status: rejected` → Raise EnrollmentError(reason)

Site Admin Workflow (Pending):

First attempt returns pending; after the admin approves, run start again:

```bash
# First attempt - returns pending
$ cd hospital-1 && ./startup/start.sh
Enrollment pending: Request abc123 queued for admin approval.
Contact your Project Admin to approve this request.

# ... Admin approves via CLI ...

# Second attempt - succeeds
$ cd hospital-1 && ./startup/start.sh
Enrollment successful. Certificate saved to startup/client.crt
```

### EnrollmentResult

The request_certificate() method returns an EnrollmentResult:

```python
@dataclass
class EnrollmentResult:
    # In-memory (use directly, no reload needed)
    private_key: Any           # RSA private key object
    certificate_pem: str       # PEM-encoded certificate
    ca_cert_pem: str           # PEM-encoded root CA

    # File paths (saved for restart persistence)
    cert_path: str             # Path to client.crt
    key_path: str              # Path to client.key
    ca_path: str               # Path to rootCA.pem
```

Design Note: Files are saved for persistence across restarts, but in-memory certificates can be used directly without reloading.

## Server Enrollment

The FL Server is also a site that requires certificates. In the Auto-Scale workflow, the server enrolls with the Certificate Service just like clients.

### Server vs Client Enrollment

| Aspect | Client Enrollment | Server Enrollment |
| --- | ----- | --- |
| Identity | EnrollmentIdentity.for_client() | EnrollmentIdentity.for_server() |
| Certificate files | client.crt, client.key | server.crt, server.key |
| Entity type | client or relay | server |
| Additional info | Organization | Hostname, FL port, Admin port |

### Server Enrollment Flow


```python
from nvflare.private.fed.client.enrollment import (
    CertRequestor,
    EnrollmentIdentity,
    EnrollmentOptions,
)

# Create server identity
identity = EnrollmentIdentity.for_server(
    name="server1",
    org="My Organization",
    host="server.example.com",    # Server hostname for certificate CN/SAN
)

# Configure options
options = EnrollmentOptions(
    timeout=30.0,
    output_dir="/workspace/startup",
)

# Create requestor
requestor = CertRequestor(
    cert_service_url="https://cert-service.example.com:8443",
    enrollment_token="eyJhbGciOiJSUzI1NiIs...",
    identity=identity,
    options=options,
)

# Perform enrollment
result = requestor.request_certificate()

# Result contains:
# - result.cert_path → server.crt
# - result.key_path → server.key
# - result.ca_path → rootCA.pem
```

### Server Package with Enrollment

Using nvflare package with enrollment options:

```bash

nvflare package  \
    -n server1  \
    -e grpc://0.0.0.0:8002:8003  \
    -t server  \
    --cert-service https://cert-service.example.com:8443  \
    --token eyJhbGciOiJSUzI1NiIs...  \
    -w ./packages

This generates:

./packages/server1/
├── startup/
│   ├── fed_server.json
│   ├── enrollment.json         # Certificate Service URL
│   ├── enrollment_token        # Server enrollment token
│   ├── start.sh
│   └── ...
└── ...

Start the server:

cd server1 && ./startup/start.sh

```

The server will:

1. Detect missing server.crt and server.key
2. Read enrollment token and Certificate Service URL
3. Submit CSR to Certificate Service
4. Receive and save server.crt, server.key, rootCA.pem
5. Continue normal startup with the obtained certificates

### Server Token Generation

Project Admin generates a server token via Certificate Service:


```bash
nvflare token generate  \
    -n server1  \
    --cert-service https://cert-service:8443  \
    --api-key $API_KEY  \
    -o server1.token

```

### Workflow Comparison


```
Manual Workflow (Small Scale)
─────────────────────────────

Site Admin (server operator):
    nvflare cert csr -n server1 -t server -o ./csr
    # Send server1.csr to Project Admin (private key stays local)

Project Admin:
    nvflare cert init -n "Project" -o ./ca          # one-time setup
    nvflare cert sign -r ./csr/server1.csr -c ./ca -o ./signed
    # Send signed server.crt, rootCA.pem, and Server URI to site admin

Site Admin:
    nvflare package -n server1 -e grpc://0.0.0.0:8002:8003 -t server \
        --cert ./signed/server.crt --rootca ./signed/rootCA.pem
    cd server1 && ./startup/start.sh

─────────────────────────────

Auto-Scale Workflow (Large Scale)
─────────────────────────────────

Project Admin:
    nvflare cert init -n "Project" -o ./ca
    # Deploy Certificate Service with rootCA
    nvflare token generate -n server1 \\
        --cert-service https://cert-service:8443 \\
        --api-key $API_KEY
    # Send token + Cert Service URL to server operator

Server Operator:
    nvflare package \\
        -n server1 \\
        -e grpc://0.0.0.0:8002:8003 \\
        -t server \\
        --cert-service https://cert-service:8443 \\
        --token "eyJhbGciOiJSUzI1NiIs..."
    cd server1 && ./startup/start.sh
    # Auto-enrollment happens, server starts with obtained certs

```

## Security Analysis

### Trust Model


```
┌───────────────────────────────────────────────────────────────────┐
│                    ROOT OF TRUST                                  │
│                                                                   │
│    Root CA (rootCA.pem + rootCA.key)                             │
│    - Created by Project Admin                                     │
│    - Private key held ONLY by Certificate Service                │
│    - Public cert distributed to all participants                 │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

> Note: In the Auto-Scale Workflow, the root CA private key is held by the Certificate Service. In the Manual Workflow (no Certificate Service), the root CA private key is held by the Project Admin who signs CSRs locally.

```
                                │
                 Signs all certificates
                                │
       ┌────────────────────────┼────────────────────────┐
       │                        │                        │
       ▼                        ▼                        ▼
┌─────────────┐          ┌─────────────┐          ┌─────────────┐
│   Server    │          │   Client    │          │   Client    │
│ Certificate │          │ Certificate │          │ Certificate │
│             │          │             │          │             │
│ Signed by   │          │ Signed by   │          │ Signed by   │
│ Root CA     │          │ Root CA     │          │ Root CA     │
└─────────────┘          └─────────────┘          └─────────────┘
```

### Key Security Properties

| Property | How It’s Achieved |
| --- | ----- |
| Private keys never transit | Generated locally by each participant |
| Root CA key protected | Only held by Certificate Service (not FL Server) |
| Tokens are tamper-proof | RS256 signature with root CA private key |
| Tokens are single-use | Tracked by Certificate Service (optional) |
| Tokens expire | Built-in expiration (exp claim) |
| mTLS between participants | All certificates signed by same root CA |
| Audit trail | All enrollments logged by Certificate Service |

### Threat Analysis

Threat 1: Compromised FL Server

Impact: Cannot issue new certificates (no root CA key)
Detection: Unusual network patterns, failed auth attempts
Response: Revoke server certificate, re-issue

Threat 2: Stolen Enrollment Token

Impact: Attacker could enroll as the legitimate participant
Mitigations:
  - Short token expiry (hours, not days)
  - Name binding (token locked to specific name)
  - IP restrictions (optional, for static environments)
  - Single-use enforcement

Threat 3: Compromised Certificate Service

Impact: Attacker could issue arbitrary certificates
Mitigations:
  - Network isolation
  - HSM for root CA key (production)
  - Audit logging
  - Access controls

Threat 4: Man-in-the-Middle

Impact: Intercept enrollment requests
Mitigations:
  - TLS for all communication
  - Certificate pinning (optional)

### Comparison: Provisioning vs Token-Based

| Aspect | Current (Provisioning) | Token-Based Enrollment |
| --- | ----- | --- |
| Private key generation | Centralized (Project Admin) | Distributed (each participant) |
| Private keys in transit | Yes (in startup kit) | Never |
| Root CA key location | Project Admin workstation | Certificate Service only |
| Adding new participants | Re-provision, redistribute | Generate token only |
| Startup kit distribution | Full kit via secure channel | Token + package command |
| Audit trail | Manual tracking | Automated logging |

## CLI Commands

This section documents the four new CLI commands for the simplified enrollment system.

### nvflare cert

Generate and manage certificates for the manual workflow.

Location: nvflare/tool/enrollment/cert_cli.py

Subcommands:

| Subcommand | Description |
| --- | ----- |
| init | Initialize a new root CA (creates rootCA.pem + rootCA.key) |
| csr | Generate a private key and certificate signing request (CSR) locally (Site Admin) |
| sign | Sign a CSR with the root CA (Project Admin) |
| site | Generate site certificate signed by root CA (legacy; prefer csr + sign) |
| api-key | Generate a secure API key for Certificate Service authentication |

nvflare cert init

Create a new root Certificate Authority.

```bash
nvflare cert init  \
    -n "My Project"  \             # Project/CA name (required)
    -o ./ca  \                      # Output directory (required)
    --org "My Organization"  \      # Organization name (optional)
    --validity 365                  # CA validity in days (default: 365)

```

Output:

```
./ca/
├── rootCA.pem        # Public certificate (distribute to all sites)
├── rootCA.key        # Private key (keep secure, for signing only)
└── state/
    └── cert.json     # Certificate state for TokenService
```

nvflare cert csr

Generate a private key and CSR locally. The private key never leaves the Site Admin's machine.

```
nvflare cert csr \
    -n hospital-1 \                  # Site name (required)
    -t client \                      # Type: server|client|relay|admin
    -o ./csr                          # Output directory (optional)
```

Output:

```
./csr/
├── hospital-1.key        # Private key (STAYS LOCAL - never send this)
└── hospital-1.csr        # Certificate signing request (send to Project Admin)
```

nvflare cert sign

Sign a CSR with the root CA. Used by the Project Admin.

```
nvflare cert sign \
    -r ./csr/hospital-1.csr \       # CSR file to sign (required)
    -c ./ca \                        # CA directory (required)
    -o ./signed                       # Output directory (optional)
```

Output:

```
./signed/
├── client.crt             # Signed certificate (send back to Site Admin)
└── rootCA.pem             # Root CA cert (send back to Site Admin)
```

nvflare cert site *(legacy)*

Generate a certificate for any site type (server, client, relay, admin) signed by the root CA. This command generates both the private key and signed certificate on the Project Admin's machine. Prefer `nvflare cert csr` + `nvflare cert sign` so that private keys are generated locally by the Site Admin and never transmitted.


```bash
# Generate server certificate
nvflare cert site  \
    -n server1  \                   # Site name (required)
    -t server  \                    # Type: server|client|relay|admin
    -c ./ca  \                      # CA directory (required)
    --host server.example.com  \    # Server hostname for SAN (optional)
    --additional_hosts localhost  \ # Additional hosts for SAN (optional)
    -o ./certs                      # Output directory (optional)

# Generate client certificate
nvflare cert site  \
    -n hospital-1  \                # Site name (required)
    -t client  \                    # Type: client
    -c ./ca  \                      # CA directory (required)
    --org "Hospital A"  \           # Organization (optional)
    -o ./certs                      # Output directory (optional)

# Generate admin certificate
nvflare cert site  \
    -n admin@org.com  \             # Admin email (required)
    -t admin  \                     # Type: admin
    -c ./ca  \                      # CA directory (required)
    --role lead  \                  # Role: lead|member|org_admin (optional)
    -o ./certs                      # Output directory (optional)

# Generate relay certificate
nvflare cert site  \
    -n relay-1  \                   # Relay name (required)
    -t relay  \                     # Type: relay
    -c ./ca  \                      # CA directory (required)
    -o ./certs                      # Output directory (optional)

```

*Options:

| Option | Description |
| --- | ----- |
| -n, --name | Site name (used as certificate CN) |
| -t, --type | Site type: server, client, relay, or admin (default: client) |
| -c, --ca_path | Path to CA directory containing rootCA.pem/rootCA.key or state/cert.json |
| -o, --output | Output directory (default: current directory) |
| --org | Organization name (default: org) |
| --valid_days | Certificate validity in days (default: 365) |
| --host | Default hostname for server SAN extension |
| --additional_hosts | Additional hostnames for server SAN extension |
| --role | Role for admin type: lead, member, org_admin, project_admin |


Output (depends on type):

```
# For server (-t server):
./certs/
├── server.crt        # Server certificate
├── server.key        # Server private key (DO NOT DISTRIBUTE)
└── rootCA.pem        # Root CA (for distribution)

# For client/relay/admin (-t client|relay|admin):
./certs/
├── client.crt        # Client certificate
├── client.key        # Client private key (DO NOT DISTRIBUTE)
└── rootCA.pem        # Root CA (for distribution)
```

> Security warning: `nvflare cert site` generates private keys on the Project Admin's machine. In the legacy workflow these keys would need to be distributed to sites, which is a security risk. Use `nvflare cert csr` + `nvflare cert sign` instead so that private keys are generated locally at each site and never transmitted.

nvflare cert api-key

Generate a secure API key for Certificate Service authentication.

```bash
# Generate and print to stdout (default: 32 bytes / 256 bits, hex format)
nvflare cert api-key

# Generate with custom length and save to file
nvflare cert api-key -l 64 -o api_key.txt

# Generate in base64 format
nvflare cert api-key --format base64
```

Options:

| Option | Description |
| --- | ----- |
| -l, --length | Key length in bytes (default: 32 = 256 bits) |
| -o, --output | Output file path (default: print to stdout) |
| --format | Output format: hex (default), base64, or urlsafe |

Usage:

After generating, use the API key with the Certificate Service:

```bash
# Set as environment variable
export NVFLARE_API_KEY='<generated-key>'

# Or use with CLI commands
nvflare token generate -n site-1 --cert-service https://... --api-key '<generated-key>'
```

### nvflare token

Generate enrollment tokens for the Auto-Scale Workflow only.

Location: nvflare/tool/enrollment/token_cli.py

Note

Manual Workflow does NOT use tokens. In the Manual Workflow, Site Admins generate private keys and CSRs locally using `nvflare cert csr`, send the CSR to the Project Admin who signs it with `nvflare cert sign`, and returns the signed certificate.

Tokens are only needed in the Auto-Scale Workflow where sites enroll dynamically via the Certificate Service.

How Token Generation Works:

In the Auto-Scale Workflow, the rootCA private key resides on the Certificate Service, not with the Project Admin. Therefore, nvflare token calls the Certificate Service API to generate signed tokens.

Subcommands:

| Subcommand | Description |
| --- | ----- |
| generate | Generate a single enrollment token |
| batch | Generate multiple tokens at once |
| info | Inspect and decode a token |

nvflare token generate

Generate a single enrollment token via the Certificate Service API.


```bash
nvflare token generate  \
    -n hospital-1  \                # Site/user name (required)
    --cert-service https://cert-service:8443  \ # Certificate Service URL
    --api-key "admin-jwt..."  \ # Admin authentication token
    -o hospital-1.token             # Output file (optional)

```


Examples:

```bash
# Client token
nvflare token generate -n hospital-1  \
    --cert-service https://cert-service:8443  \
    --api-key "$NVFLARE_API_KEY"

# Relay token
nvflare token generate -n relay-east --relay  \
    --cert-service https://cert-service:8443  \
    --api-key "$NVFLARE_API_KEY"

# User token (default role: lead)
nvflare token generate -n admin@org.com --user  \
    --cert-service https://cert-service:8443  \
    --api-key "$NVFLARE_API_KEY"

# Using environment variables
export NVFLARE_CERT_SERVICE_URL=https://cert-service:8443
export NVFLARE_API_KEY=eyJhbGciOiJSUzI1NiIs...
nvflare token generate -n hospital-1
```

nvflare token batch

Generate multiple tokens at once via the Certificate Service API.


```bash
# Using pattern
nvflare token batch  \
    --pattern "site-{001..100}"  \
    --cert-service https://cert-service:8443  \
    --api-key $NVFLARE_API_KEY  \
    -o ./tokens/

# Using names file
nvflare token batch  \
    --names-file sites.txt  \
    --cert-service https://cert-service:8443  \
    --api-key $NVFLARE_API_KEY  \
    -o ./tokens/

# Using prefix and count
nvflare token batch  \
    --prefix site-  \
    --count 100  \
    --pad 3  \
    --cert-service https://cert-service:8443  \
    --api-key $NVFLARE_API_KEY  \
    -o ./tokens/

```

Output:

./tokens/
├── site-001.token
├── site-002.token
├── ...
└── site-100.token

nvflare token info

Inspect and decode a token without verification.


```bash
nvflare token info -t eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...

Output:

Token Information:
  Subject:      hospital-1
  Subject Type: client
  Issuer:       My Project
  Issued At:    2025-01-04 10:00:00 UTC
  Expires At:   2025-01-05 10:00:00 UTC
  Token ID:     abc123-def456-789

```

### nvflare package

Generate startup kit without certificates (for token-based enrollment).

Location: nvflare/lighter/startup_kit.py

Modes:

1. Single participant mode: Generate one package from CLI arguments
2. Project file mode: Generate all packages from project.yml (without certs)

Single Participant Mode


```bash
nvflare package  \
    -n hospital-1  \                # Participant name (required)
    -e grpc://server:8002  \        # Server endpoint URI (required)
    -t client  \                    # Type: client|relay|server|admin (default: client)
    -w ./packages  \                # Output workspace directory (optional)
    --org "Hospital A"  \           # Organization (optional)
    --cert-service URL  \           # Certificate Service URL (optional)
    --token TOKEN                   # Enrollment token (optional)
```


Enrollment Options (for Auto-Scale workflow):

The Certificate Service URL and enrollment token can be configured in two ways:

| Option | When to Use | How |
| --- | ----- | --- |
| At Package Time | When URL and token are known upfront | nvflare package --cert-service URL --token TOKEN |
| At Startup Time | For K8s/Docker or when token generated later | K8s Secrets (recommended), environment variables, or files |

Option 1: Embed at Package Time (Recommended for simplicity)

When --cert-service and/or --token are provided, the package includes:

startup/enrollment.json with the Certificate Service URL
startup/enrollment_token with the enrollment token

This consolidates the package generation and enrollment setup into one command. The Site Admin just runs: cd hospital-1 && ./startup/start.sh

Option 2: Provide at Startup Time (Recommended for K8s/containers)

Generate package without enrollment options, then inject at runtime via K8s Secrets (e.g. in the startup directory or via env):

```bash
# Generate package (URL and token not known yet)
nvflare package -n hospital-1 -e grpc://server:8002 -t client

# Store token and URL in a K8s Secret
kubectl create secret generic hospital-1-enrollment \
    --from-literal=NVFLARE_CERT_SERVICE_URL=https://cert-service.example.com:8443 \
    --from-literal=NVFLARE_ENROLLMENT_TOKEN=eyJhbGciOiJSUzI1NiIs...

# In pod spec, inject from Secret:
#   envFrom:
#     - secretRef:
#         name: hospital-1-enrollment
cd hospital-1 && ./startup/start.sh
```

For production, use an External Secrets Operator to manage the Secret from your cloud secret manager.

Or place files in the startup directory:

- `startup/enrollment.json`: `{"cert_service_url": "https://..."}`
- `startup/enrollment_token`: The JWT token string

Priority Order (highest to lowest):

1. Environment variables (NVFLARE_CERT_SERVICE_URL, NVFLARE_ENROLLMENT_TOKEN)
2. Files in startup directory (enrollment.json, enrollment_token)

Environment variables always override embedded files, allowing flexibility for multi-environment deployments (dev, staging, prod).

Endpoint URI Formats:

| Format | Description |
| --- | ----- |
| grpc://host:port | gRPC with single port (admin port = fl_port + 1) |
| grpc://host:fl_port:admin_port | gRPC with explicit ports |
| http://host:port | HTTP/HTTPS protocol |
| tcp://host:port | TCP protocol |

Examples:

```bash
# Client package
nvflare package  \
    -n hospital-1  \
    -e grpc://server.example.com:8002  \
    -t client  \
    -w ./packages

# Server package (two-port format)
nvflare package  \
    -n server1  \
    -e grpc://0.0.0.0:8002:8003  \
    -t server  \
    -w ./packages

# Relay package
nvflare package  \
    -n relay-east  \
    -e grpc://server:8002  \
    -t relay  \
    --listening-host 0.0.0.0  \
    --listening-port 8102  \
    -w ./packages

# Admin package
nvflare package  \
    -n admin@org.com  \
    -e grpc://server:8003  \
    -t admin  \
    --role lead  \
    -w ./packages

# Client package with enrollment (Auto-Scale workflow)
nvflare package  \
    -n hospital-1  \
    -e grpc://server:8002  \
    -t client  \
    --cert-service https://cert-service.example.com:8443  \
    --token eyJhbGciOiJSUzI1NiIs...  \
    -w ./packages
```

Output (without enrollment options):

```
./packages/hospital-1/
├── local/
│   ├── authorization.json.default
│   └── ...
├── startup/
│   ├── fed_client.json
│   ├── start.sh
│   └── ...
└── transfer/
```

Output (with --cert-service and --token):

```
./packages/hospital-1/
├── local/
│   ├── authorization.json.default
│   ├── log_config.json.default
│   ├── privacy.json.sample
│   └── resources.json.default
├── startup/
│   ├── fed_client.json
│   ├── enrollment.json         # ← Certificate Service URL
│   ├── enrollment_token        # ← Enrollment token
│   ├── start.sh
│   ├── stop_fl.sh
│   └── sub_start.sh
└── transfer/
```

Project File Mode

Generate packages for all participants defined in a project.yml:

```bash
nvflare package -p project.yml -w ./packages
```


This filters out CertBuilder and SignatureBuilder from the project, generating packages without certificates (ready for token-based enrollment).

### nvflare enrollment

Manage pending enrollment requests (Admin CLI for Certificate Service).

Location: nvflare/tool/enrollment/enrollment_cli.py

Subcommands:

| Subcommand | Description |
| --- | ----- |
| list | List pending enrollment requests |
| info | View details of a pending request |
| approve | Approve a pending request |
| reject | Reject a pending request |
| enrolled | List enrolled entities |

Environment Variables:

- NVFLARE_CERT_SERVICE_URL: Certificate Service URL
- NVFLARE_API_KEY: Admin authentication token

nvflare enrollment list

List pending enrollment requests.


```bash
nvflare enrollment list                    # All pending
nvflare enrollment list --type client      # Sites only
nvflare enrollment list --type user        # Users only
```


Output:


```bash
Name           Type      Org          Submitted             Status
──────────────────────────────────────────────────────────────────
hospital-1     client    Hospital A   2025-01-04 10:15:00   pending
hospital-2     client    Hospital B   2025-01-04 10:20:00   pending
admin@org.com  user      Org Inc      2025-01-04 11:00:00   pending
```


nvflare enrollment info

View details of a specific pending request.


```bash
nvflare enrollment info hospital-1 --type client
```


Output:


```bash
Name:            hospital-1
Type:            client
Organization:    Hospital A
Submitted:       2025-01-04 10:15:00 UTC
Expires:         2025-01-11 10:15:00 UTC
Token Subject:   hospital-*
Source IP:       10.2.3.4
CSR Subject:     CN=hospital-1, O=Hospital A
```


nvflare enrollment approve

Approve pending enrollment requests.


```bash
# Approve single request
nvflare enrollment approve hospital-1 --type client
```



```bash
# Approve user request
nvflare enrollment approve admin@org.com --type user
```



```bash
# Bulk approve by pattern
nvflare enrollment approve --pattern "hospital-*" --type client
```


nvflare enrollment reject

Reject pending enrollment requests.


```bash
nvflare enrollment reject hospital-2 --type client  \
    --reason "Site not authorized for this project"
```


nvflare enrollment enrolled

List enrolled entities.


```bash
nvflare enrollment enrolled                # All enrolled
nvflare enrollment enrolled --type client  # Sites only
nvflare enrollment enrolled --type user    # Users only
```


Output:


```bash
Name           Type      Org          Enrolled At
────────────────────────────────────────────────────
hospital-1     client    Hospital A   2025-01-04 12:00:00
hospital-3     client    Hospital C   2025-01-04 12:30:00
admin@org.com  user      Org Inc      2025-01-04 13:00:00
```


### CLI Summary

| Command | Workflow | Purpose |
| --- | ----- | --- |
| nvflare cert init | Both | Initialize root CA (Project Admin) |
| nvflare cert csr | Manual | Generate private key + CSR locally (Site Admin) |
| nvflare cert sign | Manual | Sign a CSR with root CA (Project Admin) |
| nvflare cert site | Manual (legacy) | Generate certificate directly (prefer csr + sign) |
| nvflare token | Auto-Scale | Generate enrollment tokens |
| nvflare package | Both | Generate startup kits locally |
| nvflare enrollment | Auto-Scale | Manage pending requests (admin) |

## Workflows

Both workflows share the same CSR-based security model: private keys are always generated locally and never leave the site. The difference is how the CSR signing is orchestrated — manually by the Project Admin (Manual) or automatically by the Certificate Service using a token sent only to invited participants (Auto-Scale).

### Workflow 1: Manual (Small Scale)

For 5-10 participants. No Certificate Service needed.

```
Step 1: Site Admin              Step 2: Exchange             Step 3: Project Admin          Step 4: Site Admin
──────────────────              ────────────────             ─────────────────────          ──────────────────

Generate private key            Send CSR only                Receive CSR                    Receive signed cert
and CSR locally:                (private key stays           (no private key):              + rootCA.pem
                                local):                                                     + Server URI
nvflare cert csr \                                          nvflare cert sign \
   -n hospital-1 \              hospital-1.csr               -r hospital-1.csr \
   -t client \               ──────────────────►             -c ./ca \                    Generate startup kit:
   -o ./csr                                                   -o ./signed
                                                                  │                         nvflare package \
Output:                                                           ▼                            -n hospital-1 \
  ./csr/                         ◄──────────────────         signed/                            -e grpc://server:8002 \
  ├── hospital-1.key             Send back:                  ├── client.crt                     -t client \
  │   (STAYS LOCAL)              • client.crt                └── rootCA.pem                     --cert ./signed/client.crt \
  └── hospital-1.csr             • rootCA.pem                                                   --rootca ./signed/rootCA.pem
                                 • Server URI                                                       │
                                   (host + ports)                                                   ▼
                                                                                              cd hospital-1 && \
                                                                                                ./startup/start.sh
```

### Workflow 2: Auto-Scale (Large Scale)

For 10+ participants or dynamic environments.


```
Phase 1: Setup (Project Admin)
──────────────────────────────

1. nvflare cert init -n "Project" -o ./ca

2. Deploy Certificate Service:
   - Configure with rootCA.pem + rootCA.key
   - Start on https://cert-service.example.com

Phase 2: Generate Tokens (Project Admin)
────────────────────────────────────────

# Tokens are generated via Certificate Service API
# (rootCA private key is on the service, not local)

nvflare token generate -n hospital-1 \\
    --cert-service https://cert-service.example.com \\
    --api-key $API_KEY

# Or batch generate:
nvflare token batch \\
    --pattern "hospital-{001..100}" \\
    --cert-service https://cert-service.example.com \\
    --api-key $API_KEY

Distribute to each site:
   - Token string
   - Certificate Service URL

Phase 3: Site Enrollment (Site Admin)
────────────────────────────────────────

Option A: Consolidated command (recommended)

    nvflare package \\
        -n hospital-1 \\
        -e grpc://server:8002 \\
        -t client \\
        --cert-service https://cert-service.example.com \\
        --token "eyJhbGciOiJSUzI1NiIs..."

    cd hospital-1 && ./startup/start.sh

Option B: K8s Secret injection (for K8s/containers)

    nvflare package -n hospital-1 -e grpc://server:8002 -t client

    # Store token and URL in K8s Secret:
    #   kubectl create secret generic hospital-1-enrollment \
    #       --from-literal=NVFLARE_ENROLLMENT_TOKEN=eyJ... \
    #       --from-literal=NVFLARE_CERT_SERVICE_URL=https://cert-service.example.com
    #
    # Pod spec: envFrom: [{secretRef: {name: hospital-1-enrollment}}]

    cd hospital-1 && ./startup/start.sh

(Auto-enrollment happens at startup)

## Implementation Details
```


This section provides an overview of all implementation components. Detailed API documentation and code examples are in the sections referenced below.

### Component Overview

| Component | Location | Purpose |
| --- | ----- | --- |
| Certificate Service | nvflare/cert_service/ | HTTP service for token generation and CSR signing |
| CertService | cert_service.py | Core logic: token validation, policy evaluation, CSR signing |
| CertServiceApp | app.py | HTTP/Flask wrapper for CertService |
| EnrollmentStore | store.py | Tracks enrolled entities and pending requests (SQLite/PostgreSQL) |
| Client Enrollment | nvflare/security/enrollment/ | Client-side enrollment components |
| CertRequestor | cert_requestor.py | Generates keys, creates CSR, submits to Certificate Service via HTTP |
| EnrollmentIdentity | cert_requestor.py | Pydantic model for client/server/relay/admin identity with validation |
| EnrollmentOptions | cert_requestor.py | Pydantic model for configuration options (timeout, retry_count, output_dir) |
| EnrollmentResult | cert_requestor.py | Return value containing cert paths and in-memory objects |
| Helper functions | __init__.py | get_enrollment_token(), get_cert_service_url(), enroll() |
| CLI Tools | nvflare/tool/enrollment/ | Command-line interfaces |
| token_cli | token_cli.py | nvflare token command |
| cert_cli | cert_cli.py | nvflare cert command |
| enrollment_cli | enrollment_cli.py | nvflare enrollment command |
| TokenService | token_service.py | JWT token generation logic |
| Package Generator | nvflare/lighter/ | Startup kit generation |
| startup_kit | startup_kit.py | nvflare package command |

### Certificate Service Components

See Certificate Service section for:

HTTP API endpoints
Configuration file format
EnrollmentStore interface
Deployment options

### Client Enrollment Components

See Client Enrollment and Server Enrollment sections for:

CertRequestor usage
EnrollmentOptions configuration
EnrollmentResult dataclass
Auto-enrollment flow

### CLI Components

See CLI Commands section for:

nvflare cert - Certificate generation (Manual Workflow)
nvflare token - Token generation via Certificate Service API
nvflare package - Startup kit generation
nvflare enrollment - Pending request management

### FLARE Integration

### Auto-Enrollment Integration Points

Auto-enrollment is integrated into three FLARE components:

1. FederatedClientBase (nvflare/private/fed/client/fed_client_base.py)
2. FederatedServer (nvflare/private/fed/server/fed_server.py)
3. AdminAPI (nvflare/fuel/hci/client/api.py)

Each component calls _auto_enroll_if_needed() at startup.

**Client/Server Auto-Enrollment Flow:**


```python
# In FederatedClientBase._auto_enroll_if_needed() and FederatedServer._auto_enroll_if_needed()


from nvflare.security.enrollment import (
    get_enrollment_token, get_cert_service_url, enroll, EnrollmentIdentity
)

def _auto_enroll_if_needed(self) -> bool:
    """Perform automatic enrollment if certificates don't exist but token is available."""

    # 1. Check for existing certificates
    cert_path = self.client_args.get(SecureTrainConst.SSL_CERT)
    if cert_path and os.path.exists(cert_path):
        return False  # Already enrolled

    # 2. Get token (from env var or file)
    token = get_enrollment_token(startup_dir)
    if not token:
        self.logger.debug("No enrollment token found, skipping auto-enrollment")
        return False

    # 3. Get Certificate Service URL (from env var or enrollment.json)
    cert_service_url = get_cert_service_url(startup_dir)
    if not cert_service_url:
        raise RuntimeError("Certificate Service URL required for enrollment")

    # 4. Perform enrollment
    identity = EnrollmentIdentity.for_client(name=self.client_name, org=self.org)
    result = enroll(cert_service_url, token, identity, startup_dir)

    # 5. Update runtime args with new certificate paths
    self.client_args[SecureTrainConst.SSL_CERT] = result.cert_path
    self.client_args[SecureTrainConst.PRIVATE_KEY] = result.key_path
    self.client_args[SecureTrainConst.SSL_ROOT_CERT] = result.ca_path
    return True

**Helper Functions** (`nvflare/security/enrollment/__init__.py`):


```python
def get_enrollment_token(startup_dir: str = None) -> str:
    """Get token from NVFLARE_ENROLLMENT_TOKEN env var or enrollment_token file."""
    token = os.environ.get("NVFLARE_ENROLLMENT_TOKEN")
    if token:
        return token.strip()
    if startup_dir:
        token_file = os.path.join(startup_dir, "enrollment_token")
        if os.path.exists(token_file):
            with open(token_file, "r") as f:
                return f.read().strip()
    return None
```


```python
def get_cert_service_url(startup_dir: str = None) -> str:
    """Get URL from NVFLARE_CERT_SERVICE_URL env var or enrollment.json file."""
    url = os.environ.get("NVFLARE_CERT_SERVICE_URL")
    if url:
        return url.strip()
    if startup_dir:
        config_file = os.path.join(startup_dir, "enrollment.json")
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
                return config.get("cert_service_url")
    return None

def enroll(cert_service_url: str, token: str, identity: EnrollmentIdentity,
           output_dir: str = ".") -> EnrollmentResult:
    """Convenience function to perform enrollment."""
    options = EnrollmentOptions(output_dir=output_dir)
    requestor = CertRequestor(
        cert_service_url=cert_service_url,
        enrollment_token=token,
        identity=identity,
        options=options,
    )
    return requestor.request_certificate()
```

**No Changes to Existing FLARE Code**

The token-based enrollment is **additive** and does not modify:

- Existing authentication mechanisms
- CellNet communication
- FL training workflows
- Job execution

It simply provides an alternative way to obtain certificates before normal FLARE operations begin.

### Key Design Notes

1. **HTTP-based, not CellNet**: The Certificate Service uses standard HTTP/REST, not CellNet. This keeps certificate management separate from FL operations.
2. **No authentication bypass**: Sites must obtain valid certificates before connecting to the FL Server. There is no unauthenticated path.
3. **Token is not for authentication**: The enrollment token grants eligibility to request a certificate. The certificate is then used for mTLS authentication.
4. **Stateless tokens**: Tokens are JWTs with embedded policies. No token state is tracked; only enrolled sites and pending requests are tracked.

## Backward Compatibility

The token-based enrollment system is fully backward compatible:

1. **Existing provisioned deployments**: Continue to work unchanged
2. **No migration required**: Can adopt gradually
3. **Coexistence**: Some sites provisioned, some enrolled

## Future Enhancements

1. **Certificate rotation**: Automatic renewal before expiry
2. **Revocation**: CRL or OCSP support
3. **Dashboard integration**: Web UI for token and enrollment management
4. **Notification webhooks**: Alert admin when requests are pending

