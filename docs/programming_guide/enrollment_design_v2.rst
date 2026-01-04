.. _enrollment_design_v2:

#################################################
FLARE Simplified Enrollment System
#################################################

*Version: 2.0*
*Status: Draft*
*Author: NVIDIA FLARE Team*

.. contents:: Table of Contents
   :local:
   :depth: 3

***********************
Executive Summary
***********************

This document describes a simplified enrollment system for NVIDIA FLARE
that reduces the complexity of participant onboarding while maintaining
strong security.

**Problem Statement**

The current provisioning workflow requires:

1. Project Admin to generate complete startup kits for every participant
2. Secure distribution of startup kits (containing private keys)
3. Manual coordination for adding new participants

**Proposed Solution**

Two new workflows that eliminate centralized startup kit generation:

1. **Manual Workflow** (5-10 participants)
   
   - Project Admin generates certificates using CLI (``nvflare cert``)
   - Certificates distributed via secure channel (email, USB)
   - Sites generate their own startup kits locally (``nvflare package``)
   - No additional infrastructure required

2. **Auto-Scale Workflow** (10+ participants or dynamic)
   
   - Project Admin deploys a Certificate Service
   - Sites receive enrollment tokens instead of certificates
   - Sites auto-enroll at startup via HTTP
   - Supports dynamic addition of participants

**Key Benefits**

- **Private keys generated locally**: Never transmitted over network
- **Simplified distribution**: Certificates (Manual) or tokens (Auto-Scale) instead of full startup kits
- **Sites generate own packages**: No centralized startup kit creation
- **Flexible scaling**: Manual for small, auto-scale for large deployments
- **Clear separation**: FL workloads vs certificate management

**Design Goals Addressed**

This design was created with the following goals in mind:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Goal
     - How Addressed
   * - **5-minute setup**
     - Single command for site operators: ``nvflare package --cert-service --token``
   * - **Easier than provisioning**
     - No ``project.yml`` needed; no distribution of startup kits
   * - **Add clients mid-project**
     - Generate token + send to site. Zero touch for existing participants
   * - **Dynamic K8s scaling**
     - Batch tokens + StatefulSet with auto-enrollment at startup
   * - **No DevOps expertise**
     - Simple CLI commands with sensible defaults
   * - **Private keys stay local**
     - Keys generated on each site, never transmitted

**Complexity Comparison**

.. list-table::
   :widths: 35 20 25 20
   :header-rows: 1

   * - Scenario
     - Current Provisioning
     - Manual Workflow
     - Auto-Scale Workflow
   * - Add 1 new client
     - 4 steps
     - 3 steps
     - **1 command**
   * - Add 100 clients (K8s)
     - 100 kits + distribution
     - 100 certs + distribution
     - **1 batch token + deploy**
   * - Site operator setup
     - Receive kit, extract, start
     - Receive certs, run package, start
     - **1 command + start**
   * - Project admin effort
     - Manage all kits
     - Generate certs on demand
     - **Generate tokens only**

***********************
Background
***********************

Current Provisioning Workflow
=============================

Today, NVIDIA FLARE uses ``nvflare provision`` to create startup kits:

.. code-block:: text

    Project Admin                    Distribution              Site Operator
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

The Provisioning Problem
========================

The provisioning workflow has been consistently identified as a major source
of friction for NVFLARE users:

**Rigidity**

In NVFLARE, you cannot simply "connect" a client to a server. You must
"provision" the entire network upfront. A project admin defines a ``project.yml``
file listing every participant, then generates startup kits that must be
securely distributed out-of-band.

If a new client wants to join mid-experiment, the admin must perform manual
dynamic-provisioning with the same rootCA - a process that is not familiar to DevOps.

**Opacity**

Errors in the ``project.yml`` (e.g., mismatching port definitions or domain names)
often result in cryptic "Connection Refused" or "Cannot find path to server"
errors, making debugging difficult.

**DevOps Burden**

Compared with frameworks like OpenFL or Flower, which support dynamic joining
of clients via simple token authentication, NVFLARE's provisioning feels
archaic to researchers used to cloud-native flexibility. This architecture
impedes rapid prototyping and requires substantial DevOps expertise.

**FLARE Dashboard**

Flare Dashboard simplifies the startup kit distribution. But still requires
users to sign up the site, get approval (manual) and download the startup kit.
The process is still pretty rigid.

**Summary of Limitations**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Limitation
     - Description
   * - **Private keys in transit**
     - Keys are generated centrally and must be distributed securely
   * - **Manual distribution**
     - Each startup kit must be sent to the recipient via secure channel
   * - **Static pre-shared trust**
     - All participants must be known upfront; adding new ones requires
       manual dynamic-provisioning with the same rootCA
   * - **Scalability**
     - Difficult to manage 100+ participants with manual distribution
   * - **DevOps expertise required**
     - Configuration errors lead to cryptic failures

How the New Design Addresses These Concerns
===========================================

The following table shows how each limitation is addressed by the new workflows:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Concern
     - Manual Workflow
     - Auto-Scale Workflow
   * - **Adding client mid-project**
     - Generate cert with ``nvflare cert client``. rootCA.pem, site.crt and site.key are sent to the client.
     - Generate token with ``nvflare token generate``, send to site. Site auto-enrolls.
   * - **Private keys in transit**
     - Keys still distributed, but only certs (not full startup kits)
     - **Keys generated locally, never transmitted**
   * - **Manual distribution**
     - Distribute certs only (smaller, simpler)
     - **Distribute tokens only** (short strings, can be sent via any channel)
   * - **Static pre-shared trust**
     - New participants statically
     - **Dynamic enrollment** - sites join on-demand with tokens
   * - **Scalability**
     - Better for 5-10 participants
     - **Scales to 100+ participants** with automated enrollment
   * - **DevOps expertise**
     - Simple CLI commands
     - Simple CLI + one-time Certificate Service setup

**Example: Adding a Client Mid-Project**

*Current Provisioning:*

.. code-block:: text

    1. Update project.yml with new participant
    2. Run nvflare provision (or dynamic-provision with same rootCA)
    3. Locate the new participant's startup kit
    4. Securely distribute the entire startup kit

*Manual Workflow:*

.. code-block:: shell

    # Project Admin (30 seconds)
    nvflare cert client -n new-hospital -c ./ca -o ./certs
    # Send new-hospital.crt, new-hospital.key, rootCA.pem to site
    
    # Site Operator (2 minutes)
    nvflare package -n new-hospital -e grpc://server:8002 -t client
    cp received_certs/* new-hospital/startup/
    cd new-hospital && ./startup/start.sh

*Auto-Scale Workflow:*

.. code-block:: shell

    # Project Admin (30 seconds)
    nvflare token generate -n new-hospital \
        --cert-service https://cert-service:8443 \
        --api-key $API_KEY
    # Send token + Certificate Service URL to site
    
    # Site Operator (1 command!)
    nvflare package \
        -n new-hospital \
        -e grpc://server:8002 \
        -t client \
        --cert-service https://cert-service:8443 \
        --token "eyJhbGciOiJSUzI1NiIs..."
    cd new-hospital && ./startup/start.sh
    # Auto-enrollment happens, site joins the federation

**Key Improvement**: Neither workflow requires touching existing participants
or re-provisioning the network.

**Example: Kubernetes Dynamic Scaling to 100 Clients**

This example shows how to use the Auto-Scale workflow to dynamically scale
a federated learning deployment to 100 clients in Kubernetes.

*Step 1: Project Admin - Batch Generate Tokens (one-time)*

.. code-block:: shell

    # Generate 100 tokens via Certificate Service API
    nvflare token batch \
        --pattern "site-{001..100}" \
        --cert-service https://cert-service.example.com:8443 \
        --api-key $NVFLARE_API_KEY \
        -o ./tokens/
    
    # Result: tokens/site-001.token, tokens/site-002.token, ..., tokens/site-100.token

*Step 2: Create Kubernetes Secret with All Tokens*

.. code-block:: shell

    # Create one secret containing all 100 tokens
    kubectl create secret generic flare-tokens \
        --from-file=./tokens/ \
        --namespace=flare

A Kubernetes **Secret** stores sensitive data. To use it in a pod, you mount it
as a **volume**. Each key in the secret becomes a file in the mount path:

.. code-block:: text

    Secret "flare-tokens"              Mounted Volume at /tokens/
    ─────────────────────              ──────────────────────────
    key: site-001.token    ──────►     file: /tokens/site-001.token
    key: site-002.token    ──────►     file: /tokens/site-002.token
    ...                                ...
    key: site-100.token    ──────►     file: /tokens/site-100.token

*Step 3: Kubernetes StatefulSet*

In a StatefulSet, pods are named ``flare-client-0``, ``flare-client-1``, etc.
Each pod derives its site name from its ordinal and reads the corresponding token.

.. code-block:: yaml

    # flare-client-statefulset.yaml
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
                  # Extract pod ordinal from hostname (flare-client-0 -> 0)
                  ORDINAL=${HOSTNAME##*-}
                  
                  # Convert to site name with zero-padding (0 -> site-001)
                  SITE_NAME=$(printf "site-%03d" $((ORDINAL + 1)))
                  echo "Generating package for: $SITE_NAME"
                  
                  # Generate startup kit
                  nvflare package -n $SITE_NAME \
                      -e grpc://flare-server.flare.svc:8002 \
                      -t client \
                      -o /workspace
                  
                  # Save site name for main container
                  echo $SITE_NAME > /workspace/site_name.txt
              volumeMounts:
                - name: workspace
                  mountPath: /workspace
          containers:
            - name: flare-client
              image: nvflare/nvflare:latest
              command: ["/bin/sh", "-c"]
              args:
                - |
                  # Read site name from init container
                  SITE_NAME=$(cat /workspace/site_name.txt)
                  
                  # Read token from mounted secret
                  # Secret files are named: site-001.token, site-002.token, etc.
                  export NVFLARE_ENROLLMENT_TOKEN=$(cat /tokens/${SITE_NAME}.token)
                  export NVFLARE_CERT_SERVICE_URL="https://cert-service.flare.svc:8443"
                  
                  echo "Starting $SITE_NAME with token from /tokens/${SITE_NAME}.token"
                  
                  # Start client (auto-enrollment happens here)
                  cd /workspace && ./startup/start.sh
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
                secretName: flare-tokens  # Contains all 100 tokens

*Step 4: Deploy and Scale*

.. code-block:: shell

    # Deploy initial set
    kubectl apply -f flare-client-statefulset.yaml
    
    # Scale up to 50 clients
    kubectl scale statefulset flare-client --replicas=50 -n flare
    
    # Scale up to 100 clients
    kubectl scale statefulset flare-client --replicas=100 -n flare
    
    # Each pod:
    # 1. Generates its startup kit (init container)
    # 2. Auto-enrolls with Certificate Service using its token
    # 3. Joins the federation automatically

*Key Benefits for Kubernetes:*

- **No pre-built images**: Clients generate packages at startup
- **Dynamic scaling**: Scale up/down with ``kubectl scale``
- **Token-based identity**: Each pod has its own token in a secret
- **Auto-enrollment**: No manual certificate distribution
- **Stateless pods**: Can be replaced/recreated without re-provisioning

PKI and Certificate Concepts
============================

Before diving into the design, here are the key PKI concepts:

**Root CA (Certificate Authority)**

- **rootCA.pem**: Public certificate - distributed to all participants for verification
- **rootCA.key**: Private key - used to sign other certificates, must be protected

**Participant Certificates**

- **client.crt / server.crt**: Public certificate signed by root CA
- **client.key / server.key**: Private key for the participant

**CSR (Certificate Signing Request)**

A request containing a public key and identity information, submitted to a CA
for signing. The private key never leaves the requestor's machine.

**JWT (JSON Web Token)**

A signed token containing claims. Used for enrollment tokens that embed
approval policies.

***********************
Design Overview
***********************

Architecture
============

The token-based enrollment system consists of:

.. code-block:: text

                                    PROJECT ADMIN
                                    ─────────────
                                          │
                              nvflare token generate
                              nvflare token batch
                              nvflare token info
                                          │
                                    HTTPS API
                                          │
                                          ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                       CERTIFICATE SERVICE                                │
    │                                                                          │
    │  ┌────────────────────────────────────────────────────────────────────┐ │
    │  │                    CertServiceApp (HTTP)                           │ │
    │  │                                                                    │ │
    │  │  POST /api/v1/token   ─► Token generation (for nvflare token CLI) │ │
    │  │  GET  /api/v1/token   ─► Token info                               │ │
    │  │  POST /api/v1/enroll  ─► CSR signing (for CertRequestor)          │ │
    │  │  GET  /api/v1/pending ─► List pending requests                    │ │
    │  │                                                                    │ │
    │  └────────────────────────────┬───────────────────────────────────────┘ │
    │                               │                                          │
    │  ┌────────────────────────────▼───────────────────────────────────────┐ │
    │  │                    CertService (Core Logic)                        │ │
    │  │                                                                    │ │
    │  │  TokenService  ─► JWT generation with embedded policy             │ │
    │  │  Token validation, Policy evaluation, CSR signing                 │ │
    │  │                                                                    │ │
    │  │  Holds rootCA.key (never leaves this service)                     │ │
    │  └────────────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────────┘
                                            │
                            HTTPS (TLS - Let's Encrypt)
                                            │
          ┌─────────────────────────────────┼─────────────────────────────────┐
          │                                 │                                 │
          ▼                                 ▼                                 ▼
    ┌───────────────┐              ┌───────────────┐              ┌───────────────┐
    │   FL Server   │              │   FL Client   │              │   FL Client   │
    │               │              │               │              │               │
    │ CertRequestor │              │ CertRequestor │              │ CertRequestor │
    │               │              │               │              │               │
    │ 1. Gen keys   │◄── mTLS ───►│ 1. Gen keys   │              │ 1. Gen keys   │
    │ 2. Create CSR │              │ 2. Create CSR │              │ 2. Create CSR │
    │ 3. Submit CSR │              │ 3. Submit CSR │              │ 3. Submit CSR │
    │ 4. Get cert   │              │ 4. Get cert   │              │ 4. Get cert   │
    └───────────────┘              └───────────────┘              └───────────────┘

Key Components
==============

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Component
     - Location
     - Responsibility
   * - **TokenService**
     - ``nvflare/tool/enrollment/token_service.py``
     - Generate JWT enrollment tokens with embedded policies
   * - **CertService**
     - ``nvflare/cert_service/cert_service.py``
     - Validate tokens, evaluate policies, sign CSRs
   * - **CertServiceApp**
     - ``nvflare/cert_service/app.py``
     - HTTP wrapper exposing CertService via REST API
   * - **CertRequestor**
     - ``nvflare/private/fed/client/enrollment/cert_requestor.py``
     - Client-side: generate keys, create CSR, submit for signing

***********************
Enrollment Token
***********************

Token Structure
===============

Enrollment tokens are JWTs (JSON Web Tokens) signed with the root CA private key.
They are tamper-proof and contain all information needed for enrollment.

**JWT Structure:**

.. code-block:: json

    {
      "header": {
        "alg": "RS256",
        "typ": "JWT"
      },
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
            "rules": [
              { "name": "auto-approve", "match": {}, "action": "approve" }
            ]
          }
        }
      },
      "signature": "..."
    }

**Token Claims:**

.. list-table::
   :widths: 15 20 65
   :header-rows: 1

   * - Claim
     - Example
     - Description
   * - ``jti``
     - UUID
     - Unique token identifier (for audit/tracking)
   * - ``sub``
     - "hospital-1"
     - Subject - the participant name or pattern
   * - ``subject_type``
     - "client"
     - Participant type: client, server, relay, admin
   * - ``iss``
     - "MyProjectCA"
     - Issuer - extracted from root CA certificate
   * - ``iat``
     - timestamp
     - Issued at time
   * - ``exp``
     - timestamp
     - Expiration time
   * - ``policy``
     - {...}
     - Embedded approval policy (see below)
   * - ``roles``
     - ["lead"]
     - Roles for admin tokens

Token Security
==============

**Why JWT with RS256?**

1. **Tamper-proof**: Tokens are signed with the root CA private key
2. **Self-contained**: No database lookup needed to validate
3. **Decentralized validation**: Any service with the public key can verify
4. **Expiration**: Built-in expiry prevents indefinite use

**Security Properties:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - How It's Achieved
   * - Cannot be forged
     - Signed with root CA private key (RS256)
   * - Cannot be tampered
     - Signature verification detects any modification
   * - Cannot be reused indefinitely
     - Expiration time (``exp`` claim)
   * - Single-use enforcement
     - Certificate Service tracks used tokens
   * - Scoped to participant
     - Subject (``sub``) specifies who can use it

**Attack Mitigations:**

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Attack
     - Risk
     - Mitigation
   * - Token theft
     - Attacker uses stolen token
     - Short expiry + single-use + name binding
   * - Token forgery
     - Attacker creates fake token
     - RS256 signature verification
   * - Token tampering
     - Attacker modifies claims
     - JWT signature detects changes
   * - Replay attack
     - Reuse of valid token
     - Single-use tracking + expiration
   * - Brute force
     - Guess valid tokens
     - UUID-based JTI + rate limiting

***********************
Approval Policy
***********************

Policy Structure
================

The approval policy is embedded in the token and evaluated during enrollment.

**Policy Schema:**

.. code-block:: yaml

    # approval_policy.yaml
    
    metadata:
      project: "my-fl-project"
      description: "Enrollment policy for hospital network"
      version: "1.0"
    
    token:
      validity: "7d"              # Token expiration
    
    site:
      name_pattern: "^hospital-[0-9]+$"  # Allowed site names
    
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

Policy Elements
===============

**Metadata**

Identifies the policy scope and version.

.. code-block:: yaml

    metadata:
      project: "my-fl-project"
      description: "Policy description"
      version: "1.0"

**Token Configuration**

Controls token lifetime.

.. code-block:: yaml

    token:
      validity: "7d"     # Supports: 30m, 2h, 7d, etc.

**Site Constraints**

Restricts which site names are allowed.

.. code-block:: yaml

    site:
      name_pattern: "^hospital-[0-9]+$"   # Regex pattern

**User Constraints**

For admin tokens, controls allowed roles.

.. code-block:: yaml

    user:
      allowed_roles:
        - lead
        - member
      default_role: lead

**Approval Rules**

Rules are evaluated in order - first match wins.

.. code-block:: yaml

    approval:
      method: policy
      rules:
        - name: rule_name
          description: "Human-readable description"
          match:
            site_name_pattern: "^pattern-.*"   # Optional
            source_ips:                        # Optional
              - "10.0.0.0/8"
          action: approve | reject | pending
          message: "Reason message"
          log: true

**Match Conditions**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Condition
     - Description
   * - ``site_name_pattern``
     - Wildcard pattern for site name (e.g., "hospital-*")
   * - ``source_ips``
     - CIDR ranges for client IP (optional, for static environments)

**Actions**

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Action
     - Description
   * - ``approve``
     - Automatically approve and sign the certificate
   * - ``reject``
     - Reject the enrollment request
   * - ``pending``
     - Queue for manual approval; admin must approve/reject via CLI

Policy Evaluation Flow
======================

.. code-block:: text

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

***********************
CSR Signing Process
***********************

CSR Generation (Client-Side)
============================

The ``CertRequestor`` generates a key pair and CSR locally:

.. code-block:: python

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

**Key point**: The private key is generated locally and never transmitted.

CSR Signing (Certificate Service)
=================================

The ``CertService`` validates the request and signs the CSR:

.. code-block:: python

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

Certificate Contents
====================

The signed certificate includes:

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Field
     - OID
     - Value
   * - Common Name (CN)
     - 2.5.4.3
     - Participant name (e.g., "hospital-1")
   * - Organization (O)
     - 2.5.4.10
     - Organization name (optional)
   * - Organizational Unit (OU)
     - 2.5.4.11
     - Participant type (client, server, relay, admin)
   * - Unstructured Name
     - 1.2.840.113549.1.9.2
     - Role for admin tokens (lead, member, etc.)

***********************
Certificate Service
***********************

Overview
========

The Certificate Service is a standalone HTTP service that handles enrollment.
It is deployed separately from the FL Server.

**Why Separate?**

1. **Security isolation**: Root CA private key is not on FL Server
2. **Scalability**: Can handle many concurrent enrollments
3. **Audit**: Centralized logging of all certificate issuance
4. **Blast radius**: If FL Server is compromised, attacker cannot issue certs

Architecture
============

.. code-block:: text

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

HTTP API
========

**POST /api/v1/enroll**

Enrollment endpoint. Returns signed certificate and root CA.

*Request:*

.. code-block:: json

    {
        "token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
        "csr": "-----BEGIN CERTIFICATE REQUEST-----\n...",
        "metadata": {
            "name": "hospital-1",
            "type": "client",
            "org": "Hospital A"
        }
    }

*Response (200 OK):*

.. code-block:: json

    {
        "certificate": "-----BEGIN CERTIFICATE-----\n...",
        "ca_cert": "-----BEGIN CERTIFICATE-----\n..."
    }

*Response (202 Accepted - Pending Manual Approval):*

.. code-block:: json

    {
        "status": "pending",
        "request_id": "abc123-def456-789",
        "message": "Enrollment request queued for manual approval",
        "poll_url": "/api/v1/enroll/abc123-def456-789"
    }

*Error Responses:*

- **401 Unauthorized**: Invalid or expired token
- **403 Forbidden**: Policy rejection
- **400 Bad Request**: Invalid request format

**GET /api/v1/enroll/{request_id}**

Poll for pending request status.

*Response (200 - Still Pending):*

.. code-block:: json

    {
        "status": "pending",
        "submitted_at": "2025-01-04T10:15:00Z"
    }

*Response (200 - Approved):*

.. code-block:: json

    {
        "status": "approved",
        "certificate": "-----BEGIN CERTIFICATE-----\n...",
        "ca_cert": "-----BEGIN CERTIFICATE-----\n..."
    }

*Response (200 - Rejected):*

.. code-block:: json

    {
        "status": "rejected",
        "reason": "Site not authorized for this project"
    }

*Response (404):*

Request ID not found or expired.

**GET /api/v1/pending** (Admin Only)

List all pending enrollment requests. Requires admin authentication.

*Query Parameters:*

- ``type`` (optional): Filter by entity type (client, relay, user)

*Response:*

.. code-block:: json

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

**POST /api/v1/pending/{name}/approve** (Admin Only)

Approve a pending enrollment request.

*Query Parameters:*

- ``type`` (required): Entity type (client, relay, user)

*Response (200):*

.. code-block:: json

    {
        "status": "approved",
        "name": "hospital-1",
        "entity_type": "client",
        "certificate_issued": true
    }

**POST /api/v1/pending/{name}/reject** (Admin Only)

Reject a pending enrollment request.

*Query Parameters:*

- ``type`` (required): Entity type (client, relay, user)

*Request:*

.. code-block:: json

    {
        "reason": "Not authorized for this project"
    }

*Response (200):*

.. code-block:: json

    {
        "status": "rejected",
        "name": "hospital-1",
        "entity_type": "client"
    }

**GET /api/v1/enrolled** (Admin Only)

List all enrolled entities.

*Query Parameters:*

- ``type`` (optional): Filter by entity type

*Response:*

.. code-block:: json

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

**POST /api/v1/token** (Admin Only)
Generate an enrollment token. Used by ``nvflare token generate`` when the
rootCA private key is on the Certificate Service.

*Request:*

.. code-block:: json

    {
        "name": "hospital-1",
        "entity_type": "client",
        "valid_days": 7,
        "policy_override": {}
    }

*Response (200 OK):*

.. code-block:: json

    {
        "token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
        "subject": "hospital-1",
        "expires_at": "2025-01-11T10:00:00Z"
    }

*Request (Batch):*

.. code-block:: json

    {
        "names": ["site-001", "site-002", "site-003"],
        "entity_type": "client",
        "valid_days": 7
    }

*Response (200 OK - Batch):*

.. code-block:: json

    {
        "tokens": [
            {"name": "site-001", "token": "eyJhbGci..."},
            {"name": "site-002", "token": "eyJhbGci..."},
            {"name": "site-003", "token": "eyJhbGci..."}
        ]
    }

**GET /health**

Health check endpoint.

*Response:*

.. code-block:: json

    {"status": "healthy"}

Pending Request Storage
=======================

Pending requests are stored by the Certificate Service until:

- Approved by admin (certificate issued)
- Rejected by admin
- Expired (default: 7 days timeout)

Storage options:

1. **File-based** (simple deployments)
2. **SQLite** (single-node service)
3. **PostgreSQL** (production, multi-node)

Admin CLI for Pending Requests
==============================

Administrators manage pending requests via CLI:

.. code-block:: bash

    # List all pending enrollment requests
    nvflare enrollment list
    
    # Output:
    # Name           Type      Org          Submitted             Token Subject    Status
    # ──────────────────────────────────────────────────────────────────────────────────────
    # hospital-1     client    Hospital A   2025-01-04 10:15:00   hospital-*       pending
    # hospital-2     client    Hospital B   2025-01-04 10:20:00   hospital-*       pending
    # admin@org.com  user      Org Inc      2025-01-04 11:00:00   *@org.com        pending
    
    # Filter by type
    nvflare enrollment list --type client    # Sites only
    nvflare enrollment list --type user      # Users only
    
    # View details of a specific request
    nvflare enrollment info hospital-1 --type client
    
    # Output:
    # Name:            hospital-1
    # Type:            client
    # Organization:    Hospital A
    # Submitted:       2025-01-04 10:15:00 UTC
    # Expires:         2025-01-11 10:15:00 UTC
    # Token Subject:   hospital-*
    # Source IP:       10.2.3.4
    # CSR Subject:     CN=hospital-1, O=Hospital A
    
    # View user request
    nvflare enrollment info admin@org.com --type user
    
    # Output:
    # Name:            admin@org.com
    # Type:            user
    # Organization:    Org Inc
    # Role:            lead
    # Submitted:       2025-01-04 11:00:00 UTC
    # ...
    
    # Approve a pending request (specify type)
    nvflare enrollment approve hospital-1 --type client
    nvflare enrollment approve admin@org.com --type user
    
    # Reject a pending request with reason
    nvflare enrollment reject hospital-2 --type client --reason "Not authorized"
    
    # Bulk approve matching pattern
    nvflare enrollment approve --pattern "hospital-*" --type client
    
    # List enrolled entities
    nvflare enrollment enrolled                    # All
    nvflare enrollment enrolled --type client      # Sites only
    nvflare enrollment enrolled --type user        # Users only
    
    # Configuration (environment variables)
    # NVFLARE_CERT_SERVICE_URL - Certificate Service URL
    # NVFLARE_API_KEY - Admin authentication token

Configuration
=============

The Certificate Service is configured via a YAML file. This configuration is
loaded at startup and controls all aspects of the service.

**Configuration File:**

.. code-block:: yaml

    # cert_service_config.yaml
    
    server:
      host: 0.0.0.0
      port: 8443
      tls:
        cert: /path/to/service.crt   # Public TLS cert (Let's Encrypt)
        key: /path/to/service.key
    
    ca:
      cert: /path/to/rootCA.pem      # FLARE root CA (public)
      key: /path/to/rootCA.key       # FLARE root CA (private)
    
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

**Configuration Sections Explained:**

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Section
     - Purpose
   * - ``server``
     - HTTP server binding and TLS configuration for the service endpoint
   * - ``ca``
     - FLARE root CA used to sign enrollment certificates
   * - ``policy``
     - Approval policy for enrollment requests
   * - ``storage``
     - Backend for tracking enrolled entities and pending requests
   * - ``pending``
     - Pending request timeout and cleanup settings
   * - ``audit``
     - Audit logging for compliance

**How CertServiceApp Uses Configuration:**

.. code-block:: python

    # nvflare/cert_service/app.py
    
    import yaml
    from flask import Flask
    from nvflare.cert_service.cert_service import CertService
    from nvflare.cert_service.store import create_enrollment_store
    
    
    class CertServiceApp:
        """HTTP wrapper for CertService."""
        
        def __init__(self, config_path: str):
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
            
            # Initialize enrollment store (SQLite or PostgreSQL)
            self.store = create_enrollment_store(self.config["storage"])
            
            # Initialize CertService with CA and policy
            self.cert_service = CertService(
                root_ca_cert=self.config["ca"]["cert"],
                root_ca_key=self.config["ca"]["key"],
                policy_file=self.config.get("policy", {}).get("file"),
                enrollment_store=self.store,
                pending_timeout=self.config["pending"]["timeout"],
            )
            
            # Create Flask app
            self.app = Flask(__name__)
            self._register_routes()
            
            # Start cleanup scheduler
            self._start_cleanup_scheduler(
                interval=self.config["pending"]["cleanup_interval"]
            )
        
        def _register_routes(self):
            @self.app.route("/api/v1/enroll", methods=["POST"])
            def enroll():
                # Handle enrollment request
                ...
            
            @self.app.route("/api/v1/pending", methods=["GET"])
            def list_pending():
                # List pending requests (admin only)
                ...
            
            # ... other routes ...
        
        def run(self):
            """Start the HTTPS server."""
            self.app.run(
                host=self.config["server"]["host"],
                port=self.config["server"]["port"],
                ssl_context=(
                    self.config["server"]["tls"]["cert"],
                    self.config["server"]["tls"]["key"],
                ),
            )

**Starting the Certificate Service:**

.. code-block:: shell

    # Option 1: Direct Python
    python -m nvflare.cert_service.app --config /path/to/cert_service_config.yaml
    
    # Option 2: Using CLI (if implemented)
    nvflare cert-service start --config /path/to/cert_service_config.yaml
    
    # Option 3: Docker
    docker run -d \
        -v /path/to/config:/config \
        -v /path/to/ca:/ca:ro \
        -p 8443:8443 \
        nvflare/cert-service:latest \
        --config /config/cert_service_config.yaml

**Configuration Flow:**

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────────┐
    │                    cert_service_config.yaml                         │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      CertServiceApp.__init__()                      │
    │                                                                     │
    │   server: ──────► Flask app binding (host, port, TLS)              │
    │                                                                     │
    │   ca: ──────────► CertService (root CA for signing)                │
    │                                                                     │
    │   policy: ──────► CertService (approval rules)                     │
    │                                                                     │
    │   storage: ─────► EnrollmentStore (SQLite or PostgreSQL)           │
    │                                                                     │
    │   pending: ─────► Cleanup scheduler (timeout, interval)            │
    │                                                                     │
    │   audit: ───────► Audit logger (file, enabled)                     │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        HTTPS Server Running                         │
    │                                                                     │
    │   POST /api/v1/enroll          ─► Token validation + CSR signing   │
    │   GET  /api/v1/enroll/{id}     ─► Check pending status             │
    │   GET  /api/v1/pending         ─► List pending (admin)             │
    │   POST /api/v1/pending/{n}/approve ─► Approve request (admin)      │
    │   GET  /health                 ─► Health check                     │
    └─────────────────────────────────────────────────────────────────────┘

**Two TLS Certificates Explained:**

The configuration has two different TLS certificates:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Config Key
     - Purpose
     - Example
   * - ``server.tls.cert``
     - HTTPS endpoint TLS (public trust)
     - Let's Encrypt, DigiCert, etc.
   * - ``ca.cert``
     - FLARE root CA (signs FL certs)
     - Self-signed, project-specific

The ``server.tls`` certificate enables browsers and clients to trust the
Certificate Service endpoint (like any HTTPS website).

The ``ca`` certificate is the FLARE project's root CA that signs FL participant
certificates. These are separate trust domains.

Enrollment Store
================

The Certificate Service tracks enrolled sites and pending requests using a
pluggable storage backend. Default is SQLite.

**Entity Types**

Enrollment applies to both sites and users:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Entity Type
     - Examples
     - Description
   * - ``client``
     - hospital-1, site-001
     - FL client sites
   * - ``relay``
     - relay-east, relay-1
     - Hierarchical FL relay nodes
   * - ``server``
     - server1
     - FL server (Manual workflow only)
   * - ``user``
     - admin@org.com, researcher-1
     - FLARE Console users (with roles)

**Abstract Interface**

.. code-block:: python

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

.. code-block:: python

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
                    "DELETE FROM pending_requests WHERE expires_at < ?",
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

**PostgreSQL Implementation (Production)**

Located in ``nvflare/app_opt/cert_service/postgres_store.py``:

.. code-block:: python

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
                        CREATE TABLE IF NOT EXISTS enrolled_sites (
                            site_name TEXT PRIMARY KEY,
                            enrolled_at TIMESTAMP NOT NULL
                        );
                        
                        CREATE TABLE IF NOT EXISTS pending_requests (
                            site_name TEXT PRIMARY KEY,
                            participant_type TEXT NOT NULL,
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

**Factory Function**

.. code-block:: python

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

**Storage Comparison**

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Feature
     - SQLite (Default)
     - PostgreSQL
   * - Deployment
     - Single node
     - Multi-node / HA
   * - Dependencies
     - Built-in (Python stdlib)
     - Requires ``psycopg2``
   * - Concurrency
     - Limited (file locks)
     - Full ACID
   * - Backup
     - Copy file
     - pg_dump / replication
   * - Use Case
     - Dev, small production
     - Large production, HA

Deployment Options
==================

1. **Standalone container** (Docker/K8s)
2. **Cloud-managed** (AWS, Azure, GCP)

***********************
Client Enrollment
***********************

CertRequestor
=============

The ``CertRequestor`` handles client-side enrollment:

.. code-block:: python

    from nvflare.private.fed.client.enrollment import (
        CertRequestor,
        EnrollmentIdentity,
        EnrollmentOptions,
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

EnrollmentOptions Configuration
===============================

``EnrollmentOptions`` can be configured via:

1. **Direct instantiation** (for testing/scripts)
2. **FLARE client configuration** (``fed_client.json``)
3. **Environment variables** (for containerized deployments)

**Option 1: Direct Instantiation**

.. code-block:: python

    options = EnrollmentOptions(
        timeout=30.0,
        output_dir="/workspace/startup",
    )

**Option 2: From FLARE Client Configuration**

Enrollment settings in ``fed_client.json`` (timeouts only, NOT the URL):

.. code-block:: json

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

.. note::

    The ``cert_service_url`` is intentionally NOT in ``fed_client.json`` because:
    
    1. The Certificate Service URL is not known when the startup kit is generated
    2. The URL may change between environments (dev, staging, prod)
    3. The URL should be provided at deployment time, not package time
    
    Use environment variables or a separate file for the URL (see below).

Then load from ``client_args``:

.. code-block:: python

    @staticmethod
    def from_client_args(client_args: dict, output_dir: str) -> "EnrollmentOptions":
        """Create EnrollmentOptions from FLARE client configuration.
        
        Args:
            client_args: Client configuration dictionary (from fed_client.json)
            output_dir: Directory to save certificates
        
        Returns:
            EnrollmentOptions configured from client_args
        """
        enrollment_config = client_args.get("enrollment", {})
        
        return EnrollmentOptions(
            timeout=enrollment_config.get("timeout", 30.0),
            output_dir=output_dir,
            max_retries=enrollment_config.get("max_retries", 3),
            retry_delay=enrollment_config.get("retry_delay", 5.0),
        )

**Option 3: From Environment Variables (Recommended for URL)**

The Certificate Service URL and token are provided via environment variables,
which are set at deployment time (not package time):

.. code-block:: bash

    # Required for enrollment
    NVFLARE_CERT_SERVICE_URL=https://cert-service.example.com:8443
    NVFLARE_ENROLLMENT_TOKEN=eyJhbGciOiJSUzI1NiIs...
    
    # Optional (have sensible defaults)
    NVFLARE_ENROLLMENT_TIMEOUT=30.0
    NVFLARE_ENROLLMENT_MAX_RETRIES=3
    NVFLARE_ENROLLMENT_RETRY_DELAY=5.0

.. code-block:: python

    @staticmethod
    def from_env(output_dir: str) -> "EnrollmentOptions":
        """Create EnrollmentOptions from environment variables.
        
        Environment Variables:
            NVFLARE_ENROLLMENT_TIMEOUT: HTTP request timeout (default: 30.0)
            NVFLARE_ENROLLMENT_MAX_RETRIES: Max retry attempts (default: 3)
            NVFLARE_ENROLLMENT_RETRY_DELAY: Delay between retries (default: 5.0)
        
        Returns:
            EnrollmentOptions configured from environment
        """
        import os
        
        return EnrollmentOptions(
            timeout=float(os.getenv("NVFLARE_ENROLLMENT_TIMEOUT", "30.0")),
            output_dir=output_dir,
            max_retries=int(os.getenv("NVFLARE_ENROLLMENT_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("NVFLARE_ENROLLMENT_RETRY_DELAY", "5.0")),
        )

**Option 4: From Separate Enrollment File**

For non-containerized deployments, a separate ``enrollment.json`` can be placed
in the startup directory at deployment time.

**Consolidated Package Command (Recommended)**

The ``nvflare package`` command can include the Certificate Service URL and
token, creating everything in one step:

.. code-block:: bash

    # Site operator runs single command with all info from Project Admin
    nvflare package \
        -n hospital-1 \
        -e grpc://server.example.com:8002 \
        -t client \
        --cert-service https://cert-service.example.com:8443 \
        --token eyJhbGciOiJSUzI1NiIs... \
        -o ./

This generates:

.. code-block:: text

    ./hospital-1/
    ├── startup/
    │   ├── fed_client.json
    │   ├── enrollment.json         # Created with cert_service_url
    │   ├── enrollment.token        # Created with token
    │   ├── start.sh
    │   └── ...
    └── ...

**Manual File Creation (Alternative)**

If the package was already generated without enrollment info:

.. code-block:: bash

    # Create enrollment.json
    cat > ./hospital-1/startup/enrollment.json << EOF
    {
        "cert_service_url": "https://cert-service.example.com:8443"
    }
    EOF
    
    # Place token
    echo "eyJhbGciOiJSUzI1..." > ./hospital-1/startup/enrollment.token

**File contents:**

.. code-block:: json

    {
        "cert_service_url": "https://cert-service.example.com:8443",
        "timeout": 30.0,
        "max_retries": 3,
        "retry_delay": 5.0
    }

.. note::

    For containerized/K8s deployments, use environment variables instead
    (``NVFLARE_CERT_SERVICE_URL``, ``NVFLARE_ENROLLMENT_TOKEN``), which is
    simpler and more flexible than files.

**EnrollmentOptions Dataclass**

.. code-block:: python

    from dataclasses import dataclass, field
    from typing import Optional
    import os
    
    
    @dataclass
    class EnrollmentOptions:
        """Configuration options for certificate enrollment.
        
        Args:
            timeout: HTTP request timeout in seconds
            output_dir: Directory to save certificates
            max_retries: Maximum number of retry attempts on failure
            retry_delay: Delay between retries in seconds
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

**Usage in FederatedClientBase**

When integrating with FLARE client startup:

.. code-block:: python

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
            token_file = os.path.join(startup_dir, "enrollment.token")
            if os.path.exists(token_file):
                with open(token_file) as f:
                    token = f.read().strip()
        
        if not token:
            raise ValueError(
                "No enrollment token found. Set NVFLARE_ENROLLMENT_TOKEN "
                "or place token in startup/enrollment.token"
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

**What Goes Where**

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Configuration
     - When Known
     - Where Specified
   * - ``cert_service_url``
     - **After** Certificate Service is deployed
     - Environment variable or ``enrollment.json``
   * - ``enrollment_token``
     - **After** token is generated by Project Admin
     - Environment variable or ``enrollment.token`` file
   * - ``timeout``, ``max_retries``
     - At package generation time (optional)
     - ``fed_client.json`` or environment
   * - ``client_name``, ``org``
     - At package generation time
     - ``fed_client.json``

**Configuration Precedence**

.. code-block:: text

    cert_service_url:
        1. NVFLARE_CERT_SERVICE_URL (env)
        2. enrollment.json
        3. (NOT in fed_client.json)
    
    enrollment_token:
        1. NVFLARE_ENROLLMENT_TOKEN (env)
        2. enrollment.token (file)
    
    timeout, max_retries, retry_delay:
        1. Environment variables (highest)
        2. enrollment.json
        3. fed_client.json
        4. Default values (lowest)

**Deployment Workflow**

.. code-block:: text

    Option A: Consolidated (Recommended)
    ─────────────────────────────────────
    
    Site operator receives from Project Admin:
        - Token string
        - Cert Service URL
        - Server endpoint
    
    Single command:
        nvflare package \
            -n hospital-1 \
            -e grpc://server:8002 \
            -t client \
            --cert-service https://cert-service:8443 \
            --token eyJhbGciOiJSUzI1...
    
    Start client:
        cd hospital-1 && ./startup/start.sh
    
    ─────────────────────────────────────
    
    Option B: Environment Variables (K8s)
    ─────────────────────────────────────
    
    Package generated without enrollment info:
        nvflare package -n hospital-1 -e grpc://server:8002 -t client
    
    Set environment and start:
        export NVFLARE_CERT_SERVICE_URL=https://cert-service:8443
        export NVFLARE_ENROLLMENT_TOKEN=eyJhbGciOiJSUzI1...
        cd hospital-1 && ./startup/start.sh

Enrollment Flow
===============

**Auto-Approved Flow**

.. code-block:: text

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
         │────────────────────────────────────►│
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
         │◄────────────────────────────────────│
         │
         │  8. Save files:
         │     - client.crt (certificate)
         │     - client.key (private key)
         │     - rootCA.pem (root CA)
         │
         │  9. Return EnrollmentResult
         │     (in-memory + file paths)
         ▼

**Pending (Manual Approval) Flow**

When policy action is ``pending``, the request is queued for admin approval.
The client returns immediately - **no polling loop**. The site operator must
restart the enrollment process later (after admin approval).

.. code-block:: text

    CertRequestor                 Certificate Service              Admin
    ─────────────                 ───────────────────              ─────
         │
         │  1-3. Same as above
         │────────────────────────────►│
         │                             │
         │                             │  4. Validate token (JWT)
         │                             │
         │                             │  5. Check policy → pending
         │                             │
         │                             │  6. Store pending request
         │  202: { status: pending,    │     key: (site_name, type)
         │         request_id: ... }   │
         │◄────────────────────────────┤
         │                             │
         │  Return EnrollmentPending   │
         │  (site startup fails)       │
         ▼                             │
                                       │          GET /pending
    [Site operator waits for admin]    │◄─────────────────────────────┤
                                       │  [List pending requests]     │
                                       │─────────────────────────────►│
                                       │                              │
                                       │    POST /pending/{name}/approve
                                       │◄─────────────────────────────┤
                                       │                              │
                                       │  Sign certificate            │
                                       │  Mark site as enrolled       │
    [Site operator restarts]           │
         │                             │
         │  POST /api/v1/enroll        │
         │  (new token or same)        │
         │────────────────────────────►│
         │                             │  Lookup by site_name:
         │  200: { certificate,        │  already enrolled → return cert
         │         ca_cert }           │
         │◄────────────────────────────┤
         │
         │  8-9. Save + return result
         ▼

**Key Design Decisions:**

1. **No polling loop**: Client returns immediately with ``EnrollmentPending`` error
2. **Server tracks by site name**: Uses ``(name, entity_type)`` as unique key, not token
3. **Re-submission on restart**: Site operator restarts process after admin approval
4. **Server matches by site**: Re-submitted request for same site finds existing approved record
5. **Timeout**: Pending requests expire after 7 days if not approved/rejected
6. **No token state tracking**: Only enrolled sites and pending requests are tracked (O(sites))

**Return Values:**

- ``status: approved`` → Return ``EnrollmentResult`` (success)
- ``status: pending`` → Raise ``EnrollmentPending(request_id, message)``
- ``status: rejected`` → Raise ``EnrollmentError(reason)``

**Site Operator Workflow (Pending):**

.. code-block:: bash

    # First attempt - returns pending
    $ cd hospital-1 && ./startup/start.sh
    Enrollment pending: Request abc123 queued for admin approval.
    Contact your Project Admin to approve this request.
    
    # ... Admin approves via CLI ...
    
    # Second attempt - succeeds
    $ cd hospital-1 && ./startup/start.sh
    Enrollment successful. Certificate saved to startup/client.crt

EnrollmentResult
================

The ``request_certificate()`` method returns an ``EnrollmentResult``:

.. code-block:: python

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

**Design Note**: Files are saved for persistence across restarts, but
in-memory certificates can be used directly without reloading.

***********************
Server Enrollment
***********************

The FL Server is also a site that requires certificates. In the Auto-Scale
workflow, the server enrolls with the Certificate Service just like clients.

Server vs Client Enrollment
===========================

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - Client Enrollment
     - Server Enrollment
   * - Identity
     - ``EnrollmentIdentity.for_client()``
     - ``EnrollmentIdentity.for_server()``
   * - Certificate files
     - ``client.crt``, ``client.key``
     - ``server.crt``, ``server.key``
   * - Entity type
     - ``client`` or ``relay``
     - ``server``
   * - Additional info
     - Organization
     - Hostname, FL port, Admin port

Server Enrollment Flow
======================

.. code-block:: python

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

Server Package with Enrollment
==============================

Using ``nvflare package`` with enrollment options:

.. code-block:: bash

    nvflare package \
        -n server1 \
        -e grpc://0.0.0.0:8002:8003 \
        -t server \
        --cert-service https://cert-service.example.com:8443 \
        --token eyJhbGciOiJSUzI1NiIs... \
        -o ./packages

This generates:

.. code-block:: text

    ./packages/server1/
    ├── startup/
    │   ├── fed_server.json
    │   ├── enrollment.json         # Certificate Service URL
    │   ├── enrollment.token        # Server enrollment token
    │   ├── start.sh
    │   └── ...
    └── ...

Start the server:

.. code-block:: bash

    cd server1 && ./startup/start.sh

The server will:

1. Detect missing ``server.crt`` and ``server.key``
2. Read enrollment token and Certificate Service URL
3. Submit CSR to Certificate Service
4. Receive and save ``server.crt``, ``server.key``, ``rootCA.pem``
5. Continue normal startup with the obtained certificates

Server Token Generation
=======================

Project Admin generates a server token via Certificate Service:

.. code-block:: bash

    nvflare token generate \
        -n server1 \
        --cert-service https://cert-service:8443 \
        --api-key $API_KEY \
        -o server1.token

Workflow Comparison
===================

.. code-block:: text

    Manual Workflow (Small Scale)
    ─────────────────────────────
    
    Project Admin:
        nvflare cert init -n "Project" -o ./ca
        nvflare cert server -n server1 -c ./ca --host server.example.com
        # Send server.crt, server.key, rootCA.pem to server operator
    
    Server Operator:
        nvflare package -n server1 -e grpc://0.0.0.0:8002:8003 -t server
        # Copy received certs to startup/
        cd server1 && ./startup/start.sh
    
    ─────────────────────────────
    
    Auto-Scale Workflow (Large Scale)
    ─────────────────────────────────
    
    Project Admin:
        nvflare cert init -n "Project" -o ./ca
        # Deploy Certificate Service with rootCA
        nvflare token generate -n server1 \
            --cert-service https://cert-service:8443 \
            --api-key $API_KEY
        # Send token + Cert Service URL to server operator
    
    Server Operator:
        nvflare package \
            -n server1 \
            -e grpc://0.0.0.0:8002:8003 \
            -t server \
            --cert-service https://cert-service:8443 \
            --token "eyJhbGciOiJSUzI1NiIs..."
        cd server1 && ./startup/start.sh
        # Auto-enrollment happens, server starts with obtained certs

***********************
Security Analysis
***********************

Trust Model
===========

.. code-block:: text

    ┌───────────────────────────────────────────────────────────────────┐
    │                    ROOT OF TRUST                                  │
    │                                                                   │
    │    Root CA (rootCA.pem + rootCA.key)                             │
    │    - Created by Project Admin                                     │
    │    - Private key held ONLY by Certificate Service                │
    │    - Public cert distributed to all participants                 │
    │                                                                   │
    └───────────────────────────────────────────────────────────────────┘
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

Key Security Properties
=======================

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - How It's Achieved
   * - **Private keys never transit**
     - Generated locally by each participant
   * - **Root CA key protected**
     - Only held by Certificate Service (not FL Server)
   * - **Tokens are tamper-proof**
     - RS256 signature with root CA private key
   * - **Tokens are single-use**
     - Tracked by Certificate Service (optional)
   * - **Tokens expire**
     - Built-in expiration (exp claim)
   * - **mTLS between participants**
     - All certificates signed by same root CA
   * - **Audit trail**
     - All enrollments logged by Certificate Service

Threat Analysis
===============

**Threat 1: Compromised FL Server**

- **Impact**: Cannot issue new certificates (no root CA key)
- **Detection**: Unusual network patterns, failed auth attempts
- **Response**: Revoke server certificate, re-issue

**Threat 2: Stolen Enrollment Token**

- **Impact**: Attacker could enroll as the legitimate participant
- **Mitigations**:
  - Short token expiry (hours, not days)
  - Name binding (token locked to specific name)
  - IP restrictions (optional, for static environments)
  - Single-use enforcement

**Threat 3: Compromised Certificate Service**

- **Impact**: Attacker could issue arbitrary certificates
- **Mitigations**:
  - Network isolation
  - HSM for root CA key (production)
  - Audit logging
  - Access controls

**Threat 4: Man-in-the-Middle**

- **Impact**: Intercept enrollment requests
- **Mitigations**:
  - TLS for all communication
  - Certificate pinning (optional)

Comparison: Provisioning vs Token-Based
=======================================

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Aspect
     - Current (Provisioning)
     - Token-Based Enrollment
   * - Private key generation
     - Centralized (Project Admin)
     - Distributed (each participant)
   * - Private keys in transit
     - Yes (in startup kit)
     - **Never**
   * - Root CA key location
     - Project Admin workstation
     - Certificate Service only
   * - Adding new participants
     - Re-provision, redistribute
     - Generate token only
   * - Startup kit distribution
     - Full kit via secure channel
     - Token + package command
   * - Audit trail
     - Manual tracking
     - Automated logging

***********************
CLI Commands
***********************

This section documents the four new CLI commands for the simplified enrollment system.

nvflare cert
============

Generate and manage certificates for the manual workflow.

**Location:** ``nvflare/tool/enrollment/cert_cli.py``

**Subcommands:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Subcommand
     - Description
   * - ``init``
     - Initialize a new root CA (creates rootCA.pem + rootCA.key)
   * - ``server``
     - Generate server certificate signed by root CA
   * - ``client``
     - Generate client certificate signed by root CA

**nvflare cert init**

Create a new root Certificate Authority.

.. code-block:: shell

    nvflare cert init \
        -n "My Project" \              # Project/CA name (required)
        -o ./ca \                       # Output directory (required)
        --org "My Organization" \       # Organization name (optional)
        --validity 365                  # CA validity in days (default: 365)

*Output:*

.. code-block:: text

    ./ca/
    ├── rootCA.pem        # Public certificate (distribute to all sites)
    ├── rootCA.key        # Private key (keep secure, for signing only)
    └── state/
        └── cert.json     # Certificate state for TokenService

**nvflare cert server**

Generate a server certificate signed by the root CA.

.. code-block:: shell

    nvflare cert server \
        -n server1 \                    # Server name (required)
        -c ./ca \                       # CA directory (required)
        --host server.example.com \     # Server hostname (required)
        --port 8002 \                   # FL port (default: 8002)
        --admin-port 8003 \             # Admin port (default: 8003)
        -o ./certs                      # Output directory (optional)

*Output:*

.. code-block:: text

    ./certs/
    ├── server1.crt       # Server certificate
    └── server1.key       # Server private key

**nvflare cert client**

Generate a client certificate signed by the root CA.

.. code-block:: shell

    nvflare cert client \
        -n hospital-1 \                 # Client/site name (required)
        -c ./ca \                       # CA directory (required)
        --org "Hospital A" \            # Organization (optional)
        -o ./certs                      # Output directory (optional)

*Output:*

.. code-block:: text

    ./certs/
    ├── hospital-1.crt    # Client certificate
    └── hospital-1.key    # Client private key

nvflare token
=============

Generate enrollment tokens for the **Auto-Scale Workflow only**.

**Location:** ``nvflare/tool/enrollment/token_cli.py``

.. note::

    **Manual Workflow does NOT use tokens.** In the Manual Workflow, Project Admin
    signs certificates directly using ``nvflare cert client`` and distributes them.
    
    Tokens are only needed in the Auto-Scale Workflow where sites enroll
    dynamically via the Certificate Service.

**How Token Generation Works:**

In the Auto-Scale Workflow, the rootCA private key resides on the Certificate
Service, not with the Project Admin. Therefore, ``nvflare token`` calls the
Certificate Service API to generate signed tokens.

**Subcommands:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Subcommand
     - Description
   * - ``generate``
     - Generate a single enrollment token
   * - ``batch``
     - Generate multiple tokens at once
   * - ``info``
     - Inspect and decode a token

**nvflare token generate**

Generate a single enrollment token via the Certificate Service API.

.. code-block:: shell

    nvflare token generate \
        -n hospital-1 \                 # Site/user name (required)
        --cert-service https://cert-service:8443 \  # Certificate Service URL
        --api-key "admin-jwt..." \  # Admin authentication token
        -o hospital-1.token             # Output file (optional)

*Examples:*

.. code-block:: shell

    # Client token
    nvflare token generate -n hospital-1 \
        --cert-service https://cert-service:8443 \
        --api-key "$NVFLARE_API_KEY"
    
    # Relay token
    nvflare token generate -n relay-east --relay \
        --cert-service https://cert-service:8443 \
        --api-key "$NVFLARE_API_KEY"
    
    # User token (default role: lead)
    nvflare token generate -n admin@org.com --user \
        --cert-service https://cert-service:8443 \
        --api-key "$NVFLARE_API_KEY"
    
    # Using environment variables
    export NVFLARE_CERT_SERVICE_URL=https://cert-service:8443
    export NVFLARE_API_KEY=eyJhbGciOiJSUzI1NiIs...
    nvflare token generate -n hospital-1

**nvflare token batch**

Generate multiple tokens at once via the Certificate Service API.

.. code-block:: shell

    # Using pattern
    nvflare token batch \
        --pattern "site-{001..100}" \
        --cert-service https://cert-service:8443 \
        --api-key $NVFLARE_API_KEY \
        -o ./tokens/
    
    # Using names file
    nvflare token batch \
        --names-file sites.txt \
        --cert-service https://cert-service:8443 \
        --api-key $NVFLARE_API_KEY \
        -o ./tokens/
    
    # Using prefix and count
    nvflare token batch \
        --prefix site- \
        --count 100 \
        --pad 3 \
        --cert-service https://cert-service:8443 \
        --api-key $NVFLARE_API_KEY \
        -o ./tokens/

*Output:*

.. code-block:: text

    ./tokens/
    ├── site-001.token
    ├── site-002.token
    ├── ...
    └── site-100.token

**nvflare token info**

Inspect and decode a token without verification.

.. code-block:: shell

    nvflare token info -t eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...

*Output:*

.. code-block:: text

    Token Information:
      Subject:      hospital-1
      Subject Type: client
      Issuer:       My Project
      Issued At:    2025-01-04 10:00:00 UTC
      Expires At:   2025-01-05 10:00:00 UTC
      Token ID:     abc123-def456-789

nvflare package
===============

Generate startup kit without certificates (for token-based enrollment).

**Location:** ``nvflare/lighter/startup_kit.py``

**Modes:**

1. **Single participant mode**: Generate one package from CLI arguments
2. **Project file mode**: Generate all packages from project.yml (without certs)

**Single Participant Mode**

.. code-block:: shell

    nvflare package \
        -n hospital-1 \                 # Participant name (required)
        -e grpc://server:8002 \         # Server endpoint URI (required)
        -t client \                     # Type: client|relay|server|user (required)
        -o ./packages \                 # Output directory (optional)
        --org "Hospital A" \            # Organization (optional)
        --cert-service URL \            # Certificate Service URL (optional)
        --token TOKEN                   # Enrollment token (optional)

**Enrollment Options (for Auto-Scale workflow):**

When ``--cert-service`` and ``--token`` are provided, the package includes:

- ``startup/enrollment.json`` with the Certificate Service URL
- ``startup/enrollment.token`` with the enrollment token

This consolidates the package generation and enrollment setup into one command.

*Endpoint URI Formats:*

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Format
     - Description
   * - ``grpc://host:port``
     - gRPC with single port (admin port = fl_port + 1)
   * - ``grpc://host:fl_port:admin_port``
     - gRPC with explicit ports
   * - ``http://host:port``
     - HTTP/HTTPS protocol
   * - ``tcp://host:port``
     - TCP protocol

*Examples:*

.. code-block:: shell

    # Client package
    nvflare package \
        -n hospital-1 \
        -e grpc://server.example.com:8002 \
        -t client \
        -o ./packages
    
    # Server package (two-port format)
    nvflare package \
        -n server1 \
        -e grpc://0.0.0.0:8002:8003 \
        -t server \
        -o ./packages
    
    # Relay package
    nvflare package \
        -n relay-east \
        -e grpc://server:8002 \
        -t relay \
        --listening-host 0.0.0.0 \
        --listening-port 8102 \
        -o ./packages
    
    # User/Admin package
    nvflare package \
        -n admin@org.com \
        -e grpc://server:8003 \
        -t user \
        -o ./packages
    
    # Client package with enrollment (Auto-Scale workflow)
    nvflare package \
        -n hospital-1 \
        -e grpc://server:8002 \
        -t client \
        --cert-service https://cert-service.example.com:8443 \
        --token eyJhbGciOiJSUzI1NiIs... \
        -o ./packages

*Output (without enrollment options):*

.. code-block:: text

    ./packages/hospital-1/
    ├── local/
    │   ├── authorization.json.default
    │   └── ...
    ├── startup/
    │   ├── fed_client.json
    │   ├── start.sh
    │   └── ...
    └── transfer/

*Output (with --cert-service and --token):*

.. code-block:: text

    ./packages/hospital-1/
    ├── local/
    │   ├── authorization.json.default
    │   ├── log_config.json.default
    │   ├── privacy.json.sample
    │   └── resources.json.default
    ├── startup/
    │   ├── fed_client.json
    │   ├── enrollment.json         # ← Certificate Service URL
    │   ├── enrollment.token        # ← Enrollment token
    │   ├── start.sh
    │   ├── stop_fl.sh
    │   └── sub_start.sh
    └── transfer/

**Project File Mode**

Generate packages for all participants defined in a project.yml:

.. code-block:: shell

    nvflare package -p project.yml -o ./packages

This filters out ``CertBuilder`` and ``SignatureBuilder`` from the project,
generating packages without certificates (ready for token-based enrollment).

nvflare enrollment
==================

Manage pending enrollment requests (Admin CLI for Certificate Service).

**Location:** ``nvflare/tool/enrollment/enrollment_cli.py``

**Subcommands:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Subcommand
     - Description
   * - ``list``
     - List pending enrollment requests
   * - ``info``
     - View details of a pending request
   * - ``approve``
     - Approve a pending request
   * - ``reject``
     - Reject a pending request
   * - ``enrolled``
     - List enrolled entities

**Environment Variables:**

- ``NVFLARE_CERT_SERVICE_URL``: Certificate Service URL
- ``NVFLARE_API_KEY``: Admin authentication token

**nvflare enrollment list**

List pending enrollment requests.

.. code-block:: shell

    nvflare enrollment list                    # All pending
    nvflare enrollment list --type client      # Sites only
    nvflare enrollment list --type user        # Users only

*Output:*

.. code-block:: text

    Name           Type      Org          Submitted             Status
    ──────────────────────────────────────────────────────────────────
    hospital-1     client    Hospital A   2025-01-04 10:15:00   pending
    hospital-2     client    Hospital B   2025-01-04 10:20:00   pending
    admin@org.com  user      Org Inc      2025-01-04 11:00:00   pending

**nvflare enrollment info**

View details of a specific pending request.

.. code-block:: shell

    nvflare enrollment info hospital-1 --type client

*Output:*

.. code-block:: text

    Name:            hospital-1
    Type:            client
    Organization:    Hospital A
    Submitted:       2025-01-04 10:15:00 UTC
    Expires:         2025-01-11 10:15:00 UTC
    Token Subject:   hospital-*
    Source IP:       10.2.3.4
    CSR Subject:     CN=hospital-1, O=Hospital A

**nvflare enrollment approve**

Approve pending enrollment requests.

.. code-block:: shell

    # Approve single request
    nvflare enrollment approve hospital-1 --type client
    
    # Approve user request
    nvflare enrollment approve admin@org.com --type user
    
    # Bulk approve by pattern
    nvflare enrollment approve --pattern "hospital-*" --type client

**nvflare enrollment reject**

Reject pending enrollment requests.

.. code-block:: shell

    nvflare enrollment reject hospital-2 --type client \
        --reason "Site not authorized for this project"

**nvflare enrollment enrolled**

List enrolled entities.

.. code-block:: shell

    nvflare enrollment enrolled                # All enrolled
    nvflare enrollment enrolled --type client  # Sites only
    nvflare enrollment enrolled --type user    # Users only

*Output:*

.. code-block:: text

    Name           Type      Org          Enrolled At
    ────────────────────────────────────────────────────
    hospital-1     client    Hospital A   2025-01-04 12:00:00
    hospital-3     client    Hospital C   2025-01-04 12:30:00
    admin@org.com  user      Org Inc      2025-01-04 13:00:00

CLI Summary
===========

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Command
     - Workflow
     - Purpose
   * - ``nvflare cert``
     - Manual
     - Generate root CA and signed certificates
   * - ``nvflare token``
     - Auto-Scale
     - Generate enrollment tokens
   * - ``nvflare package``
     - Both
     - Generate startup kits (without certificates)
   * - ``nvflare enrollment``
     - Auto-Scale
     - Manage pending requests (admin)

***********************
Workflows
***********************

Workflow 1: Manual (Small Scale)
================================

For 5-10 participants. No Certificate Service needed.

.. code-block:: text

    Step 1: Project Admin            Step 2: Distribute       Step 3: Site Operator
    ────────────────────            ────────────────         ─────────────────────
    
    nvflare cert init               Email/USB/               pip install nvflare
        │                           Secure Channel
        ▼                                │                   nvflare package \
    rootCA.pem + rootCA.key              │                      -n hospital-1 \
        │                                │                      -e grpc://server:8002 \
        ▼                                │                      -t client
    nvflare cert server                  │                          │
        │                                │                          ▼
        ▼                                │                   hospital-1/startup/
    server.crt ─────────────────────────►│                      ├── fed_client.json
    server.key                           │                      └── (no certs yet)
        │                                │                          │
        ▼                                │                   Copy received certs:
    nvflare cert client                  │                      ├── client.crt
        │                                │                      ├── client.key
        ▼                                │                      └── rootCA.pem
    hospital-1.crt ─────────────────────►│                          │
    hospital-1.key                       ▼                          ▼
                                                              cd hospital-1 && \
                                                                ./startup/start.sh

Workflow 2: Auto-Scale (Large Scale)
====================================

For 10+ participants or dynamic environments.

.. code-block:: text

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
    
    nvflare token generate -n hospital-1 \
        --cert-service https://cert-service.example.com \
        --api-key $API_KEY
    
    # Or batch generate:
    nvflare token batch \
        --pattern "hospital-{001..100}" \
        --cert-service https://cert-service.example.com \
        --api-key $API_KEY
    
    Distribute to each site:
       - Token string
       - Certificate Service URL
    
    Phase 3: Site Enrollment (Site Operator)
    ────────────────────────────────────────
    
    Option A: Consolidated command (recommended)
    
        nvflare package \
            -n hospital-1 \
            -e grpc://server:8002 \
            -t client \
            --cert-service https://cert-service.example.com \
            --token "eyJhbGciOiJSUzI1NiIs..."
        
        cd hospital-1 && ./startup/start.sh
    
    Option B: Environment variables (for K8s/containers)
    
        nvflare package -n hospital-1 -e grpc://server:8002 -t client
        
        export NVFLARE_ENROLLMENT_TOKEN="eyJ..."
        export NVFLARE_CERT_SERVICE_URL="https://cert-service.example.com"
        
        cd hospital-1 && ./startup/start.sh
    
    (Auto-enrollment happens at startup)

***********************
Implementation Details
***********************

This section provides an overview of all implementation components. Detailed
API documentation and code examples are in the sections referenced below.

Component Overview
==================

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Component
     - Location
     - Purpose
   * - **Certificate Service**
     - ``nvflare/cert_service/``
     - HTTP service for token generation and CSR signing
   * - CertService
     - ``cert_service.py``
     - Core logic: token validation, policy evaluation, CSR signing
   * - CertServiceApp
     - ``app.py``
     - HTTP/Flask wrapper for CertService
   * - EnrollmentStore
     - ``store.py``
     - Tracks enrolled entities and pending requests (SQLite/PostgreSQL)
   * - **Client Enrollment**
     - ``nvflare/private/fed/client/enrollment/``
     - Client-side enrollment components
   * - CertRequestor
     - ``cert_requestor.py``
     - Submits CSR to Certificate Service via HTTP
   * - EnrollmentIdentity
     - ``cert_requestor.py``
     - Client/server/relay/user identity for enrollment
   * - EnrollmentOptions
     - ``cert_requestor.py``
     - Configuration options (timeout, output_dir)
   * - **CLI Tools**
     - ``nvflare/tool/enrollment/``
     - Command-line interfaces
   * - token_cli
     - ``token_cli.py``
     - ``nvflare token`` command
   * - cert_cli
     - ``cert_cli.py``
     - ``nvflare cert`` command
   * - enrollment_cli
     - ``enrollment_cli.py``
     - ``nvflare enrollment`` command
   * - TokenService
     - ``token_service.py``
     - JWT token generation logic
   * - **Package Generator**
     - ``nvflare/lighter/``
     - Startup kit generation
   * - startup_kit
     - ``startup_kit.py``
     - ``nvflare package`` command

Certificate Service Components
==============================

See `Certificate Service`_ section for:

- HTTP API endpoints
- Configuration file format
- EnrollmentStore interface
- Deployment options

Client Enrollment Components
============================

See `Client Enrollment`_ and `Server Enrollment`_ sections for:

- CertRequestor usage
- EnrollmentOptions configuration
- EnrollmentResult dataclass
- Auto-enrollment flow

CLI Components
==============

See `CLI Commands`_ section for:

- ``nvflare cert`` - Certificate generation (Manual Workflow)
- ``nvflare token`` - Token generation via Certificate Service API
- ``nvflare package`` - Startup kit generation
- ``nvflare enrollment`` - Pending request management

FLARE Integration
=================

**Client Auto-Enrollment (Auto-Scale Workflow)**

When a client or server starts without certificates, it performs auto-enrollment:

.. code-block:: python

    # In FederatedClientBase or server startup
    
    def _perform_auto_enrollment(self, client_args: dict, startup_dir: str):
        """Perform automatic enrollment if certificates are missing."""
        
        # 1. Check for existing certificates
        cert_path = os.path.join(startup_dir, "client.crt")
        if os.path.exists(cert_path):
            return  # Already enrolled
        
        # 2. Get Certificate Service URL and token
        cert_service_url = os.getenv("NVFLARE_CERT_SERVICE_URL")
        token = os.getenv("NVFLARE_ENROLLMENT_TOKEN")
        
        if not cert_service_url or not token:
            # Check files
            ...
        
        # 3. Perform enrollment
        identity = EnrollmentIdentity.for_client(
            name=client_args.get("client_name"),
            org=client_args.get("org"),
        )
        
        requestor = CertRequestor(
            cert_service_url=cert_service_url,
            enrollment_token=token,
            identity=identity,
            options=EnrollmentOptions.from_client_args(client_args, startup_dir),
        )
        
        result = requestor.request_certificate()
        
        # 4. Use in-memory certificates for this session
        # Files are saved for restart persistence

**No Changes to Existing FLARE Code**

The token-based enrollment is **additive** and does not modify:

- Existing authentication mechanisms
- CellNet communication
- FL training workflows
- Job execution

It simply provides an alternative way to obtain certificates before normal
FLARE operations begin.

Key Design Notes
================

1. **HTTP-based, not CellNet**: The Certificate Service uses standard HTTP/REST,
   not CellNet. This keeps certificate management separate from FL operations.

2. **No authentication bypass**: Sites must obtain valid certificates before
   connecting to the FL Server. There is no unauthenticated path.

3. **Token is not for authentication**: The enrollment token grants eligibility
   to request a certificate. The certificate is then used for mTLS authentication.

4. **Stateless tokens**: Tokens are JWTs with embedded policies. No token state
   is tracked; only enrolled sites and pending requests are tracked.

***********************
Backward Compatibility
***********************

The token-based enrollment system is fully backward compatible:

1. **Existing provisioned deployments**: Continue to work unchanged
2. **No migration required**: Can adopt gradually
3. **Coexistence**: Some sites provisioned, some enrolled

***********************
Future Enhancements
***********************

1. **Certificate rotation**: Automatic renewal before expiry
2. **Revocation**: CRL or OCSP support
3. **HSM integration**: Hardware security module for root CA
4. **Dashboard integration**: Web UI for token and enrollment management
5. **Notification webhooks**: Alert admin when requests are pending

***********************
Appendix
***********************

Glossary
========

- **Root CA**: Root Certificate Authority, the trust anchor
- **CSR**: Certificate Signing Request
- **JWT**: JSON Web Token
- **mTLS**: Mutual TLS (client and server both present certificates)
- **RS256**: RSA signature with SHA-256

Related Documents
=================

- :ref:`token_command` - Token CLI documentation
- :ref:`package_command` - Package CLI documentation
- :ref:`provisioning` - Standard provisioning documentation
