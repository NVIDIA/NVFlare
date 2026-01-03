.. _enrollment_design_v2:

#################################################
FLARE Enrollment System Design (v2)
#################################################

*Version: 2.0*
*Status: Draft*
*Author: NVIDIA FLARE Team*

***********************
Executive Summary
***********************

This document describes a redesigned enrollment system for NVIDIA FLARE that
prioritizes security, simplicity, and scalability. The design splits into two
distinct workflows based on deployment scale:

1. **Manual Workflow**: For small deployments (5-10 sites), Project Admin
   manually distributes signed certificates via secure channels (email, etc.).

2. **Auto-Scale Workflow**: For large deployments (>10 sites) or dynamic
   environments, a dedicated Certificate Service handles token-based enrollment.


***********************
Goals
***********************

1. **5-minute setup** for small deployments
2. **Secure by design** - minimize attack surface, follow PKI best practices
3. **Scalable** - support hundreds of dynamic participants
4. **Backward compatible** - existing provisioned deployments continue to work
5. **Clear separation of concerns** - FL Server vs Certificate Authority

***********************
Non-Goals
***********************

1. Replacing the existing provisioning system (remains for full-control scenarios)
2. Certificate revocation (out of scope)

***********************
Architecture Overview
***********************

Key Design Principles
=====================

1. **Root CA private key (rootCA.key) never leaves Project Admin infrastructure**
2. **Certificate Service is the single trust anchor** for auto-scale workflow
3. **Eliminate startup kit provision and distribution** to simplify the workflow

Workflow Comparison
===================

.. list-table::
   :widths: 18 27 27 28
   :header-rows: 1

   * - Aspect
     - Current (Provision)
     - Manual (CLI)
     - Auto-Scale
   * - Target
     - Any size
     - 5+ sites
     - 10+ sites or dynamic
   * - Infrastructure
     - None
     - None
     - Certificate Service
   * - Project Admin tools
     - ``nvflare provision``
     - ``nvflare cert``, ``nvflare package``
     - Certificate Service
   * - What's distributed
     - Complete startup kits
     - Signed certs only
     - Token + Cert Service URL
   * - Startup kit generation
     - By Project Admin
     - By each site locally
     - By each site locally
   * - Private keys generated
     - At Project Admin site
     - At Project Admin site
     - At each site locally ✓
   * - Private keys in transit
     - Yes (in startup kit)
     - Yes (via email/USB)
     - **Never** ✓
   * - rootCA.pem
     - In startup kit
     - Distributed with certs
     - Downloaded from Cert Service
   * - rootCA.key
     - At Project Admin
     - At Project Admin
     - At Cert Service
   * - Adding new sites
     - Provision + distribute startup kit
     - Create cert + deliver manually
     - Generate token only ✓
   * - Setup complexity
     - Low
     - Low
     - Medium (Cert Service) 

What Gets Distributed
=====================

.. list-table::
   :widths: 15 30 30 25
   :header-rows: 1

   * - Workflow
     - What Project Admin Distributes
     - Private Keys Transit
     - rootCA.key Location
   * - **Current (Provision)**
     - Complete startup kits (certs + configs)
     - Yes (in kit)
     - Project Admin
   * - **Manual (CLI)**
     - Signed certs + rootCA.pem only
     - Yes (email/USB)
     - Project Admin
   * - **Auto-Scale**
     - Token + Cert Service URL only
     - **Never** ✓
     - Cert Service

**Key points**:

- **rootCA.pem** (public certificate): Distributed to all sites for verification
- **rootCA.key** (private key): NEVER leaves Project Admin / Cert Service
- **Auto-Scale advantage**: Private keys generated locally, never transmitted

Distribution per site:

.. list-table::
   :widths: 12 30 30 28
   :header-rows: 1

   * - Site Type
     - Current (Provision)
     - Manual (CLI)
     - Auto-Scale
   * - **Server**
     - Full startup kit
     - server.crt, server.key, rootCA.pem
     - Same as Manual (or via Cert Service)
   * - **Client**
     - Full startup kit
     - client.crt, client.key, rootCA.pem
     - Token → enrolls → client.crt, client.key, rootCA.pem

***********************
Workflow 1: Manual
***********************

**CLI Only - No Certificate Service Required**

Target Audience
===============

- Teams with 5+ sites
- Proof-of-concept deployments
- Air-gapped environments
- Maximum control over certificate distribution
- **No additional infrastructure** - just CLI tools

Process Overview
================

.. code-block:: text

    Step 1: Project Admin                    Step 2: Distribute         Step 3: Site Operator
    ─────────────────────                    ──────────────────         ─────────────────────
    
    nvflare cert init                        Email/USB/Secure Channel   pip install nvflare
        │                                           │                          │
        ▼                                           │                          ▼
    rootCA.pem + rootCA.key                         │                   nvflare package
        │                                           │                      -n site-1
        ▼                                           │                      -e grpc://server:8002
    nvflare cert server -n server1                  │                          │
        │                                           │                          ▼
        ▼                                           │                   site-1/startup/
    server.crt + server.key ───────────────────────►│                      ├── fed_client.json
        │                                           │                      └── (no certs yet)
        ▼                                           │                          │
    nvflare cert client -n site-1                   │                          │
        │                                           │                          ▼
        ▼                                           │                   Copy received certs:
    site-1.crt + site-1.key ───────────────────────►│                      ├── client.crt
                                                    │                      ├── client.key
                                                    ▼                      └── rootCA.pem
                                                                               │
                                                                               ▼
                                                                          Start client

CLI Commands (Manual Workflow)
==============================

.. code-block:: shell

    # Project Admin: Initialize root CA
    nvflare cert init -n "My Project" -o ./project_ca
    
    # Project Admin: Generate server certificate
    nvflare cert server -n server1 -c ./project_ca -o ./server_certs \
        --host server.example.com
    
    # Project Admin: Generate client certificate (NEW)
    nvflare cert client -n hospital-1 -c ./project_ca -o ./hospital1_certs \
        --org "Hospital A"
    
    # Distribute via secure channel (email, USB, etc.):
    #   To server: server_certs/server.crt, server.key, rootCA.pem
    #   To client: hospital1_certs/hospital-1.crt, hospital-1.key, rootCA.pem
    
    # Site Operator: Generate startup kit
    nvflare package -n hospital-1 -e grpc://server.example.com:8002
    
    # Site Operator: Copy received certificates
    cp hospital-1.crt hospital-1/startup/client.crt
    cp hospital-1.key hospital-1/startup/client.key
    cp rootCA.pem hospital-1/startup/
    
    # Site Operator: Start
    cd hospital-1 && ./startup/start.sh

Security Properties (Manual)
============================

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Property
     - Status
   * - rootCA.key (private)
     - Project Admin only - NEVER distributed ✓
   * - rootCA.pem (public)
     - Distributed to all sites ✓
   * - Site private keys in transit
     - Yes (via secure channel)
   * - Who can issue certs
     - Project Admin only ✓
   * - Blast radius if site compromised
     - Individual site only ✓

***********************
Workflow 2: Auto-Scale
***********************

**Requires Certificate Service Infrastructure**

Target Audience
===============

- Large deployments (10+ sites)
- Dynamic environments (K8s, cloud auto-scaling)
- Sites that join/leave frequently
- Enterprise deployments requiring audit trails
- Organizations willing to deploy Certificate Service

Architecture
============

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                  PROJECT ADMIN INFRASTRUCTURE                            │
    │                                                                          │
    │   ┌─────────────────────────────────────────────────────────────────┐   │
    │   │                    Certificate Service                           │   │
    │   │                                                                  │   │
    │   │  TLS: Public CA cert (Let's Encrypt) ← Sites verify via std PKI │   │
    │   │  Signing: FLARE rootCA.key ← Never leaves this service          │   │
    │   │                                                                  │   │
    │   │  ┌─────────────────────────────────────────────────────────┐    │   │
    │   │  │ GET  /api/v1/ca-cert   → Returns rootCA.pem             │    │   │
    │   │  │ POST /api/v1/enroll    → Validates token, signs CSR     │    │   │
    │   │  └─────────────────────────────────────────────────────────┘    │   │
    │   │                                                                  │   │
    │   └─────────────────────────────────────────────────────────────────┘   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
                                       │
                     HTTPS (verified by public CA - no rootCA.pem needed!)
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         │                             │                             │
         ▼                             ▼                             ▼
    ┌─────────────┐           ┌─────────────────┐           ┌─────────────┐
    │  FL Server  │           │  Client Site    │           │  Client Site│
    │             │           │                 │           │             │
    │ Gets from   │           │ 1. Connect to   │           │ Same flow   │
    │ Cert Svc:   │           │    Cert Service │           │             │
    │ • rootCA.pem│◄─ mTLS ──►│ 2. Download     │           │             │
    │ • server.crt│           │    rootCA.pem   │           │             │
    │             │           │ 3. Send CSR     │           │             │
    │ NO root key!│           │ 4. Get cert     │           │             │
    └─────────────┘           └─────────────────┘           └─────────────┘

**Key insight**: Sites verify Certificate Service using standard public PKI
(Let's Encrypt, etc.), then download FLARE's rootCA.pem from the service.
No manual rootCA.pem distribution needed!

Certificate Service
===================

A standalone HTTP service that handles enrollment. **Separate from FL Server**.

**Two Certificates:**

1. **Public TLS cert** (Let's Encrypt, DigiCert, etc.)
   - Used for HTTPS
   - Sites can verify without any prior trust setup
   - Solves the bootstrap problem

2. **FLARE Root CA** (rootCA.pem + rootCA.key)
   - Used for signing FL participant certificates
   - Private key never leaves Certificate Service
   - Public cert served via API endpoint

**Key Responsibilities:**

1. Serve rootCA.pem (GET /api/v1/ca-cert)
2. Token validation (JWT verification)
3. Policy evaluation (site name patterns, approval rules)
4. CSR signing (holds rootCA.key)
5. Audit logging

**API Endpoints:**

.. code-block:: text

    GET /api/v1/ca-cert
    └── Response:
        └── 200: PEM-encoded rootCA.pem
    
    POST /api/v1/enroll
    ├── Request:
    │   ├── token: JWT enrollment token
    │   ├── csr: PEM-encoded CSR
    │   └── metadata: { site_name, org, ... }
    │
    └── Response:
        ├── 200: { certificate: PEM-encoded signed cert, ca_cert: rootCA.pem }
        ├── 401: Invalid/expired token
        ├── 403: Policy rejection
        └── 500: Server error

**Deployment Options:**

1. **Standalone container** (Docker/K8s)
2. **FLARE Dashboard integration** (existing web service)
3. **Cloud-managed** (AWS Certificate Manager, HashiCorp Vault)

Process Overview (Auto-Scale)
=============================

.. code-block:: text

    Phase 1: Setup (Project Admin)
    ──────────────────────────────
    
    1. Initialize root CA
       nvflare cert init -n "Project" -o ./ca
    
    2. Deploy Certificate Service (with public TLS cert)
       - Obtain TLS cert from public CA (Let's Encrypt, etc.)
       - Configure with rootCA.pem + rootCA.key
       - Configure approval policy
       - Start on https://cert-service.mycompany.com
    
    3. Enroll FL Server via Certificate Service
       - Server runs: nvflare package -n server1 -e grpc://localhost:8002 -t server
       - Server gets server.crt via enrollment (or manual cert generation)
       - Server downloads rootCA.pem from Certificate Service
    
    Phase 2: Generate Tokens (Project Admin)
    ────────────────────────────────────────
    
    nvflare token generate -s hospital-1 -c ./ca -o hospital1.token
    nvflare token generate -s hospital-2 -c ./ca -o hospital2.token
    ...
    
    Distribute to sites (minimal!):
    ┌────────────────────────────────────────────────────────┐
    │  ONLY TWO THINGS:                                      │
    │  1. Token (unique per site)                            │
    │  2. Certificate Service URL                            │
    │                                                        │
    │  NO rootCA.pem distribution - downloaded automatically │
    └────────────────────────────────────────────────────────┘
    
    Phase 3: Site Enrollment (Site Operator)
    ────────────────────────────────────────
    
    1. Generate startup kit
       nvflare package -n hospital-1 \
           -e grpc://fl-server.mycompany.com:8002 \
           --cert-service https://cert-service.mycompany.com
    
    2. Configure enrollment token
       export NVFLARE_ENROLLMENT_TOKEN="<token_from_admin>"
    
    3. Start client (auto-enrollment happens automatically)
       cd hospital-1 && ./startup/start.sh
       
       Startup flow:
       ┌─────────────────────────────────────────────────────┐
       │ 1. Connect to Certificate Service                   │
       │    (verified by public CA - Let's Encrypt, etc.)    │
       │                                                     │
       │ 2. Download rootCA.pem                              │
       │    GET https://cert-service/api/v1/ca-cert          │
       │                                                     │
       │ 3. Generate key pair locally                        │
       │    (private key never leaves this machine)          │
       │                                                     │
       │ 4. Send CSR + token to Certificate Service          │
       │    POST https://cert-service/api/v1/enroll          │
       │                                                     │
       │ 5. Receive signed certificate                       │
       │                                                     │
       │ 6. Connect to FL Server (verified by rootCA.pem)    │
       │    Ready for federated learning!                    │
       └─────────────────────────────────────────────────────┘

Security Properties (Auto-Scale)
================================

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Property
     - Status
   * - rootCA.key (private)
     - Certificate Service only - NEVER distributed ✓
   * - rootCA.pem (public)
     - Downloaded from Cert Service during enrollment ✓
   * - Site private keys in transit
     - Never - generated locally ✓
   * - Who can issue certs
     - Certificate Service only ✓
   * - Bootstrap trust
     - Public CA (Let's Encrypt) for Cert Service TLS ✓
   * - Separation of concerns
     - FL workloads vs CA operations ✓
   * - Audit trail
     - All enrollments logged ✓
   * - Blast radius if FL Server compromised
     - Cannot issue new certs ✓

***********************
Component Design
***********************

CLI Commands
============

nvflare cert
------------

.. code-block:: text

    nvflare cert
    ├── init          # Initialize root CA
    │   ├── -n/--name          Project name
    │   ├── -o/--output        Output directory
    │   └── --valid_days       Validity (default: 3650)
    │
    ├── server        # Generate server certificate
    │   ├── -n/--name          Server name (required)
    │   ├── -c/--ca_path       Path to CA (required)
    │   ├── -o/--output        Output directory
    │   ├── --host             Default hostname
    │   ├── --additional_hosts Additional SANs
    │   └── --valid_days       Validity (default: 365)
    │
    └── client        # Generate client certificate (NEW - for manual workflow)
        ├── -n/--name          Client/site name (required)
        ├── -c/--ca_path       Path to CA (required)
        ├── -o/--output        Output directory
        ├── --org              Organization
        └── --valid_days       Validity (default: 365)

nvflare package
---------------

.. code-block:: text

    nvflare package
    ├── -p/--project_file   Project YAML (packages all participants)
    ├── -w/--workspace      Output workspace
    │
    │   # Single participant mode:
    ├── -n/--name           Participant name
    ├── -e/--endpoint       Connection URI
    ├── -t/--type           server, client, relay, admin
    ├── --org               Organization
    ├── --role              Role (for admin)
    └── --cert_service_url  Certificate Service URL (for auto-enrollment)

nvflare token
-------------

(Unchanged from current design)

.. code-block:: text

    nvflare token
    ├── generate      # Generate single token
    ├── batch         # Generate multiple tokens
    └── info          # Display token information

Certificate Service
===================

**Location:** ``nvflare/app_opt/cert_service/`` (optional component)

**Components:**

.. code-block:: text

    cert_service/
    ├── __init__.py
    ├── app.py              # Flask/FastAPI application
    ├── token_validator.py  # JWT validation
    ├── policy_engine.py    # Approval policy evaluation
    ├── csr_signer.py       # Certificate signing
    ├── audit_logger.py     # Audit logging
    └── config.py           # Configuration

**Configuration:**

.. code-block:: yaml

    # cert_service_config.yaml
    
    server:
      host: 0.0.0.0
      port: 8443
      tls:
        cert: /path/to/service.crt
        key: /path/to/service.key
    
    ca:
      cert: /path/to/rootCA.pem
      key: /path/to/rootCA.key  # ONLY here, not on FL Server
    
    policy:
      file: /path/to/approval_policy.yaml
    
    audit:
      enabled: true
      log_file: /var/log/cert_service/audit.log

Client Enrollment Flow
======================

**Location:** ``nvflare/private/fed/client/enrollment/``

.. code-block:: python

    class AutoEnrollment:
        """Handles automatic certificate enrollment at startup."""
        
        def __init__(
            self,
            cert_service_url: str,
            token: str,
            identity: EnrollmentIdentity,
            ca_cert_path: str,
        ):
            self.cert_service_url = cert_service_url
            self.token = token
            self.identity = identity
            self.ca_cert_path = ca_cert_path
        
        def enroll(self, output_dir: str) -> str:
            """Perform enrollment.
            
            1. Generate key pair locally
            2. Create CSR
            3. Send CSR + token to Certificate Service (HTTPS)
            4. Receive signed certificate
            5. Save certificate and key
            
            Returns:
                Path to the signed certificate
            """
            # Generate key pair (never leaves this machine)
            pri_key, pub_key = generate_keys()
            
            # Create CSR
            csr = generate_csr(
                subject=Identity(self.identity.name, self.identity.org),
                pri_key=pri_key,
            )
            
            # Send to Certificate Service via HTTPS
            response = requests.post(
                f"{self.cert_service_url}/api/v1/enroll",
                json={
                    "token": self.token,
                    "csr": serialize_csr(csr),
                    "metadata": {
                        "name": self.identity.name,
                        "org": self.identity.org,
                        "type": self.identity.participant_type,
                    }
                },
                verify=self.ca_cert_path,  # Verify service cert
            )
            
            if response.status_code != 200:
                raise EnrollmentError(response.json().get("error"))
            
            # Save certificate and key
            cert_pem = response.json()["certificate"]
            cert_path = os.path.join(output_dir, "client.crt")
            key_path = os.path.join(output_dir, "client.key")
            
            with open(cert_path, "w") as f:
                f.write(cert_pem)
            with open(key_path, "wb") as f:
                f.write(serialize_pri_key(pri_key))
            
            return cert_path

***********************
Migration Path
***********************

From Existing Provisioning
==========================

Existing deployments using ``nvflare provision`` continue to work unchanged.

.. code-block:: text

    Existing:
    ├── nvflare provision → Full startup kits with certs
    └── No changes required
    
    New Options:
    ├── Manual workflow → For small, controlled deployments
    └── Auto-scale → For large, dynamic deployments

Choosing the Right Workflow
===========================

.. list-table::
   :widths: 18 27 27 28
   :header-rows: 1

   * - Criteria
     - Current (Provision)
     - Manual (CLI)
     - Auto-Scale
   * - Number of sites
     - Any
     - 5+ (static)
     - 10+ or dynamic
   * - Site dynamics
     - Static (need to distribute startup kit)
     - Static (need to deliver certs manually)
     - Dynamic (self-service enrollment) ✓
   * - Infrastructure
     - None
     - None
     - Certificate Service
   * - Startup kit creation
     - Centralized
     - Distributed
     - Distributed
   * - Private key security
     - Keys in transit
     - Keys in transit
     - **Keys never transit** ✓
   * - Automation
     - Low
     - Low
     - High ✓
   * - Audit trail
     - Manual
     - Manual
     - Automated ✓

**When to use each:**

- **Current (Provision)**: Full control, all participants known upfront, comfortable with existing workflow
- **Manual (CLI)**: Want to eliminate startup kit distribution, sites generate own packages
- **Auto-Scale**: Large deployments, dynamic enrollment, need audit trails, want maximum security (keys never transit)

***********************
Security Considerations
***********************

Root CA Protection
==================

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Location
     - Manual Workflow
     - Auto-Scale Workflow
   * - Project Admin workstation
     - Yes
     - Yes (for init only)
   * - Certificate Service
     - N/A
     - Yes (required)
   * - FL Server
     - **Never**
     - **Never**
   * - Client sites
     - **Never**
     - **Never**

Certificate Service Hardening
=============================

1. **Network isolation**: Only expose enrollment endpoint
2. **Rate limiting**: Prevent token brute-force
3. **TLS required**: All communication encrypted
4. **Audit logging**: Log all enrollment attempts
5. **Key protection**: Consider HSM for rootCA.key

Token Security
==============

1. **Single-use**: Each token can only be used once
2. **Short-lived**: Tokens expire (configurable)
3. **Signed**: JWT with RS256, tamper-proof
4. **Scoped**: Token specifies allowed site name/pattern

***********************
Open Questions
***********************

1. **Certificate Service deployment model**: 
   Standalone vs. Dashboard integration vs. Cloud-managed?

2. **Token delivery mechanism**:
   Out-of-band (email) vs. In-band (secure API)?

3. **Certificate rotation**:
   Automatic renewal or manual re-enrollment?

4. **Revocation**:
   CRL, OCSP, or connection-time checks?

***********************
Appendix
***********************

Glossary
========

- **Root CA**: Root Certificate Authority, the trust anchor
- **CSR**: Certificate Signing Request
- **SAN**: Subject Alternative Name (for multiple hostnames)
- **JWT**: JSON Web Token (used for enrollment tokens)

Related Documents
=================

- :ref:`token_command` - Token CLI documentation
- :ref:`cert_command` - Cert CLI documentation
- :ref:`package_command` - Package CLI documentation
- :ref:`provisioning` - Standard provisioning documentation

