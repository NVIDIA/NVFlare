.. _enrollment_token_design:

#########################################
Token-Based Enrollment Design
#########################################

This document describes the design of NVIDIA FLARE's token-based enrollment system,
which enables dynamic client enrollment without pre-provisioned certificates.

.. contents:: Table of Contents
   :local:
   :depth: 2

***********************
Overview
***********************

Traditional FLARE deployments require each client to receive a pre-provisioned startup kit
containing client-specific certificates. This approach has limitations:

- **Scalability**: Generating and distributing individual startup kits is time-consuming
- **Flexibility**: Adding new clients requires re-provisioning
- **Operational Overhead**: Managing certificate distribution is complex

Token-based enrollment addresses these challenges by allowing clients to dynamically
obtain certificates using enrollment tokens (JWTs).

Architecture Overview
=====================

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                           ADMIN SIDE                                     │
    │  ┌─────────────────┐                                                    │
    │  │  TokenService   │ ─── generates ──► Enrollment Tokens (JWTs)         │
    │  │  (CLI Tool)     │                   with embedded policy              │
    │  └─────────────────┘                                                    │
    └─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ Token distributed
                                        │ out-of-band
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                           CLIENT SIDE                                    │
    │  ┌─────────────────┐      CSR + Token      ┌─────────────────────────┐  │
    │  │ CertRequestor   │ ────────────────────► │    FL Server            │  │
    │  │ (Client)        │                       │                         │  │
    │  │                 │ ◄──────────────────── │  ┌─────────────────┐    │  │
    │  │                 │   Signed Certificate  │  │   CertService   │    │  │
    │  └─────────────────┘                       │  │   (Validator)   │    │  │
    │                                            │  └─────────────────┘    │  │
    │                                            └─────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────────┘

Key Components
==============

1. **TokenService** - Generates enrollment tokens with embedded policies
2. **CertRequestor** - Client-side component that requests certificates
3. **CertService** - Server-side component that validates tokens and signs CSRs
4. **Enrollment CLI** - Command-line interface for token management

***********************
Part 1: Core Components
***********************

TokenService Design
===================

Location: ``nvflare/tool/enrollment/token_service.py``

The ``TokenService`` is responsible for generating enrollment tokens (JWTs) that
contain embedded approval policies. Tokens are signed with the root CA's private key
to prevent tampering.

Key Features
------------

- **JWT-based tokens**: Uses RS256 signing algorithm for tamper-proof tokens
- **Embedded policy**: The approval policy is included in the token payload
- **Flexible subject types**: Supports client (FL sites), admin (FLARE Console users), relay (hierarchical FL nodes), and pattern subjects
- **Batch generation**: Can generate multiple tokens at once for scalability

Class Structure
---------------

.. code-block:: python

    class TokenService:
        """Service for generating and managing enrollment tokens."""
        
        JWT_ALGORITHM = "RS256"
        SUBJECT_TYPE_PATTERN = "pattern"
        
        def __init__(self, ca_path: str):
            """Initialize with path to CA certificates."""
            
        def generate_token(
            self,
            policy: Dict[str, Any],
            subject: str,
            subject_type: str = "client",
            validity: Optional[str] = None,
            **claims
        ) -> str:
            """Generate a single enrollment token."""
            
        def generate_token_from_file(
            self,
            policy_file: str,
            subject: str,
            **kwargs
        ) -> str:
            """Generate token using policy from YAML file."""
            
        def batch_generate_tokens(
            self,
            policy_file: str,
            count: int = 0,
            name_prefix: str = "client",
            names: Optional[List[str]] = None,
            **kwargs
        ) -> List[Dict[str, str]]:
            """Generate multiple tokens in batch."""
            
        def get_token_info(self, token: str) -> Dict[str, Any]:
            """Extract information from a token (without verification)."""

Token Payload Structure
-----------------------

.. code-block:: json

    {
        "jti": "unique-token-id",
        "sub": "hospital-1",
        "subject_type": "client",
        "iss": "nvflare-enrollment",
        "iat": 1704200000,
        "exp": 1704804800,
        "roles": ["lead"],
        "source_ips": ["10.0.0.0/8"],
        "policy": {
            "metadata": {...},
            "token": {...},
            "approval": {...}
        }
    }

.. note::

    All tokens are single-use. Once a token is used for enrollment, it cannot be reused.

CertRequestor Design
====================

Location: ``nvflare/private/fed/client/enrollment/cert_requestor.py``

The ``CertRequestor`` handles the client-side enrollment workflow, including
key pair generation, CSR creation, and certificate retrieval.

Key Features
------------

- **Automated CSR generation**: Creates Certificate Signing Requests with proper attributes
- **CellNet integration**: Uses FLARE's unified network layer for communication
- **Pydantic validation**: Input validation using Pydantic models
- **Factory methods**: Convenient constructors for different participant types

Class Structure
---------------

.. code-block:: python

    class EnrollmentIdentity(BaseModel):
        """Client identity for enrollment."""
        name: str
        participant_type: str = "client"
        org: Optional[str] = None
        role: Optional[str] = None
        
        @classmethod
        def for_client(cls, name: str, org: str = None) -> "EnrollmentIdentity":
            """Create identity for client enrollment."""
            
        @classmethod
        def for_admin(cls, name: str, role: str, org: str = None) -> "EnrollmentIdentity":
            """Create identity for admin enrollment."""
            
        @classmethod
        def for_relay(cls, name: str, org: str = None) -> "EnrollmentIdentity":
            """Create identity for relay enrollment."""


    class EnrollmentOptions(BaseModel):
        """Enrollment options."""
        output_dir: str = "."
        source_ip: Optional[str] = None


    class CertRequestor:
        """Client-side certificate requestor."""
        
        def __init__(
            self,
            cell: Cell,
            enrollment_token: str,
            identity: EnrollmentIdentity,
            options: EnrollmentOptions = None
        ):
            """Initialize the certificate requestor."""
            
        def create_csr(self) -> bytes:
            """Generate key pair and create CSR."""
            
        def submit_csr(self, csr_pem: bytes) -> Optional[bytes]:
            """Submit CSR to server via CellNet."""
            
        def save_credentials(self, cert_pem: bytes) -> str:
            """Save certificate and private key to disk."""
            
        def request_certificate(self) -> str:
            """Complete enrollment workflow and return certificate path."""

CSR Subject Attributes
----------------------

The CSR includes identity information in the subject:

- **Common Name (CN)**: Client/user name
- **Organization (O)**: Organization name (optional)
- **Organizational Unit (OU)**: Participant type (client, admin, relay)
- **Unstructured Name**: Role (for admin users)

CertService Design
==================

Location: ``nvflare/private/fed/server/enrollment/cert_service.py``

The ``CertService`` handles server-side token validation, policy evaluation,
and CSR signing.

Key Features
------------

- **Token validation**: Verifies JWT signature and expiration
- **Policy evaluation**: Evaluates embedded policy rules against enrollment context
- **CSR signing**: Signs valid CSRs using the root CA
- **CellNet handler**: Processes enrollment requests via CellNet

Class Structure
---------------

.. code-block:: python

    @dataclass
    class TokenPayload:
        """Decoded enrollment token payload."""
        token_id: str
        subject: str
        subject_type: str  # "client", "admin", "relay", or "pattern"
        policy: Dict[str, Any]
        expires_at: datetime
        roles: Optional[List[str]] = None
        source_ips: Optional[List[str]] = None


    @dataclass
    class EnrollmentContext:
        """Context for enrollment request."""
        name: str
        participant_type: str  # "client", "admin", or "relay"
        org: Optional[str] = None  # Organization name for certificate
        role: Optional[str] = None  # Role for admin tokens (e.g., "lead", "member")
        source_ip: Optional[str] = None


    class ApprovalAction(Enum):
        """Approval decision actions."""
        APPROVE = "approve"
        REJECT = "reject"
        PENDING = "pending"


    @dataclass
    class ApprovalResult:
        """Result of policy evaluation."""
        action: ApprovalAction
        rule_name: Optional[str] = None
        message: Optional[str] = None


    class CertService:
        """Server-side certificate service."""
        
        def __init__(
            self,
            root_ca_cert_path: str,
            root_ca_key_path: str
        ):
            """Initialize with root CA certificates."""
            
        def validate_token(self, token: str) -> Optional[TokenPayload]:
            """Validate JWT token and return payload."""
            
        def evaluate_policy(
            self,
            token: TokenPayload,
            context: EnrollmentContext
        ) -> ApprovalResult:
            """Evaluate policy rules against enrollment context."""
            
        def sign_csr(
            self,
            csr_pem: bytes,
            token: TokenPayload
        ) -> bytes:
            """Sign CSR and return certificate PEM."""
            
        def handle_csr_enrollment(self, request: Message) -> Message:
            """CellNet handler for CSR enrollment requests."""

Policy Evaluation Flow
----------------------

.. code-block:: text

    1. Validate token signature and expiration
    2. Verify token subject matches enrollment identity:
       a. Exact match for specific tokens (client, admin, relay)
       b. Pattern match for pattern tokens (e.g., "hospital-*")
    3. Verify token type matches enrollment type:
       - Client token → client enrollment only
       - Admin token → admin enrollment only
       - Relay token → relay enrollment only
       - Pattern token → any enrollment type
    4. Evaluate policy rules in order (first-match-wins):
       a. Check site_name_pattern if specified
       b. Check source_ips if specified (strict: required if in policy)
       c. Check admin roles if applicable
    5. Return approval decision (approve/reject/pending)

Certificate Generation
----------------------

When a CSR is approved, the signed certificate includes:

- **Common Name (CN)**: Client/user name from enrollment identity
- **Organization (O)**: Organization name (if provided)
- **Role (via Subject fields)**: 
  - Admin tokens: Embedded role (lead, member, org_admin) for downstream authorization
  - Relay tokens: "relay" identifier
  - Client tokens: No special role

Policy Match Conditions
-----------------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Condition
     - Description
   * - ``site_name_pattern``
     - Glob pattern matching against client name (e.g., ``hospital-*``)
   * - ``source_ips``
     - List of CIDR ranges the client IP must match (optional, strict enforcement)

***********************
Part 2: Integration
***********************

FL Server Integration
=====================

Location: ``nvflare/private/fed/server/fed_server.py``

The ``FederatedServer`` integrates ``CertService`` to handle enrollment requests.

Initialization
--------------

.. code-block:: python

    def _init_cert_service(self, grpc_args: dict):
        """Initialize CertService if root CA key is available."""
        root_ca_cert_path = grpc_args.get(SecureTrainConst.SSL_ROOT_CERT)
        root_ca_dir = os.path.dirname(root_ca_cert_path)
        root_ca_key_path = os.path.join(root_ca_dir, "rootCA.key")
        
        if not os.path.exists(root_ca_key_path):
            self.logger.info("CSR enrollment disabled: rootCA.key not found")
            return
            
        self.cert_service = CertService(
            root_ca_cert_path=root_ca_cert_path,
            root_ca_key_path=root_ca_key_path,
        )

CellNet Handler Registration
----------------------------

.. code-block:: python

    def _register_cellnet_cbs(self):
        # ... existing handlers ...
        
        # Add CSR enrollment handler
        if self.cert_service:
            self.cell.register_request_cb(
                channel=CellChannel.SERVER_MAIN,
                topic=CellChannelTopic.CSR_ENROLLMENT,
                cb=self.cert_service.handle_csr_enrollment,
            )

Authentication Bypass
---------------------

CSR enrollment requests bypass client certificate authentication since the client
doesn't have a certificate yet:

.. code-block:: python

    def _validate_auth_headers(self, message: Message):
        topic = message.get_header(MessageHeaderKey.TOPIC)
        channel = message.get_header(MessageHeaderKey.CHANNEL)
        
        # Allow CSR enrollment without authentication
        if topic == CellChannelTopic.CSR_ENROLLMENT and channel == CellChannel.SERVER_MAIN:
            return None
            
        # ... normal authentication ...

FL Client Integration
=====================

Location: ``nvflare/private/fed/client/fed_client_base.py``

The ``FederatedClientBase`` implements transparent auto-enrollment.

Auto-Enrollment Flow
--------------------

.. code-block:: python

    def _auto_enroll_if_needed(self, location: str, scheme: str) -> bool:
        """Check if enrollment is needed and perform it."""
        # 1. Check if secure training is enabled
        if not self.secure_train:
            return False
            
        # 2. Check if certificate already exists
        ssl_cert_path = self.client_args.get(SecureTrainConst.SSL_CERT)
        if ssl_cert_path and os.path.exists(ssl_cert_path):
            return False
            
        # 3. Look for enrollment token
        token = os.environ.get("NVFLARE_ENROLLMENT_TOKEN")
        if not token:
            # Check for token file in startup directory
            token_file = os.path.join(startup_dir, "enrollment_token")
            if os.path.exists(token_file):
                with open(token_file, "r") as f:
                    token = f.read().strip()
                    
        if not token:
            return False
            
        # 4. Perform enrollment
        self._perform_enrollment(location, scheme, token)
        return True

Token Sources
-------------

The client looks for enrollment tokens in this order:

1. ``NVFLARE_ENROLLMENT_TOKEN`` environment variable
2. ``enrollment_token`` file in the startup directory

Generic Startup Kit
-------------------

With token-based enrollment, clients receive a generic startup kit containing:

- ``fed_client.json`` - Client configuration
- ``resources.json`` - Resource configuration
- ``privacy.json`` - Privacy settings
- ``start.sh``, ``stop_fl.sh`` - Startup scripts

The startup kit does **not** contain client-specific certificates (``client.crt``,
``client.key``) or signatures. Instead, the client obtains its certificate
dynamically using the enrollment token.

.. note::

   You must also copy ``rootCA.pem`` from the provisioned workspace to the
   startup kit for TLS server verification.

Cert CLI (``nvflare cert``)
---------------------------

Location: ``nvflare/tool/enrollment/cert_cli.py``

The ``nvflare cert`` command generates root CA and server certificates for
token-based enrollment workflows.

**Subcommands:**

- ``init``: Initialize a root CA for the project
- ``server``: Generate a server certificate signed by the root CA

**Command Structure:**

.. code-block:: text

    nvflare cert
    ├── init            # Initialize root CA
    │   ├── -n/--name       Project/CA name
    │   ├── -o/--output     Output directory
    │   └── --valid_days    Certificate validity
    └── server          # Generate server certificate
        ├── -n/--name       Server name (required)
        ├── -c/--ca_path    Path to CA directory (required)
        ├── -o/--output     Output directory
        ├── --org           Organization name
        ├── --host          Default host name
        ├── --additional_hosts  Additional SANs
        └── --valid_days    Certificate validity

**Key Features:**

- **Standalone CA generation**: Create root CA without full provisioning
- **Server certificate generation**: Sign server certs with root CA
- **SAN support**: Add multiple host names/IPs to server certificate
- **Compatible with TokenService**: Output format works with ``nvflare token``

**Output Structure (init):**

.. code-block:: text

    output_dir/
    ├── rootCA.pem       # Root CA certificate
    ├── rootCA.key       # Root CA private key
    └── state/
        └── cert.json    # State file for token generation

**Output Structure (server):**

.. code-block:: text

    output_dir/
    ├── server.crt       # Server certificate
    ├── server.key       # Server private key
    └── rootCA.pem       # Root CA certificate (copy)

Package CLI (``nvflare package``)
---------------------------------

Location: ``nvflare/lighter/startup_kit.py``

The ``nvflare package`` command generates generic startup kits for token-based
enrollment. It reuses the existing provisioning infrastructure but filters out
``CertBuilder`` and ``SignatureBuilder``.

**Two Modes of Operation:**

1. **Project File Mode** (``-p``): Package ALL participants from a project.yml
2. **Single Participant Mode**: Create one package using CLI arguments

**Command Structure:**

.. code-block:: text

    nvflare package
    ├── -p/--project_file   Project YAML file (packages ALL participants)
    ├── -w/--workspace      Output workspace directory
    │
    │   # Single participant mode (when no -p):
    ├── -n/--name           Participant name (required if no -p)
    ├── -e/--endpoint       Connection URI (required if no -p)
    ├── -t/--type           Package type: server, client (default), relay, admin
    ├── --org               Organization name
    ├── --role              Role for admin type
    ├── --listening_host    Listening host for relay
    └── --listening_port    Listening port for relay

**Key Features:**

- **Reuses provisioner**: Uses the same ``Provisioner`` class as ``nvflare provision``
- **Filters builders**: Automatically removes ``CertBuilder`` and ``SignatureBuilder``
- **Preserves custom builders**: Keeps any custom builders from project.yml
- **Supports all participant types**: Client, relay, and admin packages
- **Two modes**: Batch generation from project.yml or single package creation

**Implementation:**

.. code-block:: python

    def package_from_project(project_file, workspace):
        """Package all participants from a project.yml without certificates."""
        project_dict = load_yaml(project_file)
        
        # Filter out CertBuilder and SignatureBuilder
        project_dict["builders"] = filter_builders(project_dict.get("builders", []))
        
        # Use existing provisioner logic
        project = prepare_project(project_dict)
        builders = prepare_builders(project_dict)
        provisioner = Provisioner(workspace, builders)
        ctx = provisioner.provision(project)
        return ctx.get_result_location()

    def generate_single_package(name, participant_type, endpoint_info, ...):
        """Generate a single participant package without certificates."""
        # Build minimal project_dict from CLI args
        # Filter builders, run provisioner
        ...

Token CLI
=========

Location: ``nvflare/tool/enrollment/token_cli.py``

The CLI provides a user-friendly interface for token management.

Command Structure
-----------------

.. code-block:: text

    nvflare token
    ├── generate    # Generate single token
    │   ├── -s/--subject    (required) Subject name
    │   ├── --user          Generate admin/user token (default role: lead)
    │   ├── --relay         Generate relay node token
    │   ├── -r/--role       Role for user tokens
    │   └── -o/--output     Output file
    ├── batch       # Generate multiple tokens
    └── info        # Display token information

Participant Types
-----------------

.. list-table::
   :widths: 15 20 65
   :header-rows: 1

   * - Type
     - CLI Flag
     - Description
   * - ``client``
     - (default)
     - FL client site (leaf node in federation)
   * - ``admin``
     - ``--user``
     - FLARE Console user (Admin Client) with role embedded in certificate
   * - ``relay``
     - ``--relay``
     - Relay node for hierarchical FL deployments

Environment Variables
---------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Description
   * - ``NVFLARE_CA_PATH``
     - Path to provisioned root CA directory (from ``nvflare provision``)
   * - ``NVFLARE_ENROLLMENT_POLICY``
     - Path to policy file (replaces ``-p`` option)

Default Policy
--------------

If no policy is specified, a built-in default is used that auto-approves all requests
with 7-day token validity. This enables quick-start scenarios while production
deployments should use custom policies.

CellNet Topic
=============

Location: ``nvflare/fuel/f3/cellnet/defs.py``

A new CellNet topic is added for CSR enrollment:

.. code-block:: python

    class CellChannelTopic:
        # ... existing topics ...
        CSR_ENROLLMENT = "csr_enrollment"

This topic is used for communication between ``CertRequestor`` (client) and
``CertService`` (server).

***********************
Security Considerations
***********************

Token Security
==============

- **Signed with RSA**: Tokens are signed with the root CA's private key (RS256)
- **Tamper-proof**: Modifying the token invalidates the signature
- **Single-use**: Each token can only be used once for enrollment
- **Time-limited**: Tokens expire after a configurable duration (default: 7 days)

Source IP Enforcement
=====================

When ``source_ips`` is specified in the policy:

- Client **must** provide its IP address
- Server **strictly enforces** the IP must match the CIDR ranges
- If client doesn't provide IP and policy requires it, enrollment is **rejected**

This provides protection against token theft in predictable network environments
where client IP addresses are known and stable (e.g., on-premise data centers,
cloud VMs with static IPs).

Certificate Trust
=================

- Signed certificates use the same root CA as the server
- Clients are granted trust based on the policy embedded in the token
- Certificate attributes (CN, OU, etc.) are derived from the enrollment identity

***********************
Backward Compatibility
***********************

Token-based enrollment is designed to be fully backward compatible:

1. **Existing deployments**: Clients with pre-provisioned certificates continue to work
2. **Server without root CA key**: CSR enrollment is automatically disabled
3. **Client without token**: Falls back to requiring pre-provisioned certificates
4. **Mixed deployments**: Some clients can use tokens while others use traditional provisioning

***********************
Future Enhancements
***********************

Potential future improvements:

- **Token revocation**: Ability to revoke tokens before expiration
- **Audit logging**: Comprehensive logging of enrollment events
- **Notification integration**: Email/webhook notifications for enrollment events
- **Policy versioning**: Support for policy migration and updates
- **Hardware-bound tokens**: Tokens tied to specific hardware identifiers

