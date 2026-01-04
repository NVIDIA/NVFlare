.. _token_command:

#########################
Token Command
#########################

The NVIDIA FLARE :mod:`Token CLI<nvflare.tool.enrollment.token_cli>` provides commands
to generate and manage enrollment tokens for dynamic client enrollment. This enables a simplified
deployment workflow where clients can join a federation using tokens instead of pre-provisioned
certificates.

***********************
Command Usage
***********************

.. code-block::

    usage: nvflare token [-h] {generate,batch,info} ...

    Commands for generating and inspecting enrollment tokens (JWT).

    Environment variables:
      NVFLARE_CA_PATH: Default CA directory path
      NVFLARE_ENROLLMENT_POLICY: Default policy file path

    options:
      -h, --help            show this help message and exit

    token commands:
      {generate,batch,info}
        generate            Generate a single enrollment token
        batch               Generate multiple enrollment tokens
        info                Display token information

***********************
Two Modes of Operation
***********************

The token CLI supports two modes:

1. **Local Mode** (Manual Workflow): Generate tokens using local root CA from provisioning
2. **Remote Mode** (Auto-Scale Workflow): Generate tokens via Certificate Service API

***********************
Environment Variables
***********************

The token CLI supports environment variables to simplify repeated usage:

**Local Mode (Manual Workflow):**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Description
   * - ``NVFLARE_CA_PATH``
     - Path to the provisioned workspace directory (created by ``nvflare provision``).
       For example: ``/path/to/workspace/example_project``
   * - ``NVFLARE_ENROLLMENT_POLICY``
     - Path to enrollment policy YAML file (optional, uses built-in default if not set)

**Remote Mode (Auto-Scale Workflow):**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Description
   * - ``NVFLARE_CERT_SERVICE_URL``
     - URL of the Certificate Service (e.g., ``https://cert-service:8443``)
   * - ``NVFLARE_API_KEY``
     - API key for authenticating with the Certificate Service

.. note::

   **Local Mode:** The ``NVFLARE_CA_PATH`` should point to the **project workspace** created
   by provisioning, not the server's startup kit. The token service reads the root CA from
   ``state/cert.json`` in the provisioning workspace.

   **Remote Mode:** When using ``--cert-service``, the Certificate Service generates and signs
   tokens using its own root CA. The local ``-c`` option is not needed.

Setting these variables allows you to omit options from commands:

.. code-block:: shell

    # Local Mode (Manual Workflow)
    export NVFLARE_CA_PATH=/path/to/workspace/my_project
    nvflare token generate -s hospital-1

    # Remote Mode (Auto-Scale Workflow)
    export NVFLARE_CERT_SERVICE_URL=https://cert-service:8443
    export NVFLARE_API_KEY=your-api-key
    nvflare token generate -s hospital-1

***********************
Generate Single Token
***********************

The ``nvflare token generate`` command creates a single enrollment token.

Usage
=====

.. code-block::

    usage: nvflare token generate [-h] -s SUBJECT [--user | --relay]
                                  [-c CA_PATH] [-p POLICY] [-r ROLE] [-o OUTPUT]
                                  [--cert-service URL] [--api-key KEY]

Options
=======

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Option
     - Description
   * - ``-s, --subject`` (required)
     - Subject identifier: site name or user email
   * - ``--user``
     - Generate a user/admin token (for FLARE Console users). Default role: ``lead``
   * - ``--relay``
     - Generate a relay node token (for hierarchical FL)
   * - ``-c, --ca_path``
     - **Local Mode:** Path to provisioned workspace (or set ``NVFLARE_CA_PATH``)
   * - ``-p, --policy``
     - Path to policy YAML file (or set ``NVFLARE_ENROLLMENT_POLICY``, uses default if not set)
   * - ``-r, --role``
     - Role for user tokens: ``lead`` (default), ``member``, or ``org_admin``
   * - ``-o, --output``
     - Output file to save token (prints to stdout if not specified)
   * - ``--cert-service``
     - **Remote Mode:** URL of Certificate Service (or set ``NVFLARE_CERT_SERVICE_URL``)
   * - ``--api-key``
     - **Remote Mode:** API key for Certificate Service (or set ``NVFLARE_API_KEY``)

Token Types
===========

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Type
     - Description
   * - **Client** (default)
     - FL client site (leaf node). Used for hospital sites, research centers, etc.
   * - **User** (``--user``)
     - FLARE Console user (Admin Client). Role embedded in certificate for authorization.
   * - **Relay** (``--relay``)
     - Relay node for hierarchical FL deployments.

Examples
========

**Local Mode (Manual Workflow)**

Generate tokens using local root CA from provisioning:

.. code-block:: shell

    # Basic client token
    nvflare token generate -s hospital-1 -c /path/to/workspace/my_project

    # With environment variable
    export NVFLARE_CA_PATH=/path/to/workspace/my_project
    nvflare token generate -s hospital-1

**Remote Mode (Auto-Scale Workflow)**

Generate tokens via Certificate Service API:

.. code-block:: shell

    # Generate token via Certificate Service
    nvflare token generate -s hospital-1 \
        --cert-service https://cert-service:8443 \
        --api-key $API_KEY

    # With environment variables
    export NVFLARE_CERT_SERVICE_URL=https://cert-service:8443
    export NVFLARE_API_KEY=your-api-key
    nvflare token generate -s hospital-1

**Generate a user token (for FLARE Console):**

.. code-block:: shell

    # Default role is "lead"
    nvflare token generate -s admin@example.com --user

    # Specify a different role
    nvflare token generate -s researcher@example.com --user -r member

**Generate a relay token (for hierarchical FL):**

.. code-block:: shell

    nvflare token generate -s relay-1 --relay

**Save token to file:**

.. code-block:: shell

    nvflare token generate -s hospital-1 -o enrollment_token.txt

***********************
Batch Generate Tokens
***********************

The ``nvflare token batch`` command creates multiple tokens at once.

Usage
=====

.. code-block::

    usage: nvflare token batch [-h] (-n COUNT | --names NAMES [NAMES ...])
                               -o OUTPUT [-c CA_PATH] [-p POLICY]
                               [--prefix PREFIX] [--user | --relay] [-r ROLE]
                               [--cert-service URL] [--api-key KEY]

Options
=======

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Option
     - Description
   * - ``-n, --count``
     - Number of tokens to generate (used with ``--prefix``)
   * - ``--names``
     - Explicit list of subject names (alternative to ``-n``)
   * - ``-o, --output`` (required)
     - Output file to save tokens (``.csv`` or ``.txt``)
   * - ``-c, --ca_path``
     - **Local Mode:** Path to provisioned workspace (or set ``NVFLARE_CA_PATH``)
   * - ``-p, --policy``
     - Path to policy YAML file (or set ``NVFLARE_ENROLLMENT_POLICY``)
   * - ``--prefix``
     - Prefix for auto-generated names when using ``--count`` (default: ``client``)
   * - ``--user``
     - Generate user/admin tokens for all subjects
   * - ``--relay``
     - Generate relay tokens for all subjects
   * - ``-r, --role``
     - Role for user tokens (default: ``lead``)
   * - ``--cert-service``
     - **Remote Mode:** URL of Certificate Service (or set ``NVFLARE_CERT_SERVICE_URL``)
   * - ``--api-key``
     - **Remote Mode:** API key for Certificate Service (or set ``NVFLARE_API_KEY``)

Examples
========

**Local Mode (Manual Workflow):**

.. code-block:: shell

    # Generate 10 client tokens with auto-numbered names
    nvflare token batch -n 10 --prefix hospital -o tokens.csv

    # Creates tokens for: hospital-1, hospital-2, ..., hospital-10

**Remote Mode (Auto-Scale Workflow):**

.. code-block:: shell

    # Generate 100 tokens via Certificate Service
    nvflare token batch -n 100 --prefix site \
        --cert-service https://cert-service:8443 \
        --api-key $API_KEY \
        -o tokens.csv

**Generate tokens for specific sites:**

.. code-block:: shell

    nvflare token batch --names site-a site-b site-c -o tokens.csv

**Generate user tokens in batch:**

.. code-block:: shell

    nvflare token batch --names user1@org.com user2@org.com --user -o user_tokens.csv

**Generate relay tokens in batch:**

.. code-block:: shell

    nvflare token batch -n 3 --prefix relay --relay -o relay_tokens.csv

***********************
Inspect Token
***********************

The ``nvflare token info`` command displays token information without verifying the signature.

Usage
=====

.. code-block::

    usage: nvflare token info [-h] -t TOKEN [--json]

Options
=======

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Option
     - Description
   * - ``-t, --token`` (required)
     - JWT token string or path to file containing token
   * - ``--json``
     - Output in JSON format

Examples
========

**Inspect a token directly:**

.. code-block:: shell

    nvflare token info -t "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."

**Inspect a token from file:**

.. code-block:: shell

    nvflare token info -t enrollment_token.txt

**Get JSON output:**

.. code-block:: shell

    nvflare token info -t enrollment_token.txt --json

Sample output:

.. code-block:: json

    {
      "token_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "subject": "hospital-1",
      "subject_type": "client",
      "issuer": "nvflare-enrollment",
      "issued_at": "2025-01-02T10:00:00+00:00",
      "expires_at": "2025-01-09T10:00:00+00:00",
      "policy_project": "nvflare-default",
      "policy_version": "1.0"
    }

***********************
Enrollment Policy
***********************

Enrollment tokens are generated based on a policy that defines approval rules,
token validity, and constraints.

Default Policy
==============

If no policy is specified (via ``-p`` or ``NVFLARE_ENROLLMENT_POLICY``), a built-in
default policy is used:

.. code-block:: yaml

    metadata:
      project: "nvflare-default"
      description: "Built-in default policy for quick start"
      version: "1.0"

    token:
      validity: 7d
      max_uses: 1

    site:
      name_pattern: "*"

    user:
      allowed_roles:
        - lead
        - member
        - org_admin
      default_role: lead

    approval:
      method: policy
      rules:
        - name: "auto-approve-all"
          description: "Auto-approve all enrollment requests"
          match: {}
          action: approve

    notifications:
      enabled: false

Custom Policy
=============

For production deployments, create a custom policy file:

.. code-block:: yaml

    metadata:
      project: "my-federation"
      description: "Production enrollment policy"
      version: "1.0"

    token:
      validity: 7d
      max_uses: 1

    site:
      name_pattern: "hospital-*"

    user:
      allowed_roles:
        - lead
        - member
        - org_admin
      default_role: lead

    approval:
      method: policy
      rules:
        - name: "internal-auto-approve"
          description: "Auto-approve from internal network"
          match:
            site_name_pattern: "hospital-*"
            source_ips:
              - "10.0.0.0/8"
              - "172.16.0.0/12"
          action: approve

        - name: "external-pending"
          description: "Require manual approval for external"
          match:
            site_name_pattern: "*"
          action: pending

    notifications:
      enabled: true
      channels:
        email:
          enabled: true

***********************
End-to-End Workflow
***********************

This section describes the complete workflow for deploying NVIDIA FLARE with
token-based client enrollment.

Workflow Overview
=================

.. list-table::
   :widths: 15 35 50
   :header-rows: 1

   * - Role
     - Responsibility
     - Deliverables
   * - **Project Admin**
     - Provision server, generate tokens, create generic packages
     - Server startup kit, generic client packages (via ``nvflare package``), enrollment tokens
   * - **Client Operator**
     - Deploy client with token
     - Running FL client with enrolled certificate

Phase 1: Provision and Deploy Server
====================================

First, provision the FL server using the standard provisioning process. This creates
the root CA and server certificates that will be used for token signing.

**Step 1: Create project.yml**

Create a ``project.yml`` file defining your FL project. For token-based enrollment,
you only need to define the server (clients will enroll dynamically):

.. code-block:: yaml

    api_version: 3
    name: my_fl_project
    description: FL project with token-based enrollment

    participants:
      - name: server.example.com
        type: server
        org: MyOrganization
        fed_learn_port: 8002
        admin_port: 8003

      # Admin user for management
      - name: admin@example.com
        type: admin
        org: MyOrganization
        role: project_admin

    # Note: No client participants defined - they will enroll with tokens

**Step 2: Run Provisioning**

.. code-block:: shell

    nvflare provision -p project.yml -w /path/to/workspace

This generates:

- ``my_fl_project/prod_00/server.example.com/`` - Server startup kit with certificates
- ``my_fl_project/prod_00/admin@example.com/`` - Admin console startup kit
- ``my_fl_project/state/cert.json`` - Root CA certificate and private key (used for token signing)

**Step 3: Deploy and Start Server**

Copy the server startup kit to your server machine and start it:

.. code-block:: shell

    # On server machine
    cd /path/to/server.example.com/startup
    ./start.sh

The server is now running and ready to accept client enrollments.

The workspace structure after provisioning:

.. code-block:: text

    /path/to/workspace/
    └── my_fl_project/              <-- NVFLARE_CA_PATH points here
        ├── state/
        │   └── cert.json           <-- Contains root CA (used for token signing)
        └── prod_00/
            ├── server.example.com/
            │   └── startup/
            │       ├── rootCA.pem
            │       ├── server.crt
            │       └── server.key
            └── admin@example.com/
                └── startup/

Phase 2: Generate Enrollment Tokens
===================================

Using the root CA from provisioning, generate tokens for clients.

**Step 1: Set up environment (recommended)**

.. code-block:: shell

    # Point to the provisioned workspace directory
    # Example: /path/to/workspace/example_project (contains state/cert.json)
    export NVFLARE_CA_PATH=/path/to/workspace/my_project

    # Optional: use custom policy
    export NVFLARE_ENROLLMENT_POLICY=/path/to/enrollment_policy.yaml

**Step 2: Generate tokens**

.. code-block:: shell

    # Generate a single client token
    nvflare token generate -s hospital-1

    # Generate batch tokens for multiple clients
    nvflare token batch -n 10 --prefix hospital -o hospital_tokens.csv

    # Generate user tokens for admin console users
    nvflare token generate -s admin@hospital.org --user -r lead

**Step 3: Securely distribute tokens**

Distribute tokens to each client site operator through a secure channel
(encrypted email, secure file transfer, etc.).

Phase 3: Create Generic Client Package
======================================

The Project Admin creates a **generic client package** that can be distributed to all sites.
This is essentially a provisioned startup kit but **without** certificates and signatures.

**Option A: Use ``nvflare package`` command (recommended)**

The simplest way to create a generic package is using the ``nvflare package`` command:

.. code-block:: shell

    # Generate client package
    nvflare package -n site-1 -e grpc://server.example.com:8002

    # Generate admin console package
    nvflare package -n admin@example.com -e grpc://server:8002:8003 -t admin

    # Generate relay package (for hierarchical FL)
    nvflare package -n relay-east -e grpc://server:8002 -t relay --listening_port 8010

This command:

- Reuses the existing provisioning infrastructure
- Automatically filters out ``CertBuilder`` and ``SignatureBuilder``
- Preserves all other custom builders from your project.yml

**Package command options:**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Option
     - Description
   * - ``-n, --name`` (required)
     - Participant name (site name, relay name, or user email)
   * - ``-e, --endpoint`` (required)
     - Connection URI: ``scheme://host:port`` or ``scheme://host:fl_port:admin_port``
   * - ``-t, --type``
     - Package type: ``client`` (default), ``relay``, ``admin``
   * - ``-o, --output``
     - Output directory (default: ``./<name>``)
   * - ``-p, --project_file``
     - Custom project.yml (uses default if not specified)
   * - ``--org``
     - Organization name (default: ``org``)
   * - ``--role``
     - Role for admin type: ``lead`` (default), ``member``, ``org_admin``
   * - ``--listening_host``
     - Listening host for relay (default: ``localhost``)
   * - ``--listening_port``
     - Listening port for relay (default: ``8002``)

**Option B: Use custom project.yml**

For advanced customization, create a project.yml that excludes certificate builders:

.. code-block:: yaml

    api_version: 3
    name: generic_client
    description: Generic client package for token-based enrollment

    participants:
      - name: site
        type: client
        org: TBD

    # Only include builders that don't generate certificates
    builders:
      - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
      - path: nvflare.lighter.impl.static_file.StaticFileBuilder
        args:
          config_folder: config
      # Note: CertBuilder and SignatureBuilder are NOT included

Then use the package command with your custom project:

.. code-block:: shell

    nvflare package -n site-1 -e grpc://server:8002 -p custom_project.yml

**Generated package structure:**

.. code-block:: text

    site-1/
    ├── local/
    │   ├── authorization.json.default
    │   ├── log_config.json.default
    │   ├── privacy.json.sample
    │   └── resources.json.default
    ├── startup/
    │   ├── fed_client.json       <-- Connection configuration
    │   ├── start.sh
    │   ├── stop_fl.sh
    │   └── sub_start.sh
    ├── transfer/
    └── readme.txt

.. note::

   The generic package does NOT include client-specific certificates (``client.crt``,
   ``client.key``), ``rootCA.pem``, or ``signature.json``. You must copy the ``rootCA.pem``
   from your provisioned workspace to enable TLS server verification.

Phase 4: Distribute to Client Sites
===================================

The Project Admin distributes to each client site:

1. **Generic client package** (same for all sites)
2. **Site-specific enrollment token** (unique per site)

**Delivery methods:**

- Secure file transfer (SFTP, SCP)
- Encrypted email
- Secure download portal
- Container registry (for Docker deployments)

Phase 5: Client Enrollment and Startup
======================================

Each client site receives the generic package and their enrollment token.

**Step 1: Configure enrollment token**

On the client machine, set the enrollment token:

.. code-block:: shell

    # Option A: Environment variable
    export NVFLARE_ENROLLMENT_TOKEN="<token_from_project_admin>"

    # Option B: Token file in startup directory
    echo "<token>" > /path/to/site-1/startup/enrollment_token

**Step 2: Start the client**

.. code-block:: shell

    cd /path/to/site-1/startup
    ./start.sh

The client will automatically:

1. Detect missing client certificate (``client.crt``)
2. Find the enrollment token (from env var or file)
3. Generate a key pair and CSR
4. Submit CSR with token to the server
5. Receive and save the signed certificate
6. Connect to FL server with the new certificate

Upon successful enrollment, you'll see in the client log:

.. code-block:: text

    Enrollment successful. Certificate saved to: /path/to/startup/client.crt
    Successfully registered client:hospital-1

Phase 6: Verify Deployment
==========================

**Check server for connected clients:**

From the admin console:

.. code-block:: shell

    cd /path/to/admin@example.com/startup
    ./fl_admin.sh

Then run:

.. code-block:: text

    > check_status server

This displays the server status and all connected clients.

**Run preflight check:**

.. code-block:: shell

    # On server
    nvflare preflight_check -p /path/to/server

    # On clients (after enrollment)
    nvflare preflight_check -p /path/to/client

***********************
Troubleshooting
***********************

Token Generation Fails
======================

**Error: CA path is required**

.. code-block:: shell

    Error: CA path is required.

Solution: Point to the provisioned workspace directory:

.. code-block:: shell

    # Set to the project workspace (e.g., workspace/my_project)
    export NVFLARE_CA_PATH=/path/to/workspace/my_project
    
    # Or provide directly
    nvflare token generate -s hospital-1 -c /path/to/workspace/my_project

**Error: Root CA not found**

If you see an error about missing root CA, ensure you're pointing to the correct
provisioned workspace that contains ``state/cert.json``.

Token Inspection Shows Expired
==============================

If a token shows as expired when inspecting, generate a new token. Token validity is controlled
by the policy file (default: 7 days). To use a longer validity, create a custom policy with
a longer ``token.validity`` setting.

Client Enrollment Fails
=======================

If client enrollment fails with "token expired" or "invalid token":

1. Verify the token is valid using ``token info``
2. Check that server and client clocks are synchronized
3. Ensure the token subject matches the client name
4. Verify network connectivity to the server


