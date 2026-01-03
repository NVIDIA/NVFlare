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
Environment Variables
***********************

The token CLI supports environment variables to simplify repeated usage:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Description
   * - ``NVFLARE_CA_PATH``
     - Path to directory containing ``rootCA.pem`` and ``rootCA.key``
   * - ``NVFLARE_ENROLLMENT_POLICY``
     - Path to enrollment policy YAML file (optional, uses built-in default if not set)

Setting these variables allows you to omit the ``-c`` and ``-p`` options from commands:

.. code-block:: shell

    # Set once in your environment
    export NVFLARE_CA_PATH=/path/to/startup/ca
    export NVFLARE_ENROLLMENT_POLICY=/path/to/my_policy.yaml

    # Then use simplified commands
    nvflare token generate -s site-1

***********************
Generate Single Token
***********************

The ``nvflare token generate`` command creates a single enrollment token.

Usage
=====

.. code-block::

    usage: nvflare token generate [-h] -s SUBJECT [--user | --relay]
                                  [-c CA_PATH] [-p POLICY] [-r ROLE] [-o OUTPUT]

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
     - Path to CA directory (or set ``NVFLARE_CA_PATH``)
   * - ``-p, --policy``
     - Path to policy YAML file (or set ``NVFLARE_ENROLLMENT_POLICY``, uses default if not set)
   * - ``-r, --role``
     - Role for user tokens: ``lead`` (default), ``member``, or ``org_admin``
   * - ``-o, --output``
     - Output file to save token (prints to stdout if not specified)

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

**Basic Usage - Generate a client token:**

.. code-block:: shell

    nvflare token generate -s hospital-1 -c /path/to/ca

**With Environment Variables:**

.. code-block:: shell

    export NVFLARE_CA_PATH=/path/to/ca
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
     - Path to CA directory (or set ``NVFLARE_CA_PATH``)
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

Examples
========

**Generate 10 client tokens with auto-numbered names:**

.. code-block:: shell

    nvflare token batch -n 10 --prefix hospital -o tokens.csv

This creates tokens for: ``hospital-1``, ``hospital-2``, ..., ``hospital-10``

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
token-based client enrollment, starting from server provisioning.

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

- ``server.example.com/`` - Server startup kit with certificates
- ``admin@example.com/`` - Admin console startup kit
- Root CA files (``rootCA.pem``, ``rootCA.key``) in the workspace

**Step 3: Deploy and Start Server**

Copy the server startup kit to your server machine and start it:

.. code-block:: shell

    # On server machine
    cd /path/to/server.example.com/startup
    ./start.sh

The server is now running and ready to accept client enrollments.

Phase 2: Generate Enrollment Tokens
===================================

Using the root CA from provisioning, generate tokens for clients.

**Step 1: Set up environment (recommended)**

.. code-block:: shell

    # Point to the workspace containing rootCA.pem and rootCA.key
    export NVFLARE_CA_PATH=/path/to/workspace

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

Phase 3: Client Enrollment and Startup
======================================

Each client site receives a generic startup kit (without client-specific certificates)
and an enrollment token.

**Step 1: Prepare generic client startup kit**

Create a generic client startup directory containing:

- ``rootCA.pem`` - Copy from server provisioning (for server verification)
- ``fed_client.json`` - Client configuration pointing to server
- ``resources.json`` - Resource configuration (optional)

Example ``fed_client.json``:

.. code-block:: json

    {
      "servers": [
        {
          "name": "server.example.com",
          "service": {
            "target": "server.example.com:8002",
            "options": [
              ["grpc.max_send_message_length", 1073741824],
              ["grpc.max_receive_message_length", 1073741824]
            ]
          }
        }
      ],
      "client": {
        "retry_timeout": 30,
        "ssl_root_cert": "rootCA.pem"
      }
    }

**Step 2: Configure enrollment token**

On the client machine, set the enrollment token:

.. code-block:: shell

    # Option A: Environment variable
    export NVFLARE_ENROLLMENT_TOKEN="<token_from_distribution>"

    # Option B: Token file in startup directory
    echo "<token>" > /path/to/startup/enrollment_token

**Step 3: Start the client**

.. code-block:: shell

    cd /path/to/client/startup
    ./start.sh

The client will:

1. Detect missing client certificate
2. Find the enrollment token (from env var or file)
3. Generate a key pair and CSR
4. Submit CSR with token to the server
5. Receive and save the signed certificate
6. Connect to FL server with the new certificate

Upon successful enrollment, you'll see in the client log:

.. code-block:: text

    Enrollment successful. Certificate saved to: /path/to/startup/client.crt
    Successfully registered client:hospital-1

Phase 4: Verify Deployment
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
    Provide via -c/--ca_path or set NVFLARE_CA_PATH environment variable.

Solution: Either set the environment variable or provide the ``-c`` option:

.. code-block:: shell

    export NVFLARE_CA_PATH=/path/to/ca
    # or
    nvflare token generate -s site-1 -c /path/to/ca

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


