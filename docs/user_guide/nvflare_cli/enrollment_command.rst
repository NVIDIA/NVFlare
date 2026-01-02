.. _enrollment_command:

#########################
Enrollment Token Command
#########################

The NVIDIA FLARE :mod:`Enrollment CLI<nvflare.tool.enrollment.enrollment_cli>` provides commands
to generate and manage enrollment tokens for dynamic client enrollment. This enables a simplified
deployment workflow where clients can join a federation using tokens instead of pre-provisioned
certificates.

.. note::

    The enrollment feature requires PyJWT as an optional dependency.
    Install with: ``pip install PyJWT`` or ``pip install nvflare[enrollment]``

***********************
Command Usage
***********************

.. code-block::

    usage: nvflare enrollment [-h] {token} ...

    Commands for managing FLARE enrollment tokens and certificates.

    Environment variables:
      NVFLARE_CA_PATH: Default CA directory path
      NVFLARE_ENROLLMENT_POLICY: Default policy file path

    options:
      -h, --help  show this help message and exit

    enrollment commands:
      {token}     Enrollment subcommand
        token     Manage enrollment tokens

Token Subcommands
=================

.. code-block::

    usage: nvflare enrollment token [-h] {generate,batch,info} ...

    Commands for generating and inspecting enrollment tokens (JWT).

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

The enrollment CLI supports environment variables to simplify repeated usage:

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
    nvflare enrollment token generate -s site-1

***********************
Generate Single Token
***********************

The ``nvflare enrollment token generate`` command creates a single enrollment token.

Usage
=====

.. code-block::

    usage: nvflare enrollment token generate [-h] -s SUBJECT [-c CA_PATH]
                                            [-p POLICY]
                                            [-t {client,admin,relay,pattern}]
                                            [-v VALIDITY] [-r ROLES [ROLES ...]]
                                            [--source_ips SOURCE_IPS [SOURCE_IPS ...]]
                                            [-o OUTPUT]

Options
=======

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Option
     - Description
   * - ``-s, --subject`` (required)
     - Subject identifier: site name, user email, or pattern (e.g., ``hospital-*``)
   * - ``-c, --ca_path``
     - Path to CA directory (or set ``NVFLARE_CA_PATH``)
   * - ``-p, --policy``
     - Path to policy YAML file (or set ``NVFLARE_ENROLLMENT_POLICY``, uses default if not set)
   * - ``-t, --type``
     - Subject type: ``client`` (default), ``admin``, ``relay``, or ``pattern``
   * - ``-v, --validity``
     - Token validity duration (e.g., ``7d``, ``24h``, ``30m``). Defaults to policy setting.
   * - ``-r, --roles``
     - Roles for admin tokens (e.g., ``org_admin researcher``)
   * - ``--source_ips``
     - Source IP restrictions in CIDR format (e.g., ``10.0.0.0/8``)
   * - ``-o, --output``
     - Output file to save token (prints to stdout if not specified)

Examples
========

**Basic Usage - Generate a client token:**

.. code-block:: shell

    nvflare enrollment token generate -s site-1 -c /path/to/ca

**With Environment Variables:**

.. code-block:: shell

    export NVFLARE_CA_PATH=/path/to/ca
    nvflare enrollment token generate -s site-1

**Generate an admin token with roles:**

.. code-block:: shell

    nvflare enrollment token generate -s admin@example.com -t admin -r org_admin researcher

**Generate a pattern token (matches multiple sites):**

.. code-block:: shell

    nvflare enrollment token generate -s "hospital-*" -t pattern

**Generate token with IP restrictions:**

.. code-block:: shell

    nvflare enrollment token generate -s site-1 --source_ips 10.0.0.0/8 192.168.0.0/16

**Save token to file:**

.. code-block:: shell

    nvflare enrollment token generate -s site-1 -o enrollment_token.txt

***********************
Batch Generate Tokens
***********************

The ``nvflare enrollment token batch`` command creates multiple tokens at once.

Usage
=====

.. code-block::

    usage: nvflare enrollment token batch [-h]
                                         (-n COUNT | --names NAMES [NAMES ...]) -o
                                         OUTPUT [-c CA_PATH] [-p POLICY]
                                         [--prefix PREFIX]
                                         [-t {client,admin,relay}] [-v VALIDITY]

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
   * - ``-t, --type``
     - Subject type for all tokens (default: ``client``)
   * - ``-v, --validity``
     - Token validity duration (e.g., ``7d``, ``24h``)

Examples
========

**Generate 10 tokens with auto-numbered names:**

.. code-block:: shell

    nvflare enrollment token batch -n 10 --prefix hospital -o tokens.csv

This creates tokens for: ``hospital-1``, ``hospital-2``, ..., ``hospital-10``

**Generate tokens for specific sites:**

.. code-block:: shell

    nvflare enrollment token batch --names site-a site-b site-c -o tokens.csv

**Generate admin tokens in batch:**

.. code-block:: shell

    nvflare enrollment token batch --names admin1@org.com admin2@org.com -t admin -o admin_tokens.csv

***********************
Inspect Token
***********************

The ``nvflare enrollment token info`` command displays token information without verifying the signature.

Usage
=====

.. code-block::

    usage: nvflare enrollment token info [-h] -t TOKEN [--json]

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

    nvflare enrollment token info -t "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."

**Inspect a token from file:**

.. code-block:: shell

    nvflare enrollment token info -t enrollment_token.txt

**Get JSON output:**

.. code-block:: shell

    nvflare enrollment token info -t enrollment_token.txt --json

Sample output:

.. code-block:: json

    {
      "token_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "subject": "site-1",
      "subject_type": "client",
      "issuer": "nvflare-enrollment",
      "issued_at": "2025-01-02T10:00:00+00:00",
      "expires_at": "2025-01-09T10:00:00+00:00",
      "max_uses": 1,
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
        - researcher
        - org_admin
        - project_admin
      default_role: researcher

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
        - researcher
        - org_admin
      default_role: researcher

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
Workflow Example
***********************

Here's a complete workflow for token-based enrollment:

1. **Set up environment variables (optional but recommended):**

.. code-block:: shell

    export NVFLARE_CA_PATH=/path/to/server/startup
    export NVFLARE_ENROLLMENT_POLICY=/path/to/enrollment_policy.yaml

2. **Generate tokens for clients:**

.. code-block:: shell

    # Generate batch tokens for 10 hospital sites
    nvflare enrollment token batch -n 10 --prefix hospital -o hospital_tokens.csv

3. **Distribute tokens securely to each site operator**

4. **On client side, use token for enrollment:**

.. code-block:: shell

    # Set the enrollment token
    export NVFLARE_ENROLLMENT_TOKEN="<token_from_csv>"

    # Or place in startup directory
    echo "<token>" > /path/to/startup/enrollment_token

5. **Start the client - enrollment happens automatically:**

.. code-block:: shell

    nvflare client start -m /path/to/startup

The client will automatically request and receive a certificate using the enrollment token.

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
    nvflare enrollment token generate -s site-1 -c /path/to/ca

**Error: PyJWT is required**

.. code-block:: shell

    Error: PyJWT is required for enrollment token operations.

Solution: Install the PyJWT dependency:

.. code-block:: shell

    pip install PyJWT

Token Inspection Shows Expired
==============================

If a token shows as expired when inspecting, generate a new token with a longer validity:

.. code-block:: shell

    nvflare enrollment token generate -s site-1 -v 30d

Client Enrollment Fails
=======================

If client enrollment fails with "token expired" or "invalid token":

1. Verify the token is valid using ``token info``
2. Check that server and client clocks are synchronized
3. Ensure the token subject matches the client name
4. Verify network connectivity to the server

