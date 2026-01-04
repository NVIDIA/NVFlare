.. _cert_command:

#########################
Cert Command
#########################

The ``nvflare cert`` command generates certificates for token-based enrollment.
It creates root CA certificates and server certificates without requiring the
full provisioning process.

***********************
Overview
***********************

The cert command is used by **Project Administrators** to:

1. Initialize a root Certificate Authority (CA)
2. Generate server certificates signed by the root CA

These certificates are then distributed to server operators, who use
``nvflare package`` to generate their own startup kits.

***********************
Command Usage
***********************

.. code-block::

    nvflare cert {init|server|api-key} [options]

Subcommands:

- ``init``: Initialize a root CA for the project
- ``server``: Generate a server certificate signed by the root CA
- ``api-key``: Generate a secure API key for Certificate Service

***********************
init Subcommand
***********************

Initialize a new root CA for your FL project.

Syntax
======

.. code-block:: shell

    nvflare cert init [-n NAME] [-o OUTPUT] [--valid_days DAYS]

Options
=======

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Option
     - Description
   * - ``-n, --name``
     - Project/CA name (default: ``NVFlare``)
   * - ``-o, --output``
     - Output directory (default: current directory)
   * - ``--valid_days``
     - Certificate validity in days (default: 3650 = 10 years)

Output Files
============

.. code-block:: text

    output_dir/
    ├── rootCA.pem       # Root CA certificate (distribute to all sites)
    ├── rootCA.key       # Root CA private key (keep secure!)
    └── state/
        └── cert.json    # State file for nvflare token command

Example
=======

.. code-block:: shell

    # Initialize root CA for a project
    nvflare cert init -n "My FL Project" -o ./my_project_ca

    # With custom validity
    nvflare cert init -n "My FL Project" -o ./my_project_ca --valid_days 1825

***********************
server Subcommand
***********************

Generate a server certificate signed by the root CA.

Syntax
======

.. code-block:: shell

    nvflare cert server -n NAME -c CA_PATH [-o OUTPUT] [--org ORG]
                        [--host HOST] [--additional_hosts HOSTS...]
                        [--valid_days DAYS]

Options
=======

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Option
     - Description
   * - ``-n, --name`` (required)
     - Server name (used as CN and identity)
   * - ``-c, --ca_path`` (required)
     - Path to CA directory (containing rootCA.pem/key or state/cert.json)
   * - ``-o, --output``
     - Output directory (default: current directory)
   * - ``--org``
     - Organization name (default: ``org``)
   * - ``--host``
     - Default host name (default: same as server name)
   * - ``--additional_hosts``
     - Additional host names for SAN extension (space-separated)
   * - ``--valid_days``
     - Certificate validity in days (default: 365)

Output Files
============

.. code-block:: text

    output_dir/
    ├── server.crt       # Server certificate
    ├── server.key       # Server private key (keep secure!)
    └── rootCA.pem       # Root CA certificate (copy for convenience)

Example
=======

.. code-block:: shell

    # Generate server certificate
    nvflare cert server -n server1 -c ./my_project_ca -o ./server_certs

    # With custom host names (for SAN extension)
    nvflare cert server -n server1 -c ./my_project_ca -o ./server_certs \
        --host server.example.com \
        --additional_hosts server1.example.com localhost 10.0.0.1

***********************
api-key Subcommand
***********************

Generate a secure API key for Certificate Service authentication.

Syntax
======

.. code-block:: shell

    nvflare cert api-key [-l LENGTH] [-o OUTPUT] [--format FORMAT]

Options
=======

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Option
     - Description
   * - ``-l, --length``
     - Key length in bytes (default: 32 = 256 bits)
   * - ``-o, --output``
     - Output file path (default: print to stdout)
   * - ``--format``
     - Output format: ``hex`` (default), ``base64``, or ``urlsafe``

Example
=======

.. code-block:: shell

    # Generate API key and print to stdout
    nvflare cert api-key

    # Generate 64-byte key and save to file
    nvflare cert api-key -l 64 -o api_key.txt

    # Generate in base64 format
    nvflare cert api-key --format base64

Usage
=====

After generating the API key, use it to authenticate with the Certificate Service:

.. code-block:: shell

    # Option 1: Set as environment variable
    export NVFLARE_API_KEY='<generated-key>'

    # Option 2: Pass directly to CLI commands
    nvflare token generate -n site-1 --cert-service https://... --api-key '<generated-key>'
    nvflare enrollment list --cert-service https://... --api-key '<generated-key>'

    # Option 3: Add to Certificate Service config file
    # In cert_service_config.yaml:
    #   api_key: "<generated-key>"

***********************
Complete Workflow
***********************

Here's the complete workflow using Option 2 (no startup kit distribution):

Phase 1: Project Admin - Initialize CA
======================================

.. code-block:: shell

    # Create root CA
    nvflare cert init -n "Hospital FL Project" -o ./hospital_project_ca

Phase 2: Project Admin - Generate Server Certificate
====================================================

.. code-block:: shell

    # Generate server certificate
    nvflare cert server -n server1 -c ./hospital_project_ca -o ./server_certs \
        --host fl-server.hospital.org \
        --additional_hosts server1.hospital.org

Phase 3: Distribute to Server Operator
======================================

Send only 3 files to server operator:

- ``server_certs/rootCA.pem``
- ``server_certs/server.crt``
- ``server_certs/server.key``

Phase 4: Server Operator - Generate Startup Kit
===============================================

.. code-block:: shell

    # Server operator generates their own startup kit
    nvflare package -n server1 -e grpc://fl-server.hospital.org:8002:8003 -t server

    # Copy received certificates to startup folder
    cp server_certs/rootCA.pem server1/startup/
    cp server_certs/server.crt server1/startup/
    cp server_certs/server.key server1/startup/

    # Start server
    cd server1 && ./startup/start.sh

Phase 5: Project Admin - Generate Tokens for Clients
====================================================

.. code-block:: shell

    # Generate enrollment tokens
    nvflare token generate -s hospital-1 -c ./hospital_project_ca
    nvflare token generate -s hospital-2 -c ./hospital_project_ca

Phase 6: Client Operators - Generate Startup Kits
=================================================

Each client operator generates their own startup kit:

.. code-block:: shell

    # Client operator generates startup kit
    nvflare package -n hospital-1 -e grpc://fl-server.hospital.org:8002

    # Copy rootCA.pem (received from project admin with token)
    cp rootCA.pem hospital-1/startup/

    # Set enrollment token and start
    export NVFLARE_ENROLLMENT_TOKEN="<token_from_admin>"
    cd hospital-1 && ./startup/start.sh

***********************
Security Considerations
***********************

Private Key Protection
======================

- The ``rootCA.key`` file contains the root CA private key
- **Never** distribute this file
- Store in a secure location with restricted access
- Consider hardware security modules (HSM) for production

Certificate Validity
====================

- Root CA: Typically 10 years (default: 3650 days)
- Server certificates: Typically 1 year (default: 365 days)
- Plan for certificate renewal before expiration

Subject Alternative Names (SAN)
===============================

Server certificates should include all hostnames/IPs that clients may use to connect:

.. code-block:: shell

    nvflare cert server -n server1 -c ./ca \
        --host server.example.com \
        --additional_hosts server1.example.com localhost 127.0.0.1

***********************
See Also
***********************

- :ref:`package_command` - Generate startup kits
- :ref:`token_command` - Generate enrollment tokens
- :ref:`provisioning` - Standard provisioning with certificates
- :ref:`enrollment_design_v2` - Design documentation for token-based enrollment

