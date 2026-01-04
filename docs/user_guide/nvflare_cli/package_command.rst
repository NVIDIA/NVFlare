.. _package_command:

#########################
Package Command
#########################

The ``nvflare package`` command generates generic startup kit packages for token-based
enrollment. Unlike the standard ``nvflare provision`` command, this creates packages
**without** certificates, allowing clients to obtain certificates dynamically using
enrollment tokens.

***********************
Two Modes of Operation
***********************

The command supports two modes:

1. **Project File Mode** (``-p``): Package ALL participants from a project.yml
2. **Single Participant Mode**: Create one package using CLI arguments

***********************
Command Usage
***********************

.. code-block::

    usage: nvflare package [-h] [-p PROJECT_FILE] [-w WORKSPACE]
                           [-n NAME] [-e ENDPOINT] [-t {server,client,relay,admin}]
                           [--org ORG] [--role {lead,member,org_admin}]
                           [--listening_host LISTENING_HOST]
                           [--listening_port LISTENING_PORT]
                           [--cert-service URL] [--token TOKEN]

    Generate startup kit packages without certificates for dynamic enrollment.

***********************
Options
***********************

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Option
     - Description
   * - ``-p, --project_file``
     - Path to project.yml. When provided, packages ALL participants (no other args needed)
   * - ``-w, --workspace``
     - Output workspace directory (default: ``workspace``)
   * - ``-n, --name``
     - Participant name (required if no ``-p``)
   * - ``-e, --endpoint``
     - Connection URI (required if no ``-p``). Formats:
       
       - Single port: ``grpc://server:8002``
       - Two ports: ``grpc://server:8002:8003`` (fl_port:admin_port)
   * - ``-t, --type``
     - Package type: ``server``, ``client`` (default), ``relay``, ``admin``
   * - ``--org``
     - Organization name (default: ``org``)
   * - ``--role``
     - Role for admin type: ``lead`` (default), ``member``, ``org_admin``
   * - ``--listening_host``
     - Listening host for relay type (default: ``localhost``)
   * - ``--listening_port``
     - Listening port for relay type (default: ``8002``)
   * - ``--cert-service``
     - Certificate Service URL to embed in the package (for Auto-Scale workflow)
   * - ``--token``
     - Enrollment token to embed in the package (for Auto-Scale workflow)

***********************
Supported Schemes
***********************

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Scheme
     - Description
   * - ``grpc``
     - gRPC protocol (most common)
   * - ``http``
     - HTTP/HTTPS protocol
   * - ``tcp``
     - TCP protocol

***********************
Examples
***********************

Project File Mode (Package All Participants)
============================================

When you have a project.yml file, use ``-p`` to package ALL participants at once:

.. code-block:: shell

    # Package all participants from project.yml (without certificates)
    nvflare package -p my_project.yml

    # Specify custom output workspace
    nvflare package -p my_project.yml -w /path/to/output

This is similar to ``nvflare provision`` but:

- Removes ``CertBuilder`` (no certificates generated)
- Removes ``SignatureBuilder`` (no signatures generated)
- Preserves all other builders (Docker, Helm, etc.)

**Example project.yml:**

.. code-block:: yaml

    api_version: 3
    name: my_fl_project
    description: FL project for hospitals
    
    participants:
      - name: server1
        type: server
        org: nvidia
        fed_learn_port: 8002
        admin_port: 8003
      - name: hospital-1
        type: client
        org: nvidia
      - name: hospital-2
        type: client
        org: nvidia
      - name: admin@nvidia.com
        type: admin
        org: nvidia
        role: lead
    
    builders:
      - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
      - path: nvflare.lighter.impl.static_file.StaticFileBuilder
        args:
          scheme: grpc
      - path: nvflare.lighter.impl.cert.CertBuilder    # Will be filtered out
      - path: nvflare.lighter.impl.signature.SignatureBuilder  # Will be filtered out
      - path: nvflare.lighter.impl.docker.DockerBuilder  # Will be preserved!

Single Participant Mode
=======================

Without ``-p``, you can create individual packages:

Generate Server Package
-----------------------

.. code-block:: shell

    # Server package (no certificates - add them separately)
    nvflare package -n server1 -e grpc://server.example.com:8002:8003 -t server

    # After generation, copy certificates from project admin:
    # cp server.crt server.key rootCA.pem server1/startup/

Generate Client Package
-----------------------

.. code-block:: shell

    # Basic client package
    nvflare package -n hospital-1 -e grpc://fl-server.example.com:8002

    # Client with custom organization
    nvflare package -n hospital-1 -e grpc://server:8002 --org HealthNetwork

    # Client with two-port configuration
    nvflare package -n hospital-1 -e grpc://server:8002:8003

Generate Client Package with Embedded Enrollment (Auto-Scale Workflow)
----------------------------------------------------------------------

For the Auto-Scale workflow, embed the Certificate Service URL and enrollment token
directly in the package:

.. code-block:: shell

    # Embed Certificate Service URL and token
    nvflare package -n hospital-1 -e grpc://server:8002 \
        --cert-service https://cert-service:8443 \
        --token "$TOKEN"

    # Just embed URL (token provided at runtime via env var)
    nvflare package -n hospital-1 -e grpc://server:8002 \
        --cert-service https://cert-service:8443

This creates additional files in the ``startup/`` directory:

- ``enrollment.json`` - Contains the Certificate Service URL
- ``enrollment_token`` - Contains the enrollment token (if ``--token`` provided)

At runtime, the client reads these files to perform auto-enrollment. Environment
variables (``NVFLARE_CERT_SERVICE_URL``, ``NVFLARE_ENROLLMENT_TOKEN``) take precedence
over the embedded files.

Generate Admin Package
----------------------

.. code-block:: shell

    # Admin console package with default role (lead)
    nvflare package -n admin@example.com -e grpc://server:8002 -t admin

    # Admin with specific role
    nvflare package -n orgadmin@hospital.org -e grpc://server:8002 -t admin --role org_admin

Generate Relay Package
----------------------

.. code-block:: shell

    # Relay node for hierarchical FL
    nvflare package -n relay-east -e grpc://server:8002 -t relay

    # Relay with custom listening port
    nvflare package -n relay-west -e grpc://server:8002 -t relay --listening_port 8010

***********************
Generated Structure
***********************

Project File Mode Output
========================

When using ``-p``, all participants are generated in the workspace:

.. code-block:: text

    workspace/my_fl_project/prod_00/
    ├── server1/
    │   └── startup/
    │       ├── fed_server.json
    │       ├── start.sh
    │       └── ...
    ├── hospital-1/
    │   └── startup/
    │       ├── fed_client.json
    │       ├── start.sh
    │       └── ...
    ├── hospital-2/
    │   └── ...
    └── admin@nvidia.com/
        └── ...

**Note:** No certificates or signatures are included in any package.

Server Package
==============

.. code-block:: text

    server1/
    ├── local/
    │   ├── authorization.json.default
    │   ├── log_config.json.default
    │   ├── privacy.json.sample
    │   └── resources.json.default
    ├── startup/
    │   ├── fed_server.json
    │   ├── start.sh
    │   ├── stop_fl.sh
    │   └── sub_start.sh
    └── transfer/

**After generation, copy certificates from project admin:**

.. code-block:: shell

    cp rootCA.pem server.crt server.key server1/startup/

Client Package
==============

.. code-block:: text

    site-1/
    ├── local/
    │   ├── authorization.json.default
    │   ├── log_config.json.default
    │   ├── privacy.json.sample
    │   └── resources.json.default
    ├── startup/
    │   ├── fed_client.json
    │   ├── start.sh
    │   ├── stop_fl.sh
    │   └── sub_start.sh
    ├── transfer/
    └── readme.txt

**With ``--cert-service`` and ``--token`` options:**

.. code-block:: text

    site-1/
    ├── ...
    ├── startup/
    │   ├── fed_client.json
    │   ├── enrollment.json       <-- Contains cert_service_url
    │   ├── enrollment_token      <-- Contains token (if --token provided)
    │   ├── start.sh
    │   └── ...

Admin Package
=============

.. code-block:: text

    admin@example.com/
    ├── local/
    │   └── resources.json.default
    ├── startup/
    │   ├── fed_admin.json
    │   ├── fl_admin.sh
    │   └── readme.txt
    └── transfer/

Relay Package
=============

.. code-block:: text

    relay-east/
    ├── local/
    │   ├── comm_config.json
    │   └── log_config.json.default
    ├── startup/
    │   ├── fed_relay.json
    │   ├── start.sh
    │   ├── stop_fl.sh
    │   └── sub_start.sh
    └── transfer/

***********************
How It Works
***********************

The ``nvflare package`` command reuses the existing provisioning infrastructure:

1. **Loads project configuration** from the specified file or builds a minimal project
2. **Filters builders** to remove ``CertBuilder`` and ``SignatureBuilder``
3. **Preserves all other builders** (Docker, Helm, etc.) from the project file
4. **Runs the provisioner** with the filtered builder list
5. **Outputs packages** without certificates or signatures

This ensures that any custom builders defined in your project.yml are preserved
while only removing the certificate-related builders.

***********************
Comparing Modes
***********************

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Aspect
     - Project File Mode (``-p``)
     - Single Participant Mode
   * - Participants
     - All from project.yml
     - One per command
   * - Project File
     - Required
     - Not needed
   * - Custom Builders
     - Preserved from project.yml
     - Uses minimal builder set
   * - Use Case
     - Batch generation for existing project
     - Quick single package creation

***********************
Next Steps
***********************

**Manual Workflow (without --cert-service):**

1. **Copy rootCA.pem** from your provisioned server workspace to each ``startup/`` folder
   (for TLS server verification)
2. **Distribute packages** along with signed certificates to each site
3. **Start the clients** - they will connect using the pre-signed certificates

**Auto-Scale Workflow (with --cert-service and --token):**

1. **Distribute packages** - enrollment information is already embedded
2. **Start the clients** - they will automatically enroll and obtain certificates:

   .. code-block:: shell

       cd hospital-1 && ./startup/start.sh

**Auto-Scale Workflow (with --cert-service only):**

1. **Distribute packages** along with enrollment tokens to each site
2. **Set the enrollment token** on each client machine:

   .. code-block:: shell

       export NVFLARE_ENROLLMENT_TOKEN="<token>"
       # Or place token in: startup/enrollment_token

3. **Start the clients** - they will automatically enroll and obtain certificates

***********************
See Also
***********************

- :ref:`token_command` - Generate enrollment tokens
- :ref:`provisioning` - Standard provisioning with certificates
- :ref:`enrollment_design_v2` - Design documentation for token-based enrollment
