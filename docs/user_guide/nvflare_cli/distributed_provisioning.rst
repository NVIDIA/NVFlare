.. _distributed_provisioning:

##########################
Distributed Provisioning
##########################

Distributed (manual) provisioning is an alternative to centralized provisioning
(``nvflare provision``) designed for deployments where a Project Admin cannot or
should not generate private keys on behalf of each site.

In the standard ``nvflare provision`` workflow the Project Admin runs a single
command and distributes pre-packaged startup kits — including each site's private
key — to participants. This requires participants to trust the Project Admin with
their private key during transit and at rest.

Distributed provisioning uses an **inverted-trust model**:

- Each site generates its own RSA private key **locally**. The key never leaves the machine.
- The site sends only a **Certificate Signing Request (CSR)** to the Project Admin.
- The Project Admin signs the CSR and returns a certificate.
- Each site assembles its own startup kit from its local key, the signed certificate, and the root CA.

The resulting startup kits are structurally identical to those produced by
``nvflare provision`` and are fully compatible with all NVFlare runtime components.

.. note::

   Distributed provisioning uses mTLS as the sole trust anchor. No ``signature.json``
   is generated. The ``require_signed_jobs`` policy is always enabled on the server.

*****
Roles
*****

+-------------------+----------------------------------------------------------+
| Role              | Responsibility                                           |
+===================+==========================================================+
| **Project Admin** | Runs ``cert init`` (once). Signs CSRs from each site.    |
|                   | Distributes signed certificates and ``rootCA.pem``.      |
+-------------------+----------------------------------------------------------+
| **Site Admin**    | Runs ``cert csr`` to generate a local key + CSR.         |
|                   | Sends CSR to Project Admin out-of-band (email, file      |
|                   | share, etc.). Receives cert + ``rootCA.pem``.            |
|                   | Runs ``package`` to assemble the startup kit.            |
+-------------------+----------------------------------------------------------+

The ``-t`` / ``--type`` argument identifies either a **site** (FL process identity) or a
**user** (human connecting via the admin API):

**Site types** — receive ``start.sh``; no role embedded in cert ``UNSTRUCTURED_NAME``:

+---------------+---------------------------------------------------------------+
| Type          | Description                                                   |
+===============+===============================================================+
| ``server``    | FL server process identity (mTLS server endpoint)             |
+---------------+---------------------------------------------------------------+
| ``client``    | FL client (data site) process identity (mTLS client)          |
+---------------+---------------------------------------------------------------+

**User roles** — receive ``fl_admin.sh``; role embedded in cert ``UNSTRUCTURED_NAME``
and enforced by ``local/authorization.json.default`` at each site:

+---------------+---------------------------------------------------+-----------------------------+
| Role          | Description                                       | Default ``submit_job``      |
+===============+===================================================+=============================+
| ``lead``      | Lead researcher / primary job submitter.          | Allowed (any site)          |
|               | Can submit, clone, manage own jobs; operate and   |                             |
|               | deploy custom code (byoc) on own site.            |                             |
+---------------+---------------------------------------------------+-----------------------------+
| ``org_admin`` | Organization administrator. Can manage and        | **Not allowed**             |
|               | operate own site, view all jobs, but cannot       |                             |
|               | submit or clone jobs.                             |                             |
+---------------+---------------------------------------------------+-----------------------------+
| ``member``    | Read-only observer. Can view jobs and status;     | **Not allowed**             |
|               | no submit, operate, or byoc permissions.          |                             |
+---------------+---------------------------------------------------+-----------------------------+

Sites may customize ``local/authorization.json.default`` to tighten or loosen the
default policy. ``lead`` is the intended role for users who run FL experiments.
``project_admin`` is not a ``-t`` choice — the Project Admin self-provisions via
``nvflare cert init``.

*****
Steps
*****

Step 1 — Project Admin: Initialize the Root CA (once per federation)
=====================================================================

The Project Admin bootstraps the root certificate authority. This is a **one-time
operation** per federation.

.. code-block:: bash

   nvflare cert init --project <project-name> -o ./ca

Example:

.. code-block:: bash

   nvflare cert init --project my-fl-project -o ./ca

This produces:

- ``./ca/rootCA.pem`` — root CA certificate (distribute to all sites)
- ``./ca/rootCA.key`` — root CA private key (keep secret, never distribute)
- ``./ca/ca.json`` — audit metadata (serial counter)

.. attention::

   ``rootCA.key`` must be kept confidential. Anyone with access to it can sign
   certificates for any participant identity. Store it on a secure, air-gapped
   machine or in a hardware security module (HSM).

Step 2 — Site Admin: Generate a Local Key and CSR
==================================================

Each participant (server, each client, each admin user) runs this step on their
own machine. The optional ``-t`` flag embeds the proposed certificate type in the
CSR as a hint for the Project Admin.

.. code-block:: bash

   nvflare cert csr -n <participant-name> -t <type> -o ./csr

Example for a client site named ``hospital-1``:

.. code-block:: bash

   nvflare cert csr -n hospital-1 -t client -o ./csr

This produces:

- ``./csr/hospital-1.key`` — private key (permissions: 0600, never leaves this machine)
- ``./csr/hospital-1.csr`` — certificate signing request (send to Project Admin)

Example for the FL server named ``fl-server``:

.. code-block:: bash

   nvflare cert csr -n fl-server -t server -o ./csr

.. note::

   The ``-t`` flag in ``cert csr`` is a **proposal** only. The Project Admin sets
   the final type authoritatively when signing. The org admin should generate CSRs
   on behalf of participants to ensure the correct type is requested.

Step 3 — Site Admin: Send CSR to Project Admin
===============================================

Deliver ``<participant-name>.csr`` to the Project Admin through any out-of-band
channel (email, secure file transfer, shared storage, etc.). The private key file
stays on the site's machine and is never shared.

Step 4 — Project Admin: Sign the CSR
======================================

For each received CSR, the Project Admin runs:

.. code-block:: bash

   nvflare cert sign -r <participant>.csr -c ./ca -o ./signed/<participant>

The certificate type is read from the CSR's embedded proposal. The Project Admin
may override it with ``-t <type>``:

.. code-block:: bash

   nvflare cert sign -r <participant>.csr -t <type> -c ./ca -o ./signed/<participant>

The ``-t`` argument **overrides** whatever type was proposed in the CSR, ensuring
the Project Admin has final authority over certificate types.

Example — signing the ``hospital-1`` client CSR (type embedded in CSR):

.. code-block:: bash

   nvflare cert sign -r hospital-1.csr -c ./ca -o ./signed/hospital-1

Example — signing the ``fl-server`` server CSR with explicit type override:

.. code-block:: bash

   nvflare cert sign -r fl-server.csr -t server -c ./ca -o ./signed/fl-server

Each output directory contains:

- ``<name>.crt`` (e.g. ``hospital-1.crt``, ``fl-server.crt``) — signed certificate
- ``rootCA.pem`` — copy of the root CA certificate

Step 5 — Project Admin: Distribute Signed Certificates
=======================================================

Send each site their signed certificate and ``rootCA.pem``:

- ``hospital-1/hospital-1.crt`` + ``rootCA.pem`` → send to ``hospital-1``
- ``fl-server/fl-server.crt`` + ``rootCA.pem`` → send to the server site

No private keys are exchanged at this step.

Step 6 — Site Admin: Assemble the Startup Kit
==============================================

Each site runs ``nvflare package`` to assemble a startup kit from:

- The local private key (generated in Step 2)
- The signed certificate (received in Step 5)
- ``rootCA.pem`` (received in Step 5)

**Using** ``--project-file``:

For users already familiar with ``nvflare provision`` project.yaml, those who need
custom builders, or who prefer to describe all participants in a single file.
Place all received certs and ``rootCA.pem`` in one directory (named by participant CN),
then run:

.. code-block:: bash

   nvflare package -e grpc://fl-server:8002 -p ./site.yaml --dir ./certs

A minimal ``site.yaml``:

.. code-block:: yaml

   api_version: 3
   name: my-project
   description: ""
   participants:
     - name: hospital-1
       type: client
       org: hospital
     - name: alice@hospital.com
       type: admin
       org: hospital
       role: lead

**Using** ``--dir``:

Place the key, certificate, and ``rootCA.pem`` in the same directory. The participant
name is auto-detected from the ``.key`` filename, and the kit type is derived
automatically from the certificate's embedded type:

.. code-block:: bash

   nvflare package -e grpc://fl-server:8002 --dir ./hospital-1-kit

The ``-e`` / ``--endpoint`` argument sets the FL server address using one of the
supported schemes: ``grpc://``, ``tcp://``, or ``http://``. The server identity
used for mTLS validation is derived from the hostname in the endpoint.

**Explicit mode** (when files are in different locations):

.. code-block:: bash

   nvflare package \
     -n hospital-1 \
     -e grpc://fl-server:8002 \
     --cert ./signed/hospital-1/hospital-1.crt \
     --key  ./csr/hospital-1.key \
     --rootca ./signed/hospital-1/rootCA.pem

Step 7 — Start the Federation
==============================

Each site starts NVFlare using the assembled startup kit, exactly as with a
centrally provisioned kit:

.. code-block:: bash

   cd <startup-kit-dir>
   ./startup/start.sh

Admin users connect via:

.. code-block:: bash

   cd <admin-startup-kit-dir>
   ./startup/fl_admin.sh

*************************************
Complete Example: Two-Site Federation
*************************************

This example sets up a federation with one server (``fl-server``) and one client
(``hospital-1``).

**Project Admin machine:**

.. code-block:: bash

   # 1. Initialize root CA
   nvflare cert init --project my-project -o ./ca

   # 4a. Sign server CSR (type embedded in CSR; override with -t if needed)
   nvflare cert sign -r fl-server.csr -c ./ca -o ./signed/fl-server

   # 4b. Sign client CSR
   nvflare cert sign -r hospital-1.csr -c ./ca -o ./signed/hospital-1

**Server site (fl-server):**

.. code-block:: bash

   # 2. Generate key + CSR (propose type 'server')
   nvflare cert csr -n fl-server -t server -o ./csr

   # 3. Send ./csr/fl-server.csr to Project Admin

   # 6. Copy signed cert + rootCA.pem into ./csr/ (alongside fl-server.key), then:
   nvflare package -e grpc://fl-server:8002 --dir ./csr
   # Kit type is derived from the signed cert; output to workspace/project/prod_00/fl-server/

   # 7. Start
   cd workspace/project/prod_00/fl-server && ./startup/start.sh

**Client site (hospital-1):**

.. code-block:: bash

   # 2. Generate key + CSR (propose type 'client')
   nvflare cert csr -n hospital-1 -t client -o ./csr

   # 3. Send ./csr/hospital-1.csr to Project Admin

   # 6. Copy signed cert + rootCA.pem into ./csr/ (alongside hospital-1.key), then:
   nvflare package -e grpc://fl-server:8002 --dir ./csr
   # Kit type is derived from the signed cert; output to workspace/project/prod_00/hospital-1/

   # 7. Start
   cd workspace/project/prod_00/hospital-1 && ./startup/start.sh

*********************
CLI Reference Summary
*********************

``nvflare cert init``
=====================

Initialize the root CA (Project Admin, once per federation).

+------------------+--------------------------------------------------+----------+
| Argument         | Description                                      | Required |
+==================+==================================================+==========+
| ``--project``         | Project name (used as CA subject CN)        | Yes      |
+------------------+--------------------------------------------------+----------+
| ``-o`` / ``--output-dir`` | Directory to write CA files             | Yes      |
+------------------+--------------------------------------------------+----------+
| ``--org``        | Organization name for the CA certificate         | No       |
+------------------+--------------------------------------------------+----------+
| ``--force``      | Overwrite existing CA (backs up old files)       | No       |
+------------------+--------------------------------------------------+----------+

``nvflare cert csr``
====================

Generate a local private key and CSR (Site Admin).

+------------------+--------------------------------------------------+----------+
| Argument         | Description                                      | Required |
+==================+==================================================+==========+
| ``-n`` / ``--name``   | Participant name (becomes cert CN)          | Yes      |
+------------------+--------------------------------------------------+----------+
| ``-o`` / ``--output-dir`` | Directory for key and CSR files         | Yes      |
+------------------+--------------------------------------------------+----------+
| ``-t`` / ``--type``   | Proposed certificate type. Embedded in      | No       |
|                  | the CSR as a hint for the Project Admin.         |          |
|                  | The Project Admin may override with              |          |
|                  | ``cert sign -t <type>``.                         |          |
+------------------+--------------------------------------------------+----------+
| ``--org``        | Organization name                                | No       |
+------------------+--------------------------------------------------+----------+

``nvflare cert sign``
=====================

Sign a CSR with the root CA (Project Admin).

+------------------+--------------------------------------------------+----------+
| Argument         | Description                                      | Required |
+==================+==================================================+==========+
| ``-r`` / ``--csr``    | Path to the CSR file to sign                | Yes      |
+------------------+--------------------------------------------------+----------+
| ``-c`` / ``--ca-dir`` | Directory containing ``rootCA.key`` and     | Yes      |
|                  | ``rootCA.pem``                                   |          |
+------------------+--------------------------------------------------+----------+
| ``-o`` / ``--output-dir`` | Directory for signed cert and rootCA.pem | Yes     |
+------------------+--------------------------------------------------+----------+
| ``-t`` / ``--type``   | Certificate type to issue. Overrides the    | No       |
|                  | type proposed in the CSR. Required when the      |          |
|                  | CSR has no embedded type.                        |          |
+------------------+--------------------------------------------------+----------+
| ``--force``      | Overwrite existing certificate                   | No       |
+------------------+--------------------------------------------------+----------+

``nvflare package``
===================

Assemble a startup kit (Site Admin).

+------------------+--------------------------------------------------+----------+
| Argument         | Description                                      | Required |
+==================+==================================================+==========+
| ``-e`` / ``--endpoint`` | Server endpoint URI (``grpc://host:port``, | Yes      |
|                  | ``tcp://host:port``, or ``http://host:port``)    |          |
+------------------+--------------------------------------------------+----------+
| ``-t`` / ``--type``   | Kit type: ``client``, ``server``,           | No       |
|                  | ``org_admin``, ``lead``, ``member``.             |          |
|                  | In single mode, derived from the signed cert's   |          |
|                  | embedded type. Explicit ``-t`` overrides.        |          |
|                  | In yaml mode (``-p``), acts as a type filter.    |          |
+------------------+--------------------------------------------------+----------+
| ``-p`` / ``--project-file`` | Site-scoped project YAML listing all  | No       |
|                  | participants. When given, builds all matching    |          |
|                  | participants in a single ``prod_NN``. Mutually   |          |
|                  | exclusive with ``-n`` and ``--cert``/``--key``/  |          |
|                  | ``--rootca``. Requires ``--dir``.                |          |
+------------------+--------------------------------------------------+----------+
| ``--dir``        | Directory containing key, cert, and rootCA.pem.  | One of   |
|                  | Name is auto-detected from ``*.key`` filename    | ``--dir``|
|                  | (single mode) or cert files named by participant | or       |
|                  | CN (yaml mode).                                  | ``-p``   |
+------------------+--------------------------------------------------+----------+
| ``-n`` / ``--name``   | Participant name. Required when using       | No†      |
|                  | ``--cert`` / ``--key`` / ``--rootca``            |          |
+------------------+--------------------------------------------------+----------+
| ``--cert``       | Path to signed certificate                       | No†      |
+------------------+--------------------------------------------------+----------+
| ``--key``        | Path to private key                              | No†      |
+------------------+--------------------------------------------------+----------+
| ``--rootca``     | Path to ``rootCA.pem``                           | No†      |
+------------------+--------------------------------------------------+----------+
| ``-w`` / ``--workspace`` | Workspace root. Output goes to           | No       |
|                  | ``<workspace>/<project>/prod_NN/<name>/``.       |          |
|                  | Default: ``workspace``                           |          |
+------------------+--------------------------------------------------+----------+
| ``--project-name`` | Project name used in output path and in        | No       |
|                  | fed_server.json / fed_admin.json for             |          |
|                  | challenge-response auth. Default: ``project``    |          |
+------------------+--------------------------------------------------+----------+
| ``--admin-port`` | Server admin port. Default: same as service port | No       |
|                  | (single-port TLS multiplexing).                  |          |
+------------------+--------------------------------------------------+----------+
| ``--force``      | Allow re-packaging when this participant name    | No       |
|                  | appears in the most recent ``prod_NN``           |          |
|                  | directory (a new ``prod_NN`` is created).        |          |
+------------------+--------------------------------------------------+----------+

† ``-n``, ``--cert``, ``--key``, ``--rootca`` are used together as an alternative to ``--dir``
in single-participant mode. Mutually exclusive with ``-p`` / ``--project-file``.

All commands support ``--output json`` for machine-readable output and ``--schema``
to print the JSON schema for the command's arguments.

*****
Notes
*****

- The ``--dir`` mode for ``nvflare package`` locates the ``.key`` file automatically.
  Place the signed ``.crt`` and ``rootCA.pem`` in the same directory before running.
- Startup kits produced by ``nvflare package`` are structurally identical to those
  produced by ``nvflare provision`` and work with all NVFlare runtime components.
- The server identity for mTLS validation is always the hostname from ``--endpoint``.
  Ensure that the hostname in the endpoint matches the CN in the server certificate
  (i.e. the ``-n`` name used when running ``nvflare cert csr`` for the server).
