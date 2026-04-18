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

   Distributed provisioning uses project-issued certificates and ``rootCA.pem`` as
   the trust anchor. No ``signature.json`` is generated. In distributed
   provisioning, ``require_signed_jobs`` defaults to enabled on the server because
   ``rootCA.pem`` is present in the startup kit. Operators can explicitly
   override it in ``fed_server.json``, but disabling signed-job enforcement is
   strongly discouraged.

.. note::

   Study definitions are not represented in per-site ``site.yaml`` inputs for
   distributed provisioning. In the current design, studies are defined in a
   centralized ``project.yml`` and materialized into the server-side
   ``study_registry.json`` during centralized provisioning.

****************************************
Centralized vs. Distributed at a Glance
****************************************

+-----------------------------+--------------------------------------+--------------------------------------+
| Aspect                      | Centralized (``nvflare provision``)  | Distributed (manual workflow)        |
+=============================+======================================+======================================+
| Private key custody         | Admin generates/distributes keys     | Site generates key locally           |
+-----------------------------+--------------------------------------+--------------------------------------+
| Data distributed to site    | Full startup kit                     | Signed cert + ``rootCA.pem``         |
+-----------------------------+--------------------------------------+--------------------------------------+
| Data sent from site         | Nothing                              | CSR (public key only)                |
+-----------------------------+--------------------------------------+--------------------------------------+
| Project Admin workflow      | One command provisions all sites     | Sign one CSR per participant         |
+-----------------------------+--------------------------------------+--------------------------------------+
| Site Admin workflow         | Unpack and run                       | CSR -> sign -> package -> run        |
+-----------------------------+--------------------------------------+--------------------------------------+
| Participant onboarding      | Usually prepared up front            | Join independently, on demand        |
+-----------------------------+--------------------------------------+--------------------------------------+
| Adding a new site           | Dynamic provisioning with root CA    | Same flow; no impact on existing     |
+-----------------------------+--------------------------------------+--------------------------------------+
| Trust in Project Admin      | Must trust admin with private keys   | Admin never sees private keys        |
+-----------------------------+--------------------------------------+--------------------------------------+
| Support scope               | Supports CC and HE                   | Targets non-CC and non-HE            |
+-----------------------------+--------------------------------------+--------------------------------------+

*****
Roles
*****

+------------------+------------------------------------------------------------------------------------------+
| Role             | Responsibility                                                                           |
+==================+==========================================================================================+
| Project Admin    | Runs cert init once, signs incoming CSRs, and returns signed certificates and rootCA.   |
+------------------+------------------------------------------------------------------------------------------+
| Site Admin       | Runs cert csr locally, sends CSR out-of-band, receives cert and rootCA, then packages.  |
+------------------+------------------------------------------------------------------------------------------+

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

**Default authorization policy** enforced at runtime against
``local/authorization.json.default`` on each site:

+------------------------------------------+------------------+------------------+------------------+
| Permission                               | ``lead``         | ``org_admin``    | ``member``       |
+==========================================+==================+==================+==================+
| ``submit_job``                           | Any site         | —                | —                |
+------------------------------------------+------------------+------------------+------------------+
| ``clone_job``                            | Own jobs         | —                | —                |
+------------------------------------------+------------------+------------------+------------------+
| ``manage_job`` (abort / delete)          | Own jobs         | Jobs from own org| —                |
+------------------------------------------+------------------+------------------+------------------+
| ``download_job``                         | Own jobs         | Jobs from own org| —                |
+------------------------------------------+------------------+------------------+------------------+
| ``view``                                 | Any              | Any              | Any              |
+------------------------------------------+------------------+------------------+------------------+
| ``operate`` (start / stop site)          | Own site         | Own site         | —                |
+------------------------------------------+------------------+------------------+------------------+
| ``byoc`` (custom code)                   | Any              | —                | —                |
+------------------------------------------+------------------+------------------+------------------+

*****
Steps
*****

+------+-------------------+-------------------------------------------------------------------+
| Step | Who               | Action                                                            |
+======+===================+===================================================================+
| 1    | Project Admin     | ``nvflare cert init --project my-project -o ./ca``                |
|      |                   | *(one-time per federation)*                                       |
+------+-------------------+-------------------------------------------------------------------+
| 2    | Site Admin        | ``nvflare cert csr -n hospital-1 -t client -o ./csr``             |
+------+-------------------+-------------------------------------------------------------------+
| 3    | Site Admin        | Send ``hospital-1.csr`` to Project Admin (email, file share, etc.)|
+------+-------------------+-------------------------------------------------------------------+
| 4    | Project Admin     | ``nvflare cert sign -r hospital-1.csr -c ./ca -o ./signed --accept-csr-role`` |
+------+-------------------+-------------------------------------------------------------------+
| 5    | Project Admin     | Return ``hospital-1.crt`` + ``rootCA.pem`` to site admin          |
+------+-------------------+-------------------------------------------------------------------+
| 6    | Site Admin        | ``nvflare package -e grpc://fl-server:8002 --dir ./csr``          |
|      |                   | *(kit type derived from signed cert)*                             |
+------+-------------------+-------------------------------------------------------------------+
| 7    | Site Admin        | ``cd hospital-1 && ./startup/start.sh``                           |
+------+-------------------+-------------------------------------------------------------------+

Step 1 is done once per federation. Each new participant repeats steps 2–7 independently.

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

   The ``-t`` flag in ``cert csr`` is required so the CSR always carries an
   explicit site-admin-proposed type. The Project Admin must then either accept
   that proposal explicitly or override it explicitly at signing time.

Step 3 — Site Admin: Send CSR to Project Admin
===============================================

Deliver ``<participant-name>.csr`` to the Project Admin through any out-of-band
channel (email, secure file transfer, shared storage, etc.). The private key file
stays on the site's machine and is never shared.

Step 4 — Project Admin: Sign the CSR
======================================

For each received CSR, the Project Admin runs:

.. code-block:: bash

   nvflare cert sign -r <participant>.csr -c ./ca -o ./signed/<participant> --accept-csr-role

If the Project Admin does not want to accept the role proposed by the site
admin, they may override it with ``-t <type>``:

.. code-block:: bash

   nvflare cert sign -r <participant>.csr -t <type> -c ./ca -o ./signed/<participant>

The ``--accept-csr-role`` argument means the Project Admin is explicitly trusting
the role proposed by the site admin in the CSR. The ``-t`` argument
**overrides** whatever type was proposed in the CSR.

Example — signing the ``hospital-1`` client CSR while accepting the site admin's proposed type:

.. code-block:: bash

   nvflare cert sign -r hospital-1.csr -c ./ca -o ./signed/hospital-1 --accept-csr-role

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

.. note::

   ``nvflare package`` always provides its own ``WorkspaceBuilder`` and
   ``StaticFileBuilder`` (with the scheme derived from ``--endpoint``).
   If your YAML ``builders:`` section lists either of these, those entries —
   including any custom args such as ``config_folder`` — are silently ignored
   and a warning is emitted. Custom third-party builders are passed through unchanged.

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

   # 4a. Sign server CSR (accept the site-admin-proposed type)
   nvflare cert sign -r fl-server.csr -c ./ca -o ./signed/fl-server --accept-csr-role

   # 4b. Sign client CSR
   nvflare cert sign -r hospital-1.csr -c ./ca -o ./signed/hospital-1 --accept-csr-role

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
|                  | the CSR as the site-admin-proposed type.         |          |
|                  | The Project Admin must either accept it with     |          |
|                  | ``--accept-csr-role`` or override with ``-t``.   |          |
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
| ``--accept-csr-role`` | Accept the type embedded in the CSR instead | No       |
|                  | of overriding it with ``-t`` / ``--type``.      |          |
+------------------+--------------------------------------------------+----------+
| ``--valid-days`` | Certificate validity in days. Default: 1095     | No       |
|                  | (3 years).                                      |          |
+------------------+--------------------------------------------------+----------+
| ``--force``      | Overwrite existing certificate                   | No       |
+------------------+--------------------------------------------------+----------+

``nvflare package``
===================

Assemble a startup kit (Site Admin).

+--------------------------+--------------------------------------------------------------------------+----------+
| Argument                 | Description                                                              | Required |
+==========================+==========================================================================+==========+
| -e / --endpoint          | Server endpoint URI (grpc/tcp/http).                                    | Yes      |
+--------------------------+--------------------------------------------------------------------------+----------+
| -t / --type              | Optional type filter for yaml mode only. In single mode the kit type    | No       |
|                          | is always derived from the signed cert's embedded type.                 |          |
+--------------------------+--------------------------------------------------------------------------+----------+
| -p / --project-file      | Site-scoped project YAML for multi-participant mode; requires --dir.    | No       |
+--------------------------+--------------------------------------------------------------------------+----------+
| --dir                    | Directory containing key/cert/rootCA files.                             | Mode     |
+--------------------------+--------------------------------------------------------------------------+----------+
| -n / --name              | Participant name; required in explicit single mode.                     | No*      |
+--------------------------+--------------------------------------------------------------------------+----------+
| --cert                   | Path to signed certificate.                                              | No*      |
+--------------------------+--------------------------------------------------------------------------+----------+
| --key                    | Path to private key.                                                     | No*      |
+--------------------------+--------------------------------------------------------------------------+----------+
| --rootca                 | Path to rootCA.pem.                                                      | No*      |
+--------------------------+--------------------------------------------------------------------------+----------+
| -w / --workspace         | Workspace root; output under <workspace>/<project>/prod_NN/<name>/.     | No       |
+--------------------------+--------------------------------------------------------------------------+----------+
| --project-name           | Project name used in output path and in fed_server/fed_admin configs.   | No       |
+--------------------------+--------------------------------------------------------------------------+----------+
| --admin-port             | Admin port; default is the endpoint port (learner port).                | No       |
+--------------------------+--------------------------------------------------------------------------+----------+
| --force                  | Allow re-packaging when participant exists in latest prod_NN.           | No       |
+--------------------------+--------------------------------------------------------------------------+----------+

* ``-n``, ``--cert``, ``--key``, and ``--rootca`` are used together in explicit
  single-participant mode; mutually exclusive with ``-p`` / ``--project-file``.

All commands support the global ``--out-format json`` flag for machine-readable
output and ``--schema``
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
