.. _distributed_provisioning:

########################
Distributed Provisioning
########################

Distributed provisioning lets each participant create and keep its own private
key. The Project Admin signs certificate requests, but never receives any
participant private key.

Use this workflow when the federation is not using Confidential Computing or HE
startup-kit signing requirements, and when sites or users should be provisioned
independently.

The public workflow has three commands:

.. code-block:: shell

   nvflare cert request ...
   nvflare cert approve ...
   nvflare package ...

At a high level:

1. The requester creates a local request folder and request zip.
2. The requester sends only the request zip to the Project Admin.
3. The Project Admin approves the request zip and returns a signed zip.
4. The requester packages the signed zip with their local private key.

The resulting startup kit is used the same way as a centrally provisioned
startup kit.

****************************
Quick Start: Add a Site
****************************

This example provisions a client site named ``site-3`` in organization
``nvidia`` for project ``example_project``.

Project Admin initializes the project CA once:

.. code-block:: shell

   nvflare cert init --project example_project --org nvidia -o ./ca

Site Admin creates the request:

.. code-block:: shell

   nvflare cert request site site-3 --org nvidia --project example_project

This creates:

.. code-block:: text

   site-3/
     site-3.key
     site-3.csr
     site.yaml
     request.json
     site-3.request.zip

Send ``site-3/site-3.request.zip`` to the Project Admin. Do not send
``site-3.key``. The request zip does not include the private key.

Project Admin approves the request:

.. code-block:: shell

   nvflare cert approve site-3.request.zip --ca-dir ./ca

This creates ``site-3.signed.zip``. Return that signed zip to the Site Admin.

Site Admin packages the startup kit:

.. code-block:: shell

   nvflare package site-3.signed.zip -e grpc://server1:8002

The output goes under:

.. code-block:: text

   workspace/example_project/prod_00/site-3/

Start the site:

.. code-block:: shell

   cd workspace/example_project/prod_00/site-3
   ./startup/start.sh

********************************
Quick Start: Add an Org Admin
********************************

User certificates use ``kind=user`` and include a certificate role. Supported
roles are ``org-admin``, ``lead``, and ``member``.

Requester creates the org-admin request:

.. code-block:: shell

   nvflare cert request user org-admin org_admin@nvidia.com --org nvidia --project example_project

Send ``org_admin@nvidia.com/org_admin@nvidia.com.request.zip`` to the Project
Admin.

Project Admin approves it:

.. code-block:: shell

   nvflare cert approve org_admin@nvidia.com.request.zip --ca-dir ./ca

Requester packages the returned signed zip:

.. code-block:: shell

   nvflare package org_admin@nvidia.com.signed.zip -e grpc://server1:8002

The generated user startup kit contains ``startup/fl_admin.sh``.

.. code-block:: shell

   cd workspace/example_project/prod_00/org_admin@nvidia.com
   ./startup/fl_admin.sh

For a study lead certificate, use:

.. code-block:: shell

   nvflare cert request user lead lead@nvidia.com --org nvidia --project example_project
   nvflare cert approve lead@nvidia.com.request.zip --ca-dir ./ca
   nvflare package lead@nvidia.com.signed.zip -e grpc://server1:8002

.. note::

   Certificate roles are runtime certificate roles. Study-specific roles and
   study membership are assigned later by study commands, not by
   ``cert request``.

******************************
Quick Start: Add a Server
******************************

Server identity is also requested through the same workflow:

.. code-block:: shell

   nvflare cert request server server1 --org nvidia --project example_project
   nvflare cert approve server1.request.zip --ca-dir ./ca
   nvflare package server1.signed.zip -e grpc://server1:8002

The server certificate common name should match the hostname used in the
endpoint. For ``grpc://server1:8002``, the server identity should be
``server1``.

********************
Remote Transfer Flow
********************

When the Project Admin and requester are on different machines, the path names
in examples are usually flat because the zip has been copied into the current
directory.

Requester machine:

.. code-block:: shell

   nvflare cert request site site-3 --org nvidia --project example_project

Transfer this file to the Project Admin:

.. code-block:: text

   site-3/site-3.request.zip

Project Admin machine:

.. code-block:: shell

   nvflare cert approve site-3.request.zip --ca-dir ./ca

Transfer this file back to the requester:

.. code-block:: text

   site-3.signed.zip

Requester machine:

.. code-block:: shell

   nvflare package site-3.signed.zip -e grpc://server1:8002 --request-dir ./site-3

``--request-dir`` points to the local request folder that contains
``site-3.key`` and ``request.json``. If the signed zip is next to the request
folder, the command can usually find the request folder automatically.

*********************
Local Automation Flow
*********************

For local testing, use the same zip artifacts instead of switching to a
different command shape:

.. code-block:: shell

   nvflare cert init --project example_project --org nvidia -o ./ca
   nvflare cert request site site-3 --org nvidia --project example_project
   nvflare cert approve site-3/site-3.request.zip --ca-dir ./ca
   nvflare package site-3/site-3.signed.zip -e grpc://server1:8002

This is the same workflow as remote approval. The only difference is that the
zip files are not copied between machines.

********************
Using a Project File
********************

``cert request`` and ``cert approve`` do not need the project file. They only
deal with identity and certificate approval.

Use ``--project-file`` at package time when you need custom builders or
project-level packaging configuration:

.. code-block:: shell

   nvflare package site-3.signed.zip \
       -e grpc://server1:8002 \
       --project-file ./project.yml \
       --request-dir ./site-3

In signed-zip mode, the signed zip is the source of truth for participant
identity. The project file supplies custom builders and non-identity package
configuration. Only the signed participant is built, even if the project file
lists many participants.

If the signed participant appears in the project file, identity fields such as
name, organization, kind, certificate type, and certificate role must match the
signed zip. If they conflict, packaging fails.

If the signed participant is not found in the project file, packaging continues
with the signed identity and project builders, and prints a warning.

****************
Artifacts
****************

Request folder:

.. code-block:: text

   site-3/
     site-3.key          # private key, stays local
     site-3.csr          # CSR
     site.yaml           # request identity metadata
     request.json        # request metadata and hashes
     site-3.request.zip  # sent to Project Admin

Request zip:

.. code-block:: text

   request.json
   site.yaml
   site-3.csr

The request zip must not contain ``*.key``.

Signed zip:

.. code-block:: text

   signed.json
   site.yaml
   site-3.crt
   rootCA.pem

The signed zip must not contain ``*.key``. It is not a startup kit. It is an
approval response used by ``nvflare package``.

``site.yaml`` appears in both the request zip and the signed zip. It records
the requested identity in a small human-readable form.

For a site:

.. code-block:: yaml

   name: site-3
   org: nvidia
   type: client
   project: example_project
   kind: site

For a server:

.. code-block:: yaml

   name: server1
   org: nvidia
   type: server
   project: example_project
   kind: server

For a user:

.. code-block:: yaml

   name: org_admin@nvidia.com
   org: nvidia
   type: org_admin
   project: example_project
   kind: user
   cert_role: org-admin

Audit state:

- ``cert request`` records request audit metadata under
  ``~/.nvflare/cert_requests``.
- ``cert approve`` records approval audit metadata under
  ``~/.nvflare/cert_approves``.

Audit records store paths, metadata, and hashes. They do not copy private keys.

****************
Roles
****************

``nvflare cert request`` starts with an identity kind:

.. code-block:: shell

   nvflare cert request site <site-name> --org <org> --project <project>
   nvflare cert request server <server-name> --org <org> --project <project>
   nvflare cert request user <cert-role> <email> --org <org> --project <project>

Kinds:

.. list-table::
   :header-rows: 1

   * - Kind
     - Startup kit
     - Example
   * - ``site``
     - Client site kit with ``startup/start.sh``
     - ``nvflare cert request site site-3 --org nvidia --project example_project``
   * - ``server``
     - Server kit with ``startup/start.sh``
     - ``nvflare cert request server server1 --org nvidia --project example_project``
   * - ``user``
     - Admin API user kit with ``startup/fl_admin.sh``
     - ``nvflare cert request user lead lead@nvidia.com --org nvidia --project example_project``

User certificate roles:

.. list-table::
   :header-rows: 1

   * - Role
     - Intended use
   * - ``org-admin``
     - Organization administrator
   * - ``lead``
     - Lead researcher or job submitter
   * - ``member``
     - Read-only or limited user

Project names must be path-safe identifiers matching
``[A-Za-z0-9][A-Za-z0-9._-]*``. During approval, the request project must
match the CA metadata and the root CA certificate subject.

****************
Command Summary
****************

Project Admin:

.. code-block:: shell

   nvflare cert init --project example_project --org nvidia -o ./ca
   nvflare cert approve site-3.request.zip --ca-dir ./ca

Requester:

.. code-block:: shell

   nvflare cert request site site-3 --org nvidia --project example_project
   nvflare package site-3.signed.zip -e grpc://server1:8002

With explicit locations:

.. code-block:: shell

   nvflare cert request site site-3 --org nvidia --project example_project --out ./requests/site-3
   nvflare cert approve ./requests/site-3/site-3.request.zip --ca-dir ./ca --out ./signed/site-3.signed.zip
   nvflare package ./signed/site-3.signed.zip \
       -e grpc://server1:8002 \
       --request-dir ./requests/site-3 \
       -w ./workspace

****************
Notes
****************

- The private key stays in the request folder and is never sent to the Project
  Admin.
- The server endpoint belongs to ``nvflare package``, not ``cert request``.
  Endpoint changes do not require a new certificate. Re-run ``nvflare package``
  with a new ``-e`` value.
- Startup kits generated from signed zips are compatible with the normal
  NVFlare runtime.
- Standard distributed provisioning does not generate ``signature.json``.
  Trust is anchored in the signed participant certificate and ``rootCA.pem``.
