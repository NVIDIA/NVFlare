.. _distributed_provisioning:

########################
Distributed Provisioning
########################

Distributed provisioning lets each participant create and keep its own private
key. The Project Admin signs certificate requests, but never receives any
participant private key.

Use this workflow when sites, servers, or users should be provisioned
independently. The public workflow uses three commands:

.. code-block:: shell

   nvflare cert request --participant <participant.yaml>
   nvflare cert approve <request.zip> --ca-dir <ca-dir> --profile <project_profile.yaml>
   nvflare package <signed.zip> --confirm-rootca

At a high level:

1. The Project Admin creates ``project_profile.yaml`` and initializes the
   project CA from that explicit profile file.
2. The server admin decides the server host name and ports and shares them with
   the Project Admin.
3. The Project Admin records the approved server endpoint in
   ``project_profile.yaml`` before approving requests.
4. The requester creates a participant definition file using the project name.
5. The requester runs ``nvflare cert request`` and sends only the generated
   request zip to the Project Admin.
6. The Project Admin approves the request zip and returns the signed zip, which
   includes the approved server endpoint.
7. The Project Admin shares ``rootca_fingerprint_sha256`` through a trusted
   out-of-band channel.
8. The requester packages the signed zip on the machine that owns the local
   private key.

The resulting startup kit is used the same way as a centrally provisioned
startup kit.

*******************************************
Before You Start: Record Connection Details
*******************************************

Before approvals begin, two pieces of information must be in place:

**1. Project name**

The Project Admin chooses the project name when writing
``project_profile.yaml``. Every participant definition file must use this exact
name in its ``name:`` field.

**2. Server host and ports**

The server admin decides the FL server's host name (or DNS name), the
``fed_learn_port``, and the ``admin_port`` before any provisioning begins. The
server admin communicates these values to the Project Admin, who records them
in ``project_profile.yaml``. ``nvflare cert approve`` signs this endpoint into
the signed zip. Client and user participant definition files do not contain
local ``server:`` blocks.

The only value shared out-of-band for requester verification is
``rootca_fingerprint_sha256`` after approval.

*********************
Project Profile
*********************

The Project Admin creates ``project_profile.yaml`` once per federation. This is
a lightweight project profile, not the full centralized provisioning
``project.yaml``.

.. code-block:: yaml

   name: hospital_federation
   scheme: grpc
   connection_security: tls
   server:
     host: server1.hospital-central.org
     fed_learn_port: 8002
     admin_port: 8003

Fields:

- ``name``: project name. Requests must match this value.
- ``scheme``: FLARE communication driver, such as ``grpc``, ``tcp``, or
  ``http``.
- ``connection_security``: project default connection security, such as
  ``tls``, ``mtls``, or ``clear``.
- ``server``: Project Admin-approved FL server endpoint, provided by the server
  admin before approvals.

The Project Admin keeps this file local. During approval, ``scheme``, the
default ``connection_security``, and the ``server`` endpoint are copied into the
signed zip metadata so participants do not need the profile file.

********************************
Participant Definition Files
********************************

Each requester creates a participant definition file for one identity. The file
uses the same ``participants`` structure as centralized ``project.yaml``, but it
contains only the local participant.

Client site example:

.. code-block:: yaml

   name: hospital_federation
   description: Site A - Hospital Alpha

   participants:
     - name: hospital-a
       type: client
       org: hospital_alpha

User example:

.. code-block:: yaml

   name: hospital_federation

   participants:
     - name: alice@hospital-alpha.org
       type: admin
       org: hospital_alpha
       role: lead

Server example:

.. code-block:: yaml

   name: hospital_federation
   description: Central FL server for hospital network

   participants:
     - name: server1.hospital-central.org
       type: server
       org: hospital_central
       host_names:
         - 10.0.1.50
         - fl-server.internal
       connection_security: mtls

For clients and users, server endpoint fields are intentionally absent.
``nvflare package`` gets the approved endpoint from the signed zip created by
``nvflare cert approve``.

For servers, ``connection_security`` is an optional local override. It is used
only when the server startup kit is packaged from the local request folder. It
is not approved by the Project Admin and is not distributed as federation
policy. If the server definition does not set it, package uses the project
default from the signed zip.

User roles are certificate roles. Supported values are ``org_admin``, ``lead``,
and ``member``. Study-specific roles and study membership are assigned later by
study commands.

****************************
Quick Start: Add a Site
****************************

This example provisions a client site named ``hospital-a`` for project
``hospital_federation``.

Project Admin creates ``project_profile.yaml``:

.. code-block:: yaml

   name: hospital_federation
   scheme: grpc
   connection_security: tls
   server:
     host: server1.hospital-central.org
     fed_learn_port: 8002
     admin_port: 8003

Project Admin initializes the project CA once:

.. code-block:: shell

   nvflare cert init --profile project_profile.yaml -o ./ca

.. note::

   The server host and ports are chosen by the server admin. The Project Admin
   records them in ``project_profile.yaml`` before approving requests. They are
   signed into the approval zip and do not need to be copied into client or user
   participant definition files.

Site Admin creates ``hospital-a.yaml`` with the project name and participant
identity:

.. code-block:: yaml

   name: hospital_federation

   participants:
     - name: hospital-a
       type: client
       org: hospital_alpha

Site Admin creates the request:

.. code-block:: shell

   nvflare cert request --participant hospital-a.yaml

This creates:

.. code-block:: text

   hospital-a/
     hospital-a.key
     hospital-a.csr
     site.yaml
     request.json
     hospital-a.request.zip

Send ``hospital-a/hospital-a.request.zip`` to the Project Admin. Do not send
``hospital-a.key``. The request zip does not include the private key.

Project Admin approves the request:

.. code-block:: shell

   nvflare cert approve hospital-a.request.zip --ca-dir ./ca --profile project_profile.yaml

This creates ``hospital-a.signed.zip`` and prints
``rootca_fingerprint_sha256``. The signed zip already includes ``rootCA.pem``;
return the signed zip to the Site Admin and share only the fingerprint through
a trusted out-of-band channel.

Site Admin packages the startup kit:

.. code-block:: shell

   nvflare package hospital-a.signed.zip --confirm-rootca

The output goes under:

.. code-block:: text

   workspace/hospital_federation/prod_00/hospital-a/

Start the site:

.. code-block:: shell

   cd workspace/hospital_federation/prod_00/hospital-a
   ./startup/start.sh

********************************
Quick Start: Add a User
********************************

Requester creates ``alice.yaml`` with the project name and user identity:

.. code-block:: yaml

   name: hospital_federation

   participants:
     - name: alice@hospital-alpha.org
       type: admin
       org: hospital_alpha
       role: lead

Requester creates the user request:

.. code-block:: shell

   nvflare cert request --participant alice.yaml

Send ``alice@hospital-alpha.org/alice@hospital-alpha.org.request.zip`` to the
Project Admin.

Project Admin approves it:

.. code-block:: shell

   nvflare cert approve alice@hospital-alpha.org.request.zip --ca-dir ./ca --profile project_profile.yaml

Requester packages the returned signed zip:

.. code-block:: shell

   nvflare package alice@hospital-alpha.org.signed.zip --confirm-rootca

The generated user startup kit contains ``startup/fl_admin.sh``.

.. code-block:: shell

   cd workspace/hospital_federation/prod_00/alice@hospital-alpha.org
   ./startup/fl_admin.sh

******************************
Quick Start: Add a Server
******************************

The server admin first decides the host name and ports, then communicates
those values to the Project Admin. The Project Admin records them in
``project_profile.yaml`` before approvals.

Server Admin creates ``server.yaml``:

.. code-block:: yaml

   name: hospital_federation

   participants:
     - name: server1.hospital-central.org
       type: server
       org: hospital_central
       host_names:
         - 10.0.1.50
         - fl-server.internal
       connection_security: mtls

Server Admin tells the Project Admin:

.. code-block:: text

   Server host:    server1.hospital-central.org
   fed_learn_port: 8002
   admin_port:     8003

The Project Admin records these values under the ``server:`` block in
``project_profile.yaml``. ``cert approve`` signs that endpoint into every
approved signed zip.

Then use the same request, approval, and package workflow:

.. code-block:: shell

   nvflare cert request --participant server.yaml
   nvflare cert approve server1.hospital-central.org.request.zip --ca-dir ./ca --profile project_profile.yaml
   nvflare package server1.hospital-central.org.signed.zip --confirm-rootca

The server participant name follows the same validation convention as
centralized ``project.yaml`` server participants. A DNS name is recommended for
production, and ``localhost`` remains valid for local/demo workflows.
Additional DNS names or IP addresses can be added with ``host_names``.

********************
Remote Transfer Flow
********************

When the Project Admin and requester are on different machines, the zip files
are copied between machines. The private key remains on the requester machine.

Requester machine:

.. code-block:: shell

   nvflare cert request --participant hospital-a.yaml

Transfer this file to the Project Admin:

.. code-block:: text

   hospital-a/hospital-a.request.zip

Project Admin machine:

.. code-block:: shell

   nvflare cert approve hospital-a.request.zip --ca-dir ./ca --profile project_profile.yaml

Transfer this file back to the requester:

.. code-block:: text

   hospital-a.signed.zip

Requester machine:

.. code-block:: shell

   nvflare package hospital-a.signed.zip --confirm-rootca

If the signed zip is not next to the local request folder and package cannot
find the folder from local request state, specify it:

.. code-block:: shell

   nvflare package hospital-a.signed.zip --request-dir ./hospital-a --confirm-rootca

*****************************
Root CA Fingerprint Check
*****************************

``nvflare cert approve`` prints ``rootca_fingerprint_sha256``. This is the
SHA256 certificate fingerprint for the project ``rootCA.pem``. The Project
Admin should send this value to the requester through a trusted out-of-band
channel.

The signed zip already contains ``rootCA.pem``. The requester does not need a
separate root CA file before running ``nvflare package``; the out-of-band value
is only the fingerprint used to verify the root CA inside the signed zip.

``nvflare package`` always prints the fingerprint computed from the signed zip.
Use one of these options to make the package command verify it:

.. code-block:: shell

   nvflare package hospital-a.signed.zip --confirm-rootca
   nvflare package hospital-a.signed.zip \
       --expected-rootca-fingerprint SHA256:AA:BB:...

Without either option, packaging still validates the signed zip, metadata,
certificate chain, and local private-key match, but it does not prompt and does
not perform an out-of-band fingerprint comparison.

*********************
Local Automation Flow
*********************

For local testing, use the same zip artifacts instead of switching to a
different command shape:

.. code-block:: shell

   # create project_profile.yaml and participant definition files
   nvflare cert init --profile project_profile.yaml -o ./ca
   nvflare cert request --participant hospital-a.yaml
   nvflare cert approve hospital-a/hospital-a.request.zip --ca-dir ./ca --profile project_profile.yaml
   nvflare package hospital-a/hospital-a.signed.zip --confirm-rootca

This is the same workflow as remote approval. The only difference is that the
zip files are not copied between machines.

****************
Artifacts
****************

Request folder:

.. code-block:: text

   hospital-a/
     hospital-a.key          # private key, stays local
     hospital-a.csr          # CSR
     site.yaml               # full local participant definition
     request.json            # request metadata and hashes
     hospital-a.request.zip  # sent to Project Admin

Request zip:

.. code-block:: text

   request.json
   site.yaml
   hospital-a.csr

Signed zip:

.. code-block:: text

   signed.json
   signed.json.sig
   site.yaml
   hospital-a.crt
   rootCA.pem

``signed.json.sig`` is a Project Admin CA signature over ``signed.json``.
Packaging verifies it before trusting the approved endpoint, scheme, or
connection security values.

The ``site.yaml`` in the local request folder is the full participant
definition used later by ``nvflare package``. It can contain local
package-time fields, such as the server-side ``connection_security`` override.
Client and user request folders do not contain server endpoint blocks; the
approved endpoint comes from the signed zip.

The ``site.yaml`` inside the request zip and signed zip is sanitized approval
metadata. It contains the identity fields needed for approval and packaging,
but it does not contain private keys. Server endpoint fields are
injected from ``project_profile.yaml`` during approval, not from participant
definition files. Server-side ``connection_security`` overrides are excluded
from this sanitized copy because they are local package-time behavior.

The request zip and signed zip must not contain ``*.key`` files. The signed zip
is not a startup kit; it is the approval response used by ``nvflare package``.

**************
Trust Boundary
**************

The Project Admin's signature (``signed.json.sig``) covers exactly what is
inside the signed zip:

- Participant identity: name, org, type, and role.
- Signed participant certificate (``*.crt``) and root CA (``rootCA.pem``).
- Federation connection parameters: ``scheme``, ``connection_security``, and
  the server endpoint from ``project_profile.yaml``.

Everything else is **outside the approval chain** and remains local
package-time behavior:

- **Custom builders**: builders are not included in the request zip, the
  signed zip, or the signed metadata, and are never approved by the Project
  Admin. ``nvflare package`` reads any ``builders:`` block from the local
  participant definition and applies those builders at package time on the
  requester's machine — this is purely local, package-time behavior. Because
  each site configures its own builders independently without central
  coordination, features that require a matching builder configuration across
  all participants — such as homomorphic encryption (HE) or confidential
  computing (CC) — are not directly supported by the distributed provisioning
  workflow.
- **Server-side** ``connection_security`` **override**: read from the local
  server request folder at package time; not approved and not distributed.
- **Private key**: stays on the requester machine and is never sent anywhere.
- **Workspace layout**: chosen by the requester when running
  ``nvflare package``.

If your deployment requires centrally coordinated builders across all
participants (HE, CC, or other extensions), use centralized
``nvflare provision`` instead.

****************
Command Summary
****************

Project Admin:

.. code-block:: shell

   # create project_profile.yaml
   nvflare cert init --profile project_profile.yaml -o ./ca
   nvflare cert approve hospital-a.request.zip --ca-dir ./ca --profile project_profile.yaml

Requester:

.. code-block:: shell

   nvflare cert request --participant hospital-a.yaml
   nvflare package hospital-a.signed.zip --confirm-rootca

With explicit locations:

.. code-block:: shell

   nvflare cert request --participant ./defs/hospital-a.yaml --out ./requests/hospital-a
   nvflare cert approve ./requests/hospital-a/hospital-a.request.zip \
       --ca-dir ./ca \
       --profile ./project_profile.yaml \
       --out ./signed/hospital-a.signed.zip
   nvflare package ./signed/hospital-a.signed.zip \
       --request-dir ./requests/hospital-a \
       -w ./workspace \
       --confirm-rootca

****************
Notes
****************

- The private key stays in the request folder and is never sent to the Project
  Admin.
- Client and user participant definition files do not include server endpoint
  blocks. The approved endpoint comes from ``project_profile.yaml`` through the
  signed zip.
- Project-wide ``scheme`` and default ``connection_security`` come from the
  Project Admin's profile through the signed zip.
- Server ``connection_security`` overrides are resolved locally at package time
  from the original server participant definition.
- Use ``--confirm-rootca`` for interactive root CA fingerprint confirmation or
  ``--expected-rootca-fingerprint`` for non-interactive automation.
- Startup kits generated from signed zips are compatible with the normal
  NVFlare runtime.
- Standard distributed provisioning does not generate ``signature.json``.
  Trust is anchored in the signed participant certificate and ``rootCA.pem``.
- Custom builders are not part of the approval chain. Any ``builders:`` block
  in the local participant definition is applied at package time on the
  requester's machine. Features requiring coordinated builder configuration
  across all participants (HE, CC) are not directly supported; use centralized
  ``nvflare provision`` for those deployments.
