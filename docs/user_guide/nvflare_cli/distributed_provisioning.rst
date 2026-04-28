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

1. The Project Admin initializes the project CA and creates
   ``project_profile.yaml``.
2. The requester creates a participant definition file for one site, server, or
   user.
3. The requester runs ``nvflare cert request`` and sends only the generated
   request zip to the Project Admin.
4. The Project Admin approves the request zip and returns the signed zip.
5. The Project Admin shares ``rootca_fingerprint_sha256`` through a trusted
   out-of-band channel.
6. The requester packages the signed zip on the machine that owns the local
   private key.

The resulting startup kit is used the same way as a centrally provisioned
startup kit.

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

Fields:

- ``name``: project name. Requests must match this value.
- ``scheme``: FLARE communication driver, such as ``grpc``, ``tcp``, or
  ``http``.
- ``connection_security``: project default connection security, such as
  ``tls``, ``mtls``, or ``clear``.

The Project Admin keeps this file local. During approval, ``scheme`` and the
default ``connection_security`` are copied into the signed zip metadata so
participants do not need the profile file.

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
       server:
         host: server1.hospital-central.org
         fed_learn_port: 8002
         admin_port: 8003

User example:

.. code-block:: yaml

   name: hospital_federation

   participants:
     - name: alice@hospital-alpha.org
       type: admin
       org: hospital_alpha
       role: lead
       server:
         host: server1.hospital-central.org
         fed_learn_port: 8002
         admin_port: 8003

Server example:

.. code-block:: yaml

   name: hospital_federation
   description: Central FL server for hospital network

   participants:
     - name: server1.hospital-central.org
       type: server
       org: hospital_central
       fed_learn_port: 8002
       admin_port: 8003
       host_names:
         - 10.0.1.50
         - fl-server.internal
       connection_security: mtls

For clients and users, the ``server`` block tells ``nvflare package`` which FL
server endpoint to place in the startup kit. Because this endpoint is in the
participant definition, the new package flow does not require a separate
an endpoint option.

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

Project Admin initializes the project CA once:

.. code-block:: shell

   nvflare cert init --project hospital_federation -o ./ca

Project Admin creates ``project_profile.yaml``:

.. code-block:: yaml

   name: hospital_federation
   scheme: grpc
   connection_security: tls

Site Admin creates ``hospital-a.yaml``:

.. code-block:: yaml

   name: hospital_federation

   participants:
     - name: hospital-a
       type: client
       org: hospital_alpha
       server:
         host: server1.hospital-central.org
         fed_learn_port: 8002
         admin_port: 8003

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
``rootca_fingerprint_sha256``. Return the signed zip to the Site Admin and
share the fingerprint through a trusted out-of-band channel.

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

Requester creates ``alice.yaml``:

.. code-block:: yaml

   name: hospital_federation

   participants:
     - name: alice@hospital-alpha.org
       type: admin
       org: hospital_alpha
       role: lead
       server:
         host: server1.hospital-central.org
         fed_learn_port: 8002
         admin_port: 8003

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

Server Admin creates ``server.yaml``:

.. code-block:: yaml

   name: hospital_federation

   participants:
     - name: server1.hospital-central.org
       type: server
       org: hospital_central
       fed_learn_port: 8002
       admin_port: 8003
       host_names:
         - 10.0.1.50
         - fl-server.internal
       connection_security: mtls

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

   nvflare cert init --project hospital_federation -o ./ca
   # create project_profile.yaml and participant definition files
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
   site.yaml
   hospital-a.crt
   rootCA.pem

The ``site.yaml`` in the local request folder is the full participant
definition used later by ``nvflare package``. It can contain local
package-time fields, such as the server-side ``connection_security`` override.

The ``site.yaml`` inside the request zip and signed zip is sanitized approval
metadata. It contains the identity and connection fields needed for approval
and packaging, but it does not contain private keys. Server-side
``connection_security`` overrides are excluded from this sanitized copy because
they are local package-time behavior.

The request zip and signed zip must not contain ``*.key`` files. The signed zip
is not a startup kit; it is the approval response used by ``nvflare package``.

****************
Command Summary
****************

Project Admin:

.. code-block:: shell

   nvflare cert init --project hospital_federation -o ./ca
   # create project_profile.yaml
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
- Client and user server endpoints come from the local participant definition
  file's ``server`` block.
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
