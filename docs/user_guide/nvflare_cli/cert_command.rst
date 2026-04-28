.. _cert_command:

############
Cert Command
############

The ``nvflare cert`` command family manages certificate material for
distributed provisioning. It is used by both requesters and the Project Admin.

The public subcommands are:

.. code-block:: none

   usage: nvflare cert [-h] {init,request,approve} ...

   positional arguments:
     {init,request,approve}
       init                Initialize root CA for a distributed provisioning
                           federation (Project Admin only).
       request             Create a distributed provisioning request zip.
       approve             Approve a distributed provisioning request zip.

**********************
Initialize the Root CA
**********************

The Project Admin first creates a project profile:

.. code-block:: yaml

   name: hospital_federation
   scheme: grpc
   connection_security: tls

Then the Project Admin runs ``cert init`` once per federation and passes the
profile file explicitly:

.. code-block:: shell

   nvflare cert init --profile project_profile.yaml -o ./ca

This creates:

- ``./ca/rootCA.pem``: root CA certificate
- ``./ca/rootCA.key``: root CA private key; keep this secret
- ``./ca/ca.json``: CA metadata

Common ``init`` options:

- ``--profile``: project profile yaml file. Required. ``cert init`` reads only
  the profile ``name`` and uses it as the root CA certificate subject; it does
  not search for profile files automatically. ``cert approve`` validates the
  profile ``scheme`` and ``connection_security`` fields later.
- ``--org``: optional organization name for the root CA certificate's O field.
- ``-o, --output-dir``: CA output directory. Required.
- ``--valid-days``: root CA validity in days. Default: ``3650``.
- ``--force``: overwrite existing CA files after backing them up.
- ``--schema``: print JSON schema for this command.

********************
Create a Request Zip
********************

The requester runs ``cert request`` on the machine that should own the private
key. The command reads one participant definition file, creates a private key,
CSR, metadata, and request zip.

Client site request:

.. code-block:: shell

   nvflare cert request --participant hospital-a.yaml

Server request:

.. code-block:: shell

   nvflare cert request --participant server.yaml

User request:

.. code-block:: shell

   nvflare cert request --participant alice.yaml

A participant definition uses the same top-level shape as a centralized
``project.yaml``: a ``name`` field for the project and a ``participants`` list.
For distributed provisioning the ``participants`` list must contain **exactly one
entry** — one file, one identity, one request. The key is plural to stay
consistent with the centralized format, but ``cert request`` will reject the
file if the list has zero or more than one entry.

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

To request a second participant such as ``hospital-b``, create a separate
``hospital-b.yaml`` file with its own single entry and run ``cert request``
again. Each participant has its own private key and request zip.

.. note::

   The values in this file are not generated automatically. Before writing the
   participant definition, the requester must obtain the following from the
   Project Admin through a trusted out-of-band channel (email, wiki, secure
   messaging):

   - **Project name** (``name:``): matches the ``name:`` in the Project Admin's
     ``project_profile.yaml``. Every participant definition must use the same
     project name.
   - **Server host and ports** (``server.host``, ``fed_learn_port``,
     ``admin_port``): chosen by the server admin and relayed by the Project
     Admin to all client sites and users. These values must be known before any
     client or user participant definition can be written.

   Server admins do not need a ``server:`` block; they set their own host and
   ports directly in their participant definition.

For users, use ``type: admin`` and set ``role`` to ``org_admin``, ``lead``, or
``member``.

By default, ``cert request`` writes to ``./<name>/``. For ``hospital-a``:

.. code-block:: text

   hospital-a/
     hospital-a.key
     hospital-a.csr
     site.yaml
     request.json
     hospital-a.request.zip

Send only ``hospital-a.request.zip`` to the Project Admin. The private key
remains local and is not included in the zip.

Common ``request`` options:

- ``-p, --participant``: participant definition file. Required.
- ``--out``: request folder. Default: ``./<participant-name>``.
- ``--force``: overwrite existing request files.
- ``--schema``: print JSON schema for this command.

*********************
Approve a Request Zip
*********************

The Project Admin runs ``cert approve`` with the project CA and project
profile:

.. code-block:: shell

   nvflare cert approve hospital-a.request.zip --ca-dir ./ca --profile project_profile.yaml

This validates the request zip, verifies that the request project matches the
CA and project profile, signs the CSR, and creates:

.. code-block:: text

   hospital-a.signed.zip
     signed.json
     site.yaml
     hospital-a.crt
     rootCA.pem

Return the signed zip to the requester.

The signed zip already includes ``rootCA.pem``. The requester does not need to
receive or place a separate ``rootCA.pem`` file before running
``nvflare package``.

The command output includes ``rootca_fingerprint_sha256``. Share only that
fingerprint value with the requester through a trusted out-of-band channel so
they can verify the signed zip root CA during ``nvflare package``.

Use ``--out`` to choose the signed zip location:

.. code-block:: shell

   nvflare cert approve hospital-a.request.zip \
       --ca-dir ./ca \
       --profile project_profile.yaml \
       --out ./signed/hospital-a.signed.zip

Common ``approve`` options:

- ``request_zip``: request zip produced by ``nvflare cert request``. Required.
- ``-c, --ca-dir``: directory containing ``rootCA.pem``, ``rootCA.key``, and
  ``ca.json``. Required.
- ``--profile``: Project Admin's ``project_profile.yaml`` containing
  ``name``, ``scheme``, and ``connection_security``. Required.
- ``--out``: signed zip output path. Default: ``<name>.signed.zip`` next to the
  request zip.
- ``--valid-days``: participant certificate validity in days. Default:
  ``1095``.
- ``--force``: overwrite existing signed zip.
- ``--schema``: print JSON schema for this command.

****************
End-to-End Flow
****************

Requester:

.. code-block:: shell

   nvflare cert request --participant hospital-a.yaml

Project Admin:

.. code-block:: shell

   nvflare cert approve hospital-a.request.zip --ca-dir ./ca --profile project_profile.yaml

Requester:

.. code-block:: shell

   nvflare package hospital-a.signed.zip --confirm-rootca

For the full workflow, including participant definition examples and artifact
layout, see :ref:`distributed_provisioning`.

.. note::

   **Compatibility:** Request zips produced before this release do not contain
   a ``site_yaml_sha256`` integrity field and will be rejected by
   ``nvflare cert approve``. Regenerate the request zip with
   ``nvflare cert request`` before running approve.
