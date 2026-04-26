.. _cert_command:

############
Cert Command
############

The ``nvflare cert`` command family manages certificate material for distributed
provisioning. It is used by both requesters and the Project Admin.

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

The Project Admin runs ``cert init`` once per federation:

.. code-block:: shell

   nvflare cert init --project example_project --org nvidia -o ./ca

This creates:

- ``./ca/rootCA.pem``: root CA certificate
- ``./ca/rootCA.key``: root CA private key; keep this secret
- ``./ca/ca.json``: CA metadata

Common options:

- ``--project``: project name. Required.
- ``--org``: organization name.
- ``-o, --output-dir``: CA output directory. Required.
- ``--valid-days``: root CA validity in days. Default: ``3650``.
- ``--force``: overwrite existing CA files after backing them up.
- ``--schema``: print JSON schema for this command.

********************
Create a Request Zip
********************

The requester runs ``cert request`` on the machine that should own the private
key. The command creates a private key, CSR, metadata, and request zip.

Client site request:

.. code-block:: shell

   nvflare cert request site site-3 --org nvidia --project example_project

Server request:

.. code-block:: shell

   nvflare cert request server server1 --org nvidia --project example_project

User request:

.. code-block:: shell

   nvflare cert request user org-admin org_admin@nvidia.com --org nvidia --project example_project
   nvflare cert request user lead lead@nvidia.com --org nvidia --project example_project
   nvflare cert request user member member@nvidia.com --org nvidia --project example_project

By default, ``cert request`` writes to ``./<name>/``. For ``site-3``:

.. code-block:: text

   site-3/
     site-3.key
     site-3.csr
     site.yaml
     request.json
     site-3.request.zip

Send only ``site-3.request.zip`` to the Project Admin. The private key remains
local and is not included in the zip.

Common options:

- ``kind``: ``site``, ``server``, or ``user``.
- ``values``: for ``site`` and ``server``, the identity name; for ``user``,
  ``<cert-role> <email>``.
- ``--org``: organization name. Required.
- ``--project``: project name. Required.
- ``--out``: request folder. Default: ``./<name>``.
- ``--force``: overwrite existing request files.
- ``--schema``: print JSON schema for this command.

*********************
Approve a Request Zip
*********************

The Project Admin runs ``cert approve`` with the project CA:

.. code-block:: shell

   nvflare cert approve site-3.request.zip --ca-dir ./ca

This validates the request zip, signs the CSR, and creates:

.. code-block:: text

   site-3.signed.zip

Return the signed zip to the requester.

Use ``--out`` to choose the signed zip location:

.. code-block:: shell

   nvflare cert approve site-3.request.zip --ca-dir ./ca --out ./signed/site-3.signed.zip

Common options:

- ``request_zip``: request zip produced by ``nvflare cert request``. Required.
- ``-c, --ca-dir``: directory containing ``rootCA.pem``, ``rootCA.key``, and
  ``ca.json``. Required.
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

   nvflare cert request site site-3 --org nvidia --project example_project

Project Admin:

.. code-block:: shell

   nvflare cert approve site-3.request.zip --ca-dir ./ca

Requester:

.. code-block:: shell

   nvflare package site-3.signed.zip -e grpc://server1:8002 --request-dir ./site-3

For the full workflow, including package options and artifact layout, see
:ref:`distributed_provisioning`.
