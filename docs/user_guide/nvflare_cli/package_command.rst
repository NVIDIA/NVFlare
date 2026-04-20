.. _package_command:

#########################
Package Command
#########################

``nvflare package`` assembles startup kits for distributed (manual)
provisioning from a locally held private key, a signed certificate, and
``rootCA.pem``.

The command does not generate ``signature.json``. Trust is anchored in mTLS
using the signed participant certificate and the project root CA.

***********************
Command Usage
***********************

.. code-block:: none

   usage: nvflare package [-h] [-t {client,server,org_admin,lead,member}]
                          [-e ENDPOINT] [-n NAME] [--dir DIR] [--cert CERT]
                          [--key KEY] [--rootca ROOTCA] [-w WORKSPACE]
                          [--project-name PROJECT_NAME] [-p PROJECT_FILE]
                          [--admin-port ADMIN_PORT] [--force] [--schema]

*****************
Packaging Modes
*****************

Three packaging modes are supported. Choose the one that best matches how your
site stores its certificate material.

From a Working Directory
========================

Directory mode is the simplest path for a single participant. Put the private
key, signed certificate, and ``rootCA.pem`` in one directory:

.. code-block:: shell

   nvflare package -e grpc://fl-server:8002 --dir ./hospital-1-kit

Behavior in this mode:

- participant name is auto-detected from the ``.key`` filename
- kit type is derived from the signed certificate's embedded type
- no ``-t`` is needed

From Explicit File Paths
========================

Use explicit mode when the key, cert, and root CA are stored in different
locations:

.. code-block:: shell

   nvflare package -n hospital-1 -e grpc://fl-server:8002 \
     --cert ./signed/hospital-1/hospital-1.crt \
     --key ./csr/hospital-1.key \
     --rootca ./signed/hospital-1/rootCA.pem

In explicit mode:

- ``-n`` supplies the participant name
- ``--cert``, ``--key``, and ``--rootca`` are all provided explicitly
- this mode is for a single participant

From YAML
=========

Use ``--project-file`` when you want to package from either:

- a single-site YAML with ``name``, ``org``, and ``type``
- a project-style YAML compatible with ``nvflare provision``

.. code-block:: shell

   nvflare package -e grpc://fl-server:8002 -p ./site.yaml --dir ./certs

When ``--project-file`` is used:

- ``-t`` becomes an optional type filter
- certs are discovered from ``--dir`` by participant common name
- kit type is still derived from each signed certificate

****************
Main Arguments
****************

- ``-e, --endpoint``: server endpoint URI. Supported schemes are ``grpc://``,
  ``tcp://``, and ``http://``. Required for all modes.
- ``-t, --type``: optional type filter for YAML mode only. Choices:
  ``client``, ``server``, ``org_admin``, ``lead``, ``member``.
- ``-n, --name``: participant name. Auto-detected in directory mode. In admin
  kit types, this must be an email address.
- ``--dir``: working directory containing key, cert, and ``rootCA.pem`` by
  convention. Mutually exclusive with explicit ``--cert/--key/--rootca`` input.
- ``--cert``: signed certificate from the Project Admin.
- ``--key``: private key generated locally by ``nvflare cert csr``.
- ``--rootca``: ``rootCA.pem`` from the Project Admin.
- ``-w, --workspace``: workspace root directory. Output goes to
  ``<workspace>/<project-name>/prod_NN/<name>/``. Default: ``workspace``.
- ``--project-name``: project name used in output path and in generated
  challenge-response configuration. Default: ``project``.
- ``-p, --project-file``: site-scoped or project-style YAML. Mutually exclusive
  with ``-n`` and explicit ``--cert/--key/--rootca`` mode.
- ``--admin-port``: server admin port. Default: same as the service port
  (single-port mode).
- ``--force``: allow re-packaging when the same participant name appears in the
  most recent ``prod_NN`` directory. A new ``prod_NN`` is created alongside the
  existing one.
- ``--schema``: print the JSON schema for this command's arguments and exit.

*********************
Important Notes
*********************

- ``--endpoint`` is required in all packaging modes.
- In single-participant mode, the startup-kit type is derived from the signed
  certificate produced by ``nvflare cert sign``. The certificate is the source
  of truth.
- ``WorkspaceBuilder`` and ``StaticFileBuilder`` are always managed by
  ``nvflare package``. If they appear in a project YAML builders list, those
  entries and their custom arguments are ignored.
- The output path is ``<workspace>/<project-name>/prod_NN/<name>/``.
- The server hostname in ``--endpoint`` must match the hostname expected by the
  server certificate for mTLS validation.

*********************
JSON Output and Help
*********************

Use ``--schema`` for machine-readable command discovery:

.. code-block:: shell

   nvflare package --schema

The top-level CLI also supports JSON output mode:

.. code-block:: shell

   nvflare --out-format json package -e grpc://fl-server:8002 --dir ./hospital-1-kit

For end-to-end distributed provisioning workflow details, see
:ref:`distributed_provisioning`.
