.. _cert_command:

#########################
Cert Command
#########################

The ``nvflare cert`` command family supports distributed (manual)
provisioning, where each participant generates its own private key locally and
the Project Admin signs CSRs.

***********************
Command Usage
***********************

.. code-block:: none

   nvflare cert -h

   usage: nvflare cert [-h] {init,csr,sign} ...

   positional arguments:
     {init,csr,sign}
       init   Initialize root CA for distributed provisioning
       csr    Generate a private key and CSR
       sign   Sign a CSR with the root CA

*****************
Subcommands
*****************

Initialize the Root CA
======================

Run ``nvflare cert init`` once per federation as the Project Admin:

.. code-block:: shell

   nvflare cert init --project my-fl-project -o ./ca

Common options:

- ``--project``: project name. Used as the root CA certificate CN.
- ``-o, --output-dir``: directory where CA files are written.
- ``--org``: optional organization name for the CA certificate.
- ``--force``: overwrite existing CA files after backing them up.
- ``--schema``: print the JSON schema for this command and exit.

This creates:

- ``rootCA.pem``
- ``rootCA.key``
- ``ca.json``

Generate a Private Key and CSR
==============================

Run ``nvflare cert csr`` on the participant side to generate a local private
key and CSR.

Client example:

.. code-block:: shell

   nvflare cert csr -n hospital-1 -t client -o ./csr

Server example:

.. code-block:: shell

   nvflare cert csr -n fl-server -t server -o ./csr

Single-site YAML example:

.. code-block:: shell

   nvflare cert csr --project-file ./site.yaml -o ./csr

Common options:

- ``-n, --name``: participant name. Used as the certificate common name.
- ``-o, --output-dir``: output directory for the ``.key`` and ``.csr`` files.
- ``--org``: optional organization name for the certificate.
- ``--project-file``: single-site YAML with ``name``, ``org``, and ``type``.
  Use this instead of ``--name``, ``--org``, and ``--type``.
- ``-t, --type``: proposed certificate type. Choices:
  ``client``, ``server``, ``org_admin``, ``lead``, ``member``.
- ``--force``: overwrite existing key and CSR files.
- ``--schema``: print the JSON schema for this command and exit.

The private key remains local. Only the ``.csr`` file is sent to the Project
Admin.

The ``-t`` value in ``cert csr`` is a proposal only. The Project Admin may
override it when signing.

Sign a CSR
==========

Run ``nvflare cert sign`` as the Project Admin to sign a CSR with the project
root CA:

.. code-block:: shell

   nvflare cert sign -r ./csr/hospital-1.csr -c ./ca -o ./signed/hospital-1

Use ``-t`` to override the proposed type embedded in the CSR:

.. code-block:: shell

   nvflare cert sign -r ./csr/fl-server.csr -t server -c ./ca -o ./signed/fl-server

Common options:

- ``-r, --csr``: path to the CSR file received from the participant.
- ``-c, --ca-dir``: directory containing ``rootCA.pem``, ``rootCA.key``, and ``ca.json``.
- ``-o, --output-dir``: output directory for the signed cert and ``rootCA.pem`` copy.
- ``-t, --type``: authoritative cert type to issue. Choices:
  ``client``, ``server``, ``org_admin``, ``lead``, ``member``.
- ``--valid-days``: certificate validity in days. Default: ``1095``.
- ``--force``: overwrite existing signed certificate output.
- ``--schema``: print the JSON schema for this command and exit.

The type embedded in the signed certificate is the source of truth downstream.
``nvflare package`` derives startup-kit type from the signed certificate, not
from the CSR proposal.

*********************
JSON Output and Help
*********************

Use ``--schema`` for machine-readable command discovery:

.. code-block:: shell

   nvflare cert init --schema
   nvflare cert csr --schema
   nvflare cert sign --schema

The top-level CLI also supports JSON output mode:

.. code-block:: shell

   nvflare --out-format json cert init --project my-fl-project -o ./ca

Human-readable argument errors for ``cert init``, ``cert csr``, and
``cert sign`` print help first and then list the specific missing flags. JSON
mode prints only the JSON schema or JSON error envelope.

For the full distributed-provisioning workflow, see
:ref:`distributed_provisioning`.
