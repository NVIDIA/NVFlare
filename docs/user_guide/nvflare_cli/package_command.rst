.. _package_command:

###############
Package Command
###############

``nvflare package`` assembles a startup kit from a signed zip returned by the
Project Admin and the requester's local private key.

The public distributed provisioning form is:

.. code-block:: none

   usage: nvflare package [-h] [-w WORKSPACE] [--request-dir REQUEST_DIR]
                          [--expected-rootca-fingerprint EXPECTED_ROOTCA_FINGERPRINT]
                          [--confirm-rootca] [--force] [--schema]
                          input

The ``input`` positional argument is the ``*.signed.zip`` file produced by
``nvflare cert approve``.

*******************
Basic Package Flow
*******************

For a site request created in ``./hospital-a``:

.. code-block:: shell

   nvflare package hospital-a.signed.zip --confirm-rootca

To compare the signed zip root CA with the value received from the Project
Admin through a trusted out-of-band channel:

.. code-block:: shell

   nvflare package hospital-a.signed.zip --confirm-rootca

For non-interactive automation, pass the expected fingerprint explicitly:

.. code-block:: shell

   nvflare package hospital-a.signed.zip \
       --expected-rootca-fingerprint SHA256:AA:BB:...

If the signed zip is not next to the request folder and package cannot find the
folder from local request state, specify it:

.. code-block:: shell

   nvflare package hospital-a.signed.zip \
       --request-dir ./hospital-a \
       --confirm-rootca

The command validates that:

- the signed zip contains ``signed.json``, ``signed.json.sig``, ``site.yaml``,
  one signed certificate, and ``rootCA.pem``;
- ``signed.json.sig`` verifies against ``rootCA.pem`` before the signed
  endpoint, scheme, or connection security fields are trusted;
- the signed zip does not contain private keys;
- the local private key matches the signed certificate;
- the certificate chains to ``rootCA.pem``;
- local ``request.json`` metadata and signed metadata match.
- identity fields in the local request-folder ``site.yaml`` match the signed
  zip.

The command always prints ``rootca_fingerprint_sha256`` in its result. Without
``--confirm-rootca`` or ``--expected-rootca-fingerprint``, packaging does not
prompt and does not perform an out-of-band trust comparison.

The output goes under:

.. code-block:: text

   <workspace>/<project-name>/prod_NN/<identity>/

For example:

.. code-block:: text

   workspace/hospital_federation/prod_00/hospital-a/

****************************
Connection Configuration
****************************

The package command does not require an endpoint argument in the distributed
provisioning flow.

Connection values are resolved from:

- ``signed.json`` in the signed zip, which contains the Project Admin-approved
  ``scheme``, default ``connection_security``, and ``server`` endpoint from
  ``project_profile.yaml``;
- the original local participant definition in the request folder, which
  contains participant identity and package-time fields.

The server host and port fields are part of the signed approval metadata.
``nvflare package`` uses the signed ``server`` endpoint to generate startup
kits. Client and user request folders do not provide local endpoint overrides.
If the server endpoint changes after approval, update ``project_profile.yaml``
and regenerate affected signed zips.

Local package-time fields that are intentionally excluded from the signed zip,
such as custom builders and the server-side ``connection_security`` override,
remain local packaging inputs.

Client and user participant definitions do not include the server endpoint:

.. code-block:: yaml

   participants:
     - name: hospital-a
       type: client
       org: hospital_alpha

For user startup kits, the same signed endpoint from ``signed.json`` is used:

.. code-block:: yaml

   participants:
     - name: alice@hospital-alpha.org
       type: admin
       org: hospital_alpha
       role: lead

For server kits, ``connection_security`` may be set in the server participant
definition:

.. code-block:: yaml

   participants:
     - name: server1.hospital-central.org
       type: server
       org: hospital_central
       connection_security: mtls

This server-side value is a local package-time override. It is read from the
request folder when building the server kit. It is not approved by the Project
Admin and is not distributed as federation policy. If it is not set, package
uses the default ``connection_security`` from the signed zip.

**************************
Package a User Startup Kit
**************************

For a lead user:

.. code-block:: shell

   nvflare cert request --participant alice.yaml
   nvflare cert approve alice@hospital-alpha.org.request.zip --ca-dir ./ca --profile project_profile.yaml
   nvflare package alice@hospital-alpha.org.signed.zip --confirm-rootca

The generated startup kit contains:

.. code-block:: text

   startup/fl_admin.sh

Run it with:

.. code-block:: shell

   cd workspace/hospital_federation/prod_00/alice@hospital-alpha.org
   ./startup/fl_admin.sh

****************
Main Arguments
****************

- ``input``: approved ``*.signed.zip`` returned by ``nvflare cert approve``.
- ``-w, --workspace``: workspace root directory. Default: ``workspace``.
- ``--request-dir``: local request directory containing the private key,
  ``request.json``, and the full local participant definition. Use it when the
  signed zip is not next to the request folder and local request state is not
  available.
- ``--expected-rootca-fingerprint``: expected SHA256 fingerprint for
  ``rootCA.pem`` in the signed zip. The command fails if it does not match.
- ``--confirm-rootca``: print the signed zip root CA fingerprint and prompt for
  confirmation before packaging. Use ``--expected-rootca-fingerprint`` instead
  for JSON or non-interactive automation.
- ``--force``: allow packaging when the participant name already appears in the
  latest ``prod_NN`` directory. A new ``prod_NN`` is created alongside.
- ``--schema``: print JSON schema for this command.

*********************
JSON Output and Help
*********************

Use ``--schema`` for machine-readable command discovery:

.. code-block:: shell

   nvflare package --schema

The top-level CLI also supports JSON output mode:

.. code-block:: shell

   nvflare package hospital-a.signed.zip \
       --expected-rootca-fingerprint SHA256:AA:BB:... \
       --format json

For the end-to-end distributed provisioning workflow, see
:ref:`distributed_provisioning`.
